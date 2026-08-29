"""C2 bridge from post-VLM query tokens to Action DiT image blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embeddings used by each C2 attention branch."""

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        if dim % 2:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            seq_len, device=device, dtype=self.inv_freq.dtype
        )
        frequencies = torch.einsum("i,j->ij", positions, self.inv_freq)
        angles = torch.repeat_interleave(frequencies, repeats=2, dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    even = tensor[..., ::2]
    odd = tensor[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(tensor)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine[None, None, :, :]
    sine = sine[None, None, :, :]
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class ActionQueryBridgeBlock(nn.Module):
    """One fixed C2 bridge block.

    Segmentation Query attention runs parallel to native image attention. Their
    outputs are fused before Action Query attention refines the state serially.
    """

    context_mode = "c2"

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_rope: bool = True,
        use_action_query: bool = True,
        use_segmentation_query: bool = True,
        context_mode: str = "c2",
        bridge_weight: float = 1.0,
        **unsupported,
    ) -> None:
        super().__init__()
        if unsupported:
            raise ValueError(
                "Unsupported C2 bridge options: "
                + ", ".join(sorted(unsupported))
            )
        if context_mode != "c2":
            raise ValueError("V-Link supports only context_mode='c2'.")
        if not use_action_query or not use_segmentation_query:
            raise ValueError(
                "V-Link C2 requires both Action Query and Segmentation Query."
            )
        if dim % num_heads:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.bridge_weight = float(bridge_weight)

        # Names are checkpoint-compatible with the original C2 implementation.
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.seg_norm = nn.LayerNorm(dim)
        self.seg_q_proj = nn.Linear(dim, dim)
        self.seg_k_adapter = nn.Linear(dim, dim)
        self.seg_v_adapter = nn.Linear(dim, dim)
        self.seg_o_proj = nn.Linear(dim, dim)

        self.c2_segmentation_gate = nn.Parameter(torch.zeros(()))
        self.c2_fuse_proj = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            self.c2_fuse_proj.weight.zero_()
            self.c2_fuse_proj.weight[:, :dim].copy_(torch.eye(dim))
            self.c2_fuse_proj.weight[:, dim:].copy_(torch.eye(dim))
            self.c2_fuse_proj.bias.zero_()

        self.rope = RotaryPositionEmbedding(self.head_dim) if use_rope else None

    def _reshape_heads(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _cross_attention(
        self,
        x: torch.Tensor,
        context_tokens: torch.Tensor,
        norm: nn.LayerNorm,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        context_len = context_tokens.shape[1]
        query = self._reshape_heads(
            q_proj(norm(x)), batch_size, seq_len
        )
        key = self._reshape_heads(
            k_proj(context_tokens), batch_size, context_len
        )
        value = self._reshape_heads(
            v_proj(context_tokens), batch_size, context_len
        )

        if self.rope is not None:
            cosine, sine = self.rope(
                seq_len, device=x.device, dtype=x.dtype
            )
            query, _ = apply_rope(query, query, cosine, sine)
            cosine, sine = self.rope(
                context_len, device=x.device, dtype=x.dtype
            )
            _, key = apply_rope(key, key, cosine, sine)

        scores = (
            torch.matmul(query, key.transpose(-2, -1))
            / math.sqrt(self.head_dim)
        ).float()
        weights = torch.softmax(scores, dim=-1).to(dtype=x.dtype)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, dim
        )
        return o_proj(output)

    def forward(
        self,
        x: torch.Tensor,
        *,
        native_attention_output: torch.Tensor | None = None,
        action_query_tokens: torch.Tensor | None = None,
        segmentation_query_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if native_attention_output is None:
            raise ValueError("C2 requires native_attention_output.")
        if native_attention_output.shape != x.shape:
            raise ValueError(
                "native_attention_output shape must match the pre-residual "
                f"state: native={tuple(native_attention_output.shape)} "
                f"x={tuple(x.shape)}"
            )
        if action_query_tokens is None or segmentation_query_tokens is None:
            raise ValueError(
                "C2 requires Action Query and Segmentation Query tokens."
            )

        segmentation_output = self._cross_attention(
            x,
            segmentation_query_tokens,
            self.seg_norm,
            self.seg_q_proj,
            self.seg_k_adapter,
            self.seg_v_adapter,
            self.seg_o_proj,
        )
        clamped_gate = self.c2_segmentation_gate.clamp(0.0, 1.0)
        segmentation_gate = self.c2_segmentation_gate + (
            clamped_gate - self.c2_segmentation_gate
        ).detach()
        fused_output = self.c2_fuse_proj(
            torch.cat(
                [
                    native_attention_output,
                    self.bridge_weight
                    * segmentation_gate
                    * segmentation_output,
                ],
                dim=-1,
            )
        )
        hidden_states = x + fused_output
        action_output = self._cross_attention(
            hidden_states,
            action_query_tokens,
            self.norm,
            self.q_proj,
            self.k_adapter,
            self.v_adapter,
            self.o_proj,
        )
        return hidden_states + self.bridge_weight * action_output


class ActionQueryBridge(nn.Module):
    """Project query features and attach one C2 block to each image DiT layer."""

    context_mode = "c2"

    def __init__(
        self,
        action_dim: int,
        backbone_dim: int,
        num_layers: int,
        num_heads: int = 8,
        use_rope: bool = True,
        bridge_weight: float = 1.0,
        active_layer_indices: tuple[int, ...] | None = None,
        **unsupported,
    ) -> None:
        super().__init__()
        if unsupported:
            raise ValueError(
                "Unsupported C2 bridge options: "
                + ", ".join(sorted(unsupported))
            )

        self.num_dit_layers = int(num_layers)
        if active_layer_indices is None:
            active_layer_indices = tuple(range(self.num_dit_layers))
        self.active_layer_indices = tuple(
            int(index) for index in active_layer_indices
        )
        if not self.active_layer_indices:
            raise ValueError("C2 requires at least one active DiT layer.")
        if len(set(self.active_layer_indices)) != len(
            self.active_layer_indices
        ):
            raise ValueError("C2 active_layer_indices contains duplicates.")
        if any(
            index < 0 or index >= self.num_dit_layers
            for index in self.active_layer_indices
        ):
            raise ValueError(
                "C2 active layer is outside the Action DiT stack."
            )

        self.action_query_proj = (
            nn.Identity()
            if backbone_dim == action_dim
            else nn.Linear(backbone_dim, action_dim)
        )
        self.segmentation_query_proj = (
            nn.Identity()
            if backbone_dim == action_dim
            else nn.Linear(backbone_dim, action_dim)
        )
        self.blocks = nn.ModuleDict(
            {
                str(layer_index): ActionQueryBridgeBlock(
                    dim=action_dim,
                    num_heads=num_heads,
                    use_rope=use_rope,
                    bridge_weight=bridge_weight,
                )
                for layer_index in self.active_layer_indices
            }
        )

    def prepare_context(
        self,
        action_query_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_query_features is None:
            raise ValueError(
                "C2 requires post-VLM action_query_features."
            )
        return self.action_query_proj(action_query_features)

    def prepare_segmentation_context(
        self,
        segmentation_query_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if segmentation_query_features is None:
            raise ValueError(
                "C2 requires post-VLM segmentation_query_features."
            )
        return self.segmentation_query_proj(segmentation_query_features)

    def has_block(self, layer_index: int) -> bool:
        return str(int(layer_index)) in self.blocks

    def get_block(self, layer_index: int) -> ActionQueryBridgeBlock:
        key = str(int(layer_index))
        if key not in self.blocks:
            raise KeyError(f"No C2 block is active for DiT layer {layer_index}.")
        return self.blocks[key]

    def forward(
        self,
        action_tokens: torch.Tensor,
        action_query_features: torch.Tensor,
        segmentation_query_features: torch.Tensor,
        native_attention_outputs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(native_attention_outputs) != len(self.active_layer_indices):
            raise ValueError(
                "native_attention_outputs length must match active C2 layers."
            )
        action_queries = self.prepare_context(action_query_features)
        segmentation_queries = self.prepare_segmentation_context(
            segmentation_query_features
        )
        for layer_index, native_output in zip(
            self.active_layer_indices,
            native_attention_outputs,
            strict=True,
        ):
            action_tokens = self.get_block(layer_index)(
                action_tokens,
                native_attention_output=native_output,
                action_query_tokens=action_queries,
                segmentation_query_tokens=segmentation_queries,
            )
        return action_tokens
