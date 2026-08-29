"""Behavior contracts for the selected fixed C2 bridge."""

import pytest
import torch

from gr00t.model.modules.action_query_bridge import (
    ActionQueryBridge,
    ActionQueryBridgeBlock,
    RotaryPositionEmbedding,
    apply_rope,
)
from gr00t.model.modules.dit import BasicTransformerBlock


class _ConstantAttention(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden_states, **_kwargs):
        return torch.full_like(hidden_states, self.value)


class _ZeroFFN(torch.nn.Module):
    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


class _RecordingC2Bridge(torch.nn.Module):
    context_mode = "c2"

    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, x, native_attention_output=None, **_kwargs):
        self.calls.append((x.detach().clone(), native_attention_output.detach().clone()))
        return x + native_attention_output + 3.0


def test_rope_preserves_query_and_key_norms() -> None:
    rope = RotaryPositionEmbedding(dim=8)
    cosine, sine = rope(3, device=torch.device("cpu"), dtype=torch.float64)
    query = torch.randn(1, 1, 3, 8, dtype=torch.float64)
    key = torch.randn(1, 1, 3, 8, dtype=torch.float64)

    rotated_query, rotated_key = apply_rope(query, key, cosine, sine)

    torch.testing.assert_close(rotated_query.norm(dim=-1), query.norm(dim=-1))
    torch.testing.assert_close(rotated_key.norm(dim=-1), key.norm(dim=-1))


def test_c2_runs_segmentation_parallel_then_action_serially(monkeypatch) -> None:
    block = ActionQueryBridgeBlock(dim=4, num_heads=2, use_rope=True).eval()
    calls = []

    def fake_cross_attention(
        self,
        x,
        context_tokens,
        norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    ):
        branch = "segmentation" if norm is self.seg_norm else "action"
        calls.append((branch, x.detach().clone(), context_tokens.detach().clone()))
        value = 1.0 if branch == "segmentation" else 5.0
        return torch.full_like(x, value)

    monkeypatch.setattr(ActionQueryBridgeBlock, "_cross_attention", fake_cross_attention)
    with torch.no_grad():
        block.c2_fuse_proj.weight.zero_()
        block.c2_fuse_proj.weight[:, :4].copy_(2.0 * torch.eye(4))
        block.c2_fuse_proj.weight[:, 4:].copy_(3.0 * torch.eye(4))
        block.c2_fuse_proj.bias.zero_()
        block.c2_segmentation_gate.fill_(1.0)

    initial = torch.full((1, 2, 4), 10.0)
    native = torch.full_like(initial, 2.0)
    action_queries = torch.full((1, 75, 4), 7.0)
    segmentation_queries = torch.full((1, 75, 4), 11.0)

    output = block(
        initial,
        native_attention_output=native,
        action_query_tokens=action_queries,
        segmentation_query_tokens=segmentation_queries,
    )

    assert [branch for branch, _, _ in calls] == ["segmentation", "action"]
    expected_fused = initial + 2.0 * native + 3.0
    assert torch.equal(calls[1][1], expected_fused)
    assert torch.equal(output, expected_fused + 5.0)


def test_c2_initialization_preserves_native_and_action_paths(monkeypatch) -> None:
    block = ActionQueryBridgeBlock(dim=4, num_heads=2).eval()

    def fake_cross_attention(
        self,
        x,
        context_tokens,
        norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    ):
        return torch.full_like(x, 9.0)

    monkeypatch.setattr(ActionQueryBridgeBlock, "_cross_attention", fake_cross_attention)
    initial = torch.randn(2, 3, 4)
    native = torch.randn_like(initial)

    output = block(
        initial,
        native_attention_output=native,
        action_query_tokens=torch.randn(2, 75, 4),
        segmentation_query_tokens=torch.randn(2, 75, 4),
    )

    assert torch.equal(output, initial + native + 9.0)
    assert torch.equal(block.c2_segmentation_gate, torch.zeros(()))
    assert torch.equal(block.c2_fuse_proj.weight[:, :4], torch.eye(4))
    assert torch.equal(block.c2_fuse_proj.weight[:, 4:], torch.eye(4))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"context_mode": "concat"},
        {"use_action_query": False},
        {"use_segmentation_query": False},
        {"segmentation_query_drop_prob": 0.5},
    ),
)
def test_c2_rejects_removed_bridge_variants(kwargs) -> None:
    with pytest.raises(ValueError):
        ActionQueryBridgeBlock(dim=4, num_heads=2, **kwargs)


def test_basic_transformer_block_passes_native_attention_result_to_c2() -> None:
    block = BasicTransformerBlock(
        dim=4,
        num_attention_heads=2,
        attention_head_dim=2,
        cross_attention_dim=4,
        norm_type="layer_norm",
        final_dropout=False,
    )
    bridge = _RecordingC2Bridge()
    block.attn1 = _ConstantAttention(2.0)
    block.ff = _ZeroFFN()
    initial = torch.randn(1, 3, 4)

    output = block(
        initial,
        encoder_hidden_states=torch.randn(1, 5, 4),
        action_query_bridge_block=bridge,
        action_query_tokens=torch.randn(1, 75, 4),
        segmentation_query_tokens=torch.randn(1, 75, 4),
    )

    assert len(bridge.calls) == 1
    assert torch.equal(bridge.calls[0][0], initial)
    assert torch.equal(bridge.calls[0][1], 2.0 * torch.ones_like(initial))
    assert torch.allclose(output, initial + 5.0)


def test_bridge_attaches_blocks_only_to_selected_image_dit_layers() -> None:
    bridge = ActionQueryBridge(
        action_dim=8,
        backbone_dim=8,
        num_layers=8,
        num_heads=2,
        active_layer_indices=(2, 6),
    )

    assert tuple(bridge.blocks) == ("2", "6")
    assert bridge.has_block(2)
    assert bridge.has_block(6)
    assert not bridge.has_block(0)

