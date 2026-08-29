"""Eagle backbone with V-Link Action and Segmentation Query tokens."""

from __future__ import annotations

from contextlib import nullcontext
import os

import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

try:
    from transformers.modeling_utils import no_init_weights
except Exception:
    no_init_weights = None


def _build_disentangled_query_attention_mask(
    attention_mask: torch.Tensor,
    base_seq_len: int,
    action_query_num_tokens: int,
    segmentation_query_num_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Keep query groups bidirectional internally and isolated from each other."""
    if attention_mask.ndim != 2:
        raise ValueError(
            "Query attention expects a 2D padding mask before expansion."
        )
    batch_size, total_seq_len = attention_mask.shape
    expected = (
        base_seq_len
        + action_query_num_tokens
        + segmentation_query_num_tokens
    )
    if total_seq_len != expected:
        raise ValueError(
            f"Query mask length is {total_seq_len}, expected {expected}."
        )

    positions = torch.arange(total_seq_len, device=attention_mask.device)
    blocked = positions[None, :] > positions[:, None]

    action_start = base_seq_len
    action_end = action_start + action_query_num_tokens
    segmentation_start = action_end
    segmentation_end = segmentation_start + segmentation_query_num_tokens

    blocked[action_start:action_end, action_start:action_end] = False
    blocked[
        segmentation_start:segmentation_end,
        segmentation_start:segmentation_end,
    ] = False
    blocked[
        action_start:action_end,
        segmentation_start:segmentation_end,
    ] = True
    blocked[
        segmentation_start:segmentation_end,
        action_start:action_end,
    ] = True

    blocked = blocked[None, None].expand(
        batch_size, 1, total_seq_len, total_seq_len
    ).clone()
    blocked |= (attention_mask == 0)[:, None, None, :]
    additive_mask = torch.zeros(
        batch_size,
        1,
        total_seq_len,
        total_seq_len,
        device=attention_mask.device,
        dtype=dtype,
    )
    return additive_mask.masked_fill(blocked, torch.finfo(dtype).min)


class EagleBackbone(torch.nn.Module):
    """Eagle VLM with trainable query tokens and optional language-model LoRA."""

    def __init__(
        self,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        use_vlm_lora: bool = False,
        vlm_lora_rank: int = 128,
        freeze_vlm_lora: bool = False,
        use_action_query_depth: bool = False,
        action_query_num_tokens: int = 75,
        freeze_action_query: bool = False,
        use_action_query_segmentation: bool = False,
        segmentation_query_num_tokens: int = 75,
        freeze_segmentation_query: bool = False,
        disentangle_action_segmentation_queries: bool = True,
        transformers_loading_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        del reproject_vision, projector_dim, transformers_loading_kwargs

        if model_name != "nvidia/Eagle-Block2A-2B-v2":
            raise ValueError(f"Model {model_name} not supported")
        if not use_flash_attention:
            raise ValueError("Eagle-Block2A requires Flash Attention.")
        if not load_bf16:
            raise ValueError("Eagle-Block2A must be constructed in bfloat16 mode.")

        eagle_path = os.path.join(
            os.path.dirname(__file__), "nvidia", "Eagle-Block2A-2B-v2"
        )
        config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
        init_context = (
            no_init_weights() if no_init_weights is not None else nullcontext()
        )
        with init_context:
            self.model = AutoModel.from_config(
                config, trust_remote_code=True
            )

        self.use_vlm_lora = False
        self.vlm_lora_rank = int(vlm_lora_rank)
        self.freeze_vlm_lora = bool(freeze_vlm_lora)

        self.use_action_query_depth = bool(use_action_query_depth)
        self.action_query_num_tokens = int(action_query_num_tokens)
        self.freeze_action_query = bool(freeze_action_query)
        self.use_action_query_segmentation = bool(
            use_action_query_segmentation
        )
        self.segmentation_query_num_tokens = int(
            segmentation_query_num_tokens
        )
        self.freeze_segmentation_query = bool(freeze_segmentation_query)
        self.disentangle_action_segmentation_queries = bool(
            disentangle_action_segmentation_queries
        )

        hidden_size = int(self.model.language_model.config.hidden_size)
        self.action_queries = self._create_queries(
            enabled=self.use_action_query_depth,
            num_tokens=self.action_query_num_tokens,
            hidden_size=hidden_size,
            load_bf16=load_bf16,
            parameter_name="action_query_spatial_pos_embed",
        )
        self.segmentation_queries = self._create_queries(
            enabled=self.use_action_query_segmentation,
            num_tokens=self.segmentation_query_num_tokens,
            hidden_size=hidden_size,
            load_bf16=load_bf16,
            parameter_name="segmentation_query_spatial_pos_embed",
        )

        language_layers = self._get_language_model_layers()
        if select_layer > 0:
            while len(language_layers) > select_layer:
                language_layers.pop(-1)
        self.select_layer = select_layer

        if use_vlm_lora:
            self.enable_vlm_lora(self.vlm_lora_rank, reset_trainable=False)
        self.set_trainable_parameters(
            tune_llm, tune_visual, tune_top_llm_layers
        )

        if load_bf16 and trainable_params_fp32:
            for name, parameter in self.named_parameters():
                if name.startswith(
                    (
                        "action_queries.",
                        "action_query_",
                        "segmentation_queries.",
                        "segmentation_query_",
                    )
                ):
                    continue
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()

    def _create_queries(
        self,
        *,
        enabled: bool,
        num_tokens: int,
        hidden_size: int,
        load_bf16: bool,
        parameter_name: str,
    ) -> torch.nn.Embedding | None:
        if not enabled:
            setattr(self, parameter_name, None)
            return None

        queries = torch.nn.Embedding(num_tokens, hidden_size)
        position = torch.nn.Parameter(
            torch.zeros(1, num_tokens, hidden_size)
        )
        if load_bf16:
            queries = queries.to(dtype=torch.bfloat16)
            position.data = position.data.to(dtype=torch.bfloat16)
        init_std = self._initializer_range()
        with torch.no_grad():
            queries.weight.normal_(mean=0.0, std=init_std)
            position.normal_(mean=0.0, std=init_std)
        setattr(self, parameter_name, position)
        return queries

    def _initializer_range(self) -> float:
        return float(
            getattr(
                self.model.config,
                "initializer_range",
                getattr(
                    self.model.language_model.config,
                    "initializer_range",
                    0.02,
                ),
            )
        )

    # Retained alias for checkpoint setup and compatibility tests.
    def _get_action_query_initializer_range(self) -> float:
        return self._initializer_range()

    def _get_language_model_layers(self):
        language_model = self.model.language_model
        candidates = (
            language_model,
            getattr(language_model, "model", None),
            getattr(getattr(language_model, "model", None), "model", None),
            getattr(language_model, "base_model", None),
            getattr(getattr(language_model, "base_model", None), "model", None),
            getattr(
                getattr(
                    getattr(language_model, "base_model", None),
                    "model",
                    None,
                ),
                "model",
                None,
            ),
        )
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate.layers
        raise AttributeError("Could not locate Eagle decoder layers.")

    def enable_vlm_lora(
        self,
        rank: int | None = None,
        reset_trainable: bool = True,
    ) -> None:
        rank = self.vlm_lora_rank if rank is None else int(rank)
        if rank <= 0:
            raise ValueError(f"vlm_lora_rank must be positive, got {rank}")
        if getattr(self.model, "use_llm_lora", False):
            self.use_vlm_lora = True
            self.vlm_lora_rank = rank
            return

        self.model.wrap_llm_lora(r=rank, lora_alpha=2 * rank)
        self.use_vlm_lora = True
        self.vlm_lora_rank = rank
        if reset_trainable:
            self.set_trainable_parameters(
                self.tune_llm,
                self.tune_visual,
                self.tune_top_llm_layers,
            )

    def set_trainable_parameters(
        self,
        tune_llm: bool,
        tune_visual: bool,
        tune_top_llm_layers: int,
    ) -> None:
        self.tune_llm = bool(tune_llm)
        self.tune_visual = bool(tune_visual)
        self.tune_top_llm_layers = int(tune_top_llm_layers)

        for parameter in self.parameters():
            parameter.requires_grad = True
        if self.use_vlm_lora:
            for name, parameter in self.model.language_model.named_parameters():
                parameter.requires_grad = (
                    "lora_" in name and not self.freeze_vlm_lora
                )
        elif not self.tune_llm:
            self.model.language_model.requires_grad_(False)
        if not self.tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)
        if self.tune_top_llm_layers > 0 and not self.use_vlm_lora:
            for layer in self._get_language_model_layers()[
                -self.tune_top_llm_layers :
            ]:
                layer.requires_grad_(True)

        if self.freeze_action_query and self.action_queries is not None:
            self.action_queries.requires_grad_(False)
            self.action_query_spatial_pos_embed.requires_grad = False
        if (
            self.freeze_segmentation_query
            and self.segmentation_queries is not None
        ):
            self.segmentation_queries.requires_grad_(False)
            self.segmentation_query_spatial_pos_embed.requires_grad = False

    def set_frozen_modules_to_eval_mode(self) -> None:
        if not self.training:
            return
        if (
            not self.tune_llm
            and (not self.use_vlm_lora or self.freeze_vlm_lora)
        ):
            self.model.language_model.eval()
        if not self.tune_visual:
            self.model.vision_model.eval()
            self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        required = ("input_ids", "attention_mask", "pixel_values")
        vl_input = {key: vl_input[key] for key in required}
        if self.use_action_query_depth or self.use_action_query_segmentation:
            return self._forward_with_queries(vl_input)

        outputs = self.model(**vl_input, output_hidden_states=True)
        return BatchFeature(
            data={
                "backbone_features": outputs["hidden_states"][-1],
                "backbone_attention_mask": vl_input["attention_mask"] == 1,
                "image_mask": (
                    vl_input["input_ids"]
                    == self.model.config.image_token_index
                ),
            }
        )

    def _query_embeddings(
        self,
        queries: torch.nn.Embedding,
        position: torch.Tensor,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = queries.weight.unsqueeze(0) + position
        return embeddings.to(
            device=reference.device, dtype=reference.dtype
        ).expand(batch_size, -1, -1)

    def _forward_with_queries(
        self,
        vl_input: dict[str, torch.Tensor],
    ) -> BatchFeature:
        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids
        )
        vision_embeds = self.model.extract_feature(
            vl_input["pixel_values"], image_flags=None
        )

        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape(
            batch_size * seq_len, hidden_size
        )
        flat_ids = input_ids.reshape(batch_size * seq_len)
        image_indices = (
            flat_ids == self.model.image_token_index
        ).nonzero(as_tuple=False).squeeze(1)
        if vision_embeds.shape[0] != image_indices.numel():
            image_indices = image_indices[: vision_embeds.shape[0]]
            vision_embeds = vision_embeds[: image_indices.numel()]
        flat_embeds = flat_embeds.index_copy(
            0,
            image_indices,
            vision_embeds.to(
                device=flat_embeds.device, dtype=flat_embeds.dtype
            ),
        )
        input_embeds = flat_embeds.reshape(
            batch_size, seq_len, hidden_size
        )

        query_embeds = []
        query_masks = []
        if self.action_queries is not None:
            query_embeds.append(
                self._query_embeddings(
                    self.action_queries,
                    self.action_query_spatial_pos_embed,
                    batch_size,
                    input_embeds,
                )
            )
            query_masks.append(
                torch.ones(
                    batch_size,
                    self.action_query_num_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
            )
        if self.segmentation_queries is not None:
            query_embeds.append(
                self._query_embeddings(
                    self.segmentation_queries,
                    self.segmentation_query_spatial_pos_embed,
                    batch_size,
                    input_embeds,
                )
            )
            query_masks.append(
                torch.ones(
                    batch_size,
                    self.segmentation_query_num_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
            )

        multimodal_embeds = torch.cat([input_embeds, *query_embeds], dim=1)
        multimodal_mask = torch.cat(
            [attention_mask, *query_masks], dim=1
        )
        language_mask = multimodal_mask
        if self.disentangle_action_segmentation_queries:
            language_mask = _build_disentangled_query_attention_mask(
                multimodal_mask,
                seq_len,
                self.action_query_num_tokens,
                self.segmentation_query_num_tokens,
                input_embeds.dtype,
            )

        outputs = self._forward_language_model(
            inputs_embeds=multimodal_embeds,
            attention_mask=language_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        final_features = outputs.hidden_states[-1]
        cursor = seq_len
        action_query_features = final_features[
            :, cursor : cursor + self.action_query_num_tokens
        ]
        cursor += self.action_query_num_tokens
        segmentation_query_features = final_features[
            :, cursor : cursor + self.segmentation_query_num_tokens
        ]

        backbone_features = final_features[:, :seq_len]
        if not self.tune_llm and not self.use_vlm_lora:
            backbone_features = backbone_features.detach()
        return BatchFeature(
            data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": attention_mask == 1,
                "image_mask": (
                    input_ids == self.model.config.image_token_index
                ),
                "action_query_features": action_query_features,
                "segmentation_query_features": (
                    segmentation_query_features
                ),
            }
        )

    def _forward_language_model(self, **kwargs):
        """Switch Flash Attention to SDPA for the custom 4D query mask."""
        attention_mask = kwargs.get("attention_mask")
        language_config = getattr(
            self.model.language_model, "config", None
        )
        previous = (
            getattr(language_config, "_attn_implementation", None)
            if language_config is not None
            else None
        )
        restore = (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 4
            and previous == "flash_attention_2"
        )
        if restore:
            language_config._attn_implementation = "sdpa"
        try:
            return self.model.language_model(
                position_ids=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                **kwargs,
            )
        finally:
            if restore:
                language_config._attn_implementation = previous
