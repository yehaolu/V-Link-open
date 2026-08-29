from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d6Config(PretrainedConfig):
    """Configuration for GR00T-N1.6 and the fixed V-Link C2 recipe."""

    model_type: str = "Gr00tN1d6"
    model_dtype: str = "bfloat16"

    # Backbone.
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_model_type: str = "eagle"
    model_revision: str | None = None
    tune_top_llm_layers: int = 4
    backbone_embedding_dim: int = 2048
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True
    collator_overwrite_image_inputs: bool = False
    eagle_collator: bool = False
    backbone_trainable_params_fp32: bool = True

    # V-Link VLM adaptation.
    use_vlm_lora: bool = False
    vlm_lora_rank: int = 128
    freeze_vlm_lora: bool = False
    defer_vlm_lora_initialization: bool = False

    # Fixed grouped Action Query depth head.
    use_action_query_depth: bool = False
    action_query_depth_source: str = "action_query"
    action_query_depth_supervision: str = "gt_depth"
    action_query_depth_head_type: str = "grouped_action_query_depth_head"
    action_query_num_tokens: int = 75
    action_query_grid_size: tuple[int, int] = (5, 5)
    action_query_depth_target_size: tuple[int, int] = (64, 80)
    action_query_depth_num_cameras: int = 3
    action_query_depth_hidden_dim: int = 256
    action_query_depth_decoder_channels: int = 64
    action_query_depth_loss_weight: float = 0.1
    action_query_depth_loss_beta: float = 0.1
    action_query_depth_rel_weight: float = 0.1
    action_query_depth_min_depth: float = 0.1
    action_query_depth_patch_weight: float = 0.5
    action_query_depth_patch_size: tuple[int, int] = (8, 8)
    action_query_depth_patch_topk_ratio: float = 0.25
    action_query_depth_patch_mode: str = "topk"
    action_query_depth_patch_min_valid_ratio: float = 0.25
    freeze_action_query: bool = False

    # Fixed grouped Segmentation Query head.
    use_action_query_segmentation: bool = False
    action_query_segmentation_head_type: str = (
        "grouped_action_query_segmentation_head"
    )
    segmentation_query_num_tokens: int = 75
    segmentation_query_grid_size: tuple[int, int] = (5, 5)
    action_query_segmentation_target_size: tuple[int, int] = (128, 128)
    action_query_segmentation_num_cameras: int = 3
    action_query_segmentation_num_classes: int = 178
    action_query_segmentation_hidden_dim: int = 256
    action_query_segmentation_decoder_channels: int = 64
    action_query_segmentation_loss_weight: float = 0.002
    action_query_segmentation_dice_weight: float = 0.5
    action_query_segmentation_ignore_index: int = 255
    freeze_segmentation_query: bool = False
    disentangle_action_segmentation_queries: bool = True

    # Fixed C2 bridge: Seg Query runs parallel to native VLM attention, then
    # Action Query refines the fused state serially.
    use_action_query_bridge: bool = False
    action_query_bridge_num_layers: int = 0
    action_query_bridge_num_heads: int = 8
    action_query_bridge_use_rope: bool = True
    action_query_bridge_use_action_query: bool = True
    action_query_bridge_use_segmentation_query: bool = True
    action_query_bridge_keep_action_query_in_vlm: bool = True
    action_query_bridge_keep_segmentation_query_in_vlm: bool = True
    action_query_bridge_context_mode: str = "c2"
    action_query_bridge_weight: float = 1.0
    action_query_bridge_segmentation_query_drop_prob: float = 0.0
    action_query_bridge_detach_segmentation_query: bool = False
    freeze_action_query_bridge: bool = False

    action_loss_weight: float = 1.0

    # Processing.
    image_crop_size: tuple[int, int] | None = None
    image_target_size: tuple[int, int] | None = None
    shortest_image_edge: int | None = 256
    crop_fraction: float | None = 0.95
    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = False
    use_relative_action: bool = False

    # Action head.
    max_state_dim: int = 29
    max_action_dim: int = 29
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching.
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training.
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if key == "collator_overwrite_image_inputs":
                setattr(self, "eagle_collator", value)
            setattr(self, key, value)

        for config_field in self.__dataclass_fields__.values():
            if not hasattr(self, config_field.name):
                if config_field.default is not MISSING:
                    setattr(self, config_field.name, config_field.default)
                elif getattr(config_field, "default_factory", MISSING) is not MISSING:
                    setattr(self, config_field.name, config_field.default_factory())

    def validate_v_link_recipe(self) -> None:
        """Reject partial or incompatible V-Link configurations."""
        enabled = (
            self.use_action_query_depth,
            self.use_action_query_segmentation,
            self.use_action_query_bridge,
        )
        if not any(enabled):
            return
        if not all(enabled):
            raise ValueError(
                "V-Link requires depth, segmentation, and C2 bridge together."
            )

        expected = {
            "vlm_lora_rank": 128,
            "action_query_depth_source": "action_query",
            "action_query_depth_supervision": "gt_depth",
            "action_query_depth_head_type": "grouped_action_query_depth_head",
            "action_query_num_tokens": 75,
            "action_query_grid_size": (5, 5),
            "action_query_depth_num_cameras": 3,
            "action_query_segmentation_head_type": (
                "grouped_action_query_segmentation_head"
            ),
            "segmentation_query_num_tokens": 75,
            "segmentation_query_grid_size": (5, 5),
            "action_query_segmentation_num_cameras": 3,
            "disentangle_action_segmentation_queries": True,
            "action_query_bridge_num_layers": 0,
            "action_query_bridge_num_heads": 8,
            "action_query_bridge_use_rope": True,
            "action_query_bridge_use_action_query": True,
            "action_query_bridge_use_segmentation_query": True,
            "action_query_bridge_keep_action_query_in_vlm": True,
            "action_query_bridge_keep_segmentation_query_in_vlm": True,
            "action_query_bridge_context_mode": "c2",
            "action_query_bridge_weight": 1.0,
            "action_query_bridge_segmentation_query_drop_prob": 0.0,
            "action_query_bridge_detach_segmentation_query": False,
            "freeze_action_query": False,
            "freeze_segmentation_query": False,
            "use_alternate_vl_dit": True,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if not (
            self.use_vlm_lora or self.defer_vlm_lora_initialization
        ):
            mismatches.append(
                "use_vlm_lora=False (expected True during normal use)"
            )
        if mismatches:
            raise ValueError("Unsupported V-Link recipe: " + "; ".join(mismatches))

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        if is_dataclass(self):
            config = asdict(self)
        else:
            config = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            config = {key: value for key, value in config.items() if key not in exclude_keys}
        return config

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        def default(value):
            if isinstance(value, (Path, torch.dtype, torch.device)):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            return str(value)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("GrootN1d6", Gr00tN1d6Config)
