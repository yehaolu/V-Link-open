"""Train the reproducible RoboTwin2 C2 recipe."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Literal

from gr00t.configs.base_config import Config, get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.experiment import run


@dataclass
class RoboTwin2TrainConfig:
    """User-configurable runtime settings for the fixed C2 recipe."""

    stage: Literal["warmup", "full"]
    base_model_path: str
    dataset_path: Path
    output_dir: Path
    num_gpus: int = 4
    seed: int = 0
    global_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    save_total_limit: int = 10
    logging_steps: int = 10
    dataloader_num_workers: int = 8
    shard_size: int = 1024
    episode_sampling_rate: float = 0.1
    wandb_project: str = "v-link-robotwin2"
    use_wandb: bool = False


def apply_c2_recipe(model_config, stage: str) -> None:
    """Apply the fixed 75-query C2 topology and stage-specific trainability."""
    if stage not in {"warmup", "full"}:
        raise ValueError(f"stage must be warmup or full, got {stage!r}")

    model_config.tune_llm = False
    model_config.tune_visual = False
    model_config.tune_top_llm_layers = 0
    model_config.use_vlm_lora = True
    model_config.vlm_lora_rank = 128
    model_config.freeze_vlm_lora = False

    model_config.use_action_query_depth = True
    model_config.action_query_depth_source = "action_query"
    model_config.action_query_depth_supervision = "gt_depth"
    model_config.action_query_depth_head_type = "grouped_action_query_depth_head"
    model_config.action_query_depth_num_cameras = 3
    model_config.action_query_num_tokens = 75
    model_config.action_query_grid_size = (5, 5)
    model_config.action_query_depth_loss_weight = 0.1
    model_config.freeze_action_query = False

    model_config.use_action_query_segmentation = True
    model_config.action_query_segmentation_head_type = (
        "grouped_action_query_segmentation_head"
    )
    model_config.action_query_segmentation_num_cameras = 3
    model_config.segmentation_query_num_tokens = 75
    model_config.segmentation_query_grid_size = (5, 5)
    model_config.action_query_segmentation_loss_weight = 0.002
    model_config.action_query_segmentation_dice_weight = 0.5
    model_config.action_query_segmentation_num_classes = 178
    model_config.freeze_segmentation_query = False
    model_config.disentangle_action_segmentation_queries = True

    model_config.use_action_query_bridge = True
    model_config.action_query_bridge_num_layers = 0
    model_config.action_query_bridge_num_heads = 8
    model_config.action_query_bridge_use_rope = True
    model_config.action_query_bridge_use_action_query = True
    model_config.action_query_bridge_use_segmentation_query = True
    model_config.action_query_bridge_keep_action_query_in_vlm = True
    model_config.action_query_bridge_keep_segmentation_query_in_vlm = True
    model_config.action_query_bridge_context_mode = "c2"
    model_config.action_query_bridge_weight = 1.0
    model_config.action_query_bridge_segmentation_query_drop_prob = 0.0
    model_config.action_query_bridge_detach_segmentation_query = False

    is_full = stage == "full"
    model_config.action_loss_weight = 1.0 if is_full else 0.0
    model_config.tune_projector = is_full
    model_config.tune_diffusion_model = is_full
    model_config.tune_vlln = is_full
    model_config.freeze_action_query_bridge = not is_full


def _action_horizon(config: Config, embodiment_tag: str) -> int:
    action = config.data.modality_configs[embodiment_tag]["action"]
    if not action.delta_indices:
        raise ValueError("RoboTwin2 action modality must declare delta indices")
    return len(action.delta_indices)


def _register_fixed_modality_config() -> None:
    """Register the one public modality config through Python's module cache."""
    importlib.import_module("examples.RoboTwin2.modality_config_robotwin2")


def build_config(args: RoboTwin2TrainConfig) -> Config:
    """Build the GR00T runtime config for one C2 training stage."""
    _register_fixed_modality_config()
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT.value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "seed": args.seed,
                "datasets": [
                    {
                        "dataset_paths": [str(args.dataset_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    apply_c2_recipe(config.model, args.stage)
    config.model.load_bf16 = True
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.action_horizon = _action_horizon(config, embodiment_tag)
    config.model.shortest_image_edge = 256
    config.model.crop_fraction = 0.95

    default_steps = 20_000 if args.stage == "warmup" else 120_000
    default_save_steps = 20_000 if args.stage == "warmup" else 40_000
    config.training.start_from_checkpoint = args.base_model_path
    config.training.output_dir = str(args.output_dir)
    # experiment.run writes directly to output_dir and derives the logging name
    # from its basename when experiment_name is None.
    config.training.experiment_name = None
    config.training.optim = "adamw_torch"
    config.training.num_gpus = args.num_gpus
    config.training.global_batch_size = args.global_batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.learning_rate = args.learning_rate
    config.training.max_steps = default_steps
    config.training.save_steps = default_save_steps
    config.training.save_total_limit = args.save_total_limit
    config.training.logging_steps = args.logging_steps
    config.training.dataloader_num_workers = args.dataloader_num_workers
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.use_wandb = args.use_wandb
    config.training.wandb_project = args.wandb_project

    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = args.episode_sampling_rate
    config.data.num_shards_per_epoch = int(1e5)
    return config


def main() -> None:
    import tyro

    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    args = tyro.cli(RoboTwin2TrainConfig, description=__doc__)
    run(build_config(args))


if __name__ == "__main__":
    main()
