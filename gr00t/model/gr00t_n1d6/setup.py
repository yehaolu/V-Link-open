"""Training pipeline for GR00T-N1.6 and the fixed V-Link recipe."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from termcolor import colored
import torch
from transformers import AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import (
    Gr00tN1d6Processor,
)
from gr00t.model.registry import register_model


def convert_tensors_to_lists(value):
    """Recursively convert tensors and arrays for JSON serialization."""
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            key: convert_tensors_to_lists(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in value]
    return value


class Gr00tN1d6Pipeline(ModelPipeline):
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir
        self.transformers_loading_kwargs = {
            "trust_remote_code": (
                config.training.transformers_trust_remote_code
            ),
            "local_files_only": (
                config.training.transformers_local_files_only
            ),
        }
        optional = {
            "revision": self.model_config.model_revision,
            "cache_dir": config.training.transformers_cache_dir,
            "token": config.training.transformers_access_token,
        }
        self.transformers_loading_kwargs.update(
            {
                key: value
                for key, value in optional.items()
                if value is not None
            }
        )

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(
            self.save_cfg_dir
        )
        self.data_collator = self.processor.collator

    def _model_overrides(self) -> dict:
        """Fields intentionally controlled by the public V-Link recipe."""
        fields = (
            "tune_llm",
            "tune_visual",
            "tune_top_llm_layers",
            "tune_projector",
            "tune_diffusion_model",
            "tune_vlln",
            "backbone_trainable_params_fp32",
            "use_vlm_lora",
            "vlm_lora_rank",
            "freeze_vlm_lora",
            "action_loss_weight",
            "use_action_query_depth",
            "action_query_depth_source",
            "action_query_depth_supervision",
            "action_query_depth_head_type",
            "action_query_num_tokens",
            "action_query_grid_size",
            "action_query_depth_target_size",
            "action_query_depth_num_cameras",
            "action_query_depth_hidden_dim",
            "action_query_depth_decoder_channels",
            "action_query_depth_loss_weight",
            "action_query_depth_loss_beta",
            "action_query_depth_rel_weight",
            "action_query_depth_min_depth",
            "action_query_depth_patch_weight",
            "action_query_depth_patch_size",
            "action_query_depth_patch_topk_ratio",
            "action_query_depth_patch_mode",
            "action_query_depth_patch_min_valid_ratio",
            "freeze_action_query",
            "use_action_query_segmentation",
            "action_query_segmentation_head_type",
            "segmentation_query_num_tokens",
            "segmentation_query_grid_size",
            "action_query_segmentation_target_size",
            "action_query_segmentation_num_cameras",
            "action_query_segmentation_num_classes",
            "action_query_segmentation_hidden_dim",
            "action_query_segmentation_decoder_channels",
            "action_query_segmentation_loss_weight",
            "action_query_segmentation_dice_weight",
            "action_query_segmentation_ignore_index",
            "freeze_segmentation_query",
            "disentangle_action_segmentation_queries",
            "use_action_query_bridge",
            "action_query_bridge_num_layers",
            "action_query_bridge_num_heads",
            "action_query_bridge_use_rope",
            "action_query_bridge_use_action_query",
            "action_query_bridge_use_segmentation_query",
            "action_query_bridge_keep_action_query_in_vlm",
            "action_query_bridge_keep_segmentation_query_in_vlm",
            "action_query_bridge_context_mode",
            "action_query_bridge_weight",
            "action_query_bridge_segmentation_query_drop_prob",
            "action_query_bridge_detach_segmentation_query",
            "freeze_action_query_bridge",
            "state_dropout_prob",
            "state_additive_noise_scale",
            "action_horizon",
            "use_relative_action",
        )
        return {
            name: getattr(self.model_config, name)
            for name in fields
        }

    def _create_model(self):
        checkpoint = self.config.training.start_from_checkpoint
        if checkpoint is None:
            model = self.model_class(
                self.model_config,
                transformers_loading_kwargs=(
                    self.transformers_loading_kwargs
                ),
            )
        else:
            model = self.model_class.from_pretrained_by_key(
                checkpoint,
                model_config_overrides=self._model_overrides(),
                transformers_loading_kwargs=(
                    self.transformers_loading_kwargs
                ),
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            (self.save_cfg_dir / "final_model_config.json").write_text(
                model.config.to_filtered_json()
            )

        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        logging.info("Total parameters: %s", f"{total:,}")
        logging.info(
            "Trainable parameters: %s (%.2f%%)",
            f"{trainable:,}",
            100 * trainable / total,
        )
        return model

    def _processor_kwargs(self) -> dict:
        return {
            "modality_configs": self.config.data.modality_configs,
            "image_crop_size": self.model_config.image_crop_size,
            "image_target_size": self.model_config.image_target_size,
            "random_rotation_angle": (
                self.model_config.random_rotation_angle
            ),
            "color_jitter_params": (
                self.model_config.color_jitter_params
            ),
            "model_name": self.model_config.model_name,
            "model_type": self.model_config.backbone_model_type,
            "formalize_language": self.model_config.formalize_language,
            "apply_sincos_state_encoding": (
                self.model_config.apply_sincos_state_encoding
            ),
            "max_action_horizon": self.model_config.action_horizon,
            "use_albumentations": (
                self.model_config.use_albumentations_transforms
            ),
            "extra_augmentation_config": (
                self.model_config.extra_augmentation_config
            ),
            "shortest_image_edge": (
                self.model_config.shortest_image_edge
            ),
            "crop_fraction": self.model_config.crop_fraction,
            "include_depth_targets": True,
            "include_segmentation_targets": True,
            "use_relative_action": (
                self.model_config.use_relative_action
            ),
            "transformers_loading_kwargs": (
                self.transformers_loading_kwargs
            ),
        }

    def _create_dataset(self, save_cfg_dir: Path):
        checkpoint = self.config.training.start_from_checkpoint
        if checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                checkpoint,
                **self._processor_kwargs(),
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                statistics=None,
                embodiment_id_mapping=None,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                **self._processor_kwargs(),
            )

        if get_rank() == 0:
            (self.save_cfg_dir / "final_processor_config.json").write_text(
                json.dumps(
                    {
                        key: str(value)
                        for key, value in vars(processor).items()
                    },
                    indent=2,
                )
            )

        self.processor = processor
        train_dataset, eval_dataset = DatasetFactory(
            config=self.config
        ).build(processor=processor)
        statistics = convert_tensors_to_lists(
            train_dataset.get_dataset_statistics()
        )
        (save_cfg_dir / "dataset_statistics.json").write_text(
            json.dumps(statistics, indent=2)
        )
        return train_dataset, eval_dataset


register_model(Gr00tN1d6Config, Gr00tN1d6Pipeline)
