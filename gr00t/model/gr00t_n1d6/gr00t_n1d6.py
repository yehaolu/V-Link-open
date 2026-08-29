"""GR00T-N1.6 with the fixed V-Link C2 action-query recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from huggingface_hub import snapshot_download
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.checkpoint_loading import (
    checkpoint_has_vlm_lora,
    confirm_checkpoint_group_loading,
    load_pretrained_weights_by_name,
)
from gr00t.model.modules.action_query_bridge import ActionQueryBridge
from gr00t.model.modules.action_query_depth import (
    GroupedActionQueryDepthHead,
    combined_depth_loss,
    prepare_depth_target_and_mask,
)
from gr00t.model.modules.action_query_segmentation import (
    GroupedActionQuerySegmentationHead,
    combined_segmentation_loss,
    prepare_segmentation_target,
)
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


class Gr00tN1d6ActionHead(nn.Module):
    """Flow-matching action head with grouped auxiliary heads and C2."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        config.validate_v_link_recipe()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=(
                    config.attend_text_every_n_blocks
                ),
            )
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln
            else nn.Identity()
        )

        if config.use_action_query_depth:
            tokens_per_camera = (
                config.action_query_grid_size[0]
                * config.action_query_grid_size[1]
            )
            self.action_query_depth_head = GroupedActionQueryDepthHead(
                feature_dim=config.backbone_embedding_dim,
                tokens_per_camera=tokens_per_camera,
                camera_grid_size=config.action_query_grid_size,
                num_cameras=config.action_query_depth_num_cameras,
                hidden_dim=config.action_query_depth_hidden_dim,
                decoder_channels=config.action_query_depth_decoder_channels,
                target_size=config.action_query_depth_target_size,
            )
        else:
            self.action_query_depth_head = None

        if config.use_action_query_segmentation:
            tokens_per_camera = (
                config.segmentation_query_grid_size[0]
                * config.segmentation_query_grid_size[1]
            )
            self.action_query_segmentation_head = (
                GroupedActionQuerySegmentationHead(
                    feature_dim=config.backbone_embedding_dim,
                    tokens_per_camera=tokens_per_camera,
                    camera_grid_size=config.segmentation_query_grid_size,
                    max_input_tokens=config.segmentation_query_num_tokens,
                    num_classes=(
                        config.action_query_segmentation_num_classes
                    ),
                    num_cameras=(
                        config.action_query_segmentation_num_cameras
                    ),
                    hidden_dim=(
                        config.action_query_segmentation_hidden_dim
                    ),
                    decoder_channels=(
                        config.action_query_segmentation_decoder_channels
                    ),
                    target_size=(
                        config.action_query_segmentation_target_size
                    ),
                )
            )
        else:
            self.action_query_segmentation_head = None

        if config.use_action_query_bridge:
            num_layers = len(self.model.transformer_blocks)
            interval = int(config.attend_text_every_n_blocks)
            active_layers = tuple(
                index
                for index in range(num_layers)
                if index % 2 == 0 and index % (2 * interval) != 0
            )
            self.action_query_bridge = ActionQueryBridge(
                action_dim=config.input_embedding_dim,
                backbone_dim=config.backbone_embedding_dim,
                num_layers=num_layers,
                num_heads=config.action_query_bridge_num_heads,
                use_rope=config.action_query_bridge_use_rope,
                bridge_weight=config.action_query_bridge_weight,
                active_layer_indices=active_layers,
            )
        else:
            self.action_query_bridge = None

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                config.max_seq_len, self.input_embedding_dim
            )
            nn.init.normal_(
                self.position_embedding.weight, mean=0.0, std=0.02
            )

        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(
                0.02 * torch.randn(1, 1, self.input_embedding_dim)
            )
            if self.state_dropout_prob > 0
            else None
        )
        self.state_additive_noise_scale = (
            config.state_additive_noise_scale
        )
        self.beta_dist = Beta(
            config.noise_beta_alpha, config.noise_beta_beta
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            config.tune_vlln,
        )

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_vlln: bool,
    ) -> None:
        self.tune_projector = bool(tune_projector)
        self.tune_diffusion_model = bool(tune_diffusion_model)
        self.tune_vlln = bool(tune_vlln)

        self.requires_grad_(True)
        if not self.tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.mask_token is not None:
                self.mask_token.requires_grad_(False)
        if not self.tune_diffusion_model:
            self.model.requires_grad_(False)
        if not self.tune_vlln:
            self.vlln.requires_grad_(False)
        if (
            self.action_query_bridge is not None
            and self.config.freeze_action_query_bridge
        ):
            self.action_query_bridge.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self) -> None:
        if not self.training:
            return
        if not self.tune_projector:
            self.state_encoder.eval()
            self.action_encoder.eval()
            self.action_decoder.eval()
            if self.config.add_pos_embed:
                self.position_embedding.eval()
        if not self.tune_diffusion_model:
            self.model.eval()
        if (
            self.action_query_bridge is not None
            and self.config.freeze_action_query_bridge
        ):
            self.action_query_bridge.eval()

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(
            device, dtype=dtype
        )
        return (1 - sample) * self.config.noise_s

    def process_backbone_output(
        self, backbone_output: BatchFeature
    ) -> BatchFeature:
        for key in (
            "backbone_features",
            "action_query_features",
            "segmentation_query_features",
        ):
            if key in backbone_output:
                backbone_output[key] = self.vlln(backbone_output[key])
        return backbone_output

    def _depth_outputs(
        self,
        action_query_features: torch.Tensor,
        action_input: BatchFeature,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        logits = self.action_query_depth_head.predict_logits(
            action_query_features
        )
        prediction = F.softplus(logits)
        if "depth_target" not in action_input:
            if self.training:
                raise KeyError(
                    "V-Link training requires depth_target from camera_meta."
                )
            return prediction, None, {}

        target, mask = prepare_depth_target_and_mask(
            prediction,
            action_input.depth_target,
            action_input.get("depth_mask"),
        )
        loss, terms = combined_depth_loss(
            prediction,
            target,
            mask,
            beta=self.config.action_query_depth_loss_beta,
            rel_weight=self.config.action_query_depth_rel_weight,
            min_depth=self.config.action_query_depth_min_depth,
            patch_weight=self.config.action_query_depth_patch_weight,
            patch_size=self.config.action_query_depth_patch_size,
            patch_topk_ratio=(
                self.config.action_query_depth_patch_topk_ratio
            ),
            patch_mode=self.config.action_query_depth_patch_mode,
            patch_min_valid_ratio=(
                self.config.action_query_depth_patch_min_valid_ratio
            ),
        )
        terms["valid_ratio"] = (mask > 0.5).float().mean().detach()
        return prediction, loss, terms

    def _segmentation_outputs(
        self,
        segmentation_query_features: torch.Tensor,
        action_input: BatchFeature,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        logits = self.action_query_segmentation_head(
            segmentation_query_features
        )
        if "segmentation_target" not in action_input:
            if self.training:
                raise KeyError(
                    "V-Link training requires segmentation_target from "
                    "camera_meta."
                )
            return logits, None, {}

        target = prepare_segmentation_target(
            logits,
            action_input.segmentation_target,
            self.config.action_query_segmentation_ignore_index,
        )
        loss, terms = combined_segmentation_loss(
            logits,
            target,
            self.config.action_query_segmentation_ignore_index,
            self.config.action_query_segmentation_dice_weight,
        )
        return logits, loss, terms

    def _add_position_embedding(
        self, action_features: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.add_pos_embed:
            return action_features
        positions = torch.arange(
            action_features.shape[1],
            dtype=torch.long,
            device=action_features.device,
        )
        return action_features + self.position_embedding(
            positions
        ).unsqueeze(0)

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> dict:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(
            action_input.state, embodiment_id
        )
        if self.state_dropout_prob > 0:
            drop = (
                torch.rand(
                    state_features.shape[0],
                    device=state_features.device,
                )
                < self.state_dropout_prob
            )[:, None, None].to(dtype=state_features.dtype)
            state_features = (
                state_features * (1 - drop) + self.mask_token * drop
            )
        if self.training and self.state_additive_noise_scale > 0:
            state_features = state_features + (
                torch.randn_like(state_features)
                * self.state_additive_noise_scale
            )

        actions = action_input.action
        noise = torch.randn_like(actions)
        time = self.sample_time(
            actions.shape[0], actions.device, actions.dtype
        )[:, None, None]
        noisy_trajectory = (1 - time) * noise + time * actions
        velocity = actions - noise
        timestep = (
            time[:, 0, 0] * self.num_timestep_buckets
        ).long()
        action_features = self._add_position_embedding(
            self.action_encoder(
                noisy_trajectory, timestep, embodiment_id
            )
        )
        state_action = torch.cat(
            (state_features, action_features), dim=1
        )

        action_queries = backbone_output.get(
            "action_query_features"
        )
        segmentation_queries = backbone_output.get(
            "segmentation_query_features"
        )
        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=state_action,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=(
                    backbone_output.backbone_attention_mask
                ),
                timestep=timestep,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=(
                    backbone_output.backbone_attention_mask
                ),
                action_query_bridge=self.action_query_bridge,
                action_query_features=action_queries,
                segmentation_query_features=segmentation_queries,
            )
        else:
            model_output, _ = self.model(
                hidden_states=state_action,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=(
                    backbone_output.backbone_attention_mask
                ),
                timestep=timestep,
                return_all_hidden_states=True,
            )

        prediction = self.action_decoder(model_output, embodiment_id)
        predicted_actions = prediction[:, -actions.shape[1] :]
        action_mask = action_input.action_mask
        action_loss = (
            F.mse_loss(
                predicted_actions, velocity, reduction="none"
            )
            * action_mask
        )
        action_loss_scalar = action_loss.sum() / (
            action_mask.sum() + 1e-6
        )
        total_loss = (
            self.config.action_loss_weight * action_loss_scalar
            if self.config.action_loss_weight > 0
            else action_loss_scalar.new_zeros(())
        )

        depth_prediction, depth_loss, depth_terms = (
            self._depth_outputs(action_queries, action_input)
        )
        if depth_loss is not None:
            total_loss = total_loss + (
                self.config.action_query_depth_loss_weight * depth_loss
            )

        segmentation_logits, segmentation_loss, segmentation_terms = (
            self._segmentation_outputs(
                segmentation_queries, action_input
            )
        )
        if segmentation_loss is not None:
            total_loss = total_loss + (
                self.config.action_query_segmentation_loss_weight
                * segmentation_loss
            )

        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "action_loss_scalar": action_loss_scalar,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "action_query_features": action_queries,
            "segmentation_query_features": segmentation_queries,
            "depth_pred": depth_prediction,
            "depth_loss": depth_loss,
            "depth_loss_terms": depth_terms,
            "segmentation_logits": segmentation_logits,
            "segmentation_loss": segmentation_loss,
            "segmentation_loss_terms": segmentation_terms,
        }

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        return BatchFeature(
            data={
                "backbone_features": (
                    backbone_output.backbone_features
                ),
                "state_features": self.state_encoder(
                    action_input.state,
                    action_input.embodiment_id,
                ),
            }
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        batch_size = backbone_features.shape[0]
        actions = torch.randn(
            (
                batch_size,
                self.config.action_horizon,
                self.action_dim,
            ),
            dtype=backbone_features.dtype,
            device=backbone_features.device,
        )
        action_queries = backbone_output.get(
            "action_query_features"
        )
        segmentation_queries = backbone_output.get(
            "segmentation_query_features"
        )
        step_size = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            continuous_time = (
                step / float(self.num_inference_timesteps)
            )
            discrete_time = int(
                continuous_time * self.num_timestep_buckets
            )
            timesteps = torch.full(
                (batch_size,),
                discrete_time,
                device=backbone_features.device,
            )
            action_features = self._add_position_embedding(
                self.action_encoder(
                    actions, timesteps, embodiment_id
                )
            )
            state_action = torch.cat(
                (state_features, action_features), dim=1
            )
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=state_action,
                    encoder_hidden_states=backbone_features,
                    timestep=timesteps,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=(
                        backbone_output.backbone_attention_mask
                    ),
                    action_query_bridge=self.action_query_bridge,
                    action_query_features=action_queries,
                    segmentation_query_features=segmentation_queries,
                )
            else:
                model_output = self.model(
                    hidden_states=state_action,
                    encoder_hidden_states=backbone_features,
                    timestep=timesteps,
                )
            velocity = self.action_decoder(
                model_output, embodiment_id
            )[:, -self.action_horizon :]
            actions = actions + step_size * velocity

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": backbone_features,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        features = self._encode_features(
            backbone_output, action_input
        )
        return self.get_action_with_features(
            features.backbone_features,
            features.state_features,
            action_input.embodiment_id,
            backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Vision-language-action model for GR00T-N1.6 and V-Link."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    @classmethod
    def from_pretrained_by_key(
        cls,
        checkpoint_path: str | Path,
        *,
        model_config_overrides: dict | None = None,
        transformers_loading_kwargs: dict | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> "Gr00tN1d6":
        loading_kwargs = dict(
            transformers_loading_kwargs
            or {"trust_remote_code": True}
        )
        local_path = Path(checkpoint_path)
        if not local_path.exists():
            download_kwargs = {
                key: loading_kwargs[key]
                for key in (
                    "local_files_only",
                    "revision",
                    "cache_dir",
                    "token",
                )
                if loading_kwargs.get(key) is not None
            }
            local_path = Path(
                snapshot_download(
                    repo_id=str(checkpoint_path),
                    **download_kwargs,
                )
            )

        checkpoint_config = AutoConfig.from_pretrained(
            local_path, **loading_kwargs
        )
        checkpoint_uses_lora = bool(
            getattr(checkpoint_config, "use_vlm_lora", False)
        ) or checkpoint_has_vlm_lora(local_path)
        checkpoint_lora_rank = int(
            getattr(checkpoint_config, "vlm_lora_rank", 128)
        )
        checkpoint_uses_queries = bool(
            getattr(
                checkpoint_config,
                "use_action_query_depth",
                False,
            )
        )
        checkpoint_uses_segmentation = bool(
            getattr(
                checkpoint_config,
                "use_action_query_segmentation",
                False,
            )
        )
        checkpoint_uses_bridge = bool(
            getattr(
                checkpoint_config,
                "use_action_query_bridge",
                False,
            )
        )

        model_config = checkpoint_config
        for key, value in (model_config_overrides or {}).items():
            setattr(model_config, key, value)
        requested_lora = bool(
            getattr(model_config, "use_vlm_lora", False)
        )
        requested_lora_rank = int(
            getattr(model_config, "vlm_lora_rank", 128)
        )
        model_config.use_vlm_lora = checkpoint_uses_lora
        model_config.vlm_lora_rank = (
            checkpoint_lora_rank
            if checkpoint_uses_lora
            else requested_lora_rank
        )
        model_config.defer_vlm_lora_initialization = (
            requested_lora and not checkpoint_uses_lora
        )
        model_config.validate_v_link_recipe()

        model = cls(
            model_config,
            transformers_loading_kwargs=loading_kwargs,
        )
        loading_info = load_pretrained_weights_by_name(
            model,
            local_path,
            strict_lora_shape=checkpoint_uses_lora,
        )

        required_groups = []
        must_load_groups = []
        for active, present, group in (
            (
                model.backbone.action_queries is not None,
                checkpoint_uses_queries,
                "action_query_tokens",
            ),
            (
                model.backbone.segmentation_queries is not None,
                checkpoint_uses_segmentation,
                "segmentation_query_tokens",
            ),
            (
                model.action_head.action_query_depth_head is not None,
                checkpoint_uses_queries,
                "action_query_depth_head",
            ),
            (
                model.action_head.action_query_segmentation_head is not None,
                checkpoint_uses_segmentation,
                "action_query_segmentation_head",
            ),
            (
                model.action_head.action_query_bridge is not None,
                checkpoint_uses_bridge,
                "action_query_bridge",
            ),
            (
                model.backbone.use_vlm_lora,
                checkpoint_uses_lora,
                "vlm_lora",
            ),
        ):
            if active:
                required_groups.append(group)
            if present:
                must_load_groups.append(group)
        confirm_checkpoint_group_loading(
            model,
            loading_info,
            required_groups=tuple(required_groups),
            must_load_groups=tuple(must_load_groups),
        )

        if requested_lora and not model.backbone.use_vlm_lora:
            model.backbone.enable_vlm_lora(requested_lora_rank)
        model.config.use_vlm_lora = requested_lora
        model.config.vlm_lora_rank = requested_lora_rank
        model.config.defer_vlm_lora_initialization = False
        model.config.validate_v_link_recipe()

        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        return model

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        config.validate_v_link_recipe()
        loading_kwargs = dict(
            transformers_loading_kwargs
            or {"trust_remote_code": True}
        )

        use_queries = (
            config.use_action_query_depth
            and config.use_action_query_segmentation
            and config.use_action_query_bridge
        )
        self.backbone = get_backbone_cls(config)(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=(
                config.backbone_trainable_params_fp32
            ),
            use_vlm_lora=config.use_vlm_lora,
            vlm_lora_rank=config.vlm_lora_rank,
            freeze_vlm_lora=config.freeze_vlm_lora,
            use_action_query_depth=use_queries,
            action_query_num_tokens=config.action_query_num_tokens,
            freeze_action_query=config.freeze_action_query,
            use_action_query_segmentation=use_queries,
            segmentation_query_num_tokens=(
                config.segmentation_query_num_tokens
            ),
            freeze_segmentation_query=(
                config.freeze_segmentation_query
            ),
            disentangle_action_segmentation_queries=(
                config.disentangle_action_segmentation_queries
            ),
            transformers_loading_kwargs=loading_kwargs,
        )
        self.action_head = Gr00tN1d6ActionHead(config)

        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=loading_kwargs,
        )

    def prepare_input(
        self, inputs: dict
    ) -> Tuple[BatchFeature, BatchFeature]:
        if "vlm_content" in inputs:
            content = inputs["vlm_content"]
            if not isinstance(content, list):
                content = [content]
            prepared = self.collator(
                [{"vlm_content": item} for item in content]
            )["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prepared)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device(tensor):
            if torch.is_floating_point(tensor):
                return tensor.to(self.device, dtype=self.dtype)
            return tensor.to(self.device)

        return (
            tree.map_structure(to_device, backbone_inputs),
            tree.map_structure(to_device, action_inputs),
        )

    def forward(self, inputs: dict) -> dict:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        return self.action_head(
            self.backbone(backbone_inputs), action_inputs
        )

    def get_action(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        return self.action_head.get_action(
            self.backbone(backbone_inputs), action_inputs
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
