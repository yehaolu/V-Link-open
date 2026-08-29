"""Grouped Action Query depth prediction and supervision."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GroupedActionQueryDepthHead(nn.Module):
    """Decode one contiguous Action Query grid per camera."""

    def __init__(
        self,
        feature_dim: int,
        tokens_per_camera: int,
        camera_grid_size: tuple[int, int],
        num_cameras: int = 3,
        hidden_dim: int = 256,
        decoder_channels: int = 64,
        target_size: tuple[int, int] = (64, 80),
        dropout: float = 0.1,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        grid_height, grid_width = camera_grid_size
        if grid_height * grid_width != tokens_per_camera:
            raise ValueError(
                f"camera_grid_size {camera_grid_size} does not match "
                f"tokens_per_camera={tokens_per_camera}"
            )
        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")

        self.target_size = tuple(target_size)
        self.tokens_per_camera = int(tokens_per_camera)
        self.camera_grid_size = tuple(camera_grid_size)
        self.num_cameras = int(num_cameras)
        self.total_tokens = self.num_cameras * self.tokens_per_camera

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_embed = nn.Parameter(
            torch.zeros(1, self.total_tokens, hidden_dim)
        )
        nn.init.trunc_normal_(self.position_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.token_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Upsample(size=target_size, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_channels // 2),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, decoder_channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, decoder_channels // 4),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 4, 1, kernel_size=3, padding=1),
        )

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features [B, tokens, dim], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.total_tokens:
            raise ValueError(
                f"Input has {features.shape[1]} Action Query tokens, expected "
                f"{self.num_cameras} * {self.tokens_per_camera} = {self.total_tokens}"
            )

        batch_size = features.shape[0]
        grouped = self.input_proj(features).reshape(
            batch_size * self.num_cameras,
            self.tokens_per_camera,
            -1,
        )
        position_embed = self.position_embed.reshape(
            1,
            self.num_cameras,
            self.tokens_per_camera,
            -1,
        ).expand(batch_size, -1, -1, -1)
        grouped = grouped + position_embed.reshape(
            batch_size * self.num_cameras,
            self.tokens_per_camera,
            -1,
        ).to(device=grouped.device, dtype=grouped.dtype)
        grouped = self.token_encoder(grouped)

        grid_height, grid_width = self.camera_grid_size
        spatial = grouped.transpose(1, 2).reshape(
            batch_size * self.num_cameras,
            -1,
            grid_height,
            grid_width,
        )
        depth = self.decoder(spatial)
        return depth.reshape(batch_size, self.num_cameras, *self.target_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.predict_logits(features))


def prepare_depth_target_and_mask(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if mask is not None and mask.ndim == 3:
        mask = mask.unsqueeze(1)

    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if mask is None:
        valid = torch.isfinite(target) & (target > 0)
    else:
        mask = mask.to(device=prediction.device)
        valid = (mask > 0.5) & torch.isfinite(target) & (target > 0)

    if prediction.shape[-2:] != target.shape[-2:]:
        valid_float = valid.to(dtype=target.dtype)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        weighted_target = F.interpolate(
            safe_target * valid_float,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        valid_weight = F.interpolate(
            valid_float,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        target = weighted_target / valid_weight.clamp_min(1e-6)
        valid = valid_weight > 0.5

    if target.shape[1] == 1 and prediction.shape[1] != 1:
        target = target.expand(-1, prediction.shape[1], -1, -1)
    if valid.shape[1] == 1 and prediction.shape[1] != 1:
        valid = valid.expand(-1, prediction.shape[1], -1, -1)

    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return target, valid.to(device=prediction.device, dtype=prediction.dtype)


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_abs_relative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float = 0.1,
) -> torch.Tensor:
    loss = (prediction - target).abs() / target.clamp_min(min_depth)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def hard_patch_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    rel_weight: float,
    min_depth: float,
    patch_size: tuple[int, int],
    topk_ratio: float,
    mode: str,
    min_valid_ratio: float,
) -> torch.Tensor:
    if mode == "none" or topk_ratio <= 0:
        return prediction.new_zeros(())

    pixel_loss = F.smooth_l1_loss(
        prediction, target, beta=beta, reduction="none"
    )
    pixel_loss += rel_weight * (
        (prediction - target).abs() / target.clamp_min(min_depth)
    )

    patch_height, patch_width = patch_size
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    height, width = prediction.shape[-2:]
    pad_height = (patch_height - height % patch_height) % patch_height
    pad_width = (patch_width - width % patch_width) % patch_width
    if pad_height or pad_width:
        pixel_loss = F.pad(pixel_loss, (0, pad_width, 0, pad_height))
        mask = F.pad(mask, (0, pad_width, 0, pad_height))

    patch_loss_sum = F.avg_pool2d(
        pixel_loss * mask, kernel_size=patch_size, stride=patch_size
    )
    patch_valid_ratio = F.avg_pool2d(
        mask, kernel_size=patch_size, stride=patch_size
    )
    values = (
        patch_loss_sum / patch_valid_ratio.clamp_min(1e-6)
    )[patch_valid_ratio >= min_valid_ratio]
    if values.numel() == 0:
        return prediction.new_zeros(())
    if mode == "max":
        return values.max()
    if mode == "mean":
        return values.mean()
    if mode == "topk":
        count = max(1, math.ceil(values.numel() * min(max(topk_ratio, 0.0), 1.0)))
        return values.topk(k=count, largest=True).values.mean()
    raise ValueError(f"Unsupported patch loss mode: {mode}")


def combined_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.1,
    rel_weight: float = 0.1,
    min_depth: float = 0.1,
    patch_weight: float = 0.5,
    patch_size: tuple[int, int] = (8, 8),
    patch_topk_ratio: float = 0.25,
    patch_mode: str = "topk",
    patch_min_valid_ratio: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    smooth_l1 = masked_smooth_l1_loss(prediction, target, mask, beta)
    abs_relative = masked_abs_relative_loss(
        prediction, target, mask, min_depth
    )
    patch = (
        hard_patch_depth_loss(
            prediction,
            target,
            mask,
            beta=beta,
            rel_weight=rel_weight,
            min_depth=min_depth,
            patch_size=patch_size,
            topk_ratio=patch_topk_ratio,
            mode=patch_mode,
            min_valid_ratio=patch_min_valid_ratio,
        )
        if patch_weight > 0 and patch_mode != "none"
        else prediction.new_zeros(())
    )
    return smooth_l1 + rel_weight * abs_relative + patch_weight * patch, {
        "smooth_l1": smooth_l1.detach(),
        "abs_rel_loss": abs_relative.detach(),
        "patch_loss": patch.detach(),
    }
