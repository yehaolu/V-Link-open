"""Grouped Segmentation Query prediction and supervision."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GroupedActionQuerySegmentationHead(nn.Module):
    """Decode one contiguous Segmentation Query grid per camera."""

    def __init__(
        self,
        feature_dim: int,
        tokens_per_camera: int,
        camera_grid_size: tuple[int, int],
        max_input_tokens: int,
        num_classes: int,
        num_cameras: int = 3,
        hidden_dim: int = 256,
        decoder_channels: int = 64,
        target_size: tuple[int, int] = (128, 128),
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

        self.num_classes = int(num_classes)
        self.num_cameras = int(num_cameras)
        self.tokens_per_camera = int(tokens_per_camera)
        self.camera_grid_size = tuple(camera_grid_size)
        self.total_tokens = self.num_cameras * self.tokens_per_camera
        self.target_size = tuple(target_size)
        if max_input_tokens < self.total_tokens:
            raise ValueError(
                f"max_input_tokens={max_input_tokens} is smaller than "
                f"grouped token count {self.total_tokens}"
            )

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_embed = nn.Parameter(
            torch.zeros(1, max_input_tokens, hidden_dim)
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
            nn.Conv2d(decoder_channels // 4, self.num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features [B, tokens, dim], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.total_tokens:
            raise ValueError(
                f"Input has {features.shape[1]} Segmentation Query tokens, "
                f"expected {self.total_tokens}"
            )

        batch_size = features.shape[0]
        grouped = self.input_proj(features)
        grouped = grouped + self.position_embed[:, : self.total_tokens].to(
            device=grouped.device, dtype=grouped.dtype
        )
        grouped = grouped.reshape(
            batch_size * self.num_cameras,
            self.tokens_per_camera,
            -1,
        )
        grouped = self.token_encoder(grouped)

        grid_height, grid_width = self.camera_grid_size
        spatial = grouped.transpose(1, 2).reshape(
            batch_size * self.num_cameras,
            -1,
            grid_height,
            grid_width,
        )
        logits = self.decoder(spatial)
        return logits.reshape(
            batch_size,
            self.num_cameras,
            self.num_classes,
            *self.target_size,
        )


def prepare_segmentation_target(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4:
        raise ValueError(
            f"Expected segmentation target [B,V,H,W], got {tuple(target.shape)}"
        )

    target = target.to(device=logits.device, dtype=torch.long)
    num_cameras = logits.shape[1]
    if target.shape[1] == 1 and num_cameras != 1:
        target = target.expand(-1, num_cameras, -1, -1)
    elif target.shape[1] != num_cameras:
        raise ValueError(
            f"Segmentation target has {target.shape[1]} cameras but logits "
            f"have {num_cameras}"
        )

    if target.shape[-2:] != logits.shape[-2:]:
        batch_size, cameras, height, width = target.shape
        target = F.interpolate(
            target.reshape(batch_size * cameras, 1, height, width).float(),
            size=logits.shape[-2:],
            mode="nearest",
        ).reshape(batch_size, cameras, *logits.shape[-2:]).long()

    valid = (target >= 0) & (target < logits.shape[2])
    ignored = target == ignore_index
    return torch.where(
        valid | ignored, target, torch.full_like(target, ignore_index)
    )


def segmentation_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    batch_size, cameras, classes, height, width = logits.shape
    target = target.reshape(batch_size * cameras, height, width)
    if (target != ignore_index).sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(
        logits.reshape(batch_size * cameras, classes, height, width),
        target,
        ignore_index=ignore_index,
    )


def segmentation_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, cameras, classes, height, width = logits.shape
    logits = logits.reshape(batch_size * cameras, classes, height, width)
    target = target.reshape(batch_size * cameras, height, width)
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.new_zeros(())

    safe_target = target.masked_fill(~valid, 0)
    probabilities = F.softmax(logits, dim=1) * valid.unsqueeze(1)
    one_hot = (
        F.one_hot(safe_target, num_classes=classes)
        .permute(0, 3, 1, 2)
        .to(dtype=probabilities.dtype)
        * valid.unsqueeze(1)
    )
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    union = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    if present.sum() == 0:
        return logits.new_zeros(())
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice[present].mean()


def segmentation_pixel_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.new_zeros(())
    correct = (logits.argmax(dim=2) == target) & valid
    return correct.sum().float() / valid.sum().float().clamp_min(1.0)


def combined_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cross_entropy = segmentation_ce_loss(logits, target, ignore_index)
    dice = segmentation_dice_loss(logits, target, ignore_index)
    return cross_entropy + dice_weight * dice, {
        "ce_loss": cross_entropy.detach(),
        "dice_loss": dice.detach(),
        "pixel_acc": segmentation_pixel_accuracy(
            logits, target, ignore_index
        ).detach(),
        "valid_ratio": (target != ignore_index).float().mean().detach(),
    }
