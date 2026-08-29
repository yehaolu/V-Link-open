from __future__ import annotations

import pytest
import torch

from gr00t.model.modules.action_query_segmentation import (
    GroupedActionQuerySegmentationHead,
    combined_segmentation_loss,
    prepare_segmentation_target,
)


def make_small_grouped_head() -> GroupedActionQuerySegmentationHead:
    return GroupedActionQuerySegmentationHead(
        feature_dim=16,
        tokens_per_camera=4,
        camera_grid_size=(2, 2),
        num_cameras=3,
        max_input_tokens=12,
        num_classes=5,
        hidden_dim=16,
        decoder_channels=32,
        target_size=(8, 8),
        dropout=0.0,
        num_transformer_layers=1,
        num_attention_heads=4,
    )


def test_grouped_segmentation_head_shape() -> None:
    head = make_small_grouped_head().eval()

    logits = head(torch.randn(2, 12, 16))

    assert logits.shape == (2, 3, 5, 8, 8)


def test_grouped_segmentation_head_does_not_mix_camera_groups() -> None:
    torch.manual_seed(0)
    head = make_small_grouped_head().eval()
    features = torch.randn(1, 12, 16)
    changed = features.clone()
    changed[:, 4:8] += 10.0

    original = head(features)
    updated = head(changed)

    torch.testing.assert_close(
        original[:, (0, 2)],
        updated[:, (0, 2)],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(original[:, 1], updated[:, 1])


def test_grouped_segmentation_head_rejects_wrong_token_count() -> None:
    head = make_small_grouped_head()

    with pytest.raises(ValueError, match="expected 12"):
        head(torch.randn(2, 4, 16))


def test_grouped_segmentation_head_reuses_segmentation_target_and_loss() -> None:
    head = make_small_grouped_head().eval()
    logits = head(torch.randn(2, 12, 16))
    raw_target = torch.randint(0, 5, (2, 3, 4, 4))
    raw_target[:, :, 0, 0] = 255

    target = prepare_segmentation_target(logits, raw_target)
    loss, terms = combined_segmentation_loss(logits, target)

    assert target.shape == (2, 3, 8, 8)
    assert torch.isfinite(loss)
    assert set(("ce_loss", "dice_loss", "pixel_acc", "valid_ratio")) <= set(terms)
