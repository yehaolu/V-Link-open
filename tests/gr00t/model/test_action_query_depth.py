"""Tests for the fixed grouped three-camera depth head."""

import pytest
import torch

from gr00t.model.modules.action_query_depth import (
    GroupedActionQueryDepthHead,
    prepare_depth_target_and_mask,
)


def _head() -> GroupedActionQueryDepthHead:
    return GroupedActionQueryDepthHead(
        feature_dim=16,
        tokens_per_camera=25,
        camera_grid_size=(5, 5),
        num_cameras=3,
        hidden_dim=16,
        decoder_channels=32,
        target_size=(8, 8),
        dropout=0.0,
        num_attention_heads=4,
        num_transformer_layers=1,
    )


def test_grouped_depth_head_predicts_one_map_per_camera() -> None:
    prediction = _head().eval()(torch.randn(2, 75, 16))

    assert prediction.shape == (2, 3, 8, 8)


def test_grouped_depth_head_does_not_mix_camera_groups() -> None:
    torch.manual_seed(0)
    head = _head().eval()
    features = torch.randn(1, 75, 16)
    changed = features.clone()
    changed[:, 25:50] += 10.0

    original_prediction = head(features)
    changed_prediction = head(changed)

    torch.testing.assert_close(
        original_prediction[:, (0, 2)],
        changed_prediction[:, (0, 2)],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(original_prediction[:, 1], changed_prediction[:, 1])


def test_grouped_depth_head_rejects_wrong_query_count() -> None:
    with pytest.raises(ValueError, match=r"3 \* 25 = 75"):
        _head()(torch.randn(2, 25, 16))


def test_depth_target_preserves_three_camera_dimension() -> None:
    prediction = torch.ones(2, 3, 8, 8)
    target = torch.ones(2, 3, 16, 16)

    prepared_target, prepared_mask = prepare_depth_target_and_mask(prediction, target)

    assert prepared_target.shape == prediction.shape
    assert prepared_mask.shape == prediction.shape

