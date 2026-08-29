"""Checkpoint contracts retained by the fixed Warmup-to-Full workflow."""

import pytest
import torch

from gr00t.model.gr00t_n1d6.checkpoint_loading import (
    confirm_checkpoint_group_loading,
)


class _TinyBackbone(torch.nn.Module):
    def __init__(self, action_tokens: int = 75, segmentation_tokens: int = 75) -> None:
        super().__init__()
        self.action_queries = torch.nn.Embedding(action_tokens, 4)
        self.action_query_spatial_pos_embed = torch.nn.Parameter(
            torch.zeros(1, action_tokens, 4)
        )
        self.segmentation_queries = torch.nn.Embedding(segmentation_tokens, 4)
        self.segmentation_query_spatial_pos_embed = torch.nn.Parameter(
            torch.zeros(1, segmentation_tokens, 4)
        )

        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.block = torch.nn.Module()
        self.model.language_model.block.lora_A = torch.nn.Linear(4, 2, bias=False)


class _TinyModel(torch.nn.Module):
    def __init__(self, action_tokens: int = 75, segmentation_tokens: int = 75) -> None:
        super().__init__()
        self.backbone = _TinyBackbone(action_tokens, segmentation_tokens)


def test_confirmation_fails_when_checkpoint_query_group_mismatches_shape() -> None:
    model = _TinyModel(segmentation_tokens=74)
    loading_info = {
        "loaded_keys": [
            "backbone.action_queries.weight",
            "backbone.action_query_spatial_pos_embed",
        ],
        "missing_keys": [
            "backbone.segmentation_queries.weight",
            "backbone.segmentation_query_spatial_pos_embed",
        ],
        "unexpected_keys": [],
        "mismatched_keys": [
            ("backbone.segmentation_queries.weight", (75, 4), (74, 4)),
            ("backbone.segmentation_query_spatial_pos_embed", (1, 75, 4), (1, 74, 4)),
        ],
    }

    with pytest.raises(ValueError, match="SEGMENTATION_QUERY"):
        confirm_checkpoint_group_loading(
            model,
            loading_info,
            required_groups=("segmentation_query_tokens",),
        )


def test_confirmation_reports_loaded_query_and_lora_groups() -> None:
    model = _TinyModel()
    loading_info = {
        "loaded_keys": [
            "backbone.action_queries.weight",
            "backbone.action_query_spatial_pos_embed",
            "backbone.segmentation_queries.weight",
            "backbone.segmentation_query_spatial_pos_embed",
            "backbone.model.language_model.block.lora_A.weight",
        ],
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
    }

    summaries = confirm_checkpoint_group_loading(
        model,
        loading_info,
        required_groups=(
            "vlm_lora",
            "action_query_tokens",
            "segmentation_query_tokens",
        ),
    )

    assert summaries["vlm_lora"]["status"] == "loaded"
    assert summaries["action_query_tokens"]["status"] == "loaded"
    assert summaries["segmentation_query_tokens"]["status"] == "loaded"


def test_confirmation_fails_when_required_query_group_is_absent() -> None:
    model = _TinyModel()
    loading_info = {
        "loaded_keys": [],
        "missing_keys": [
            "backbone.action_queries.weight",
            "backbone.action_query_spatial_pos_embed",
        ],
        "unexpected_keys": [],
        "mismatched_keys": [],
    }

    with pytest.raises(ValueError, match="ACTION_QUERY"):
        confirm_checkpoint_group_loading(
            model,
            loading_info,
            required_groups=("action_query_tokens",),
            must_load_groups=("action_query_tokens",),
        )
