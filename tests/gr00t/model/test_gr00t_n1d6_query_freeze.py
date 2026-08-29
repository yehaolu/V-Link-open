"""Trainability contracts for the fixed query recipe."""

import torch

from examples.RoboTwin2.train import apply_c2_recipe
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6 import gr00t_n1d6 as model_module


class _DummyBackbone(torch.nn.Module):
    last_kwargs = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        type(self).last_kwargs = kwargs
        self.action_queries = torch.nn.Embedding(75, 2)
        self.action_query_spatial_pos_embed = torch.nn.Parameter(torch.zeros(1, 75, 2))
        self.segmentation_queries = torch.nn.Embedding(75, 2)
        self.segmentation_query_spatial_pos_embed = torch.nn.Parameter(
            torch.zeros(1, 75, 2)
        )


class _DummyActionHead(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config


class _DummyCollator:
    def __init__(self, **_kwargs) -> None:
        pass


def _fixed_config(stage: str) -> Gr00tN1d6Config:
    config = Gr00tN1d6Config()
    apply_c2_recipe(config, stage)
    return config


def test_fixed_recipe_constructs_both_trainable_query_groups(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "get_backbone_cls", lambda config: _DummyBackbone)
    monkeypatch.setattr(model_module, "Gr00tN1d6ActionHead", _DummyActionHead)

    from gr00t.model.gr00t_n1d6 import processing_gr00t_n1d6

    monkeypatch.setattr(
        processing_gr00t_n1d6,
        "Gr00tN1d6DataCollator",
        _DummyCollator,
    )

    model = model_module.Gr00tN1d6(_fixed_config("warmup"))

    assert _DummyBackbone.last_kwargs["use_action_query_depth"] is True
    assert _DummyBackbone.last_kwargs["use_action_query_segmentation"] is True
    assert _DummyBackbone.last_kwargs["freeze_action_query"] is False
    assert _DummyBackbone.last_kwargs["freeze_segmentation_query"] is False
    assert model.backbone.action_queries.weight.requires_grad
    assert model.backbone.segmentation_queries.weight.requires_grad


def test_warmup_freezes_bridge_parameters_and_keeps_it_in_eval_mode() -> None:
    config = _fixed_config("warmup")
    config.backbone_embedding_dim = 16
    config.hidden_size = 32
    config.input_embedding_dim = 16
    config.max_state_dim = 8
    config.max_action_dim = 8
    config.action_dim = 8
    config.action_horizon = 2
    config.action_query_depth_target_size = (8, 8)
    config.action_query_depth_hidden_dim = 16
    config.action_query_depth_decoder_channels = 32
    config.action_query_segmentation_target_size = (8, 8)
    config.action_query_segmentation_hidden_dim = 16
    config.action_query_segmentation_decoder_channels = 32
    config.action_query_segmentation_num_classes = 5
    config.diffusion_model_cfg = {
        "num_layers": 4,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "output_dim": 16,
        "final_dropout": False,
        "interleave_self_attention": True,
    }

    action_head = model_module.Gr00tN1d6ActionHead(config)
    bridge = action_head.action_query_bridge

    assert bridge is not None
    assert not any(parameter.requires_grad for parameter in bridge.parameters())

    action_head.train()
    bridge.train()
    action_head.set_frozen_modules_to_eval_mode()
    assert not bridge.training
