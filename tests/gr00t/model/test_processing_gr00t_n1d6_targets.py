from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import pytest

from gr00t.model.gr00t_n1d6.image_augmentations import (
    FractionalCenterCrop,
    FractionalRandomCrop,
    LetterBoxPad,
    apply_segmentation_with_replay,
)
from gr00t.model.gr00t_n1d6 import processing_gr00t_n1d6 as processing


FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "processor_config"


class _DummyVlmProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(padding_side="right")


class _IdentityStateActionProcessor:
    def apply(self, *, state, action, embodiment_tag):
        del embodiment_tag
        return state, action

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


def _make_target_processor(*, include_depth: bool, include_segmentation: bool):
    processor = processing.Gr00tN1d6Processor.__new__(
        processing.Gr00tN1d6Processor
    )
    modality = SimpleNamespace(modality_keys=["joint"])
    processor.modality_configs = {
        "robot": {
            "video": SimpleNamespace(modality_keys=["front", "wrist"]),
            "state": modality,
            "action": modality,
        }
    }
    processor.state_action_processor = _IdentityStateActionProcessor()
    processor.max_state_dim = 1
    processor.max_action_dim = 1
    processor.max_action_horizon = 1
    processor.formalize_language = False
    processor.use_albumentations = True
    processor.include_depth_targets = include_depth
    processor.include_segmentation_targets = include_segmentation
    processor.embodiment_id_mapping = {"robot": 0}
    processor.train_image_transform = A.ReplayCompose(
        [
            FractionalRandomCrop(crop_fraction=0.5, p=1.0),
            A.SmallestMaxSize(
                max_size=4,
                interpolation=cv2.INTER_LINEAR,
                p=1.0,
            ),
        ]
    )
    processor.train()
    processor.state_action_processor = _IdentityStateActionProcessor()
    processor._apply_vlm_processing = lambda images, language: {
        "transformed_rgb": images,
        "language": language,
    }
    return processor


def _make_message():
    front_labels = np.arange(10, 26, dtype=np.uint8).reshape(4, 4)
    wrist_labels = np.arange(100, 116, dtype=np.uint8).reshape(4, 4)

    def rgb(labels: np.ndarray) -> Image.Image:
        image = np.zeros((*labels.shape, 3), dtype=np.uint8)
        image[..., 0] = labels
        return Image.fromarray(image)

    content = SimpleNamespace(
        embodiment=SimpleNamespace(value="robot"),
        states={"joint": np.zeros((1, 1), dtype=np.float32)},
        actions={"joint": np.zeros((1, 1), dtype=np.float32)},
        images={"front": [rgb(front_labels)], "wrist": [rgb(wrist_labels)]},
        masks=None,
        text="Move!",
        camera_calibration={
            # Camera metadata deliberately uses the opposite order from RGB.
            "view_names": ["wrist", "front"],
            "depth_m": np.stack(
                [wrist_labels, front_labels], axis=0
            )[None].astype(np.float32),
            "mesh_segmentation": np.stack(
                [wrist_labels, front_labels], axis=0
            )[None].astype(np.int64),
        },
    )
    return [{"content": content}], (front_labels, wrist_labels)


def test_processor_exposes_only_the_selected_target_options() -> None:
    parameters = inspect.signature(processing.Gr00tN1d6Processor.__init__).parameters
    target_options = {name for name in parameters if name.startswith("include_")}

    assert target_options == {
        "include_depth_targets",
        "include_segmentation_targets",
    }


def test_target_options_survive_processor_serialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        processing,
        "build_processor",
        lambda model_name, transformers_loading_kwargs: _DummyVlmProcessor(),
    )
    processor = processing.Gr00tN1d6Processor.from_pretrained(
        FIXTURE_DIR,
        include_depth_targets=True,
        include_segmentation_targets=False,
    )

    processor.save_pretrained(tmp_path)
    restored = processing.Gr00tN1d6Processor.from_pretrained(tmp_path)

    config = json.loads((tmp_path / "processor_config.json").read_text())
    assert config["processor_kwargs"]["include_depth_targets"] is True
    assert config["processor_kwargs"]["include_segmentation_targets"] is False
    assert restored.include_depth_targets is True
    assert restored.include_segmentation_targets is False


def test_warmup_recipe_overrides_incompatible_checkpoint_processor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from examples.RoboTwin2.train import RoboTwin2TrainConfig, build_config
    from gr00t.model.gr00t_n1d6.setup import Gr00tN1d6Pipeline
    from transformers import AutoProcessor

    checkpoint_dir = tmp_path / "base-checkpoint"
    checkpoint_dir.mkdir()
    for filename in ("statistics.json", "embodiment_id.json"):
        shutil.copy(FIXTURE_DIR / filename, checkpoint_dir / filename)
    processor_config = json.loads(
        (FIXTURE_DIR / "processor_config.json").read_text()
    )
    processor_config["processor_kwargs"].update(
        {
            "image_crop_size": [111, 112],
            "image_target_size": [221, 222],
            "random_rotation_angle": 45,
            "color_jitter_params": {"brightness": 0.9},
            "shortest_image_edge": 999,
            "crop_fraction": 0.5,
            "max_state_dim": 128,
            "max_action_dim": 128,
            "max_action_horizon": 50,
            "apply_sincos_state_encoding": True,
            "use_relative_action": False,
        }
    )
    (checkpoint_dir / "processor_config.json").write_text(
        json.dumps(processor_config)
    )
    monkeypatch.setattr(
        processing,
        "build_processor",
        lambda model_name, transformers_loading_kwargs: _DummyVlmProcessor(),
    )

    config = build_config(
        RoboTwin2TrainConfig(
            stage="warmup",
            base_model_path=str(checkpoint_dir),
            dataset_path=tmp_path / "dataset",
            output_dir=tmp_path / "output",
        )
    )
    pipeline = Gr00tN1d6Pipeline(config, tmp_path / "output")
    processor = AutoProcessor.from_pretrained(
        checkpoint_dir,
        **pipeline._processor_kwargs(),
        **pipeline.transformers_loading_kwargs,
    )

    assert processor.image_crop_size == config.model.image_crop_size
    assert processor.image_target_size == config.model.image_target_size
    assert processor.random_rotation_angle == config.model.random_rotation_angle
    assert processor.color_jitter_params == config.model.color_jitter_params
    assert processor.shortest_image_edge == config.model.shortest_image_edge
    assert processor.crop_fraction == config.model.crop_fraction
    assert processor.max_state_dim == config.model.max_state_dim
    assert processor.max_action_dim == config.model.max_action_dim
    assert processor.max_action_horizon == 15
    assert processor.apply_sincos_state_encoding is False
    assert processor.use_relative_action is True
    assert processor.include_depth_targets is True
    assert processor.include_segmentation_targets is True


def test_processor_rejects_unknown_recipe_override(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        processing,
        "build_processor",
        lambda model_name, transformers_loading_kwargs: _DummyVlmProcessor(),
    )

    with pytest.raises(
        TypeError,
        match=r"Unexpected processor keyword argument\(s\): experimental_crop",
    ):
        processing.Gr00tN1d6Processor.from_pretrained(
            FIXTURE_DIR,
            experimental_crop=True,
        )


def test_processor_accepts_documented_hugging_face_loading_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        processing,
        "build_processor",
        lambda model_name, transformers_loading_kwargs: _DummyVlmProcessor(),
    )

    processor, unused_kwargs = processing.Gr00tN1d6Processor.from_pretrained(
        FIXTURE_DIR,
        cache_dir=tmp_path / "cache",
        force_download=False,
        local_files_only=True,
        revision="main",
        return_unused_kwargs=True,
    )

    assert isinstance(processor, processing.Gr00tN1d6Processor)
    assert unused_kwargs == {}


def test_targets_replay_rgb_geometry_and_follow_rgb_view_order() -> None:
    np.random.seed(3)
    processor = _make_target_processor(
        include_depth=True,
        include_segmentation=True,
    )
    messages, source_labels = _make_message()

    result = processor(messages)

    rgb_red = result["transformed_rgb"][:, 0].astype(np.float32)
    np.testing.assert_allclose(result["depth_target"], rgb_red, atol=1.0)
    assert result["depth_target"].shape == (2, 4, 4)
    assert result["depth_mask"].shape == (2, 4, 4)
    assert result["segmentation_target"].shape == (2, 4, 4)
    for view_index, labels in enumerate(source_labels):
        assert set(np.unique(result["segmentation_target"][view_index])).issubset(
            set(labels.ravel())
        )
        assert result["segmentation_target"][view_index, 0, 0] == rgb_red[
            view_index, 0, 0
        ]


def test_target_options_gate_target_outputs() -> None:
    processor = _make_target_processor(
        include_depth=False,
        include_segmentation=False,
    )
    messages, _ = _make_message()

    result = processor(messages)

    assert "depth_target" not in result
    assert "depth_mask" not in result
    assert "segmentation_target" not in result


def test_targets_support_the_official_random_rotation_option() -> None:
    processor = _make_target_processor(
        include_depth=True,
        include_segmentation=True,
    )
    processor.train_image_transform = A.ReplayCompose(
        [A.Rotate(limit=(15, 15), border_mode=cv2.BORDER_CONSTANT, p=1.0)]
    )
    messages, source_labels = _make_message()

    result = processor(messages)

    assert result["depth_target"].shape == (2, 4, 4)
    assert result["segmentation_target"].shape == (2, 4, 4)
    assert set(np.unique(result["segmentation_target"][0])).issubset(
        {*source_labels[0].ravel(), 255}
    )


def test_segmentation_rotation_matches_albumentations_mask_replay_pixelwise() -> None:
    segmentation = np.arange(35, dtype=np.uint8).reshape(5, 7)
    transform = A.ReplayCompose(
        [
            A.Rotate(
                limit=(30, 30),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=255,
                p=1.0,
            )
        ]
    )
    replayed = transform(
        image=np.zeros((5, 7, 3), dtype=np.uint8),
        mask=segmentation,
    )

    actual = apply_segmentation_with_replay(
        segmentation,
        replayed["replay"],
        ignore_index=255,
    )

    assert actual.dtype == segmentation.dtype
    np.testing.assert_array_equal(actual, replayed["mask"])


def test_replayed_letterbox_padding_aligns_different_camera_aspects() -> None:
    processor = _make_target_processor(
        include_depth=True,
        include_segmentation=True,
    )
    processor.train_image_transform = A.ReplayCompose(
        [
            LetterBoxPad(),
            A.SmallestMaxSize(max_size=8, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=0.5),
            A.SmallestMaxSize(max_size=8, interpolation=cv2.INTER_AREA),
        ]
    )
    images = {
        "front": [Image.fromarray(np.full((4, 8, 3), 10, dtype=np.uint8))],
        "wrist": [Image.fromarray(np.full((8, 4, 3), 100, dtype=np.uint8))],
    }
    content = SimpleNamespace(
        embodiment=SimpleNamespace(value="robot"),
        states={"joint": np.zeros((1, 1), dtype=np.float32)},
        actions={"joint": np.zeros((1, 1), dtype=np.float32)},
        images=images,
        masks=None,
        text="Move!",
        camera_calibration={
            "view_names": ["front", "wrist"],
            "depth_m": np.asarray(
                [
                    np.full((4, 8), 10, dtype=np.float32),
                    np.full((8, 4), 100, dtype=np.float32),
                ],
                dtype=object,
            ),
            "mesh_segmentation": np.asarray(
                [
                    np.full((4, 8), 1, dtype=np.uint8),
                    np.full((8, 4), 2, dtype=np.uint8),
                ],
                dtype=object,
            ),
        },
    )

    # Ragged camera arrays are normally stored separately by view; exercise
    # the aligned helper directly after reproducing processor view order.
    result = processor._get_vlm_inputs(
        image_keys=["front", "wrist"],
        images=images,
        masks=None,
        image_transform=processor.train_image_transform,
        language="Move!",
        depth_m=content.camera_calibration["depth_m"],
        mesh_segmentation=content.camera_calibration["mesh_segmentation"],
    )

    wrist_rgb_valid = result["transformed_rgb"][1, 0] > 0
    np.testing.assert_array_equal(result["depth_mask"][1].astype(bool), wrist_rgb_valid)
    np.testing.assert_array_equal(
        result["segmentation_target"][1] != 255,
        wrist_rgb_valid,
    )


def test_public_processor_aligns_ragged_per_view_camera_targets() -> None:
    processor = _make_target_processor(
        include_depth=True,
        include_segmentation=True,
    )
    processor.train_image_transform = A.ReplayCompose(
        [
            LetterBoxPad(),
            A.SmallestMaxSize(max_size=8, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=0.5),
            A.SmallestMaxSize(max_size=8, interpolation=cv2.INTER_AREA),
        ]
    )
    images = {
        "front": [Image.fromarray(np.full((4, 8, 3), 10, dtype=np.uint8))],
        "wrist": [Image.fromarray(np.full((8, 4, 3), 100, dtype=np.uint8))],
    }
    content = SimpleNamespace(
        embodiment=SimpleNamespace(value="robot"),
        states={"joint": np.zeros((1, 1), dtype=np.float32)},
        actions={"joint": np.zeros((1, 1), dtype=np.float32)},
        images=images,
        masks=None,
        text="Move!",
        camera_calibration={
            "view_names": ["wrist", "front"],
            "depth_m": [
                np.full((8, 4), 100, dtype=np.float32),
                np.full((4, 8), 10, dtype=np.float32),
            ],
            "mesh_segmentation": [
                np.full((8, 4), 2, dtype=np.uint8),
                np.full((4, 8), 1, dtype=np.uint8),
            ],
        },
    )

    result = processor([{"content": content}])

    assert result["depth_target"].shape == (2, 8, 8)
    assert result["segmentation_target"].shape == (2, 8, 8)
    wrist_rgb_valid = result["transformed_rgb"][1, 0] > 0
    np.testing.assert_array_equal(result["depth_mask"][1].astype(bool), wrist_rgb_valid)
    np.testing.assert_array_equal(
        result["segmentation_target"][1] != 255,
        wrist_rgb_valid,
    )
