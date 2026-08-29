"""Validator contracts shared with the RoboTwin2 LeRobot loader."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from examples.RoboTwin2.validate_dataset import main, validate_dataset
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader


VIEW_NAMES = np.asarray(
    ["image", "left_wrist_image", "right_wrist_image"],
    dtype="<U17",
)


def _metadata(frame_count: int = 2, **overrides) -> dict[str, np.ndarray]:
    values = {
        "intrinsics": np.broadcast_to(
            np.eye(3, dtype=np.float32), (frame_count, 3, 3, 3)
        ).copy(),
        "cam2world": np.broadcast_to(
            np.eye(4, dtype=np.float32), (frame_count, 3, 4, 4)
        ).copy(),
        "depth_u16": np.ones((frame_count, 3, 4, 5), dtype=np.uint16),
        "mesh_segmentation_u16": np.ones(
            (frame_count, 3, 4, 5), dtype=np.uint16
        ),
        "view_names": VIEW_NAMES,
        "depth_unit": np.asarray("mm"),
    }
    values.update(overrides)
    return values


def _write_metadata(tmp_path: Path, values: dict[str, np.ndarray]) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset"
    metadata_path = dataset_path / "camera_meta" / "episode_000000.npz"
    metadata_path.parent.mkdir(parents=True)
    np.savez(metadata_path, **values)
    meta_path = dataset_path / "meta"
    meta_path.mkdir()
    (meta_path / "info.json").write_text(
        json.dumps(
            {
                "camera_meta_path": "camera_meta/episode_{episode_index:06d}.npz",
                "chunks_size": 1000,
            }
        )
    )
    (meta_path / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 2}) + "\n"
    )
    return dataset_path, metadata_path


def _write_loader_dataset(
    tmp_path: Path,
    episode_count: int = 2,
) -> tuple[Path, list[Path]]:
    dataset_path = tmp_path / "dataset"
    meta_path = dataset_path / "meta"
    meta_path.mkdir(parents=True)
    camera_meta_path = (
        "loader_camera_meta/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"
    )
    (meta_path / "info.json").write_text(
        json.dumps({"camera_meta_path": camera_meta_path, "chunks_size": 1})
    )
    (meta_path / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": episode_index, "length": 2}) + "\n"
            for episode_index in range(episode_count)
        )
    )
    metadata_paths = []
    for episode_index in range(episode_count):
        metadata_path = dataset_path / camera_meta_path.format(
            episode_chunk=episode_index,
            episode_index=episode_index,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(metadata_path, **_metadata())
        metadata_paths.append(metadata_path)
    return dataset_path, metadata_paths


def test_main_resolves_every_episode_from_info_camera_meta_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, _ = _write_loader_dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", ["validate_dataset.py", str(dataset_path)])

    assert main() == 0
    assert "validated 2 loader-compatible camera-meta files" in capsys.readouterr().out


def test_main_reports_the_referenced_camera_metadata_path_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, _ = _write_loader_dataset(tmp_path, episode_count=1)
    camera_meta_path = "referenced/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"
    (dataset_path / "meta" / "info.json").write_text(
        json.dumps({"camera_meta_path": camera_meta_path, "chunks_size": 1})
    )
    expected_path = dataset_path / "referenced/chunk-000/episode_000000.npz"
    monkeypatch.setattr(sys, "argv", ["validate_dataset.py", str(dataset_path)])

    assert main() == 1
    assert str(expected_path) in capsys.readouterr().err


def test_main_reports_when_one_episode_camera_metadata_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, metadata_paths = _write_loader_dataset(tmp_path)
    missing_path = metadata_paths[-1]
    missing_path.unlink()
    monkeypatch.setattr(sys, "argv", ["validate_dataset.py", str(dataset_path)])

    assert main() == 1
    assert str(missing_path) in capsys.readouterr().err


def test_validated_metadata_is_loadable_by_robotwin_loader(tmp_path: Path) -> None:
    dataset_path, metadata_path = _write_metadata(tmp_path, _metadata())

    assert validate_dataset(dataset_path) == 1
    parsed = LeRobotEpisodeLoader._parse_camera_meta_npz(
        metadata_path,
        np.asarray([0, 1]),
    )

    assert parsed["depth_m"].shape == (2, 3, 4, 5)
    assert parsed["view_names"] == VIEW_NAMES.tolist()


def test_main_rejects_camera_metadata_shorter_than_declared_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, metadata_path = _write_metadata(tmp_path, _metadata())
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes_path.write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    monkeypatch.setattr(sys, "argv", ["validate_dataset.py", str(dataset_path)])

    assert main() == 1
    error = capsys.readouterr().err
    assert str(metadata_path) in error
    assert "declared length 3" in error
    assert "intrinsics has 2 frames" in error


@pytest.mark.parametrize(
    "field",
    ("intrinsics", "cam2world", "depth_u16", "mesh_segmentation_u16"),
)
def test_validator_requires_each_indexed_field_to_match_declared_length(
    tmp_path: Path,
    field: str,
) -> None:
    values = _metadata(frame_count=3)
    values[field] = _metadata()[field]
    dataset_path, metadata_path = _write_metadata(tmp_path, values)
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes_path.write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")

    with pytest.raises(
        ValueError,
        match=rf"{metadata_path}.*declared length 3.*{field} has 2 frames",
    ):
        validate_dataset(dataset_path)


@pytest.mark.parametrize(
    ("episode", "message"),
    (
        ({"episode_index": 0}, "episode 0 is missing required length"),
        (
            {"episode_index": 0, "length": "not-a-length"},
            "episode 0 has invalid length 'not-a-length'",
        ),
    ),
)
def test_validator_reports_missing_or_invalid_episode_length(
    tmp_path: Path,
    episode: dict[str, object],
    message: str,
) -> None:
    dataset_path, _ = _write_metadata(tmp_path, _metadata())
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes_path.write_text(json.dumps(episode) + "\n")

    with pytest.raises(ValueError, match=rf"{episodes_path}.*{message}"):
        validate_dataset(dataset_path)


@pytest.mark.parametrize(
    "missing_field",
    ("intrinsics", "cam2world", "depth_u16", "view_names", "depth_unit"),
)
def test_validator_reports_missing_loader_field(
    tmp_path: Path,
    missing_field: str,
) -> None:
    values = _metadata()
    values.pop(missing_field)
    dataset_path, _ = _write_metadata(tmp_path, values)

    with pytest.raises(ValueError, match=rf"missing required field.*{missing_field}"):
        validate_dataset(dataset_path)


def test_validator_reports_missing_segmentation_target(tmp_path: Path) -> None:
    values = _metadata()
    values.pop("mesh_segmentation_u16")
    dataset_path, _ = _write_metadata(tmp_path, values)

    with pytest.raises(ValueError, match="missing required segmentation field"):
        validate_dataset(dataset_path)


def test_validator_rejects_segmentation_without_valid_recipe_label(
    tmp_path: Path,
) -> None:
    segmentation = np.full((2, 3, 4, 5), 65535, dtype=np.uint16)
    dataset_path, metadata_path = _write_metadata(
        tmp_path,
        _metadata(mesh_segmentation_u16=segmentation),
    )

    with pytest.raises(
        ValueError,
        match=rf"{metadata_path}.*no valid segmentation supervision.*\[0, 177\]",
    ):
        validate_dataset(dataset_path)


def test_validator_accepts_recipe_label_boundaries_and_ignore_value(
    tmp_path: Path,
) -> None:
    segmentation = np.full((2, 3, 4, 5), 255, dtype=np.uint16)
    segmentation[0, 0, 0, 0] = 0
    segmentation[1, 2, 3, 4] = 177
    dataset_path, _ = _write_metadata(
        tmp_path,
        _metadata(mesh_segmentation_u16=segmentation),
    )

    assert validate_dataset(dataset_path) == 1


def test_validator_rejects_depth_without_positive_supervision(tmp_path: Path) -> None:
    depth_u16 = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    dataset_path, metadata_path = _write_metadata(
        tmp_path,
        _metadata(depth_u16=depth_u16),
    )

    with pytest.raises(
        ValueError,
        match=rf"{metadata_path}.*depth_u16.*no valid depth supervision",
    ):
        validate_dataset(dataset_path)


def test_validator_accepts_depth_with_invalid_and_positive_pixels(tmp_path: Path) -> None:
    depth_u16 = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    depth_u16[1, 2, 3, 4] = 1
    dataset_path, _ = _write_metadata(
        tmp_path,
        _metadata(depth_u16=depth_u16),
    )

    assert validate_dataset(dataset_path) == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"intrinsics": np.zeros((2, 3, 4, 4))}, "intrinsics.*expected"),
        ({"cam2world": np.zeros((2, 3, 3, 3))}, "cam2world.*expected"),
        ({"depth_u16": np.zeros((2, 2, 4, 5), dtype=np.uint16)}, "depth_u16.*expected"),
        (
            {"mesh_segmentation_u16": np.zeros((2, 3, 5, 5), dtype=np.uint16)},
            "segmentation.*match depth_u16",
        ),
        (
            {"mesh_segmentation_u16": np.asarray(1, dtype=np.uint16)},
            "mesh_segmentation_u16.*no frame dimension",
        ),
        (
            {"view_names": np.asarray(["left_wrist_image", "image", "right_wrist_image"])},
            "view_names.*expected.*image.*left_wrist_image.*right_wrist_image",
        ),
        ({"depth_unit": np.asarray("cm")}, "depth_unit.*mm or m"),
    ),
)
def test_validator_reports_invalid_loader_shape_or_value(
    tmp_path: Path,
    overrides: dict[str, np.ndarray],
    message: str,
) -> None:
    dataset_path, _ = _write_metadata(tmp_path, _metadata(**overrides))

    with pytest.raises(ValueError, match=message):
        validate_dataset(dataset_path)
