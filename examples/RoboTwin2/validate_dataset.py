#!/usr/bin/env python3
"""Validate the camera metadata required by the fixed RoboTwin2 loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


VIEW_NAMES = ("image", "left_wrist_image", "right_wrist_image")
SEGMENTATION_KEYS = (
    "mesh_segmentation",
    "mesh_segmentation_u8",
    "mesh_segmentation_u16",
    "segmentation",
    "segmentation_u8",
)
REQUIRED_FIELDS = ("intrinsics", "cam2world", "depth_u16", "view_names", "depth_unit")
SEGMENTATION_NUM_CLASSES = 178


def _require_shape(path: Path, field: str, array: np.ndarray, suffix: tuple[int, ...]) -> None:
    expected_rank = 2 + len(suffix)
    if array.ndim != expected_rank or array.shape[1] != 3 or array.shape[2:] != suffix:
        dimensions = ", ".join(str(value) for value in suffix)
        raise ValueError(
            f"{path}: {field} has shape {array.shape}; expected [frames, 3, {dimensions}]"
        )


def _validate_camera_metadata(path: Path, declared_length: int) -> None:
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in REQUIRED_FIELDS if field not in data]
        if missing:
            raise ValueError(f"{path}: missing required field(s): {', '.join(missing)}")
        segmentation_key = next((key for key in SEGMENTATION_KEYS if key in data), None)
        if segmentation_key is None:
            raise ValueError(
                f"{path}: missing required segmentation field; expected one of "
                f"{', '.join(SEGMENTATION_KEYS)}"
            )

        intrinsics = np.asarray(data["intrinsics"])
        cam2world = np.asarray(data["cam2world"])
        depth_u16 = np.asarray(data["depth_u16"])
        segmentation = np.asarray(data[segmentation_key])
        view_names = tuple(str(value) for value in data["view_names"].tolist())
        depth_unit_array = np.asarray(data["depth_unit"])

    _require_shape(path, "intrinsics", intrinsics, (3, 3))
    _require_shape(path, "cam2world", cam2world, (4, 4))
    if depth_u16.ndim != 4 or depth_u16.shape[1] != 3:
        raise ValueError(
            f"{path}: depth_u16 has shape {depth_u16.shape}; "
            "expected [frames, 3, height, width]"
        )
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"{path}: depth_u16 must have dtype uint16, got {depth_u16.dtype}")
    for field, array in (
        ("intrinsics", intrinsics),
        ("cam2world", cam2world),
        ("depth_u16", depth_u16),
        (segmentation_key, segmentation),
    ):
        if array.ndim == 0:
            raise ValueError(f"{path}: {field} has no frame dimension")
        if array.shape[0] != declared_length:
            raise ValueError(
                f"{path}: declared length {declared_length}, but {field} has "
                f"{array.shape[0]} frames"
            )
    if segmentation.shape != depth_u16.shape:
        raise ValueError(
            f"{path}: segmentation shape {segmentation.shape} must match "
            f"depth_u16 shape {depth_u16.shape}"
        )
    if not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError(
            f"{path}: segmentation must have an integer dtype, got {segmentation.dtype}"
        )
    if not np.any(depth_u16 > 0):
        raise ValueError(f"{path}: depth_u16 has no valid depth supervision (> 0)")
    if not np.any(
        (segmentation >= 0) & (segmentation < SEGMENTATION_NUM_CLASSES)
    ):
        raise ValueError(
            f"{path}: no valid segmentation supervision in label range [0, 177]"
        )

    frame_view_shape = depth_u16.shape[:2]
    for field, array in (("intrinsics", intrinsics), ("cam2world", cam2world)):
        if array.shape[:2] != frame_view_shape:
            raise ValueError(
                f"{path}: {field} frame/view shape {array.shape[:2]} does not match "
                f"depth_u16 {frame_view_shape}"
            )
    if view_names != VIEW_NAMES:
        raise ValueError(
            f"{path}: view_names are {list(view_names)}; expected exact order "
            f"{list(VIEW_NAMES)}"
        )
    if depth_unit_array.size == 0:
        raise ValueError(f"{path}: depth_unit is empty; expected mm or m")
    value = depth_unit_array.item() if depth_unit_array.size == 1 else depth_unit_array.flat[0]
    depth_unit = str(value)
    if depth_unit not in {"mm", "m"}:
        raise ValueError(f"{path}: depth_unit must be mm or m, got {depth_unit!r}")


def validate_dataset(dataset_path: Path) -> int:
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"

    with info_path.open() as file:
        info = json.load(file)
    with episodes_path.open() as file:
        episodes = [json.loads(line) for line in file]

    camera_meta_pattern = info.get("camera_meta_path")
    if not isinstance(camera_meta_pattern, str) or not camera_meta_pattern:
        raise ValueError(f"{info_path}: missing required camera_meta_path")
    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError(f"{info_path}: chunks_size must be a positive integer")
    if not episodes:
        raise ValueError(f"{episodes_path}: contains no episodes")

    files = []
    for episode_index in range(len(episodes)):
        episode = episodes[episode_index]
        if "length" not in episode:
            raise ValueError(
                f"{episodes_path}: episode {episode_index} is missing required length"
            )
        try:
            declared_length = int(episode["length"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{episodes_path}: episode {episode_index} has invalid length "
                f"{episode['length']!r}"
            ) from exc
        episode_chunk = episode_index // chunks_size
        try:
            relative_path = camera_meta_pattern.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"{info_path}: invalid camera_meta_path: {exc}") from exc
        path = dataset_path / relative_path
        if not path.exists():
            raise ValueError(
                f"{path}: camera metadata file not found for episode {episode_index}"
            )
        _validate_camera_metadata(path, declared_length)
        files.append(path)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    args = parser.parse_args()
    try:
        count = validate_dataset(args.dataset_path)
    except (OSError, ValueError) as exc:
        print(f"[V-Link] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[V-Link] validated {count} loader-compatible camera-meta files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
