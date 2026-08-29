#!/usr/bin/env python
"""
LeRobot Dataset Loader

A simplified, clean implementation for loading LeRobot datasets with video support.
This module provides the core functionality for loading episodes from LeRobot format datasets,
handling metadata parsing, video decoding, and data preprocessing for VLA training.

The LeRobotEpisodeLoader serves as the foundation for higher-level dataset classes,
providing episode-level data access with support for multi-modal data including:
- Video frames from multiple camera views
- Proprioceptive state information
- Action sequences
- Language instructions/annotations

Returns messages with VLAStepData as defined in types.py.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gr00t.data.types import ModalityConfig
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, load_initial_actions
from gr00t.utils.video_utils import get_frames_by_indices


logger = logging.getLogger(__name__)


# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"
LEROBOT_CALIBRATION_FILENAME = "calibration.json"  # fallback static-calibration source

ALLOWED_MODALITIES = ["video", "state", "action", "language", "mask", "camera_calibration"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}

LANG_KEYS = ["task", "sub_task"]


# ---------------------------------------------------------------------------
# Online transforms
#
# Ported from multi_view_vla/gr00t/data/dataset/lerobot_episode_loader.py.
# These convert raw parquet state/action vectors in-memory (e.g. quaternion
# rotations -> 6D rotation) so the resulting layout matches the post-transform
# indices declared in modality.json. The result has the same index layout as
# the offline convert_eef_quat_to_rot6d.py convert.
# ---------------------------------------------------------------------------


def _normalize_rot6d_convention(convention: str | None) -> str:
    value = "row" if convention is None else str(convention).strip().lower()
    if value not in ("row", "col"):
        raise ValueError(
            f"Unsupported rot6d convention '{convention}', expected one of: row, col"
        )
    return value


def _matrix_to_rot6d(rot_mat: np.ndarray, convention: str = "row") -> np.ndarray:
    rot = np.asarray(rot_mat, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {rot.shape}")
    convention = _normalize_rot6d_convention(convention)
    if convention == "row":
        return np.concatenate([rot[0, :], rot[1, :]], axis=0)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0)


def _rot6d_to_matrix(rot6d: np.ndarray, convention: str = "row") -> np.ndarray:
    vec6 = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    if vec6.shape[0] != 6:
        raise ValueError(f"rot6d must have length 6, got {vec6.shape[0]}")
    convention = _normalize_rot6d_convention(convention)

    vec6 = vec6.reshape(2, 3)
    basis1 = vec6[0]
    basis2 = vec6[1]
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 - np.dot(basis1, basis2) * basis1
    basis2 = basis2 / np.linalg.norm(basis2)
    basis3 = np.cross(basis1, basis2)
    if convention == "row":
        return np.vstack([basis1, basis2, basis3])
    return np.column_stack([basis1, basis2, basis3])


def _quat_xyzw_to_rot6d(quat_xyzw: np.ndarray, convention: str = "row") -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 6D rotation."""
    rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()
    return _matrix_to_rot6d(rot_mat, convention=convention)


def _convert_eef9_rot6d_convention(
    eef9: np.ndarray,
    src_convention: str,
    dst_convention: str,
) -> np.ndarray:
    src = _normalize_rot6d_convention(src_convention)
    dst = _normalize_rot6d_convention(dst_convention)
    arr = np.asarray(eef9, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 9:
        raise ValueError(f"eef9 must have length 9, got {arr.shape[0]}")
    if src == dst:
        return arr.astype(np.float32)
    rot = _rot6d_to_matrix(arr[3:], convention=src)
    rot6d = _matrix_to_rot6d(rot, convention=dst)
    return np.concatenate([arr[:3], rot6d], axis=0).astype(np.float32)


def _transform_eef_quat_to_rot6d(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    In-memory EEF quaternion -> rot6d conversion on a DataFrame.

    For each configured column (e.g. "observation.state", "action"):
    1. Extract left/right quaternion and position from source indices
    2. Convert quaternion (xyzw) to rot6d
    3. Truncate to original_length then append [left_eef(9), right_eef(9)]

    The result has the same index layout as the offline convert_eef_quat_to_rot6d.py.
    """
    columns_cfg = config.get("columns", {})
    convention = _normalize_rot6d_convention(config.get("convention", "row"))

    for col_name, idx_cfg in columns_cfg.items():
        if col_name not in df.columns:
            continue

        lq = idx_cfg["left_quat"]
        rq = idx_cfg["right_quat"]
        lp = idx_cfg["left_pos"]
        rp = idx_cfg["right_pos"]
        orig_len = idx_cfg.get("original_length")

        raw_series = df[col_name].tolist()
        extended = []
        for row_vec in raw_series:
            arr = np.asarray(row_vec, dtype=np.float32)
            if orig_len is not None:
                arr = arr[:orig_len]

            left_rot6d = _quat_xyzw_to_rot6d(arr[lq[0] : lq[1]], convention=convention)
            right_rot6d = _quat_xyzw_to_rot6d(arr[rq[0] : rq[1]], convention=convention)
            left_eef = np.concatenate([arr[lp[0] : lp[1]], left_rot6d])
            right_eef = np.concatenate([arr[rp[0] : rp[1]], right_rot6d])
            extended.append(np.concatenate([arr, left_eef, right_eef]))

        df[col_name] = extended

    return df


def _parse_eef9_ranges(ranges_cfg: Any) -> list[tuple[int, int]]:
    if ranges_cfg is None:
        return []
    if isinstance(ranges_cfg, (list, tuple)):
        if len(ranges_cfg) == 2 and all(
            isinstance(v, (int, np.integer)) for v in ranges_cfg
        ):
            return [(int(ranges_cfg[0]), int(ranges_cfg[1]))]
        ranges: list[tuple[int, int]] = []
        for item in ranges_cfg:
            ranges.extend(_parse_eef9_ranges(item))
        return ranges
    if isinstance(ranges_cfg, dict):
        if "indices" in ranges_cfg:
            return _parse_eef9_ranges(ranges_cfg["indices"])
        if "ranges" in ranges_cfg:
            return _parse_eef9_ranges(ranges_cfg["ranges"])
        ranges = []
        for value in ranges_cfg.values():
            ranges.extend(_parse_eef9_ranges(value))
        return ranges
    raise ValueError(f"Invalid eef9 range config: {ranges_cfg!r}")


def _transform_eef_rot6d_convert_convention(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert eef9 rot6d convention for configured columns/ranges."""
    src = _normalize_rot6d_convention(config.get("src_convention", "col"))
    dst = _normalize_rot6d_convention(config.get("dst_convention", "row"))
    columns_cfg = config.get("columns", {})
    if src == dst:
        return df

    for col_name, ranges_cfg in columns_cfg.items():
        if col_name not in df.columns:
            continue
        ranges = _parse_eef9_ranges(ranges_cfg)
        if not ranges:
            continue

        converted_rows = []
        for row_vec in df[col_name].tolist():
            arr = np.asarray(row_vec, dtype=np.float32).copy()
            for start, end in ranges:
                if end - start != 9:
                    raise ValueError(
                        f"EEF segment must have dim 9, got [{start}, {end}) in column '{col_name}'"
                    )
                arr[start:end] = _convert_eef9_rot6d_convention(
                    arr[start:end],
                    src_convention=src,
                    dst_convention=dst,
                )
            converted_rows.append(arr)
        df[col_name] = converted_rows

    return df


def apply_online_transforms(df: pd.DataFrame, transforms: list[dict]) -> pd.DataFrame:
    """Apply a list of online transform configs to a raw parquet DataFrame.

    Args:
        df: Raw DataFrame loaded from parquet.
        transforms: List of transform dicts from modality.json["transforms"].

    Returns:
        Transformed DataFrame (vectors may be extended with new dimensions).
    """
    for transform_cfg in transforms:
        transform_type = transform_cfg.get("type")
        if transform_type == "eef_quat_to_rot6d":
            df = _transform_eef_quat_to_rot6d(df, transform_cfg)
        elif transform_type == "eef_rot6d_convert_convention":
            df = _transform_eef_rot6d_convert_convention(df, transform_cfg)
        else:
            logger.warning(f"Unknown transform type '{transform_type}', skipping")
    return df


def _rec_defaultdict() -> defaultdict:
    """Factory that creates an infinitely nestable defaultdict."""
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree):
    """Recursively turn a (nested) defaultdict into a regular dict."""
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


class LeRobotEpisodeLoader:
    """
    Episode-level data loader for LeRobot format datasets.

    This class handles the loading and preprocessing of individual episodes from LeRobot datasets.
    It manages metadata parsing, video decoding, and data extraction across multiple modalities
    (video, state, action, language) while maintaining compatibility with the VLA training pipeline.

    Key responsibilities:
    - Parse LeRobot metadata files (info.json, episodes.jsonl, etc.)
    - Load and decode video data using configurable backends
    - Extract and process multi-modal data according to modality configurations
    - Provide dataset statistics for normalization
    - Handle initial action loading for policy initialization

    Args:
        dataset_path: Path to dataset root directory containing meta/ and data files
        modality_configs: Dictionary mapping modality names to ModalityConfig objects
                         that specify temporal sampling and data keys to load
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ... )
        >>> episode_data = loader[0]  # Load first episode as DataFrame
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize LeRobot episode loader with dataset path and modality configurations.

        The initialization process involves:
        1. Loading all metadata files from the dataset
        2. Parsing and validating modality configurations
        3. Computing effective episode lengths based on action horizon
        """
        self.dataset_path = Path(dataset_path)
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Load metadata files and parse dataset structure
        self._load_metadata()

        # Set up modality configs after metadata is loaded
        self.modality_configs = self._parse_and_validate_modality_configs(modality_configs)

        # Compute effective episode lengths accounting for action horizon
        self.episode_lengths = self.get_episode_lengths()

    def _load_metadata(self) -> None:
        """
        Load all metadata files including dataset statistics.

        Parses the standard LeRobot metadata structure:
        - info.json: Dataset configuration and file patterns
        - episodes.jsonl: Per-episode metadata (length, timestamps, etc.)
        - tasks.jsonl: Task descriptions and mappings
        - modality.json: Modality structure and data layout
        - stats.json: Dataset statistics for normalization
        """
        meta_dir = self.dataset_path / LEROBOT_META_DIR_NAME

        # Load dataset configuration
        info_path = meta_dir / LEROBOT_INFO_FILENAME
        with open(info_path, "r") as f:
            self.info_meta = json.load(f)

        # Load episode metadata (one episode per line)
        episodes_path = meta_dir / LEROBOT_EPISODES_FILENAME
        with open(episodes_path, "r") as f:
            self.episodes_metadata = [json.loads(line) for line in f]

        # Load task descriptions and create mapping
        tasks_path = meta_dir / LEROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks_data = [json.loads(line) for line in f]
            self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        # Load modality structure information
        modality_path = meta_dir / LEROBOT_MODALITY_FILENAME
        with open(modality_path, "r") as f:
            self.modality_meta = json.load(f)

        # Load dataset statistics for normalization
        stats_path = meta_dir / LEROBOT_STATS_FILE_NAME
        assert stats_path.exists(), (
            f"{stats_path} does not exist for {self.dataset_path}, please use gr00t/data/stats.py to generate it"
        )
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        relative_stats_path = meta_dir / LEROBOT_RELATIVE_STATS_FILE_NAME
        if relative_stats_path.exists():
            with open(relative_stats_path, "r") as f:
                self.stats["relative_action"] = json.load(f)

        # Extract key configuration parameters
        self.feature_config = self.info_meta.get("features", {})
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.mask_path_pattern = self.info_meta.get("mask_path")
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta.get("fps", 30)

    def get_episode_lengths(self):
        """
        Compute original episode lengths.

        Returns:
            List of original episode lengths
        """
        episode_lengths = []
        for ep_meta in self.episodes_metadata:
            episode_lengths.append(int(ep_meta["length"]))
        return episode_lengths

    def get_episode_length(self, idx: int) -> int:
        """Get the length of a specific episode."""
        return self.episode_lengths[idx]

    def _parse_and_validate_modality_configs(
        self,
        modality_configs: dict[str, ModalityConfig],
    ) -> dict[str, ModalityConfig]:
        """
        Parse and validate modality configurations, filling in defaults where needed.

        For missing modality configs, creates default configurations:
        - video: All available camera views with single timestep
        - state: All available state keys with single timestep
        - action: All available action keys with 16-step horizon
        - language: Must be explicitly configured if needed

        Args:
            modality_configs: User-provided modality configurations

        Returns:
            Complete and validated modality configurations

        Raises:
            ValueError: If invalid modalities are specified
            AssertionError: If language modality configuration is invalid
        """
        # Validate all modality configurations
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")
            if modality == "language":
                # Language modality has special constraints
                assert len(modality_configs[modality].modality_keys) == 1, (
                    "Language modality must have exactly one key"
                )
                assert modality_configs[modality].delta_indices == [0], (
                    "Only single timestep is supported for language modality"
                )

        # Validate video modality_keys against modality.json.
        # Each key in modality_configs["video"].modality_keys must exist in
        # modality.json["video"], otherwise _load_video_data will fail with
        # a confusing KeyError when trying to resolve the original video key.
        if "video" in modality_configs and "video" in self.modality_meta:
            config_keys = set(modality_configs["video"].modality_keys)
            meta_keys = set(self.modality_meta["video"].keys())
            missing_keys = config_keys - meta_keys
            if missing_keys:
                raise ValueError(
                    f"Video modality_keys {sorted(missing_keys)} in modality_config "
                    f"not found in modality.json. "
                    f"modality_config expects: {sorted(config_keys)}, "
                    f"modality.json defines: {sorted(meta_keys)}. "
                    f"Please ensure modality.json and your modality_config use the "
                    f"same video key names."
                )

        return modality_configs

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _extract_joint_groups(
        self,
        df: pd.DataFrame,
        joint_groups: list[str],
        modality_type: str = "state",
    ) -> pd.DataFrame:
        """
        Extract specific joint groups from data arrays based on modality metadata.

        Uses the modality metadata to slice the appropriate indices from the raw data arrays,
        allowing for flexible joint group extraction (e.g., arm joints, gripper state).

        Args:
            df: DataFrame containing the raw episode data
            joint_groups: List of joint group names to extract (e.g., ["arm", "gripper"])
            modality_type: Type of modality ("state" or "action")

        Returns:
            DataFrame with columns for each requested joint group containing sliced arrays
        """
        modality_info = self.modality_meta.get(modality_type, {})
        joint_data = pd.DataFrame()

        for group_name in joint_groups:
            if group_name in modality_info:
                group_info = modality_info[group_name]
                start_idx = group_info["start"]
                end_idx = group_info["end"]
                original_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                # Slice the array data for this joint group
                if isinstance(df[original_key].iloc[0], np.ndarray):
                    joint_data[group_name] = df[original_key].map(lambda x: x[start_idx:end_idx])
                else:
                    joint_data[group_name] = df[original_key]  # for strings and scalars
            else:
                print(
                    f"Warning: Joint group '{group_name}' not found in {modality_type} modality. Available groups: {list(modality_info.keys())}"
                )

        return joint_data

    def _uses_eef_modalities(self) -> bool:
        """Whether current loader config requires EEF-specific online transforms."""
        # Heuristic 1: modality keys contain "eef"
        for modality in ("state", "action"):
            cfg = self.modality_configs.get(modality)
            if cfg is None:
                continue
            for key in cfg.modality_keys:
                if "eef" in str(key).lower():
                    return True

        # Heuristic 2: action configs explicitly declare EEF type/format
        action_cfg = self.modality_configs.get("action")
        if action_cfg is None or action_cfg.action_configs is None:
            return False
        for cfg in action_cfg.action_configs:
            type_name = str(getattr(cfg.type, "name", "")).upper()
            format_name = str(getattr(cfg.format, "name", "")).upper()
            if type_name == "EEF" or format_name == "XYZ_ROT6D":
                return True
        return False

    def _apply_online_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply modality.json["transforms"] to the raw parquet DataFrame.

        EEF-specific transforms (eef_quat_to_rot6d, eef_rot6d_convert_convention)
        are only applied when the loader config actually consumes EEF modalities;
        otherwise they are skipped so non-EEF configs are unaffected.
        """
        transforms = self.modality_meta.get("transforms", [])
        if not transforms:
            return df

        if not self._uses_eef_modalities():
            filtered = []
            skipped = []
            for t in transforms:
                ttype = str(t.get("type", ""))
                if ttype in ("eef_quat_to_rot6d", "eef_rot6d_convert_convention"):
                    skipped.append(ttype)
                    continue
                filtered.append(t)
            if skipped:
                logger.info(
                    "Skipping EEF online transforms for non-EEF modality config: %s",
                    skipped,
                )
            transforms = filtered

        return apply_online_transforms(df, transforms)

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        """
        Load and process parquet data for a specific episode.

        Handles the complete data loading pipeline:
        1. Load raw parquet file based on chunking structure
        2. Process language annotations (convert task indices to strings)
        3. Extract state and action joint groups

        Args:
            episode_index: Index of the episode to load

        Returns:
            Processed DataFrame with all modality data
        """
        # Load raw parquet data using chunking pattern
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        parquet_path = self.dataset_path / parquet_filename
        original_df = pd.read_parquet(parquet_path)

        # Apply online transforms (e.g. quat->rot6d) before joint group extraction
        # so that the modality.json start/end slice indices refer to the
        # post-transform layout.
        original_df = self._apply_online_transforms(original_df)

        loaded_df = pd.DataFrame()

        # Process language annotations (convert task indices to task strings)
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                # these keys will be loaded separately from episodes.jsonl
                if key in LANG_KEYS:
                    continue
                assert key.startswith("annotation.")
                subkey = key.replace("annotation.", "")
                assert subkey in self.modality_meta["annotation"], (
                    f"Key {subkey} not found in language modality"
                )
                original_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                loaded_df[f"language.{key}"] = original_df[original_key].apply(
                    lambda x: self.tasks_map[x]
                )

        # Extract joint groups for state and action modalities
        for modality_type in ["state", "action"]:
            if modality_type not in self.modality_configs:
                continue
            joint_groups_df = self._extract_joint_groups(
                original_df,
                self.modality_configs[modality_type].modality_keys,
                modality_type,
            )
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]

        return loaded_df

    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load video data for all configured camera views at specified indices.

        Uses the configured video backend to decode video frames at the exact indices
        needed for the episode, supporting multiple camera views simultaneously.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of indices to extract frames at

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames
        """
        video_data = {}

        if not self.video_path_pattern or "video" not in self.modality_configs:
            return video_data

        chunk_idx = episode_index // self.chunk_size
        image_keys = self.modality_configs["video"].modality_keys

        for image_key in image_keys:
            # Resolve the original key used in video file naming
            original_key = self.modality_meta["video"][image_key].get(
                "original_key", f"observation.images.{image_key}"
            )
            assert original_key in self.feature_config, (
                f"Original key {original_key} not found in feature config"
            )

            # Construct video file path using pattern
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx,
                video_key=original_key,
                episode_index=episode_index,
            )
            video_path = self.dataset_path / video_filename

            # Decode video frames at specified timestamps
            video_data[image_key] = get_frames_by_indices(
                str(video_path),
                indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs or {},
            )

        return video_data

    def _load_mask_file(self, mask_path: Path, indices: np.ndarray) -> np.ndarray:
        """Load masks from npz/npy file at specified indices."""
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")
        suffix = mask_path.suffix.lower()
        if suffix not in {".npz", ".npy"}:
            raise ValueError(f"Only .npz or .npy mask files are supported: {mask_path}")

        if suffix == ".npy":
            masks = np.load(mask_path)
        else:
            npz_data = np.load(mask_path)
            if "arr_0" in npz_data:
                masks = npz_data["arr_0"]
            elif len(npz_data.files) == 1:
                masks = npz_data[npz_data.files[0]]
            else:
                raise ValueError(f"Mask npz must contain a single array or 'arr_0': {mask_path}")

        if masks.ndim == 2:
            masks = masks[None, ...]

        return masks[indices]

    def _load_mask_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load mask data for all configured mask views at specified indices.
        """
        mask_data = {}

        if not self.mask_path_pattern or "mask" not in self.modality_configs:
            return mask_data

        chunk_idx = episode_index // self.chunk_size
        mask_keys = self.modality_configs["mask"].modality_keys

        for mask_key in mask_keys:
            mask_meta = self.modality_meta.get("mask", {}).get(mask_key, {})
            original_key = mask_meta.get("original_key", mask_key)
            mask_filename = self.mask_path_pattern.format(
                episode_chunk=chunk_idx,
                episode_index=episode_index,
                mask_key=original_key,
                video_key=original_key,
            )
            mask_path = self.dataset_path / mask_filename
            mask_data[mask_key] = self._load_mask_file(mask_path, indices)

        return mask_data

    # ------------------------------------------------------------------
    # Camera calibration (independent of __getitem__ / DataFrame path)
    # ------------------------------------------------------------------
    def get_camera_calibration(
        self, episode_index: int, indices: np.ndarray
    ) -> dict[str, Any] | None:
        """Load per-frame camera calibration for an episode.

        Returns a normalized dict::

            {
                "K":          float32 [N, V, 3, 3]   # OpenCV intrinsics
                "cam2world":  float32 [N, V, 4, 4]   # OpenGL, verbatim from npz
                "depth_m":    float32 [N, V, H, W] | None   # mm -> m, depth_u16==0 -> NaN
                "mesh_segmentation": int64 [N, V, H, W] | None
                "view_names": list[str]   length V
            }

        Field names: raw npz ``intrinsics`` -> ``K``, raw ``depth_u16`` (mm) ->
        ``depth_m`` (m, NaN where invalid). ``cam2world`` keeps its raw name and
        OpenGL frame; this loader does NOT invert or flip it.

        Source preference: ``info.json["camera_meta_path"]`` -> ``meta/calibration.json``.
        If neither exists, returns ``None`` with a warning *unless*
        ``camera_calibration`` is in ``modality_configs``, in which case
        the missing source is a hard failure.
        """
        has_consumer = "camera_calibration" in self.modality_configs
        indices = np.asarray(indices)

        camera_meta_pattern = self.info_meta.get("camera_meta_path")
        npz_path: Path | None = None
        if camera_meta_pattern:
            chunk_idx = episode_index // self.chunk_size
            npz_path = self.dataset_path / camera_meta_pattern.format(
                episode_chunk=chunk_idx, episode_index=episode_index
            )
            if npz_path.exists():
                calib = self._parse_camera_meta_npz(npz_path, indices)
                self._validate_calibration_view_names(calib["view_names"])
                return calib

        json_path = self.dataset_path / LEROBOT_META_DIR_NAME / LEROBOT_CALIBRATION_FILENAME
        if json_path.exists():
            calib = self._parse_camera_meta_json(json_path, indices)
            self._validate_calibration_view_names(calib["view_names"])
            return calib

        msg = (
            f"camera calibration source not found for episode {episode_index}: "
            f"tried {npz_path if camera_meta_pattern else '<info.json has no camera_meta_path>'} "
            f"and {json_path}"
        )
        if has_consumer:
            raise FileNotFoundError(msg)
        logger.warning("%s; returning None.", msg)
        return None

    @staticmethod
    def _parse_camera_meta_npz(npz_path: Path, indices: np.ndarray) -> dict[str, Any]:
        """Parse a per-episode camera_meta npz; pure function for unit tests."""
        with np.load(npz_path, allow_pickle=False) as data:
            intrinsics = np.asarray(data["intrinsics"][indices], dtype=np.float32)
            cam2world = np.asarray(data["cam2world"][indices], dtype=np.float32)
            depth_u16 = np.asarray(data["depth_u16"][indices])
            mesh_segmentation = None
            for segmentation_key in (
                "mesh_segmentation",
                "mesh_segmentation_u8",
                "mesh_segmentation_u16",
                "segmentation",
                "segmentation_u8",
            ):
                if segmentation_key in data:
                    # Keep labels in their compact stored dtype while they are
                    # in DataLoader workers. The model casts to torch.long only
                    # when computing cross entropy.
                    mesh_segmentation = np.asarray(data[segmentation_key][indices])
                    break
            view_names = [str(v) for v in data["view_names"].tolist()]
            depth_unit_arr = data["depth_unit"]
            depth_unit = (
                str(depth_unit_arr.item())
                if depth_unit_arr.size == 1
                else str(depth_unit_arr[0])
            )

        if depth_unit == "mm":
            scale = 1.0 / 1000.0
        elif depth_unit == "m":
            scale = 1.0
        else:
            raise ValueError(f"Unsupported depth_unit: {depth_unit!r}")
        depth_m = depth_u16.astype(np.float32) * scale
        depth_m[depth_u16 == 0] = np.nan

        return {
            "K": intrinsics,
            "cam2world": cam2world,
            "depth_m": depth_m,
            "mesh_segmentation": mesh_segmentation,
            "view_names": view_names,
        }

    @staticmethod
    def _parse_camera_meta_json(json_path: Path, indices: np.ndarray) -> dict[str, Any]:
        """Static-calibration fallback: meta/calibration.json broadcast to per-frame.

        Schema: ``{view_name: {"K": [3,3], "cam2world": [4,4], "static": bool}}``.
        depth is not provided by this fallback; ``depth_m`` is ``None``.
        """
        with open(json_path) as f:
            static_calib = json.load(f)
        view_names = list(static_calib.keys())
        V = len(view_names)
        N = int(np.asarray(indices).shape[0])
        K = np.stack([np.asarray(static_calib[v]["K"], dtype=np.float32) for v in view_names])
        cam2world = np.stack(
            [np.asarray(static_calib[v]["cam2world"], dtype=np.float32) for v in view_names]
        )
        return {
            "K": np.broadcast_to(K[None], (N, V, 3, 3)).copy(),
            "cam2world": np.broadcast_to(cam2world[None], (N, V, 4, 4)).copy(),
            "depth_m": None,
            "mesh_segmentation": None,
            "view_names": view_names,
        }

    def _validate_calibration_view_names(self, view_names: list[str]) -> None:
        """Every configured video key must appear in calibration view_names."""
        if "video" not in self.modality_configs:
            return
        keys = self.modality_configs["video"].modality_keys
        missing = [k for k in keys if k not in view_names]
        if missing:
            raise ValueError(
                f"Configured video keys {missing} not present in calibration "
                f"view_names {view_names}; calibration must cover every kept "
                f"video key."
            )

    def get_dataset_statistics(self) -> dict[str, Any]:
        """
        Extract dataset statistics for normalization from loaded metadata.

        Constructs a nested dictionary containing statistics (mean, std, min, max, q01, q99)
        for each joint group in state and action modalities. These statistics are used
        by processors for data normalization during training.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}
        """
        mapping = {"state": "observation.state", "action": "action"}
        dataset_statistics = _rec_defaultdict()

        for modality in mapping.keys():  # state, action
            for joint_key in self.modality_configs[modality].modality_keys:
                # Determine which statistics key to use
                if self.modality_meta[modality][joint_key].get("original_key", None) is not None:
                    stats_key = self.modality_meta[modality][joint_key]["original_key"]
                else:
                    stats_key = mapping[modality]

                # Extract the relevant slice of statistics
                start_idx, end_idx = (
                    self.modality_meta[modality][joint_key]["start"],
                    self.modality_meta[modality][joint_key]["end"],
                )
                for stat_type in self.stats[stats_key].keys():  # mean, std, min, max, q01, q99
                    dataset_statistics[modality][joint_key][stat_type] = self.stats[stats_key][
                        stat_type
                    ][start_idx:end_idx]
        stats = _to_plain_dict(dataset_statistics)
        # Directly add relative action stats
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def create_language_from_meta(
        self, episode_meta: dict, nframes: int, lang_key: str
    ) -> list[str]:
        if lang_key == "task":
            meta_language = random.choice(episode_meta["tasks"])
            new_languages = [meta_language] * nframes
        elif lang_key == "sub_task":
            action_delta_indices = self.modality_configs["action"].delta_indices
            action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            new_languages = [[] for _ in range(nframes)]
            sub_tasks = episode_meta["sub_tasks"]
            for sub_task in sub_tasks:
                start_idx, end_idx, sub_text = (
                    sub_task["start"],
                    sub_task["end"],
                    sub_task["text"],
                )
                horizon = action_horizon // 2
                for i in range(start_idx - horizon, end_idx):
                    if i < 0:
                        continue
                    new_languages[i].append(sub_text)
            new_languages = [i if len(i) > 0 else [""] for i in new_languages]
            new_languages = [random.choice(i) for i in new_languages]
        else:
            raise ValueError(f"Language key {lang_key} not supported")
        return new_languages

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """
        Load complete episode data as a processed DataFrame.

        Combines parquet data loading and video decoding to create a unified DataFrame
        containing all modality data for the episode. Video frames are converted to
        PIL Images and stored in the DataFrame.

        Args:
            idx: Episode index to load

        Returns:
            DataFrame with columns for all modalities and timestamps, with video frames
            as PIL Images ready for further processing

        Raises:
            IndexError: If episode index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        # Load and parse the parquet data
        df = self._load_parquet_data(episode_id)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(episode_meta, len(df), lang_key)
                df["language." + lang_key] = new_languages

        # Use actual dataframe length (might be less than nominal)
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        # Load synchronized video data
        video_data = self._load_video_data(episode_id, np.arange(actual_length))

        # Add video frames to dataframe as PIL Images
        for key in video_data.keys():
            assert len(video_data[key]) == len(df), (
                f"Video data for {key} has length {len(video_data[key])} but dataframe has length {len(df)}"
            )
            df[f"video.{key}"] = [frame for frame in video_data[key]]

        # Load synchronized mask data
        mask_data = self._load_mask_data(episode_id, np.arange(actual_length))
        for key in mask_data.keys():
            assert len(mask_data[key]) == len(df), (
                f"Mask data for {key} has length {len(mask_data[key])} but dataframe has length {len(df)}"
            )
            df[f"mask.{key}"] = [mask for mask in mask_data[key]]

        return df

    def get_initial_actions(self):
        """
        Load initial actions for policy initialization if available.

        Returns:
            List containing initial action dictionaries, or empty list if not available
        """
        meta_dirpath = self.dataset_path / LEROBOT_META_DIR_NAME
        initial_actions_path = meta_dirpath / INITIAL_ACTIONS_FILENAME
        if initial_actions_path.exists():
            initial_actions = load_initial_actions(initial_actions_path)
            return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        else:
            return []
