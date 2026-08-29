from __future__ import annotations

import numpy as np
import pytest

from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import (
    _reorder_camera_views_to_image_keys,
)


def test_reorders_camera_views_to_match_image_keys() -> None:
    camera_values = np.asarray(
        [
            np.full((2, 2), 10, dtype=np.float32),
            np.full((2, 2), 20, dtype=np.float32),
            np.full((2, 2), 30, dtype=np.float32),
        ]
    )
    camera_calibration = {
        "view_names": ["wrist_image", "image", "unused_view"],
    }

    reordered = _reorder_camera_views_to_image_keys(
        camera_values,
        image_keys=["image", "wrist_image"],
        camera_calibration=camera_calibration,
        value_name="depth_m",
    )

    assert reordered.shape == (2, 2, 2)
    assert np.array_equal(reordered[0], camera_values[1])
    assert np.array_equal(reordered[1], camera_values[0])


def test_reorder_keeps_legacy_order_when_view_names_are_absent() -> None:
    camera_values = np.asarray(
        [
            np.full((2, 2), 10, dtype=np.float32),
            np.full((2, 2), 20, dtype=np.float32),
        ]
    )

    reordered = _reorder_camera_views_to_image_keys(
        camera_values,
        image_keys=["image", "wrist_image"],
        camera_calibration={},
        value_name="depth_m",
    )

    assert np.array_equal(reordered, camera_values)


def test_reorder_rejects_missing_requested_view() -> None:
    camera_values = np.asarray(
        [
            np.full((2, 2), 10, dtype=np.float32),
            np.full((2, 2), 20, dtype=np.float32),
        ]
    )

    with pytest.raises(KeyError, match="view_names does not cover"):
        _reorder_camera_views_to_image_keys(
            camera_values,
            image_keys=["image", "wrist_image"],
            camera_calibration={"view_names": ["image", "side_image"]},
            value_name="depth_m",
        )
