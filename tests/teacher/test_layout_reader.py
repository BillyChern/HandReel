"""Tests for the baseline layout reader, against synthetic layout JSON fixtures.

Expected numbers below are computed independently of ``layout_reader.py`` (by hand, following the
same ``P`` change-of-basis documented in the module's docstring and verified against the real
whisking demo's ``layout_camera_frame_optimized.json``), not by re-deriving them from the module
under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastdo.teacher.layout_reader import read_baseline_layout, read_keyframe_layout


def _write(tmp_path: Path, data: dict, name: str = "layout.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _object(frame_idx: int, new_quat: list[float], translation: list[float],
           intrinsics_normalized: dict[str, float] | None = None, scale: float = 0.25) -> dict:
    obj = {
        "frame_idx": frame_idx,
        "local_to_scene": {"new_quat": new_quat, "translation": translation, "scale": [scale, scale, scale]},
    }
    if intrinsics_normalized is not None:
        obj["intrinsics_normalized"] = intrinsics_normalized
    return obj


_INTR = {"fx_norm": 0.5, "fy_norm": 0.5, "cx_norm": 0.5, "cy_norm": 0.5}


def test_pose_composition_matches_hand_derived_values(tmp_path: Path) -> None:
    # Identity rotation, translation (tx,ty,tz)=(1,2,3), per-frame translation_scale=2,
    # mesh_scale/mesh_scale_original=0.5/0.25=2 -> combined scale factor 4.
    data = {
        "objects": [_object(0, [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0], _INTR)],
        "translation_scale_optimization": {
            "mesh_scale": 0.5, "mesh_scale_original": 0.25,
            "per_frame": [{"frame_idx": 0, "translation_scale": 2.0}],
        },
    }
    path = _write(tmp_path, data)

    poses, mesh_scale, k_matrix = read_baseline_layout(path, image_width=100, image_height=50)

    assert poses.shape == (1, 4, 4) and poses.dtype.is_floating_point
    assert mesh_scale == pytest.approx(0.5)

    # t_pose = [tz,tx,ty] * 4 = [12,4,8]; t_cam = t_pose @ P with P=[[0,0,1],[-1,0,0],[0,-1,0]].
    expected_t = np.array([-4.0, -8.0, 12.0])
    np.testing.assert_allclose(poses[0, :3, 3].numpy(), expected_t, atol=1e-5)

    # R_pose = I -> R_cam = P.T.
    expected_r = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(poses[0, :3, :3].numpy(), expected_r, atol=1e-6)
    np.testing.assert_allclose(poses[0, 3, :].numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    # fx_norm=fy_norm=cx_norm=cy_norm=0.5, width=100, height=50.
    expected_k = np.array([[50.0, 0.0, 50.0], [0.0, 25.0, 25.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(k_matrix.numpy(), expected_k, atol=1e-5)


def test_frames_are_sorted_by_frame_idx(tmp_path: Path) -> None:
    data = {
        "objects": [
            _object(2, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0], _INTR),
            _object(0, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], _INTR),
            _object(1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0], _INTR),
        ],
    }
    path = _write(tmp_path, data)

    poses, _, _ = read_baseline_layout(path)

    # tz increases with frame_idx (0,1,2); tz maps to translation_camera_frame[..., 2] under P.
    assert poses.shape == (3, 4, 4)
    np.testing.assert_allclose(poses[:, 2, 3].numpy(), [0.0, 1.0, 2.0], atol=1e-5)


def test_defaults_without_translation_scale_optimization_block(tmp_path: Path) -> None:
    data = {"objects": [_object(0, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0], _INTR, scale=0.7)]}
    path = _write(tmp_path, data)

    poses, mesh_scale, _ = read_baseline_layout(path)

    assert mesh_scale == pytest.approx(0.7)  # falls back to local_to_scene.scale[0]
    # translation_scale defaults to 1, scale_ratio defaults to 1: t_pose = [1,0,0] @ P = [0,0,1].
    np.testing.assert_allclose(poses[0, :3, 3].numpy(), [0.0, 0.0, 1.0], atol=1e-5)


def test_median_k_used_when_some_frames_missing_intrinsics(tmp_path: Path) -> None:
    data = {
        "objects": [
            _object(0, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], {"fx_norm": 0.4, "fy_norm": 0.4,
                                                                 "cx_norm": 0.5, "cy_norm": 0.5}),
            _object(1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], None),  # missing (like frame 125 in the real demo)
            _object(2, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], {"fx_norm": 0.6, "fy_norm": 0.6,
                                                                 "cx_norm": 0.5, "cy_norm": 0.5}),
        ],
    }
    path = _write(tmp_path, data)

    _, _, k_matrix = read_baseline_layout(path, image_width=10, image_height=10)

    # median of fx in {4.0, 6.0} = 5.0
    assert k_matrix[0, 0].item() == pytest.approx(5.0)


def test_raises_on_empty_objects(tmp_path: Path) -> None:
    path = _write(tmp_path, {"objects": []})
    with pytest.raises(ValueError, match="empty"):
        read_baseline_layout(path)


def test_raises_when_no_frame_has_intrinsics(tmp_path: Path) -> None:
    data = {"objects": [_object(0, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], None)]}
    path = _write(tmp_path, data)
    with pytest.raises(ValueError, match="intrinsics_normalized"):
        read_baseline_layout(path)


def _optimized_layout() -> dict:
    return {
        "objects": [_object(4, [1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 1.5], _INTR),
                    _object(5, [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0], _INTR, scale=0.25),
                    _object(6, [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 9.0], _INTR)],
        "translation_scale_optimization": {
            "mesh_scale": 0.5, "mesh_scale_original": 0.25, "ref_frame": 5,
            "per_frame": [{"frame_idx": 5, "translation_scale": 2.0}],
        },
    }


def test_keyframe_layout_is_the_unscaled_reference_frame_pose(tmp_path: Path) -> None:
    path = _write(tmp_path, _optimized_layout())

    pose, scale = read_keyframe_layout(path)
    poses, mesh_scale, _ = read_baseline_layout(path)

    assert scale == pytest.approx(0.25)                      # mesh_scale_original, not the optimized 0.5
    # No translation_scale (2) and no mesh-scale ratio (2): t_pose = [3,1,2]; t_cam = t_pose @ P.
    np.testing.assert_allclose(pose[:3, 3].numpy(), [-1.0, -2.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(poses[1, :3, 3].numpy(), [-4.0, -8.0, 12.0], atol=1e-5)   # the optimized answer
    np.testing.assert_allclose(pose[:3, :3].numpy(), poses[1, :3, :3].numpy(), atol=1e-6)
    assert mesh_scale == pytest.approx(0.5)


def test_keyframe_layout_rejects_tracked_frames(tmp_path: Path) -> None:
    path = _write(tmp_path, _optimized_layout())
    assert read_keyframe_layout(path, frame_idx=5)[1] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="ref_frame 5"):
        read_keyframe_layout(path, frame_idx=6)

