"""Tests for `fastdo.bench.headtohead`: SE(3) trajectory interpolation, pose errors, silhouettes, IoU."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fastdo.bench.headtohead import (
    axis_tilt_errors,
    interpolate_poses,
    mask_iou,
    principal_axis,
    render_silhouette,
    rotation_errors,
    translation_errors,
)
from fastdo.geometry.lie import se3_exp


def _pose(xi: list[float]) -> torch.Tensor:
    return se3_exp(torch.tensor(xi, dtype=torch.float64)).float()


def test_interpolate_poses_hits_solved_frames_and_holds_the_ends() -> None:
    solved = torch.stack([_pose([0, 0, 1, 0, 0, 0]), _pose([0.3, 0, 1, 0, 0, 0.6]), _pose([0.3, 0.3, 1, 0.2, 0, 0.6])])
    full = interpolate_poses(solved, [2, 5, 8], n_frames=11)
    assert full.shape == (11, 4, 4)
    for i, f in enumerate([2, 5, 8]):
        assert torch.allclose(full[f], solved[i], atol=1e-5)
    assert torch.allclose(full[0], solved[0]) and torch.allclose(full[1], solved[0])   # before first: hold
    assert torch.allclose(full[9], solved[2]) and torch.allclose(full[10], solved[2])  # after last: hold


def test_interpolate_poses_is_geodesic_between_solved_frames() -> None:
    a = _pose([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    b = a @ _pose([0.3, 0.0, 0.0, 0.0, 0.0, 0.9])   # 0.9 rad about z, with a screw translation
    full = interpolate_poses(torch.stack([a, b]), [0, 3], n_frames=4)
    third = a @ _pose([0.1, 0.0, 0.0, 0.0, 0.0, 0.3])
    assert torch.allclose(full[1], third, atol=1e-5)


def test_translation_and_rotation_errors() -> None:
    a = torch.stack([_pose([0, 0, 1, 0, 0, 0]), _pose([0, 0, 1, 0, 0, 0])])
    b = torch.stack([_pose([0.03, 0, 1, 0, 0, 0]), _pose([0, 0, 0, 0.2, 0, 0])])
    b[1, :3, 3] = torch.tensor([0.0, 0.0, 1.0])       # pure 0.2 rad rotation, same translation
    assert torch.allclose(translation_errors(a, b), torch.tensor([0.03, 0.0]), atol=1e-6)
    rot = rotation_errors(a, b)
    assert float(rot[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(rot[1]) == pytest.approx(0.2, abs=1e-5)


def test_render_silhouette_fills_a_projected_triangle() -> None:
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    verts = np.array([[-0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [-0.1, 0.1, 1.0], [0.0, 0.0, -1.0]], np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3]], np.int64)       # second face crosses behind the camera: skipped
    sil = render_silhouette(verts, faces, K, (80, 100))
    assert sil.dtype == bool and sil.shape == (80, 100)
    assert sil[35, 45] and not sil[48, 58] and not sil[0, 0]
    assert abs(int(sil.sum()) - 200) < 40                       # right triangle, 20 px legs


def test_mask_iou() -> None:
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2, :2] = True
    b[:2, 1:3] = True
    assert mask_iou(a, b) == pytest.approx(2 / 6)
    assert math.isnan(mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)))


def test_axis_tilt_ignores_spin_about_the_axis() -> None:
    pts = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0], [0.02, 0.0, 0.0], [-0.02, 0.0, 0.01]])
    axis = principal_axis(pts)
    assert abs(abs(float(axis[1])) - 1.0) < 1e-6
    base = torch.stack([_pose([0, 0, 1, 0, 0, 0])] * 2)
    other = torch.stack([_pose([0, 0, 0, 0, 1.2, 0]), _pose([0, 0, 0, 0, 0, math.pi / 2])])  # spin about y; 90 deg tilt
    tilt = axis_tilt_errors(base, other, axis)
    assert float(tilt[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(tilt[1]) == pytest.approx(math.pi / 2, abs=1e-5)
