"""Tests for `fastdo.robot.symmetry`: which grasp rotations an object allows, and the rotated grasp."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.robot.symmetry import grasp_rotation, rotate_grasp, rotational_symmetries


def _axes(mesh: trimesh.Trimesh, scale: float = 1.0):
    return rotational_symmetries(mesh.vertices, mesh.faces, scale, step_deg=30, n_samples=8000)


def test_a_cylinder_allows_every_angle_about_its_axis_only() -> None:
    """Known answer: a 1.5 cm radius, 18 cm cylinder (a screwdriver handle) maps onto itself at every angle about its
    long axis; a 90 deg turn about a cross axis moves its ends by ~6 cm, so none is allowed there."""
    cyl = trimesh.creation.cylinder(radius=0.015, height=0.18, sections=96)
    axes = _axes(cyl)
    long = axes[2]
    assert abs(abs(long.axis[2]) - 1.0) < 1e-3 and long.extent_m == pytest.approx(0.18, abs=0.005)
    assert long.allowed_deg == (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)
    assert max(long.mismatch_m.values()) < 0.002          # the 8,000-sample floor
    for cross in axes[:2]:
        assert 90 not in cross.allowed_deg and cross.mismatch_m[90] > 0.02


def test_a_box_allows_only_half_turns_and_an_asymmetric_object_nothing() -> None:
    box = trimesh.creation.box(extents=[0.10, 0.06, 0.04])
    for ax in _axes(box):
        assert ax.allowed_deg == (0, 180), ax.allowed_deg
    lump = trimesh.util.concatenate([box, trimesh.creation.box(extents=[0.03, 0.03, 0.03],
                                                               transform=trimesh.transformations.translation_matrix(
                                                                   [0.04, 0.02, 0.03]))])
    assert all(ax.allowed_deg == (0,) for ax in _axes(lump))


def test_tolerance_is_the_registered_three_millimetres() -> None:
    """The 90 deg turn about the long axis of a 6 x 5.9 cm cross-section box is allowed (mismatch 2.7 mm at 8,000
    samples), of a 6 x 5 cm one refused (6.0 mm)."""
    near = trimesh.creation.box(extents=[0.20, 0.060, 0.059])
    far = trimesh.creation.box(extents=[0.20, 0.060, 0.050])
    assert 90 in _axes(near)[2].allowed_deg
    assert 90 not in _axes(far)[2].allowed_deg


def test_rotated_grasp_keeps_the_hand_on_the_same_surface() -> None:
    """The wrist moves rigidly with the object frame rotated about the axis: its distance to the axis and its height
    along it are unchanged, the object poses are untouched, and angle 0 is the identity."""
    rng = np.random.default_rng(0)
    n = 6
    obj = np.tile(np.eye(4), (n, 1, 1))
    obj[:, :3, :3] = Rotation.random(n, random_state=1).as_matrix()
    obj[:, :3, 3] = rng.normal(0, 0.2, (n, 3))
    wrist = np.tile(np.eye(4), (n, 1, 1))
    wrist[:, :3, :3] = Rotation.random(n, random_state=2).as_matrix()
    wrist[:, :3, 3] = obj[:, :3, 3] + rng.normal(0, 0.05, (n, 3))
    axis, centre = np.array([0.0, 0.0, 1.0]), np.array([0.01, -0.02, 0.0])
    rot = grasp_rotation(axis, centre, np.radians(70))
    out = rotate_grasp(wrist, obj, rot)
    rel_a = np.linalg.solve(obj, wrist)[:, :3, 3] - centre
    rel_b = np.linalg.solve(obj, out)[:, :3, 3] - centre
    assert np.allclose(np.linalg.norm(rel_a[:, :2], axis=1), np.linalg.norm(rel_b[:, :2], axis=1))
    assert np.allclose(rel_a[:, 2], rel_b[:, 2])
    ang = np.arctan2(rel_b[:, 1], rel_b[:, 0]) - np.arctan2(rel_a[:, 1], rel_a[:, 0])
    assert np.allclose(np.mod(ang, 2 * np.pi), np.radians(70))
    assert np.allclose(rotate_grasp(wrist, obj, grasp_rotation(axis, centre, 0.0)), wrist)
