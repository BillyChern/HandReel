"""Tests for `fastdo.robot.placement`: gravity alignment, egocentric heading -> robot +x, object on the table top."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.robot import placement as P
from fastdo.robot.placement import (
    camera_heading,
    gravity_rotation,
    place_on_table,
    transform_points,
)

_CUBE = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)


def _camera(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """cam_to_world with optical axis `forward` and image-up `up` (OpenCV: +y_cam points down)."""
    z = forward / np.linalg.norm(forward)
    y = -(up - up @ z * z)
    y /= np.linalg.norm(y)
    out = np.eye(4)
    out[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    return out


@pytest.mark.parametrize("up", [[0, 0, 1.0], [0, -1.0, 0], [0.3, 0.2, -0.9], [0, 0, -1.0]])
def test_gravity_rotation_maps_up_to_z(up: list[float]) -> None:
    rot = gravity_rotation(np.asarray(up))
    assert np.allclose(rot @ np.asarray(up) / np.linalg.norm(up), [0, 0, 1], atol=1e-12)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12) and np.linalg.det(rot) == pytest.approx(1.0)


def test_object_lands_on_the_table_under_the_view_direction() -> None:
    """Known answer: world up = -y_w (OpenCV-like world), camera looking along +z_w, tilted 30 deg down. The heading
    must come out as robot +x, the object centre at table_xy and its lowest point on the table top."""
    up_w = np.array([0.0, -1.0, 0.0])
    fwd = np.array([0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])     # 30 deg below the horizon
    c2w = _camera(fwd, up_w)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    pose[:3, 3] = [0.1, 0.3, 0.6]
    pl = place_on_table(pose, _CUBE, 0.05, up_w, c2w, (0.35, -0.2), 0.8, clearance=0.0)
    rot = pl.T_rv[:3, :3]
    assert np.allclose(rot @ up_w, [0, 0, 1], atol=1e-12)
    heading = rot @ fwd
    assert np.allclose(heading[:2] / np.linalg.norm(heading[:2]), [1, 0], atol=1e-12)
    pts = transform_points(pl.T_rv, (0.05 * _CUBE) @ pose[:3, :3].T + pose[:3, 3])
    assert pts[:, 2].min() == pytest.approx(0.8, abs=1e-12)
    assert np.allclose(pl.object_start[:2, 3], [0.35, -0.2], atol=1e-12)
    assert np.allclose(pl.object_start, pl.T_rv @ pose, atol=1e-12)


def test_heading_sign_mutation_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: a negated heading sends the human's view direction to robot -x."""
    real = P.camera_heading
    monkeypatch.setattr(P, "camera_heading", lambda c2w, r, m=0.2: -real(c2w, r, m))
    up_w = np.array([0.0, 0.0, 1.0])
    fwd = np.array([1.0, 1.0, -0.3])
    pl = P.place_on_table(np.eye(4), _CUBE, 0.05, up_w, _camera(fwd, up_w), (0.3, 0.0), 0.8)
    heading = pl.T_rv[:3, :3] @ fwd
    assert heading[0] < 0


def test_looking_down_falls_back_to_image_up() -> None:
    up_w = np.array([0.0, 0.0, 1.0])
    c2w = _camera(np.array([0.0, 0.01, -1.0]), np.array([0.0, 1.0, 0.0]))     # straight down, image-up = +y_w
    h = camera_heading(c2w, gravity_rotation(up_w))
    assert np.allclose(h, [0, 1], atol=1e-6)


def test_yaw_offset_rotates_about_z() -> None:
    up_w = np.array([0.0, 0.0, 1.0])
    c2w = _camera(np.array([1.0, 0.0, 0.0]), up_w)
    a = place_on_table(np.eye(4), _CUBE, 0.05, up_w, c2w, (0.3, 0.0), 0.8)
    b = place_on_table(np.eye(4), _CUBE, 0.05, up_w, c2w, (0.3, 0.0), 0.8, yaw_offset=np.pi / 2)
    assert b.yaw - a.yaw == pytest.approx(np.pi / 2)
    assert np.allclose(b.T_rv[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    with pytest.raises(ValueError):
        gravity_rotation(np.zeros(3))


@pytest.mark.parametrize("pitch", [0.0, 25.0, 40.0, 70.0])
def test_pitched_up_recovers_world_up(pitch: float) -> None:
    """Known answer: a camera pitched down by `pitch` in a z-up world; the assumed pitch gives back +z exactly, the
    opposite sign (mutation) is off by twice the pitch."""
    th = np.radians(pitch)
    cam = _camera(np.array([np.cos(th), 0.0, -np.sin(th)]), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(P.pitched_up(cam, pitch), [0.0, 0.0, 1.0], atol=1e-12)
    wrong = P.pitched_up(cam, -pitch)
    assert np.degrees(np.arccos(np.clip(wrong[2], -1.0, 1.0))) == pytest.approx(2 * pitch, abs=1e-6)


def test_footprint_must_stay_on_the_table() -> None:
    """A 6 cm cube on a table whose near edge is x = 0.18: centred 3 cm in it touches the edge (refused with the 2 cm
    margin, accepted without), centred at 0.30 it is on the table; off the far side and off the y range it is not."""
    pose = np.eye(4)
    table = ((0.18, 0.78), (-0.60, 0.40))
    for x, margin, inside in ((0.21, 0.02, False), (0.211, 0.0, True), (0.30, 0.02, True), (0.77, 0.02, False)):
        pose[:3, 3] = [x, 0.0, 0.83]
        assert P.footprint_inside(pose, _CUBE, 0.03, *table, margin=margin) is inside, (x, margin)
    pose[:3, 3] = [0.30, 0.39, 0.83]
    assert not P.footprint_inside(pose, _CUBE, 0.03, *table)
