"""Known-answer and mutation tests for fastdo.bench.v2d_orientation (V2D ego-device orientation hypotheses)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.bench import v2d_track3 as v2d
from fastdo.bench.v2d_orientation import (
    RZ_PI,
    conj_180,
    edge_ncc,
    image_up_cosine,
    rotate_calib_180,
    rotate_pixels_180,
    sobel_magnitude,
)

# ego_cam_a-like numbers with every odd term non-zero (tangential, thin prism, tilt)
_DIST = np.array([4.69, 1.31, 0.0012, -0.0009, 0.015, 5.06, 2.73, 0.159, 0.0011, -0.0004, 0.0007, 0.0003,
                  0.0189, 0.0012])


def _cam(dist: np.ndarray = _DIST) -> v2d.CameraCalib:
    k = np.array([[1150.0, 0.0, 994.8], [0.0, 1149.0, 767.5], [0.0, 0.0, 1.0]])
    return v2d.CameraCalib(name="A", resolution=(2028, 1520), K=k, distortion_model="Perspective",
                           distortion=dist.copy(), hfov_deg=108.0, rotation_deg=180,
                           extrinsics_to_stereo_left=None)


def _points(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.3, 1.5, n)
    xy = rng.uniform(-0.6, 0.6, (n, 2)) * z[:, None]
    return np.column_stack([xy, z])


def _project(x: np.ndarray, cam: v2d.CameraCalib) -> np.ndarray:
    import cv2

    uv, _ = cv2.projectPoints(x.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.distortion)
    return uv.reshape(-1, 2)


def test_rotate_calib_180_matches_the_rotated_image() -> None:
    """Projecting Rz(pi) X with the rotated calibration lands on the 180-deg-rotated pixel of X (14 terms)."""
    cam = _cam()
    x = _points()
    uv = _project(x, cam)
    uv_rot = _project(x @ RZ_PI[:3, :3].T, rotate_calib_180(cam))
    assert np.max(np.abs(uv_rot - rotate_pixels_180(uv, cam.resolution))) < 1e-6


def test_rotate_calib_180_is_an_involution() -> None:
    cam = _cam()
    back = rotate_calib_180(rotate_calib_180(cam))
    np.testing.assert_allclose(back.K, cam.K, atol=1e-12)
    np.testing.assert_array_equal(back.distortion, cam.distortion)
    assert back.rotation_deg == cam.rotation_deg


@pytest.mark.parametrize("idx", [2, 3, 8, 12])
def test_keeping_an_odd_term_unflipped_is_detected(idx: int) -> None:
    """Mutation: forgetting to negate any one odd term (p1, p2, s1, tau_x) moves pixels measurably."""
    cam = _cam()
    x = _points()
    good = rotate_calib_180(cam)
    d = good.distortion.copy()
    d[idx] *= -1.0
    bad = dataclasses.replace(good, distortion=d)
    ref = rotate_pixels_180(_project(x, cam), cam.resolution)
    assert np.max(np.abs(_project(x @ RZ_PI[:3, :3].T, bad) - ref)) > 0.05


def test_principal_point_without_flip_is_detected() -> None:
    """Mutation: rotating the distortion but keeping cx, cy moves pixels by ~2|cx - (W-1)/2| (here ~37 px)."""
    cam = _cam()
    good = rotate_calib_180(cam)
    bad = dataclasses.replace(good, K=cam.K.copy())
    x = _points()
    ref = rotate_pixels_180(_project(x, cam), cam.resolution)
    assert np.median(np.abs(_project(x @ RZ_PI[:3, :3].T, bad) - ref)[:, 0]) > 30.0


def test_conj_180_extrinsic_projects_consistently() -> None:
    """A point seen by camera c and projected into camera a: the rotated frames give the rotated pixel."""
    a = _cam(np.zeros(14))
    a_T_c = np.eye(4)
    a_T_c[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.3, 0.2], degrees=True).as_matrix()
    a_T_c[:3, 3] = [-0.0374, 0.0002, -0.0002]
    x_c = _points()
    uv = _project(x_c @ a_T_c[:3, :3].T + a_T_c[:3, 3], a)
    t2 = conj_180(a_T_c)
    x_c2 = x_c @ RZ_PI[:3, :3].T
    uv2 = _project(x_c2 @ t2[:3, :3].T + t2[:3, 3], rotate_calib_180(a))
    assert np.max(np.abs(uv2 - rotate_pixels_180(uv, a.resolution))) < 1e-6
    assert t2[0, 3] == pytest.approx(0.0374) and t2[2, 3] == pytest.approx(-0.0002)


def _head_camera(roll_deg: float) -> np.ndarray:
    """World z up; camera looking down ~50 deg, upright at roll 0 (Rx(-140) alone is upside down: its camera +y has
    a +z world component); rolled about its optical axis by roll_deg."""
    look = Rotation.from_euler("x", -140, degrees=True).as_matrix()
    roll = Rotation.from_euler("z", 180.0 + roll_deg, degrees=True).as_matrix()
    t = np.eye(4)
    t[:3, :3] = roll @ look
    t[:3, 3] = -t[:3, :3] @ np.array([0.0, -0.45, 1.25])
    return t


def test_image_up_cosine_separates_upright_from_upside_down() -> None:
    upright = image_up_cosine(_head_camera(0.0))
    tilted = image_up_cosine(_head_camera(25.0))
    flipped = image_up_cosine(RZ_PI @ _head_camera(0.0))
    assert upright > 0.5 and tilted > 0.5 and flipped < -0.5
    assert image_up_cosine(np.stack([_head_camera(0.0)] * 3)).shape == (3,)


def test_edge_ncc_prefers_the_right_hypothesis() -> None:
    """Values sampled at the correct projections correlate ~1; the 180-deg-wrong hypothesis does not."""
    import cv2

    rng = np.random.default_rng(4)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (400, 640)).astype(np.float32), (0, 0), 3.0)
    edges = sobel_magnitude(img)
    uv = np.column_stack([rng.uniform(5, 634, 3000), rng.uniform(5, 394, 3000)])
    vals = np.asarray([edges[round(v), round(u)] for u, v in uv])
    right = edge_ncc(vals, edges, np.round(uv))
    wrong = edge_ncc(vals, edges, rotate_pixels_180(np.round(uv), (640, 400)))
    assert right > 0.99 and abs(wrong) < 0.1
    assert np.isnan(edge_ncc(vals[:10], edges, uv[:10]))
