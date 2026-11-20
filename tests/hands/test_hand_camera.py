"""Tests for `fastdo.hands.hand_camera`: HaWoR outputs moved into the projection camera
by a depth-preserving principal-point shear, and the hard error on an uncorrected principal-point mismatch.

The slot tests run the real `fastdo.hands.hawor_core.run_heads` (right slots and flipped left slots, H4) with the seeded
`FakeHawor` of `test_hawor_core`, whose `get_trans` is HaWoR's full-frame translation; only the predicted weak-perspective
camera is fixed, so that each slot's translation is a known point seen at a known pixel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from fastdo.hands.hand_camera import (
    CAMERA_KEY,
    HAWOR_CAMERA_KEY,
    PP_TOLERANCE_PX,
    hawor_camera,
    principal_point_shear,
    require_hands_camera,
)
from fastdo.hands.hawor_core import WINDOW, run_heads
from fastdo.hands.hawor_plan import ClipGeometry, Slot
from tests.hands.test_hawor_core import OPS, FakeHawor

W, H = 640, 480
# DexYCB serial 836212060125 (DEV): principal point (308.55, 248.52), so HaWoR's (320, 240) is (+11.45, -8.52) px away.
DEV_K = np.array([[616.640869140625, 0.0, 308.548095703125], [0.0, 616.2581787109375, 248.52310180664062],
                  [0.0, 0.0, 1.0]])


def _project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return np.stack([K[0, 0] * p[..., 0] / p[..., 2] + K[0, 2], K[1, 1] * p[..., 1] / p[..., 2] + K[1, 2]], axis=-1)


def _with_principal_point(K: np.ndarray, cx: float, cy: float) -> np.ndarray:
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, 2], out[1, 2] = cx, cy
    return out


def _points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-0.3, 0.3, n), rng.uniform(-0.2, 0.2, n), rng.uniform(0.25, 2.0, n)], axis=-1)


def test_hawor_camera_is_the_image_centre_with_the_single_focal_length() -> None:
    np.testing.assert_array_equal(hawor_camera(DEV_K, W, H),
                                  [[DEV_K[0, 0], 0.0, 320.0], [0.0, DEV_K[0, 0], 240.0], [0.0, 0.0, 1.0]])
    assert hawor_camera(DEV_K.astype(np.float32), 641, 479)[:2, 2].tolist() == [320.5, 239.5]


def test_shear_reprojection_identity_and_unchanged_depth() -> None:
    points = _points(4000, seed=52).reshape(40, 100, 3)
    sheared = principal_point_shear(points, (W / 2.0, H / 2.0), DEV_K)

    hawor_pixels = _project(points, _with_principal_point(DEV_K, W / 2.0, H / 2.0))
    assert sheared.shape == points.shape and sheared.dtype == np.float64
    assert float(np.abs(_project(sheared, DEV_K) - hawor_pixels).max()) < 1e-6          # the same pixel, through K
    np.testing.assert_array_equal(sheared[..., 2], points[..., 2])                       # depth is untouched
    # The evaluated form (M4): X' = X + Z (cx_H - cx_K) / fx and Y' = Y + Z (cy_H - cy_K) / fy.
    np.testing.assert_allclose(sheared[..., 0], points[..., 0] + points[..., 2] * (320.0 - DEV_K[0, 2]) / DEV_K[0, 0],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(sheared[..., 1], points[..., 1] + points[..., 2] * (240.0 - DEV_K[1, 2]) / DEV_K[1, 1],
                               rtol=0, atol=1e-15)
    # Without the shear, every point lands exactly (cx_K - W/2, cy_K - H/2) = (-11.45, +8.52) px off.
    offset = _project(points, DEV_K) - hawor_pixels
    np.testing.assert_allclose(offset[..., 0], DEV_K[0, 2] - 320.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(offset[..., 1], DEV_K[1, 2] - 240.0, rtol=0, atol=1e-9)


def test_shear_keeps_float32_and_non_finite_rows() -> None:
    points = _points(30, seed=1).astype(np.float32).reshape(10, 3, 3)
    points[4] = np.nan
    sheared = principal_point_shear(points, (320.0, 240.0), DEV_K)
    assert sheared.dtype == np.float32 and bool(np.isnan(sheared[4]).all())
    np.testing.assert_array_equal(sheared[..., 2], points[..., 2])
    finite = np.isfinite(points).all(axis=-1)
    error = _project(sheared, DEV_K) - _project(points, _with_principal_point(DEV_K, 320.0, 240.0))
    assert float(np.abs(error[finite]).max()) < 1e-3                                     # float32 storage


def test_shear_is_a_bitwise_no_op_when_the_principal_points_match() -> None:
    moge2_like = np.array([[612.0, 0.0, 320.0], [0.0, 598.5, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)   # fx != fy
    points = _points(60, seed=3).astype(np.float32).reshape(20, 3, 3)
    sheared = principal_point_shear(points, (W / 2.0, H / 2.0), moge2_like)
    assert sheared.dtype == np.float32 and sheared is not points
    np.testing.assert_array_equal(sheared, points)


class _FixedCam(nn.Module):
    """`FakeHawor.mano_head` with its predicted camera `(s, tx, ty)` replaced by `cam` (one row per slot, in order)."""

    def __init__(self, inner: nn.Module, cam: Tensor) -> None:
        super().__init__()
        self.inner, self.cam = inner, cam

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pose, betas, _ = self.inner(x)
        return pose, betas, self.cam


@pytest.mark.parametrize("side", ["right", "left"])
def test_right_and_flipped_left_slots_reach_the_true_point_after_the_shear(side: str) -> None:
    """A point P (projection-camera frame) seen at its pixel (u, v) through K. HaWoR's network, which sees the mirrored
    image for a left slot (column W - 1 - u, plan centre W - cx - 1), predicts the weak-perspective camera of that pixel
    at P's depth; `run_heads` un-mirrors it and returns the translation in HaWoR's camera. The shear must return P."""
    flip = side == "left"
    K = np.array(DEV_K, copy=True)
    K[1, 1] = K[0, 0]                    # HaWoR has one focal length; with fy = fx the round trip is exact
    f = float(K[0, 0])
    rng = np.random.default_rng(7 if flip else 5)
    truth, slots, cams = [], [], []
    for q in range(WINDOW):
        point = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15), rng.uniform(0.35, 1.2)])
        u, v = _project(point, K)
        box_cx, box_cy, scale = u + rng.uniform(-15, 15), v + rng.uniform(-15, 15), rng.uniform(0.4, 1.2)
        center = (W - box_cx - 1.0 if flip else box_cx, box_cy)                          # hawor_plan.plan_clip
        bs = 200.0 * scale * (2.0 * f / (200.0 * scale * point[2]))
        u_seen = W - 1.0 - u if flip else u
        cams.append([2.0 * f / (200.0 * scale * point[2]), (u_seen - center[0]) * 2.0 / bs, (v - center[1]) * 2.0 / bs])
        slots.append(Slot(0, side, q, q, 0.0, center, scale, flip, q))
        truth.append(point)
    model = FakeHawor()
    setattr(model, "mano_head", _FixedCam(model.mano_head, torch.tensor(cams, dtype=torch.float32)))  # noqa: B010
    bank = torch.randn(WINDOW, 1280, 16, 12, generator=torch.Generator().manual_seed(11))

    trans = run_heads(model, OPS, bank, slots, [ClipGeometry(W, H, f)]).trans.numpy()
    points = np.stack(truth)

    hawor_pixels = _project(trans, hawor_camera(K, W, H))
    assert float(np.abs(hawor_pixels - _project(points, K)).max()) < 1e-3                # HaWoR's own camera: exact
    unsheared = _project(trans, K) - _project(points, K)
    np.testing.assert_allclose(unsheared, np.broadcast_to([K[0, 2] - 320.0, K[1, 2] - 240.0], unsheared.shape),
                               rtol=0, atol=1e-3)                                          # (-11.45, +8.52) px, both sides
    sheared = principal_point_shear(trans.astype(np.float64), (W / 2.0, H / 2.0), K)
    assert float(np.abs(_project(sheared, K) - _project(points, K)).max()) < 1e-3        # un-mirroring error: 1 px
    np.testing.assert_allclose(sheared, points, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(sheared[:, 2], trans[:, 2].astype(np.float64))


def _hands(**camera: np.ndarray) -> dict[str, np.ndarray]:
    return {"right_joints": np.zeros((2, 21, 3), np.float32), **camera}


def test_require_hands_camera_rejects_uncorrected_hawor_outputs_under_an_offset_principal_point() -> None:
    with pytest.raises(ValueError, match=r"principal point.*\(320(\.0*)?, 240(\.0*)?\).*-11\.45.*\+?8\.52"):
        require_hands_camera(_hands(), DEV_K, W, H, "hands.npz")
    require_hands_camera(_hands(), _with_principal_point(DEV_K, 320.4, 239.3), W, H, "hands.npz")   # within 1 px
    assert PP_TOLERANCE_PX == 1.0


def test_require_hands_camera_accepts_moge2_intrinsics_without_a_record() -> None:
    require_hands_camera(_hands(), np.array([[590.0, 0, 320.0], [0, 588.0, 240.0], [0, 0, 1]]), W, H, "hands.npz")


def test_require_hands_camera_checks_the_recorded_output_camera() -> None:
    recorded = _hands(**{CAMERA_KEY: DEV_K, HAWOR_CAMERA_KEY: hawor_camera(DEV_K, W, H)})
    require_hands_camera(recorded, DEV_K.astype(np.float32), W, H, "hands.npz")
    with pytest.raises(ValueError, match="camera"):
        require_hands_camera(recorded, _with_principal_point(DEV_K, 320.0, 240.0), W, H, "hands.npz")
    other_focal = np.array(DEV_K, copy=True)
    other_focal[0, 0] += 5.0
    with pytest.raises(ValueError, match="camera"):
        require_hands_camera(recorded, other_focal, W, H, "hands.npz")
