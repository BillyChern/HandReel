"""Tests for `fastdo.retarget.rate`: engine key frames, C2 natural splines, the
sign-continuous quaternion spline, and the input upsamplers the full-rate kinematic reference is solved on
(object SE(3) by spline; wrist SE(3) by spline with the hand vertices lerped in the wrist frame). No real data.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from fastdo.retarget.rate import (
    engine_key_frames,
    natural_spline,
    quaternion_spline,
    upsample_hand_track,
    upsample_object_poses,
)


def _wxyz(rot: Rotation) -> np.ndarray:
    return rot.as_quat()[:, [3, 0, 1, 2]]


def _poses(rot: Rotation, trans: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (trans.shape[0], 1, 1))
    out[:, :3, :3] = rot.as_matrix()
    out[:, :3, 3] = trans
    return out


def test_engine_key_frames() -> None:
    keys = engine_key_frames(138, 3)
    assert len(keys) == 47 and keys.dtype == np.int64 and keys[-3:].tolist() == [132, 135, 137]
    k139 = engine_key_frames(139, 3)
    assert len(k139) == 47 and k139[-1] == 138
    k40 = engine_key_frames(40, 3)
    assert len(k40) == 14 and k40[-1] == 39
    assert np.array_equal(engine_key_frames(10, 1), np.arange(10))
    with pytest.raises(ValueError):
        engine_key_frames(10, 0)
    with pytest.raises(ValueError):
        engine_key_frames(1, 3)


def test_natural_spline_interpolates_keys_and_reproduces_affine_motion() -> None:
    keys = engine_key_frames(40, 3)
    values = np.random.default_rng(0).normal(size=(14, 5))
    out = natural_spline(values, keys, np.arange(40))
    assert out.shape == (40, 5)
    assert np.abs(out[keys] - values).max() <= 1e-12
    affine = 2.0 + 0.5 * keys.astype(np.float64)
    assert np.abs(natural_spline(affine, keys, np.arange(40)) - (2.0 + 0.5 * np.arange(40))).max() <= 1e-12
    with pytest.raises(ValueError):
        natural_spline(values[:3], np.array([0, 5, 5]), np.arange(6))
    with pytest.raises(ValueError):
        natural_spline(values[:1], np.array([0]), np.arange(6))


def test_natural_spline_has_zero_second_derivative_at_both_ends() -> None:
    """The rate composition is a C2 *natural* spline: zero second derivative at the first and last key. A not-a-knot
    spline also interpolates the keys and reproduces affine motion, so only this test tells the two apart. The keys
    end with a 1-frame interval and the values are random; each second derivative is the 4-point difference
    `(2 f0 - 5 f1 + 4 f2 - f3) / h^2` over samples inside one cubic piece, which is exact for a cubic. Both ends are
    ~0 relative to the curvature at the interior keys, and the output equals scipy's `CubicSpline(bc_type="natural")`."""
    keys = engine_key_frames(41, 3)                                           # 0, 3, ..., 39, 40
    values = np.random.default_rng(1).normal(size=(keys.shape[0], 4))

    def second_derivative(at: int, towards: int) -> np.ndarray:
        h = (float(towards) - float(at)) / 3.0
        f = natural_spline(values, keys, float(at) + h * np.arange(4.0))
        return (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / h ** 2

    interior = np.stack([second_derivative(keys[i], keys[i + 1]) for i in range(1, keys.shape[0] - 1)])
    scale = float(np.abs(interior).max())
    assert scale >= 0.1
    for end in (second_derivative(keys[0], keys[1]), second_derivative(keys[-1], keys[-2])):
        assert float(np.abs(end).max()) <= 1e-9 * scale
    frames = np.linspace(0.0, 40.0, 161)
    reference = CubicSpline(keys.astype(np.float64), values, axis=0, bc_type="natural")
    assert np.abs(natural_spline(values, keys, frames) - reference(frames)).max() <= 1e-12


def test_quaternion_spline_is_sign_invariant_and_unit(caplog: pytest.LogCaptureFixture) -> None:
    key_frames, out_frames = engine_key_frames(40, 3), np.arange(40)
    quat = _wxyz(Rotation.from_rotvec(0.2 * np.arange(14)[:, None] * np.array([0.3, 0.5, 0.81])))
    assert quat[0, 0] >= 0.0 and bool((np.sum(quat[1:] * quat[:-1], axis=1) > 0).all())  # key 0 sign pinned
    flipped = quat.copy()
    flipped[1::2] *= -1
    caplog.set_level(logging.WARNING, logger="fastdo.retarget.rate")
    base = quaternion_spline(quat, key_frames, out_frames)
    out = quaternion_spline(flipped, key_frames, out_frames)
    assert np.abs(out - base).max() <= 1e-12
    assert np.abs(np.linalg.norm(out, axis=1) - 1.0).max() <= 1e-12
    assert np.abs(out[key_frames] - quat).max() <= 1e-12
    assert not [r for r in caplog.records if r.name == "fastdo.retarget.rate"]

    caplog.clear()
    big = _wxyz(Rotation.from_rotvec(np.radians([0.0, 10.0, 110.0])[:, None] * np.array([0.0, 0.0, 1.0])))
    quaternion_spline(big, np.array([0, 5, 10]), np.arange(11))
    warnings = [r for r in caplog.records if r.name == "fastdo.retarget.rate" and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "(1, 2)" in warnings[0].getMessage()


def test_upsample_object_poses_contract() -> None:
    """Positions by natural spline, rotations by the quaternion spline: key rows are the inputs, affine
    translation is reproduced on every frame, and every output rotation is the matrix of `quaternion_spline`."""
    keys = engine_key_frames(138, 3)
    f = keys.astype(np.float64)
    rot = Rotation.from_rotvec(0.05 * f[:, None] * np.array([0.0, 0.6, 0.8]))
    trans = np.stack([0.01 * f, np.sin(0.1 * f), np.full_like(f, 0.3)], axis=1)
    poses = _poses(rot, trans)
    poses[1::2, :3, :3] = Rotation.from_quat(-rot.as_quat()[1::2]).as_matrix()   # same rotations either way
    out = upsample_object_poses(poses, keys, 138)
    assert out.shape == (138, 4, 4) and out.dtype == np.float64
    assert np.abs(out[keys] - poses).max() <= 1e-12
    assert np.array_equal(out[:, 3], np.tile([0.0, 0.0, 0.0, 1.0], (138, 1)))
    assert np.abs(out[:, :3, :3] @ out[:, :3, :3].transpose(0, 2, 1) - np.eye(3)).max() <= 1e-12
    assert np.abs(out[:, 0, 3] - 0.01 * np.arange(138)).max() <= 1e-12
    expected_rot = Rotation.from_quat(quaternion_spline(_wxyz(rot), keys, np.arange(138))[:, [1, 2, 3, 0]])
    assert np.abs(out[:, :3, :3] - expected_rot.as_matrix()).max() <= 1e-12
    with pytest.raises(ValueError):
        upsample_object_poses(poses[:-1], keys, 138)                      # 46 poses, 47 keys
    with pytest.raises(ValueError):
        upsample_object_poses(poses[:-1], keys[:-1], 138)                 # last key 135, not 137
    with pytest.raises(ValueError):
        upsample_object_poses(poses, keys + 1, 139)                        # keys start at 1


def test_upsample_hand_track_lerps_vertices_in_the_wrist_frame() -> None:
    """The wrist SE(3) follows `upsample_object_poses`; hand vertices are lerped per vertex in the wrist frame
    between keys and mapped back with the upsampled wrist (no MANO layer in the uv env)."""
    keys, n_frames = engine_key_frames(10, 3), 10
    f = keys.astype(np.float64)
    rng = np.random.default_rng(3)
    base, drift = rng.normal(size=(5, 3)) * 0.05, rng.normal(size=(5, 3)) * 0.01
    local_keys = base[None] + f[:, None, None] * drift[None] + (f % 2)[:, None, None] * 0.02   # not affine in f
    wrist = _poses(Rotation.from_rotvec(0.3 * f[:, None] * np.array([0.0, 0.0, 1.0])),
                   np.stack([0.1 * f, 0.02 * f ** 2, np.zeros_like(f)], axis=1))
    verts = np.einsum("kij,kvj->kvi", wrist[:, :3, :3], local_keys) + wrist[:, None, :3, 3]
    wrist_out, verts_out = upsample_hand_track(wrist, verts, keys, n_frames)
    assert wrist_out.shape == (10, 4, 4) and verts_out.shape == (10, 5, 3) and verts_out.dtype == np.float64
    assert np.array_equal(wrist_out, upsample_object_poses(wrist, keys, n_frames))
    assert np.abs(verts_out[keys] - verts).max() <= 1e-12
    for frame, (i, alpha) in {1: (0, 1.0 / 3.0), 5: (1, 2.0 / 3.0), 8: (2, 2.0 / 3.0)}.items():
        local = (1.0 - alpha) * local_keys[i] + alpha * local_keys[i + 1]
        expected = local @ wrist_out[frame, :3, :3].T + wrist_out[frame, :3, 3]
        assert np.abs(verts_out[frame] - expected).max() <= 1e-12, frame
    with pytest.raises(ValueError):
        upsample_hand_track(wrist, verts[:-1], keys, n_frames)
