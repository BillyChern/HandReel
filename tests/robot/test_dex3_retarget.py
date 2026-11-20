"""Tests for `fastdo.robot.dex3_retarget`: Unitree's DexPilot-style MANO -> Dex3-1 retargeting.

Known answers: Dex3-1's own forward-kinematics keypoints, retargeted, give back its joint angles (exactly without the
`norm_delta` pull; exactly when warm-started at the answer with it; to ~0.01 rad along a smooth walk). The same holds
through the MANO path (hand frame from MCP joints, keypoints in that frame). Mutations: a wrong hand-frame handedness
and a wrong hardware order both break the recovery. Plus the gradient, the pinch projection's hysteresis and the
rate-matched low-pass.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.robot import dex3_retarget as R
from fastdo.robot.dex3_retarget import (
    Dex3DexPilot,
    DexPilotConfig,
    dex3_tip_positions,
    dexpilot_vectors,
    mano_hand_frame,
    mano_keypoints,
    rate_matched_alpha,
)

_EXACT = DexPilotConfig(norm_delta=0.0)


def _samples(g1, n: int, seed: int) -> np.ndarray:
    """Joint samples away from the limits whose thumb-index/middle distances stay out of the pinch zone (5 cm)."""
    lim = Dex3DexPilot(g1).limits
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = rng.uniform(lim[:, 0] + 0.05, lim[:, 1] - 0.05)
        v = dexpilot_vectors(dex3_tip_positions(g1, q)[0])
        if np.linalg.norm(v[:2], axis=1).min() > 0.05:
            out.append(q)
    return np.asarray(out)


def test_gradient_matches_finite_differences(g1) -> None:
    r = Dex3DexPilot(g1)
    q_true, q = _samples(g1, 2, seed=3)
    ref, w = r._reference(dexpilot_vectors(dex3_tip_positions(g1, q_true)[0]))
    last = r.limits.mean(axis=1)
    _, grad = r.objective(q, ref, w, last)
    eps = 1e-6
    num = np.array([(r.objective(q + eps * e, ref, w, last)[0] - r.objective(q - eps * e, ref, w, last)[0]) / (2 * eps)
                    for e in np.eye(7)])
    assert np.abs(num - grad).max() < 1e-7


def test_own_fk_recovers_joint_angles_cold_start(g1) -> None:
    """Known answer: without the norm_delta pull, from Unitree's mid-range start, the joints come back to 1e-4 rad."""
    for q_true in _samples(g1, 12, seed=0):
        q = Dex3DexPilot(g1, _EXACT).solve(dexpilot_vectors(dex3_tip_positions(g1, q_true)[0]))
        assert np.abs(q - q_true).max() < 1e-4


def test_own_fk_recovers_joint_angles_with_default_config(g1) -> None:
    """With norm_delta (default), the answer is a fixed point when the warm start is the answer."""
    for q_true in _samples(g1, 6, seed=1):
        r = Dex3DexPilot(g1)
        r.last_q = q_true.copy()
        q = r.solve(dexpilot_vectors(dex3_tip_positions(g1, q_true)[0]))
        assert np.abs(q - q_true).max() < 1e-5


def test_sequence_walk_tracks_with_default_config(g1) -> None:
    lim = Dex3DexPilot(g1).limits
    rng = np.random.default_rng(2)
    q = _samples(g1, 1, seed=2)[0]
    walk = []
    while len(walk) < 100:                       # a random walk that stays out of the pinch zone
        step = np.clip(q + rng.normal(0, 0.02, 7), lim[:, 0] + 0.02, lim[:, 1] - 0.02)
        if np.linalg.norm(dexpilot_vectors(dex3_tip_positions(g1, step)[0])[:2], axis=1).min() > 0.05:
            q = step
            walk.append(q.copy())
    walk = np.asarray(walk)
    r = Dex3DexPilot(g1)
    r.last_q = walk[0].copy()
    out = r.retarget_sequence(dexpilot_vectors(dex3_tip_positions(g1, walk)), low_pass=False)
    err = np.abs(out - walk).max(axis=1)
    assert np.median(err) < 0.02


def _synthetic_mano(g1, q_true: np.ndarray, rot: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """`(1,21,3)` OpenPose joints whose `mano_hand_frame` is `rot` and whose tips are Dex3's FK tips."""
    joints = np.zeros((21, 3))
    joints[0] = wrist
    joints[9] = wrist + rot @ [0.0, -0.09, 0.0]
    joints[5] = wrist + rot @ [0.0, -0.085, -0.02]
    joints[17] = wrist + rot @ [0.0, -0.07, 0.04]
    tips = dex3_tip_positions(g1, q_true)[0][1:]
    joints[[4, 8, 12]] = wrist + tips @ rot.T
    return joints[None]


def _mano_retarget(g1, joints: np.ndarray) -> np.ndarray:
    frame = mano_hand_frame(joints)
    kp = mano_keypoints(joints, None)
    kp_hand = np.einsum("tvi,tij->tvj", kp - joints[:, None, 0], frame)
    return Dex3DexPilot(g1, _EXACT).solve(dexpilot_vectors(kp_hand)[0])


def test_mano_path_recovers_joint_angles(g1) -> None:
    rot = Rotation.from_euler("xyz", [0.4, -1.1, 2.0]).as_matrix()
    for q_true in _samples(g1, 4, seed=4):
        joints = _synthetic_mano(g1, q_true, rot, np.array([0.3, -0.2, 0.9]))
        assert np.allclose(mano_hand_frame(joints)[0], rot, atol=1e-9)
        assert np.abs(_mano_retarget(g1, joints) - q_true).max() < 1e-4


def test_mano_frame_handedness_mutation_is_caught(g1, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: a hand frame with x and z flipped (still a rotation) retargets to different joints."""
    real = R.mano_hand_frame
    monkeypatch.setattr(R, "mano_hand_frame", lambda j: real(j) * np.array([-1.0, 1.0, -1.0]))
    rot = Rotation.from_euler("xyz", [0.4, -1.1, 2.0]).as_matrix()
    q_true = _samples(g1, 1, seed=4)[0]
    joints = _synthetic_mano(g1, q_true, rot, np.zeros(3))
    frame = R.mano_hand_frame(joints)
    kp_hand = np.einsum("tvi,tij->tvj", mano_keypoints(joints, None) - joints[:, None, 0], frame)
    q = Dex3DexPilot(g1, _EXACT).solve(dexpilot_vectors(kp_hand)[0])
    assert np.abs(q - q_true).max() > 0.05


def test_hardware_order_mutation_is_caught(g1) -> None:
    """Mutation: reading the solution in pinocchio order (index, middle, thumb) is not the hardware command."""
    q_true = _samples(g1, 1, seed=5)[0]
    q = Dex3DexPilot(g1, _EXACT).solve(dexpilot_vectors(dex3_tip_positions(g1, q_true)[0]))
    pin_order = q[[3, 4, 5, 6, 0, 1, 2]]
    assert np.abs(q - q_true).max() < 1e-4 and np.abs(pin_order - q_true).max() > 0.05


def test_mano_keypoints_use_tip_vertices() -> None:
    joints = np.zeros((1, 21, 3))
    verts = np.zeros((1, 778, 3))
    verts[0, 745], verts[0, 317], verts[0, 444] = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    assert np.array_equal(mano_keypoints(joints, verts)[0, 1:], np.eye(3))


def test_pinch_projection_and_hysteresis(g1) -> None:
    r = Dex3DexPilot(g1)
    target = np.zeros((6, 3))
    target[3:] = [[0.0, -0.1, 0.0], [0.0, -0.15, 0.0], [0.0, -0.16, 0.0]]
    for dist, projected in ((0.02, True), (0.04, True), (0.06, False), (0.04, False)):
        target[0] = [dist, 0.0, 0.0]
        target[1] = [0.0, 0.0, 0.08]
        ref, weight = r._reference(target.copy())
        assert bool(r.projected[0]) is projected, dist
        if projected:
            assert np.allclose(ref[0], [1e-4, 0.0, 0.0], atol=1e-8) and weight[0] == 200.0
        else:
            assert np.allclose(ref[0], target[0]) and weight[0] == 1.0
        assert np.all(weight[3:] == 6.0)


def test_rate_matched_low_pass(g1) -> None:
    assert rate_matched_alpha(0.2, 100.0, 100.0) == pytest.approx(0.2)
    assert rate_matched_alpha(0.2, 100.0, 10.0) == pytest.approx(1 - 0.8 ** 10)
    r = Dex3DexPilot(g1, rate_hz=100.0)
    q_a, q_b = _samples(g1, 2, seed=6)
    first = r.step(dexpilot_vectors(dex3_tip_positions(g1, q_a)[0]))
    second = r.step(dexpilot_vectors(dex3_tip_positions(g1, q_b)[0]))
    assert np.allclose(second, first + 0.2 * (r.last_q - first))


def test_invalid_rows_are_held(g1) -> None:
    q_true = _samples(g1, 2, seed=7)
    vec = dexpilot_vectors(dex3_tip_positions(g1, q_true))
    out = Dex3DexPilot(g1, _EXACT).retarget_sequence(np.stack([vec[0], vec[1], vec[1]]),
                                                    valid=np.array([False, True, False]), low_pass=False)
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[2], out[1])
    with pytest.raises(ValueError):
        Dex3DexPilot(g1).retarget_sequence(vec, valid=np.zeros(2, dtype=bool))
