"""Tests for `fastdo.robot.trajectory`: 250/100 Hz sampling, uniform time scaling under velocity and acceleration
limits, position clipping, gravity feed-forward, and the finite-difference checker (with mutations)."""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from fastdo.robot import g1_constants as C
from fastdo.robot.trajectory import (
    ExportConfig,
    TrajectoryLimits,
    build_trajectory,
    check_trajectory,
    gravity_torque,
)

_NO_SMOOTH = ExportConfig(smooth_sigma_s=0.0, hold_s=0.5)


def _ramp(n: int, dt: float, amp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(n) * dt
    s = np.sin(np.linspace(0, np.pi / 2, n)) * amp
    arm = np.array([-0.3, -0.3, 0.1, 0.8, 0.0, 0.0, 0.0]) + s[:, None]
    hand = np.array([0.0, 0.0, -0.3, 0.3, 0.3, 0.3, 0.3]) + 0.5 * s[:, None] * np.array([0, 0, -1, 1, 1, 1, 1])
    return t, arm, hand


def test_slow_motion_is_sampled_unscaled(g1) -> None:
    """Known answer: a motion already within the limits keeps time scale 1; the streams are uniform at 250 / 100 Hz,
    hold the end poses for hold_s with zero velocity, and pass through the input end points."""
    t, arm, hand = _ramp(31, 0.1, 0.2)
    traj = build_trajectory(g1, t, arm, hand, _NO_SMOOTH)
    assert traj.time_scale == pytest.approx(1.0)
    assert np.allclose(np.diff(traj.arm_t), 1 / 250) and np.allclose(np.diff(traj.hand_t), 1 / 100)
    assert traj.duration == pytest.approx(3.0 + 2 * 0.5, abs=1 / 250)
    hold = traj.arm_t < 0.5
    assert np.allclose(traj.arm_q[hold], arm[0]) and np.allclose(traj.arm_dq[hold], 0.0)
    assert np.allclose(traj.arm_q[-1], arm[-1], atol=1e-9) and np.allclose(traj.hand_q[-1], hand[-1], atol=1e-9)
    assert check_trajectory(traj, TrajectoryLimits()) == []


def test_fast_motion_is_slowed_to_the_limits(g1) -> None:
    t, arm, hand = _ramp(11, 0.1, 1.5)                    # 1.5 rad in 1 s: above the 1 rad/s arm cap
    limits = TrajectoryLimits()
    traj = build_trajectory(g1, t, arm, hand, _NO_SMOOTH, limits)
    assert traj.time_scale > 1.5
    assert check_trajectory(traj, limits) == []
    v = np.abs(np.diff(traj.arm_q, axis=0)) * 250
    assert v.max() <= max(limits.arm_vmax) * (1 + 1e-3)
    assert traj.hand_t[-1] == pytest.approx(traj.arm_t[-1], abs=0.01)


def test_positions_are_clipped_into_the_command_limits(g1) -> None:
    t, arm, hand = _ramp(21, 0.1, 0.1)
    arm[:, 3] = 2.5                                       # elbow above its 2.0944 rad limit
    hand[:, 1] = 0.9                                      # thumb_1 above the 0.611 rad intersection limit
    limits = TrajectoryLimits()
    traj = build_trajectory(g1, t, arm, hand, _NO_SMOOTH, limits)
    assert limits.arm_bounds()[3, 1] - 0.011 < traj.arm_q[:, 3].max() <= limits.arm_bounds()[3, 1]
    assert C.RIGHT_HAND_COMMAND_LIMITS[1][1] - 0.011 < traj.hand_q[:, 1].max() <= C.RIGHT_HAND_COMMAND_LIMITS[1][1]
    assert check_trajectory(traj, limits) == []


def test_feedforward_is_the_gravity_torque(g1) -> None:
    t, arm, hand = _ramp(11, 0.1, 0.2)
    traj = build_trajectory(g1, t, arm, hand, _NO_SMOOTH)
    k = len(traj.arm_t) // 2
    m = g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = g1.default_qpos()
    d.qpos[g1.right_arm_qpos] = traj.arm_q[k]
    d.qpos[g1.right_hand_qpos] = traj.hand_q[np.searchsorted(traj.hand_t, traj.arm_t[k])]   # nearest hand sample
    mujoco.mj_forward(m, d)
    assert np.allclose(traj.arm_tau_ff[k], d.qfrc_bias[g1.right_arm_dof], atol=0.05)
    assert np.allclose(gravity_torque(g1, traj.arm_q[k:k + 1], d.qpos[g1.right_hand_qpos][None])[0],
                       d.qfrc_bias[g1.right_arm_dof], atol=1e-9)


def test_checker_catches_mutations(g1) -> None:
    """Mutations: a 0.05 rad jump in one 4 ms tick, a sample outside the limits, a non-uniform clock."""
    t, arm, hand = _ramp(11, 0.1, 0.2)
    limits = TrajectoryLimits()
    traj = build_trajectory(g1, t, arm, hand, _NO_SMOOTH, limits)
    jump = traj.arm_q.copy()
    jump[100:, 2] += 0.05
    assert any("velocity" in e for e in check_trajectory(replace(traj, arm_q=jump), limits))
    out = traj.hand_q.copy()
    out[50, 0] = 2.0
    assert any("limits" in e for e in check_trajectory(replace(traj, hand_q=out), limits))
    clock = traj.arm_t.copy()
    clock[10:] += 0.001
    assert any("uniform" in e for e in check_trajectory(replace(traj, arm_t=clock), limits))


def test_bad_inputs_raise(g1) -> None:
    t, arm, hand = _ramp(5, 0.1, 0.1)
    with pytest.raises(ValueError):
        build_trajectory(g1, t[::-1], arm, hand)
    with pytest.raises(ValueError):
        build_trajectory(g1, t, arm[:, :6], hand)
    arm[2, 0] = np.nan
    with pytest.raises(ValueError):
        build_trajectory(g1, t, arm, hand)
