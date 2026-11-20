"""Robot trajectory export: per-frame joints -> smoothed, limit-respecting 250 Hz arm / 100 Hz hand streams.

Steps (`build_trajectory`):
1. Smoothing: a zero-phase Gaussian filter of `smooth_sigma_s` seconds over the source frames, per joint; the knots
   are then clipped `knot_margin` inside the command limits (a sample clipped at a limit later is a velocity kink,
   which no time scale removes).
2. Interpolation: a clamped cubic spline per joint (zero velocity at both ends).
3. Speed: ONE uniform time-scale factor `s >= 1` for arm and hand together, the smallest with
   max|dq|/s <= v_max and max|ddq|/s^2 <= a_max on every joint (checked on a 1 kHz grid), so the path is unchanged and
   arm and hand stay synchronised; the motion is slowed, never reshaped.
4. Sampling at 250 Hz (arm, `rt/arm_sdk` rate of xr_teleoperate) and 100 Hz (hand, its `rt/dex3` rate), with
   `hold_s` of the first and last pose prepended and appended.
5. Position limits: samples are clipped into the command limits (arm limits shrunk by `arm_margin`). Then every
   stream is re-checked with finite differences (`check_trajectory`); a clip that creates a violation grows `s` by 5%
   and the export is redone (at most 30 times, then ValueError).
6. Feed-forward torque: the gravity torque `tau_ff = g(q)` of the right arm at each commanded sample (MuJoCo RNE with
   zero velocity and acceleration), xr_teleoperate's `pin.rnea(q, 0, 0)`.

Limits (`TrajectoryLimits`) default to conservative first-hardware-run values, NOT tuned: arm 1.0 rad/s and 5 rad/s^2
(the URDF allows 37 / 22 rad/s; Unitree's arm7 SDK example moves at 0.5 rad/s), hand 2.0 rad/s and 20 rad/s^2 (URDF
3.14 rad/s for thumb_0, 12 elsewhere); each is also capped by the URDF velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from fastdo.robot import g1_constants as C
from fastdo.robot.g1_model import G1Upper

__all__ = ["ExportConfig", "RobotTrajectory", "TrajectoryLimits", "build_trajectory", "check_trajectory",
           "gravity_torque", "left_park_torque"]


@dataclass(frozen=True)
class TrajectoryLimits:
    arm_vmax: tuple[float, ...] = tuple(min(1.0, v) for v in C.ARM_URDF_VELOCITY)
    arm_amax: tuple[float, ...] = (5.0,) * 7
    hand_vmax: tuple[float, ...] = tuple(min(2.0, v) for v in C.HAND_URDF_VELOCITY)
    hand_amax: tuple[float, ...] = (20.0,) * 7
    arm_margin: float = 0.02
    arm_limits: tuple[tuple[float, float], ...] = field(default=C.RIGHT_ARM_COMMAND_LIMITS)
    hand_limits: tuple[tuple[float, float], ...] = field(default=C.RIGHT_HAND_COMMAND_LIMITS)

    def arm_bounds(self) -> np.ndarray:
        return np.asarray(self.arm_limits) + np.array([self.arm_margin, -self.arm_margin])

    def hand_bounds(self) -> np.ndarray:
        return np.asarray(self.hand_limits, dtype=np.float64)


@dataclass(frozen=True)
class ExportConfig:
    arm_rate_hz: float = 250.0
    hand_rate_hz: float = 100.0
    smooth_sigma_s: float = 0.1
    hold_s: float = 0.5
    check_rate_hz: float = 1000.0
    knot_margin: float = 0.01


@dataclass(frozen=True)
class RobotTrajectory:
    arm_t: np.ndarray        # (N,) s
    arm_q: np.ndarray        # (N,7) C.RIGHT_ARM_JOINTS order
    arm_dq: np.ndarray       # (N,7)
    arm_tau_ff: np.ndarray   # (N,7) N m
    hand_t: np.ndarray       # (M,) s
    hand_q: np.ndarray       # (M,7) hardware order (C.RIGHT_HAND_JOINTS)
    hand_dq: np.ndarray      # (M,7)
    time_scale: float        # s >= 1 (step 3)
    hold_s: float
    source_t: np.ndarray     # (K,) source frame times, relative to the first

    def source_time(self, t: np.ndarray | float) -> np.ndarray:
        """Source-clip time (s from its first frame) of trajectory time `t` (hold segments clamp)."""
        tau = (np.asarray(t, dtype=np.float64) - self.hold_s) / self.time_scale
        return np.clip(tau, 0.0, float(self.source_t[-1]))

    @property
    def duration(self) -> float:
        return float(self.arm_t[-1])


def gravity_torque(g1: G1Upper, q_arm: np.ndarray, q_hand: np.ndarray) -> np.ndarray:
    """`(N,7)` right-arm gravity torque at `(N,7)` arm and hand joints (left arm parked, zero velocity)."""
    m = g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = g1.default_qpos()
    out = np.zeros((len(q_arm), 7))
    res = np.zeros(m.nv)
    for k in range(len(q_arm)):
        d.qpos[g1.right_arm_qpos] = q_arm[k]
        d.qpos[g1.right_hand_qpos] = q_hand[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_rne(m, d, 0, res)
        out[k] = res[g1.right_arm_dof]
    return out


def left_park_torque(g1: G1Upper) -> np.ndarray:
    """`(7,)` gravity torque of the left arm at `C.LEFT_ARM_PARK` (fixed base: independent of the right arm)."""
    m = g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = g1.default_qpos()
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    res = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 0, res)
    dof = m.jnt_dofadr[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.LEFT_ARM_JOINTS]]
    return res[dof]


def _fd(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    v = np.diff(x, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    return v, a


def check_trajectory(traj: RobotTrajectory, limits: TrajectoryLimits, tol: float = 1e-6) -> list[str]:
    """Violations of positions, finite-difference velocities and accelerations, and sampling rates (empty = valid)."""
    out: list[str] = []
    for name, t, q, bounds, vmax, amax in (
            ("arm", traj.arm_t, traj.arm_q, limits.arm_bounds(), limits.arm_vmax, limits.arm_amax),
            ("hand", traj.hand_t, traj.hand_q, limits.hand_bounds(), limits.hand_vmax, limits.hand_amax)):
        if not np.isfinite(q).all():
            out.append(f"{name}: non-finite samples")
            continue
        dt = np.diff(t)
        if dt.size == 0 or np.ptp(dt) > 1e-9:
            out.append(f"{name}: sampling is not uniform")
            continue
        low = q < bounds[:, 0] - tol
        high = q > bounds[:, 1] + tol
        if low.any() or high.any():
            out.append(f"{name}: {int(low.sum() + high.sum())} samples outside the command limits")
        v, a = _fd(q, float(dt[0]))
        vr = np.abs(v).max(axis=0) / np.asarray(vmax)
        ar = np.abs(a).max(axis=0) / np.asarray(amax)
        if (vr > 1.0 + 1e-3).any():
            out.append(f"{name}: velocity {vr.max():.3f}x the limit (joint {int(vr.argmax())})")
        if (ar > 1.0 + 1e-3).any():
            out.append(f"{name}: acceleration {ar.max():.3f}x the limit (joint {int(ar.argmax())})")
    return out


def _scale_factor(spl: CubicSpline, duration: float, vmax: np.ndarray, amax: np.ndarray, rate: float) -> float:
    tt = np.linspace(0.0, duration, max(int(duration * rate) + 1, 2))
    v = np.abs(spl(tt, 1)).max(axis=0)
    a = np.abs(spl(tt, 2)).max(axis=0)
    return float(max(1.0, (v / vmax).max(), np.sqrt((a / amax).max())))


def _sample(spl: CubicSpline, s: float, duration: float, hold: float, rate: float, bounds: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = hold * 2.0 + s * duration
    n = int(np.floor(total * rate + 1e-9)) + 1
    t = np.arange(n) / rate
    tau = np.clip((t - hold) / s, 0.0, duration)
    moving = (t >= hold) & (t <= hold + s * duration)
    q = np.clip(spl(tau), bounds[:, 0], bounds[:, 1])
    dq = np.where(moving[:, None], spl(tau, 1) / s, 0.0)
    return t, q, dq


def build_trajectory(g1: G1Upper, source_t: np.ndarray, q_arm: np.ndarray, q_hand: np.ndarray,
                     cfg: ExportConfig | None = None, limits: TrajectoryLimits | None = None) -> RobotTrajectory:
    """Steps 1-6 of the module docstring for `(K,)` source times and `(K,7)` arm and hand joints.

    Raises:
        ValueError: On non-increasing times, mismatched shapes, non-finite joints, or no feasible time scale.
    """
    cfg = ExportConfig() if cfg is None else cfg
    limits = TrajectoryLimits() if limits is None else limits
    t = np.asarray(source_t, dtype=np.float64)
    t = t - t[0]
    qa = np.asarray(q_arm, dtype=np.float64)
    qh = np.asarray(q_hand, dtype=np.float64)
    if t.ndim != 1 or len(t) < 2 or (np.diff(t) <= 0).any():
        raise ValueError("source times must be strictly increasing, at least two")
    if qa.shape != (len(t), 7) or qh.shape != (len(t), 7):
        raise ValueError(f"expected (K,7) joints for K = {len(t)}, got {qa.shape} and {qh.shape}")
    if not (np.isfinite(qa).all() and np.isfinite(qh).all()):
        raise ValueError("non-finite joints")
    sigma = cfg.smooth_sigma_s / float(np.median(np.diff(t)))
    if sigma > 0:
        qa = gaussian_filter1d(qa, sigma, axis=0, mode="nearest")
        qh = gaussian_filter1d(qh, sigma, axis=0, mode="nearest")
    m = cfg.knot_margin
    qa = np.clip(qa, limits.arm_bounds()[:, 0] + m, limits.arm_bounds()[:, 1] - m)
    qh = np.clip(qh, limits.hand_bounds()[:, 0] + m, limits.hand_bounds()[:, 1] - m)
    arm_spl = CubicSpline(t, qa, bc_type="clamped")
    hand_spl = CubicSpline(t, qh, bc_type="clamped")
    duration = float(t[-1])
    s = max(_scale_factor(arm_spl, duration, np.asarray(limits.arm_vmax), np.asarray(limits.arm_amax),
                          cfg.check_rate_hz),
            _scale_factor(hand_spl, duration, np.asarray(limits.hand_vmax), np.asarray(limits.hand_amax),
                          cfg.check_rate_hz))
    for _ in range(30):
        arm_t, arm_q, arm_dq = _sample(arm_spl, s, duration, cfg.hold_s, cfg.arm_rate_hz, limits.arm_bounds())
        hand_t, hand_q, hand_dq = _sample(hand_spl, s, duration, cfg.hold_s, cfg.hand_rate_hz, limits.hand_bounds())
        traj = RobotTrajectory(arm_t=arm_t, arm_q=arm_q, arm_dq=arm_dq, arm_tau_ff=np.zeros_like(arm_q),
                               hand_t=hand_t, hand_q=hand_q, hand_dq=hand_dq, time_scale=s, hold_s=cfg.hold_s,
                               source_t=t)
        if not check_trajectory(traj, limits):
            hand_at_arm = np.clip(hand_spl(np.clip((arm_t - cfg.hold_s) / s, 0.0, duration)),
                                  limits.hand_bounds()[:, 0], limits.hand_bounds()[:, 1])
            tau = gravity_torque(g1, arm_q, hand_at_arm)
            return replace(traj, arm_tau_ff=tau)
        s *= 1.05
    raise ValueError(f"no feasible time scale: {check_trajectory(traj, limits)}")
