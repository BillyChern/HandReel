"""Grasp closure for the kinematic robot path.

Open-loop kinematic replay puts the retargeted Dex3 fingertips ON the object surface with no normal force: the
provisional lab clips that passed S1 and P14 moved their objects 0.5 of 24.1 cm and 0.2 of 18.1 cm. Here, over the
contact phase (task rows c0..c1), every finger gets the same FLEXION BIAS (`approach.FLEXION` signs, on its two
flexion joints; thumb_0 never):

    bias(t) = a * r(t),   r = smoothstep((t - t[c0]) / ramp_s), and after c1 (if the task continues) it ramps
                          back down over ramp_s with the same smoothstep (the hand releases before the retreat)

added to the retargeted hand rows and clipped to the command limits minus the export's knot margin (so the bias can
never push a joint past a limit). The amplitude `a` is the SMALLEST on [0, `max_bias`] (bisection, `iters` rounds)
for which the SIMULATED fingertip penetration (each finger's distal link into the object, MuJoCo soft contact: its
depth is the contact-force proxy) reaches `target_m` during the contact phase on EVERY finger that touches the object
at the full bias. One common amplitude, not one per finger: with per-finger amplitudes the thumb reached the target
by being pushed (amplitude ~0.01 rad) and the sphere known answer was squeezed out sideways instead of lifted.
When the target is not reached at `max_bias`, `max_bias` is kept and `reached` is False. The bias is BAKED
into the exported trajectory (the robot plays it open loop); the package's replay simulates exactly that trajectory.

`CLOSURE_TARGET_M` is the frozen target depth (see its comment for the provenance).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fastdo.robot.approach import FLEXION, smoothstep
from fastdo.robot.g1_model import G1Upper
from fastdo.robot.replay import ReplayScene, SuccessCriteria, replay
from fastdo.robot.trajectory import (
    ExportConfig,
    RobotTrajectory,
    TrajectoryLimits,
    build_trajectory,
)

__all__ = ["CLOSURE_TARGET_M", "FINGERS", "ClosureConfig", "close_grasp", "closure_bias", "finger_body_masks"]

# FROZEN target fingertip penetration (m). Provenance: a development tuning run on 6 provisional clips ONLY, grid
# 0.5/1/2/3/4 mm. 0 of 6 clips succeeded at any target. 2 mm won the committed tie-break
# (summed lift fraction 1.136 against 0.636 at 1 mm; 2-4 mm tie, the smallest is taken). At >= 2 mm the bisection
# saturates at max_bias on every clip, so the value is NOT a measured optimum: it is the rule's pick on a flat
# objective. It is frozen here before any production-pin clip is scored.
CLOSURE_TARGET_M = 0.002

# (name, its flexion joints in hardware order, its distal link)
FINGERS: tuple[tuple[str, tuple[int, int], str], ...] = (
    ("thumb", (1, 2), "right_hand_thumb_2_link"),
    ("index", (3, 4), "right_hand_index_1_link"),
    ("middle", (5, 6), "right_hand_middle_1_link"),
)


@dataclass(frozen=True)
class ClosureConfig:
    target_m: float = CLOSURE_TARGET_M
    ramp_s: float = 0.3
    max_bias: float = 1.0           # rad, before the command-limit clip
    iters: int = 7
    limit_margin: float = 0.05      # rad: a biased joint stops this far inside its command limit. At the export's
                                    #   0.01-rad knot margin the object pushed the simulated thumb_2 0.023 rad past its
                                    #   MJCF limit (replay `joint_limits`, provisional clip 144357)


def finger_body_masks(scene: ReplayScene) -> tuple[np.ndarray, ...]:
    m = scene.g1.model
    out = []
    for _, _, link in FINGERS:
        mask = np.zeros(m.nbody, dtype=bool)
        mask[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)] = True
        out.append(mask)
    return tuple(out)


def closure_bias(times: np.ndarray, span: tuple[int, int], amplitude: np.ndarray, ramp_s: float) -> np.ndarray:
    """`(T,7)` flexion bias rows (module docstring) for the finger amplitudes `(3,)` (rad; `close_grasp` uses one
    common value)."""
    t = np.asarray(times, dtype=np.float64)
    c0, c1 = span
    r = smoothstep((t - t[c0]) / ramp_s)
    if c1 < len(t) - 1:
        r = r * (1.0 - smoothstep((t - t[c1]) / ramp_s))
    out = np.zeros((len(t), 7))
    for (_, joints, _), a in zip(FINGERS, np.asarray(amplitude, dtype=np.float64), strict=True):
        for j in joints:
            out[:, j] = FLEXION[j] * a * r
    return out


def _biased(hand_q: np.ndarray, bias: np.ndarray, limits: TrajectoryLimits, export: ExportConfig,
            limit_margin: float = 0.05) -> np.ndarray:
    """The hand rows plus the bias, a biased joint stopping `limit_margin` inside its command limit (never moved
    against the bias direction by that stop), and every joint inside the limits minus the knot margin."""
    b = limits.hand_bounds()
    q = np.asarray(hand_q, dtype=np.float64)
    m = max(limit_margin, export.knot_margin)
    eff = np.clip(q + bias, b[:, 0] + m, b[:, 1] - m) - q
    eff = np.where(np.sign(eff) == np.sign(bias), eff, 0.0)
    return np.clip(q + eff, b[:, 0] + export.knot_margin, b[:, 1] - export.knot_margin)


def _phase_depth(traj: RobotTrajectory, logs: dict, times: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    """`(3,)` deepest fingertip penetration over the contact phase (trajectory time of rows c0..c1)."""
    t0 = float(times[0])
    lo = traj.hold_s + traj.time_scale * (float(times[span[0]]) - t0)
    hi = traj.hold_s + traj.time_scale * (float(times[span[1]]) - t0)
    sel = (logs["t"] >= lo - 1e-9) & (logs["t"] <= hi + 1e-9)
    fd = np.asarray(logs["finger_depth"])
    return fd[sel].max(axis=0) if sel.any() else np.zeros(fd.shape[1])


def close_grasp(g1: G1Upper, scene: ReplayScene, times: np.ndarray, q_arm: np.ndarray, hand_q: np.ndarray,
                span: tuple[int, int], object_ref: np.ndarray, object_start: np.ndarray, ref_rows: np.ndarray,
                criteria: SuccessCriteria, export: ExportConfig, limits: TrajectoryLimits,
                cfg: ClosureConfig | None = None) -> tuple[np.ndarray, dict]:
    """`(biased hand rows (T,7), record)` (module docstring)."""
    cfg = ClosureConfig() if cfg is None else cfg
    masks = finger_body_masks(scene)

    def run(amp: np.ndarray) -> tuple[np.ndarray, np.ndarray, RobotTrajectory, dict]:
        rows = _biased(hand_q, closure_bias(times, span, amp, cfg.ramp_s), limits, export, cfg.limit_margin)
        traj = build_trajectory(g1, times, q_arm, rows, export, limits)
        _, logs = replay(scene, traj, object_ref, object_start, criteria, ref_rows=ref_rows, finger_bodies=masks)
        return rows, _phase_depth(traj, logs, times, span), traj, logs

    n = len(FINGERS)
    _, d0, _, _ = run(np.zeros(n))
    _, dmax, _, _ = run(np.full(n, cfg.max_bias))
    touching = dmax > 0.0                        # fingers the full bias brings onto the object

    def reached(d: np.ndarray) -> bool:
        return bool(touching.any() and (d[touching] >= cfg.target_m).all())

    lo, hi = 0.0, cfg.max_bias
    if reached(d0):
        hi = 0.0
    elif reached(dmax):
        for _ in range(cfg.iters):
            mid = 0.5 * (lo + hi)
            _, dm, _, _ = run(np.full(n, mid))
            if reached(dm):
                hi = mid
            else:
                lo = mid
    rows, dfin, _, _ = run(np.full(n, hi))
    rec = {"rule": "common flexion bias, smallest amplitude reaching the target depth", "target_m": cfg.target_m,
           "ramp_s": cfg.ramp_s, "max_bias_rad": cfg.max_bias, "limit_margin_rad": cfg.limit_margin,
           "iters": cfg.iters, "span_task_rows": [int(span[0]), int(span[1])],
           "amplitude_rad": float(hi), "touching_at_max": {f[0]: bool(x) for f, x in zip(FINGERS, touching, strict=True)},
           "depth_without_m": {f[0]: float(x) for f, x in zip(FINGERS, d0, strict=True)},
           "depth_at_max_m": {f[0]: float(x) for f, x in zip(FINGERS, dmax, strict=True)},
           "depth_final_m": {f[0]: float(x) for f, x in zip(FINGERS, dfin, strict=True)},
           "reached": reached(dfin),
           "simulations": cfg.iters + 3}
    return rows, rec
