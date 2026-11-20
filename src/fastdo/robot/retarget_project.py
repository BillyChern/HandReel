"""Penetration-free projection of a retargeted Dex3 grasp.

The planned Dex3 grasps of the S3 development clips lie 9-53 mm INSIDE their objects, because the DexPilot-style
retarget (`dex3_retarget`) never sees the object. This projection adds the object.

The target keypoints are the RETARGET's own Dex3 keypoints (wrist and the three
fingertips, `dex3_retarget.dex3_tip_positions`) in the object frame, not the human's. The retarget already minimises
the human keypoint loss, and the packages carry no human keypoints. So this is the minimal-change projection of the
retarget onto the penetration-free set, and its keypoint deviation is exactly how far non-penetration moves the hand
from the retarget. It is reported per grasp.

    min over (dq, dt, dr):  ||K(q0 + dq, X(dt, dr)) - K0||^2 / k_scale^2  +  w * max(0, -clearance - a)^2 / a^2

- X(dt, dr) = base_obj0 @ [R(dr) | dt]: the Dex3 base in the object frame, moved in its own frame.
- dt and dr are projected onto the 60 mm / 20 deg norm balls (a fixed bound, never tuned).
- The hand stays inside the command limits minus `limit_margin_rad`.
- a = `aim_frac` x pen_m: a penalty converges onto its boundary from OUTSIDE, so it aims inside the limit; feasibility
  is judged at pen_m itself.
- Penalty continuation: w = 1, 10, ... for at most `rounds` rounds (bounded Powell, warm-started), stopping at the
  first feasible round.
- Starts: the retarget itself, and backed out along the direction away from the object centre by 0.5 and 1.0 x the
  shift bound. The best result is feasible first, then has the least keypoint change. Evaluated hand poses are
  quantised to `HandClearance`'s 1e-9 cache key, which makes the result independent of evaluation history: an
  unquantised Powell run was order-dependent on deep pushes, 2026-10-17.
- A start that is already feasible (clearance >= -pen_m) is returned UNCHANGED.
- The result is `feasible` only if the final clearance is >= -pen_m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from fastdo.robot import g1_constants as C
from fastdo.robot.approach import HandClearance
from fastdo.robot.dex3_retarget import dex3_tip_positions
from fastdo.robot.grasp_closure import CLOSURE_TARGET_M

__all__ = ["ProjectConfig", "ProjectResult", "project_grasp"]


@dataclass(frozen=True)
class ProjectConfig:
    pen_m: float = CLOSURE_TARGET_M          # the frozen grasp-closure penetration target (2 mm)
    max_shift_m: float = 0.06                # the base-pose bound
    max_rot_rad: float = float(np.radians(20.0))
    limit_margin_rad: float = 0.05
    k_scale_m: float = 0.01                  # keypoint error unit (1 cm)
    aim_frac: float = 0.5                    # the penalty aims at aim_frac * pen_m (a penalty converges from outside)
    rounds: int = 6
    maxfev: int = 3000


@dataclass(frozen=True)
class ProjectResult:
    hand_q: np.ndarray
    base_obj: np.ndarray
    clearance_start_m: float
    clearance_m: float
    feasible: bool
    keypoint_rms_mm: float       # RMS over wrist + 3 tips of the keypoint displacement (object frame)
    keypoint_max_mm: float
    shift_m: float
    rot_deg: float
    dq_norm_rad: float
    rounds: int


def _ball(v: np.ndarray, r: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= r else v * (r / n)


def _keypoints(clr: HandClearance, base_obj: np.ndarray, q: np.ndarray) -> np.ndarray:
    k = dex3_tip_positions(clr.scene.g1, np.asarray(q, float)[None])[0]          # (4, 3) in the Dex3 base frame
    return k @ np.asarray(base_obj, float)[:3, :3].T + np.asarray(base_obj, float)[:3, 3]


def project_grasp(clr: HandClearance, base_obj0: np.ndarray, hand_q0: np.ndarray,
                  cfg: ProjectConfig | None = None, centre: np.ndarray | None = None) -> ProjectResult:
    """Project one grasp onto the penetration-free set with minimal keypoint change (module docstring). `centre`
    (object frame, default the origin) orients the back-out start."""
    cfg = ProjectConfig() if cfg is None else cfg
    lim = np.asarray(C.RIGHT_HAND_COMMAND_LIMITS, float)
    lo, hi = lim[:, 0] + cfg.limit_margin_rad, lim[:, 1] - cfg.limit_margin_rad
    q0 = np.clip(np.asarray(hand_q0, float), lo, hi)
    b0 = np.asarray(base_obj0, float)
    c0 = clr.clearance(b0, q0)
    if c0 >= -cfg.pen_m:
        return ProjectResult(q0, b0, c0, c0, True, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    k0 = _keypoints(clr, b0, q0)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt, dr = _ball(x[7:10], cfg.max_shift_m), _ball(x[10:13], cfg.max_rot_rad)
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec(dr).as_matrix()
        t[:3, 3] = dt
        # quantised to HandClearance's 1e-9 cache key, so equal inputs give equal outputs whatever was evaluated first
        return np.round(np.clip(q0 + x[:7], lo, hi), 9), b0 @ t

    def parts(x: np.ndarray) -> tuple[float, float]:
        q, b = unpack(x)
        dk = _keypoints(clr, b, q) - k0
        return float(np.sum(dk * dk)) / cfg.k_scale_m ** 2, clr.clearance(b, q)

    bounds = ([(a - c, d - c) for a, c, d in zip(lo, q0, hi, strict=True)]
              + [(-cfg.max_shift_m, cfg.max_shift_m)] * 3 + [(-cfg.max_rot_rad, cfg.max_rot_rad)] * 3)
    # starts: the retarget itself, and backed out along the base's direction away from the object centre by half and
    # all of the shift bound (the natural escape from a deep interpenetration); best = feasible, then least keypoint
    c_base = (np.linalg.inv(b0) @ np.append(np.zeros(3) if centre is None else np.asarray(centre, float), 1.0))[:3]
    u = -c_base / max(float(np.linalg.norm(c_base)), 1e-12)
    starts = [np.zeros(13)] + [np.concatenate([np.zeros(7), f * cfg.max_shift_m * u, np.zeros(3)]) for f in (0.5, 1.0)]
    best: tuple[tuple[bool, float], np.ndarray, int] | None = None
    for x in starts:
        w, done = 1.0, 0
        for done in range(1, cfg.rounds + 1):
            def cost(z: np.ndarray, w: float = w) -> float:
                kc, cl = parts(z)
                aim = cfg.aim_frac * cfg.pen_m
                return kc + w * max(0.0, -cl - aim) ** 2 / aim ** 2

            x = np.asarray(minimize(cost, x, method="Powell", bounds=bounds,
                                    options={"maxfev": cfg.maxfev, "xtol": 1e-5, "ftol": 1e-9}).x, float)
            if parts(x)[1] >= -cfg.pen_m:
                break
            w *= 10.0
        kc, cl = parts(x)
        key = (cl >= -cfg.pen_m, -kc if cl >= -cfg.pen_m else cl)
        if best is None or key > best[0]:
            best = (key, x, done)
    _, x, done = best
    q, b = unpack(x)
    cl = clr.clearance(b, q)
    dk = np.linalg.norm(_keypoints(clr, b, q) - k0, axis=1)
    rel = np.linalg.inv(b0) @ b
    return ProjectResult(q, b, c0, cl, bool(cl >= -cfg.pen_m), float(np.sqrt(np.mean(dk ** 2)) * 1e3),
                         float(dk.max() * 1e3), float(np.linalg.norm(rel[:3, 3])),
                         float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel[:3, :3]).as_rotvec()))),
                         float(np.linalg.norm(q - q0)), done)
