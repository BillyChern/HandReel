"""Opposition-seeking grasp refinement: a kinematic (simulation-free) baseline.

No retargeted full-bias contact set of the S3 clips was found in force closure (epsilon 0) or able to hold its
weight. This module refines ONE grasp (the Dex3 hand pose and the Dex3 base pose in the object frame at the
contact phase) to maximise the Ferrari-Canny epsilon (`fastdo.robot.force_closure`), staying near the human grasp.

Contact model (kinematic, in the object frame; no simulation): every collision geom of the hand
(`approach.HandClearance.hand_geoms`) within `contact_tol_m` of the object is a contact at the closest OBJECT-surface
point, with the normal from the hand into the object (`physics.grasp_preload.find_contacts`, `mj_geomDistance`).

Variables x = [dq (7), dt (3), dr (3)]: hand joint offset; base translation (m) and rotation vector (rad) applied in
the Dex3 BASE frame, base_obj(x) = base_obj_ref @ [R(dr) | dt]. Bounds: the hand stays inside the command limits minus
`limit_margin_rad`; |dt_k| <= `max_shift_m` and |dr_k| <= `max_rot_rad` per component (the 15 mm / 10 deg
bound is enforced on the norms after the solve, refusing otherwise). Cost (minimised, bounded Powell):

    -Q + w_reach * sum_f max(0, gap_f - tol)^2 / tol^2        (each finger's distal link onto the surface)
             + w_pen * max(0, -min_gap - pen_m)^2 / pen_m^2          (no penetration beyond the grasp-closure target)
             + w_dev * (||dq||^2 / q_scale^2 + ||dt||^2 / max_shift^2 + ||dr||^2 / max_rot^2)

Q is `force_closure.signed_quality`: epsilon in force closure, minus the wrench
hull's distance from the origin otherwise, so the cost is not flat when every finger touches without closure. The
result is ACCEPTED only if it raises epsilon above the start's and respects every bound; otherwise the start is
returned unchanged with `improved = False` (the refinement never makes a grasp worse). The deviation from the human
grasp is always reported (a refinement far from the human grasp is no longer imitation).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from fastdo.physics.grasp_preload import find_contacts
from fastdo.robot import g1_constants as C
from fastdo.robot.approach import HandClearance
from fastdo.robot.force_closure import contact_wrenches, epsilon_metric, signed_quality
from fastdo.robot.grasp_closure import CLOSURE_TARGET_M, FINGERS

__all__ = ["GraspContactModel", "RefineConfig", "RefineResult", "refine_grasp"]


@dataclass(frozen=True)
class RefineConfig:
    mu: float = 0.5                    # effective friction mu' = MuJoCo friction 1.0 / s_mu 2.0
    contact_tol_m: float = 0.002       # a hand geom this close to the object is a contact
    pen_m: float = CLOSURE_TARGET_M    # allowed penetration (the frozen grasp-closure target depth)
    max_shift_m: float = 0.015
    max_rot_rad: float = float(np.radians(10.0))
    limit_margin_rad: float = 0.05
    q_scale_rad: float = 0.5
    w_reach: float = 0.05
    w_pen: float = 100.0               # a steep barrier: acceptance requires min_gap >= -pen_m (set on the KA scene)
    w_dev: float = 0.002
    maxfev: int = 4000


@dataclass(frozen=True)
class RefineResult:
    hand_q: np.ndarray
    base_obj: np.ndarray
    epsilon_start: float
    epsilon: float
    improved: bool
    dq_norm_rad: float
    shift_m: float
    rot_deg: float
    contacts: int
    fingers: tuple[str, ...]
    reason: str


class GraspContactModel:
    """Kinematic contacts and epsilon of the Dex3 hand at (base pose in the object frame, hand pose)."""

    def __init__(self, clearance: HandClearance, com: np.ndarray, char_len: float, cfg: RefineConfig) -> None:
        self.clr, self.com, self.L, self.cfg = clearance, np.asarray(com, float), float(char_len), cfg
        m = clearance.m
        self.obj_geoms = np.zeros(m.ngeom, dtype=bool)
        self.obj_geoms[clearance.object_geom] = True
        self.hand_mask = np.zeros(m.ngeom, dtype=bool)
        self.hand_mask[clearance.hand_geoms] = True
        names = {link: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link) for _, _, link in FINGERS}
        self.finger_geoms = {f: [int(g) for g in clearance.hand_geoms if m.geom_bodyid[g] == names[link]]
                             for f, _, link in FINGERS}

    def _place(self, base_obj: np.ndarray, hand_q: np.ndarray) -> None:
        clr, m, d = self.clr, self.clr.m, self.clr.d
        rel = clr.geoms_in_base(hand_q)
        d.qpos[:] = clr.scene.g1.default_qpos()
        oq = clr.scene.object_qpos
        d.qpos[oq:oq + 7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_kinematics(m, d)
        poses = np.einsum("ij,gjk->gik", np.asarray(base_obj, float), rel)
        for k, g in enumerate(clr.hand_geoms):
            d.geom_xpos[g] = poses[k, :3, 3]
            d.geom_xmat[g] = poses[k, :3, :3].ravel()

    def measure(self, base_obj: np.ndarray, hand_q: np.ndarray) -> dict:
        """{epsilon, min_gap, finger_gap {finger: gap}, contacts, fingers}: all in the object frame."""
        self._place(base_obj, hand_q)
        cs = find_contacts(self.clr.m, self.clr.d, self.hand_mask, self.obj_geoms, self.clr.distmax)
        gaps = {int(c.hand_geom): c.gap for c in cs}
        near = [c for c in cs if c.gap <= self.cfg.contact_tol_m]
        eps, quality = 0.0, -1.0
        if near:
            w = contact_wrenches(np.array([c.p_obj for c in near]), np.array([c.normal for c in near]), self.com,
                                 self.cfg.mu, self.L)
            eps, quality = epsilon_metric(w), signed_quality(w)
        fgap = {f: min((gaps.get(g, self.clr.distmax) for g in gs), default=self.clr.distmax)
                for f, gs in self.finger_geoms.items()}
        touching = tuple(f for f, gp in fgap.items() if gp <= self.cfg.contact_tol_m)
        return {"epsilon": eps, "quality": quality, "min_gap": min(gaps.values(), default=self.clr.distmax), "finger_gap": fgap,
                "contacts": len(near), "fingers": touching}


def _base(base_obj_ref: np.ndarray, dt: np.ndarray, dr: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(dr).as_matrix()
    t[:3, 3] = dt
    return np.asarray(base_obj_ref, float) @ t


def refine_grasp(model: GraspContactModel, base_obj_ref: np.ndarray, hand_ref: np.ndarray,
                 cfg: RefineConfig | None = None) -> RefineResult:
    """Refine one grasp (module docstring). Never returns a grasp with a lower epsilon than the start."""
    cfg = model.cfg if cfg is None else cfg
    lim = np.asarray(C.RIGHT_HAND_COMMAND_LIMITS, float)
    lo, hi = lim[:, 0] + cfg.limit_margin_rad, lim[:, 1] - cfg.limit_margin_rad
    q0 = np.clip(np.asarray(hand_ref, float), lo, hi)
    start = model.measure(base_obj_ref, q0)

    def ball(v: np.ndarray, r: float) -> np.ndarray:          # the norm bound, as a projection (always feasible)
        n = float(np.linalg.norm(v))
        return v if n <= r else v * (r / n)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return q0 + x[:7], ball(x[7:10], cfg.max_shift_m), ball(x[10:13], cfg.max_rot_rad)

    def cost(x: np.ndarray) -> float:
        q, dt, dr = unpack(x)
        s = model.measure(_base(base_obj_ref, dt, dr), q)
        tol = cfg.contact_tol_m
        reach = sum(max(0.0, g - tol) ** 2 for g in s["finger_gap"].values()) / tol ** 2
        pen = max(0.0, -s["min_gap"] - cfg.pen_m) ** 2 / cfg.pen_m ** 2
        dev = (float(x[:7] @ x[:7]) / cfg.q_scale_rad ** 2 + float(dt @ dt) / cfg.max_shift_m ** 2
               + float(dr @ dr) / cfg.max_rot_rad ** 2)
        return -s["quality"] + cfg.w_reach * reach + cfg.w_pen * pen + cfg.w_dev * dev

    bounds = ([(a - b, c - b) for a, b, c in zip(lo, q0, hi, strict=True)]
              + [(-cfg.max_shift_m, cfg.max_shift_m)] * 3 + [(-cfg.max_rot_rad, cfg.max_rot_rad)] * 3)
    sol = minimize(cost, np.zeros(13), method="Powell", bounds=bounds,
                   options={"maxfev": cfg.maxfev, "xtol": 1e-4, "ftol": 1e-7})
    q, dt, dr = unpack(np.asarray(sol.x, float))
    q = np.clip(q, lo, hi)
    base = _base(base_obj_ref, dt, dr)
    end = model.measure(base, q)
    shift, rot = float(np.linalg.norm(dt)), float(np.degrees(np.linalg.norm(dr)))
    ok_bounds = shift <= cfg.max_shift_m + 1e-9 and rot <= np.degrees(cfg.max_rot_rad) + 1e-9
    ok_pen = end["min_gap"] >= -cfg.pen_m - 1e-6
    improved = bool(end["epsilon"] > start["epsilon"] and ok_bounds and ok_pen)
    if not improved:
        why = ("no epsilon gain" if end["epsilon"] <= start["epsilon"]
               else "wrist offset beyond the 15 mm / 10 deg bound" if not ok_bounds else "penetration beyond pen_m")
        return RefineResult(q0, np.asarray(base_obj_ref, float), start["epsilon"], start["epsilon"], False, 0.0, 0.0,
                            0.0, start["contacts"], start["fingers"], why)
    return RefineResult(q, base, start["epsilon"], end["epsilon"], True, float(np.linalg.norm(q - q0)), shift, rot,
                        end["contacts"], end["fingers"], "epsilon raised")
