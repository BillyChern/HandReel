"""Synthesised pre-grasp approach for the kinematic robot path.

DexYCB lab clips start with the hand 5-9 rows (0.5-0.9 s) from the grasp. The vertical 8-cm approach of
`fastdo.robot.reach.approach_retreat` then starts with the Dex3 fingers inside the object: on the two provisional
lab clips that passed S1 and P14, the trajectory start was 29.1 / 37.9 mm inside the object (after the placement
envelope), so no home could be checked. This module replaces the rows BEFORE the first contact row c0:

- **Direction:** the mean, over the contact rows, of the unit vector from the palm (the Dex3 `base_link` origin, where
  the Dex3 URDF puts `right_hand_palm_link`) to the object's hull centroid, in the OBJECT frame (so it is placement
  invariant). The hand retreats along its negative: away from the object, the way the palm faces it.
- **Distance:** the smallest `d` on a `step_m` grid from `clear_m` up to `max_m` such that (i) at the approach start
  the robot hand (every collision geom of the right-wrist subtree, at the start hand pose) clears the object by at
  least `clear_m` + `envelope_m` (the placement envelope the transfer planner subtracts), and (ii) every
  sample of the approach strictly before c0 (each row and `substeps` interpolated poses between rows) has a
  positive clearance: NO penetration before contact.
- **Path:** wrist translation `d * f(t)` along the retreat direction, `f` a smoothstep from 1 at the first task row
  to 0 at c0 (orientation held at the c0 pose). Two finger schedules are tried at each distance, in this order:
  "grasp" (the c0 hand pose throughout: a pure translation) and "open" (the flexion joints (`FLEXION` signs) at their
  extension command limit at f = 1, back to the c0 hand pose with the same `f`). The first (distance, schedule) that
  passes is taken. A wrapped grasp (the lab clips) needs "open"; a fingertip touch from above (the synthetic box)
  needs "grasp", because fingers closing with the wrist sweep through the object. The approach spans at least
  `min_s` (the caller pads synthetic rows when the clip has fewer).
- **Contact pose that penetrates:** the human contact row retargeted to Dex3 can already penetrate the object (the
  synthetic box: 19 mm, the fingertip sites sit inside the pads). Then the approach arrives one row EARLY (row c0-1)
  at a PRE-CONTACT pose: the c0 pose backed out along the retreat direction by the smallest `e` (1 mm steps, at most
  `max_backout_m`) at which it clears by `pre_clear_m`; row c0 keeps the human contact pose. So no row before c0
  penetrates; the move into the (penetrating) grasp happens between rows c0-1 and c0.
- **Collision check:** MuJoCo `mj_geomDistance` (convex hulls, as the replay collides them) between every right-hand
  geom and the object mesh, in the object frame.

`plan_approach` returns an `ApproachPlan` with `ok` False and the reason when no distance passes; the caller then
refuses the clip (nothing is weakened: the existing checks run on the planned rows unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from fastdo.robot import g1_constants as C
from fastdo.robot.replay import SceneConfig, build_replay_scene

__all__ = ["FLEXION", "ApproachConfig", "ApproachError", "ApproachPlan", "HandClearance", "open_hand",
           "plan_approach", "smoothstep"]

# closing direction of each right Dex3 joint, hardware order (thumb_0 abducts: not a flexion joint). Measured with
# `dex3_tip_positions`: +0.2 rad on index_0/1, middle_0/1 and -0.2 rad on thumb_1/2 bring the thumb tip and the
# index/middle tips together.
FLEXION: tuple[int, ...] = (0, -1, -1, 1, 1, 1, 1)


class ApproachError(ValueError):
    """No synthesised approach passes the collision check: the clip is refused."""


@dataclass(frozen=True)
class ApproachConfig:
    clear_m: float = 0.010          # start clearance beyond the placement envelope (>= 10 mm)
    envelope_m: float = 0.0         # the placement envelope the transfer planner subtracts (set by the caller)
    min_s: float = 0.5              # the approach interpolates into contact over at least this long
    step_m: float = 0.005
    max_m: float = 0.25
    substeps: int = 4               # interpolated poses checked between consecutive rows
    distmax: float = 0.30
    pre_clear_m: float = 0.001      # clearance of the pre-contact pose when the contact pose penetrates
    max_backout_m: float = 0.06


@dataclass(frozen=True)
class ApproachPlan:
    ok: bool
    distance_m: float | None
    direction_obj: np.ndarray                  # (3,) unit retreat direction, object frame
    start_clearance_m: float | None
    min_clearance_before_contact_m: float | None
    penetrating_samples: int | None            # samples before c0 with clearance <= 0 at the chosen distance
    base_obj: np.ndarray = field(repr=False)   # (A+1,4,4) planned Dex3 base poses in the object frame, rows 0..A
    hand_q: np.ndarray = field(repr=False)     # (A+1,7) planned hand joints (A = c0, or c0-1 with a back-out)
    reason: str | None = None
    schedule: str | None = None                # "grasp" or "open" (module docstring, "Path")
    contact_clearance_m: float | None = None   # clearance of the c0 (human contact) pose itself
    backout_m: float = 0.0                     # pre-contact back-out (module docstring)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "distance_m": self.distance_m, "schedule": self.schedule,
                "contact_clearance_m": self.contact_clearance_m, "backout_m": self.backout_m,
                "arrival_row": None if self.base_obj is None else len(self.base_obj) - 1, "direction_obj": np.round(self.direction_obj, 4).tolist(),
                "start_clearance_m": self.start_clearance_m,
                "min_clearance_before_contact_m": self.min_clearance_before_contact_m,
                "penetrating_samples": self.penetrating_samples, "reason": self.reason}


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def open_hand(q: np.ndarray) -> np.ndarray:
    """The hand pose with every flexion joint at its extension command limit (thumb_0 kept)."""
    lim = np.asarray(C.RIGHT_HAND_COMMAND_LIMITS, dtype=np.float64)
    out = np.array(q, dtype=np.float64, copy=True)
    for j, s in enumerate(FLEXION):
        if s > 0:
            out[..., j] = lim[j, 0]
        elif s < 0:
            out[..., j] = lim[j, 1]
    return out


class HandClearance:
    """Signed distance (m, negative = penetration) between the right Dex3 hand at a hand pose and the object, with
    the Dex3 base at a pose in the OBJECT frame (module docstring, "Collision check")."""

    def __init__(self, mesh_vertices: np.ndarray, mesh_faces: np.ndarray, scale: float, root: str | None = None,
                 distmax: float = 0.30) -> None:
        self.scene = build_replay_scene(mesh_vertices, mesh_faces, scale, np.eye(4), SceneConfig(), root=root)
        m = self.scene.g1.model
        self.m, self.d, self.distmax = m, mujoco.MjData(m), distmax
        wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        sub = np.zeros(m.nbody, dtype=bool)
        sub[wrist] = True
        for b in range(wrist + 1, m.nbody):
            sub[b] = sub[m.body_parentid[b]]
        coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.hand_geoms = np.flatnonzero(sub[m.geom_bodyid] & coll & self.scene.robot_geoms)
        self.object_geom = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"))
        self._cache: dict[tuple, np.ndarray] = {}

    def geoms_in_base(self, hand_q: np.ndarray) -> np.ndarray:
        """`(G,4,4)` poses of the hand geoms in the Dex3 base frame at `hand_q` (rigid in the arm joints)."""
        key = tuple(np.round(np.asarray(hand_q, dtype=np.float64), 9))
        if key in self._cache:
            return self._cache[key]
        g1, m, d = self.scene.g1, self.m, self.d
        d.qpos[:] = g1.default_qpos()
        d.qpos[g1.right_hand_qpos] = hand_q
        mujoco.mj_kinematics(m, d)
        ws = np.eye(4)
        ws[:3, :3] = d.site_xmat[g1.site_base].reshape(3, 3)
        ws[:3, 3] = d.site_xpos[g1.site_base]
        inv = np.linalg.inv(ws)
        out = np.tile(np.eye(4), (len(self.hand_geoms), 1, 1))
        for k, g in enumerate(self.hand_geoms):
            out[k, :3, :3] = d.geom_xmat[g].reshape(3, 3)
            out[k, :3, 3] = d.geom_xpos[g]
        out = np.einsum("ij,gjk->gik", inv, out)
        if len(self._cache) < 4096:
            self._cache[key] = out
        return out

    def clearance(self, base_obj: np.ndarray, hand_q: np.ndarray) -> float:
        rel = self.geoms_in_base(hand_q)
        m, d = self.m, self.d
        d.qpos[:] = self.scene.g1.default_qpos()
        oq = self.scene.object_qpos
        d.qpos[oq:oq + 7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_kinematics(m, d)
        poses = np.einsum("ij,gjk->gik", np.asarray(base_obj, dtype=np.float64), rel)
        best = self.distmax
        for k, g in enumerate(self.hand_geoms):
            d.geom_xpos[g] = poses[k, :3, 3]
            d.geom_xmat[g] = poses[k, :3, :3].ravel()
            best = min(best, float(mujoco.mj_geomDistance(m, d, int(g), self.object_geom, self.distmax, None)))
        return best


def retreat_direction(base_obj_contact: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Unit retreat direction (object frame): minus the mean palm-to-centroid unit vector over the contact rows."""
    v = np.asarray(centroid, dtype=np.float64)[None] - np.asarray(base_obj_contact)[:, :3, 3]
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    mean = v.mean(axis=0)
    n = float(np.linalg.norm(mean))
    if n < 1e-6:
        raise ApproachError("the palm-to-object directions over the contact rows cancel: no approach direction")
    return -mean / n


def _path(times: np.ndarray, c0: int, base_c0: np.ndarray, hand_c0: np.ndarray, u: np.ndarray, dist: float,
          substeps: int, schedule: str = "open") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(row base poses (c0+1,4,4), row hands (c0+1,7), sample bases (S,4,4), sample hands (S,7))`; the samples are
    the rows before c0 and `substeps` interpolations after each, all strictly before c0."""
    t = np.asarray(times[: c0 + 1], dtype=np.float64)
    span = max(float(t[c0] - t[0]), 1e-9)
    opened = open_hand(hand_c0) if schedule == "open" else np.asarray(hand_c0, dtype=np.float64)

    def at(tt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = smoothstep((t[c0] - tt) / span)
        base = np.repeat(base_c0[None], len(tt), axis=0)
        base[:, :3, 3] = base_c0[:3, 3] + dist * f[:, None] * u
        hand = hand_c0 + f[:, None] * (opened - hand_c0)
        return base, hand

    rows_b, rows_h = at(t)
    if c0 == 0:
        return rows_b, rows_h, rows_b[:0], rows_h[:0]
    frac = np.arange(substeps + 1) / (substeps + 1)
    ts = (t[:-1, None] + frac[None] * np.diff(t)[:, None]).ravel()
    sb, sh = at(ts)
    return rows_b, rows_h, sb, sh


def plan_approach(times: np.ndarray, c0: int, base_obj: np.ndarray, hand_q: np.ndarray, contact_rows: np.ndarray,
                  centroid: np.ndarray, clearance: HandClearance, cfg: ApproachConfig | None = None) -> ApproachPlan:
    """The synthesised approach onto task row `c0` (module docstring). `base_obj` `(T,4,4)`: the Dex3 base (wrist
    target) in the object frame per task row; `contact_rows`: task rows in contact (their palm directions)."""
    cfg = ApproachConfig() if cfg is None else cfg
    t = np.asarray(times, dtype=np.float64)
    if float(t[c0] - t[0]) < cfg.min_s - 1e-9:
        raise ApproachError(f"the approach spans {t[c0] - t[0]:.2f} s < {cfg.min_s} s: pad the rows first")
    u = retreat_direction(np.asarray(base_obj)[np.asarray(contact_rows)], centroid)
    base_c0, hand_c0 = np.asarray(base_obj[c0], dtype=np.float64), np.asarray(hand_q[c0], dtype=np.float64)
    contact_cl = clearance.clearance(base_c0, hand_c0)
    arrive, backout, target = c0, 0.0, base_c0
    common = {"direction_obj": u, "contact_clearance_m": contact_cl}
    if contact_cl <= 0.0:
        if c0 < 1:
            raise ApproachError("the contact pose penetrates and there is no row before it to arrive on")
        for e in np.arange(0.001, cfg.max_backout_m + 1e-9, 0.001):
            cand = base_c0.copy()
            cand[:3, 3] = base_c0[:3, 3] + e * u
            if clearance.clearance(cand, hand_c0) >= cfg.pre_clear_m:
                arrive, backout, target = c0 - 1, float(e), cand
                break
        else:
            return ApproachPlan(ok=False, distance_m=None, start_clearance_m=None,
                                min_clearance_before_contact_m=None, penetrating_samples=None,
                                base_obj=np.asarray(base_obj[: c0 + 1]), hand_q=np.asarray(hand_q[: c0 + 1]),
                                reason=f"the contact pose penetrates by {-contact_cl * 1000:.1f} mm and backing it out "
                                       f"{cfg.max_backout_m * 100:.0f} cm along the retreat does not clear it",
                                **common)
    common["backout_m"] = backout
    need = cfg.clear_m + cfg.envelope_m
    best = None
    for dist in np.arange(cfg.clear_m, cfg.max_m + 1e-9, cfg.step_m):
        for schedule in ("grasp", "open"):
            rb, rh, sb, sh = _path(t, arrive, target, hand_c0, u, float(dist), cfg.substeps, schedule)
            if arrive < c0:                                   # the arrival row itself is before contact too
                sb, sh = np.concatenate([sb, rb[-1:]]), np.concatenate([sh, rh[-1:]])
            start = clearance.clearance(rb[0], rh[0])
            if start < need:
                continue
            cl = np.array([clearance.clearance(b, h) for b, h in zip(sb, sh, strict=True)])
            pen = int((cl <= 0.0).sum())
            best = (float(dist), start, float(cl.min()) if cl.size else start, pen, rb, rh)
            if pen == 0:
                return ApproachPlan(ok=True, distance_m=float(dist), start_clearance_m=start,
                                    min_clearance_before_contact_m=best[2], penetrating_samples=0, base_obj=rb,
                                    hand_q=rh, schedule=schedule, **common)
    if best is None:
        return ApproachPlan(ok=False, distance_m=None, start_clearance_m=None,
                            min_clearance_before_contact_m=None, penetrating_samples=None,
                            base_obj=np.asarray(base_obj[: c0 + 1]), hand_q=np.asarray(hand_q[: c0 + 1]),
                            reason=f"no retreat up to {cfg.max_m * 100:.0f} cm clears the object by "
                                   f"{need * 1000:.1f} mm at the start", **common)
    return ApproachPlan(ok=False, distance_m=best[0], start_clearance_m=best[1],
                        min_clearance_before_contact_m=best[2], penetrating_samples=best[3], base_obj=best[4],
                        hand_q=best[5], reason=f"every clearing retreat up to {cfg.max_m * 100:.0f} cm penetrates the "
                                               "object before contact", **common)
