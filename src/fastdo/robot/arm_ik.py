"""Right-arm inverse kinematics of the G1: a wrist SE(3) target -> 7 arm joints, per frame.

The controlled frame is the site `right_dex3_base` (the Dex3 URDF `base_link`, the frame the hand retargeting works
in), so one wrist target serves both the arm IK and the hand retargeting.

Per frame, damped least squares on the SE(3) log-map error (Gauss-Newton with Levenberg damping):
- error   xi = log(T(q)^-1 T*) in the BODY frame (`fastdo.geometry.lie.se3_log`, translation through the inverse left
  Jacobian, so translation and rotation are coupled exactly as on SE(3));
- Jacobian J_b = [R^T J_p; R^T J_r] restricted to the 7 right-arm dofs (MuJoCo `mj_jacSite`), since
  T(q + dq) ~= T(q) exp(J_b dq);
- weights W = diag(1, 1, 1, w_r, w_r, w_r): `rot_weight` w_r = 1/sqrt(50) m/rad, the ratio of xr_teleoperate's
  `G1_29_ArmIK` cost (50 * translation^2 + rotation^2);
- step  (J_b^T W^2 J_b + (lambda^2 + mu) I) dq = J_b^T W^2 xi + mu (q_prev - q): Levenberg damping lambda plus a
  temporal-continuity pull mu toward the previous frame's solution (xr_teleoperate's 0.1 ||q - q_last||^2 analogue);
- joint limits: every iterate is clipped into the command limits shrunk by `limit_margin`;
- warm start from the previous frame's solution (temporal continuity), the first frame from `q_init`;
- joint-speed cap (with `times`): each frame's iterate is also clipped to q_prev +- `max_joint_speed` * dt, so a target
  that crosses a wrist singularity or a joint limit costs a few frames of tracking error instead of a branch flip
  (a smoke-test clip had 2.2 rad single-frame wrist-roll flips, which became a 12x slower export);
- restarts: when the warm-started frame leaves a weighted residual above `restart_tol`, the frame is also solved from
  each of `seeds` and the best result replaces it only if it cuts the weighted residual by `restart_gain` (a single
  outlier target can otherwise pull the warm start into a branch the following frames cannot leave).
- collision avoidance (`IkCollider`, optional): every iteration runs MuJoCo collision detection on a copy of the model
  whose geom margins are `d_safe`; each contact between a geom of the right arm/hand and any other robot geom or the
  table (the object is never in the IK model) with distance d < `d_safe` adds the soft row
  sqrt(w_c) n^T (J_2 - J_1) dq = sqrt(w_c) (d_safe - d), n the contact normal, J_i the point Jacobians of the two
  bodies at the contact point on the 7 arm dofs. Skipped pairs: both inside the hand (their distance does not depend on
  the arm), and ROBOT bodies at most `min_joint_hops` (2) joints apart along the kinematic tree (e.g. torso-shoulder,
  elbow-wrist pitch: by design 8-10 mm apart at every angle, not collisions; with legs and waist welded, MuJoCo's own
  parent-child filter no longer applies to the torso, whose weld parent is the world). The hop filter never applies
  to the table: it hangs off the world body too, 1-2 hops from the shoulder pitch/roll links, and a hop count
  through the world says nothing about the distance to it. A kinematic IK that ignores the
  torso commands the arm through it (in a smoke test the shoulder roll was commanded into the torso and the arm
  stalled 0.5 rad from its target). The fingers are posed at the frame's retargeted hand joints (`hand_q`), not at 0:
  with the hand at 0 the smoke-test IK reported 1 cm clearance on every row while the replay's thumb (at its
  retargeted angle) pressed up to 5.9 mm into the right shoulder links (1,516 physics steps deeper than 1 mm).
A frame whose target is out of reach ends at the damped least-squares compromise; its residual is reported, not hidden.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from fastdo.geometry.lie import se3_log
from fastdo.robot import g1_constants as C
from fastdo.robot.g1_model import G1Upper

__all__ = ["ArmIkConfig", "ArmIkResult", "IkCollider", "body_jacobian", "site_pose", "solve_arm_ik",
           "solve_arm_ik_frame"]


@dataclass(frozen=True)
class ArmIkConfig:
    rot_weight: float = 1.0 / np.sqrt(50.0)
    damping: float = 1e-3
    continuity: float = 1e-4
    limit_margin: float = 0.02
    max_iter: int = 100
    max_step: float = 0.2
    tol: float = 1e-9
    max_joint_speed: float = 3.0
    restart_tol: float = 0.01
    restart_gain: float = 0.8
    seeds: tuple[tuple[float, ...], ...] = ((0.0,) * 7, (-0.5, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0),
                                            (0.3, -0.5, 0.5, 0.5, 0.5, 0.0, 0.0), (-1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class ArmIkResult:
    q: np.ndarray            # (T,7) right-arm joints, `C.RIGHT_ARM_JOINTS` order
    pos_err: np.ndarray      # (T,) metres, |p(q) - p*|
    rot_err: np.ndarray      # (T,) radians, angle of R(q)^T R*
    iters: np.ndarray        # (T,) iterations used
    clearance: np.ndarray    # (T,) min distance (m) of arm geoms to other robot geoms / table, capped at d_safe;
                             #      NaN without a collider


def _joint_hops(model: mujoco.MjModel) -> np.ndarray:
    """`(nbody, nbody)` number of joints on the tree path between two bodies."""
    nb = model.nbody
    chains = []
    for b in range(nb):
        chain, hops, cur = {}, 0, b
        while True:
            chain[cur] = hops
            if cur == 0:
                break
            hops += int(model.body_jntnum[cur])
            cur = int(model.body_parentid[cur])
        chains.append(chain)
    out = np.zeros((nb, nb), dtype=np.int64)
    for a in range(nb):
        for b in range(a + 1, nb):
            common = min((ca + chains[b][k] for k, ca in chains[a].items() if k in chains[b]))
            out[a, b] = out[b, a] = common
    return out


def _subtree(model: mujoco.MjModel, root: int) -> np.ndarray:
    out = np.zeros(model.nbody, dtype=bool)
    out[root] = True
    for b in range(root + 1, model.nbody):
        out[b] = out[model.body_parentid[b]]
    return out


class IkCollider:
    """Soft collision rows for the arm IK (module docstring). `exclude_bodies`: bodies never considered (object)."""

    def __init__(self, g1: G1Upper, d_safe: float = 0.01, weight: float = 10.0,
                 exclude_bodies: tuple[int, ...] = (), min_joint_hops: int = 2) -> None:
        self.g1 = g1
        self.model = copy.deepcopy(g1.model)
        self.model.geom_margin[:] = d_safe
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = g1.default_qpos()
        self.d_safe, self.sqrt_w = d_safe, float(np.sqrt(weight))
        m = self.model
        sid = mujoco.mj_name2id
        arm = _subtree(m, sid(m, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"))
        hand = _subtree(m, sid(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"))
        self.arm_geom = arm[m.geom_bodyid]
        self.hand_geom = hand[m.geom_bodyid]
        self.robot_geom = _subtree(m, sid(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))[m.geom_bodyid]
        skip = np.zeros(m.nbody, dtype=bool)
        skip[list(exclude_bodies)] = True
        self.skip_geom = skip[m.geom_bodyid] | (m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        # Narrow-phase only the pairs the rows can use (a 10x cheaper mj_collision): other robot geoms and the table
        # (bit 1) collide with the arm (bit 2) and the hand (bit 4); the arm also with the hand; nothing else.
        live = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        hand_g, arm_g = self.hand_geom, self.arm_geom & ~self.hand_geom
        m.geom_contype[:] = np.where(hand_g, 4, np.where(arm_g, 2, 1)) * live
        m.geom_conaffinity[:] = np.where(hand_g, 1, np.where(arm_g, 1 | 4, 0)) * live
        self.near = _joint_hops(m) <= min_joint_hops
        self._j1 = np.zeros((3, m.nv))
        self._j2 = np.zeros((3, m.nv))

    def rows(self, q_arm: np.ndarray, q_hand: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """`(A (k,7), b (k,), min distance)` at `q_arm`, with the right hand at `q_hand` (package enum order) when given
        (else at its last value, initially 0)."""
        m, d = self.model, self.data
        d.qpos[self.g1.right_arm_qpos] = q_arm
        if q_hand is not None:
            d.qpos[self.g1.right_hand_qpos] = q_hand
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        a_rows, b_rows, dmin = [], [], self.d_safe
        cols = self.g1.right_arm_dof
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if self.skip_geom[g1] or self.skip_geom[g2] or not (self.arm_geom[g1] or self.arm_geom[g2]):
                continue
            if self.hand_geom[g1] and self.hand_geom[g2]:
                continue
            if self.robot_geom[g1] and self.robot_geom[g2] and self.near[m.geom_bodyid[g1], m.geom_bodyid[g2]]:
                continue                                   # adjacent robot links; the table is never skipped
            dist = float(c.dist)
            if dist >= self.d_safe:
                continue
            dmin = min(dmin, dist)
            mujoco.mj_jac(m, d, self._j1, None, c.pos, int(m.geom_bodyid[g1]))
            mujoco.mj_jac(m, d, self._j2, None, c.pos, int(m.geom_bodyid[g2]))
            n = c.frame[:3]
            a_rows.append(self.sqrt_w * (n @ (self._j2[:, cols] - self._j1[:, cols])))
            b_rows.append(self.sqrt_w * (self.d_safe - dist))
        if not a_rows:
            return np.zeros((0, 7)), np.zeros(0), dmin
        return np.asarray(a_rows), np.asarray(b_rows), dmin


class _ArmFK:
    def __init__(self, g1: G1Upper) -> None:
        self.g1 = g1
        self.data = mujoco.MjData(g1.model)
        self.data.qpos[:] = g1.default_qpos()
        self._jp = np.zeros((3, g1.model.nv))
        self._jr = np.zeros((3, g1.model.nv))

    def set(self, q_arm: np.ndarray, q_hand: np.ndarray | None = None) -> None:
        self.data.qpos[self.g1.right_arm_qpos] = q_arm
        if q_hand is not None:
            self.data.qpos[self.g1.right_hand_qpos] = q_hand
        mujoco.mj_kinematics(self.g1.model, self.data)

    def pose(self) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = self.data.site_xmat[self.g1.site_base].reshape(3, 3)
        out[:3, 3] = self.data.site_xpos[self.g1.site_base]
        return out

    def body_jac(self) -> np.ndarray:
        mujoco.mj_comPos(self.g1.model, self.data)
        mujoco.mj_jacSite(self.g1.model, self.data, self._jp, self._jr, self.g1.site_base)
        rot = self.data.site_xmat[self.g1.site_base].reshape(3, 3)
        cols = self.g1.right_arm_dof
        return np.vstack([rot.T @ self._jp[:, cols], rot.T @ self._jr[:, cols]])


def site_pose(g1: G1Upper, q_arm: np.ndarray) -> np.ndarray:
    """`(T,4,4)` world poses of `right_dex3_base` for `(T,7)` right-arm joints (other joints at `default_qpos`)."""
    fk = _ArmFK(g1)
    out = []
    for q in np.atleast_2d(q_arm):
        fk.set(q)
        out.append(fk.pose())
    return np.stack(out)


def body_jacobian(g1: G1Upper, q_arm: np.ndarray) -> np.ndarray:
    """`(6,7)` body Jacobian [v; w] of `right_dex3_base` at `q_arm`."""
    fk = _ArmFK(g1)
    fk.set(q_arm)
    return fk.body_jac()


def _weighted_residual(fk: _ArmFK, q: np.ndarray, target: np.ndarray, rot_weight: float,
                       collider: IkCollider | None = None, q_hand: np.ndarray | None = None) -> float:
    fk.set(q)
    xi = _log_err(fk.pose(), target)
    sq = np.sum(xi[:3] ** 2) + rot_weight ** 2 * np.sum(xi[3:] ** 2)
    if collider is not None:
        sq += float(np.sum(collider.rows(q, q_hand)[1] ** 2))
    return float(np.sqrt(sq))


def _log_err(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    rel = np.linalg.solve(current, target)
    return se3_log(torch.from_numpy(rel)).numpy()


def solve_arm_ik_frame(fk: _ArmFK, target: np.ndarray, q0: np.ndarray, q_prev: np.ndarray, lo: np.ndarray,
                       hi: np.ndarray, cfg: ArmIkConfig, collider: IkCollider | None = None,
                       q_hand: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """One frame of the damped least-squares solve (module docstring). Returns `(q, iterations)`. `q_hand`: the
    frame's right-hand joints for the collision rows."""
    w2 = np.array([1.0, 1.0, 1.0] + [cfg.rot_weight ** 2] * 3)
    q = np.clip(q0, lo, hi)
    reg = (cfg.damping ** 2 + cfg.continuity) * np.eye(7)
    it = 0
    for it in range(1, cfg.max_iter + 1):
        fk.set(q)
        xi = _log_err(fk.pose(), target)
        jac = fk.body_jac()
        lhs = jac.T @ (w2[:, None] * jac) + reg
        rhs = jac.T @ (w2 * xi) + cfg.continuity * (q_prev - q)
        if collider is not None:
            a_c, b_c, _ = collider.rows(q, q_hand)
            lhs += a_c.T @ a_c
            rhs += a_c.T @ b_c
        dq = np.linalg.solve(lhs, rhs)
        norm = float(np.abs(dq).max())
        if norm > cfg.max_step:
            dq *= cfg.max_step / norm
        q_new = np.clip(q + dq, lo, hi)
        if float(np.abs(q_new - q).max()) < cfg.tol:
            q = q_new
            break
        q = q_new
    return q, it


def solve_arm_ik(g1: G1Upper, targets: np.ndarray, q_init: np.ndarray | None = None,
                 cfg: ArmIkConfig | None = None, valid: np.ndarray | None = None,
                 collider: IkCollider | None = None, times: np.ndarray | None = None,
                 hand_q: np.ndarray | None = None) -> ArmIkResult:
    """`(T,4,4)` world targets of `right_dex3_base` -> `ArmIkResult`. Invalid rows hold the previous solution.

    With a `collider` (built on `g1`'s model) every iteration also gets the collision rows, with the fingers at
    `hand_q` `(T,7)` (package enum order; None: the collider's current hand pose, initially 0); with `times` `(T,)` (s)
    the joint-speed cap applies (module docstring).
    """
    cfg = ArmIkConfig() if cfg is None else cfg
    targets = np.asarray(targets, dtype=np.float64)
    t = targets.shape[0]
    lim = np.asarray(C.RIGHT_ARM_COMMAND_LIMITS, dtype=np.float64)
    lo, hi = lim[:, 0] + cfg.limit_margin, lim[:, 1] - cfg.limit_margin
    mask = np.ones(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    fk = _ArmFK(g1)
    q_prev = np.clip(np.zeros(7) if q_init is None else np.asarray(q_init, dtype=np.float64), lo, hi)
    qs = np.zeros((t, 7))
    iters = np.zeros(t, dtype=np.int64)
    pos_err = np.full(t, np.nan)
    rot_err = np.full(t, np.nan)
    clearance = np.full(t, np.nan)
    last_t: float | None = None
    hands = None if hand_q is None else np.asarray(hand_q, dtype=np.float64)
    if hands is not None and hands.shape != (t, 7):
        raise ValueError(f"hand_q must be ({t}, 7), got {hands.shape}")
    for k in range(t):
        qh = None if hands is None else hands[k]
        if mask[k]:
            lo_k, hi_k = lo, hi
            if times is not None and last_t is not None:
                cap = cfg.max_joint_speed * float(times[k] - last_t)
                lo_k, hi_k = np.maximum(lo, q_prev - cap), np.minimum(hi, q_prev + cap)
            q_warm, iters[k] = solve_arm_ik_frame(fk, targets[k], q_prev, q_prev, lo_k, hi_k, cfg, collider, qh)
            res = _weighted_residual(fk, q_warm, targets[k], cfg.rot_weight, collider, qh)
            if res > cfg.restart_tol:
                for seed in cfg.seeds:
                    q_s, it_s = solve_arm_ik_frame(fk, targets[k], np.clip(seed, lo_k, hi_k), q_prev, lo_k, hi_k, cfg,
                                                   collider, qh)
                    iters[k] += it_s
                    res_s = _weighted_residual(fk, q_s, targets[k], cfg.rot_weight, collider, qh)
                    if res_s < cfg.restart_gain * res:
                        q_warm, res = q_s, res_s
            last_t = None if times is None else float(times[k])
            q_prev = q_warm
            fk.set(q_prev)
            pose = fk.pose()
            pos_err[k] = float(np.linalg.norm(pose[:3, 3] - targets[k, :3, 3]))
            cos = (np.trace(pose[:3, :3].T @ targets[k, :3, :3]) - 1.0) / 2.0
            rot_err[k] = float(np.arccos(np.clip(cos, -1.0, 1.0)))
            if collider is not None:
                clearance[k] = collider.rows(q_prev, qh)[2]
        qs[k] = q_prev
    return ArmIkResult(q=qs, pos_err=pos_err, rot_err=rot_err, iters=iters, clearance=clearance)
