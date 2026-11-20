"""C2 tier-0 controller terms, computed on CPU from the scene (MuJoCo) and DO AS I DO's kinematic reference.

- Gravity feedforward: the wrist-z position servo (kp_z) sags by the supported weight / kp_z; command it back.
- Squeeze: at each reference frame, hand links (explicit contact <pair>s with the object) within `contact_dist_m`
  give contact points, normals (hand -> object) and gaps; a grasp-force NNLS over linearised friction-pyramid edges
  balances the object's reference wrench m(a_ref - g); each contact gets a commanded move along its normal (gap +
  penetration), and damped least squares maps the moves onto the finger joints. The penetration part is returned
  per finger, so the tier-0b search can scale fingers linearly.

Single-hand scenes (`right` / `left`): actuators 0-5 are the wrist (pos x/y/z slides, rot x/y/z), 6..nu-1 the
fingers, the object is a free joint at the end of qpos. Pure numpy + mujoco (no torch, no warp).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import nnls

from fastdo.physics.planner_config import SqueezeConfig

__all__ = [
    "Contact",
    "FINGERS",
    "HandInfo",
    "contacts_at",
    "finger_of",
    "frames_to_steps",
    "grasp_force_nnls",
    "hand_info",
    "squeeze_frame",
    "squeeze_trajectory",
]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GRAVITY = 9.81


def finger_of(body_name: str) -> int | None:
    """Finger index of a link body name (`right_index_MP` -> 1); None for the palm / wrist / anything else."""
    low = body_name.lower()
    for i, f in enumerate(FINGERS):
        if f in low:
            return i
    return None


def _subtree(model: mujoco.MjModel, root: int) -> list[int]:
    out = []
    for b in range(model.nbody):
        p = b
        while p != 0 and p != root:
            p = int(model.body_parentid[p])
        if p == root:
            out.append(b)
    return out


@dataclass(frozen=True)
class HandInfo:
    side: str
    nu: int
    finger_ctrl: np.ndarray      # actuator indices of the finger joints
    finger_dof: np.ndarray       # their dof indices (columns of the Jacobian)
    z_ctrl: int                  # actuator of the wrist z slide
    kp_z: float
    m_hand: float                # mass carried by the z slide (gravcomp-weighted)
    m_obj: float
    obj_body: int
    pairs: tuple[tuple[int, int], ...]   # (hand geom, object geom)
    hand_geom_finger: dict[int, int | None]


def hand_info(model: mujoco.MjModel, side: str) -> HandInfo:
    jz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_pos_z")
    if jz < 0 or model.jnt_type[jz] != mujoco.mjtJoint.mjJNT_SLIDE or not np.allclose(model.jnt_axis[jz], [0, 0, 1]):
        raise ValueError(f"{side}_pos_z must be a world-z slide joint")
    acts = [a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == jz]
    if len(acts) != 1:
        raise ValueError("expected one actuator on the wrist z slide")
    z_ctrl = acts[0]
    sub = _subtree(model, int(model.jnt_bodyid[jz]))
    m_hand = float(sum(model.body_mass[b] * (1.0 - model.body_gravcomp[b]) for b in sub))
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    if obj_body < 0:
        raise ValueError(f"no {side}_object body")
    hand_set = set(sub)
    pairs, hgf = [], {}
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if b1 in hand_set and b2 == obj_body:
            pairs.append((g1, g2))
        elif b2 in hand_set and b1 == obj_body:
            pairs.append((g2, g1))
    for g, _ in pairs:
        hgf[g] = finger_of(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "")
    finger_ctrl = np.array([a for a in range(model.nu) if a >= 6], dtype=int)
    finger_dof = np.array([int(model.jnt_dofadr[int(model.actuator_trnid[a, 0])]) for a in finger_ctrl], dtype=int)
    return HandInfo(side=side, nu=int(model.nu), finger_ctrl=finger_ctrl, finger_dof=finger_dof, z_ctrl=z_ctrl,
                    kp_z=float(model.actuator_gainprm[z_ctrl, 0]), m_hand=m_hand,
                    m_obj=float(model.body_mass[obj_body]), obj_body=obj_body, pairs=tuple(pairs),
                    hand_geom_finger=hgf)


@dataclass(frozen=True)
class Contact:
    hand_geom: int
    finger: int
    p_hand: np.ndarray
    p_obj: np.ndarray
    normal: np.ndarray           # unit, from the hand into the object
    gap: float                   # signed distance (negative = penetrating)


def contacts_at(model: mujoco.MjModel, data: mujoco.MjData, info: HandInfo, dist_max: float) -> list[Contact]:
    """Per finger-link geom, its closest object geom if within `dist_max` (palm/wrist links excluded)."""
    best: dict[int, tuple[float, np.ndarray]] = {}
    fromto = np.zeros(6)
    for gh, go in info.pairs:
        if info.hand_geom_finger.get(gh) is None:
            continue
        d = mujoco.mj_geomDistance(model, data, gh, go, 2.0 * dist_max + 0.01, fromto)
        if d <= dist_max and (gh not in best or d < best[gh][0]):
            best[gh] = (float(d), fromto.copy())
    out = []
    com = data.xipos[info.obj_body].copy()
    for gh, (d, ft) in best.items():
        v = ft[3:6] - ft[0:3]
        nv = float(np.linalg.norm(v))
        if nv > 1e-7:
            n = v / nv if d >= 0 else -v / nv
        else:
            n = com - ft[0:3]
            n = n / max(float(np.linalg.norm(n)), 1e-9)
        out.append(Contact(gh, int(info.hand_geom_finger[gh]), ft[0:3].copy(), ft[3:6].copy(), n, d))
    return out


def grasp_force_nnls(contacts: list[Contact], com: np.ndarray, force_req: np.ndarray, mu: float,
                     rho: float = 1e-3) -> tuple[np.ndarray, float]:
    """Normal force per contact balancing `force_req` (and zero torque about `com`) with pyramid edges, alpha >= 0.

    Returns (f_n per contact, relative residual |G a - w| / |w|). Least-norm tie-break by a rho * I block."""
    if not contacts:
        return np.zeros(0), 1.0
    cols = []
    for c in contacts:
        t1 = np.cross(c.normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(c.normal, [0.0, 1.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(c.normal, t1)
        for t in (t1, -t1, t2, -t2):
            e = c.normal + mu * t
            cols.append(np.concatenate([e, np.cross(c.p_obj - com, e)]))
    g = np.stack(cols, axis=1)                       # (6, 4 n_c)
    w = np.concatenate([force_req, np.zeros(3)])
    a_aug = np.vstack([g, rho * np.eye(g.shape[1])])
    b_aug = np.concatenate([w, np.zeros(g.shape[1])])
    alpha, _ = nnls(a_aug, b_aug, maxiter=50 * g.shape[1])
    res = float(np.linalg.norm(g @ alpha - w) / max(np.linalg.norm(w), 1e-9))
    return alpha.reshape(len(contacts), 4).sum(axis=1), res


def squeeze_frame(model: mujoco.MjModel, data: mujoco.MjData, info: HandInfo, contacts: list[Contact],
                  f_n: np.ndarray, cfg: SqueezeConfig) -> tuple[np.ndarray, np.ndarray]:
    """(gap term (n_f,), per-finger penetration terms (5, n_f)) of finger-joint offsets for one frame (DLS)."""
    n_f = len(info.finger_dof)
    gap_q = np.zeros(n_f)
    fing_q = np.zeros((len(FINGERS), n_f))
    if not contacts:
        return gap_q, fing_q
    rows = []
    jacp = np.zeros((3, model.nv))
    for c in contacts:
        mujoco.mj_jac(model, data, jacp, None, c.p_hand, int(model.geom_bodyid[c.hand_geom]))
        rows.append(c.normal @ jacp[:, info.finger_dof])
    a = np.stack(rows)                                           # (n_c, n_f)
    pinv = a.T @ np.linalg.inv(a @ a.T + cfg.dls_lambda ** 2 * np.eye(len(contacts)))
    gaps = np.clip([c.gap for c in contacts], -cfg.move_max_m, cfg.move_max_m - cfg.pen_min_m)
    gap_q = pinv @ gaps
    pen = np.clip(cfg.delta_base_m + f_n / cfg.k_contact_n_per_m, cfg.pen_min_m, cfg.pen_max_m)
    for fi in range(len(FINGERS)):
        b = np.array([pen[i] if c.finger == fi else 0.0 for i, c in enumerate(contacts)])
        if b.any():
            fing_q[fi] = pinv @ b
    return gap_q, fing_q


def squeeze_trajectory(model: mujoco.MjModel, qpos_frames: np.ndarray, info: HandInfo, cfg: SqueezeConfig,
                       ref_dt: float, mu: float) -> dict[str, np.ndarray]:
    """Per reference frame: gap term (F, n_f), finger terms (F, 5, n_f), held (F,), contacts (F,), residual (F,)."""
    data = mujoco.MjData(model)
    n_frames = qpos_frames.shape[0]
    n_f = len(info.finger_dof)
    gap = np.zeros((n_frames, n_f))
    fing = np.zeros((n_frames, len(FINGERS), n_f))
    held = np.zeros(n_frames)
    n_c = np.zeros(n_frames, dtype=int)
    resid = np.ones(n_frames)
    coms = np.zeros((n_frames, 3))
    per_frame = []
    for f in range(n_frames):
        data.qpos[:] = qpos_frames[f]
        mujoco.mj_forward(model, data)
        coms[f] = data.xipos[info.obj_body]
        per_frame.append(contacts_at(model, data, info, cfg.contact_dist_m))
    acc = np.zeros_like(coms)
    if n_frames >= 3:
        acc[1:-1] = (coms[2:] - 2 * coms[1:-1] + coms[:-2]) / ref_dt ** 2
        k = max(1, cfg.smooth_frames)
        kern = np.ones(k) / k
        acc = np.stack([np.convolve(acc[:, i], kern, mode="same") for i in range(3)], axis=1)
    for f in range(n_frames):
        cs = per_frame[f]
        n_c[f] = len(cs)
        if not cs:
            continue
        data.qpos[:] = qpos_frames[f]
        mujoco.mj_forward(model, data)
        force_req = info.m_obj * (acc[f] + np.array([0.0, 0.0, GRAVITY]))
        f_n, resid[f] = grasp_force_nnls(cs, coms[f], force_req, mu)
        gap[f], fing[f] = squeeze_frame(model, data, info, cs, f_n, cfg)
        held[f] = 1.0
    return {"gap": gap, "finger": fing, "held": held, "n_contacts": n_c, "residual": resid}


def frames_to_steps(per_frame: np.ndarray, warmup_steps: int, steps_per_frame: int, total_steps: int,
                    ramp_steps: int, smooth_frames: int) -> np.ndarray:
    """(F, ...) per reference frame -> (total_steps, ...) per sim step on DO AS I DO's time base.

    Frames are smoothed (moving average), placed at sim steps warmup + f * (F*spf - 1)/(F - 1) (their
    align_corners resampling) and linearly interpolated; after the last frame the last value holds. During the
    warmup the frame-0 value ramps in linearly over its last `ramp_steps` steps."""
    x = np.asarray(per_frame, dtype=np.float64)
    n_frames = x.shape[0]
    flat = x.reshape(n_frames, -1)
    k = max(1, smooth_frames)
    if k > 1 and n_frames > 1:
        pad = np.concatenate([np.repeat(flat[:1], k // 2, 0), flat, np.repeat(flat[-1:], k - 1 - k // 2, 0)])
        flat = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(flat.shape[1])], 1)
    out = np.zeros((total_steps, flat.shape[1]))
    span = max(n_frames * steps_per_frame - 1, 1)
    pos = warmup_steps + np.arange(n_frames) * (span / max(n_frames - 1, 1))
    t = np.arange(total_steps, dtype=np.float64)
    post = t >= warmup_steps
    for i in range(flat.shape[1]):
        out[post, i] = np.interp(t[post], pos, flat[:, i])
    if ramp_steps > 0:
        r0 = max(warmup_steps - ramp_steps, 0)
        ramp = np.clip((t - r0) / max(warmup_steps - r0, 1), 0.0, 1.0)
        out[~post] = ramp[~post, None] * flat[0][None, :]
    return out.reshape((total_steps,) + x.shape[1:])
