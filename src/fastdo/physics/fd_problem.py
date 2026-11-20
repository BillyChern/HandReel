"""D4' problem definition: DO AS I DO's executor semantics and reward, on CPU MuJoCo in float64.

Clean-room implementation against the MuJoCo API and the published objective (DO AS I DO `get_reward`, read, not
copied). Pure numpy + mujoco; no torch, no warp. Single-hand scenes only (actuators = the first nu dofs, one free
object joint at the end of qpos).

Executor semantics (`ExecutorSpec`, `rollout`): from the start state, control row s is applied at step s. For
s < W (= warmup steps) the object weld is active with its own solver parameters and gravity is ZERO; from s = W on
the weld is off and gravity is the default. Controls are rounded to float32 (the GPU executor's control dtype).

Windows: a rollout may cover rows s0 .. s0+T-1 from a recorded state (receding-horizon use); costs then use the
absolute row for the reference, the warmup multiplier and the gates, and the terminal factor on the window's last row.

Reward (`step_costs`), for executed control row s, with x_{s+1} the state after the step and ref row s+1 (the
alignment of DO AS I DO's rollout: abs_step = s+1):
- a_s = || w (.) dq(x_{s+1}, ref_{s+1}) ||_2, w = (base_pos, base_rot, joint, obj_pos, obj_rot) scales on the nv-dim
  difference (robot dims subtract, object position subtracts, object rotation = quat_sub axis-angle);
- b_s = || qvel(x_{s+1}) - qvel_ref_{s+1} ||_2 with scale `vel_scale`;
- h_s = max(0, max_c pen_c - margin) over hand-object contacts of the step's collision pass (pre-step positions x_s,
  as in MJWarp/MuJoCo where contacts are computed before integration), cost `pen_scale` * h_s^2;
- p_s = pedestal mismatch (binary, DO AS I DO's in-hand / near-pedestal gates at sim step s+1; zero in the warmup).
Per-step cost c_s = a_s + vel_scale*b_s + pen_scale*h_s^2 + ped_scale*p_s, weighted by m_s = warmup_mult if s+1 < W
else 1, and by `terminal_scale` on the last row. J = sum_s m_s c_s (whole clip; DO AS I DO averages per horizon).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

__all__ = ["ExecutorSpec", "RewardSpec", "Rollout", "StepCosts", "build_executor", "dq_obj_rot", "quat_sub",
           "rollout", "set_phase", "step_costs"]

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
TERMINATE_Z = -1.0
TERMINATE_EVERY = 100          # the executor checks termination after every control block


@dataclass(eq=False)
class ExecutorSpec:
    model: mujoco.MjModel           # own copy: weld relpose written, gravity switched per step
    q0: np.ndarray
    ctrl0: np.ndarray
    release_step: int               # W
    gravity: np.ndarray
    weld_ids: tuple[int, ...]
    n_steps: int                    # executed control rows T
    obj_qadr: int                   # qpos address of the object free joint (7 entries)


@dataclass(eq=False)
class RewardSpec:
    qpos_ref: np.ndarray            # (Tref, nq), Tref >= T + 1
    qvel_ref: np.ndarray            # (Tref, nv)
    w_dq: np.ndarray                # (nv,)
    vel_scale: float
    pen_scale: float
    pen_margin: float
    terminal_scale: float
    warmup_mult: float
    hand_geom: np.ndarray           # (ngeom,) bool: penetration hand set ("collision_hand_" geoms)
    object_geom: np.ndarray         # (ngeom,) bool: penetration object set ("object" geoms)
    ped_scale: float = 0.0
    side_object_geom: np.ndarray | None = None   # object (+ supports) of the side, for the mismatch checks
    pedestal_geom: np.ndarray | None = None
    gate_in_hand: np.ndarray | None = None       # (Tref,) bool
    gate_near_ped: np.ndarray | None = None


@dataclass(eq=False)
class Rollout:  # noqa: D101 (fields below)
    qpos: np.ndarray                # (T, nq) state after each step
    qvel: np.ndarray
    ctrl: np.ndarray                # (T, nu) applied (float32-rounded) controls
    states: np.ndarray              # (T, nstate) full integration state BEFORE each step
    pen: np.ndarray                 # (T,) max hand-object penetration of the step's collision pass
    pen_grad: dict = field(default_factory=dict)   # s -> (nv,) d pen / d q-tangent at x_s, where pen > margin
    ho_touch: np.ndarray | None = None
    ped_touch: np.ndarray | None = None
    terminated_at: int | None = None    # executed rows before termination (None = whole clip)
    finite: bool = True
    executor_terminated_at: int | None = None   # where the executor WOULD have stopped (also when terminate=False)
    s0: int = 0                         # absolute control row of the first recorded row


@dataclass(eq=False)
class StepCosts:
    a: np.ndarray
    b: np.ndarray
    h: np.ndarray
    ped: np.ndarray
    mult: np.ndarray                # warmup x terminal multipliers m_s
    total: float
    dq: np.ndarray                  # (T, nv) unweighted differences (robot, obj pos, obj rot)
    dv: np.ndarray                  # (T, nv)


def _relpose(model: mujoco.MjModel, data: mujoco.MjData, eq_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Weld-to-world relpose that pins body eq_obj1id at its current pose (DO AS I DO's `set_weld_target`)."""
    body = int(model.eq_obj1id[eq_id])
    anchor = np.array(model.eq_data[eq_id, 0:3])
    conj = np.zeros(4)
    mujoco.mju_negQuat(conj, data.xquat[body])
    rel = np.zeros(3)
    mujoco.mju_rotVecQuat(rel, anchor - data.xpos[body], conj)
    return rel, conj


def build_executor(model: mujoco.MjModel, q0: np.ndarray, ctrl0: np.ndarray, release_step: int,
                   gravity: np.ndarray, weld_ids: list[int], n_steps: int) -> ExecutorSpec:
    """Copy `model`, write the weld target at the start pose (float32-rounded q0, as the GPU executor holds it)."""
    m = copy.copy(model)
    d = mujoco.MjData(m)
    q0 = np.asarray(q0, np.float64).astype(np.float32).astype(np.float64)
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d)
    for e in weld_ids:
        rel, conj = _relpose(m, d, e)
        m.eq_data[e, 3:6] = rel
        m.eq_data[e, 6:10] = conj
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(f"single-hand scene expected one free joint, got {len(free)}")
    return ExecutorSpec(model=m, q0=q0, ctrl0=np.asarray(ctrl0, np.float64), release_step=int(release_step),
                        gravity=np.asarray(gravity, np.float64), weld_ids=tuple(int(i) for i in weld_ids),
                        n_steps=int(n_steps), obj_qadr=int(m.jnt_qposadr[free[0]]))


def set_phase(ex: ExecutorSpec, m: mujoco.MjModel, d: mujoco.MjData, s: int) -> None:
    """Weld and gravity for control row s (weld on and zero gravity before W)."""
    welded = s < ex.release_step
    m.opt.gravity[:] = 0.0 if welded else ex.gravity
    for e in ex.weld_ids:
        d.eq_active[e] = 1 if welded else 0


def start_data(ex: ExecutorSpec, m: mujoco.MjModel | None = None) -> mujoco.MjData:
    m = ex.model if m is None else m
    d = mujoco.MjData(m)
    d.qpos[:] = ex.q0
    d.qvel[:] = 0.0
    d.ctrl[:] = ex.ctrl0.astype(np.float32)
    return d


def _contact_pen(m: mujoco.MjModel, d: mujoco.MjData, rw: RewardSpec, want_grad: bool):
    """(max hand-object penetration, its d pen/d q-tangent or None, hand-object touching, pedestal touching)."""
    best, best_i = 0.0, -1
    ho = ped = False
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        dist = float(c.dist)
        if rw.hand_geom[g1] and rw.object_geom[g2] or rw.object_geom[g1] and rw.hand_geom[g2]:
            pen = max(-dist, 0.0) if np.isfinite(dist) else 0.0
            if pen > best:
                best, best_i = pen, i
        if rw.side_object_geom is not None and dist <= 0.0:
            so1, so2 = rw.side_object_geom[g1], rw.side_object_geom[g2]
            if (rw.hand_geom[g1] and so2) or (so1 and rw.hand_geom[g2]):
                ho = True
            if rw.pedestal_geom is not None and ((rw.pedestal_geom[g1] and so2) or (so1 and rw.pedestal_geom[g2])):
                ped = True
    grad = None
    if want_grad and best_i >= 0 and best - rw.pen_margin > 0.0:
        c = d.contact[best_i]
        n = np.array(c.frame[0:3])
        jp1 = np.zeros((3, m.nv))
        jp2 = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp1, None, c.pos, int(m.geom_bodyid[int(c.geom[0])]))
        mujoco.mj_jac(m, d, jp2, None, c.pos, int(m.geom_bodyid[int(c.geom[1])]))
        grad = -(n @ (jp2 - jp1))      # dist grows when geom2 moves along +n relative to geom1; pen = -dist
    return best, grad, ho, ped


def rollout(ex: ExecutorSpec, rw: RewardSpec | None, ctrls: np.ndarray, want_grad: bool = True,
            m: mujoco.MjModel | None = None, terminate: bool = True, s0: int = 0,
            state0: np.ndarray | None = None) -> Rollout:
    """Execute `ctrls` (T, nu) as control rows s0 .. s0+T-1 with the executor's semantics; record what D4' needs.

    The start is the clip's start state (s0 = 0) or `state0`, a full integration state recorded before row s0.
    `terminate=True` is the executor (stop after the control block in which the object fell below z = -1 m);
    `terminate=False` simulates every row (the planner's objective: a drop costs its tracking error, finitely)."""
    m = ex.model if m is None else m
    if state0 is None:
        if s0 != 0:
            raise ValueError("a rollout that does not start at row 0 needs its start state")
        d = start_data(ex, m)
    else:
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, state0, STATE_SPEC)
    T = int(ctrls.shape[0])
    nq, nv, nu = m.nq, m.nv, m.nu
    ns = mujoco.mj_stateSize(m, STATE_SPEC)
    out = Rollout(qpos=np.zeros((T, nq)), qvel=np.zeros((T, nv)), ctrl=np.zeros((T, nu)), states=np.zeros((T, ns)),
                  pen=np.zeros(T), ho_touch=np.zeros(T, bool), ped_touch=np.zeros(T, bool), s0=int(s0))
    u32 = np.asarray(ctrls, np.float64).astype(np.float32).astype(np.float64)
    z_adr = ex.obj_qadr + 2
    for s in range(T):
        set_phase(ex, m, d, s0 + s)
        d.ctrl[:] = u32[s]
        mujoco.mj_getState(m, d, out.states[s], STATE_SPEC)
        mujoco.mj_step(m, d)
        out.qpos[s], out.qvel[s], out.ctrl[s] = d.qpos, d.qvel, u32[s]
        if rw is not None:
            pen, grad, ho, ped = _contact_pen(m, d, rw, want_grad)
            out.pen[s], out.ho_touch[s], out.ped_touch[s] = pen, ho, ped
            if grad is not None:
                out.pen_grad[s] = grad
        bad = any(d.warning[w].number > 0 for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                                                      mujoco.mjtWarning.mjWARN_BADQVEL))
        if bad or not (
                np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            out.finite = False
            out.terminated_at = s + 1
            break
        if (s0 + s + 1) % TERMINATE_EVERY == 0 and d.qpos[z_adr] < TERMINATE_Z and out.executor_terminated_at is None:
            out.executor_terminated_at = s + 1
            if terminate:
                out.terminated_at = s + 1
                break
    if out.terminated_at is not None:
        k = out.terminated_at
        for name in ("qpos", "qvel", "ctrl", "states", "pen", "ho_touch", "ped_touch"):
            setattr(out, name, getattr(out, name)[:k])
    return out


def _quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(u, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(v, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def quat_sub(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) of qb^-1 qa (wxyz), with DO AS I DO's convention: angle 2*atan2(|v|, w) wrapped to
    (-pi, pi] by subtracting 2*pi above pi; zero vector part gives zero."""
    qn = np.array(qb, dtype=np.float64, copy=True)
    qn[..., 1:] *= -1.0
    q = _quat_mul(qn, np.asarray(qa, np.float64))
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    ang = 2.0 * np.arctan2(s, q[..., 0])
    ang = np.where(ang > np.pi, ang - 2.0 * np.pi, ang)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = v * (ang / s)[..., None]
    return np.where((s > 0.0)[..., None], out, 0.0)


def dq_obj_rot(q_obj: np.ndarray, q_ref: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """(T, 3, 3) d quat_sub(q (x) exp(delta), q_ref) / d delta at delta = 0 (MuJoCo's free-joint tangent is local),
    by central differences of the closed form."""
    out = np.zeros(q_obj.shape[:-1] + (3, 3))
    for i in range(3):
        dv = np.zeros(4)
        dv[0] = np.cos(h / 2.0)
        dv[1 + i] = np.sin(h / 2.0)
        dm = dv.copy()
        dm[1 + i] *= -1.0
        out[..., :, i] = (quat_sub(_quat_mul(q_obj, dv), q_ref) - quat_sub(_quat_mul(q_obj, dm), q_ref)) / (2 * h)
    return out


def _ped_mismatch(ro: Rollout, rw: RewardSpec, W: int) -> np.ndarray:
    T = ro.qpos.shape[0]
    out = np.zeros(T)
    if rw.ped_scale <= 0 or rw.gate_in_hand is None or rw.pedestal_geom is None or not rw.pedestal_geom.any():
        return out
    n = rw.gate_in_hand.shape[0]
    for s in range(T):
        k = ro.s0 + s + 1
        if k < W:
            continue
        i = min(k, n - 1)
        ih, npd = bool(rw.gate_in_hand[i]), bool(rw.gate_near_ped[i])
        if ih and not npd:
            out[s] = 0.0 if ro.ho_touch[s] else 1.0
        elif npd and not ih:
            out[s] = 0.0 if ro.ped_touch[s] else 1.0
    return out


def step_costs(ex: ExecutorSpec, rw: RewardSpec, ro: Rollout) -> StepCosts:
    """Per-step cost terms of the executed rows and the whole-clip objective J (see the module docstring)."""
    T = ro.qpos.shape[0]
    a0 = ex.obj_qadr
    nr = a0                                   # robot qpos dims (= robot dofs)
    o = ro.s0
    ref = rw.qpos_ref[o + 1:o + T + 1]
    dq = np.zeros((T, ro.qvel.shape[1]))
    dq[:, :nr] = ro.qpos[:, :nr] - ref[:, :nr]
    dq[:, nr:nr + 3] = ro.qpos[:, a0:a0 + 3] - ref[:, a0:a0 + 3]
    dq[:, nr + 3:nr + 6] = quat_sub(ro.qpos[:, a0 + 3:a0 + 7], ref[:, a0 + 3:a0 + 7])
    dv = ro.qvel - rw.qvel_ref[o + 1:o + T + 1]
    a = np.linalg.norm(dq * rw.w_dq, axis=1)
    b = np.linalg.norm(dv, axis=1)
    h = np.maximum(ro.pen - rw.pen_margin, 0.0)
    ped = _ped_mismatch(ro, rw, ex.release_step)
    mult = np.where(np.arange(o + 1, o + T + 1) < ex.release_step, rw.warmup_mult, 1.0)
    mult[-1] *= rw.terminal_scale
    c = a + rw.vel_scale * b + rw.pen_scale * h * h + rw.ped_scale * ped
    total = float(np.sum(mult * c)) if (ro.finite and ro.terminated_at is None) else float("inf")
    return StepCosts(a=a, b=b, h=h, ped=ped, mult=mult, total=total, dq=dq, dv=dv)
