"""Quasi-dynamic convex contact step: one time step as a convex QP over generalised velocities.

Model (Pang-Tedrake quasi-dynamic contact with Anitescu's convex relaxation; CQDC). State q (MuJoCo qpos); the
decision variable is the generalised velocity v (nv) over one step of length h, split into the actuated hand dofs
v_a (each driven by one MuJoCo affine position actuator) and the unactuated object dofs v_u (free joints):

    minimise_v   1/2 v^T Q v - b^T v
    subject to   phi_i / h + (J_n,i + mu_i d_ik^T J_t,i) v >= 0      every contact i, pyramid edge k = 0..K-1
                 joint limits  lo <= q_j + h v_j <= hi                 (limited hand joints near a bound)

    Q = blockdiag( h K_p + K_v ,  M_uu / h )
    b = [ g_a u + b0_a - K_p q_a - c_a(q) + p_a(q) ;  -c_u(q) + p_u(q) ]

- K_p = -gear^2 biasprm[1] (position stiffness), K_v = -gear^2 biasprm[2] + dof_damping (velocity damping),
  g_a = gear gainprm[0], b0_a = gear biasprm[0]: the actuator force gear (gainprm0 u + biasprm0 + biasprm1 L +
  biasprm2 Ldot) with L = gear q, evaluated implicitly at q + h v. The hand term is the stiffness tracking energy
  (1/2h) ||q_a + h v_a - q_eq(u)||^2_{K_p} + (1/2) v_a^T K_v v_a (an overdamped servo; hand inertia neglected).
- M_uu: the object block of MuJoCo's mass matrix at q; the object term is the quasi-dynamic energy
  (1/2h) v_u^T M_uu v_u (the object starts every step at rest: no momentum is carried over).
- c(q) = qfrc_bias at zero velocity (gravity), p(q) = qfrc_passive at zero velocity (springs, gravcomp).
- Contacts: MuJoCo's own collision query at the CURRENT q (`mj_fwdPosition`), with every pair/geom margin widened by
  `query_margin` so contacts about to close are seen; phi_i = the contact's signed distance (penetration scaled by
  `phi_gain`); n_i = frame row 0 (geom1 -> geom2); J_* = n/t rows of J(p, body2) - J(p, body1) at the contact point.
  Edge directions d_ik = cos(2 pi k/K) t1 + sin(2 pi k/K) t2 (frame rows 1, 2); mu_i = the contact's sliding friction.
- KKT: Q v = b + G^T z, z >= 0, z (G v - g) = 0, so the contact force on body2 is sum_k z_ik (n_i + mu_i d_ik): inside
  the polyhedral Coulomb cone. The relaxation (Anitescu) lets a sliding contact separate at normal speed mu |v_t|.

Next state: q+ = mj_integratePos(q, v, h). Solver: `fastdo.physics.dense_qp.solve_qp` (primal-dual IPM) on the
Jacobi-scaled problem y = D^-1 v, D = diag(Q)^-1/2, with every inequality row scaled to unit norm (the object's
rotational block of Q is ~1e-2 against ~1e2 for the hand, and dense_qp's tolerance is absolute).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from fastdo.physics.dense_qp import QPResult, solve_qp

__all__ = ["ContactSet", "QDConfig", "QDModel", "StepQP", "StepResult", "assemble", "build_model",
           "ctrl_from_command_velocity", "edge_angles", "query_contacts", "step"]


@dataclass(frozen=True)
class QDConfig:
    h: float = 0.005               # step length (s)
    n_edges: int = 4               # polyhedral friction-cone edges per contact
    query_margin: float = 0.005    # added to every geom / pair margin for the contact query (m)
    phi_gain: float = 1.0          # fraction of a penetration corrected within one step
    limit_margin: float = 0.05     # joint-limit rows only for joints within this distance of a bound
    tol: float = 1e-7              # IPM tolerance (on the scaled problem)
    max_iter: int = 100


@dataclass(eq=False)
class QDModel:
    m: mujoco.MjModel              # own copy, margins widened
    d: mujoco.MjData
    cfg: QDConfig
    act_dof: np.ndarray            # (nu,) dof of each actuator
    act_qadr: np.ndarray           # (nu,) its qpos address
    act_gain: np.ndarray           # (nu,) d tau / d u
    act_bias0: np.ndarray          # (nu,)
    k_pos: np.ndarray              # (nu,)
    k_vel: np.ndarray              # (nu,)
    obj_dof: np.ndarray            # dofs of the free joints
    lim_dof: np.ndarray            # limited hinge/slide joints: dof, qpos address, range
    lim_qadr: np.ndarray
    lim_lo: np.ndarray
    lim_hi: np.ndarray
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (K, 2) cos, sin


@dataclass(eq=False)
class ContactSet:
    geom: np.ndarray               # (nc, 2)
    dist: np.ndarray               # (nc,)
    mu: np.ndarray                 # (nc,)
    frame: np.ndarray              # (nc, 3, 3) rows n, t1, t2
    jrel: np.ndarray               # (nc, 3, nv) J(p, body2) - J(p, body1)
    M: np.ndarray                  # (nv, nv) mass matrix at q
    bias: np.ndarray               # (nv,) qfrc_bias at zero velocity
    passive: np.ndarray            # (nv,) qfrc_passive at zero velocity


@dataclass(eq=False)
class StepQP:
    Q: np.ndarray
    b: np.ndarray
    G: np.ndarray
    g: np.ndarray
    db_du: np.ndarray              # (nv, nu)
    row_contact: np.ndarray        # (m,) contact index of each row, -1 = joint limit


@dataclass(eq=False)
class StepResult:
    v: np.ndarray
    z: np.ndarray                  # (m,) row multipliers, original units (contact force = sum z (n + mu d))
    active: np.ndarray             # (m,) bool: scaled multiplier > scaled slack (the IPM's partition)
    q_next: np.ndarray
    qp: StepQP
    sol: QPResult                  # the solver's result on the SCALED problem
    contacts: ContactSet
    t_query: float
    t_assemble: float
    t_solve: float


def edge_angles(n_edges: int) -> np.ndarray:
    th = 2.0 * np.pi * np.arange(n_edges) / n_edges
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def build_model(model: mujoco.MjModel, cfg: QDConfig | None = None) -> QDModel:
    """Validate the actuator/joint structure and copy the model with widened contact margins."""
    cfg = QDConfig() if cfg is None else cfg
    m = copy.copy(model)
    m.geom_margin[:] = m.geom_margin + cfg.query_margin
    if m.npair:
        m.pair_margin[:] = m.pair_margin + cfg.query_margin
    hinge_slide = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    act_dof, gain, b0, kp, kv = [], [], [], [], []
    for a in range(m.nu):
        if (int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT)
                or int(m.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE)
                or int(m.actuator_gaintype[a]) != int(mujoco.mjtGain.mjGAIN_FIXED)
                or int(m.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_AFFINE)):
            raise ValueError(f"actuator {a}: only joint-transmission affine position actuators are supported")
        j = int(m.actuator_trnid[a, 0])
        if int(m.jnt_type[j]) not in hinge_slide:
            raise ValueError(f"actuator {a} drives a joint that is not a hinge/slide")
        gr = float(m.actuator_gear[a, 0])
        dof = int(m.jnt_dofadr[j])
        act_dof.append(dof)
        gain.append(gr * float(m.actuator_gainprm[a, 0]))
        b0.append(gr * float(m.actuator_biasprm[a, 0]))
        kp.append(-gr * gr * float(m.actuator_biasprm[a, 1]))
        kv.append(-gr * gr * float(m.actuator_biasprm[a, 2]) + float(m.dof_damping[dof]))
    if len(set(act_dof)) != len(act_dof):
        raise ValueError("two actuators on one dof")
    obj = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == free:
            obj.extend(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + 6))
        elif int(m.jnt_dofadr[j]) not in act_dof:
            raise ValueError(f"joint {j} is neither actuated nor free")
    if np.any(np.asarray(kp) <= 0):
        raise ValueError("every actuated dof needs a positive position stiffness")
    lim = [j for j in range(m.njnt) if m.jnt_limited[j] and int(m.jnt_type[j]) in hinge_slide]
    act_qadr = [int(m.jnt_qposadr[int(m.dof_jntid[k])]) for k in act_dof]
    return QDModel(m=m, d=mujoco.MjData(m), cfg=cfg, act_dof=np.array(act_dof, int),
                   act_qadr=np.array(act_qadr, int), act_gain=np.array(gain),
                   act_bias0=np.array(b0), k_pos=np.array(kp), k_vel=np.array(kv), obj_dof=np.array(obj, int),
                   lim_dof=np.array([m.jnt_dofadr[j] for j in lim], int),
                   lim_qadr=np.array([m.jnt_qposadr[j] for j in lim], int),
                   lim_lo=np.array([m.jnt_range[j, 0] for j in lim]), lim_hi=np.array([m.jnt_range[j, 1] for j in lim]),
                   edges=edge_angles(cfg.n_edges))


def query_contacts(qm: QDModel, q: np.ndarray) -> ContactSet:
    """MuJoCo's collision query, mass matrix and zero-velocity bias/passive forces at q."""
    m, d = qm.m, qm.d
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d)
    mujoco.mj_fwdVelocity(m, d)
    M = np.zeros((m.nv, m.nv))
    if hasattr(d, "qM"):            # MuJoCo <= 3.4 (DO AS I DO's): mj_fullM(m, dst, qM)
        mujoco.mj_fullM(m, M, d.qM)
    else:                           # newer: mj_fullM(m, d, dst)
        mujoco.mj_fullM(m, d, M)
    keep = [i for i in range(d.ncon) if d.contact[i].exclude in (0, 1)]
    nc = len(keep)
    geom = np.zeros((nc, 2), int)
    dist, mu = np.zeros(nc), np.zeros(nc)
    frame = np.zeros((nc, 3, 3))
    jrel = np.zeros((nc, 3, m.nv))
    jp = np.zeros((3, m.nv))
    for r, i in enumerate(keep):
        c = d.contact[i]
        geom[r] = c.geom
        dist[r] = c.dist
        mu[r] = c.friction[0]
        frame[r] = np.asarray(c.frame).reshape(3, 3)
        for sgn, gi in ((-1.0, int(c.geom[0])), (1.0, int(c.geom[1]))):
            body = int(m.geom_bodyid[gi])
            if body == 0:
                continue
            mujoco.mj_jac(m, d, jp, None, c.pos, body)
            jrel[r] += sgn * jp
    return ContactSet(geom=geom, dist=dist, mu=mu, frame=frame, jrel=jrel, M=M, bias=d.qfrc_bias.copy(),
                      passive=d.qfrc_passive.copy())


def assemble(qm: QDModel, cs: ContactSet, q: np.ndarray, ctrl: np.ndarray) -> StepQP:
    """The step's QP data (see the module docstring)."""
    cfg, m = qm.cfg, qm.m
    h, nv = cfg.h, m.nv
    Q = np.zeros((nv, nv))
    a = qm.act_dof
    Q[a, a] = h * qm.k_pos + qm.k_vel
    o = qm.obj_dof
    Q[np.ix_(o, o)] = cs.M[np.ix_(o, o)] / h
    b = cs.passive - cs.bias
    q_act = np.asarray(q)[qm.act_qadr]
    b[a] += qm.act_gain * np.asarray(ctrl, float) + qm.act_bias0 - qm.k_pos * q_act
    db_du = np.zeros((nv, m.nu))
    db_du[a, np.arange(m.nu)] = qm.act_gain
    rows, rhs, owner = [], [], []
    if cs.dist.size:
        phi = np.where(cs.dist >= 0.0, cs.dist, cfg.phi_gain * cs.dist)
        jn = np.einsum("cj,cjv->cv", cs.frame[:, 0, :], cs.jrel)
        jt1 = np.einsum("cj,cjv->cv", cs.frame[:, 1, :], cs.jrel)
        jt2 = np.einsum("cj,cjv->cv", cs.frame[:, 2, :], cs.jrel)
        E = qm.edges
        g_rows = (jn[:, None, :] + cs.mu[:, None, None]
                  * (E[None, :, 0, None] * jt1[:, None, :] + E[None, :, 1, None] * jt2[:, None, :]))
        rows.append(g_rows.reshape(-1, nv))
        rhs.append(np.repeat(-phi / h, E.shape[0]))
        owner.append(np.repeat(np.arange(cs.dist.size), E.shape[0]))
    if qm.lim_dof.size:
        qj = np.asarray(q)[qm.lim_qadr]
        for lo_side in (True, False):
            gap = qj - qm.lim_lo if lo_side else qm.lim_hi - qj
            sel = np.flatnonzero(gap < cfg.limit_margin)
            if sel.size == 0:
                continue
            gap_eff = np.where(gap[sel] >= 0.0, gap[sel], cfg.phi_gain * gap[sel])
            R = np.zeros((sel.size, nv))
            R[np.arange(sel.size), qm.lim_dof[sel]] = 1.0 if lo_side else -1.0
            rows.append(R)
            rhs.append(-gap_eff / h)
            owner.append(np.full(sel.size, -1))
    G = np.concatenate(rows) if rows else np.zeros((0, nv))
    g = np.concatenate(rhs) if rhs else np.zeros(0)
    return StepQP(Q=Q, b=b, G=G, g=g, db_du=db_du,
                  row_contact=np.concatenate(owner).astype(int) if owner else np.zeros(0, int))


def ctrl_from_command_velocity(qm: QDModel, q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Control whose actuator equilibrium is q_a + h w (the hand's commanded joint velocity w over one step)."""
    q_act = np.asarray(q)[qm.act_qadr]
    q_eq = q_act + qm.cfg.h * np.asarray(w, float)
    return (qm.k_pos * q_eq - qm.act_bias0) / qm.act_gain


def step(qm: QDModel, q: np.ndarray, ctrl: np.ndarray) -> StepResult:
    """One quasi-dynamic step from q under control `ctrl` (MuJoCo actuator units)."""
    t0 = time.perf_counter()
    cs = query_contacts(qm, q)
    t1 = time.perf_counter()
    qp = assemble(qm, cs, q, ctrl)
    t2 = time.perf_counter()
    D = 1.0 / np.sqrt(np.diag(qp.Q))
    Qs = qp.Q * D[:, None] * D[None, :]
    if qp.G.shape[0]:
        Gs = qp.G * D[None, :]
        rn = np.linalg.norm(Gs, axis=1)
        rn = np.where(rn > 0.0, rn, 1.0)
        sol = solve_qp(Qs, -D * qp.b, G=Gs / rn[:, None], h=qp.g / rn, tol=qm.cfg.tol, max_iter=qm.cfg.max_iter)
        s_scaled = (Gs / rn[:, None]) @ sol.x - qp.g / rn
        z, active = sol.z / rn, sol.z > np.maximum(s_scaled, 0.0)
    else:
        sol = solve_qp(Qs, -D * qp.b, tol=qm.cfg.tol, max_iter=qm.cfg.max_iter)
        z, active = np.zeros(0), np.zeros(0, bool)
    v = D * sol.x
    t3 = time.perf_counter()
    q_next = np.array(q, dtype=np.float64, copy=True)
    mujoco.mj_integratePos(qm.m, q_next, v, qm.cfg.h)
    return StepResult(v=v, z=z, active=active, q_next=q_next, qp=qp, sol=sol, contacts=cs, t_query=t1 - t0,
                      t_assemble=t2 - t1, t_solve=t3 - t2)
