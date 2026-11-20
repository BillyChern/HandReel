"""T0': a deterministic, physically consistent transform of DO AS I DO's kinematic warm start that PRE-LOADS the grasp.

Actuator model (MuJoCo position actuators, DO AS I DO's hand): tau_a = kp (u - q) - kv qdot. Hand equation of motion
with contact forces f_i acting ON THE OBJECT at points p_i (the hand feels -f_i):
    M qdd + c(q, qdot) = tau_a - sum_i J_i^T f_i            (J_i: 3 x n_a point Jacobian of the hand at p_i)
so the control that realises a hand motion q(t) while exerting f on the object is
    u = q + Kp^-1 [ M qdd + c + Kv qdot + sum_i J_i^T f_i ]                                         (1)
T0' evaluates (1) on the reference rows, with q replaced by the GAP-CLOSED configuration q* = q_ref + dq_geo:
- Gap closure (QP 1): selected hand links (within `select_m` of the object in the reference) must touch:
      min ||dq||^2 + rho |slack|^2  s.t.  p0 <= n_i^T J_i dq - g_i (+-slack) <= p_max  (g_i signed gap, n_i hand->object)
  over the finger joints (the wrist stays on the reference). Under MuJoCo's soft contact the steady penetration is
  f_n R / (K I) (efc_R, efc_KBIP), ~1e-5 m per N for these scenes, so the commanded overlap p0 only absorbs the
  executor's tracking lag; the FORCE comes from (1).
- Grasp forces (QP 2), pyramidal cone with a safety-reduced friction mu' = mu / s_mu, 4 edges per contact
  (e_ik = n_i + mu' t_ik, alpha_ik >= 0, force on the object f_i = sum_k alpha_ik e_ik):
      min || Kp^-1 J^T E alpha ||^2 + eps ||alpha||^2 + lam || G E alpha - s_w w_req ||^2
      s.t. alpha >= 0,  1^T alpha_i >= f_min   (every selected contact pushes)
  with G the 6 x 3n grasp map about the object COM and w_req = [ m (a_ref - g) ; I alpha_ref + w x I w ] (world;
  the reference object path). The soft equality keeps the QP feasible when the contacts cannot balance the demand
  (reported as the residual); a contact set without force closure gets internal force 0 beyond f_min.
The joint offset of T0' is the minimal-norm  dq_geo + Kp^-1 (J^T f)  on top of the dynamics feed-forward.

Pure numpy + mujoco (no third-party code). Single-hand scenes: actuators 0-5 wrist, 6.. fingers, object free joint.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fastdo.physics.dense_qp import solve_qp

__all__ = ["GraspContact", "PreloadConfig", "actuator_gains", "closure_offset", "find_contacts", "flexion_mask",
           "grasp_forces", "object_wrench_demand", "preload_controls", "pyramid_edges"]


@dataclass(frozen=True)
class PreloadConfig:
    select_m: float = 0.004          # a hand link within this of the object (reference) is a grasp contact
    p0_m: float = 0.0005             # commanded overlap of a selected contact (closure lower bound)
    pmax_m: float = 0.002            # closure upper bound
    f_min_n: float = 0.5             # minimum normal force per selected contact (internal pre-load)
    s_mu: float = 2.0                # friction safety factor
    s_w: float = 1.5                 # wrench-demand safety factor
    eps: float = 1e-6
    lam: float = 1e4
    stride: int = 5                  # rows between solves (linear interpolation in between)
    smooth: int = 21                 # moving average (rows) of the offset
    ramp_steps: int = 100            # the pre-load ramps in over the last 0.5 s of the warmup
    dynamics_ff: bool = True         # include M qdd + c + Kv qdot (actuator-model feed-forward)
    closure: bool = True
    forces: bool = True
    finger_offset_max: float = 0.3   # rad, per finger joint
    wrist_offset_max: float = 0.05   # m / rad
    hold_m: float = 0.0              # hysteresis: a link selected at the previous solve stays selected within this
    flex_rad: float = 0.0            # comparator: + this on every finger flexion joint (ramped like the pre-load)


FLEX_TOKENS = ("CMC_FE", "MCP_FE", "PIP", "DIP", "_IP")


def flexion_mask(model: mujoco.MjModel) -> np.ndarray:
    """Finger flexion actuators (positive = closing for DO AS I DO's Sharpa hand: ranges start at or near 0)."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "" for a in range(model.nu)]
    return np.array([a >= 6 and any(t in n for t in FLEX_TOKENS) for a, n in enumerate(names)])


@dataclass(frozen=True)
class GraspContact:
    hand_geom: int
    body: int
    p_hand: np.ndarray
    p_obj: np.ndarray
    normal: np.ndarray               # unit, from the hand into the object
    gap: float                       # signed distance (negative = penetrating)


def actuator_gains(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dof index, kp, kv) per actuator of a position-actuated hand (gain kp, bias [0, -kp, -kv])."""
    dof = np.array([int(model.jnt_dofadr[int(model.actuator_trnid[a, 0])]) for a in range(model.nu)])
    kp = np.asarray(model.actuator_gainprm[:, 0], float).copy()
    kv = -np.asarray(model.actuator_biasprm[:, 2], float).copy()
    if not np.allclose(-model.actuator_biasprm[:, 1], kp):
        raise ValueError("expected position actuators (biasprm[1] = -kp)")
    return dof, kp, kv


def pyramid_edges(normal: np.ndarray, mu: float) -> np.ndarray:
    """(4, 3) edges n +- mu t1, n +- mu t2 of the linearised friction cone at a contact."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    t1 = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(n, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.stack([n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2])


def find_contacts(model: mujoco.MjModel, data: mujoco.MjData, hand_geoms: np.ndarray, obj_geoms: np.ndarray,
                  dist_max: float) -> list[GraspContact]:
    """Per hand collision geom, its closest object geom if within `dist_max` (`mj_geomDistance`; kinematics of
    `data` must be current)."""
    best: dict[int, tuple[float, np.ndarray]] = {}
    ft = np.zeros(6)
    for gh in np.flatnonzero(hand_geoms):
        for go in np.flatnonzero(obj_geoms):
            d = mujoco.mj_geomDistance(model, data, int(gh), int(go), 2.0 * dist_max + 0.01, ft)
            if d <= dist_max and (gh not in best or d < best[gh][0]):
                best[int(gh)] = (float(d), ft.copy())
    out = []
    for gh, (d, f) in sorted(best.items()):
        v = f[3:6] - f[0:3]
        nv = float(np.linalg.norm(v))
        if nv > 1e-9:
            n = v / nv if d >= 0 else -v / nv
        else:   # touching exactly: fall back to the direction to the object geom centre
            go = int(np.flatnonzero(obj_geoms)[0])
            n = data.geom_xpos[go] - f[0:3]
            n = n / max(float(np.linalg.norm(n)), 1e-12)
        out.append(GraspContact(gh, int(model.geom_bodyid[gh]), f[0:3].copy(), f[3:6].copy(), n, d))
    return out


def object_wrench_demand(mass: float, inertia: np.ndarray, xmat: np.ndarray, acc: np.ndarray, omega: np.ndarray,
                         alpha: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """[F; tau] (world) the contacts must supply: F = m (a - g), tau = R (I alpha_b + w_b x I w_b); body-frame
    angular velocity/acceleration and principal inertia (MuJoCo's body_inertia with body_iquat folded into xmat)."""
    F = mass * (np.asarray(acc, float) - np.asarray(gravity, float))
    Iw = inertia * omega
    tau_b = inertia * alpha + np.cross(omega, Iw)
    return np.concatenate([F, np.asarray(xmat, float).reshape(3, 3) @ tau_b])


def _grasp_map(contacts: list[GraspContact], com: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    cols = []
    for c, E in zip(contacts, edges):
        r = c.p_obj - com
        for e in E:
            cols.append(np.concatenate([e, np.cross(r, e)]))
    return np.stack(cols, axis=1)   # (6, 4 n_c)


def grasp_forces(contacts: list[GraspContact], com: np.ndarray, w_req: np.ndarray, mu: float, cfg: PreloadConfig,
                 jac_hand: list[np.ndarray] | None = None, kp: np.ndarray | None = None) -> dict:
    """QP 2: pyramid edge forces alpha >= 0 (force on the object per contact = E_i^T alpha_i).

    `jac_hand[i]` (3, n_a) and `kp` (n_a,) weight the objective by the joint offset they cause (Kp^-1 J^T f);
    without them the objective is the sum of squared normal forces. Returns per-contact force vectors on the
    object, normal forces, the joint torque J^T f (if Jacobians are given) and the wrench residual."""
    n_c = len(contacts)
    if n_c == 0:
        return {"f": np.zeros((0, 3)), "fn": np.zeros(0), "tau": None, "residual": float(np.linalg.norm(w_req)),
                "status": "no_contacts"}
    mu_s = mu / cfg.s_mu
    edges = [pyramid_edges(c.normal, mu_s) for c in contacts]
    G = _grasp_map(contacts, com, edges)                       # (6, 4n)
    nv = 4 * n_c
    S = np.zeros((n_c, nv))                                    # normal force of contact i = sum of its alphas
    for i in range(n_c):
        S[i, 4 * i:4 * i + 4] = 1.0
    if jac_hand is not None and kp is not None:
        B = np.zeros((jac_hand[0].shape[1], nv))               # Kp^-1 J^T E
        for i, (J, E) in enumerate(zip(jac_hand, edges)):
            B[:, 4 * i:4 * i + 4] = (J.T @ E.T) / kp[:, None]
        H = B.T @ B
    else:
        H = S.T @ S
    w = cfg.s_w * np.asarray(w_req, float)
    H = H + cfg.eps * np.eye(nv) + cfg.lam * G.T @ G
    c = -cfg.lam * G.T @ w
    Gin = np.vstack([np.eye(nv), S])
    hin = np.concatenate([np.zeros(nv), np.full(n_c, cfg.f_min_n)])
    # scale for conditioning
    sc = max(1.0, float(np.abs(np.diag(H)).max()))
    res = solve_qp(H / sc, c / sc, G=Gin, h=hin, tol=1e-10, max_iter=150)
    a = np.maximum(res.x, 0.0)
    f = np.stack([edges[i].T @ a[4 * i:4 * i + 4] for i in range(n_c)])
    tau = None
    if jac_hand is not None:
        tau = sum(J.T @ fi for J, fi in zip(jac_hand, f))
    return {"f": f, "fn": S @ a, "tau": tau, "residual": float(np.linalg.norm(G @ a - w) / max(np.linalg.norm(w), 1e-9)),
            "status": res.status}


def closure_offset(jn: np.ndarray, gaps: np.ndarray, cfg: PreloadConfig, weight: np.ndarray | None = None,
                   lo: np.ndarray | None = None, hi: np.ndarray | None = None, rho: float = 1e5) -> tuple[np.ndarray, str]:
    """QP 1 (soft band): min dq^T W dq + rho (|s1|^2 + |s2|^2) over (dq, s1 >= 0, s2 >= 0) s.t.
    jn dq - gap + s1 >= p0,  jn dq - gap - s2 <= pmax  (rows of `jn` = n_i^T J_i), optional box lo <= dq <= hi.
    The slacks keep it feasible when the selected contacts cannot all be closed inside the band (e.g. two contacts
    on one link with conflicting normals); rho = 1e5 prices 1 mm of band violation like 0.1 rad of joint offset."""
    nc, n = jn.shape
    W = np.ones(n) if weight is None else np.asarray(weight, float)
    H = np.diag(np.concatenate([W, np.full(2 * nc, rho)]))
    I, Z = np.eye(nc), np.zeros((nc, nc))
    Gin = [np.hstack([jn, I, Z]), np.hstack([-jn, Z, I]), np.hstack([np.zeros((2 * nc, n)), np.eye(2 * nc)])]
    hin = [gaps + cfg.p0_m, -(gaps + cfg.pmax_m), np.zeros(2 * nc)]
    if lo is not None:
        E = np.hstack([np.eye(n), np.zeros((n, 2 * nc))])
        Gin += [E, -E]
        hin += [lo, -hi]
    res = solve_qp(H, np.zeros(n + 2 * nc), G=np.vstack(Gin), h=np.concatenate(hin), tol=1e-12, max_iter=200)
    status = res.status if res.status != "max_iter" or res.primal_residual < 1e-7 else "max_iter"
    return res.x[:n], ("optimal" if res.primal_residual < 1e-7 else status)


def _smooth_rows(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.shape[0] < 2:
        return x
    pad = np.concatenate([np.repeat(x[:1], k // 2, 0), x, np.repeat(x[-1:], k - 1 - k // 2, 0)])
    ker = np.ones(k) / k
    return np.stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(x.shape[1])], 1)


def _object_path(model: mujoco.MjModel, qpos_ref: np.ndarray, obj_qadr: int, k: int) -> dict:
    """Reference object COM acceleration (world), body angular velocity / acceleration, by smoothed differences."""
    dt = float(model.opt.timestep)
    p = qpos_ref[:, obj_qadr:obj_qadr + 3]
    q = qpos_ref[:, obj_qadr + 3:obj_qadr + 7]
    acc = np.zeros_like(p)
    acc[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt ** 2
    w = np.zeros_like(p)
    dq = np.zeros(4)
    qc = np.zeros(4)
    for t in range(q.shape[0] - 1):
        mujoco.mju_negQuat(qc, q[t])
        mujoco.mju_mulQuat(dq, qc, q[t + 1])
        mujoco.mju_quat2Vel(w[t], dq, dt)
    w = _smooth_rows(w, k)
    al = np.zeros_like(w)
    al[1:] = (w[1:] - w[:-1]) / dt
    return {"acc": _smooth_rows(acc, k), "omega": w, "alpha": _smooth_rows(al, k)}


def preload_controls(model: mujoco.MjModel, qpos_ref: np.ndarray, ctrl_ref: np.ndarray, release_step: int,
                     gravity: np.ndarray, side: str, cfg: PreloadConfig, n_steps: int | None = None) -> dict:
    """T0'(ctrl_ref): rows of `ctrl_ref` plus the dynamics feed-forward and the grasp pre-load of eq. (1).

    `qpos_ref` (T_ref, nq) is the executor's reference (warmup interpolation included) on the sim time base,
    `release_step` = W (zero gravity before it, as the executor). Returns {"ctrl": (T, nu), diagnostics}."""
    T = int(n_steps or ctrl_ref.shape[0])
    Tr = qpos_ref.shape[0]
    nu = model.nu
    dof, kp, kv = actuator_gains(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    obj_geoms = np.array([int(model.geom_bodyid[g]) == ob and int(model.geom_group[g]) == 3 for g in range(model.ngeom)])
    hand_geoms = np.array(["collision_hand_" in n and "arm_cyl" not in n for n in names])
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    oq = int(model.jnt_qposadr[free[0]])
    mu = float(model.pair_friction[0, 0]) if model.npair else 1.0
    path = _object_path(model, qpos_ref, oq, cfg.smooth)
    dt = float(model.opt.timestep)
    qd = np.zeros((Tr, model.nv))
    qdd = np.zeros((Tr, model.nv))
    qd[1:, dof] = (qpos_ref[1:, dof] - qpos_ref[:-1, dof]) / dt          # hand joints are hinges/slides: qpos = dof
    qd[:, dof] = _smooth_rows(qd[:, dof], cfg.smooth)
    qdd[1:, dof] = (qd[1:, dof] - qd[:-1, dof]) / dt
    qdd[:, dof] = _smooth_rows(qdd[:, dof], cfg.smooth)
    finger = np.arange(nu) >= 6
    m = model
    d = mujoco.MjData(m)
    g_saved = m.opt.gravity.copy()
    rows = list(range(0, T, max(1, cfg.stride)))
    if rows[-1] != T - 1:
        rows.append(T - 1)
    ff = np.zeros((len(rows), nu))
    geo = np.zeros((len(rows), nu))
    frc = np.zeros((len(rows), nu))
    n_contacts = np.zeros(len(rows), int)
    resid = np.full(len(rows), np.nan)
    fn_sum = np.zeros(len(rows))
    closure_fail = 0
    jacp = np.zeros((3, m.nv))
    held: set[int] = set()
    try:
        for k, t in enumerate(rows):
            tr = min(t, Tr - 1)
            grav = np.zeros(3) if t < release_step else np.asarray(gravity, float)
            m.opt.gravity[:] = grav
            d.qpos[:] = qpos_ref[tr]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            if cfg.dynamics_ff:
                d.qvel[:] = qd[tr]
                d.qacc[:] = qdd[tr]
                out = np.zeros(m.nv)
                mujoco.mj_rne(m, d, 1, out)
                ff[k] = (out[dof] + kv * qd[tr, dof]) / kp
                d.qvel[:] = 0.0
            cs = (find_contacts(m, d, hand_geoms, obj_geoms, max(cfg.select_m, cfg.hold_m))
                  if (cfg.closure or cfg.forces) else [])
            cs = [c for c in cs if c.gap <= cfg.select_m or c.hand_geom in held]
            held = {c.hand_geom for c in cs}
            n_contacts[k] = len(cs)
            if not cs:
                continue
            jac = []
            for c in cs:
                mujoco.mj_jac(m, d, jacp, None, c.p_hand, c.body)
                jac.append(jacp[:, dof].copy())
            if cfg.closure:
                jn = np.stack([c.normal @ J for c, J in zip(cs, jac)])
                use = np.linalg.norm(jn[:, finger], axis=1) > 1e-6
                if use.any():
                    q0 = qpos_ref[tr, dof][finger]
                    lo = np.maximum(-cfg.finger_offset_max, m.jnt_range[m.actuator_trnid[finger, 0], 0] - q0)
                    hi = np.minimum(cfg.finger_offset_max, m.jnt_range[m.actuator_trnid[finger, 0], 1] - q0)
                    lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
                    dq, st = closure_offset(jn[use][:, finger], np.array([c.gap for c, u in zip(cs, use) if u]),
                                            cfg, lo=lo, hi=hi)
                    if st == "optimal":
                        geo[k, finger] = dq
                    else:
                        closure_fail += 1
            if cfg.forces:
                com = d.xipos[ob].copy()
                w_req = object_wrench_demand(float(m.body_mass[ob]), m.body_inertia[ob], d.ximat[ob],
                                             path["acc"][tr], path["omega"][tr], path["alpha"][tr], grav)
                gf = grasp_forces(cs, com, w_req, mu, cfg, jac_hand=jac, kp=kp)
                frc[k] = gf["tau"] / kp
                resid[k] = gf["residual"]
                fn_sum[k] = float(gf["fn"].sum())
    finally:
        m.opt.gravity[:] = g_saved
    grasp = geo + frc
    grasp[:, ~finger] = np.clip(grasp[:, ~finger], -cfg.wrist_offset_max, cfg.wrist_offset_max)
    grasp[:, finger] = np.clip(grasp[:, finger], -cfg.finger_offset_max, cfg.finger_offset_max)
    tt = np.arange(T)
    interp = lambda a: np.stack([np.interp(tt, rows, a[:, i]) for i in range(nu)], 1)   # noqa: E731
    g_rows = _smooth_rows(interp(grasp), cfg.smooth)
    f_rows = _smooth_rows(interp(ff), cfg.smooth)
    ramp = np.clip((tt - (release_step - cfg.ramp_steps)) / max(cfg.ramp_steps, 1), 0.0, 1.0)
    base = np.asarray(ctrl_ref[:T] if ctrl_ref.shape[0] >= T else
                      np.concatenate([ctrl_ref, np.repeat(ctrl_ref[-1:], T - ctrl_ref.shape[0], 0)]), float)
    g_rows = g_rows + cfg.flex_rad * flexion_mask(m)[None, :].astype(float)
    u = base + f_rows + ramp[:, None] * g_rows
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    u = np.clip(u, lo, hi)
    return {"ctrl": u, "rows": np.array(rows), "n_contacts": n_contacts, "residual": resid, "fn_sum": fn_sum,
            "closure_fail": closure_fail, "offset_geo": geo, "offset_force": frc, "offset_ff": ff}
