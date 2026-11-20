"""D4': exact-engine finite-difference Gauss-Newton / Levenberg-Marquardt on DO AS I DO's knot vector.

Variables. z in R^{K x nu}, u_s(z) = u0_s + sum_k phi_k(s) sigma (.) z_k: K = ceil(T / 40) knots, phi = linear
interpolation with align_corners (DO AS I DO's `interp`), sigma = DO AS I DO's knot noise per actuator (units).

Least-squares structure.
J(z) = sum_s m_s [a_s + beta b_s + kappa h_s^2] + J_binary, with a_s = ||w (.) dq_s||, b_s = ||dv_s||.
The norms are not squares, so each GN iteration k minimises the MM majoriser (IRLS):
  ||e|| <= ||e||^2 / (2 f) + f / 2 for every f > 0 (equality iff f = ||e||),
with f = max(||e(z_k)||, floor). The residual vector
  r_s = [ sqrt(m_s / f_a,s) w (.) dq_s ; sqrt(m_s beta / f_b,s) dv_s ; sqrt(2 m_s kappa) h_s ]
gives Q_k(z) = (||r(z)||^2 + sum_s m_s (f_a,s + beta f_b,s)) / 2 >= J_smooth(z), with equality and equal gradient
(J^T r = grad J_smooth) at z_k when the floors are inactive. (Erratum, 2026-10-05T03:50Z: the stage-0 runs used
sqrt(m_s kappa) h_s, which halves the penetration term in the GN model; see the result doc, section 9.) Gauss-Newton: r(z_k + d) ~ r_k + J_k d with J_k = dr/dz from chained exact-engine transition Jacobians
(fd_linearize) and analytic row Jacobians; J^T J is never formed, products J v and J^T w are a forward-tangent
and a reverse-adjoint pass over the stored (A_s, B_s).
Step. LM: (J^T J + lambda I) d = -J^T r by conjugate gradients (cap `cg_max_iters`, relative residual `cg_rtol`).
Gain ratio rho = (J(z_k) - J(z_k + d)) / pred, pred = (||r_k||^2 - ||r_k + J d||^2) / 2.
Acceptance is NEVER-WORSE on the exact objective from a full executor rollout: accept iff J(z_k + d) < J(z_k)
(then lambda *= max(1/3, 1 - (2 rho - 1)^3), nu = 2); else lambda *= nu, nu *= 2 and re-solve on the same
linearisation. lambda_0 = lambda0_scale * ||J g||^2 / ||g||^2 (curvature along the gradient g = J^T r).
Stopping. S1 `max_iters` linearisations; S2 `stall_checks` consecutive accepted iterations each with relative
decrease < `rel_tol`; S3 `max_trials` consecutive rejections in one iteration (stalled); S4 an optional caller
callback (e.g. a certificate). Every iterate is a full executor rollout, so the returned trajectory is the certified
CPU float64 replay of the returned controls.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fastdo.physics.fd_linearize import FDPool, Linearization, linearize
from fastdo.physics.fd_problem import ExecutorSpec, RewardSpec, Rollout, StepCosts, dq_obj_rot, rollout, step_costs
from fastdo.physics.planner_config import D4PrimeConfig

__all__ = ["GNModel", "IterRecord", "KnotBasis", "LMResult", "conjugate_gradient", "knot_sigma", "optimize"]


class KnotBasis:
    """Linear knot interpolation, F.interpolate(mode='linear', align_corners=True) of K knots onto K * n steps,
    truncated to T rows."""

    def __init__(self, n_steps: int, knot_steps: int) -> None:
        self.T = int(n_steps)
        self.K = max(2, int(math.ceil(self.T / knot_steps)))
        L = self.K * knot_steps
        p = np.arange(self.T) * (self.K - 1) / (L - 1)
        k = np.minimum(np.floor(p).astype(int), self.K - 2)
        self.k = k
        self.alpha = (p - k)[:, None]

    def apply(self, theta: np.ndarray) -> np.ndarray:
        return (1.0 - self.alpha) * theta[self.k] + self.alpha * theta[self.k + 1]

    def adjoint(self, g: np.ndarray) -> np.ndarray:
        out = np.zeros((self.K, g.shape[1]))
        np.add.at(out, self.k, (1.0 - self.alpha) * g)
        np.add.at(out, self.k + 1, self.alpha * g)
        return out


def knot_sigma(nu: int, cfg: D4PrimeConfig) -> np.ndarray:
    s = np.full(nu, cfg.sigma_joint)
    s[0:3] = cfg.sigma_pos
    s[3:6] = cfg.sigma_rot
    return s


class GNModel:
    """IRLS residual r(z_k) and matrix-free J v / J^T w around one nominal rollout and its linearisation."""

    def __init__(self, ex: ExecutorSpec, rw: RewardSpec, ro: Rollout, sc: StepCosts, lin: Linearization,
                 basis: KnotBasis, sigma: np.ndarray, cfg: D4PrimeConfig) -> None:
        T, nv = sc.dq.shape
        self.T, self.nv, self.basis, self.sigma = T, nv, basis, sigma
        self.A, self.B = lin.A, lin.B
        nr = ex.obj_qadr
        cq = np.sqrt(sc.mult / np.maximum(sc.a, cfg.a_floor))
        cv = np.sqrt(sc.mult * rw.vel_scale / np.maximum(sc.b, cfg.b_floor))
        cp = np.sqrt(2.0 * sc.mult * rw.pen_scale)       # Q = ||r||^2 / 2 must carry m kappa h^2 in full
        # d dq / d q-tangent: identity except the object rotation block
        D = np.broadcast_to(np.eye(nv), (T, nv, nv)).copy()
        a0 = ex.obj_qadr
        ref_q = rw.qpos_ref[ro.s0 + 1:ro.s0 + T + 1, a0 + 3:a0 + 7]
        D[:, nr + 3:nr + 6, nr + 3:nr + 6] = dq_obj_rot(ro.qpos[:, a0 + 3:a0 + 7], ref_q)
        self.Mq = (cq[:, None] * rw.w_dq[None, :])[:, :, None] * D                # (T, nv, nv) on x_{s+1}
        self.cv = cv                                                              # on qvel of x_{s+1}
        P = np.zeros((T, nv))                                                     # pen rows on x_s
        for s, g in ro.pen_grad.items():
            if sc.h[s] > 0.0:
                P[s] = cp[s] * g
        self.Mp = P
        self.c = (cq, cv, cp)
        self.r = (cq[:, None] * rw.w_dq[None, :] * sc.dq, cv[:, None] * sc.dv, cp * sc.h)

    @staticmethod
    def dot(x: tuple, y: tuple) -> float:
        return float(sum(np.vdot(a, b) for a, b in zip(x, y)))

    def jv(self, z: np.ndarray) -> tuple:
        du = self.basis.apply(z * self.sigma)
        Bdu = np.einsum("tij,tj->ti", self.B, du)
        T, nx = Bdu.shape
        X = np.zeros((T + 1, nx))
        A = self.A
        dx = np.zeros(nx)
        for s in range(T):
            dx = A[s] @ dx + Bdu[s]
            X[s + 1] = dx
        nv = self.nv
        out_q = np.einsum("tij,tj->ti", self.Mq, X[1:, :nv])
        out_v = self.cv[:, None] * X[1:, nv:2 * nv]
        out_p = np.einsum("ti,ti->t", self.Mp, X[:-1, :nv])
        return out_q, out_v, out_p

    def jtw(self, w: tuple) -> np.ndarray:
        wq, wv, wp = w
        nv = self.nv
        T, nx = self.T, self.A.shape[1]
        inj_next = np.zeros((T, nx))
        inj_next[:, :nv] = np.einsum("tij,ti->tj", self.Mq, wq)
        inj_next[:, nv:2 * nv] = self.cv[:, None] * wv
        inj_cur = self.Mp * wp[:, None]
        G = np.zeros((T, nx))
        lam = np.zeros(nx)
        A = self.A
        for s in range(T - 1, -1, -1):
            lam = lam + inj_next[s]
            G[s] = lam
            lam = A[s].T @ lam
            lam[:nv] += inj_cur[s]
        gu = np.einsum("tij,ti->tj", self.B, G)
        return self.basis.adjoint(gu) * self.sigma


def conjugate_gradient(matvec: Callable[[np.ndarray], np.ndarray], b: np.ndarray, max_iters: int,
                       rtol: float) -> tuple[np.ndarray, int]:
    """CG for an SPD operator; returns (x, iterations)."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(np.vdot(r, r))
    stop = (rtol ** 2) * rr
    it = 0
    for it in range(1, max_iters + 1):
        Ap = matvec(p)
        pAp = float(np.vdot(p, Ap))
        if pAp <= 0.0:
            break
        a = rr / pAp
        x += a * p
        r -= a * Ap
        rr_new = float(np.vdot(r, r))
        if rr_new <= stop:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, it


@dataclass
class IterRecord:
    it: int
    J: float
    accepted: bool
    trials: int
    lam: float
    rho: float
    pred: float
    cg_iters: int
    t_rollout_s: float
    t_fd_wall_s: float
    t_fd_cpu_s: float
    t_solve_s: float
    fd_steps: int
    rollout_steps: int
    extra: dict = field(default_factory=dict)


@dataclass
class LMResult:
    z: np.ndarray
    ctrls: np.ndarray
    rollout: Rollout
    costs: StepCosts
    history: list[IterRecord]
    stop: str
    J0: float


def optimize(ex: ExecutorSpec, rw: RewardSpec, u0: np.ndarray, cfg: D4PrimeConfig, pool: FDPool,
             callback: Callable[[int, Rollout, StepCosts], bool] | None = None,
             log: Callable[[IterRecord], None] | None = None) -> LMResult:
    """Run D4' from the warm start `u0` (T, nu). `callback(it, rollout, costs)` returning True stops (S4)."""
    if not cfg.enabled:
        raise RuntimeError("D4' is opt-in: pass D4PrimeConfig(enabled=True)")
    T, nu = u0.shape
    basis = KnotBasis(T, cfg.knot_steps)
    sigma = knot_sigma(nu, cfg)
    z = np.zeros((basis.K, nu))
    ctrl = lambda zz: u0 + basis.apply(zz * sigma)  # noqa: E731
    t0 = time.perf_counter()
    ro = rollout(ex, rw, ctrl(z), terminate=False)
    sc = step_costs(ex, rw, ro)
    t_ro = time.perf_counter() - t0
    J0 = sc.total
    hist: list[IterRecord] = []
    lam, nu_f = None, 2.0
    small = 0
    stop = "max_iters"
    if callback is not None and callback(0, ro, sc):
        return LMResult(z, ctrl(z), ro, sc, hist, "callback", J0)
    if not math.isfinite(J0):
        return LMResult(z, ctrl(z), ro, sc, hist, "nonfinite_start", J0)
    for it in range(1, cfg.max_iters + 1):
        lin = linearize(pool, ro, cfg.fd_eps, cfg.fd_centered)
        t1 = time.perf_counter()
        gm = GNModel(ex, rw, ro, sc, lin, basis, sigma, cfg)
        g = gm.jtw(gm.r)
        if lam is None:
            jg = gm.jv(g)
            lam = cfg.lambda0_scale * gm.dot(jg, jg) / max(float(np.vdot(g, g)), 1e-300)
        rr = gm.dot(gm.r, gm.r)
        t_solve, t_roll, rsteps, cg_total = time.perf_counter() - t1, 0.0, 0, 0
        accepted, rho, pred, trials = False, float("nan"), float("nan"), 0
        for trials in range(1, cfg.max_trials + 1):
            t2 = time.perf_counter()
            mv = lambda v: gm.jtw(gm.jv(v)) + lam * v  # noqa: E731,B023
            d, cg_it = conjugate_gradient(mv, -g, cfg.cg_max_iters, cfg.cg_rtol)
            cg_total += cg_it
            jd = gm.jv(d)
            pred = 0.5 * (rr - gm.dot(tuple(a + b for a, b in zip(gm.r, jd)), tuple(a + b for a, b in zip(gm.r, jd))))
            t_solve += time.perf_counter() - t2
            t3 = time.perf_counter()
            ro_new = rollout(ex, rw, ctrl(z + d), terminate=False)
            sc_new = step_costs(ex, rw, ro_new)
            t_roll += time.perf_counter() - t3
            rsteps += ro_new.qpos.shape[0]
            actual = sc.total - sc_new.total
            rho = actual / pred if pred > 0 else float("nan")
            if math.isfinite(sc_new.total) and actual > 0.0:
                accepted = True
                rel = actual / max(sc.total, 1e-300)
                z, ro, sc = z + d, ro_new, sc_new
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * min(rho, 1.0) - 1.0) ** 3) if math.isfinite(rho) else 1.0
                nu_f = 2.0
                small = small + 1 if rel < cfg.rel_tol else 0
                break
            lam *= nu_f
            nu_f *= 2.0
        rec = IterRecord(it=it, J=sc.total, accepted=accepted, trials=trials, lam=float(lam), rho=float(rho),
                         pred=float(pred), cg_iters=cg_total, t_rollout_s=t_roll, t_fd_wall_s=lin.wall_s,
                         t_fd_cpu_s=lin.cpu_s, t_solve_s=t_solve, fd_steps=lin.fd_steps, rollout_steps=rsteps)
        hist.append(rec)
        if log is not None:
            log(rec)
        if not accepted:
            stop = "stalled"
            break
        if callback is not None and callback(it, ro, sc):
            stop = "callback"
            break
        if small >= cfg.stall_checks:
            stop = "converged"
            break
    return LMResult(z=z, ctrls=ctrl(z), rollout=ro, costs=sc, history=hist, stop=stop, J0=J0)
