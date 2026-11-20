"""Covariance-weighted robust Frechet mean on R^3 x SO(3)/G and the label-free drift test (design 4, R1.4-R1.5).

Objective: F(M) = sum_i rho_k(e_i(M)), e_i^2 = min_g r_i(M; g)^T Sigma_i^-1 r_i(M; g), r_i = [p_i - pbar;
Log(O_i S_g Obar^T)], rho_k Huber. (Whitening the rotation residual through J_l^-1(theta_i), as R1.4 asks, leaves e_i
unchanged: J_l(theta) theta = theta. It changes only the Jacobian, which uses J_r^-1.) Gauss-Newton on the update
pbar += dp, Obar <- Exp(dphi) Obar with dr/d(dp, dphi) = -blockdiag(I, J_r^-1(theta_i)), IRLS Huber weights frozen
per step, and backtracking: a step is accepted only if F (representatives re-chosen at the trial point) decreases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastdo.reference.config import ReferenceConfig
from fastdo.reference.so3 import angle_deg, exp, jr_inv, log, project

__all__ = ["DriftTest", "MeanResult", "alpha_from_pool", "ctrl_mean", "drift_test", "floor_covariance",
           "quotient_angle_deg", "residuals", "robust_mean"]


@dataclass(frozen=True)
class MeanResult:
    p: np.ndarray            # (3,)
    O: np.ndarray            # (3,3)
    labels: np.ndarray       # (n,) representative index per row
    e: np.ndarray            # (n,) whitened residual norm
    w: np.ndarray            # (n,) Huber weight
    r: np.ndarray            # (n,6) residuals with the chosen representatives
    H: np.ndarray            # (6,6) information at the solution (weights frozen)
    iterations: int
    converged: bool
    reason: str              # "" | "gn_failed:<why>"
    F_history: tuple[float, ...]

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.H)


def floor_covariance(cfg: ReferenceConfig) -> np.ndarray:
    return np.diag([cfg.floor_pos_m ** 2] * 3 + [np.radians(cfg.floor_rot_deg) ** 2] * 3)


def residuals(p: np.ndarray, O: np.ndarray, pbar: np.ndarray, Obar: np.ndarray, elements: np.ndarray
              ) -> np.ndarray:
    """`(n,K,6)` residuals of every row under every representative."""
    rp = np.asarray(p) - pbar
    X = np.einsum("nij,kjl,ml->nkim", O, elements, Obar)      # O_i S_g Obar^T
    th = log(X)
    return np.concatenate([np.broadcast_to(rp[:, None, :], th.shape), th], axis=-1)


def _rho(e: np.ndarray, k: float) -> np.ndarray:
    """Huber loss (convex; R1.4)."""
    if not np.isfinite(k):
        return 0.5 * e * e
    return np.where(e <= k, 0.5 * e * e, k * e - 0.5 * k * k)


def _weight(e: np.ndarray, k: float) -> np.ndarray:
    """IRLS weight of the Huber loss: 1 inside k, k/e outside."""
    return np.where(e <= k, 1.0, k / np.maximum(e, 1e-300))


def _evaluate(p, O, Lam, pbar, Obar, elements, k):
    R = residuals(p, O, pbar, Obar, elements)                  # (n,K,6)
    e2 = np.einsum("nki,nij,nkj->nk", R, Lam, R)
    lab = np.argmin(e2 + 1e-15 * np.arange(e2.shape[1]), axis=1)
    e = np.sqrt(np.maximum(e2[np.arange(len(p)), lab], 0.0))
    return float(_rho(e, k).sum()), lab, e, R[np.arange(len(p)), lab]


def quotient_angle_deg(O: np.ndarray, Obar: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """`(n,)` min over g of angle(O_i S_g, Obar)."""
    X = np.einsum("nij,kjl->nkil", O, elements)
    return angle_deg(X, Obar[None, None]).min(axis=1)


def _medoid(O: np.ndarray, elements: np.ndarray) -> np.ndarray:
    n = len(O)
    tot = np.zeros(n)
    for j in range(n):
        tot[j] = quotient_angle_deg(O, O[j], elements).sum()
    return O[int(np.argmin(tot))]


def robust_mean(p: np.ndarray, O: np.ndarray, Sigma: np.ndarray, elements: np.ndarray, cfg: ReferenceConfig,
                huber: bool = True) -> MeanResult:
    """The robust mean of the module docstring (`huber=False`: plain weighted least squares; a mutation hook)."""
    p, O = np.asarray(p, dtype=np.float64), np.asarray(O, dtype=np.float64)
    n = len(p)
    Lam = np.linalg.inv(np.asarray(Sigma, dtype=np.float64))
    k = cfg.huber_k if huber else np.inf
    pbar, Obar = np.median(p, axis=0), _medoid(O, elements)
    F, lab, e, r = _evaluate(p, O, Lam, pbar, Obar, elements, k)
    hist = [F]
    H = np.eye(6)
    for it in range(1, cfg.gn_max_iter + 1):
        w = _weight(e, k)
        J = np.zeros((n, 6, 6))
        J[:, :3, :3] = np.eye(3)
        J[:, 3:, 3:] = jr_inv(r[:, 3:])
        JL = np.einsum("nji,njk->nik", J, Lam)
        H = np.einsum("n,nij,njk->ik", w, JL, J)
        g = np.einsum("n,nij,nj->i", w, JL, r)
        if not np.isfinite(H).all() or np.linalg.cond(H) > cfg.gn_max_cond:
            return MeanResult(pbar, Obar, lab, e, w, r, H, it, False, "gn_failed:ill_conditioned", tuple(hist))
        delta = np.linalg.solve(H, g)
        if not np.isfinite(delta).all():
            return MeanResult(pbar, Obar, lab, e, w, r, H, it, False, "gn_failed:non_finite_step", tuple(hist))
        step = 1.0
        for _ in range(cfg.gn_max_halvings + 1):
            p_try = pbar + step * delta[:3]
            O_try = exp(step * delta[3:]) @ Obar
            F_try, lab_t, e_t, r_t = _evaluate(p, O, Lam, p_try, O_try, elements, k)
            if F_try < F:
                break
            step *= 0.5
        else:
            grad = float(np.sqrt(g @ np.linalg.solve(H, g)))
            if grad < 1e-9 or np.linalg.norm(delta) < cfg.gn_tol:
                return MeanResult(pbar, Obar, lab, e, w, r, H, it, True, "", tuple(hist))
            return MeanResult(pbar, Obar, lab, e, w, r, H, it, False, "gn_failed:no_descent", tuple(hist))
        pbar, Obar, F, lab, e, r = p_try, project(O_try), F_try, lab_t, e_t, r_t
        hist.append(F)
        if np.linalg.norm(step * delta[:3]) < cfg.gn_tol and np.linalg.norm(step * delta[3:]) < cfg.gn_tol:
            w = _weight(e, k)
            J[:, 3:, 3:] = jr_inv(r[:, 3:])
            JL = np.einsum("nji,njk->nik", J, Lam)
            H = np.einsum("n,nij,njk->ik", w, JL, J)
            return MeanResult(pbar, Obar, lab, e, w, r, H, it, True, "", tuple(hist))
    w = _weight(e, k)
    return MeanResult(pbar, Obar, lab, e, w, r, H, cfg.gn_max_iter, False, "gn_failed:max_iterations", tuple(hist))


@dataclass(frozen=True)
class DriftTest:
    p_trend: float
    p_half: float | None
    trend_stat: float
    half_stat: float | None
    half_disc_mm: float | None      # absolute half-mean discrepancy (reported only)
    half_disc_deg: float | None

    def refuse(self, alpha: float) -> bool:
        return self.p_trend <= alpha or (self.p_half is not None and self.p_half <= alpha)


def drift_test(times: np.ndarray, r: np.ndarray, w: np.ndarray, cfg: ReferenceConfig, seed: int) -> DriftTest:
    """REV 1 (design R1.5), SUPERSEDED in rev 2 by `fastdo.reference.motion` (no magnitude floor; kept only so the
    Stage A record can be reproduced). Permutation trend and split-half tests on floor-whitened, Huber-clipped
    residuals."""
    tau = np.asarray(times, dtype=np.float64)
    n = len(tau)
    scale = np.array([cfg.floor_pos_m] * 3 + [np.radians(cfg.floor_rot_deg)] * 3)
    z = np.asarray(w)[:, None] * np.asarray(r) / scale
    c = tau - tau.mean()
    den = float(c @ c) if float(c @ c) > 0 else 1.0
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(n) for _ in range(cfg.drift_permutations)])
    zp = z[perms]                                               # (B,n,6)
    t_obs = float(np.sum((c @ z) ** 2) / den)
    t_perm = np.sum(np.einsum("n,bnd->bd", c, zp) ** 2, axis=1) / den
    p_t = (1.0 + float(np.sum(t_perm >= t_obs - 1e-12))) / (cfg.drift_permutations + 1.0)
    p_h = h_obs = dmm = ddeg = None
    if n >= cfg.split_half_min_vote:
        h = n // 2
        h_obs = float(np.sum((z[:h].mean(0) - z[n - h:].mean(0)) ** 2))
        h_perm = np.sum((zp[:, :h].mean(1) - zp[:, n - h:].mean(1)) ** 2, axis=1)
        p_h = (1.0 + float(np.sum(h_perm >= h_obs - 1e-12))) / (cfg.drift_permutations + 1.0)
        rr = np.asarray(r)
        dmm = float(np.linalg.norm(rr[:h, :3].mean(0) - rr[n - h:, :3].mean(0)) * 1e3)
        ddeg = float(np.degrees(np.linalg.norm(rr[:h, 3:].mean(0) - rr[n - h:, 3:].mean(0))))
    return DriftTest(p_t, p_h, t_obs, h_obs, dmm, ddeg)


def ctrl_mean(p: np.ndarray, O: np.ndarray, labels: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CTRL estimator: coordinate-wise median of the centres; SVD-projected arithmetic mean of the
    representative rotations O_i S_{g_i}."""
    rep = np.einsum("nij,njk->nik", O, elements[np.asarray(labels)])
    return np.median(p, axis=0), project(rep.mean(axis=0))


def alpha_from_pool(segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], cfg: ReferenceConfig
                    ) -> tuple[float, int]:
    """Label-free alpha_D (R1.4): over segments `(p, O, Sigma_w, elements)` with n >= min_vote, the floor-only robust
    mean, then median over rows of (n/(n-1)) r^T Sigma_w^-1 r / chi2_6 median. Returns (alpha, rows); alpha = 1 with
    fewer than 20 rows."""
    q = []
    S0 = floor_covariance(cfg)
    for p, O, Sw, el in segments:
        n = len(p)
        if n < cfg.min_vote:
            continue
        m = robust_mean(p, O, np.broadcast_to(S0, (n, 6, 6)), el, cfg)
        Lw = np.linalg.inv(Sw)
        q.extend((n / (n - 1.0)) * np.einsum("ni,nij,nj->n", m.r, Lw, m.r))
    if len(q) < 20:
        return 1.0, len(q)
    return float(np.median(q) / cfg.chi2_6_median), len(q)
