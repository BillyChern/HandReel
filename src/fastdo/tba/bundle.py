"""R1: temporal object bundle adjustment with free anchor depths.

Unknowns, per window: the relative pose ``M_k`` (``X_k = M_k X_key``) of every frame with at least ``min_live``
visible anchored keyframe tracks (the keyframe itself is the identity, which fixes the rigid gauge), and the depth
``d_i`` of each anchored keyframe query along its keyframe ray (``Y_i = d_i r_i``). Frames with fewer live tracks
are not unknowns: no parallax information reaches them, and they keep BASE's pose.

Cost (all terms dimensionless, so the problem is EXACTLY invariant under the gauge g_beta, KA-G):
- tracks: ``sum_{k,i} w_ki |pi(K (R_k Y_i + t_k)) - u_ki|^2`` with Cauchy IRLS weights ``w = 1 / (1 + r^2 / c^2)``,
  one round per entry of ``gnc_c_px`` (default four rounds at the production solver's 4 px; config docstring);
- anchor prior: ``((d_i - d_i^B) / (prior_rel_sd d_i^B))^2``, centred at BASE's anchor;
- weak motion priors: second differences of the anchor-cloud centroid ``R_k ybar + t_k`` (sd ``acc_rel_sd`` times the
  window's median anchor depth) and of the frame-to-frame rotation (sd ``rot_acc_sd_deg``), over consecutive rows.

Lengths are normalised by the window's median BASE anchor depth before the solve and restored after it, and anchors
are parametrised by ``a_i = log(d_i / d_i^B)``: a window scaled by beta is then the same numerical problem.
The track scale-free gauge (all d_i and t_k times lambda) is fixed only softly by the prior; the stage's explicit
gauge (`gauge_k` or `gauge_consensus`) is applied afterwards. Each round is solved by Levenberg-Marquardt on the dense
normal equations (`_levenberg_marquardt`; the WIP's scipy TRF/lsmr took 43 s on a 75-row HOI4D window, this 3.5 s,
same optimum to 1e-6 relative cost).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from fastdo.tba.config import TbaConfig
from fastdo.tba.geometry import project, rotvec, rotvec_to_matrix

__all__ = ["BundleProblem", "BundleResult", "apply_log_gauge", "gauge_consensus", "gauge_k", "pair_depths",
           "solve_bundle"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundleProblem:
    """One window's BA inputs (camera-frame, metres; module docstring).

    Attributes:
        K: (3,3) intrinsics used by the BA.
        tracks: (T,J,2) tracked pixels.
        vis: (T,J) bool tracker visibility.
        key: Keyframe row.
        rays: (J,3) keyframe rays ``K^-1 [q, 1]`` (z = 1).
        anchored: (J,) bool: the ray hits BASE's posed template at the keyframe.
        d_base: (J,) BASE anchor depths (NaN where not anchored).
        M0: (T,4,4) BASE relative poses ``X_k = M0_k X_key`` (NaN rows where BASE has no pose).
    """

    K: np.ndarray
    tracks: np.ndarray
    vis: np.ndarray
    key: int
    rays: np.ndarray
    anchored: np.ndarray
    d_base: np.ndarray
    M0: np.ndarray


@dataclass(frozen=True)
class BundleResult:
    """The BA solution in BASE's (soft) gauge, before the stage's explicit gauge.

    Attributes:
        depths: (J,) anchor depths (NaN where not anchored).
        M: (T,4,4) relative poses; rows that are not BA unknowns are ``M0``'s.
        ba_frames: (T,) bool: rows whose pose the BA estimated (the keyframe included).
        weights: (T,J) final Cauchy weights at ``cauchy_c_px`` (0 where not observed or not a BA row).
        z_ref: Median BASE anchor depth (the normalising length).
        nfev: Function evaluations, summed over the GNC stages.
        cost_init: Robust track cost at BASE (c = cauchy_c_px), px^2 units.
        cost_final: The same at the solution.
        solved: False when the window had nothing to solve (fewer than 2 BA rows or 4 anchors): an identity result.
    """

    depths: np.ndarray
    M: np.ndarray
    ba_frames: np.ndarray
    weights: np.ndarray
    z_ref: float
    nfev: int
    cost_init: float
    cost_final: float
    solved: bool


def _ba_rows(p: BundleProblem, cfg: TbaConfig) -> np.ndarray:
    finite = np.isfinite(p.M0).all(axis=(1, 2))
    live = (p.vis & p.anchored[None, :]).sum(axis=1) >= cfg.min_live
    rows = finite & live
    rows[p.key] = True
    return rows


def _cauchy_cost(r2: np.ndarray, c: float) -> float:
    return float(np.sum(c * c * np.log1p(r2 / (c * c))))


class _Problem:
    """The normalised least-squares problem (closure over the fixed arrays)."""

    def __init__(self, p: BundleProblem, cfg: TbaConfig, rows: np.ndarray) -> None:
        self.p, self.cfg = p, cfg
        self.anc = np.flatnonzero(p.anchored)
        self.z_ref = float(np.median(p.d_base[self.anc]))
        self.free = np.flatnonzero(rows & (np.arange(len(rows)) != p.key))
        self.col_of_frame = {int(k): 6 * j for j, k in enumerate(self.free)}
        self.col_of_anchor = {int(i): 6 * len(self.free) + j for j, i in enumerate(self.anc)}
        self.n_x = 6 * len(self.free) + len(self.anc)
        obs = p.vis & p.anchored[None, :] & rows[:, None]
        obs[p.key] = False                              # the keyframe reprojects exactly onto its own queries
        self.ok, self.oi = np.nonzero(obs)
        self.uv = p.tracks[self.ok, self.oi].astype(np.float64)
        self.yb = (p.d_base[self.anc] / self.z_ref)[:, None] * p.rays[self.anc]       # normalised BASE anchors
        self.ybar = self.yb.mean(axis=0)
        self.R0 = p.M0[:, :3, :3].copy()
        self.t0 = p.M0[:, :3, 3] / self.z_ref
        self.R0[p.key], self.t0[p.key] = np.eye(3), 0.0
        trip = [k for k in range(1, len(rows) - 1) if rows[k - 1] and rows[k] and rows[k + 1]]
        self.trip = np.asarray(trip, np.int64)
        self.anc_pos = np.full(len(p.anchored), -1, np.int64)
        self.anc_pos[self.anc] = np.arange(len(self.anc))
        self.w = np.ones(len(self.ok))

    def x0(self) -> np.ndarray:
        x = np.zeros(self.n_x)
        for k, c in self.col_of_frame.items():
            x[c + 3:c + 6] = self.t0[k]
        return x

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R, t = self.R0.copy(), self.t0.copy()
        if len(self.free):
            blk = x[:6 * len(self.free)].reshape(-1, 6)
            R[self.free] = rotvec_to_matrix(blk[:, :3]) @ self.R0[self.free]
            t[self.free] = blk[:, 3:]
        a = x[6 * len(self.free):]
        return R, t, a

    def track_residual(self, x: np.ndarray) -> np.ndarray:
        R, t, a = self.unpack(x)
        Y = self.yb * np.exp(a)[:, None]
        X = np.einsum("nij,nj->ni", R[self.ok], Y[self.anc_pos[self.oi]]) + t[self.ok]
        z = np.where(X[:, 2] > 1e-6, X[:, 2], 1e-6)
        uv = project(np.concatenate([X[:, :2], z[:, None]], axis=1), self.p.K)
        return uv - self.uv                                                    # (n_obs, 2)

    def fun(self, x: np.ndarray) -> np.ndarray:
        R, t, a = self.unpack(x)
        parts = [(np.sqrt(self.w)[:, None] * self.track_residual(x)).ravel(),
                 np.expm1(a) / self.cfg.prior_rel_sd]
        if len(self.trip):
            k = self.trip
            c = np.einsum("nij,j->ni", R, self.ybar) + t
            parts.append(((c[k + 1] - 2.0 * c[k] + c[k - 1]) / self.cfg.acc_rel_sd).ravel())
            parts.append(self._rot_acc(R).ravel())
        return np.concatenate(parts)

    def _rot_acc(self, R: np.ndarray) -> np.ndarray:
        k = self.trip
        w1 = rotvec(R[k + 1] @ np.swapaxes(R[k], 1, 2))
        w0 = rotvec(R[k] @ np.swapaxes(R[k - 1], 1, 2))
        return (w1 - w0) / math.radians(self.cfg.rot_acc_sd_deg)

    def jac(self, x: np.ndarray) -> csr_matrix:
        """Analytic Jacobian of `fun` (track, prior and centroid rows exact; rotation-acceleration rows by central
        differences over three interleaved frame colours, each row touching one frame of each colour)."""
        R, t, a = self.unpack(x)
        n_f, n_obs, n_anc, n_t = len(self.free), len(self.ok), len(self.anc), len(self.trip)
        fcol = np.full(len(R), -1, np.int64)
        fcol[self.free] = 6 * np.arange(n_f)
        Jl = np.tile(np.eye(3), (len(R), 1, 1))
        if n_f:
            Jl[self.free] = _left_jacobian(x[:6 * n_f].reshape(-1, 6)[:, :3])
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def put(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
            r, c, v = np.broadcast_arrays(r, c, v)
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(v.ravel())

        # tracks
        Y = self.yb * np.exp(a)[:, None]
        Z = np.einsum("nij,nj->ni", R[self.ok], Y[self.anc_pos[self.oi]])
        X = Z + t[self.ok]
        z = np.where(X[:, 2] > 1e-6, X[:, 2], 1e-6)
        fx, fy = self.p.K[0, 0], self.p.K[1, 1]
        A = np.zeros((n_obs, 2, 3))
        A[:, 0, 0], A[:, 0, 2] = fx / z, -fx * X[:, 0] / z ** 2
        A[:, 1, 1], A[:, 1, 2] = fy / z, -fy * X[:, 1] / z ** 2
        A *= np.sqrt(self.w)[:, None, None]
        r_obs = 2 * np.arange(n_obs)[:, None] + np.arange(2)[None, :]          # (n_obs, 2)
        fc = fcol[self.ok]
        has = fc >= 0
        if has.any():
            dw = -np.einsum("nij,njk,nkl->nil", A[has], _skew(Z[has]), Jl[self.ok[has]])
            put(np.repeat(r_obs[has][:, :, None], 3, 2), fc[has, None, None] + np.arange(3)[None, None, :], dw)
            put(np.repeat(r_obs[has][:, :, None], 3, 2), fc[has, None, None] + 3 + np.arange(3)[None, None, :],
                A[has])
        acol = 6 * n_f + self.anc_pos[self.oi]
        put(r_obs, np.repeat(acol[:, None], 2, 1), np.einsum("nij,nj->ni", A, Z))
        # anchor prior
        base = 2 * n_obs
        put(base + np.arange(n_anc), 6 * n_f + np.arange(n_anc), np.exp(a) / self.cfg.prior_rel_sd)
        base += n_anc
        if n_t:
            k = self.trip
            sd = self.cfg.acc_rel_sd
            u = np.einsum("nij,j->ni", R, self.ybar)
            for coef, kk in ((1.0, k + 1), (-2.0, k), (1.0, k - 1)):
                c = fcol[kk]
                m = c >= 0
                if not m.any():
                    continue
                rr = base + 3 * np.flatnonzero(m)[:, None] + np.arange(3)[None, :]          # (n,3)
                dw = -coef / sd * np.einsum("nij,njk->nik", _skew(u[kk[m]]), Jl[kk[m]])
                put(np.repeat(rr[:, :, None], 3, 2), c[m, None, None] + np.arange(3)[None, None, :], dw)
                put(rr, c[m, None] + 3 + np.arange(3)[None, :], np.full(rr.shape, coef / sd))
            base_r = base + 3 * n_t
            h = 1e-6
            for colour in range(3):
                sel = self.free[self.free % 3 == colour]
                if not len(sel):
                    continue
                for comp in range(3):
                    xp, xm = x.copy(), x.copy()
                    xp[fcol[sel] + comp] += h
                    xm[fcol[sel] + comp] -= h
                    d = (self._rot_acc(self.unpack(xp)[0]) - self._rot_acc(self.unpack(xm)[0])) / (2 * h)
                    for off in (-1, 0, 1):
                        kk = k + off
                        m = np.isin(kk, sel)
                        if not m.any():
                            continue
                        rr = base_r + 3 * np.flatnonzero(m)[:, None] + np.arange(3)[None, :]
                        put(rr, np.repeat((fcol[kk[m]] + comp)[:, None], 3, 1), d[m])
        n_res = 2 * n_obs + n_anc + 6 * n_t
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_res, self.n_x)).tocsr()


def _skew(v: np.ndarray) -> np.ndarray:
    out = np.zeros((*v.shape[:-1], 3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def _left_jacobian(w: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian: exp(w + d) ~ exp(J_l(w) d) exp(w)."""
    th = np.linalg.norm(w, axis=-1)[:, None, None]
    W = _skew(w)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    c1 = np.where(small, 0.5, (1.0 - np.cos(ths)) / ths ** 2)
    c2 = np.where(small, 1.0 / 6.0, (ths - np.sin(ths)) / ths ** 3)
    return np.eye(3)[None] + c1 * W + c2 * (W @ W)


def _levenberg_marquardt(fun, jac, x: np.ndarray, max_iter: int, ftol: float = 1e-10,
                         xtol: float = 1e-10) -> tuple[np.ndarray, int]:
    """Levenberg-Marquardt on ``0.5 |fun(x)|^2`` with the sparse Jacobian's normal equations solved densely.

    The unknowns are a few hundred (6 per BA row + 1 per anchor), so ``J^T J`` is small and dense while ``J`` has
    thousands of rows: a Cholesky solve per step is ~100x cheaper than scipy's iterative trust-region (lsmr) on a
    75-row window. Marquardt damping ``A + mu diag(A)`` (diag floored), mu x1/3 on an accepted step and x2 on a
    rejected one. Deterministic. Returns (x, residual evaluations).
    """
    f = fun(x)
    cost = 0.5 * float(f @ f)
    nfev, mu = 1, None
    for _ in range(max_iter):
        J = jac(x)
        A = (J.T @ J).toarray()
        g = J.T @ f
        d = np.maximum(np.diag(A), 1e-12 * max(float(np.diag(A).max()), 1e-300))
        if mu is None:
            mu = 1e-3
        improved = False
        for _ in range(30):
            try:
                step = -np.linalg.solve(A + mu * np.diag(d), g)
            except np.linalg.LinAlgError:
                mu *= 4.0
                continue
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1
            c_new = 0.5 * float(f_new @ f_new)
            if np.isfinite(c_new) and c_new <= cost:
                rel = (cost - c_new) / max(cost, 1e-300)
                small = float(np.linalg.norm(step)) <= xtol * (float(np.linalg.norm(x)) + xtol)
                x, f, cost = x_new, f_new, c_new
                mu = max(mu / 3.0, 1e-12)
                improved = True
                if rel < ftol or small:
                    return x, nfev
                break
            mu *= 2.0
        if not improved:
            break
    return x, nfev


def solve_bundle(p: BundleProblem, cfg: TbaConfig) -> BundleResult:
    """R1 on one window (module docstring). Deterministic; CPU only."""
    rows = _ba_rows(p, cfg)
    n_t, n_j = p.vis.shape
    identity = BundleResult(depths=np.where(p.anchored, p.d_base, np.nan), M=p.M0.copy(), ba_frames=rows,
                            weights=np.zeros((n_t, n_j)), z_ref=float("nan"), nfev=0, cost_init=float("nan"),
                            cost_final=float("nan"), solved=False)
    if int(rows.sum()) < 2 or int(p.anchored.sum()) < 4:
        return identity
    prob = _Problem(p, cfg, rows)
    if len(prob.ok) == 0:
        return identity
    x = prob.x0()
    c_fin = cfg.cauchy_c_px
    r2 = (prob.track_residual(x) ** 2).sum(axis=1)
    cost_init = _cauchy_cost(r2, c_fin)
    nfev = 0
    for c in (cfg.gnc_c_px if cfg.robust else (c_fin,)):
        r2 = (prob.track_residual(x) ** 2).sum(axis=1)
        prob.w = 1.0 / (1.0 + r2 / (c * c)) if cfg.robust else np.ones_like(r2)
        x, n = _levenberg_marquardt(prob.fun, prob.jac, x, cfg.max_nfev)
        nfev += n
    r2 = (prob.track_residual(x) ** 2).sum(axis=1)
    weights = np.zeros((n_t, n_j))
    weights[prob.ok, prob.oi] = 1.0 / (1.0 + r2 / (c_fin * c_fin))
    R, t, a = prob.unpack(x)
    M = p.M0.copy()
    M[rows, :3, :3] = R[rows]
    M[rows, :3, 3] = t[rows] * prob.z_ref
    M[rows, 3] = (0.0, 0.0, 0.0, 1.0)
    depths = np.full(n_j, np.nan)
    depths[prob.anc] = p.d_base[prob.anc] * np.exp(a)
    return BundleResult(depths=depths, M=M, ba_frames=rows, weights=weights, z_ref=prob.z_ref, nfev=nfev,
                        cost_init=cost_init, cost_final=_cauchy_cost(r2, c_fin), solved=True)


def pair_depths(M: np.ndarray, depths: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """(T,J) camera z of anchor ``i`` in row ``k`` under relative poses ``M`` (NaN where undefined)."""
    Y = depths[:, None] * rays
    X = np.einsum("kij,nj->kni", M[:, :3, :3], Y) + M[:, None, :3, 3]
    return X[..., 2]


def _pairs(p: BundleProblem, res: BundleResult) -> np.ndarray:
    """Gauge voters ("median over (k, live i)"): every LIVE pair, i.e. a visible track of an
    anchored keyframe query, in a row whose pose the BA estimated (the keyframe included). No weight or residual
    filter: the median is the registered robustness, and a filter computed at the estimate would be one more selector
    for the displaced-selector test. An unsolved window votes with its keyframe only, whose ratio is exactly 0."""
    return p.vis & p.anchored[None, :] & res.ba_frames[:, None]


def gauge_k(p: BundleProblem, res: BundleResult, d_ref: np.ndarray | None = None) -> float:
    """BASE's window gauge held: ``l_K = median over (k, live i) of log(z_BASE / z_BA)`` (0 if no pair).

    ``z_BASE`` is anchor i's camera depth in row k under BASE: its template hit at the keyframe (`d_ref`; default
    `p.d_base`, the prior centre, which IS the hit unless a mutation moved it) carried by BASE's relative pose
    ``M0_k``; ``z_BA`` the same under the BA.
    """
    m = _pairs(p, res)
    ref = p.d_base if d_ref is None else np.asarray(d_ref, np.float64)
    zb = pair_depths(p.M0, np.where(p.anchored, ref, np.nan), p.rays)
    za = pair_depths(res.M, res.depths, p.rays)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.log(zb[m] / za[m])
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else 0.0


def gauge_consensus(p: BundleProblem, res: BundleResult, z_obs: np.ndarray) -> float:
    """The draft's pointmap-consensus gauge (TBA-V-C): ``median log(z_obs / z_BA)`` over live pairs with a reading."""
    m = _pairs(p, res) & np.isfinite(z_obs)
    za = pair_depths(res.M, res.depths, p.rays)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.log(z_obs[m] / za[m])
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else 0.0


def apply_log_gauge(res: BundleResult, ell: float) -> BundleResult:
    """g_exp(ell) of a BA solution: depths and BA-row translations times exp(ell) (non-BA rows are BASE's)."""
    lam = math.exp(float(ell))
    M = res.M.copy()
    M[res.ba_frames, :3, 3] *= lam
    return BundleResult(depths=res.depths * lam, M=M, ba_frames=res.ba_frames, weights=res.weights,
                        z_ref=res.z_ref, nfev=res.nfev, cost_init=res.cost_init, cost_final=res.cost_final,
                        solved=res.solved)
