"""Error-state iterated RTS smoother on SE(3) x R^6 (+ an optional hand point) for the pose smoother.

Exact model:

- State per row k: the object's CENTROID-frame pose T_k = (R_k, c_k) in SE(3) (camera frame; c_k is the posed
  template centroid) and its body twist xi_k = (omega_k, v_k) in R^6; optionally the hand wrist p_k, u_k in R^3 x R^3.
- Process (constant twist, integrated white noise on the twist): T_{k+1} = T_k Exp(D_k xi_k + a_k),
  xi_{k+1} = xi_k + b_k, (a_k, b_k) ~ N(0, Q_k), Q_k = [[D^3/3, D^2/2], [D^2/2, D]] (x) diag(q_w^2 I3, q_v^2 I3);
  the hand point is the same model on R^3 with q_h.
- Measurements (BASE's row): rotation  r_R = Log(R_k^T R^m_k) ~ N(0, s_R^2 I3) (body frame);
  centre r_c = c^m_k - c_k ~ N(0, S_k), S_k = s_par_k^2 u u^T + s_perp_k^2 (I - u u^T), u = c^m_k / |c^m_k|,
  s_par_k = a_par |c^m_k|, s_perp_k = a_perp |c^m_k| (ANISOTROPIC, camera frame, relative to depth);
  hand r_p = p^m_k - p_k ~ N(0, S^h_k) with (a_par_h, a_perp_h).
- Grasp rigidity (contact rows only, production flags): g_k = R_k v_k - u_k - (R_k omega_k) x (c_k - p_k) ~ N(0, q_c^2 I3).
- Robust: Huber on each block's per-dof Mahalanobis RMS, k = `huber_k`, by IRLS (`reweights` rounds).

Each Gauss-Newton step linearises about the current trajectory with body perturbations T = Tbar Exp(delta),
xi = xibar + beta, and runs one forward Kalman pass and one backward RTS pass on the error state. The transition is
delta_{k+1} = Ad(Gbar_k^-1) delta_k + J_r(phibar_k)(D_k beta_k + D_k xibar_k - phibar_k + a_k), with
Gbar_k = Tbar_k^-1 Tbar_{k+1}, phibar_k = Log(Gbar_k): the covariance is ADJOINT-TRANSPORTED. One KF + RTS pass solves
the linearised MAP exactly, so the iteration is Gauss-Newton on the MAP (Bell 1994).

Noise reference: the centre covariance of row k is evaluated at the CURRENT ESTIMATE's ray and at the
window's median measured range zbar, never at row k's own measurement. Self-weighting (variance from |c^m_k|, ray
from c^m_k) gives nearer-than-true rows more weight and pulls the window toward the camera by ~1% at the screening
noise level (measured on synthetic windows; `noise_ref="measured"` keeps it as the mutation).

Gauge: the smoother reads no metric source, so it must not move the window's gauge. After the
solve, gamma = log(sum_k w_k |chat_k| / sum_k w_k |c^m_k|) (w = final centre-block Huber weights over measured rows)
and every centre moves along its own ray by exp(-gamma); the hand likewise with its own gamma. All noise scales are
proportional to zbar, so the estimator is exactly equivariant under g_lambda (KA-G).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastdo.smooth.lie import (
    se3_adjoint,
    se3_exp,
    se3_jr,
    se3_log,
    so3_exp,
    so3_hat,
    so3_jl_inv,
    so3_jr,
    so3_log,
)

__all__ = ["Noise", "SmoothResult", "smooth_se3"]


@dataclass(frozen=True)
class Noise:
    """Absolute noise of one window (metres, radians, per row); built by `fastdo.smooth.config` from depth."""

    s_rot: float
    a_par: float
    a_perp: float
    q_w: float
    q_v: float
    a_par_h: float = float("nan")
    a_perp_h: float = float("nan")
    q_h: float = float("nan")
    q_c: float = float("nan")
    huber_k: float = 1.345
    reweights: int = 3
    gn_iters: int = 8
    diffuse: float = 1e3
    noise_ref: str = "estimate"
    process: str = "se3"


@dataclass(frozen=True)
class SmoothResult:
    R: np.ndarray            # (T, 3, 3)
    c: np.ndarray            # (T, 3) gauge-held centres
    hand: np.ndarray | None  # (T, 3) gauge-held wrists, or None
    gamma: float             # removed object gauge component (log), before the hold
    gamma_hand: float
    w_c: np.ndarray          # (T,) final centre-block weights (1 = inlier; NaN = no measurement)
    w_R: np.ndarray
    n_gn: int
    w_h: np.ndarray | None = None   # final wrist-block weights (NaN = no measurement), None without a hand
    w_g: np.ndarray | None = None   # final grasp-factor weights (NaN = no factor on that row)


def _aniso(u: np.ndarray, s_par: float, s_perp: float) -> np.ndarray:
    P = np.outer(u, u)
    return s_par ** 2 * P + s_perp ** 2 * (np.eye(3) - P)


def _fill(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Rows without a measurement copy the nearest measured row (initialisation only)."""
    idx = np.flatnonzero(ok)
    near = idx[np.abs(np.arange(len(x))[:, None] - idx[None, :]).argmin(axis=1)]
    return x[near].copy()


def _q_block(D: float, q: np.ndarray) -> np.ndarray:
    n = len(q)
    Qc = np.diag(q ** 2)
    out = np.zeros((2 * n, 2 * n))
    out[:n, :n] = D ** 3 / 3.0 * Qc
    out[:n, n:] = out[n:, :n] = D ** 2 / 2.0 * Qc
    out[n:, n:] = D * Qc
    return out


def _ref_ray(c_est: np.ndarray, c_meas: np.ndarray) -> np.ndarray:
    """The ray at which a row's centre covariance is evaluated: the CURRENT ESTIMATE's. A separate
    function so that KA-MAP-REG can mutate it (the measured ray) and show that the test sees the reference."""
    return c_est / np.linalg.norm(c_est)


def _gauge(c_hat: np.ndarray, c_m: np.ndarray, w: np.ndarray) -> float:
    """The window-gauge component of a correction: log of the weighted mean range ratio (linear in metric range,
    like the scorer's least-squares lambda); the hold divides every range by exp(gamma)."""
    return float(np.log(np.sum(w * np.linalg.norm(c_hat, axis=1)) / np.sum(w * np.linalg.norm(c_m, axis=1))))


def smooth_se3(R_m: np.ndarray, c_m: np.ndarray, dt: np.ndarray, noise: Noise, hand_m: np.ndarray | None = None,
               contact: np.ndarray | None = None, gauge_hold: bool = True) -> SmoothResult:
    """Smooth one window (module docstring). Rows with any non-finite entry carry no measurement.

    Args:
        R_m: (T, 3, 3) measured rotations; c_m: (T, 3) measured centroid positions (camera frame, metres).
        dt: (T-1,) row spacing D_k in rows (1 for a regular window).
        hand_m: (T, 3) measured wrist positions (NaN = missing), or None for the object-only smoother.
        contact: (T,) bool rows where grasp rigidity applies (needs `hand_m`).
        gauge_hold: False only for the KA mutation (the hold disabled).
    """
    R_m, c_m = np.asarray(R_m, np.float64), np.asarray(c_m, np.float64)
    T = len(c_m)
    ok = np.isfinite(c_m).all(1) & np.isfinite(R_m).all(axis=(1, 2))
    if ok.sum() < 3:
        raise ValueError("fewer than 3 measured rows: nothing to smooth")
    use_h = hand_m is not None
    hand_m = np.asarray(hand_m, np.float64) if use_h else None
    ok_h = np.isfinite(hand_m).all(1) if use_h else np.zeros(T, bool)
    if use_h and ok_h.sum() < 3:
        use_h, ok_h = False, np.zeros(T, bool)
    con = (np.asarray(contact, bool) & ok & ok_h) if (use_h and contact is not None) else np.zeros(T, bool)
    n = 18 if use_h else 12
    rng_m = np.linalg.norm(c_m, axis=1)
    zbar = float(np.median(rng_m[ok]))

    # initial trajectory: measurements, finite-difference twists
    Rb, cb = _fill(R_m, ok), _fill(c_m, ok)
    Tb = np.tile(np.eye(4), (T, 1, 1))
    Tb[:, :3, :3], Tb[:, :3, 3] = Rb, cb
    se3 = noise.process == "se3"
    if noise.process not in ("se3", "so3xr3"):
        raise ValueError(f"unknown process model {noise.process!r}")
    xb = np.zeros((T, 6))
    for k in range(T - 1):
        if se3:
            xb[k] = se3_log(np.linalg.inv(Tb[k]) @ Tb[k + 1]) / dt[k]
        else:
            xb[k, :3] = so3_log(Rb[k].T @ Rb[k + 1]) / dt[k]
            xb[k, 3:] = (cb[k + 1] - cb[k]) / dt[k]
    xb[-1] = xb[-2] if T > 1 else 0.0
    if use_h:
        pb = _fill(hand_m, ok_h)
        ub = np.zeros((T, 3))
        ub[:-1] = (pb[1:] - pb[:-1]) / dt[:, None]
        ub[-1] = ub[-2]
    w_R, w_c, w_h, w_g = (np.ones(T) for _ in range(4))
    qd = np.array([noise.q_w] * 3 + [noise.q_v] * 3)
    zbar_h = float(np.median(np.linalg.norm(hand_m[ok_h], axis=1))) if use_h else float("nan")
    ref = noise.noise_ref

    def S_c(k: int) -> np.ndarray:
        """Centre covariance of row k, evaluated at a reference that does NOT carry row k's own noise (s.2.3)."""
        if ref == "measured":                      # mutation / ablation only: self-weighting, biased inward
            return _aniso(c_m[k] / rng_m[k], noise.a_par * rng_m[k], noise.a_perp * rng_m[k])
        return _aniso(_ref_ray(Tb[k, :3, 3], c_m[k]), noise.a_par * zbar, noise.a_perp * zbar)

    def S_h(k: int) -> np.ndarray:
        if ref == "measured":
            r = float(np.linalg.norm(hand_m[k]))
            return _aniso(hand_m[k] / r, noise.a_par_h * r, noise.a_perp_h * r)
        return _aniso(_ref_ray(pb[k], hand_m[k]), noise.a_par_h * zbar_h, noise.a_perp_h * zbar_h)

    big = noise.diffuse ** 2
    P0 = np.diag(big * np.concatenate([[noise.s_rot ** 2] * 3, [(noise.a_par * zbar) ** 2] * 3,
                                       [noise.s_rot ** 2] * 3, [(noise.a_par * zbar) ** 2] * 3]
                                      + ([[(noise.a_par_h * zbar) ** 2] * 6] if use_h else [])))
    n_gn = 0

    def residuals(k: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """(r, H, Sigma, block) at the current linearisation point; H = d r / d e."""
        out = []
        R, c = Tb[k, :3, :3], Tb[k, :3, 3]
        if ok[k]:
            r = so3_log(R.T @ R_m[k])
            H = np.zeros((3, n))
            H[:, 0:3] = -so3_jl_inv(r)
            out.append((r, H, noise.s_rot ** 2 * np.eye(3), "R"))
            H = np.zeros((3, n))
            H[:, 3:6] = -R if se3 else -np.eye(3)
            out.append((c_m[k] - c, H, S_c(k), "c"))
        if use_h and ok_h[k]:
            H = np.zeros((3, n))
            H[:, 12:15] = -np.eye(3)
            out.append((hand_m[k] - pb[k], H, S_h(k), "h"))
        if con[k]:
            om, v = xb[k, :3], xb[k, 3:]
            Rw = R @ om
            d = c - pb[k]
            H = np.zeros((3, n))
            if se3:      # body linear velocity: camera-frame centroid velocity is R v
                g = R @ v - ub[k] - np.cross(Rw, d)
                H[:, 0:3] = -R @ so3_hat(v) - so3_hat(d) @ R @ so3_hat(om)
                H[:, 3:6] = -so3_hat(Rw) @ R
                H[:, 9:12] = R
            else:        # camera-frame linear velocity
                g = v - ub[k] - np.cross(Rw, d)
                H[:, 0:3] = -so3_hat(d) @ R @ so3_hat(om)
                H[:, 3:6] = -so3_hat(Rw)
                H[:, 9:12] = np.eye(3)
            H[:, 6:9] = so3_hat(d) @ R
            H[:, 12:15] = so3_hat(Rw)
            H[:, 15:18] = -np.eye(3)
            out.append((g, H, noise.q_c ** 2 * np.eye(3), "g"))
        return out

    weights = {"R": w_R, "c": w_c, "h": w_h, "g": w_g}
    for rnd in range(noise.reweights + 1):
        for _ in range(noise.gn_iters):
            n_gn += 1
            F_l, u_l, Qd_l = [], [], []
            for k in range(T - 1):
                F = np.eye(n)
                u = np.zeros(n)
                Gm = np.eye(n)
                if se3:
                    G = np.linalg.inv(Tb[k]) @ Tb[k + 1]
                    phi = se3_log(G)
                    Jr = se3_jr(phi)
                    F[:6, :6] = se3_adjoint(np.linalg.inv(G))       # adjoint transport of the error
                    F[:6, 6:12] = dt[k] * Jr
                    u[:6] = Jr @ (dt[k] * xb[k] - phi)
                    Gm[:6, :6] = Jr
                else:
                    GR = Tb[k, :3, :3].T @ Tb[k + 1, :3, :3]
                    phi = so3_log(GR)
                    Jr = so3_jr(phi)
                    F[:3, :3] = GR.T                                # SO(3) adjoint of GR^-1
                    F[:3, 6:9] = dt[k] * Jr
                    F[3:6, 9:12] = dt[k] * np.eye(3)
                    u[:3] = Jr @ (dt[k] * xb[k, :3] - phi)
                    u[3:6] = Tb[k, :3, 3] + dt[k] * xb[k, 3:] - Tb[k + 1, :3, 3]
                    Gm[:3, :3] = Jr
                u[6:12] = xb[k] - xb[k + 1]
                Q = np.zeros((n, n))
                Qo = _q_block(dt[k], qd)
                idx = np.r_[0:12]
                Q[np.ix_(idx, idx)] = Qo
                if use_h:
                    F[12:15, 15:18] = dt[k] * np.eye(3)
                    u[12:15] = pb[k] + dt[k] * ub[k] - pb[k + 1]
                    u[15:18] = ub[k] - ub[k + 1]
                    Q[12:, 12:] = _q_block(dt[k], np.full(3, noise.q_h))
                F_l.append(F)
                u_l.append(u)
                Qd_l.append(Gm @ Q @ Gm.T)
            xs_f, Ps_f, xs_p, Ps_p = [], [], [], []
            x, P = np.zeros(n), P0.copy()
            for k in range(T):
                if k > 0:
                    x = F_l[k - 1] @ x + u_l[k - 1]
                    P = F_l[k - 1] @ P @ F_l[k - 1].T + Qd_l[k - 1]
                xs_p.append(x.copy())
                Ps_p.append(P.copy())
                blocks = residuals(k)
                if blocks:
                    r = np.concatenate([b[0] for b in blocks])
                    H = np.vstack([b[1] for b in blocks])
                    S = np.zeros((len(r), len(r)))
                    o = 0
                    for b in blocks:
                        S[o:o + 3, o:o + 3] = b[2] / max(weights[b[3]][k], 1e-9)
                        o += 3
                    # GN: minimise |r + H e|^2_S  <=>  pseudo-measurement z = -r = H e + v
                    Sy = H @ P @ H.T + S
                    K = np.linalg.solve(Sy, H @ P).T
                    x = x + K @ (-r - H @ x)
                    P = P - K @ H @ P
                    P = 0.5 * (P + P.T)
                xs_f.append(x.copy())
                Ps_f.append(P.copy())
            xs = [xs_f[-1]]
            for k in range(T - 2, -1, -1):
                Gk = np.linalg.solve(Ps_p[k + 1], F_l[k] @ Ps_f[k]).T
                xs.append(xs_f[k] + Gk @ (xs[-1] - xs_p[k + 1]))
            e = np.asarray(xs[::-1])
            for k in range(T):
                if se3:
                    Tb[k] = Tb[k] @ se3_exp(e[k, :6])
                else:
                    Tb[k, :3, :3] = Tb[k, :3, :3] @ so3_exp(e[k, :3])
                    Tb[k, :3, 3] = Tb[k, :3, 3] + e[k, 3:6]
                xb[k] = xb[k] + e[k, 6:12]
                if use_h:
                    pb[k] = pb[k] + e[k, 12:15]
                    ub[k] = ub[k] + e[k, 15:18]
            if np.max(np.abs(e)) < 1e-10:
                break
        if rnd == noise.reweights:
            break
        for k in range(T):
            for r, _, S, blk in residuals(k):
                m = float(np.sqrt(r @ np.linalg.solve(S, r) / 3.0))
                weights[blk][k] = 1.0 if m <= noise.huber_k else noise.huber_k / m

    c_hat = Tb[:, :3, 3].copy()
    gamma = _gauge(c_hat[ok], c_m[ok], w_c[ok])
    hand_hat, gamma_h = None, float("nan")
    if use_h:
        hand_hat = pb.copy()
        gamma_h = _gauge(hand_hat[ok_h], hand_m[ok_h], w_h[ok_h])
    if gauge_hold:
        c_hat *= np.exp(-gamma)
        if use_h:
            hand_hat *= np.exp(-gamma_h)
    wc = np.where(ok, w_c, np.nan)
    wr = np.where(ok, w_R, np.nan)
    wh = np.where(ok_h, w_h, np.nan) if use_h else None
    wg = np.where(con, w_g, np.nan) if use_h else None
    return SmoothResult(R=Tb[:, :3, :3].copy(), c=c_hat, hand=hand_hat, gamma=gamma, gamma_hand=gamma_h, w_c=wc,
                        w_R=wr, n_gn=n_gn, w_h=wh, w_g=wg)

