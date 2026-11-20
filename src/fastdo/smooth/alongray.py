"""AR1D: the pure along-ray 1-D RTS smoother (C-SMOOTH generalised).

Each row's posed-template centroid c_k moves ALONG ITS OWN RAY to a robust RTS smooth r_hat_k of its range
r_k = |c_k| under a constant-velocity model with row spacing D_k; rotation, bearing and size are unchanged.
Differences from C-SMOOTH (`fastdo.tba.comparators.c_smooth`, kept bit-for-bit as the control):
1. noise relative to the window's median range zbar (meas sd a_par*zbar, velocity sd b*a_par*zbar per row), one
   pooled constant for every domain instead of a per-corpus GT-derived mm value; this makes it gauge-equivariant;
2. row spacing D_k from frame_idx (C-SMOOTH assumed 1);
3. a DIFFUSE initial state (sd `diffuse` x the measurement sd), so the linear MAP satisfies
   sum_k w_k (r_hat_k - r_k) = 0 exactly (Lemma 1): the constant direction is in the null space of the CV
   prior's precision;
4. the gauge hold of `fastdo.smooth.rts` (exactly a no-op up to round-off when Lemma 1 holds; it removes only the
   residual from the finite diffuse prior).

AR1D-HO (`alongray_joint`; candidate C2, the matched dumb JOINT control): the object's
centroid range and the contact-side wrist range, each a 1-D constant-velocity series along its own rays, coupled on
the contact stage's rows by a 1-D rigidity factor on the range-velocity difference, v_o,k - v_h,k ~ N(0, q_c^2), with
C1's constants (`fastdo.smooth.rts.Noise`: object meas a_par zbar, velocity sd q_v; wrist meas a_par_h zbar_h,
velocity sd q_h; coupling q_c). The problem is linear-Gaussian, so one dense solve of the normal equations is the
exact MAP (no linearisation); Huber IRLS on each block as C1. The coupling acts only on velocities and the CV prior
annihilates constants, so Lemma 1 holds PER BODY: sum_k w_k (r_hat_k - r_k) = 0 for the object and for the wrist
separately (KA-LEMMA1-HO); the hold is a no-op up to round-off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["JointRanges", "alongray_joint", "alongray_ranges", "rts_cv_1d"]


def rts_cv_1d(y: np.ndarray, dt: np.ndarray, meas_sd: float, vel_sd: float, weights: np.ndarray,
              diffuse: float = 1e3) -> np.ndarray:
    """RTS smoother of a scalar series, constant velocity, spacing dt (T-1,); NaN rows carry no measurement."""
    n = len(y)
    H = np.array([1.0, 0.0])
    x = np.zeros(2)
    P = np.diag([(diffuse * meas_sd) ** 2, (diffuse * meas_sd) ** 2])
    first = int(np.flatnonzero(np.isfinite(y))[0])
    x[0] = y[first]
    xf, Pf, xp, Pp, Fs = [], [], [], [], []
    for k in range(n):
        if k > 0:
            D = float(dt[k - 1])
            F = np.array([[1.0, D], [0.0, 1.0]])
            Q = vel_sd ** 2 * np.array([[D ** 3 / 3.0, D ** 2 / 2.0], [D ** 2 / 2.0, D]])
            x, P = F @ x, F @ P @ F.T + Q
            Fs.append(F)
        xp.append(x.copy())
        Pp.append(P.copy())
        if np.isfinite(y[k]):
            r = meas_sd ** 2 / max(float(weights[k]), 1e-9)
            s = H @ P @ H + r
            g = P @ H / s
            x = x + g * (y[k] - H @ x)
            P = P - np.outer(g, H @ P)
        xf.append(x.copy())
        Pf.append(P.copy())
    xs = [xf[-1]]
    for k in range(n - 2, -1, -1):
        G = np.linalg.solve(Pp[k + 1], Fs[k] @ Pf[k]).T
        xs.append(xf[k] + G @ (xs[-1] - xp[k + 1]))
    return np.asarray(xs[::-1])[:, 0]


def alongray_ranges(c_m: np.ndarray, dt: np.ndarray, a_par: float, band: float, huber_k: float = 1.345,
                    reweights: int = 3, gauge_hold: bool = True, diffuse: float = 1e3
                    ) -> tuple[np.ndarray, float, np.ndarray]:
    """Smoothed centres (T, 3) (NaN rows stay NaN), the removed gauge component gamma, and the final weights."""
    c_m = np.asarray(c_m, np.float64)
    ok = np.isfinite(c_m).all(1)
    rng = np.where(ok, np.linalg.norm(np.where(ok[:, None], c_m, 0.0), axis=1), np.nan)
    zbar = float(np.median(rng[ok]))
    meas = a_par * zbar
    w = np.ones(len(c_m))
    sm = rng
    for i in range(reweights + 1):
        sm = rts_cv_1d(rng, dt, meas, band * meas, w, diffuse)
        if i == reweights:
            break
        r = np.abs(rng - sm) / meas
        w = np.where(np.isfinite(r) & (r > huber_k), huber_k / np.where(r > 0, r, 1.0), 1.0)
    # Lemma 1 holds for the weights of the final solve
    gamma = float(np.log(np.sum(w[ok] * sm[ok]) / np.sum(w[ok] * rng[ok])))
    if gauge_hold:
        sm = sm * np.exp(-gamma)
    out = np.full_like(c_m, np.nan)
    out[ok] = c_m[ok] * (sm[ok] / rng[ok])[:, None]
    return out, gamma, w


@dataclass(frozen=True)
class JointRanges:
    c: np.ndarray            # (T, 3) smoothed object centres (NaN where not delivered)
    hand: np.ndarray | None  # (T, 3) smoothed wrists (NaN where not measured), None = object only (fallback)
    gamma: float             # object gauge component removed by the hold (Lemma 1: ~0)
    gamma_hand: float
    w_c: np.ndarray          # (T,) final weights per block (NaN = no residual on that row)
    w_h: np.ndarray
    w_g: np.ndarray


def _cv_rows(T: int, dt: np.ndarray, q: float, off: int, n: int) -> np.ndarray:
    """Whitened CV process residuals of one body (positions at off+k, velocities at off+T+k) as rows of A."""
    A = np.zeros((2 * (T - 1), n))
    for k in range(T - 1):
        D = float(dt[k])
        L = np.linalg.cholesky(q ** 2 * np.array([[D ** 3 / 3.0, D ** 2 / 2.0], [D ** 2 / 2.0, D]]))
        Li = np.linalg.inv(L)
        blk = np.zeros((2, n))
        blk[0, off + k + 1], blk[0, off + k], blk[0, off + T + k] = 1.0, -1.0, -D
        blk[1, off + T + k + 1], blk[1, off + T + k] = 1.0, -1.0
        A[2 * k:2 * k + 2] = Li @ blk
    return A


def alongray_joint(c_m: np.ndarray, h_m: np.ndarray | None, contact: np.ndarray | None, dt: np.ndarray, noise,
                   gauge_hold: bool = True) -> JointRanges:
    """AR1D-HO (module docstring). `noise` is C1's `Noise` for the window (a_par, q_v, a_par_h, q_h, q_c, huber_k,
    reweights); the object's measurement sd is a_par x zbar and the wrist's a_par_h x zbar_h (their median ranges).
    With fewer than 3 measured wrist rows it is AR1D-with-C1-constants on the object alone (hand None)."""
    c_m = np.asarray(c_m, np.float64)
    T = len(c_m)
    ok = np.isfinite(c_m).all(1)
    r_o = np.where(ok, np.linalg.norm(np.where(ok[:, None], c_m, 0.0), axis=1), np.nan)
    zbar = float(np.median(r_o[ok]))
    use_h = h_m is not None
    if use_h:
        h_m = np.asarray(h_m, np.float64)
        ok_h = np.isfinite(h_m).all(1)
        use_h = ok_h.sum() >= 3
    if not use_h:
        ok_h = np.zeros(T, bool)
        h_m = np.full((T, 3), np.nan)
    r_h = np.where(ok_h, np.linalg.norm(np.where(ok_h[:, None], h_m, 0.0), axis=1), np.nan)
    zbar_h = float(np.median(r_h[ok_h])) if use_h else float("nan")
    con = (np.asarray(contact, bool) & ok & ok_h) if (use_h and contact is not None) else np.zeros(T, bool)
    nb = 2 if use_h else 1
    n = 2 * T * nb
    sd_o, sd_h = noise.a_par * zbar, (noise.a_par_h * zbar_h if use_h else 1.0)
    A_p = [_cv_rows(T, dt, noise.q_v, 0, n)]
    if use_h:
        A_p.append(_cv_rows(T, dt, noise.q_h, 2 * T, n))
    A_prior = np.vstack(A_p)
    io, ih, ig = np.flatnonzero(ok), np.flatnonzero(ok_h), np.flatnonzero(con)
    A_o = np.zeros((len(io), n))
    A_o[np.arange(len(io)), io] = 1.0
    A_h = np.zeros((len(ih), n))
    A_g = np.zeros((len(ig), n))
    if use_h:
        A_h[np.arange(len(ih)), 2 * T + ih] = 1.0
        A_g[np.arange(len(ig)), T + ig] = 1.0
        A_g[np.arange(len(ig)), 3 * T + ig] = -1.0
    w_o, w_h, w_g = np.ones(len(io)), np.ones(len(ih)), np.ones(len(ig))
    x = np.zeros(n)
    for rnd in range(noise.reweights + 1):
        a_o = A_o * (np.sqrt(w_o) / sd_o)[:, None]
        a_h = A_h * (np.sqrt(w_h) / sd_h)[:, None]
        a_g = A_g * (np.sqrt(w_g) / noise.q_c)[:, None]
        A = np.vstack([A_prior, a_o, a_h, a_g])
        b = np.concatenate([np.zeros(len(A_prior)), r_o[io] * np.sqrt(w_o) / sd_o, r_h[ih] * np.sqrt(w_h) / sd_h,
                            np.zeros(len(ig))])
        x = np.linalg.solve(A.T @ A, A.T @ b)
        if rnd == noise.reweights:
            break
        hub = lambda m: np.where(m > noise.huber_k, noise.huber_k / np.where(m > 0, m, 1.0), 1.0)
        w_o = hub(np.abs(x[io] - r_o[io]) / sd_o)
        if use_h:
            w_h = hub(np.abs(x[2 * T + ih] - r_h[ih]) / sd_h)
            w_g = hub(np.abs(x[T + ig] - x[3 * T + ig]) / noise.q_c)
    sm_o = x[:T]
    gamma = float(np.log(np.sum(w_o * sm_o[io]) / np.sum(w_o * r_o[io])))
    gamma_h = float("nan")
    if gauge_hold:
        sm_o = sm_o * np.exp(-gamma)
    c_hat = np.full_like(c_m, np.nan)
    c_hat[ok] = c_m[ok] * (sm_o[ok] / r_o[ok])[:, None]
    hand = None
    if use_h:
        sm_h = x[2 * T:3 * T]
        gamma_h = float(np.log(np.sum(w_h * sm_h[ih]) / np.sum(w_h * r_h[ih])))
        if gauge_hold:
            sm_h = sm_h * np.exp(-gamma_h)
        hand = np.full_like(h_m, np.nan)
        hand[ok_h] = h_m[ok_h] * (sm_h[ok_h] / r_h[ok_h])[:, None]
    full = lambda idx, w: np.where(np.isin(np.arange(T), idx), np.bincount(idx, w, T) if len(idx) else 0.0, np.nan)
    return JointRanges(c=c_hat, hand=hand, gamma=gamma, gamma_hand=gamma_h, w_c=full(io, w_o), w_h=full(ih, w_h),
                       w_g=full(ig, w_g))
