"""Noise-calibrated motion test.

One test serves two callers:
- the conditioner's drift refusal (a rest segment is refused `moved`), on the residuals of its robust mean;
- the lead-motion test of the batch screen, on the lead rows' world poses.

Decision: `moved` iff the motion is SIGNIFICANT (a permutation trend test and a permutation split-half test, combined by
Bonferroni so the joint level is <= alpha = 0.05 one-sided) AND its estimated MAGNITUDE exceeds the floor (20 mm centre
or 15 deg rotation: the batch screen's tolerance, the module's action cap). Significance alone is scale-free and fires
on sub-millimetre MoCap trends (a defect of an earlier version); the magnitude alone fires on jitter.

Statistics (rows in time order, tau = row time):
- residuals r_i (6,) = [centre - reference; Log(O_i S_g Oref^T)] (world-left rotation), physical units (m, rad);
- whitening W_i: the inverse Cholesky factor of a per-row 6x6 covariance (`cov`), or the ROBUST RESIDUAL SCALE of the
  rows themselves (default): per block (centre, rotation) s^2 = median_i ||d_i - mean(d)||^2 / (2 * chi2_3 median),
  d_i = r_{i+1} - r_i (first differences: a linear drift adds a constant to d, which the centring removes, so the
  scale is not inflated by the motion it is testing);
- z_i = W_i (r_i - a), a the Huber IRLS location of the rows (so the result does not depend on the residuals'
  reference), Huber-clipped to norm <= k (3.55 = sqrt(chi2_6 0.95)): one gross row has bounded influence;
- TREND T = ||sum_i (tau_i - taubar) z_i||^2 / sum_i (tau_i - taubar)^2; SPLIT-HALF H = ||mean(z first floor(n/2)) -
  mean(z last floor(n/2))||^2 (n >= 6 only);
- null: B = 1999 random permutations of the rows' time slots; p = (1 + #{stat_perm >= stat_obs}) / (B + 1);
- combined p = min(1, 2 min(p_T, p_H)) (p_T alone when n < 6); significant iff combined p <= alpha.
Magnitude (physical units, rigid-motion equivariant): a Huber IRLS line fit r_i ~ a + b (tau_i - taubar) under the
same whitening gives the trend's end-to-end change b * (tau_max - tau_min); a Huber IRLS location of each half gives
the split-half discrepancy. magnitude = max(trend end-to-end, half discrepancy), per block (mm, deg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastdo.reference.config import ReferenceConfig
from fastdo.reference.so3 import angle_deg, log

__all__ = ["MotionTest", "motion_test", "motion_test_residuals", "pose_residuals", "robust_scale"]

_CHI2_3_MEDIAN = 2.365974


@dataclass(frozen=True)
class MotionTest:
    n: int
    decided: bool                   # False: fewer than `min_rows` rows (no decision; the caller refuses or skips)
    moved: bool                     # significant AND exceeds_floor
    significant: bool
    exceeds_floor: bool
    p_value: float | None           # Bonferroni-combined
    p_trend: float | None
    p_half: float | None
    trend_stat: float | None
    half_stat: float | None
    trend_mm: float | None          # end-to-end change of the robust trend line (centre, mm)
    trend_deg: float | None
    half_disc_mm: float | None      # robust half-location discrepancy (centre, mm)
    half_disc_deg: float | None
    magnitude_mm: float | None
    magnitude_deg: float | None
    floor_mm: float
    floor_deg: float
    scale_mm: float | None          # robust-mode whitening scale (None when a covariance whitened the rows)
    scale_deg: float | None
    scale_source: str               # "robust_residual" | "covariance" | "given"
    reason: str = ""

    def refuse(self, alpha: float | None = None) -> bool:
        """Rev-1 call site compatibility: a segment is refused `moved` iff the rev-2 decision says moved."""
        return self.moved

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def robust_scale(r: np.ndarray, cfg: ReferenceConfig) -> np.ndarray:
    """(2,) robust per-block noise scale (m, rad) from first differences (module docstring), floored."""
    r = np.asarray(r, dtype=np.float64)
    d = np.diff(r, axis=0)
    out = np.empty(2)
    floors = (cfg.motion_min_scale_m, np.radians(cfg.motion_min_scale_deg))
    for b, sl in enumerate((slice(0, 3), slice(3, 6))):
        if len(d) == 0:
            out[b] = floors[b]
            continue
        dc = d[:, sl] - d[:, sl].mean(axis=0)
        s2 = float(np.median(np.sum(dc * dc, axis=1))) / (2.0 * _CHI2_3_MEDIAN)
        out[b] = max(np.sqrt(max(s2, 0.0)), floors[b])
    return out


def _whiteners(r: np.ndarray, cov: np.ndarray | None, scale: np.ndarray | None, cfg: ReferenceConfig
               ) -> tuple[np.ndarray, np.ndarray | None, str]:
    """(n,6,6) whitening matrices, the robust scale used (or None) and its source."""
    n = len(r)
    if cov is not None:
        C = np.asarray(cov, dtype=np.float64)
        C = 0.5 * (C + np.transpose(C, (0, 2, 1)))
        L = np.linalg.cholesky(C)
        return np.linalg.inv(L), None, "covariance"
    src = "given" if scale is not None else "robust_residual"
    s = np.asarray(scale, dtype=np.float64) if scale is not None else robust_scale(r, cfg)
    W = np.diag(np.repeat(1.0 / s, 3))
    return np.broadcast_to(W, (n, 6, 6)).copy(), s, src


def _clip(z: np.ndarray, k: float) -> np.ndarray:
    """Huber clip of each row's whitened residual to norm <= k."""
    nz = np.linalg.norm(z, axis=1)
    return z * np.minimum(1.0, k / np.maximum(nz, 1e-300))[:, None]


def _irls(X: np.ndarray, r: np.ndarray, W: np.ndarray, k: float, iters: int = 60) -> np.ndarray:
    """Huber IRLS: argmin_theta sum_i rho_k(||W_i (r_i - X_i theta)||), X (n,6,m)."""
    WX = np.einsum("nij,njm->nim", W, X)
    Wr = np.einsum("nij,nj->ni", W, r)
    w = np.ones(len(r))
    theta = np.zeros(X.shape[2])
    for _ in range(iters):
        A = np.einsum("n,nim,nil->ml", w, WX, WX)
        g = np.einsum("n,nim,ni->m", w, WX, Wr)
        new = np.linalg.lstsq(A, g, rcond=None)[0]
        e = np.linalg.norm(Wr - np.einsum("nim,m->ni", WX, new), axis=1)
        w = np.minimum(1.0, k / np.maximum(e, 1e-300))
        if np.max(np.abs(new - theta)) < 1e-12:
            theta = new
            break
        theta = new
    return theta


def _block_norms(v: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(v[:3]) * 1e3), float(np.degrees(np.linalg.norm(v[3:])))


def _split_half(z: np.ndarray, zp: np.ndarray, B: int, min_rows: int) -> tuple[float, float] | None:
    """(p, stat) of the permutation split-half test, or None when n < `min_rows`."""
    n = len(z)
    if n < min_rows:
        return None
    h = n // 2
    obs = float(np.sum((z[:h].mean(0) - z[n - h:].mean(0)) ** 2))
    perm = np.sum((zp[:, :h].mean(1) - zp[:, n - h:].mean(1)) ** 2, axis=1)
    return (1.0 + float(np.sum(perm >= obs - 1e-12))) / (B + 1.0), obs


def _combine(p_t: float, p_h: float | None) -> float:
    """Bonferroni over the two tests (joint level <= alpha)."""
    return p_t if p_h is None else min(1.0, 2.0 * min(p_t, p_h))


def motion_test_residuals(times: np.ndarray, r: np.ndarray, cfg: ReferenceConfig | None = None, seed: int = 0,
                          cov: np.ndarray | None = None, scale: np.ndarray | None = None) -> MotionTest:
    """The motion test on precomputed residuals `r (n,6)` (m, rad) at `times (n,)` (module docstring).

    `cov (n,6,6)` whitens each row by its covariance; otherwise `scale (2,)` (m, rad) if given, else the robust
    residual scale of the rows themselves."""
    cfg = cfg or ReferenceConfig()
    tau = np.asarray(times, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = len(tau)
    floor_mm, floor_deg = cfg.motion_floor_m * 1e3, cfg.motion_floor_deg
    if n < cfg.motion_min_rows:
        return MotionTest(n, False, False, False, False, None, None, None, None, None, None, None, None, None,
                          None, None, floor_mm, floor_deg, None, None, "none", "too_few_rows")
    W, s, src = _whiteners(r, cov, scale, cfg)
    k = cfg.huber_k
    loc = _irls(np.broadcast_to(np.eye(6), (n, 6, 6)).copy(), r, W, k)     # robust centre: clip about it
    z = _clip(np.einsum("nij,nj->ni", W, r - loc), k)
    c = tau - tau.mean()
    den = float(c @ c) if float(c @ c) > 0 else 1.0
    B = cfg.drift_permutations
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(np.arange(n), (B, 1)), axis=1)
    zp = z[perms]
    t_obs = float(np.sum((c @ z) ** 2) / den)
    t_perm = np.sum(np.einsum("n,bnd->bd", c, zp) ** 2, axis=1) / den
    p_t = (1.0 + float(np.sum(t_perm >= t_obs - 1e-12))) / (B + 1.0)
    sh = _split_half(z, zp, B, cfg.split_half_min_vote)
    p_h, h_obs = (None, None) if sh is None else sh
    p = _combine(p_t, p_h)
    X = np.zeros((n, 6, 12))
    X[:, :, :6] = np.eye(6)
    X[:, :, 6:] = c[:, None, None] * np.eye(6)
    b = _irls(X, r, W, k)[6:]
    tr_mm, tr_deg = _block_norms(b * float(np.ptp(tau)))
    hd_mm = hd_deg = 0.0
    if n >= cfg.split_half_min_vote:
        h = n // 2
        I6 = np.broadcast_to(np.eye(6), (h, 6, 6))
        a1 = _irls(I6.copy(), r[:h], W[:h], k)
        a2 = _irls(I6.copy(), r[n - h:], W[n - h:], k)
        hd_mm, hd_deg = _block_norms(a2 - a1)
    mag_mm, mag_deg = max(tr_mm, hd_mm), max(tr_deg, hd_deg)
    sig = p <= cfg.drift_alpha
    exceeds = mag_mm > floor_mm or mag_deg > floor_deg
    return MotionTest(n, True, bool(sig and exceeds), bool(sig), bool(exceeds), p, p_t, p_h, t_obs, h_obs, tr_mm,
                      tr_deg, hd_mm if sh is not None else None, hd_deg if sh is not None else None, mag_mm, mag_deg,
                      floor_mm, floor_deg, None if s is None else float(s[0] * 1e3),
                      None if s is None else float(np.degrees(s[1])), src)


def _medoid_index(D: np.ndarray) -> int:
    return int(np.argmin(D.sum(axis=1)))


def pose_residuals(p: np.ndarray, O: np.ndarray, elements: np.ndarray | None = None) -> np.ndarray:
    """(n,6) residuals of world poses to their medoid reference (rigid-motion equivariant): centre minus the centre
    medoid; Log(O_i S_g Oref^T) with Oref the rotation medoid under the symmetry quotient and g_i the nearest
    representative."""
    p = np.asarray(p, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    G = np.eye(3)[None] if elements is None else np.asarray(elements, dtype=np.float64)
    pref = p[_medoid_index(np.linalg.norm(p[:, None] - p[None], axis=2))]
    X = np.einsum("nij,kjl->nkil", O, G)                                    # O_i S_g
    ang = np.stack([angle_deg(X, O[j][None, None]).min(axis=1) for j in range(len(O))], axis=1)
    Oref = O[_medoid_index(ang)]
    g = np.argmin(angle_deg(X, Oref[None, None]), axis=1)
    rot = log(np.einsum("nij,nkj->nik", X[np.arange(len(O)), g], np.broadcast_to(Oref, O.shape)))
    return np.concatenate([p - pref, rot], axis=1)


def motion_test(times: np.ndarray, p: np.ndarray, O: np.ndarray, cfg: ReferenceConfig | None = None, seed: int = 0,
                cov: np.ndarray | None = None, elements: np.ndarray | None = None,
                scale: np.ndarray | None = None) -> MotionTest:
    """Lead-motion entry point: is the object MOVING over these lead rows?

    Args:
        times: (n,) row times (s), increasing.
        p: (n,3) world positions of the object's reference CENTRE (m). Pass the template's surface-centroid centre
            (`SymmetryGroup.c_star` posed), not the arbitrary SAM 3D origin: an off-centre origin turns rotation noise
            into position noise by its lever arm.
        O: (n,3,3) world rotations.
        cov: optional (n,6,6) per-row world covariance of [centre; world-left rotation] (m, rad). Without it the rows
            are whitened by their own robust residual scale (module docstring).
        elements: optional (K,3,3) symmetry group (index 0 = identity); rotation residuals use the quotient.
        scale: optional (2,) whitening scale (m, rad) instead of the robust estimate (ignored when `cov` is given).

    Returns:
        `MotionTest`: `moved` (significant AND magnitude > floor), the combined and per-test p-values, the magnitude
        estimates (mm, deg) and the floor. `decided` is False with fewer than `cfg.motion_min_rows` rows.
    """
    return motion_test_residuals(times, pose_residuals(p, O, elements), cfg, seed, cov=cov, scale=scale)
