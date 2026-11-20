"""Comparator arms on BASE's stored output: C-SMOOTH and C-CONS. Never promoted.

- **C-SMOOTH** (the dumb matched control): BASE with each row's camera-frame template centre moved ALONG ITS OWN RAY to
  a robust Rauch-Tung-Striebel smooth of the range ``|c_k|`` over the rows (constant-velocity model), size and rotation
  unchanged. Noise fixed a priori from step 0's within-window sd of the signed along-ray error (DexYCB 63.7 mm, HOI4D
  21.5 mm): measurement sd = that value, velocity random-walk sd per row = a quarter of it; three Huber reweightings (k
  = 1.345 measurement sd).
- **C-CONS** (lambda-only): BASE scaled by the pointmap-consensus gauge g_exp(l) (no BA, no carving),
  ``l = median over (k, live i) of log(z_obs / z_BASE)``, with z_BASE the camera depth of BASE's keyframe anchor i in
  row k and z_obs the stored pointmap depth at the tracked pixel.
"""

from __future__ import annotations

import math

import numpy as np

from fastdo.bench.metric_frame import apply_gauge
from fastdo.tba.bundle import pair_depths
from fastdo.tba.geometry import relative_poses

__all__ = ["c_cons", "c_smooth", "consensus_gauge_base", "rts_smooth"]

HUBER_K = 1.345


def rts_smooth(y: np.ndarray, meas_sd: float, vel_sd: float, weights: np.ndarray | None = None) -> np.ndarray:
    """RTS smoother of a scalar series under a constant-velocity model (unit row spacing); NaN rows are skipped.

    `weights` (T,) in (0, 1] inflate the measurement variance of row k to ``meas_sd^2 / w_k`` (robust reweighting).
    """
    y = np.asarray(y, np.float64)
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = vel_sd ** 2 * np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]])
    H = np.array([1.0, 0.0])
    first = int(np.flatnonzero(np.isfinite(y))[0])
    x = np.array([y[first], 0.0])
    Pm = np.diag([meas_sd ** 2, (10.0 * vel_sd) ** 2])
    xs_f, Ps_f, xs_p, Ps_p = [], [], [], []
    for k in range(n):
        if k > 0:
            x, Pm = F @ x, F @ Pm @ F.T + Q
        xs_p.append(x.copy())
        Ps_p.append(Pm.copy())
        if np.isfinite(y[k]):
            r = meas_sd ** 2 / max(w[k], 1e-6)
            s = H @ Pm @ H + r
            g = Pm @ H / s
            x = x + g * (y[k] - H @ x)
            Pm = Pm - np.outer(g, H @ Pm)
        xs_f.append(x.copy())
        Ps_f.append(Pm.copy())
    xs = [xs_f[-1]]
    for k in range(n - 2, -1, -1):
        G = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs.append(xs_f[k] + G @ (xs[-1] - xs_p[k + 1]))
    return np.asarray(xs[::-1])[:, 0]


def c_smooth(poses: np.ndarray, scale: float, centroid: np.ndarray, meas_sd_m: float,
             reweights: int = 3) -> np.ndarray:
    """C-SMOOTH poses (module docstring): rows keep rotation; translations move each centre along its own ray."""
    poses = np.asarray(poses, np.float64)
    ok = np.isfinite(poses).all(axis=(1, 2))
    c = np.full((len(poses), 3), np.nan)
    c[ok] = scale * poses[ok, :3, :3] @ np.asarray(centroid, np.float64) + poses[ok, :3, 3]
    rng = np.linalg.norm(c, axis=1)
    w = np.ones(len(poses))
    sm = rng
    for _ in range(reweights + 1):
        sm = rts_smooth(rng, meas_sd_m, 0.25 * meas_sd_m, w)
        r = np.abs(rng - sm) / meas_sd_m
        w = np.where(np.isfinite(r) & (r > HUBER_K), HUBER_K / np.where(r > 0, r, 1.0), 1.0)
    out = poses.copy()
    u = c[ok] / rng[ok, None]
    out[ok, :3, 3] = poses[ok, :3, 3] + (sm[ok] - rng[ok])[:, None] * u
    return out


def consensus_gauge_base(poses: np.ndarray, key: int, rays: np.ndarray, d_base: np.ndarray, vis: np.ndarray,
                         z_obs: np.ndarray) -> float:
    """C-CONS's ``l`` (module docstring); 0 when no live anchored track has a pointmap reading."""
    anchored = np.isfinite(d_base)
    zb = pair_depths(relative_poses(poses, key), np.where(anchored, d_base, np.nan), rays)
    m = vis & anchored[None, :] & np.isfinite(z_obs) & np.isfinite(zb)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.log(z_obs[m] / zb[m])
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else 0.0


def c_cons(poses: np.ndarray, scale: float, ell: float) -> tuple[np.ndarray, float]:
    """C-CONS: g_exp(ell) of BASE (every translation and the scale)."""
    return apply_gauge(poses, scale, math.exp(float(ell)))
