"""Short-gap filling of an ego-motion trajectory.

`fastdo.geometry.ego_motion` writes one Sim(3) camera pose per row, X_t = [s_t R_t | u_t], and a `valid` mask for the
rows that joined the pose graph. `world_frame.apply_ego_motion` refuses a window unless EVERY row is valid. In one
diagnosis, 10 of 104 stored windows pass the window-level self-check but are refused for a few off-trajectory rows.

`fill_gaps` fills each INTERIOR run of invalid rows of length <= `max_gap_rows` between its two bracketing valid rows.
It interpolates along the geodesic of the product Lie group SE(3) x R+:

    (R_t, u_t) = X_a Exp(tau Log(X_a^-1 X_b))   (SE(3), body-frame geodesic)
    s_t        = s_a^(1 - tau) s_b^tau            (R+, geometric)
    tau        = (t - a) / (b - a)

This is exact for a constant-twist motion with geometric scale drift (test known answer). It never extrapolates:
leading and trailing invalid rows stay invalid. Rows it fills are flagged, so a consumer can report them, and valid
rows are returned bit-identical.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from fastdo.smooth.lie import se3_exp, se3_log

__all__ = ["fill_gaps", "split_sim3"]


def split_sim3(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, u) of a Sim(3) matrix [s R | u] (s = cube root of det of the 3x3 block)."""
    m = np.asarray(x, float)[:3, :3]
    s = float(np.cbrt(np.linalg.det(m)))
    return s, m / s, np.asarray(x, float)[:3, 3].copy()


def _se3(R: np.ndarray, u: np.ndarray) -> np.ndarray:
    g = np.eye(4)
    g[:3, :3], g[:3, 3] = R, u
    return g


def fill_gaps(c2w: np.ndarray, valid: np.ndarray, max_gap_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(c2w_out, valid_out, filled)`: interior gaps of <= `max_gap_rows` rows filled on the SE(3) x R+ geodesic
    (module docstring); everything else unchanged."""
    c2w = np.asarray(c2w, float)
    valid = np.asarray(valid, bool)
    out, ok, filled = c2w.copy(), valid.copy(), np.zeros_like(valid)
    if max_gap_rows <= 0:
        return out, ok, filled
    idx = np.flatnonzero(valid)
    for a, b in pairwise(idx):
        gap = b - a - 1
        if gap == 0 or gap > max_gap_rows:
            continue
        sa, Ra, ua = split_sim3(c2w[a])
        sb, Rb, ub = split_sim3(c2w[b])
        ga, gb = _se3(Ra, ua), _se3(Rb, ub)
        xi = se3_log(np.linalg.inv(ga) @ gb)
        for t in range(a + 1, b):
            tau = (t - a) / (b - a)
            g = ga @ se3_exp(tau * xi)
            s = sa ** (1.0 - tau) * sb ** tau
            out[t, :3, :3] = s * g[:3, :3]
            out[t, :3, 3] = g[:3, 3]
            out[t, 3] = (0.0, 0.0, 0.0, 1.0)
            ok[t] = filled[t] = True
    return out, ok, filled
