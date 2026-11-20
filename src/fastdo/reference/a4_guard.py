"""A4 guard: the ALONG-RAY gross-wander guard of a pre-contact lead.

Along-ray translation rest is NOT testable from monocular video, so A4 does not test it; this guard only catches GROSS
along-ray failure (the stored leads that wander by tens of centimetres to a metre). It does not claim rest within 2 cm
along the ray.

Statistic, on the BINDING lead's rows in time order:
- c = the geometric median of the lead's centres (c*, world); u_i = unit(c - C_i), C_i row i's camera centre;
- s_i = u_i . (p_i - c), the along-ray coordinate of row i (m);
- s~ = the 3-row running median of s (a centred window shrinking at the ends, `running_median3`);
- stat = max_i |s~_i - s~_last| (the largest along-ray excursion from the lead's last row, the row nearest contact).
Decision: REFUSE (gross along-ray wander) iff stat > tau(L), L = the lead's row count, tau(L) the q99 of the
statistic over NOISE-ONLY leads of L rows from a block bootstrap of the measured along-ray residuals
(L = 5..45). A lead longer than 45 rows uses tau(45) (the largest tabulated value; the
statistic grows with L, so this refuses at least as often as an extrapolation would); fewer than 5 rows: undecided.
"""

from __future__ import annotations

import numpy as np

from fastdo.reference.a4 import geometric_median

__all__ = ["L_MAX", "L_MIN", "along_ray_guard", "along_ray_p7", "along_ray_series", "running_median3"]

L_MIN, L_MAX = 5, 45


def running_median3(x: np.ndarray) -> np.ndarray:
    """3-row running median of a 1-D series, the window shrinking at the ends (k=3)."""
    x = np.asarray(x, dtype=np.float64)
    return np.array([np.median(x[max(0, i - 1): i + 2]) for i in range(len(x))]) if len(x) else x


def along_ray_series(p: np.ndarray, cam_centres: np.ndarray) -> np.ndarray:
    """(n,) along-ray coordinates s_i (m) of the lead's centres about their geometric median (module docstring)."""
    p = np.asarray(p, dtype=np.float64)
    c = geometric_median(p)
    u = c - np.asarray(cam_centres, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return np.einsum("ni,ni->n", u, p - c)


def along_ray_p7(s: np.ndarray) -> float:
    """max_i |s~_i - s~_last| of a 1-D series (any unit)."""
    sm = running_median3(s)
    return float(np.max(np.abs(sm - sm[-1])))


def along_ray_guard(p: np.ndarray, cam_centres: np.ndarray, tau_mm: dict[int, float]) -> dict:
    """The guard's decision on one lead: {decided, refuse, L, stat_mm, tau_mm, tau_L}. `tau_mm` maps L -> tau(L) in mm
    (the committed table)."""
    n = len(p)
    if n < L_MIN:
        return {"decided": False, "refuse": False, "L": n, "stat_mm": None, "tau_mm": None, "tau_L": None}
    stat = along_ray_p7(along_ray_series(p, cam_centres)) * 1e3
    key = min(n, L_MAX)
    tau = float(tau_mm[key])
    return {"decided": True, "refuse": bool(stat > tau), "L": n, "stat_mm": stat, "tau_mm": tau, "tau_L": key}
