"""Knot-space control residuals.

Knot residuals U_i in R^{K_i x R} at rows t_k = W + 40 k (k = 0, 1, ... while t_k < T), U_i[k] = u_exec(t_k) -
ctrl_ref(t_k); the control of every executor row is ctrl_ref + the residual LINEARLY interpolated between knots,
held constant after the last knot. Before the first knot (the welded warmup, rows < W) the residual ramps linearly
from 0 at row 0 (the warm start's own placed start, whose control is exact) to U[0] at W: an implementation choice
made before the re-knot known answer ran, because a residual of 0 through the warmup would jump by
U[0] at the release and a held U[0] would move the hand away from the placed start at row 0.

The fallback representation (if the 40-row global knots fail the re-knot known answer) is knots at every
replan block boundary plus the 40-row interior knots: `knot_rows(..., block=ctrl_steps)`.

Named representations (`representation_knots`): the registered `knot40` (the grid above) FAILS its re-knot known
answer; three candidates are re-tested, in order of parsimony (`R2_CANDIDATES`), on a holdout of TUNE runs:
`knot40_from0` (the 40-row grid from row 0), `two_rate` (20-row knots in [0, W), 40-row knots from W) and
`knot20_from0` (a 20-row grid from row 0). With a knot at row 0 the start residual is that knot's value (no ramp).

Per-dimension scales (`KnotScaler`) are frozen on the training set: the std of every residual dimension over all
valid knots, floored at `floor`; samples are produced in scaled units and mapped back with `inverse`.
Padding: a batch of clips is padded to K_max knots with a mask (`pad`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "KNOT_SPACING",
    "R2_CANDIDATES",
    "REPRESENTATIONS",
    "KnotScaler",
    "cond_features",
    "knot_rows",
    "knots_from_controls",
    "pad",
    "representation_knots",
    "residual_rows",
    "rows_from_knots",
]

KNOT_SPACING = 40
R2_CANDIDATES: tuple[str, ...] = ("knot40_from0", "two_rate", "knot20_from0")      # order of parsimony
REPRESENTATIONS: tuple[str, ...] = ("knot40", *R2_CANDIDATES)


def knot_rows(W: int, T: int, spacing: int = KNOT_SPACING, block: int | None = None, start: int | None = None,
              block_from: int | None = None) -> np.ndarray:
    """Knot rows t_k = start + spacing k < T (start = W, the registered grid); with `block`, also every block
    boundary in [block_from, T) (block_from = W by default)."""
    s0 = int(W) if start is None else int(start)
    rows = set(range(s0, int(T), int(spacing)))
    if block is not None:
        b0 = int(W) if block_from is None else int(block_from)
        first = -(-b0 // int(block)) * int(block)
        rows |= set(range(first, int(T), int(block)))
    return np.array(sorted(rows), np.int64)


def representation_knots(name: str, W: int, T: int) -> np.ndarray:
    """Knot rows of a named representation (module docstring) for a clip with release row W and T rows."""
    W, T = int(W), int(T)
    if name == "knot40":
        return knot_rows(W, T)
    if name == "knot40_from0":
        return knot_rows(W, T, start=0)
    if name == "two_rate":
        return np.array(sorted(set(range(0, min(W, T), KNOT_SPACING // 2)) | set(range(W, T, KNOT_SPACING))), np.int64)
    if name == "knot20_from0":
        return knot_rows(W, T, spacing=KNOT_SPACING // 2, start=0)
    raise ValueError(f"unknown knot representation {name!r}")


def residual_rows(knots: np.ndarray, U: np.ndarray, n_rows: int) -> np.ndarray:
    """(n_rows, R) residual of every executor row from knot values U (K, R) at rows `knots` (module docstring)."""
    knots = np.asarray(knots, np.int64)
    U = np.asarray(U, np.float64)
    if knots.ndim != 1 or U.shape[0] != knots.shape[0] or knots.size == 0:
        raise ValueError("one residual row per knot")
    if np.any(np.diff(knots) <= 0) or knots[0] < 0:
        raise ValueError("knot rows must increase and be >= 0")
    if knots[0] == 0:                     # a knot at row 0 sets the start residual itself
        x, y = knots, U
    else:
        x = np.concatenate([[0], knots])
        y = np.concatenate([np.zeros((1, U.shape[1])), U])
    r = np.arange(n_rows)
    return np.stack([np.interp(r, x, y[:, j]) for j in range(U.shape[1])], 1)


def rows_from_knots(ctrl_ref: np.ndarray, knots: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Controls ctrl_ref + residual (rows of `ctrl_ref`; the residual acts on the first U.shape[1] columns)."""
    ctrl_ref = np.asarray(ctrl_ref, np.float64)
    out = ctrl_ref.copy()
    out[:, :U.shape[1]] += residual_rows(knots, U, ctrl_ref.shape[0])
    return out


def knots_from_controls(u_exec: np.ndarray, ctrl_ref: np.ndarray, knots: np.ndarray, width: int | None = None
                        ) -> np.ndarray:
    """U[k] = u_exec(t_k) - ctrl_ref(t_k) over the first `width` columns (all by default)."""
    w = u_exec.shape[1] if width is None else int(width)
    k = np.asarray(knots, np.int64)
    if k.max() >= min(u_exec.shape[0], ctrl_ref.shape[0]):
        raise ValueError("a knot lies beyond the executed controls")
    return np.asarray(u_exec, np.float64)[k, :w] - np.asarray(ctrl_ref, np.float64)[k, :w]


def pad(Us: list[np.ndarray], k_max: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(B, K_max, R) zero-padded knot residuals and the (B, K_max) validity mask."""
    k_max = k_max or max(u.shape[0] for u in Us)
    R = Us[0].shape[1]
    x = np.zeros((len(Us), k_max, R))
    m = np.zeros((len(Us), k_max))
    for i, u in enumerate(Us):
        if u.shape[0] > k_max:
            raise ValueError(f"{u.shape[0]} knots > K_max {k_max}")
        x[i, :u.shape[0]] = u
        m[i, :u.shape[0]] = 1.0
    return x, m


@dataclass(frozen=True)
class KnotScaler:
    scale: np.ndarray           # (R,)

    @staticmethod
    def fit(Us: list[np.ndarray], floor: float = 1e-3) -> KnotScaler:
        allk = np.concatenate([np.asarray(u, np.float64) for u in Us])
        return KnotScaler(np.maximum(allk.std(0), floor))

    def transform(self, U: np.ndarray) -> np.ndarray:
        return np.asarray(U, np.float64) / self.scale

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, np.float64) * self.scale


def cond_features(entry: dict, knots: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The prior's conditioning of one clip, from its `RefTable` entry:
    per knot (K, 6 + R + 3F + F + 1): xi(T^ref_W^-1 T^ref_{t_k}), q^ref_{t_k}, the reference fingertips in the
    object frame, d^ref_f, kappa; global (5,): mass, the bounding box of the object points, side (+1 right, -1 left);
    the object points (P, 3). Fingertips are the kinematic reference's robot tips (`tips_ref`), i.e.
    "MANO tips relative to the object" as retargeted."""
    from fastdo.amortised.se3 import object_error, quat_conj, quat_rotate

    k = np.asarray(knots, np.int64)
    W = int(entry["W"])
    p, q = np.asarray(entry["obj_p"], np.float64), np.asarray(entry["obj_q"], np.float64)
    rho, om = object_error(p[k], q[k], np.repeat(p[W:W + 1], k.size, 0), np.repeat(q[W:W + 1], k.size, 0))
    tips = np.asarray(entry["tips_ref"], np.float64)[k]
    local = quat_rotate(quat_conj(q[k])[:, None], tips - p[k][:, None])
    per = np.concatenate([rho, om, np.asarray(entry["q_ref"], np.float64)[k], local.reshape(k.size, -1),
                          np.asarray(entry["dref"], np.float64)[k], np.asarray(entry["kappa"], np.float64)[k, None]], 1)
    pts = np.asarray(entry["points"], np.float64)
    glob = np.concatenate([[float(entry["mass"])], pts.max(0) - pts.min(0), [1.0 if side == "right" else -1.0]])
    return per, glob, pts
