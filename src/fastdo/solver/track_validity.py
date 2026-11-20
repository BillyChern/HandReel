"""Per-frame track validity from the Tomasi-Kanade rank constraint, using tracks alone.

Why this exists. Measured on 38 DexYCB clips and 82,844 track-frame entries against an exact
oracle, CoTracker's error is **bimodal**,
not gradual: sub-pixel while locked (0.82 px median) and catastrophic once lost (37.5 px median,
263 px p90), with **half of every clip's tracks dead by frame ~14 of 24** and only 7.6% surviving a
24-frame clip. The loss times are consistent across clips (half-life quartiles [10, 15]), so the
track term is excellent early and mostly outliers late, in every clip alike.

What this computes. Under an affine camera a rigid object's **uncentred** track matrix satisfies
`M = [A | t] [X; 1]`, which has rank 4, so an entry that does not fit a rank-4 model is inconsistent
with any rigid motion. The per-frame inlier fraction `S(k)` then estimates how much of the track
term at frame `k` is still real. It separates exact tracks from CoTracker at **92.1%** accuracy,
returning `S(k) = 1` on exact input.

Three traps, each measured rather than assumed:

  * **Least squares cannot be used.** Its breakdown point is 0, and at the measured 40.8%
    contamination it fits the LOST tracks: as a detector it scored AUC **0.2485**, i.e. inverted.
  * **Do not centre.** Tomasi-Kanade's rank-3 form needs the CENTROID, because subtracting it is
    what makes the per-frame translation vanish. A *median* centre does not, and leaves a rank-1
    per-frame offset the model must absorb -- on exact tracks the best possible fit degrades from
    0.213 px to 0.924 px. The uncentred rank-4 form avoids the choice entirely and fits best
    (0.131 px median, p90 0.964).
  * **Structure fitted only from the clean early frames is ill-conditioned** (6 frames excite the
    third direction to `sigma_3/sigma_1` ~ 0.006), and a minimal-sample RANSAC on it, followed by a
    redescending loss with no basin, is worse than useless. The early frames are used to
    INITIALISE; each frame's camera is warm-started from the previous one (the motion is smooth,
    which is free conditioning) and the structure is re-estimated from inlier-weighted entries of
    all frames.

Achievable floor, measured: even the best possible rank-4 fit to EXACT tracks has p99 5.3 px over
38 clips, so `inlier_px` cannot be set tight; 6 px is the shipped value and lost tracks sit at
37.5 px, far from it.

No depth, no ground truth and no solve enter this file.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

RANK = 4                    # uncentred: M = [A | t] [X; 1]; see the module docstring.
_MIN_TRACKS = 8
_MIN_SCALE_PX = 0.5


def _wls(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted least squares: `target` (n,d) ~ `design` (n,k) @ C; returns C (k,d)."""
    root = np.sqrt(np.maximum(weight, 0.0))[:, None]
    coef, *_ = np.linalg.lstsq(design * root, target * root, rcond=None)
    return coef


def _tukey(residual: np.ndarray, cutoff: float) -> np.ndarray:
    """Tukey biweight, redescending to exactly zero at `cutoff`."""
    u = np.clip(residual / max(cutoff, 1e-9), 0.0, 1.0)
    return (1.0 - u**2) ** 2


def frame_validity(tracks: np.ndarray, vis: np.ndarray | None = None, k0: int = 8,
                   inlier_px: float = 6.0, irls_iters: int = 6, outer_iters: int = 3) -> np.ndarray:
    """Per-frame fraction of tracks still consistent with a rigid motion.

    Args:
        tracks: (T, N, 2) track positions in pixels.
        vis: (T, N) visibility, or None for all-visible. Invisible entries are excluded from the
            fits and from the returned fractions' denominators.
        k0: Frames used to initialise the structure (contamination is lowest early).
        inlier_px: Residual below which an entry counts as consistent.
        irls_iters: IRLS refinements of each frame's camera.
        outer_iters: Alternations between per-frame cameras and the structure.

    Returns:
        (T,) float64 in [0, 1]. All ones when the input is too small to fit (fewer than
        `_MIN_TRACKS` tracks), so a caller can multiply by it unconditionally.
    """
    X = np.asarray(tracks, dtype=np.float64)
    if X.ndim != 3 or X.shape[2] != 2:
        raise ValueError(f"tracks must be (T,N,2), got {X.shape}")
    T, N, _ = X.shape
    if N < _MIN_TRACKS or T < 2:
        logger.debug("track_validity: %d tracks over %d frames is too small to fit; returning ones", N, T)
        return np.ones(T)
    W0 = np.ones((T, N)) if vis is None else np.asarray(vis, dtype=bool).astype(np.float64)

    k0 = int(min(max(k0, 2), T))
    M0 = np.concatenate([X[:k0, :, 0], X[:k0, :, 1]], axis=0)
    _u, s, vt = np.linalg.svd(M0, full_matrices=False)
    rank = min(RANK, vt.shape[0])
    B = (vt[:rank] * s[:rank, None]).T                          # (N, rank) structure, INITIAL only

    resid = np.zeros((T, N))
    for outer in range(outer_iters):
        cameras = np.zeros((T, rank, 2))
        prev: np.ndarray | None = None
        for k in range(T):
            Y, w0 = X[k], W0[k]
            cam = prev if prev is not None else _wls(B, Y, w0)   # warm start: the motion is smooth
            for _ in range(irls_iters):
                r = np.linalg.norm(Y - B @ cam, axis=1)
                scale = max(1.4826 * float(np.median(r[w0 > 0])) if (w0 > 0).any() else 0.0, _MIN_SCALE_PX)
                cam = _wls(B, Y, w0 * _tukey(r, max(4.0 * scale, inlier_px)))
            cameras[k], prev = cam, cam
            resid[k] = np.linalg.norm(Y - B @ cam, axis=1)
        if outer == outer_iters - 1:
            break
        cutoff = max(4.0 * max(1.4826 * float(np.median(resid)), _MIN_SCALE_PX), inlier_px)
        w = W0 * _tukey(resid, cutoff)
        design = np.concatenate([cameras[:, :, 0], cameras[:, :, 1]], axis=0)        # (2T, rank)
        target = np.concatenate([X[:, :, 0], X[:, :, 1]], axis=0)                    # (2T, N)
        wstack = np.concatenate([w, w], axis=0)
        B = np.stack([_wls(design, target[:, j:j + 1], wstack[:, j])[:, 0] for j in range(N)])

    ok = (resid < inlier_px) & (W0 > 0)
    denom = np.maximum(W0.sum(axis=1), 1.0)
    return np.clip(ok.sum(axis=1) / denom, 0.0, 1.0)
