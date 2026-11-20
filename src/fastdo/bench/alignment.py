"""Similarity alignment for protocol-conformant reconstruction metrics: Umeyama fit and scaled ICP.

`scaled_icp` follows PyTorch3D's `iterative_closest_point(estimate_scale=True, allow_reflection=False)`, the
alignment MCC-HO / DiffHOI / G-HOP (and so DO AS I DO Table 2) apply before F-score and Chamfer: identity
initialisation, nearest neighbours of the transformed prediction in the ground truth, one Umeyama similarity
refit per iteration, and a stop on the relative RMSE change (tested before the refit here and after it in
PyTorch3D; `scaled_icp` documents the difference and the measured parity). Several (pred_k, gt_k) pairs share ONE
similarity while nearest neighbours of pred_k are searched only in gt_k (a per-clip gauge fit).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

__all__ = ["SimilarityFit", "scaled_icp", "umeyama_similarity"]

_RMSE_EPS = 1e-12   # metres; stands in for a previous RMSE of exactly 0 in the relative-change stop test


@dataclass(frozen=True)
class SimilarityFit:
    """The similarity x -> scale * rotation @ x + translation returned by `scaled_icp`.

    `iterations` counts the Umeyama refits composing the result (0: identity); `rmse` (metres) is the
    nearest-neighbour RMSE at the returned alignment; `converged` is False when `max_iterations` was reached.
    """

    scale: float
    rotation: np.ndarray      # (3,3), det = +1
    translation: np.ndarray   # (3,)
    iterations: int
    rmse: float
    converged: bool


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares (s, R, t) minimising sum_i |s R src_i + t - dst_i|^2 (Umeyama 1991), R a proper rotation.

    Raises:
        ValueError: `src` and `dst` are not (N,3) arrays of equal N, N < 3, or `src` has zero spread.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or src.shape != dst.shape:
        raise ValueError(f"src and dst must be (N,3) arrays of equal N, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"a similarity needs at least 3 point pairs, got {n}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float((xs**2).sum() / n)
    if var_s <= 0.0:
        raise ValueError("src points are all identical; the similarity is undefined")
    u, sig, vt = np.linalg.svd(xd.T @ xs / n)
    d = np.ones(3)
    sign = float(np.sign(np.linalg.det(u @ vt)))
    d[2] = sign if sign != 0.0 else 1.0          # reflection guard; a zero sign counts as +1
    rotation = u @ np.diag(d) @ vt
    scale = float((sig * d).sum() / var_s)
    return scale, rotation, mu_d - scale * rotation @ mu_s


def _cloud(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (N,3) array, got shape {arr.shape}")
    return arr


def _transform(clouds: Sequence[np.ndarray], scale: float, rotation: np.ndarray, translation: np.ndarray
               ) -> list[np.ndarray]:
    return [scale * c @ rotation.T + translation for c in clouds]


def scaled_icp(pred: Sequence[np.ndarray], gt: Sequence[np.ndarray], max_iterations: int = 100,
               relative_rmse_thr: float = 1e-6) -> tuple[list[np.ndarray], SimilarityFit]:
    """One similarity aligning every pred_k to gt_k by ICP with scaling; returns the aligned predictions and the fit.

    Starting from the identity similarity, each iteration runs four steps:
    1. match: transform every pred_k with the current (s, R, t) and take each point's nearest neighbour in gt_k
       only (one cKDTree per gt_k, built once);
    2. RMSE: the RMSE of the current alignment over those pairs, pooled over all k;
    3. stop test (from the second iteration on): stop when |prev_rmse - rmse| / prev_rmse < `relative_rmse_thr`,
       prev_rmse being the step-2 RMSE of the previous iteration (a previous RMSE of exactly 0 counts as
       `_RMSE_EPS`); the current alignment is returned as is, without a further refit;
    4. refit: otherwise fit ONE similarity with `umeyama_similarity` on the concatenated (pred, matched gt)
       pairs; it is the current alignment of the next iteration.
    When `max_iterations` refits pass without a stop, one more match gives the RMSE of the returned alignment
    and `converged` is False. The step-2 RMSE is non-increasing in exact arithmetic (the refit is least-squares
    optimal for its pairs and rematching only shortens distances), so the absolute value in step 3 differs from
    a signed test `(prev_rmse - rmse) / prev_rmse < thr` only by float noise.

    PyTorch3D's `iterative_closest_point(estimate_scale=True)` orders the loop differently: it matches, refits
    FIRST, then takes the RMSE of the refit cloud against the pre-refit neighbours, applies the signed test
    `(prev_rmse - rmse) / prev_rmse <= thr` to that RMSE and returns the refit when it stops. The two loops
    therefore test different RMSE sequences and may stop at different iterations. The measured parity holds
    despite the ordering difference: max |ours - PyTorch3D| point deviation 0.023 mm (float32 PyTorch3D, CPU)
    over the 9 cases of the parity script (box / cylinder / annulus at 3 / 8 / 15 deg), with identical F-5,
    F-10 and Chamfer in all 9.

    Raises:
        ValueError: `pred` and `gt` differ in length or are empty, a cloud is not (N,3), or max_iterations < 1.
    """
    preds = [_cloud(p, f"pred[{k}]") for k, p in enumerate(pred)]
    gts = [_cloud(g, f"gt[{k}]") for k, g in enumerate(gt)]
    if not preds or len(preds) != len(gts):
        raise ValueError(f"pred and gt must be equally long non-empty sequences, got {len(preds)} and {len(gts)}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    trees = [cKDTree(g) for g in gts]
    pred_all = np.concatenate(preds)

    def match(clouds: list[np.ndarray]) -> tuple[float, np.ndarray]:
        dists, idx = zip(*(tree.query(c, k=1, workers=1) for c, tree in zip(clouds, trees, strict=True)),
                         strict=True)
        rmse = float(np.sqrt(np.mean(np.concatenate(dists) ** 2)))
        return rmse, np.concatenate([g[i] for g, i in zip(gts, idx, strict=True)])

    scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
    prev_rmse: float | None = None
    iterations, converged = 0, False
    for _ in range(max_iterations):
        current = _transform(preds, scale, rotation, translation)
        rmse, matched = match(current)
        if prev_rmse is not None and abs(prev_rmse - rmse) / max(prev_rmse, _RMSE_EPS) < relative_rmse_thr:
            converged = True
            break
        prev_rmse = rmse
        scale, rotation, translation = umeyama_similarity(pred_all, matched)
        iterations += 1
    else:
        current = _transform(preds, scale, rotation, translation)
        rmse, _matched = match(current)
        logger.debug("scaled ICP stopped at max_iterations=%d with RMSE %.3e m", max_iterations, rmse)
    fit = SimilarityFit(scale=scale, rotation=rotation, translation=translation, iterations=iterations, rmse=rmse,
                        converged=converged)
    return current, fit
