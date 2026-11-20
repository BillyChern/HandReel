# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Pose-aligned protocol metrics, reported beside ``e_par`` for contrast.

Each of these aligns the prediction before scoring, and each is invariant to (or confounded by) the gauge
``g_lam`` that ``e_par`` measures:

- :func:`mpjpe_root_aligned`: hand MPJPE after subtracting each side's root joint. Exactly invariant when the
  hand is translated by ``(lam - 1)`` times its root at fixed size (what a hand-depth error does).
- :func:`pa_mpjpe`: hand MPJPE after a per-frame similarity (Procrustes with scale). Invariant to any gauge.
- :func:`object_cd_after_scaled_icp`: object Chamfer (cm^2) after one scaled ICP over the frames. A similarity
  absorbs ``g_lam``, so it is invariant in algebra (exactly, up to rounding, with ``normalise_init=True``).

:func:`scaled_icp` follows PyTorch3D's ``iterative_closest_point(estimate_scale=True, allow_reflection=False)``
(identity initialisation, nearest neighbours in the ground truth, one Umeyama refit per iteration, a stop on the
relative RMSE change, tested here before the refit). The arithmetic follows the authors' development
implementation operation for operation (checked bit for bit outside this package).

Units: metres in; MPJPE in metres; Chamfer in cm^2 (metres x 100, mean squared nearest-neighbour distance both ways).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

__all__ = [
    "SimilarityFit",
    "chamfer_cm2",
    "mpjpe_root_aligned",
    "object_cd_after_scaled_icp",
    "pa_mpjpe",
    "scaled_icp",
    "umeyama_similarity",
]

_RMSE_EPS = 1e-12   # metres; stands in for a previous RMSE of exactly 0 in the relative-change stop test


@dataclass(frozen=True)
class SimilarityFit:
    """The similarity ``x -> scale * rotation @ x + translation`` returned by :func:`scaled_icp`."""

    scale: float
    rotation: np.ndarray      # (3, 3), det = +1
    translation: np.ndarray   # (3,)
    iterations: int
    rmse: float
    converged: bool


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares ``(s, R, t)`` minimising ``sum_i |s R src_i + t - dst_i|^2`` (Umeyama 1991), R proper.

    Raises:
        ValueError: inputs are not (N, 3) of equal N, N < 3, or ``src`` has zero spread.
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
    """ONE similarity aligning every ``pred_k`` to ``gt_k`` by ICP with scaling (neighbours of ``pred_k`` in ``gt_k``).

    Raises:
        ValueError: ``pred`` and ``gt`` differ in length or are empty, a cloud is not (N, 3), or max_iterations < 1.
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


def chamfer_cm2(source: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Chamfer distance in cm^2: metres x 100, mean squared nearest-neighbour distance both ways, summed."""
    s = np.asarray(source, dtype=np.float64) * 100.0
    t = np.asarray(target, dtype=np.float64) * 100.0
    d_ts, _ = cKDTree(s).query(t)
    d_st, _ = cKDTree(t).query(s)
    return float(np.mean(np.square(d_ts)) + np.mean(np.square(d_st)))


def object_cd_after_scaled_icp(pred: Sequence[np.ndarray], gt: Sequence[np.ndarray],
                               max_iterations: int = 100, centre: bool = True,
                               normalise_init: bool = False) -> tuple[float, float]:
    """Object Chamfer (cm^2) after ONE scaled ICP over the given frames: ``(mean over frames, fitted scale)``.

    ``centre`` subtracts each cloud's centroid first. ``normalise_init`` (needs ``centre``) first scales the
    predictions by rms(gt) / rms(pred), pooled over frames, so the result is gauge-invariant by construction; the
    returned scale includes that factor.

    Raises:
        ValueError: ``normalise_init`` without ``centre``, or invalid clouds (see :func:`scaled_icp`).
    """
    pre = 1.0
    if centre:
        pred = [np.asarray(p, np.float64) - np.asarray(p, np.float64).mean(0) for p in pred]
        gt = [np.asarray(g, np.float64) - np.asarray(g, np.float64).mean(0) for g in gt]
        if normalise_init:
            rp = math.sqrt(np.mean(np.concatenate([np.sum(p ** 2, 1) for p in pred])))
            rg = math.sqrt(np.mean(np.concatenate([np.sum(g ** 2, 1) for g in gt])))
            pre = rg / rp
            pred = [pre * p for p in pred]
    elif normalise_init:
        raise ValueError("normalise_init needs centre=True")
    aligned, fit = scaled_icp(list(pred), list(gt), max_iterations=max_iterations)
    vals = [chamfer_cm2(a, np.asarray(g, np.float64)) for a, g in zip(aligned, gt, strict=True)]
    return float(np.mean(vals)), float(fit.scale) * pre


def _joints_pair(pred_joints: np.ndarray, gt_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two (T, J, 3) arrays of the SAME shape; refused (never broadcast) otherwise."""
    p = np.asarray(pred_joints, dtype=np.float64)
    g = np.asarray(gt_joints, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 3 or p.shape != g.shape:
        raise ValueError(f"pred and gt joints must be (T, J, 3) arrays of the same shape, got {p.shape} and {g.shape}")
    return p, g


def mpjpe_root_aligned(pred_joints: np.ndarray, gt_joints: np.ndarray, root: int = 0) -> np.ndarray:
    """Per-frame MPJPE (m) after subtracting each side's root joint. (T, J, 3) inputs of the same shape.

    Raises:
        ValueError: inputs are not (T, J, 3) of the same shape.
    """
    p, g = _joints_pair(pred_joints, gt_joints)
    pr = p - p[:, root:root + 1]
    gr = g - g[:, root:root + 1]
    out: np.ndarray = np.linalg.norm(pr - gr, axis=2).mean(axis=1)
    return out


def pa_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray) -> np.ndarray:
    """Per-frame MPJPE (m) after a per-frame similarity of the prediction onto the GT.

    NaN on a frame that is not finite, or whose similarity is undefined (fewer than 3 joints, or all predicted
    joints at one point), so one degenerate frame does not abort the whole score.

    Raises:
        ValueError: inputs are not (T, J, 3) of the same shape (checked before any frame is scored).
    """
    p, g = _joints_pair(pred_joints, gt_joints)
    out = np.full(len(p), np.nan)
    for t in range(len(p)):
        if not (np.isfinite(p[t]).all() and np.isfinite(g[t]).all()):
            continue
        try:
            s, r, tr = umeyama_similarity(p[t], g[t])
        except ValueError:
            logger.debug("pa_mpjpe: frame %d has an undefined similarity; scored NaN", t)
            continue
        out[t] = float(np.linalg.norm(s * p[t] @ r.T + tr - g[t], axis=1).mean())
    return out
