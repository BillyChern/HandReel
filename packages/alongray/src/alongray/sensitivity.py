# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Window- and frame-scope alignments side by side, and what each one reports for a PURE gauge.

Every alignment maps the prediction onto the truth before the error is read; the error is the mean over the used
rows of ``|aligned_p - g|`` (the MPJPE / mean-centre-error convention) and its RMS. Window-scope alignments fit ONE
transform over every frame; frame-scope alignments fit one per frame.

| name | fitted | error it reports for a pure gauge ``p = lam g`` |
|---|---|---|
| ``none`` | nothing | ``|lam - 1| <|g|>`` |
| ``translation`` | one offset per window | ``|lam - 1|`` x the mean distance of g from its window mean (0 for a static point) |
| ``scale`` | one scale ``kappa`` about the CAMERA centre per window (``kappa p ~ g``) | 0 |
| ``sim3`` | one similarity per window (Umeyama) | 0 |
| ``root_frame`` | per-frame translation (a root joint, or the centroid) | ``|lam - 1|`` x the mean root-relative size |
| ``sim3_frame`` | per-frame similarity (PA-MPJPE) | 0 |

``scale`` fits the prediction onto the truth; it differs from ``BodyScore.aligned["centre_rms_after_lam"]``, which
fits the truth's orbit to the prediction (``p - lam g``). Frame-scope alignments need several points per frame
(``root_frame`` 2, ``sim3_frame`` 3) and are None for a centre-only body; ``sim3`` is None when the true points are
collinear to a relative tolerance of 1e-6 (second singular value of the centred points below 1e-6 x the first).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alongray.aligned import umeyama_similarity

__all__ = ["ALIGNMENTS", "aligned_errors", "gauge_sensitivity"]

ALIGNMENTS = ("none", "translation", "scale", "sim3", "root_frame", "sim3_frame")
_COLLINEAR_RTOL = 1e-6


def _prep(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.ndim == 2:
        p, g = p[:, None], g[:, None]
    if p.shape != g.shape or p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError(f"pred {np.shape(pred)} and gt {np.shape(gt)} must match as (T, 3) or (T, N, 3)")
    ok = np.isfinite(p).all(axis=(1, 2)) & np.isfinite(g).all(axis=(1, 2))
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        if v.shape != (p.shape[0],):
            raise ValueError(f"valid must be a (T,) = ({p.shape[0]},) frame mask, got shape {v.shape}")
        ok &= v
    if not ok.any():
        raise ValueError("no frame is valid and finite in both pred and gt")
    return p[ok], g[ok]


def _stats(diff: np.ndarray) -> dict[str, float]:
    n = np.linalg.norm(diff.reshape(-1, 3), axis=1)
    return {"mean": float(n.mean()), "rms": float(np.sqrt(np.mean(n * n)))}


def _spans_a_plane(x: np.ndarray) -> bool:
    sv = np.linalg.svd(x - x.mean(0), compute_uv=False)
    return bool(sv[0] > 0 and sv[1] > _COLLINEAR_RTOL * sv[0])


def aligned_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None,
                   root: int | None = None) -> dict[str, Any]:
    """Error of one body under every alignment of :data:`ALIGNMENTS` (module docstring).

    Args:
        pred, gt: (T, 3) reference points or (T, N, 3) corresponding points (joints), camera frame, metres.
        valid: optional (T,) frame mask; frames non-finite anywhere are dropped too.
        root: root point index for ``root_frame`` (0 = MANO wrist); None = the per-frame centroid.

    Returns:
        ``{name: {"mean", "rms", ...fit} or None}`` plus ``n_frames`` and ``n_points``.

    Raises:
        ValueError: shapes differ or are not (T, 3) / (T, N, 3), ``valid`` is not (T,), or no frame is usable.
    """
    p, g = _prep(pred, gt, valid)
    t_n, n_pts = p.shape[:2]
    fp, fg = p.reshape(-1, 3), g.reshape(-1, 3)
    out: dict[str, Any] = {"n_frames": int(t_n), "n_points": int(n_pts), "none": _stats(p - g)}
    off = (fg - fp).mean(0)
    out["translation"] = {**_stats(p + off - g), "offset": off.tolist()}
    kappa = float(np.sum(fp * fg) / np.sum(fp * fp))
    out["scale"] = {**_stats(kappa * p - g), "kappa": kappa}
    out["sim3"] = None
    if len(fp) >= 3 and _spans_a_plane(fg):
        try:
            s, r, t = umeyama_similarity(fp, fg)
            out["sim3"] = {**_stats(s * p @ r.T + t - g), "scale": s}
        except ValueError:          # a prediction with zero spread: the similarity is undefined
            pass
    if n_pts >= 2:
        pr = p - (p.mean(1, keepdims=True) if root is None else p[:, root:root + 1])
        gr = g - (g.mean(1, keepdims=True) if root is None else g[:, root:root + 1])
        out["root_frame"] = _stats(pr - gr)
    else:
        out["root_frame"] = None
    out["sim3_frame"] = None
    if n_pts >= 3:
        try:
            res = []
            for k in range(t_n):
                s, r, t = umeyama_similarity(p[k], g[k])
                res.append(s * p[k] @ r.T + t - g[k])
            out["sim3_frame"] = _stats(np.stack(res))
        except ValueError:          # a frame whose predicted points have zero spread
            pass
    return out


def gauge_sensitivity(gt: np.ndarray, lams: tuple[float, ...] = (0.8, 0.9, 1.1, 1.25),
                      valid: np.ndarray | None = None, root: int | None = None) -> dict[str, dict[str, float | None]]:
    """Each alignment's mean error when the prediction is the truth moved by a PURE gauge ``g_lam: x -> lam x``.

    A pure gauge leaves every projected point in place (pinhole camera, isolated points), so a non-zero entry is error
    the projected points cannot reveal: that metric mixes the gauge in. Returns ``{alignment: {repr(lam): mean error or None}}``.

    Raises:
        ValueError: a ``lam`` is not finite and positive, or :func:`aligned_errors` refuses the input.
    """
    g = np.asarray(gt, dtype=np.float64)
    table: dict[str, dict[str, float | None]] = {a: {} for a in ALIGNMENTS}
    for lam in lams:
        if not (np.isfinite(lam) and lam > 0):
            raise ValueError(f"lam must be finite and > 0, got {lam!r}")
        e = aligned_errors(lam * g, g, valid, root)
        for a in ALIGNMENTS:
            table[a][repr(float(lam))] = None if e[a] is None else e[a]["mean"]
    return table
