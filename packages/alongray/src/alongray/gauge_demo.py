# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""The monocular scale gauge, shown on synthetic points: observability education, not error estimation.

For a calibrated pinhole camera, pi(K x) = pi(K lam x) for every lam > 0: a whole scene scaled about the camera
centre projects to the same pixels, so one image cannot fix its metric scale. This module builds such a pair of scenes
and two negative controls, in which the gauge is broken by metric information:

- a hand anchor of fixed metric size (it can follow the scaling only by changing size, so its pixels move);
- a second camera at a fixed, non-zero metric baseline (the scaling about camera 1 moves pixels in camera 2).

It estimates no error of any method, and it does not show that a complete rendered objective (occlusion, contact,
priors) is invariant. The gauge freedom of reconstruction is classical: B. Triggs, P. McLauchlan, R. Hartley and
A. Fitzgibbon, "Bundle Adjustment: A Modern Synthesis", Vision Algorithms: Theory and Practice, 2000,
https://perception.inrialpes.fr/Publications/2000/TMHF00/Triggs-va99.pdf .
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alongray.score import intrinsics_matrix

__all__ = ["LABEL", "MIN_DEPTH_M", "REFERENCE", "fixed_anchor_control", "project", "scaled_scene_pair",
           "second_camera_control"]

MIN_DEPTH_M = 1e-3
LABEL = "observability education on synthetic points, not error estimation"
REFERENCE = ("Triggs, McLauchlan, Hartley and Fitzgibbon, Bundle Adjustment: A Modern Synthesis (2000), "
             "https://perception.inrialpes.fr/Publications/2000/TMHF00/Triggs-va99.pdf")


def _points(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite (N, 3) array of camera-frame points in metres")
    return a


def _depth_ok(x: np.ndarray, name: str) -> None:
    if np.any(x[:, 2] <= MIN_DEPTH_M):
        raise ValueError(f"{name}: depth at or below {MIN_DEPTH_M} m (near-zero or behind the camera) is refused")


def _lam(lam: float) -> float:
    lam = float(lam)
    if not (np.isfinite(lam) and lam > 0):
        raise ValueError(f"lam must be finite and positive, got {lam!r}")
    return lam


def project(k: Any, x: Any) -> np.ndarray:
    """Pinhole pixels (N, 2) of camera-frame points (N, 3); depths must exceed MIN_DEPTH_M."""
    kk = intrinsics_matrix(k)
    x = _points(x, "points")
    _depth_ok(x, "points")
    h = x @ kk.T
    return h[:, :2] / h[:, 2:3]


def scaled_scene_pair(points: Any, lam: float, k: Any) -> dict[str, Any]:
    """A scene and the same scene scaled by ``lam`` about the camera centre, with their largest pixel difference."""
    lam = _lam(lam)
    x = _points(points, "points")
    _depth_ok(x, "points")
    xs = lam * x
    _depth_ok(xs, "scaled points")
    d = float(np.abs(project(k, x) - project(k, xs)).max())
    return {"points": x, "scaled_points": xs, "lam": lam, "max_pixel_difference": d,
            "metric_depth_ratio": float(np.median(xs[:, 2] / x[:, 2])), "label": LABEL, "reference": REFERENCE}


def fixed_anchor_control(points: Any, anchor: Any, lam: float, k: Any) -> dict[str, Any]:
    """Negative control: the scene is scaled, while a metric anchor (e.g. a hand of known size) keeps its shape and
    size and only moves its centroid to ``lam`` times its depth. The anchor's pixels then change."""
    lam = _lam(lam)
    pair = scaled_scene_pair(points, lam, k)
    a = _points(anchor, "anchor")
    _depth_ok(a, "anchor")
    c = a.mean(axis=0)
    moved = a - c + lam * c
    _depth_ok(moved, "moved anchor")
    return {"lam": lam, "scene_max_pixel_difference": pair["max_pixel_difference"],
            "anchor_max_pixel_difference": float(np.abs(project(k, a) - project(k, moved)).max()),
            "label": LABEL, "reference": REFERENCE}


def second_camera_control(points: Any, lam: float, k: Any, baseline: Any) -> dict[str, Any]:
    """Negative control: a second camera displaced by a fixed metric ``baseline`` (camera-1 frame, no rotation) sees
    the scaled scene at different pixels unless the baseline is zero."""
    lam = _lam(lam)
    pair = scaled_scene_pair(points, lam, k)
    b = np.asarray(baseline, np.float64).reshape(3)
    if not np.isfinite(b).all():
        raise ValueError("baseline must be finite")
    x, xs = pair["points"], pair["scaled_points"]
    d2 = float(np.abs(project(k, x - b) - project(k, xs - b)).max())
    return {"lam": lam, "baseline_m": b.tolist(), "camera1_max_pixel_difference": pair["max_pixel_difference"],
            "camera2_max_pixel_difference": d2, "label": LABEL, "reference": REFERENCE}
