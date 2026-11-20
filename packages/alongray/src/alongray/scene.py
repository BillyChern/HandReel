# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Several bodies in one window: one gauge per body, one shared scene scale, and the relative gauge.

Each rigid body (the object, each hand) has its own gauge ``lam_b``. One similarity about the camera (one ``lam_s``
for every body) removes only the common part. What is left is the RELATIVE gauge ``log(lam_a / lam_b)``: it moves
one body along its rays relative to the other while, under a pinhole camera, every projected point of both stays put,
so the projected points cannot reveal this hand-object placement error. Occlusion and contact between the bodies
are not modelled; they can change pixels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alongray.core import gauge_decomposition
from alongray.score import reference_points

__all__ = ["SceneGauge", "scene_gauge"]


@dataclass(frozen=True)
class SceneGauge:
    """Scene-level gauge readings (metres).

    Attributes:
        lam: Per-body least-squares scale (as in :func:`alongray.e_par_split`).
        shared_lam: ONE least-squares scale fitted over every body's reference points together.
        relative_log_gauge: ``{"a/b": log(lam_a / lam_b)}`` for every pair, in the order given.
        shared_off_orbit: Per body, RMS along-ray ``a_t - shared_lam Z_t``.
        residual_after_shared_scale: Per body, ``shared_off_orbit / shared_lam``: the body's ``e_par`` after the
            prediction is divided by the shared scale (compare with the body's own ``oracle_floor``).
    """

    lam: dict[str, float]
    shared_lam: float
    relative_log_gauge: dict[str, float]
    shared_off_orbit: dict[str, float] = field(default_factory=dict)
    residual_after_shared_scale: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Plain-Python dict."""
        return {"lam": self.lam, "shared_lam": self.shared_lam, "relative_log_gauge": self.relative_log_gauge,
                "shared_off_orbit": self.shared_off_orbit,
                "residual_after_shared_scale": self.residual_after_shared_scale}


def _finite(p: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(p).all(1) & np.isfinite(g).all(1)
    return p[ok], g[ok]


def scene_gauge(bodies: Mapping[str, tuple[np.ndarray, np.ndarray]], hand_root: int = 0) -> SceneGauge:
    """Per-body and shared gauges of ``{name: (pred, gt)}`` (shapes as :func:`alongray.reference_points`).

    Raises:
        ValueError: fewer than one body, a body fails :func:`alongray.gauge_decomposition`, or the shared scale is
            not finite and positive.
    """
    if not bodies:
        raise ValueError("no body to score")
    pairs = {n: _finite(reference_points(p, hand_root), reference_points(g, hand_root)) for n, (p, g) in bodies.items()}
    lam = {n: gauge_decomposition(p, g).lam for n, (p, g) in pairs.items()}
    num = sum(float(np.sum(p * g)) for p, g in pairs.values())
    den = sum(float(np.sum(g * g)) for _, g in pairs.values())
    shared = num / den
    if not (math.isfinite(shared) and shared > 0):
        raise ValueError(f"the shared scale {shared!r} is not finite and positive")
    off, res = {}, {}
    for n, (p, g) in pairs.items():
        z = np.linalg.norm(g, axis=1)
        o = np.einsum("ij,ij->i", p, g) / z - shared * z
        off[n] = float(np.sqrt(np.mean(o * o)))
        res[n] = off[n] / shared
    names = list(lam)
    rel = {f"{a}/{b}": math.log(lam[a] / lam[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return SceneGauge(lam=lam, shared_lam=shared, relative_log_gauge=rel, shared_off_orbit=off,
                      residual_after_shared_scale=res)
