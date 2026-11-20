"""Per-output wall QC for the SAM 3D fast postprocess.

The decoder hands FlexiCubes a thin negative shell, so the extracted surface is double-walled: an outer wall and an
inner wall 1-3 mm under it. The postprocess's mincut normally deletes the inner wall, but reduced-step samplers leave
it far more often (13/90 DexYCB runs at 8 steps against 0/24 at 25). Samples on a kept inner wall carry inward
normals, which breaks both the solver's front-facing test and its pseudonormal SDF, so a kept wall is a quality
failure, not a cosmetic one.

An inside-outside test on the output alone cannot find it: a double-walled shell is closed and consistently oriented,
exactly like a genuinely thin object (the whisk's wires). The postprocess's own decision plus one global ratio can.

**Definition (fixed by the design; computed on `fast_postprocess`'s output, before the y-up rotation).**
- `removed` = faces removed by the mincut / faces after decimation;
- `rho` = total mesh area / area of the convex hull of the mesh's vertices;
- **flag if and only if `removed < max_removed` and `rho >= min_area_ratio`** (defaults 0.25 and 1.40).

A single-walled closed object loses its inner wall, so the mincut removes 37-76% of the decimated faces and `rho`
stays near 1 (DexYCB singles: 0.93-1.19). A kept inner wall roughly doubles the area against the hull (walls:
1.60-2.38) and leaves the mincut with almost nothing to cut (0.001-0.21). 1.40 sits near the geometric mean of the two
`rho` groups, a factor of about 1.17 from each.

**Fallback.** A flagged request is re-run once, in the same worker, at 25+25 Euler steps with `fast_postprocess` (the
same seed), and the fallback output is used whether or not it flags again (`fallback` / `fallback_flagged`).
There is no cap on fallbacks; their rate is a reported throughput metric.

This module runs under the SAM 3D environment's interpreter beside `sam3d_worker.py`, so it imports nothing from
`fastdo`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_MAX_REMOVED",
    "DEFAULT_MIN_AREA_RATIO",
    "FALLBACK_DROPPED",
    "FALLBACK_SOLVER",
    "FALLBACK_STEPS",
    "QC_FALLBACK",
    "QC_FALLBACK_FLAGGED",
    "QC_PASS",
    "WALL_QC_MODES",
    "WallQcThresholds",
    "area_ratio",
    "fallback_params",
    "flagged",
    "removed_fraction",
    "statistics",
]

DEFAULT_MAX_REMOVED = 0.25
DEFAULT_MIN_AREA_RATIO = 1.40
WALL_QC_MODES: tuple[str, ...] = ("off", "fallback")
# `qc` in the worker reply and in shape.json.
QC_PASS, QC_FALLBACK, QC_FALLBACK_FLAGGED = "pass", "fallback", "fallback_flagged"
# The path the fallback re-runs: 25+25 Euler with fast_postprocess.
FALLBACK_STEPS = 25
FALLBACK_SOLVER = "euler"
FALLBACK_DROPPED = ("sparse_structure_shortcut", "structured_latent_solver",   # the sampler options
                    "sparse_structure_solver")                                  # and the SS solver


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass(frozen=True)
class WallQcThresholds:
    """The QC's two thresholds; the defaults are the design's calibrated pair.

    Raises:
        ValueError: On a `max_removed` outside [0, 1] or a `min_area_ratio` that is not positive (either naming its
            `ShapeConfig` field).
    """

    max_removed: float = DEFAULT_MAX_REMOVED
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO

    def __post_init__(self) -> None:
        if not _finite(self.max_removed) or not 0.0 <= float(self.max_removed) <= 1.0:
            raise ValueError(f"wall_qc_max_removed must be a finite number in [0, 1], got {self.max_removed!r}")
        if not _finite(self.min_area_ratio) or float(self.min_area_ratio) <= 0.0:
            raise ValueError(f"wall_qc_min_area_ratio must be a finite number > 0, got {self.min_area_ratio!r}")


def removed_fraction(faces_decimated: int, faces_removed_mincut: int) -> float:
    """`faces_removed_mincut / faces_decimated`; 0.0 when the decimated mesh is empty (nothing could be cut)."""
    decimated = int(faces_decimated)
    return float(int(faces_removed_mincut) / decimated) if decimated > 0 else 0.0


def area_ratio(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Total mesh area over the area of the convex hull of `vertices`; 0.0 for a degenerate hull."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    hull_area = float(mesh.convex_hull.area)
    return float(mesh.area) / hull_area if hull_area > 0.0 else 0.0


def statistics(vertices: np.ndarray, faces: np.ndarray, *, faces_decimated: int,
               faces_removed_mincut: int) -> dict[str, Any]:
    """The QC's inputs for one postprocess output: the two face counts, `removed` and `area_ratio`."""
    return {"faces_decimated": int(faces_decimated), "faces_removed_mincut": int(faces_removed_mincut),
            "removed": removed_fraction(faces_decimated, faces_removed_mincut),
            "area_ratio": area_ratio(vertices, faces)}


def flagged(stats: Mapping[str, Any], thresholds: WallQcThresholds) -> bool:
    """Whether `stats` looks double-walled: `removed < max_removed` and `area_ratio >= min_area_ratio`."""
    return bool(float(stats["removed"]) < float(thresholds.max_removed)
                and float(stats["area_ratio"]) >= float(thresholds.min_area_ratio))


def fallback_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """`params` re-pointed at the fallback path: 25+25 Euler steps with today's sampler (the sampler options
    dropped), the fast postprocess, the same seed, the QC off.

    The QC is off in the fallback because its output is used either way; the worker labels it `fallback_flagged` when
    it would have flagged again, from statistics it computes itself.
    """
    fallback = dict(params) | {"sparse_structure_steps": FALLBACK_STEPS, "structured_latent_steps": FALLBACK_STEPS,
                               "solver": FALLBACK_SOLVER, "mesh_postprocess_impl": "fast", "wall_qc": "off"}
    for name in FALLBACK_DROPPED:       # the fallback path samples with today's sampler, never the request's
        fallback.pop(name, None)
    return fallback
