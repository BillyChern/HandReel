# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""alongray: the unaligned along-ray error e_par of hand-object reconstructions and its exact gauge split.

``e_par^2 = (lam - 1)^2 <Z^2> + <o^2>`` splits the uncorrected error in quadrature into a gauge part, along the truth's
scale direction, and an off-orbit term orthogonal to it. These are generally not the gain and the remainder of the
best rescaling of the prediction about the camera centre (:func:`kappa_star`, :func:`e_par_min`). See
:mod:`alongray.core` for the definitions.
"""

from alongray.aligned import (
    SimilarityFit,
    chamfer_cm2,
    mpjpe_root_aligned,
    object_cd_after_scaled_icp,
    pa_mpjpe,
    scaled_icp,
    umeyama_similarity,
)
from alongray.change import WindowChange, summarize_changes, window_change
from alongray.core import (
    MIN_ROWS,
    AlongRay,
    EParSplit,
    GaugeDecomposition,
    along_ray_components,
    apply_gauge,
    e_par_min,
    e_par_split,
    gauge_decomposition,
    kappa_star,
)
from alongray.pool import (
    NOT_ESTIMABLE,
    cluster_bootstrap,
    pool_rows,
    pool_splits,
    pooled_shares,
    split_from_moments,
    split_of_row,
)
from alongray.scene import SceneGauge, scene_gauge
from alongray.score import (
    BodyScore,
    Score,
    intrinsics_matrix,
    reference_points,
    reprojection_error_px,
    score,
    score_body,
)
from alongray.sensitivity import ALIGNMENTS, aligned_errors, gauge_sensitivity

__version__ = "0.4.1"

__all__ = [
    "ALIGNMENTS",
    "MIN_ROWS",
    "NOT_ESTIMABLE",
    "AlongRay",
    "BodyScore",
    "EParSplit",
    "GaugeDecomposition",
    "SceneGauge",
    "Score",
    "SimilarityFit",
    "WindowChange",
    "__version__",
    "aligned_errors",
    "along_ray_components",
    "apply_gauge",
    "chamfer_cm2",
    "cluster_bootstrap",
    "e_par_min",
    "e_par_split",
    "gauge_decomposition",
    "gauge_sensitivity",
    "intrinsics_matrix",
    "kappa_star",
    "mpjpe_root_aligned",
    "object_cd_after_scaled_icp",
    "pa_mpjpe",
    "pool_rows",
    "pool_splits",
    "pooled_shares",
    "reference_points",
    "reprojection_error_px",
    "scaled_icp",
    "scene_gauge",
    "score",
    "score_body",
    "split_from_moments",
    "split_of_row",
    "summarize_changes",
    "umeyama_similarity",
    "window_change",
]
