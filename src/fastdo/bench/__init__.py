"""Benchmarking: profiler, metrics, human-study tooling."""

from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget

from .alignment import SimilarityFit, scaled_icp, umeyama_similarity
from .dexycb_eval import (
    ALIGNMENTS,
    SequenceMetrics,
    evaluate_sequence,
    evaluate_sequence_protocols,
    sample_mesh_points,
)
from .metrics import RetargetErrors, chamfer_distance, f_score, retarget_errors
from .profiler import Profiler, ProfileReport, StageRecord
from .yield_report import YieldReport, compute_yield_report

__all__ = [
    "ALIGNMENTS",
    "ProfileReport",
    "Profiler",
    "RetargetErrors",
    "SequenceMetrics",
    "SimilarityFit",
    "StageRecord",
    "YieldReport",
    "apply_cpu_budget",
    "apply_gpu_budget",
    "chamfer_distance",
    "compute_yield_report",
    "evaluate_sequence",
    "evaluate_sequence_protocols",
    "f_score",
    "retarget_errors",
    "sample_mesh_points",
    "scaled_icp",
    "umeyama_similarity",
]
