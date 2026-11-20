"""DexYCB reconstruction metrics: F-5, F-10 and Chamfer between posed predicted and GT meshes.

`evaluate_sequence` is the strict metric (camera frame, no alignment). `evaluate_sequence_protocols` adds the
scaled-ICP protocols of MCC-HO / DiffHOI / G-HOP, which DO AS I DO Table 2 follows, per frame and per clip.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from fastdo.bench.alignment import scaled_icp
from fastdo.bench.metrics import chamfer_distance, f_score
from fastdo.io.clip_record import ClipRecord

logger = logging.getLogger(__name__)

__all__ = ["ALIGNMENTS", "SequenceMetrics", "evaluate_sequence", "evaluate_sequence_protocols",
           "sample_mesh_points"]

ALIGNMENTS: tuple[str, ...] = ("none", "icp_frame", "icp_clip")


@dataclass(frozen=True)
class SequenceMetrics:
    f5: float
    f10: float
    chamfer_cm2: float
    n_frames: int
    solver_seconds: float


def sample_mesh_points(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def _posed(points: np.ndarray, pose: np.ndarray, scale: float) -> np.ndarray:
    return scale * points @ pose[:3, :3].T + pose[:3, 3]


def evaluate_sequence(record: ClipRecord, gt_vertices: np.ndarray, gt_faces: np.ndarray, pred_vertices: np.ndarray,
                      pred_faces: np.ndarray, pred_poses: np.ndarray, pred_scale: float, n_points: int = 10000,
                      seed: int = 0, solver_seconds: float = 0.0) -> SequenceMetrics:
    # Both point clouds are sampled with the same seed (not an independent seed + 1) so that an
    # identical mesh + identical pose ("perfect prediction") yields bit-identical point clouds and
    # therefore an exact-zero Chamfer/F-score floor. With independent seeds, sampling noise alone
    # (~2 mm mean nearest-neighbor distance at n_points=2000 on a 5 cm-radius sphere -- measured
    # on a known sphere) is comparable to the F-5 threshold (5 mm) and swamps genuine reconstruction
    # error. When gt/pred meshes differ (the real evaluation case), the differing geometry itself
    # provides independent sampling; no seed offset is needed to avoid a degenerate correspondence.
    gt_pts = sample_mesh_points(gt_vertices, gt_faces, n_points, seed)
    pred_pts = sample_mesh_points(pred_vertices, pred_faces, n_points, seed)
    f5s, f10s, cds = [], [], []
    for k in range(record.n_frames):
        if record.confidence[k] <= 0:
            continue
        gt = _posed(gt_pts, record.object_pose[k].astype(np.float64), 1.0)
        pred = _posed(pred_pts, pred_poses[k].astype(np.float64), pred_scale)
        f5s.append(f_score(pred, gt, 0.005))
        f10s.append(f_score(pred, gt, 0.010))
        cds.append(chamfer_distance(pred, gt, squared=True) * 1e4)
    if not f5s:
        raise ValueError(f"no frame with confidence > 0 in {record.clip_id}")
    m = SequenceMetrics(f5=float(np.mean(f5s)), f10=float(np.mean(f10s)), chamfer_cm2=float(np.mean(cds)),
                        n_frames=len(f5s), solver_seconds=solver_seconds)
    logger.info("%s: F5 %.3f F10 %.3f CD %.3f cm^2 over %d frames", record.clip_id, m.f5, m.f10, m.chamfer_cm2, m.n_frames)
    return m


def _nn_distances(src: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Distance from each `src` point to its nearest neighbour in the cloud of `tree`, as `fastdo.bench.metrics`."""
    d, _ = tree.query(src, k=1, workers=1)
    return np.asarray(d, dtype=np.float64)


def _f_score(d_pg: np.ndarray, d_gp: np.ndarray, threshold: float) -> float:
    """`fastdo.bench.metrics.f_score` from the pred->gt and gt->pred distances, with its exact arithmetic."""
    precision = float((d_pg <= threshold).mean())
    recall = float((d_gp <= threshold).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _frame_means(preds: list[np.ndarray], gts: list[np.ndarray], gt_trees: list[cKDTree],
                 solver_seconds: float) -> SequenceMetrics:
    """F-5 / F-10 / squared Chamfer (cm^2) per (pred, gt) frame pair, averaged, as `evaluate_sequence` scores.

    Each frame's two directed nearest-neighbour distance arrays are computed once, from one cKDTree per cloud
    (`gt_trees[k]` is the tree of `gts[k]`, built once and shared by all protocols); F-5, F-10 and the squared
    Chamfer derive from them with the arithmetic of `f_score` / `chamfer_distance`, so the numbers are bitwise
    equal to calling those (2 queries per frame instead of 6).
    """
    f5s: list[float] = []
    f10s: list[float] = []
    cds: list[float] = []
    for pred, gt, gt_tree in zip(preds, gts, gt_trees, strict=True):
        d_pg = _nn_distances(pred, gt_tree)
        d_gp = _nn_distances(gt, cKDTree(pred))
        f5s.append(_f_score(d_pg, d_gp, 0.005))
        f10s.append(_f_score(d_pg, d_gp, 0.010))
        cds.append(float((d_pg**2).mean() + (d_gp**2).mean()) * 1e4)
    return SequenceMetrics(f5=float(np.mean(f5s)), f10=float(np.mean(f10s)), chamfer_cm2=float(np.mean(cds)),
                           n_frames=len(f5s), solver_seconds=solver_seconds)


def evaluate_sequence_protocols(record: ClipRecord, gt_vertices: np.ndarray, gt_faces: np.ndarray,
                                pred_vertices: np.ndarray, pred_faces: np.ndarray, pred_poses: np.ndarray,
                                pred_scale: float, n_points: int = 10000, seed: int = 0,
                                solver_seconds: float = 0.0) -> dict[str, SequenceMetrics]:
    """The DexYCB object metrics under each alignment protocol in `ALIGNMENTS`, on one shared point sampling.

    Points are sampled and posed exactly as `evaluate_sequence` does, over the frames with confidence > 0:
    - "none": the strict camera-frame metric, bitwise equal to `evaluate_sequence`.
    - "icp_frame": each frame's posed prediction aligned to its GT by its own `scaled_icp` (the MCC-HO / DiffHOI /
      G-HOP protocol DO AS I DO Table 2 follows), then scored; forgives any per-frame similarity.
    - "icp_clip": one `scaled_icp` similarity shared by all frames (the similarity gauge), so tracking errors
      between frames still count.

    Raises:
        ValueError: no frame has confidence > 0.
    """
    t0 = time.perf_counter()
    gt_pts = sample_mesh_points(gt_vertices, gt_faces, n_points, seed)
    pred_pts = sample_mesh_points(pred_vertices, pred_faces, n_points, seed)
    keep = [k for k in range(record.n_frames) if record.confidence[k] > 0]
    if not keep:
        raise ValueError(f"no frame with confidence > 0 in {record.clip_id}")
    gts = [_posed(gt_pts, record.object_pose[k].astype(np.float64), 1.0) for k in keep]
    preds = [_posed(pred_pts, pred_poses[k].astype(np.float64), pred_scale) for k in keep]
    t_sample = time.perf_counter()
    gt_trees = [cKDTree(g) for g in gts]
    out = {"none": _frame_means(preds, gts, gt_trees, solver_seconds)}
    t_none = time.perf_counter()
    frame_fits = [scaled_icp([p], [g]) for p, g in zip(preds, gts, strict=True)]
    out["icp_frame"] = _frame_means([aligned[0] for aligned, _fit in frame_fits], gts, gt_trees, solver_seconds)
    t_frame = time.perf_counter()
    clip_aligned, clip_fit = scaled_icp(preds, gts)
    out["icp_clip"] = _frame_means(clip_aligned, gts, gt_trees, solver_seconds)
    t_clip = time.perf_counter()
    iters = [fit.iterations for _aligned, fit in frame_fits]
    logger.info("%s over %d frames: F5 none %.3f / icp_frame %.3f / icp_clip %.3f; per-frame ICP iterations mean %.1f"
                " max %d (%d not converged); clip ICP scale %.4f, %d iterations, converged %s", record.clip_id,
                len(keep), out["none"].f5, out["icp_frame"].f5, out["icp_clip"].f5, float(np.mean(iters)), max(iters),
                sum(not fit.converged for _aligned, fit in frame_fits), clip_fit.scale, clip_fit.iterations,
                clip_fit.converged)
    logger.info("%s protocol seconds: sampling %.2f, none %.2f, icp_frame %.2f, icp_clip %.2f", record.clip_id,
                t_sample - t0, t_none - t_sample, t_frame - t_none, t_clip - t_frame)
    return out
