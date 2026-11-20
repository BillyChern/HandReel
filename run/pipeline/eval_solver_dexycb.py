"""Evaluate the M1 trajectory solver (`solve_trajectories`) on DexYCB with oracle or real inputs.

Oracle mode (`dexycb.mode=oracle`, the default) uses ground-truth object segmentation,
ground-truth depth and GT-mesh-derived tracks instead of a real perception front end (SAM 2 /
MoGe / CoTracker), to isolate the solver's own accuracy from perception error. Real mode
(`dexycb.mode=real`) drives the actual open-weights front end via
`fastdo.io.datasets.dexycb_real.build_real_observations`: ground truth is used only for the
keyframe mask prompt and the hand-scale anchor (see that module's docstring), never as a
per-frame oracle signal. Both modes report mean F-5 / F-10 / Chamfer under three protocols
(`fastdo.bench.evaluate_sequence_protocols`): strict (camera frame, no alignment), scaled ICP per
frame and scaled ICP per clip. The DO AS I DO baseline's reported numbers (F-5 0.71, F-10 0.93,
Chamfer 0.66) are printed only beside the per-frame scaled-ICP block, the protocol its Table 2
follows. `dexycb.protocol_metrics=false` skips the two ICP protocols (their fields are then null).

`dexycb.batch_size` solves that many sequences at once as one decoupled batch; each row's
`seconds` is then the batch wall time divided by its number of clips (`batch_clips`). The default 1
solves sequences one at a time. A batch's sequences are prepared (observations, in real mode the front end, and the
initialization) just before that batch is solved, so at most one batch of prepared inputs is held at once. If a
batched solve raises one of `_SKIPPED_ERRORS`, its sequences are retried one at a time (those rows have
`batch_clips` 1), and a sequence that fails alone is skipped and listed in `failed_sequences`, as before batching.
`seconds` stops after the results reach the host (see `TrajectorySolution`), so it is not like-for-like with
figures measured before batching. `batch_clips` is the sub-batch the solver planned (`TrajectorySolution.plan`),
so a batch that the CUDA budget or the card's free memory split is not reported as one; `card_limited`
marks a split by the card.

Examples:
  uv run python run/pipeline/eval_solver_dexycb.py \
      paths.data=${FASTDO_ROOT}/data paths.outputs=${FASTDO_ROOT}/outputs \
      dexycb.max_sequences=3 dexycb.mode=real
  uv run python run/pipeline/eval_solver_dexycb.py \
      paths.data=${FASTDO_ROOT}/data paths.outputs=${FASTDO_ROOT}/outputs \
      dexycb.max_sequences=3 dexycb.batch_size=3
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import trimesh
import yaml
from omegaconf import DictConfig
from torch import Tensor

from fastdo.bench import Profiler, SequenceMetrics, evaluate_sequence, sample_mesh_points
from fastdo.bench.confidence_calibration import frame_translation_errors
from fastdo.bench.dexycb_eval import evaluate_sequence_protocols
from fastdo.bench.dexycb_report import PROTOCOL_KEYS, summary_lines
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.geometry.lie import se3_exp
from fastdo.io.clip_record import ClipRecord
from fastdo.io.datasets.dexycb import DexYCBReader
from fastdo.io.datasets.dexycb_frames import DexYCBFrames
from fastdo.io.datasets.dexycb_real import build_real_observations, load_real_inputs
from fastdo.shape.mesh_utils import canonical_samples
from fastdo.solver import (
    GpuBudgetError,
    KeyframeRegistration,
    MeshSamples,
    SolverConfig,
    TrajectoryObservations,
    TrajectorySolution,
    anchors_from_pose,
    gate_init_poses,
    pnp_initialize,
    project,
    register_keyframe_pose,
    solve_trajectories,
)
from fastdo.solver import transform_points as _tf_points

logger = logging.getLogger(__name__)

_BASELINE = {"f5": 0.71, "f10": 0.93, "chamfer_cm2": 0.66}
_BASELINE_LABEL = "DO AS I DO Table 2 (scaled-ICP protocol, their shapes)"
_STRIDE = 3                # DexYCB is 30 fps; every 3rd frame -> 10 fps, matching the engine.
_N_ANCHORS = 64
_N_MESH = 2048
_DEPTH_TOL_M = 0.01        # 1 cm oracle visibility depth test.
_N_TRACK_QUERIES = 96      # mode="real": CoTracker query points sampled in the keyframe object mask.
_N_ANCHOR_SURFACE = 16384  # mode="real": dense mesh samples the keyframe query pixels are anchored on.
# mode="real" keyframe: the first 10 fps frame. Fixed, so no ground truth decides which frame is prompted;
# DexYCB clips open with the object resting on the table, before the hand reaches and occludes it.
_REAL_KEYFRAME_INDEX = 0
# A sequence that raises one of these is skipped and listed in `failed_sequences` (per-sequence semantics); anything
# else aborts. `torch.linalg.LinAlgError` from main's per-sequence loop, `GpuBudgetError` from the batched solve.
_SKIPPED_ERRORS: tuple[type[Exception], ...] = (FileNotFoundError, ValueError, KeyError, torch.linalg.LinAlgError,
                                                GpuBudgetError)


def _grasped_object_id(meta: dict[str, Any]) -> int:
    """The grasped object's YCB class id from a DexYCB `meta.yml` (`ycb_ids[ycb_grasp_ind]`)."""
    ycb_ids = list(meta["ycb_ids"])
    return int(ycb_ids[int(meta["ycb_grasp_ind"])])


def build_oracle_observations(masks: Tensor, hand_masks: Tensor, pointmaps: Tensor, object_poses: Tensor,
                              K: Tensor, anchor_points: Tensor, depth_tol: float = _DEPTH_TOL_M
                              ) -> TrajectoryObservations:
    """Builds oracle `TrajectoryObservations` from GT masks/depth/poses and GT-mesh anchors.

    `tracks` are `anchor_points` (J,3, canonical/metric object frame) projected into every frame
    with the ground-truth object pose and camera intrinsics. `track_vis` requires the projected
    pixel to (a) land inside the frame in front of the camera, (b) fall inside the GT object mask,
    and (c) be within `depth_tol` metres of the recorded GT depth at that pixel -- a coarse
    (self-)occlusion test: an anchor on the far/occluded side of the object projects behind
    whatever surface the depth camera actually saw at that pixel.
    """
    T, H, W = masks.shape
    if object_poses.shape[0] != T:
        raise ValueError(f"object_poses has {object_poses.shape[0]} frames, masks has {T}")
    cam_pts = _tf_points(anchor_points.double(), object_poses.double(), 1.0)   # (T,J,3)
    raw_tracks = project(cam_pts, K.double())                                  # (T,J,2) float64
    u = raw_tracks[..., 0].round().long()
    v = raw_tracks[..., 1].round().long()
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (cam_pts[..., 2] > 0)
    uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
    frame_idx = torch.arange(T, device=masks.device).view(T, 1).expand_as(uc)
    mask_hit = masks[frame_idx, vc, uc]
    depth_at = pointmaps[frame_idx, vc, uc, 2].double()
    depth_ok = (depth_at - cam_pts[..., 2]).abs() < depth_tol   # NaN (invalid depth) compares False
    track_vis = in_bounds & mask_hit & depth_ok
    return TrajectoryObservations(intrinsics=K, masks=masks, hand_masks=hand_masks, pointmaps=pointmaps,
                                  tracks=raw_tracks.float(), track_vis=track_vis, anchors=anchor_points)


def _perturb_poses(poses: Tensor, seed: int, trans_std: float = 0.02, rot_std_deg: float = 10.0) -> Tensor:
    """GT poses perturbed by ~2 cm / ~10 deg (isotropic Gaussian direction), for the `gt_perturbed`
    initialization ablation: `T_init = Exp(xi) @ T_gt` (left perturbation)."""
    g = torch.Generator().manual_seed(seed)
    t = poses.shape[0]
    xi = torch.zeros(t, 6, dtype=torch.float64)
    xi[:, :3] = trans_std * torch.nn.functional.normalize(torch.randn(t, 3, generator=g).double(), dim=1)
    xi[:, 3:] = math.radians(rot_std_deg) * torch.nn.functional.normalize(torch.randn(t, 3, generator=g).double(), dim=1)
    return (se3_exp(xi) @ poses.double()).float()


def _sub_record(record: ClipRecord, idx: np.ndarray) -> ClipRecord:
    """`record` restricted to frames `idx` (hand tracks dropped: unused by `evaluate_sequence`)."""
    sub = ClipRecord(clip_id=record.clip_id, frame_idx=record.frame_idx[idx], timestamps=record.timestamps[idx],
                     intrinsics=record.intrinsics, cam_to_world=record.cam_to_world[idx],
                     object_mesh_path=record.object_mesh_path, object_pose=record.object_pose[idx],
                     object_scale=record.object_scale, gravity_cam=record.gravity_cam,
                     confidence=record.confidence[idx], gate_score=record.gate_score,
                     hand_left=None, hand_right=None)
    sub.validate()
    return sub


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class SequenceRow:
    sequence: str
    n_frames: int
    f5: float
    f10: float
    chamfer_cm2: float
    scale: float
    seconds: float
    iterations: int
    mean_confidence: float
    shape_source: str          # the predicted shape scored; "gt_mesh": the GT mesh itself (oracle shape)
    batch_clips: int           # clips in the planned sub-batch this row was solved in (1 after a retry or split)
    card_limited: bool = False  # the card's free memory narrowed that sub-batch
    # f5 / f10 / chamfer_cm2 above are the strict metric; these are the scaled-ICP protocols
    # (None when dexycb.protocol_metrics=false).
    f5_icp_frame: float | None = None
    f10_icp_frame: float | None = None
    chamfer_icp_frame_cm2: float | None = None
    f5_icp_clip: float | None = None
    f10_icp_clip: float | None = None
    chamfer_icp_clip_cm2: float | None = None
    # Per solved frame, the solver's confidence and |t_pred - t_gt| (m; None without a GT pose).
    frame_confidence: tuple[float, ...] = ()
    frame_translation_error_m: tuple[float | None, ...] = ()


def _sequence_row(seq: str, record: ClipRecord, gt_vertices: np.ndarray, gt_faces: np.ndarray,
                  sol: TrajectorySolution, *, protocol_metrics: bool = False, batch_clips: int = 1,
                  card_limited: bool = False) -> SequenceRow:
    """The results row of one solve of the GT mesh: strict metrics, plus both scaled-ICP protocols if enabled
    (`protocol_metrics`, off by default; the CLI passes `dexycb.protocol_metrics`, true in the config).
    `batch_clips` / `card_limited` describe the planned sub-batch the clip was solved in (`_solve_batch`)."""
    poses = sol.poses.numpy()
    frame: SequenceMetrics | None = None
    clip: SequenceMetrics | None = None
    if protocol_metrics:
        by_protocol = evaluate_sequence_protocols(record, gt_vertices, gt_faces, gt_vertices, gt_faces, poses,
                                                  sol.scale, solver_seconds=sol.seconds)
        strict, frame, clip = by_protocol["none"], by_protocol["icp_frame"], by_protocol["icp_clip"]
    else:
        strict = evaluate_sequence(record, gt_vertices, gt_faces, gt_vertices, gt_faces, poses, sol.scale,
                                   solver_seconds=sol.seconds)
    return SequenceRow(sequence=seq, n_frames=strict.n_frames, f5=strict.f5, f10=strict.f10,
                       chamfer_cm2=strict.chamfer_cm2, scale=sol.scale, seconds=sol.seconds,
                       iterations=sol.iterations, mean_confidence=float(sol.confidence.mean()), shape_source="gt_mesh",
                       batch_clips=batch_clips, card_limited=card_limited,
                       f5_icp_frame=None if frame is None else frame.f5,
                       f10_icp_frame=None if frame is None else frame.f10,
                       chamfer_icp_frame_cm2=None if frame is None else frame.chamfer_cm2,
                       f5_icp_clip=None if clip is None else clip.f5,
                       f10_icp_clip=None if clip is None else clip.f10,
                       chamfer_icp_clip_cm2=None if clip is None else clip.chamfer_cm2,
                       frame_confidence=tuple(float(c) for c in sol.confidence.tolist()),
                       frame_translation_error_m=tuple(frame_translation_errors(poses, record.object_pose,
                                                                               record.confidence > 0)))


@dataclass(frozen=True)
class PreparedSequence:
    """One sequence's solver inputs (oracle or real observations, initial poses and scale) plus what scoring needs."""

    sequence: str
    record: ClipRecord
    gt_vertices: np.ndarray
    gt_faces: np.ndarray
    mesh: MeshSamples
    obs: TrajectoryObservations
    init_poses: Tensor
    init_scale: float = 1.0


def _prepare_sequence(reader: DexYCBReader, frames: DexYCBFrames, seq: str, dexycb_root: Path, mode: str,
                      init_mode: str, max_frames: int, seed: int, profiler: Profiler) -> PreparedSequence:
    """`seq`'s `mode` observations (oracle or real) and its `init_mode` initialization, ready for `_solve_batch`."""
    record = reader.load(seq)
    subject, sequence, _serial = seq.split("/")
    meta = yaml.safe_load((dexycb_root / subject / sequence / "meta.yml").read_text())
    object_id = _grasped_object_id(meta)

    K = torch.as_tensor(record.intrinsics)
    idx_all = np.arange(record.n_frames, dtype=np.int64)[::_STRIDE]

    gt_mesh = cast("trimesh.Trimesh", trimesh.load(record.object_mesh_path, process=False, force="mesh"))
    gt_vertices = np.asarray(gt_mesh.vertices, dtype=np.float64)
    gt_faces = np.asarray(gt_mesh.faces, dtype=np.int64)
    mesh_points, mesh_normals = canonical_samples(gt_vertices, gt_faces, _N_MESH, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    registration: KeyframeRegistration | None = None
    if mode == "real":
        idx = idx_all[:max_frames]
        sub = _sub_record(record, idx)
        obs, anchor_valid, registration = _real_observations(record, frames, seq, idx, object_id, K, gt_mesh,
                                                             mesh_points, mesh_normals, seed, device, profiler)
        anchor_points = obs.anchors
        object_poses = torch.as_tensor(sub.object_pose)  # GT: read only by the gt/gt_perturbed init ablations
    elif mode == "oracle":
        masks_full, hands_full, pmaps_full = frames.load_observations(seq, K, object_id=object_id, stride=_STRIDE)
        n_use = min(len(idx_all), int(masks_full.shape[0]), max_frames)
        idx, masks, hands, pmaps = idx_all[:n_use], masks_full[:n_use], hands_full[:n_use], pmaps_full[:n_use]
        sub = _sub_record(record, idx)
        anchor_points = torch.as_tensor(sample_mesh_points(gt_vertices, gt_faces, _N_ANCHORS, seed=seed + 1),
                                        dtype=torch.float32)
        anchor_valid = torch.ones(_N_ANCHORS, dtype=torch.bool)
        object_poses = torch.as_tensor(sub.object_pose)
        obs = build_oracle_observations(masks, hands, pmaps, object_poses, K, anchor_points)
    else:
        raise ValueError(f"unknown dexycb.mode {mode!r} (expected 'oracle' or 'real')")

    init_scale = 1.0
    if init_mode == "gt_perturbed":
        init_poses = _perturb_poses(object_poses, seed=seed)
    elif init_mode == "gt":
        init_poses = object_poses                     # exact ground truth, no perturbation (the oracle gate)
    elif registration is None:
        with profiler.stage(f"{seq}/pnp_init"):
            init_poses, _n_inliers = pnp_initialize(anchor_points, anchor_valid, obs.tracks, obs.track_vis, K,
                                                    scale=1.0)
    else:
        with profiler.stage(f"{seq}/pnp_init"):
            init_poses = _real_init_poses(obs, anchor_valid, K, registration, mesh_points, _REAL_KEYFRAME_INDEX)
        init_scale = registration.scale               # 1.0: rigid registration of the metric mesh

    return PreparedSequence(sequence=seq, record=sub, gt_vertices=gt_vertices, gt_faces=gt_faces,
                            mesh=MeshSamples(points=mesh_points, normals=mesh_normals), obs=obs,
                            init_poses=init_poses, init_scale=init_scale)


def _real_observations(record: ClipRecord, frames: DexYCBFrames, seq: str, idx: np.ndarray, object_id: int,
                       K: Tensor, gt_mesh: trimesh.Trimesh, mesh_points: Tensor, mesh_normals: Tensor, seed: int,
                       device: str, profiler: Profiler
                       ) -> tuple[TrajectoryObservations, Tensor, KeyframeRegistration]:
    """`mode=real` observations with anchors on the object mesh, their validity, and the keyframe registration.

    Ground truth enters only through `load_real_inputs` (keyframe labels as the SAM 2 prompt, hand
    joints as the metric-scale anchor). The keyframe pose comes from a rigid registration of the object
    mesh (the shape, as in oracle mode) to the keyframe's SAM 2 mask and MoGe-2 pointmap, and the
    CoTracker queries are anchored on that posed mesh so `pnp_initialize` and the track term see
    mesh-frame anchors. (A similarity registration, `estimate_scale=True`, was measured no better on the
    three gate sequences: the mesh is metric here, and partial views of box-like objects leave the
    fitted scale poorly conditioned.)
    """
    hand = record.hand_right if record.hand_right is not None else record.hand_left
    if hand is None:
        raise ValueError(f"{seq}: no hand track to anchor the MoGe-2 metric scale on")
    k = _REAL_KEYFRAME_INDEX
    inputs = load_real_inputs(frames, seq, idx, k, hand)
    real = build_real_observations(inputs.frames, inputs.seg, object_id, inputs.joint_3d_cam, inputs.joint_valid,
                                   K, k, n_track_queries=_N_TRACK_QUERIES, device=device, profiler=profiler,
                                   stage_prefix=seq)
    obs = real.observations
    with profiler.stage(f"{seq}/keyframe_register"):
        observed = obs.pointmaps[k][obs.masks[k] & torch.isfinite(obs.pointmaps[k]).all(-1)]
        reg = register_keyframe_pose(mesh_points, mesh_normals, observed, K, obs.masks[k], scale=1.0,
                                     seed=seed, device=device)
        dense_points, dense_normals = canonical_samples(np.asarray(gt_mesh.vertices), np.asarray(gt_mesh.faces),
                                                        _N_ANCHOR_SURFACE, seed=seed + 2)
        hw = (int(obs.masks.shape[1]), int(obs.masks.shape[2]))
        anchors, valid = anchors_from_pose(dense_points, dense_normals, reg.pose, reg.scale, K, hw, obs.tracks[k])
    logger.info("%s: keyframe registration scale %.4f score %.4f m, %d/%d queries anchored on the mesh", seq,
                reg.scale, reg.score, int(valid.sum()), int(valid.numel()))
    return real.anchored(anchors, valid), valid, reg


def _real_init_poses(obs: TrajectoryObservations, anchor_valid: Tensor, K: Tensor, reg: KeyframeRegistration,
                     mesh_points: Tensor, keyframe_index: int) -> Tensor:
    """`mode=real` initialization: PnP on the real tracks, gated against each frame's own pointmap
    (`gate_init_poses`); when PnP solves no frame at all, every frame starts from the gated keyframe pose."""
    T = obs.tracks.shape[0]
    try:
        poses, n_inliers = pnp_initialize(obs.anchors, anchor_valid, obs.tracks, obs.track_vis, K, scale=reg.scale)
    except ValueError:
        logger.warning("pnp_initialize solved no frame; seeding every frame from the keyframe registration")
        poses, n_inliers = reg.pose[None].expand(T, 4, 4).clone(), torch.zeros(T, dtype=torch.int64)
    gated, _accepted = gate_init_poses(poses, n_inliers, mesh_points, reg.scale, obs, keyframe_index, reg.pose)
    return gated


def _failure(sequence: str, exc: BaseException) -> dict[str, str]:
    """One `failed_sequences` entry of the report."""
    return {"sequence": sequence, "error": f"{type(exc).__name__}: {exc}"}


def run_isolated[Item, Row](items: list[Item], run: Callable[[list[Item]], list[Row]],
                            name: Callable[[Item], str], failed: list[dict[str, str]] | None = None) -> list[Row]:
    """`run(items)` with per-sequence failure isolation (BASE's skip semantics).

    If the call raises one of `_SKIPPED_ERRORS` (`GpuBudgetError` included), every item is retried alone, so
    one bad sequence does not take its batch-mates down. An item that fails alone is logged and skipped, and recorded
    in `failed` when given (the report's `failed_sequences`). Any other exception propagates."""
    try:
        return run(items)
    except _SKIPPED_ERRORS as exc:
        if len(items) == 1:
            logger.exception("sequence %s failed; skipping", name(items[0]))
            if failed is not None:
                failed.append(_failure(name(items[0]), exc))
            return []
        logger.exception("batch %s failed; retrying its %d sequences one at a time",
                         "+".join(name(item) for item in items), len(items))
    return [row for item in items for row in run_isolated([item], run, name, failed)]


def _solve_batch(batch: list[PreparedSequence], solver_cfg: SolverConfig, init_mode: str, profiler: Profiler,
                 *, protocol_metrics: bool = False, failed: list[dict[str, str]] | None = None) -> list[SequenceRow]:
    """One `solve_trajectories` call over `batch` (its errors reach `run_isolated`), then per-sequence scoring
    (`_sequence_row` with `protocol_metrics`); a sequence whose scoring raises one of `_SKIPPED_ERRORS` is skipped
    without solving the batch again, and recorded in `failed` when given."""
    frames = str(sum(p.init_poses.shape[0] for p in batch))
    with profiler.stage("+".join(p.sequence for p in batch), mode=init_mode, n_frames=frames):
        sols = solve_trajectories([p.mesh for p in batch], [p.obs for p in batch], [p.init_poses for p in batch],
                                  [p.init_scale for p in batch], solver_cfg)
    rows: list[SequenceRow] = []
    for prep, sol in zip(batch, sols, strict=True):
        clips = len(sol.plan.clips) if sol.plan is not None else len(batch)     # the planned sub-batch
        try:
            rows.append(_sequence_row(prep.sequence, prep.record, prep.gt_vertices, prep.gt_faces, sol,
                                      protocol_metrics=protocol_metrics, batch_clips=clips,
                                      card_limited=sol.plan is not None and sol.plan.card_limited))
        except _SKIPPED_ERRORS as exc:
            logger.exception("scoring sequence %s failed; skipping", prep.sequence)
            if failed is not None:
                failed.append(_failure(prep.sequence, exc))
    return rows


def _stage_seconds(profiler: Profiler, seqs: list[str]) -> dict[str, float]:
    """Mean seconds per stage kind over sequences (`<seq>/<kind>` stages; the bare `<seq>` stage is the solve).

    A batched solve's stage is named by its sequences joined with "+" (`_solve_batch`); it counts once per sequence,
    at its wall time divided by its number of sequences."""
    known = set(seqs)
    by_kind: dict[str, list[float]] = {}
    for rec in profiler.report().stages:
        parts = rec.name.split("+")
        if all(part in known for part in parts):
            by_kind.setdefault("solver", []).extend([rec.seconds / len(parts)] * len(parts))
        else:
            by_kind.setdefault(rec.name.rsplit("/", 1)[-1], []).append(rec.seconds)
    return {kind: float(np.mean(v)) for kind, v in by_kind.items()}


def _select_diverse_sequences(dexycb_root: Path, seqs: list[str], n: int) -> list[str]:
    """Up to `n` sequences, preferring distinct grasped objects (one per object where possible)."""
    if n <= 0 or n >= len(seqs):
        return seqs
    chosen: list[str] = []
    object_of: dict[str, int] = {}
    seen_objects: set[int] = set()
    for seq in seqs:
        subject, sequence, _serial = seq.split("/")
        key = f"{subject}/{sequence}"
        if key not in object_of:
            meta = yaml.safe_load((dexycb_root / subject / sequence / "meta.yml").read_text())
            object_of[key] = _grasped_object_id(meta)
        obj_id = object_of[key]
        if obj_id not in seen_objects:
            seen_objects.add(obj_id)
            chosen.append(seq)
            if len(chosen) >= n:
                return chosen
    for seq in seqs:
        if seq not in chosen:
            chosen.append(seq)
            if len(chosen) >= n:
                break
    return chosen


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    orig_cwd = Path(hydra.utils.get_original_cwd())
    dexycb_root = _resolve(orig_cwd, str(cfg.dexycb.root))
    mano_pca = _resolve(orig_cwd, str(cfg.dexycb.mano_pca))
    out_dir = _resolve(orig_cwd, str(cfg.paths.outputs)) / "eval"
    mode = str(cfg.dexycb.mode)
    init_mode = str(cfg.dexycb.init)
    max_frames = int(cfg.dexycb.max_frames)
    max_sequences = int(cfg.dexycb.max_sequences)
    batch_size = max(1, int(cfg.dexycb.get("batch_size", 1)))
    seed = int(cfg.seed)
    protocol_metrics = bool(cfg.dexycb.protocol_metrics)
    fused = bool(cfg.dexycb.get("fused_sinkhorn", False))

    reader = DexYCBReader(dexycb_root, mano_pca=mano_pca)
    frames = DexYCBFrames(dexycb_root)
    all_seqs = reader.sequences()
    seqs = _select_diverse_sequences(dexycb_root, all_seqs, max_sequences)
    logger.info("evaluating %d/%d sequences (mode=%s, init=%s, max_frames=%d, batch_size=%d)", len(seqs),
                len(all_seqs), mode, init_mode, max_frames, batch_size)

    profiler = Profiler(clip_id=f"dexycb_{mode}", system=f"m1_solver_{init_mode}", n_frames=max_frames, fps=10.0)
    solver_cfg = SolverConfig(device="cuda" if torch.cuda.is_available() else "cpu", seed=seed, fused_sinkhorn=fused)
    rows: list[SequenceRow] = []
    failed: list[dict[str, str]] = []

    def solve(batch: list[PreparedSequence]) -> list[SequenceRow]:
        return _solve_batch(batch, solver_cfg, init_mode, profiler, protocol_metrics=protocol_metrics, failed=failed)

    pending: list[PreparedSequence] = []
    for i, seq in enumerate(seqs):
        try:
            pending.append(_prepare_sequence(reader, frames, seq, dexycb_root, mode, init_mode, max_frames, seed,
                                             profiler))
        except _SKIPPED_ERRORS as exc:
            logger.exception("sequence %s failed; skipping", seq)
            failed.append(_failure(seq, exc))
        if pending and (len(pending) == batch_size or i == len(seqs) - 1):
            rows.extend(run_isolated(pending, solve, lambda p: p.sequence, failed))
            pending = []
    if not rows:
        raise RuntimeError("no sequence evaluated successfully")
    if failed:
        logger.warning("%d/%d sequences failed; the mean covers only the %d evaluated ones", len(failed),
                       len(seqs), len(rows))

    keys: tuple[str, ...] = ("f5", "f10", "chamfer_cm2", "scale", "seconds", "iterations", "mean_confidence")
    keys += PROTOCOL_KEYS if protocol_metrics else ()
    means = {k: float(np.mean([getattr(r, k) for r in rows])) for k in keys}
    mean_row: dict[str, float | int | str | None] = {**means, **{k: None for k in PROTOCOL_KEYS if k not in means},
                                                     "sequence": "mean", "batch_size": batch_size,
                                                     "fused_sinkhorn": int(fused),
                                                     "min_batch_clips": min(r.batch_clips for r in rows),
                                                     "card_limited": int(any(r.card_limited for r in rows)),
                                                     "n_frames": int(np.mean([r.n_frames for r in rows])),
                                                     "n_evaluated": len(rows), "n_failed": len(failed)}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dexycb_{mode}.json"
    report = {"mode": mode, "init_mode": init_mode, "confidence_scale": solver_cfg.confidence_scale,
              "confidence_mass_floor": solver_cfg.confidence_mass_floor, "rows": [asdict(r) for r in rows],
              "mean": mean_row, "failed_sequences": failed, "stage_seconds_mean": _stage_seconds(profiler, seqs)}
    out_path.write_text(json.dumps(report, indent=2))
    profile_path = out_dir / f"dexycb_{mode}_profile.json"
    profiler.report().to_json(profile_path)

    for line in summary_lines(means, len(rows), len(failed), _BASELINE, _BASELINE_LABEL,
                              shape_sources=[r.shape_source for r in rows]):
        print(line)
    logger.info("wrote %s and %s", out_path, profile_path)


if __name__ == "__main__":
    main()
