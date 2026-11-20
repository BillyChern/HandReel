"""The `solve` stage worker: real-mode initialization and M1 solve, on the engine's artifacts.

`solve` runs after `tracks` (`fastdo.engine.manifest.STAGE_NAMES`). It repeats `run/pipeline/eval_solver_dexycb.py`'s
real mode (`_real_observations`, `_real_init_poses`, then `solve_trajectory`) in the same order:

1. the per-process CUDA cap, before any allocation (`WorkerContext.budget`);
2. a stale `solve.npz` and `solve.json` are removed, so a failed solve leaves nothing for `qc` to read;
3. `read_solve_inputs`: every per-row artifact must have the decode rows and `tracks.npz` must sit on the
   mask's keyframe; `assemble_real_observations` builds the unanchored observations;
4. the keyframe's masked, finite pointmap points;
5. the registration's initial scale (`registration_init_scale`);
6. `register_keyframe_pose` of the shape stage's `samples.npz`;
7. `anchors_from_pose` of the keyframe track pixels on `canonical_samples(mesh.obj, n_anchor_surface, seed + 2)`;
8. `RealObservations.anchored`;
9. `pnp_initialize`; when it solves no frame, every frame starts from the registration (logged at WARNING);
10. `gate_init_poses`;
11. `solve_fn(mesh, obs, init, reg.scale, cfg.solver_config())` (default `solve_trajectory`);
12. the finiteness check, `write_solve_artifact` with `final_track_weights`, then `solve.json`.

Steps 1-10 are `prepare_solve` and step 12 is `finish_solve`, so a batched stage runs its own solve between them.
The only differences from that real mode: the inputs are stage artifacts; the mesh samples are `samples.npz` (canonical
frame, unit bounding-box diagonal), so the registration starts at `s0` instead of 1.0 (for a metric template
`1 / normalization_scale`, which is that mode's 1.0 expressed in the canonical frame); errors propagate, so
`run_clip` marks the stage failed; the poses are canonical-frame poses (`fastdo.engine.qc_calibration.
template_frame_poses` maps a template's back to its own frame); `device` is used as given; every decode row is solved;
the hands (and so the metric scale) come from HaWoR. `hand_masks` stay all False, as in that real mode.

`solve.json` (tmp file then `os.replace`, after `solve.npz`): `clip_id`, `keyframe_index`, `keyframe_frame_idx`,
`n_rows`; `valid_range` `[0, b]` only when a driftqc prefix was applied (key absent otherwise); `registration` (`scale_source`, `init_scale`, `scale`, `score`, `estimate_scale`); `n_queries`, `n_anchored`,
`n_pnp_solved` (frames with `n_inliers >= pnp_min_inliers`), `pnp_fallback`, `n_init_accepted`; `solver` (`scale`,
`iterations`, `seconds`, `overrides`); `fix_scale` and `device`; with `depth_anchor_gate` set also the gate's
`depth_anchor_gate` block (`fastdo.engine.depth_anchor_gate`; the key is absent without a gate, so solve.json is then
unchanged); with `per_frame_reanchor` set also its `per_frame_reanchor` block (flagged, the moved frames and
their translation shifts; `fastdo.engine.per_frame_reanchor`, applied in `finish_solve` after the gate and before
anything is written, so both solve paths honour it); with `fixed_scale` set also its `fixed_scale` block (the
per-window file's record: scale, instance, K, fused size); with `contact_ray_smoother` set also its
`contact_ray_smoother` block (`fastdo.engine.contact_ray_smoother`; applied in `finish_solve` after G and after the
final track weights, so `track_weight` stays the solver's IRLS weights, as the contact stage leaves them for `qc`);
with `grasp_ray_constant` set also its `grasp_ray_constant` block (`fastdo.engine.grasp_ray_constant`; applied at
the smoother's place: one log factor per clip on `poses` (and, in gauge mode, `scale`), the covariance's translation
rows/columns by the same factor).
`qc` reads only `solve.npz`.

`fixed_scale` (default None: step 5 and the solve are today's): step 5 reads the window's scale from
`<artifact_dir>/<fixed_scale.file>` (`fastdo.engine.instance_scale.read_fixed_scale`; missing = error) instead of
`registration_init_scale` (source "fixed"), and `cfg.solver_config()` carries `fix_scale=True`, so the production
registration, anchors, PnP, init gate, solve and depth-anchor gate all run at that scale and the LM never moves it.

driftqc: with `driftqc_enabled` True, `read_solve_inputs` solves on the clip's `valid_range.json` prefix, sliced through
the single slicing helper `fastdo.engine.valid_range.slice_artifact` before the decode-row check; ACCEPT (range [0,
N-1]) takes the unmodified path. With it False (the default) a leftover `driftqc.json` or `valid_range.json` is a
ValueError and nothing is ever sliced.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch
import trimesh
from torch import Tensor

from fastdo.engine.branch_select import BRANCH_KEY, apply_branch_select
from fastdo.engine.contact_ray_smoother import (
    SMOOTHER_KEY,
    ContactRaySmootherConfig,
    smoother_from_artifacts,
)
from fastdo.engine.depth_anchor_gate import GATE_KEY, apply_depth_anchor_gate
from fastdo.engine.geometry_io import is_masked, masked_object_points
from fastdo.engine.grasp_ray_constant import (
    GRASP_KEY,
    GraspRayConstantConfig,
    grasp_constant_from_artifacts,
)
from fastdo.engine.instance_scale import read_fixed_scale
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.per_frame_reanchor import REANCHOR_KEY, TRACKER_VIS_KEY, apply_per_frame_reanchor
from fastdo.engine.solve_stage_config import (
    SCALE_SOURCES,
    BranchSelectConfig,
    DepthAnchorGateConfig,
    FixedScaleConfig,
    InitGateConfig,
    PerFrameReanchorConfig,
    RegistrationConfig,
    SolveStageConfig,
)
from fastdo.engine.source_domain import refuse_declared_egocentric
from fastdo.engine.valid_range import RowRange, resolve_valid_range, slice_artifact
from fastdo.engine.worker_support import WorkerContext, load_npz, read_frame_idx, require_artifact
from fastdo.engine.workers_geometry import GEOMETRY_FILE
from fastdo.engine.workers_hands import HANDS_FILE
from fastdo.engine.workers_qc import SOLVE_FILE, check_decode_rows, write_json, write_solve_artifact
from fastdo.engine.workers_shape import (
    MESH_FILE,
    SAMPLES_FILE,
    TEMPLATE_BACKEND,
    read_mesh_samples,
    read_shape_meta,
)
from fastdo.engine.workers_video import MASK_FILE, TRACKS_FILE
from fastdo.geometry.intrinsics import read_intrinsics
from fastdo.io.datasets.dexycb_real import RealObservations, assemble_real_observations
from fastdo.shape.mesh_utils import canonical_samples
from fastdo.solver.aniso import corrected_template_points
from fastdo.solver.api import (
    MeshSamples,
    SolverConfig,
    TrajectoryObservations,
    TrajectorySolution,
    solve_trajectory,
)
from fastdo.solver.batch import ObjectPixels
from fastdo.solver.init import pnp_initialize, scale_from_extent, scale_from_lateral_extent
from fastdo.solver.keyframe import (
    KeyframeRegistration,
    anchors_from_pose,
    gate_init_poses,
    register_keyframe_hypotheses,
    register_keyframe_pose,
)
from fastdo.solver.terms import track_term

__all__ = [
    "SCALE_SOURCES",
    "SOLVE_META_FILE",
    "BranchSelectConfig",
    "ContactRaySmootherConfig",
    "DepthAnchorGateConfig",
    "FixedScaleConfig",
    "InitGateConfig",
    "PerFrameReanchorConfig",
    "RegistrationConfig",
    "SolveCall",
    "SolveFn",
    "SolveInputs",
    "SolveStageConfig",
    "apply_contact_ray_smoother",
    "apply_grasp_ray_constant",
    "call_from_registration",
    "final_track_weights",
    "finish_solve",
    "make_solve_worker",
    "prepare_solve",
    "read_solve_inputs",
    "registration_init_scale",
]

logger = logging.getLogger(__name__)

SOLVE_META_FILE = "solve.json"
# The largest factor by which a metric template may declare the object BIGGER than the
# keyframe pointmap's extent before the registration refuses. ONE-sided, see `registration_init_scale`.
# Measured on declared/observed (qc18): metric templates reach at most 2.376 (n=71), unit-normalized
# ones start at 4.353 (n=3). 3.0 sits in that gap, margins 1.26x and 1.45x -- the metric-side one is thin.
METRIC_TEMPLATE_MAX_RATIO = 3.0
SolveFn = Callable[[MeshSamples, TrajectoryObservations, Tensor, float, SolverConfig], TrajectorySolution]
FIXED_SCALE_KEY, FIXED_SOURCE = "fixed_scale", "fixed"   # solve.json block / context key, registration source
_FINITE_FIELDS = ("poses", "scale", "covariance", "per_frame_cost", "confidence")
_SHAPE_FIELDS = ("shape_log_stretch", "shape_axes", "shape_centroid")   # anisotropic solves only: checked when present


def registration_init_scale(scale_source: str, shape_meta: Mapping[str, Any], pointmap_k: Tensor, mask_k: Tensor,
                            points: Tensor) -> tuple[float, str]:
    """`(initial registration scale, resolved source)`.

    `metric_template`: `1 / normalization_scale`, the metres per canonical unit of a metric template. `extent`:
    `scale_from_extent` of the keyframe's masked pointmap (the provisional generated-shape default).
    `lateral`: `scale_from_lateral_extent` (opt-in).
    `auto`: `metric_template` for a `"template"` shape (`shape.json` `backend`), else `extent`.

    Raises:
        ValueError: On an unknown source, or `metric_template` for a shape that is not a template.
    """
    backend = shape_meta.get("backend")
    source = scale_source
    if source == "auto":
        source = "metric_template" if backend == TEMPLATE_BACKEND else "extent"
    if source == "metric_template":
        if backend != TEMPLATE_BACKEND:
            raise ValueError(f"registration.scale_source metric_template needs a {TEMPLATE_BACKEND!r} shape, but "
                             f"shape.json records backend {backend!r}: only a metric template's normalization_scale "
                             "converts canonical units to metres")
        declared = 1.0 / float(shape_meta["normalization_scale"])
        observed = scale_from_extent(pointmap_k, mask_k, points)
        # An earlier version inferred metric-ness from the BACKEND NAME, so a mesh normalized to unit diagonal read
        # as "one metre across" -- a silent 4.5x error on DO AS I DO's template.obj. The solver cannot undo
        # it: it moves scale by a median 3.1% because the image cannot constrain the scale-depth orbit.
        # So cross-check the declared scale against the one INDEPENDENT ruler already here,
        # the keyframe pointmap's extent, and refuse rather than proceed.
        #
        # The check is ONE-SIDED, because the two failure modes live on opposite sides. The first
        # version was two-sided at 2.0 (calibrated on n=5) and refused 5 of the 71 labelled clips whose
        # templates ARE metric -- in every one the template was right and the extent was wrong, reading
        # hand-held YCB objects as up to 6.5 m. Depth-edge ("flying") pixels only ever ADD extent, so that
        # pathology is always observed > declared; rank correlation of its error with the masked depth
        # spread is +0.649. A normalized mesh declares ~1 m for a ~0.2 m object: always observed < declared.
        # Hand occlusion also lowers the observed extent, but boundedly (worst metric clip: 2.376x).
        if observed > 0 and declared / observed > METRIC_TEMPLATE_MAX_RATIO:
            raise ValueError(
                f"registration: the template declares {declared:.4f} m per canonical unit "
                f"(1/normalization_scale) but the keyframe pointmap's extent implies only {observed:.4f} m -- "
                f"the template claims an object {declared / observed:.2f}x bigger than the camera sees, "
                f"above {METRIC_TEMPLATE_MAX_RATIO}x; these disagree the way a mesh normalized to unit size "
                f"rather than metric does. The solver cannot recover the scale afterwards (it moves scale by "
                f"~3%). Supply a metric mesh, or set registration.scale_source=extent or "
                f"registration.estimate_scale=true.")
        return declared, source
    if source == "extent":
        return scale_from_extent(pointmap_k, mask_k, points), source
    if source == "lateral":
        # The extent ruler with depth entering only through a median (`scale_from_lateral_extent`).
        # Opt-in until it is shown to improve FINAL quality, not just the intermediate ruler.
        return scale_from_lateral_extent(pointmap_k, mask_k, points), source
    raise ValueError(f"unknown registration.scale_source {scale_source!r}; expected one of {SCALE_SOURCES}")


def final_track_weights(anchors: Tensor, tracks: Tensor, track_vis: Tensor, solution: TrajectorySolution, K: Tensor,
                        cauchy_c: float) -> Tensor:
    """`(T,J)` float32 Cauchy IRLS weights of the solver's track term at the returned poses and scale.

    `vis / (1 + r^2 / cauchy_c^2)`, not scaled by `lam_track`: the quantity `track_inlier_ratio` thresholds at 0.5
    (residual below `cauchy_c` px). Invisible (and unanchored) tracks weigh exactly 0. An anisotropic solve's
    anchors are corrected like the template (`aniso.corrected_template_points`, the identity for a flag-off solve).
    """
    anchors = corrected_template_points(anchors.detach().cpu().double(), solution)
    term = track_term(anchors, tracks.detach().cpu().double(), track_vis.detach().cpu().bool(),
                      solution.poses.detach().cpu().double(),
                      torch.tensor(math.log(solution.scale), dtype=torch.float64), K.detach().cpu().double(), cauchy_c)
    return term.weight.float()


@dataclass(frozen=True)
class SolveInputs:
    """One clip's solve-stage inputs (`read_solve_inputs`)."""

    observations: RealObservations
    frame_idx: np.ndarray          # (T,) int64 decode rows
    keyframe_index: int
    K: Tensor                      # (3,3) float32
    mesh: MeshSamples              # samples.npz: canonical surface samples and normals
    dense_points: Tensor           # (n_anchor_surface,3) canonical samples of mesh.obj
    dense_normals: Tensor
    shape_meta: dict[str, Any]
    valid_range: RowRange | None = None   # the driftqc prefix applied, or None (identity path)
    # (H,W,3) the keyframe's pointmap. Read it, never `observations.observations.pointmaps[k]`: from a masked
    # artifact those are a NaN placeholder and the object pixels live in `object_pixels`.
    keyframe_pointmap: Tensor | None = None
    # (T,) int64 per-frame count of tracks.npz `vis` (the tracker's own visibility, every query), after slicing
    tracker_visible_count: np.ndarray | None = None


def read_solve_inputs(artifact_dir: Path, n_anchor_surface: int, seed: int,
                      driftqc_enabled: bool = False) -> SolveInputs:
    """Reads the decode rows, `mask.npz`, `geometry.npz`, `tracks.npz`, `intrinsics.npz` and the shape stage's
    `samples.npz`, `mesh.obj` and `shape.json`; `dense_*` is `canonical_samples(mesh.obj, n_anchor_surface, seed + 2)`.
    `resolve_valid_range(artifact_dir, frame_idx, driftqc_enabled)` runs first: None leaves the body unchanged; a
    `RowRange` slices the decode rows and `mask.npz`, `geometry.npz` and `tracks.npz` (`slice_artifact`) before the
    decode-row check. The sliced arrays are views and are never written to. A masked `geometry.npz` is never made
    dense: its object points become the observations' `object_pixels`, its keyframe `keyframe_pointmap`, and
    `pointmaps` is a zero-memory all-NaN (T,H,W,3) expand -- NaN is what the dense rebuild held off the mask.

    Raises:
        FileNotFoundError: If an artifact is missing (naming the stage that writes it).
        ValueError: If the rows of `mask.npz`, `geometry.npz` or `tracks.npz` are not the decode rows,
            `tracks.npz`'s `keyframe_index` differs from `mask.npz`'s, or the artifacts' shapes disagree; with
            `driftqc_enabled` False if driftqc artifacts are present, and with it True if `valid_range.json` is absent
            or does not match its `driftqc.json` and the decode rows (`resolve_valid_range`); if a masked
            `geometry.npz` does not match `mask.npz` (`masked_object_points`) or its keyframe.
    """
    artifact_dir = Path(artifact_dir)
    frame_idx = read_frame_idx(artifact_dir)
    rng = resolve_valid_range(artifact_dir, frame_idx, driftqc_enabled)
    mask = load_npz(require_artifact(artifact_dir / MASK_FILE, "mask"))
    geometry = load_npz(require_artifact(artifact_dir / GEOMETRY_FILE, "geometry"))
    tracks = load_npz(require_artifact(artifact_dir / TRACKS_FILE, "tracks"))
    if rng is not None and is_masked(geometry):
        # A masked artifact keeps only object pixels; driftqc's valid range comes from a stage that reads
        # tracked (possibly off-object) pixels, and slice_artifact knows only the dense keys. Never both.
        raise ValueError(f"{GEOMETRY_FILE} is masked but driftqc is enabled: driftqc needs dense pointmaps; "
                         "write geometry with pointmap_storage='dense'")
    if rng is not None:
        frame_idx = frame_idx[rng.as_slice()]
        mask = slice_artifact(mask, MASK_FILE, rng)
        geometry = slice_artifact(geometry, GEOMETRY_FILE, rng)
        tracks = slice_artifact(tracks, TRACKS_FILE, rng)
    for name, data in ((MASK_FILE, mask), (GEOMETRY_FILE, geometry), (TRACKS_FILE, tracks)):
        check_decode_rows(data["frame_idx"], frame_idx, name)
    k = int(mask["keyframe_index"])
    if int(tracks["keyframe_index"]) != k:
        raise ValueError(f"{TRACKS_FILE} keyframe_index {int(tracks['keyframe_index'])} is not {MASK_FILE}'s "
                         f"keyframe_index {k}: the track queries must sit on the keyframe the shape is registered on")
    K = torch.from_numpy(read_intrinsics(artifact_dir)[0])
    obj = cast("trimesh.Trimesh", trimesh.load(str(require_artifact(artifact_dir / MESH_FILE, "shape")), process=False,
                                               force="mesh"))
    dense_points, dense_normals = canonical_samples(np.asarray(obj.vertices), np.asarray(obj.faces), n_anchor_surface,
                                                    seed=seed + 2)
    masks = torch.from_numpy(mask["masks"].astype(bool))
    object_pixels = None
    if is_masked(geometry):
        points, offsets = masked_object_points(geometry, mask["masks"])
        keyframe = np.asarray(geometry["keyframe_pointmap"], dtype=np.float32)
        if (int(geometry["keyframe_index"]) != k or keyframe.shape != (*masks.shape[1:], 3)
                or not np.array_equal(points[offsets[k]:offsets[k + 1]], keyframe[masks[k].numpy()], equal_nan=True)):
            raise ValueError(f"{GEOMETRY_FILE}'s keyframe_pointmap is not {MASK_FILE}'s keyframe {k}")
        object_pixels = ObjectPixels(torch.from_numpy(points), torch.from_numpy(offsets))
        keyframe_pointmap = torch.from_numpy(keyframe)
        pointmaps = torch.full((1, *masks.shape[1:], 3), math.nan).expand(*masks.shape, 3)
    else:
        pointmaps = torch.from_numpy(np.asarray(geometry["pointmaps"], dtype=np.float32))
        keyframe_pointmap = pointmaps[k]
    observations = assemble_real_observations(
        K, masks, pointmaps, torch.from_numpy(tracks["tracks"].astype(np.float32)),
        torch.from_numpy(tracks["vis"].astype(bool)), torch.from_numpy(tracks["query_uv"].astype(np.float32)), k,
        keyframe_pointmap=keyframe_pointmap, object_pixels=object_pixels)
    return SolveInputs(observations=observations, frame_idx=frame_idx, keyframe_index=k, K=K,
                       mesh=read_mesh_samples(artifact_dir), dense_points=dense_points, dense_normals=dense_normals,
                       shape_meta=read_shape_meta(artifact_dir), valid_range=rng, keyframe_pointmap=keyframe_pointmap,
                       tracker_visible_count=np.asarray(tracks["vis"]).astype(bool).sum(1).astype(np.int64))


@dataclass(frozen=True)
class SolveCall:
    """`prepare_solve`'s result: `solve_fn`'s arguments, and in `context` what `finish_solve` needs (`frame_idx`, `K`,
    `keyframe_index`, `registration`, `scale_source`, `init_scale`, `n_queries`, `n_anchored`, `n_pnp_solved`,
    `pnp_fallback`, `n_init_accepted`, and `valid_range` only when driftqc is enabled)."""

    mesh: MeshSamples
    obs: TrajectoryObservations
    init_poses: Tensor
    init_scale: float
    context: Mapping[str, Any]


def prepare_solve(clip_id: str, artifact_dir: Path, cfg: SolveStageConfig, driftqc_enabled: bool = False) -> SolveCall:
    """Steps 1-10 of the module docstring: the solver's inputs and initialization for one clip. `driftqc_enabled` is
    passed to `read_solve_inputs`; `context["valid_range"]` (the applied prefix or None) is set only when it is True,
    so a disabled call never yields a context carrying the key."""
    WorkerContext().budget()
    artifact_dir = Path(artifact_dir)
    for name in (SOLVE_FILE, SOLVE_META_FILE):
        (artifact_dir / name).unlink(missing_ok=True)
    marks = [time.perf_counter()]      # per-phase seconds, logged (not written: artifacts stay byte-identical)
    inputs = read_solve_inputs(artifact_dir, cfg.n_anchor_surface, cfg.seed, driftqc_enabled)
    marks.append(time.perf_counter())
    real, k, K, mesh = inputs.observations, inputs.keyframe_index, inputs.K, inputs.mesh
    base = real.observations
    pointmap_k = base.pointmaps[k] if inputs.keyframe_pointmap is None else inputs.keyframe_pointmap
    observed = pointmap_k[base.masks[k] & torch.isfinite(pointmap_k).all(-1)]
    fixed = None
    if cfg.fixed_scale is None:
        init_scale, scale_source = registration_init_scale(cfg.registration.scale_source, inputs.shape_meta,
                                                           pointmap_k, base.masks[k], mesh.points)
    else:                              # the fused instance size's scale, held by the solve (`solver_config`)
        init_scale, fixed = read_fixed_scale(artifact_dir / cfg.fixed_scale.file)
        scale_source = FIXED_SOURCE
    r = cfg.registration
    if cfg.branch_select is None:
        reg = register_keyframe_pose(mesh.points, mesh.normals, observed, K, base.masks[k], scale=init_scale,
                                     estimate_scale=r.estimate_scale, scale_range=r.scale_range, n_obs=r.n_obs,
                                     n_mesh=r.n_mesh, n_iters=r.n_iters, seed=cfg.seed, device=cfg.device)
        marks.append(time.perf_counter())
        return call_from_registration(clip_id, inputs, reg, init_scale, scale_source, cfg, driftqc_enabled, fixed,
                                      marks)
    # every run's top-k distinct hypotheses; the production run's element 0 is register_keyframe_pose's result
    b = cfg.branch_select
    regs: list[tuple[tuple[int, float], int, KeyframeRegistration, float]] = []
    for offset, mult in b.runs():
        s0 = init_scale if (offset, mult) == (0, 1.0) else init_scale * mult
        hyps = register_keyframe_hypotheses(mesh.points, mesh.normals, observed, K, base.masks[k], scale=s0,
                                            estimate_scale=r.estimate_scale, scale_range=r.scale_range, n_obs=r.n_obs,
                                            n_mesh=r.n_mesh, n_iters=r.n_iters, seed=cfg.seed + offset,
                                            device=cfg.device, top_k=b.top_k, min_rot_deg=b.min_rot_deg,
                                            min_shift=b.min_shift)
        regs.extend(((offset, mult), rank, h, s0) for rank, h in enumerate(hyps))
    marks.append(time.perf_counter())
    call = call_from_registration(clip_id, inputs, regs[0][2], init_scale, scale_source, cfg, driftqc_enabled, fixed,
                                  marks)
    others = tuple(call_from_registration(f"{clip_id}#b{i}", inputs, h, s0, scale_source, cfg, driftqc_enabled, fixed)
                   for i, (_run, _rank, h, s0) in enumerate(regs[1:], start=1))
    meta = tuple(MappingProxyType({"run": run, "rank": rank, "reg_score": float(h.score)}) for run, rank, h, _ in regs)
    branch = MappingProxyType({"candidates": others, "meta": meta, "dense": (inputs.dense_points, inputs.dense_normals)})
    return dataclasses.replace(call, context=MappingProxyType({**call.context, BRANCH_KEY: branch}))


def call_from_registration(clip_id: str, inputs: SolveInputs, reg: KeyframeRegistration, init_scale: float,
                           scale_source: str, cfg: SolveStageConfig, driftqc_enabled: bool = False,
                           fixed: Mapping[str, Any] | None = None, marks: list[float] | None = None) -> SolveCall:
    """Steps 7-10 of the module docstring (anchors, `RealObservations.anchored`, PnP, init gate) from a keyframe
    registration `reg`: `prepare_solve`'s tail, and `fastdo.engine.branch_select`'s per-hypothesis call.
    `marks` are `prepare_solve`'s phase clocks (read inputs, registration); without them the timing is this tail's."""
    marks = list(marks) if marks is not None else [time.perf_counter()] * 3
    real, k, K, mesh = inputs.observations, inputs.keyframe_index, inputs.K, inputs.mesh
    base = real.observations
    hw = (int(base.masks.shape[1]), int(base.masks.shape[2]))
    anchors, valid = anchors_from_pose(inputs.dense_points, inputs.dense_normals, reg.pose, reg.scale, K, hw,
                                       base.tracks[k], max_px=cfg.anchor_max_px)
    obs = real.anchored(anchors, valid)
    marks.append(time.perf_counter())
    t = int(obs.tracks.shape[0])
    try:
        poses, n_inliers = pnp_initialize(obs.anchors, valid, obs.tracks, obs.track_vis, K, scale=reg.scale,
                                          min_inliers=cfg.pnp_min_inliers)
        fallback = False
    except ValueError:
        logger.warning("%s: pnp_initialize solved no frame; seeding every frame from the keyframe registration", clip_id)
        poses, n_inliers, fallback = reg.pose[None].expand(t, 4, 4).clone(), torch.zeros(t, dtype=torch.int64), True
    marks.append(time.perf_counter())
    g = cfg.gate
    init, accepted = gate_init_poses(poses, n_inliers, mesh.points, reg.scale, obs, k, reg.pose,
                                     min_inliers=g.min_inliers, max_offset=g.max_offset, min_points=g.min_points)
    marks.append(time.perf_counter())
    timing = dict(zip(("read_inputs", "registration", "anchors", "pnp", "gate"),
                      (b - a for a, b in itertools.pairwise(marks)), strict=True))
    context = MappingProxyType({
        "frame_idx": inputs.frame_idx, "K": K, "keyframe_index": k, "registration": reg, "scale_source": scale_source,
        "init_scale": init_scale, "n_queries": int(valid.numel()), "n_anchored": int(valid.sum()),
        "n_pnp_solved": int((n_inliers >= cfg.pnp_min_inliers).sum()), "pnp_fallback": fallback,
        "n_init_accepted": int(accepted.sum()), "timing": MappingProxyType(timing), **({"valid_range": inputs.valid_range} if driftqc_enabled else {}),
        **({} if cfg.per_frame_reanchor is None else {TRACKER_VIS_KEY: inputs.tracker_visible_count}),
        **({} if fixed is None else {FIXED_SCALE_KEY: MappingProxyType(dict(fixed))})})
    logger.info("%s: registration (%s, s0 %.4f) scale %.4f score %.4f m; %d/%d queries anchored; PnP solved %d/%d "
                "frames%s; gate kept %d", clip_id, scale_source, init_scale, reg.scale, reg.score, context["n_anchored"],
                context["n_queries"], context["n_pnp_solved"], t, " (fallback)" if fallback else "",
                context["n_init_accepted"])
    return SolveCall(mesh=mesh, obs=obs, init_poses=init, init_scale=reg.scale, context=context)


def finish_solve(clip_id: str, artifact_dir: Path, call: SolveCall, solution: TrajectorySolution,
                 cfg: SolveStageConfig, solve_seconds: float | None = None, *,
                 allow_egocentric_dead_frames: bool = False) -> None:
    """Step 12 of the module docstring: `solve.npz`, then `solve.json`.

    Raises:
        FloatingPointError: Naming the field, if `solution` has a non-finite value in `poses`, `scale`, `covariance`,
            `per_frame_cost` or `confidence`; nothing is written then.
        ValueError: If `cfg.depth_anchor_gate` is set but `call` carries no gate record (the solve path that produced
            `solution` did not run the depth-anchor gate), or the other way round; nothing is written then.
        KeyError: If `cfg.per_frame_reanchor` is set but `call` was prepared without it (no visible-track counts).
        DeclaredEgocentric: If V2's dead-frame branch would run (`per_frame_reanchor` set, `dead_frame_branch` not
            False) on a clip whose `source_domain.json` declares it egocentric, and `allow_egocentric_dead_frames` (the
            source profile's study override) is False; nothing is written then.

    With `cfg.per_frame_reanchor` set, G replaces `solution` by its re-anchored copy first (after the gate:
    `solution` is the kept one), so solve.npz, the track weights and solve.json describe the re-anchored poses.
    """
    gate = call.context.get(GATE_KEY)
    branch = call.context.get(BRANCH_KEY)
    if (branch is None) != (cfg.branch_select is None) or (branch is not None and "chosen" not in branch):
        raise ValueError(f"{clip_id}: solve_stage.branch_select is {cfg.branch_select} but the solve call "
                         f"{'carries no' if branch is None else 'carries an unselected'} branch record: this solve "
                         "path does not honour branch_select; nothing is written")
    if (gate is None) != (cfg.depth_anchor_gate is None):
        raise ValueError(f"{clip_id}: solve_stage.depth_anchor_gate is {cfg.depth_anchor_gate} but the solve call "
                         f"{'carries no' if gate is None else 'carries a'} gate record: this solve path does not "
                         "honour the gate; nothing is written")
    pfr = cfg.per_frame_reanchor
    if pfr is not None and pfr.dead_frame_branch is not False and not allow_egocentric_dead_frames:
        refuse_declared_egocentric(artifact_dir, clip_id, "V2's dead-frame branch (solve_stage.per_frame_reanchor."
                                   f"dead_frame_branch={pfr.dead_frame_branch})")
    for name in _FINITE_FIELDS + tuple(f for f in _SHAPE_FIELDS if getattr(solution, f, None) is not None):
        value = getattr(solution, name)
        finite = bool(torch.isfinite(value).all()) if isinstance(value, Tensor) else math.isfinite(float(value))
        if not finite:
            raise FloatingPointError(f"{clip_id}: the solver returned a non-finite {name}; no {SOLVE_FILE} is written")
    reanchor = None
    if cfg.per_frame_reanchor is not None:
        solution, reanchor = apply_per_frame_reanchor(call.mesh, call.obs, solution, call.context,
                                                      cfg.per_frame_reanchor, impl=cfg.per_frame_reanchor_impl)
        if not bool(torch.isfinite(solution.poses).all()):
            raise FloatingPointError(f"{clip_id}: the per-frame re-anchor produced non-finite poses; no {SOLVE_FILE} "
                                     "is written")
        logger.info("per-frame re-anchor %s: certificate %s (%s) flagged %s; %d dead frames; moved %d/%d frames, "
                    "max shift %.4f m", clip_id, reanchor["certificate"], reanchor["certificate_source"],
                    reanchor["flagged"], len(reanchor["dead_frames"]), len(reanchor["frames"]), reanchor["n_frames"],
                    reanchor["max_shift_m"])
    artifact_dir, ctx = Path(artifact_dir), call.context
    fixed = ctx.get(FIXED_SCALE_KEY)
    if (fixed is None) != (cfg.fixed_scale is None):
        raise ValueError(f"{clip_id}: solve_stage.fixed_scale is {cfg.fixed_scale} but the solve call "
                         f"{'carries no' if fixed is None else 'carries a'} fixed-scale record; nothing is written")
    solver_cfg = cfg.solver_config()
    weight = final_track_weights(call.obs.anchors, call.obs.tracks, call.obs.track_vis, solution, ctx["K"],
                                 solver_cfg.cauchy_c)
    t0 = time.perf_counter()
    smoother = None
    if cfg.contact_ray_smoother is not None:       # after G and AFTER the track weights (module docstring)
        solution, smoother = apply_contact_ray_smoother(artifact_dir, solution, cfg.contact_ray_smoother,
                                                        ctx["frame_idx"], ctx.get("valid_range"))
        logger.info("contact-ray smoother %s: side %s, %d gated rows, active %s, shift median %s", clip_id,
                    smoother.get("side"), smoother["n_gated"], smoother["active"], smoother.get("shift_median"))
    grasp = None
    if cfg.grasp_ray_constant is not None:         # at the smoother's place (they are mutually exclusive)
        solution, grasp = apply_grasp_ray_constant(artifact_dir, solution, cfg.grasp_ray_constant,
                                                   ctx["frame_idx"], ctx.get("valid_range"))
        logger.info("grasp-ray constant %s (%s): side %s, %d gated rows, active %s, log constant %.4f", clip_id,
                    cfg.grasp_ray_constant.mode, grasp.get("side"), grasp["n_gated"], grasp["active"],
                    grasp["log_constant"])
    write_solve_artifact(artifact_dir, solution, ctx["frame_idx"], weight)
    timing = {**ctx.get("timing", {}), **({} if solve_seconds is None else {"solve_fn": solve_seconds}),
              "write_npz": time.perf_counter() - t0}
    reg, frame_idx, k = ctx["registration"], ctx["frame_idx"], int(ctx["keyframe_index"])
    rng = ctx.get("valid_range")  # absent in contexts built before E4 (L-4)
    write_json(artifact_dir / SOLVE_META_FILE, {
        "clip_id": clip_id, "keyframe_index": k, "keyframe_frame_idx": int(frame_idx[k]), "n_rows": int(frame_idx.size),
        **({} if rng is None else {"valid_range": [rng.start, rng.stop]}),
        "registration": {"scale_source": ctx["scale_source"], "init_scale": float(ctx["init_scale"]),
                         "scale": float(reg.scale), "score": float(reg.score),
                         "estimate_scale": cfg.registration.estimate_scale},
        **{key: ctx[key] for key in ("n_queries", "n_anchored", "n_pnp_solved", "pnp_fallback", "n_init_accepted")},
        "solver": {"scale": float(solution.scale), "iterations": int(solution.iterations),
                   "seconds": float(solution.seconds), "overrides": dict(cfg.solver)},
        "fix_scale": cfg.fix_scale, "device": cfg.device, **({} if gate is None else {GATE_KEY: dict(gate)}),
        **({} if reanchor is None else {REANCHOR_KEY: dict(reanchor)}),
        **({} if fixed is None else {FIXED_SCALE_KEY: dict(fixed)}),
        **({} if branch is None else {BRANCH_KEY: dict(branch)}),
        **({} if smoother is None else {SMOOTHER_KEY: smoother}),
        **({} if grasp is None else {GRASP_KEY: grasp})})
    logger.info("solve timing %s: %s", clip_id, json.dumps({key: round(float(v), 4) for key, v in timing.items()}))
    logger.info("solve of %r: scale %.4f, %d fine iterations, %.2f s", clip_id, solution.scale, solution.iterations,
                solution.seconds)


def apply_contact_ray_smoother(artifact_dir: Path, solution: TrajectorySolution, cfg: ContactRaySmootherConfig,
                               frame_idx: np.ndarray, valid_range: RowRange | None = None
                               ) -> tuple[TrajectorySolution, dict[str, Any]]:
    """The contact-gated along-ray smoother (`fastdo.engine.contact_ray_smoother`) on a kept solution.

    Reads the clip's `samples.npz` points and `hands.npz` (sliced to `valid_range` when a driftqc prefix applies; its
    rows must be `frame_idx`). Only `poses` changes, and only when the smoother is active: an inactive clip gets the
    SAME `solution` object back, so its solve.npz is byte-identical to the smoother-off stage.

    Raises:
        ValueError: If `hands.npz` rows differ from the solve's rows.
    """
    artifact_dir = Path(artifact_dir)
    mesh = np.asarray(load_npz(require_artifact(artifact_dir / SAMPLES_FILE, "shape"))["points"], dtype=np.float64)
    hands = load_npz(require_artifact(artifact_dir / HANDS_FILE, "hands"))
    if valid_range is not None:
        hands = slice_artifact(hands, HANDS_FILE, valid_range)
    check_decode_rows(np.asarray(hands["frame_idx"], dtype=np.int64), np.asarray(frame_idx, dtype=np.int64),
                      HANDS_FILE)
    poses = solution.poses.detach().cpu().numpy()
    res = smoother_from_artifacts(artifact_dir, poses, float(solution.scale), mesh, hands, cfg)
    if not res.active:
        return solution, res.record
    moved = torch.from_numpy(res.poses).to(dtype=solution.poses.dtype, device=solution.poses.device)
    return dataclasses.replace(solution, poses=moved), res.record


def apply_grasp_ray_constant(artifact_dir: Path, solution: TrajectorySolution, cfg: GraspRayConstantConfig,
                             frame_idx: np.ndarray, valid_range: RowRange | None = None
                             ) -> tuple[TrajectorySolution, dict[str, Any]]:
    """The per-clip grasp-onset along-ray constant (`fastdo.engine.grasp_ray_constant`) on a kept solution.

    Reads `samples.npz` and `hands.npz` as `apply_contact_ray_smoother`. An inactive clip gets the SAME `solution`
    object back (solve.npz byte-identical to the stage without the key); an active one gets new `poses`, `scale`
    (gauge mode) and `covariance` (translation rows/columns times the factor), in the input dtypes and device.

    Raises:
        ValueError: If `hands.npz` rows differ from the solve's rows.
    """
    artifact_dir = Path(artifact_dir)
    samples = load_npz(require_artifact(artifact_dir / SAMPLES_FILE, "shape"))
    mesh = np.asarray(samples["points"], dtype=np.float64)
    normals = np.asarray(samples["normals"], dtype=np.float64)
    hands = load_npz(require_artifact(artifact_dir / HANDS_FILE, "hands"))
    if valid_range is not None:
        hands = slice_artifact(hands, HANDS_FILE, valid_range)
    check_decode_rows(np.asarray(hands["frame_idx"], dtype=np.int64), np.asarray(frame_idx, dtype=np.int64),
                      HANDS_FILE)
    poses = solution.poses.detach().cpu().numpy()
    res = grasp_constant_from_artifacts(artifact_dir, poses, float(solution.scale), mesh, normals, hands, cfg)
    if not res.active:
        return solution, res.record
    lam = float(np.exp(res.log_constant))
    jac = torch.ones(solution.covariance.shape[-1], dtype=torch.float64)
    jac[:3] = lam
    cov = solution.covariance.detach().to(torch.float64)
    cov = (cov * jac[:, None] * jac[None, :]).to(dtype=solution.covariance.dtype, device=solution.covariance.device)
    moved = torch.from_numpy(res.poses).to(dtype=solution.poses.dtype, device=solution.poses.device)
    return dataclasses.replace(solution, poses=moved, scale=float(res.scale), covariance=cov), res.record


def make_solve_worker(cfg: SolveStageConfig | None = None, solve_fn: SolveFn = solve_trajectory, *,
                      driftqc_enabled: bool = False, allow_egocentric_dead_frames: bool = False) -> StageFn:
    """The `solve` stage worker: `prepare_solve`, `solve_fn(..., cfg.solver_config())`, with `cfg.depth_anchor_gate` set
    the depth-anchor gate (`apply_depth_anchor_gate`: a flagged clip's re-solve is a second `solve_fn` call from the
    same `SolveCall`), then `finish_solve`. Building it allocates nothing.

    Args:
        cfg: The stage config (`SolveStageConfig()`: its defaults on `cuda`).
        solve_fn: The solve (`solve_trajectory`; tests inject fakes).
        driftqc_enabled: `driftqc_stage.enabled` (default False: never slices; leftover driftqc artifacts raise).
        allow_egocentric_dead_frames: Passed to `finish_solve`: only the source profile's study override
            sets it. The profile itself is applied by `fastdo.engine.source_profile.make_profiled_solve_worker`.
    """
    cfg = cfg if cfg is not None else SolveStageConfig()

    def solve_worker(clip_id: str, artifact_dir: Path) -> None:
        call = prepare_solve(clip_id, artifact_dir, cfg, driftqc_enabled)
        t0 = time.perf_counter()
        solution = solve_fn(call.mesh, call.obs, call.init_poses, call.init_scale, cfg.solver_config())
        many = lambda m, o, p, s, c: [solve_fn(*args, c) for args in zip(m, o, p, s, strict=True)]
        if cfg.depth_anchor_gate is not None:
            (call,), (solution,) = apply_depth_anchor_gate(
                [clip_id], [call], [solution], cfg.depth_anchor_gate, cfg.solver_config(), many)
        if cfg.branch_select is not None:
            (call,), (solution,) = apply_branch_select([clip_id], [call], [solution], cfg.branch_select,
                                                       cfg.solver_config(), cfg.depth_anchor_gate, many)
        finish_solve(clip_id, artifact_dir, call, solution, cfg, solve_seconds=time.perf_counter() - t0,
                     allow_egocentric_dead_frames=allow_egocentric_dead_frames)

    return solve_worker
