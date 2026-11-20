"""Builds `TrajectoryObservations` for DexYCB's `mode: real` evaluation front end.

Unlike `run/pipeline/eval_solver_dexycb.py`'s `build_oracle_observations` (ground-truth masks,
ground-truth depth, GT-mesh-derived tracks -- isolates the solver's own accuracy), this module
drives the actual open-weights front end: SAM 2.1 mask propagation, MoGe-2 pointmaps with a
hand-anchored metric scale, and CoTracker3 tracking. Ground truth is used in exactly two places,
each standing in for something a deployed system also gets from outside the video, never as a
per-frame oracle signal:

1. **The keyframe mask prompt.** `seg[keyframe_index] == object_id` seeds SAM 2's propagation,
   the role a click/detector prompt plays in deployment; every other frame's mask is SAM 2's own
   output, and no other frame's labels are read (`test_build_real_observations_ignores_non_
   keyframe_labels` scrambles them and checks the observations are unchanged).
2. **The hand-scale anchor.** `joint_3d_cam`/`joint_valid` (DexYCB's per-frame hand joints) feed
   `hand_anchored_scale`, which corrects MoGe's per-frame metric-scale error -- the role HaWoR
   joints play on the whisking demo (`run/pipeline/whisking_headtohead.py`).

The camera intrinsics `K` (calibration, not a label) fix MoGe's field of view, and the pointmaps are
rebuilt through `K` (`run_moge(intrinsics=K)`), so they share `K`'s focal length *and* principal point
with the track projections, the silhouette term and the hand anchor.

This builder has no shape, so it cannot anchor the CoTracker queries in the object's canonical frame,
which `pnp_initialize` and `solve_trajectory`'s track term need. It therefore returns
`RealObservations`: `observations` carries NaN anchors and an all-False `track_vis` (so it cannot be
mistaken for solver-ready input: PnP on it solves no frame), the tracks' own 2D visibility separately,
and the keyframe camera-frame points under the queries. A caller with a shape (the DexYCB
evaluation's mesh, a SAM 3D / TRELLIS mesh) anchors the queries on it
(`fastdo.solver.keyframe.register_keyframe_pose` + `anchors_from_pose`) and calls
`RealObservations.anchored(anchors, valid)` for the solver's `TrajectoryObservations`.

`hand_masks` is an all-False placeholder (no hand segmentation in this front end), so the solver's
hand-occluder exclusion is a no-op in `mode: real`.

`assemble_real_observations` is the post-model half of `build_real_observations` (everything after its
three model calls). The engine's solve stage (`fastdo.engine.workers_solve`) calls it on its stage
artifacts, so both front ends build their observations the same way.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from fastdo.bench import Profiler
from fastdo.geometry.moge_model import run_moge
from fastdo.geometry.moge_pointmaps import (
    apply_metric_scale,
    clip_median_scale,
    hand_anchored_scale,
)
from fastdo.io.clip_record import HandTrack
from fastdo.io.datasets.dexycb_frames import DexYCBFrames
from fastdo.shape.masks_sam2 import propagate_masks_sam2
from fastdo.solver import TrajectoryObservations
from fastdo.solver.batch import ObjectPixels
from fastdo.tracks.cotracker import (
    sample_query_points_in_mask,
    track_points_cotracker,
    tracks_in_masks,
)

logger = logging.getLogger(__name__)

__all__ = ["RealObservations", "RealSequenceInputs", "assemble_real_observations", "build_real_observations",
           "load_real_inputs"]

_QUERY_SEED = 0  # sample_query_points_in_mask's RNG seed: reproducible query sampling


@dataclass(frozen=True)
class RealSequenceInputs:
    """`build_real_observations`' per-sequence inputs, as read from a DexYCB camera directory."""

    frames: Tensor         # (T,H,W,3) uint8 RGB
    seg: Tensor            # (T,H,W) int64; GT labels at the keyframe only, zeros at every other frame
    joint_3d_cam: Tensor   # (T,21,3) float32 camera-frame hand joints, metres
    joint_valid: Tensor    # (T,21) bool


@dataclass(frozen=True)
class RealObservations:
    """`build_real_observations`' result: the front end's observations before the tracks are anchored.

    `observations.anchors` is all NaN and `observations.track_vis` all False until `anchored` is called
    with mesh-frame anchors, so the unanchored result gives the solver no track correspondences.
    """

    observations: TrajectoryObservations   # masks, hand_masks, metric pointmaps, tracks; no anchors yet
    track_vis: Tensor                      # (T,J) bool: tracked, inside the image and that frame's mask
    keyframe_points: Tensor                # (J,3) keyframe camera-frame metric points under the queries

    def anchored(self, anchors: Tensor, anchor_valid: Tensor) -> TrajectoryObservations:
        """Solver-ready observations with `(J,3)` canonical `anchors`, visible only where `anchor_valid`."""
        if tuple(anchors.shape) != tuple(self.keyframe_points.shape):
            raise ValueError(f"anchors must be {tuple(self.keyframe_points.shape)}, got {tuple(anchors.shape)}")
        return replace(self.observations, anchors=anchors, track_vis=self.track_vis & anchor_valid[None])


def load_real_inputs(frames: DexYCBFrames, seq: str, idx: np.ndarray, keyframe_index: int,
                     hand: HandTrack) -> RealSequenceInputs:
    """Loads the RGB frames `idx`, the keyframe's labels and the hand joints for `mode: real`.

    Only `labels_<idx[keyframe_index]>.npz` is opened: no other frame's GT segmentation is read, so
    the returned `seg` cannot leak per-frame labels into the front end.

    Args:
        frames: DexYCB per-frame file access.
        seq: `"<subject>/<sequence>/<serial>"`.
        idx: `(T,)` source frame indices to load (e.g. the 10 fps stride).
        keyframe_index: Position in `idx` of the keyframe whose labels are loaded.
        hand: The sequence's hand track (full length); joints/validity are taken at `idx`.

    Returns:
        A `RealSequenceInputs`.
    """
    paths = frames.frame_paths(seq)
    colors = np.stack([np.asarray(Image.open(paths[int(i)].color).convert("RGB"), dtype=np.uint8) for i in idx])
    seg = np.zeros(colors.shape[:3], dtype=np.int64)
    with np.load(paths[int(idx[keyframe_index])].labels) as labels:
        seg[keyframe_index] = labels["seg"]
    valid = torch.from_numpy(np.asarray(hand.valid)[idx])
    return RealSequenceInputs(frames=torch.from_numpy(colors), seg=torch.from_numpy(seg),
                              joint_3d_cam=torch.from_numpy(np.asarray(hand.joints)[idx]),
                              joint_valid=valid[:, None].expand(-1, hand.joints.shape[1]).clone())


def build_real_observations(frames: Tensor, seg: Tensor, object_id: int, joint_3d_cam: Tensor,
                            joint_valid: Tensor, K: Tensor, keyframe_index: int, n_track_queries: int = 96,
                            device: str = "cuda", profiler: Profiler | None = None,
                            stage_prefix: str = "frontend") -> RealObservations:
    """Builds real-front-end observations for one DexYCB sequence (tracks not yet anchored on a shape).

    Args:
        frames: `(T,H,W,3)` uint8 RGB frames.
        seg: `(T,H,W)` GT segmentation labels; only `seg[keyframe_index]` is read (the prompt).
        object_id: The grasped object's YCB class id (matches `seg`'s per-pixel labels).
        joint_3d_cam: `(T,J,3)` per-frame hand joints, camera frame, metres (the scale anchor).
        joint_valid: `(T,J)` bool per-joint validity.
        K: `(3,3)` camera intrinsics.
        keyframe_index: Frame index carrying the mask prompt and the track queries.
        n_track_queries: CoTracker query points sampled inside the keyframe object mask.
        device: torch device string for SAM 2 / MoGe-2 / CoTracker.
        profiler: Optional profiler; records `<stage_prefix>/{sam2,moge,cotracker}` stages.
        stage_prefix: Stage-name prefix for `profiler`.

    Returns:
        `RealObservations`: `masks` from SAM 2, `hand_masks` all False, metric `pointmaps` (MoGe-2
        rebuilt through `K`, times the hand-anchored scale, NaN where invalid), `tracks` from
        CoTracker; `track_vis` additionally requires the tracked pixel to lie inside the image and
        that frame's SAM 2 mask and the keyframe point under the query to be finite; and
        `keyframe_points` the keyframe camera-frame points under the queries.

    Raises:
        ValueError: If `keyframe_index` is out of range, `seg`'s frame count differs from
            `frames`', the keyframe mask is empty, or no frame yields a hand-anchored scale.
    """
    t = int(frames.shape[0])
    if not 0 <= keyframe_index < t:
        raise ValueError(f"keyframe_index {keyframe_index} out of range [0,{t})")
    if seg.shape[0] != t:
        raise ValueError(f"seg has {seg.shape[0]} frames, frames has {t}")

    def stage(name: str) -> AbstractContextManager[None]:
        return profiler.stage(f"{stage_prefix}/{name}") if profiler is not None else nullcontext()

    keyframe_mask = seg[keyframe_index] == object_id
    with stage("sam2"):
        masks = propagate_masks_sam2(frames, keyframe_index, keyframe_mask, device=device).to(torch.bool)

    with stage("moge"):
        raw = run_moge(frames, device=device, intrinsics=K)
    scale = clip_median_scale(hand_anchored_scale(raw, joint_3d_cam, K, joint_valid))
    pointmaps = apply_metric_scale(raw, scale)
    logger.info("hand-anchored metric scale: median %.4f, range [%.4f, %.4f]", float(scale.median()),
                float(scale.min()), float(scale.max()))

    gen = torch.Generator().manual_seed(_QUERY_SEED)
    query_uv = sample_query_points_in_mask(keyframe_mask, n_track_queries, generator=gen)
    with stage("cotracker"):
        tracks, track_vis = track_points_cotracker(frames, keyframe_index, query_uv, device=device)
    return assemble_real_observations(K, masks, pointmaps, tracks, track_vis, query_uv, keyframe_index)


def assemble_real_observations(K: Tensor, masks: Tensor, pointmaps: Tensor, tracks: Tensor, tracker_vis: Tensor,
                               query_uv: Tensor, keyframe_index: int, keyframe_pointmap: Tensor | None = None,
                               object_pixels: ObjectPixels | None = None) -> RealObservations:
    """Unanchored `RealObservations` from the front end's outputs (the post-model half of `build_real_observations`).

    Args:
        K: `(3,3)` camera intrinsics.
        masks: `(T,H,W)` bool object masks.
        pointmaps: `(T,H,W,3)` metric camera-frame pointmaps, NaN where invalid.
        tracks: `(T,J,2)` tracked pixels `(x,y)` of the keyframe queries.
        tracker_vis: `(T,J)` the tracker's own visibility.
        query_uv: `(J,2)` keyframe query pixels.
        keyframe_index: Frame carrying the queries.
        keyframe_pointmap: `(H,W,3)` the keyframe's pointmap, read instead of `pointmaps[keyframe_index]` when given
            (with `object_pixels`, `pointmaps` may be a NaN placeholder).
        object_pixels: Passed through to the observations (`TrajectoryObservations.object_pixels`).

    Returns:
        `hand_masks` all False; `keyframe_points` the keyframe pointmap under the rounded queries; `track_vis` the
        tracker's visibility, inside the image and that frame's mask, with a finite keyframe point; and unanchored
        observations (all-False `track_vis`, NaN anchors) until `RealObservations.anchored`.

    Raises:
        ValueError: If `keyframe_index` is out of range or the T/J shapes of the inputs disagree.
    """
    t, j = int(masks.shape[0]), int(query_uv.shape[0])
    if not 0 <= keyframe_index < t:
        raise ValueError(f"keyframe_index {keyframe_index} out of range [0,{t})")
    shapes = {"masks": tuple(masks.shape), "pointmaps": tuple(pointmaps.shape), "tracks": tuple(tracks.shape),
              "tracker_vis": tuple(tracker_vis.shape), "query_uv": tuple(query_uv.shape)}
    h, w = shapes["masks"][1:3] if masks.dim() == 3 else (-1, -1)
    expected = {"masks": (t, h, w), "pointmaps": (t, h, w, 3), "tracks": (t, j, 2), "tracker_vis": (t, j),
                "query_uv": (j, 2)}
    if keyframe_pointmap is not None:
        shapes["keyframe_pointmap"], expected["keyframe_pointmap"] = tuple(keyframe_pointmap.shape), (h, w, 3)
    if object_pixels is not None:
        shapes["object_pixels"], expected["object_pixels"] = (object_pixels.n_frames,), (t,)
    if shapes != expected:
        raise ValueError(f"front-end outputs disagree on T={t} frames, J={j} queries: shapes {shapes}")
    hand_masks = torch.zeros_like(masks)
    keyframe = pointmaps[keyframe_index] if keyframe_pointmap is None else keyframe_pointmap
    keyframe_points = _sample_pointmap(keyframe, query_uv)
    track_vis = (tracker_vis.to(torch.bool) & tracks_in_masks(masks, tracks)
                 & torch.isfinite(keyframe_points).all(-1)[None])
    unanchored = TrajectoryObservations(intrinsics=K, masks=masks, hand_masks=hand_masks, pointmaps=pointmaps,
                                        tracks=tracks, track_vis=torch.zeros_like(track_vis),
                                        anchors=torch.full_like(keyframe_points, float("nan")),
                                        object_pixels=object_pixels)
    return RealObservations(observations=unanchored, track_vis=track_vis, keyframe_points=keyframe_points)


def _sample_pointmap(pointmap: Tensor, query_uv: Tensor) -> Tensor:
    """`(J,3)` pointmap values at the rounded query pixels (NaN where the pointmap is invalid)."""
    h, w = pointmap.shape[:2]
    u = query_uv[:, 0].round().long().clamp(0, w - 1)
    v = query_uv[:, 1].round().long().clamp(0, h - 1)
    return pointmap[v, u]
