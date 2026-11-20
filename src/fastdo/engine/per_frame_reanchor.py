"""Per-frame re-anchor "G": move a solved frame's object onto its own pointmap observation.

Motivation (clip 135841): once every CoTracker track has died (keyframe queries only), the pointmap term has lost
capture and the smooth prior carries the object to 1.8x its depth and off its own mask, while the clip-level certificate
averages the alive and dead frames' opposite departures away. DO AS I DO never propagates a frame's translation: every
frame is placed from that frame's pointmap. G is that idea restricted to the frames whose solve disagrees with its own
observation. It runs after the production solve and the certificate gate (`depth_anchor_gate`), before `finish_solve`
writes solve.npz, so the contact stage reads the re-anchored poses. Label-free: it reads the clip's masks, masked
pointmap, raw tracker visibility and intrinsics only, so it runs on internet video as on lab clips.

PER FRAME t (float64 on the CPU, on copies):
  z_obs(t)  `batch.frame_depths`: median z of the frame's masked finite in-front pointmap points (the certificate's
            and the depth anchor's z_obs), NaN below `DEPTH_MIN_POINTS`
  p(t)      s R c + t, c the mean of the clip's `MeshSamples.points` (the template the solve and contact use)
  d(t)      log max(p_z, 1e-6) - log z_obs                       (the certificate's d_t)
  q(t)      |proj(p) - mask centroid| / r_mask, r_mask = sqrt(mask area / pi), the centroid the pixel-index mean
  nvis(t)   tracks.npz `vis` summed over the queries (the tracker's own visibility, every query)
A frame whose d, q or p_z is not finite or p_z <= 0 is never moved.

CLIP FLAG = the production solve's certificate |median_t d_t| >= `threshold` (the depth-anchor gate's; that gate's own
recorded certificate when the gate ran, else `depth_certificate` of the solution, which is then the production solve).
  flagged     every frame with |d| > band or q > perp_k is placed at z_obs on the camera ray through
              its mask centroid: p' = z_obs K^-1 (u_m, v_m, 1)
  unflagged   only DEAD frames (nvis < vis_min): |d| > band slides p along its own ray to the band
              edge, z' = z_obs exp(sign(d) band); then q > perp_k moves it onto the mask-centroid ray at that depth
The translation moves by p' - p; rotation and scale are untouched. An unflagged clip without a dead frame out of the
band keeps the solution object itself (its solve.npz is then byte-identical to the stage without G).

SAFETY OPTIONS (each None by default = the G above, byte for byte; `PerFrameReanchorConfig`):
  max_log_shift       after p' is computed, if |log p'_z / p_z| > max_log_shift, p' slides along its own camera ray
                      to depth p_z exp(+-max_log_shift) (the move keeps its direction, its depth change is capped)
  occluded_area_frac  a frame whose mask area < occluded_area_frac x the clip's median mask area (over frames with a
                      non-empty mask) is LOW-VISIBILITY; such a frame is moved only if p'_z <= p_z (toward the camera or
                      sideways), never away from it. Diagnosis: where G harmed by +54 mm, dead frames whose mask
                      had collapsed to 3-17% of the clip median read the background (pointmap 59-74% too far) while the
                      solve was within 10% of GT, and G pushed the object 0.27-0.50 m away. A mask that has lost most
                      of its object can over-read depth (background behind it) far more than under-read it.
OPTION (None by default = the branches above, byte for byte):
  dead_frame_branch   False disables the unflagged (dead-frame) branch: only a FLAGGED clip is re-anchored. The
                      dead-frame branch moved frames on 29/53 fresh Ego4D windows while egocentric HOI4D almost never
                      reaches it.

IMPLEMENTATION `impl` (`solve_stage.per_frame_reanchor_impl`; "reference" by default = the code above, byte for
byte): "vectorized" computes the SAME `FrameObservations` without the per-frame `np.nonzero` loop and the global
finite-point copy (measured: 0.72-0.80 s per window on the single gpu-class thread, dominated by those passes over
the native-resolution masks). Exactness, not tolerance: the mask centroid is (sum of pixel indices) / count, where the
index sums are integers accumulated exactly (row and column projections in unsigned integers, then int64) and every
partial sum stays below 2**53, so the float64 division reproduces `np.nonzero(...)[i].mean()` bit for bit (numpy's
mean of an integer array is its exact float64 sum divided by the count); z_obs selects the same two middle order
statistics with `np.partition` instead of a full sort and averages them in float64 as `frame_depths` does. The masks
must be canonical booleans (bytes 0/1), which every production path builds (`astype(bool)` / a comparison); a
non-boolean mask tensor is canonicalized with `!= 0` first. Pinned by tests/engine/test_per_frame_reanchor_vectorized.py.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from fastdo.engine.depth_anchor_gate import GATE_KEY, depth_certificate
from fastdo.engine.solve_stage_config import REANCHOR_IMPLS, SAFETY_OPTIONS, PerFrameReanchorConfig
from fastdo.solver.api import MeshSamples, TrajectoryObservations, TrajectorySolution
from fastdo.solver.batch import DEPTH_MIN_POINTS, frame_depths, valid_pixels, valid_pixels_of

__all__ = ["MIN_POSITIVE_DEPTH", "REANCHOR_IMPLS", "REANCHOR_KEY", "TRACKER_VIS_KEY", "FrameObservations",
           "apply_per_frame_reanchor", "frame_observations", "frame_observations_vectorized", "low_visibility",
           "mask_moments", "reanchor_poses"]

REANCHOR_KEY = "per_frame_reanchor"          # the solve.json block
TRACKER_VIS_KEY = "tracker_visible_count"    # the SolveCall context key (T,) int64, set by prepare_solve
MIN_POSITIVE_DEPTH = 1e-6                    # d's log floor on the solved depth
_SAME_DEPTH_RTOL = 1e-9                      # a sideways move (mask ray at the same depth) is not "away" (float noise)


@dataclasses.dataclass(frozen=True)
class FrameObservations:
    """One clip's per-frame observations for G (module docstring): all (T,) or (T,2) float64 numpy, `K` (3,3)."""

    z_obs: np.ndarray
    mask_centroid: np.ndarray
    r_mask: np.ndarray
    nvis: np.ndarray
    K: np.ndarray


def frame_observations(obs: TrajectoryObservations, tracker_visible: np.ndarray, K: Any) -> FrameObservations:
    """`FrameObservations` of `obs` (masks and masked pointmap), the raw per-frame visible-track counts and `K`.

    Raises:
        ValueError: If `tracker_visible` does not have one entry per frame.
    """
    points, counts = valid_pixels_of(obs)
    z_obs = frame_depths(points, torch.cat([counts.new_zeros(1), counts.cumsum(0)]), DEPTH_MIN_POINTS).numpy()
    masks = obs.masks.detach().cpu().numpy().astype(bool)
    n = masks.shape[0]
    nvis = np.asarray(tracker_visible, dtype=np.int64).reshape(-1)
    if nvis.shape[0] != n:
        raise ValueError(f"per-frame re-anchor: {nvis.shape[0]} visible-track counts for {n} frames")
    centroid = np.full((n, 2), np.nan)
    for t in range(n):
        if masks[t].any():
            vv, uu = np.nonzero(masks[t])
            centroid[t] = (uu.mean(), vv.mean())
    area = masks.reshape(n, -1).sum(1).astype(float)
    K64 = np.asarray(K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else K, dtype=np.float64)
    return FrameObservations(z_obs=np.asarray(z_obs, np.float64), mask_centroid=centroid,
                             r_mask=np.sqrt(area / np.pi), nvis=nvis, K=K64)


def mask_moments(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame of canonical (T,H,W) bool `masks`, the pixel count and the sums of the pixels' column (u) and
    row (v) indices, all (T,) int64 and exact: row counts come from one contiguous pass (unsigned accumulators that
    cannot overflow: a row holds at most W pixels), column counts only over each frame's occupied row span."""
    n, h, w = masks.shape
    u8 = masks.view(np.uint8)
    rows = u8.sum(axis=2, dtype=np.uint16 if w < 2**16 else np.uint32).astype(np.int64)   # (T,H) pixels per row
    count = rows.sum(axis=1)
    sum_v = rows @ np.arange(h, dtype=np.int64)
    sum_u = np.zeros(n, dtype=np.int64)
    columns = np.arange(w, dtype=np.int64)
    for t in np.flatnonzero(count):
        occupied = np.flatnonzero(rows[t])
        cols = u8[t, occupied[0]:occupied[-1] + 1].sum(axis=0, dtype=np.uint16 if h < 2**16 else np.uint32)
        sum_u[t] = cols.astype(np.int64) @ columns
    return count, sum_u, sum_v


def _frame_depths_exact(obs: TrajectoryObservations, min_points: int) -> np.ndarray:
    """`frame_depths(*valid_pixels_of(obs))` as (T,) float64, from the object pixels in place: per frame the
    median z of its rows that are all finite with z > 0 (the same set `valid_pixels_of` then `frame_depths` keep), the
    two middle order statistics by `np.partition`, NaN below `min_points`."""
    px = obs.object_pixels
    if px is None:
        points, counts = valid_pixels(obs.masks, obs.pointmaps)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts.numpy(), dtype=np.int64)])
    else:
        points, offsets = px.points, px.offsets.detach().cpu().numpy()
    pts = points.detach().cpu().numpy()
    z_all = pts[:, 2]
    # all three coordinates finite and z > 0, column by column (one row-wise `.all` pass is ~5x slower)
    ok = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(z_all) & (z_all > 0)
    out = np.full(offsets.shape[0] - 1, np.nan, dtype=np.float64)
    for k in range(out.shape[0]):
        start, stop = int(offsets[k]), int(offsets[k + 1])
        z = z_all[start:stop][ok[start:stop]]
        n = z.shape[0]
        if n < min_points:
            continue
        lo, hi = (n - 1) // 2, n // 2
        # the order statistics of the stored values; widening to float64 is exact and monotonic, so they are the
        # reference's (which widens first, then sorts), and the mean of the two is taken in float64 as it does
        part = np.partition(z, (lo, hi))
        out[k] = (np.float64(part[lo]) + np.float64(part[hi])) / 2
    return out


def frame_observations_vectorized(obs: TrajectoryObservations, tracker_visible: np.ndarray, K: Any
                                  ) -> FrameObservations:
    """`frame_observations(obs, tracker_visible, K)`, bit for bit, without the per-frame loops over the dense
    masks (module docstring, IMPLEMENTATION).

    Raises:
        ValueError: If `tracker_visible` does not have one entry per frame.
    """
    z_obs = _frame_depths_exact(obs, DEPTH_MIN_POINTS)
    masks_t = obs.masks.detach().cpu()
    masks = masks_t.numpy() if masks_t.dtype == torch.bool else (masks_t != 0).numpy()
    n = masks.shape[0]
    nvis = np.asarray(tracker_visible, dtype=np.int64).reshape(-1)
    if nvis.shape[0] != n:
        raise ValueError(f"per-frame re-anchor: {nvis.shape[0]} visible-track counts for {n} frames")
    count, sum_u, sum_v = mask_moments(np.ascontiguousarray(masks))
    centroid = np.full((n, 2), np.nan)
    seen = count > 0
    centroid[seen, 0] = sum_u[seen] / count[seen]
    centroid[seen, 1] = sum_v[seen] / count[seen]
    area = count.astype(float)
    K64 = np.asarray(K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else K, dtype=np.float64)
    return FrameObservations(z_obs=z_obs, mask_centroid=centroid, r_mask=np.sqrt(area / np.pi), nvis=nvis, K=K64)


def _frame_stats(poses: np.ndarray, scale: float, c: np.ndarray, ob: FrameObservations
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = scale * poses[:, :3, :3] @ c + poses[:, :3, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.log(np.maximum(p[:, 2], MIN_POSITIVE_DEPTH)) - np.log(ob.z_obs)
        uv = p @ ob.K.T
        uv = uv[:, :2] / uv[:, 2:3]
        q = np.linalg.norm(uv - ob.mask_centroid, axis=1) / ob.r_mask
    return p, d, q


def low_visibility(ob: FrameObservations, area_frac: float | None) -> np.ndarray:
    """(T,) bool: frames whose mask area is below `area_frac` x the median area of the clip's non-empty masks (all
    False when `area_frac` is None or no mask is non-empty)."""
    area = np.pi * ob.r_mask ** 2
    seen = area[area > 0]
    if area_frac is None or seen.size == 0:
        return np.zeros(area.shape[0], dtype=bool)
    return area < area_frac * float(np.median(seen))


def reanchor_poses(poses: np.ndarray, scale: float, c: np.ndarray, ob: FrameObservations, flagged: bool,
                   cfg: PerFrameReanchorConfig) -> tuple[np.ndarray, list[int], np.ndarray]:
    """G on float64 `(T,4,4)` poses: `(new poses, moved frames, (n_moved,3) translation shifts in metres)`."""
    p, d, q = _frame_stats(poses, scale, c, ob)
    Kinv = np.linalg.inv(ob.K)
    out = poses.copy()
    moved: list[int] = []
    shifts: list[np.ndarray] = []
    low_vis = low_visibility(ob, cfg.occluded_area_frac)
    for t in range(poses.shape[0]):
        if not (np.isfinite(d[t]) and np.isfinite(q[t]) and p[t, 2] > 0):
            continue
        if not flagged and (cfg.dead_frame_branch is False or ob.nvis[t] >= cfg.vis_min):
            continue
        a_out, q_out = abs(d[t]) > cfg.band, q[t] > cfg.perp_k
        if not (a_out or q_out):
            continue
        ray = Kinv @ np.array([*ob.mask_centroid[t], 1.0])
        if flagged:
            new = ob.z_obs[t] * ray / ray[2]
        else:
            new = p[t].copy()
            if a_out:
                new = new * (ob.z_obs[t] * math.exp(math.copysign(cfg.band, d[t])) / new[2])
            if q_out:
                new = new[2] * ray / ray[2]
        if cfg.max_log_shift is not None:
            step = math.log(new[2] / p[t, 2])
            if abs(step) > cfg.max_log_shift:
                new = new * (p[t, 2] * math.exp(math.copysign(cfg.max_log_shift, step)) / new[2])
        if low_vis[t] and new[2] > p[t, 2] * (1.0 + _SAME_DEPTH_RTOL):
            continue
        out[t, :3, 3] = poses[t, :3, 3] + (new - p[t])
        moved.append(t)
        shifts.append(new - p[t])
    return out, moved, np.asarray(shifts, dtype=np.float64).reshape(-1, 3)


def _clip_certificate(mesh: MeshSamples, obs: TrajectoryObservations, solution: TrajectorySolution,
                      context: Mapping[str, Any]) -> tuple[float | None, str]:
    """The production solve's certificate value: the depth-anchor gate's record (its FIRST solve) when present, else the
    certificate of `solution` (then the production solve)."""
    gate = context.get(GATE_KEY)
    if gate is not None:
        return gate["certificate"]["value"], "depth_anchor_gate"
    return depth_certificate(mesh.points, obs, solution).value, "solution"


def apply_per_frame_reanchor(mesh: MeshSamples, obs: TrajectoryObservations, solution: TrajectorySolution,
                             context: Mapping[str, Any], cfg: PerFrameReanchorConfig, impl: str = "reference"
                             ) -> tuple[TrajectorySolution, MappingProxyType[str, Any]]:
    """G on one clip's kept solution: `(solution to write, solve.json's per_frame_reanchor block)`.

    Returns `solution` itself when no frame moves; otherwise a copy whose `poses` (same dtype and device) carry the
    moved translations, every other field unchanged. `impl` (one of `REANCHOR_IMPLS`) picks how the per-frame
    observations are computed; both give the same bytes, and the block does not record it.

    Raises:
        KeyError: If `context` lacks the raw visible-track counts (`TRACKER_VIS_KEY`, set by `prepare_solve`).
        ValueError: On an anisotropic solution: G is defined and tested for the isotropic template only; on an
            unknown `impl`.
    """
    t0 = time.perf_counter()
    if impl not in REANCHOR_IMPLS:
        raise ValueError(f"per-frame re-anchor impl must be one of {REANCHOR_IMPLS}, got {impl!r}")
    if getattr(solution, "shape_log_stretch", None) is not None:
        raise ValueError("per-frame re-anchor is defined for an isotropic solve; this solution is anisotropic")
    observe = frame_observations_vectorized if impl == "vectorized" else frame_observations
    ob = observe(obs, np.asarray(context[TRACKER_VIS_KEY]), context["K"])
    value, source = _clip_certificate(mesh, obs, solution, context)
    flagged = value is not None and value >= cfg.threshold
    poses = solution.poses.detach().cpu().numpy().astype(np.float64)
    c = mesh.points.detach().cpu().numpy().astype(np.float64).mean(0)
    new, moved, shifts = reanchor_poses(poses, float(solution.scale), c, ob, flagged, cfg)
    kept = solution
    if moved:
        dtype, device = solution.poses.dtype, solution.poses.device
        kept = dataclasses.replace(solution, poses=torch.from_numpy(new).to(dtype=dtype, device=device))
    dead = [int(t) for t in np.flatnonzero(ob.nvis < cfg.vis_min)]
    base = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in SAFETY_OPTIONS}
    safety = cfg.safety_options()
    if "occluded_area_frac" in safety:
        safety["low_visibility_frames"] = [int(t) for t in np.flatnonzero(low_visibility(ob, cfg.occluded_area_frac))]
    record = {**base, **safety, "min_points": DEPTH_MIN_POINTS, "certificate": value,
              "certificate_source": source, "flagged": bool(flagged),
              "mode": "flagged: out-of-band/off-mask frames to z_obs on the mask ray" if flagged
              else "unflagged: dead frames only, band edge then mask ray",
              "n_frames": int(poses.shape[0]), "dead_frames": dead, "frames": [int(t) for t in moved],
              "shifts_m": shifts.round(9).tolist(),
              "max_shift_m": float(np.linalg.norm(shifts, axis=1).max()) if moved else 0.0,
              "seconds": time.perf_counter() - t0}
    return kept, MappingProxyType(record)
