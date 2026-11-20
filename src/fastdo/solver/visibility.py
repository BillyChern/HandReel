"""Visibility-aware template measure: which canonical mesh samples are seen
from the camera at the current pose/scale.

Restricting the mesh-side OT marginal to this set is the fix for the partial-observation bias
documented in `api.py`'s module docstring and diagnosed on DexYCB: at the
coarse/mid annealing levels the mesh marginal is nearly hard, so an unrestricted uniform measure over
the *whole* template (including the self-occluded far side) forces observed points to feed mass to
mesh samples that could not possibly have produced them, biasing the fit toward the camera. Restricting
the template measure to the visible set makes a near-hard marginal correct again.

Three additions on top of the original mechanism, all inside this
module, with no `eps`/`rho` retuning:

1. `exclude_hand_occluded`/`hand_occluded`/`dilate_hand_masks`: a visible-by-geometry template sample
   whose projected pixel falls under the (dilated) hand mask is hidden by the hand and must also leave
   the mesh-side marginal, at every level - this is what actually removes the coverage pull toward the
   hand-covered part of the visible surface on the cracker-box sequence. The *hand* mask (not the
   object mask) is used because it is a real occluder regardless of the current pose estimate; the
   object mask is not (a wrong init would project the whole template outside the object mask and empty
   the visible set).
2. `visible_samples` now takes a facing-test slack `delta`: the front-facing test becomes
   `n_j . v_j > -delta` (`v_j` the unit sample-to-camera direction) instead of the strict `> 0`. `delta`
   is a **cosine slack, unitless** - not an angle in degrees or radians, even though "delta=0.25 admits
   samples up to about 14 degrees past the strict boundary" is a useful intuition for a specific value;
   at `delta = 0` this is exactly the original strict test (a superset of the true visible set only when
   `delta > 0`). Each level anneals its own slack independently via `SolverConfig.vis_delta_coarse/mid/
   fine` - **`SolverConfig` is the single source of truth for the shipped per-level values**, not this
   docstring or any report prose, which can and did drift across several tuning rounds.
   Likewise the z-buffer margin is annealed (`vis_margin_coarse/mid/fine`, fraction of diameter) and the z-buffer
   itself defaults to full
   resolution (`downsample = 1`): a coarser buffer with a tight margin can reject genuinely-visible
   samples on faces seen at a grazing angle, whose depth spans tens of millimetres within one coarse
   pixel cell.
3. `foreshortening_weights`: the observed measure (masked pointmap pixels) is uniform over *image
   pixels*, i.e. it is the pushforward under projection of the surface measure weighted by
   `cos(theta)/z^2` (theta the angle between the surface normal and the view ray - the standard solid-
   angle/foreshortening density; a flat patch at grazing incidence subtends far fewer pixels per unit
   area than one seen face-on, and depth also shrinks the subtended angle as `1/z^2`). A template
   measure that is uniform *per sample* instead over-weights grazing/far samples relative to what the
   camera actually observed. `api.py` uses this weighting (instead of a uniform 1/|V_k|) at the mid and
   fine levels only; the coarse level keeps uniform mass (capture range first, foreshortening bias
   second).
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from fastdo.solver.terms import project, transform_points

logger = logging.getLogger(__name__)

# Fewer than this many samples visible (object out of frame, or a degenerate pose) is treated as "no
# usable visibility signal this frame": fall back to the full template rather than starve the OT term.
_MIN_VISIBLE = 32


def _rotate_normals(poses: Tensor, normals: Tensor) -> Tensor:
    """Camera-frame normals (T,M,3) for canonical `normals` (M,3), or per-frame (T,M,3)."""
    R = poses[:, :3, :3]
    if normals.dim() == 3:
        return torch.einsum("tij,tmj->tmi", R, normals)
    return torch.einsum("tij,mj->tmi", R, normals)


def _sparse_zbuffer_at(lin: Tensor, z: Tensor) -> Tensor:
    """`zbuf[lin]` for `zbuf = scatter_reduce(full(inf), lin, z, "amin")`, without the dense buffer.

    Sorting the pixel indices groups equal pixels into consecutive segments; a segment id per sorted entry (cumsum of
    "differs from its predecessor") replaces the pixel index, so the buffer has at most `lin.numel()` cells. Every
    size is fixed by `lin.numel()`, so nothing syncs with the host. `amin` is order-independent and exact, so the
    result is bit-identical to the dense buffer's."""
    order = torch.argsort(lin)
    s = lin[order]
    starts = torch.ones_like(s, dtype=torch.bool)
    starts[1:] = s[1:] != s[:-1]
    seg = starts.cumsum(0) - 1
    buf = torch.full_like(z, float("inf"))
    buf.scatter_reduce_(0, seg, z[order], reduce="amin")
    out = torch.empty_like(z)
    out[order] = buf[seg]
    return out


def visible_samples(points: Tensor, normals: Tensor, poses: Tensor, log_scale: Tensor, K: Tensor,
                    image_hw: tuple[int, int], downsample: int = 1, margin: float = 0.02,
                    delta: float = 0.0, *, frame_hw: Tensor | None = None,
                    sample_valid: Tensor | None = None, diameter: Tensor | None = None,
                    sparse: bool = False) -> Tensor:
    """Boolean visibility mask (T,M) for canonical mesh samples `points` (M,3) with unit `normals` (M,3).

    A sample is visible at frame `k` when it is:
      (a) front-facing with slack `delta`: `n_j . v_j > -delta`, `v_j` the unit sample-to-camera
          direction (`normalize(-p_j)`; the camera sits at the origin). `delta = 0` is the strict test
          (the original mechanism); `delta > 0` widens the set to a superset of the truly
          visible samples, trading some of the fix's bias reduction for capture range at a pose/scale
          that is not yet correct (see `SolverConfig.vis_delta_coarse/mid/fine`).
      (b) inside the image once projected with `K`;
      (c) not self-occluded: a `1/downsample`-resolution z-buffer built from every front-facing,
          in-image candidate keeps `z_j <= z_buf(u_j, v_j) + margin * s * diameter`, where `diameter` is
          the canonical (unscaled) extent of `points` and `s = log_scale.exp()` is the current scale, so
          the slack is a fraction of the object's *physical* size. `downsample = 1` (full resolution) is
          the default: a coarser buffer can rejects genuinely-visible grazing-angle
          samples whose depth varies a lot within one coarse cell.

    Fully batched over T (frames) and M (samples): no Python loop. Uses `scatter_reduce_(...,
    reduce="amin")` for the z-buffer, matching the pattern of `synthetic._splat_depth` but batched over
    frames via a per-frame pixel offset instead of looped.

    A frame with fewer than `_MIN_VISIBLE` samples surviving all three tests (the object is out of frame,
    e.g. behind the camera) falls back to the full template (all True) for that frame, logged at DEBUG,
    so the OT term is never handed an empty mesh-side marginal.

    `points`/`normals` may be per-frame (T,M,3), `log_scale` and `diameter`
    per-frame (T,), and `K` per-frame (T,3,3). `image_hw` then sizes the z-buffer (the batch maximum)
    while `frame_hw` (T,2) holds each frame's own (H,W) for the in-image test. `sample_valid` (M,) or
    (T,M) marks real samples of a padded template: padding is never a candidate, and a degenerate
    frame falls back to its valid samples rather than to every column. `diameter` defaults to the
    extent of `points` (M,3).

    `sparse=True` (`SolverConfig.sparse_zbuffer`): the z-buffer is only ever read at the T*M pixels the samples
    land on, so it is built over those pixels alone (`_sparse_zbuffer_at`) instead of all T*Hd*Wd of them. `amin`
    over the same values per pixel is exact, so the mask is identical; memory falls from 8 bytes per pixel per frame
    (2.2 GB per 100-frame clip at 1440x1920) to O(T*M), with no host synchronisation.
    """
    T = poses.shape[0]
    M = points.shape[-2]
    device, dtype = poses.device, poses.dtype
    points = points.to(device=device, dtype=dtype)
    normals = normals.to(device=device, dtype=dtype)
    # Kept as 0-dim tensors (not `float(...)`) until they are combined with `margin` below: this is
    # called once per accepted LM step (an earlier tuning of that hot path aimed to avoid host/device
    # synchronization), and a `float(...)`/`.item()` call blocks on the whole CUDA queue draining.
    if diameter is None:
        diameter = (points.max(0).values - points.min(0).values).norm()
    s = log_scale.exp()

    p = transform_points(points, poses, s)                        # (T,M,3) camera-frame positions
    n = _rotate_normals(poses, normals)                           # (T,M,3): rotation only (unit-preserving)
    view_dir = torch.nn.functional.normalize(-p, dim=-1)          # unit sample-to-camera direction
    cos = (n * view_dir).sum(-1)                                  # (T,M); equals the old sign-only test at delta=0
    front = cos > -delta

    H, W = image_hw
    Hd, Wd = max(1, H // downsample), max(1, W // downsample)
    Kd = K.clone().to(device=device, dtype=dtype)
    Kd[..., 0, 0] = Kd[..., 0, 0] / downsample
    Kd[..., 1, 1] = Kd[..., 1, 1] / downsample
    Kd[..., 0, 2] = Kd[..., 0, 2] / downsample
    Kd[..., 1, 2] = Kd[..., 1, 2] / downsample
    uv = project(p, Kd)                                            # (T,M,2) low-res pixel coords
    u = uv[..., 0].round().long()
    v = uv[..., 1].round().long()
    if frame_hw is None:
        in_image = (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd) & (p[..., 2] > 0)
    else:
        hw_d = (frame_hw.to(device) // downsample).clamp_min(1)
        in_image = (u >= 0) & (u < hw_d[:, 1:2]) & (v >= 0) & (v < hw_d[:, 0:1]) & (p[..., 2] > 0)
    candidate = front & in_image
    if sample_valid is not None:
        candidate = candidate & sample_valid.to(device)

    z = p[..., 2]
    u_c, v_c = u.clamp(0, Wd - 1), v.clamp(0, Hd - 1)
    t_idx = torch.arange(T, device=device).view(T, 1).expand(T, M)
    lin = (t_idx * (Hd * Wd) + v_c * Wd + u_c).reshape(-1)
    z_or_inf = torch.where(candidate, z, torch.full_like(z, float("inf"))).reshape(-1)
    if sparse:
        zbuf_at = _sparse_zbuffer_at(lin, z_or_inf).view(T, M)
    else:
        zbuf = torch.full((T * Hd * Wd,), float("inf"), dtype=dtype, device=device)
        zbuf.scatter_reduce_(0, lin, z_or_inf, reduce="amin")
        zbuf_at = zbuf[lin].view(T, M)
    zbuf_slack = margin * s * diameter                               # 0-dim (or (T,)) tensor, no host sync
    if zbuf_slack.dim() == 1:
        zbuf_slack = zbuf_slack[:, None]
    visible = candidate & (z <= zbuf_at + zbuf_slack)

    n_visible = visible.sum(dim=1)
    degenerate = n_visible < _MIN_VISIBLE
    # `.any()`/`.sum()` on a GPU tensor still just queues a kernel; only the `bool(...)`/`int(...)`
    # host-side read below would sync, so gate it behind an `isEnabledFor` check (no sync) that is
    # normally False and skips the sync entirely on the hot (per-accepted-step) path.
    if logger.isEnabledFor(logging.DEBUG) and bool(degenerate.any()):
        logger.debug("visible_samples: %d/%d frame(s) degenerate (< %d visible samples); using the "
                    "full template for them", int(degenerate.sum()), T, _MIN_VISIBLE)
    fallback = torch.ones_like(visible) if sample_valid is None else sample_valid.to(device).expand(T, M)
    return torch.where(degenerate.view(T, 1), fallback, visible)


def dilate_hand_masks(hand_masks: Tensor, dilate_px: int) -> Tensor:
    """Binary-dilate each frame's hand mask by `dilate_px` pixels (max-pool over a
    `2*dilate_px + 1` square structuring element). Depends only on the clip's own hand masks, not on
    any pose/scale estimate, so callers compute this once per clip (see `api.py`), not per LM step.
    """
    if dilate_px <= 0:
        return hand_masks
    k = 2 * dilate_px + 1
    x = hand_masks.to(torch.float32).unsqueeze(1)                   # (T,1,H,W)
    dilated = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=dilate_px)
    return dilated.squeeze(1) > 0.5


def hand_occluded(points: Tensor, poses: Tensor, log_scale: Tensor, K: Tensor,
                  hand_mask_dilated: Tensor) -> Tensor:
    """(T,M) bool: True where `points`' full-resolution projected pixel falls inside the (already-
    dilated) hand mask, i.e. the sample is hidden by the hand this frame. The hand mask is an occluder
    that is meaningful at any pose (unlike the object mask, which is only meaningful once the pose
    estimate is already close), so this is safe to apply at every annealing level, including coarse.
    Per-frame `points` (T,M,3), `log_scale` (T,) and `K` (T,3,3) work unchanged; a hand mask
    padded to a larger batch image is False outside each frame's own image, so no per-frame bound is
    needed here.
    """
    T = poses.shape[0]
    M = points.shape[-2]
    s = log_scale.exp()
    p = transform_points(points, poses, s)
    uv = project(p, K)
    H, W = hand_mask_dilated.shape[-2], hand_mask_dilated.shape[-1]
    u = uv[..., 0].round().long()
    v = uv[..., 1].round().long()
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (p[..., 2] > 0)
    u_c, v_c = u.clamp(0, W - 1), v.clamp(0, H - 1)
    t_idx = torch.arange(T, device=points.device).view(T, 1).expand(T, M)
    hit = hand_mask_dilated[t_idx, v_c, u_c]
    return hit & in_image


def exclude_hand_occluded(mask: Tensor, points: Tensor, poses: Tensor, log_scale: Tensor, K: Tensor,
                          hand_mask_dilated: Tensor, min_visible: int = _MIN_VISIBLE) -> Tensor:
    """`mask` (T,M) with hand-occluded samples additionally excluded, per frame falling back to the
    unmodified `mask` (not the full template) if fewer than `min_visible` samples would remain - heavy
    hand occlusion is exactly the case this exists for (the cracker-box sequence), so degenerating all
    the way to "everything visible" would defeat the point; reverting to the geometry-only visible set
    is the more conservative fallback.
    """
    occluded = hand_occluded(points, poses, log_scale, K, hand_mask_dilated)
    combined = mask & ~occluded
    n_visible = combined.sum(dim=1)
    degenerate = n_visible < min_visible
    if logger.isEnabledFor(logging.DEBUG) and bool(degenerate.any()):
        logger.debug("exclude_hand_occluded: %d/%d frame(s) would drop below %d visible samples after "
                    "hand exclusion; keeping the geometry-only visible set for them",
                    int(degenerate.sum()), mask.shape[0], min_visible)
    return torch.where(degenerate.view(-1, 1), mask, combined)


def foreshortening_weights(points: Tensor, normals: Tensor, poses: Tensor, log_scale: Tensor) -> Tensor:
    """(T,M) >= 0 unnormalized template density `max(n_j . v_j, 0) / z_j^2` (`v_j` the unit sample-to-
    camera direction, `z_j` the camera-frame depth): the surface-measure density whose pushforward
    under projection is uniform over image pixels, matching the observed (masked-pointmap-pixel)
    measure. A per-sample-uniform template measure over-weights samples seen at a grazing angle or far
    from the camera relative to what the camera actually observed; this is the correction (mid/fine
    levels only - see `api.py`). Not normalized to sum to 1: callers (`pointmap_term`'s `mesh_mask`
    handling) normalize per frame.
    """
    s = log_scale.exp()
    p = transform_points(points, poses, s)
    n = _rotate_normals(poses, normals)
    view_dir = torch.nn.functional.normalize(-p, dim=-1)
    cos = (n * view_dir).sum(-1).clamp_min(0.0)
    z2 = (p[..., 2] ** 2).clamp_min(1e-12)
    return cos / z2
