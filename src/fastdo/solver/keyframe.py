"""Keyframe pose from a known shape and a real pointmap, and mesh-frame anchors for 2D tracks.

A real front end (SAM 2 masks, MoGe pointmaps, CoTracker tracks) gives pixel tracks but no 3D
anchor in the object's canonical frame, which `pnp_initialize` and `track_term` need. Given the
object's shape (canonical surface samples + normals) and one keyframe pose, every keyframe query
pixel is anchored on the visible mesh surface (`anchors_from_pose`). When no pose for that keyframe
is available from a shape model, `register_keyframe_pose` estimates it from the keyframe's masked
pointmap alone: a multi-hypothesis trimmed ICP over a fixed icosahedral rotation grid, scored by a
symmetric visible-surface Chamfer plus a silhouette term against the keyframe mask. With
`estimate_scale=True` the fit is a similarity, so a monocular pointmap whose metric scale disagrees
with the shape's (a few to tens of percent for MoGe-2) still yields the right rotation and anchors,
with the pose and scale expressed in the pointmap's own metric. `gate_init_poses` screens per-frame
PnP initializations against that pointmap. Nothing here looks at a ground-truth pose.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import Tensor

from fastdo.solver.api import TrajectoryObservations
from fastdo.solver.terms import project, transform_points
from fastdo.solver.visibility import visible_samples

logger = logging.getLogger(__name__)

__all__ = ["KeyframeRegistration", "anchors_from_pose", "gate_init_poses", "register_keyframe_hypotheses",
           "register_keyframe_pose"]

_TRIM_FACTOR = 2.5          # correspondences farther than 2.5x the median residual are dropped per ICP step
_TRUNCATE_FRAC = 0.1        # score: residuals truncated at 10% of the (scaled) diameter
_SILHOUETTE_WEIGHT = 0.1    # score: outside-mask fraction, in units of the (scaled) diameter
_MASK_DILATE_PX = 2


@dataclass(frozen=True)
class KeyframeRegistration:
    """`register_keyframe_pose`'s result: the posing `scale * R y + t` of canonical samples `y`."""

    pose: Tensor    # (4,4) float32 object-to-camera transform (rotation + translation, scale not baked in)
    scale: float    # metric scale of the canonical samples in the observation's frame
    score: float    # truncated symmetric Chamfer + silhouette penalty, metres (lower is better)


def _rotation_hypotheses(dtype: torch.dtype) -> Tensor:
    """120 rotations: the icosahedral group and a copy offset by 36 degrees about (1,1,1)."""
    group = Rotation.create_group("I")
    offset = Rotation.from_rotvec([math.radians(36.0) / math.sqrt(3.0)] * 3)
    mats = torch.cat([torch.as_tensor(group.as_matrix()), torch.as_tensor((group * offset).as_matrix())])
    return mats.to(dtype)


def _diameter(points: Tensor) -> float:
    return float((points.max(0).values - points.min(0).values).norm())


def _umeyama(src: Tensor, dst: Tensor, w: Tensor, scale: Tensor, bounds: tuple[float, float] | None
             ) -> tuple[Tensor, Tensor, Tensor]:
    """Batched weighted similarity fit `dst ~ s R src + t` over (H,n,3) pairs with (H,n) weights.

    `bounds=None` keeps the per-hypothesis `scale` (H,) fixed (Kabsch rotation, then the matching
    translation); otherwise `s` is the Umeyama estimate clamped to `bounds`. Returns `(R, t, s)`.
    """
    wsum = w.sum(1, keepdim=True).clamp_min(1e-9)
    mu_s = (w[..., None] * src).sum(1) / wsum
    mu_d = (w[..., None] * dst).sum(1) / wsum
    cov = torch.einsum("hn,hni,hnj->hij", w, src - mu_s[:, None], dst - mu_d[:, None])
    U, S, Vh = torch.linalg.svd(cov)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(1, 2)))
    signs = torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1)
    R = V @ torch.diag_embed(signs) @ U.transpose(1, 2)
    s = scale
    if bounds is not None:
        var_src = (w * ((src - mu_s[:, None]) ** 2).sum(-1)).sum(1).clamp_min(1e-12)
        s = ((S * signs).sum(-1) / var_src).clamp(bounds[0], bounds[1])
    t = mu_d - s[:, None] * (R @ mu_s[..., None])[..., 0]
    return R, t, s


def _residuals(obs: Tensor, Y: Tensor, Yn: Tensor, R: Tensor, t: Tensor, s: Tensor
               ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Nearest-neighbour residuals obs->front-facing mesh and front-facing mesh->obs, per hypothesis."""
    P = s[:, None, None] * torch.einsum("hij,mj->hmi", R, Y) + t[:, None]
    front = (torch.einsum("hij,mj->hmi", R, Yn) * -P).sum(-1) > 0          # (H,M) camera-facing samples
    D = torch.cdist(obs.expand(R.shape[0], -1, -1), P)                     # (H,N,M)
    d_mesh, i_mesh = D.min(1)                                              # (H,M), over every sample
    # Masked in place after the mesh-side minimum, so one (H,N,M) float64 tensor is alive at a time; the values are
    # those of a masked copy. The copy doubled the peak (2 x 7.5 GiB at 2048 x 4096).
    d_obs, j_obs = D.masked_fill_(~front[:, None, :], math.inf).min(2)      # (H,N)
    return d_obs, j_obs, d_mesh, i_mesh, front


def _fit_hypotheses(points: Tensor, normals: Tensor, observed: Tensor, K: Tensor, mask: Tensor, scale: float,
                    estimate_scale: bool, scale_range: tuple[float, float], n_obs: int, n_mesh: int, n_iters: int,
                    seed: int, device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The batched multi-hypothesis ICP of `register_keyframe_pose`: `(R, t, s, score)` over all 120 hypotheses.

    Args:
        points: `(M,3)` canonical surface samples.
        normals: `(M,3)` outward unit normals, index-aligned with `points`.
        observed: `(N,3)` finite camera-frame points of the object in the keyframe.
        K: `(3,3)` pixel intrinsics.
        mask: `(H,W)` bool keyframe object mask (scores the projected silhouette).
        scale: Scale applied to `points` before posing (the initial scale when `estimate_scale`).
        estimate_scale: Also fit the scale (similarity ICP), after a rigid first third of the iterations.
        scale_range: Bounds on the fitted scale, as multiples of `scale`.
        n_obs: Observed points subsampled per ICP step.
        n_mesh: Mesh samples subsampled per ICP step.
        n_iters: ICP iterations per hypothesis (all hypotheses run batched).
        seed: RNG seed for both subsamplings.
        device: torch device for the batched ICP.

    Returns:
        `(R (H,3,3), t (H,3), s (H,), score (H,))` float64 on `device`; score is +inf for a hypothesis with no
        front-facing sample.

    Raises:
        ValueError: If fewer than 10 observed points survive the outlier filter.
    """
    gen = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    diam = scale * _diameter(points)
    obs_all = observed.to(dtype)
    if obs_all.shape[0] >= 10:
        centre = obs_all.median(dim=0).values
        obs_all = obs_all[(obs_all - centre).norm(dim=1) < diam * scale_range[1]]   # flying pixels / leakage
    if obs_all.shape[0] < 10:
        raise ValueError(f"only {obs_all.shape[0]} observed keyframe points; need at least 10")
    obs = obs_all[torch.randperm(obs_all.shape[0], generator=gen)[:n_obs]].to(device)
    sel = torch.randperm(points.shape[0], generator=gen)[:n_mesh]
    Y = points[sel].to(dtype).to(device)
    Yn = F.normalize(normals[sel].to(dtype), dim=-1).to(device)

    R = _rotation_hypotheses(dtype).to(device)
    s = torch.full((R.shape[0],), scale, dtype=dtype, device=device)
    c_obs = obs.median(dim=0).values
    ray = c_obs / c_obs.norm().clamp_min(1e-9)
    t = (c_obs + 0.25 * diam * ray)[None] - scale * (R @ Y.mean(0))           # centroid slightly behind the surface
    floor = 0.005 * diam
    for it in range(n_iters):
        d_obs, j_obs, d_mesh, i_mesh, front = _residuals(obs, Y, Yn, R, t, s)
        thr_obs = _TRIM_FACTOR * d_obs.nan_to_num(posinf=math.nan).nanmedian(1).values.clamp_min(floor)
        d_mesh_front = torch.where(front, d_mesh, torch.full_like(d_mesh, math.nan))
        thr_mesh = _TRIM_FACTOR * d_mesh_front.nanmedian(1).values.nan_to_num(nan=floor).clamp_min(floor)
        w = torch.cat([(d_obs <= thr_obs[:, None]).to(dtype),
                       (front & (d_mesh <= thr_mesh[:, None])).to(dtype)], dim=1)
        src = torch.cat([Y[j_obs], Y.expand(R.shape[0], -1, -1)], dim=1)
        dst = torch.cat([obs.expand(R.shape[0], -1, -1), obs[i_mesh]], dim=1)
        fit_scale = estimate_scale and it >= n_iters // 3
        R, t, s = _umeyama(src, dst, w, s, (scale * scale_range[0], scale * scale_range[1]) if fit_scale else None)

    d_obs, _j, d_mesh, _i, front = _residuals(obs, Y, Yn, R, t, s)
    tau = _TRUNCATE_FRAC * diam
    n_front = front.sum(1)
    score = (d_obs.clamp_max(tau).mean(1)
             + (d_mesh.clamp_max(tau) * front).sum(1) / n_front.clamp_min(1)
             + _SILHOUETTE_WEIGHT * diam * _outside_fraction(Y, R, t, s, front, K.to(device, dtype),
                                                             mask.to(device)))
    score = torch.where(n_front > 0, score, torch.full_like(score, math.inf))
    return R, t, s, score


def _registration(R: Tensor, t: Tensor, s: Tensor, score: Tensor, h: int) -> KeyframeRegistration:
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = R[h].float().cpu()
    pose[:3, 3] = t[h].float().cpu()
    return KeyframeRegistration(pose=pose, scale=float(s[h]), score=float(score[h]))


def register_keyframe_pose(points: Tensor, normals: Tensor, observed: Tensor, K: Tensor, mask: Tensor,
                           scale: float = 1.0, estimate_scale: bool = False,
                           scale_range: tuple[float, float] = (0.5, 2.0), n_obs: int = 512, n_mesh: int = 1024,
                           n_iters: int = 50, seed: int = 0, device: str = "cpu") -> KeyframeRegistration:
    """Registers a canonical shape to one keyframe's masked camera-frame pointmap (module docstring).

    Args and Raises: as `register_keyframe_hypotheses`. Returns the best-scoring hypothesis (`argmin`: the first on a
    tie), which is always `register_keyframe_hypotheses(...)[0]`.
    """
    R, t, s, score = _fit_hypotheses(points, normals, observed, K, mask, scale, estimate_scale, scale_range, n_obs,
                                     n_mesh, n_iters, seed, device)
    best = int(score.argmin())
    logger.info("register_keyframe_pose: best hypothesis %d/%d, scale %.4f, score %.4f m", best, R.shape[0],
                float(s[best]), float(score[best]))
    return _registration(R, t, s, score, best)


def register_keyframe_hypotheses(points: Tensor, normals: Tensor, observed: Tensor, K: Tensor, mask: Tensor,
                                 scale: float = 1.0, estimate_scale: bool = False,
                                 scale_range: tuple[float, float] = (0.5, 2.0), n_obs: int = 512, n_mesh: int = 1024,
                                 n_iters: int = 50, seed: int = 0, device: str = "cpu", top_k: int = 1,
                                 min_rot_deg: float = 10.0, min_shift: float = 0.005) -> list[KeyframeRegistration]:
    """The registration's `top_k` best DISTINCT hypotheses, best first (branch enumeration).

    The same batched ICP as `register_keyframe_pose` (identical operations, so element 0 is its result bit for bit).
    The 120 hypotheses converge into a few basins; walking them in score order (a stable sort, so ties keep grid
    order), a hypothesis is kept only if it differs from every kept one by more than `min_rot_deg` degrees of rotation
    OR more than `min_shift` metres of posed-template-centroid position.

    Args:
        points: `(M,3)` canonical surface samples.
        normals: `(M,3)` outward unit normals, index-aligned with `points`.
        observed: `(N,3)` finite camera-frame points of the object in the keyframe.
        K: `(3,3)` pixel intrinsics.
        mask: `(H,W)` bool keyframe object mask (scores the projected silhouette).
        scale: Scale applied to `points` before posing (the initial scale when `estimate_scale`).
        estimate_scale: Also fit the scale (similarity ICP), after a rigid first third of the iterations.
        scale_range: Bounds on the fitted scale, as multiples of `scale`.
        n_obs: Observed points subsampled per ICP step.
        n_mesh: Mesh samples subsampled per ICP step.
        n_iters: ICP iterations per hypothesis (all hypotheses run batched).
        seed: RNG seed for both subsamplings.
        device: torch device for the batched ICP.
        top_k: Number of distinct hypotheses returned (fewer when fewer finite-score basins exist).
        min_rot_deg: Rotation (degrees) beyond which two hypotheses are distinct.
        min_shift: Posed-centroid distance (metres) beyond which two hypotheses are distinct.

    Returns:
        Up to `top_k` `KeyframeRegistration`s, ascending score.

    Raises:
        ValueError: If fewer than 10 observed points survive the outlier filter, or `top_k < 1`.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    R, t, s, score = _fit_hypotheses(points, normals, observed, K, mask, scale, estimate_scale, scale_range, n_obs,
                                     n_mesh, n_iters, seed, device)
    order = torch.sort(score.cpu(), stable=True).indices.tolist()
    c = points.to(R.dtype).to(R.device).mean(0)
    centroid = s[:, None] * (R @ c) + t                                        # (H,3) posed template centroids
    cos_min = math.cos(math.radians(min_rot_deg))
    kept: list[int] = []
    for h in order:
        if not math.isfinite(float(score[h])):
            break
        dup = False
        for j in kept:
            cos = min(1.0, max(-1.0, float(((R[h].T @ R[j]).trace() - 1.0) / 2.0)))
            if cos >= cos_min and float((centroid[h] - centroid[j]).norm()) <= min_shift:
                dup = True
                break
        if not dup:
            kept.append(h)
            if len(kept) == top_k:
                break
    if not kept:                                   # every score infinite: argmin's choice, as register_keyframe_pose
        kept = [int(score.argmin())]
    logger.info("register_keyframe_hypotheses: kept %s of %d (scores %s)", kept, R.shape[0],
                [round(float(score[h]), 5) for h in kept])
    return [_registration(R, t, s, score, h) for h in kept]


def _outside_fraction(Y: Tensor, R: Tensor, t: Tensor, s: Tensor, front: Tensor, K: Tensor, mask: Tensor
                      ) -> Tensor:
    """Fraction of front-facing posed samples projecting outside the (dilated) keyframe mask."""
    H, W = mask.shape
    dil = F.max_pool2d(mask.float()[None, None], 2 * _MASK_DILATE_PX + 1, stride=1,
                       padding=_MASK_DILATE_PX)[0, 0] > 0
    uv = project(s[:, None, None] * torch.einsum("hij,mj->hmi", R, Y) + t[:, None], K)
    u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    hit = dil[v.clamp(0, H - 1), u.clamp(0, W - 1)] & inside
    return ((~hit) & front).sum(1) / front.sum(1).clamp_min(1)


def anchors_from_pose(points: Tensor, normals: Tensor, pose: Tensor, scale: float, K: Tensor,
                      image_hw: tuple[int, int], query_uv: Tensor, max_px: float = 2.0) -> tuple[Tensor, Tensor]:
    """Anchors keyframe query pixels on the visible surface of a posed canonical shape.

    Each query pixel takes the visible sample (`visible_samples`: front-facing, in-image, z-buffer
    unoccluded) whose projection is nearest to it, valid when that distance is at most `max_px`.

    Args:
        points: `(M,3)` canonical surface samples (dense enough to cover the silhouette at `max_px`).
        normals: `(M,3)` outward unit normals, index-aligned with `points`.
        pose: `(4,4)` keyframe object-to-camera transform.
        scale: Metric scale applied to `points` before posing.
        K: `(3,3)` pixel intrinsics.
        image_hw: `(H,W)` image size.
        query_uv: `(J,2)` keyframe pixel coordinates `(x,y)`.
        max_px: Maximum pixel distance from a query to its anchor's projection.

    Returns:
        `(anchors, valid)`: `(J,3)` float32 canonical anchors (zeros where invalid) and `(J,)` bool.
    """
    P = points.double()
    poses = pose.double()[None]
    log_scale = torch.tensor(math.log(scale), dtype=torch.float64)
    vis = visible_samples(P, normals.double(), poses, log_scale, K.double(), image_hw)[0]
    cam = transform_points(P, poses, scale)[0]
    uv = project(cam, K.double())
    dist = torch.cdist(query_uv.double(), uv).masked_fill(~vis[None], math.inf)   # (J,M)
    best, idx = dist.min(dim=1)
    valid = best <= max_px
    anchors = torch.where(valid[:, None], points[idx], torch.zeros_like(points[idx]))
    return anchors.float(), valid


def gate_init_poses(poses: Tensor, n_inliers: Tensor, points: Tensor, scale: float, obs: TrajectoryObservations,
                    keyframe_index: int, keyframe_pose: Tensor, min_inliers: int = 6, max_offset: float = 1.0,
                    min_points: int = 20) -> tuple[Tensor, Tensor]:
    """Screens per-frame PnP initializations against each frame's own masked pointmap.

    Real tracks drift under fast motion and occlusion, and EPnP on a small, drifting correspondence
    set can land metres away; the solver's pointmap term has no capture range from there. A frame's
    pose is accepted when PnP solved it directly (`n_inliers >= min_inliers`) and the posed shape
    centroid lies within `max_offset * scale * diameter` of the frame's masked-pointmap median point.
    A directly solved frame with fewer than `min_points` masked pointmap points (heavy occlusion, where
    drift is likeliest) has no such check of its own; it is accepted only when its posed centroid lies
    within the same distance of the nearest frame that passed the pointmap check (or the keyframe).
    The keyframe always takes `keyframe_pose`. A rejected frame takes the rotation of the nearest
    accepted frame, translated so the shape centroid sits a quarter diameter behind the frame's
    observed median point along its viewing ray (the registration's own translation start); with
    fewer than `min_points` pointmap points it keeps the nearest accepted pose unchanged.

    Args:
        poses: `(T,4,4)` per-frame initial poses (e.g. `pnp_initialize`'s).
        n_inliers: `(T,)` PnP inlier counts (0 where unsolved).
        points: `(M,3)` canonical surface samples.
        scale: Scale of `points` in the pointmap metric.
        obs: The clip's observations (`masks`, and `object_pixels` or else `pointmaps`, are read).
        keyframe_index: Frame whose pose is `keyframe_pose`.
        keyframe_pose: `(4,4)` keyframe object-to-camera transform.
        min_inliers: Minimum PnP inliers for a direct solve.
        max_offset: Accepted centroid offset, in (scaled) diameters.
        min_points: Minimum masked pointmap points for the centroid test and the fallback translation.

    Returns:
        `(poses, accepted)`: `(T,4,4)` float32 gated poses and `(T,)` bool accepted-as-given mask.
    """
    T = poses.shape[0]
    diam = scale * _diameter(points)
    centroid = points.double().mean(0)
    out = poses.double().clone()
    out[keyframe_index] = keyframe_pose.double()
    observed: list[Tensor | None] = []
    px = obs.object_pixels
    for k in range(T):
        # mask first: finite-check only object pixels (`object_pixels` holds exactly those when set)
        pts = obs.pointmaps[k][obs.masks[k]] if px is None else px.frame(k)
        pts = pts[torch.isfinite(pts).all(-1)]
        observed.append(pts.double().median(dim=0).values if pts.shape[0] >= min_points else None)
    posed_centroids = scale * out[:, :3, :3] @ centroid + out[:, :3, 3]    # (T,3)
    solved = n_inliers >= min_inliers
    accepted = torch.zeros(T, dtype=torch.bool)
    for k in range(T):
        c_obs = observed[k]
        if bool(solved[k]) and c_obs is not None:
            accepted[k] = bool((posed_centroids[k] - c_obs).norm() <= max_offset * diam)
    accepted[keyframe_index] = True
    checked = torch.nonzero(accepted).flatten()                               # verified on their own pointmap
    for k in range(T):
        if bool(solved[k]) and observed[k] is None and k != keyframe_index:
            ref = int(checked[(checked - k).abs().argmin()])
            accepted[k] = bool((posed_centroids[k] - posed_centroids[ref]).norm() <= max_offset * diam)
    good = torch.nonzero(accepted).flatten()
    for k in torch.nonzero(~accepted).flatten().tolist():
        out[k] = out[int(good[(good - k).abs().argmin()])]
        c_obs = observed[k]
        if c_obs is not None:
            ray = c_obs / c_obs.norm().clamp_min(1e-9)
            out[k, :3, 3] = c_obs + 0.25 * diam * ray - scale * out[k, :3, :3] @ centroid
    logger.info("gate_init_poses: %d/%d frames keep their PnP pose", int(accepted.sum()), T)
    return out.float(), accepted
