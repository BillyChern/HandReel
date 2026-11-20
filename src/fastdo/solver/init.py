"""Closed-form initialization: canonical anchors from the keyframe and per-frame PnP from tracks."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _diameter(points: Tensor) -> float:
    return float((points.max(0).values - points.min(0).values).norm())


def assign_anchors(surface_points: Tensor, pointmap_k0: Tensor, tracks_k0: Tensor, vis_k0: Tensor,
                   pose_k0: Tensor, scale: float, max_dist: float = 0.02) -> tuple[Tensor, Tensor]:
    """Assign each keyframe track to its nearest canonical surface sample.

    Each visible track pixel in the keyframe is unprojected through the keyframe pointmap to a
    camera-frame 3D point, mapped into the canonical (unit-scale, object) frame using the
    keyframe pose and metric scale, and snapped to its nearest neighbour among `surface_points`.

    Args:
        surface_points: Canonical (unit-scale) surface samples, (M,3).
        pointmap_k0: Keyframe camera-frame pointmap, (H,W,3), with non-finite entries where depth
            is unknown.
        tracks_k0: Keyframe 2D pixel track locations, (J,2).
        vis_k0: Per-track visibility mask in the keyframe, (J,).
        pose_k0: Keyframe object-to-camera transform, (4,4).
        scale: Metric scale factor mapping canonical points to metres.
        max_dist: Maximum snap distance, as a fraction of the canonical shape's diameter, for an
            assignment to be considered valid.

    Returns:
        A tuple `(anchors, valid)`: `anchors` is (J,3), the nearest canonical surface point for
        every track (regardless of validity); `valid` is (J,), a boolean mask that is True only
        where the track is visible, has a finite unprojected point, and snapped within
        `max_dist * diameter(surface_points)`.
    """
    H, W = pointmap_k0.shape[:2]
    u = tracks_k0[:, 0].round().long().clamp(0, W - 1)
    v = tracks_k0[:, 1].round().long().clamp(0, H - 1)
    p_cam = pointmap_k0[v, u]                                     # (J,3)
    ok = vis_k0 & torch.isfinite(p_cam).all(-1)
    R, t = pose_k0[:3, :3], pose_k0[:3, 3]
    canon = ((p_cam - t) @ R) / scale                             # R^T (p - t) / s
    d = torch.cdist(torch.nan_to_num(canon), surface_points)
    dist, j = d.min(dim=1)
    ok = ok & (dist < max_dist * _diameter(surface_points))
    anchors = surface_points[j]
    return anchors, ok


def scale_from_extent(pointmap_k0: Tensor, mask_k0: Tensor, surface_points: Tensor) -> float:
    """Estimate the metric scale factor from the observed keyframe point extent.

    Compares the robust (5th-95th percentile) extent of the valid, masked keyframe pointmap
    points against the canonical shape's diameter to recover the metric scale multiplier.

    Args:
        pointmap_k0: Keyframe camera-frame pointmap, (H,W,3), with non-finite entries where depth
            is unknown.
        mask_k0: Per-pixel validity mask selecting the object in the keyframe, (H,W).
        surface_points: Canonical (unit-scale) surface samples, (M,3).

    Returns:
        The estimated metric scale factor (observed extent divided by 0.9 times the canonical
        diameter).

    Raises:
        ValueError: If fewer than 10 valid pointmap points fall inside the mask.
    """
    pts = pointmap_k0[mask_k0 & torch.isfinite(pointmap_k0).all(-1)]
    if pts.shape[0] < 10:
        raise ValueError("too few valid pointmap points inside the mask for scale initialization")
    lo, hi = torch.quantile(pts, 0.05, dim=0), torch.quantile(pts, 0.95, dim=0)
    observed = float((hi - lo).norm())
    return observed / (0.9 * _diameter(surface_points))


# E|P v| / |v| for a uniformly oriented 3D vector v projected onto a plane: sqrt(2/3). Derived, not fitted --
# measured: the lateral/3D bias ratio of the two rulers below at 0.815 on 71 metric clips.
_ISOTROPIC_PROJECTION = (2.0 / 3.0) ** 0.5


def scale_from_lateral_extent(pointmap_k0: Tensor, mask_k0: Tensor, surface_points: Tensor) -> float:
    """`scale_from_extent`'s estimate with depth entering only through a MEDIAN.

    The 3D extent is dominated by its worst axis, and at a silhouette the pointmap's depth is interpolated
    between object and background ("flying pixels") -- contamination that only ever ADDS extent. It reached
    45x on a hand-held object (rank correlation +0.649 with the masked depth spread), and the solver
    cannot repair an initial scale: it moves scale by ~3%. Here the extent is taken in NORMALISED
    IMAGE coordinates (x/z, y/z) -- where a mask is accurate and no depth edge can inflate it -- and lifted
    to metres by the median masked depth, which tolerates up to half the pixels being off-object where the
    per-axis 5-95% quantile tolerates 5%. Worst case over the same 71 clips: 2.7x instead of 45x.

    The 0.9 convention is `scale_from_extent`'s, so the two rulers share an expected bias and differ only
    in scatter; `_ISOTROPIC_PROJECTION` accounts for measuring a projected rather than a 3D diagonal.

    Raises:
        ValueError: If fewer than 10 valid, in-front pointmap points fall inside the mask.
    """
    pts = pointmap_k0[mask_k0 & torch.isfinite(pointmap_k0).all(-1) & (pointmap_k0[..., 2] > 0)]
    if pts.shape[0] < 10:
        raise ValueError("too few valid pointmap points inside the mask for scale initialization")
    z = pts[:, 2]
    image_plane = pts[:, :2] / z[:, None]
    lo, hi = torch.quantile(image_plane, 0.05, dim=0), torch.quantile(image_plane, 0.95, dim=0)
    observed = float((hi - lo).norm()) * float(z.median())
    return observed / (0.9 * _ISOTROPIC_PROJECTION * _diameter(surface_points))


def pnp_initialize(anchors: Tensor, anchor_valid: Tensor, tracks: Tensor, vis: Tensor, K: Tensor, scale: float,
                   min_inliers: int = 6) -> tuple[Tensor, Tensor]:
    """Solve a per-frame object pose from 2D-3D track correspondences via RANSAC PnP.

    For each frame, the canonical anchors visible in that frame (masked by `anchor_valid` and
    `vis`) are lifted to metric scale and matched against their tracked pixel locations with
    `cv2.solvePnPRansac` (EPnP) followed by `cv2.solvePnPRefineLM` on the inlier set. A frame is
    treated as unsolved (and later copies the nearest directly solved frame's pose) whenever it
    has fewer than `min_inliers` visible correspondences, RANSAC/refinement fails or raises
    `cv2.error` (OpenCV degeneracy, e.g. a near-planar or collinear point set), or the resulting
    rotation/translation is non-finite or not a valid rotation (`|det(R) - 1| >= 1e-3`).

    Args:
        anchors: Canonical (object-frame, unit-scale) 3D anchor points, (J,3).
        anchor_valid: Per-anchor validity mask from `assign_anchors`, (J,).
        tracks: Per-frame 2D pixel tracks for each anchor, (T,J,2).
        vis: Per-frame per-anchor visibility mask, (T,J).
        K: Camera intrinsics matrix, (3,3).
        scale: Metric scale factor mapping canonical anchors to metres.
        min_inliers: Minimum number of RANSAC inliers (and visible correspondences) required for
            a frame to count as directly solved.

    Returns:
        A tuple `(poses, n_inliers)`: `poses` is (T,4,4) object-to-camera transforms (float32),
        with unsolved frames set to the nearest directly solved frame's pose; `n_inliers` is
        (T,), the inlier count per frame (0 for frames that never attempted or failed a solve).

    Raises:
        ValueError: If no frame solves directly (RANSAC/refinement never produces a valid,
            sufficiently-inlying, non-degenerate pose on any frame).
    """
    T = tracks.shape[0]
    obj = (anchors * scale).double().cpu().numpy()
    Kn = K.double().cpu().numpy()
    poses = torch.zeros(T, 4, 4, dtype=torch.float32)
    n_in = torch.zeros(T, dtype=torch.int64)
    for k in range(T):
        sel = (vis[k] & anchor_valid).cpu().numpy()
        if sel.sum() < min_inliers:
            continue
        img = tracks[k][torch.as_tensor(sel)].double().cpu().numpy()
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj[sel], img, Kn, None, iterationsCount=200,
                                                     reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inl is None or len(inl) < min_inliers:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(obj[sel][inl[:, 0]], img[inl[:, 0]], Kn, None, rvec, tvec)
            Rm, _ = cv2.Rodrigues(rvec)
        except cv2.error:
            logger.debug("pnp_initialize: frame %d raised cv2.error, treating as unsolved", k)
            continue
        if not (np.isfinite(Rm).all() and np.isfinite(tvec).all()):
            logger.debug("pnp_initialize: frame %d produced a non-finite pose, treating as unsolved", k)
            continue
        if abs(np.linalg.det(Rm) - 1) >= 1e-3:
            logger.debug("pnp_initialize: frame %d produced a non-orthonormal rotation, treating as unsolved", k)
            continue
        poses[k, :3, :3] = torch.as_tensor(Rm, dtype=torch.float32)
        poses[k, :3, 3] = torch.as_tensor(tvec[:, 0], dtype=torch.float32)
        poses[k, 3, 3] = 1.0
        n_in[k] = len(inl)
    solved = torch.nonzero(n_in >= min_inliers).flatten()
    if solved.numel() == 0:
        raise ValueError("PnP failed on every frame; need a keyframe pose from the shape model")
    for k in range(T):
        if n_in[k] < min_inliers:
            nearest = solved[(solved - k).abs().argmin()]
            poses[k] = poses[nearest]
    logger.info("pnp_initialize: %d/%d frames solved directly", int(solved.numel()), T)
    return poses, n_in
