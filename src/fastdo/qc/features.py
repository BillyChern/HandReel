"""Per-clip QC features: reprojection IoU, Laplace std, OT residual, hand penetration, track inliers.

Rasterizer note: `fastdo.solver.terms_image` exposes no
rasterizer -- its silhouette term works on distance transforms and boundary pixels, never on a rendered
occupancy image -- so reprojection IoU uses the minimal point-splat rasterizer below. Posed canonical
surface samples are projected with `fastdo.solver.terms.{transform_points, project}` (the solver's own
camera model), splatted to their nearest pixel, and closed (dilate, then erode, by the same radius).
The radius starts at a 1-pixel dilation and grows with the projected sample spacing, because
a fixed 1-pixel dilation measured IoU 0.44-0.95 against an exact analytic sphere silhouette over
1k-20k random surface samples (speckle holes at solver density, a 1-pixel outward halo when dense),
while the adaptive closing measured 0.982-0.990 over the same range.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import astuple, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import Tensor

from fastdo.solver.api import MeshSamples
from fastdo.solver.sdf import SdfGrid, sdf_lookup
from fastdo.solver.terms import project, transform_points

logger = logging.getLogger(__name__)

__all__ = [
    "QcFeatures",
    "laplace_std_ok_frac",
    "ot_residual_rms_frac",
    "penetration_depth_m",
    "qc_features",
    "render_silhouettes",
    "reprojection_iou",
    "track_inlier_ratio",
]


@dataclass(frozen=True)
class QcFeatures:
    reprojection_iou: float        # mean per-frame IoU of the posed-mesh splat vs the object mask
    laplace_std_ok_frac: float     # fraction of frames with Laplace translation std below threshold
    ot_residual_rms_frac: float    # RMS over frames of the per-frame data cost, / object diameter
    penetration_depth_m: float     # max hand-vertex penetration into the object, metres
    track_inlier_ratio: float      # fraction of track IRLS weights above 0.5

    def to_array(self) -> np.ndarray:
        """Feature vector (5,) float64 in field order (the order `fastdo.qc.acceptance` standardizes)."""
        return np.asarray(astuple(self), dtype=np.float64)


def _splat_radii(points: Tensor, p_cam: Tensor, scale: float, K: Tensor, spacing_factor: float,
                 max_radius_px: int) -> Tensor:
    """Per-frame closing radius (T,) long: `spacing_factor` x the projected median nearest-neighbour
    spacing of the samples at the frame's median depth, clamped to [1, max_radius_px].

    The median NN spacing of random area samples underestimates their largest gaps; the measured
    factor needed to close an analytic sphere's splat was 1.8-2.3 (1k-4k samples), hence 2.5.
    """
    pts = points.detach().cpu().numpy().astype(np.float64)
    if pts.shape[0] < 2:
        return torch.ones(p_cam.shape[0], dtype=torch.long)
    nn_dist, _ = cKDTree(pts).query(pts, k=2)
    spacing_m = abs(scale) * float(np.median(nn_dist[:, 1]))
    z = p_cam[..., 2].where(p_cam[..., 2] > 0, torch.full_like(p_cam[..., 2], float("nan")))
    z_med = torch.nanmedian(z, dim=1).values.nan_to_num(nan=float("inf"))                # (T,)
    focal = float(max(float(K[0, 0]), float(K[1, 1])))
    px = spacing_factor * focal * spacing_m / z_med.clamp_min(1e-6)
    return px.ceil().clamp(1, max_radius_px).long()


def _max_filter(img: Tensor, radius: int) -> Tensor:
    """Max over a (2r+1)^2 square, separably (a row pass then a column pass: O(r) per pixel, not O(r^2))."""
    k = 2 * radius + 1
    rows = F.max_pool2d(img, (1, k), stride=1, padding=(0, radius))
    return F.max_pool2d(rows, (k, 1), stride=1, padding=(radius, 0))


def _close(img: Tensor, radius: int) -> Tensor:
    """Binary closing of a (1,1,h,w) float image with a (2r+1)^2 square. Erosion pads with foreground,
    so pixels at the image border are never eroded by the out-of-image region (a clipped object)."""
    return -_max_filter(-_max_filter(img, radius), radius)


def _silhouette_crops(points: Tensor, poses: Tensor, scale: float, K: Tensor, image_hw: tuple[int, int],
                      spacing_factor: float, max_radius_px: int) -> Iterator[tuple[int, int, int, int, int, Tensor]]:
    """`(k, v0, v1, u0, u1, crop (v1-v0, u1-u0) bool)` for every frame whose splat is non-empty: the closed splat on
    its padded bounding box, the only region where frame k's silhouette can be True."""
    H, W = image_hw
    T = poses.shape[0]
    p = transform_points(points.to(torch.float64), poses.to(torch.float64), float(scale))   # (T,M,3)
    uv = project(p, K.to(torch.float64))
    u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (p[..., 2] > 0)
    radii = _splat_radii(points, p, scale, K, spacing_factor, max_radius_px)
    for k in range(T):
        uk, vk = u[k][ok[k]], v[k][ok[k]]
        if uk.numel() == 0:
            continue
        r = int(radii[k])
        pad = 2 * r + 1
        u0, u1 = max(int(uk.min()) - pad, 0), min(int(uk.max()) + pad + 1, W)
        v0, v1 = max(int(vk.min()) - pad, 0), min(int(vk.max()) + pad + 1, H)
        crop = torch.zeros(v1 - v0, u1 - u0, dtype=torch.bool)
        crop[vk - v0, uk - u0] = True
        yield k, v0, v1, u0, u1, _close_binary(crop, r)


def _box_count(x: Tensor, r: int) -> Tensor:
    """Count of ones of the (h,w) integer image `x` in each (2r+1)^2 window, zero-padded, exactly: separable
    sliding sums from cumulative sums, O(1) per pixel whatever the radius."""
    k = 2 * r + 1
    c = F.pad(x, (r + 1, r)).cumsum(1)
    x = c[:, k:] - c[:, :-k]
    c = F.pad(x, (0, 0, r + 1, r)).cumsum(0)
    return c[k:, :] - c[:-k, :]


def _close_binary(img: Tensor, radius: int) -> Tensor:
    """`_close` for an (h,w) bool image, bit-exactly and O(1) per pixel: dilation is "any one in the window",
    erosion is "no in-image zero in the window" (so, as in `_close`, out-of-image pixels act as foreground and a
    clipped object's border is not eroded). `_close`'s former max-pooling was 4.5 s of a 7.5 s qc stage at S2 size."""
    dilated = _box_count(img.to(torch.int32), radius) > 0
    return _box_count((~dilated).to(torch.int32), radius) == 0


def render_silhouettes(points: Tensor, poses: Tensor, scale: float, K: Tensor, image_hw: tuple[int, int],
                       spacing_factor: float = 2.5, max_radius_px: int = 32) -> Tensor:
    """Point-splat silhouettes (T,H,W) bool of canonical samples `points` (M,3) posed by `poses` (T,4,4).

    Each sample is transformed (`s R y + t`) and projected with `K`; samples in front of the camera and
    inside the image mark their nearest pixel; each frame's splat is then closed on its bounding box
    (padded by the radius, so the work scales with the object's image size, not the image's).
    """
    H, W = image_hw
    out = torch.zeros(poses.shape[0], H, W, dtype=torch.bool)
    for k, v0, v1, u0, u1, crop in _silhouette_crops(points, poses, scale, K, image_hw, spacing_factor, max_radius_px):
        out[k, v0:v1, u0:u1] = crop
    return out


def reprojection_iou(masks: Tensor, poses: Tensor, scale: float, points: Tensor, K: Tensor) -> float:
    """Mean over frames of IoU(rendered silhouette, `masks` (T,H,W)); frames whose union is empty (object
    neither rendered nor segmented) carry no evidence and are skipped; 0.0 if every frame is empty.

    The mask is the segmenter's visible-object mask, so a hand occluding the object lowers the IoU of
    a correct pose (the render ignores occluders); the acceptance model is fitted with that bias in.

    Counted, not rendered at full size: the silhouette is False outside each frame's crop, so
    |S & M| is counted on the crop and |S | M| = |S| + |M| - |S & M|. Integer counts, so the value is exactly the
    full-image one; building the (T,H,W) silhouette and two full-image passes cost 6.8 s of a 7.5 s qc stage at
    S2's 100 x 1080x1920.
    """
    if masks.dim() != 3 or masks.shape[0] != poses.shape[0]:
        raise ValueError(f"masks must be (T,H,W) with T = poses.shape[0]: {tuple(masks.shape)}")
    m = masks.to(device="cpu", dtype=torch.bool)
    T = int(m.shape[0])
    mask_px = torch.count_nonzero(m.reshape(T, -1), dim=1)
    inter = torch.zeros(T, dtype=torch.long)
    sil_px = torch.zeros(T, dtype=torch.long)
    for k, v0, v1, u0, u1, crop in _silhouette_crops(points, poses, scale, K, (int(m.shape[1]), int(m.shape[2])),
                                                      2.5, 32):
        inter[k] = torch.count_nonzero(crop & m[k, v0:v1, u0:u1])
        sil_px[k] = torch.count_nonzero(crop)
    inter_f, union = inter.double(), (sil_px + mask_px - inter).double()
    valid = union > 0
    if not bool(valid.any()):
        logger.warning("reprojection_iou: no frame has a rendered or segmented object; returning 0.0")
        return 0.0
    return float((inter_f[valid] / union[valid]).mean())


def laplace_std_ok_frac(covariance: Tensor, threshold: float = 0.01) -> float:
    """Fraction of frames whose translation std `sqrt(trace(Sigma[:3,:3]))` is below `threshold` (m).

    `covariance` is the solver's per-frame Laplace block, (T,6,6) as `solve_trajectory` returns it or
    (T,7,7) with the shared log-scale; the first three parameters are the left se(3) perturbation's
    translation `rho`. Under a left perturbation the object origin moves by `rho + theta x t`, so this
    block also absorbs rotation uncertainty scaled by the object's distance -- by design it flags any
    frame whose pose is loose at the camera-frame origin, not only a loose object centre.
    """
    if covariance.dim() != 3 or covariance.shape[1] < 3 or covariance.shape[2] < 3:
        raise ValueError(f"covariance must be (T,d,d) with d >= 3: {tuple(covariance.shape)}")
    if covariance.shape[0] == 0:
        return 0.0
    block = covariance[:, :3, :3].to(torch.float64)
    std = torch.diagonal(block, dim1=1, dim2=2).sum(-1).clamp_min(0.0).sqrt()
    return float((std < threshold).double().mean())


def ot_residual_rms_frac(per_frame_cost: Tensor, diameter: float) -> float:
    """`sqrt(mean_t cost_t^2) / diameter` for the solver's per-frame RMS data cost (metres)."""
    if not diameter > 0.0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    cost = per_frame_cost.to(torch.float64).flatten()
    if cost.numel() == 0 or not bool(torch.isfinite(cost).all()):
        raise ValueError("per_frame_cost must be non-empty and finite")
    return float((cost**2).mean().sqrt()) / diameter


def penetration_depth_m(hand_verts: Tensor | None, sdf: SdfGrid | None, poses: Tensor, scale: float) -> float:
    """Max over frames and hand vertices of `max(-phi, 0)`, metres; 0.0 without hands or an SDF.

    Camera-frame vertices (T,V,3) map to the canonical frame as `R^T (h - t) / s` (as `contact_term`
    does); `phi` is canonical-frame distance, so the metric depth is `s * max(-phi, 0)`. Non-finite
    vertices (frames with no hand estimate) are ignored.
    """
    if hand_verts is None or sdf is None:
        return 0.0
    h = hand_verts.to(torch.float64)
    if h.dim() != 3 or h.shape[0] != poses.shape[0] or h.shape[2] != 3:
        raise ValueError(f"hand_verts must be (T,V,3) with T = poses.shape[0]: {tuple(h.shape)}")
    finite = torch.isfinite(h).all(-1)                                              # (T,V)
    if not bool(finite.any()):
        return 0.0
    P = poses.to(torch.float64)
    R, t = P[:, :3, :3], P[:, :3, 3]
    h = torch.where(finite[..., None], h, t[:, None, :])
    y = torch.einsum("tji,tvj->tvi", R, h - t[:, None, :]) / float(scale)
    phi, _ = sdf_lookup(sdf, y)
    depth = (-phi).clamp_min(0.0).masked_fill(~finite, 0.0)
    return abs(float(scale)) * float(depth.max())


def track_inlier_ratio(track_weight: Tensor, threshold: float = 0.5) -> float:
    """Fraction of all track IRLS weights strictly above `threshold` (invisible tracks weigh 0)."""
    w = track_weight.flatten()
    if w.numel() == 0:
        return 0.0
    return float((w > threshold).double().mean())


def qc_features(masks: Tensor, poses: Tensor, scale: float, mesh: MeshSamples, covariance: Tensor,
                per_frame_cost: Tensor, K: Tensor, diameter: float, hand_verts: Tensor | None,
                sdf: SdfGrid | None, track_weight: Tensor, laplace_std_threshold: float = 0.01) -> QcFeatures:
    """All five per-clip QC features for one solved clip (CPU; inputs are moved to CPU float64)."""
    cpu_poses = poses.detach().to("cpu", torch.float64)
    qc = QcFeatures(
        reprojection_iou=reprojection_iou(masks.detach().cpu(), cpu_poses, scale, mesh.points.detach().cpu(),
                                          K.detach().cpu()),
        laplace_std_ok_frac=laplace_std_ok_frac(covariance.detach().cpu(), laplace_std_threshold),
        ot_residual_rms_frac=ot_residual_rms_frac(per_frame_cost.detach().cpu(), diameter),
        penetration_depth_m=penetration_depth_m(None if hand_verts is None else hand_verts.detach().cpu(),
                                                sdf, cpu_poses, scale),
        track_inlier_ratio=track_inlier_ratio(track_weight.detach().cpu()),
    )
    if not all(math.isfinite(x) for x in astuple(qc)):
        raise ValueError(f"non-finite QC feature: {qc}")
    logger.debug("qc_features: %s", qc)
    return qc
