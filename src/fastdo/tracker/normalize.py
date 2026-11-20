"""Scale-free tracker inputs from the feature tokens (`fastdo.tracker.features`).

The features leave the raw track block in pixel units. `track_image_scale` is the apparent object size `D_img = sqrt(fx
fy) L / max(z, 0.05 L)` in pixels; dividing the track gradient and curvature by `D_img^2` and the track residual by
`D_img` makes them commensurable with the canonical pointmap block and invariant under a scene similarity and a pixel
rescale. A fixed elementwise squash (floor 1e-3, the features' histogram lower edge) only compresses dynamic range, so
it keeps every invariance exact.

Float64, device-agnostic, no host read and no data-dependent Python branch (validation reads shapes only). Frame
and track normalization zero non-finite rows; point and cluster tokens are NOT sanitized (the features' convention), so
`build_tracker_inputs` can detect a non-finite row and mask its frame.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from fastdo.solver.terms import TermOutput
from fastdo.tracker.features import frame_feature_dim

__all__ = ["CLUSTER_TOKEN_DIM", "POINT_TOKEN_DIM", "cluster_geometry", "cluster_tokens", "normalize_frame_tokens",
           "normalize_track_tokens", "point_tokens", "track_image_scale"]

logger = logging.getLogger(__name__)

F64 = torch.float64
POINT_TOKEN_DIM: int = 7
CLUSTER_TOKEN_DIM: int = 5
_INPUT_FLOOR = 1e-3          # = 10 ** features._LOG10_LO, the histogram's lower edge in units of L (tied by test N7)
_MASS_FLOOR = 1e-12          # = features._MASS_FLOOR, the barycenter clamp (tied by test N7)
_DEPTH_FLOOR = 0.05          # relative depth floor of D_img, in units of L


def _per_frame(value: Tensor | float, n_frames: int, device: torch.device, name: str) -> Tensor:
    """A float, 0-dim or (T,) value -> (T,) float64 on `device` (no host read)."""
    if isinstance(value, Tensor):
        if value.dim() > 1 or (value.dim() == 1 and value.shape[0] != n_frames):
            raise ValueError(f"{name} must be a float, 0-dim or ({n_frames},), got shape {tuple(value.shape)}")
        return value.to(device=device, dtype=F64).expand(n_frames)
    return torch.full((n_frames,), float(value), dtype=F64, device=device)


def track_image_scale(K: Tensor, poses: Tensor, length_scale: Tensor | float) -> Tensor:
    """(T,) float64 apparent object size in pixels: `sqrt(fx fy) * L / max(z, 0.05 L)`, `z` the origin depth."""
    if poses.dim() != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (T,4,4), got {tuple(poses.shape)}")
    T, dev = poses.shape[0], poses.device
    if K.shape != (3, 3) and K.shape != (T, 3, 3):
        raise ValueError(f"K must be (3,3) or ({T},3,3), got {tuple(K.shape)}")
    Kd = K.to(device=dev, dtype=F64)
    f = torch.sqrt(Kd[..., 0, 0] * Kd[..., 1, 1]).expand(T)
    L = _per_frame(length_scale, T, dev, "length_scale")
    z = poses[:, 2, 3].to(F64)
    return f * L / torch.maximum(z, _DEPTH_FLOOR * L)


def normalize_frame_tokens(frame_tokens: Tensor, image_scale: Tensor, n_params: int = 7, n_bins: int = 16,
                           n_clusters: int = 64) -> tuple[Tensor, Tensor]:
    """((T,F) float64 squashed tokens, (T,) bool finite); non-finite rows (or bad `image_scale`) become 0.

    `asinh(g_pm / 1e-3) | log1p(diagH_pm / 1e-3) | asinh(g_tr / D^2 / 1e-3) | log1p(diagH_tr / D^2 / 1e-3)`, the
    remaining columns (histogram, cluster mass, support fractions) bit-equal.
    """
    width = frame_feature_dim(n_params, n_bins, n_clusters)
    if frame_tokens.dim() != 2 or frame_tokens.shape[1] != width:
        raise ValueError(f"frame_tokens must be (T,{width}), got {tuple(frame_tokens.shape)}")
    T, P = frame_tokens.shape[0], n_params
    if image_scale.shape != (T,):
        raise ValueError(f"image_scale must be ({T},), got {tuple(image_scale.shape)}")
    x = frame_tokens.to(F64)
    d = image_scale.to(device=x.device, dtype=F64)
    d2 = (d * d)[:, None]
    out = torch.cat([torch.asinh(x[:, 0:P] / _INPUT_FLOOR), torch.log1p(x[:, P:2 * P] / _INPUT_FLOOR),
                     torch.asinh(x[:, 2 * P:3 * P] / d2 / _INPUT_FLOOR),
                     torch.log1p(x[:, 3 * P:4 * P] / d2 / _INPUT_FLOOR), x[:, 4 * P:]], dim=1)
    finite = torch.isfinite(x).all(1) & torch.isfinite(d) & (d > 0) & torch.isfinite(out).all(1)
    return torch.where(finite[:, None], out, torch.zeros_like(out)), finite


def _normalize_track(track_tokens: Tensor, image_scale: Tensor) -> tuple[Tensor, Tensor]:
    """((T,J,6) normalized, (T,J) bool row ok): column 3 `log1p(residual px / D / 1e-3)`, the rest bit-equal."""
    if track_tokens.dim() != 3 or track_tokens.shape[-1] != 6:
        raise ValueError(f"track_tokens must be (T,J,6), got {tuple(track_tokens.shape)}")
    T = track_tokens.shape[0]
    if image_scale.shape != (T,):
        raise ValueError(f"image_scale must be ({T},), got {tuple(image_scale.shape)}")
    t = track_tokens.to(F64)
    d = image_scale.to(device=t.device, dtype=F64)
    col3 = torch.log1p(t[..., 3] / d[:, None] / _INPUT_FLOOR)
    out = torch.cat([t[..., 0:3], col3[..., None], t[..., 4:6]], dim=-1)
    ok = torch.isfinite(t).all(-1) & (torch.isfinite(d) & (d > 0))[:, None] & torch.isfinite(out).all(-1)
    return torch.where(ok[..., None], out, torch.zeros_like(out)), ok


def normalize_track_tokens(track_tokens: Tensor, image_scale: Tensor) -> Tensor:
    """(T,J,6) float64; a non-finite row, or every row of a frame with a bad `image_scale`, becomes all 0."""
    return _normalize_track(track_tokens, image_scale)[0]


def point_tokens(X: Tensor, pm: TermOutput, row_mass: Tensor, poses: Tensor, length_scale: Tensor | float) -> Tensor:
    """(T,N,7) float64 `[R^T (x - t) / L (3), asinh(R^T (x - pbar) / L / 1e-3) (3), N * row_mass (1)]`.

    Invalid rows (`pm.weight <= 0`) are 0; the matched residual is 0 where the row mass is at or below the
    barycenter clamp. The residual is read back from the pointmap convention `sqrt(N w) (x - pbar)`. Nothing else is
    sanitized: a non-finite `pm.residual`, `X` or `row_mass` on a valid row stays in that row.
    """
    if pm.residual.dim() != 3 or pm.residual.shape[-1] != 3:
        raise ValueError(f"need a point-to-point (T,N,3) residual, got {tuple(pm.residual.shape)}")
    T, N = pm.weight.shape
    if X.shape != (T, N, 3):
        raise ValueError(f"X must be ({T},{N},3), got {tuple(X.shape)}")
    if row_mass.shape != (T, N):
        raise ValueError(f"row_mass must be ({T},{N}), got {tuple(row_mass.shape)}")
    if poses.shape != (T, 4, 4):
        raise ValueError(f"poses must be ({T},4,4), got {tuple(poses.shape)}")
    dev = pm.residual.device
    valid = pm.weight > 0
    rm = row_mass.to(device=dev, dtype=F64)
    m = N * rm
    matched = valid & (rm > _MASS_FLOOR)
    pose = poses.to(device=dev, dtype=F64)
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    L = _per_frame(length_scale, T, dev, "length_scale")
    Xc = torch.where(valid[..., None], X.to(device=dev, dtype=F64), torch.zeros((), dtype=F64, device=dev))
    y = ((Xc - t[:, None]) @ R) / L[:, None, None]
    rc = (pm.residual.to(F64) @ R) / (m.clamp_min(_MASS_FLOOR).sqrt() * L[:, None])[..., None]
    rc = torch.where(matched[..., None], rc, torch.zeros_like(rc))
    out = torch.cat([y, torch.asinh(rc / _INPUT_FLOOR), m[..., None]], dim=-1)
    return torch.where(valid[..., None], out, torch.zeros_like(out))


def cluster_geometry(points: Tensor, cluster_id: Tensor, template_diameter: float,
                     n_clusters: int = 64) -> tuple[Tensor, Tensor]:
    """((C,3) member mean / diameter, 0 for an empty cluster; (C,) member fraction), float64, device ops only.

    Pass the samples the plan's columns index (`mesh_cluster_ids`' contract) with their ids. Counts use
    `index_add_` rather than `bincount`, whose output size is a host read.
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (M,3), got {tuple(points.shape)}")
    M = points.shape[0]
    if cluster_id.shape != (M,):
        raise ValueError(f"cluster_id must be ({M},), got {tuple(cluster_id.shape)}")
    pts = points.to(F64)
    ids = cluster_id.to(pts.device)
    counts = torch.zeros(n_clusters, dtype=F64, device=pts.device).index_add_(0, ids, torch.ones_like(pts[:, 0]))
    sums = torch.zeros(n_clusters, 3, dtype=F64, device=pts.device).index_add_(0, ids, pts)
    mean = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1.0)[:, None], torch.zeros_like(sums))
    return mean / template_diameter, counts / M


def cluster_tokens(frame_tokens: Tensor, centroids: Tensor, fraction: Tensor, n_params: int = 7,
                   n_bins: int = 16) -> Tensor:
    """(T,C,5) float64 `[plan column mass (1), centroid / diameter (3), member fraction (1)]`, not sanitized.

    `centroids` (C,3) or (T,C,3) and `fraction` (C,) or (T,C); per-clip values of a batch are gathered with
    `[layout.clip_index]` (clusters.py's padding rule).
    """
    if centroids.dim() not in (2, 3) or centroids.shape[-1] != 3:
        raise ValueError(f"centroids must be (C,3) or (T,C,3), got {tuple(centroids.shape)}")
    C = centroids.shape[-2]
    lo = 4 * n_params + n_bins
    width = frame_feature_dim(n_params, n_bins, C)
    if frame_tokens.dim() != 2 or frame_tokens.shape[1] != width:
        raise ValueError(f"frame_tokens must be (T,{width}) for C={C}, got {tuple(frame_tokens.shape)}")
    T, dev = frame_tokens.shape[0], frame_tokens.device
    if centroids.dim() == 3 and centroids.shape[0] != T:
        raise ValueError(f"per-frame centroids have {centroids.shape[0]} frames, the tokens {T}")
    if fraction.shape != (C,) and fraction.shape != (T, C):
        raise ValueError(f"fraction must be ({C},) or ({T},{C}), got {tuple(fraction.shape)}")
    mass = frame_tokens[:, lo:lo + C].to(F64)
    cen = centroids.to(device=dev, dtype=F64).expand(T, C, 3)
    frac = fraction.to(device=dev, dtype=F64).expand(T, C)
    return torch.cat([mass[..., None], cen, frac[..., None]], dim=-1)
