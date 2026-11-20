"""Occlusion-aware silhouette and hand-contact residual terms with analytic Jacobians."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.geometry.lie import hat
from fastdo.solver._bspline import bspline_prefilter_2d, bspline_sample_with_grad
from fastdo.solver.sdf import SdfGrid, sdf_lookup
from fastdo.solver.terms import (
    TermOutput,
    point_param_jacobian,
    project,
    projection_jacobian,
    transform_points,
)


def distance_transform(mask: Tensor) -> Tensor:
    """Pixel distance to the nearest True pixel of `mask` (H,W) bool; 0 inside the mask."""
    from scipy.ndimage import distance_transform_edt

    m = mask.cpu().numpy().astype(bool)
    if not m.any():
        return torch.full(mask.shape, float(max(mask.shape)), dtype=torch.float32)
    return torch.as_tensor(distance_transform_edt(~m), dtype=torch.float32)


def boundary_pixels(mask: Tensor, exclude: Tensor) -> Tensor:
    """(u,v) pixel coordinates of the `mask` boundary that are not within 2 px of `exclude`."""
    m = mask.float()[None, None]
    eroded = -F.max_pool2d(-m, 3, stride=1, padding=1)
    border = (m - eroded)[0, 0] > 0
    near = F.max_pool2d(exclude.float()[None, None], 5, stride=1, padding=2)[0, 0] > 0
    vs, us = torch.nonzero(border & ~near, as_tuple=True)
    return torch.stack([us, vs], dim=1).to(torch.float32)


def pad_boundaries(boundaries: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Stack per-frame boundary pixel lists into a (T,Bmax,2) tensor plus a (T,Bmax) validity mask."""
    bmax = max(int(b.shape[0]) for b in boundaries)
    out = torch.zeros(len(boundaries), bmax, 2, dtype=torch.float32)
    valid = torch.zeros(len(boundaries), bmax, dtype=torch.bool)
    for k, b in enumerate(boundaries):
        out[k, : b.shape[0]] = b
        valid[k, : b.shape[0]] = True
    return out, valid


def _sample(img: Tensor, uv: Tensor) -> Tensor:
    """img (T,C,H,W), uv (T,n,2) pixel coords -> (T,n,C), bilinear, border-clamped."""
    H, W = img.shape[-2:]
    gx = uv[..., 0] / (W - 1) * 2.0 - 1.0
    gy = uv[..., 1] / (H - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)[:, None].to(img.dtype)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0].transpose(1, 2)


@dataclass(frozen=True)
class DtSplines:
    """Cached cubic B-spline coefficients of one or more (H,W) distance-transform maps.

    Produced by `prefilter_distance_transforms`; pass the result to `silhouette_term` instead of a
    raw `dist_transforms` tensor to reuse the prefilter's O(H+W) tridiagonal solve (see
    `fastdo.solver._bspline`) across many calls against the same masks -- e.g. repeated
    Gauss-Newton iterations of the trajectory solver against a fixed set of per-frame masks.
    """

    coef: Tensor  # (T,1,H,W) float64 prefiltered coefficients


def prefilter_distance_transforms(dist_transforms: Tensor) -> DtSplines:
    """Prefilter (T,H,W) distance-transform maps once into cacheable cubic B-spline coefficients."""
    return DtSplines(coef=bspline_prefilter_2d(dist_transforms.to(torch.float64)[:, None]))


def _sample_with_grad(dt_splines: DtSplines, uv: Tensor) -> tuple[Tensor, Tensor]:
    """Prefiltered dt_splines.coef (T,1,H,W) sampled at uv (T,n,2) -> (value (T,n), grad (T,n,2)).

    Uses an exact (interpolating) cubic B-spline -- see `fastdo.solver._bspline` -- rather than
    `grid_sample`'s "bilinear" (C0: a real derivative jump at every integer pixel boundary) or
    "bicubic" (cubic convolution: only C1, its *second* derivative still jumps at pixel boundaries).
    A projected point landing a small fraction of a pixel from such a boundary makes a
    finite-difference check straddle it, and either discontinuity is enough on its own to disagree
    with the analytic Jacobian there by an O(1) amount for this term's sensitivity (K's focal
    length propagated through `projection_jacobian`). The B-spline is C2, which removes that
    failure mode; its analytic gradient matches a finite-difference check everywhere, not just
    mid-cell. Trade-off: a cubic spline is not shape-preserving, so it can slightly overshoot or
    undershoot the raw per-pixel distance transform near a sharp feature such as the mask boundary
    (where the true field's slope has a real jump) -- see `silhouette_term`.
    """
    return bspline_sample_with_grad(dt_splines.coef.to(uv.dtype), uv)


def silhouette_term(surface_points: Tensor, surface_normals: Tensor, dist_transforms: Tensor | DtSplines,
                    boundaries: Tensor, boundary_valid: Tensor, hand_masks: Tensor, poses: Tensor,
                    log_scale: Tensor, K: Tensor, lam_out: float, lam_cov: float) -> TermOutput:
    """Occlusion-aware silhouette term: out-of-mask penalty for visible surface points plus a
    coverage term pulling each observed silhouette boundary pixel toward the nearest visible
    projected surface point. Residual rows: [M out-of-mask rows | Bmax coverage rows], d = 1.

    `dist_transforms` accepts either a raw (T,H,W) tensor (prefiltered internally on every call) or
    the `DtSplines` returned by `prefilter_distance_transforms`, to reuse that prefilter's work
    across repeated calls against the same masks. Internally, the out-of-mask residual is a cubic
    B-spline resampling of the distance transform (see `_sample_with_grad`), which is not
    shape-preserving: it can slightly overshoot or undershoot the raw per-pixel distance transform
    a cell or two inside the mask boundary, where the true field's slope has a real jump. This
    trades a small amount of value fidelity there for a gradient that is C2 (matches a
    finite-difference check everywhere, not just away from pixel grid lines).
    """
    dtype = poses.dtype
    s = log_scale.exp()
    p = transform_points(surface_points.to(dtype), poses, s)                       # (T,M,3)
    n_cam = torch.einsum("tij,mj->tmi", poses[:, :3, :3], surface_normals.to(dtype))
    uv = project(p, K)
    dt_splines = dist_transforms if isinstance(dist_transforms, DtSplines) else \
        prefilter_distance_transforms(dist_transforms)
    H, W = dt_splines.coef.shape[-2:]
    inside = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
    hand = _sample(hand_masks.to(dtype)[:, None], uv)[..., 0] > 0.5
    vis = ((n_cam * p).sum(-1) < 0) & inside & ~hand                                # (T,M)
    dt, g = _sample_with_grad(dt_splines, uv)                                        # (T,M), (T,M,2)
    J_uv = projection_jacobian(p, K) @ point_param_jacobian(p, poses[:, :3, 3])     # (T,M,2,7)
    root_out = lam_out ** 0.5
    r_out = root_out * dt[..., None]
    J_out = root_out * (g[..., None, :] @ J_uv)                                      # (T,M,1,7)
    w_out = vis.to(dtype)
    # coverage: each boundary pixel is pulled to the nearest visible projected point
    b = boundaries.to(dtype)
    d = torch.cdist(b, uv).masked_fill(~vis[:, None, :], float("inf"))
    dmin, j = d.min(dim=2)                                                           # (T,B)
    finite = torch.isfinite(dmin)
    j = j.masked_fill(~finite, 0)
    uv_near = torch.gather(uv, 1, j[..., None].expand(-1, -1, 2))
    J_near = torch.gather(J_uv, 1, j[..., None, None].expand(-1, -1, 2, 7))
    dist = dmin.masked_fill(~finite, 1.0).clamp_min(1e-9)
    root_cov = lam_cov ** 0.5
    r_cov = root_cov * dist[..., None]
    J_cov = root_cov * (((uv_near - b) / dist[..., None])[..., None, :] @ J_near)
    w_cov = boundary_valid.to(dtype) * finite.to(dtype)
    r_cov = r_cov * finite[..., None]
    J_cov = J_cov * finite[..., None, None]
    return TermOutput(residual=torch.cat([r_out, r_cov], 1), jacobian=torch.cat([J_out, J_cov], 1),
                      weight=torch.cat([w_out, w_cov], 1))


def contact_term(sdf: SdfGrid, hand_verts: Tensor, fingertip_idx: Tensor, in_hand: Tensor, poses: Tensor,
                 log_scale: Tensor, lam_pen: float, lam_con: float, delta: float = 0.005) -> TermOutput:
    """Hand-contact term against a canonical-frame SDF: penetration penalty for all hand vertices
    plus a contact-encouraging penalty for fingertip vertices while a hand-object contact is
    flagged (`in_hand`). Residual rows: [V penetration rows | F contact rows], d = 1.
    """
    dtype = poses.dtype
    s = log_scale.exp()
    R, t = poses[:, :3, :3], poses[:, :3, 3]
    h = hand_verts.to(dtype)
    V = h.shape[1]
    y = torch.einsum("tji,tvj->tvi", R, h - t[:, None, :]) / s                     # R^T (h - t) / s
    phi, grad = sdf_lookup(sdf, y)                                                   # (T,V), (T,V,3)
    Rt = R.transpose(1, 2)[:, None]                                                  # (T,1,3,3)
    d_rho = (-Rt / s).expand(-1, V, -1, -1)
    d_theta = (Rt @ hat(h)) / s
    dy = torch.cat([d_rho, d_theta, -y[..., None]], dim=-1)                          # (T,V,3,7)
    dphi = grad[..., None, :] @ dy                                                   # (T,V,1,7)
    root_pen = lam_pen ** 0.5
    pen = (phi < 0)
    r_pen = root_pen * (-phi).clamp_min(0.0)[..., None]
    J_pen = -root_pen * dphi * pen[..., None, None]
    w_pen = torch.ones_like(phi)
    root_con = lam_con ** 0.5
    phi_f, dphi_f = phi[:, fingertip_idx], dphi[:, fingertip_idx]
    gate = in_hand.to(dtype)[:, None]
    con = (phi_f > delta) & in_hand[:, None]
    r_con = root_con * (phi_f - delta).clamp_min(0.0)[..., None] * gate[..., None]
    J_con = root_con * dphi_f * con[..., None, None]
    w_con = gate.expand(-1, phi_f.shape[1])
    return TermOutput(residual=torch.cat([r_pen, r_con], 1), jacobian=torch.cat([J_pen, J_con], 1),
                      weight=torch.cat([w_pen, w_con], 1))
