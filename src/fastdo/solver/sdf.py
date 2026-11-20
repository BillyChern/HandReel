"""Signed-distance grids of canonical meshes with trilinear lookup and gradients.

Backend: `libigl` (pseudonormal signed distance, prebuilt wheel available for this platform;
no source compilation was required). Sign convention: negative inside the mesh, positive
outside, matching libigl's `signed_distance` output directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SdfGrid:
    values: Tensor   # (G,G,G) indexed [ix, iy, iz]
    origin: Tensor   # (3,) world coordinate of index (0,0,0)
    step: float      # cubic cell size

    @property
    def resolution(self) -> int:
        return int(self.values.shape[0])


def mesh_sdf_grid(vertices: np.ndarray, faces: np.ndarray, resolution: int = 64, padding: float = 0.15) -> SdfGrid:
    """Build a cubic signed-distance grid around `vertices`/`faces` using libigl's pseudonormal SDF."""
    import igl

    lo, hi = vertices.min(0), vertices.max(0)
    extent = float((hi - lo).max())
    centre = (lo + hi) / 2.0
    # Margin is applied on *both* sides of the half-extent so `padding` controls the fraction of
    # the object's own size available beyond its surface for off-mesh queries (e.g. a hand vertex
    # sitting a few centimetres away from the object); half-extent alone leaves too little room.
    half = extent * (0.5 + 2.0 * padding)
    step = 2.0 * half / (resolution - 1)
    # Anchor the grid so the mesh centroid lands exactly on a lattice sample (index k), regardless
    # of resolution parity. The exact SDF of a compact mesh is non-smooth at interior medial points
    # near the centroid (a distance function's gradient is only piecewise defined there), so a
    # centroid that falls *between* samples biases trilinear interpolation; snapping it to a sample
    # removes that bias while leaving the requested half-extent and cell size unchanged.
    k = (resolution - 1) // 2
    origin = centre - k * step
    ax = origin[None, :] + step * np.arange(resolution)[:, None]                   # (G,3) per-axis coords
    gx, gy, gz = np.meshgrid(ax[:, 0], ax[:, 1], ax[:, 2], indexing="ij")
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float64)
    sd, _, _, _ = igl.signed_distance(pts, vertices.astype(np.float64), faces.astype(np.int64),
                                      sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
    values = torch.as_tensor(np.asarray(sd).reshape(resolution, resolution, resolution), dtype=torch.float64)
    return SdfGrid(values=values, origin=torch.as_tensor(origin, dtype=torch.float64), step=float(step))


def sdf_lookup(sdf: SdfGrid, y: Tensor) -> tuple[Tensor, Tensor]:
    """Trilinear phi(y) and grad phi(y) for y (T,n,3); coordinates outside the grid are clamped.

    `grad` is the *exact* derivative of the trilinear interpolant `grid_sample` evaluates, obtained
    by differentiating through the same lookup (via autograd) rather than by separately trilinearly
    interpolating a precomputed central-difference field. A precomputed node-centred central
    difference is the average of the two cell slopes on either side of a grid node, which differs
    from the true (single-cell) slope of the trilinear interpolant away from the nodes themselves --
    exactly the field `contact_term`/`silhouette_term` differentiate through, so consistency here is
    what the analytic-vs-finite-difference Jacobian tests rely on.
    """
    T, n, _ = y.shape
    vals = sdf.values.to(y.dtype)
    G = sdf.resolution
    chans = vals.permute(2, 1, 0)[None, None].expand(T, 1, -1, -1, -1)             # (T,1,D=z,H=y,W=x)
    y_local = y.detach().clone().requires_grad_(True)
    norm = (y_local - sdf.origin.to(y.dtype)) / (sdf.step * (G - 1)) * 2.0 - 1.0    # (T,n,3) in [-1,1]
    grid = norm.clamp(-1.0, 1.0).view(T, 1, 1, n, 3)                                # (x,y,z) order
    with torch.enable_grad():
        out = F.grid_sample(chans, grid, mode="bilinear", padding_mode="border", align_corners=True)
        phi = out[:, 0, 0, 0, :]                                                    # (T,n)
        # `grad` is detached below, so no outer graph ever needs create_graph=True here.
        (grad,) = torch.autograd.grad(phi.sum(), y_local)
    return phi.detach(), grad.detach()


def sdf_values(sdf: SdfGrid, y: Tensor) -> Tensor:
    """Trilinear phi(y) for y (T,n,3), without the gradient.

    The same normalization, border clamp and `grid_sample` call as `sdf_lookup`, under `torch.no_grad()`: bitwise
    equal to `sdf_lookup(sdf, y)[0]`, without building or differentiating an autograd graph.
    """
    T, n, _ = y.shape
    vals = sdf.values.to(y.dtype)
    G = sdf.resolution
    chans = vals.permute(2, 1, 0)[None, None].expand(T, 1, -1, -1, -1)
    with torch.no_grad():
        norm = (y - sdf.origin.to(y.dtype)) / (sdf.step * (G - 1)) * 2.0 - 1.0
        grid = norm.clamp(-1.0, 1.0).view(T, 1, 1, n, 3)
        out = F.grid_sample(chans, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[:, 0, 0, 0, :]
