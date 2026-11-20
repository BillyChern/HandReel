"""R2: hand-aware silhouette carving of the template over the window.

Evidence per decode row k (pixels are image facts, so they use the STORED intrinsics whatever K the BA uses):
- object region: the pixels hit by the row's stored `geometry.npz` object points (the masked pointmap reprojects onto
  the integer pixel grid with the stored K: KA-MASK), closed with a disc of radius `close_px`;
- UNKNOWN: HaWoR hand silhouettes (stored vertices, both sides, valid rows) rasterised and dilated by a disc of radius
  `hand_dilate_px`, plus everything outside the image. Production masks exclude occluding hand pixels, so a voxel
  behind the hand must cast no vote: carving there would remove the grasp region.

Carving (only over-extension is removed; nothing is added): every template voxel (`sdf < 0` on the stored canonical
SDF grid, upsampled `voxel_upsample` times) is projected with the row's pose; a row that projects it in-image and not
UNKNOWN "sees" it and votes "outside" when the pixel is not in the object region. A voxel seen by >= `min_views` rows
with an outside fraction >= `vote_q` is carved. If more than `cap` of the template volume would go, the stage
ABSTAINS (template unchanged): gate the action, not just the evidence. A carved template is re-meshed by marching
cubes on ``max(sdf, d_carved)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from fastdo.tba.config import TbaConfig
from fastdo.tba.geometry import project

__all__ = ["CarveResult", "Evidence", "VoxelGrid", "carve_template", "disc", "hand_silhouette", "object_region",
           "remesh", "voxel_grid"]

logger = logging.getLogger(__name__)


class Evidence(Protocol):
    """Per-row image evidence (`fastdo.tba.window.FrameEvidence` reads it from stored artefacts)."""

    image_hw: tuple[int, int]

    def region(self, k: int) -> np.ndarray: ...       # (H,W) bool object region

    def unknown(self, k: int) -> np.ndarray: ...      # (H,W) bool UNKNOWN pixels (no vote)


def disc(radius: int) -> np.ndarray:
    """A (2r+1)^2 uint8 disc structuring element (r = 0: a single pixel)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def object_region(points: np.ndarray, K: np.ndarray, image_hw: tuple[int, int], close_px: int) -> np.ndarray:
    """(H,W) bool: pixels of camera-frame points (rounded projections), closed by a disc of radius `close_px`."""
    h, w = image_hw
    m = np.zeros((h, w), np.uint8)
    p = np.asarray(points, np.float64)
    p = p[np.isfinite(p).all(axis=1) & (p[:, 2] > 0)]
    if len(p):
        uv = np.rint(project(p, K)).astype(np.int64)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        m[uv[ok, 1], uv[ok, 0]] = 1
    if close_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disc(close_px))
    return m.astype(bool)


def hand_silhouette(vertices: list[np.ndarray], faces: np.ndarray, K: np.ndarray, image_hw: tuple[int, int],
                    dilate_px: int) -> np.ndarray:
    """(H,W) bool union of rasterised hand meshes (camera frame), dilated by a disc of radius `dilate_px`."""
    h, w = image_hw
    m = np.zeros((h, w), np.uint8)
    for v in vertices:
        v = np.asarray(v, np.float64)
        if not np.isfinite(v).all():
            continue
        tri = v[np.asarray(faces, np.int64)]                                   # (F,3,3)
        front = (tri[..., 2] > 1e-3).all(axis=1)
        uv = project(tri[front], K)
        pts = np.rint(uv * 16.0).astype(np.int32)                              # 4 fractional bits (cv2 shift)
        keep = (np.abs(pts) < 2 ** 26).all(axis=(1, 2))
        for t in pts[keep]:
            cv2.fillConvexPoly(m, t, 1, lineType=cv2.LINE_8, shift=4)
    if dilate_px > 0:
        m = cv2.dilate(m, disc(dilate_px))
    return m.astype(bool)


@dataclass(frozen=True)
class VoxelGrid:
    """A canonical SDF lattice (negative inside): ``values[ix, iy, iz]`` at ``origin + step * (ix, iy, iz)``."""

    values: np.ndarray
    origin: np.ndarray
    step: float

    def centres(self) -> np.ndarray:
        g = self.values.shape
        ax = [self.origin[d] + self.step * np.arange(g[d]) for d in range(3)]
        gx, gy, gz = np.meshgrid(*ax, indexing="ij")
        return np.stack([gx, gy, gz], axis=-1)


def voxel_grid(values: np.ndarray, origin: np.ndarray, step: float, upsample: int) -> VoxelGrid:
    """The stored SDF lattice, trilinearly upsampled `upsample` times over the same extent."""
    values = np.asarray(values, np.float64)
    if upsample == 1:
        return VoxelGrid(values, np.asarray(origin, np.float64), float(step))
    g = values.shape[0]
    n = (g - 1) * upsample + 1
    idx = np.arange(n) / upsample
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing="ij")
    fine = map_coordinates(values, np.stack([ii, jj, kk]), order=1, mode="nearest")
    return VoxelGrid(fine, np.asarray(origin, np.float64), float(step) / upsample)


@dataclass(frozen=True)
class CarveResult:
    """Carving outcome. `carved` (N_inside,) bool over `inside_idx` (flat lattice indices of template voxels)."""

    inside_idx: np.ndarray
    carved: np.ndarray
    n_seen: np.ndarray
    frac_out: np.ndarray
    removed_frac: float
    abstained: bool

    @property
    def applied(self) -> bool:
        return (not self.abstained) and bool(self.carved.any())


def carve_template(grid: VoxelGrid, poses: np.ndarray, scale: float, K: np.ndarray, evidence: Evidence,
                   cfg: TbaConfig, rows: np.ndarray | None = None) -> CarveResult:
    """Silhouette votes over the window's rows (module docstring). `rows` (T,) bool limits the voting rows."""
    flat = grid.values.reshape(-1)
    inside_idx = np.flatnonzero(flat < 0.0)
    pts = grid.centres().reshape(-1, 3)[inside_idx]
    h, w = evidence.image_hw
    n_seen = np.zeros(len(pts), np.int64)
    n_out = np.zeros(len(pts), np.int64)
    poses = np.asarray(poses, np.float64)
    use = np.isfinite(poses).all(axis=(1, 2))
    if rows is not None:
        use &= np.asarray(rows, bool)
    for k in np.flatnonzero(use):
        X = float(scale) * pts @ poses[k, :3, :3].T + poses[k, :3, 3]
        front = X[:, 2] > 1e-6
        uv = np.full((len(X), 2), -1, np.int64)
        uv[front] = np.rint(project(X[front], K)).astype(np.int64)
        inimg = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        unk = evidence.unknown(k)
        reg = evidence.region(k)
        sel = np.flatnonzero(inimg)
        sel = sel[~unk[uv[sel, 1], uv[sel, 0]]]
        n_seen[sel] += 1
        n_out[sel] += ~reg[uv[sel, 1], uv[sel, 0]]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_seen > 0, n_out / np.maximum(n_seen, 1), 0.0)
    carved = (n_seen >= cfg.min_views) & (frac >= cfg.vote_q)
    removed = float(carved.sum()) / max(len(pts), 1)
    abstained = removed > cfg.cap
    return CarveResult(inside_idx=inside_idx, carved=carved & (not abstained), n_seen=n_seen, frac_out=frac,
                       removed_frac=removed, abstained=abstained)


def remesh(grid: VoxelGrid, res: CarveResult) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes of ``max(sdf, d_carved)`` (canonical frame), outward-oriented. Requires `res.applied`."""
    import igl

    shape = grid.values.shape
    carved = np.zeros(int(np.prod(shape)), bool)
    carved[res.inside_idx[res.carved]] = True
    carved = carved.reshape(shape)
    d_in = distance_transform_edt(carved)
    d_out = distance_transform_edt(~carved)
    d_c = np.where(carved, d_in - 0.5, -(d_out - 0.5)) * grid.step
    field = np.maximum(grid.values, d_c)
    # igl indexes S(x + y nx + z nx ny): x fastest (Fortran order of the [ix, iy, iz] array)
    S = np.asarray(field, np.float64).reshape(-1, order="F")
    GV = np.ascontiguousarray(grid.centres().reshape(-1, 3, order="F"))
    V, F, _ = igl.marching_cubes(S, GV, shape[0], shape[1], shape[2], 0.0)
    V, F = np.asarray(V, np.float64), np.asarray(F, np.int64)
    if len(F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        if float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) < 0.0:   # signed volume * 6
            F = F[:, ::-1].copy()
    return V, F
