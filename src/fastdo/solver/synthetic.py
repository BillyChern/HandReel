"""Synthetic hand-occluded object clips with exact ground truth for solver tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from torch import Tensor

from fastdo.geometry.lie import se3_exp


@dataclass(frozen=True)
class SyntheticConfig:
    n_frames: int = 60
    image_hw: tuple[int, int] = (240, 320)
    focal: float = 300.0
    shape: str = "torus"
    n_surface: int = 2048
    n_tracks: int = 64
    occlusion: float = 0.0        # fraction of the image width covered by the occluder disc (0 = none)
    depth_noise: float = 0.002    # metres, Gaussian on pointmap z
    motion_scale: float = 0.02    # per-frame se(3) step std (m, rad)
    distance: float = 0.6         # initial object distance along z (m)
    scale: float = 0.12           # metric size multiplier of the unit-normalized mesh


@dataclass(frozen=True)
class SyntheticClip:
    mesh_vertices: Tensor
    mesh_faces: Tensor
    surface_points: Tensor
    surface_normals: Tensor
    poses_gt: Tensor
    scale_gt: float
    intrinsics: Tensor
    image_hw: tuple[int, int]
    masks: Tensor
    hand_masks: Tensor
    pointmaps: Tensor
    tracks: Tensor
    track_vis: Tensor
    anchors: Tensor


def _unit_mesh(shape: str) -> trimesh.Trimesh:
    if shape == "torus":
        mesh = trimesh.creation.torus(major_radius=1.0, minor_radius=0.35, major_sections=64, minor_sections=32)
    elif shape == "box":
        mesh = trimesh.creation.box(extents=(1.0, 0.6, 0.4))
    else:
        raise ValueError(f"unknown shape {shape}")
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1.0 / mesh.bounding_box.extents.max())
    return mesh


def _smooth_trajectory(n: int, step: float, distance: float, gen: torch.Generator, focal: float,
                       image_hw: tuple[int, int]) -> Tensor:
    """Random-walk SE(3) trajectory, low-passed, with a frustum-bounce guard.

    Without a bound, the random walk can drift the object centre out of the camera's view for a run of
    frames (confirmed for n_frames=100, seed=3, box: fully outside a 320x240 image by frame 70) or
    behind/too-far-from the camera; a `solve_trajectory` fit then has zero pointmap/track observations
    for those frames and can only extrapolate via the smoothness prior, which is a test-fixture defect
    (an information limit unrelated to solver correctness), not a solver bug (a diagnostic of the
    solver tests showed it). While integrating the low-passed increments, if the candidate next centre would
    project outside the central 80% of the image or leave the [0.35, 1.0] m depth band, the remaining
    increments' translational part `xi[k:, :3]` is negated (a "bounce" back toward the frustum) and the
    step is re-taken. This is a no-op for any clip that already stays inside those bounds (verified by
    `test_frustum_guard_is_noop_for_bounded_default_clip`), so existing fixtures/tolerances are
    unaffected unless they actually left the frustum.
    """
    xi = torch.randn(n, 6, generator=gen, dtype=torch.float64) * step
    xi[:, 3:] *= 2.0                                  # rotations a bit faster than translations
    kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    for _ in range(3):                                # low-pass the increments
        xi = torch.nn.functional.conv1d(xi.T.unsqueeze(0), kernel.view(1, 1, 3).expand(6, 1, 3),
                                        padding=1, groups=6).squeeze(0).T
    H, W = image_hw
    u_lo, u_hi = 0.1 * W, 0.9 * W
    v_lo, v_hi = 0.1 * H, 0.9 * H
    T = torch.eye(4, dtype=torch.float64)
    T[2, 3] = distance
    poses = [T]
    for k in range(1, n):
        cand = se3_exp(xi[k]) @ poses[-1]
        centre = cand[:3, 3]
        z = float(centre[2])
        if z > 1e-9:
            u = focal * float(centre[0]) / z + W / 2
            v = focal * float(centre[1]) / z + H / 2
        else:
            u = v = float("inf")
        in_frustum = (u_lo <= u <= u_hi) and (v_lo <= v <= v_hi) and (0.35 <= z <= 1.0)
        if not in_frustum:
            xi[k:, :3] = -xi[k:, :3]                  # bounce: negate the remaining translational steps
            cand = se3_exp(xi[k]) @ poses[-1]
        poses.append(cand)
    return torch.stack(poses).float()


def _project(p: Tensor, K: Tensor) -> Tensor:
    z = p[..., 2].clamp_min(1e-6)
    return torch.stack([K[0, 0] * p[..., 0] / z + K[0, 2], K[1, 1] * p[..., 1] / z + K[1, 2]], dim=-1)


def _splat_depth(p: Tensor, K: Tensor, hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Z-buffer point splat: returns (pointmap (H,W,3) with NaN holes, mask (H,W))."""
    H, W = hw
    uv = _project(p, K)
    u = uv[:, 0].round().long()
    v = uv[:, 1].round().long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (p[:, 2] > 0)
    u, v, p = u[ok], v[ok], p[ok]
    lin = v * W + u
    depth = torch.full((H * W,), float("inf"), dtype=p.dtype)
    depth.scatter_reduce_(0, lin, p[:, 2], reduce="amin")
    best = depth[lin] == p[:, 2]
    pointmap = torch.full((H * W, 3), float("nan"), dtype=p.dtype)
    pointmap[lin[best]] = p[best]
    mask = torch.isfinite(depth).view(H, W)
    return pointmap.view(H, W, 3), mask


def _occluder(k: int, n: int, hw: tuple[int, int], frac: float) -> Tensor:
    H, W = hw
    if frac <= 0.0:
        return torch.zeros(H, W, dtype=torch.bool)
    radius = 0.5 * frac * W
    cx = W * (0.25 + 0.5 * k / max(n - 1, 1))
    cy = H * 0.5
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return ((xx - cx) ** 2 + (yy - cy) ** 2) < radius**2


def make_synthetic_clip(cfg: SyntheticConfig, seed: int) -> SyntheticClip:
    gen = torch.Generator().manual_seed(seed)
    mesh = _unit_mesh(cfg.shape)
    dense, face_idx = trimesh.sample.sample_surface(mesh, 20000, seed=seed)
    dense_t = torch.as_tensor(np.asarray(dense), dtype=torch.float32)
    sparse_idx = torch.randperm(dense_t.shape[0], generator=gen)[: cfg.n_surface]
    surface = dense_t[sparse_idx]
    normals = torch.as_tensor(np.asarray(mesh.face_normals)[np.asarray(face_idx)], dtype=torch.float32)[sparse_idx]
    anchors = dense_t[torch.randperm(dense_t.shape[0], generator=gen)[: cfg.n_tracks]]
    H, W = cfg.image_hw
    K = torch.tensor([[cfg.focal, 0.0, W / 2], [0.0, cfg.focal, H / 2], [0.0, 0.0, 1.0]])
    poses = _smooth_trajectory(cfg.n_frames, cfg.motion_scale, cfg.distance, gen, cfg.focal, cfg.image_hw)
    masks, hands, pmaps, tracks, vis = [], [], [], [], []
    for k in range(cfg.n_frames):
        T = poses[k]
        p_dense = cfg.scale * dense_t @ T[:3, :3].T + T[:3, 3]
        pmap, mask = _splat_depth(p_dense, K, cfg.image_hw)
        occ = _occluder(k, cfg.n_frames, cfg.image_hw, cfg.occlusion)
        mask = mask & ~occ
        pmap[~mask] = float("nan")
        if cfg.depth_noise > 0:
            noise = torch.randn(H, W, generator=gen) * cfg.depth_noise
            pmap[..., 2] = pmap[..., 2] + noise
            pmap[..., 0] = (torch.arange(W).float()[None, :] - K[0, 2]) / K[0, 0] * pmap[..., 2]
            pmap[..., 1] = (torch.arange(H).float()[:, None] - K[1, 2]) / K[1, 1] * pmap[..., 2]
        p_anchor = cfg.scale * anchors @ T[:3, :3].T + T[:3, 3]
        uv = _project(p_anchor, K)
        u = uv[:, 0].round().long().clamp(0, W - 1)
        v = uv[:, 1].round().long().clamp(0, H - 1)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        front = torch.isfinite(pmap[v, u, 2]) & ((pmap[v, u, 2] - p_anchor[:, 2]).abs() < 0.01)
        masks.append(mask)
        hands.append(occ)
        pmaps.append(pmap)
        tracks.append(uv)
        vis.append(inside & front & ~occ[v, u])
    return SyntheticClip(
        mesh_vertices=torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32),
        mesh_faces=torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int64),
        surface_points=surface,
        surface_normals=normals,
        poses_gt=poses,
        scale_gt=cfg.scale,
        intrinsics=K,
        image_hw=cfg.image_hw,
        masks=torch.stack(masks),
        hand_masks=torch.stack(hands),
        pointmaps=torch.stack(pmaps),
        tracks=torch.stack(tracks),
        track_vis=torch.stack(vis),
        anchors=anchors,
    )
