"""Tests for `visible_samples`, the visibility-aware template measure (design 11.9)."""

from __future__ import annotations

import math

import numpy as np
import torch
import trimesh
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver.synthetic import _unit_mesh
from fastdo.solver.visibility import (
    dilate_hand_masks,
    exclude_hand_occluded,
    foreshortening_weights,
    hand_occluded,
    visible_samples,
)

DT = torch.float64


def _K(focal: float = 1200.0, hw: tuple[int, int] = (960, 1280)) -> Tensor:
    H, W = hw
    return torch.tensor([[focal, 0.0, W / 2], [0.0, focal, H / 2], [0.0, 0.0, 1.0]], dtype=DT)


def _sample(mesh: trimesh.Trimesh, n: int, seed: int) -> tuple[Tensor, Tensor]:
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = np.asarray(mesh.face_normals)[np.asarray(face_idx)]
    return (torch.as_tensor(np.asarray(pts), dtype=DT), torch.as_tensor(normals, dtype=DT))


def test_box_corner_view_half_visible_no_backface_all_in_image() -> None:
    """A box seen from a generic (non-degenerate) corner view: exactly one face of each +/-axis pair
    is front-facing, so front-facing surface area (and, for a large enough sample, count) is ~half the
    total regardless of the box's aspect ratio; the z-buffer must not additionally reject any of a
    convex shape's front-facing samples (a box has no self-occlusion)."""
    mesh = _unit_mesh("box")
    points, normals = _sample(mesh, 4000, seed=0)
    scale = 0.12
    # A generic rotation (not axis-aligned) so no face normal is exactly edge-on to the camera.
    theta = torch.tensor([0.55, 0.35, 0.0], dtype=DT)
    R = se3_exp(torch.cat([torch.zeros(3, dtype=DT), theta]))[:3, :3]
    pose = torch.eye(4, dtype=DT)
    pose[:3, :3] = R
    pose[2, 3] = 0.6
    poses = pose.unsqueeze(0)
    log_scale = torch.tensor(np.log(scale), dtype=DT)
    K = _K()

    vis = visible_samples(points, normals, poses, log_scale, K, image_hw=(960, 1280))
    assert vis.shape == (1, points.shape[0])
    frac = float(vis[0].float().mean())
    assert 0.40 <= frac <= 0.60, frac

    # No back-facing sample (by the dot test) is ever marked visible.
    s = float(log_scale.exp())
    R0 = poses[0, :3, :3]
    p_cam = s * (points @ R0.T) + poses[0, :3, 3]
    n_cam = normals @ R0.T
    front = (n_cam * (-p_cam)).sum(-1) > 0
    assert bool((vis[0] & ~front).sum() == 0)

    # Every visible sample projects inside the image.
    u = K[0, 0] * p_cam[:, 0] / p_cam[:, 2] + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / p_cam[:, 2] + K[1, 2]
    inside = (u >= 0) & (u < 1280) & (v >= 0) & (v < 960)
    assert bool((vis[0] & ~inside).sum() == 0)


def test_torus_far_inner_wall_rejected_by_zbuffer_vs_bruteforce_raytest() -> None:
    """The torus's far inner wall (facing the camera by the dot test, but hidden behind the near ring)
    must be rejected by the z-buffer occlusion test; cross-check against a brute-force ray intersection
    against the actual mesh (transformed into the camera frame).

    Viewed straight down the hole axis (identity rotation) the hole is empty space, so both inner walls
    are simultaneously visible with no real self-occlusion (confirmed empirically: zero front-facing
    samples rejected at full z-buffer resolution regardless of sample count). Tilting 90 degrees to an
    edge-on view puts genuine tube-on-tube self-occlusion in frame.

    A later change set the production z-buffer default to full resolution (`downsample=1`, no per-pixel
    binning) to stop it from rejecting genuinely-visible grazing-angle samples on a *box* (see
    `SolverConfig`'s docstring). For a smoothly curved, self-occluding shape like this torus, a
    per-sample z-buffer can only detect that sample A occludes sample B when some *other* sample lands
    at (very nearly) A's own projected pixel - so with only a few thousand samples spread over a large
    projected footprint at full resolution, most occlusion goes undetected purely from under-sampling
    (empirically confirmed: at the 3000-sample count used elsewhere in this file, only ~5% of clearly
    (>2cm) occluded ground-truth samples are rejected at `downsample=1`). This is a sample-density
    property of a discrete z-buffer, not a `visible_samples` bug: at high enough sample density (dense
    enough that two samples plausibly land in the same full-resolution pixel), the production default
    correctly rejects the great majority of clearly-occluded samples while keeping the great majority
    of clearly-unoccluded ones - verified here at 30k samples (`n_mesh` in real solves is 1-2k, but this
    test is about validating the z-buffer *mechanism*, not exercising production sample counts)."""
    torch.manual_seed(0)
    mesh = _unit_mesh("torus")
    points, normals = _sample(mesh, 30_000, seed=1)
    scale = 0.12
    theta = torch.tensor([math.radians(90.0), 0.0, 0.0], dtype=DT)
    R = se3_exp(torch.cat([torch.zeros(3, dtype=DT), theta]))[:3, :3]
    pose = torch.eye(4, dtype=DT)
    pose[:3, :3] = R
    pose[2, 3] = 0.6                      # edge-on view: real tube-on-tube self-occlusion in frame
    poses = pose.unsqueeze(0)
    log_scale = torch.tensor(np.log(scale), dtype=DT)
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=DT)
    image_hw = (480, 640)

    vis = visible_samples(points, normals, poses, log_scale, K, image_hw)[0]   # downsample=1 (default)

    s = float(log_scale.exp())
    R0 = poses[0, :3, :3]
    t0 = poses[0, :3, 3]
    p_cam = s * (points @ R0.T) + t0
    n_cam = normals @ R0.T
    front = (n_cam * (-p_cam)).sum(-1) > 0

    # Mesh transformed into the camera frame for ray-casting ground truth.
    verts_cam = (s * (torch.as_tensor(np.asarray(mesh.vertices), dtype=DT) @ R0.T) + t0).numpy()
    cam_mesh = trimesh.Trimesh(vertices=verts_cam, faces=np.asarray(mesh.faces), process=False)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(cam_mesh)

    gen = torch.Generator().manual_seed(3)
    idx = torch.randperm(points.shape[0], generator=gen)[:200]
    p_np = p_cam[idx].numpy()
    dist = np.linalg.norm(p_np, axis=1)
    origins = np.zeros_like(p_np)
    directions = p_np / dist[:, None]
    locations, index_ray, _index_tri = intersector.intersects_location(origins, directions,
                                                                        multiple_hits=True)
    first_hit = np.full(p_np.shape[0], np.inf)
    if locations.shape[0] > 0:
        hit_dist = np.linalg.norm(locations, axis=1)
        for r, d in zip(index_ray, hit_dist, strict=True):
            first_hit[r] = min(first_hit[r], d)
    gap_mm = (dist - first_hit) * 1000.0
    front_np = front[idx].numpy()
    vis_np = vis[idx].numpy()
    # "Clear" cases only (>2cm gap = unambiguously occluded; <2mm gap = unambiguously not occluded):
    # avoids the borderline/grazing samples right at the visibility boundary, where any discretization
    # (ours or the ray test's own numerical tangency) disagrees with the other some of the time.
    clear_occluded = front_np & (gap_mm > 20.0)
    clear_visible = front_np & (gap_mm < 2.0)
    assert int(clear_occluded.sum()) > 15, "test scene should exercise clear self-occlusion"
    assert int(clear_visible.sum()) > 15, "test scene should exercise clear non-occlusion"
    assert float((~vis_np[clear_occluded]).mean()) > 0.8, "clearly-occluded samples must mostly be rejected"
    assert float(vis_np[clear_visible].mean()) > 0.7, "clearly-visible samples must mostly stay visible"

    # And there must be at least one sample that IS front-facing but IS rejected by the z-buffer
    # (the far inner wall), i.e. this test scene actually exercises self-occlusion.
    assert bool((front & ~vis).any())


def test_degenerate_frame_falls_back_to_full_template() -> None:
    """A frame where the object is entirely out of view (e.g. behind the camera) has fewer than 32
    visible samples; `visible_samples` must fall back to the full template (all True) for that frame."""
    mesh = _unit_mesh("box")
    points, normals = _sample(mesh, 256, seed=2)
    pose_ok = torch.eye(4, dtype=DT)
    pose_ok[2, 3] = 0.6
    pose_behind = torch.eye(4, dtype=DT)
    pose_behind[2, 3] = -0.6              # object behind the camera: nothing projects in front
    poses = torch.stack([pose_ok, pose_behind])
    log_scale = torch.tensor(np.log(0.12), dtype=DT)
    K = _K()

    vis = visible_samples(points, normals, poses, log_scale, K, image_hw=(960, 1280))
    assert bool(vis[1].all())
    assert not bool(vis[0].all())


def test_delta_widens_facing_set_but_strict_delta0_matches_old_behaviour() -> None:
    """`delta` widens the front-facing test to `n_j . v_j > -delta`. Grazing/slightly
    back-facing samples (cos in (-0.5, 0)) must be excluded at delta=0 (the original strict test,
    unchanged) and at least some of them must become visible once delta is opened up to 0.5 (the
    coarse-level default) - this is the capture-range mechanism the solver relies on."""
    mesh = _unit_mesh("box")
    points, normals = _sample(mesh, 4000, seed=0)
    scale = 0.12
    theta = torch.tensor([0.55, 0.35, 0.0], dtype=DT)
    R = se3_exp(torch.cat([torch.zeros(3, dtype=DT), theta]))[:3, :3]
    pose = torch.eye(4, dtype=DT)
    pose[:3, :3] = R
    pose[2, 3] = 0.6
    poses = pose.unsqueeze(0)
    log_scale = torch.tensor(np.log(scale), dtype=DT)
    K = _K()

    s = float(log_scale.exp())
    p_cam = s * (points @ R.T) + poses[0, :3, 3]
    n_cam = normals @ R.T
    view_dir = torch.nn.functional.normalize(-p_cam, dim=-1)
    cos = (n_cam * view_dir).sum(-1)
    grazing_back = (cos > -0.5) & (cos < 0.0)
    assert int(grazing_back.sum()) > 20, "scene should contain some grazing/back-facing samples"

    vis_strict = visible_samples(points, normals, poses, log_scale, K, image_hw=(960, 1280), delta=0.0)[0]
    assert bool((vis_strict[grazing_back]).any()) is False

    vis_wide = visible_samples(points, normals, poses, log_scale, K, image_hw=(960, 1280), delta=0.5)[0]
    assert bool((vis_wide[grazing_back]).any())


def test_dilate_hand_masks_expands_square_block() -> None:
    """A single True pixel dilates to an exact `(2*dilate_px+1)^2` square block (max-pool structuring
    element), and `dilate_px=0` is a no-op."""
    mask = torch.zeros(1, 21, 21, dtype=torch.bool)
    mask[0, 10, 10] = True
    dilated = dilate_hand_masks(mask, dilate_px=3)
    assert dilated.dtype == torch.bool
    assert int(dilated[0].sum()) == 7 * 7
    assert bool(dilated[0, 10 - 3 : 10 + 4, 10 - 3 : 10 + 4].all())
    assert not bool(dilated[0, 6, 10])   # just outside the 7x7 block (row 10-4)

    noop = dilate_hand_masks(mask, dilate_px=0)
    assert torch.equal(noop, mask)


def test_exclude_hand_occluded_removes_projected_samples_and_falls_back_when_scarce() -> None:
    """Samples that project under the (dilated) hand mask are excluded from an all-visible input mask;
    a frame where the hand mask covers everything falls back to the pre-exclusion mask rather than
    emptying the visible set entirely."""
    K = _K(focal=500.0, hw=(480, 640))
    H, W = 480, 640
    poses = torch.eye(4, dtype=DT).unsqueeze(0)
    poses[0, 2, 3] = 0.6
    log_scale = torch.tensor(0.0, dtype=DT)
    # Two points, both at z=0.6 (pure-translation pose): x=-0.1 projects to pixel column 237 (deep in
    # the left half), x=+0.1 to column 403 (deep in the right half) - both comfortably clear of the
    # dilated mask boundary near column 320 (+/-3 px) so there is no rounding/dilation ambiguity.
    points = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=DT)
    mask = torch.ones(1, points.shape[0], dtype=torch.bool)

    hand_mask = torch.zeros(1, H, W, dtype=torch.bool)
    hand_mask[:, :, : W // 2] = True                   # left half only
    hand_dilated = dilate_hand_masks(hand_mask, dilate_px=3)

    occluded = hand_occluded(points, poses, log_scale, K, hand_dilated)[0]
    assert bool(occluded[0]) and not bool(occluded[1])   # left point occluded, right point not

    combined = exclude_hand_occluded(mask, points, poses, log_scale, K, hand_dilated, min_visible=1)
    assert torch.equal(combined[0], torch.tensor([False, True]))

    # Hand mask covering the WHOLE image would leave 0 visible samples; with min_visible=1 that is
    # degenerate (both points would be hand-occluded), so it must fall back to `mask` unchanged.
    hand_all = torch.ones(1, H, W, dtype=torch.bool)
    fallback = exclude_hand_occluded(mask, points, poses, log_scale, K, hand_all, min_visible=1)
    assert torch.equal(fallback[0], mask[0])


def test_foreshortening_weights_favour_face_on_over_grazing_and_zero_backfacing() -> None:
    """`max(n.v, 0) / z^2`: a face-on sample outweighs a grazing one at the same depth, and a strictly
    back-facing sample gets exactly zero weight."""
    poses = torch.eye(4, dtype=DT).unsqueeze(0)
    poses[0, 2, 3] = 0.6
    log_scale = torch.tensor(0.0, dtype=DT)
    # All three canonical points sit at camera-frame z = 0.6 (poses is a pure translation), i.e. the
    # same depth, so any weight difference is due to the normal direction alone.
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=DT)
    grazing_normal = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, -0.1], dtype=DT), dim=0)
    normals = torch.stack([
        torch.tensor([0.0, 0.0, -1.0], dtype=DT),            # face-on: points straight at the camera
        grazing_normal,                                       # grazing
        torch.tensor([0.0, 0.0, 1.0], dtype=DT),              # strictly back-facing
    ])

    w = foreshortening_weights(points, normals, poses, log_scale)[0]
    assert w[0] > w[1] > 0.0
    assert w[2] == 0.0
