"""Tests for `fastdo.solver.keyframe`: keyframe pose registration and mesh-frame track anchors."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import so3_exp
from fastdo.solver.api import TrajectoryObservations
from fastdo.solver.keyframe import anchors_from_pose, gate_init_poses, register_keyframe_pose
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip
from fastdo.solver.terms import transform_points


def _posed_nn_error(points: Tensor, pose_a: Tensor, pose_b: Tensor, scale: float) -> float:
    """Mean nearest-neighbour distance between two posings of the same samples (symmetry-aware)."""
    a = transform_points(points.double(), pose_a[None].double(), scale)[0]
    b = transform_points(points.double(), pose_b[None].double(), scale)[0]
    return float(torch.cdist(a, b).min(dim=1).values.mean())


def _axis_angle(axis: list[float], degrees: float) -> Tensor:
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm() * math.radians(degrees)
    return so3_exp(a).float()


@pytest.mark.parametrize("shape", ["box", "torus"])
def test_register_keyframe_pose_recovers_ground_truth(shape: str) -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=2, image_hw=(240, 320), shape=shape, n_surface=2048,
                                               depth_noise=0.001), seed=11)
    mask = clip.masks[0]
    observed = clip.pointmaps[0][mask & torch.isfinite(clip.pointmaps[0]).all(-1)]
    # Re-express the canonical shape in a frame rotated by an arbitrary 100 deg, so the true keyframe
    # rotation (identity in the fixture) is not trivially one of the registration's grid hypotheses.
    q = _axis_angle([0.3, -0.5, 0.8], 100.0)
    points, normals = clip.surface_points @ q, clip.surface_normals @ q       # rows: q^T y
    true_pose = clip.poses_gt[0].clone()
    true_pose[:3, :3] = true_pose[:3, :3] @ q

    reg = register_keyframe_pose(points, normals, observed, clip.intrinsics, mask, scale=clip.scale_gt,
                                 n_obs=256, n_mesh=512, n_iters=40, seed=0, device="cpu")

    assert reg.pose.shape == (4, 4)
    assert math.isfinite(reg.score) and reg.scale == clip.scale_gt        # rigid unless asked to fit scale
    # 0.12 m object, 1 mm depth noise: the registered posing must overlay the true one to a few mm.
    assert _posed_nn_error(points, reg.pose, true_pose, clip.scale_gt) < 0.004


def test_register_keyframe_pose_rejects_empty_observation() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=2, image_hw=(120, 160), n_surface=256), seed=1)
    with pytest.raises(ValueError, match="observed"):
        register_keyframe_pose(clip.surface_points, clip.surface_normals, torch.zeros(3, 3), clip.intrinsics,
                               clip.masks[0], scale=clip.scale_gt)


def test_anchors_from_pose_matches_ground_truth_anchors() -> None:
    # Frame 5, not frame 0: the fixture's frame-0 pose is the identity, which views the box's side faces
    # exactly edge-on, where one pixel spans centimetres of depth and no anchor rule is well defined.
    clip = make_synthetic_clip(SyntheticConfig(n_frames=6, image_hw=(240, 320), shape="box", n_surface=4096,
                                               n_tracks=64, depth_noise=0.0), seed=11)
    k = 5
    anchors, valid = anchors_from_pose(clip.surface_points, clip.surface_normals, clip.poses_gt[k],
                                       clip.scale_gt, clip.intrinsics, clip.image_hw, clip.tracks[k])

    assert anchors.shape == (64, 3) and valid.shape == (64,)
    visible = clip.track_vis[k]
    ok = valid & visible
    assert int(ok.sum()) >= int(0.8 * visible.sum())
    err = (anchors[ok] - clip.anchors[ok]).norm(dim=1)
    assert float(err.median()) < 0.03                  # unit-scale shape: within a sample spacing
    # The anchor lies on the surface the camera sees at that pixel: posed depth matches the pointmap.
    cam = transform_points(anchors.double(), clip.poses_gt[k][None].double(), clip.scale_gt)[0]
    h, w = clip.image_hw
    u = clip.tracks[k][:, 0].round().long().clamp(0, w - 1)
    v = clip.tracks[k][:, 1].round().long().clamp(0, h - 1)
    dz = (cam[:, 2] - clip.pointmaps[k][v, u, 2].double()).abs()
    assert float((dz[ok] < 0.01).float().mean()) >= 0.8


def test_anchors_from_pose_flags_pixels_off_the_object() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=2, image_hw=(240, 320), n_surface=2048), seed=7)
    far_corner = torch.tensor([[0.0, 0.0]])            # the object never covers the image corner
    _anchors, valid = anchors_from_pose(clip.surface_points, clip.surface_normals, clip.poses_gt[0],
                                        clip.scale_gt, clip.intrinsics, clip.image_hw, far_corner)
    assert not bool(valid[0])


def test_register_keyframe_pose_estimates_a_depth_scale_error() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=2, image_hw=(240, 320), shape="box", n_surface=2048,
                                               depth_noise=0.001), seed=11)
    mask = clip.masks[0]
    k_err = 1.2                                          # the pointmap reads 20% too far, as monocular depth can
    observed = k_err * clip.pointmaps[0][mask & torch.isfinite(clip.pointmaps[0]).all(-1)]
    q = _axis_angle([0.3, -0.5, 0.8], 100.0)
    points, normals = clip.surface_points @ q, clip.surface_normals @ q
    true_pose = clip.poses_gt[0].clone()
    true_pose[:3, :3] = true_pose[:3, :3] @ q
    true_pose[:3, 3] = k_err * true_pose[:3, 3]           # the same object, seen in the scaled pointmap's metric

    reg = register_keyframe_pose(points, normals, observed, clip.intrinsics, mask, scale=clip.scale_gt,
                                 estimate_scale=True, n_obs=256, n_mesh=512, n_iters=40, seed=0, device="cpu")

    assert reg.scale == pytest.approx(k_err * clip.scale_gt, rel=0.05)   # partial-view similarity ICP: few % low
    assert _posed_nn_error(points, reg.pose, true_pose, reg.scale) < 0.006


def test_gate_init_poses_replaces_implausible_pnp_frames() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=5, image_hw=(240, 320), n_surface=2048), seed=3)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    pnp = clip.poses_gt.clone()
    pnp[3, :3, 3] += torch.tensor([0.0, 0.0, 5.0])      # EPnP landed 5 m behind the object
    n_inliers = torch.tensor([40, 40, 40, 40, 2])        # frame 4 never solved directly

    poses, accepted = gate_init_poses(pnp, n_inliers, clip.surface_points, clip.scale_gt, obs, 0, clip.poses_gt[0])

    assert accepted.tolist() == [True, True, True, False, False]
    assert torch.allclose(poses[:3], clip.poses_gt[:3])
    assert torch.allclose(poses[3, :3, :3], clip.poses_gt[2, :3, :3])       # nearest accepted rotation
    for k in (3, 4):                                    # translation re-seated on the frame's own pointmap
        assert float((poses[k, :3, 3] - clip.poses_gt[k, :3, 3]).norm()) < 0.5 * clip.scale_gt


def test_gate_init_poses_checks_sparse_frames_against_the_nearest_verified_frame() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=5, image_hw=(240, 320), n_surface=2048), seed=3)
    pointmaps = clip.pointmaps.clone()
    pointmaps[[2, 3]] = float("nan")                     # frames 2 and 3: too few pointmap points to check
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    pnp = clip.poses_gt.clone()
    pnp[3, :3, 3] += torch.tensor([0.4, 0.0, 3.0])       # frame 3: EPnP far off, on a frame with no pointmap
    n_inliers = torch.full((5,), 40)

    poses, accepted = gate_init_poses(pnp, n_inliers, clip.surface_points, clip.scale_gt, obs, 0, clip.poses_gt[0])

    assert accepted.tolist() == [True, True, True, False, True]    # frame 2 agrees with frame 1; frame 3 does not
    assert torch.allclose(poses[2], clip.poses_gt[2])
    assert torch.allclose(poses[3], poses[4]) or torch.allclose(poses[3], poses[2])   # nearest accepted pose

