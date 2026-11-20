import torch

from fastdo.geometry.lie import se3_exp
from fastdo.solver.synthetic import SyntheticConfig, _smooth_trajectory, make_synthetic_clip


def test_shapes_and_determinism() -> None:
    cfg = SyntheticConfig(n_frames=8, image_hw=(120, 160), n_surface=512, n_tracks=16)
    a = make_synthetic_clip(cfg, seed=1)
    b = make_synthetic_clip(cfg, seed=1)
    assert a.poses_gt.shape == (8, 4, 4) and a.masks.shape == (8, 120, 160)
    assert a.pointmaps.shape == (8, 120, 160, 3) and a.tracks.shape == (8, 16, 2)
    assert torch.equal(a.poses_gt, b.poses_gt) and torch.equal(a.masks, b.masks)


def test_pointmap_consistent_with_pose() -> None:
    cfg = SyntheticConfig(n_frames=4, image_hw=(120, 160), n_surface=2048, n_tracks=16, depth_noise=0.0)
    clip = make_synthetic_clip(cfg, seed=2)
    for k in range(4):
        pts = clip.pointmaps[k][clip.masks[k]]              # camera-frame points
        assert pts.shape[0] > 50
        T = clip.poses_gt[k]
        canon = (pts - T[:3, 3]) @ T[:3, :3] / clip.scale_gt  # back to canonical frame
        d = torch.cdist(canon, clip.surface_points).min(dim=1).values
        assert d.median() < 0.02                                # within 2% of the unit-normalized shape


def test_occluder_removes_observations() -> None:
    cfg = SyntheticConfig(n_frames=4, image_hw=(120, 160), n_surface=512, n_tracks=32)
    full = make_synthetic_clip(cfg, seed=3)
    occ = make_synthetic_clip(SyntheticConfig(**{**cfg.__dict__, "occlusion": 0.6}), seed=3)
    assert occ.masks.sum() < full.masks.sum()
    assert occ.track_vis.float().mean() < full.track_vis.float().mean()
    assert occ.hand_masks.any()
    assert torch.equal(occ.poses_gt, full.poses_gt)


def test_tracks_project_anchors() -> None:
    cfg = SyntheticConfig(n_frames=3, image_hw=(120, 160), n_surface=512, n_tracks=16, depth_noise=0.0)
    clip = make_synthetic_clip(cfg, seed=4)
    K = clip.intrinsics
    for k in range(3):
        T = clip.poses_gt[k]
        p = clip.scale_gt * clip.anchors @ T[:3, :3].T + T[:3, 3]
        uv = torch.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], dim=1)
        vis = clip.track_vis[k]
        assert torch.allclose(uv[vis], clip.tracks[k][vis], atol=1e-4)


def _smooth_trajectory_unguarded(n: int, step: float, distance: float, gen: torch.Generator) -> torch.Tensor:
    """Verbatim copy of `_smooth_trajectory` as it existed before the frustum-bounce guard was
    added (no `focal`/`image_hw` args, no bounce). Used only as a golden reference to prove the guard is
    a no-op for a clip whose random walk never approaches the image border (an explicit
    regression requirement)."""
    xi = torch.randn(n, 6, generator=gen, dtype=torch.float64) * step
    xi[:, 3:] *= 2.0
    kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    for _ in range(3):
        xi = torch.nn.functional.conv1d(xi.T.unsqueeze(0), kernel.view(1, 1, 3).expand(6, 1, 3),
                                        padding=1, groups=6).squeeze(0).T
    T = torch.eye(4, dtype=torch.float64)
    T[2, 3] = distance
    poses = [T]
    for k in range(1, n):
        poses.append(se3_exp(xi[k]) @ poses[-1])
    return torch.stack(poses).float()


def test_frustum_guard_keeps_t100_box_seed3_observable() -> None:
    """The T=100/seed=3/box fixture (`test_runtime_budget_100_frames`'s clip) used to leave the
    320x240 image entirely by frame 70, leaving frames 70-99 with zero mask pixels (the
    root cause of a solver-test diagnostic). The frustum-bounce guard in `_smooth_trajectory`
    must keep at least `_MIN_VALID_OBS`(=20)-worth of valid mask pixels in every frame."""
    cfg = SyntheticConfig(n_frames=100, image_hw=(240, 320), n_surface=2048, n_tracks=64, occlusion=0.3,
                          shape="box")
    clip = make_synthetic_clip(cfg, seed=3)
    n_valid = clip.masks.flatten(1).sum(1)
    assert int(n_valid.min()) >= 20, n_valid.tolist()


def test_frustum_guard_is_noop_for_bounded_default_clip() -> None:
    """Regression: for a clip whose random walk never approaches the image border (T=8, default
    motion_scale/distance/focal/image_hw), the frustum guard must not change a single pose - bit-identical
    to the pre-guard `_smooth_trajectory`."""
    ref = _smooth_trajectory_unguarded(8, 0.02, 0.6, torch.Generator().manual_seed(7))
    guarded = _smooth_trajectory(8, 0.02, 0.6, torch.Generator().manual_seed(7), focal=300.0,
                                 image_hw=(240, 320))
    assert torch.equal(ref, guarded)
