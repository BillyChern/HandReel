import math

import cv2
import pytest
import torch

from fastdo.geometry.lie import se3_inv, se3_log
from fastdo.solver.init import assign_anchors, pnp_initialize, scale_from_extent
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip


def test_assign_anchors_matches_ground_truth() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=3, image_hw=(240, 320), n_surface=2048, n_tracks=32,
                                               depth_noise=0.0), seed=7)
    anchors, valid = assign_anchors(clip.surface_points, clip.pointmaps[0], clip.tracks[0], clip.track_vis[0],
                                    clip.poses_gt[0], clip.scale_gt)
    assert valid.sum() >= 0.8 * clip.track_vis[0].sum()
    err = (anchors[valid] - clip.anchors[valid]).norm(dim=1)
    assert err.median() < 0.03                      # snapped to the nearest of 2048 samples on a unit shape


def test_scale_from_extent_within_30_percent() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=2, image_hw=(240, 320), n_surface=1024), seed=8)
    s = scale_from_extent(clip.pointmaps[0], clip.masks[0], clip.surface_points)
    assert 0.7 < s / clip.scale_gt < 1.3


def test_pnp_initialize_close_to_truth() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=12, image_hw=(240, 320), n_surface=1024, n_tracks=64,
                                               occlusion=0.3), seed=9)
    valid = torch.ones(clip.anchors.shape[0], dtype=torch.bool)
    poses, n_in = pnp_initialize(clip.anchors, valid, clip.tracks, clip.track_vis, clip.intrinsics, clip.scale_gt)
    assert poses.shape == (12, 4, 4)
    solvable = clip.track_vis.sum(dim=1) >= 6           # frames PnP can possibly solve
    solved = n_in >= 6
    assert solvable.sum() >= 6, "fixture must leave at least 6 solvable frames"
    assert bool(solved[solvable].all()), "every frame with >= 6 visible tracks must be solved"
    assert (~solved | solvable).all()                    # never claims a solve without enough tracks
    rel = se3_log(poses.double() @ se3_inv(clip.poses_gt.double()))
    assert rel[solved, :3].norm(dim=1).max() < 0.02
    assert math.degrees(rel[solved, 3:].norm(dim=1).max()) < 5.0
    # unsolved frames inherit the nearest solved frame's pose
    for k in torch.nonzero(~solved).flatten().tolist():
        nearest = torch.nonzero(solved).flatten()
        j = int(nearest[(nearest - k).abs().argmin()])
        assert torch.equal(poses[k], poses[j])


def test_pnp_initialize_treats_cv2_error_as_unsolved(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=5, image_hw=(240, 320), n_surface=1024, n_tracks=32,
                                               occlusion=0.0), seed=11)
    valid = torch.ones(clip.anchors.shape[0], dtype=torch.bool)
    # every frame here has >= 6 visible tracks, so solvePnPRansac is called exactly once per
    # frame, in frame order; raise on the call matching the target frame's index.
    assert (clip.track_vis.sum(dim=1) >= 6).all()
    bad_frame = 2
    real_solve = cv2.solvePnPRansac
    calls = {"n": 0}

    def flaky_solve(*args: object, **kwargs: object) -> object:
        idx = calls["n"]
        calls["n"] += 1
        if idx == bad_frame:
            raise cv2.error("degenerate")
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(cv2, "solvePnPRansac", flaky_solve)
    poses, n_in = pnp_initialize(clip.anchors, valid, clip.tracks, clip.track_vis, clip.intrinsics, clip.scale_gt)
    assert calls["n"] == 5
    assert int(n_in[bad_frame]) == 0
    others = [k for k in range(5) if k != bad_frame]
    assert (n_in[others] >= 6).all()
    solved = torch.nonzero(n_in >= 6).flatten()
    nearest = solved[(solved - bad_frame).abs().argmin()]
    assert torch.equal(poses[bad_frame], poses[nearest])


def test_pnp_initialize_raises_when_nothing_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=4, image_hw=(240, 320), n_surface=512, n_tracks=16,
                                               occlusion=0.0), seed=12)
    valid = torch.ones(clip.anchors.shape[0], dtype=torch.bool)

    def always_raise(*args: object, **kwargs: object) -> object:
        raise cv2.error("degenerate")

    monkeypatch.setattr(cv2, "solvePnPRansac", always_raise)
    with pytest.raises(ValueError):
        pnp_initialize(clip.anchors, valid, clip.tracks, clip.track_vis, clip.intrinsics, clip.scale_gt)


def test_scale_from_extent_raises_on_too_few_points() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=2, image_hw=(240, 320), n_surface=256), seed=13)
    mask = torch.zeros_like(clip.masks[0])
    valid_idx = torch.nonzero(clip.masks[0])
    assert valid_idx.shape[0] >= 3
    for y, x in valid_idx[:3].tolist():
        mask[y, x] = True
    with pytest.raises(ValueError):
        scale_from_extent(clip.pointmaps[0], mask, clip.surface_points)
