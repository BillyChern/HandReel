"""Tests for `fastdo.io.datasets.dexycb_real.build_real_observations` (fully faked front end)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor

from fastdo.bench import Profiler
from fastdo.io.clip_record import HandTrack
from fastdo.io.datasets import dexycb_real
from fastdo.io.datasets.dexycb_frames import DexYCBFrames


def _fake_propagate_broadcast(frames: Tensor, keyframe_index: int, keyframe_mask: Tensor,
                              device: str = "cuda") -> Tensor:
    """Fake SAM 2: the keyframe mask, unchanged, at every frame."""
    return keyframe_mask.unsqueeze(0).expand(frames.shape[0], -1, -1).clone()


def _make_fake_run_moge(pointmap: Tensor) -> Any:
    def _fake_run_moge(frames: Tensor, device: str = "cuda", model: Any = None,
                       intrinsics: Tensor | None = None) -> Tensor:
        return pointmap.unsqueeze(0).expand(frames.shape[0], -1, -1, -1).clone()

    return _fake_run_moge


def _fake_track_stationary(frames: Tensor, keyframe_index: int, query_uv: Tensor, device: str = "cuda",
                           model: Any = None) -> tuple[Tensor, Tensor]:
    """Fake CoTracker: every query point stays put for the whole clip, fully visible."""
    t, j = frames.shape[0], query_uv.shape[0]
    return query_uv.unsqueeze(0).expand(t, j, 2).clone(), torch.ones(t, j, dtype=torch.bool)


def _clip(t: int, h: int, w: int, keyframe_index: int, object_id: int) -> dict[str, Any]:
    """Synthetic clip: object square at the keyframe, hand joints 1 m in front of the camera."""
    seg = torch.zeros(t, h, w, dtype=torch.uint8)
    seg[keyframe_index, h // 4: h // 2, w // 4: w // 2] = object_id
    joints = torch.zeros(t, 21, 3)
    joints[..., 2] = 1.0
    K = torch.tensor([[10.0, 0.0, w / 2], [0.0, 10.0, h / 2], [0.0, 0.0, 1.0]])
    return {"frames": torch.zeros(t, h, w, 3, dtype=torch.uint8), "seg": seg, "object_id": object_id,
            "joint_3d_cam": joints, "joint_valid": torch.ones(t, 21, dtype=torch.bool), "K": K,
            "keyframe_index": keyframe_index}


def _patch_front_end(monkeypatch: pytest.MonkeyPatch, pointmap: Tensor, propagate: Any = None,
                     track: Any = None) -> None:
    monkeypatch.setattr(dexycb_real, "propagate_masks_sam2", propagate or _fake_propagate_broadcast)
    monkeypatch.setattr(dexycb_real, "run_moge", _make_fake_run_moge(pointmap))
    monkeypatch.setattr(dexycb_real, "track_points_cotracker", track or _fake_track_stationary)


def test_build_real_observations_keyframe_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Tensor] = {}

    def _spy_propagate(frames: Tensor, keyframe_index: int, keyframe_mask: Tensor,
                       device: str = "cuda") -> Tensor:
        captured["mask"] = keyframe_mask.clone()
        return _fake_propagate_broadcast(frames, keyframe_index, keyframe_mask, device)

    clip = _clip(t=4, h=16, w=16, keyframe_index=2, object_id=5)
    clip["seg"][0, 0:4, 0:4] = 5                    # another frame's labels must NOT be the prompt
    _patch_front_end(monkeypatch, torch.ones(16, 16, 3), propagate=_spy_propagate)

    dexycb_real.build_real_observations(**clip, n_track_queries=8, device="cpu")

    expected = clip["seg"][2] == 5
    assert bool(expected.any())
    assert torch.equal(captured["mask"], expected)


def test_build_real_observations_scale_matches_hand_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    t, h, w, k_true = 3, 12, 12, 2.5
    metric = torch.zeros(h, w, 3)
    metric[..., 2] = 1.0                           # every pixel at 1 m, same depth as the hand joints
    _patch_front_end(monkeypatch, metric / k_true)  # what MoGe reports before the metric correction

    obs = dexycb_real.build_real_observations(**_clip(t, h, w, keyframe_index=1, object_id=7),
                                              n_track_queries=8, device="cpu").observations

    corrected = obs.pointmaps[..., 2] / (metric[..., 2] / k_true)
    assert torch.allclose(corrected, torch.full_like(corrected, k_true), rtol=0.01)


def test_build_real_observations_rebuilds_moge_points_through_the_camera_intrinsics(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Tensor | None] = {}

    def _spy_run_moge(frames: Tensor, device: str = "cuda", model: Any = None,
                      intrinsics: Tensor | None = None) -> Tensor:
        seen["intrinsics"] = intrinsics
        return torch.ones(frames.shape[0], 16, 16, 3)

    clip = _clip(t=3, h=16, w=16, keyframe_index=0, object_id=4)
    clip["K"] = torch.tensor([[12.0, 0.0, 6.5], [0.0, 12.5, 9.25], [0.0, 0.0, 1.0]])   # off-centre principal point
    _patch_front_end(monkeypatch, torch.ones(16, 16, 3))
    monkeypatch.setattr(dexycb_real, "run_moge", _spy_run_moge)

    dexycb_real.build_real_observations(**clip, n_track_queries=8, device="cpu")

    assert seen["intrinsics"] is not None and torch.equal(seen["intrinsics"], clip["K"])


def test_build_real_observations_ignores_non_keyframe_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _clip(t=4, h=16, w=16, keyframe_index=1, object_id=3)
    _patch_front_end(monkeypatch, torch.ones(16, 16, 3))
    a = dexycb_real.build_real_observations(**clip, n_track_queries=8, device="cpu")

    scrambled = dict(clip)
    seg = clip["seg"].clone()
    seg[[0, 2, 3]] = torch.randint(0, 8, (3, 16, 16), dtype=torch.uint8)    # per-frame GT labels: garbage
    scrambled["seg"] = seg
    b = dexycb_real.build_real_observations(**scrambled, n_track_queries=8, device="cpu")

    for name in ("masks", "pointmaps", "tracks"):
        assert torch.equal(getattr(a.observations, name), getattr(b.observations, name)), name
    assert torch.equal(a.track_vis, b.track_vis)
    assert torch.equal(a.keyframe_points, b.keyframe_points)


def test_build_real_observations_visibility_requires_track_inside_propagated_mask(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _clip(t=3, h=16, w=16, keyframe_index=0, object_id=2)

    def _drifting_track(frames: Tensor, keyframe_index: int, query_uv: Tensor, device: str = "cuda",
                        model: Any = None) -> tuple[Tensor, Tensor]:
        tracks, vis = _fake_track_stationary(frames, keyframe_index, query_uv)
        tracks[1] = torch.tensor([0.0, 0.0])        # frame 1: inside the image, outside the object mask
        tracks[2] = torch.tensor([99.0, 99.0])      # frame 2: outside the image altogether
        return tracks, vis

    _patch_front_end(monkeypatch, torch.ones(16, 16, 3), track=_drifting_track)
    real = dexycb_real.build_real_observations(**clip, n_track_queries=8, device="cpu")

    assert bool(real.track_vis[0].all())
    assert not bool(real.track_vis[1].any())
    assert not bool(real.track_vis[2].any())


def test_build_real_observations_are_not_solver_ready_until_anchored(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _clip(t=3, h=16, w=16, keyframe_index=0, object_id=2)
    _patch_front_end(monkeypatch, torch.ones(16, 16, 3))
    real = dexycb_real.build_real_observations(**clip, n_track_queries=8, device="cpu")

    unanchored = real.observations
    assert torch.isnan(unanchored.anchors).all() and not bool(unanchored.track_vis.any())
    assert bool(real.track_vis.all())                                    # the tracks' own 2D visibility
    assert torch.equal(real.keyframe_points, torch.ones(8, 3))           # camera-frame points, kept apart

    mesh_anchors = torch.randn(8, 3)
    valid = torch.tensor([True, False] * 4)
    obs = real.anchored(mesh_anchors, valid)
    assert torch.equal(obs.anchors, mesh_anchors)
    assert torch.equal(obs.track_vis, real.track_vis & valid[None])
    assert obs.pointmaps is unanchored.pointmaps and obs.tracks is unanchored.tracks
    with pytest.raises(ValueError, match="anchors"):
        real.anchored(torch.zeros(5, 3), valid)


def test_build_real_observations_records_front_end_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_front_end(monkeypatch, torch.ones(16, 16, 3))
    profiler = Profiler(clip_id="c", system="s", n_frames=4, fps=10.0, sync_cuda=False)
    dexycb_real.build_real_observations(**_clip(t=4, h=16, w=16, keyframe_index=0, object_id=1),
                                        n_track_queries=8, device="cpu", profiler=profiler, stage_prefix="seq")
    names = [s.name for s in profiler.report().stages]
    assert names == ["seq/sam2", "seq/moge", "seq/cotracker"]


def _bitwise_equal(a: Tensor | None, b: Tensor | None) -> bool:
    """`torch.equal`, with NaN equal to NaN (the unanchored observations' anchors are all NaN); None (an unset
    optional field, e.g. `object_pixels`) equals only None."""
    if a is None or b is None:
        return a is b
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        return torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(), b.nan_to_num())
    return torch.equal(a, b)


def test_assemble_real_observations_is_the_post_model_half_of_build(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Tensor] = {}

    def spy_propagate(frames: Tensor, keyframe_index: int, keyframe_mask: Tensor, device: str = "cuda") -> Tensor:
        seen["masks"] = _fake_propagate_broadcast(frames, keyframe_index, keyframe_mask, device)
        return seen["masks"].clone()

    def spy_track(frames: Tensor, keyframe_index: int, query_uv: Tensor, device: str = "cuda",
                  model: Any = None) -> tuple[Tensor, Tensor]:
        tracks, vis = _fake_track_stationary(frames, keyframe_index, query_uv)
        tracks[2] = tracks[2] + 3.0                  # drift out of the mask on one frame
        vis[3, :2] = False
        seen["query_uv"], seen["tracks"], seen["vis"] = query_uv.clone(), tracks.clone(), vis.clone()
        return tracks, vis

    metric_scale = dexycb_real.apply_metric_scale

    def spy_scale(raw: Tensor, scale: Tensor) -> Tensor:
        seen["pointmaps"] = metric_scale(raw, scale)
        return seen["pointmaps"].clone()

    pointmap = torch.ones(16, 16, 3)
    pointmap[4:6, 4:6] = float("nan")                # NaN keyframe points under some queries
    clip = _clip(t=4, h=16, w=16, keyframe_index=1, object_id=3)
    _patch_front_end(monkeypatch, pointmap, propagate=spy_propagate, track=spy_track)
    monkeypatch.setattr(dexycb_real, "apply_metric_scale", spy_scale)

    built = dexycb_real.build_real_observations(**clip, n_track_queries=8, device="cpu")
    inputs = (clip["K"], seen["masks"].to(torch.bool), seen["pointmaps"], seen["tracks"], seen["vis"], seen["query_uv"])
    assembled = dexycb_real.assemble_real_observations(*inputs, 1)

    for field in dataclasses.fields(built.observations):
        assert _bitwise_equal(getattr(built.observations, field.name), getattr(assembled.observations, field.name)), \
            field.name
    assert torch.equal(built.track_vis, assembled.track_vis) and not bool(built.track_vis.all())
    assert _bitwise_equal(built.keyframe_points, assembled.keyframe_points)
    with pytest.raises(ValueError):
        dexycb_real.assemble_real_observations(*inputs, 4)          # keyframe_index = T
    K, masks, pointmaps, tracks, vis, query_uv = inputs
    with pytest.raises(ValueError):
        dexycb_real.assemble_real_observations(K, masks, pointmaps, tracks[:-1], vis, query_uv, 1)


def test_load_real_inputs_reads_rgb_hand_joints_and_only_the_keyframe_labels(tmp_path: Path) -> None:
    seq = "20200709-subject-01/20200709_141754/836212060125"
    d = tmp_path / seq
    d.mkdir(parents=True)
    for i in range(9):
        Image.fromarray(np.full((8, 10, 3), 10 * i, np.uint8)).save(d / f"color_{i:06d}.jpg", quality=100)
    seg = np.zeros((8, 10), np.uint8)
    seg[2:4, 3:8] = 5
    np.savez(d / "labels_000003.npz", seg=seg)        # only the keyframe's labels exist on disk
    joints = np.arange(9 * 21 * 3, dtype=np.float32).reshape(9, 21, 3)
    valid = np.array([True, True, True, False, True, True, True, True, True])
    hand = HandTrack(params=np.zeros((9, 48), np.float32), betas=np.zeros(10, np.float32),
                     trans=np.zeros((9, 3), np.float32), joints=joints, valid=valid)

    inputs = dexycb_real.load_real_inputs(DexYCBFrames(tmp_path), seq, np.array([0, 3, 6]), 1, hand)

    assert inputs.frames.shape == (3, 8, 10, 3) and inputs.frames.dtype == torch.uint8
    assert abs(int(inputs.frames[2, 0, 0, 0]) - 60) <= 2                      # frame 6's RGB
    assert torch.equal(inputs.seg[1], torch.from_numpy(seg.astype(np.int64)))
    assert not bool(inputs.seg[[0, 2]].any())                                  # non-keyframes: never loaded
    assert torch.equal(inputs.joint_3d_cam, torch.from_numpy(joints[[0, 3, 6]]))
    assert inputs.joint_valid.shape == (3, 21)
    assert bool(inputs.joint_valid[0].all()) and not bool(inputs.joint_valid[1].any())
