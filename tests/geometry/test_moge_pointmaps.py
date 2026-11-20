"""Tests for the MoGe pointmap loader and hand-anchored metric scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.geometry.moge_pointmaps import (
    apply_metric_scale,
    clip_median_scale,
    hand_anchor_pixels,
    hand_anchored_scale,
    hand_anchored_scale_from_samples,
    load_pointmaps,
)


def test_loads_batch_style_six_digit_names(tmp_path: Path) -> None:
    a = np.zeros((4, 5, 3), dtype=np.float32)
    a[..., 2] = 1.0
    b = np.ones((4, 5, 3), dtype=np.float32) * 2.0
    np.save(tmp_path / "000000_pointmap.npy", a)
    np.save(tmp_path / "000003_pointmap.npy", b)

    out = load_pointmaps(tmp_path, [0, 3])

    assert out.shape == (2, 4, 5, 3)
    np.testing.assert_allclose(out[0].numpy(), a)
    np.testing.assert_allclose(out[1].numpy(), b)


def test_loads_single_image_style_four_digit_names(tmp_path: Path) -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    np.save(tmp_path / "0125_pointmap.npy", a)

    out = load_pointmaps(tmp_path, [125])

    assert out.shape == (1, 2, 2, 3)


def test_non_finite_values_become_nan(tmp_path: Path) -> None:
    a = np.zeros((2, 2, 3), dtype=np.float32)
    a[0, 0] = [np.inf, 1.0, -np.inf]
    np.save(tmp_path / "000000_pointmap.npy", a)

    out = load_pointmaps(tmp_path, [0])

    # inf/-inf channels become NaN; the one finite channel (1.0) is untouched.
    np.testing.assert_equal(np.isnan(out[0, 0, 0].numpy()), [True, False, True])
    assert out[0, 0, 0, 1].item() == pytest.approx(1.0)


def test_companion_mask_marks_invalid_pixels_nan(tmp_path: Path) -> None:
    a = np.ones((2, 2, 3), dtype=np.float32)
    valid = np.array([[True, False], [True, True]])
    np.save(tmp_path / "000000_pointmap.npy", a)
    np.save(tmp_path / "000000_mask.npy", valid)

    out = load_pointmaps(tmp_path, [0]).numpy()

    assert not np.isnan(out[0, 0, 0]).any()
    assert np.isnan(out[0, 0, 1]).all()


def test_missing_frame_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pointmaps(tmp_path, [7])


def test_empty_frame_idx_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        load_pointmaps(tmp_path, [])


def test_inconsistent_shapes_raise(tmp_path: Path) -> None:
    np.save(tmp_path / "000000_pointmap.npy", np.zeros((2, 2, 3), dtype=np.float32))
    np.save(tmp_path / "000001_pointmap.npy", np.zeros((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="inconsistent shapes"):
        load_pointmaps(tmp_path, [0, 1])


# --- hand_anchored_scale / apply_metric_scale -------------------------------------------------

_H, _W = 100, 100
_FX = _FY = 1.0
_CX = _CY = 50.0
_INTRINSICS = torch.tensor([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]])


def _place_joints(pointmaps: torch.Tensor, frame: int, joints_real: list[tuple[float, float, float]],
                  scale: float) -> None:
    """Writes `scale * joint` into the pointmap at each joint's exact-integer projected pixel
    (fx=fy=1, so pixel = (X/Z + cx, Y/Z + cy); with Z=1 and integer X this lands exactly)."""
    for x, y, z in joints_real:
        u, v = int(_FX * x / z + _CX), int(_FY * y / z + _CY)
        pointmaps[frame, v, u] = torch.tensor([x * scale, y * scale, z * scale])


def test_hand_anchored_scale_recovers_one_over_known_global_scale() -> None:
    s = 2.5  # raw pointmap = s * real (MoGe's own global scale ambiguity)
    t, j = 2, 3
    # Z held constant *within* each frame (varied *across* frames instead) so the 3 joints'
    # X in {0,1,2} project to 3 distinct pixels (X/Z + cx) rather than colliding under the
    # projection's integer truncation, which a varying-Z-within-a-frame choice can cause.
    joints_frame0 = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]
    joints_frame1 = [(0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (2.0, 0.0, 2.0)]

    pointmaps = torch.full((t, _H, _W, 3), float("nan"))
    _place_joints(pointmaps, 0, joints_frame0, s)
    _place_joints(pointmaps, 1, joints_frame1, s)

    hand_joints_cam = torch.zeros(t, j, 3)
    hand_joints_cam[0] = torch.tensor(joints_frame0)
    hand_joints_cam[1] = torch.tensor(joints_frame1)
    valid = torch.ones(t, j, dtype=torch.bool)

    k = hand_anchored_scale(pointmaps, hand_joints_cam, _INTRINSICS, valid)

    assert k.shape == (t,)
    np.testing.assert_allclose(k.numpy(), [1.0 / s, 1.0 / s], atol=1e-6)


def test_hand_anchored_scale_nan_when_no_valid_joints() -> None:
    pointmaps = torch.zeros(1, _H, _W, 3)
    hand_joints_cam = torch.ones(1, 3, 3)
    valid = torch.zeros(1, 3, dtype=torch.bool)  # baseline: no usable hand -> frame is skipped

    k = hand_anchored_scale(pointmaps, hand_joints_cam, _INTRINSICS, valid)

    assert bool(torch.isnan(k[0]))


def test_hand_anchored_scale_excludes_out_of_bounds_joint() -> None:
    s = 2.0
    joints_in = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]
    joint_out = (10_000.0, 0.0, 1.0)  # projects far outside the (100,100) pointmap
    pointmaps = torch.full((1, _H, _W, 3), float("nan"))
    _place_joints(pointmaps, 0, joints_in, s)

    hand_joints_cam = torch.tensor([[*joints_in, joint_out]])
    valid = torch.ones(1, 3, dtype=torch.bool)

    k = hand_anchored_scale(pointmaps, hand_joints_cam, _INTRINSICS, valid)

    # matches the 2-joint-only estimate: the out-of-bounds joint contributes nothing, it does
    # not crash and does not silently sample an unrelated (clamped) pixel.
    np.testing.assert_allclose(k.numpy(), [1.0 / s], atol=1e-6)


def test_hand_anchored_scale_excludes_nan_pointmap_sample() -> None:
    s = 2.0
    joint_good = (0.0, 0.0, 1.0)
    joint_bad = (1.0, 0.0, 1.0)  # its pixel is left NaN in the pointmap (simulates a bad sample)
    pointmaps = torch.full((1, _H, _W, 3), float("nan"))
    _place_joints(pointmaps, 0, [joint_good], s)

    hand_joints_cam = torch.tensor([[joint_good, joint_bad]])
    valid = torch.ones(1, 2, dtype=torch.bool)

    k = hand_anchored_scale(pointmaps, hand_joints_cam, _INTRINSICS, valid)

    np.testing.assert_allclose(k.numpy(), [1.0 / s], atol=1e-6)


def test_hand_anchored_scale_degenerate_depth_is_nan() -> None:
    pointmaps = torch.zeros(1, _H, _W, 3)  # pointmap z is exactly 0 everywhere -> |h_pm_z| < 1e-6
    hand_joints_cam = torch.tensor([[[0.0, 0.0, 1.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)

    k = hand_anchored_scale(pointmaps, hand_joints_cam, _INTRINSICS, valid)

    assert bool(torch.isnan(k[0]))


def test_apply_metric_scale_broadcasts_per_frame() -> None:
    pointmaps = torch.ones(2, 3, 3, 3)
    k = torch.tensor([2.0, 0.5])

    out = apply_metric_scale(pointmaps, k)

    assert out.shape == pointmaps.shape
    np.testing.assert_allclose(out[0].numpy(), np.full((3, 3, 3), 2.0))
    np.testing.assert_allclose(out[1].numpy(), np.full((3, 3, 3), 0.5))


def test_clip_median_scale_pools_finite_frames() -> None:
    k = torch.tensor([float("nan"), 1.2, 0.8, 1.25, 1.3, float("nan")])
    out = clip_median_scale(k)
    assert out.shape == k.shape
    assert torch.allclose(out, torch.full((6,), 1.2))    # torch's lower median of [0.8, 1.2, 1.25, 1.3]
    assert torch.isnan(k[0])                              # input left untouched


def test_clip_median_scale_raises_without_any_valid_frame() -> None:
    with pytest.raises(ValueError, match="no valid frame"):
        clip_median_scale(torch.full((3,), float("nan")))


def _hand_anchored_scale_before_137(pointmaps: torch.Tensor, hand_joints_cam: torch.Tensor, intrinsics: torch.Tensor,
                                    valid: torch.Tensor) -> torch.Tensor:
    """`hand_anchored_scale` verbatim as it was before the split into pixels + formula: the oracle."""
    t, h, w = pointmaps.shape[:3]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    k = torch.full((t,), float("nan"), dtype=torch.float32)
    for frame in range(t):
        real_z: list[float] = []
        pm_z: list[float] = []
        for j in range(hand_joints_cam.shape[1]):
            if not bool(valid[frame, j]):
                continue
            joint = hand_joints_cam[frame, j]
            if not bool(torch.isfinite(joint).all()):
                continue
            x, y, z = float(joint[0]), float(joint[1]), float(joint[2])
            if z <= 0.0:
                continue
            u, v = int(fx * x / z + cx), int(fy * y / z + cy)
            if not (0 <= u < w and 0 <= v < h):
                continue
            sample = pointmaps[frame, v, u]
            if not bool(torch.isfinite(sample).all()):
                continue
            real_z.append(z)
            pm_z.append(float(sample[2]))
        if not real_z:
            continue
        h_pm_z = float(np.mean(pm_z))
        if abs(h_pm_z) < 1e-6:
            continue
        k[frame] = float(np.mean(real_z)) / h_pm_z
    return k


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hand_anchored_scale_split_is_bit_identical_to_the_original(dtype: torch.dtype) -> None:
    """Pixels + formula == the original per-joint loop, bit for bit, through every exclusion branch (invalid,
    non-finite, behind the camera, out of bounds, NaN sample in one channel, degenerate depth, empty frame)."""
    gen = torch.Generator().manual_seed(7)
    t, h, w, j = 9, 20, 30, 21
    K = torch.tensor([[25.0, 0.0, 14.3], [0.0, 26.0, 9.7], [0.0, 0.0, 1.0]])
    pointmaps = (torch.rand(t, h, w, 3, generator=gen, dtype=torch.float64) + 0.3).to(dtype)
    pointmaps[torch.rand(t, h, w, generator=gen) < 0.15] = float("nan")
    pointmaps[..., 0][torch.rand(t, h, w, generator=gen) < 0.05] = float("nan")   # one channel only
    pointmaps[5, ..., 2] = 1e-9                                                  # degenerate depth frame
    joints = torch.randn(t, j, 3, generator=gen) * torch.tensor([0.4, 0.3, 0.3]) + torch.tensor([0.0, 0.0, 0.5])
    joints[0, :4, 2] = -0.2                                                      # behind the camera
    joints[1, 3] = float("nan")
    valid = torch.rand(t, j, generator=gen) > 0.2
    valid[7] = False                                                             # no joint at all

    expected = _hand_anchored_scale_before_137(pointmaps, joints, K, valid)
    got = hand_anchored_scale(pointmaps, joints, K, valid)

    assert got.dtype == torch.float32
    assert got.numpy().tobytes() == expected.numpy().tobytes()
    assert torch.isnan(expected[[5, 7]]).all() and torch.isfinite(expected).sum() >= 5   # branches reached
    frame, v, u, z = hand_anchor_pixels(joints, K, valid, (t, h, w))
    assert bool(((v >= 0) & (v < h) & (u >= 0) & (u < w)).all()) and int(frame.numel()) < int(valid.sum())
    samples = pointmaps[frame, v, u]
    assert hand_anchored_scale_from_samples(t, frame, z, samples).numpy().tobytes() == expected.numpy().tobytes()
