# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""The one-call API: per-body lam, pure-gauge contrast against the aligned protocols, inputs and intrinsics."""

from __future__ import annotations

import numpy as np
import pytest

from alongray import e_par_split, mpjpe_root_aligned, pa_mpjpe, reprojection_error_px, score

K = np.array([[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])


def _traj(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([0.05, -0.03, 0.55])
    return base + 0.08 * rng.standard_normal((n, 3)) * np.array([1.0, 1.0, 0.5])


def _rot(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _scene(n: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = _traj(n, seed)
    shape = 0.04 * rng.standard_normal((600, 3))
    poses = np.tile(np.eye(4), (n, 1, 1))
    for t in range(n):
        poses[t, :3, :3] = _rot(seed + t)
    poses[:, :3, 3] = centres
    pts = np.stack([shape @ poses[t, :3, :3].T + centres[t] for t in range(n)])
    hand_local = 0.03 * rng.standard_normal((21, 3))
    hand_local[0] = 0.0
    wrist = centres + np.array([0.0, 0.08, 0.02])
    joints = np.stack([hand_local @ _rot(100 + t).T + wrist[t] for t in range(n)])
    return {"poses": poses, "points": pts, "joints": joints}


def test_pure_gauge_moves_e_par_and_no_aligned_protocol() -> None:
    """Object g_k (centres and size times k) and a hand-depth error (hand translated by (k-1) wrist, fixed size)."""
    s = _scene()
    k = 1.1
    pred_poses = s["poses"].copy()
    pred_poses[:, :3, 3] *= k
    pred_pts = k * s["points"]
    wrist = s["joints"][:, 0]
    pred_joints = s["joints"] + (k - 1.0) * wrist[:, None, :]
    res = score(pred_poses, s["poses"], pred_joints, s["joints"], K,
                pred_object_points=pred_pts, gt_object_points=s["points"])
    for name in ("object", "hand"):
        b = res[name]
        assert b.lam == pytest.approx(k, rel=1e-13)
        assert b.off_orbit_part == pytest.approx(0.0, abs=1e-13)
        assert b.e_par == pytest.approx(b.gauge_part, rel=1e-12)
        assert b.e_par == pytest.approx((k - 1.0) * b.z_rms, rel=1e-12)
        assert b.e_par > 0.05                                         # about 60 mm at 0.6 m
        assert b.reproj_rms_px == pytest.approx(0.0, abs=1e-9)        # image-exact: no pixel moves
        assert b.aligned["centre_rms_after_lam"] == pytest.approx(0.0, abs=1e-13)
        assert b.aligned["centre_ate_sim3"] == pytest.approx(0.0, abs=1e-12)
    assert res["hand"].aligned["mpjpe_root_aligned"] == pytest.approx(0.0, abs=1e-14)
    assert res["hand"].aligned["pa_mpjpe"] == pytest.approx(0.0, abs=1e-12)
    cd0 = score(s["poses"], s["poses"], pred_object_points=s["points"], gt_object_points=s["points"])
    assert res["object"].aligned["chamfer_cm2_after_scaled_icp"] == pytest.approx(
        cd0["object"].aligned["chamfer_cm2_after_scaled_icp"], abs=1e-9)


def test_each_body_gets_its_own_lambda() -> None:
    s = _scene()
    pred_poses = s["poses"].copy()
    pred_poses[:, :3, 3] *= 0.9
    res = score(pred_poses, s["poses"], 1.2 * s["joints"][:, 0], s["joints"][:, 0])
    assert res["object"].lam == pytest.approx(0.9, rel=1e-13)
    assert res["hand"].lam == pytest.approx(1.2, rel=1e-13)
    assert "mpjpe_root_aligned" not in res["hand"].aligned           # a (T, 3) wrist has no joints


def test_score_matches_the_low_level_split_and_off_orbit_error_is_seen() -> None:
    rng = np.random.default_rng(4)
    gt = _traj(25, 4)
    pred = 1.05 * gt + rng.normal(0, 0.01, gt.shape)
    b = score(pred, gt, k=(600.0, 610.0, 320.0, 240.0))["object"]
    sp = e_par_split(pred, gt)
    assert (b.e_par, b.gauge_part, b.off_orbit_part, b.lam, b.z_rms, b.oracle_floor, b.n_frames) == (
        sp.e_par, sp.gauge_part, sp.off_orbit_part, sp.lam, sp.z_rms, sp.oracle_floor, sp.n_frames)
    assert b.reproj_rms_px > 0.5 and b.off_orbit_part > 0.001
    assert b.e_3d ** 2 == pytest.approx(b.e_par ** 2 + b.e_perp ** 2, rel=1e-12)


def test_reprojection_uses_the_intrinsics() -> None:
    gt = np.array([[0.0, 0.0, 1.0]])
    pred = np.array([[0.01, 0.0, 1.0]])
    assert reprojection_error_px(pred, gt, K)[0] == pytest.approx(6.0)
    assert np.isnan(reprojection_error_px(np.array([[0.0, 0.0, -1.0]]), gt, K)[0])


def test_input_errors() -> None:
    gt = _traj()
    with pytest.raises(ValueError):
        score(gt, gt, pred_hand=gt)
    with pytest.raises(ValueError):
        score(gt, gt, k=np.eye(2))
    with pytest.raises(ValueError):
        score(gt[:, :2], gt[:, :2])
    with pytest.raises(ValueError):
        score(gt, gt, pred_object_points=[gt])


def test_object_shapes_and_pose_bottom_row_are_checked() -> None:
    s = _scene()
    with pytest.raises(ValueError, match="centres or"):
        score(s["joints"], s["joints"])                 # (T, J, 3) is not an object input
    bad = s["poses"].copy()
    bad[2, 3, :] = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="bottom row"):
        score(bad, s["poses"])
    with pytest.raises(ValueError, match="wrists or"):
        score(s["poses"], s["poses"], s["poses"], s["poses"])
    with pytest.raises(ValueError, match="root index"):
        score(s["poses"], s["poses"], s["joints"], s["joints"], hand_root=21)
    nan_pose = s["poses"].copy()
    nan_pose[1] = np.nan                                # a missing frame is not a bad bottom row
    assert score(nan_pose, s["poses"])["object"].n_frames == len(nan_pose) - 1


def test_fy_is_not_fx_through_tuple_intrinsics() -> None:
    gt = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    k = (600.0, 700.0, 320.0, 240.0)
    np.testing.assert_allclose(reprojection_error_px(gt + [0.0, 0.01, 0.0], gt, k), [7.0, 3.5], rtol=1e-12)
    np.testing.assert_allclose(reprojection_error_px(gt + [0.01, 0.0, 0.0], gt, k), [6.0, 3.0], rtol=1e-12)
    b = score(gt + [0.0, 0.01, 0.0], gt, k=k)["object"]
    assert b.reproj_rms_px == pytest.approx(np.sqrt((7.0 ** 2 + 3.5 ** 2) / 2), rel=1e-12)
    assert b.n_reproj_frames == 2


def _hand_root3(lam: float = 1.2, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The whole hand translated by (lam - 1) times joint 3, plus a 10 mm error on joint 0 only."""
    s = _scene(seed=seed)
    gt = s["joints"]
    pred = gt + (lam - 1.0) * gt[:, 3:4]
    pred[:, 0] += np.array([0.01, 0.0, 0.0])
    return pred, gt


def test_hand_root_selects_the_reference_joint() -> None:
    pred, gt = _hand_root3()
    obj = _traj()
    h3 = score(obj, obj, pred, gt, hand_root=3)["hand"]
    h0 = score(obj, obj, pred, gt)["hand"]
    assert h3.lam == pytest.approx(1.2, rel=1e-12) and h3.off_orbit_part == pytest.approx(0.0, abs=1e-12)
    assert h0.off_orbit_part > 1e-3
    assert h3.aligned["mpjpe_root_aligned"] == pytest.approx(0.01 / 21, rel=1e-9)
    assert h0.aligned["mpjpe_root_aligned"] == pytest.approx(0.01 * 20 / 21, rel=1e-9)


def test_reprojection_warns_when_no_frame_is_in_front() -> None:
    gt = _traj() * np.array([1.0, 1.0, -1.0])           # an OpenGL-style -z forward camera
    with pytest.warns(UserWarning, match=r"\+z forward"):
        b = score(1.05 * gt, gt, k=K)["object"]
    assert b.n_reproj_frames == 0 and np.isnan(b.reproj_rms_px)
    assert b.lam == pytest.approx(1.05, rel=1e-12)       # e_par itself does not depend on the axis convention


def test_one_degenerate_hand_frame_does_not_abort_pa_mpjpe() -> None:
    s = _scene()
    pred = s["joints"].copy()
    pred[4] = pred[4, :1]                               # every joint of frame 4 at the wrist
    h = score(s["poses"], s["poses"], pred, s["joints"])["hand"]
    assert np.isfinite(h.aligned["pa_mpjpe"])


@pytest.mark.parametrize("n_frames", [21, 1])
def test_joints_against_a_wrist_trajectory_are_refused_not_broadcast(n_frames: int) -> None:
    """T == J (21) or T == 1 used to broadcast (T, J, 3) against (T, 3) into a silent MPJPE of 0.6-0.7 m."""
    s = _scene(n=n_frames)
    obj = s["poses"]
    with pytest.raises(ValueError, match="same .T, J, 3. shape"):
        score(obj, obj, s["joints"], s["joints"][:, 0])
    with pytest.raises(ValueError, match="same .T, J, 3. shape"):
        score(obj, obj, s["joints"][:, 0], s["joints"])
    for fn in (mpjpe_root_aligned, pa_mpjpe):
        with pytest.raises(ValueError, match="same shape"):
            fn(s["joints"], s["joints"][:, 0])
    assert score(obj, obj, s["joints"], s["joints"])["hand"].aligned["pa_mpjpe"] == pytest.approx(0.0, abs=1e-12)


def test_mismatched_joint_sets_are_refused() -> None:
    s = _scene()
    with pytest.raises(ValueError, match=r"same joint set and order"):
        score(s["poses"], s["poses"], s["joints"][:, :16], s["joints"])
    with pytest.raises(ValueError, match="same shape"):
        pa_mpjpe(s["joints"][:, :16], s["joints"])


def test_object_may_mix_poses_and_centres() -> None:
    s = _scene()
    b = score(s["poses"], s["poses"][:, :3, 3])["object"]
    assert b.e_par == pytest.approx(0.0, abs=1e-15) and b.lam == pytest.approx(1.0, rel=1e-15)


def test_zero_filled_prediction_warns() -> None:
    gt = _traj()
    pred = 1.05 * gt
    pred[4] = 0.0
    with pytest.warns(UserWarning, match="zero-filled"):
        score(pred, gt)
