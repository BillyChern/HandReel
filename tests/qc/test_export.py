"""Tests for `fastdo.qc.export`: `ClipRecord` export, the camera-to-world conversions, the world-frame
retargeting stage hook and the `qc` stage worker's `qc.json`.

CPU only. `retarget_kinematic` is replaced by a capturing fake. The engine-to-reference rate composition and the masking
of invalid hand rows are tested in `tests/qc/test_retarget_stage.py`, the solver's own masking in
`tests/retarget/test_kinematic_fast_mask.py`, and the workers' artifact plumbing in `tests/engine/test_workers_qc.py`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

import fastdo.qc.export as export_module
from fastdo.engine.workers import make_qc_worker
from fastdo.io.clip_record import ClipRecord
from fastdo.io.datasets.dexycb import _DEFAULT_GRAVITY_CAM
from fastdo.qc import AcceptanceRule, ConformalCalibration, QcFeatures
from fastdo.qc.export import (
    DEFAULT_GRAVITY_CAM,
    export_clip_record,
    gravity_aligned_cam_to_world,
    points_to_world,
    pose_to_world,
    run_retarget_stage,
)
from fastdo.retarget.kinematic import RetargetConfig
from fastdo.retarget.wrist_target import ik_site_wrist_poses
from tests.engine.fakes_qc import write_solved_clip
from tests.qc.records import (
    K_PIX,
    N_VERTS,
    CapturingKinematic,
    hand,
    hand_labels,
    hand_verts,
    make_record,
    rigid,
    solution,
    tiny_sdf,
)


def _assert_identical(a: ClipRecord, b: ClipRecord) -> None:
    for name in ("frame_idx", "timestamps", "intrinsics", "cam_to_world", "object_pose", "gravity_cam", "confidence"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), name
    assert (a.clip_id, a.object_mesh_path, a.object_scale, a.gate_score) == \
        (b.clip_id, b.object_mesh_path, b.object_scale, b.gate_score)
    for side in ("hand_left", "hand_right"):
        ha, hb = getattr(a, side), getattr(b, side)
        assert (ha is None) == (hb is None), side
        if ha is not None:
            for key in ("params", "betas", "trans", "joints", "valid"):
                x, y = getattr(ha, key), getattr(hb, key)
                assert x.dtype == y.dtype and np.array_equal(x, y), f"{side}.{key}"


def test_export_clip_record_validates_and_roundtrips(tmp_path: Path) -> None:
    t = 6
    sol = solution(t, seed=3)
    frame_idx = 12 + 3 * np.arange(t, dtype=np.int64)
    timestamps = frame_idx / 29.97
    cam_to_world = rigid(t, 4)
    gravity = torch.tensor([0.1, -0.98, 0.17])
    left, right = hand(t, 5), hand(t, 6, valid=np.array([True, True, False, True, True, True]))

    record = export_clip_record("clip_a", sol, "/artifacts/clip_a/mesh.obj", K_PIX, cam_to_world, gravity, left, right,
                                0.73, frame_idx, timestamps)
    record.validate()
    assert record.n_frames == t and record.clip_id == "clip_a" and record.object_mesh_path == "/artifacts/clip_a/mesh.obj"
    assert record.object_scale == sol.scale and record.gate_score == 0.73
    assert np.array_equal(record.object_pose, sol.poses.numpy()) and np.array_equal(record.confidence, sol.confidence.numpy())
    assert np.array_equal(record.intrinsics, K_PIX.numpy())
    assert np.array_equal(record.cam_to_world, cam_to_world.numpy().astype(np.float32))
    assert np.array_equal(record.gravity_cam, gravity.numpy()) and np.array_equal(record.timestamps, timestamps)
    assert record.hand_left is left and record.hand_right is right

    saved = tmp_path / "exported.npz"
    again = export_clip_record("clip_a", sol, "/artifacts/clip_a/mesh.obj", K_PIX, cam_to_world, gravity, left, right,
                               0.73, frame_idx, timestamps, path=saved)
    _assert_identical(record, again)
    _assert_identical(record, ClipRecord.load(saved))
    record.save(tmp_path / "record.npz")
    _assert_identical(record, ClipRecord.load(tmp_path / "record.npz"))


def test_export_clip_record_requires_gravity_cam() -> None:
    t = 4
    with pytest.raises(TypeError, match="gravity_cam"):
        export_clip_record("c", solution(t), "m.obj", K_PIX, rigid(t, 0),  # type: ignore[call-arg]
                           hand_left=None, hand_right=None, gate_score=0.5, frame_idx=np.arange(t),
                           timestamps=np.arange(t) / 10.0)


def test_export_clip_record_without_hands(tmp_path: Path) -> None:
    t = 5
    frame_idx = np.arange(t, dtype=np.int64) * 3
    record = export_clip_record("bare", solution(t), "m.obj", K_PIX, rigid(t, 1), torch.from_numpy(DEFAULT_GRAVITY_CAM),
                                None, None, 0.1, frame_idx, frame_idx / 30.0)
    record.validate()
    assert record.hand_left is None and record.hand_right is None
    assert np.array_equal(DEFAULT_GRAVITY_CAM, _DEFAULT_GRAVITY_CAM) and DEFAULT_GRAVITY_CAM.dtype == np.float32
    record.save(tmp_path / "bare.npz")
    _assert_identical(record, ClipRecord.load(tmp_path / "bare.npz"))


def test_export_clip_record_rejects_hand_rows_that_differ_from_solution_rows() -> None:
    """hands.npz rows are the decode rows, so the hand T must equal the solution T."""
    t = 5
    with pytest.raises(ValueError, match="hand_right"):
        export_clip_record("c", solution(t), "m.obj", K_PIX, rigid(t, 1), torch.from_numpy(DEFAULT_GRAVITY_CAM), None,
                           hand(t + 1, 2), 0.1, np.arange(t), np.arange(t) / 10.0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_frame_conversion_roundtrips_to_known_world_values(dtype: torch.dtype) -> None:
    """float64 must round-trip at 1e-10 (measured 3e-16 poses, 2e-15 points): a float32 cast inside a float64
    conversion errs by ~1e-7, which the float32 tolerance 1e-5 would miss. The whisking reference is that sensitive:
    float32-rounded world knots alone move it by up to 0.25 rad, the whole float32 record path by 0.45 rad
    (measured)."""
    tol = 1e-10 if dtype == torch.float64 else 1e-5
    t = 5
    obj_world, wrist_world = rigid(t, 1), rigid(t, 2)
    verts_world = torch.from_numpy(np.random.default_rng(0).normal(size=(t, N_VERTS, 3)))
    cam_to_world = rigid(t, 3)
    world_to_cam = torch.linalg.inv(cam_to_world)
    obj_cam, wrist_cam = world_to_cam @ obj_world, world_to_cam @ wrist_world
    verts_cam = torch.einsum("tij,tvj->tvi", world_to_cam[:, :3, :3], verts_world) + world_to_cam[:, None, :3, 3]
    assert (obj_cam - obj_world).abs().max() > 0.1 and (verts_cam - verts_world).abs().max() > 0.1

    c2w = cam_to_world.to(dtype)
    for cam, world in ((obj_cam, obj_world), (wrist_cam, wrist_world)):
        out = pose_to_world(c2w, cam.to(dtype))
        assert out.shape == (t, 4, 4) and out.dtype == dtype
        assert (out.double() - world).abs().max() <= tol
    points = points_to_world(c2w, verts_cam.to(dtype))
    assert points.shape == (t, N_VERTS, 3) and points.dtype == dtype
    assert (points.double() - verts_world).abs().max() <= tol
    with pytest.raises(ValueError):
        pose_to_world(c2w[:-1], obj_cam.to(dtype))
    with pytest.raises(ValueError):
        points_to_world(c2w, verts_cam.to(dtype)[..., :2])


def test_gravity_aligned_cam_to_world_maps_up_to_world_z() -> None:
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    for up in ([0.0, -1.0, 0.0], [0.2, -0.9, 0.3], [0.0, 0.0, -1.0], [0.0, 0.0, 2.0]):
        c2w = gravity_aligned_cam_to_world(torch.tensor(up, dtype=torch.float64), 3)
        assert c2w.shape == (3, 4, 4) and c2w.dtype == torch.float64
        last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(3, 4)
        assert torch.equal(c2w[0], c2w[2]) and torch.equal(c2w[:, 3], last_row)
        R = c2w[0, :3, :3]
        assert (R @ R.T - torch.eye(3, dtype=torch.float64)).abs().max() <= 1e-12
        assert abs(float(torch.linalg.det(R)) - 1.0) <= 1e-12 and float(c2w[0, :3, 3].abs().max()) == 0.0
        u = torch.tensor(up, dtype=torch.float64)
        assert (R @ (u / u.norm()) - ez).abs().max() <= 1e-12, up
    R = gravity_aligned_cam_to_world(torch.from_numpy(DEFAULT_GRAVITY_CAM), 1)[0, :3, :3]
    assert (R @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) - torch.tensor([0.0, 1.0, 0.0])).abs().max() <= 1e-12
    with pytest.raises(ValueError):
        gravity_aligned_cam_to_world(torch.zeros(3), 2)


def test_run_retarget_stage_skips_rejected_clip(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    t = 5
    record = make_record(np.arange(t), hand_right=hand(t, 1), confidence=torch.zeros(t))
    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)
    caplog.set_level(logging.INFO, logger="fastdo.qc.export")
    out = run_retarget_stage(record, hand_verts(t, 2), hand_labels(), cast(Any, None), tiny_sdf(), torch.zeros(8, 3),
                             torch.zeros(8, 3), 30.0, RetargetConfig())
    assert out is None and spy.calls == []
    assert any("confidence" in r.getMessage() for r in caplog.records if r.name == "fastdo.qc.export")


def test_run_retarget_stage_calls_kinematic_with_world_frame_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consecutive source frames, so the engine rate is the reference rate and the conversion is all that happens."""
    t = 7
    record = make_record(40 + np.arange(t), seed=2, hand_right=hand(t, 3))
    assert record.hand_right is not None
    verts_cam, labels, cfg = hand_verts(t, 4), hand_labels(), RetargetConfig(seed=3)
    sdf, surface, normals, scene = tiny_sdf(), torch.zeros(8, 3), torch.ones(8, 3), cast(Any, object())
    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)

    out = run_retarget_stage(record, verts_cam, labels, scene, sdf, surface, normals, 30.0, cfg)

    assert len(spy.calls) == 1 and out is spy.results[0]
    args, kwargs = spy.calls[0]
    obj, wrist, scale, verts = args[0], args[1], args[2], args[3]
    assert obj.shape == wrist.shape == (t, 4, 4) and verts.shape == (t, N_VERTS, 3) and obj.dtype == torch.float64
    c2w = torch.from_numpy(record.cam_to_world).double()
    obj_cam = torch.from_numpy(record.object_pose).double()
    assert (obj - c2w @ obj_cam).abs().max() <= 1e-12
    assert (obj - obj_cam).abs().max() > 0.05          # the fixture's camera moves: camera-frame poses never reach it
    wrist_cam = ik_site_wrist_poses(torch.from_numpy(record.hand_right.joints).double(),       # the default target
                                    torch.from_numpy(np.ascontiguousarray(record.hand_right.params[:, :3])).double())
    assert (wrist - c2w @ wrist_cam).abs().max() <= 1e-12
    expected_verts = torch.einsum("tij,tvj->tvi", c2w[:, :3, :3], verts_cam.double()) + c2w[:, None, :3, 3]
    assert (verts - expected_verts).abs().max() <= 1e-12 and (verts - verts_cam.double()).abs().max() > 0.05
    assert scale == record.object_scale and args[4] is labels and args[5] is scene and args[6] is sdf
    assert args[7] is surface and args[8] is normals and args[9] == 30.0 and args[10] is cfg
    assert torch.equal(kwargs["frame_valid"], torch.ones(t, dtype=torch.bool))


def _rule(threshold: float, weights: list[float]) -> AcceptanceRule:
    return AcceptanceRule(calibration=ConformalCalibration(threshold=threshold, alpha=0.1, n_calibration=20),
                          logistic_weights=np.asarray(weights), feature_mean=np.zeros(5), feature_std=np.ones(5))


PASSING = QcFeatures(reprojection_iou=0.9, laplace_std_ok_frac=1.0, ot_residual_rms_frac=0.02, penetration_depth_m=0.0,
                     track_inlier_ratio=0.9)


def test_qc_worker_confidence_in_unit_interval(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    write_solved_clip(clip, t=6)
    worker = make_qc_worker(_rule(0.5, [0.0, 2.0, 1.0, -1.0, -1.0, 1.0]), features_fn=lambda *a, **k: PASSING)

    worker("clip0", clip)

    qc = json.loads((clip / "qc.json").read_text())
    assert qc["accepted"] is True and 0.0 <= qc["confidence"] <= 1.0
    assert qc["features"] == dataclasses.asdict(PASSING) and qc["record"] == "clip_record.npz"
    ClipRecord.load(clip / "clip_record.npz").validate()
