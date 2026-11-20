"""Tests for the D6 `qc` and `retarget` stage workers: artifact plumbing, `qc.json`, `retarget.json` and the fallback
marker (artifact files and fakes only; no model, no GPU, no MuJoCo scene)."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import fastdo.qc.export as export_module
from fastdo.engine import workers_retarget
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, run_clip
from fastdo.engine.worker_support import load_npz, save_npz
from fastdo.engine.workers import (
    RetargetStageConfig,
    make_qc_worker,
    make_retarget_worker,
    read_solve_artifact,
)
from fastdo.engine.workers_qc import decode_timestamps
from fastdo.io.clip_record import ClipRecord
from fastdo.qc import AcceptanceRule, ConformalCalibration, QcFeatures
from fastdo.qc.export import (
    DEFAULT_GRAVITY_CAM,
    gravity_aligned_cam_to_world,
    points_to_world,
    pose_to_world,
)
from fastdo.retarget.kinematic import KinematicTrajectory, RetargetConfig
from fastdo.retarget.wrist_target import WRIST_TARGETS, wrist_target_poses
from tests.engine.fakes import random_frames, write_decode
from tests.engine.fakes_qc import IMAGE_H, IMAGE_W, write_solved_clip
from tests.qc.records import CapturingKinematic, rigid

PASSING = QcFeatures(reprojection_iou=0.9, laplace_std_ok_frac=1.0, ot_residual_rms_frac=0.02, penetration_depth_m=0.0,
                     track_inlier_ratio=0.9)
STAGE_CFG = RetargetStageConfig(scene_xml="/nonexistent/scene.xml", mano_right_pkl="/fake/MANO_RIGHT.pkl",
                                mano_left_pkl="/fake/MANO_LEFT.pkl")


def test_retarget_stage_config_from_the_hydra_block() -> None:
    root = OmegaConf.load(Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml")
    cfg = RetargetStageConfig.from_mapping(root.retarget_stage)
    assert cfg.scene_xml == str(root.isolation.demo_scene_xml) and cfg.mano_right_pkl.endswith("MANO_RIGHT.pkl")
    assert cfg.mano_left_pkl.endswith("MANO_LEFT.pkl") and cfg.side is None and cfg.reference_frequency is None
    assert (cfg.mano_right_pkl, cfg.mano_left_pkl) == (str(root.isolation.mano_right_pkl), str(root.isolation.mano_left_pkl))
    assert Path(cfg.mano_right_pkl).parent == Path(str(root.isolation.mano_dir))
    assert Path(cfg.mano_left_pkl).parent == Path(str(root.isolation.mano_left_dir))    # HaWoR's data_left tree
    assert cfg.retarget == RetargetConfig() and cfg.mano_pkl_path("right") == cfg.mano_right_pkl
    assert root.retarget_stage.wrist_target == "ik_site" and cfg.wrist_target == "ik_site" == STAGE_CFG.wrist_target
    custom = RetargetStageConfig.from_mapping({"scene_xml": "s.xml", "mano_pkl_path": {"right": "r.pkl"},
                                               "side": "right", "wrist_target": "mcp", "reference_frequency": 50,
                                               "retarget": {"seed": 3}})
    assert (custom.reference_frequency, custom.retarget.seed, custom.mano_left_pkl, custom.wrist_target) == \
        (50.0, 3, "", "mcp")
    with pytest.raises(ValueError):
        custom.mano_pkl_path("left")
    with pytest.raises(ValueError):
        RetargetStageConfig(scene_xml="s.xml", mano_right_pkl="r.pkl", side="both")
    with pytest.raises(ValueError, match="wrist_target"):
        RetargetStageConfig(scene_xml="s.xml", mano_right_pkl="r.pkl", wrist_target="palm")


def _rule(accepting: bool) -> AcceptanceRule:
    sign = 1.0 if accepting else -1.0
    return AcceptanceRule(calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
                          logistic_weights=sign * np.array([0.0, 2.0, 1.0, -1.0, -1.0, 1.0]), feature_mean=np.zeros(5),
                          feature_std=np.ones(5))


class _CaptureFeatures:
    def __init__(self) -> None:
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> QcFeatures:
        self.args, self.kwargs = args, kwargs
        return PASSING


def test_qc_worker_feeds_artifacts_to_features_and_exports_the_record(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    right_valid = np.array([True, True, False, True, True, True])
    solution = write_solved_clip(clip, right_valid=right_valid)
    capture = _CaptureFeatures()

    make_qc_worker(_rule(True), features_fn=capture, laplace_std_threshold=0.02)("clip0", clip)

    masks, poses, scale, mesh, covariance, per_frame_cost, K, diameter, hand_verts, sdf, track_weight = capture.args
    assert capture.kwargs == {"laplace_std_threshold": 0.02}
    assert torch.equal(masks, torch.from_numpy(load_npz(clip / "mask.npz")["masks"]))
    assert torch.equal(poses, solution.poses) and scale == solution.scale and torch.equal(covariance, solution.covariance)
    assert torch.equal(per_frame_cost, solution.per_frame_cost) and K.shape == (3, 3) and float(K[0, 0]) == 40.0
    points = mesh.points.double()
    assert diameter == pytest.approx(0.2 * float((points.max(0).values - points.min(0).values).norm()), rel=1e-12)
    hands = load_npz(clip / "hands.npz")
    assert hand_verts.shape == (6, 2 * 778, 3)
    assert torch.equal(hand_verts[:, :778], torch.from_numpy(hands["left_vertices"]).double())
    assert bool(hand_verts[2, 778:].isnan().all()) and not bool(hand_verts[[0, 1, 3, 4, 5], 778:].isnan().any())
    assert float(sdf.step) == pytest.approx(1.0 / 3.0) and track_weight.shape == (6, 16)

    qc = json.loads((clip / "qc.json").read_text())
    assert qc["accepted"] is True and 0.0 <= qc["confidence"] <= 1.0 and qc["diameter_m"] == diameter
    assert (qc["camera_source"], qc["gravity_source"]) == ("gravity_aligned_static", "default")
    record = ClipRecord.load(clip / "clip_record.npz")
    assert record.frame_idx.tolist() == [0, 3, 6, 9, 12, 15] and np.allclose(record.timestamps, record.frame_idx / 30.0)
    assert np.array_equal(record.gravity_cam, DEFAULT_GRAVITY_CAM) and record.object_mesh_path == str(clip / "mesh.obj")
    assert np.array_equal(record.cam_to_world, gravity_aligned_cam_to_world(torch.from_numpy(DEFAULT_GRAVITY_CAM), 6)
                          .numpy().astype(np.float32))
    assert record.hand_right is not None and record.hand_right.valid.tolist() == right_valid.tolist()
    assert np.array_equal(record.confidence, solution.confidence.numpy()) and np.isnan(record.gate_score)


def test_qc_worker_runs_the_real_features_and_reads_geometry_camera(tmp_path: Path) -> None:
    clip = tmp_path / "clip1"
    write_solved_clip(clip, seed=1)
    cam_to_world, gravity = rigid(6, 3).float(), np.array([0.0, -0.8, 0.6], dtype=np.float32)
    save_npz(clip / "geometry.npz", frame_idx=3 * np.arange(6, dtype=np.int64), cam_to_world=cam_to_world.numpy(),
             gravity_cam=gravity)
    (clip / "gate.json").write_text(json.dumps({"score": 0.625}))

    make_qc_worker(_rule(True))("clip1", clip)

    qc = json.loads((clip / "qc.json").read_text())
    assert set(qc["features"]) == set(PASSING.__dataclass_fields__) and all(np.isfinite(list(qc["features"].values())))
    assert (qc["camera_source"], qc["gravity_source"]) == ("geometry", "geometry")
    assert qc["accepted"] is True                    # the record assertions below must not become vacuous
    record = ClipRecord.load(clip / "clip_record.npz")
    assert np.array_equal(record.cam_to_world, cam_to_world.numpy()) and np.array_equal(record.gravity_cam, gravity)
    assert record.gate_score == 0.625


def test_qc_worker_requires_hands_in_the_projection_camera(tmp_path: Path) -> None:
    """`hands.npz` outputs must be in the camera the qc stage uses (`intrinsics.npz`). A recorded camera that is not
    K, or HaWoR outputs without a recorded correction under a principal point more than 1 px off centre, is an error."""
    from fastdo.geometry.intrinsics import camera_matrix, write_intrinsics
    from fastdo.hands.hand_camera import CAMERA_KEYS

    clip = tmp_path / "clip_cam"
    write_solved_clip(clip, seed=4)
    write_intrinsics(clip, camera_matrix(40.0, 40.0, 4.0, 20.0), "given")          # hands.npz was written under 15.5, 11.5
    with pytest.raises(ValueError, match="camera"):
        make_qc_worker(_rule(True), features_fn=lambda *a, **k: PASSING)("clip_cam", clip)
    hands = load_npz(clip / "hands.npz")
    save_npz(clip / "hands.npz", **{k: v for k, v in hands.items() if k not in CAMERA_KEYS})   # no recorded camera
    with pytest.raises(ValueError, match="principal point"):
        make_qc_worker(_rule(True), features_fn=lambda *a, **k: PASSING)("clip_cam", clip)
    assert not (clip / "qc.json").exists() or not json.loads((clip / "qc.json").read_text()).get("accepted", False)


def test_qc_worker_rejection_removes_a_stale_record(tmp_path: Path) -> None:
    clip = tmp_path / "clip2"
    write_solved_clip(clip, seed=2)
    (clip / "clip_record.npz").write_bytes(b"stale")
    make_qc_worker(_rule(False), features_fn=lambda *a, **k: PASSING)("clip2", clip)
    qc = json.loads((clip / "qc.json").read_text())
    assert qc["accepted"] is False and 0.0 <= qc["confidence"] <= 1.0 and qc["record"] is None
    assert not (clip / "clip_record.npz").exists()


def test_qc_worker_requires_matching_rows(tmp_path: Path) -> None:
    clip = tmp_path / "clip3"
    write_solved_clip(clip, seed=3)
    hands = load_npz(clip / "hands.npz")
    save_npz(clip / "hands.npz", **{**hands, "frame_idx": hands["frame_idx"] + 1})
    with pytest.raises(ValueError, match="hands.npz"):
        make_qc_worker(_rule(True), features_fn=lambda *a, **k: PASSING)("clip3", clip)
    assert read_solve_artifact(clip).frame_idx.tolist() == [0, 3, 6, 9, 12, 15]


def test_decode_timestamps_use_the_stored_stride(tmp_path: Path) -> None:
    """A one-row decode has no row step, so without the stored `stride` its source rate cannot be recovered."""
    one, many = random_frames(1, IMAGE_H, IMAGE_W, 0), random_frames(4, IMAGE_H, IMAGE_W, 0)
    write_decode(tmp_path / "one", one, frame_idx=[12], fps=10.0, stride=3)
    frame_idx, timestamps = decode_timestamps(tmp_path / "one")
    assert frame_idx.tolist() == [12] and timestamps.tolist() == [12 / 30.0]
    write_decode(tmp_path / "one_without_stride", one, frame_idx=[12], fps=10.0)
    with pytest.raises(ValueError, match="one-row"):
        decode_timestamps(tmp_path / "one_without_stride")
    write_decode(tmp_path / "rows_without_stride", many, frame_idx=[0, 3, 6, 9], fps=10.0)
    assert np.allclose(decode_timestamps(tmp_path / "rows_without_stride")[1], [0.0, 0.1, 0.2, 0.3], rtol=0.0, atol=1e-12)
    write_decode(tmp_path / "rows_off_stride", many, frame_idx=[0, 2, 4, 6], fps=10.0, stride=3)
    with pytest.raises(ValueError, match="stride"):
        decode_timestamps(tmp_path / "rows_off_stride")


def _accepted_clip(tmp_path: Path, right_valid: np.ndarray | None = None) -> Path:
    clip = tmp_path / "clip0"
    write_solved_clip(clip, right_valid=right_valid)
    make_qc_worker(_rule(True), features_fn=lambda *a, **k: PASSING)("clip0", clip)
    return clip


def _zero_labels(path: str) -> torch.Tensor:
    return torch.zeros(778, dtype=torch.int8)


def _forbidden_solve(calls: list[Any]) -> Callable[..., KinematicTrajectory]:
    """A `stage_fn` that records a call and fails it: the tests using it assert that it is never reached."""
    def solve(*args: Any, **kwargs: Any) -> KinematicTrajectory:
        calls.append(args)
        raise AssertionError("the solve must not run")
    return solve


def _no_solve(calls: list[Any]) -> Any:
    return make_retarget_worker(STAGE_CFG, scene=cast(Any, object()), stage_fn=_forbidden_solve(calls),
                                label_fn=_zero_labels)


def test_retarget_worker_skips_a_clip_rejected_by_qc(tmp_path: Path) -> None:
    clip = tmp_path / "clip0"
    write_solved_clip(clip)
    make_qc_worker(_rule(False), features_fn=lambda *a, **k: PASSING)("clip0", clip)
    (clip / "trajectory_kinematic.npz").write_bytes(b"stale")
    calls: list[Any] = []

    make_retarget_worker(STAGE_CFG, stage_fn=_forbidden_solve(calls))("clip0", clip)   # the scene never loads either

    report = json.loads((clip / "retarget.json").read_text())
    assert report["status"] == "skipped" and report["retarget_fallback_needed"] is False and calls == []
    assert not (clip / "trajectory_kinematic.npz").exists()

    accepted = _accepted_clip(tmp_path / "zero_confidence")        # accepted, but the record's confidence is all zero
    record = ClipRecord.load(accepted / "clip_record.npz")
    dataclasses.replace(record, confidence=np.zeros_like(record.confidence)).save(accepted / "clip_record.npz")
    _no_solve(calls)("clip0", accepted)
    report = json.loads((accepted / "retarget.json").read_text())
    assert (report["status"], report["reason"], calls) == ("skipped", "mean tracker confidence is not positive", [])


def test_retarget_worker_writes_full_rate_reference_and_masked_counts(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _accepted_clip(tmp_path, right_valid=np.array([True, True, False, True, True, True]))
    hands = load_npz(clip / "hands.npz")
    hands["right_vertices"][2] = np.nan          # a masked row: its content must not reach the solve (nor fail the checks)
    save_npz(clip / "hands.npz", **hands)
    label_paths: list[str] = []

    def fake_labels(path: str) -> torch.Tensor:
        label_paths.append(str(path))
        return torch.zeros(778, dtype=torch.int8)

    monkeypatch.setattr(workers_retarget, "mano_finger_labels", fake_labels)
    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)

    make_retarget_worker(STAGE_CFG, scene=cast(Any, object()))("clip0", clip)

    (args, kwargs), = spy.calls
    assert label_paths == [STAGE_CFG.mano_right_pkl] and args[9] == 30.0
    report = json.loads((clip / "retarget.json").read_text())
    assert (report["status"], report["side"], report["retarget_fallback_needed"]) == ("done", "right", False)
    assert (report["n_rows"], report["n_reference_frames"], report["masked_rows"], report["masked_frames"]) == (6, 16, 1, 5)
    assert report["frequency"] == 30.0 and report["trajectory"] == "trajectory_kinematic.npz"
    trajectory = load_npz(clip / "trajectory_kinematic.npz")
    assert trajectory["qpos"].shape == (16, 35) and float(trajectory["frequency"]) == 30.0
    assert (~kwargs["frame_valid"]).sum() == 5 and bool(torch.isfinite(args[3]).all())

    record = ClipRecord.load(clip / "clip_record.npz")
    verts_world = points_to_world(torch.from_numpy(record.cam_to_world).double(),
                                  torch.from_numpy(load_npz(clip / "hands.npz")["right_vertices"]).double())
    valid_rows = np.array([0, 1, 3, 4, 5])
    assert (args[3][3 * valid_rows] - verts_world[valid_rows]).abs().max() <= 1e-9     # hands.npz vertices, not ClipRecord


def test_retarget_worker_masks_rows_whose_object_is_unobserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row with zero object support (`confidence == 0`: below `MIN_VALID_OBS`, the solver's "unobserved" flag) carries
    a pose made only by the temporal priors; it is masked exactly like a hand-invalid row. On the Ego4D pilot
    ~30% of emitted rows had no support at all, some with the object extrapolated behind the camera."""
    clip = _accepted_clip(tmp_path)                                # every hand row valid
    record = ClipRecord.load(clip / "clip_record.npz")
    confidence = record.confidence.copy()
    confidence[2] = 0.0
    object_pose = record.object_pose.copy()
    object_pose[2, 2, 3] = -0.2                                    # an extrapolated pose, behind the camera
    dataclasses.replace(record, confidence=confidence, object_pose=object_pose).save(clip / "clip_record.npz")
    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)

    make_retarget_worker(STAGE_CFG, scene=cast(Any, object()), label_fn=_zero_labels)("clip0", clip)

    (_args, kwargs), = spy.calls
    report = json.loads((clip / "retarget.json").read_text())
    assert (report["status"], report["masked_rows"], report["masked_frames"]) == ("done", 1, 5)
    assert (~kwargs["frame_valid"]).sum() == 5


@pytest.mark.parametrize("configured", [None, "ik_site", "mcp"])
def test_retarget_worker_selects_the_configured_wrist_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            configured: str | None) -> None:
    """The IK-site target unless `wrist_target: mcp`; `retarget.json` records the choice."""
    clip = _accepted_clip(tmp_path)
    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)
    cfg = STAGE_CFG if configured is None else dataclasses.replace(STAGE_CFG, wrist_target=configured)

    make_retarget_worker(cfg, scene=cast(Any, object()), label_fn=_zero_labels)("clip0", clip)

    (args, _), = spy.calls
    record = ClipRecord.load(clip / "clip_record.npz")
    assert record.hand_right is not None
    c2w, joints = torch.from_numpy(record.cam_to_world).double(), torch.from_numpy(record.hand_right.joints).double()
    rotvec = torch.from_numpy(np.ascontiguousarray(record.hand_right.params[:, :3])).double()
    targets = {name: pose_to_world(c2w, wrist_target_poses(name, joints, rotvec)) for name in WRIST_TARGETS}
    chosen = configured or "ik_site"
    other = "mcp" if chosen == "ik_site" else "ik_site"
    key_rows = args[1][3 * np.arange(6)]
    assert (key_rows - targets[chosen]).abs().max() <= 1e-6 and (key_rows - targets[other]).abs().max() > 1e-2
    assert json.loads((clip / "retarget.json").read_text())["wrist_target"] == chosen


@pytest.mark.parametrize("outcome", ["raises", "nan_qpos", "inf_qvel"])
def test_retarget_worker_failure_marks_stage_failed_with_fallback_marker(tmp_path: Path, outcome: str) -> None:
    """A solve that raises, or returns without raising but non-finite, takes the failed / fallback path."""
    clip = _accepted_clip(tmp_path)
    (clip / "trajectory_kinematic.npz").write_bytes(b"stale")

    def solver(*args: Any, **kwargs: Any) -> KinematicTrajectory:
        if outcome == "raises":
            raise FloatingPointError("LM diverged")
        qpos, qvel = np.zeros((16, 35)), np.zeros((16, 34))
        if outcome == "nan_qpos":
            qpos[7, 3] = np.nan
        else:
            qvel[7, 3] = np.inf
        return KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=30.0)

    worker = make_retarget_worker(STAGE_CFG, scene=cast(Any, object()), stage_fn=solver, label_fn=_zero_labels)
    manifest = ClipManifest.create(["clip0"], ["source.mp4"], [6])
    run_clip("clip0", EngineConfig(artifact_root=tmp_path, stages=("retarget",)), {"retarget": worker}, manifest)

    expected = "LM diverged" if outcome == "raises" else "non-finite"
    status = manifest.status("clip0", "retarget")
    assert status.state == "failed" and expected in status.error
    report = json.loads((clip / "retarget.json").read_text())
    assert (report["status"], report["retarget_fallback_needed"], report["error_type"]) == \
        ("failed", True, "FloatingPointError")
    assert expected in report["error"] and not (clip / "trajectory_kinematic.npz").exists()


@pytest.mark.parametrize(("defect", "match"), [
    ("rows", "right_vertices"), ("vertex_count", "right_vertices"), ("nan_vertex_on_valid_row", "right_vertices"),
    ("nan_joint_on_valid_row", "hand_right.joints"), ("nan_global_orient_on_valid_row", "global orient"),
    ("nan_object_pose", "object_pose"), ("nan_camera", "cam_to_world"), ("hands_camera", "camera"),
])
def test_retarget_worker_contract_errors_fail_without_the_fallback_marker(tmp_path: Path, defect: str, match: str) -> None:
    """Malformed vertices, or a non-finite record field the stage reads, are data bugs, not solver failures: no
    fallback marker, solver never called. A NaN joint on a valid row used to fail inside the upsampling and
    set the marker."""
    clip = _accepted_clip(tmp_path)
    hands, record = load_npz(clip / "hands.npz"), ClipRecord.load(clip / "clip_record.npz")
    assert record.hand_right is not None
    verts, joints, params = hands["right_vertices"], record.hand_right.joints.copy(), record.hand_right.params.copy()
    pose, camera = record.object_pose.copy(), record.cam_to_world.copy()
    if defect == "rows":
        verts = verts[:-1]
    elif defect == "vertex_count":
        verts = verts[:, :-1]
    elif defect == "nan_vertex_on_valid_row":
        verts[4, 10, 1] = np.nan
    elif defect == "nan_joint_on_valid_row":
        joints[4, 9, 1] = np.nan
    elif defect == "nan_global_orient_on_valid_row":
        params[4, 1] = np.nan
    elif defect == "nan_object_pose":
        pose[1, 0, 3] = np.nan
    elif defect == "hands_camera":                   # hands.npz outputs are not in this clip's camera
        from fastdo.geometry.intrinsics import camera_matrix, write_intrinsics
        write_intrinsics(clip, camera_matrix(40.0, 40.0, 4.0, 20.0), "given")
    else:
        camera[5, 2, 2] = np.nan
    save_npz(clip / "hands.npz", **{**hands, "right_vertices": verts})
    dataclasses.replace(record, object_pose=pose, cam_to_world=camera, hand_right=dataclasses.replace(
        record.hand_right, joints=joints, params=params)).save(clip / "clip_record.npz")
    (clip / "trajectory_kinematic.npz").write_bytes(b"stale")
    calls: list[Any] = []

    with pytest.raises(ValueError, match=match):
        _no_solve(calls)("clip0", clip)

    report = json.loads((clip / "retarget.json").read_text())
    assert (report["status"], report["retarget_fallback_needed"], report["error_type"]) == ("failed", False, "ValueError")
    assert calls == [] and not (clip / "trajectory_kinematic.npz").exists()


def test_retarget_worker_input_composition_errors_fail_without_the_fallback_marker(tmp_path: Path,
                                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Any error while preparing the solver inputs, not only ValueError, leaves the marker False."""
    clip = _accepted_clip(tmp_path)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(workers_retarget, "prepare_retarget_stage", broken)
    calls: list[Any] = []
    with pytest.raises(np.linalg.LinAlgError):
        _no_solve(calls)("clip0", clip)
    report = json.loads((clip / "retarget.json").read_text())
    assert (report["status"], report["retarget_fallback_needed"], report["error_type"]) == ("failed", False, "LinAlgError")
    assert calls == []


def test_retarget_worker_warns_when_reference_frequency_relabels_the_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         caplog: pytest.LogCaptureFixture) -> None:
    """`reference_frequency` labels the source-frame grid; a label >1% off the 30 fps source rate warns."""
    clip = _accepted_clip(tmp_path)
    monkeypatch.setattr(export_module, "retarget_kinematic", CapturingKinematic())
    caplog.set_level(logging.WARNING, logger="fastdo.engine.workers_retarget")
    for label, warns in ((50.0, True), (30.2, False), (None, False)):
        caplog.clear()
        cfg = dataclasses.replace(STAGE_CFG, reference_frequency=label)
        make_retarget_worker(cfg, scene=cast(Any, object()), label_fn=_zero_labels)("clip0", clip)
        report = json.loads((clip / "retarget.json").read_text())
        expected = 30.0 if label is None else label
        assert report["status"] == "done" and report["source_frequency"] == pytest.approx(30.0, rel=1e-12)
        assert report["frequency"] == pytest.approx(expected, rel=1e-12)
        assert float(load_npz(clip / "trajectory_kinematic.npz")["frequency"]) == pytest.approx(expected, rel=1e-12)
        assert load_npz(clip / "trajectory_kinematic.npz")["qpos"].shape == (16, 35)      # the grid never changes
        assert any("relabels" in r.getMessage() for r in caplog.records) is warns, label


# --- a flagged window: "Its labels are then computed from the pts." ------------------------------------

def test_an_unflagged_window_still_labels_from_the_arithmetic(tmp_path: Path) -> None:
    """1.2 chooses DERIVED normally and MEASURED only on disagreement. The derived value is exactly uniform
    where pts carry container jitter, so recording pts must not quietly change every clean window's labels."""
    frames = random_frames(4, IMAGE_H, IMAGE_W, 0)
    write_decode(tmp_path / "clean", frames, frame_idx=[0, 3, 6, 9], fps=10.0, stride=3,
                 pts=[0.0001, 0.1002, 0.1999, 0.2998], timebase_mismatch=False)
    _, timestamps = decode_timestamps(tmp_path / "clean")
    assert np.allclose(timestamps, [0.0, 0.1, 0.2, 0.3], rtol=0.0, atol=1e-12)


def test_a_flagged_window_labels_from_the_recorded_pts(tmp_path: Path) -> None:
    """The one-frame-shift case, in numbers: four Ego4D videos start at 21.029 ms, and a window decoded ONE FRAME OFF
    writes exactly the `frame_idx` the audit expects. Its rows really sit at 0.021 + (n+1)/30 s; the
    arithmetic claims n/30. Only the measured pts describe the frames that are actually in the file, so a
    flagged window labels from them -- the arithmetic is the quantity the flag disputes."""
    fps, start = 30.0, 0.021029
    idx = [0, 1, 2, 3]
    shifted = [start + (n + 1) / fps for n in idx]                   # where the decoded rows really are
    frames = random_frames(4, IMAGE_H, IMAGE_W, 0)
    write_decode(tmp_path / "shifted", frames, frame_idx=idx, fps=fps, stride=1,
                 pts=shifted, timebase_mismatch=True)

    frame_idx, timestamps = decode_timestamps(tmp_path / "shifted")
    assert frame_idx.tolist() == idx
    assert np.allclose(timestamps, shifted, rtol=0.0, atol=1e-12)
    derived = np.asarray(idx) / fps
    assert not np.allclose(timestamps, derived, atol=1e-3)           # and it is NOT the arithmetic
    assert np.max(np.abs(timestamps - derived)) > 1 / fps            # 1.2's own bound, exceeded


def test_a_flagged_window_without_pts_refuses(tmp_path: Path) -> None:
    """1.2 says a flagged window labels from the pts. If the flag is set and the pts are missing or the
    wrong length, there is nothing to label it with -- and silently falling back to the arithmetic would
    restore exactly the behaviour the flag exists to prevent, invisibly."""
    frames = random_frames(4, IMAGE_H, IMAGE_W, 0)
    write_decode(tmp_path / "no_pts", frames, frame_idx=[0, 1, 2, 3], fps=30.0, stride=1,
                 timebase_mismatch=True)
    with pytest.raises(ValueError, match="absent"):
        decode_timestamps(tmp_path / "no_pts")

    write_decode(tmp_path / "short_pts", frames, frame_idx=[0, 1, 2, 3], fps=30.0, stride=1,
                 pts=[0.0, 0.033], timebase_mismatch=True)
    with pytest.raises(ValueError, match="not"):
        decode_timestamps(tmp_path / "short_pts")


def test_an_archive_without_the_flag_is_unchanged(tmp_path: Path) -> None:
    """The whole-file decode path writes no pts and no flag, and archives predate both. They must keep
    labelling from the arithmetic rather than raising."""
    frames = random_frames(4, IMAGE_H, IMAGE_W, 0)
    write_decode(tmp_path / "old", frames, frame_idx=[0, 3, 6, 9], fps=10.0, stride=3)
    assert np.allclose(decode_timestamps(tmp_path / "old")[1], [0.0, 0.1, 0.2, 0.3], rtol=0.0, atol=1e-12)


def test_a_flagged_window_with_a_stride_defect_still_raises(tmp_path: Path) -> None:
    """Unevenly spaced rows are a STRIDE defect, not a timebase one. Having pts to fall back on must not
    let that one through quietly -- two different failures, and only one of them 1.2 forgives."""
    frames = random_frames(4, IMAGE_H, IMAGE_W, 0)
    write_decode(tmp_path / "wonky", frames, frame_idx=[0, 2, 4, 6], fps=10.0, stride=3,
                 pts=[0.0, 0.2, 0.4, 0.6], timebase_mismatch=True)
    with pytest.raises(ValueError, match="stride"):
        decode_timestamps(tmp_path / "wonky")


# --- the solve can place geometry behind the camera, and nothing checked -------------------------

def test_min_camera_z_is_the_smallest_depth_over_frames_and_points() -> None:
    """`p = scale * R @ y + t`, so this is `min_k min_i p_z`. Computed without materialising the
    (T, N, 3) cloud -- only the z row of each rotation contributes."""
    from fastdo.engine.workers_qc import min_camera_z

    poses = np.stack([np.eye(4), np.eye(4)])
    poses[0, 2, 3], poses[1, 2, 3] = 1.0, 0.5           # two frames, translated along z
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, -0.1]])

    assert min_camera_z(poses, 1.0, points) == pytest.approx(0.4)     # 0.5 + 1.0 * (-0.1)
    assert min_camera_z(poses, 2.0, points) == pytest.approx(0.3)     # scale multiplies the offset
    assert min_camera_z(poses, 0.0, points) == pytest.approx(0.5)     # collapsed in place: p -> t


def test_total_collapse_is_safe_and_mild_collapse_is_not() -> None:
    """The result that decides the discriminator: at scale ~1e-28 the object shrinks IN PLACE, so
    `p -> t` and the points sit wherever the translation is -- in front. The lethal band is MILD
    collapse, where the object is still large enough to swing a point across the plane. So a gate keyed
    on recovered SCALE would have to fire on a narrow interior band while passing values a thousand
    times smaller; min z is the quantity that separates them."""
    from fastdo.engine.workers_qc import min_camera_z

    poses = np.stack([np.eye(4)])
    poses[0, 2, 3] = 0.66                                # the measured median translation
    points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    assert min_camera_z(poses, 1e-28, points) == pytest.approx(0.66)   # total collapse: SAFE
    assert min_camera_z(poses, 1e-3, points) > 0.0                     # still in front at 1e-3 here
    assert min_camera_z(poses, 1.0, points) < 0.0                      # a full-size object crosses it


def test_min_camera_z_refuses_a_malformed_input() -> None:
    from fastdo.engine.workers_qc import min_camera_z

    poses = np.stack([np.eye(4)])
    with pytest.raises(ValueError, match="mesh points must be"):
        min_camera_z(poses, 1.0, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="poses must be"):
        min_camera_z(np.eye(4), 1.0, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="at least one pose"):
        min_camera_z(poses, 1.0, np.zeros((0, 3)))
