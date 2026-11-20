"""driftqc engine hooks: solve, qc and retarget on the driftqc prefix.

Clips are the 24-row synthetic fixture of tests/engine/fakes_driftqc.py, run through the real `driftqc` worker (so every
valid_range.json here is a record the driftqc writer produced): contraction None gives ACCEPT [0, 23], contraction 9
gives TRUNCATE [0, 7] and contraction 6 gives REJECT (driftqc.json only). The frozen files are read under FROZEN_ROOT
(env FASTDO_DRIFTQC_ROOT); a test skips only when they are absent.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import shutil
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.engine import workers_qc, workers_retarget, workers_solve
from fastdo.engine.batched_stages import SolveBatchStage
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.valid_range import DRIFTQC_FILE, VALID_RANGE_FILE, RowRange
from fastdo.engine.worker_support import load_npz, read_frame_idx, save_npz
from fastdo.engine.workers_driftqc import DriftQcStageConfig, DriftRejected, make_driftqc_worker
from fastdo.engine.workers_qc import (
    QcStageConfig,
    make_qc_worker,
    make_qc_worker_from_config,
    read_camera_frame,
    read_qc_inputs,
)
from fastdo.engine.workers_retarget import RetargetStageConfig, make_retarget_worker
from fastdo.engine.workers_solve import (
    finish_solve,
    make_solve_worker,
    prepare_solve,
    read_solve_inputs,
)
from fastdo.io.clip_record import ClipRecord
from fastdo.qc import AcceptanceRule, ConformalCalibration, QcFeatures
from fastdo.qc.rule_io import save_acceptance_rule
from fastdo.retarget.kinematic import KinematicTrajectory
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations, TrajectorySolution
from tests.engine.fakes_driftqc import N_ROWS, driftqc_clip, write_driftqc_solve_inputs
from tests.engine.fakes_solve import SMALL_SOLVE, RecordingSolve, fake_solution

FROZEN_ROOT = Path(os.environ.get("FASTDO_DRIFTQC_ROOT", "/opt/fastdo/handreel"))
FROZEN_ABSENT = ("frozen driftqc files absent (not shipped): set FASTDO_DRIFTQC_ROOT to a directory holding "
                 "temp/driftqc/constants_v2_2.json, temp/trackweight/constants_v2.json and "
                 "temp/trackweight/code/v1_weights.py")
CONSTANTS = FROZEN_ROOT / "temp/driftqc/constants_v2_2.json"
TRACKWEIGHT_CONSTANTS = FROZEN_ROOT / "temp/trackweight/constants_v2.json"
DETECTOR = FROZEN_ROOT / "temp/trackweight/code/v1_weights.py"
requires_frozen = pytest.mark.skipif(
    not (CONSTANTS.is_file() and TRACKWEIGHT_CONSTANTS.is_file() and DETECTOR.is_file()),
    reason=FROZEN_ABSENT,
)

PREFIX = 8  # TRUNCATE [0, 7] rows
_OBS_PER_ROW = ("masks", "hand_masks", "pointmaps", "tracks", "track_vis")
_PER_ROW_ARTIFACTS = ["mask.npz", "geometry.npz", "tracks.npz"]


@pytest.fixture(scope="module")
def driftqc_worker() -> StageFn:
    if not (CONSTANTS.is_file() and TRACKWEIGHT_CONSTANTS.is_file() and DETECTOR.is_file()):
        pytest.skip(FROZEN_ABSENT)
    return make_driftqc_worker(DriftQcStageConfig(enabled=True, constants_path=str(CONSTANTS),
                                                  trackweight_constants_path=str(TRACKWEIGHT_CONSTANTS),
                                                  detector_path=str(DETECTOR)))


def _driftqc_dir(root: Path, clip_id: str, contraction_from: int | None, worker: StageFn) -> Path:
    """The solve inputs of a 24-row clip plus the real driftqc worker's records (a REJECT is swallowed here)."""
    artifact_dir = write_driftqc_solve_inputs(root / clip_id, driftqc_clip(0), contraction_from)
    try:
        worker(clip_id, artifact_dir)
    except DriftRejected:
        assert contraction_from == 6
    return artifact_dir


def _range(artifact_dir: Path) -> list[int]:
    return list(json.loads((artifact_dir / VALID_RANGE_FILE).read_text())["range"])


def _solve_json(artifact_dir: Path) -> dict[str, Any]:
    return dict(json.loads((artifact_dir / "solve.json").read_text()))


def _without_driftqc(artifact_dir: Path, dest: Path) -> Path:
    """A copy of the clip dir with driftqc.json and valid_range.json removed (the disabled-run twin)."""
    shutil.copytree(artifact_dir, dest)
    for name in (DRIFTQC_FILE, VALID_RANGE_FILE):
        (dest / name).unlink(missing_ok=True)
    return dest


def _assert_obs_rows(obs: TrajectoryObservations, init_poses: Tensor, rows: int) -> None:
    for name in _OBS_PER_ROW:
        assert int(getattr(obs, name).shape[0]) == rows, name
    assert int(init_poses.shape[0]) == rows


def _slice_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    original = workers_solve.slice_artifact

    def spy(arrays: Any, artifact: str, rng: RowRange) -> dict[str, np.ndarray]:
        calls.append(artifact)
        return original(arrays, artifact, rng)

    monkeypatch.setattr(workers_solve, "slice_artifact", spy)
    return calls


@requires_frozen
def test_solve_on_prefix_uses_sliced_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            driftqc_worker: StageFn) -> None:
    artifact_dir = _driftqc_dir(tmp_path / "on", "clip1", 9, driftqc_worker)
    assert _range(artifact_dir) == [0, PREFIX - 1]
    decode = read_frame_idx(artifact_dir)
    full = read_solve_inputs(_without_driftqc(artifact_dir, tmp_path / "full" / "clip1"), SMALL_SOLVE.n_anchor_surface,
                             SMALL_SOLVE.seed)
    calls = _slice_spy(monkeypatch)

    sliced = read_solve_inputs(artifact_dir, SMALL_SOLVE.n_anchor_surface, SMALL_SOLVE.seed, driftqc_enabled=True)
    assert sliced.valid_range == RowRange(stop=PREFIX - 1, n_rows=N_ROWS) and full.valid_range is None
    assert calls == _PER_ROW_ARTIFACTS
    assert np.array_equal(sliced.frame_idx, decode[:PREFIX])
    for name in _OBS_PER_ROW:  # the prefix of the full inputs, never recomputed
        assert np.array_equal(getattr(sliced.observations.observations, name).numpy(),
                              getattr(full.observations.observations, name)[:PREFIX].numpy(), equal_nan=True), name
    assert np.array_equal(sliced.observations.track_vis.numpy(), full.observations.track_vis[:PREFIX].numpy())
    assert np.array_equal(sliced.observations.keyframe_points.numpy(), full.observations.keyframe_points.numpy(),
                          equal_nan=True)

    solve = RecordingSolve()
    make_solve_worker(SMALL_SOLVE, solve_fn=solve, driftqc_enabled=True)("clip1", artifact_dir)
    [(_, obs, init_poses, _, _)] = solve.calls
    _assert_obs_rows(obs, init_poses, PREFIX)
    assert np.array_equal(load_npz(artifact_dir / "solve.npz")["frame_idx"], decode[:PREFIX])
    meta = _solve_json(artifact_dir)
    assert meta["valid_range"] == [0, PREFIX - 1] and meta["n_rows"] == PREFIX


@requires_frozen
def test_accept_solve_arrays_equal_disabled(tmp_path: Path, driftqc_worker: StageFn) -> None:
    """G0a: ACCEPT takes the identity path, so the real CPU solve equals the disabled run on a copy without records."""
    with_records = _driftqc_dir(tmp_path / "on", "clip0", None, driftqc_worker)
    assert _range(with_records) == [0, N_ROWS - 1]
    without = _without_driftqc(with_records, tmp_path / "off" / "clip0")

    make_solve_worker(SMALL_SOLVE, driftqc_enabled=True)("clip0", with_records)
    make_solve_worker(SMALL_SOLVE)("clip0", without)

    npz_on, npz_off = load_npz(with_records / "solve.npz"), load_npz(without / "solve.npz")
    assert npz_on.keys() == npz_off.keys()
    assert int(npz_on["frame_idx"].size) == N_ROWS
    for key in npz_on.keys() - {"seconds"}:
        assert npz_on[key].dtype == npz_off[key].dtype and np.array_equal(npz_on[key], npz_off[key]), key
    meta_on, meta_off = _solve_json(with_records), _solve_json(without)
    assert "valid_range" not in meta_on and "valid_range" not in meta_off
    for meta in (meta_on, meta_off):
        meta["solver"].pop("seconds")
    assert meta_on == meta_off


@requires_frozen
def test_batched_solve_stage_inherits_slicing(tmp_path: Path, driftqc_worker: StageFn) -> None:
    truncated = _driftqc_dir(tmp_path, "clip1", 9, driftqc_worker)
    accepted = _driftqc_dir(tmp_path, "clip0", None, driftqc_worker)
    seen: list[tuple[TrajectoryObservations, Tensor]] = []

    def solve_many(meshes: list[MeshSamples], obs: list[TrajectoryObservations], inits: list[Tensor],
                   scales: list[float], config: SolverConfig) -> list[TrajectorySolution]:
        seen.extend(zip(obs, inits, strict=True))
        return [fake_solution(init, scale) for init, scale in zip(inits, scales, strict=True)]

    stage = SolveBatchStage(partial(prepare_solve, cfg=SMALL_SOLVE, driftqc_enabled=True),
                            partial(finish_solve, cfg=SMALL_SOLVE), SMALL_SOLVE.solver_config(), solve_many=solve_many)
    outcome = stage.run_batch([("clip1", truncated), ("clip0", accepted)], "t0-solve-00000")

    assert outcome.errors == {}
    assert len(seen) == 2
    _assert_obs_rows(*seen[0], PREFIX)
    _assert_obs_rows(*seen[1], N_ROWS)
    assert _solve_json(truncated)["valid_range"] == [0, PREFIX - 1]
    assert "valid_range" not in _solve_json(accepted)
    assert int(load_npz(truncated / "solve.npz")["frame_idx"].size) == PREFIX


@requires_frozen
def test_stale_valid_range_without_driftqc_json_raises(tmp_path: Path, driftqc_worker: StageFn) -> None:
    """valid_range.json is honoured only with its matching driftqc.json, and never with driftqc disabled."""
    artifact_dir = _driftqc_dir(tmp_path, "clip1", 9, driftqc_worker)
    (artifact_dir / DRIFTQC_FILE).unlink()
    solve = RecordingSolve()
    with pytest.raises(ValueError, match="driftqc.json absent"):
        make_solve_worker(SMALL_SOLVE, solve_fn=solve, driftqc_enabled=True)("clip1", artifact_dir)
    with pytest.raises(ValueError, match="stale driftqc artifacts"):
        make_solve_worker(SMALL_SOLVE, solve_fn=solve)("clip1", artifact_dir)
    assert solve.calls == []
    assert not (artifact_dir / "solve.npz").exists() and not (artifact_dir / "solve.json").exists()


@requires_frozen
def test_disabled_with_driftqc_artifacts_raises_never_slices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             driftqc_worker: StageFn) -> None:
    """A matching driftqc.json and valid_range.json [0,7] with driftqc_enabled False is a ValueError, never a
    slice; the same dir with driftqc_enabled True solves on the prefix."""
    artifact_dir = _driftqc_dir(tmp_path, "clip1", 9, driftqc_worker)
    rejected = _driftqc_dir(tmp_path, "clip2", 6, driftqc_worker)  # a REJECT leftover: driftqc.json only
    assert not (rejected / VALID_RANGE_FILE).exists()
    calls = _slice_spy(monkeypatch)
    solve = RecordingSolve()

    for clip_id, clip_dir in (("clip1", artifact_dir), ("clip2", rejected)):
        with pytest.raises(ValueError, match="stale driftqc artifacts"):
            make_solve_worker(SMALL_SOLVE, solve_fn=solve)(clip_id, clip_dir)
        with pytest.raises(ValueError, match="stale driftqc artifacts"):
            prepare_solve(clip_id, clip_dir, SMALL_SOLVE)
        with pytest.raises(ValueError, match="stale driftqc artifacts"):
            read_solve_inputs(clip_dir, SMALL_SOLVE.n_anchor_surface, SMALL_SOLVE.seed, driftqc_enabled=False)
    assert calls == [] and solve.calls == []
    assert not (artifact_dir / "solve.npz").exists()

    make_solve_worker(SMALL_SOLVE, solve_fn=solve, driftqc_enabled=True)("clip1", artifact_dir)
    [(_, obs, init_poses, _, _)] = solve.calls
    _assert_obs_rows(obs, init_poses, PREFIX)
    assert calls == _PER_ROW_ARTIFACTS


@requires_frozen
def test_finish_solve_without_valid_range_context_key(tmp_path: Path, driftqc_worker: StageFn) -> None:
    """finish_solve reads context['valid_range'] with .get; a context without it (or with None) writes no key.
    A disabled prepare_solve gives no key, so the None variant is built here by hand."""
    truncated = _driftqc_dir(tmp_path / "t", "clip1", 9, driftqc_worker)
    plain = _without_driftqc(truncated, tmp_path / "p" / "clip1")

    call = prepare_solve("clip1", plain, SMALL_SOLVE)
    assert "valid_range" not in call.context  # the key is set only with driftqc enabled
    with_none = dataclasses.replace(call, context=MappingProxyType({**call.context, "valid_range": None}))
    for variant in (call, with_none):
        finish_solve("clip1", plain, variant, fake_solution(call.init_poses, call.init_scale), SMALL_SOLVE)
        assert "valid_range" not in _solve_json(plain)

    call_t = prepare_solve("clip1", truncated, SMALL_SOLVE, driftqc_enabled=True)
    assert call_t.context["valid_range"] == RowRange(stop=PREFIX - 1, n_rows=N_ROWS)
    finish_solve("clip1", truncated, call_t, fake_solution(call_t.init_poses, call_t.init_scale), SMALL_SOLVE)
    assert _solve_json(truncated)["valid_range"] == [0, PREFIX - 1]


# ---- qc and retarget on the prefix ---------------------------------------------------------------------------

_PASSING = QcFeatures(reprojection_iou=0.9, laplace_std_ok_frac=1.0, ot_residual_rms_frac=0.02, penetration_depth_m=0.0,
                      track_inlier_ratio=0.9)
_ACCEPTING = AcceptanceRule(calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
                            logistic_weights=np.array([0.0, 2.0, 1.0, -1.0, -1.0, 1.0]), feature_mean=np.zeros(5),
                            feature_std=np.ones(5))
_RETARGET_CFG = RetargetStageConfig(scene_xml="/nonexistent/scene.xml", mano_right_pkl="/fake/MANO_RIGHT.pkl",
                                    mano_left_pkl="/fake/MANO_LEFT.pkl")
_HOOK_FILES = ("workers_solve.py", "workers_qc.py", "workers_retarget.py")


def _passing(*args: Any, **kwargs: Any) -> QcFeatures:
    return _PASSING


def _qc_worker(driftqc_enabled: bool = False) -> StageFn:
    return make_qc_worker(_ACCEPTING, features_fn=_passing, driftqc_enabled=driftqc_enabled)


def _solved_dir(root: Path, clip_id: str, contraction_from: int | None, worker: StageFn) -> Path:
    """`_driftqc_dir` plus a fake solve (RecordingSolve) with driftqc enabled."""
    artifact_dir = _driftqc_dir(root, clip_id, contraction_from, worker)
    make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve(), driftqc_enabled=True)(clip_id, artifact_dir)
    return artifact_dir


class _RecordingKinematic:
    """A retarget `stage_fn`: records the ClipRecord row count and returns a finite one-frame trajectory."""

    def __init__(self) -> None:
        self.rows: list[int] = []

    def __call__(self, record: ClipRecord, *args: Any) -> KinematicTrajectory:
        self.rows.append(record.n_frames)
        return KinematicTrajectory(qpos=np.zeros((1, 35)), qvel=np.zeros((1, 34)), frequency=float(args[6]))


def _retarget_worker(stage_fn: Any, driftqc_enabled: bool = False) -> StageFn:
    return make_retarget_worker(_RETARGET_CFG, scene=object(),  # type: ignore[arg-type]
                                stage_fn=stage_fn, label_fn=lambda path: torch.zeros(778, dtype=torch.int8),
                                driftqc_enabled=driftqc_enabled)


def _hook_slice_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    for module in (workers_qc, workers_retarget):
        original = module.slice_artifact

        def spy(arrays: Any, artifact: str, rng: RowRange, _original: Any = original) -> dict[str, np.ndarray]:
            calls.append(artifact)
            return dict(_original(arrays, artifact, rng))

        monkeypatch.setattr(module, "slice_artifact", spy)
    return calls


@requires_frozen
def test_qc_record_on_prefix_rows_and_timestamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 driftqc_worker: StageFn) -> None:
    artifact_dir = _solved_dir(tmp_path, "clip1", 9, driftqc_worker)
    decode = read_frame_idx(artifact_dir)
    with np.load(artifact_dir / "decode.npz", allow_pickle=False) as data:
        fps, stride = float(data["fps"]), int(data["stride"])
    assert stride == 3
    calls = _hook_slice_spy(monkeypatch)

    inputs = read_qc_inputs(artifact_dir, driftqc_enabled=True)
    assert inputs.valid_range == RowRange(stop=PREFIX - 1, n_rows=N_ROWS)
    assert calls == ["hands.npz", "mask.npz"]
    assert int(inputs.masks.shape[0]) == PREFIX and np.array_equal(inputs.frame_idx, decode[:PREFIX])
    for side in ("left", "right"):
        assert inputs.hands.tracks[side].params.shape[0] == PREFIX
        assert inputs.hands.vertices[side].shape[0] == PREFIX

    _qc_worker(driftqc_enabled=True)("clip1", artifact_dir)
    record = ClipRecord.load(artifact_dir / "clip_record.npz")
    record.validate()
    assert record.clip_id == "clip1"  # the clip id is unchanged
    assert np.array_equal(record.frame_idx, decode[:PREFIX])
    expected = decode[:PREFIX] / (fps * 3)
    assert record.timestamps.dtype == expected.dtype and record.timestamps.tobytes() == expected.tobytes()
    assert record.hand_right is not None and record.hand_right.params.shape[0] == PREFIX
    qc = json.loads((artifact_dir / "qc.json").read_text())
    assert qc["accepted"] is True and qc["n_frames"] == PREFIX


@requires_frozen
def test_accept_qc_json_equals_disabled(tmp_path: Path, driftqc_worker: StageFn) -> None:
    """ACCEPT takes the identity path: qc.json (real features, fake solve) equals the disabled run's. That qc.json
    equality is unconditional; the clip_record.npz array comparison runs only if the fixture is accepted, because the
    verdict of real features under this rule is not fixed by the spec."""
    with_records = _driftqc_dir(tmp_path / "on", "clip0", None, driftqc_worker)
    assert _range(with_records) == [0, N_ROWS - 1]
    without = _without_driftqc(with_records, tmp_path / "off" / "clip0")
    rule_path = tmp_path / "rule.json"
    save_acceptance_rule(rule_path, dataclasses.replace(_ACCEPTING, logistic_weights=np.array([10.0, 0, 0, 0, 0, 0])))

    make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve(), driftqc_enabled=True)("clip0", with_records)
    make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve())("clip0", without)
    make_qc_worker_from_config(QcStageConfig(rule_path=str(rule_path)), driftqc_enabled=True)("clip0", with_records)
    make_qc_worker_from_config(QcStageConfig(rule_path=str(rule_path)))("clip0", without)

    qc_on = json.loads((with_records / "qc.json").read_text())
    qc_off = json.loads((without / "qc.json").read_text())
    assert qc_on["n_frames"] == N_ROWS and qc_on == qc_off
    if qc_on["accepted"]:
        rec_on, rec_off = load_npz(with_records / "clip_record.npz"), load_npz(without / "clip_record.npz")
        assert rec_on.keys() == rec_off.keys()
        for key in rec_on.keys() - {"object_mesh_path"}:
            assert np.array_equal(rec_on[key], rec_off[key], equal_nan=rec_on[key].dtype.kind == "f"), key


@requires_frozen
def test_camera_frame_sliced_when_present(tmp_path: Path, driftqc_worker: StageFn) -> None:
    artifact_dir = _solved_dir(tmp_path, "clip1", 9, driftqc_worker)
    decode = read_frame_idx(artifact_dir)
    rng = RowRange(stop=PREFIX - 1, n_rows=N_ROWS)
    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (N_ROWS, 1, 1))
    cam_to_world[:, 0, 3] = np.arange(N_ROWS, dtype=np.float32)
    gravity = np.array([0.0, -0.8, 0.6], dtype=np.float32)
    geometry = load_npz(artifact_dir / "geometry.npz")
    save_npz(artifact_dir / "geometry.npz", **geometry, cam_to_world=cam_to_world, gravity_cam=gravity)

    camera = read_camera_frame(artifact_dir, decode[:PREFIX], rng)
    assert camera.cam_to_world.shape == (PREFIX, 4, 4) and np.array_equal(camera.cam_to_world, cam_to_world[:PREFIX])
    assert (camera.camera_source, camera.gravity_source) == ("geometry", "geometry")
    assert np.array_equal(camera.gravity_cam, gravity)
    with pytest.raises(ValueError, match="decode rows"):  # the unsliced geometry rows are not the prefix rows
        read_camera_frame(artifact_dir, decode[:PREFIX])
    assert read_camera_frame(artifact_dir, decode).cam_to_world.shape == (N_ROWS, 4, 4)

    _qc_worker(driftqc_enabled=True)("clip1", artifact_dir)
    record = ClipRecord.load(artifact_dir / "clip_record.npz")
    assert np.array_equal(record.cam_to_world, cam_to_world[:PREFIX]) and np.array_equal(record.gravity_cam, gravity)


@requires_frozen
def test_retarget_hands_check_on_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        driftqc_worker: StageFn) -> None:
    artifact_dir = _solved_dir(tmp_path, "clip1", 9, driftqc_worker)
    _qc_worker(driftqc_enabled=True)("clip1", artifact_dir)
    stage = _RecordingKinematic()

    _retarget_worker(stage, driftqc_enabled=True)("clip1", artifact_dir)
    report = json.loads((artifact_dir / "retarget.json").read_text())
    assert report["status"] == "done" and report["n_rows"] == PREFIX and stage.rows == [PREFIX]

    monkeypatch.setattr(workers_retarget, "slice_artifact", lambda arrays, artifact, rng: dict(arrays))  # mutant
    with pytest.raises(ValueError, match=r"hands.npz rows are not the decode rows: frame_idx"):
        _retarget_worker(stage, driftqc_enabled=True)("clip1", artifact_dir)
    report = json.loads((artifact_dir / "retarget.json").read_text())
    assert report["status"] == "failed" and report["retarget_fallback_needed"] is False and stage.rows == [PREFIX]


@requires_frozen
def test_qc_and_retarget_disabled_with_driftqc_artifacts_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               driftqc_worker: StageFn) -> None:
    """qc and retarget with driftqc_enabled False on a dir holding driftqc artifacts raise and never slice."""
    artifact_dir = _solved_dir(tmp_path, "clip1", 9, driftqc_worker)
    _qc_worker(driftqc_enabled=True)("clip1", artifact_dir)
    qc_bytes = (artifact_dir / "qc.json").read_bytes()
    rejected = _driftqc_dir(tmp_path, "clip2", 6, driftqc_worker)  # a REJECT leftover: driftqc.json only
    rule_path = tmp_path / "rule.json"
    save_acceptance_rule(rule_path, _ACCEPTING)
    calls = _hook_slice_spy(monkeypatch)
    stage = _RecordingKinematic()

    for clip_id, clip_dir in (("clip1", artifact_dir), ("clip2", rejected)):
        with pytest.raises(ValueError, match="stale driftqc artifacts"):
            read_qc_inputs(clip_dir)
        with pytest.raises(ValueError, match="stale driftqc artifacts"):
            _qc_worker()(clip_id, clip_dir)
        with pytest.raises(ValueError, match="stale driftqc artifacts"):
            make_qc_worker_from_config(QcStageConfig(rule_path=str(rule_path)), features_fn=_passing)(clip_id, clip_dir)
    assert (artifact_dir / "qc.json").read_bytes() == qc_bytes and not (rejected / "qc.json").exists()
    with pytest.raises(ValueError, match="stale driftqc artifacts"):
        _retarget_worker(stage)("clip1", artifact_dir)
    report = json.loads((artifact_dir / "retarget.json").read_text())
    assert report["status"] == "failed" and "stale driftqc artifacts" in report["error"]
    assert calls == [] and stage.rows == []
    assert not (artifact_dir / "trajectory_kinematic.npz").exists()

    _retarget_worker(stage, driftqc_enabled=True)("clip1", artifact_dir)  # the same dir, enabled: the prefix
    assert stage.rows == [PREFIX] and calls == ["hands.npz"]

    (artifact_dir / DRIFTQC_FILE).unlink()  # valid_range.json without its driftqc.json
    with pytest.raises(ValueError, match="driftqc.json absent"):
        read_qc_inputs(artifact_dir, driftqc_enabled=True)
    with pytest.raises(ValueError, match="driftqc.json absent"):
        _retarget_worker(stage, driftqc_enabled=True)("clip1", artifact_dir)


def test_single_helper_ast() -> None:
    """Every row slice in the hook modules goes through slice_artifact: `as_slice()` is only ever used to subscript
    `frame_idx` or `timestamps`, no slice is built from a range's `stop`, and each module binds the valid_range hooks."""
    engine = Path(workers_solve.__file__).parent
    for name in _HOOK_FILES:
        tree = ast.parse((engine / name).read_text(), filename=name)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        n_as_slice = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "as_slice":
                n_as_slice += 1
                parent = parents.get(node)
                assert isinstance(parent, ast.Subscript) and parent.slice is node, (name, node.lineno)
                assert isinstance(parent.value, ast.Name) and parent.value.id in {"frame_idx", "timestamps"}, \
                    (name, node.lineno)
            if isinstance(node, ast.Slice):
                bounds = [b for b in (node.lower, node.upper, node.step) if b is not None]
                assert not any(isinstance(sub, ast.Attribute) and sub.attr == "stop"
                               for b in bounds for sub in ast.walk(b)), (name, node.lineno)
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    and node.module == "fastdo.engine.valid_range" for alias in node.names}
        assert {"resolve_valid_range", "slice_artifact"} <= imported, (name, imported)
        assert n_as_slice >= (0 if name == "workers_retarget.py" else 1), name  # retarget slices hands only
