"""driftqc engine end to end.

A three-clip manifest of 24-row synthetic clips runs `with_driftqc(_STAGES)` through `run_manifest`: the front end is
faked (tests/engine/fakes_solve.py), the tracks stage writes the contraction fixture, `driftqc` is the real worker on
the frozen files, solve is `RecordingSolve`, qc is the real worker and retarget is the real worker with an injected
kinematic stage and finger labels, all built with `driftqc_enabled=True`. Seed 0 is healthy (ACCEPT [0, 23]), seed 1
is contracted from row 9 (TRUNCATE [0, 7]) and seed 2 from row 6 (REJECT "short"). The frozen files are read under
FROZEN_ROOT (env FASTDO_DRIFTQC_ROOT); the test skips only when they are absent.

The manifest test runs driftqc enabled only, so it cannot catch a disabled-path bypass (valid_range.json honoured with
driftqc off); the disabled-path tests in test_driftqc_solve_qc.py and test_driftqc_default_identity.py cover that path.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.engine import workers_qc, workers_retarget, workers_solve
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, StageFn, run_manifest
from fastdo.engine.valid_range import DRIFTQC_FILE, VALID_RANGE_FILE, RowRange
from fastdo.engine.worker_support import load_npz, read_frame_idx, save_npz
from fastdo.engine.workers_driftqc import (
    DriftQcStageConfig,
    DriftRejected,
    load_driftqc_constants,
    load_frozen_detector,
    make_driftqc_worker,
    with_driftqc,
)
from fastdo.engine.workers_qc import QcStageConfig, make_qc_worker_from_config
from fastdo.engine.workers_retarget import RetargetStageConfig, make_retarget_worker
from fastdo.engine.workers_solve import finish_solve, make_solve_worker, prepare_solve
from fastdo.io.clip_record import ClipRecord
from fastdo.qc import AcceptanceRule, ConformalCalibration
from fastdo.qc.rule_io import save_acceptance_rule
from fastdo.retarget.kinematic import KinematicTrajectory
from fastdo.solver.synthetic import SyntheticClip
from tests.engine.fakes_driftqc import (
    N_ROWS,
    contracted_tracks,
    driftqc_clip,
    write_driftqc_solve_inputs,
)
from tests.engine.fakes_solve import (
    SMALL_SOLVE,
    RecordingSolve,
    fake_solution,
    synthetic_front_end_stage_fns,
)

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

_STAGES = ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve", "qc", "retarget")
_STRIDE = 3
# clip id -> (seed, contraction_from, driftqc decision, kept rows or None for REJECT)
_CLIPS: dict[str, tuple[int, int | None, str, int | None]] = {
    "clip0": (0, None, "ACCEPT", N_ROWS),
    "clip1": (1, 9, "TRUNCATE", 8),
    "clip2": (2, 6, "REJECT", None),
}


def _contracted_tracks_stage(clips: dict[str, SyntheticClip], contraction: dict[str, int | None]) -> StageFn:
    """The fake tracks worker writing the E3 fixture's tracks.npz (rows >= contraction_from moved toward the centroid)."""

    def worker(clip_id: str, artifact_dir: Path) -> None:
        clip = clips[clip_id]
        tracks = contracted_tracks(clip, contraction[clip_id])
        save_npz(Path(artifact_dir) / "tracks.npz", tracks=tracks, vis=clip.track_vis.numpy().astype(bool),
                 query_uv=tracks[0].copy(), frame_idx=_STRIDE * np.arange(N_ROWS, dtype=np.int64),
                 keyframe_index=np.array(0, dtype=np.int64))

    return worker


class _RecordingKinematic:
    """A retarget `stage_fn`: records (clip id, ClipRecord rows) and returns a finite one-frame trajectory."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, int]] = []

    def __call__(self, record: ClipRecord, *args: Any) -> KinematicTrajectory:
        self.rows.append((record.clip_id, record.n_frames))
        return KinematicTrajectory(qpos=np.zeros((1, 35)), qvel=np.zeros((1, 34)), frequency=float(args[6]))


def _reference_onset(clip_dir: Path) -> int | None:
    """The onset re-derived from the clip's saved mask/geometry/tracks arrays through the frozen detector's
    `sg_onset(sg_stat(...))`, independent of the engine worker (the `_reference_m` pattern)."""
    constants = load_driftqc_constants(CONSTANTS, TRACKWEIGHT_CONSTANTS)
    detector = load_frozen_detector(DETECTOR)
    mask, geo, tr = (load_npz(clip_dir / name) for name in ("mask.npz", "geometry.npz", "tracks.npz"))
    m = detector.sg_stat(geo["pointmaps"].astype(np.float64), mask["masks"].astype(bool),
                         tr["query_uv"].astype(np.float64), tr["tracks"].astype(np.float64), tr["vis"].astype(bool),
                         constants.pair_floor_m, constants.min_pairs, constants.patch_r, kf=0)
    onset = detector.sg_onset(np.asarray(m), constants.tau_abs, constants.K_consec)
    return None if onset is None else int(onset)


def _spy_resolve(monkeypatch: pytest.MonkeyPatch, seen: list[tuple[str, str, bool, RowRange | None]]) -> None:
    """Records (hook module, clip dir name, driftqc_enabled, returned range) for the solve, qc and retarget hooks."""
    for module in (workers_solve, workers_qc, workers_retarget):
        original = module.resolve_valid_range

        def spy(artifact_dir: Path, frame_idx: np.ndarray, driftqc_enabled: bool, *, _name: str = module.__name__,
                _original: Any = original) -> RowRange | None:
            rng = _original(artifact_dir, frame_idx, driftqc_enabled)
            seen.append((_name, Path(artifact_dir).name, driftqc_enabled, rng))
            return rng  # type: ignore[no-any-return]

        monkeypatch.setattr(module, "resolve_valid_range", spy)


@requires_frozen
def test_three_clip_manifest_with_driftqc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = {clip_id: driftqc_clip(seed) for clip_id, (seed, *_) in _CLIPS.items()}
    fns = synthetic_front_end_stage_fns(clips, tmp_path / "templates", stride=_STRIDE)
    fns["tracks"] = _contracted_tracks_stage(clips, {c: spec[1] for c, spec in _CLIPS.items()})
    fns["driftqc"] = make_driftqc_worker(DriftQcStageConfig(
        enabled=True, constants_path=str(CONSTANTS), trackweight_constants_path=str(TRACKWEIGHT_CONSTANTS),
        detector_path=str(DETECTOR)))
    solve = RecordingSolve()
    fns["solve"] = make_solve_worker(SMALL_SOLVE, solve_fn=solve, driftqc_enabled=True)
    rule_path = tmp_path / "rule.json"
    save_acceptance_rule(rule_path, AcceptanceRule(
        calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
        logistic_weights=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), feature_mean=np.zeros(5), feature_std=np.ones(5)))
    fns["qc"] = make_qc_worker_from_config(QcStageConfig(rule_path=str(rule_path)), driftqc_enabled=True)
    kinematic = _RecordingKinematic()
    fns["retarget"] = make_retarget_worker(
        RetargetStageConfig(scene_xml="/nonexistent/scene.xml", mano_right_pkl="/fake/MANO_RIGHT.pkl",
                            mano_left_pkl="/fake/MANO_LEFT.pkl"), scene=object(),  # type: ignore[arg-type]
        stage_fn=kinematic, label_fn=lambda path: torch.zeros(778, dtype=torch.int8), driftqc_enabled=True)
    stages = with_driftqc(_STAGES)
    assert stages == ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "driftqc", "solve", "qc",
                      "retarget")
    root, manifest_path = tmp_path / "engine", tmp_path / "manifest.parquet"
    manifest = ClipManifest.create(list(_CLIPS), ["synthetic"] * 3, [N_ROWS] * 3)
    seen: list[tuple[str, str, bool, RowRange | None]] = []
    _spy_resolve(monkeypatch, seen)

    run_manifest(manifest, EngineConfig(artifact_root=root, stages=stages, manifest_path=manifest_path), fns)

    saved = ClipManifest.load(manifest_path)
    for clip_id, (_, _, decision, rows) in _CLIPS.items():
        clip_dir = root / clip_id
        doc = json.loads((clip_dir / DRIFTQC_FILE).read_text())
        assert doc["decision"] == decision, clip_id
        decode = read_frame_idx(clip_dir)
        assert decode.tolist() == (_STRIDE * np.arange(N_ROWS)).tolist()
        if rows is None:  # clip 2: driftqc failed with the prefix, nothing downstream ran
            for m in (manifest, saved):
                status = m.status(clip_id, "driftqc")
                assert (status.state, status.error) == ("failed", "drift rejected: short")
                assert [m.status(clip_id, s).state for s in ("solve", "qc", "retarget")] == ["pending"] * 3
            assert all(manifest.status(clip_id, s).state == "done" for s in _STAGES[:7])
            assert _reference_onset(clip_dir) == 6
            assert (doc["onset"], doc["cut_b"], doc["range"]) == (6, 4, None)
            assert not any((clip_dir / name).exists() for name in (VALID_RANGE_FILE, "solve.json", "qc.json",
                                                                   "clip_record.npz", "retarget.json"))
            continue
        for m in (manifest, saved):
            for stage in stages:
                status = m.status(clip_id, stage)
                assert (status.state, status.error) == ("done", ""), (clip_id, stage, status.error)
        assert json.loads((clip_dir / VALID_RANGE_FILE).read_text())["range"] == [0, rows - 1]
        solve_meta = json.loads((clip_dir / "solve.json").read_text())
        if rows == N_ROWS:  # ACCEPT: the full range takes the identity path, solve.json has no valid_range key
            assert _reference_onset(clip_dir) is None
            assert doc["onset"] is None and "valid_range" not in solve_meta
        else:
            assert _reference_onset(clip_dir) == 9
            assert (doc["onset"], doc["cut_b"]) == (9, 7) and solve_meta["valid_range"] == [0, rows - 1]
        record = ClipRecord.load(clip_dir / "clip_record.npz")
        record.validate()
        assert record.clip_id == clip_id and record.n_frames == rows
        assert np.array_equal(record.frame_idx, decode[:rows])
        qc = json.loads((clip_dir / "qc.json").read_text())
        assert qc["n_frames"] == rows and qc["accepted"] is True, clip_id  # retarget must reach its hook
        report = json.loads((clip_dir / "retarget.json").read_text())
        assert (report["status"], report["n_rows"]) == ("done", rows), (clip_id, report)

    assert [int(call[2].shape[0]) for call in solve.calls] == [N_ROWS, 8]  # obs T and init poses per solved clip
    assert [int(call[1].tracks.shape[0]) for call in solve.calls] == [N_ROWS, 8]
    assert kinematic.rows == [("clip0", N_ROWS), ("clip1", 8)]
    assert {(name.rsplit(".", 1)[1], clip, enabled) for name, clip, enabled, _ in seen} == {
        (module, clip, True) for module in ("workers_solve", "workers_qc", "workers_retarget")
        for clip in ("clip0", "clip1")}
    assert all(rng == (RowRange(stop=7, n_rows=N_ROWS) if clip == "clip1" else None) for _, clip, _, rng in seen)


def _driftqc_worker() -> StageFn:
    return make_driftqc_worker(DriftQcStageConfig(
        enabled=True, constants_path=str(CONSTANTS), trackweight_constants_path=str(TRACKWEIGHT_CONSTANTS),
        detector_path=str(DETECTOR)))


@requires_frozen
def test_prepare_solve_valid_range_key_only_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """prepare_solve puts 'valid_range' in call.context only with driftqc enabled. With driftqc disabled every
    dir either raises (leftover driftqc artifacts of any decision, or valid_range.json alone) or gives a
    context without the key, so a key-carrying context cannot come out of a disabled prepare_solve; finish_solve on a
    context without the key never slices: full-row solve.npz, no valid_range in solve.json.

    The disabled-path solve.npz and solve.json are also compared for exact equality with a default-path run
    (`make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve())`, no driftqc kwarg) on a pristine copy of the same
    inputs. The whole-pipeline default identity is covered by test_default_run_equals_hooks_stubbed
    (tests/engine/test_driftqc_default_identity.py)."""
    worker, clip = _driftqc_worker(), driftqc_clip(0)
    dirs: dict[str, Path] = {}
    for name, contraction in (("plain", 9), ("accept", None), ("truncate", 9), ("reject", 6)):
        dirs[name] = write_driftqc_solve_inputs(tmp_path / name / "clip1", clip, contraction)
        if name == "plain":  # the same contracted inputs, driftqc never run
            shutil.copytree(dirs[name], tmp_path / "default" / "clip1")
            continue
        try:
            worker("clip1", dirs[name])
        except DriftRejected:
            assert name == "reject"
    dirs["orphan"] = shutil.copytree(dirs["truncate"], tmp_path / "orphan" / "clip1")
    (dirs["orphan"] / DRIFTQC_FILE).unlink()  # valid_range.json without its driftqc.json
    sliced: list[str] = []
    original_slice = workers_solve.slice_artifact

    def slice_spy(arrays: Any, artifact: str, rng: RowRange) -> dict[str, np.ndarray]:
        sliced.append(artifact)
        return original_slice(arrays, artifact, rng)

    monkeypatch.setattr(workers_solve, "slice_artifact", slice_spy)

    for name in ("accept", "truncate", "reject", "orphan"):
        for kwargs in ({}, {"driftqc_enabled": False}):
            with pytest.raises(ValueError, match="stale driftqc artifacts"):
                prepare_solve("clip1", dirs[name], SMALL_SOLVE, **kwargs)
    plain = dirs["plain"]
    for kwargs in ({}, {"driftqc_enabled": False}):
        call = prepare_solve("clip1", plain, SMALL_SOLVE, **kwargs)
        assert "valid_range" not in call.context and int(call.obs.tracks.shape[0]) == N_ROWS
    finish_solve("clip1", plain, call, fake_solution(call.init_poses, call.init_scale), SMALL_SOLVE)
    assert sliced == []
    with np.load(plain / "solve.npz", allow_pickle=False) as data:
        assert np.array_equal(data["frame_idx"], read_frame_idx(plain)) and data["poses"].shape[0] == N_ROWS
    meta = json.loads((plain / "solve.json").read_text())
    assert "valid_range" not in meta and meta["n_rows"] == N_ROWS
    default = tmp_path / "default" / "clip1"
    make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve())("clip1", default)
    with np.load(plain / "solve.npz", allow_pickle=False) as got, np.load(default / "solve.npz", allow_pickle=False) as ref:
        assert sorted(got.files) == sorted(ref.files)
        for key in ref.files:
            assert got[key].dtype == ref[key].dtype and np.array_equal(got[key], ref[key], equal_nan=True), key
    assert meta == json.loads((default / "solve.json").read_text())

    accept = prepare_solve("clip1", dirs["accept"], SMALL_SOLVE, driftqc_enabled=True)
    assert "valid_range" in accept.context and accept.context["valid_range"] is None and sliced == []
    truncate = prepare_solve("clip1", dirs["truncate"], SMALL_SOLVE, driftqc_enabled=True)
    assert truncate.context["valid_range"] == RowRange(stop=7, n_rows=N_ROWS) and sliced
