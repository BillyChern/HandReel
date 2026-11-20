"""End to end: a three-clip manifest reaches `qc` through the real solve stage (CPU; the front end
is faked from synthetic clips, the shape stage is the real template worker)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.bench.yield_report import compute_yield_report, read_qc_verdicts
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, clips_per_gpu_hour_report, run_manifest
from fastdo.engine.workers_qc import QcStageConfig, make_qc_worker_from_config, read_solve_artifact
from fastdo.engine.workers_solve import RegistrationConfig, SolveStageConfig, make_solve_worker
from fastdo.io.clip_record import ClipRecord
from fastdo.qc import AcceptanceRule, ConformalCalibration
from fastdo.qc.rule_io import save_acceptance_rule
from fastdo.solver.synthetic import make_synthetic_clip
from tests.engine.fakes_solve import SMALL_CLIP, synthetic_front_end_stage_fns

_STAGES = ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve", "qc")


def test_three_clip_manifest_reaches_qc_through_the_solve_stage(tmp_path: Path) -> None:
    clips = {f"clip{s}": make_synthetic_clip(SMALL_CLIP, seed=s) for s in range(3)}
    fns = synthetic_front_end_stage_fns(clips, tmp_path / "templates")
    fns["solve"] = make_solve_worker(SolveStageConfig(
        device="cpu", seed=0, registration=RegistrationConfig(n_obs=128, n_mesh=256, n_iters=10), n_anchor_surface=4096,
        solver={"n_obs": 256, "n_mesh": 256, "iters_coarse": 3, "iters_mid": 2, "iters_fine": 3}))
    rule_path = tmp_path / "rule.json"
    save_acceptance_rule(rule_path, AcceptanceRule(
        calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
        logistic_weights=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), feature_mean=np.zeros(5), feature_std=np.ones(5)))
    fns["qc"] = make_qc_worker_from_config(QcStageConfig(rule_path=str(rule_path)))
    root, manifest_path, ids = tmp_path / "engine", tmp_path / "manifest.parquet", list(clips)
    manifest = ClipManifest.create(ids, ["synthetic"] * 3, [6] * 3)

    run_manifest(manifest, EngineConfig(artifact_root=root, stages=_STAGES, manifest_path=manifest_path), fns)

    rule_sha = hashlib.sha256(rule_path.read_bytes()).hexdigest()
    for clip_id in ids:
        for stage in _STAGES:
            status = manifest.status(clip_id, stage)
            assert (status.state, status.error) == ("done", ""), (clip_id, stage, status.error)
        solve = read_solve_artifact(root / clip_id)
        assert solve.solution.poses.shape == (6, 4, 4) and bool(torch.isfinite(solve.solution.poses).all())
        assert solve.frame_idx.tolist() == [0, 3, 6, 9, 12, 15]
        weight = solve.track_weight
        assert weight.shape == (6, 24) and float(weight.min()) >= 0.0 and float(weight.max()) <= 1.0
        meta = json.loads((root / clip_id / "solve.json").read_text())
        assert meta["registration"]["scale_source"] == "metric_template"
        qc = json.loads((root / clip_id / "qc.json").read_text())
        assert qc["accepted"] is True and qc["rule"]["sha256"] == rule_sha
        ClipRecord.load(root / clip_id / "clip_record.npz").validate()

    verdicts = read_qc_verdicts(root, ids)
    assert sum(verdicts.values()) == 3
    saved = ClipManifest.load(manifest_path)
    report = compute_yield_report(saved, gate_built=False, qc_verdicts=verdicts)
    assert (report.n_qc_accepted, report.yield_fraction) == (3, 1.0)
    total = sum(saved.status(c, s).seconds for c in ids for s in saved.stage_names())
    assert clips_per_gpu_hour_report(saved)["overall"] == pytest.approx(3 * 3600 / total, rel=1e-12)
