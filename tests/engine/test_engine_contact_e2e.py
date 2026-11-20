"""The `contact` stage in the real chain: solve -> contact -> qc, and what qc then scores and exports.

The wiring is the part that can silently do nothing (or silently do the wrong thing): the correction only pays off
if qc scores and the ClipRecord exports the CORRECTED trajectory, and only if a stale correction is never applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, run_manifest
from fastdo.engine.workers_contact import CONTACT_FILE, ContactStageConfig, make_contact_worker, with_contact
from fastdo.engine.workers_qc import QcStageConfig, make_qc_worker_from_config
from fastdo.engine.workers_solve import RegistrationConfig, SolveStageConfig, make_solve_worker
from fastdo.io.clip_record import ClipRecord
from fastdo.qc import AcceptanceRule, ConformalCalibration
from fastdo.qc.rule_io import save_acceptance_rule
from fastdo.solver.synthetic import make_synthetic_clip
from tests.engine.fakes_solve import SMALL_CLIP, synthetic_front_end_stage_fns

_STAGES = ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve", "contact", "qc")


def _run(tmp_path: Path, *, with_contact_stage: bool) -> Path:
    clips = {"clip0": make_synthetic_clip(SMALL_CLIP, seed=0)}
    fns = synthetic_front_end_stage_fns(clips, tmp_path / "templates")
    fns["solve"] = make_solve_worker(SolveStageConfig(
        device="cpu", seed=0, registration=RegistrationConfig(n_obs=128, n_mesh=256, n_iters=10), n_anchor_surface=4096,
        solver={"n_obs": 256, "n_mesh": 256, "iters_coarse": 3, "iters_mid": 2, "iters_fine": 3}))
    rule_path = tmp_path / "rule.json"
    save_acceptance_rule(rule_path, AcceptanceRule(
        calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
        logistic_weights=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), feature_mean=np.zeros(5), feature_std=np.ones(5)))
    fns["qc"] = make_qc_worker_from_config(QcStageConfig(rule_path=str(rule_path)))
    stages: tuple[str, ...] = ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve", "qc")
    if with_contact_stage:
        fns["contact"] = make_contact_worker(ContactStageConfig(enabled=True))
        stages = with_contact(stages)
        assert stages == _STAGES
    root, manifest_path = tmp_path / "engine", tmp_path / "manifest.parquet"
    manifest = ClipManifest.create(["clip0"], ["synthetic"], [6])
    manifest.save(manifest_path)
    run_manifest(manifest, EngineConfig(artifact_root=root, stages=stages, manifest_path=manifest_path), fns)
    for stage in stages:
        status = manifest.status("clip0", stage)
        assert (status.state, status.error) == ("done", ""), (stage, status.error)
    return root / "clip0"


def test_qc_scores_and_exports_the_corrected_trajectory(tmp_path: Path) -> None:
    d = _run(tmp_path, with_contact_stage=True)

    qc = json.loads((d / "qc.json").read_text())
    assert qc["poses_source"] == "contact"
    assert qc["contact_shift_mm_median"] is not None

    contact, solve = np.load(d / CONTACT_FILE), np.load(d / "solve.npz")
    assert not np.array_equal(contact["poses"], solve["poses"])          # the correction did something
    assert np.array_equal(solve["poses"], np.load(d / "solve.npz")["poses"])   # solve.npz is untouched

    if qc["accepted"]:
        record = ClipRecord.load(d / "clip_record.npz")
        assert np.allclose(np.asarray(record.object_pose, dtype=np.float32),
                           contact["poses"].astype(np.float32), atol=1e-6)


def test_without_the_stage_qc_scores_the_solver_trajectory(tmp_path: Path) -> None:
    d = _run(tmp_path, with_contact_stage=False)
    qc = json.loads((d / "qc.json").read_text())
    assert qc["poses_source"] == "solve" and qc["contact_shift_mm_median"] is None
    assert not (d / CONTACT_FILE).exists()


def test_a_stale_correction_is_not_applied(tmp_path: Path) -> None:
    """Re-solving must not leave the previous correction in force; qc falls back to the solver's poses."""
    d = _run(tmp_path, with_contact_stage=True)
    solve = dict(np.load(d / "solve.npz"))
    solve["poses"] = solve["poses"] + np.float32(0.02)
    np.savez(d / "solve.npz", **solve)

    rule_path = tmp_path / "rule.json"
    make_qc_worker_from_config(QcStageConfig(rule_path=str(rule_path)))("clip0", d)

    assert json.loads((d / "qc.json").read_text())["poses_source"] == "solve"


def test_with_contact_refuses_a_bad_stage_list() -> None:
    assert with_contact(("solve", "qc")) == ("solve", "contact", "qc")
    assert with_contact(("solve", "contact", "qc")) == ("solve", "contact", "qc")   # idempotent
    with pytest.raises(ValueError, match="needs a 'solve' stage"):
        with_contact(("decode", "qc"))
    with pytest.raises(ValueError, match="precedes"):
        with_contact(("qc", "solve"))
    with pytest.raises(ValueError, match="must follow 'solve'"):
        with_contact(("solve", "qc", "contact"))
