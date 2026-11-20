"""Tests for the gate stage (`fastdo.engine.workers_gate`): stub gates on fake encoder archives, no weights."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.bench.yield_report import compute_yield_report
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import STAGE_GPU, EngineConfig, StageFn, run_clip
from fastdo.engine.workers_gate import (
    GATE_REJECTED_PREFIX,
    GateRejected,
    GateStageConfig,
    gate_outcome_counts,
    make_gate_worker,
    read_gate_json,
)
from fastdo.engine.workers_qc import _gate_score
from fastdo.gate.bundle import GateBundle
from fastdo.gate.calibrate import GateCalibration
from fastdo.gate.model import GateConfig, InteractionGate
from tests.gate.fakes import StubGate, write_encoder_npz


def _bundle(score: float, threshold: float = 0.5, crop: str = "letterbox", token_dim: int = 16,
            model: InteractionGate | None = None) -> GateBundle:
    gate = StubGate([score], token_dim=token_dim) if model is None else model
    calibration = GateCalibration(threshold=threshold, alpha=0.05, k=2, n_pos=41, n_neg=0,
                                  guaranteed_recall=1 - 2 / 42, empirical_recall=1.0, empirical_fpr=None)
    return GateBundle(model=gate, config=gate.cfg, calibration=calibration, crop=crop, hand_conf_source="none",
                      card={}, path="/stub/gate_bundle.pt", sha256="ab" * 32)


def _encoder(artifact_dir: Path, crop: str = "letterbox", dim: int = 16) -> None:
    tokens = np.random.default_rng(0).standard_normal((30, dim)).astype(np.float32)
    write_encoder_npz(artifact_dir, tokens, np.arange(0, 90, 3), crop)


def test_gate_worker_accepts_and_writes_gate_json(tmp_path: Path) -> None:
    _encoder(tmp_path)

    make_gate_worker(GateStageConfig(), _bundle(0.8))("c0", tmp_path)

    gate = read_gate_json(tmp_path)
    assert gate["score"] == pytest.approx(0.8) and gate["score"] == gate["p_interact"]
    assert gate["accepted"] is True and gate["threshold"] == 0.5 and gate["version"] == 1
    assert gate["sampled_rows"] == [0, 4, 8, 12, 17, 21, 25, 29]
    assert gate["sampled_frame_idx"] == [0, 12, 24, 36, 51, 63, 75, 87]
    assert gate["keyframe_row"] in gate["sampled_rows"] and len(gate["keyframe_scores"]) == 8
    assert gate["keyframe_frame_idx"] == 3 * gate["keyframe_row"]
    assert gate["hand_conf_source"] == "none" and gate["rigid_calibrated"] is False and gate["crop"] == "letterbox"
    assert gate["bundle"] == {"path": "/stub/gate_bundle.pt", "sha256": "ab" * 32, "version": 1}
    assert _gate_score(tmp_path) == pytest.approx(0.8)


def test_gate_worker_rejects_below_threshold(tmp_path: Path) -> None:
    _encoder(tmp_path)

    with pytest.raises(GateRejected) as info:
        make_gate_worker(GateStageConfig(), _bundle(0.3))("c1", tmp_path)

    assert str(info.value) == "gate rejected: p_interact 0.300000 < threshold 0.500000"
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert gate["accepted"] is False and gate["score"] == pytest.approx(0.3)
    make_gate_worker(GateStageConfig(), _bundle(0.5))("c1", tmp_path)                 # 0.5 >= 0.5 is accepted
    assert read_gate_json(tmp_path)["accepted"] is True


def _fake_encoder(crop: str = "letterbox", dim: int = 16) -> StageFn:
    def worker(clip_id: str, artifact_dir: Path) -> None:
        _encoder(artifact_dir, crop, dim)

    return worker


def test_run_clip_stops_at_gate_rejection(tmp_path: Path) -> None:
    cfg = EngineConfig(artifact_root=tmp_path / "engine", stages=("encoder", "gate", "mask"))
    manifest = ClipManifest.create(["c0", "c1"], ["test", "test"], [90, 90])
    accept, reject = make_gate_worker(GateStageConfig(), _bundle(0.8)), make_gate_worker(GateStageConfig(), _bundle(0.3))
    mask_calls: list[str] = []

    def gate(clip_id: str, artifact_dir: Path) -> None:
        (accept if clip_id == "c0" else reject)(clip_id, artifact_dir)

    fns: dict[str, StageFn] = {"encoder": _fake_encoder(), "gate": gate,
                               "mask": lambda clip_id, artifact_dir: mask_calls.append(clip_id)}
    for clip_id in manifest.clip_ids():
        run_clip(clip_id, cfg, fns, manifest)

    assert mask_calls == ["c0"]
    rejected = manifest.status("c1", "gate")
    assert rejected.state == "failed" and rejected.error.startswith(GATE_REJECTED_PREFIX)
    assert manifest.status("c1", "mask").state == "pending"
    assert compute_yield_report(manifest, gate_built=True).n_gate_accepted == 1
    assert gate_outcome_counts(manifest) == {"accepted": 1, "rejected": 1, "error": 0, "pending": 0}


def test_gate_errors_are_not_rejections(tmp_path: Path) -> None:
    worker = make_gate_worker(GateStageConfig(), _bundle(0.8))
    cases: dict[str, StageFn | None] = {"missing": None, "crop": _fake_encoder(crop="center"),
                                       "dim": _fake_encoder(dim=15)}
    manifest = ClipManifest.create(list(cases), ["test"] * 3, [90] * 3)
    for clip_id, encoder in cases.items():
        fns: dict[str, StageFn] = {"gate": worker} if encoder is None else {"encoder": encoder, "gate": worker}
        run_clip(clip_id, EngineConfig(artifact_root=tmp_path, stages=("encoder", "gate")), fns, manifest)

    errors = {clip_id: manifest.status(clip_id, "gate").error for clip_id in cases}
    assert all(manifest.status(clip_id, "gate").state == "failed" for clip_id in cases)
    assert "run the 'encoder' stage" in errors["missing"]
    assert "'center'" in errors["crop"] and "letterbox" in errors["crop"]
    assert "15" in errors["dim"] and "16" in errors["dim"]
    assert gate_outcome_counts(manifest) == {"accepted": 0, "rejected": 0, "error": 3, "pending": 0}
    assert not list(tmp_path.rglob("gate.json"))


def test_gate_stage_config_no_accept_all_default(tmp_path: Path) -> None:
    assert GateStageConfig.from_mapping({"model_path": None}) == GateStageConfig()
    assert GateStageConfig.from_mapping({"model_path": tmp_path}).model_path == str(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        GateStageConfig.from_mapping({"model_path": None, "threshold": 0.5})
    with pytest.raises(ValueError, match="accept-all"):
        make_gate_worker(GateStageConfig())
    with pytest.raises(FileNotFoundError):
        make_gate_worker(GateStageConfig(model_path=str(tmp_path / "missing.pt")))
    assert STAGE_GPU["gate"] is False


@pytest.mark.timing
def test_gate_stage_is_cheap(tmp_path: Path) -> None:
    torch.manual_seed(0)
    worker = make_gate_worker(GateStageConfig(), _bundle(0.0, threshold=0.0, model=InteractionGate(
        GateConfig(token_dim=2048)).eval()))
    write_encoder_npz(tmp_path, np.random.default_rng(0).standard_normal((100, 2048)), np.arange(100) * 3,
                      "letterbox")
    worker("warm", tmp_path)
    seconds: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        worker("c", tmp_path)
        seconds.append(time.perf_counter() - start)
    assert statistics.median(seconds) < 0.05
