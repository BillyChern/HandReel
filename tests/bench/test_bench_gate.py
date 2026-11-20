"""Tests for the gate wiring of the throughput benchmark (`bench_stages`, `bench_throughput.s2_status`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fastdo.bench.bench_clips import BenchClip
from fastdo.bench.bench_stages import build_stage_fns
from fastdo.bench.yield_report import GATE_UNBUILT
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.workers_frames import EncodeWorker
from fastdo.gate.bundle import save_gate_bundle
from fastdo.gate.calibrate import GateCalibration
from fastdo.gate.model import GateConfig, InteractionGate
from run.pipeline import bench_throughput as bench

_REPO = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, *overrides: str) -> DictConfig:
    base = [f"paths.root={_REPO}", f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}",
            f"whisking.demo_root={tmp_path / 'demo'}", "bench.doh_root=null", "bench.whisking_copies=1"]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[*base, *overrides])


def _saved_bundle(path: Path) -> Path:
    torch.manual_seed(0)
    calibration = GateCalibration(threshold=0.25, alpha=0.05, k=2, n_pos=41, n_neg=0, guaranteed_recall=1 - 2 / 42,
                                  empirical_recall=1.0, empirical_fpr=None)
    save_gate_bundle(path, InteractionGate(GateConfig(token_dim=16, hidden=8)), calibration, "letterbox", {})
    return path


def test_gate_registered_only_with_model_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    clips = bench.build_clips(cfg, _REPO)

    fns, status = build_stage_fns(cfg, _REPO, clips)

    assert "gate" not in fns and status["gate"] == GATE_UNBUILT
    default_encoder = fns["encoder"]
    assert isinstance(default_encoder, EncodeWorker) and default_encoder.crop == "center"
    path = _saved_bundle(tmp_path / "gate_bundle.pt")
    gate_cfg = _cfg(tmp_path, "bench.stages=[decode,encoder,gate]", f"gate_stage.model_path={path}")
    with_gate, gate_status = build_stage_fns(gate_cfg, _REPO, clips)
    assert "gate" in with_gate and gate_status["gate"].startswith("computed (gate bundle")
    assert "threshold 0.2500" in gate_status["gate"] and "on 41 contact-centred 100DOH train positives" in gate_status["gate"]
    encoder = with_gate["encoder"]
    assert isinstance(encoder, EncodeWorker) and encoder.crop == "letterbox"
    with pytest.raises(FileNotFoundError):
        build_stage_fns(_cfg(tmp_path, "bench.stages=null", f"gate_stage.model_path={tmp_path / 'missing.pt'}"),
                        _REPO, clips)


def test_gate_without_encoder_fails_fast(tmp_path: Path) -> None:
    path = _saved_bundle(tmp_path / "gate_bundle.pt")
    cfg = _cfg(tmp_path, "bench.stages=[decode,gate,mask]", f"gate_stage.model_path={path}")

    with pytest.raises(ValueError, match="encoder") as info:
        build_stage_fns(cfg, _REPO, bench.build_clips(cfg, _REPO))

    assert "gate" in str(info.value)
    fns, _ = build_stage_fns(_cfg(tmp_path, "bench.stages=[decode,encoder,gate]", f"gate_stage.model_path={path}"),
                             _REPO, bench.build_clips(cfg, _REPO))
    assert "gate" in fns


def test_s2_status_counts_gate_rejection_as_finished(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    clips = [BenchClip(f"c{i}", "100doh", "video", tmp_path / f"c{i}.mp4", 300, "train") for i in range(2)]
    manifest = ClipManifest.create(["c0", "c1"], ["100doh:train"] * 2, [300, 300])
    for stage in ("decode", "encoder", "gate", "mask"):
        manifest.mark("c0", stage, "done", 0.1)
    for stage in ("decode", "encoder"):
        manifest.mark("c1", stage, "done", 0.1)
    manifest.mark("c1", "gate", "failed", 0.01, error="gate rejected: p_interact 0.100000 < threshold 0.250000")

    cost = bench.s2_status(cfg, clips, manifest, ["decode", "encoder", "gate"], gate_status=bench.GATE_BUILT)
    full = bench.s2_status(cfg, clips, manifest, ["decode", "encoder", "gate", "mask"], gate_status=bench.GATE_BUILT)

    assert cost.startswith(bench.GATE_COST_ONLY + ": measured over the registered stages")
    assert cost == f"{bench.GATE_COST_ONLY}: measured over the registered stages ['decode', 'encoder', 'gate'] on 2 " \
                   "raw-stream clips"
    assert not cost.startswith(("measured", "unmeasured"))
    assert full.startswith("measured over the registered stages")
    manifest.mark("c1", "gate", "failed", 0.01, error="[Errno 2] encoder.npz not found; run the 'encoder' stage")
    errored = bench.s2_status(cfg, clips, manifest, ["decode", "encoder", "gate"], gate_status=bench.GATE_BUILT)
    assert errored.startswith("unmeasured") and "c1 stopped at gate" in errored


# --- the card had no false-positive column, and promotion named no bound -------------------------

def test_false_positive_rates_split_in_fold_from_out_of_fold() -> None:
    """The defect: a reader of the card could infer only the FIT rate. On the shipped gate
    that was 0.004 against 0.123 out of fold -- a 30x understatement of the deployable number -- so the
    two must be reported separately and `out_of_fold` must pool exactly `val` and `val_seq`."""
    import importlib.util
    import sys
    from pathlib import Path

    import torch

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("train_gate_mod", repo / "run/pipeline/train_gate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_gate_mod"] = mod
    spec.loader.exec_module(mod)

    class _DS:
        def __init__(self, interact, scores):
            self.labels = type("L", (), {"interact": torch.tensor(interact)})()
            self._scores = torch.tensor(scores)

    data = {                                    # negatives only matter; positives are ignored
        "fit": _DS([0.0] * 4 + [1.0], [0.0, 0.0, 0.0, 0.9, 1.0]),      # 1 of 4 above 0.5
        "val": _DS([0.0] * 4, [0.0, 0.9, 0.9, 0.9]),                   # 3 of 4
        "val_seq": _DS([0.0] * 4, [0.0, 0.0, 0.0, 0.9]),               # 1 of 4
        "calib": _DS([1.0, 1.0], [1.0, 1.0]),                          # no negatives: omitted entirely
    }
    out = mod.false_positive_rates(data, lambda ds: ds._scores, 0.5)

    assert out["by_partition"]["fit"]["rate"] == 0.25
    assert out["by_partition"]["val"]["rate"] == 0.75
    assert "calib" not in out["by_partition"], "a partition with no negatives has no false-positive rate"
    assert out["n_out_of_fold"] == 8
    assert out["out_of_fold"] == 0.5            # (3 + 1) / 8 -- val and val_seq ONLY, never fit
    assert mod.false_positive_rates(data, lambda ds: ds._scores, None)["out_of_fold"] is None


def test_the_card_records_whether_the_fpr_was_bounded_at_all() -> None:
    """A bound is opt-in so this breaks no existing run -- but its ABSENCE is now recorded. The defect
    was not that the bound was loose; it was that there was no column, so nobody could tell."""
    import json
    from pathlib import Path

    path = Path("/opt/fastdo/handreel/outputs/gate/gate_card.json")
    if not path.exists() and not (path.parents[2] / ".git").exists():   # no checkout there: the card is not shipped
        pytest.skip(f"the development host's gate card is not shipped: {path}")
    card = json.loads(path.read_text())
    # the shipped card predates this change, which is itself the point being pinned
    assert "false_positive_rates" not in card
    assert card["promoted"] is True and card["reasons"] == []
    assert card["min_val_auroc"] == 0.9, "recall and ranking were the only bars"
