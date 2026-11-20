"""Tests for `run/pipeline/bench_throughput.py`: faked stage workers, plus one real GPU run."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
import trimesh
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastdo.bench import ProfileReport
from fastdo.bench.yield_report import read_yield_report
from fastdo.engine.batched_stages import BatchedStage
from fastdo.engine.manifest import STAGE_NAMES, ClipManifest
from fastdo.engine.orchestrator import StageFn
from fastdo.io.datasets.dexycb import ycb_object_name
from fastdo.qc import AcceptanceRule, ConformalCalibration
from fastdo.qc.rule_io import save_acceptance_rule
from run.pipeline import bench_throughput as bench
from tests.bench.dexycb_tree import write_fake_dexycb
from tests.bench.fake_doh import install_file_access_spy, write_fake_doh_bundle
from tests.engine.fakes_batched import SleepBatchStage

_REPO = Path(__file__).resolve().parents[2]
_WHISKING_VIDEO = _REPO / "third_party/do-as-i-do/reconstruction/whisking/whisking.mp4"
_IDS = ["bench_whisking_000", "bench_whisking_001", "bench_whisking_002"]
_REAL_STAGES = ["decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks"]


def _cfg(tmp_path: Path, *overrides: str, fake_demo: bool = True) -> DictConfig:
    """The bench config for a test, with the execution path PINNED.

    `run_bench` selects its path from `bench.batched.enabled` alone, and that key defaults to true, so
    an unpinned config would silently move every test here off the single-stream reference path (which stays the
    reference). Each caller that wants a mode passes `bench.batched.enabled=...` itself; every other call gets
    single-stream explicitly, so a future flip in either direction cannot change what this suite exercises.
    """
    base = [f"paths.root={_REPO}", f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}"]
    if not any(o.startswith("bench.batched.enabled") for o in overrides):
        base.append("bench.batched.enabled=false")
    if fake_demo:
        base.append(f"whisking.demo_root={tmp_path / 'demo'}")
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[*base, *overrides])


def test_the_execution_path_is_pinned_by_the_tests_not_inherited_from_the_default(tmp_path: Path) -> None:
    """`run_bench` selects its path from `bench.batched.enabled` alone, which defaults to true.

    `_cfg` therefore pins single-stream unless the caller names a mode, so the reference path keeps its coverage and
    a future flip (either way) cannot silently change what this suite exercises.
    """
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        default = compose(config_name="config", overrides=[f"paths.root={_REPO}"])
    assert bool(OmegaConf.select(default, "bench.batched.enabled", default=False)) is True
    assert bool(OmegaConf.select(_cfg(tmp_path), "bench.batched.enabled", default=False)) is False
    assert bool(OmegaConf.select(_cfg(tmp_path, "bench.batched.enabled=true"), "bench.batched.enabled")) is True


def _fake_stage_fns(fail: frozenset[tuple[str, str]] = frozenset(), with_gate: bool = False) -> dict[str, StageFn]:
    """Every stage but the unbuilt gate (unless `with_gate`); the qc fake writes `qc.json`, accepting clips 000 and
    002."""

    def make(stage: str) -> StageFn:
        def worker(clip_id: str, artifact_dir: Path) -> None:
            time.sleep(0.002)
            if (clip_id, stage) in fail:
                raise RuntimeError(f"fake {stage} failure")
            if stage == "qc":
                accepted = clip_id.endswith(("000", "002"))
                (artifact_dir / "qc.json").write_text(json.dumps({"clip_id": clip_id, "accepted": accepted}))

        return worker

    return {stage: make(stage) for stage in STAGE_NAMES if with_gate or stage != "gate"}


def _payload(tmp_path: Path) -> dict[str, object]:
    return dict(json.loads((tmp_path / "bench" / "yield_report.json").read_text()))


@pytest.mark.parametrize("doh_root", ["null", "empty"])
def test_bench_throughput_reports_100doh_unavailable(tmp_path: Path, caplog: pytest.LogCaptureFixture,
                                                     doh_root: str) -> None:
    if doh_root == "empty":
        (tmp_path / "doh").mkdir()
        doh_root = str(tmp_path / "doh")
    cfg = _cfg(tmp_path, f"bench.doh_root={doh_root}", "bench.whisking_copies=3")

    with caplog.at_level(logging.WARNING):
        report = bench.run_bench(cfg, _REPO, stage_fns=_fake_stage_fns())

    assert any("S2" in r.getMessage() and "unmeasured" in r.getMessage() and "100DOH" in r.getMessage()
               for r in caplog.records)
    assert (tmp_path / "bench" / "manifest.parquet").is_file()
    payload = _payload(tmp_path)
    assert report.n_clips == payload["n_clips"] == 3
    assert ClipManifest.load(tmp_path / "bench" / "manifest.parquet").clip_ids() == _IDS
    assert str(payload["s2_raw_stream"]).startswith("unmeasured")
    assert payload["gate_status"] == "unbuilt"
    assert payload["n_gate_accepted"] == 3 and report.n_gate_accepted == 3
    assert payload["n_qc_accepted"] == 2 and report.yield_fraction == pytest.approx(2 / 3)
    assert "QC-only" in str(payload["yield_basis"])
    # epic_kitchens_100 and ego4d are in SOURCES so a run that HAS windowed clips reports them; a run
    # without them reports 0 rather than omitting the column, which keeps the report shape constant.
    assert payload["sources"] == {"whisking": 3, "dexycb": 0, "100doh": 0,
                                  "epic_kitchens_100": 0, "ego4d": 0, "hoi4d": 0,
                                  "oakink2_ego": 0, "oakink2_allo": 0}
    assert payload["clip_splits"] == {}
    assert payload["complete_pipeline"] is False
    assert any("not end to end" in r.getMessage() and "gate" in r.getMessage() for r in caplog.records)


def test_two_clips_per_gpu_hour_formulas_agree(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=3")

    report = bench.run_bench(cfg, _REPO, stage_fns=_fake_stage_fns(frozenset({(_IDS[1], "tracks")})))

    manifest = ClipManifest.load(tmp_path / "bench" / "manifest.parquet")
    for clip_id in _IDS:
        profile = ProfileReport.from_json(tmp_path / "engine" / clip_id / "profile.json")
        column_seconds = sum(manifest.status(clip_id, stage).seconds for stage in STAGE_NAMES)
        assert column_seconds > 0.0
        assert abs(profile.clips_per_gpu_hour() - 3600.0 / column_seconds) <= 1e-9
    assert manifest.status(_IDS[1], "tracks").state == "failed" and manifest.status(_IDS[1], "solve").state == "pending"
    total = sum(report.per_stage_seconds.values())
    assert report.clips_per_gpu_hour == pytest.approx(3 * 3600.0 / total, rel=1e-12)
    payload = _payload(tmp_path)
    assert float(str(payload["profile_formula_max_abs_diff"])) <= 1e-9
    assert payload["profile_formula_clips_compared"] == 3
    assert payload["n_qc_accepted"] == 2                    # clip 001 stopped at its failed tracks stage
    # default bench.stages leaves the encoder out; clip 001's failed tracks leaves tracks off the list
    assert payload["stages_done_on_every_clip"] == ["decode", "mask", "shape", "intrinsics", "hands", "geometry"]


def test_bench_throughput_runs_only_100doh_train_clips(tmp_path: Path, caplog: pytest.LogCaptureFixture,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    cfg = _cfg(tmp_path, f"bench.doh_root={bundle.root}", "bench.whisking_copies=1")
    seen: list[Path] = []

    def count_frames(path: Path) -> int:
        seen.append(path)
        return 250

    spy = install_file_access_spy(monkeypatch)
    with caplog.at_level(logging.WARNING):
        report = bench.run_bench(cfg, _REPO, stage_fns=_fake_stage_fns(with_gate=True), count_frames=count_frames)
    monkeypatch.undo()

    assert report.n_clips == 3 and seen == [bundle.train_paths[1]]
    for marker in bundle.held_out_markers:
        assert spy.touched(marker) == []
    table = pq.read_table(tmp_path / "bench" / "manifest.parquet")
    assert table.column("clip_id").to_pylist() == ["bench_whisking_000", *bundle.train_ids]
    assert table.column("source").to_pylist() == ["whisking", "100doh:train", "100doh:train"]
    assert not any("unmeasured" in r.getMessage() for r in caplog.records)
    payload = _payload(tmp_path)
    assert payload["sources"] == {"whisking": 1, "dexycb": 0, "100doh": 2,
                                  "epic_kitchens_100": 0, "ego4d": 0, "hoi4d": 0,
                                  "oakink2_ego": 0, "oakink2_allo": 0}
    assert payload["clip_splits"] == dict.fromkeys(bundle.train_ids, "train")
    assert str(payload["s2_raw_stream"]).startswith("measured")         # the (fake) gate ran: S2 can be measured
    assert not [p for p in tmp_path.rglob("*.mp4") if bundle.root not in p.parents]     # nothing copied


def test_s2_stays_unmeasured_when_100doh_clips_do_not_finish(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    cfg = _cfg(tmp_path, f"bench.doh_root={bundle.root}", "bench.whisking_copies=0")

    with caplog.at_level(logging.WARNING):
        bench.run_bench(cfg, _REPO, stage_fns=_fake_stage_fns(frozenset({(bundle.train_ids[0], "mask")})),
                        count_frames=lambda path: 250)

    s2 = str(_payload(tmp_path)["s2_raw_stream"])
    assert s2.startswith("unmeasured: 1 of 2 raw-stream clips") and "stopped at mask" in s2
    assert any("S2" in r.getMessage() and "unmeasured: 1 of 2" in r.getMessage() for r in caplog.records)


def test_encoder_without_its_package_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "encoder" not in bench.pipeline_stages(_cfg(tmp_path))       # default stages until transformers exists
    bench.check_stage_packages(["decode", "encoder"], find_spec=lambda name: object())
    with pytest.raises(RuntimeError, match="transformers"):
        bench.check_stage_packages(["decode", "encoder"], find_spec=lambda name: None)
    monkeypatch.setattr(bench, "STAGE_PACKAGES", {"encoder": "fastdo_no_such_package_d12"})

    with pytest.raises(RuntimeError, match="encoder"):
        bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.stages=[decode,encoder]"), _REPO)
    assert not (tmp_path / "bench").exists()


def test_bench_stages_run_in_pipeline_order_and_reject_unknown(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1", "bench.stages=[tracks,decode]")

    bench.run_bench(cfg, _REPO, stage_fns=_fake_stage_fns())

    manifest = ClipManifest.load(tmp_path / "bench" / "manifest.parquet")
    states = {stage: manifest.status("bench_whisking_000", stage).state for stage in STAGE_NAMES}
    assert [stage for stage, state in states.items() if state == "done"] == ["decode", "tracks"]
    status = _payload(tmp_path)["stage_status"]
    assert isinstance(status, dict)
    assert (status["decode"], status["mask"], status["gate"]) == ("injected", "excluded (bench.stages)",
                                                                  "excluded (bench.stages)")
    with pytest.raises(ValueError, match="bogus"):
        bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.stages=[decode,bogus]"), _REPO,
                        stage_fns=_fake_stage_fns())


def _saved_rule(path: Path) -> Path:
    save_acceptance_rule(path, AcceptanceRule(calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
                                              logistic_weights=np.zeros(6), feature_mean=np.zeros(5),
                                              feature_std=np.ones(5)))
    return path


def test_real_stage_fns_leave_unbuilt_stages_unregistered(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=2")
    clips = bench.build_clips(cfg, _REPO)

    fns, status = bench.build_stage_fns(cfg, _REPO, clips)

    assert set(fns) == {"decode", "encoder", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve"}
    assert status["gate"] == "unbuilt"
    assert status["solve"].startswith("computed (M1 solver")
    assert status["qc"].startswith("not registered: qc_stage.rule_path is null")
    assert status["retarget"].startswith("not registered")
    assert status["hands"].startswith("computed") and status["shape"] == "computed (sam3d)"
    _, borrowed = bench.build_stage_fns(_cfg(tmp_path, "bench.hands_source=precomputed"), _REPO, clips)
    assert borrowed["hands"].startswith("borrowed")
    rule = _saved_rule(tmp_path / "rule.json")
    with_qc, with_qc_status = bench.build_stage_fns(_cfg(tmp_path, f"qc_stage.rule_path={rule}"), _REPO, clips)
    assert "qc" in with_qc and with_qc_status["qc"].startswith("computed (rule ")
    assert (with_qc_status["gate"], with_qc_status["retarget"]) == (status["gate"], status["retarget"])


def test_bench_refuses_the_qc_stage_without_a_rule(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "bench.doh_root=null", "qc_stage.rule_path=null")
    assert "qc" in bench.pipeline_stages(cfg)                           # the default bench.stages

    with pytest.raises(ValueError, match=r"qc_stage\.rule_path") as info:
        bench.run_bench(cfg, _REPO)

    assert "bench.stages" in str(info.value) and "accept-all" in str(info.value)     # names both keys
    assert not (tmp_path / "bench" / "manifest.parquet").exists()
    assert not (tmp_path / "engine").exists() or not any((tmp_path / "engine").iterdir())


def test_dexycb_template_shape_is_labelled_borrowed(tmp_path: Path) -> None:
    seq = write_fake_dexycb(tmp_path / "dexycb")
    template = tmp_path / "dexycb" / "models" / ycb_object_name(5) / "textured_simple.obj"
    overrides = (f"dexycb.root={tmp_path / 'dexycb'}", f"bench.dexycb_sequences=['{seq}']", "bench.whisking_copies=0",
                 "bench.doh_root=null")
    cfg = _cfg(tmp_path, *overrides, "bench.dexycb_shape=template")
    clips = bench.build_clips(cfg, _REPO)

    with pytest.raises(FileNotFoundError, match="textured_simple.obj"):
        bench.build_stage_fns(cfg, _REPO, clips)
    template.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.05, 0.02)).export(str(template))
    _, status = bench.build_stage_fns(cfg, _REPO, clips)

    assert status["shape"].startswith("borrowed (GT YCB template for dexycb clips")
    with pytest.raises(ValueError, match="dexycb_shape"):
        bench.build_stage_fns(_cfg(tmp_path, *overrides, "bench.dexycb_shape=mesh"), _REPO, clips)


@pytest.mark.parametrize(("with_gate", "gate_status"), [(False, "unbuilt"), (True, "excluded (bench.stages)")])
def test_s2_is_unmeasured_unless_the_gate_ran(tmp_path: Path, with_gate: bool, gate_status: str) -> None:
    """S2 is measured only when the interaction gate ran on the clips.
    With the gate unbuilt, or built but excluded by `bench.stages`, it is unmeasured and names that reason, although
    every 100DOH clip finished every stage that ran."""
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    cfg = _cfg(tmp_path, f"bench.doh_root={bundle.root}", "bench.whisking_copies=0", "bench.stages=[decode]")

    bench.run_bench(cfg, _REPO, stage_fns=_fake_stage_fns(with_gate=with_gate), count_frames=lambda path: 250)

    manifest = ClipManifest.load(tmp_path / "bench" / "manifest.parquet")
    assert all(manifest.status(clip_id, "decode").state == "done" for clip_id in bundle.train_ids)
    payload = _payload(tmp_path)
    s2 = str(payload["s2_raw_stream"])
    assert payload["gate_status"] == gate_status
    assert s2.startswith("unmeasured: 2 of 2") and f"the interaction gate is {gate_status}" in s2
    assert ("unbuilt" in s2) is not with_gate
    clips = bench.build_clips(cfg, _REPO, count_frames=lambda path: 250)
    not_run = bench.s2_status(cfg, clips, manifest, ["decode"], gate_status=bench.GATE_BUILT)   # gate not among the run
    assert not_run.startswith("unmeasured") and "did not run" in not_run


def _real_run_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "needs the GPU"
    for pkg in ("moge", "sam2"):
        if importlib.util.find_spec(pkg) is None:
            return False, f"needs the frontend extra ({pkg})"
    if not _WHISKING_VIDEO.is_file():
        return False, "needs the shipped whisking demo"
    if not os.environ.get("FASTDO_SAM3D_PYTHON"):
        return False, "needs FASTDO_SAM3D_PYTHON (bench.shape.backend sam3d)"
    return True, ""


@pytest.mark.timing
@pytest.mark.skipif(not _real_run_available()[0], reason=_real_run_available()[1])
def test_bench_throughput_runs_on_whisking_fallback(tmp_path: Path) -> None:
    """The GPU run: the default `bench.stages` without qc (no saved AcceptanceRule), whisking x3."""
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=3",
               "bench.stages=[decode,gate,mask,shape,intrinsics,hands,geometry,tracks,solve,retarget]", fake_demo=False)

    bench.main(cfg)

    assert (tmp_path / "bench" / "manifest.parquet").is_file()
    payload = read_yield_report(tmp_path / "bench" / "yield_report.json")
    assert payload["n_clips"] == 3
    manifest = ClipManifest.load(tmp_path / "bench" / "manifest.parquet")
    errors = {(c, s): manifest.status(c, s).error for c in _IDS for s in _REAL_STAGES
              if manifest.status(c, s).state != "done"}
    assert errors == {}
    assert payload["stage_states"]["encoder"] == {"done": 0, "failed": 0, "pending": 3}
    assert {"encoder", "gate", "qc", "retarget"} <= set(payload["unmeasured_stages"])
    assert payload["stages_done_on_every_clip"][:len(_REAL_STAGES)] == _REAL_STAGES
    assert payload["profile_formula_clips_compared"] == 3 and payload["profile_formula_max_abs_diff"] <= 1e-9


def test_yield_report_records_gate_outcomes(tmp_path: Path) -> None:
    """A gate rejection stops its clip and is counted as rejected, not as an error, beside `n_gate_accepted`."""
    from fastdo.engine.workers_gate import GateRejected

    fns = _fake_stage_fns(with_gate=True)

    def gate(clip_id: str, artifact_dir: Path) -> None:
        if clip_id == "bench_whisking_001":
            raise GateRejected(0.1, 0.5)

    fns["gate"] = gate
    bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=3"), _REPO, stage_fns=fns)

    payload = _payload(tmp_path)
    assert payload["gate_status"] == "built" and payload["n_gate_accepted"] == 2
    assert payload["gate_outcomes"] == {"accepted": 2, "rejected": 1, "error": 0, "pending": 0}
    manifest = ClipManifest.load(tmp_path / "bench" / "manifest.parquet")
    assert manifest.status("bench_whisking_001", "mask").state == "pending"


def test_bench_throughput_batched_mode_matches_single_stream(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """`bench.batched.enabled` runs the same fakes through `run_manifest_batched`, one encoder batch for 3 clips."""
    # The promoted per-stage budgets are measured under a 21 GiB per-process cap and `check_stage_budgets` now
    # validates against the cap this process actually applies, so composing that config takes the same cap.
    monkeypatch.setenv("FASTDO_GPU_MEM_GB", "21")
    stages = "bench.stages=[decode,encoder,mask,shape,intrinsics,hands,geometry,tracks,solve,qc,retarget]"
    runs = {}
    for mode in ("false", "true"):
        fns: dict[str, StageFn | BatchedStage] = {**_fake_stage_fns()}
        if mode == "true":
            fns["encoder"] = SleepBatchStage(0.005)
        # The promoted default holds encoder at max_batch 1, labelled unmeasured: no measurement
        # covers a real encoder batch. This test drives a *fake* BatchedStage to check the orchestration wiring, so
        # it lifts that cap explicitly -- exactly the deliberate edit the label is there to force -- and declares the
        # batch's memory, without which a batched stage is now refused outright. 3 x 2.0 GiB
        # fits the 6.0 GiB budget, so the three clips are one batch, as the assertions below expect.
        cfg = _cfg(tmp_path / mode, "bench.doh_root=null", "bench.whisking_copies=3", stages,
                   f"bench.batched.enabled={mode}", "bench.batched.stages.encoder.unmeasured=false",
                   "bench.batched.stages.encoder.max_batch=3", "+bench.batched.stages.encoder.budget_gb=6.0",
                   "+bench.batched.stages.encoder.fixed_cost_gb=2.0")
        runs[mode] = (bench.run_bench(cfg, _REPO, stage_fns=fns), _payload(tmp_path / mode))
    (single, single_payload), (batched, payload) = runs["false"], runs["true"]
    assert single.n_clips == batched.n_clips == 3 and single.n_qc_accepted == batched.n_qc_accepted == 2
    assert single_payload["stage_states"] == payload["stage_states"]
    assert (single_payload["execution"], payload["execution"]) == ("single_stream", "batched")
    apportionment = payload["apportionment"]
    assert isinstance(apportionment, dict) and set(apportionment["unattributed_ns"].values()) == {0}
    assert float(str(payload["profile_formula_max_abs_diff"])) <= 1e-9   # ProfileReport's sum() is compensated
    manifest = ClipManifest.load(tmp_path / "true" / "bench" / "manifest.parquet")
    assert len({manifest.status(c, "encoder").batch_id for c in _IDS}) == 1 and manifest.status(_IDS[0], "encoder").batch_id
    bad = _cfg(tmp_path / "bad", "bench.doh_root=null", stages, "bench.batched.enabled=false")
    with pytest.raises(TypeError, match="encoder"):
        bench.run_bench(bad, _REPO, stage_fns={**_fake_stage_fns(), "encoder": SleepBatchStage(0.005)})
    assert not (tmp_path / "bad" / "bench" / "manifest.parquet").exists()


def test_the_yield_report_records_both_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F1's provenance half: `gpu_mem_cap_gb` is the raw environment, None on a default-env run, while the config
    asserts `cap_gb: 32` (formerly 21) -- so a run could disagree with itself about the cap it ran under. The report
    now carries the cap the process applies and the cap the per-stage budgets are declared against, alongside the raw
    value."""
    monkeypatch.delenv("FASTDO_GPU_MEM_GB", raising=False)
    bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1"), _REPO,
                    stage_fns=_fake_stage_fns())
    payload = _payload(tmp_path)
    assert payload["gpu_mem_cap_gb"] is None            # the environment says nothing ...
    assert payload["applied_gpu_mem_cap_gb"] == 12.0    # ... but the process still caps itself at gpu_budget default
    assert payload["declared_cap_gb"] == 32.0           # and the promoted budgets are declared against 32

    monkeypatch.setenv("FASTDO_GPU_MEM_GB", "32")
    bench.run_bench(_cfg(tmp_path / "capped", "bench.doh_root=null", "bench.whisking_copies=1"), _REPO,
                    stage_fns=_fake_stage_fns())
    capped = _payload(tmp_path / "capped")
    assert (capped["gpu_mem_cap_gb"], capped["applied_gpu_mem_cap_gb"], capped["declared_cap_gb"]) == ("32", 32.0, 32.0)


@pytest.mark.timing
@pytest.mark.skipif(not _real_run_available()[0], reason=_real_run_available()[1])
def test_bench_throughput_batched_on_whisking(tmp_path: Path) -> None:
    """The GPU run in batched mode (whisking x3, no qc rule).

    Runs under `FASTDO_GPU_MEM_GB=21`, the cap the promoted budgets were measured under: `check_stage_budgets`
    validates against the cap the process applies, so a smaller one refuses this config at load (solve's 13.39 GiB).
    """
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=3", "bench.batched.enabled=true",
               "bench.stages=[decode,gate,mask,shape,intrinsics,hands,geometry,tracks,solve,retarget]", fake_demo=False)
    bench.main(cfg)
    payload = read_yield_report(tmp_path / "bench" / "yield_report.json")
    assert (tmp_path / "bench" / "manifest.parquet").is_file() and payload["n_clips"] == 3
    assert payload["execution"] == "batched"


def test_contact_stage_is_opt_in_like_driftqc() -> None:
    """The contact correction must not appear in a run that did not ask for it, and must be namable when it did."""
    from omegaconf import OmegaConf

    from run.pipeline.bench_throughput import known_stages, pipeline_stages

    off = OmegaConf.create({"bench": {"stages": None}})
    assert "contact" not in known_stages(off)

    on = OmegaConf.create({"bench": {"stages": ["solve", "contact", "qc"]}, "contact_stage": {"enabled": True}})
    assert known_stages(on).index("contact") == known_stages(on).index("solve") + 1
    assert pipeline_stages(on) == ("solve", "contact", "qc")

    asked_but_off = OmegaConf.create({"bench": {"stages": ["solve", "contact"]}})
    with pytest.raises(ValueError, match="contact_stage.enabled is false"):
        pipeline_stages(asked_but_off)

    bad = OmegaConf.create({"bench": {"stages": None}, "contact_stage": {"enabled": True, "cell": "nonsense"}})
    with pytest.raises(ValueError, match="cell"):
        known_stages(bad)


def test_prompt_stage_is_opt_in_and_precedes_mask() -> None:
    """S2's P3 stage must sit before mask (which consumes its artifact) and never appear unasked."""
    from omegaconf import OmegaConf

    from run.pipeline.bench_throughput import known_stages, pipeline_stages

    off = OmegaConf.create({"bench": {"stages": None}})
    assert "prompt" not in known_stages(off)

    on = OmegaConf.create({"bench": {"stages": None}, "prompt_stage": {"enabled": True, "min_score": 0.1}})
    stages = known_stages(on)
    assert stages.index("prompt") == stages.index("mask") - 1
    assert stages.index("decode") < stages.index("prompt")

    asked_but_off = OmegaConf.create({"bench": {"stages": ["decode", "prompt", "mask"]}})
    with pytest.raises(ValueError, match="prompt_stage.enabled is false"):
        pipeline_stages(asked_but_off)


# -- resume: a killed run costs the interrupted clip, not the corpus ---------------------------------------------

def _counting_stage_fns(calls: list[tuple[str, str]], fail: frozenset[tuple[str, str]] = frozenset(),
                        ) -> dict[str, StageFn]:
    """Fake stages that RECORD every (clip, stage) they run, so a resumed run's skips are observable."""

    def make(stage: str) -> StageFn:
        def worker(clip_id: str, artifact_dir: Path) -> None:
            calls.append((clip_id, stage))
            if (clip_id, stage) in fail:
                raise RuntimeError(f"fake {stage} failure")
            if stage == "qc":
                (artifact_dir / "qc.json").write_text(json.dumps({"clip_id": clip_id, "accepted": True}))

        return worker

    return {stage: make(stage) for stage in STAGE_NAMES if stage != "gate"}


def test_a_resumed_run_reruns_only_what_did_not_finish(tmp_path: Path) -> None:
    """The point of resume: on a shared card a long run WILL be killed, so the rerun must cost the lost work only."""
    stages = "bench.stages=[decode,mask,shape]"
    first: list[tuple[str, str]] = []
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=3", stages)
    bench.run_bench(cfg, _REPO, stage_fns=_counting_stage_fns(first, fail=frozenset({(_IDS[2], "shape")})))
    assert (_IDS[2], "shape") in first and len(first) == 9            # every clip-stage was attempted

    second: list[tuple[str, str]] = []
    resumed = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=3", stages,
                   "bench.overwrite=false", "bench.resume=true")
    report = bench.run_bench(resumed, _REPO, stage_fns=_counting_stage_fns(second))

    assert second == [(_IDS[2], "shape")]                             # ONLY the stage that failed
    assert report.n_clips == 3


def test_resume_without_a_previous_manifest_is_a_normal_run(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=2", "bench.stages=[decode,mask]",
               "bench.overwrite=false", "bench.resume=true")
    bench.run_bench(cfg, _REPO, stage_fns=_counting_stage_fns(calls))
    assert len(calls) == 4


def test_resume_and_overwrite_contradict_each_other(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1", "bench.stages=[decode]",
               "bench.overwrite=true", "bench.resume=true")
    with pytest.raises(ValueError, match="contradict"):
        bench.run_bench(cfg, _REPO, stage_fns=_counting_stage_fns([]))


def test_a_resumed_run_refuses_a_different_clip_set(tmp_path: Path) -> None:
    base = ["bench.doh_root=null", "bench.stages=[decode]"]
    bench.run_bench(_cfg(tmp_path, *base, "bench.whisking_copies=3"), _REPO, stage_fns=_counting_stage_fns([]))
    cfg = _cfg(tmp_path, *base, "bench.whisking_copies=2", "bench.overwrite=false", "bench.resume=true")
    with pytest.raises(ValueError, match="different settings"):      # the clip count is a config key: refused first
        bench.run_bench(cfg, _REPO, stage_fns=_counting_stage_fns([]))
    # the manifest's own check stays: the same config can still yield another clip set (an inventory file changed)
    with pytest.raises(ValueError, match="same clip set"):
        bench.resumed_manifest(bench.build_clips(cfg, _REPO), tmp_path / "bench" / bench.MANIFEST_FILE, True, ["decode"])


def test_a_resumed_run_refuses_changed_settings(tmp_path: Path) -> None:
    """Resume reuses stages marked done, so it must be the SAME run: a changed stage list or a changed stage
    setting would leave results of the old settings beside results of the new ones."""
    base = ["bench.doh_root=null", "bench.whisking_copies=2"]
    bench.run_bench(_cfg(tmp_path, *base, "bench.stages=[decode]"), _REPO, stage_fns=_counting_stage_fns([]))
    again = ["bench.overwrite=false", "bench.resume=true"]
    for changed in (["bench.stages=[decode,mask]"], ["bench.stages=[decode]", "solve_stage.registration.n_obs=256"]):
        calls: list[tuple[str, str]] = []
        with pytest.raises(ValueError, match="different settings"):
            bench.run_bench(_cfg(tmp_path, *base, *changed, *again), _REPO, stage_fns=_counting_stage_fns(calls))
        assert calls == []                                            # refused before any stage ran
    bench.run_bench(_cfg(tmp_path, *base, "bench.stages=[decode]", *again), _REPO, stage_fns=_counting_stage_fns([]))


def test_a_resumed_run_refuses_a_manifest_without_a_settings_record(tmp_path: Path) -> None:
    base = ["bench.doh_root=null", "bench.whisking_copies=1", "bench.stages=[decode]"]
    bench.run_bench(_cfg(tmp_path, *base), _REPO, stage_fns=_counting_stage_fns([]))
    (tmp_path / "bench" / bench.RESUME_SETTINGS_FILE).unlink()
    with pytest.raises(ValueError, match="no settings record"):
        bench.run_bench(_cfg(tmp_path, *base, "bench.overwrite=false", "bench.resume=true"), _REPO,
                        stage_fns=_counting_stage_fns([]))


def test_a_resumed_run_refuses_a_rule_file_replaced_at_the_same_path(tmp_path: Path) -> None:
    """The config names the acceptance rule by path. Verdicts computed under one rule must not be pooled with
    verdicts under another, so the settings record covers the BYTES of the named rule files, not only their paths."""
    rule = tmp_path / "rule.json"
    rule.write_text('{"threshold": 1}')
    base = ["bench.doh_root=null", "bench.whisking_copies=1", "bench.stages=[decode]", f"qc_stage.rule_path={rule}"]
    again = ["bench.overwrite=false", "bench.resume=true"]
    bench.run_bench(_cfg(tmp_path, *base), _REPO, stage_fns=_counting_stage_fns([]))
    bench.run_bench(_cfg(tmp_path, *base, *again), _REPO, stage_fns=_counting_stage_fns([]))      # same bytes: resumes
    rule.write_text('{"threshold": 2}')
    with pytest.raises(ValueError, match="different settings"):
        bench.run_bench(_cfg(tmp_path, *base, *again), _REPO, stage_fns=_counting_stage_fns([]))
    rule.unlink()                                                             # an absent file is another setting too
    with pytest.raises(ValueError, match="different settings"):
        bench.run_bench(_cfg(tmp_path, *base, *again), _REPO, stage_fns=_counting_stage_fns([]))


def test_resume_without_a_manifest_does_not_adopt_clip_directories_that_exist(tmp_path: Path) -> None:
    """With an empty output directory there is nothing to compare settings with, so resume is a normal run: a
    populated artifact root is refused as it is without resume, and files of an older run are not kept."""
    base = ["bench.doh_root=null", "bench.whisking_copies=1", "bench.stages=[decode]"]
    bench.run_bench(_cfg(tmp_path, *base), _REPO, stage_fns=_counting_stage_fns([]))
    (tmp_path / "bench" / bench.MANIFEST_FILE).unlink()
    (tmp_path / "bench" / bench.RESUME_SETTINGS_FILE).unlink()
    assert (tmp_path / "engine" / _IDS[0]).is_dir()
    calls: list[tuple[str, str]] = []
    with pytest.raises(FileExistsError):
        bench.run_bench(_cfg(tmp_path, *base, "bench.overwrite=false", "bench.resume=true"), _REPO,
                        stage_fns=_counting_stage_fns(calls))
    assert calls == []


def test_resume_keeps_the_prompt_the_previous_run_produced(tmp_path: Path) -> None:
    """A resumed run must not overwrite a stage-produced mask_prompt.npz with the dataset label."""
    base = ["bench.doh_root=null", "bench.whisking_copies=1", "bench.stages=[decode]"]
    bench.run_bench(_cfg(tmp_path, *base), _REPO, stage_fns=_counting_stage_fns([]))
    marker = tmp_path / "engine" / _IDS[0] / "mask_prompt.npz"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"produced by the prompt stage, not the label")

    cfg = _cfg(tmp_path, *base, "bench.overwrite=false", "bench.resume=true")
    bench.run_bench(cfg, _REPO, stage_fns=_counting_stage_fns([]))
    assert marker.read_bytes() == b"produced by the prompt stage, not the label"


def test_build_clips_reaches_the_windowed_builder(tmp_path: Path) -> None:
    """The registered runner must be able to BUILD an S2 clip list, not merely contain a builder for one.

    P3 and P4 both shipped complete and tested and were still unrunnable, because nothing in the registered
    path could invoke them. This walks the real route -- config key -> `build_clips` -> `windowed_clips` --
    on real files, so an inventory that the runner cannot reach fails here instead of on the S2 run.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    for name, fps in (("a.mp4", 50), ("b.mp4", 25)):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"testsrc=size=32x32:rate={fps}:duration=25", str(videos / name)], check=True)
    inventory = tmp_path / "windows.json"
    inventory.write_text(json.dumps([
        {"video_id": "P18_03", "path": str(videos / "a.mp4"), "source": "epic_kitchens_100", "split": "TEST"},
        {"video_id": "ego_x", "path": str(videos / "b.mp4"), "source": "ego4d", "split": "CALIB"},
    ]))
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1",
               f"bench.windows_inventory={inventory}")

    clips = bench.build_clips(cfg, _REPO)

    windows = [c for c in clips if c.window is not None]
    assert windows, "build_clips produced no windowed clips: the inventory never reached windowed_clips"
    # 25 s of video, 10 s windows after a seeded offset -> 1 or 2 windows each; both videos represented
    assert {c.source for c in windows} == {"epic_kitchens_100", "ego4d"}
    # the operating rate, not the run stride: 50 fps -> 5, 25 fps -> 3 (round(2.5)), and bench.stride is 3
    assert {c.source: c.decode_stride for c in windows} == {"epic_kitchens_100": 5, "ego4d": 3}
    assert all(c.n_frames == c.window[1] - c.window[0] for c in windows)
    # and the whisking clips still come through unwindowed
    assert any(c.window is None for c in clips)


def test_s2_status_counts_internet_video_as_a_raw_stream(tmp_path: Path) -> None:
    """S2 is raw-stream yield and cost. 100DOH was the only such source for a long time, so `s2_status`
    tested `source == "100doh"` -- and reported S2 UNMEASURED on a run made entirely of Ego4D windows,
    which is the run S2 exists to measure. Observed on the first S2 pilot, which called 4 Ego4D windows
    'the whisking/DexYCB fallback set'."""
    from fastdo.bench.bench_clips import BenchClip

    cfg = _cfg(tmp_path, "bench.doh_root=null")
    windows = [BenchClip(clip_id=f"v:k{k:03d}", source="ego4d", kind="video", path=Path("/v.mp4"),
                         n_frames=300, split="CALIB", window=(k * 300, (k + 1) * 300), decode_stride=3)
               for k in range(4)]

    assert "no raw-stream clips" not in bench.s2_status(cfg, windows)
    assert bench.s2_status(cfg, windows) == "pending: 4 raw-stream clips selected"

    # a lab-only clip list is still correctly reported as having no raw stream
    lab = [BenchClip(clip_id="w0", source="whisking", kind="video", path=Path("/w.mp4"), n_frames=100)]
    assert bench.s2_status(cfg, lab).startswith("unmeasured: no raw-stream clips")


def test_a_prompt_refusal_finishes_a_window_just_as_a_gate_rejection_does(tmp_path: Path) -> None:
    """A gate rejection FINISHES a clip, so `s2_status` does not count it unfinished.
    The prompt stage (S2's P3) also refuses by raising -- `run_clip` marks it "failed" -- but `s2_status`
    knew only about the gate, so a window the detector legitimately declined read as an unfinished clip
    and the whole run reported UNMEASURED.

    This is the same bug as `test_s2_status_counts_internet_video_as_a_raw_stream` one stage over: a
    correct S2 run, reported as not measured, because a new S2 stage was not taught to an old predicate.
    The pilot's refusal rate was 2 of 4, so this fires on a realistic CALIB run, not an exotic one.
    """
    from fastdo.bench.bench_clips import BenchClip
    from fastdo.engine.workers_prompt import PROMPT_REFUSED_PREFIX

    cfg = _cfg(tmp_path, "bench.doh_root=null")
    windows = [BenchClip(clip_id=f"v:k{k:03d}", source="ego4d", kind="video", path=Path("/v.mp4"),
                         n_frames=300, split="CALIB", window=(k * 300, (k + 1) * 300), decode_stride=3)
               for k in range(2)]
    stages = ["decode", "prompt", "gate", "mask"]     # a stage after the gate, so this is S2 and not GATE_COST_ONLY
    manifest = ClipManifest.create([c.clip_id for c in windows], ["ego4d"] * 2, [300] * 2)
    for clip in windows:
        manifest.mark(clip.clip_id, "decode", "done", 1.0)
    # one window builds a prompt and passes the gate; the other is REFUSED by the detector
    for stage in ("prompt", "gate", "mask"):
        manifest.mark("v:k000", stage, "done", 1.0)
    manifest.mark("v:k001", "prompt", "failed", 1.0,
                  error=f"{PROMPT_REFUSED_PREFIX}: v:k001: no confident hand-object contact on rows [0]")

    status = bench.s2_status(cfg, windows, manifest, stages, gate_status=bench.GATE_BUILT)
    assert status.startswith("measured"), status

    # a genuine prompt ERROR is still unfinished -- the point is to tell the two apart, not to excuse both
    broken = ClipManifest.create([c.clip_id for c in windows], ["ego4d"] * 2, [300] * 2)
    for clip in windows:
        broken.mark(clip.clip_id, "decode", "done", 1.0)
    for stage in ("prompt", "gate", "mask"):
        broken.mark("v:k000", stage, "done", 1.0)
    broken.mark("v:k001", "prompt", "failed", 1.0, error="CUDA out of memory")
    assert bench.s2_status(cfg, windows, broken, stages, gate_status=bench.GATE_BUILT).startswith("unmeasured")


def test_the_yield_report_carries_the_prompt_outcomes(tmp_path: Path) -> None:
    """Gate B criterion 5 is the prompt refusal rate, and a counter nothing calls is not a measurement.

    `prompt_outcome_counts` classifies the rows (its own unit tests cover that); this asserts the number
    actually REACHES the report, which is the half this project has got wrong four times. Here no prompt
    stage is registered, so every clip is pending -- the point is that the key is emitted and totals.
    """
    bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=3"), _REPO,
                    stage_fns=_fake_stage_fns(with_gate=True))

    outcomes = _payload(tmp_path)["prompt_outcomes"]
    assert set(outcomes) == {"built", "refused", "error", "pending"}      # type: ignore[arg-type]
    assert sum(outcomes.values()) == 3                                    # type: ignore[union-attr]


def test_the_completeness_report_covers_stages_outside_STAGE_NAMES(tmp_path: Path) -> None:
    """`stages_done_on_every_clip` and the gap WARNING both iterated STAGE_NAMES, which does not contain
    S2's `prompt` (with_prompt inserts it into the run's stage list, not into that tuple). So the prompt
    stage could never be reported complete, and -- worse -- could never be named as the gap, which is the
    one job the warning has. Third instance of "a new S2 stage meets an old predicate"."""
    bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=2"), _REPO,
                    stage_fns=_fake_stage_fns(with_gate=True))

    manifest = ClipManifest.load(tmp_path / "bench" / "manifest.parquet")
    for clip_id in manifest.clip_ids():
        manifest.mark(clip_id, "prompt", "done", 1.0)
    manifest.save(tmp_path / "bench" / "manifest.parquet")

    from fastdo.bench.yield_report import stage_order
    assert "prompt" in stage_order(manifest)                    # the manifest carries it...
    from fastdo.engine.manifest import STAGE_NAMES
    assert "prompt" not in STAGE_NAMES                          # ...and the canonical tuple does not


def test_the_yield_report_carries_the_timebase_flags(tmp_path: Path) -> None:
    """The timebase flag has to REACH the report, or the guard is present on disk and absent from analysis.

    The windowing rule says a flagged window "stays in every analysis". Recording `timebase_mismatch`
    in `decode.npz` and never reading it back would satisfy the letter and lose the point -- the same
    unreachable-feature failure this repo has paid for repeatedly.

    `unrecorded` must be its own bucket, not folded into `clean`: these clips decode whole files (no
    window), so the check is undefined for them and they have not passed it, they did not take it.
    """
    bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1"), _REPO,
                    stage_fns=_fake_stage_fns())
    flags = _payload(tmp_path)["timebase_flags"]

    assert set(flags) == {"flagged", "clean", "unrecorded"}
    assert sum(flags.values()) > 0, "every clip of the run must land in exactly one bucket"
    # This bench decodes WHOLE FILES (whisking copies, no window), and the check is defined for windows,
    # so every clip should be `unrecorded`. Asserting that rather than the weaker `flagged == 0`, which
    # is trivially true when nothing took the check -- the same vacuous-pass shape being guarded against.
    assert flags["unrecorded"] == sum(flags.values())
    assert flags["flagged"] == 0 and flags["clean"] == 0


def test_every_run_writes_the_decode_index_audit(tmp_path: Path) -> None:
    """The persisted index must be written BY THE RUN, not by an operator who remembers to.

    `decode.npz` carries Gate A criterion 2's `frame_idx` in the same file as the pixels, and two separate
    rules delete it: the Ego4D licence's cleanup and the artifact pruning. The gate's fallback to
    a persisted index existed, correct and tested, while NOTHING wrote the file -- so on a real pruned run
    the gate would have gone quiet rather than failed.

    This bench's decode stage is a fake that writes no `decode.npz`, which makes it the more useful
    assertion: the audit must name every clip and record each one as an ERROR, not invent an entry. A
    writer that quietly emitted plausible indices for windows it never read would be worse than no writer
    at all, because the gate would then pass on them.
    """
    import json

    from fastdo.engine.decode_index import DECODE_INDEX_FILE, index_entry, load_index_audit

    bench.run_bench(_cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1"), _REPO,
                    stage_fns=_fake_stage_fns())

    payload = _payload(tmp_path)
    audit_path = Path(payload["decode_index_audit"])
    assert audit_path.name == DECODE_INDEX_FILE and audit_path.is_file(), payload["decode_index_audit"]

    raw = json.loads(audit_path.read_text())
    assert raw, "the audit must name the run's clips, not be an empty object"
    assert all(entry == {"error": "no decode.npz"} for entry in raw.values()), raw
    assert load_index_audit(audit_path) == {}          # and an error entry never reads as usable

    # ...and with a real decode present, the same writer produces exactly the fields the gate reads.
    artifact_dir = audit_path.parent / "artifacts" / next(iter(raw))
    if artifact_dir.is_dir():
        np.savez(artifact_dir / "decode.npz", frame_idx=np.arange(0, 12, 3, dtype=np.int64),
                 stride=np.int64(3), frames=np.zeros((4, 2, 2, 3), np.uint8))
        assert index_entry(artifact_dir) == {"n": 4, "first": 0, "last": 9, "stride": 3,
                                             "contiguous_stride": True}


def test_the_windowed_builder_can_take_the_screened_frame_count(tmp_path: Path, monkeypatch) -> None:
    """The packet-table route must be REACHABLE from the registered runner, and must not change the windows.

    `probe_windowed_clips` counts DECODED frames by default, which is O(video): 13.55 s for a 203 s file and
    up to 235.7 s for a 1033 s one. Placing the windows of the 21-video Ego4D draw therefore costs ~70 min
    BEFORE any window is known to be worth processing. The packet-table count is permitted for a source whose
    `nb_frames` has been validated against decoded counts (exact on all 21), and the trim screen makes it
    safe per file -- but `windowed_clips_from_inventory` was never handed the runner's `count_frames`, so the
    permission was unreachable from the config. That is the same defect as P3 and P4: built, tested, unrunnable.

    Both halves are asserted here. Reachability alone would pass while silently changing eligibility, and an
    equivalence taken from a note is not one this test observed.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    for name, fps in (("a.mp4", 50), ("b.mp4", 25)):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"testsrc=size=32x32:rate={fps}:duration=25", str(videos / name)], check=True)
    inventory = tmp_path / "windows.json"
    inventory.write_text(json.dumps([
        {"video_id": "P18_03", "path": str(videos / "a.mp4"), "source": "epic_kitchens_100", "split": "TEST"},
        {"video_id": "ego_x", "path": str(videos / "b.mp4"), "source": "ego4d", "split": "CALIB"},
    ]))

    from fastdo.io.video import count_decoded_frames as real_decoded
    from fastdo.io.video import screened_frame_count as real_screened
    decoded_calls: list[Path] = []
    screened_calls: list[Path] = []

    def decoded_spy(path: Path) -> int:
        decoded_calls.append(Path(path))
        return real_decoded(path)

    def screened_spy(path: Path) -> int:
        screened_calls.append(Path(path))
        return real_screened(path)

    def ids(*overrides: str) -> list[str]:
        cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1",
                   f"bench.windows_inventory={inventory}", *overrides)
        return sorted(c.clip_id for c in bench.build_clips(cfg, _REPO, decoded_spy) if c.window is not None)

    default_ids = ids()
    assert decoded_calls, "the default route must still count decoded frames"

    monkeypatch.setattr(bench, "screened_frame_count", screened_spy)
    decoded_calls.clear()
    screened_ids = ids("bench.windows_frame_count=screened")

    assert screened_calls, "bench.windows_frame_count=screened never reached screened_frame_count"
    assert decoded_calls == [], f"the screened route still paid the O(video) decode for {decoded_calls}"
    assert screened_ids == default_ids, "the screened route changed the window set; it must keep it"


def test_an_unknown_windows_frame_count_route_is_refused(tmp_path: Path) -> None:
    """A typo must not silently fall back to the expensive route, nor to a permission the run did not opt into."""
    inventory = tmp_path / "windows.json"
    inventory.write_text(json.dumps([]))
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1",
               f"bench.windows_inventory={inventory}", "bench.windows_frame_count=packet")
    with pytest.raises(ValueError, match="windows_frame_count"):
        bench.build_clips(cfg, _REPO)


def _split_inventory(tmp_path: Path) -> Path:
    """Two real videos, one CALIB and one TEST, as an inventory the runner can build from."""
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)
    for name in ("calib.mp4", "test.mp4"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=size=32x32:rate=25:duration=25", str(videos / name)], check=True)
    inventory = tmp_path / "splits.json"
    inventory.write_text(json.dumps([
        {"video_id": "ego_c", "path": str(videos / "calib.mp4"), "source": "ego4d", "split": "CALIB"},
        {"video_id": "ego_t", "path": str(videos / "test.mp4"), "source": "ego4d", "split": "TEST"},
    ]))
    return inventory


def test_a_truncated_run_refuses_to_spend_test_windows(tmp_path: Path) -> None:
    """Freeze-1: `bench.windows_max_per_video` must be null for the TEST run -- so a run that SETS it is by
    construction not the TEST run, and must not open TEST content.

    TEST is one-shot and runs automatically once CALIB is clean, so there is no round-trip in which someone
    notices that a pilot already spent it. The rule existed only as prose in the freeze note, and prose does
    not fail loudly: a bounded pilot over the real 21-video inventory selects 24 TEST windows against 18
    CALIB ones, and nothing would have said so.
    """
    inventory = _split_inventory(tmp_path)
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1",
               f"bench.windows_inventory={inventory}", "bench.windows_max_per_video=1")

    with pytest.raises(ValueError, match="TEST"):
        bench.build_clips(cfg, _REPO)


def test_a_truncated_run_over_calib_only_is_allowed(tmp_path: Path) -> None:
    """The guard must block spending TEST, not block bounded pilots -- which are how CALIB gets run at all."""
    inventory = _split_inventory(tmp_path)
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1",
               f"bench.windows_inventory={inventory}", "bench.windows_max_per_video=1",
               "bench.windows_splits=[CALIB]")

    windows = [c for c in bench.build_clips(cfg, _REPO) if c.window is not None]

    assert windows, "the CALIB-only pilot produced no windows"
    assert {c.split for c in windows} == {"CALIB"}


def test_an_untruncated_run_may_still_include_test(tmp_path: Path) -> None:
    """The freeze's own TEST run sets max_per_video to null; the guard must not stand in its way."""
    inventory = _split_inventory(tmp_path)
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1",
               f"bench.windows_inventory={inventory}")

    splits = {c.split for c in bench.build_clips(cfg, _REPO) if c.window is not None}

    assert "TEST" in splits and "CALIB" in splits


def test_hoi4d_windows_reach_the_registered_runner(tmp_path: Path) -> None:
    """`bench.hoi4d_windows` -> `build_clips` -> windowed HOI4D clips, and `prepare_artifacts` writes the GT
    object label of the window's first frame as decode row 0's prompt and the SHIPPED K as camera.json."""
    from fastdo.engine.workers_video import read_mask_prompt
    from fastdo.geometry.intrinsics import read_camera_json
    from fastdo.io.datasets.hoi4d import HOI4DReader
    from tests.io.hoi4d_tree import SEQ, K, write_tree

    root = write_tree(tmp_path / "hoi4d")
    reader = HOI4DReader(root)
    reader.video_path(SEQ).parent.mkdir(parents=True)
    reader.video_path(SEQ).write_bytes(b"")
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps({"windows": [{"sequence": SEQ, "start": 2, "stop": 10, "shift_px": [4.0, -1.0],
                                                "pool": "dev"}]}))
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=0", f"hoi4d.root={root}",
               f"bench.hoi4d_windows={windows}")
    (clip,) = bench.build_clips(cfg, _REPO)
    assert clip.clip_id == "bench_hoi4d_ZY20210800001_H1_C2_N01_S10_s01_T1_f002" and clip.window == (2, 10)
    prompts = bench.prepare_artifacts(cfg, _REPO, [clip], tmp_path / "engine", write_inputs=True)
    assert "frame 2" in prompts[clip.clip_id]
    row, mask = read_mask_prompt(tmp_path / "engine" / clip.clip_id)
    assert row == 0 and np.array_equal(mask, reader.object_mask(SEQ, 2)[0])
    cam = read_camera_json(tmp_path / "engine" / clip.clip_id / "camera.json")
    np.testing.assert_allclose(np.asarray(cam), K, atol=1e-4)      # the shipped K: shift_px corrects only GT poses


def test_windows_ids_keeps_only_the_named_windows_and_refuses_test(tmp_path: Path) -> None:
    """`bench.windows_ids` hand-picks windows; unknown ids and TEST ids are refused, null keeps every window."""
    inventory = _split_inventory(tmp_path)
    base = ("bench.doh_root=null", "bench.whisking_copies=1", f"bench.windows_inventory={inventory}")
    every = [c.clip_id for c in bench.build_clips(_cfg(tmp_path, *base), _REPO) if c.window is not None]
    got = [c.clip_id for c in bench.build_clips(_cfg(tmp_path, *base, "bench.windows_ids=[ego_c:k001]"), _REPO)
           if c.window is not None]
    assert got == ["ego_c:k001"] and len(every) > 1
    with pytest.raises(ValueError, match="not selected"):
        bench.build_clips(_cfg(tmp_path, *base, "bench.windows_ids=[ego_c:k099]"), _REPO)
    with pytest.raises(ValueError, match="TEST"):
        bench.build_clips(_cfg(tmp_path, *base, "bench.windows_ids=[ego_t:k000]"), _REPO)


def test_a_missing_baseline_submodule_names_the_command_that_fetches_it(tmp_path: Path) -> None:
    """Without the DO AS I DO submodule the whisking prompt has no layout to read: the error says how to fetch it
    instead of a bare FileNotFoundError on the layout file."""
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1")       # demo_root: an empty tmp path
    clips = bench.build_clips(cfg, _REPO)
    with pytest.raises(FileNotFoundError, match=r"git submodule update --init --recursive") as info:
        bench.prepare_artifacts(cfg, _REPO, clips, tmp_path / "engine", write_inputs=True)
    assert "layout_camera_frame_optimized.json" in str(info.value)


def _bench_manifest(out_dir: Path, failed: list[bool]) -> None:
    ids = [f"clip_{i}" for i in range(len(failed))]
    manifest = ClipManifest.create(ids, ["whisking"] * len(ids), [10] * len(ids))
    for clip_id, fails in zip(ids, failed, strict=True):
        manifest.mark(clip_id, "decode", "done", 1.0)
        manifest.mark(clip_id, "mask", "failed" if fails else "done", 0.5, "No CUDA GPUs are available" if fails else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(out_dir / bench.MANIFEST_FILE)


@pytest.mark.parametrize(("failed", "code"), [([True, True, True], 1), ([True, False, True], 0), ([False], 0)])
def test_the_benchmark_exits_1_only_when_every_clip_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           caplog: pytest.LogCaptureFixture, failed: list[bool],
                                                           code: int) -> None:
    """A run in which every clip failed a stage must not exit 0 with only a WARNING; a partial failure still does
    (its report and WARNING describe it)."""
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(bench, "apply_cpu_budget", lambda: None)       # keep this process's thread settings
    monkeypatch.setattr(bench, "apply_gpu_budget", lambda: None)
    monkeypatch.setattr(bench, "run_bench", lambda cfg, root: _bench_manifest(tmp_path / "bench", failed))
    if code:
        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as info:
            bench.main(cfg)
        assert info.value.code == 1
        assert "every clip failed (3 of 3)" in caplog.text and "mask: No CUDA GPUs are available" in caplog.text
    else:
        bench.main(cfg)
    assert bench.exit_status(ClipManifest.load(tmp_path / "bench" / bench.MANIFEST_FILE)) == code
