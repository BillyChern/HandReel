"""Tests for `fastdo.bench.bench_batched`: artifact comparison and the `bench.batched` Hydra block."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastdo.bench import gpu_budget
from fastdo.bench.bench_batched import (
    batched_engine_config,
    build_batched_stages,
    compare_artifact_dirs,
    main,
)
from fastdo.engine.batched_runner import (
    BatchedEngineConfig,
    MemoryLedgerConfig,
    StageBatchConfig,
    StageBudgetError,
    applied_cap_gb,
    check_stage_budgets,
    run_manifest_batched,
)
from fastdo.engine.batched_stages import FrameBatchedStage
from fastdo.engine.batching import dynamic_batches
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, StageFn
from tests.engine.fakes_batched import RowMeanWorker

_REPO = Path(__file__).resolve().parents[2]

# The promoted per-stage caps, each measured in one named row of
# outputs/bench/d14_rr_batched/batches.jsonl. `max_batch` is the batch size that row actually ran, `budget_gb` its
# `peak_gpu_mem_gb` (device GiB of the whole batch) rounded up to 2 dp, and `fixed_cost_gb` that same peak divided
# by the batch size, rounded down -- because `dynamic_batches` sums `fixed_cost_gb` per clip over a group.
MEASURED = {
    "mask":     (StageBatchConfig(max_batch=2, budget_gb=4.38, fixed_cost_gb=2.18), 4.376729088, "mask-00002"),
    "tracks":   (StageBatchConfig(max_batch=2, budget_gb=8.20, fixed_cost_gb=4.09), 8.193717760, "tracks-00000"),
    "hands":    (StageBatchConfig(max_batch=6, budget_gb=5.77, fixed_cost_gb=0.96), 5.764341354, "hands-00000"),
    "geometry": (StageBatchConfig(max_batch=4, budget_gb=7.27, fixed_cost_gb=1.81), 7.269983744, "geometry-00000"),
    "solve":    (StageBatchConfig(max_batch=6, budget_gb=13.39, fixed_cost_gb=2.23), 13.382537216, "solve-00000"),
}
CAP_GB = 21.0          # the per-process device cap every one of those rows was measured under


@pytest.fixture(autouse=True)
def _applied_cap_is_the_measured_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the cap this process applies to the 21 GiB the evidence was measured under.

    `check_stage_budgets` validates against the cap `apply_gpu_budget` actually installs
    (`FASTDO_GPU_MEM_GB`, else 12.0), not against the declared `bench.batched.cap_gb` alone, so the promoted config
    only loads under a 21 GiB process. Tests that exercise a different applied cap set the variable themselves.
    """
    monkeypatch.setenv(gpu_budget.ENV_VAR, str(int(CAP_GB)))


def _tree(root: Path) -> Path:
    rng = np.random.default_rng(0)
    for i in range(2):
        clip = root / f"clip{i}"
        (clip / "hands_job").mkdir(parents=True)
        np.savez(clip / "geometry.npz", value=(0.01 * rng.standard_normal((4, 3))).astype(np.float32),
                 frame_idx=np.arange(4, dtype=np.int64))
        np.savez(clip / "mask.npz", masks=rng.random((4, 24, 32)) > 0.5, frame_idx=np.arange(4, dtype=np.int64))
        np.savez(clip / "tracks.npz", tracks=np.full((4, 2, 2), np.nan, np.float32))
        np.savez(clip / "encoder.npz", tokens=np.ones((4, 8), np.float32), crop=np.array("center"))
        (clip / "solve.json").write_text(json.dumps({"scale": 1.0, "solver": {"seconds": 2.0}}))
        (clip / "profile.json").write_text(json.dumps({"total_seconds": 1.0}))
        (clip / "hands_job" / "request.json").write_text("{}")
    return root


def _copy(a: Path, tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(a, tmp_path / name))


def _edit(path: Path, key: str, change) -> None:  # type: ignore[no-untyped-def]
    with np.load(path) as data:
        arrays = {k: data[k].copy() for k in data.files}
    change(arrays[key])
    np.savez(path, **arrays)  # type: ignore[arg-type]


def test_compare_artifact_dirs_reports_differences(tmp_path: Path) -> None:
    a = _tree(tmp_path / "a")
    same = compare_artifact_dirs(a, _copy(a, tmp_path, "same"))
    assert same["n_mismatched"] == 0 and same["missing_in_a"] == [] == same["missing_in_b"] and same["n_files"] == 10

    b = _copy(a, tmp_path, "float")
    _edit(b / "clip0" / "geometry.npz", "value", lambda v: v.__setitem__((1, 2), v[1, 2] + np.float32(1e-3)))
    report = compare_artifact_dirs(a, b)
    assert report["arrays"]["clip0/geometry.npz"]["value"]["max_abs_diff"] == pytest.approx(1e-3, rel=1e-4)
    assert report["n_mismatched"] == 1

    b = _copy(a, tmp_path, "mask")
    _edit(b / "clip1" / "mask.npz", "masks", lambda m: m.__setitem__((2, 3, 4), not m[2, 3, 4]))
    report = compare_artifact_dirs(a, b)
    assert report["arrays"]["clip1/mask.npz"]["masks"]["mismatch_fraction"] == 1 / 3072 and report["n_mismatched"] == 1

    b = _copy(a, tmp_path, "crop")
    np.savez(b / "clip0" / "encoder.npz", tokens=np.ones((4, 8), np.float32), crop=np.array("letterbox"))
    assert compare_artifact_dirs(a, b)["arrays"]["clip0/encoder.npz"]["crop"]["equal"] is False

    b = _copy(a, tmp_path, "missing")
    (b / "clip0" / "tracks.npz").unlink()
    assert compare_artifact_dirs(a, b)["missing_in_b"] == ["clip0/tracks.npz"]

    b = _copy(a, tmp_path, "ignored")
    (b / "clip0" / "profile.json").write_text("{}")
    (b / "clip0" / "hands_job" / "request.json").write_text('{"x": 1}')
    (b / "clip1" / "solve.json").write_text(json.dumps({"scale": 1.0, "solver": {"seconds": 9.0}}))
    out = tmp_path / "report.json"
    assert main(["compare", str(a), str(b), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["n_mismatched"] == 0
    (b / "clip1" / "solve.json").write_text(json.dumps({"scale": 2.0}))
    assert compare_artifact_dirs(a, b)["n_mismatched"] == 1


# Production declares a ledger and compose MERGES a dict override into it; delete the node first so a
# test's `{resident_gb: R}` is the whole block.
_LEDGER_RESET = "~bench.batched.memory_ledger"


def _cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[f"paths.root={_REPO}", *overrides])


def test_batched_engine_config_from_the_hydra_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    block = cfg.bench.batched
    assert block.enabled is True and block.retry_singly is True      # promoted
    assert block.max_clips_per_wave == 6      # was 8; solve and hands are measured only to batch 6 (condition 2)
    assert block.cap_gb == 32                 # the evidence rows were measured under 21 (CAP_GB)

    batched = batched_engine_config(cfg.bench, EngineConfig(artifact_root=tmp_path))

    assert (batched.max_clips_per_wave, batched.retry_singly) == (6, True)
    assert block.prefetch_decode is True and batched.prefetch_decode is True         # promoted
    opted_out = _cfg("bench.batched.prefetch_decode=false").bench
    assert batched_engine_config(opted_out, EngineConfig(artifact_root=tmp_path)).prefetch_decode is False
    assert batched.batch_config("decode") == StageBatchConfig() and "decode" not in batched.stage_batches
    # encoder never ran in the rr family: it stays unbatched and labelled, never silently promoted (condition 3).
    assert batched.stage_batches["encoder"] == StageBatchConfig(max_batch=1, unmeasured=True)
    assert set(batched.stage_batches) == {"encoder", *MEASURED}

    for stage, (expected, peak_gb, _row) in MEASURED.items():
        sb = batched.stage_batches[stage]
        assert sb == expected, stage
        assert not sb.unmeasured, stage
        # the budget covers its row's measured peak and still fits the cap that peak was measured under
        assert peak_gb <= sb.resolved_budget_gb() <= CAP_GB, stage
        budget, size = sb.resolved_budget_gb(), sb.max_batch
        # the planner reproduces exactly the measured batch: `max_batch` clips are one group, and the budget alone
        # (max_batch raised out of the way) still refuses the next clip, so the two caps agree rather than compete.
        assert [len(g) for g in dynamic_batches(list(range(size)), size, budget,
                                                lambda _c, gb=sb.fixed_cost_gb: gb)] == [size], stage
        assert [len(g) for g in dynamic_batches(list(range(size + 1)), size + 1, budget,
                                                lambda _c, gb=sb.fixed_cost_gb: gb)] == [size, 1], stage

    monkeypatch.setenv("FASTDO_GPU_MEM_GB", "21")
    assert batched.stage_batches["encoder"].resolved_budget_gb() == 21.0   # no budget of its own: the process cap
    monkeypatch.delenv("FASTDO_GPU_MEM_GB")
    assert batched.stage_batches["encoder"].resolved_budget_gb() == 12.0
    # encoder no longer carries a budget_gb of its own, so the NaN case rides on a stage that declares one
    for override in ("bench.batched.stages.mask.max_batch=0", "bench.batched.stages.mask.budget_gb=nan"):
        with pytest.raises(ValueError):
            batched_engine_config(_cfg(override).bench, EngineConfig(artifact_root=tmp_path))


def test_budgets_over_the_cap_and_unmeasured_promotion_are_refused(tmp_path: Path) -> None:
    engine = EngineConfig(artifact_root=tmp_path)
    # solve is the stage nearest the cap (13.38 GB measured at batch 6): a budget past 21 GiB is refused by name.
    with pytest.raises(StageBudgetError, match=r"'solve' declares budget_gb 22.0.*21"):
        batched_engine_config(_cfg("bench.batched.stages.solve.budget_gb=22.0").bench, engine)
    # and raising max_batch alone commits to more than the cap even with a legal budget
    with pytest.raises(StageBudgetError, match=r"'solve' plans max_batch 12 .*26.76 GiB, above the 21"):
        batched_engine_config(_cfg("bench.batched.stages.solve.max_batch=12",
                                   "bench.batched.stages.solve.budget_gb=21.0").bench, engine)
    # a stage no measurement covers cannot be enabled by raising its cap ...
    with pytest.raises(ValueError, match="unmeasured stage may not declare max_batch 4"):
        batched_engine_config(_cfg("bench.batched.stages.encoder.max_batch=4").bench, engine)
    # ... and dropping the label is necessary but no longer sufficient: a batched stage that declares neither
    # budget_gb nor fixed_cost_gb is refused too, so the promoting edit has to carry the measurement it claims
    with pytest.raises(StageBudgetError, match=r"'encoder' declares max_batch 4 with neither budget_gb nor"):
        batched_engine_config(_cfg("bench.batched.stages.encoder.max_batch=4",
                                   "bench.batched.stages.encoder.unmeasured=false").bench, engine)
    admitted = batched_engine_config(_cfg("bench.batched.stages.encoder.max_batch=4",
                                          "bench.batched.stages.encoder.unmeasured=false",
                                          "+bench.batched.stages.encoder.budget_gb=8.0",
                                          "+bench.batched.stages.encoder.fixed_cost_gb=2.0").bench, engine)
    assert admitted.stage_batches["encoder"] == StageBatchConfig(max_batch=4, budget_gb=8.0, fixed_cost_gb=2.0)


def test_unknown_stage_keys_raise_and_stagefn_fallbacks_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    misspelt = OmegaConf.create({"batched": {"stages": {"mask": {"max_batch": 2}, "trakcs": {"max_batch": 2}}}})
    with pytest.raises(ValueError, match=r"unknown stage keys \['trakcs'\]"):
        batched_engine_config(misspelt, EngineConfig(artifact_root=tmp_path))

    def plain(clip_id: str, artifact_dir: Path) -> None:
        return None

    with caplog.at_level(logging.WARNING, logger="fastdo.bench.bench_batched"):
        out = build_batched_stages(_cfg(), {"encoder": plain, "geometry": RowMeanWorker("geometry")},
                                   tmp_path / "jobs", tmp_path / ".gpu.lock")
    assert out["encoder"] is plain and isinstance(out["geometry"], FrameBatchedStage)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "encoder" in warnings[0] and "plain StageFn" in warnings[0]


def test_the_guard_validates_against_the_cap_the_process_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 caplog: pytest.LogCaptureFixture) -> None:
    """The cap in force is `apply_gpu_budget`'s, not the declared `bench.batched.cap_gb`.

    An earlier solver OOM happened under `FASTDO_GPU_MEM_GB=20`; certifying a plan against a cap the process does not
    apply is how that happens again.
    """
    engine = EngineConfig(artifact_root=tmp_path)
    assert applied_cap_gb() == CAP_GB                              # the autouse fixture's 21, as the runtime reads it
    assert batched_engine_config(_cfg().bench, engine).stage_batches["solve"].budget_gb == 13.39

    # solve's measured 13.39 fits the declared 21 and does NOT fit a 12 GiB process: refused, naming both caps
    monkeypatch.setenv(gpu_budget.ENV_VAR, "12")
    with pytest.raises(StageBudgetError,
                       match=r"'solve' declares budget_gb 13.39, above the 12.0 GiB.*cap_gb 32.0.*FASTDO_GPU_MEM_GB"):
        batched_engine_config(_cfg().bench, engine)

    # unset is not "uncapped": apply_gpu_budget installs gpu_budget.DEFAULT_GB, and the guard resolves the same way
    monkeypatch.delenv(gpu_budget.ENV_VAR)
    assert applied_cap_gb() == gpu_budget.DEFAULT_GB == 12.0
    with pytest.raises(StageBudgetError, match=r"'solve' declares budget_gb 13.39, above the 12.0 GiB"):
        batched_engine_config(_cfg().bench, engine)

    # a non-positive value opts the process cap out (apply_gpu_budget's own rule): the declared cap is all there is
    monkeypatch.setenv(gpu_budget.ENV_VAR, "0")
    assert applied_cap_gb() is None
    # (the production ledger needs a cap, so this no-cap check runs with the ledger off)
    assert batched_engine_config(_cfg("bench.batched.memory_ledger=null").bench, engine).stage_batches["solve"] \
        .budget_gb == 13.39

    # a process cap above the declared one leaves the declared one binding, and the disagreement is stated, not hidden
    monkeypatch.setenv(gpu_budget.ENV_VAR, "40")
    with caplog.at_level(logging.WARNING, logger="fastdo.engine.batched_runner"):
        batched_engine_config(_cfg().bench, engine)
    assert any("32.0" in m and "40.0" in m for m in (r.getMessage() for r in caplog.records))   # declared 32
    with pytest.raises(StageBudgetError, match=r"'solve' declares budget_gb 33.0, above the 32.0 GiB"):
        batched_engine_config(_cfg("bench.batched.stages.solve.budget_gb=33.0").bench, engine)


def test_a_batched_stage_that_declares_no_memory_at_all_is_refused(tmp_path: Path) -> None:
    """`planned_peak_gb()` is `max_batch * fixed_cost_gb` = 0.0 without a fixed cost and
    `budget_gb` was checked only when set, so a block declaring NEITHER passed at any batch size -- precisely
    an earlier `solve: {max_batch: 8}` placeholder, and the shape a batched `shape` stage would arrive in.
    """
    engine = EngineConfig(artifact_root=tmp_path)
    with pytest.raises(StageBudgetError,
                       match=r"'solve' declares max_batch 8 with neither budget_gb nor fixed_cost_gb"):
        check_stage_budgets({"solve": StageBatchConfig(max_batch=8)}, CAP_GB)
    with pytest.raises(StageBudgetError, match=r"'shape' declares max_batch 8 with neither"):
        batched_engine_config(_cfg("+bench.batched.stages.shape={max_batch: 8}").bench, engine)
    # a per-frame term alone does not declare the frame-independent floor `planned_peak_gb` plans against
    with pytest.raises(StageBudgetError, match="neither budget_gb nor fixed_cost_gb"):
        check_stage_budgets({"shape": StageBatchConfig(max_batch=2, cost_per_frame_gb=0.1)}, CAP_GB)
    # either field alone is a declaration the cap can bind, and both are legal
    check_stage_budgets({"shape": StageBatchConfig(max_batch=8, fixed_cost_gb=1.0)}, CAP_GB)
    check_stage_budgets({"shape": StageBatchConfig(max_batch=8, budget_gb=8.0)}, CAP_GB)


def test_declared_unmeasured_at_batch_one_is_not_what_that_refusal_catches(tmp_path: Path) -> None:
    """`encoder` declares no memory either, and must keep passing. The distinction is
    `max_batch`: a stage held at 1 runs one clip per group, so there is no group for a budget to bound."""
    check_stage_budgets({"encoder": StageBatchConfig(max_batch=1, unmeasured=True)}, CAP_GB)
    check_stage_budgets({"shape": StageBatchConfig(max_batch=1)}, CAP_GB)       # unlabelled, but still not batched
    batched = batched_engine_config(_cfg().bench, EngineConfig(artifact_root=tmp_path))
    assert batched.stage_batches["encoder"] == StageBatchConfig(max_batch=1, unmeasured=True)


def test_pipeline_knobs_reach_the_runner(tmp_path: Path) -> None:
    """`pipeline_waves`, `pipeline_depth` and `pipeline_resources` flow from `bench.batched` into the runner,
    which then runs each stage on its resource class's thread; malformed values are refused at config load."""
    engine = EngineConfig(artifact_root=tmp_path, stages=("decode", "shape", "hands", "qc"))
    default = batched_engine_config(_cfg().bench, engine)
    assert (default.pipeline_waves, default.pipeline_depth) == (True, 4)        # depth 4
    # the measured worker classes; their main-process torch.cuda calls allocate nothing (run/conf/config.yaml)
    assert dict(default.pipeline_resources) == {"shape": "sam3d_worker", "hands": "hawor_worker", "geometry": "geom",
                                                "prompt": "prompt", "mask": "mask"}
    off = batched_engine_config(_cfg("bench.batched.pipeline_waves=false").bench, engine)
    assert off.pipeline_waves is False                                          # the knob still turns it off
    cfg = _cfg("bench.batched.pipeline_waves=true", "bench.batched.pipeline_depth=3",
               "bench.batched.pipeline_resources.shape=sam3d_worker",
               "bench.batched.pipeline_resources.hands=hawor_worker")
    batched = batched_engine_config(cfg.bench, engine)
    assert (batched.pipeline_waves, batched.pipeline_depth) == (True, 3)
    assert dict(batched.pipeline_resources) == {"shape": "sam3d_worker", "hands": "hawor_worker", "geometry": "geom",
                                                "prompt": "prompt", "mask": "mask"}   # overrides merge onto the default

    threads: dict[str, set[str]] = {}

    def fake(stage: str) -> StageFn:
        def fn(clip_id: str, artifact_dir: Path) -> None:
            threads.setdefault(stage, set()).add(threading.current_thread().name)
        return fn

    ids = ["a", "b", "c"]
    manifest = ClipManifest.create(ids, [f"{c}.mp4" for c in ids], [3, 3, 3])
    report = run_manifest_batched(manifest, replace(batched, max_clips_per_wave=1, run_id="p0"),
                                  {stage: fake(stage) for stage in engine.stages})
    assert threads == {"decode": {"fastdo-pipeline-decode"}, "shape": {"fastdo-pipeline-sam3d_worker"},
                       "hands": {"fastdo-pipeline-hawor_worker"}, "qc": {"fastdo-pipeline-qc"}}
    assert report.n_batches == 12 and all(manifest.status(c, "qc").state == "done" for c in ids)

    with pytest.raises(ValueError, match="unknown stage keys"):
        batched_engine_config(_cfg("+bench.batched.pipeline_resources.bogus=x").bench, engine)
    with pytest.raises(ValueError, match="runs in-process"):       # only SAM 3D runs in a worker subprocess
        batched_engine_config(_cfg("bench.batched.pipeline_resources.shape=sam3d_worker",
                                   "bench.shape.backend=trellis").bench, engine)
    with pytest.raises(ValueError, match="pipeline_depth"):
        batched_engine_config(_cfg("bench.batched.pipeline_depth=0").bench, engine)


def test_memory_ledger_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                              caplog: pytest.LogCaptureFixture) -> None:
    """`bench.batched.memory_ledger` is off by default; `{resident_gb: R}` reaches the runner and its capacity is
    the applied cap minus R; a bad R, a stray key, a disabled cap or a `*_worker` class on an in-process stage is
    refused at config load."""
    engine = EngineConfig(artifact_root=tmp_path)
    # production declares the measured-cost ledger; the engine default stays off
    prod = batched_engine_config(_cfg().bench, engine).memory_ledger
    assert prod is not None and prod.resident_gb == 7.0 and dict(prod.costs_gb) == {
        "solve": 9.0, "tracks": 9.0, "mask": 3.9, "geometry": 2.3, "intrinsics": 2.0, "prompt": 0.5}
    assert batched_engine_config(_cfg("bench.batched.memory_ledger=null").bench, engine).memory_ledger is None
    assert BatchedEngineConfig(engine).memory_ledger is None

    on = batched_engine_config(_cfg(_LEDGER_RESET, "+bench.batched.memory_ledger={resident_gb:3.5}").bench, engine)
    assert on.memory_ledger == MemoryLedgerConfig(resident_gb=3.5)
    assert on.memory_ledger.capacity_gb() == CAP_GB - 3.5                     # the autouse fixture applies 21 GiB
    monkeypatch.setenv(gpu_budget.ENV_VAR, "32")
    assert on.memory_ledger.capacity_gb() == 28.5                              # read from the cap applied at run time
    assert batched_engine_config(_cfg(_LEDGER_RESET, "+bench.batched.memory_ledger={resident_gb:0}").bench, engine) \
        .memory_ledger == MemoryLedgerConfig(resident_gb=0.0)

    for bad in ("{resident_gb:32}", "{resident_gb:40.0}", "{resident_gb:-1}", "{resident_gb:nan}",
                "{resident_gb:null}", "{resident_gb:'3'}", "{resident_gb:3,capacity_gb:20}", "{}", "4"):
        with pytest.raises(ValueError, match="memory_ledger"):
            batched_engine_config(_cfg(_LEDGER_RESET, f"+bench.batched.memory_ledger={bad}").bench, engine)
    costs = batched_engine_config(_cfg(_LEDGER_RESET, "+bench.batched.memory_ledger={resident_gb:7,costs_gb:{tracks:21.5,solve:22}}")
                                  .bench, engine).memory_ledger
    assert costs is not None and (costs.resident_gb, dict(costs.costs_gb)) == (7, {"tracks": 21.5, "solve": 22.0})
    for bad in ("{resident_gb:1,costs_gb:{bogus:1}}", "{resident_gb:1,costs_gb:{tracks:-1}}",
                "{resident_gb:1,costs_gb:{tracks:nan}}", "{resident_gb:1,costs_gb:4}", "{costs_gb:{tracks:1}}"):
        with pytest.raises(ValueError, match="memory_ledger"):
            batched_engine_config(_cfg(_LEDGER_RESET, f"+bench.batched.memory_ledger={bad}").bench, engine)
    monkeypatch.setenv(gpu_budget.ENV_VAR, "0")                    # the cap's opt-out: nothing to subtract from
    with pytest.raises(ValueError, match="memory_ledger needs the per-process device cap"):
        batched_engine_config(_cfg(_LEDGER_RESET, "+bench.batched.memory_ledger={resident_gb:1}").bench, engine)
    monkeypatch.setenv(gpu_budget.ENV_VAR, "21")
    with pytest.raises(ValueError, match=r"in-process stage\(s\) \['geometry'\]"):
        batched_engine_config(_cfg(_LEDGER_RESET, "+bench.batched.memory_ledger={resident_gb:1}",
                                   "bench.batched.pipeline_resources.geometry=geom_worker").bench, engine)
    with pytest.raises(ValueError, match="MemoryLedgerConfig"):
        BatchedEngineConfig(engine, memory_ledger={"resident_gb": 1.0})  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="fastdo.bench.bench_batched"):
        off = batched_engine_config(_cfg(_LEDGER_RESET, "+bench.batched.memory_ledger={resident_gb:1}",
                                         "bench.batched.pipeline_waves=false").bench, engine)
    assert off.memory_ledger == MemoryLedgerConfig(resident_gb=1.0)
    assert any("ignores the ledger" in r.getMessage() for r in caplog.records)
