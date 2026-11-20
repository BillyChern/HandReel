"""driftqc default identity.

With the driftqc stage disabled, every default engine artifact must be unchanged. The proof here is differential, so it
survives main moving: run A is the default pipeline (valid_range hooks spied, never stubbed); run B is the same pipeline
with the valid_range hooks stubbed to the pre-change path and `fastdo.engine.workers_driftqc` blocked. A later change to
a default artifact changes A and B alike. The patches use `raising=False`, so they are no-ops where a hook name does not
exist.

The module also holds the opt-in config and bench wiring tests, and the differential test asserts that run A really
called `resolve_valid_range` from the solve and qc hooks (with `driftqc_enabled` False), so a bypass that never calls a
hook turns it red.

A second pair with the real CPU solve (`SMALL_SOLVE`) is omitted: two runs of A with the real solve do not agree under
the artifact digest, because `solve.npz` stores the solver's wall-clock `seconds`.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastdo.bench import bench_batched, bench_stages
from fastdo.engine import source_profile, workers_solve
from fastdo.engine.batched_stages import SolveBatchStage
from fastdo.engine.manifest import STAGE_NAMES
from fastdo.engine.orchestrator import STAGE_GPU, StageFn
from fastdo.engine.source_domain import SourceDomain, write_domain_record
from fastdo.engine.worker_support import load_npz, read_frame_idx
from fastdo.engine.workers_driftqc import DriftQcStageConfig, make_driftqc_worker, with_driftqc
from fastdo.engine.workers_solve import SolveStageConfig, read_solve_inputs
from fastdo.io.datasets.dexycb_real import assemble_real_observations
from fastdo.qc import AcceptanceRule, ConformalCalibration
from fastdo.qc.rule_io import save_acceptance_rule
from run.pipeline import bench_throughput
from tests.engine.driftqc_digest import ENGINE_DIR, run_default_pipeline
from tests.engine.fakes_driftqc import driftqc_clip, write_driftqc_solve_inputs
from tests.engine.fakes_solve import SMALL_SOLVE, fake_solution

_VALID_RANGE = "fastdo.engine.valid_range"
_HOOK_MODULES = (_VALID_RANGE, "fastdo.engine.workers_solve", "fastdo.engine.workers_qc",
                 "fastdo.engine.workers_retarget")
_DRIFTQC_ARTIFACTS = ("driftqc.json", "valid_range.json")
_REPO = Path(__file__).resolve().parents[2]
FROZEN_ROOT = Path(os.environ.get("FASTDO_DRIFTQC_ROOT", "/opt/fastdo/handreel"))
FROZEN_ABSENT = ("frozen driftqc files absent (not shipped): set FASTDO_DRIFTQC_ROOT to a directory holding "
                 "temp/driftqc/constants_v2_2.json, temp/trackweight/constants_v2.json and "
                 "temp/trackweight/code/v1_weights.py")
_FROZEN_REL = {"constants_path": "temp/driftqc/constants_v2_2.json",
               "trackweight_constants_path": "temp/trackweight/constants_v2.json",
               "detector_path": "temp/trackweight/code/v1_weights.py"}
requires_frozen = pytest.mark.skipif(not all((FROZEN_ROOT / rel).is_file() for rel in _FROZEN_REL.values()),
                                     reason=FROZEN_ABSENT)


def _import_optional(name: str) -> ModuleType | None:
    """The module, or None when that module itself does not exist yet (a missing dependency inside it still raises)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None


def _patch_hook(mp: pytest.MonkeyPatch, name: str, make: Callable[[Any], Any]) -> None:
    """Replaces `name` in valid_range.py and in every hook module that bound it (`from ... import`), with
    `make(original)`. A no-op while valid_range.py does not exist (the E0 base)."""
    source = _import_optional(_VALID_RANGE)
    if source is None or not hasattr(source, name):
        return
    original = getattr(source, name)
    replacement = make(original)
    for module_name in _HOOK_MODULES:
        module = _import_optional(module_name)
        if module is not None and getattr(module, name, None) is original:
            mp.setattr(module, name, replacement, raising=False)


def _install_spies(mp: pytest.MonkeyPatch, enabled_seen: list[object], slice_calls: list[str],
                   resolve_callers: list[str] | None = None) -> None:
    def spy_resolve(original: Any) -> Any:
        signature = inspect.signature(original)

        def spy(*args: Any, **kwargs: Any) -> Any:
            enabled_seen.append(signature.bind(*args, **kwargs).arguments["driftqc_enabled"])
            caller = inspect.currentframe()
            if resolve_callers is not None and caller is not None and caller.f_back is not None:
                resolve_callers.append(str(caller.f_back.f_globals.get("__name__")))
            return original(*args, **kwargs)

        return spy

    def spy_slice(original: Any) -> Any:
        def spy(*args: Any, **kwargs: Any) -> Any:
            slice_calls.append(repr(args[1:2]))
            return original(*args, **kwargs)

        return spy

    _patch_hook(mp, "resolve_valid_range", spy_resolve)
    _patch_hook(mp, "slice_artifact", spy_slice)


def _install_stubs(mp: pytest.MonkeyPatch) -> None:
    def refuse(name: str) -> Callable[[Any], Any]:
        def make(original: Any) -> Any:
            def stub(*args: Any, **kwargs: Any) -> Any:
                raise AssertionError(f"{name} called in the hooks-stubbed arm")

            return stub

        return make

    _patch_hook(mp, "resolve_valid_range", lambda original: lambda *a, **k: None)
    _patch_hook(mp, "read_valid_range", refuse("read_valid_range"))
    _patch_hook(mp, "slice_artifact", refuse("slice_artifact"))
    mp.setitem(sys.modules, "fastdo.engine.workers_driftqc", None)  # type: ignore[arg-type]


def _driftqc_files(root: Path) -> list[Path]:
    return sorted(p for name in _DRIFTQC_ARTIFACTS for p in root.rglob(name))


def test_digest_helper_is_deterministic(tmp_path: Path) -> None:
    first = run_default_pipeline(tmp_path / "a1")
    second = run_default_pipeline(tmp_path / "a2")
    assert first == second


def test_default_run_equals_hooks_stubbed(tmp_path: Path) -> None:
    enabled_seen: list[object] = []
    slice_calls: list[str] = []
    resolve_callers: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        _install_spies(mp, enabled_seen, slice_calls, resolve_callers)
        digests_a = run_default_pipeline(tmp_path / "a")
    with pytest.MonkeyPatch.context() as mp:
        _install_stubs(mp)
        digests_b = run_default_pipeline(tmp_path / "b")

    for clip_id in ("clip0", "clip1", "clip2"):  # the comparison is not vacuous: solve and qc ran in both arms
        for artifact in ("solve.npz", "solve.json", "qc.json", "clip_record.npz"):
            assert f"{clip_id}/{artifact}" in digests_a, (clip_id, artifact)
    # the real hooks ran in arm A (a bypass that never calls resolve_valid_range fails here), once per
    # clip in each of the solve and qc hooks, always with driftqc disabled.
    assert len(enabled_seen) >= 1
    for hook in ("fastdo.engine.workers_solve", "fastdo.engine.workers_qc"):
        assert resolve_callers.count(hook) >= 3, (hook, resolve_callers)
    assert all(enabled is False for enabled in enabled_seen), enabled_seen
    assert slice_calls == []
    assert _driftqc_files(tmp_path / "a" / ENGINE_DIR) == []
    assert digests_a == digests_b


# --- opt-in config and bench wiring ------------------------------------


def _compose(*overrides: str) -> DictConfig:
    from hydra.core.global_hydra import GlobalHydra

    # An earlier test module (e.g. an import of sam2) can leave GlobalHydra initialised; start from a clear one.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[f"paths.root={_REPO}", *overrides])


def _enabled(base: Path | None = FROZEN_ROOT) -> list[str]:
    """Overrides enabling driftqc: frozen-file paths under `base`, or repo-relative ones when `base` is None.

    Dense pointmaps too: production stores them masked, and driftqc samples off-mask pixels."""
    return ["driftqc_stage.enabled=true", "geometry_stage.pointmap_storage=dense",
            *(f"driftqc_stage.{key}={rel if base is None else base / rel}" for key, rel in _FROZEN_REL.items())]


def _saved_rule(path: Path) -> Path:
    save_acceptance_rule(path, AcceptanceRule(
        calibration=ConformalCalibration(threshold=0.5, alpha=0.1, n_calibration=20),
        logistic_weights=np.zeros(6), feature_mean=np.zeros(5), feature_std=np.ones(5)))
    return path


def test_default_config_driftqc_disabled() -> None:
    raw = OmegaConf.load(_REPO / "run/conf/config.yaml")
    assert isinstance(raw, DictConfig)
    assert DriftQcStageConfig.from_mapping(raw.driftqc_stage).enabled is False
    assert DriftQcStageConfig.from_mapping(raw.driftqc_stage) == DriftQcStageConfig()
    assert set(raw.driftqc_stage) == {"enabled", *_FROZEN_REL}
    assert "driftqc" not in raw.bench.stages
    cfg = _compose()
    fns, labels = bench_stages.build_stage_fns(cfg, _REPO, [])
    assert "driftqc" not in fns and "driftqc" not in labels
    assert tuple(labels) == STAGE_NAMES
    assert "driftqc" not in bench_throughput.pipeline_stages(cfg)


def test_stage_names_unchanged() -> None:
    """driftqc stays out of the DEFAULT pipeline, which is what "identity" means here.

    This used to also assert `set(STAGE_GPU) == set(STAGE_NAMES)`, i.e. that no opt-in stage may declare
    whether it uses the GPU. That invariant is why `prompt` had no entry and ran detectron2 and SAM 2 on the
    shared card outside the GPU lock: `STAGE_GPU.get(stage, False)` answered False for it. STAGE_GPU now
    covers every stage the engine CAN run, a superset of the default ones; being declared CPU-only in it does
    not put driftqc into any pipeline, which the assertions below still pin.
    """
    assert "driftqc" not in STAGE_NAMES
    assert STAGE_GPU["driftqc"] is False             # declared, and declared CPU-only
    assert set(STAGE_GPU) >= set(STAGE_NAMES)


def test_no_driftqc_files_when_disabled(tmp_path: Path) -> None:
    digests = run_default_pipeline(tmp_path)
    for clip_id in ("clip0", "clip1", "clip2"):
        assert f"{clip_id}/solve.json" in digests
        solve_json = json.loads((tmp_path / ENGINE_DIR / clip_id / "solve.json").read_text())
        assert "valid_range" not in solve_json
    assert _driftqc_files(tmp_path) == []


def test_valid_range_absent_takes_identity_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_dir = write_driftqc_solve_inputs(tmp_path / "clip0", driftqc_clip(0), None)
    calls: list[object] = []
    original = workers_solve.slice_artifact

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[1:2])
        return original(*args, **kwargs)

    monkeypatch.setattr(workers_solve, "slice_artifact", spy)
    inputs = read_solve_inputs(clip_dir, 64, 0)

    assert calls == [] and inputs.valid_range is None
    frame_idx = read_frame_idx(clip_dir)
    assert inputs.frame_idx.dtype == frame_idx.dtype and inputs.frame_idx.tobytes() == frame_idx.tobytes()
    mask, geometry, tracks = (load_npz(clip_dir / name) for name in ("mask.npz", "geometry.npz", "tracks.npz"))
    expected = assemble_real_observations(
        inputs.K, torch.from_numpy(mask["masks"].astype(bool)), torch.from_numpy(geometry["pointmaps"].astype(np.float32)),
        torch.from_numpy(tracks["tracks"].astype(np.float32)), torch.from_numpy(tracks["vis"].astype(bool)),
        torch.from_numpy(tracks["query_uv"].astype(np.float32)), int(mask["keyframe_index"]))
    pairs = [(f"observations.{f.name}", getattr(inputs.observations.observations, f.name),
              getattr(expected.observations, f.name)) for f in dataclasses.fields(expected.observations)]
    pairs += [("track_vis", inputs.observations.track_vis, expected.track_vis),
              ("keyframe_points", inputs.observations.keyframe_points, expected.keyframe_points)]
    for name, got, want in pairs:
        if isinstance(want, torch.Tensor):
            assert isinstance(got, torch.Tensor) and got.dtype == want.dtype and got.shape == want.shape, name
            assert got.numpy().tobytes() == want.numpy().tobytes(), name
        else:
            assert got == want, name


def test_bench_stages_refuse_driftqc_when_disabled(tmp_path: Path) -> None:
    cfg = _compose(f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}",
                   "bench.stages=[decode,tracks,driftqc,solve]")
    with pytest.raises(ValueError, match=r"driftqc_stage\.enabled"):
        bench_throughput.pipeline_stages(cfg)
    with pytest.raises(ValueError, match=r"driftqc_stage\.enabled"):
        bench_throughput.run_bench(cfg, _REPO, stage_fns={})
    assert not (tmp_path / "bench" / "manifest.parquet").exists()


def test_bench_stages_order_when_enabled(tmp_path: Path) -> None:
    order = bench_throughput.pipeline_stages(_compose(*_enabled(), "bench.stages=[qc,solve,driftqc,tracks,decode]"))
    assert order == ("decode", "tracks", "driftqc", "solve", "qc")
    assert bench_throughput.pipeline_stages(_compose(*_enabled(), "bench.stages=null")) == with_driftqc(STAGE_NAMES)
    assert "driftqc" not in bench_throughput.pipeline_stages(_compose(*_enabled()))   # the default list omits it
    with pytest.raises(ValueError, match="bogus"):
        bench_throughput.pipeline_stages(_compose(*_enabled(), "bench.stages=[decode,bogus]"))

    ran: list[tuple[str, str]] = []

    def fake(stage: str) -> StageFn:
        def worker(clip_id: str, artifact_dir: Path) -> None:
            ran.append((clip_id, stage))

        return worker

    cfg = _compose(*_enabled(), f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}",
                   f"whisking.demo_root={tmp_path / 'demo'}", "bench.doh_root=null", "bench.whisking_copies=1",
                   # this asserts single-stream stage ORDER, so it pins the path rather than inheriting the
                   # promoted `bench.batched.enabled: true` default
                   "bench.batched.enabled=false", "bench.stages=[decode,tracks,driftqc,solve]")
    bench_throughput.run_bench(cfg, _REPO, stage_fns={s: fake(s) for s in ("decode", "tracks", "driftqc", "solve")})
    assert [stage for _, stage in ran] == ["decode", "tracks", "driftqc", "solve"]
    report = json.loads((tmp_path / "bench" / "yield_report.json").read_text())
    assert report["stages_run"] == ["decode", "tracks", "driftqc", "solve"]
    assert report["stage_status"]["driftqc"] == "injected"


@requires_frozen
def test_build_stage_fns_registers_driftqc_only_when_enabled() -> None:
    fns, labels = bench_stages.build_stage_fns(_compose(*_enabled()), _REPO, [])
    assert callable(fns["driftqc"])
    assert labels["driftqc"].startswith("computed (driftqc v1 TRUNC")
    assert tuple(labels) == with_driftqc(STAGE_NAMES)
    relative, _ = bench_stages.build_stage_fns(_compose(*_enabled(None)), FROZEN_ROOT, [])   # resolved against root
    assert callable(relative["driftqc"])
    default, default_labels = bench_stages.build_stage_fns(_compose(), _REPO, [])
    assert "driftqc" not in default and "driftqc" not in default_labels


@pytest.mark.parametrize("enabled", [False, True], ids=["default", "enabled"])
def test_build_stage_fns_passes_driftqc_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                enabled: bool) -> None:
    seen: dict[str, object] = {}

    def recorder(stage: str) -> Callable[..., StageFn]:
        def make(*args: Any, **kwargs: Any) -> StageFn:
            seen[stage] = kwargs.get("driftqc_enabled", "<not passed>")
            return lambda clip_id, artifact_dir: None

        return make

    monkeypatch.setattr(bench_stages, "make_profiled_solve_worker", recorder("solve"))   # the profiled worker
    monkeypatch.setattr(bench_stages, "make_qc_worker_from_config", recorder("qc"))
    monkeypatch.setattr(bench_stages, "make_driftqc_worker", lambda cfg: (lambda clip_id, artifact_dir: None))
    overrides = [f"qc_stage.rule_path={_saved_rule(tmp_path / 'rule.json')}", *(_enabled() if enabled else [])]
    cfg = _compose(*overrides)

    fns, _ = bench_stages.build_stage_fns(cfg, _REPO, [])

    assert seen == {"solve": enabled, "qc": enabled}
    assert "retarget" not in fns                         # bench never registers retarget (NOT_REGISTERED)
    assert ("driftqc" in fns) is enabled
    batched = bench_batched.build_batched_stages(cfg, fns, tmp_path / "jobs", tmp_path / "gpu.lock")
    stage = batched["solve"]
    assert isinstance(stage.prepare, functools.partial) and isinstance(stage.finish, functools.partial)
    # the profiled halves around prepare_solve / finish_solve (fastdo.engine.source_profile)
    assert stage.prepare.func is source_profile._profiled_prepare
    assert stage.finish.func is source_profile._profiled_finish
    solve_cfg = SolveStageConfig.from_mapping(cfg.solve_stage)
    resolver = stage.prepare.keywords["resolver"]
    assert resolver.solve == solve_cfg and stage.finish.keywords == {"resolver": resolver}
    assert stage.prepare.keywords["driftqc_enabled"] is enabled   # finish reads call.context["valid_range"]


@requires_frozen
def test_bench_built_workers_refuse_stale_driftqc_artifacts_when_disabled(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the config wiring: the default config's solve (single and batched) and qc workers raise on a leftover
    TRUNCATE record and never slice.

    A hook bypass must fail on this test's own assertions, independent of CUDA and of missing inputs. So solve
    runs on `solve_stage.device=cpu`, any `slice_artifact` call raises an AssertionError at once, and the qc leg runs
    on its own dir that holds a full-row default-path `solve.npz` (a bypassed qc hook then fails the regex, not with
    FileNotFoundError)."""
    clip_dir = write_driftqc_solve_inputs(tmp_path / "engine" / "clip0", driftqc_clip(0), 9)
    qc_dir = write_driftqc_solve_inputs(tmp_path / "qc" / "clip0", driftqc_clip(0), 9)
    call = workers_solve.prepare_solve("clip0", qc_dir, SMALL_SOLVE)
    workers_solve.finish_solve("clip0", qc_dir, call, fake_solution(call.init_poses, call.init_scale), SMALL_SOLVE)
    make_driftqc_worker(DriftQcStageConfig(enabled=True, **{k: str(FROZEN_ROOT / v) for k, v in _FROZEN_REL.items()}))(
        "clip0", clip_dir)
    assert json.loads((clip_dir / "valid_range.json").read_text())["range"] == [0, 7]
    for name in _DRIFTQC_ARTIFACTS:  # same inputs and decode rows, so the record is valid for qc_dir too
        shutil.copyfile(clip_dir / name, qc_dir / name)
    slice_calls: list[str] = []
    _install_spies(monkeypatch, [], slice_calls)

    def refuse_slice(original: Any) -> Any:
        def spy(*args: Any, **kwargs: Any) -> Any:
            original(*args, **kwargs)  # records the call in slice_calls
            raise AssertionError(f"slice_artifact called with driftqc disabled: {slice_calls}")

        return spy

    _patch_hook(monkeypatch, "slice_artifact", refuse_slice)
    cfg = _compose(f"qc_stage.rule_path={_saved_rule(tmp_path / 'rule.json')}", "solve_stage.device=cpu")
    assert cfg.solve_stage.device == "cpu"
    fns, _ = bench_stages.build_stage_fns(cfg, _REPO, [])
    write_domain_record(clip_dir, "clip0", "dexycb", SourceDomain.THIRD_PERSON)   # a bench clip is declared
    batched = bench_batched.build_batched_stages(cfg, fns, tmp_path / "jobs", tmp_path / "gpu.lock")
    solve_stage = batched["solve"]
    assert isinstance(solve_stage, SolveBatchStage)

    for run in (lambda: fns["solve"]("clip0", clip_dir), lambda: solve_stage.prepare("clip0", clip_dir),
                lambda: fns["qc"]("clip0", qc_dir)):
        with pytest.raises(ValueError, match="stale driftqc artifacts"):
            run()
    assert slice_calls == []
    assert not (clip_dir / "solve.npz").exists() and not (qc_dir / "qc.json").exists()
