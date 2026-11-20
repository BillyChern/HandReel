"""`bench.batched.prewarm` (`fastdo.engine.prewarm`) loads the shape and hands worker-subprocess models
at run start instead of on first use, with nothing else changed.

Covered: the once-and-hand-over slot; the runner starts each listed hook on its own thread after its card admission, only
when listed and registered, and a failing hook never fails the run; the shape worker's prewarm performs the very load
its first clip would, outside the process-wide model-load lock, is used by the first clip (one load), falls back to the
first-use load on failure, and does nothing after the first clip; the hands stage sends the warm driver one zero-clip
request with the real calls' options, a real call waits for it, and it is a no-op without a warm driver or after a
call; the warm driver itself (the real `_hawor_batched_driver.serve` on a fake model API) serves real requests after a
zero-clip prewarm request with the bytes fresh processes write; artifacts and batch logs of a pipelined run are
identical with and without prewarm; the config key defaults to `[]`.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import threading
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
import trimesh
from omegaconf import OmegaConf

from fastdo.bench.bench_batched import compare_artifact_dirs, prewarm_stages
from fastdo.engine import batched_stages as bs
from fastdo.engine import workers_shape
from fastdo.engine.apportion import read_batch_log
from fastdo.engine.batched_runner import BatchedEngineConfig, run_manifest_batched
from fastdo.engine.batched_stages import HandsBatchStage
from fastdo.engine.prewarm import PrewarmableStageFn, PrewarmSlot, start_prewarm
from fastdo.engine.worker_support import WorkerContext
from fastdo.engine.workers_shape import (
    MESH_FILE,
    SAMPLES_FILE,
    SDF_FILE,
    SHAPE_META_FILE,
    make_shape_worker,
)
from fastdo.shape import ShapeConfig, ShapeResult
from tests.engine.fakes import random_frames, write_decode
from tests.engine.test_batched_stages import HANDS_CFG
from tests.engine.test_pipelined_runner import IDS, Probe, build, config, new_manifest, run_guarded
from tests.hands.test_hawor_core import _clips
from tests.hands.test_hawor_driver_cpu import _request, _RngApi

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
TIMEOUT_S = 30.0


# --- the slot ----------------------------------------------------------------------------------------------------------


def test_slot_hands_the_prewarmed_value_over_once() -> None:
    slot = PrewarmSlot("x")
    assert slot.prewarm(lambda: "model") and slot.state == "done"
    assert slot.take() == "model" and slot.state == "taken"
    assert slot.take() is None
    calls: list[int] = []
    assert not slot.prewarm(lambda: calls.append(1))           # after first use: nothing loads
    assert calls == []


def test_slot_without_prewarm_or_after_a_failure_gives_none() -> None:
    assert PrewarmSlot("x").take() is None

    def boom() -> None:
        raise RuntimeError("load failed")

    slot = PrewarmSlot("y")
    assert slot.prewarm(boom) is False and slot.take() is None


def test_take_waits_for_a_prewarm_in_progress() -> None:
    slot, release, got = PrewarmSlot("x"), threading.Event(), []
    loader = threading.Thread(target=slot.prewarm, args=(lambda: release.wait() and "model",))
    loader.start()
    while slot.state != "running":
        pass
    taker = threading.Thread(target=lambda: got.append(slot.take()))
    taker.start()
    taker.join(0.2)
    assert taker.is_alive() and got == []                      # blocked on the load
    release.set()
    taker.join(TIMEOUT_S)
    loader.join(TIMEOUT_S)
    assert got == ["model"]


def test_start_prewarm_runs_hooks_on_their_own_threads(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[tuple[str, str]] = []

    class _Hooked:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name, self.fail = name, fail

        def prewarm(self) -> None:
            seen.append((self.name, threading.current_thread().name))
            if self.fail:
                raise RuntimeError("no card")

    order: list[str] = []
    stages = {"shape": _Hooked("shape"), "hands": _Hooked("hands", fail=True), "mask": object()}
    with caplog.at_level(logging.WARNING):
        threads = start_prewarm(stages, ["shape", "hands", "mask", "tracks"], before=order.append)
        for t in threads:
            t.join(TIMEOUT_S)
    assert sorted(seen) == [("hands", "fastdo-prewarm-hands"), ("shape", "fastdo-prewarm-shape")]
    assert sorted(order) == ["hands", "shape"]
    text = caplog.text
    assert "prewarm hands raised RuntimeError" in text and "'mask' is a object without a prewarm() hook" in text
    assert "'tracks' is not registered" in text


# --- the shape worker ---------------------------------------------------------------------------------------------

BASE_CFG = ShapeConfig(backend="sam3d", deterministic=False, spconv_kernel_pin=False)
PREP: dict[str, Any] = {"sdf_resolution": 16, "sdf_padding": 0.15, "n_samples": 64, "samples_seed": 0}


class _LoadingFake:
    """A cube backend with a `load()` that records its thread, can block on an event and can fail once."""

    def __init__(self, log: list[str], gate: threading.Event | None = None, fail: bool = False) -> None:
        box = trimesh.creation.box(extents=(0.3, 0.2, 0.4))
        self.vertices, self.faces = np.asarray(box.vertices) + 0.1, np.asarray(box.faces)
        self.log, self.gate, self.fail, self.loaded = log, gate, fail, False

    def load(self) -> _LoadingFake:
        self.log.append(threading.current_thread().name)
        if self.gate is not None:
            assert self.gate.wait(TIMEOUT_S)
        if self.fail:
            raise RuntimeError("SAM 3D worker failed to load")
        self.loaded = True
        return self

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        assert self.loaded
        return ShapeResult(self.vertices, self.faces, None, {"decode": 0.5})


def _clip(d: Path) -> None:
    frames = random_frames(3, h=10, w=12, seed=7)
    masks = np.zeros((3, 10, 12), dtype=bool)
    masks[:, 2:8, 3:9] = True
    write_decode(d, frames)
    np.savez(d / "mask.npz", masks=masks, frame_idx=np.arange(3, dtype=np.int64) * 3,
             keyframe_index=np.array(1, dtype=np.int64))


def _factory(monkeypatch: pytest.MonkeyPatch, backends: list[_LoadingFake]) -> list[_LoadingFake]:
    built: list[_LoadingFake] = []

    def make(cfg: ShapeConfig) -> _LoadingFake:
        built.append(backends[len(built)])
        return built[-1]

    monkeypatch.setattr(workers_shape, "make_shape_backend", make)
    return built


def _artifacts(d: Path) -> dict[str, bytes]:
    meta = json.loads((d / SHAPE_META_FILE).read_text())
    return {name: (d / name).read_bytes() for name in (MESH_FILE, SDF_FILE, SAMPLES_FILE)} | {
        SHAPE_META_FILE: json.dumps(meta, sort_keys=True).encode()}


def test_shape_prewarm_is_the_first_clips_load_and_changes_no_byte(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    built = _factory(monkeypatch, [_LoadingFake(log), _LoadingFake(log)])
    plain_dir, warm_dir = tmp_path / "plain", tmp_path / "warm"
    for d in (plain_dir, warm_dir):
        _clip(d)
    plain = make_shape_worker(BASE_CFG, **PREP)
    plain("clip", plain_dir)
    warm = make_shape_worker(BASE_CFG, **PREP)
    assert isinstance(warm, PrewarmableStageFn)
    thread = threading.Thread(target=warm.prewarm, name="fastdo-prewarm-shape")
    thread.start()
    thread.join(TIMEOUT_S)
    warm("clip", warm_dir)
    warm("clip", warm_dir)                                     # a later clip reuses it
    assert len(built) == 2 and log == ["MainThread", "fastdo-prewarm-shape"]
    assert _artifacts(plain_dir) == _artifacts(warm_dir)


def test_shape_prewarm_holds_no_model_load_lock_and_the_first_clip_waits(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    gate, log = threading.Event(), []
    built = _factory(monkeypatch, [_LoadingFake(log, gate)])
    _clip(tmp_path)
    worker = make_shape_worker(BASE_CFG, **PREP)
    assert isinstance(worker, PrewarmableStageFn)
    loader = threading.Thread(target=worker.prewarm, name="fastdo-prewarm-shape")
    loader.start()
    while not log:
        pass
    # another worker's in-process model loads while SAM 3D is still loading (WorkerContext serializes loads)
    other = run_guarded(lambda: WorkerContext(loader=lambda: "mask model").model())
    assert other == {"result": "mask model"}
    clip = threading.Thread(target=worker, args=("clip", tmp_path))
    clip.start()
    clip.join(0.2)
    assert clip.is_alive() and not (tmp_path / SHAPE_META_FILE).exists()      # waiting for the prewarm
    # ... and while the first clip waits for it, loads elsewhere still proceed (it waits outside the lock too)
    other = run_guarded(lambda: WorkerContext(loader=lambda: "tracks model").model())
    assert other == {"result": "tracks model"}
    gate.set()
    clip.join(TIMEOUT_S)
    loader.join(TIMEOUT_S)
    assert (tmp_path / SHAPE_META_FILE).is_file() and len(built) == 1


def test_a_failed_shape_prewarm_falls_back_to_the_first_use_load(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    built = _factory(monkeypatch, [_LoadingFake(log, fail=True), _LoadingFake(log)])
    _clip(tmp_path)
    worker = make_shape_worker(BASE_CFG, **PREP)
    assert isinstance(worker, PrewarmableStageFn)
    worker.prewarm()
    worker("clip", tmp_path)
    assert len(built) == 2 and built[1].loaded and (tmp_path / SHAPE_META_FILE).is_file()


def test_a_shape_prewarm_after_the_first_clip_loads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = _factory(monkeypatch, [_LoadingFake([]), _LoadingFake([])])
    _clip(tmp_path)
    worker = make_shape_worker(BASE_CFG, **PREP)
    assert isinstance(worker, PrewarmableStageFn)
    worker("clip", tmp_path)
    worker.prewarm()
    assert len(built) == 1


def test_a_cached_shape_worker_offers_no_prewarm(tmp_path: Path) -> None:
    assert not hasattr(make_shape_worker(BASE_CFG, cache_dir=tmp_path / "cache", **PREP), "prewarm")


# --- the hands stage ----------------------------------------------------------------------------------------------


class _Warm:
    """A stand-in `WarmHandsDriver`: records each request's body and thread; `gate` blocks a request."""

    def __init__(self, cfg: Any, log_dir: Path) -> None:
        self.cfg, self.log_dir = cfg, log_dir
        self.requests: list[tuple[dict[str, Any], str]] = []
        self.gate: threading.Event | None = None

    def request(self, request: Path, timeout: float, log_path: Path) -> int:
        self.requests.append((json.loads(Path(request).read_text()), threading.current_thread().name))
        if self.gate is not None:
            assert self.gate.wait(TIMEOUT_S)
        return 0

    def close(self) -> None:
        pass


def test_hands_prewarm_sends_one_zero_clip_request_with_the_real_options(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bs, "WarmHandsDriver", _Warm)
    calls: list[Any] = []
    monkeypatch.setattr(bs, "run_hands_call", lambda inputs, cfg, job, lock, warm=None: (
        calls.append((inputs, warm)), (type("R", (), {"peak_cuda_gb": 0.0})(), {}))[1])
    monkeypatch.setattr(bs, "hands_focal_length", lambda d: 500.0)
    stage = HandsBatchStage(replace(HANDS_CFG, warm_driver=True), tmp_path / "jobs", tmp_path / ".gpu.lock",
                            lambda c, d: None)
    assert isinstance(stage.warm, _Warm)
    stage.prewarm()
    (body, _thread), = stage.warm.requests
    assert body["clips"] == [] and body["hawor_root"] == str(Path(HANDS_CFG.hawor_root).absolute())
    real = bs.write_request(tmp_path / "probe", [], replace(HANDS_CFG, warm_driver=True))
    assert body["options"] == json.loads(real.read_text())["options"]
    assert (tmp_path / "jobs" / HandsBatchStage.PREWARM_JOB / "request.json").is_file()
    stage.run_batch([("c0", tmp_path / "c0")], "hands-00000")
    stage.prewarm()                                            # after a real call: nothing sent
    assert len(stage.warm.requests) == 1 and len(calls) == 1 and calls[0][1] is stage.warm


def test_a_hands_call_waits_for_the_prewarm_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bs, "WarmHandsDriver", _Warm)
    order: list[str] = []
    monkeypatch.setattr(bs, "run_hands_call", lambda inputs, cfg, job, lock, warm=None: (
        order.append("call"), (type("R", (), {"peak_cuda_gb": 0.0})(), {}))[1])
    monkeypatch.setattr(bs, "hands_focal_length", lambda d: 500.0)
    stage = HandsBatchStage(replace(HANDS_CFG, warm_driver=True), tmp_path / "jobs", tmp_path / ".gpu.lock",
                            lambda c, d: None)
    assert isinstance(stage.warm, _Warm)
    stage.warm.gate = threading.Event()
    pre = threading.Thread(target=stage.prewarm)
    pre.start()
    while not stage.warm.requests:
        pass
    call = threading.Thread(target=stage.run_batch, args=([("c0", tmp_path / "c0")], "hands-00000"))
    call.start()
    call.join(0.2)
    assert call.is_alive() and order == []
    stage.warm.gate.set()
    call.join(TIMEOUT_S)
    pre.join(TIMEOUT_S)
    assert order == ["call"]


def test_hands_prewarm_without_a_warm_driver_does_nothing(tmp_path: Path) -> None:
    stage = HandsBatchStage(HANDS_CFG, tmp_path / "jobs", tmp_path / ".gpu.lock", lambda c, d: None)
    assert stage.warm is None
    stage.prewarm()
    assert not (tmp_path / "jobs").exists()


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    import importlib.util
    import sys

    from tests.hands.test_hawor_driver_cpu import DRIVER_FILE

    spec = importlib.util.spec_from_file_location("fastdo_test_hawor_driver_258", DRIVER_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_warm_driver_serves_real_requests_after_a_zero_clip_prewarm_byte_for_byte(driver: ModuleType,
                                                                                    tmp_path: Path) -> None:
    """The real serve loop (fake models whose detections read torch's RNG): a zero-clip first request loads and warms
    up the models, and the real requests after it write exactly the bytes fresh processes write."""
    from tests.hands.test_hawor_driver_cpu import REPO

    shared = driver.load_shared_modules(REPO)
    clips = _clips()
    fresh, warm = tmp_path / "fresh", tmp_path / "warm"
    for i in range(2):
        (fresh / f"r{i}").mkdir(parents=True)
        torch.manual_seed(0)
        assert driver.run_request(_request(fresh / f"r{i}", clips), fresh / f"r{i}" / "response.json", shared,
                                  _RngApi(clips), device="cpu") == 0
    (warm / "prewarm").mkdir(parents=True)
    empty = _request(warm / "prewarm", []) | {"hawor_root": str(tmp_path)}
    (warm / "prewarm" / "request.json").write_text(json.dumps(empty))
    paths = [str(warm / "prewarm" / "request.json")]
    for i in range(2):
        (warm / f"r{i}").mkdir(parents=True)
        body = _request(warm / f"r{i}", clips) | {"hawor_root": str(tmp_path)}
        (warm / f"r{i}" / "request.json").write_text(json.dumps(body))
        paths.append(str(warm / f"r{i}" / "request.json"))
    api, protocol = _RngApi(clips), io.StringIO()
    torch.manual_seed(0)

    code = driver.serve(io.StringIO("".join(f"{p}\n" for p in paths)), protocol, shared, api, device="cpu")

    assert code == 0 and api.warm_ups == 1
    assert [json.loads(line) for line in protocol.getvalue().splitlines()] == [
        {"request": p, "code": 0, "serving": True} for p in paths]
    pre = json.loads((warm / "prewarm" / "response.json").read_text())
    assert pre["clips"] == [] and pre["error"] == "" and pre["load_seconds"] > 0.0
    for i in range(2):
        for c in range(2):
            assert ((warm / f"r{i}" / "out" / f"clip{c}.npz").read_bytes()
                    == (fresh / f"r{i}" / "out" / f"clip{c}.npz").read_bytes()), (i, c)


# --- the pipelined runner ---------------------------------------------------------------------------------------------


class _Prewarmed:
    """Wraps a fake stage with a `prewarm()` that records when and where it ran."""

    def __init__(self, inner: Any, log: list[tuple[str, str]], fail: bool = False) -> None:
        self.inner, self.log, self.fail = inner, log, fail

    def __call__(self, clip_id: str, artifact_dir: Path) -> None:
        self.log.append(("call", threading.current_thread().name))
        self.inner(clip_id, artifact_dir)

    def prewarm(self) -> None:
        self.log.append(("prewarm", threading.current_thread().name))
        if self.fail:
            raise RuntimeError("prewarm exploded")


def _run(root: Path, prewarm: tuple[str, ...], log: list[tuple[str, str]], fail: bool = False) -> Any:
    stages = build(Probe())
    stages["shape"] = _Prewarmed(stages["shape"], log, fail)
    cfg: BatchedEngineConfig = dataclasses.replace(config(root, True, depth=2, persist=False), prewarm=prewarm)
    box = run_guarded(lambda: run_manifest_batched(new_manifest(), cfg, stages))
    assert "exc" not in box, box.get("exc")
    return box["result"]


def _log_keys(path: Path) -> list[tuple[object, ...]]:
    return [(r.batch_id, r.stage, r.clip_ids, r.weights, r.errors, r.retry_of) for r in read_batch_log(path)]


def test_runner_prewarms_listed_stages_and_changes_no_artifact(tmp_path: Path) -> None:
    off_log: list[tuple[str, str]] = []
    on_log: list[tuple[str, str]] = []
    off = _run(tmp_path / "off", (), off_log)
    on = _run(tmp_path / "on", ("shape", "retarget"), on_log)
    assert all(kind == "call" for kind, _ in off_log)
    assert on_log.count(("prewarm", "fastdo-prewarm-shape")) == 1
    assert len([k for k, _ in on_log if k == "call"]) == len(IDS)
    report = compare_artifact_dirs(tmp_path / "off", tmp_path / "on")
    assert report["n_mismatched"] == 0 and not report["missing_in_a"] and not report["missing_in_b"]
    assert sorted(_log_keys(off.batch_log)) == sorted(_log_keys(on.batch_log))    # line ORDER interleaves


def test_a_failing_prewarm_never_fails_the_run(tmp_path: Path) -> None:
    log: list[tuple[str, str]] = []
    _run(tmp_path, ("shape",), log, fail=True)
    assert ("prewarm", "fastdo-prewarm-shape") in log and len([k for k, _ in log if k == "call"]) == len(IDS)


def test_prewarm_config_validates() -> None:
    engine = config(Path("/tmp/x"), True).engine
    assert BatchedEngineConfig(engine).prewarm == ()
    assert BatchedEngineConfig(engine, prewarm=["shape", "hands"]).prewarm == ("shape", "hands")  # type: ignore[arg-type]
    for bad in (("shape", "shape"), ("",), (3,)):
        with pytest.raises(ValueError, match="prewarm must be distinct stage names"):
            BatchedEngineConfig(engine, prewarm=bad)  # type: ignore[arg-type]


def test_bench_key_defaults_to_empty_and_is_checked() -> None:
    cfg = OmegaConf.load(_ROOT_CONFIG)
    assert prewarm_stages(cfg.bench) == tuple(OmegaConf.to_container(cfg.bench.batched.prewarm))  # type: ignore[arg-type]
    for value, ok in (([], ()), (None, ()), (["shape", "hands"], ("shape", "hands"))):
        bench = OmegaConf.create({"batched": {"prewarm": value, "pipeline_waves": True}})
        assert prewarm_stages(bench) == ok
    for bad in (["shape", "shape"], ["sahpe"], "shape"):
        with pytest.raises(ValueError, match="bench.batched.prewarm"):
            prewarm_stages(OmegaConf.create({"batched": {"prewarm": bad, "pipeline_waves": True}}))


def test_the_bench_entry_point_reaches_both_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachability at the registered entry point: the composed production config with the key set gives a runner
    config that lists the stages and batched stages that offer the hooks (nothing loads while building)."""
    from hydra import compose, initialize_config_dir

    from fastdo.bench.bench_batched import batched_engine_config, build_batched_stages
    from fastdo.bench.bench_stages import build_stage_fns
    from fastdo.engine.orchestrator import EngineConfig

    monkeypatch.setenv("FASTDO_GPU_MEM_GB", "32")           # the production config is certified at the 32 GB cap
    repo = Path(__file__).resolve().parents[2]
    base = [f"paths.root={repo}", f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}",
            "bench.hands_source=batched"]
    with initialize_config_dir(config_dir=str(repo / "run/conf"), version_base=None):
        on = compose(config_name="config", overrides=[*base, "bench.batched.prewarm=[shape,hands]"])
    fns, _ = build_stage_fns(on, repo, [])
    stages = build_batched_stages(on, fns, tmp_path / "jobs", tmp_path / ".gpu.lock")
    engine = EngineConfig(artifact_root=tmp_path / "engine", stages=("shape", "hands"))
    assert batched_engine_config(on.bench, engine).prewarm == ("shape", "hands")
    assert isinstance(stages["shape"], PrewarmableStageFn) and isinstance(stages["hands"], HandsBatchStage)
    assert callable(stages["hands"].prewarm)
