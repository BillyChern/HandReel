"""Tests for the declared-cost device-memory ledger (`fastdo.engine.pipelined_runner.MemoryLedger`) and its use
by the wave-pipelined runner.

Unit tests drive `MemoryLedger` from plain threads. The runner tests use CPU-only fakes with three in-process CUDA
classes (`mask`, `geom`, `gpu`) and two subprocess worker classes (`sam3d_worker`, `hawor_worker`). A shared `Probe`
adds each gated call's declared cost, computed here from the test's own table (never from the code under test), to a
running sum while the call is inside, and records every moment the sum passed the capacity while more than one gated
call ran. Blocking scenarios are driven by events and barriers; nothing waits on a sleep.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fastdo.bench import gpu_budget
from fastdo.bench.bench_batched import compare_artifact_dirs
from fastdo.engine import pipelined_runner
from fastdo.engine.apportion import BatchRecord, LedgerBatchRecord, read_batch_log
from fastdo.engine.batched_runner import (
    BatchedEngineConfig,
    MemoryLedgerConfig,
    StageBatchConfig,
    run_manifest_batched,
)
from fastdo.engine.batched_stages import BatchedStage, BatchOutcome
from fastdo.engine.batching import dynamic_batches
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, StageFn
from fastdo.engine.pipelined_runner import (
    MemoryLedger,
    PipelinedRunner,
    _Stopped,
    runs_cuda_in_process,
)
from tests.engine.fakes import random_frames, write_decode

TIMEOUT_S = 30.0
CAP_GB = 12.0                   # FASTDO_GPU_MEM_GB pinned by the autouse fixture
ROWS = {"c0": 4, "c1": 7, "c2": 5, "c3": 9, "c4": 6, "c5": 3, "c6": 8}
IDS = list(ROWS)
STAGES = ("decode", "mask", "shape", "hands", "geometry", "tracks", "qc")
PREV = {stage: STAGES[i - 1] for i, stage in enumerate(STAGES) if i}
RESOURCES = {"shape": "sam3d_worker", "hands": "hawor_worker", "geometry": "geom", "mask": "mask"}
WAVE = 2
# shape and hands declare costs far above any capacity: they are subprocess classes, so the ledger must ignore them
BATCHES = {"mask": StageBatchConfig(max_batch=2, budget_gb=4.0, fixed_cost_gb=2.0),
           "geometry": StageBatchConfig(max_batch=2, budget_gb=10.0, fixed_cost_gb=1.0, cost_per_frame_gb=0.1),
           "tracks": StageBatchConfig(max_batch=1, budget_gb=3.0),
           "shape": StageBatchConfig(max_batch=1, budget_gb=50.0),
           "hands": StageBatchConfig(max_batch=2, budget_gb=80.0, fixed_cost_gb=40.0)}
GATED = {"mask", "geometry", "tracks"}
FIELDS = [f.name for f in fields(BatchRecord)]


def declared(stage: str, ids: Sequence[str]) -> float:
    """The design's declared cost, written out independently: fixed x n + per-frame x frames, else budget_gb."""
    if stage == "mask":
        return 2.0 * len(ids)
    if stage == "geometry":
        return 1.0 * len(ids) + 0.1 * sum(ROWS[c] for c in ids)
    assert stage == "tracks"
    return 3.0


@pytest.fixture(autouse=True)
def _cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gpu_budget.ENV_VAR, str(CAP_GB))


# --- MemoryLedger ---------------------------------------------------------------------------------------------------

def _wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not predicate():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.001)


def _spawn(fn: Callable[[], Any]) -> tuple[threading.Thread, dict[str, Any]]:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            box["exc"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


def test_concurrent_reservations_never_exceed_capacity() -> None:
    """8 threads x 60 attempts of random costs (some above capacity): whenever two or more attempts hold the ledger,
    their costs sum to at most the capacity; an oversized one only ever runs alone."""
    capacity = 5.0
    ledger = MemoryLedger(capacity)
    lock = threading.Lock()
    inside: list[float] = []
    violations: list[tuple[float, ...]] = []
    peak = [0.0, 0]

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(60):
            cost = rng.choice([0.0, 0.5, 1.25, 2.0, 3.0, 7.0])
            ledger.acquire(cost)
            try:
                with lock:
                    inside.append(cost)
                    if len(inside) > 1 and sum(inside) > capacity + 1e-9:
                        violations.append(tuple(inside))
                    peak[0], peak[1] = max(peak[0], sum(inside)), max(peak[1], len(inside))
                time.sleep(rng.random() * 1e-4)
                with lock:
                    inside.remove(cost)
            finally:
                ledger.release(cost)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads)
    assert violations == []
    assert peak[1] >= 2                             # the ledger did admit concurrent attempts
    assert (ledger.reserved_gb, ledger.n_holders, ledger.n_waiting) == (0.0, 0, 0)


def test_an_oversized_attempt_runs_alone_and_release_wakes_waiters() -> None:
    ledger = MemoryLedger(2.0)
    assert ledger.acquire(5.0) >= 0                 # empty ledger: an attempt above capacity still starts
    thread, box = _spawn(lambda: ledger.acquire(0.0))   # even a zero-cost attempt waits behind it
    _wait_for(lambda: ledger.n_waiting == 1)
    assert "result" not in box
    ledger.release(5.0)
    thread.join(TIMEOUT_S)
    assert box["result"] >= 0 and ledger.n_holders == 1
    ledger.release(0.0)

    ledger.acquire(1.5)
    thread, box = _spawn(lambda: ledger.acquire(1.0))   # 1.5 + 1.0 > 2.0: waits
    _wait_for(lambda: ledger.n_waiting == 1)
    ledger.acquire(0.5)                                 # 1.5 + 0.5 fits beside it
    ledger.release(1.5)
    thread.join(TIMEOUT_S)                              # 0.5 + 1.0 fits: the release woke it
    assert "result" in box and ledger.reserved_gb == pytest.approx(1.5)
    ledger.release(0.5)
    ledger.release(1.0)
    assert ledger.reserved_gb == 0.0                    # reset exactly when the last holder leaves


def test_halt_interrupts_waiting_and_refuses_new_attempts() -> None:
    ledger = MemoryLedger(1.0)
    ledger.acquire(1.0)
    thread, box = _spawn(lambda: ledger.acquire(0.5))
    _wait_for(lambda: ledger.n_waiting == 1)
    ledger.halt()
    thread.join(TIMEOUT_S)
    assert not thread.is_alive() and isinstance(box.get("exc"), _Stopped)
    assert ledger.n_waiting == 0 and ledger.n_holders == 1        # the waiter reserved nothing
    ledger.release(1.0)
    with pytest.raises(_Stopped):
        ledger.acquire(0.0)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, "1.0", True])
def test_invalid_costs_and_capacities_are_refused(bad: Any) -> None:
    ledger = MemoryLedger(1.0)
    with pytest.raises(ValueError, match="cost_gb"):
        ledger.acquire(bad)
    with pytest.raises(ValueError, match="cost_gb"):
        ledger.release(bad)
    with pytest.raises(ValueError, match="capacity_gb"):
        MemoryLedger(bad)
    with pytest.raises(ValueError, match="capacity_gb"):
        MemoryLedger(0.0)
    with pytest.raises(RuntimeError, match="without"):
        ledger.release(0.5)


def test_declared_cost_is_the_grouping_cost() -> None:
    """`declared_cost_gb` is the sum `dynamic_batches` closes a group on, else `budget_gb`, else 0."""
    frames = {"a": 40, "b": 162, "c": 49, "d": 7}
    sb = StageBatchConfig(max_batch=3, budget_gb=9.0, fixed_cost_gb=1.81, cost_per_frame_gb=0.013)
    for group in dynamic_batches(list(frames), sb.max_batch, sb.resolved_budget_gb(),
                                 lambda c: sb.clip_cost_gb(frames[c])):
        running = 0.0
        for c in group:
            running += 1.81 + 0.013 * frames[c]
        assert sb.declared_cost_gb([frames[c] for c in group]) == running
    assert StageBatchConfig(max_batch=2, budget_gb=4.38, fixed_cost_gb=2.18).declared_cost_gb([10, 99]) == 4.36
    assert StageBatchConfig(budget_gb=3.0).declared_cost_gb([10, 20]) == 3.0
    assert StageBatchConfig(max_batch=1, unmeasured=True).declared_cost_gb([10]) == 0.0


@pytest.mark.parametrize(("stage", "cls", "expected"), [
    ("mask", "gpu", True), ("mask", "mask", True), ("geometry", "geom", True), ("prompt", "prompt", True),
    ("shape", "sam3d_worker", False), ("hands", "hawor_worker", False), ("shape", "gpu", True),
    ("decode", "decode", False), ("qc", "qc", False)])
def test_in_process_rule(stage: str, cls: str, expected: bool) -> None:
    assert runs_cuda_in_process(stage, cls) is expected


# --- the runner -----------------------------------------------------------------------------------------------------

class Probe:
    """Every call's stage and clips in start order, plus the declared cost held by the gated calls inside."""

    def __init__(self, capacity: float, overrides: dict[str, float] | None = None) -> None:
        self.capacity = capacity
        self.overrides = overrides or {}
        self.lock = threading.Lock()
        self.starts: list[tuple[str, tuple[str, ...]]] = []
        self.inside: list[float] = []
        self.violations: list[tuple[float, ...]] = []
        self.max_gated = 0

    def enter(self, stage: str, ids: Sequence[str]) -> None:
        with self.lock:
            self.starts.append((stage, tuple(ids)))
            if stage in GATED:
                self.inside.append(self.cost(stage, ids))
                self.max_gated = max(self.max_gated, len(self.inside))
                if len(self.inside) > 1 and sum(self.inside) > self.capacity + 1e-9:
                    self.violations.append(tuple(self.inside))

    def leave(self, stage: str, ids: Sequence[str]) -> None:
        if stage in GATED:
            with self.lock:
                self.inside.remove(self.cost(stage, ids))

    def cost(self, stage: str, ids: Sequence[str]) -> float:
        return self.overrides[stage] if stage in self.overrides else declared(stage, ids)

    def order(self, stage: str) -> list[tuple[str, ...]]:
        return [ids for s, ids in self.starts if s == stage]


Hook = Callable[[str, Sequence[str]], None]


def _write(stage: str, clip_id: str, artifact_dir: Path) -> None:
    if stage == "decode":
        write_decode(artifact_dir, random_frames(ROWS[clip_id], seed=int(clip_id[1:])), stride=3)
        return
    prev = artifact_dir / ("decode.npz" if PREV[stage] == "decode" else f"{PREV[stage]}.json")
    digest = hashlib.sha256(prev.read_bytes()).hexdigest()[:16]
    (artifact_dir / f"{stage}.json").write_text(json.dumps({"clip_id": clip_id, "stage": stage, "prev": digest}))


def _body(stage: str, ids: Sequence[str], probe: Probe, sleep_s: float, hook: Hook | None,
          write: Callable[[], None]) -> None:
    probe.enter(stage, ids)
    try:
        if hook is not None:
            hook(stage, ids)
        if sleep_s:
            time.sleep(sleep_s)
        write()
    finally:
        probe.leave(stage, ids)


def make_fn(stage: str, probe: Probe, sleep_s: float, hook: Hook | None) -> StageFn:
    def fn(clip_id: str, artifact_dir: Path) -> None:
        _body(stage, [clip_id], probe, sleep_s, hook, lambda: _write(stage, clip_id, Path(artifact_dir)))
    return fn


class Batch:
    def __init__(self, stage: str, probe: Probe, sleep_s: float, hook: Hook | None) -> None:
        self.stage, self.probe, self.sleep_s, self.hook = stage, probe, sleep_s, hook

    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        def write() -> None:
            for clip_id, artifact_dir in clips:
                _write(self.stage, clip_id, Path(artifact_dir))
        _body(self.stage, [c for c, _ in clips], self.probe, self.sleep_s, self.hook, write)
        return BatchOutcome()


def build(probe: Probe, sleep_s: float = 0.0, hook: Hook | None = None) -> dict[str, StageFn | BatchedStage]:
    stages: dict[str, StageFn | BatchedStage] = {s: make_fn(s, probe, sleep_s, hook) for s in STAGES}
    stages["mask"] = Batch("mask", probe, sleep_s, hook)
    stages["geometry"] = Batch("geometry", probe, sleep_s, hook)
    return stages


def config(root: Path, capacity: float | None, depth: int = 3,
           costs: dict[str, float] | None = None) -> BatchedEngineConfig:
    ledger = None if capacity is None else MemoryLedgerConfig(resident_gb=CAP_GB - capacity, costs_gb=costs or {})
    return BatchedEngineConfig(EngineConfig(artifact_root=root, stages=STAGES), BATCHES, max_clips_per_wave=WAVE,
                               run_id="t0", pipeline_waves=True, pipeline_depth=depth, pipeline_resources=RESOURCES,
                               memory_ledger=ledger)


def new_manifest() -> ClipManifest:
    return ClipManifest.create(IDS, [f"{c}.mp4" for c in IDS], [3 * ROWS[c] for c in IDS])


def run_guarded(fn: Callable[[], Any]) -> dict[str, Any]:
    thread, box = _spawn(fn)
    thread.join(TIMEOUT_S)
    assert not thread.is_alive(), "the run hung"
    return box


def _key(record: BatchRecord) -> tuple[object, ...]:
    return (record.batch_id, record.stage, record.clip_ids, record.weights, record.errors, record.retry_of,
            record.batched, record.peak_gpu_mem_gb)


@pytest.mark.parametrize("capacity", [6.5, 2.5])
def test_ledger_bounds_the_declared_cost_and_changes_only_scheduling(tmp_path: Path, capacity: float) -> None:
    """6.5 GiB admits some pairs of in-process attempts (geometry beside tracks) and refuses others (mask pairs beside
    anything); 2.5 GiB makes every mask pair, geometry pair and tracks attempt oversized, so each runs alone."""
    probes = {flag: Probe(capacity) for flag in ("off", "on")}
    manifests = {flag: new_manifest() for flag in probes}
    for flag, probe in probes.items():
        cfg = config(tmp_path / flag, capacity if flag == "on" else None)
        stages = build(probe, sleep_s=0.005)
        box = run_guarded(functools.partial(run_manifest_batched, manifests[flag], cfg, stages))
        assert "exc" not in box, box.get("exc")

    on, off = probes["on"], probes["off"]
    assert on.violations == []
    if capacity == 2.5:
        assert on.max_gated == 1
    for stage in STAGES:                            # per-stage order unchanged
        assert on.order(stage) == off.order(stage), stage
    diff = compare_artifact_dirs(tmp_path / "off", tmp_path / "on")
    assert diff["n_mismatched"] == 0 and diff["missing_in_a"] == [] == diff["missing_in_b"]
    for clip_id in IDS:
        for stage in STAGES:
            a, b = manifests["off"].status(clip_id, stage), manifests["on"].status(clip_id, stage)
            assert (a.state, a.batch_id) == (b.state, b.batch_id) and a.state == "done", (clip_id, stage)
    records = {f: read_batch_log(tmp_path / f / "batches.jsonl") for f in probes}
    assert sorted(map(_key, records["off"])) == sorted(map(_key, records["on"]))     # batch ids and groups unchanged

    for record in records["on"]:                    # the wait is recorded exactly on gated attempts
        assert isinstance(record, LedgerBatchRecord) is (record.stage in GATED), record.batch_id
        if isinstance(record, LedgerBatchRecord):
            assert isinstance(record.ledger_wait_ns, int) and record.ledger_wait_ns >= 0
    for line in (tmp_path / "off" / "batches.jsonl").read_text().splitlines():
        assert list(json.loads(line)) == FIELDS     # today's keys, today's order
    for line in (tmp_path / "on" / "batches.jsonl").read_text().splitlines():
        payload = json.loads(line)
        assert list(payload) == FIELDS + (["ledger_wait_ns"] if payload["stage"] in GATED else [])
    assert all(type(r) is BatchRecord for r in records["off"])


def test_the_probe_sees_an_unbounded_overlap_without_the_ledger(tmp_path: Path) -> None:
    """Power check for the probe: with the ledger off, mask(wave 1) is held inside until geometry(wave 0) is inside,
    a 4.0 + 3.1 GiB moment the 6.5 GiB ledger would never admit."""
    probe = Probe(6.5)
    geometry_inside = threading.Event()

    def hook(stage: str, ids: Sequence[str]) -> None:
        if stage == "geometry" and "c0" in ids:
            geometry_inside.set()
        if stage == "mask" and "c2" in ids:
            assert geometry_inside.wait(TIMEOUT_S)

    box = run_guarded(lambda: run_manifest_batched(new_manifest(), config(tmp_path, None), build(probe, hook=hook)))
    assert "exc" not in box and probe.violations


def test_subprocess_worker_classes_are_not_gated(tmp_path: Path) -> None:
    """At 2.5 GiB every in-process attempt is oversized and runs alone. shape (declared 50 GiB) and hands (40 GiB a
    clip) must still run INSIDE in-process attempts: each rendezvous needs both calls inside at once, which is
    impossible if the ledger gated the worker class, so a gated worker breaks a barrier and fails a clip."""
    probe = Probe(2.5)
    with_shape, with_hands = threading.Barrier(2, timeout=TIMEOUT_S), threading.Barrier(2, timeout=TIMEOUT_S)

    def hook(stage: str, ids: Sequence[str]) -> None:
        if (stage == "shape" and ids == ["c0"]) or (stage == "mask" and "c2" in ids):
            with_shape.wait()
        if (stage == "hands" and ids == ["c0"]) or (stage == "mask" and "c4" in ids):
            with_hands.wait()

    manifest = new_manifest()
    box = run_guarded(lambda: run_manifest_batched(manifest, config(tmp_path, 2.5), build(probe, hook=hook)))
    assert "exc" not in box
    assert all(manifest.status(c, "qc").state == "done" for c in IDS)
    assert probe.violations == [] and probe.max_gated == 1
    records = read_batch_log(tmp_path / "batches.jsonl")
    assert {r.stage for r in records if isinstance(r, LedgerBatchRecord)} == GATED


def test_halt_releases_a_class_thread_waiting_on_the_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """geometry(wave 0) holds the ledger and raises KeyboardInterrupt once mask(wave 1) is waiting on it: the run
    halts before the reservation is released, so the waiter never starts, and the caller gets the exception. (The
    staged release makes the ordering observable: releasing first would admit the waiter.)"""
    geometry_inside = threading.Event()
    ledgers: list[MemoryLedger] = []

    class Staged(MemoryLedger):
        """Holds the mask class's second request until geometry(wave 0) is inside, so it really waits."""

        def __init__(self, capacity_gb: float) -> None:
            super().__init__(capacity_gb)
            self.mask_requests = 0
            ledgers.append(self)

        def acquire(self, cost_gb: float) -> int:
            if threading.current_thread().name == "fastdo-pipeline-mask":
                self.mask_requests += 1
                if self.mask_requests == 2:
                    assert geometry_inside.wait(TIMEOUT_S)
            return super().acquire(cost_gb)

        def release(self, cost_gb: float) -> None:
            """geometry's release lets the waiter settle before the exception travels on, so a release that came
            before the halt would visibly start mask(wave 1)."""
            super().release(cost_gb)
            if threading.current_thread().name == "fastdo-pipeline-geom":
                _wait_for(lambda: self.n_waiting == 0)

    def hook(stage: str, ids: Sequence[str]) -> None:
        if stage == "geometry" and "c0" in ids:
            geometry_inside.set()
            _wait_for(lambda: ledgers[0].n_waiting == 1)
            raise KeyboardInterrupt("geometry")

    monkeypatch.setattr(pipelined_runner, "MemoryLedger", Staged)
    probe = Probe(2.5)
    manifest = new_manifest()
    runner = PipelinedRunner(manifest, config(tmp_path, 2.5, depth=2), build(probe, hook=hook))
    box = run_guarded(runner.run)

    assert isinstance(box.get("exc"), KeyboardInterrupt) and "result" not in box
    assert ("mask", ("c2", "c3")) not in probe.starts
    assert ledgers[0].n_waiting == 0 and ledgers[0].n_holders == 0
    records = read_batch_log(tmp_path / "batches.jsonl")
    assert not [r for r in records if r.stage == "geometry" or (r.stage == "mask" and "c2" in r.clip_ids)]
    assert manifest.status("c0", "geometry").state == "pending"


# --- per-attempt cost overrides ---------------------------------------------------------------------

def test_cost_overrides_are_validated_and_read_only() -> None:
    ledger = MemoryLedgerConfig(resident_gb=7.0, costs_gb={"tracks": 21.5, "solve": 22})
    assert dict(ledger.costs_gb) == {"tracks": 21.5, "solve": 22.0}
    assert ledger.attempt_cost_gb("tracks", 8.18) == 21.5 and ledger.attempt_cost_gb("mask", 4.36) == 4.36
    with pytest.raises(TypeError):
        ledger.costs_gb["mask"] = 1.0  # type: ignore[index]
    assert MemoryLedgerConfig(resident_gb=7.0).costs_gb == {}                  # `{resident_gb}` alone stays valid
    with pytest.raises(ValueError, match=r"unknown stage keys \['bogus'\]"):
        MemoryLedgerConfig(resident_gb=7.0, costs_gb={"bogus": 1.0})
    for bad in (-1.0, math.nan, math.inf, True, "3"):
        with pytest.raises(ValueError, match="costs_gb values"):
            MemoryLedgerConfig(resident_gb=7.0, costs_gb={"tracks": bad})     # type: ignore[dict-item]
    with pytest.raises(ValueError, match="costs_gb must map stage"):
        MemoryLedgerConfig(resident_gb=7.0, costs_gb=[("tracks", 1.0)])        # type: ignore[arg-type]


class Spy(MemoryLedger):
    """Records the cost each class thread reserved."""

    seen: ClassVar[list[tuple[str, float]]] = []

    def acquire(self, cost_gb: float) -> int:
        Spy.seen.append((threading.current_thread().name.removeprefix("fastdo-pipeline-"), cost_gb))
        return super().acquire(cost_gb)


def test_an_override_replaces_the_declared_cost_in_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """tracks declares 3.0 and geometry ~3 GiB a pair; overridden to 5.0 and 0.5 they are what the ledger reserves,
    mask (not listed) keeps its declared 2.0 a clip, and the 6.5 GiB bound holds on the overridden costs."""
    Spy.seen = []
    monkeypatch.setattr(pipelined_runner, "MemoryLedger", Spy)
    overrides = {"tracks": 5.0, "geometry": 0.5}
    probe = Probe(6.5, overrides)
    manifest = new_manifest()
    box = run_guarded(functools.partial(run_manifest_batched, manifest, config(tmp_path, 6.5, costs=overrides),
                                        build(probe, sleep_s=0.005)))
    assert "exc" not in box and all(manifest.status(c, "qc").state == "done" for c in IDS)
    by_class: dict[str, set[float]] = {}
    for cls, cost in Spy.seen:
        by_class.setdefault(cls, set()).add(cost)
    assert by_class == {"gpu": {5.0}, "geom": {0.5}, "mask": {4.0, 2.0}}     # c6 is a wave of one: 2.0
    assert probe.violations == []


def test_an_oversized_override_runs_alone_and_is_logged_once(tmp_path: Path,
                                                             caplog: pytest.LogCaptureFixture) -> None:
    """tracks overridden to 100 GiB, far above the 6.5 GiB capacity: every tracks attempt still runs, but with no
    other gated attempt inside; the start-up warning names it once."""
    overrides = {"tracks": 100.0}
    probe = Probe(6.5, overrides)
    manifest = new_manifest()
    with caplog.at_level("WARNING", logger="fastdo.engine.pipelined_runner"):
        box = run_guarded(functools.partial(run_manifest_batched, manifest, config(tmp_path, 6.5, costs=overrides),
                                            build(probe, sleep_s=0.005)))
    assert "exc" not in box and all(manifest.status(c, "tracks").state == "done" for c in IDS)
    assert probe.violations == []                  # 100 + anything > 6.5: tracks was always alone among gated calls
    oversized = [r for r in caplog.records if "costs_gb['tracks']" in r.getMessage()]
    assert len(oversized) == 1 and "runs alone" in oversized[0].getMessage()
