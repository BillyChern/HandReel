"""Tests for `pipeline_waves`: the wave-pipelined batched runner (`fastdo.engine.pipelined_runner`).

The fakes are CPU-only and deterministic: `decode` writes `decode.npz`, every later stage writes `<stage>.json` holding
a digest of the previous stage's artifact (so a stage that ran before its predecessor committed would fail or differ),
and a shared `Probe` records every call's stage, clips, thread and `perf_counter_ns` interval plus the number of calls
running per resource class. Overlap and ordering are asserted on those instrumented intervals and counts; the one
end-to-end wall-clock comparison takes each arm's fastest of interleaved passes, since load only inflates a wall;
blocking scenarios are driven by events, never by sleeps.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from fastdo.bench.bench_batched import compare_artifact_dirs
from fastdo.bench.profiler import ProfileReport
from fastdo.engine.apportion import BatchRecord, check_apportionment, read_batch_log
from fastdo.engine.batched_runner import (
    BatchedEngineConfig,
    BatchRunReport,
    StageBatchConfig,
    run_manifest_batched,
)
from fastdo.engine.batched_stages import BatchedStage, BatchOutcome
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, StageFn, gpu_lock, hold_gpu_lock_for_process
from fastdo.engine.pipelined_runner import GPU_CLASS, PipelinedRunner, stage_classes
from tests.engine.fakes import random_frames, write_decode

ROWS = {"c0": 4, "c1": 7, "c2": 5, "c3": 9, "c4": 6, "c5": 3, "c6": 8}
IDS = list(ROWS)
STAGES = ("decode", "mask", "shape", "hands", "geometry", "qc")
RESOURCES = {"shape": "sam3d_worker", "hands": "hawor_worker"}
CLASSES = stage_classes(STAGES, RESOURCES)
WAVE = 2
WAVES = [IDS[i:i + WAVE] for i in range(0, len(IDS), WAVE)]
WAVE_OF = {clip_id: k for k, wave in enumerate(WAVES) for clip_id in wave}
PREV = {stage: STAGES[i - 1] for i, stage in enumerate(STAGES) if i}
TIMEOUT_S = 30.0


def test_stage_classes_follow_the_design() -> None:
    assert CLASSES == {"decode": "decode", "mask": GPU_CLASS, "shape": "sam3d_worker", "hands": "hawor_worker",
                       "geometry": GPU_CLASS, "qc": "qc"}
    assert stage_classes(STAGES, {})["shape"] == GPU_CLASS == stage_classes(STAGES, {})["hands"]


@dataclass(frozen=True)
class Call:
    stage: str
    clip_ids: tuple[str, ...]
    thread: str
    start_ns: int
    end_ns: int


class Probe:
    """Thread-safe log of every fake call, with the peak number of calls running per class and overall."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[Call] = []
        self.active: Counter[str] = Counter()
        self.max_active: Counter[str] = Counter()
        self.running = self.max_running = 0

    @contextmanager
    def call(self, stage: str, ids: Sequence[str]) -> Iterator[None]:
        cls = CLASSES[stage]
        with self.lock:
            self.active[cls] += 1
            self.max_active[cls] = max(self.max_active[cls], self.active[cls])
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            with self.lock:
                self.active[cls] -= 1
                self.running -= 1
                self.calls.append(Call(stage, tuple(ids), threading.current_thread().name, start, end))


def _write(stage: str, clip_id: str, artifact_dir: Path) -> None:
    if stage == "decode":
        write_decode(artifact_dir, random_frames(ROWS[clip_id], seed=int(clip_id[1:])), stride=3)
        return
    prev = artifact_dir / ("decode.npz" if PREV[stage] == "decode" else f"{PREV[stage]}.json")
    digest = hashlib.sha256(prev.read_bytes()).hexdigest()[:16]        # FileNotFoundError if run out of order
    (artifact_dir / f"{stage}.json").write_text(json.dumps({"clip_id": clip_id, "stage": stage, "prev": digest}))


Hook = Callable[[str, str], None]


def make_stage(stage: str, probe: Probe, sleep_s: float = 0.0, fail: Collection[str] = (),
               interrupt: Collection[str] = (), hook: Hook | None = None) -> StageFn:
    """A one-clip fake: `hook(stage, clip)` first, then `KeyboardInterrupt` for `interrupt`, the sleep, a
    `RuntimeError` for `fail`, else the artifact."""
    def fn(clip_id: str, artifact_dir: Path) -> None:
        with probe.call(stage, [clip_id]):
            if hook is not None:
                hook(stage, clip_id)
            if clip_id in interrupt:
                raise KeyboardInterrupt(clip_id)
            if sleep_s:
                time.sleep(sleep_s)
            if clip_id in fail:
                raise RuntimeError(f"fake {stage} failure {clip_id}")
            _write(stage, clip_id, Path(artifact_dir))

    return fn


class ChainBatch:
    """A `BatchedStage` fake: one call per group; a group of more than one clip holding `poison` fails as a unit."""

    def __init__(self, stage: str, probe: Probe, sleep_s: float = 0.0, poison: str | None = None) -> None:
        self.stage, self.probe, self.sleep_s, self.poison = stage, probe, sleep_s, poison

    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        ids = [clip_id for clip_id, _ in clips]
        with self.probe.call(self.stage, ids):
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.poison in ids and len(ids) > 1:
                raise RuntimeError("poison batch")
            for clip_id, artifact_dir in clips:
                _write(self.stage, clip_id, Path(artifact_dir))
        return BatchOutcome()


def build(probe: Probe, sleep_s: float = 0.0, fail: Collection[str] = (), poison: str | None = None,
          interrupt: dict[str, Collection[str]] | None = None, hook: Hook | None = None,
          ) -> dict[str, StageFn | BatchedStage]:
    interrupt = interrupt or {}
    stages: dict[str, StageFn | BatchedStage] = {
        stage: make_stage(stage, probe, sleep_s, fail if stage == "mask" else (), interrupt.get(stage, ()), hook)
        for stage in STAGES}
    stages["geometry"] = ChainBatch("geometry", probe, sleep_s, poison)
    return stages


def config(root: Path, pipeline: bool, depth: int = 2, persist: bool = True) -> BatchedEngineConfig:
    """`persist` False leaves `manifest_path` unset: no parquet save per commit (~20-40 ms each on this host)."""
    engine = EngineConfig(artifact_root=root, stages=STAGES,
                          manifest_path=root / "manifest.parquet" if persist else None)
    return BatchedEngineConfig(engine, {"geometry": StageBatchConfig(max_batch=2)}, max_clips_per_wave=WAVE, run_id="t0",
                               pipeline_waves=pipeline, pipeline_depth=depth, pipeline_resources=RESOURCES)


def new_manifest() -> ClipManifest:
    return ClipManifest.create(IDS, [f"{c}.mp4" for c in IDS], [3 * ROWS[c] for c in IDS])


def run_one(root: Path, pipeline: bool, stages: dict[str, StageFn | BatchedStage], depth: int = 2,
            persist: bool = False) -> tuple[ClipManifest, BatchRunReport]:
    manifest = new_manifest()
    return manifest, run_manifest_batched(manifest, config(root, pipeline, depth, persist), stages)


def run_guarded(fn: Callable[[], Any]) -> dict[str, Any]:
    """`fn()` on a helper thread, failing the test instead of hanging it; returns `result` or `exc`."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            box["exc"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT_S)
    assert not thread.is_alive(), "the run hung"
    return box


def _key(record: BatchRecord) -> tuple[object, ...]:
    return (record.batch_id, record.run_id, record.stage, record.clip_ids, record.weights, record.errors,
            record.retry_of, record.batched, record.peak_gpu_mem_gb)


# --- invariants 1 and 6: identical results, failures and retries as before ---------------------------------------

@pytest.mark.parametrize(("depth", "failures", "persist"), [(1, False, False), (2, False, False), (3, False, False),
                                                        (1, True, False), (2, True, True), (3, True, False)])
def test_pipeline_matches_the_sequential_run(tmp_path: Path, depth: int, failures: bool, persist: bool) -> None:
    kwargs: dict[str, Any] = {"fail": ("c1",), "poison": "c3"} if failures else {}
    runs = {flag: run_one(tmp_path / str(flag), flag, build(Probe(), **kwargs), depth, persist)
            for flag in (False, True)}
    off, on = tmp_path / "False", tmp_path / "True"
    (m_off, _), (m_on, report) = runs[False], runs[True]

    diff = compare_artifact_dirs(off, on)
    assert diff["n_mismatched"] == 0 and diff["missing_in_a"] == [] == diff["missing_in_b"]
    for clip_id in IDS:
        for stage in STAGES:
            a, b = m_off.status(clip_id, stage), m_on.status(clip_id, stage)
            assert (a.state, a.error, a.batch_id) == (b.state, b.error, b.batch_id), (clip_id, stage)
        p_off, p_on = (ProfileReport.from_json(root / clip_id / "profile.json") for root in (off, on))
        assert [(s.name, s.meta) for s in p_off.stages] == [(s.name, s.meta) for s in p_on.stages], clip_id
        for record in p_on.stages:
            assert record.seconds == m_on.status(clip_id, record.name).seconds
    if persist:
        saved = ClipManifest.load(on / "manifest.parquet")
        assert all(saved.status(c, s) == m_on.status(c, s) for c in IDS for s in STAGES)
    records_off, records_on = read_batch_log(off / "batches.jsonl"), read_batch_log(on / "batches.jsonl")
    assert sorted(map(_key, records_off)) == sorted(map(_key, records_on))           # order-insensitive
    assert len(records_on) == len(records_off) == report.n_batches
    assert set(check_apportionment(m_on, records_on).unattributed_ns.values()) <= {0}

    if failures:           # invariant 6: the clip fails alone, the batch-mate is retried singly, other waves go on
        assert m_on.status("c1", "mask").state == "failed" and "fake mask failure c1" in m_on.status("c1", "mask").error
        assert all(m_on.status("c1", s).state == "pending" for s in STAGES[STAGES.index("shape"):])
        geometry = [r for r in records_on if r.stage == "geometry"]
        unit = [r for r in geometry if "c3" in r.clip_ids and len(r.clip_ids) > 1]
        assert len(unit) == 1 and all(e == "poison batch" for e in unit[0].errors)
        assert {r.clip_ids for r in geometry if r.retry_of == unit[0].batch_id} == {("c2",), ("c3",)}
        assert report.n_retries == 2
    done = [c for c in IDS if not (failures and c == "c1")]
    assert all(m_on.status(c, "qc").state == "done" for c in done)


# --- invariants 2, 3 and 4: stage order per wave, one attempt per class, real overlap ------------------------------

def _busy_ns(calls: Sequence[Call]) -> int:
    """Length of the union of the calls' intervals: the time during which at least one attempt was running."""
    busy, end = 0, None
    for start, stop in sorted((c.start_ns, c.end_ns) for c in calls):
        if end is None or start > end:
            busy, end = busy + stop - start, stop
        elif stop > end:
            busy, end = busy + stop - end, stop
    return busy


def _run_wall_ns(report: BatchRunReport) -> int:
    """The run's own wall, as the report accounts for it (`overlapped_ns` is off the critical path)."""
    return sum(report.wall_ns_per_stage.values()) - report.overlapped_ns + report.overhead_ns


MAX_PAIRS = 6       # interleaved (pipelined, sequential) passes for the end-to-end wall claim of invariant 4


def test_order_exclusion_and_overlap(tmp_path: Path) -> None:
    sleep_s = 0.02
    probe, seq_probe = Probe(), Probe()
    t0 = time.perf_counter_ns()
    _, report = run_one(tmp_path / "on0", True, build(probe, sleep_s), depth=2)
    stopwatch_ns = time.perf_counter_ns() - t0
    _, sequential = run_one(tmp_path / "off0", False, build(seq_probe, sleep_s), depth=2)

    by_wave_stage: dict[tuple[int, str], list[Call]] = {}
    for call in probe.calls:
        by_wave_stage.setdefault((WAVE_OF[call.clip_ids[0]], call.stage), []).append(call)
    for k in range(len(WAVES)):        # invariant 2: stage s+1 of a wave starts after stage s of it ended
        for stage, nxt in pairwise(STAGES):
            assert min(c.start_ns for c in by_wave_stage[(k, nxt)]) > max(c.end_ns for c in by_wave_stage[(k, stage)])

    for cls in set(CLASSES.values()):  # invariant 3: never two attempts of one class at once
        assert probe.max_active[cls] == 1, cls
        spans = sorted((c.start_ns, c.end_ns) for c in probe.calls if CLASSES[c.stage] == cls)
        assert all(nxt[0] >= prev[1] for prev, nxt in pairwise(spans)), cls
        assert len({c.thread for c in probe.calls if CLASSES[c.stage] == cls}) == 1, cls

    # invariant 4: calls of different classes did run at once, and the pass took measurably less wall.
    assert probe.max_running >= 2
    gpu = [c for c in probe.calls if CLASSES[c.stage] == GPU_CLASS]
    workers = [c for c in probe.calls if CLASSES[c.stage] in ("sam3d_worker", "hawor_worker")]
    assert any(g.start_ns < w.end_ns and w.start_ns < g.end_ns for g in gpu for w in workers)

    # (a) On the probe's own intervals. The sequential runner runs its attempts back to back, so their timeline is
    # their summed duration; the pipelined one must cover its attempts in at most 85% of theirs. This leaves out the
    # runner's commit time between attempts, which is CPU-bound and which a loaded host stretches arbitrarily
    # (measured under cgroup CPU pressure: up to 0.9 s with NO attempt running, and `overlapped_ns` floored at 0 in
    # 5 of 20 runs), while the sleep-based attempts overlap regardless (busy/summed 0.60-0.73 on the same host).
    attempts_ns = sum(c.end_ns - c.start_ns for c in probe.calls)
    busy_ns = _busy_ns(probe.calls)
    assert seq_probe.max_running == 1
    assert _busy_ns(seq_probe.calls) == sum(c.end_ns - c.start_ns for c in seq_probe.calls)
    assert busy_ns < 0.85 * attempts_ns, (busy_ns / 1e9, attempts_ns / 1e9)

    # (b) The report accounts for it: its wall covers every attempt and no more than the call, and `overlapped_ns` is
    # at least the saving the probe and a stopwatch prove. That bound is derived, not fixed: positive on a quiet host
    # (the attempts outweigh the whole call), vacuous once load stretched the commits past it, which is exactly when
    # the runner's accounting legitimately floors `overlapped_ns` at 0.
    wall_on = _run_wall_ns(report)
    assert busy_ns <= wall_on <= stopwatch_ns
    assert report.overhead_ns >= 0 and report.overlapped_ns >= attempts_ns - stopwatch_ns

    # (c) End to end, pipelined against sequential: this is what catches runner overhead eating the overlap (50 ms of
    # extra hand-off latency per stage fails it while (a) passes). A loaded host only inflates a wall, so each arm's
    # fastest pass estimates its unloaded wall; passes are interleaved so both arms see the same host, and more are
    # added only while the claim is not yet shown. It is NOT the overlap check: an attempt-serialised runner still
    # hides its commits behind other classes' attempts and can pass it on a loaded host; max_running and (a) fail it.
    walls_on, walls_off = [wall_on], [_run_wall_ns(sequential)]
    for k in range(1, MAX_PAIRS):
        if min(walls_on) < 0.85 * min(walls_off):
            break
        walls_on.append(_run_wall_ns(run_one(tmp_path / f"on{k}", True, build(Probe(), sleep_s), depth=2)[1]))
        walls_off.append(_run_wall_ns(run_one(tmp_path / f"off{k}", False, build(Probe(), sleep_s), depth=2)[1]))
    assert min(walls_on) < 0.85 * min(walls_off), ([w / 1e9 for w in walls_on], [w / 1e9 for w in walls_off])


# --- invariant 5: the depth bound ------------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pipeline_depth_bounds_the_waves_in_flight(tmp_path: Path, depth: int) -> None:
    probe = Probe()
    second_wave_started = threading.Event()

    def hook(stage: str, clip_id: str) -> None:
        if stage == "decode" and clip_id == WAVES[1][0]:
            second_wave_started.set()
        if depth >= 2 and stage == "qc" and clip_id == WAVES[0][0]:   # force the overlap the bound allows
            assert second_wave_started.wait(TIMEOUT_S)

    run_one(tmp_path, True, build(probe, hook=hook), depth=depth)
    spans = {k: (min(c.start_ns for c in probe.calls if WAVE_OF[c.clip_ids[0]] == k),
                 max(c.end_ns for c in probe.calls if WAVE_OF[c.clip_ids[0]] == k)) for k in range(len(WAVES))}
    in_flight = max(sum(1 for s, e in spans.values() if s <= t < e) for t, _ in spans.values())
    assert in_flight <= depth
    assert in_flight == min(depth, 2) if depth <= 2 else in_flight >= 2


# --- invariant 7: a BaseException on a class thread -------------------------------------------------------------

def test_base_exception_on_a_class_thread_halts_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hands (hawor_worker thread) raises on wave 0 while shape (sam3d_worker thread) is inside wave 1's first clip:
    shape finishes and commits that attempt, nothing starts afterwards, and the caller gets the exception."""
    probe = Probe()
    shape_inside, halted = threading.Event(), threading.Event()
    halted_at: list[int] = []
    original = PipelinedRunner._stop

    def stop(self: PipelinedRunner, exc: BaseException | None) -> None:
        original(self, exc)
        halted_at.append(time.perf_counter_ns())
        halted.set()

    monkeypatch.setattr(PipelinedRunner, "_stop", stop)
    first_of_wave1 = WAVES[1][0]

    def hook(stage: str, clip_id: str) -> None:
        if stage == "shape" and clip_id == first_of_wave1:
            shape_inside.set()
            assert halted.wait(TIMEOUT_S)          # the attempt in flight when the run halts
        if stage == "hands" and clip_id == "c0":
            assert shape_inside.wait(TIMEOUT_S)

    manifest = new_manifest()
    stages = build(probe, interrupt={"hands": ("c0",)}, hook=hook)
    box = run_guarded(lambda: run_manifest_batched(manifest, config(tmp_path, True, persist=True), stages))

    assert isinstance(box.get("exc"), KeyboardInterrupt) and "result" not in box
    assert all(call.start_ns < halted_at[0] for call in probe.calls)          # no attempt started after the halt
    assert manifest.status(first_of_wave1, "shape").state == "done"          # the attempt in flight committed
    assert manifest.status("c0", "hands").state == "pending"                 # the raising attempt committed nothing
    records = read_batch_log(tmp_path / "batches.jsonl")
    assert not [r for r in records if r.stage == "hands"]
    assert [r.clip_ids for r in records if r.stage == "shape"] == [("c0",), ("c1",), (first_of_wave1,)]
    called = {(c.stage, c.clip_ids[0]) for c in probe.calls}
    assert ("shape", WAVES[1][1]) not in called and ("hands", "c1") not in called
    assert not [c for c in probe.calls if c.stage in ("geometry", "qc") or WAVE_OF[c.clip_ids[0]] >= 2]
    assert ClipManifest.load(tmp_path / "manifest.parquet").status(first_of_wave1, "shape").state == "done"


# --- GPU lock: held once per run, and a stage that nests it on a class thread does not deadlock ---------------------

def test_nested_gpu_lock_on_a_class_thread_passes_through(tmp_path: Path) -> None:
    probe = Probe()
    lock_path = tmp_path / ".gpu.lock"
    inner = make_stage("hands", probe)

    def hands(clip_id: str, artifact_dir: Path) -> None:        # as `run_hands_call` does inside HandsBatchStage
        with gpu_lock(lock_path):
            inner(clip_id, artifact_dir)

    stages = build(probe) | {"hands": hands}
    manifest = new_manifest()
    box = run_guarded(lambda: run_manifest_batched(manifest, config(tmp_path, True, persist=False), stages))
    assert "exc" not in box
    assert all(manifest.status(c, "qc").state == "done" for c in IDS)
    assert {c.thread for c in probe.calls if c.stage == "hands"} == {"fastdo-pipeline-hawor_worker"}
    with gpu_lock(lock_path, blocking=False):           # released after the run
        pass


def test_hold_gpu_lock_for_process_is_shared_by_threads_and_released(tmp_path: Path) -> None:
    path = tmp_path / ".gpu.lock"
    entered = threading.Event()

    def other_thread() -> None:
        with gpu_lock(path, blocking=False):
            entered.set()

    with hold_gpu_lock_for_process(path):
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(TIMEOUT_S)
        assert entered.is_set()
        with pytest.raises(RuntimeError, match="already held"), hold_gpu_lock_for_process(path):
            pass
    blocked: list[BaseException] = []
    holder_in, release = threading.Event(), threading.Event()

    def holder() -> None:
        with gpu_lock(path):
            holder_in.set()
            release.wait(TIMEOUT_S)

    thread = threading.Thread(target=holder)
    thread.start()
    assert holder_in.wait(TIMEOUT_S)
    try:                          # after the hold, another thread's flock excludes this one again
        with gpu_lock(path, blocking=False):
            pass
    except BlockingIOError as exc:
        blocked.append(exc)
    finally:
        release.set()
        thread.join(TIMEOUT_S)
    assert blocked


# --- invariant 8: the default leaves the sequential path alone ----------------------------------------------------

def test_default_config_never_builds_the_pipelined_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BatchedEngineConfig(EngineConfig(artifact_root=tmp_path))
    assert (cfg.pipeline_waves, cfg.pipeline_depth, dict(cfg.pipeline_resources)) == (False, 2, {})

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("the pipelined runner was built with pipeline_waves off")

    monkeypatch.setattr(PipelinedRunner, "__init__", refuse)
    probe = Probe()
    manifest = new_manifest()
    run_manifest_batched(manifest, config(tmp_path, False, persist=False), build(probe))
    assert all(manifest.status(c, "qc").state == "done" for c in IDS)
    assert {c.thread for c in probe.calls} == {threading.current_thread().name}


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_pipeline_depth_must_be_a_positive_integer(tmp_path: Path, bad: Any) -> None:
    with pytest.raises(ValueError, match="pipeline_depth"):
        BatchedEngineConfig(EngineConfig(artifact_root=tmp_path), pipeline_depth=bad)
    with pytest.raises(ValueError, match="pipeline_resources"):
        BatchedEngineConfig(EngineConfig(artifact_root=tmp_path), pipeline_resources={"shape": ""})


def test_prefetch_decode_is_subsumed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    probe = Probe()
    manifest = new_manifest()
    cfg = BatchedEngineConfig(config(tmp_path, True, persist=False).engine, {"geometry": StageBatchConfig(max_batch=2)},
                              max_clips_per_wave=WAVE, run_id="t0", prefetch_decode=True, pipeline_waves=True,
                              pipeline_resources=RESOURCES)
    with caplog.at_level("INFO", logger="fastdo.engine.pipelined_runner"):
        run_manifest_batched(manifest, cfg, build(probe))
    assert any("subsumes prefetch_decode" in r.getMessage() for r in caplog.records)
    assert {c.thread for c in probe.calls if c.stage == "decode"} == {"fastdo-pipeline-decode"}
    assert all(manifest.status(c, "qc").state == "done" for c in IDS)
