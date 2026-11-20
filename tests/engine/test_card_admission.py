"""Tests for card-level admission and the patient CUDA-OOM retry (`fastdo.engine.card_admission`) in both
batched runners, their `batches.jsonl` wait keys, and the `bench.batched` config blocks.

The card is a fake: `card_admission.card_free_gb` reads a `FakeCard` whose free memory is a function of a fake clock,
and `card_admission.wait_ns` advances that clock instead of sleeping, so every wait is exact and nothing sleeps. The
stages are CPU-only StageFns (decode -> mask -> shape -> solve, mask batched by 2) that write a JSON file each; shape
runs in the `sam3d_worker` class, as in production, and raises the SAM 3D worker's OOM message while the fake
card is short.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fastdo.bench import gpu_budget
from fastdo.bench.bench_batched import batched_engine_config
from fastdo.engine import apportion, card_admission
from fastdo.engine.apportion import (
    AdmissionBatchRecord,
    BatchRecord,
    LedgerBatchRecord,
    check_apportionment,
    read_batch_log,
)
from fastdo.engine.batched_runner import (
    BatchedEngineConfig,
    CardAdmissionConfig,
    MemoryLedgerConfig,
    OomRetryConfig,
    StageBatchConfig,
    run_manifest_batched,
)
from fastdo.engine.batched_stages import BatchOutcome
from fastdo.engine.card_admission import is_cuda_oom
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, StageFn
from tests.engine.fakes import random_frames, write_decode

_REPO = Path(__file__).resolve().parents[2]
TIMEOUT_S = 30.0
ROWS = {"c0": 4, "c1": 7, "c2": 5, "c3": 6}
IDS = list(ROWS)
STAGES = ("decode", "mask", "shape", "solve")
RESOURCES = {"shape": "sam3d_worker"}
BATCHES = {"mask": StageBatchConfig(max_batch=2, budget_gb=4.0, fixed_cost_gb=2.0)}
FIELDS = [f.name for f in fields(BatchRecord)]

# Verbatim (truncated) from a production batches.jsonl: a worker reply, an in-process OOM and
# SAM 3D's sparse-conv allocation failure.
SAM3D_OOM = ("SAM 3D worker failed on request 20: OutOfMemoryError: CUDA out of memory. Tried to allocate 196.00 MiB. "
             "GPU 0 has a total capacity of 79.19 GiB of which 134.25 MiB is free. Process 2637243 has 35.34 GiB")
SOLVE_OOM = ("CUDA out of memory. Tried to allocate 1.91 GiB. GPU 0 has a total capacity of 79.19 GiB of which "
             "1.89 GiB is free. Process 2637243 has 35.34 GiB memory in use. 32.00 GiB allowed; Of the allocated")
SPCONV_OOM = "SAM 3D worker failed on request 22: RuntimeError: Cuda error: 2[cudaMalloc(&m_gpuPtr, bytes);]"


@pytest.fixture(autouse=True)
def _cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gpu_budget.ENV_VAR, "32")


class FakeCard:
    """Free card GiB as a function of fake seconds; `wait_ns` advances the clock. Records every reading."""

    def __init__(self, free_at: Callable[[float], float]) -> None:
        self.free_at = free_at
        self.t = 0.0
        self.lock = threading.Lock()
        self.reads: list[tuple[float, bool]] = []
        self.sleeps: list[float] = []

    def free(self, in_process: bool) -> float:
        with self.lock:
            self.reads.append((self.t, in_process))
            return self.free_at(self.t)

    def now_free(self) -> float:
        with self.lock:
            return self.free_at(self.t)

    def sleep(self, wake: threading.Event, seconds: float) -> int:
        with self.lock:
            self.t += seconds
            self.sleeps.append(seconds)
        return round(seconds * 1e9)


@pytest.fixture
def card(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable[[float], float]], FakeCard]:
    def install(free_at: Callable[[float], float]) -> FakeCard:
        fake = FakeCard(free_at)
        monkeypatch.setattr(card_admission, "card_free_gb", fake.free)
        monkeypatch.setattr(card_admission, "wait_ns", fake.sleep)
        return fake
    return install


class Calls:
    """Every stage call in start order, with the fake card's free GiB at the call."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.log: list[tuple[str, tuple[str, ...], float]] = []

    def add(self, stage: str, ids: Sequence[str], free: float) -> None:
        with self.lock:
            self.log.append((stage, tuple(ids), free))

    def of(self, stage: str) -> list[tuple[tuple[str, ...], float]]:
        return [(ids, free) for s, ids, free in self.log if s == stage]


Fail = Callable[[str, Sequence[str], float], str | None]


def _write(stage: str, clip_id: str, artifact_dir: Path) -> None:
    if stage == "decode":
        write_decode(artifact_dir, random_frames(ROWS[clip_id], seed=int(clip_id[1:])), stride=3)
    else:
        (artifact_dir / f"{stage}.json").write_text(json.dumps({"clip_id": clip_id, "stage": stage}))


def build(calls: Calls, fake: FakeCard | None, fail: Fail | None = None) -> dict[str, Any]:
    """StageFns for every stage and a batched mask; `fail(stage, ids, free)` returns the message to raise, or None."""
    def free() -> float:
        return math.inf if fake is None else fake.now_free()

    def fn(stage: str) -> StageFn:
        def call(clip_id: str, artifact_dir: Path) -> None:
            now = free()
            calls.add(stage, [clip_id], now)
            if fail is not None and (message := fail(stage, [clip_id], now)):
                raise RuntimeError(message)
            _write(stage, clip_id, Path(artifact_dir))
        return call

    class Mask:
        def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
            ids = [c for c, _ in clips]
            now = free()
            calls.add("mask", ids, now)
            if fail is not None and (message := fail("mask", ids, now)):
                raise RuntimeError(message)
            for clip_id, artifact_dir in clips:
                _write("mask", clip_id, Path(artifact_dir))
            return BatchOutcome()

    stages: dict[str, Any] = {s: fn(s) for s in STAGES}
    stages["mask"] = Mask()
    return stages


def config(root: Path, admission: CardAdmissionConfig | None = None, retry: OomRetryConfig | None = None,
           pipelined: bool = True, ledger: MemoryLedgerConfig | None = None) -> BatchedEngineConfig:
    return BatchedEngineConfig(EngineConfig(artifact_root=root, stages=STAGES), BATCHES, max_clips_per_wave=2,
                               run_id="t0", pipeline_waves=pipelined, pipeline_depth=2, pipeline_resources=RESOURCES,
                               memory_ledger=ledger, card_admission=admission, oom_retry=retry)


def new_manifest() -> ClipManifest:
    return ClipManifest.create(IDS, [f"{c}.mp4" for c in IDS], [3 * ROWS[c] for c in IDS])


def run(manifest: ClipManifest, cfg: BatchedEngineConfig, stages: dict[str, Any]) -> dict[str, Any]:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["report"] = run_manifest_batched(manifest, cfg, stages)
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            box["exc"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT_S)
    assert not thread.is_alive(), "the run hung"
    return box


def lines(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "batches.jsonl").read_text().splitlines()]


def rows_of(root: Path, stage: str) -> list[dict[str, Any]]:
    return [row for row in lines(root) if row["stage"] == stage]


# --- units ----------------------------------------------------------------------------------------------------------

def test_oom_classifier_on_the_214_messages() -> None:
    for message in (SAM3D_OOM, SOLVE_OOM, SPCONV_OOM, "torch.OutOfMemoryError: boom", "CUDA error: out of memory",
                    "HaWoR driver exited with code 1: ... CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"):
        assert is_cuda_oom(message), message
    for message in ("", "poison batch", "prompt refused: c0:k000: no confident hand-object contact",
                    "gate rejected: CUDA out of memory quoted in a verdict",
                    "solve needs 3.10 GiB but the CUDA budget has 1.20 GiB left (cap 32.00 GiB from apply_gpu_budget)",
                    "Cuda error: 700[cudaMemcpy]"):
        assert not is_cuda_oom(message), message


def test_backoff_schedule_doubles_to_its_cap() -> None:
    policy = OomRetryConfig(max_total_wait_s=100.0, backoff_s=(1.5, 10.0))
    assert [policy.delay_s(k) for k in range(6)] == [1.5, 3.0, 6.0, 10.0, 10.0, 10.0]
    assert policy.delay_s(10_000) == 10.0


def test_wait_ns_is_cut_short_by_the_wake_event() -> None:
    wake = threading.Event()
    wake.set()
    start = time.monotonic()
    assert card_admission.wait_ns(wake, 30.0) >= 0
    assert time.monotonic() - start < 5.0


def test_card_free_gb_is_none_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)    # holds on a GPU host too
    assert card_admission.card_free_gb(True) is None and card_admission.card_free_gb(False) is None


def test_records_round_trip_with_their_wait_keys(tmp_path: Path) -> None:
    base = BatchRecord("b0", "r", "shape", ("c0",), (4,), 10, (10,), ("",), "", False, 0.0, "2026-09-26T00:00:00Z")
    ledger = LedgerBatchRecord.of(replace(base, batch_id="b1"), 7)
    both = AdmissionBatchRecord.with_waits(replace(ledger, batch_id="b2"), card_wait_ns=5)
    both = AdmissionBatchRecord.with_waits(both, oom_backoff_ns=3, card_wait_ns=None)    # None keeps the card wait
    only_backoff = AdmissionBatchRecord.with_waits(replace(base, batch_id="b3"), oom_backoff_ns=0)
    records: list[BatchRecord] = [base, ledger, both, only_backoff]
    assert (both.ledger_wait_ns, both.card_wait_ns, both.oom_backoff_ns) == (7, 5, 3)
    log = tmp_path / "batches.jsonl"
    apportion.append_batch_records(log, records)
    keys = [list(json.loads(line)) for line in log.read_text().splitlines()]
    assert keys == [FIELDS, [*FIELDS, "ledger_wait_ns"], [*FIELDS, "ledger_wait_ns", "card_wait_ns", "oom_backoff_ns"],
                    [*FIELDS, "oom_backoff_ns"]]
    assert read_batch_log(log) == records
    assert [type(r) for r in read_batch_log(log)] == [BatchRecord, LedgerBatchRecord, AdmissionBatchRecord,
                                                      AdmissionBatchRecord]
    with pytest.raises(TypeError, match="unknown wait keys"):
        AdmissionBatchRecord.with_waits(base, bogus_ns=1)


# --- card admission -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pipelined", [True, False])
def test_admission_waits_then_proceeds(tmp_path: Path, card: Callable[..., FakeCard], pipelined: bool) -> None:
    """The card has 1 GiB free until t = 1.5 s, then 10. shape (4 GiB, a worker class: card-free reading) and mask (2
    GiB via its class `gpu`, in-process reading) wait in 0.5 s polls; no call starts short; the rows carry the waits."""
    fake = card(lambda t: 1.0 if t < 1.5 else 10.0)
    calls = Calls()
    admission = CardAdmissionConfig(min_free_gb={"shape": 4.0, "gpu": 2.0}, poll_s=0.5, max_wait_s=60.0)
    manifest = new_manifest()
    box = run(manifest, config(tmp_path, admission, pipelined=pipelined), build(calls, fake))
    assert "exc" not in box, box.get("exc")
    assert all(manifest.status(c, "solve").state == "done" for c in IDS)
    for stage, need in (("shape", 4.0), ("mask", 2.0), ("solve", 2.0)):
        assert calls.of(stage) and all(free >= need for _, free in calls.of(stage)), stage
    assert calls.of("decode")[0][1] == 1.0                   # decode is not gated: it ran on the short card
    assert {inp for _, inp in fake.reads} == {True, False}
    waits = {row["batch_id"]: row.get("card_wait_ns") for row in lines(tmp_path)}
    assert sum(w or 0 for w in waits.values()) == 3 * 500_000_000   # 3 polls in all, whoever took them
    for row in lines(tmp_path):
        gated = row["stage"] in ("mask", "shape", "solve")
        assert list(row) == FIELDS + (["card_wait_ns"] if gated else []), row
    assert rows_of(tmp_path, "mask")[0]["card_wait_ns"] == 1_500_000_000   # the first gated attempt took all 3
    check_apportionment(manifest, read_batch_log(tmp_path / "batches.jsonl"))


def test_admission_gives_up_after_max_wait(tmp_path: Path, card: Callable[..., FakeCard],
                                           caplog: pytest.LogCaptureFixture) -> None:
    """The card never frees: every shape attempt waits exactly max_wait_s (4 polls of 0.5 s), then runs anyway."""
    fake = card(lambda t: 1.0)
    calls = Calls()
    admission = CardAdmissionConfig(min_free_gb={"sam3d_worker": 4.0}, poll_s=0.5, max_wait_s=2.0)
    manifest = new_manifest()
    with caplog.at_level(logging.WARNING, logger="fastdo.engine.card_admission"):
        box = run(manifest, config(tmp_path, admission), build(calls, fake))
    assert "exc" not in box, box.get("exc")
    assert [ids for ids, _ in calls.of("shape")] == [(c,) for c in IDS]
    assert all(manifest.status(c, "solve").state == "done" for c in IDS)
    assert [row["card_wait_ns"] for row in rows_of(tmp_path, "shape")] == [2_000_000_000] * len(IDS)
    assert all("card_wait_ns" not in row for row in lines(tmp_path) if row["stage"] != "shape")
    assert fake.sleeps == [0.5] * 4 * len(IDS) and {inp for _, inp in fake.reads} == {False}
    assert sum("dispatching anyway" in r.getMessage() for r in caplog.records) == len(IDS)


class FakeCache:
    """`release_cached_gb` stand-in: returns `gib` once, and from then on the fake card has that much more
    free; later calls return 0.0, as an emptied cache does."""

    def __init__(self, fake: FakeCard, gib: float) -> None:
        self.gib = gib
        self.calls = 0
        base = fake.free_at
        fake.free_at = lambda t: base(t) + (gib if self.calls else 0.0)

    def release(self) -> float:
        self.calls += 1
        return self.gib if self.calls == 1 else 0.0


def test_a_worker_wait_reclaims_this_process_cache_first(tmp_path: Path, card: Callable[..., FakeCard],
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         caplog: pytest.LogCaptureFixture) -> None:
    """The card never frees by itself (1 GiB) but this process holds 5 GiB of cache. The first short shape
    reading releases it, re-reads 6 GiB and admits at once: no poll sleeps, no dispatch-anyway, 0 ns waits."""
    fake = card(lambda t: 1.0)
    cache = FakeCache(fake, 5.0)
    monkeypatch.setattr(card_admission, "release_cached_gb", cache.release)
    calls = Calls()
    admission = CardAdmissionConfig(min_free_gb={"sam3d_worker": 4.0}, poll_s=0.5, max_wait_s=2.0)
    manifest = new_manifest()
    with caplog.at_level(logging.INFO, logger="fastdo.engine.card_admission"):
        box = run(manifest, config(tmp_path, admission), build(calls, fake))
    assert "exc" not in box, box.get("exc")
    assert all(free >= 4.0 for _, free in calls.of("shape")) and len(calls.of("shape")) == len(IDS)
    assert fake.sleeps == [] and cache.calls == 1
    assert [row["card_wait_ns"] for row in rows_of(tmp_path, "shape")] == [0] * len(IDS)
    assert sum("released 5.00 GiB" in r.getMessage() for r in caplog.records) == 1
    assert not any("dispatching anyway" in r.getMessage() for r in caplog.records)


def test_in_process_waits_and_reclaim_off_never_empty_the_cache(tmp_path: Path, card: Callable[..., FakeCard],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-process reading already counts the cache, so a short mask (class `gpu`) wait never reclaims; with
    `reclaim_cache` off a short worker wait does not either. Both then wait exactly as plain admission does."""
    fake = card(lambda t: 1.0 if t < 1.0 else 10.0)
    cache = FakeCache(fake, 5.0)
    monkeypatch.setattr(card_admission, "release_cached_gb", cache.release)
    admission = CardAdmissionConfig(min_free_gb={"gpu": 2.0}, poll_s=0.5, max_wait_s=60.0)
    box = run(new_manifest(), config(tmp_path / "a", admission), build(Calls(), fake))
    assert "exc" not in box, box.get("exc")
    assert cache.calls == 0 and fake.sleeps == [0.5, 0.5]

    fake.t, fake.sleeps = 0.0, []
    off = CardAdmissionConfig(min_free_gb={"sam3d_worker": 4.0}, poll_s=0.5, max_wait_s=60.0, reclaim_cache=False)
    box = run(new_manifest(), config(tmp_path / "b", off), build(Calls(), fake))
    assert "exc" not in box, box.get("exc")
    assert cache.calls == 0 and fake.sleeps == [0.5, 0.5]


def test_release_cached_gb_is_none_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)    # holds on a GPU host too
    assert card_admission.release_cached_gb() is None


def test_halt_ends_an_admission_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real clock: shape(wave 0) polls a card that never frees (max_wait_s 30 s) while mask(wave 1) raises
    KeyboardInterrupt. The halt wakes the poll at once; shape never starts and the caller gets the interrupt."""
    monkeypatch.setattr(card_admission, "card_free_gb", lambda in_process: 1.0)
    calls = Calls()
    polling = threading.Event()
    real_wait = card_admission.wait_ns

    def wait(wake: threading.Event, seconds: float) -> int:
        polling.set()
        return real_wait(wake, seconds)

    monkeypatch.setattr(card_admission, "wait_ns", wait)

    def fail(stage: str, ids: Sequence[str], free: float) -> str | None:
        if stage == "mask" and "c2" in ids:
            assert polling.wait(TIMEOUT_S)
            raise KeyboardInterrupt
        return None

    admission = CardAdmissionConfig(min_free_gb={"shape": 4.0}, poll_s=0.05, max_wait_s=30.0)
    start = time.monotonic()
    box = run(new_manifest(), config(tmp_path, admission), build(calls, None, fail))
    assert isinstance(box.get("exc"), KeyboardInterrupt)
    assert time.monotonic() - start < 10.0
    assert calls.of("shape") == []


# --- patient OOM retry ----------------------------------------------------------------------------------------------

def _short(need: float, message: str) -> Fail:
    """Raises `message` from shape while the fake card has less than `need` GiB free."""
    def fail(stage: str, ids: Sequence[str], free: float) -> str | None:
        return message if stage == "shape" and free < need else None
    return fail


@pytest.mark.parametrize("pipelined", [True, False])
def test_oom_retry_backs_off_and_succeeds_when_memory_frees(tmp_path: Path, card: Callable[..., FakeCard],
                                                            pipelined: bool) -> None:
    """shape OOMs while the card is short (until t = 6 s). c0's one-clip group, which today would be lost for good,
    retries alone after 1, 2 and 4 s and succeeds at t = 7; every later shape attempt runs on the freed card."""
    fake = card(lambda t: 1.0 if t < 6.0 else 10.0)
    calls = Calls()
    retry = OomRetryConfig(max_total_wait_s=100.0, backoff_s=(1.0, 8.0))
    manifest = new_manifest()
    box = run(manifest, config(tmp_path, retry=retry, pipelined=pipelined), build(calls, fake, _short(4.0, SAM3D_OOM)))
    assert "exc" not in box, box.get("exc")
    assert all(manifest.status(c, "solve").state == "done" for c in IDS)
    assert fake.sleeps == [1.0, 2.0, 4.0]
    shape = rows_of(tmp_path, "shape")
    c0 = [row for row in shape if row["clip_ids"] == ["c0"]]
    assert [row["oom_backoff_ns"] for row in c0] == [0, 1_000_000_000, 2_000_000_000, 4_000_000_000]
    assert [bool(row["errors"][0]) for row in c0] == [True, True, True, False]
    assert [row["retry_of"] for row in c0] == ["", c0[0]["batch_id"], c0[1]["batch_id"], c0[2]["batch_id"]]
    assert manifest.status("c0", "shape").batch_id == c0[-1]["batch_id"]
    for row in lines(tmp_path):                             # every row carries the key while oom_retry is on
        assert list(row) == [*FIELDS, "oom_backoff_ns"]
    check = check_apportionment(manifest, read_batch_log(tmp_path / "batches.jsonl"))
    assert check.unattributed_ns.get("shape", 0) == 0     # the chain is walked whole: every attempt is attributed


def test_oom_retry_stops_at_the_budget(tmp_path: Path, card: Callable[..., FakeCard],
                                       caplog: pytest.LogCaptureFixture) -> None:
    """The card never frees: c0 backs off 1, 2, then the 2 s left of its 5 s budget, and fails for good; the clips
    after it do the same, and nothing downstream of shape runs."""
    fake = card(lambda t: 1.0)
    calls = Calls()
    retry = OomRetryConfig(max_total_wait_s=5.0, backoff_s=(1.0, 4.0))
    manifest = new_manifest()
    with caplog.at_level(logging.WARNING, logger="fastdo.engine.batched_runner"):
        box = run(manifest, config(tmp_path, retry=retry), build(calls, fake, _short(4.0, SPCONV_OOM)))
    assert "exc" not in box, box.get("exc")
    assert fake.sleeps == [1.0, 2.0, 2.0] * len(IDS)
    for clip_id in IDS:
        status = manifest.status(clip_id, "shape")
        assert status.state == "failed" and is_cuda_oom(status.error)
        assert manifest.status(clip_id, "solve").state != "done"
        chain = [row for row in rows_of(tmp_path, "shape") if row["clip_ids"] == [clip_id]]
        assert [row["oom_backoff_ns"] for row in chain] == [0, 1_000_000_000, 2_000_000_000, 2_000_000_000]
        assert status.batch_id == chain[-1]["batch_id"]
    assert calls.of("solve") == []
    assert sum("giving up" in r.getMessage() for r in caplog.records) == len(IDS)
    check_apportionment(manifest, read_batch_log(tmp_path / "batches.jsonl"))


def test_each_retry_is_readmitted_and_admission_waits_count_toward_the_budget(
        tmp_path: Path, card: Callable[..., FakeCard]) -> None:
    """Admission (max_wait_s 3) and retry (budget 5, backoff 1) together on a card that never frees: each shape
    attempt waits 3 s for the card first; c0's chain is first (3) + retry (1 + 3 = 4 waited) + retry (1 + 3 = 8 >= 5):
    three attempts, each re-admitted."""
    fake = card(lambda t: 1.0)
    admission = CardAdmissionConfig(min_free_gb={"shape": 4.0}, poll_s=1.0, max_wait_s=3.0)
    retry = OomRetryConfig(max_total_wait_s=5.0, backoff_s=(1.0, 1.0))
    manifest = new_manifest()
    box = run(manifest, config(tmp_path, admission, retry), build(Calls(), fake, _short(4.0, SAM3D_OOM)))
    assert "exc" not in box, box.get("exc")
    chain = [row for row in rows_of(tmp_path, "shape") if row["clip_ids"] == ["c0"]]
    assert [(row["card_wait_ns"], row["oom_backoff_ns"]) for row in chain] == [
        (3_000_000_000, 0), (3_000_000_000, 1_000_000_000), (3_000_000_000, 1_000_000_000)]
    assert list(chain[0]) == [*FIELDS, "card_wait_ns", "oom_backoff_ns"]
    assert manifest.status("c0", "shape").state == "failed"


def test_a_group_oom_retries_singly_at_once_then_backs_off(tmp_path: Path, card: Callable[..., FakeCard]) -> None:
    """mask (batched by 2) OOMs as a group until t = 1: its single retries start at once, as today (a smaller
    workload); c0's single retry OOMs too (t = 0) and backs off 1 s before it succeeds; c1 then runs on the freed
    card."""
    fake = card(lambda t: 1.0 if t < 1.0 else 10.0)

    def fail(stage: str, ids: Sequence[str], free: float) -> str | None:
        return SOLVE_OOM if stage == "mask" and free < 4.0 else None

    manifest = new_manifest()
    retry = OomRetryConfig(max_total_wait_s=10.0, backoff_s=(1.0, 1.0))
    box = run(manifest, config(tmp_path, retry=retry, pipelined=False), build(Calls(), fake, fail))
    assert "exc" not in box, box.get("exc")
    mask = rows_of(tmp_path, "mask")
    first = [(row["clip_ids"], bool(row["errors"][0]), row["oom_backoff_ns"]) for row in mask[:4]]
    assert first == [(["c0", "c1"], True, 0), (["c0"], True, 0), (["c0"], False, 1_000_000_000), (["c1"], False, 0)]
    assert mask[1]["retry_of"] == mask[0]["batch_id"] and mask[2]["retry_of"] == mask[1]["batch_id"]
    assert mask[3]["retry_of"] == mask[0]["batch_id"]
    assert all(manifest.status(c, "solve").state == "done" for c in IDS)
    check_apportionment(manifest, read_batch_log(tmp_path / "batches.jsonl"))


@pytest.mark.parametrize("pipelined", [True, False])
def test_non_oom_errors_keep_todays_handling(tmp_path: Path, card: Callable[..., FakeCard], pipelined: bool) -> None:
    """A non-OOM failure of a one-clip group is final at once, and a failed group gets one immediate single retry
    per clip: with oom_retry on, the same batch ids, groups, retries and errors as with it off; no backoff."""
    def fail(stage: str, ids: Sequence[str], free: float) -> str | None:
        if stage == "shape" and ids == ["c1"]:
            return "poison clip"
        if stage == "mask" and ("c2" in ids):
            return "poison batch"
        return None

    retry = OomRetryConfig(max_total_wait_s=100.0, backoff_s=(1.0, 8.0))
    runs = {}
    for flag, policy in (("off", None), ("on", retry)):
        fake = card(lambda t: 10.0)
        manifest = new_manifest()
        box = run(manifest, config(tmp_path / flag, retry=policy, pipelined=pipelined), build(Calls(), fake, fail))
        assert "exc" not in box, box.get("exc")
        assert fake.sleeps == []
        runs[flag] = (manifest, lines(tmp_path / flag))

    def key(row: dict[str, Any]) -> tuple[object, ...]:
        return row["batch_id"], tuple(row["clip_ids"]), tuple(row["errors"]), row["retry_of"]

    assert sorted(map(key, runs["off"][1])) == sorted(map(key, runs["on"][1]))
    assert all(row["oom_backoff_ns"] == 0 for row in runs["on"][1])
    for clip_id in IDS:
        for stage in STAGES:
            a, b = runs["off"][0].status(clip_id, stage), runs["on"][0].status(clip_id, stage)
            assert (a.state, a.batch_id, a.error) == (b.state, b.batch_id, b.error), (clip_id, stage)
    assert runs["on"][0].status("c1", "shape").state == "failed"
    assert runs["on"][0].status("c2", "mask").state == "failed"


def test_ledger_gated_rows_carry_all_three_waits_in_order(tmp_path: Path, card: Callable[..., FakeCard]) -> None:
    fake = card(lambda t: 10.0)
    ledger = MemoryLedgerConfig(resident_gb=2.0)
    admission = CardAdmissionConfig(min_free_gb={"mask": 1.0, "shape": 1.0})
    retry = OomRetryConfig(max_total_wait_s=10.0, backoff_s=(1.0, 1.0))
    manifest = new_manifest()
    box = run(manifest, config(tmp_path, admission, retry, ledger=ledger), build(Calls(), fake))
    assert "exc" not in box, box.get("exc")
    expected = {"decode": [*FIELDS, "oom_backoff_ns"],
                "mask": [*FIELDS, "ledger_wait_ns", "card_wait_ns", "oom_backoff_ns"],
                "shape": [*FIELDS, "card_wait_ns", "oom_backoff_ns"],
                "solve": [*FIELDS, "ledger_wait_ns", "oom_backoff_ns"]}
    for row in lines(tmp_path):
        assert list(row) == expected[row["stage"]], row
    assert all(isinstance(r, AdmissionBatchRecord) for r in read_batch_log(tmp_path / "batches.jsonl"))


# --- default off ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pipelined", [True, False])
def test_default_off_touches_no_new_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                pipelined: bool) -> None:
    """With both keys None every admission entry point raises if called; an OOM on a one-clip group is final at once
    and a group's OOM clips get today's single retries; every row has exactly today's keys."""
    def boom(*args: object, **kwargs: object) -> Any:
        raise AssertionError("a card-admission entry point ran with the feature off")

    for name in ("card_free_gb", "wait_ns", "wait_for_card", "is_cuda_oom"):
        monkeypatch.setattr(card_admission, name, boom)
    monkeypatch.setattr(AdmissionBatchRecord, "with_waits", classmethod(boom))

    def fail(stage: str, ids: Sequence[str], free: float) -> str | None:
        if stage == "shape" and ids == ["c0"]:
            return SAM3D_OOM
        if stage == "mask" and "c2" in ids and len(ids) > 1:
            return SOLVE_OOM
        return None

    manifest = new_manifest()
    cfg = config(tmp_path, pipelined=pipelined)
    assert cfg.card_admission is None and cfg.oom_retry is None
    assert BatchedEngineConfig(EngineConfig(artifact_root=tmp_path)).card_admission is None
    assert BatchedEngineConfig(EngineConfig(artifact_root=tmp_path)).oom_retry is None
    calls = Calls()
    box = run(manifest, cfg, build(calls, None, fail))
    assert "exc" not in box, box.get("exc")
    assert [ids for ids, _ in calls.of("shape")].count(("c0",)) == 1        # lost for good, as today
    assert manifest.status("c0", "shape").state == "failed"
    mask = rows_of(tmp_path, "mask")
    assert [(r["clip_ids"], r["retry_of"] != "") for r in mask] == [
        (["c0", "c1"], False), (["c2", "c3"], False), (["c2"], True), (["c3"], True)]
    for row in lines(tmp_path):
        assert list(row) == FIELDS
    assert all(type(r) is BatchRecord for r in read_batch_log(tmp_path / "batches.jsonl"))


# --- config ---------------------------------------------------------------------------------------------------------

_BLOCKS = ("bench.batched.card_admission=", "bench.batched.oom_retry=")


def _cfg(*overrides: str) -> DictConfig:
    """Composes run/conf with `overrides`. Both blocks default to a mapping, and a Hydra mapping override
    MERGES into it; a block override here REPLACES it (null first, as run/conf documents), which is what these tests
    specify."""
    replaced = [x for o in overrides for x in ((o.split("=", 1)[0] + "=null", o)
                                               if o.startswith(_BLOCKS) and not o.endswith("=null") else (o,))]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[f"paths.root={_REPO}", *replaced])


def test_config_blocks_parse_and_refuse(tmp_path: Path) -> None:
    engine = EngineConfig(artifact_root=tmp_path)
    prod = batched_engine_config(_cfg().bench, engine)                              # production: ON
    assert prod.card_admission == CardAdmissionConfig(
        min_free_gb={"solve": 8.5, "tracks": 8.0, "mask": 4.5, "geometry": 3.0, "intrinsics": 3.0, "sam3d_worker": 4.0,
                     "hawor_worker": 4.0}, poll_s=5.0, max_wait_s=300.0)
    assert prod.card_admission.reclaim_cache is True
    assert prod.oom_retry == OomRetryConfig(max_total_wait_s=900.0, backoff_s=(10.0, 120.0))
    off = batched_engine_config(_cfg("bench.batched.card_admission=null", "bench.batched.oom_retry=null").bench, engine)
    assert off.card_admission is None and off.oom_retry is None                       # null restores the old path
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):   # a plain override MERGES
        merged = compose(config_name="config", overrides=[f"paths.root={_REPO}",
                                                          "bench.batched.card_admission={min_free_gb:{solve:1}}"])
    one = batched_engine_config(merged.bench, engine).card_admission
    assert one is not None and one.threshold_gb("solve", "solve") == 1.0 and one.threshold_gb("tracks", "tracks") == 8.0

    on = batched_engine_config(_cfg(
        "bench.batched.card_admission={min_free_gb:{shape:4,sam3d_worker:3,geom:3.0,solve:8.5},poll_s:2,max_wait_s:0}",
        "bench.batched.oom_retry={max_total_wait_s:900,backoff_s:[10,120]}").bench, engine)
    assert on.card_admission == CardAdmissionConfig(
        min_free_gb={"shape": 4.0, "sam3d_worker": 3.0, "geom": 3.0, "solve": 8.5}, poll_s=2.0, max_wait_s=0.0)
    assert on.card_admission.threshold_gb("shape", "sam3d_worker") == 4.0         # the stage key wins
    assert on.card_admission.threshold_gb("geometry", "geom") == 3.0
    assert on.card_admission.threshold_gb("mask", "mask") is None
    assert on.oom_retry == OomRetryConfig(max_total_wait_s=900.0, backoff_s=(10.0, 120.0))
    defaults = batched_engine_config(_cfg("bench.batched.card_admission={min_free_gb:{hawor_worker:4}}").bench, engine)
    assert defaults.card_admission is not None
    assert (defaults.card_admission.poll_s, defaults.card_admission.max_wait_s) == (5.0, 600.0)
    assert defaults.card_admission.reclaim_cache is True                            # on with admission
    off = batched_engine_config(_cfg("bench.batched.card_admission={min_free_gb:{shape:4},reclaim_cache:false}").bench,
                                engine)
    assert off.card_admission is not None and off.card_admission.reclaim_cache is False

    for bad in ("{min_free_gb:{decode:1}}", "{min_free_gb:{qc:1}}", "{min_free_gb:{bogus:1}}",
                "{min_free_gb:{geom_worker:1}}", "{min_free_gb:{shape:-1}}", "{min_free_gb:{shape:nan}}",
                "{min_free_gb:{shape:true}}", "{min_free_gb:{shape:'4'}}", "{min_free_gb:{}}", "{min_free_gb:4}",
                "{min_free_gb:{shape:4},poll_s:0}", "{min_free_gb:{shape:4},poll_s:inf}",
                "{min_free_gb:{shape:4},max_wait_s:-1}", "{min_free_gb:{shape:4},min_wait_s:1}", "{poll_s:1}", "4",
                "{min_free_gb:{shape:4},reclaim_cache:1}", "{min_free_gb:{shape:4},reclaim_cache:'yes'}"):
        with pytest.raises(ValueError, match="card_admission"):
            batched_engine_config(_cfg(f"bench.batched.card_admission={bad}").bench, engine)
    for bad in ("{max_total_wait_s:0,backoff_s:[1,2]}", "{max_total_wait_s:nan,backoff_s:[1,2]}",
                "{max_total_wait_s:10,backoff_s:[0,2]}", "{max_total_wait_s:10,backoff_s:[3,2]}",
                "{max_total_wait_s:10,backoff_s:[1]}", "{max_total_wait_s:10,backoff_s:[1,2,3]}",
                "{max_total_wait_s:10,backoff_s:1}", "{max_total_wait_s:10,backoff_s:[1,inf]}",
                "{max_total_wait_s:10}", "{backoff_s:[1,2]}", "{max_total_wait_s:10,backoff_s:[1,2],retries:3}",
                "{max_total_wait_s:true,backoff_s:[1,2]}", "10"):
        with pytest.raises(ValueError, match="oom_retry"):
            batched_engine_config(_cfg(f"bench.batched.oom_retry={bad}").bench, engine)
    with pytest.raises(ValueError, match="CardAdmissionConfig"):
        BatchedEngineConfig(engine, card_admission={"min_free_gb": {"shape": 4.0}})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="OomRetryConfig"):
        BatchedEngineConfig(engine, oom_retry={"max_total_wait_s": 1.0})  # type: ignore[arg-type]
