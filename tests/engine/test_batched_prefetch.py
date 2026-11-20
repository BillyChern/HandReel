"""Tests for `prefetch_decode`: wave k+1's decode on one background thread while wave k runs.

Every run is compared against the same run with the flag off: artifacts, manifest states and batch ids, the ordered
batch log (all fields but the timing ones: `wall_ns`, `share_ns`, `started_utc`, `peak_gpu_mem_gb`) and the per-clip
profile records (all but `seconds` and the peak) must match.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import pytest

from fastdo.bench.bench_batched import compare_artifact_dirs
from fastdo.bench.profiler import ProfileReport
from fastdo.engine.apportion import BatchRecord, apportion_ns, check_apportionment, read_batch_log
from fastdo.engine.batched_runner import (
    BatchedEngineConfig,
    BatchRunReport,
    StageBatchConfig,
    run_manifest_batched,
)
from fastdo.engine.batched_stages import BatchedStage, ConcurrentClipStage, FrameBatchedStage
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, StageFn
from fastdo.engine.worker_support import BatchedWorker
from tests.engine.fakes import random_frames, write_decode
from tests.engine.fakes_batched import RowMeanWorker, SlotWorkerFactory

ROWS = {"c0": 4, "c1": 7, "c2": 5, "c3": 9, "c4": 6, "c5": 3, "c6": 8}
IDS = list(ROWS)
STAGES = ("decode", "encoder", "tracks", "qc")
PREFETCH_THREAD = "fastdo-decode-prefetch"


class FakeDecode(BatchedWorker):
    """Writes `decode.npz` with `ROWS[c]` rows (manifest `n_frames` is 3x that, so weights show which one was read).

    `fail_when(ids)` raises `RuntimeError("poison decode")`; a clip of `interrupt` raises `KeyboardInterrupt`;
    `on_clip(clip_id)` runs first. Records `(ids, thread name)` of every call.
    """

    def __init__(self, fail_when: Callable[[Sequence[str]], bool] | None = None, interrupt: Collection[str] = (),
                 on_clip: Callable[[str], None] | None = None) -> None:
        self.fail_when, self.interrupt, self.on_clip = fail_when, set(interrupt), on_clip
        self.calls: list[tuple[list[str], str]] = []

    def run_many(self, clips: Sequence[tuple[str, Path]]) -> None:
        ids = [clip_id for clip_id, _ in clips]
        self.calls.append((ids, threading.current_thread().name))
        for clip_id in ids:
            if self.on_clip is not None:
                self.on_clip(clip_id)
            if clip_id in self.interrupt:
                raise KeyboardInterrupt(clip_id)
        if self.fail_when is not None and self.fail_when(ids):
            raise RuntimeError("poison decode")
        for clip_id, artifact_dir in clips:
            write_decode(Path(artifact_dir), random_frames(ROWS[clip_id], seed=int(clip_id[1:])), stride=3)


def _qc(clip_id: str, artifact_dir: Path) -> None:
    (artifact_dir / "qc.json").write_text(json.dumps({"clip_id": clip_id, "accepted": True}))


def run_one(root: Path, prefetch: bool, decode: FakeDecode, wave: int, decode_batch: int,
            encoder: StageFn | BatchedStage | None = None, pre_done: Collection[str] = (),
            expect: type[BaseException] | None = None) -> tuple[ClipManifest, BatchRunReport | None]:
    manifest = ClipManifest.create(IDS, [f"{c}.mp4" for c in IDS], [3 * ROWS[c] for c in IDS])
    for clip_id in pre_done:                       # a resumed run: decode already done and on disk
        decode(clip_id, root / clip_id)
        manifest.mark(clip_id, "decode", "done", 1.0)
    decode.calls.clear()
    stages: dict[str, StageFn | BatchedStage] = {
        "decode": FrameBatchedStage(decode) if decode_batch > 1 else decode,
        "encoder": encoder if encoder is not None else FrameBatchedStage(RowMeanWorker("encoder")),
        "tracks": ConcurrentClipStage(SlotWorkerFactory("tracks")), "qc": _qc}
    cfg = BatchedEngineConfig(EngineConfig(artifact_root=root, stages=STAGES, manifest_path=root / "manifest.parquet"),
                              {"decode": StageBatchConfig(max_batch=decode_batch),
                               "encoder": StageBatchConfig(max_batch=3), "tracks": StageBatchConfig(max_batch=2)},
                              max_clips_per_wave=wave, run_id="t0", prefetch_decode=prefetch)
    if expect is None:
        return manifest, run_manifest_batched(manifest, cfg, stages)
    with pytest.raises(expect):
        run_manifest_batched(manifest, cfg, stages)
    return manifest, None


def _key(record: BatchRecord) -> tuple[object, ...]:
    return (record.batch_id, record.run_id, record.stage, record.clip_ids, record.weights, record.errors,
            record.retry_of, record.batched)


def assert_same(off: Path, on: Path, m_off: ClipManifest, m_on: ClipManifest) -> list[BatchRecord]:
    """The flag-on run recorded and wrote what the flag-off run did (module docstring); returns its batch log."""
    report = compare_artifact_dirs(off, on)
    assert report["n_mismatched"] == 0 and report["missing_in_a"] == [] == report["missing_in_b"]
    for clip_id in IDS:
        for stage in STAGES:
            a, b = m_off.status(clip_id, stage), m_on.status(clip_id, stage)
            assert (a.state, a.error, a.batch_id) == (b.state, b.error, b.batch_id), (clip_id, stage)
        assert (off / clip_id / "profile.json").is_file() == (on / clip_id / "profile.json").is_file()
        if (on / clip_id / "profile.json").is_file():
            p_off, p_on = (ProfileReport.from_json(root / clip_id / "profile.json") for root in (off, on))
            assert [(s.name, s.meta) for s in p_off.stages] == [(s.name, s.meta) for s in p_on.stages], clip_id
            for stage_record in p_on.stages:
                assert stage_record.seconds == m_on.status(clip_id, stage_record.name).seconds
    assert ClipManifest.load(on / "manifest.parquet").status("c0", "decode").state == m_on.status("c0", "decode").state
    records_off, records_on = read_batch_log(off / "batches.jsonl"), read_batch_log(on / "batches.jsonl")
    assert [_key(r) for r in records_off] == [_key(r) for r in records_on]
    for record in records_on:
        assert record.share_ns == tuple(apportion_ns(record.wall_ns, list(record.weights)))
    assert set(check_apportionment(m_on, records_on).unattributed_ns.values()) <= {0}
    return records_on


def _poison_group(ids: Sequence[str]) -> bool:
    return len(ids) > 1 and "c3" in ids


def _poison(ids: Sequence[str]) -> bool:
    return "c3" in ids


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
@pytest.mark.parametrize("decode_batch", [1, 2], ids=["stagefn", "batched2"])
@pytest.mark.parametrize("wave", [2, 3])
def test_prefetch_matches_the_sequential_run(tmp_path: Path, wave: int, decode_batch: int, resume: bool) -> None:
    pre_done = ("c2", "c3", "c5") if resume else ()     # c2+c3: a whole wave of 2 already decoded
    decodes = {flag: FakeDecode() for flag in (False, True)}
    runs = {flag: run_one(tmp_path / str(flag), flag, decodes[flag], wave, decode_batch, pre_done=pre_done)
            for flag in (False, True)}

    records = assert_same(tmp_path / "False", tmp_path / "True", runs[False][0], runs[True][0])
    off, on = runs[False][1], runs[True][1]
    assert off is not None and on is not None
    assert (on.n_batches, on.n_retries, on.solve_batched) == (off.n_batches, off.n_retries, off.solve_batched)
    assert on.wall_ns_per_stage.keys() == off.wall_ns_per_stage.keys() and off.overlapped_ns == 0
    first_wave = set(IDS[:wave])
    prefetched = [r for r in records if r.stage == "decode" and not set(r.clip_ids) & first_wave]
    assert on.overlapped_ns == sum(r.wall_ns for r in prefetched) > 0
    for record in records:                              # decode weighs n_frames: its frames were read BEFORE it ran
        assert record.weights == tuple((3 if record.stage == "decode" else 1) * ROWS[c] for c in record.clip_ids)
    # the first wave decodes on the main thread; every later one on the prefetch thread; a done clip never decodes
    threads = {c: name for ids, name in decodes[True].calls for c in ids}
    assert set(threads) == set(IDS) - set(pre_done) and sorted(c for ids, _ in decodes[True].calls for c in ids) == \
        sorted(threads)
    assert {c: name == PREFETCH_THREAD for c, name in threads.items()} == {c: c not in first_wave for c in threads}
    assert not any(t.name == PREFETCH_THREAD for t in threading.enumerate())


def test_decode_overlaps_the_previous_waves_stages(tmp_path: Path) -> None:
    second_wave_decoding = threading.Event()
    waited: list[bool] = []
    encoder_worker = RowMeanWorker("encoder")

    def encoder(clip_id: str, artifact_dir: Path) -> None:
        if clip_id in ("c0", "c1"):                 # wave 1 holds its GPU stage until wave 2's decode has started
            waited.append(second_wave_decoding.wait(timeout=10))
        encoder_worker(clip_id, artifact_dir)

    def on_clip(clip_id: str) -> None:
        if clip_id == "c2":
            second_wave_decoding.set()

    t0 = time.perf_counter()
    _, run = run_one(tmp_path, True, FakeDecode(on_clip=on_clip), 2, 1, encoder=encoder)
    assert waited == [True, True] and time.perf_counter() - t0 < 10
    assert run is not None and run.overlapped_ns > 0


@pytest.mark.parametrize(("poison", "decode_batch", "retries"),
                         [(_poison, 1, 0), (_poison_group, 2, 2), (_poison, 2, 2)],
                         ids=["stagefn_fails", "group_fails_retry_passes", "group_and_retry_fail"])
def test_a_failing_prefetched_decode_matches_today(tmp_path: Path, poison: Callable[[Sequence[str]], bool],
                                                   decode_batch: int, retries: int) -> None:
    decodes = {flag: FakeDecode(fail_when=poison) for flag in (False, True)}
    runs = {flag: run_one(tmp_path / str(flag), flag, decodes[flag], 3, decode_batch) for flag in (False, True)}

    records = assert_same(tmp_path / "False", tmp_path / "True", runs[False][0], runs[True][0])
    manifest, run = runs[True]
    assert run is not None and run.n_retries == retries
    decode_c3 = [(r.batch_id, r.clip_ids, r.errors, r.retry_of) for r in records if "c3" in r.clip_ids and
                 r.stage == "decode"]
    if decode_batch == 1:
        assert decode_c3 == [("t0-decode-00003", ("c3",), ("poison decode",), "")]
        assert manifest.status("c3", "decode").state == "failed"
    else:                                           # wave [c3 c4 c5] -> groups [c3 c4] [c5]; the group fails
        assert decode_c3[0] == ("t0-decode-00002", ("c3", "c4"), ("poison decode",) * 2, "")
        assert [r[3] for r in decode_c3[1:]] == ["t0-decode-00002"]
        # the retries took batch ids 00003-00004, so the prefetched [c5] (00003 on the thread) reran as 00005
        c5 = [(r.batch_id, r.clip_ids) for r in records if r.stage == "decode" and r.clip_ids == ("c5",)]
        assert c5 == [("t0-decode-00005", ("c5",))]
        assert [name for ids, name in decodes[True].calls if ids == ["c5"]] == [PREFETCH_THREAD, "MainThread"]
        expected = "done" if poison is _poison_group else "failed"
        assert manifest.status("c3", "decode").state == expected and manifest.status("c4", "decode").state == "done"


def test_a_keyboard_interrupt_in_prefetched_decode_propagates(tmp_path: Path) -> None:
    decodes = {flag: FakeDecode(interrupt=("c3",)) for flag in (False, True)}
    runs = {flag: run_one(tmp_path / str(flag), flag, decodes[flag], 2, 1, expect=KeyboardInterrupt)
            for flag in (False, True)}

    assert_same(tmp_path / "False", tmp_path / "True", runs[False][0], runs[True][0])
    manifest = runs[True][0]
    assert manifest.status("c2", "decode").state == "done" and manifest.status("c3", "decode").state == "pending"
    assert [name for ids, name in decodes[True].calls if ids == ["c3"]] == [PREFETCH_THREAD]
    assert not any(t.name == PREFETCH_THREAD for t in threading.enumerate())


def test_a_main_thread_interrupt_joins_the_prefetch_and_commits_none_of_it(tmp_path: Path) -> None:
    worker = RowMeanWorker("encoder", kill_when=lambda ids: "c1" in ids)
    manifest, _ = run_one(tmp_path, True, FakeDecode(), 2, 1, encoder=FrameBatchedStage(worker), expect=SystemExit)

    assert not any(t.name == PREFETCH_THREAD for t in threading.enumerate())
    assert [manifest.status(c, "decode").state for c in ("c0", "c1", "c2", "c3")] == ["done", "done", "pending",
                                                                                      "pending"]
    assert {r.stage for r in read_batch_log(tmp_path / "batches.jsonl")} == {"decode"}
    assert all("c2" not in r.clip_ids for r in read_batch_log(tmp_path / "batches.jsonl"))


def test_flag_off_or_no_leading_decode_starts_no_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        caplog: pytest.LogCaptureFixture) -> None:
    started: list[str] = []

    class Spy(threading.Thread):                   # every thread of the process, the stages' slot pools included
        def start(self) -> None:
            started.append(self.name)
            super().start()

    monkeypatch.setattr(threading, "Thread", Spy)
    _, run = run_one(tmp_path / "on", True, FakeDecode(), 2, 1)
    assert run is not None and started.count(PREFETCH_THREAD) == 3      # the control: waves 2-4 of 4
    started.clear()
    _, run = run_one(tmp_path / "off", False, FakeDecode(), 2, 1)
    assert run is not None and run.overlapped_ns == 0 and PREFETCH_THREAD not in started

    manifest = ClipManifest.create(IDS[:4], [f"{c}.mp4" for c in IDS[:4]], [ROWS[c] for c in IDS[:4]])
    for clip_id in IDS[:4]:
        FakeDecode()(clip_id, tmp_path / "late" / clip_id)
    cfg = BatchedEngineConfig(EngineConfig(artifact_root=tmp_path / "late", stages=("encoder", "decode")),
                              max_clips_per_wave=2, run_id="t0", prefetch_decode=True)
    with caplog.at_level(logging.WARNING, logger="fastdo.engine.batched_runner"):
        late = run_manifest_batched(manifest, cfg, {"encoder": RowMeanWorker("encoder"), "decode": FakeDecode()})
    assert late.overlapped_ns == 0 and all(manifest.status(c, "decode").state == "done" for c in IDS[:4])
    assert PREFETCH_THREAD not in started
    assert sum("prefetch_decode is set" in r.getMessage() for r in caplog.records) == 1
