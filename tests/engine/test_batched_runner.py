"""Tests for `fastdo.engine.batched_runner.run_manifest_batched`: stage waves against the single-stream reference,
frame-proportional apportionment, failure isolation with a single retry, and readiness."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from fastdo.bench.bench_batched import compare_artifact_dirs
from fastdo.bench.profiler import ProfileReport
from fastdo.bench.yield_report import clip_rate_agreement, compute_yield_report
from fastdo.engine.apportion import apportion_ns, check_apportionment, read_batch_log
from fastdo.engine.batched_runner import (
    BatchedEngineConfig,
    StageBatchConfig,
    ready_clips,
    run_manifest_batched,
)
from fastdo.engine.batched_stages import (
    ConcurrentClipStage,
    FrameBatchedStage,
    HandsBatchStage,
    SolveBatchStage,
)
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, StageFn, run_manifest
from fastdo.engine.workers import make_encode_worker, make_hands_worker, make_tracks_worker
from fastdo.solver.api import SolverConfig
from tests.engine.fakes import FakeDinov2, FakeOnlineTracker
from tests.engine.fakes_batched import (
    FakeHandsDriver,
    FakeSolveMany,
    RowMeanWorker,
    SlotWorkerFactory,
    fail_for,
    finish_fake_solve,
    prepare_fake_solve,
    write_clip_inputs,
)
from tests.engine.test_hands_batch import HANDS_CFG

ROWS = (4, 7, 5, 9, 6)
IDS = [f"c{i}" for i in range(len(ROWS))]
STAGES = ("decode", "encoder", "mask", "hands", "geometry", "tracks", "solve", "qc")


def setup_clips(root: Path, rows: tuple[int, ...] = ROWS) -> ClipManifest:
    ids = [f"c{i}" for i in range(len(rows))]
    for seed, (clip_id, n) in enumerate(zip(ids, rows, strict=True)):
        write_clip_inputs(root, clip_id, n, seed=seed)
    return ClipManifest.create(ids, [f"{clip_id}.mp4" for clip_id in ids], list(rows))


def batched_cfg(root: Path, stages: tuple[str, ...], max_batch: int, run_id: str = "t0",
                manifest_path: Path | None = None, **kwargs: object) -> BatchedEngineConfig:
    engine = EngineConfig(artifact_root=root, stages=stages, manifest_path=manifest_path)
    return BatchedEngineConfig(engine, {s: StageBatchConfig(max_batch=max_batch) for s in stages}, run_id=run_id,
                               **kwargs)  # type: ignore[arg-type]


def _noop(clip_id: str, artifact_dir: Path) -> None:
    return None


def _qc(clip_id: str, artifact_dir: Path) -> None:
    (artifact_dir / "qc.json").write_text(json.dumps({"clip_id": clip_id, "accepted": True}))


def solve_fn(fake: FakeSolveMany) -> StageFn:
    """The single-stream solve: `prepare_fake_solve`, one clip through `fake`, `finish_fake_solve`."""
    def run(clip_id: str, artifact_dir: Path) -> None:
        call = prepare_fake_solve(clip_id, artifact_dir)
        (solution,) = fake([call.mesh], [call.obs], [call.init_poses], [call.init_scale], SolverConfig())
        finish_fake_solve(clip_id, artifact_dir, call, solution)
    return run


def _tracks() -> StageFn:
    return fail_for(("c3",), "tracks", make_tracks_worker(n_queries=4, device="cpu"))


@pytest.mark.parametrize("wave", [None, 2], ids=["one_wave", "waves_of_2"])
def test_batching_equivalence_bitwise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wave: int | None) -> None:
    monkeypatch.setattr("fastdo.tracks.cotracker._load_cotracker_model", lambda device: FakeOnlineTracker())
    monkeypatch.setattr("fastdo.engine.workers_hands.run_hands_batched", FakeHandsDriver())
    monkeypatch.setattr("fastdo.engine.hands_batch.run_hands_batched", FakeHandsDriver())
    single, batched = tmp_path / "single", tmp_path / "batched"
    single_dino, batched_dino, walker = FakeDinov2(), FakeDinov2(), RowMeanWorker()
    reference = setup_clips(single)
    run_manifest(reference, EngineConfig(artifact_root=single, stages=STAGES), {
        "decode": _noop, "encoder": make_encode_worker(device="cpu", crop_size=14, max_batch=4, model=single_dino),
        "mask": SlotWorkerFactory("mask")(), "hands": make_hands_worker(HANDS_CFG), "geometry": RowMeanWorker(),
        "tracks": _tracks(), "solve": solve_fn(FakeSolveMany()), "qc": _qc})
    manifest = setup_clips(batched)
    run_manifest_batched(manifest, batched_cfg(batched, STAGES, 3, max_clips_per_wave=wave), {
        "decode": _noop,
        "encoder": FrameBatchedStage(make_encode_worker(device="cpu", crop_size=14, max_batch=4, model=batched_dino)),
        "mask": ConcurrentClipStage(SlotWorkerFactory("mask")),
        "hands": HandsBatchStage(HANDS_CFG, batched / ".hands_jobs", batched / ".gpu.lock", make_hands_worker(None)),
        "geometry": FrameBatchedStage(walker), "tracks": ConcurrentClipStage(_tracks),
        "solve": SolveBatchStage(prepare_fake_solve, finish_fake_solve, SolverConfig(), FakeSolveMany()), "qc": _qc})

    report = compare_artifact_dirs(single, batched)
    assert report["n_mismatched"] == 0 and report["missing_in_a"] == [] == report["missing_in_b"]
    assert report["n_files"] >= 5 * 7 - 2
    for clip_id in IDS:
        for stage in STAGES:
            a, b = reference.status(clip_id, stage), manifest.status(clip_id, stage)
            assert (a.state, a.error) == (b.state, b.error), (clip_id, stage)
    c3 = [manifest.status("c3", stage) for stage in ("tracks", "solve", "qc")]
    assert [(s.state, s.error) for s in c3] == [("failed", "fake tracks failure c3"), ("pending", ""), ("pending", "")]
    assert single_dino.batch_sizes == [4, 4, 3, 4, 1, 4, 4, 1, 4, 2]
    if wave is None:
        assert batched_dino.batch_sizes == [4, 4, 4, 4, 4, 4, 4, 3]
    assert any(len(set(ids)) > 1 for ids in walker.passes)


def test_apportionment_invariant_and_headline_rate(tmp_path: Path) -> None:
    manifest = setup_clips(tmp_path)
    tracks = SlotWorkerFactory("tracks", sleep_s=0.05, meet=("c0", "c1"))
    cfg = BatchedEngineConfig(EngineConfig(artifact_root=tmp_path, stages=("encoder", "tracks")),
                              {"encoder": StageBatchConfig(max_batch=3), "tracks": StageBatchConfig(max_batch=2)},
                              run_id="t0")

    run = run_manifest_batched(manifest, cfg, {"encoder": FrameBatchedStage(RowMeanWorker("encoder", sleep_s=0.02)),
                                               "tracks": ConcurrentClipStage(tracks)})

    records = read_batch_log(run.batch_log)
    assert run.batch_log == tmp_path / "batches.jsonl" and (run.n_batches, run.n_retries) == (5, 0)
    assert [(r.batch_id, r.clip_ids, r.weights) for r in records] == [
        ("t0-encoder-00000", ("c0", "c1", "c2"), (4, 7, 5)), ("t0-encoder-00001", ("c3", "c4"), (9, 6)),
        ("t0-tracks-00000", ("c0", "c1"), (4, 7)), ("t0-tracks-00001", ("c2", "c3"), (5, 9)),
        ("t0-tracks-00002", ("c4",), (6,))]
    for record in records:
        assert sum(record.share_ns) == record.wall_ns
        assert record.share_ns == tuple(apportion_ns(record.wall_ns, list(record.weights)))
        for clip_id, share in zip(record.clip_ids, record.share_ns, strict=True):
            status = manifest.status(clip_id, record.stage)
            assert status.seconds == share / 1e9 and status.batch_id == record.batch_id
    check = check_apportionment(manifest, records)
    assert check.unattributed_ns == {"encoder": 0, "tracks": 0} and check.stage_total_ns == check.batch_wall_ns
    total_ns = sum(record.wall_ns for record in records)
    assert run.wall_ns_per_stage == check.batch_wall_ns and run.overhead_ns >= 0
    seconds = [manifest.status(c, s).seconds for c in IDS for s in ("encoder", "tracks")]
    assert abs(math.fsum(seconds) - total_ns / 1e9) <= 1e-9
    assert compute_yield_report(manifest).clips_per_gpu_hour == pytest.approx(3600 * 5 / (total_ns / 1e9), rel=1e-12)
    for clip_id, rows in zip(IDS, ROWS, strict=True):
        profile = ProfileReport.from_json(tmp_path / clip_id / "profile.json")
        assert [record.name for record in profile.stages] == ["encoder", "tracks"]
        for record in profile.stages:
            status = manifest.status(clip_id, record.name)
            assert record.seconds == status.seconds and record.meta["batch_id"] == status.batch_id
            assert record.meta["frames"] == str(rows)
    assert clip_rate_agreement(manifest, tmp_path) == (0.0, 5)
    met = records[2]                        # c0 and c1 passed Barrier(2): they overlapped inside one wall
    assert tracks.calls["c0"] == tracks.calls["c1"] == 1 and sum(met.share_ns) == met.wall_ns


@pytest.mark.parametrize("case", ["clip_error", "unit_error", "fails_alone", "no_retry"])
def test_failure_isolation_retries_singly(tmp_path: Path, case: str) -> None:
    rows = (4, 7, 5)
    fake = FakeSolveMany()
    if case == "clip_error":
        tracks: object = ConcurrentClipStage(SlotWorkerFactory("tracks", fail_clips=("c1",)))
    else:
        poison = (lambda ids: "c1" in ids) if case == "fails_alone" else (lambda ids: len(ids) > 1 and "c1" in ids)
        tracks = FrameBatchedStage(RowMeanWorker("tracks", fail_when=poison))
    manifest = setup_clips(tmp_path / "batched", rows)
    cfg = batched_cfg(tmp_path / "batched", ("tracks", "solve"), 3, retry_singly=case != "no_retry")
    stages = {"tracks": tracks, "solve": SolveBatchStage(prepare_fake_solve, finish_fake_solve, SolverConfig(), fake)}

    run = run_manifest_batched(manifest, cfg, stages)  # type: ignore[arg-type]

    records = [r for r in read_batch_log(run.batch_log) if r.stage == "tracks"]
    states = {c: (manifest.status(c, "tracks").state, manifest.status(c, "tracks").error) for c in ("c0", "c1", "c2")}
    if case == "clip_error":
        assert [(r.batch_id, r.clip_ids, r.retry_of) for r in records] == [
            ("t0-tracks-00000", ("c0", "c1", "c2"), ""), ("t0-tracks-00001", ("c1",), "t0-tracks-00000")]
        assert states == {"c0": ("done", ""), "c1": ("failed", "fake tracks failure c1"), "c2": ("done", "")}
        assert [manifest.status(c, "tracks").batch_id for c in ("c0", "c1", "c2")] == [
            "t0-tracks-00000", "t0-tracks-00001", "t0-tracks-00000"]
        assert manifest.status("c1", "tracks").seconds == (records[0].share_ns[1] + records[1].wall_ns) / 1e9
        assert len(fake.calls) == 1 and [p.shape[0] for p in fake.calls[0][2]] == [4, 5]
        assert manifest.status("c1", "solve").state == "pending" and run.n_retries == 1

        def interrupt(clip_id: str, artifact_dir: Path) -> None:
            raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            run_manifest_batched(setup_clips(tmp_path / "ki", rows), batched_cfg(tmp_path / "ki", ("a",), 3),
                                 {"a": interrupt})
    elif case == "unit_error":
        assert len(records) == 4 and [r.retry_of for r in records[1:]] == ["t0-tracks-00000"] * 3
        assert set(states.values()) == {("done", "")}
        run_manifest(setup_clips(tmp_path / "single", rows), EngineConfig(tmp_path / "single", ("tracks", "solve")),
                     {"tracks": RowMeanWorker("tracks"), "solve": solve_fn(FakeSolveMany())})
        report = compare_artifact_dirs(tmp_path / "single", tmp_path / "batched")
        assert report["n_mismatched"] == 0 and not report["missing_in_a"] and not report["missing_in_b"]
    elif case == "fails_alone":
        assert len(records) == 4
        assert states == {"c0": ("done", ""), "c1": ("failed", "poison batch"), "c2": ("done", "")}
    else:
        assert len(records) == 1 and set(states.values()) == {("failed", "poison batch")} and fake.calls == []


def test_ready_clips_follows_run_clip_semantics(tmp_path: Path) -> None:
    manifest = ClipManifest.create(["c0", "c1", "c2"], ["s0", "s1", "s2"], [4, 4, 4])
    manifest.mark("c0", "a", "done", 1.0)
    manifest.mark("c1", "a", "failed", 1.0, error="x")
    engine = EngineConfig(artifact_root=tmp_path, stages=("a", "gate", "b"))
    ids = manifest.clip_ids()

    assert ready_clips(manifest, engine, "b", {"a", "b"}, ids) == ["c0"]
    manifest.mark("c0", "b", "done", 1.0)
    assert ready_clips(manifest, engine, "b", {"a", "b"}, ids) == []
    assert ready_clips(manifest, replace(engine, skip_done=False), "b", {"a", "b"}, ids) == ["c0"]
