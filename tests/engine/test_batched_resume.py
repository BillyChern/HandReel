"""Resumability of `run_manifest_batched` after a kill inside a batch or inside its single retry."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, tzinfo
from pathlib import Path

import pytest

from fastdo.engine.apportion import check_apportionment, read_batch_log
from fastdo.engine.batched_runner import run_manifest_batched
from fastdo.engine.batched_stages import ConcurrentClipStage, FrameBatchedStage
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.refusal import PROMPT_REFUSED_PREFIX
from tests.engine.fakes_batched import RowMeanWorker, SlotWorkerFactory
from tests.engine.test_batched_runner import batched_cfg, setup_clips


def test_resume_after_killed_batch_in_batch(tmp_path: Path) -> None:
    root, path = tmp_path / "artifacts", tmp_path / "manifest.parquet"
    manifest = setup_clips(root, (4, 4, 4, 4))
    cfg = batched_cfg(root, ("a", "b"), 2, manifest_path=path)
    killer = RowMeanWorker("b", kill_when=lambda ids: "c2" in ids)

    with pytest.raises(SystemExit):
        run_manifest_batched(manifest, cfg, {"a": FrameBatchedStage(RowMeanWorker("a")),
                                             "b": FrameBatchedStage(killer)})

    loaded = ClipManifest.load(path)
    assert all(loaded.status(c, "a").state == "done" for c in loaded.clip_ids())
    assert [loaded.status(c, "b").state for c in loaded.clip_ids()] == ["done", "done", "pending", "pending"]
    log = path.with_name("batches.jsonl")
    assert [r.batch_id for r in read_batch_log(log)] == ["t0-a-00000", "t0-a-00001", "t0-b-00000"]

    healthy_a, healthy_b = RowMeanWorker("a"), RowMeanWorker("b")
    run_manifest_batched(loaded, replace(cfg, run_id="t1"), {"a": FrameBatchedStage(healthy_a),
                                                            "b": FrameBatchedStage(healthy_b)})

    records = read_batch_log(log)
    assert healthy_a.calls == [] and healthy_b.calls == [["c2", "c3"]]
    assert len(records) == 4 and records[-1].batch_id == "t1-b-00000"
    reloaded = ClipManifest.load(path)
    assert all(reloaded.status(c, s).state == "done" for c in reloaded.clip_ids() for s in ("a", "b"))
    assert check_apportionment(reloaded, records).unattributed_ns == {"a": 0, "b": 0}


def test_resume_after_killed_batch_in_retry(tmp_path: Path) -> None:
    root, path = tmp_path / "artifacts", tmp_path / "manifest.parquet"
    manifest = setup_clips(root, (4, 7, 5))
    cfg = batched_cfg(root, ("a",), 3, manifest_path=path)

    with pytest.raises(SystemExit):
        run_manifest_batched(manifest, cfg, {"a": ConcurrentClipStage(SlotWorkerFactory("a", kill_clips=("c1",)))})

    loaded = ClipManifest.load(path)
    assert [loaded.status(c, "a").state for c in loaded.clip_ids()] == ["done", "pending", "done"]
    log = path.with_name("batches.jsonl")
    assert len(read_batch_log(log)) == 1

    run_manifest_batched(loaded, replace(cfg, run_id="t1"), {"a": ConcurrentClipStage(SlotWorkerFactory("a"))})

    records = read_batch_log(log)
    assert [(r.batch_id, r.clip_ids) for r in records[1:]] == [("t1-a-00000", ("c1",))]
    reloaded = ClipManifest.load(path)
    assert reloaded.status("c1", "a").state == "done" and reloaded.status("c1", "a").batch_id == "t1-a-00000"
    check = check_apportionment(reloaded, records)
    assert check.unattributed_ns["a"] == records[0].share_ns[1] > 0
    assert check.batch_wall_ns["a"] == records[0].wall_ns + records[1].wall_ns


def test_run_id_already_in_the_log_is_refused_or_suffixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class SameSecond(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> SameSecond:
            return cls(2026, 9, 14, 17, 0, 0, tzinfo=UTC)

    monkeypatch.setattr("fastdo.engine.batched_runner.datetime", SameSecond)
    root, path = tmp_path / "artifacts", tmp_path / "manifest.parquet"
    manifest = setup_clips(root, (4, 4, 4, 4))
    cfg = batched_cfg(root, ("a",), 2, run_id="", manifest_path=path)       # "" takes the default <second>-<pid>
    with pytest.raises(SystemExit):
        run_manifest_batched(manifest, cfg, {"a": FrameBatchedStage(RowMeanWorker("a", kill_when=lambda ids: "c2" in ids))})

    loaded = ClipManifest.load(path)
    run = run_manifest_batched(loaded, cfg, {"a": FrameBatchedStage(RowMeanWorker("a"))})   # resumed in the same second

    records = read_batch_log(path.with_name("batches.jsonl"))
    base = f"20260914T170000Z-{os.getpid()}"
    assert run.run_id == f"{base}-1" and [r.batch_id for r in records] == [f"{base}-a-00000", f"{base}-1-a-00000"]
    assert check_apportionment(ClipManifest.load(path), records).unattributed_ns == {"a": 0}
    with pytest.raises(ValueError, match=f"run_id '{base}-1' already has records"):
        run_manifest_batched(loaded, replace(cfg, run_id=f"{base}-1"), {"a": FrameBatchedStage(RowMeanWorker("a"))})


def test_the_batched_runner_also_stops_re_running_refusals(tmp_path: Path) -> None:
    """The batched path must apply the same verdict-vs-fault policy as `run_manifest`.

    Fixing only the sequential runner would leave the batched one paying for every refusal on every
    resume, and every existing test would still pass -- the two paths select their work independently.
    """
    root, path = tmp_path / "artifacts", tmp_path / "manifest.parquet"
    manifest = setup_clips(root, (4, 4))
    cfg = batched_cfg(root, ("a",), 2, manifest_path=path)

    class Refuser(RowMeanWorker):
        def run_many(self, clips: Sequence[tuple[str, Path]]) -> None:
            self.calls.append([clip_id for clip_id, _ in clips])
            raise RuntimeError(f"{PROMPT_REFUSED_PREFIX}: no interaction keyframe")

    first = Refuser("a")
    run_manifest_batched(manifest, cfg, {"a": FrameBatchedStage(first)})
    assert first.calls, "the stage should have been attempted once"
    assert all(manifest.status(c, "a").state == "failed" for c in manifest.clip_ids())

    again = Refuser("a")
    reloaded = ClipManifest.load(path)
    run_manifest_batched(reloaded, batched_cfg(root, ("a",), 2, run_id="t1", manifest_path=path),
                         {"a": FrameBatchedStage(again)})
    assert again.calls == [], "a refused clip must not be re-batched on resume"
