"""Tests for `fastdo.engine.orchestrator`.

Stage workers here are fakes/spies (`StageFn = Callable[[str, Path], None]`); the real workers are
tested elsewhere. Several tests exercise `gpu_lock` directly (a real cross-process mutual-exclusion
check via `subprocess`, a same-thread re-entrancy check, and a cross-thread contention check),
so the GPU file lock has its own mutual-exclusion test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

import fastdo.bench.profiler as profiler_module
from fastdo.engine.manifest import STAGE_NAMES, ClipManifest
from fastdo.engine.orchestrator import (
    EngineConfig,
    clips_per_gpu_hour_report,
    gpu_lock,
    run_clip,
    run_manifest,
)
from fastdo.engine.refusal import GATE_REJECTED_PREFIX, PROMPT_REFUSED_PREFIX


def test_run_clip_builds_its_own_profiler_per_clip(tmp_path: Path) -> None:
    manifest = ClipManifest.create(["clip0", "clip1"], ["s0", "s1"], [5, 7])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("decode",))

    def noop(_clip_id: str, _artifact_dir: Path) -> None:
        return None

    stage_fns = {"decode": noop}

    original_init = profiler_module.Profiler.__init__
    with mock.patch.object(
        profiler_module.Profiler, "__init__", autospec=True, side_effect=original_init,
    ) as spy:
        run_clip("clip0", cfg, stage_fns, manifest)
        run_clip("clip1", cfg, stage_fns, manifest)

    assert spy.call_count == 2

    report0 = json.loads((tmp_path / "clip0" / "profile.json").read_text())
    report1 = json.loads((tmp_path / "clip1" / "profile.json").read_text())
    assert report0["clip_id"] == "clip0"
    assert report1["clip_id"] == "clip1"
    assert report0["clip_id"] != report1["clip_id"]


def test_orchestrator_resumable(tmp_path: Path) -> None:
    manifest = ClipManifest.create(["clip0"], ["s"], [5])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("a", "b"))

    a_calls: list[str] = []
    b_state = {"raised": False}

    def worker_a(clip_id: str, _artifact_dir: Path) -> None:
        a_calls.append(clip_id)

    def worker_b(_clip_id: str, _artifact_dir: Path) -> None:
        if not b_state["raised"]:
            b_state["raised"] = True
            raise RuntimeError("transient failure")

    stage_fns = {"a": worker_a, "b": worker_b}

    run_manifest(manifest, cfg, stage_fns)
    assert manifest.status("clip0", "a").state == "done"
    assert manifest.status("clip0", "b").state == "failed"
    assert manifest.status("clip0", "b").error == "transient failure"

    run_manifest(manifest, cfg, stage_fns)
    assert manifest.status("clip0", "a").state == "done"
    assert manifest.status("clip0", "b").state == "done"
    assert a_calls == ["clip0"]


def test_orchestrator_isolates_clip_failure(tmp_path: Path) -> None:
    manifest = ClipManifest.create(
        ["clip1", "clip2", "clip3"], ["s", "s", "s"], [5, 5, 5],
    )
    cfg = EngineConfig(artifact_root=tmp_path, stages=("a", "b"))

    def worker_a(clip_id: str, _artifact_dir: Path) -> None:
        if clip_id == "clip2":
            raise ValueError("boom")

    def worker_b(_clip_id: str, _artifact_dir: Path) -> None:
        return None

    stage_fns = {"a": worker_a, "b": worker_b}

    run_manifest(manifest, cfg, stage_fns)  # must not raise

    for clip_id in ("clip1", "clip3"):
        assert manifest.status(clip_id, "a").state == "done"
        assert manifest.status(clip_id, "b").state == "done"

    assert manifest.status("clip2", "a").state == "failed"
    assert manifest.status("clip2", "a").error != ""
    # run_clip stops at the first failed stage: "b" was never attempted for clip2.
    assert manifest.status("clip2", "b").state == "pending"


def test_orchestrator_resumes_after_simulated_process_kill(tmp_path: Path) -> None:
    """`manifest_path` set -> every stage transition is saved immediately, so a process kill
    mid-batch loses at most the one stage in flight, never anything already marked.

    A worker raising `SystemExit` (not a plain `Exception`) simulates the process itself dying:
    `run_clip`'s `except Exception` deliberately does not catch it, so it propagates all the way
    out of `run_manifest`. A brand-new `ClipManifest` loaded from disk (a fresh object, not the
    original in-memory one -- simulating a real restart) must then resume exactly from the next
    pending stage/clip: no redone work, nothing skipped.
    """
    clip_ids = ["clip0", "clip1", "clip2", "clip3"]
    manifest = ClipManifest.create(clip_ids, ["s"] * 4, [5, 5, 5, 5])
    manifest_path = tmp_path / "manifest.parquet"
    cfg = EngineConfig(artifact_root=tmp_path, stages=("a", "b"), manifest_path=manifest_path)

    a_calls: list[str] = []
    b_calls: list[str] = []

    def worker_a(clip_id: str, _artifact_dir: Path) -> None:
        a_calls.append(clip_id)

    def worker_b_kills_on_clip2(clip_id: str, _artifact_dir: Path) -> None:
        b_calls.append(clip_id)
        if clip_id == "clip2":
            raise SystemExit("simulated process kill")

    with pytest.raises(SystemExit):
        run_manifest(manifest, cfg, {"a": worker_a, "b": worker_b_kills_on_clip2})

    # Fully processed before the kill.
    for clip_id in ("clip0", "clip1"):
        assert manifest.status(clip_id, "a").state == "done"
        assert manifest.status(clip_id, "b").state == "done"
    # clip2's "a" stage completed (and was saved) before "b" killed the process.
    assert manifest.status("clip2", "a").state == "done"
    assert manifest.status("clip2", "b").state == "pending"
    # clip3 was never reached.
    assert manifest.status("clip3", "a").state == "pending"

    # Simulate restarting the whole process: a brand new ClipManifest object, loaded from disk,
    # entirely replaces the in-memory one.
    reloaded = ClipManifest.load(manifest_path)
    for clip_id in ("clip0", "clip1"):
        assert reloaded.status(clip_id, "a").state == "done"
        assert reloaded.status(clip_id, "b").state == "done"
    assert reloaded.status("clip2", "a").state == "done"
    assert reloaded.status("clip2", "b").state == "pending"
    assert reloaded.status("clip3", "a").state == "pending"

    a_calls.clear()
    b_calls.clear()

    def worker_b_fixed(clip_id: str, _artifact_dir: Path) -> None:
        b_calls.append(clip_id)

    run_manifest(reloaded, cfg, {"a": worker_a, "b": worker_b_fixed})

    for clip_id in clip_ids:
        assert reloaded.status(clip_id, "a").state == "done"
        assert reloaded.status(clip_id, "b").state == "done"

    # Already-done work is never redone; only the interrupted/untouched clips re-run.
    assert sorted(a_calls) == ["clip3"]
    assert sorted(b_calls) == ["clip2", "clip3"]


def test_clips_per_gpu_hour_report() -> None:
    """Seconds in every state, `n` = the manifest's clips, `3600 / (total / n)`."""
    manifest = ClipManifest.create(["clip0", "clip1"], ["s", "s"], [10, 10])
    manifest.mark("clip0", "decode", "done", seconds=10.0)
    manifest.mark("clip1", "decode", "done", seconds=20.0)
    manifest.mark("clip0", "encoder", "done", seconds=5.0)
    manifest.mark("clip1", "encoder", "failed", seconds=1.0, error="x")

    report = clips_per_gpu_hour_report(manifest)
    assert report["decode"] == 240.0        # 3600 / (30 / 2)
    assert report["encoder"] == 1200.0      # the failed stage's 1.0 s is spent compute: 3600 / (6 / 2)
    assert report["overall"] == 200.0       # 3600 / (36 / 2)
    for stage in STAGE_NAMES:
        if stage not in ("decode", "encoder"):
            assert report[stage] == float("inf")


@pytest.mark.parametrize("n", [1, 2, 4, 10])
def test_clips_per_gpu_hour_report_is_per_clip(n: int) -> None:
    """N identical clips report the one-clip rates. D1's formula (done rows only, divided by the whole manifest's
    seconds) gives overall 800 / N and an infinite solve rate here, so it fails for every N."""
    manifest = ClipManifest.create([f"clip{i}" for i in range(n)], ["s"] * n, [10] * n)
    for clip_id in manifest.clip_ids():
        manifest.mark(clip_id, "decode", "done", seconds=3.0)
        manifest.mark(clip_id, "tracks", "done", seconds=1.5)
        manifest.mark(clip_id, "solve", "failed", seconds=0.5, error="x")

    report = clips_per_gpu_hour_report(manifest)

    expected = {"decode": 1200.0, "tracks": 2400.0, "solve": 7200.0, "overall": 720.0}
    assert set(expected) <= set(report) and set(STAGE_NAMES) <= set(report)
    for key, value in report.items():
        assert value == expected.get(key, float("inf")), key


def test_clips_per_gpu_hour_report_all_zero_is_infinite() -> None:
    manifest = ClipManifest.create(["clip0"], ["s"], [10])
    report = clips_per_gpu_hour_report(manifest)
    assert report["overall"] == float("inf")
    empty = clips_per_gpu_hour_report(ClipManifest.create([], [], []))   # n == 0: no ZeroDivisionError
    assert set(STAGE_NAMES) <= set(empty) and all(value == float("inf") for value in empty.values())


def test_run_clip_holds_gpu_lock_during_gpu_marked_stage(tmp_path: Path) -> None:
    """`encoder` is GPU-marked in `STAGE_GPU`; `run_clip` must already hold `gpu_lock` for it.

    Checked from a *different* thread than the one running `run_clip`: `gpu_lock` is deliberately
    re-entrant within a single thread (see `test_gpu_lock_is_reentrant_within_same_thread`), so a
    same-thread non-blocking re-acquire would now succeed as a no-op and prove nothing. A separate
    thread's `_lock_depth` is empty, so it genuinely contends for the OS-level flock.
    """
    manifest = ClipManifest.create(["clip0"], ["s"], [5])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("encoder",))
    lock_path = tmp_path / ".gpu.lock"

    contention_seen = threading.Event()
    probe_finished = threading.Event()

    def encoder_worker(_clip_id: str, _artifact_dir: Path) -> None:
        def probe() -> None:
            try:
                with gpu_lock(lock_path, blocking=False):
                    pass
            except BlockingIOError:
                contention_seen.set()
            finally:
                probe_finished.set()

        probe_thread = threading.Thread(target=probe)
        probe_thread.start()
        probe_thread.join(timeout=5.0)

    run_clip("clip0", cfg, {"encoder": encoder_worker}, manifest)
    assert probe_finished.is_set()
    assert contention_seen.is_set(), "a different thread must see the lock held during the stage"
    assert manifest.status("clip0", "encoder").state == "done"


def test_gpu_lock_is_reentrant_within_same_thread(tmp_path: Path) -> None:
    """Nested `with gpu_lock(path):` in the *same* thread must not self-deadlock.

    Before the fix, each nested call did its own `os.open` (a distinct open file description) and
    then blocked on `flock`, waiting forever for the outer call's still-held lock -- this test
    would hang without the thread-local depth counter, so it runs the nested acquisition in a
    background thread and asserts it completes well within a 5s timeout.
    """
    lock_path = tmp_path / ".gpu.lock"
    entered_inner = threading.Event()
    finished = threading.Event()

    def body() -> None:
        # Deliberately nested (not combined into one `with a, b:`) to visually match the
        # "nested gpu_lock acquisition" this test is about -- behaviourally identical either way.
        with gpu_lock(lock_path), gpu_lock(lock_path):
            entered_inner.set()
        finished.set()

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "nested gpu_lock acquisition hung (self-deadlock)"
    assert entered_inner.is_set()
    assert finished.is_set()

    # Fully released afterward (depth back to 0): a fresh non-blocking acquire must succeed.
    with gpu_lock(lock_path, blocking=False):
        pass


def test_gpu_lock_serializes_across_processes(tmp_path: Path) -> None:
    """Two separate OS processes must never hold `gpu_lock` on the same path at once."""
    lock_path = tmp_path / ".gpu.lock"
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"

    code = textwrap.dedent(
        """
        import json, sys, time
        from pathlib import Path
        from fastdo.engine.orchestrator import gpu_lock

        lock_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
        hold = float(sys.argv[3])
        with gpu_lock(lock_path):
            acquired = time.time()
            time.sleep(hold)
            released = time.time()
        out_path.write_text(json.dumps({"acquired": acquired, "released": released}))
        """
    )

    p1 = subprocess.Popen([sys.executable, "-c", code, str(lock_path), str(out1), "0.4"])
    time.sleep(0.1)
    p2 = subprocess.Popen([sys.executable, "-c", code, str(lock_path), str(out2), "0.1"])

    assert p1.wait(timeout=15) == 0
    assert p2.wait(timeout=15) == 0

    r1 = json.loads(out1.read_text())
    r2 = json.loads(out2.read_text())

    overlap = min(r1["released"], r2["released"]) - max(r1["acquired"], r2["acquired"])
    assert overlap <= 0.0, "both processes held the gpu lock at the same time"


def test_a_worker_can_declare_its_failure_fatal_to_the_run(tmp_path: Path) -> None:
    """The GPU guard SIGKILLing a model subprocess is not this clip's problem, it is the card's.

    Marching to the next clip meets the same card: on 2026-09-17 the guard killed a HaWoR driver every ~12 s while
    the peer sat at 74 GB, and the run would have marked most of a 140-clip corpus failed in minutes. A worker
    marks such an exception `fatal_to_run`; the orchestrator records THIS clip's failure (so a resumable rerun
    retries it) and then stops instead of burning the corpus.
    """
    seen: list[str] = []

    def killed(clip_id: str, artifact_dir: Path) -> None:
        seen.append(clip_id)
        exc = RuntimeError("HaWoR driver exited with code -9")
        exc.fatal_to_run = True                                        # type: ignore[attr-defined]
        raise exc

    manifest = ClipManifest.create(["c0", "c1", "c2"], ["s", "s", "s"], [3, 3, 3])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("decode",), system="t")
    with pytest.raises(RuntimeError, match="code -9"):
        run_manifest(manifest, cfg, {"decode": killed})

    assert seen == ["c0"], f"the run must stop at the first fatal clip, not continue: {seen}"
    assert manifest.status("c0", "decode").state == "failed"           # recorded, so a rerun retries it
    assert manifest.status("c1", "decode").state == "pending"          # never attempted


def test_an_ordinary_stage_failure_is_still_isolated_per_clip(tmp_path: Path) -> None:
    """The default stays: one clip's failure must not stop the corpus."""
    seen: list[str] = []

    def flaky(clip_id: str, artifact_dir: Path) -> None:
        seen.append(clip_id)
        if clip_id == "c0":
            raise RuntimeError("just this clip")

    manifest = ClipManifest.create(["c0", "c1", "c2"], ["s", "s", "s"], [3, 3, 3])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("decode",), system="t")
    run_manifest(manifest, cfg, {"decode": flaky})

    assert seen == ["c0", "c1", "c2"]
    assert manifest.status("c0", "decode").state == "failed"
    assert manifest.status("c2", "decode").state == "done"


def test_the_prompt_stage_holds_the_gpu_lock(tmp_path: Path) -> None:
    """`prompt` runs Hands23 and SAM 2's image predictor on the shared card (1.5 GB peak measured), so it must
    hold the lock. It was absent from `STAGE_GPU` for its whole first outing and `.get(stage, False)` answered
    False, so it ran two models outside the lock. Same cross-thread contention probe as `encoder` above."""
    manifest = ClipManifest.create(["clip0"], ["s"], [5])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("prompt",))
    lock_path = tmp_path / ".gpu.lock"

    contention_seen = threading.Event()
    probe_finished = threading.Event()

    def prompt_worker(_clip_id: str, _artifact_dir: Path) -> None:
        def probe() -> None:
            try:
                with gpu_lock(lock_path, blocking=False):
                    pass
            except BlockingIOError:
                contention_seen.set()
            finally:
                probe_finished.set()

        probe_thread = threading.Thread(target=probe)
        probe_thread.start()
        probe_thread.join(timeout=5.0)

    run_clip("clip0", cfg, {"prompt": prompt_worker}, manifest)
    assert probe_finished.is_set()
    assert contention_seen.is_set(), "the prompt stage must hold the GPU lock while its models are resident"
    assert manifest.status("clip0", "prompt").state == "done"


def test_a_stage_that_never_declared_its_gpu_use_is_locked(tmp_path: Path) -> None:
    """The mechanism, not just the instance: `STAGE_GPU.get(stage, False)` failed OPEN, so a stage nobody
    declared ran unlocked -- which is how P3's prompt stage ran two models on the shared card. The default is
    now True, which fails SAFE, and every stage the engine can really run is declared so nothing relies on it."""
    from fastdo.engine.orchestrator import stage_uses_gpu

    assert stage_uses_gpu("prompt") is True
    assert stage_uses_gpu("contact") is False
    assert stage_uses_gpu("driftqc") is False
    assert stage_uses_gpu("a_new_stage_someone_forgot") is True      # fails safe, not open

    # and every stage the engine can actually run is declared, so the fallback is never load-bearing
    from fastdo.engine.orchestrator import STAGE_GPU
    from fastdo.engine.workers_contact import with_contact
    from fastdo.engine.workers_driftqc import with_driftqc
    from fastdo.engine.workers_prompt import with_prompt

    every = with_contact(with_prompt(with_driftqc(STAGE_NAMES)))
    undeclared = [stage for stage in every if stage not in STAGE_GPU]
    assert undeclared == [], (f"{undeclared} can be run by the engine but never declared GPU use; they would "
                              f"fall back to the safe default instead of saying what they do")


def test_a_refused_clip_is_not_re_run_but_a_failed_one_is(tmp_path: Path) -> None:
    """Resume must retry faults and NOT retry verdicts, which look identical in the manifest.

    A refusal and a crash both land as `failed`; only the error prefix separates them. Before
    this, every resume re-ran the refusing stage to obtain the identical refusal -- 182.7 s per resume on
    the 45-window Ego4D run, for 13 refusals whose verdict is a deterministic function of artifacts
    already on disk.

    Both halves matter. Skipping faults too would silently abandon clips an OOM or a GPU-guard kill
    took down, which is the opposite and worse bug.
    """
    manifest = ClipManifest.create(["refused", "broke"], ["s0", "s0"], [5, 5])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("prompt",))
    calls: list[str] = []

    def worker(clip_id: str, _artifact_dir: Path) -> None:
        calls.append(clip_id)
        if clip_id == "refused":
            raise RuntimeError(f"{PROMPT_REFUSED_PREFIX}: no interaction keyframe")
        raise RuntimeError("CUDA out of memory")

    run_manifest(manifest, cfg, {"prompt": worker})
    assert sorted(calls) == ["broke", "refused"]
    assert manifest.status("refused", "prompt").state == "failed"
    assert manifest.status("broke", "prompt").state == "failed"

    calls.clear()
    run_manifest(manifest, cfg, {"prompt": worker})
    assert calls == ["broke"], "a refusal must not be re-run; a fault must be"


def test_retry_refusals_re_runs_them_for_a_changed_gate(tmp_path: Path) -> None:
    """The escape hatch exists because a CHANGED rule makes the old verdict no longer the answer."""
    manifest = ClipManifest.create(["refused"], ["s0"], [5])
    calls: list[str] = []

    def refuse(clip_id: str, _artifact_dir: Path) -> None:
        calls.append(clip_id)
        raise RuntimeError(f"{GATE_REJECTED_PREFIX}: p_interact 0.100000 < threshold 0.500000")

    run_manifest(manifest, EngineConfig(artifact_root=tmp_path, stages=("gate",)), {"gate": refuse})
    calls.clear()

    run_manifest(manifest, EngineConfig(artifact_root=tmp_path, stages=("gate",), retry_refusals=True),
                 {"gate": refuse})
    assert calls == ["refused"]


def test_a_refusal_finishes_the_clip_across_every_later_stage(tmp_path: Path) -> None:
    """A clip refused at `prompt` must not be re-admitted by its still-pending LATER stages.

    This is why `refused` is a per-CLIP question. Filtering stage by stage would leave `gate` (never
    run, so `pending`) putting the clip straight back into the work set, and the saving would be zero
    while every test of the refusing stage still passed.
    """
    manifest = ClipManifest.create(["refused"], ["s0"], [5])
    cfg = EngineConfig(artifact_root=tmp_path, stages=("prompt", "gate"))
    calls: list[str] = []

    def prompt(clip_id: str, _artifact_dir: Path) -> None:
        calls.append(f"prompt:{clip_id}")
        raise RuntimeError(f"{PROMPT_REFUSED_PREFIX}: no interaction keyframe")

    def gate(clip_id: str, _artifact_dir: Path) -> None:
        calls.append(f"gate:{clip_id}")

    run_manifest(manifest, cfg, {"prompt": prompt, "gate": gate})
    assert manifest.status("refused", "gate").state == "pending", "the refusal should stop the chain"
    calls.clear()

    run_manifest(manifest, cfg, {"prompt": prompt, "gate": gate})
    assert calls == [], "a still-pending later stage must not re-admit a refused clip"
