"""Wave-pipelined batched orchestration: `run_manifest_batched` with `BatchedEngineConfig.pipeline_waves`.

The sequential runner (`fastdo.engine.batched_runner._Runner`) finishes every stage of wave k before wave k+1 starts,
although wave k+1 depends on nothing in wave k. Here each stage belongs to a resource class (`stage_classes`): the
in-process `gpu` class, a worker-subprocess class named in `pipeline_resources` (e.g. `shape: sam3d_worker`), or, for a
stage `stage_uses_gpu` calls CPU-only, a class of its own named after the stage. Every class has ONE thread, which
repeatedly takes the lowest-index in-flight wave whose next registered stage belongs to it and runs that stage for the
wave exactly as `_Runner._stage` does (`ready_clips`, `_groups`, `_group`, single retries). So:

- per wave, a stage starts only after the wave's previous registered stage (in `engine.stages` order) committed;
- no two attempts of one class overlap, in particular no two in-process CUDA stages;
- a stage's waves run in wave order on one thread (wave k reaches every stage no later than wave k+1, and the class
  takes the lowest ready index), so the per-stage `seq`, and with it every batch id, is the sequential run's;
- at most `pipeline_depth` waves have started and not finished; a wave is admitted when one finishes.

The run holds `hold_gpu_lock_for_process(<artifact_root>/.gpu.lock)` once, for its whole duration: other processes on
the root are excluded, and a stage that nests `gpu_lock` (`run_hands_call`) passes through on any thread. Attempts take
no per-attempt file lock. Only `gpu`-class attempts reset and read the CUDA peak; any other class records
`max(0.0, outcome.peak_gpu_mem_gb)`, as the decode prefetch does, since its CUDA work (if any) is in a subprocess.
`prefetch_decode` is subsumed: decode is a CPU class that runs ahead on its own.

Every commit (`_commit`: batch log, manifest mark and save, profiler adds, walls, counters), every `seq` step and every
manifest read (`ready_clips`, `_groups`) happens under one lock. `batches.jsonl` line ORDER may interleave across
stages; each line's content is the sequential run's. A wave's `profile.json` files are written when it finishes.

Failure: an `Exception` inside a stage is a unit failure with single retries, as before. Anything that escapes a stage
on a class thread (a `BaseException` such as `KeyboardInterrupt`, or a commit's `OSError`) halts the run: no attempt
starts after it, every thread finishes (and commits) the attempt it is in, the threads are joined, and the first such
exception is re-raised on the calling thread. The attempt that raised commits nothing. A `KeyboardInterrupt` on the
calling thread halts the run the same way.

Memory ledger (`BatchedEngineConfig.memory_ledger`, off by default): `pipeline_depth` > 2 lets several
in-process CUDA classes hold their working sets at once, which OOMed the solve at the 32 GB cap. With the
ledger on, an attempt of a stage that runs CUDA IN-PROCESS (`runs_cuda_in_process`: `stage_uses_gpu` and a class whose
name does not end in `_worker`; today `gpu`, `geom`, `prompt` and `mask`) reserves its group's DECLARED cost,
`StageBatchConfig.declared_cost_gb` (fixed x clips + per-frame x frames, else `budget_gb`, else 0; a per-attempt
`MemoryLedgerConfig.costs_gb` entry replaces it for its stage), from a
`MemoryLedger` of capacity `applied_cap_gb() - resident_gb`, and releases it when the call returns or raises. A
`*_worker` class (`sam3d_worker`, `hawor_worker`) is a worker SUBPROCESS: its device memory is another process's, its
main-process side allocates nothing (run/conf/config.yaml), so it is never gated. The wait comes after the batch id is
assigned and before the clock starts: each class keeps its one thread, so a class's order and every batch id are the
ledger-off run's, `wall_ns` excludes the wait, and only the gated attempts' `batches.jsonl` rows gain `ledger_wait_ns`
(`LedgerBatchRecord`). With the ledger off nothing in this paragraph runs and the rows are today's. A gated attempt
from which a `BaseException` escapes halts the run BEFORE releasing its reservation, and the halt wakes every waiter
with `_Stopped`, so no attempt starts on the memory a failing one freed and no thread is left waiting. Deadlock-free:
a thread reserves at most one cost, never while waiting, and an attempt above the capacity runs once the ledger empties.
Costs are DECLARATIONS, not measurements: a stage that declares nothing is admitted at 0 GiB (logged at start).

Prewarm (`BatchedEngineConfig.prewarm`, empty by default; `fastdo.engine.prewarm`): once the run holds the GPU
lock and before the first wave is admitted, each listed registered stage with a `prewarm()` hook gets a daemon thread
that waits for its stage's card admission and then calls the hook (a worker subprocess's model load), overlapping wave
0's first stages. The hooks never touch a wave, a batch id or the manifest; the stage's first attempt waits for its
prewarm. The threads are joined before the run returns. With the list empty nothing in this paragraph runs.

Card admission and patient OOM retry (`BatchedEngineConfig.card_admission` / `oom_retry`, off by default,
`fastdo.engine.card_admission`): the ledger cannot see memory other processes hold on the card. With
`card_admission` on, an attempt of a listed stage or class (subprocess workers and in-process stages alike) polls the
card's free memory after its batch id is taken, AFTER its ledger reservation (the reading is then closest to the call,
and the reservation already bounds this process's concurrent in-process work), and before its clock starts; the wait is
bounded by `max_wait_s`, so a reservation is never held indefinitely. Its row gains `card_wait_ns`. The patient OOM
retry is `_Runner._group`'s: its backoff sleeps on the class's own thread, so only that class waits and every other
class keeps running. A halt sets the wake event, which ends every admission poll and backoff at once, and the attempt
after it raises `_Stopped`; a clip whose chain a halt cut short stays pending, as any clip whose attempt a halt stopped.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from fastdo.bench.profiler import Profiler, _cuda_peak_gb
from fastdo.engine.apportion import BatchRecord, LedgerBatchRecord
from fastdo.engine.batched_runner import (
    BatchedEngineConfig,
    BatchRunReport,
    _Runner,
    _Timed,
    ready_clips,
)
from fastdo.engine.batched_stages import BatchedStage
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import StageFn, hold_gpu_lock_for_process, stage_uses_gpu
from fastdo.engine.prewarm import start_prewarm

__all__ = ["GPU_CLASS", "WORKER_CLASS_SUFFIX", "MemoryLedger", "PipelinedRunner", "runs_cuda_in_process",
           "stage_classes"]

logger = logging.getLogger(__name__)

GPU_CLASS = "gpu"
WORKER_CLASS_SUFFIX = "_worker"


def stage_classes(stages: Sequence[str], resources: Mapping[str, str]) -> dict[str, str]:
    """Resource class of each of `stages`: `resources[stage]`, else `gpu` when `stage_uses_gpu(stage)`, else the
    stage's own name (one CPU class per CPU-only stage). Only the `gpu` class measures the in-process CUDA peak."""
    return {stage: resources.get(stage) or (GPU_CLASS if stage_uses_gpu(stage) else stage) for stage in stages}


def runs_cuda_in_process(stage: str, cls: str) -> bool:
    """Whether `stage`, in resource class `cls`, allocates device memory in THIS process: it uses the GPU
    (`stage_uses_gpu`) and its class is not a worker-subprocess class, i.e. its name does not end in `_worker`
    (`sam3d_worker`, `hawor_worker`). Only such stages are gated by the memory ledger."""
    return stage_uses_gpu(stage) and not cls.endswith(WORKER_CLASS_SUFFIX)


class _Stopped(BaseException):
    """Raised at a class thread's next attempt once the run halts; caught by that thread, never re-raised."""


def _checked_gb(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)


class MemoryLedger:
    """Declared device GiB reserved by the in-process attempts running now (module docstring).

    `acquire(cost)` blocks until `reserved + cost <= capacity` OR nothing is reserved, then adds `cost`; so an attempt
    whose cost alone passes the capacity still starts once the ledger is empty, runs alone, and nothing deadlocks.
    `release(cost)` subtracts it and wakes every waiter; the sum resets to exactly 0.0 when the last holder leaves (no
    float residue). `halt()` makes every waiting and every later `acquire` raise `_Stopped`. Waiters are not queued:
    whichever fits when woken goes first. Costs must be finite and >= 0.
    """

    def __init__(self, capacity_gb: float) -> None:
        self.capacity_gb = _checked_gb(capacity_gb, "capacity_gb")
        if self.capacity_gb <= 0.0:
            raise ValueError(f"capacity_gb must be > 0, got {capacity_gb!r}")
        self._cond = threading.Condition()
        self._reserved = 0.0
        self._holders = 0
        self._waiting = 0
        self._halted = False

    @property
    def reserved_gb(self) -> float:
        with self._cond:
            return self._reserved

    @property
    def n_holders(self) -> int:
        with self._cond:
            return self._holders

    @property
    def n_waiting(self) -> int:
        with self._cond:
            return self._waiting

    def acquire(self, cost_gb: float) -> int:
        """Reserves `cost_gb`; returns the `perf_counter_ns` spent waiting. Raises `_Stopped` once halted."""
        cost = _checked_gb(cost_gb, "cost_gb")
        start = time.perf_counter_ns()
        with self._cond:
            self._waiting += 1
            try:
                while not self._halted and self._holders and self._reserved + cost > self.capacity_gb:
                    self._cond.wait()
            finally:
                self._waiting -= 1
            if self._halted:
                raise _Stopped
            self._reserved += cost
            self._holders += 1
        return time.perf_counter_ns() - start

    def release(self, cost_gb: float) -> None:
        """Returns `cost_gb` (what the matching `acquire` reserved) and wakes the waiters."""
        cost = _checked_gb(cost_gb, "cost_gb")
        with self._cond:
            if self._holders == 0:
                raise RuntimeError("MemoryLedger.release without a matching acquire")
            self._holders -= 1
            self._reserved = self._reserved - cost if self._holders else 0.0
            self._cond.notify_all()

    def halt(self) -> None:
        with self._cond:
            self._halted = True
            self._cond.notify_all()


@dataclass
class _Wave:
    """One in-flight wave: `next` indexes `PipelinedRunner.order`; `busy` while one of its stages runs."""

    index: int
    clip_ids: list[str]
    profilers: dict[str, Profiler]
    next: int = 0
    busy: bool = False


class PipelinedRunner(_Runner):
    """`_Runner` with the waves pipelined across resource classes (module docstring). Reuses `_groups`, `_group`,
    `_record` and `_commit` unchanged; overrides only where concurrency needs it."""

    def __init__(self, manifest: ClipManifest, cfg: BatchedEngineConfig,
                 stages: Mapping[str, StageFn | BatchedStage]) -> None:
        super().__init__(manifest, cfg, stages)
        self.order = [stage for stage in self.engine.stages if stage in self.stages]
        self.classes = stage_classes(self.order, cfg.pipeline_resources)
        ignored = sorted(set(cfg.pipeline_resources) - set(self.order))
        if ignored:
            logger.info("pipeline_resources names stage(s) %s this run does not register; ignored", ignored)
        self._cond = threading.Condition()
        self._queue: deque[list[str]] = deque()
        self._active: list[_Wave] = []
        self._admitted = 0
        self._halt = False
        self._error: BaseException | None = None
        self._ledger: MemoryLedger | None = None
        self._gated: frozenset[str] = frozenset()
        if cfg.memory_ledger is not None:
            self._ledger = MemoryLedger(cfg.memory_ledger.capacity_gb())
            self._gated = frozenset(s for s in self.order if runs_cuda_in_process(s, self.classes[s]))
            costs = cfg.memory_ledger.costs_gb
            undeclared = sorted(s for s in self._gated
                                if s not in costs and cfg.batch_config(s).declared_cost_gb([1]) == 0.0)
            logger.info("memory ledger: %.2f GiB (applied cap minus resident %.2f GiB) gates %s; per-attempt "
                        "costs_gb overrides %s", self._ledger.capacity_gb, cfg.memory_ledger.resident_gb,
                        sorted(self._gated), dict(sorted(costs.items())))
            for stage, gb in sorted(costs.items()):
                if stage not in self._gated:
                    logger.warning("memory ledger: costs_gb names %r, which this run does not gate (not registered, "
                                   "or a subprocess class); ignored", stage)
                elif gb > self._ledger.capacity_gb:
                    logger.warning("memory ledger: costs_gb[%r] = %.2f GiB exceeds the %.2f GiB capacity; each of its "
                                   "attempts runs alone", stage, gb, self._ledger.capacity_gb)
            if undeclared:
                logger.warning("memory ledger: stage(s) %s declare no device cost and are admitted at 0 GiB",
                               undeclared)

    def _prefetch_enabled(self) -> bool:
        if self.cfg.prefetch_decode:
            logger.info("pipeline_waves subsumes prefetch_decode: decode runs ahead as its own resource class")
        return False

    def run(self) -> BatchRunReport:
        start = time.perf_counter_ns()
        waves = self._waves()
        if waves:
            for stage in self.engine.stages:
                if stage not in self.stages and stage not in self.warned:
                    self.warned.add(stage)
                    logger.warning("no stage worker registered for %r; leaving it pending", stage)
        self._queue.extend(waves)
        names = list(dict.fromkeys(self.classes[stage] for stage in self.order))
        with hold_gpu_lock_for_process(self.lock_path):
            prewarms = self._start_prewarm() if waves else []
            with self._cond:
                self._admit()
            threads = [threading.Thread(target=self._serve, args=(name,), name=f"fastdo-pipeline-{name}")
                       for name in names]
            for thread in threads:
                thread.start()
            try:
                for thread in threads:
                    thread.join()
            except BaseException:       # the caller was interrupted: halt, let every thread finish its attempt
                self._stop(None)
                for thread in threads:
                    thread.join()
                raise
            finally:
                for thread in prewarms:     # a stage that ran waited for its prewarm; this only covers one that did not
                    thread.join()
        if self._error is not None:
            raise self._error
        wall = time.perf_counter_ns() - start
        total = sum(self.walls.values())
        overlapped = max(0, total - wall)
        overhead = max(0, wall - (total - overlapped))
        return BatchRunReport(self.run_id, self.log, self.n_batches, self.n_retries, dict(self.walls), overhead,
                              self.solve_batched, overlapped)

    def _start_prewarm(self) -> list[threading.Thread]:
        """`start_prewarm` of `cfg.prewarm`'s registered stages (module docstring of `fastdo.engine.prewarm`),
        each after the card admission its stage's attempts take; `[]` when the list is empty."""
        if not self.cfg.prewarm:
            return []
        wanted = [stage for stage in self.cfg.prewarm if stage in self.order]
        skipped = sorted(set(self.cfg.prewarm) - set(wanted))
        if skipped:
            logger.warning("prewarm lists stage(s) %s this run does not register; ignored", skipped)
        return start_prewarm(self.stages, wanted, before=self._card_wait_prewarm)

    def _card_wait_prewarm(self, stage: str) -> None:
        """Waits, on the prewarm thread, for the card admission `stage`'s attempts would wait for (nothing when off)."""
        card_ns = _Runner._card_wait(self, stage)
        if card_ns:
            logger.info("prewarm %s: waited %.1f s for card admission", stage, card_ns / 1e9)

    # --- scheduling: a docstring starting "(locked)" means the caller holds `self._cond` ---

    def _admit(self) -> None:
        """(locked) Starts queued waves while fewer than `pipeline_depth` are in flight; a wave with no registered
        stage finishes at once, as `_Runner._window` would write its profiles."""
        root = self.engine.artifact_root
        while self._queue and len(self._active) < self.cfg.pipeline_depth and not self._halt:
            clip_ids = self._queue.popleft()
            profilers = {c: Profiler(c, self.engine.system, self.manifest.n_frames(c), self.engine.fps)
                         for c in clip_ids}
            for clip_id in clip_ids:
                (root / clip_id).mkdir(parents=True, exist_ok=True)
            wave = _Wave(self._admitted, clip_ids, profilers)
            self._admitted += 1
            self._active.append(wave)
            if not self.order:
                self._finish(wave)

    def _finish(self, wave: _Wave) -> None:
        """(locked) Writes the wave's `profile.json` files and retires it."""
        for clip_id in wave.clip_ids:
            wave.profilers[clip_id].report().to_json(self.engine.artifact_root / clip_id / "profile.json")
        self._active.remove(wave)

    def _take(self, name: str) -> _Wave | None:
        """(locked) The lowest-index idle wave whose next stage is class `name`'s, marked busy; waits for one. None
        once the run halts or no wave is in flight or queued."""
        while not self._halt:
            for wave in self._active:
                if not wave.busy and self.classes[self.order[wave.next]] == name:
                    wave.busy = True
                    return wave
            if not self._active and not self._queue:
                return None
            self._cond.wait()
        return None

    def _advance(self, wave: _Wave) -> None:
        """(locked) The wave's current stage committed: move to its next stage, or finish it and admit another."""
        wave.busy = False
        wave.next += 1
        if wave.next == len(self.order):
            self._finish(wave)
            self._admit()
        self._cond.notify_all()

    def _stop(self, exc: BaseException | None) -> None:
        """Halts the run; keeps the first exception that escaped a class thread. Wakes the ledger's waiters too."""
        with self._cond:
            if exc is not None and self._error is None:
                self._error = exc
            self._halt = True
            self._cond.notify_all()
        self._wake.set()
        if self._ledger is not None:
            self._ledger.halt()

    def _serve(self, name: str) -> None:
        """One resource class's thread (module docstring)."""
        while True:
            with self._cond:
                wave = self._take(name)
            if wave is None:
                return
            try:
                self._run_stage(self.order[wave.next], wave)
                with self._cond:
                    self._advance(wave)
            except _Stopped:
                return
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread by `run`
                self._stop(exc)
                return

    def _run_stage(self, stage: str, wave: _Wave) -> None:
        """`_Runner._stage` for one wave, with the manifest reads under the lock."""
        with self._cond:
            ready = ready_clips(self.manifest, self.engine, stage, self.stages.keys(), wave.clip_ids)
            frames, groups = self._groups(stage, ready) if ready else ({}, [])
        for group in groups:
            self._group(stage, self.stages[stage], group, frames, wave.profilers)

    # --- `_Runner` overrides ---

    def _next_batch_id(self, stage: str) -> str:
        with self._cond:
            return super()._next_batch_id(stage)

    def _attempt(self, stage: str, impl: StageFn | BatchedStage, group: list[str], frames: Mapping[str, int],
                 retry_of: str, totals: dict[str, int], attempts: dict[str, list[str]],
                 ) -> tuple[BatchRecord, dict[str, str]]:
        """`_Runner._attempt` without a per-attempt file lock (the run holds it), measuring the CUDA peak only on the
        `gpu` class; raises `_Stopped` instead of starting once the run halts. A stage the memory ledger gates
        reserves its declared cost after the batch id is taken and before the clock starts (module docstring)."""
        if self._halt:
            raise _Stopped
        batch_id = self._next_batch_id(stage)
        if stage not in self._gated:
            card_ns = self._card_wait(stage)
            record, errors = self._timed_attempt(stage, impl, group, frames, retry_of, totals, attempts, batch_id)
            return self._with_card_wait(record, card_ns), errors
        assert self._ledger is not None and self.cfg.memory_ledger is not None
        cost = self.cfg.memory_ledger.attempt_cost_gb(
            stage, self.cfg.batch_config(stage).declared_cost_gb([frames[c] for c in group]))
        wait_ns = self._ledger.acquire(cost)
        try:
            card_ns = self._card_wait(stage)
            record, errors = self._timed_attempt(stage, impl, group, frames, retry_of, totals, attempts, batch_id)
        except _Stopped:                # halted while waiting for the card: nothing ran, the halt is already recorded
            raise
        except BaseException as exc:
            self._stop(exc)             # halt BEFORE the release: no waiter may start on the memory this attempt held
            raise
        finally:
            self._ledger.release(cost)
        return self._with_card_wait(LedgerBatchRecord.of(record, wait_ns), card_ns), errors

    def _card_wait(self, stage: str) -> int | None:
        """`_Runner._card_wait`, raising `_Stopped` instead of starting when the run halted during the wait."""
        card_ns = super()._card_wait(stage)
        if card_ns is not None and self._halt:
            raise _Stopped
        return card_ns

    def _timed_attempt(self, stage: str, impl: StageFn | BatchedStage, group: list[str], frames: Mapping[str, int],
                       retry_of: str, totals: dict[str, int], attempts: dict[str, list[str]], batch_id: str,
                       ) -> tuple[BatchRecord, dict[str, str]]:
        """The call, its clock and its record, for the batch id `_attempt` took."""
        started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        measure = self.classes[stage] == GPU_CLASS
        if measure:
            _cuda_peak_gb(reset=True)
        t0 = time.perf_counter_ns()
        outcome = self._unit_call(impl, group, batch_id)
        device_peak = _cuda_peak_gb(reset=False) if measure else 0.0
        wall_ns = time.perf_counter_ns() - t0
        return self._record(stage, group, frames, retry_of, totals, attempts,
                            _Timed(batch_id, started, outcome, wall_ns), device_peak)

    def _commit(self, record: BatchRecord, final: list[str], errors: Mapping[str, str], frames: Mapping[str, int],
                profilers: Mapping[str, Profiler], totals: Mapping[str, int],
                attempts: Mapping[str, list[str]]) -> None:
        with self._cond:
            super()._commit(record, final, errors, frames, profilers, totals, attempts)
