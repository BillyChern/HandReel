"""Batched stage adapters for `fastdo.engine.batched_runner`: one `run_batch` call per group of clips.

- `FrameBatchedStage` (encoder, geometry): the worker's `run_many` shares forward passes across the group's clips.
- `ConcurrentClipStage` (mask, tracks): the group's clips run at once, one slot worker each, on a `SlotPool` kept for
  the adapter's lifetime, so a stage loads at most `n_slots` models however many groups and waves run.
- `HandsBatchStage`: one `run_hands_call` (one HaWoR driver load) for the group's clips without a precomputed npz;
  with `hands.warm_driver` every group is a request to one warm driver the stage owns until `close()`.
- `SolveBatchStage`: `prepare_solve` per clip, one `solve_trajectories` call for the group, then `finish_solve` per
  clip. Without `fastdo.solver.solve_trajectories` it falls back to a per-clip loop inside the same timed batch. With a
  `depth_anchor_gate`, the group's flagged clips are re-solved in one more call (`apply_depth_anchor_gate`).

`run_batch` raising means the whole group failed as a unit; per-clip failures go in `BatchOutcome.errors`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from torch import Tensor

import fastdo.solver
from fastdo.engine.batching import SlotPool
from fastdo.engine.branch_select import apply_branch_select
from fastdo.engine.depth_anchor_gate import apply_depth_anchor_gate
from fastdo.engine.hands_batch import hands_focal_length, run_hands_call
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.solve_stage_config import BranchSelectConfig, DepthAnchorGateConfig
from fastdo.engine.worker_support import DECODE_FILE, BatchedWorker
from fastdo.engine.workers_hands import HANDS_OUTPUT, HAWOR_FILE
from fastdo.engine.workers_solve import SolveCall
from fastdo.hands.hawor_batched import (
    HandsClipInput,
    HandsConfig,
    WarmHandsDriver,
    call_timeout_s,
    write_request,
)
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations, TrajectorySolution

__all__ = [
    "BatchOutcome",
    "BatchedStage",
    "ConcurrentClipStage",
    "FrameBatchedStage",
    "HandsBatchStage",
    "SolveBatchStage",
    "SolveManyFn",
    "resolve_solve_many",
]

logger = logging.getLogger(__name__)

SolveManyFn = Callable[[list[MeshSamples], list[TrajectoryObservations], list[Tensor], list[float], SolverConfig],
                       list[TrajectorySolution]]
FALLBACK_WARNING = "solve_trajectories unavailable: solving clips one at a time"
_warned: set[str] = set()


@dataclass(frozen=True)
class BatchOutcome:
    """Result of one `run_batch`: `clip_id -> error` for the clips that failed alone, the adapter's own peak GiB, and
    whether the compute was actually batched."""

    errors: Mapping[str, str] = field(default_factory=dict)
    peak_gpu_mem_gb: float = 0.0
    batched: bool = True


@runtime_checkable
class BatchedStage(Protocol):
    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome: ...


class FrameBatchedStage:
    """`worker.run_many(clips)`; an exception is never caught here (the group fails as a unit)."""

    def __init__(self, worker: BatchedWorker) -> None:
        self.worker = worker

    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        self.worker.run_many(clips)
        return BatchOutcome()


class ConcurrentClipStage:
    """The group's clips on one `SlotPool` (clip `i` on slot `i`); errors come from `ClipRun.error`."""

    def __init__(self, make_worker: Callable[[], StageFn]) -> None:
        self.pool = SlotPool(make_worker)

    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        return BatchOutcome(errors={run.clip_id: run.error for run in self.pool.run_group(clips) if run.error})


class HandsBatchStage:
    """Clips with `<artifact_dir>/all_hand_meshes.npz` run through `precomputed` one at a time, each caught
    separately; the rest make one `run_hands_call` with job dir `job_root / batch_id` under `gpu_lock(lock_path)`.
    With `hands_cfg.warm_driver` those calls go to one `WarmHandsDriver` (logs in `job_root`) that the stage
    keeps until `close()`, which the stage's owner calls at the end of the run.

    `prewarm()` (`fastdo.engine.prewarm`; the runner calls it only when `bench.batched.prewarm` lists `hands`):
    with a warm driver, sends it one request with NO clips (job dir `job_root / PREWARM_JOB`), which makes the server
    load and warm up its models exactly as its first request would, capture its post-warm-up RNG state and serve
    nothing; every real request then restores that state (`_hawor_batched_driver.serve_request`), as the first real
    request of an unprewarmed server starts from it, so the hands are unchanged. A real call waits for a prewarm in
    progress; a prewarm after the first real call does nothing. Without a warm driver there is nothing to keep loaded
    and `prewarm()` does nothing."""

    PREWARM_JOB = "prewarm"

    def __init__(self, hands_cfg: HandsConfig, job_root: Path, lock_path: Path, precomputed: StageFn) -> None:
        self.hands_cfg, self.job_root, self.lock_path, self.precomputed = hands_cfg, Path(job_root), lock_path, precomputed
        self.warm = WarmHandsDriver(hands_cfg, self.job_root) if hands_cfg.warm_driver else None
        self._warm_lock = threading.Lock()      # one request at a time on the warm driver (prewarm vs a call)
        self._called = False

    def close(self) -> None:
        """Ends the warm driver, if any. Safe to call twice."""
        if self.warm is not None:
            self.warm.close()

    def prewarm(self) -> None:
        """Load and warm up the warm driver's models now (class docstring)."""
        if self.warm is None:
            logger.info("prewarm hands: no warm driver (hands.warm_driver false); nothing to keep loaded")
            return
        with self._warm_lock:
            if self._called:
                logger.info("prewarm hands: skipped (the stage already called the driver)")
                return
            start = time.perf_counter()
            job = self.job_root / self.PREWARM_JOB
            request = write_request(job, [], self.hands_cfg)
            code = self.warm.request(request, call_timeout_s(self.hands_cfg, 0), job / "driver.log")
        if code == 0:
            logger.info("prewarm hands: warm driver loaded in %.1f s", time.perf_counter() - start)
        else:
            logger.warning("prewarm hands: the warm driver answered code %d after %.1f s; the first call loads it",
                           code, time.perf_counter() - start)

    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        errors: dict[str, str] = {}
        inputs: list[HandsClipInput] = []
        for clip_id, artifact_dir in clips:
            artifact_dir = Path(artifact_dir)
            if (artifact_dir / HAWOR_FILE).is_file():
                try:
                    self.precomputed(clip_id, artifact_dir)
                except Exception as exc:  # noqa: BLE001 - one clip's failure never fails its batch-mates
                    errors[clip_id] = str(exc)
                continue
            try:
                focal = hands_focal_length(artifact_dir)
            except (OSError, ValueError, KeyError) as exc:
                errors[clip_id] = str(exc)                 # run_clip's `str(exc)`: same error text in both modes
                continue
            inputs.append(HandsClipInput(clip_id, artifact_dir / DECODE_FILE, focal, artifact_dir / HANDS_OUTPUT))
        if not inputs:
            return BatchOutcome(errors=errors)
        with self._warm_lock:
            self._called = True
            report, call_errors = run_hands_call(inputs, self.hands_cfg, self.job_root / batch_id, self.lock_path,
                                                 warm=self.warm)
        return BatchOutcome(errors=errors | call_errors, peak_gpu_mem_gb=report.peak_cuda_gb)


def _solve_one_at_a_time(meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
                         init_scales: list[float], config: SolverConfig) -> list[TrajectorySolution]:
    return [fastdo.solver.solve_trajectory(mesh, o, poses, scale, config)      # looked up per call: patchable
            for mesh, o, poses, scale in zip(meshes, obs, init_poses, init_scales, strict=True)]


def resolve_solve_many() -> tuple[SolveManyFn, bool]:
    """`(fastdo.solver.solve_trajectories, True)` when it is callable now, else a per-clip loop and False (logged once)."""
    batched = getattr(fastdo.solver, "solve_trajectories", None)
    if callable(batched):
        return cast("SolveManyFn", batched), True
    if FALLBACK_WARNING not in _warned:
        _warned.add(FALLBACK_WARNING)
        logger.warning(FALLBACK_WARNING)
    return _solve_one_at_a_time, False


def _executed_sub_batches(solutions: Sequence[TrajectorySolution]) -> list[int] | str:
    """Clip counts of the batched solver's consecutive `BatchPlan`s (one plan object per sub-batch), or "n/a".
    Logging only: an unreadable plan is logged and never fails a batch whose solve already succeeded."""
    try:
        plans = [getattr(solution, "plan", None) for solution in solutions]
        executed = [len(p.clips) for i, p in enumerate(plans) if p is not None and (i == 0 or plans[i - 1] is not p)]
    except Exception as exc:  # noqa: BLE001 - diagnostics must not turn a solved batch into a unit failure
        logger.warning("could not read the solve sub-batch plans (%s: %s)", type(exc).__name__, exc)
        return "n/a"
    return executed or "n/a"


class SolveBatchStage:
    """`prepare` per clip, one `solve_many` call over the prepared clips in order, `finish` per clip.

    `depth_anchor_gate` (`solve_stage.depth_anchor_gate`; None = no gate): after the group's production solve,
    `apply_depth_anchor_gate` re-solves the flagged clips in one more `solve_many` call (anchor config), and `finish`
    gets each clip's gated call and kept solution. It must be the gate `finish` was built with: `finish_solve` refuses
    a call whose gate record disagrees with its config.

    `branch_select` (`solve_stage.branch_select`; None = one hypothesis): after the gate, `apply_branch_select`
    solves (and gates) every clip's other registration hypotheses in one more `solve_many` call and `finish` gets the
    chosen call and solution; `finish_solve` refuses a call whose branch record disagrees with its config."""

    def __init__(self, prepare: Callable[[str, Path], SolveCall],
                 finish: Callable[[str, Path, SolveCall, TrajectorySolution], None], config: SolverConfig,
                 solve_many: SolveManyFn | None = None, depth_anchor_gate: DepthAnchorGateConfig | None = None,
                 branch_select: BranchSelectConfig | None = None) -> None:
        self.prepare, self.finish, self.config, self._solve_many = prepare, finish, config, solve_many
        self.depth_anchor_gate = depth_anchor_gate
        self.branch_select = branch_select

    def solve_many(self) -> tuple[SolveManyFn, bool]:
        """The solve call and whether it batches: the injected function (True), else `resolve_solve_many()`."""
        return (self._solve_many, True) if self._solve_many is not None else resolve_solve_many()

    def run_batch(self, clips: Sequence[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        errors: dict[str, str] = {}
        prepared: list[tuple[str, Path, SolveCall]] = []
        for clip_id, artifact_dir in clips:
            try:
                prepared.append((clip_id, Path(artifact_dir), self.prepare(clip_id, Path(artifact_dir))))
            except Exception as exc:  # noqa: BLE001 - one clip's failure never fails its batch-mates
                errors[clip_id] = str(exc)
        solve_many, batched = self.solve_many()
        if not prepared:
            return BatchOutcome(errors=errors, batched=batched)
        calls = [call for _, _, call in prepared]
        solutions = solve_many([c.mesh for c in calls], [c.obs for c in calls], [c.init_poses for c in calls],
                               [c.init_scale for c in calls], self.config)
        if len(solutions) != len(calls):
            raise RuntimeError(f"solve_many returned {len(solutions)} solutions for {len(calls)} clips")
        logger.info("solve batch %s: %d clips planned in one call (batched %s), executed sub-batches %s", batch_id,
                    len(calls), batched, _executed_sub_batches(solutions))
        if self.depth_anchor_gate is not None:
            gated, solutions = apply_depth_anchor_gate([c for c, _, _ in prepared], calls, solutions,
                                                       self.depth_anchor_gate, self.config, solve_many)
            prepared = [(clip_id, artifact_dir, call) for (clip_id, artifact_dir, _), call in
                        zip(prepared, gated, strict=True)]
        if self.branch_select is not None:       # every clip's other hypotheses in one more call
            chosen, solutions = apply_branch_select([c for c, _, _ in prepared], [c for _, _, c in prepared], solutions,
                                                    self.branch_select, self.config, self.depth_anchor_gate, solve_many)
            prepared = [(clip_id, artifact_dir, call) for (clip_id, artifact_dir, _), call in
                        zip(prepared, chosen, strict=True)]
        for (clip_id, artifact_dir, call), solution in zip(prepared, solutions, strict=True):
            try:
                self.finish(clip_id, artifact_dir, call, solution)
            except Exception as exc:  # noqa: BLE001 - one clip's failure never fails its batch-mates
                errors[clip_id] = str(exc)
        return BatchOutcome(errors=errors, batched=batched)
