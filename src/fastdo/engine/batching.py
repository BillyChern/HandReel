"""GPU-budget-aware dynamic batching for the engine's batched stage entry points.

The front-end stages batch in one of two ways, and the difference is deliberate:

- **Frame-batched** (`encoder`, `geometry`): DINOv2 and MoGe-2 are stateless per image, so crops or frames
  from several clips are concatenated into one forward pass. `dynamic_batches` sizes those passes: it closes
  a group as soon as the next item would push the group's estimated memory past the budget, so groups shrink
  as per-item cost grows, down to a single item, and never raise.
- **Clip-batched** (`mask`, `tracks`): SAM 2's video predictor and CoTracker3's online predictor are
  single-clip and stateful (SAM 2 keeps a per-video inference state with its memory bank; CoTracker3 online
  keeps its queries and window position on the model object itself), so their tensors are never concatenated
  across clips. "Dynamic batching" for them means running several clips' predictor instances at the same
  time, as many as fit the budget: `run_clips_concurrently` groups clips with `dynamic_batches` (a clip's
  cost is one predictor instance plus its activations) and runs each group's clips in parallel threads, one
  predictor instance per concurrent slot. Slots are created on demand and reused by later groups, so each
  instance still loads its model once. The slots live in a `SlotPool`, which the batched runner
keeps for a stage's whole run, across every group and wave.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from fastdo.engine.orchestrator import StageFn

__all__ = ["ClipRun", "SlotPool", "dynamic_batches", "run_clips_concurrently"]

logger = logging.getLogger(__name__)


def dynamic_batches[T](items: list[T], max_batch: int, gpu_budget_gb: float,
                    cost_fn: Callable[[T], float] = lambda _: 1.0) -> list[list[T]]:
    """Greedily groups `items`, in order, under a per-group item cap and a GPU memory budget.

    A group closes when it already holds `max_batch` items or when adding the next item would push the
    running cost estimate (the sum of `cost_fn` over the group) above `gpu_budget_gb`. An item whose own cost
    exceeds the budget still gets a group of its own: the batch shrinks to 1 and nothing raises.

    Args:
        items: Items to group; their order is preserved.
        max_batch: Largest group size, >= 1.
        gpu_budget_gb: Memory budget of one group in GiB (compared against summed `cost_fn` values).
        cost_fn: Estimated GiB one item adds to a group, finite and >= 0.

    Returns:
        Consecutive groups whose concatenation is exactly `items`.

    Raises:
        ValueError: If `max_batch < 1` or `cost_fn` returns a negative or non-finite cost.
    """
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    batches: list[list[T]] = []
    current: list[T] = []
    running = 0.0
    for item in items:
        cost = float(cost_fn(item))
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"cost_fn must return a finite cost >= 0, got {cost}")
        if current and (len(current) >= max_batch or running + cost > gpu_budget_gb):
            batches.append(current)
            current, running = [], 0.0
        current.append(item)
        running += cost
    if current:
        batches.append(current)
    return batches


@dataclass(frozen=True)
class ClipRun:
    """Outcome of one clip in `run_clips_concurrently`: wall seconds and the error text ("" on success)."""

    clip_id: str
    seconds: float
    error: str


def run_clips_concurrently(make_worker: Callable[[], StageFn], clips: Sequence[tuple[str, Path]],
                           max_concurrent: int, gpu_budget_gb: float,
                           cost_fn: Callable[[tuple[str, Path]], float] = lambda _: 1.0) -> list[ClipRun]:
    """Runs a clip-batched stage: groups of clips whose predictor instances fit the budget run together.

    Clips are grouped with `dynamic_batches(clips, max_concurrent, gpu_budget_gb, cost_fn)`. Group `g` runs
    its clips in parallel threads, clip `i` of the group on slot worker `i`; slot workers come from
    `make_worker()` on first need and are reused by later groups, so at most `max(len(group))` worker
    instances (and models) ever exist. A worker exception is caught per clip and reported in
    `ClipRun.error`, like `fastdo.engine.orchestrator.run_clip`; recording it in a manifest is the caller's job.

    Returns:
        One `ClipRun` per clip, in the order of `clips`.
    """
    pool = SlotPool(make_worker)
    runs: list[ClipRun] = []
    for group in dynamic_batches(list(clips), max_concurrent, gpu_budget_gb, cost_fn):
        runs.extend(pool.run_group(group))
    return runs


class SlotPool:
    """Concurrent slot workers of a clip-batched stage, made on first need and kept for the pool's lifetime.

    Args:
        make_worker: Builds one slot's `StageFn` (one predictor instance; it must never run two clips at once).
    """

    def __init__(self, make_worker: Callable[[], StageFn]) -> None:
        self._make_worker = make_worker
        self._slots: list[StageFn] = []

    @property
    def n_slots(self) -> int:
        """Slot workers made so far: the largest group run on this pool."""
        return len(self._slots)

    def run_group(self, clips: Sequence[tuple[str, Path]]) -> list[ClipRun]:
        """Runs clip `i` on slot `i`, each in its own thread, and waits for every thread.

        Returns:
            One `ClipRun` per clip, in input order; a slot's `Exception` becomes that clip's `ClipRun.error`.

        Raises:
            BaseException: The first `SystemExit`/`KeyboardInterrupt`/... a slot raised, re-raised in the calling
                thread once every thread of the group has finished; it is never recorded as a clip error.
        """
        while len(self._slots) < len(clips):
            self._slots.append(self._make_worker())
        runs: list[ClipRun | None] = [None] * len(clips)
        fatal: list[BaseException | None] = [None] * len(clips)

        def target(i: int, clip_id: str, artifact_dir: Path) -> None:
            try:
                runs[i] = _timed_run(self._slots[i], clip_id, artifact_dir)
            except BaseException as exc:  # noqa: BLE001 - handed to the calling thread below, never swallowed
                fatal[i] = exc

        threads = [threading.Thread(target=target, args=(i, clip_id, Path(artifact_dir)), name=f"slot-{i}")
                   for i, (clip_id, artifact_dir) in enumerate(clips)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for exc in fatal:
            if exc is not None:
                raise exc
        return [run for run in runs if run is not None]


def _timed_run(worker: StageFn, clip_id: str, artifact_dir: Path) -> ClipRun:
    start = time.perf_counter()
    try:
        worker(clip_id, artifact_dir)
    except Exception as exc:  # noqa: BLE001 - isolated per clip, reported in ClipRun.error
        logger.warning("clip %r failed in a concurrent group: %s", clip_id, exc)
        return ClipRun(clip_id, time.perf_counter() - start, str(exc))
    return ClipRun(clip_id, time.perf_counter() - start, "")
