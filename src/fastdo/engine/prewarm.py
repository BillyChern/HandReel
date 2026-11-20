"""Model prewarm at run start: load a stage's worker-subprocess model beside wave 0's first stages.

Measured: the SAM 3D worker's pipeline load (49.5 s) and the warm HaWoR driver's load and warm-up sit on the
critical path of the pipelined run's fill, because both load on their stage's first use. A stage listed in
`bench.batched.prewarm` (default `[]`: nothing changes) that exposes a `prewarm()` method is loaded on its own
background thread as soon as the pipelined run starts (`fastdo.engine.pipelined_runner`), overlapping wave 0's decode,
prompt and mask.

Contract of a stage's `prewarm()`:
- it loads exactly what the stage's first use would load, the same way, so the outputs are unchanged (prewarm is
  EXACT: every artifact byte-identical);
- the stage's first real use waits for a prewarm in progress and then uses its result, and a prewarm that starts after
  the stage already loaded does nothing (no second model);
- a failed prewarm is logged at WARNING and the stage loads on first use as it always did;
- it never holds `fastdo.engine.worker_support`'s process-wide model-load lock: a worker-subprocess load only waits on
  another process, and holding that lock would serialize wave 0's in-process model loads behind it.

`PrewarmSlot` implements the once-and-hand-over part for a closure-based worker; `start_prewarm` runs the hooks.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["PrewarmSlot", "PrewarmableStageFn", "prewarm_hook", "start_prewarm"]

logger = logging.getLogger(__name__)


class PrewarmSlot:
    """One model loaded at most once, either by `prewarm` (a background thread) or on first use (`take`).

    States: idle -> running -> done (prewarm) and idle/done -> taken (first use). `take` returns the prewarmed value
    (waiting while it loads) or None, after which the caller loads as it always did; once taken, `prewarm` is a no-op.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._state = "idle"
        self._value: Any = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def prewarm(self, load: Callable[[], Any]) -> bool:
        """Runs `load()` on the calling thread unless the slot was already used; True when it ran and succeeded."""
        with self._lock:
            if self._state != "idle":
                logger.info("prewarm %s: skipped (slot %s)", self.name, self._state)
                return False
            self._state = "running"
        value, ok = None, False
        start = time.perf_counter()
        try:
            value, ok = load(), True
            logger.info("prewarm %s: loaded in %.1f s", self.name, time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - a failed prewarm falls back to the first-use load
            logger.warning("prewarm %s failed after %.1f s (%s: %s); the stage loads on first use", self.name,
                           time.perf_counter() - start, type(exc).__name__, exc)
        finally:
            with self._lock:
                self._value, self._state = value, "done"
            self._done.set()
        return ok

    def wait(self) -> None:
        """Blocks while a prewarm is running (returns at once otherwise). Call it outside any model-load lock."""
        with self._lock:
            running = self._state == "running"
        if running:
            start = time.perf_counter()
            self._done.wait()
            logger.info("prewarm %s: first use waited %.1f s for it", self.name, time.perf_counter() - start)

    def take(self) -> Any:
        """The prewarmed value (None if there is none: never prewarmed, or failed); the slot is closed afterwards."""
        self.wait()
        with self._lock:
            value, self._value = self._value, None
            self._state = "taken"
        return value


def prewarm_hook(impl: object) -> Callable[[], Any] | None:
    """`impl.prewarm` when the stage offers one, else None."""
    hook = getattr(impl, "prewarm", None)
    return hook if callable(hook) else None


def start_prewarm(stages: Mapping[str, object], names: Sequence[str],
                  before: Callable[[str], None] | None = None) -> list[threading.Thread]:
    """Starts one daemon thread per stage of `names` with a `prewarm()` hook; `before(stage)` runs first on that
    thread (the pipelined runner's card admission). A listed stage without a hook, or not registered, is logged and
    skipped. An `Exception` escaping a hook is logged; it never reaches the run."""
    threads = []
    for stage in names:
        impl = stages.get(stage)
        hook = None if impl is None else prewarm_hook(impl)
        if hook is None:
            logger.warning("prewarm: stage %r is %s; nothing to prewarm", stage,
                           "not registered" if impl is None else f"a {type(impl).__name__} without a prewarm() hook")
            continue

        def run(stage: str = stage, hook: Callable[[], Any] = hook) -> None:
            try:
                if before is not None:
                    before(stage)
                hook()
            except Exception as exc:  # noqa: BLE001 - a prewarm never fails the run
                logger.warning("prewarm %s raised %s: %s", stage, type(exc).__name__, exc)

        thread = threading.Thread(target=run, name=f"fastdo-prewarm-{stage}", daemon=True)
        thread.start()
        logger.info("prewarm: %s started", stage)
        threads.append(thread)
    return threads


class PrewarmableStageFn:
    """A plain `StageFn` that also offers `prewarm()` (the runner sees a `StageFn`: it has no `run_batch`); it carries
    the wrapped function's `__name__`, `__doc__` and `__wrapped__` (`functools.update_wrapper`)."""

    def __init__(self, fn: Callable[[str, Any], None], prewarm: Callable[[], Any]) -> None:
        self._fn = fn
        self.prewarm = prewarm
        functools.update_wrapper(self, fn)

    def __call__(self, clip_id: str, artifact_dir: Any) -> None:
        self._fn(clip_id, artifact_dir)
