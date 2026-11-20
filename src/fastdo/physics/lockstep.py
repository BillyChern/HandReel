"""Cooperative lock-step scheduling of several physics clips in one process.

Each clip's MPC loop is a Python generator that yields after issuing one step of GPU work (a rollout step or an
executed substep) and wherever it would otherwise block on the device (`wait_event`). `run_lockstep` resumes the
clips round-robin, switching the current torch and warp stream to the clip's own stream first, so the host keeps
every clip's queue fed while the GPU overlaps their kernels. Nothing is shared between clips except the device:
each clip has its own model, data, config, reference and random stream (`RngSlot`), so a clip's result is the one it
would produce alone (up to the nondeterminism its executor already has).

Torch is imported lazily so the scheduling logic is testable without CUDA.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any, Self

__all__ = ["Lane", "RngSlot", "drive_single", "run_lockstep", "wait_event"]


IDLE_SLEEP_S = 50e-6


def wait_event(event: Any) -> Generator[Any, None, None]:
    """Yield the event until it (a CUDA event, anything with `.query()`) has completed; never blocks the host.

    Yielding the event (a plain step yields None) tells the driver this resume made no progress: `drive_single`
    then blocks on `event.synchronize()`, and `run_lockstep` sleeps briefly when a whole round made none."""
    while not event.query():
        yield event


def drive_single(gen: Generator[Any, None, Any]) -> Any:
    """Run one clip generator to completion, BLOCKING on each yielded event (no busy-wait); returns its value."""
    try:
        while True:
            waiting = next(gen)
            if waiting is not None:
                waiting.synchronize()
    except StopIteration as stop:
        return stop.value


class RngSlot:
    """A clip's private state of the default CUDA generator.

    DO AS I DO draws from the default generator (`torch.randn_like`, `torch.rand`), seeded once per clip by
    `process_config`. With several clips in one process, `with slot:` loads this clip's (seed, offset) into the
    default generator and stores it back on exit, so each clip consumes exactly the stream it would alone. Inactive
    (single clip) it does nothing.
    """

    def __init__(self, active: bool, get_state: Callable[[], Any] | None = None,
                 set_state: Callable[[Any], None] | None = None, device: Any = "cuda") -> None:
        self.active = active
        self.state: Any = None
        if active and (get_state is None or set_state is None):
            import torch

            dev = torch.device(device)
            if dev.type != "cuda":
                raise ValueError("RngSlot holds a CUDA generator state; got device " + str(dev))
            dev = torch.device("cuda", torch.cuda.current_device() if dev.index is None else dev.index)

            def get_state() -> Any:
                return torch.cuda.get_rng_state(dev)

            def set_state(state: Any) -> None:
                torch.cuda.set_rng_state(state, dev)
        self._get, self._set = get_state, set_state

    def capture(self) -> None:
        """Take the default generator's current state as this clip's (call right after the clip's setup)."""
        if self.active:
            self.state = self._get()

    def __enter__(self) -> Self:
        if self.active:
            if self.state is None:
                raise RuntimeError("RngSlot used before capture()")
            self._set(self.state)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.active:
            self.state = self._get()


@dataclass
class Lane:
    """One clip in the lock-step loop."""

    name: str
    gen: Generator[Any, None, Any]
    streams: Any = None               # opaque, handed to `set_streams`
    result: Any = None
    error: BaseException | None = None
    resumes: int = 0
    done: bool = field(default=False)


def run_lockstep(lanes: Iterable[Lane], set_streams: Callable[[Lane], None] | None = None) -> list[Lane]:
    """Resume every unfinished lane once per round until all finish. A lane that raises is finished with `error`
    set (the others continue): one clip's failure is that clip's outcome, as it would be in its own process."""
    lanes = list(lanes)
    active = [ln for ln in lanes if not ln.done]
    while active:
        progressed = False
        for ln in list(active):
            if set_streams is not None:
                set_streams(ln)
            try:
                waiting = next(ln.gen)
                ln.resumes += 1
                progressed |= waiting is None
            except StopIteration as stop:
                ln.result, ln.done = stop.value, True
                active.remove(ln)
                progressed = True
            except Exception as exc:  # noqa: BLE001 -- recorded per lane, re-raised by the caller if wanted
                ln.error, ln.done = exc, True
                active.remove(ln)
                progressed = True
        if not progressed:
            time.sleep(IDLE_SLEEP_S)            # every lane is waiting on its device: do not spin a core
    return lanes
