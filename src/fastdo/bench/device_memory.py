"""This process's CUDA memory as NVML reports it, sampled while a workload runs.

The shared-host guard and the GPU run queue act on `nvidia-smi --query-compute-apps` per-process `used_memory`.
That figure counts memory the caching allocator never sees: the CUDA context and its kernel images, library handles
and the workspaces libraries allocate themselves (cuSOLVER, MAGMA), and the memory the driver allocates when it
launches a kernel (a Triton kernel's per-block shared memory and per-thread local storage).
`torch.cuda.max_memory_allocated` and `max_memory_reserved` see none of it, and `set_per_process_memory_fraction`
does not bound it. `ProcessMemorySampler` polls the same query as the guard from a daemon thread, keeps this
process's peak, and can stop the process past a limit before the guard has to kill it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Self

__all__ = ["ProcessMemorySampler", "nvidia_smi_process_mib"]

logger = logging.getLogger(__name__)
Query = Callable[[], str]


def _nvidia_smi() -> str:
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=False, timeout=10).stdout


def nvidia_smi_process_mib(pid: int | None = None, query: Query = _nvidia_smi) -> tuple[int, int] | None:
    """`(MiB, rows)` for `pid` (default: this process): its `used_memory` summed over its NVML rows, as the guard sums
    them, and the number of rows. None when nvidia-smi is unavailable or lists no row for the pid."""
    key = str(os.getpid() if pid is None else pid)
    try:
        out = query()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi unavailable: %s", exc)
        return None
    rows = [int(parts[1]) for parts in ([x.strip() for x in line.split(",")] for line in out.splitlines())
            if len(parts) == 2 and parts[0] == key and parts[1].isdigit()]
    return (sum(rows), len(rows)) if rows else None


def _exit_past_limit(mib: int) -> None:
    logger.error("process GPU memory %d MiB is past the sampler's limit: exiting with status 3", mib)
    os._exit(3)


class ProcessMemorySampler:
    """Context manager polling `nvidia_smi_process_mib` every `interval_s` seconds from a daemon thread.

    `peak_mib` (None until a row is seen) and `max_rows` hold the largest sample, including one taken on exit. When a
    sample exceeds `stop_mib`, `on_limit(mib)` runs once on the sampler thread; the default logs and exits the process
    with status 3, so it releases the GPU on its own."""

    def __init__(self, interval_s: float = 0.25, stop_mib: int | None = None,
                 on_limit: Callable[[int], None] = _exit_past_limit, query: Query = _nvidia_smi,
                 pid: int | None = None) -> None:
        self.interval_s = interval_s
        self.stop_mib = stop_mib
        self.peak_mib: int | None = None
        self.max_rows = 0
        self.samples = 0
        self._on_limit = on_limit
        self._query = query
        self._pid = pid
        self._limit_hit = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> int | None:
        """Take one sample now and fold it into the peak; returns its MiB."""
        reading = nvidia_smi_process_mib(self._pid, self._query)
        self.samples += 1
        if reading is None:
            return None
        mib, rows = reading
        self.peak_mib = mib if self.peak_mib is None else max(self.peak_mib, mib)
        self.max_rows = max(self.max_rows, rows)
        if self.stop_mib is not None and mib > self.stop_mib and not self._limit_hit:
            self._limit_hit = True
            self._on_limit(mib)
        return mib

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def __enter__(self) -> Self:
        self.sample()
        self._thread = threading.Thread(target=self._run, name="gpu-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
