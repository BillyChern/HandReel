"""Card-level admission and patient CUDA-OOM retry for the batched runners. Both are opt-in.

Why. The memory ledger (`fastdo.engine.pipelined_runner.MemoryLedger`) bounds the DECLARED device cost of this
process's in-process attempts against its own per-process cap. It cannot see the card. In one measured run another
tenant held 35.34 GiB of the 79.19 GiB card, and attempts OOMed at the card level while this process was well under
its 32 GiB cap: the SAM 3D worker ("134.25 MiB is free"), but also 4-6 clip solve groups ("1.89 GiB is free" with
16.90 GiB in use), geometry and intrinsics. A one-clip group gets no retry at all, and a single retry (`retry_singly`)
runs at once, into the same full card. So windows were lost for good: 8 of 51 in one pass, 3 of 51 in another.

`card_admission` (`CardAdmissionConfig`). Before an attempt of a listed stage starts, the runner polls the card's free
memory every `poll_s` until it is at least that stage's `min_free_gb`. After `max_wait_s` it dispatches anyway and logs
a WARNING: admission only defers work, it never refuses it. A `min_free_gb` key is a stage name or a resource-class name
(`fastdo.engine.pipelined_runner.stage_classes`), and a stage key wins over its class's key. Unlisted stages are not
gated. In-process stages may be listed as well as the `*_worker` subprocess classes, because the card-level OOMs
above hit both. What "free" means depends on who allocates (`card_free_gb`):
- An in-process stage allocates through this process's caching allocator, which reuses its own cached blocks, so the
  reading is `fastdo.solver.memory.card_headroom_bytes`, as the solve's multi-clip planner reads it: the card's free
  memory plus this process's reserved-but-unallocated cache.
- A `*_worker` subprocess cannot use this process's cache, so the reading is the card's free memory alone
  (`torch.cuda.mem_get_info`).
Both readings are taken in the main process, which initialises CUDA anyway: the in-process stages run in it, and
`apply_gpu_budget` installs the cap there. Off CUDA the reading is None and admission passes at once (logged once).
Reclaim. A worker's reading is the card's free memory, and part of what fills the card is this
process's own reserved-but-unallocated cache: one run showed a SAM 3D request waiting 300 s, three times,
for 4 GiB while this process sat at 19.2 GiB and a foreign tenant at 36.4 GiB. The caching allocator gives that cache
back only on this process's OWN out-of-memory, never for another process's. So with `reclaim_cache` on (the default)
a worker-class wait that finds the card short first calls `release_cached_gb` (`torch.cuda.empty_cache`), logs the
GiB it returned and reads the card again, once per poll. An in-process reading already counts the cache, so it never
reclaims. Emptying the cache changes no stage output: it frees only blocks no tensor holds; the next allocation
pays a `cudaMalloc` instead.
Admission is best-effort. Two classes that pass admission together both see the same free memory, and a foreign tenant
can allocate between the check and the call. The patient retry is the backstop.

`oom_retry` (`OomRetryConfig`). The runner's existing retry (`_Runner._group`) retries each failed clip of a
multi-clip group once, alone, and immediately. With `oom_retry` on, that path is extended, not duplicated:
- a clip whose attempt failed with a CUDA OOM (`is_cuda_oom`) is retried alone, one-clip groups included (and with
  `retry_singly` off, which then still governs non-OOM failures);
- a retry that repeats the previous workload (the clip alone again) first waits an exponential backoff
  (`OomRetryConfig.delay_s`); the first retry after a multi-clip group stays immediate, as today, because it is a
  smaller workload;
- every retry is re-admitted by `card_admission` when that is on;
- the chain continues while the clip's total wait (backoff sleeps plus the retries' admission waits) is below
  `max_total_wait_s`, and each backoff is clipped to the budget left.
Each retry's `retry_of` names the previous attempt, so `check_apportionment` walks the whole chain. A non-OOM error
keeps today's handling: one immediate single retry after a multi-clip group, else final.

Timing. Waits happen after the attempt's batch id is taken and before its clock starts, so `wall_ns` excludes them
and batch ids are unchanged. `batches.jsonl` rows gain `card_wait_ns` (rows of admission-gated stages) and
`oom_backoff_ns` (every row while `oom_retry` is on; 0 unless a backoff preceded the attempt), after `ledger_wait_ns`
(`fastdo.engine.apportion.AdmissionBatchRecord`). With both keys null nothing in this module runs and the rows are
today's. A halt of the pipelined runner cuts every wait short (`wait_ns` on the runner's wake event).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from fastdo.engine.refusal import is_refusal

__all__ = ["CardAdmissionConfig", "OomRetryConfig", "card_free_gb", "is_cuda_oom", "release_cached_gb",
           "wait_for_card", "wait_ns"]

logger = logging.getLogger(__name__)

# Messages of a device-memory allocation failure, matched anywhere in a recorded error (a worker's reply or a
# subprocess log tail is quoted inside the runner's message). From measured batches.jsonl logs: torch's
# "CUDA out of memory" (in-process stages and the SAM 3D worker's "OutOfMemoryError: ..." reply), and SAM 3D's
# sparse-conv "Cuda error: 2[cudaMalloc(...)]" (2 = cudaErrorMemoryAllocation). The driver-level and library spellings
# of the same failure are listed too. `GpuBudgetError` ("the CUDA budget has ... left") is NOT here: it is a refusal
# against this process's own cap, which the memory ledger governs, and waiting for the card does not change it.
_OOM_PATTERNS = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|cudaErrorMemoryAllocation|"
    r"Cuda error: 2\[cudaMalloc|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED",
    re.IGNORECASE)

_NO_CUDA_LOGGED = threading.Event()


def _finite(value: object, name: str, *, positive: bool) -> float:
    """`value` as a float when it is a finite real number >= 0 (> 0 with `positive`); a bool is not a number."""
    if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) \
            and (value > 0.0 if positive else value >= 0.0):
        return float(value)
    raise ValueError(f"{name} must be a finite number {'>' if positive else '>='} 0, got {value!r}")


@dataclass(frozen=True)
class CardAdmissionConfig:
    """`bench.batched.card_admission` (module docstring): `min_free_gb` maps a stage or a resource class to the
    card GiB that must be free before one of its attempts starts; `poll_s` > 0 is the polling period and
    `max_wait_s` >= 0 the longest wait before dispatching anyway; `reclaim_cache` empties this process's allocator
    cache before a worker-class wait. Stored read-only."""

    min_free_gb: Mapping[str, float]
    poll_s: float = 5.0
    max_wait_s: float = 600.0
    reclaim_cache: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.reclaim_cache, bool):
            raise ValueError(f"bench.batched.card_admission.reclaim_cache must be a bool, got {self.reclaim_cache!r}")
        if not isinstance(self.min_free_gb, Mapping) or not self.min_free_gb:  # a ValueError, as every other refusal
            raise ValueError("bench.batched.card_admission.min_free_gb must map at least one stage or class to "
                             f"GiB, got {self.min_free_gb!r}")
        bad_keys = sorted(repr(k) for k in self.min_free_gb if not (isinstance(k, str) and k))
        if bad_keys:
            raise ValueError(f"bench.batched.card_admission.min_free_gb keys must be stage or class names, got "
                             f"{bad_keys}")
        values = {k: _finite(v, f"bench.batched.card_admission.min_free_gb[{k!r}]", positive=False)
                  for k, v in self.min_free_gb.items()}
        object.__setattr__(self, "min_free_gb", MappingProxyType(values))
        object.__setattr__(self, "poll_s", _finite(self.poll_s, "bench.batched.card_admission.poll_s", positive=True))
        object.__setattr__(self, "max_wait_s",
                           _finite(self.max_wait_s, "bench.batched.card_admission.max_wait_s", positive=False))

    def threshold_gb(self, stage: str, cls: str) -> float | None:
        """`min_free_gb[stage]`, else `min_free_gb[cls]`, else None (the stage is not gated)."""
        return self.min_free_gb.get(stage, self.min_free_gb.get(cls))


@dataclass(frozen=True)
class OomRetryConfig:
    """`bench.batched.oom_retry` (module docstring): a clip's backoff sleeps plus its retries' admission waits
    stop the chain once they reach `max_total_wait_s` > 0; `backoff_s = (initial, max)` with 0 < initial <= max."""

    max_total_wait_s: float
    backoff_s: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_total_wait_s",
                           _finite(self.max_total_wait_s, "bench.batched.oom_retry.max_total_wait_s", positive=True))
        pair = self.backoff_s
        if isinstance(pair, (str, bytes)) or not hasattr(pair, "__len__") or len(pair) != 2:
            raise ValueError(f"bench.batched.oom_retry.backoff_s must be [initial, max], got {pair!r}")
        initial = _finite(pair[0], "bench.batched.oom_retry.backoff_s[0]", positive=True)
        cap = _finite(pair[1], "bench.batched.oom_retry.backoff_s[1]", positive=True)
        if cap < initial:
            raise ValueError(f"bench.batched.oom_retry.backoff_s must satisfy initial <= max, got {list(pair)}")
        object.__setattr__(self, "backoff_s", (initial, cap))

    def delay_s(self, k: int) -> float:
        """The `k`-th backoff (0-based): `min(max, initial * 2**k)`."""
        initial, cap = self.backoff_s
        return cap if k >= 64 else min(cap, initial * 2.0 ** k)


def is_cuda_oom(error: str) -> bool:
    """Whether a recorded clip error is a device-memory allocation failure (`_OOM_PATTERNS`). A stage's refusal
    (`fastdo.engine.refusal.is_refusal`) never is, whatever its text quotes."""
    return bool(error) and not is_refusal(error) and _OOM_PATTERNS.search(error) is not None


def card_free_gb(in_process: bool) -> float | None:
    """GiB an attempt could allocate on the current CUDA device now (module docstring): `card_headroom_bytes` for an
    in-process stage, the card's free memory for a subprocess worker. None off CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    if in_process:
        from fastdo.solver.memory import card_headroom_bytes

        headroom = card_headroom_bytes("cuda")
        return None if headroom is None else headroom / 2**30
    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def release_cached_gb() -> float | None:
    """Empties this process's CUDA caching-allocator cache; returns the GiB of reserved memory it gave back to
    the card, None off CUDA. Only blocks no live tensor holds are freed."""
    import torch

    if not torch.cuda.is_available():
        return None
    before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    return max(0, before - torch.cuda.memory_reserved()) / 2**30


def wait_ns(wake: threading.Event, seconds: float) -> int:
    """Sleeps up to `seconds`, or until `wake` is set (a halt); returns the `perf_counter_ns` slept."""
    start = time.perf_counter_ns()
    if seconds > 0.0:
        wake.wait(seconds)
    return time.perf_counter_ns() - start


def wait_for_card(stage: str, need_gb: float, in_process: bool, cfg: CardAdmissionConfig,
                  wake: threading.Event) -> int:
    """Polls `card_free_gb(in_process)` every `cfg.poll_s` until it is >= `need_gb`, `cfg.max_wait_s` has been waited
    (then a WARNING, and the caller dispatches anyway), or `wake` is set; returns the ns waited. Off CUDA: 0 at once."""
    waited = 0
    limit = round(cfg.max_wait_s * 1e9)
    while True:
        free = card_free_gb(in_process)
        if free is None:
            if not _NO_CUDA_LOGGED.is_set():
                _NO_CUDA_LOGGED.set()
                logger.warning("card_admission is on but CUDA is unavailable in this process: not gating")
            return waited
        if free < need_gb and cfg.reclaim_cache and not in_process:
            released = release_cached_gb()
            if released:
                logger.info("card admission: %s released %.2f GiB of this process's allocator cache (%.2f GiB free "
                            "< %.2f)", stage, released, free, need_gb)
                free = card_free_gb(in_process)
                if free is None:
                    return waited
        if free >= need_gb:
            if waited:
                logger.info("card admission: %s admitted after %.1f s (%.2f GiB free >= %.2f)", stage, waited / 1e9,
                            free, need_gb)
            return waited
        if waited >= limit or wake.is_set():
            if not wake.is_set():
                logger.warning("card admission: %s waited %.1f s (max_wait_s) with %.2f GiB free < %.2f GiB; "
                               "dispatching anyway", stage, waited / 1e9, free, need_gb)
            return waited
        if not waited:
            logger.info("card admission: %s waits for %.2f GiB free on the card (%.2f now)", stage, need_gb, free)
        waited += wait_ns(wake, min(cfg.poll_s, (limit - waited) / 1e9))
