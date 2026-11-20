"""Stage-wave batched orchestration over a `ClipManifest`: `run_manifest`'s semantics, with cross-clip batches.

`run_manifest_batched` takes the clips `run_manifest` would run, in manifest order, in windows of
`max_clips_per_wave`. Inside a window it walks `engine.stages` in order; each stage collects its ready clips
(`ready_clips`, the walk of `run_clip`), groups them and runs every group before the next stage starts. A plain
`StageFn` runs one clip per batch; a `BatchedStage` gets `dynamic_batches` groups under its `StageBatchConfig`.

One attempt holds `gpu_lock(<artifact_root>/.gpu.lock)` for a GPU stage and is timed with `perf_counter_ns` (CUDA
peak reset before, `synchronize` before the clock stops). Its wall is apportioned to its clips by frames
(`fastdo.engine.apportion.apportion_ns`). An `Exception` fails the whole group; a `BaseException` propagates with
nothing logged or marked. After an attempt on several clips each failed clip is retried once, alone
(`retry_singly`). Commit order per attempt: the `BatchRecord` is appended (fsync), final clips are marked with their
`batch_id`, and the manifest is saved once. A kill inside a batch therefore loses only that batch.

`prefetch_decode` (opt-in): when the first stage is a CPU-only `decode`, wave k+1's `decode` groups run on ONE
background thread while wave k runs its later stages, so decode leaves the critical path. The thread only calls the
stage and times it (`perf_counter_ns`; no GPU lock and no CUDA peak, whose counters belong to the main thread's stage);
every commit happens on the main thread when wave k+1 reaches `decode`, in `_attempt`/`_commit` order with their content
and batch ids. A retry that shifts the batch ids makes every later group rerun there, and a `BaseException` from the
thread is re-raised there once the groups before it are committed. The recorded decode wall is the prefetch wall, which
overlapped other stages: it counts in `wall_ns_per_stage` and `overlapped_ns`, not in `overhead_ns`. At most one wave
is prefetched, and only the clips `ready_clips` selects for it.

`pipeline_waves` (opt-in): the waves run through `fastdo.engine.pipelined_runner` instead, where each resource
class (in-process `gpu`, a worker subprocess, a CPU stage) has one thread and up to `pipeline_depth` waves are in
flight, so wave k+1's stages overlap wave k's on other classes. Per-stage batch ids and artifacts are unchanged; the
run holds the GPU lock once for its whole duration, and `prefetch_decode` is subsumed (decode is a CPU class that runs
ahead). With the flag off, as by default, nothing here changes.

`memory_ledger` (opt-in, `pipeline_waves` only): a `MemoryLedgerConfig` gives the pipelined runner a
declared-cost device-memory ledger (`fastdo.engine.pipelined_runner.MemoryLedger`) that in-process CUDA attempts
reserve their group's `StageBatchConfig.declared_cost_gb` from. The sequential runner never reads it: it already runs
one GPU attempt at a time.

`card_admission` and `oom_retry` (opt-in, both runners; `fastdo.engine.card_admission`): an attempt of a listed
stage waits for the card's free memory before it starts, and a clip that failed with a CUDA OOM is retried alone with
exponential backoff until its wait budget is spent, extending the single retry below. With both None, as by default,
nothing here changes.

`prewarm` (opt-in, `pipeline_waves` only; `fastdo.engine.prewarm`): the listed stages' `prewarm()` hooks start on
background threads when the pipelined run starts. The sequential runner never reads it. With `()`, as by default,
nothing here changes.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import zipfile
from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

from fastdo.bench import gpu_budget
from fastdo.bench.profiler import Profiler, _cuda_peak_gb
from fastdo.engine import card_admission
from fastdo.engine.apportion import (
    AdmissionBatchRecord,
    BatchRecord,
    append_batch_records,
    apportion_ns,
    read_batch_log,
)
from fastdo.engine.batched_stages import (
    BatchedStage,
    BatchOutcome,
    HandsBatchStage,
    SolveBatchStage,
)
from fastdo.engine.batching import dynamic_batches
from fastdo.engine.card_admission import CardAdmissionConfig, OomRetryConfig
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import STAGE_GPU, EngineConfig, StageFn, gpu_lock, stage_uses_gpu
from fastdo.engine.worker_support import read_frame_idx

__all__ = ["BATCH_LOG", "HOST_CAP_GB", "BatchRunReport", "BatchedEngineConfig", "CardAdmissionConfig",
           "MemoryLedgerConfig", "OomRetryConfig", "StageBatchConfig", "StageBudgetError", "applied_cap_gb",
           "check_stage_budgets", "clip_frames", "ready_clips", "run_manifest_batched"]

logger = logging.getLogger(__name__)

BATCH_LOG = "batches.jsonl"

# Per-process device cap of this shared H100 host, and the cap every batching measurement was taken under
# (`gpu_mem_cap_gb` in outputs/bench/d14_rr_{single_r1,single_r2,batched}/yield_report.json). A `bench.batched`
# block that declares its own `cap_gb` overrides this; it is only the fallback for a block that omits one. Either
# way it is a DECLARED cap: `check_stage_budgets` validates against the lower of it and `applied_cap_gb()`, the cap
# this process really installs.
HOST_CAP_GB = 21.0


def applied_cap_gb() -> float | None:
    """The per-process device cap this process actually installs, resolved exactly as `apply_gpu_budget` resolves it.

    `FASTDO_GPU_MEM_GB`, else `gpu_budget.DEFAULT_GB` (12.0) -- never the `bench.batched.cap_gb` a config declares.
    None when the value is non-positive, which is `apply_gpu_budget`'s opt-out: no cap is installed, so there is no
    applied cap to validate against.

    Raises:
        ValueError: When the environment variable is not a number (as `apply_gpu_budget` would).
    """
    gb = float(os.environ.get(gpu_budget.ENV_VAR, gpu_budget.DEFAULT_GB))
    return None if gb <= 0.0 else gb


class StageBudgetError(ValueError):
    """A stage's batch memory does not fit the per-process device cap in force, or it declares none at all.

    Raised for a declared `budget_gb` or planned peak above the cap, and for a stage that batches (`max_batch > 1`)
    while declaring neither `budget_gb` nor `fixed_cost_gb`, which no cap can bind.

    A `ValueError`, so every caller that already refuses a malformed `bench.batched` block keeps refusing this one.
    """


@dataclass(frozen=True)
class StageBatchConfig:
    """Grouping of one batched stage: `budget_gb` None reads `FASTDO_GPU_MEM_GB` (else 12.0) at call time.

    `fixed_cost_gb` and `cost_per_frame_gb` are PER-CLIP device costs. `dynamic_batches` sums them over a candidate
    group and closes the group when the sum would pass `budget_gb`, so a group's declared cost is that sum, never
    either term alone: a batch of 2 under `fixed_cost_gb: 4.09` declares 8.18 GiB, not 4.09.

    `unmeasured` marks a stage no measurement covers. Such a stage may not declare `max_batch > 1`, so promoting it
    has to be an explicit edit that drops the flag, never a silent side effect of raising a cap.
    """

    max_batch: int = 1
    budget_gb: float | None = None
    cost_per_frame_gb: float = 0.0
    fixed_cost_gb: float = 0.0
    unmeasured: bool = False

    def __post_init__(self) -> None:
        if int(self.max_batch) != self.max_batch or self.max_batch < 1:
            raise ValueError(f"max_batch must be an integer >= 1, got {self.max_batch}")
        if self.budget_gb is not None and not (math.isfinite(self.budget_gb) and self.budget_gb > 0.0):
            raise ValueError(f"budget_gb must be finite and > 0 when set, got {self.budget_gb}")
        for name in ("cost_per_frame_gb", "fixed_cost_gb"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        if self.unmeasured and self.max_batch > 1:
            raise ValueError(f"an unmeasured stage may not declare max_batch {self.max_batch}: no measurement covers "
                             "a batch above 1 for it. Measure it first, then drop `unmeasured` in the same edit")

    def resolved_budget_gb(self) -> float:
        if self.budget_gb is not None:
            return float(self.budget_gb)
        return float(os.environ.get(gpu_budget.ENV_VAR, gpu_budget.DEFAULT_GB))

    def planned_peak_gb(self) -> float:
        """Frame-independent device GiB of the largest group this config plans: `max_batch * fixed_cost_gb`.

        A lower bound on a real group's declared cost whenever `cost_per_frame_gb` is non-zero, which is why
        `check_stage_budgets` checks `budget_gb` against the cap as well.
        """
        return float(self.max_batch) * float(self.fixed_cost_gb)

    def clip_cost_gb(self, frames: int) -> float:
        """One clip's declared device GiB, `fixed_cost_gb + cost_per_frame_gb * frames`: the `cost_fn` `_groups`
        hands `dynamic_batches`."""
        return self.fixed_cost_gb + self.cost_per_frame_gb * frames

    def declared_cost_gb(self, frames: Sequence[int]) -> float:
        """A group's declared device GiB, for clips of `frames` decode rows each.

        When `fixed_cost_gb` or `cost_per_frame_gb` is set: the sum of `clip_cost_gb` over the group, accumulated in
        order from 0.0 exactly as `dynamic_batches` accumulates the running cost it closes the group on. Else the
        declared `budget_gb` (a stage that states only a whole-group figure). Else 0.0: the stage declares nothing,
        and `resolved_budget_gb`'s fallback to the process cap is not a declaration.
        """
        if self.fixed_cost_gb or self.cost_per_frame_gb:
            total = 0.0
            for n in frames:
                total += self.clip_cost_gb(n)
            return total
        return 0.0 if self.budget_gb is None else float(self.budget_gb)


def check_stage_budgets(stage_batches: Mapping[str, StageBatchConfig], cap_gb: float) -> None:
    """Refuses any stage whose declared batch memory passes the per-process device cap actually in force.

    The cap in force is the LOWER of `cap_gb` -- the block's declared `bench.batched.cap_gb`, the cap the evidence was
    measured under -- and `applied_cap_gb()`, the cap `apply_gpu_budget` installs in this process from
    `FASTDO_GPU_MEM_GB`. Checking only the declared one certifies plans the process cannot allocate (one such plan was a
    solver OOM under `FASTDO_GPU_MEM_GB=20`); checking only the applied one would let a config whose own measurement
    basis is wrong pass on a roomier host. Neither is preferred silently: a disagreement is logged, and every refusal
    names both caps.

    A stage that batches (`max_batch > 1`) and declares NEITHER `budget_gb` nor `fixed_cost_gb` is itself an error:
    `planned_peak_gb()` is then 0.0 and `resolved_budget_gb()` falls back to the process cap, so no declared figure
    bounds the group and this check would pass it at any batch size -- the exact defect shape the guard exists to
    catch. `cost_per_frame_gb` alone does not declare it either, `planned_peak_gb()` being frame-independent. A stage
    held at `max_batch: 1` declares nothing because it batches nothing: that is the deliberately unmeasured `encoder`,
    and it keeps passing.

    Otherwise both the declared `budget_gb` (what `dynamic_batches` may fill) and `planned_peak_gb()` (what `max_batch`
    alone already commits to) must fit, so neither a raised budget nor a raised batch size can quietly plan a group the
    process cannot allocate.

    Raises:
        StageBudgetError: Naming the stage, the offending figure and the cap in force.
    """
    if not (math.isfinite(cap_gb) and cap_gb > 0.0):
        raise ValueError(f"cap_gb must be finite and > 0, got {cap_gb}")
    applied = applied_cap_gb()
    effective = cap_gb if applied is None else min(cap_gb, applied)
    if applied is not None and applied != cap_gb:
        logger.warning("bench.batched.cap_gb %s disagrees with the %s GiB this process applies from %s: every stage "
                       "budget is validated against the lower, %s GiB", cap_gb, applied, gpu_budget.ENV_VAR, effective)
        source = (f"the lower of bench.batched.cap_gb {cap_gb} and the {applied} GiB this process applies from "
                  f"{gpu_budget.ENV_VAR}")
    else:
        source = f"bench.batched.cap_gb {cap_gb}"
    for stage, sb in sorted(stage_batches.items()):
        if sb.max_batch > 1 and sb.budget_gb is None and sb.fixed_cost_gb == 0.0:
            raise StageBudgetError(f"stage {stage!r} declares max_batch {sb.max_batch} with neither budget_gb nor "
                                   "fixed_cost_gb: its planned peak is 0.0 GiB and its budget falls back to the "
                                   "process cap, so no declared figure bounds the group and no cap can refuse it at "
                                   "any batch size. Declare the measured budget_gb and fixed_cost_gb, or hold the "
                                   "stage at max_batch 1 (`unmeasured: true` when no measurement covers it)")
        if sb.budget_gb is not None and sb.budget_gb > effective:
            raise StageBudgetError(f"stage {stage!r} declares budget_gb {sb.budget_gb}, above the {effective} GiB "
                                   f"per-process device cap ({source}): lower it, or run under a cap the evidence was "
                                   "actually measured under")
        planned = sb.planned_peak_gb()
        if planned > effective:
            raise StageBudgetError(f"stage {stage!r} plans max_batch {sb.max_batch} x fixed_cost_gb "
                                   f"{sb.fixed_cost_gb} = {planned:.2f} GiB, above the {effective} GiB per-process "
                                   f"device cap ({source})")


def _is_gb(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0


@dataclass(frozen=True)
class MemoryLedgerConfig:
    """`bench.batched.memory_ledger`: the device GiB outside any attempt's declared cost.

    `resident_gb` is what the process keeps on the card between attempts (loaded weights, the allocator's cache), fixed
    from a measured floor (`torch.cuda.memory_allocated` after warm-up). The ledger's capacity is the
    applied per-process cap minus it, read when the runner starts (`capacity_gb`).

    `costs_gb`: `stage -> GiB` of ONE attempt (the whole group, whatever its size), which REPLACES that stage's
    `StageBatchConfig.declared_cost_gb` in the ledger. The stage declarations under-state Ego4D (tracks declares
    4.09/clip, a 2-clip attempt measured ~18.7 GiB above resident; solve 2.23/clip, a 5-clip attempt ~19.2), so this
    key fixes these from measured peaks. A stage it does not list keeps its declared cost. Keys must be `STAGE_GPU`
    stages, values finite and >= 0; a value above the capacity is allowed (that stage runs alone). Stored read-only.
    """

    resident_gb: float
    costs_gb: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_gb(self.resident_gb):
            raise ValueError(f"bench.batched.memory_ledger.resident_gb must be a finite number >= 0, got "
                             f"{self.resident_gb!r}")
        if not isinstance(self.costs_gb, Mapping):      # a ValueError, as every other bench.batched refusal
            raise ValueError(f"bench.batched.memory_ledger.costs_gb must map stage -> GiB, got "  # noqa: TRY004
                             f"{self.costs_gb!r}")
        unknown = sorted(str(stage) for stage in self.costs_gb if stage not in STAGE_GPU)
        if unknown:
            raise ValueError(f"bench.batched.memory_ledger.costs_gb has unknown stage keys {unknown}; valid keys: "
                             f"{sorted(STAGE_GPU)}")
        bad = {stage: gb for stage, gb in self.costs_gb.items() if not _is_gb(gb)}
        if bad:
            raise ValueError(f"bench.batched.memory_ledger.costs_gb values must be finite numbers >= 0, got {bad}")
        object.__setattr__(self, "costs_gb", MappingProxyType({s: float(gb) for s, gb in self.costs_gb.items()}))

    def attempt_cost_gb(self, stage: str, declared_gb: float) -> float:
        """The ledger cost of one attempt of `stage`: `costs_gb[stage]` when listed, else `declared_gb`."""
        return self.costs_gb.get(stage, declared_gb)

    def capacity_gb(self) -> float:
        """`applied_cap_gb() - resident_gb`.

        Raises:
            ValueError: When no cap is applied (`FASTDO_GPU_MEM_GB` <= 0): the ledger has nothing to subtract from.
                Or when `resident_gb` is not below the applied cap, which leaves no capacity for any attempt.
        """
        cap = applied_cap_gb()
        if cap is None:
            raise ValueError(f"bench.batched.memory_ledger needs the per-process device cap, but {gpu_budget.ENV_VAR} "
                             f"= {os.environ.get(gpu_budget.ENV_VAR)!r} disables it: set a positive cap or set "
                             "memory_ledger to null")
        if self.resident_gb >= cap:
            raise ValueError(f"bench.batched.memory_ledger.resident_gb {self.resident_gb} must be below the {cap} GiB "
                             f"per-process device cap this process applies ({gpu_budget.ENV_VAR})")
        return cap - float(self.resident_gb)


@dataclass(frozen=True)
class BatchedEngineConfig:
    """`engine` plus per-stage grouping, the wave size, the single-retry switch, the log path, the run id, the
    decode prefetch switch and the wave-pipelining switch (module docstring).

    `pipeline_depth` caps the waves started and not finished while `pipeline_waves` is on (decode.npz is 0.6-1.5 GB a
    window). `pipeline_resources` names a stage's resource class; a stage it does not name goes to `gpu` when
    `stage_uses_gpu`, else to a class of its own (`fastdo.engine.pipelined_runner.stage_classes`). `memory_ledger`
    turns on the pipelined runner's declared-cost ledger; None, the default, leaves every path as it was, and
    the sequential runner ignores it. `card_admission` and `oom_retry` (`fastdo.engine.card_admission`) turn on
    card-level admission and the patient OOM retry in both runners; None, the default, leaves every path as it was.
    """

    engine: EngineConfig
    stage_batches: Mapping[str, StageBatchConfig] = field(default_factory=dict)
    max_clips_per_wave: int | None = None
    retry_singly: bool = True
    batch_log: Path | None = None
    run_id: str | None = None
    prefetch_decode: bool = False
    pipeline_waves: bool = False
    pipeline_depth: int = 2
    pipeline_resources: Mapping[str, str] = field(default_factory=dict)
    memory_ledger: MemoryLedgerConfig | None = None
    card_admission: CardAdmissionConfig | None = None
    oom_retry: OomRetryConfig | None = None
    prewarm: tuple[str, ...] = ()        # stages whose prewarm() the pipelined runner starts at run start

    def __post_init__(self) -> None:
        if self.max_clips_per_wave is not None and self.max_clips_per_wave < 1:
            raise ValueError(f"max_clips_per_wave must be >= 1 when set, got {self.max_clips_per_wave}")
        if isinstance(self.pipeline_depth, bool) or int(self.pipeline_depth) != self.pipeline_depth \
                or self.pipeline_depth < 1:
            raise ValueError(f"pipeline_depth must be an integer >= 1, got {self.pipeline_depth!r}")
        for stage, cls in self.pipeline_resources.items():
            if not (isinstance(cls, str) and cls):
                raise ValueError(f"pipeline_resources[{stage!r}] must be a non-empty class name, got {cls!r}")
        if self.memory_ledger is not None and not isinstance(self.memory_ledger, MemoryLedgerConfig):
            raise ValueError(f"memory_ledger must be a MemoryLedgerConfig or None, got {self.memory_ledger!r}")
        if self.card_admission is not None and not isinstance(self.card_admission, CardAdmissionConfig):
            raise ValueError(f"card_admission must be a CardAdmissionConfig or None, got {self.card_admission!r}")
        if self.oom_retry is not None and not isinstance(self.oom_retry, OomRetryConfig):
            raise ValueError(f"oom_retry must be an OomRetryConfig or None, got {self.oom_retry!r}")
        prewarm = tuple(self.prewarm)
        if any(not isinstance(stage, str) or not stage for stage in prewarm) or len(set(prewarm)) != len(prewarm):
            raise ValueError(f"prewarm must be distinct stage names, got {self.prewarm!r}")
        object.__setattr__(self, "prewarm", prewarm)

    def batch_config(self, stage: str) -> StageBatchConfig:
        """`stage_batches[stage]`, or `StageBatchConfig()` for a stage it does not list."""
        return self.stage_batches.get(stage, StageBatchConfig())

    def resolved_batch_log(self) -> Path:
        if self.batch_log is not None:
            return Path(self.batch_log)
        if self.engine.manifest_path is not None:
            return self.engine.manifest_path.with_name(BATCH_LOG)
        return self.engine.artifact_root / BATCH_LOG


@dataclass(frozen=True)
class BatchRunReport:
    """`n_batches` counts every attempt (retries included); `overhead_ns` is the runner's wall minus the attempts'.
    `solve_batched` is None when no `solve` batch ran, else whether every one of them batched its compute.
    `overlapped_ns` is the prefetched decode wall: in `wall_ns_per_stage`, but off the critical path. With
    `pipeline_waves` it is the stage wall that ran concurrently, the attempts' sum minus the run's wall floored
    at 0, and `overhead_ns` is floored at 0 as well."""

    run_id: str
    batch_log: Path
    n_batches: int
    n_retries: int
    wall_ns_per_stage: dict[str, int]
    overhead_ns: int
    solve_batched: bool | None
    overlapped_ns: int = 0


@dataclass(frozen=True)
class _Timed:
    """One attempt's call and clock; `_record` turns it into the `BatchRecord`."""

    batch_id: str
    started: str
    outcome: BatchOutcome
    wall_ns: int


@dataclass
class _Prefetch:
    """Wave k+1's decode: the thread appends one `_Timed` per group in order and stops at a `BaseException`."""

    clip_ids: list[str]
    ready: list[str]
    frames: dict[str, int]
    groups: list[list[str]]
    base_seq: int
    thread: threading.Thread | None = None
    results: list[_Timed] = field(default_factory=list)
    interrupt: BaseException | None = None


def clip_frames(manifest: ClipManifest, artifact_root: Path, clip_id: str) -> int:
    """Decode rows of `<artifact_root>/<clip_id>/decode.npz` when it exists and reads, else `manifest.n_frames`."""
    artifact_dir = Path(artifact_root) / clip_id
    if (artifact_dir / "decode.npz").is_file():
        if (artifact_dir / "decode.npz").stat().st_size == 0:     # a placeholder worker's (--stub) empty artifact
            logger.debug("decode.npz of %r is an empty placeholder; weighting it by n_frames", clip_id)
            return manifest.n_frames(clip_id)
        try:
            return int(read_frame_idx(artifact_dir).shape[0])
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            logger.warning("could not read the decode rows of %r (%s); weighting it by n_frames", clip_id, exc)
    return manifest.n_frames(clip_id)


def ready_clips(manifest: ClipManifest, engine: EngineConfig, stage: str, registered: Collection[str],
                clip_ids: Sequence[str]) -> list[str]:
    """Clips of `clip_ids` that `run_clip` would run `stage` on now: not skipped as done, and every earlier
    registered stage of `engine.stages` done (an unregistered stage blocks nothing)."""
    earlier = [s for s in engine.stages[:engine.stages.index(stage)] if s in registered]
    return [c for c in clip_ids
            if not (engine.skip_done and manifest.status(c, stage).state == "done")
            and all(manifest.status(c, s).state == "done" for s in earlier)]


class _Runner:
    def __init__(self, manifest: ClipManifest, cfg: BatchedEngineConfig,
                 stages: Mapping[str, StageFn | BatchedStage]) -> None:
        self.manifest, self.cfg, self.stages, self.engine = manifest, cfg, stages, cfg.engine
        self.log = cfg.resolved_batch_log()
        self.run_id = _unused_run_id(cfg.run_id, {record.run_id for record in read_batch_log(self.log)})
        self.lock_path = self.engine.artifact_root / ".gpu.lock"
        self.seq: dict[str, int] = {}
        self.walls: dict[str, int] = {}
        self.n_batches = self.n_retries = 0
        self.solve_batched: bool | None = None
        self.warned: set[str] = set()
        self.overlapped_ns = 0
        self.pending: _Prefetch | None = None
        self.prefetch = self._prefetch_enabled()
        self._wake = threading.Event()      # set by a halt (the pipelined runner) to cut a card-admission wait short
        self._card_need = self._card_thresholds()

    def _card_thresholds(self) -> dict[str, tuple[float, bool]]:
        """`stage -> (min_free_gb, in_process)` for every stage of `engine.stages` that `card_admission` lists by
        its own name or its resource class's; `{}` when it is off. `in_process` picks the card reading
        (`card_admission.card_free_gb`)."""
        admission = self.cfg.card_admission
        if admission is None:
            return {}
        # imported here: pipelined_runner imports this module
        from fastdo.engine.pipelined_runner import runs_cuda_in_process, stage_classes

        classes = stage_classes(self.engine.stages, self.cfg.pipeline_resources)
        need: dict[str, tuple[float, bool]] = {}
        for stage, cls in classes.items():
            gb = admission.threshold_gb(stage, cls)
            if gb is not None:
                need[stage] = (gb, runs_cuda_in_process(stage, cls))
        unused = sorted(set(admission.min_free_gb) - set(classes) - set(classes.values()))
        if unused:
            logger.warning("card_admission: min_free_gb names %s, no stage or class of this run; ignored", unused)
        logger.info("card admission: %s, poll %.1f s, give up after %.1f s",
                    {s: f"{gb:.2f} GiB {'in-process' if inp else 'card-free'}" for s, (gb, inp) in need.items()},
                    admission.poll_s, admission.max_wait_s)
        return need

    def _prefetch_enabled(self) -> bool:
        if not self.cfg.prefetch_decode:
            return False
        if not self.engine.stages or self.engine.stages[0] != "decode" or "decode" not in self.stages:
            logger.warning("prefetch_decode is set, but the first stage is not a registered 'decode' (stages %s): "
                           "running without it", list(self.engine.stages))
            return False
        if stage_uses_gpu("decode"):
            raise ValueError("prefetch_decode: 'decode' takes the GPU lock, so it may not run beside a GPU stage")
        return True

    def _waves(self) -> list[list[str]]:
        """The clips to run, in manifest order, in windows of `max_clips_per_wave`."""
        candidates = [c for c in self.manifest.clip_ids()
                      if any(self.manifest.status(c, s).state != "done" for s in self.engine.stages)]
        if not self.engine.retry_refusals:
            # Same policy as `run_manifest`: a stage's VERDICT is not work to redo, a fault is. Applied
            # here too, or the batched path keeps paying for refusals the sequential path stopped paying.
            refused = self.manifest.refused(self.engine.stages)
            if refused:
                logger.info("skipping %d clip(s) a stage already refused", len(refused))
            candidates = [c for c in candidates if c not in refused]
        size = self.cfg.max_clips_per_wave or max(len(candidates), 1)
        return [candidates[first:first + size] for first in range(0, len(candidates), size)]

    def run(self) -> BatchRunReport:
        start = time.perf_counter_ns()
        waves = self._waves()
        try:
            for k, wave in enumerate(waves):
                self._window(wave, waves[k + 1] if k + 1 < len(waves) else [])
        finally:
            if self.pending is not None and self.pending.thread is not None:
                self.pending.thread.join()      # the main thread failed: let the decode finish, commit none of it
        overhead = time.perf_counter_ns() - start - (sum(self.walls.values()) - self.overlapped_ns)
        return BatchRunReport(self.run_id, self.log, self.n_batches, self.n_retries, dict(self.walls), overhead,
                              self.solve_batched, self.overlapped_ns)

    def _window(self, clip_ids: list[str], next_ids: list[str]) -> None:
        root, engine = self.engine.artifact_root, self.engine
        profilers = {c: Profiler(c, engine.system, self.manifest.n_frames(c), engine.fps) for c in clip_ids}
        for clip_id in clip_ids:
            (root / clip_id).mkdir(parents=True, exist_ok=True)
        prefetched, self.pending = self.pending, None
        for stage in engine.stages:
            if stage not in self.stages:
                if stage not in self.warned:
                    self.warned.add(stage)
                    logger.warning("no stage worker registered for %r; leaving it pending", stage)
                continue
            if prefetched is not None and stage == "decode":
                self._adopt_prefetch(prefetched, clip_ids, profilers)
            else:
                ready = ready_clips(self.manifest, engine, stage, self.stages.keys(), clip_ids)
                if ready:
                    self._stage(stage, ready, profilers)
            if self.prefetch and stage == "decode" and next_ids:
                self.pending = self._launch_prefetch(next_ids)
        for clip_id in clip_ids:
            profilers[clip_id].report().to_json(root / clip_id / "profile.json")

    def _launch_prefetch(self, clip_ids: list[str]) -> _Prefetch | None:
        """Starts the decode thread on the next wave's `ready_clips`, with the frames and groups `_stage` would take
        now, before any of their `decode.npz` is written; None when no clip is ready."""
        ready = ready_clips(self.manifest, self.engine, "decode", self.stages.keys(), clip_ids)
        if not ready:
            return None
        for clip_id in clip_ids:
            (self.engine.artifact_root / clip_id).mkdir(parents=True, exist_ok=True)
        frames, groups = self._groups("decode", ready)
        pf = _Prefetch(clip_ids, ready, frames, groups, self.seq.get("decode", 0))
        pf.thread = threading.Thread(target=self._prefetch_decode, args=(pf,), name="fastdo-decode-prefetch")
        pf.thread.start()
        return pf

    def _prefetch_decode(self, pf: _Prefetch) -> None:
        """The thread: calls and times each group as `_attempt` does; touches no manifest, log, profiler or seq."""
        impl = self.stages["decode"]
        try:
            for i, group in enumerate(pf.groups):
                batch_id = self._batch_id("decode", pf.base_seq + i)
                started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                t0 = time.perf_counter_ns()
                outcome = self._unit_call(impl, group, batch_id)
                pf.results.append(_Timed(batch_id, started, outcome, time.perf_counter_ns() - t0))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread by `_adopt_prefetch`
            pf.interrupt = exc

    def _adopt_prefetch(self, pf: _Prefetch, clip_ids: list[str], profilers: Mapping[str, Profiler]) -> None:
        """Joins the thread and commits its groups exactly as `_stage` would have run them (module docstring)."""
        assert pf.thread is not None
        pf.thread.join()
        ready = ready_clips(self.manifest, self.engine, "decode", self.stages.keys(), clip_ids)
        if pf.clip_ids != clip_ids or pf.ready != ready:
            raise RuntimeError(f"decode prefetch ran {pf.ready} of wave {pf.clip_ids}, but wave {clip_ids} is ready "
                               f"for {ready}")
        impl = self.stages["decode"]
        for i, group in enumerate(pf.groups):
            if i == len(pf.results):
                assert pf.interrupt is not None
                raise pf.interrupt
            adopt = self.seq.get("decode", 0) == pf.base_seq + i    # False once a retry has taken this batch id
            self._group("decode", impl, group, pf.frames, profilers, pf.results[i] if adopt else None)
        if pf.interrupt is not None:
            raise pf.interrupt

    def _groups(self, stage: str, ready: list[str]) -> tuple[dict[str, int], list[list[str]]]:
        impl = self.stages[stage]
        frames = {c: clip_frames(self.manifest, self.engine.artifact_root, c) for c in ready}
        if isinstance(impl, BatchedStage):
            sb = self.cfg.batch_config(stage)
            cap = min(sb.max_batch, impl.hands_cfg.max_clips_per_call) if isinstance(impl, HandsBatchStage) \
                else sb.max_batch
            groups = dynamic_batches(ready, cap, sb.resolved_budget_gb(),
                                     lambda c: sb.clip_cost_gb(frames[c]))
        else:
            groups = [[c] for c in ready]
        return frames, groups

    def _stage(self, stage: str, ready: list[str], profilers: Mapping[str, Profiler]) -> None:
        frames, groups = self._groups(stage, ready)
        for group in groups:
            self._group(stage, self.stages[stage], group, frames, profilers)

    def _group(self, stage: str, impl: StageFn | BatchedStage, group: list[str], frames: Mapping[str, int],
               profilers: Mapping[str, Profiler], done: _Timed | None = None) -> None:
        """One group's first attempt -- run here, or `done` on the prefetch thread -- then its single retries."""
        totals: dict[str, int] = dict.fromkeys(group, 0)
        attempts: dict[str, list[str]] = {c: [] for c in group}
        if done is None:
            first, errors = self._attempt(stage, impl, group, frames, "", totals, attempts)
        else:
            if self._next_batch_id(stage) != done.batch_id:
                raise RuntimeError(f"prefetched batch {done.batch_id} is out of order")
            self.overlapped_ns += done.wall_ns
            first, errors = self._record(stage, group, frames, "", totals, attempts, done, 0.0)
        if self.cfg.oom_retry is not None:
            self._group_patiently(stage, impl, group, frames, profilers, totals, attempts, first, errors)
            return
        retry = [c for c in group if errors.get(c)] if len(group) > 1 and self.cfg.retry_singly else []
        self._commit(first, [c for c in group if c not in retry], errors, frames, profilers, totals, attempts)
        for clip_id in retry:
            record, errors = self._attempt(stage, impl, [clip_id], frames, first.batch_id, totals, attempts)
            self._commit(record, [clip_id], errors, frames, profilers, totals, attempts)

    def _group_patiently(self, stage: str, impl: StageFn | BatchedStage, group: list[str], frames: Mapping[str, int],
                         profilers: Mapping[str, Profiler], totals: dict[str, int], attempts: dict[str, list[str]],
                         first: BatchRecord, errors: Mapping[str, str]) -> None:
        """`_group`'s retries with `oom_retry` on (`fastdo.engine.card_admission`): today's single retries, plus
        a CUDA-OOM clip retried alone (one-clip groups too) with backoff until its wait budget is spent. A retry that
        will be retried again is committed with no final clip, so the manifest is marked once, by the chain's last."""
        policy = self.cfg.oom_retry
        assert policy is not None
        budget_ns = round(policy.max_total_wait_s * 1e9)
        single = len(group) > 1 and self.cfg.retry_singly
        retry = [c for c in group if errors.get(c) and (single or card_admission.is_cuda_oom(errors[c]))]
        first = AdmissionBatchRecord.with_waits(first, oom_backoff_ns=0)
        self._commit(first, [c for c in group if c not in retry], errors, frames, profilers, totals, attempts)
        for clip_id in retry:
            previous, waited_ns, n_backoffs = first, 0, 0
            backoff = len(group) == 1        # the same workload again: wait first. After a group: at once, as today
            while True:
                slept = 0
                if backoff:
                    delay = min(policy.delay_s(n_backoffs), (budget_ns - waited_ns) / 1e9)
                    logger.warning("clip %r stage %r: CUDA OOM in batch %s; retrying alone in %.1f s (waited %.1f of "
                                   "%.1f s)", clip_id, stage, previous.batch_id, delay, waited_ns / 1e9,
                                   policy.max_total_wait_s)
                    slept = card_admission.wait_ns(self._wake, delay)
                    n_backoffs += 1
                attempt, clip_errors = self._attempt(stage, impl, [clip_id], frames, previous.batch_id, totals,
                                                     attempts)
                record = AdmissionBatchRecord.with_waits(attempt, oom_backoff_ns=slept)
                waited_ns += slept + (record.card_wait_ns or 0)
                oom = card_admission.is_cuda_oom(clip_errors[clip_id])
                again = oom and waited_ns < budget_ns
                if oom and not again:
                    logger.warning("clip %r stage %r: CUDA OOM again in batch %s after %.1f s of waiting "
                                   "(max_total_wait_s %.1f); giving up", clip_id, stage, record.batch_id,
                                   waited_ns / 1e9, policy.max_total_wait_s)
                self._commit(record, [] if again else [clip_id], clip_errors, frames, profilers, totals, attempts)
                if not again:
                    break
                previous, backoff = record, True

    def _batch_id(self, stage: str, seq: int) -> str:
        return f"{self.run_id}-{stage}-{seq:05d}"

    def _next_batch_id(self, stage: str) -> str:
        seq = self.seq.get(stage, 0)
        self.seq[stage] = seq + 1
        return self._batch_id(stage, seq)

    def _unit_call(self, impl: StageFn | BatchedStage, group: list[str], batch_id: str) -> BatchOutcome:
        """`_call`, an `Exception` failing every clip of the group; a `BaseException` propagates."""
        try:
            return self._call(impl, [(c, self.engine.artifact_root / c) for c in group], batch_id)
        except Exception as exc:  # noqa: BLE001 - a unit failure: every clip of the group fails with it
            logger.warning("batch %s (%d clips) failed as a unit: %s", batch_id, len(group), exc)
            return BatchOutcome(errors=dict.fromkeys(group, str(exc)), batched=_is_batched(impl))

    def _attempt(self, stage: str, impl: StageFn | BatchedStage, group: list[str], frames: Mapping[str, int],
                 retry_of: str, totals: dict[str, int], attempts: dict[str, list[str]],
                 ) -> tuple[BatchRecord, dict[str, str]]:
        batch_id = self._next_batch_id(stage)
        card_ns = self._card_wait(stage)
        started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        lock: AbstractContextManager[None] = gpu_lock(self.lock_path) if stage_uses_gpu(stage) else nullcontext()
        with lock:
            _cuda_peak_gb(reset=True)
            t0 = time.perf_counter_ns()
            outcome = self._unit_call(impl, group, batch_id)
            device_peak = _cuda_peak_gb(reset=False)
            wall_ns = time.perf_counter_ns() - t0
        record, errors = self._record(stage, group, frames, retry_of, totals, attempts,
                                      _Timed(batch_id, started, outcome, wall_ns), device_peak)
        return self._with_card_wait(record, card_ns), errors

    def _card_wait(self, stage: str) -> int | None:
        """The ns this attempt of `stage` waited for card admission; None when `stage` is not gated (always,
        with `card_admission` off). Called after the batch id is taken and before the clock starts."""
        if stage not in self._card_need:
            return None
        assert self.cfg.card_admission is not None
        need_gb, in_process = self._card_need[stage]
        return card_admission.wait_for_card(stage, need_gb, in_process, self.cfg.card_admission, self._wake)

    @staticmethod
    def _with_card_wait(record: BatchRecord, card_ns: int | None) -> BatchRecord:
        """`record` itself for an ungated attempt, else an `AdmissionBatchRecord` carrying `card_wait_ns`."""
        return record if card_ns is None else AdmissionBatchRecord.with_waits(record, card_wait_ns=card_ns)

    def _record(self, stage: str, group: list[str], frames: Mapping[str, int], retry_of: str,
                totals: dict[str, int], attempts: dict[str, list[str]], timed: _Timed, device_peak: float,
                ) -> tuple[BatchRecord, dict[str, str]]:
        """The attempt's `BatchRecord` and per-clip errors; tallies its shares and batch id into `totals`/`attempts`.
        A prefetched attempt passes `device_peak` 0.0: decode allocates nothing on the card."""
        batch_id, outcome, wall_ns = timed.batch_id, timed.outcome, timed.wall_ns
        weights = [frames[c] for c in group]
        shares = apportion_ns(wall_ns, weights)
        errors = {c: outcome.errors.get(c, "") for c in group}
        for clip_id, share in zip(group, shares, strict=True):
            totals[clip_id] += share
            attempts[clip_id].append(batch_id)
            if errors[clip_id]:
                logger.warning("clip %r failed at stage %r in batch %s: %s", clip_id, stage, batch_id, errors[clip_id])
        record = BatchRecord(batch_id=batch_id, run_id=self.run_id, stage=stage, clip_ids=tuple(group),
                             weights=tuple(weights), wall_ns=wall_ns, share_ns=tuple(shares),
                             errors=tuple(errors[c] for c in group), retry_of=retry_of, batched=outcome.batched,
                             peak_gpu_mem_gb=max(device_peak, float(outcome.peak_gpu_mem_gb)),
                             started_utc=timed.started)
        return record, errors

    @staticmethod
    def _call(impl: StageFn | BatchedStage, clips: list[tuple[str, Path]], batch_id: str) -> BatchOutcome:
        if isinstance(impl, BatchedStage):
            return impl.run_batch(clips, batch_id)
        (clip_id, artifact_dir), = clips
        cast("StageFn", impl)(clip_id, artifact_dir)
        return BatchOutcome(batched=False)

    def _commit(self, record: BatchRecord, final: list[str], errors: Mapping[str, str], frames: Mapping[str, int],
                profilers: Mapping[str, Profiler], totals: Mapping[str, int],
                attempts: Mapping[str, list[str]]) -> None:
        append_batch_records(self.log, [record])
        stage = record.stage
        self.n_batches += 1
        self.n_retries += bool(record.retry_of)
        self.walls[stage] = self.walls.get(stage, 0) + record.wall_ns
        if stage == "solve":
            self.solve_batched = record.batched if self.solve_batched is None else self.solve_batched and record.batched
        for clip_id in final:
            seconds, error = totals[clip_id] / 1e9, errors[clip_id]
            self.manifest.mark(clip_id, stage, "failed" if error else "done", seconds, error=error,
                               batch_id=record.batch_id)
            profilers[clip_id].add(stage, seconds, peak_gpu_mem_gb=record.peak_gpu_mem_gb, batch_id=record.batch_id,
                                   batch_size=str(len(record.clip_ids)), frames=str(frames[clip_id]),
                                   attempts=",".join(attempts[clip_id]))
        if self.engine.manifest_path is not None:
            self.manifest.save(self.engine.manifest_path)


def _unused_run_id(requested: str | None, taken: Collection[str]) -> str:
    """`requested`, refused with `ValueError` when the batch log already holds it; else `<UTC second>-<pid>`, with
    `-1`, `-2`, ... appended while that id is taken (a resume within the same second must not reuse batch ids)."""
    if requested:
        if requested in taken:
            raise ValueError(f"run_id {requested!r} already has records in the batch log; pass a new run_id")
        return requested
    base = run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    n = 0
    while run_id in taken:
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def _is_batched(impl: StageFn | BatchedStage) -> bool:
    if isinstance(impl, SolveBatchStage):
        return impl.solve_many()[1]
    return isinstance(impl, BatchedStage)


def run_manifest_batched(manifest: ClipManifest, cfg: BatchedEngineConfig,
                         stages: Mapping[str, StageFn | BatchedStage]) -> BatchRunReport:
    """Runs every clip with a non-`done` stage of `cfg.engine.stages` in stage waves (module docstring).

    A clip's stage seconds are its frame-proportional shares of every attempt it joined, summed in integer ns and
    divided by 1e9; the same float goes into the manifest column and the clip's `profile.json` record.

    Raises:
        ValueError: When `cfg.run_id` already has records in the batch log, or the log is malformed.
        BaseException: Whatever non-`Exception` a worker raised (`SystemExit`, `KeyboardInterrupt`), unrecorded.
    """
    if cfg.pipeline_waves:
        # imported here: pipelined_runner imports this module
        from fastdo.engine.pipelined_runner import PipelinedRunner

        return PipelinedRunner(manifest, cfg, stages).run()
    return _Runner(manifest, cfg, stages).run()
