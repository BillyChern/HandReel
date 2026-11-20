"""Resumable per-clip orchestrator: walks stage workers over a `ClipManifest`.

Stage workers are injected as `dict[str, StageFn]` so this module is fully testable without any
real front-end or model code (the stage modules supply the real workers). `ClipManifest` is the single
source of truth for stage state; `run_clip` keeps no state beyond one clip's own `Profiler`, which
it constructs fresh, exactly once, per call -- a bare `ProfileReport` carries no clip identity of
its own, so it is never shared or reused across clips. Cross-clip aggregation (the design's headline
clips-per-GPU-hour metric) goes exclusively through `clips_per_gpu_hour_report`, reading the
manifest's own `{stage}_seconds` columns -- never through a shared `Profiler`/`ProfileReport`.
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from fastdo.bench.profiler import Profiler
from fastdo.bench.yield_report import stage_order
from fastdo.engine.manifest import STAGE_NAMES, ClipManifest

__all__ = [
    "STAGE_GPU",
    "EngineConfig",
    "StageFn",
    "clips_per_gpu_hour_report",
    "gpu_lock",
    "hold_gpu_lock_for_process",
    "run_clip",
    "run_manifest",
    "stage_uses_gpu",
]

logger = logging.getLogger(__name__)

StageFn = Callable[[str, Path], None]

# Stages that call into a loaded model, or allocate on the card, on the shared H100 and must therefore serialize behind
# `gpu_lock` (one heavy job at a time on the shared card). `solve` is gated: its default device is `cuda`, and the M1
# solver allocates its Sinkhorn and z-buffer tensors on the shared card. `decode` (frame I/O), `gate` (a tiny MLP) and
# `qc`/`retarget` (CPU-bound glue) are not gated. A `MappingProxyType` so nothing downstream can mutate the shared
# classification by accident.
STAGE_GPU: Mapping[str, bool] = MappingProxyType(
    {
        "decode": False,
        "encoder": True,
        "gate": False,
        "mask": True,
        "shape": True,
        "intrinsics": True,  # MoGe-2 forward on a few keyframes when the clip has no camera.json
        "hands": True,
        "geometry": True,
        "tracks": True,
        "solve": True,
        "qc": False,
        "retarget": False,
        # The opt-in stages. `prompt` IS a GPU stage: it runs Hands23 (detectron2 X-101 FPN) and SAM 2's image
        # predictor on `prompt_stage.device`, measured at 1.5 GB peak. It was once absent here, and
        # `.get(stage, False)` silently answered False, so it ran two models on the shared card outside the lock.
        # `contact` (a numpy slide over a precomputed grid) and `driftqc` (CPU glue, asserted torch-free) are
        # genuinely CPU-only.
        "prompt": True,
        "contact": False,
        "driftqc": False,
        "tba": False,          # opt-in: numpy/scipy BA + carving on stored artefacts, CPU only
    }
)


def stage_uses_gpu(stage: str) -> bool:
    """Whether `stage` must hold the GPU lock. An UNDECLARED stage is treated as a GPU stage.

    The old spelling, `STAGE_GPU.get(stage, False)`, failed OPEN: a stage nobody added to the map ran
    unlocked. That is how `prompt` came to run detectron2 and SAM 2's image predictor on the shared card
    outside the lock for its whole first outing, and nothing reported it -- an unguarded stage is silent
    until it collides with another job.

    So the default is True, which fails SAFE: the cost of over-locking a CPU-only stage is that it
    serialises against GPU work, which is slow; the cost of under-locking a GPU stage is two models on a
    shared card at once, which is a guard kill. `test_a_stage_that_never_declared_its_gpu_use_is_locked`
    pins that every stage the engine can actually run is declared, so no real stage relies on this
    fallback -- it exists to catch the one nobody thought about.
    """
    if stage in STAGE_GPU:
        return STAGE_GPU[stage]
    logger.warning("stage %r has no STAGE_GPU entry: taking the GPU lock for it (the safe default). Declare it "
                   "in fastdo.engine.orchestrator.STAGE_GPU; known stages: %s", stage, sorted(STAGE_GPU))
    return True

_GPU_LOCK_NAME = ".gpu.lock"


@dataclass(frozen=True)
class EngineConfig:
    """Static, immutable settings for one orchestrator run.

    `retry_refusals` re-runs clips a stage DECLINED on the merits (a gate rejection, a prompt refusal).
    The default `False` does not: a refusal is a deterministic function of artifacts already on disk, so
    re-running it spends the stage's seconds to obtain the identical verdict -- measured at **182.7 s**
    per resume on the 45-window Ego4D run (13 refusals, all at `prompt`). Set it to `True` after CHANGING
    a gate's rule or threshold,
    which is the one case where the old verdict is no longer the answer the code would give.

    `manifest_path`, when set, makes every stage transition durable: `run_clip` saves `manifest`
    to this path (atomic tmp + `os.replace`, per `ClipManifest.save`) right after each `.mark`
    call, so a process kill mid-batch loses at most the one stage that was in flight, never
    anything already marked. `None` (the default) keeps the manifest in-memory only, e.g. for
    tests that don't care about durability across a restart.
    """

    artifact_root: Path
    stages: tuple[str, ...] = STAGE_NAMES
    skip_done: bool = True
    fps: float = 10.0
    system: str = "engine"
    manifest_path: Path | None = None
    retry_refusals: bool = False


# Per-thread re-entrancy depth for `gpu_lock`, keyed by resolved lock-file path. `flock` is scoped
# to the open file description, not the process, so a naive nested `with gpu_lock(same_path):` in
# the *same* thread would self-deadlock (the inner `os.open` gets a distinct OFD and blocks
# forever on the outer's still-held lock). This makes nested acquisition in one thread a no-op
# past the first (outermost) real `flock`; a *different* thread still contends for the OS lock
# normally, since the depth counter is thread-local.
_lock_depth = threading.local()

# Lock files this PROCESS holds through `hold_gpu_lock_for_process`: while a key is here, `gpu_lock` on it passes
# through on EVERY thread. Empty unless a pipelined batched run is in progress, so `gpu_lock` is otherwise unchanged.
_process_held: set[str] = set()
_process_held_guard = threading.Lock()


@contextmanager
def gpu_lock(path: Path, blocking: bool = True) -> Iterator[None]:
    """Exclusive, cross-process file lock serializing GPU stage workers on the shared host.

    Held around any stage marked `gpu=True` in `STAGE_GPU` so only one heavy job runs at a time
    (Global Constraints: "outputs/engine/.gpu.lock"). `blocking=False` raises `BlockingIOError`
    immediately when another process already holds the lock, instead of waiting for it.
    Re-entrant within one thread (see `_lock_depth`); a different thread in the same process
    still blocks/contends for the lock like any other caller -- unless the process holds the lock
    through `hold_gpu_lock_for_process`, in which case every thread passes through.
    """
    key = str(path.resolve())
    with _process_held_guard:
        held_by_process = key in _process_held
    if held_by_process:
        yield
        return
    depth_map: dict[str, int] | None = getattr(_lock_depth, "depth", None)
    if depth_map is None:
        depth_map = {}
        _lock_depth.depth = depth_map

    if depth_map.get(key, 0) > 0:
        depth_map[key] += 1
        try:
            yield
        finally:
            depth_map[key] -= 1
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
    try:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, flags)
        depth_map[key] = 1
        try:
            yield
        finally:
            depth_map[key] = 0
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def hold_gpu_lock_for_process(path: Path) -> Iterator[None]:
    """`gpu_lock(path)` taken once, by the calling thread, for the whole block, and shared by every thread of the process.

    The pipelined batched runner runs stages on several threads of one process. Its own class threads provide
    in-process exclusion, and this one `flock` excludes other processes on the artifact root. A per-thread `gpu_lock`
    would not work there: a stage that nests `gpu_lock` itself (`run_hands_call`) on a thread other than the holder
    opens a second file description and blocks on the first forever. So while this block is open, `gpu_lock(path)`
    passes through on every thread. Not re-entrant: a second hold of the same path raises `RuntimeError`.
    """
    key = str(path.resolve())
    with gpu_lock(path):
        with _process_held_guard:
            if key in _process_held:
                raise RuntimeError(f"{key} is already held for this process")
            _process_held.add(key)
        try:
            yield
        finally:
            with _process_held_guard:
                _process_held.discard(key)


def _mark_and_persist(
    manifest: ClipManifest,
    cfg: EngineConfig,
    clip_id: str,
    stage: str,
    state: str,
    seconds: float,
    error: str = "",
) -> None:
    """`manifest.mark(...)`, then immediately `manifest.save(cfg.manifest_path)` if set.

    This is what makes a stage transition durable rather than only in-memory: called right after
    every `.mark`, so at most the one stage currently in flight can be lost to a process kill.
    """
    manifest.mark(clip_id, stage, state, seconds, error=error)
    if cfg.manifest_path is not None:
        manifest.save(cfg.manifest_path)


def run_clip(
    clip_id: str,
    cfg: EngineConfig,
    stage_fns: dict[str, StageFn],
    manifest: ClipManifest,
) -> None:
    """Run every stage in `cfg.stages` for one clip, timed and mirrored into `manifest`.

    Builds its own `Profiler` (never shared across clips). A stage already `"done"` is skipped
    when `cfg.skip_done`. A stage worker's exception marks that stage `"failed"` with `str(exc)`
    and stops this clip's processing -- the exception is isolated here, not re-raised, so
    `run_manifest` can continue with the next clip. Deliberately `except Exception`, not
    `BaseException`: a worker raising `SystemExit`/`KeyboardInterrupt` (or any other non-`Exception`
    signal) is treated as the process itself being killed and is allowed to propagate all the way
    out, rather than being swallowed as an ordinary stage failure. The clip's own
    `profiler.report()` is persisted to `<artifact_root>/<clip_id>/profile.json` for per-clip
    debugging.
    """
    profiler = Profiler(
        clip_id=clip_id, system=cfg.system, n_frames=manifest.n_frames(clip_id), fps=cfg.fps,
    )
    clip_dir = cfg.artifact_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.artifact_root / _GPU_LOCK_NAME

    for stage in cfg.stages:
        if cfg.skip_done and manifest.status(clip_id, stage).state == "done":
            continue
        if stage not in stage_fns:
            logger.warning("no stage worker registered for %r; leaving %r pending", stage, clip_id)
            continue

        worker = stage_fns[stage]
        use_gpu_lock = stage_uses_gpu(stage)
        try:
            with profiler.stage(stage):
                if use_gpu_lock:
                    with gpu_lock(lock_path):
                        worker(clip_id, clip_dir)
                else:
                    worker(clip_id, clip_dir)
        except Exception as exc:          # stage workers may raise anything; isolated per clip
            seconds = profiler.report().stages[-1].seconds
            _mark_and_persist(manifest, cfg, clip_id, stage, "failed", seconds, error=str(exc))
            logger.warning("clip %r failed at stage %r: %s", clip_id, stage, exc)
            if getattr(exc, "fatal_to_run", False):
                # A worker may declare its failure fatal to the RUN, not just to this clip -- the GPU guard
                # SIGKILLing a model subprocess is the case that matters, because the next clip meets the same
                # card. The clip's failure is already recorded, so a resumable rerun retries it.
                logger.error("clip %r: %r is fatal to the run, stopping: %s", clip_id, stage, exc)
                raise
            break
        else:
            seconds = profiler.report().stages[-1].seconds
            _mark_and_persist(manifest, cfg, clip_id, stage, "done", seconds)

    profiler.report().to_json(clip_dir / "profile.json")


def run_manifest(manifest: ClipManifest, cfg: EngineConfig, stage_fns: dict[str, StageFn]) -> None:
    """Run `run_clip` once for every clip with at least one non-`"done"` stage in `cfg.stages`.

    Never constructs or holds a `Profiler` itself: cross-clip aggregation goes exclusively through
    `clips_per_gpu_hour_report`.
    """
    pending_ids: set[str] = set()
    for stage in cfg.stages:
        pending_ids.update(manifest.pending(stage))
    if not cfg.retry_refusals:
        refused = manifest.refused(cfg.stages) & pending_ids
        if refused:
            logger.info("skipping %d clip(s) a stage already refused; set retry_refusals to re-run them",
                        len(refused))
        pending_ids -= refused
    for clip_id in manifest.clip_ids():
        if clip_id in pending_ids:
            run_clip(clip_id, cfg, stage_fns, manifest)


def clips_per_gpu_hour_report(manifest: ClipManifest) -> dict[str, float]:
    """Per-stage and overall clips per GPU-hour, from the manifest's own `{stage}_seconds` columns.

    `n` is every clip of the manifest, and a stage's total sums its seconds over every clip in every state: a failed
    stage's seconds are compute the engine spent, and a clip's `profile.json` records them too, as
    `compute_yield_report` does. The rate is `3600 * n / total`, written `3600 / (total / n)`
    and summed in `fastdo.bench.yield_report.stage_order`, exactly as `compute_yield_report` writes it, so the two agree
    bit for bit on the same totals. `"overall"` is the same over the total of every stage. `inf` when
    a total is 0 or the manifest is empty. For one clip this is `3600 / total`, the same convention as
    `fastdo.bench.profiler.ProfileReport.clips_per_gpu_hour`, and N identical clips give the one-clip values.
    """
    clip_ids = manifest.clip_ids()
    n = len(clip_ids)
    report: dict[str, float] = {}
    total = 0.0
    for stage in stage_order(manifest):
        stage_total = 0.0
        for clip_id in clip_ids:
            stage_total += manifest.status(clip_id, stage).seconds
        report[stage] = _clips_per_hour(stage_total, n)
        total += stage_total
    report["overall"] = _clips_per_hour(total, n)
    return report


def _clips_per_hour(total_seconds: float, n_clips: int) -> float:
    return float("inf") if n_clips == 0 or total_seconds <= 0.0 else 3600.0 / (total_seconds / n_clips)
