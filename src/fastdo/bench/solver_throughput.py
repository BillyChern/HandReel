"""Throughput of the batched trajectory solver: clips per second against the batch size B.

`synthetic_dexycb_like` builds the benchmark clips: DexYCB-like synthetic sequences (25 frames at 480x640,
focal 600 px, 1024 template samples, 64 tracks, occlusion 0.3, box shape) whose initial poses are perturbed per
frame by 3 cm / 10 deg, with a +10 % initial scale error. `measure_throughput` solves the first B clips together
(`solve_trajectories`) for each batch size and reports the best of `repeats` synchronized wall times, clips per
second, peak CUDA memory and each clip's LM iterations per annealing level. A batch size that exceeds the
process's CUDA budget is reported with its error instead of aborting the whole curve.

Not imported by `fastdo.bench` itself: importing the solver stack is kept out of the package's import path.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from fastdo.bench.device_memory import ProcessMemorySampler
from fastdo.geometry.lie import se3_exp
from fastdo.solver.api import (
    MeshSamples,
    SolverConfig,
    TrajectoryObservations,
    TrajectorySolution,
    clip_dims,
    plan_solve,
    solve_trajectories,
)
from fastdo.solver.memory import BatchPlan, GpuBudgetError

# The planner's own test (`memory._FUSED`): the fused Sinkhorn needs triton importable.
_FUSED = importlib.util.find_spec("triton") is not None
from fastdo.solver.synthetic import SyntheticClip, SyntheticConfig, make_synthetic_clip

__all__ = ["ClipInputs", "ThroughputRow", "allocator_conf", "measure_one_at_a_time", "measure_throughput", "perturbed_inputs",
           "synthetic_dexycb_like"]

logger = logging.getLogger(__name__)
ClipInputs = tuple[MeshSamples, TrajectoryObservations, Tensor, float]


@dataclass(frozen=True)
class ThroughputRow:
    """One batch size: `walls` (s) of every repeat, `clips_per_second` from the fastest, `peak_gb` of CUDA memory
    allocated, `iterations[b]` clip b's LM iterations per level (coarse, mid, fine), `sub_batches` the sizes
    `solve_trajectories` splits the B clips into under the process's CUDA budget (`api.plan_solve`; `(B,)`
    when they fit together) and `estimate_gb` their largest estimated peak; `error` when it failed.

    `reserved_gb` is the caching allocator's peak reserved memory (what `estimate_gb` x slack bounds);
    `nvml_peak_gb` / `nvml_rows` the largest NVML per-process reading while the repeats ran (the guard's figure,
    non-allocator memory included; None off CUDA or without nvidia-smi); `alloc_conf` the PYTORCH_CUDA_ALLOC_CONF in
    effect and `fused_sinkhorn` the solver's Sinkhorn path. `card_limited` when the card's free memory
    narrowed a sub-batch (`memory.BatchPlan.card_limited`; `plan_solve` applies the solve's own rule)."""

    batch_size: int
    walls: tuple[float, ...]
    clips_per_second: float
    peak_gb: float
    iterations: tuple[tuple[int, ...], ...]
    sub_batches: tuple[int, ...] = ()
    estimate_gb: float = 0.0
    error: str = ""
    reserved_gb: float = 0.0
    nvml_peak_gb: float | None = None
    nvml_rows: int = 0
    alloc_conf: str = ""
    fused_sinkhorn: bool = False
    card_limited: bool = False


def perturbed_inputs(clip: SyntheticClip, seed: int, trans: float = 0.03, rot_deg: float = 10.0) -> ClipInputs:
    """Solver inputs for a synthetic clip: every frame's GT pose perturbed by a `trans` m translation and a `rot_deg`
    rotation in random directions (left-multiplied), initial scale 1.1x the true scale."""
    g = torch.Generator().manual_seed(seed)
    T = clip.poses_gt.shape[0]
    xi = torch.zeros(T, 6)
    xi[:, :3] = trans * torch.nn.functional.normalize(torch.randn(T, 3, generator=g), dim=1)
    xi[:, 3:] = math.radians(rot_deg) * torch.nn.functional.normalize(torch.randn(T, 3, generator=g), dim=1)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    init = se3_exp(xi.double()).float() @ clip.poses_gt
    return MeshSamples(clip.surface_points, clip.surface_normals), obs, init, clip.scale_gt * 1.1


def synthetic_dexycb_like(n_clips: int, n_frames: int = 25, n_surface: int = 1024, seed: int = 200,
                          image_hw: tuple[int, int] = (480, 640)) -> list[ClipInputs]:
    """`n_clips` DexYCB-like clips; clip i uses synthetic seed `seed + i` and perturbation seed `seed + 100 + i`."""
    return [perturbed_inputs(make_synthetic_clip(SyntheticConfig(n_frames=n_frames, image_hw=image_hw, focal=600.0,
                                                                 n_surface=n_surface, n_tracks=64, occlusion=0.3,
                                                                 shape="box"), seed=seed + i), seed=seed + 100 + i)
            for i in range(n_clips)]


def allocator_conf() -> str:
    """The caching allocator's configuration in effect: PYTORCH_ALLOC_CONF, else its deprecated name
    PYTORCH_CUDA_ALLOC_CONF, else empty."""
    return os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


def _solve(inputs: Sequence[ClipInputs], cfg: SolverConfig) -> list[TrajectorySolution]:
    return solve_trajectories([x[0] for x in inputs], [x[1] for x in inputs], [x[2] for x in inputs],
                              [x[3] for x in inputs], cfg)


def fused_effective(cfg: SolverConfig) -> bool:
    """Whether the fused Sinkhorn kernels actually RUN for `cfg` -- the flag, triton importable, and a CUDA device (the
    memory planner's own expression). A throughput row reports this, not the flag: since the flag defaults on, a CPU
    row echoing it would claim kernels that cannot run there."""
    return cfg.fused_sinkhorn and _FUSED and torch.device(cfg.device).type == "cuda"


def measure_one_at_a_time(inputs: Sequence[ClipInputs], cfg: SolverConfig, repeats: int = 2) -> float:
    """Clips/s solving each of `inputs` ALONE (best of `repeats` per clip): the like-for-like baseline for a batch of
    the same clips. `measure_throughput`'s B=1 row solves only `inputs[0]`, the easiest clip of the synthetic set
    (LM iterations (10,1,1) against up to (10,5,10) in the batch), which overstates B=1 against a lockstep batch that
    runs until its slowest clip is done."""
    cuda = torch.device(cfg.device).type == "cuda"
    _solve(inputs[: min(2, len(inputs))], cfg)                  # warm-up, as in measure_throughput
    total = 0.0
    for clip in inputs:
        best = float("inf")
        for _ in range(repeats):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _solve([clip], cfg)
            if cuda:
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        total += best
    return len(inputs) / total


def measure_throughput(inputs: Sequence[ClipInputs], batch_sizes: Sequence[int], cfg: SolverConfig,
                       repeats: int = 2, stop_mib: int | None = None) -> list[ThroughputRow]:
    """Solve `inputs[:B]` together `repeats` times for each B (after one warm-up solve of two clips). On CUDA the
    repeats run under a `ProcessMemorySampler`; `stop_mib` makes it exit the process past that NVML reading."""
    if max(batch_sizes) > len(inputs):
        raise ValueError(f"batch size {max(batch_sizes)} needs that many clips, got {len(inputs)}")
    cuda = torch.device(cfg.device).type == "cuda"

    def synchronize() -> None:
        if cuda:
            torch.cuda.synchronize()

    _solve(inputs[: min(2, len(inputs))], cfg)                  # warm-up: kernel compilation, allocator pools
    rows: list[ThroughputRow] = []
    for B in batch_sizes:
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        walls: list[float] = []
        sols: list[TrajectorySolution] = []
        plans: list[BatchPlan] = []
        error = ""
        sampler = ProcessMemorySampler(stop_mib=stop_mib) if cuda else None
        try:
            chosen = inputs[:B]
            plans = plan_solve(clip_dims([x[0] for x in chosen], [x[1] for x in chosen], [x[2] for x in chosen], cfg),
                               cfg)
            with sampler if sampler is not None else contextlib.nullcontext():
                for _ in range(repeats):
                    synchronize()
                    t0 = time.perf_counter()
                    sols = _solve(chosen, cfg)
                    synchronize()
                    walls.append(time.perf_counter() - t0)
        except (torch.OutOfMemoryError, GpuBudgetError) as exc:
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            logger.warning("B=%d: %s", B, error)
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        nvml_mib = sampler.peak_mib if sampler is not None else None
        rate = B / min(walls) if walls and not error else 0.0
        row = ThroughputRow(batch_size=B, walls=tuple(walls), clips_per_second=rate, peak_gb=peak,
                            iterations=tuple(tuple(len(a) for a in s.accepted) for s in sols),
                            sub_batches=tuple(len(p.clips) for p in plans),
                            estimate_gb=max((p.peak_bytes / 2**30 for p in plans), default=0.0), error=error,
                            reserved_gb=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0.0,
                            nvml_peak_gb=None if nvml_mib is None else nvml_mib / 1024,
                            nvml_rows=sampler.max_rows if sampler is not None else 0,
                            alloc_conf=allocator_conf(), fused_sinkhorn=fused_effective(cfg),
                            card_limited=any(p.card_limited for p in plans))
        rows.append(row)
        logger.info("B=%2d: %.3f clips/s, walls %s s, peak allocated %.2f / reserved %.2f / NVML %s GB (estimate %.2f GB, "
                    "sub-batches %s), LM iterations per clip %s", B, rate, [f"{w:.3f}" for w in walls], peak,
                    row.reserved_gb, "n/a" if row.nvml_peak_gb is None else f"{row.nvml_peak_gb:.2f}", row.estimate_gb,
                    row.sub_batches, list(row.iterations))
    return rows
