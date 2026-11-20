"""On CUDA: a clip `plan_batches` admits under a budget fits that budget, and keyframe
registration holds one distance tensor at a time.

Skipped without CUDA; marked `timing` (GPU, contention-sensitive, outside the merge gate). Every cap set here is at
most the process cap the session applied (`apply_gpu_budget`, FASTDO_GPU_MEM_GB <= 12); a case whose tightest admitted
budget would exceed it is skipped, never run above the session's cap.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator

import pytest
import torch

from fastdo.solver import batch_eval, keyframe, memory
from fastdo.solver.api import (
    MeshSamples,
    SolverConfig,
    TrajectoryObservations,
    clip_dims,
    solve_trajectory,
)
from fastdo.solver.synthetic import SyntheticClip, SyntheticConfig, make_synthetic_clip

logger = logging.getLogger(__name__)
_GiB = 2**30


@pytest.fixture
def restore_cap() -> Iterator[float]:
    """The session's CUDA fraction, restored after the test lowers it."""
    fraction = torch.cuda.get_per_process_memory_fraction(0)
    yield fraction
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(fraction, 0)


def _session_cap_bytes(fraction: float) -> int:
    return int((fraction if 0.0 < fraction <= 1.0 else 1.0) * torch.cuda.get_device_properties(0).total_memory)


def _inputs(clip: SyntheticClip) -> tuple[MeshSamples, TrajectoryObservations]:
    return (MeshSamples(clip.surface_points, clip.surface_normals),
            TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                   pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                   anchors=clip.anchors))


def _solve_at_tightest_admitted_cap(clip: SyntheticClip, cfg: SolverConfig, session_cap: int) -> dict[str, float]:
    """Plan under the session cap, lower the cap to the smallest one that still admits that plan, solve, and check
    the allocated peak against the estimate and the reserved peak against the cap."""
    mesh, obs = _inputs(clip)
    dims = clip_dims([mesh], [obs], [clip.poses_gt], cfg)
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0)
    headroom = session_cap - memory.NON_ALLOCATOR_RESERVE_BYTES - allocated
    (planned,) = memory.plan_batches(dims, cfg, headroom, batch_eval._REFRESH_ELEMENTS)
    cap = memory.NON_ALLOCATOR_RESERVE_BYTES + allocated + math.ceil(planned.peak_bytes * memory.ALLOCATOR_SLACK) + (
        2 << 20)
    if cap > session_cap:
        pytest.skip(f"the tightest admitted cap {cap / _GiB:.2f} GiB exceeds the session cap {session_cap / _GiB:.2f}")
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    sol = solve_trajectory(mesh, obs, clip.poses_gt, clip.scale_gt, cfg)      # an OutOfMemoryError fails the test
    seconds = time.perf_counter() - t0
    peak_allocated = torch.cuda.max_memory_allocated(0) - allocated
    peak_reserved = torch.cuda.max_memory_reserved(0)
    assert sol.plan is not None and sol.plan.refresh_elements == planned.refresh_elements
    assert sol.plan.peak_bytes == planned.peak_bytes
    assert peak_allocated <= planned.peak_bytes, (peak_allocated / _GiB, planned.peak_bytes / _GiB)
    assert peak_reserved <= cap, (peak_reserved / _GiB, cap / _GiB)
    row = {"chunk_frames": planned.refresh_elements // (cfg.n_obs * dims[0].n_mesh), "cap_gib": cap / _GiB,
           "estimate_gib": planned.peak_bytes / _GiB, "allocated_gib": peak_allocated / _GiB,
           "reserved_gib": peak_reserved / _GiB, "seconds": seconds}
    logger.info("tightest admitted cap: %s", {k: round(v, 3) for k, v in row.items()})
    return row


@pytest.mark.timing
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
def test_admitted_plans_fit_their_budget_on_cuda(restore_cap: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default float32 Sinkhorn at 4x counts (T=16, n_obs 4096, n_mesh 8192, 720x1280: 8-frame chunks beside the
    cached plan) and an eager float64 Sinkhorn at 2048 x 4096, one shot (T=8: 1.79 GiB allocated against the old 1.55
    GiB estimate) and in 4-frame chunks (T=16: 2.064 GiB against 2.042 GiB without the cuBLAS workspace), each solve
    at the smallest cap their plan is admitted under. Then keyframe registration at 3x its counts (1536 x 3072, the
    scaled R4b replay that ran out of memory on the masked copy) peaks near one (120, N, M) float64 distance tensor."""
    session_cap = _session_cap_bytes(restore_cap)
    big = make_synthetic_clip(SyntheticConfig(n_frames=16, image_hw=(720, 1280), focal=900.0, n_surface=8192,
                                              n_tracks=96, shape="box"), seed=7)
    f32 = _solve_at_tightest_admitted_cap(big, SolverConfig(n_obs=4096, n_mesh=8192, device="cuda", iters_coarse=3,
                                                            iters_mid=2, iters_fine=3), session_cap)
    assert f32["chunk_frames"] < 16                                  # the chunked regime, not a one-shot refresh
    f64_cfg = SolverConfig(n_obs=2048, n_mesh=4096, device="cuda", iters_coarse=3, iters_mid=2, iters_fine=3,
                           sinkhorn_float32=False)
    small = make_synthetic_clip(SyntheticConfig(n_frames=8, image_hw=(480, 640), focal=600.0, n_surface=4096,
                                                n_tracks=64, shape="box"), seed=8)
    assert _solve_at_tightest_admitted_cap(small, f64_cfg, session_cap)["chunk_frames"] >= 8     # one shot
    longer = make_synthetic_clip(SyntheticConfig(n_frames=16, image_hw=(480, 640), focal=600.0, n_surface=4096,
                                                 n_tracks=64, shape="box"), seed=9)
    monkeypatch.setattr(batch_eval, "_REFRESH_ELEMENTS", 4 * 2048 * 4096)   # the planner's chunk bound: 4 frames
    assert _solve_at_tightest_admitted_cap(longer, f64_cfg, session_cap)["chunk_frames"] == 4

    torch.cuda.set_per_process_memory_fraction(restore_cap, 0)
    torch.cuda.empty_cache()
    valid = big.masks[0] & torch.isfinite(big.pointmaps[0]).all(-1)
    n_obs, n_mesh = 1536, 3072
    assert int(valid.sum()) >= n_obs
    one_tensor = 120 * n_obs * n_mesh * 8
    base = torch.cuda.memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0)
    keyframe.register_keyframe_pose(big.surface_points, big.surface_normals, big.pointmaps[0][valid], big.intrinsics,
                                    big.masks[0], scale=big.scale_gt, n_obs=n_obs, n_mesh=n_mesh, n_iters=3,
                                    device="cuda")
    peak = torch.cuda.max_memory_allocated(0) - base
    logger.info("registration 3x: peak %.3f GiB, one distance tensor %.3f GiB", peak / _GiB, one_tensor / _GiB)
    assert peak <= 1.25 * one_tensor, (peak / _GiB, one_tensor / _GiB)
