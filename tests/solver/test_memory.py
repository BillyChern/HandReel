"""CUDA memory planning of the batched solve (`fastdo.solver.memory`), checked on CPU.

The estimate is checked against the allocator: torch.profiler's memory events report every CPU allocation and free
(C++-internal temporaries included), so the live-bytes curve of a real `solve_trajectories` run gives its measured
peak. Device-side peak = the `DeviceBatch` tensors + the peak above the live set at the end of `prepare_batch`
(host-only packing tensors excluded, as they never reach the GPU). The planner is checked for the budget bound,
consecutive maximal groups and the fail-fast error; the solve is checked to split under a small budget without
changing any result.
"""

from __future__ import annotations

import time
from dataclasses import fields

import pytest
import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from fastdo.solver import api, batch, batch_eval, memory
from fastdo.solver.api import (
    SolverConfig,
    TrajectorySolution,
    clip_dims,
    solve_trajectories,
    solve_trajectory,
)
from fastdo.solver.memory import (
    ALLOCATOR_SLACK,
    BatchPlan,
    ClipDims,
    GpuBudgetError,
    batch_peak_bytes,
    plan_batches,
)
from tests.solver.test_batch import Inputs, _clip, _inputs, _unzip

MAX_REFRESH = 1 << 28
# The dense-plan, dense-z-buffer memory model these tests were derived on (the earlier defaults); the moments and
# sparse z-buffer paths
# have their own estimate tests (test_plan_moments.py, test_sparse_zbuffer.py and the sparse case below).
_DENSE = {"plan_moments": False, "sparse_zbuffer": False}


def _device_bytes(db: batch.DeviceBatch) -> int:
    seen: dict[int, int] = {}
    for f in fields(db):
        value = getattr(db, f.name)
        if isinstance(value, Tensor):
            st = value.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
    return sum(seen.values())


def _measured_peak(monkeypatch: pytest.MonkeyPatch, inputs: list, cfg: SolverConfig, refresh: int) -> int:
    """Device-side peak bytes of `solve_trajectories` on CPU (see the module docstring)."""
    marks: list[tuple[int, int]] = []
    real_prepare = api.prepare_batch

    def prepare(packed: batch.PackedBatch, c: SolverConfig) -> batch.DeviceBatch:
        db = real_prepare(packed, c)
        marks.append((time.time_ns(), _device_bytes(db)))
        return db

    monkeypatch.setattr(api, "prepare_batch", prepare)
    monkeypatch.setattr(batch_eval, "_REFRESH_ELEMENTS", refresh)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        solve_trajectories(*_unzip(inputs), cfg)
    events = sorted((e.start_ns(), e.nbytes()) for e in prof.profiler.kineto_results.events() if e.name() == "[memory]")
    assert len(marks) == 1 and events
    t_prep, db_bytes = marks[0]
    live = base = peak = 0
    for t_ns, n_bytes in events:
        live += n_bytes
        if t_ns <= t_prep:
            base = peak = live
        peak = max(peak, live)
    return db_bytes + peak - base


@pytest.mark.parametrize(("lengths", "n_obs", "n_mesh", "float32", "chunk_frames"),
                         [((6,), 128, 256, True, None), ((6,), 256, 512, True, None), ((5, 6, 7), 128, 256, True, None),
                          ((5, 6, 7), 128, 256, False, None), ((5, 6, 7), 128, 256, True, 2),
                          ((4,), 256, 2048, False, None), ((6,), 256, 2048, False, 2), ((6,), 256, 2048, True, 2)])
def test_estimate_bounds_measured_cpu_peak(monkeypatch: pytest.MonkeyPatch, lengths: tuple[int, ...], n_obs: int,
                                           n_mesh: int, float32: bool, chunk_frames: int | None) -> None:
    """The estimate is an upper bound on the allocator peak of a real solve and at most 2x above it: one clip (plan
    cache freed before a refresh), a larger N x M, three clips (partial refreshes beside the cached plan), an eager
    float64 Sinkhorn, and two-frame Sinkhorn chunks. At N x M = 256 x 2048 the per-entry
    Sinkhorn transients dominate the per-frame terms; the float64 cases (one shot, and in chunks beside the cached
    plan) and the float32 chunked case keep the estimate within 2x at that ratio after the float64 term went from 24
    to 32 bytes per entry. The CPU allocator stayed under the old 24 here; the CUDA underestimate is guarded by
    `test_memory_gpu.py`."""
    inputs = [_inputs(_clip(T, n_mesh, seed=170 + T, image_hw=(120, 160), n_tracks=16), seed=T) for T in lengths]
    cfg = SolverConfig(**_DENSE, n_obs=n_obs, n_mesh=n_mesh, iters_coarse=4, iters_mid=2, iters_fine=3, device="cpu",
                       sinkhorn_float32=float32)
    refresh = MAX_REFRESH if chunk_frames is None else chunk_frames * n_obs * n_mesh
    measured = _measured_peak(monkeypatch, inputs, cfg, refresh)
    estimate = batch_peak_bytes(clip_dims(*_unzip(inputs)[:3], cfg), cfg, refresh)
    assert measured <= estimate <= 2.0 * measured, (measured, estimate, estimate / measured)


@pytest.mark.parametrize(("lengths", "moments"), [((6,), False), ((5, 6, 7), False), ((5, 6, 7), True)])
def test_sparse_zbuffer_estimate_bounds_measured_cpu_peak(monkeypatch: pytest.MonkeyPatch, lengths: tuple[int, ...],
                                                          moments: bool) -> None:
    """With the sparse z-buffer (and a larger image, where the dense one would dominate) the estimate stays an
    upper bound on the allocator peak and within 2x of it."""
    inputs = [_inputs(_clip(T, 512, seed=206 + T, image_hw=(480, 640), n_tracks=16), seed=T) for T in lengths]
    cfg = SolverConfig(n_obs=128, n_mesh=512, iters_coarse=4, iters_mid=2, iters_fine=3, device="cpu",
                       sparse_zbuffer=True, plan_moments=moments)
    measured = _measured_peak(monkeypatch, inputs, cfg, MAX_REFRESH)
    estimate = batch_peak_bytes(clip_dims(*_unzip(inputs)[:3], cfg), cfg, MAX_REFRESH)
    assert measured <= estimate <= 2.0 * measured, (measured, estimate, estimate / measured)


def _dims(n: int) -> list[ClipDims]:
    return [ClipDims(n_frames=20 + 5 * (i % 3), n_mesh=1024 * (1 + i % 2), image_hw=(480, 640), n_tracks=64)
            for i in range(n)]


def test_plan_batches_fits_the_budget_with_maximal_consecutive_groups() -> None:
    cfg = SolverConfig(**_DENSE, device="cuda")
    dims = _dims(7)
    one_frame = cfg.n_obs * max(d.n_mesh for d in dims)
    budget = int(batch_peak_bytes(dims[:2], cfg, MAX_REFRESH) * ALLOCATOR_SLACK * 1.3)
    plans = plan_batches(dims, cfg, budget, MAX_REFRESH)
    assert [c for p in plans for c in p.clips] == list(range(7))
    assert any(len(p.clips) > 1 for p in plans) and len(plans) > 1
    for p in plans:
        group = [dims[i] for i in p.clips]
        assert p.peak_bytes == batch_peak_bytes(group, cfg, p.refresh_elements)
        assert p.peak_bytes * ALLOCATOR_SLACK <= budget and p.refresh_elements <= MAX_REFRESH
        if p.clips[-1] + 1 < len(dims):                      # maximal: the next clip does not fit even in 1-frame chunks
            wider = group + [dims[p.clips[-1] + 1]]
            assert batch_peak_bytes(wider, cfg, one_frame) * ALLOCATOR_SLACK > budget
    assert any(p.refresh_elements < MAX_REFRESH for p in plans)    # chunks shrink before a batch is split


def test_plan_batches_without_budget_and_fail_fast() -> None:
    cfg = SolverConfig(**_DENSE, device="cuda")
    dims = _dims(3)
    (only,) = plan_batches(dims, cfg, None, MAX_REFRESH)
    assert only.clips == (0, 1, 2) and only.refresh_elements == MAX_REFRESH
    with pytest.raises(GpuBudgetError, match=r"clip 0 \(T=20.*CUDA budget leaves 0\.12 GiB"):
        plan_batches(dims, cfg, 1 << 27, MAX_REFRESH)
    with pytest.raises(ValueError, match="no clips"):
        plan_batches([], cfg, None, MAX_REFRESH)


def test_plan_batches_prefers_a_one_shot_refresh_for_one_clip() -> None:
    """One clip refreshed in one chunk frees its cached plan first, so it needs less than a chunk one frame shorter
    (which holds the new plan's buffer beside the chunk): a budget that fits the one-shot refresh must plan it."""
    cfg = SolverConfig(**_DENSE, device="cuda")
    dims = [ClipDims(n_frames=20, n_mesh=1024, image_hw=(480, 640), n_tracks=64)]
    one_shot = batch_peak_bytes(dims, cfg, MAX_REFRESH)
    assert batch_peak_bytes(dims, cfg, 19 * cfg.n_obs * 1024) > one_shot
    (plan,) = plan_batches(dims, cfg, int(one_shot * ALLOCATOR_SLACK) + 1, MAX_REFRESH)
    assert plan.refresh_elements == MAX_REFRESH and plan.peak_bytes == one_shot


def test_solve_splits_under_a_small_budget_without_changing_results(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = [_inputs(_clip(T, 128, seed=190 + T), seed=T) for T in (5, 6, 7)]
    cfg = SolverConfig(**_DENSE, n_obs=64, n_mesh=128, device="cpu", sinkhorn_float32=False)
    whole = solve_trajectories(*_unzip(inputs), cfg)
    dims = clip_dims(*_unzip(inputs)[:3], cfg)
    budget = int(batch_peak_bytes(dims[:2], cfg, MAX_REFRESH) * ALLOCATOR_SLACK)
    groups: list[tuple[int, int]] = []
    real_solve = api._solve_batch

    def spy(meshes, obs, poses, scales, c, refresh_elements):
        groups.append((len(meshes), refresh_elements))
        return real_solve(meshes, obs, poses, scales, c, refresh_elements)

    monkeypatch.setattr(api, "allocator_headroom_bytes", lambda device: budget)
    monkeypatch.setattr(api, "_solve_batch", spy)
    split = solve_trajectories(*_unzip(inputs), cfg)
    assert [n for n, _ in groups] == [2, 1], groups
    for one, many in zip(whole, split, strict=True):
        assert many.accepted == one.accepted
        assert torch.allclose(many.poses, one.poses, atol=1e-6) and abs(many.scale - one.scale) < 1e-9
    monkeypatch.setattr(api, "allocator_headroom_bytes", lambda device: 1 << 10)
    monkeypatch.setattr(api, "prepare_batch", lambda *a: pytest.fail("allocated before checking the budget"))
    with pytest.raises(GpuBudgetError):
        solve_trajectories(*_unzip(inputs), cfg)


def test_require_allocator_bytes_reads_the_fraction_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    memory.require_allocator_bytes(1 << 40, "cpu", "a CPU tensor")          # no budget off CUDA

    class Props:
        total_memory = 80 << 30

    monkeypatch.setattr(memory, "_cuda_index", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda idx: 0.05)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: Props())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda idx: 1 << 30)
    memory.require_allocator_bytes(2 << 30, "cuda", "a plan")               # 4 GiB cap - 1 GiB allocated = 3 GiB left
    with pytest.raises(GpuBudgetError, match=r"a plan needs 3\.50 GiB but the CUDA budget has 3\.00 GiB left"):
        memory.require_allocator_bytes(7 << 29, "cuda", "a plan")


def _shaped_inputs(n_frames: int, n_mesh: int) -> Inputs:
    """Inputs with DexYCB-like sizes (480x640, 64 tracks) and no content: planning reads only sizes (`clip_dims`)."""
    hw = (480, 640)
    masks = torch.zeros(1, 1, 1, dtype=torch.bool).expand(n_frames, *hw)
    obs = api.TrajectoryObservations(intrinsics=torch.eye(3), masks=masks, hand_masks=masks,
                                     pointmaps=torch.zeros(1, 1, 1, 3).expand(n_frames, *hw, 3),
                                     tracks=torch.zeros(n_frames, 64, 2),
                                     track_vis=torch.zeros(n_frames, 64, dtype=torch.bool), anchors=torch.zeros(64, 3))
    mesh = api.MeshSamples(points=torch.zeros(n_mesh, 3), normals=torch.zeros(n_mesh, 3))
    return mesh, obs, torch.eye(4).expand(n_frames, 4, 4), 1.0


def _fake_cuda_readings(monkeypatch: pytest.MonkeyPatch, cap_bytes: int, free_bytes: int) -> list[int]:
    """CUDA memory readings for a CPU run: an `apply_gpu_budget` cap of `cap_bytes` on a 64 GiB card (a fraction exact
    in binary) with nothing allocated, and `free_bytes` free on the card. Returns the list `mem_get_info` appends to."""
    reads: list[int] = []
    total = 64 << 30

    class Props:
        total_memory = total

    def mem_get_info(idx: int) -> tuple[int, int]:
        reads.append(idx)
        return free_bytes, total

    monkeypatch.setattr(memory, "_cuda_index", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda idx: cap_bytes / total)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: Props())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda idx: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    return reads


def _record_sub_batches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Replace the device solve by a stub that records each planned sub-batch as (clips, Sinkhorn chunk)."""
    groups: list[tuple[int, int]] = []

    def fake_solve(meshes: list, obs: list, poses: list, scales: list, cfg: SolverConfig,
                   refresh_elements: int) -> list[TrajectorySolution]:
        groups.append((len(meshes), refresh_elements))
        empty = torch.zeros(0)
        return [TrajectorySolution(poses=empty, scale=1.0, covariance=empty, per_frame_cost=empty, confidence=empty,
                                   iterations=0, seconds=0.0) for _ in meshes]

    monkeypatch.setattr(api, "_solve_batch", fake_solve)
    return groups


def _plan_of(sol: TrajectorySolution) -> BatchPlan:
    assert sol.plan is not None
    return sol.plan


@pytest.mark.parametrize("card", ["almost_full", "tight"])
def test_one_clip_plan_ignores_the_cards_free_memory(monkeypatch: pytest.MonkeyPatch, card: str) -> None:
    """A one-clip solve (`solve_trajectory`) plans from the `apply_gpu_budget` cap alone. With the card
    reporting almost nothing free, the old min(cap, card) budget raised `GpuBudgetError`; with just enough free for
    one-frame chunks, it planned smaller chunks. Now both give the cap-only plan, `mem_get_info` is never read, and
    the result records the plan with no card reading."""
    cfg = SolverConfig(**_DENSE, device="cpu")                                   # shipped sizes: n_obs 1024, n_mesh 2048
    mesh, obs, init, scale = _shaped_inputs(25, 2048)
    dims = clip_dims([mesh], [obs], [init], cfg)
    budget = (4 << 30) - memory.NON_ALLOCATOR_RESERVE_BYTES
    (expected,) = plan_batches(dims, cfg, budget, batch_eval._REFRESH_ELEMENTS)
    if card == "almost_full":
        free = 1 << 20
        with pytest.raises(GpuBudgetError):                           # what planning from the card would do
            plan_batches(dims, cfg, free, batch_eval._REFRESH_ELEMENTS)
    else:
        free = int(batch_peak_bytes(dims, cfg, cfg.n_obs * 2048) * ALLOCATOR_SLACK) + 1
        (from_card,) = plan_batches(dims, cfg, free, batch_eval._REFRESH_ELEMENTS)
        assert from_card.refresh_elements < expected.refresh_elements
    reads = _fake_cuda_readings(monkeypatch, 4 << 30, free)
    groups = _record_sub_batches(monkeypatch)
    assert memory.allocator_headroom_bytes("cuda") == budget
    sol = solve_trajectory(mesh, obs, init, scale, cfg)
    assert groups == [(1, expected.refresh_elements)] and sol.plan == expected
    assert expected.budget_bytes == budget and expected.card_bytes is None and not expected.card_limited
    assert reads == []


def test_card_free_memory_bounds_only_multi_clip_sub_batches_and_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """For more than one clip the card's free memory is read once and may narrow the sub-batches. With
    room on the card for one clip, three clips that fit the cap together are solved one at a time, each with the chunk
    a one-clip solve gets, and every result records the card reading and `card_limited`. With a free card the three
    clips are one sub-batch."""
    cfg = SolverConfig(**_DENSE, device="cpu")
    inputs = [_shaped_inputs(25, 2048) for _ in range(3)]
    dims = clip_dims(*_unzip(inputs)[:3], cfg)
    cap = 16 << 30
    budget = cap - memory.NON_ALLOCATOR_RESERVE_BYTES
    (alone,) = plan_batches(dims[:1], cfg, budget, batch_eval._REFRESH_ELEMENTS)
    assert [p.clips for p in plan_batches(dims, cfg, budget, batch_eval._REFRESH_ELEMENTS)] == [(0, 1, 2)]
    free = int(alone.peak_bytes * ALLOCATOR_SLACK) + 1                  # room for one clip, not for two
    reads = _fake_cuda_readings(monkeypatch, cap, free)
    groups = _record_sub_batches(monkeypatch)
    plans = [_plan_of(s) for s in solve_trajectories(*_unzip(inputs), cfg)]
    assert len(reads) == 1 and groups == [(1, alone.refresh_elements)] * 3
    assert [p.clips for p in plans] == [(0,), (1,), (2,)]
    assert all(p.budget_bytes == budget and p.card_bytes == free for p in plans)
    assert [p.card_limited for p in plans] == [True, True, False]

    reads = _fake_cuda_readings(monkeypatch, cap, 64 << 30)
    groups.clear()
    plans = [_plan_of(s) for s in solve_trajectories(*_unzip(inputs), cfg)]
    assert len(reads) == 1 and groups == [(3, batch_eval._REFRESH_ELEMENTS)]
    assert all(p.clips == (0, 1, 2) and p.card_bytes == 64 << 30 and not p.card_limited for p in plans)


def test_cuda_estimate_adds_the_cublas_workspace_and_keeps_the_1x_chunks() -> None:
    """On CUDA the estimate also counts the cuBLAS workspace the caching allocator keeps
    (`_CUDA_WORKSPACE_BYTES`, 32 MiB measured on the H100); off CUDA there is none. At the shipped counts (n_obs 1024,
    n_mesh 2048) on the whisking clip's sizes (T=46, 720x1280, 96 tracks) the plan under 12 and 20 GiB caps is still
    one refresh of every frame."""
    dims = [ClipDims(n_frames=46, n_mesh=2048, image_hw=(720, 1280), n_tracks=96)]
    on_cuda, on_cpu = SolverConfig(**_DENSE, device="cuda"), SolverConfig(**_DENSE, device="cpu")
    for refresh in (MAX_REFRESH, 4 * on_cuda.n_obs * 2048):
        extra = batch_peak_bytes(dims, on_cuda, refresh) - batch_peak_bytes(dims, on_cpu, refresh)
        assert extra == memory._CUDA_WORKSPACE_BYTES == 32 << 20
    for cap_gib in (12, 20):
        budget = (cap_gib << 30) - memory.NON_ALLOCATOR_RESERVE_BYTES
        (plan,) = plan_batches(dims, on_cuda, budget, batch_eval._REFRESH_ELEMENTS)
        assert plan.refresh_elements == batch_eval._REFRESH_ELEMENTS
