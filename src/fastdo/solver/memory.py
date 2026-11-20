"""CUDA memory planning for the batched solve: an explicit peak estimate and a batching bound.

`fastdo.bench.gpu_budget.apply_gpu_budget` caps a process with `torch.cuda.set_per_process_memory_fraction`. That
fraction bounds the caching allocator's reserved segments: every tensor, cuBLAS/cuBLASLt workspaces, CUB scratch
space and the fused Triton kernels' outputs. It does not bound the CUDA context with its lazily loaded kernel
images, the cuBLAS/cuSOLVER handle state, or Triton's loaded kernel modules. An OOM measured during development put that
non-PyTorch memory at 0.94 GiB for one solver process. MAGMA's batched routines, which torch's default heuristic
picks for a batched `torch.cholesky_solve`, would add pinned host arrays and `magma_malloc` workspaces outside the
fraction. The banded solve does not call them (`_block_thomas`). The fraction also cannot make a batch
fit: a batch that is too large
fails mid-solve with a generic OutOfMemoryError.

`plan_batches` therefore decides before any device allocation. Clips are grouped into consecutive sub-batches, and
each sub-batch gets a Sinkhorn chunk bound (`refresh_elements`) whose `batch_peak_bytes` fits the budget. A clip that
cannot fit alone, even with one-frame chunks, raises `GpuBudgetError`. `allocator_headroom_bytes` reads the budget
from the fraction `apply_gpu_budget` set, minus `NON_ALLOCATOR_RESERVE_BYTES` and the memory already allocated.
It never reads the card's free memory, so a one-clip plan depends on that budget alone. `card_headroom_bytes` reads
the card, and `plan_batches` lets it bound only sub-batches of more than one clip.

Peak model (`batch_peak_bytes`), for T frames in a sub-batch of B clips, N = n_obs observed points, and M, P, J the
sub-batch's largest template, image (H x W pixels) and track count (every per-frame tensor is padded to them).
E = T N M is the number of dense plan entries. A refresh runs over chunks of `per` frames (E_c = per N M entries).
A one-clip, one-chunk refresh ("one shot") frees the cached plan first. Otherwise the cached plan, or the new
plan's buffer, is held beside the chunk (8 E). Peak = persistent + the largest phase, plus on CUDA the cuBLAS
workspace the caching allocator keeps (`_CUDA_WORKSPACE_BYTES`, 32 MiB measured):
- persistent: the device batch, the Sinkhorn potentials, the visibility mask and the kNN plan (`_PER_*` per frame);
- Sinkhorn: held + 16 bytes per refreshed entry (32 for an eager float64 solve): four Sinkhorn-dtype tensors, the
  cost matrix beside a logsumexp's argument, shifted argument and exp (for float32 also the cost beside the plan
  before and after its float64 cast) + the chunk's marginals and kNN scratch;
- terms: the plan (8 E) + the chunk's cost matrix and, unless one shot, its float64 plan + residuals, Jacobians and
  normal-equation blocks for every frame (`_TERMS_*`);
- trial: the plan + a subset trial's gathered rows (8 E_c, B > 1) + the same term transients;
- visibility: held + the z-buffer (8 bytes per pixel; `sparse_zbuffer`: `_VIS_SPARSE_M` bytes per sample instead)
  and the (T, M) visibility maps of every refreshed frame.
Plan moments (`SolverConfig.plan_moments`): the cache holds the plan's canonical moments (B (T,N,3) and w (T,N) float64,
`_MOMENT_BYTES` = 32 per frame per observed point) instead of the plan, so every `8 E` above becomes `32 T N`. The kNN
plan leaves the persistent set, and its chunk's candidates (`_PER_KNN` per candidate) are counted beside the refresh and
the terms of that chunk instead. The refreshed chunk's cost matrix and float64 plan are counted beside the terms (an
upper bound: `pointmap_term` frees them after the reduction, before building the terms); a one-shot refresh has the
same bound as before, and the savings are the trial phase and every phase of a chunked refresh.
Anisotropic scale (`SolverConfig.anisotropic_scale`, zero when off): every term's Jacobian carries 9 columns instead of
7 plus the stretch columns' own temporaries (`_ANISO_N`, `_ANISO_J` per frame beside the terms of the terms and trial
phases), and a refresh builds the stretched template and its normals, A (y - c) and normalize(A^-1 n) (`_ANISO_M` per
frame beside the visibility pass and the Sinkhorn solve).
The coefficients are upper bounds read off `batch_eval`/`terms`/`visibility`/`ot`. `tests/solver/test_memory.py`
checks them against allocator peaks measured on CPU. `ALLOCATOR_SLACK` covers the caching allocator's block rounding
and split blocks.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fastdo.solver.api import SolverConfig

__all__ = ["ALLOCATOR_SLACK", "NON_ALLOCATOR_RESERVE_BYTES", "BatchPlan", "ClipDims", "GpuBudgetError",
           "allocator_headroom_bytes", "batch_peak_bytes", "card_headroom_bytes", "plan_batches",
           "require_allocator_bytes"]

_FUSED = importlib.util.find_spec("triton") is not None   # `ot_fused` runs the CUDA Sinkhorn when triton imports
NON_ALLOCATOR_RESERVE_BYTES = 1 << 30     # CUDA context, library handles, Triton modules (0.94 GiB measured)
ALLOCATOR_SLACK = 1.25                    # estimate x slack must fit: block rounding and split blocks
# Bytes per frame per unit (N observed points, M template samples, P image pixels, J tracks, knn candidates).
_PER_N, _PER_M, _PER_P, _PER_J, _PER_KNN, _PER_FRAME = 33, 65, 1, 41, 16, 2048    # persistent tensors
_SINKHORN_N, _SINKHORN_M, _SINKHORN_KNN = 40, 40, 64                             # per refreshed chunk frame
_TERMS_N, _TERMS_M, _TERMS_J = 700, 30, 600                                     # building terms + normal blocks
_VIS_P, _VIS_M = 8, 300                                                          # z-buffer, (T,M) visibility maps
_VIS_SPARSE_M = 64                        # sparse z-buffer per (frame, sample): indices, order, segments, buffer
_PLAN_BYTES = 8                           # cached dense plan, float64
_MOMENT_BYTES = 32                        # cached moments per frame per observed point, B (3) + w (1) float64
# Anisotropic scale, per frame per observed point / track: the (.,3,3) stretch columns and their product, the 9-column
# Jacobian's two extra columns through `pointmap_term`'s scaling and masking and `term_blocks`; per template sample: the
# stretched samples and normals (float64 (T,M,3) each) and one of their temporaries.
_ANISO_N, _ANISO_J, _ANISO_M = 300, 250, 96
# The cuBLAS workspace the caching allocator keeps once a CUDA solve has run (32.00 MiB still
# allocated after a solve returned on the H100). A float64 solve in 4-frame chunks peaked 23 MiB above the model
# without it. CUDA only, since it does not exist off the GPU.
_CUDA_WORKSPACE_BYTES = 32 << 20


class GpuBudgetError(RuntimeError):
    """A solve or kernel output cannot fit the process's CUDA budget (`apply_gpu_budget`)."""


@dataclass(frozen=True)
class ClipDims:
    """Host-known sizes of one clip: frames, template samples after subsampling, image (H, W), tracks."""

    n_frames: int
    n_mesh: int
    image_hw: tuple[int, int]
    n_tracks: int


@dataclass(frozen=True)
class BatchPlan:
    """Consecutive clips solved together, their Sinkhorn chunk bound, and their estimated peak (bytes).

    What the sub-batch was planned against. `budget_bytes` is `allocator_headroom_bytes` (None: no
    budget). `card_bytes` is `card_headroom_bytes` (None: not read, i.e. one clip or off CUDA). `card_limited` is
    True when the budget alone would have planned this sub-batch differently: with the next clip, or a larger
    chunk."""

    clips: tuple[int, ...]
    refresh_elements: int
    peak_bytes: int
    budget_bytes: int | None = None
    card_bytes: int | None = None
    card_limited: bool = False


def _refresh_bytes(cfg: SolverConfig) -> int:
    """Transient bytes per refreshed plan entry: four Sinkhorn-dtype (T,N,M) tensors, 16 bytes for float32 and 32 for
    a float64 Sinkhorn on the eager path (the default, and always off CUDA or without triton;
    `SolverConfig.fused_sinkhorn` opts into the fused kernels, kept at 16). An eager half-step's `torch.logsumexp`
    holds its argument, the argument minus its maxima and their exp beside the cost matrix; float32's cast of the
    plan (cost 4 + plan 4 + float64 plan 8) needs no more. float64 was 24, and a one-shot
    float64 solve (T=8, 2048 x 4096) peaked at 1.82 GiB allocated on CUDA against its 1.55 GiB estimate; the CPU
    allocator stayed under 24 at 256 x 2048, so the CUDA test (`tests/solver/test_memory_gpu.py`) guards it."""
    # Fusion is NOT discounted for a float64 Sinkhorn. It used to return 16 here whenever the fused
    # path applied, which was harmless while `fused_sinkhorn` defaulted off; with it on, the float64 one-shot solve
    # in test_memory_gpu peaked at 1.52 GiB against a 1.08 GiB estimate. Only float32 fused is measured to fit at 16
    # (7.48 GB at B=16, fused or eager), so float64 keeps the eager path's conservative 32.
    return 16 if cfg.sinkhorn_float32 else 32


def _cost_bytes(cfg: SolverConfig) -> int:
    """Bytes per refreshed entry still alive while the chunk's terms are built: its Sinkhorn-dtype cost matrix and
    the chunk's float64 plan before it is written into the plan buffer."""
    return (4 if cfg.sinkhorn_float32 else 8) + _PLAN_BYTES


def _vis_frame_bytes(cfg: SolverConfig, P: int, M: int) -> int:
    """Visibility-pass transients per refreshed frame: the z-buffer (dense: 8 bytes per pixel; sparse: per
    sample) and the (T, M) visibility maps."""
    return (_VIS_SPARSE_M * M if cfg.sparse_zbuffer else _VIS_P * P) + _VIS_M * M


def _aniso_terms_bytes(cfg: SolverConfig, T: int, N: int, J: int) -> int:
    """Anisotropic-scale term transients (module docstring); 0 with `anisotropic_scale` off."""
    return T * (_ANISO_N * N + _ANISO_J * J) if cfg.anisotropic_scale else 0


def _aniso_template_bytes(cfg: SolverConfig, T: int, M: int) -> int:
    """Stretched-template transients of a refresh (module docstring); 0 with `anisotropic_scale` off."""
    return T * _ANISO_M * M if cfg.anisotropic_scale else 0


def batch_peak_bytes(dims: Sequence[ClipDims], cfg: SolverConfig, refresh_elements: int) -> int:
    """Estimated peak device bytes of solving `dims` together with Sinkhorn chunks of `refresh_elements` entries:
    the persistent tensors plus the largest of four phases, each with only the transients alive at that moment
    (see the module docstring)."""
    T, B = sum(d.n_frames for d in dims), len(dims)
    N, M = cfg.n_obs, max(d.n_mesh for d in dims)
    P = max(d.image_hw[0] * d.image_hw[1] for d in dims)
    J = max(d.n_tracks for d in dims)
    knn = min(cfg.knn_fine or 0, M)
    E = T * N * M
    per = T if cfg.correspondence == "nn" else min(T, max(1, refresh_elements // (N * M)))  # nn is not chunked
    E_c, one_shot = per * N * M, B == 1 and per >= T
    if cfg.plan_moments:
        return _moments_peak_bytes(dims, cfg, T, N, M, P, J, knn, per, one_shot)
    held = 0 if one_shot else _PLAN_BYTES * E              # the cached plan (or the new plan's buffer) beside a chunk
    persistent = T * (_PER_N * N + _PER_M * M + _PER_P * P + _PER_J * J + _PER_KNN * knn * N + _PER_FRAME)
    sinkhorn = held + _refresh_bytes(cfg) * E_c + per * (_SINKHORN_N * N + _SINKHORN_M * M + _SINKHORN_KNN * knn * N)
    sinkhorn += _aniso_template_bytes(cfg, T, M)
    chunk_plan = 0 if one_shot else _PLAN_BYTES            # one shot: the refreshed plan is the cache itself
    terms = (_PLAN_BYTES * E + (_cost_bytes(cfg) - _PLAN_BYTES + chunk_plan) * E_c
             + T * (_TERMS_N * N + _TERMS_M * M + _TERMS_J * J)) + _aniso_terms_bytes(cfg, T, N, J)
    trial = (_PLAN_BYTES * (E + (E_c if B > 1 else 0)) + T * (_TERMS_N * N + _TERMS_M * M + _TERMS_J * J)
             + _aniso_terms_bytes(cfg, T, N, J))
    # A refresh of every frame releases the plan before the visibility pass; only a partial refresh (B > 1) holds it.
    visibility = ((_PLAN_BYTES * E if B > 1 else 0) + T * _vis_frame_bytes(cfg, P, M) + _aniso_template_bytes(cfg, T, M)
                  if cfg.visibility else 0)
    workspace = _CUDA_WORKSPACE_BYTES if torch.device(cfg.device).type == "cuda" else 0
    return workspace + persistent + max(sinkhorn, terms, trial, visibility)


def _moments_peak_bytes(dims: Sequence[ClipDims], cfg: SolverConfig, T: int, N: int, M: int, P: int, J: int, knn: int,
                        per: int, one_shot: bool) -> int:
    """`batch_peak_bytes` with `SolverConfig.plan_moments`: the same phases with the cached moments (32 T N) in
    place of the cached plan (8 E), and the chunk's kNN candidates transient (module docstring)."""
    B, E_c = len(dims), per * N * M
    moments = _MOMENT_BYTES * T * N                         # the cached moments, or the new moments' buffer
    held = 0 if one_shot else moments
    chunk_knn = _PER_KNN * knn * N * per                    # the chunk's kNN plan until it is reduced
    persistent = T * (_PER_N * N + _PER_M * M + _PER_P * P + _PER_J * J + _PER_FRAME)
    frame_terms = T * (_TERMS_N * N + _TERMS_M * M + _TERMS_J * J) + _aniso_terms_bytes(cfg, T, N, J)
    sinkhorn = held + _refresh_bytes(cfg) * E_c + per * (_SINKHORN_N * N + _SINKHORN_M * M + _SINKHORN_KNN * knn * N)
    sinkhorn += chunk_knn + _aniso_template_bytes(cfg, T, M)
    terms = moments + _cost_bytes(cfg) * E_c + chunk_knn + frame_terms
    trial = moments * (2 if B > 1 else 1) + frame_terms     # a subset trial gathers its rows of the moments
    visibility = ((moments if B > 1 else 0) + T * _vis_frame_bytes(cfg, P, M) + _aniso_template_bytes(cfg, T, M)
                  if cfg.visibility else 0)
    workspace = _CUDA_WORKSPACE_BYTES if torch.device(cfg.device).type == "cuda" else 0
    return workspace + persistent + max(sinkhorn, terms, trial, visibility)


def _fit(dims: Sequence[ClipDims], cfg: SolverConfig, budget: int, max_refresh: int) -> tuple[int, int] | None:
    """Largest whole-frame chunk bound (<= `max_refresh`) whose estimate x slack fits `budget`, with its estimate.
    The default bound is tried first: for one clip a single chunk over every frame frees the cached plan and can
    need less than a chunk one frame shorter. Below it the estimate grows with the chunk, so it is bisected."""
    def fits(chunk: int) -> bool:
        return batch_peak_bytes(dims, cfg, chunk) * ALLOCATOR_SLACK <= budget

    N, M = cfg.n_obs, max(d.n_mesh for d in dims)
    T = sum(d.n_frames for d in dims)
    frames = max(1, min(T, max_refresh // (N * M)))
    if fits(max_refresh):
        return max_refresh, batch_peak_bytes(dims, cfg, max_refresh)
    lo, hi = 1, frames - 1
    if hi < 1 or not fits(N * M):
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(mid * N * M) else (lo, mid - 1)
    return lo * N * M, batch_peak_bytes(dims, cfg, lo * N * M)


def plan_batches(dims: Sequence[ClipDims], cfg: SolverConfig, budget: int | None, max_refresh: int,
                 card_budget: int | None = None) -> list[BatchPlan]:
    """Greedy consecutive sub-batches that fit `budget` bytes (None: no budget, one batch of every clip).

    A clip on its own is planned against `budget` alone, so its chunk, and whether it raises, never depend on
    `card_budget`. `card_budget` (the card's free memory, `card_headroom_bytes`) also bounds every sub-batch of more
    than one clip. Each plan records both and whether the card reading changed it (`BatchPlan.card_limited`)."""
    if not dims:
        raise ValueError("no clips to plan")
    if budget is None:
        return [BatchPlan(tuple(range(len(dims))), max_refresh, batch_peak_bytes(dims, cfg, max_refresh),
                          card_bytes=card_budget)]
    group_budget = budget if card_budget is None else min(budget, card_budget)
    plans: list[BatchPlan] = []
    start = 0
    while start < len(dims):
        fit = _fit(dims[start:start + 1], cfg, budget, max_refresh)
        if fit is None:
            d = dims[start]
            need = batch_peak_bytes(dims[start:start + 1], cfg, cfg.n_obs * d.n_mesh) * ALLOCATOR_SLACK
            raise GpuBudgetError(
                f"clip {start} (T={d.n_frames}, n_obs={cfg.n_obs}, M={d.n_mesh}, image {d.image_hw[0]}x"
                f"{d.image_hw[1]}) needs about {need / 2**30:.2f} GiB even with one-frame Sinkhorn chunks, but the "
                f"CUDA budget leaves {budget / 2**30:.2f} GiB (apply_gpu_budget / FASTDO_GPU_MEM_GB, minus "
                f"{NON_ALLOCATOR_RESERVE_BYTES / 2**30:.2f} GiB for non-allocator memory and what is already allocated)")
        end = start + 1
        while end < len(dims) and (wider := _fit(dims[start:end + 1], cfg, group_budget, max_refresh)) is not None:
            fit, end = wider, end + 1
        card_limited = group_budget < budget and (
            (end < len(dims) and _fit(dims[start:end + 1], cfg, budget, max_refresh) is not None)
            or (end - start > 1 and _fit(dims[start:end], cfg, budget, max_refresh) != fit))
        plans.append(BatchPlan(tuple(range(start, end)), fit[0], fit[1], budget, card_budget, card_limited))
        start = end
    return plans


def _cuda_index(device: torch.device | str) -> int | None:
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _cap_bytes(idx: int) -> int:
    """The allocator cap `apply_gpu_budget` set (`set_per_process_memory_fraction`); the whole card when unset."""
    fraction = torch.cuda.get_per_process_memory_fraction(idx)
    return int((fraction if 0.0 < fraction <= 1.0 else 1.0) * torch.cuda.get_device_properties(idx).total_memory)


def allocator_headroom_bytes(device: torch.device | str) -> int | None:
    """Bytes the solver may still allocate on `device` under the `apply_gpu_budget` / FASTDO_GPU_MEM_GB fraction: its
    cap minus the non-allocator reserve and the memory already allocated. None off CUDA.

    Only the process's own budget enters, never the card's free memory: the card is shared, so a
    one-clip solve planned from it would chunk, split or raise depending on other processes. `card_headroom_bytes`
    reads the card for grouping several clips."""
    idx = _cuda_index(device)
    if idx is None:
        return None
    return max(0, _cap_bytes(idx) - NON_ALLOCATOR_RESERVE_BYTES - torch.cuda.memory_allocated(idx))


def card_headroom_bytes(device: torch.device | str) -> int | None:
    """What the card itself leaves this process on `device`: its free memory (`torch.cuda.mem_get_info`) plus the
    memory this process has reserved but not allocated. None off CUDA. `plan_batches` lets it bound only sub-batches
    of more than one clip, and `BatchPlan.card_bytes` records it."""
    idx = _cuda_index(device)
    if idx is None:
        return None
    free, _total = torch.cuda.mem_get_info(idx)
    return max(0, free + torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx))


def require_allocator_bytes(n_bytes: int, device: torch.device | str, what: str) -> None:
    """Raise `GpuBudgetError` before allocating `n_bytes` that the fraction cap cannot hold (a no-op off CUDA)."""
    idx = _cuda_index(device)
    if idx is None:
        return
    cap = _cap_bytes(idx)
    left = cap - torch.cuda.memory_allocated(idx)
    if n_bytes > left:
        raise GpuBudgetError(f"{what} needs {n_bytes / 2**30:.2f} GiB but the CUDA budget has {left / 2**30:.2f} GiB "
                             f"left (cap {cap / 2**30:.2f} GiB from apply_gpu_budget / FASTDO_GPU_MEM_GB)")
