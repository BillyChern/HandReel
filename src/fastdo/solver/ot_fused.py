"""Fused CUDA kernels for the dense log-domain Sinkhorn solve.

The batched-solve profile shows that once B clips share one solve, the GPU work per
clip is dominated by the dense coarse/mid-level Sinkhorn: every half-iteration
`f_i = coef * logsumexp_j((g_j - C_ij) / eps + log b_j)` runs about seven full passes over the (T,N,M)
cost tensor in eager PyTorch (broadcast subtract, divide, add, then the max / subtract / exp / sum of
`torch.logsumexp`), each allocating a (T,N,M) temporary. The Triton kernels below evaluate the same
expression per row (or per column) of C in one pass with no temporaries: one program per (frame, row),
vectorized over the row, computing max-shifted log-sum-exp. `plan_from_potentials` materializes the
final plan `exp((f_i + g_j - C_ij) / eps + log a_i + log b_j)` the same way, one row per program.

These are the same floating-point expressions evaluated in a different reduction order, so results agree
with the eager expressions to float round-off (the reduction order of the row sum differs), not bit for
bit. They run only when the caller opts in (`ot.sinkhorn_unbalanced_potentials(fused=True)`,
`SolverConfig.fused_sinkhorn`), and only for CUDA tensors that do not require gradients.

Device memory outside the caching allocator. Triton launches each program as a CUDA
block of 32 x num_warps = 128 threads with `metadata.shared` bytes of per-block shared memory (16 bytes for
these kernels). The CUDA driver allocates the loaded module and whatever a launch needs itself, so
`torch.cuda.set_per_process_memory_fraction` does not bound it; `require_allocator_bytes` only guards the
plan tensor. The first `plan_from_potentials` was one elementwise program over 2^16 entries: its cached H100
module was 1.7-4.7 MB against 12-43 KB for the row kernels, and a fresh assembly of it for sm_90 did not
finish in 12 minutes on one CPU core, where a row kernel takes 0.3 s. Every GPU job the shared-host guard
killed at 7.5-11 GB of process memory under a 4 GiB fraction had launched that kernel; whether its launch
memory caused those readings is not established without a GPU. The plan
kernel is now a row kernel like the other two, and the fused path stays opt-in until it is measured.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fastdo.solver.memory import require_allocator_bytes

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton ships with the CUDA torch wheels this project pins
    triton = None

__all__ = ["fused_available", "lse_cols", "lse_rows", "plan_from_potentials"]


def fused_available(*tensors: Tensor) -> bool:
    """True when the fused kernels apply: triton importable, every tensor on CUDA, no gradient tracked."""
    if triton is None or not all(t.is_cuda for t in tensors):
        return False
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))


if triton is not None:

    @triton.jit
    def _lse_kernel(C, pot, logm, eps, out, n_red, n_keep, s_t, s_keep, s_red, sp_t, sp_red, sl_t, sl_red,
                    BLOCK: tl.constexpr):  # pragma: no cover - compiled for the GPU
        r = tl.program_id(0).cast(tl.int64)
        t = r // n_keep
        kept = r % n_keep
        red = tl.arange(0, BLOCK).cast(tl.int64)
        valid = red < n_red
        c = tl.load(C + t * s_t + kept * s_keep + red * s_red, mask=valid, other=0.0)
        p = tl.load(pot + t * sp_t + red * sp_red, mask=valid, other=0.0)
        lm = tl.load(logm + t * sl_t + red * sl_red, mask=valid, other=0.0)
        e = tl.load(eps + t)
        x = tl.where(valid, (p - c) / e + lm, float("-inf"))
        m = tl.max(x, axis=0)
        tl.store(out + r, m + tl.log(tl.sum(tl.exp(x - m), axis=0)))

    @triton.jit
    def _plan_kernel(C, f, g, loga, logb, eps, out, N, M, s_t, s_i, s_j, sf_t, sf_i, sg_t, sg_j, sa_t, sa_i, sb_t, sb_j,
                     BLOCK: tl.constexpr):  # pragma: no cover - compiled for the GPU
        r = tl.program_id(0).cast(tl.int64)                  # one program per (frame, row): r = t * N + i
        t = r // N
        i = r % N
        j = tl.arange(0, BLOCK).cast(tl.int64)
        valid = j < M
        c = tl.load(C + t * s_t + i * s_i + j * s_j, mask=valid, other=0.0)
        fv = tl.load(f + t * sf_t + i * sf_i)
        gv = tl.load(g + t * sg_t + j * sg_j, mask=valid, other=0.0)
        la = tl.load(loga + t * sa_t + i * sa_i)
        lb = tl.load(logb + t * sb_t + j * sb_j, mask=valid, other=0.0)
        e = tl.load(eps + t)
        tl.store(out + r * M + j, tl.exp((fv + gv - c) / e + la + lb), mask=valid)


def _eps_rows(eps: float | Tensor, C: Tensor) -> Tensor:
    if isinstance(eps, Tensor):
        return eps.to(C.dtype).reshape(C.shape[0]).contiguous()
    return torch.full((C.shape[0],), eps, dtype=C.dtype, device=C.device)


def lse_rows(C: Tensor, g: Tensor, log_b: Tensor, eps: float | Tensor) -> Tensor:
    """`logsumexp_j((g[t,j] - C[t,i,j]) / eps[t] + log_b[t,j])` -> (T,N) for C (T,N,M), g/log_b (T,M)."""
    T, N, M = C.shape
    out = torch.empty(T, N, dtype=C.dtype, device=C.device)
    _lse_kernel[(T * N,)](C, g, log_b, _eps_rows(eps, C), out, M, N, C.stride(0), C.stride(1), C.stride(2),
                          g.stride(0), g.stride(1), log_b.stride(0), log_b.stride(1),
                          BLOCK=triton.next_power_of_2(M))
    return out


def lse_cols(C: Tensor, f: Tensor, log_a: Tensor, eps: float | Tensor) -> Tensor:
    """`logsumexp_i((f[t,i] - C[t,i,j]) / eps[t] + log_a[t,i])` -> (T,M) for C (T,N,M), f/log_a (T,N)."""
    T, N, M = C.shape
    out = torch.empty(T, M, dtype=C.dtype, device=C.device)
    _lse_kernel[(T * M,)](C, f, log_a, _eps_rows(eps, C), out, N, M, C.stride(0), C.stride(2), C.stride(1),
                          f.stride(0), f.stride(1), log_a.stride(0), log_a.stride(1),
                          BLOCK=triton.next_power_of_2(N))
    return out


def plan_from_potentials(C: Tensor, f: Tensor, g: Tensor, log_a: Tensor, log_b: Tensor,
                         eps: float | Tensor) -> Tensor:
    """`exp((f[t,i] + g[t,j] - C[t,i,j]) / eps[t] + log_a[t,i] + log_b[t,j])` -> (T,N,M), one pass."""
    T, N, M = C.shape
    # The one (T,N,M) output of the fused path: fail with the budget in the message before compiling or launching.
    require_allocator_bytes(T * N * M * C.element_size(), C.device, f"the fused Sinkhorn plan ({T}x{N}x{M})")
    out = torch.empty(T, N, M, dtype=C.dtype, device=C.device)             # contiguous: row r starts at r * M
    _plan_kernel[(T * N,)](
        C, f, g, log_a, log_b, _eps_rows(eps, C), out, N, M, C.stride(0), C.stride(1), C.stride(2),
        f.stride(0), f.stride(1), g.stride(0), g.stride(1), log_a.stride(0), log_a.stride(1), log_b.stride(0),
        log_b.stride(1), BLOCK=triton.next_power_of_2(M))
    return out
