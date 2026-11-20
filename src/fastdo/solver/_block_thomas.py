"""Clip-batched block-Thomas recursion kernels behind `fastdo.solver.banded`.

Every kernel carries a leading clip dimension: `D` (B,N,b,b), `U` (B,N-1,b,b), right-hand sides
(B,N,b,K). Clips are independent chains (the coupling between two clips is exactly zero, see
`banded.ClipLayout`), so one sequential loop over the N block positions eliminates all B clips with one
batched kernel per step: the number of kernel launches does not grow with B. Blocks are 6x6 per frame,
or 12x12 over the frame pairs of `pair_frames` when a velocity prior couples frames k and k+2.

Device linear algebra without host synchronization. The LM loop allows one host read
per iteration, so the recursion must not synchronize, and whatever a library does internally is invisible to
Tensor hooks and to `torch.cuda.set_sync_debug_mode`. Only two operators are used:
- `torch.linalg.cholesky_ex` without its error check (a CUDA `torch.linalg.cholesky` reads its `info` flag
  back to the host). A non-positive-definite damped block gives NaN steps, which the LM rejects like any other
  cost-increasing step. torch's default heuristic may pick MAGMA for batched inputs when MAGMA is built in
  (this torch 2.9.1 wheel has it), so a solve pins cuSOLVER with `prefer_cusolver`. `api.solve_trajectories`
  enters it once per solve; the kernels here never touch the process-wide flag, so a direct CUDA
  caller of `banded.solve_block_tridiagonal` gets torch's default heuristic unless it enters the pin itself.
- `cholesky_solve` below: two `torch.linalg.solve_triangular` calls, which run on cuBLAS trsm / trsmBatched
  (no MAGMA option). `torch.cholesky_solve` is avoided on purpose. With the default heuristic, torch 2.9
  sends it to MAGMA whenever the batch holds more than one matrix, which is every B >= 2 solve, and to
  cuSOLVER only for a single matrix. A MAGMA batched call runs on its own queue with pinned host pointer
  arrays and can allocate `magma_malloc` workspaces. Hooks do not see its synchronizations, and
  `set_per_process_memory_fraction` does not govern its workspaces. On CPU (LAPACK) the two triangular solves
  gave results bitwise equal to `torch.cholesky_solve`.
`tests/solver/test_banded_batch.py` checks the operator set on CPU. On CUDA,
`test_batch.py::test_host_syncs_per_iteration` counts the profiler's runtime synchronize and cudaFree events.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import torch
from torch import Tensor


class _BackendPin:
    """State of `prefer_cusolver`'s process-wide pin: how many solves hold it and the preference to restore."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.holders = 0
        self.previous = torch._C._LinalgBackend.Default


_PIN = _BackendPin()


def _is_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


@contextlib.contextmanager
def prefer_cusolver(device: torch.device | str) -> Iterator[None]:
    """Pin cuSOLVER as the CUDA linear-algebra backend for a whole solve, then restore the caller's preference, also
    when the solve raises. `api.solve_trajectories` enters it once per solve, outside the LM loop, so
    the flag is set and restored once however many factorizations the solve runs.

    The flag is process-wide. Solves that overlap on several threads share one pin: a lock-guarded count makes the
    first entry save and set the flag and the last exit restore it, so no solve restores the flag under another
    solve that is still running. Code outside the solver that sets the flag during a solve is not guarded against.
    A CPU device leaves the flag alone."""
    if not _is_cuda(device):
        yield
        return
    with _PIN.lock:
        if _PIN.holders == 0:
            _PIN.previous = torch.backends.cuda.preferred_linalg_library()
            torch.backends.cuda.preferred_linalg_library("cusolver")
        _PIN.holders += 1
    try:
        yield
    finally:
        with _PIN.lock:
            _PIN.holders -= 1
            if _PIN.holders == 0:
                torch.backends.cuda.preferred_linalg_library(_PIN.previous)


def cholesky_solve(L: Tensor, rhs: Tensor) -> Tensor:
    """`(L L^T)^{-1} rhs` for lower Cholesky factors `L` (...,b,b) and `rhs` (...,b,K), by forward then backward
    triangular substitution (see the module docstring for why not `torch.cholesky_solve`)."""
    y = torch.linalg.solve_triangular(L, rhs, upper=False)
    return torch.linalg.solve_triangular(L.mT, y, upper=True)


def damped_diag_blocks(D: Tensor, lam: Tensor, floor: float) -> Tensor:
    """`lam * diag(H)` touches only the scalar diagonal entries of H, never the off-diagonal entries of
    a block itself; matches `lm.py`'s dense/banded damping convention exactly. `D` is (B,N,b,b) and
    `lam` (B,), one damping per clip."""
    diag_vals = torch.diagonal(D, dim1=-2, dim2=-1)                      # (B,N,b)
    eye = torch.eye(D.shape[-1], dtype=D.dtype, device=D.device)
    return D + lam.reshape(-1, 1, 1, 1) * torch.diag_embed(diag_vals) + floor * eye


def pair_frames(D: Tensor, U: Tensor, U2: Tensor, c: Tensor, g: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Regroup the pentadiagonal 6x6-block pose system of every clip into block-tridiagonal 12x12
    super-blocks over the frame pairs (2j, 2j+1). `D` (B,W,6,6), `U`
    (B,W-1,6,6), `U2` (B,W-2,6,6), `c`/`g` (B,W,6) with W even (`ClipLayout` pads every clip to W frames
    with identity dummy frames). Super-diagonal block j is [[D_2j, U_2j], [U_2j^T, D_2j+1]]; the block
    coupling pair j to pair j+1 is [[U2_2j, 0], [U_2j+1, U2_2j+1]] (frames 2j and 2j+3 are three apart,
    hence uncoupled). Returns (DD (B,S,12,12), UU (B,S-1,12,12), c (B,S,12), g (B,S,12)), S = W/2."""
    B, W = D.shape[0], D.shape[1]
    S = W // 2
    DD = torch.cat([torch.cat([D[:, 0::2], U[:, 0::2]], dim=-1),
                    torch.cat([U[:, 0::2].transpose(-1, -2), D[:, 1::2]], dim=-1)], dim=-2)
    UU = torch.cat([torch.cat([U2[:, 0::2], torch.zeros_like(U2[:, 0::2])], dim=-1),
                    torch.cat([U[:, 1::2], U2[:, 1::2]], dim=-1)], dim=-2)
    return DD, UU, c.reshape(B, S, 12, *c.shape[3:]), g.reshape(B, S, 12)   # c may carry K border columns


def forward_schur(D: Tensor, U: Tensor, lam: Tensor, floor: float) -> tuple[Tensor, Tensor, Tensor]:
    """Block-Cholesky (Thomas) forward pass, sequential over the N block positions and batched over the
    B clips: `D` (B,N,b,b), `U` (B,N-1,b,b), `lam` (B,). Returns `Dtilde` (B,N,b,b) (the Schur-reduced
    diagonal blocks `D_k'`), their lower Cholesky factors `L` (B,N,b,b), and the
    gain blocks `M[:, k] = Dtilde_k^{-1} U_k` (B,N-1,b,b) reused by the solve's back-substitution and
    the covariance recursion. Sequences are built as lists and `torch.stack`ed (autograd-safe). It never touches
    the process-wide linear-algebra backend flag (`prefer_cusolver` is entered once per solve)."""
    N = D.shape[1]
    Dd = damped_diag_blocks(D, lam, floor)
    Dtilde_list = [Dd[:, 0]]
    M_list: list[Tensor] = []
    L_list = [torch.linalg.cholesky_ex(Dd[:, 0]).L]
    for k in range(1, N):
        Mk1 = cholesky_solve(L_list[-1], U[:, k - 1])                # Dtilde_{k-1}^{-1} U_{k-1}
        M_list.append(Mk1)
        Dtilde_k = Dd[:, k] - U[:, k - 1].transpose(-1, -2) @ Mk1
        Dtilde_list.append(Dtilde_k)
        L_list.append(torch.linalg.cholesky_ex(Dtilde_k).L)
    Dtilde = torch.stack(Dtilde_list, dim=1)
    L = torch.stack(L_list, dim=1)
    B, b = D.shape[0], D.shape[-1]
    M = (torch.stack(M_list, dim=1) if M_list
         else torch.zeros(B, 0, b, b, dtype=D.dtype, device=D.device))
    return Dtilde, L, M


def thomas_solve(L: Tensor, M: Tensor, rhs: Tensor) -> Tensor:
    """Given `forward_schur`'s factors `L` (B,N,b,b) and gains `M` (B,N-1,b,b), solve the damped
    block-tridiagonal system `A X = rhs` for `rhs` (B,N,b,K) (K simultaneous right-hand sides) by forward
    substitution (eliminating the sub-diagonal via `M`) then backward substitution. `Dtilde_k^{-1} Bt_k`
    does not depend on the backward recursion once `Bt` is known, so it is one batched `cholesky_solve`
    over all (clip, block) positions."""
    N = rhs.shape[1]
    Bt_list = [rhs[:, 0]]
    for k in range(1, N):
        Bt_list.append(rhs[:, k] - M[:, k - 1].transpose(-1, -2) @ Bt_list[-1])
    Bt = torch.stack(Bt_list, dim=1)
    Y = cholesky_solve(L, Bt)
    X_list: list[Tensor] = [Y[:, N - 1]]
    for k in range(N - 2, -1, -1):
        X_list.append(Y[:, k] - M[:, k] @ X_list[-1])
    X_list.reverse()
    return torch.stack(X_list, dim=1)


def rts_covariances(L: Tensor, M: Tensor) -> Tensor:
    """Block marginals of the damped block-tridiagonal inverse via the backward Rauch-Tung-Striebel
    recursion `Sigma_{N-1} = Dtilde_{N-1}^{-1}`, `Sigma_k = Dtilde_k^{-1} + M_k Sigma_{k+1} M_k^T`,
    batched over clips: `L` (B,N,b,b), `M` (B,N-1,b,b) -> (B,N,b,b)."""
    B, N, b = L.shape[0], L.shape[1], L.shape[-1]
    eye = torch.eye(b, dtype=L.dtype, device=L.device).expand(B, N, b, b).contiguous()
    Dinv = cholesky_solve(L, eye)
    Sigma_list = [Dinv[:, N - 1]]
    for k in range(N - 2, -1, -1):
        Sigma_list.append(Dinv[:, k] + M[:, k] @ Sigma_list[-1] @ M[:, k].transpose(-1, -2))
    Sigma_list.reverse()
    return torch.stack(Sigma_list, dim=1)
