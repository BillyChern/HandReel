"""Generic Levenberg-Marquardt over per-frame vector-space blocks, block-tridiagonal + banded solve.

Unlike `fastdo.solver.lm` (SE(3)^T x R+ with a *dense* 6T+1 normal-equations solve, one shared
scale parameter), this module makes no assumption about the state space beyond "a vector `x_k` of
fixed dimension `n` per frame `k`" -- the kinematic retargeting uses it with `x_k = q_k[robot_dof]` (28 MuJoCo hand+
wrist dofs), and other callers may reuse it with a different `n`. State is `x (T, n)`.

Per-frame terms (`BlockTerm`) only ever couple a residual to its OWN frame's block (block-diagonal
contribution to the Gauss-Newton Hessian); the only cross-frame coupling comes from `SmoothTerm`,
a pairwise residual between adjacent frames `k`/`k+1` (block-tridiagonal off-diagonal contribution).
The resulting `T*n x T*n` normal-equations matrix is therefore block-tridiagonal with bandwidth
`2n-1`: assembled directly into `scipy.linalg.solveh_banded`'s upper-form banded storage (no dense
`T*n x T*n` matrix is ever built), so this scales to long trajectories the dense M1 solver would not.

Convention shared with `fastdo.solver.lm`/`fastdo.solver.terms.TermOutput`: a term's `residual`/
`jacobian` already carry any `sqrt(lambda)` weighting baked in (see `fastdo.solver.terms`'s module
docstring); `weight` is an additional 0/1-ish inclusion mask (e.g. to drop a residual row without
changing the tensor's shape across frames), not a second copy of the same scaling.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import solveh_banded
from threadpoolctl import ThreadpoolController
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "BlockLMConfig",
    "BlockLMResult",
    "BlockTerm",
    "Evaluate",
    "SmoothTerm",
    "assemble_block_system",
    "freeze_active_bounds",
    "lm_optimize_blocks",
    "solve_block_tridiagonal",
    "total_block_cost",
]


@dataclass(frozen=True)
class BlockTerm:
    """One residual term, block-diagonal in the per-frame state (no cross-frame coupling).

    `residual`/`jacobian` are already flattened over whatever per-residual structure the caller's
    term has (e.g. 3 components per point): `residual (T, R)`, `jacobian (T, R, n)`. `weight (T, R)`
    is a 0/1-ish inclusion mask (see module docstring); the cost contribution is
    `(weight * residual**2).sum()`.
    """

    residual: Tensor
    jacobian: Tensor
    weight: Tensor


@dataclass(frozen=True)
class SmoothTerm:
    """Pairwise residual between adjacent frames `k`/`k+1`: the only cross-frame coupling.

    `residual (T-1, n)`; `jac_k`/`jac_k1` (T-1, n, n) are d(residual)/d(x_k) and d(residual)/d(x_{k+1});
    `weight (T-1, n)` is the same 0/1-ish inclusion mask convention as `BlockTerm`.
    """

    residual: Tensor
    jac_k: Tensor
    jac_k1: Tensor
    weight: Tensor


Evaluate = Callable[[Tensor, bool], tuple[list[BlockTerm], SmoothTerm | None]]


@dataclass(frozen=True)
class BlockLMConfig:
    """`blas_threads` limits BLAS threads around the banded solve (None: no limit). With
    `final_refresh=False` the accepted step that ends the loop is not refreshed (lever L12, `lm_optimize_blocks`)."""

    max_iter: int = 30
    lam0: float = 1e-2
    lam_up: float = 5.0
    lam_down: float = 3.0
    rel_tol: float = 1e-4
    blas_threads: int | None = None
    final_refresh: bool = True


@dataclass(frozen=True)
class BlockLMResult:
    x: Tensor
    cost: float
    iterations: int


def _weighted_jr(term_or_smooth_residual: Tensor, weight: Tensor) -> Tensor:
    sw = weight.clamp_min(0.0).sqrt().to(torch.float64)
    return term_or_smooth_residual.to(torch.float64) * sw


def assemble_block_system(terms: list[BlockTerm], smooth: SmoothTerm | None, n_frames: int, n_block: int,
                          ) -> tuple[Tensor, Tensor | None, Tensor]:
    """Block-diagonal Hessian, block-off-diagonal (from smoothness), and gradient.

    Returns `(diag (T,n,n), offdiag (T-1,n,n) or None, g (T,n))`. `offdiag[k]` couples frame `k`'s
    row block to frame `k+1`'s column block (`H[k,k+1]`); the matrix is symmetric, so `H[k+1,k] =
    offdiag[k].T` is implied, not stored separately (matches `scipy.linalg.solveh_banded`'s
    upper-form convention used downstream).
    """
    T, n = n_frames, n_block
    device = terms[0].residual.device if terms else (smooth.residual.device if smooth else "cpu")
    diag = torch.zeros(T, n, n, dtype=torch.float64, device=device)
    g = torch.zeros(T, n, dtype=torch.float64, device=device)
    for term in terms:
        sw = term.weight.clamp_min(0.0).sqrt().to(torch.float64)                 # (T,R)
        Jw = term.jacobian.to(torch.float64) * sw[..., None]                     # (T,R,n)
        rw = term.residual.to(torch.float64) * sw                                # (T,R)
        diag += torch.einsum("trn,trm->tnm", Jw, Jw)
        g += torch.einsum("trn,tr->tn", Jw, rw)

    offdiag: Tensor | None = None
    if smooth is not None:
        sw = smooth.weight.clamp_min(0.0).sqrt().to(torch.float64)               # (T-1,n)
        Jk = smooth.jac_k.to(torch.float64) * sw[..., None]                      # (T-1,n,n)
        Jk1 = smooth.jac_k1.to(torch.float64) * sw[..., None]                    # (T-1,n,n)
        rw = smooth.residual.to(torch.float64) * sw                              # (T-1,n)
        diag[:-1] += torch.einsum("kan,kam->knm", Jk, Jk)
        diag[1:] += torch.einsum("kan,kam->knm", Jk1, Jk1)
        offdiag = torch.einsum("kan,kam->knm", Jk, Jk1)                          # (T-1,n,n)
        g[:-1] += torch.einsum("kan,ka->kn", Jk, rw)
        g[1:] += torch.einsum("kan,ka->kn", Jk1, rw)
    return diag, offdiag, g


def total_block_cost(terms: list[BlockTerm], smooth: SmoothTerm | None) -> float:
    cost = torch.zeros((), dtype=torch.float64)
    for term in terms:
        w = term.weight.clamp_min(0.0).to(torch.float64)
        cost = cost + (w * term.residual.to(torch.float64) ** 2).sum()
    if smooth is not None:
        w = smooth.weight.clamp_min(0.0).to(torch.float64)
        cost = cost + (w * smooth.residual.to(torch.float64) ** 2).sum()
    return float(cost)


@functools.lru_cache(maxsize=16)
def _banded_indices(n_frames: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices into `ab.ravel()` (`ab` of shape `(2n, T n)`) of the diagonal-block upper triangles (frames in
    order, `np.triu_indices(n)` within a block) and of the off-diagonal blocks (row-major `(T-1, n, n)`).
    Read-only: the arrays are shared by every call with the same shape."""
    M, u = n_frames * n, 2 * n - 1
    iu, ju = np.triu_indices(n)
    frame = np.arange(n_frames)[:, None]
    rows, cols = frame * n + iu[None, :], frame * n + ju[None, :]
    diag_flat = ((u + rows - cols) * M + cols).ravel()
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    kframe = np.arange(max(n_frames - 1, 0))[:, None]
    off_rows, off_cols = kframe * n + ii.ravel()[None, :], kframe * n + n + jj.ravel()[None, :]
    off_flat = ((u + off_rows - off_cols) * M + off_cols).ravel()
    diag_flat.setflags(write=False)
    off_flat.setflags(write=False)
    return diag_flat, off_flat


@functools.cache
def _blas_controller() -> ThreadpoolController:
    """One reused controller: `threadpool_limits` rescans the loaded libraries on every enter (0.37 ms vs 0.009 ms)."""
    return ThreadpoolController()


def _banded_from_blocks(diag: np.ndarray, offdiag: np.ndarray | None, n: int, lam: float) -> np.ndarray:
    """Upper-form banded storage (`scipy.linalg.solveh_banded`'s `ab`) for LM-damped block-tridiagonal H.

    `ab[u + i - j, j] = H[i, j]` for `max(0, j-u) <= i <= j`, `u = 2n-1`. Damping `lam * diag(H)` is
    applied to each frame's own diagonal block before it is written into `ab` (a fixed `1e-9` floor
    keeps the solve well-posed for a structurally unobserved dof, matching `fastdo.solver.lm.lm_step`).
    """
    T = diag.shape[0]
    M = T * n
    u = 2 * n - 1
    ab = np.zeros((u + 1) * M, dtype=np.float64)
    diag_flat, off_flat = _banded_indices(T, n)
    iu, ju = np.triu_indices(n)                                     # within-block upper triangle
    damped = diag.copy()
    diag_idx = np.arange(n)
    damped[:, diag_idx, diag_idx] += lam * diag[:, diag_idx, diag_idx] + 1e-9
    ab[diag_flat] = damped[:, iu, ju].ravel()
    if offdiag is not None:
        ab[off_flat] = offdiag.reshape(-1)
    return ab.reshape(u + 1, M)


def solve_block_tridiagonal(diag: Tensor, offdiag: Tensor | None, g: Tensor, lam: float,
                            blas_threads: int | None = None) -> Tensor:
    """LM-damped block-tridiagonal normal-equations solve via `scipy.linalg.solveh_banded`.

    `diag (T,n,n)`, `offdiag (T-1,n,n)` or `None`, `g (T,n)` -> `delta (T,n)` solving
    `(H + lam*diag(H) + 1e-9 I) delta = -g` where `H` is the block-tridiagonal matrix `diag`/`offdiag`
    represent (CPU float64 throughout; `scipy` has no batched/GPU banded solver). With `blas_threads`, the LAPACK
    call runs under that BLAS thread limit (single-threaded is faster for these small bands on a shared host).
    """
    T, n = g.shape
    diag_np = diag.detach().cpu().numpy()
    off_np = offdiag.detach().cpu().numpy() if offdiag is not None else None
    ab = _banded_from_blocks(diag_np, off_np, n, lam)
    rhs = (-g.detach().cpu().numpy().reshape(T * n)).astype(np.float64)
    if blas_threads is None:
        sol = solveh_banded(ab, rhs, lower=False)
    else:
        with _blas_controller().limit(limits=blas_threads, user_api="blas"):
            sol = solveh_banded(ab, rhs, lower=False)
    return torch.from_numpy(sol.reshape(T, n)).to(g.dtype).to(g.device)


def _box(bounds: tuple[Tensor, Tensor], n: int, like: Tensor) -> tuple[Tensor, Tensor]:
    """`(lower, upper)` as `(n,)` tensors of `like`'s dtype; ValueError unless `(n,)` each with `lower <= upper`."""
    lower, upper = (torch.as_tensor(b, dtype=like.dtype, device=like.device) for b in bounds)
    if tuple(lower.shape) != (n,) or tuple(upper.shape) != (n,):
        raise ValueError(f"bounds must be two ({n},) tensors, got {tuple(lower.shape)} and {tuple(upper.shape)}")
    if bool(torch.isnan(lower).any() | torch.isnan(upper).any()) or bool((lower > upper).any()):
        raise ValueError("bounds must satisfy lower <= upper with no NaN (use -inf / inf for an unbounded entry)")
    return lower, upper


def freeze_active_bounds(diag: Tensor, offdiag: Tensor | None, g: Tensor, active: Tensor,
                         ) -> tuple[Tensor, Tensor | None, Tensor]:
    """The normal equations with every `active (T,n)` entry held fixed: its row and column of `H` zeroed (diagonal and
    off-diagonal blocks), its diagonal set to 1 and its gradient to 0, so the LM step leaves it at 0 and the other
    entries solve the reduced system exactly (the projected-Newton active-set step). Inputs are not modified."""
    if not bool(active.any()):
        return diag, offdiag, g
    keep = (~active).to(diag.dtype)                                                  # (T,n)
    diag = diag * keep[:, :, None] * keep[:, None, :] + torch.diag_embed(active.to(diag.dtype))
    if offdiag is not None:
        offdiag = offdiag * keep[:-1, :, None] * keep[1:, None, :]
    return diag, offdiag, g * keep


def lm_optimize_blocks(x0: Tensor, evaluate: Evaluate, cfg: BlockLMConfig,
                       on_final_accept: Callable[[Tensor], None] | None = None,
                       bounds: tuple[Tensor, Tensor] | None = None) -> BlockLMResult:
    """Block LM: state `x (T,n)`, `evaluate(x, refresh) -> (terms, smooth)` (see `Evaluate`).

    `refresh` mirrors `fastdo.solver.lm.lm_optimize`'s own convention: `True` at the initial
    evaluation and after every accepted step (the caller may use it to refresh a cached, otherwise-
    fixed quantity such as an OT plan -- the majorize-minimize / envelope-theorem pattern this
    module's terms are built around); `False` for a trial step, where a cheaper, fixed-plan
    evaluation suffices to decide accept/reject.

    With `cfg.final_refresh=False`, an accepted step that ends the loop (`rel < cfg.rel_tol` or the last iteration)
    calls `on_final_accept(x)` instead of `evaluate(x, True)`, and the returned `cost` is that trial's. `x` and
    `iterations` are unchanged for a deterministic `evaluate` (the final refresh's plan and Jacobians
    are never read).

    `bounds = (lower (n,), upper (n,))` makes this a projected LM on the box `lower <= x_k <= upper` (every
    frame, `-inf`/`inf` for an unbounded entry): `x0` is projected onto the box, every step is computed with the
    active bounds frozen (`freeze_active_bounds`: an entry at a bound whose gradient points out of the box), and the
    trial is projected onto the box before its cost is evaluated, so every evaluated and returned `x` is feasible.
    `bounds=None` runs the unbounded loop unchanged.
    """
    x = x0.clone()
    n = x.shape[1]
    box = None if bounds is None else _box(bounds, n, x)
    if box is not None:
        x = torch.maximum(torch.minimum(x, box[1]), box[0])
    lam = cfg.lam0
    terms, smooth = evaluate(x, True)
    cost = total_block_cost(terms, smooth)
    it = 0
    system: tuple[Tensor, Tensor | None, Tensor] | None = None
    for it in range(1, cfg.max_iter + 1):
        # A rejected trial leaves `x`, `terms` and `smooth` unchanged, hence the (frozen) normal
        # equations too: they are assembled once per accepted state, not once per iteration.
        if system is None:
            diag, offdiag, g = assemble_block_system(terms, smooth, x.shape[0], n)
            if box is not None:
                active = ((x <= box[0]) & (g > 0.0)) | ((x >= box[1]) & (g < 0.0))
                diag, offdiag, g = freeze_active_bounds(diag, offdiag, g, active)
            system = (diag, offdiag, g)
        diag, offdiag, g = system
        delta = solve_block_tridiagonal(diag, offdiag, g, lam, blas_threads=cfg.blas_threads)
        trial = x + delta.to(x.dtype)
        if box is not None:
            trial = torch.maximum(torch.minimum(trial, box[1]), box[0])
        t_terms, t_smooth = evaluate(trial, False)
        trial_cost = total_block_cost(t_terms, t_smooth)
        if trial_cost < cost:
            rel = (cost - trial_cost) / max(cost, 1e-12)
            x = trial
            lam = max(lam / cfg.lam_down, 1e-9)
            if not cfg.final_refresh and (rel < cfg.rel_tol or it == cfg.max_iter):
                if on_final_accept is not None:
                    on_final_accept(x)
                cost = trial_cost
                logger.debug("iter %d accepted (final, no refresh) cost %.6g", it, cost)
                break
            terms, smooth = evaluate(x, True)
            system = None
            cost = total_block_cost(terms, smooth)
            logger.debug("iter %d accepted cost %.6g lam %.2e", it, cost, lam)
            if rel < cfg.rel_tol:
                break
        else:
            lam *= cfg.lam_up
            logger.debug("iter %d rejected trial %.6g > %.6g lam %.2e", it, trial_cost, cost, lam)
            if lam > 1e6:
                break
    return BlockLMResult(x=x, cost=cost, iterations=it)
