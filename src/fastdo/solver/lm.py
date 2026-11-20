"""Levenberg-Marquardt on SE(3)^T x R+ with block-tridiagonal normal equations (6T+1 unknowns).

`lm_optimize` solved each step with an O(n) arrow-banded elimination (`lm_step(...,
solver="banded")`) via scipy's `solveh_banded` rather than an O(n^3) dense factorization, exploiting the
block-tridiagonal (in the 6T pose coordinates) + dense-arrow (the one shared log-scale parameter)
structure that `assemble_normal_equations` always produces.

The single-threaded-BLAS fix for `solveh_banded` (see `_lm_step_banded`) moved from an
import-time `os.environ.setdefault` (which silently mutated the process environment for every importer
of `fastdo`, including ones that never call the banded solve path) to a `threadpoolctl.threadpool_limits`
context manager scoped to the `solveh_banded` call itself.

`assemble_normal_equations`'s Python loop over frames and the dense pseudo-inverse covariance
are replaced by `fastdo.solver.banded` (design Section 11.6): `BlockTridiagonal` assembly is a batched
contraction, `solve_block_tridiagonal` is a plain-torch block-Thomas recursion plus a rank-one Schur
complement for the shared scale, and `marginal_covariances` gives per-frame covariance blocks in O(T)
via the backward (Rauch-Tung-Striebel smoother, "RTS") recursion. `lm_optimize` now assembles and solves through that
block form directly and never densifies inside its loop; `assemble_normal_equations` remains as a thin
wrapper (`densify(assemble_block_tridiagonal(...))`) kept for tests, small problems, and `LMResult.H`.
`lm_step`'s dense and scipy-`"banded"` paths (`_lm_step_banded`) are kept verbatim as the reference
implementations the equivalence tests in `test_lm.py` check the new block path against; `lm_optimize`
itself no longer calls either.

An evaluate callback may return a `banded.Priors` (constant-pose triple plus the
constant-velocity quadruple of `terms.velocity_term`) in place of the bare smoothness triple;
`total_cost` adds the velocity residual and the block solve regroups frame pairs (see `banded.py`). The
scipy reference path (`_BAND_U = 11`) only covers the tridiagonal system without a velocity prior.

`lm_optimize_batch` runs B concatenated clips (`banded.ClipLayout`) as one decoupled system.
Each clip keeps its own damping `lam_b`, accept/reject decision and convergence flag, all computed on the
device from per-clip costs; a converged clip is frozen by masking its steps (`torch.where`), and the
loop reads its per-clip flags back to the host exactly once per iteration. The loop keeps per-frame
normal-equation blocks, so each trial evaluates only the active clips' frames and each refresh only the
accepted clips' frames (`BatchEvaluate`'s `frames`): a stopped clip, or one whose trial was rejected,
costs no evaluation and keeps its cached correspondences. `lm_optimize` is the B = 1 case of the same loop.

Anisotropic scale (`SolverConfig.anisotropic_scale`): the per-clip state may be (B,3) log-stretches instead of (B,)
log-scales. The loop is unchanged apart from broadcasting its per-clip masks over the extra axis; the evaluate callback
returns terms with 9 Jacobian columns (the isotropy prior among them), so assembly, merit and acceptance include the
prior.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast

import numpy as np
import threadpoolctl
import torch
from scipy.linalg import solveh_banded
from torch import Tensor

from fastdo.geometry.lie import apply_left
from fastdo.solver.banded import (
    BlockTridiagonal,
    ClipLayout,
    PriorsLike,
    as_priors,
    assemble_block_tridiagonal,
    assemble_from_blocks,
    clip_sum,
    densify,
    solve_block_tridiagonal,
    term_blocks,
)
from fastdo.solver.terms import TermOutput

logger = logging.getLogger(__name__)
Evaluate = Callable[[Tensor, Tensor, bool], tuple[list[TermOutput], PriorsLike]]


class BatchEvaluate(Protocol):
    """`evaluate(poses (T,4,4), log_scales (B,) or (B,3), refresh, frames)`, positional. `refresh` is False for a trial
    evaluation (cached correspondences) and True to refresh them. `frames` (None: every frame) is a device
    index of whole clips' frames in clip order (`ClipLayout.frames_of`): only those frames are evaluated
    (and, with `refresh`, only their cached correspondences change), and the data terms come back for those
    frames only, in that order. The priors always cover every frame."""

    def __call__(self, poses: Tensor, log_scales: Tensor, refresh: bool, frames: Tensor | None = None, /
                 ) -> tuple[list[TermOutput], PriorsLike]: ...

# Bandwidth (number of super/sub-diagonals) of the 6T-dim pose block of H, in the parameter layout
# [xi_0, ..., xi_{T-1}]: a frame's own 6x6 diagonal block spans offsets 0..5 and the smoothness term's
# cross block between frame k and k+1 spans up to offset 11 (row 6k, column 6k+11). Only used by the
# reference `_lm_step_banded` (scipy) path now; `lm_optimize` itself uses the block-Thomas path.
_BAND_U = 11


@dataclass(frozen=True)
class LMConfig:
    max_iter: int = 30
    lam0: float = 1e-2
    lam_up: float = 5.0
    lam_down: float = 3.0
    rel_tol: float = 1e-4
    fix_scale: bool = False
    # A trial step whose per-clip MEDIAN object depth is at or below this (metres) is rejected like a
    # cost increase. The median, not every frame: unobserved rows may legitimately be extrapolated behind the camera
    # and must not stall the solve of the observed ones. None = no guard (the historical behaviour).
    min_median_depth: float | None = None


@dataclass(frozen=True)
class LMResult:
    """Result of `lm_optimize`.

    `system` is the `BlockTridiagonal` Gauss-Newton system (`J^T W J` in the 6T+1 global parameter
    layout) assembled at the *returned* `poses`/`log_scale` -- i.e. from correspondences refreshed at
    the final state, not from whichever earlier iterate happened to produce the last accepted step.
    Downstream consumers use `marginal_covariances(system)` as the precision matrix of a Laplace
    approximation to parameter covariance. `H` densifies `system` on access (no longer a stored
    dense (6T+1)^2 tensor) for tests and callers that want the whole matrix.
    """

    poses: Tensor
    log_scale: Tensor
    cost: float
    iterations: int
    system: BlockTridiagonal

    @property
    def H(self) -> Tensor:
        H, _g = densify(self.system)
        return H


@dataclass(frozen=True)
class BatchLMResult:
    """Result of `lm_optimize_batch` for B clips. `poses` (T,4,4), `log_scale` (B,) and `cost` (B,) stay
    on the device; `iterations` and `accepted` (each clip's accept/reject decision at every iteration
    it was active) are host-side; `loop_iterations` counts the shared loop's iterations, i.e. its host
    synchronizations. `system` is assembled at the returned state, as `LMResult.system`."""

    poses: Tensor
    log_scale: Tensor
    cost: Tensor
    iterations: tuple[int, ...]
    accepted: tuple[tuple[bool, ...], ...]
    loop_iterations: int
    system: BlockTridiagonal


def assemble_normal_equations(terms: list[TermOutput], smooth: PriorsLike, n_frames: int
                              ) -> tuple[Tensor, Tensor]:
    """Thin wrapper around `assemble_block_tridiagonal`/`densify`: kept for tests, small
    problems, and any caller wanting the dense (6T+1)^2 system directly. `lm_optimize`'s hot loop calls
    `assemble_block_tridiagonal` itself and never densifies."""
    return densify(assemble_block_tridiagonal(terms, smooth, n_frames))


def total_cost(terms: list[TermOutput], smooth: PriorsLike) -> Tensor:
    cost = torch.zeros((), dtype=torch.float64)
    for term in terms:
        w = term.weight.clamp_min(0.0).to(torch.float64)
        cost = cost + (w * (term.residual.to(torch.float64) ** 2).sum(-1)).sum()
        if term.extra is not None:                # the unbalanced marginal penalty, flag-gated
            cost = cost + term.extra.to(torch.float64).sum()
    priors = as_priors(smooth)
    for prior in (priors.smooth, priors.velocity):
        if prior is not None:
            cost = cost + (prior[0].to(torch.float64) ** 2).sum()
    return cost


def clip_costs(terms: list[TermOutput], smooth: PriorsLike, layout: ClipLayout, frames: Tensor | None = None
               ) -> Tensor:
    """Per-clip total cost (B,) float64: `total_cost` summed per frame, then per clip over
    `layout.clip_index` (prior residual k is charged to frame k's clip; across a clip boundary it is 0).
    With `frames` the data terms cover only those frames (`BatchEvaluate`), and the clips they do not
    touch carry their prior cost alone."""
    ci = layout.clip_index
    rows = ci if frames is None else ci.index_select(0, frames)
    cost = torch.zeros(layout.n_clips, dtype=torch.float64, device=ci.device)
    for term in terms:
        if term.merit is not None:                # the robust loss the step descends, not Σ w·‖r‖²
            per_frame = term.merit.to(torch.float64)
        else:
            w = term.weight.clamp_min(0.0).to(torch.float64)
            per_frame = (w * (term.residual.to(torch.float64) ** 2).sum(-1)).flatten(1).sum(1)
        if term.extra is not None:                # per frame already, so it adds straight on
            per_frame = per_frame + term.extra.to(torch.float64)
        cost = cost + clip_sum(per_frame, rows, layout.n_clips)
    priors = as_priors(smooth)
    for prior, rows in ((priors.smooth, ci[:-1]), (priors.velocity, ci[:max(ci.shape[0] - 2, 0)])):
        if prior is not None:
            cost = cost + clip_sum((prior[0].to(torch.float64) ** 2).sum(-1), rows, layout.n_clips)
    return cost


def _pose_block_to_upper_banded(A: Tensor, u: int) -> np.ndarray:
    """Symmetric (n,n) tensor -> scipy `solveh_banded` upper-form array (u+1,n): row `i` holds the
    diagonal at offset `u-i` (row `u` is the main diagonal). Only the O(u*n) band is ever moved off the
    GPU, not the full O(n^2) dense matrix.
    """
    n = A.shape[0]
    ab = torch.zeros(u + 1, n, dtype=A.dtype, device=A.device)
    for row in range(u + 1):
        offset = u - row
        d = torch.diagonal(A, offset=offset)
        ab[row, offset:offset + d.shape[0]] = d
    return ab.cpu().numpy()


def _lm_step_banded(H: Tensor, g: Tensor, lam: float, u: int = _BAND_U) -> Tensor:
    """Arrow-banded solve for H = [[A, c], [c^T, d]] with A the (6T,6T) block-tridiagonal pose block
    (bandwidth `u`) and (c, d) the dense arrow row/column coupling every frame to the one shared
    log-scale parameter. Eliminates the arrow via the Schur complement: `A x1 = -g_pose`, `A x2 = c`
    (one banded solve, two right-hand sides, O(u^2 n) instead of the dense solve's O(n^3)), then
    `delta_sigma = (-g_sigma - c^T x1) / (d - c^T x2)` and `delta_pose = x1 - delta_sigma * x2`. LM
    damping (`lam * diag(H)`) and the `1e-9` ridge are applied to the diagonal exactly as in the dense
    path (`lm_step` with `solver="dense"`); the two must match to 1e-9 (see
    `test_lm_step_banded_matches_dense_on_assembled_system` in test_lm.py).

    Raises `ValueError` if the pose block has a nonzero entry beyond offset `u`
    (for example the (k, k+2) blocks of a velocity-prior system, which reach offset 17), which the band
    extraction below would otherwise silently drop.
    """
    device, dtype = H.device, H.dtype
    n = H.shape[0]
    if bool(torch.triu(H[:-1, :-1], diagonal=u + 1).ne(0).any()):
        raise ValueError(f"H has nonzero pose-block entries beyond bandwidth {u}; use solve_block_tridiagonal")
    damped_diag = torch.diagonal(H) * (1.0 + lam) + 1e-9
    A = H[:-1, :-1].clone()
    A[torch.arange(n - 1, device=device), torch.arange(n - 1, device=device)] = damped_diag[:-1]
    c = H[:-1, -1]
    d = damped_diag[-1]
    ab = _pose_block_to_upper_banded(A, u)
    rhs = torch.stack([-g[:-1], c], dim=1).cpu().numpy()
    # `solveh_banded` (LAPACK via OpenBLAS) is called once per accepted LM step on a tiny system
    # (bandwidth 12, n up to ~6*300+1) - single-threaded LAPACK is not just adequate but *faster* than
    # multi-threaded here, since OpenBLAS's default thread count (this build's MAX_THREADS=64, or the
    # host's core count) massively oversubscribes a problem this small. On a host already running other
    # CPU/BLAS-heavy processes (this is a shared host) that oversubscription compounds into
    # severe thread scheduling contention: an uncapped call was observed to make an unrelated fast
    # (CPU-only, <1s in isolation) test take 10+ minutes wall-clock (an earlier workaround set an import-time
    # `OPENBLAS_NUM_THREADS=4` env var). The fix lives here - `threadpoolctl.threadpool_limits(1)`, scoped to just
    # this call - so importing `fastdo` no longer silently mutates the process-wide BLAS thread count for
    # every importer; only this call site is capped (to 1, not 4 - re-verified clean on the
    # whole suite), and only for its own duration.
    with threadpoolctl.threadpool_limits(limits=1):
        sol = torch.as_tensor(np.ascontiguousarray(solveh_banded(ab, rhs, lower=False)), dtype=dtype,
                              device=device)
    x1, x2 = sol[:, 0], sol[:, 1]
    delta_sigma = (-g[-1] - c @ x1) / (d - c @ x2)
    delta_pose = x1 - delta_sigma * x2
    return torch.cat([delta_pose, delta_sigma.reshape(1)])


def lm_step(H: Tensor, g: Tensor, lam: float, solver: str = "dense") -> Tensor:
    """Solve `(H + lam*diag(H) + 1e-9 I) delta = -g`.

    `solver="banded"` assumes H has the arrow/block-tridiagonal structure `assemble_normal_equations`
    produces (bandwidth `_BAND_U` in the 6T pose coordinates, plus a dense arrow row/column for the
    shared log-scale) and solves it in O(n) via `_lm_step_banded` instead of an O(n^3) dense
    factorization; `lm_optimize` uses it for exactly this reason (the dense solve dominated
    wall-clock at T=100). It defaults to `"dense"` here (unchanged from the original) because
    `lm_step` is also exercised directly on arbitrary dense systems without that structure (see
    `test_lm_step_reduces_quadratic` in test_lm.py); a banded solve on a non-block-tridiagonal H would
    silently drop out-of-band entries and be wrong, so the assumption is only ever applied where the
    caller (`lm_optimize`) knows it holds. Falls back to the dense path for `H` too small to hold the
    assumed bandwidth. A system with the constant-velocity prior (`BlockTridiagonal.U2`, e.g.
    `LMResult.H` from `solve_trajectory`'s defaults) has bandwidth 17, outside this path's assumption;
    `_lm_step_banded` rejects it with `ValueError` rather than dropping the (k, k+2) blocks. Solve such
    systems with `banded.solve_block_tridiagonal` (or `solver="dense"`).
    """
    if solver == "banded" and H.shape[0] > _BAND_U + 2:
        return _lm_step_banded(H, g, lam)
    # lam * diag(H) only damps parameters whose Hessian diagonal is nonzero; a direction with an
    # exactly zero diagonal entry (e.g. a fully unobserved parameter) gets no stabilisation from
    # this term and relies solely on the fixed 1e-9 * I floor below to keep the solve well-posed.
    A = H + lam * torch.diag(torch.diagonal(H)) + 1e-9 * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    return torch.linalg.solve(A, -g)


def apply_update(poses: Tensor, log_scale: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
    T = poses.shape[0]
    xi = delta[: 6 * T].reshape(T, 6).to(poses.dtype)
    new_poses = apply_left(poses, xi)
    return new_poses, log_scale + delta[-1].to(log_scale.dtype)


def _fix_scale(sys: BlockTridiagonal) -> BlockTridiagonal:
    """`cfg.fix_scale` semantics (preserved exactly from the dense path): zero the scale
    row/column and set the diagonal entry to 1, so `solve_block_tridiagonal`'s Schur complement always
    solves `delta_sigma = 0` and `marginal_covariances`'s rank-one scale correction vanishes (its
    denominator becomes `1 - 0 = 1` and its numerator `x = A^{-1} @ 0 = 0`), exactly like the dense
    path's `H[-1,-1] = 1`, `H[-1,:] = H[:,-1] = 0`, `g[-1] = 0`."""
    if sys.border > 1:                                               # identity corner, zero border
        eye = torch.eye(sys.border, dtype=sys.h_ss.dtype, device=sys.h_ss.device)
        return replace(sys, c=torch.zeros_like(sys.c), h_ss=eye.expand_as(sys.h_ss).clone(),
                       g_s=torch.zeros_like(sys.g_s))
    return replace(sys, c=torch.zeros_like(sys.c), h_ss=torch.ones_like(sys.h_ss),
                   g_s=torch.zeros_like(sys.g_s))


def lm_optimize_batch(poses0: Tensor, log_scale0: Tensor, evaluate: BatchEvaluate, cfg: LMConfig,
                      layout: ClipLayout) -> BatchLMResult:
    """Levenberg-Marquardt over B concatenated clips with per-clip damping, acceptance and convergence.

    Per clip b this is exactly `lm_optimize`'s rule: a trial with lower cost is accepted (`lam_b /=
    lam_down`, correspondences refreshed, stop if the relative decrease is below `rel_tol`), otherwise
    rejected (`lam_b *= lam_up`, stop once `lam_b > 1e6`). A stopped clip's steps are masked out, so its
    poses stay bit-identical while other clips iterate. The per-clip flags are read back once per
    iteration (the loop's only host synchronization). The loop keeps every term's per-frame normal-equation
    blocks (`banded.term_blocks`), so a trial evaluates only the active clips' frames and the refresh (run
    only if some clip accepted) only the accepted clips' frames, whose block rows it overwrites: a stopped
    or rejected clip costs no evaluation.
    """
    ci = layout.clip_index
    B, T = layout.n_clips, poses0.shape[0]
    poses, log_scale = poses0.clone(), log_scale0.clone()
    # A (B,3) log-stretch state takes the per-clip masks over its trailing axis; a (B,) one uses them as is.
    per_state = (lambda m: m) if log_scale.dim() == 1 else (lambda m: m[:, None])
    lam = torch.full((B,), cfg.lam0, dtype=torch.float64, device=poses.device)
    active = torch.ones(B, dtype=torch.bool, device=poses.device)
    terms, smooth = evaluate(poses, log_scale, True)
    blocks = [term_blocks(term) for term in terms]
    cost = clip_costs(terms, smooth, layout)
    del terms
    active_host = [True] * B
    iterations = [0] * B
    history: list[list[bool]] = [[] for _ in range(B)]
    subsets: dict[tuple[int, ...], Tensor | None] = {}               # one index tensor per clip subset

    def frames_of(flags: list[bool]) -> Tensor | None:
        key = tuple(b for b in range(B) if flags[b])
        if key not in subsets:
            subsets[key] = layout.frames_of(key)
        return subsets[key]

    it = 0
    for it in range(1, cfg.max_iter + 1):
        sys = _assemble(blocks, smooth, T, layout, cfg.fix_scale)
        delta_pose, delta_sigma = solve_block_tridiagonal(sys, lam)
        trial_poses = torch.where(active[ci][:, None, None], apply_left(poses, delta_pose.to(poses.dtype)), poses)
        trial_ls = torch.where(per_state(active), log_scale + delta_sigma.to(log_scale.dtype), log_scale)
        trial_frames = frames_of(active_host)
        t_terms, t_smooth = evaluate(trial_poses, trial_ls, False, trial_frames)
        trial_cost = clip_costs(t_terms, t_smooth, layout, trial_frames)
        del t_terms
        accept = active & (trial_cost < cost)
        if cfg.min_median_depth is not None:
            depth = torch.stack([z.median() for z in torch.split(trial_poses[:, 2, 3], list(layout.lengths))])
            accept = accept & (depth > cfg.min_median_depth)
        rel = (cost - trial_cost) / cost.clamp_min(1e-12)
        lam = torch.where(accept, (lam / cfg.lam_down).clamp_min(1e-9), torch.where(active, lam * cfg.lam_up, lam))
        next_active = active & ~torch.where(accept, rel < cfg.rel_tol, lam > 1e6)
        poses = torch.where(accept[ci][:, None, None], trial_poses, poses)
        log_scale = torch.where(per_state(accept), trial_ls, log_scale)
        accept_host, next_host = cast("list[list[bool]]", torch.stack([accept, next_active]).cpu().tolist())
        for b in range(B):
            if active_host[b]:
                history[b].append(accept_host[b])
                if not next_host[b]:
                    iterations[b] = it
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("iter %d accepted %s active %s", it, accept_host, next_host)
        if any(accept_host):
            frames = frames_of(accept_host)
            terms, smooth = evaluate(poses, log_scale, True, frames)
            new_blocks = [term_blocks(term) for term in terms]
            blocks = new_blocks if frames is None else [
                (H.index_copy(0, frames, H_new), g.index_copy(0, frames, g_new))
                for (H, g), (H_new, g_new) in zip(blocks, new_blocks, strict=True)]
            cost = torch.where(accept, clip_costs(terms, smooth, layout, frames), cost)
            del terms, new_blocks
        active, active_host = next_active, next_host
        if not any(active_host):
            break
    for b in range(B):
        if active_host[b]:
            iterations[b] = it
    # The system is assembled from `blocks`/`smooth`, which are always in sync with `poses`/`log_scale`
    # (both change together on an accepted step), never carried over from the loop.
    sys = _assemble(blocks, smooth, T, layout, cfg.fix_scale)
    return BatchLMResult(poses=poses, log_scale=log_scale, cost=cost, iterations=tuple(iterations),
                         accepted=tuple(tuple(h) for h in history), loop_iterations=it, system=sys)


def _assemble(blocks: list[tuple[Tensor, Tensor]], smooth: PriorsLike, n_frames: int, layout: ClipLayout,
              fix_scale: bool) -> BlockTridiagonal:
    sys = assemble_from_blocks(blocks, smooth, n_frames, layout)
    return _fix_scale(sys) if fix_scale else sys


def lm_optimize(poses0: Tensor, log_scale0: Tensor, evaluate: Evaluate, cfg: LMConfig) -> LMResult:
    """Single-clip LM: the B = 1 case of `lm_optimize_batch`, with a scalar log-scale and a
    `refresh: bool` evaluate callback (a single clip is always evaluated whole, so `frames` is None).
    `LMResult.system` is a layout-free single-clip system."""
    layout = ClipLayout.build((poses0.shape[0],), poses0.device)

    def batch_evaluate(poses: Tensor, log_scales: Tensor, refresh: bool, frames: Tensor | None = None
                       ) -> tuple[list[TermOutput], PriorsLike]:
        if frames is not None:
            raise ValueError("lm_optimize evaluates its single clip whole; a frame subset was requested")
        return evaluate(poses, log_scales[0], refresh)

    res = lm_optimize_batch(poses0, log_scale0.reshape(1, *log_scale0.shape), batch_evaluate, cfg, layout)  # or (3,)
    sys = replace(res.system, h_ss=res.system.h_ss[0], g_s=res.system.g_s[0], layout=None)
    return LMResult(poses=res.poses, log_scale=res.log_scale[0], cost=float(res.cost[0]),
                    iterations=res.iterations[0], system=sys)
