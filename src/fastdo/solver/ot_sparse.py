"""kNN-truncated log-domain unbalanced Sinkhorn.

The fine-level pointmap term's dense Sinkhorn touches every (observed point, mesh sample) pair each
iteration - O(T N M). Design 11.5 shows the fine-level plan is local (measured mean dropped row mass
2.4e-4 at k=32): restricting each observed point's kernel to its k nearest VISIBLE mesh samples turns
this into a well-posed unbalanced OT problem in its own right (cost +inf outside the candidate set, so
the fixed-plan majorization-minimization argument applies unchanged - truncation is a change of
objective, not an approximation to descend), at O(T N k) per iteration.

`SparsePlan` is the truncated plan `pointmap_term` (`terms.py`) hands back on its Sinkhorn-recompute
path when `knn` is set, and accepts back on its cached-plan (`refresh=False`) path unchanged - see that
module's docstring. Dual potentials stay dense-shaped (`f` (T,N), `g` (T,M)) throughout: `g` is always
indexed by the GLOBAL mesh-sample id, never by a candidate "slot", so warm-starting across a refresh
whose candidate set changed (visibility/pose moved) needs no explicit gather/rebase step - a sample
that re-enters candidacy simply reads whatever `g` last held there (0 if it never received a value; see
`test_ot_sparse.py::test_warm_start_across_candidate_change_matches_cold_solve`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fastdo.solver.ot import Param, _per_batch, _safe_log_mass, _tau

__all__ = ["SparsePlan", "knn_tail_mass", "plan_barycenters_sparse", "sinkhorn_unbalanced_sparse"]


@dataclass(frozen=True)
class SparsePlan:
    """A kNN-truncated OT plan: `values[t,n,:]` is the plan mass observed point `n` (frame `t`) puts on
    the `k` candidate mesh samples named by `indices[t,n,:]` - GLOBAL column ids in `[0, M)`, not slot
    positions. Columns absent from a row's candidate set carry (implicitly) zero mass."""

    indices: Tensor   # (T, N, k) int64
    values: Tensor    # (T, N, k)


def _segment_plan(idx: Tensor, num_segments: int) -> tuple[Tensor, Tensor]:
    """`(order, lengths)` for `_segment_sum`: each row's entries sorted by segment id (stable) and the
    flattened per-(row, segment) entry counts. Computed once per Sinkhorn call, since `idx` is fixed across
    its iterations. The counts are integer atomics, which are exact in any order."""
    order = torch.argsort(idx, dim=1, stable=True)
    counts = torch.zeros(idx.shape[0], num_segments, dtype=torch.long, device=idx.device)
    counts.scatter_add_(1, idx, torch.ones_like(idx))
    return order, counts.flatten()


def _segment_sum(values: Tensor, plan: tuple[Tensor, Tensor], num_segments: int) -> Tensor:
    """Per-(row, segment) sum of `values` (B, L) in a FIXED order: a `segment_reduce` over the sorted entries.

    `scatter_add_` sums floats with atomics, in a different order on every run; through the LM's step
    decisions that moved one Ego4D pose ~1 cm on bit-identical inputs. The deterministic `scatter_add_`
    re-sorts on every call (17x slower here); sorting once per Sinkhorn call costs ~3.4x one atomic scatter."""
    order, lengths = plan
    ordered = torch.gather(values, 1, order).flatten()
    return torch.segment_reduce(ordered, "sum", lengths=lengths, unsafe=True).reshape(values.shape[0], num_segments)


def _segment_logsumexp(values: Tensor, idx: Tensor, num_segments: int,
                       plan: tuple[Tensor, Tensor] | None = None) -> Tensor:
    """logsumexp of `values` (B, L) grouped by integer segment id `idx` (B, L) in `[0, num_segments)`,
    per batch row, -> (B, num_segments). Segments touched by no entry return -inf (never NaN): a
    per-segment max via `scatter_reduce_(..., "amax")` (numerically stable shift), `exp`, `_segment_sum`,
    `log`, add the max back - the batched, GPU-friendly "segment log-sum-exp" for
    the sparse Sinkhorn's column (mesh-side) potential update.
    """
    B = values.shape[0]
    seg_max = torch.full((B, num_segments), float("-inf"), dtype=values.dtype, device=values.device)
    seg_max.scatter_reduce_(1, idx, values, reduce="amax", include_self=True)
    shifted = torch.exp(values - torch.gather(seg_max, 1, idx))
    seg_sum = _segment_sum(shifted, plan if plan is not None else _segment_plan(idx, num_segments), num_segments)
    has_entries = torch.isfinite(seg_max)
    return torch.where(has_entries, seg_max + torch.log(seg_sum.clamp_min(1e-300)), seg_max)


def sinkhorn_unbalanced_sparse(C_k: Tensor, idx: Tensor, a: Tensor, b: Tensor, eps: Param, rho_a: Param,
                               rho_b: Param, n_iter: int,
                               potentials: tuple[Tensor, Tensor] | None = None
                               ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """kNN-truncated counterpart of `ot.sinkhorn_unbalanced_potentials`, restricted to each row's `k`
    candidates (cost +inf outside `idx`, see the module docstring for why this is exact, not
    approximate). `C_k`/`idx` (T,N,k): squared distances and GLOBAL column indices of each observed
    point's candidate set (`idx` should not repeat a column *within* one row). `a` (T,N)/`b` (T,M) are
    the same dense marginals `sinkhorn_unbalanced_potentials` takes; `b` may contain exact zeros (a
    masked-out mesh sample), handled the same NaN-safe way via `_safe_log_mass`. `eps`/`rho_a`/
    `rho_b` may be per-frame (T,) tensors, as in `ot.sinkhorn_unbalanced_potentials`.

    Returns `(values (T,N,k), (f (T,N), g (T,M)))`: `values` is the plan mass at each candidate (a
    `SparsePlan`'s `values` field once paired with `idx`); `f`/`g` are dense-shaped dual potentials, `g`
    GLOBAL-indexed (see module docstring on warm-starting across a candidate change). Columns touched by
    no candidate this call score `g_j = 0` exactly - the unbalanced problem's ordinary treatment of a
    target this call never tried to explain (the "far mesh points" case), not a special case.

    The row (observed-point) update is a plain logsumexp over each row's k (gathered, contiguous)
    candidates. The column (mesh-side) update is the segment log-sum-exp of `_segment_logsumexp` over
    the M mesh columns, since many rows can share a candidate.
    """
    T, N, k = C_k.shape
    M = b.shape[1]
    log_a = torch.log(a.clamp_min(1e-30))                      # (T,N)
    log_b_full = _safe_log_mass(b)                              # (T,M)
    idx_flat = idx.reshape(T, N * k)
    log_b = torch.gather(log_b_full, 1, idx_flat).reshape(T, N, k)
    if potentials is None:
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)
    else:
        f, g = potentials
    ta, tb = _tau(rho_a, eps), _tau(rho_b, eps)
    coef_a, coef_b = _per_batch(-ta * eps, C_k, 2), _per_batch(-tb * eps, C_k, 2)
    eps_c = _per_batch(eps, C_k, 3)
    plan = _segment_plan(idx_flat, M)
    for _ in range(n_iter):
        g_at = torch.gather(g, 1, idx_flat).reshape(T, N, k)
        f = coef_a * torch.logsumexp((g_at - C_k) / eps_c + log_b, dim=2)
        col_terms = ((f[:, :, None] - C_k) / eps_c + log_a[:, :, None]).reshape(T, N * k)
        raw = _segment_logsumexp(col_terms, idx_flat, M, plan)  # -inf where no candidate touched a column
        g = torch.where(torch.isfinite(raw), coef_b * raw, torch.zeros_like(raw))
    g_at = torch.gather(g, 1, idx_flat).reshape(T, N, k)
    values = torch.exp((f[:, :, None] + g_at - C_k) / eps_c + log_a[:, :, None] + log_b)
    return values, (f, g)


def plan_barycenters_sparse(values: Tensor, idx: Tensor, Yk: Tensor) -> tuple[Tensor, Tensor]:
    """Per-source barycenter of targets under a truncated plan: `values`/`idx` (T,N,k) (a `SparsePlan`'s
    fields), `Yk` (T,M,3) the per-frame target points (or any other (T,M,3)-shaped, GLOBAL-column-
    indexed field, e.g. canonical normals broadcast to T frames for the point-to-plane residual).
    Returns `(ybar (T,N,3), mass w (T,N))`, the sparse analogue of `ot.plan_barycenters`.
    """
    T, N, k = values.shape
    idx_exp = idx.reshape(T, N * k, 1).expand(-1, -1, Yk.shape[-1])
    Y_gathered = torch.gather(Yk, 1, idx_exp).reshape(T, N, k, Yk.shape[-1])
    w = values.sum(dim=2)
    ybar = (values.unsqueeze(-1) * Y_gathered).sum(dim=2) / w.clamp_min(1e-12).unsqueeze(-1)
    return ybar, w


def knn_tail_mass(P_dense: Tensor, idx: Tensor) -> Tensor:
    """Diagnostic: fraction of each row's dense-plan mass that lies
    OUTSIDE its `idx` (T,N,k) candidate set, `P_dense` (T,N,M) -> (T,N). A row with zero total mass
    (e.g. an invalid observed point) reports zero tail mass (nothing to drop)."""
    total = P_dense.sum(dim=2)
    in_mass = torch.gather(P_dense, 2, idx).sum(dim=2)
    return ((total - in_mass) / total.clamp_min(1e-12)).clamp(0.0, 1.0)
