"""Per-frame tracker feature tokens from the solver's own linearization.

Pure transforms of quantities the solver's evaluate callback already computes: the per-term `TermOutput`s, the OT
plan, the poses, the log-scale and the template samples. Everything here is device-agnostic, float64, never reads
back to the host and has no data-dependent Python branch (validation reads shapes and Python arguments only), so it
is safe inside a device-side loop. Nothing is sanitized: every reduction is per frame, so a non-finite input in
frame k stays in row k.

Callers pass data terms only: the pointmap term `pm` and the RAW `track_term` output `tr_raw` (Cauchy weight, no
`lam_track / scale2`). A prior term (e.g. design 11.10's ray-scale prior) never enters a token.

Recovering the solver weighting. Per frame, with `n_pm`/`n_tr` the positive-weight counts (clamped to 1),
`w_t = lam_track / scale2`, `Mk` from `frame_normal_blocks` and `g_solver`/`H_solver` the `term_blocks` sums over
`[pm, TermOutput(tr_raw.residual, tr_raw.jacobian, w_t * tr_raw.weight)]`:
`Mk^T g_solver = L^2 n_pm g_pm + w_t n_tr g_tr` and `diag(Mk^T H_solver Mk) = L^2 n_pm diagH_pm + w_t n_tr diagH_tr`.
Off-diagonal entries of H are not in the token; call `frame_normal_blocks` for them.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from fastdo.geometry.lie import se3_adjoint
from fastdo.solver.banded import term_blocks
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import TermOutput
from fastdo.tracker.clusters import mesh_cluster_mass

__all__ = ["FRAME_FEATURE_DIM", "TRACK_TOKEN_DIM", "build_frame_tokens", "frame_feature_dim",
           "frame_gradient_curvature", "frame_normal_blocks", "plan_row_mass", "pointmap_residual_histogram",
           "track_tokens"]

logger = logging.getLogger(__name__)

F64 = torch.float64
TRACK_TOKEN_DIM: int = 6
_MASS_FLOOR = 1e-12          # the barycenter clamp (ot.plan_barycenters): below it pbar is not a barycenter
_LOG10_LO, _LOG10_HI = -3.0, 0.0   # histogram range: matched distance / length scale in 10^[-3, 0]


def frame_feature_dim(n_params: int = 7, n_bins: int = 16, n_clusters: int = 64) -> int:
    """4 * n_params + n_bins + n_clusters + 2."""
    return 4 * n_params + n_bins + n_clusters + 2


FRAME_FEATURE_DIM: int = frame_feature_dim()


def _per_frame(value: Tensor | float, n_frames: int, device: torch.device, name: str) -> Tensor:
    """A float, 0-dim or (T,) value -> (T,) float64 on `device` (no host read)."""
    if isinstance(value, Tensor):
        if value.dim() > 1 or (value.dim() == 1 and value.shape[0] != n_frames):
            raise ValueError(f"{name} must be a float, 0-dim or ({n_frames},), got shape {tuple(value.shape)}")
        return value.to(device=device, dtype=F64).expand(n_frames)
    return torch.full((n_frames,), float(value), dtype=F64, device=device)


def _check_terms(terms: list[TermOutput]) -> tuple[int, int]:
    """(T, P) shared by every term, from shapes only."""
    if not terms:
        raise ValueError("need at least one term")
    T, P = terms[0].jacobian.shape[0], terms[0].jacobian.shape[-1]
    for t in terms:
        if t.jacobian.dim() != 4 or t.jacobian.shape[0] != T or t.jacobian.shape[-1] != P:
            raise ValueError(f"terms disagree on (T, P): {tuple(t.jacobian.shape)} vs T={T}, P={P}")
        if t.residual.shape[0] != T or t.weight.shape[0] != T:
            raise ValueError("a term's residual/weight frame count differs from its jacobian's")
    return T, P


def _canonical_map(poses: Tensor, length: Tensor, n_params: int) -> Tensor:
    """Mk = blockdiag(Ad_T, I) @ diag(L, L, L, 1, ...): (T,P,P) float64."""
    T = length.shape[0]
    if poses.shape != (T, 4, 4):
        raise ValueError(f"poses must have shape ({T},4,4), got {tuple(poses.shape)}")
    Mk = torch.zeros(T, n_params, n_params, dtype=F64, device=length.device)
    Mk[:, :6, :6] = se3_adjoint(poses.to(device=length.device, dtype=F64))
    Mk.diagonal(dim1=-2, dim2=-1)[:, 6:] = 1.0
    Mk[:, :, :3] = Mk[:, :, :3] * length[:, None, None]
    return Mk


def frame_normal_blocks(terms: list[TermOutput], poses: Tensor | None = None,
                        length_scale: Tensor | float | None = None, *,
                        metric_residual: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """(H (T,P,P), g (T,P), n_obs (T,)), float64; solver frame, or canonical when poses and length_scale are given.

    `H`/`g` sum `term_blocks` over `terms`; `n_obs` counts positive-weight rows (a count, not a weight sum: weights
    carry term-specific units). Solver frame: the left-perturbation columns `[rho, theta, sigma, ...]` of
    `TermOutput.jacobian`. Canonical: with `Mk = blockdiag(Ad_T, I) diag(L, L, L, 1, ...)` (the body-frame
    perturbation `T Exp(xi)`, translation in units of L), `H_c = Mk^T H Mk` and `g_c = Mk^T g`, both divided by
    `L^2` when `metric_residual` (metric residuals become unit-free); pixel residuals (the raw track term) pass
    `metric_residual=False`. Canonical blocks are invariant to a similarity scaling of the scene; metric-residual
    blocks also to camera placement.
    """
    T, P = _check_terms(terms)
    if (poses is None) != (length_scale is None):
        raise ValueError("canonical mode needs both poses and length_scale (or neither for the solver frame)")
    H, g = term_blocks(terms[0])
    n_obs = (terms[0].weight > 0).flatten(1).sum(1).to(F64)
    for t in terms[1:]:
        Ht, gt = term_blocks(t)
        H, g = H + Ht, g + gt
        n_obs = n_obs + (t.weight > 0).flatten(1).sum(1).to(F64)
    if poses is None or length_scale is None:
        if not metric_residual:
            raise ValueError("metric_residual=False only has meaning in canonical mode")
        return H, g, n_obs
    if P < 6:
        raise ValueError(f"canonical mode needs at least the 6 pose columns, got P={P}")
    L = _per_frame(length_scale, T, H.device, "length_scale")
    Mk = _canonical_map(poses, L, P)
    MkT = Mk.transpose(1, 2)
    H_c = MkT @ H @ Mk
    g_c = (MkT @ g.unsqueeze(-1)).squeeze(-1)
    if metric_residual:
        L2 = L * L
        H_c = H_c / L2[:, None, None]
        g_c = g_c / L2[:, None]
    return H_c, g_c, n_obs


def frame_gradient_curvature(terms: list[TermOutput], poses: Tensor | None = None,
                             length_scale: Tensor | float | None = None, *,
                             metric_residual: bool = True) -> tuple[Tensor, Tensor]:
    """(g (T,P), diag(H) (T,P)) per observation, float64 (`frame_normal_blocks` divided by n_obs clamped to 1)."""
    H, g, n_obs = frame_normal_blocks(terms, poses, length_scale, metric_residual=metric_residual)
    n = n_obs.clamp_min(1.0)[:, None]
    return g / n, torch.diagonal(H, dim1=-2, dim2=-1) / n


def plan_row_mass(plan: Tensor | SparsePlan) -> Tensor:
    """(T,N) row sums of a dense or kNN plan."""
    if isinstance(plan, SparsePlan):
        return plan.values.to(F64).sum(-1)
    return plan.to(F64).sum(-1)


def pointmap_residual_histogram(term: TermOutput, row_mass: Tensor, length_scale: Tensor | float,
                                n_bins: int = 16) -> Tensor:
    """(T,n_bins) float64 counts of valid rows' matched distance / length_scale on 10^[-3,0] log bins.

    The matched distance reads the pointmap residual convention back: `|residual| / sqrt(N * row_mass)` on rows
    with mass above the barycenter clamp, `+inf` on valid but unmatched rows. Bin `j` counts the rows with exactly
    `j` of the `n_bins - 1` log-spaced edges strictly below `x`, so `x <= 1e-3` is bin 0 and `x > 1` (also inf and
    NaN) the last bin. Invalid rows are not counted: each row of the output sums to the frame's valid count.
    """
    if term.residual.dim() != 3 or term.residual.shape[-1] != 3:
        raise ValueError(f"need a point-to-point (T,N,3) residual, got {tuple(term.residual.shape)}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if row_mass.shape != term.weight.shape:
        raise ValueError(f"row_mass {tuple(row_mass.shape)} must match the weight {tuple(term.weight.shape)}")
    T, N = term.weight.shape
    dev = term.residual.device
    valid = term.weight > 0
    mass = row_mass.to(device=dev, dtype=F64)
    matched = valid & (mass > _MASS_FLOOR)
    dist = torch.linalg.vector_norm(term.residual.to(F64), dim=-1) / (N * mass).clamp_min(_MASS_FLOOR).sqrt()
    dist = torch.where(matched, dist, torch.full_like(dist, float("inf")))
    x = dist / _per_frame(length_scale, T, dev, "length_scale")[:, None]
    edges = torch.logspace(_LOG10_LO, _LOG10_HI, n_bins - 1, dtype=F64, device=dev)
    bins = torch.bucketize(x, edges)
    return torch.zeros(T, n_bins, dtype=F64, device=dev).scatter_add_(1, bins, valid.to(F64))


def track_tokens(anchors: Tensor, term: TermOutput, template_diameter: Tensor | float) -> Tensor:
    """(T,J,6) float64: [anchor/diameter (3), residual norm px (1), weight (1), visible (1)].

    `anchors` are canonical template coordinates, (J,3) or (T,J,3). `term` is the raw `track_term` output; the
    residual norm is exactly 0 where the weight is not positive.
    """
    if term.residual.dim() != 3 or term.residual.shape[-1] != 2:
        raise ValueError(f"need a (T,J,2) pixel residual, got {tuple(term.residual.shape)}")
    T, J = term.weight.shape
    if anchors.dim() not in (2, 3) or anchors.shape[-2:] != (J, 3) or (anchors.dim() == 3 and anchors.shape[0] != T):
        raise ValueError(f"anchors must be ({J},3) or ({T},{J},3), got {tuple(anchors.shape)}")
    dev = term.residual.device
    D = _per_frame(template_diameter, T, dev, "template_diameter")
    canon = (anchors.to(device=dev, dtype=F64) / D[:, None, None]).expand(T, J, 3)
    visible = term.weight > 0
    norm = torch.linalg.vector_norm(term.residual.to(F64), dim=-1)
    norm = torch.where(visible, norm, torch.zeros_like(norm))
    return torch.cat([canon, norm[..., None], term.weight.to(F64)[..., None], visible.to(F64)[..., None]], dim=-1)


def build_frame_tokens(pm: TermOutput, tr_raw: TermOutput, plan: Tensor | SparsePlan, cluster_id: Tensor,
                       template_diameter: Tensor | float, poses: Tensor, log_scale: Tensor,
                       n_bins: int = 16, n_clusters: int = 64) -> Tensor:
    """(T, frame_feature_dim(P, n_bins, n_clusters)) float64; tr_raw is the raw track_term output.

    With `L = exp(log_scale) * template_diameter` and `P = pm.jacobian.shape[-1]`, the columns are
    `g_pm [0,P) | diagH_pm [P,2P)` (canonical pointmap block per valid row), `g_tr [2P,3P) | diagH_tr [3P,4P)`
    (canonical raw track block, `metric_residual=False`, per visible track), the matched-distance histogram
    normalized by the valid count, the raw cluster column mass, the pointmap valid fraction and the visible-track
    fraction. Every column is invariant under a similarity scaling of the scene with the same intrinsics.
    """
    T, _P = _check_terms([pm, tr_raw])
    if tr_raw.residual.shape[-1] != 2:
        raise ValueError(f"tr_raw must be the (T,J,2) pixel track term, got {tuple(tr_raw.residual.shape)}")
    plan_shape = plan.values.shape if isinstance(plan, SparsePlan) else plan.shape
    if len(plan_shape) != 3 or plan_shape[:2] != pm.weight.shape:
        raise ValueError(f"plan {tuple(plan_shape)} and pm weight {tuple(pm.weight.shape)} disagree on (T,N)")
    dev = pm.jacobian.device
    if log_scale.dim() > 1 or (log_scale.dim() == 1 and log_scale.shape[0] != T):
        raise ValueError(f"log_scale must be 0-dim or ({T},), got {tuple(log_scale.shape)}")
    L = log_scale.to(device=dev, dtype=F64).exp() * _per_frame(template_diameter, T, dev, "template_diameter")
    g_pm, h_pm = frame_gradient_curvature([pm], poses, L)
    g_tr, h_tr = frame_gradient_curvature([tr_raw], poses, L, metric_residual=False)
    N, J = pm.weight.shape[1], tr_raw.weight.shape[1]
    n_pm = (pm.weight > 0).sum(1).to(F64)
    hist = pointmap_residual_histogram(pm, plan_row_mass(plan), L, n_bins) / n_pm.clamp_min(1.0)[:, None]
    mass = mesh_cluster_mass(plan, cluster_id, n_clusters)
    valid_frac = n_pm / max(N, 1)
    track_frac = (tr_raw.weight > 0).sum(1).to(F64) / max(J, 1)
    return torch.cat([g_pm, h_pm, g_tr, h_tr, hist, mass, valid_frac[:, None], track_frac[:, None]], dim=1)
