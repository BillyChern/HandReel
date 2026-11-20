"""Differentiable weighted LM step and the tracker-to-solver glue.

`weighted_lm_step` is the exact Marquardt step `(H + lam Diag(H) + floor I) delta = -g` of the block solver
(`banded.solve_block_tridiagonal`) with the network's multiplicative weight overrides and per-clip damping, and it is
differentiable in both. The blocks are formed linearly in the effective weight `w = max(weight, 0) * override`
(`J^T (w J)`), never through `term_blocks`' `sqrt(w) J`, whose derivative is infinite at `w = 0` and gives NaN
gradients at every zero base weight or zero override. Autograd through the block-Thomas recursion is a fixed, finite
composition of smooth maps, so it equals the implicit-function derivative of `A delta = -g` up to round-off.

Preconditions (documented, not checked: a check would read tensors back to the host): overrides are `>= 0` (the tracker
gives `[1e-3, 1]` on valid slots and exactly 0 on masked ones) and damping is `>= 0`. The gradient is exactly 0 where
the base weight is 0. It is large where a floor-level weight (1e-3) carries a direction's only curvature; training
excludes clips whose data weights are all 0 and always assembles with a temporal prior.

Units of the glue: the weights override `pm.weight` and the solver-weighted track weight
(the product commutes with `lam_track / scale2`); `damping` is `lam` per clip; `xi_hat` is the body-frame
`[rho / L, theta]`; `dual_f` / `dual_g` are potentials divided by the frame's eps, `dual_g` per template cluster.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver.banded import (
    ClipLayout,
    PriorsLike,
    assemble_from_blocks,
    solve_block_tridiagonal,
)
from fastdo.solver.ot import _tau
from fastdo.solver.terms import TermOutput
from fastdo.tracker.heads import TrackerOutput
from fastdo.tracker.pack import clips_to_frames

__all__ = [
    "tracker_increment_poses",
    "tracker_sinkhorn_init",
    "tracker_weight_overrides",
    "weighted_lm_step",
    "weighted_term_blocks",
]

logger = logging.getLogger(__name__)

F64 = torch.float64


def weighted_term_blocks(term: TermOutput, override: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """`(H (T,7,7), g (T,7))` float64, linear in `w = term.weight.clamp_min(0) * override` (None -> override 1).

    `H = sum_k w_k J_k^T J_k` and `g = sum_k w_k J_k^T r_k` per frame, equal to `banded.term_blocks` of the same
    weights, with bounded derivatives in `w` everywhere (no `sqrt`). The override is neither clamped nor read back.
    """
    J, r, weight = term.jacobian, term.residual, term.weight
    if J.dim() != 4 or J.shape[-1] != 7:
        raise ValueError(f"jacobian must have shape (T,n,d,7), got {tuple(J.shape)}")
    if weight.dim() != 2 or tuple(J.shape[:2]) != tuple(weight.shape) or tuple(r.shape) != tuple(J.shape[:3]):
        raise ValueError(f"inconsistent term shapes: residual {tuple(r.shape)}, jacobian {tuple(J.shape)}, "
                         f"weight {tuple(weight.shape)}")
    w = weight.to(F64).clamp_min(0.0)
    if override is not None:
        if tuple(override.shape) != tuple(weight.shape):
            raise ValueError(f"override must have the weight's shape {tuple(weight.shape)}, "
                             f"got {tuple(override.shape)}")
        w = w * override.to(F64)
    J64 = J.to(F64)
    H = torch.einsum("tnda,tn,tndb->tab", J64, w, J64)
    g = torch.einsum("tnda,tn,tnd->ta", J64, w, r.to(F64))
    return H, g


def weighted_lm_step(terms: Sequence[TermOutput], smooth: PriorsLike, overrides: Sequence[Tensor | None],
                     damping: Tensor | float, n_frames: int, layout: ClipLayout | None = None,
                     floor: float = 1e-9) -> tuple[Tensor, Tensor]:
    """`(delta_pose (T,6), delta_sigma (B,) or 0-dim)`: `solve_block_tridiagonal(assemble_from_blocks(...))`.

    `overrides[i]` multiplies `terms[i].weight` (None = 1). `damping` is a float, or a Tensor `(layout.n_clips,)`
    with a layout and `()` / `(1,)` without one; a tensor keeps its autograd graph. Priors do not depend on the
    weights. An empty `terms` with a prior is allowed.
    """
    if len(overrides) != len(terms):
        raise ValueError(f"need one override per term: {len(terms)} terms, {len(overrides)} overrides")
    for i, term in enumerate(terms):
        if term.weight.shape[0] != n_frames:
            raise ValueError(f"term {i} has {term.weight.shape[0]} frames, expected n_frames={n_frames}")
    if layout is not None and layout.n_frames != n_frames:
        raise ValueError(f"layout has {layout.n_frames} frames, expected n_frames={n_frames}")
    if isinstance(damping, Tensor):
        allowed = [(layout.n_clips,)] if layout is not None else [(), (1,)]
        if tuple(damping.shape) not in allowed:
            raise ValueError(f"damping must have shape {' or '.join(map(str, allowed))}, "
                             f"got {tuple(damping.shape)}")
    blocks = [weighted_term_blocks(term, override) for term, override in zip(terms, overrides, strict=True)]
    return solve_block_tridiagonal(assemble_from_blocks(blocks, smooth, n_frames, layout), damping, floor)


def tracker_weight_overrides(output: TrackerOutput, layout: ClipLayout) -> tuple[Tensor, Tensor]:
    """`(point (T,N), track (T,J))` = `clips_to_frames` of the point and track weights (dtype kept; B1 casts)."""
    return clips_to_frames(output.point_weights, layout), clips_to_frames(output.track_weights, layout)


def _per_frame(value: Tensor | float, n_frames: int, like: Tensor, name: str) -> Tensor:
    """A float, 0-dim or `(T,)` value -> `(T,)` in `like`'s dtype and device (no host read)."""
    if isinstance(value, Tensor):
        if value.dim() > 1 or (value.dim() == 1 and value.shape[0] != n_frames):
            raise ValueError(f"{name} must be a float, 0-dim or ({n_frames},), got shape {tuple(value.shape)}")
        return value.to(device=like.device, dtype=like.dtype).expand(n_frames)
    return torch.full((n_frames,), float(value), dtype=like.dtype, device=like.device)


def tracker_sinkhorn_init(output: TrackerOutput, layout: ClipLayout, cluster_id: Tensor | None,
                          eps: Tensor | float, drive: Literal["g", "f"] = "g", C: Tensor | None = None,
                          a: Tensor | None = None, rho_b: Tensor | float | None = None) -> tuple[Tensor, Tensor]:
    """Sinkhorn warm start `(f (T,N), g (T,M))` in potential units from the tracker's dual heads.

    The Sinkhorn loop updates f first, so an `f` init is overwritten by the first half-step and is inert.
    - `drive="g"`: `(zeros, eps * dual_g gathered by cluster_id)`; `cluster_id` is `(M,)` or `(T,M)` (values not
      checked). Invisible or padded samples receive their cluster's value, which is harmless (log mass -1e30).
    - `drive="f"`: `f0 = eps * dual_f` and `g0 = -tau_b eps LSE_i((f0_i - C_ij)/eps + log a_i)`, one g half-step,
      so the loop's first f update sees the network's `dual_f`. Needs `C (T,N,M)`, `a (T,N)` and `rho_b` (float or
      `(T,)`, inf = balanced); `cluster_id` is unused.
    """
    fe = clips_to_frames(output.dual_f, layout)
    T = fe.shape[0]
    like = torch.zeros((), dtype=F64, device=fe.device)
    e = _per_frame(eps, T, like, "eps")
    if drive == "g":
        if cluster_id is None:
            raise ValueError("drive='g' needs cluster_id (M,) or (T,M)")
        gc = clips_to_frames(output.dual_g, layout)
        if cluster_id.dim() == 1:
            ids = cluster_id.expand(T, cluster_id.shape[0])
        elif cluster_id.dim() == 2 and cluster_id.shape[0] == T:
            ids = cluster_id
        else:
            raise ValueError(f"cluster_id must have shape (M,) or ({T},M), got {tuple(cluster_id.shape)}")
        g = e[:, None] * gc.gather(1, ids.to(device=gc.device, dtype=torch.long)).to(F64)
        return torch.zeros(fe.shape, dtype=F64, device=fe.device), g
    if drive != "f":
        raise ValueError(f"drive must be 'g' or 'f', got {drive!r}")
    if C is None or a is None or rho_b is None:
        raise ValueError("drive='f' needs C (T,N,M), a (T,N) and rho_b")
    if C.dim() != 3 or tuple(C.shape[:2]) != tuple(fe.shape) or tuple(a.shape) != tuple(fe.shape):
        raise ValueError(f"C must be ({T},{fe.shape[1]},M) and a ({T},{fe.shape[1]}), "
                         f"got {tuple(C.shape)} and {tuple(a.shape)}")
    f0 = e[:, None] * fe.to(F64)
    rho = rho_b if not isinstance(rho_b, Tensor) else _per_frame(rho_b, T, like, "rho_b")
    tau_b = _tau(rho, e)
    log_a = torch.log(a.to(F64).clamp_min(1e-30))
    lse = torch.logsumexp((f0[:, :, None] - C.to(F64)) / e[:, None, None] + log_a[:, :, None], dim=1)
    return f0, (-tau_b * e)[:, None] * lse


def tracker_increment_poses(poses: Tensor, log_scale: Tensor, output: TrackerOutput, layout: ClipLayout,
                            length_scale: Tensor | float) -> tuple[Tensor, Tensor]:
    """`(poses (T,4,4), log_scale (B,))`: `T_k Exp(diag(L_k,L_k,L_k,1,1,1) xi_hat_k)` and `log_scale + sigma_hat`.

    The body-frame increment equals the solver's left increment `Ad_T diag(L,L,L,1,1,1) xi_hat`. `length_scale` is
    a float, 0-dim or `(T,)` (the features' `_per_frame` rule). A zero `xi_hat` returns the poses bit-exactly.
    """
    xi = clips_to_frames(output.xi_hat, layout).to(poses.dtype)
    if xi.shape[0] != poses.shape[0]:
        raise ValueError(f"poses have {poses.shape[0]} frames, the layout {xi.shape[0]}")
    L = _per_frame(length_scale, xi.shape[0], poses, "length_scale")
    scaled = torch.cat([xi[:, :3] * L[:, None], xi[:, 3:]], dim=-1)
    return poses @ se3_exp(scaled), log_scale + output.sigma_hat.to(log_scale.dtype)
