"""Canonical moments of a transport plan: its sufficient statistics under the Sim(3) action.

Between two Sinkhorn refreshes the plan P (T,N,M) is held fixed while LM evaluates the pointmap term at trial poses
(R, t, s). The term reads P only through the transported point of each observation,

    pbar_i = sum_j P_ij (s R y_j + t) / w_i,    w_i = sum_j P_ij,

and because the Sim(3) action is affine this factorises as pbar_i = s R B_i + t with B_i = sum_j P_ij y_j / w_i, the
barycenter in CANONICAL (untransformed) template coordinates. The point-to-plane normal is a direction field, so it
picks up R alone: nbar_i = R Nbar_i. `(B, w, Nbar)` are T*N*(3+1+3) numbers where P is T*N*M (a kNN plan T*N*k*2), and
they give the residual and its Jacobian at every pose between refreshes. Only the refresh itself needs C and P.

Zero-mass convention, mirrored exactly from `ot.plan_barycenters` / `ot_sparse.plan_barycenters_sparse`: both divide by
`max(w, 1e-12)`, so the old transported point is `(P @ (s R Y + t)) / max(w, 1e-12) = s R B + (w / max(w, 1e-12)) t`.
The translation's factor is exactly 1.0 on every row with w >= 1e-12 (w / w in IEEE arithmetic) and exactly 0.0 on a
zero-mass row, where pbar is 0 as before; `transported_points` applies that factor rather than a bare `+ t`. (A
zero-mass row has zero residual and Jacobian either way, `scale_w = sqrt(N w) = 0`, but a row with 0 < w < 1e-12 does
not, so the factor is kept rather than argued away.) The canonical normal is reduced to the unit vector the old
`pointmap_term` built from the same plan, dense or sparse, with the same operations, so rotating it is the only change.

Numerics: algebraically identical to evaluating the plan at each pose; the sums are reassociated (P @ Y before the
transform instead of after), so terms differ from the plan path at float round-off only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fastdo.solver.ot import plan_barycenters
from fastdo.solver.ot_sparse import SparsePlan, plan_barycenters_sparse

if TYPE_CHECKING:
    from fastdo.solver.aniso import Stretch

__all__ = ["MASS_FLOOR", "PlanMoments", "plan_moments", "take_moments", "transported_points"]

MASS_FLOOR = 1e-12   # the clamp `plan_barycenters` / `plan_barycenters_sparse` divide the row mass by


@dataclass(frozen=True)
class PlanMoments:
    """The canonical moments of a plan: `bary` (T,N,3) = (P @ Y) / max(w, 1e-12) in canonical coordinates,
    `mass` (T,N) = the row mass w = P.sum(-1), and, only when reduced for the point-to-plane residual, `normal` (T,N,3),
    the unit canonical barycentric normal (None otherwise)."""

    bary: Tensor
    mass: Tensor
    normal: Tensor | None = None


def _per_frame(field: Tensor, T: int) -> Tensor:
    return field if field.dim() == 3 else field.unsqueeze(0).expand(T, -1, -1)


def plan_moments(plan: Tensor | SparsePlan, Y: Tensor, normals: Tensor | None = None) -> PlanMoments:
    """Reduce a dense (T,N,M) or kNN `SparsePlan` to its canonical moments against the canonical template `Y` (M,3) or
    per-frame (T,M,3). With `normals` ((M,3) or (T,M,3), unit canonical normals paired with `Y`) the unit barycentric
    normal is kept too, built exactly as `terms.pointmap_term`'s point-to-plane branch builds it from the plan."""
    if isinstance(plan, SparsePlan):
        T = plan.values.shape[0]
        bary, w = plan_barycenters_sparse(plan.values, plan.indices, _per_frame(Y, T))
        normal = None
        if normals is not None:
            # `plan_barycenters_sparse` already divides by the clamped mass and the old branch divides once more before
            # normalising; a positive per-row factor, kept so the unit vector is bit-identical to the plan path's.
            raw, _ = plan_barycenters_sparse(plan.values, plan.indices, _per_frame(normals, T))
            normal = torch.nn.functional.normalize(raw / w.clamp_min(MASS_FLOOR)[..., None], dim=-1)
        return PlanMoments(bary=bary, mass=w, normal=normal)
    bary, w = plan_barycenters(plan, Y)
    normal = None
    if normals is not None:
        normal = torch.nn.functional.normalize((plan @ normals) / w.clamp_min(MASS_FLOOR)[..., None], dim=-1)
    return PlanMoments(bary=bary, mass=w, normal=normal)


def transported_points(m: PlanMoments, poses: Tensor, scale: Tensor | float, stretch: Stretch | None = None) -> Tensor:
    """The transported point pbar (T,N,3) of every observation at `poses` (T,4,4) and `scale` (scalar or per-frame
    (T,)): s R B + (w / max(w, 1e-12)) t, the value `plan_barycenters` gives on the transformed template (module
    docstring). Same R, t, s convention as `terms.transform_points`.

    Anisotropic `stretch` (`fastdo.solver.aniso`): the template is y -> A (y - c) before the similarity, and the moments
    stay sufficient: P @ (R A (Y - c) + t) / max(w, 1e-12) = R A (B - keep c) + keep t with keep = w / max(w, 1e-12), so
    pbar = s R A (B - keep c) + keep t (the solver passes s = 1: the stretch carries the scale)."""
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    if isinstance(scale, Tensor) and scale.dim() == 1:
        scale = scale[:, None, None]
    keep_t = m.mass / m.mass.clamp_min(MASS_FLOOR)                       # 1.0 exactly unless w < 1e-12
    if stretch is None:
        return scale * torch.einsum("tij,tnj->tni", R, m.bary) + keep_t[..., None] * t[:, None, :]
    local = torch.einsum("tij,tnj->tni", stretch.matrix(), m.bary - keep_t[..., None] * stretch.centroid[:, None, :])
    return scale * torch.einsum("tij,tnj->tni", R, local) + keep_t[..., None] * t[:, None, :]


def take_moments(m: PlanMoments, rows: Tensor | slice) -> PlanMoments:
    """The moments of frames `rows` (a slice view, or an `index_select` of a device index)."""
    def pick(x: Tensor) -> Tensor:
        return x[rows] if isinstance(rows, slice) else x.index_select(0, rows)
    return PlanMoments(bary=pick(m.bary), mass=pick(m.mass), normal=None if m.normal is None else pick(m.normal))
