"""Residual terms (pointmap OT, point tracks, Lie-group smoothness) with analytic Jacobians.

Parameter columns per frame: [rho(3), theta(3), sigma] = left se(3) perturbation + shared log-scale. With the
anisotropic `stretch` (`fastdo.solver.aniso`, `SolverConfig.anisotropic_scale`) the one log-scale column becomes the
clip's three log-stretches: [rho(3), theta(3), l(3)].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, overload

import torch
from torch import Tensor

from fastdo.geometry.lie import hat, se3_adjoint, se3_inv, se3_log
from fastdo.solver.aniso import stretch_jacobian
from fastdo.solver.moments import PlanMoments, plan_moments, transported_points
from fastdo.solver.ot import Param, plan_barycenters, sinkhorn_unbalanced_potentials
from fastdo.solver.ot_sparse import SparsePlan, plan_barycenters_sparse, sinkhorn_unbalanced_sparse

if TYPE_CHECKING:
    from fastdo.solver.aniso import Stretch

logger = logging.getLogger(__name__)

_NO_POTENTIALS = object()   # sentinel: distinguishes "potentials kwarg not passed" from "passed as None"
# Below this row sum, a frame's mesh_mask carries no usable mesh-side signal (e.g. every visible sample
# happens to foreshorten to ~0 weight); see pointmap_term's docstring for the fallback this triggers.
_ZERO_WEIGHT_ROW_TOL = 1e-12


@dataclass(frozen=True)
class TermOutput:
    residual: Tensor   # (T, n, d)
    jacobian: Tensor   # (T, n, d, 7), or (T, n, d, 9) with the stretch ([rho, theta, l])
    weight: Tensor     # (T, n) IRLS weights on squared residuals (0 = ignore)
    # (T,) per-frame cost ADDED to the merit function; not a residual. It carries no Jacobian and never
    # reaches `term_blocks`, so it cannot change the Gauss-Newton direction -- only which steps are
    # accepted. `None` (the default) is the shipped behaviour; see `SolverConfig.unbalanced_merit`.
    extra: Tensor | None = None
    # (T,) per-frame cost that REPLACES `Σ weight·‖residual‖²` in the merit. Like `extra` it never reaches
    # `term_blocks`, so the Gauss-Newton direction is unchanged -- only which steps are accepted. `None` (the
    # default) is the shipped behaviour.
    merit: Tensor | None = None


def generalised_kl(p: Tensor, q: Tensor) -> Tensor:
    """`sum_i [p_i log(p_i/q_i) - p_i + q_i]` over the last dim: the unbalanced-OT marginal penalty.

    Exactly zero when `p == q`, and finite as `p_i -> 0`, where the limit is `q_i` rather than `-inf` --
    which is the regime that matters here, a plan that has abandoned its mass. Rows with `p_i == 0`
    contribute `q_i` without ever evaluating `log 0`, and `q_i == 0` (reachable for a hidden mesh sample)
    is clamped rather than allowed to produce a NaN.
    """
    ratio = (p.clamp_min(1e-30) / q.clamp_min(1e-30)).log()
    return (torch.where(p > 0, p * ratio, torch.zeros_like(p)) - p + q).sum(-1)


def _mass_deficiency(rho_x: Param, w: Tensor, a: Tensor) -> Tensor:
    """(T,) `rho_x * KL(w || a)`, defined as exactly 0 wherever `rho_x` is infinite.

    `rho_x = inf` is BALANCED OT: the row marginal is enforced exactly, so no observation mass can be
    abandoned and the deficiency is structurally zero. Computing it anyway multiplies `inf` by Sinkhorn's
    finite-iteration marginal error and yields `inf` or `NaN`, poisoning the merit of every balanced solve.
    """
    kl = generalised_kl(w, a)
    rho = torch.as_tensor(rho_x, dtype=kl.dtype, device=kl.device)
    return torch.where(torch.isinf(rho), torch.zeros_like(kl), rho * kl)


def transform_points(Y: Tensor, poses: Tensor, scale: Tensor | float) -> Tensor:
    """Canonical points Y (M,3) or (T,M,3) -> camera frame (T,M,3): s R y + t. `scale` is a scalar or a
    per-frame (T,) tensor (frames of different clips carry their own clip's scale)."""
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    if isinstance(scale, Tensor) and scale.dim() == 1:
        scale = scale[:, None, None]
    if Y.dim() == 2:
        Y = Y.unsqueeze(0).expand(poses.shape[0], -1, -1)
    return scale * torch.einsum("tij,tmj->tmi", R, Y) + t[:, None, :]


def point_param_jacobian(p: Tensor, t: Tensor, axes: Tensor | None = None) -> Tensor:
    """d p / d [rho, theta, sigma] for camera points p (T,n,3) with frame translations t (T,3): (T,n,3,7).

    Anisotropic scale: with `axes` W = R U (T,3,3) (`aniso.Stretch.rotated_axes`), p = R A (y - c) + t and the scale
    column becomes the three stretch columns d p / d l (`aniso.stretch_jacobian`): (T,n,3,9)."""
    eye = torch.eye(3, dtype=p.dtype, device=p.device).expand(*p.shape[:-1], 3, 3)
    if axes is not None:
        return torch.cat([eye, -hat(p), stretch_jacobian(p, t, axes)], dim=-1)
    d_sigma = (p - t[:, None, :]).unsqueeze(-1)          # d(s R y + t)/d sigma = s R y = p - t
    return torch.cat([eye, -hat(p), d_sigma], dim=-1)


def _intrinsics(K: Tensor, p: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(fx, fy, cx, cy) broadcastable against `p[..., 0]`: `K` is (3,3), or (T,3,3) per frame for p (T,...,3)."""
    if K.dim() == 2:
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    shape = (K.shape[0],) + (1,) * (p.dim() - 2)
    return (K[:, 0, 0].reshape(shape), K[:, 1, 1].reshape(shape), K[:, 0, 2].reshape(shape),
            K[:, 1, 2].reshape(shape))


def project(p: Tensor, K: Tensor) -> Tensor:
    z = p[..., 2].clamp_min(1e-6)
    fx, fy, cx, cy = _intrinsics(K, p)
    return torch.stack([fx * p[..., 0] / z + cx, fy * p[..., 1] / z + cy], dim=-1)


def projection_jacobian(p: Tensor, K: Tensor) -> Tensor:
    """d pi / d p: (...,2,3)."""
    z = p[..., 2].clamp_min(1e-6)
    zero = torch.zeros_like(z)
    fx, fy, _cx, _cy = _intrinsics(K, p)
    row0 = torch.stack([fx / z, zero, -fx * p[..., 0] / (z * z)], dim=-1)
    row1 = torch.stack([zero, fy / z, -fy * p[..., 1] / (z * z)], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _squared(dist: Tensor) -> Tensor:
    """`dist ** 2`, in place when no gradient is tracked (a batch of clips must not hold two (T,N,M) float64
    distance tensors at once); bit-identical either way."""
    return dist.pow_(2) if not dist.requires_grad else dist**2


def _squared_distances(A: Tensor, B: Tensor) -> Tensor:
    """Squared Euclidean distances `(T,N,M)` between `A` (T,N,3) and `B` (T,M,3), from explicit differences.

    Numerically the same as `torch.cdist(A, B, compute_mode="donot_use_mm_for_euclid_dist") ** 2` (no
    `x^2 + y^2 - 2xy` cancellation; no square root to undo), but summed one coordinate at a time with
    in-place ops instead of through cdist's generic pairwise kernel: measured ~50x faster on the whisking
    solve's (46,1024,2048) float32 matrix (4 ms vs 160-270 ms per call), max relative difference 3e-7,
    with a peak of two `(T,N,M)` buffers.
    """
    C = (A[:, :, None, 0] - B[:, None, :, 0]).square_()
    for i in (1, 2):
        C.add_((A[:, :, None, i] - B[:, None, :, i]).square_())
    return C


@overload
def pointmap_term(X: Tensor, X_valid: Tensor, Y: Tensor, poses: Tensor, log_scale: Tensor, eps: Param,
                  rho_x: Param, rho_y: Param, n_sinkhorn: int, plan: Tensor | SparsePlan | PlanMoments | None = None,
                  mesh_mask: Tensor | None = None, normals: Tensor | None = None,
                  point_to_plane: bool = False, *,
                  sinkhorn_dtype: torch.dtype | None = None,
                  knn: int | None = None, mesh_valid: Tensor | None = None, fused_sinkhorn: bool = False,
                  with_deficiency: bool = False, moments: bool = False, stretch: Stretch | None = None
                  ) -> tuple[TermOutput, Tensor | SparsePlan | PlanMoments]: ...


@overload
def pointmap_term(X: Tensor, X_valid: Tensor, Y: Tensor, poses: Tensor, log_scale: Tensor, eps: Param,
                  rho_x: Param, rho_y: Param, n_sinkhorn: int, plan: Tensor | SparsePlan | PlanMoments | None = None,
                  mesh_mask: Tensor | None = None, *,
                  potentials: tuple[Tensor, Tensor] | None, normals: Tensor | None = None,
                  point_to_plane: bool = False, sinkhorn_dtype: torch.dtype | None = None,
                  knn: int | None = None, mesh_valid: Tensor | None = None, fused_sinkhorn: bool = False,
                  with_deficiency: bool = False, moments: bool = False, stretch: Stretch | None = None
                  ) -> tuple[TermOutput, Tensor | SparsePlan | PlanMoments,
                             tuple[Tensor | None, Tensor | None]]: ...


def pointmap_term(X: Tensor, X_valid: Tensor, Y: Tensor, poses: Tensor, log_scale: Tensor, eps: Param,
                  rho_x: Param, rho_y: Param, n_sinkhorn: int, plan: Tensor | SparsePlan | PlanMoments | None = None,
                  mesh_mask: Tensor | None = None, normals: Tensor | None = None,
                  point_to_plane: bool = False,
                  potentials: object = _NO_POTENTIALS, sinkhorn_dtype: torch.dtype | None = None,
                  knn: int | None = None, mesh_valid: Tensor | None = None, fused_sinkhorn: bool = False,
                  with_deficiency: bool = False, moments: bool = False, stretch: Stretch | None = None
                  ) -> (tuple[TermOutput, Tensor | SparsePlan | PlanMoments]
                        | tuple[TermOutput, Tensor | SparsePlan | PlanMoments, tuple[Tensor | None, Tensor | None]]):
    """Partial-OT pointmap term. X (T,N,3) observed camera points (NaN allowed where invalid), Y (M,3).

    `mesh_mask` (T,M), when given, restricts and/or reweights the mesh-side marginal. It may
    be a plain boolean visibility mask, or (the foreshortening weighting) any
    nonnegative *continuous* per-sample weight - e.g. `max(n_j.v_j, 0)/z_j^2` times a boolean mask - the
    two are handled identically: the mesh-side mass is `b = mesh_mask / mesh_mask.sum(dim=1,
    keepdim=True)` (a plain 0/1 mask reduces to the original uniform `1/|V_k|` on the visible set;
    zeroed-out samples are excluded from Sinkhorn regardless of whether the zero came from the mask or
    from a clamped weight). Only affects the branch that actually runs Sinkhorn (`plan is None`); a plan
    supplied directly is used as-is regardless of `mesh_mask`. See `ot.sinkhorn_unbalanced_potentials`'s
    docstring for how exact zeros in `b` are handled in the log domain without NaN.

    A frame whose `mesh_mask` row sums to (numerically) zero - e.g. `visibility.py`'s own z-buffer found
    no visible samples at all (see `visible_samples`'s `_MIN_VISIBLE` fallback, which normally prevents
    this by falling back to the full template first) or, with the foreshortening weighting above, every
    visible sample happens to be at exactly grazing incidence so the `max(...,0)` clamp zeros all of
    them - falls back to the uniform full template (`b = 1/M`) for that frame, logged at DEBUG. This is
    the same "never hand Sinkhorn an empty marginal" policy as `visible_samples`'s own fallback, just
    applied at the point where a *continuous* weight (not just the boolean mask) could still end up
    all-zero for a frame; documenting both together here since they are two instances of one policy.

    If `point_to_plane` is True, `normals` (M,3) unit canonical normals paired with `Y` are required;
    the residual is the scalar projection of the point-to-point vector onto the barycentric surface
    normal (T,N,1) instead of the full 3-vector (T,N,3). This discards the tangential component of
    the residual, which under wide vertex spacing is dominated by surface-sampling error rather than
    the depth signal the observations actually constrain.

    `potentials`, when passed (even as `None`), switches the return value from the legacy 2-tuple
    `(TermOutput, plan)` to a 3-tuple `(TermOutput, plan, (f, g))` that also hands back the Sinkhorn
    dual potentials. Passing a prior `(f, g)` pair (e.g. from the previous LM iteration's solve at a
    nearby pose) warm-starts the Sinkhorn recompute so far fewer iterations are needed to converge; a
    plan supplied directly (skipping the Sinkhorn recompute entirely) passes the input potentials
    through unchanged since no dual ascent ran.

    `sinkhorn_dtype`, when given, runs only the Sinkhorn solve (the cost matrix, dual potentials and
    log-sum-exp iterations - the dominant GPU cost at T=100, the runtime-budget scenario) in
    that dtype (e.g. `torch.float32`), casting the resulting plan and potentials back to `X`'s own
    dtype before the barycenter/residual/Jacobian computation, which always stays in `X`'s dtype.
    Defaults to `None` (no cast, Sinkhorn runs in `X`'s own dtype exactly as before) because
    `test_pointmap_residual_zero_at_truth_and_jacobian`'s finite-difference Jacobian check needs full
    float64 precision throughout (an h=1e-6 perturbation is meaningless in float32).

    `knn`, when given, restricts the Sinkhorn recompute (the `plan is None`
    branch only - like `mesh_mask`, a plan supplied directly is used as-is regardless of `knn`) to each
    observed point's `knn` nearest samples with nonzero `mesh_mask` weight: hidden columns are set to
    +inf before `torch.topk` so a hidden back-face sample is never a candidate. The
    returned `plan` becomes a `SparsePlan` (GLOBAL-indexed candidate columns, never candidate-slot
    indexed) instead of a dense (T,N,M) tensor; `plan_barycenters_sparse` computes barycenters from it
    exactly as `plan_barycenters` does for a dense plan, and the cached-plan (`plan=` supplied directly)
    path accepts a `SparsePlan` the same way it accepts a dense one. `knn >= M` is clamped to `M` (every
    column is a candidate - the exactness configuration `test_ot_sparse.py` checks against the dense
    solver); `None` or `0` skip the candidate search entirely and behave exactly as before this
    parameter existed (a dense plan, unchanged).

    `Y` may be per-frame (T,M,3) (and `normals` (T,M,3)), `log_scale` per-frame
    (T,), and `eps`/`rho_x`/`rho_y` per-frame (T,) tensors. `mesh_valid` (T,M) bool marks real samples
    of a padded template: it replaces the all-ones template in the uniform marginal (`mesh_mask=None`)
    and in the zero-row fallback, so padding never receives mesh mass. `fused_sinkhorn` opts the dense Sinkhorn
    into `ot_fused`'s CUDA kernels (`ot.sinkhorn_unbalanced_potentials`'s `fused`).

    `with_deficiency` (default False) also returns `extra`, the (T,) `rho_x * KL(P1 || a)` the plan
    owes for observation mass it abandoned, which `lm.total_cost` / `lm.clip_costs` add to the merit. It is
    a keyword the CALLER must opt into -- not a field stripped later by one caller -- so every path that
    does not ask gets exactly the shipped behaviour.

    Plan moments (`SolverConfig.plan_moments`): `plan` may also be a `moments.PlanMoments`, the plan's canonical moments
    (B, w, and the unit canonical normal for `point_to_plane`); the term is then evaluated from pbar = s R B + t and
    nbar = R Nbar without any (T,N,M) or (T,N,k) tensor (`fastdo.solver.moments` has the algebra and the zero-mass
    factor on t). `moments=True` hands back `PlanMoments` in place of the plan: a freshly solved (or a supplied
    dense/sparse) plan is reduced against the CANONICAL `Y` (and `normals` when `point_to_plane`), dropped, and the
    term is built from the moments, so a refresh and the trials that reuse its moments evaluate one function. Moments
    reduced without normals refuse `point_to_plane` (ValueError) rather than silently change the residual.
    `with_deficiency` reads only the row mass w, which the moments carry exactly (the same `P.sum(-1)`). The residual,
    Jacobian (the same left-perturbation formula at the same pbar) and returned structures keep their shapes and
    meaning; values differ from the plan path at float round-off (reassociated sums).

    Anisotropic `stretch` (`aniso.Stretch` of these T frames, `SolverConfig.anisotropic_scale`): the template enters as
    y -> A (y - c), so the transformed template is R A (y - c) + t, and the caller passes `log_scale` = 0 (the stretch
    carries the scale). Moments are still reduced against the CANONICAL `Y`, so they stay valid at every trial stretch
    (`moments.transported_points`), and the Jacobian's scale column becomes the three stretch columns (T,N,3,9). The
    point-to-plane residual is not implemented under a stretch (its normal would also depend on l, and the batched
    callback does not use it): `point_to_plane` with a `stretch` raises ValueError.
    """
    if point_to_plane and stretch is not None:
        raise ValueError("point_to_plane is not implemented with an anisotropic stretch")
    if point_to_plane and normals is None:
        raise ValueError("normals (M,3) is required when point_to_plane=True")
    if isinstance(plan, PlanMoments) and point_to_plane and plan.normal is None:
        raise ValueError("point_to_plane needs PlanMoments reduced with normals (plan_moments(..., normals=...))")
    T, N, _ = X.shape
    M = Y.shape[-2]
    s = log_scale.exp()
    # The moments path never forms the transformed template; it transforms the (T,N,3) barycenters instead.
    Yk = (transform_points(Y if stretch is None else stretch.apply(Y), poses, s)
          if not isinstance(plan, PlanMoments) else None)                                    # (T,M,3)
    Xc = torch.where(X_valid[..., None], X, torch.zeros_like(X))
    want_potentials = potentials is not _NO_POTENTIALS
    init = cast("tuple[Tensor, Tensor] | None",
               potentials) if (want_potentials and potentials is not None) else None
    f: Tensor | None
    g: Tensor | None
    # `a` is needed on BOTH paths, not just where Sinkhorn runs. The merit function charges
    # `rho_x * KL(P1 || a)` for the observation mass the plan abandons, and a trial step is scored against
    # a STALE cached plan, so the marginal it is compared to has to be the same one. It depends only on
    # `X_valid`, so it is identical either way and costs one normalisation on the cached path.
    a = X_valid.to(X.dtype)
    a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)
    if plan is None:
        assert Yk is not None
        valid_f = mesh_valid.to(X.dtype) if mesh_valid is not None else None
        if mesh_mask is None and valid_f is None:
            b = torch.full((T, M), 1.0 / M, dtype=X.dtype, device=X.device)
        else:
            bm = mesh_mask.to(X.dtype) if mesh_mask is not None else cast("Tensor", valid_f)
            zero_rows = bm.sum(dim=1, keepdim=True).squeeze(-1) < _ZERO_WEIGHT_ROW_TOL
            # See the docstring above: fall back to the uniform full template for a frame whose
            # mesh_mask carries no usable signal at all, consistent with visible_samples's own
            # _MIN_VISIBLE fallback. Applied unconditionally (a no-op `where` when no row is zero)
            # rather than behind a `.any()` check, to avoid a host/device sync on this hot per-refresh
            # path (see visibility.py's own docstring for why that matters here).
            bm = torch.where(zero_rows[:, None], torch.ones_like(bm) if valid_f is None else valid_f, bm)
            if logger.isEnabledFor(logging.DEBUG) and bool(zero_rows.any()):
                logger.debug("pointmap_term: %d/%d frame(s) have an all-zero mesh_mask row; falling "
                            "back to the uniform full template for them", int(zero_rows.sum()), T)
            b = bm / bm.sum(dim=1, keepdim=True).clamp_min(1.0)
        sd = sinkhorn_dtype if sinkhorn_dtype is not None else X.dtype
        # Cost matrix computed in float64 with the default (matmul-based, x^2+y^2-2xy) cdist, then cast
        # to the Sinkhorn dtype. The mm path loses precision to catastrophic
        # cancellation for nearby points (the reason for the explicit non-mm kernel below), but
        # only ~1e-16*|x|^2 in float64 - negligible next to eps (as small as 1e-4 * diameter^2 at the
        # fine level; test_ot_sparse.py's test_matmul_cdist_float64_matches_precision_mode confirms this
        # to 1e-10 relative on unit-diameter points). Measured ~100x faster than the non-mm kernel on
        # this GPU/torch build: computing in float64 and casting down
        # is NOT the throwaway-of-float32's-speed-benefit it would be for a fast kernel, because the
        # non-mm kernel it replaces was never fast to begin with. float64 is unsupported on MPS (not a
        # target backend for this CUDA-only codebase, but checked defensively rather than assumed): the
        # slow, precision-mode float32 kernel is kept for that one case only.
        if X.dtype == torch.float32 and X.device.type == "mps":
            C = torch.cdist(Xc.to(sd), Yk.to(sd), compute_mode="donot_use_mm_for_euclid_dist") ** 2
        else:
            C = _squared(torch.cdist(Xc.double(), Yk.double())).to(sd)
        init_sd = (init[0].to(sd), init[1].to(sd)) if init is not None else None
        a_sd, b_sd = a.to(sd), b.to(sd)
        k_eff = min(knn, M) if knn else 0
        if k_eff > 0:
            # Candidate search only over samples with
            # nonzero mesh weight - a hidden back-face sample is never a candidate, so hide it from
            # torch.topk by setting its search distance to +inf. C_k itself gathers from the REAL
            # (unmasked) cost matrix: a frame with fewer visible samples than knn can still fill its
            # candidate slots with masked-out columns, which is harmless (their log-mass is already
            # -inf via `_safe_log_mass` inside `sinkhorn_unbalanced_sparse`, the same mechanism the
            # dense path already relies on) rather than needing a second "no mass" code path here.
            C_search = torch.where(b_sd[:, None, :] > 0, C, torch.full_like(C, float("inf")))
            cand_idx = torch.topk(-C_search, k_eff, dim=2).indices
            C_k = torch.gather(C, 2, cand_idx)
            values, (f, g) = sinkhorn_unbalanced_sparse(C_k, cand_idx, a_sd, b_sd, eps, rho_x, rho_y,
                                                        n_sinkhorn, potentials=init_sd)
            plan = SparsePlan(indices=cand_idx, values=values.to(X.dtype))
            f, g = f.to(X.dtype), g.to(X.dtype)
            if moments:
                del C_search, C_k                                         # dense search buffer, see below
        else:
            plan, f, g = sinkhorn_unbalanced_potentials(C, a_sd, b_sd, eps, rho_x, rho_y, n_sinkhorn,
                                                        init=init_sd, fused=fused_sinkhorn)
            plan, f, g = plan.to(X.dtype), f.to(X.dtype), g.to(X.dtype)
        if moments:
            del C                                                         # the refresh chunk ends here
    else:
        f, g = init if init is not None else (None, None)
    if moments and not isinstance(plan, PlanMoments):
        # Reduce to canonical moments and drop the plan before any term is built (fastdo.solver.moments).
        plan = plan_moments(plan, Y, normals if point_to_plane else None)
    nbar_c: Tensor | None = None
    if isinstance(plan, PlanMoments):
        pbar, w, nbar_c = transported_points(plan, poses, s, stretch), plan.mass, plan.normal   # (T,N,3), (T,N), (T,N,3)
    elif isinstance(plan, SparsePlan):
        pbar, w = plan_barycenters_sparse(plan.values, plan.indices, cast("Tensor", Yk))   # (T,N,3), (T,N)
    else:
        pbar, w = plan_barycenters(plan, cast("Tensor", Yk))                                # (T,N,3), (T,N)
    scale_w = (N * w).clamp_min(0.0).sqrt()[..., None]                    # unit weight for a balanced plan
    # The deficiency the pose objective never saw. `scale_w` above is exactly why the degeneracy
    # exists -- under unbalanced OT a model far from the data transports no mass, `w -> 0`, and the
    # residual vanishes however wrong the pose is, while `weight` below stays at full validity. This is
    # the `rho_x * KL(P1 || a)` the Sinkhorn already paid for, so that abandoning the observations costs
    # what it should. It is summed into the merit only behind `SolverConfig.unbalanced_merit`.
    #
    # The mesh-side `rho_y * KL(P^T 1 || b)` is deliberately NOT included: `b` depends on `mesh_mask`,
    # which the CACHED path is not given (`batch_eval`'s `cached`), so it would differ between a trial
    # step and the baseline it is compared against -- and an inconsistent term is worse than a missing
    # one. It is also 30-100x smaller than the row term in measurement, and the row term is the one that
    # creates the degeneracy.
    extra = _mass_deficiency(rho_x, w, a) if with_deficiency else None     # (T,) or None
    t = poses[:, :3, 3]
    if point_to_plane:
        assert normals is not None
        R = poses[:, :3, :3]
        # Barycentric normal averaged in the canonical frame (like pbar's ybar), then rotated into
        # the camera frame; unlike pbar it does not pick up translation or scale (a direction field).
        # With moments the unit canonical normal was built at the reduction, by these same operations.
        if nbar_c is None:
            if isinstance(plan, SparsePlan):
                normals_bc = normals if normals.dim() == 3 else normals.unsqueeze(0).expand(T, -1, -1)
                nbar_raw, _ = plan_barycenters_sparse(plan.values, plan.indices, normals_bc)
            else:
                nbar_raw = cast("Tensor", plan) @ normals
            nbar_c = torch.nn.functional.normalize(nbar_raw / w.clamp_min(1e-12)[..., None], dim=-1)
        nbar = torch.einsum("tij,tnj->tni", R, nbar_c)                    # (T,N,3)
        diff = Xc - pbar
        residual = scale_w * (nbar * diff).sum(-1, keepdim=True)          # (T,N,1)
        J = scale_w[..., None] * torch.cat([
            -nbar,
            torch.cross(nbar, Xc, dim=-1),
            -(nbar * (pbar - t[:, None, :])).sum(-1, keepdim=True),
        ], dim=-1)[..., None, :]                                          # (T,N,1,7)
    else:
        residual = scale_w * (Xc - pbar)
        J = -scale_w[..., None] * point_param_jacobian(pbar, t, None if stretch is None else stretch.rotated_axes(poses))
    weight = X_valid.to(X.dtype)
    residual = torch.where(X_valid[..., None], residual, torch.zeros_like(residual))
    J = torch.where(X_valid[..., None, None], J, torch.zeros_like(J))
    out = TermOutput(residual=residual, jacobian=J, weight=weight, extra=extra)
    if not want_potentials:
        return out, plan
    return out, plan, (f, g)


def track_term(anchors: Tensor, tracks: Tensor, vis: Tensor, poses: Tensor, log_scale: Tensor, K: Tensor,
               cauchy_c: float, cauchy_merit: bool = False, stretch: Stretch | None = None) -> TermOutput:
    """Reprojection of canonical anchors (J,3) against 2D tracks (T,J,2) with Cauchy IRLS weights.

    `tracks` may carry NaN placeholders wherever `vis` is False; those entries are sanitized before
    use so `residual`, `jacobian`, and `weight` are always finite (weight is exactly 0, not NaN).

    `weight = 1/(1 + r²/c²)` is exactly `ρ′(s)`, s = r², for the Cauchy loss `ρ(s) = c²·log1p(s/c²)` (normalised
    to `≈ s` near zero, the scale of the least-squares terms), so the weighted Gauss-Newton step descends Cauchy. The
    shipped merit `Σ w·s`, with `w` re-evaluated at the trial point, is instead the Geman-McClure loss. With
    `cauchy_merit` the term also returns `merit = Σ vis·ρ(s)` per frame, so LM can accept on the function its step
    descends. The residual, Jacobian and weight are identical either way.

    Anisotropic `stretch`: the anchors are template points under the same transform, R A (a - c) + t (`log_scale` = 0),
    and the Jacobian carries the three stretch columns (T,J,2,9).
    """
    p = transform_points(anchors if stretch is None else stretch.apply(anchors), poses, log_scale.exp())   # (T,J,3)
    tracks_c = torch.where(vis[..., None], tracks, torch.zeros_like(tracks))
    residual = project(p, K) - tracks_c                                   # (T,J,2)
    axes = None if stretch is None else stretch.rotated_axes(poses)
    J = projection_jacobian(p, K) @ point_param_jacobian(p, poses[:, :3, 3], axes)   # (T,J,2,7) or (T,J,2,9)
    r2 = (residual * residual).sum(-1)
    weight = vis.to(residual.dtype) / (1.0 + r2 / (cauchy_c * cauchy_c))
    residual = torch.where(vis[..., None], residual, torch.zeros_like(residual))
    merit = None
    if cauchy_merit:
        c2 = cauchy_c * cauchy_c
        # From the SANITISED residual, and masked by vis: an occluded entry costs exactly nothing.
        merit = (vis.to(residual.dtype) * c2 * torch.log1p((residual * residual).sum(-1) / c2)).sum(-1)
    return TermOutput(residual=residual, jacobian=J, weight=weight, merit=merit)


# Below this camera-frame depth (metres) the anchor's log is taken of the clamp. The value is `project`'s clamp.
_ANCHOR_MIN_DEPTH = 1e-6


def depth_anchor_term(centroid: Tensor, log_depth_obs: Tensor, depth_valid: Tensor, poses: Tensor, log_scale: Tensor,
                      lam: float, band: float, stretch: Stretch | None = None) -> TermOutput:
    """Dead-zone absolute depth anchor: one scalar residual per frame, r_t = sqrt(lam) h(d_t).

    d_t = log z_sol(t) - log z_obs(t), z_sol the camera-frame z of the posed template centroid (s R c + t)_z and
    z_obs = exp(`log_depth_obs`) the frame's observed object depth (`batch.frame_depths`). The dead zone is
    h(d) = sign(d) max(0, |d| - band): inside the band (and on frames with `depth_valid` False) the residual AND its
    Jacobian are exact zeros, selected by `torch.where` rather than multiplied by a zero weight, so an in-band frame
    adds exact zeros to the normal equations and to the merit. Outside the band the penalty lam (|d| - band)^2 is convex
    and never redescends: it restores a solve the redescending pointmap term has abandoned (an absorbing state, seen
    on one DexYCB development clip: 1.45 m solved against a 0.68-0.74 m pointmap).

    Jacobian (T,1,1,7): with the left perturbation [rho, theta, sigma], d z / d param is row z of
    `point_param_jacobian` at the posed centroid p ([e_z, -(hat p)_z, (p - t)_z]), and d h / d param = (1 / z) d z /
    d param outside the band, with the sigma (and stretch) columns ZEROED: the anchor never changes the object's size.
    For a centroid at or behind the camera (z below `_ANCHOR_MIN_DEPTH`) the log is taken of
    the clamp and the Jacobian uses the clamped z: a steep forward push, not the zero gradient of a clamp.

    `centroid` (T,3) is each frame's canonical template centroid; `log_depth_obs`/`depth_valid` (T,) (finite everywhere;
    invalid frames carry any finite placeholder). The weight is `depth_valid` (0/1): lam lives in the residual.
    Anisotropic `stretch`: the centroid is transformed like any template point, R A (c - c_p) + t with `log_scale` = 0,
    and the Jacobian carries the three stretch columns (T,1,1,9)."""
    c = centroid[:, None, :]
    p = transform_points(c if stretch is None else stretch.apply(c), poses, log_scale.exp())       # (T,1,3)
    z = p[..., 2].clamp_min(_ANCHOR_MIN_DEPTH)                                                    # (T,1)
    d = z.log() - log_depth_obs[:, None]
    excess = d.abs() - band
    outside = (excess > 0) & depth_valid[:, None]                                                 # (T,1)
    root = lam ** 0.5
    zero = torch.zeros((), dtype=p.dtype, device=p.device)
    h = torch.where(outside, torch.sign(d) * excess, zero)
    axes = None if stretch is None else stretch.rotated_axes(poses)
    dz = point_param_jacobian(p, poses[:, :3, 3], axes)[..., 2, :]                                # (T,1,7|9)
    # The anchor moves the along-ray PLACEMENT only: its sigma and stretch columns are zeroed, so it never changes the
    # object's size. With them in, a fixture without tracks drove the scale to ~0 (a smaller object nearer the camera
    # satisfies the same centroid depth); size stays with the terms that see it.
    dz = torch.cat([dz[..., :6], torch.zeros_like(dz[..., 6:])], dim=-1)
    J = torch.where(outside[..., None], (root / z)[..., None] * dz, zero)
    return TermOutput(residual=(root * h)[..., None], jacobian=J[..., None, :], weight=depth_valid.to(p.dtype)[:, None])


def _se3_ad(xi: Tensor) -> Tensor:
    """se(3) bracket operator ad_xi (...,6,6) for xi = [rho, theta], such that ad_xi @ eta = [xi, eta]."""
    rho, theta = xi[..., :3], xi[..., 3:]
    K_theta = hat(theta)
    K_rho = hat(rho)
    zero = torch.zeros_like(K_theta)
    top = torch.cat([K_theta, K_rho], dim=-1)
    bottom = torch.cat([zero, K_theta], dim=-1)
    return torch.cat([top, bottom], dim=-2)


# Bernoulli numbers B_0..B_10 (B_odd>=3 are zero): coefficients of the J_l^{-1}(xi) = sum B_n/n! ad_xi^n series.
_BERNOULLI = (1.0, -0.5, 1.0 / 6, 0.0, -1.0 / 30, 0.0, 1.0 / 42, 0.0, -1.0 / 30, 0.0, 5.0 / 66)


def _se3_left_jacobian_inv(xi: Tensor) -> Tensor:
    """Inverse left Jacobian of the SE(3) exponential map at xi (...,6) -> (...,6,6).

    Satisfies Log(Exp(eps) Exp(xi)) ~= xi + J_l^{-1}(xi) @ eps for small eps; computed via the
    convergent Bernoulli series in the se(3) adjoint operator ad_xi, which reduces exactly to the
    familiar closed form on the so(3) block when rho = 0.
    """
    ad = _se3_ad(xi)
    eye = torch.eye(6, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 6, 6)
    result = torch.zeros_like(eye)
    power = eye
    fact = 1.0
    for n, b in enumerate(_BERNOULLI):
        if n > 0:
            fact *= n
            power = power @ ad
        if b != 0.0:
            result = result + (b / fact) * power
    return result


def smoothness_term(poses: Tensor, lam_v: float) -> tuple[Tensor, Tensor, Tensor]:
    """Constant-pose prior r_k = sqrt(lam) Log(T_k^-1 T_{k+1}); Jacobians w.r.t. xi_k and xi_{k+1}.

    The residual is a left perturbation of T_k composed with a left perturbation of T_{k+1}, so
    differentiating through Log(...) needs the inverse left Jacobian of the log map, not just the
    group Adjoint: d Log(Exp(eps) rel)/d eps = J_l^{-1}(Log(rel)) at eps = 0. Concretely, with
    r_k = sqrt(lam) Log(rel) and Ad = Ad(T_k^-1):

        d r_k / d xi_{k+1} =  sqrt(lam) * J_l^{-1}(r_k) @ Ad
        d r_k / d xi_k     = -sqrt(lam) * J_l^{-1}(r_k) @ Ad

    `J_l^{-1}` is the inverse left Jacobian of SE(3) at the (unscaled) log-residual, computed here
    by `_se3_left_jacobian_inv` via its convergent Bernoulli series in the se(3) adjoint operator
    ad_xi (this is exact, not merely a first-order approximation like the group Adjoint alone).
    """
    return relative_motion_term(poses[:-1], poses[1:], lam_v)


def relative_motion_term(from_poses: Tensor, to_poses: Tensor, lam_v: float) -> tuple[Tensor, Tensor, Tensor]:
    """`smoothness_term` over explicit pose pairs (P,4,4), (P,4,4): r = sqrt(lam) Log(from^-1 to) and its Jacobians
    w.r.t. the `from` and `to` left perturbations (the batched solve evaluates only the pairs inside one clip)."""
    inv_k = se3_inv(from_poses)
    rel = inv_k @ to_poses
    root = lam_v ** 0.5
    log_rel = se3_log(rel)
    r = root * log_rel
    Ad = se3_adjoint(inv_k)
    corr = _se3_left_jacobian_inv(log_rel) @ Ad
    return r, -root * corr, root * corr


def velocity_term(poses: Tensor, lam_v: float, dt: float = 1.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Constant-velocity (second-difference) prior on SE(3).

    Convention: a_j = Log(T_j^-1 T_{j+1}) is the body-frame relative motion of `smoothness_term` (same
    factor inverted). Design 11.7 writes the spatial form Log(T_{j+1} T_j^-1); both vanish on exactly the
    constant-twist trajectories T_k = Exp(k xi) T_0 = T_0 Exp(k xi'), and this form keeps the
    constant-pose term's invariance to a fixed change of camera frame. For k = 1..T-2 (stored at k-1):

        r_k = s (a_k - a_{k-1}),    s = sqrt(lam_v / dt^3).

    `lam_v` is the inverse acceleration-noise density 1/q_c, so in the solver's frame units (dt = 1) the
    weight is 1/(dt^3 q_c). With C_j = J_l^-1(a_j) Ad(T_j^-1), `smoothness_term`'s
    identities d a_j/d xi_{j+1} = C_j = -d a_j/d xi_j give the three (T-2,6,6) left-perturbation blocks

        J_prev = d r_k/d xi_{k-1} =  s C_{k-1}
        J_mid  = d r_k/d xi_k     = -s (C_k + C_{k-1})
        J_next = d r_k/d xi_{k+1} =  s C_k

    returned as the `Smooth2` quadruple `(r, J_prev, J_mid, J_next)` (`fastdo.solver.banded`). Fewer than
    three frames give empty (0,6) / (0,6,6) tensors.
    """
    if poses.shape[0] < 3:
        empty = poses.new_zeros(0, 6, 6)
        return poses.new_zeros(0, 6), empty, empty.clone(), empty.clone()
    a, da_own, da_next = smoothness_term(poses, 1.0)       # a_j; d a_j/d xi_j = -C_j; d a_j/d xi_{j+1} = C_j
    s = (lam_v / dt**3) ** 0.5
    return s * (a[1:] - a[:-1]), -s * da_own[:-1], s * (da_own[1:] - da_next[:-1]), s * da_next[1:]
