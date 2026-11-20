"""Contact-transport objective.

Unbalanced OT between the human contact measure (`fastdo.retarget.contact.ContactMeasure`, band-
weighted atoms on the object surface) and a UNIFORM measure over the robot's sampled surface points
(the contact measure's band weights are *not* reused here -- they stay a diagnostic/reward signal for
later tasks). With the resulting plan held fixed, the OT term reduces to a per-robot-point weighted
least-squares residual (envelope theorem: the "fixed-plan majorizer"), suitable
as a `fastdo.solver.lm_blocks.BlockTerm` in the per-frame kinematic LM (`fastdo.retarget.kinematic`).

Frame convention: both `contact_transport_term`'s inputs and its own internal cost live in the
*canonical* (unit-scale object) frame -- the same frame `fastdo.retarget.contact` uses. Robot points
are expected already forward-kinematics'd in the *world* frame; `robot_canonical_points` maps them
(and their Jacobian) into canonical coordinates using the object's own (held-fixed per frame) pose,
read out of `q`'s free-joint block exactly as `contact.robot_contact_measure` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fastdo.geometry.lie import quat_wxyz_to_matrix
from fastdo.retarget.contact import ContactMeasure
from fastdo.solver.lm_blocks import BlockTerm
from fastdo.solver.ot import (
    plan_barycenters,
    sinkhorn_unbalanced,
    sinkhorn_unbalanced_uniform_b,
    sinkhorn_unbalanced_uniform_b_scaling,
)
from fastdo.solver.sdf import SdfGrid, sdf_lookup

__all__ = ["InbandMeasure", "canonical_diameter", "contact_transport_term", "contact_transport_term_inband",
           "inband_measure", "robot_canonical_points"]

_DEGENERATE_NORMAL_NORM = 1e-6


def canonical_diameter(points: Tensor) -> float:
    """Bounding-box diagonal of `points (M,3)` -- the `D` in "canonical-diameter^2 units"."""
    return float((points.max(dim=0).values - points.min(dim=0).values).norm())


def robot_canonical_points(q: Tensor, world: Tensor, jac: Tensor, mesh_scale: float) -> tuple[Tensor, Tensor]:
    """Map already-FK'd robot points/Jacobian into the canonical (unit-scale object) frame.

    Args:
        q: `(T, 35)` qpos (only the object free-joint block `q[:, 28:35]` -- pos + wxyz quat -- is
            read; it is treated as a per-frame constant, matching `contact.robot_contact_measure`).
        world: `(T, U, 3)` world-frame points, e.g. from `RobotScene.point_kinematics`.
        jac: `(T, U, 3, n)` world-frame positional Jacobian of `world` over the robot dof block
            (`n` = `RobotScene.robot_dof` width, 28 for the Sharpa Wave scene).

    Returns:
        `(points, jac)`: `points (T, U, 3)` canonical-frame positions (`r_u(q)`, *not* projected onto
        the nearest surface sample -- the raw FK point keeps the Jacobian exact);
        `jac (T, U, 3, n)` the canonical-frame Jacobian, `(R_obj^T / mesh_scale) @ jac`.
    """
    dtype = torch.float64
    q_t = q.to(dtype) if isinstance(q, Tensor) else torch.as_tensor(q, dtype=dtype)
    obj_pos = q_t[:, 28:31]
    R_obj = quat_wxyz_to_matrix(q_t[:, 31:35])
    points = torch.einsum("tji,tuj->tui", R_obj, world.to(dtype) - obj_pos[:, None, :]) / mesh_scale
    canon_jac = torch.einsum("tji,tujk->tuik", R_obj, jac.to(dtype)) / mesh_scale
    return points, canon_jac


def contact_transport_term(human: ContactMeasure, robot_points: Tensor, robot_jac: Tensor,
                           robot_labels: Tensor, eps: float, rho_h: float, rho_r: float,
                           lambda_label: float, n_sinkhorn: int = 50, plan: Tensor | None = None,
                           contact_margin: float = 0.0, surface_normals: Tensor | None = None,
                           sdf: SdfGrid | None = None,
                           ) -> tuple[BlockTerm, Tensor]:
    """Fixed-plan contact-transport residual, with an optional outward contact margin.

    Cost (v1, squared Euclidean + label mismatch): `C[v,u] = ||human.points[v] - robot_points[u]||^2
    + lambda_label * 1[human.labels[v] != robot_labels[u]]`. Marginals: `a = human.weights` (the raw,
    unnormalized band weights -- `mu_H`), `b = 1/U` uniform (`mu_R`). Given the resulting
    plan `P (T,V,U)` (recomputed here if `plan is None`, else reused as-is -- the fixed-plan/envelope-
    theorem majorizer), the residual per robot point `u` is `sqrt(U * w_u) * (robot_points[u] -
    pbar_u)` with `w_u = sum_v P[v,u]`, `pbar_u = sum_v P[v,u] * human.points[v] / w_u`; points with
    `w_u < 1e-6 / U` are dropped (zero residual, zero weight).

    Contact margin: with the fixed-plan target `pbar_u` sitting exactly on the object surface (`human.
    points` are themselves nearest-surface-sample projections), sampling noise alone places the
    robot's actual surface point inside the mesh on a large fraction of frames even though clearance
    is excellent -- the human contact band (`sigma`=4mm, `tau`=8mm in `human_contact_measure`) already
    tolerates a few mm of slack, so offsetting the target *outward* by a small `contact_margin` costs
    essentially nothing in contact fidelity while giving the robot surface a non-penetration margin.
    When `contact_margin != 0`, `pbar_u <- pbar_u + contact_margin * nbar_u`, where `nbar_u` is the
    plan-weighted mean of the human contact atoms' surface normals (`surface_normals` indexed by
    `human.sample_idx`, i.e. the normals of the *same* nearest mesh samples `human_contact_measure`
    itself projected onto), normalized; if that weighted mean is degenerate (near zero -- e.g. atoms
    on opposite sides of a thin part cancel out), falls back to the (already outward-pointing) SDF
    gradient direction at `pbar_u`. The residual/Jacobian formulas are otherwise unchanged (the
    offset is a constant per plan refresh, exactly like `pbar_u` itself under the envelope theorem).
    The OT cost is then measured to the offset surface too, `C[v,u] = ||human.points[v] +
    contact_margin * surface_normals[human.sample_idx[v]] - robot_points[u]||^2 + label term`, so the
    plan matches robot samples to the clearance band they are pulled onto (default margin 3 mm in
    `fastdo.retarget.kinematic.RetargetConfig`).

    Args:
        human: The human contact measure for this batch of `T` frames (`ContactMeasure.points`
            already canonical-frame, `ContactMeasure.weights` the raw band weights `mu_H`).
        robot_points: `(T, U, 3)` canonical-frame robot points (raw FK, not surface-projected), e.g.
            from `robot_canonical_points`.
        robot_jac: `(T, U, 3, n)` canonical-frame Jacobian of `robot_points` over the `n`-dim state.
        robot_labels: `(U,)` finger label per robot point (shared across frames).
        eps: Sinkhorn entropic regularization (canonical-diameter^2 units).
        rho_h: Unbalanced KL weight on the human marginal (moderate -- "most human contact mass
            should be reproduced").
        rho_r: Unbalanced KL weight on the robot marginal (small -- "the robot may touch elsewhere
            at low cost").
        lambda_label: Weight of the finger-label mismatch indicator in the OT cost.
        n_sinkhorn: Sinkhorn iterations.
        plan: A previously computed plan `(T, V, U)` to reuse (fixed-plan mode); `None` recomputes it
            from the current `human`/`robot_points`/labels.
        contact_margin: Outward offset applied to `pbar_u`, in the *same* (canonical) units as
            `robot_points` -- the caller converts from metres (dividing by `mesh_scale`) before
            passing this in, matching every other canonical-frame quantity in this module. `0.0`
            (default) reproduces the original (un-offset) behaviour exactly, and needs neither
            `surface_normals` nor `sdf`.
        surface_normals: `(M, 3)` canonical-frame unit normals aligned with the `surface (M,3)` array
            `human_contact_measure` projects onto (i.e. `surface_normals[human.sample_idx]` gives
            each atom's own surface normal). Required when `contact_margin != 0`.
        sdf: Canonical-unit-mesh signed-distance grid, used only as the degenerate-normal fallback
            direction (its gradient already points outward). Required when `contact_margin != 0`.

    Returns:
        `(BlockTerm, plan)`: `BlockTerm.residual`/`.jacobian`/`.weight` flattened over `(U, 3)` to
        `(T, U*3)`/`(T, U*3, n)`/`(T, U*3)`; `plan` is the plan used (freshly computed, or the one
        passed in), for the caller to cache and reuse on the next fixed-plan (trial) evaluation.
    """
    dtype = torch.float64
    T, U, _ = robot_points.shape
    human_points = human.points.to(dtype)
    robot_points = robot_points.to(dtype)
    if contact_margin != 0.0 and (surface_normals is None or sdf is None):
        raise ValueError("surface_normals and sdf are required when contact_margin != 0")

    if plan is None:
        # The cost is measured to the offset surface the targets live on.
        atoms = human_points if surface_normals is None or contact_margin == 0.0 else \
            human_points + contact_margin * surface_normals.to(dtype)[human.sample_idx]
        cost = torch.cdist(atoms, robot_points, compute_mode="donot_use_mm_for_euclid_dist") ** 2
        mismatch = (human.labels.to(torch.int64)[:, None] != robot_labels.to(torch.int64)[None, :]).to(dtype)
        cost = cost + lambda_label * mismatch[None]
        a = human.weights.to(dtype)
        b = torch.full((T, U), 1.0 / U, dtype=dtype)
        plan = sinkhorn_unbalanced(cost, a, b, eps, rho_h, rho_r, n_sinkhorn)

    plan_t = plan.transpose(1, 2)                                                # (T,U,V)
    pbar, w_u = plan_barycenters(plan_t, human_points)                           # (T,U,3), (T,U)

    if contact_margin != 0.0:
        assert surface_normals is not None and sdf is not None   # checked above
        atom_normals = surface_normals.to(dtype)[human.sample_idx]              # (T,V,3)
        nbar_raw, _ = plan_barycenters(plan_t, atom_normals)                     # (T,U,3), unnormalized
        nbar_norm = nbar_raw.norm(dim=-1, keepdim=True)
        degenerate = nbar_norm.squeeze(-1) < _DEGENERATE_NORMAL_NORM
        nbar = nbar_raw / nbar_norm.clamp_min(1e-12)
        if degenerate.any():
            _, grad = sdf_lookup(sdf, pbar)                                      # (T,U,3), points outward
            grad_dir = grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            nbar = torch.where(degenerate[..., None], grad_dir, nbar)
        pbar = pbar + contact_margin * nbar

    mask = w_u >= (1e-6 / U)
    scale_w = torch.sqrt((float(U) * w_u).clamp_min(0.0))                       # (T,U)

    residual = scale_w[..., None] * (robot_points - pbar)                        # (T,U,3)
    residual = torch.where(mask[..., None], residual, torch.zeros_like(residual))
    jac = scale_w[..., None, None] * robot_jac.to(dtype)                         # (T,U,3,n)
    jac = torch.where(mask[..., None, None], jac, torch.zeros_like(jac))
    weight = torch.repeat_interleave(mask.to(dtype), repeats=3, dim=1)           # (T,U*3)

    n = robot_jac.shape[-1]
    term = BlockTerm(
        residual=residual.reshape(T, U * 3),
        jacobian=jac.reshape(T, U * 3, n),
        weight=weight,
    )
    return term, plan


@dataclass(frozen=True)
class InbandMeasure:
    """The human contact measure restricted to its in-band atoms, gathered per frame.

    `V_b = max(1, max_t #{v : w_tv > 0})` rows per frame, ordered by band weight (`torch.topk`). Rows past a frame's
    own in-band count are padding: `valid` False, weight 0, never transported."""

    index: Tensor      # (T, V_b) int64, atom indices into the measure's second axis
    valid: Tensor      # (T, V_b) bool, band weight > 0
    nonempty: Tensor   # (T,) bool, the frame has at least one in-band atom
    points: Tensor     # (T, V_b, 3) canonical atom points
    normals: Tensor    # (T, V_b, 3) surface normals of the atoms' nearest samples
    weights: Tensor    # (T, V_b) band weights, 0 on invalid rows
    labels: Tensor     # (T, V_b) int8 finger labels


def inband_measure(human: ContactMeasure, surface_normals: Tensor) -> InbandMeasure:
    """Gather the atoms with band weight > 0 of every frame (`InbandMeasure`)."""
    dtype = torch.float64
    weights = human.weights.to(dtype)
    counts = (weights > 0).sum(dim=1)
    v_b = max(1, int(counts.max())) if weights.shape[0] > 0 else 1
    index = torch.topk(weights, v_b, dim=1).indices
    gathered = torch.gather(weights, 1, index)
    valid = gathered > 0
    return InbandMeasure(
        index=index, valid=valid, nonempty=counts > 0,
        points=torch.gather(human.points.to(dtype), 1, index[..., None].expand(-1, -1, 3)),
        normals=surface_normals.to(dtype)[torch.gather(human.sample_idx, 1, index)],
        weights=torch.where(valid, gathered, torch.zeros_like(gathered)),
        labels=human.labels.to(torch.int8)[index],
    )


def contact_transport_term_inband(measure: InbandMeasure, robot_points: Tensor, robot_jac: Tensor | None,
                                  robot_labels: Tensor, eps: float, rho_h: float, rho_r: float, lambda_label: float,
                                  n_sinkhorn: int, plan: Tensor | None, contact_margin: float, sdf: SdfGrid,
                                  n_dof: int = 28, sinkhorn: str = "log", robot_jac_frames: Tensor | None = None,
                                  ) -> tuple[BlockTerm, Tensor]:
    """`contact_transport_term` on the in-band rows only; same residual, weight and
    Jacobian conventions.

    A zero-weight atom's row of the unbalanced plan is zero (its KL term is finite only then), so dropping those
    rows leaves the model's minimizer unchanged; against the log-clamped all-rows solver the legacy path runs the
    difference is measured, not certified
    (`tests/retarget/test_kinematic_fast.py::test_inband_rows_exactness_on_whisking`).
    With `plan=None` the plan is solved by `sinkhorn_unbalanced_uniform_b` on non-empty frames only; empty frames
    get an all-zero plan, so their transport weights are 0. The plan is `(T, V_b, U)`; point and normal
    barycentres come from one `bmm`. `robot_jac=None` (a trial evaluation) gives an expanded zero Jacobian.

    The cost matrix is built and the barycentres are taken only on the frames that need them (non-empty frames,
    frames with transported mass): bit-identical. `sinkhorn="scaling"` solves the plan with
    `sinkhorn_unbalanced_uniform_b_scaling` (the same iterations in the scaling domain, equal up to round-off).
    With `robot_jac_frames` (`(T,)` bool), `robot_jac` holds only those frames' rows; every frame with transported
    mass must be among them (ValueError otherwise), since only those frames' Jacobian rows are nonzero.
    """
    dtype = torch.float64
    T, U, _ = robot_points.shape
    robot_points = robot_points.to(dtype)
    if plan is None:
        plan = torch.zeros(T, measure.index.shape[1], U, dtype=dtype)
        if bool(measure.nonempty.any()):
            ne = measure.nonempty
            # The cost only on the frames Sinkhorn solves (per-frame values unchanged).
            pts, nrm = measure.points[ne], measure.normals[ne]
            atoms = pts if contact_margin == 0.0 else pts + contact_margin * nrm
            cost = torch.cdist(atoms, robot_points[ne], compute_mode="donot_use_mm_for_euclid_dist") ** 2
            mismatch = measure.labels[ne].to(torch.int64)[:, :, None] != robot_labels.to(torch.int64)[None, None, :]
            cost = cost + lambda_label * mismatch.to(dtype)
            solve = sinkhorn_unbalanced_uniform_b_scaling if sinkhorn == "scaling" else sinkhorn_unbalanced_uniform_b
            plan[ne] = solve(cost, measure.weights[ne], measure.valid[ne], eps, rho_h, rho_r, n_sinkhorn)

    plan_t = plan.transpose(1, 2)                                                # (T,U,V_b)
    w_u = plan_t.sum(dim=2)
    # Barycentres only on frames with transported mass; elsewhere every row is masked below, so the
    # residual, Jacobian and weight are the same zeros (the per-frame bmm and SDF lookups are unchanged).
    live = (w_u > 0).any(dim=1)
    pbar = torch.zeros(T, U, 3, dtype=dtype)
    if bool(live.any()):
        lp = plan_t[live]
        wl = w_u[live]
        bary = (lp @ torch.cat([measure.points[live], measure.normals[live]], dim=2)) / wl.clamp_min(1e-12)[..., None]
        pl = bary[..., :3]
        if contact_margin != 0.0:
            nbar_raw = bary[..., 3:]
            nbar_norm = nbar_raw.norm(dim=-1, keepdim=True)
            degenerate = nbar_norm.squeeze(-1) < _DEGENERATE_NORMAL_NORM
            nbar = nbar_raw / nbar_norm.clamp_min(1e-12)
            if degenerate.any():
                _, grad = sdf_lookup(sdf, pl)                                    # (T',U,3), points outward
                nbar = torch.where(degenerate[..., None], grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-12),
                                   nbar)
            pl = pl + contact_margin * nbar
        pbar[live] = pl

    mask = w_u >= (1e-6 / U)
    scale_w = torch.sqrt((float(U) * w_u).clamp_min(0.0))
    zero = torch.zeros((), dtype=dtype)
    residual = torch.where(mask[..., None], scale_w[..., None] * (robot_points - pbar), zero)
    if robot_jac is None:
        jac = zero.expand(T, U * 3, n_dof)
    else:
        # Frames without transported mass have an all-False mask, i.e. an all-zero Jacobian.
        jac = torch.zeros(T, U, 3, n_dof, dtype=dtype)
        if bool(live.any()):
            if robot_jac_frames is None:
                live_jac = robot_jac[live]
            else:
                if bool((live & ~robot_jac_frames).any()):
                    raise ValueError("robot_jac_frames misses a frame with transported mass")
                live_jac = robot_jac[live[robot_jac_frames]]
            jac[live] = torch.where(mask[live][..., None, None], scale_w[live][..., None, None] * live_jac.to(dtype),
                                    zero)
        jac = jac.reshape(T, U * 3, n_dof)
    weight = torch.repeat_interleave(mask.to(dtype), repeats=3, dim=1)
    return BlockTerm(residual=residual.reshape(T, U * 3), jacobian=jac, weight=weight), plan
