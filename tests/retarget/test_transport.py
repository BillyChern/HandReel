"""Tests for `fastdo.retarget.transport`: contact-transport term.

The "reaches a target contact measure" synthetic test uses a hand-rolled analytic
2-link planar arm (no MuJoCo) as the "fake hand": both its forward kinematics and Jacobian are
plain closed-form trig, so the test is a from-scratch check of `contact_transport_term` +
`fastdo.solver.lm_blocks.lm_optimize_blocks` working together, independent of `RobotScene`.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import trimesh
from torch import Tensor

from fastdo.retarget.contact import ContactMeasure
from fastdo.retarget.transport import (
    canonical_diameter,
    contact_transport_term,
    robot_canonical_points,
)
from fastdo.solver.lm_blocks import BlockLMConfig, lm_optimize_blocks
from fastdo.solver.ot import sinkhorn_unbalanced
from fastdo.solver.sdf import SdfGrid, mesh_sdf_grid

DT = torch.float64
L1, L2 = 0.6, 0.4


@lru_cache(maxsize=1)
def _unit_box_sdf() -> SdfGrid:
    """A small, cached synthetic SDF (unit box) for the contact_margin degenerate-fallback tests."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return mesh_sdf_grid(np.asarray(box.vertices), np.asarray(box.faces), resolution=24)
# (link index, fraction along the link) for each sampled "robot surface point": one point on link 1
# (the base) and two well-separated points on link 2 (mid + tip). Correspondence is unambiguous with
# this spacing; a denser sampling (adjacent points a few mm apart) makes the toy 2-DOF system's
# entropic-OT correspondence genuinely ambiguous between neighbours, which is a fundamentally
# different (harder, ill-posed) problem from what this test checks -- the real whisking-demo test
# (`test_kinematic.py`) exercises hundreds of points with real geometry instead.
POINT_FRACS: tuple[tuple[int, float], ...] = ((0, 1.0), (1, 0.5), (1, 1.0))
MID_INDEX = 1  # POINT_FRACS[1] == (1, 0.5), the link-2 midpoint
TIP_INDEX = 2  # POINT_FRACS[2] == (1, 1.0), the link-2 tip


def _two_link_fk(theta: Tensor) -> tuple[Tensor, Tensor]:
    """theta (T,2) -> (points (T,U,3), jac (T,U,3,2)), U = len(POINT_FRACS). Closed-form, no MuJoCo."""
    t1, t2 = theta[:, 0], theta[:, 1]
    c1, s1 = torch.cos(t1), torch.sin(t1)
    c12, s12 = torch.cos(t1 + t2), torch.sin(t1 + t2)
    zero = torch.zeros_like(c1)
    pts, jacs = [], []
    for link, frac in POINT_FRACS:
        if link == 0:
            p = torch.stack([frac * L1 * c1, frac * L1 * s1, zero], dim=-1)
            d1 = torch.stack([-frac * L1 * s1, frac * L1 * c1, zero], dim=-1)
            d2 = torch.zeros_like(d1)
        else:
            p = torch.stack([L1 * c1 + frac * L2 * c12, L1 * s1 + frac * L2 * s12, zero], dim=-1)
            d1 = torch.stack([-L1 * s1 - frac * L2 * s12, L1 * c1 + frac * L2 * c12, zero], dim=-1)
            d2 = torch.stack([-frac * L2 * s12, frac * L2 * c12, zero], dim=-1)
        pts.append(p)
        jacs.append(torch.stack([d1, d2], dim=-1))
    return torch.stack(pts, dim=1), torch.stack(jacs, dim=1)


def test_canonical_diameter_matches_bounding_box_diagonal() -> None:
    pts = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 1.0, 0.0]], dtype=DT)
    assert math.isclose(canonical_diameter(pts), 5.0, rel_tol=1e-9)


def test_robot_canonical_points_identity_pose_is_just_world_over_scale() -> None:
    T, U = 2, 3
    world = torch.randn(T, U, 3, dtype=DT)
    jac = torch.randn(T, U, 3, 4, dtype=DT)
    q = torch.zeros(T, 35, dtype=DT)
    q[:, 31] = 1.0   # identity wxyz quaternion, zero object translation
    scale = 2.0
    points, canon_jac = robot_canonical_points(q, world, jac, mesh_scale=scale)
    assert torch.allclose(points, world / scale, atol=1e-12)
    assert torch.allclose(canon_jac, jac / scale, atol=1e-12)


def test_contact_transport_term_drops_low_mass_robot_points() -> None:
    """A robot point transported essentially zero mass gets residual=0 and weight=0, not NaN."""
    T, V, U = 1, 1, 2
    human = ContactMeasure(
        points=torch.zeros(T, V, 3, dtype=DT),
        raw_points=torch.zeros(T, V, 3, dtype=DT),
        weights=torch.ones(T, V, dtype=DT),
        labels=torch.zeros(V, dtype=torch.int8),
        sample_idx=torch.zeros(T, V, dtype=torch.int64),
        sdf_values=torch.zeros(T, V, dtype=DT),
    )
    # robot point 0 sits at the human atom; point 1 is far away and should get ~0 transported mass
    # once eps is small (entropic OT still smears a little mass everywhere, but well under the
    # 1e-6/U cutoff for a point this far given a tight eps).
    robot_points = torch.tensor([[[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]]], dtype=DT)
    robot_jac = torch.zeros(T, U, 3, 2, dtype=DT)
    robot_labels = torch.zeros(U, dtype=torch.int8)

    term, plan = contact_transport_term(human, robot_points, robot_jac, robot_labels,
                                        eps=1e-4, rho_h=1.0, rho_r=1e-2, lambda_label=0.0, n_sinkhorn=200)
    assert plan.shape == (T, V, U)
    assert torch.isfinite(term.residual).all()
    # far point's 3 residual components are exactly zero-weighted
    assert torch.equal(term.weight[0, 3:6], torch.zeros(3, dtype=DT))
    assert torch.equal(term.residual[0, 3:6], torch.zeros(3, dtype=DT))
    # near point should have been assigned essentially all the mass and near-zero residual
    assert term.weight[0, 0] > 0.0
    assert term.residual[0, :3].norm() < 1e-3


def test_contact_transport_term_contact_margin_offsets_target_along_surface_normal() -> None:
    """With a single human atom (all plan mass, one robot point), the fixed-plan target
    `pbar_u` should shift by exactly `contact_margin * normal` -- checked against the closed-form
    residual, not just "it changed"."""
    T, V, U = 1, 1, 1
    human = ContactMeasure(
        points=torch.zeros(T, V, 3, dtype=DT),
        raw_points=torch.zeros(T, V, 3, dtype=DT),
        weights=torch.ones(T, V, dtype=DT),
        labels=torch.zeros(V, dtype=torch.int8),
        sample_idx=torch.zeros(T, V, dtype=torch.int64),   # -> surface_normals[0]
        sdf_values=torch.zeros(T, V, dtype=DT),
    )
    surface_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=DT)   # (M=1,3)
    robot_points = torch.zeros(T, U, 3, dtype=DT)                 # robot sits exactly at the atom
    robot_jac = torch.zeros(T, U, 3, 2, dtype=DT)
    robot_labels = torch.zeros(U, dtype=torch.int8)
    margin = 0.1

    term, plan = contact_transport_term(
        human, robot_points, robot_jac, robot_labels, eps=1e-4, rho_h=1.0, rho_r=1e-2,
        lambda_label=0.0, n_sinkhorn=200, contact_margin=margin, surface_normals=surface_normals,
        sdf=_unit_box_sdf(),
    )
    # Only one atom/one point: w_u == mass(human) == 1, so scale_w = sqrt(U*w_u) = 1; pbar = (0,0,0)
    # before the offset, (0,0,margin) after -- residual = robot_points - pbar_offset = -(0,0,margin).
    expected = torch.tensor([0.0, 0.0, -margin], dtype=DT)
    assert torch.allclose(term.residual[0].reshape(3), expected, atol=1e-3)
    assert plan.shape == (T, V, U)


def test_contact_transport_term_measures_the_ot_cost_to_the_offset_surface() -> None:
    """With `contact_margin != 0` the plan is solved on the cost to the offset atoms
    `points + margin * normal(sample)`, the band the targets live in, not to the surface itself."""
    gen = torch.Generator().manual_seed(8)
    T, V, U, M = 2, 6, 5, 4
    human = ContactMeasure(
        points=torch.randn(T, V, 3, generator=gen, dtype=DT) * 0.3,
        raw_points=torch.zeros(T, V, 3, dtype=DT),
        weights=torch.rand(T, V, generator=gen, dtype=DT),
        labels=torch.tensor([0, 1, 1, 2, 0, 1], dtype=torch.int8),
        sample_idx=torch.randint(0, M, (T, V), generator=gen),
        sdf_values=torch.zeros(T, V, dtype=DT),
    )
    normals = torch.randn(M, 3, generator=gen, dtype=DT)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    robot_points = torch.randn(T, U, 3, generator=gen, dtype=DT) * 0.3
    robot_labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.int8)
    margin, eps, rho_h, rho_r, lam_label = 0.15, 1e-2, 1.0, 1e-2, 0.05

    _, plan = contact_transport_term(human, robot_points, torch.zeros(T, U, 3, 2, dtype=DT), robot_labels,
                                     eps=eps, rho_h=rho_h, rho_r=rho_r, lambda_label=lam_label, n_sinkhorn=80,
                                     contact_margin=margin, surface_normals=normals, sdf=_unit_box_sdf())
    offset_atoms = human.points + margin * normals[human.sample_idx]
    cost = torch.cdist(offset_atoms, robot_points, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    cost = cost + lam_label * (human.labels[:, None].long() != robot_labels[None, :].long()).to(DT)[None]
    expected = sinkhorn_unbalanced(cost, human.weights, torch.full((T, U), 1.0 / U, dtype=DT), eps, rho_h,
                                   rho_r, 80)
    assert torch.allclose(plan, expected, rtol=1e-12, atol=1e-15)
    _, plan_surface = contact_transport_term(human, robot_points, torch.zeros(T, U, 3, 2, dtype=DT),
                                             robot_labels, eps=eps, rho_h=rho_h, rho_r=rho_r,
                                             lambda_label=lam_label, n_sinkhorn=80)
    assert not torch.allclose(plan, plan_surface)


def test_contact_transport_term_zero_margin_matches_no_margin_default() -> None:
    """contact_margin=0.0 (explicit) must reproduce the exact same residual as omitting it."""
    torch.manual_seed(3)
    T, V, U = 2, 5, 4
    human = ContactMeasure(
        points=torch.randn(T, V, 3, dtype=DT) * 0.3,
        raw_points=torch.randn(T, V, 3, dtype=DT) * 0.3,
        weights=torch.rand(T, V, dtype=DT),
        labels=torch.zeros(V, dtype=torch.int8),
        sample_idx=torch.randint(0, 3, (T, V), dtype=torch.int64),
        sdf_values=torch.zeros(T, V, dtype=DT),
    )
    robot_points = torch.randn(T, U, 3, dtype=DT) * 0.3
    robot_jac = torch.randn(T, U, 3, 2, dtype=DT)
    robot_labels = torch.zeros(U, dtype=torch.int8)

    term_a, _ = contact_transport_term(human, robot_points, robot_jac, robot_labels, eps=1e-2,
                                       rho_h=1.0, rho_r=1e-2, lambda_label=0.0, n_sinkhorn=100)
    term_b, _ = contact_transport_term(human, robot_points, robot_jac, robot_labels, eps=1e-2,
                                       rho_h=1.0, rho_r=1e-2, lambda_label=0.0, n_sinkhorn=100,
                                       contact_margin=0.0)
    assert torch.equal(term_a.residual, term_b.residual)
    assert torch.equal(term_a.jacobian, term_b.jacobian)


def test_contact_transport_term_raises_without_normals_or_sdf_when_margin_nonzero() -> None:
    T, V, U = 1, 1, 1
    human = ContactMeasure(
        points=torch.zeros(T, V, 3, dtype=DT), raw_points=torch.zeros(T, V, 3, dtype=DT),
        weights=torch.ones(T, V, dtype=DT), labels=torch.zeros(V, dtype=torch.int8),
        sample_idx=torch.zeros(T, V, dtype=torch.int64), sdf_values=torch.zeros(T, V, dtype=DT),
    )
    robot_points = torch.zeros(T, U, 3, dtype=DT)
    robot_jac = torch.zeros(T, U, 3, 2, dtype=DT)
    robot_labels = torch.zeros(U, dtype=torch.int8)
    try:
        contact_transport_term(human, robot_points, robot_jac, robot_labels, eps=1e-4, rho_h=1.0,
                               rho_r=1e-2, lambda_label=0.0, contact_margin=0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when contact_margin != 0 without surface_normals/sdf")


def test_contact_transport_term_falls_back_to_sdf_gradient_when_normals_cancel() -> None:
    """Two atoms with opposite normals, both fully explained by the same robot point: the
    plan-weighted mean normal is ~0 (degenerate), so the offset must fall back to the SDF gradient
    direction at pbar (here: straight up, `+z`, since pbar sits directly below the box's centre)."""
    T, V, U = 1, 2, 1
    human = ContactMeasure(
        points=torch.zeros(T, V, 3, dtype=DT),               # both atoms at the same canonical point
        raw_points=torch.zeros(T, V, 3, dtype=DT),
        weights=torch.ones(T, V, dtype=DT),
        labels=torch.zeros(V, dtype=torch.int8),
        sample_idx=torch.tensor([[0, 1]], dtype=torch.int64),
        sdf_values=torch.zeros(T, V, dtype=DT),
    )
    surface_normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=DT)  # cancel exactly
    robot_points = torch.zeros(T, U, 3, dtype=DT)
    robot_jac = torch.zeros(T, U, 3, 2, dtype=DT)
    robot_labels = torch.zeros(U, dtype=torch.int8)
    margin = 0.2

    term, plan = contact_transport_term(
        human, robot_points, robot_jac, robot_labels, eps=1e-4, rho_h=1.0, rho_r=1e-2,
        lambda_label=0.0, n_sinkhorn=200, contact_margin=margin, surface_normals=surface_normals,
        sdf=_unit_box_sdf(),
    )
    # pbar = (0,0,0) is the box's centre; both atoms' normals cancel (degenerate), so the offset must
    # come entirely from the SDF-gradient fallback -- a *unit* vector times `margin`. The residual
    # also carries the usual sqrt(U*w_u) mass scaling (both atoms have zero-cost correspondence here,
    # so most/all of their mass is transported to the single robot point); check against that scale
    # factor computed from the plan actually returned, rather than assuming an exact w_u.
    assert torch.isfinite(term.residual).all()
    w_u = float(plan.sum(dim=1)[0, 0])
    expected_norm = math.sqrt(1 * w_u) * margin
    assert math.isclose(float(term.residual[0].reshape(3).norm()), expected_norm, rel_tol=1e-2)


def test_synthetic_two_link_hand_reaches_target_contact_via_transport() -> None:
    """A fake 2-link hand with analytic FK reaches a target contact measure (< 1mm).

    Two human contact atoms (link-2 mid + tip, both at the ground-truth pose) give the 2-DOF system
    a well-posed correspondence problem, exactly mirroring how a real hand's contact patch carries
    many atoms rather than one -- annealing eps and refreshing the OT plan on every accepted step
    (the same schedule shape `fastdo.retarget.kinematic.retarget_kinematic` uses).
    """
    torch.manual_seed(0)
    theta_gt = torch.tensor([[0.9, -0.5]], dtype=DT)
    target_points, _ = _two_link_fk(theta_gt)
    human = ContactMeasure(
        points=target_points[:, [MID_INDEX, TIP_INDEX], :],
        raw_points=target_points[:, [MID_INDEX, TIP_INDEX], :],
        weights=torch.ones(1, 2, dtype=DT),
        labels=torch.zeros(2, dtype=torch.int8),
        sample_idx=torch.zeros(1, 2, dtype=torch.int64),
        sdf_values=torch.zeros(1, 2, dtype=DT),
    )
    robot_labels = torch.zeros(len(POINT_FRACS), dtype=torch.int8)
    d2 = canonical_diameter(target_points[0]) ** 2

    state: dict[str, Tensor | None] = {"plan": None}

    def make_evaluate(eps: float):
        def evaluate(theta: Tensor, refresh: bool):
            points, jac = _two_link_fk(theta)
            term, plan = contact_transport_term(
                human, points, jac, robot_labels, eps=eps, rho_h=1.0 * d2, rho_r=1e-2 * d2,
                lambda_label=0.0, n_sinkhorn=500, plan=None if refresh else state["plan"],
            )
            state["plan"] = plan
            return [term], None
        return evaluate

    x = torch.tensor([[0.1, 0.1]], dtype=DT)
    for eps in (5e-1 * d2, 1e-1 * d2, 2e-2 * d2, 5e-3 * d2, 1e-3 * d2, 2e-4 * d2, 4e-5 * d2):
        state["plan"] = None
        res = lm_optimize_blocks(x, make_evaluate(eps), BlockLMConfig(max_iter=40, rel_tol=1e-12))
        x = res.x

    final_points, _ = _two_link_fk(x)
    dist_to_target = (final_points[0, TIP_INDEX] - target_points[0, TIP_INDEX]).norm()
    assert dist_to_target < 1e-3, f"tip did not reach the target contact atom: {dist_to_target}"


def test_inband_transport_term_reproduces_the_all_rows_term_synthetic() -> None:
    """Only atoms with band weight > 0 enter Sinkhorn; the term matches the all-rows one."""
    from fastdo.retarget.transport import contact_transport_term_inband, inband_measure

    gen = torch.Generator().manual_seed(0)
    T, V, U, M = 6, 64, 96, 256
    human_points = torch.rand(T, V, 3, generator=gen, dtype=DT) - 0.5
    robot_points = torch.rand(T, U, 3, generator=gen, dtype=DT) - 0.5
    robot_jac = torch.randn(T, U, 3, 28, generator=gen, dtype=DT)
    weights = torch.exp(-2.0 * torch.rand(T, V, generator=gen, dtype=DT))
    weights = torch.where(torch.rand(T, V, generator=gen, dtype=DT) < 0.8, torch.zeros_like(weights), weights)
    weights[3] = 0.0
    human_labels = torch.randint(0, 6, (V,), generator=gen).to(torch.int8)
    robot_labels = torch.randint(0, 6, (U,), generator=gen).to(torch.int8)
    sample_idx = torch.randint(0, M, (T, V), generator=gen)
    normals = torch.randn(M, 3, generator=gen, dtype=DT)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    human = ContactMeasure(points=human_points, raw_points=human_points, weights=weights, labels=human_labels,
                           sample_idx=sample_idx, sdf_values=torch.zeros(T, V, dtype=DT))
    margin = 0.003 / 0.18086324855404615
    measure = inband_measure(human, normals)
    for eps in (0.055, 0.0055):
        ref, _ = contact_transport_term(human, robot_points, robot_jac, robot_labels, eps=eps, rho_h=1.1, rho_r=0.011,
                                        lambda_label=0.05, n_sinkhorn=25, contact_margin=margin,
                                        surface_normals=normals, sdf=_unit_box_sdf())
        got, plan = contact_transport_term_inband(measure, robot_points, robot_jac, robot_labels, eps, 1.1, 0.011,
                                                  0.05, 25, None, margin, _unit_box_sdf())
        assert float((got.residual - ref.residual).abs().max()) <= 1e-10
        assert torch.equal(got.weight, ref.weight)
        assert float((got.jacobian - ref.jacobian).abs().max()) <= 1e-10
        assert torch.equal(got.weight[3], torch.zeros_like(got.weight[3]))
        assert plan.shape == (6, 15, 96)
        trial, same_plan = contact_transport_term_inband(measure, robot_points, None, robot_labels, eps, 1.1, 0.011,
                                                         0.05, 25, plan, margin, _unit_box_sdf())
        assert same_plan is plan and torch.equal(trial.residual, got.residual) and not trial.jacobian.any()


def test_inband_measure_gathers_by_weight() -> None:
    from fastdo.retarget.transport import inband_measure

    T, V = 2, 5
    weights = torch.tensor([[0.0, 0.5, 0.0, 0.2, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=DT)
    points = torch.arange(T * V * 3, dtype=DT).reshape(T, V, 3)
    sample_idx = torch.arange(T * V).reshape(T, V)
    normals = torch.randn(T * V, 3, generator=torch.Generator().manual_seed(1), dtype=DT)
    human = ContactMeasure(points=points, raw_points=points, weights=weights, labels=torch.arange(V).to(torch.int8),
                           sample_idx=sample_idx, sdf_values=torch.zeros(T, V, dtype=DT))
    m = inband_measure(human, normals)
    # V_b = max(1, max_t #{v : w_tv > 0}) = 3: frame 0 has three positive weights (0.5, 0.2, 0.9).
    assert m.index.shape == (2, 3) and m.index[0].tolist() == [4, 1, 3]
    assert m.valid.tolist() == [[True, True, True], [False, False, False]]
    assert m.nonempty.tolist() == [True, False]
    assert m.weights[0].tolist() == [0.9, 0.5, 0.2] and torch.equal(m.weights[1], torch.zeros(3, dtype=DT))
    rows = [4, 1, 3]
    assert torch.equal(m.points[0], points[0, rows]) and torch.equal(m.normals[0], normals[sample_idx[0, rows]])
    assert m.labels.dtype == torch.int8 and m.labels[0].tolist() == rows
