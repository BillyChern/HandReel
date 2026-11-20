"""Tests for `fastdo.retarget.kinematic_terms`: penetration/joint-limit/wrist-prior/smoothness terms.

Every analytic Jacobian here (penetration, joint limits) gets a central-finite-difference check;
`wrist_prior_term`'s Jacobian *is* a central finite difference (by design),
so its test instead cross-checks the function's own FD against an independently-coded FD in the test
at a different step size, plus the structural fact that hand-dof columns must be exactly zero.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from torch import Tensor

from fastdo.geometry.lie import se3_inv, se3_log
from fastdo.retarget.kinematic_terms import (
    contact_band_term,
    human_wrist_poses,
    joint_limit_term,
    penetration_term,
    primitive_penetration_term,
    robot_joint_limits,
    smoothness_term,
    wrist_prior_term,
    wrist_terminal_body,
)
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.primitives import CapsuleDistance, capsule_min_distance
from fastdo.solver.lm_blocks import BlockTerm
from fastdo.solver.sdf import mesh_sdf_grid

DT = torch.float64
REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "sharpa"
             / "right" / "whisking" / "0" / "scene.xml")
whisking_scene_missing = pytest.mark.skipif(not SCENE_XML.exists(), reason="baseline submodule scene.xml not checked out")


def _linear_fk(q: Tensor, A: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """q (T,n), A (U,3,n), b (U,3) -> (points (T,U,3), jac (T,U,3,n)) for a fixed *linear* map."""
    T = q.shape[0]
    points = torch.einsum("uij,tj->tui", A, q) + b[None]
    jac = A[None].expand(T, -1, -1, -1)
    return points, jac


def test_penetration_term_jacobian_matches_central_finite_differences() -> None:
    torch.manual_seed(0)
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=40, padding=0.5)

    T, U, n = 2, 5, 4
    A = torch.randn(U, 3, n, dtype=DT) * 0.15
    b = torch.randn(U, 3, dtype=DT) * 0.2   # near the box centre: comfortably inside (phi < 0)
    q0 = torch.randn(T, n, dtype=DT) * 0.3

    def residual(q: Tensor) -> Tensor:
        points, jac = _linear_fk(q, A, b)
        return penetration_term(points, jac, sdf, lam_pen=2.0).residual

    points0, jac0 = _linear_fk(q0, A, b)
    term = penetration_term(points0, jac0, sdf, lam_pen=2.0)
    assert (points0.norm(dim=-1) < 0.5).any(), "test setup should place at least one point near/inside the box"

    h = 1e-5
    for k in range(n):
        qp, qm = q0.clone(), q0.clone()
        qp[:, k] += h
        qm[:, k] -= h
        fd = (residual(qp) - residual(qm)) / (2 * h)
        assert torch.allclose(term.jacobian[:, :, k], fd, atol=1e-4, rtol=1e-3), f"dof {k} mismatch"


def _sphere_phi(center: Tensor, radius: float) -> Callable[[Tensor], tuple[Tensor, Tensor]]:
    def phi(y: Tensor) -> tuple[Tensor, Tensor]:
        d = y - center
        n = d.norm(dim=-1)
        return n - radius, d / n.clamp_min(1e-15)[..., None]
    return phi


def test_primitive_penetration_term_jacobian_matches_central_finite_differences() -> None:
    """Envelope-theorem Jacobian (grad phi(p*) . ((1 - t*) J_a + t* J_b), t* fixed) vs central finite
    differences of the residual `sqrt(lam) relu(clearance - d_min(q))`, on random linear "kinematics"
    with capsules partly inside an analytic sphere SDF (exact search: many golden steps). Inactive
    capsules (d_min > clearance) must get a zero residual and a zero Jacobian row."""
    gen = torch.Generator().manual_seed(5)
    T, P, n = 3, 12, 5
    R, clearance, lam = 0.3, 0.01, 7.0
    A_a = torch.randn(P, 3, n, generator=gen, dtype=DT) * 0.05
    A_b = torch.randn(P, 3, n, generator=gen, dtype=DT) * 0.05
    direction = torch.randn(P, 3, generator=gen, dtype=DT)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    mid = direction * (R + 0.02 + 0.08 * torch.rand(P, 1, generator=gen, dtype=DT))
    tangent = torch.linalg.cross(direction, torch.randn(P, 3, generator=gen, dtype=DT))
    tangent = 0.1 * tangent / tangent.norm(dim=-1, keepdim=True)
    b_a, b_b = mid - tangent, mid + tangent
    radius = torch.full((P,), 0.05, dtype=DT)
    q0 = torch.randn(T, n, generator=gen, dtype=DT) * 0.2
    phi = _sphere_phi(torch.zeros(3, dtype=DT), R)

    def ends(q: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pa, ja = _linear_fk(q, A_a, b_a)
        pb, jb = _linear_fk(q, A_b, b_b)
        return pa, pb, ja, jb

    def residual(q: Tensor) -> Tensor:
        pa, pb, ja, jb = ends(q)
        term, _ = primitive_penetration_term(pa, pb, ja, jb, radius, phi, lam, clearance, n_golden=60)
        return term.residual

    term, dist = primitive_penetration_term(*ends(q0), radius, phi, lam, clearance, n_golden=60)
    active = dist.d_min < clearance
    assert active.any() and (~active).any() and (dist.d_min < 0).any(), "setup: need active, inactive, penetrating"
    assert torch.allclose(term.residual, lam ** 0.5 * torch.relu(clearance - dist.d_min))
    assert torch.equal(term.jacobian[~active], torch.zeros_like(term.jacobian[~active]))
    assert term.residual.shape == (T, P) and term.jacobian.shape == (T, P, n)

    h = 1e-6
    for k in range(n):
        qp, qm = q0.clone(), q0.clone()
        qp[:, k] += h
        qm[:, k] -= h
        fd = (residual(qp) - residual(qm)) / (2 * h)
        assert torch.allclose(term.jacobian[:, :, k], fd, rtol=1e-4, atol=1e-7), f"dof {k} mismatch"


def test_primitive_penetration_term_skips_the_jacobian_for_a_trial_evaluation() -> None:
    gen = torch.Generator().manual_seed(6)
    T, P, n = 2, 3, 4
    pa = torch.randn(T, P, 3, generator=gen, dtype=DT) * 0.2
    pb = pa + 0.05
    jac = torch.randn(T, P, 3, n, generator=gen, dtype=DT)
    radius = torch.full((P,), 0.05, dtype=DT)
    phi = _sphere_phi(torch.zeros(3, dtype=DT), 0.3)
    full, _ = primitive_penetration_term(pa, pb, jac, jac, radius, phi, 2.0, 0.01)
    trial, _ = primitive_penetration_term(pa, pb, jac, jac, radius, phi, 2.0, 0.01, need_jacobian=False)
    assert torch.equal(full.residual, trial.residual)
    assert torch.equal(trial.jacobian, torch.zeros(T, P, n, dtype=DT))


def test_contact_band_term_pulls_each_bodys_closest_cover_to_the_margin() -> None:
    """`sqrt(lam_band * W_b) * (d_min - margin)` on the closest cover of each body `b` whose robot
    samples carry transport mass `W_b` (sum of U * w_u over the body's samples); the body's other covers
    and mass-free bodies contribute nothing. Jacobian: envelope theorem, checked by central FD with the
    plan mass held fixed (the fixed-plan majorizer the LM uses)."""
    gen = torch.Generator().manual_seed(9)
    T, n, R, lam, margin = 3, 4, 0.3, 5.0, 0.02
    cover_bodies = np.array([10, 10, 11, 12])
    sample_bodies = np.array([10, 10, 11, 11, 12, 13])
    P = cover_bodies.size
    A_a = torch.randn(P, 3, n, generator=gen, dtype=DT) * 0.03
    A_b = torch.randn(P, 3, n, generator=gen, dtype=DT) * 0.03
    direction = torch.nn.functional.normalize(torch.randn(P, 3, generator=gen, dtype=DT), dim=-1)
    mid = direction * torch.tensor([[0.33], [0.45], [0.36], [0.40]], dtype=DT)   # cover 0 well below cover 1
    tangent = 0.08 * torch.nn.functional.normalize(torch.linalg.cross(direction, torch.randn(P, 3, dtype=DT,
                                                   generator=gen)), dim=-1)
    radius = torch.full((P,), 0.02, dtype=DT)
    mass = torch.rand(T, sample_bodies.size, generator=gen, dtype=DT) + 0.5
    mass[0, 4] = 0.0                                                        # body 12 carries no mass at t=0
    q0 = torch.randn(T, n, generator=gen, dtype=DT) * 0.1
    phi = _sphere_phi(torch.zeros(3, dtype=DT), R)

    def evaluate(q: Tensor, need_jacobian: bool = True) -> tuple[BlockTerm, CapsuleDistance]:
        pa, ja = _linear_fk(q, A_a, mid - tangent)
        pb, jb = _linear_fk(q, A_b, mid + tangent)
        dist = capsule_min_distance(phi, pa, pb, radius, n_golden=60)
        return contact_band_term(dist, ja, jb, cover_bodies, sample_bodies, mass, lam, margin,
                                 need_jacobian=need_jacobian), dist

    term, dist = evaluate(q0)
    body_mass = torch.stack([mass[:, :2].sum(1), mass[:, :2].sum(1), mass[:, 2:4].sum(1), mass[:, 4]], dim=1)
    closest = torch.tensor([True, False, True, True]).expand(T, P)
    assert torch.all(dist.d_min[:, 0] < dist.d_min[:, 1])
    expected = torch.sqrt(lam * body_mass) * (dist.d_min - margin) * closest
    assert torch.allclose(term.residual, expected)
    assert torch.equal(term.residual[:, 1], torch.zeros(T, dtype=DT)) and term.residual[0, 3] == 0.0
    trial, _ = evaluate(q0, need_jacobian=False)
    assert torch.equal(trial.residual, term.residual) and not trial.jacobian.any()

    h = 1e-6
    for k in range(n):
        qp, qm = q0.clone(), q0.clone()
        qp[:, k] += h
        qm[:, k] -= h
        fd = (evaluate(qp)[0].residual - evaluate(qm)[0].residual) / (2 * h)
        assert torch.allclose(term.jacobian[:, :, k], fd, rtol=1e-4, atol=1e-8), f"dof {k} mismatch"


def test_joint_limit_term_jacobian_matches_central_finite_differences() -> None:
    torch.manual_seed(1)
    n = 6
    q_min = torch.full((n,), -1.0, dtype=DT)
    q_max = torch.full((n,), 1.0, dtype=DT)
    limited = torch.ones(n, dtype=DT)
    # half comfortably in range, half comfortably out of range on either side (away from the kink at
    # exactly q_min/q_max, where relu's derivative is discontinuous).
    q0 = torch.tensor([[0.0, 1.5, -1.5, 0.5, -0.5, 2.0]], dtype=DT)

    def residual(q: Tensor) -> Tensor:
        return joint_limit_term(q, q_min, q_max, limited, lam_lim=3.0).residual

    term = joint_limit_term(q0, q_min, q_max, limited, lam_lim=3.0)
    h = 1e-6
    for k in range(n):
        qp, qm = q0.clone(), q0.clone()
        qp[:, k] += h
        qm[:, k] -= h
        fd = (residual(qp) - residual(qm)) / (2 * h)
        assert torch.allclose(term.jacobian[0, :, k], fd[0], atol=1e-6), f"dof {k} mismatch"


def test_joint_limit_term_zero_inside_range_and_unlimited_dof_contributes_nothing() -> None:
    q_min = torch.tensor([-1.0, -1.0, -1.0], dtype=DT)
    q_max = torch.tensor([1.0, 1.0, 1.0], dtype=DT)
    limited = torch.tensor([1.0, 1.0, 0.0], dtype=DT)   # dof 2 unlimited
    q = torch.tensor([[0.0, 5.0, 5.0]], dtype=DT)         # dof 1 violates, dof 2 would too if limited
    term = joint_limit_term(q, q_min, q_max, limited, lam_lim=1.0)
    assert term.residual[0, 0] == 0.0
    assert term.residual[0, 1] > 0.0
    # dof 2 is unlimited: excluded via weight=0 (the raw residual/jacobian values are irrelevant --
    # `weight` is what zeroes their contribution to the assembled cost/Hessian, per BlockTerm's own
    # convention; see the function's docstring for why zeroing only the Jacobian would be worse).
    assert term.weight[0, 2] == 0.0


def test_smoothness_term_accepts_per_dof_weights() -> None:
    """Wrist translation (metres) and joint angles (radians) need different weights, so
    `lam_q` may be a `(n,)` vector; each dof's residual and Jacobian scale by its own sqrt(lam)."""
    q = torch.tensor([[0.0, 0.0], [0.01, 0.2], [0.03, 0.1]], dtype=DT)
    lam = torch.tensor([1e5, 1e3], dtype=DT)
    term = smoothness_term(q, lam)
    assert torch.allclose(term.residual, lam.sqrt() * (q[1:] - q[:-1]))
    assert torch.allclose(term.jac_k[0], -torch.diag(lam.sqrt()))
    assert torch.allclose(term.jac_k1[1], torch.diag(lam.sqrt()))


def test_smoothness_term_matches_closed_form() -> None:
    q = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]], dtype=DT)
    lam_q = 4.0
    term = smoothness_term(q, lam_q)
    expected = (lam_q ** 0.5) * (q[1:] - q[:-1])
    assert torch.allclose(term.residual, expected)
    n = 2
    assert torch.allclose(term.jac_k[0], -(lam_q ** 0.5) * torch.eye(n, dtype=DT))
    assert torch.allclose(term.jac_k1[0], (lam_q ** 0.5) * torch.eye(n, dtype=DT))


def test_human_wrist_poses_gives_orthonormal_rotation_and_wrist_position() -> None:
    # A hand-built, unambiguous right-hand configuration: wrist at origin, middle MCP straight up
    # (+z), index MCP to the -x side, ring MCP to the +x side (so y_aux = index - ring points -x).
    joints = torch.zeros(1, 21, 3, dtype=DT)
    joints[0, 0] = torch.tensor([0.0, 0.0, 0.0])     # wrist
    joints[0, 9] = torch.tensor([0.0, 0.0, 1.0])     # middle MCP
    joints[0, 5] = torch.tensor([-1.0, 0.0, 0.5])    # index MCP
    joints[0, 13] = torch.tensor([1.0, 0.0, 0.5])    # ring MCP

    pose = human_wrist_poses(joints, is_right=True)
    assert pose.shape == (1, 4, 4)
    R = pose[0, :3, :3]
    assert torch.allclose(R.T @ R, torch.eye(3, dtype=DT), atol=1e-9)
    assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0, dtype=DT), atol=1e-9)
    assert torch.allclose(pose[0, :3, 3], joints[0, 0])
    # z axis should point from wrist toward middle MCP, i.e. +z here
    assert R[:, 2][2] > 0.9


@whisking_scene_missing
def test_robot_joint_limits_matches_model_jnt_range_for_the_whisking_scene() -> None:
    scene = RobotScene.from_xml(SCENE_XML)
    q_min, q_max, limited = robot_joint_limits(scene.model, scene.robot_dof)
    n = scene.robot_dof.stop - scene.robot_dof.start
    assert q_min.shape == (n,) and q_max.shape == (n,) and limited.shape == (n,)
    # Every one of the 28 Sharpa robot dofs is range-limited (checked directly against the compiled
    # model in development); the wrist slide joints range +-5, the thumb CMC flexion 6 is [-0.1745, 1.9199].
    assert torch.all(limited == 1.0)
    assert torch.allclose(q_min[:3], torch.full((3,), -5.0, dtype=DT))
    assert torch.allclose(q_max[:3], torch.full((3,), 5.0, dtype=DT))
    assert q_min[6] == pytest.approx(-0.1745, abs=1e-3)
    assert q_max[6] == pytest.approx(1.9199, abs=1e-3)


@whisking_scene_missing
def test_wrist_prior_term_hand_dof_columns_are_exactly_zero_and_fd_is_consistent() -> None:
    scene = RobotScene.from_xml(SCENE_XML)
    wrist_body = wrist_terminal_body(scene)
    assert wrist_body  # non-empty

    T = 2
    q = torch.zeros(T, scene.model.nq, dtype=DT)
    q[:, 31] = 1.0        # identity object quaternion (wxyz)
    torch.manual_seed(2)
    q[:, :6] = torch.randn(T, 6, dtype=DT) * 0.05   # small, physically reasonable wrist qpos

    fk = scene.fk(q.numpy())
    T_wrist = torch.from_numpy(np.asarray(fk[wrist_body])).to(DT)
    # An arbitrary but fixed target pose, offset from the current wrist pose.
    target = T_wrist.clone()
    target[:, 0, 3] += 0.02

    term = wrist_prior_term(q, scene, wrist_body, target, lam_wrist=2.0, fd_eps=1e-5)
    assert term.residual.shape == (T, 6)
    assert term.jacobian.shape == (T, 6, 28)
    assert torch.equal(term.jacobian[:, :, 6:], torch.zeros(T, 6, 22, dtype=DT))

    # Independently-coded FD at a different step size, as a consistency cross-check.
    def residual_fn(qq: np.ndarray) -> Tensor:
        fkq = scene.fk(qq)
        Tw = torch.from_numpy(np.asarray(fkq[wrist_body])).to(DT)
        return (2.0 ** 0.5) * se3_log(se3_inv(Tw) @ target)

    q_np = q.numpy()
    h = 2e-5
    jac_ref = torch.zeros(T, 6, 6, dtype=DT)
    for k in range(6):
        qp, qm = q_np.copy(), q_np.copy()
        qp[:, k] += h
        qm[:, k] -= h
        jac_ref[:, :, k] = (residual_fn(qp) - residual_fn(qm)) / (2 * h)
    assert torch.allclose(term.jacobian[:, :, :6], jac_ref, atol=1e-4, rtol=1e-2)


@whisking_scene_missing
def test_wrist_prior_closed_form_matches_the_scene_fk_term() -> None:
    """`Trans(q0..q2) Rx(q3) Ry(q4) Rz(q5)` with the 13 central-difference configurations in one
    batched call reproduces the `scene.fk` term."""
    from fastdo.retarget.batched_kinematics import wrist_qpos0
    from fastdo.retarget.kinematic_terms import wrist_prior_term_closed_form

    scene = RobotScene.from_xml(SCENE_XML)
    wrist_body = wrist_terminal_body(scene)
    rng = np.random.default_rng(5)
    q = np.zeros((10, 35))
    q[:, 31] = 1.0
    q[:, 0:3] = rng.uniform(-0.5, 0.5, (10, 3))
    q[:, 3:6] = rng.uniform(-np.pi, np.pi, (10, 3))
    q_t = q.copy()
    q_t[:, :6] += 0.1 * rng.normal(size=(10, 6))
    targets = torch.from_numpy(scene.fk(q_t)[wrist_body])
    ref = wrist_prior_term(torch.from_numpy(q), scene, wrist_body, targets, 100.0, 1e-5, need_jacobian=True)
    got = wrist_prior_term_closed_form(torch.from_numpy(q[:, :6]), targets, 100.0, 1e-5, need_jacobian=True,
                                       qpos0=wrist_qpos0(scene))
    assert float((got.residual - ref.residual).abs().max()) <= 1e-12
    assert float((got.jacobian - ref.jacobian).abs().max()) <= 1e-8
    assert torch.equal(got.jacobian[:, :, 6:], torch.zeros(10, 6, 22, dtype=DT))
    trial = wrist_prior_term_closed_form(torch.from_numpy(q[:, :6]), targets, 100.0, 1e-5, need_jacobian=False,
                                         qpos0=wrist_qpos0(scene))
    assert float((trial.residual - ref.residual).abs().max()) <= 1e-12
    assert trial.jacobian.shape == (10, 6, 28) and not trial.jacobian.any()


def test_primitive_term_with_a_given_distance_equals_the_searched_term() -> None:
    """Passing the already searched `CapsuleDistance` gives the identical term."""
    gen = torch.Generator().manual_seed(5)
    T, P, n = 3, 12, 5
    A_a = torch.randn(P, 3, n, generator=gen, dtype=DT) * 0.05
    A_b = torch.randn(P, 3, n, generator=gen, dtype=DT) * 0.05
    b_a = torch.randn(P, 3, generator=gen, dtype=DT) * 0.3
    b_b = b_a + 0.1 * torch.randn(P, 3, generator=gen, dtype=DT)
    radius = torch.full((P,), 0.05, dtype=DT)
    q0 = torch.randn(T, n, generator=gen, dtype=DT) * 0.2
    phi = _sphere_phi(torch.zeros(3, dtype=DT), 0.3)
    pa, ja = _linear_fk(q0, A_a, b_a)
    pb, jb = _linear_fk(q0, A_b, b_b)
    searched, dist = primitive_penetration_term(pa, pb, ja, jb, radius, phi, 7.0, 0.01)
    assert (dist.d_min < 0.01).any()
    given, same = primitive_penetration_term(pa, pb, ja, jb, radius, phi, 7.0, 0.01,
                                             distance=capsule_min_distance(phi, pa, pb, radius))
    assert torch.equal(searched.residual, given.residual) and torch.equal(searched.jacobian, given.jacobian)
    assert torch.equal(same.d_min, dist.d_min)
