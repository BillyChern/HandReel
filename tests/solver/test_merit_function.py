"""The merit function must not reward abandoning the data.

Under unbalanced OT a model far from the observations transports no mass, so `scale_w = sqrt(N*w) -> 0`
and the pointmap residual vanishes *however wrong the pose is* -- while `weight` stays at full validity.
Measured on 15 real clips: 14 detach somewhere along their gauge orbit, and on 14/14 the quantity LM
accepts steps on is 0.006-0.49x its value at the solve while the unbalanced functional the clip's own
Sinkhorn minimises is 1.8-21x WORSE.

`test_detaching_lowers_the_shipped_cost_and_raises_the_merit` pins BOTH halves, so it cannot pass for the
wrong reason: if someone "fixes" the defect by making the residual not vanish, the first assertion fails
and the test says so instead of quietly going green.
"""
from __future__ import annotations

import math

import pytest
import torch

from fastdo.solver.terms import TermOutput, generalised_kl, pointmap_term

# A cube is a stand-in for the object template; `rho`/`eps` are the shipped fine-level values already
# multiplied by diameter^2 the way `batch_eval` multiplies them by `scale2` (rho_x 0.1, rho_y 1e-3).
#
# The object's SIZE RELATIVE TO ITS DEPTH is load-bearing, not cosmetic. `rho` scales as diameter^2, so a
# 1 m cube at 1 m depth gets rho_x = 0.3 against a transport cost of only 0.64 -- and the plan keeps 21%
# of its mass instead of abandoning it, so the fixture never reaches the regime under test. A hand-held
# object is ~0.2 m at ~0.8 m, which is what the 15 measured clips look like and what this reproduces.
_EXTENT = 0.2
_DIAM2 = (_EXTENT * math.sqrt(3.0)) ** 2
EPS, RHO_X, RHO_Y, N_SINKHORN = 1e-4 * _DIAM2, 0.1 * _DIAM2, 1e-3 * _DIAM2, 200


def _clip(n_side: int = 6, frames: int = 2, noise: float = 0.004
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A cube template one metre down the optical axis, observed with a little noise.

    The noise matters: at an exact fit the residual is already 0 and detaching could not lower it, so a
    noiseless fixture would hide the very defect under test. A real solve always stops at a nonzero cost.
    """
    g = torch.linspace(-_EXTENT / 2, _EXTENT / 2, n_side, dtype=torch.float64)
    Y = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    poses = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
    poses[:, 2, 3] = 1.0
    log_scale = torch.zeros(frames, dtype=torch.float64)
    X = (Y + poses[0, :3, 3]).unsqueeze(0).repeat(frames, 1, 1)
    X = X + noise * torch.linspace(-1.0, 1.0, X.shape[1], dtype=torch.float64)[None, :, None]
    return X, torch.ones(X.shape[:2], dtype=torch.bool), Y, poses, log_scale


def _costs(X, X_valid, Y, poses, log_scale) -> tuple[float, float, float]:
    """`(what LM accepts on, the added deficiency, kept OT mass)` at this configuration."""
    pm, plan = pointmap_term(X, X_valid, Y, poses, log_scale, EPS, RHO_X, RHO_Y, N_SINKHORN,
                           with_deficiency=True)
    shipped = float((pm.weight * (pm.residual**2).sum(-1)).sum())
    assert pm.extra is not None, "pointmap_term must always compute the deficiency; the flag gates its USE"
    mass = float(plan.sum(-1).sum(-1).mean()) if isinstance(plan, torch.Tensor) else float("nan")
    return shipped, float(pm.extra.sum()), mass


def _along_gauge_orbit(poses: torch.Tensor, log_scale: torch.Tensor, lam: float):
    """`(t, s) -> (lambda t, lambda s)`: the orbit that leaves the image pointwise invariant."""
    moved = poses.clone()
    moved[:, :3, 3] *= lam
    return moved, log_scale + math.log(lam)


def test_detaching_lowers_the_shipped_cost_and_raises_the_merit() -> None:
    X, X_valid, Y, poses, log_scale = _clip()
    fit_shipped, fit_extra, fit_mass = _costs(X, X_valid, Y, poses, log_scale)
    det_poses, det_ls = _along_gauge_orbit(poses, log_scale, 0.2)
    det_shipped, det_extra, det_mass = _costs(X, X_valid, Y, det_poses, det_ls)

    # The model really has left the data behind, rather than merely moved a little.
    assert det_mass < 0.1 * fit_mass, f"not detached: mass {fit_mass:.4f} -> {det_mass:.4f}"

    # THE DEFECT, pinned: the quantity LM accepts steps on FALLS when the model abandons the data.
    assert det_shipped < fit_shipped, (
        f"expected the shipped cost to fall on detaching ({fit_shipped:.6g} -> {det_shipped:.6g}); if this "
        "now rises, the residual no longer vanishes with the row mass and this test needs rewriting")

    # THE FIX: with the deficiency added, detaching is correctly scored as much worse.
    assert det_shipped + det_extra > fit_shipped + fit_extra


def test_the_deficiency_is_negligible_where_the_plan_keeps_its_mass() -> None:
    """The fix must not perturb a solve that already fits: a balanced plan owes almost nothing."""
    X, X_valid, Y, poses, log_scale = _clip(noise=0.0)
    _, extra, _ = _costs(X, X_valid, Y, poses, log_scale)
    assert extra < 0.05 * RHO_X, f"a fitting clip should owe ~no deficiency, got {extra:.6g}"


def test_generalised_kl_is_zero_at_equality_and_saturates_at_total_abandonment() -> None:
    q = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64)
    assert float(generalised_kl(q, q)) == pytest.approx(0.0, abs=1e-12)
    # p -> 0 is the regime that matters: the limit is sum(q), NOT -inf, and no `log 0` is evaluated.
    assert float(generalised_kl(torch.zeros_like(q), q)) == pytest.approx(1.0)
    assert math.isfinite(float(generalised_kl(torch.zeros_like(q), torch.zeros_like(q))))


def test_total_cost_adds_extra_only_when_present() -> None:
    from fastdo.solver.lm import total_cost

    r = torch.zeros(3, 4, 3, dtype=torch.float64)
    w = torch.ones(3, 4, dtype=torch.float64)
    j = torch.zeros(3, 4, 3, 7, dtype=torch.float64)
    base = TermOutput(residual=r, jacobian=j, weight=w)
    with_extra = TermOutput(residual=r, jacobian=j, weight=w,
                            extra=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert float(total_cost([base], None)) == pytest.approx(0.0)
    assert float(total_cost([with_extra], None)) == pytest.approx(6.0)


def test_the_gauss_newton_system_never_sees_extra() -> None:
    """The load-bearing claim of the unbalanced merit: only the ACCEPTANCE TEST changes, never the step direction.

    By the envelope theorem the shipped Jacobian is already the gradient of the full unbalanced
    functional, so the fix must not touch it. Enforced here rather than asserted in a docstring: if
    `term_blocks` ever starts reading `extra`, this fails.
    """
    from fastdo.solver.banded import term_blocks

    X, X_valid, Y, poses, log_scale = _clip()
    pm, _ = pointmap_term(X, X_valid, Y, poses, log_scale, EPS, RHO_X, RHO_Y, N_SINKHORN,
                           with_deficiency=True)
    assert pm.extra is not None and float(pm.extra.abs().sum()) > 0.0, "need a nonzero extra to test"
    H_with, g_with = term_blocks(pm)
    H_without, g_without = term_blocks(TermOutput(pm.residual, pm.jacobian, pm.weight))
    assert torch.equal(H_with, H_without)
    assert torch.equal(g_with, g_without)


def test_the_shipped_default_computes_no_deficiency_for_any_caller() -> None:
    """The gate lives at the SOURCE. A first version gated it in one caller (`batch_eval.evaluate`), and
    `test_lm`, which calls `pointmap_term` directly, silently got the new merit and failed -- a default
    that one caller can bypass is not a default."""
    X, X_valid, Y, poses, log_scale = _clip()
    pm, _ = pointmap_term(X, X_valid, Y, poses, log_scale, EPS, RHO_X, RHO_Y, N_SINKHORN)
    assert pm.extra is None


def test_balanced_ot_owes_exactly_zero_rather_than_inf_or_nan() -> None:
    """`rho = inf` is balanced OT, a legitimate setting (`test_lm` uses it): no mass can be abandoned, so
    the deficiency is structurally 0. Naively `inf * (Sinkhorn's finite-iteration marginal error)`."""
    X, X_valid, Y, poses, log_scale = _clip()
    pm, _ = pointmap_term(X, X_valid, Y, poses, log_scale, EPS, float("inf"), float("inf"), N_SINKHORN,
                          with_deficiency=True)
    assert pm.extra is not None
    assert torch.equal(pm.extra, torch.zeros_like(pm.extra))
