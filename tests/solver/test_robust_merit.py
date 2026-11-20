"""LM's acceptance merit must be able to match the robust loss its step descends.

The track term's IRLS weight `w = 1/(1 + r²/c²)` is EXACTLY `ρ′(s)` (s = r²) for the Cauchy loss
`ρ(s) = c²·log(1 + s/c²)` -- the normalisation with `ρ ≈ s` near zero, i.e. on the same scale as the least-squares
terms it is summed with. So the weighted Gauss-Newton step is majorize-minimize for Cauchy. But `clip_costs` accepts on
`Σ w·s` with `w` re-evaluated at the trial point, which is `Σ c²·s/(c² + s)`: the Geman-McClure loss.

`robust_merit="cauchy"` makes the merit the function the step descends. It must change ONLY step acceptance: the
residual, Jacobian and weight -- everything the Gauss-Newton system is built from -- are untouched, and the shipped
default ("irls") is byte-for-byte what it was.
"""
from __future__ import annotations

import math

import pytest
import torch

from fastdo.solver import lm as lm_module
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations, solve_trajectory
from fastdo.solver.banded import ClipLayout
from fastdo.solver.lm import clip_costs
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip
from fastdo.solver.terms import TermOutput, track_term

C = 4.0   # the shipped `SolverConfig.cauchy_c`


def _track_fixture(offset_px: float, n: int = 12, frames: int = 3, seed: int = 0):
    """Anchors one metre down the optical axis, with tracks displaced `offset_px` from their projections."""
    g = torch.Generator().manual_seed(seed)
    anchors = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.2
    poses = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
    poses[:, 2, 3] = 1.0
    log_scale = torch.zeros(frames, dtype=torch.float64)
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    p = anchors[None] + poses[:, None, :3, 3]
    uv = torch.stack([K[0, 0] * p[..., 0] / p[..., 2] + K[0, 2], K[1, 1] * p[..., 1] / p[..., 2] + K[1, 2]], -1)
    direction = torch.nn.functional.normalize(torch.randn(frames, n, 2, generator=g, dtype=torch.float64), dim=-1)
    tracks = uv + offset_px * direction
    vis = torch.ones(frames, n, dtype=torch.bool)
    vis[0, 0] = False                                   # an occluded entry must cost exactly nothing
    return anchors, tracks, vis, poses, log_scale, K


def test_the_irls_weight_is_exactly_the_derivative_of_the_cauchy_merit() -> None:
    """The mathematical core: if this fails, the Cauchy merit is not the function the step descends."""
    s = torch.linspace(0.0, 400.0, 81, dtype=torch.float64, requires_grad=True)
    rho = C * C * torch.log1p(s / (C * C))
    (grad,) = torch.autograd.grad(rho.sum(), s)
    assert torch.allclose(grad, 1.0 / (1.0 + s.detach() / (C * C)), rtol=0, atol=1e-12)


def test_the_cauchy_merit_is_c2_log1p_of_the_same_residuals() -> None:
    anchors, tracks, vis, poses, ls, K = _track_fixture(offset_px=6.0)
    out = track_term(anchors, tracks, vis, poses, ls, K, C, cauchy_merit=True)
    assert out.merit is not None and out.merit.shape == (poses.shape[0],)
    r2 = (out.residual**2).sum(-1)
    expected = (vis.to(r2.dtype) * C * C * torch.log1p(r2 / (C * C))).sum(-1)
    assert torch.allclose(out.merit, expected, rtol=0, atol=1e-12)
    assert float(out.residual[0, 0].abs().sum()) == 0.0     # the occluded entry contributes nothing


def test_the_step_is_unchanged_by_the_flag() -> None:
    """Only ACCEPTANCE may change. Residual, Jacobian and weight feed the Gauss-Newton system; they must be identical."""
    args = _track_fixture(offset_px=25.0)
    shipped, cauchy = track_term(*args, C), track_term(*args, C, cauchy_merit=True)
    assert torch.equal(shipped.residual, cauchy.residual)
    assert torch.equal(shipped.jacobian, cauchy.jacobian)
    assert torch.equal(shipped.weight, cauchy.weight)
    assert shipped.merit is None, "the shipped default must not compute a merit override"


def test_the_two_merits_agree_near_zero_and_part_on_outliers() -> None:
    """Prediction 1's mechanism (they agree where residuals are small) and the defect (they part on outliers)."""
    for offset, agree in ((0.05, True), (37.5, False)):
        out = track_term(*_track_fixture(offset_px=offset), C, cauchy_merit=True)
        gm = (out.weight * (out.residual**2).sum(-1)).sum(-1)
        assert out.merit is not None
        rel = float(((out.merit - gm).abs() / gm.clamp_min(1e-300)).max())
        if agree:
            assert rel < 1e-3, f"at {offset} px the losses should agree to second order, differ by {rel:.2e}"
        else:
            # A dead track at 37.5 px: Geman-McClure saturates near c² = 16, Cauchy c² log(1 + 88) ≈ 72.
            assert rel > 3.0, f"at {offset} px the losses should part by >4x, differ by only {rel:.2f}"


def test_clip_costs_replaces_the_weighted_sum_with_the_merit_and_still_adds_extra() -> None:
    residual = torch.ones(2, 3, 2, dtype=torch.float64)
    weight = torch.full((2, 3), 0.5, dtype=torch.float64)
    jac = torch.zeros(2, 3, 2, 7, dtype=torch.float64)
    layout = ClipLayout.build((2,), device="cpu")
    shipped = clip_costs([TermOutput(residual, jac, weight)], None, layout)
    assert float(shipped[0]) == pytest.approx(2 * 3 * 0.5 * 2.0)            # Σ w·‖r‖² = 6
    merit = torch.tensor([7.0, 11.0], dtype=torch.float64)
    extra = torch.tensor([1.0, 2.0], dtype=torch.float64)
    replaced = clip_costs([TermOutput(residual, jac, weight, extra=extra, merit=merit)], None, layout)
    assert float(replaced[0]) == pytest.approx(7.0 + 11.0 + 1.0 + 2.0)


def test_the_flag_reaches_the_merit_through_the_registered_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachability at the real entry point (`feedback-a-permission-is-not-a-capability`): with the flag set, the
    track term that `solve_trajectory` hands to `clip_costs` must carry a merit; with the default it must not."""
    clip = make_synthetic_clip(SyntheticConfig(n_frames=6, image_hw=(120, 160), focal=150.0, n_surface=128,
                                               n_tracks=16, shape="box"), seed=3)
    mesh = MeshSamples(clip.surface_points, clip.surface_normals)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    seen: list[bool] = []
    real = lm_module.clip_costs

    def spy(terms, *a, **k):                             # type: ignore[no-untyped-def]
        seen.extend(t.merit is not None for t in terms[1:])   # terms[0] is the pointmap term
        return real(terms, *a, **k)

    monkeypatch.setattr(lm_module, "clip_costs", spy)
    for flag, want in (("irls", False), ("cauchy", True)):
        seen.clear()
        cfg = SolverConfig(n_obs=64, n_mesh=128, device="cpu", iters_coarse=1, iters_mid=1, iters_fine=1,
                           robust_merit=flag)
        solve_trajectory(mesh, obs, clip.poses_gt, clip.scale_gt, cfg)
        assert seen, "clip_costs never saw a track term -- the fixture does not exercise the path"
        assert all(s == want for s in seen), f"robust_merit={flag!r}: merit present = {sorted(set(seen))}"


def test_an_unknown_robust_merit_is_refused() -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=3, image_hw=(80, 100), focal=100.0, n_surface=64,
                                               n_tracks=8, shape="box"), seed=4)
    mesh = MeshSamples(clip.surface_points, clip.surface_normals)
    obs = TrajectoryObservations(intrinsics=clip.intrinsics, masks=clip.masks, hand_masks=clip.hand_masks,
                                 pointmaps=clip.pointmaps, tracks=clip.tracks, track_vis=clip.track_vis,
                                 anchors=clip.anchors)
    cfg = SolverConfig(n_obs=32, n_mesh=64, device="cpu", robust_merit="huber")
    with pytest.raises(ValueError, match="robust_merit"):
        solve_trajectory(mesh, obs, clip.poses_gt, clip.scale_gt, cfg)


def test_the_constant_is_the_consistent_one() -> None:
    """Pins the normalisation that had to be corrected: c²·log1p(s/c²) ≈ s near zero, NOT s/2."""
    s = 1e-6
    assert C * C * math.log1p(s / (C * C)) == pytest.approx(s, rel=1e-6)
