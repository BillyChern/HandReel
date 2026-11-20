"""Feature-token CPU tests 1-9: the term_blocks contract, frame blocks, the residual histogram and track tokens."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import apply_left
from fastdo.solver.banded import assemble_block_tridiagonal, term_blocks
from fastdo.solver.terms import TermOutput, pointmap_term
from fastdo.tracker.features import (
    frame_gradient_curvature,
    frame_normal_blocks,
    plan_row_mass,
    pointmap_residual_histogram,
    track_tokens,
)
from tests.tracker.synthetic_terms import make_terms_fixture, random_terms

F64 = torch.float64


@pytest.mark.parametrize("P", [7, 8])
def test_term_blocks_contract_matches_independent_oracle(P: int) -> None:
    t = random_terms(T=5, n=11, d=3, P=P, seed=1)
    wp = t.weight.clamp_min(0)
    H_ref = torch.einsum("tndp,tndq,tn->tpq", t.jacobian, t.jacobian, wp)
    g_ref = torch.einsum("tndp,tnd,tn->tp", t.jacobian, t.residual, wp)
    H, g = term_blocks(t)
    assert H.shape == (5, P, P) and g.shape == (5, P)
    assert H.dtype == F64 and g.dtype == F64
    torch.testing.assert_close(H, H_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(g, g_ref, rtol=1e-12, atol=1e-12)
    t32 = TermOutput(t.residual.float(), t.jacobian.float(), t.weight.float())
    H32, g32 = term_blocks(t32)
    assert H32.dtype == F64 and g32.dtype == F64


def test_term_blocks_matches_assemble_block_tridiagonal() -> None:
    fx = make_terms_fixture()
    Hk, gk = term_blocks(fx.pm)
    sys = assemble_block_tridiagonal([fx.pm], None, 6)
    torch.testing.assert_close(sys.D, Hk[:, :6, :6], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(sys.c, Hk[:, :6, 6], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(sys.g, gk[:, :6], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(sys.h_ss, Hk[:, 6, 6].sum(), rtol=1e-12, atol=0.0)
    torch.testing.assert_close(sys.g_s, gk[:, 6].sum(), rtol=1e-12, atol=0.0)
    H, g, n_obs = frame_normal_blocks([fx.pm])
    assert torch.equal(H, Hk) and torch.equal(g, gk)
    assert torch.equal(n_obs, torch.full((6,), 96.0, dtype=F64))


def test_frame_gradient_curvature_is_per_observation() -> None:
    fx = make_terms_fixture()
    pm = fx.pm
    pm2 = TermOutput(torch.cat([pm.residual, pm.residual], 1), torch.cat([pm.jacobian, pm.jacobian], 1),
                     torch.cat([pm.weight, pm.weight], 1))
    for a, b in zip(frame_gradient_curvature([pm2]), frame_gradient_curvature([pm]), strict=True):
        assert (a - b).abs().max() <= 1e-12 * b.abs().max()
    n_obs = frame_normal_blocks([pm, fx.tr])[2]
    expected = 96.0 + (fx.tr.weight > 0).sum(1).to(F64)
    assert torch.equal(n_obs, expected)
    assert torch.equal(n_obs, 96.0 + (fx.tr_raw.weight > 0).sum(1).to(F64))


def _dense_gauss_newton_step(H: Tensor, g: Tensor) -> Tensor:
    """Undamped dense (6T+1) Gauss-Newton step with one shared log-scale, from per-frame (T,7,7)/(T,7) blocks."""
    T = H.shape[0]
    Hd = torch.zeros(6 * T + 1, 6 * T + 1, dtype=F64)
    for k in range(T):
        Hd[6 * k:6 * k + 6, 6 * k:6 * k + 6] = H[k, :6, :6]
        Hd[6 * k:6 * k + 6, -1] = H[k, :6, 6]
        Hd[-1, 6 * k:6 * k + 6] = H[k, :6, 6]
    Hd[-1, -1] = H[:, 6, 6].sum()
    gd = torch.cat([g[:, :6].flatten(), g[:, 6].sum().reshape(1)])
    return torch.linalg.solve(Hd, -gd)


def test_frame_gradient_curvature_zero_at_fixed_plan_optimum() -> None:
    fx = make_terms_fixture(T=4)
    xi = torch.tensor([0.01, -0.01, 0.005, 0.02, -0.01, 0.015], dtype=F64).expand(4, 6)
    p0, ls0 = apply_left(fx.poses, xi), fx.log_scale

    def pm_at(p: Tensor, ls: Tensor) -> TermOutput:
        return pointmap_term(fx.X, fx.X_valid, fx.Y, p, ls, fx.eps, fx.rho_x, fx.rho_y, 0, plan=fx.plan)[0]

    p, ls = p0, ls0
    for _ in range(6):
        H, g, _n = frame_normal_blocks([pm_at(p, ls)])
        d = _dense_gauss_newton_step(H, g)
        p = apply_left(p, d[:-1].view(4, 6))
        ls = ls + d[-1]
    g0 = frame_gradient_curvature([pm_at(p0, ls0)])[0]
    g1 = frame_gradient_curvature([pm_at(p, ls)])[0]
    assert g1[:, :6].abs().max() <= 1e-6 * g0[:, :6].abs().max()
    assert g1[:, 6].sum().abs() <= 1e-6 * g0[:, 6].sum().abs()


def test_frame_normal_blocks_frame_arguments() -> None:
    fx = make_terms_fixture()
    L = fx.log_scale.exp() * fx.template_diameter
    with pytest.raises(ValueError):
        frame_normal_blocks([fx.pm], poses=fx.poses)
    with pytest.raises(ValueError):
        frame_normal_blocks([fx.pm], length_scale=L)
    with pytest.raises(ValueError):
        frame_normal_blocks([fx.pm], metric_residual=False)
    with pytest.raises(ValueError):
        frame_normal_blocks([random_terms(3, 4, 3, P=5, seed=0)], torch.eye(4, dtype=F64).expand(3, 4, 4), 1.0)
    with pytest.raises(ValueError):
        frame_normal_blocks([random_terms(3, 4, 3, P=7, seed=0), random_terms(4, 4, 3, P=7, seed=0)])


@pytest.mark.parametrize("knn", [None, 16])
def test_pointmap_residual_histogram_sums_to_valid_count(knn: int | None) -> None:
    fx = make_terms_fixture(occluded_frame=2, knn=knn)
    L = fx.log_scale.exp() * fx.template_diameter
    hist = pointmap_residual_histogram(fx.pm, plan_row_mass(fx.plan), L)
    assert hist.shape == (6, 16) and hist.dtype == F64
    assert torch.equal(hist.sum(1), (fx.pm.weight > 0).sum(1).to(F64))
    assert torch.equal(hist[2], torch.zeros(16, dtype=F64))


def test_pointmap_residual_histogram_bins() -> None:
    x = [5e-4, 1e-3, 2e-3, 0.03, 1.0, 3.0]
    residual = torch.zeros(1, 8, 3, dtype=F64)
    for i, xi in enumerate(x):
        residual[0, i, 0] = xi * 0.25
    residual[0, 6:, 0] = 0.1
    term = TermOutput(residual, torch.zeros(1, 8, 3, 7, dtype=F64),
                      torch.tensor([[1.0] * 7 + [0.0]], dtype=F64))
    row_mass = torch.tensor([[1 / 8] * 6 + [0.0, 1 / 8]], dtype=F64)
    hist = pointmap_residual_histogram(term, row_mass, 0.25)
    expected = torch.zeros(1, 16, dtype=F64)
    for xi in x:
        expected[0, sum(1 for j in range(15) if 10 ** (-3 + 3 * j / 14) < xi)] += 1.0
    expected[0, 15] += 1.0                                   # row 6: valid but unmatched -> +inf -> last bin
    assert [sum(1 for j in range(15) if 10 ** (-3 + 3 * j / 14) < xi) for xi in (5e-4, 1e-3, 1.0, 3.0)] == \
        [0, 0, 14, 15]
    assert torch.equal(hist, expected)
    assert hist.sum().eq(7.0).all()


def test_pointmap_residual_histogram_rejects_bad_shapes() -> None:
    fx = make_terms_fixture()
    rm = plan_row_mass(fx.plan)
    T, N = fx.pm.weight.shape
    p2p = TermOutput(torch.zeros(T, N, 1, dtype=F64), torch.zeros(T, N, 1, 7, dtype=F64), fx.pm.weight)
    with pytest.raises(ValueError):
        pointmap_residual_histogram(p2p, rm, 1.0)
    with pytest.raises(ValueError):
        pointmap_residual_histogram(fx.pm, torch.ones(T, N + 1, dtype=F64), 1.0)
    with pytest.raises(ValueError):
        pointmap_residual_histogram(fx.pm, rm, 1.0, n_bins=1)


def test_track_tokens_visibility_matches_input() -> None:
    fx = make_terms_fixture()
    tr = fx.tr_raw
    tok = track_tokens(fx.anchors, tr, fx.template_diameter)
    assert tok.shape == (6, 16, 6) and tok.dtype == F64
    assert torch.equal(tok[..., 5], (tr.weight > 0).to(F64))
    assert torch.equal(tok[..., 4], tr.weight)
    vis = tr.weight > 0
    assert torch.equal(tok[..., 3][~vis], torch.zeros_like(tok[..., 3][~vis]))
    assert torch.equal(tok[..., 3][vis], torch.linalg.vector_norm(tr.residual, dim=-1)[vis])
    assert torch.equal(tok[..., :3], (fx.anchors / fx.template_diameter).expand(6, 16, 3))
    assert torch.equal(track_tokens(fx.anchors.expand(6, 16, 3), tr, fx.template_diameter), tok)
    bad = TermOutput(torch.zeros(6, 16, 3, dtype=F64), torch.zeros(6, 16, 3, 7, dtype=F64), tr.weight)
    with pytest.raises(ValueError):
        track_tokens(fx.anchors, bad, fx.template_diameter)
