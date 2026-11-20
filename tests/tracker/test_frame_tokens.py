"""Token CPU tests 10-15: canonical invariance, the 110-column token layout, occlusion, NaN locality, host reads."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import TermOutput, pointmap_term, track_term
from fastdo.tracker.clusters import mesh_cluster_ids, mesh_cluster_mass
from fastdo.tracker.features import (
    FRAME_FEATURE_DIM,
    build_frame_tokens,
    frame_feature_dim,
    frame_gradient_curvature,
    frame_normal_blocks,
    plan_row_mass,
    pointmap_residual_histogram,
    track_tokens,
)
from tests.tracker.synthetic_terms import CAUCHY_C, TermsFixture, make_terms_fixture

F64 = torch.float64
BLOCKS = ((0, 7), (7, 14), (14, 21), (21, 28))


def _rel_close(a: Tensor, ref: Tensor, rel: float) -> bool:
    return bool((a - ref).abs().max() <= rel * ref.abs().max())


def _tokens(fx: TermsFixture, cluster_id: Tensor) -> Tensor:
    return build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, cluster_id, fx.template_diameter, fx.poses, fx.log_scale)


def test_canonical_blocks_invariant_to_camera_frame() -> None:
    fx = make_terms_fixture()
    G = se3_exp(torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], dtype=F64))
    X2 = fx.X @ G[:3, :3].T + G[:3, 3]
    poses2 = G @ fx.poses
    pm2 = pointmap_term(X2, fx.X_valid, fx.Y, poses2, fx.log_scale, fx.eps, fx.rho_x, fx.rho_y, 0, plan=fx.plan)[0]
    L = fx.log_scale.exp() * fx.template_diameter
    H1, g1, _ = frame_normal_blocks([fx.pm], fx.poses, L)
    H2, g2, _ = frame_normal_blocks([pm2], poses2, L)
    assert _rel_close(H2, H1, 1e-9) and _rel_close(g2, g1, 1e-9)
    gs1, gs2 = frame_normal_blocks([fx.pm])[1], frame_normal_blocks([pm2])[1]
    assert (gs2 - gs1).abs().max() > 0.1 * gs1.abs().max()


def test_canonical_blocks_invariant_to_global_scale() -> None:
    fx = make_terms_fixture()
    a = 2.0
    poses3 = fx.poses.clone()
    poses3[:, :3, 3] *= a
    ls3 = fx.log_scale + math.log(a)
    pm3 = pointmap_term(a * fx.X, fx.X_valid, fx.Y, poses3, ls3, fx.eps, fx.rho_x, fx.rho_y, 0, plan=fx.plan)[0]
    L = fx.log_scale.exp() * fx.template_diameter
    H1, g1, _ = frame_normal_blocks([fx.pm], fx.poses, L)
    H3, g3, _ = frame_normal_blocks([pm3], poses3, a * L)
    assert _rel_close(H3, H1, 1e-12) and _rel_close(g3, g1, 1e-12)
    rm = plan_row_mass(fx.plan)
    assert torch.equal(pointmap_residual_histogram(pm3, rm, a * L), pointmap_residual_histogram(fx.pm, rm, L))


@pytest.mark.parametrize("knn", [None, 16])
def test_build_frame_tokens_invariant_to_global_scale_with_tracks(knn: int | None) -> None:
    fx = make_terms_fixture(knn=knn)
    cluster_id = mesh_cluster_ids(fx.Y, 64, 0)
    a = 2.0
    poses3 = fx.poses.clone()
    poses3[:, :3, 3] *= a
    ls3 = fx.log_scale + math.log(a)
    pm3 = pointmap_term(a * fx.X, fx.X_valid, fx.Y, poses3, ls3, a**2 * fx.eps, a**2 * fx.rho_x, a**2 * fx.rho_y, 0,
                        plan=fx.plan)[0]
    tr_raw3 = track_term(fx.anchors, fx.tracks, fx.track_vis, poses3, ls3, fx.K, CAUCHY_C)
    tok = _tokens(fx, cluster_id)
    tok3 = build_frame_tokens(pm3, tr_raw3, fx.plan, cluster_id, fx.template_diameter, poses3, ls3)
    for lo, hi in BLOCKS:
        ref_max = tok[:, lo:hi].abs().max()
        bound = 1e-12 * ref_max if ref_max > 0 else torch.tensor(1e-300, dtype=F64)
        assert (tok3[:, lo:hi] - tok[:, lo:hi]).abs().max() <= bound, (lo, hi)
    assert torch.equal(tok3[:, 28:], tok[:, 28:])
    # Contrast: the old solver-weighted combined block [pm, tr] is not similarity invariant.
    L2 = (fx.log_scale.exp() * fx.template_diameter) ** 2
    L, L3 = fx.log_scale.exp() * fx.template_diameter, ls3.exp() * fx.template_diameter
    tr3 = TermOutput(tr_raw3.residual, tr_raw3.jacobian, tr_raw3.weight / (a**2 * L2))
    old = frame_gradient_curvature([fx.pm, fx.tr], fx.poses, L)[0]
    old3 = frame_gradient_curvature([pm3, tr3], poses3, L3)[0]
    assert (old3 - old).abs().max() > 0.1 * old.abs().max()


def test_build_frame_tokens_layout() -> None:
    fx = make_terms_fixture(knn=16)
    cluster_id = mesh_cluster_ids(fx.Y, 64, 0)
    tok = _tokens(fx, cluster_id)
    assert FRAME_FEATURE_DIM == frame_feature_dim() == 110 and frame_feature_dim(8) == 114
    assert tok.shape == (6, 110) and tok.dtype == F64
    L = fx.log_scale.exp() * fx.template_diameter
    g_pm, h_pm = frame_gradient_curvature([fx.pm], fx.poses, L)
    g_tr, h_tr = frame_gradient_curvature([fx.tr_raw], fx.poses, L, metric_residual=False)
    assert torch.equal(tok[:, 0:7], g_pm) and torch.equal(tok[:, 7:14], h_pm)
    assert torch.equal(tok[:, 14:21], g_tr) and torch.equal(tok[:, 21:28], h_tr)
    hist = pointmap_residual_histogram(fx.pm, plan_row_mass(fx.plan), L)
    assert torch.equal(tok[:, 28:44], hist / 96)
    assert torch.equal(tok[:, 44:108], mesh_cluster_mass(fx.plan, cluster_id))
    assert torch.equal(tok[:, 108], torch.ones(6, dtype=F64))
    assert torch.equal(tok[:, 109], (fx.tr_raw.weight > 0).sum(1).to(F64) / 16)
    tok_e = build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, cluster_id.expand(6, 256), fx.template_diameter,
                               fx.poses, fx.log_scale.expand(6))
    assert torch.equal(tok_e, tok)
    tr = fx.tr_raw
    bad_d = TermOutput(torch.zeros(6, 16, 3, dtype=F64), torch.zeros(6, 16, 3, 7, dtype=F64), tr.weight)
    bad_t = TermOutput(tr.residual[:5], tr.jacobian[:5], tr.weight[:5])
    for bad in (bad_d, bad_t):
        with pytest.raises(ValueError):
            build_frame_tokens(fx.pm, bad, fx.plan, cluster_id, fx.template_diameter, fx.poses, fx.log_scale)


@pytest.mark.parametrize("knn", [None, 16])
def test_build_frame_tokens_finite_on_fully_occluded_frame(knn: int | None) -> None:
    fx = make_terms_fixture(occluded_frame=3, knn=knn)
    tok = _tokens(fx, mesh_cluster_ids(fx.Y, 64, 0))
    assert torch.isfinite(tok).all()
    assert torch.equal(tok[3, 0:44], torch.zeros(44, dtype=F64))
    torch.testing.assert_close(tok[3, 44:108], torch.zeros(64, dtype=F64), rtol=0.0, atol=1e-12)
    assert torch.equal(tok[3, 108:], torch.zeros(2, dtype=F64))
    others = [k for k in range(6) if k != 3]
    assert torch.equal(tok[others, 108], torch.ones(5, dtype=F64))


def test_non_finite_input_stays_in_its_frame() -> None:
    fx = make_terms_fixture(knn=16)
    cluster_id = mesh_cluster_ids(fx.Y, 64, 0)
    clean = _tokens(fx, cluster_id)
    jac = fx.pm.jacobian.clone()
    jac[4, 0, 0, 0] = float("nan")
    assert isinstance(fx.plan, SparsePlan)
    values = fx.plan.values.clone()
    values[4, 0, 0] = float("nan")
    tok = build_frame_tokens(TermOutput(fx.pm.residual, jac, fx.pm.weight), fx.tr_raw,
                             SparsePlan(indices=fx.plan.indices, values=values), cluster_id, fx.template_diameter,
                             fx.poses, fx.log_scale)
    others = [k for k in range(6) if k != 4]
    assert torch.equal(tok[others], clean[others])
    assert torch.isnan(tok[4, 0:14]).any()
    assert torch.isnan(tok[4, 44:108]).any()
    assert torch.equal(tok[4, 14:28], clean[4, 14:28])


def test_build_frame_tokens_reads_nothing_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    fx = make_terms_fixture(knn=16)
    cluster_id = mesh_cluster_ids(fx.Y, 64, 0)
    ref_tok = _tokens(fx, cluster_id)
    ref_track = track_tokens(fx.anchors, fx.tr_raw, fx.template_diameter)

    def host_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("host read")

    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__float__", "__int__"):
            m.setattr(torch.Tensor, name, host_read)
        tok = _tokens(fx, cluster_id)
        track = track_tokens(fx.anchors, fx.tr_raw, fx.template_diameter)
    assert tok.shape == (6, FRAME_FEATURE_DIM) and track.shape == (6, 16, 6)
    assert torch.isfinite(tok).all() and torch.isfinite(track).all()
    assert torch.equal(tok, ref_tok) and torch.equal(track, ref_track)
