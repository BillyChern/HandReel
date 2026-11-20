"""Model contract tests C1-C4 on the tokens: clamped count, solver-weight recovery, per-frame ids, per-frame count.

Written and run on BASE before any tracker-model source. Each test pins a feature-token contract that a
surviving feature-token mutant broke without failing the feature-token tests.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_adjoint, se3_exp
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import TermOutput, pointmap_term, track_term
from fastdo.tracker.clusters import mesh_cluster_ids, mesh_cluster_mass
from fastdo.tracker.features import (
    build_frame_tokens,
    frame_gradient_curvature,
    frame_normal_blocks,
    plan_row_mass,
    pointmap_residual_histogram,
)
from tests.tracker.synthetic_terms import CAUCHY_C, make_terms_fixture, random_terms

F64 = torch.float64


def _with_weight(t: TermOutput, weight: Tensor) -> TermOutput:
    return TermOutput(t.residual, t.jacobian, weight)


def test_frame_gradient_curvature_equals_normal_blocks_over_clamped_count() -> None:
    t1, t2 = random_terms(5, 12, 3, 7, seed=1), random_terms(5, 9, 2, 7, seed=2)
    w1, w2 = t1.weight.clone(), t2.weight.clone()
    w1[2], w2[2] = -w1[2].abs(), -w2[2].abs()           # frame 2: no positive weight in either term
    w1[3], w2[3] = -w1[3].abs(), -w2[3].abs()
    w1[3, 0] = 0.3                                       # frame 3: exactly one positive weight (0.3) in t1
    t1, t2 = _with_weight(t1, w1), _with_weight(t2, w2)
    poses = se3_exp(0.3 * torch.randn(5, 6, generator=torch.Generator().manual_seed(3), dtype=F64))
    length_scale = 0.1 * torch.arange(1, 6, dtype=F64)
    modes: list[tuple[tuple[Tensor, Tensor] | tuple[None, None], bool]] = [
        ((None, None), True), ((poses, length_scale), True), ((poses, length_scale), False)]
    for terms in ([t1], [t1, t2]):
        counts = [sum(int(float(t.weight[k, i]) > 0) for t in terms for i in range(t.weight.shape[1]))
                  for k in range(5)]
        for (p, ls), metric in modes:
            g, h = frame_gradient_curvature(terms, p, ls, metric_residual=metric)
            H_nb, g_nb, n_obs = frame_normal_blocks(terms, p, ls, metric_residual=metric)
            n = n_obs.clamp_min(1.0)[:, None]
            assert torch.equal(g, g_nb / n)
            assert torch.equal(h, torch.diagonal(H_nb, dim1=-2, dim2=-1) / n)
            assert n_obs.tolist() == [float(c) for c in counts]
            assert torch.isfinite(g).all() and torch.isfinite(h).all()
            assert torch.equal(g[2], torch.zeros(7, dtype=F64)) and torch.equal(h[2], torch.zeros(7, dtype=F64))
            assert torch.equal(g[3], g_nb[3]) and torch.equal(h[3], torch.diagonal(H_nb[3]))
            assert float(n_obs[3]) == 1.0
            assert math.isclose(float(sum(t.weight[3].clamp_min(0).sum() for t in terms)), 0.3, rel_tol=1e-15)
            off = H_nb - torch.diag_embed(torch.diagonal(H_nb, dim1=-2, dim2=-1))
            assert float(off.abs().max()) > 1e-3


def _per_frame_rel(lhs: Tensor, rhs: Tensor) -> Tensor:
    """(T,) max|lhs - rhs| / max|rhs| per frame (rows flattened)."""
    return (lhs - rhs).flatten(1).abs().amax(1) / rhs.flatten(1).abs().amax(1)


def _canonical_diag(H: Tensor, Mk: Tensor) -> Tensor:
    """(T,P) diag(Mk^T H Mk)."""
    return torch.diagonal(Mk.transpose(1, 2) @ H @ Mk, dim1=-2, dim2=-1)


@pytest.mark.parametrize("a", [1.0, 2.0])
@pytest.mark.parametrize("per_frame_scale", [False, True], ids=["ls0dim", "lsT"])
@pytest.mark.parametrize("knn", [None, 16], ids=["dense", "sparse"])
def test_solver_weighting_recovered_from_tokens(knn: int | None, per_frame_scale: bool, a: float) -> None:
    extra = [None, 3] if (a == 1.0 and not per_frame_scale) else [None]   # the occluded-frame extra case
    for occluded in extra:
        fx = make_terms_fixture(knn=knn, occluded_frame=occluded)
        T = fx.poses.shape[0]
        poses = fx.poses.clone()
        poses[:, :3, 3] *= a
        log_scale0 = fx.log_scale + math.log(a)
        if a == 1.0:
            pm, tr_raw = fx.pm, fx.tr_raw
        else:
            pm = pointmap_term(a * fx.X, fx.X_valid, fx.Y, poses, log_scale0, a**2 * fx.eps, a**2 * fx.rho_x,
                               a**2 * fx.rho_y, 0, plan=fx.plan)[0]
            tr_raw = track_term(fx.anchors, fx.tracks, fx.track_vis, poses, log_scale0, fx.K, CAUCHY_C)
        log_scale = log_scale0 + 0.05 * torch.arange(T, dtype=F64) if per_frame_scale else log_scale0
        lam_track = 0.37
        scale2 = (0.8 * math.exp(float(log_scale0)) * fx.template_diameter) ** 2   # deliberately not L^2
        w_t = lam_track / scale2
        tr_w = TermOutput(tr_raw.residual, tr_raw.jacobian, w_t * tr_raw.weight)
        tok = build_frame_tokens(pm, tr_raw, fx.plan, mesh_cluster_ids(fx.Y, 64, 0), fx.template_diameter, poses,
                                 log_scale)
        H_s, g_s, _ = frame_normal_blocks([pm, tr_w])
        L = log_scale.exp().expand(T) * fx.template_diameter
        Mk = torch.zeros(T, 7, 7, dtype=F64)
        Mk[:, :6, :6] = se3_adjoint(poses)
        Mk[:, 6, 6] = 1.0
        Mk[:, :, :3] *= L[:, None, None]
        MkT = Mk.transpose(1, 2)
        n_pm = (pm.weight > 0).sum(1).clamp_min(1).to(F64)[:, None]
        n_tr = (tr_raw.weight > 0).sum(1).clamp_min(1).to(F64)[:, None]
        L1, L2 = L[:, None], (L * L)[:, None]

        g_lhs = (MkT @ g_s[..., None])[..., 0]
        rows = [k for k in range(T) if k != occluded]
        cases = {
            "g": (g_lhs, L2 * n_pm * tok[:, 0:7] + w_t * n_tr * tok[:, 14:21]),
            "H": (_canonical_diag(H_s, Mk), L2 * n_pm * tok[:, 7:14] + w_t * n_tr * tok[:, 21:28]),
        }
        H_p, g_p, _ = frame_normal_blocks([pm])
        cases["g_pm"] = ((MkT @ g_p[..., None])[..., 0], L2 * n_pm * tok[:, 0:7])
        cases["H_pm"] = (_canonical_diag(H_p, Mk), L2 * n_pm * tok[:, 7:14])
        for name, (lhs, rhs) in cases.items():
            assert bool((_per_frame_rel(lhs, rhs)[rows] <= 1e-12).all()), (name, _per_frame_rel(lhs, rhs))
            if occluded is not None:
                assert torch.equal(lhs[occluded], torch.zeros(7, dtype=F64)), name
                assert torch.equal(rhs[occluded], torch.zeros(7, dtype=F64)), name
        if occluded is not None:
            continue
        w_bad = lam_track / (L * L)[:, None]
        mutants = {
            "L for L^2, g": (g_lhs, L1 * n_pm * tok[:, 0:7] + w_t * n_tr * tok[:, 14:21]),
            "L for L^2, g_pm": (cases["g_pm"][0], L1 * n_pm * tok[:, 0:7]),
            "L for L^2, H_pm": (cases["H_pm"][0], L1 * n_pm * tok[:, 7:14]),
            "lam/L^2 for w_t, g": (g_lhs, L2 * n_pm * tok[:, 0:7] + w_bad * n_tr * tok[:, 14:21]),
            "lam/L^2 for w_t, H": (cases["H"][0], L2 * n_pm * tok[:, 7:14] + w_bad * n_tr * tok[:, 21:28]),
        }
        for name, (lhs, rhs) in mutants.items():
            assert bool((_per_frame_rel(lhs, rhs) > 1e-3).all()), (name, _per_frame_rel(lhs, rhs))


@pytest.mark.parametrize("knn", [None, 16], ids=["dense", "sparse"])
def test_mesh_cluster_mass_per_frame_ids_follow_a_column_permutation(knn: int | None) -> None:
    fx = make_terms_fixture(knn=knn)
    ids = mesh_cluster_ids(fx.Y, 64, 0)
    k = 2
    pi = torch.randperm(256, generator=torch.Generator().manual_seed(7))
    inv = torch.argsort(pi)
    plan2: Tensor | SparsePlan
    if isinstance(fx.plan, SparsePlan):
        indices2 = fx.plan.indices.clone()
        indices2[k] = inv[fx.plan.indices[k]]
        plan2 = SparsePlan(indices=indices2, values=fx.plan.values)
    else:
        plan2 = fx.plan.clone()
        plan2[k] = fx.plan[k][:, pi]
    ids_TM = ids.expand(6, 256).clone()
    ids_TM[k] = ids[pi]
    ref = mesh_cluster_mass(fx.plan, ids)
    out = mesh_cluster_mass(plan2, ids_TM)
    others = [j for j in range(6) if j != k]
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-15)
    assert torch.equal(out[others], ref[others])
    wrong = mesh_cluster_mass(plan2, ids)                 # shared ids with a permuted plan: row k must change
    assert float((wrong[k] - ref[k]).abs().max()) > 1e-3 * float(ref[k].abs().max())
    assert torch.equal(wrong[others], ref[others])
    tok_ref = build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, ids, fx.template_diameter, fx.poses, fx.log_scale)
    tok = build_frame_tokens(fx.pm, fx.tr_raw, plan2, ids_TM, fx.template_diameter, fx.poses, fx.log_scale)
    torch.testing.assert_close(tok[:, 44:108], tok_ref[:, 44:108], rtol=1e-12, atol=1e-15)
    assert torch.equal(tok[:, 0:28], tok_ref[:, 0:28]) and torch.equal(tok[:, 108:110], tok_ref[:, 108:110])
    atol = 0.0 if knn else 1e-12
    torch.testing.assert_close(tok[:, 28:44], tok_ref[:, 28:44], rtol=0.0, atol=atol)


@pytest.mark.parametrize("knn", [None, 16], ids=["dense", "sparse"])
def test_histogram_normalized_by_per_frame_valid_count(knn: int | None) -> None:
    fx = make_terms_fixture()
    X_valid2 = torch.zeros(6, 96, dtype=torch.bool)
    X_valid2[0] = True
    X_valid2[1, 0:50] = True
    X_valid2[2, 7] = True
    X_valid2[4, 0:17] = True
    X_valid2[5] = True
    pm2, plan2 = pointmap_term(fx.X, X_valid2, fx.Y, fx.poses, fx.log_scale, fx.eps, fx.rho_x, fx.rho_y, 50,
                               knn=knn)
    ids = mesh_cluster_ids(fx.Y, 64, 0)
    tok = build_frame_tokens(pm2, fx.tr_raw, plan2, ids, fx.template_diameter, fx.poses, fx.log_scale)
    L = fx.log_scale.exp() * fx.template_diameter
    counts = pointmap_residual_histogram(pm2, plan_row_mass(plan2), L)
    divisor = torch.tensor([96, 50, 1, 1, 17, 96], dtype=F64)
    assert torch.equal(tok[:, 28:44], counts / divisor[:, None])
    sums = tok[:, 28:44].sum(1)
    torch.testing.assert_close(sums, torch.tensor([1, 1, 1, 0, 1, 1], dtype=F64), rtol=0.0, atol=1e-12)
    assert torch.equal(tok[3, 28:44], torch.zeros(16, dtype=F64))
    assert tok[:, 108].tolist() == [1.0, 50 / 96, 1 / 96, 0.0, 17 / 96, 1.0]
    assert float((tok[1, 28:44] - counts[1] / 96).abs().max()) > 0.1
