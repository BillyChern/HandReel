"""Truncated unbalanced-Sinkhorn gradient check and its report (checks K1-K10).

QUICK_CONFIG problem unless stated. K7 is the recorded QUICK snapshot plus checks that follow from theorems (Hoelder,
the entrywise plan bound, the kappa contraction). There are 7 probes: 3 translations,
3 rotations about the observed centroid and 1 scale, each scaled to ||E||_inf = eps.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from fastdo.solver.ot import _tau, sinkhorn_unbalanced_potentials
from fastdo.tracker import (
    GradientEstimates,
    implicit_cost_gradient,
    implicit_value_gradient,
    truncated_sinkhorn_gradient_check,
)
from fastdo.tracker.gradient_check_report import (
    INIT_DRIVE,
    QUICK_CONFIG,
    GradientCheckConfig,
    GradientCheckProblem,
    _decide,
    build_gradient_check_problem,
    main,
    run_gradient_check_report,
    warm_start,
)
from fastdo.tracker.sinkhorn_check import (
    ConvergedReference,
    converged_reference,
    dual_gradients,
    sinkhorn_iterate,
)

DT = torch.float64
N_ITERS = (2, 3, 4, 5)
ESTIMATORS = ("detached", "unrolled", "implicit")
GROUPS = ("T3", "R3", "S1", "all7", "max_group")
# Recorded QUICK_CONFIG run (dual_noise, g drive): relative bias against FD of the converged value, n = 2..5.
SNAPSHOT: dict[str, dict[str, list[float]]] = {
    "detached": {"T3": [1.2504077635950643e-02, 8.349804311228384e-03, 5.281025165617864e-03, 2.965687190474177e-03],
                 "R3": [4.931261759870014e-02, 2.0632398383448235e-02, 8.990358708379128e-03, 4.021783243632232e-03],
                 "S1": [9.816723101990214e-03, 3.0649005231135566e-03, 3.4339610076372204e-03, 2.1501531790350022e-03]},
    "unrolled": {"T3": [3.875494450735256e-02, 1.1461425517722997e-02, 3.007951761325721e-03, 7.541652084665138e-04],
                 "R3": [9.194005038161238e-02, 2.0147059237769462e-02, 4.020574517847772e-03, 8.237986297211361e-04],
                 "S1": [2.705257496227662e-02, 7.169938242162005e-03, 1.7256420777566992e-03, 3.7814313834354766e-04]},
    "implicit": {"T3": [3.012638642381789e-04, 7.559530364233122e-05, 1.884854939075151e-05, 4.671580614750881e-06],
                 "R3": [4.3690763912636245e-04, 9.49238118911491e-05, 2.188260734669735e-05, 5.193028346115149e-06],
                 "S1": [6.38439749705131e-04, 1.5087930958955696e-04, 3.6529245806365234e-05, 8.886550630062631e-06]},
}
SNAPSHOT_DECISION = "detached"
TOP_KEYS = {"schema", "created_utc", "torch_version", "scope", "config", "problem", "tau_a", "tau_b", "kappa",
            "protocol", "cases", "decision",
            *(f"{prefix}n_iter_{n}" for prefix in ("", "unrolled_", "implicit_") for n in N_ITERS)}
PROTOCOL_KEYS = {"envelope_vs_fd", "unrolled_vs_fd_truncated_max", "grad_g_over_grad_f_max",
                 "contraction_ratio_max", "contraction_ratio_over_kappa_max",
                 "barycentric_planfixed_vs_exact_rel", "barycentric_planfixed_vs_exact_cos",
                 "barycentric_groups"}
DECISION_KEYS = {"threshold", "budget_n_iter", "primary_init", "decision_inits", "metric", "bias_at_budget",
                 "material", "ot_gradient", "dual_init_detached", "applies_to"}
PROBLEM_KEYS = {"N", "M", "n_visible", "n_valid_obs", "scale2", "length_scale", "eps", "rho_a", "rho_b",
                "probe_groups", "synthetic_default_dtype"}
# QUICK-sized report over g- and f-driven inits, so the max-over-inits decision is exercised (review I-2).
MULTI_INITS = ("cluster_oracle", "f_oracle", "dual_noise", "lm_warm", "cold")
MULTI_CONFIG = dataclasses.replace(QUICK_CONFIG, inits=MULTI_INITS, decision_inits=MULTI_INITS)


def _bias(G: Tensor, ref: Tensor) -> float:
    return float((G - ref).norm() / ref.norm())


def _close(x: Tensor, ref: Tensor, rel: float) -> None:
    err = float((x - ref).abs().max())
    assert err <= rel * float(ref.abs().max()), (err, float(ref.abs().max()))


@pytest.fixture(scope="module")
def quick() -> tuple[GradientCheckProblem, ConvergedReference]:
    p, cfg = build_gradient_check_problem(QUICK_CONFIG), QUICK_CONFIG
    ref = converged_reference(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, p.directions, cfg.h, cfg.n_converged,
                              cfg.n_converged_fd)
    return p, ref


@pytest.fixture(scope="module")
def quick_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    root = tmp_path_factory.mktemp("report")
    return run_gradient_check_report(root / "tracker" / "gradient_check.json", MULTI_CONFIG), root


def _init(p: GradientCheckProblem, ref: ConvergedReference, kind: str) -> tuple[Tensor, Tensor]:
    return warm_start(p, kind, QUICK_CONFIG, (ref.f, ref.g))


def _check(p: GradientCheckProblem, ref: ConvergedReference, kind: str, n: int) -> GradientEstimates:
    return truncated_sinkhorn_gradient_check(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, n, _init(p, ref, kind),
                                             p.directions, h=QUICK_CONFIG.h, drive=INIT_DRIVE[kind],
                                             reference=ref)


def test_truncated_sinkhorn_gradient_check_reports_finite_numbers(
        quick: tuple[GradientCheckProblem, ConvergedReference]) -> None:
    p, ref = quick
    est = _check(p, ref, "dual_noise", 5)
    for name in GradientEstimates._fields:
        value = getattr(est, name)
        assert value.shape == (7,) and bool(torch.isfinite(value).all()), name
    init, cfg = _init(p, ref, "dual_noise"), QUICK_CONFIG
    fresh = truncated_sinkhorn_gradient_check(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, 5, init, p.directions,
                                              n_converged=cfg.n_converged, n_converged_fd=cfg.n_converged_fd)
    assert torch.equal(fresh.fd_converged, est.fd_converged) and torch.equal(fresh.detached, est.detached)
    bad = [(p.C.float(), p.directions), (p.C.expand(2, -1, -1), p.directions), (p.C, p.directions[:, 0])]
    for C, E in bad:
        with pytest.raises(ValueError):
            truncated_sinkhorn_gradient_check(C, p.a, p.b, p.eps, p.rho_a, p.rho_b, 2, init, E)
    with pytest.raises(ValueError):
        truncated_sinkhorn_gradient_check(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, 0, init, p.directions)
    with pytest.raises(ValueError):
        warm_start(p, "unknown", QUICK_CONFIG, (ref.f, ref.g))
    with pytest.raises(ValueError, match="reference"):
        truncated_sinkhorn_gradient_check(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, 2, init, p.directions[:3],
                                          reference=ref)


def test_envelope_and_finite_difference_protocol(
        quick: tuple[GradientCheckProblem, ConvergedReference]) -> None:
    p, ref = quick
    assert _bias(ref.envelope, ref.fd_converged) <= 1e-7
    for n in N_ITERS:
        est = _check(p, ref, "dual_noise", n)
        assert _bias(est.unrolled, est.fd_truncated) <= 1e-6, n
    est = _check(p, ref, "f_noise", 3)                                   # the f drive differentiates g0(C)
    assert _bias(est.unrolled, est.fd_truncated) <= 1e-6


@pytest.mark.parametrize("n", [1, 2, 5])
def test_last_half_step_is_the_g_update(quick: tuple[GradientCheckProblem, ConvergedReference],
                                        n: int) -> None:
    p, ref = quick
    for kind in ("dual_noise", "f_noise", "cold"):
        init = _init(p, ref, kind)
        _, f, g = sinkhorn_iterate(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, n, init, INIT_DRIVE[kind])
        grad_f, grad_g = dual_gradients(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, f, g)
        assert float(grad_f.abs().max()) > 0, kind
        assert float(grad_g.abs().max()) <= 1e-10 * float(grad_f.abs().max()), kind
    g0 = _init(p, ref, "dual_noise")[1]                  # the f init is overwritten by the first f half-step
    kept = sinkhorn_unbalanced_potentials(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, n, init=(ref.f, g0))
    zeroed = sinkhorn_unbalanced_potentials(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, n,
                                            init=(torch.zeros_like(ref.f), g0))
    assert all(torch.equal(x, y) for x, y in zip(kept, zeroed, strict=True))


@pytest.mark.parametrize("eps_scale2", [1e-3, 1e-4])
def test_potential_contraction_bound(eps_scale2: float) -> None:
    cfg = dataclasses.replace(QUICK_CONFIG, eps_scale2=eps_scale2)
    p = build_gradient_check_problem(cfg)
    _, fs, gs = sinkhorn_unbalanced_potentials(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, cfg.n_converged)
    kappa = float(_tau(p.rho_a, p.eps)) * float(_tau(p.rho_b, p.eps))
    assert abs(kappa - p.rho_a / (p.rho_a + p.eps) * p.rho_b / (p.rho_b + p.eps)) <= 1e-15
    S = p.b[0] > 0
    for kind in ("dual_noise", "cold"):
        init = warm_start(p, kind, cfg, (fs, gs))
        errs = []
        for n in range(1, 7):
            _, f, g = sinkhorn_iterate(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, n, init, "g")
            errs.append((float((f - fs).abs().max()), float((g - gs)[:, S].abs().max())))
        slack = 1e-12 * p.eps
        for (f0, g0), (f1, g1) in itertools.pairwise(errs):
            assert f1 <= kappa * (1 + 1e-9) * f0 + slack, (kind, f0, f1)
            assert g1 <= kappa * (1 + 1e-9) * g0 + slack, (kind, g0, g1)


def _dual_value(z_f: Tensor, z_g: Tensor, C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float,
                rho_b: float) -> Tensor:
    """The dual objective D(f,g;C) on rows R = {a > 0} and columns S = {b > 0}."""
    R, S = a[0] > 0, b[0] > 0
    a_R, b_S, C_RS = a[0, R], b[0, S], C[0][R][:, S]
    return (-rho_a * (a_R * (torch.exp(-z_f / rho_a) - 1)).sum() - rho_b * (b_S * (torch.exp(-z_g / rho_b) - 1)).sum()
            - eps * (a_R[:, None] * b_S[None, :] * (torch.exp((z_f[:, None] + z_g[None, :] - C_RS) / eps) - 1)).sum())


@pytest.mark.parametrize("invalid_rows", [0, 2])
def test_implicit_closed_form_matches_autograd_hessian(invalid_rows: int) -> None:
    cfg = GradientCheckConfig(n_frames_clip=2, n_obs=6, n_surface=16, eps_scale2=1e-2, rho_a_scale2=0.1,
                              rho_b_scale2=1e-2, inits=("dual_noise",), primary_init="dual_noise",
                              decision_inits=("dual_noise",))
    p = build_gradient_check_problem(cfg)
    a = p.a.clone()
    a[0, :invalid_rows] = 0.0
    a = a / a.sum()
    R, S = a[0] > 0, p.b[0] > 0
    assert p.cluster_id is None and 2 <= int(S.sum()) < 16
    _, fs, gs = sinkhorn_unbalanced_potentials(p.C, a, p.b, p.eps, p.rho_a, p.rho_b, 500)
    gen = torch.Generator().manual_seed(5)
    init = (fs + 0.3 * p.eps * torch.randn(fs.shape, generator=gen, dtype=DT),
            gs + 0.3 * p.eps * torch.randn(gs.shape, generator=gen, dtype=DT))
    P, f, g = sinkhorn_iterate(p.C, a, p.b, p.eps, p.rho_a, p.rho_b, 2, init, "g")
    E = p.eps * torch.randn((3, *p.C.shape), generator=gen, dtype=DT)
    got = implicit_value_gradient(p.C, a, p.b, p.eps, p.rho_a, p.rho_b, P, f, g, E)
    matrix = implicit_cost_gradient(p.C, a, p.b, p.eps, p.rho_a, p.rho_b, P, f, g)
    assert matrix.shape == p.C.shape
    _close((matrix[None] * E).sum((1, 2, 3)), got, 1e-13)
    n_R, z0 = int(R.sum()), torch.cat([f[0, R], g[0, S]])

    def dual(z: Tensor, C: Tensor) -> Tensor:
        return _dual_value(z[:n_R], z[n_R:], C, a, p.b, p.eps, p.rho_a, p.rho_b)

    def grad_z(C: Tensor) -> Tensor:
        z = z0.clone().requires_grad_(True)
        return torch.autograd.grad(dual(z, C), z, create_graph=True)[0]

    hess = torch.autograd.functional.hessian(lambda z: dual(z, p.C), z0)
    step = torch.linalg.solve(hess, grad_z(p.C).detach())
    ref = torch.stack([(P * E[k]).sum() - torch.autograd.functional.jvp(grad_z, p.C, E[k])[1] @ step
                       for k in range(3)])
    _close(got, ref, 1e-10)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_unrolled_minus_detached_is_the_row_residual_term(
        quick: tuple[GradientCheckProblem, ConvergedReference], n: int) -> None:
    p, ref = quick
    est, init = _check(p, ref, "dual_noise", n), _init(p, ref, "dual_noise")
    K, R = p.directions.shape[0], p.a[0] > 0

    def rows(z: Tensor) -> Tensor:
        Cz = p.C + (z[:, None, None, None] * p.directions).sum(0)
        return sinkhorn_iterate(Cz, p.a, p.b, p.eps, p.rho_a, p.rho_b, n, init, "g")[0][0].sum(1)

    zero = torch.zeros(K, dtype=DT)
    _, f, _ = sinkhorn_iterate(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, n, init, "g")
    alpha = f[0, R] + p.rho_a * torch.log(rows(zero)[R] / p.a[0, R])
    predicted = (alpha[:, None] * torch.autograd.functional.jacobian(rows, zero)[R]).sum(0)
    _close(est.unrolled - est.detached, predicted, 1e-8)


def test_bias_bounds_and_snapshot(quick: tuple[GradientCheckProblem, ConvergedReference],
                                  quick_report: tuple[dict[str, Any], Path]) -> None:
    case = quick_report[0]["cases"]["dual_noise"]
    for est, groups in SNAPSHOT.items():
        for group, values in groups.items():
            for n, want in zip(N_ITERS, values, strict=True):
                got = case[est][group][f"n_iter_{n}"]
                assert abs(got - want) <= 1e-6 * abs(want), (est, group, n, got, want)
    p, ref = quick
    S, mass = p.b[0] > 0, float(ref.P.sum())
    kappa = p.rho_a / (p.rho_a + p.eps) * p.rho_b / (p.rho_b + p.eps)
    E_inf = p.directions.abs().flatten(1).max(1).values
    for kind in ("dual_noise", "f_noise"):
        init, deltas = _init(p, ref, kind), []
        for n in range(1, 7):
            P, f, g = sinkhorn_iterate(p.C, p.a, p.b, p.eps, p.rho_a, p.rho_b, n, init, INIT_DRIVE[kind])
            delta = max(float((f - ref.f).abs().max()), float((g - ref.g)[:, S].abs().max()))
            deltas.append(delta)
            l1 = float((P - ref.P).abs().sum())                                           # (ii)
            assert l1 <= mass * math.expm1(2 * delta / p.eps) * (1 + 1e-9), (kind, n)
            detached = (P[None] * p.directions).sum((1, 2, 3))                             # (i) Hoelder
            assert bool(((detached - ref.envelope).abs() <= E_inf * l1 * (1 + 1e-9)).all()), (kind, n)
        for d0, d1 in itertools.pairwise(deltas):                          # (iii)
            assert d1 <= kappa * (1 + 1e-9) * d0 + 1e-12 * p.eps, (kind, d0, d1)


def test_gradient_check_report_writes_expected_keys(quick_report: tuple[dict[str, Any], Path]) -> None:
    report, root = quick_report
    path = root / "tracker" / "gradient_check.json"
    assert sorted(x.relative_to(root).as_posix() for x in root.rglob("*")) == [
        "tracker", "tracker/gradient_check.json"]
    assert json.loads(path.read_text()) == report
    assert set(report) == TOP_KEYS and set(report["protocol"]) == PROTOCOL_KEYS
    assert set(report["decision"]) == DECISION_KEYS and set(report["problem"]) == PROBLEM_KEYS
    assert report["schema"] == "fastdo.tracker.gradient_check/v1" and report["scope"] == "dense_float64"
    assert report["created_utc"].endswith("Z")
    config = report["config"]
    assert set(config) == {field.name for field in dataclasses.fields(GradientCheckConfig)}
    assert all(isinstance(config[k], float) for k in ("eps_scale2", "rho_a_scale2", "rho_b_scale2"))
    case = report["cases"]["dual_noise"]
    assert set(case) == {"drive", "z0_sup_err_g_support_over_eps", *ESTIMATORS} and case["drive"] == "g"
    for est in ESTIMATORS:
        assert set(case[est]) == set(GROUPS)
        for group in GROUPS:
            assert set(case[est][group]) == {f"n_iter_{n}" for n in N_ITERS}
    for prefix, est in (("", "detached"), ("unrolled_", "unrolled"), ("implicit_", "implicit")):
        for n in N_ITERS:
            value = report[f"{prefix}n_iter_{n}"]
            assert isinstance(value, float) and math.isfinite(value) and value >= 0
            assert value == case[est]["max_group"][f"n_iter_{n}"]
    decision, cases, proto = report["decision"], report["cases"], report["protocol"]
    assert decision["ot_gradient"] in {"detached", "implicit", "unrolled", "none"}
    assert decision["dual_init_detached"] is True and decision["applies_to"] == "ot_objective"
    assert decision["ot_gradient"] == _decide(decision["bias_at_budget"], decision["threshold"])
    assert decision["decision_inits"] == list(MULTI_INITS) and set(cases) == set(MULTI_INITS)
    for est in ESTIMATORS:                                  # max over inits of the max probe group
        assert decision["bias_at_budget"][est] == max(cases[k][est]["max_group"]["n_iter_5"] for k in MULTI_INITS)
    assert any(decision["bias_at_budget"][est] > case[est]["max_group"]["n_iter_5"] for est in ESTIMATORS)
    assert cases["f_oracle"]["drive"] == "f" and "z0_sup_err_f_over_eps" in cases["f_oracle"]
    assert cases["lm_warm"]["z0_sup_err_g_support_over_eps"] > 0.1 and INIT_DRIVE["f_noise"] == "f"
    assert all(cases["f_oracle"][est]["max_group"]["n_iter_2"] <= 1e-6 for est in ESTIMATORS)   # FD floor
    assert all(isinstance(v, bool) for v in decision["material"].values())
    assert report["problem"]["synthetic_default_dtype"] == "float32"
    assert proto["envelope_vs_fd"] <= 1e-7 and proto["grad_g_over_grad_f_max"] <= 1e-10
    assert 0 < proto["contraction_ratio_over_kappa_max"] <= 1 + 1e-9
    assert proto["contraction_ratio_over_kappa_max"] == proto["contraction_ratio_max"] / report["kappa"]
    assert proto["barycentric_planfixed_vs_exact_rel"] > 1e-3
    assert 0 < proto["unrolled_vs_fd_truncated_max"] <= 1e-6


def test_problem_is_independent_of_the_default_dtype(
        quick: tuple[GradientCheckProblem, ConvergedReference], tmp_path: Path) -> None:
    (p, _), previous = quick, torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)                  # review I-1: the clip's noise stream is pinned
    try:
        q = build_gradient_check_problem(QUICK_CONFIG)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(q.C, p.C) and torch.equal(q.directions, p.directions)
    big = build_gradient_check_problem(dataclasses.replace(QUICK_CONFIG, n_surface=256))   # r2 I-1: clusters of 2+
    gs, vis = torch.randn(1, 256, generator=torch.Generator().manual_seed(3), dtype=DT), big.b[0] > 0
    assert big.cluster_id is not None and int(torch.bincount(big.cluster_id[vis]).max()) >= 2
    g0 = warm_start(big, "cluster_oracle", QUICK_CONFIG, (torch.zeros_like(big.a), gs))[1][0]   # visible-sample mean
    for c in torch.unique(big.cluster_id).tolist():
        members, both = big.cluster_id == c, (big.cluster_id == c) & vis
        want = gs[0, both].mean() if bool(both.any()) else torch.zeros((), dtype=DT)
        assert float((g0[members] - want).abs().max()) <= 1e-12, c
    with pytest.raises(ValueError, match="no iterate"):                 # nothing to validate
        run_gradient_check_report(tmp_path / "unused.json", dataclasses.replace(
            QUICK_CONFIG, inits=("f_oracle",), primary_init="f_oracle", decision_inits=("f_oracle",)))


def test_decision_rule() -> None:
    assert _decide({"detached": 0.43, "implicit": 0.05, "unrolled": 2.4}, 0.1) == "implicit"
    assert _decide({"detached": 0.43, "implicit": 0.15, "unrolled": 2.4}, 0.1) == "none"
    assert _decide({"detached": 0.1, "implicit": 0.05, "unrolled": 0.01}, 0.1) == "detached"
    assert _decide({"detached": 0.2, "implicit": 0.3, "unrolled": 0.01}, 0.1) == "unrolled"


def test_report_cli_quick(tmp_path: Path) -> None:
    repo_out = Path(__file__).resolve().parents[2] / "outputs" / "tracker" / "gradient_check.json"
    before = repo_out.stat().st_mtime_ns if repo_out.exists() else None
    target = tmp_path / "g.json"
    assert main(["--out", str(target), "--quick"]) == 0
    report = json.loads(target.read_text())
    assert set(report) == TOP_KEYS and report["decision"]["ot_gradient"] == SNAPSHOT_DECISION
    assert (repo_out.stat().st_mtime_ns if repo_out.exists() else None) == before
