import torch

from fastdo.solver.ot import (
    ot_cost,
    ot_objective,
    plan_barycenters,
    sinkhorn_unbalanced,
    sinkhorn_unbalanced_potentials,
)

torch.manual_seed(0)


def _clouds(n: int = 64, m: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.randn(1, m, 3, dtype=torch.float64)
    x = y[:, :n] + 0.01 * torch.randn(1, n, 3, dtype=torch.float64)
    return x, y


def test_balanced_marginals() -> None:
    x, y = _clouds(64, 64)
    C = torch.cdist(x, y) ** 2
    a = torch.full((1, 64), 1 / 64, dtype=torch.float64)
    b = torch.full((1, 64), 1 / 64, dtype=torch.float64)
    P = sinkhorn_unbalanced(C, a, b, eps=0.01, rho_a=float("inf"), rho_b=float("inf"), n_iter=200)
    assert torch.allclose(P.sum(2), a, atol=1e-4)
    assert torch.allclose(P.sum(1), b, atol=1e-4)


def test_unbalanced_leaves_hidden_targets_unmatched() -> None:
    x, y = _clouds(64, 96)                     # x samples only the first 64 of 96 targets
    C = torch.cdist(x, y) ** 2
    a = torch.full((1, 64), 1 / 64, dtype=torch.float64)
    b = torch.full((1, 96), 1 / 96, dtype=torch.float64)
    P = sinkhorn_unbalanced(C, a, b, eps=0.001, rho_a=10.0, rho_b=0.01, n_iter=300)
    hidden_mass = P[:, :, 64:].sum()
    total = P.sum()
    assert hidden_mass / total < 0.05
    assert torch.allclose(P.sum(2), a, atol=2e-3)   # every observed point is explained


def test_barycenters_recover_correspondence() -> None:
    x, y = _clouds(32, 32)
    C = torch.cdist(x, y) ** 2
    a = torch.full((1, 32), 1 / 32, dtype=torch.float64)
    b = torch.full((1, 32), 1 / 32, dtype=torch.float64)
    P = sinkhorn_unbalanced(C, a, b, eps=0.001, rho_a=float("inf"), rho_b=float("inf"), n_iter=300)
    ybar, w = plan_barycenters(P, y)
    assert torch.allclose(w, a, atol=1e-4)
    assert (ybar - y[:, :32]).norm(dim=-1).max() < 0.03


def test_envelope_gradient_matches_finite_difference() -> None:
    x, y = _clouds(24, 40)
    a = torch.full((1, 24), 1 / 24, dtype=torch.float64)
    b = torch.full((1, 40), 1 / 40, dtype=torch.float64)
    eps, rho_a, rho_b = 0.02, 10.0, 0.1

    def objective(shift: torch.Tensor) -> torch.Tensor:
        C = torch.cdist(x + shift, y) ** 2
        P = sinkhorn_unbalanced(C, a, b, eps, rho_a, rho_b, n_iter=2000)
        return ot_objective(C, P, a, b, eps, rho_a, rho_b)[0]

    s = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    C0 = torch.cdist(x + s, y) ** 2
    P0 = sinkhorn_unbalanced(C0, a, b, eps, rho_a, rho_b, n_iter=2000).detach()
    g = torch.autograd.grad(ot_cost(C0, P0)[0], s)[0]          # envelope: d objective / d shift = <dC/d shift, P*>
    num = torch.zeros(3, dtype=torch.float64)
    h = 1e-4
    for i in range(3):
        d = torch.zeros(3, dtype=torch.float64); d[i] = h
        num[i] = (objective(s.detach() + d) - objective(s.detach() - d)) / (2 * h)
    assert torch.allclose(g, num, atol=2e-4, rtol=2e-2), (g, num)


def _kl_reference(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    mask = p > 0
    terms = torch.where(mask, p * torch.log(torch.where(mask, p, torch.ones_like(p)) / q), torch.zeros_like(p)) - p + q
    return terms.flatten(1).sum(1)


def test_ot_objective_skips_marginal_terms_when_rho_is_inf() -> None:
    x, y = _clouds(16, 20)
    C = torch.cdist(x, y) ** 2
    a = torch.full((1, 16), 1 / 16, dtype=torch.float64)
    b = torch.full((1, 20), 1 / 20, dtype=torch.float64)
    P = sinkhorn_unbalanced(C, a, b, eps=0.05, rho_a=1.0, rho_b=1.0, n_iter=100)   # deliberately unbalanced plan
    balanced_obj = ot_objective(C, P, a, b, 0.05, float("inf"), float("inf"))
    entropic_only = ot_cost(C, P) + 0.05 * _kl_reference(P, a[:, :, None] * b[:, None, :])
    assert torch.allclose(balanced_obj, entropic_only, atol=1e-10)
    with_marginals = ot_objective(C, P, a, b, 0.05, 1.0, 1.0)
    assert (with_marginals > balanced_obj).all()          # KL marginal penalties are non-negative and non-zero here
    only_a = ot_objective(C, P, a, b, 0.05, 1.0, float("inf"))
    assert (balanced_obj < only_a).all() and (only_a < with_marginals).all()


def test_sinkhorn_finite_with_all_zero_mesh_marginal_row() -> None:
    """A batch element whose entire mesh-side marginal `b` is zero (a frame with no usable
    mesh_mask signal, before terms.py's own full-template fallback) must not produce NaN/inf: the
    all-zero `b` row's plan should end up ~0 everywhere (nothing to match against), while an ordinary
    row in the same batch is unaffected."""
    x, y = _clouds(20, 30)
    x2 = torch.cat([x, x], dim=0)                         # (2,20,3): batch of 2 identical point sets
    y2 = torch.cat([y, y], dim=0)
    C = torch.cdist(x2, y2) ** 2
    a = torch.full((2, 20), 1.0 / 20, dtype=torch.float64)
    b = torch.full((2, 30), 1.0 / 30, dtype=torch.float64)
    b[0] = 0.0                                            # frame 0: empty mesh-side marginal
    for eps in (1e-1, 1e-3, 1e-6):
        P, f, g = sinkhorn_unbalanced_potentials(C, a, b, eps=eps, rho_a=0.1, rho_b=1e-3, n_iter=50)
        assert torch.isfinite(P).all() and torch.isfinite(f).all() and torch.isfinite(g).all()
        assert float(P[0].sum()) < 1e-9, (eps, float(P[0].sum()))
        assert float(P[1].sum()) > 0.1, (eps, float(P[1].sum()))


def test_uniform_b_sinkhorn_matches_general_on_valid_rows() -> None:
    """Uniform-b Sinkhorn with merged constants and inert invalid rows matches the general
    solver on the valid rows; invalid rows of its plan are exactly zero."""
    from fastdo.solver.ot import sinkhorn_unbalanced_uniform_b

    dt = torch.float64
    gen = torch.Generator().manual_seed(0)
    B, N, M = 5, 14, 96
    x = torch.rand(B, N, 3, generator=gen, dtype=dt) - 0.5
    y = torch.rand(B, M, 3, generator=gen, dtype=dt) - 0.5
    lx = torch.randint(0, 6, (B, N), generator=gen)
    ly = torch.randint(0, 6, (B, M), generator=gen)
    C = torch.cdist(x, y) ** 2 + 0.05 * (lx[:, :, None] != ly[:, None, :]).to(dt)
    a = torch.exp(-2.0 * torch.rand(B, N, generator=gen, dtype=dt))
    a = torch.where(torch.rand(B, N, generator=gen, dtype=dt) < 0.3, torch.zeros_like(a), a)
    b = torch.full((B, M), 1.0 / M, dtype=dt)
    valid = a > 0
    assert (~valid).any()
    for eps in (0.055, 0.0055):
        plan_u = sinkhorn_unbalanced_uniform_b(C, a, valid, eps, 1.1, 0.011, 25)
        plan_g, _, _ = sinkhorn_unbalanced_potentials(C, a, b, eps, 1.1, 0.011, 25)
        assert float((plan_u - plan_g).abs()[valid].max()) <= 1e-12 * float(plan_g.max())
        assert torch.equal(plan_u[~valid], torch.zeros_like(plan_u[~valid]))
