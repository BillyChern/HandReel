"""The underflow-avoiding exp / logsumexp / Sinkhorn are bit-identical to the torch expressions they replace."""

from __future__ import annotations

import torch

from fastdo.solver.ot import (
    _LOG_ZERO,
    _exp_exact,
    _logsumexp_exact,
    _sinkhorn_unbalanced_uniform_b_torch,
    sinkhorn_unbalanced_uniform_b,
)


def test_exp_exact_bitwise_across_the_underflow_range() -> None:
    z = torch.cat([torch.linspace(-800.0, 0.0, 400_001, dtype=torch.float64),
                   torch.tensor([-1e30, -746.0, -745.2, -708.4, -700.0, -699.99, float("-inf"), float("nan")],
                                dtype=torch.float64)])
    z = z[torch.randperm(z.numel(), generator=torch.Generator().manual_seed(0))]
    e, r = _exp_exact(z), torch.exp(z)
    assert torch.equal(torch.isnan(e), torch.isnan(r))
    keep = ~torch.isnan(r)
    assert torch.equal(e[keep], r[keep])
    assert torch.equal(e.view(torch.int64)[keep], r.view(torch.int64)[keep])   # also the sign of zero


def test_exp_is_position_and_length_independent_in_the_subnormal_band() -> None:
    """`_exp_exact` recomputes [-746, -700) entries on a gathered tensor; that is exact only if `torch.exp`'s result
    for an entry does not depend on where it sits (vector body vs tail)."""
    g = torch.Generator().manual_seed(3)
    for n in (*range(1, 40), 63, 64, 65, 1025):
        z = -700.0 - 46.0 * torch.rand(n, generator=g, dtype=torch.float64)
        full = torch.exp(z)
        for s in range(min(n, 5)):
            assert torch.equal(torch.exp(z[s:].clone()), full[s:])
        assert torch.equal(torch.stack([torch.exp(z[i:i + 1].clone())[0] for i in range(n)]), full)


def test_logsumexp_exact_bitwise() -> None:
    g = torch.Generator().manual_seed(1)
    for dim in (1, 2):
        x = -torch.rand(13, 21, 96, generator=g, dtype=torch.float64) * 3000.0
        x[:, 3] = _LOG_ZERO
        x[2] = _LOG_ZERO                                           # a whole slice at the sentinel
        assert torch.equal(_logsumexp_exact(x, dim), torch.logsumexp(x, dim=dim))


def test_sinkhorn_default_bitwise_against_the_torch_expressions() -> None:
    g = torch.Generator().manual_seed(2)
    for eps in (0.05, 0.005, 0.0005):
        C = torch.rand(17, 20, 96, generator=g, dtype=torch.float64)
        a = torch.rand(17, 20, generator=g, dtype=torch.float64)
        valid = torch.rand(17, 20, generator=g) < 0.55
        valid[:, 0] = True
        valid[4] = False                                           # a frame with no valid row
        ref = _sinkhorn_unbalanced_uniform_b_torch(C, a, valid, eps, 1.0, 0.01, 25)
        out = sinkhorn_unbalanced_uniform_b(C, a, valid, eps, 1.0, 0.01, 25)
        assert torch.equal(out.view(torch.int64), ref.view(torch.int64))


def test_sinkhorn_default_keeps_autograd_on_the_torch_path() -> None:
    g = torch.Generator().manual_seed(4)
    C = torch.rand(3, 6, 12, generator=g, dtype=torch.float64, requires_grad=True)
    a = torch.rand(3, 6, generator=g, dtype=torch.float64)
    valid = torch.ones(3, 6, dtype=torch.bool)
    out = sinkhorn_unbalanced_uniform_b(C, a, valid, 0.005, 1.0, 0.01, 5)
    out.sum().backward()
    assert C.grad is not None and bool(torch.isfinite(C.grad).all())
    assert torch.equal(out.detach(), _sinkhorn_unbalanced_uniform_b_torch(C.detach(), a, valid, 0.005, 1.0, 0.01, 5))
