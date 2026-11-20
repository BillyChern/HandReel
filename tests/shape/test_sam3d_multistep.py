"""The AB2 multistep solver (CPU; no SAM 3D import)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch

from fastdo.shape.sam3d_multistep import AdamsBashforth2, cfg_regime, tree_map


def _euler(dyn, x, times):
    for t0, t1 in pairwise(times):
        x = x + dyn(x, t0) * (t1 - t0)
    return x


def _grid(n: int) -> torch.Tensor:
    return torch.linspace(0, 1, n + 1, dtype=torch.float64)


def test_exact_for_velocity_linear_in_t():
    """v = a + b t: AB2 integrates the linear interpolant exactly after its Euler start, so the error is the start
    step's alone (b h^2 / 2), independent of the rest of the grid."""
    a, b, n = 0.3, -1.7, 10
    x = AdamsBashforth2().solve(lambda x, t: a + b * t + 0 * x, torch.zeros((), dtype=torch.float64), _grid(n))
    h = 1 / n
    assert float(x) - (a + b / 2) == pytest.approx(-b * h**2 / 2, abs=1e-12)     # the Euler start's error, only


@pytest.mark.parametrize("dyn", [
    lambda x, t: -2.0 * x + torch.cos(3 * t),
    lambda x, t: torch.sin(4 * t) * x,
])
def test_second_order_convergence_vs_euler_first_order(dyn):
    x0 = torch.tensor([1.0, -0.5], dtype=torch.float64)
    ref = AdamsBashforth2().solve(dyn, x0, _grid(4096))
    errs_ab = [float((AdamsBashforth2().solve(dyn, x0, _grid(n)) - ref).abs().max()) for n in (16, 32, 64)]
    errs_eu = [float((_euler(dyn, x0, _grid(n)) - ref).abs().max()) for n in (16, 32, 64)]
    rate_ab = np.log2(errs_ab[1] / errs_ab[2])
    rate_eu = np.log2(errs_eu[1] / errs_eu[2])
    assert 1.7 < rate_ab < 2.4, errs_ab
    assert 0.8 < rate_eu < 1.2, errs_eu
    # the point of the lever: AB2 at 12 steps beats production's Euler at 25 on a smooth field
    assert float((AdamsBashforth2().solve(dyn, x0, _grid(12)) - ref).abs().max()) < \
        float((_euler(dyn, x0, _grid(25)) - ref).abs().max())


def test_restart_at_regime_change_matches_piecewise_integration():
    """A velocity that jumps at t = 0.5 (guidance switch): with the regime the solver restarts there, so the second
    half is integrated from x(0.5) exactly as a fresh AB2 run would; without it, the stale guided velocity leaks in."""
    def dyn(x, t):
        return (2.0 if float(t) <= 0.5 else -1.0) * torch.ones_like(x) * (1 + t)

    x0 = torch.zeros(3, dtype=torch.float64)
    grid = _grid(12)
    regime = cfg_regime((0, 500), 1000.0, 1.0)
    solver = AdamsBashforth2(regime)
    with_restart = solver.solve(dyn, x0, grid)
    assert solver.stats == {"steps": 12, "restarts": 2, "right_limits": 0}
    # steps start at nodes k/12; k <= 6 (t <= 0.5) are guided, k >= 7 not: a fresh AB2 over nodes 0..7, then a fresh
    # AB2 from x(7/12) over nodes 7..12, is the same computation
    first = AdamsBashforth2().solve(dyn, x0, grid[:8])
    second = AdamsBashforth2().solve(dyn, first, grid[7:])
    assert torch.allclose(with_restart, second, atol=1e-12, rtol=0)
    without = AdamsBashforth2().solve(dyn, x0, grid)
    assert not torch.allclose(without, with_restart)


def test_cfg_regime_matches_get_strength_rule():
    r = cfg_regime([0, 500], 1000.0, 1)
    assert r(torch.tensor(0.0)) and r(torch.tensor(0.5)) and not r(torch.tensor(0.5 + 1e-6)) and not r(torch.tensor(1.0))
    assert cfg_regime([0, 500], 1000.0, 0)(torch.tensor(0.1)) is False       # strength 0: never guided
    assert cfg_regime(None, 1000.0, 1.0)(0.2) is False


def test_pytree_latents_and_extra_args_pass_through():
    seen = []

    def dyn(x, t, d, *, cond):
        seen.append((float(d), cond))
        return {"a": torch.ones_like(x["a"]), "b": [2 * torch.ones_like(x["b"][0])]}

    x0 = {"a": torch.zeros(2, dtype=torch.float64), "b": [torch.zeros(1, dtype=torch.float64)]}
    out = AdamsBashforth2().solve(dyn, x0, _grid(4), 0.25, cond="c")
    assert torch.allclose(out["a"], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(out["b"][0], 2 * torch.ones(1, dtype=torch.float64))
    assert seen == [(0.25, "c")] * 4


def test_tree_map_structure():
    assert tree_map(lambda a, b: a + b, {"x": (1, [2])}, {"x": (10, [20])}) == {"x": (11, [22])}


def test_right_limit_restores_second_order_through_a_closed_switch():
    """The field jumps at t = 0.5 with the switch node INCLUDED in the guided side (SAM 3D's closed interval). Without
    the right limit the step (0.5, 0.5+h] uses the guided field: first order. With it: second order again."""
    def dyn(x, t):
        g = 2.0 if float(t) * 1000.0 <= 500.0 else -1.0
        return g * (1 + torch.cos(3 * t)) * torch.ones_like(x) + 0 * x

    def exact():                                          # integrate each regime separately, on a fine grid
        x = torch.zeros(1, dtype=torch.float64)
        x = AdamsBashforth2().solve(lambda x, t: 2.0 * (1 + torch.cos(3 * t)) * torch.ones_like(x), x,
                                    torch.linspace(0, 0.5, 4097, dtype=torch.float64))
        return AdamsBashforth2().solve(lambda x, t: -1.0 * (1 + torch.cos(3 * t)) * torch.ones_like(x), x,
                                       torch.linspace(0.5, 1.0, 4097, dtype=torch.float64))

    ref = exact()
    regime = cfg_regime((0, 500), 1000.0, 1.0)
    x0 = torch.zeros(1, dtype=torch.float64)
    plain = [float((AdamsBashforth2(regime).solve(dyn, x0, _grid(n)) - ref).abs()) for n in (16, 32, 64)]
    fixed = [float((AdamsBashforth2(regime, right_limit_eps=1e-9).solve(dyn, x0, _grid(n)) - ref).abs())
             for n in (16, 32, 64)]
    assert 0.8 < np.log2(plain[1] / plain[2]) < 1.2, plain            # first order: the switch step dominates
    assert 1.7 < np.log2(fixed[1] / fixed[2]) < 2.5, fixed            # second order restored
    assert fixed[0] < plain[0] / 5
    s = AdamsBashforth2(regime, right_limit_eps=1e-9)
    s.solve(dyn, x0, _grid(16))
    assert s.stats == {"steps": 16, "restarts": 2, "right_limits": 1}
