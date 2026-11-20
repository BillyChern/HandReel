"""`lm_optimize_blocks(bounds=...)`, the projected active-set block LM the retarget stage's `joint_limits:
projected` runs on (the Sharpa joint box).

- On a separable quadratic whose unconstrained minimum lies outside the box, the bounded solve lands exactly on the
  projection of that minimum (the constrained optimum), and EVERY evaluated state is inside the box.
- Mutation check: the same problem without bounds leaves the box (so the feasibility assertion can fail).
- With smoothness coupling, entries at a bound stay there while the free entries still reach their coupled optimum.
- `bounds=None` is the unbounded loop, bitwise; malformed bounds raise.
"""

from __future__ import annotations

import pytest
import torch

from fastdo.solver.lm_blocks import (
    BlockLMConfig,
    BlockTerm,
    SmoothTerm,
    assemble_block_system,
    freeze_active_bounds,
    lm_optimize_blocks,
)

T, N = 5, 4
TARGET = torch.tensor([[1.5, -2.0, 0.3, 0.0]], dtype=torch.float64).expand(T, N)
LOWER = torch.tensor([-1.0, -1.0, -1.0, float("-inf")], dtype=torch.float64)
UPPER = torch.tensor([1.0, 1.0, 1.0, float("inf")], dtype=torch.float64)


def _quadratic(target: torch.Tensor, seen: list[torch.Tensor], smooth_weight: float = 0.0):
    def evaluate(x: torch.Tensor, refresh: bool) -> tuple[list[BlockTerm], SmoothTerm | None]:
        seen.append(x.clone())
        term = BlockTerm(residual=x - target, jacobian=torch.eye(N, dtype=x.dtype).expand(T, N, N),
                         weight=torch.ones_like(x))
        smooth = None
        if smooth_weight > 0.0:
            s = smooth_weight ** 0.5
            eye = torch.eye(N, dtype=x.dtype).expand(T - 1, N, N)
            smooth = SmoothTerm(residual=s * (x[1:] - x[:-1]), jac_k=-s * eye, jac_k1=s * eye,
                                weight=torch.ones(T - 1, N, dtype=x.dtype))
        return [term], smooth

    return evaluate


def _inside(x: torch.Tensor) -> bool:
    return bool(((x >= LOWER) & (x <= UPPER)).all())


def test_bounded_solve_lands_on_the_projected_minimum_and_never_leaves_the_box() -> None:
    seen: list[torch.Tensor] = []
    res = lm_optimize_blocks(torch.zeros(T, N, dtype=torch.float64), _quadratic(TARGET, seen),
                             BlockLMConfig(max_iter=50, rel_tol=1e-12), bounds=(LOWER, UPPER))
    expected = torch.maximum(torch.minimum(TARGET, UPPER), LOWER)
    assert torch.allclose(res.x, expected, atol=1e-9, rtol=0.0)
    assert (res.x[:, 0] == 1.0).all() and (res.x[:, 1] == -1.0).all()        # exactly AT the bound, not near it
    assert seen and all(_inside(x) for x in seen)


def test_mutation_without_bounds_the_same_problem_leaves_the_box() -> None:
    seen: list[torch.Tensor] = []
    res = lm_optimize_blocks(torch.zeros(T, N, dtype=torch.float64), _quadratic(TARGET, seen),
                             BlockLMConfig(max_iter=50, rel_tol=1e-12))
    assert not _inside(res.x) and torch.allclose(res.x, TARGET, atol=1e-6, rtol=0.0)


def test_coupled_problem_keeps_active_entries_at_their_bound() -> None:
    target = TARGET.clone()
    target[2:, 2] = 0.9                               # the free third column now varies over frames
    seen: list[torch.Tensor] = []
    res = lm_optimize_blocks(torch.zeros(T, N, dtype=torch.float64), _quadratic(target, seen, smooth_weight=2.0),
                             BlockLMConfig(max_iter=200, rel_tol=1e-14), bounds=(LOWER, UPPER))
    assert all(_inside(x) for x in seen)
    assert (res.x[:, 0] == 1.0).all() and (res.x[:, 1] == -1.0).all()
    # the unbounded coupled optimum of the third column is inside the box, so the bounded solve must reach it too
    free = lm_optimize_blocks(torch.zeros(T, N, dtype=torch.float64), _quadratic(target, [], smooth_weight=2.0),
                              BlockLMConfig(max_iter=200, rel_tol=1e-14))
    assert torch.allclose(res.x[:, 2:], free.x[:, 2:], atol=1e-6, rtol=0.0)


def test_an_infeasible_start_is_projected_first() -> None:
    seen: list[torch.Tensor] = []
    lm_optimize_blocks(torch.full((T, N), 5.0, dtype=torch.float64), _quadratic(TARGET, seen),
                       BlockLMConfig(max_iter=3), bounds=(LOWER, UPPER))
    assert _inside(seen[0])


def test_bounds_none_is_the_unbounded_loop_bitwise() -> None:
    a = lm_optimize_blocks(torch.zeros(T, N, dtype=torch.float64), _quadratic(TARGET, [], 1.0), BlockLMConfig())
    b = lm_optimize_blocks(torch.zeros(T, N, dtype=torch.float64), _quadratic(TARGET, [], 1.0), BlockLMConfig(),
                           bounds=None)
    assert torch.equal(a.x, b.x) and a.iterations == b.iterations and a.cost == b.cost


def test_freeze_active_bounds_zeroes_rows_and_columns() -> None:
    terms, smooth = _quadratic(TARGET, [], 1.0)(torch.zeros(T, N, dtype=torch.float64), True)
    diag, off, g = assemble_block_system(terms, smooth, T, N)
    active = torch.zeros(T, N, dtype=torch.bool)
    active[1, 2] = True
    d2, o2, g2 = freeze_active_bounds(diag, off, g, active)
    assert d2[1, 2, 2] == 1.0 and (d2[1, 2, [0, 1, 3]] == 0).all() and (d2[1, [0, 1, 3], 2] == 0).all()
    assert o2 is not None and (o2[0, :, 2] == 0).all() and (o2[1, 2, :] == 0).all() and g2[1, 2] == 0.0
    assert torch.equal(d2[0], diag[0]) and torch.equal(g2[3], g[3])          # other frames untouched
    assert freeze_active_bounds(diag, off, g, torch.zeros_like(active))[0] is diag


@pytest.mark.parametrize("bounds", [
    (torch.zeros(3, dtype=torch.float64), torch.ones(N, dtype=torch.float64)),
    (torch.ones(N, dtype=torch.float64), torch.zeros(N, dtype=torch.float64)),
    (torch.full((N,), float("nan"), dtype=torch.float64), torch.ones(N, dtype=torch.float64)),
])
def test_malformed_bounds_raise(bounds: tuple[torch.Tensor, torch.Tensor]) -> None:
    with pytest.raises(ValueError, match="bounds"):
        lm_optimize_blocks(torch.zeros(T, N, dtype=torch.float64), _quadratic(TARGET, []), BlockLMConfig(),
                           bounds=bounds)


def test_the_active_set_makes_a_coupled_bounded_step_exact() -> None:
    """Within-frame coupling: min |A x - b|^2 over x_0 <= 0 with A's columns correlated. Freezing the active x_0 makes
    each step the exact reduced solve, so the constrained optimum is reached in a few iterations; projecting the
    unconstrained step alone (x_1 computed as if x_0 moved) converges far more slowly (mutation-checked)."""
    a = torch.tensor([[1.0, 0.0], [0.9, 0.436]], dtype=torch.float64)
    b = a @ torch.tensor([1.0, 1.0], dtype=torch.float64)            # unconstrained optimum (1, 1): x_0 outside the box
    x1_star = float(a[:, 1] @ b / (a[:, 1] @ a[:, 1]))              # constrained optimum (0, x1_star)

    def evaluate(x: torch.Tensor, refresh: bool) -> tuple[list[BlockTerm], SmoothTerm | None]:
        r = x @ a.T - b
        return [BlockTerm(residual=r, jacobian=a.expand(x.shape[0], 2, 2), weight=torch.ones_like(r))], None

    lo = torch.tensor([float("-inf"), float("-inf")], dtype=torch.float64)
    hi = torch.tensor([0.0, float("inf")], dtype=torch.float64)
    res = lm_optimize_blocks(torch.zeros(1, 2, dtype=torch.float64), evaluate,
                             BlockLMConfig(max_iter=3, rel_tol=0.0), bounds=(lo, hi))
    assert res.x[0, 0] == 0.0 and abs(float(res.x[0, 1]) - x1_star) < 1e-4
