"""Second-order multistep ODE solver for SAM 3D's flow-matching samplers.

SAM 3D integrates dx/dt = v(x, t) from noise (t = 0) to data (t = 1) on a uniform grid with explicit Euler, one
velocity evaluation (NFE) per step; its local error is O(h^2) and its global error O(h). The variable-step two-step
Adams-Bashforth method (AB2) keeps ONE evaluation per step and reuses the previous step's velocity:

    x_{n+1} = x_n + h_n [ (1 + h_n / (2 h_{n-1})) v_n  -  (h_n / (2 h_{n-1})) v_{n-1} ],

which integrates the linear interpolant of v through (t_{n-1}, v_{n-1}), (t_n, v_n) exactly: local error O(h^3),
global O(h^2). At equal NFE it is a strictly higher-order method; at half the NFE its global error is of the order of
Euler's at the full count whenever v is smooth in t.

v is NOT smooth everywhere here. Classifier-free guidance is switched on for `interval[0] <= t * time_scale <=
interval[1]` only (SAM 3D's `get_strength`), so the field jumps where the regime changes; a multistep formula that
mixes a guided and an unguided velocity is inconsistent there. `AdamsBashforth2` therefore restarts, taking one Euler
step, at the first step and whenever `regime(t)` differs from the previous step's (the error of the jump itself is
the same O(h) that Euler's own grid places on it).

**Right limit at a switch node (`right_limit_eps`).** SAM 3D's guidance interval is closed (`lo <= t*scale <= hi`), so
on a grid with a node exactly at the switch (t = 0.5 for any even step count), the velocity evaluated AT that node is the
guided one, and a one-evaluation method applies it over the whole step (0.5, 0.5 + h], where the exact flow is unguided:
an O(h) error that no order of accuracy elsewhere removes (measured: AB2 at 12 steps only modestly better than Euler at
12). With `right_limit_eps` > 0, a step whose start node's regime differs from its interior's (tested at the step
midpoint) evaluates the field at t0 + eps (the right limit), and the restart follows the interior regime. eps enters
SAM 3D's time embedding as eps * time_scale, i.e. 1e-3 for eps 1e-6.

`solve_iter` is the one method SAM 3D's samplers call (`FlowMatching.generate_iter`,
`ShortCut.generate_iter` with its extra `d` argument, passed through `*args`). Latents may be tensors or pytrees of
tensors (dict / list / tuple). This module imports nothing from SAM 3D, so it is CPU-testable.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from itertools import pairwise
from typing import Any

__all__ = ["AdamsBashforth2", "cfg_regime", "tree_map"]


def tree_map(fn: Callable[..., Any], *trees: Any) -> Any:
    """`fn` applied leaf-wise over matching dict / list / tuple structures (tensors or arrays at the leaves)."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(tree_map(fn, *leaves) for leaves in zip(*trees, strict=True))
    return fn(*trees)


def cfg_regime(interval: Any, time_scale: float, strength: Any) -> Callable[[Any], Hashable]:
    """SAM 3D's guidance switch as a regime function: True where `get_strength` would guide at t (a single
    `(lo, hi)` interval on `t * time_scale`, and a positive scalar `strength`), else False."""
    if interval is None or isinstance(interval, dict) or not isinstance(strength, (int, float)) or strength <= 0:
        return lambda t: False

    lo, hi = float(interval[0]), float(interval[1])

    def regime(t: Any) -> bool:
        s = float(t) * float(time_scale)
        return lo <= s <= hi

    return regime


class AdamsBashforth2:
    """Variable-step AB2 with an Euler start and an Euler restart at every regime change (module docstring)."""

    def __init__(self, regime: Callable[[Any], Hashable] | None = None, right_limit_eps: float = 0.0) -> None:
        self.regime = regime
        self.right_limit_eps = float(right_limit_eps)
        self.stats = {"steps": 0, "restarts": 0, "right_limits": 0}

    def solve_iter(self, dynamics_fn: Callable[..., Any], x_init: Any, times: Any, *args: Any,
                   **kwargs: Any) -> Iterator[tuple[Any, Any]]:
        x = x_init
        prev: tuple[Any, Any, Hashable] | None = None
        for t0, t1 in pairwise(times):
            h = t1 - t0
            t_eval, r = t0, (self.regime(t0) if self.regime is not None else None)
            if self.regime is not None and self.right_limit_eps > 0:
                r_mid = self.regime((t0 + t1) / 2)
                if r_mid != r:                   # the start node sits on the switch: use the interior's field
                    t_eval, r = t0 + self.right_limit_eps, r_mid
                    self.stats["right_limits"] += 1
            v = dynamics_fn(x, t_eval, *args, **kwargs)
            self.stats["steps"] += 1
            if prev is None or prev[2] != r:
                self.stats["restarts"] += 1
                x = tree_map(lambda a, b, h=h: a + b * h, x, v)
            else:
                v_prev, h_prev, _ = prev
                c = h / (2 * h_prev)
                x = tree_map(lambda a, b, bp, c=c, h=h: a + ((1 + c) * b - c * bp) * h, x, v, v_prev)
            prev = (v, h, r)
            yield x, t0

    def solve(self, dynamics_fn: Callable[..., Any], x_init: Any, times: Any, *args: Any, **kwargs: Any) -> Any:
        x = x_init
        for x, _ in self.solve_iter(dynamics_fn, x_init, times, *args, **kwargs):
            pass
        return x
