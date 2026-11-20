# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""The best gauge move of the prediction, ``kappa_star``, and the least ``e_par`` it reaches, ``e_par_min``.

With ``A = <Z^2>``, ``B = <o^2>`` and ``e_par(kappa)^2 = (kappa lam - 1)^2 A + kappa^2 B``, the minimiser is
``kappa* = lam A / (lam^2 A + B)`` and the minimum ``e_par_min^2 = A B / (lam^2 A + B)``. The gauge part and the
off-orbit term split the uncorrected error; they are not the gain and the remainder of that best move.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alongray import e_par_min, e_par_split, kappa_star, score


def _trajectories(rng: np.random.Generator, lam: float, off_over_z: float, t: int,
                  perp: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Truth and prediction with least-squares scale ``lam`` and ``<o^2>^(1/2) = off_over_z <Z^2>^(1/2)`` exactly.

    The prediction is ``lam c_gt + o_t u_t + w_t`` with ``sum_t Z_t o_t = 0`` and ``w_t`` perpendicular to the ray.
    """
    d = rng.normal(size=(t, 3))
    d[:, 2] = np.abs(d[:, 2]) + 1.0
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    z = rng.uniform(0.3, 1.5, t)
    g = z[:, None] * u
    r = rng.normal(size=t)
    o = r - (np.sum(z * r) / np.sum(z * z)) * z
    o *= off_over_z * math.sqrt(np.mean(z * z)) / math.sqrt(np.mean(o * o))
    w = rng.normal(scale=perp, size=(t, 3))
    w -= np.einsum("ij,ij->i", w, u)[:, None] * u
    return lam * g + o[:, None] * u + w, g


def _random_case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lam = float(rng.uniform(0.3, 3.0))
    off_over_z = float(math.exp(rng.uniform(math.log(0.01), math.log(2.0))))
    return _trajectories(rng, lam, off_over_z, int(rng.integers(3, 61)))


@pytest.mark.parametrize("seed", range(8))
def test_e_par_of_kappa_is_the_closed_form(seed: int) -> None:
    p, g = _random_case(seed)
    sp = e_par_split(p, g)
    a, b = sp.z_rms ** 2, sp.off_orbit_part ** 2
    for k in np.random.default_rng(100 + seed).uniform(0.05, 4.0, 10):
        want = math.sqrt((k * sp.lam - 1.0) ** 2 * a + k * k * b)
        assert e_par_split(p, g, kappa=float(k)).e_par == pytest.approx(want, rel=1e-10)


@pytest.mark.filterwarnings("ignore:per-body scale lam:UserWarning")   # the grid's small kappa leave [0.1, 10]
@pytest.mark.parametrize("seed", range(8))
def test_kappa_star_and_e_par_min_match_a_grid_search(seed: int) -> None:
    """Property test: random lam, off-orbit size and T; a grid over kappa against the closed forms."""
    p, g = _random_case(seed)
    sp = e_par_split(p, g)
    ks, em = kappa_star(sp), e_par_min(sp)
    grid = np.linspace(1e-3, 3.0 / sp.lam, 3001)
    h = float(grid[1] - grid[0])
    e = np.array([e_par_split(p, g, kappa=float(k)).e_par for k in grid])
    assert e.min() >= em * (1.0 - 1e-9)                                   # nothing on the grid beats the minimum
    assert abs(float(grid[np.argmin(e)]) - ks) <= h                       # the grid's best is next to kappa*
    curv = sp.lam ** 2 * sp.z_rms ** 2 + sp.off_orbit_part ** 2           # e^2 = em^2 + curv (k - ks)^2
    assert e.min() ** 2 <= em ** 2 + curv * h * h / 4.0 + 1e-12
    assert e_par_split(p, g, kappa=ks).e_par == pytest.approx(em, rel=1e-10)


def test_known_answer() -> None:
    """lam = 2, A = 1, B = 1: kappa* = 2/5, e_par_min^2 = 1/5, oracle floor 1/2, e_par^2 = 2."""
    p, g = _trajectories(np.random.default_rng(7), 2.0, 1.0, 20)
    g, p = g / math.sqrt(np.mean(np.sum(g * g, axis=1))), p / math.sqrt(np.mean(np.sum(g * g, axis=1)))
    sp = e_par_split(p, g)
    assert (sp.lam, sp.z_rms, sp.off_orbit_part) == pytest.approx((2.0, 1.0, 1.0), rel=1e-12)
    assert kappa_star(sp) == pytest.approx(0.4, rel=1e-12)
    assert e_par_min(sp) == pytest.approx(math.sqrt(0.2), rel=1e-12)
    assert sp.oracle_floor == pytest.approx(0.5, rel=1e-12)
    assert sp.e_par == pytest.approx(math.sqrt(2.0), rel=1e-12)


@pytest.mark.parametrize(("lam", "off_over_z", "remainder_above_off"), [(2.0, 1.0, False), (0.5, 0.5, True)])
def test_the_parts_are_not_the_gain_and_remainder_of_the_best_move(lam: float, off_over_z: float,
                                                                   remainder_above_off: bool) -> None:
    """The off-orbit term can lie above or below the least error a gauge move reaches, and the gauge part is not
    the error that move removes, in RMS or in mean square."""
    p, g = _trajectories(np.random.default_rng(3), lam, off_over_z, 30)
    sp = e_par_split(p, g)
    em = e_par_min(sp)
    assert (em > sp.off_orbit_part * (1 + 1e-6)) is remainder_above_off
    assert (em < sp.off_orbit_part * (1 - 1e-6)) is (not remainder_above_off)
    assert abs((sp.e_par - em) - sp.gauge_part) > 1e-3 * sp.e_par
    assert abs((sp.e_par ** 2 - em ** 2) - sp.gauge_part ** 2) > 1e-3 * sp.e_par ** 2


@pytest.mark.parametrize("seed", range(6))
def test_e_par_min_is_below_the_oracle_floor_and_the_uncorrected_error(seed: int) -> None:
    p, g = _random_case(seed)
    sp = e_par_split(p, g)
    em = e_par_min(sp)
    assert em < sp.oracle_floor * (1.0 - 1e-12)
    assert em <= sp.e_par * (1.0 + 1e-12)
    assert 0.0 < kappa_star(sp) < 1.0 / sp.lam                            # it shrinks past the oracle when B > 0


@pytest.mark.parametrize("lam", [0.6, 1.0, 1.7])
def test_pure_scale_error_kappa_star_is_the_oracle_and_e_par_min_is_zero(lam: float) -> None:
    rng = np.random.default_rng(11)
    g = rng.normal(size=(12, 3)) * 0.1 + np.array([0.0, 0.0, 0.7])
    sp = e_par_split(lam * g, g)
    assert kappa_star(sp) == pytest.approx(1.0 / lam, rel=1e-12)
    assert e_par_min(sp) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_a_gauge_move_of_the_prediction_rescales_kappa_star_and_keeps_e_par_min(seed: int) -> None:
    p, g = _random_case(seed)
    sp, sk = e_par_split(p, g), e_par_split(p, g, kappa=1.37)
    assert kappa_star(sk) == pytest.approx(kappa_star(sp) / 1.37, rel=1e-10)
    assert e_par_min(sk) == pytest.approx(e_par_min(sp), rel=1e-10)
    assert sk.off_orbit_part == pytest.approx(1.37 * sp.off_orbit_part, rel=1e-10)    # the off-orbit term moves


def test_body_scores_are_accepted() -> None:
    p, g = _random_case(5)
    b = score(p, g)["object"]
    sp = e_par_split(p, g)
    assert kappa_star(b) == pytest.approx(kappa_star(sp), rel=1e-12)
    assert e_par_min(b) == pytest.approx(e_par_min(sp), rel=1e-12)


@pytest.mark.parametrize(("lam", "z_rms", "off"), [(0.0, 1.0, 0.1), (-1.0, 1.0, 0.1), (math.nan, 1.0, 0.1),
                                                   (1.0, 0.0, 0.1), (1.0, math.inf, 0.1), (1.0, 1.0, -0.1),
                                                   (1.0, 1.0, math.nan)])
@pytest.mark.parametrize("fn", [kappa_star, e_par_min])
def test_invalid_splits_are_refused(fn, lam: float, z_rms: float, off: float) -> None:
    class Split:
        pass

    s = Split()
    s.lam, s.z_rms, s.off_orbit_part = lam, z_rms, off
    with pytest.raises(ValueError):
        fn(s)
