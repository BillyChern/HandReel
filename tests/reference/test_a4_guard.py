"""A4 (c): the along-ray gross-wander guard (`fastdo.reference.a4_guard`), synthetic known answers."""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.reference import a4_guard as G

CENTRE = np.array([0.1, -0.05, 0.9])
U = CENTRE / np.linalg.norm(CENTRE)


def test_running_median_matches_the_p7_convention() -> None:
    x = np.array([0.0, 10.0, 0.0, 0.0, 5.0, 6.0])
    assert G.running_median3(x).tolist() == [5.0, 0.0, 0.0, 0.0, 5.0, 5.5]


def test_statistic_is_the_largest_excursion_from_the_last_row() -> None:
    s = np.array([0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0])
    assert G.along_ray_p7(s) == pytest.approx(0.3)
    s = np.array([0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1])            # measured from the LAST row, not the first
    assert G.along_ray_p7(s) == pytest.approx(0.2)
    assert G.along_ray_p7(np.array([0.0, 0.0, 1.0, 0.0, 0.0])) == 0.0          # one spike is removed by the median
    assert G.along_ray_p7(np.array([0.0, 1.0, 0.0, 0.0, 0.0])) == 0.5          # the end window shrinks (P7's rule)


def test_only_the_along_ray_component_counts() -> None:
    """A static camera at the origin: a 4 cm wander along the ray gives ~38 mm (the end rows use 2-row medians); the
    same wander perpendicular to the ray gives < 2 mm (the ray to the lead's median tilts by ~1 deg)."""
    n = 20
    f = np.linspace(0, 1, n)
    v = np.cross(U, [0.0, 1.0, 0.0])
    v /= np.linalg.norm(v)
    cams = np.zeros((n, 3))
    a = G.along_ray_p7(G.along_ray_series(CENTRE + 0.04 * f[:, None] * U, cams))
    b = G.along_ray_p7(G.along_ray_series(CENTRE + 0.04 * f[:, None] * v, cams))
    assert a == pytest.approx(0.04 * 18 / 19, rel=1e-6)
    assert b < 0.002


def test_guard_decision_and_table_lookup() -> None:
    tau = {L: 100.0 + L for L in range(G.L_MIN, G.L_MAX + 1)}
    n = 20
    f = np.linspace(0, 1, n)
    cams = np.zeros((n, 3))
    d = G.along_ray_guard(CENTRE + 0.5 * f[:, None] * U, cams, tau)
    assert d["decided"] and d["refuse"] and d["tau_mm"] == 120.0 and d["stat_mm"] > 400
    d = G.along_ray_guard(CENTRE + 0.05 * f[:, None] * U, cams, tau)
    assert d["decided"] and not d["refuse"]
    d = G.along_ray_guard(np.tile(CENTRE, (60, 1)), np.zeros((60, 3)), tau)
    assert d["tau_L"] == 45 and d["tau_mm"] == 145.0
    assert not G.along_ray_guard(np.tile(CENTRE, (4, 1)), np.zeros((4, 3)), tau)["decided"]


def test_invariance_to_a_world_rigid_motion() -> None:
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    p = CENTRE + rng.normal(0, 0.03, (15, 3))
    cams = rng.normal(0, 0.05, (15, 3))
    Q = Rotation.from_rotvec([0.3, -1.0, 0.4]).as_matrix()
    t = np.array([1.0, 2.0, -3.0])
    a = G.along_ray_p7(G.along_ray_series(p, cams))
    b = G.along_ray_p7(G.along_ray_series(p @ Q.T + t, cams @ Q.T + t))
    assert a == pytest.approx(b, rel=1e-9)
