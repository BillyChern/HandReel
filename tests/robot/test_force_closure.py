"""Known answers for fastdo.robot.force_closure: opposing contacts on a
sphere are in force closure and the same contacts on one side are not; the static hold residual of a cradle."""
from __future__ import annotations

import itertools

import numpy as np

from fastdo.robot.force_closure import contact_wrenches, epsilon_metric, static_hold_residual

R = 0.04
MU = 0.5


def _sphere_contacts(az_deg: list[float], el_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    a, e = np.radians(az_deg), np.radians(el_deg)
    p = R * np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.full_like(a, np.sin(e))], axis=1)
    return p, -p / R                                   # normals from the hand INTO the object: towards the centre


def test_three_opposing_fingertips_on_a_sphere_are_in_force_closure() -> None:
    p, n = _sphere_contacts([0.0, 120.0, 240.0])
    assert epsilon_metric(contact_wrenches(p, n, np.zeros(3), MU, R)) > 0.0


def test_the_same_contacts_on_one_side_are_not() -> None:
    p, n = _sphere_contacts([0.0, 30.0, 60.0])
    assert epsilon_metric(contact_wrenches(p, n, np.zeros(3), MU, R)) == 0.0


def test_a_single_contact_and_too_few_wrenches_give_zero() -> None:
    p, n = _sphere_contacts([0.0])
    assert epsilon_metric(contact_wrenches(p, n, np.zeros(3), MU, R)) == 0.0
    assert epsilon_metric(np.zeros((0, 6))) == 0.0


def test_epsilon_grows_with_friction() -> None:
    p, n = _sphere_contacts([0.0, 120.0, 240.0])
    lo = epsilon_metric(contact_wrenches(p, n, np.zeros(3), 0.3, R))
    hi = epsilon_metric(contact_wrenches(p, n, np.zeros(3), 0.8, R))
    assert 0.0 < lo < hi


def test_a_bottom_cradle_holds_gravity_but_is_not_in_force_closure() -> None:
    # three contacts under the sphere, pushing up: they hold the weight but cannot resist a downward pull
    p, n = _sphere_contacts([0.0, 120.0, 240.0], el_deg=-60.0)
    w = contact_wrenches(p, n, np.zeros(3), MU, R)
    up = np.array([0.0, 0.0, 0.2 * 9.81, 0.0, 0.0, 0.0])
    assert static_hold_residual(w, up) < 1e-6
    assert static_hold_residual(w, -up) > 0.99
    assert epsilon_metric(w) == 0.0


def test_one_sided_contacts_cannot_hold_a_sideways_load() -> None:
    p, n = _sphere_contacts([0.0])                    # one contact on +x pushing towards -x
    w = contact_wrenches(p, n, np.zeros(3), MU, R)
    assert static_hold_residual(w, np.array([1.0, 0, 0, 0, 0, 0])) > 0.99
    assert static_hold_residual(np.zeros((0, 6)), np.array([1.0, 0, 0, 0, 0, 0])) == 1.0


def test_signed_quality_is_epsilon_inside_and_minus_the_hull_distance_outside() -> None:
    from fastdo.robot.force_closure import signed_quality

    p, n = _sphere_contacts([0.0, 120.0, 240.0])
    w = contact_wrenches(p, n, np.zeros(3), MU, R)
    assert np.isclose(signed_quality(w), epsilon_metric(w))
    # one-sided sets are outside closure: Q < 0, and the further from opposition, the more negative
    q_arc = [signed_quality(contact_wrenches(*_sphere_contacts([0.0, 30.0, a]), np.zeros(3), MU, R))
             for a in (60.0, 90.0)]
    assert q_arc[0] < q_arc[1] < 0.0
    q1 = 1.0 / np.sqrt(1.0 + MU ** 2)                 # a lone pyramid's hull distance: |mean of its 4 unit edges|
    q_one = signed_quality(contact_wrenches(*_sphere_contacts([0.0]), np.zeros(3), MU, R))
    assert np.isclose(q_one, -q1)
    # continuity at the boundary: opening the third contact towards opposition raises Q monotonically
    qs = [signed_quality(contact_wrenches(*_sphere_contacts([0.0, 120.0, a]), np.zeros(3), MU, R))
          for a in (150.0, 180.0, 210.0, 240.0)]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(qs)), qs
    assert signed_quality(np.zeros((0, 6))) < 0.0
