# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Regression tests (items F2, F3, F10): pooling denominators with integer ids and group mismatches, and test gaps
in the moment refusals and in alongray.change. The tests write no files."""

from __future__ import annotations

import numpy as np
import pytest

from alongray.change import summarize_changes, window_change
from alongray.pool import pool_rows, split_from_moments

MOM = {"pp": 1.0, "pz": 0.9, "zz": 1.0, "n": 5}


def test_integer_unit_ids_are_matched_to_the_string_manifest():
    rows = [{"unit_id": 1, "moments": MOM}, {"unit_id": 2, "moments": MOM}]
    out = pool_rows(rows, assigned=["1", "2"])["all"]
    assert (out["n_assigned"], out["n_scored"], out["n_missing"]) == (2, 2, 0)
    full = pool_rows(rows, assigned=["1", "2"], failure_policy="refuse")["all"]   # a complete cohort is accepted
    assert full["n_scored"] == 2


def test_a_row_in_a_different_group_than_the_manifest_is_refused():
    rows = [{"unit_id": "u1", "corpus": "D", "moments": MOM}, {"unit_id": "u2", "corpus": "D", "moments": MOM},
            {"unit_id": "u3", "corpus": "H", "moments": MOM}]
    with pytest.raises(ValueError, match="different group"):
        pool_rows(rows, group_key="corpus", assigned={"u1": "D", "u2": "H", "u3": "H"})
    ok = pool_rows(rows, group_key="corpus", assigned={"u1": "D", "u2": "D", "u3": "H"})
    assert ok["D"]["n_assigned"] == 2 and ok["H"]["n_missing"] == 0


def test_the_z1_refusal_is_its_own_check():
    with pytest.raises(ValueError, match="positive mean range"):
        split_from_moments(1.0, 0.9, 1.0, 5, z1=1.5, a1=0.9)


def test_constant_range_moments_whose_mean_error_exceeds_e_par_are_refused():
    """z1^2 = zz (constant range, the det = 0 branch), a1^2 <= pp, but (a1 - z1)^2 = 2.25 > e_par^2 = 1."""
    with pytest.raises(ValueError, match="squared mean error"):
        split_from_moments(4.0, 2.0, 1.0, 5, z1=1.0, a1=-0.5)


def _ids(n):
    return np.arange(n)


def test_a_nonfinite_truth_is_not_also_counted_as_a_degenerate_ray():
    g = np.column_stack([np.linspace(-0.1, 0.1, 5), np.zeros(5), np.linspace(0.5, 0.7, 5)])
    g[1] = np.nan                                                 # non-finite truth
    g[3] = 0.0                                                    # degenerate ray
    b = np.where(np.isfinite(g), 1.1 * g, 0.5) + [0, 0, 0.5]
    w = window_change(b, 1.2 * b, g, _ids(5), _ids(5), _ids(5), unit_id="u")
    assert (w.flow["n_nonfinite_gt"], w.flow["n_ray_degenerate"], w.flow["n_dropped"]) == (1, 1, 2)


def _const(unit, eb, ea, T=3):
    g = np.tile([[0.05, 0.0, 0.5]], (T, 1))
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    return window_change(g + eb * u, g + ea * u, g, _ids(T), _ids(T), _ids(T), unit_id=unit)


def test_zero_change_windows_enter_the_error_medians():
    """Windows with frames but no change still carry errors: the medians run over every window with frames."""
    ws = [_const("a", 0.01, 0.03), _const("b", 0.05, 0.05), _const("c", 0.02, 0.01)]
    assert ws[1].status == "zero_change"
    s = summarize_changes(ws)
    assert s["difference_of_median_centre"] == pytest.approx(0.03 - 0.02, abs=1e-12)
    assert s["median_paired_delta_centre"] == pytest.approx(0.0, abs=1e-12)


def test_a_share_of_exactly_0_9_counts_as_at_least_0_9():
    T = 4
    g = np.tile([[0.0, 0.0, 0.5]], (T, 1))
    d = np.tile([[2.0 ** -6, 0.0, 3 * 2.0 ** -6]], (T, 1))          # (d.u)^2 / |d|^2 = 9 / 10, exactly representable
    w = window_change(g, g + d, g, _ids(T), _ids(T), _ids(T), unit_id="u")
    assert w.share_along == 0.9
    assert summarize_changes([w])["n_windows_share_ge_0.9"] == 1
