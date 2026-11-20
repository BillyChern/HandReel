# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Known answers for alongray.change, the change between two predictions of one unit (the solve-change estimand).

Every expected value below is derived by hand from the construction, not by calling the code under test. The tests
write no files.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alongray.change import summarize_changes, window_change

RNG = np.random.default_rng(595)


def _gt(T: int) -> np.ndarray:
    """Off-axis camera-frame centres in metres, all well in front of the camera."""
    return np.column_stack([RNG.uniform(-0.15, 0.15, T), RNG.uniform(-0.1, 0.1, T), RNG.uniform(0.4, 0.9, T)])


def _ids(T: int) -> np.ndarray:
    return np.arange(0, 3 * T, 3)


def _perp(u: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each row of u."""
    a = np.where(np.abs(u[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    w = np.cross(u, a)
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def _wc(before, after, gt, ids=None, **kw):
    ids = _ids(len(gt)) if ids is None else ids
    return window_change(before, after, gt, ids, ids, ids, unit_id=kw.pop("unit_id", "u0"), **kw)


# --- the per-window share ------------------------------------------------------------------------------------


def test_pure_along_ray_change_has_share_one():
    g = _gt(12)
    w = _wc(1.1 * g, 1.25 * g, g)
    assert w.status == "scored"
    assert w.share_along == pytest.approx(1.0, abs=1e-12)
    assert w.sum_across_sq == pytest.approx(0.0, abs=1e-15)
    assert w.sum_change_sq == pytest.approx(float((0.15 ** 2) * (g ** 2).sum()), rel=1e-12)


def test_pure_across_ray_change_has_share_zero():
    g = _gt(10)
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    d = 0.02 * _perp(u)
    w = _wc(g.copy(), g + d, g)
    assert w.share_along == pytest.approx(0.0, abs=1e-12)
    assert w.sum_change_sq == pytest.approx(10 * 0.02 ** 2, rel=1e-12)


def test_share_is_the_ratio_of_sums_within_the_window():
    """Equal |d| at angles theta_t to the ray: share = mean cos^2 theta. Unequal |d|: the energy-weighted ratio."""
    g = _gt(4)
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    w_ = _perp(u)
    th = np.array([0.0, math.pi / 6, math.pi / 4, math.pi / 2])
    r = np.array([0.01, 0.01, 0.01, 0.01])
    d = r[:, None] * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w_)
    w = _wc(g.copy(), g + d, g)
    assert w.share_along == pytest.approx((1 + 0.75 + 0.5 + 0) / 4, abs=1e-12)
    r2 = np.array([0.03, 0.01, 0.01, 0.01])                       # one big along-ray frame dominates the window
    d2 = r2[:, None] * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w_)
    w2 = _wc(g.copy(), g + d2, g)
    expect = (9e-4 * 1 + 1e-4 * (0.75 + 0.5 + 0)) / (9e-4 + 3e-4)
    assert w2.share_along == pytest.approx(expect, abs=1e-12)
    assert w2.share_along != pytest.approx(float(np.mean(np.cos(th) ** 2)), abs=1e-3)


def test_gt_ray_and_before_ray_give_different_known_answers():
    """`before` sits off the true ray by a known angle alpha; the change lies along `before`'s own ray."""
    T = 6
    g = np.tile([[0.0, 0.0, 0.6]], (T, 1))
    alpha = 0.3
    b_dir = np.array([math.sin(alpha), 0.0, math.cos(alpha)])
    before = np.tile(0.6 * b_dir, (T, 1))
    after = before + 0.05 * b_dir
    wg = _wc(before, after, g, ray="gt")
    wb = _wc(before, after, g, ray="before")
    assert wb.share_along == pytest.approx(1.0, abs=1e-12)
    assert wg.share_along == pytest.approx(math.cos(alpha) ** 2, abs=1e-12)
    assert (wg.ray, wb.ray) == ("gt", "before")
    # the error terms stay on the TRUE ray whichever ray splits the change
    expect = abs(0.65 * math.cos(alpha) - 0.6) - abs(0.6 * math.cos(alpha) - 0.6)
    assert wg.delta_abs_along_mean == pytest.approx(expect, abs=1e-12)
    assert wb.delta_abs_along_mean == pytest.approx(expect, abs=1e-12)


def test_a_non_positive_min_range_is_refused():
    g = _gt(5)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="min_range"):
            _wc(g, 1.1 * g, g, min_range=bad)


def test_unknown_ray_is_refused():
    g = _gt(5)
    with pytest.raises(ValueError, match="ray"):
        _wc(g, 1.1 * g, g, ray="after")


# --- correspondence and time IDs -----------------------------------------------------------------------------


def test_rows_are_joined_on_frame_ids_not_on_position():
    g = _gt(8)
    ids = _ids(8)
    before, after = 1.05 * g, 1.2 * g
    ref = window_change(before, after, g, ids, ids, ids, unit_id="u")
    perm = RNG.permutation(8)
    w = window_change(before, after[perm], g, ids, ids[perm], ids, unit_id="u")
    assert w.share_along == ref.share_along
    assert w.delta_centre_mean == ref.delta_centre_mean
    assert w.frame_ids_used == tuple(int(i) for i in ids)


def test_a_mismatch_of_frame_id_sets_is_refused_never_intersected():
    g = _gt(6)
    ids = _ids(6)
    shifted = ids + 1
    with pytest.raises(ValueError, match="frame ids"):
        window_change(g, 1.1 * g, g, ids, shifted, ids, unit_id="u")
    with pytest.raises(ValueError, match="frame ids"):
        window_change(g, 1.1 * g, g[:-1], ids, ids, ids[:-1], unit_id="u")


def test_duplicate_or_non_integer_frame_ids_are_refused():
    g = _gt(5)
    ids = _ids(5)
    dup = ids.copy()
    dup[-1] = dup[0]
    with pytest.raises(ValueError, match="duplicate"):
        window_change(g, 1.1 * g, g, dup, dup, dup, unit_id="u")
    with pytest.raises(ValueError, match="integer"):
        window_change(g, 1.1 * g, g, ids + 0.5, ids + 0.5, ids + 0.5, unit_id="u")
    with pytest.raises(ValueError, match="frame ids"):
        window_change(g, 1.1 * g, g, ids[:-1], ids, ids, unit_id="u")


def test_only_t_by_3_arrays_are_accepted():
    g = _gt(5)
    ids = _ids(5)
    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        window_change(g[:, :2], g[:, :2], g[:, :2], ids, ids, ids, unit_id="u")
    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        window_change(g[None], g[None], g[None], ids, ids, ids, unit_id="u")


# --- paired finite masks and dropped denominators --------------------------------------------------------------


def test_a_nonfinite_row_in_any_input_drops_that_frame_from_every_quantity_and_is_counted():
    g = _gt(7)
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    before = g + 0.01 * u
    after = g + 0.03 * u + 0.01 * _perp(u)
    after_nan = after.copy()
    after_nan[2] = np.nan
    after_nan[4, 0] = np.nan
    before_nan = before.copy()
    before_nan[5, 1] = np.inf
    g_nan = g.copy()
    g_nan[0, 2] = np.nan
    w = _wc(before_nan, after_nan, g_nan)
    keep = [1, 3, 6]
    ref = _wc(before[keep], after[keep], g[keep], ids=_ids(7)[keep])
    assert w.flow["n_input"] == 7 and w.flow["n_used"] == 3 and w.flow["n_dropped"] == 4
    assert (w.flow["n_nonfinite_gt"], w.flow["n_nonfinite_before"], w.flow["n_nonfinite_after"]) == (1, 1, 2)
    assert w.dropped_ids == tuple(int(i) for i in _ids(7)[[0, 2, 4, 5]])
    assert w.share_along == pytest.approx(ref.share_along, abs=1e-14)
    assert w.delta_centre_mean == pytest.approx(ref.delta_centre_mean, abs=1e-14)
    assert w.delta_abs_along_mean == pytest.approx(ref.delta_abs_along_mean, abs=1e-14)
    # paired: the error means use the SAME frames for before and after (not each side's own finite frames)
    assert w.e_before_mean == pytest.approx(0.01, abs=1e-12)


def test_a_true_point_at_the_camera_centre_is_dropped_and_counted():
    g = _gt(5)
    g[3] = 0.0
    w = _wc(1.1 * np.where(g == 0, 0.5, g), 1.2 * np.where(g == 0, 0.5, g), g)
    assert w.flow["n_ray_degenerate"] == 1 and w.flow["n_used"] == 4


def test_a_window_with_no_change_has_no_share_and_says_so():
    g = _gt(5)
    w = _wc(1.1 * g, 1.1 * g, g)
    assert w.status == "zero_change" and w.share_along is None
    assert w.flow["n_zero_change"] == 5


def test_a_window_with_no_usable_frame_is_not_scored():
    g = _gt(3)
    w = _wc(np.full_like(g, np.nan), g, g)
    assert w.status == "no_frames" and w.share_along is None and w.flow["n_used"] == 0


# --- signed and absolute components ----------------------------------------------------------------------------


def test_signed_and_absolute_along_ray_components_are_separate():
    """before 10 cm too far, after 10 cm too near: |along| is unchanged, the signed change is -20 cm."""
    g = _gt(9)
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    w = _wc(g + 0.1 * u, g - 0.1 * u, g)
    assert w.delta_abs_along_mean == pytest.approx(0.0, abs=1e-12)
    assert w.delta_centre_mean == pytest.approx(0.0, abs=1e-12)
    assert w.signed_change_along_mean == pytest.approx(-0.2, abs=1e-12)
    assert w.abs_change_along_mean == pytest.approx(0.2, abs=1e-12)
    assert w.signed_along_before_mean == pytest.approx(0.1, abs=1e-12)
    assert w.signed_along_after_mean == pytest.approx(-0.1, abs=1e-12)


# --- pooling over windows ------------------------------------------------------------------------------------


def test_median_of_window_shares_differs_from_the_pooled_ratio_and_both_are_labelled():
    g = np.tile([[0.0, 0.0, 0.5]], (4, 1)) + np.column_stack([np.zeros(4), np.zeros(4), 0.05 * np.arange(4)])
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    p = _perp(u)
    ws = []
    # per-frame change r (cos th u + sin th p) with constant r and th: share cos^2 th, sum r^2 T
    for name, r, th in (("a", 0.01, math.acos(math.sqrt(0.2))), ("b", 0.01, math.acos(math.sqrt(0.92))),
                        ("c", 0.05, 0.0)):
        d = r * (math.cos(th) * u + math.sin(th) * p)
        ws.append(_wc(g.copy(), g + d, g, unit_id=name))
    s = summarize_changes(ws)
    assert s["median_window_share"] == pytest.approx(0.92, abs=1e-12)
    tot = 4 * (0.01 ** 2 + 0.01 ** 2 + 0.05 ** 2)
    along = 4 * (0.2 * 0.01 ** 2 + 0.92 * 0.01 ** 2 + 0.05 ** 2)
    assert s["pooled_share_ratio_of_sums"] == pytest.approx(along / tot, abs=1e-12)
    assert s["n_windows_share_ge_0.9"] == 2
    assert "median over windows" in s["estimand"]["median_window_share"]
    assert "not the paper" in s["estimand"]["pooled_share_ratio_of_sums"]


def test_median_of_paired_differences_is_not_the_difference_of_medians():
    """e_before = (1, 2, 3) cm, e_after = (3, 1, 2) cm: paired deltas (2, -1, -1) -> median -1 cm; medians 2 and 2."""
    g = np.tile([[0.05, 0.0, 0.5]], (3, 1))
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    ws = [_wc(g + eb * u, g + ea * u, g, unit_id=n)
          for n, eb, ea in (("a", 0.01, 0.03), ("b", 0.02, 0.01), ("c", 0.03, 0.02))]
    s = summarize_changes(ws)
    assert s["median_paired_delta_centre"] == pytest.approx(-0.01, abs=1e-12)
    assert s["difference_of_median_centre"] == pytest.approx(0.0, abs=1e-12)
    assert s["n_windows_worse"] == 1 and s["n_windows_better"] == 2


def test_spearman_of_paired_deltas_known_answers():
    g = np.tile([[0.05, 0.0, 0.5]], (3, 1))
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    p = _perp(u)
    # along-ray error deltas increase with the centre deltas -> rho(de, da) = 1; the perp deltas decrease -> -1
    ws = []
    for i, (da, dp) in enumerate(((0.01, 0.03), (0.02, 0.02), (0.04, 0.01), (0.08, 0.0))):
        before = g + 0.1 * u + 0.05 * p
        after = g + (0.1 + da) * u + (0.05 + dp) * p
        ws.append(_wc(before, after, g, unit_id=f"w{i}"))
    s = summarize_changes(ws)
    de = [w.delta_centre_mean for w in ws]
    assert de == sorted(de)                                   # the construction's premise
    assert s["spearman_delta_centre_vs_delta_abs_along"] == pytest.approx(1.0, abs=1e-12)
    assert s["spearman_delta_centre_vs_delta_perp"] == pytest.approx(-1.0, abs=1e-12)


def test_denominators_assigned_scored_zero_change_no_frames_and_missing():
    g = _gt(5)
    ws = [_wc(1.1 * g, 1.2 * g, g, unit_id="a"), _wc(1.1 * g, 1.1 * g, g, unit_id="b"),
          _wc(np.full_like(g, np.nan), g, g, unit_id="c")]
    s = summarize_changes(ws, assigned=["a", "b", "c", "d", "e"])
    assert (s["n_assigned"], s["n_windows"], s["n_scored"], s["n_zero_change"], s["n_no_frames"], s["n_missing"]) == \
        (5, 3, 1, 1, 1, 2)
    assert s["missing_ids"] == ["d", "e"]
    assert s["frames"]["n_input"] == 15 and s["frames"]["n_used"] == 10 and s["frames"]["n_dropped"] == 5


def test_duplicate_windows_and_unassigned_windows_are_refused():
    g = _gt(5)
    w = _wc(1.1 * g, 1.2 * g, g, unit_id="a")
    with pytest.raises(ValueError, match="duplicate"):
        summarize_changes([w, w])
    with pytest.raises(ValueError, match="not assigned"):
        summarize_changes([w], assigned=["b"])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_changes([w], assigned=["a", "a"])


def test_windows_scored_with_different_rays_are_not_pooled():
    g = _gt(5)
    with pytest.raises(ValueError, match="ray"):
        summarize_changes([_wc(1.1 * g, 1.2 * g, g, unit_id="a", ray="gt"),
                           _wc(1.1 * g, 1.2 * g, g, unit_id="b", ray="before")])


def test_summary_records_its_conventions():
    g = _gt(5)
    s = summarize_changes([_wc(1.1 * g, 1.2 * g, g, unit_id="a")])
    conv = s["conventions"]
    assert conv["frame"] == "camera frame, camera centre at the origin"
    assert conv["units"] == "metres"
    assert conv["change_ray"] == "gt"
    assert conv["error_ray"] == "gt"
    assert "identical" in conv["correspondence"]
    assert s["spearman_ties"] == "average ranks"
