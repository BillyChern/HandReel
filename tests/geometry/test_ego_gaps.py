"""Known answers for the ego-motion gap filler (`fastdo.geometry.ego_gaps`)."""
from __future__ import annotations

import numpy as np
import pytest

from fastdo.geometry.ego_gaps import fill_gaps, split_sim3
from fastdo.smooth.lie import se3_exp

T = 40


def _sim3_path(xi: np.ndarray, s0: float = 1.0, ratio: float = 1.0) -> np.ndarray:
    """X_t = [s_t R_t | u_t] with (R_t, u_t) = Exp(t xi) (a constant-twist SE(3) geodesic) and s_t = s0 ratio^t."""
    out = np.zeros((T, 4, 4))
    for t in range(T):
        g = se3_exp(t * xi)
        out[t, :3, :3] = s0 * ratio ** t * g[:3, :3]
        out[t, :3, 3] = g[:3, 3]
        out[t, 3, 3] = 1.0
    return out


XI = np.array([0.02, -0.01, 0.03, 0.01, 0.002, -0.004])


def test_a_constant_twist_gap_is_filled_exactly() -> None:
    truth = _sim3_path(XI, 1.1, 1.003)
    valid = np.ones(T, bool)
    valid[10:15] = False
    out, ok, filled = fill_gaps(truth, valid, max_gap_rows=8)
    assert ok.all() and filled.sum() == 5 and filled[10:15].all()
    assert np.abs(out - truth).max() < 1e-9


def test_valid_rows_are_returned_bit_identical() -> None:
    truth = _sim3_path(XI, 0.9, 1.01)
    valid = np.ones(T, bool)
    valid[[3, 4, 20]] = False
    out, _, _ = fill_gaps(truth, valid, max_gap_rows=4)
    assert np.array_equal(out[valid], truth[valid])


def test_a_gap_longer_than_the_cap_is_left_invalid() -> None:
    truth = _sim3_path(XI)
    valid = np.ones(T, bool)
    valid[10:20] = False
    out, ok, filled = fill_gaps(truth, valid, max_gap_rows=9)
    assert not ok[10:20].any() and not filled.any()
    assert np.array_equal(out[10:20], truth[10:20])          # untouched: the caller still refuses those rows


def test_leading_and_trailing_gaps_are_never_extrapolated() -> None:
    truth = _sim3_path(XI)
    valid = np.ones(T, bool)
    valid[:2] = False
    valid[-3:] = False
    _, ok, filled = fill_gaps(truth, valid, max_gap_rows=10)
    assert not ok[:2].any() and not ok[-3:].any() and not filled.any()


def test_no_gap_is_a_no_op_and_zero_cap_fills_nothing() -> None:
    truth = _sim3_path(XI, 1.2, 0.998)
    out, ok, filled = fill_gaps(truth, np.ones(T, bool), max_gap_rows=5)
    assert np.array_equal(out, truth) and ok.all() and not filled.any()
    valid = np.ones(T, bool)
    valid[7] = False
    _, ok0, filled0 = fill_gaps(truth, valid, max_gap_rows=0)
    assert not ok0[7] and not filled0.any()


def test_split_recovers_scale_and_rotation() -> None:
    x = _sim3_path(XI, 1.3, 1.0)[5]
    s, R, u = split_sim3(x)
    assert s == pytest.approx(1.3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(s * R, x[:3, :3]) and np.allclose(u, x[:3, 3])


def test_interpolation_stays_on_the_rotation_group_for_a_large_turn() -> None:
    truth = _sim3_path(np.array([0.2, 0.1, -0.15, 0.05, 0.0, 0.02]))
    valid = np.ones(T, bool)
    valid[5:9] = False
    out, _, _ = fill_gaps(truth, valid, max_gap_rows=4)
    for t in range(5, 9):
        s, R, _ = split_sim3(out[t])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10) and s == pytest.approx(1.0)


def test_a_gap_exactly_at_the_cap_is_filled() -> None:
    truth = _sim3_path(XI)
    valid = np.ones(T, bool)
    valid[10:14] = False
    _, ok, filled = fill_gaps(truth, valid, max_gap_rows=4)
    assert ok.all() and filled.sum() == 4


def test_an_offset_body_frame_geodesic_is_filled_exactly() -> None:
    """X_t = G0 Exp(t xi): still a body-frame geodesic, but G0 does not commute with Exp(t xi), so a world-frame
    interpolation (Log(X_b X_a^-1)) would miss it."""
    g0 = se3_exp(np.array([0.7, -0.4, 0.3, 0.5, -0.2, 0.1]))
    truth = np.einsum("ij,tjk->tik", g0, _sim3_path(XI))
    valid = np.ones(T, bool)
    valid[12:17] = False
    out, ok, _ = fill_gaps(truth, valid, max_gap_rows=5)
    assert ok.all() and np.abs(out - truth).max() < 1e-9
