"""SO(3)/SE(3) maps of the smoother: inverses and Jacobians against central differences, across the series
branches; each with a mutation that fires."""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.smooth import lie
from fastdo.smooth.lie import (
    se3_adjoint,
    se3_exp,
    se3_jr,
    se3_jr_inv,
    se3_log,
    so3_exp,
    so3_jl_inv,
    so3_jr,
    so3_log,
)

ANGLES = (0.0, 1e-7, 1e-4, 9.9e-3, 1.01e-2, 4.9e-2, 5.1e-2, 0.5, 2.0, 3.1)


def _num(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    cols = []
    for i in range(len(x)):
        d = np.zeros(len(x))
        d[i] = h
        cols.append((f(x + d) - f(x - d)) / (2 * h))
    return np.stack(cols, 1)


def _xi(th: float, seed: int) -> np.ndarray:
    xi = np.random.default_rng(seed).normal(size=6)
    xi[:3] *= th / np.linalg.norm(xi[:3])
    return xi


@pytest.mark.parametrize("th", ANGLES)
def test_exp_log_inverse_and_right_jacobians(th: float) -> None:
    xi = _xi(th, 1)
    T = se3_exp(xi)
    assert np.allclose(se3_log(T), xi, atol=1e-11)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-14)
    Jn = _num(lambda d: se3_log(np.linalg.inv(T) @ se3_exp(xi + d)), np.zeros(6))
    assert np.abs(Jn - se3_jr(xi)).max() < 1e-8
    assert np.allclose(se3_jr_inv(xi) @ se3_jr(xi), np.eye(6), atol=1e-12)
    Jn3 = _num(lambda d: so3_log(so3_exp(xi[:3]).T @ so3_exp(xi[:3] + d)), np.zeros(3))
    assert np.abs(Jn3 - so3_jr(xi[:3])).max() < 1e-8


@pytest.mark.parametrize("th", ANGLES)
def test_left_jacobian_inverse_of_the_rotation_residual(th: float) -> None:
    """Log(Exp(-phi) Exp(r)) ~ r - J_l^-1(r) phi: the measurement Jacobian of `fastdo.smooth.rts`."""
    r = _xi(th, 2)[:3]
    Jn = _num(lambda p: so3_log(so3_exp(-p) @ so3_exp(r)), np.zeros(3))
    assert np.abs(Jn + so3_jl_inv(r)).max() < (1e-6 if th > 3.0 else 1e-8)   # FD is ill-conditioned near pi


def test_adjoint_conjugates_the_exponential() -> None:
    T = se3_exp(_xi(0.7, 3))
    y = _xi(0.4, 4)
    assert np.allclose(T @ se3_exp(y) @ np.linalg.inv(T), se3_exp(se3_adjoint(T) @ y), atol=1e-12)


def test_log_near_pi_is_stable() -> None:
    a = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    for th in (np.pi - 1e-9, np.pi - 1e-7, np.pi):
        R = so3_exp(th * a)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-8)


def test_mutation_q_series_sign_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """The small-angle series of Barfoot's Q had its signs wrong in the first draft; flipping them back must fail."""
    xi = _xi(9e-3, 5)
    T = se3_exp(xi)
    good = np.abs(_num(lambda d: se3_log(np.linalg.inv(T) @ se3_exp(xi + d)), np.zeros(6)) - se3_jr(xi)).max()
    assert good < 1e-8
    orig = lie._q

    def bad_q(phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
        P, Rh = lie.so3_hat(phi), lie.so3_hat(rho)
        t2 = float(phi @ phi)
        a, b, c = 1 / 6 - t2 / 120, -1 / 24 + t2 / 720, -1 / 120 + t2 / 2520
        return (0.5 * Rh + a * (P @ Rh + Rh @ P + P @ Rh @ P) + b * (P @ P @ Rh + Rh @ P @ P - 3 * P @ Rh @ P)
                + c * (P @ Rh @ P @ P + P @ P @ Rh @ P))

    monkeypatch.setattr(lie, "_q", bad_q)
    bad = np.abs(_num(lambda d: se3_log(np.linalg.inv(T) @ se3_exp(xi + d)), np.zeros(6)) - lie.se3_jr(xi)).max()
    assert bad > 1e-6
    monkeypatch.setattr(lie, "_q", orig)
