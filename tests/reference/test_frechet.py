"""Robust Frechet mean, drift test and alpha: known answers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from fastdo.reference import consolidate_rows
from fastdo.reference import frechet as FM
from fastdo.reference.so3 import angle_deg, exp, log

E1 = np.eye(3)[None]


def _rows(n, rng, Sigma, p0=None, O0=None):
    """`n` rows of a static pose with world-left noise drawn from `Sigma` (6x6)."""
    p0 = np.zeros(3) if p0 is None else p0
    O0 = np.eye(3) if O0 is None else O0
    L = np.linalg.cholesky(Sigma)
    x = rng.normal(size=(n, 6)) @ L.T
    return p0 + x[:, :3], exp(x[:, 3:]) @ O0


def _sigma(rng, pos=0.003, rot_deg=0.8):
    A = rng.normal(size=(6, 6))
    C = A @ A.T / 6 + np.eye(6)
    d = np.array([pos] * 3 + [np.radians(rot_deg)] * 3)
    return C * np.outer(d, d)


def test_static_noisy_object_matches_the_crlb(cfg) -> None:
    """iid Gaussian noise with a known (correlated) covariance: the estimate's Mahalanobis error against the
    information n Sigma^-1 averages 6 (chi^2_6) and its empirical covariance matches Sigma / n within 35%."""
    rng = np.random.default_rng(0)
    Sigma = _sigma(rng)
    n, trials = 30, 120
    errs = []
    O0 = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    p0 = np.array([0.1, -0.05, 0.6])
    for _ in range(trials):
        p, O = _rows(n, rng, Sigma, p0, O0)
        m = FM.robust_mean(p, O, np.broadcast_to(Sigma, (n, 6, 6)), E1, cfg, huber=False)
        assert m.converged
        errs.append(np.concatenate([m.p - p0, log(m.O @ O0.T)]))
    errs = np.array(errs)
    info = n * np.linalg.inv(Sigma)
    d2 = np.einsum("ti,ij,tj->t", errs, info, errs)
    assert 6 * 0.75 < d2.mean() < 6 * 1.3
    emp = np.cov(errs.T)
    ratio = np.diag(emp) / np.diag(Sigma / n)
    assert np.all((ratio > 0.65) & (ratio < 1.35))


def test_huber_mean_within_three_standard_errors(cfg) -> None:
    rng = np.random.default_rng(1)
    Sigma = _sigma(rng)
    n = 40
    p, O = _rows(n, rng, Sigma)
    m = FM.robust_mean(p, O, np.broadcast_to(Sigma, (n, 6, 6)), E1, cfg)
    se = np.sqrt(np.diag(Sigma) / n)
    err = np.concatenate([m.p, log(m.O)])
    assert np.all(np.abs(err) < 3 * se)


def test_weighted_mean_matches_the_closed_form_in_1d(cfg) -> None:
    """Position-only noise with per-row variances, identical rotations, all inliers: the mean is the inverse-variance
    weighted average (1e-9)."""
    rng = np.random.default_rng(2)
    n = 8
    var = rng.uniform(1e-6, 2e-5, n)
    x = rng.normal(0, 1, n) * np.sqrt(var)
    p = np.zeros((n, 3))
    p[:, 0] = x
    O = np.tile(np.eye(3), (n, 1, 1))
    S = np.tile(np.diag([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]), (n, 1, 1))
    S[:, 0, 0] = var
    m = FM.robust_mean(p, O, S, E1, cfg)
    assert np.max(m.e) < cfg.huber_k
    want = np.sum(x / var) / np.sum(1 / var)
    assert abs(m.p[0] - want) < 1e-9


def test_objective_decreases_monotonically(cfg) -> None:
    rng = np.random.default_rng(3)
    Sigma = _sigma(rng, 0.01, 5.0)
    p, O = _rows(25, rng, Sigma)
    p[3] += 0.3
    m = FM.robust_mean(p, O, np.broadcast_to(Sigma, (25, 6, 6)), E1, cfg)
    assert m.converged and np.all(np.diff(m.F_history) <= 1e-15)


def test_one_far_outlier_moves_the_mean_little(cfg) -> None:
    """One 20 cm row among 12 (design test 2; Huber's bounded influence is k * 5 mm / 11 = 1.6 mm here): the mean
    moves < 2 mm and the row is NOT replaced."""
    rng = np.random.default_rng(4)
    n = 12
    p, O = _rows(n, rng, np.diag([1e-6] * 3 + [1e-5] * 3))
    p[6] += [0.2, 0.0, 0.0]
    S0 = np.broadcast_to(FM.floor_covariance(cfg), (n, 6, 6))
    rows = np.arange(n)
    rep = consolidate_rows(rows, p, O, S0, np.ones(n), np.arange(n) / 10, E1, cfg)
    clean = np.delete(p, 6, axis=0).mean(0)
    assert rep.status == "consolidated"
    assert np.linalg.norm(rep.p - clean) < 0.002
    assert 6 in rep.outlier_kept and 6 not in rep.replaced


def test_a_moved_object_is_refused(cfg) -> None:
    """A 5 cm step in mid-segment (hands far): the drift test refuses `moved`."""
    rng = np.random.default_rng(5)
    n = 20
    p, O = _rows(n, rng, np.diag([4e-6] * 3 + [1e-5] * 3))
    p[10:] += [0.05, 0, 0]
    S0 = np.broadcast_to(FM.floor_covariance(cfg), (n, 6, 6))
    rep = consolidate_rows(np.arange(n), p, O, S0, np.ones(n), np.arange(n) / 10, E1, cfg)
    assert rep.status == "refused"            # bimodal: drift, few inliers or a flat Huber optimum (gn_failed)
    p2, O2 = _rows(n, rng, np.diag([4e-6] * 3 + [1e-5] * 3))
    p2 += np.linspace(0, 0.03, n)[:, None] * [1, 0, 0]          # slow 3 cm drift, every row within the cap
    rep2 = consolidate_rows(np.arange(n), p2, O2, S0, np.ones(n), np.arange(n) / 10, E1, cfg)
    assert rep2.status == "refused" and rep2.reason == "moved"


def test_static_segment_passes_the_drift_test(cfg) -> None:
    rng = np.random.default_rng(6)
    n = 20
    p, O = _rows(n, rng, np.diag([4e-6] * 3 + [1e-5] * 3))
    S0 = np.broadcast_to(FM.floor_covariance(cfg), (n, 6, 6))
    rep = consolidate_rows(np.arange(n), p, O, S0, np.ones(n), np.arange(n) / 10, E1, cfg)
    assert rep.status == "consolidated" and len(rep.replaced) == n


def test_quotient_mean_with_flipped_representatives(cfg) -> None:
    """Half the rows carry a 180 deg relabelling about z: the quotient mean recovers the true rotation."""
    rng = np.random.default_rng(7)
    G = np.stack([np.eye(3), Rotation.from_rotvec([0, 0, np.pi]).as_matrix()])
    n = 12
    O0 = Rotation.from_rotvec([0.2, 0.4, -0.1]).as_matrix()
    p, O = _rows(n, rng, np.diag([1e-6] * 3 + [1e-5] * 3), O0=O0)
    O[::2] = O[::2] @ G[1]
    m = FM.robust_mean(p, O, np.broadcast_to(FM.floor_covariance(cfg), (n, 6, 6)), G, cfg)
    best = min(angle_deg(m.O @ g, O0) for g in G)
    assert best < 0.5


def test_alpha_is_recovered(cfg) -> None:
    """True noise = 4 x the stored covariance -> alpha ~ 4 within 20% at 2,000 rows (dof-corrected)."""
    rng = np.random.default_rng(8)
    segs = []
    for _ in range(200):
        Sw = _sigma(rng, 0.004, 1.0)
        p, O = _rows(10, rng, 4.0 * Sw)
        segs.append((p, O, np.broadcast_to(Sw, (10, 6, 6)), E1))
    alpha, rows = FM.alpha_from_pool(segs, cfg)
    assert rows == 2000 and abs(alpha / 4.0 - 1) < 0.2


def test_ctrl_mean_uses_the_same_representatives(cfg) -> None:
    rng = np.random.default_rng(9)
    G = np.stack([np.eye(3), Rotation.from_rotvec([0, 0, np.pi]).as_matrix()])
    p, O = _rows(9, rng, np.diag([1e-6] * 3 + [1e-5] * 3))
    O[1::2] = O[1::2] @ G[1]
    labels = np.array([0, 1] * 4 + [0])
    pc, Oc = FM.ctrl_mean(p, O, labels, G)
    assert np.allclose(pc, np.median(p, axis=0)) and angle_deg(Oc, np.eye(3)) < 0.5
