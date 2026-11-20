"""Small dense convex QP by a primal-dual interior-point method (Mehrotra predictor-corrector).

    minimise   1/2 x^T H x + c^T x
    subject to A x = b          (m_eq rows)
               G x >= h         (m_in rows)

H must be symmetric positive semidefinite and the reduced KKT matrix nonsingular (H positive definite on the null
space of A, A of full row rank). Pure numpy; sized for tens to a few hundred variables (the grasp pre-load).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["QPResult", "solve_qp"]


@dataclass(frozen=True)
class QPResult:
    x: np.ndarray
    y: np.ndarray          # equality multipliers
    z: np.ndarray          # inequality multipliers (>= 0)
    status: str            # "optimal" | "max_iter" | "infeasible"
    iterations: int
    primal_residual: float
    dual_residual: float
    gap: float


def _kkt_solve(H, A, G, w, r1, r2, r3):
    """Solve [H + G^T W G, A^T; A, 0] [dx; -dy] = [r1 + G^T W r3 ... ] reduced system (see solve_qp)."""
    n, m = H.shape[0], A.shape[0]
    K = H + G.T @ (w[:, None] * G)
    rhs1 = r1 + G.T @ (w * r3)
    reg = 1e-12 * max(1.0, float(np.abs(np.diag(K)).max(initial=1.0)))
    if m == 0:
        M, rhs = K + reg * np.eye(n), rhs1
    else:
        M = np.block([[K + reg * np.eye(n), A.T], [A, -reg * np.eye(m)]])
        rhs = np.concatenate([rhs1, r2])
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return sol[:n], -sol[n:]


def solve_qp(H: np.ndarray, c: np.ndarray, A: np.ndarray | None = None, b: np.ndarray | None = None,
             G: np.ndarray | None = None, h: np.ndarray | None = None, tol: float = 1e-9,
             max_iter: int = 100) -> QPResult:
    """Solve the QP above; returns the last iterate with a status (never raises on non-convergence)."""
    H = np.asarray(H, float)
    n = H.shape[0]
    c = np.asarray(c, float)
    A = np.zeros((0, n)) if A is None else np.atleast_2d(np.asarray(A, float))
    b = np.zeros(0) if b is None else np.asarray(b, float)
    G = np.zeros((0, n)) if G is None else np.atleast_2d(np.asarray(G, float))
    h = np.zeros(0) if h is None else np.asarray(h, float)
    m_in = G.shape[0]
    x = np.zeros(n)
    y = np.zeros(A.shape[0])
    s = np.ones(m_in)
    z = np.ones(m_in)
    if m_in:   # start: least-squares feasible-ish x, slacks shifted positive
        s = np.maximum(G @ x - h, 1.0)
    scale = 1.0 + max(np.abs(c).max(initial=0), np.abs(b).max(initial=0), np.abs(h).max(initial=0))
    status, it = "max_iter", 0
    for it in range(1, max_iter + 1):
        rd = H @ x + c - A.T @ y - G.T @ z            # dual residual
        rp = A @ x - b                                 # equality residual
        ri = G @ x - s - h                             # inequality residual
        mu = float(s @ z) / m_in if m_in else 0.0
        pres = max(np.abs(rp).max(initial=0), np.abs(ri).max(initial=0))
        dres = np.abs(rd).max(initial=0)
        if pres <= tol * scale and dres <= tol * scale and mu <= tol * scale:
            status = "optimal"
            break
        w = z / s if m_in else np.zeros(0)

        def direction(rs: np.ndarray):
            # Newton on: H dx - A^T dy - G^T dz = -rd; A dx = -rp; G dx - ds = -ri; Z ds + S dz = -rs
            # ds = G dx + ri; dz = -(rs + Z ds)/S = -(rs/S) - w (G dx + ri)
            r1 = -rd - G.T @ (rs / s) if m_in else -rd
            r3 = -ri if m_in else np.zeros(0)
            dx, dy = _kkt_solve(H, A, G, w, r1, -rp, r3)
            ds = G @ dx + ri if m_in else np.zeros(0)
            dz = -(rs / s) - w * ds if m_in else np.zeros(0)
            return dx, dy, ds, dz

        def step(v, dv):
            neg = dv < 0
            return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0

        dx, dy, ds, dz = direction(s * z)                       # affine (predictor)
        if m_in:
            a_aff = min(step(s, ds), step(z, dz))
            mu_aff = float((s + a_aff * ds) @ (z + a_aff * dz)) / m_in
            sigma = (mu_aff / max(mu, 1e-300)) ** 3
            # never target complementarity far below the tolerance: a collapsed gap stalls the dual residual
            target = max(sigma * mu, 0.1 * tol * scale)
            dx, dy, ds, dz = direction(s * z + ds * dz - target)   # corrector
            alpha = 0.99 * min(step(s, ds), step(z, dz))
            alpha = min(alpha, 1.0)
        else:
            alpha = 1.0
        x = x + alpha * dx
        y = y + alpha * dy
        if m_in:
            s = np.maximum(s + alpha * ds, 1e-300)
            z = np.maximum(z + alpha * dz, 1e-300)
        if m_in and float(np.abs(x).max(initial=0)) > 1e12:
            status = "infeasible"
            break
    rd = H @ x + c - A.T @ y - G.T @ z
    pres = max(np.abs(A @ x - b).max(initial=0), np.abs(np.minimum(G @ x - h, 0)).max(initial=0) if m_in else 0.0)
    return QPResult(x=x, y=y, z=z, status=status, iterations=it, primal_residual=float(pres),
                    dual_residual=float(np.abs(rd).max(initial=0)), gap=float(s @ z) if m_in else 0.0)
