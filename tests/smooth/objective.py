"""The joint MAP objective, written independently of `fastdo.smooth.rts`, with the noise reference and the
Huber weights FROZEN (KA-MAP-REG).

The registered estimator evaluates each row's centre covariance at the CURRENT ESTIMATE's ray (and the window's
median measured range) and reweights by Huber, so Gauss-Newton does not minimise one fixed objective. Its fixed point
is claimed to be the minimiser of THIS objective with S_k and the weights frozen at the smoother's own output
(rays of the output poses, final weights). `frozen_inputs` builds those frozen quantities from a `SmoothResult`;
`residuals` is the objective; `check_fixed_point` minimises it with scipy from the smoother's output and returns how
far the poses move (metres). Used by tests/smooth/test_joint.py (synthetic windows).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from fastdo.smooth.lie import se3_exp, se3_log, so3_log


def _aniso(u: np.ndarray, s_par: float, s_perp: float) -> np.ndarray:
    u = u / np.linalg.norm(u)
    P = np.outer(u, u)
    return s_par ** 2 * P + s_perp ** 2 * (np.eye(3) - P)


def frozen_inputs(res: Any, cm: np.ndarray, hm: np.ndarray | None, nz: Any) -> dict[str, Any]:
    """Covariances at the OUTPUT's rays (estimate reference) and the output's final weights (NaN -> 0 = absent)."""
    ok = np.isfinite(cm).all(1)
    zbar = float(np.median(np.linalg.norm(cm[ok], axis=1)))
    S_c = np.array([_aniso(res.c[k], nz.a_par * zbar, nz.a_perp * zbar) for k in range(len(cm))])
    out: dict[str, Any] = {"S_c": S_c, "w_R": np.nan_to_num(res.w_R), "w_c": np.nan_to_num(res.w_c)}
    if hm is not None:
        ok_h = np.isfinite(hm).all(1)
        zh = float(np.median(np.linalg.norm(hm[ok_h], axis=1)))
        out["S_h"] = np.array([_aniso(res.hand[k], nz.a_par_h * zh, nz.a_perp_h * zh) for k in range(len(hm))])
        out["w_h"] = np.nan_to_num(res.w_h)
        out["w_g"] = np.nan_to_num(res.w_g)
    return out


def residuals(x: np.ndarray, Rm: np.ndarray, cm: np.ndarray, hm: np.ndarray | None, con: np.ndarray, nz: Any,
              fz: dict[str, Any], dt: np.ndarray) -> np.ndarray:
    """SE(3) process, rotation + anisotropic centre + wrist + grasp-rigidity residuals, each whitened by its frozen
    covariance and multiplied by sqrt(frozen weight). Rows with a zero weight or no measurement contribute 0."""
    T = len(cm)
    nh = hm is not None
    n = 18 if nh else 12
    X = x.reshape(T, n)
    Ts = [se3_exp(X[k, :6]) for k in range(T)]
    out = []
    for k in range(T):
        R, c = Ts[k][:3, :3], Ts[k][:3, 3]
        if fz["w_c"][k] > 0:
            out.append(np.sqrt(fz["w_R"][k]) * so3_log(R.T @ Rm[k]) / nz.s_rot)
            out.append(np.sqrt(fz["w_c"][k]) * np.linalg.solve(np.linalg.cholesky(fz["S_c"][k]), cm[k] - c))
        if nh and fz["w_h"][k] > 0:
            out.append(np.sqrt(fz["w_h"][k]) * np.linalg.solve(np.linalg.cholesky(fz["S_h"][k]), hm[k] - X[k, 12:15]))
        if nh and con[k] and fz["w_g"][k] > 0:
            om, v = X[k, 6:9], X[k, 9:12]
            g = R @ v - X[k, 15:18] - np.cross(R @ om, c - X[k, 12:15])
            out.append(np.sqrt(fz["w_g"][k]) * g / nz.q_c)
    q = np.array([nz.q_w] * 3 + [nz.q_v] * 3)
    for k in range(T - 1):
        D = float(dt[k])
        Q = np.block([[np.diag(q ** 2) * D ** 3 / 3, np.diag(q ** 2) * D ** 2 / 2],
                      [np.diag(q ** 2) * D ** 2 / 2, np.diag(q ** 2) * D]])
        f = np.r_[se3_log(np.linalg.inv(Ts[k]) @ Ts[k + 1]) - D * X[k, 6:12], X[k + 1, 6:12] - X[k, 6:12]]
        out.append(np.linalg.solve(np.linalg.cholesky(Q), f))
        if nh:
            Qh = nz.q_h ** 2 * np.block([[np.eye(3) * D ** 3 / 3, np.eye(3) * D ** 2 / 2],
                                         [np.eye(3) * D ** 2 / 2, np.eye(3) * D]])
            fh = np.r_[X[k + 1, 12:15] - X[k, 12:15] - D * X[k, 15:18], X[k + 1, 15:18] - X[k, 15:18]]
            out.append(np.linalg.solve(np.linalg.cholesky(Qh), fh))
    return np.concatenate(out)


def check_fixed_point(res: Any, Rm: np.ndarray, cm: np.ndarray, hm: np.ndarray | None, con: np.ndarray, nz: Any,
                      dt: np.ndarray) -> float:
    """Max movement (m) of the object centres and wrists when scipy minimises the frozen objective from the
    smoother's output (twists by finite differences of the output; the smoother does not return them)."""
    T = len(cm)
    nh = hm is not None
    n = 18 if nh else 12
    X = np.zeros((T, n))
    Ms = []
    for k in range(T):
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = res.R[k], res.c[k]
        Ms.append(M)
        X[k, :6] = se3_log(M)
        if nh:
            X[k, 12:15] = res.hand[k]
    for k in range(T - 1):
        X[k, 6:12] = se3_log(np.linalg.inv(Ms[k]) @ Ms[k + 1]) / dt[k]
        if nh:
            X[k, 15:18] = (res.hand[k + 1] - res.hand[k]) / dt[k]
    X[-1, 6:12] = X[-2, 6:12]
    if nh:
        X[-1, 15:18] = X[-2, 15:18]
    fz = frozen_inputs(res, cm, hm, nz)
    sol = least_squares(lambda x: residuals(x, Rm, cm, hm, con, nz, fz, dt), X.ravel(), method="lm", xtol=1e-15,
                        ftol=1e-15, gtol=1e-15, max_nfev=100000)
    Y = sol.x.reshape(T, n)
    c_opt = np.array([se3_exp(Y[k, :6])[:3, 3] for k in range(T)])
    move = float(np.abs(c_opt - res.c).max())
    if nh:
        move = max(move, float(np.abs(Y[:, 12:15] - res.hand).max()))
    return move
