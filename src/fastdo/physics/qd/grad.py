"""Implicit differentiation of the quasi-dynamic step's solution map.

At a solution (v, z) of  min 1/2 v^T Q v - b^T v  s.t.  G v >= g  with active set A (z_i > s_i, s = G v - g, the
interior-point partition), the KKT conditions Q v - b - G_A^T z_A = 0, G_A v = g_A hold on a neighbourhood where A
does not change (strict complementarity). Differentiating in b:

    Q dv - G_A^T dz_A = db,   G_A dv = 0   =>   dv = N (N^T Q N)^{-1} N^T db,   N = an orthonormal basis of null(G_A)

(the null-space form stays exact when the pyramid rows of a contact, or of several coincident contacts, are linearly
dependent). One linear solve per step; d v / d u = (d v / d b) (d b / d u), and for the commanded hand velocity w
(u = q_eq^{-1}(q_a + h w)) d v / d w = (d v / d u) diag(h k_pos / gain).
"""

from __future__ import annotations

import numpy as np

from fastdo.physics.qd.model import QDModel, StepResult

__all__ = ["active_set", "dv_db", "dv_dcmd_vel", "dv_dctrl"]


def active_set(res: StepResult) -> np.ndarray:
    """Rows judged active at the IPM solution: multiplier larger than slack (both in the solver's scaled units)."""
    return res.active


def dv_db(res: StepResult, rank_tol: float = 1e-10) -> np.ndarray:
    """(nv, nv) sensitivity of the step's velocity to the linear term b."""
    Q = res.qp.Q
    act = active_set(res)
    GA = res.qp.G[act] if act.any() else np.zeros((0, Q.shape[0]))
    if GA.shape[0] == 0:
        return np.linalg.inv(Q)
    _, sv, vt = np.linalg.svd(GA, full_matrices=True)
    rank = int(np.sum(sv > rank_tol * max(1.0, float(sv.max(initial=0.0)))))
    N = vt[rank:].T
    if N.shape[1] == 0:
        return np.zeros_like(Q)
    return N @ np.linalg.solve(N.T @ Q @ N, N.T)


def dv_dctrl(res: StepResult) -> np.ndarray:
    """(nv, nu) d v / d u."""
    return dv_db(res) @ res.qp.db_du


def dv_dcmd_vel(qm: QDModel, res: StepResult) -> np.ndarray:
    """(nv, nu) d v / d w for the commanded hand joint velocity w (`model.ctrl_from_command_velocity`)."""
    return dv_dctrl(res) * (qm.cfg.h * qm.k_pos / qm.act_gain)[None, :]
