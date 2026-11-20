"""DO AS I DO's per-step reward and tracking errors, vectorized over (world, row) samples.

The terms are `fastdo.physics.fd_problem.step_costs` (equal to DO AS I DO's `get_reward` to 1.1e-7 on 1,614
states), evaluated on arbitrary samples instead of one contiguous trajectory, so a batch of worlds at
different rows of different episodes is scored in one call. For executed control row s with x_{s+1} the state after
the step (DO AS I DO's `abs_step = s + 1`):

    cost_s = a_s + vel_scale b_s + pen_scale h_s^2 + ped_scale p_s,        get_reward(x_{s+1}) = -cost_s,
    a_s = || w (.) dq(x_{s+1}, ref_{s+1}) ||,  b_s = || qvel(x_{s+1}) - qvel_ref_{s+1} ||,
    h_s = max(0, pen_s - margin)  (max hand-object penetration of the step's collision pass),
    p_s = pedestal mismatch at sim step s+1 (in-hand / near-pedestal gates; zero while s+1 < W),

and the executor's rollout weights the warmup by m_s = warmup_mult when s+1 < W (else 1). `contact_terms` is the
contact part of `fd_problem._contact_pen` (no gradient) over MuJoCo's struct-of-arrays contact view.

Tracking errors are DO AS I DO's `compute_object_tracking_error` per frame: executed row s (state x_{s+1}) against
reference row s (the executor compares `qpos_traj[i]` with `qpos_ref[i]`), position norm and |quat_sub| angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from fastdo.physics.fd_problem import RewardSpec, quat_sub

__all__ = ["RowCosts", "contact_terms", "row_costs", "row_tracking"]


@dataclass(frozen=True, eq=False)
class RowCosts:
    """Per-sample terms; `reward` is DO AS I DO's get_reward value, `weighted` = mult * reward."""

    a: np.ndarray
    b: np.ndarray
    h: np.ndarray
    ped: np.ndarray
    cost: np.ndarray
    mult: np.ndarray
    reward: np.ndarray
    weighted: np.ndarray


def contact_terms(d: mujoco.MjData, rw: RewardSpec) -> tuple[float, bool, bool]:
    """(max hand-object penetration, hand touches the side's object, side's object touches a pedestal) of the
    contacts currently in `d` (after `mj_step`: the step's collision pass at the pre-step positions)."""
    n = int(d.ncon)
    if n == 0:
        return 0.0, False, False
    con = d.contact
    g = np.asarray(con.geom, dtype=np.int64).reshape(n, 2)
    dist = np.asarray(con.dist, dtype=np.float64).reshape(n)
    g1, g2 = g[:, 0], g[:, 1]
    hand1, hand2 = rw.hand_geom[g1], rw.hand_geom[g2]
    ho_pair = (hand1 & rw.object_geom[g2]) | (rw.object_geom[g1] & hand2)
    pen = np.where(np.isfinite(dist), np.maximum(-dist, 0.0), 0.0)
    best = float(pen[ho_pair].max()) if ho_pair.any() else 0.0
    ho = ped = False
    if rw.side_object_geom is not None:
        touch = dist <= 0.0
        so1, so2 = rw.side_object_geom[g1], rw.side_object_geom[g2]
        ho = bool((touch & ((hand1 & so2) | (so1 & hand2))).any())
        if rw.pedestal_geom is not None:
            ped = bool((touch & ((rw.pedestal_geom[g1] & so2) | (so1 & rw.pedestal_geom[g2]))).any())
    return best, ho, ped


def _ped_mismatch(rw: RewardSpec, rows: np.ndarray, ho: np.ndarray, ped: np.ndarray, W: int) -> np.ndarray:
    out = np.zeros(rows.shape[0])
    if rw.ped_scale <= 0 or rw.gate_in_hand is None or rw.pedestal_geom is None or not rw.pedestal_geom.any():
        return out
    k = rows + 1
    i = np.minimum(k, rw.gate_in_hand.shape[0] - 1)
    ih, npd = rw.gate_in_hand[i].astype(bool), rw.gate_near_ped[i].astype(bool)
    live = k >= W
    out = np.where(live & ih & ~npd & ~ho, 1.0, out)
    out = np.where(live & npd & ~ih & ~ped, 1.0, out)
    return out


def row_costs(rw: RewardSpec, obj_qadr: int, W: int, rows: np.ndarray, qpos: np.ndarray, qvel: np.ndarray,
              pen: np.ndarray, ho: np.ndarray, ped: np.ndarray) -> RowCosts:
    """Reward terms of B samples: control row `rows` (B,) int, post-step state `qpos` (B, nq) / `qvel` (B, nv), and
    the step's contact terms `pen` (B,), `ho`, `ped` (B,) bool (module docstring)."""
    rows = np.asarray(rows, np.int64)
    a0 = obj_qadr
    ref = rw.qpos_ref[rows + 1]
    dq = np.zeros((rows.shape[0], qvel.shape[1]))
    dq[:, :a0] = qpos[:, :a0] - ref[:, :a0]
    dq[:, a0:a0 + 3] = qpos[:, a0:a0 + 3] - ref[:, a0:a0 + 3]
    dq[:, a0 + 3:a0 + 6] = quat_sub(qpos[:, a0 + 3:a0 + 7], ref[:, a0 + 3:a0 + 7])
    a = np.linalg.norm(dq * rw.w_dq, axis=1)
    b = np.linalg.norm(qvel - rw.qvel_ref[rows + 1], axis=1)
    h = np.maximum(np.asarray(pen, np.float64) - rw.pen_margin, 0.0)
    pm = _ped_mismatch(rw, rows, np.asarray(ho, bool), np.asarray(ped, bool), W)
    cost = a + rw.vel_scale * b + rw.pen_scale * h * h + rw.ped_scale * pm
    mult = np.where(rows + 1 < W, rw.warmup_mult, 1.0)
    reward = -cost
    return RowCosts(a=a, b=b, h=h, ped=pm, cost=cost, mult=mult, reward=reward, weighted=mult * reward)


def row_tracking(rw: RewardSpec, obj_qadr: int, rows: np.ndarray, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position error m, rotation error rad) of executed rows `rows` (state x_{s+1}) against reference row s, as
    DO AS I DO's `compute_object_tracking_error` (reference in float32, its executor's dtype)."""
    rows = np.asarray(rows, np.int64)
    a0 = obj_qadr
    ref = rw.qpos_ref[rows].astype(np.float32).astype(np.float64)
    pos = np.linalg.norm(qpos[:, a0:a0 + 3] - ref[:, a0:a0 + 3], axis=1)
    rot = np.linalg.norm(quat_sub(qpos[:, a0 + 3:a0 + 7], ref[:, a0 + 3:a0 + 7]), axis=1)
    return pos, rot
