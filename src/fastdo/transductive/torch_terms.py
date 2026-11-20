"""Batched torch versions of the transductive per-row terms, for the GPU backend (one clip = one pool of worlds).

Op-for-op the numpy definitions of `fastdo.transductive.reward` (DO AS I DO's reward, tracking errors, contact
terms) and `fastdo.transductive.obs` (the observation), evaluated for a pool of worlds that sit at different rows of
the same clip: every reference row is a gather by the per-world row index. `PoolRef` holds one clip's reference and
masks on the device. The numpy versions remain the reference implementation; `tests/transductive/test_batch_group.py`
checks these against them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.config import SUCCESS_POS_M, SUCCESS_ROT_RAD, EnvConfig
from fastdo.transductive.obs import delta_scale, joint_scale

__all__ = ["PoolRef", "StackedRef", "contact_terms_t", "observe_s", "observe_t", "quat_sub_t", "row_terms_s",
           "row_terms_t"]


def quat_sub_t(qa: Tensor, qb: Tensor) -> Tensor:
    """`fastdo.physics.fd_problem.quat_sub` (DO AS I DO's convention): axis-angle of qb^-1 qa, (..., 3)."""
    w1, x1, y1, z1 = qb[..., 0], -qb[..., 1], -qb[..., 2], -qb[..., 3]
    w2, x2, y2, z2 = qa.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    v = torch.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)
    s = torch.linalg.vector_norm(v, dim=-1)
    ang = 2.0 * torch.atan2(s, w)
    ang = torch.where(ang > torch.pi, ang - 2.0 * torch.pi, ang)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.where((s > 0)[..., None], v * (ang / safe)[..., None], torch.zeros_like(v))


@dataclass(eq=False)
class PoolRef:
    """One clip's reference and masks on `device` (dtype float of the backend)."""

    qpos_ref: Tensor
    qvel_ref: Tensor
    ctrl_ref: Tensor
    w_dq: Tensor
    hand: Tensor           # (ngeom,) bool
    obj: Tensor
    side_obj: Tensor
    ped_geom: Tensor
    gate_in_hand: Tensor   # (Tref,) bool (zeros when the clip has no gate)
    gate_near_ped: Tensor
    ped_on: bool
    a0: int
    R: int
    W: int
    T: int
    max_sim: int
    vel_scale: float
    pen_scale: float
    pen_margin: float
    ped_scale: float
    warmup_mult: float

    @staticmethod
    def build(sp: ClipSpec, device: torch.device | str, dtype: torch.dtype) -> PoolRef:
        rw = sp.rw
        f = lambda x: torch.as_tensor(np.asarray(x, np.float64), dtype=dtype, device=device)
        b = lambda x: torch.as_tensor(np.asarray(x, bool), device=device)
        ng = sp.ex.model.ngeom
        zeros_g = np.zeros(ng, bool)
        gate_ok = rw.gate_in_hand is not None and rw.pedestal_geom is not None and bool(np.any(rw.pedestal_geom))
        n_ref = rw.qpos_ref.shape[0]
        return PoolRef(qpos_ref=f(rw.qpos_ref), qvel_ref=f(rw.qvel_ref), ctrl_ref=f(sp.ctrl_ref), w_dq=f(rw.w_dq),
                       hand=b(rw.hand_geom), obj=b(rw.object_geom),
                       side_obj=b(rw.side_object_geom if rw.side_object_geom is not None else zeros_g),
                       ped_geom=b(rw.pedestal_geom if rw.pedestal_geom is not None else zeros_g),
                       gate_in_hand=b(rw.gate_in_hand if gate_ok else np.zeros(n_ref, bool)),
                       gate_near_ped=b(rw.gate_near_ped if gate_ok else np.zeros(n_ref, bool)),
                       ped_on=bool(gate_ok and rw.ped_scale > 0), a0=sp.ex.obj_qadr, R=sp.n_robot, W=sp.W, T=sp.T,
                       max_sim=sp.max_sim_steps, vel_scale=float(rw.vel_scale), pen_scale=float(rw.pen_scale),
                       pen_margin=float(rw.pen_margin), ped_scale=float(rw.ped_scale),
                       warmup_mult=float(rw.warmup_mult))


def contact_terms_t(p: PoolRef, dist: Tensor, geom: Tensor, worldid: Tensor, nacon: Tensor, n: int
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """Per-world (max hand-object penetration, hand touches side object, side object touches a pedestal) from a
    flat contact buffer (`reward.contact_terms` per world; entries beyond `nacon` ignored)."""
    L = dist.shape[0]
    valid = torch.arange(L, device=dist.device) < nacon.reshape(()).to(torch.long)
    ng = p.hand.shape[0]
    g1, g2 = geom[:, 0].long().clamp(0, ng - 1), geom[:, 1].long().clamp(0, ng - 1)
    wid = worldid.long().clamp(0, n - 1)
    d = dist.to(p.qpos_ref.dtype)
    h1, h2 = p.hand[g1], p.hand[g2]
    ho_pair = (h1 & p.obj[g2]) | (p.obj[g1] & h2)
    pen = torch.where(torch.isfinite(d), torch.clamp(-d, min=0.0), torch.zeros_like(d))
    pen = torch.where(valid & ho_pair, pen, torch.zeros_like(pen))
    best = torch.zeros(n, dtype=d.dtype, device=d.device).scatter_reduce(0, wid, pen, reduce="amax")
    touch = valid & (d <= 0.0)
    s1, s2 = p.side_obj[g1], p.side_obj[g2]
    ho = touch & ((h1 & s2) | (s1 & h2))
    pd = touch & ((p.ped_geom[g1] & s2) | (s1 & p.ped_geom[g2]))
    z = torch.zeros(n, dtype=d.dtype, device=d.device)
    return (best, z.scatter_reduce(0, wid, ho.to(d.dtype), reduce="amax") > 0.5,
            z.scatter_reduce(0, wid, pd.to(d.dtype), reduce="amax") > 0.5)


def row_terms_t(p: PoolRef, rows: Tensor, qpos: Tensor, qvel: Tensor, pen: Tensor, ho: Tensor, ped: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
    """(weighted reward m_s get_reward_s, position error, rotation error) of each world's executed row `rows`."""
    a0 = p.a0
    ref = p.qpos_ref[rows + 1]
    dq = torch.cat([qpos[:, :a0] - ref[:, :a0], qpos[:, a0:a0 + 3] - ref[:, a0:a0 + 3],
                    quat_sub_t(qpos[:, a0 + 3:a0 + 7], ref[:, a0 + 3:a0 + 7])], dim=1)
    a = torch.linalg.vector_norm(dq * p.w_dq, dim=1)
    b = torch.linalg.vector_norm(qvel - p.qvel_ref[rows + 1], dim=1)
    h = torch.clamp(pen - p.pen_margin, min=0.0)
    cost = a + p.vel_scale * b + p.pen_scale * h * h
    if p.ped_on:
        k = rows + 1
        i = torch.clamp(k, max=p.gate_in_hand.shape[0] - 1)
        ih, npd = p.gate_in_hand[i], p.gate_near_ped[i]
        live = k >= p.W
        mis = (live & ih & ~npd & ~ho) | (live & npd & ~ih & ~ped)
        cost = cost + p.ped_scale * mis.to(cost.dtype)
    mult = torch.where(rows + 1 < p.W, torch.full_like(cost, p.warmup_mult), torch.ones_like(cost))
    tref = p.qpos_ref[rows].to(torch.float32).to(cost.dtype)
    pos = torch.linalg.vector_norm(qpos[:, a0:a0 + 3] - tref[:, a0:a0 + 3], dim=1)
    rot = torch.linalg.vector_norm(quat_sub_t(qpos[:, a0 + 3:a0 + 7], tref[:, a0 + 3:a0 + 7]), dim=1)
    return -mult * cost, pos, rot


def observe_t(cfg: EnvConfig, p: PoolRef, rows: Tensor, qpos: Tensor, qvel: Tensor, delta_prev: Tensor,
              sums: Tensor, contact: Tensor, clip_index: int, n_clips: int, alpha: Tensor) -> Tensor:
    """`fastdo.transductive.obs.observe` for every world of the pool (float32 (n, obs_dim))."""
    R, a0, dt, dev = p.R, p.a0, qpos.dtype, qpos.device
    last = p.qpos_ref.shape[0] - 1
    js = torch.as_tensor(joint_scale(cfg, R), dtype=dt, device=dev)
    vs = torch.full((R,), 10.0, dtype=dt, device=dev)
    vs[:3], vs[3:6] = 1.0, 5.0
    ps, rs = cfg.obs_scale_pos, cfg.obs_scale_rot
    q, ob_p, ob_q = qpos[:, :R], qpos[:, a0:a0 + 3], qpos[:, a0 + 3:a0 + 7]
    r = p.qpos_ref[torch.clamp(rows, max=last)]
    parts = [(ob_p - r[:, a0:a0 + 3]) / ps, quat_sub_t(ob_q, r[:, a0 + 3:a0 + 7]) / rs,
             qvel[:, a0:a0 + 3], qvel[:, a0 + 3:a0 + 6] / 5.0, (q - r[:, :R]) / js, qvel[:, :R] / vs,
             (p.ctrl_ref[torch.clamp(rows + cfg.rows_per_step, max=p.ctrl_ref.shape[0] - 1)] - q) / js]
    for d in cfg.future_rows:
        f = p.qpos_ref[torch.clamp(rows + d, max=last)]
        parts += [(f[:, a0:a0 + 3] - ob_p) / ps, quat_sub_t(f[:, a0 + 3:a0 + 7], ob_q) / rs, (f[:, :R] - q) / js]
    T = float(p.T)
    ds = torch.as_tensor(delta_scale(cfg, R), dtype=dt, device=dev)
    ph = torch.stack([torch.clamp(p.W - rows, min=0).to(dt) / max(p.W, 1), rows.to(dt) / T], dim=1)
    budget = torch.stack([sums[:, 0] / (SUCCESS_POS_M * T), sums[:, 1] / (SUCCESS_ROT_RAD * T)], dim=1)
    cont = torch.stack([contact[:, 0] / 0.005, contact[:, 1], contact[:, 2]], dim=1)
    parts += [(ob_p - q[:, :3]) / (2 * ps), delta_prev / ds, cont, ph, budget,
              (alpha[:, None] if cfg.assist_observed else torch.zeros_like(alpha[:, None])).to(dt)]
    if cfg.clip_onehot:
        oh = torch.zeros(qpos.shape[0], n_clips, dtype=dt, device=dev)
        oh[:, clip_index] = 1.0
        parts.append(oh)
    return torch.clamp(torch.cat(parts, dim=1), -cfg.obs_clip, cfg.obs_clip).to(torch.float32)


@dataclass(eq=False)
class StackedRef:
    """The references of C clips of one hand side, stacked on `device` for per-world gathers by (clip, row): rows
    past a clip's end repeat its last row (so a clamp to the stacked length equals the clip's own clamp)."""

    qpos_ref: Tensor       # (C, Tm, nq)
    qvel_ref: Tensor       # (C, Tm, nv)
    ctrl_ref: Tensor       # (C, Tm, nu)
    w_dq: Tensor           # (C, nv)
    gate_in_hand: Tensor   # (C, Tm) bool
    gate_near_ped: Tensor
    ped_on: Tensor         # (C,) bool
    W: Tensor              # (C,) long
    T: Tensor
    max_sim: Tensor
    ctrl_steps: Tensor
    terminate_z: Tensor    # (C,) float
    vel_scale: Tensor
    pen_scale: Tensor
    pen_margin: Tensor
    ped_scale: Tensor
    warmup_mult: Tensor
    a0: int
    R: int

    @staticmethod
    def build(specs: list[ClipSpec], device: torch.device | str, dtype: torch.dtype) -> StackedRef:
        a0, R = specs[0].ex.obj_qadr, specs[0].n_robot
        if any(sp.ex.obj_qadr != a0 or sp.n_robot != R for sp in specs):
            raise ValueError("stacked clips need one robot layout (one hand side)")
        Tm = max(sp.rw.qpos_ref.shape[0] for sp in specs)

        def pad(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return np.concatenate([x, np.repeat(x[-1:], Tm - x.shape[0], axis=0)]) if x.shape[0] < Tm else x

        refs = [PoolRef.build(sp, "cpu", torch.float64) for sp in specs]
        f = lambda xs: torch.as_tensor(np.stack(xs), dtype=dtype, device=device)
        lg = lambda xs: torch.as_tensor(np.asarray(xs, np.int64), device=device)
        b = lambda xs: torch.as_tensor(np.stack(xs).astype(bool), device=device)
        return StackedRef(
            qpos_ref=f([pad(sp.rw.qpos_ref) for sp in specs]), qvel_ref=f([pad(sp.rw.qvel_ref) for sp in specs]),
            ctrl_ref=f([pad(sp.ctrl_ref) for sp in specs]), w_dq=f([sp.rw.w_dq for sp in specs]),
            gate_in_hand=b([pad(r.gate_in_hand.numpy()) for r in refs]),
            gate_near_ped=b([pad(r.gate_near_ped.numpy()) for r in refs]),
            ped_on=torch.as_tensor([r.ped_on for r in refs], device=device),
            W=lg([sp.W for sp in specs]), T=lg([sp.T for sp in specs]), max_sim=lg([sp.max_sim_steps for sp in specs]),
            ctrl_steps=lg([sp.ctrl_steps for sp in specs]), terminate_z=f([sp.terminate_z for sp in specs]),
            vel_scale=f([r.vel_scale for r in refs]), pen_scale=f([r.pen_scale for r in refs]),
            pen_margin=f([r.pen_margin for r in refs]), ped_scale=f([r.ped_scale for r in refs]),
            warmup_mult=f([r.warmup_mult for r in refs]), a0=a0, R=R)


def row_terms_s(p: StackedRef, clip: Tensor, rows: Tensor, qpos: Tensor, qvel: Tensor, pen: Tensor, ho: Tensor,
                ped: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """`row_terms_t` for worlds of different clips: (weighted reward, position error, rotation error)."""
    a0 = p.a0
    Tm = p.qpos_ref.shape[1]
    k = torch.clamp(rows + 1, max=Tm - 1)
    ref = p.qpos_ref[clip, k]
    dq = torch.cat([qpos[:, :a0] - ref[:, :a0], qpos[:, a0:a0 + 3] - ref[:, a0:a0 + 3],
                    quat_sub_t(qpos[:, a0 + 3:a0 + 7], ref[:, a0 + 3:a0 + 7])], dim=1)
    a = torch.linalg.vector_norm(dq * p.w_dq[clip], dim=1)
    b = torch.linalg.vector_norm(qvel - p.qvel_ref[clip, k], dim=1)
    h = torch.clamp(pen - p.pen_margin[clip], min=0.0)
    cost = a + p.vel_scale[clip] * b + p.pen_scale[clip] * h * h
    W = p.W[clip]
    live = (rows + 1) >= W
    ih, npd = p.gate_in_hand[clip, k], p.gate_near_ped[clip, k]
    mis = ((live & ih & ~npd & ~ho) | (live & npd & ~ih & ~ped)) & p.ped_on[clip]
    cost = cost + p.ped_scale[clip] * mis.to(cost.dtype)
    mult = torch.where(rows + 1 < W, p.warmup_mult[clip], torch.ones_like(cost))
    tref = p.qpos_ref[clip, torch.clamp(rows, max=Tm - 1)].to(torch.float32).to(cost.dtype)
    pos = torch.linalg.vector_norm(qpos[:, a0:a0 + 3] - tref[:, a0:a0 + 3], dim=1)
    rot = torch.linalg.vector_norm(quat_sub_t(qpos[:, a0 + 3:a0 + 7], tref[:, a0 + 3:a0 + 7]), dim=1)
    return -mult * cost, pos, rot


def observe_s(cfg: EnvConfig, p: StackedRef, clip: Tensor, onehot_id: Tensor, rows: Tensor, qpos: Tensor,
              qvel: Tensor, delta_prev: Tensor, sums: Tensor, contact: Tensor, n_clips: int, alpha: Tensor) -> Tensor:
    """`fastdo.transductive.obs.observe` for worlds of different clips (float32 (n, obs_dim))."""
    R, a0, dt, dev = p.R, p.a0, qpos.dtype, qpos.device
    Tm = p.qpos_ref.shape[1]
    js = torch.as_tensor(joint_scale(cfg, R), dtype=dt, device=dev)
    vs = torch.full((R,), 10.0, dtype=dt, device=dev)
    vs[:3], vs[3:6] = 1.0, 5.0
    ps, rs = cfg.obs_scale_pos, cfg.obs_scale_rot
    q, ob_p, ob_q = qpos[:, :R], qpos[:, a0:a0 + 3], qpos[:, a0 + 3:a0 + 7]
    r = p.qpos_ref[clip, torch.clamp(rows, max=Tm - 1)]
    parts = [(ob_p - r[:, a0:a0 + 3]) / ps, quat_sub_t(ob_q, r[:, a0 + 3:a0 + 7]) / rs,
             qvel[:, a0:a0 + 3], qvel[:, a0 + 3:a0 + 6] / 5.0, (q - r[:, :R]) / js, qvel[:, :R] / vs,
             (p.ctrl_ref[clip, torch.clamp(rows + cfg.rows_per_step, max=Tm - 1)] - q) / js]
    for d in cfg.future_rows:
        f = p.qpos_ref[clip, torch.clamp(rows + d, max=Tm - 1)]
        parts += [(f[:, a0:a0 + 3] - ob_p) / ps, quat_sub_t(f[:, a0 + 3:a0 + 7], ob_q) / rs, (f[:, :R] - q) / js]
    T = p.T[clip].to(dt)
    W = p.W[clip]
    ds = torch.as_tensor(delta_scale(cfg, R), dtype=dt, device=dev)
    ph = torch.stack([torch.clamp(W - rows, min=0).to(dt) / torch.clamp(W, min=1).to(dt), rows.to(dt) / T], dim=1)
    budget = torch.stack([sums[:, 0] / (SUCCESS_POS_M * T), sums[:, 1] / (SUCCESS_ROT_RAD * T)], dim=1)
    cont = torch.stack([contact[:, 0] / 0.005, contact[:, 1], contact[:, 2]], dim=1)
    parts += [(ob_p - q[:, :3]) / (2 * ps), delta_prev / ds, cont, ph, budget,
              (alpha[:, None] if cfg.assist_observed else torch.zeros_like(alpha[:, None])).to(dt)]
    if cfg.clip_onehot:
        parts.append(torch.nn.functional.one_hot(onehot_id, n_clips).to(dt))
    return torch.clamp(torch.cat(parts, dim=1), -cfg.obs_clip, cfg.obs_clip).to(torch.float32)
