"""Environment v2 terms ON THE DEVICE: torch ports of
`fastdo.amortised.tracker_obs` (reference table, observation, reward, step penalty, training termination),
`fastdo.amortised.se3.object_error` and `fastdo.amortised.contact_ref.GridSDF.query`, batched over worlds.

The host functions stay the SPECIFICATION: every function here mirrors one of them operation by operation, in
float64 (the table, states and intermediate terms), and casts to the host function's output dtype at the same
point. Equality with the host functions is a known answer (`tests/amortised/test_m9_dev_terms.py`, CPU device,
float64): equal to a few ulp, not bitwise (libm `exp`/`sin`/`atan2` and vectorised kernels differ between numpy and
torch), so a float32 observation can differ by one ulp from the host's. The replay rebuild on the device path uses
THIS function on both sides (acting and minibatch rebuild), so the rebuild still equals what the policy saw.

`DevTable.from_host(table)` uploads a `RefTable`; `DevSDF.from_grids(sdfs)` stacks the clips' grids (one grid per
clip, padded to the largest resolution) and answers queries by clip index. `TermDev` is `TermState` over all worlds:
`row(...)` feeds ONE executed row per world (the caller loops over the k rows of a step in order) and returns the
end reason code per world (`END_NONE`, `END_CLIP`, `END_ENVELOPE`, `END_FUTILE`), with `TermState.rows`' order of
checks (clip end, then envelope, then futility; the latter two only in training).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastdo.amortised.tracker_obs import RefTable, TrackerConfig

__all__ = ["END_CLIP", "END_ENVELOPE", "END_FUTILE", "END_NONE", "DevSDF", "DevTable", "TermDev", "group_t",
           "object_error_t", "observe_t", "quat_conj_t", "quat_rotate_t", "row_rewards_t", "step_penalty_t"]

END_NONE, END_CLIP, END_ENVELOPE, END_FUTILE = 0, 1, 2, 3
_SMALL = 1e-4
F64 = torch.float64


# ------------------------------------------------------------------------------------------------ SE(3), torch
def _dot3(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _cross(a: Tensor, b: Tensor) -> Tensor:
    return torch.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def quat_conj_t(q: Tensor) -> Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def _quat_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, av, bw, bv = a[..., 0], a[..., 1:], b[..., 0], b[..., 1:]
    w = aw * bw - _dot3(av, bv)
    v = aw[..., None] * bv + bw[..., None] * av + _cross(av, bv)
    return torch.cat([w[..., None], v], -1)


def quat_rotate_t(q: Tensor, v: Tensor) -> Tensor:
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * _cross(u, v)
    return v + w * t + _cross(u, t)


def _so3_log_quat(q: Tensor) -> Tensor:
    q = torch.where(q[..., :1] < 0, -q, q)
    w, v = q[..., 0], q[..., 1:]
    s = torch.sqrt(_dot3(v, v))
    th = 2.0 * torch.atan2(s, w)
    small = s < _SMALL
    ws = torch.where(small, w, torch.ones_like(w))
    k = torch.where(small, 2.0 / ws * (1.0 - s * s / (3.0 * w * w)), th / torch.where(small, torch.ones_like(s), s))
    return k[..., None] * v


def _v_inv_apply(om: Tensor, x: Tensor) -> Tensor:
    th = torch.sqrt(_dot3(om, om))
    small = th < 1e-3
    t = torch.where(small, torch.ones_like(th), th)
    t2 = th * th
    C = torch.where(small, 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0,
                    (1.0 - t * torch.sin(t) / (2.0 * (1.0 - torch.cos(t)))) / (t * t))
    wx = _cross(om, x)
    return x - 0.5 * wx + C[..., None] * _cross(om, wx)


def object_error_t(p: Tensor, q: Tensor, p_ref: Tensor, q_ref: Tensor) -> tuple[Tensor, Tensor]:
    """`se3.object_error` on tensors: (rho, omega) of (T^ref)^-1 T."""
    qi = quat_conj_t(q_ref)
    qe = _quat_mul(qi, q)
    pe = quat_rotate_t(qi, p - p_ref)
    om = _so3_log_quat(qe)
    return _v_inv_apply(om, pe), om


def group_t(n: int, vals: tuple[float, float, float], device: torch.device | str) -> Tensor:
    s = torch.full((n,), float(vals[2]), dtype=F64, device=device)
    s[:3], s[3:6] = float(vals[0]), float(vals[1])
    return s


# ------------------------------------------------------------------------------------------------ table and SDF
@dataclass(frozen=True, eq=False)
class DevTable:
    """`RefTable` on a device (float64 rows; int64 indices)."""

    offset: Tensor
    n_rows: Tensor
    W: Tensor
    T: Tensor
    obj_p: Tensor
    obj_q: Tensor
    obj_v: Tensor
    q_ref: Tensor
    ctrl_ref: Tensor
    tips_ref: Tensor
    dref: Tensor
    kappa: Tensor
    points: Tensor
    mass: Tensor
    friction: Tensor

    @staticmethod
    def from_host(t: RefTable, device: torch.device | str) -> DevTable:
        f = lambda a, dt=F64: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        i = lambda a: f(a, torch.int64)
        return DevTable(offset=i(t.offset), n_rows=i(t.n_rows), W=i(t.W), T=i(t.T), obj_p=f(t.obj_p),
                        obj_q=f(t.obj_q), obj_v=f(t.obj_v), q_ref=f(t.q_ref), ctrl_ref=f(t.ctrl_ref),
                        tips_ref=f(t.tips_ref), dref=f(t.dref), kappa=f(t.kappa), points=f(t.points),
                        mass=f(t.mass), friction=f(t.friction))

    @property
    def R(self) -> int:
        return int(self.q_ref.shape[1])

    @property
    def F(self) -> int:
        return int(self.tips_ref.shape[1])

    def idx(self, clip: Tensor, row: Tensor) -> Tensor:
        return self.offset[clip] + torch.minimum(torch.clamp(row, min=0), self.n_rows[clip] - 1)


@dataclass(frozen=True, eq=False)
class DevSDF:
    """The clips' `GridSDF`s stacked (C, G, G, G), zero-padded to the largest G; `size` holds each clip's own G."""

    values: Tensor
    origin: Tensor           # (C, 3)
    step: Tensor             # (C,)
    size: Tensor             # (C,) int64

    @staticmethod
    def from_grids(sdfs: Sequence[object], device: torch.device | str) -> DevSDF:
        G = max(int(s.values.shape[0]) for s in sdfs)
        vals = np.zeros((len(sdfs), G, G, G))
        for c, s in enumerate(sdfs):
            g = int(s.values.shape[0])
            vals[c, :g, :g, :g] = s.values
        f = lambda a, dt=F64: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        return DevSDF(values=f(vals), origin=f(np.stack([s.origin for s in sdfs])),
                      step=f([float(s.step) for s in sdfs]), size=f([int(s.values.shape[0]) for s in sdfs],
                                                                     torch.int64))

    def query(self, clip: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """`GridSDF.query` of clip `clip[b]` at points p (B, ..., 3): (phi (B, ...), grad (B, ..., 3))."""
        extra = p.dim() - 2
        shp = lambda x: x.reshape(x.shape[0], *([1] * extra), *x.shape[1:])
        org, st = shp(self.origin[clip]), shp(self.step[clip])[..., None]
        Gm1 = shp((self.size[clip] - 1).to(F64))[..., None]
        u = (p - org) / st
        uc = torch.minimum(torch.clamp(u, min=0.0), Gm1)
        i = torch.minimum(torch.floor(uc).to(torch.int64), (Gm1 - 1).to(torch.int64))
        f = uc - i.to(F64)
        x, y, z = i[..., 0], i[..., 1], i[..., 2]
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
        cb = shp(clip).expand_as(x)
        V = self.values
        c = {(a, b, e): V[cb, x + a, y + b, z + e] for a in (0, 1) for b in (0, 1) for e in (0, 1)}
        gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        phi = (c[0, 0, 0] * gx * gy * gz + c[1, 0, 0] * fx * gy * gz + c[0, 1, 0] * gx * fy * gz
               + c[1, 1, 0] * fx * fy * gz + c[0, 0, 1] * gx * gy * fz + c[1, 0, 1] * fx * gy * fz
               + c[0, 1, 1] * gx * fy * fz + c[1, 1, 1] * fx * fy * fz)
        dx = ((c[1, 0, 0] - c[0, 0, 0]) * gy * gz + (c[1, 1, 0] - c[0, 1, 0]) * fy * gz
              + (c[1, 0, 1] - c[0, 0, 1]) * gy * fz + (c[1, 1, 1] - c[0, 1, 1]) * fy * fz)
        dy = ((c[0, 1, 0] - c[0, 0, 0]) * gx * gz + (c[1, 1, 0] - c[1, 0, 0]) * fx * gz
              + (c[0, 1, 1] - c[0, 0, 1]) * gx * fz + (c[1, 1, 1] - c[1, 0, 1]) * fx * fz)
        dz = ((c[0, 0, 1] - c[0, 0, 0]) * gx * gy + (c[1, 0, 1] - c[1, 0, 0]) * fx * gy
              + (c[0, 1, 1] - c[0, 1, 0]) * gx * fy + (c[1, 1, 1] - c[1, 1, 0]) * fx * fy)
        grad = torch.stack([dx, dy, dz], -1) / st
        far = torch.any(u != uc, dim=-1)
        pc = org + uc * st
        gn = torch.clamp(torch.linalg.vector_norm(grad, dim=-1, keepdim=True), min=1e-12)
        q = pc - phi[..., None] * grad / gn
        off = p - q
        dist = torch.linalg.vector_norm(off, dim=-1)
        phi = torch.where(far, dist, phi)
        grad = torch.where(far[..., None], off / torch.clamp(dist, min=1e-12)[..., None], grad)
        return phi, grad

    def tip_sdf(self, clip: Tensor, tips: Tensor, obj_p: Tensor, obj_q: Tensor) -> Tensor:
        """`tracker_env.tip_sdf` per world: (B, F, 4) SDF distance and unit WORLD-frame gradient."""
        qi = quat_conj_t(obj_q)[:, None, :]
        local = quat_rotate_t(qi, tips - obj_p[:, None, :])
        phi, g = self.query(clip, local)
        g = g / torch.clamp(torch.sqrt((g * g).sum(-1, keepdim=True)), min=1e-12)
        return torch.cat([phi[..., None], quat_rotate_t(obj_q[:, None, :], g)], -1)


# ------------------------------------------------------------------------------------------------ observation
def observe_t(table: DevTable, rec: dict[str, Tensor], cfg: TrackerConfig) -> tuple[Tensor, Tensor]:
    """`tracker_obs.observe` on device tensors: (actor, critic) float32 observations of records (B, ...)."""
    dev = table.obj_p.device
    R, K, k = table.R, cfg.horizon, cfg.k
    c = rec["clip"].to(torch.int64)
    s = rec["row"].to(torch.int64)
    t = s - 1
    qp = rec["qpos"].to(F64)
    qv = rec["qvel"].to(F64)
    q, p, qo = qp[:, :R], qp[:, R:R + 3], qp[:, R + 3:R + 7]
    wr = rec["wrist"].to(F64)
    pw, qwi = wr[:, :3], quat_conj_t(wr[:, 3:7])
    to_wrist = lambda x: quat_rotate_t(qwi[:, None], x - pw[:, None])
    js, ds = group_t(R, cfg.q_scale, dev), group_t(R, cfg.delta_max, dev)
    vs = group_t(R, (1.0, 5.0, 10.0), dev)
    i0 = table.idx(c, t)
    rho, om = object_error_t(p, qo, table.obj_p[i0], table.obj_q[i0])
    ov = qv[:, R:R + 6]
    vr = table.obj_v[i0]
    v_err = quat_rotate_t(quat_conj_t(table.obj_q[i0]), ov[:, :3] - vr[:, :3])
    w_err = ov[:, 3:6] - vr[:, 3:6]
    parts = [rho / cfg.s_p, om / cfg.s_th, v_err, w_err / 5.0, (q - table.q_ref[i0]) / js, qv[:, :R] / vs,
             (table.ctrl_ref[table.idx(c, t + k)] - q) / js]
    for j in range(1, K + 1):
        ij = table.idx(c, t + j * k)
        fr, fo = object_error_t(table.obj_p[ij], table.obj_q[ij], p, qo)
        parts += [fr / cfg.s_p, fo / cfg.s_th, (table.q_ref[ij] - q) / js]
    pts = table.points[c]
    now = p[:, None] + quat_rotate_t(qo[:, None], pts)
    i1 = table.idx(c, t + k)
    nxt = table.obj_p[i1][:, None] + quat_rotate_t(table.obj_q[i1][:, None], pts)
    B = c.shape[0]
    parts += [to_wrist(now).reshape(B, -1) / cfg.pts_scale, to_wrist(nxt).reshape(B, -1) / cfg.pts_scale]
    ts = rec["tip_sdf"].to(F64)
    grad_w = quat_rotate_t(qwi[:, None], ts[..., 1:4])
    parts += [ts[..., 0] / cfg.sdf_scale, grad_w.reshape(B, -1), table.dref[i0] / cfg.sdf_scale]
    ct = rec["contact"].to(F64)
    W, T = table.W[c].to(F64), table.T[c].to(F64)
    kap = torch.stack([table.kappa[table.idx(c, t + j * k)] for j in range(1, K + 1)], 1)
    sf = s.to(F64)
    parts += [torch.stack([ct[:, 0] / 0.005, ct[:, 1], ct[:, 2]], 1), rec["prev_delta"].to(F64) / ds,
              torch.stack([torch.clamp(W - sf, min=0.0) / torch.clamp(W, min=1.0), sf / T], 1),
              rec["alpha"].to(F64)[:, None], kap]
    actor = torch.clamp(torch.cat([x.reshape(B, -1) for x in parts], 1), -cfg.obs_clip, cfg.obs_clip)
    priv = [table.mass[c][:, None], table.friction[c][:, None], rec["tip_force"].to(F64) / cfg.force_scale,
            rec["obj_vel"].to(F64)]
    critic = torch.cat([actor, torch.clamp(torch.cat(priv, 1), -cfg.obs_clip, cfg.obs_clip)], 1)
    return actor.to(torch.float32), critic.to(torch.float32)


# ------------------------------------------------------------------------------------------------ reward
def row_rewards_t(table: DevTable, cfg: TrackerConfig, clip: Tensor, rows: Tensor, qpos: Tensor, tips: Tensor,
                  d_tip: Tensor, pen: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """`tracker_obs.row_rewards` for one executed row per world: clip, rows (B,), qpos (B, nq), tips (B, F, 3),
    d_tip (B, F), pen (B,) -> (reward, |rho|, theta) (B,)."""
    R = table.R
    i = table.idx(clip, rows)
    rho, om = object_error_t(qpos[:, R:R + 3], qpos[:, R + 3:R + 7], table.obj_p[i], table.obj_q[i])
    rn, th = torch.sqrt((rho * rho).sum(-1)), torch.sqrt((om * om).sum(-1))
    e_tip = torch.sqrt(((tips - table.tips_ref[i]) ** 2).sum(-1)).mean(-1)
    lam = 1.0 / group_t(R, cfg.q_scale, qpos.device) ** 2
    dq2 = ((qpos[:, :R] - table.q_ref[i]) ** 2 * lam).sum(-1)
    track = (cfg.w_o * torch.exp(-rn / cfg.s_p - th / cfg.s_th) + cfg.w_h * torch.exp(-e_tip / cfg.s_h)
             + cfg.w_q * torch.exp(-dq2))
    contact = cfg.w_c * torch.exp(-torch.clamp(d_tip - table.dref[i], min=0.0) / cfg.s_c).mean(-1)
    pen_cost = cfg.w_pen * torch.clamp(pen - cfg.m_pen, min=0.0) / cfg.m_pen
    return table.kappa[i] * track + contact - pen_cost, rn, th


def step_penalty_t(cfg: TrackerConfig, delta: Tensor, delta_prev: Tensor) -> Tensor:
    ds = group_t(delta.shape[-1], cfg.delta_max, delta.device)
    return -cfg.w_a * (((delta - delta_prev) / ds) ** 2).sum(-1)


# ------------------------------------------------------------------------------------------------ termination
class TermDev:
    """`TermState` over all worlds on a device (module docstring)."""

    def __init__(self, n: int, device: torch.device | str) -> None:
        self.env_count = torch.zeros(n, dtype=torch.int64, device=device)
        self.pos_sum = torch.zeros(n, dtype=F64, device=device)

    def reset(self, mask: Tensor) -> None:
        self.env_count = torch.where(mask, torch.zeros_like(self.env_count), self.env_count)
        self.pos_sum = torch.where(mask, torch.zeros_like(self.pos_sum), self.pos_sum)

    def row(self, cfg: TrackerConfig, live: Tensor, rows: Tensor, rho_n: Tensor, theta: Tensor, p_err: Tensor,
            W: Tensor, T: Tensor, evaluation: bool) -> Tensor:
        """Feed one executed row of every LIVE world (others untouched); the end code per world (module docstring)."""
        code = torch.full_like(rows, END_NONE)
        clip_end = live & (rows + 1 >= T)
        code = torch.where(clip_end, torch.full_like(code, END_CLIP), code)
        if evaluation:
            return code
        go = live & ~clip_end
        out = (rho_n > cfg.env_pos) | (theta > cfg.env_rot)          # the welded warmup (rows < W)
        if cfg.env_post_pos is None:                                  # post-release term dropped
            post = torch.zeros_like(out)
        else:
            post = (rho_n > cfg.env_post_pos) | (theta > cfg.env_post_rot)
        out = torch.where(rows < W, out, post)
        cnt = torch.where(out, self.env_count + 1, torch.zeros_like(self.env_count))
        self.env_count = torch.where(go, cnt, self.env_count)
        env_end = go & (self.env_count >= cfg.env_rows)
        code = torch.where(env_end, torch.full_like(code, END_ENVELOPE), code)
        go = go & ~env_end
        lo = W + cfg.grace
        n_win = torch.clamp(T - W - cfg.grace, min=1).to(F64)
        bound = cfg.fut_pos * n_win * (1.0 + 1e-9)
        acc = go & (rows >= lo)
        self.pos_sum = torch.where(acc, self.pos_sum + p_err, self.pos_sum)
        fut = acc & (self.pos_sum > bound)
        return torch.where(fut, torch.full_like(code, END_FUTILE), code)
