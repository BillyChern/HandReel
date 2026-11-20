"""Environment v2 on a BATCHED backend with device observation, reward and termination.

`DeviceTrackerEnv` has `TrackerEnv`'s contract (the same `StepOut`, record fields, reset hook, evaluation mode) and
semantics, row for row, but steps every world together on `batch_group.PoolsBackend`:
- `backend="warp"`: one `sim_warp.WarpPool` per clip, DO AS I DO's simulator (mujoco_warp 0.0.1; the `retargeting`
  env), CUDA or warp's CPU device; fingertips by `mujoco_warp.kinematics` on a separate FK `Data` (the simulation
  data are never touched), per-finger normal force by `mujoco_warp.contact_force`;
- `backend="cpu-pool"`: one `batch_group.CpuPool` per clip (CPU MuJoCo, float64), the test reference: with it the
  device env executes EXACTLY `TrackerEnv`'s rows (same controls, same simulator), so the two are compared directly.
Worlds are bound to clips for good (per-clip pools cannot rebind; `world_clip` must be grouped, non-decreasing), so
a reset hook must return each world's own clip (it chooses alpha and the start row; RSI is supported).

Semantics kept from `TrackerEnv` (the specification):
- controls: computed on the HOST in float64 exactly as `TrackerEnv.step` does (first-order hold of the residual over
  k rows, ctrl_ref + residual, rounded to float32), so a world's applied controls are bit-identical to the CPU env's
  for the same action;
- per row, in order: the training termination (`TermDev`: clip end, envelope, futility; evaluation: clip end only),
  then the executor's own ends (invalid state, then the drop rule at block ends). Rows after a training end are still
  executed (as `CpuWorlds` executes the whole block) until the block or an executor end; the record of the step is
  the state after the LAST EXECUTED row (min(k, T - s0) rows, or the executor-end row);
- reward = the step's rows up to the end summed in order, plus the step penalty; float32;
- an ended world gets its final record (post-step state, prev_delta = the step's target) and is reset at once.
Lock-step differences (no effect on any kept quantity): a world keeps stepping garbage rows after its executor end
or past its clip end until the step's k rows are done; those rows are discarded (and counted in `rows` executed only
up to where `TrackerEnv` would have executed).

Records: fingertips, SDF features and rewards are computed on the device in float64 (`tracker_dev`); the record
fields are cast to their storage dtypes on the HOST with numpy (as `TrackerEnv._record` does), and the device copy
of the record (`record_dev`) is uploaded from those host arrays, so the device observation reads exactly the stored
bits (the replay rebuild on the device path is the same function on the same bits).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.amortised.tracker_dev import (
    END_CLIP,
    END_ENVELOPE,
    END_FUTILE,
    DevSDF,
    DevTable,
    TermDev,
    row_rewards_t,
    step_penalty_t,
)
from fastdo.amortised.tracker_env import StepOut, TipSpec, fk_hand
from fastdo.amortised.tracker_obs import RefTable, TrackerConfig, _group, record_fields
from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.sim_cpu import _object_body

__all__ = ["BACKENDS", "CpuKin", "DeviceTrackerEnv", "WarpKin", "grouped_world_clip"]

BACKENDS = ("warp", "cpu-pool")
_TERM_NAMES = {END_CLIP: "clip_end", END_ENVELOPE: "envelope", END_FUTILE: "futile"}
F64 = torch.float64


def grouped_world_clip(n_worlds: int, n_clips: int) -> list[int]:
    """World -> clip for per-clip pools: contiguous blocks, sizes differing by at most one (clip c gets worlds
    [c n / C, (c + 1) n / C))."""
    if n_worlds < n_clips:
        raise ValueError(f"{n_worlds} worlds < {n_clips} clips")
    return [i * n_clips // n_worlds for i in range(n_worlds)]


def _obj_geom_mask(m: mujoco.MjModel, spec: ClipSpec) -> np.ndarray:
    b = _object_body(spec)[0]
    return np.asarray(m.geom_bodyid == b)


def _finger_of_body(m: mujoco.MjModel, tip: TipSpec) -> np.ndarray:
    out = np.full(m.nbody, -1, np.int64)
    for f, b in enumerate(tip.tip_body):
        if out[b] < 0:
            out[b] = f
    return out


class CpuKin:
    """Fingertip FK and per-finger normal force of a `CpuPool` (CPU MuJoCo; `TrackerEnv`'s own functions)."""

    def __init__(self, pool: Any, spec: ClipSpec, tip: TipSpec, device: torch.device) -> None:
        self.pool, self.spec, self.tip, self.dev = pool, spec, tip, device
        self.m = spec.ex.model
        self.scratch = mujoco.MjData(self.m)
        self.obj = _obj_geom_mask(self.m, spec)

    def fk(self, q: Tensor) -> tuple[Tensor, Tensor]:
        wr, tips = fk_hand(self.m, self.scratch, q.detach().cpu().numpy().astype(np.float64), self.tip)
        return torch.as_tensor(wr, dtype=F64, device=self.dev), torch.as_tensor(tips, dtype=F64, device=self.dev)

    def tip_force(self, mask: Tensor) -> Tensor:
        n = self.pool.n
        out = np.zeros((n, len(self.tip.tip_body)))
        f6 = np.zeros(6)
        for i in np.flatnonzero(mask.detach().cpu().numpy()):
            d = self.pool.sim.data[i]
            for k in range(int(d.ncon)):
                g1, g2 = int(d.contact.geom1[k]), int(d.contact.geom2[k])
                other = g2 if self.obj[g1] else (g1 if self.obj[g2] else -1)
                if other < 0:
                    continue
                b = int(self.m.geom_bodyid[other])
                if b in self.tip.tip_body:
                    mujoco.mj_contactForce(self.m, d, k, f6)
                    out[i, self.tip.tip_body.index(b)] += f6[0]
        return torch.as_tensor(out, dtype=F64, device=self.dev)


class WarpKin:
    """Fingertip FK (`mujoco_warp.kinematics` on a separate FK Data of the pool's worlds) and per-finger normal force
    (`mujoco_warp.contact_force` on the pool's contact buffer) of a `WarpPool`."""

    def __init__(self, pool: Any, spec: ClipSpec, tip: TipSpec, device: torch.device) -> None:
        import mujoco_warp as mjw
        import warp as wp

        self.pool, self.spec, self.tip, self.dev = pool, spec, tip, device
        self._wp, self._mjw = wp, mjw
        m = spec.ex.model
        d0 = mujoco.MjData(m)
        d0.qpos[:] = spec.ex.q0
        with wp.ScopedDevice(pool.device), pool._scope():
            self.fd = mjw.put_data(m, d0, nworld=pool.n, nconmax=1, njmax=1)
            ncap = int(pool.cdist.shape[0])
            self.ids = wp.array(np.arange(ncap, dtype=np.int32), dtype=int)
            self.force = wp.zeros(ncap, dtype=wp.spatial_vector)
        t = wp.to_torch
        self.fq, self.xpos, self.xquat, self.xmat = t(self.fd.qpos), t(self.fd.xpos), t(self.fd.xquat), t(self.fd.xmat)
        self.force_t = t(self.force)
        pdev = self.fq.device
        self.wb = int(tip.wrist_body)
        self.tb = torch.as_tensor(np.asarray(tip.tip_body), device=pdev)
        self.off = torch.as_tensor(tip.tip_offset, dtype=torch.float32, device=pdev)
        self.obj = torch.as_tensor(_obj_geom_mask(m, spec), device=pdev)
        self.bodyid = torch.as_tensor(np.asarray(m.geom_bodyid, np.int64), device=pdev)
        self.finger = torch.as_tensor(_finger_of_body(m, tip), device=pdev)

    def fk(self, q: Tensor) -> tuple[Tensor, Tensor]:
        self.fq[:] = q.to(self.fq.device, torch.float32)
        with self._wp.ScopedDevice(self.pool.device), self.pool._scope():
            self._mjw.kinematics(self.pool.mw, self.fd)
        xp, xq, xm = self.xpos, self.xquat, self.xmat.reshape(self.xmat.shape[0], -1, 3, 3)
        wr = torch.cat([xp[:, self.wb], xq[:, self.wb]], -1)
        tips = xp[:, self.tb] + torch.einsum("nfij,fj->nfi", xm[:, self.tb], self.off)
        return wr.to(self.dev, F64), tips.to(self.dev, F64)

    def tip_force(self, mask: Tensor) -> Tensor:
        p = self.pool
        with self._wp.ScopedDevice(p.device), p._scope():
            self._mjw.contact_force(p.mw, p.dw, self.ids, False, self.force)
        fn = self.force_t[:, 0].to(torch.float64)
        g1, g2 = p.cgeom[:, 0].long(), p.cgeom[:, 1].long()
        valid = torch.arange(fn.shape[0], device=fn.device) < p.nacon[0]
        other = torch.where(self.obj[g1], g2, torch.where(self.obj[g2], g1, torch.full_like(g1, -1)))
        fing = torch.where(other >= 0, self.finger[self.bodyid[torch.clamp(other, min=0)]], torch.full_like(g1, -1))
        keep = valid & (fing >= 0)
        out = torch.zeros(p.n, len(self.tip.tip_body), dtype=torch.float64, device=fn.device)
        out.index_put_((p.cwid.long()[keep], fing[keep]), fn[keep], accumulate=True)
        return out.to(self.dev) * mask.to(self.dev, F64)[:, None]


class DeviceTrackerEnv:
    """Worlds `world_clip[i]` over `specs` on a batched backend (module docstring)."""

    def __init__(self, cfg: TrackerConfig, specs: Sequence[ClipSpec], table: RefTable, tips: Sequence[TipSpec],
                 sdfs: Sequence[Any], world_clip: Sequence[int], *, backend: str = "warp",
                 device: str | torch.device = "cuda:0", sim_device: str | None = None, evaluation: bool = False,
                 reset_hook: Callable[[int], tuple[int, float, int]] | None = None, assist: bool = False,
                 settle: int = 20) -> None:
        from fastdo.transductive.batch_group import CpuPool, PoolsBackend

        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r} not in {BACKENDS}")
        wc = np.asarray(world_clip, np.int64)
        if wc.size == 0 or np.any(np.diff(wc) < 0):
            raise ValueError("world_clip must be non-empty and grouped by clip (non-decreasing)")
        self.cfg, self.specs, self.table, self.tips = cfg, list(specs), table, list(tips)
        self.evaluation, self.reset_hook = bool(evaluation), reset_hook
        self.dev = torch.device(device)
        if self.dev.type == "cuda":
            self.stream = torch.cuda.Stream(self.dev)        # warp's captured graphs run on a non-default stream
            torch.cuda.set_stream(self.stream)
        self.world_clip = wc
        self.n, self.R = int(wc.size), table.R
        m0 = self.specs[0].ex.model
        self.fields = record_fields(m0.nq, m0.nv, self.R, table.F)
        clips = sorted(set(wc.tolist()))
        pools, kins = [], []
        for c in clips:
            n_c = int((wc == c).sum())
            if backend == "warp":
                from fastdo.transductive.sim_warp import WarpPool

                pool = WarpPool(self.specs[c], n_c, device=sim_device or str(device), assist=assist, settle=settle)
                kins.append(WarpKin(pool, self.specs[c], self.tips[c], self.dev))
            else:
                pool = CpuPool(self.specs[c], n_c, assist_wn=10.0, settle=settle)
                kins.append(CpuKin(pool, self.specs[c], self.tips[c], self.dev))
            pools.append(pool)
        self.backend = PoolsBackend(pools, clips)
        self.kins, self.slices = kins, list(self.backend.sl)
        self.dt = DevTable.from_host(table, self.dev)
        self.sdf = DevSDF.from_grids(list(sdfs), self.dev)
        self.clip_t = torch.as_tensor(wc, device=self.dev)
        self.ctrl_full = [np.asarray(sp.ctrl_ref, np.float64) for sp in self.specs]
        self.T = np.array([sp.T for sp in self.specs], np.int64)
        self.ctrl_steps = torch.as_tensor([sp.ctrl_steps for sp in self.specs], device=self.dev)
        self.term_z = torch.as_tensor([sp.terminate_z for sp in self.specs], dtype=F64, device=self.dev)
        self.z_adr = self.R + 2
        self._ds = _group(self.R, cfg.delta_max)
        self.delta_prev = np.zeros((self.n, self.R))
        self.alpha = np.zeros(self.n)
        self.next_alpha = np.zeros(self.n)
        self.start_row = np.zeros(self.n, np.int64)
        self.row = np.zeros(self.n, np.int64)
        self.term = TermDev(self.n, self.dev)
        self.contact = torch.zeros(self.n, 3, dtype=F64, device=self.dev)
        self.tforce = torch.zeros(self.n, len(self.tips[0].tip_body), dtype=F64, device=self.dev)
        self.record_dev: dict[str, Tensor] = {}

    # ------------------------------------------------------------------------------------------ helpers
    def _state(self) -> tuple[Tensor, Tensor]:
        q, v = self.backend.state()
        return q.to(self.dev, F64), v.to(self.dev, F64)

    def _fk(self, q: Tensor) -> tuple[Tensor, Tensor]:
        out = [k.fk(q[sl]) for k, sl in zip(self.kins, self.slices, strict=True)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])

    def _tip_force(self, mask: Tensor) -> Tensor:
        return torch.cat([k.tip_force(mask[sl]) for k, sl in zip(self.kins, self.slices, strict=True)])

    def _records(self, q: Tensor, v: Tensor, contact: Tensor, tforce: Tensor, prev_delta: np.ndarray,
                 alpha: np.ndarray, row: np.ndarray) -> dict[str, np.ndarray]:
        """Host records (`TrackerEnv._record` for every world): float64 terms on the device, cast on the host."""
        R = self.R
        wr, tips = self._fk(q)
        ts = self.sdf.tip_sdf(self.clip_t, tips, q[:, R:R + 3], q[:, R + 3:R + 7])
        h = lambda x: x.detach().cpu().numpy()
        raw = {"qpos": h(q), "qvel": h(v), "wrist": h(wr), "tips": h(tips), "tip_sdf": h(ts), "contact": h(contact),
               "prev_delta": prev_delta, "alpha": alpha, "clip": self.world_clip, "row": row,
               "tip_force": h(tforce), "obj_vel": h(v[:, R:R + 6])}
        return {k: np.asarray(raw[k]).astype(dt) for k, (_, dt) in self.fields.items()}

    def _upload(self, rec: dict[str, np.ndarray]) -> None:
        self.record_dev = {k: torch.as_tensor(a, device=self.dev) for k, a in rec.items()}

    def _reset(self, worlds: np.ndarray) -> None:
        mask = np.zeros(self.n, bool)
        mask[worlds] = True
        row = np.zeros(self.n, np.int64)
        for i in worlds:
            if self.reset_hook is not None:
                clip, alpha, r = self.reset_hook(int(i))
                if int(clip) != int(self.world_clip[i]):
                    raise ValueError(f"world {i} is bound to clip {self.world_clip[i]}, the hook chose {clip} "
                                     "(per-clip pools cannot rebind)")
                self.next_alpha[i] = float(alpha)
                row[i] = int(r)
        mt = torch.as_tensor(mask, device=self.dev)
        self.backend.reset(mt, self.clip_t, torch.as_tensor(row, device=self.dev))
        self.start_row[mask] = row[mask]
        self.row[mask] = row[mask]
        self.delta_prev[mask] = 0.0
        self.alpha[mask] = self.next_alpha[mask]
        self.term.reset(mt)
        self.contact[mt] = 0.0
        self.tforce[mt] = 0.0

    def _sim_alpha(self) -> Tensor:
        a = np.zeros(self.n) if self.evaluation else self.alpha
        return torch.as_tensor(a, dtype=F64, device=self.dev)

    def records(self) -> dict[str, np.ndarray]:
        q, v = self._state()
        return self._records(q, v, self.contact, self.tforce, self.delta_prev, self.alpha, self.row)

    def reset_all(self) -> dict[str, np.ndarray]:
        self._reset(np.arange(self.n))
        rec = self.records()
        self._upload(rec)
        return rec

    # ------------------------------------------------------------------------------------------ step
    def controls(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(target (n, R), float32 control rows (n, k, nu), rows each world executes min(k, T - s0)) on the HOST,
        exactly as `TrackerEnv.step` computes them."""
        cfg, R, n = self.cfg, self.R, self.n
        target = self._ds * np.tanh(np.asarray(actions, np.float64))
        frac = np.arange(1, cfg.k + 1) / cfg.k
        res = self.delta_prev[:, None, :] + frac[None, :, None] * (target - self.delta_prev)[:, None, :]
        kk = np.minimum(cfg.k, self.T[self.world_clip] - self.row)
        nu = self.ctrl_full[0].shape[1]
        u = np.zeros((n, cfg.k, nu))
        for c in np.unique(self.world_clip):
            w = np.flatnonzero(self.world_clip == c)
            ref = self.ctrl_full[c]
            idx = np.minimum(self.row[w][:, None] + np.arange(cfg.k)[None, :], ref.shape[0] - 1)
            u[w] = ref[idx]
        u[:, :, :R] += res
        return target, u.astype(np.float32), kk

    def step(self, actions: np.ndarray) -> StepOut:
        cfg, R, n, dev = self.cfg, self.R, self.n, self.dev
        target, u32, kk = self.controls(actions)
        ctrl = torch.as_tensor(u32, device=dev)
        kk_t = torch.as_tensor(kk, device=dev)
        s0 = torch.as_tensor(self.row, device=dev)
        Wc, Tc = self.dt.W[self.clip_t], self.dt.T[self.clip_t]
        alpha_sim = self._sim_alpha()
        nq = self.specs[0].ex.model.nq
        qbuf = torch.zeros(n, cfg.k, nq, dtype=F64, device=dev)
        false = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
        exec_end, term_end = false(), false()
        term_code = torch.zeros(n, dtype=torch.int64, device=dev)
        exec_code = torch.zeros(n, dtype=torch.int64, device=dev)      # 1 invalid, 2 drop
        used = kk_t.clone()
        n_exec = kk_t.clone()
        rew = torch.zeros(n, dtype=F64, device=dev)
        snap_q, snap_v = self._state()
        snap_c, snap_tf = self.contact.clone(), self.tforce.clone()
        for j in range(cfg.k):
            rows = s0 + j
            executing = (j < kk_t) & ~exec_end
            # a world past its executor end or its clip end is not stepped where the backend can skip it (CpuPool:
            # a row that differs from the world's own), and never with the assist or beyond the reference (WarpPool)
            rows_sim = torch.where(executing, rows, torch.full_like(rows, -1))
            a_sim = torch.where(executing, alpha_sim, torch.zeros_like(alpha_sim))
            pen, ho, ped, bad = self.backend.step_row(ctrl[:, j], rows_sim, a_sim)
            q, v = self._state()
            qbuf[:, j] = q
            pen, bad = pen.to(dev, F64), bad.to(dev)
            _, tips = self._fk(q)
            d_tip = self.sdf.tip_sdf(self.clip_t, tips, q[:, R:R + 3], q[:, R + 3:R + 7])[..., 0]
            r, rn, th = row_rewards_t(self.dt, cfg, self.clip_t, rows, q, tips, d_tip, pen)
            i_ref = self.dt.idx(self.clip_t, rows)
            p_err = torch.sqrt(((q[:, R:R + 3] - self.dt.obj_p[i_ref]) ** 2).sum(-1))
            live = executing & ~term_end
            code = self.term.row(cfg, live, rows, rn, th, p_err, Wc, Tc, self.evaluation)
            rew = torch.where(live, rew + r, rew)
            new_term = live & (code > 0)
            term_code = torch.where(new_term, code, term_code)
            used = torch.where(new_term, torch.full_like(used, j + 1), used)
            term_end |= new_term
            invalid = executing & bad
            blk_end = (rows + 1) % self.ctrl_steps[self.clip_t] == 0
            drop = executing & ~bad & blk_end & (q[:, self.z_adr] < self.term_z[self.clip_t])
            new_exec = invalid | drop
            exec_code = torch.where(invalid, torch.ones_like(exec_code),
                                    torch.where(drop, torch.full_like(exec_code, 2), exec_code))
            n_exec = torch.where(new_exec, torch.full_like(n_exec, j + 1), n_exec)
            used = torch.where(new_exec & ~term_end, torch.full_like(used, j + 1), used)
            exec_end |= new_exec
            snap = executing & (new_exec | (kk_t == j + 1))
            if bool(snap.any()):
                tf = self._tip_force(snap)
                ct = torch.stack([pen, ho.to(dev, F64), ped.to(dev, F64)], 1)
                s1 = snap[:, None]
                snap_q, snap_v = torch.where(s1, q, snap_q), torch.where(s1, v, snap_v)
                snap_c, snap_tf = torch.where(s1, ct, snap_c), torch.where(s1, tf, snap_tf)
        pen_step = step_penalty_t(cfg, torch.as_tensor(target, device=dev), torch.as_tensor(self.delta_prev,
                                                                                               device=dev))
        reward = (rew + pen_step).to(torch.float32)
        h = lambda x: x.detach().cpu().numpy()
        term_h, exec_h, used_h, nexec_h = h(term_code), h(exec_code), h(used), h(n_exec)
        q_h = h(qbuf)
        ends: list[str | None] = [None] * n
        for i in np.flatnonzero((term_h > 0) | (exec_h > 0)):
            ends[i] = _TERM_NAMES[int(term_h[i])] if term_h[i] > 0 else ("invalid" if exec_h[i] == 1 else "drop")
        ended = np.array([e is not None for e in ends])
        self.delta_prev = target.copy()
        self.row = self.row + nexec_h
        self.contact, self.tforce = snap_c, snap_tf
        final_rec = self._records(snap_q, snap_v, snap_c, snap_tf, self.delta_prev, self.alpha, self.row)
        final = {k: np.where(ended.reshape(-1, *([1] * (a.ndim - 1))), a, np.zeros_like(a))
                 for k, a in final_rec.items()}
        qrows = [q_h[i, :int(used_h[i] if ended[i] else nexec_h[i])] for i in range(n)]
        crows = [u32[i, :int(used_h[i] if ended[i] else nexec_h[i])].astype(np.float64) for i in range(n)]
        done = np.flatnonzero(ended)
        if done.size:
            self._reset(done)
            rec = self.records()
        else:
            rec = final_rec
        self._upload(rec)
        return StepOut(reward=h(reward), term=ended.astype(np.float32), trunc=np.zeros(n, np.float32), end=ends,
                       final=final, record=rec, rows=nexec_h.astype(np.int64), qpos_rows=qrows, ctrl_rows=crows)

    def get_state(self) -> dict[str, np.ndarray]:
        """The device path does not snapshot simulator state (one unit per arm, no resume)."""
        return {}

    def set_state(self, st: dict[str, np.ndarray]) -> None:
        raise NotImplementedError("the device backend does not resume a simulator state")
