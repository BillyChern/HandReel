"""The episode logic of `fastdo.transductive.group.WorldGroup`, batched in torch over ALL worlds at once (GPU
path), on top of a `BatchBackend`: one batched simulator step for every world, whatever its clip.

Backend interface (`BatchBackend`, the seam a slot-union model plugs into, e.g. a one-model-many-clips batch):
- `reset(mask, clip, row)`: (re)start the masked worlds on clip `clip[w]` at row `row[w]` (0 = the executor's start
  state; > 0 = reference-state initialisation, `sim_cpu.CpuWorlds.reset`); a backend that cannot rebind a world to
  another clip or start mid-clip raises;
- `state()`: (qpos, qvel) of every world; `step_row(ctrl, rows, alpha)`: one executor row for every world (the
  backend applies each world's phase: weld and zero gravity before W or during an RSI settle, the assist wrench),
  returning per-world contact terms (max hand-object penetration, hand-object touch, object-pedestal touch) in the
  backend's own geom numbering, and an invalid-state flag.
Implementations: `PoolsBackend` (per-clip pools side by side: `sim_warp.WarpPool` on the GPU or `CpuPool`; clip
binding fixed, RSI on CpuPool only), `CpuBatchBackend` (CPU MuJoCo worlds of any clip, rebindable, RSI: the
reference for a union backend). Everything clip-specific the group needs (references, gates, time base, reward
scales) is gathered per world from a `torch_terms.StackedRef` by (clip, row).

Per row: the applied control `ctrl_ref[clip, s] + residual` (float32), DO AS I DO's weighted reward and the tracking
errors, the applied-control buffer, RSI early termination, the executor's drop rule at block ends; a world that ends
mid-step (invalid, RSI early termination) is masked for the rest of the step. Per step: the episode-end rules, the
scorer's criteria at the clip's end, successes (unassisted episodes that started at row 0), auto-reset with the next
assist level and the next start row (`config.draw_start`, the same per-world draw as the numpy path), the observation.
`tests/transductive/test_batch_group.py` checks the semantics against `WorldGroup`.

Resume on this path restarts every episode (`snapshot`/`restore` carry no world state). The CPU path resumes exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.config import SUCCESS_POS_M, SUCCESS_ROT_RAD, EnvConfig, draw_start
from fastdo.transductive.group import END_CODES
from fastdo.transductive.obs import delta_scale
from fastdo.transductive.sim_cpu import CpuWorlds
from fastdo.transductive.torch_terms import StackedRef, observe_s, row_terms_s

__all__ = ["BatchBackend", "BatchGroup", "CpuBatchBackend", "CpuPool", "Pool", "PoolsBackend"]


class BatchBackend(Protocol):
    """n worlds, each bound to one clip of the group, stepped together one row at a time (module docstring)."""

    n: int

    def reset(self, mask: Tensor, clip: Tensor, row: Tensor) -> None: ...
    def state(self) -> tuple[Tensor, Tensor]: ...
    def step_row(self, ctrl: Tensor, rows: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


class Pool(Protocol):
    """n worlds of ONE clip (`PoolsBackend` puts several side by side)."""

    spec: ClipSpec
    n: int

    def reset(self, mask: Tensor, row: Tensor | None = None) -> None: ...
    def state(self) -> tuple[Tensor, Tensor]: ...
    def step_row(self, ctrl: Tensor, rows: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


class _CpuWorldsRows:
    """Shared by the CPU backends: `CpuWorlds` stepped one row per world, masked worlds skipped."""

    sim: CpuWorlds
    n: int
    dtype: torch.dtype
    settle: int

    def state(self) -> tuple[Tensor, Tensor]:
        q = np.stack([self.sim.qpos(i) for i in range(self.n)])
        v = np.stack([self.sim.qvel(i) for i in range(self.n)])
        return torch.as_tensor(q, dtype=self.dtype), torch.as_tensor(v, dtype=self.dtype)

    def step_row(self, ctrl: Tensor, rows: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        u = ctrl.detach().cpu().numpy().astype(np.float64)
        pen, ho, ped, bad = np.zeros(self.n), np.zeros(self.n, bool), np.zeros(self.n, bool), np.zeros(self.n, bool)
        for i in range(self.n):
            self.sim.alpha[i] = float(alpha[i])
            if int(self.sim.row[i]) != int(rows[i]):    # ended earlier in this step (masked by the group)
                bad[i] = True
                continue
            blk = self.sim.step_rows(i, u[i:i + 1])
            pen[i], ho[i], ped[i] = blk.pen[0], blk.ho[0], blk.ped[0]
            bad[i] = blk.end == "invalid"           # a "drop" is re-derived by the group's own block-end rule
        t = lambda x, dt: torch.as_tensor(x, dtype=dt)
        return t(pen, self.dtype), t(ho, torch.bool), t(ped, torch.bool), t(bad, torch.bool)


class CpuPool(_CpuWorldsRows):
    """The CPU backend as a single-clip pool (test reference for `PoolsBackend`)."""

    def __init__(self, spec: ClipSpec, n: int, assist_wn: float, dtype: torch.dtype = torch.float64,
                 sim_cls: type[CpuWorlds] = CpuWorlds, settle: int = 20, assist_track: bool = True) -> None:
        self.spec, self.n, self.dtype, self.settle = spec, n, dtype, settle
        self.sim = sim_cls([spec], [0] * n, assist_wn=assist_wn, assist_track=assist_track)

    def reset(self, mask: Tensor, row: Tensor | None = None) -> None:
        for i in np.flatnonzero(mask.cpu().numpy()):
            self.sim.reset(int(i), 0 if row is None else int(row[i]), self.settle)


class CpuBatchBackend(_CpuWorldsRows):
    """CPU MuJoCo worlds of ANY clip of the group (`CpuWorlds` over all clips; `reset` rebinds a world to the clip
    it is given): the interface a slot-union backend implements, on the CPU."""

    def __init__(self, specs: Sequence[ClipSpec], world_clip: Sequence[int], assist_wn: float,
                 dtype: torch.dtype = torch.float64, settle: int = 20, sim_cls: type[CpuWorlds] = CpuWorlds,
                 assist_track: bool = True) -> None:
        self.n, self.dtype, self.settle = len(world_clip), dtype, settle
        self.sim = sim_cls(list(specs), list(world_clip), assist_wn=assist_wn, assist_track=assist_track)

    def reset(self, mask: Tensor, clip: Tensor, row: Tensor) -> None:
        for i in np.flatnonzero(mask.cpu().numpy()):
            self.sim.rebind(int(i), int(clip[i]))
            self.sim.reset(int(i), int(row[i]), self.settle)


class PoolsBackend:
    """Per-clip pools side by side (pool k = clip `clips[k]`); the world -> clip binding is fixed."""

    def __init__(self, pools: Sequence[Pool], clips: Sequence[int]) -> None:
        self.pools, self.clips = list(pools), [int(c) for c in clips]
        self.sl, o = [], 0
        for p in self.pools:
            self.sl.append(slice(o, o + p.n))
            o += p.n
        self.n = o
        self.world_clip = np.concatenate([np.full(p.n, c) for p, c in zip(self.pools, self.clips, strict=True)])

    def reset(self, mask: Tensor, clip: Tensor, row: Tensor) -> None:
        wc = torch.as_tensor(self.world_clip, device=clip.device)
        if bool((mask & (clip != wc)).any()):
            raise NotImplementedError("per-clip pools cannot rebind a world to another clip (a union backend can)")
        for p, sl in zip(self.pools, self.sl, strict=True):
            m = mask[sl]
            if bool(m.any()):
                r = row[sl]
                p.reset(m, r if bool((r[m] > 0).any()) else None)

    def state(self) -> tuple[Tensor, Tensor]:
        st = [p.state() for p in self.pools]
        return torch.cat([s[0] for s in st]), torch.cat([s[1] for s in st])

    def step_row(self, ctrl: Tensor, rows: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        out = [p.step_row(ctrl[sl], rows[sl], alpha[sl]) for p, sl in zip(self.pools, self.sl, strict=True)]
        return tuple(torch.cat([o[k] for o in out]) for k in range(4))  # type: ignore[return-value]


class BatchGroup:
    """All worlds of one hand side on one backend (module docstring). `specs[c]` is clip c of the group;
    `clip_ids[c]` its global index (the one-hot); `world_clip` the initial clip of every world."""

    def __init__(self, cfg: EnvConfig, backend: BatchBackend, specs: Sequence[ClipSpec], world_clip: Sequence[int],
                 clip_ids: Sequence[int], n_clips: int, device: torch.device | str, dtype: torch.dtype = torch.float32,
                 world_ids: Sequence[int] | None = None) -> None:
        self.cfg, self.backend, self.specs, self.n_clips = cfg, backend, list(specs), int(n_clips)
        self.dev, self.dt = torch.device(device), dtype
        self.ref = StackedRef.build(self.specs, self.dev, dtype)
        K = cfg.rows_per_step
        if any(sp.ctrl_steps % K or sp.T % K for sp in self.specs):
            raise ValueError("rows_per_step must divide every clip's ctrl_steps and T")
        n = backend.n
        self.R = self.ref.R
        z = lambda *s, dt=torch.float64: torch.zeros(*s, dtype=dt, device=self.dev)
        self.clip = torch.as_tensor(np.asarray(world_clip, np.int64), device=self.dev)
        self.onehot = torch.as_tensor(np.asarray(clip_ids, np.int64), device=self.dev)   # clip -> global id
        self.world_ids = np.arange(n) if world_ids is None else np.asarray(world_ids, np.int64)
        self.rows, self.start = z(n, dt=torch.long), z(n, dt=torch.long)
        self.episode = np.zeros(n, np.int64)
        self.delta = z(n, self.R, dt=dtype)
        self.sums, self.raw = z(n, 5), z(n)
        self.contact = z(n, 3, dt=dtype)
        self.alpha, self.next_alpha = z(n), z(n)   # float64: the curriculum compares levels exactly (EnvBook)
        Tm = max(sp.T for sp in self.specs)
        self.buf = torch.zeros(n, Tm, self.specs[0].nu, dtype=torch.float32, device=self.dev)
        self.dmax = torch.as_tensor(delta_scale(cfg, self.R), dtype=dtype, device=self.dev)
        self.next_clip = self.clip.clone()         # clip of each world's next episode (a union may reassign it)

    @property
    def n_worlds(self) -> int:
        return int(self.rows.shape[0])

    def _reset(self, mask: Tensor) -> None:
        self.clip = torch.where(mask, self.next_clip, self.clip)
        Tc = self.ref.T.cpu().numpy()
        clip_h, start = self.clip.cpu().numpy(), np.zeros(self.n_worlds, np.int64)
        for w in np.flatnonzero(mask.cpu().numpy()):
            start[w] = draw_start(self.cfg, int(Tc[clip_h[w]]), int(self.world_ids[w]), int(self.episode[w]))
            self.episode[w] += 1
        st = torch.as_tensor(start, device=self.dev)
        self.backend.reset(mask, self.clip, st)
        self.start = torch.where(mask, st, self.start)
        self.rows = torch.where(mask, st, self.rows)
        self.delta[mask] = 0.0
        self.sums[mask] = 0.0
        self.raw[mask] = 0.0
        self.contact[mask] = 0.0
        self.alpha[mask] = self.next_alpha[mask]

    def _obs(self) -> Tensor:
        q, v = self.backend.state()
        return observe_s(self.cfg, self.ref, self.clip, self.onehot[self.clip], self.rows, q.to(self.dev, self.dt),
                         v.to(self.dev, self.dt), self.delta, self.sums[:, :2].to(self.dt), self.contact,
                         self.n_clips, self.alpha.to(self.dt))

    def reset_all(self, alpha: Tensor | None = None) -> Tensor:
        if alpha is not None:
            self.next_alpha[:] = alpha.to(torch.float64)
        self._reset(torch.ones(self.n_worlds, dtype=torch.bool, device=self.dev))
        return self._obs()

    def step(self, actions: Tensor, next_alpha: Tensor | None = None
             ) -> tuple[Tensor, Tensor, Tensor, dict[str, np.ndarray], list[dict[str, Any]]]:
        cfg, K, p = self.cfg, self.cfg.rows_per_step, self.ref
        if next_alpha is not None:
            self.next_alpha[:] = next_alpha.to(torch.float64)
        target = self.dmax * torch.tanh(actions.to(self.dev, self.dt))
        n = self.n_worlds
        b = lambda: torch.zeros(n, dtype=torch.bool, device=self.dev)
        executed = torch.zeros(n, dtype=torch.long, device=self.dev)
        invalid, drop, et = b(), b(), b()
        raw_step = torch.zeros(n, dtype=torch.float64, device=self.dev)
        alpha_ep, clip, rsi = self.alpha.clone(), self.clip, self.start > 0
        idx = torch.arange(n, device=self.dev)
        W, T, max_sim = p.W[clip], p.T[clip], p.max_sim[clip]
        for j in range(K):
            res = target if cfg.hold == "zoh" else self.delta + (j + 1) / K * (target - self.delta)
            rows = self.rows.clone()                   # this row's index (self.rows advances below)
            live = ~(invalid | drop | et)
            ctrl = p.ctrl_ref[clip, rows].clone()
            ctrl[:, :self.R] += res
            c32 = ctrl.to(torch.float32)
            pen, ho, ped, bad = (x.to(self.dev) for x in self.backend.step_row(c32, rows, self.alpha.to(self.dt)))
            q, v = (x.to(self.dev, self.dt) for x in self.backend.state())
            w, pos, rot = row_terms_s(p, clip, rows, q, v, pen.to(self.dt), ho, ped)
            lf = live.to(torch.float64)
            raw_step += w.to(torch.float64) * lf
            post = ((rows >= W) & (rows < torch.minimum(T, max_sim))).to(torch.float64) * lf
            self.sums += torch.stack([pos.double() * lf, rot.double() * lf, pos.double() * post,
                                      rot.double() * post, post], dim=1)
            rc = torch.clamp(rows, max=self.buf.shape[1] - 1)      # no boolean indexing: no host sync per row
            self.buf[idx, rc] = torch.where(live[:, None], c32, self.buf[idx, rc])
            self.contact = torch.where(live[:, None], torch.stack([pen.to(self.dt), ho.to(self.dt),
                                                                  ped.to(self.dt)], 1), self.contact)
            self.rows += live.long()
            executed += live.long()
            invalid |= live & bad
            if cfg.rsi_et:
                et |= live & ~bad & rsi & ((pos > SUCCESS_POS_M) | (rot > SUCCESS_ROT_RAD))
            blk_end = (rows + 1) % p.ctrl_steps[clip] == 0
            drop |= live & ~bad & ~et & blk_end & (q[:, p.a0 + 2] < p.terminate_z[clip])
        self.delta = target.clone()
        self.raw += raw_step
        Tf = T.to(torch.float64)                  # futility thresholds in float64, as the numpy path
        ended = invalid | drop | et
        clip_end = ~ended & (self.rows >= T)
        futile = ~ended & ~clip_end & ~rsi & ((self.sums[:, 0] >= SUCCESS_POS_M * Tf)
                                             | (self.sums[:, 1] >= SUCCESS_ROT_RAD * Tf))
        if not cfg.futility:
            futile = torch.zeros_like(futile)
        done = ended | clip_end | futile
        complete = (clip_end | drop) & (self.rows >= max_sim) & ~rsi
        nrows = torch.clamp(self.rows, min=1).double()
        whole_p, whole_r = self.sums[:, 0] / nrows, self.sums[:, 1] / nrows
        npost = torch.clamp(self.sums[:, 4], min=1.0)
        crit_p = complete & (whole_p < SUCCESS_POS_M) & (whole_r < SUCCESS_ROT_RAD)
        crit_s = complete & (self.sums[:, 4] > 0) & (self.sums[:, 2] / npost < SUCCESS_POS_M) & (
            self.sums[:, 3] / npost < SUCCESS_ROT_RAD)
        free = alpha_ep == 0.0
        primary, strict = crit_p & free, crit_s & free
        rew = cfg.alive_bonus * executed.double() + raw_step + cfg.success_bonus * primary.double()
        end = torch.zeros(n, dtype=torch.long, device=self.dev)
        for name, m in (("clip_end", clip_end), ("futile", futile), ("drop", drop), ("invalid", invalid),
                        ("early_term", et)):
            end = torch.where(m, torch.full_like(end, END_CODES[name]), end)
        host = {k: x.cpu().numpy() for k, x in (("done", done), ("primary", primary), ("strict", strict),
                                                ("end", end), ("rows", executed), ("ep_rows", self.rows.clone()),
                                                ("whole_p", whole_p), ("whole_r", whole_r),
                                                ("post_p", self.sums[:, 2] / npost), ("post_r", self.sums[:, 3] / npost),
                                                ("raw", raw_step), ("alpha", alpha_ep), ("ap", crit_p & ~free),
                                                ("ep_raw", self.raw.clone()), ("clip", self.onehot[clip]))}
        successes = []
        for w in np.flatnonzero(host["primary"]):
            nr = int(host["ep_rows"][w])
            successes.append({"task": self.specs[int(clip[w])].task, "clip": int(host["clip"][w]), "world": int(w),
                              "ctrl": self.buf[w, :nr].cpu().numpy(),
                              "whole_pos_m": float(host["whole_p"][w]), "whole_rot_rad": float(host["whole_r"][w]),
                              "post_pos_m": float(host["post_p"][w]), "post_rot_rad": float(host["post_r"][w]),
                              "strict": bool(host["strict"][w]), "raw_return": float(host["ep_raw"][w])})
        d = host["done"]
        info = {"raw_reward": host["raw"], "rows": host["rows"], "end": np.where(d, host["end"], 0),
                "primary": host["primary"], "strict": host["strict"], "clip": host["clip"],
                "ep_whole_pos": np.where(d, host["whole_p"], np.nan), "ep_rows": np.where(d, host["ep_rows"], 0),
                "alpha": host["alpha"], "assisted_primary": host["ap"]}
        if bool(done.any()):
            self._reset(done)
        return self._obs(), (cfg.reward_scale * rew).to(torch.float32), done, info, successes

    def snapshot(self) -> dict[str, Any]:
        return {}

    def restore(self, snap: dict[str, Any]) -> Tensor:
        return self.reset_all()
