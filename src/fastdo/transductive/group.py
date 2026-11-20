"""Episode logic of a group of worlds on one simulator backend: residual -> control rows, reward, episode end,
success bookkeeping, observation, auto-reset. Runs inside a worker process (or in the learner for tests).

Per world and policy step (`EnvConfig` for the definitions): the residual target from the action, the applied rows
`ctrl_ref[s .. s+k) + residual`, the backend's `step_block`, DO AS I DO's reward and tracking errors on the executed
rows, the episode-end rules, and at the clip's end the scorer's success criteria on the training rollout:
PRIMARY = whole-trace means over the executed rows below 0.1 m / 0.5 rad with the clip completed in the scorer's
sense (executed rows >= max_sim_steps: an executor termination in the last block still completes the clip); STRICT
= the same on rows [W, min(T, max_sim_steps)). RSI episodes (`EnvConfig`: start at a reference row, early termination on the scorer's per-row thresholds, no futility)
never succeed. A PRIMARY success hands back the episode's applied control rows (float32,
exactly what was executed) for certification by deterministic re-execution (`fastdo.transductive.certify`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.config import SUCCESS_POS_M, SUCCESS_ROT_RAD, EnvConfig, draw_start
from fastdo.transductive.obs import delta_scale, observe
from fastdo.transductive.reward import row_costs, row_tracking
from fastdo.transductive.sim_cpu import RowBlock

__all__ = ["END_CODES", "Backend", "WorldGroup"]

END_CODES = {"running": 0, "clip_end": 1, "drop": 2, "invalid": 3, "futile": 4, "incomplete": 5, "early_term": 6}


class Backend(Protocol):
    """A simulator over independent worlds (`fastdo.transductive.sim_cpu.CpuWorlds` and the GPU backend)."""

    world_clip: np.ndarray
    row: np.ndarray

    @property
    def n_worlds(self) -> int: ...
    def spec(self, i: int) -> ClipSpec: ...
    def reset(self, i: int, row: int = 0, settle: int = 20) -> None: ...
    def qpos(self, i: int) -> np.ndarray: ...
    def qvel(self, i: int) -> np.ndarray: ...
    def step_rows(self, i: int, ctrls: np.ndarray) -> RowBlock: ...
    def get_state(self, i: int) -> np.ndarray: ...
    def set_state(self, i: int, state: np.ndarray) -> None: ...


class WorldGroup:
    """Worlds of `sim`; `clip_ids[c]` is the global index of the backend's clip c (for the one-hot)."""

    def __init__(self, cfg: EnvConfig, sim: Backend, clip_ids: Sequence[int], n_clips: int,
                 world_ids: Sequence[int] | None = None) -> None:
        self.cfg, self.sim = cfg, sim
        self.clip_ids = np.asarray(clip_ids, np.int64)
        self.n_clips = int(n_clips)
        n = sim.n_worlds
        R = sim.spec(0).n_robot
        if any(sim.spec(i).n_robot != R for i in range(n)):
            raise ValueError("one group needs one robot layout (separate groups per hand side)")
        self.R = R
        self.delta = np.zeros((n, R))
        self.sums = np.zeros((n, 5))                      # pos, rot, post pos, post rot, post rows
        self.raw = np.zeros(n)
        self.contact = [(0.0, False, False)] * n
        self.buf = [np.zeros((sim.spec(i).T, sim.spec(i).nu), np.float32) for i in range(n)]
        self.next_alpha = np.zeros(n)                      # assist level of each world's NEXT episode
        self.world_ids = np.arange(n) if world_ids is None else np.asarray(world_ids, np.int64)
        self.start = np.zeros(n, np.int64)                 # start row of the current episode (> 0: RSI)
        self.episode = np.zeros(n, np.int64)               # episodes started so far (the RSI draw's counter)

    @property
    def n_worlds(self) -> int:
        return self.sim.n_worlds

    def _reset_world(self, i: int) -> None:
        row = draw_start(self.cfg, self.sim.spec(i).T, int(self.world_ids[i]), int(self.episode[i]))
        self.episode[i] += 1
        self.start[i] = row
        self.sim.reset(i, row, self.cfg.rsi_settle)
        self.sim.alpha[i] = self.next_alpha[i]
        self.delta[i] = 0.0
        self.sums[i] = 0.0
        self.raw[i] = 0.0
        self.contact[i] = (0.0, False, False)

    def _obs(self, i: int) -> np.ndarray:
        sp = self.sim.spec(i)
        s = int(self.sim.row[i])
        return observe(self.cfg, sp, s, np.asarray(self.sim.qpos(i)), np.asarray(self.sim.qvel(i)), self.delta[i],
                       (self.sums[i, 0], self.sums[i, 1]), self.contact[i],
                       int(self.clip_ids[self.sim.world_clip[i]]), self.n_clips, float(self.sim.alpha[i]))

    def reset_all(self, alpha: np.ndarray | None = None) -> np.ndarray:
        if alpha is not None:
            self.next_alpha[:] = alpha
        for i in range(self.n_worlds):
            self._reset_world(i)
        return np.stack([self._obs(i) for i in range(self.n_worlds)])

    def residual_rows(self, i: int, target: np.ndarray, k: int) -> np.ndarray:
        """(k, R) residual applied over the next k rows, ending at `target` after rows_per_step rows."""
        K = self.cfg.rows_per_step
        if self.cfg.hold == "zoh":
            return np.repeat(target[None], k, axis=0)
        f = (np.arange(1, k + 1) / K)[:, None]
        return self.delta[i][None] + f * (target - self.delta[i])[None]

    def step(self, actions: np.ndarray, next_alpha: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
        """Advance every world by one policy step (actions -> residual rows); a world that resets starts its next
        episode at assist level `next_alpha` (default: unchanged). Returns (obs, reward, done, info, successes)."""
        if next_alpha is not None:
            self.next_alpha[:] = next_alpha
        rows = []
        for i in range(self.n_worlds):
            sp = self.sim.spec(i)
            s0 = int(self.sim.row[i])
            k = min(self.cfg.rows_per_step, sp.T - s0)
            target = delta_scale(self.cfg, self.R) * np.tanh(np.asarray(actions[i], np.float64))
            res = self.residual_rows(i, target, k)
            u = sp.ctrl_ref[s0:s0 + k].copy()
            u[:, :self.R] += res
            rows.append((u, target, res))
        return self._advance(rows)

    def step_controls(self, ctrls: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                   dict[str, np.ndarray], list[dict[str, Any]]]:
        """Advance every world by explicit control rows (known answers, certification); `None` or an empty array
        for a world ends its episode as incomplete (no controls left: the run stops short of the clip)."""
        return self._advance([(None if u is None else np.asarray(u, np.float64), None, None) for u in ctrls])

    def _advance(self, rows: list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
        n, cfg = self.n_worlds, self.cfg
        rew, done = np.zeros(n), np.zeros(n, bool)
        info = {"raw_reward": np.zeros(n), "rows": np.zeros(n, np.int64), "end": np.zeros(n, np.int64),
                "primary": np.zeros(n, bool), "strict": np.zeros(n, bool), "clip": self.clip_ids[self.sim.world_clip],
                "ep_whole_pos": np.full(n, np.nan), "ep_rows": np.zeros(n, np.int64),
                "alpha": self.sim.alpha.copy(), "assisted_primary": np.zeros(n, bool)}
        successes: list[dict[str, Any]] = []
        out_obs = []
        for i in range(n):
            sp = self.sim.spec(i)
            s0 = int(self.sim.row[i])
            u, target, res = rows[i]
            if u is None or u.shape[0] == 0:
                blk = None
                m = 0
                end: str | None = "incomplete"
            else:
                blk = self.sim.step_rows(i, u)
                m = blk.rows.shape[0]
                end = blk.end
            if target is not None:
                self.delta[i] = target if m == u.shape[0] else res[m - 1]
            raw = 0.0
            rsi = self.start[i] > 0
            if blk is not None and m:
                pos, rot = row_tracking(sp.rw, sp.ex.obj_qadr, blk.rows, blk.qpos)
                viol = (pos > SUCCESS_POS_M) | (rot > SUCCESS_ROT_RAD)
                if rsi and cfg.rsi_et and viol.any():     # RSI early termination at the first violating row
                    m = int(np.argmax(viol)) + 1
                    for name in ("rows", "qpos", "qvel", "pen", "ho", "ped", "ctrl"):
                        setattr(blk, name, getattr(blk, name)[:m])
                    pos, rot = pos[:m], rot[:m]
                    end = "early_term"
                self.buf[i][s0:s0 + m] = blk.ctrl
                rc = row_costs(sp.rw, sp.ex.obj_qadr, sp.W, blk.rows, blk.qpos, blk.qvel, blk.pen, blk.ho, blk.ped)
                post = (blk.rows >= sp.W) & (blk.rows < min(sp.T, sp.max_sim_steps))
                self.sums[i] += [pos.sum(), rot.sum(), pos[post].sum(), rot[post].sum(), post.sum()]
                raw = float(rc.weighted.sum())
                self.contact[i] = (float(blk.pen[-1]), bool(blk.ho[-1]), bool(blk.ped[-1]))
            self.raw[i] += raw
            r = cfg.alive_bonus * m + raw
            s1 = s0 + m
            T = sp.T
            if end is None and s1 >= T:
                end = "clip_end"
            elif end is None and cfg.futility and not rsi and (self.sums[i, 0] >= SUCCESS_POS_M * T
                                                   or self.sums[i, 1] >= SUCCESS_ROT_RAD * T):
                end = "futile"
            info["raw_reward"][i], info["rows"][i] = raw, m
            if end is not None:
                done[i] = True
                info["end"][i] = END_CODES[end]
                info["ep_rows"][i] = s1
                info["ep_whole_pos"][i] = self.sums[i, 0] / s1 if s1 else np.nan
                # the scorer's completed_clip is executed rows >= max_sim_steps: a drop in the last block still counts
                if end in ("clip_end", "drop") and s1 >= sp.max_sim_steps and not rsi:
                    r += self._finish(i, sp, s1, info, successes)
                self._reset_world(i)
            rew[i] = cfg.reward_scale * r
            out_obs.append(self._obs(i))
        return np.stack(out_obs), rew, done, info, successes

    def _finish(self, i: int, sp: ClipSpec, n: int, info: dict[str, np.ndarray],
                successes: list[dict[str, Any]]) -> float:
        """The scorer's criteria on a world that executed n >= max_sim_steps rows (whole-trace means over the n
        executed rows, STRICT over [W, max_sim_steps)); returns the success bonus earned."""
        whole_p, whole_r = self.sums[i, 0] / n, self.sums[i, 1] / n
        npost = self.sums[i, 4]
        post_p = self.sums[i, 2] / npost if npost else np.inf
        post_r = self.sums[i, 3] / npost if npost else np.inf
        primary = bool(whole_p < SUCCESS_POS_M and whole_r < SUCCESS_ROT_RAD)
        strict = bool(post_p < SUCCESS_POS_M and post_r < SUCCESS_ROT_RAD)
        if self.sim.alpha[i] > 0.0:                   # assisted: curriculum bookkeeping only, never a success
            info["assisted_primary"][i] = primary
            return 0.0
        info["primary"][i], info["strict"][i] = primary, strict
        if not primary:
            return 0.0
        successes.append({"task": sp.task, "clip": int(self.clip_ids[self.sim.world_clip[i]]), "world": i,
                          "ctrl": self.buf[i][:n].copy(),
                          "whole_pos_m": float(whole_p), "whole_rot_rad": float(whole_r),
                          "post_pos_m": float(post_p), "post_rot_rad": float(post_r), "strict": strict,
                          "raw_return": float(self.raw[i])})
        return self.cfg.success_bonus

    def snapshot(self) -> dict[str, Any]:
        """Exact episode state of every world (backend states, residuals, sums, applied-control buffers)."""
        return {"sim": [self.sim.get_state(i) for i in range(self.n_worlds)], "alpha": self.sim.alpha.copy(),
                "next_alpha": self.next_alpha.copy(), "delta": self.delta.copy(), "start": self.start.copy(),
                "episode": self.episode.copy(),
                "sums": self.sums.copy(), "raw": self.raw.copy(), "contact": list(self.contact),
                "buf": [b.copy() for b in self.buf]}

    def restore(self, snap: dict[str, Any]) -> np.ndarray:
        for i, st in enumerate(snap["sim"]):
            self.sim.set_state(i, st)
        self.sim.alpha[:], self.next_alpha[:] = snap["alpha"], snap["next_alpha"]
        self.start[:], self.episode[:] = snap["start"], snap["episode"]
        self.delta[:], self.sums[:], self.raw[:] = snap["delta"], snap["sums"], snap["raw"]
        self.contact = [tuple(c) for c in snap["contact"]]
        self.buf = [np.asarray(b, np.float32).copy() for b in snap["buf"]]
        return np.stack([self._obs(i) for i in range(self.n_worlds)])
