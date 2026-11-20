"""Host-resident, state-compressed replay with observation REBUILD (design §4.2 "State-compressed, HOST-resident
replay"; milestones M3/M4).

Layout: a ring of `capacity` vectorised steps x `n_worlds` worlds. Each slot holds the RECORD of the pre-step state
(the compressed per-transition fields, e.g. qpos fp32, fingertips fp16, clip id and row int32), the action, the
reward and the termination / truncation flags of that step. Observations are never stored: `sample` gathers the
records and the caller's `observe(records)` rebuilds them (bitwise equal to the observation the policy saw, because
the environment computes its observation from exactly these records with the same function).

n-step transitions are aggregated at SAMPLE time with FlashSAC's semantics (`fastdo.amortised.sac.nstep_aggregate`):
for (step s, world w), j* = the first done in s .. s+n-1 (n-1 if none); the next record is the record of step
s + j* + 1 of the same world, or, if step s + j* ended the episode, the FINAL record stored with it (the post-step
state before the auto-reset), as the vendored memory-efficient buffer keeps final observations. Final records live in
a dense POOL of `final_capacity` records with an index ring `final_idx` (capacity, n_worlds) (-1 = no episode end);
both writing (`add`) and reading (`gather`) are vectorised over worlds (no Python loop per world, for M9's 8k worlds).
A pool entry is reused round-robin; overwriting one that a step still in the ring references raises (size the pool by
the episode-end rate: >= one end per T/k steps per world; the default, capacity x n_worlds, can never overflow).
A step s is sampleable once step s + n has been written and while step s is still in the ring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from fastdo.amortised.sac import nstep_aggregate

__all__ = ["HostReplay"]


class HostReplay:
    def __init__(self, fields: Mapping[str, tuple[tuple[int, ...], np.dtype]], n_worlds: int, capacity: int,
                 act_dim: int, n_step: int, gamma: float, final_capacity: int | None = None) -> None:
        if capacity <= n_step + 1:
            raise ValueError("capacity must exceed n_step + 1 vectorised steps")
        self.n_worlds, self.capacity, self.n_step, self.gamma = int(n_worlds), int(capacity), int(n_step), float(gamma)
        self.rec = {k: np.zeros((capacity, n_worlds, *shape), dtype) for k, (shape, dtype) in fields.items()}
        self.action = np.zeros((capacity, n_worlds, act_dim), np.float32)
        self.reward = np.zeros((capacity, n_worlds), np.float32)
        self.term = np.zeros((capacity, n_worlds), np.float32)
        self.trunc = np.zeros((capacity, n_worlds), np.float32)
        fcap = int(final_capacity) if final_capacity is not None else capacity * n_worlds
        if fcap < n_worlds:
            raise ValueError("final_capacity must hold at least one episode end per world")
        self.final = {k: np.zeros((fcap, *shape), dtype) for k, (shape, dtype) in fields.items()}
        self.final_idx = np.full((capacity, n_worlds), -1, np.int32)
        self._final_owner = np.full((fcap, 2), -1, np.int32)        # (slot, world) that references each pool entry
        self._final_next = 0
        self.steps = 0

    def __len__(self) -> int:
        return min(self.steps, self.capacity) * self.n_worlds

    @property
    def nbytes_per_transition(self) -> int:
        per = sum(a.dtype.itemsize * int(np.prod(a.shape[2:])) for a in self.rec.values())
        return per + 4 * (self.action.shape[-1] + 3)

    def add(self, rec: Mapping[str, np.ndarray], action: np.ndarray, reward: np.ndarray, term: np.ndarray,
            trunc: np.ndarray, final: Mapping[str, np.ndarray] | None = None) -> None:
        """One vectorised step: `rec` = records of the PRE-step states (n_worlds, ...); `final` = records of the
        post-step states (n_worlds, ...), read for the worlds whose episode ended (term or trunc)."""
        slot = self.steps % self.capacity
        for k, a in self.rec.items():
            a[slot] = rec[k]
        self.action[slot], self.reward[slot] = action, reward
        self.term[slot], self.trunc[slot] = term, trunc
        self.final_idx[slot] = -1                                   # the overwritten step's ends are gone
        done = np.flatnonzero((np.asarray(term) > 0) | (np.asarray(trunc) > 0))
        if done.size:
            if final is None:
                raise ValueError("episode ends need their final records")
            fcap = self._final_owner.shape[0]
            if done.size > fcap:
                raise ValueError(f"{done.size} episode ends > final_capacity {fcap}")
            pos = (self._final_next + np.arange(done.size)) % fcap
            own = self._final_owner[pos]
            live = (own[:, 0] >= 0) & (self.final_idx[np.maximum(own[:, 0], 0), np.maximum(own[:, 1], 0)] == pos)
            if live.any():
                raise ValueError("final-record pool overflow: an entry still referenced by a step in the ring would "
                                 "be overwritten (raise final_capacity)")
            for k, a in self.final.items():
                a[pos] = np.asarray(final[k])[done]
            self.final_idx[slot, done] = pos
            self._final_owner[pos, 0], self._final_owner[pos, 1] = slot, done
            self._final_next = int((self._final_next + done.size) % fcap)
        self.steps += 1

    def sample_range(self) -> tuple[int, int]:
        """[lo, hi) of sampleable vectorised steps."""
        return max(0, self.steps - self.capacity), self.steps - self.n_step

    def sample_indices(self, rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.sample_range()
        if hi <= lo:
            raise ValueError("not enough steps to sample")
        return rng.integers(lo, hi, size=batch), rng.integers(0, self.n_worlds, size=batch)

    def gather(self, steps: np.ndarray, worlds: np.ndarray) -> dict[str, object]:
        """The n-step batch of (steps, worlds): records, next records, action, R, term, trunc, j*."""
        steps, worlds = np.asarray(steps, np.int64), np.asarray(worlds, np.int64)
        lo, hi = self.sample_range()
        if steps.size and (steps.min() < lo or steps.max() >= hi):
            raise IndexError(f"steps outside the sampleable range [{lo}, {hi})")
        n, cap = self.n_step, self.capacity
        win = (steps[None, :] + np.arange(n)[:, None]) % cap
        rew = torch.as_tensor(self.reward[win, worlds[None, :]])
        term = torch.as_tensor(self.term[win, worlds[None, :]])
        trunc = torch.as_tensor(self.trunc[win, worlds[None, :]])
        R, jstar, t_out, tr_out = nstep_aggregate(rew, term, trunc, self.gamma)
        j = jstar.numpy()
        slot = steps % cap
        rec = {k: a[slot, worlds] for k, a in self.rec.items()}
        nxt_slot = (steps + j + 1) % cap
        nxt = {k: a[nxt_slot, worlds].copy() for k, a in self.rec.items()}
        end_slot = (steps + j) % cap
        ended = np.flatnonzero((t_out.numpy() > 0) | (tr_out.numpy() > 0))
        if ended.size:
            pos = self.final_idx[end_slot[ended], worlds[ended]]
            if (pos < 0).any():
                raise RuntimeError("an episode end without its final record")
            for k, a in nxt.items():
                a[ended] = self.final[k][pos]
        return {"rec": rec, "next_rec": nxt, "action": self.action[slot, worlds], "reward": R, "terminated": t_out,
                "truncated": tr_out, "jstar": jstar}

    _STATE = ("action", "reward", "term", "trunc", "final_idx", "_final_owner")

    def state_dict(self) -> dict[str, np.ndarray]:
        """Every array and counter (resume, M5): records as `rec.<field>`, final pool as `final.<field>`."""
        out = {f"rec.{k}": a for k, a in self.rec.items()} | {f"final.{k}": a for k, a in self.final.items()}
        out |= {k: getattr(self, k) for k in self._STATE}
        out["counters"] = np.array([self.steps, self._final_next], np.int64)
        return out

    def load_state_dict(self, st: dict[str, np.ndarray]) -> None:
        for k, a in self.rec.items():
            a[...] = st[f"rec.{k}"]
        for k, a in self.final.items():
            a[...] = st[f"final.{k}"]
        for k in self._STATE:
            getattr(self, k)[...] = st[k]
        self.steps, self._final_next = (int(x) for x in st["counters"])

    def sample(self, rng: np.random.Generator, batch: int,
               observe: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None = None) -> dict[str, object]:
        out = self.gather(*self.sample_indices(rng, batch))
        if observe is not None:
            out["obs"] = observe(out["rec"])
            out["next_obs"] = observe(out["next_rec"])
        return out
