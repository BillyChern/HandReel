"""Tracker training loop on the DEVICE backend.

`DeviceTrainer` is `fastdo.amortised.trainer.Trainer` with four seams replaced, everything else (replay, learner
updates at the pinned UTD, assist schedule, success store, inline CPU certification, episode log, stats) unchanged:
- environment: `tracker_gpu.DeviceTrackerEnv` (`backend` "warp" on CUDA or warp's CPU device; "cpu-pool" = CPU
  MuJoCo, the test reference), worlds bound to clips in contiguous blocks (`grouped_world_clip`): the clip sampler
  does not choose a world's clip (per-clip pools cannot rebind), it still records failures and certified clips;
- acting: one eval-mode actor pass for all worlds on the learner's device (`acting.act_batch`) with the vectorised
  counter-based noise `acting.noise_hash` (candidates record `noise="hash"`, so the CPU certificate regenerates it);
- observation: `tracker_dev.observe_t` on the device, for acting and for the replay rebuild (the HOST replay's
  sampled records are uploaded per minibatch: a state-compressed, HOST-resident replay);
- learner: `SAC` on the device; the reward normaliser keeps its running state on the host.

Not supported on this path (it runs one 15-min unit per arm, with no extension): resume. `resume` refuses;
`save_checkpoint` still writes the learner, replay and log state for the record, with an empty environment state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastdo.amortised.acting import act_batch, episode_seed, noise_hash
from fastdo.amortised.tracker_dev import observe_t
from fastdo.amortised.tracker_gpu import DeviceTrackerEnv, grouped_world_clip
from fastdo.amortised.trainer import Episode, Trainer, TrainerConfig, TrainInputs

__all__ = ["DeviceTrainer"]


class DeviceTrainer(Trainer):
    """Module docstring. `backend`/`device`/`sim_device` as `DeviceTrackerEnv`."""

    def __init__(self, cfg: TrainerConfig, inp: TrainInputs, out_dir: Path, *, backend: str = "warp",
                 device: str = "cuda:0", sim_device: str | None = None) -> None:
        self.backend_name, self.dev, self.sim_device = backend, torch.device(device), sim_device
        self.world_clip = np.asarray(grouped_world_clip(cfg.n_worlds, len(inp.tasks)), np.int64)
        super().__init__(cfg, inp, out_dir)

    # ------------------------------------------------------------------------------------------ seams
    def _agent_device(self) -> str:
        return str(self.dev)

    def _make_env(self) -> DeviceTrackerEnv:
        cfg, inp = self.cfg, self.inp
        return DeviceTrackerEnv(cfg.tracker, inp.specs, self.table, inp.tips, inp.sdfs, self.world_clip,
                                backend=self.backend_name, device=self.dev, sim_device=self.sim_device,
                                reset_hook=self._hook, assist=cfg.assist.enabled)

    def _hook(self, i: int) -> tuple[int, float, int]:
        """`Trainer._hook` with the world's own clip (no sampler draw); alpha and RSI as on the CPU path."""
        clip = int(self.world_clip[i])
        alpha = self.assist.world_alpha(i, clip)
        row = 0
        if self.cfg.rsi_prob > 0.0 and self.rng.random() < self.cfg.rsi_prob:
            sp = self.inp.specs[clip]
            row = int(self.rng.integers(sp.W, max(sp.W + 1, sp.T - 2 * self.cfg.tracker.k)))
        e = self.n_episodes
        self.n_episodes += 1
        self._pending[i] = Episode(clip=clip, episode=e, seed=episode_seed(self.cfg.seed, e), start_row=row,
                                   alpha=alpha)
        return clip, alpha, row

    def _act(self) -> np.ndarray:
        obs, _ = observe_t(self.env.dt, self.env.record_dev, self.cfg.tracker)
        seeds = np.array([e.seed for e in self.eps], np.int64)
        steps = np.array([len(e.versions) for e in self.eps], np.int64)
        noise = torch.from_numpy(noise_hash(seeds, steps, self.table.R)).to(self.dev)
        return act_batch(self.behaviour, obs, noise).detach().cpu().numpy().astype(np.float32)

    def _observe_batch(self, rec: dict[str, np.ndarray]) -> tuple[Any, Any]:
        up = {k: torch.as_tensor(np.ascontiguousarray(v), device=self.dev) for k, v in rec.items()}
        return observe_t(self.env.dt, up, self.cfg.tracker)

    def _batch(self, b: dict[str, Any]) -> dict[str, Any]:
        d = self.dev
        return {"actor_obs": b["obs"][0], "critic_obs": b["obs"][1], "next_actor_obs": b["next_obs"][0],
                "next_critic_obs": b["next_obs"][1], "action": torch.as_tensor(b["action"]).to(d),
                "reward": b["reward"], "terminated": torch.as_tensor(b["terminated"]).to(d)}

    def _candidate_noise(self) -> str:
        return "hash"

    def resume(self) -> bool:
        if self.latest(self.out) is not None:
            raise RuntimeError("the device trainer does not resume (one unit per arm, design §8.2); "
                               f"{self.out} already holds a checkpoint")
        return False
