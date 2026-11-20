"""The GPU frontend of the transductive environment: one `BatchGroup` of `WarpPool`s (one pool per clip) in the
learner's process, all tensors on the device; the same `TrackingEnv` protocol and bookkeeping (`EnvBook`) as the CPU
`TransductiveEnv`, so `fastdo.transductive.trainer.train` drives either.

`backend="warp"`: DO AS I DO's simulator (mujoco_warp 0.0.1; `retargeting` env; CUDA, or warp's CPU device for
probes), one pool per clip. `backend="cpu-pool"`: the CPU backend as per-clip pools (`CpuPool`); `"cpu-batch"`: CPU
worlds of any clip behind the union-style interface (`CpuBatchBackend`). Both CPU variants are test references.
On CUDA the env first makes a dedicated torch stream current (warp's captured step graphs run on it; see
`fastdo.transductive.sim_warp`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.transductive.batch_group import BatchGroup, CpuBatchBackend, CpuPool, PoolsBackend
from fastdo.transductive.clip import ClipSpec, load_clip
from fastdo.transductive.config import EnvConfig
from fastdo.transductive.env import EnvBook
from fastdo.transductive.obs import obs_dim

__all__ = ["BatchedEnv"]


class BatchedEnv(EnvBook):
    """`TrackingEnv` over `tasks` on one device (module docstring)."""

    def __init__(self, cfg: EnvConfig, source: dict[str, ClipSpec] | Path | str, tasks: Sequence[str],
                 worlds_per_clip: int, backend: str = "warp", device: str = "cuda:0",
                 sim_device: str | None = None) -> None:
        dev = torch.device(device)
        if dev.type == "cuda":
            self.stream = torch.cuda.Stream(dev)
            torch.cuda.set_stream(self.stream)
        self._init_book(cfg, tasks, np.repeat(np.arange(len(tasks)), worlds_per_clip), worlds_per_clip, dev)
        specs = [source[t] if isinstance(source, dict) else load_clip(Path(source), t) for t in self.tasks]
        clips = list(range(len(specs)))
        if backend == "warp":
            from fastdo.transductive.sim_warp import WarpPool

            sd = sim_device or str(device)
            be: Any = PoolsBackend([WarpPool(sp, worlds_per_clip, device=sd, assist_wn=cfg.assist_wn,
                                             assist=cfg.assist, settle=cfg.rsi_settle,
                                             assist_track=cfg.assist_track) for sp in specs], clips)
            dtype = torch.float32
        elif backend == "cpu-pool":
            be = PoolsBackend([CpuPool(sp, worlds_per_clip, assist_wn=cfg.assist_wn, settle=cfg.rsi_settle,
                                       assist_track=cfg.assist_track)
                               for sp in specs], clips)
            dtype = torch.float64
        elif backend == "cpu-batch":
            be = CpuBatchBackend(specs, self.world_clip.tolist(), cfg.assist_wn, settle=cfg.rsi_settle,
                                 assist_track=cfg.assist_track)
            dtype = torch.float64
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.group = BatchGroup(cfg, be, specs, self.world_clip.tolist(), clips, len(specs), dev, dtype)
        self._obs_dim = obs_dim(cfg, specs[0].n_robot, self.n_clips)
        self._act_dim = specs[0].n_robot

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def priv_dim(self) -> int:
        return 0

    @property
    def act_dim(self) -> int:
        return self._act_dim

    def privileged(self) -> Tensor:
        return torch.zeros(self.n_worlds, 0, device=self.device)

    def _alpha(self) -> Tensor:
        return torch.as_tensor(self.next_alpha(), device=self.device)

    def reset(self, ref: Any = None, world_ids: Tensor | None = None) -> Tensor:
        if ref is not None or world_ids is not None:
            raise NotImplementedError("the transductive env resets all worlds onto their own clips")
        return self.group.reset_all(self._alpha())

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        obs, rew, done, info, succ = self.group.step(action, self._alpha())
        self._account(info, succ)
        return obs, rew.to(self.device), done, info

    def snapshot(self) -> dict[str, Any]:
        return self.group.snapshot()

    def restore(self, snap: dict[str, Any]) -> Tensor:
        return self.group.reset_all(self._alpha())

    def close(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
