"""Training schedules: the per-clip assist level and the clip sampler.

Assist (training only; the object wrench of `fastdo.transductive.sim_cpu.CpuWorlds._assist`): per clip, alpha_c
starts at `start` (1) and decays by `step` (0.1) each time at least `pass_frac` (50%) of a block of `window` (16)
consecutive ASSISTED episodes of that clip pass (non-overlapping blocks); one world in `unassisted_every` (4) always
runs alpha = 0 (worlds w with w % 4 == 0), so unassisted episodes exist from the start. With `enabled=False` every
world runs alpha = 0. Only alpha = 0 episodes from row 0 may produce stored successes (`SuccessStore.offer`).

Clip sampler: p(i) proportional to eps + 1[i not certified] (1 + failure rate over its last `window` (16) episodes),
eps = 0.05; an empty history has failure rate 0. A certified clip (and, by the null-motion exclusion, an
MT-null clip after its first certified success) keeps weight eps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["AssistConfig", "AssistSchedule", "ClipSampler"]


@dataclass(frozen=True)
class AssistConfig:
    enabled: bool = True
    start: float = 1.0
    step: float = 0.1
    window: int = 16
    pass_frac: float = 0.5
    unassisted_every: int = 4


class AssistSchedule:
    def __init__(self, cfg: AssistConfig, n_clips: int) -> None:
        self.cfg = cfg
        self.level = np.full(n_clips, cfg.start if cfg.enabled else 0.0)
        self.block: list[list[bool]] = [[] for _ in range(n_clips)]
        self.decays = np.zeros(n_clips, np.int64)

    def world_alpha(self, world: int, clip: int) -> float:
        if not self.cfg.enabled or world % self.cfg.unassisted_every == 0:
            return 0.0
        return float(self.level[clip])

    def record(self, clip: int, alpha: float, passed: bool) -> None:
        """One finished episode of `clip` run at `alpha`; only assisted episodes count."""
        if alpha <= 0.0:
            return
        b = self.block[clip]
        b.append(bool(passed))
        if len(b) >= self.cfg.window:
            if sum(b) >= self.cfg.pass_frac * len(b):
                self.decays[clip] += 1
                self.level[clip] = max(0.0, round(self.cfg.start - self.cfg.step * self.decays[clip], 10))
            b.clear()

    def state(self) -> dict[str, Any]:
        return {"level": self.level.tolist(), "block": [list(b) for b in self.block], "decays": self.decays.tolist()}

    def load_state(self, st: dict[str, Any]) -> None:
        self.level = np.asarray(st["level"], np.float64)
        self.block = [list(b) for b in st["block"]]
        self.decays = np.asarray(st["decays"], np.int64)


class ClipSampler:
    def __init__(self, n_clips: int, eps: float = 0.05, window: int = 16) -> None:
        self.eps, self.window = float(eps), int(window)
        self.certified = np.zeros(n_clips, bool)
        self.hist: list[deque[bool]] = [deque(maxlen=self.window) for _ in range(n_clips)]

    def weights(self) -> np.ndarray:
        fail = np.array([np.mean(h) if h else 0.0 for h in self.hist])
        return self.eps + (~self.certified) * (1.0 + fail)

    def draw(self, rng: np.random.Generator) -> int:
        w = self.weights()
        return int(rng.choice(len(w), p=w / w.sum()))

    def record(self, clip: int, failed: bool) -> None:
        self.hist[clip].append(bool(failed))

    def state(self) -> dict[str, Any]:
        return {"certified": self.certified.tolist(), "hist": [list(h) for h in self.hist]}

    def load_state(self, st: dict[str, Any]) -> None:
        self.certified = np.asarray(st["certified"], bool)
        self.hist = [deque(h, maxlen=self.window) for h in st["hist"]]
