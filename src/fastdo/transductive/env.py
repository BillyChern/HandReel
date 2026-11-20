"""The transductive residual environment over many clips: a `fastdo.policy.protocol.TrackingEnv`.

Worlds are laid out clip by clip (`worlds_per_clip` each, global world w belongs to clip w // worlds_per_clip) and
split into contiguous chunks, one `WorldGroup` per worker process (`n_workers = 0`: one group in this process). A
world's trajectory depends only on its own actions (the backend's worlds are independent), so results do not depend
on `n_workers`. The policy's PPO learner (`fastdo.policy.ppo`, `fastdo.policy.rollout`) drives it unchanged: actions
are unsquashed Gaussian samples, `step` auto-resets, and there are no truncations (the clip's end is terminal), so
`info` carries no time-outs.

Bookkeeping in the learner process: env rows (executor rows, 5 ms each) and policy steps per clip, episode ends by
reason, and the PRIMARY successes the groups hand back, each stamped with its clip's cumulative env rows at the
step that completed it (`pop_successes`).
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.transductive.clip import ClipSpec, load_clip
from fastdo.transductive.config import EnvConfig
from fastdo.transductive.group import END_CODES, WorldGroup
from fastdo.transductive.obs import obs_dim

__all__ = ["EnvBook", "TransductiveEnv"]


def _build_group(cfg: EnvConfig, source: Any, tasks: Sequence[str], world_clip: Sequence[int],
                 world_ids: Sequence[int]) -> WorldGroup:
    from fastdo.transductive.sim_cpu import CpuWorlds

    used = sorted({int(c) for c in world_clip})
    specs = [source[tasks[c]] if isinstance(source, dict) else load_clip(Path(source), tasks[c]) for c in used]
    local = {c: j for j, c in enumerate(used)}
    sim = CpuWorlds(specs, [local[int(c)] for c in world_clip], assist_wn=cfg.assist_wn,
                    assist_track=cfg.assist_track)
    return WorldGroup(cfg, sim, used, len(tasks), world_ids)


def _worker(conn: Any, cfg_dict: dict, source: Any, tasks: list[str], world_clip: list[int],
            world_ids: list[int]) -> None:
    group = _build_group(EnvConfig.from_dict(cfg_dict), source, tasks, world_clip, world_ids)
    conn.send(("ready", None))
    while True:
        cmd, arg = conn.recv()
        try:
            if cmd == "reset":
                out: Any = group.reset_all(arg)
            elif cmd == "step":
                out = group.step(*arg)
            elif cmd == "snapshot":
                out = group.snapshot()
            elif cmd == "restore":
                out = group.restore(arg)
            elif cmd == "close":
                conn.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command {cmd}")
            conn.send(("ok", out))
        except Exception as exc:  # noqa: BLE001 - handed to the learner, which raises it
            import traceback

            conn.send(("error", f"{exc!r}\n{traceback.format_exc()}"))


class EnvBook:
    """Learner-side bookkeeping shared by the CPU (`TransductiveEnv`) and GPU (`fastdo.transductive.batch_env`)
    frontends: per-clip env rows / policy steps / episode ends / training successes, the per-clip assist curriculum,
    and the success hand-off (each stamped with its clip's cumulative env rows)."""

    def _init_book(self, cfg: EnvConfig, tasks: Sequence[str], world_clip: np.ndarray, worlds_per_clip: int,
                   device: str | torch.device) -> None:
        self.cfg, self.tasks = cfg, list(tasks)
        self.device = torch.device(device)
        self.n_clips = len(self.tasks)
        self.world_clip = world_clip
        n = world_clip.shape[0]
        self.rows_per_clip = np.zeros(self.n_clips, np.int64)
        self.steps_per_clip = np.zeros(self.n_clips, np.int64)
        self.ends = np.zeros((self.n_clips, len(END_CODES)), np.int64)
        self.primary_per_clip = np.zeros(self.n_clips, np.int64)
        self.strict_per_clip = np.zeros(self.n_clips, np.int64)
        self.assist_alpha = np.full(self.n_clips, cfg.assist_start if cfg.assist else 0.0)
        self.assist_n = np.zeros(self.n_clips, np.int64)
        self.assist_pass = np.zeros(self.n_clips, np.int64)
        self._free = (np.arange(n) % worlds_per_clip) % cfg.assist_free_every == 0
        self._successes: list[dict[str, Any]] = []

    @property
    def n_worlds(self) -> int:
        return int(self.world_clip.shape[0])

    def next_alpha(self) -> np.ndarray:
        """Assist level of each world's next episode: its clip's current level, 0 on the unassisted worlds."""
        return np.where(self._free, 0.0, self.assist_alpha[self.world_clip])

    def _update_assist(self, info: dict[str, np.ndarray]) -> None:
        """The curriculum (`EnvConfig` docstring): per clip, promote after a window of assisted episodes at the
        current level whose PRIMARY rate (under the assist) reaches `assist_promote`."""
        cfg = self.cfg
        ended = (info["end"] > 0) & (info["alpha"] > 0.0)
        for w in np.flatnonzero(ended):
            c = int(info["clip"][w])
            if info["alpha"][w] != self.assist_alpha[c]:
                continue                                  # an episode started before the last promotion
            self.assist_n[c] += 1
            self.assist_pass[c] += int(info["assisted_primary"][w])
            if self.assist_n[c] >= cfg.assist_window:
                if self.assist_pass[c] >= cfg.assist_promote * self.assist_n[c]:
                    self.assist_alpha[c] = max(0.0, round(self.assist_alpha[c] - cfg.assist_step, 6))
                self.assist_n[c] = self.assist_pass[c] = 0

    def _account(self, info: dict[str, np.ndarray], successes: list[dict[str, Any]]) -> None:
        np.add.at(self.rows_per_clip, info["clip"], info["rows"])
        np.add.at(self.steps_per_clip, info["clip"], 1)
        ended = info["end"] > 0
        np.add.at(self.ends, (info["clip"][ended], info["end"][ended]), 1)
        np.add.at(self.primary_per_clip, info["clip"], info["primary"].astype(np.int64))
        np.add.at(self.strict_per_clip, info["clip"], info["strict"].astype(np.int64))
        if self.cfg.assist:
            self._update_assist(info)
        for s in successes:
            c = int(s["clip"])
            s["task"] = self.tasks[c]
            s["env_rows_clip"] = int(self.rows_per_clip[c])
            s["env_rows_total"] = int(self.rows_per_clip.sum())
            self._successes.append(s)

    def pop_successes(self) -> list[dict[str, Any]]:
        out, self._successes = self._successes, []
        return out

    def counters(self) -> dict[str, np.ndarray]:
        return {"rows_per_clip": self.rows_per_clip.copy(), "steps_per_clip": self.steps_per_clip.copy(),
                "ends": self.ends.copy(), "primary_per_clip": self.primary_per_clip.copy(),
                "strict_per_clip": self.strict_per_clip.copy(), "assist_alpha": self.assist_alpha.copy(),
                "assist_n": self.assist_n.copy(), "assist_pass": self.assist_pass.copy()}

    def set_counters(self, c: dict[str, np.ndarray]) -> None:
        for k, v in c.items():
            setattr(self, k, np.asarray(v, np.float64 if k == "assist_alpha" else np.int64).copy())


class TransductiveEnv(EnvBook):
    """`TrackingEnv` over `tasks` (module docstring). `source` is a dict task -> ClipSpec or a clip cache root."""

    def __init__(self, cfg: EnvConfig, source: dict[str, ClipSpec] | Path | str, tasks: Sequence[str],
                 worlds_per_clip: int, n_workers: int = 0, device: str | torch.device = "cpu") -> None:
        self._init_book(cfg, tasks, np.repeat(np.arange(len(tasks)), worlds_per_clip), worlds_per_clip, device)
        n = self.world_clip.shape[0]
        first = source[self.tasks[0]] if isinstance(source, dict) else load_clip(Path(source), self.tasks[0])
        self._obs_dim = obs_dim(cfg, first.n_robot, self.n_clips)
        self._act_dim = first.n_robot
        self._procs: list[Any] = []
        self._conns: list[Any] = []
        self._local: WorldGroup | None = None
        chunks = [c for c in np.array_split(np.arange(n), max(n_workers, 1)) if c.size]
        self._chunks = chunks
        if n_workers <= 0:
            self._local = _build_group(cfg, source, self.tasks, self.world_clip.tolist(), list(range(n)))
        else:
            ctx = mp.get_context("spawn")
            src = str(source) if not isinstance(source, dict) else source
            for c in chunks:
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, cfg.as_dict(), src, self.tasks,
                                                      self.world_clip[c].tolist(), c.tolist()), daemon=True)
                p.start()
                self._procs.append(p)
                self._conns.append(parent)
            for conn in self._conns:
                self._recv(conn)

    # -- TrackingEnv protocol
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

    def _recv(self, conn: Any) -> Any:
        status, out = conn.recv()
        if status == "error":
            raise RuntimeError(f"worker failed: {out}")
        return out

    def _call(self, cmd: str, args: Sequence[Any] | None = None) -> list[Any]:
        if self._local is not None:
            fn = {"reset": self._local.reset_all, "step": lambda a: self._local.step(*a),
                  "snapshot": lambda a: self._local.snapshot(), "restore": self._local.restore}[cmd]
            return [fn(args[0] if args else None)]
        for j, conn in enumerate(self._conns):
            conn.send((cmd, None if args is None else args[j]))
        return [self._recv(conn) for conn in self._conns]

    def reset(self, ref: Any = None, world_ids: Tensor | None = None) -> Tensor:
        if ref is not None or world_ids is not None:
            raise NotImplementedError("the transductive env resets all worlds onto their own clips")
        alpha = self.next_alpha()
        obs = np.concatenate(self._call("reset", [alpha[c] for c in self._chunks]))
        return torch.as_tensor(obs, device=self.device)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        a = action.detach().to("cpu", torch.float64).numpy()
        alpha = self.next_alpha()
        parts = self._call("step", [(a[c], alpha[c]) for c in self._chunks])
        obs = np.concatenate([p[0] for p in parts])
        rew = np.concatenate([p[1] for p in parts])
        done = np.concatenate([p[2] for p in parts])
        info = {k: np.concatenate([p[3][k] for p in parts]) for k in parts[0][3]}
        for c, p in zip(self._chunks, parts, strict=True):
            for s in p[4]:
                s["world"] = int(c[s["world"]])
        self._account(info, [s for p in parts for s in p[4]])
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device)
        return t(obs, torch.float32), t(rew, torch.float32), t(done, torch.bool), info

    # -- checkpoint support
    def snapshot(self) -> dict[str, Any]:
        """Exact state of every world, in global world order (independent of the worker layout)."""
        parts = self._call("snapshot")
        return {k: (np.concatenate([p[k] for p in parts]) if isinstance(parts[0][k], np.ndarray)
                    else [x for p in parts for x in p[k]]) for k in parts[0]}

    def restore(self, snap: dict[str, Any]) -> Tensor:
        chunks = [np.arange(self.n_worlds)] if self._local is not None else self._chunks
        args = [{k: (v[c] if isinstance(v, np.ndarray) else [v[int(i)] for i in c]) for k, v in snap.items()}
                for c in chunks]
        obs = np.concatenate(self._call("restore", args))
        return torch.as_tensor(obs, device=self.device)

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
                conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=10)
        self._procs, self._conns = [], []
