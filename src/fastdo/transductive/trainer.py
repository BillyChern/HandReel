"""Transductive PPO training loop: the PPO residual learner on `TransductiveEnv`, seeded, checkpointed and resumable.

Reused unchanged from `fastdo.policy`: `ActorCritic` (Gaussian actor, zero-mean initial residual), `collect_rollout`,
`compute_gae`, `ppo_update`, `PpoConfig`, `save_checkpoint`/`load_checkpoint`. Added here:

- determinism: one torch generator (seed) drives the action noise and the minibatch permutations; the environment
  has no randomness (DO AS I DO's executor, no perturbation), so a run is a function of (seed, configs, clips);
- resume: every `checkpoint_every` updates the model and optimizer (`checkpoints/update_XXXXXX.pt`) and the trainer
  state (`checkpoints/state_XXXXXX.npz`, no pickle: generator state, every world's exact simulator/episode state, counters, the
  last observation) are written atomically; `train(..., resume=True)` continues from the newest pair and reproduces
  the uninterrupted run exactly (CPU backend);
- logging (`log.jsonl`, one line per update): PPO diagnostics, env rows (executor rows) and policy steps in total
  and per clip, episode ends by reason, PRIMARY/STRICT training-rollout successes per clip;
- learning dynamics: observation normalization (`ObsNorm`, running statistics, saved with the state and in the
  checkpoint meta) and an adaptive learning rate on the measured KL (x/÷1.5 outside [0.5, 2] x `desired_kl`): the
  first CPU smoke on a real clip ran at approx_kl 0.2-1.1 and clip fraction 0.6-0.75 with a fixed 3e-4;
- the success store: every PRIMARY training rollout's applied controls, up to `store_per_clip` per clip in the order
  they occurred, as `successes/<task>/<k>.npz` (+ `successes/index.jsonl` with the env rows of its clip at that
  point), for certification (`fastdo.transductive.certify`).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastdo.policy.checkpoint import load_checkpoint, save_checkpoint
from fastdo.policy.networks import ActorCritic
from fastdo.policy.ppo import PpoConfig, compute_gae, ppo_update
from fastdo.policy.rollout import EpisodeTracker, collect_rollout, flatten_rollout
from fastdo.transductive.env import TransductiveEnv

__all__ = ["ObsNorm", "TrainConfig", "train"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Run settings. The run stops after `n_updates` updates or once `max_env_rows` executor rows were simulated."""

    seed: int = 0
    n_updates: int = 100
    max_env_rows: int | None = None
    checkpoint_every: int = 10
    init_log_std: float = -1.0
    store_per_clip: int = 4
    obs_norm: bool = True
    obs_norm_clip: float = 5.0
    desired_kl: float | None = 0.01
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    ppo: PpoConfig = field(default_factory=lambda: PpoConfig(n_steps=32, n_epochs=5, n_minibatches=4, gamma=0.99,
                                                             lam=0.95, lr=3e-4, hidden=(256, 256, 256)))

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ppo"]["hidden"] = list(self.ppo.hidden)
        return d


def _latest(ckpt_dir: Path) -> int | None:
    ups = sorted(int(p.stem.split("_")[1]) for p in ckpt_dir.glob("state_*.npz"))
    ups = [u for u in ups if (ckpt_dir / f"update_{u:06d}.pt").is_file()]
    return ups[-1] if ups else None


def _flat(xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    shapes = np.array([x.shape for x in xs], np.int64)
    return np.concatenate([np.asarray(x).ravel() for x in xs]), shapes


def _unflat(flat: np.ndarray, shapes: np.ndarray) -> list[np.ndarray]:
    out, o = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(flat[o:o + n].reshape(tuple(int(v) for v in sh)))
        o += n
    return out


def _save_state(path: Path, gen: torch.Generator, counters: dict[str, np.ndarray], snap: dict[str, Any],
                update: int, extra: dict[str, np.ndarray]) -> None:
    """Trainer state as a plain npz (no pickle): generator, counters and the env snapshot."""
    arrays = {"generator": gen.get_state().numpy(), "update": np.int64(update),
              **{f"counter_{k}": v for k, v in counters.items()}, **extra}
    if snap:                                  # the CPU path's exact world states (the GPU path restarts episodes)
        sim, sim_sh = _flat(snap["sim"])
        buf, buf_sh = _flat(snap["buf"])
        arrays.update({"env_sim": sim, "env_sim_shapes": sim_sh, "env_buf": buf, "env_buf_shapes": buf_sh,
                       "env_delta": snap["delta"], "env_sums": snap["sums"], "env_raw": snap["raw"],
                       "env_contact": np.array(snap["contact"], np.float64), "env_alpha": snap["alpha"],
                       "env_next_alpha": snap["next_alpha"], "env_start": snap["start"],
                       "env_episode": snap["episode"]})
    b = io.BytesIO()
    np.savez(b, **arrays)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(b.getvalue())
    os.replace(tmp, path)


def _load_state(path: Path) -> tuple[torch.Tensor, dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if "env_sim" not in a:
        snap: dict[str, Any] = {}
        counters = {k[len("counter_"):]: v for k, v in a.items() if k.startswith("counter_")}
        return torch.from_numpy(a["generator"].copy()), counters, snap, a
    snap = {"sim": _unflat(a["env_sim"], a["env_sim_shapes"]),
            "buf": [x.astype(np.float32) for x in _unflat(a["env_buf"], a["env_buf_shapes"])],
            "delta": a["env_delta"], "sums": a["env_sums"], "raw": a["env_raw"], "alpha": a["env_alpha"],
            "next_alpha": a["env_next_alpha"], "start": a["env_start"], "episode": a["env_episode"],
            "contact": [(float(c[0]), bool(c[1]), bool(c[2])) for c in a["env_contact"]]}
    counters = {k[len("counter_"):]: v for k, v in a.items() if k.startswith("counter_")}
    return torch.from_numpy(a["generator"].copy()), counters, snap, a


class ObsNorm:
    """Running mean/variance of the observation (parallel Welford over each batch), deterministic; the normalized
    observation is (o - mean) / sqrt(var + 1e-8), clipped to +-clip. Updated on every batch the learner sees during
    training (statistics include the batch they normalize), frozen when evaluating."""

    def __init__(self, dim: int, device: torch.device, clip: float) -> None:
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(dim, dtype=torch.float64, device=device)
        self.clip = clip

    def update(self, x: torch.Tensor) -> None:
        x = x.to(torch.float64)
        n = float(x.shape[0])
        bm = x.mean(0)
        bm2 = ((x - bm) ** 2).sum(0)
        tot = self.count + n
        delta = bm - self.mean
        self.mean = self.mean + delta * (n / tot)
        self.m2 = self.m2 + bm2 + delta * delta * (self.count * n / tot)
        self.count = tot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        var = self.m2 / torch.clamp(self.count, min=1.0)
        y = (x.to(torch.float64) - self.mean) / torch.sqrt(var + 1e-8)
        return torch.clamp(y, -self.clip, self.clip).to(x.dtype)

    def state(self) -> dict[str, np.ndarray]:
        return {"norm_count": self.count.cpu().numpy(), "norm_mean": self.mean.cpu().numpy(),
                "norm_m2": self.m2.cpu().numpy()}

    def load(self, a: dict[str, np.ndarray]) -> None:
        dev = self.mean.device
        self.count = torch.as_tensor(a["norm_count"], dtype=torch.float64, device=dev)
        self.mean = torch.as_tensor(a["norm_mean"], dtype=torch.float64, device=dev)
        self.m2 = torch.as_tensor(a["norm_m2"], dtype=torch.float64, device=dev)


class _NormEnv:
    """The learner's view of the env: observations normalized by `ObsNorm` (updated as they arrive)."""

    def __init__(self, env: TransductiveEnv, norm: ObsNorm | None) -> None:
        self.env, self.norm = env, norm
        self.obs_dim, self.priv_dim, self.act_dim, self.n_worlds = env.obs_dim, env.priv_dim, env.act_dim, env.n_worlds

    def privileged(self) -> torch.Tensor:
        return self.env.privileged()

    def _n(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.norm is None:
            return obs
        if update:
            self.norm.update(obs)
        return self.norm(obs)

    def reset(self, ref: Any = None, world_ids: Any = None) -> torch.Tensor:
        return self._n(self.env.reset(ref, world_ids))

    def step(self, action: torch.Tensor):
        obs, rew, done, info = self.env.step(action)
        return self._n(obs), rew, done, info


class _Store:
    """The success store (module docstring)."""

    def __init__(self, root: Path, cap: int, upto: int) -> None:
        """Open the store of a run resumed at update `upto` (0 = fresh): entries written after that update (the
        updates a resume replays) are dropped, so a resumed run stores exactly what an uninterrupted one does."""
        self.root, self.cap = root, cap
        root.mkdir(parents=True, exist_ok=True)
        self.count: dict[str, int] = {}
        idx = root / "index.jsonl"
        keep = []
        if idx.is_file():
            for line in idx.read_text().splitlines():
                if line.strip():
                    m = json.loads(line)
                    if m["update"] <= upto:
                        keep.append(line)
                        self.count[m["task"]] = self.count.get(m["task"], 0) + 1
                    else:
                        (root / m["task"] / f"{m['k']:04d}.npz").unlink(missing_ok=True)
            idx.write_text("".join(x + "\n" for x in keep))

    def add(self, s: dict[str, Any], update: int) -> bool:
        t = s["task"]
        k = self.count.get(t, 0)
        if k >= self.cap:
            return False
        d = self.root / t
        d.mkdir(parents=True, exist_ok=True)
        meta = {k2: v for k2, v in s.items() if k2 != "ctrl"} | {"update": update, "k": k}
        np.savez(d / f"{k:04d}.npz", ctrl=np.asarray(s["ctrl"], np.float32),
                 meta_json=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        with (self.root / "index.jsonl").open("a") as f:
            f.write(json.dumps(meta) + "\n")
        self.count[t] = k + 1
        return True


def _checkpoint(ck: Path, update: int, model: ActorCritic, opt: torch.optim.Optimizer, norm: ObsNorm | None,
                gen: torch.Generator, env: Any, cfg: TrainConfig) -> None:
    c = env.counters()
    meta = {"update": update, "env_rows": int(c["rows_per_clip"].sum()), "seed": cfg.seed}
    if norm is not None:
        meta["obs_norm"] = {k: v.tolist() for k, v in norm.state().items()} | {"clip": norm.clip}
    save_checkpoint(ck / f"update_{update:06d}.pt", model, opt, meta)
    _save_state(ck / f"state_{update:06d}.npz", gen, c, env.snapshot(), update, norm.state() if norm is not None else {})


def train(env: TransductiveEnv | Any, out_dir: Path, cfg: TrainConfig, resume: bool = True,
          env_cfg: dict[str, Any] | None = None) -> ActorCritic:
    """Train (module docstring); returns the model, with its observation normalizer as `model.obs_norm`. The
    learner runs on `env.device`."""
    out_dir = Path(out_dir)
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    dev = env.device
    p = cfg.ppo
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=p.hidden, init_log_std=cfg.init_log_std).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=p.lr)
    norm = ObsNorm(env.obs_dim, dev, cfg.obs_norm_clip) if cfg.obs_norm else None
    lenv = _NormEnv(env, norm)
    start = 0
    last = _latest(ck) if resume else None
    if last is not None:
        loaded = load_checkpoint(ck / f"update_{last:06d}.pt", dev)
        model.load_state_dict(loaded.model.state_dict())
        opt.load_state_dict(loaded.optimizer_state)
        gstate, counters, snap, arrays = _load_state(ck / f"state_{last:06d}.npz")
        gen.set_state(gstate)
        env.set_counters(counters)
        if norm is not None:
            norm.load(arrays)
        obs = lenv._n(env.restore(snap).to(dev), update=False)   # already counted when it was first seen
        start = last
        logger.info("resumed from update %d (%d env rows)", last, int(env.rows_per_clip.sum()))
    else:
        (out_dir / "config.json").write_text(json.dumps({"train": cfg.as_dict(), "env": env_cfg or {},
                                                         "tasks": env.tasks, "n_worlds": env.n_worlds,
                                                         "obs_dim": env.obs_dim, "act_dim": env.act_dim}, indent=1))
        obs = lenv.reset()
    store = _Store(out_dir / "successes", cfg.store_per_clip, start)
    logp = out_dir / "log.jsonl"
    if logp.is_file():                        # drop the log lines of updates the resume replays
        lines = [x for x in logp.read_text().splitlines() if x.strip() and json.loads(x)["update"] <= start]
        logp.write_text("".join(x + "\n" for x in lines))
    tracker = EpisodeTracker(env.n_worlds, dev)
    log = logp.open("a")
    last_saved = start
    for update in range(start, cfg.n_updates):
        if cfg.max_env_rows is not None and int(env.rows_per_clip.sum()) >= cfg.max_env_rows:
            if last_saved < update:              # the budget is reached: checkpoint where the run stops
                _checkpoint(ck, update, model, opt, norm, gen, env, cfg)
            break
        t0 = time.perf_counter()
        ro, obs = collect_rollout(lenv, model, obs, p.n_steps, p.gamma, tracker, gen)
        t1 = time.perf_counter()
        adv, ret = compute_gae(ro.rewards, ro.values, ro.dones, ro.last_value, p.gamma, p.lam)
        stats = ppo_update(model, opt, flatten_rollout(ro, adv, ret), p, gen)
        t2 = time.perf_counter()
        lr = opt.param_groups[0]["lr"]
        if cfg.desired_kl is not None:                 # adaptive step size on the measured KL (rsl_rl's rule)
            if stats["approx_kl"] > 2.0 * cfg.desired_kl:
                lr = max(lr / 1.5, cfg.lr_min)
            elif stats["approx_kl"] < 0.5 * cfg.desired_kl:
                lr = min(lr * 1.5, cfg.lr_max)
            for g in opt.param_groups:
                g["lr"] = lr
        stored = sum(store.add(s, update + 1) for s in env.pop_successes())
        ep_r, ep_l = tracker.pop()
        c = env.counters()
        line = {"update": update + 1, "env_rows": int(c["rows_per_clip"].sum()),
                "policy_steps": int(c["steps_per_clip"].sum()), "mean_step_reward": tracker.pop_mean_step_reward(),
                "episode_return": float(ep_r.mean()) if ep_r.numel() else math.nan,
                "episode_length": float(ep_l.double().mean()) if ep_l.numel() else math.nan,
                "n_episodes": int(ep_r.numel()), "action_std": float(model.action_std().detach().mean()),
                "collect_s": t1 - t0, "update_s": t2 - t1, "stored": int(stored), "lr": lr, **stats,
                "rows_per_clip": c["rows_per_clip"].tolist(), "primary_per_clip": c["primary_per_clip"].tolist(),
                "strict_per_clip": c["strict_per_clip"].tolist(), "ends": c["ends"].sum(0).tolist(),
                "assist_alpha": c["assist_alpha"].tolist()}
        log.write(json.dumps(line, default=float) + "\n")
        log.flush()
        if (update + 1) % cfg.checkpoint_every == 0 or update + 1 == cfg.n_updates:
            _checkpoint(ck, update + 1, model, opt, norm, gen, env, cfg)
            last_saved = update + 1
    log.close()
    model.obs_norm = norm            # the normalizer the policy was trained with (None when disabled)
    return model
