"""Tracker training loop on the CPU executor: acting, replay, learner, assist, clip sampler, success store,
inline certification, checkpoint and resume.

One vectorised step: the behaviour actor (the learner's actor copied every `act_sync_every` steps, a content-addressed
version, `fastdo.amortised.acting`) acts batch-invariantly with each world's counter-based episode noise; the
environment (`TrackerEnv`, DO AS I DO's executor) executes k rows per world; the transition is stored in the
host replay with the PRE-step record; then `UtdCounter` grants u x worlds gradient updates (u pinned per TRANSITION,
FlashSAC's 2/2048), once `learning_starts` steps are sampleable. A world whose episode ended is restarted at once by
the reset hook: the clip sampler draws its next clip, the assist schedule its alpha, RSI (probability `rsi_prob`)
its start row. Every episode that ran the clip to its end is scored by the clip's `scorer` (the registered scorer on
real clips); a passing alpha = 0, row-0 episode is offered to the `SuccessStore` and, with `certify_inline`,
certified closed loop on the CPU at once (a certified clip drops to the sampler's floor weight).

Determinism and resume: everything random is seeded (numpy for the hook and replay sampling, torch's global generator
for the learner's samples, counter-based acting noise), MuJoCo on CPU is deterministic, acting is batch-of-one and
the process runs single-threaded torch (`threads`). A checkpoint (every `checkpoint_every` steps, atomic directory
rename, the last two kept) holds the learner, behaviour actor, replay, every world's exact state and current record,
the episodes in flight (their rows and versions), schedules, counters, generator states and the episode log length;
`Trainer.resume` continues from the newest one and reproduces the uninterrupted run bit for bit (the log is cut back
to the checkpoint, the store and certificates are idempotent). The offer outcome in the log and the stats comes from
the trainer's own checkpointed state (`stored_keys`), not from the store's files: a candidate stored after the
checkpoint that was resumed from is found again on disk with identical bytes ("exists") and is reported "stored", as
the uninterrupted run reports it.

Prefix trace (`fastdo.amortised.prefix_trace`): with a `PrefixTrace` attached (`attach_trace`), every inline
certificate is written as a certificate row (the trace's run-level source, its clip, certified or not) and every
`trace.every` vectorised steps a budget checkpoint row. A certificate's row is written
AFTER the step that produced it has counted its rows (the episode's last rows are part of the budget it cost), so the
trace is a true prefix: a clip counts at budget c only if it was certified by the rows executed up to c. Nothing else
changes: without a trace the loop is unchanged, and the trace never feeds back into training.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastdo.amortised.acting import PolicyVersions, act, episode_seed, step_noise
from fastdo.amortised.replay import HostReplay
from fastdo.amortised.sac import SAC, SacConfig, UtdCounter
from fastdo.amortised.schedule import AssistConfig, AssistSchedule, ClipSampler
from fastdo.amortised.success import Candidate, SuccessStore, certify, trajectory_sha
from fastdo.amortised.tracker_env import TipSpec, TrackerEnv
from fastdo.amortised.tracker_obs import RefTable, TrackerConfig, obs_dims, observe
from fastdo.transductive.clip import ClipSpec

__all__ = ["Episode", "TrainInputs", "Trainer", "TrainerConfig"]


@dataclass(frozen=True)
class TrainerConfig:
    seed: int = 0
    n_worlds: int = 4
    max_steps: int = 1000                 # vectorised steps
    learning_starts: int = 16             # sampleable vectorised steps before the first update
    replay_capacity: int = 20_000         # vectorised steps
    final_capacity: int | None = None
    act_sync_every: int = 25
    checkpoint_every: int = 0             # vectorised steps; 0 = never
    rsi_prob: float = 0.0
    sampler_eps: float = 0.05
    sampler_window: int = 16
    certify_inline: bool = True
    threads: int | None = 1               # torch threads of the process (None: leave unchanged)
    assist: AssistConfig = field(default_factory=AssistConfig)
    sac: SacConfig = field(default_factory=SacConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(eq=False)
class TrainInputs:
    """Clips in table order: specs, Sharpa/toy tip specs, object SDFs, `clip_table_entry` dicts, task names and one
    scorer per clip (executed qpos rows -> dict with "success")."""

    specs: Sequence[ClipSpec]
    tips: Sequence[TipSpec]
    sdfs: Sequence[Any]
    entries: Sequence[dict[str, Any]]
    tasks: Sequence[str]
    scorers: Sequence[Callable[[np.ndarray], dict[str, Any]]]


@dataclass
class Episode:
    clip: int
    episode: int
    seed: int
    start_row: int
    alpha: float
    versions: list[str] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    ctrl: list[np.ndarray] = field(default_factory=list)

    def head(self) -> dict[str, Any]:
        return {"clip": self.clip, "episode": self.episode, "seed": self.seed, "start_row": self.start_row,
                "alpha": self.alpha, "versions": list(self.versions)}


def _cat(xs: list[np.ndarray], width: int) -> np.ndarray:
    return np.concatenate(xs) if xs else np.zeros((0, width))


class Trainer:
    def __init__(self, cfg: TrainerConfig, inp: TrainInputs, out_dir: Path) -> None:
        if cfg.threads is not None:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.cfg, self.inp, self.out = cfg, inp, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.table = RefTable.stack(list(inp.entries))
        R, F, P = self.table.R, self.table.F, self.table.P
        a_dim, c_dim = obs_dims(cfg.tracker, R, F, P)
        self.agent = SAC(a_dim, c_dim, R, cfg.sac, device=self._agent_device())
        self.behaviour = copy.deepcopy(self.agent.actor)
        self.versions = PolicyVersions()
        self.store = SuccessStore(self.out / "successes")
        self.rng = np.random.default_rng(cfg.seed)
        self.assist = AssistSchedule(cfg.assist, len(inp.tasks))
        self.sampler = ClipSampler(len(inp.tasks), cfg.sampler_eps, cfg.sampler_window)
        self.utd = UtdCounter(cfg.sac.updates_per_transition, cfg.n_worlds)
        self.env = self._make_env()
        self.replay = HostReplay(self.env.fields, cfg.n_worlds, cfg.replay_capacity, R, cfg.sac.n_steps,
                                 cfg.sac.gamma, cfg.final_capacity)
        self.step = 0
        self.n_episodes = 0
        self.ver: str | None = None
        self.eps: list[Episode | None] = [None] * cfg.n_worlds
        self._pending: dict[int, Episode] = {}
        self.stored_keys: set[tuple[str, int]] = set()      # candidates THIS run has stored (checkpointed)
        self.stats: dict[str, Any] = {"rows": 0, "updates": 0, "ends": {}, "offers": {}, "certified": 0,
                                      "certificates": 0, "alpha0_full": 0, "alpha0_pass": 0, "assisted_full": 0,
                                      "assisted_pass": 0}
        self.log_path = self.out / "log.jsonl"
        self.log_lines = 0
        self.rec: dict[str, np.ndarray] | None = None
        self.on_episode: Callable[[Episode, str, dict[str, Any] | None], None] | None = None
        self.trace: Any = None                               # a `prefix_trace.PrefixTrace` (`attach_trace`)
        self._trace_pending: list[tuple[int, bool, int, str]] = []
        self._clip_rows: list[int] = [0] * len(inp.tasks)

    # ------------------------------------------------------------------------------------------ backend seams
    def _agent_device(self) -> str:
        """The learner's device (the CPU path; `trainer_dev.DeviceTrainer` overrides)."""
        return "cpu"

    def _make_env(self) -> TrackerEnv:
        """The environment (the CPU executor; `trainer_dev.DeviceTrainer` overrides)."""
        cfg, inp = self.cfg, self.inp
        return TrackerEnv(cfg.tracker, inp.specs, self.table, inp.tips, inp.sdfs, [0] * cfg.n_worlds,
                          reset_hook=self._hook)

    def _act(self) -> np.ndarray:
        """This step's actions (n_worlds, R): batch-of-one acting with each episode's counter-based noise."""
        obs, _ = observe(self.table, self.rec, self.cfg.tracker)
        noise = torch.stack([step_noise(e.seed, len(e.versions), self.table.R) for e in self.eps])
        return act(self.behaviour, obs, noise)

    def _observe_batch(self, rec: dict[str, np.ndarray]) -> tuple[Any, Any]:
        """Observations of sampled records (the replay rebuild)."""
        return observe(self.table, rec, self.cfg.tracker)

    def _batch(self, b: dict[str, Any]) -> dict[str, Any]:
        """One sampled n-step batch as the learner's update input."""
        return {"actor_obs": torch.as_tensor(b["obs"][0]), "critic_obs": torch.as_tensor(b["obs"][1]),
                "next_actor_obs": torch.as_tensor(b["next_obs"][0]),
                "next_critic_obs": torch.as_tensor(b["next_obs"][1]),
                "action": torch.as_tensor(b["action"]), "reward": b["reward"], "terminated": b["terminated"]}

    def _candidate_noise(self) -> str:
        return "torch"

    # ------------------------------------------------------------------------------------------ episodes
    def _hook(self, i: int) -> tuple[int, float, int]:
        clip = self.sampler.draw(self.rng)
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

    def _log(self, row: dict[str, Any]) -> None:
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=float) + "\n")
        self.log_lines += 1

    def _finish(self, i: int, ep: Episode, reason: str) -> None:
        sp = self.inp.specs[ep.clip]
        qpos, ctrl = _cat(ep.qpos, sp.ex.model.nq), _cat(ep.ctrl, sp.ex.model.nu)
        st = self.stats
        st["ends"][reason] = st["ends"].get(reason, 0) + 1
        score = self.inp.scorers[ep.clip](qpos) if reason == "clip_end" else None
        passed = bool(score is not None and score.get("success"))
        if reason == "clip_end":
            key = "alpha0" if ep.alpha <= 0.0 else "assisted"
            st[f"{key}_full"] += 1
            st[f"{key}_pass"] += int(passed)
        self.assist.record(ep.clip, ep.alpha, passed)
        self.sampler.record(ep.clip, not passed)
        offer = None
        if passed:
            cand = Candidate(task=self.inp.tasks[ep.clip], clip=ep.clip, episode=ep.episode, noise_seed=ep.seed,
                             start_row=ep.start_row, alpha=ep.alpha, versions=tuple(ep.versions),
                             rows=int(qpos.shape[0]), traj_sha=trajectory_sha(qpos, ctrl),
                             score={k: v for k, v in score.items() if isinstance(v, (bool, int, float, str))},
                             noise=self._candidate_noise())
            stored, offer = self.store.offer(cand, self.versions)
            key = (cand.task, cand.episode)
            if offer == "exists" and key not in self.stored_keys:
                offer = "stored"            # identical bytes left by this run's own future (resume from an older ckpt)
            if offer == "stored":
                self.stored_keys.add(key)
            st["offers"][offer] = st["offers"].get(offer, 0) + 1
            if stored and self.cfg.certify_inline:
                self._certify(cand)
        row = {"step": self.step, "world": i, "clip": ep.clip, "episode": ep.episode, "seed": ep.seed,
               "start_row": ep.start_row, "alpha": ep.alpha, "task": self.inp.tasks[ep.clip], "reason": reason,
               "rows": int(qpos.shape[0]), "success": passed, "offer": offer, "versions": len(set(ep.versions))}
        self._log(row)
        if self.on_episode is not None:
            self.on_episode(ep, reason, score)
        self.versions.release(ep.versions)

    def _certify(self, cand: Candidate) -> None:
        c, inp = cand.clip, self.inp
        cert = certify(cand, inp.specs[c], inp.tips[c], inp.sdfs[c], inp.entries[c], self.cfg.tracker,
                       copy.deepcopy(self.agent.actor).cpu(), self.store.versions(), inp.scorers[c])
        d = self.out / "certificates" / cand.task
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{cand.episode:09d}.npz", qpos=cert.qpos, ctrl=cert.ctrl)
        summ = cert.summary() | {"equals_training_rollout": cert.sha == cand.traj_sha}
        (d / f"{cand.episode:09d}.json").write_text(json.dumps(summ, sort_keys=True, default=float))
        self.stats["certificates"] += 1
        if cert.certified:
            self.stats["certified"] += 1
            self.sampler.certified[c] = True
        if self.trace is not None:                   # written after this step's rows are counted (`_trace_step`)
            self._trace_pending.append((c, bool(cert.certified), int(cand.episode), cert.sha))

    # ------------------------------------------------------------------------------------------ prefix trace
    def attach_trace(self, trace: Any) -> None:
        """Write this run's certificates and budget checkpoints to `trace` (module docstring)."""
        if [r.task for r in trace.refs] != list(self.inp.tasks):
            raise ValueError("the trace's references must be this run's clips, in table order")
        self.trace = trace

    def _trace_step(self, rows: np.ndarray, clips: list[int]) -> None:
        for i, c in enumerate(clips):
            self._clip_rows[c] += int(rows[i])
        total = int(self.stats["rows"])
        for c, ok, ep, sha in self._trace_pending:
            self.trace.certificate(c, rows=total, certified=ok, step=self.step, episode=ep,
                                   clip_rows=self._clip_rows[c], sha=sha)
        self._trace_pending.clear()
        if self.step % self.trace.every == 0:
            self.trace.checkpoint(rows=total, step=self.step)

    def close_trace(self) -> None:
        """The trace's final checkpoint (its last point) at the run's current budget."""
        if self.trace is not None:
            self.trace.close(rows=int(self.stats["rows"]), step=self.step)

    def _adopt_pending(self, worlds: Sequence[int]) -> None:
        for i in worlds:
            self.eps[i] = self._pending.pop(i)

    # ------------------------------------------------------------------------------------------ loop
    def start(self) -> None:
        self.rec = self.env.reset_all()
        self._adopt_pending(range(self.cfg.n_worlds))

    def _sync_behaviour(self) -> None:
        if self.ver is None or self.step % self.cfg.act_sync_every == 0:
            self.behaviour.load_state_dict(self.agent.actor.state_dict())
            self.ver = self.versions.snapshot(self.behaviour)

    def one_step(self) -> None:
        self._sync_behaviour()
        a = self._act()
        for e in self.eps:
            e.versions.append(self.ver)
        self.versions.retain([self.ver] * len(self.eps))
        out = self.env.step(a)
        r, te, tr = (torch.as_tensor(x) for x in (out.reward, out.term, out.trunc))
        if self.agent.normalizer is not None:
            self.agent.normalizer.update(r, te, tr)
        self.replay.add(self.rec, a, out.reward, out.term, out.trunc, final=out.final)
        clips = [e.clip for e in self.eps] if self.trace is not None else []
        ended = []
        for i, e in enumerate(self.eps):
            e.qpos.append(out.qpos_rows[i])
            e.ctrl.append(out.ctrl_rows[i])
            if out.end[i] is not None:
                self._finish(i, e, out.end[i])
                ended.append(i)
        self._adopt_pending(ended)
        self.rec = out.record
        self.stats["rows"] += int(out.rows.sum())
        self.step += 1
        if self.trace is not None:
            self._trace_step(out.rows, clips)
        self._learn()
        self.versions.gc(keep=[self.ver])

    def _learn(self) -> None:
        k = self.utd.step()
        lo, hi = self.replay.sample_range()
        if hi - lo < self.cfg.learning_starts:
            return
        for _ in range(k):
            b = self.replay.sample(self.rng, self.cfg.sac.batch_size, observe=self._observe_batch)
            self.agent.update(self._batch(b))
            self.stats["updates"] += 1

    def run(self, stop_after: int | None = None) -> dict[str, Any]:
        """Run to `max_steps` (or `stop_after` more steps), checkpointing on schedule; returns the summary."""
        if self.rec is None:
            self.start()
        t0 = time.perf_counter()
        target = self.cfg.max_steps if stop_after is None else min(self.cfg.max_steps, self.step + stop_after)
        while self.step < target:
            self.one_step()
            if self.cfg.checkpoint_every and self.step % self.cfg.checkpoint_every == 0:
                self.save_checkpoint()
        return self.summary() | {"wall_s": time.perf_counter() - t0}

    def summary(self) -> dict[str, Any]:
        return {"step": self.step, "episodes": self.n_episodes, **self.stats,
                "assist_level": self.assist.level.tolist(), "sampler_weights": self.sampler.weights().tolist(),
                "certified_clips": [t for t, c in zip(self.inp.tasks, self.sampler.certified, strict=True) if c]}

    def digest(self) -> str:
        """sha256 over the learner, the replay, the store and the log (the resume known answer)."""
        h = hashlib.sha256()
        for k, v in self.agent.state_dict().items():
            h.update(k.encode())
            h.update(pickle.dumps(_canon(v)))
        for k, a in sorted(self.replay.state_dict().items()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(a).tobytes())
        for c in self.store.candidates() if self.store.root.is_dir() else []:
            h.update(c.to_json().encode())
        if self.log_path.is_file():
            h.update(self.log_path.read_bytes())
        return h.hexdigest()

    # ------------------------------------------------------------------------------------------ checkpoints
    def save_checkpoint(self) -> Path:
        root = self.out / "ckpt"
        root.mkdir(exist_ok=True)
        final = root / f"step_{self.step:09d}"
        tmp = root / f".tmp_{self.step:09d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        torch.save({"agent": self.agent.state_dict(), "behaviour": self.behaviour.state_dict(),
                    "torch_rng": torch.get_rng_state()}, tmp / "agent.pt")
        np.savez(tmp / "replay.npz", **self.replay.state_dict())
        np.savez(tmp / "env.npz", **self.env.get_state())
        np.savez(tmp / "rec.npz", **self.rec)
        rows = {}
        for i, e in enumerate(self.eps):
            sp = self.inp.specs[e.clip]
            rows[f"q{i}"] = _cat(e.qpos, sp.ex.model.nq)
            rows[f"c{i}"] = _cat(e.ctrl, sp.ex.model.nu)
        np.savez(tmp / "episodes.npz", **rows)
        snaps = set(self.versions.refs) | {self.ver}
        self.versions.write(snaps, tmp / "snapshots")
        state = {"step": self.step, "n_episodes": self.n_episodes, "ver": self.ver, "utd_credit": self.utd.credit,
                 "rng": self.rng.bit_generator.state, "episodes": [e.head() for e in self.eps],
                 "assist": self.assist.state(), "sampler": self.sampler.state(), "stats": self.stats,
                 "log_lines": self.log_lines, "refs": self.versions.refs, "config": self.cfg.as_dict(),
                 "stored_keys": sorted([t, e] for t, e in self.stored_keys)}
        (tmp / "trainer.json").write_text(json.dumps(state, default=float))
        tmp.rename(final)
        for old in sorted(root.glob("step_*"))[:-2]:
            shutil.rmtree(old)
        return final

    @staticmethod
    def latest(out_dir: Path) -> Path | None:
        c = sorted((Path(out_dir) / "ckpt").glob("step_*"))
        return c[-1] if c else None

    def resume(self) -> bool:
        """Load the newest checkpoint under the output directory; False if there is none (a fresh start)."""
        ck = self.latest(self.out)
        if ck is None:
            return False
        st = json.loads((ck / "trainer.json").read_text())
        blob = torch.load(ck / "agent.pt", weights_only=True)
        self.agent.load_state_dict(blob["agent"])
        self.behaviour.load_state_dict(blob["behaviour"])
        torch.set_rng_state(blob["torch_rng"])
        with np.load(ck / "replay.npz") as z:
            self.replay.load_state_dict({k: z[k] for k in z.files})
        with np.load(ck / "env.npz") as z:
            self.env.set_state({k: z[k] for k in z.files})
        with np.load(ck / "rec.npz") as z:
            self.rec = {k: z[k] for k in z.files}
        raw = {p.stem: p.read_bytes() for p in (ck / "snapshots").glob("*.bin")}
        self.versions.load_state(raw, {k: int(v) for k, v in st["refs"].items()})
        self.step, self.n_episodes, self.ver = int(st["step"]), int(st["n_episodes"]), st["ver"]
        self.utd.credit = float(st["utd_credit"])
        self.rng.bit_generator.state = st["rng"]
        self.assist.load_state(st["assist"])
        self.sampler.load_state(st["sampler"])
        self.stats = st["stats"]
        self.stored_keys = {(str(t), int(e)) for t, e in st.get("stored_keys", [])}
        with np.load(ck / "episodes.npz") as z:
            self.eps = []
            for i, h in enumerate(st["episodes"]):
                e = Episode(clip=h["clip"], episode=h["episode"], seed=h["seed"], start_row=h["start_row"],
                            alpha=h["alpha"], versions=list(h["versions"]))
                q, c = z[f"q{i}"], z[f"c{i}"]
                e.qpos, e.ctrl = ([q], [c]) if q.shape[0] else ([], [])
                self.eps.append(e)
        self.log_lines = int(st["log_lines"])
        if self.log_path.is_file():
            lines = self.log_path.read_text().splitlines(keepends=True)[:self.log_lines]
            self.log_path.write_text("".join(lines))
        return True


def _canon(v: Any) -> Any:
    """A picklable, value-only form of a state dict entry (tensors as numpy)."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return v
