"""Acting: seeded per-episode exploration noise, content-addressed policy versions, batch-invariant acting.

A certificate re-executes a candidate episode closed loop "with the same weights and noise". Both are made exact here:

- NOISE is counter-based: the standard-normal vector of policy step j of an episode with seed s is drawn from a torch
  CPU generator seeded with a hash of (s, j) (`step_noise`). Nothing depends on how many other worlds drew noise
  before, so a certifier regenerates step j directly from the stored seed. `episode_seed(run_seed, episode)` gives
  every episode of a run its own seed.
- WEIGHTS are versioned by content: `PolicyVersions.snapshot(actor)` serialises the actor's state dict (parameters
  and BatchNorm buffers, fixed key order, raw little-endian bytes) and names it by its sha256. A candidate stores the
  version of every policy step; the certifier loads each version and REFUSES bytes whose sha256 is not the stored name
  (`load`), so a certificate can never be produced by other weights than the ones that acted.
- ACTING is batch-invariant (`act`): every world's action is computed in a batch of one (the actor in eval mode, BN
  running statistics), so a world's action does not depend on which other worlds acted beside it, and the training
  rollout on the CPU executor equals its certificate bit for bit (GPU training rollouts are later certified on CPU
  without that guarantee; a CPU false negative only loses an acceptance).

The behaviour actor is the learner's actor copied every `act_sync_every` vectorised steps (an implementation
choice): it bounds the number of distinct versions per episode, hence the snapshot bytes a stored
candidate carries.

DEVICE PATH (`fastdo.amortised.trainer_dev`; an implementation choice): `step_noise` costs
~28 us per world and step on the host (a torch generator seeded through sha256), i.e. ~0.23 s per vectorised step at
8,192 worlds, more than the simulator. The device trainer therefore draws its exploration noise with `noise_hash`, a
vectorised counter-based normal (splitmix64 of (seed, step, dim) -> two uniforms -> Box-Muller, numpy uint64/float64
on the host, cast to float32): a pure function of (seed, step, dim), so the certifier regenerates it exactly from the
stored seed. A candidate records which generator acted (`NOISE_KINDS`; `step_noise` = "torch", the CPU path's, the
default) and `noise_for` dispatches on it. `act_batch` acts for all worlds in ONE eval-mode pass on the actor's device
(mathematically the batch-of-one action; numerically not guaranteed bit-equal, which the CPU certificate does not
need: a GPU rollout is certified by its own CPU re-execution).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from fastdo.amortised.sac import Actor

__all__ = ["NOISE_KINDS", "PolicyVersions", "VersionError", "act", "act_batch", "episode_seed", "noise_for",
           "noise_hash", "state_bytes", "step_noise"]

NOISE_KINDS = ("torch", "hash")

_MASK63 = (1 << 63) - 1


class VersionError(ValueError):
    """Snapshot bytes whose sha256 differs from the version they are stored under (or a missing version)."""


def _mix(*xs: int) -> int:
    h = hashlib.sha256(b"fastdo.amortised.acting:" + b":".join(str(int(x)).encode() for x in xs))
    return int.from_bytes(h.digest()[:8], "little") & _MASK63


def episode_seed(run_seed: int, episode: int) -> int:
    """The noise seed of global episode number `episode` of a run seeded `run_seed`."""
    return _mix(run_seed, episode)


def step_noise(seed: int, step: int, dim: int) -> torch.Tensor:
    """Standard-normal noise (dim,) float32 of policy step `step` of the episode with noise seed `seed`."""
    g = torch.Generator(device="cpu")
    g.manual_seed(_mix(seed, step))
    return torch.randn(dim, generator=g, dtype=torch.float32)


_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _splitmix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser of uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x + _GOLD
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def noise_hash(seeds: np.ndarray, steps: np.ndarray, dim: int) -> np.ndarray:
    """Standard-normal noise (n, dim) float32 of policy step `steps[i]` of the episode with noise seed `seeds[i]`
    (module docstring, DEVICE PATH): a pure, vectorised function of (seed, step, d)."""
    s = np.asarray(seeds, np.int64).astype(np.uint64)
    t = np.asarray(steps, np.int64).astype(np.uint64)
    key = _splitmix(_splitmix(s) ^ t)[:, None]
    d = np.arange(dim, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        a = _splitmix(key + np.uint64(2) * d)
        b = _splitmix(key + np.uint64(2) * d + np.uint64(1))
    u1 = ((a >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53
    u2 = ((b >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def noise_for(kind: str, seed: int, step: int, dim: int) -> torch.Tensor:
    """The (dim,) float32 noise of one policy step under generator `kind` (`NOISE_KINDS`)."""
    if kind == "torch":
        return step_noise(seed, step, dim)
    if kind == "hash":
        return torch.from_numpy(noise_hash(np.array([seed]), np.array([step]), dim)[0])
    raise ValueError(f"unknown noise kind {kind!r}")


def state_bytes(actor: torch.nn.Module) -> bytes:
    """Canonical bytes of a module's state dict: for each key in order, the key, dtype, shape and raw data."""
    buf = io.BytesIO()
    for k, v in actor.state_dict().items():
        t = v.detach().cpu().contiguous()
        buf.write(f"{k}|{t.dtype}|{tuple(t.shape)}\n".encode())
        buf.write(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes())
    return buf.getvalue()


def _to_state(actor_template: torch.nn.Module, raw: bytes) -> dict[str, torch.Tensor]:
    """Inverse of `state_bytes` against a module of the same architecture."""
    out: dict[str, torch.Tensor] = {}
    pos = 0
    for k, v in actor_template.state_dict().items():
        nl = raw.index(b"\n", pos)
        head = raw[pos:nl].decode()
        if head != f"{k}|{v.dtype}|{tuple(v.shape)}":
            raise VersionError(f"snapshot layout differs at {k}: {head}")
        pos = nl + 1
        n = v.numel() * v.element_size()
        arr = np.frombuffer(raw[pos:pos + n], dtype=v.detach().cpu().numpy().dtype).reshape(tuple(v.shape))
        out[k] = torch.from_numpy(arr.copy())
        pos += n
    if pos != len(raw):
        raise VersionError("trailing bytes in a snapshot")
    return out


class PolicyVersions:
    """Content-addressed actor snapshots, held in memory with reference counts and written to `root` on demand."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else None
        self._raw: dict[str, bytes] = {}
        self._refs: dict[str, int] = {}

    def snapshot(self, actor: torch.nn.Module) -> str:
        raw = state_bytes(actor)
        sha = hashlib.sha256(raw).hexdigest()
        self._raw.setdefault(sha, raw)
        return sha

    def retain(self, shas: Iterable[str]) -> None:
        for s in shas:
            self._refs[s] = self._refs.get(s, 0) + 1

    def release(self, shas: Iterable[str]) -> None:
        for s in shas:
            self._refs[s] -= 1
            if self._refs[s] <= 0:
                del self._refs[s]

    def gc(self, keep: Iterable[str] = ()) -> None:
        """Drop in-memory snapshots that are neither referenced nor in `keep`."""
        live = set(self._refs) | set(keep)
        for s in [s for s in self._raw if s not in live]:
            del self._raw[s]

    def raw(self, sha: str) -> bytes:
        if sha in self._raw:
            raw = self._raw[sha]
        elif self.root is not None and (self.root / f"{sha}.bin").is_file():
            raw = (self.root / f"{sha}.bin").read_bytes()
        else:
            raise VersionError(f"no snapshot {sha[:16]}")
        if hashlib.sha256(raw).hexdigest() != sha:
            raise VersionError(f"snapshot {sha[:16]}: bytes do not hash to their version")
        return raw

    def write(self, shas: Iterable[str], root: Path | None = None) -> None:
        """Persist snapshots as `<root>/<sha>.bin` (content-addressed; an existing file is kept)."""
        root = Path(root) if root is not None else self.root
        if root is None:
            raise ValueError("no snapshot root")
        root.mkdir(parents=True, exist_ok=True)
        for s in set(shas):
            p = root / f"{s}.bin"
            if not p.is_file():
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_bytes(self.raw(s))
                tmp.replace(p)

    def load(self, actor: torch.nn.Module, sha: str) -> None:
        """Load version `sha` into `actor`; refuses bytes that do not hash to `sha` (module docstring)."""
        actor.load_state_dict(_to_state(actor, self.raw(sha)))

    def state(self) -> dict[str, bytes]:
        return dict(self._raw)

    def load_state(self, raw: dict[str, bytes], refs: dict[str, int]) -> None:
        self._raw, self._refs = dict(raw), dict(refs)

    @property
    def refs(self) -> dict[str, int]:
        return dict(self._refs)


def act(actor: Actor, obs: np.ndarray, noise: torch.Tensor | None) -> np.ndarray:
    """Actions in [-1, 1] (n, A) for observations (n, D): one batch-of-one eval-mode pass per row (module docstring).
    `noise` (n, A) standard normal; None = the deterministic mean action (tanh of the mean)."""
    out = np.zeros((obs.shape[0], actor.act_dim), np.float32)
    with torch.no_grad():
        for i in range(obs.shape[0]):
            o = torch.as_tensor(np.asarray(obs[i:i + 1], np.float32))
            if noise is None:
                mean, _ = actor.mean_std(o, training=False)
                out[i] = torch.tanh(mean)[0].numpy()
            else:
                out[i] = actor.sample(o, training=False, noise=noise[i:i + 1])[0][0].numpy()
    return out


def act_batch(actor: Actor, obs: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
    """Actions in [-1, 1] (n, A) for observations (n, D) in ONE eval-mode pass on the actor's device (module
    docstring, DEVICE PATH); `noise` (n, A) or None for the deterministic mean action."""
    with torch.no_grad():
        if noise is None:
            mean, _ = actor.mean_std(obs, training=False)
            return torch.tanh(mean)
        return actor.sample(obs, training=False, noise=noise)[0]
