"""Toy `TrackingEnv` for the PPO tests: a 1-D point mass tracking a sinusoid.

Per world (resampled at every episode start from the environment's own seeded generator): phase
phi ~ U(0, 2 pi), amplitude A ~ U(amplitude), mass m ~ U(mass). The reference is
x_ref(t) = A sin(omega t + phi), and an episode starts on it (x = x_ref(0), v = v_ref(0)), as the MuJoCo
environments start after the warm-up brings the hand to the first reference frame. The raw action a is
squashed in the environment, force = u_max tanh(a), and the mass follows semi-implicit Euler at `dt`:
v <- v + dt force / m, x <- x + dt v.

Actor observation (4): [x - x_ref, v - v_ref, A sin(phase), A cos(phase)], from which a linear feedback
plus feedforward (x_ref'' = -omega^2 A sin(phase)) law tracks; privileged critic features (2): [m - 1,
k / episode_steps], the mass being unobservable to the actor. Reward dt exp(-((x - x_ref) / sigma)^2) (a
tracking rate times the step, as in legged-robot PPO setups), so an episode return lies in
(0, episode_steps dt]. An episode terminates when |x - x_ref| > max_error (a failure: done, not a time-out)
and is truncated after `episode_steps` (done and `info["time_outs"]`); either way the world auto-resets.

`SlipChainEnv` (the SMC tests) is a grasp that slips independently of the action, and `TwistChainEnv` with
`TwistCritic` a chain whose particles differ in value; see their docstrings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fastdo.policy.networks import ActorCritic
from fastdo.policy.protocol import TIME_OUTS_KEY
from fastdo.policy.reference import PolicyReference

__all__ = ["SlipChainConfig", "SlipChainEnv", "ToyEnvConfig", "ToyTrackingEnv", "TwistChainConfig", "TwistChainEnv",
           "TwistCritic"]


@dataclass(frozen=True)
class ToyEnvConfig:
    n_worlds: int = 64
    episode_steps: int = 100
    dt: float = 0.05
    omega: float = math.pi
    amplitude: tuple[float, float] = (0.25, 0.5)
    mass: tuple[float, float] = (0.8, 1.25)
    u_max: float = 10.0
    sigma: float = 0.25
    max_error: float = 0.5
    seed: int = 0


class ToyTrackingEnv:
    obs_dim = 4
    priv_dim = 2
    act_dim = 1

    def __init__(self, cfg: ToyEnvConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ToyEnvConfig()
        self.n_worlds = self.cfg.n_worlds
        self._gen = torch.Generator().manual_seed(self.cfg.seed)
        n = self.n_worlds
        self.x = torch.zeros(n)
        self.v = torch.zeros(n)
        self.phase0 = torch.zeros(n)
        self.amp = torch.zeros(n)
        self.mass = torch.ones(n)
        self.k = torch.zeros(n, dtype=torch.long)
        self.reset(None, None)

    def _reference(self) -> tuple[Tensor, Tensor, Tensor]:
        phase = self.phase0 + self.cfg.omega * self.cfg.dt * self.k.to(self.phase0.dtype)
        return self.amp * torch.sin(phase), self.amp * self.cfg.omega * torch.cos(phase), phase

    def _obs(self) -> Tensor:
        x_ref, v_ref, phase = self._reference()
        return torch.stack(
            [self.x - x_ref, self.v - v_ref, self.amp * torch.sin(phase), self.amp * torch.cos(phase)],
            dim=-1,
        )

    def reset(self, ref: Any, world_ids: Tensor | None) -> Tensor:
        ids = torch.arange(self.n_worlds) if world_ids is None else world_ids
        u = torch.rand(ids.numel(), 3, generator=self._gen)
        (a_lo, a_hi), (m_lo, m_hi) = self.cfg.amplitude, self.cfg.mass
        self.phase0[ids] = 2.0 * math.pi * u[:, 0]
        self.amp[ids] = a_lo + (a_hi - a_lo) * u[:, 1]
        self.mass[ids] = m_lo + (m_hi - m_lo) * u[:, 2]
        self.k[ids] = 0
        x_ref, v_ref, _ = self._reference()
        self.x[ids] = x_ref[ids]
        self.v[ids] = v_ref[ids]
        return self._obs()

    def privileged(self) -> Tensor:
        return torch.stack([self.mass - 1.0, self.k.float() / self.cfg.episode_steps], dim=-1)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        cfg = self.cfg
        force = cfg.u_max * torch.tanh(action[:, 0])
        self.v = self.v + cfg.dt * force / self.mass
        self.x = self.x + cfg.dt * self.v
        self.k = self.k + 1
        x_ref, _, _ = self._reference()
        error = self.x - x_ref
        reward = cfg.dt * torch.exp(-((error / cfg.sigma) ** 2))
        failed = error.abs() > cfg.max_error
        time_outs = (self.k >= cfg.episode_steps) & ~failed
        done = time_outs | failed
        if bool(done.any()):
            self.reset(None, done.nonzero().squeeze(-1))
        return self._obs(), reward, done, {TIME_OUTS_KEY: time_outs}


@dataclass(frozen=True)
class SlipChainConfig:
    """The slip chain of the SMC tests (`SlipChainEnv`)."""

    n_worlds: int = 16
    n_chunks: int = 20
    chunk_steps: int = 1
    hz: float = 2.0            # chunk_steps / hz = 0.5 s, SmcConfig.chunk_seconds
    p_slip: float = 0.2
    drop: float = 1.0
    seed: int = 0


class SlipChainEnv:
    """A grasp that slips independently of the action: the SMC-versus-best-of-K toy of design Section 9b.

    A clip has `n_chunks` chunks of `chunk_steps` steps (`n_chunks chunk_steps + 1` reference frames, no warm-up). The
    object sits exactly on the reference until, on the first step of any chunk, the grasp slips with probability
    `p_slip` from the environment's own generator: the object is then `drop` below its reference and the episode
    fails (done, not a time-out) and auto-resets, so a failure is absorbing for that episode. The step onto the last
    frame is a time-out. Each episode draws a lineage `tag`; `info["q"] = [tag, k]` names the episode and frame of every
    recorded step, so an SMC output that stitches different lineages or skips frames is visible. The observation is
    `[k / (T - 1), 1]`, the same for every world on frame k (a critic sees no lineage), `priv_dim` is 0, and the reward
    is `1 / hz` per step. `get_state`/`set_state` copy `(k, tag)` between worlds; the slip generator is shared.
    """

    obs_dim = 2
    priv_dim = 0
    act_dim = 1

    def __init__(self, cfg: SlipChainConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else SlipChainConfig()
        self.n_worlds = self.cfg.n_worlds
        self._gen = torch.Generator().manual_seed(self.cfg.seed)
        self.n_frames = self.cfg.n_chunks * self.cfg.chunk_steps + 1
        pose = torch.eye(4, dtype=torch.float64).repeat(self.n_frames, 1, 1)
        pose[:, 0, 3] = 0.1 * torch.arange(self.n_frames, dtype=torch.float64) / self.cfg.hz
        pose[:, 2, 3] = 0.1
        self.reference = PolicyReference(q_ref=torch.zeros(self.n_frames, 1, dtype=torch.float64), obj_pose=pose,
                                         log_scale=0.0, contact_plan=torch.zeros(self.n_frames, 6, dtype=torch.bool),
                                         covariance=None, hz=self.cfg.hz)
        self.k = torch.zeros(self.n_worlds, dtype=torch.long)
        self.tag = torch.zeros(self.n_worlds, dtype=torch.float64)
        self.reset(None, None)

    def _obs(self) -> Tensor:
        phase = self.k.float() / (self.n_frames - 1)
        return torch.stack([phase, torch.ones_like(phase)], dim=-1)

    def reset(self, ref: Any, world_ids: Tensor | None) -> Tensor:
        ids = torch.arange(self.n_worlds) if world_ids is None else world_ids.long().flatten()
        self.k[ids] = 0
        self.tag[ids] = torch.rand(ids.numel(), dtype=torch.float64, generator=self._gen)
        return self._obs()

    def privileged(self) -> Tensor:
        return torch.zeros(self.n_worlds, 0)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        cfg = self.cfg
        if action.shape != (self.n_worlds, self.act_dim):
            raise ValueError(f"action must be ({self.n_worlds}, {self.act_dim}), got {tuple(action.shape)}")
        draw = torch.rand(self.n_worlds, generator=self._gen)
        slipped = (self.k % cfg.chunk_steps == 0) & (draw < cfg.p_slip)
        k_next = self.k + 1
        pose = self.reference.obj_pose[k_next].clone()
        pose[:, 2, 3] -= cfg.drop * slipped.to(pose.dtype)
        time_outs = (k_next == self.n_frames - 1) & ~slipped
        done = slipped | time_outs
        reward = torch.full((self.n_worlds,), 1.0 / cfg.hz)
        info = {TIME_OUTS_KEY: time_outs, "terminated": slipped, "obj_pose": pose, "k": k_next.clone(),
                "q": torch.stack([self.tag, k_next.to(torch.float64)], dim=-1)}
        self.k = k_next
        if bool(done.any()):
            self.reset(None, done.nonzero()[:, 0])
        return self._obs(), reward, done, info

    def world_kinematics(self) -> tuple[Tensor, Tensor]:
        """`info["q"] = [tag, k]` and the reference object pose of every world's current state (the particle system's
        reset-state hook)."""
        return torch.stack([self.tag, self.k.to(torch.float64)], dim=-1), self.reference.obj_pose[self.k].clone()

    def get_state(self, world_ids: Tensor) -> dict[str, Tensor]:
        ids = world_ids.long().flatten()
        return {"k": self.k[ids].clone(), "tag": self.tag[ids].clone()}

    def set_state(self, world_ids: Tensor, state: dict[str, Tensor]) -> None:
        ids = world_ids.long().flatten()
        self.k[ids] = state["k"]
        self.tag[ids] = state["tag"]


@dataclass(frozen=True)
class TwistChainConfig:
    """The critic-twist chain of the SMC tests (`TwistChainEnv`)."""

    n_worlds: int = 16
    n_chunks: int = 8
    chunk_steps: int = 2
    hz: float = 4.0            # chunk_steps / hz = 0.5 s, SmcConfig.chunk_seconds
    drift_std: float = 0.0
    p_doom: float = 0.0
    seed: int = 0
    init_std: float = 0.0


class TwistChainEnv(SlipChainEnv):
    """Particles whose twisted potentials differ: the SMC checks that `run_smc` weights by them.

    The clip, observation, lineage `tag` and time-out are `SlipChainEnv`'s (never a slip). A world also carries a
    feature `y` (`init_std N(0, 1)` at episode start, 0 by default) and a `doomed` flag, its privileged features
    `[y, doomed]`. Every step adds
    `drift_std N(0, 1)` to `y` and pays the new `y` as reward. On the first step of a chunk a doomed episode fails
    (done, not a time-out) and auto-resets, and a healthy one becomes doomed with probability `p_doom`. `trace` holds,
    per step of the current clip, the worlds' `tag`, `y` and `doomed` before the step and `reward`, `y_next` and
    `doomed_next` after it (before an auto-reset); `copies` the number of worlds each `set_state` wrote; `n_failures`
    counts failures over all clips. `reset(ref, None)` starts a clip and clears `trace` and `copies`.
    """

    priv_dim = 2

    def __init__(self, cfg: TwistChainConfig | None = None) -> None:
        tcfg = cfg if cfg is not None else TwistChainConfig()
        self.twist_cfg = tcfg
        self.y = torch.zeros(tcfg.n_worlds, dtype=torch.float64)
        self.doomed = torch.zeros(tcfg.n_worlds, dtype=torch.bool)
        self.trace: list[dict[str, Tensor]] = []
        self.copies: list[int] = []
        self.n_failures = 0
        super().__init__(SlipChainConfig(n_worlds=tcfg.n_worlds, n_chunks=tcfg.n_chunks, chunk_steps=tcfg.chunk_steps,
                                         hz=tcfg.hz, p_slip=0.0, seed=tcfg.seed))

    def reset(self, ref: Any, world_ids: Tensor | None) -> Tensor:
        ids = torch.arange(self.n_worlds) if world_ids is None else world_ids.long().flatten()
        self.y[ids], self.doomed[ids] = 0.0, False
        if self.twist_cfg.init_std > 0.0:           # no draw by default, so the other chains' random streams are unchanged
            self.y[ids] = self.twist_cfg.init_std * torch.randn(ids.numel(), generator=self._gen, dtype=torch.float64)
        if world_ids is None:
            self.trace, self.copies = [], []
        return super().reset(ref, world_ids)

    def privileged(self) -> Tensor:
        return torch.stack([self.y, self.doomed.to(torch.float64)], dim=-1)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        cfg = self.twist_cfg
        if action.shape != (self.n_worlds, self.act_dim):
            raise ValueError(f"action must be ({self.n_worlds}, {self.act_dim}), got {tuple(action.shape)}")
        row = {"tag": self.tag.clone(), "y": self.y.clone(), "doomed": self.doomed.clone()}
        first = self.k % cfg.chunk_steps == 0
        failed = first & self.doomed
        doom_draw = torch.rand(self.n_worlds, generator=self._gen)
        self.doomed = self.doomed | (first & (doom_draw < cfg.p_doom))
        self.y = self.y + cfg.drift_std * torch.randn(self.n_worlds, generator=self._gen, dtype=torch.float64)
        k_next = self.k + 1
        time_outs = (k_next == self.n_frames - 1) & ~failed
        done = failed | time_outs
        reward = self.y.clone()
        self.trace.append({**row, "reward": reward, "y_next": self.y.clone(), "doomed_next": self.doomed.clone()})
        info = {TIME_OUTS_KEY: time_outs, "terminated": failed, "obj_pose": self.reference.obj_pose[k_next].clone(),
                "k": k_next.clone(), "q": torch.stack([self.tag, k_next.to(torch.float64)], dim=-1)}
        self.k, self.n_failures = k_next, self.n_failures + int(failed.sum())
        if bool(done.any()):
            self.reset(None, done.nonzero()[:, 0])
        return self._obs(), reward, done, info

    def get_state(self, world_ids: Tensor) -> dict[str, Tensor]:
        ids = world_ids.long().flatten()
        return {**super().get_state(ids), "y": self.y[ids].clone(), "doomed": self.doomed[ids].clone()}

    def set_state(self, world_ids: Tensor, state: dict[str, Tensor]) -> None:
        ids = world_ids.long().flatten()
        super().set_state(ids, state)
        self.y[ids], self.doomed[ids] = state["y"], state["doomed"]
        self.copies.append(ids.numel())


class TwistCritic(ActorCritic):
    """An `ActorCritic` whose critic is `V = value_scale y - doom_penalty doomed` on `TwistChainEnv`'s privileged
    features `[y, doomed]` (its actor is irrelevant: the chain ignores actions)."""

    def __init__(self, value_scale: float, doom_penalty: float) -> None:
        super().__init__(TwistChainEnv.obs_dim, TwistChainEnv.priv_dim, TwistChainEnv.act_dim, hidden=(4,))
        self.value_scale, self.doom_penalty = value_scale, doom_penalty

    def value(self, obs: Tensor, priv: Tensor) -> Tensor:
        return self.value_scale * priv[:, 0].to(torch.float64) - self.doom_penalty * priv[:, 1].to(torch.float64)
