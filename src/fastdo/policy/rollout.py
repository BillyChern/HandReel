"""Batched on-policy rollout collection for PPO over a `TrackingEnv`.

`collect_rollout` steps all worlds `n_steps` times with the stochastic actor and stores, per step t and
world i, the actor observation o_t, the privileged features p_t, the unsquashed action a_t ~ pi(. | o_t),
its log-probability, the critic value V(o_t, p_t), the reward and the done flag, plus the bootstrap value
V(o_T, p_T) of the state after the last step.

Truncations. For worlds whose episode ended by time-out (`info["time_outs"]`, see
`fastdo.policy.protocol`), the stored reward is r_t + gamma V(o_t, p_t): the environment has already
auto-reset, so the value of the true final state s_{t+1} is unavailable, and the value of the state the last
action was taken in stands in for it (the rsl_rl convention). The done flag still cuts GAE there, so a
truncation is bootstrapped once instead of being treated as a failure with zero future reward. A time-out must end
its episode (`time_outs` a subset of `done`, `fastdo.policy.protocol`): otherwise gamma V(o_t, p_t) would be added
to a step whose episode continues and whose next value GAE also bootstraps. `collect_rollout` checks this once per
rollout (one host sync) and raises ValueError.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from fastdo.policy.networks import ActorCritic
from fastdo.policy.protocol import TIME_OUTS_KEY, TrackingEnv

__all__ = ["EpisodeTracker", "PpoBatch", "Rollout", "collect_rollout", "flatten_rollout"]


@dataclass(frozen=True)
class Rollout:
    """One rollout of `T` steps over `N` worlds (all tensors on the environment's device)."""

    obs: Tensor         # (T, N, obs_dim)
    priv: Tensor        # (T, N, priv_dim)
    actions: Tensor     # (T, N, act_dim), unsquashed
    logp: Tensor        # (T, N)
    values: Tensor      # (T, N)
    rewards: Tensor     # (T, N), time-outs bootstrapped
    dones: Tensor       # (T, N) bool
    last_value: Tensor  # (N,), V(o_T, p_T)


@dataclass(frozen=True)
class PpoBatch:
    """A flattened rollout with its GAE advantages and returns, the input of `ppo_update`."""

    obs: Tensor         # (B, obs_dim)
    priv: Tensor        # (B, priv_dim)
    actions: Tensor     # (B, act_dim)
    logp_old: Tensor    # (B,)
    values_old: Tensor  # (B,)
    advantages: Tensor  # (B,), unnormalized (normalized per minibatch in the loss)
    returns: Tensor     # (B,), advantages + values_old

    def __len__(self) -> int:
        return int(self.logp_old.shape[0])

    def index(self, idx: Tensor) -> PpoBatch:
        """The sub-batch at integer indices `idx`."""
        return PpoBatch(obs=self.obs[idx], priv=self.priv[idx], actions=self.actions[idx],
                        logp_old=self.logp_old[idx], values_old=self.values_old[idx],
                        advantages=self.advantages[idx], returns=self.returns[idx])


class EpisodeTracker:
    """Accumulates per-world undiscounted episode returns and lengths across rollouts."""

    def __init__(self, n_worlds: int, device: torch.device | str = "cpu") -> None:
        self._return = torch.zeros(n_worlds, dtype=torch.float64, device=device)
        self._length = torch.zeros(n_worlds, dtype=torch.long, device=device)
        self._done_returns: list[Tensor] = []
        self._done_lengths: list[Tensor] = []
        self._reward_sum = torch.zeros((), dtype=torch.float64, device=device)
        self._world_steps = 0

    def update(self, reward: Tensor, done: Tensor) -> None:
        """Add one step's raw rewards (n_worlds,) and close the episodes whose `done` is set."""
        self._return += reward.to(self._return.dtype)
        self._length += 1
        self._reward_sum += reward.to(self._return.dtype).sum()
        self._world_steps += int(reward.numel())
        if bool(done.any()):
            self._done_returns.append(self._return[done].clone())
            self._done_lengths.append(self._length[done].clone())
            self._return[done] = 0.0
            self._length[done] = 0

    def pop(self) -> tuple[Tensor, Tensor]:
        """Returns (float64) and lengths (int64) of the episodes completed since the last pop."""
        device = self._return.device
        returns = (torch.cat(self._done_returns) if self._done_returns
                   else torch.zeros(0, dtype=torch.float64, device=device))
        lengths = (torch.cat(self._done_lengths) if self._done_lengths
                   else torch.zeros(0, dtype=torch.long, device=device))
        self._done_returns, self._done_lengths = [], []
        return returns, lengths

    def pop_mean_step_reward(self) -> float:
        """Mean raw reward per world-step since the last call (NaN if no step was recorded)."""
        mean = float(self._reward_sum) / self._world_steps if self._world_steps else math.nan
        self._reward_sum.zero_()
        self._world_steps = 0
        return mean


@torch.no_grad()
def collect_rollout(
    env: TrackingEnv,
    model: ActorCritic,
    obs: Tensor,
    n_steps: int,
    gamma: float,
    tracker: EpisodeTracker | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Rollout, Tensor]:
    """Step `env` `n_steps` times from observation `obs` (n_worlds, obs_dim) with the stochastic actor.

    Returns the rollout and the observation to continue from. `tracker`, when given, receives the raw
    (un-bootstrapped) rewards and done flags.

    Time-outs bootstrap with the rsl_rl convention: a time-out step's stored reward gains `gamma V(o_t)`, the value of
    the observation the action was taken from, not of the terminal state s_T (the environment auto-resets and reports
    no terminal observation). With the potential-based shaping of design Section 5 this pairs the kept
    `gamma Phi(s_T)` with the shaped critic at s_{T-1}: a one-step approximation of rule (2), not an exact V(s_T)
    bootstrap.

    Raises:
        ValueError: If the environment flagged a time-out on a step that did not end the episode.
    """
    n = env.n_worlds
    device, dtype = obs.device, obs.dtype
    obs_buf = torch.empty(n_steps, n, env.obs_dim, dtype=dtype, device=device)
    priv_buf = torch.empty(n_steps, n, env.priv_dim, dtype=dtype, device=device)
    act_buf = torch.empty(n_steps, n, env.act_dim, dtype=dtype, device=device)
    logp_buf = torch.empty(n_steps, n, dtype=dtype, device=device)
    value_buf = torch.empty(n_steps, n, dtype=dtype, device=device)
    reward_buf = torch.empty(n_steps, n, dtype=dtype, device=device)
    done_buf = torch.empty(n_steps, n, dtype=torch.bool, device=device)
    orphan_time_out = torch.zeros((), dtype=torch.bool, device=device)
    for t in range(n_steps):
        priv = env.privileged()
        action, logp, value = model.act(obs, priv, generator=generator)
        obs_buf[t], priv_buf[t], act_buf[t], logp_buf[t], value_buf[t] = obs, priv, action, logp, value
        obs, reward, done, info = env.step(action)
        done = done.to(torch.bool)
        if tracker is not None:
            tracker.update(reward, done)
        time_outs = info.get(TIME_OUTS_KEY)
        if time_outs is not None:
            time_outs = time_outs.to(torch.bool)
            orphan_time_out |= (time_outs & ~done).any()
            reward = reward + gamma * value * time_outs.to(value.dtype)
        reward_buf[t], done_buf[t] = reward, done
    if bool(orphan_time_out):
        raise ValueError("env reported info['time_outs'] on a step whose done is False; time_outs must be a subset "
                         "of done (fastdo.policy.protocol)")
    last_value = model.value(obs, env.privileged())
    rollout = Rollout(obs=obs_buf, priv=priv_buf, actions=act_buf, logp=logp_buf, values=value_buf,
                      rewards=reward_buf, dones=done_buf, last_value=last_value)
    return rollout, obs


def flatten_rollout(rollout: Rollout, advantages: Tensor, returns: Tensor) -> PpoBatch:
    """Flatten the (T, N) leading dimensions of a rollout and its GAE output into a `PpoBatch`."""
    return PpoBatch(
        obs=rollout.obs.flatten(0, 1),
        priv=rollout.priv.flatten(0, 1),
        actions=rollout.actions.flatten(0, 1),
        logp_old=rollout.logp.flatten(),
        values_old=rollout.values.flatten(),
        advantages=advantages.flatten(),
        returns=returns.flatten(),
    )
