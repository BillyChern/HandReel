"""Proximal policy optimization with an asymmetric actor-critic (policy design, Section 8).

Own implementation, no RL framework and no simulator import: the learner sees the environment only through
`fastdo.policy.protocol.TrackingEnv`.

Generalized advantage estimation (Schulman et al. 2016), per world, over a rollout of T steps:

    delta_t = r_t + gamma (1 - done_t) V(s_{t+1}) - V(s_t),        V(s_T) = last_value,
    A_t     = sum_l (gamma lam)^l prod_{j<l} (1 - done_{t+j}) delta_{t+l}
            = delta_t + gamma lam (1 - done_t) A_{t+1},             R_t = A_t + V(s_t),

so an episode end cuts both the bootstrap and the advantage sum (time-outs are bootstrapped in the reward by
`fastdo.policy.rollout.collect_rollout`).

Loss on a minibatch (Schulman et al. 2017), with the advantages normalized per minibatch,
A <- (A - mean A) / (std A + 1e-8), the ratio rho = exp(logp - logp_old) of the unsquashed Gaussian (the tanh
squashing happens in the environment, so no Jacobian term appears, see `fastdo.policy.networks`):

    L_pi = -mean min(rho A, clip(rho, 1 - clip, 1 + clip) A)
    L_V  = 0.5 mean max((V - R)^2, (V_old + clip(V - V_old, -value_clip, value_clip) - R)^2)
    L    = L_pi + value_coef L_V - ent_coef mean H[pi(. | o)],

minimized with Adam for `n_epochs` passes of `n_minibatches` random minibatches per rollout, with the global
gradient norm clipped to `max_grad_norm`, and the log-std projected back into its bounds after every step.
`value_clip` is in the units of the return: with value_clip 0.2 the critic moves at most 0.2 toward its
target per update on each sample, so rewards should be scaled such that returns are O(1-10) (e.g. per-step
rewards multiplied by the control period); `value_clip = math.inf` disables the value clipping.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from fastdo.policy.networks import ActorCritic
from fastdo.policy.protocol import TrackingEnv
from fastdo.policy.rollout import EpisodeTracker, PpoBatch, collect_rollout, flatten_rollout

__all__ = [
    "PpoConfig",
    "clipped_surrogate",
    "clipped_value_loss",
    "compute_gae",
    "ppo_loss",
    "ppo_update",
    "train",
]

logger = logging.getLogger(__name__)

_ADV_EPS = 1e-8
_N_LOG_LINES = 10


@dataclass(frozen=True)
class PpoConfig:
    """PPO hyper-parameters. `n_steps` is per world; a rollout holds `n_steps * n_worlds` samples.
    `value_coef` weights the value loss against the surrogate (the two networks share no parameters, so
    it matters only through the joint gradient-norm clip). `value_clip` is in return units; `math.inf`
    turns the clipped value loss into the plain 0.5 (V - R)^2."""

    n_steps: int = 32
    n_epochs: int = 5
    n_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    ent_coef: float = 0.0
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    hidden: tuple[int, ...] = (512, 512, 512)
    value_coef: float = 1.0


def compute_gae(
    rewards: Tensor, values: Tensor, dones: Tensor, last_value: Tensor, gamma: float, lam: float
) -> tuple[Tensor, Tensor]:
    """GAE advantages and returns over the leading time dimension.

    Args:
        rewards: (T, ...) rewards r_t.
        values: (T, ...) critic values V(s_t) of the states the actions were taken in.
        dones: (T, ...) bool or {0, 1}; done_t means the episode ended with step t.
        last_value: (...) value V(s_T) of the state after the last step.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        (advantages (T, ...), returns = advantages + values (T, ...)).

    Raises:
        ValueError: If `rewards`, `values` or `last_value` holds a NaN or an infinity (for example the NaN
            value `ActorCritic.act` returns without privileged features), which would otherwise pass
            silently into every advantage before it and into the losses. The check syncs once per call.
    """
    inputs = {"rewards": rewards, "values": values, "last_value": last_value}
    finite = torch.stack([torch.isfinite(x).all() for x in inputs.values()])
    if not bool(finite.all()):
        bad = [name for name, ok in zip(inputs, finite.tolist(), strict=True) if not ok]
        raise ValueError(f"compute_gae: non-finite {', '.join(bad)}")
    not_done = 1.0 - dones.to(values.dtype)
    advantages = torch.empty_like(values)
    gae = torch.zeros_like(values[0])
    next_value = last_value.to(values.dtype)
    for t in reversed(range(values.shape[0])):
        delta = rewards[t] + gamma * not_done[t] * next_value - values[t]
        gae = delta + gamma * lam * not_done[t] * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


def clipped_surrogate(logp: Tensor, logp_old: Tensor, advantages: Tensor, clip: float) -> Tensor:
    """-mean min(rho A, clip(rho, 1 - clip, 1 + clip) A), rho = exp(logp - logp_old); `advantages` as given
    (normalize before calling). The gradient vanishes on samples where the clipped branch is the minimum."""
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages
    return -torch.min(unclipped, clipped).mean()


def clipped_value_loss(values: Tensor, values_old: Tensor, returns: Tensor, value_clip: float) -> Tensor:
    """0.5 mean max((V - R)^2, (V_old + clip(V - V_old, -value_clip, value_clip) - R)^2)."""
    values_clipped = values_old + (values - values_old).clamp(-value_clip, value_clip)
    return 0.5 * torch.max((values - returns) ** 2, (values_clipped - returns) ** 2).mean()


def ppo_loss(model: ActorCritic, batch: PpoBatch, cfg: PpoConfig) -> tuple[Tensor, dict[str, Tensor]]:
    """Total PPO loss on one minibatch (advantages normalized here) and detached diagnostics:
    `policy_loss`, `value_loss`, `entropy`, `approx_kl` (the k3 estimator mean((rho - 1) - log rho)) and
    `clip_fraction` (share of samples with |rho - 1| > clip)."""
    logp, entropy, value = model.evaluate(batch.obs, batch.priv, batch.actions)
    adv = batch.advantages
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + _ADV_EPS)
    policy_loss = clipped_surrogate(logp, batch.logp_old, adv, cfg.clip)
    value_loss = clipped_value_loss(value, batch.values_old, batch.returns, cfg.value_clip)
    entropy_mean = entropy.mean()
    loss = policy_loss + cfg.value_coef * value_loss - cfg.ent_coef * entropy_mean
    with torch.no_grad():
        log_ratio = logp - batch.logp_old
        stats = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy_mean.detach(),
            "approx_kl": (torch.expm1(log_ratio) - log_ratio).mean(),
            "clip_fraction": ((torch.exp(log_ratio) - 1.0).abs() > cfg.clip).to(logp.dtype).mean(),
        }
    return loss, stats


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    cfg: PpoConfig,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """`n_epochs` passes over `batch`, each split into `n_minibatches` random minibatches, one Adam step per
    minibatch with the global gradient norm clipped to `max_grad_norm`, each followed by
    `model.clamp_log_std_()`.

    Returns the minibatch means of the `ppo_loss` diagnostics, the mean pre-clip `grad_norm`, and the
    rollout's `explained_variance` 1 - Var(R - V_old) / Var(R). `generator` (on its own device) drives the
    minibatch permutations.
    """
    n = len(batch)
    device = batch.logp_old.device
    perm_device = generator.device if generator is not None else device
    sums: dict[str, Tensor] = {}
    count = 0
    for _ in range(cfg.n_epochs):
        perm = torch.randperm(n, generator=generator, device=perm_device).to(device)
        for idx in perm.chunk(cfg.n_minibatches):
            loss, stats = ppo_loss(model, batch.index(idx), cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            stats["grad_norm"] = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).detach()
            optimizer.step()
            model.clamp_log_std_()
            for key, val in stats.items():
                sums[key] = sums[key] + val if key in sums else val.clone()
            count += 1
    out = {key: float(val) / count for key, val in sums.items()}
    var_returns = float(batch.returns.var())
    residual = float((batch.returns - batch.values_old).var())
    out["explained_variance"] = 1.0 - residual / var_returns if var_returns > 0.0 else math.nan
    return out


def train(
    env: TrackingEnv,
    model: ActorCritic,
    cfg: PpoConfig,
    n_updates: int,
    logger_fn: Callable[[dict[str, float]], None] | None = None,
    generator: torch.Generator | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> ActorCritic:
    """Run `n_updates` PPO iterations (rollout of `cfg.n_steps` steps, GAE, `ppo_update`) on `env`.

    The model must live on the environment's device. `logger_fn`, when given, receives per update the
    `ppo_update` diagnostics plus `update`, `env_steps`, `mean_step_reward` (raw reward per world-step),
    `episode_return`/`episode_length` (means over episodes completed during the rollout, NaN if none),
    `n_episodes`, `action_std` (mean of `model.action_std()`), `collect_seconds` and `update_seconds`.
    The module's log lines are subsampled (about 10 per call) and report the maximal `approx_kl` and
    `clip_fraction` since the previous line; take step-size statistics from `logger_fn`, which sees every
    update. `generator` drives the action noise and the minibatch permutations (it must be on the model's
    device); None uses the global RNG. `optimizer`, when given, is stepped instead of a fresh Adam at
    `cfg.lr`: keep it across calls and checkpoint its `state_dict` with the model to resume. Every call
    starts with `env.reset(None, None)`; episodes cut by that reset were already bootstrapped from the
    previous rollout's last value, so only the episode statistics lose them.
    Returns `model`, trained in place.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    obs = env.reset(None, None)
    tracker = EpisodeTracker(env.n_worlds, obs.device)
    env_steps = 0
    log_every = max(1, n_updates // _N_LOG_LINES)
    kl_max = clip_max = 0.0
    for update in range(n_updates):
        t0 = time.perf_counter()
        rollout, obs = collect_rollout(env, model, obs, cfg.n_steps, cfg.gamma, tracker, generator)
        t1 = time.perf_counter()
        advantages, returns = compute_gae(rollout.rewards, rollout.values, rollout.dones,
                                          rollout.last_value, cfg.gamma, cfg.lam)
        stats = ppo_update(model, optimizer, flatten_rollout(rollout, advantages, returns), cfg, generator)
        t2 = time.perf_counter()
        env_steps += cfg.n_steps * env.n_worlds
        ep_returns, ep_lengths = tracker.pop()
        stats.update(
            update=float(update),
            env_steps=float(env_steps),
            mean_step_reward=tracker.pop_mean_step_reward(),
            episode_return=float(ep_returns.mean()) if ep_returns.numel() else math.nan,
            episode_length=float(ep_lengths.double().mean()) if ep_lengths.numel() else math.nan,
            n_episodes=float(ep_returns.numel()),
            action_std=float(model.action_std().detach().mean()),
            collect_seconds=t1 - t0,
            update_seconds=t2 - t1,
        )
        if logger_fn is not None:
            logger_fn(stats)
        kl_max, clip_max = max(kl_max, stats["approx_kl"]), max(clip_max, stats["clip_fraction"])
        if update % log_every == 0 or update == n_updates - 1:
            logger.info("PPO update %d/%d: step reward %.4f, episode return %.4f, value loss %.4g, "
                        "kl %.4f (max since last line %.4f), clip %.3f (max %.3f), std %.3f", update + 1,
                        n_updates, stats["mean_step_reward"], stats["episode_return"], stats["value_loss"],
                        stats["approx_kl"], kl_max, stats["clip_fraction"], clip_max, stats["action_std"])
            kl_max = clip_max = 0.0
    return model
