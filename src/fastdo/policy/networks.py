"""Asymmetric actor-critic for the tracking policy (policy design, Section 7).

Actor: an ELU MLP from the observation o to the mean mu(o) of a diagonal Gaussian with a state-independent
log standard deviation, pi(a | o) = N(a; mu(o), diag(exp(2 log_std))). Critic: an ELU MLP of the same width
from the concatenation [o, p] of the observation and the privileged features p (contact forces, object
velocity; features the deployed policy cannot see), V(o, p).

The action is the unsquashed Gaussian sample. The environment applies the squashing
u = q_ref + delta_max tanh(a) (`fastdo.policy.protocol`), so PPO's probability ratios are ratios of Gaussian
densities of the same unsquashed sample: the tanh Jacobian log|det du/da| is a function of `a` alone and
cancels in pi_new(a | o) / pi_old(a | o), so `act`/`evaluate` return the plain Gaussian log-probability
without a correction term. The entropy is likewise the pre-squash Gaussian entropy (the entropy bonus is
off by default, `PpoConfig.ent_coef = 0`).

Initialization: orthogonal weights (gain sqrt(2) in hidden layers), zero biases, gain 0.01 on the actor's
output so the initial mean residual is ~0 (the policy starts on the reference), gain 1 on the critic's.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["LOG_STD_MAX", "LOG_STD_MIN", "ActorCritic"]

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
_HIDDEN_GAIN = math.sqrt(2.0)
_ACTOR_OUT_GAIN = 0.01
_CRITIC_OUT_GAIN = 1.0


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_dim
    for h in hidden:
        linear = nn.Linear(width, h)
        nn.init.orthogonal_(linear.weight, gain=_HIDDEN_GAIN)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.ELU()]
        width = h
    head = nn.Linear(width, out_dim)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian actor over observations and a critic over observations plus privileged features.

    Args:
        obs_dim: Actor observation width (`TrackingEnv.obs_dim`).
        priv_dim: Privileged feature width (`TrackingEnv.priv_dim`, may be 0).
        act_dim: Action width (`TrackingEnv.act_dim`).
        hidden: Hidden layer widths shared by both networks (`PpoConfig.hidden`).
        init_log_std: Initial value of every log standard deviation, in [LOG_STD_MIN, LOG_STD_MAX]; the
            log-std is clamped to that range when used and projected into it after every optimizer step
            (`clamp_log_std_`).

    Raises:
        ValueError: If `init_log_std` lies outside [LOG_STD_MIN, LOG_STD_MAX].
    """

    def __init__(
        self,
        obs_dim: int,
        priv_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (512, 512, 512),
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        if not LOG_STD_MIN <= init_log_std <= LOG_STD_MAX:
            raise ValueError(f"init_log_std must lie in [{LOG_STD_MIN}, {LOG_STD_MAX}], got {init_log_std}")
        self.obs_dim, self.priv_dim, self.act_dim = obs_dim, priv_dim, act_dim
        self.actor = _mlp(obs_dim, hidden, act_dim, _ACTOR_OUT_GAIN)
        self.critic = _mlp(obs_dim + priv_dim, hidden, 1, _CRITIC_OUT_GAIN)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def mean_action(self, obs: Tensor) -> Tensor:
        """Gaussian mean mu(o), (N, act_dim)."""
        return self.actor(obs)

    def _log_std(self) -> Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def action_std(self) -> Tensor:
        """The standard deviation actually sampled with, exp of the clamped log-std, (act_dim,)."""
        return self._log_std().exp()

    def clamp_log_std_(self) -> None:
        """Project the log-std parameter into [LOG_STD_MIN, LOG_STD_MAX] in place.

        Call after every optimizer step (`fastdo.policy.ppo.ppo_update` does). The clamp used in `act` and
        `evaluate` passes the gradient on the bounds but not strictly beyond them, so a parameter that the
        optimizer's momentum carried past a bound would otherwise receive no gradient to come back.
        """
        with torch.no_grad():
            self.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

    def value(self, obs: Tensor, priv: Tensor) -> Tensor:
        """Critic value V(o, p), (N,)."""
        return self.critic(torch.cat([obs, priv], dim=-1)).squeeze(-1)

    def _log_prob(self, mean: Tensor, log_std: Tensor, action: Tensor) -> Tensor:
        z = (action - mean) * torch.exp(-log_std)
        return (-0.5 * z**2 - log_std - 0.5 * math.log(2.0 * math.pi)).sum(-1)

    def act(
        self,
        obs: Tensor,
        priv: Tensor | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an unsquashed action for each world.

        Returns `(action (N, act_dim), logp (N,), value (N,))`. `deterministic` returns the mean. The value
        needs the privileged features: without `priv` (deployment, where they do not exist) it is NaN when
        `priv_dim > 0`, and `fastdo.policy.ppo.compute_gae` rejects it rather than letting it reach a loss.
        With `priv_dim == 0` the critic sees the observation alone and `act(obs)` returns its value.
        """
        mean = self.mean_action(obs)
        log_std = self._log_std()
        if deterministic:
            action = mean
        else:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            action = mean + noise * torch.exp(log_std)
        logp = self._log_prob(mean, log_std, action)
        if priv is None and self.priv_dim == 0:
            priv = obs.new_zeros((*obs.shape[:-1], 0))
        if priv is None:
            value = torch.full(logp.shape, math.nan, dtype=logp.dtype, device=logp.device)
        else:
            value = self.value(obs, priv)
        return action, logp, value

    def evaluate(self, obs: Tensor, priv: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Log-probability (N,), entropy (N,) and value (N,) of given unsquashed actions, differentiable."""
        mean = self.mean_action(obs)
        log_std = self._log_std()
        logp = self._log_prob(mean, log_std, action)
        entropy_per_dim = 0.5 + 0.5 * math.log(2.0 * math.pi) + log_std
        entropy = entropy_per_dim.sum(-1).expand(logp.shape)
        return logp, entropy, self.value(obs, priv)
