"""The batched tracking-environment protocol shared by the tracking environments and the PPO learner.

`fastdo.policy.ppo` is written against this protocol only, so it never imports a simulator; the
MuJoCo Warp environment and its plain-MuJoCo CPU twin and the toy environment of the PPO tests
implement it. Conventions every implementation follows:

- All tensors live on the environment's device; observations, privileged features and rewards are
  floating point, `done` is `torch.bool`. Every leading dimension is `n_worlds`.
- Actions are the raw, unbounded policy outputs (a Gaussian sample). The environment squashes them (the
  MuJoCo environments command the PD targets `u = q_ref + delta_max * tanh(action)`), so the learner's
  log-probabilities are those of the unsquashed Gaussian and need no tanh Jacobian correction.
- `step` auto-resets: a world whose `done` is True has already been reset, so the returned observation
  (and the next `privileged()`) belong to the first state of its new episode, while `reward` is the
  reward of the transition that ended the old one.
- `done` marks the end of an episode for any reason. When some of those ends are truncations (the clip or
  the episode time limit ran out, not a failure), `info["time_outs"]` is a bool tensor (n_worlds,) that
  is True exactly there, so the learner can bootstrap through them; an environment without truncations
  may omit the key. Other `info` entries (for example `obj_pose`, `contact_force`, `k`) are for logging
  and evaluation; the learner ignores them.

`StatefulTrackingEnv` adds the batched world-state copy with which the particle system of `fastdo.policy.smc`
resamples.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor

__all__ = ["TIME_OUTS_KEY", "StatefulTrackingEnv", "TrackingEnv"]

TIME_OUTS_KEY = "time_outs"


@runtime_checkable
class TrackingEnv(Protocol):
    """A batch of `n_worlds` independent tracking episodes stepped in lockstep.

    The dimensions are read-only properties so that implementations may expose them either as plain
    instance attributes or as properties.
    """

    @property
    def obs_dim(self) -> int:
        """Width of the actor observation."""
        ...

    @property
    def priv_dim(self) -> int:
        """Width of the privileged features that only the critic sees (may be 0)."""
        ...

    @property
    def act_dim(self) -> int:
        """Width of the (unsquashed) action."""
        ...

    @property
    def n_worlds(self) -> int:
        """Number of parallel worlds."""
        ...

    def reset(self, ref: Any, world_ids: Tensor | None) -> Tensor:
        """Start new episodes and return the observation of all worlds, (n_worlds, obs_dim).

        `ref` is the reference to track (a `fastdo.policy.reference.PolicyReference` for the MuJoCo
        environments; typed `Any` so this module does not depend on it); None lets the environment choose
        from its own reference set. `world_ids` (int64, (M,)) restricts the reset to those worlds, the
        others keep their state; None resets every world.
        """
        ...

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """Advance every world by one policy step with `action` (n_worlds, act_dim), unsquashed.

        Returns `(obs (n_worlds, obs_dim), reward (n_worlds,), done (n_worlds,) bool, info)` with the
        auto-reset and `info["time_outs"]` conventions of the module docstring.
        """
        ...

    def privileged(self) -> Tensor:
        """Privileged critic features (n_worlds, priv_dim) of the state the last returned observation
        describes (after any auto-reset)."""
        ...


@runtime_checkable
class StatefulTrackingEnv(TrackingEnv, Protocol):
    """A `TrackingEnv` whose worlds' states can be copied between worlds (`fastdo.policy.env_state` for the
    MuJoCo environments)."""

    def get_state(self, world_ids: Tensor) -> Any:
        """Copies of the states of worlds `world_ids` (int64, (M,), repeats allowed), in the order given:
        everything that determines each world's next step and what that step returns."""
        ...

    def set_state(self, world_ids: Tensor, state: Any) -> None:
        """Write the M states of `state` (from `get_state`) into the distinct worlds `world_ids`, in order."""
        ...
