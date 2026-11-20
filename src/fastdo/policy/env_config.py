"""Configuration and observation layout of the batched tracking environments (the environments are in
`fastdo.policy.env` and `fastdo.policy.env_warp`)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fastdo.policy.rewards import RewardConfig
from fastdo.policy.scene import N_LABELS

__all__ = ["N_FINGERS", "EnvConfig", "obs_layout"]

N_FINGERS = 22
_TICK_TOL = 1e-6


@dataclass(frozen=True, kw_only=True)
class EnvConfig:
    """Environment settings.

    Warm-up (`fastdo.policy.warmup`). `warmup_duration` is in seconds: the baseline's do_as_i_do value, 3.0 s
    (`warmup_duration`; its 600 `warmup_steps` count 5 ms simulator steps). Here that is `warmup_ctrl_steps = 150`
    control steps at 50 Hz, i.e. `warmup_sim_steps = 1500` simulator steps at `1/(hz n_substeps) = 2 ms`, and
    `warmup_duration * hz` must be an integer. With `warmup_open_hand` the hand starts open and backed off
    `warmup_clearance` (the baseline's `warmup_min_clearance`, 0.1 m) from the object and is driven to reference
    frame 0 over the whole warm-up; without it, the hand starts on frame 0.

    Evaluation (`fastdo.policy.rewards.clip_success`, W = `warmup_ctrl_steps`): stack the reset state, then
    `info["obj_pose"]` after every step. Element `W + j` is then the state at reference frame j (element W ends the
    warm-up on frame 0; `info["k"]` names each step's frame), the aligned reference is `obj_pose[max(i - W, 0)]`, and
    `warmup_steps = W`. With the per-step `info["obj_pose"]` alone (no reset state), element `W - 1 + j` is frame j
    and `warmup_steps = W - 1`.

    Control. `kp, kd` are the finger PD gains and `wrist_kp, wrist_kd` the wrist's (the scene's wrist stiffness,
    near-critical damping). Reward. `reward_scale` multiplies `step_reward` (0.02 keeps clip returns in ~1-10 with
    the default reward weights, `fastdo.policy.env` docstring) and `warmup_reward_scale` further scales warm-up steps
    (the baseline's `warmup_rew_multiplier`, 0.1 in its do_as_i_do config). `seed` drives reference randomization and
    reference-set draws; `nconmax, njmax` are MuJoCo Warp's per-world contact and constraint buffers (the
    baseline's values; sizing in `fastdo.policy.env_warp`); a world that overflows them fails."""

    n_worlds: int
    hz: float = 50.0
    n_substeps: int = 10
    horizon: int = 10
    delta_max: float = 0.3
    kp: float
    kd: float
    warmup_duration: float = 3.0
    randomize: bool
    warmup_open_hand: bool = True
    warmup_clearance: float = 0.1
    wrist_kp: float = 1000.0
    wrist_kd: float = 100.0
    reward_scale: float = 0.02
    warmup_reward_scale: float = 0.1
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int = 0
    nconmax: int = 120
    njmax: int = 500

    def __post_init__(self) -> None:
        positive = {"n_worlds": self.n_worlds, "hz": self.hz, "n_substeps": self.n_substeps, "horizon": self.horizon,
                    "delta_max": self.delta_max, "kp": self.kp, "wrist_kp": self.wrist_kp,
                    "reward_scale": self.reward_scale, "nconmax": self.nconmax, "njmax": self.njmax}
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        non_negative = {"kd": self.kd, "wrist_kd": self.wrist_kd, "warmup_reward_scale": self.warmup_reward_scale,
                        "warmup_duration": self.warmup_duration, "warmup_clearance": self.warmup_clearance}
        for name, value in non_negative.items():
            if not value >= 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        ticks = self.warmup_duration * self.hz
        if abs(ticks - round(ticks)) > _TICK_TOL:
            raise ValueError(f"warmup_duration * hz must be a whole number of control steps, got {ticks}")

    @property
    def warmup_ctrl_steps(self) -> int:
        """Warm-up length in control steps (the index `clip_success` takes on environment trajectories)."""
        return round(self.warmup_duration * self.hz)

    @property
    def warmup_sim_steps(self) -> int:
        return self.warmup_ctrl_steps * self.n_substeps


def obs_layout(horizon: int = 10) -> dict[str, slice]:
    """Name -> slice of the actor observation (`fastdo.policy.env` docstring), contiguous from 0 in this order."""
    widths = (("wrist_pose", 9), ("wrist_twist", 6), ("finger_q", N_FINGERS), ("finger_qd", N_FINGERS),
              ("obj_rot", 6), ("obj_twist", 6), ("obj_window", 6 * horizon), ("wrist_window", 6 * horizon),
              ("finger_window", N_FINGERS * horizon), ("contact_plan", N_LABELS * (horizon + 1)), ("shape", 10),
              ("prev_action", 6 + N_FINGERS), ("phase", 2))
    starts = np.cumsum([0, *(width for _, width in widths)])[:-1].tolist()
    return {name: slice(start, start + width) for (name, width), start in zip(widths, starts, strict=True)}
