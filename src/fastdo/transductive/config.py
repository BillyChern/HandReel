"""Settings of the transductive residual environment; frozen, plain data (checkpointed as a dict)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["SUCCESS_POS_M", "SUCCESS_ROT_RAD", "EnvConfig"]

SUCCESS_POS_M = 0.1      # DO AS I DO's success thresholds (PRIMARY / STRICT)
SUCCESS_ROT_RAD = 0.5


@dataclass(frozen=True, kw_only=True)
class EnvConfig:
    """Environment settings.

    Control. The policy acts every `rows_per_step` executor rows (5 ms each; 10 rows = 20 Hz). Its unbounded action a
    becomes the residual target `delta = delta_max (.) tanh(a)` with `delta_max` = (wrist position m, wrist rotation
    rad, finger rad) per dimension group; `hold="foh"` ramps the applied residual linearly from the previous target
    to the new one over the step's rows (continuous controls, like DO AS I DO's knot interpolation), `"zoh"` holds
    it. The applied control of row s is `ctrl_ref[s] + residual_s`: a zero action is DO AS I DO's kinematic warm
    start (T0) exactly.

    Reward (per policy step, summed over its rows s): `reward_scale * (alive_bonus + m_s get_reward_s)`, with
    get_reward DO AS I DO's registered per-step reward (`fastdo.transductive.reward`) and m_s its warmup multiplier;
    `success_bonus * reward_scale` is added on the step that completes a PRIMARY success. The alive bonus offsets the
    negative cost so that ending an episode early is never a gain. `info["raw_reward"]` is the sum of
    m_s get_reward_s alone (the known-answer quantity).

    Episode end. The executor's termination (object below z = -1 m at a block end; invalid state) and the clip's end
    (row T) end an episode; the end of the clip is terminal, not a truncation (nothing follows it). With `futility`,
    an episode also ends as soon as PRIMARY success has become impossible: the whole-trace mean error is the sum over
    T rows divided by T, so once the partial sum of position (rotation) errors reaches 0.1 T (0.5 T) no continuation
    can pass. This never ends an episode that could still succeed.

    Observation. Fixed per-group scales (`obs_scale_*`), clamped to +-`obs_clip`; `future_rows` are the reference
    look-ahead offsets in rows; `clip_onehot` appends the clip's one-hot id (transductive: the policy may specialise
    per clip because it only ever converts the clips it trains on).

    Assist curriculum (training only; `assist=True`). DO AS I DO's executor drops the object at the release unless a
    precise grasp is formed (80% of REF's own correction already fails), a cliff that unassisted exploration
    does not cross (smoke run: 2M rows on one clip, every episode dropped at the release). With the assist, an
    episode of assist level alpha in [0, 1] applies, from the release on, a wrench on the object toward its reference
    (a virtual object controller decayed by a curriculum, the idea of DexMachina, arXiv 2505.24853, re-derived):
    F = alpha m (-g + wn^2 (p_ref - p) + 2 wn (v_ref - v)), tau = alpha I (wn^2 R e_rot + 2 wn (w_ref - w)) at the
    object's COM, with m its mass, I its largest principal inertia, R e_rot the world-frame rotation error. Each
    clip has a level alpha_c (start `assist_start`); after every `assist_window` assisted episodes of a clip at its
    current level, alpha_c drops by `assist_step` if at least `assist_promote` of them met PRIMARY (under the assist).
    One world in `assist_free_every` of each clip always runs unassisted (alpha = 0), and ONLY unassisted episodes
    can produce successes: every stored rollout is an executor rollout without any assist. With
    `assist_observed=False` the observation's assist entry is held at 0 (the policy cannot tell assisted from
    unassisted episodes). With `assist_track=False` the assist is gravity compensation only (F = -alpha m g, no
    tracking terms, no torque): the hand must still carry the object along the reference, with part of its weight.

    Reference-state initialisation (training only; `rsi_frac > 0`, DeepMimic-style): an episode starts, with
    probability `rsi_frac`, at a random row r (a multiple of `rows_per_step` in [rows_per_step, T - rsi_min_rows])
    from the reference state (`sim_cpu.CpuWorlds.reset`: reference qpos/qvel, the object welded at its reference
    pose for `rsi_settle` rows after the release); with `rsi_et` it ends as soon as a row's object error exceeds the
    scorer's thresholds (0.1 m or 0.5 rad). RSI episodes never produce successes and have no futility rule; the
    draw for world w's e-th episode is `numpy default_rng([seed, w, e])` (`draw_start`), so it does not depend on the
    worker layout. Successes come only from episodes that start at row 0 (the executor's start), unassisted."""

    rows_per_step: int = 10
    delta_max: tuple[float, float, float] = (0.05, 0.10, 0.40)
    hold: str = "foh"
    alive_bonus: float = 0.5
    reward_scale: float = 0.01
    success_bonus: float = 0.0
    futility: bool = True
    future_rows: tuple[int, ...] = (10, 20, 40, 80)
    clip_onehot: bool = True
    obs_scale_pos: float = 0.05
    obs_scale_rot: float = 0.3
    obs_scale_joint: float = 0.3
    obs_clip: float = 10.0
    assist: bool = False
    assist_wn: float = 10.0
    assist_start: float = 1.0
    assist_step: float = 0.1
    assist_window: int = 16
    assist_promote: float = 0.5
    assist_free_every: int = 4
    assist_observed: bool = True
    assist_track: bool = True
    rsi_frac: float = 0.0
    rsi_settle: int = 20
    rsi_min_rows: int = 100
    rsi_et: bool = True
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rows_per_step < 1:
            raise ValueError("rows_per_step must be >= 1")
        if self.hold not in ("foh", "zoh"):
            raise ValueError(f"hold must be 'foh' or 'zoh', got {self.hold!r}")
        if len(self.delta_max) != 3 or min(self.delta_max) <= 0:
            raise ValueError("delta_max needs three positive scales (wrist pos, wrist rot, fingers)")
        if not self.future_rows or min(self.future_rows) < 1:
            raise ValueError("future_rows must be positive offsets")
        if not 0.0 <= self.rsi_frac <= 1.0 or self.rsi_settle < 0 or self.rsi_min_rows < 1:
            raise ValueError("rsi_frac must be in [0, 1], rsi_settle >= 0, rsi_min_rows >= 1")
        if self.assist_free_every < 1 or not 0.0 <= self.assist_start <= 1.0:
            raise ValueError("assist_free_every must be >= 1 and assist_start in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["delta_max"] = list(self.delta_max)
        d["future_rows"] = list(self.future_rows)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EnvConfig:
        d = dict(d)
        d["delta_max"] = tuple(d["delta_max"])
        d["future_rows"] = tuple(d["future_rows"])
        return EnvConfig(**d)


def draw_start(cfg: EnvConfig, T: int, world: int, episode: int) -> int:
    """Start row of world `world`'s `episode`-th episode: 0, or an RSI row (`EnvConfig` docstring)."""
    if cfg.rsi_frac <= 0.0:
        return 0
    import numpy as np

    rng = np.random.default_rng([int(cfg.seed), int(world), int(episode)])
    if rng.random() >= cfg.rsi_frac:
        return 0
    K = cfg.rows_per_step
    hi = (T - cfg.rsi_min_rows) // K
    return int(rng.integers(1, hi + 1)) * K if hi >= 1 else 0
