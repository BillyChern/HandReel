"""Behaviour-cloning warm start from the baseline MPC's executed rollouts.

Artifact (`trajectory_mjwp.npz`, written by `retargeting/pipeline/optimize_physics.py`, which `np.savez`es one stacked
entry per executed control chunk). Consumed keys, with `C` chunks of `n = ctrl_steps` (100) simulator steps:

- `qpos (C, n, nq)`, nq 35 = robot 28 + object free joint 7, and `qvel (C, n, nv)`: the state after each step;
- `ctrl (C, n, na)`, na 28: the `mj_data.ctrl` commanded for that step;
- `time (C, n)`: the simulation time after the step, `(s + 1) sim_dt` for absolute step `s`;
- `sim_step (C,)`: the step count after the chunk, so the chunk covers absolute steps `sim_step - n .. sim_step - 1`.

`load_mpc_states` orders chunks by `sim_step`, never by file position, and checks that they tile steps `0 .. C n - 1`;
row `j` of its output is then simulator step `j`. `ctrl` holds joint position targets (every Sharpa actuator is a
`general` actuator on one joint with `force = kp (ctrl - q) - kv qdot`), which MuJoCo clamps to `ctrlrange`; the
model timestep is `sim_dt` = 5 ms. This is the quantity the policy environment commands as its PD target `u`.

Baseline reference time base. `retargeting/utils/io.py:load_data` puts the kinematic reference on the simulator grid
with `align_to_sim_dt`: with `f = round(ref_dt / sim_dt)` (7 for the whisking `ref_dt` 0.0333) a T-frame reference
becomes `f T` samples of `F.interpolate(linear, align_corners=True)` after `warmup_steps` warm-up samples, ignoring the
npz's own `frequency` (50 Hz for the whisking clip): source frame i sits at post-warm-up step `i (f T - 1) / (T - 1)`,
35.2 ms per frame at T = 135 (28.4 Hz). `mpc_reference_frequency` is the source rate at which `resample_reference(kin,
..., hz=1/sim_dt)` reproduces those samples, the MPC's nominal control `q_mpc`.

Dataset on the environment's control grid (`bc_dataset`; the environment's 50 Hz clock is
authoritative). `ref` is the environment's reference (frames k at `ref.hz`); env frame k is the instant the MPC reached
at fractional post-warm-up step `p_k = k steps_per_frame` (`mpc_steps_per_frame`). MPC quantities are read there by
linear interpolation between rows: resampled in reference time, never by rollout step index.
- Residual. On the simulator grid `r_j = clamp(ctrl_j) - q_mpc[j - W]`, the MPC's command against its own same-step
  nominal control, with `ctrl` clamped to the actuator range first (default on, as MuJoCo and mujoco_warp do).
- Target base. The environment's action from `o_k` commands `q_ref[k + 1] + delta` over the period k -> k + 1, so
  the resampled residual is shifted by one control period: `delta_k = r(p_{k+1})`, and the PD target it reconstructs,
  `target_k = q_ref[k + 1] + delta_k`, is `q_ref[k + 1]` exactly for a zero-residual rollout.
- Saturation. `delta_k` is clipped to `saturation delta_max` (0.95) before the tanh-squashed loss, since 2-3 % of
  finger residuals exceed delta_max and would drive pre-tanh means to infinity; `BcDataset.clipped_fraction` reports it.
- Observation. `o_k = fastdo.policy.env_obs.observation_from_state` on the MPC state at the start of the period
  (rollout row `W + p_k - 1`, qpos and qvel interpolated, the object quaternion renormalized, qvel time-rescaled, see
  "Velocity time base"), frame k, `t = W_env + k`
  (after the environment's warm-up), the previous target residual `delta_{k-1}` (`delta_{-1}` from `r(p_0)`), the
  canonical x-axis carried from zeros, the reference window of `ref` from a `ReferenceBank` and zero contact forces
  (not logged; they enter only the privileged features).
- Pairs k = 0..K-1 need `p_{k+1}` inside both the post-warm-up rollout and the nominal: warm-up rows and rows past the
  last reference frame never enter.

Velocity time base (the same resampling applied to the observation). The MPC plays the reference
`c = steps_per_frame dt ref.hz` times slower than the environment (1.761 on the whisking clip: 35.2 ms against 20 ms per
frame), so its logged qvel, taken on its 28.4 Hz clock, is `1 / c` of the velocity of the same path on the environment's
50 Hz clock. With `rescale_velocity` (default) the interpolated qvel is multiplied by `c`: the observation's velocity
channels (wrist twist, finger qd, object twist; the privileged object twist) are those of the MPC path x_env(tau) =
x_mpc(c tau) replayed at the reference rate, the exact derivative of that time change for every joint and the free
joint's angular velocity.
Positions and residuals are positions and need no rescaling. What the rescaling cannot undo is dynamics: the MPC's
velocity-dependent PD lag and the object's gravity and contact timing belong to the slower playback, so the pairs remain
a distribution shift for the environment's actor (whether BC helps is not yet measured). `BcDataset.velocity_scale`
records `c` (1 without rescaling).

Two defaults (both kept until BC is measured against PPO):
- BC target. The target point-samples the MPC command at the start of the next control period: `delta_k = r(p_{k+1})`,
  the residual read at the single instant `p_{k+1}`, not averaged over the period k -> k + 1. This is consistent with
  the environment's PD target `q_ref[k + 1] + delta`, which the action holds over that period.
- Velocities. Rescaling to the environment's 50 Hz clock (`BcConfig.rescale_velocity`, `policy.bc.rescale_velocity`) is
  on by default, because the policy observes 50 Hz velocities.
- The pending measurement of whether BC shortens PPO also measures both (against a period-averaged target and
  unscaled velocities). Neither default changes before then.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.policy.bank import ReferenceBank
from fastdo.policy.env_obs import observation_from_state, observation_state, shape_descriptor
from fastdo.policy.networks import ActorCritic
from fastdo.policy.reference import PolicyReference
from fastdo.policy.scene import N_LABELS, PolicyScene

logger = logging.getLogger(__name__)

__all__ = ["BASELINE_SIM_DT", "BASELINE_WARMUP_STEPS", "BcConfig", "BcDataset", "MpcRollout", "bc_dataset",
           "bc_pretrain", "load_mpc_rollout", "load_mpc_states", "mpc_reference_frequency", "mpc_steps_per_frame"]

BASELINE_SIM_DT = 0.005        # s, the baseline's simulator step (`sim_dt`)
BASELINE_WARMUP_STEPS = 600    # simulator steps = 3.0 s welded warm-up (`warmup_steps`)
_REQUIRED_KEYS = ("qpos", "qvel", "ctrl", "time", "sim_step")
_TIME_TOL = 1e-6               # s; the baseline accumulates `time += sim_dt`, so float drift is ~1e-13
_RATE_TOL = 1e-6


@dataclass(frozen=True)
class MpcRollout:
    """One executed MPC rollout: `q (S, nq)`, `qvel (S, nv)` after simulator step j, `ctrl (S, na)` for step j, `dt`."""

    q: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    dt: float


@dataclass(frozen=True)
class BcConfig:
    """`bc_dataset` settings: the environment's `delta_max`, `horizon` and warm-up length in control steps, the MPC's
    warm-up in simulator steps, the target saturation (fraction of `delta_max`), ctrl clamping and whether observation
    velocities are rescaled to the environment's clock (module docstring, "Velocity time base")."""

    delta_max: float = 0.3
    horizon: int = 10
    env_warmup_steps: int = 150
    mpc_warmup_steps: int = BASELINE_WARMUP_STEPS
    saturation: float = 0.95
    clamp_ctrl: bool = True
    rescale_velocity: bool = True

    def __post_init__(self) -> None:
        if not self.delta_max > 0.0 or not 0.0 < self.saturation <= 1.0 or self.horizon < 1:
            raise ValueError(f"need delta_max > 0, saturation in (0, 1] and horizon >= 1, got {self}")
        if self.mpc_warmup_steps < 1 or self.env_warmup_steps < 0:
            raise ValueError(f"need mpc_warmup_steps >= 1 and env_warmup_steps >= 0, got {self}")


@dataclass(frozen=True)
class BcDataset:
    """`obs (K, obs_dim)` and `action (K, na)` (float32, the clipped residuals), `target (K, na)` float64 the PD
    targets `q_ref[k + 1] + action`, `frames (K,)`, the fraction of residual entries the saturation clipped and the
    factor applied to the MPC's qvel in the observations (`velocity_scale`, module docstring)."""

    obs: Tensor
    action: Tensor
    target: Tensor
    frames: Tensor
    clipped_fraction: float
    velocity_scale: float


def load_mpc_states(path: str | Path) -> MpcRollout:
    """Read a baseline `trajectory_mjwp.npz` (module docstring); row `j` is simulator step `j` and `dt` the least-squares
    step of `time = (j + 1) dt`.

    Raises:
        ValueError: A key is missing; the shapes disagree; `sim_step` has duplicates or does not tile steps
            `0 .. S - 1` in whole chunks; or `time` is not `(j + 1) dt` to `_TIME_TOL`.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in _REQUIRED_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{path}: missing keys {missing} (need {list(_REQUIRED_KEYS)})")
        arrays = {k: np.asarray(data[k], dtype=np.float64) for k in ("qpos", "qvel", "ctrl", "time")}
        sim_step = np.asarray(data["sim_step"])
    time = arrays["time"]
    if time.ndim != 2 or time.shape[0] < 1 or time.shape[1] < 1:
        raise ValueError(f"time must be (n_chunks, ctrl_steps), got {time.shape}")
    n_chunks, n = time.shape
    for key in ("qpos", "qvel", "ctrl"):
        if arrays[key].ndim != 3 or arrays[key].shape[:2] != (n_chunks, n):
            raise ValueError(f"{key} {arrays[key].shape} must be ({n_chunks}, {n}, dim) like time")
    if sim_step.shape != (n_chunks,):
        raise ValueError(f"sim_step must be ({n_chunks},), got {sim_step.shape}")
    order = np.argsort(sim_step, kind="stable")
    steps = sim_step[order].astype(np.int64)
    if np.any(np.diff(steps) == 0):
        raise ValueError(f"duplicate chunks: sim_step {np.sort(sim_step).tolist()}")
    if not np.array_equal(steps, n * np.arange(1, n_chunks + 1)):
        raise ValueError(f"chunks must be contiguous from simulator step 0 in chunks of {n}: sorted sim_step "
                         f"{steps.tolist()}")
    t = time[order].reshape(-1)
    ticks = np.arange(1, n_chunks * n + 1, dtype=np.float64)
    dt = float(t @ ticks / (ticks @ ticks))
    worst = float(np.abs(t - ticks * dt).max())
    if not dt > 0.0 or worst > _TIME_TOL:
        raise ValueError(f"time must be (step + 1) dt on a uniform grid: dt {dt:.6g}, worst deviation {worst:.3g} s")
    rows = {k: arrays[k][order].reshape(n_chunks * n, -1) for k in ("qpos", "qvel", "ctrl")}
    logger.debug("load_mpc_states: %d chunks x %d steps, dt %.6g s", n_chunks, n, dt)
    return MpcRollout(q=rows["qpos"], qvel=rows["qvel"], ctrl=rows["ctrl"], dt=dt)


def load_mpc_rollout(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """`(q (S, nq), ctrl (S, na), dt)` of `load_mpc_states`."""
    rollout = load_mpc_states(path)
    return rollout.q, rollout.ctrl, rollout.dt


def mpc_reference_frequency(n_frames: int, ref_dt: float, sim_dt: float = BASELINE_SIM_DT) -> float:
    """Source frequency of a `n_frames`-frame kinematic reference as the baseline MPC plays it (module docstring):
    `(n_frames - 1) / ((f n_frames - 1) sim_dt)` with `f = round(ref_dt / sim_dt)`.

    Raises:
        ValueError: `n_frames < 2`, a non-positive step, or `ref_dt < sim_dt` (the baseline then strides).
    """
    if n_frames < 2:
        raise ValueError(f"need at least 2 reference frames, got {n_frames}")
    if not (sim_dt > 0.0 and ref_dt > 0.0):
        raise ValueError(f"ref_dt and sim_dt must be positive, got {ref_dt}, {sim_dt}")
    if ref_dt < sim_dt:
        raise ValueError(f"ref_dt {ref_dt} < sim_dt {sim_dt}: the baseline strides through the reference; unsupported")
    factor = max(1, round(ref_dt / sim_dt))
    return (n_frames - 1) / ((factor * n_frames - 1) * sim_dt)


def mpc_steps_per_frame(n_source_frames: int, ref_dt: float, source_hz: float, env_hz: float,
                        sim_dt: float = BASELINE_SIM_DT) -> float:
    """MPC simulator steps per environment frame: env frame k is source position `k source_hz / env_hz`, which the MPC
    plays `1 / (mpc_reference_frequency sim_dt)` steps per source frame apart (7.045 for the whisking clip)."""
    if not (source_hz > 0.0 and env_hz > 0.0):
        raise ValueError(f"rates must be positive, got {source_hz}, {env_hz}")
    return (source_hz / env_hz) / (mpc_reference_frequency(n_source_frames, ref_dt, sim_dt) * sim_dt)


def _interpolate(x: Tensor, position: Tensor) -> Tensor:
    """Rows of `x (R, d)` at fractional `position (M,)` in [0, R - 1], linearly."""
    base = position.floor().long().clamp(0, max(int(x.shape[0]) - 2, 0))
    weight = (position - base.to(position.dtype))[:, None]
    upper = (base + 1).clamp_max(int(x.shape[0]) - 1)
    return (1.0 - weight) * x[base] + weight * x[upper]


def bc_dataset(scene: PolicyScene, rollout: MpcRollout, mpc_ref: PolicyReference, ref: PolicyReference,
               steps_per_frame: float, cfg: BcConfig) -> BcDataset:
    """Control-grid (observation, residual) pairs from one MPC rollout (module docstring, "Dataset").

    `mpc_ref` is the MPC's nominal on the simulator grid (`mpc_ref.hz == 1 / rollout.dt`); `ref` the environment's
    reference; `steps_per_frame` from `mpc_steps_per_frame`.

    Raises:
        ValueError: On mismatched shapes or rates, a rollout too short for one pair, or non-finite outputs.
    """
    q, qvel, ctrl = (torch.as_tensor(a, dtype=torch.float64) for a in (rollout.q, rollout.qvel, rollout.ctrl))
    na, W = scene.nq_robot, cfg.mpc_warmup_steps
    if ctrl.shape[1] != na or q.shape[0] != ctrl.shape[0] or qvel.shape[0] != ctrl.shape[0]:
        raise ValueError(f"rollout q {tuple(q.shape)}, qvel {tuple(qvel.shape)}, ctrl {tuple(ctrl.shape)} must share S "
                         f"with {na} ctrl columns")
    if abs(mpc_ref.hz * rollout.dt - 1.0) > _RATE_TOL or not steps_per_frame > 0.0:
        raise ValueError(f"mpc_ref.hz {mpc_ref.hz} must be 1/dt = {1.0 / rollout.dt} and steps_per_frame > 0")
    n_post = min(int(ctrl.shape[0]) - W, mpc_ref.num_frames)        # post-warm-up rows with a nominal
    if n_post < 2:
        raise ValueError(f"rollout of {ctrl.shape[0]} steps has no post-warm-up rows past warmup_steps {W}")
    u = ctrl[W:W + n_post]
    if cfg.clamp_ctrl:
        low, high = (torch.as_tensor(b, dtype=torch.float64) for b in (scene.ctrl_low, scene.ctrl_high))
        u = torch.minimum(torch.maximum(u, low), high)
    residual = u - mpc_ref.q_ref[:n_post, :na].to(torch.float64)
    n_pairs = min(int(np.floor((n_post - 1) / steps_per_frame + 1e-9)), ref.num_frames - 1)
    if n_pairs < 1:
        raise ValueError(f"{n_post} post-warm-up rows hold no control period at {steps_per_frame} steps per frame")
    frames = torch.arange(n_pairs, dtype=torch.long)
    position = torch.arange(n_pairs + 1, dtype=torch.float64) * steps_per_frame           # p_0 .. p_K
    bound = cfg.saturation * cfg.delta_max
    raw = _interpolate(residual, position)                                                  # r(p_k), k = 0..K
    delta = raw.clamp(-bound, bound)
    clipped = float((raw[1:].abs() > bound).double().mean())
    state_q = _interpolate(q, W - 1 + position[:-1])
    velocity_scale = steps_per_frame * rollout.dt * ref.hz if cfg.rescale_velocity else 1.0
    state_v = _interpolate(qvel, W - 1 + position[:-1]) * velocity_scale        # to the environment's clock
    quat = slice(scene.obj_qpos + 3, scene.obj_qpos + 7)
    state_q[:, quat] = state_q[:, quat] / state_q[:, quat].norm(dim=1, keepdim=True)
    bank = ReferenceBank(1, na, N_LABELS, ref.hz, torch.device("cpu"))
    bank.validate(ref)
    bank.write(0, ref)
    shape, gravity = shape_descriptor(scene, torch.device("cpu")), torch.as_tensor(scene.gravity, dtype=torch.float32)
    prev_x, rows = torch.zeros(1, 3), []
    for k in range(n_pairs):
        kk = torch.tensor([k])
        state = observation_state(scene, bank, state_q[k:k + 1].float(), state_v[k:k + 1].float(), kk,
                                  kk + cfg.env_warmup_steps, torch.zeros(1, N_LABELS), cfg.horizon,
                                  cfg.env_warmup_steps, gravity, shape)
        window = state.reference.obj.contact_plan.reshape(1, cfg.horizon + 1, N_LABELS)
        obs, _, prev_x = observation_from_state(state, prev_x, delta[k:k + 1].float(), window)
        rows.append(obs)
    action = delta[1:]                                                                      # delta_k = r(p_{k+1})
    target = ref.q_ref[1:n_pairs + 1, :na].to(torch.float64) + action
    obs_all = torch.cat(rows)
    if not (torch.isfinite(obs_all).all() and torch.isfinite(action).all()):
        raise ValueError("non-finite observations or residual actions")
    logger.info("bc_dataset: %d control-grid pairs (%.4g MPC steps per frame, %d warm-up rows excluded), |residual| "
                "mean %.4g max %.4g, %.2f%% of entries clipped at %.3g, qvel x %.4g", n_pairs, steps_per_frame, W,
                float(raw[1:].abs().mean()), float(raw[1:].abs().max()), 100.0 * clipped, bound, velocity_scale)
    return BcDataset(obs=obs_all, action=action.float(), target=target, frames=frames, clipped_fraction=clipped,
                     velocity_scale=velocity_scale)


def bc_pretrain(model: ActorCritic, dataset: tuple[Tensor, Tensor], n_epochs: int, lr: float, *,
                delta_max: float | None = None, batch_size: int = 4096, seed: int = 0) -> dict[str, float]:
    """Regress the actor's mean onto the residual actions with Adam; the critic and log-std are not touched.

    The prediction is `mean_action(obs)` or, with `delta_max`, the environment's squashed residual
    `delta_max tanh(mean_action(obs))` (`fastdo.policy.protocol`), so the loss is taken in the units the environment
    commands. Minibatches of `batch_size` are drawn without replacement per epoch from a generator seeded by `seed`.

    Returns:
        `mse_initial` and `mse_final` (mean squared error over the whole dataset before and after training),
        `n_samples`, `n_epochs`, `n_updates`.

    Raises:
        ValueError: On shape mismatches with the model, an empty dataset or non-positive hyperparameters.
    """
    obs, target = dataset
    if obs.ndim != 2 or target.ndim != 2 or obs.shape[0] != target.shape[0] or obs.shape[0] < 1:
        raise ValueError(f"dataset must be (M, obs_dim), (M, act_dim) with M >= 1, got {tuple(obs.shape)}, "
                         f"{tuple(target.shape)}")
    if obs.shape[1] != model.obs_dim or target.shape[1] != model.act_dim:
        raise ValueError(f"model expects obs {model.obs_dim} / act {model.act_dim}, got {obs.shape[1]} / "
                         f"{target.shape[1]}")
    if n_epochs < 1 or not lr > 0.0 or batch_size < 1 or (delta_max is not None and not delta_max > 0.0):
        raise ValueError(f"need n_epochs >= 1, lr > 0, batch_size >= 1, delta_max > 0 or None; got {n_epochs}, "
                         f"{lr}, {batch_size}, {delta_max}")
    param = model.log_std
    obs, target = obs.to(param.device, param.dtype), target.to(param.device, param.dtype)
    n = int(obs.shape[0])

    def predict(x: Tensor) -> Tensor:
        mean = model.mean_action(x)
        return mean if delta_max is None else delta_max * torch.tanh(mean)

    def dataset_mse() -> float:
        with torch.no_grad():
            sse = sum(float(F.mse_loss(predict(obs[i:i + batch_size]), target[i:i + batch_size], reduction="sum"))
                      for i in range(0, n, batch_size))
        return sse / (n * model.act_dim)

    optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    initial, n_updates = dataset_mse(), 0
    for _ in range(n_epochs):
        perm = torch.randperm(n, generator=gen).to(param.device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = F.mse_loss(predict(obs[idx]), target[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n_updates += 1
    final = dataset_mse()
    logger.info("bc_pretrain: %d samples, %d epochs, %d updates, MSE %.4e -> %.4e", n, n_epochs, n_updates, initial,
                final)
    return {"mse_initial": initial, "mse_final": final, "n_samples": float(n), "n_epochs": float(n_epochs),
            "n_updates": float(n_updates)}
