"""Twisted sequential Monte Carlo with the tracking policy as proposal, and its no-resampling baseline best-of-K
(policy design, Section 9b).

Particles. `K = SmcConfig.n_particles` particles live in the K worlds of one `StatefulTrackingEnv` (a `TrackingEnv`
whose worlds' states can be copied; `fastdo.policy.env_state` for the MuJoCo environments). Every particle tracks the
clip `ref`: with `reference_noise` and a covariance on `ref`, each world's episode starts on its own posterior draw
(`sample_reference_perturbation`, design Section 6), and every step applies the policy mean plus
`N(0, action_noise_std^2)` in the unsquashed action space. All randomness of the particle system comes from
`generator`. An environment built with `randomize=True` would perturb the references a second time and is rejected.

Schedule. With `W` warm-up control steps (`env.cfg.warmup_ctrl_steps`, 0 for an environment without one) and `T`
reference frames at `ref.hz`, a clip is `S = W + T - 1` steps and step `W - 1 + j` ends on frame `j` (the `EnvConfig`
evaluation note). The `T - 1` steps after the warm-up form `N = ceil((T - 1) / c)` chunks of `c = round(chunk_seconds
hz)` steps; chunk 0 also holds the warm-up. Every world's episode must end on step `S - 1` with a time-out: an earlier
time-out, or a live particle without one there, means the environment's warm-up or clip length differs from `ref`
(RuntimeError).

Potentials and resampling. After chunk j >= 1 with `H` steps, particle i has the twisted log-potential

    log G_i = R_i + gamma^H V(s_i,end) - V(s_i,start),     R_i = sum_h gamma^h r_i,h,

and chunk 0 has `R_i + gamma^H V(s_i,end)`: the particles start from the untwisted initial distribution, so the first
incremental weight carries no division by V(s_0) (design 9b amendment a; `subtract_start_potential` restores the
difference form for the ablation). Without `twist` the potential is `R_i` alone; `V` is the `ActorCritic` critic on the
observation and privileged features, `V(s_end) = 0` after the last chunk, where the clip is over, and a particle whose
episode ended in the chunk (a failure) has `-inf`. The weights are `w_i ~ exp(log G_i / lambda)` with
`lambda = max(ess_temperature(log G, ess_fraction), lambda_floor_c sigma_v)` (amendment d): the ESS is `ess_fraction K`
unless the floor binds, and every live particle weighs the same (`lambda = inf`) when their potentials are equal. After every
chunk but the last `run_smc` draws ancestors by `systematic_resample`; a particle with offspring keeps its own world,
and its extra offspring are copied into the worlds of particles without any, the fewest copies for those offspring
counts (`n_resamples` counts the chunks after which a copy happened). `run_best_of_k` computes the same per-chunk
diagnostics and never resamples. If every particle has failed, the clip stops: that chunk reports ESS 0, later chunks
NaN.

Output. The trajectory is the ancestral path of one final particle: among the particles alive at the end, in
decreasing order of the undiscounted return along their path, the first whose object trajectory passes `clip_success`
against `ref` (warm-up `W - 1` steps); `success` is whether one does (a failed particle's path is never a success:
`clip_success` averages the error over the clip, so a short drop could pass it). `qpos (S, nq + 7)` is `info["q"]`
followed by the object free-joint qpos of `info["obj_pose"]` (the MuJoCo scene's full qpos, 35 columns) and
`obj_pose (S, 4, 4)`; both float64, NaN on steps never simulated. `start_qpos` is that path's reset state (before step
0, the state of the world that simulated its chunk 0) in the same layout, read from `env.world_kinematics()` right after
the reset; empty for an environment without that method.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from fastdo.policy.networks import ActorCritic
from fastdo.policy.protocol import TIME_OUTS_KEY, StatefulTrackingEnv
from fastdo.policy.randomize import perturb_reference, sample_reference_perturbation
from fastdo.policy.reference import PolicyReference
from fastdo.policy.rewards import clip_success
from fastdo.policy.scene import object_qpos

logger = logging.getLogger(__name__)

__all__ = ["SmcConfig", "SmcResult", "chunk_bounds", "chunk_horizon", "chunk_log_potentials", "ess_temperature",
           "run_best_of_k", "run_smc", "systematic_resample"]

_BISECTION_STEPS = 200
_LOG_ESS_TOL = 1e-10          # |log ESS - log target| that ends the bisection
_EQUAL_REL_TOL = 1e-6         # live potentials within 1e-6 max(1, |v|) are equal (float32 rewards and critic values)
_LOW_PAD, _HIGH_PAD = 5.0, 30.0
_WARNED_ATTR = "_smc_warned_no_covariance"   # per-environment flag (logging only), as BatchedTrackingEnv._draw


@dataclass(frozen=True)
class SmcConfig:
    """Particle-system settings (module docstring); `chunk_seconds` is rounded to whole steps at the reference rate.

    `subtract_start_potential` weights chunk 0 in the difference form `R_0 + gamma^H V(s_end) - V(s_0)` of later chunks
    (the original formula, kept for the ablation) instead of `R_0 + gamma^H V(s_end)` (design 9b amendment a, default).
    `lambda_floor_c` and `sigma_v` set the tempering floor `lambda_j = max(lambda_ESS, lambda_floor_c sigma_v)`
    (amendment d): `sigma_v` is the critic's calibrated chunk-residual sd (`fastdo.policy.smc_calibration`), 0 (no
    floor) until calibrated."""

    n_particles: int = 64
    chunk_seconds: float = 0.5
    ess_fraction: float = 0.5
    gamma: float = 0.99
    twist: bool = True
    action_noise_std: float = 0.1
    reference_noise: bool = True
    subtract_start_potential: bool = False
    lambda_floor_c: float = 1.0
    sigma_v: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lambda_floor_c) and self.lambda_floor_c >= 0.0 and math.isfinite(self.sigma_v)
                and self.sigma_v >= 0.0):
            raise ValueError(f"lambda_floor_c and sigma_v must be finite and >= 0, got {self.lambda_floor_c}, "
                             f"{self.sigma_v}")
        if self.n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {self.n_particles}")
        if not self.chunk_seconds > 0.0:
            raise ValueError(f"chunk_seconds must be positive, got {self.chunk_seconds}")
        if not 0.0 < self.ess_fraction <= 1.0:
            raise ValueError(f"ess_fraction must lie in (0, 1], got {self.ess_fraction}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must lie in (0, 1], got {self.gamma}")
        if not self.action_noise_std >= 0.0:
            raise ValueError(f"action_noise_std must be non-negative, got {self.action_noise_std}")


@dataclass(frozen=True)
class SmcResult:
    """One clip (module docstring): the selected path's `qpos (S, nq)` and `obj_pose (S, 4, 4)` (float64), its
    `success`, the per-chunk `ess_per_chunk (N,)` and `lambda_per_chunk (N,)`, and `n_resamples`. `n_success` counts the
    final particles whose own path passes `clip_success` (for best-of-K: successful independent rollouts), and
    `chunk_residuals (N, K)` holds each live particle's `R + gamma^H V(s_end) - V(s_start)` per chunk (NaN for dead
    particles, unsimulated chunks or without `twist`), the input of `fastdo.policy.smc_calibration`.
`start_qpos (nq + 7,)` is the selected path's reset state (empty when the environment has no `world_kinematics`)."""

    qpos: np.ndarray
    obj_pose: np.ndarray
    success: bool
    ess_per_chunk: np.ndarray
    lambda_per_chunk: np.ndarray
    n_resamples: int
    n_success: int = 0
    chunk_residuals: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    start_qpos: np.ndarray = field(default_factory=lambda: np.zeros(0))


def ess_temperature(log_potentials: Tensor, ess_fraction: float) -> float:
    """Temperature lambda at which w = exp(log_potentials / lambda) has ESS (sum w)^2 / sum w^2 = ess_fraction K.

    Bisection on log lambda: the ESS of tempered weights grows monotonically with lambda, from the number of particles
    tied at the maximum (lambda -> 0) to the number of finite potentials (lambda -> inf). `-inf` potentials (dead
    particles) weigh 0 at every lambda. Returns `inf` when the finite potentials are equal (within
    `1e-6 max(1, |v|)`) or too few are finite to reach the target, and the bracket's lower end `min gap e^-5` when the
    tie at the maximum alone reaches it.

    Raises:
        ValueError: If `ess_fraction` is outside (0, 1], a potential is NaN or +inf, or none is finite.
    """
    if not 0.0 < ess_fraction <= 1.0:
        raise ValueError(f"ess_fraction must lie in (0, 1], got {ess_fraction}")
    v = log_potentials.detach().to("cpu", torch.float64).flatten().numpy()
    if bool(np.isnan(v).any()) or bool(np.isposinf(v).any()):
        raise ValueError("log-potentials must be finite or -inf")
    live = v[np.isfinite(v)]
    if live.size == 0:
        raise ValueError("every log-potential is -inf: no live particle")
    target = ess_fraction * v.size
    gaps = float(live.max()) - live
    spread = float(gaps.max())
    if spread <= _EQUAL_REL_TOL * max(1.0, float(np.abs(live).max())) or live.size <= target:
        return math.inf

    def log_ess(log_lam: float) -> float:
        z = -gaps / math.exp(log_lam)
        return float(2.0 * np.logaddexp.reduce(z) - np.logaddexp.reduce(2.0 * z))

    log_target = math.log(target)
    lo, hi = math.log(float(gaps[gaps > 0.0].min())) - _LOW_PAD, math.log(spread) + _HIGH_PAD
    if log_ess(lo) >= log_target:
        return math.exp(lo)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        error = log_ess(mid) - log_target
        if abs(error) <= _LOG_ESS_TOL or hi - lo <= 1e-14:
            return math.exp(mid)
        lo, hi = (mid, hi) if error < 0.0 else (lo, mid)
    return math.exp(0.5 * (lo + hi))


def systematic_resample(weights: Tensor, generator: torch.Generator) -> Tensor:
    """Ancestor indices (K,) int64 on the weights' device, sorted: one u ~ U[0, 1) from `generator`, positions
    (u + i) / K, and ancestor i the first particle whose cumulative normalized weight exceeds position i. Unbiased
    (E[offspring_j] = K w_j / sum w), with offspring in {floor, ceil} of K w_j / sum w; zero weights never reproduce.

    Raises:
        ValueError: If the weights are empty, negative or non-finite, or sum to zero.
    """
    w = weights.detach().to(torch.float64).flatten()
    k = w.numel()
    if k == 0 or not bool(torch.isfinite(w).all()) or bool((w < 0.0).any()) or not float(w.sum()) > 0.0:
        raise ValueError("weights must be non-empty, finite, non-negative and not all zero")
    cdf = torch.cumsum(w / w.sum(), dim=0)
    u = float(torch.rand((), dtype=torch.float64, generator=generator, device=generator.device))
    positions = (u + torch.arange(k, dtype=torch.float64, device=w.device)) / k
    last_positive = int(torch.nonzero(w > 0.0)[-1, 0])            # rounding may leave cdf[-1] just below a position
    return torch.searchsorted(cdf, positions, right=True).clamp_max(last_positive)


def _keep_in_place(ancestors: Tensor) -> Tensor:
    """The ancestors' offspring counts, rearranged so that every particle with offspring is its own world's ancestor:
    only the extra offspring move, into the worlds of the particles without offspring."""
    index = torch.arange(ancestors.numel(), device=ancestors.device)
    counts = torch.bincount(ancestors, minlength=ancestors.numel())
    out = index.clone()
    out[counts == 0] = torch.repeat_interleave(index, (counts - 1).clamp_min(0))
    return out


def chunk_log_potentials(chunk_return: Tensor, v_start: Tensor, v_end: Tensor, n_steps: int, gamma: float,
                         alive: Tensor, twist: bool) -> Tensor:
    """Log-potentials (K,) float64 of one chunk: `R + gamma^n_steps V_end - V_start` (`R` without `twist`), `-inf` for
    dead particles and non-finite values."""
    log_g = chunk_return.to(torch.float64)
    if twist:
        log_g = log_g + gamma**n_steps * v_end.to(torch.float64) - v_start.to(torch.float64)
    return torch.where(alive & torch.isfinite(log_g), log_g, torch.full_like(log_g, -math.inf))


def run_smc(env: StatefulTrackingEnv, model: ActorCritic, ref: PolicyReference, cfg: SmcConfig,
            generator: torch.Generator) -> SmcResult:
    """Twisted SMC on one clip (module docstring). `model` must live on the environment's device."""
    return _run_particles(env, model, ref, cfg, generator, resample=True)


def run_best_of_k(env: StatefulTrackingEnv, model: ActorCritic, ref: PolicyReference, k: int,
                  generator: torch.Generator, cfg: SmcConfig | None = None) -> SmcResult:
    """K independent particles with the noise model of `cfg` (default `SmcConfig()`, `n_particles` replaced by `k`),
    never resampled: the best-of-K baseline of design Section 9b."""
    base = cfg if cfg is not None else SmcConfig()
    return _run_particles(env, model, ref, dataclasses.replace(base, n_particles=k), generator, resample=False)


def chunk_horizon(cfg: SmcConfig, hz: float) -> int:
    """The chunk horizon `c = round(chunk_seconds hz)` steps (at least 1); chunk 0 also holds the warm-up."""
    return max(1, round(cfg.chunk_seconds * hz))


def chunk_bounds(env: StatefulTrackingEnv, ref: PolicyReference, cfg: SmcConfig) -> tuple[int, tuple[int, ...]]:
    """(W, bounds): chunk j is steps [bounds[j], bounds[j + 1]) and bounds[-1] = S (module docstring)."""
    if env.n_worlds != cfg.n_particles:
        raise ValueError(f"n_particles ({cfg.n_particles}) must equal the environment's n_worlds ({env.n_worlds})")
    env_cfg = getattr(env, "cfg", None)
    if bool(getattr(env_cfg, "randomize", False)):
        raise ValueError("the particle system draws each particle's reference itself (SmcConfig.reference_noise): "
                         "build the environment with randomize=False")
    if ref.num_frames < 2:
        raise ValueError(f"the reference needs at least 2 frames, got {ref.num_frames}")
    warmup, frames = int(getattr(env_cfg, "warmup_ctrl_steps", 0)), ref.num_frames
    chunk = chunk_horizon(cfg, ref.hz)
    n_chunks = math.ceil((frames - 1) / chunk)
    return warmup, (0, *(warmup + min((j + 1) * chunk, frames - 1) for j in range(n_chunks)))


def _start(env: StatefulTrackingEnv, ref: PolicyReference, cfg: SmcConfig, generator: torch.Generator) -> Tensor:
    covariance = ref.covariance
    if cfg.reference_noise and covariance is None and not getattr(env, _WARNED_ATTR, False):
        logger.warning("reference_noise=True but the reference has no covariance; the particles track it unperturbed")
        try:
            setattr(env, _WARNED_ATTR, True)            # once per environment
        except AttributeError:                          # an environment that takes no attributes warns every run
            logger.debug("%s takes no attributes: the no-covariance warning repeats", type(env).__name__)
    if not cfg.reference_noise or covariance is None:
        return env.reset(ref, None)
    obs = torch.empty(0)
    for world in range(cfg.n_particles):
        draw = perturb_reference(ref, sample_reference_perturbation(covariance, ref.hz, generator))
        obs = env.reset(draw, torch.tensor([world]))
    return obs


def _start_kinematics(env: StatefulTrackingEnv) -> tuple[Tensor, Tensor] | None:
    """Every world's reset state as `(info["q"], info["obj_pose"])` of `env.world_kinematics()`, or None without it."""
    read = getattr(env, "world_kinematics", None)
    if read is None:
        return None
    q, pose = read()
    return q.to(torch.float64).clone(), pose.to(torch.float64).clone()


def _value(model: ActorCritic, env: StatefulTrackingEnv, obs: Tensor) -> Tensor:
    return model.value(obs, env.privileged()).to(torch.float64)


@torch.no_grad()
def _run_particles(env: StatefulTrackingEnv, model: ActorCritic, ref: PolicyReference, cfg: SmcConfig,
                   generator: torch.Generator, resample: bool) -> SmcResult:
    warmup, bounds = chunk_bounds(env, ref, cfg)
    n_particles, n_steps, n_chunks = cfg.n_particles, bounds[-1], len(bounds) - 1
    obs = _start(env, ref, cfg, generator)
    start = _start_kinematics(env)
    device = obs.device
    identity = torch.arange(n_particles, device=device)
    zeros = torch.zeros(n_particles, dtype=torch.float64, device=device)
    alive = torch.ones(n_particles, dtype=torch.bool, device=device)
    path_return, v_prev = zeros.clone(), (_value(model, env, obs) if cfg.twist else zeros)
    q_buf = torch.empty(0)
    pose_buf = torch.full((n_steps, n_particles, 4, 4), math.nan, dtype=torch.float64, device=device)
    bad_time_out = torch.zeros((), dtype=torch.bool, device=device)
    ancestry: list[Tensor] = []
    ess, lam = np.full(n_chunks, math.nan), np.full(n_chunks, math.nan)
    residuals = np.full((n_chunks, n_particles), math.nan)
    n_resamples, n_simulated = 0, 0
    for j in range(n_chunks):
        chunk_return = zeros.clone()
        for h, step in enumerate(range(bounds[j], bounds[j + 1])):
            noise = torch.randn(n_particles, env.act_dim, generator=generator, device=generator.device).to(device)
            obs, reward, done, info = env.step(model.mean_action(obs) + cfg.action_noise_std * noise)
            if step == 0:
                if "q" not in info or "obj_pose" not in info:
                    raise KeyError("the particle system records info['q'] and info['obj_pose'] of every step")
                q_buf = torch.full((n_steps, n_particles, info["q"].shape[-1]), math.nan, dtype=torch.float64,
                                   device=device)
            q_buf[step], pose_buf[step] = info["q"], info["obj_pose"]
            r = reward.to(torch.float64)
            chunk_return, path_return = chunk_return + cfg.gamma**h * r, path_return + r
            time_outs = info.get(TIME_OUTS_KEY, torch.zeros_like(done)).to(torch.bool)
            if step < n_steps - 1:
                alive, bad_time_out = alive & ~done, bad_time_out | time_outs.any()
            else:
                alive = alive & ~(done & ~time_outs)
                bad_time_out = bad_time_out | (alive & ~time_outs).any()
        n_simulated, last = j + 1, j == n_chunks - 1
        if bool(bad_time_out):
            raise RuntimeError(f"episodes must end with a time-out on step {n_steps - 1} (W {warmup} + T "
                               f"{ref.num_frames} - 1): the environment's warm-up or clip length differs")
        if not bool(alive.any()):
            ess[j] = 0.0
            break
        v_end = _value(model, env, obs) if cfg.twist and not last else zeros
        n_h = bounds[j + 1] - bounds[j]
        v_start = v_prev if j > 0 or cfg.subtract_start_potential else zeros      # chunk 0: amendment (a)
        log_g = chunk_log_potentials(chunk_return, v_start, v_end, n_h, cfg.gamma, alive, cfg.twist)
        if cfg.twist:                                   # the difference form on every chunk: the critic's residual
            diff = chunk_log_potentials(chunk_return, v_prev, v_end, n_h, cfg.gamma, alive, True).cpu().numpy()
            residuals[j] = np.where(np.isfinite(diff), diff, math.nan)
        anc = identity
        if bool(torch.isfinite(log_g).any()):
            lam[j] = max(ess_temperature(log_g, cfg.ess_fraction), cfg.lambda_floor_c * cfg.sigma_v)
            if math.isinf(lam[j]):
                weights = torch.isfinite(log_g).to(torch.float64)          # uniform over the live particles
            else:
                weights = torch.exp((log_g - log_g.max()) / lam[j])
            ess[j] = float(weights.sum() ** 2 / (weights * weights).sum())
            if resample and not last:
                anc = _keep_in_place(systematic_resample(weights, generator))
                moved = anc != identity
                if bool(moved.any()):
                    obs = obs[anc]                             # gathered before set_state writes the rows
                    env.set_state(identity[moved], env.get_state(anc[moved]))
                    n_resamples += 1
        else:
            ess[j] = 0.0
            logger.warning("chunk %d: live particles but no finite log-potential (non-finite critic?); not resampled", j)
        if not last:
            ancestry.append(anc)
        alive, path_return, v_prev = alive[anc], path_return[anc], v_end[anc]
    return _select(ref, warmup, bounds[:n_simulated + 1], q_buf, pose_buf, ancestry,
                   alive if n_simulated == n_chunks else torch.zeros_like(alive), path_return, ess, lam, n_resamples,
                   residuals, start)


def _select(ref: PolicyReference, warmup: int, bounds: tuple[int, ...], q_buf: Tensor, pose_buf: Tensor,
            ancestry: list[Tensor], alive: Tensor, path_return: Tensor, ess: np.ndarray, lam: np.ndarray,
            n_resamples: int, residuals: np.ndarray, start: tuple[Tensor, Tensor] | None = None) -> SmcResult:
    """Reconstruct every final particle's ancestral path and pick the output (module docstring)."""
    n_steps, n_particles = pose_buf.shape[0], pose_buf.shape[1]
    q_path, pose_path = torch.full_like(q_buf, math.nan), torch.full_like(pose_buf, math.nan)
    owner = torch.arange(n_particles, device=pose_buf.device)      # world that simulated chunk j of each final particle
    for j in reversed(range(len(bounds) - 1)):
        s0, s1 = bounds[j], bounds[j + 1]
        q_path[s0:s1], pose_path[s0:s1] = q_buf[s0:s1, owner], pose_buf[s0:s1, owner]
        if j > 0:
            owner = ancestry[j - 1][owner]
    frames = np.clip(np.arange(n_steps) - warmup + 1, 0, ref.num_frames - 1)
    ref_pose = ref.obj_pose.detach().to("cpu", torch.float64).numpy()[frames]
    poses = pose_path.cpu().numpy()
    order = torch.argsort(path_return, descending=True).tolist()
    live = [i for i, ok in zip(order, alive[order].tolist(), strict=True) if ok]
    passed = [i for i in live if clip_success(poses[:, i], ref_pose, max(warmup - 1, 0))[0]]
    chosen, success = (passed[0], True) if passed else ((live[0] if live else order[0]), False)
    pose = poses[:, chosen]
    obj_q = np.full((n_steps, 7), math.nan)
    rows = np.isfinite(pose).all(axis=(1, 2))
    if bool(rows.any()):
        obj_q[rows] = object_qpos(pose[rows])
    logger.debug("particle system: success %s, %d live of %d, %d resampling copies, mean ESS %.2f", success, len(live),
                 n_particles, n_resamples, float(np.nanmean(ess)) if bool(np.isfinite(ess).any()) else math.nan)
    start_qpos = np.zeros(0)
    if start is not None:                            # owner: the world that simulated chunk 0 of each final particle
        world = int(owner[chosen])
        start_obj = object_qpos(start[1][world].cpu().numpy()[None])[0]
        start_qpos = np.concatenate([start[0][world].cpu().numpy(), start_obj])
    return SmcResult(qpos=np.concatenate([q_path[:, chosen].cpu().numpy(), obj_q], axis=1), obj_pose=pose,
                     success=success, ess_per_chunk=ess, lambda_per_chunk=lam, n_resamples=n_resamples,
                     n_success=len(passed), chunk_residuals=residuals, start_qpos=start_qpos)
