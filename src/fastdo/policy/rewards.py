"""Batched tracking reward, termination and the baseline success metric (policy design, Section 5).

    r_k = w_pose exp(-d_k^2 / sigma_pose^2)                                   object pose tracking
        + w_joint exp(-|q_k - q_ref,k|^2 / sigma_joint^2)                     joint tracking
        + w_contact (1/L) sum_l [a_kl c_kl - lambda_contact (1 - a_kl) c_kl]  contact-plan consistency
        - w_drop 1[drop_k] - w_action |u_k - u_{k-1}|^2                       drop and action-rate penalties

with the product metric on SO(3) x R^3

    d_k^2 = w_rot |Log_SO3(R_ref,k^T R_k)|^2 + w_tr |p_k - p_ref,k|^2,

`a_kl` the reference contact plan and `c_kl = 1[contact_force_kl > f_min]`. `d_k` is invariant under the
simultaneous left action of any g in SE(3) on both poses, and its two components are exactly the per-frame
quantities `fastdo.bench.metrics.retarget_errors` averages (E_pos and the geodesic E_rot), so the training
reward and the evaluation criterion share one metric. The se(3) log is deliberately not used: its
translational part `J_l^{-1}(omega) R_ref^T (p - p_ref)` equals `|p - p_ref|` only at omega = 0.

The joint metric Q is the identity. Heights are world z: the simulator's gravity is -z, so the drop test
is invariant under the gravity-fixing group (yaw and translations) but not under all of SE(3), as it
should be. Everything is batched over N worlds without host synchronization; reference indices past the
last frame are clamped to it, as in `future_window`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastdo.bench.metrics import RetargetErrors, retarget_errors
from fastdo.geometry.lie import so3_log
from fastdo.policy.reference import PolicyReference

logger = logging.getLogger(__name__)

__all__ = [
    "REWARD_TERMS",
    "RewardConfig",
    "StepObservation",
    "clip_success",
    "pose_errors",
    "shaping_potential",
    "shaping_term",
    "step_reward",
    "terminated",
]

REWARD_TERMS = ("pose", "joint", "contact", "drop", "action_rate")
_UP = 2                                                    # world z; MuJoCo's default gravity is (0, 0, -g)


@dataclass(frozen=True)
class RewardConfig:
    """Section 5 weights. `max_pos_error` (m) is the tracking-loss termination threshold on |p - p_ref|.

    Shaping (`shaping_potential`, `shaping_term`): `w_phi` and `d_phi` (m) set the potential Phi = -w_phi d / d_phi;
    `gamma` is the discount of the shaped MDP and must equal the learner's (PPO's) discount for the invariance."""

    w_pose: float = 1.0
    w_rot: float = 1.0
    w_tr: float = 10.0
    sigma_pose: float = 0.05
    w_joint: float = 0.5
    sigma_joint: float = 0.3
    w_contact: float = 0.5
    lambda_contact: float = 0.5
    f_min: float = 0.5
    w_drop: float = 10.0
    w_action: float = 0.01
    drop_height: float = 0.1
    max_pos_error: float = 0.3
    w_phi: float = 1.0
    d_phi: float = 0.1
    gamma: float = 0.99

    def __post_init__(self) -> None:
        for name in ("sigma_pose", "sigma_joint", "drop_height", "max_pos_error", "d_phi"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not self.w_phi >= 0.0 or not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"need w_phi >= 0 and gamma in (0, 1], got {self.w_phi}, {self.gamma}")


@dataclass(frozen=True)
class StepObservation:
    """What the environment measured after one control step, batched over N worlds.

    `q (N,nq)` joint positions in the layout of `PolicyReference.q_ref`; `qd (N,nv)` joint velocities;
    `obj_pose (N,4,4)` object poses; `obj_twist (N,6)` object velocity `[v, omega]`; `contact_force (N,L)`
    per-finger-label contact force magnitudes against the object (N); `action (N,na)` and `prev_action
    (N,na)` the current and previous controls; `k (N,)` integer reference step of each world.
    """

    q: Tensor
    qd: Tensor
    obj_pose: Tensor
    obj_twist: Tensor
    contact_force: Tensor
    action: Tensor
    prev_action: Tensor
    k: Tensor


def _check_batch(ref: PolicyReference, obs: StepObservation) -> int:
    """Validate shapes (no host sync) and return N."""
    n = int(obs.k.shape[0]) if obs.k.ndim == 1 else -1
    T = ref.num_frames
    nq = int(ref.q_ref.shape[-1])
    n_labels = int(ref.contact_plan.shape[-1])
    na = int(obs.action.shape[-1]) if obs.action.ndim == 2 else -1
    expected = {
        "k": (n,), "q": (n, nq), "obj_pose": (n, 4, 4), "obj_twist": (n, 6),
        "contact_force": (n, n_labels), "action": (n, na), "prev_action": (n, na),
    }
    bad = [f"{name} {tuple(getattr(obs, name).shape)} != {shape}" for name, shape in expected.items()
           if tuple(getattr(obs, name).shape) != shape]
    if obs.qd.ndim != 2 or int(obs.qd.shape[0]) != n:
        bad.append(f"qd {tuple(obs.qd.shape)} is not (N={n},nv)")
    if obs.k.dtype.is_floating_point or obs.k.dtype == torch.bool:
        bad.append(f"k must be an integer tensor, got {obs.k.dtype}")
    if ref.q_ref.shape[0] != T or ref.contact_plan.shape[0] != T:
        bad.append(f"reference lengths differ: obj_pose {T}, q_ref {ref.q_ref.shape[0]}, "
                   f"contact_plan {ref.contact_plan.shape[0]}")
    if n < 0 or na < 0 or bad:
        raise ValueError("StepObservation does not match the reference: " + "; ".join(bad))
    return n


def _reference_index(ref: PolicyReference, obs: StepObservation) -> Tensor:
    return obs.k.to(device=ref.obj_pose.device, dtype=torch.long).clamp(0, ref.num_frames - 1)


def pose_errors(obj_pose: Tensor, ref_pose: Tensor) -> tuple[Tensor, Tensor]:
    """Position error `|p - p_ref|` (m) and rotation angle `|Log_SO3(R_ref^T R)|` (rad), shapes (...,).

    These are the per-frame E_pos and E_rot of `fastdo.bench.metrics.retarget_errors` (its geodesic
    `arccos((tr(R R_ref^T) - 1)/2)` is the same angle), computed here with the atan2-based `so3_log`.
    """
    dp = obj_pose[..., :3, 3] - ref_pose[..., :3, 3]
    omega = so3_log(ref_pose[..., :3, :3].transpose(-1, -2) @ obj_pose[..., :3, :3])
    return torch.linalg.vector_norm(dp, dim=-1), torch.linalg.vector_norm(omega, dim=-1)


def _dropped(cfg: RewardConfig, obj_pose: Tensor, ref_pose: Tensor) -> Tensor:
    return ref_pose[:, _UP, 3] - obj_pose[:, _UP, 3] > cfg.drop_height


def step_reward(cfg: RewardConfig, ref: PolicyReference, obs: StepObservation
                ) -> tuple[Tensor, dict[str, Tensor]]:
    """Section 5 reward `(N,)` in the dtype of `obs.obj_pose`, and its terms for logging.

    The dict holds the signed contributions `pose`, `joint`, `contact`, `drop`, `action_rate`
    (`REWARD_TERMS`, which sum to the reward) and the diagnostics `e_pos`, `e_rot` (`pose_errors`).
    """
    _check_batch(ref, obs)
    dtype = obs.obj_pose.dtype
    idx = _reference_index(ref, obs)
    ref_pose = ref.obj_pose[idx].to(obs.obj_pose)

    dp = obs.obj_pose[:, :3, 3] - ref_pose[:, :3, 3]
    omega = so3_log(ref_pose[:, :3, :3].transpose(-1, -2) @ obs.obj_pose[:, :3, :3])
    rot_sq, tr_sq = (omega * omega).sum(-1), (dp * dp).sum(-1)
    pose = cfg.w_pose * torch.exp(-(cfg.w_rot * rot_sq + cfg.w_tr * tr_sq) / cfg.sigma_pose**2)

    dq = obs.q.to(dtype) - ref.q_ref[idx].to(dtype)
    joint = cfg.w_joint * torch.exp(-(dq * dq).sum(-1) / cfg.sigma_joint**2)

    touching = (obs.contact_force > cfg.f_min).to(dtype)
    planned = ref.contact_plan[idx].to(dtype)
    if touching.shape[-1] > 0:
        per_label = planned * touching - cfg.lambda_contact * (1.0 - planned) * touching
        contact = cfg.w_contact * per_label.mean(-1)
    else:
        contact = torch.zeros_like(pose)

    drop = -cfg.w_drop * _dropped(cfg, obs.obj_pose, ref_pose).to(dtype)
    du = (obs.action - obs.prev_action).to(dtype)
    action_rate = -cfg.w_action * (du * du).sum(-1)

    terms = {"pose": pose, "joint": joint, "contact": contact, "drop": drop, "action_rate": action_rate}
    reward = pose + joint + contact + drop + action_rate
    terms["e_pos"] = torch.sqrt(tr_sq)
    terms["e_rot"] = torch.linalg.vector_norm(omega, dim=-1)
    return reward, terms


def terminated(cfg: RewardConfig, ref: PolicyReference, obs: StepObservation) -> Tensor:
    """`(N,)` bool: the object fell `drop_height` below its reference height, or tracking is lost
    (`|p - p_ref| > max_pos_error`, the translational component of the product-metric logarithm)."""
    _check_batch(ref, obs)
    ref_pose = ref.obj_pose[_reference_index(ref, obs)].to(obs.obj_pose)
    e_pos = torch.linalg.vector_norm(obs.obj_pose[:, :3, 3] - ref_pose[:, :3, 3], dim=-1)
    return _dropped(cfg, obs.obj_pose, ref_pose) | (e_pos > cfg.max_pos_error)


def clip_success(pred_pose: np.ndarray, ref_pose: np.ndarray, warmup_steps: int
                 ) -> tuple[bool, RetargetErrors]:
    """Baseline success (E_pos < 0.1 m and E_rot < 0.5 rad) on the steps after the welded warm-up:
    `retarget_errors(pred_pose[warmup_steps:], ref_pose[warmup_steps:])` for (T,4,4) trajectories."""
    if pred_pose.shape != ref_pose.shape or pred_pose.ndim != 3 or pred_pose.shape[1:] != (4, 4):
        raise ValueError(f"trajectories must both be (T,4,4): {pred_pose.shape} vs {ref_pose.shape}")
    if not 0 <= warmup_steps < pred_pose.shape[0]:
        raise ValueError(f"warmup_steps must be in [0, {pred_pose.shape[0]}), got {warmup_steps}")
    errors = retarget_errors(pred_pose[warmup_steps:], ref_pose[warmup_steps:])
    logger.debug("clip_success: E_pos %.4f m, E_rot %.4f rad over %d steps -> %s", errors.e_pos,
                 errors.e_rot, pred_pose.shape[0] - warmup_steps, errors.success)
    return errors.success, errors


def shaping_potential(cfg: RewardConfig, obj_pose: Tensor, ref_pose: Tensor) -> Tensor:
    """Shaping potential Phi(s) = -w_phi d / d_phi (design Section 5), shape (...,), with d the product-metric distance
    `sqrt(w_rot |Log_SO3(R_ref^T R)|^2 + w_tr |p - p_ref|^2)` of `step_reward`'s pose term. Smooth in the pose, so it
    keeps a gradient where the pose kernel is numerically flat (4e-18 at the 0.1 m success boundary)."""
    dp = obj_pose[..., :3, 3] - ref_pose[..., :3, 3]
    omega = so3_log(ref_pose[..., :3, :3].transpose(-1, -2) @ obj_pose[..., :3, :3])
    d = torch.sqrt(cfg.w_rot * (omega * omega).sum(-1) + cfg.w_tr * (dp * dp).sum(-1))
    return -cfg.w_phi * d / cfg.d_phi


def shaping_term(cfg: RewardConfig, phi: Tensor, phi_next: Tensor, absorbing: Tensor) -> Tensor:
    """Potential-based shaping F = gamma Phi(s') - Phi(s) (Ng, Harada and Russell, ICML 1999), which leaves the optimal
    policies unchanged. `Phi(s')` is replaced by 0 exactly where the transition ends in an absorbing termination
    (`absorbing`: a drop, tracking loss or invalid state); a clip-end time-out keeps it, because the learner bootstraps
    V(s') there. Violating either rule would pay for ending episodes early."""
    kept = torch.where(absorbing, torch.zeros_like(phi_next), phi_next)
    return cfg.gamma * kept - phi
