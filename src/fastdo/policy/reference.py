"""Policy reference and its chart at the current pose (policy design, Section 4).

Twist convention (inherited from `fastdo.geometry.lie`): `se3_exp`/`se3_log` use `xi = [rho, theta]`, the
translational part first and the rotation (axis-angle) second, and a pose is a (4,4) homogeneous transform.

`resample_reference` turns a kinematic reference (C3b, typically 10 fps) into the policy's control rate
(50 Hz) with geodesic SE(3) interpolation `T(t) = T_i Exp(t Log(T_i^{-1} T_{i+1}))`, which reproduces screw
motions exactly; joints and covariances use the same linear weights, the contact plan the nearest frame.
`future_window` expresses the next H reference poses as body-frame twists `Log(T_cur^{-1} T_ref,k+h)` in the
normal chart at the *current* pose; the encoding is invariant under any left action applied to both.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp, se3_inv, se3_log

if TYPE_CHECKING:  # the kinematic module pulls MuJoCo and the transport stack; only its type is needed
    from fastdo.retarget.kinematic import KinematicTrajectory

logger = logging.getLogger(__name__)

__all__ = ["PolicyReference", "future_window", "resample_reference"]

_LENGTH_SLACK = 1e-9   # absorbs float error in (T_src - 1) * hz / f_src when it is an exact integer


@dataclass(frozen=True)
class PolicyReference:
    """A clip's tracking reference at the policy rate.

    `q_ref (T,nq)` robot joint reference; `obj_pose (T,4,4)` object poses; `log_scale` the object's log
    scale; `contact_plan (T,L)` bool, which finger labels should touch; `covariance (T,7,7) | None` the
    reconstruction posterior on `(xi = [rho, theta] as a left perturbation, log s)`; `hz` the rate.
    """

    q_ref: Tensor
    obj_pose: Tensor
    log_scale: float
    contact_plan: Tensor
    covariance: Tensor | None
    hz: float

    @property
    def num_frames(self) -> int:
        return int(self.obj_pose.shape[0])


def _resample_grid(n_src: int, f_src: float, hz: float, device: torch.device, dtype: torch.dtype
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """Grid t_k = k / hz over [0, (n_src-1)/f_src] -> base index i, weight alpha, nearest index (n,)."""
    n_out = math.floor((n_src - 1) * hz / f_src + _LENGTH_SLACK) + 1
    k = torch.arange(n_out, device=device, dtype=dtype)
    s = k * f_src / hz                                                      # position in source frames
    i = torch.floor(s + _LENGTH_SLACK).long().clamp(0, max(n_src - 2, 0))
    alpha = (s - i.to(dtype)).clamp(0.0, 1.0)
    if n_src == 1:
        alpha = torch.zeros_like(alpha)
    nearest = torch.floor(s + 0.5).long().clamp(0, n_src - 1)
    return i, alpha, nearest


def resample_reference(kin: KinematicTrajectory, obj_pose: Tensor, contact_plan: Tensor,
                       covariance: Tensor | None, hz: float = 50.0, *, log_scale: float = 0.0
                       ) -> PolicyReference:
    """Resample a kinematic reference at `kin.frequency` to `hz` (design Section 4).

    Object poses follow the SE(3) geodesic between consecutive source frames, `q_ref` and `covariance` are
    interpolated linearly with the same weights (convex combinations keep the covariance PSD; left
    perturbations live in the world frame, so no chart transport is needed), and `contact_plan` takes the
    nearest source frame. The output spans the same time interval, `floor((T-1) hz / f) + 1` frames, in the
    dtype and device of `obj_pose`. `log_scale` (keyword-only, default 0) is carried through unchanged.

    `q_ref` is interpolated linearly in every coordinate, including any free-joint quaternion block or
    Euler-angle wrist coordinates in `kin.qpos`; `obj_pose` is the authority for the object pose.
    """
    if hz <= 0.0 or kin.frequency <= 0.0:
        raise ValueError(f"rates must be positive, got hz={hz}, kin.frequency={kin.frequency}")
    n_src = int(obj_pose.shape[0])
    if obj_pose.shape != (n_src, 4, 4) or n_src < 1:
        raise ValueError(f"obj_pose must be (T,4,4), got {tuple(obj_pose.shape)}")
    if kin.qpos.shape[0] != n_src or contact_plan.shape[0] != n_src:
        raise ValueError(f"length mismatch: qpos {kin.qpos.shape[0]}, obj_pose {n_src}, "
                         f"contact_plan {contact_plan.shape[0]}")
    if covariance is not None and covariance.shape != (n_src, 7, 7):
        raise ValueError(f"covariance must be (T,7,7) with T={n_src}, got {tuple(covariance.shape)}")
    dtype, device = obj_pose.dtype, obj_pose.device
    i, alpha, nearest = _resample_grid(n_src, float(kin.frequency), float(hz), device, dtype)
    j = (i + 1).clamp_max(n_src - 1)

    if n_src > 1:
        delta = se3_log(se3_inv(obj_pose[:-1]) @ obj_pose[1:])                 # (T-1,6) body-frame steps
        poses = obj_pose[i] @ se3_exp(alpha[:, None] * delta[i])
    else:
        poses = obj_pose[i].clone()

    q = torch.as_tensor(kin.qpos, dtype=dtype, device=device)
    w = alpha[:, None]
    q_ref = (1.0 - w) * q[i] + w * q[j]

    cov_out: Tensor | None = None
    if covariance is not None:
        c = covariance.to(dtype=dtype, device=device)
        cov_out = (1.0 - w[:, :, None]) * c[i] + w[:, :, None] * c[j]

    plan = contact_plan.to(device=device).bool()[nearest]
    logger.debug("resample_reference: %d frames at %.3g Hz -> %d frames at %.3g Hz",
                 n_src, kin.frequency, poses.shape[0], hz)
    return PolicyReference(q_ref=q_ref, obj_pose=poses, log_scale=float(log_scale), contact_plan=plan,
                           covariance=cov_out, hz=float(hz))


def future_window(ref: PolicyReference, k: Tensor, cur_pose: Tensor, cur_q: Tensor, horizon: int = 10
                  ) -> tuple[Tensor, Tensor]:
    """The next `horizon` reference steps in the chart at the current pose (design Section 4).

    Returns `twists (N,H,6) = Log(T_cur^{-1} T_ref,min(k+h, T-1))` and `dq (N,H,nq) = q_ref,min(k+h,T-1) -
    q_cur` for `h = 1..H`; indices past the last reference frame are clamped to it. Replacing both the
    reference poses and `cur_pose` by `g T` for any `g` in SE(3) leaves the twists unchanged.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    n = int(k.shape[0])
    if cur_pose.shape != (n, 4, 4) or cur_q.shape[0] != n:
        raise ValueError(f"batch mismatch: k {n}, cur_pose {tuple(cur_pose.shape)}, "
                         f"cur_q {tuple(cur_q.shape)}")
    T = ref.num_frames
    steps = torch.arange(1, horizon + 1, device=k.device, dtype=torch.long)
    idx = (k.long()[:, None] + steps[None]).clamp(0, T - 1)                      # (N,H)
    ref_pose = ref.obj_pose[idx]                                             # (N,H,4,4)
    twists = se3_log(se3_inv(cur_pose)[:, None] @ ref_pose)
    dq = ref.q_ref[idx] - cur_q[:, None]
    return twists, dq
