"""Posterior-aware reference randomization (policy design, Section 6).

A reference is perturbed along the reconstruction posterior instead of by iid jitter. The scale, which the
solver estimates as one scalar per clip, is perturbed once with its full variance `s = Sigma[6,6]`, and every
frame's pose perturbation is drawn from its conditional given that draw:

    log s <- log s + e_s,                e_s ~ N(0, s)   (once per clip),
    T_ref,k <- Exp(eps_k) T_ref,k,      eps_k = Sigma_k[:6,6] / s * e_s + C_k z_k,
    C_k = chol(Sigma_k[:6,:6] - Sigma_k[:6,6] Sigma_k[6,:6] / s + 1e-12 I)   (Schur complement),

so each frame's joint vector (eps_k, e_s) ~ N(0, Sigma_k) exactly, pose-scale cross terms included (`eps_k` is
a left perturbation `[rho, theta]` in `fastdo.geometry.lie`'s ordering). The earlier `log s + mean_k eps_k[6]`
shrank the scale's standard deviation toward sqrt(4 tau / (T Delta)) on long clips (0.40x at T = 230, 50 Hz,
tau = 0.2 s). `z_k` are six independent copies of the stationary critically damped second-order Gauss-Markov
process

    dx = v dt,  dv = (-x/tau^2 - 2v/tau) dt + sigma dW,  sigma^2 = 4/tau^3,
    P_inf = diag(1, 1/tau^2),  rho(Delta) = (1 + Delta/tau) exp(-Delta/tau),

sampled exactly: (x, v)_0 ~ N(0, P_inf), (x, v)_{k+1} = Phi (x, v)_k + w_k with Phi = expm(A Delta) and
w_k ~ N(0, P_inf - Phi P_inf Phi^T). Every frame's z_k has unit variance, consecutive frames correlate by
rho(1/hz) (0.995 at 50 Hz, tau = 0.2 s).
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.policy.reference import PolicyReference

__all__ = ["gauss_markov_discretization", "perturb_reference", "sample_reference_perturbation"]

_COV_JITTER = 1e-12
_N_PATHS = 6                  # pose coordinates; the scale is one draw per clip
_SCALE = 6                    # index of log s in the 7-vector
_SCALE_VAR_REL_TOL = 1e-5     # Sigma_k[6,6] must agree across frames (float32 interpolation stays far below)
_SCALE_COLUMN_REL_TOL = 1e-12


def gauss_markov_discretization(delta: float, tau: float) -> tuple[Tensor, Tensor]:
    """Exact float64 discretization of the unit-variance critically damped process over a step `delta`:
    the transition `Phi = expm(A delta)` with `A = [[0, 1], [-1/tau^2, -2/tau]]` and the process-noise
    covariance `Q = P_inf - Phi P_inf Phi^T`, `P_inf = diag(1, 1/tau^2)` (both (2,2))."""
    if delta <= 0.0 or tau <= 0.0:
        raise ValueError(f"delta and tau must be positive, got delta={delta}, tau={tau}")
    A = torch.tensor([[0.0, 1.0], [-1.0 / tau**2, -2.0 / tau]], dtype=torch.float64)
    Phi = torch.linalg.matrix_exp(A * delta)
    P_inf = torch.diag(torch.tensor([1.0, 1.0 / tau**2], dtype=torch.float64))
    Q = P_inf - Phi @ P_inf @ Phi.T
    return Phi, 0.5 * (Q + Q.T)


def _gauss_markov_paths(n_steps: int, delta: float, tau: float, generator: torch.Generator) -> Tensor:
    """Position components (n_steps, 6) of six independent stationary unit-variance paths, float64."""
    Phi, Q = gauss_markov_discretization(delta, tau)
    L_q = torch.linalg.cholesky(Q)
    L_inf = torch.diag(torch.tensor([1.0, 1.0 / tau], dtype=torch.float64))
    noise = torch.randn(n_steps, _N_PATHS, 2, dtype=torch.float64, generator=generator,
                        device=generator.device).cpu()
    state = noise[0] @ L_inf.T                                             # (6,2) ~ N(0, P_inf)
    out = torch.empty(n_steps, _N_PATHS, dtype=torch.float64)
    out[0] = state[:, 0]
    for k in range(1, n_steps):
        state = state @ Phi.T + noise[k] @ L_q.T
        out[k] = state[:, 0]
    return out


def sample_reference_perturbation(covariance: Tensor, hz: float, generator: torch.Generator,
                                  tau: float = 0.2) -> Tensor:
    """One perturbation path `eps (T,7)`: `eps[:, :6]` the per-frame pose perturbations and `eps[:, 6]` the clip's
    single log-scale draw `e_s` repeated on every frame, so that each `eps_k ~ N(0, covariance_k)` jointly and
    the pose part decorrelates as `(1 + Delta/tau) exp(-Delta/tau)` between frames `Delta = 1/hz` apart (module
    docstring). `covariance_k[6,6]` must be the same on every frame. Returned in the covariance's dtype and
    device; the random draws (`e_s` first, then the paths) come from `generator` (CPU or its own device)."""
    if covariance.ndim != 3 or covariance.shape[1:] != (7, 7):
        raise ValueError(f"covariance must be (T,7,7), got {tuple(covariance.shape)}")
    if covariance.shape[0] < 1:
        raise ValueError("covariance must have at least one frame")
    if hz <= 0.0:
        raise ValueError(f"hz must be positive, got {hz}")
    n_steps = int(covariance.shape[0])
    cov = covariance.detach().to("cpu", torch.float64)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    scale_var = cov[:, _SCALE, _SCALE]
    s = float(scale_var.mean())
    spread = float((scale_var - s).abs().max())
    if s < 0.0 or spread > _SCALE_VAR_REL_TOL * max(abs(s), 1e-300):
        raise ValueError(f"covariance[:, 6, 6] must be one non-negative log-scale variance shared by all frames, "
                         f"got range [{float(scale_var.min()):.6g}, {float(scale_var.max()):.6g}]")
    cross = cov[:, :_SCALE, _SCALE]                                          # (T,6)
    if s > 0.0:
        gain = cross / s
        schur = cov[:, :_SCALE, :_SCALE] - cross[:, :, None] * cross[:, None, :] / s
    elif float(cross.abs().max()) > 0.0:
        raise ValueError("covariance is not positive semidefinite: zero log-scale variance with cross terms")
    else:                                                                    # known scale
        gain = torch.zeros_like(cross)
        schur = cov[:, :_SCALE, :_SCALE]
    schur = schur + _COV_JITTER * torch.eye(_SCALE, dtype=torch.float64)
    C, info = torch.linalg.cholesky_ex(schur)
    if bool((info != 0).any()):
        bad = torch.nonzero(info).flatten().tolist()
        raise ValueError(f"covariance is not positive definite at frames {bad[:10]}")
    e_s = math.sqrt(s) * float(torch.randn((), dtype=torch.float64, generator=generator,
                                           device=generator.device))
    z = _gauss_markov_paths(n_steps, 1.0 / hz, tau, generator)
    pose = gain * e_s + (C @ z.unsqueeze(-1)).squeeze(-1)
    eps = torch.cat([pose, torch.full((n_steps, 1), e_s, dtype=torch.float64)], dim=-1)
    return eps.to(dtype=covariance.dtype, device=covariance.device)


def perturb_reference(ref: PolicyReference, eps: Tensor) -> PolicyReference:
    """Apply `eps (T,7)` from `sample_reference_perturbation`: object poses `Exp(eps[:, :6]) T` (left,
    world-frame perturbation) and log scale `+ e_s`, where `eps[:, 6]` must hold the one clip-level draw `e_s`
    on every frame. Joints, contact plan, covariance and rate are shared with `ref` (not copied)."""
    if eps.shape != (ref.num_frames, 7):
        raise ValueError(f"eps must be ({ref.num_frames},7), got {tuple(eps.shape)}")
    scale = eps[:, _SCALE]
    e_s = float(scale[0])
    if float((scale - e_s).abs().max()) > _SCALE_COLUMN_REL_TOL * max(1.0, abs(e_s)):
        raise ValueError("eps[:, 6] must repeat one clip-level log-scale perturbation on every frame")
    delta = se3_exp(eps[:, :_SCALE].to(ref.obj_pose))
    return dataclasses.replace(ref, obj_pose=delta @ ref.obj_pose, log_scale=ref.log_scale + e_s)
