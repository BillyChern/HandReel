"""Posterior-aware reference randomization (design Section 6)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp, se3_inv, se3_log
from fastdo.policy.randomize import (
    gauss_markov_discretization,
    perturb_reference,
    sample_reference_perturbation,
)
from fastdo.policy.reference import PolicyReference

DT = torch.float64
HZ, TAU, T_REF, N_SAMPLES = 50.0, 0.2, 30, 4000


def _rho(delta: float, tau: float) -> float:
    return (1.0 + delta / tau) * math.exp(-delta / tau)


def _posterior_like_covariance(T: int, gen: torch.Generator) -> Tensor:
    """Non-diagonal (T,7,7) covariances with solver-like scales: ~5 mm, ~0.05 rad, ~0.02 log-scale."""
    std = torch.tensor([5e-3, 5e-3, 8e-3, 5e-2, 4e-2, 6e-2, 2e-2], dtype=DT)
    A = torch.randn(T, 7, 7, dtype=DT, generator=gen)
    C = A @ A.transpose(-1, -2) / 7.0 + 0.2 * torch.eye(7, dtype=DT)
    d = torch.diagonal(C, dim1=-2, dim2=-1).sqrt()
    corr = C / (d[:, :, None] * d[:, None, :])
    return corr * (std[:, None] * std[None, :])


@pytest.fixture(scope="module")
def samples() -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(0)
    cov = _posterior_like_covariance(T_REF, torch.Generator().manual_seed(1))
    eps = torch.stack([sample_reference_perturbation(cov, HZ, gen, tau=TAU) for _ in range(N_SAMPLES)])
    return cov, eps


def _whitened(cov: Tensor, eps: Tensor) -> tuple[Tensor, Tensor]:
    """Invert the conditional sampler: `z (N,T,6) = C_k^{-1} (eps_k[:6] - Sigma_k[:6,6] / s e_s)`, `C_k` the
    Cholesky factor of the Schur complement, and the standardized scale draw `e_s / sqrt(s)` (N,)."""
    s = cov[:, 6, 6].mean()
    cross = cov[:, :6, 6]
    schur = cov[:, :6, :6] - cross[:, :, None] * cross[:, None, :] / s
    C = torch.linalg.cholesky(schur + 1e-12 * torch.eye(6, dtype=DT))
    e_s = eps[:, 0, 6]
    resid = eps[..., :6] - (cross / s)[None] * e_s[:, None, None]
    z = torch.linalg.solve_triangular(C, resid.unsqueeze(-1), upper=False).squeeze(-1)
    return z, e_s / s.sqrt()


def test_gauss_markov_discretization_constants() -> None:
    for delta, tau in ((0.02, 0.2), (0.1, 0.2), (0.02, 0.05), (1.0, 0.3)):
        Phi, Q = gauss_markov_discretization(delta, tau)
        assert Phi.dtype == DT and Phi.shape == (2, 2) and Q.shape == (2, 2)
        e = math.exp(-delta / tau)
        closed = torch.tensor([[(1 + delta / tau) * e, delta * e],
                               [-delta / tau**2 * e, (1 - delta / tau) * e]], dtype=DT)
        assert torch.allclose(Phi, closed, atol=1e-12, rtol=1e-12)
        P_inf = torch.diag(torch.tensor([1.0, 1.0 / tau**2], dtype=DT))
        assert torch.allclose(Phi @ P_inf @ Phi.T + Q, P_inf, atol=1e-12)
        assert torch.linalg.eigvalsh(Q).min() > 0.0
        # Continuous Lyapunov equation A P + P A^T + sigma^2 e2 e2^T = 0 with sigma^2 = 4/tau^3.
        A = torch.tensor([[0.0, 1.0], [-1.0 / tau**2, -2.0 / tau]], dtype=DT)
        G = torch.tensor([[0.0, 0.0], [0.0, 4.0 / tau**3]], dtype=DT)
        assert (A @ P_inf + P_inf @ A.T + G).abs().amax() < 1e-9
        # Stationary autocorrelation of the position component: (Phi P_inf)[0,0] = rho(delta).
        assert abs(float((Phi @ P_inf)[0, 0]) - _rho(delta, tau)) < 1e-12


def test_perturbation_marginals_match_covariance(samples: tuple[Tensor, Tensor]) -> None:
    cov, eps = samples
    assert eps.shape == (N_SAMPLES, T_REF, 7) and eps.dtype == DT
    # The log-scale perturbation is one draw per clip, repeated on every frame.
    assert torch.equal(eps[..., 6], eps[:, :1, 6].expand(N_SAMPLES, T_REF))
    # Per-frame JOINT covariance of (eps_k[:6], e_s), cross terms included (zero-mean estimator).
    emp = torch.einsum("nki,nkj->kij", eps, eps) / N_SAMPLES
    rel = torch.linalg.matrix_norm(emp - cov) / torch.linalg.matrix_norm(cov)
    assert rel.max() < 0.10, f"per-frame relative Frobenius error up to {rel.max():.3f}"
    # Full scale variance (the review finding: mean_k eps_k[6] kept only 0.73x of it at T = 30, 0.16x at 230).
    var_s = float((eps[:, 0, 6] ** 2).mean())
    assert abs(var_s / float(cov[0, 6, 6]) - 1.0) < 0.10, var_s / float(cov[0, 6, 6])

    z, _ = _whitened(cov, eps)
    num = (z[:, :-1] * z[:, 1:]).sum()
    den = torch.sqrt((z[:, :-1] ** 2).sum() * (z[:, 1:] ** 2).sum())
    corr = float(num / den)
    target = _rho(1.0 / HZ, TAU)
    assert abs(corr - target) < 0.02, f"consecutive-step correlation {corr:.4f} vs {target:.4f}"


def test_perturbation_whitened_paths_unit_variance_and_lag_profile(
    samples: tuple[Tensor, Tensor],
) -> None:
    cov, eps = samples
    z, u_s = _whitened(cov, eps)
    assert z.shape == (N_SAMPLES, T_REF, 6)
    # Tolerances are ~7 standard errors of the pooled estimators, whose frames are correlated:
    # SE = sqrt(c * sum_jk rho(|j-k| Delta)^2 / (N T^2)) with c = 2 for mean z^2 (0.017 at N = 4000) and
    # c = 1 for the cross moments (0.012); 10 seeds measured max deviations of 0.043 (variance), 0.037
    # (cross-coordinate) and 0.033 (path-scale).
    var = (z**2).mean(dim=(0, 1))                                           # per coordinate
    assert (var - 1.0).abs().max() < 0.12, var
    # Cross-coordinate independence of the six paths, and independence from the scale draw.
    zc = torch.einsum("nki,nkj->ij", z, z) / (N_SAMPLES * T_REF)
    off = zc - torch.diag(torch.diagonal(zc))
    assert off.abs().max() < 0.09
    assert (torch.einsum("nki,n->i", z, u_s) / (N_SAMPLES * T_REF)).abs().max() < 0.09
    for lag in (1, 5, 10):
        num = (z[:, :-lag] * z[:, lag:]).sum()
        den = torch.sqrt((z[:, :-lag] ** 2).sum() * (z[:, lag:] ** 2).sum())
        target = _rho(lag / HZ, TAU)
        assert abs(float(num / den) - target) < 0.02, (lag, float(num / den), target)


def test_sample_reference_perturbation_is_seeded_and_validates() -> None:
    cov = _posterior_like_covariance(8, torch.Generator().manual_seed(3))
    a = sample_reference_perturbation(cov, HZ, torch.Generator().manual_seed(7))
    b = sample_reference_perturbation(cov, HZ, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        sample_reference_perturbation(cov[:, :6, :6], HZ, torch.Generator())
    with pytest.raises(ValueError):
        sample_reference_perturbation(cov, 0.0, torch.Generator())
    varying = cov.clone()
    varying[3, 6, 6] *= 1.5                                                 # the clip has ONE scale variance
    with pytest.raises(ValueError):
        sample_reference_perturbation(varying, HZ, torch.Generator())
    known_scale = cov.clone()
    known_scale[:, 6, :] = 0.0
    known_scale[:, :, 6] = 0.0
    eps = sample_reference_perturbation(known_scale, HZ, torch.Generator().manual_seed(8))
    assert bool(torch.isfinite(eps).all()) and torch.equal(eps[:, 6], torch.zeros(8, dtype=DT))
    assert float(eps[:, :6].abs().max()) > 0.0


def test_perturb_reference_applies_left_exponential_and_scale() -> None:
    gen = torch.Generator().manual_seed(4)
    T = 12
    poses = se3_exp(torch.randn(T, 6, dtype=DT, generator=gen))
    ref = PolicyReference(q_ref=torch.zeros(T, 3, dtype=DT), obj_pose=poses, log_scale=0.1,
                          contact_plan=torch.zeros(T, 2, dtype=torch.bool), covariance=None, hz=HZ)
    eps = 0.05 * torch.randn(T, 7, dtype=DT, generator=gen)
    eps[:, 6] = 0.03                                                        # one clip-level scale draw

    out = perturb_reference(ref, eps)

    assert torch.allclose(out.obj_pose, se3_exp(eps[:, :6]) @ poses, atol=1e-12)
    assert torch.allclose(se3_log(out.obj_pose @ se3_inv(poses)), eps[:, :6], atol=1e-10)
    assert out.log_scale == pytest.approx(0.13, abs=1e-15)
    assert out.q_ref is ref.q_ref and out.contact_plan is ref.contact_plan and out.hz == ref.hz
    assert ref.log_scale == 0.1 and torch.equal(ref.obj_pose, poses)       # input untouched
    with pytest.raises(ValueError):
        perturb_reference(ref, eps[:-1])
    per_frame_scale = eps.clone()
    per_frame_scale[5, 6] += 0.01                                           # not one draw per clip
    with pytest.raises(ValueError):
        perturb_reference(ref, per_frame_scale)
