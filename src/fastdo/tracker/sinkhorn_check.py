"""Truncated unbalanced-Sinkhorn gradient check.

The gradient of the entropic OT value `OT_eps(C) = min_P F(P; C)` is the converged plan `P*` (envelope theorem). At a
2-5 iteration budget from predicted duals that no longer holds exactly. For cost directions `E_k` this module compares
three estimators at the n-iterate `z_n = (f_n, g_n)` against the finite difference of the converged value:
- `detached`: `<P_n, E>`, duals and plan held constant (the plan's `detach_duals=True`);
- `unrolled`: `d/dt V_n(C + t E)` by autograd through the n iterations, with `V_n(C) = F(P_n(C); C)`;
- `implicit`: `<P_n, E> + (d_C grad_z D[E])^T (-Hess_zz D)^{-1} grad_z D` at `z_n` (Ablin, Peyre and Moreau 2020),
  with `D` the unbalanced dual restricted to rows `R = {a > 0}` and columns `S = {b > 0}`.

The init is a constant (not a function of C). The Sinkhorn loop updates f first, so `init[0]` is overwritten by the
first f half-step. Drive `"f"` therefore runs one g half-step from `init[0]` inside the differentiated map (`g0(C)`)
before the n iterations, so the network's `dual_f` is what the loop sees.

Scope: dense float64 CPU. The production fine level runs kNN Sinkhorn in float32; the training stage
re-verifies on that path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal, NamedTuple

import torch
from torch import Tensor

from fastdo.solver.ot import _safe_log_mass, _tau, ot_objective, sinkhorn_unbalanced_potentials

__all__ = [
    "ConvergedReference",
    "GradientEstimates",
    "converged_reference",
    "dual_gradients",
    "g_half_step",
    "implicit_cost_gradient",
    "implicit_value_gradient",
    "sinkhorn_iterate",
    "truncated_sinkhorn_gradient_check",
]

logger = logging.getLogger(__name__)

F64 = torch.float64
Drive = Literal["g", "f"]


class GradientEstimates(NamedTuple):
    detached: Tensor        # (K,) <P_n, E_k>: duals and plan held constant
    unrolled: Tensor        # (K,) d/dz V_n(C + z E) through the n iterations, init constant
    implicit: Tensor        # (K,) M2 closed form at the n-iterate
    fd_truncated: Tensor    # (K,) central FD of V_n (init constant): validates `unrolled`
    fd_converged: Tensor    # (K,) central FD of V_inf: the reference
    envelope: Tensor        # (K,) <P*, E_k> at the converged plan


class ConvergedReference(NamedTuple):
    """The converged state of one problem, computed once and shared by every `(init, n)` (checker minor M6)."""

    P: Tensor               # (1,N,M) converged plan
    f: Tensor               # (1,N)
    g: Tensor               # (1,M)
    envelope: Tensor        # (K,) <P*, E_k>
    fd_converged: Tensor    # (K,) central FD of the converged value


def _check_problem(C: Tensor, a: Tensor, b: Tensor, directions: Tensor) -> None:
    if C.dim() != 3 or C.shape[0] != 1:
        raise ValueError(f"C must have shape (1,N,M), got {tuple(C.shape)}")
    if tuple(a.shape) != (1, C.shape[1]) or tuple(b.shape) != (1, C.shape[2]):
        raise ValueError(f"a must be (1,{C.shape[1]}) and b (1,{C.shape[2]}), got {tuple(a.shape)}, {tuple(b.shape)}")
    if directions.dim() != 4 or tuple(directions.shape[1:]) != tuple(C.shape):
        raise ValueError(f"directions must have shape (K,{','.join(map(str, C.shape))}), got {tuple(directions.shape)}")
    if any(x.dtype != F64 for x in (C, a, b, directions)):
        raise ValueError("C, a, b and directions must be float64")


def g_half_step(C: Tensor, a: Tensor, f: Tensor, eps: float, rho_b: float) -> Tensor:
    """`g = -tau_b eps LSE_i((f_i - C_ij)/eps + log a_i)`: the solver's g update [ot.py], `(B,M)`."""
    tau_b = float(_tau(rho_b, eps))
    log_a = torch.log(a.clamp_min(1e-30))
    return -tau_b * eps * torch.logsumexp((f[:, :, None] - C) / eps + log_a[:, :, None], dim=1)


def sinkhorn_iterate(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float, n_iter: int,
                     init: tuple[Tensor, Tensor], drive: Drive = "g") -> tuple[Tensor, Tensor, Tensor]:
    """`(P_n, f_n, g_n)`: n solver iterations from `init` (drive "g"), or from `(f0, g_half_step(f0))` (drive "f")."""
    f0, g0 = init
    if drive == "f":
        g0 = g_half_step(C, a, f0, eps, rho_b)
    elif drive != "g":
        raise ValueError(f"drive must be 'g' or 'f', got {drive!r}")
    return sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter, init=(f0, g0))


def _value(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float, n_iter: int,
           init: tuple[Tensor, Tensor], drive: Drive) -> Tensor:
    P, _, _ = sinkhorn_iterate(C, a, b, eps, rho_a, rho_b, n_iter, init, drive)
    return ot_objective(C, P, a, b, eps, rho_a, rho_b)[0]


def _central_fd(value: Callable[[Tensor], Tensor], C: Tensor, directions: Tensor, h: float) -> Tensor:
    return torch.stack([(value(C + h * E) - value(C - h * E)) / (2 * h) for E in directions])


def _contract(G: Tensor, directions: Tensor) -> Tensor:
    return (G[None] * directions).sum(dim=(1, 2, 3))


def converged_reference(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float,
                        directions: Tensor, h: float = 1e-4, n_converged: int = 5000,
                        n_converged_fd: int = 3000) -> ConvergedReference:
    """`(P*, f*, g*)` from `n_converged` cold iterations, `<P*, E_k>` and the central FD (step `h`) of the
    converged value, each side warm-started from `(f*, g*)` for `n_converged_fd` iterations."""
    _check_problem(C, a, b, directions)
    with torch.no_grad():
        P, f, g = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_converged)
        fd = _central_fd(lambda Cz: _value(Cz, a, b, eps, rho_a, rho_b, n_converged_fd, (f, g), "g"),
                         C, directions, h)
    return ConvergedReference(P=P, f=f, g=g, envelope=_contract(P, directions), fd_converged=fd)


def dual_gradients(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float, f: Tensor,
                   g: Tensor) -> tuple[Tensor, Tensor]:
    """`(grad_f D (1,N), grad_g D (1,M))` = `(a e^{-f/rho_a} - P1, b e^{-g/rho_b} - P^T 1)`, in the log domain.

    Both are differences of two positive masses `u - v = -u expm1(log v - log u)`, evaluated from log masses so that
    fine-level marginals that underflow stay exact. Rows off `R = {a > 0}` and columns off `S = {b > 0}` are 0.
    """
    log_a, log_b = torch.log(a.clamp_min(1e-30)), _safe_log_mass(b)
    log_r = log_a + f / eps + torch.logsumexp((g[:, None, :] - C) / eps + log_b[:, None, :], dim=2)
    log_c = log_b + g / eps + torch.logsumexp((f[:, :, None] - C) / eps + log_a[:, :, None], dim=1)
    log_ea, log_eb = log_a - f / rho_a, log_b - g / rho_b
    grad_f = -torch.exp(log_ea) * torch.expm1(log_r - log_ea)
    grad_g = -torch.exp(log_eb) * torch.expm1(log_c - log_eb)
    return torch.where(a > 0, grad_f, torch.zeros_like(grad_f)), torch.where(b > 0, grad_g, torch.zeros_like(grad_g))


def implicit_cost_gradient(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float, P: Tensor,
                           f: Tensor, g: Tensor) -> Tensor:
    """`(1,N,M)` implicit OT-value gradient at `(P, f, g)`: `P * (1 + (u_i + w_j)/eps)` (checker I5).

    `(u_R, w_S) = solve(-Hess D, grad_z D)` on rows `R = {a > 0}` and columns `S = {b > 0}` (0 elsewhere), with
    `-Hess D = [[diag(ea/rho_a + r/eps), P_RS/eps], [P_RS^T/eps, diag(eb/rho_b + c/eps)]]`, `r`, `c` the marginals of
    `P_RS`, `ea = exp(log a - f/rho_a)` and `eb = exp(log b - g/rho_b)`. Dense solve: report-scale problems only.
    """
    if C.shape[0] != 1:
        raise ValueError(f"batch size must be 1, got C {tuple(C.shape)}")
    R, S = a[0] > 0, b[0] > 0
    P0 = P[0]
    P_RS = P0[R][:, S]
    r, c = P_RS.sum(1), P_RS.sum(0)
    ea = torch.exp(torch.log(a[0, R]) - f[0, R] / rho_a)
    eb = torch.exp(torch.log(b[0, S]) - g[0, S] / rho_b)
    n_R = r.shape[0]
    top = torch.cat([torch.diag(ea / rho_a + r / eps), P_RS / eps], dim=1)
    bottom = torch.cat([P_RS.T / eps, torch.diag(eb / rho_b + c / eps)], dim=1)
    sol = torch.linalg.solve(torch.cat([top, bottom], dim=0), torch.cat([ea - r, eb - c]))
    u = torch.zeros(P0.shape[0], dtype=P0.dtype, device=P0.device).masked_scatter(R, sol[:n_R])
    w = torch.zeros(P0.shape[1], dtype=P0.dtype, device=P0.device).masked_scatter(S, sol[n_R:])
    return (P0 * (1.0 + (u[:, None] + w[None, :]) / eps))[None]


def implicit_value_gradient(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float,
                            P: Tensor, f: Tensor, g: Tensor, directions: Tensor) -> Tensor:
    """`(K,)` `G_imp[E_k]`: `implicit_cost_gradient` contracted with `directions (K,1,N,M)`."""
    return _contract(implicit_cost_gradient(C, a, b, eps, rho_a, rho_b, P, f, g), directions)


def truncated_sinkhorn_gradient_check(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float,
                                      n_iter: int, init: tuple[Tensor, Tensor], directions: Tensor,
                                      h: float = 1e-4, n_converged: int = 5000, n_converged_fd: int = 3000,
                                      drive: Drive = "g", reference: ConvergedReference | None = None
                                      ) -> GradientEstimates:
    """The three estimators at the n-iterate and their two FD references.

    `C (1,N,M)`, `a (1,N)`, `b (1,M)`, `directions (K,1,N,M)`, all float64. `reference`, when given, is the problem's
    `converged_reference` (it must come from the same `C, a, b, eps, rho, directions, h`); otherwise it is computed.
    """
    _check_problem(C, a, b, directions)
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if reference is None:
        reference = converged_reference(C, a, b, eps, rho_a, rho_b, directions, h, n_converged, n_converged_fd)
    elif (tuple(reference.P.shape) != tuple(C.shape) or tuple(reference.envelope.shape) != (directions.shape[0],)
          or tuple(reference.fd_converged.shape) != (directions.shape[0],)):
        raise ValueError(f"reference does not match the problem: P {tuple(reference.P.shape)} vs C {tuple(C.shape)}, "
                         f"envelope {tuple(reference.envelope.shape)} vs K = {directions.shape[0]}")
    const = (init[0].detach(), init[1].detach())
    with torch.no_grad():
        P, f, g = sinkhorn_iterate(C, a, b, eps, rho_a, rho_b, n_iter, const, drive)
        detached = _contract(P, directions)
        fd_truncated = _central_fd(lambda Cz: _value(Cz, a, b, eps, rho_a, rho_b, n_iter, const, drive),
                                   C, directions, h)
        implicit = implicit_value_gradient(C, a, b, eps, rho_a, rho_b, P, f, g, directions)
    z = torch.zeros(directions.shape[0], dtype=F64, requires_grad=True)
    Cz = C + (z[:, None, None, None] * directions).sum(0)
    (unrolled,) = torch.autograd.grad(_value(Cz, a, b, eps, rho_a, rho_b, n_iter, const, drive), z)
    logger.debug("truncated_sinkhorn_gradient_check: N=%d M=%d K=%d n_iter=%d drive=%s", C.shape[1], C.shape[2],
                 directions.shape[0], n_iter, drive)
    return GradientEstimates(detached=detached, unrolled=unrolled, implicit=implicit, fd_truncated=fd_truncated,
                             fd_converged=reference.fd_converged, envelope=reference.envelope)
