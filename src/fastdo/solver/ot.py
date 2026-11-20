"""Log-domain unbalanced entropic optimal transport (Sinkhorn), batched.

The fused CUDA kernels (`ot_fused`) are imported inside `sinkhorn_unbalanced_potentials`, and only when a call opts
in (`fused=True`): `ot_fused` imports triton at import time, which no importer of `fastdo.solver` should pay for.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# A Sinkhorn parameter (eps or a marginal penalty rho): one Python float for the whole batch, or a (B,)
# tensor with one value per batch element (frames of different clips carry different
# diameter^2 normalizations).
Param = float | Tensor


def _tau(rho: Param, eps: Param) -> Param:
    if isinstance(rho, Tensor):
        rho_t, eps_t = rho, (eps if isinstance(eps, Tensor) else torch.full_like(rho, eps))
    elif isinstance(eps, Tensor):
        rho_t, eps_t = torch.full_like(eps, rho), eps
    else:
        return 1.0 if math.isinf(rho) else rho / (rho + eps)
    return torch.where(torch.isinf(rho_t), torch.ones_like(rho_t), rho_t / (rho_t + eps_t))


def _per_batch(x: Param, like: Tensor, ndim: int) -> Param:
    """`x` ready to broadcast against a (B, ...) tensor of `ndim` dims in `like`'s dtype: a float passes
    through unchanged (exactly the expressions of the scalar-only version), a (B,) tensor is cast (computed in float64
    first, so a cast float32 value equals what the scalar path multiplies by) and reshaped."""
    if not isinstance(x, Tensor):
        return x
    return x.to(like.dtype).reshape(-1, *([1] * (ndim - 1)))


_LOG_ZERO = -1e30   # sentinel for log(0): finite (no NaN from -inf +/- inf), but dwarfs every (g-C)/eps
                    # term the solver ever produces (bounded by O(max C / eps), at most ~1e5 even at the
                    # finest annealing level), so it still zeroes exp(...) cleanly. Plain
                    # `log(b.clamp_min(eps))` is not negative enough for this: at the fine level eps is
                    # as small as ~1e-6 in diameter^2 units, so (g-C)/eps can reach the same ~1e5
                    # magnitude that a merely-small clamp (e.g. 1e-30, log ~ -69) would fail to suppress.


def _safe_log_mass(x: Tensor) -> Tensor:
    """log(x) with exact zeros mapped to `_LOG_ZERO` instead of computed via a clamp-then-log.

    Used for a masked marginal (e.g. the visibility-aware template measure): an excluded
    entry's mass is exactly 0, and must be excluded from the log-domain Sinkhorn updates without ever
    producing NaN (-inf + inf, or 0 * inf) and without the exclusion leaking back in as real mass
    through an insufficiently negative clamp. Entries with `x > 0` behave exactly like plain
    `torch.log(x.clamp_min(1e-300))` (unchanged from before this masking existed).
    """
    return torch.where(x > 0, torch.log(x.clamp_min(1e-300)), torch.full_like(x, _LOG_ZERO))


def sinkhorn_unbalanced_potentials(C: Tensor, a: Tensor, b: Tensor, eps: Param, rho_a: Param, rho_b: Param,
                                   n_iter: int = 50, init: tuple[Tensor, Tensor] | None = None,
                                   fused: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """Log-domain Sinkhorn returning the plan and its dual potentials (f, g), optionally warm-started.

    Minimizes <C,P> + eps KL(P|a x b) + rho_a KL(P1|a) + rho_b KL(P^T1|b) over P (B,N,M).
    rho = inf makes that marginal a hard constraint (balanced on that side). `init`, when given, is a
    prior (f, g) pair (e.g. from a nearby pose/eps) to warm-start the iteration instead of the zero
    potentials a cold start would use; far fewer iterations are then needed to converge (the dual
    ascent is a contraction, so a close warm start needs only a handful of corrective iterations; verified
    to match a cold long run within 1e-6).

    `b` may contain exact zeros (the visibility-aware template measure masks out the mesh samples
    not visible from the current pose this way); `_safe_log_mass` maps those to a large-but-finite
    negative log-mass rather than `-inf`, so those columns of the plan converge to (numerically) zero
    without ever producing NaN.

    `eps`, `rho_a`, `rho_b` may each be a (B,) tensor, one value per batch element (the packed
    clips); a float keeps the original scalar expressions. `fused=True` runs the half-steps and the final plan
    as fused one-pass kernels (`ot_fused`, the same expressions up to float round-off) when they apply (CUDA, no
    autograd). It is opt-in until the kernels pass their CUDA tests and their launch memory,
    which the CUDA driver allocates outside the caching allocator, is measured (see `ot_fused`).
    """
    log_a = torch.log(a.clamp_min(1e-30))
    log_b = _safe_log_mass(b)
    if init is None:
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)
    else:
        f, g = init
    ta, tb = _tau(rho_a, eps), _tau(rho_b, eps)
    coef_a, coef_b = _per_batch(-ta * eps, C, 2), _per_batch(-tb * eps, C, 2)
    if fused:
        from fastdo.solver import ot_fused  # lazy: it imports triton (module docstring)

        if ot_fused.fused_available(C, log_a, log_b, f, g):
            for _ in range(n_iter):
                f = coef_a * ot_fused.lse_rows(C, g, log_b, eps)
                g = coef_b * ot_fused.lse_cols(C, f, log_a, eps)
            return ot_fused.plan_from_potentials(C, f, g, log_a, log_b, eps), f, g
    eps_c = _per_batch(eps, C, 3)
    for _ in range(n_iter):
        f = coef_a * torch.logsumexp((g[:, None, :] - C) / eps_c + log_b[:, None, :], dim=2)
        g = coef_b * torch.logsumexp((f[:, :, None] - C) / eps_c + log_a[:, :, None], dim=1)
    P = torch.exp((f[:, :, None] + g[:, None, :] - C) / eps_c + log_a[:, :, None] + log_b[:, None, :])
    return P, f, g


def sinkhorn_unbalanced(C: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float,
                        n_iter: int = 50) -> Tensor:
    """Transport plan P (B,N,M) minimizing <C,P> + eps KL(P|a x b) + rho_a KL(P1|a) + rho_b KL(P^T1|b).

    rho = inf makes that marginal a hard constraint (balanced on that side). Thin wrapper around
    `sinkhorn_unbalanced_potentials` (cold start) that drops the dual potentials.
    """
    P, _f, _g = sinkhorn_unbalanced_potentials(C, a, b, eps, rho_a, rho_b, n_iter)
    return P


_EXP_FAST_MIN = -700.0     # torch's CPU exp is fast on [-700, 0]; below about -708 it takes a ~15-300x slower path
_EXP_ZERO_BELOW = -746.0   # torch.exp(z) is exactly +0.0 for every z < -746 (checked densely on this torch build)


def _exp_exact(z: Tensor) -> Tensor:
    """`torch.exp(z)` bit for bit, without the slow underflow path: exp on `max(z, -700)`; entries below -700
    set to +0, and the rare entries in [-746, -700) (tiny / subnormal results) recomputed by `torch.exp` itself.
    NaN passes through `clamp` and `exp` unchanged. Bit identity rests on `torch.exp` being elementwise (its result
    for an entry does not depend on the tensor's length or the entry's position), which
    `tests/solver/test_ot_fastexp_393.py` checks."""
    e = torch.exp(z.clamp(min=_EXP_FAST_MIN))
    low = z < _EXP_FAST_MIN
    if bool(low.any()):
        mid = low & (z >= _EXP_ZERO_BELOW)
        e.masked_fill_(low, 0.0)
        if bool(mid.any()):
            e[mid] = torch.exp(z[mid])
    return e


def _logsumexp_exact(x: Tensor, dim: int) -> Tensor:
    """`torch.logsumexp(x, dim)` bit for bit for FINITE `x` (ATen: amax, exp(x - m), sum, log, + m) via `_exp_exact`.
    ATen also zeroes an infinite max; that case cannot arise here (`_LOG_ZERO` is finite), so it is not replicated."""
    m = x.amax(dim=dim, keepdim=True)
    return _exp_exact(x - m).sum(dim=dim).log_().add_(m.squeeze(dim))


def sinkhorn_unbalanced_uniform_b(C: Tensor, a: Tensor, row_valid: Tensor, eps: float, rho_a: float, rho_b: float,
                                  n_iter: int) -> Tensor:
    """`sinkhorn_unbalanced` for a uniform `b = 1/M` with invalid rows made inert.

    Scaled potentials `fe = f / eps`, `ge = g / eps` from zero, with `C / eps` and the log masses merged into two
    constants once per call: `Cb = C / eps + log M`, `Ca = C / eps - log_a`. The iteration order is that of
    `sinkhorn_unbalanced_potentials`. `log_a` is `log(a)` (clamped at 1e-30) on `row_valid` rows and -1e30 elsewhere,
    so an invalid row contributes exactly 0 to every column update and to the plan. `C (B,N,M)`, `a (B,N)`,
    `row_valid (B,N)` bool -> plan `(B,N,M)` on `C.device` in `C.dtype`.

    On CPU the column half-step and the plan go through `_exp_exact` / `_logsumexp_exact`, which return
    `torch.exp` / `torch.logsumexp` bit for bit but skip the exp underflow slow path (about a third of these entries
    sit at or below -708): byte-identical retarget qpos, CPU time about 0.9x. The row half-step never underflows and
    keeps `torch.logsumexp`. Other devices, and a call that records autograd (`_exp_exact` writes in place), run
    the plain torch expressions.
    """
    M = C.shape[2]
    log_m = math.log(M)
    log_a = torch.where(row_valid, torch.log(a.clamp_min(1e-30)), torch.full_like(a, _LOG_ZERO))
    Cb = C / eps + log_m
    Ca = C / eps - log_a[:, :, None]
    ta, tb = _tau(rho_a, eps), _tau(rho_b, eps)
    fe = torch.zeros_like(a)
    ge = torch.zeros(C.shape[0], M, dtype=C.dtype, device=C.device)
    fast = C.device.type == "cpu" and not (torch.is_grad_enabled() and (C.requires_grad or a.requires_grad))
    exp, lse_cols = (_exp_exact, _logsumexp_exact) if fast else (torch.exp, torch.logsumexp)
    for _ in range(n_iter):
        fe = -ta * torch.logsumexp(ge[:, None, :] - Cb, dim=2)
        ge = -tb * lse_cols(fe[:, :, None] - Ca, dim=1)
    return exp(fe[:, :, None] + ge[:, None, :] - Ca - log_m)


def _sinkhorn_unbalanced_uniform_b_torch(C: Tensor, a: Tensor, row_valid: Tensor, eps: float, rho_a: float,
                                         rho_b: float, n_iter: int) -> Tensor:
    """`sinkhorn_unbalanced_uniform_b` without the fast exp (plain `torch.exp` / `torch.logsumexp`): the
    bit-identity oracle of `tests/solver/test_ot_fastexp_393.py`, never called by production code."""
    M = C.shape[2]
    log_m = math.log(M)
    log_a = torch.where(row_valid, torch.log(a.clamp_min(1e-30)), torch.full_like(a, _LOG_ZERO))
    Cb = C / eps + log_m
    Ca = C / eps - log_a[:, :, None]
    ta, tb = _tau(rho_a, eps), _tau(rho_b, eps)
    fe = torch.zeros_like(a)
    ge = torch.zeros(C.shape[0], M, dtype=C.dtype, device=C.device)
    for _ in range(n_iter):
        fe = -ta * torch.logsumexp(ge[:, None, :] - Cb, dim=2)
        ge = -tb * torch.logsumexp(fe[:, :, None] - Ca, dim=1)
    return torch.exp(fe[:, :, None] + ge[:, None, :] - Ca - log_m)


_SCALING_RANGE = 1e-280     # a stabilized row/column sum outside [1e-280, 1e280] (or non-finite) -> log-domain re-solve
_ABSORB_DRIFT = 200.0       # re-absorb once a potential moves more than this from the absorbed one
_SINKHORN_STATS = {"calls": 0, "absorptions": 0, "fallbacks": 0}   # diagnostics, never read by the solver


def _sums_in_range(x: Tensor) -> Tensor:
    """`(B, K)` -> `(B,)` True where every entry lies in `[_SCALING_RANGE, 1 / _SCALING_RANGE]` (False on NaN)."""
    return ((x >= _SCALING_RANGE) & (x <= 1.0 / _SCALING_RANGE)).all(dim=1)


def sinkhorn_unbalanced_uniform_b_scaling(C: Tensor, a: Tensor, row_valid: Tensor, eps: float, rho_a: float,
                                          rho_b: float, n_iter: int) -> Tensor:
    """`sinkhorn_unbalanced_uniform_b`'s `n_iter` iterations evaluated in the stabilized scaling domain with
    absorption: the same potentials up to float round-off, NOT bit-identical (opt-in,
    `RetargetConfig.sinkhorn`).

    With `S = C / eps`, `la = log a` (valid rows; -1e30 elsewhere), the iteration is that of the log domain,

        fe_i = -ta * (LSE_j(ge_j - S_ij) - log M),    ge_j = -tb * LSE_i(fe_i - S_ij + la_i),

    with each LSE rewritten exactly around absorbed potentials `(alpha, beta)`: the row kernel
    `Kr_ij = exp(beta_j - S_ij + nr_i)` and the column kernel `Kc_ij = exp(alpha_i - S_ij + la_i + nc_j)`, with
    the normalizers `nr_i = -LSE_j(beta_j - S_ij)` and `nc_j = -LSE_i(alpha_i - S_ij + la_i)` (unit row / column
    sums), give

        LSE_j(ge_j - S_ij) = -nr_i + log sum_j exp(ge_j - beta_j) Kr_ij
        LSE_i(fe_i - S_ij + la_i) = -nc_j + log sum_i exp(fe_i - alpha_i) Kc_ij.

    An iteration is then two batched matrix-vector products instead of two log-sum-exp passes over `(B, N, M)`, and
    every sum lies in `[exp(-200), exp(200)]` while no potential has moved more than 200 from its absorbed value; a
    batch element that moves further is re-absorbed (two LSE and two `exp` passes over its kernels). The first
    absorption is at the zero start. The plan is the log-domain expression `exp(fe + ge - S + la - log M)`. A batch
    element whose sums still leave `[1e-280, 1e280]` is re-solved by `sinkhorn_unbalanced_uniform_b` itself.
    Shapes and conventions as there.
    """
    M = C.shape[2]
    log_m = math.log(M)
    log_a = torch.where(row_valid, torch.log(a.clamp_min(1e-30)), torch.full_like(a, _LOG_ZERO))
    S = C / eps
    Ca = S - log_a[:, :, None]
    ta, tb = _tau(rho_a, eps), _tau(rho_b, eps)
    fe = torch.zeros_like(a)
    ge = torch.zeros(C.shape[0], M, dtype=C.dtype, device=C.device)
    alpha, beta = fe.clone(), ge.clone()
    Kr, Kc = torch.empty_like(S), torch.empty_like(S)
    nr, nc = torch.empty_like(fe), torch.empty_like(ge)
    zero = torch.zeros((), dtype=C.dtype, device=C.device)

    def absorb(idx: Tensor) -> None:
        alpha[idx], beta[idx] = fe[idx], ge[idx]
        row_exp = beta[idx][:, None, :] - S[idx]
        col_exp = alpha[idx][:, :, None] - Ca[idx]
        nr[idx] = -torch.logsumexp(row_exp, dim=2)
        nc[idx] = -torch.logsumexp(col_exp, dim=1)
        Kr[idx] = torch.exp(row_exp + nr[idx][:, :, None])
        Kc[idx] = torch.exp(col_exp + nc[idx][:, None, :])
        _SINKHORN_STATS["absorptions"] += int(idx.sum())

    _SINKHORN_STATS["calls"] += 1
    bad = torch.zeros(C.shape[0], dtype=torch.bool, device=C.device)
    absorb(~bad)
    for _ in range(n_iter):
        row = (Kr @ torch.exp(ge - beta)[:, :, None]).squeeze(2)                # (B,N)
        fe = -ta * (torch.log(row) - nr - log_m)
        u = torch.where(row_valid, torch.exp(fe - alpha), zero)
        col = (u[:, None, :] @ Kc).squeeze(1)                                   # (B,M)
        ge = -tb * (torch.log(col) - nc)
        bad |= ~(_sums_in_range(row) & _sums_in_range(col))
        drift = torch.maximum(torch.where(row_valid, (fe - alpha).abs(), zero).amax(dim=1),
                              (ge - beta).abs().amax(dim=1))
        redo = (drift > _ABSORB_DRIFT) & ~bad
        if bool(redo.any()):
            absorb(redo)
    plan = torch.exp(fe[:, :, None] + ge[:, None, :] - Ca - log_m)
    if bool(bad.any()):
        _SINKHORN_STATS["fallbacks"] += int(bad.sum())
        plan[bad] = sinkhorn_unbalanced_uniform_b(C[bad], a[bad], row_valid[bad], eps, rho_a, rho_b, n_iter)
    return plan


def plan_barycenters(P: Tensor, Y: Tensor) -> tuple[Tensor, Tensor]:
    """Per-source barycenter of targets under P: returns (ybar (B,N,3), mass w (B,N))."""
    w = P.sum(dim=2)
    ybar = (P @ Y) / w.clamp_min(1e-12)[..., None]
    return ybar, w


def ot_cost(C: Tensor, P: Tensor) -> Tensor:
    """Transport cost <C,P> per batch element."""
    return (C * P).sum(dim=(1, 2))


def _kl(p: Tensor, q: Tensor) -> Tensor:
    """Generalized KL(p|q) = sum p log(p/q) - p + q, summed over the last dims, per batch element."""
    ratio = torch.where(p > 0, p / q.clamp_min(1e-300), torch.ones_like(p))
    return (torch.where(p > 0, p * torch.log(ratio), torch.zeros_like(p)) - p + q).flatten(1).sum(1)


def ot_objective(C: Tensor, P: Tensor, a: Tensor, b: Tensor, eps: float, rho_a: float, rho_b: float) -> Tensor:
    """Full unbalanced entropic objective <C,P> + eps KL(P|a x b) + rho_a KL(P1|a) + rho_b KL(P^T1|b), per batch."""
    obj = ot_cost(C, P) + eps * _kl(P, a[:, :, None] * b[:, None, :])
    if not math.isinf(rho_a):
        obj = obj + rho_a * _kl(P.sum(2), a)
    if not math.isinf(rho_b):
        obj = obj + rho_b * _kl(P.sum(1), b)
    return obj
