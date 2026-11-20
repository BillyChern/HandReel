"""Exact (interpolating) cubic B-spline resampling of a 2D grid, with an analytic gradient.

Used internally by `silhouette_term` to sample the per-pixel distance-transform field at the
continuous, sub-pixel coordinates projected surface points land on. `torch.nn.functional.grid_sample`
offers "bilinear" (C0: its derivative has a real jump at every integer pixel boundary) and "bicubic"
(cubic convolution: only C1, its *second* derivative still jumps at pixel boundaries). A projected
point that happens to land a small fraction of a pixel from such a boundary makes a finite-difference
check straddle it, and the second-derivative jump alone is enough to disagree with the analytic
Jacobian by an O(1) amount there (verified empirically against this module's test fixtures). A true
cubic B-spline is C2 continuous, which removes that failure mode; reproducing the exact sample values
requires a prefiltering deconvolution step first (Unser, "Splines: A Perfect Fit for Signal/Image
Processing", 1999), implemented here via a tridiagonal (Thomas-algorithm) solve per axis.

Boundary convention (binding, one convention used consistently by both halves below): whole-sample
mirror symmetry, i.e. the coefficient sequence is treated as if extended by reflection *about* the
first/last sample itself (index -1 aliases to index 1, index N aliases to index N-2, and so on with
period 2*(N-1)). `_prefilter_1d` solves the tridiagonal system for exactly this boundary (its first
and last rows encode c[-1] = c[1] and c[N] = c[N-2]); `bspline_sample_with_grad` evaluates with the
matching mirrored index, not a clamped one -- clamping is a *different* (and inconsistent) boundary
condition from what the prefilter solved for, and reproduces the border samples incorrectly (e.g. a
[0, 1, 2, 3] signal's interpolant would read back -0.3, not 0.0, at the left edge under clamped
evaluation against a mirror-boundary prefilter).

Because it is not shape-preserving -- a cubic spline overshoots/undershoots slightly near a sharp
kink in the underlying samples, trading exact shape fidelity for the C2 smoothness above -- resampled
values a cell or two inside a sharp feature (e.g. just inside a distance transform's mask boundary,
where the field's slope has a real jump) can differ from the raw per-pixel value by a small amount.
This is a deliberate, documented trade-off for `silhouette_term`'s residual (see its docstring).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _prefilter_1d(s: Tensor, dim: int) -> Tensor:
    """Cubic B-spline prefilter along `dim`: samples -> coefficients (mirror boundary)."""
    s = s.movedim(dim, -1)
    n = s.shape[-1]
    if n == 1:
        return s.movedim(-1, dim)                                 # a single sample needs no filtering
    sub = torch.ones(n, dtype=s.dtype, device=s.device)
    diag = torch.full((n,), 4.0, dtype=s.dtype, device=s.device)
    sup = torch.ones(n, dtype=s.dtype, device=s.device)
    diag[0] = 2.0
    diag[-1] = 2.0
    sup[-1] = 0.0                                                # last row has no c[n] term
    rhs = 6.0 * s.clone()
    rhs[..., 0] = 3.0 * s[..., 0]
    rhs[..., -1] = 3.0 * s[..., -1]
    cp = torch.zeros(n, dtype=s.dtype, device=s.device)
    dp = torch.zeros_like(rhs)
    cp[0] = sup[0] / diag[0]
    dp[..., 0] = rhs[..., 0] / diag[0]
    for i in range(1, n):                                        # forward sweep (Thomas algorithm)
        pivot = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / pivot
        dp[..., i] = (rhs[..., i] - sub[i] * dp[..., i - 1]) / pivot
    out = torch.zeros_like(rhs)
    out[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):                                # back substitution
        out[..., i] = dp[..., i] - cp[i] * out[..., i + 1]
    return out.movedim(-1, dim)


def bspline_prefilter_2d(img: Tensor) -> Tensor:
    """Convert a (...,H,W) sample grid to separable cubic B-spline coefficients."""
    coef = _prefilter_1d(img, -1)
    coef = _prefilter_1d(coef, -2)
    return coef


def _basis(t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Uniform cubic B-spline weights for taps at floor(x)-1 .. floor(x)+2, offset t in [0,1)."""
    t2 = t * t
    t3 = t2 * t
    w0 = (1.0 - t) ** 3 / 6.0
    w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    w3 = t3 / 6.0
    return w0, w1, w2, w3


def _basis_deriv(t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """d/dt of `_basis`."""
    t2 = t * t
    d0 = -((1.0 - t) ** 2) / 2.0
    d1 = (3.0 * t2 - 4.0 * t) / 2.0
    d2 = (-3.0 * t2 + 2.0 * t + 1.0) / 2.0
    d3 = t2 / 2.0
    return d0, d1, d2, d3


def _mirror_index(idx: Tensor, n: int) -> Tensor:
    """Whole-sample mirror reflection of `idx` into [0, n), period 2*(n-1) -- see module docstring."""
    if n == 1:
        return torch.zeros_like(idx)
    period = 2 * (n - 1)
    idx = idx % period                                            # Python/torch floor-mod: always >= 0
    return torch.where(idx >= n, period - idx, idx)


def bspline_sample_with_grad(coef: Tensor, uv: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate the C2 cubic B-spline interpolant of `coef` (T,1,H,W) at pixel coords uv (T,n,2).

    Returns (value (T,n), d value/d uv (T,n,2)); out-of-grid taps use the mirrored index described
    in the module docstring (consistent with `_prefilter_1d`'s boundary), not a clamped one.
    """
    frames, _, height, width = coef.shape
    u, v = uv[..., 0], uv[..., 1]
    iu, iv = torch.floor(u), torch.floor(v)
    tu, tv = u - iu, v - iv
    wu, wv = torch.stack(_basis(tu), dim=-1), torch.stack(_basis(tv), dim=-1)
    du, dv = torch.stack(_basis_deriv(tu), dim=-1), torch.stack(_basis_deriv(tv), dim=-1)
    iu_l, iv_l = iu.long(), iv.long()
    frame_idx = torch.arange(frames, device=coef.device)[:, None].expand_as(u)
    val = torch.zeros_like(u)
    grad_u = torch.zeros_like(u)
    grad_v = torch.zeros_like(u)
    for a in range(4):
        row = _mirror_index(iv_l + a - 1, height)
        for b in range(4):
            col = _mirror_index(iu_l + b - 1, width)
            c = coef[frame_idx, 0, row, col]
            val = val + wv[..., a] * wu[..., b] * c
            grad_u = grad_u + wv[..., a] * du[..., b] * c
            grad_v = grad_v + dv[..., a] * wu[..., b] * c
    return val, torch.stack([grad_u, grad_v], dim=-1)
