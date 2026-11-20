"""Anisotropic similarity registration.

Measured: a generated (SAM 3D) template's proportions are samples, not measurements: two seeds of one window
move the sorted principal extents by a median |dlog| of 0.07 / 0.09 / 0.18, while the solver's one scale moves by
0.031. One scale cannot absorb an aspect-ratio error, so `SolverConfig.anisotropic_scale` replaces it with a
body-frame diagonal stretch in the template's own principal frame:

    p_k = R_k A (y - c) + t_k,     A = U diag(exp l) U^T,     l in R^3 per clip.

`U` (columns u_i, descending eigenvalue) and `c` are the principal axes and centroid of the clip's template samples,
fixed per clip (`principal_frame`). l is estimated jointly with the per-frame poses; l_1 = l_2 = l_3 = log s is today's
Sim(3) with c absorbed into t. Everything downstream of the transform stays exact:

- points p = R A (y - c) + t, normals n' = normalize(R A^-1 n) (inverse transpose of a symmetric map);
- plan moments: pbar = R A (B - c) + t with B the canonical barycenter, so no new cache;
- dp/dl_i = sigma_i (u_i^T (y - c)) R u_i = (w_i^T (p - t)) w_i with w_i = R u_i (`stretch_jacobian`): like today's
  scale column (p - t) it needs only the transformed point and the pose, so every term builds it from what it has;
- the prior E = lam * sum_i (l_i - mean l)^2 = lam * l^T P l, P = I - 11^T/3, is the residual sqrt(lam) P l with
  Jacobian sqrt(lam) P (`prior_residuals`): it adds lam P to the 3x3 border block and lam P l to its gradient.

Pose convention at the boundary. Internally t is where the CENTROID c goes (the design's model). `batch.prepare_batch`
converts the caller's isotropic init (p = s0 R y + t0) exactly with t = t0 + s0 R c, and `api` converts the result
back to the isotropic convention with the geometric-mean scale s = exp(mean l): t_out = t - s R c. So the reported
`poses`/`scale` keep today's meaning (p = s R y + t_out on the isotropic part), and the exact object is
p = s R Y' + t_out with Y' = `corrected_template_points(y, solution)` = c + U diag(exp(l - mean l)) U^T (y - c), the
template's proportions corrected about its centroid (volume-preserving, the volume lives in s). Inside the solve the
centroid-referenced t is what LM's median-depth guard (`LMConfig.min_median_depth`) reads: the centroid's depth.

Not done here (follow-ups): the design's Fisher diagnostic (the 3x3 Schur block of l, exposed in solve.json), the full
6-DOF SPD stretch, the point-to-plane residual under a stretch, and the contact / retarget / QC-diameter consumers,
which still read the raw template with `scale` (they should go through `corrected_template_points`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = ["ANISO_DOF", "Stretch", "corrected_template_normals", "corrected_template_points", "isotropy_projector",
           "principal_frame", "prior_residuals", "stretch_jacobian"]

ANISO_DOF = 3   # per-clip border columns with the anisotropic scale on (today: 1, the log-scale)


def principal_frame(points: Tensor) -> tuple[Tensor, Tensor]:
    """`(U (3,3), c (3,))` float64 on the CPU: the principal axes (columns, ordered by DESCENDING eigenvalue) and the
    centroid of template samples `points` (M,3), which are area-uniform surface samples, so the plain mean and
    covariance are the design's area-weighted ones.

    Deterministic: the covariance is formed and diagonalised in float64 on the CPU; each of u_1, u_2 is signed so its
    largest-magnitude component is positive, and u_3 = u_1 x u_2 makes U a rotation. The signs never change the model
    (A = U diag U^T only sees u_i u_i^T); they only make the reported `shape_axes` reproducible. A template with
    (near-)equal principal extents, a cube or a sphere, has no preferred frame: U is then whatever LAPACK returns for
    that covariance (deterministic on one build), and the stretch lives in that frame."""
    pts = points.detach().to("cpu", torch.float64)
    c = pts.mean(0)
    X = pts - c
    cov = X.T @ X / max(pts.shape[0], 1)
    _evals, evecs = torch.linalg.eigh(cov)                              # ascending
    U = evecs.flip(-1).clone()                                          # descending
    for i in range(2):
        j = int(U[:, i].abs().argmax())
        if float(U[j, i]) < 0.0:
            U[:, i] = -U[:, i]
    U[:, 2] = torch.linalg.cross(U[:, 0], U[:, 1])
    return U, c


@dataclass(frozen=True)
class Stretch:
    """The per-frame (T, ...) stretch of the frames being evaluated: `axes` U (T,3,3) (columns u_i), `centroid` c (T,3)
    and `log_stretch` l (T,3), each frame carrying its clip's values. A = U diag(exp l) U^T."""

    axes: Tensor
    centroid: Tensor
    log_stretch: Tensor

    def take(self, rows: Tensor | slice) -> Stretch:
        """The stretch of frames `rows` (a slice view, or an `index_select` of a device index)."""
        def pick(x: Tensor) -> Tensor:
            return x[rows] if isinstance(rows, slice) else x.index_select(0, rows)
        return Stretch(axes=pick(self.axes), centroid=pick(self.centroid), log_stretch=pick(self.log_stretch))

    def matrix(self, inverse: bool = False) -> Tensor:
        """A (T,3,3), or A^-1 = U diag(exp(-l)) U^T with `inverse`."""
        sigma = (-self.log_stretch if inverse else self.log_stretch).exp()
        return (self.axes * sigma[:, None, :]) @ self.axes.transpose(-1, -2)

    def apply(self, Y: Tensor) -> Tensor:
        """A (y - c) for template points `Y` (M,3) or per-frame (T,M,3) -> (T,M,3)."""
        if Y.dim() == 2:
            Y = Y.unsqueeze(0)
        return torch.einsum("tij,tmj->tmi", self.matrix(), Y - self.centroid[:, None, :])

    def apply_normals(self, normals: Tensor) -> Tensor:
        """Unit normals of the stretched surface, normalize(A^-1 n), for `normals` (M,3) or (T,M,3) -> (T,M,3)."""
        if normals.dim() == 2:
            normals = normals.unsqueeze(0)
        return torch.nn.functional.normalize(torch.einsum("tij,tmj->tmi", self.matrix(inverse=True), normals), dim=-1)

    def mean_scale(self) -> Tensor:
        """(T,) exp(mean l), the geometric-mean scale: the stand-in wherever a SCALAR scale enters (the visibility
        z-buffer slack margin * s * diameter), and the `scale` a solve reports."""
        return self.log_stretch.mean(-1).exp()

    def rotated_axes(self, poses: Tensor) -> Tensor:
        """W = R U (T,3,3): the principal axes in the camera frame, columns w_i = R u_i."""
        return poses[:, :3, :3] @ self.axes


def stretch_jacobian(p: Tensor, t: Tensor, rotated_axes: Tensor) -> Tensor:
    """dp/dl (T,n,3,3) for camera points `p` (T,n,3) = R A (y - c) + t, frame translations `t` (T,3) and
    `rotated_axes` W = R U (T,3,3). Column i is sigma_i (u_i^T (y - c)) R u_i = (w_i^T (p - t)) w_i, since
    R^T (p - t) = A (y - c) and u_i^T A = sigma_i u_i^T. It sums over i to p - t, today's scale column."""
    a = torch.einsum("tni,tij->tnj", p - t[:, None, :], rotated_axes)   # a_i = w_i . (p - t)
    return rotated_axes[:, None, :, :] * a[:, :, None, :]


def isotropy_projector(dtype: torch.dtype, device: torch.device | str) -> Tensor:
    """P = I - 11^T/3 (3,3): removes the isotropic part of l; l^T P l = sum_i (l_i - mean l)^2."""
    return torch.eye(3, dtype=dtype, device=device) - torch.full((3, 3), 1.0 / 3.0, dtype=dtype, device=device)


def prior_residuals(log_stretch: Tensor) -> tuple[Tensor, Tensor]:
    """The isotropy prior's unit-weight residual `(r (T,1,3), J (T,1,3,9))` per frame, r = P l and J = [0 | P] over the
    [rho, theta, l] columns. `batch_eval` weights frame k by lam_b / T_b (its clip's prior weight over the clip's frame
    count), so a clip's frames sum to lam_b * l^T P l, with Hessian lam_b P and gradient lam_b P l in the border."""
    T = log_stretch.shape[0]
    P = isotropy_projector(log_stretch.dtype, log_stretch.device)
    r = (log_stretch @ P).unsqueeze(1)                                   # P symmetric
    J = torch.zeros(T, 1, 3, 6 + ANISO_DOF, dtype=log_stretch.dtype, device=log_stretch.device)
    J[..., 6:] = P
    return r, J


def _shape_fields(solve: Any) -> tuple[Tensor, Tensor, Tensor] | None:
    fields = (getattr(solve, "shape_log_stretch", None), getattr(solve, "shape_axes", None),
              getattr(solve, "shape_centroid", None))
    if all(f is None for f in fields):
        return None
    if any(f is None for f in fields):
        raise ValueError("a solve with stretch shape fields needs all of shape_log_stretch, shape_axes, shape_centroid")
    return tuple(torch.as_tensor(f) for f in fields)                     # type: ignore[return-value]


def corrected_template_points(points: Tensor, solve: Any, metric: bool = False) -> Tensor:
    """The template with its proportions corrected by an anisotropic solve, the single place downstream applies A.

    `points` (..., 3) canonical template points; `solve` a `TrajectorySolution` (or anything with its `scale` and the
    `shape_*` fields, e.g. a read `solve.npz`). Returns Y' = c + U diag(exp(l - mean l)) U^T (y - c), the drop-in
    template for the reported poses and scale: p = scale R Y' + t is exactly the solved object. With `metric`,
    returns scale * Y' (metres in the object frame). A solve without shape fields (flag off) returns `points` itself
    (or scale * points), so a consumer can call this unconditionally.

    Follow-up (not done here): the contact and retarget consumers still read the raw template with `scale`."""
    fields = _shape_fields(solve)
    if fields is None:
        return points * float(solve.scale) if metric else points
    ell, U, c = (f.to(device=points.device, dtype=points.dtype) for f in fields)
    rel = (ell - ell.mean()).exp()
    A = (U * rel[None, :]) @ U.T
    out = c + (points - c) @ A.T
    return out * float(solve.scale) if metric else out


def corrected_template_normals(normals: Tensor, solve: Any) -> Tensor:
    """Unit normals of `corrected_template_points`' surface: normalize(A^-1 n) (unchanged without shape fields)."""
    fields = _shape_fields(solve)
    if fields is None:
        return normals
    ell, U, _c = (f.to(device=normals.device, dtype=normals.dtype) for f in fields)
    A_inv = (U * (ell.mean() - ell).exp()[None, :]) @ U.T
    return torch.nn.functional.normalize(normals @ A_inv.T, dim=-1)
