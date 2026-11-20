# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""The unaligned along-ray error ``e_par`` and its exact split into a gauge part and an off-orbit term.

All inputs are camera-frame positions in metres with the camera centre at the origin, so the true viewing ray of a
point ``c_gt`` is ``c_gt / |c_gt|``. No scale, rotation or translation alignment is applied anywhere in this module.

Definitions for one rigid body with predicted and true reference-point trajectories ``c_t`` and ``c_gt,t``
(``t = 1..T``, frames finite in both), ``<.>`` the mean over frames:

- gauge orbit of the truth: ``g_lam: c -> lam c`` (with the body's size times ``lam``). Under a pinhole camera it is
  image-exact for the body's isolated projected points: each stays where it is, so no function of them can see it
  (occlusion or contact with another body can change pixels);
- least-squares orbit point: ``lam = sum_t <c_t, c_gt,t> / sum_t |c_gt,t|^2``;
- true range ``Z_t = |c_gt,t|`` and the along-ray part of the off-orbit residual
  ``o_t = (c_t - lam c_gt,t) . c_gt,t / Z_t``;
- signed along-ray error ``a_t = (c_t - c_gt,t) . c_gt,t / Z_t = (lam - 1) Z_t + o_t``;
- ``e_par = <a^2>^(1/2)``.

Because ``sum_t Z_t o_t = 0`` is the normal equation of ``lam``, the cross term vanishes and

    e_par^2 = (lam - 1)^2 <Z^2> + <o^2>          (exact),

the gauge part ``|lam - 1| <Z^2>^(1/2)`` and the off-orbit term ``<o^2>^(1/2)``. These split the UNCORRECTED error in
quadrature into components parallel and orthogonal to the truth's scale direction ``Z``. In mean square they are
exactly what a move of the TRUTH to its nearest orbit point ``lam c_gt`` takes away and leaves. They are generally not
the gain and the remainder of rescaling the PREDICTION, because a gauge move ``g_kappa`` of the prediction rescales
``o`` with it:

    e_par(kappa)^2 = (kappa lam - 1)^2 <Z^2> + kappa^2 <o^2>.

The oracle ``g_{1/lam}`` zeroes the gauge part and leaves ``oracle_floor = <o^2>^(1/2) / lam``. The minimum over
``kappa`` lies at :func:`kappa_star` ``= lam <Z^2> / (lam^2 <Z^2> + <o^2>)`` and is :func:`e_par_min`
``= (<Z^2> <o^2> / (lam^2 <Z^2> + <o^2>))^(1/2)``: below ``oracle_floor`` whenever ``<o^2> > 0``, and above or below
the off-orbit term depending on ``lam`` and ``<o^2> / <Z^2>`` (above it exactly when ``(1 - lam^2) <Z^2> > <o^2>``).

Scale or constant shift? Within a short window the true range barely changes, so the scale direction (``a ~ Z``) and
the constant-depth-shift direction (``a ~ 1``) are nearly the same vector. :class:`EParSplit` therefore also reports
``collinearity = <Z>^2 / <Z^2>`` (1 = indistinguishable), ``shift_part = |<a>|`` (what ONE constant along-ray shift
removes) and ``shift_scale_part`` (what a joint least-squares fit of ``a`` on ``span{1, Z}`` removes). A pure constant
shift ``d`` puts ``collinearity * d^2`` of its mean square into the gauge part, so the gauge part is evidence of a
scale error only when ``gauge_part^2`` clearly exceeds ``shift_part^2``; ``collinearity < 1`` is necessary, not
sufficient. With fewer than :data:`MIN_ROWS` frames the split is all gauge (one frame) or nearly so by construction;
such a split carries ``low_n = True`` and warns.

The arithmetic follows the authors' development implementation operation for operation, so the two agree bit for
bit (a parity suite kept outside this package).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    "MIN_ROWS",
    "AlongRay",
    "EParSplit",
    "GaugeDecomposition",
    "along_ray_components",
    "apply_gauge",
    "e_par_min",
    "e_par_split",
    "gauge_decomposition",
    "kappa_star",
]

_CONVENTION_HINT = ("check that pred and gt are in the same camera frame (camera at the origin, +z forward), in the "
                    "same units (metres), and that missing frames are NaN rather than zero-filled")
_Z_RMS_WARN = 10.0              # metres: an RMS true range above this is more likely millimetres
_LAM_WARN = (0.1, 10.0)         # a per-body scale outside this is more likely a convention or unit error
MIN_ROWS = 3                    # a split over fewer frames is flagged ``low_n`` (one frame: all gauge by construction)


def _traj3(x: np.ndarray, name: str) -> np.ndarray:
    """A (T, 3) float64 trajectory; anything else is refused rather than reshaped."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must be a (T, 3) array, got shape {a.shape}")
    return a


@dataclass(frozen=True)
class AlongRay:
    """Per-frame decomposition of ``pred - gt`` (camera frame, metres).

    Attributes:
        signed_along: (T,) component along the unit TRUE ray ``gt / |gt|``; positive = predicted too far.
        perp: (T,) norm of the perpendicular remainder.
        total: (T,) ``|pred - gt|``.
        log_depth_ratio: (T,) ``log(|pred| / |gt|)``, the gauge's own coordinate.
    """

    signed_along: np.ndarray
    perp: np.ndarray
    total: np.ndarray
    log_depth_ratio: np.ndarray


def along_ray_components(pred: np.ndarray, gt: np.ndarray) -> AlongRay:
    """Split camera-frame position errors (T, 3) into along-ray and perpendicular parts (NaN rows stay NaN).

    Raises:
        ValueError: inputs are not (T, 3) of equal T, or a ground-truth point sits at the camera centre.
    """
    p = _traj3(pred, "pred")
    g = _traj3(gt, "gt")
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} and gt {g.shape} must match")
    rng = np.linalg.norm(g, axis=1)
    if np.any(rng[np.isfinite(rng)] <= 0.0):
        raise ValueError("a ground-truth point sits at the camera centre; the ray is undefined")
    u = g / rng[:, None]
    d = p - g
    along = np.einsum("ij,ij->i", d, u)
    perp = np.linalg.norm(d - along[:, None] * u, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ldr = np.log(np.linalg.norm(p, axis=1) / rng)
    return AlongRay(signed_along=along, perp=perp, total=np.linalg.norm(d, axis=1), log_depth_ratio=ldr)


@dataclass(frozen=True)
class GaugeDecomposition:
    """``pred_t = lam * gt_t + off_t`` with ``lam`` the least-squares orbit point (one scalar per trajectory).

    Attributes:
        lam: Orbit coordinate of the prediction (1.0 = the truth's own point of the orbit).
        on_orbit_rms: RMS over frames of ``|(lam - 1) gt_t|`` (m): the uncorrected error's component along the truth's
            orbit, in quadrature with ``off_orbit_rms``. It is not what the best gauge move of the prediction
            removes, because that move also rescales the off-orbit residual (module docstring).
        off_orbit_rms: RMS over frames of ``|pred_t - lam gt_t|`` (m, full 3-D): the distance from the prediction to
            the nearest point of the truth's orbit. It is not the error left after the best gauge move of the
            prediction, which can be smaller or larger.
        log_size_ratio: ``log(s_pred / s_gt)`` when sizes are given, else NaN.
        orbit_coherence: ``log_size_ratio - log(lam)``: 0 when size and distance are wrong by the same factor.
        n_frames: Frames used (finite in both).
    """

    lam: float
    on_orbit_rms: float
    off_orbit_rms: float
    log_size_ratio: float
    orbit_coherence: float
    n_frames: int


def gauge_decomposition(pred_t: np.ndarray, gt_t: np.ndarray, pred_size: float | None = None,
                        gt_size: float | None = None) -> GaugeDecomposition:
    """Least-squares split of camera-frame trajectories (T, 3) into on- and off-orbit parts.

    ``lam = sum_t <pred_t, gt_t> / sum_t |gt_t|^2`` minimises ``sum_t |pred_t - lam gt_t|^2``.

    Raises:
        ValueError: inputs are not (T, 3) of equal T, no frame is finite in both, or ``lam`` is not finite and
            positive (a sign of mixed conventions, units or zero-filled frames).
    """
    p = _traj3(pred_t, "pred")
    g = _traj3(gt_t, "gt")
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} and gt {g.shape} must match")
    ok = np.isfinite(p).all(axis=1) & np.isfinite(g).all(axis=1)
    if not ok.any():
        raise ValueError("no frame is finite in both trajectories")
    p, g = p[ok], g[ok]
    gg = float(np.sum(g * g))
    if not gg > 0.0:
        raise ValueError(f"every finite ground-truth point sits at the camera centre; {_CONVENTION_HINT}")
    lam = float(np.sum(p * g) / gg)
    if not (np.isfinite(lam) and lam > 0.0):
        raise ValueError(f"the least-squares scale lam = {lam!r} is not finite and positive; {_CONVENTION_HINT}")
    on = np.linalg.norm((lam - 1.0) * g, axis=1)
    off = np.linalg.norm(p - lam * g, axis=1)
    if pred_size is not None and gt_size is not None:
        lsr = float(np.log(pred_size / gt_size))
        coh = lsr - float(np.log(lam))
    else:
        lsr, coh = float("nan"), float("nan")
    return GaugeDecomposition(lam=lam, on_orbit_rms=float(np.sqrt(np.mean(on ** 2))),
                              off_orbit_rms=float(np.sqrt(np.mean(off ** 2))), log_size_ratio=lsr,
                              orbit_coherence=coh, n_frames=int(ok.sum()))


def apply_gauge(poses: np.ndarray, scale: float, lam: float) -> tuple[np.ndarray, float]:
    """``g_lam`` on (..., 4, 4) poses and a body scale: every translation and the scale times ``lam``.

    Raises:
        ValueError: ``lam`` is not finite and positive.
    """
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"lam must be finite and > 0, got {lam!r}")
    out = np.array(poses, dtype=np.float64, copy=True)
    out[..., :3, 3] *= lam
    return out, float(scale) * lam


@dataclass(frozen=True)
class EParSplit:
    """``e_par`` and its exact split (metres) for one body.

    Attributes:
        e_par: RMS over frames of the signed along-ray error (unaligned).
        gauge_part: ``|lam - 1| * z_rms``, the uncorrected error's component along the truth's scale direction. It
            is not the reduction in ``e_par`` that the best gauge move of the prediction achieves (:func:`e_par_min`).
        off_orbit_part: ``rms(o)``, the along-ray component of the off-orbit residual, orthogonal to that direction.
            It scales with a gauge move of the prediction, so it is neither ``e_par`` after the oracle correction
            (``oracle_floor``) nor the least ``e_par`` over gauge moves (:func:`e_par_min`).
        lam: Per-body least-squares orbit point (scale) of the prediction.
        z_rms: RMS over frames of the true range ``|c_gt|``.
        oracle_floor: ``off_orbit_part / lam``: ``e_par`` after the oracle correction ``g_{1/lam}``. This is NOT
            the minimum of ``e_par(kappa)`` over ``kappa`` (that minimum is lower whenever ``off_orbit_part > 0``);
            it is what removing the gauge part, and nothing else, leaves. The minimum is :func:`e_par_min`.
        n_frames: Frames used (finite in both).
        shift_part: ``|<a>|``, the RMS one constant along-ray shift removes.
        shift_scale_part: the RMS removed by a joint least-squares fit of ``a`` on ``span{1, Z}``.
        collinearity: ``<Z>^2 / <Z^2>`` in (0, 1]; near 1 = a scale and a constant shift are indistinguishable.
        low_n: ``n_frames < MIN_ROWS``; such a split says little (one frame: off_orbit_part = 0 by construction).

    ``e_par ** 2 == gauge_part ** 2 + off_orbit_part ** 2`` up to float rounding (module docstring).
    """

    e_par: float
    gauge_part: float
    off_orbit_part: float
    lam: float
    z_rms: float
    oracle_floor: float
    n_frames: int
    shift_part: float = float("nan")
    shift_scale_part: float = float("nan")
    collinearity: float = float("nan")
    low_n: bool = False


def e_par_split(c_pred: np.ndarray, c_gt: np.ndarray, kappa: float = 1.0) -> EParSplit:
    """``e_par`` of ``kappa * c_pred`` against ``c_gt`` (T, 3) with its exact gauge / off-orbit split.

    ``kappa`` applies a gauge move ``g_kappa`` to the prediction first (1.0 = score the prediction as given), so
    ``e_par(kappa)^2 = (kappa lam - 1)^2 <Z^2> + kappa^2 <o^2>``. Frames that are not finite in both are dropped.

    Warns (``UserWarning``) when ``z_rms`` exceeds 10 (likely millimetres), ``lam`` leaves [0.1, 10], or fewer than
    ``MIN_ROWS`` frames are used (``low_n``).

    Raises:
        ValueError: inputs are not (T, 3) of equal T, ``kappa`` is not finite and positive, no frame is finite in
            both, a true point sits at the camera centre, or ``lam`` is not finite and positive.
    """
    if not (np.isfinite(kappa) and kappa > 0.0):
        raise ValueError(f"kappa must be finite and > 0, got {kappa!r}")
    p = _traj3(c_pred, "c_pred") * float(kappa)
    g = _traj3(c_gt, "c_gt")
    dec = gauge_decomposition(p, g)
    ok = np.isfinite(p).all(1) & np.isfinite(g).all(1)
    p, g = p[ok], g[ok]
    ar = along_ray_components(p, g)
    z = np.linalg.norm(g, axis=1)
    o = np.einsum("ij,ij->i", p - dec.lam * g, g / z[:, None])
    e_par = float(np.sqrt(np.mean(ar.signed_along ** 2)))
    z_rms = float(np.sqrt(np.mean(z ** 2)))
    off = float(np.sqrt(np.mean(o ** 2)))
    if z_rms > _Z_RMS_WARN:
        warnings.warn(f"RMS true range {z_rms:.3g} > {_Z_RMS_WARN:g}: inputs look like millimetres, not metres",
                      UserWarning, stacklevel=2)
    if not _LAM_WARN[0] <= dec.lam <= _LAM_WARN[1]:
        warnings.warn(f"per-body scale lam = {dec.lam:.3g} is outside [{_LAM_WARN[0]:g}, {_LAM_WARN[1]:g}]: "
                      f"{_CONVENTION_HINT}", UserWarning, stacklevel=2)
    if dec.n_frames < MIN_ROWS:
        warnings.warn(f"only {dec.n_frames} frame(s): with fewer than {MIN_ROWS} the split is all gauge or nearly "
                      "so by construction (low_n)", UserWarning, stacklevel=2)
    shift, both, coll = _shift_reading(ar.signed_along, z)
    return EParSplit(e_par=e_par, gauge_part=abs(dec.lam - 1.0) * z_rms, off_orbit_part=off, lam=dec.lam,
                     z_rms=z_rms, oracle_floor=off / dec.lam, n_frames=dec.n_frames,
                     shift_part=shift, shift_scale_part=both, collinearity=coll, low_n=dec.n_frames < MIN_ROWS)


class _SplitLike(Protocol):
    lam: float
    z_rms: float
    off_orbit_part: float


def _quadratic(split: _SplitLike) -> tuple[float, float, float]:
    """``(lam, z_rms, off_orbit_part)`` of an :class:`EParSplit` or ``BodyScore``, checked."""
    lam, z, off = float(split.lam), float(split.z_rms), float(split.off_orbit_part)
    if not (math.isfinite(lam) and lam > 0.0 and math.isfinite(z) and z > 0.0 and math.isfinite(off) and off >= 0.0):
        raise ValueError(f"need finite lam > 0, z_rms > 0 and off_orbit_part >= 0, got {lam!r}, {z!r}, {off!r}")
    return lam, z, off


def kappa_star(split: _SplitLike) -> float:
    """The gauge move ``kappa`` of the prediction that minimises ``e_par(kappa)``: ``lam A / (lam^2 A + B)``.

    ``A = z_rms^2`` and ``B = off_orbit_part^2``; ``e_par(kappa)^2 = (kappa lam - 1)^2 A + kappa^2 B`` is a convex
    quadratic in ``kappa``. It equals the oracle ``1 / lam`` only when ``B = 0`` and is smaller otherwise. It reads the
    ground truth: a reference, not a correction a method can run.

    Raises:
        ValueError: ``lam`` or ``z_rms`` is not finite and positive, or ``off_orbit_part`` is negative or not finite.
    """
    lam, z, off = _quadratic(split)
    a, b = z * z, off * off
    return lam * a / (lam * lam * a + b)


def e_par_min(split: _SplitLike) -> float:
    """The least ``e_par`` over gauge moves of the prediction, ``e_par(kappa_star)``: ``(A B / (lam^2 A + B))^(1/2)``.

    Below ``oracle_floor`` whenever ``off_orbit_part > 0``; it may lie above or below ``off_orbit_part``. Like
    :func:`kappa_star` it reads the ground truth.

    Raises:
        ValueError: as :func:`kappa_star`.
    """
    lam, z, off = _quadratic(split)
    return z * off / math.hypot(lam * z, off)


def _shift_reading(a: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """``(|<a>|, RMS of the projection of a on span{1, Z}, <Z>^2 / <Z^2>)`` for signed along-ray errors ``a``."""
    x = np.stack([np.ones_like(z), z], axis=1)
    beta, *_ = np.linalg.lstsq(x, a, rcond=None)
    both = float(np.sqrt(np.mean((x @ beta) ** 2)))
    return abs(float(np.mean(a))), both, float(np.mean(z) ** 2 / np.mean(z * z))
