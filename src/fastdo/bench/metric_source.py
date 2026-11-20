"""One metric measurement per window as the gauge's metric source (the result below), and protocol sensitivity.

For one rigid body, write ``l = log lambda`` for the window gauge of
``g_lambda: (s, t_k) -> (lambda s, lambda t_k)``. A metric source ``m`` reads ``-l`` (the correction that would undo
the gauge) with bias ``b`` and variance ``sigma^2``. The result, for a single body with sources ``{(m_i, sigma_i^2)}``:

- GLS estimate ``d_hat = sum_i m_i / sigma_i^2 / sum_i 1 / sigma_i^2`` (:func:`gls_fuse`); its bias is the
  precision-weighted average of the sources' biases (2(ii)) and its variance ``1 / sum_i sigma_i^-2`` never rises
  when a source is added (2(iii));
- the plug-in of one exact source (``sigma -> 0``) puts all the weight on it: ``d_hat = m``.

The known-object-size source (:func:`size_reading`, :func:`known_size_correction`): the posed template's metric
extent ``s_hat * D(template)`` is compared with the measured extent ``D_true`` of the real object;
``kappa = D_true / (s_hat * D(template))`` and the correction is ``g_kappa`` of the object (image-exact: every
pixel of the object stays where it is). Applied to a camera-frame centre trajectory, ``c -> kappa c``.

E1 (metric along-ray RMS) decomposes EXACTLY (:func:`e1_split`): with ``lam`` the least-squares orbit point of
``metric_frame.gauge_decomposition``, ``Z_t = |c_gt,t|`` and ``o_t`` the along-ray component of the off-orbit
remainder ``c_pred,t - lam c_gt,t``, the cross term ``sum_t Z_t o_t`` vanishes by the normal equation, so

    E1^2 = (lam - 1)^2 mean(Z^2) + mean(o^2),

and after ``g_kappa``: ``E1(kappa)^2 = (kappa lam - 1)^2 mean(Z^2) + kappa^2 mean(o^2)``. The first term is the
gauge (lambda) part, the second the off-orbit floor that no gauge correction removes.

Protocol metrics (:func:`object_cd_after_scaled_icp`, :func:`mpjpe_root_aligned`, :func:`pa_mpjpe`) let a study inject a
pure gauge error and measure what each literature protocol sees.

Units: metres in; E1 and MPJPE out in metres (callers convert); Chamfer in cm^2 as HOLD reports it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fastdo.bench.alignment import scaled_icp, umeyama_similarity
from fastdo.bench.metric_frame import along_ray_components, gauge_decomposition, hold_chamfer_cm2

logger = logging.getLogger(__name__)

__all__ = [
    "SIZE_KINDS",
    "E1Split",
    "caliper_extents",
    "caliper_log_sd",
    "e1_split",
    "expected_e1",
    "gls_fuse",
    "known_size_correction",
    "mpjpe_root_aligned",
    "object_cd_after_scaled_icp",
    "pa_mpjpe",
    "size_reading",
]

SIZE_KINDS = ("obb_gm", "obb_max", "rms_radius", "diameter")


def caliper_extents(points: np.ndarray) -> np.ndarray:
    """Edge lengths (m, ascending) of the (approximately) minimum-volume oriented box of a point set.

    This is what three caliper readings of the real object give (L, W, H), independent of any pose.
    """
    import trimesh

    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) < 4:
        raise ValueError(f"need at least 4 points for an oriented box, got {len(p)}")
    hull = trimesh.convex.convex_hull(p)
    _, ext = trimesh.bounds.oriented_bounds(hull)
    return np.sort(np.asarray(ext, dtype=np.float64))


def size_reading(points: np.ndarray, kind: str) -> float:
    """A rotation-invariant metric size of a point set (m).

    Kinds:
        obb_gm: geometric mean of the three oriented-box extents (three caliper readings; the primary kind);
        obb_max: the longest oriented-box extent (one caliper reading);
        rms_radius: RMS distance of the points from their centroid (needs the full surface, not a caliper);
        diameter: the largest distance between two points (convex hull vertices).
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if kind == "obb_gm":
        return float(np.exp(np.mean(np.log(caliper_extents(p)))))
    if kind == "obb_max":
        return float(caliper_extents(p)[-1])
    if kind == "rms_radius":
        return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, axis=1))))
    if kind == "diameter":
        import trimesh
        from scipy.spatial.distance import pdist

        hv = np.asarray(trimesh.convex.convex_hull(p).vertices, dtype=np.float64)
        return float(pdist(hv).max())
    raise ValueError(f"unknown size kind {kind!r}; expected one of {SIZE_KINDS}")


def known_size_correction(template_points: np.ndarray, scale: float, true_points: np.ndarray,
                          kind: str = "obb_gm") -> float:
    """``kappa = D_true / (scale * D(template))``: the object gauge move that makes the posed template's metric size
    equal to the measured size of the real object. Both point sets are pose-free (any rigid frame)."""
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be finite and > 0, got {scale!r}")
    d_t = size_reading(np.asarray(template_points, np.float64) * float(scale), kind)
    d_g = size_reading(true_points, kind)
    return float(d_g / d_t)


def caliper_log_sd(extents: np.ndarray, sd_m: float, kind: str = "obb_gm") -> float:
    """Delta-method sd of ``log D`` when each caliper extent carries independent N(0, sd_m^2) error."""
    e = np.asarray(extents, dtype=np.float64)
    if kind == "obb_gm":
        return float(sd_m * math.sqrt(np.sum(1.0 / e ** 2)) / len(e))
    if kind == "obb_max":
        return float(sd_m / e[-1])
    raise ValueError(f"caliper error is defined for obb_gm / obb_max, got {kind!r}")


def gls_fuse(readings: Sequence[float], variances: Sequence[float]) -> tuple[float, float]:
    """The result above, for one body: GLS combination ``(sum m_i / v_i) / (sum 1 / v_i)`` and its variance ``1 / sum 1/v_i``.

    A variance of 0 marks an exact source: the estimate is that reading (several exact sources must agree).
    """
    m = np.asarray(readings, dtype=np.float64)
    v = np.asarray(variances, dtype=np.float64)
    if m.shape != v.shape or m.size == 0:
        raise ValueError("readings and variances must be equally long and non-empty")
    if np.any(v < 0) or not np.all(np.isfinite(m)):
        raise ValueError("variances must be >= 0 and readings finite")
    exact = v == 0
    if exact.any():
        vals = m[exact]
        if np.ptp(vals) > 1e-12:
            raise ValueError("two exact sources disagree")
        return float(vals[0]), 0.0
    w = 1.0 / v
    return float(np.sum(w * m) / np.sum(w)), float(1.0 / np.sum(w))


@dataclass(frozen=True)
class E1Split:
    """E1 and its exact split (metres), for a centre trajectory after the gauge move ``kappa``.

    Attributes:
        e1: RMS over frames of the signed along-ray error.
        gauge_part: ``|kappa lam - 1| * rms(Z)``, the lambda part.
        offorbit_part: ``kappa * rms(o)``, the off-orbit along-ray floor.
        lam: orbit coordinate after the move (``kappa * lam_base``).
        z_rms: rms over frames of the true range.
    """

    e1: float
    gauge_part: float
    offorbit_part: float
    lam: float
    z_rms: float


def e1_split(c_pred: np.ndarray, c_gt: np.ndarray, kappa: float = 1.0) -> E1Split:
    """E1 of ``kappa * c_pred`` against ``c_gt`` with its exact gauge / off-orbit split (module docstring)."""
    p = np.asarray(c_pred, dtype=np.float64).reshape(-1, 3) * float(kappa)
    g = np.asarray(c_gt, dtype=np.float64).reshape(-1, 3)
    dec = gauge_decomposition(p, g)
    ok = np.isfinite(p).all(1) & np.isfinite(g).all(1)
    p, g = p[ok], g[ok]
    ar = along_ray_components(p, g)
    z = np.linalg.norm(g, axis=1)
    o = np.einsum("ij,ij->i", p - dec.lam * g, g / z[:, None])
    e1 = float(np.sqrt(np.mean(ar.signed_along ** 2)))
    z_rms = float(np.sqrt(np.mean(z ** 2)))
    return E1Split(e1=e1, gauge_part=abs(dec.lam - 1.0) * z_rms, offorbit_part=float(np.sqrt(np.mean(o ** 2))),
                   lam=dec.lam, z_rms=z_rms)


def expected_e1(c_pred: np.ndarray, c_gt: np.ndarray, kappa0: float, sigma: float = 0.0,
                two_point: float | None = None) -> float:
    """RMS-expected E1 (m) of ``kappa c_pred`` when the measured correction is ``kappa = kappa0 * exp(eps)``.

    ``eps ~ N(0, sigma^2)`` (closed form: E[e^eps] = e^{sigma^2/2}, E[e^{2 eps}] = e^{2 sigma^2}), or, with
    ``two_point = m``, ``eps = +-log(1 + m)`` with equal probability (a look-alike object m larger or smaller).
    Returns ``sqrt(E[E1^2])``.
    """
    p = np.asarray(c_pred, dtype=np.float64).reshape(-1, 3)
    g = np.asarray(c_gt, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(p).all(1) & np.isfinite(g).all(1)
    p, g = p[ok], g[ok]
    z = np.linalg.norm(g, axis=1)
    a = np.einsum("ij,ij->i", p, g / z[:, None])          # along-true-ray range of the prediction
    pp, pz, zz = float(np.mean(a * a)), float(np.mean(a * z)), float(np.mean(z * z))
    if two_point is not None:
        ks = [kappa0 * (1.0 + two_point), kappa0 / (1.0 + two_point)]
        ms = [k * k * pp - 2.0 * k * pz + zz for k in ks]
        return float(math.sqrt(max(np.mean(ms), 0.0)))
    m1, m2 = math.exp(sigma ** 2 / 2.0), math.exp(2.0 * sigma ** 2)
    return float(math.sqrt(max(kappa0 ** 2 * m2 * pp - 2.0 * kappa0 * m1 * pz + zz, 0.0)))


def object_cd_after_scaled_icp(pred: Sequence[np.ndarray], gt: Sequence[np.ndarray],
                               max_iterations: int = 100, centre: bool = True,
                               normalise_init: bool = False) -> tuple[float, float]:
    """Object Chamfer (cm^2, HOLD units) after ONE scaled ICP (scale, rotation, translation) over the given frames.

    The AGILE / HOLD object protocol ("after ICP alignment of scale, rotation, and translation"); returns
    ``(mean over frames of the aligned Chamfer, the fitted scale)``. ``centre`` subtracts each cloud's centroid
    first (shape evaluation in a centred frame). Without it, ICP from the identity can stall when the prediction
    sits centimetres away along the ray (a 1.2x gauge at 0.6 m: 12 cm), which is an initialisation artefact, not a
    property of the protocol; the unit test keeps that case as a documented counter-example.
    ``normalise_init`` (requires ``centre``) first scales the predictions by rms(gt) / rms(pred), pooled over
    frames, so the ICP starts from the same normalised cloud whatever the prediction's scale: the result is then
    invariant to a gauge by construction (up to float rounding); the returned scale includes that factor.
    """
    pre = 1.0
    if centre:
        pred = [np.asarray(p, np.float64) - np.asarray(p, np.float64).mean(0) for p in pred]
        gt = [np.asarray(g, np.float64) - np.asarray(g, np.float64).mean(0) for g in gt]
        if normalise_init:
            rp = math.sqrt(np.mean(np.concatenate([np.sum(p ** 2, 1) for p in pred])))
            rg = math.sqrt(np.mean(np.concatenate([np.sum(g ** 2, 1) for g in gt])))
            pre = rg / rp
            pred = [pre * p for p in pred]
    elif normalise_init:
        raise ValueError("normalise_init needs centre=True")
    aligned, fit = scaled_icp(list(pred), list(gt), max_iterations=max_iterations)
    vals = [hold_chamfer_cm2(a, np.asarray(g, np.float64)) for a, g in zip(aligned, gt, strict=True)]
    return float(np.mean(vals)), float(fit.scale) * pre


def mpjpe_root_aligned(pred_joints: np.ndarray, gt_joints: np.ndarray, root: int = 0) -> np.ndarray:
    """Per-frame MPJPE (m) after subtracting each side's root joint (MANO joint 0, the wrist). (T, J, 3) inputs."""
    p = np.asarray(pred_joints, dtype=np.float64)
    g = np.asarray(gt_joints, dtype=np.float64)
    pr = p - p[:, root:root + 1]
    gr = g - g[:, root:root + 1]
    return np.linalg.norm(pr - gr, axis=2).mean(axis=1)


def pa_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray) -> np.ndarray:
    """Per-frame MPJPE (m) after a per-frame similarity (Procrustes with scale) of the prediction onto the GT."""
    p = np.asarray(pred_joints, dtype=np.float64)
    g = np.asarray(gt_joints, dtype=np.float64)
    out = np.full(len(p), np.nan)
    for t in range(len(p)):
        if not (np.isfinite(p[t]).all() and np.isfinite(g[t]).all()):
            continue
        s, r, tr = umeyama_similarity(p[t], g[t])
        out[t] = float(np.linalg.norm(s * p[t] @ r.T + tr - g[t], axis=1).mean())
    return out
