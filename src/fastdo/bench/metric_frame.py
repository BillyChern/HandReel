"""Metric-frame scorers: quantities that see along-ray error, which ICP-aligned protocols forgive.

Every function here scores the PHYSICAL property in metres in the frame it is given, with no scale or rigid
alignment unless the name says so:

- :func:`along_ray_components` splits a camera-frame position error into its component along the true viewing ray
  and the perpendicular remainder (image functionals are blind to the first, not the second).
- :func:`gauge_decomposition` splits a trajectory error into the part ON the gauge orbit of the truth,
  ``g_lambda: (s, t_k) -> (lambda s, lambda t_k)`` (one scalar per window, image-exact), and the OFF-orbit
  remainder. A pure multiplicative depth bias lands exactly on the orbit; production errors do not,
  and the remedy used here is "project onto the orbit, then pick lambda".
- :func:`cd_h_hold` is HOLD's hand-relative Chamfer (zc-alexfan/hold@065c0e1e, ``eval_cd_f_right`` +
  ``calculate_chamfer_f_scores``, MIT): object vertices minus the MANO wrist (joint 0), translation only, 3,000
  random vertices per side after one ``torch.manual_seed(1)``, metres x 100, squared distances, cm^2, NaN on invalid
  GT frames, nanmean over frames. It is the CD_h that 4D-HOF / AGILE cite. Sampling order is HOLD's exactly so a
  reference run can be matched bit for bit.
- :func:`pairwise_object_errors` scores two objects' relative placement, which is invariant to the (unknown) camera
  pose: V2D Track 3 has no camera-to-mocap extrinsic, so these are its frame-free endpoints.

Units: metres in, metres out, except CD_h (cm^2) as HOLD reports it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

__all__ = [
    "AlongRay",
    "GaugeDecomposition",
    "along_ray_components",
    "apply_gauge",
    "cd_h_hold",
    "gauge_decomposition",
    "hold_chamfer_cm2",
    "pairwise_object_errors",
]


@dataclass(frozen=True)
class AlongRay:
    """Per-frame decomposition of ``pred - gt`` (camera frame, metres).

    Attributes:
        signed_along: (T,) component along the unit TRUE ray ``gt / |gt|``; positive = predicted too far.
        perp: (T,) norm of the perpendicular remainder.
        total: (T,) ``|pred - gt|``.
        log_depth_ratio: (T,) ``log(|pred| / |gt|)`` (range ratio, the gauge's own coordinate).
    """

    signed_along: np.ndarray
    perp: np.ndarray
    total: np.ndarray
    log_depth_ratio: np.ndarray


def along_ray_components(pred: np.ndarray, gt: np.ndarray) -> AlongRay:
    """Split camera-frame position errors (T, 3) into along-ray and perpendicular parts (NaN rows stay NaN)."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    g = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
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
    """``pred_t = lam * gt_t + off_t`` with ``lam`` the least-squares orbit point (one scalar per window).

    Attributes:
        lam: The orbit coordinate of the prediction (1.0 = on the truth's own point of the orbit).
        on_orbit_rms: RMS over frames of ``|(lam - 1) gt_t|`` (m): what an image-exact gauge correction can remove.
        off_orbit_rms: RMS over frames of ``|pred_t - lam gt_t|`` (m): what no gauge correction can remove.
        log_size_ratio: ``log(s_pred / s_gt)`` when sizes are given, else NaN.
        orbit_coherence: ``log_size_ratio - log(lam)``: 0 when size and distance are wrong by the same factor, as
            a pure depth bias gives; production reads "right size, wrong distance".
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
    """Least-squares split of camera-frame centre trajectories (T, 3) into on- and off-orbit parts.

    ``lam = sum_t <pred_t, gt_t> / sum_t |gt_t|^2`` minimises ``sum_t |pred_t - lam gt_t|^2``.
    """
    p = np.asarray(pred_t, dtype=np.float64).reshape(-1, 3)
    g = np.asarray(gt_t, dtype=np.float64).reshape(-1, 3)
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} and gt {g.shape} must match")
    ok = np.isfinite(p).all(axis=1) & np.isfinite(g).all(axis=1)
    if not ok.any():
        raise ValueError("no frame is finite in both trajectories")
    p, g = p[ok], g[ok]
    lam = float(np.sum(p * g) / np.sum(g * g))
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
    """``g_lam`` of the paper's eq. (gauge): every translation and the scale times ``lam``; rotations unchanged."""
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"lam must be finite and > 0, got {lam!r}")
    out = np.array(poses, dtype=np.float64, copy=True)
    out[..., :3, 3] *= lam
    return out, float(scale) * lam


def hold_chamfer_cm2(source: np.ndarray, target: np.ndarray) -> float:
    """HOLD's ``calculate_chamfer_f_scores`` Chamfer term: metres x 100, mean squared NN distance both ways (cm^2)."""
    s = np.asarray(source, dtype=np.float64) * 100.0
    t = np.asarray(target, dtype=np.float64) * 100.0
    d_ts, _ = cKDTree(s).query(t)
    d_st, _ = cKDTree(t).query(s)
    return float(np.mean(np.square(d_ts)) + np.mean(np.square(d_st)))


def cd_h_hold(pred_obj_verts: np.ndarray, pred_wrist: np.ndarray, gt_obj_verts: np.ndarray, gt_wrist: np.ndarray,
              gt_valid: np.ndarray | None = None, n_sample: int = 3000, seed: int = 1) -> tuple[float, np.ndarray]:
    """HOLD-compatible hand-relative Chamfer distance (cm^2): ``(sequence nanmean, per-frame values)``.

    Args:
        pred_obj_verts: (T, Vp, 3) predicted object vertices, camera frame, metres.
        pred_wrist: (T, 3) predicted MANO joint 0 (wrist), same frame.
        gt_obj_verts: (T, Vg, 3) ground-truth object vertices, ground-truth camera frame.
        gt_wrist: (T, 3) ground-truth wrist.
        gt_valid: (T,) bool; invalid frames are NaN (HOLD sets them after computing). None = all valid.
        n_sample: Vertices drawn per side per frame (HOLD: 3000, without replacement via randperm).
        seed: ``torch.manual_seed`` called ONCE before the frame loop (HOLD's ``eval_cd_f_right``).

    HOLD draws ``randperm(n_pred)`` then ``randperm(n_gt)`` per frame, and draws NOTHING on a frame whose prediction
    is NaN; this function reproduces that order exactly, so the subsample of every later frame matches.
    """
    import torch

    pv = np.asarray(pred_obj_verts, dtype=np.float64)
    gv = np.asarray(gt_obj_verts, dtype=np.float64)
    pw = np.asarray(pred_wrist, dtype=np.float64).reshape(-1, 3)
    gw = np.asarray(gt_wrist, dtype=np.float64).reshape(-1, 3)
    n_t = pv.shape[0]
    if not (gv.shape[0] == pw.shape[0] == gw.shape[0] == n_t):
        raise ValueError("pred/gt vertices and wrists must share the frame axis")
    valid = np.ones(n_t, dtype=bool) if gt_valid is None else np.asarray(gt_valid, dtype=bool).reshape(-1)
    rel_p = pv - pw[:, None, :]
    rel_g = gv - gw[:, None, :]
    torch.manual_seed(seed)
    out = np.full(n_t, np.nan)
    for t in range(n_t):
        if not np.isfinite(rel_p[t].mean()):
            continue  # HOLD: no randperm is drawn for a NaN prediction
        idx_p = torch.randperm(rel_p.shape[1])[:n_sample].numpy()
        idx_g = torch.randperm(rel_g.shape[1])[:n_sample].numpy()
        out[t] = hold_chamfer_cm2(rel_p[t, idx_p], rel_g[t, idx_g])
    out[~valid] = np.nan
    with np.errstate(invalid="ignore"):
        value = float(np.nanmean(out)) if np.isfinite(out).any() else float("nan")
    return value, out


def pairwise_object_errors(pred_a: np.ndarray, pred_b: np.ndarray, gt_a: np.ndarray,
                           gt_b: np.ndarray) -> dict[str, np.ndarray]:
    """Camera-pose-free errors of object A placed relative to object B, per frame.

    Args:
        pred_a, pred_b: (T, 4, 4) predicted object-to-frame poses, both in ONE (any) predicted frame, metres; the
            translation is taken as the object's centre (pass centre-offset poses if the origin is not the centre).
        gt_a, gt_b: (T, 4, 4) ground truth in one (any) ground-truth frame (V2D: the episode's mocap world).

    Returns:
        ``dist_err`` (signed, |c_A - c_B| predicted minus true), ``rel_pos_err`` (|position of A in B's frame,
        predicted minus true|; meaningful only when B has no rotational symmetry) and ``dist_true``. Both errors are
        invariant to any rigid transform applied to all predicted poses together (and to all GT poses together), so
        they need no camera-to-world registration; a global gauge ``lambda`` changes ``dist_err`` by
        ``(lambda - 1) * dist_true`` exactly.
    """
    pa, pb, ga, gb = (np.asarray(x, dtype=np.float64) for x in (pred_a, pred_b, gt_a, gt_b))
    d_pred = np.linalg.norm(pa[:, :3, 3] - pb[:, :3, 3], axis=1)
    d_true = np.linalg.norm(ga[:, :3, 3] - gb[:, :3, 3], axis=1)
    rel_p = np.einsum("tji,tj->ti", pb[:, :3, :3], pa[:, :3, 3] - pb[:, :3, 3])
    rel_g = np.einsum("tji,tj->ti", gb[:, :3, :3], ga[:, :3, 3] - gb[:, :3, 3])
    return {"dist_err": d_pred - d_true, "rel_pos_err": np.linalg.norm(rel_p - rel_g, axis=1), "dist_true": d_true}
