# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""One-call scoring of predicted against true hand and object trajectories in camera coordinates.

:func:`score` returns, per body, ``e_par`` with its exact split (gauge part, off-orbit term, per-body ``lam``),
the unaligned 3-D and perpendicular errors, the image reprojection error when intrinsics are given (exactly blind
to the gauge, like every image functional), and the pose-aligned protocol metrics for contrast.

Each body is scored at ONE reference point per frame: the object centre (pass (T, 3) centres, or (T, 4, 4)
object-to-camera poses whose translation is the centre) and the hand root (pass (T, 3) wrists, or (T, J, 3)
joints with ``hand_root`` the root index; MANO joint 0 is the wrist). The per-body ``lam`` is the
least-squares orbit point of that trajectory. Input conventions are in the README ("Input conventions").

Rows are frames. Without frame ids, row ``t`` of the prediction and row ``t`` of the truth are taken to be the same
frame (synchronized row by row; nothing checks it). With ``pred_ids`` / ``gt_ids`` (one integer id per row) the rows
are joined on the ids by the rules of :mod:`alongray.change`: the two id sets must be identical (no intersection, no
trimming, no shift), duplicate or non-integer ids are refused, and the joined rows are taken in increasing id order.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from alongray.aligned import mpjpe_root_aligned, object_cd_after_scaled_icp, pa_mpjpe, umeyama_similarity
from alongray.change import _ids
from alongray.core import along_ray_components, e_par_split, gauge_decomposition

__all__ = ["BodyScore", "Score", "intrinsics_matrix", "reference_points", "reprojection_error_px", "score",
           "score_body"]


@dataclass(frozen=True)
class BodyScore:
    """Scores of one body (metres unless the name says otherwise).

    Attributes:
        e_par: Unaligned RMS signed along-ray error of the reference point.
        gauge_part: ``|lam - 1| * z_rms``; ``e_par**2 == gauge_part**2 + off_orbit_part**2``.
        off_orbit_part: RMS along-ray component of ``c_pred - lam c_gt``.
        lam: Per-body least-squares scale (orbit point) of the predicted trajectory.
        z_rms: RMS true range.
        oracle_floor: ``e_par`` after the oracle gauge correction ``g_{1/lam}`` (not the minimum over all gauges).
        e_perp: Unaligned RMS error perpendicular to the true ray.
        e_3d: Unaligned RMS 3-D position error.
        n_frames: Frames used (finite in both).
        reproj_rms_px: RMS image reprojection error of the reference point (pixels), NaN without intrinsics.
        n_reproj_frames: Frames in ``reproj_rms_px`` (z > 0 in both; 0 without intrinsics).
        aligned: Pose-aligned protocol metrics, for contrast (keys documented in :func:`score_body`).
        shift_part, shift_scale_part, collinearity, low_n: the constant-shift reading and the few-frames flag of
            :class:`alongray.core.EParSplit`.
        aligned_n_frames: For each key of ``aligned``, the frames its value was computed on (0 when it is NaN). The
            centre metrics use the frames whose reference point is finite in both; the joint metrics the frames
            whose every joint is finite in both; the Chamfer the frames whose clouds are finite in both.
        n_frames_input: Frames given (before any frame is dropped); ``n_frames`` of them enter ``e_par``.
    """

    e_par: float
    gauge_part: float
    off_orbit_part: float
    lam: float
    z_rms: float
    oracle_floor: float
    e_perp: float
    e_3d: float
    n_frames: int
    reproj_rms_px: float = float("nan")
    n_reproj_frames: int = 0
    aligned: dict[str, float] = field(default_factory=dict)
    shift_part: float = float("nan")
    shift_scale_part: float = float("nan")
    collinearity: float = float("nan")
    low_n: bool = False
    aligned_n_frames: dict[str, int] = field(default_factory=dict)
    n_frames_input: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Plain-Python dict (JSON-serialisable; NaN stays NaN)."""
        return asdict(self)


@dataclass(frozen=True)
class Score:
    """Per-body scores keyed by body name (``"object"``, ``"hand"``)."""

    bodies: dict[str, BodyScore]

    def __getitem__(self, name: str) -> BodyScore:
        return self.bodies[name]

    def as_dict(self) -> dict[str, Any]:
        """``{body: BodyScore.as_dict()}``."""
        return {k: v.as_dict() for k, v in self.bodies.items()}


_KINDS = ("any", "object", "hand")


def reference_points(x: np.ndarray, root: int = 0, kind: str = "any") -> np.ndarray:
    """(T, 3) reference trajectory from (T, 3) points, (T, 4, 4) poses (translation) or (T, J, 3) joints (``root``).

    ``kind="object"`` accepts only (T, 3) and (T, 4, 4); ``kind="hand"`` only (T, 3) and (T, J, 3). The bottom row
    of every finite (4, 4) pose must be ``[0, 0, 0, 1]``.

    Raises:
        ValueError: any other shape, a bad pose bottom row, or ``root`` outside ``[0, J)``.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 3:
        return a
    if a.ndim == 3 and a.shape[1:] == (4, 4) and kind != "hand":
        fin = np.isfinite(a).all(axis=(1, 2))
        if not np.allclose(a[fin, 3, :], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9):
            raise ValueError("a (4, 4) pose's bottom row is not [0, 0, 0, 1]; pass object-to-camera rigid poses")
        return a[:, :3, 3]
    if a.ndim == 3 and a.shape[2] == 3 and kind != "object":
        if not 0 <= root < a.shape[1]:
            raise ValueError(f"root index {root} is outside [0, {a.shape[1]}) joints")
        return a[:, root]
    allowed = {"any": "(T,3) points, (T,4,4) poses or (T,J,3) joints", "object": "(T,3) centres or (T,4,4) poses",
               "hand": "(T,3) wrists or (T,J,3) joints"}[kind]
    raise ValueError(f"expected {allowed}, got shape {a.shape}")


def intrinsics_matrix(k: np.ndarray | Sequence[float]) -> np.ndarray:
    """A (3, 3) pinhole matrix from a (3, 3) matrix or ``(fx, fy, cx, cy)``.

    The matrix must be finite, have positive focal lengths, ``K[1, 0] = 0`` and the canonical bottom row
    ``[0, 0, 1]``: the reprojection divides by the camera-frame depth z, which is the homogeneous coordinate only
    for that row. A skew ``K[0, 1]`` is allowed.

    Raises:
        ValueError: any other shape, a non-finite entry, non-positive focal lengths, a non-zero ``K[1, 0]`` or a
            bottom row other than ``[0, 0, 1]``.
    """
    a = np.asarray(k, dtype=np.float64)
    if a.shape == (4,):
        if not np.isfinite(a).all():
            raise ValueError(f"intrinsics must be finite, got (fx, fy, cx, cy) = {tuple(a.tolist())}")
        fx, fy, cx, cy = a
        a = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if a.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3,3) or (fx, fy, cx, cy), got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("intrinsics must be finite")
    if not np.array_equal(a[2], [0.0, 0.0, 1.0]):
        raise ValueError(f"the intrinsics' bottom row must be [0, 0, 1], got {a[2].tolist()}: the reprojection "
                         "divides by camera z")
    if a[1, 0] != 0.0:
        raise ValueError(f"K[1, 0] must be 0 for a pinhole matrix, got {a[1, 0]!r}")
    if not (a[0, 0] > 0 and a[1, 1] > 0):
        raise ValueError("focal lengths must be positive")
    return a


def reprojection_error_px(pred: np.ndarray, gt: np.ndarray, k: np.ndarray | Sequence[float]) -> np.ndarray:
    """Per-frame pixel distance between the projections of (T, 3) camera-frame points (NaN if either z <= 0)."""
    kk = intrinsics_matrix(k)
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape != g.shape:
        raise ValueError(f"pred and gt must be (T, 3) arrays of equal T, got {p.shape} and {g.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        up = (p @ kk.T)[:, :2] / p[:, 2:3]
        ug = (g @ kk.T)[:, :2] / g[:, 2:3]
    err = np.linalg.norm(up - ug, axis=1)
    err[~((p[:, 2] > 0) & (g[:, 2] > 0))] = np.nan
    return np.asarray(err, dtype=np.float64)


def _is_joints(x: np.ndarray) -> bool:
    """True for a (T, J, 3)-like 3-D array that is not a stack of (4, 4) poses."""
    shape = np.shape(x)
    return len(shape) == 3 and shape[1:] != (4, 4)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")


def _ate_sim3(p: np.ndarray, g: np.ndarray) -> float:
    """RMS centre error after one similarity over the trajectory (NaN when it is undefined)."""
    if len(p) < 3:
        return float("nan")
    try:
        s, r, t = umeyama_similarity(p, g)
    except ValueError:
        return float("nan")
    return _rms(np.linalg.norm(s * p @ r.T + t - g, axis=1))


def _frame_join(pred_ids: Any, gt_ids: Any, n_pred: int, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """Row orders that join pred and gt on their frame ids, in increasing id order (module docstring).

    Raises:
        ValueError: ids that are not one integer per row, duplicate ids, or id sets that differ.
    """
    fp, fg = _ids(pred_ids, n_pred, "pred"), _ids(gt_ids, n_gt, "gt")
    op, og = np.argsort(fp, kind="stable"), np.argsort(fg, kind="stable")
    if not np.array_equal(fp[op], fg[og]):
        only_p, only_g = np.setdiff1d(fp, fg), np.setdiff1d(fg, fp)
        raise ValueError(f"pred and gt frame ids must be the same set (no intersection, no trimming, no shift): "
                         f"{only_p.size} id(s) only in pred {only_p[:5].tolist()}, {only_g.size} only in gt "
                         f"{only_g[:5].tolist()}")
    return op, og


def _n_rows(x: Any) -> int:
    return int(np.shape(x)[0]) if np.ndim(x) else 0


def score_body(pred: np.ndarray, gt: np.ndarray, k: np.ndarray | Sequence[float] | None = None, *,
               root: int = 0, kind: str = "any", pred_points: Sequence[np.ndarray] | np.ndarray | None = None,
               gt_points: Sequence[np.ndarray] | np.ndarray | None = None, pred_ids: Any = None,
               gt_ids: Any = None) -> BodyScore:
    """Score one body. ``pred`` / ``gt``: (T, 3), (T, 4, 4) or (T, J, 3), camera frame, metres (``kind`` restricts
    the shapes, see :func:`reference_points`). Warns when intrinsics are given but no frame has z > 0 in both.

    ``aligned`` holds, for contrast:
        ``centre_rms_after_lam``: 3-D RMS of ``c_pred - lam c_gt`` (the trajectory after the per-body scale);
        ``centre_ate_sim3``: RMS centre error after one similarity (scale, rotation, translation) over the frames;
        ``mpjpe_root_aligned``, ``pa_mpjpe``: mean over frames, when (T, J, 3) joints are given;
        ``chamfer_cm2_after_scaled_icp``: when ``pred_points`` / ``gt_points`` (per-frame (V, 3) clouds) are given,
        one centred scaled ICP over the frames finite in both (NaN when there is none).
    ``aligned_n_frames`` gives each of these its own frame count.

    ``pred_ids`` / ``gt_ids`` (both or neither): one integer frame id per row of ``pred`` / ``gt`` (and of the
    clouds); the rows are joined on them (module docstring). Without them the rows are taken as synchronized.

    Raises:
        ValueError: bad shapes, bad or mismatched frame ids, no frame is finite in both, a true point sits at the
            camera centre, or the per-body scale is not finite and positive.
    """
    if (pred_ids is None) != (gt_ids is None):
        raise ValueError("pass both pred_ids and gt_ids, or neither")
    if pred_ids is not None:
        op, og = _frame_join(pred_ids, gt_ids, _n_rows(pred), _n_rows(gt))
        pred, gt = np.asarray(pred)[op], np.asarray(gt)[og]
        if pred_points is not None and gt_points is not None and len(pred_points) == len(op) \
                and len(gt_points) == len(og):
            pred_points, gt_points = [pred_points[i] for i in op], [gt_points[i] for i in og]
    if (_is_joints(pred) or _is_joints(gt)) and np.shape(pred) != np.shape(gt):
        raise ValueError(f"hand joints must have the same (T, J, 3) shape in pred and gt, got {np.shape(pred)} "
                         f"and {np.shape(gt)} (same joint set and order on both sides)")
    p_all = reference_points(pred, root, kind)
    g_all = reference_points(gt, root, kind)
    if p_all.shape != g_all.shape:
        raise ValueError(f"pred {np.shape(pred)} and gt {np.shape(gt)} must have the same number of frames")
    if np.any(np.all(p_all == 0.0, axis=1)):
        warnings.warn("a predicted reference point is exactly (0, 0, 0): missing frames should be NaN, not "
                      "zero-filled", UserWarning, stacklevel=2)
    sp = e_par_split(p_all, g_all)
    ok = np.isfinite(p_all).all(1) & np.isfinite(g_all).all(1)
    p, g = p_all[ok], g_all[ok]
    ar = along_ray_components(p, g)
    aligned = {"centre_rms_after_lam": gauge_decomposition(p, g).off_orbit_rms, "centre_ate_sim3": _ate_sim3(p, g)}
    n_al = {k: (len(p) if np.isfinite(v) else 0) for k, v in aligned.items()}
    pj, gj = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pj.ndim == 3 and pj.shape[1:] != (4, 4) and pj.shape[1] > 1:
        for name, per_frame in (("mpjpe_root_aligned", mpjpe_root_aligned(pj, gj, root)),
                                ("pa_mpjpe", pa_mpjpe(pj, gj))):
            aligned[name] = _nanmean(per_frame)
            n_al[name] = int(np.isfinite(per_frame).sum())
    if (pred_points is None) != (gt_points is None):
        raise ValueError("pass both pred_points and gt_points, or neither")
    if pred_points is not None and gt_points is not None:
        if not (len(pred_points) == len(gt_points) == p_all.shape[0]):
            raise ValueError(f"pred_points and gt_points must have one cloud per frame each: {p_all.shape[0]} "
                             f"frames, got {len(pred_points)} and {len(gt_points)} clouds")
        keep = [t for t in range(len(gt_points))
                if np.isfinite(pred_points[t]).all() and np.isfinite(gt_points[t]).all()]
        cd = float("nan")
        if keep:
            cd, _ = object_cd_after_scaled_icp([pred_points[t] for t in keep], [gt_points[t] for t in keep])
        aligned["chamfer_cm2_after_scaled_icp"] = cd
        n_al["chamfer_cm2_after_scaled_icp"] = len(keep) if np.isfinite(cd) else 0
    reproj, n_reproj = float("nan"), 0
    if k is not None:
        err = reprojection_error_px(p, g, k)
        n_reproj = int(np.isfinite(err).sum())
        reproj = _rms(err)
        if n_reproj == 0:
            warnings.warn("no frame projects with z > 0 in both pred and gt: the reprojection needs +z forward "
                          "(OpenCV camera axes)", UserWarning, stacklevel=2)
    return BodyScore(e_par=sp.e_par, gauge_part=sp.gauge_part, off_orbit_part=sp.off_orbit_part, lam=sp.lam,
                     z_rms=sp.z_rms, oracle_floor=sp.oracle_floor, e_perp=_rms(ar.perp), e_3d=_rms(ar.total),
                     n_frames=sp.n_frames, reproj_rms_px=reproj, n_reproj_frames=n_reproj,
                     aligned=aligned, shift_part=sp.shift_part, shift_scale_part=sp.shift_scale_part,
                     collinearity=sp.collinearity, low_n=sp.low_n, aligned_n_frames=n_al,
                     n_frames_input=int(p_all.shape[0]))


def score(pred_object: np.ndarray, gt_object: np.ndarray, pred_hand: np.ndarray | None = None,
          gt_hand: np.ndarray | None = None, k: np.ndarray | Sequence[float] | None = None, *,
          hand_root: int = 0, pred_object_points: Sequence[np.ndarray] | np.ndarray | None = None,
          gt_object_points: Sequence[np.ndarray] | np.ndarray | None = None, pred_ids: Any = None,
          gt_ids: Any = None) -> Score:
    """Score an object and (optionally) a hand. Each body gets its own ``lam``; see :func:`score_body`.

    Args:
        pred_object, gt_object: (T, 3) object centres or (T, 4, 4) object-to-camera poses, metres; nothing else.
        pred_hand, gt_hand: (T, 3) wrists or (T, J, 3) joints, metres; both or neither.
        k: intrinsics, (3, 3) or (fx, fy, cx, cy); only the reprojection contrast uses them (``e_par`` and its
            split need only camera-frame coordinates, whose rays all pass through the origin).
        hand_root: root joint index for (T, J, 3) hands.
        pred_object_points, gt_object_points: optional per-frame object clouds for the aligned Chamfer.
        pred_ids, gt_ids: optional integer frame ids, one per row of every prediction / truth array (both or
            neither). The two sets must be identical, without duplicates; rows are joined on them in increasing id
            order. Without them row t of every array is taken to be the same frame (synchronized row by row).

    Raises:
        ValueError: only one of the two hand arrays (or id arrays) is given, or a body fails :func:`score_body`.
    """
    if (pred_hand is None) != (gt_hand is None):
        raise ValueError("pass both pred_hand and gt_hand, or neither")
    bodies = {"object": score_body(pred_object, gt_object, k, kind="object", pred_points=pred_object_points,
                                   gt_points=gt_object_points, pred_ids=pred_ids, gt_ids=gt_ids)}
    if pred_hand is not None and gt_hand is not None:
        bodies["hand"] = score_body(pred_hand, gt_hand, k, root=hand_root, kind="hand", pred_ids=pred_ids,
                                    gt_ids=gt_ids)
    return Score(bodies=bodies)
