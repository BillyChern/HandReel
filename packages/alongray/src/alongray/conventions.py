# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Convention handling for :mod:`alongray.coverage`: conversion to one canonical frame and empirical checks.

Canonical frame: metres, OpenCV camera axes (x right, y down, z forward), frame IDs in the protocol's index base.
A method's arrays are converted only from conventions it DECLARES. Undeclared or unrecognised conventions are never
guessed: the affected arrays are left out of the empirical checks and the report says UNKNOWN.

The empirical checks run on frames that two tracks share. They detect conventions that were declared wrongly; they do
not score either method, and every threshold below is a screening constant, not a measurement confidence:

- unit or scale: the median per-frame ratio ``|p_b| / |p_a|``; flagged when ``|log10 ratio| >= UNIT_LOG10_TOL``
  (a factor of about 3.2, far beyond the depth errors of ordinary reconstructions, well inside a mm/m or cm/m swap);
- frame or transform: the median ``|p_b - p_a|`` over the median ``|p_a|``; flagged above ``DISCREPANCY_TOL``;
- object origin (needs rotations in one object frame): two constant offsets are fitted to the position difference,
  one in the object frame (``r_i = R_a,i^T (p_b,i - p_a,i)``, ``d = mean r``) and one in the camera frame (``c_i =
  p_b,i - p_a,i``). The origin is flagged only when the object-frame fit explains at least ``ORIGIN_EXPLAINED`` of
  ``sum |r|^2``, ``|d| >= ORIGIN_TOL_M``, AND its residual is below ``ORIGIN_WIN_RATIO`` times the camera-frame
  residual. A unit swap, an axis flip, a global scale, a mirror or a camera-frame offset is fitted at least as well
  in the camera frame, and without rotation spread the two fits cannot be told apart; none of these is an origin;
- rotation (only when the protocol lists ``object_rotation`` under ``modality.compare``, which declares both
  methods' rotations to be in one object frame): the median geodesic angle of ``R_a^T R_b``, symmetry-reduced where
  the protocol declares a group, flagged above ``ROTATION_TOL_DEG`` (a quaternion-order or axis-convention error);
- symmetry: per-frame geodesic angle of ``R_a^T R_b``, raw and minimised over the protocol's symmetry group; flagged
  when the two medians differ by more than ``SYMMETRY_TOL_DEG``;
- camera axes (per track): the share of positions in front of the camera (``z > 0``) below ``FRONT_SHARE``.

Not caught, by design of the thresholds: a few-percent global scale (a 3% scale gives ratio 1.03 and no flag; even
3.0x escapes ``UNIT_OR_SCALE_SUSPECT`` and reads as ``FRAME_OR_TRANSFORM_SUSPECT``), a mirror or a constant
camera-frame offset below ``DISCREPANCY_TOL``, and rotation disagreements below ``ROTATION_TOL_DEG``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["DISCREPANCY_TOL", "FRONT_SHARE", "ORIGIN_EXPLAINED", "ORIGIN_TOL_M", "ORIGIN_WIN_RATIO", "ROTATION_TOL_DEG",
           "SYMMETRY_TOL_DEG", "UNIT_LOG10_TOL", "UNIT_TO_M", "Track", "angle_deg", "pair_residuals", "summarise_pair",
           "to_canonical"]

UNIT_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
GL_TO_CV = np.diag([1.0, -1.0, -1.0])
SO3_TOL = 1e-5
UNIT_LOG10_TOL = 0.5
DISCREPANCY_TOL = 0.5
ORIGIN_TOL_M = 0.005
ORIGIN_EXPLAINED = 0.9
ORIGIN_WIN_RATIO = 0.25
ROTATION_TOL_DEG = 60.0
SYMMETRY_TOL_DEG = 5.0
FRONT_SHARE = 0.95
CLOCK_TOL_FRAMES = 0.5


@dataclass(frozen=True)
class Track:
    """One unit's delivered frames in the canonical frame (``position``/``rotation`` None when not usable)."""

    frame_ids: np.ndarray
    position: np.ndarray | None
    rotation: np.ndarray | None
    notes: tuple[str, ...] = field(default_factory=tuple)
    roundtrip_m: float = 0.0
    nonfinite: int = 0


def angle_deg(r: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees of rotation matrices (..., 3, 3), stable near 0 and near 180 degrees."""
    fro = np.linalg.norm(r - np.eye(3), axis=(-2, -1))
    return np.degrees(2.0 * np.arcsin(np.clip(fro / (2.0 * np.sqrt(2.0)), 0.0, 1.0)))


def _is_so3(r: np.ndarray) -> bool:
    eye = np.broadcast_to(np.eye(3), r.shape)
    return bool(np.allclose(np.swapaxes(r, -1, -2) @ r, eye, atol=SO3_TOL)
                and np.allclose(np.linalg.det(r), 1.0, atol=SO3_TOL))


def _rigid(m: np.ndarray) -> bool:
    return (m.ndim == 3 and m.shape[1:] == (4, 4) and _is_so3(m[:, :3, :3])
            and np.allclose(m[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=SO3_TOL))


def _clock_note(frames: np.ndarray, stamps: np.ndarray | None, fps: float | None) -> str | None:
    if stamps is None or fps is None or len(frames) < 2:
        return None
    if stamps.shape != frames.shape or not np.isfinite(stamps).all():
        raise ValueError("timestamps must be finite and have one entry per frame id")
    implied = (stamps - stamps[0]) * fps + frames[0]
    worst = float(np.max(np.abs(implied - frames)))
    return None if worst <= CLOCK_TOL_FRAMES else f"CLOCK_MISMATCH: timestamps imply frame ids up to {worst:.2f} " \
                                                  f"frames away from the declared ids at {fps:g} fps"


def to_canonical(arrays: Mapping[str, np.ndarray], conv: Mapping[str, Any], protocol_base: int,
                 fps: float | None = None) -> Track:
    """Convert one unit's arrays (keys ``frame_ids``, optional ``object_position``, ``object_rotation``,
    ``camera_from_world``, ``timestamps``) to the canonical frame using only declared conventions.

    Raises:
        ValueError: malformed arrays (shapes, non-integer or duplicate frame ids, non-finite timestamps).
    """
    if "frame_ids" not in arrays:
        raise ValueError("unit file has no 'frame_ids' array")
    raw = np.asarray(arrays["frame_ids"])
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("'frame_ids' must be a 1-D integer array")
    if len(np.unique(raw)) != len(raw):
        raise ValueError("duplicate frame ids inside one unit file are refused (never deduplicated silently)")
    notes: list[str] = []
    base = conv.get("frame_index_base")
    if base not in (0, 1):
        notes.append("frame_index_base undeclared: ids taken as given")
        base = protocol_base
    frames = raw.astype(np.int64) - int(base) + int(protocol_base)
    clock = _clock_note(frames, None if "timestamps" not in arrays else np.asarray(arrays["timestamps"], float), fps)
    if clock:
        notes.append(clock)
    n = len(frames)
    pos = np.asarray(arrays["object_position"], float) if "object_position" in arrays else None
    rot = np.asarray(arrays["object_rotation"], float) if "object_rotation" in arrays else None
    if pos is not None and pos.shape != (n, 3):
        raise ValueError(f"object_position must be ({n}, 3), got {pos.shape}")
    if rot is not None and rot.shape != (n, 3, 3):
        raise ValueError(f"object_rotation must be ({n}, 3, 3), got {rot.shape}")
    finite = np.ones(n, bool)
    for a in (pos, rot):
        if a is not None:
            finite &= np.isfinite(a.reshape(n, int(np.prod(a.shape[1:])))).all(1)
    frames, nonfinite = frames[finite], int((~finite).sum())
    pos = None if pos is None else pos[finite]
    rot = None if rot is None else rot[finite]
    if rot is not None and len(rot) and not _is_so3(rot):
        notes.append("ROTATION_NOT_SO3: rotations left out")
        rot = None
    factor = UNIT_TO_M.get(str(conv.get("length_unit")))
    if factor is None:
        notes.append(f"length_unit {conv.get('length_unit')!r} unknown: positions left out")
        pos = None
    roundtrip = 0.0
    frame = conv.get("frame")
    if frame == "world":
        if "camera_from_world" not in arrays:
            raise ValueError("frame 'world' needs a 'camera_from_world' (N, 4, 4) array")
        m = np.asarray(arrays["camera_from_world"], float)
        if m.shape != (n, 4, 4):
            raise ValueError(f"camera_from_world must be ({n}, 4, 4), got {m.shape}")
        m = m[finite]
        if not _rigid(m):
            notes.append("CAMERA_TRANSFORM_NOT_RIGID: positions and rotations left out")
            pos = rot = None
        else:
            rc, tc = m[:, :3, :3], m[:, :3, 3] * (factor or 1.0)
            if pos is not None:
                pw = pos * factor
                pc = np.einsum("nij,nj->ni", rc, pw) + tc
                back = np.einsum("nji,nj->ni", rc, pc - tc)
                roundtrip = float(np.max(np.abs(back - pw))) if len(pw) else 0.0
                pos = pc / factor
            rot = None if rot is None else rc @ rot
    elif frame != "camera":
        notes.append(f"frame {frame!r} unknown: positions and rotations left out")
        pos = rot = None
    axes = conv.get("camera_axes")
    if axes == "opengl":
        pos = None if pos is None else pos @ GL_TO_CV.T
        rot = None if rot is None else GL_TO_CV @ rot
    elif axes != "opencv":
        notes.append(f"camera_axes {axes!r} unknown: positions and rotations left out")
        pos = rot = None
    if pos is not None:
        pos = pos * factor
        if len(pos) and float(np.mean(pos[:, 2] > 0)) < FRONT_SHARE:
            notes.append(f"CAMERA_AXES_SUSPECT: only {np.mean(pos[:, 2] > 0):.2f} of positions have z > 0 in the "
                         "canonical OpenCV frame")
    return Track(frames, pos, rot, tuple(notes), roundtrip, nonfinite)


def pair_residuals(pa: np.ndarray | None, pb: np.ndarray | None, ra: np.ndarray | None, rb: np.ndarray | None,
                   group: Sequence[np.ndarray] | None) -> dict[str, np.ndarray]:
    """Per-frame residuals of two tracks on the same frames (arrays aligned row by row)."""
    out: dict[str, np.ndarray] = {}
    if pa is not None and pb is not None:
        na, nb = np.linalg.norm(pa, axis=1), np.linalg.norm(pb, axis=1)
        ok = (na > 0) & (nb > 0)
        out["log10_ratio"] = np.log10(nb[ok] / na[ok])
        out["abs_diff"], out["norm_a"] = np.linalg.norm(pb - pa, axis=1), na
        if ra is not None:
            out["offset_obj"] = np.einsum("nji,nj->ni", ra, pb - pa)
            out["offset_cam"] = pb - pa
    if ra is not None and rb is not None:
        rel = np.swapaxes(ra, -1, -2) @ rb
        out["raw_deg"] = angle_deg(rel)
        if group:
            out["sym_deg"] = np.min([angle_deg(np.swapaxes(ra @ g, -1, -2) @ rb) for g in group], axis=0)
    return out


def _origin(r: np.ndarray, c: np.ndarray) -> tuple[dict[str, Any], bool]:
    """Constant object-frame (``r``) against constant camera-frame (``c``) offset fits of one object's rows."""
    d = r.mean(0)
    tot = float((r * r).sum())
    res_obj, res_cam = float(((r - d) ** 2).sum()), float(((c - c.mean(0)) ** 2).sum())
    explained = 1.0 if tot == 0 else 1.0 - res_obj / tot
    wins = res_obj < ORIGIN_WIN_RATIO * res_cam - 1e-9 * tot
    out = {"offset_mm": (d * 1e3).tolist(), "norm_mm": float(np.linalg.norm(d) * 1e3), "explained": explained,
           "object_vs_camera_residual": res_obj / res_cam if res_cam > 0 else None}
    return out, bool(wins and explained >= ORIGIN_EXPLAINED and np.linalg.norm(d) >= ORIGIN_TOL_M)


def summarise_pair(res: Mapping[str, list[np.ndarray]], compare_rotation: bool = False
                   ) -> tuple[dict[str, Any], list[str]]:
    """Pool per-unit residual lists (``res[key]`` is a list of arrays; ``offset_obj`` and ``offset_cam`` are pooled
    per object by the caller and passed as ``offset_by_object`` and ``camera_offset_by_object``) into a summary and
    its flags. ``compare_rotation``: the protocol declares rotations comparable (one object frame)."""
    summ: dict[str, Any] = {}
    flags: list[str] = []
    cat = {k: np.concatenate(v) for k, v in res.items() if not k.endswith("_by_object") and v}
    if "log10_ratio" in cat and cat["log10_ratio"].size:
        lr = float(np.median(cat["log10_ratio"]))
        summ["median_scale_ratio"] = 10.0 ** lr
        if abs(lr) >= UNIT_LOG10_TOL:
            flags.append(f"UNIT_OR_SCALE_SUSPECT: median |p_b|/|p_a| = {10.0 ** lr:.4g}")
        disc = float(np.median(cat["abs_diff"]) / max(float(np.median(cat["norm_a"])), 1e-300))
        summ["median_relative_discrepancy"] = disc
        if disc > DISCREPANCY_TOL and abs(lr) < UNIT_LOG10_TOL:
            flags.append(f"FRAME_OR_TRANSFORM_SUSPECT: median |p_b - p_a| / |p_a| = {disc:.3g}")
    cams = dict(res.get("camera_offset_by_object", {}))
    for obj, offs in sorted(dict(res.get("offset_by_object", {})).items()):
        r = np.concatenate(offs)
        if not len(r):
            continue
        o, hit = _origin(r, np.concatenate(cams[obj]))
        summ.setdefault("origin_offset_mm", {})[obj] = o
        if hit:
            flags.append(f"ORIGIN_OFFSET_SUSPECT: object {obj}: constant object-frame offset {o['norm_mm']:.1f} mm "
                         f"explains {o['explained']:.3f} of the position difference and fits better than a constant "
                         "camera-frame offset")
    if "raw_deg" in cat and cat["raw_deg"].size:
        summ["median_rotation_deg"] = float(np.median(cat["raw_deg"]))
        if "sym_deg" in cat:
            summ["median_rotation_deg_symmetry_reduced"] = float(np.median(cat["sym_deg"]))
            if summ["median_rotation_deg"] - summ["median_rotation_deg_symmetry_reduced"] > SYMMETRY_TOL_DEG:
                flags.append("SYMMETRY_SENSITIVE: rotation agreement depends on the symmetry handling (raw "
                             f"{summ['median_rotation_deg']:.1f} deg, reduced "
                             f"{summ['median_rotation_deg_symmetry_reduced']:.1f} deg)")
        ang = summ.get("median_rotation_deg_symmetry_reduced", summ["median_rotation_deg"])
        if compare_rotation and ang > ROTATION_TOL_DEG:
            flags.append(f"ROTATION_DISAGREEMENT: median geodesic angle {ang:.1f} deg > {ROTATION_TOL_DEG:g} deg with "
                         "rotations declared in one object frame (quaternion order or axis convention?)")
    return summ, flags
