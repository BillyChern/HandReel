# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""Loads per-frame MoGe camera-frame pointmaps produced by the baseline's ``get_pointmap_dir.py``.

``get_pointmap_dir.py`` (``third_party/do-as-i-do/reconstruction/scripts``) writes one
``<name>_pointmap.npy`` (H,W,3 float32, camera-frame XYZ) per input frame. Its two modes use
different naming: single-image mode (``--image``) keeps the input's own stem (e.g.
``0125_pointmap.npy``); batch mode (``--image_dir``) numbers frames from the directory's own
filenames, which for this repo's ``all_frames/`` extraction is 6-digit zero-padded (e.g.
``000000_pointmap.npy``). This loader tries the batch (6-digit), single-image (4-digit) and bare
integer namings, in that order, so it works against either output layout.

MoGe itself (verified on the whisking demo's output) does
not encode per-pixel invalidity in the saved array -- every pixel gets a dense estimate, no NaN,
no sentinel. The adapter contract still calls for NaN-marked invalid pixels, so this loader (a)
propagates any non-finite values already present in the array to NaN across all 3 channels, and
(b) additionally honours an optional companion ``<name>_mask.npy`` / ``<name>_valid.npy`` boolean
array (same H,W) when present, marking pixels false there as NaN too -- for MoGe outputs that
carry no such companion file (the common case here), only (a) applies.

**``load_pointmaps`` returns raw MoGe scale, not metres.** MoGe's monocular depth has an unknown
global scale ambiguity per clip; ``load_pointmaps`` passes the array through unchanged, so any
downstream code that assumes metric depth (e.g. ``fastdo.solver.init.scale_from_extent``, which
compares a pointmap's observed extent against a canonical shape's diameter in metres) is wrong
unless the pointmap has first been corrected. ``hand_anchored_scale`` + ``apply_metric_scale``
below reproduce the baseline's own correction (``reconstruction/scripts/
optimize_translation_scale.py``) for exactly this purpose: apply ``apply_metric_scale(pointmaps,
hand_anchored_scale(...))`` before treating a loaded pointmap as metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "apply_metric_scale",
    "clip_median_scale",
    "hand_anchor_pixels",
    "hand_anchored_scale",
    "hand_anchored_scale_from_samples",
    "load_pointmaps",
]


def load_pointmaps(directory: Path, frame_idx: Sequence[int]) -> Tensor:
    """Loads and stacks per-frame pointmaps.

    Args:
        directory: Directory holding ``<name>_pointmap.npy`` files (and optional
            ``<name>_mask.npy``/``<name>_valid.npy`` companions).
        frame_idx: Frame indices to load, in the order they should appear in the output.

    Returns:
        ``(N,H,W,3)`` float32, camera-frame XYZ, with NaN at invalid pixels. **Raw MoGe scale,
        not metres** -- see the module docstring; apply ``hand_anchored_scale`` +
        ``apply_metric_scale`` before any metric use.

    Raises:
        ValueError: If ``frame_idx`` is empty, a loaded array is not ``(H,W,3)``, or the loaded
            pointmaps do not all share the same shape.
        FileNotFoundError: If no pointmap file is found for a requested frame index.
    """
    directory = Path(directory)
    if not frame_idx:
        raise ValueError("frame_idx is empty")

    maps: list[np.ndarray] = []
    for i in frame_idx:
        pointmap_path = _resolve_pointmap_path(directory, i)
        arr = np.load(pointmap_path).astype(np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{pointmap_path}: expected (H,W,3), got {arr.shape}")
        arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)

        mask_path = _companion_mask_path(pointmap_path)
        if mask_path is not None:
            valid = np.load(mask_path).astype(bool)
            arr = np.where(valid[..., None], arr, np.nan).astype(np.float32)
        maps.append(arr)

    shapes = {m.shape for m in maps}
    if len(shapes) > 1:
        raise ValueError(f"{directory}: pointmaps for {list(frame_idx)} have inconsistent shapes: {shapes}")

    return torch.from_numpy(np.stack(maps, axis=0))


def _resolve_pointmap_path(directory: Path, i: int) -> Path:
    for name in (f"{i:06d}_pointmap.npy", f"{i:04d}_pointmap.npy", f"{i}_pointmap.npy"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no pointmap for frame {i} under {directory} (tried 6-digit, 4-digit and bare naming)")


def _companion_mask_path(pointmap_path: Path) -> Path | None:
    base = pointmap_path.stem.removesuffix("_pointmap")
    for suffix in ("_mask.npy", "_valid.npy"):
        candidate = pointmap_path.with_name(base + suffix)
        if candidate.is_file():
            return candidate
    return None


def hand_anchored_scale(pointmaps: Tensor, hand_joints_cam: Tensor, intrinsics: Tensor,
                        valid: Tensor, depth_scale: Tensor | None = None) -> Tensor:
    """Per-frame pointmap-to-metric scale factor, hand-anchored (raw MoGe scale -> metres).

    Reproduces ``reconstruction/scripts/optimize_translation_scale.py``'s core formula (module
    docstring lines 10-16; per-frame loop lines 453-464): treat a HaWoR hand as ground truth,
    compare its depth against the *same* depth read off the pointmap at the hand's own pixels, and
    take the ratio. The baseline does this with per-pixel raycasting through a hand mask against
    the full HaWoR hand mesh (``h_real``/``h_pm`` = mean over raycast-hit pixels, lines 441-454);
    this adapter has no hand mesh/mask/raycasting machinery, so it uses HaWoR's 3D joints directly
    as the "real" anchor points and projects them into the pointmap's pixel grid instead of
    raycasting -- same statistic (mean of the *z* component, since only ``h_real[2]`` /
    ``h_pm[2]`` feed the baseline's ``k = h_real_z / h_pm_z``, line 464; taking the mean of the
    z-components directly is identical to taking the z-component of the mean vector), same
    formula, coarser anchor set (~21 joints vs. thousands of raycast hits).

    Per-joint inclusion in a frame's mean (all deviations from the baseline documented, since the
    baseline's raycast hits have no equivalent failure mode):
    - ``valid[t, j]`` must be True and ``hand_joints_cam[t, j]`` must be finite.
    - The joint's depth (``Z``) must be positive (a joint behind the camera has no sane pixel
      projection).
    - Projecting with ``project_to_pixels``'s own formula (``u = int(fx*X/Z + cx)``, ``v =
      int(fy*Y/Z + cy)``, truncating like ``.astype(int)`` at lines 212-213) must land *inside*
      the pointmap's ``(H,W)`` bounds. The baseline never needs this check -- its raycast hits
      only ever come from pixels already inside a hand mask, hence already in-bounds -- but a
      HaWoR joint can legitimately project outside the frame (e.g. an occluded/out-of-view hand),
      so an out-of-bounds joint is excluded here rather than clamped into the image like the
      baseline's mesh-vertex projection does (clamping would silently read an unrelated pixel).
    - The pointmap sample at that pixel must be finite (this adapter's own ``load_pointmaps`` can
      produce NaN pixels; the baseline's dense pointmaps never did, so it has no such check).

    Fallback when a frame has zero usable joints, or when ``abs(h_pm_z) < 1e-6`` (the exact
    degenerate-depth guard at lines 355/460, there causing the baseline to ``skip`` -- i.e. leave
    that frame's scale unset rather than fabricate one): this frame's output is NaN, the direct
    analogue of "unset" for a dense ``(T,)`` tensor. Callers must decide their own policy for NaN
    frames (e.g. nearest valid neighbour), matching the baseline leaving them for a later stage.

    Args:
        pointmaps: ``(T,H,W,3)`` raw (non-metric) camera-frame pointmaps, as returned by
            ``load_pointmaps``.
        hand_joints_cam: ``(T,J,3)`` camera-frame hand joints in metres (e.g.
            ``HandTrack.joints``).
        intrinsics: ``(3,3)`` pixel intrinsics matching the pointmaps' pixel grid.
        valid: ``(T,J)`` bool, per-joint validity (e.g. ``HandTrack.valid`` broadcast, or a
            per-joint mask if available).
        depth_scale: ``(J,)`` per-joint factor ``f_j`` (the per-side hand-depth correction), or None (the
            default: the formula above, unchanged). Each joint is treated as rescaled about the camera centre,
            ``X_j -> f_j * X_j``: its pixel is unchanged (``fx * f X / (f Z) = fx * X / Z``) and its depth becomes
            ``f_j * Z_j``, so with one ``f`` for every joint each frame's ``k`` is multiplied by exactly ``f``. See
            ``hand_anchor_pixels``.

    Returns:
        ``(T,)`` float32; ``pointmaps * k[:, None, None, None]`` (see ``apply_metric_scale``) is
        the metric pointmap. NaN where no scale could be estimated for that frame.
    """
    t, h, w = (int(d) for d in pointmaps.shape[:3])
    frame, v, u, z = hand_anchor_pixels(hand_joints_cam, intrinsics, valid, (t, h, w), depth_scale=depth_scale)
    return hand_anchored_scale_from_samples(t, frame, z, pointmaps[frame, v, u].cpu())


def hand_anchor_pixels(hand_joints_cam: Tensor, intrinsics: Tensor, valid: Tensor,
                       image_shape: tuple[int, int, int],
                       depth_scale: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The pixels of ``(T,H,W)`` = ``image_shape`` pointmaps that ``hand_anchored_scale`` reads: every joint of
    frames ``0..T-1`` that passes its checks before the read.

    Split out of ``hand_anchored_scale`` so a caller holding only those pixels of the pointmaps (the
    geometry stage gathers them on the GPU instead of copying whole frames to the host) gets the same scale.

    With ``depth_scale`` (``(J,)``, one factor per joint column), every check and pixel is computed from the
    recorded joints exactly as without it, and only the returned depth is ``f_j * z``. That is the joint rescaled
    about the camera centre (``X -> f X`` projects to the same pixel), in exact arithmetic: rescaling the float32
    joints first would round ``fX / fZ`` and could move a truncated pixel by one. ``None`` (the default) returns
    ``z`` itself, the path every existing run took.

    Returns:
        ``(frame, v, u, z)``, each ``(P,)``: int64 frame, row and column of every valid, finite, in-front,
        in-bounds joint, in (frame, joint) order, and float64 ``z``, the joint's depth exactly as the scale's
        mean reads it. ``hand_anchored_scale_from_samples(T, frame, z, pointmaps[frame, v, u])`` is
        ``hand_anchored_scale(pointmaps, ...)``.
    """
    t, height, width = image_shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    frames: list[int] = []
    vs: list[int] = []
    us: list[int] = []
    zs: list[float] = []
    factors = None if depth_scale is None else [float(f) for f in depth_scale.reshape(-1).tolist()]
    if factors is not None and len(factors) != int(hand_joints_cam.shape[1]):
        raise ValueError(f"depth_scale must have one factor per joint ({int(hand_joints_cam.shape[1])}), "
                         f"got {len(factors)}")
    for frame in range(t):
        for j in range(hand_joints_cam.shape[1]):
            if not bool(valid[frame, j]):
                continue
            joint = hand_joints_cam[frame, j]
            if not bool(torch.isfinite(joint).all()):
                continue
            x, y, z = float(joint[0]), float(joint[1]), float(joint[2])
            if z <= 0.0:
                continue
            u, v = int(fx * x / z + cx), int(fy * y / z + cy)
            if not (0 <= u < width and 0 <= v < height):
                continue
            frames.append(frame)
            vs.append(v)
            us.append(u)
            zs.append(z if factors is None else z * factors[j])
    return (torch.tensor(frames, dtype=torch.int64), torch.tensor(vs, dtype=torch.int64),
            torch.tensor(us, dtype=torch.int64), torch.tensor(zs, dtype=torch.float64))


def hand_anchored_scale_from_samples(n_frames: int, frame: Tensor, z: Tensor, samples: Tensor) -> Tensor:
    """``hand_anchored_scale``'s formula on ``hand_anchor_pixels``' ``frame``/``z`` and ``samples`` ``(P,3)``, the
    raw pointmap values at those pixels: per frame, the joints whose sample is finite give
    ``mean(z) / mean(sample z)`` (NaN when none do, or when ``abs(mean(sample z)) < 1e-6``).

    Returns:
        ``(n_frames,)`` float32, NaN where no scale could be estimated.
    """
    k = torch.full((n_frames,), float("nan"), dtype=torch.float32)
    usable = torch.isfinite(samples).all(dim=-1)
    for f in range(n_frames):
        rows = (frame == f) & usable
        if not bool(rows.any()):
            continue
        # Python floats in joint order, as the per-joint loop collected them: np.mean's summation order is unchanged.
        real_z: list[float] = z[rows].tolist()
        pm_z: list[float] = samples[rows, 2].tolist()
        h_pm_z = float(np.mean(pm_z))
        if abs(h_pm_z) < 1e-6:
            continue
        k[f] = float(np.mean(real_z)) / h_pm_z

    return k


def apply_metric_scale(pointmaps: Tensor, k: Tensor) -> Tensor:
    """Applies a per-frame scale factor (from ``hand_anchored_scale``) to raw pointmaps.

    Args:
        pointmaps: ``(T,H,W,3)`` raw camera-frame pointmaps.
        k: ``(T,)`` per-frame scale factor.

    Returns:
        ``(T,H,W,3)``: ``pointmaps * k[:, None, None, None]``.
    """
    return pointmaps * k[:, None, None, None]


def clip_median_scale(k: Tensor) -> Tensor:
    """Pools per-frame scale factors from ``hand_anchored_scale`` into one clip-level factor.

    A frame's anchor is ~21 hand joints read against the pointmap at their projected pixels; joints
    hidden behind the object or off the hand's visible surface read the wrong depth, so single
    frames can be tens of percent off. The median over frames (NaN frames ignored) is the
    low-variance estimate; it cannot follow per-frame drift of the monocular metric scale, which
    the solver's global scale does not model either.

    Args:
        k: ``(T,)`` per-frame scale, NaN where unset.

    Returns:
        ``(T,)`` tensor filled with the median of the finite entries of ``k``.

    Raises:
        ValueError: If no frame has a valid scale.
    """
    valid = torch.isfinite(k)
    if not bool(valid.any()):
        raise ValueError("hand_anchored_scale produced no valid frame; cannot fix the metric scale")
    return torch.full_like(k, float(k[valid].median()))
