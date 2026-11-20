"""Reads the DO AS I DO baseline's per-frame object layout JSON into a metric SE(3) trajectory.

Format and pose composition verified against the baseline's own scripts (the real demo data,
``third_party/do-as-i-do/reconstruction/whisking/obj_tracking_out/whisk/combined_visualization/
layout_camera_frame_optimized.json``, 138 frames):

- ``reconstruction/scripts/convert_layout_to_camera_frame.py`` converts a pose-frame
  (x-forward, y-left, z-up) translation/quaternion pair into the camera frame
  (x-right, y-down, z-forward) by reordering the translation ``[tx,ty,tz] -> [tz,tx,ty]`` and
  applying the change-of-basis matrix ``P = [[0,0,1],[-1,0,0],[0,-1,0]]``: for row vectors,
  ``t_cam = t_pose @ P``; for rotation matrices, ``R_cam = P.T @ R_pose`` (equivalently
  ``t_cam = P.T @ t_pose`` for a column vector -- both give the same numbers, verified below).
  Crucially, both this script and ``visualize_3d.py`` read the pose-frame quaternion from
  ``local_to_scene["new_quat"]``, *not* ``local_to_scene["quat_wxyz"]`` (a different, unrelated
  quaternion present in the same dict -- using it instead reproduces nothing).
- ``reconstruction/scripts/visualize_3d.py``'s ``load_layout`` auto-detects an already-converted
  ("optimized") layout by the presence of ``local_to_scene.translation_camera_frame`` /
  ``quat_wxyz_camera_frame`` and, in that case, uses them directly with no further scaling --
  i.e. those two fields are the final answer once present.
- The "optimized" layout additionally holds a per-frame metric-scale correction under
  ``translation_scale_optimization``: ``mesh_scale`` (the final, optimization-refined uniform
  mesh scale) vs. ``mesh_scale_original`` (the pre-optimization ``local_to_scene.scale``), and
  ``per_frame[i] = {"frame_idx", "translation_scale", ...}``, one multiplicative correction per
  frame. Numerically verifying every one of the whisking demo's 138 frames against the stored
  ``translation_camera_frame`` establishes the exact composition (max residual 2.2e-16, i.e.
  float64 machine precision):

      t_cam = ((tz, tx, ty) * translation_scale[frame_idx] * (mesh_scale / mesh_scale_original)) @ P

  i.e. the per-frame ``translation_scale`` and the mesh-scale ratio are applied to the raw
  pose-frame translation *before* the ``P`` change of basis (scalar multiplication commutes with
  the linear ``P`` map, so this is equivalent to scaling ``translation_camera_frame`` itself by
  the same two factors). This reader recomputes the conversion from the raw ``new_quat`` /
  ``translation`` / per-frame ``translation_scale`` fields (rather than reading the precomputed
  ``*_camera_frame`` fields) so it exercises -- and is testable against -- the actual pipeline
  math, not just a pass-through of already-converted numbers.
- Frame 125 (the keyframe/reference frame in the whisking demo) is the one entry missing
  ``intrinsics_normalized``; the remaining 137 frames' values vary slightly frame to frame (the
  per-frame camera focal length is itself re-estimated, not fixed). Since this reader returns a
  single ``K``, it denormalizes every frame that does carry ``intrinsics_normalized`` and returns
  the per-component median, which is robust to both the missing keyframe entry and the small
  per-frame jitter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["read_baseline_layout", "read_keyframe_layout"]

# Pose-frame (x-fwd, y-left, z-up) -> camera-frame (x-right, y-down, z-fwd) change of basis.
# Byte-for-byte the ``P`` matrix in convert_layout_to_camera_frame.py.
_P = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def read_baseline_layout(path: Path, image_width: int = 1280, image_height: int = 720,
                         ) -> tuple[Tensor, float, Tensor]:
    """Parses a baseline ``layout_camera_frame_optimized.json`` into a metric trajectory.

    Args:
        path: Path to the layout JSON (``obj_tracking_out/<object>/combined_visualization/
            layout_camera_frame_optimized.json`` or an equivalent un-optimized ``objects`` +
            optional ``translation_scale_optimization`` document).
        image_width: Pixel width used to denormalize ``intrinsics_normalized``.
        image_height: Pixel height used to denormalize ``intrinsics_normalized``.

    Returns:
        ``(poses, mesh_scale, K)``: ``poses`` is ``(T,4,4)`` float32 object-to-camera transforms
        (rotation and translation only -- ``mesh_scale`` is *not* baked in, matching
        ``visualize_3d.py``'s separate ``--scale`` argument), sorted by ascending frame index;
        ``mesh_scale`` is the scalar that multiplies canonical (unit-scale) mesh vertices before
        applying ``poses`` to place them in the camera frame; ``K`` is the ``(3,3)`` float32
        pixel intrinsics matrix (per-frame median when ``intrinsics_normalized`` varies).

    Raises:
        ValueError: If ``path`` has no ``objects`` entries, or none of them carry
            ``intrinsics_normalized``.
    """
    data = json.loads(Path(path).read_text())
    objects = sorted(data["objects"], key=_frame_index)
    if not objects:
        raise ValueError(f"{path}: 'objects' is empty")

    ts_by_frame, mesh_scale, scale_ratio = _translation_scale_table(
        data.get("translation_scale_optimization"), objects[0]["local_to_scene"])

    poses = np.zeros((len(objects), 4, 4), dtype=np.float32)
    intrinsics: list[np.ndarray] = []
    n_missing_intrinsics = 0
    for i, obj in enumerate(objects):
        frame_idx = _frame_index(obj)
        pose = obj["local_to_scene"]
        ts = ts_by_frame.get(frame_idx, 1.0)
        r_cam, t_cam = _pose_to_camera_frame(pose["new_quat"], pose["translation"], ts * scale_ratio)
        poses[i, :3, :3] = r_cam
        poses[i, :3, 3] = t_cam
        poses[i, 3, 3] = 1.0

        intr = obj.get("intrinsics_normalized")
        if intr is None:
            n_missing_intrinsics += 1
        else:
            intrinsics.append(_denormalize_intrinsics(intr, image_width, image_height))

    if not intrinsics:
        raise ValueError(f"{path}: no object entry has 'intrinsics_normalized'")
    if n_missing_intrinsics:
        logger.warning("%s: %d/%d frames missing 'intrinsics_normalized'; using the median K "
                       "over the remaining frames", path, n_missing_intrinsics, len(objects))
    k_matrix = np.median(np.stack(intrinsics, axis=0), axis=0).astype(np.float32)

    return torch.from_numpy(poses), float(mesh_scale), torch.from_numpy(k_matrix)


def read_keyframe_layout(path: Path, frame_idx: int | None = None) -> tuple[Tensor, float]:
    """The shape stage's own (pre-optimization) keyframe layout from a baseline layout JSON.

    `read_baseline_layout` returns the baseline's *final* answer: every frame's translation carries the
    per-frame ``translation_scale`` and the ``mesh_scale / mesh_scale_original`` ratio, and the scale is
    the optimized ``mesh_scale``. A head-to-head that starts from those values starts from the
    baseline's result. This reads what SAM 3D itself produced at the reference frame instead:
    ``local_to_scene``'s raw ``new_quat`` / ``translation`` with no scale correction (the same
    ``P`` change of basis as `read_baseline_layout`, via the shared `_pose_to_camera_frame`), and
    ``mesh_scale_original`` (the pre-optimization ``local_to_scene.scale``).

    Args:
        path: Layout JSON path (as for `read_baseline_layout`).
        frame_idx: The reference frame. `None` takes ``translation_scale_optimization.ref_frame``. With
            an optimization block, only its ``ref_frame`` is accepted: every other frame's
            ``local_to_scene`` is a tracked pose, not the shape stage's layout.

    Returns:
        ``(pose, scale)``: ``(4,4)`` float32 object-to-camera transform (scale not baked in) and the
        shape stage's uniform mesh scale.

    Raises:
        ValueError: If ``frame_idx`` is missing (no ``ref_frame`` to default to), is not the
            optimization's ``ref_frame``, or has no layout entry.
    """
    data = json.loads(Path(path).read_text())
    tso = data.get("translation_scale_optimization")
    ref = None if tso is None or "ref_frame" not in tso else int(tso["ref_frame"])
    if frame_idx is None:
        if ref is None:
            raise ValueError(f"{path}: no translation_scale_optimization.ref_frame; pass frame_idx")
        frame_idx = ref
    if ref is not None and frame_idx != ref:
        raise ValueError(f"{path}: frame {frame_idx} is a tracked pose; the shape stage's layout is ref_frame {ref}")
    entries = [obj for obj in data["objects"] if _frame_index(obj) == frame_idx]
    if not entries:
        raise ValueError(f"{path}: no layout entry for frame {frame_idx}")
    lts = entries[0]["local_to_scene"]
    r_cam, t_cam = _pose_to_camera_frame(lts["new_quat"], lts["translation"], 1.0)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3], pose[:3, 3] = r_cam, t_cam
    scale = float(lts["scale"][0])
    if tso is not None and "mesh_scale_original" in tso and abs(float(tso["mesh_scale_original"]) - scale) > 1e-6:
        logger.warning("%s: local_to_scene.scale %.6f differs from mesh_scale_original %.6f; using the latter",
                       path, scale, float(tso["mesh_scale_original"]))
        scale = float(tso["mesh_scale_original"])
    return torch.from_numpy(pose), scale


def _frame_index(obj: dict[str, Any]) -> int:
    """Baseline layouts key frames by ``frame_idx``; ``visualize_3d.py`` also accepts ``frame_index``."""
    if "frame_idx" in obj:
        return int(obj["frame_idx"])
    if "frame_index" in obj:
        return int(obj["frame_index"])
    raise KeyError("layout object entry has neither 'frame_idx' nor 'frame_index'")


def _translation_scale_table(tso: dict[str, Any] | None, fallback_local_to_scene: dict[str, Any],
                             ) -> tuple[dict[int, float], float, float]:
    """Returns ``(per-frame translation_scale, mesh_scale, mesh_scale / mesh_scale_original)``.

    Falls back to ``translation_scale=1`` everywhere and the un-optimized ``local_to_scene.scale``
    when the layout has no ``translation_scale_optimization`` block (e.g. a raw
    ``convert_layout_to_camera_frame.py`` output that was never run through the optimizer).
    """
    if tso is None:
        scale = fallback_local_to_scene.get("scale")
        mesh_scale = float(scale[0]) if scale else 1.0
        return {}, mesh_scale, 1.0
    mesh_scale = float(tso["mesh_scale"])
    mesh_scale_original = float(tso.get("mesh_scale_original", mesh_scale))
    scale_ratio = mesh_scale / mesh_scale_original if mesh_scale_original else 1.0
    table = {int(e["frame_idx"]): float(e["translation_scale"]) for e in tso.get("per_frame", [])}
    return table, mesh_scale, scale_ratio


def _pose_to_camera_frame(quat_wxyz_pose: list[float], translation_pose: list[float], scale: float,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces ``convert_rotation``/``convert_translation`` from
    ``convert_layout_to_camera_frame.py``: pose-frame (x-fwd, y-left, z-up) rotation + translation
    -> camera-frame (x-right, y-down, z-fwd), with ``translation_pose`` additionally scaled by
    ``scale`` (the per-frame ``translation_scale`` times the mesh-scale ratio) before conversion.
    """
    qw, qx, qy, qz = quat_wxyz_pose
    r_pose = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    r_cam = _P.T @ r_pose
    tx, ty, tz = translation_pose
    t_pose = np.array([tz, tx, ty], dtype=np.float64) * scale
    t_cam = _P.T @ t_pose
    return r_cam.astype(np.float32), t_cam.astype(np.float32)


def _denormalize_intrinsics(intr: dict[str, float], width: int, height: int) -> np.ndarray:
    """``intrinsics_normalized`` -> pixel ``K``: fx/cx scale by width, fy/cy scale by height."""
    fx = float(intr["fx_norm"]) * width
    fy = float(intr["fy_norm"]) * height
    cx = float(intr["cx_norm"]) * width
    cy = float(intr["cy_norm"]) * height
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
