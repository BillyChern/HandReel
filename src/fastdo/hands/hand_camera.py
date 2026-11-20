"""Camera consistency of HaWoR outputs.

HaWoR's full-frame translation (`hawor.py` `get_trans`, run by `fastdo.hands.hawor_core.run_heads`) uses the clip's focal
length `K[0,0]` and the image centre `(W/2, H/2)` as principal point. Its camera-frame joints, vertices and translations
therefore satisfy `u = f X/Z + W/2`, whatever principal point the clip's camera `K` has. Projecting them through a given
calibration moves every point by `(cx - W/2, cy - H/2)` pixels: (-11.45, +8.52) on DexYCB serial 836212060125, which put
22-31 % of the hand-anchored scale's joint pixels on background.

`principal_point_shear` puts such points in the projection camera, leaving depth unchanged:

    X' = X + Z (cx_H - cx_K) / fx_K,    Y' = Y + Z (cy_H - cy_K) / fy_K,    Z' = Z.

A point's pixel through `K` then equals its pixel through `K` with HaWoR's principal point, exactly. This is the
evaluated form of the fix: it is applied to HaWoR outputs computed with the current image centre, and
HaWoR's inputs, including its CLIFF bounding-box feature, do not change.
- **Focal length.** HaWoR uses `fx` on both axes and the shear leaves the focal lengths alone. The remaining vertical term
  `(fy - fx)/fy (v - H/2)` is at most 0.15 px on DexYCB. A MoGe-2 camera (internet video, HOI4D's primary row) has its
  principal point at `(W/2, H/2)`, so the shear is the identity there, bitwise, even though its `fx != fy`.
- **Non-rigid.** The shear distorts a hand mesh by up to 3.2 mm on DexYCB DEV; a rigid root translation would leave up to
  2.4 px of reprojection error instead. The shear is used for every consumer (geometry scale, retarget,
  penetration QC).

`fastdo.engine.workers_hands.write_hands_artifact` applies the shear and records both cameras in `hands.npz`:
`CAMERA_KEY`, the camera the outputs are in, and `HAWOR_CAMERA_KEY`, HaWoR's own. A consumer that uses hand outputs
together with `K` calls `require_hands_camera`. It raises when the outputs are not in `K`: either the recorded camera
differs, or no camera is recorded (an older `hands.npz`, in HaWoR's camera) and `K`'s principal point is more
than `PP_TOLERANCE_PX` from the image centre.

numpy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "CAMERA_KEY",
    "CAMERA_KEYS",
    "HAWOR_CAMERA_KEY",
    "PP_TOLERANCE_PX",
    "hawor_camera",
    "principal_point",
    "principal_point_shear",
    "require_hands_camera",
]

CAMERA_KEY = "camera_K"                    # (3,3) float64: the camera the hand outputs are expressed in
HAWOR_CAMERA_KEY = "hawor_camera_K"        # (3,3) float64: the camera HaWoR computed them in
CAMERA_KEYS: tuple[str, ...] = (CAMERA_KEY, HAWOR_CAMERA_KEY)
PP_TOLERANCE_PX = 1.0                      # a larger uncorrected principal-point difference is a hard error


def _camera(K: Any, name: str) -> np.ndarray:
    k = np.asarray(K, dtype=np.float64)
    if k.shape != (3, 3) or not bool(np.isfinite(k).all()) or not (k[0, 0] > 0.0 and k[1, 1] > 0.0):
        raise ValueError(f"{name} must be a finite (3,3) pinhole matrix with positive focal lengths, got {k.tolist()}")
    return k


def principal_point(K: Any) -> np.ndarray:
    """`(cx, cy)` of a `(3,3)` pinhole matrix, float64."""
    k = _camera(K, "K")
    return np.array([k[0, 2], k[1, 2]])


def hawor_camera(K: Any, width: int, height: int) -> np.ndarray:
    """The `(3,3)` float64 camera HaWoR computes in for a clip with camera `K` and `width` x `height` frames: focal
    `K[0,0]` on both axes, principal point `(width / 2, height / 2)` (`hawor_core.run_heads`)."""
    f = float(_camera(K, "K")[0, 0])
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def principal_point_shear(points: np.ndarray, from_principal_point: Sequence[float] | np.ndarray, K: Any) -> np.ndarray:
    """`(..., 3)` camera-frame points seen through principal point `from_principal_point`, moved into camera `K`.

    The result has `points`' shape and dtype. The arithmetic is float64; non-finite rows stay non-finite; depth is
    copied unchanged. When the principal points are equal the result is a bitwise copy.

    Raises:
        ValueError: If `points` is not a floating `(..., 3)` array or `K` is not a valid pinhole matrix.
    """
    arr = np.asarray(points)
    if arr.ndim < 1 or arr.shape[-1] != 3 or arr.dtype.kind != "f":
        raise ValueError(f"points must be a floating (..., 3) array, got {arr.shape} {arr.dtype}")
    k = _camera(K, "K")
    cx_h, cy_h = (float(v) for v in np.asarray(from_principal_point, dtype=np.float64).reshape(2))
    dx, dy = cx_h - k[0, 2], cy_h - k[1, 2]
    if dx == 0.0 and dy == 0.0:
        return arr.copy()
    p = arr.astype(np.float64)
    p[..., 0] += p[..., 2] * (dx / k[0, 0])
    p[..., 1] += p[..., 2] * (dy / k[1, 1])
    return p.astype(arr.dtype, copy=False)


def _fmt(xy: np.ndarray) -> str:
    return f"({xy[0]:.2f}, {xy[1]:.2f})"


def require_hands_camera(hands: Mapping[str, Any], K: Any, width: int, height: int, source: str) -> None:
    """Checks that hand outputs (`hands.npz` arrays) are in the projection camera `K` of `width` x `height` frames.

    - With `CAMERA_KEY` recorded, it must equal `K`: principal point and focal lengths within `PP_TOLERANCE_PX`.
    - Without it, the outputs are HaWoR's, uncorrected, with principal point `(width / 2, height / 2)`; `K`'s principal
      point must be within `PP_TOLERANCE_PX` of it on both axes.

    Raises:
        ValueError: Naming `source`, both principal points and the pixel offset, when the check fails.
    """
    k = _camera(K, "K")
    if CAMERA_KEY in hands:
        recorded = _camera(hands[CAMERA_KEY], f"{source} {CAMERA_KEY}")
        pp_delta = principal_point(recorded) - principal_point(k)
        focal_delta = np.array([recorded[0, 0] - k[0, 0], recorded[1, 1] - k[1, 1]])
        if max(float(np.abs(pp_delta).max()), float(np.abs(focal_delta).max())) > PP_TOLERANCE_PX:
            raise ValueError(
                f"{source}: the hand outputs are in the camera recorded as {CAMERA_KEY} (principal point "
                f"{_fmt(principal_point(recorded))}, focal {_fmt(recorded[[0, 1], [0, 1]])}), not in the projection "
                f"camera K (principal point {_fmt(principal_point(k))}, focal {_fmt(k[[0, 1], [0, 1]])}): off by "
                f"{pp_delta[0]:+.2f}, {pp_delta[1]:+.2f} px (tolerance {PP_TOLERANCE_PX:g} px). Rerun the hands stage "
                f"with this clip's intrinsics")
        return
    hawor_pp = np.array([width / 2.0, height / 2.0])
    delta = principal_point(k) - hawor_pp
    if float(np.abs(delta).max()) > PP_TOLERANCE_PX:
        raise ValueError(
            f"{source} records no camera correction ({CAMERA_KEY} absent), so its HaWoR outputs use HaWoR's principal "
            f"point {_fmt(hawor_pp)}, but the projection camera K has principal point {_fmt(principal_point(k))}: "
            f"every hand point would land {delta[0]:+.2f}, {delta[1]:+.2f} px off (tolerance {PP_TOLERANCE_PX:g} px)"
            f". Rerun the hands stage, whose writer shears the outputs into K")
