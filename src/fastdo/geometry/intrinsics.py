"""Camera intrinsics for the engine's `intrinsics` stage: given calibration or MoGe-2's estimate.

The `intrinsics` stage runs after `shape` and before `hands` and `geometry`
(`fastdo.engine.manifest.STAGE_NAMES`), because the geometry stage's metric scale projects hand joints
through `K`, and no earlier stage produced one. Its artifact, `intrinsics.npz`, holds exactly:

- `K`: `(3,3)` float32 pixel intrinsics of the decoded frames;
- `source`: a 0-d unicode array, one of `INTRINSICS_SOURCES` -- `"given"` when the clip ships a
  `camera.json` calibration (`fx`, `fy`, `cx`, `cy` in pixels), `"moge2"` when `K` was estimated.

It loads without pickle. MoGe-2's `infer(...)["intrinsics"]` (pinned rev `b942f00`, `moge/model/v2.py`) is
normalized by the image size: `fx/W`, `fy/H`, and `cx = cy = 0.5`. `pixel_intrinsics` turns a stack of those
into one pixel `K` by the entry-wise median over frames, so a single bad keyframe cannot drag the focal length.

Principal-point convention:
`pixel_intrinsics` maps the normalized `cx = cy = 0.5` to `0.5 * W` and `0.5 * H` (640 and 360 at 1280x720), the
values `test_pixel_intrinsics_median` pins and the baseline MoGe run's `0125_intrinsics.txt` lists.

Focal debias (`geometry_stage.intrinsics_focal_scale`; ON in the shipped `run/conf/config.yaml`, which sets the pooled
1.1040399273866555; null = off; these functions default to None = off): MoGe-2's focal length is
biased LOW on lab cameras (median log error DexYCB -11.7%, HOI4D -4.75%, OakInk2 egocentric about -12%; within-clip
spread about 1.2%: a bias, not noise), and HaWoR's hand depth, hence the object's along-ray depth, scales with it.
With `focal_scale = s`, `pixel_intrinsics` multiplies `fx` and `fy` by `s` and leaves the principal point alone;
`write_intrinsics` then also records `focal_scale` (0-d float64). `None` (or exactly 1.0) is the uncorrected path,
byte for byte. `s` is a CAMERA-DOMAIN constant fitted on labelled lab video; its transfer to internet video is
unverified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

__all__ = [
    "CAMERA_JSON",
    "FOCAL_SCALE_RANGE",
    "INTRINSICS_FILE",
    "INTRINSICS_SOURCES",
    "camera_matrix",
    "check_focal_scale",
    "pixel_intrinsics",
    "read_camera_json",
    "read_intrinsics",
    "write_intrinsics",
]

INTRINSICS_SOURCES: tuple[str, ...] = ("given", "moge2")
INTRINSICS_FILE = "intrinsics.npz"
CAMERA_JSON = "camera.json"
# the focal multiplier's admissible range (a correction of the measured lab biases, -12% .. +11%, with margin)
FOCAL_SCALE_RANGE: tuple[float, float] = (0.8, 1.25)


def check_focal_scale(value: object, name: str = "focal_scale") -> float | None:
    """`value` as a float, or None when it is off (None or exactly 1.0, the byte-identical path).

    Raises:
        ValueError: On a bool, a non-number, a non-finite number, or one outside `FOCAL_SCALE_RANGE`.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError(f"{name} must be null or a number in {FOCAL_SCALE_RANGE}, got {value!r}")
    s = float(value)
    lo, hi = FOCAL_SCALE_RANGE
    if not (np.isfinite(s) and lo <= s <= hi):
        raise ValueError(f"{name} must be null or a number in {FOCAL_SCALE_RANGE}, got {value!r}")
    return None if s == 1.0 else s


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """`(3,3)` float32 pinhole matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def pixel_intrinsics(normalized: np.ndarray, width: int, height: int, focal_scale: float | None = None) -> np.ndarray:
    """Pixel `K` from `(N,3,3)` MoGe-2 normalized intrinsics.

    Takes the median over the N frames of each of `fx/W`, `fy/H`, `cx/W`, `cy/H` separately, then scales by
    `W`, `H`, `W`, `H`. `focal_scale` (`check_focal_scale`; None = off) then multiplies `fx` and `fy`.

    Raises:
        ValueError: If `normalized` is not `(N,3,3)` with N >= 1, or `focal_scale` is invalid.
    """
    scale = check_focal_scale(focal_scale)
    arr = np.asarray(normalized, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3) or arr.shape[0] < 1:
        raise ValueError(f"normalized intrinsics must be (N,3,3) with N >= 1, got {arr.shape}")
    fx = float(np.median(arr[:, 0, 0])) * width
    fy = float(np.median(arr[:, 1, 1])) * height
    cx = float(np.median(arr[:, 0, 2])) * width
    cy = float(np.median(arr[:, 1, 2])) * height
    if scale is not None:
        fx, fy = fx * scale, fy * scale
    return camera_matrix(fx, fy, cx, cy)


def read_camera_json(path: Path) -> np.ndarray:
    """`camera_matrix` from a JSON file with pixel `fx`, `fy`, `cx`, `cy` keys."""
    data = json.loads(Path(path).read_text())
    return camera_matrix(float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"]))


def write_intrinsics(artifact_dir: Path, K: np.ndarray, source: str, focal_scale: float | None = None) -> None:
    """Writes `<artifact_dir>/intrinsics.npz` (uncompressed, tmp file then `os.replace`).

    `focal_scale` is provenance only: when it is on (`check_focal_scale` not None) the file also holds it as a
    0-d float64; `K` must already include it. Off, the file holds exactly `K` and `source`.

    Raises:
        ValueError: If `source` is not in `INTRINSICS_SOURCES`, `K` is not a finite `(3,3)` matrix, or `focal_scale`
            is invalid.
    """
    scale = check_focal_scale(focal_scale)
    if source not in INTRINSICS_SOURCES:
        raise ValueError(f"source must be one of {INTRINSICS_SOURCES}, got {source!r}")
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"K must be a finite (3,3) matrix, got shape {K.shape}")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / INTRINSICS_FILE
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as f:
            extra = {} if scale is None else {"focal_scale": np.array(scale, dtype=np.float64)}
            np.savez(f, K=K, source=np.array(source), **extra)
        os.replace(tmp, path)
    except BaseException:                  # any failure or interrupt, as `engine.worker_support.save_npz` does
        tmp.unlink(missing_ok=True)
        raise


def read_intrinsics(artifact_dir: Path) -> tuple[np.ndarray, str]:
    """`(K, source)` from `<artifact_dir>/intrinsics.npz`.

    Raises:
        FileNotFoundError: If the artifact is absent (the `intrinsics` stage has not run for this clip).
    """
    path = Path(artifact_dir) / INTRINSICS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the 'intrinsics' stage before 'geometry'")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["K"], dtype=np.float32), str(data["source"])
