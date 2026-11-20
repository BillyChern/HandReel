"""driftqc row-range slicing helper: the single place a driftqc prefix is applied to engine artifacts.

- `read_valid_range` validates `valid_range.json` against its `driftqc.json` and the clip's decode rows and returns an
  inclusive prefix `RowRange`, or None for the identity path (file absent, or range [0, N-1]).
- `resolve_valid_range` is the only entry the engine hooks call. With driftqc disabled it never slices: leftover
  driftqc artifacts are a ValueError, and without them the result is None.
- `slice_artifact` is pure row indexing. Per-row keys (`PER_ROW`) are sliced `[0, stop]`, clip-level keys
  (`CLIP_LEVEL`) are passed through as the same object, and any other key is a ValueError. No value is ever used in
  arithmetic, so `frame_scale` is sliced like `frame_idx` and never divided or multiplied.
- `slice_observations` applies the same prefix to cache `TrajectoryObservations` (used by the S7 runner).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from fastdo.engine.workers_hands import SIDES
from fastdo.hands.hand_camera import CAMERA_KEYS

if TYPE_CHECKING:
    from fastdo.solver.api import TrajectoryObservations

__all__ = [
    "CLIP_LEVEL",
    "DRIFTQC_FILE",
    "PER_ROW",
    "VALID_RANGE_FILE",
    "VALID_RANGE_KEYS",
    "RowRange",
    "frame_idx_sha256",
    "read_valid_range",
    "resolve_valid_range",
    "slice_artifact",
    "slice_observations",
]

VALID_RANGE_FILE = "valid_range.json"
DRIFTQC_FILE = "driftqc.json"  # defined here, re-exported by workers_driftqc
VALID_RANGE_VERSION = 1
# The valid_range.json schema, frozen in E1; E3's writer imports it.
VALID_RANGE_KEYS = ("version", "clip_id", "segment_id", "n_rows", "frame_idx_sha256", "range", "decision",
                    "driftqc_sha256")

_HAND_PER_ROW = ("params", "trans", "joints", "valid", "vertices", "infilled")
_HAND_CLIP_LEVEL = ("betas", "faces")

# Per artifact: keys whose leading axis is the decode rows (sliced) and clip-level keys (passed through by identity).
PER_ROW: Mapping[str, frozenset[str]] = MappingProxyType({
    "mask.npz": frozenset({"masks", "frame_idx"}),
    "geometry.npz": frozenset({"pointmaps", "frame_idx", "frame_scale", "cam_to_world"}),
    "tracks.npz": frozenset({"tracks", "vis", "frame_idx"}),
    "hands.npz": frozenset({"frame_idx"} | {f"{s}_{k}" for s in SIDES for k in _HAND_PER_ROW}),
    # ruler options add the per-frame contact cost and gate (written whenever the ruler is not the default)
    "contact.npz": frozenset({"poses", "shift_mm", "frame_idx", "min_cost_mm", "gated"}),
})
CLIP_LEVEL: Mapping[str, frozenset[str]] = MappingProxyType({
    "mask.npz": frozenset({"keyframe_index"}),
    "geometry.npz": frozenset({"scale", "gravity_cam"}),
    "tracks.npz": frozenset({"query_uv", "keyframe_index"}),
    "hands.npz": frozenset({f"{s}_{k}" for s in SIDES for k in _HAND_CLIP_LEVEL} | set(CAMERA_KEYS)),   # cameras
    "contact.npz": frozenset({"side", "n_corrected", "solve_sha256", "constants", "centroid_obj", "bias_free",
                              "touch_gate_mm", "geometric"}),
})

_OBS_PER_ROW = ("masks", "hand_masks", "pointmaps", "tracks", "track_vis")  # intrinsics, anchors: clip-level


@dataclass(frozen=True)
class RowRange:
    """Inclusive prefix [0, stop] of `n_rows` decode rows (v1: start is always 0)."""

    stop: int
    n_rows: int

    def __post_init__(self) -> None:
        if not 0 <= self.stop < self.n_rows:
            raise ValueError(f"RowRange stop {self.stop} outside [0, {self.n_rows - 1}]")

    def as_slice(self) -> slice:
        """`slice(0, stop + 1)`."""
        return slice(0, self.stop + 1)

    @property
    def start(self) -> int:
        """Always 0 in v1."""
        return 0


def frame_idx_sha256(frame_idx: np.ndarray) -> str:
    """sha256 of `np.asarray(frame_idx, np.int64).reshape(-1).tobytes()`."""
    return hashlib.sha256(np.asarray(frame_idx, np.int64).reshape(-1).tobytes()).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_valid_range(artifact_dir: Path, decode_idx: np.ndarray) -> RowRange | None:
    """The clip's driftqc prefix, or None for the identity path.

    Returns None if valid_range.json is absent OR its range is [0, N-1] (callers then run the unmodified code).

    Raises:
        ValueError: If driftqc.json is absent; sha256(driftqc.json bytes) != valid_range.json["driftqc_sha256"];
            n_rows != decode_idx.size; frame_idx_sha256 mismatch; range is not [0, b] with 0 <= b <= N-1;
            version != 1.
    """
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / VALID_RANGE_FILE
    if not path.is_file():
        return None
    record: Any = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"{path}: expected a JSON object")  # noqa: TRY004 - spec: every malformed record is ValueError
    missing = [key for key in VALID_RANGE_KEYS if key not in record]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    version = record["version"]
    if not _is_int(version) or version != VALID_RANGE_VERSION:
        raise ValueError(f"{path}: version {version!r} != {VALID_RANGE_VERSION}")
    driftqc_path = artifact_dir / DRIFTQC_FILE
    if not driftqc_path.is_file():
        raise ValueError(f"{path}: {DRIFTQC_FILE} absent next to {VALID_RANGE_FILE}")
    driftqc_sha = hashlib.sha256(driftqc_path.read_bytes()).hexdigest()
    if record["driftqc_sha256"] != driftqc_sha:
        raise ValueError(f"{path}: driftqc_sha256 {record['driftqc_sha256']!r} != sha256 of {DRIFTQC_FILE} "
                         f"{driftqc_sha}")
    n = int(np.asarray(decode_idx).size)
    n_rows = record["n_rows"]
    if not _is_int(n_rows) or n_rows != n:
        raise ValueError(f"{path}: n_rows {n_rows!r} != decode rows {n}")
    if record["frame_idx_sha256"] != frame_idx_sha256(decode_idx):
        raise ValueError(f"{path}: frame_idx_sha256 does not match the clip's decode frame_idx")
    rng = record["range"]
    if (not isinstance(rng, list) or len(rng) != 2 or not all(_is_int(v) for v in rng) or rng[0] != 0
            or not 0 <= rng[1] <= n - 1):
        raise ValueError(f"{path}: range {rng!r} is not [0, b] with 0 <= b <= {n - 1}")
    stop = int(rng[1])
    if stop == n - 1:
        return None
    return RowRange(stop=stop, n_rows=n)


def resolve_valid_range(artifact_dir: Path, decode_idx: np.ndarray, driftqc_enabled: bool) -> RowRange | None:
    """The only entry the engine hooks call.

    driftqc_enabled False: never slices; ValueError("stale driftqc artifacts: ...") if driftqc.json or
    valid_range.json exists, else None. True: valid_range.json must exist (else ValueError "driftqc enabled but
    valid_range.json absent"), then `read_valid_range(artifact_dir, decode_idx)`.
    """
    artifact_dir = Path(artifact_dir)
    if not driftqc_enabled:
        present = [name for name in (DRIFTQC_FILE, VALID_RANGE_FILE) if (artifact_dir / name).exists()]
        if present:
            raise ValueError(f"stale driftqc artifacts: {present} in {artifact_dir} with driftqc disabled; "
                             f"remove them or enable driftqc_stage")
        return None
    if not (artifact_dir / VALID_RANGE_FILE).is_file():
        raise ValueError(f"driftqc enabled but valid_range.json absent in {artifact_dir}")
    return read_valid_range(artifact_dir, decode_idx)


def slice_artifact(arrays: Mapping[str, np.ndarray], artifact: str, rng: RowRange) -> dict[str, np.ndarray]:
    """Row-indexed copy of one artifact's arrays.

    out[k] = arrays[k][rng.as_slice()] for k in PER_ROW[artifact] (after checking arrays[k].shape[0] == rng.n_rows),
    out[k] = arrays[k] (same object) for k in CLIP_LEVEL[artifact]. No arithmetic of any kind (frame_scale included).
    Dtypes and trailing shapes are preserved.

    Raises:
        ValueError: On an artifact without a key classification, a key in neither set (forces explicit
            classification), or a per-row length != n_rows.
    """
    if artifact not in PER_ROW:
        raise ValueError(f"no row classification for artifact {artifact!r}; known: {sorted(PER_ROW)}")
    per_row, clip_level = PER_ROW[artifact], CLIP_LEVEL[artifact]
    unknown = sorted(set(arrays) - per_row - clip_level)
    if unknown:
        raise ValueError(f"{artifact}: unclassified keys {unknown}")
    rows = rng.as_slice()
    out: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key in clip_level:
            out[key] = value
            continue
        if value.ndim == 0 or value.shape[0] != rng.n_rows:
            raise ValueError(f"{artifact}[{key!r}] has leading shape {value.shape[:1]}, expected ({rng.n_rows},)")
        out[key] = value[rows]
    return out


def slice_observations(obs: TrajectoryObservations, rng: RowRange) -> TrajectoryObservations:
    """Per-row tensors (masks, hand_masks, pointmaps, tracks, track_vis) sliced [0, stop]; intrinsics (3,3) and
    anchors (J,3) passed through. Used by the S7 runner on cache observations; the engine uses slice_artifact.
    `object_pixels`, when set, keeps the kept frames' segment prefix.

    Raises:
        ValueError: If a per-row tensor's leading size (or `object_pixels`' frame count) != rng.n_rows.
    """
    rows = rng.as_slice()
    sliced: dict[str, Any] = {}
    for name in _OBS_PER_ROW:
        tensor = getattr(obs, name)
        if tensor.ndim == 0 or int(tensor.shape[0]) != rng.n_rows:
            raise ValueError(f"observations.{name} has leading shape {tuple(tensor.shape[:1])}, "
                             f"expected ({rng.n_rows},)")
        sliced[name] = tensor[rows]
    px = obs.object_pixels
    if px is not None:                     # the object pixels of the kept frames, a prefix of the packed points
        if px.n_frames != rng.n_rows:
            raise ValueError(f"observations.object_pixels has {px.n_frames} frames, expected {rng.n_rows}")
        sliced["object_pixels"] = dataclasses.replace(px, points=px.points[:int(px.offsets[rows.stop])],
                                                      offsets=px.offsets[:rows.stop + 1])
    return dataclasses.replace(obs, **sliced)
