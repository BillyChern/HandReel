"""HOI4D as a benchmark source: windows, keyframe prompt, object-pose registration, GT gate and GT adapter.

A HOI4D clip is ONE 10 s window of a 20 s sequence video (the S2 window length, `WINDOW_SECONDS`), decoded at the
operating stride of a 15 fps video (`operating_stride(15) = 2`, 7.5 fps, 75 rows). Window placement is fixed by the
action labels alone, before any pipeline output: it starts `LEAD_S` before the first contact segment (grasp, push,
...), clamped into the video, so the prompt frame shows the object before the hand reaches it and the window holds
the manipulation. The clip id is `bench_hoi4d_<seq with / -> _>_f<start:03d>`.

Engine inputs mirror DexYCB's (bench_clips.dexycb_prompt / dexycb_intrinsics): the keyframe prompt is the GT object
label of the window's first frame (decode row 0), and camera.json is the shipped calibration. The GT object pose is
corrected by the sequence's measured object-pose registration (`estimate_registration`; `fastdo.io.datasets.hoi4d`
explains why the object pose, not the camera, carries it).

GT adapter: `hoi4d_gt(reader, seq, shift_px)` returns `(object_pose (T,4,4) float64, valid (T,) bool)`, registered
and indexed by ABSOLUTE source frame, the
exact contract of qc23.load_gt, so the aligned post-contact scorer (qc23.align_to_model + score_aligned), the direct
template scorer (qc21.score) and centroid/along-ray read-outs run unchanged; `hoi4d_template_mesh` is the metric CAD
model (the DexYCB template analogue).

`gt_gate` is the GT-quality gate of a window, computed from GT alone (poses, masks, CAD, K): see `GATE` for the
thresholds.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from fastdo.bench.bench_clips import WINDOW_SECONDS, BenchClip, operating_stride
from fastdo.engine.source_domain import SourceDomain
from fastdo.io.datasets.hoi4d import (
    FPS,
    HOI4DReader,
    contact_time,
    correct_object_pose,
    parse_sequence_id,
)
from fastdo.io.datasets.hoi4d_gt_check import check_frame, silhouette

__all__ = [
    "GATE",
    "LEAD_S",
    "MIN_REGISTRATION_FRAMES",
    "SOURCE",
    "HOI4DWindow",
    "contact_window",
    "estimate_registration",
    "gt_gate",
    "hoi4d_clips",
    "hoi4d_gt",
    "hoi4d_prompt",
    "hoi4d_template_mesh",
    "load_windows",
    "window_clip_id",
]

logger = logging.getLogger(__name__)

SOURCE = "hoi4d"
LEAD_S = 2.0
STRIDE = operating_stride(FPS)
MIN_REGISTRATION_FRAMES = 5
_PREFIX = "bench_hoi4d_"

GATE: Mapping[str, float] = MappingProxyType({
    "min_valid_fraction": 0.9,      # objpose valid on >= 90% of the window's decode rows (pose gate)
    "min_prompt_px": 500.0,         # the prompt frame's object label (qc23.MIN_MASK_PX, the input floor)
    "max_registration_px": 25.0,    # |sequence registration shift|: > 25 px (~19 mm at 0.8 m) is not a registration
    "max_lateral_mm": 10.0,         # median residual lateral GT offset after registration
    "min_iou_best": 0.7,            # median best-shift IoU: rotation/shape agree once translation is removed
    "min_check_frames": 4.0,        # check frames with >= min_prompt_px object pixels
})
"""The GT-quality gate of a window (module docstring). Frozen with the split; changing it re-draws the split."""


@dataclass(frozen=True)
class HOI4DWindow:
    """One benchmark window: source frames `[start, stop)` of `sequence`, with its registration shift (px)."""

    sequence: str
    start: int
    stop: int
    shift_px: tuple[float, float] = (0.0, 0.0)

    @property
    def clip_id(self) -> str:
        return window_clip_id(self.sequence, self.start)


def window_clip_id(seq: str, start: int) -> str:
    parse_sequence_id(seq)
    return f"{_PREFIX}{seq.replace('/', '_')}_f{int(start):03d}"


def contact_window(events: Sequence[tuple[str, float, float]], n_frames: int, fps: float = FPS,
                   window_s: float = WINDOW_SECONDS, lead_s: float = LEAD_S) -> tuple[int, int]:
    """`[start, stop)` source frames: `window_s` long, starting `lead_s` before the first contact segment, clamped into
    `[0, n_frames)`. No contact segment: the window is centred on the video.

    Raises:
        ValueError: If the video is shorter than one window.
    """
    length = round(window_s * fps)
    if n_frames < length:
        raise ValueError(f"{n_frames} frames is shorter than one {window_s} s window ({length} frames)")
    t = contact_time(events)
    start = (n_frames - length) // 2 if t is None else round((t - lead_s) * fps)
    start = min(max(start, 0), n_frames - length)
    return start, start + length


def decode_rows(start: int, stop: int, stride: int = STRIDE) -> np.ndarray:
    """The source frames a windowed decode keeps: `start, start + stride, ... < stop`."""
    return np.arange(int(start), int(stop), int(stride), dtype=np.int64)


def hoi4d_template_mesh(reader: HOI4DReader, seq: str) -> Path:
    """The instance's metric CAD model (the DexYCB template analogue). Raises FileNotFoundError when absent."""
    path = reader.mesh_path(seq)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the CAD model of {seq}")
    return path


def hoi4d_prompt(reader: HOI4DReader, seq: str, frame: int) -> np.ndarray:
    """(H,W) bool GT object label of source frame `frame` (decode row 0 of a window starting there).

    Raises:
        ValueError: If the object has no pixel on that frame.
    """
    obj, _ = reader.object_mask(seq, frame)
    if not obj.any():
        raise ValueError(f"{seq}: the object has no labelled pixel on frame {frame}")
    return obj


def hoi4d_gt(reader: HOI4DReader, seq: str, shift_px: Sequence[float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """`(object_pose (T,4,4) float64, valid (T,) bool)` by absolute source frame -- qc23.load_gt's contract -- with the
    object pose corrected by the window's registration `shift_px` (the split's `shift_px`)."""
    rec = reader.load(seq, pose_shift_px=(float(shift_px[0]), float(shift_px[1])))
    return np.asarray(rec.object_pose, dtype=np.float64), np.asarray(rec.confidence) > 0


def _mesh(reader: HOI4DReader, seq: str) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = cast("trimesh.Trimesh", trimesh.load(hoi4d_template_mesh(reader, seq), force="mesh", process=False))
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def _checks(reader: HOI4DReader, seq: str, frames: Sequence[int], K: np.ndarray,
            mesh: tuple[np.ndarray, np.ndarray], image_hw: tuple[int, int],
            shift_px: Sequence[float] = (0.0, 0.0)) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in frames:
        pose, ok = reader.objpose(seq, int(t))
        if not ok:
            out.append({"t": int(t), "valid": False})
            continue
        pose = correct_object_pose(pose, K, shift_px)
        obj, other = reader.object_mask(seq, int(t))
        chk = check_frame(silhouette(*mesh, pose, K, image_hw), obj, other, float(pose[2, 3]), float(K[0, 0]))
        out.append({"t": int(t), "valid": True, **chk.as_dict()})
    return out


def estimate_registration(reader: HOI4DReader, seq: str, n_frames: int = 20, min_px: float = GATE["min_prompt_px"],
                          min_iou_best: float = 0.8, image_hw: tuple[int, int] = (1080, 1920)) -> dict[str, Any]:
    """The sequence's RGB-vs-pose registration from its GT alone: the median best-shift (dx, dy) px of the CAD
    silhouette onto the annotated mask, under the shipped K, over `n_frames` frames spread over the whole sequence
    (independent of any window), using frames with >= `min_px` object pixels and best-shift IoU >= `min_iou_best`.

    Returns `{shift_px: [dx, dy] | None, n_used, mad_px: [x, y]}`; `shift_px` is None with < MIN_REGISTRATION_FRAMES.
    """
    total = reader.n_frames(seq)
    frames = np.unique(np.linspace(0, total - 1, n_frames).round().astype(int)).tolist()
    K = reader.intrinsics(seq).astype(np.float64)
    rows = _checks(reader, seq, frames, K, _mesh(reader, seq), image_hw)
    used = [r for r in rows if r["valid"] and r["obj_px"] >= min_px and r["iou_best"] >= min_iou_best]
    if len(used) < MIN_REGISTRATION_FRAMES:
        return {"shift_px": None, "n_used": len(used), "mad_px": None}
    s = np.array([[r["shift_px"][1], r["shift_px"][0]] for r in used], float)    # (dx, dy)
    med = np.median(s, axis=0)
    return {"shift_px": [float(med[0]), float(med[1])], "n_used": len(used),
            "mad_px": np.median(np.abs(s - med), axis=0).round(2).tolist()}


def gt_gate(reader: HOI4DReader, seq: str, start: int, stop: int, shift_px: Sequence[float] | None,
            n_check: int = 8, image_hw: tuple[int, int] = (1080, 1920)) -> dict[str, Any]:
    """The window's GT-quality gate (`GATE`), from GT alone, on the object pose corrected by `shift_px` under the
    shipped K. `shift_px` None (registration failed) fails the gate.

    Check frames: `n_check` decode rows spread over the window. Returns the measurements, the failed criteria
    (`reasons`, empty when it passes) and `pass`.
    """
    rows = decode_rows(start, stop)
    valid = np.array([reader.objpose(seq, int(t))[1] for t in rows])
    prompt_px = int(reader.object_mask(seq, int(rows[0]))[0].sum())
    out: dict[str, Any] = {"valid_fraction": float(valid.mean()), "prompt_px": prompt_px,
                           "shift_px": None if shift_px is None else [float(v) for v in shift_px]}
    reasons: list[str] = []
    if out["valid_fraction"] < GATE["min_valid_fraction"]:
        reasons.append("valid_fraction")
    if prompt_px < GATE["min_prompt_px"]:
        reasons.append("prompt_px")
    if shift_px is None:
        reasons.append("registration")
    elif math.hypot(*shift_px) > GATE["max_registration_px"]:
        reasons.append("registration_px")
    if shift_px is not None:
        K = reader.intrinsics(seq).astype(np.float64)
        picks = rows[np.unique(np.linspace(0, len(rows) - 1, n_check).round().astype(int))]
        checks = [c for c in _checks(reader, seq, picks.tolist(), K, _mesh(reader, seq), image_hw, shift_px)
                  if c["valid"] and c["obj_px"] >= GATE["min_prompt_px"]]
        out["n_check"] = len(checks)
        if len(checks) < GATE["min_check_frames"]:
            reasons.append("check_frames")
        if checks:
            out.update(lateral_mm=float(np.median([c["lateral_mm"] for c in checks])),
                       iou_best=float(np.median([c["iou_best"] for c in checks])),
                       iou_occ=float(np.median([c["iou_occ"] for c in checks])),
                       iou_raw=float(np.median([c["iou_raw"] for c in checks])))
            if out["lateral_mm"] > GATE["max_lateral_mm"]:
                reasons.append("lateral_mm")
            if out["iou_best"] < GATE["min_iou_best"]:
                reasons.append("iou_best")
    out["reasons"], out["pass"] = reasons, not reasons
    return out


def load_windows(entries: Sequence[Mapping[str, Any]]) -> list[HOI4DWindow]:
    """`HOI4DWindow`s from split/inventory entries `{sequence, start, stop, shift_px}`.

    Raises:
        ValueError: On a missing key, a malformed id, a bad range or a repeated clip id.
    """
    out: list[HOI4DWindow] = []
    for e in entries:
        missing = [k for k in ("sequence", "start", "stop", "shift_px") if k not in e]
        if missing:
            raise ValueError(f"HOI4D window entry missing {missing}: {dict(e)}")
        start, stop = int(e["start"]), int(e["stop"])
        if not 0 <= start < stop:
            raise ValueError(f"{e['sequence']}: window must be 0 <= start < stop, got [{start}, {stop})")
        dx, dy = (float(v) for v in e["shift_px"])
        out.append(HOI4DWindow(str(e["sequence"]), start, stop, (dx, dy)))
    ids = [w.clip_id for w in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"repeated HOI4D clip ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return out


def hoi4d_clips(reader: HOI4DReader, windows: Sequence[HOI4DWindow]) -> list[BenchClip]:
    """One windowed video `BenchClip` per window (decode stride `STRIDE`). Raises FileNotFoundError on a missing video."""
    clips: list[BenchClip] = []
    for w in windows:
        path = reader.video_path(w.sequence)
        if not path.is_file():
            raise FileNotFoundError(f"HOI4D video {path} not found")
        clips.append(BenchClip(w.clip_id, SOURCE, "video", path, w.stop - w.start, window=(w.start, w.stop),
                               decode_stride=STRIDE, domain=SourceDomain.EGOCENTRIC))   # head-mounted camera
    return clips
