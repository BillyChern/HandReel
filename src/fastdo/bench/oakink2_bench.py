"""OakInk2 as a benchmark source: CONTIGUOUS 10 s windows per sequence, with GT object identity per window.

LICENCE: OakInk2 is CC BY-SA 4.0. SHARE-ALIKE: redistributed derivatives (window lists with GT, masks, meshes, poses)
carry the same licence and attribution (`fastdo.io.datasets.oakink2`). Read-only: nothing is written under the root.

Why this route exists. The stage-0 grouping gate validates
a label-free ADJACENCY key: window k and k+1 of one video are linked when the object mask at the cut overlaps.
Precision needs adjacent windows whose masks can sit on DIFFERENT objects. DexYCB and HOI4D are one object per sequence
and one window per sequence. OakInk2's multi-object task programs are the only local identity-known source where
consecutive windows change object.

Windows. Image index i of view v (`OakInk2Reader.frame_files`, ~30 fps).
- Window k of a sequence covers images `[o + L k, o + L (k + 1))`, with `L = round(WINDOW_SECONDS * FPS) = 300` and
  `o = round(window_offset(seq) * FPS)`: S2's seeded per-video head offset, a function of the name alone.
- Only complete windows are kept.
- Decode rows are every `STRIDE = operating_stride(30) = 3`rd image from the window start (10 fps; 100 rows).
- The last row of window k and the first row of k+1 are therefore 3 images (~0.1 s) apart, as in S2.
- Clip ids: `bench_oakink2_<ego|allo>_<seq with '++' -> '--'>_k<NNN>`.

Source domain (`fastdo.engine.source_domain`):
- `oakink2_ego` is EGOCENTRIC: the `egocentric` view is head-mounted, and its extrinsics move within a sequence.
- `oakink2_allo` is THIRD_PERSON: the static `allocentric_top` rig camera.
- The domain follows the VIEW, so the view is part of the source name.

GT identity (`window_identity`, from the task program alone, no pixels).
- Each program step assigns a MOCAP-id interval per hand to that hand's objects.
- A window's `interaction` is, per object, the number of mocap ids in the window covered by any hand interval naming it.
- `primary` is the object with the most interaction (ties: the lexicographically smallest id), or None when no hand
  interacts in the window.
- `first_objects` / `last_objects` are the objects whose interval covers the window's first / last decode row.
- This identifies what the HANDS manipulate. Which object the pipeline's MASK covers is measured after the mask pass,
  against GT silhouettes.

No prompt is written for these clips: the production prompt stage (`prompt_stage.enabled: true`) chooses the object,
as on Ego4D. No `camera.json` is written: intrinsics are estimated, as on internet video.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.bench.bench_clips import (
    WINDOW_SECONDS,
    BenchClip,
    _read_rgb,
    operating_stride,
    window_offset,
)
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.source_domain import SourceDomain
from fastdo.engine.worker_support import DECODE_FILE, WorkerContext, save_npz
from fastdo.io.datasets.oakink2 import FPS, OakInk2Reader, Primitive, refuse_reserved

__all__ = ["DOMAIN_OF_SOURCE", "SOURCES", "SOURCE_ALLO", "SOURCE_EGO", "STRIDE", "VIEW_OF_SOURCE", "WINDOW_IMAGES",
           "OakInk2Window", "contiguous_windows", "image_window_files", "load_windows", "make_image_window_decode_worker",
           "oakink2_clips", "sequence_windows", "window_identity"]

SOURCE_EGO, SOURCE_ALLO = "oakink2_ego", "oakink2_allo"
VIEW_OF_SOURCE: Mapping[str, str] = {SOURCE_EGO: "egocentric", SOURCE_ALLO: "allocentric_top"}
DOMAIN_OF_SOURCE: Mapping[str, SourceDomain] = {SOURCE_EGO: SourceDomain.EGOCENTRIC,
                                                SOURCE_ALLO: SourceDomain.THIRD_PERSON}
SOURCES = (SOURCE_EGO, SOURCE_ALLO)
WINDOW_IMAGES = round(WINDOW_SECONDS * FPS)
STRIDE = operating_stride(FPS)
_TAG = {SOURCE_EGO: "ego", SOURCE_ALLO: "allo"}


@dataclass(frozen=True)
class OakInk2Window:
    """Window `k` of `sequence` in `source`'s view: image indices `[start, stop)`."""

    sequence: str
    source: str
    k: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        refuse_reserved(self.sequence)
        if self.source not in VIEW_OF_SOURCE:
            raise ValueError(f"unknown OakInk2 source {self.source!r}; expected one of {sorted(VIEW_OF_SOURCE)}")
        if not 0 <= self.start < self.stop:
            raise ValueError(f"{self.sequence}: window must be 0 <= start < stop, got [{self.start}, {self.stop})")

    @property
    def clip_id(self) -> str:
        return f"bench_oakink2_{_TAG[self.source]}_{self.sequence.replace('++', '--')}_k{self.k:03d}"

    @property
    def rows(self) -> np.ndarray:
        """The image indices a decode keeps: `start, start + STRIDE, ... < stop`."""
        return np.arange(self.start, self.stop, STRIDE, dtype=np.int64)


def contiguous_windows(n_images: int, offset: int, length: int = WINDOW_IMAGES) -> list[tuple[int, int, int]]:
    """`(k, start, stop)` of every COMPLETE window `[offset + length k, offset + length (k + 1))` inside `[0, n_images)`.

    Raises:
        ValueError: If `length < 1` or `offset < 0`.
    """
    if length < 1 or offset < 0:
        raise ValueError(f"length must be >= 1 and offset >= 0, got {length} and {offset}")
    out, k = [], 0
    while offset + length * (k + 1) <= n_images:
        out.append((k, offset + length * k, offset + length * (k + 1)))
        k += 1
    return out


def sequence_windows(reader: OakInk2Reader, seq: str, source: str = SOURCE_EGO) -> list[OakInk2Window]:
    """Every complete window of `seq` in `source`'s view (module docstring)."""
    n = len(reader.frame_files(seq, VIEW_OF_SOURCE[source]))
    offset = round(window_offset(seq) * FPS)
    return [OakInk2Window(seq, source, k, a, b) for k, a, b in contiguous_windows(n, offset)]


def _covered(interval: tuple[int, int] | None, lo: int, hi: int) -> int:
    """Mocap ids of `[lo, hi]` inside the closed `interval`."""
    if interval is None:
        return 0
    return max(0, min(interval[1], hi) - max(interval[0], lo) + 1)


def _active(program: Sequence[Primitive], fid: int) -> list[str]:
    objs: set[str] = set()
    for p in program:
        for iv, obs in ((p.lh, p.obj_lh), (p.rh, p.obj_rh)):
            if iv is not None and iv[0] <= fid <= iv[1]:
                objs.update(obs)
    return sorted(objs)


def window_identity(program: Sequence[Primitive], frame_ids: np.ndarray, window: OakInk2Window) -> dict[str, Any]:
    """The window's GT identity from the task program (module docstring): `interaction`, `primary`,
    `first_objects`, `last_objects`, and the mocap ids of its first/last decode rows."""
    rows = window.rows
    lo, hi = int(frame_ids[window.start]), int(frame_ids[window.stop - 1])
    inter: Counter[str] = Counter()
    for p in program:
        for iv, obs in ((p.lh, p.obj_lh), (p.rh, p.obj_rh)):
            n = _covered(iv, lo, hi)
            for o in obs:
                if n:
                    inter[o] += n
    primary = min(inter, key=lambda o: (-inter[o], o)) if inter else None
    first_id, last_id = int(frame_ids[rows[0]]), int(frame_ids[rows[-1]])
    return {"interaction": dict(sorted(inter.items())), "primary": primary,
            "first_row_mocap_id": first_id, "last_row_mocap_id": last_id,
            "first_objects": _active(program, first_id), "last_objects": _active(program, last_id)}


def load_windows(entries: Sequence[Mapping[str, Any]]) -> list[OakInk2Window]:
    """`OakInk2Window`s from entries `{sequence, source, k, start, stop}`.

    Raises:
        ValueError: On a missing key, a bad range, an unknown source or a repeated clip id.
    """
    out: list[OakInk2Window] = []
    for e in entries:
        missing = [key for key in ("sequence", "source", "k", "start", "stop") if key not in e]
        if missing:
            raise ValueError(f"OakInk2 window entry missing {missing}: {dict(e)}")
        out.append(OakInk2Window(str(e["sequence"]), str(e["source"]), int(e["k"]), int(e["start"]), int(e["stop"])))
    ids = [w.clip_id for w in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"repeated OakInk2 clip ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return out


def oakink2_clips(reader: OakInk2Reader, windows: Sequence[OakInk2Window]) -> list[BenchClip]:
    """One `image_window` `BenchClip` per window: `path` is the view's PNG directory, `window` the image-index range,
    `decode_stride` STRIDE and `domain` the view's.

    Raises:
        FileNotFoundError: If a window reaches past the view's last image.
    """
    clips: list[BenchClip] = []
    for w in windows:
        view = VIEW_OF_SOURCE[w.source]
        n = len(reader.frame_files(w.sequence, view))
        if w.stop > n:
            raise FileNotFoundError(f"{w.clip_id}: window ends at image {w.stop}, the view has {n}")
        clips.append(BenchClip(w.clip_id, w.source, "image_window", reader.view_dir(w.sequence, view),
                               w.stop - w.start, window=(w.start, w.stop), decode_stride=STRIDE,
                               domain=DOMAIN_OF_SOURCE[w.source]))
    return clips


def image_window_files(directory: Path, start: int, stop: int, stride: int) -> tuple[list[Path], np.ndarray]:
    """`(files, image indices)` of an `image_window` clip: every `stride`-th PNG of `[start, stop)` in mocap-id order."""
    files = sorted(Path(directory).glob("*.png"), key=lambda p: int(p.stem))
    if stop > len(files):
        raise FileNotFoundError(f"{directory}: window [{start}, {stop}) past the last of {len(files)} images")
    idx = np.arange(start, stop, stride, dtype=np.int64)
    return [files[int(i)] for i in idx], idx


def make_image_window_decode_worker(sources: Mapping[str, tuple[Path, int, int, int]], fps: float = FPS) -> StageFn:
    """The `decode` stage for `image_window` clips: `clip_id -> (PNG dir, start, stop, stride)` into `decode.npz` with
    the video decode stage's keys: `frames` (N,H,W,3) uint8, `frame_idx` (N,) int64 IMAGE indices,
    `fps = fps / stride`, `stride`."""
    table = {str(k): (Path(v[0]), int(v[1]), int(v[2]), int(v[3])) for k, v in sources.items()}
    ctx = WorkerContext()

    def decode_worker(clip_id: str, artifact_dir: Path) -> None:
        if clip_id not in table:
            raise KeyError(f"no image window registered for clip {clip_id!r}")
        ctx.budget()
        directory, start, stop, stride = table[clip_id]
        files, idx = image_window_files(directory, start, stop, stride)
        frames = np.stack([_read_rgb(f) for f in files])
        save_npz(Path(artifact_dir) / DECODE_FILE, frames=frames, frame_idx=idx,
                 fps=np.array(fps / stride, dtype=np.float64), stride=np.array(stride, dtype=np.int64))

    return decode_worker
