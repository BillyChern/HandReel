"""Clip sources of the throughput benchmark (`run/pipeline/bench_throughput.py`).

- `whisking_clips`: the shipped whisking demo repeated `copies` times under distinct clip ids.
- `dexycb_clips`: explicit `"<subject>/<sequence>/<serial>"` DexYCB ids under the configured root; which sequences run
  is configuration, never a listing of the dataset. Their frames are colour JPEGs, decoded by
  `make_image_sequence_decode_worker` into the video decode stage's `decode.npz` contract.
- `doh_clips`: the official train-split clips listed in `<bench.doh_root>/manifest.json` (the 100DOH bundle's layout:
  `clips/<category>_<video_id>_<start_seconds>_<end_seconds>.mp4`); none when that root is None. Entries of any other
  split (the test clips reserved for the final evaluation) are dropped from the parsed manifest before any path is
  built, so their files are never listed, stat-ed or opened, and each selected clip carries its split into the
  benchmark manifest row (`BenchClip.manifest_source`, `"100doh:train"`).

Keyframe prompts and calibration. No engine stage writes `mask_prompt.npz` yet (a keyframe detector is later work), so
the benchmark writes one per clip before the run, untimed, from a stand-in from outside the video: for whisking the SAM
3D pre-optimization layout's silhouette on the reference frame (`whisking_prompt`, the whisking head-to-head's prompt),
for DexYCB the ground-truth label map of the first frame (`dexycb_prompt`, the real mode's prompt; ground truth is read
on that frame only). DexYCB's calibration becomes the clip's `camera.json` (`write_camera_json`), which the intrinsics
stage uses as given. 100DOH clips have no prompt source yet.

`dexycb_template_mesh` is the grasped object's GT YCB mesh, the same file `DexYCBReader.load` records as
`object_mesh_path`: the shape of a calibration run (`bench.dexycb_shape: template`).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import trimesh
from PIL import Image

from fastdo.bench.headtohead import render_silhouette
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.source_domain import SOURCE_DOMAINS, SourceDomain, domain_of_source, parse_domain
from fastdo.engine.worker_support import DECODE_FILE, WorkerContext, save_npz
from fastdo.geometry.intrinsics import CAMERA_JSON
from fastdo.io.datasets.dexycb import (  # meta.yml needs DexYCB's own YAML loader
    _load_intrinsics,
    load_dexycb_yaml,
    ycb_object_name,
)
from fastdo.io.datasets.dexycb_frames import DexYCBFrames
from fastdo.io.video import count_decoded_frames, probe_fps, sample_frame_indices
from fastdo.teacher.layout_reader import read_baseline_layout, read_keyframe_layout

__all__ = [
    "DEXYCB_FPS",
    "DOH_BENCH_SPLIT",
    "DOH_MANIFEST",
    "OPERATING_FPS",
    "RAW_STREAM_SOURCES",
    "S2_WINDOW_SEED",
    "SOURCES",
    "WINDOWED_KINDS",
    "WINDOW_SECONDS",
    "BenchClip",
    "dexycb_clips",
    "dexycb_intrinsics",
    "dexycb_prompt",
    "dexycb_template_mesh",
    "doh_clip_name",
    "doh_clip_path",
    "doh_clips",
    "doh_train_entries",
    "keyframe_row",
    "make_image_sequence_decode_worker",
    "operating_stride",
    "probe_windowed_clips",
    "whisking_clips",
    "whisking_prompt",
    "window_offset",
    "windowed_clips",
    "windowed_clips_from_inventory",
    "write_camera_json",
]

logger = logging.getLogger(__name__)

SOURCES: tuple[str, ...] = ("whisking", "dexycb", "100doh", "epic_kitchens_100", "ego4d", "hoi4d", "oakink2_ego",
                           "oakink2_allo")

WINDOWED_KINDS: tuple[str, ...] = ("video", "image_window")
"""Clip kinds that honour `BenchClip.window`: a video file, or an `image_window` (a range of one PNG directory)."""

RAW_STREAM_SOURCES: tuple[str, ...] = ("100doh", "epic_kitchens_100", "ego4d")
"""The sources S2 is about: raw video with no pose label, as opposed to the lab corpora.

S2 is "raw-stream yield and cost", and for a long time 100DOH was the only such source in the benchmark, so
`s2_status` tested `source == "100doh"` directly. With EK-100 and Ego4D windows real, that test silently
reports S2 as UNMEASURED on a run made entirely of internet video -- which is the run S2 exists to measure.
"""
DOH_MANIFEST = "manifest.json"
DOH_BENCH_SPLIT = "train"    # the only 100DOH split the benchmark reads; "test" clips are reserved for final evaluation
DEXYCB_FPS = 30.0            # RealSense D415 capture rate (`fastdo.io.datasets.dexycb`)
DEXYCB_TEMPLATE_FILE = "textured_simple.obj"
_PREFIX = "bench"


@dataclass(frozen=True)
class BenchClip:
    """One benchmark clip: `path` is a video file (`kind` "video") or a DexYCB camera directory ("image_sequence").

    `split` is the source dataset's official split when it has one (100DOH), else None.

    `domain` is the clip's typed source domain (`fastdo.engine.source_domain`): given by the builder of the route the
    clip entered by (a windowed S2 clip is egocentric), else `SOURCE_DOMAINS[source]`; after construction it is
    never None. A source the table does not know must be given a domain explicitly (ValueError otherwise).
    """

    clip_id: str
    source: str
    kind: str
    path: Path
    n_frames: int
    split: str | None = None
    window: tuple[int, int] | None = None
    """Source frame range `[start, stop)` to decode, or None for the whole file.

    The unit of work on EK-100 and Ego4D is a ~10 s window of a long video, not the video. Without this the
    decode worker's window path is unreachable -- it accepts `(path, start, stop)` and nothing could build one.
    Only `kind == "video"` honours it; an image sequence has no such range.
    """

    decode_stride: int | None = None
    """Keep every n-th source frame for THIS clip, or None to use the run's `bench.stride`.

    S2 fixes the OPERATING rate, not the stride: `s_v = max(1, floor(fps_v / 10 + 0.5))` is 5 at 50 fps and 6 at
    59.94 fps, so one EK-100 run needs both and a single run-wide stride cannot express it. `bench_stages` builds one decode worker per distinct stride.
    """

    domain: SourceDomain | None = None
    """`egocentric` or `third_person`; None at construction means `SOURCE_DOMAINS[source]` (see the class doc)."""

    def __post_init__(self) -> None:
        if self.decode_stride is not None and self.decode_stride < 1:
            raise ValueError(f"{self.clip_id}: decode_stride must be >= 1, got {self.decode_stride}")
        if self.window is not None:
            start, stop = self.window
            if self.kind not in WINDOWED_KINDS:
                raise ValueError(f"{self.clip_id}: only a video or image_window clip can carry a window, got kind "
                                 f"{self.kind!r}")
            if not 0 <= start < stop:
                raise ValueError(f"{self.clip_id}: window must be 0 <= start < stop, got [{start}, {stop})")
        domain = domain_of_source(self.source) if self.domain is None else parse_domain(self.domain)
        known = SOURCE_DOMAINS.get(self.source)
        if known is not None and known is not domain:
            raise ValueError(f"{self.clip_id}: source {self.source!r} is {known.value}, not {domain.value} "
                             "(SOURCE_DOMAINS); a clip's domain cannot contradict its source")
        object.__setattr__(self, "domain", domain)

    @property
    def decode_source(self) -> Path | tuple[Path, int, int]:
        """What `make_decode_worker` should decode: the path, or `(path, start, stop)` for a windowed clip."""
        return self.path if self.window is None else (self.path, self.window[0], self.window[1])

    @property
    def manifest_source(self) -> str:
        """The benchmark manifest's `source` value: `"<source>:<split>"` when the clip has a split, else `source`."""
        return self.source if self.split is None else f"{self.source}:{self.split}"


def whisking_clips(video: Path, n_frames: int, copies: int) -> list[BenchClip]:
    """`copies` clips of the whisking video, ids `bench_whisking_000`, `bench_whisking_001`, ...

    Raises:
        ValueError: If `copies < 0`.
    """
    if copies < 0:
        raise ValueError(f"copies must be >= 0, got {copies}")
    return [BenchClip(f"{_PREFIX}_whisking_{i:03d}", "whisking", "video", Path(video), int(n_frames))
            for i in range(copies)]


def doh_train_entries(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The `official_split == "train"` entries of `<root>/manifest.json`, and the count of entries left out per split.

    Only `manifest.json` is read. Entries without `official_split` count under `"<missing>"` and are left out too.

    Raises:
        ValueError: If the manifest is not a JSON list of objects.
    """
    entries = json.loads((Path(root) / DOH_MANIFEST).read_text())
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{Path(root) / DOH_MANIFEST} must be a JSON list of clip entries")
    train = [entry for entry in entries if entry.get("official_split") == DOH_BENCH_SPLIT]
    held_out = Counter(str(entry.get("official_split", "<missing>")) for entry in entries
                       if entry.get("official_split") != DOH_BENCH_SPLIT)
    return train, dict(sorted(held_out.items()))


def doh_clip_name(entry: Mapping[str, Any]) -> str:
    """`<category>_<video_id>_<start_seconds>_<end_seconds>`, the bundle's clip file stem."""
    return f"{entry['category']}_{entry['video_id']}_{int(entry['start_seconds'])}_{int(entry['end_seconds'])}"


def doh_clip_path(root: Path, entry: Mapping[str, Any]) -> Path:
    """The entry's clip file: its `clip_path` relative to `root`, else `clips/<doh_clip_name>.mp4`. Pure path
    arithmetic: nothing on disk is touched.

    Raises:
        ValueError: If `clip_path` is absolute or leaves `root`.
    """
    rel = PurePosixPath(str(entry.get("clip_path") or f"clips/{doh_clip_name(entry)}.mp4"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"100DOH clip_path {rel} must stay under bench.doh_root")
    return Path(root) / rel


def doh_clips(root: Path | None, count_frames: Callable[[Path], int] = count_decoded_frames) -> list[BenchClip]:
    """The train-split clips of the 100DOH bundle at `root`, in manifest order; `[]` when `root` is None or has no
    `manifest.json` (logged). The directory is never listed.

    Test-split entries are dropped before any path is built (module docstring). `n_frames` is the entry's
    `decoded_frames` when recorded, else `count_frames` of the (train) clip file.

    Raises:
        FileNotFoundError: If `root` is set but is not a directory, or a train entry's clip file is missing.
    """
    if root is None:
        return []
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"bench.doh_root {root} is not a directory")
    if not (root / DOH_MANIFEST).is_file():
        logger.warning("bench.doh_root %s has no %s: no 100DOH clip is selected without its official_split", root,
                       DOH_MANIFEST)
        return []
    train, held_out = doh_train_entries(root)
    logger.info("100DOH %s: %d %s-split clips selected; left out, never opened: %s", root / DOH_MANIFEST, len(train),
                DOH_BENCH_SPLIT, held_out)
    clips: list[BenchClip] = []
    for entry in train:
        path = doh_clip_path(root, entry)
        if not path.is_file():
            raise FileNotFoundError(f"100DOH train clip {path} listed in {root / DOH_MANIFEST} is missing")
        recorded = entry.get("decoded_frames")
        n_frames = int(recorded) if isinstance(recorded, int) and recorded > 0 else int(count_frames(path))
        clips.append(BenchClip(f"{_PREFIX}_100doh_{doh_clip_name(entry)}", "100doh", "video", path, n_frames,
                               split=DOH_BENCH_SPLIT))
    return clips


def dexycb_clips(root: Path, seq_ids: Sequence[str]) -> list[BenchClip]:
    """One clip per `"<subject>/<sequence>/<serial>"` id; `n_frames` counts the camera's colour frames.

    Raises:
        FileNotFoundError: If a sequence has no colour frames.
    """
    frames = DexYCBFrames(Path(root))
    return [BenchClip(f"{_PREFIX}_dexycb_{seq.replace('/', '_')}", "dexycb", "image_sequence", Path(root) / seq,
                      len(frames.frame_paths(seq))) for seq in seq_ids]


def keyframe_row(n_frames: int, stride: int, keyframe_frame_idx: int) -> int:
    """The decode row nearest source frame `keyframe_frame_idx`, among `sample_frame_indices(n_frames, stride)`.

    Raises:
        ValueError: If `n_frames < 1`.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    rows = np.asarray(sample_frame_indices(n_frames, stride), dtype=np.int64)
    return int(np.argmin(np.abs(rows - int(keyframe_frame_idx))))


def whisking_prompt(layout_json: Path, mesh_path: Path, keyframe_frame_idx: int,
                    image_hw: tuple[int, int] = (720, 1280)) -> np.ndarray:
    """`(H,W)` bool silhouette of the SAM 3D mesh at its pre-optimization keyframe layout (pose and scale), projected
    through the layout's intrinsics: the whisking head-to-head's keyframe prompt."""
    _, _, K = read_baseline_layout(Path(layout_json), image_width=image_hw[1], image_height=image_hw[0])
    pose, scale = read_keyframe_layout(Path(layout_json), int(keyframe_frame_idx))
    mesh = cast("trimesh.Trimesh", trimesh.load(str(mesh_path), process=False, force="mesh"))
    P = pose.double().numpy()
    posed = scale * np.asarray(mesh.vertices, dtype=np.float64) @ P[:3, :3].T + P[:3, 3]
    return np.asarray(render_silhouette(posed, np.asarray(mesh.faces, dtype=np.int64), K, image_hw), dtype=bool)


def dexycb_prompt(root: Path, seq_id: str, frame: int = 0) -> np.ndarray:
    """`(H,W)` bool pixels of the grasped object (`ycb_ids[ycb_grasp_ind]`) in colour frame `frame`'s label map.

    Raises:
        ValueError: If the object has no pixel on that frame.
    """
    object_id = _grasped_object_id(root, seq_id)
    labels = DexYCBFrames(Path(root)).frame_paths(seq_id)[frame].labels
    with np.load(labels, allow_pickle=False) as data:
        mask = np.asarray(data["seg"]) == object_id
    if not mask.any():
        raise ValueError(f"{labels}: grasped object {object_id} has no pixel on frame {frame}")
    return mask


def _grasped_object_id(root: Path, seq_id: str) -> int:
    meta = load_dexycb_yaml(Path(root) / Path(seq_id).parent / "meta.yml")
    return int(list(meta["ycb_ids"])[int(meta["ycb_grasp_ind"])])


def dexycb_template_mesh(root: Path, seq_id: str) -> Path:
    """`<root>/models/<ycb_object_name(ycb_ids[ycb_grasp_ind])>/textured_simple.obj`, the grasped object's GT mesh.

    Raises:
        FileNotFoundError: If the file is absent.
    """
    path = Path(root) / "models" / ycb_object_name(_grasped_object_id(root, seq_id)) / DEXYCB_TEMPLATE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the GT YCB mesh ({DEXYCB_TEMPLATE_FILE}) of {seq_id}'s grasped object")
    return path


def dexycb_intrinsics(root: Path, seq_id: str) -> np.ndarray:
    """`(3,3)` float32 colour intrinsics of the sequence's camera (`calibration/intrinsics/<serial>_*.yml`)."""
    return _load_intrinsics(Path(root), seq_id.split("/")[-1])


def write_camera_json(artifact_dir: Path, K: np.ndarray) -> None:
    """Writes `<artifact_dir>/camera.json` with pixel `fx`, `fy`, `cx`, `cy` (read by `read_camera_json`)."""
    K = np.asarray(K, dtype=np.float64)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / CAMERA_JSON
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}))
    os.replace(tmp, path)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def make_image_sequence_decode_worker(sequences: Mapping[str, Path], stride: int = 3,
                                      fps: float = DEXYCB_FPS) -> StageFn:
    """The `decode` stage for image-sequence clips: every `stride`-th `color_*.jpg` (sorted) into `decode.npz`, with the
    video decode stage's keys: `frames` (N,H,W,3) uint8, `frame_idx` (N,) int64, `fps = fps / stride`, `stride`.

    Args:
        sequences: `clip_id -> directory of color_*.jpg frames`.
        stride: Keep every `stride`-th frame, starting at frame 0.
        fps: Capture rate of the sequence.

    Raises:
        ValueError: If `stride < 1`, at construction.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    table = {str(clip_id): Path(path) for clip_id, path in sequences.items()}
    ctx = WorkerContext()

    def decode_worker(clip_id: str, artifact_dir: Path) -> None:
        if clip_id not in table:
            raise KeyError(f"no image sequence registered for clip {clip_id!r}")
        ctx.budget()
        colors = sorted(table[clip_id].glob("color_*.jpg"))
        if not colors:
            raise FileNotFoundError(f"no color_*.jpg frames under {table[clip_id]}")
        rows = sample_frame_indices(len(colors), stride)
        frames = np.stack([_read_rgb(colors[row]) for row in rows])
        save_npz(Path(artifact_dir) / DECODE_FILE, frames=frames, frame_idx=np.asarray(rows, dtype=np.int64),
                 fps=np.array(fps / stride, dtype=np.float64), stride=np.array(stride, dtype=np.int64))
        logger.info("decoded %d of %d frames of %r (stride %d)", len(rows), len(colors), clip_id, stride)

    return decode_worker


# --- S2 windowing, detector-free ------------------------------------------------------------------------------

S2_WINDOW_SEED = "20260914"      # the windowing seed; the offset is a function of it and the video id
WINDOW_SECONDS = 10.0            # window length AND stride: [a, a+10), no overlap
OPERATING_FPS = 10.0             # the rate the pipeline is validated at; s_v is chosen to hold it, not fixed


def window_offset(video_id: str, seed: str = S2_WINDOW_SEED) -> float:
    """`o_v = random.Random(f"{seed}:{video_id}").uniform(0, 10)` -- the fixed per-video head offset.

    Seeded on the video id alone, so it is reproducible from the id and independent of any pixel, annotation or
    pipeline output. Changing `seed` changes every window placement and invalidates the registered S2 windows.
    """
    return random.Random(f"{seed}:{video_id}").uniform(0.0, WINDOW_SECONDS)


def operating_stride(fps: float, operating_fps: float = OPERATING_FPS) -> int:
    """`s_v = max(1, floor(fps_v / 10 + 0.5))`: 5 at 50 fps, 6 at 59.94 fps.

    The S2 windowing rule fixes the OPERATING RATE, not the stride, so videos of different capture rates get
    different strides and land at about the same rows per second.

    Raises:
        ValueError: If `fps` or `operating_fps` is not positive.
    """
    if not fps > 0 or not operating_fps > 0:
        raise ValueError(f"fps and operating_fps must be positive, got {fps} and {operating_fps}")
    return max(1, math.floor(fps / operating_fps + 0.5))


def windowed_clips(video_id: str, path: Path, duration_s: float, fps: float, source: str,
                   seed: str = S2_WINDOW_SEED, split: str | None = None) -> list[BenchClip]:
    """Every eligible 10 s window of one long video, as `BenchClip`s the engine can actually decode.

    Implements the S2 windowing rule exactly: offset `o_v`, windows `[o_v + 10k, o_v + 10(k+1))` eligible while
    `a + 10 <= duration_s`, source frames `[ceil(a * fps), ceil((a + 10) * fps))`, stride `s_v`, and the window id
    `"{video_id}:k{k:03d}"` used verbatim as the clip id so the artifact directory and the reported id are the
    same string.

    **Egocentric by route:** every window is `SourceDomain.EGOCENTRIC` (the S2 windowed route is egocentric
    video); a source `SOURCE_DOMAINS` calls third-person raises.

    **Detector-free by construction:** the only inputs are the id, the duration and the frame rate. No pixel,
    annotation or pipeline output can reach window placement, which is what makes the yield denominator
    well defined. `duration_s` and `fps` are measured at run time, after the freeze.

    Args:
        video_id: The source video's id, e.g. `"P18_03"`. May not contain a path separator.
        path: The video file.
        duration_s: `D_v`, decoded frames / avg_frame_rate.
        fps: `fps_v`, ffprobe `avg_frame_rate`.
        source: The manifest `source` for these clips, e.g. `"epic_kitchens_100"`.
        seed: The window seed; see `window_offset`.
        split: The dataset split, carried onto every window.

    Returns:
        The eligible windows in order, `k = 0, 1, ...`; empty when no whole window fits. `n_frames` is the
        window's SOURCE frame count (`stop - start`), not its decode-row count, matching every other builder
        here: the profiler reports `n_frames / fps` as the seconds of video a clip represents, which is the
        denominator of the S2 yield law. Rows are `n_frames / s_v` and come back on `decode.npz`.

    Raises:
        ValueError: On a video_id containing a path separator, or a non-positive `fps`.
    """
    if "/" in video_id or os.sep in video_id:
        raise ValueError(f"video_id becomes a directory name, so it may not contain a separator: {video_id!r}")
    stride = operating_stride(fps)
    offset = window_offset(video_id, seed)
    clips: list[BenchClip] = []
    k = 0
    while offset + WINDOW_SECONDS * (k + 1) <= duration_s:
        start_s = offset + WINDOW_SECONDS * k
        start, stop = math.ceil(start_s * fps), math.ceil((start_s + WINDOW_SECONDS) * fps)
        clips.append(BenchClip(clip_id=f"{video_id}:k{k:03d}", source=source, kind="video", path=Path(path),
                               n_frames=stop - start, split=split, window=(start, stop), decode_stride=stride,
                               domain=SourceDomain.EGOCENTRIC))
        k += 1
    return clips


def probe_windowed_clips(video_id: str, path: Path, source: str, split: str | None = None,
                         seed: str = S2_WINDOW_SEED, fps_of: Callable[[Path], float] = probe_fps,
                         count_frames: Callable[[Path], int] = count_decoded_frames) -> list[BenchClip]:
    """`windowed_clips` with `D_v` and `fps_v` MEASURED from the file, the way the windowing rule defines them.

    `D_v` is DECODED frames / `avg_frame_rate`, not the container's `nb_frames`: a container count includes
    non-decoding packets, so using it would place the last window past the end of the video and the run would
    fail on a clip the windowing rule never declared eligible. Both probes are run-time measurements, which
    the frozen rule permits.

    That reasoning is measured, not cautionary: whisking reports 172 packets against 138 decodable frames,
    and on Ego4D `fee6098f` a container DURATION admits an 88th window whose last index is 26664 in a
    26660-frame file. A container count is nonetheless PERMITTED for a source validated against
    decoded counts -- `nb_frames` reproduced `D_v` exactly on all 21 Ego4D videos, giving the identical
    window set. Pass `count_frames=fastdo.io.video.screened_frame_count` to take that route, which screens
    each file for container-level trimming and falls back to decoding otherwise. The DEFAULT stays the
    decoded count, so a run opts in rather than inheriting a permission scoped to one corpus.

    Raises:
        ValueError: If `avg_frame_rate` does not probe as positive.
    """
    path = Path(path)
    fps = fps_of(path)
    if not fps > 0:
        raise ValueError(f"{path}: avg_frame_rate probed as {fps}, so no window can be placed")
    return windowed_clips(video_id, path, count_frames(path) / fps, fps, source, seed=seed, split=split)


def _spread_indices(n: int, k: int) -> list[int]:
    """`k` indices spread evenly over `range(n)`, both ends included; all of them when `k >= n`.

    Systematic rather than random, to match the draw's own systematic design and stay reproducible without a
    second seed. It is unbiased in expectation for a refusal rate unless the content is periodic at the
    sampling period, which for 10 s windows of hours-long unscripted video it is not.
    """
    if n <= 0 or k < 1:
        raise ValueError(f"n must be positive and k at least 1, got n={n}, k={k}")
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def windowed_clips_from_inventory(inventory: Path, root: Path | None = None, seed: str = S2_WINDOW_SEED,
                                  max_per_video: int | None = None,
                                  fps_of: Callable[[Path], float] = probe_fps,
                                  count_frames: Callable[[Path], int] = count_decoded_frames) -> list[BenchClip]:
    """Every window of every video in a JSON inventory -- the registered runner's route to a windowed clip list.

    The inventory is a list of objects with `video_id`, `path`, `source` and an optional `split`; `path` is
    resolved against `root` when relative. It names videos only: the windows themselves come from
    `probe_windowed_clips`, so nothing about the sampling is editable from the inventory except which videos
    take part, which is what the draw already fixes.

    `max_per_video` keeps `k` windows of each video, SPREAD EVENLY over the video, for a BOUNDED PILOT. It
    is not a registered S2 run: the S2 numbers need every eligible window, so the yield denominator of a
    truncated run is not the corpus.

    The spread is the point. Taking the first `k` -- which is what this did at first -- samples the OPENING
    of every video, and in egocentric footage the first seconds are disproportionately walking in, the camera
    settling and the scene being set up, which is the least likely place to find a hand holding an object.
    The first S2 pilot refused both opening windows of one video and passed both of another, a 50% refusal
    rate that sat exactly on the refusal gate's registered bound and may have been mostly an artefact of where it
    sampled. A head-biased pilot can fail that gate for reasons that have nothing to do with the pipeline. Setting it logs a warning saying so, and the default is None.

    Raises:
        ValueError: If the file is not a list of objects, an entry lacks a required key or names an unknown
            `source`, or two entries share a `video_id`.
        FileNotFoundError: If the inventory or one of its videos does not exist.
    """
    inventory = Path(inventory)
    entries = json.loads(inventory.read_text())
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{inventory}: a window inventory is a list of objects, got {type(entries).__name__}")
    clips: list[BenchClip] = []
    seen: set[str] = set()
    for entry in entries:
        missing = [k for k in ("video_id", "path", "source") if k not in entry]
        if missing:
            raise ValueError(f"{inventory}: an entry is missing {missing}: {entry}")
        video_id, source = str(entry["video_id"]), str(entry["source"])
        if source not in SOURCES:
            raise ValueError(f"{inventory}: {video_id} names source {source!r}; known sources: {SOURCES}")
        if video_id in seen:
            raise ValueError(f"{inventory}: {video_id} appears twice; window ids would collide")
        seen.add(video_id)
        path = Path(entry["path"])
        path = path if path.is_absolute() or root is None else Path(root) / path
        if not path.is_file():
            raise FileNotFoundError(f"{inventory}: {video_id} names {path}, which is not a file")
        windows = probe_windowed_clips(video_id, path, source, split=entry.get("split"), seed=seed,
                                       fps_of=fps_of, count_frames=count_frames)
        if max_per_video is not None:
            if max_per_video < 1:
                raise ValueError(f"max_per_video must be >= 1 when set, got {max_per_video}")
            kept = _spread_indices(len(windows), max_per_video)
            logger.warning("%s: keeping %d of %d windows, spread evenly (k = %s) -- a TRUNCATED run, whose "
                           "window count is NOT the S2 corpus and whose yield denominator is therefore not "
                           "the corpus's", video_id, len(kept), len(windows), [windows[i].clip_id[-4:] for i in kept])
            windows = [windows[i] for i in kept]
        logger.info("%s: %d windows (%s)", video_id, len(windows), source)
        clips.extend(windows)
    return clips
