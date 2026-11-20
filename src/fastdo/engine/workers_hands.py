"""The `hands` stage worker: `hands.npz` from a HaWoR-format hand npz.

Two paths, both writing `hands.npz` rows that are exactly `decode.npz`'s rows:
- precomputed: a clip with `<artifact_dir>/all_hand_meshes.npz` (or a `sources` entry) is read by
  `write_hands_artifact`;
- batched HaWoR: otherwise `fastdo.hands.hawor_batched.run_hands_batched` runs this one clip on `decode.npz` with
  focal length `K[0,0]` from `intrinsics.npz` (HaWoR takes one focal length and the image centre). It writes
  `<artifact_dir>/hands_hawor.npz`, and `write_hands_artifact` reads that back. The engine normally runs many clips per
  model load through `fastdo.engine.hands_batch.run_hands_stage`; this worker then retries failed clips one at a time.

Rows are selected from the source npz by its `frame_idx` key (`arange(T_source)` when absent), so the hand T equals the
solution T downstream (`ClipRecord.validate`, the retarget worker). Keys, per side (`left`, `right`):

- `{side}_params` (T,48), `{side}_trans` (T,3), `{side}_joints` (T,21,3) float32 and `{side}_valid` (T,) bool, the
  `HandTrack` fields of `load_hawor_npz`, row-selected;
- `{side}_betas` (10,) float32: the mean of the source's per-frame betas over the selected valid rows, else over the
  selected rows;
- `{side}_vertices` (T,778,3) float32 and `{side}_faces` (1552,3) int32, the source's raw mesh arrays
  (`HandTrack` carries no vertices; the retarget stage reads them from here);
- `{side}_infilled` (T,) bool: the source key, row-selected, else all False;

plus `frame_idx` (T,) int64, the selected source frame indices, and the two cameras (below). Written uncompressed,
tmp file then `os.replace`. On the batched path, invalid rows (`{side}_valid` False) hold finite copies of the nearest
valid row and must be masked.

Camera (`fastdo.hands.hand_camera`). A HaWoR npz, batched or precomputed, is in
HaWoR's camera: focal length `K[0,0]` and principal point `(W/2, H/2)` of the decoded frames. The writer moves
`{side}_joints`, `{side}_vertices` and `{side}_trans` into the projection camera `K` of `intrinsics.npz` with the
depth-preserving principal-point shear. `params`, `betas`, `valid` and `infilled` are unchanged. It records
`camera_K` (3,3) float64, the camera the outputs are in (`K`), and `hawor_camera_K` (3,3) float64, HaWoR's camera.
The shear is a bitwise no-op when `K`'s principal point is the image centre (MoGe-2 intrinsics). A source in any other
camera, such as ground-truth hands already in `K`, must not be passed through this writer. Consumers check the record
with `fastdo.hands.hand_camera.require_hands_camera`.
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import (
    DECODE_FILE,
    WorkerContext,
    read_frame_idx,
    require_artifact,
    rows_by_frame_index,
    save_npz,
)
from fastdo.geometry.intrinsics import INTRINSICS_FILE, read_intrinsics
from fastdo.hands.hand_camera import (
    CAMERA_KEY,
    HAWOR_CAMERA_KEY,
    hawor_camera,
    principal_point,
    principal_point_shear,
)
from fastdo.hands.hawor_batched import HandsClipInput, HandsConfig, run_hands_batched
from fastdo.hands.hawor_npz import load_hawor_npz

__all__ = ["HANDS_FILE", "HANDS_JOB_DIR", "HANDS_OUTPUT", "HAWOR_FILE", "SIDES", "decode_frame_size",
           "make_hands_worker", "write_hands_artifact"]

logger = logging.getLogger(__name__)

HANDS_FILE = "hands.npz"
HAWOR_FILE = "all_hand_meshes.npz"
HANDS_OUTPUT = "hands_hawor.npz"                                     # the batched driver's per-clip output
HANDS_JOB_DIR = "hands_job"
SIDES: tuple[str, ...] = ("left", "right")


def decode_frame_size(artifact_dir: Path) -> tuple[int, int]:
    """`(width, height)` of the `(N,H,W,3)` frames in `<artifact_dir>/decode.npz`, read from the array header only.

    Raises:
        FileNotFoundError: If `decode.npz` is absent.
        ValueError: If its `frames` array is not 4-D.
    """
    path = require_artifact(Path(artifact_dir) / DECODE_FILE, "decode")
    with zipfile.ZipFile(path) as archive, archive.open("frames.npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape = np.lib.format.read_array_header_1_0(f)[0]
        else:
            shape = np.lib.format.read_array_header_2_0(f)[0]
    if len(shape) != 4:
        raise ValueError(f"{path}: frames must be (N,H,W,3), got {tuple(shape)}")
    return int(shape[2]), int(shape[1])


def _cameras(artifact_dir: Path, K: np.ndarray | None, image_size: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """`(projection camera, HaWoR camera)`, both (3,3) float64."""
    if K is None:
        try:
            K, _ = read_intrinsics(artifact_dir)
        except FileNotFoundError:
            raise FileNotFoundError(f"{artifact_dir / INTRINSICS_FILE} not found: the hands writer puts HaWoR outputs in "
                                    f"the clip's projection camera; run the intrinsics stage first") from None
    width, height = decode_frame_size(artifact_dir) if image_size is None else (int(image_size[0]), int(image_size[1]))
    camera = np.asarray(K, dtype=np.float64)
    return camera, hawor_camera(camera, width, height)


def write_hands_artifact(hawor_npz: Path, artifact_dir: Path, frame_idx: np.ndarray | None = None,
                         K: np.ndarray | None = None, image_size: tuple[int, int] | None = None) -> None:
    """Writes `<artifact_dir>/hands.npz` from a HaWoR npz, one row per requested source frame index, with the hand
    outputs moved from HaWoR's camera into the projection camera (module docstring).

    Args:
        hawor_npz: A HaWoR `all_hand_meshes.npz` (both hands), in HaWoR's camera.
        artifact_dir: The clip's artifact directory.
        frame_idx: Source frame indices to select; `None` reads them from `<artifact_dir>/decode.npz`.
        K: The `(3,3)` projection camera; `None` reads `<artifact_dir>/intrinsics.npz`.
        image_size: `(width, height)` of the frames HaWoR ran on; `None` reads the `decode.npz` frames' header.

    Raises:
        ValueError: If the source lacks a requested frame index (the message names the first one), repeats a frame
            index, or has inconsistent row counts across `frame_idx`, `{side}_vertices`, `{side}_joints`,
            `{side}_infilled` (when present) and per-frame `(T,10)` `{side}_betas`.
        FileNotFoundError: If `decode.npz` is absent and `frame_idx` or `image_size` is None, or if `K` is None and
            `intrinsics.npz` is absent.
    """
    hawor_npz, artifact_dir = Path(hawor_npz), Path(artifact_dir)
    wanted = read_frame_idx(artifact_dir) if frame_idx is None else np.asarray(frame_idx, dtype=np.int64).reshape(-1)
    tracks = {side: load_hawor_npz(hawor_npz, side) for side in SIDES}
    t_source = int(tracks["right"].params.shape[0])
    with np.load(hawor_npz, allow_pickle=False) as data:
        source = {key: np.asarray(data[key]) for key in data.files if not key.endswith(("_rot", "_hand_pose"))}
    source_idx = np.asarray(source.get("frame_idx", np.arange(t_source)), dtype=np.int64).reshape(-1)
    row_counts: dict[str, int] = {"frame_idx": int(source_idx.shape[0])}
    for side in SIDES:
        per_row = [f"{side}_vertices", f"{side}_joints"]
        per_row += [key for key in (f"{side}_infilled",) if key in source]      # optional
        if source[f"{side}_betas"].ndim == 2:                                    # (T,10) per frame; (10,) is per clip
            per_row.append(f"{side}_betas")
        row_counts |= {key: int(source[key].shape[0]) if source[key].ndim else 0 for key in per_row}
    if any(count != t_source for count in row_counts.values()):
        raise ValueError(f"{hawor_npz}: inconsistent row counts {row_counts}, expected {t_source} everywhere")
    rows = rows_by_frame_index(source_idx, wanted, str(hawor_npz))
    camera, hawor_K = _cameras(artifact_dir, K, image_size)
    hawor_pp = principal_point(hawor_K)

    arrays: dict[str, np.ndarray] = {"frame_idx": source_idx[rows], CAMERA_KEY: camera, HAWOR_CAMERA_KEY: hawor_K}
    for side in SIDES:
        track = tracks[side]
        valid = track.valid[rows]
        betas = np.asarray(source[f"{side}_betas"], dtype=np.float32)
        if betas.ndim == 2:
            selected = betas[rows]
            betas = (selected[valid] if valid.any() else selected).mean(axis=0)
        infilled = source.get(f"{side}_infilled")
        arrays |= {
            f"{side}_params": track.params[rows],
            f"{side}_betas": betas.astype(np.float32),
            f"{side}_trans": principal_point_shear(track.trans[rows], hawor_pp, camera),
            f"{side}_joints": principal_point_shear(track.joints[rows], hawor_pp, camera),
            f"{side}_valid": valid,
            f"{side}_vertices": principal_point_shear(source[f"{side}_vertices"][rows].astype(np.float32), hawor_pp,
                                                      camera),
            f"{side}_faces": source[f"{side}_faces"].astype(np.int32),
            f"{side}_infilled": (np.zeros(t_source, bool) if infilled is None else infilled.astype(bool))[rows],
        }
    save_npz(artifact_dir / HANDS_FILE, **arrays)
    logger.info("wrote %d hand rows from %s (principal point %s -> %s)", rows.size, hawor_npz, hawor_pp.tolist(),
                principal_point(camera).tolist())


def make_hands_worker(cfg: HandsConfig | None = None, sources: Mapping[str, Path] | None = None) -> StageFn:
    """The `hands` stage worker (see the module docstring).

    Args:
        cfg: Settings of the batched HaWoR path; None allows precomputed npz files only.
        sources: Optional `clip_id -> all_hand_meshes.npz`; a clip not listed uses `<artifact_dir>/all_hand_meshes.npz`.

    Without a precomputed npz the worker raises `RuntimeError` when `cfg` is None, `ValueError` when the clip has no
    `intrinsics.npz`, and `RuntimeError` with the driver's message when the batched clip fails.
    """
    table = {} if sources is None else {str(clip_id): Path(path) for clip_id, path in sources.items()}
    ctx = WorkerContext()

    def hands_worker(clip_id: str, artifact_dir: Path) -> None:
        ctx.budget()
        artifact_dir = Path(artifact_dir)
        npz = table.get(clip_id, artifact_dir / HAWOR_FILE)
        if npz.is_file():
            write_hands_artifact(npz, artifact_dir)
            return
        if cfg is None:
            raise RuntimeError("hands_worker needs a HandsConfig for the batched HaWoR path")
        try:
            K, _ = read_intrinsics(artifact_dir)
        except FileNotFoundError:
            raise ValueError("hands_worker needs intrinsics.npz; run the intrinsics stage first") from None
        clip = HandsClipInput(clip_id, artifact_dir / DECODE_FILE, float(K[0, 0]), artifact_dir / HANDS_OUTPUT)
        (result,) = run_hands_batched([clip], cfg, artifact_dir / HANDS_JOB_DIR).clips
        if result.status != "done":
            raise RuntimeError(result.error)
        write_hands_artifact(artifact_dir / HANDS_OUTPUT, artifact_dir)

    return hands_worker
