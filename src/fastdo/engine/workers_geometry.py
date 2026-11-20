"""Camera stage workers: `intrinsics` (one K per clip) and `geometry` (metric MoGe-2 pointmaps).

`intrinsics` runs after `shape` and before `hands` and `geometry` (`fastdo.engine.manifest.STAGE_NAMES`). It writes
`intrinsics.npz` (`fastdo.geometry.intrinsics`): the clip's `camera.json` calibration when the artifact directory
has one (`source="given"`; the model is not loaded), else the entry-wise median of MoGe-2's normalized intrinsics on
`n_keyframes` evenly spaced decode rows, in pixels (`source="moge2"`).

`geometry` runs MoGe-2 on every decode row (the engine's 10 fps keyframes) through the stage's `K`,
`run_moge(frames, device, model, intrinsics=K)`, so the pointmaps and the metric scale use one camera. It
rescales them to metres with one clip-level factor from `scale_fn(pointmaps, hand_joints, K, valid)`: a
per-frame result, like the default `hand_anchored_scale`, is pooled with `clip_median_scale`, and a single number is
used as is. The hand joints are the `hands.npz` joints of `hand_sides`, rows matched to the decode rows by source frame
index. The default is the right hand alone, the protocol of the whisking head-to-head (`whisking.hand_side`, the demo's
`anchor_hand`), where the hand-anchored scale's 5-21% bias was measured. `hand_sides=SIDES` pools both hands' joints: a
different statistic, to which those bias numbers do not carry over. A clip whose selected hands leave no usable joint
gets a NaN scale and fails with a ValueError naming `hand_sides`. The joints must be in the stage's `K`:
`fastdo.hands.hand_camera.require_hands_camera` raises a ValueError when `hands.npz` records another camera, or records
none (HaWoR's image-centre principal point) while `K`'s principal point is more than 1 px from the image centre.
`geometry.npz` holds `pointmaps` (N,H,W,3) float32 metres (NaN where MoGe marked a pixel invalid), `frame_idx` (N,)
int64, `scale` 0-d float32 and `frame_scale` (N,) float32 (the scale function's per-frame output, NaN where unset).

`geometry` is frame-batched: `GeometryWorker.run_many` streams the decode rows of clips that share a camera (frame size
and `K`), in clip order, through `run_moge` calls of at most `max_frames_per_pass` frames. A call can span clips, and a
clip longer than `max_frames_per_pass` is split across calls. Clips with different cameras cannot share a call, since
MoGe takes the field of view per call. MoGe's batch size bounds GPU memory either way (`moge_batch_size`, default
`moge_model._BATCH_SIZE` = 2; a per-worker knob, `geometry_stage.moge_batch_size`, not bit-neutral).
`moge_cuda_graph` (default False, `geometry_stage.moge_cuda_graph`) makes each worker install
`fastdo.geometry.moge_cuda_graph.graphed_moge_forward` on its loaded model once: the MoGe-2 forward is captured per
input signature and replayed, the same kernels as eager batch 2; opt-in until its GPU bit-identity gate passes. False
leaves the model untouched. `moge_overlap` (default False, `geometry_stage.moge_overlap`; the geometry worker only)
pipelines its MoGe batches (`run_moge_batches(overlap=True)`, `fastdo.geometry.moge_overlap`): batch k's CPU shift
solve runs while batch k+1's forward is on the GPU, meant bit-identical; opt-in until its GPU gate passes.
`moge_num_tokens` (default None, `geometry_stage.moge_num_tokens`; BOTH workers) is MoGe-2's base ViT token count
(`infer(num_tokens=...)`, on the serial and the overlapped path alike). None never reaches `infer` (its default, 3600:
byte-identical to before); an int in `moge_model.NUM_TOKENS_RANGE` is faster but moves the depth, so it waits
on the pose gate and an Ego4D gauge.
`max_frames_per_pass` bounds each call's host arrays (the call's frames and MoGe's float pointmaps). It does not bound
the artifact: a clip's `pointmaps` are one (N,H,W,3) float32 array (about 11 GB for 1000 frames at 720p), which the
worker assembles, rescales in place and writes, one clip at a time. Each worker loads MoGe-2 once, on first use, after
the CUDA cap; pass one loaded `model` to both factories to share a single copy.

With `pointmap_storage="masked"` and the default `hand_anchored_scale`, a call whose clips are all whole in it
(the production case: one S2 window per call) never brings the full-resolution pointmaps to the host. Per MoGe
batch it gathers on the device only each row's object pixels, the keyframe row and the hand-joint pixels the scale
reads (`run_moge_batches`), then rescales and writes them: the same `geometry.npz` arrays, bit for bit, `row_stride`
included. A clip split across calls or a custom `scale_fn` (whose pixels are unknown) takes the dense path above.
The gathered path reads `mask.npz` before running MoGe, so a bad mask fails the call before inference, not after.

`hand_depth_scale` (default None = off; `geometry_stage.hand_depth_scale`) is a per-side relative correction of
the hand depth the metric scale is anchored on. The anchor is, per frame, k_t = mean_j Z_j / mean_j P_t(u_j, v_j)_z
(`hand_anchored_scale`): the hand joints' depth over the raw pointmap's depth at the joints' own pixels. Rescaling a
joint about the camera centre, X_j -> f X_j, leaves its pixel unchanged (u = fx fX / fZ + cx) and multiplies its depth
by f. With one side anchoring (right, left, or auto), every k_t is multiplied by f, so is their median (f > 0), and so
is every metric pointmap: THE METRIC SCALE MOVES BY f, NOT 1/f. HaWoR placing a side's hands a relative r too far,
Z_hawor = (1 + r) Z_true, inflates k by (1 + r); f_side = 1 / (1 + r_side) removes it. Measured: r_right = +2.1%
(f < 1: the right-anchored scene shrinks) and r_left = -1.1% (f > 1: it grows). With both sides pooled (`SIDES`), each
joint's depth takes its own side's f. The rescale is applied as "the recorded joints' pixels, depth x f"
(`hand_anchor_pixels(depth_scale=...)`): the same thing in exact arithmetic, without float32 rounding moving a
truncated pixel. Factors of exactly 1.0 on the side(s) a clip uses are no correction and take the uncorrected path
(byte-identical, no provenance key); otherwise `geometry.npz` also records `hand_depth_scale` (one float64 per entry of
`hand_sides`). Only this stage's anchor changes: the contact stage reads `hands.npz` itself and is not corrected.

`ego_motion` (default None / disabled, `geometry_stage.ego_motion`) additionally writes `ego_motion.npz`, the
window's camera trajectory (`fastdo.engine.geometry_ego`, `fastdo.geometry.ego_motion`): background pixels are
tracked on the decode rows before MoGe runs, the gathered path copies only those pointmap samples off the device (the
dense path reads them from its arrays), and the Sim(3) pose graph runs after the clip's metric scale is applied.
`geometry.npz` is unchanged byte for byte either way. It needs `row_stride` 1.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.engine.geometry_ego import finish_clip, plan_for_clip
from fastdo.engine.geometry_io import check_masked_shapes, masked_payload, pack_masked
from fastdo.engine.npz_rows import npz_array_shape
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import (
    DECODE_FILE,
    BatchedWorker,
    WorkerContext,
    load_npz,
    read_decode,
    read_decode_rows,
    require_artifact,
    rows_by_frame_index,
    save_npz,
)
from fastdo.engine.workers_hands import HANDS_FILE, SIDES
from fastdo.geometry import moge_cuda_graph as moge_graph
from fastdo.geometry import moge_model
from fastdo.geometry.ego_motion import EgoMotionConfig
from fastdo.geometry.ego_tracks import EgoPlan
from fastdo.geometry.intrinsics import (
    CAMERA_JSON,
    check_focal_scale,
    pixel_intrinsics,
    read_camera_json,
    read_intrinsics,
    write_intrinsics,
)
from fastdo.geometry.moge_model import run_moge, run_moge_batches
from fastdo.geometry.moge_pointmaps import (
    clip_median_scale,
    hand_anchor_pixels,
    hand_anchored_scale,
    hand_anchored_scale_from_samples,
)
from fastdo.hands.hand_camera import require_hands_camera

__all__ = [
    "DEFAULT_HAND_SIDES",
    "DEFAULT_MAX_FRAMES_PER_PASS",
    "GEOMETRY_FILE",
    "HAND_DEPTH_SCALE_RANGE",
    "HAND_SIDES_AUTO",
    "GeometryWorker",
    "ScaleFn",
    "check_hand_depth_scale",
    "keyframe_rows",
    "make_geometry_worker",
    "make_intrinsics_worker",
]

logger = logging.getLogger(__name__)

GEOMETRY_FILE = "geometry.npz"
POINTMAP_STORAGES: tuple[str, ...] = ("dense", "masked")
# The measured protocol: the whisking demo's anchor hand (`whisking.hand_side` in run/conf/config.yaml).
DEFAULT_HAND_SIDES: tuple[str, ...] = ("right",)
# `hand_sides="auto"`: use the right hand whenever it has ANY valid frame, and the left only when it has none.
# Deliberately right-PREFERRING rather than "whichever side has more valid frames": that makes it bit-identical
# on every clip the fixed ("right",) default already handles, so it recovers the dropped clips without moving
# any measured number. Measured on DexYCB subjects 03-06, 64 of 162 clips (39.5%) have no valid right hand and
# a fully valid left one, and every one of them is dropped at this stage AFTER the whole front end has been
# paid.
HAND_SIDES_AUTO = "auto"
# A per-side factor outside this range is refused. The measured side biases are -1.1% and +2.1%; a factor of,
# say, 1.021 or 0.021 typed as a percentage, or 2.1, is a configuration error, not a calibration.
HAND_DEPTH_SCALE_RANGE: tuple[float, float] = (0.8, 1.25)
# `(pointmaps (N,H,W,3), hand_joints (N,J,3), K (3,3), valid (N,J)) -> (N,) per-frame scale or one scale`.
ScaleFn = Callable[..., Tensor | float]


def check_hand_depth_scale(hand_depth_scale: Any) -> dict[str, float] | None:
    """`hand_depth_scale` validated: None, or a mapping with exactly the keys of `SIDES`, each a real number
    (not a bool) in `HAND_DEPTH_SCALE_RANGE`. Returns None for None and for all factors exactly 1.0 (no correction).

    Raises:
        ValueError: On anything else.
    """
    if hand_depth_scale is None:
        return None
    try:
        keys = set(hand_depth_scale.keys())
    except AttributeError:
        raise ValueError(f"hand_depth_scale must be None or a mapping {{left: f, right: f}}, got "
                         f"{hand_depth_scale!r}") from None
    if keys != set(SIDES):
        raise ValueError(f"hand_depth_scale must have exactly the keys {sorted(SIDES)}, got {sorted(map(str, keys))}")
    lo, hi = HAND_DEPTH_SCALE_RANGE
    factors: dict[str, float] = {}
    for side in SIDES:
        f = hand_depth_scale[side]
        if isinstance(f, bool) or not isinstance(f, (int, float, np.integer, np.floating)):
            raise ValueError(f"hand_depth_scale[{side!r}] must be a number, got {f!r}")  # noqa: TRY004 - config errors are ValueErrors
        if not (math.isfinite(float(f)) and lo <= float(f) <= hi):
            raise ValueError(f"hand_depth_scale[{side!r}] must be finite and in [{lo}, {hi}], got {f!r}")
        factors[side] = float(f)
    return None if all(f == 1.0 for f in factors.values()) else factors


def keyframe_rows(n: int, n_keyframes: int) -> np.ndarray:
    """Evenly spaced decode rows: `np.unique(np.round(np.linspace(0, n - 1, n_keyframes)).astype(np.int64))`."""
    if n < 1 or n_keyframes < 1:
        raise ValueError(f"n and n_keyframes must be >= 1, got {n} and {n_keyframes}")
    return np.unique(np.round(np.linspace(0, n - 1, n_keyframes)).astype(np.int64))


def _check_moge_batch_size(moge_batch_size: int | None) -> int | None:
    """`moge_batch_size` unchanged if `None` (MoGe's own default) or an int >= 1.

    Raises:
        ValueError: On anything else (bools included), at worker construction.
    """
    if moge_batch_size is not None and (not isinstance(moge_batch_size, int) or isinstance(moge_batch_size, bool)
                                        or moge_batch_size < 1):
        raise ValueError(f"moge_batch_size must be an int >= 1 or None, got {moge_batch_size!r}")
    return moge_batch_size


def _check_moge_cuda_graph(moge_cuda_graph: bool) -> bool:
    """`moge_cuda_graph` unchanged if a bool.

    Raises:
        ValueError: On anything else, at worker construction.
    """
    if not isinstance(moge_cuda_graph, bool):
        raise ValueError(f"moge_cuda_graph must be a bool, got {moge_cuda_graph!r}")  # noqa: TRY004  # as siblings
    return moge_cuda_graph


def _check_moge_overlap(moge_overlap: bool) -> bool:
    """`moge_overlap` unchanged if a bool.

    Raises:
        ValueError: On anything else, at worker construction.
    """
    if not isinstance(moge_overlap, bool):
        raise ValueError(f"moge_overlap must be a bool, got {moge_overlap!r}")  # noqa: TRY004  # as siblings
    return moge_overlap


def _check_moge_num_tokens(moge_num_tokens: int | None) -> int | None:
    """`moge_num_tokens` unchanged if `None` (MoGe's default, 3600) or an int in `moge_model.NUM_TOKENS_RANGE`.

    Raises:
        ValueError: On anything else.
    """
    return moge_model.check_num_tokens(moge_num_tokens, "moge_num_tokens")


def _tokens_kwarg(moge_num_tokens: int | None) -> dict[str, int]:
    """The `num_tokens` keyword for the `moge_model` calls: absent for `None`, so the default worker makes exactly the
    calls it makes without the knob."""
    return {} if moge_num_tokens is None else {"num_tokens": moge_num_tokens}


def _moge(ctx: WorkerContext, cuda_graph: bool) -> Any:
    """The worker's MoGe-2 model; with `cuda_graph`, its graphed forward installed (idempotent, so the first
    call installs it and later calls, or a second worker sharing the model, reuse the same runner and graphs)."""
    model = ctx.model()
    if cuda_graph:
        moge_graph.graphed_moge_forward(model)
    return model


def _log_graph_stats(model: Any, what: str) -> None:
    """Logs (and resets) the graph runner's counters after one MoGe call; a warning if any forward ran eager."""
    runner = moge_graph.installed_runner(model)
    if runner is None:
        return
    stats = runner.take_stats()
    log = logger.warning if stats["eager_calls"] else logger.info
    log("MoGe-2 CUDA graph (%s): %d capture(s) in %.2f s, %d replay(s), %d eager call(s)", what, stats["captures"],
        stats["capture_seconds"], stats["replays"], stats["eager_calls"])


def make_intrinsics_worker(n_keyframes: int = 3, device: str = "cuda", model: Any = None,
                           moge_batch_size: int | None = None, moge_cuda_graph: bool = False,
                           moge_num_tokens: int | None = None, focal_scale: float | None = None) -> StageFn:
    """The `intrinsics` stage worker (see the module docstring); nothing loads until MoGe-2 is first needed.

    `moge_batch_size` is MoGe-2's frames per forward on the keyframes (`None`: `moge_model._BATCH_SIZE`, today's 2);
    a clip with `camera.json` never runs MoGe, so it cannot act there. `moge_cuda_graph` installs the graphed
    forward on the worker's model at its first MoGe call; False (the default) leaves the model untouched.
    `moge_num_tokens` is MoGe-2's base ViT tokens on the keyframes (`None`: not passed, MoGe's own 3600); like
    the batch size it cannot act on a clip with `camera.json`. `focal_scale` (opt-in; None = off, byte-identical)
    multiplies the ESTIMATED fx and fy (`pixel_intrinsics`) and is recorded in `intrinsics.npz`; a `camera.json`
    calibration is used exactly as stated.

    Raises:
        ValueError: On a `moge_batch_size` that is not an int >= 1 or None, a non-bool `moge_cuda_graph`, a
            `moge_num_tokens` that is not None or an int in `moge_model.NUM_TOKENS_RANGE`, or a `focal_scale` that
            `check_focal_scale` refuses, at construction.
    """
    focal_scale = check_focal_scale(focal_scale, "focal_scale")
    moge_batch_size = _check_moge_batch_size(moge_batch_size)
    cuda_graph = _check_moge_cuda_graph(moge_cuda_graph)
    moge_num_tokens = _check_moge_num_tokens(moge_num_tokens)
    ctx = WorkerContext(loader=lambda: moge_model._load_model(device), model=model)

    def intrinsics_worker(clip_id: str, artifact_dir: Path) -> None:
        ctx.budget()
        artifact_dir = Path(artifact_dir)
        camera = artifact_dir / CAMERA_JSON
        if camera.is_file():
            write_intrinsics(artifact_dir, read_camera_json(camera), "given")
            return
        decode_path = require_artifact(artifact_dir / DECODE_FILE, "decode")
        n, height, width = npz_array_shape(decode_path, "frames")[:3]      # read the keyframe rows only
        rows = keyframe_rows(n, n_keyframes)
        frames, _ = read_decode_rows(decode_path, rows)
        model = _moge(ctx, cuda_graph)
        normalized = moge_model.moge_intrinsics(torch.from_numpy(frames), device=device, model=model,
                                                batch_size=moge_batch_size, **_tokens_kwarg(moge_num_tokens))
        if cuda_graph:
            _log_graph_stats(model, f"intrinsics of {clip_id}")
        K = pixel_intrinsics(normalized.numpy(), width=width, height=height, focal_scale=focal_scale)
        write_intrinsics(artifact_dir, K, "moge2", focal_scale=focal_scale)
        logger.info("MoGe-2 intrinsics of %r from rows %s: fx %.1f fy %.1f (focal_scale %s)", clip_id, rows.tolist(),
                    K[0, 0], K[1, 1], focal_scale)

    intrinsics_worker.focal_scale = focal_scale  # type: ignore[attr-defined]  # reachability tests read it
    return intrinsics_worker


@dataclass(frozen=True)
class _ClipInputs:
    artifact_dir: Path
    frames: np.ndarray       # (N,H,W,3) uint8
    frame_idx: np.ndarray    # (N,) int64 source frame indices
    K: np.ndarray            # (3,3) float32 pixel intrinsics
    joints: Tensor           # (N,21*len(hand_sides),3) float32 camera-frame joints, in `hand_sides` order
    valid: Tensor            # (N,21*len(hand_sides)) bool
    hand_sides: tuple[str, ...] = DEFAULT_HAND_SIDES   # the sides actually used, after `HAND_SIDES_AUTO`
    rows: np.ndarray | None = None   # the decode rows `frames` holds under `row_stride` > 1; None = all of them
    # the per-side depth factors of `hand_sides` and their per-joint (21*len(hand_sides),) float64 expansion;
    # None when there is no correction (off, or every factor of the sides used is exactly 1.0)
    side_depth_scale: tuple[float, ...] | None = None
    depth_scale: Tensor | None = None

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def camera(self) -> tuple[int, int, bytes]:
        return int(self.frames.shape[1]), int(self.frames.shape[2]), self.K.tobytes()


def strided_rows(n: int, keyframe_index: int, row_stride: int) -> np.ndarray:
    """Rows 0, k, 2k, ... of `n`, plus the keyframe (registration and the masked storage need its pointmap)."""
    rows = set(range(0, n, row_stride))
    rows.add(int(keyframe_index))
    return np.array(sorted(rows), dtype=np.int64)


def _auto_hand_side(hands: Any, rows: np.ndarray) -> tuple[str, ...]:
    """The side to anchor on under `HAND_SIDES_AUTO`: right if it has any valid frame, else left.

    Falls back to `DEFAULT_HAND_SIDES` when neither side has one, so a clip with no usable hand at all still
    fails with the stage's own message rather than silently anchoring on nothing.
    """
    for side in ("right", "left"):
        if bool(np.asarray(hands[f"{side}_valid"], dtype=bool)[rows].any()):
            return (side,)
    return DEFAULT_HAND_SIDES


def _read_inputs(artifact_dir: Path, hand_sides: Sequence[str] | str,
                 hand_depth_scale: dict[str, float] | None = None) -> _ClipInputs:
    frames, frame_idx = read_decode(artifact_dir)
    K, _ = read_intrinsics(artifact_dir)
    hands_path = require_artifact(artifact_dir / HANDS_FILE, "hands")
    joints, valid = [], []
    with np.load(hands_path, allow_pickle=False) as hands:
        require_hands_camera(hands, K, int(frames.shape[2]), int(frames.shape[1]), str(hands_path))
        rows = rows_by_frame_index(hands["frame_idx"], frame_idx, str(hands_path))
        sides = _auto_hand_side(hands, rows) if hand_sides == HAND_SIDES_AUTO else tuple(hand_sides)
        for side in sides:
            side_joints = np.asarray(hands[f"{side}_joints"], dtype=np.float32)[rows]
            side_valid = np.asarray(hands[f"{side}_valid"], dtype=bool)[rows]
            joints.append(side_joints)
            valid.append(np.repeat(side_valid[:, None], side_joints.shape[1], axis=1))
    side_scale = None if hand_depth_scale is None else tuple(hand_depth_scale[side] for side in sides)
    depth_scale = None
    if side_scale is not None and any(f != 1.0 for f in side_scale):
        depth_scale = torch.from_numpy(np.concatenate([np.full(j.shape[1], f, np.float64)
                                                       for j, f in zip(joints, side_scale, strict=True)]))
    else:
        side_scale = None
    return _ClipInputs(artifact_dir, frames, frame_idx, K, torch.from_numpy(np.concatenate(joints, axis=1)),
                       torch.from_numpy(np.concatenate(valid, axis=1)), sides, side_depth_scale=side_scale,
                       depth_scale=depth_scale)


class GeometryWorker(BatchedWorker):
    """The `geometry` stage (see the module docstring); build it with `make_geometry_worker`."""

    def __init__(self, scale_fn: ScaleFn, max_frames_per_pass: int, device: str, model: Any,
                 hand_sides: Sequence[str] | str = DEFAULT_HAND_SIDES, pointmap_storage: str = "dense",
                 row_stride: int = 1, moge_batch_size: int | None = None, moge_cuda_graph: bool = False,
                 moge_overlap: bool = False, moge_num_tokens: int | None = None,
                 hand_depth_scale: Any = None, ego_motion: EgoMotionConfig | None = None) -> None:
        self.hand_depth_scale = check_hand_depth_scale(hand_depth_scale)
        if self.hand_depth_scale is not None and scale_fn is not hand_anchored_scale:
            raise ValueError("hand_depth_scale corrects hand_anchored_scale's anchor; a custom scale_fn would ignore it")
        self.moge_batch_size = _check_moge_batch_size(moge_batch_size)
        self.ego = ego_motion if ego_motion is not None and ego_motion.enabled else None
        if self.ego is not None and row_stride != 1:
            raise ValueError(f"geometry_stage.ego_motion needs row_stride 1 (every decode row), got {row_stride!r}")
        self.moge_cuda_graph = _check_moge_cuda_graph(moge_cuda_graph)
        self.moge_overlap = _check_moge_overlap(moge_overlap)
        self.moge_num_tokens = _check_moge_num_tokens(moge_num_tokens)
        if not isinstance(row_stride, int) or isinstance(row_stride, bool) or row_stride < 1:
            raise ValueError(f"row_stride must be an int >= 1, got {row_stride!r}")
        self.row_stride = row_stride
        if pointmap_storage not in POINTMAP_STORAGES:
            raise ValueError(f"pointmap_storage must be one of {POINTMAP_STORAGES}, got {pointmap_storage!r}")
        self.pointmap_storage = pointmap_storage
        if max_frames_per_pass < 1:
            raise ValueError(f"max_frames_per_pass must be >= 1, got {max_frames_per_pass}")
        sides: tuple[str, ...] | str
        if hand_sides == HAND_SIDES_AUTO:
            sides = HAND_SIDES_AUTO
        else:
            sides = tuple(hand_sides)
            if not sides or len(set(sides)) != len(sides) or any(side not in SIDES for side in sides):
                raise ValueError(f"hand_sides must be {HAND_SIDES_AUTO!r} or distinct entries of {SIDES}, "
                                 f"got {sides}")
        self.scale_fn, self.max_frames_per_pass, self.device, self.hand_sides = scale_fn, max_frames_per_pass, device, sides
        self._ctx = WorkerContext(loader=lambda: moge_model._load_model(device), model=model)

    def run_many(self, clips: Sequence[tuple[str, Path]]) -> None:
        """Pointmaps for every clip; clips sharing a camera share `run_moge` calls."""
        self._ctx.budget()
        by_camera: dict[tuple[int, int, bytes], list[_ClipInputs]] = {}
        for _, artifact_dir in clips:
            inputs = _read_inputs(Path(artifact_dir), self.hand_sides, self.hand_depth_scale)
            if self.row_stride > 1:
                inputs = self._strided(inputs)
            by_camera.setdefault(inputs.camera, []).append(inputs)
        for group in by_camera.values():
            self._run_group(group)

    def _strided(self, clip: _ClipInputs) -> _ClipInputs:
        """The clip restricted to `strided_rows`; the keyframe comes from `mask.npz`, pinned to the decode rows."""
        mask = load_npz(require_artifact(clip.artifact_dir / "mask.npz", "mask"))
        if not np.array_equal(np.asarray(mask["frame_idx"]), np.asarray(clip.frame_idx)):
            raise ValueError(f"{clip.artifact_dir}: mask.npz rows are not the decode rows; refusing to take its keyframe")
        rows = strided_rows(clip.n_frames, int(mask["keyframe_index"]), self.row_stride)
        return replace(clip, frames=clip.frames[rows], joints=clip.joints[rows], valid=clip.valid[rows], rows=rows)

    def _run_group(self, group: list[_ClipInputs]) -> None:
        """Streams one camera's rows, in clip order, through calls of at most `max_frames_per_pass` frames."""
        K = torch.from_numpy(group[0].K)
        buffers: list[np.ndarray | None] = [None] * len(group)
        segments: list[tuple[int, int, int]] = []    # (clip position, first row, end row) of the pending call
        room = self.max_frames_per_pass
        for position, clip in enumerate(group):
            row = 0
            while row < clip.n_frames:
                take = min(room, clip.n_frames - row)
                segments.append((position, row, row + take))
                row, room = row + take, room - take
                if room == 0:
                    self._run_pass(group, segments, K, buffers)
                    segments, room = [], self.max_frames_per_pass
        if segments:
            self._run_pass(group, segments, K, buffers)

    def _run_pass(self, group: list[_ClipInputs], segments: list[tuple[int, int, int]], K: Tensor,
                  buffers: list[np.ndarray | None]) -> None:
        """One `run_moge` call over `segments`; writes every clip whose last row it holds."""
        pieces = [group[position].frames[first:end] for position, first, end in segments]
        frames = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        if self._gathers(group, segments):
            self._run_gathered_pass([group[position] for position, _, _ in segments], frames, K)
            return
        model = self._model()
        raw = run_moge(torch.from_numpy(frames), device=self.device, model=model, intrinsics=K,
                       batch_size=self.moge_batch_size, overlap=self.moge_overlap,
                       **_tokens_kwarg(self.moge_num_tokens)).numpy()
        self._graph_stats(model, len(frames))
        offset = 0
        for position, first, end in segments:
            clip, rows = group[position], end - first
            part = raw[offset:offset + rows]
            offset += rows
            if first == 0 and end == clip.n_frames:
                self._write(clip, torch.from_numpy(part), K)          # the whole clip in one call: no copy
                continue
            buffer = buffers[position]
            if buffer is None:
                buffer = buffers[position] = np.empty((clip.n_frames, *raw.shape[1:]), dtype=np.float32)
            buffer[first:end] = part
            if end == clip.n_frames:
                buffers[position] = None
                self._write(clip, torch.from_numpy(buffer), K)

    def _model(self) -> Any:
        """The worker's MoGe-2 model, graphed when `moge_cuda_graph`."""
        return _moge(self._ctx, self.moge_cuda_graph)

    def _graph_stats(self, model: Any, n_frames: int) -> None:
        if self.moge_cuda_graph:
            _log_graph_stats(model, f"geometry pass of {n_frames} frames")

    def _gathers(self, group: list[_ClipInputs], segments: list[tuple[int, int, int]]) -> bool:
        """Take the gathered masked path for this call? Only with masked storage, the default scale (the one
        whose pixels are known: a custom `scale_fn` may read anything), and every clip whole in this call -- the
        production case, one S2 window per call (`DEFAULT_MAX_FRAMES_PER_PASS`). Otherwise the dense path runs."""
        return (self.pointmap_storage == "masked" and self.scale_fn is hand_anchored_scale
                and all(first == 0 and end == group[position].n_frames for position, first, end in segments))

    def _run_gathered_pass(self, clips: list[_ClipInputs], frames: np.ndarray, K: Tensor) -> None:
        """One `run_moge_batches` call that copies to the host only what the masked `geometry.npz` and the
        hand-anchored scale read -- each row's object pixels, the keyframe, the hand-joint pixels -- instead of the
        full-resolution pointmaps; then `_write`'s scale, rescale and payload on those values, bit-identically."""
        h, w = int(frames.shape[1]), int(frames.shape[2])
        gathers = []
        for clip in clips:
            mask = self._read_mask(clip)
            gathers.append(_Gather.for_clip(clip, K, h, w, mask, self._plan(clip, mask[0])))
        owners = [(gather, row) for gather in gathers for row in range(gather.clip.n_frames)]   # call row -> clip row

        def sink(start: int, points: Tensor) -> None:
            rows = owners[start:start + int(points.shape[0])]
            masks = np.stack([gather.masks[gather.decode_row(row)] for gather, row in rows])
            picked = points[torch.from_numpy(masks).to(points.device)].cpu().numpy()   # row-major, frame-major
            at = 0
            for b, (gather, row) in enumerate(rows):
                at = gather.take(row, points[b], picked, at)

        model = self._model()
        run_moge_batches(torch.from_numpy(frames), sink, device=self.device, model=model, intrinsics=K,
                         batch_size=self.moge_batch_size, overlap=self.moge_overlap,
                         **_tokens_kwarg(self.moge_num_tokens))
        self._graph_stats(model, len(frames))
        for gather in gathers:
            self._write_gathered(gather)

    def _write_gathered(self, gather: _Gather) -> None:
        """`_write` (masked storage, `hand_anchored_scale`) on a clip's gathered values: the same arrays."""
        clip, n = gather.clip, gather.clip.n_frames
        frame, _, _, z = gather.anchors
        scale, frame_scale = self._clip_scale(clip, hand_anchored_scale_from_samples(n, frame, z,
                                                                                     torch.from_numpy(gather.samples)))
        # `raw.mul_(scale)` on the rows' values: the same float32 multiply on the same values, so the same bits.
        points = torch.from_numpy(gather.points).mul_(scale).numpy()
        keyframe = torch.from_numpy(gather.keyframe).mul_(scale).numpy()
        if clip.rows is not None:
            # `row_stride`: as `_write`, rows MoGe-2 did not see are NaN -- set after the rescale, as there.
            for row in np.setdiff1d(np.arange(len(clip.frame_idx)), clip.rows):
                points[gather.offsets[row]:gather.offsets[row + 1]] = np.nan
            full_scale = torch.full((len(clip.frame_idx),), math.nan, dtype=torch.float32)
            full_scale[torch.from_numpy(clip.rows)] = frame_scale
            frame_scale, n = full_scale, len(clip.frame_idx)
        self._save(clip, masked_payload(points, keyframe, gather.masks, gather.keyframe_index), scale, frame_scale, n)
        if gather.ego is not None:
            # the same float32 multiply `points` got, so the dense path's samples are these bits
            plan, plan_seconds = gather.ego
            finish_clip(clip.artifact_dir, plan, torch.from_numpy(gather.ego_samples).mul_(scale).numpy(),
                        clip.frame_idx, self.ego, plan_seconds)

    def _write(self, clip: _ClipInputs, raw: Tensor, K: Tensor) -> None:
        n = int(raw.shape[0])
        if clip.depth_scale is None:
            scaled = self.scale_fn(raw, clip.joints, K, clip.valid)
        else:   # construction guarantees scale_fn is hand_anchored_scale
            scaled = hand_anchored_scale(raw, clip.joints, K, clip.valid, depth_scale=clip.depth_scale)
        scale, frame_scale = self._clip_scale(clip, scaled)
        # In place: `apply_metric_scale(raw, full(n, scale))` without a second (N,H,W,3) copy; `raw` is this worker's.
        pointmaps = raw.mul_(scale)
        dense = pointmaps.numpy().astype(np.float32, copy=False)
        if clip.rows is not None:
            # `row_stride`: rows MoGe-2 did not see carry NaN, the readers' own "no observation"; `frame_idx` stays whole.
            full = np.full((len(clip.frame_idx), *dense.shape[1:]), np.nan, dtype=np.float32)
            full[clip.rows] = dense
            full_scale = torch.full((len(clip.frame_idx),), math.nan, dtype=torch.float32)
            full_scale[torch.from_numpy(clip.rows)] = frame_scale
            dense, frame_scale, n = full, full_scale, len(clip.frame_idx)
        if self.pointmap_storage == "masked":
            # only object pixels (and the keyframe) reach the solver; writing and reading the rest cost
            # 11.9 s of a 58.2 s window. The mask is read here, pinned by digest, and must be the decode rows.
            masks, keyframe_index = self._read_mask(clip)
            stored = pack_masked(dense, masks, keyframe_index)
        else:
            stored = {"pointmaps": dense}
        self._save(clip, stored, scale, frame_scale, n)
        ego = self._plan(clip)
        if ego is not None:
            plan, plan_seconds = ego
            finish_clip(clip.artifact_dir, plan, dense[plan.frame, plan.v, plan.u], clip.frame_idx, self.ego,
                        plan_seconds)

    def _plan(self, clip: _ClipInputs, masks: np.ndarray | None = None) -> tuple[EgoPlan, float] | None:
        """The clip's ego-motion tracker plan (None when `ego_motion` is off)."""
        if self.ego is None:
            return None
        return plan_for_clip(clip.artifact_dir, clip.frames, clip.frame_idx, clip.K, self.ego, masks)

    def _clip_scale(self, clip: _ClipInputs, scaled: Tensor | float) -> tuple[float, Tensor]:
        """`scale_fn`'s output -> (the clip's scale, per-frame scale (n,) float32), checked finite and positive."""
        n = clip.n_frames
        frame_scale = torch.as_tensor(scaled, dtype=torch.float32)
        frame_scale = frame_scale.detach().cpu().reshape(-1)
        if frame_scale.numel() == 1:
            scale = float(frame_scale[0])
            frame_scale = torch.full((n,), scale, dtype=torch.float32)
        elif frame_scale.numel() == n:
            # All-NaN pools to NaN here, so the check below names `hand_sides` (clip_median_scale would not).
            finite = frame_scale[torch.isfinite(frame_scale)]
            scale = float(clip_median_scale(frame_scale)[0]) if finite.numel() else math.nan
        else:
            raise ValueError(f"scale_fn must return one scale or one per frame ({n}), got {frame_scale.numel()}")
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(f"{clip.artifact_dir}: the metric scale must be finite and positive, got {scale} "
                             f"(hand_sides={clip.hand_sides}, requested {self.hand_sides}; a NaN means no "
                             f"usable joint on those hands)")
        return scale, frame_scale

    @staticmethod
    def _read_mask(clip: _ClipInputs) -> tuple[np.ndarray, int]:
        """`mask.npz`'s (masks (N,H,W) bool, keyframe decode row), refused unless its rows are the decode rows."""
        mask = load_npz(require_artifact(clip.artifact_dir / "mask.npz", "mask"))
        if not np.array_equal(np.asarray(mask["frame_idx"]), np.asarray(clip.frame_idx)):
            raise ValueError(f"{clip.artifact_dir}: mask.npz rows are not the geometry rows; refusing to cut "
                             "the pointmaps with another clip's mask")
        return np.asarray(mask["masks"], dtype=bool), int(mask["keyframe_index"])

    def _save(self, clip: _ClipInputs, stored: dict[str, np.ndarray], scale: float, frame_scale: Tensor,
              n: int) -> None:
        # hand-depth-scale provenance, only when a correction was applied (so an uncorrected file keeps its exact bytes)
        extra = ({} if clip.side_depth_scale is None
                 else {"hand_depth_scale": np.array(clip.side_depth_scale, dtype=np.float64)})
        save_npz(clip.artifact_dir / GEOMETRY_FILE, **stored,
                 frame_idx=clip.frame_idx, scale=np.array(scale, dtype=np.float32),
                 frame_scale=frame_scale.numpy().astype(np.float32),
                 hand_sides=np.array(clip.hand_sides), **extra)      # provenance: which hand actually set the scale
        logger.info("geometry for %s: %d frames, metric scale %.4f", clip.artifact_dir, n, scale)


@dataclass(frozen=True)
class _Gather:
    """The host arrays one clip's gathered masked pass fills, batch by batch, from the device pointmaps.

    `points` holds each decode row's object pixels at `offsets[row]:offsets[row + 1]` (`pointmaps[masks]` order),
    `keyframe` the keyframe row, `samples` the raw points at `hand_anchor_pixels` (`anchors`), all unscaled."""
    clip: _ClipInputs
    masks: np.ndarray             # (N,H,W) bool, `mask.npz` over every decode row
    keyframe_index: int           # decode row
    offsets: np.ndarray           # (N+1,) int64
    anchors: tuple[Tensor, Tensor, Tensor, Tensor]   # (frame, v, u, z) over the clip's (possibly strided) rows
    points: np.ndarray            # (P,3) float32
    keyframe: np.ndarray          # (H,W,3) float32
    samples: np.ndarray           # (A,3) float32
    ego: tuple[EgoPlan, float] | None = None       # the ego-motion plan and its seconds, when enabled
    ego_samples: np.ndarray | None = None          # (S,3) float32 raw pointmap values at the plan's pixels
    ego_rows: dict[int, np.ndarray] | None = None  # decode row -> indices into the plan's flat samples

    @classmethod
    def for_clip(cls, clip: _ClipInputs, K: Tensor, h: int, w: int, mask: tuple[np.ndarray, int],
                 ego: tuple[EgoPlan, float] | None = None) -> _Gather:
        masks, keyframe_index = mask
        check_masked_shapes((len(clip.frame_idx), h, w, 3), masks.shape, keyframe_index)
        counts = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.int64)
        offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(counts)])
        anchors = hand_anchor_pixels(clip.joints, K, clip.valid, (clip.n_frames, h, w), depth_scale=clip.depth_scale)
        ego_samples = ego_rows = None
        if ego is not None:
            plan = ego[0]
            ego_samples = np.full((len(plan.frame), 3), np.nan, np.float32)
            ego_rows = {int(r): np.nonzero(plan.frame == r)[0] for r in np.unique(plan.frame)}
        return cls(clip, masks, keyframe_index, offsets, anchors, np.empty((int(offsets[-1]), 3), np.float32),
                   np.empty((h, w, 3), np.float32), np.empty((int(anchors[0].numel()), 3), np.float32),
                   ego, ego_samples, ego_rows)

    def decode_row(self, row: int) -> int:
        """The decode row of the clip's `row`-th MoGe row (they differ under `row_stride`)."""
        return row if self.clip.rows is None else int(self.clip.rows[row])

    def take(self, row: int, points: Tensor, picked: np.ndarray, at: int) -> int:
        """Stores row `row`'s values from its device `points` (H,W,3) and its object pixels `picked[at:]`;
        returns the end of those pixels in `picked`."""
        decode = self.decode_row(row)
        first, end = int(self.offsets[decode]), int(self.offsets[decode + 1])
        self.points[first:end] = picked[at:at + end - first]
        if decode == self.keyframe_index:
            self.keyframe[...] = points.cpu().numpy()
        frame, v, u, _ = self.anchors
        hits = torch.nonzero(frame == row).reshape(-1)
        if hits.numel():
            index = torch.stack([v[hits], u[hits]]).to(points.device)
            self.samples[hits.numpy()] = points[index[0], index[1]].cpu().numpy()
        if self.ego_rows is not None and self.ego_samples is not None and self.ego is not None:
            mine = self.ego_rows.get(decode)
            if mine is not None and mine.size:
                plan = self.ego[0]
                ev = torch.from_numpy(plan.v[mine]).to(points.device)
                eu = torch.from_numpy(plan.u[mine]).to(points.device)
                self.ego_samples[mine] = points[ev, eu].cpu().numpy()
        return at + end - first


# One S2 window (100 decode rows) per call. At 64 every window straddled two calls and was reassembled through a
# host buffer: 2.1 s/window of copying (2.17 -> 0.075 s at 100 rows, 4 synthetic 1080p windows), bit-identical outputs.
# Keep it EVEN: MoGe runs `_BATCH_SIZE` = 2 frames at a time from each call's start, so an even pass size pairs the same
# rows as any other even size, whatever the clip lengths. (At `moge_batch_size` 8 a 100-row pass is 8x12 + 4, so
# which rows share a forward depends on where passes start; deterministic, but not the same grouping as 64-row passes.)
DEFAULT_MAX_FRAMES_PER_PASS = 100


def make_geometry_worker(scale_fn: ScaleFn = hand_anchored_scale,
                         max_frames_per_pass: int = DEFAULT_MAX_FRAMES_PER_PASS,
                         device: str = "cuda", model: Any = None,
                         hand_sides: Sequence[str] | str = DEFAULT_HAND_SIDES,
                         pointmap_storage: str = "dense", row_stride: int = 1,
                         moge_batch_size: int | None = None, moge_cuda_graph: bool = False,
                         moge_overlap: bool = False, moge_num_tokens: int | None = None,
                         hand_depth_scale: Any = None,
                         ego_motion: EgoMotionConfig | None = None) -> GeometryWorker:
    """The `geometry` stage worker; nothing loads or allocates until its first call.

    Args:
        scale_fn: Metric-scale function, `hand_anchored_scale` by default (R3: a calibrated scale fusion plugs in
            here).
        max_frames_per_pass: Most frames in one `run_moge` call; a call can span clips sharing a camera, and a
            longer clip is split across calls.
        device: Torch device MoGe-2 loads on.
        model: A `MoGeModel`-compatible model to use instead of loading `Ruicheng/moge-2-vitl`.
        hand_sides: Hands whose joints feed `scale_fn`: distinct entries of `SIDES`, or `HAND_SIDES_AUTO`.
            The default, the right hand alone, is the measured protocol, under which the 5-21% hand-anchored
            scale bias was measured; `SIDES` pools both hands. `HAND_SIDES_AUTO` keeps the right hand wherever
            it is usable and falls back to the left only when it is not, which recovers the 39.5% of DexYCB
            clips this stage currently drops for having the wrong hand while leaving every other clip
            bit-identical.

            **The default stays `("right",)` on purpose.** The fallback is opt-in until a study
            says whether a left hand carries the same scale bias as a right one: switching it on by default
            would raise yield and silently change the POPULATION behind every quality number, which is the one
            way to make the comparison to DO AS I DO meaningless. `geometry.npz` records the side actually used
            (`hand_sides`) so that study can stratify.

        pointmap_storage: `dense` (the default, every existing run) writes the full (N,H,W,3) pointmaps; `masked`
            writes only each frame's object pixels plus the keyframe (`fastdo.engine.geometry_io`), which the
            solve reads back identically. Never `masked` with driftqc enabled: driftqc reads tracked pixels that
            can leave the mask, and refuses a masked artifact.

        row_stride: Run MoGe-2 on decode rows 0, k, 2k, ... and the keyframe (from `mask.npz`) only, writing NaN
            rows elsewhere. 1 (the default) is every row, byte-identical to before. Measured:
            stride 2 is non-inferior on 71 labelled clips, stride 4 is not. The metric scale is then pooled over the
            computed rows only. Rows without pointmaps get `confidence == 0` from the solver, which retargeting masks
            -- adopting a stride needs row support redefined from masks and tracks first.

        moge_batch_size: MoGe-2 frames per forward pass. `None` (the default) is `moge_model._BATCH_SIZE`, 2:
            byte-identical to before. 8 saves ~0.8 s per 1080p window but changes the pointmaps in the 4th-7th
            significant digit (fp16 kernel choice), so it waits on the 71-clip DexYCB gate.

        moge_cuda_graph: Install `graphed_moge_forward` on the worker's model at its first MoGe call (the forward
            captured per input signature and replayed; the same kernels as eager). False (the default) leaves the model
            untouched: byte-identical to before. Opt-in until its GPU bit-identity gate and an Ego4D pair comparison
            pass.

        moge_overlap: Run each MoGe call's batches as a 2-stage pipeline (`run_moge_batches(overlap=True)`):
            batch k's CPU shift solve and host copies overlap batch k+1's forward, through a verbatim transcription
            of `MoGeModel.infer` (the loaded model must be a `MoGeModel`; it refuses other models and a moge whose
            source moved). False (the default) is the serial path: byte-identical to before. Opt-in until
            its GPU bit-identity gate and an Ego4D comparison pass. The intrinsics worker has no such knob.

        moge_num_tokens: MoGe-2's base ViT tokens (`infer(num_tokens=...)`) on both the serial and the overlapped
            path. `None` (the default) is not passed to `infer`: its own resolution_level 9 = 3600 tokens,
            byte-identical to before. An int in `moge_model.NUM_TOKENS_RANGE` ([1200, 3600]) is faster (2500 is
            -26% MoGe time at 1440x1920) but moves the depth, so it waits on the pose gate
            and an Ego4D gauge. The intrinsics worker takes the same knob.

        hand_depth_scale: None (the default: no correction, byte-identical to before) or `{left: f_L, right: f_R}`, each
            in `HAND_DEPTH_SCALE_RANGE`. The anchor's joints of side s are rescaled about the camera centre by f_s,
            which multiplies the clip's metric scale by f_s (module docstring). f_s = 1 / (1 + r_s) cancels a relative
            hand-depth bias r_s. Factors of exactly 1.0 on the sides a clip uses change nothing. Opt-in. The contact
            stage is NOT corrected. Requires the default `scale_fn`.

        ego_motion: `geometry_stage.ego_motion`; when enabled, also write `ego_motion.npz`, the window's camera
            trajectory from its full-frame pointmaps (module docstring). None or disabled (the default) computes
            nothing; `geometry.npz` is byte-identical either way. Requires `row_stride` 1.

    Raises:
        ValueError: On `max_frames_per_pass < 1`, bad `hand_sides`, an unknown `pointmap_storage`, a `row_stride`
            or `moge_batch_size` that is not an int >= 1, a non-bool `moge_cuda_graph` or `moge_overlap`, or a
            `moge_num_tokens` that is not None or an int in [1200, 3600], or a bad `hand_depth_scale` (or one with a
            custom `scale_fn`), at construction.
    """
    return GeometryWorker(scale_fn, max_frames_per_pass, device, model, hand_sides, pointmap_storage, row_stride,
                          moge_batch_size, moge_cuda_graph, moge_overlap, moge_num_tokens, hand_depth_scale,
                          ego_motion)
