"""Frame stage workers: `decode` writes decode.npz; `encoder` writes DINOv2 tokens, frame-batched across clips.

Artifacts are uncompressed `np.savez` archives written through a tmp file and `os.replace`, loadable without pickle:

- `decode.npz`: `frames` (N,H,W,3) uint8 RGB, `frame_idx` (N,) int64 strictly increasing source frame indices,
  `fps` 0-d float64 = `probe_fps(source) / stride`, and `stride` 0-d int64 (the qc stage's timestamps need it even
  for a one-row clip). The default backend is ffmpeg: decord misreads
  edit-listed MP4s such as the whisking demo (`fastdo.io.video`), so `"decord"` stays opt-in.
- `encoder.npz`: `tokens` (N, 2*hidden) float32, one `encode_crops` descriptor (CLS + mean patch token, bf16
  forward) per decode row, `frame_idx` (N,) int64, and `crop`, a 0-d unicode array naming the crop. The default
  crop, `"center"`, is the frame's centred square resized to `crop_size` (the design's DINOv2 ViT-L/14 at 336).
  `"letterbox"` resizes the whole frame so its long side is `crop_size` and pads the rest with the
  ImageNet mean, so the frame edges (where 13 of the 43 100DOH train clips hold most of the hand box) stay in view; the
  interaction gate records the crop it was calibrated on and refuses other crops. The design's hand-object ROI crops
  need boxes that no earlier stage produces yet.

The encoder is frame-batched: `EncodeWorker.run_many` concatenates the crops of several clips into shared forward
passes, grouped by `dynamic_batches` so one pass holds at most `max_batch` crops and at most `gpu_budget_gb` of
estimated activation memory (`crop_cost_gb` per crop, on top of the weights). Called as a `StageFn`, it is the
one-clip case. The model loads once per worker, on first use, after the CUDA cap.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.encoder import dinov2
from fastdo.engine.batching import dynamic_batches
from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import (
    DECODE_FILE,
    BatchedWorker,
    WorkerContext,
    read_decode,
    save_npz,
)
from fastdo.io.video import decode_frames, decode_window, probe_fps

__all__ = ["CROP_MODES", "ENCODER_FILE", "LETTERBOX_FILL", "EncodeWorker", "center_crops", "letterbox_crops",
           "make_decode_worker", "make_encode_worker", "timebase_flags", "timebase_mismatch"]

logger = logging.getLogger(__name__)

ENCODER_FILE = "encoder.npz"
CROP_MODES: tuple[str, ...] = ("center", "letterbox")
LETTERBOX_FILL: tuple[int, int, int] = (round(255 * dinov2.IMAGENET_MEAN[0]), round(255 * dinov2.IMAGENET_MEAN[1]),
                                        round(255 * dinov2.IMAGENET_MEAN[2]))   # (124, 116, 104): normalizes to ~0
_RESIZE_CHUNK = 16  # frames per resize call, bounding the float copy of a 720p clip


def timebase_mismatch(frame_idx: Sequence[int], pts: Sequence[float], fps: float,
                      start_time: float = 0.0) -> bool:
    """Does any decoded row sit further than one frame interval from where its index says it should?

    The window protocol, verbatim: "If any row has |pts - n / fps_v| > 1 / fps_v, the window is flagged
    `timebase_mismatch`." This is the guard for the off-by-one-frame decode defect -- a window decoded one frame off
    still writes the `frame_idx` the audit expects, so only a MEASURED time can contradict it. Worked
    through on the four Ego4D videos that start at 21.029 ms: a correct decode gives 0.021 s against the
    0.0333 s bound and passes; a one-frame shift gives 0.0543 s and is flagged.

    `start_time` is offered but defaults to 0.0, matching the protocol's formula exactly. Folding the
    container's start time in would make the test tighter, and would also change a protocol
    quantity -- so it is a parameter a caller must ask for, not a silent improvement.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive to place a row in time, got {fps}")
    tol = 1.0 / fps
    return any(abs(float(t) - (start_time + int(n) / fps)) > tol for n, t in zip(frame_idx, pts))


def timebase_flags(artifact_root: Path, clip_ids: Sequence[str]) -> dict[str, int]:
    """`{"flagged", "clean", "unrecorded"}` over `clip_ids`, read from each clip's `decode.npz`.

    The window protocol says a flagged window "stays in every analysis", which only means anything if the
    flag REACHES the analysis. Recording it and never reading it would leave the guard
    present on disk and absent from every report -- the failure this project has paid for repeatedly.

    `unrecorded` counts clips whose decode predates the flag or was a whole-file decode (the check is
    defined for windows), and is reported rather than folded into `clean`: a missing measurement is not a
    passing one.
    """
    counts = {"flagged": 0, "clean": 0, "unrecorded": 0}
    for clip_id in clip_ids:
        path = Path(artifact_root) / clip_id / DECODE_FILE
        if not path.is_file():
            counts["unrecorded"] += 1
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if "timebase_mismatch" not in data.files:
                    counts["unrecorded"] += 1
                else:
                    counts["flagged" if int(data["timebase_mismatch"]) else "clean"] += 1
        except (OSError, ValueError):
            counts["unrecorded"] += 1
    return counts


def make_decode_worker(sources: Mapping[str, Path | tuple[Path, int, int]], stride: int = 3,
                       backend: str = "ffmpeg", local_probe: bool = False,
                       packet_table: bool = False,
                       scale_height: int | None = None) -> StageFn:
    """The `decode` stage: decodes each clip's source video at `stride` into `decode.npz`.

    Args:
        sources: `clip_id -> video path`, or `clip_id -> (video path, start, stop)` for a WINDOW of that
            video (source frame indices, `stop` exclusive). One long source file therefore yields many
            windows, each decoding only its own range; the whole file is decoded only
            for a bare path. Built from the same lists the caller passed to `ClipManifest.create`.
        stride: Keep every `stride`-th decoded frame, from frame 0 (whole file) or from `start` (window).
        local_probe: Address each window with a window-local ffprobe instead of the whole-file frame
            table, and fall back to the table whenever that probe cannot cover the window. It changes no
            decoded frame; it changes only what ingest costs, and NOT always for the better. Amortised
            over EVERY eligible window of a video the whole-file table wins (2.75 s/window against
            4.24 s); the local probe wins when a video contributes FEW windows (10.3x at 4 per video,
            41.4x at one), because it removes the dependence on the video's length. Break-even is about
            34 windows per video. Leave it off for an exhaustive pass; turn it on for a sampled one, and
            only for a source whose frame rate has been compared (Ego4D: 20 of 20 windows exact).
        packet_table: On a frame-table cache MISS, address the window from the container's packet
            table when the trim screen passes (`fastdo.io.video.packet_table_is_frame_table`). The table
            itself is 144x cheaper (13.55 s against 0.09 s, measured); full windowed ingest is 1.6x-3.0x
            faster depending on video length, because the per-window decode (1.14 s) is paid either way.
            Opt-in: the screen rests on two genuine negatives.
        scale_height: Downscale each decoded row to this height, the window protocol's
            `scale=-2:720:flags=area` (the protocol fixes it at 720). `None` (the default) decodes at source size.
            WINDOWS ONLY -- a whole-file decode ignores it, because `decode_frames` has no scaled path and
            silently decoding one route scaled and the other not would be worse than not offering it.
            The saving is DISK, not time: 2.25x fewer bytes on 1920x1080, and about 7% off the decode.
            A source already at or below the target is left alone rather than enlarged.
        backend: `fastdo.io.video.decode_frames` backend, `"ffmpeg"` by default. A window always uses
            ffmpeg, whose seek-then-select path is exact on edit-listed and variable-rate files.

    Raises:
        ValueError: If `stride < 1`, at construction.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if scale_height is not None and scale_height < 1:
        raise ValueError(f"scale_height must be >= 1 or None, got {scale_height}")
    table: dict[str, tuple[Path, int | None, int | None]] = {}
    for clip_id, entry in sources.items():
        if isinstance(entry, tuple):
            path, start, stop = entry
            table[str(clip_id)] = (Path(path), int(start), int(stop))
        else:
            table[str(clip_id)] = (Path(entry), None, None)
    ctx = WorkerContext()

    def decode_worker(clip_id: str, artifact_dir: Path) -> None:
        if clip_id not in table:
            raise KeyError(f"no source video registered for clip {clip_id!r}")
        ctx.budget()
        source, start, stop = table[clip_id]
        pts: list[float] | None = None
        if start is None:
            frames, idx = decode_frames(source, stride=stride, backend=backend)
        else:
            frames, idx, pts = decode_window(source, start, stop, stride=stride, local_probe=local_probe,
                                             packet_table=packet_table, scale_height=scale_height,
                                             return_times=True)
        frame_idx = np.asarray(idx, dtype=np.int64)
        if frame_idx.size > 1 and not bool((np.diff(frame_idx) > 0).all()):
            raise ValueError(f"{source}: decoded frame indices are not strictly increasing")
        source_fps = probe_fps(source)
        fps = source_fps / stride
        extra: dict[str, np.ndarray] = {}
        if pts is not None:
            # The window protocol: "The window decoder records each row's best-effort pts." It is recorded
            # rather than derived because the derived value is what the timebase check exists to VERIFY --
            # `workers_qc.decode_timestamps` computes `frame_idx / (fps * stride)`, so checking that
            # against itself would pass on a window decoded one frame off.
            extra["pts"] = np.asarray(pts, dtype=np.float64)
            extra["timebase_mismatch"] = np.array(
                int(timebase_mismatch(frame_idx.tolist(), pts, source_fps)), dtype=np.int64)
        save_npz(Path(artifact_dir) / DECODE_FILE, frames=np.ascontiguousarray(frames.numpy(), dtype=np.uint8),
                 frame_idx=frame_idx, fps=np.array(fps, dtype=np.float64), stride=np.array(stride, dtype=np.int64),
                 **extra)
        logger.info("decoded %d frames of %r (stride %d, %.4f fps%s)", frame_idx.size, clip_id, stride, fps,
                    "" if start is None else f", window [{start}, {stop})")

    return decode_worker


def _check_frames(frames: Tensor, size: int) -> tuple[int, int, int]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {tuple(frames.shape)}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])


def center_crops(frames: Tensor, size: int) -> Tensor:
    """`(N,3,size,size)` uint8: each `(N,H,W,3)` uint8 frame's centred square, resized (bilinear, antialiased).

    Raises:
        ValueError: If `frames` is not `(N,H,W,3)` or `size < 1`.
    """
    n, h, w = _check_frames(frames, size)
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    out = torch.empty((n, 3, size, size), dtype=torch.uint8)
    for start in range(0, n, _RESIZE_CHUNK):
        chunk = frames[start:start + _RESIZE_CHUNK, top:top + side, left:left + side].permute(0, 3, 1, 2).float()
        if side != size:
            chunk = F.interpolate(chunk, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        out[start:start + _RESIZE_CHUNK] = chunk.round().clamp(0, 255).to(torch.uint8)
    return out


def letterbox_crops(frames: Tensor, size: int) -> Tensor:
    """`(N,3,size,size)` uint8: each whole frame resized (bilinear, antialiased) so its long side is `size`, centred
    at `top = (size - h') // 2`, `left = (size - w') // 2`, on a `LETTERBOX_FILL` background.

    Raises:
        ValueError: If `frames` is not `(N,H,W,3)` or `size < 1`.
    """
    n, h, w = _check_frames(frames, size)
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out = torch.tensor(LETTERBOX_FILL, dtype=torch.uint8).view(1, 3, 1, 1).repeat(n, 1, size, size)
    for start in range(0, n, _RESIZE_CHUNK):
        chunk = frames[start:start + _RESIZE_CHUNK].permute(0, 3, 1, 2).float()
        if (new_h, new_w) != (h, w):
            chunk = F.interpolate(chunk, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        out[start:start + _RESIZE_CHUNK, :, top:top + new_h, left:left + new_w] = chunk.round().clamp(0, 255).to(
            torch.uint8)
    return out


class EncodeWorker(BatchedWorker):
    """The `encoder` stage (see the module docstring); build it with `make_encode_worker`."""

    def __init__(self, crop_size: int, max_batch: int, gpu_budget_gb: float, crop_cost_gb: float, device: str,
                 model: Any, crop: str = "center") -> None:
        if crop_size < 1 or max_batch < 1:
            raise ValueError(f"crop_size and max_batch must be >= 1, got {crop_size} and {max_batch}")
        if crop not in CROP_MODES:
            raise ValueError(f"crop must be one of {CROP_MODES}, got {crop!r}")
        self.crop_size, self.max_batch, self.crop = crop_size, max_batch, crop
        self.gpu_budget_gb, self.crop_cost_gb = gpu_budget_gb, crop_cost_gb
        self._ctx = WorkerContext(loader=lambda: dinov2.load_dinov2(device), model=model)

    def run_many(self, clips: Sequence[tuple[str, Path]]) -> None:
        """Encodes every decode row of `clips`, sharing forward passes across clip boundaries."""
        self._ctx.budget()
        crop_fn = letterbox_crops if self.crop == "letterbox" else center_crops
        dirs = [Path(artifact_dir) for _, artifact_dir in clips]
        crops: list[Tensor] = []
        frame_idx: list[np.ndarray] = []
        for artifact_dir in dirs:
            frames, idx = read_decode(artifact_dir)
            crops.append(crop_fn(torch.from_numpy(frames), self.crop_size))
            frame_idx.append(idx)
        items = [(clip, row) for clip, block in enumerate(crops) for row in range(int(block.shape[0]))]
        tokens: list[list[Tensor]] = [[] for _ in dirs]
        model = self._ctx.model() if items else None
        for batch in dynamic_batches(items, self.max_batch, self.gpu_budget_gb, lambda _: self.crop_cost_gb):
            stack = torch.stack([crops[clip][row] for clip, row in batch])
            pooled = dinov2.encode_crops(model, stack, batch_size=len(batch))
            for (clip, _), descriptor in zip(batch, pooled, strict=True):
                tokens[clip].append(descriptor)
        for artifact_dir, rows, idx in zip(dirs, tokens, frame_idx, strict=True):
            save_npz(artifact_dir / ENCODER_FILE, tokens=torch.stack(rows).numpy().astype(np.float32),
                     frame_idx=idx, crop=np.array(self.crop))


def make_encode_worker(crop_size: int = 336, max_batch: int = 32, gpu_budget_gb: float = 4.0,
                       crop_cost_gb: float = 0.05, device: str = "cuda", model: Any = None,
                       crop: str = "center") -> EncodeWorker:
    """The `encoder` stage worker; nothing loads or allocates until its first call.

    Args:
        crop_size: Square crop side fed to DINOv2 (a multiple of its 14 px patch).
        max_batch: Most crops per forward pass.
        gpu_budget_gb: Activation memory one forward pass may use, in GiB, on top of the weights.
        crop_cost_gb: Estimated activation GiB per crop (DINOv2-L at 336 px under bf16 is well below 0.05).
        device: Torch device the model loads on.
        model: A `Dinov2Model`-compatible model to use instead of loading `facebook/dinov2-large`.
        crop: `"center"` (the centred square, the default) or `"letterbox"` (module docstring).

    Raises:
        ValueError: If `crop` is not in `CROP_MODES`.
    """
    return EncodeWorker(crop_size, max_batch, gpu_budget_gb, crop_cost_gb, device, model, crop)
