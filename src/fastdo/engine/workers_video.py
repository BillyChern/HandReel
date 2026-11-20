"""Video stage workers: `mask` (SAM 2.1 propagation) and `tracks` (CoTracker3 online), clip-batched.

Both wrap single-clip, stateful video predictors: SAM 2 keeps a per-video inference state with its memory bank, and
CoTracker3 online keeps its queries and window position on the model object. Their tensors are therefore never
concatenated across clips, unlike the `encoder` and `geometry` stages. Dynamic batching for them is clip-level:
`fastdo.engine.batching.run_clips_concurrently(lambda: make_mask_worker(...), clips, max_concurrent,
gpu_budget_gb, cost_fn)` runs as many clips' predictor instances at once as the budget allows, one worker (and one
loaded model) per concurrent slot. One worker instance must not run two clips at the same time.

Inputs: `decode.npz` and `mask_prompt.npz`, which holds `keyframe_index` (0-d int64, a decode row) and `mask`
((H,W) bool, the object on that row). No stage writes the prompt yet (the design's open-vocabulary detector on the
keyframe is later work), so the caller writes it with `write_mask_prompt`.

Artifacts (uncompressed, tmp file then `os.replace`):
- `mask.npz`: `masks` (N,H,W) bool at native resolution, `frame_idx` (N,) int64, `keyframe_index` 0-d int64.
- `tracks.npz`: `tracks` (N,J,2) float32 pixel (x,y), `vis` (N,J) bool, `query_uv` (J,2) float32 sampled inside the
  prompt mask with a seeded generator, `frame_idx` (N,) int64, `keyframe_index` 0-d int64.

`make_mask_worker(variant=...)` takes a `fastdo.shape.sam2_variants` variant string. It defaults to `"large"` at
native resolution until the recorded sweep (`run/pipeline/bench_sam2_variants.py`) selects the
default with `choose_sam2_variant`. `make_mask_worker(mask_compile=...)` is `bench.mask_compile`
(`fastdo.shape.sam2_compile`): `none` (default) is the eager predictor; the vos modes load SAM 2.1's compiled VOS
predictor (not bit-exact, so gated on pose accuracy).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastdo.engine.orchestrator import StageFn
from fastdo.engine.worker_support import WorkerContext, read_decode, save_npz
from fastdo.shape import masks_sam2
from fastdo.shape.sam2_compile import check_mask_compile
from fastdo.shape.sam2_variants import DEFAULT_SAM2_VARIANT, parse_sam2_variant, propagate_masks_at
from fastdo.tracks import cotracker
from fastdo.tracks.cotracker import sample_query_points_in_mask, track_points_cotracker

__all__ = [
    "MASK_FILE",
    "PROMPT_FILE",
    "TRACKS_FILE",
    "make_mask_worker",
    "make_tracks_worker",
    "read_mask_prompt",
    "write_mask_prompt",
]

logger = logging.getLogger(__name__)

PROMPT_FILE = "mask_prompt.npz"
MASK_FILE = "mask.npz"
TRACKS_FILE = "tracks.npz"


def write_mask_prompt(artifact_dir: Path, keyframe_index: int, mask: np.ndarray) -> None:
    """Writes `<artifact_dir>/mask_prompt.npz`: the object mask on decode row `keyframe_index`.

    Raises:
        ValueError: If `keyframe_index < 0`, or `mask` is not a 2-D mask with at least one pixel set.
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H,W), got shape {mask.shape}")
    if not mask.any():
        raise ValueError("mask prompt is empty: SAM 2 and the track queries need at least one object pixel")
    if keyframe_index < 0:
        raise ValueError(f"keyframe_index must be >= 0, got {keyframe_index}")
    save_npz(Path(artifact_dir) / PROMPT_FILE, keyframe_index=np.array(keyframe_index, dtype=np.int64), mask=mask)


def read_mask_prompt(artifact_dir: Path) -> tuple[int, np.ndarray]:
    """`(keyframe_index, mask (H,W) bool)` from `<artifact_dir>/mask_prompt.npz`.

    Raises:
        FileNotFoundError: If the prompt has not been written.
    """
    path = Path(artifact_dir) / PROMPT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; write the clip's keyframe object mask with write_mask_prompt")
    with np.load(path, allow_pickle=False) as data:
        return int(data["keyframe_index"]), np.asarray(data["mask"], dtype=bool)


def make_mask_worker(variant: str = DEFAULT_SAM2_VARIANT, bf16: bool = True, device: str = "cuda",
                     predictor: Any = None, mask_compile: str = "none") -> StageFn:
    """The `mask` stage worker: SAM 2.1 propagation of the clip's prompt mask to every decode row.

    Args:
        variant: `"<size>"` or `"<size>@<short side px>"` (`fastdo.shape.sam2_variants.parse_sam2_variant`).
        bf16: Propagate under bf16 autocast on CUDA.
        device: Torch device the predictor loads on.
        predictor: A `SAM2VideoPredictor`-compatible object to use instead of loading `variant`'s checkpoint.
        mask_compile: `bench.mask_compile` (`fastdo.shape.sam2_compile.MASK_COMPILE_MODES`). `none` loads the eager
            predictor with exactly the call it always had; a vos mode loads upstream's compiled VOS predictor.

    Raises:
        ValueError: On an unknown variant or `mask_compile`, at construction.
    """
    parsed = parse_sam2_variant(variant)
    if check_mask_compile(mask_compile) == "none":
        def load() -> Any:
            return masks_sam2.load_sam2_predictor(device=device, model_id=parsed.model_id)
    else:
        def load() -> Any:
            return masks_sam2.load_sam2_predictor(device=device, model_id=parsed.model_id, mask_compile=mask_compile)
    ctx = WorkerContext(loader=load, model=predictor)

    def mask_worker(clip_id: str, artifact_dir: Path) -> None:
        ctx.budget()
        artifact_dir = Path(artifact_dir)
        frames, frame_idx = read_decode(artifact_dir)
        keyframe_index, prompt = read_mask_prompt(artifact_dir)
        masks = propagate_masks_at(torch.from_numpy(frames), keyframe_index, torch.from_numpy(prompt),
                                   short_side=parsed.short_side, device=device, predictor=ctx.model(), bf16=bf16)
        save_npz(artifact_dir / MASK_FILE, masks=masks.numpy().astype(bool), frame_idx=frame_idx,
                 keyframe_index=np.array(keyframe_index, dtype=np.int64))
        logger.info("propagated %s masks over %d frames of %r", parsed.label, len(frame_idx), clip_id)

    return mask_worker


def make_tracks_worker(n_queries: int = 96, seed: int = 0, device: str = "cuda", model: Any = None) -> StageFn:
    """The `tracks` stage worker: CoTracker3 online tracks of queries sampled in the clip's prompt mask.

    Args:
        n_queries: Query points sampled inside the keyframe prompt mask.
        seed: Seed of the query-sampling generator.
        device: Torch device the model loads on.
        model: A `CoTrackerOnlinePredictor`-compatible model to use instead of loading `cotracker3_online`.
    """
    ctx = WorkerContext(loader=lambda: cotracker._load_cotracker_model(device), model=model)

    def tracks_worker(clip_id: str, artifact_dir: Path) -> None:
        ctx.budget()
        artifact_dir = Path(artifact_dir)
        frames, frame_idx = read_decode(artifact_dir)
        keyframe_index, prompt = read_mask_prompt(artifact_dir)
        generator = torch.Generator().manual_seed(seed)
        query_uv = sample_query_points_in_mask(torch.from_numpy(prompt), n_queries, generator=generator)
        tracks, vis = track_points_cotracker(torch.from_numpy(frames), keyframe_index, query_uv, device=device,
                                             model=ctx.model())
        save_npz(artifact_dir / TRACKS_FILE, tracks=tracks.numpy().astype(np.float32), vis=vis.numpy().astype(bool),
                 query_uv=query_uv.numpy().astype(np.float32), frame_idx=frame_idx,
                 keyframe_index=np.array(keyframe_index, dtype=np.int64))
        logger.info("tracked %d queries over %d frames of %r", n_queries, len(frame_idx), clip_id)

    return tracks_worker
