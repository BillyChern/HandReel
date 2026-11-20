"""Runs the open MoGe-2 depth/pointmap model on decoded RGB frames.

Unlike `fastdo.geometry.moge_pointmaps.load_pointmaps` (which loads pointmaps the baseline's own
`get_pointmap_dir.py` already computed and saved to disk), this module runs MoGe-2 itself, so a
front end with no precomputed baseline artifacts (e.g. DexYCB's `mode: real`) can still get
per-frame camera-frame pointmaps. Mirrors `fastdo.shape.masks_sam2`'s lazy-import pattern: the
`moge` package (the `frontend` extra in `pyproject.toml`, pinned to a pre-MoGe-3 commit of
`microsoft/MoGe` to avoid its later `flex-gemm` compiled-CUDA-extension dependency and cu130-only
torch index) is imported only inside `_load_model`, so this module -- and anything that imports
`fastdo.geometry` -- still imports with only the base install; only calling `run_moge` without an
injected `model` needs the extra.

The checkpoint is the open `Ruicheng/moge-2-vitl` (MoGe-2 ViT-L, ungated on Hugging Face).

`run_moge` returns the model's own point map unscaled. MoGe-2 predicts metric scale, but its
per-frame metric estimate is not exact, so callers that need metres apply
`hand_anchored_scale`/`apply_metric_scale` (`fastdo.geometry.moge_pointmaps`) first, exactly as
for `load_pointmaps`' raw MoGe-1 output. Pixels MoGe masks out as invalid (it writes `inf` there)
come back as NaN, matching `load_pointmaps`' invalid-pixel convention.

Camera model: MoGe's `infer` rebuilds its point map from the predicted depth with the principal
point at the image centre (pixel index `((W-1)/2, (H-1)/2)`) and, without `fov_x`, its own focal
length. A calibrated camera's principal point is generally elsewhere (DexYCB: ~11 px off), and a
point map built around the wrong centre is shifted sideways by `dc * Z / f` (13 mm at 0.75 m for
11 px at f = 617 px) against every term that projects with the real `K`. So when the caller has
pixel intrinsics, `run_moge(intrinsics=K)` passes `K`'s FoV to MoGe and re-expresses the point map
through `K` itself: `X = (u - cx) Z / fx`, `Y = (v - cy) Z / fy`, keeping MoGe's depth `Z`. The
result then satisfies `project(pointmap[v, u], K) == (u, v)`, the same convention as the ground-truth
depth back-projection in `fastdo.io.datasets.dexycb_frames`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.geometry import moge_overlap

logger = logging.getLogger(__name__)

__all__ = ["MODEL_ID", "NUM_TOKENS_RANGE", "check_num_tokens", "fov_x_from_intrinsics", "moge_intrinsics", "run_moge",
           "run_moge_batches"]

MODEL_ID = "Ruicheng/moge-2-vitl"
# Frames per MoGe-2 forward pass. Bounds peak activation memory for the shared GPU: measured on 1280x720
# whisking frames, batch 1 / 2 / 4 peak at 3.2 / 4.0 / >6.2 GiB reserved, and the shared host now caps our
# processes at 6 GiB each (8 GiB combined); batch 2 is ~2x faster than batch 1 at the same accuracy.
# The default of every `batch_size=None` below, read at call time. `geometry_stage.moge_batch_size` overrides it
# per worker; batch 8 saves ~0.8 s per 1080p window but is NOT bit-identical (fp16 kernel choice), so it stays
# opt-in until it passes the 71-clip DexYCB gate.
_BATCH_SIZE = 2
# The explicit `num_tokens` values accepted (inclusive): MoGe-2's own `num_tokens_range` (pinned rev `b942f00`,
# `moge/model/v2.py`). `None` -- every existing call -- is `infer`'s default, resolution_level 9 = the top of the range
# (3600 base ViT tokens), and is NOT passed to `infer` at all, so it stays byte-identical. Fewer tokens are faster but
# move the depth (measured at 1440x1920: 2500 tokens -26% MoGe time, median depth ratio 0.97 vs 3600): a QUALITY lever.
NUM_TOKENS_RANGE = (1200, 3600)


def check_num_tokens(num_tokens: int | None, what: str = "num_tokens") -> int | None:
    """`num_tokens` unchanged if `None` (MoGe's default, 3600) or an int in `NUM_TOKENS_RANGE` (bools refused).

    Raises:
        ValueError: On anything else, naming `what`.
    """
    lo, hi = NUM_TOKENS_RANGE
    if num_tokens is not None and (not isinstance(num_tokens, int) or isinstance(num_tokens, bool)
                                   or not lo <= num_tokens <= hi):
        raise ValueError(f"{what} must be None (MoGe's default, {hi}) or an int in [{lo}, {hi}], got {num_tokens!r}")
    return num_tokens


def _token_kwargs(num_tokens: int | None) -> dict[str, Any]:
    """`infer`'s `num_tokens` keyword: absent for `None`, so the default call is unchanged."""
    return {} if num_tokens is None else {"num_tokens": num_tokens}


def _resolve_batch_size(batch_size: int | None) -> int:
    """`batch_size`, or `_BATCH_SIZE` for `None`; an int >= 1 (bools refused).

    Raises:
        ValueError: On anything else.
    """
    if batch_size is None:
        return _BATCH_SIZE
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1 (or None for {_BATCH_SIZE}), got {batch_size!r}")
    return batch_size


def fov_x_from_intrinsics(K: Tensor, width: int) -> float:
    """Horizontal field of view in degrees for pixel intrinsics `K` at image width `width`.

    MoGe's `infer(fov_x=...)` takes the FoV in degrees; passing the known camera FoV makes MoGe's
    depth consistent with `K`'s focal length (MoGe otherwise re-estimates it).
    """
    return math.degrees(2.0 * math.atan(0.5 * width / float(K[0, 0])))


def run_moge(frames: Tensor, device: str = "cuda", model: Any = None, intrinsics: Tensor | None = None,
             batch_size: int | None = None, overlap: bool = False, num_tokens: int | None = None) -> Tensor:
    """Runs MoGe-2 on a batch of RGB frames to get camera-frame pointmaps.

    Args:
        frames: `(N,H,W,3)` uint8 RGB frames (float frames are taken to be in [0,1] already).
        device: torch device string for the model and its inputs.
        model: A `moge.model.v2.MoGeModel`-compatible object exposing `.infer(image (B,3,H,W),
            **kwargs) -> dict` with a `"points"` entry of shape `(B,H,W,3)`. `None` loads the real
            `Ruicheng/moge-2-vitl` model via `_load_model`; tests inject a fake instead.
        intrinsics: Optional known `(3,3)` pixel intrinsics of the frames. When given, MoGe gets
            their horizontal FoV (`fov_x_from_intrinsics`) and the returned points are rebuilt from
            MoGe's depth through these intrinsics (see the module docstring), so they project back
            onto their own pixels with `K`. `None` returns MoGe's own point map (its estimated focal
            length, principal point at the image centre).
        batch_size: Frames per `infer` call; `None` is `_BATCH_SIZE` (2). Not bit-neutral on the real model.
        overlap: Pipeline the batches (`run_moge_batches`); needs a `MoGeModel` (its `infer` transcribed).
        num_tokens: MoGe-2's base ViT tokens, `infer(num_tokens=...)`; `None` (the default) is not passed, i.e.
            `infer`'s own resolution_level 9 = 3600. Else an int in `NUM_TOKENS_RANGE`. Not depth-neutral.

    Returns:
        `(N,H,W,3)` float32 camera-frame XYZ on the CPU, the model's own depth unscaled, NaN where
        the model marked a pixel invalid.

    Raises:
        ValueError: If `frames` is not `(N,H,W,3)`, `intrinsics` is not `(3,3)`, `batch_size` is not an int >= 1,
            `num_tokens` is not None or an int in `NUM_TOKENS_RANGE`, or `overlap` is not a bool or is True with a
            model whose `infer` is not `MoGeModel.infer`.
        RuntimeError: If `model` is `None` and the `moge` package is not installed, or `overlap` is True and the
            installed moge source differs from the transcribed one (`moge_overlap.PINNED_SOURCE_SHA256`).
    """
    _check_inputs(frames, intrinsics)
    batch_size = _resolve_batch_size(batch_size)
    check_num_tokens(num_tokens)
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    out = np.empty((n, h, w, 3), dtype=np.float32)

    def to_host(start: int, points: Tensor) -> None:
        out[start:start + int(points.shape[0])] = points.cpu().numpy()

    run_moge_batches(frames, to_host, device=device, model=model, intrinsics=intrinsics, batch_size=batch_size,
                     overlap=overlap, num_tokens=num_tokens)
    return torch.from_numpy(out)


def run_moge_batches(frames: Tensor, sink: Callable[[int, Tensor], None], device: str = "cuda", model: Any = None,
                     intrinsics: Tensor | None = None, batch_size: int | None = None, overlap: bool = False,
                     num_tokens: int | None = None) -> None:
    """`run_moge` without the host copy: each batch's points go to `sink(start, points)` on the model's device.

    `points` is the `(B,H,W,3)` float32 block of rows `start:start+B` that `run_moge` would copy to the host,
    bit for bit (`run_moge` is this function with a sink that copies). It is only valid during the call, which
    runs under `torch.inference_mode()`. The geometry stage's masked storage gathers only the pixels it
    keeps, instead of copying 2.5-5.9 GB of full-resolution pointmaps per 100-row window to the host. `batch_size`
    is `run_moge`'s: the blocks are `batch_size` rows (`None` = `_BATCH_SIZE`), the last one possibly shorter.

    `overlap=True` runs the batches as a 2-stage pipeline (`fastdo.geometry.moge_overlap`): batch k's CPU shift
    solve and the rest of its post-processing (then this function's own invalid mask, back-projection and `sink`)
    run while batch k+1's forward is on the GPU, through a verbatim transcription of `MoGeModel.infer`, so the blocks
    are meant to be bit-identical to `overlap=False`. It needs a model whose `infer` is `MoGeModel.infer`; the forward
    may be the CUDA-graphed one (`fastdo.geometry.moge_cuda_graph`). False (the default) is the serial path, unchanged.
    Opt-in until its GPU bit-identity gate and an Ego4D comparison pass.

    `num_tokens` is `run_moge`'s, on both paths: `None` never reaches `infer` (its default, 3600 tokens); an int
    is passed as `infer(num_tokens=...)`, and the overlapped path uses it exactly as `infer` would.

    Raises:
        ValueError: If `frames` is not `(N,H,W,3)`, `intrinsics` is not `(3,3)`, `batch_size` is not an int >= 1,
            `num_tokens` is not None or an int in `NUM_TOKENS_RANGE`, or `overlap` is not a bool or is True with a
            model whose `infer` is not `MoGeModel.infer`.
        RuntimeError: If `model` is `None` and the `moge` package is not installed, or `overlap` is True and the
            installed moge source differs from the transcribed one (`moge_overlap.PINNED_SOURCE_SHA256`).
    """
    _check_inputs(frames, intrinsics)
    batch_size = _resolve_batch_size(batch_size)
    if not isinstance(overlap, bool):
        raise ValueError(f"overlap must be a bool, got {overlap!r}")  # noqa: TRY004  # as the batch_size check
    check_num_tokens(num_tokens)
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    if model is None:
        model = _load_model(device)
    infer_kwargs: dict[str, Any] = {} if intrinsics is None else {"fov_x": fov_x_from_intrinsics(intrinsics, w)}
    infer_kwargs.update(_token_kwargs(num_tokens))

    # The invalid-pixel mask and the back-projection run per batch on the model's device: done afterwards in numpy
    # on the whole (N,H,W,3) array they cost 6.7 s of a 12.0 s call at S2's 100 x 1080x1920 (the model ~4.5 s).
    # Same float32 operations in the same order as `_points_through_intrinsics`, so the result is bit-identical.
    grid: tuple[Tensor, Tensor] | None = None

    def emit(start: int, points: Tensor) -> None:
        nonlocal grid
        points = torch.as_tensor(points).detach().to(torch.float32)
        invalid = ~torch.isfinite(points).all(dim=-1)
        if intrinsics is None:
            points = points.masked_fill(invalid[..., None], math.nan)
        else:
            if grid is None:
                grid = _pixel_grid(h, w, points.device)
            points = _back_project(points[..., 2].masked_fill(invalid, math.nan), intrinsics, grid)
        sink(start, points)

    with torch.inference_mode():
        if overlap:
            moge_overlap.run_overlapped(frames, device, model, batch_size, infer_kwargs.get("fov_x"), emit,
                                        num_tokens=num_tokens)
            return
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            emit(start, model.infer(_model_input(frames[start:end], device), **infer_kwargs)["points"])


def _check_inputs(frames: Tensor, intrinsics: Tensor | None) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {tuple(frames.shape)}")
    if intrinsics is not None and tuple(intrinsics.shape) != (3, 3):
        raise ValueError(f"intrinsics must be (3,3), got {tuple(intrinsics.shape)}")


def moge_intrinsics(frames: Tensor, device: str = "cuda", model: Any = None, batch_size: int | None = None,
                    num_tokens: int | None = None) -> Tensor:
    """MoGe-2's own normalized camera intrinsics for each frame.

    Args:
        frames: `(N,H,W,3)` uint8 RGB frames (float frames are taken to be in [0,1] already).
        device: torch device string for the model and its inputs.
        model: A `MoGeModel`-compatible object whose `.infer(image (B,3,H,W))` returns an `"intrinsics"`
            entry of shape `(B,3,3)`. `None` loads the real model via `_load_model`.
        batch_size: Frames per `infer` call; `None` is `_BATCH_SIZE` (2).
        num_tokens: `infer(num_tokens=...)`; `None` (the default) is not passed (MoGe's own 3600).

    Returns:
        `(N,3,3)` float32 on the CPU, exactly `infer(batch)["intrinsics"]` in `batch_size` chunks. These
        are normalized by the image size (pinned rev `b942f00`, `moge/model/v2.py` `infer`): `fx/W`, `fy/H`,
        `cx = cy = 0.5`; `fastdo.geometry.intrinsics.pixel_intrinsics` converts them to pixels.

    Raises:
        ValueError: If `frames` is not `(N,H,W,3)`, `batch_size` is not an int >= 1, or `num_tokens` is not None or an
            int in `NUM_TOKENS_RANGE`.
        RuntimeError: If `model` is `None` and the `moge` package is not installed.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {tuple(frames.shape)}")
    batch_size = _resolve_batch_size(batch_size)
    token_kwargs = _token_kwargs(check_num_tokens(num_tokens))
    n = int(frames.shape[0])
    if model is None:
        model = _load_model(device)
    out = np.empty((n, 3, 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            K = model.infer(_model_input(frames[start:end], device), **token_kwargs)["intrinsics"]
            out[start:end] = torch.as_tensor(K).detach().to("cpu", torch.float32).numpy()
    return torch.from_numpy(out)


def _model_input(frames: Tensor, device: str) -> Tensor:
    """`(B,H,W,3)` frames -> MoGe's `(B,3,H,W)` float input in [0,1] on `device`."""
    return moge_overlap.model_input_on_device(frames.to(device=device))


def _pixel_grid(h: int, w: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """`(vv, uu)` float32 `(H,W)` pixel row and column coordinates on `device`."""
    return torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                          torch.arange(w, dtype=torch.float32, device=device), indexing="ij")


def _back_project(depth: Tensor, K: Tensor, grid: tuple[Tensor, Tensor]) -> Tensor:
    """`(B,H,W,3)` camera points from `(B,H,W)` depth: `_points_through_intrinsics` in torch, on `depth`'s device."""
    vv, uu = grid
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    # Divide by 0-d DEVICE tensors: CUDA's `tensor / python_float` multiplies by the reciprocal, 1 ulp off numpy's
    # true division on ~36% of pixels; tensor / tensor is IEEE division on every device.
    fx_t, fy_t = (torch.tensor(v, dtype=torch.float32, device=depth.device) for v in (fx, fy))
    return torch.stack([(uu - cx) * depth / fx_t, (vv - cy) * depth / fy_t, depth], dim=-1)


def _points_through_intrinsics(depth: np.ndarray, K: Tensor) -> np.ndarray:
    """`(N,H,W,3)` camera points back-projected from `(N,H,W)` depth with pixel intrinsics `K`."""
    h, w = depth.shape[1:]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    vv, uu = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return np.stack([(uu - cx) * depth / fx, (vv - cy) * depth / fy, depth], axis=-1).astype(np.float32)


def _load_model(device: str) -> Any:
    """Loads the real MoGe-2 model -- untyped since `moge` ships no type stubs.

    Raises:
        RuntimeError: If `moge` is not importable, with an actionable install hint (rather than a
            bare `ModuleNotFoundError`), mirroring `fastdo.shape.masks_sam2._load_predictor`.
    """
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise RuntimeError(
            "moge is not installed. It ships in this project's optional 'frontend' extra: run "
            "`uv sync --extra dev --extra frontend`. Weights download from (or are read from the "
            f"local Hugging Face cache for) https://huggingface.co/{MODEL_ID}."
        ) from exc
    from fastdo.model_pins import hf_revision

    revision = hf_revision(MODEL_ID)        # the commit scripts/fetch_weights.py downloads, not whatever `main` is now
    logger.info("loading MoGe-2 checkpoint %s @ %s on %s", MODEL_ID, revision, device)
    return MoGeModel.from_pretrained(MODEL_ID, revision=revision).to(device).eval()
