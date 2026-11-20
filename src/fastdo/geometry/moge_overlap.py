# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions transcribed from MoGe (https://github.com/microsoft/MoGe, commit b942f00b, moge/model/v2.py
# `MoGeModel.infer` and moge/utils/geometry_torch.py `recover_focal_shift`),
# Copyright (c) Microsoft Corporation, MIT License; see licenses/MoGe-LICENSE.txt and THIRD_PARTY_LICENSES.md.
"""Overlapped MoGe-2 inference (opt-in): hide the per-batch CPU shift solve and the host copies.

`MoGeModel.infer` is serial per batch: the forward, then
`recover_focal_shift` copies a 64x64 downsample of the points to the host (a sync) and runs a scipy Levenberg-Marquardt
fit per frame while the GPU idles, then the rest of the post-processing. `run_overlapped` runs the batches as a 2-stage
software pipeline, A(0), A(1), B(0), A(2), B(1), ..., B(last), so B(k)'s CPU solve runs while A(k+1)'s forward is on
the GPU:

- `stage_a`: upload the batch (on CUDA from a pinned per-batch copy, `non_blocking`, on a copy stream the compute stream
  waits on), `model.forward` under `infer`'s autocast/inference-mode context (the CUDA-graphed forward when installed),
  then `infer`'s GPU work up to `recover_focal_shift`'s host copy; that copy goes into pinned buffers
  (`non_blocking`) followed by an event.
- `stage_b`: wait on the event, the per-frame solve with MoGe's own `solve_optimal_shift` / `solve_optimal_focal_shift`
  on identical arrays, then the rest of `infer`'s post-processing.

`stage_a` + `stage_b` are a VERBATIM transcription of `MoGeModel.infer` (with `recover_focal_shift` inlined and split at
its `.cpu()`) at the pinned moge commit `PINNED_MOGE_COMMIT`, for infer's defaults (`resolution_level=9`,
`force_projection=True`, `apply_mask=True`, `use_fp16=True`) and a 4-D image, with `fov_x` and `num_tokens`
as arguments: every op in the same order with the same dtypes. `num_tokens=None` (the default) takes infer's own
resolution_level-9 token formula, verbatim; an int is used as `infer(num_tokens=N)` uses it. Library functions `infer`
calls (`normalized_view_plane_uv`, the scipy solves, `utils3d.pt.intrinsics_from_focal_center`,
`utils3d.pt.depth_map_to_point_map`) are called, not copied. Two guards
enforce it in code (`check_transcribed_model`): the model's `infer` must BE `moge.model.v2.MoGeModel.infer` (so the
overlap replaces exactly the function transcribed), and the sha256 of the installed source of the transcribed moge
functions must equal `PINNED_SOURCE_SHA256`; on a moge upgrade the overlap refuses to run until the transcription is
re-checked and the pin updated.

Differences that do not change a value: the host copies go to pinned memory laid out as `.cpu()` lays them out
(`torch.empty_like`, preserved strides); the frames are uploaded per batch from a pinned copy instead of synchronously
from pageable memory. On the CPU every copy is the identity, as `.cpu()` is there, and the order A(0), A(1), B(0), ...
is kept, so CPU tests exercise the pipeline itself.

This module imports torch and the standard library only; moge, utils3d and scipy are imported when first needed.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "PINNED_MOGE_COMMIT",
    "PINNED_SOURCE_SHA256",
    "check_transcribed_model",
    "infer_transcribed",
    "model_input_on_device",
    "moge_source_sha256",
    "run_overlapped",
    "stage_a",
    "stage_b",
]

# `moge` 2.0.0 as installed by the `frontend` extra (`pyproject.toml`), and the sha256 of the source of the functions
# transcribed below (`moge_source_sha256`). Both pinned by tests/geometry/test_moge_overlap.py.
PINNED_MOGE_COMMIT = "b942f00bdc2a2a23ebb474fbe034d487e6dcceec"
PINNED_SOURCE_SHA256 = "9a756543c8b23da9d3105ec8ca5f05a79a0aeb7d92e1b7ab10cebbd49220c15c"

# `MoGeModel.infer`'s defaults (`run_moge_batches` passes at most `fov_x` and `num_tokens`) and
# `recover_focal_shift`'s default `downsample_size`.
_RESOLUTION_LEVEL = 9
_FORCE_PROJECTION = True
_APPLY_MASK = True
_USE_FP16 = True
_DOWNSAMPLE_SIZE = (64, 64)


def _transcribed_functions() -> tuple[Callable[..., Any], ...]:
    from moge.model.v2 import MoGeModel
    from moge.utils import geometry_numpy, geometry_torch
    return (MoGeModel.infer, geometry_torch.recover_focal_shift, geometry_numpy.solve_optimal_shift,
            geometry_numpy.solve_optimal_focal_shift)


@functools.cache
def moge_source_sha256() -> str:
    """sha256 of the installed source of `MoGeModel.infer`, `recover_focal_shift`, `solve_optimal_shift` and
    `solve_optimal_focal_shift`, concatenated in that order."""
    digest = hashlib.sha256()
    for fn in _transcribed_functions():
        digest.update(inspect.getsource(fn).encode())
    return digest.hexdigest()


def check_transcribed_model(model: Any) -> None:
    """Refuses a model whose `infer` is not the transcribed `MoGeModel.infer`, or an installed moge whose source moved.

    Raises:
        ValueError: If `type(model).infer` is not `moge.model.v2.MoGeModel.infer`, or the instance overrides `infer`.
        RuntimeError: If the installed source of the transcribed functions differs from `PINNED_SOURCE_SHA256`.
    """
    from moge.model.v2 import MoGeModel
    if getattr(type(model), "infer", None) is not MoGeModel.infer or "infer" in vars(model):
        raise ValueError("overlap=True replaces moge's MoGeModel.infer with a transcription of it, so it needs a model "
                         f"whose infer IS MoGeModel.infer; got {type(model).__name__}")
    if moge_source_sha256() != PINNED_SOURCE_SHA256:
        raise RuntimeError(f"the installed moge source of MoGeModel.infer/recover_focal_shift/solve_optimal_*shift "
                           f"has sha256 {moge_source_sha256()}, not the transcribed {PINNED_SOURCE_SHA256} (moge commit "
                           f"{PINNED_MOGE_COMMIT}): re-check fastdo.geometry.moge_overlap against it and re-pin")


def model_input_on_device(chunk: Tensor) -> Tensor:
    """`(B,H,W,3)` frames already on the model's device -> MoGe's `(B,3,H,W)` float input in [0,1]."""
    image = chunk.permute(0, 3, 1, 2)
    return image.float() / 255.0 if chunk.dtype == torch.uint8 else image.float()


class _Uploader:
    """`moge_model._model_input` per batch; on CUDA from a pinned copy, `non_blocking` on a copy stream."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

    def __call__(self, chunk: Tensor) -> Tensor:
        if self.stream is None or chunk.device.type != "cpu":
            return model_input_on_device(chunk.to(device=self.device))
        pinned = chunk.pin_memory()                               # same sizes and strides; the host allocator caches
        with torch.cuda.stream(self.stream):
            on_device = pinned.to(device=self.device, non_blocking=True)
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        on_device.record_stream(current)
        return model_input_on_device(on_device)


def _to_host(t: Tensor) -> Tensor:
    """`t.cpu()` issued asynchronously: a pinned buffer laid out as `.cpu()` lays it out. The identity on the CPU."""
    if t.device.type == "cpu":
        return t
    host = torch.empty_like(t, device="cpu", pin_memory=True)
    host.copy_(t, non_blocking=True)
    return host


@dataclass
class InFlight:
    """One batch between `stage_a` and `stage_b`: `infer`'s locals at `recover_focal_shift`'s host copy."""

    device_type: str
    aspect_ratio: float
    points: Tensor | None                 # infer's `points` (B,H,W,3) float32, before the shift
    normal: Tensor | None
    mask_binary: Tensor | None
    metric_scale: Tensor | None
    fov_x: Any
    # recover_focal_shift's locals
    shape: torch.Size | None = None
    rfs_points: Tensor | None = None      # its reshaped `points`
    rfs_mask: Tensor | None = None        # its reshaped `mask` (None iff infer's mask_binary is None)
    rfs_focal: Tensor | None = None       # its reshaped `focal` (None iff fov_x is None)
    uv_lr_host: Tensor | None = None
    points_lr_host: Tensor | None = None
    focal_host: Tensor | None = None
    mask_lr_host: Tensor | None = None
    ready: torch.cuda.Event | None = None


def stage_a(model: Any, image: Tensor, fov_x: Any = None, num_tokens: int | None = None) -> InFlight:
    """`MoGeModel.infer(image, fov_x=fov_x, num_tokens=num_tokens)` up to `recover_focal_shift`'s host copy, which is
    issued asynchronously.

    `image` is `(B,3,H,W)`. Call under `torch.inference_mode()` (`infer` is decorated with it).
    """
    if image.dim() != 4:
        raise ValueError(f"image must be (B,3,H,W), got {tuple(image.shape)}")
    # --- MoGeModel.infer ---
    image = image.to(dtype=model.dtype, device=model.device)

    original_height, original_width = image.shape[-2:]
    aspect_ratio = original_width / original_height

    # Determine the number of base tokens to use (resolution_level=9)
    if num_tokens is None:
        min_tokens, max_tokens = model.num_tokens_range
        num_tokens = int(min_tokens + (_RESOLUTION_LEVEL / 9) * (max_tokens - min_tokens))

    # Forward pass
    with torch.autocast(device_type=model.device.type, dtype=torch.float16,
                        enabled=_USE_FP16 and model.dtype != torch.float16):
        output = model.forward(image, num_tokens=num_tokens)
    points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])

    # Always process the output in fp32 precision
    points, normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x,  # noqa: C417
                                                     [points, normal, mask, metric_scale, fov_x])
    device_type = model.device.type
    with torch.autocast(device_type=device_type, dtype=torch.float32):
        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None
        state = InFlight(device_type, aspect_ratio, points, normal, mask_binary, metric_scale, fov_x)
        if points is not None:
            if fov_x is None:
                _recover_focal_shift_device(state, points, mask_binary, None)
            else:
                focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(
                    torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _recover_focal_shift_device(state, points, mask_binary, focal)
    return state


def _recover_focal_shift_device(state: InFlight, points: Tensor, mask: Tensor | None, focal: Tensor | None) -> None:
    """`recover_focal_shift(points, mask, focal)` up to and including its host copies (issued asynchronously)."""
    from moge.utils.geometry_torch import normalized_view_plane_uv
    downsample_size = _DOWNSAMPLE_SIZE
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)
    mask_lr: Any  # declared for mypy only; the lines below are verbatim

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size,
                                                      mode='nearest').squeeze(1) > 0

    state.shape, state.rfs_points, state.rfs_mask, state.rfs_focal = shape, points, mask, focal
    state.uv_lr_host = _to_host(uv_lr)
    state.points_lr_host = _to_host(points_lr.detach())
    state.focal_host = _to_host(focal) if focal is not None else None
    state.mask_lr_host = None if mask is None else _to_host(mask_lr)
    if points.device.type == "cuda":
        state.ready = torch.cuda.Event()
        state.ready.record(torch.cuda.current_stream(points.device))


def _recover_focal_shift_host(state: InFlight) -> tuple[Tensor, Tensor]:
    """The rest of `recover_focal_shift`: the per-frame scipy solve on the host arrays; returns `(focal, shift)`."""
    from moge.utils.geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift
    assert state.shape is not None and state.rfs_points is not None and state.uv_lr_host is not None
    assert state.points_lr_host is not None
    shape, points, mask, focal = state.shape, state.rfs_points, state.rfs_mask, state.rfs_focal

    focal_np: Any  # these four declared for mypy only; the lines below are verbatim
    mask_lr_np: Any
    optim_shift: Any
    optim_focal: Any
    uv_lr_np = state.uv_lr_host.numpy()
    points_lr_np = state.points_lr_host.numpy()
    focal_np = state.focal_host.numpy() if focal is not None and state.focal_host is not None else None
    mask_lr_np = None if mask is None or state.mask_lr_host is None else state.mask_lr_host.numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def stage_b(state: InFlight) -> dict[str, Tensor]:
    """The rest of `MoGeModel.infer` after `stage_a`: returns exactly `infer`'s dict (`points`, `intrinsics`, `depth`,
    `mask`, `normal`, the ones that are not None). Call under `torch.inference_mode()`."""
    import utils3d
    if state.ready is not None:
        state.ready.synchronize()
    aspect_ratio = state.aspect_ratio
    points, normal, mask_binary, metric_scale = state.points, state.normal, state.mask_binary, state.metric_scale
    # --- MoGeModel.infer, inside its `torch.autocast(device_type, dtype=torch.float32)` block ---
    with torch.autocast(device_type=state.device_type, dtype=torch.float32):
        if points is not None:
            focal, shift = _recover_focal_shift_host(state)
            fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            intrinsics = utils3d.pt.intrinsics_from_focal_center(
                fx, fy, torch.tensor(0.5, device=points.device, dtype=points.dtype),
                torch.tensor(0.5, device=points.device, dtype=points.dtype))
            points[..., 2] += shift[..., None, None]
            if mask_binary is not None:
                mask_binary &= points[..., 2] > 0        # in case depth is contains negative values
            depth = points[..., 2].clone()
        else:
            depth, intrinsics = None, None

        # If projection constraint is forced, recompute the point map using the actual depth map & intrinsics
        if _FORCE_PROJECTION and depth is not None:
            points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)

        # Apply metric scale
        if metric_scale is not None:
            if points is not None:
                points *= metric_scale[:, None, None, None]
            if depth is not None:
                depth *= metric_scale[:, None, None]

        # Apply mask
        if _APPLY_MASK and mask_binary is not None:
            points = torch.where(mask_binary[..., None], points, torch.inf) if points is not None else None
            depth = torch.where(mask_binary, depth, torch.inf) if depth is not None else None
            normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None

    return_dict = {
        'points': points,
        'intrinsics': intrinsics,
        'depth': depth,
        'mask': mask_binary,
        'normal': normal
    }
    return {k: v for k, v in return_dict.items() if v is not None}


def infer_transcribed(model: Any, image: Tensor, fov_x: Any = None, num_tokens: int | None = None) -> dict[str, Tensor]:
    """`stage_b(stage_a(model, image, fov_x, num_tokens))` under inference mode: `model.infer(image, fov_x=fov_x,
    num_tokens=num_tokens)`, unpipelined."""
    with torch.inference_mode():
        return stage_b(stage_a(model, image, fov_x, num_tokens))


def _pipelined(items: Iterator[tuple[int, Tensor]], model: Any, fov_x: Any,
               num_tokens: int | None = None) -> Iterator[tuple[int, dict[str, Tensor]]]:
    """A(0), A(1), B(0), A(2), B(1), ..., B(last): yields `(start, infer output)` per batch, in order."""
    pending: tuple[int, InFlight] | None = None
    for start, image in items:
        state = stage_a(model, image, fov_x, num_tokens)
        if pending is not None:
            yield pending[0], stage_b(pending[1])
        pending = (start, state)
    if pending is not None:
        yield pending[0], stage_b(pending[1])


def run_overlapped(frames: Tensor, device: str, model: Any, batch_size: int, fov_x: Any,
                   emit: Callable[[int, Tensor], None], num_tokens: int | None = None) -> None:
    """`emit(start, model.infer(frames[start:start+batch_size], fov_x=fov_x, num_tokens=num_tokens)["points"])` per
    batch, pipelined.

    `emit(k)` runs inside B(k), after A(k+1) was enqueued. Call under `torch.inference_mode()`.

    Raises:
        ValueError, RuntimeError: From `check_transcribed_model`.
    """
    check_transcribed_model(model)
    upload = _Uploader(device)
    n = int(frames.shape[0])
    batches = ((start, upload(frames[start:min(start + batch_size, n)])) for start in range(0, n, batch_size))
    for start, output in _pipelined(batches, model, fov_x, num_tokens):
        emit(start, output["points"])
