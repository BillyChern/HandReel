"""CUDA-graph replay of the MoGe-2 forward (opt-in until its GPU gate passes).

`MoGeModel.infer` calls `self.forward(image, num_tokens=...)` once per batch, inside `torch.autocast(fp16)` and
`torch.inference_mode()`. At the production batch of 2 and 3600 tokens, that forward launches ~1,285 kernels, and
about 1.6 s of a 100-row window is launch/CPU time between them. `graphed_moge_forward(model)` installs, on the
model instance, a `GraphedMoGeForward` in place of `model.forward`: per call signature it runs the eager forward twice
on a side stream (warm-up), captures one `torch.cuda.CUDAGraph`, and from then on copies the image into the graph's
static input, replays it and returns a dict of CLONES of its outputs. The replay runs the same kernels on the same
values as the eager batch-2 forward, so it is meant to be bit-identical to it (batch 8 is not: other fp16 kernels).
All graphs share one memory pool (`torch.cuda.graph_pool_handle()`), so a tail batch adds little. This is the
SAM 3D graph pattern (`fastdo.shape.sam3d_cuda_graph`); the structure and the safety rules are the same.

The signature is: the image's shape, stride, dtype and device; `num_tokens`; the autocast state (enabled, dtype); the
inference-mode state; `model.onnx_compatible_mode`; and the backend flags that choose kernels (cuDNN, TF32, reduced-
precision reductions, the SDPA backends, deterministic algorithms), so a replay never answers a call whose eager
forward would have launched other kernels.

What the MoGe-2 forward does (read at the pinned rev: `moge/model/v2.py` `MoGeModel.forward`, `moge/model/modules.py`
`DINOv2Encoder`/`ConvStack`/`MLP`, `moge/model/dinov2/` `DinoVisionTransformer`, `NestedTensorBlock`,
`MemEffAttention`, `PatchEmbed`, `LayerScale`, `Mlp`; `moge/utils/geometry_torch.py` `normalized_view_plane_uv`) and
how each capture hazard is handled:

- Host syncs / `.item()`: none on the forward path. The only `.item()` in DINOv2 is the drop-path schedule in
  `DinoVisionTransformer.__init__`; the `assert`s in `PatchEmbed.forward` and `interpolate_pos_encoding` compare
  Python ints taken from shapes. The geometry stage's syncs (`recover_focal_shift`'s per-frame CPU scipy solve, the
  sink's `.cpu()`) are in `infer` after `forward` returns, outside the graph.
- CPU tensors: none. `image_mean`/`image_std` are device buffers; `normalized_view_plane_uv` builds its UV grid with
  `torch.linspace(..., device=device)` on the device (no host-to-device copy, unlike the SAM 3D graph's timestep
  frequencies). So no replacement function is needed; a capture that meets an unsupported operation raises and that
  signature stays eager (below).
- Data-dependent shapes: `num_tokens` as a TENSOR makes `base_h`/`base_w` device tensors (`.round().long()`) that then
  size `F.interpolate`, `unflatten` and the UV grids -- a host read and shapes known only at run time. Such a call is
  never captured (eager). The production `infer` passes a Python int, which is in the signature, so `base_h`/`base_w`,
  the aspect ratio and every Python-level shape branch (`interpolate_pos_encoding`'s square fast path, the neck's
  per-level sizes) are fixed per signature.
- RNG: none in eval mode (Dropout p=0, `DropPath`/stochastic depth only when `training` with a drop rate, and the ViT-L
  checkpoint's rate is 0; `F.scaled_dot_product_attention` with no dropout). `model.training` makes a call eager
  (training-mode blocks may call `torch.randperm`). As in the SAM 3D graph, the CPU, CUDA and Python RNG states are
  saved before the warm-up and restored after the capture; a warm-up that changed any of them marks the signature eager
  forever.
- Attention: xFormers is not installed, so `MemEffAttention` runs `Attention.forward`'s
  `F.scaled_dot_product_attention` (capturable; the backend choice is a host-side decision on fixed shapes and the
  SDPA flags, which are in the signature). With xFormers present the same capture would include its kernels.
- Autocast: `forward` runs under `infer`'s `torch.autocast(fp16)`; the weight casts are recorded inside the graph with
  autocast's cast cache off during warm-up and capture (a cast cached before the capture would be freed when `infer`'s
  autocast block exits, under a live graph).
- Input strides: `moge_model._model_input` hands MoGe a permuted (channels-last-strided) image and `infer`'s
  `.to(dtype, device)` keeps it, and `F.interpolate` takes stride-dependent kernels; the static input is allocated with
  the call's own strides (`torch.empty_strided`) and the stride is in the signature.
- Outputs: `infer` writes into its outputs in place (`points[..., 2] += shift` after `.float()`, a no-op on float32
  outputs), so every replay returns clones (`preserve_format`: the eager output's strides) and the graph's static
  outputs are never exposed.
- Parameters: a graph reads the weights at their addresses. The parameter and buffer addresses are checked on every
  call (~0.04 ms for MoGe-2's 483 tensors); if any moved (`model.to`, `.half()`), every graph is dropped and recaptured.
  Replacing Parameter OBJECTS is not supported (nothing in fastdo does either).

Anything the capture does not support runs the eager forward and is counted in `stats["eager_calls"]`: grad mode,
training mode, a non-CUDA input or one requiring grad, a tensor `num_tokens`, arguments other than `(image,
num_tokens)`, a warm-up that consumed randomness, a capture that raises, or a non-dict output. Calls are serialised by
a lock, and each call's stream waits for the previous replay's (copy, replay, clone), so the static buffers are never
shared by two calls in flight.

This module imports torch and the standard library only.
"""

from __future__ import annotations

import logging
import random
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = ["WARMUP_CALLS", "GraphedMoGeForward", "eager_moge_forward", "graphed_moge_forward", "installed_runner"]

logger = logging.getLogger(__name__)

WARMUP_CALLS = 2        # eager forwards on the capture stream before a capture (torch's own advice)

TensorSig = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]


def _sig(x: Any) -> TensorSig | None:
    if not isinstance(x, Tensor) or not x.is_cuda or x.requires_grad:
        return None
    return tuple(x.shape), tuple(x.stride()), x.dtype, x.device


def _static(x: Tensor) -> Tensor:
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device).copy_(x)


def _backend_flags() -> tuple[Any, ...]:
    """The global switches that change which kernels an eager forward launches."""
    cuda = torch.backends.cuda
    cudnn_sdp = getattr(cuda, "cudnn_sdp_enabled", lambda: None)()
    return (torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32, cuda.matmul.allow_tf32,
            cuda.matmul.allow_fp16_reduced_precision_reduction, cuda.matmul.allow_bf16_reduced_precision_reduction,
            cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled(), cudnn_sdp,
            torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision())


def _rng_states(device: torch.device) -> tuple[Tensor, Tensor, Any]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(device), random.getstate()


def _set_rng_states(states: tuple[Tensor, Tensor, Any], device: torch.device) -> None:
    torch.set_rng_state(states[0])
    torch.cuda.set_rng_state(states[1], device)
    random.setstate(states[2])


def _same_rng(a: tuple[Tensor, Tensor, Any], b: tuple[Tensor, Tensor, Any]) -> bool:
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    image: Tensor
    outputs: dict[str, Tensor]

    def replay(self, image: Tensor) -> dict[str, Tensor]:
        self.image.copy_(image)
        self.graph.replay()
        return {name: out.clone() for name, out in self.outputs.items()}


class GraphedMoGeForward:
    """A MoGe-2 model's `forward`, captured once per call signature and replayed (module docstring).

    Holds the model weakly: installed on it as `model.forward`, it makes no reference cycle.

    Args:
        model: A `moge.model.v2.MoGeModel` (or any module whose `forward(image, num_tokens)` returns a dict of tensors).
        warmup: Eager forwards on the capture stream before each capture.

    Attributes:
        stats: Counters since the last `take_stats`: `captures`, `capture_seconds` (warm-up plus capture, synchronised),
            `replays` and `eager_calls`.
    """

    def __init__(self, model: torch.nn.Module, warmup: int = WARMUP_CALLS) -> None:
        self._model = weakref.ref(model)
        self._forward: Callable[..., Any] = type(model).forward        # the class's eager forward, unbound
        self.warmup = warmup
        self.graphs: dict[Any, _Graph | None] = {}      # None: this signature stays eager
        self.streams: dict[torch.device, tuple[torch.cuda.Stream, Any]] = {}     # device -> (capture stream, pool)
        self.stats = self._zero_stats()
        self._tensors: list[Tensor] | None = None       # the parameters and buffers the graphs read
        self._addresses: tuple[int, ...] = ()
        self._lock = threading.Lock()
        self._last: torch.cuda.Event | None = None      # recorded after the previous replay's clones

    @staticmethod
    def _zero_stats() -> dict[str, Any]:
        return {"captures": 0, "capture_seconds": 0.0, "replays": 0, "eager_calls": 0}

    def take_stats(self) -> dict[str, Any]:
        """The counters since the previous call, then resets them."""
        with self._lock:
            stats, self.stats = self.stats, self._zero_stats()
        return stats

    def _module(self) -> torch.nn.Module:
        model = self._model()
        if model is None:
            raise RuntimeError("the MoGe model this graph runner belongs to no longer exists")
        return model

    def eager(self, image: Any, *args: Any, **kwargs: Any) -> Any:
        """The model's own (class) forward."""
        return self._forward(self._module(), image, *args, **kwargs)

    def signature(self, image: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """The graph key of one call, or None when the call is not capturable."""
        if kwargs and (args or set(kwargs) != {"num_tokens"}):
            return None
        if not kwargs and len(args) != 1:
            return None
        num_tokens = kwargs["num_tokens"] if kwargs else args[0]
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            return None                                 # a tensor num_tokens sizes the forward at run time
        model = self._module()
        if torch.is_grad_enabled() or model.training:
            return None
        image_sig = _sig(image)
        if image_sig is None:
            return None
        autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        return (image_sig, num_tokens, autocast, torch.is_inference_mode_enabled(),
                bool(getattr(model, "onnx_compatible_mode", False)), _backend_flags())

    def __call__(self, image: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            key = self.signature(image, args, kwargs)
            if key is not None:
                self._check_weights()
                if key not in self.graphs:
                    self.graphs[key] = self._capture(image, kwargs["num_tokens"] if kwargs else args[0])
            graph = None if key is None else self.graphs[key]
            if graph is None:
                self.stats["eager_calls"] += 1
            else:
                self.stats["replays"] += 1
                stream = torch.cuda.current_stream(image.device)
                if self._last is not None:
                    stream.wait_event(self._last)       # the previous call's clones have read the static outputs
                outputs = graph.replay(image)
                self._last = torch.cuda.Event()
                self._last.record(stream)
                return outputs
        return self.eager(image, *args, **kwargs)

    def _check_weights(self) -> None:
        """Drops every graph if a parameter or buffer moved since the graphs were captured."""
        if self._tensors is not None and tuple(t.data_ptr() for t in self._tensors) == self._addresses:
            return
        if self._tensors is not None and self.graphs:
            logger.warning("MoGe-2 parameters moved since the CUDA graphs were captured; dropping %d graph(s)",
                           len(self.graphs))
            self.graphs.clear()
        model = self._module()
        self._tensors = [*model.parameters(), *model.buffers()]
        self._addresses = tuple(t.data_ptr() for t in self._tensors)

    def _capture(self, image: Tensor, num_tokens: int) -> _Graph | None:
        start = time.perf_counter()
        device = image.device
        if device not in self.streams:
            self.streams[device] = (torch.cuda.Stream(device), torch.cuda.graph_pool_handle())
        stream, pool = self.streams[device]
        static_image = _static(image)

        def run() -> Any:
            return self.eager(static_image, num_tokens=num_tokens)

        rng = _rng_states(device)
        outputs: Any = None
        graph: torch.cuda.CUDAGraph | None = None
        enabled, dtype = torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
        cast_cache_off = torch.autocast("cuda", enabled=True, dtype=dtype, cache_enabled=False) if enabled \
            else nullcontext()
        try:
            with cast_cache_off:
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup):
                        run()
                torch.cuda.current_stream(device).wait_stream(stream)
                if not _same_rng(_rng_states(device), rng):
                    logger.warning("the MoGe-2 forward drew random numbers; image %s stays eager",
                                   tuple(image.shape))
                    return None
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
                    outputs = run()
        except RuntimeError as exc:
            logger.warning("capturing the MoGe-2 forward failed for image %s; that signature stays eager: %s",
                           tuple(image.shape), exc)
            return None
        finally:
            _set_rng_states(rng, device)
            torch.cuda.synchronize(device)
            self.stats["captures"] += 1
            self.stats["capture_seconds"] += time.perf_counter() - start
        if not isinstance(outputs, dict) or not all(isinstance(v, Tensor) for v in outputs.values()):
            return None
        return _Graph(graph, static_image, outputs)


_RUNNERS: weakref.WeakKeyDictionary[torch.nn.Module, GraphedMoGeForward] = weakref.WeakKeyDictionary()


def graphed_moge_forward(model: torch.nn.Module) -> GraphedMoGeForward:
    """Installs the model's `GraphedMoGeForward` as `model.forward` and returns it; idempotent.

    The runner (and its graphs) is created on first use and kept for the model's lifetime, so installing again after
    `eager_moge_forward` reuses the captured graphs.

    Raises:
        TypeError: If `model` is not a `torch.nn.Module`.
        RuntimeError: If `model.forward` is already overridden on the instance by something else.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"graphed_moge_forward needs a torch.nn.Module (MoGeModel), got {type(model).__name__}")
    runner = _RUNNERS.get(model)
    current = vars(model).get("forward")
    if current is not None:
        if current is runner:
            return runner
        raise RuntimeError("the MoGe-2 model's forward is already overridden on the instance")
    if runner is None:
        runner = _RUNNERS[model] = GraphedMoGeForward(model)
    model.forward = runner
    return runner


def installed_runner(model: Any) -> GraphedMoGeForward | None:
    """The `GraphedMoGeForward` installed as `model.forward`, or None."""
    current = vars(model).get("forward") if hasattr(model, "__dict__") else None
    return current if isinstance(current, GraphedMoGeForward) else None


def eager_moge_forward(model: torch.nn.Module) -> None:
    """Restores the class forward if `graphed_moge_forward` installed the runner (its graphs are kept)."""
    if installed_runner(model) is not None:
        del model.forward
