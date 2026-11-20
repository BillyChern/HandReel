# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# This file reproduces one short expression of SAM 3D Objects (https://github.com/facebookresearch/sam-3d-objects),
# a SAM Material of Meta Platforms, Inc.; that expression is distributed under the SAM License (licenses/SAM_LICENSE),
# the rest of the file under Apache-2.0. See THIRD_PARTY_LICENSES.md.
"""CUDA-graph replay of SAM 3D's sparse-structure flow backbone.

The sparse-structure flow calls its dense DiT backbone (`ss_generator.reverse_fn.backbone`, a
`SparseStructureFlowTdfyWrapper`) once or twice per Euler step at fixed shapes: the (1, 4096, 8) shape latent, the pose
tokens, the step's time and shortcut size and the request's condition tokens. Eager, each call launches thousands of
small kernels and the flow is launch-bound (GPU utilisation 34-44% in production). `graphed_ss_backbone` swaps the
backbone's `forward` for a `GraphedSsBackbone` for one block: per call signature (every input's shape, stride, dtype and
device, the `cfg` flag, the autocast state) it runs the eager forward twice on a side stream (warm-up), captures one
`torch.cuda.CUDAGraph`, and from then on copies the inputs into the graph's static buffers, replays it and returns
clones of its outputs. Graphs live as long as the backbone, so a worker captures once and replays for every request.

What stays exactly the eager computation:
- the captured kernels are the eager forward's own, with autocast's weight-cast cache off inside warm-up and capture
  (a cast cached before the capture would be freed when the pipeline's autocast block exits, under a live graph);
- `TimestepEmbedder.timestep_embedding` builds its frequencies on the CPU and copies them to the device on every call,
  which a capture cannot record. Inside warm-up and capture it is replaced by the same expression with those
  frequencies copied once and kept on the device; each warm-up call checks the replacement against the original,
  bit for bit, and raises on any difference;
- the CPU, CUDA and Python RNG states are saved before the warm-up and restored after the capture, so the extra
  forwards consume no randomness; a forward that changed any of them is never graphed (that signature stays eager).
Anything the capture does not support (grad mode, a non-CUDA input, a keyword other than `d` and `cfg`, a non-dict
output) runs the eager forward and is counted in `stats["eager_calls"]`.

This module imports torch, numpy and the standard library only: the SAM 3D worker imports it from its own directory.
"""

from __future__ import annotations

import importlib
import random
import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["TIMESTEP_EMBEDDER_MODULE", "WARMUP_CALLS", "GraphedSsBackbone", "graphed_ss_backbone"]

TIMESTEP_EMBEDDER_MODULE = "sam3d_objects.model.backbone.tdfy_dit.models.timestep_embedder"
WARMUP_CALLS = 2                        # eager forwards on the capture stream before a capture (torch's own advice)
_KWARGS = frozenset({"d", "cfg"})       # what the CFG wrapper and the shortcut generator pass to the backbone

TensorSig = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]


def _sig(x: Any) -> TensorSig | None:
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.requires_grad:
        return None
    return tuple(x.shape), tuple(x.stride()), x.dtype, x.device


def _static(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device).copy_(x)


@contextmanager
def _device_timestep_frequencies(freqs: dict[tuple[int, float, torch.device], torch.Tensor]) -> Iterator[None]:
    """`TimestepEmbedder.timestep_embedding` with its CPU-built frequencies copied to the device once and cached in
    `freqs`; outside a capture every call is checked against the original, bit for bit."""
    cls = importlib.import_module(TIMESTEP_EMBEDDER_MODULE).TimestepEmbedder
    saved = cls.__dict__["timestep_embedding"]
    original = saved.__func__

    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
        # The original's expression, line for line, except that `freqs` is looked up instead of rebuilt.
        half = dim // 2
        key = (dim, max_period, t.device)
        if key not in freqs:
            freqs[key] = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
                                   / half).to(device=t.device)
        args = t[:, None].float() * freqs[key][None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if not torch.cuda.is_current_stream_capturing() and not torch.equal(embedding, original(t, dim, max_period)):
            raise RuntimeError("the device-frequency timestep_embedding differs from SAM 3D's own")
        return embedding

    cls.timestep_embedding = staticmethod(timestep_embedding)
    try:
        yield
    finally:
        cls.timestep_embedding = saved


def _rng_states(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, Any]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(device), random.getstate()


def _set_rng_states(states: tuple[torch.Tensor, torch.Tensor, Any], device: torch.device) -> None:
    torch.set_rng_state(states[0])
    torch.cuda.set_rng_state(states[1], device)
    random.setstate(states[2])


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    latents: dict[str, torch.Tensor]
    t: torch.Tensor
    args: tuple[torch.Tensor, ...]
    d: torch.Tensor | None
    outputs: dict[str, torch.Tensor]

    def replay(self, latents: Mapping[str, torch.Tensor], t: torch.Tensor, args: tuple[Any, ...],
               d: torch.Tensor | None) -> dict[str, torch.Tensor]:
        for name, value in latents.items():
            self.latents[name].copy_(value)
        self.t.copy_(t)
        for static, value in zip(self.args, args, strict=True):
            static.copy_(value)
        if self.d is not None and d is not None:        # the signature makes both None or both tensors
            self.d.copy_(d)
        self.graph.replay()
        return {name: out.clone() for name, out in self.outputs.items()}


class GraphedSsBackbone:
    """The backbone's forward, captured once per call signature and replayed (module docstring).

    Args:
        backbone: `ss_generator.reverse_fn.backbone`.
        warmup: Eager forwards on the capture stream before each capture.

    Attributes:
        stats: Counters since the last `take_stats`: `captures`, `capture_seconds` (warm-up plus capture, synchronised),
            `replays` and `eager_calls`.
    """

    def __init__(self, backbone: torch.nn.Module, warmup: int = WARMUP_CALLS) -> None:
        self.forward: Callable[..., Any] = type(backbone).forward.__get__(backbone)    # the eager forward
        self.warmup = warmup
        self.graphs: dict[Any, _Graph | None] = {}      # None: this signature stays eager (a forward used the RNG)
        self.freqs: dict[tuple[int, float, torch.device], torch.Tensor] = {}
        self.pool: Any = None
        self.stream: torch.cuda.Stream | None = None
        self.stats = self._zero_stats()

    @staticmethod
    def _zero_stats() -> dict[str, Any]:
        return {"captures": 0, "capture_seconds": 0.0, "replays": 0, "eager_calls": 0}

    def take_stats(self) -> dict[str, Any]:
        """The counters since the previous call, then resets them."""
        stats, self.stats = self.stats, self._zero_stats()
        return stats

    def signature(self, latents: Any, t: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        """The graph key of one call, or None when the call is not capturable."""
        if torch.is_grad_enabled() or not isinstance(latents, dict) or not set(kwargs) <= _KWARGS:
            return None
        if not isinstance(kwargs.get("cfg", False), bool):
            return None
        sigs = [(name, _sig(value)) for name, value in latents.items()]
        t_sig, arg_sigs = _sig(t), tuple(_sig(a) for a in args)
        d = kwargs.get("d")
        d_sig = None if d is None else _sig(d)
        if (any(s is None for _, s in sigs) or t_sig is None or any(s is None for s in arg_sigs)
                or (d is not None and d_sig is None) or len({s[3] for _, s in sigs if s} | {t_sig[3]}) != 1):
            return None
        autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        return tuple(sigs), t_sig, arg_sigs, d_sig, bool(kwargs.get("cfg", False)), autocast

    def __call__(self, latents: Any, t: Any, *args: Any, **kwargs: Any) -> Any:
        key = self.signature(latents, t, args, kwargs)
        if key is not None and key not in self.graphs:
            self.graphs[key] = self._capture(latents, t, args, kwargs)
        graph = None if key is None else self.graphs[key]
        if graph is None:
            self.stats["eager_calls"] += 1
            return self.forward(latents, t, *args, **kwargs)
        self.stats["replays"] += 1
        return graph.replay(latents, t, args, kwargs.get("d"))

    def _capture(self, latents: dict[str, torch.Tensor], t: torch.Tensor, args: tuple[torch.Tensor, ...],
                 kwargs: Mapping[str, Any]) -> _Graph | None:
        start = time.perf_counter()
        device = t.device
        if self.stream is None:
            self.stream, self.pool = torch.cuda.Stream(device), torch.cuda.graph_pool_handle()
        static_latents = {name: _static(value) for name, value in latents.items()}
        static_t, static_args = _static(t), tuple(_static(a) for a in args)
        d = kwargs.get("d")
        static_d = None if d is None else _static(d)
        cfg = {"cfg": True} if kwargs.get("cfg", False) else {}

        def run() -> Any:
            return self.forward(static_latents, static_t, *static_args,
                                **({} if static_d is None else {"d": static_d}), **cfg)

        rng = _rng_states(device)
        outputs: Any = None
        graph: torch.cuda.CUDAGraph | None = None
        try:
            with torch.autocast("cuda", enabled=torch.is_autocast_enabled("cuda"),
                                dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False), \
                    _device_timestep_frequencies(self.freqs):
                self.stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(self.stream):
                    for _ in range(self.warmup):
                        run()
                torch.cuda.current_stream(device).wait_stream(self.stream)
                after = _rng_states(device)
                if not (torch.equal(after[0], rng[0]) and torch.equal(after[1], rng[1]) and after[2] == rng[2]):
                    return None                     # the forward draws random numbers: never graphed
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool, stream=self.stream, capture_error_mode="thread_local"):
                    outputs = run()
        finally:
            _set_rng_states(rng, device)
            torch.cuda.synchronize(device)
            self.stats["captures"] += 1
            self.stats["capture_seconds"] += time.perf_counter() - start
        if not isinstance(outputs, dict) or not all(isinstance(v, torch.Tensor) for v in outputs.values()):
            return None
        return _Graph(graph, static_latents, static_t, static_args, static_d, outputs)


_RUNNERS: weakref.WeakKeyDictionary[torch.nn.Module, GraphedSsBackbone] = weakref.WeakKeyDictionary()


@contextmanager
def graphed_ss_backbone(ss_generator: Any) -> Iterator[GraphedSsBackbone]:
    """Routes `ss_generator.reverse_fn.backbone`'s calls through its `GraphedSsBackbone` (created on first use and
    kept, with its graphs, for the backbone's lifetime) for one block; the eager forward is restored on exit.

    Raises:
        RuntimeError: If the backbone's `forward` is already overridden on the instance (no nesting).
    """
    backbone = ss_generator.reverse_fn.backbone
    if "forward" in vars(backbone):
        raise RuntimeError("the SAM 3D sparse-structure backbone's forward is already overridden")
    runner = _RUNNERS.get(backbone)
    if runner is None:
        runner = _RUNNERS[backbone] = GraphedSsBackbone(backbone)
    runner.take_stats()                 # the block's counters start at zero, whatever an earlier block left
    backbone.forward = runner
    try:
        yield runner
    finally:
        del backbone.forward
