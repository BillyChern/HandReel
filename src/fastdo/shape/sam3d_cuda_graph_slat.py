# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# This file reproduces one short expression of SAM 3D Objects (https://github.com/facebookresearch/sam-3d-objects),
# a SAM Material of Meta Platforms, Inc.; that expression is distributed under the SAM License (licenses/SAM_LICENSE),
# the rest of the file under Apache-2.0. See THIRD_PARTY_LICENSES.md.
"""CUDA-graph replay of SAM 3D's structured-latent flow transformer blocks.

The structured-latent (SLAT) flow calls its backbone (`slat_generator.reverse_fn.backbone`, a `SLatFlowModelTdfyWrapper`)
76 times per request at 25 Euler steps (guidance doubles the calls for t <= 0.5). Each call runs the sparse-conv IO
blocks and then 24 `ModulatedSparseTransformerCrossBlock`s, `for block in self.blocks: h = block(h, t_emb, cond)`, whose
small kernels make the flow launch-bound (~246k launches per request, the largest kernels ~4 s of 16-23 s).
`graphed_slat_blocks` shadows the backbone's `blocks` attribute for one block of code with a `GraphedSlatBlocks`, so the
forward's loop sees a single "block", the runner, which runs the whole stack. Per call signature (the features' and both
condition tensors' shape, stride, dtype and device, the token layout, the attention backend and the autocast state) it
runs the eager stack twice on a side stream (warm-up), captures one `torch.cuda.CUDAGraph`, and from then on copies the
three inputs into the graph's static buffers, replays it and returns `h.replace(<a clone of the output features>)`. The
input and output blocks, the embedders and everything outside the stack stay eager.

Graphs live for one block of code, i.e. one request's SLAT phase: the coords, hence the token count, layout and
`cu_seqlens`, are fixed within a request and change across them. On exit every graph, the `cu_seqlens` cache and the
graph pool are dropped and, when anything was captured, `torch.cuda.empty_cache()` returns the dropped pool's memory (a
released private pool is never reused by later allocations and would otherwise stay reserved until the allocator next
runs out); its seconds are `stats["release_seconds"]`.

What stays exactly the eager computation:
- the captured kernels are the eager stack's own, with autocast's weight-cast cache off inside warm-up and capture
  (as in `sam3d_cuda_graph`);
- `sparse_scaled_dot_product_attention` (flash_attn backend) builds `cu_seqlens` on the CPU from the layout and copies it
  to the device on every call, which a capture cannot record. Inside warm-up and capture the attention module's
  reference to it is replaced by the same two flash_attn calls with those tensors cached on the device per seqlen
  tuple. Each warm-up call checks every cached tensor against the original expression and the result against the
  original function, bit for bit, and raises on any difference. A call form the replacement does not handle runs the
  original and keeps that signature eager;
- the output is rebuilt as `h.replace(features)`: warm-up checks that the eager stack's output is exactly that, every
  attribute of the SparseTensor and of its backend data other than the features identical (or an equal plain value);
- inputs whose address is not 256-byte aligned run eager (cuBLAS chooses kernels by alignment, static buffers are
  aligned);
- the CPU, CUDA and Python RNG states are saved before the warm-up and restored after the capture; a stack that changed
  any of them is never graphed.
Anything the capture does not support (grad mode, a non-CUDA or grad-requiring input, an attention backend other than
flash_attn, SAM 3D's sparse DEBUG mode, a block other than full attention without RoPE or checkpointing, an unhandled
attention call, a structurally different output) runs the eager stack and is counted in `stats["eager_calls"]`.

This module imports torch and the standard library (and its sibling `sam3d_cuda_graph`): the SAM 3D worker imports it
from its own directory.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

if __package__:
    from .sam3d_cuda_graph import WARMUP_CALLS, _rng_states, _set_rng_states, _static
else:                                   # run from the SAM 3D worker's directory: the sibling file, never `fastdo`
    from sam3d_cuda_graph import (  # type: ignore[no-redef]
        WARMUP_CALLS,
        _rng_states,
        _set_rng_states,
        _static,
    )

__all__ = [
    "ATTENTION_MODULE",
    "FULL_ATTN_MODULE",
    "PATCH_SENTINEL",
    "GraphedSlatBlocks",
    "device_cu_seqlens_attention",
    "graphed_slat_blocks",
]

_SPARSE = "sam3d_objects.model.backbone.tdfy_dit.modules.sparse"
ATTENTION_MODULE = f"{_SPARSE}.attention.modules"     # SparseMultiHeadAttention resolves the function here
FULL_ATTN_MODULE = f"{_SPARSE}.attention.full_attn"   # the function, its ATTN backend, flash_attn and SparseTensor
PATCH_SENTINEL = "__fastdo_slat_graph__"
ALIGNMENT = 256                         # bytes; the caching allocator's blocks are 512-aligned
_STREAMS: dict[torch.device, torch.cuda.Stream] = {}   # one capture stream per device for the process lifetime


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _tensor_sig(x: Any) -> tuple[Any, ...] | None:
    """shape, stride, dtype, device of a capturable input; None otherwise."""
    if (not isinstance(x, torch.Tensor) or not x.is_cuda or x.requires_grad
            or x.data_ptr() % ALIGNMENT != 0):
        return None
    return tuple(x.shape), tuple(x.stride()), x.dtype, x.device


def _autocast_state() -> tuple[bool, torch.dtype]:
    return torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")


def _original_cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    # `full_attn.sparse_scaled_dot_product_attention`'s expression, verbatim.
    return torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlens), dim=0)]).int().to(device)


def _seqlens(s: Any) -> list[int]:
    return [s.layout[i].stop - s.layout[i].start for i in range(s.shape[0])]


def _device_attention(original: Callable[..., Any], full_attn: Any, cache: dict[Any, torch.Tensor],
                      unhandled: list[str]) -> Callable[..., Any]:
    """`original` with its flash_attn `cu_seqlens` looked up in `cache` instead of rebuilt on the CPU (module
    docstring). Outside a capture every call is checked against the original, bit for bit."""

    def cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
        key = (tuple(seqlens), device)
        capturing = _capturing(device)
        if key not in cache:
            if capturing:               # warm-up always fills the cache first; a new length here is a bug
                raise RuntimeError(f"cu_seqlens for {key[0]} first requested inside a CUDA-graph capture")
            cache[key] = _original_cu_seqlens(seqlens, device)
        if not capturing and not torch.equal(cache[key], _original_cu_seqlens(seqlens, device)):
            raise RuntimeError("the cached cu_seqlens differ from SAM 3D's own")
        return cache[key]

    def attention(*args: Any, **kwargs: Any) -> Any:
        sparse, flash = full_attn.SparseTensor, getattr(full_attn, "flash_attn", None)
        q = args[0] if args else None
        if full_attn.ATTN != "flash_attn" or flash is None or kwargs or not isinstance(q, sparse):
            unhandled.append(f"backend {full_attn.ATTN!r}, {len(args)} args, kwargs {sorted(kwargs)}")
            return original(*args, **kwargs)
        if len(args) == 1 and len(q.shape) == 4 and q.shape[1] == 3:               # self-attention, packed qkv
            seqlens = _seqlens(q)
            out = flash.flash_attn_varlen_qkvpacked_func(q.feats, cu_seqlens(seqlens, q.device), max(seqlens))
        elif (len(args) == 2 and len(q.shape) == 3 and isinstance(args[1], torch.Tensor)
              and args[1].dim() == 5 and q.shape[0] == args[1].shape[0]):          # cross-attention, dense kv
            q_seqlen = _seqlens(q)
            n, length, _, heads, channels = args[1].shape
            kv_seqlen = [length] * n
            out = flash.flash_attn_varlen_kvpacked_func(
                q.feats, args[1].reshape(n * length, 2, heads, channels), cu_seqlens(q_seqlen, q.device),
                cu_seqlens(kv_seqlen, q.device), max(q_seqlen), max(kv_seqlen))
        else:
            unhandled.append(f"{len(args)} args of shapes {[tuple(a.shape) for a in args]}")
            return original(*args)
        result = q.replace(out)
        if not _capturing(q.device):
            expected = original(*args)
            if not (torch.equal(result.feats, expected.feats) and result.shape == expected.shape
                    and result.layout == expected.layout):
                raise RuntimeError("the device-cu_seqlens attention differs from SAM 3D's own")
        return result

    setattr(attention, PATCH_SENTINEL, True)
    return attention


@contextmanager
def device_cu_seqlens_attention(cache: dict[Any, torch.Tensor], unhandled: list[str]) -> Iterator[None]:
    """Replaces `SparseMultiHeadAttention`'s `sparse_scaled_dot_product_attention` for one block and restores the
    original by identity on exit. Call forms it does not handle are appended to `unhandled` and run the original.

    Raises:
        RuntimeError: If the replacement is already installed (no nesting).
    """
    attention: Any = importlib.import_module(ATTENTION_MODULE)
    full_attn = importlib.import_module(FULL_ATTN_MODULE)
    saved = attention.sparse_scaled_dot_product_attention
    if getattr(saved, PATCH_SENTINEL, False):
        raise RuntimeError("the device-cu_seqlens SAM 3D attention is already installed")
    attention.sparse_scaled_dot_product_attention = _device_attention(saved, full_attn, cache, unhandled)
    try:
        yield
    finally:
        attention.sparse_scaled_dot_product_attention = saved


def _capturable_block(block: Any) -> bool:
    """Full attention without RoPE or activation checkpointing in both attention layers (what the graph assumes)."""
    if getattr(block, "use_checkpoint", True) is not False:
        return False
    for name in ("self_attn", "cross_attn"):
        attn = getattr(block, name, None)
        if getattr(attn, "attn_mode", None) != "full" or getattr(attn, "use_rope", True) is not False:
            return False
    return True


def _same_value(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if isinstance(a, torch.Tensor | dict) or type(a) is not type(b):
        return False                    # tensors and containers of state must be the same object
    try:
        return bool(a == b)
    except (RuntimeError, TypeError, ValueError):
        return False


def _same_fields(a: Any, b: Any, feats: torch.Tensor, skip: tuple[str, ...] = ()) -> bool:
    fa = {k: v for k, v in vars(a).items() if k not in skip and v is not feats}
    fb = {k: v for k, v in vars(b).items() if k not in skip and v is not feats}
    return fa.keys() == fb.keys() and all(_same_value(fa[k], fb[k]) for k in fa)


def _is_replace_of(out: Any, base: Any) -> bool:
    """Whether `out` is exactly `base.replace(out.feats)`: what the replay returns for the eager output."""
    if type(out) is not type(base) or not hasattr(out, "data"):
        return False
    expected = base.replace(out.feats)
    return (_same_fields(out, expected, out.feats, skip=("data",))
            and _same_fields(out.data, expected.data, out.feats))


@dataclass
class _SlatGraph:
    graph: torch.cuda.CUDAGraph
    feats: torch.Tensor
    t_emb: torch.Tensor
    cond: torch.Tensor
    out: torch.Tensor

    def replay(self, h: Any, t_emb: torch.Tensor, cond: torch.Tensor) -> Any:
        self.feats.copy_(h.feats)
        self.t_emb.copy_(t_emb)
        self.cond.copy_(cond)
        self.graph.replay()
        return h.replace(self.out.clone())


class GraphedSlatBlocks:
    """The transformer-block stack, captured once per call signature and replayed (module docstring).

    Iterating it yields itself once, so `for block in self.blocks: h = block(h, t_emb, cond)` runs the whole stack
    in one call.

    Args:
        blocks: The backbone's `blocks` ModuleList, run eagerly in order.
        warmup: Eager stack runs on the capture stream before each capture (at least 1: warm-up fills the
            `cu_seqlens` cache the capture reads).

    Attributes:
        stats: Counters since the last `take_stats`: `captures`, `capture_seconds` (warm-up plus capture,
            synchronised), `replays`, `eager_calls` and `release_seconds` (dropping the graphs, `close`).
    """

    def __init__(self, blocks: Any, warmup: int = WARMUP_CALLS) -> None:
        if warmup < 1:
            raise ValueError(f"warmup must be >= 1 (it fills the cu_seqlens cache), got {warmup}")
        self.blocks = blocks
        self.warmup = warmup
        self.capturable = all(_capturable_block(block) for block in blocks)
        self.graphs: dict[Any, _SlatGraph | None] = {}     # None: this signature stays eager
        self.cu_seqlens: dict[Any, torch.Tensor] = {}
        self.pool: Any = None
        self.stats = self._zero_stats()

    @staticmethod
    def _zero_stats() -> dict[str, Any]:
        return {"captures": 0, "capture_seconds": 0.0, "replays": 0, "eager_calls": 0, "release_seconds": 0.0}

    def take_stats(self) -> dict[str, Any]:
        """The counters since the previous call, then resets them."""
        stats, self.stats = self.stats, self._zero_stats()
        return stats

    def __iter__(self) -> Iterator[GraphedSlatBlocks]:
        yield self

    def eager(self, h: Any, t_emb: torch.Tensor, cond: torch.Tensor) -> Any:
        for block in self.blocks:
            h = block(h, t_emb, cond)
        return h

    def signature(self, h: Any, t_emb: Any, cond: Any) -> Any:
        """The graph key of one call, or None when the call is not capturable."""
        if not self.capturable or torch.is_grad_enabled():
            return None
        sigs = (_tensor_sig(getattr(h, "feats", None)), _tensor_sig(t_emb), _tensor_sig(cond))
        if any(s is None for s in sigs) or len({s[3] for s in sigs if s is not None}) != 1:
            return None
        full_attn = importlib.import_module(FULL_ATTN_MODULE)
        basic = sys.modules.get(getattr(full_attn.SparseTensor, "__module__", ""))
        if (full_attn.ATTN != "flash_attn" or getattr(full_attn, "DEBUG", True) or getattr(basic, "DEBUG", True)
                or not isinstance(h, full_attn.SparseTensor)):
            return None
        layout = tuple((s.start, s.stop, s.step) for s in h.layout)
        return sigs, tuple(h.shape), layout, _autocast_state()

    def __call__(self, h: Any, t_emb: Any, cond: Any) -> Any:
        key = self.signature(h, t_emb, cond)
        if key is not None and key not in self.graphs:
            self.graphs[key] = self._capture(h, t_emb, cond)
        graph = None if key is None else self.graphs[key]
        if graph is None:
            self.stats["eager_calls"] += 1
            return self.eager(h, t_emb, cond)
        self.stats["replays"] += 1
        return graph.replay(h, t_emb, cond)

    def _capture(self, h: Any, t_emb: torch.Tensor, cond: torch.Tensor) -> _SlatGraph | None:
        start = time.perf_counter()
        device = h.feats.device
        stream = _STREAMS.get(device)
        if stream is None:
            stream = _STREAMS[device] = torch.cuda.Stream(device)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        static_h = h.replace(_static(h.feats))
        static_t, static_cond = _static(t_emb), _static(cond)

        def run() -> Any:
            return self.eager(static_h, static_t, static_cond)

        rng = _rng_states(device)
        unhandled: list[str] = []
        out: Any = None
        graph: torch.cuda.CUDAGraph | None = None
        try:
            with torch.autocast("cuda", enabled=torch.is_autocast_enabled("cuda"),
                                dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False), \
                    device_cu_seqlens_attention(self.cu_seqlens, unhandled):
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup):
                        out = run()
                torch.cuda.current_stream(device).wait_stream(stream)
                after = _rng_states(device)
                if (unhandled or not _is_replace_of(out, static_h) or not torch.equal(after[0], rng[0])
                        or not torch.equal(after[1], rng[1]) or after[2] != rng[2]):
                    return None             # never graphed: this signature stays eager
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool, stream=stream, capture_error_mode="thread_local"):
                    out = run()
        finally:
            _set_rng_states(rng, device)
            torch.cuda.synchronize(device)
            self.stats["captures"] += 1
            self.stats["capture_seconds"] += time.perf_counter() - start
        return _SlatGraph(graph, static_h.feats, static_t, static_cond, out.feats)

    def close(self) -> None:
        """Drops every graph, the cu_seqlens cache and the pool; releases the pool's memory if anything was captured."""
        captured = any(graph is not None for graph in self.graphs.values())
        self.graphs.clear()
        self.cu_seqlens.clear()
        self.pool = None
        if captured:
            start = time.perf_counter()
            torch.cuda.empty_cache()
            self.stats["release_seconds"] += time.perf_counter() - start


@contextmanager
def graphed_slat_blocks(slat_generator: Any) -> Iterator[GraphedSlatBlocks]:
    """Routes `slat_generator.reverse_fn.backbone`'s transformer blocks through a new `GraphedSlatBlocks` for one
    block of code: the runner shadows the backbone's `blocks` in its instance `__dict__` (its registered ModuleList,
    parameters and state dict are untouched) and is removed on exit, then closed (every graph dropped).

    Raises:
        RuntimeError: If the backbone's `blocks` is already shadowed (no nesting).
    """
    model = slat_generator.reverse_fn.backbone
    if "blocks" in vars(model):
        raise RuntimeError("the SAM 3D structured-latent backbone's blocks are already overridden")
    runner = GraphedSlatBlocks(model.blocks)
    vars(model)["blocks"] = runner
    try:
        yield runner
    finally:
        del vars(model)["blocks"]
        runner.close()
