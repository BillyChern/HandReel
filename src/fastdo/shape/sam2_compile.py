"""Compile modes of SAM 2.1's VOS-optimised video predictor: the `bench.mask_compile` knob.

`sam2.build_sam.build_sam2_video_predictor_hf(model_id, vos_optimized=True)` builds `SAM2VideoPredictorVOS`, whose
constructor wraps five forwards (`VOS_COMPILED`: image encoder, memory encoder, memory attention, prompt encoder,
mask decoder) in `torch.compile(..., mode="max-autotune")`. That upstream mode crashes on our propagation loop
(inductor's CUDA-graph trees overwrite a memory-attention output the loop still holds), so the knob offers:

  none          the eager production predictor (`vos_optimized` not passed): the default, byte-identical to before
  vos_nocg      every compile call made with mode "max-autotune-no-cudagraphs": fused, autotuned kernels, eager
                launches (mask stage -18%, 145 s cold compile)
  vos_cg_clone  upstream's mode, each compiled callable wrapped in `CudagraphStep`: a
                `torch.compiler.cudagraph_mark_step_begin()` before every call and its tensor outputs cloned, the
                fix the error names (-22%, 41 s warm compile)
  vos_det       inductor's default mode (no GEMM autotuning) with `triton.autotune_pointwise` off, which in
                torch 2.9 also pins every reduction to ONE config: no kernel is chosen by timing, so the compiled
                kernels -- and the masks -- should not depend on the process (vos_nocg's max-autotune does)

The vos modes are NOT bit-exact against `none` (per-clip mask IoU 0.997, worst frame 0.94); they need a pose
gate before any promotion.

`compile_mode_patch(mode)` rewrites `torch.compile` only for the duration of the predictor's build -- the compile
CALLS happen in the constructor, compilation itself is lazy (first forward) -- and restores the original on exit,
also when the build raises. Only callables defined in the `sam2` package are rewritten; any other compile call made
meanwhile passes through untouched. The context yields the qualnames it rewrote, so the loader can refuse a build in
which the knob reached nothing (an upstream change would otherwise leave upstream's crashing mode in place).

Compile cost is per predictor, i.e. per process and per concurrent mask slot. Inductor caches compiled kernels under
`TORCHINDUCTOR_CACHE_DIR` (default `/tmp/torchinductor_<user>`, shared by every process of the user); set it per
process (e.g. `TORCHINDUCTOR_CACHE_DIR=<run dir>/inductor`) to keep concurrent runs from sharing or racing on one
cache, or share one deliberately to pay the autotuning once.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Callable, Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["DET_OPTIONS", "MASK_COMPILE_MODES", "NOCG_MODE", "VOS_COMPILED", "CudagraphStep", "check_mask_compile",
           "compile_mode_patch"]

MASK_COMPILE_MODES: tuple[str, ...] = ("none", "vos_nocg", "vos_cg_clone", "vos_det")
NOCG_MODE = "max-autotune-no-cudagraphs"
# torch.compile takes `mode` OR `options`, never both; these are compile-time inductor settings, so they bind the
# compiled callables only (nothing global is toggled, unlike `torch.use_deterministic_algorithms`).
DET_OPTIONS: dict[str, bool] = {"triton.autotune_pointwise": False, "max_autotune": False}
# The forwards `SAM2VideoPredictorVOS` compiles (sam2's `_compile_all_components` + `compile_image_encoder`).
VOS_COMPILED: frozenset[str] = frozenset({"ImageEncoder.forward", "MemoryEncoder.forward", "MemoryAttention.forward",
                                          "PromptEncoder.forward", "MaskDecoder.forward"})


def check_mask_compile(value: Any) -> str:
    """`value` as a `MASK_COMPILE_MODES` entry.

    Raises:
        ValueError: On anything else (including a non-string).
    """
    if not isinstance(value, str) or value not in MASK_COMPILE_MODES:
        raise ValueError(f"bench.mask_compile must be one of {MASK_COMPILE_MODES}, got {value!r}")
    return value


def _clone_tensors(out: Any) -> Any:
    return torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: t.clone(), out)


class CudagraphStep:
    """Calls a CUDA-graph-compiled callable as one graph step and returns clones of its tensor outputs.

    `torch.compiler.cudagraph_mark_step_begin()` before the call tells inductor's CUDA-graph trees a new step starts
    (so replaying does not wait for the previous outputs to die); the clones own their memory, so the next replay
    cannot overwrite a tensor the caller still holds. Non-tensor leaves pass through; containers are rebuilt.
    """

    def __init__(self, compiled: Callable[..., Any],
                 mark_step: Callable[[], None] = torch.compiler.cudagraph_mark_step_begin) -> None:
        self.compiled, self._mark_step = compiled, mark_step
        functools.update_wrapper(self, compiled)          # __name__/__qualname__/__doc__, __wrapped__ = compiled

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._mark_step()
        return _clone_tensors(self.compiled(*args, **kwargs))

    def __repr__(self) -> str:
        return f"CudagraphStep({self.compiled!r})"


def _is_sam2(fn: Any) -> bool:
    module = getattr(fn, "__module__", None) or ""
    return module == "sam2" or module.startswith("sam2.")


@contextlib.contextmanager
def compile_mode_patch(mode: str, module: Any = torch) -> Iterator[list[str]]:
    """Within the context, `module.compile` applies `mode` to sam2's compile calls; the original is restored on exit.

    Args:
        mode: `vos_nocg`, `vos_cg_clone` or `vos_det` (`none` patches nothing: use the eager build instead).
        module: The object whose `compile` attribute is patched; `torch` except in tests.

    Yields:
        The qualnames of the callables rewritten so far (a live list, complete when the context exits).

    Raises:
        ValueError: On `none` or an unknown mode.
    """
    if check_mask_compile(mode) == "none":
        raise ValueError("compile_mode_patch needs a vos mode; mask_compile 'none' builds the eager predictor")
    original = module.compile
    rewritten: list[str] = []

    def compile_(fn: Any = None, *args: Any, **kwargs: Any) -> Any:
        if fn is None or not _is_sam2(fn):               # decorator form or another package: untouched
            return original(fn, *args, **kwargs)
        rewritten.append(str(getattr(fn, "__qualname__", repr(fn))))
        if mode == "vos_nocg":
            return original(fn, *args, **{**kwargs, "mode": NOCG_MODE})
        if mode == "vos_det":
            rest = {k: v for k, v in kwargs.items() if k != "mode"}
            return original(fn, *args, **{**rest, "options": dict(DET_OPTIONS)})
        return CudagraphStep(original(fn, *args, **kwargs))

    module.compile = compile_
    try:
        yield rewritten
    finally:
        module.compile = original
        logger.info("mask_compile %s rewrote %d torch.compile calls: %s", mode, len(rewritten), sorted(rewritten))
