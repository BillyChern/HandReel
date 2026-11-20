"""Deterministic spconv kernel selection for SAM 3D's sparse convolutions.

SAM 3D's 3x3x3 fp16 SubM convs run spconv's MaskImplicitGemm. With `spconv.constants.SPCONV_CPP_GEMM = True` (spconv
2.3.6), `spconv.pytorch.ops.implicit_gemm` asks the C++ tuner `CONV_CPP`, which times every candidate kernel on the first
call per key (dtypes, out and in channels, GPU arch; no point count) and keeps the fastest for the life of the process.
Kernel families are within 0.1-5% of each other, so the winner depends on the first request's point count and on timing
noise, and kernels with different tile shapes change the output bytes (det49: 3 states per clip over 6 fresh stock
processes and 4 request orders).

`pin_spconv_kernels` makes the choice a pure function of the candidate list. It is installed once per process, before
the pipeline loads, and never per request (the kernel cache lives for the process):
- P4 `spconv.pytorch.ops.SPCONV_CPP_GEMM = False` (read at call time): implicit gemm takes spconv's Python path;
- P5 the instance attribute `spconv.algo.CONV.tune_and_cache`: caches `fixed_rule` of the C++ candidate set under
  spconv's own key, without executing or timing a kernel. Forward only; other op types raise.
It refuses `SPCONV_ALGO=native`: SAM 3D then builds `ConvAlgo.Native` convs, whose timing tuner the pin does not cover.
Gate (12 fresh processes x 4 orders x 3 clips under CPU load): one mesh.obj/sdf.npz/samples.npz state per clip; first
request -2.84 s against stock, later requests within noise; peak reserved 20.63 GiB, as stock.

Standard library only at import: the SAM 3D worker imports it from its own directory; spconv and cumm load lazily.
"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Callable
from typing import Any

__all__ = ["PATCH_SENTINEL", "PIN_RULE_VERSION", "SUPPORTED_SPCONV", "UNPINNED_SPCONV_ALGO", "candidates", "fixed_rule",
           "pin_spconv_kernels", "pinned_kernels"]

PIN_RULE_VERSION = 2                    # bump with any change of `candidates` or `fixed_rule`
PATCH_SENTINEL = "__fastdo_det__"       # the attribute sam3d_determinism also sets on its wrappers
SUPPORTED_SPCONV = frozenset({"2.3.6"})
_OPS_ATTRS = ("SPCONV_CPP_GEMM", "implicit_gemm", "CONV")
_CONV_ATTRS = ("kc_forward_cache", "get_all_available", "lock", "get_tuned_algo")
UNPINNED_SPCONV_ALGO = "native"         # SAM 3D's SPCONV_ALGO value that routes its convs past the pin


def candidates(conv: Any, inp: Any, weight: Any, output: Any, layout_i: Any, layout_w: Any, layout_o: Any, arch: Any,
               op_type: Any, mask_width: int, use_tf32: bool, bias: Any, scale: Any,
               fp32_accum: bool | None = None) -> list[Any]:
    """The C++ tuner's candidate set: `conv.get_all_available(..., fp32_accum)` with None read as False (the C++
    default), without "Simt" kernels for fp16 input and weight on arch >= 7.0 (the Python version compares a str with
    an enum, so it never drops them)."""
    tv = importlib.import_module("cumm.tensorview")
    avail = conv.get_all_available(inp, weight, output, layout_i, layout_w, layout_o, arch, op_type, mask_width,
                                   fp32_accum=bool(fp32_accum), use_tf32=use_tf32, bias=bias, scale=scale)
    if inp.dtype == tv.float16 and weight.dtype == tv.float16 and tuple(arch) >= (7, 0):
        avail = [desp for desp in avail if desp.algo != "Simt"]
    return list(avail)


def fixed_rule(avail: list[Any]) -> Any:
    """The greatest `(min_arch, accumulator dtype == input dtype, tile m*n*k, name)`: the newest-architecture kernel,
    then the input-dtype accumulator, then the largest tile; the name breaks ties. Independent of the list order."""

    def rank(desp: Any) -> tuple[tuple[int, ...], bool, int, str]:
        return (tuple(desp.min_arch), int(desp.dacc) == int(desp.dtype_input), math.prod(tuple(desp.tile_shape)[:3]),
                str(desp))

    return max(avail, key=rank)


def pin_spconv_kernels() -> Callable[[], None]:
    """Installs P4 and P5 (module docstring) and returns `restore`; a second call while installed is a no-op.

    Raises:
        RuntimeError: Before any change, on an unsupported spconv version, a missing patch point, a non-empty
            `CONV.kc_forward_cache` (kernels already chosen by timing) or `SPCONV_ALGO=native`.
    """
    modules: list[Any] = [importlib.import_module(n) for n in ("spconv", "spconv.pytorch.ops", "spconv.algo")]
    spconv, ops, algo = modules
    conv: Any = getattr(algo, "CONV", None)
    current = getattr(conv, "__dict__", {}).get("tune_and_cache")
    if getattr(current, PATCH_SENTINEL, False) and getattr(ops, "SPCONV_CPP_GEMM", None) is False:
        return lambda: None
    problems = [] if getattr(spconv, "__version__", None) in SUPPORTED_SPCONV else [
        f"spconv {getattr(spconv, '__version__', None)!r} is not one of {sorted(SUPPORTED_SPCONV)}"]
    problems += [f"spconv.pytorch.ops has no {name}" for name in _OPS_ATTRS if not hasattr(ops, name)]
    problems += [f"spconv.algo.CONV has no {name}" for name in _CONV_ATTRS if not hasattr(conv, name)]
    problems += [] if hasattr(algo, "BestConvAlgoByProfile") else ["spconv.algo has no BestConvAlgoByProfile"]
    if hasattr(ops, "CONV") and ops.CONV is not conv:
        problems.append("spconv.pytorch.ops.CONV is not spconv.algo.CONV")
    if hasattr(conv, "kc_forward_cache") and conv.kc_forward_cache:
        problems.append(f"spconv.algo.CONV already cached {len(conv.kc_forward_cache)} timed kernels")
    if os.environ.get("SPCONV_ALGO") == UNPINNED_SPCONV_ALGO:
        problems.append("SPCONV_ALGO=native makes SAM 3D build ConvAlgo.Native convs, whose timing tuner is not pinned")
    if problems:
        raise RuntimeError("cannot pin spconv kernels: " + "; ".join(problems))
    conv_op_type = importlib.import_module("cumm.conv.bases").ConvOpType
    tv = importlib.import_module("cumm.tensorview")

    def pinned(op_type: Any, inp: Any, weight: Any, output: Any, layout_i: Any, layout_w: Any, layout_o: Any,
               arch: Any, mask: Any, mask_argsort: Any, indices: Any, reverse_mask: bool, mask_filter: int = 0xffffffff,
               mask_width: int = -1, mask_output: Any = None, alpha: float = 1.0, beta: float = 0.0, stream: int = 0,
               fp32_accum: bool | None = None, use_tf32: bool = True, bias: Any = None,
               scale: Any = None) -> tuple[Any, float]:
        if op_type != conv_op_type.kForward:
            raise NotImplementedError(f"the spconv kernel pin covers the forward pass only, got {op_type}")
        avail = candidates(conv, inp, weight, output, layout_i, layout_w, layout_o, arch, op_type, mask_width,
                           use_tf32, tv.Tensor() if bias is None else bias, tv.Tensor() if scale is None else scale,
                           fp32_accum)
        if not avail:
            raise ValueError(f"no spconv kernel for {op_type} (dtypes {inp.dtype}/{weight.dtype}, arch {arch})")
        kv = weight.dim(1) if weight.ndim == 3 else math.prod(weight.shape[1:-1])
        res = algo.BestConvAlgoByProfile(fixed_rule(avail), arch, splitk=1)
        key = (inp.dtype, weight.dtype, output.dtype, output.dim(1), inp.dim(1), arch[0], arch[1], -1, kv > 32)
        with conv.lock:
            conv.kc_forward_cache[key] = res
        return res, 0.0

    setattr(pinned, PATCH_SENTINEL, True)
    saved_flag = ops.SPCONV_CPP_GEMM
    done: list[bool] = []
    ops.SPCONV_CPP_GEMM = False
    conv.tune_and_cache = pinned

    def restore() -> None:
        if done:
            return
        done.append(True)
        ops.SPCONV_CPP_GEMM = saved_flag
        if vars(conv).get("tune_and_cache") is pinned:
            del conv.tune_and_cache

    return restore


def pinned_kernels() -> dict[str, str]:
    """`{"i<dtype>w<dtype>o<dtype>:k<out>:c<in>:sm<arch>": kernel name}` from spconv's Python forward kernel cache."""
    conv = importlib.import_module("spconv.algo").CONV
    with conv.lock:
        items = list(conv.kc_forward_cache.items())
    return dict(sorted((f"i{k[0]}w{k[1]}o{k[2]}:k{k[3]}:c{k[4]}:sm{k[5]}{k[6]}", str(res.algo_desp)) for k, res in items))
