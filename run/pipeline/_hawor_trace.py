"""Diagnostic trace for the batched HaWoR driver; OFF unless `FASTDO_HAWOR_TRACE=<path.jsonl>` is set.

Loaded by path from `_hawor_batched_driver.py` (the driver imports no `fastdo` package). With the variable unset every
function here returns immediately and nothing is hooked, so the production path is unchanged.

Levels (`FASTDO_HAWOR_TRACE_LEVEL`, default 1):
  1  the process's numerics flags at load and after warm-up; per detector chunk a sha1 of the raw detections
     (already on the host in `detect_clip`, so no extra device sync); per clip group sha1s of the backbone features,
     the heads outputs and the MANO outputs.
  2  level 1 plus a forward hook on every top-level layer of the YOLO detector and every block of the HaWoR backbone,
     recording a sha1 of each call's output (forces a device sync per layer call: diagnosis only).
Each record is one JSON line: {"seq", "req", "kind", ...}. `req` counts the requests this process has served.

Injection (diagnosis only): `FASTDO_HAWOR_NMS_TRUNCATE="<clip_id>|<chunk>|<first_empty>"` makes the detector's NMS return
no detections for images `first_empty..` of predict call `chunk` (0-based) of that clip -- exactly what ultralytics'
NMS wall-clock limit leaves when it breaks after image `first_empty - 1` -- to test that mechanism against a recorded
output. The NMS itself is not changed; its outputs for those images are replaced by empty tensors.

Runtime licences (nothing copied into this repository): HaWoR CC BY-NC-ND 4.0, ultralytics AGPL-3.0, MANO licence.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import torch

_PATH = os.environ.get("FASTDO_HAWOR_TRACE", "")
_LEVEL = int(os.environ.get("FASTDO_HAWOR_TRACE_LEVEL", "1") or 1)
_TRUNCATE = os.environ.get("FASTDO_HAWOR_NMS_TRUNCATE", "")
_state: dict[str, Any] = {"seq": 0, "req": 0, "calls": {}, "clip": None, "chunk": -1, "wrapped": False}


def enabled(level: int = 1) -> bool:
    return bool(_PATH) and _LEVEL >= level


def new_request() -> None:
    if enabled():
        _state["req"] += 1
        _state["calls"] = {}


def current_clip(clip_id: str) -> None:
    """The driver is about to run the detector on `clip_id` (predict calls counted from 0)."""
    _state["clip"], _state["chunk"] = clip_id, -1
    if _TRUNCATE and not _state["wrapped"]:
        _wrap_nms()


def _wrap_nms() -> None:
    from ultralytics.utils import nms

    clip_id, chunk, first = _TRUNCATE.split("|")
    original = nms.non_max_suppression

    def truncating(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        if _state["clip"] is not None:
            _state["chunk"] += 1
            if _state["clip"] == clip_id and _state["chunk"] == int(chunk):
                out = [o if i < int(first) else o[:0] for i, o in enumerate(out)]
                if _PATH:
                    _write("nms_truncated", clip=clip_id, chunk=int(chunk), first_empty=int(first))
        return out

    nms.non_max_suppression = truncating
    _state["wrapped"] = True


def _write(kind: str, **fields: Any) -> None:
    _state["seq"] += 1
    with open(_PATH, "a") as f:
        f.write(json.dumps({"seq": _state["seq"], "req": _state["req"], "kind": kind, **fields}) + "\n")


def digest(x: Any) -> str:
    """sha1 over the bytes of a tensor / array / nested sequence of them (dtype and shape included)."""
    h = hashlib.sha1()

    def add(v: Any) -> None:
        if isinstance(v, torch.Tensor):
            t = v.detach()
            h.update(f"{t.dtype}{tuple(t.shape)}".encode())
            h.update(t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(f"{v.dtype}{v.shape}".encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, (list, tuple)):
            for item in v:
                add(item)
        elif isinstance(v, dict):
            for k in sorted(v):
                h.update(str(k).encode())
                add(v[k])
        elif hasattr(v, "__dataclass_fields__"):
            for k in v.__dataclass_fields__:
                h.update(k.encode())
                add(getattr(v, k))
        else:
            h.update(repr(v).encode())

    add(x)
    return h.hexdigest()[:16]


def flags(where: str) -> None:
    """The process's numerics flags (cuDNN / TF32 / deterministic algorithms / cuBLAS workspace)."""
    if not enabled():
        return
    _write("flags", where=where, cudnn_enabled=torch.backends.cudnn.enabled,
           cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
           cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
           matmul_precision=torch.get_float32_matmul_precision(),
           bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
           fp16_reduced=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
           deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
           cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           cudnn_version=torch.backends.cudnn.version(), torch=torch.__version__,
           sdp_flash=torch.backends.cuda.flash_sdp_enabled(), sdp_mem=torch.backends.cuda.mem_efficient_sdp_enabled())


def record(kind: str, value: Any, **fields: Any) -> None:
    if enabled():
        _write(kind, h=digest(value), **fields)


def _hook(name: str):
    def fn(module: torch.nn.Module, args: Any, output: Any) -> None:
        n = _state["calls"].get(name, 0)
        _state["calls"][name] = n + 1
        _write("layer", module=name, call=n, h=digest(output))
    return fn


def install_hooks(models: Any) -> None:
    """Level 2: forward hooks on the YOLO layers and the HaWoR backbone blocks (diagnosis only)."""
    if not enabled(2):
        return
    try:
        layers = models.yolo.model.model                        # ultralytics DetectionModel.model (nn.Sequential)
        for i, layer in enumerate(layers):
            layer.register_forward_hook(_hook(f"yolo.{i}.{type(layer).__name__}"))
    except AttributeError as exc:
        _write("hook_error", target="yolo", error=str(exc))
    backbone = getattr(models.hawor, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if blocks is not None:
        for i, block in enumerate(blocks):
            block.register_forward_hook(_hook(f"backbone.block{i}"))
    else:
        _write("hook_error", target="backbone", error="no .blocks")


if __name__ == "__main__":
    import sys

    print("usage: not a command. Diagnostic trace of the batched HaWoR driver, loaded by path from "
          "run/pipeline/_hawor_batched_driver.py; set FASTDO_HAWOR_TRACE=<path.jsonl> to enable it.")
    raise SystemExit(0 if sys.argv[1:] in ([], ["-h"], ["--help"]) else 2)
