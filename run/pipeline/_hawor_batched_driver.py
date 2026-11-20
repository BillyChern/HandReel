"""Batched HaWoR hand-mesh recovery driver. Runs in the baseline's hawor env, never in the uv env.

    <hawor_python> _hawor_batched_driver.py --request REQ --response RESP
    <hawor_python> _hawor_batched_driver.py --serve
    <hawor_python> _hawor_batched_driver.py --self-check --response RESP
    <hawor_python> _hawor_batched_driver.py --import-check

Launch it with the hawor env's own python binary and cwd = the HaWoR root, never through `conda run` (the env's
activation sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`). The request/response contract and exit codes are documented in
`fastdo.hands.hawor_batched`. This file imports no `fastdo` package: `load_shared_modules` loads the shared modules
(`src/fastdo/hands/{hawor_boxes,hawor_plan,hawor_core,hand_gaps,hand_npz}.py`, `src/fastdo/bench/gpu_budget.py`) by
path under `fastdo_shared_<stem>`, and `_hawor_driver_models.py` holds everything that imports HaWoR.

Per call: load once, warm up, then stream clips. Each clip is read, detected, box-filtered, planned and cropped on its
own (a failure there fails that clip only); its crops join a feature FIFO shared by the clip group. A group closes at
`clips_per_group` clips or before `max_crops_per_group` would be exceeded; its heads, gap fill, MANO and writes then run
and `response.json` is rewritten. A failure in a group's backbone, heads or MANO fails every clip of the group.

Warm mode (`--serve`): one process serves many requests. It reads one request.json path per stdin line, writes
that request's response to the sibling `response.json`, and prints one JSON line `{"request", "code", "serving"}` to
stdout when the request is finished; every other stdout write is redirected to stderr, so stdout carries only these
lines. The first request loads and warms up the models exactly as a fresh call does and reports those seconds; later
requests reuse them and report `load_seconds = warmup_seconds = 0`, and their `peak_cuda_gb` is measured from the
request's start (resident weights included). Each request runs today's code path (`_Call.run`) on a fresh `_Call`,
after the torch (CPU and CUDA), numpy and `random` RNG states are restored to their values right after warm-up -- the
state a fresh process starts its clips from -- and is followed by `torch.cuda.empty_cache()`, so only the weights stay
resident. A load failure answers code 2 with `serving: false` and ends the process; EOF ends it with exit code 0.

Batch invariance (opt-in `options.batch_invariant`, absent = false): a window's hands must not depend on its batch
mates. When on, every backbone, heads and MANO call runs at one fixed shape -- `FeatureBank(pad_to_batch=True)`,
`run_heads(pad_last=True)`, and MANO in chunks of exactly `options.mano_rows_per_batch` rows (default 512), the last one
padded by repeating its first row -- and padded outputs are dropped. The response then carries `"batch_invariant": true`
and `"mano_rows_per_batch"`. When off, the calls and the response are exactly those of before.

Timing independence (opt-in `options.deterministic`, absent = false): ultralytics' NMS gives up after a
wall-clock budget and returns no detections for the rest of a predict call; when on, the driver removes that budget
(`_hawor_driver_models.disable_nms_time_limit`) after loading and on every warm request, and the response carries
`"deterministic": true`. When off, the calls and the response are exactly those of before.

Runtime licences (nothing copied into this repository): HaWoR CC BY-NC-ND 4.0, ultralytics AGPL-3.0, MANO licence.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import numpy as np
import torch

logger = logging.getLogger("hawor_batched_driver")

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_FILES: dict[str, str] = {
    "hawor_boxes": "src/fastdo/hands/hawor_boxes.py",
    "hawor_plan": "src/fastdo/hands/hawor_plan.py",
    "hawor_core": "src/fastdo/hands/hawor_core.py",
    "hand_gaps": "src/fastdo/hands/hand_gaps.py",
    "hand_npz": "src/fastdo/hands/hand_npz.py",
    "gpu_budget": "src/fastdo/bench/gpu_budget.py",
}
MODELS_FILE = "run/pipeline/_hawor_driver_models.py"
TRACE_FILE = "run/pipeline/_hawor_trace.py"                        # diagnostic trace, off unless its env var is set
SIDES = ("left", "right")
TIMED = ("read", "detect", "box_filter", "crop", "backbone", "heads", "gap_fill", "mano", "write")
DEFAULT_MANO_ROWS_PER_BATCH = 512                                  # MANO chunk rows when batch_invariant is on


def load_module(name: str, path: Path) -> ModuleType:
    """Loads a file as module `name`, registered in `sys.modules` before execution (frozen dataclasses need it)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_shared_modules(repo_root: Path) -> dict[str, ModuleType]:
    """The six shared modules, keyed by file stem, loaded as `fastdo_shared_<stem>`."""
    return {stem: load_module(f"fastdo_shared_{stem}", Path(repo_root) / rel) for stem, rel in SHARED_FILES.items()}


_TRACE: ModuleType | None = None                   # loaded by main() only when FASTDO_HAWOR_TRACE is set


def _trace(fn: str, *args: Any, **kwargs: Any) -> None:
    """Calls `_hawor_trace.<fn>` when the diagnostic trace is on; nothing otherwise."""
    if _TRACE is not None:
        getattr(_TRACE, fn)(*args, **kwargs)


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


class _Clip:
    """Per-clip bookkeeping during a call (mutable state, not configuration)."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.clip_id = str(spec["clip_id"])
        self.spec = spec
        self.status, self.error, self.n_frames = "pending", "", 0
        self.seconds = {key: 0.0 for key in TIMED}
        self.frames: np.ndarray | None = None
        self.frame_idx = np.zeros(0, np.int64)
        self.fps = 0.0
        self.geom: Any = None
        self.boxes: dict[str, np.ndarray] = {}
        self.slots: list[Any] = []
        self.n_crops = 0
        self.rows: dict[str, dict[str, np.ndarray]] = {}
        self.filled: dict[str, np.ndarray] = {}
        self.mesh: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fail(self, stage: str, exc: BaseException) -> None:
        self.status, self.error = "failed", f"{stage}: {type(exc).__name__}: {exc}"
        self.frames = None
        logger.warning("clip %s failed at %s: %s", self.clip_id, stage, exc)

    def result(self) -> dict[str, Any]:
        seconds = dict(self.seconds) | {"total": float(sum(self.seconds.values()))}
        return {"clip_id": self.clip_id, "status": self.status, "error": self.error, "n_frames": self.n_frames,
                "seconds": seconds}


def _read_frames(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        frames, frame_idx, fps = np.asarray(data["frames"]), np.asarray(data["frame_idx"]), float(data["fps"])
    if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"{path}: frames must be non-empty (N,H,W,3) uint8, got {frames.shape} {frames.dtype}")
    if frame_idx.shape != (frames.shape[0],) or (frame_idx.size > 1 and not bool((np.diff(frame_idx) > 0).all())):
        raise ValueError(f"{path}: frame_idx must be ({frames.shape[0]},) and strictly increasing")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: fps must be positive, got {fps}")
    return frames, frame_idx.astype(np.int64), fps


class _Call:
    """One request: models loaded once, clips streamed into groups."""

    def __init__(self, request: Mapping[str, Any], response: Path, shared: Mapping[str, ModuleType], api: Any,
                 device: str) -> None:
        self.opts, self.response, self.shared, self.api, self.device = request["options"], response, shared, api, device
        # absent from a default request (only written when on), so an old request.json reads as off.
        self.batch_invariant = bool(self.opts.get("batch_invariant", False))
        self.mano_rows = int(self.opts.get("mano_rows_per_batch", DEFAULT_MANO_ROWS_PER_BATCH))
        self.body: dict[str, Any] = {"version": 1, "error": "", "load_seconds": 0.0, "warmup_seconds": 0.0,
                                     "compute_seconds": 0.0, "peak_cuda_gb": 0.0, "clips": []}
        if self.batch_invariant:                        # recorded only when on: the off response is as before
            self.body |= {"batch_invariant": True, "mano_rows_per_batch": self.mano_rows}
        # absent = off. On, YOLO's NMS runs without its wall-clock limit (`disable_nms_time_limit`, applied in
        # `load` / `adopt`), so a stalled driver can no longer drop the detections of a chunk's trailing frames.
        self.deterministic = bool(self.opts.get("deterministic", False))
        if self.deterministic:
            self.body |= {"deterministic": True}
        self.finished: list[_Clip] = []
        self.compute_start: float | None = None
        self.models: Any = None

    def write_response(self) -> None:
        if self.compute_start is not None:
            self.body["compute_seconds"] = time.perf_counter() - self.compute_start
        if torch.device(self.device).type == "cuda" and torch.cuda.is_available():
            self.body["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 2**30
        self.body["clips"] = [clip.result() for clip in self.finished]
        tmp = self.response.with_name(self.response.name + ".tmp")
        tmp.write_text(json.dumps(self.body, indent=1))
        os.replace(tmp, self.response)

    def load(self, hawor_root: Path) -> bool:
        start = time.perf_counter()
        try:
            self.models = self.api.load_models(hawor_root, self.shared)
            if self.deterministic:                                 # before the warm-up's first predict
                self.api.disable_nms_time_limit()
            _sync(self.device)
            self.body["load_seconds"] = time.perf_counter() - start
            _trace("flags", "after_load")
            _trace("install_hooks", self.models)
            start = time.perf_counter()
            self.api.warm_up(self.models, self.shared, int(self.opts["det_frames_per_batch"]))
            _sync(self.device)
            _trace("flags", "after_warmup")
            self.body["warmup_seconds"] = time.perf_counter() - start
        except Exception as exc:
            logger.exception("loading failed")
            self.body["error"] = f"{type(exc).__name__}: {exc}"
            self.write_response()
            return False
        self.compute_start = time.perf_counter()
        self.write_response()
        return True

    def adopt(self, models: Any) -> None:
        """Warm mode: this request runs on models an earlier request loaded and warmed up (load and warm-up 0 s)."""
        self.models = models
        if self.deterministic:                                     # idempotent; a warm server keeps it
            self.api.disable_nms_time_limit()
        self.compute_start = time.perf_counter()
        self.write_response()

    def _timed(self, clip: _Clip, key: str, start: float) -> None:
        _sync(self.device)
        clip.seconds[key] += time.perf_counter() - start

    def prepare(self, clip: _Clip) -> None:
        """Read, detect and box-filter one clip; failures stay with the clip."""
        boxes = self.shared["hawor_boxes"]
        for stage in ("read", "detect", "box_filter"):
            start = time.perf_counter()
            try:
                if stage == "read":
                    clip.frames, clip.frame_idx, clip.fps = _read_frames(str(clip.spec["frames_path"]))
                    clip.n_frames = int(clip.frame_idx.size)
                    height, width = int(clip.frames.shape[1]), int(clip.frames.shape[2])
                    clip.geom = self.shared["hawor_plan"].ClipGeometry(width, height, float(clip.spec["focal_px"]))
                elif stage == "detect":
                    _trace("current_clip", clip.clip_id)
                    raw = self.api.detect_clip(self.models, self.shared, clip.frames, float(self.opts["det_conf"]),
                                               int(self.opts["det_frames_per_batch"]))
                    _trace("record", "detect", raw, clip=clip.clip_id)
                else:
                    clip.boxes = {side: boxes.filter_hand_boxes(raw[side], clip.frame_idx) for side in SIDES}
                    clip.n_crops = sum(int(np.isfinite(clip.boxes[side]).all(axis=1).sum()) for side in SIDES)
            except Exception as exc:  # noqa: BLE001 - confined to this clip
                clip.fail(stage, exc)
                return
            finally:
                self._timed(clip, stage, start)

    def _new_bank(self) -> Any:
        """A clip group's feature FIFO; padded to fixed-size backbone batches when batch_invariant."""
        core = self.shared["hawor_core"]
        if self.batch_invariant:
            return core.FeatureBank(self.models.hawor, batch=int(self.opts["crops_per_backbone_batch"]),
                                    bf16=bool(self.opts["bf16"]), pad_to_batch=True)
        return core.FeatureBank(self.models.hawor, batch=int(self.opts["crops_per_backbone_batch"]),
                                bf16=bool(self.opts["bf16"]))

    def run(self, clip_specs: Sequence[Mapping[str, Any]]) -> int:
        plan, core = self.shared["hawor_plan"], self.shared["hawor_core"]
        group: list[_Clip] = []
        bank = self._new_bank()
        backbone_seconds = 0.0
        for spec in clip_specs:
            clip = _Clip(spec)
            self.prepare(clip)
            if clip.status == "failed":
                self.finished.append(clip)
                continue
            if plan.closes_group(len(group), bank.n_items, clip.n_crops, int(self.opts["clips_per_group"]),
                                 int(self.opts["max_crops_per_group"])):
                self.finish_group(group, bank, backbone_seconds)
                group, backbone_seconds = [], 0.0
                bank = self._new_bank()
            start = time.perf_counter()
            try:
                items, clip.slots = plan.plan_clip(len(group), clip.frame_idx, clip.boxes, clip.geom, bank.n_items,
                                                   temporal=str(self.opts["temporal"]))
                crops = core.crop_hands(torch.from_numpy(clip.frames), items, device=self.device,
                                        frames_per_chunk=int(self.opts["upload_frames_per_chunk"]))
            except Exception as exc:  # noqa: BLE001 - confined to this clip
                self._timed(clip, "crop", start)
                clip.fail("crop", exc)
                self.finished.append(clip)
                continue
            self._timed(clip, "crop", start)
            clip.frames = None
            group.append(clip)
            try:
                backbone_seconds += bank.push(crops)
            except Exception as exc:  # noqa: BLE001 - the shared backbone fails the whole group
                self.fail_group(group, "backbone", exc)
                group, backbone_seconds = [], 0.0
                bank = self._new_bank()
        if group:
            self.finish_group(group, bank, backbone_seconds)
        self.write_response()
        return 0 if all(clip.status == "done" for clip in self.finished) else 1

    def fail_group(self, group: Sequence[_Clip], stage: str, exc: BaseException) -> None:
        for clip in group:
            clip.fail(stage, exc)
            self.finished.append(clip)
        self.write_response()

    def finish_group(self, group: Sequence[_Clip], bank: Any, backbone_seconds: float) -> None:
        core, gaps, npz = self.shared["hawor_core"], self.shared["hand_gaps"], self.shared["hand_npz"]
        stage = "backbone"
        try:
            backbone_seconds += bank.flush()
            if _TRACE is not None:
                _trace("record", "features", bank.features(), clips=[c.clip_id for c in group])
            stage, start = "heads", time.perf_counter()
            slots = [slot for clip in group for slot in clip.slots]
            geoms, windows = [c.geom for c in group], int(self.opts["windows_per_heads_batch"])
            if self.batch_invariant:                              # the last heads batch at the full shape
                out = core.run_heads(self.models.hawor, self.models.ops, bank.features(), slots, geoms,
                                     windows_per_batch=windows, pad_last=True)
            else:
                out = core.run_heads(self.models.hawor, self.models.ops, bank.features(), slots, geoms,
                                     windows_per_batch=windows)
            _trace("record", "heads", out, clips=[c.clip_id for c in group])
            rows = {(i, side): core.gather_rows(out, slots, i, side, clip.n_frames)
                    for i, clip in enumerate(group) for side in SIDES}
            del out
            _sync(self.device)
            heads_seconds = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001 - shared stages fail the whole group
            self.fail_group(group, stage, exc)
            return
        for i, clip in enumerate(group):
            start = time.perf_counter()
            try:
                for side in SIDES:
                    clip.rows[side], clip.filled[side] = gaps.fill_interior_gaps(rows[(i, side)], clip.frame_idx,
                                                                                 int(self.opts["max_fill_gap"]))
            except Exception as exc:  # noqa: BLE001 - confined to this clip
                clip.fail("gap_fill", exc)
            clip.seconds["gap_fill"] += time.perf_counter() - start
        live = [clip for clip in group if clip.status != "failed"]
        try:
            mano_seconds = self._mano(live)
        except Exception as exc:  # noqa: BLE001 - shared MANO fails the group
            self.fail_group(live, "mano", exc)
            self.finished.extend(clip for clip in group if clip.status == "failed" and clip not in self.finished)
            self.write_response()
            return
        _trace("record", "mano", [clip.mesh.get(side) for clip in live for side in SIDES],
               clips=[c.clip_id for c in live])
        self._split_shared(group, backbone_seconds, heads_seconds, mano_seconds)
        for clip in group:
            if clip.status != "failed":
                start = time.perf_counter()
                try:
                    sides = {side: npz.assemble_side(clip.rows[side], *clip.mesh[side], clip.filled[side],
                                                     clip.boxes[side]) for side in SIDES}
                    npz.write_hand_npz(Path(clip.spec["out_npz"]), clip.frame_idx, clip.fps, sides["left"],
                                       sides["right"], self.models.faces)
                    clip.status = "done"
                except Exception as exc:  # noqa: BLE001 - confined to this clip
                    clip.fail("write", exc)
                clip.seconds["write"] += time.perf_counter() - start
            clip.rows, clip.mesh = {}, {}
            self.finished.append(clip)
        self.write_response()

    def _mano(self, live: Sequence[_Clip]) -> float:
        start = time.perf_counter()
        for side in SIDES:
            picks = [(clip, np.flatnonzero(clip.rows[side]["valid"])) for clip in live]
            total = sum(int(rows.size) for _, rows in picks)
            cat = {key: np.concatenate([clip.rows[side][key][rows] for clip, rows in picks]) if picks else None
                   for key in ("rot", "hand_pose", "betas", "trans")}
            vertices = np.zeros((0, 778, 3), np.float32)
            joints = np.zeros((0, 21, 3), np.float32)
            if total and self.batch_invariant:
                vertices, joints = self._mano_fixed(side, cat)
            elif total:
                vertices, joints = self.api.run_mano(self.models, side, cat["rot"], cat["hand_pose"], cat["betas"],
                                                     cat["trans"])
            offset = 0
            for clip, rows in picks:
                v = np.zeros((clip.n_frames, 778, 3), np.float32)
                j = np.zeros((clip.n_frames, 21, 3), np.float32)
                if rows.size:
                    v[rows], j[rows] = vertices[offset:offset + rows.size], joints[offset:offset + rows.size]
                    offset += int(rows.size)
                clip.mesh[side] = (v, j)
        _sync(self.device)
        return time.perf_counter() - start

    def _mano_fixed(self, side: str, cat: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """MANO over `cat`'s rows in calls of exactly `mano_rows_per_batch` rows; the last chunk is padded by
        repeating its own first row, and padded outputs are dropped."""
        total, rows = int(cat["rot"].shape[0]), self.mano_rows
        vertices, joints = [], []
        for start in range(0, total, rows):
            index = np.arange(start, min(start + rows, total))
            keep = int(index.size)
            if keep < rows:
                index = np.concatenate([index, np.full(rows - keep, start, dtype=index.dtype)])
            v, j = self.api.run_mano(self.models, side, cat["rot"][index], cat["hand_pose"][index],
                                     cat["betas"][index], cat["trans"][index])
            vertices.append(np.asarray(v)[:keep])
            joints.append(np.asarray(j)[:keep])
        return np.concatenate(vertices), np.concatenate(joints)

    @staticmethod
    def _split_shared(group: Sequence[_Clip], backbone: float, heads: float, mano: float) -> None:
        def share(values: list[int]) -> list[float]:
            total = sum(values)
            return [v / total if total else 1.0 / len(values) for v in values]

        live_valid = [sum(int(c.rows[s]["valid"].sum()) for s in SIDES) if c.status != "failed" else 0 for c in group]
        for clip, wb, wh, wm in zip(group, share([c.n_crops for c in group]), share([len(c.slots) for c in group]),
                                    share(live_valid)):
            clip.seconds["backbone"] += backbone * wb
            clip.seconds["heads"] += heads * wh
            clip.seconds["mano"] += mano * wm


def run_request(request: Mapping[str, Any], response: Path, shared: Mapping[str, ModuleType], api: Any,
                device: str = "cuda") -> int:
    """Processes one request; returns the exit code (0 all done, 1 some clip failed, 2 load failure)."""
    call = _Call(request, Path(response), shared, api, device)
    if not call.load(Path(request["hawor_root"])):
        return 2
    return call.run(request["clips"])


@dataclass(frozen=True)
class _Warm:
    """What a `--serve` process keeps between requests: the models and the RNG states right after warm-up."""

    models: Any
    hawor_root: str
    rng: tuple[Any, ...]


def _rng_state(device: str) -> tuple[Any, ...]:
    cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if cuda else None, np.random.get_state(),
            random.getstate())


def _set_rng_state(state: tuple[Any, ...]) -> None:
    cpu, cuda, np_state, py_state = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    np.random.set_state(np_state)
    random.setstate(py_state)


def _write_error(response: Path, error: str) -> None:
    body = {"version": 1, "error": error, "load_seconds": 0.0, "warmup_seconds": 0.0, "compute_seconds": 0.0,
            "peak_cuda_gb": 0.0, "clips": []}
    tmp = response.with_name(response.name + ".tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, response)


def serve_request(request_path: Path, shared: Mapping[str, ModuleType], api: Any, device: str,
                  warm: _Warm | None) -> tuple[int, _Warm | None]:
    """One `--serve` request (module docstring); returns its exit code and the state to keep (None: nothing loaded)."""
    response = request_path.with_name("response.json")
    request = json.loads(request_path.read_text())
    _trace("new_request")
    if request.get("version") != 1:
        _write_error(response, f"unsupported request version {request.get('version')!r}")
        return 2, warm
    cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    call = _Call(request, response, shared, api, device)
    if warm is None:
        if not call.load(Path(request["hawor_root"])):
            return 2, None
        warm = _Warm(call.models, str(request["hawor_root"]), _rng_state(device))
    elif str(request["hawor_root"]) != warm.hawor_root:
        _write_error(response, f"this warm driver loaded {warm.hawor_root}, the request names {request['hawor_root']}")
        return 2, warm
    else:
        _set_rng_state(warm.rng)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        call.adopt(warm.models)
    code = call.run(request["clips"])
    del call
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()                                   # transient blocks back; only the weights stay
    return code, warm


def serve(lines: Iterable[str], protocol: TextIO, shared: Mapping[str, ModuleType], api: Any,
          device: str = "cuda") -> int:
    """The `--serve` loop: one request path per line until EOF; returns the process exit code."""
    warm: _Warm | None = None
    for line in lines:
        path = line.strip()
        if not path:
            continue
        code, warm = serve_request(Path(path), shared, api, device, warm)
        logger.info("served %s: code %d", path, code)
        protocol.write(json.dumps({"request": path, "code": code, "serving": warm is not None}) + "\n")
        protocol.flush()
        if warm is None:                                           # loading failed: nothing to serve with
            return 2
    return 0


def import_check(shared: Mapping[str, ModuleType]) -> int:
    """Builds one instance of each shared dataclass and loads the models file, without HaWoR or CUDA."""
    plan, core, npz = shared["hawor_plan"], shared["hawor_core"], shared["hand_npz"]
    plan.ClipGeometry(1280, 720, 999.0)
    plan.CropItem(0, 0, "right", (640.0, 360.0), 1.0, False)
    plan.Slot(0, "right", 0, 0, 0.0, (640.0, 360.0), 1.0, False, 0)
    core.HaworOps(rot6d_to_rotmat=torch.clone, rotmat_to_aa=torch.clone)
    empty = torch.zeros(0)
    core.HeadOutputs(rotmat=empty, aa=empty, betas=empty, trans=empty)
    z = np.zeros((1, 3), np.float32)
    npz.SideArrays(vertices=np.zeros((1, 778, 3), np.float32), joints=np.zeros((1, 21, 3), np.float32), trans=z, rot=z,
                   hand_pose=np.zeros((1, 45), np.float32), betas=np.zeros((1, 10), np.float32),
                   valid=np.ones(1, bool), infilled=np.zeros(1, bool), box=np.zeros((1, 5), np.float32))
    load_module("fastdo_driver_models", REPO_ROOT / MODELS_FILE)
    heavy = sorted(name for name in ("lib", "hawor", "scripts", "ultralytics", "smplx") if name in sys.modules)
    if heavy or torch.cuda.is_initialized():
        print(f"import-check failed: imported {heavy}, cuda initialized {torch.cuda.is_initialized()}")
        return 1
    print("import-check ok")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--serve", action="store_true", help="warm mode: request paths on stdin, one per line")
    args = parser.parse_args(argv)
    protocol: TextIO | None = None
    if args.serve:                             # stdout carries completion lines only; any other write goes to stderr
        sys.stdout.flush()
        protocol = os.fdopen(os.dup(1), "w")
        os.dup2(2, 1)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    shared = load_shared_modules(REPO_ROOT)
    if args.import_check:
        return import_check(shared)
    if not args.serve and (args.response is None or (args.request is None and not args.self_check)):
        parser.error("--response and one of --request / --self-check are required (or --serve)")
    shared["gpu_budget"].apply_cpu_budget()
    shared["gpu_budget"].apply_gpu_budget()
    api = load_module("fastdo_driver_models", REPO_ROOT / MODELS_FILE)
    if os.environ.get("FASTDO_HAWOR_TRACE") or os.environ.get("FASTDO_HAWOR_NMS_TRUNCATE"):   # diagnosis only
        global _TRACE
        _TRACE = load_module("fastdo_driver_trace", REPO_ROOT / TRACE_FILE)
    if protocol is not None:
        return serve(sys.stdin, protocol, shared, api)
    body: dict[str, Any] = {"version": 1, "error": ""}
    if args.self_check:
        try:
            body["self_check"] = api.self_check(api.load_models(Path.cwd(), shared), shared)
            code = 0
        except Exception as exc:
            logger.exception("self-check failed")
            body["error"], code = f"{type(exc).__name__}: {exc}", 2
        args.response.write_text(json.dumps(body, indent=1))
        return code
    request = json.loads(args.request.read_text())
    if request.get("version") != 1:
        body |= {"error": f"unsupported request version {request.get('version')!r}", "clips": []}
        args.response.write_text(json.dumps(body))
        return 2
    return run_request(request, args.response, shared, api)


if __name__ == "__main__":
    sys.exit(main())
