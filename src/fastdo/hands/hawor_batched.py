"""Batched hand-mesh recovery at 10 fps with HaWoR, uv side: config, request and response files, the driver call.

HaWoR, its YOLOv8 hand detector (ultralytics) and MANO are never imported in the uv environment. `run_hands_batched`
starts `run/pipeline/_hawor_batched_driver.py` with the baseline's hawor env python (`hands.hawor_python`) and
cwd = `hands.hawor_root`, then exchanges JSON files with it. One call loads the models once for all of its clips.

Runtime licences (the repository copies none of their code):
- HaWoR code and weights: CC BY-NC-ND 4.0;
- ultralytics, the detector runtime: AGPL-3.0;
- MANO model files: the MANO licence.
This stage is the research default behind `hands.backend: hawor_batched`. Shipping it as the engine default is the
user's decision.

File contract:
- `request.json`: `{"version": 1, "hawor_root": str, "options": {det_conf, det_frames_per_batch, bf16, temporal,
  crops_per_backbone_batch, windows_per_heads_batch, clips_per_group, max_crops_per_group, upload_frames_per_chunk,
  max_fill_gap[, batch_invariant, mano_rows_per_batch][, deterministic]}, "clips": [{clip_id, frames_path, focal_px,
  out_npz}]}`; all
  paths absolute. The bracketed options (`BATCH_INVARIANT_KEYS`) are written only when `hands.batch_invariant` is
  true, so a default request is byte-identical to one written before they existed; a driver reads their absence as
  off.
- Frames npz (the decode stage's `decode.npz` is read in place): `frames` (N,H,W,3) uint8 RGB, `frame_idx` (N,) int64
  strictly increasing, `fps` float64; it loads with `allow_pickle=False`.
- `response.json`: `{"version": 1, "error": str, "load_seconds", "warmup_seconds", "compute_seconds", "peak_cuda_gb",
  "clips": [{clip_id, status: "done"|"failed", error, n_frames, seconds: {read, detect, box_filter, crop, backbone,
  heads, gap_fill, mano, write, total}}]}`, plus `"batch_invariant": true, "mano_rows_per_batch"` when the request
  turned it on. It is rewritten atomically after loading and after every clip group, and lists the clips
  finished so far.
- Exit codes: 0 when all clips are done, 1 when some clip failed, 2 when loading failed (`error` set, `clips` empty).
- Output: `fastdo.hands.hand_npz`'s 23-key npz, one row per frames-npz row.

`run_hands_batched` raises `RuntimeError` on exit 2 or any code other than 0 or 1 (the message is the response error,
else the last 20 lines of `driver.log`), and `TimeoutError` after `call_timeout_s`, having killed the driver's process
group. A `done` clip whose npz fails `validate_hand_npz`, or whose row count differs from `n_frames`, is reported
`failed`.

Batch invariance (opt-in `hands.batch_invariant`, default false): batched HaWoR runs each model at a batch size
the OTHER clips of a group set (the backbone FIFO's remainder, the last heads batch, one MANO call over the group's
rows), and cuBLAS/cuDNN pick kernels by shape, so a window's hands could move with its batch mates. When on,
the driver runs every one of those calls at one fixed shape: the backbone remainder padded with zero crops to
`crops_per_backbone_batch`, the last heads batch padded to `windows_per_heads_batch` windows with copies of its first
window, and MANO in chunks of exactly `mano_rows_per_batch` rows (the last padded with copies of its first row);
padded outputs are dropped. Off, the driver makes exactly the calls it made before.

Timing independence (opt-in `hands.deterministic`, default false): ultralytics' NMS stops after `2.0 + 0.05 x
batch` seconds of wall clock (2.8 s per 16-frame detector call) and returns NO detections for the rest of that call, so a
stalled driver silently drops both hands on a chunk's trailing frames (observed: window f836a2b5:k074, rows 77-79).
When on, `options.deterministic` is written into the request and the driver calls the NMS with `max_time_img=inf`
(`_hawor_driver_models.disable_nms_time_limit`); outputs change only where the limit would have cut a call short.

Warm mode (opt-in `hands.warm_driver`): a `WarmHandsDriver` keeps ONE `--serve` driver process for its owner's
lifetime, so the ~10 s of model load and warm-up is paid once per run instead of once per call. Each call still writes
`request.json`, reads `response.json` and `driver.log` (the server log's slice for that request) in its job dir, and
gets a request's exit code from the server's completion line, or the server's own exit code when it died mid-request;
the same `RuntimeError`/`TimeoutError` rules then apply. A timeout or any interruption kills the server's process group,
and the next call starts a fresh server. The server's weights stay resident on the GPU between calls.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import select
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Self

from omegaconf import DictConfig

from fastdo.hands.hand_npz import validate_hand_npz
from fastdo.hands.hawor_plan import TEMPORAL_MODES

__all__ = [
    "BATCH_INVARIANT_KEYS",
    "DETERMINISTIC_KEYS",
    "DRIVER_SCRIPT",
    "OPTION_KEYS",
    "SECONDS_KEYS",
    "HandsCallReport",
    "HandsClipInput",
    "HandsClipResult",
    "HandsConfig",
    "WarmHandsDriver",
    "call_timeout_s",
    "driver_command",
    "driver_env",
    "hands_config_from_cfg",
    "read_response",
    "release_cuda_cache",
    "run_hands_batched",
    "serve_command",
    "verify_clip_result",
    "write_request",
]

logger = logging.getLogger(__name__)

DRIVER_SCRIPT: Path = Path(__file__).resolve().parents[3] / "run" / "pipeline" / "_hawor_batched_driver.py"
BACKEND = "hawor_batched"
OPTION_KEYS: tuple[str, ...] = ("det_conf", "det_frames_per_batch", "bf16", "temporal", "crops_per_backbone_batch",
                                "windows_per_heads_batch", "clips_per_group", "max_crops_per_group",
                                "upload_frames_per_chunk", "max_fill_gap")
BATCH_INVARIANT_KEYS: tuple[str, ...] = ("batch_invariant", "mano_rows_per_batch")   # in options only when on
DETERMINISTIC_KEYS: tuple[str, ...] = ("deterministic",)                              # in options only when on
SECONDS_KEYS: tuple[str, ...] = ("read", "detect", "box_filter", "crop", "backbone", "heads", "gap_fill", "mano",
                                 "write", "total")
_POSITIVE_INTS = ("det_frames_per_batch", "crops_per_backbone_batch", "windows_per_heads_batch", "clips_per_group",
                  "max_crops_per_group", "upload_frames_per_chunk", "max_clips_per_call", "mano_rows_per_batch")
_LOG_TAIL_LINES = 20


@dataclass(frozen=True)
class HandsConfig:
    """Settings of the batched hands stage (`cfg.hands`). A call's timeout is `timeout_s_base + timeout_s_per_clip *
    n_clips` (the base is a field so tests can shorten it)."""

    hawor_python: Path
    hawor_root: Path
    det_conf: float = 0.2
    det_frames_per_batch: int = 16
    bf16: bool = True
    temporal: str = "source_grid"
    crops_per_backbone_batch: int = 64
    windows_per_heads_batch: int = 16
    clips_per_group: int = 8
    max_crops_per_group: int = 4096
    upload_frames_per_chunk: int = 256
    max_clips_per_call: int = 256
    max_fill_gap: int = 120
    timeout_s_base: float = 120.0
    timeout_s_per_clip: float = 20.0
    warm_driver: bool = False                  # one `--serve` driver per owner (`WarmHandsDriver`), opt-in
    batch_invariant: bool = False              # fixed-shape backbone/heads/MANO calls (module docstring), opt-in
    mano_rows_per_batch: int = 512             # MANO rows per call when batch_invariant; unused otherwise
    deterministic: bool = False                # YOLO NMS without its wall-clock limit (module docstring), opt-in

    def __post_init__(self) -> None:
        if self.temporal not in TEMPORAL_MODES:
            raise ValueError(f"temporal must be one of {TEMPORAL_MODES}, got {self.temporal!r}")
        for name in _POSITIVE_INTS:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not 0.0 <= self.det_conf <= 1.0 or self.max_fill_gap < 0:
            raise ValueError(f"need 0 <= det_conf <= 1 and max_fill_gap >= 0, got {self.det_conf}, {self.max_fill_gap}")
        if self.timeout_s_base <= 0 or self.timeout_s_per_clip < 0:
            raise ValueError(f"need timeout_s_base > 0 and timeout_s_per_clip >= 0, got {self.timeout_s_base}, "
                             f"{self.timeout_s_per_clip}")


def hands_config_from_cfg(cfg: DictConfig) -> HandsConfig:
    """`HandsConfig` from `cfg.hands`; keys not given keep their defaults.

    Raises:
        ValueError: If `cfg.hands.backend` is set to anything other than `"hawor_batched"`.
    """
    section = cfg.hands
    backend = section.get("backend", BACKEND)
    if backend != BACKEND:
        raise ValueError(f"hands.backend is {backend!r}; this config reader serves {BACKEND!r} only")
    values: dict[str, Any] = {}
    for f in fields(HandsConfig):
        if f.name in section:
            values[f.name] = Path(str(section[f.name])) if f.name in ("hawor_python", "hawor_root") else section[f.name]
    return HandsConfig(**values)


@dataclass(frozen=True)
class HandsClipInput:
    clip_id: str
    frames_path: Path
    focal_px: float
    out_npz: Path


@dataclass(frozen=True)
class HandsClipResult:
    clip_id: str
    status: str
    error: str
    n_frames: int
    seconds: dict[str, float]


@dataclass(frozen=True)
class HandsCallReport:
    returncode: int
    load_seconds: float
    warmup_seconds: float
    compute_seconds: float
    peak_cuda_gb: float
    clips: tuple[HandsClipResult, ...]
    batch_invariant: bool = False              # the driver ran fixed-shape calls (the response's flag)
    deterministic: bool = False                # the driver ran its NMS without the time limit (the response's flag)


def call_timeout_s(cfg: HandsConfig, n_clips: int) -> float:
    """Wall-clock limit of one driver call."""
    return float(cfg.timeout_s_base + cfg.timeout_s_per_clip * n_clips)


def _write_json(path: Path, body: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1))
    os.replace(tmp, path)


def write_request(job_dir: Path, clips: Sequence[HandsClipInput], cfg: HandsConfig) -> Path:
    """Writes `<job_dir>/request.json` with absolute paths and returns its path.

    Raises:
        ValueError: If two clips share a `clip_id`.
    """
    ids = [clip.clip_id for clip in clips]
    if len(set(ids)) != len(ids):
        raise ValueError(f"clip ids must be unique within a call, got {ids}")
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    body = {"version": 1, "hawor_root": str(Path(cfg.hawor_root).absolute()),
            "options": {key: getattr(cfg, key) for key in OPTION_KEYS}
            # only when on, so a default request is byte-identical on the wire.
            | ({key: getattr(cfg, key) for key in BATCH_INVARIANT_KEYS} if cfg.batch_invariant else {})
            | ({key: getattr(cfg, key) for key in DETERMINISTIC_KEYS} if cfg.deterministic else {}),
            "clips": [{"clip_id": c.clip_id, "frames_path": str(Path(c.frames_path).absolute()),
                       "focal_px": float(c.focal_px), "out_npz": str(Path(c.out_npz).absolute())} for c in clips]}
    path = job_dir / "request.json"
    _write_json(path, body)
    return path


def driver_command(cfg: HandsConfig, request: Path, response: Path) -> list[str]:
    return ["nice", "-n", "15", str(cfg.hawor_python), str(DRIVER_SCRIPT), "--request", str(request), "--response",
            str(response)]


def serve_command(cfg: HandsConfig) -> list[str]:
    """The warm driver's command line."""
    return ["nice", "-n", "15", str(cfg.hawor_python), str(DRIVER_SCRIPT), "--serve"]


def driver_env(base: Mapping[str, str]) -> dict[str, str]:
    """The driver's environment: `base` without the weights-only override and `PYTHONPATH`, with the GPU and CPU budget
    defaults and BLAS threads set to `FASTDO_CPU_THREADS`."""
    env = {k: v for k, v in base.items() if k not in ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "PYTHONPATH")}
    env.setdefault("FASTDO_GPU_MEM_GB", "12")
    env.setdefault("FASTDO_CPU_THREADS", "4")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = env["FASTDO_CPU_THREADS"]
    return env


def read_response(path: Path, returncode: int) -> HandsCallReport:
    """Parses a driver `response.json`; `returncode` is recorded as given."""
    body = json.loads(Path(path).read_text())
    clips = tuple(HandsClipResult(clip_id=str(c["clip_id"]), status=str(c["status"]), error=str(c.get("error", "")),
                                  n_frames=int(c.get("n_frames", 0)),
                                  seconds={str(k): float(v) for k, v in dict(c.get("seconds", {})).items()})
                  for c in body.get("clips", []))
    return HandsCallReport(returncode=int(returncode), load_seconds=float(body.get("load_seconds", 0.0)),
                           warmup_seconds=float(body.get("warmup_seconds", 0.0)),
                           compute_seconds=float(body.get("compute_seconds", 0.0)),
                           peak_cuda_gb=float(body.get("peak_cuda_gb", 0.0)), clips=clips,
                           batch_invariant=bool(body.get("batch_invariant", False)),
                           deterministic=bool(body.get("deterministic", False)))


def verify_clip_result(clip: HandsClipInput, result: HandsClipResult) -> HandsClipResult:
    """`result`, turned `failed` when it claims `done` but its npz does not validate or has another row count."""
    if result.status == "failed":
        return result
    if result.status != "done":
        return replace(result, status="failed", error=f"unknown status {result.status!r}: {result.error}")
    try:
        rows = validate_hand_npz(clip.out_npz)
    except Exception as exc:  # noqa: BLE001 - a missing, truncated or malformed npz all fail the clip
        return replace(result, status="failed", error=f"{type(exc).__name__}: {exc}")
    if rows != result.n_frames:
        return replace(result, status="failed", error=f"{clip.out_npz}: {rows} rows, driver reported {result.n_frames}")
    return result


def _log_tail(log: Path) -> str:
    try:
        return "\n".join(log.read_text(errors="replace").splitlines()[-_LOG_TAIL_LINES:])
    except OSError:
        return f"(no {log})"


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


class WarmHandsDriver:
    """One `--serve` HaWoR driver kept across calls (module docstring). Started on the first `request`; `close()`
    (also a context manager and an `atexit` hook) ends it. Server logs are `<log_dir>/warm_driver_<n>.log`."""

    def __init__(self, cfg: HandsConfig, log_dir: Path) -> None:
        self.cfg, self.log_dir = cfg, Path(log_dir)
        self.spawns = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: Path | None = None
        self._buf = b""
        atexit.register(self.close)

    @property
    def pid(self) -> int | None:
        """The live server's pid (its process group id too), else None."""
        return self._proc.pid if self._proc is not None and self._proc.poll() is None else None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(self, request: Path, timeout: float, log_path: Path) -> int:
        """Runs one request; returns its exit code (the completion line's, else the server's exit code).

        Raises:
            subprocess.TimeoutExpired: After `timeout` s without a completion line; the server's group is killed first.
            RuntimeError: If the server answers something other than this request's completion line.
        """
        proc = self._ensure()
        assert proc.stdin is not None and proc.stdout is not None and self._log is not None
        log, offset = self._log, self._log.stat().st_size
        try:
            try:
                proc.stdin.write(f"{request}\n".encode())
                proc.stdin.flush()
            except OSError:                                        # BrokenPipeError: it died before reading it
                return self._reap()
            deadline = time.monotonic() + timeout
            while b"\n" not in self._buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._kill()
                    raise subprocess.TimeoutExpired(serve_command(self.cfg), timeout)
                ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
                chunk = os.read(proc.stdout.fileno(), 1 << 16) if ready else None
                if chunk == b"" or (chunk is None and proc.poll() is not None):
                    return self._reap()
                self._buf += chunk or b""
            line, _, self._buf = self._buf.partition(b"\n")
            try:
                ack = json.loads(line)
                if ack["request"] != str(request):
                    raise ValueError(f"completion line for {ack['request']!r}")
                code, serving = int(ack["code"]), bool(ack["serving"])
            except (ValueError, KeyError, TypeError) as exc:
                self._kill()
                raise RuntimeError(f"warm HaWoR driver sent {line[:200]!r} for {request}: {exc}") from None
            if not serving:                                        # a load failure: the server exits by itself
                self._reap(wait_s=30.0)
            return code
        except BaseException:
            self._kill()
            raise
        finally:
            _copy_tail(log, offset, log_path)

    def close(self) -> None:
        """Ends the server (EOF on stdin, then SIGKILL to its group after 30 s). Safe to call twice."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None and proc.stdin is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=30.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._kill()

    def _ensure(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            logger.warning("warm HaWoR driver %d exited with code %s between calls; starting a new one",
                           self._proc.pid, self._proc.returncode)
            self._kill()
        self.spawns += 1
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log, self._buf = self.log_dir / f"warm_driver_{self.spawns:02d}.log", b""
        with self._log.open("wb") as log:
            self._proc = subprocess.Popen(serve_command(self.cfg), cwd=Path(self.cfg.hawor_root).absolute(),
                                          env=driver_env(os.environ), start_new_session=True,
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=log, bufsize=0)
        logger.info("warm HaWoR driver started: pid %d, log %s", self._proc.pid, self._log)
        return self._proc

    def _reap(self, wait_s: float | None = None) -> int:
        """The server's exit code once it has exited (killed after `wait_s`, if given); the next call starts anew."""
        proc = self._proc
        assert proc is not None
        try:
            returncode = proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            self._kill()
            return int(proc.returncode)
        self._kill()
        return int(returncode)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        _kill_group(proc)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _copy_tail(log: Path, offset: int, dest: Path) -> None:
    """`log` from byte `offset` on, written to `dest` (a request's own `driver.log` in warm mode)."""
    try:
        with log.open("rb") as source:
            source.seek(offset)
            Path(dest).write_bytes(source.read())
    except OSError as exc:
        logger.warning("could not copy the warm driver log %s to %s: %s", log, dest, exc)


def release_cuda_cache() -> float:
    """Hands this process's CACHED CUDA blocks back before the driver starts; returns the GiB released.

    The driver is a separate process with its own allocator, so while it runs the card holds BOTH. Measured
    2026-09-17 on the shared card: our engine process held 6758 MiB -- overwhelmingly cached blocks from the mask
    and shape stages, not live tensors -- while the driver held 7330 MiB, and the guard killed us at a 14 GB total
    for a job budgeted at 8. Cached blocks cost nothing to release (the allocator re-grows on the next stage)
    and holding them across a subprocess call is pure risk on a card we share.

    Returns 0.0, and does nothing, when torch or CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:                                   # a CPU-only caller has no cache to release
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    before = torch.cuda.memory_reserved()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return max(0.0, (before - torch.cuda.memory_reserved()) / 2**30)


def run_hands_batched(clips: Sequence[HandsClipInput], cfg: HandsConfig, job_dir: Path,
                      warm: WarmHandsDriver | None = None) -> HandsCallReport:
    """One driver call for `clips` (see the module docstring); results come back in `clips`' order. With `warm`, the
    call is one request to that warm server instead of a fresh driver process.

    Raises:
        RuntimeError: On a load failure or an unexpected exit code.
        TimeoutError: When the call exceeds `call_timeout_s`; the driver's process group is killed first.
    """
    job_dir = Path(job_dir)
    request = write_request(job_dir, clips, cfg)
    response, log_path = job_dir / "response.json", job_dir / "driver.log"
    response.unlink(missing_ok=True)
    timeout = call_timeout_s(cfg, len(clips))
    released = release_cuda_cache()          # the driver's memory is ADDED to ours, so give back what we can first
    logger.info("HaWoR driver call: %d clips, timeout %.0f s, released %.2f GiB of cache, log %s",
                len(clips), timeout, released, log_path)
    timed_out = f"HaWoR driver exceeded {timeout:.0f} s for {len(clips)} clips; see {log_path}"
    if warm is not None:
        try:
            returncode = warm.request(request, timeout, log_path)
        except subprocess.TimeoutExpired:
            raise TimeoutError(timed_out) from None
    else:
        with log_path.open("wb") as log:
            proc = subprocess.Popen(driver_command(cfg, request, response), cwd=Path(cfg.hawor_root).absolute(),
                                    env=driver_env(os.environ), start_new_session=True, stdout=log,
                                    stderr=subprocess.STDOUT)
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                raise TimeoutError(timed_out) from None
            except BaseException:
                _kill_group(proc)
                raise
    if returncode not in (0, 1) or not response.is_file():
        error = ""
        if response.is_file():
            try:
                error = str(json.loads(response.read_text()).get("error", ""))
            except ValueError:
                error = ""
        exc = RuntimeError(f"HaWoR driver exited with code {returncode}: {error or _log_tail(log_path)}")
        if returncode == -signal.SIGKILL:
            # SIGKILL on a shared card means the GPU guard, not this clip. Marching to the next clip repeats it
            # immediately: on 2026-09-17 the guard killed a driver every ~12 s while the peer sat at 74 GB, and
            # without this flag the run would have marked most of a 140-clip corpus failed in a few minutes.
            # `fatal_to_run` tells the orchestrator to record THIS clip's failure and then stop the run, so a
            # resumable rerun retries the untouched clips instead of re-solving a corpus of failures.
            exc.fatal_to_run = True                                    # type: ignore[attr-defined]
        raise exc
    report = read_response(response, returncode)
    by_id = {result.clip_id: result for result in report.clips}
    results = tuple(verify_clip_result(clip, by_id[clip.clip_id]) if clip.clip_id in by_id
                    else HandsClipResult(clip.clip_id, "failed", "the driver reported no result for this clip", 0, {})
                    for clip in clips)
    return replace(report, clips=results)
