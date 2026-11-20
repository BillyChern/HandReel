"""Per-stage wall-clock and GPU-memory profiler producing JSON reports."""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageRecord:
    """Timing of one pipeline stage."""

    name: str
    seconds: float
    peak_gpu_mem_gb: float = 0.0
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class ProfileReport:
    """All stage timings for one clip processed by one system."""

    clip_id: str
    system: str
    n_frames: int
    fps: float
    stages: list[StageRecord] = field(default_factory=list)

    def total_seconds(self) -> float:
        return float(sum(s.seconds for s in self.stages))

    def gpu_hours(self) -> float:
        return self.total_seconds() / 3600.0

    def clips_per_gpu_hour(self) -> float:
        total = self.total_seconds()
        return float("inf") if total <= 0.0 else 3600.0 / total

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = asdict(self)
        payload["total_seconds"] = self.total_seconds()
        payload["clips_per_gpu_hour"] = self.clips_per_gpu_hour()
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> ProfileReport:
        raw = json.loads(path.read_text())
        stages = [StageRecord(**s) for s in raw["stages"]]
        return cls(
            clip_id=raw["clip_id"], system=raw["system"], n_frames=int(raw["n_frames"]),
            fps=float(raw["fps"]), stages=stages,
        )


def _cuda_peak_gb(reset: bool) -> float:
    """This stage's peak CUDA allocation, or 0.0 when the process has never initialized CUDA.

    The guard is `is_initialized`, NOT `is_available`: `torch.cuda.synchronize()` CREATES the primary
    context when none exists (measured: 516 MiB on an H100), so gating on availability made every
    CPU-only stage pay a GPU context for a measurement that is zero by construction. A process that has
    not initialized CUDA has allocated nothing, so 0.0 is the exact answer, not an approximation -- and
    for the first GPU stage, whose entry therefore skips the reset, `max_memory_allocated` still reports
    that stage's true peak, since nothing was allocated before it.
    """
    try:
        import torch
    except ImportError:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not torch.cuda.is_initialized():
            return 0.0
    torch.cuda.synchronize()
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 1e9


class Profiler:
    """Collects StageRecords via a context manager or explicit durations."""

    def __init__(self, clip_id: str, system: str, n_frames: int, fps: float,
                 sync_cuda: bool = True) -> None:
        self._report = ProfileReport(clip_id=clip_id, system=system, n_frames=n_frames, fps=fps)
        self._sync_cuda = sync_cuda

    @contextmanager
    def stage(self, name: str, **meta: str) -> Iterator[None]:
        if self._sync_cuda:
            _cuda_peak_gb(reset=True)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            peak = _cuda_peak_gb(reset=False) if self._sync_cuda else 0.0
            secs = time.perf_counter() - t0
            self.add(name, secs, peak_gpu_mem_gb=peak, **meta)

    def add(self, name: str, seconds: float, peak_gpu_mem_gb: float = 0.0, **meta: str) -> None:
        rec = StageRecord(name=name, seconds=float(seconds), peak_gpu_mem_gb=float(peak_gpu_mem_gb),
                          meta=dict(meta))
        self._report.stages.append(rec)
        logger.info("stage %-28s %9.2f s  peak %.1f GB", name, seconds, peak_gpu_mem_gb)

    def report(self) -> ProfileReport:
        return self._report
