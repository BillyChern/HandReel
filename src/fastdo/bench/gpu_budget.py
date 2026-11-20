"""Per-process CUDA memory cap for runs on a shared GPU host.

A GPU may be shared with other jobs, so every HandReel process caps its own CUDA allocations. The
cap is applied from the test session fixture and from the pipeline entry points, never at import time.
"""

from __future__ import annotations

import logging
import os

import torch

__all__ = ["CPU_ENV_VAR", "DEFAULT_CPU_THREADS", "DEFAULT_GB", "ENV_VAR", "apply_cpu_budget", "apply_gpu_budget"]

logger = logging.getLogger(__name__)

ENV_VAR = "FASTDO_GPU_MEM_GB"
DEFAULT_GB = 12.0
CPU_ENV_VAR = "FASTDO_CPU_THREADS"
DEFAULT_CPU_THREADS = 4


def apply_gpu_budget(gb: float | None = None, device: int = 0) -> float | None:
    """Cap this process's CUDA allocations on `device` at `gb` GiB.

    `gb` defaults to the `FASTDO_GPU_MEM_GB` environment variable, else 12 GiB. A non-positive value
    opts out. Returns the fraction of device memory applied, or None when CUDA is unavailable or the
    cap was skipped.
    """
    if not torch.cuda.is_available():
        return None
    if gb is None:
        gb = float(os.environ.get(ENV_VAR, DEFAULT_GB))
    if gb <= 0:
        logger.info("CUDA per-process cap disabled (%s=%s)", ENV_VAR, gb)
        return None
    total = float(torch.cuda.get_device_properties(device).total_memory)
    fraction = min(1.0, gb * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    logger.info("CUDA per-process cap %.1f GiB (%.3f of device %d)", gb, fraction, device)
    return fraction


def apply_cpu_budget(threads: int | None = None) -> int:
    """Cap this process's torch intra-op threads (and the BLAS thread env vars for child processes).

    Every test session and pipeline entry point caps its threads, so several workers can share a host.
    `threads` defaults to `FASTDO_CPU_THREADS`, else 4. Returns the thread count applied.
    """
    if threads is None:
        threads = int(os.environ.get(CPU_ENV_VAR, DEFAULT_CPU_THREADS))
    threads = max(1, threads)
    torch.set_num_threads(threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    logger.info("CPU budget: %d torch threads", threads)
    return threads
