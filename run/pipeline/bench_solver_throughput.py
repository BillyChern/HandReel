"""Throughput curve of the batched trajectory solver: clips per second at each batch size B.

Solves B DexYCB-like synthetic clips together (`fastdo.bench.solver_throughput`) and writes
`<paths.outputs>/bench/solver_throughput.json` with every row, the speed-up over B = 1 and the device name.
The process's CUDA budget (`FASTDO_GPU_MEM_GB`, default 12 GB) and CPU budget (`FASTDO_CPU_THREADS`) apply.

Example:
  uv run python run/pipeline/bench_solver_throughput.py bench.batch_sizes=[1,2,4,8,16] \
      paths.outputs=<out_dir> hydra.run.dir=<out_dir>/hydra
"""

from __future__ import annotations

import os

# Expandable segments keep a large batch's alternating plan/cost allocations from fragmenting the capped pool;
# set before torch initializes CUDA. torch 2.9 reads PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF is its deprecated
# name); a value set under either name by the caller wins.
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.bench.solver_throughput import allocator_conf, measure_throughput, synthetic_dexycb_like
from fastdo.solver import SolverConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    bench = cfg.bench
    sizes = [int(b) for b in bench.batch_sizes]
    cuda = torch.cuda.is_available()
    inputs = synthetic_dexycb_like(max(sizes), n_frames=int(bench.n_frames), n_surface=int(bench.n_surface),
                                   seed=int(bench.seed))
    fused = bool(bench.get("fused_sinkhorn", False))
    stop_mib = int(float(bench.get("stop_gb", 0)) * 1024) or None
    rows = measure_throughput(inputs, sizes, SolverConfig(device="cuda" if cuda else "cpu", fused_sinkhorn=fused),
                              repeats=int(bench.repeats), stop_mib=stop_mib)
    base = next((r.clips_per_second for r in rows if r.batch_size == 1 and not r.error), 0.0)
    speedup = {r.batch_size: (r.clips_per_second / base if base > 0 else 0.0) for r in rows}
    out = Path(cfg.paths.outputs) / "bench" / "solver_throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "n_frames": int(bench.n_frames),
               "n_surface": int(bench.n_surface), "fused_sinkhorn": fused,
               "alloc_conf": allocator_conf(), "rows": [asdict(r) for r in rows],
               "speedup_vs_b1": speedup}
    out.write_text(json.dumps(payload, indent=2))
    for r in rows:
        nvml = "n/a" if r.nvml_peak_gb is None else f"{r.nvml_peak_gb:5.2f}"
        logger.info("B=%2d  %7.3f clips/s  x%.2f vs B=1  allocated %5.2f / reserved %5.2f / NVML %s GB (estimate %5.2f "
                    "GB)  sub-batches %s  %s", r.batch_size, r.clips_per_second, speedup[r.batch_size], r.peak_gb,
                    r.reserved_gb, nvml, r.estimate_gb, r.sub_batches, r.error)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
