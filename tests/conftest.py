import importlib.util

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cuda_budget() -> None:
    """Shared-host rule: every test process caps its torch threads (FASTDO_CPU_THREADS, default 4) and CUDA allocations
    (FASTDO_GPU_MEM_GB, default 12 GiB). Without torch installed there is nothing to cap, so tests that need only numpy
    and scipy (e.g. tests/run/test_e1_stats.py) run in an environment without it."""
    if importlib.util.find_spec("torch") is None:
        return
    from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget

    apply_cpu_budget()
    apply_gpu_budget()
