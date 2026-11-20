"""Where the tests find the baseline's hawor conda env and HaWoR checkout; tests that need them skip otherwise."""

from __future__ import annotations

import os
from pathlib import Path

HAWOR_PYTHON = Path(os.environ.get("FASTDO_HAWOR_PYTHON", "/opt/fastdo/miniforge3/envs/hawor/bin/python"))


def hawor_root() -> Path | None:
    """`$FASTDO_HAWOR_ROOT` when set, else None."""
    value = os.environ.get("FASTDO_HAWOR_ROOT", "")
    return Path(value) if value else None


def cpu_subprocess_env() -> dict[str, str]:
    """This process's environment for a CPU-only hawor-python helper: no PYTHONPATH, no weights-only override."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")}
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    return env
