"""The documented policy entry points run as scripts, `uv run python run/pipeline/<script>.py
+policy=default policy.run=<name>`, and not only as `run.pipeline.*` modules under pytest.

Each script runs in a subprocess with the test interpreter (the project environment `uv run` uses), from a working
directory outside the repository and without PYTHONPATH, through Hydra's `--cfg job` path: the module imports and the
configuration composes, but no job (training or evaluation) runs. CPU only."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.policy.env_fixtures import REPO_ROOT

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("script", ["eval_policy.py", "train_policy.py"])
def test_policy_script_composes_its_config_as_a_script(script: str, tmp_path: Path) -> None:
    """`python run/pipeline/<script> +policy=default policy.run=entry_probe --cfg job` exits 0 and prints the composed
    policy group with the run name; nothing is written under the repository's outputs."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, str(REPO_ROOT / "run" / "pipeline" / script), "+policy=default",
               "policy.run=entry_probe", f"paths.outputs={tmp_path / 'outputs'}",
               f"hydra.run.dir={tmp_path / 'hydra'}", "--cfg", "job"]
    done = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600, check=False)
    logger.info("%s --cfg job: exit %d, %d bytes of output", script, done.returncode, len(done.stdout))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "policy:" in done.stdout and "run: entry_probe" in done.stdout
    assert "law_min_clips" in done.stdout
    assert not (tmp_path / "outputs").exists()
