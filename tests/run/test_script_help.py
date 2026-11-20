"""`--help` of the scripts that are libraries or start a long campaign: a usage line, never a run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_LIBRARY_MODULES = ["run/pipeline/_hawor_driver_models.py", "run/pipeline/_hawor_trace.py", "run/study/e1_stats.py"]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-s", "-B", str(_REPO / script), *args], capture_output=True, text=True,
                          timeout=120, cwd=_REPO, check=False)


@pytest.mark.parametrize("script", _LIBRARY_MODULES)
def test_a_library_module_prints_a_usage_line(script: str) -> None:
    help_run = _run(script, "--help")
    assert help_run.returncode == 0 and help_run.stdout.startswith("usage: not a command."), help_run.stderr
    assert _run(script, "--bogus").returncode == 2


def test_the_mutant_check_answers_help_without_running_mutants() -> None:
    """`--help` used to be ignored and started the whole campaign (one pytest run per mutant)."""
    help_run = _run("packages/alongray/tools/mutate_split.py", "--help")
    assert help_run.returncode == 0 and help_run.stdout.startswith("usage: mutate_split.py"), help_run.stderr
    assert "BASELINE_unmutated" not in help_run.stdout and "rc=" not in help_run.stdout      # no mutant row
    assert _run("packages/alongray/tools/mutate_split.py", "--bogus").returncode == 2
