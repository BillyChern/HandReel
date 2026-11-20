"""packages/alongray declares the three console entry points the paper's supplement names (`alongray`, `alongray-pool`
and the validator `alongray-coverage`), and each one answers --help. No install needed: the package is pure Python."""
from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "packages" / "alongray"
SCRIPTS = tomllib.loads((PKG / "pyproject.toml").read_text())["project"]["scripts"]


def test_the_supplement_s_entry_points_are_declared() -> None:
    assert SCRIPTS == {"alongray": "alongray.cli:main", "alongray-pool": "alongray.pool:main",
                       "alongray-coverage": "alongray.coverage:main"}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_each_entry_point_answers_help(name: str) -> None:
    module, func = SCRIPTS[name].split(":")
    code = (f"import sys; sys.path.insert(0, {str(PKG / 'src')!r}); import {module} as m; "
            f"sys.argv = [{name!r}, '--help']; sys.exit(m.{func}())")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "usage:" in r.stdout.lower()


def test_coverage_scores_nothing_and_reads_a_protocol() -> None:
    """The validator's own help text: it compares conventions and unit flow, and prints no score."""
    sys.path.insert(0, str(PKG / "src"))
    try:
        cov = importlib.import_module("alongray.coverage")
    finally:
        sys.path.remove(str(PKG / "src"))
    doc = (cov.__doc__ or "").lower()
    assert "protocol" in doc and "score" in doc
