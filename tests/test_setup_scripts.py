"""run/setup/*.sh read FASTDO_CONDA_BASE (the config's paths.conda_base), with no literal miniforge root: unset, they
stop with a message naming it."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "run" / "setup"
SCRIPTS = ("build_baseline_envs.sh", "verify_baseline_envs.sh")
_DEFAULT = re.compile(r'^CONDA_BASE="\$\{FASTDO_CONDA_BASE:\?[^}]+\}"', re.MULTILINE)


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_literal_conda_root_outside_the_default(name: str) -> None:
    code = [ln for ln in (SETUP / name).read_text().splitlines() if not ln.lstrip().startswith("#")]
    literal = [ln for ln in code if "miniforge3" in ln or "/opt/" in ln]
    assert literal == [], literal
    assert any('source "$CONDA_BASE/etc/profile.d/conda.sh"' in ln for ln in code)


@pytest.mark.parametrize("name", SCRIPTS)
@pytest.mark.parametrize("env,expected", [({"FASTDO_CONDA_BASE": "/x/conda"}, "/x/conda"), ({}, None)])
def test_the_conda_base_line_follows_the_variable(name: str, env: dict[str, str], expected: str | None) -> None:
    line = _DEFAULT.search((SETUP / name).read_text())
    assert line is not None
    run = subprocess.run(["bash", "-c", f'{line.group(0)}; printf %s "$CONDA_BASE"'], env={"PATH": "/usr/bin:/bin", **env},
                         capture_output=True, text=True, check=False)
    if expected is None:
        assert run.returncode != 0 and "set FASTDO_CONDA_BASE" in run.stderr
    else:
        assert run.returncode == 0 and run.stdout == expected


HELP_SCRIPTS = (SETUP / "build_baseline_envs.sh", SETUP / "make_mjwarp314_site.sh", SETUP / "verify_baseline_envs.sh",
                SETUP.parents[1] / "release" / "env" / "verify.sh")


@pytest.mark.parametrize("flag", ["--help", "-h"])
@pytest.mark.parametrize("script", HELP_SCRIPTS, ids=lambda p: p.name)
def test_help_prints_usage_before_any_check_or_side_effect(script: Path, flag: str, tmp_path: Path) -> None:
    """--help needs no FASTDO_CONDA_BASE and creates nothing, here (the cwd) or under a conda root that is set."""
    work, conda = tmp_path / "cwd", tmp_path / "conda"
    work.mkdir()
    conda.mkdir()
    root_temp = SETUP.parents[1] / "temp"
    had_temp = root_temp.exists()
    for env in ({"PATH": "/usr/bin:/bin"}, {"PATH": "/usr/bin:/bin", "FASTDO_CONDA_BASE": str(conda)}):
        run = subprocess.run(["bash", str(script), flag], cwd=work, env=env, capture_output=True, text=True,
                             check=False, timeout=30)
        assert run.returncode == 0, run.stderr
        assert "Usage:" in run.stdout, run.stdout
        assert "FASTDO_CONDA_BASE to your conda" not in run.stderr
    assert list(work.iterdir()) == [] and list(conda.iterdir()) == []
    assert root_temp.exists() == had_temp


def test_build_baseline_envs_refuses_an_unknown_target_before_any_side_effect(tmp_path: Path) -> None:
    run = subprocess.run(["bash", str(SETUP / "build_baseline_envs.sh"), "bogus"], cwd=tmp_path,
                         env={"PATH": "/usr/bin:/bin", "FASTDO_CONDA_BASE": str(tmp_path)}, capture_output=True,
                         text=True, check=False, timeout=30)
    assert run.returncode == 1 and "usage" in run.stderr
    assert list(tmp_path.iterdir()) == []
