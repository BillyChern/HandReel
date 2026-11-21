"""scripts/fetch_baseline.sh: the DO AS I DO baseline at its pinned commit, in a git clone that records no submodule
commit for third_party/do-as-i-do (the public repository), in a copy without git metadata (an archive), and in a
checkout that records the submodule commit.

These tests run the script in a scratch copy of the repository root and never touch the network: the dry runs, the
refusals, and the check of a checkout at the wrong commit. The real clone is checked by
`test_fetches_and_verifies_the_pinned_baseline`, which needs network access and runs only with FASTDO_TEST_NETWORK=1.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fetch_baseline.sh"
PIN = "824591b808c342b20079c3b4198a8c2bdf88c74e"
URL = "https://github.com/malik-group/do-as-i-do.git"


def _env() -> dict[str, str]:
    """The caller's environment without GIT_* variables, so no git command here can reach another repository."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _root(tmp_path: Path) -> Path:
    """A scratch repository root as an archive unpacks it: root files, the scripts, an empty baseline directory."""
    root = tmp_path / "handreel"
    (root / "scripts").mkdir(parents=True)
    (root / "third_party" / "do-as-i-do").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    shutil.copy2(SCRIPT, root / "scripts" / "fetch_baseline.sh")
    shutil.copy2(REPO / "scripts" / "setup_all.sh", root / "scripts" / "setup_all.sh")
    return root


def _run(root: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "scripts/fetch_baseline.sh", *args], cwd=root, env=env or _env(),
                          capture_output=True, text=True, timeout=600, check=False)


def test_the_pin_and_url_are_the_ones_the_script_and_readme_name() -> None:
    # 2026-11-21: the release records no submodule for third_party/do-as-i-do and ships no .gitmodules (it named only
    # that submodule); the script clones the baseline itself
    text = SCRIPT.read_text()
    assert re.search(rf"^PIN={PIN}$", text, re.MULTILINE)
    assert URL in text
    assert PIN in (REPO / "README.md").read_text()
    assert not (REPO / ".gitmodules").exists()


def test_the_pin_is_the_gitlink_of_this_commit() -> None:
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout (an archive): there is no gitlink to compare with")
    out = subprocess.run(["git", "ls-tree", "HEAD", "third_party/do-as-i-do"], cwd=REPO, env=_env(),
                         capture_output=True, text=True, check=True).stdout.split()
    if not out:
        pytest.skip("this commit records no submodule commit for third_party/do-as-i-do (the public repository)")
    assert out[:3] == ["160000", "commit", PIN]


def test_dry_run_prints_the_pinned_clone_and_fetches_nothing(tmp_path: Path) -> None:
    root = _root(tmp_path)
    r = _run(root, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert f"+ git clone --no-checkout {URL} third_party/do-as-i-do" in r.stdout
    assert f"+ git -C third_party/do-as-i-do checkout --detach {PIN}" in r.stdout
    assert "+ git -C third_party/do-as-i-do submodule update --init --recursive" in r.stdout
    assert list((root / "third_party" / "do-as-i-do").iterdir()) == []


def test_a_mirror_url_replaces_the_clone_url_only(tmp_path: Path) -> None:
    root = _root(tmp_path)
    r = _run(root, "--dry-run", env={**_env(), "FASTDO_BASELINE_URL": "https://mirror.example.invalid/do-as-i-do.git"})
    assert r.returncode == 0, r.stderr
    assert "+ git clone --no-checkout https://mirror.example.invalid/do-as-i-do.git third_party/do-as-i-do" in r.stdout
    assert f"checkout --detach {PIN}" in r.stdout


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(root), *args],
                   env=_env(), capture_output=True, text=True, check=True)


def test_a_git_clone_without_the_gitlink_clones_the_baseline(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "third_party" / "do-as-i-do").rmdir()          # git keeps no empty directory
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "x")
    r = _run(root, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert f"+ git clone --no-checkout {URL} third_party/do-as-i-do" in r.stdout
    assert f"+ git -C third_party/do-as-i-do checkout --detach {PIN}" in r.stdout


def test_a_git_checkout_with_the_gitlink_updates_the_submodule(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{PIN},third_party/do-as-i-do")
    r = _run(root, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "+ git submodule update --init --recursive -- third_party/do-as-i-do" in r.stdout
    assert "git clone" not in r.stdout


def test_outside_the_repository_root_it_stops(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "pyproject.toml").unlink()
    assert _run(root, "--dry-run").returncode == 2


def test_a_non_empty_directory_that_is_not_a_checkout_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "third_party" / "do-as-i-do" / "stray.txt").write_text("x")
    r = _run(root)
    assert r.returncode == 1
    assert "not empty and not a git checkout" in r.stderr


def test_a_checkout_at_another_commit_fails_the_check(tmp_path: Path) -> None:
    root = _root(tmp_path)
    dest = root / "third_party" / "do-as-i-do"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(dest)]
    subprocess.run([*git, "init", "-q"], env=_env(), check=True)
    (dest / "README").write_text("not the baseline\n")
    subprocess.run([*git, "add", "README"], env=_env(), check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], env=_env(), check=True)
    for args in (("--check",), ()):
        r = _run(root, *args)
        assert r.returncode == 1, args
        assert f"not the pinned {PIN}" in r.stderr


def test_without_a_checkout_the_check_fails(tmp_path: Path) -> None:
    r = _run(_root(tmp_path), "--check")
    assert r.returncode == 1
    assert "is not a git checkout" in r.stderr


@pytest.mark.skipif(os.environ.get("FASTDO_TEST_NETWORK") != "1",
                    reason="needs network access to GitHub and GitLab; set FASTDO_TEST_NETWORK=1 to run it")
def test_fetches_and_verifies_the_pinned_baseline(tmp_path: Path) -> None:
    root = _root(tmp_path)
    r = _run(root)
    assert r.returncode == 0, r.stderr
    assert f"ok: third_party/do-as-i-do at {PIN}" in r.stdout
    assert (root / "third_party/do-as-i-do/reconstruction/modules/sam-3d-objects").is_dir()
    assert _run(root, "--check").returncode == 0
