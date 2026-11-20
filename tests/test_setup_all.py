"""scripts/setup_all.sh, which wraps release/QUICKSTART.md step 2: valid bash, and its --dry-run prints every step,
runs none, and reports the preflight checks (no conda, network or GPU needed here)."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "setup_all.sh"


def _run(*args: str, cwd: Path, path: str = "/usr/bin:/bin") -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, check=False,
                          env={"PATH": path, "HOME": str(cwd)})


def _fake_root(tmp_path: Path) -> tuple[Path, str]:
    """A directory that passes the preflight: the root files, the submodule directory and a `conda` on PATH."""
    root, bin_dir = tmp_path / "root", tmp_path / "bin"
    (root / "third_party/do-as-i-do/reconstruction/modules/sam-3d-objects").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (root / "uv.lock").write_text("")
    bin_dir.mkdir()
    # the fake conda lists no env and fails on every other call
    (bin_dir / "conda").write_text('#!/bin/sh\nif [ "$1 $2" = "env list" ]; then echo "# conda environments:"; '
                                   'echo "base  /opt/conda"; exit 0; fi\nexit 99\n')
    (bin_dir / "conda").chmod(0o755)
    return root, f"{bin_dir}:/usr/bin:/bin"


def test_the_script_parses() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, check=False).returncode == 0


def test_dry_run_prints_quickstart_step_2_and_runs_nothing(tmp_path: Path) -> None:
    root, path = _fake_root(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    r = _run("--dry-run", "--hawor", str(tmp_path / "hawor"), "--hands23", str(tmp_path / "h23"), cwd=root, path=path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "preflight: ok" in out and "FAIL" not in out
    for step in ("uv sync --frozen --python 3.12 --extra dev --extra frontend",
                 "conda create -y -n sam3d --file release/env/sam3d.conda.txt",
                 "conda create -y -n hawor --file release/env/hawor.conda.txt",
                 "git clone https://github.com/malik-group/HaWoR.git",
                 "2c3fa0c06e54a1b192c1ff692eaa89d526cd6317",
                 "nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae",
                 "pip install --no-deps -e third_party/do-as-i-do/reconstruction/modules/sam-3d-objects",
                 "torch-scatter==2.1.2", "setup.py install",
                 "a2f4a8771ab77e8411c26b27f24f9489a28a2453", "--extra viser",
                 "conda run -n sam3d python -m pip install --no-deps",
                 "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl", FLASH_ATTN_SHA256):
        assert step in out, step
    sam3d = [i for i, line in enumerate(out.splitlines()) if "sam-3d-objects" in line or "flash_attn" in line]
    assert len(sam3d) == 2 and "sam-3d-objects" in out.splitlines()[sam3d[0]]   # flash_attn after the sam3d build
    assert sorted(tmp_path.rglob("*")) == before                       # nothing created


def test_a_failed_conda_listing_is_not_an_empty_list(tmp_path: Path) -> None:
    """If `conda env list` fails, the script cannot know whether sam3d or hawor exists: the preflight fails instead
    of going on to `conda create -y`, which would replace an existing env. An env that conda lists is refused too."""
    root, path = _fake_root(tmp_path)
    conda = tmp_path / "bin" / "conda"
    conda.write_text("#!/bin/sh\nexit 99\n")
    args = ("--hawor", str(tmp_path / "hawor"), "--hands23", str(tmp_path / "h23"))
    r = _run("--dry-run", *args, cwd=root, path=path)
    assert "preflight: FAIL" in r.stdout and "conda env list failed" in r.stdout and "preflight: ok" not in r.stdout
    live = _run(*args, cwd=root, path=path)
    assert live.returncode == 2 and "conda env list failed" in live.stderr and "conda create" not in live.stdout
    conda.write_text('#!/bin/sh\nif [ "$1 $2" = "env list" ]; then echo "sam3d  /x/envs/sam3d"; exit 0; fi\nexit 99\n')
    r = _run("--dry-run", *args, cwd=root, path=path)
    assert "a conda env named sam3d already exists" in r.stdout
    r = _run("--dry-run", "--replace-envs", *args, cwd=root, path=path)       # the listing is not needed then
    assert "preflight: ok" in r.stdout


FLASH_ATTN_SHA256 = "3d41b2fc55753faa7f45d6568ea73a96b96afb48b82994ab9b49bcbcb6c87588"


def test_one_flash_attn_wheel_everywhere() -> None:
    """The sam3d lock, its exporter, setup_all.sh and QUICKSTART step 2 name the same wheel URL and sha256."""
    import importlib.util
    import re

    spec = importlib.util.spec_from_file_location("export_env_locks", REPO / "release/env/export_env_locks.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    url = ex.FLASH_ATTN_WHEEL
    assert url.endswith("#sha256=" + FLASH_ATTN_SHA256) and url.startswith("https://github.com/Dao-AILab/")
    lock = (REPO / "release/env/sam3d.pip.txt").read_text()
    assert ex.ALWAYS_BUILD["sam3d"][0] in lock.splitlines()
    assert not re.search(r"^flash.attn==", lock, re.MULTILINE | re.IGNORECASE)   # never a pin pip would build
    assert f'FLASH_ATTN_WHEEL="{url}"' in SCRIPT.read_text()
    assert url in (REPO / "release/QUICKSTART.md").read_text()


def test_without_hands23_the_optional_venv_is_skipped(tmp_path: Path) -> None:
    root, path = _fake_root(tmp_path)
    r = _run("--dry-run", cwd=root, path=path)
    assert r.returncode == 0 and "detectron2" not in r.stdout


def test_dry_run_reports_every_failed_preflight_check_and_still_prints_the_plan(tmp_path: Path) -> None:
    r = _run("--dry-run", cwd=tmp_path)                                  # no root files, no submodule, no conda
    assert r.returncode == 2, r.stdout + r.stderr
    for msg in ("run from the repository root", "the DO AS I DO submodule is missing", "conda not found"):
        assert f"preflight: FAIL: {msg}" in r.stdout and msg in r.stderr, msg
    assert "uv sync --frozen --python 3.12" in r.stdout and "setup.py install" in r.stdout
    assert "done. Next" not in r.stdout and list(tmp_path.iterdir()) == []


def test_bad_arguments_and_a_wrong_directory_are_refused(tmp_path: Path) -> None:
    assert _run("--bogus", cwd=tmp_path).returncode == 2
    r = _run(cwd=tmp_path)                                               # not the repository root, not a dry run
    assert r.returncode == 2 and "repository root" in r.stderr


def test_an_existing_env_is_never_replaced_without_the_flag(tmp_path: Path) -> None:
    """`conda create -y -n sam3d` replaces an env of that name; the script refuses unless --replace-envs."""
    root, path = _fake_root(tmp_path)
    conda = Path(path.split(":")[0]) / "conda"
    conda.write_text("#!/bin/sh\nif [ \"$1 $2\" = \"env list\" ]; then\n"
                     "printf '# conda environments:\\nbase  *  /x\\nhawor     /x/envs/hawor\\n'; exit 0; fi\nexit 99\n")
    r = _run("--dry-run", cwd=root, path=path)
    assert r.returncode == 2 and "preflight: FAIL: a conda env named hawor already exists" in r.stdout, r.stdout
    assert "sam3d already exists" not in r.stdout
    r = _run("--dry-run", "--replace-envs", cwd=root, path=path)
    assert r.returncode == 0 and "preflight: ok" in r.stdout and "are replaced" in r.stdout, r.stdout + r.stderr
    r = _run(cwd=root, path=path)                                        # a real run stops before its first step
    assert r.returncode == 2 and "hawor already exists" in r.stderr and "uv sync" not in r.stdout
