"""R5: release/env/verify.sh checks what QUICKSTART builds and nothing else; release/env/check_weights.py finds each
weight file where the pipeline reads it. Fake conda and uv executables; tiny files; no GPU, no network."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VERIFY = REPO / "release" / "env" / "verify.sh"
_SPEC = importlib.util.spec_from_file_location("check_weights", REPO / "release" / "env" / "check_weights.py")
CW = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = CW
_SPEC.loader.exec_module(CW)


def _manifest(payload: bytes) -> dict:
    return {"models": [
        {"key": "m", "files": [{"file": "w.pt", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                                "expected_at": "$HF_HOME/hub/w.pt"}]},
        {"key": "h", "files": [{"file": "h.ckpt", "size": 1, "sha256": "0" * 64,
                                "expected_at": "$FASTDO_HAWOR_ROOT/weights/h.ckpt"}]},
        {"key": "hands23", "files": [{"file": "x", "size": 1, "sha256": "0" * 64,
                                      "expected_at": "$FASTDO_HANDS23_ROOT/x"}]}]}


def _hub(torch_home: Path) -> None:
    from fastdo.model_pins import COTRACKER_HUB_COMMIT

    for code in (f"facebookresearch_co-tracker_{COTRACKER_HUB_COMMIT}", "facebookresearch_dinov2_main"):
        (torch_home / "hub" / code).mkdir(parents=True)
        (torch_home / "hub" / code / "hubconf.py").write_text("")


def test_check_weights_reports_each_file(tmp_path: Path) -> None:
    payload = b"weights"
    env = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch")}
    results = {line: ok for ok, line in CW.checks(_manifest(payload), env)}
    assert any("missing at" in line and not ok for line, ok in results.items())
    assert any("FASTDO_HAWOR_ROOT is not set" in line for line in results)        # unset: a failure, not a guess
    assert not any(line.startswith("hands23/") for line in results)                # opt-in only
    (tmp_path / "hf" / "hub").mkdir(parents=True)
    (tmp_path / "hf" / "hub" / "w.pt").write_bytes(payload)
    (tmp_path / "hawor" / "weights").mkdir(parents=True)
    (tmp_path / "hawor" / "weights" / "h.ckpt").write_bytes(b"x")
    _hub(tmp_path / "torch")
    env["FASTDO_HAWOR_ROOT"] = str(tmp_path / "hawor")
    assert all(ok for ok, _ in CW.checks(_manifest(payload), env))
    sha = [line for ok, line in CW.checks(_manifest(payload), env, sha=True) if not ok]
    assert sha == [f"h/h.ckpt: {tmp_path / 'hawor/weights/h.ckpt'} sha256 differs from {'0' * 64}"]
    (tmp_path / "hf" / "hub" / "w.pt").write_bytes(payload + b"!")
    assert any("bytes, expected" in line for ok, line in CW.checks(_manifest(payload), env) if not ok)
    assert any(line.startswith("hands23/") for _, line in CW.checks(_manifest(payload), env, with_hands23=True))



def test_a_torch_hub_code_dir_is_ok_only_when_its_pin_file_matches(tmp_path: Path) -> None:
    """A cached `facebookresearch_dinov2_main` (or any hub dir) without fetch_weights.py's pin file is reported
    UNVERIFIED, never plain OK; a pin file of another commit, or files changed since the unpack, fail."""
    from fastdo import model_pins

    env = {"TORCH_HOME": str(tmp_path / "torch")}
    _hub(tmp_path / "torch")
    hub = tmp_path / "torch" / "hub"
    dinov2 = hub / "facebookresearch_dinov2_main"
    cotracker = hub / f"facebookresearch_co-tracker_{model_pins.COTRACKER_HUB_COMMIT}"

    def hub_lines() -> dict[str, tuple[bool, str]]:
        return {name: next((ok, line) for ok, line in CW.checks({"models": []}, env) if name in line)
                for name in (dinov2.name, cotracker.name)}

    for ok, line in hub_lines().values():
        assert ok and " UNVERIFIED: " in line and CW.label(ok, line) == "UNVERIFIED "
    model_pins.write_code_pin(dinov2, "facebookresearch/dinov2", model_pins.DINOV2_HUB_COMMIT)
    model_pins.write_code_pin(cotracker, "facebookresearch/co-tracker", model_pins.COTRACKER_HUB_COMMIT)
    (dinov2 / "__pycache__").mkdir()
    (dinov2 / "__pycache__" / "hubconf.cpython-311.pyc").write_bytes(b"import writes this")   # not content
    for ok, line in hub_lines().values():
        assert ok and "UNVERIFIED" not in line and CW.label(ok, line) == "OK   "
    (dinov2 / "hubconf.py").write_text("# changed")
    ok, line = hub_lines()[dinov2.name]
    assert not ok and "mismatch: files changed" in line and CW.label(ok, line) == "FAIL "
    model_pins.write_code_pin(cotracker, "facebookresearch/co-tracker", "0" * 40)
    ok, line = hub_lines()[cotracker.name]
    assert not ok and "mismatch: unpacked as" in line


def test_an_hf_snapshot_link_to_a_wrong_blob_fails(tmp_path: Path) -> None:
    """The HF cache layout: `snapshots/<rev>/<file>` is a link to `blobs/<sha>`. A blob of the wrong size fails the
    plain check, and one of the right size but another digest fails --sha256."""
    payload = b"weights"
    snap = tmp_path / "hf" / "hub" / "w.pt"
    blob = tmp_path / "hf" / "blobs" / "b"
    blob.parent.mkdir(parents=True)
    snap.parent.mkdir(parents=True)
    snap.symlink_to(blob)
    env = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "torch")}
    only = {"m"}
    blob.write_bytes(payload + b"!")
    assert [line for ok, line in CW.checks(_manifest(payload), env, only=only) if not ok] == [
        f"m/w.pt: {snap} has {len(payload) + 1} bytes, expected {len(payload)}"]
    blob.write_bytes(b"x" * len(payload))
    assert all(ok for ok, _ in CW.checks(_manifest(payload), env, only=only))
    assert [line for ok, line in CW.checks(_manifest(payload), env, sha=True, only=only) if not ok] == [
        f"m/w.pt: {snap} sha256 differs from {hashlib.sha256(payload).hexdigest()}"]

def test_the_shipped_manifest_paths_all_expand(tmp_path: Path) -> None:
    import json

    manifest = json.loads((REPO / "release" / "models.json").read_text())
    env = {k: str(tmp_path / k) for k in ("HF_HOME", "TORCH_HOME", "FASTDO_HAWOR_ROOT", "FASTDO_MANO_DIR",
                                          "FASTDO_MANO_LEFT_DIR", "FASTDO_HANDS23_ROOT")}
    lines = [line for _, line in CW.checks(manifest, env, with_hands23=True)]
    assert lines and not [line for line in lines if "is not set" in line]


def _fakes(tmp_path: Path, conda_rc: int) -> dict[str, str]:
    bin_dir, conda = tmp_path / "bin", tmp_path / "conda" / "bin"
    bin_dir.mkdir()
    conda.mkdir(parents=True)
    log = tmp_path / "calls.log"
    (bin_dir / "uv").write_text(f"#!/bin/sh\necho \"uv $*\" >> '{log}'\nexit 0\n")
    (conda / "conda").write_text(f"#!/bin/sh\necho \"conda $*\" >> '{log}'\nexit {conda_rc}\n")
    for f in (bin_dir / "uv", conda / "conda"):
        f.chmod(0o755)
    hawor = tmp_path / "hawor" / "lib" / "models"
    hawor.mkdir(parents=True)
    (hawor / "hawor.py").write_text("")
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "FASTDO_CONDA_BASE": str(tmp_path / "conda"),
            "FASTDO_HAWOR_ROOT": str(tmp_path / "hawor"), "HOME": str(tmp_path)}


@pytest.mark.parametrize("conda_rc,expected", [(0, 0), (1, 1)])
def test_verify_checks_only_the_quickstart_envs(tmp_path: Path, conda_rc: int, expected: int) -> None:
    env = _fakes(tmp_path, conda_rc)
    run = subprocess.run(["bash", str(VERIFY), "--no-gpu"], env=env, capture_output=True, text=True, timeout=120,
                         check=False)
    assert run.returncode == expected, run.stdout + run.stderr
    calls = (tmp_path / "calls.log").read_text()
    envs = {line.split()[3] for line in calls.splitlines() if line.startswith("conda run -n")}
    assert envs == {"sam3d", "hawor"}                                   # never sam3, tapnet or retargeting
    assert "release/env/check_weights.py" in calls and "cuda.is_available" not in calls
    assert "--with-hands23" not in calls                                # Hands23 only with FASTDO_HANDS23_ROOT


def test_verify_needs_the_conda_root(tmp_path: Path) -> None:
    env = {k: v for k, v in _fakes(tmp_path, 0).items() if k != "FASTDO_CONDA_BASE"}
    run = subprocess.run(["bash", str(VERIFY)], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode != 0 and "FASTDO_CONDA_BASE" in run.stderr


def test_verify_script_is_executable_text() -> None:
    assert os.access(VERIFY, os.R_OK) and VERIFY.read_text().startswith("#!/usr/bin/env bash")


def test_sha256sum_list_feeds_sha256sum_c(tmp_path: Path) -> None:
    """S4: digests are checked by `sha256sum -c`, not compared by eye."""
    import shutil

    payload = b"weights"
    env = {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "t"), "FASTDO_HAWOR_ROOT": str(tmp_path)}
    (tmp_path / "hf" / "hub").mkdir(parents=True)
    (tmp_path / "hf" / "hub" / "w.pt").write_bytes(payload)
    lines = CW.sha256sum_list(_manifest(payload), env, only={"m"})
    assert lines == [f"{hashlib.sha256(payload).hexdigest()}  {tmp_path / 'hf/hub/w.pt'}"]
    if shutil.which("sha256sum"):
        ok = subprocess.run(["sha256sum", "-c", "-"], input="\n".join(lines) + "\n", text=True, capture_output=True,
                            check=False)
        assert ok.returncode == 0, ok.stdout + ok.stderr
        (tmp_path / "hf" / "hub" / "w.pt").write_bytes(b"other")
        bad = subprocess.run(["sha256sum", "-c", "-"], input="\n".join(lines) + "\n", text=True, capture_output=True,
                             check=False)
        assert bad.returncode != 0
    with pytest.raises(SystemExit, match="FASTDO_MANO_DIR"):
        import json

        CW.sha256sum_list(json.loads((REPO / "release" / "models.json").read_text()), {}, only={"mano"})


_FSPEC = importlib.util.spec_from_file_location("check_flash_attn", REPO / "release" / "env" / "check_flash_attn.py")
FA = importlib.util.module_from_spec(_FSPEC)
_FSPEC.loader.exec_module(FA)
MISSING = (None, "ModuleNotFoundError: No module named 'flash_attn'")


@pytest.mark.parametrize("name", ["NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB", "NVIDIA H200"])
def test_flash_attn_missing_on_a_card_sam3d_forces_it_to_fails(name: str) -> None:
    """F1: SAM 3D sets ATTN_BACKEND=flash_attn when the device-0 name contains A100/H100/H200."""
    ok, line = FA.decide((name, (9, 0)), *MISSING)
    assert not ok and line.startswith("FAIL") and name in line and "No module named 'flash_attn'" in line
    assert "release/env/sam3d.pip.txt" in line
    ok, line = FA.decide((name, (9, 0)), "2.8.3", None)
    assert ok and line.startswith("OK") and "2.8.3" in line and "pins" not in line


def test_flash_attn_elsewhere_is_a_note_not_a_failure() -> None:
    ok, line = FA.decide(("NVIDIA GeForce RTX 4090", (8, 9)), *MISSING)
    assert ok and line.startswith("NOTE") and "sdpa" in line and "not run" in line
    ok, line = FA.decide(("Tesla V100-SXM2-32GB", (7, 0)), *MISSING)
    assert ok and line.startswith("OK") and "sdpa" in line
    ok, line = FA.decide(None, *MISSING)                                  # --no-gpu
    assert ok and line.startswith("NOTE") and "A100/H100/H200" in line
    ok, line = FA.decide(("NVIDIA H100 80GB HBM3", (9, 0)), "2.7.4", None)
    assert ok and "pins 2.8.3" in line


def test_flash_attn_status_reports_the_import_error() -> None:
    version, error = FA.flash_attn_status("fastdo_no_such_module_for_this_test")
    assert version is None and error.startswith("ModuleNotFoundError")
    version, error = FA.flash_attn_status("json")
    assert error is None and version


def test_the_sam3d_rule_matches_sam3d_objects() -> None:
    """The card list is SAM 3D's own (skips without the DO AS I DO baseline)."""
    src = (REPO / "third_party/do-as-i-do/reconstruction/modules/sam-3d-objects/sam3d_objects/pipeline/"
           "inference_pipeline.py")
    if not src.exists():
        pytest.skip("the DO AS I DO baseline is not in third_party/do-as-i-do (bash scripts/fetch_baseline.sh)")
    text = src.read_text()
    assert all(f'"{card}" in gpu_name' in text for card in FA.FLASH_ATTN_CARDS)
    assert 'os.environ["ATTN_BACKEND"] = "flash_attn"' in text


def test_verify_runs_the_flash_attn_check_in_sam3d(tmp_path: Path) -> None:
    env = _fakes(tmp_path, 0)
    run = subprocess.run(["bash", str(VERIFY), "--no-gpu"], env=env, capture_output=True, text=True, timeout=120,
                         check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert "conda run -n sam3d python release/env/check_flash_attn.py --no-gpu" in calls
    (tmp_path / "calls.log").unlink()
    run = subprocess.run(["bash", str(VERIFY)], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert "conda run -n sam3d python release/env/check_flash_attn.py" in (tmp_path / "calls.log").read_text()
