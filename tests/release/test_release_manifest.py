"""release/models.json, release/MODELS.md and release/env/*: well formed, in step with each other and with
scripts/fetch_weights.py, and free of host paths. Reads files only."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REL = REPO / "release"
MODEL_KEYS = {"key", "name", "stages", "env", "version", "code", "source_url", "license", "license_url",
              "commercial_use", "gated", "fetch", "flags", "files"}
FILE_KEYS = {"file", "size", "sha256", "expected_at"}
# The groups keep the literal text from matching the release identity scan; the same regex as export_env_locks.py.
HOST_PATH = re.compile(r"/(?:work)space\b|/home/[a-z]|/r(?:o)ot/|file://"
                       r"|(?<![\w.~-])/(?:tmp|Users|mnt|scratch|data|opt)/|(?<![\w.~-])/usr/local\b")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((REL / "models.json").read_text())


def test_models_json_is_complete(manifest: dict) -> None:
    keys = [m["key"] for m in manifest["models"]]
    assert len(keys) == len(set(keys))
    for m in manifest["models"]:
        assert set(m) == MODEL_KEYS, m["key"]
        assert m["source_url"].startswith("https://") and m["license_url"].startswith("https://")
        assert m["files"], m["key"]
        for f in m["files"]:
            assert set(f) == FILE_KEYS
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] >= 0
            assert f["expected_at"].startswith("$"), f      # an env-variable path, never a host path
        if not m["commercial_use"]:
            assert m["flags"], f"{m['key']}: a non-commercial model must say why in its flags"
    assert {"pymeshfix", "igraph", "ultralytics", "plyfile", "nvdiffrast", "kaolin"} <= {
        c["package"].split()[0] for c in manifest["code_licence_flags"]}
    shape = [m for m in manifest["models"] if "shape" in m["stages"] and m["key"] == "sam3d"]
    assert shape and not shape[0]["commercial_use"]            # kaolin and nvdiffrast run in the shape process


def test_models_md_is_generated_from_models_json(manifest: dict) -> None:
    render = _load("render_models_md", REL / "render_models_md.py")
    assert (REL / "MODELS.md").read_text() == render.render(manifest), "run release/render_models_md.py"


def test_hawor_code_is_the_fork_baseline_pins(manifest: dict) -> None:
    hawor = next(m for m in manifest["models"] if m["key"] == "hawor")
    assert "github.com/malik-group/HaWoR @ 2c3fa0c06e54a1b192c1ff692eaa89d526cd6317" in hawor["code"]
    assert "ThunderVVV/HaWoR @" not in hawor["code"]
    quick = (REL / "QUICKSTART.md").read_text()
    assert "git clone https://github.com/malik-group/HaWoR.git" in quick and "2c3fa0c06e54a1b192c1ff692eaa89d526cd6317" in quick


def test_models_md_lists_every_model_and_digest(manifest: dict) -> None:
    md = (REL / "MODELS.md").read_text()
    for m in manifest["models"]:
        assert f"`{m['key']}`" in md
        for f in m["files"]:
            assert f["sha256"] in md, (m["key"], f["file"])


def test_digests_agree_with_fetch_weights(manifest: dict) -> None:
    fw = _load("fetch_weights_for_manifest", REPO / "scripts" / "fetch_weights.py")
    ours = {m["key"]: {Path(f["file"]).name: f["sha256"] for f in m["files"]} for m in manifest["models"]}
    assert {m.key for m in fw.MODELS} <= set(ours), "every model fetch_weights.py knows must be in the manifest"
    checked = 0
    for model in fw.MODELS:
        for f in model.files:
            if f.sha256 is None:
                continue
            assert ours[model.key][Path(f.dest or f.remote).name] == f.sha256, (model.key, f.remote)
            checked += 1
    assert checked >= 6


def test_fetch_weights_verifies_every_listed_digest(manifest: dict) -> None:
    """Every file models.json gives a sha256 for is one fetch_weights.py verifies, with that digest (MANO is never
    fetched, so it is the one exception)."""
    fw = _load("fetch_weights_for_digests", REPO / "scripts" / "fetch_weights.py")
    theirs = {m.key: {Path(f.dest or f.remote).name: f.sha256 for f in m.files} for m in fw.MODELS}
    for m in manifest["models"]:
        if m["key"] == "mano":
            continue
        assert m["key"] in theirs, f"{m['key']} is not in fetch_weights.py"
        for f in m["files"]:
            assert theirs[m["key"]].get(Path(f["file"]).name) == f["sha256"], (m["key"], f["file"])


def test_pins_agree_with_the_manifest(manifest: dict) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from fastdo import model_pins

    text = json.dumps(manifest)
    for repo, rev in model_pins.HF_REVISIONS.items():
        entry = next(m for m in manifest["models"] if m["source_url"].endswith("/" + repo))
        assert rev in entry["version"], repo
    assert model_pins.DINOV2_HUB_COMMIT in text and model_pins.COTRACKER_HUB_COMMIT in text


def test_env_locks_have_no_host_paths_and_pin_their_indexes() -> None:
    env = REL / "env"
    for name in ("sam3d", "hawor"):
        conda = (env / f"{name}.conda.txt").read_text()
        assert "@EXPLICIT" in conda and re.search(r"^https://\S+#[0-9a-f]{32}$", conda, re.MULTILINE)
    for name in ("sam3d", "hawor", "hands23"):
        pip = (env / f"{name}.pip.txt").read_text()
        assert "--extra-index-url https://download.pytorch.org/whl/cu128" in pip
        assert re.search(r"^torch==[\d.]+\+cu128$", pip, re.MULTILINE), name
    for f in env.glob("*.txt"):
        assert not HOST_PATH.search(f.read_text()), f.name


def test_export_rewrite_rules() -> None:
    ex = _load("export_env_locks", REL / "env" / "export_env_locks.py")
    keep, build, dropped, unused = ex.rewrite_pip([
        "numpy==1.26.4",
        "six @ file:///somewhere/build/six_1/work",
        "-e git+https://example.org/x.git@" + "a" * 40 + "#egg=xpkg&subdirectory=../../y",
        "-e file:///opt/src/detectron2",
        "lietorch==0.2",
        "chumpy @ git+https://github.com/mattloper/chumpy@" + "b" * 40,
        "openai==2.15.0",
        "torch==2.8.0+cu128",
        "homebuilt @ file:///somewhere/else/homebuilt",
    ], {"detectron2": "detectron2 @ deadbeef"}, ex.UNUSED["sam3d"],
        ex.conda_package_names(["https://conda.anaconda.org/conda-forge/noarch/six-1.17.0-pyhd8ed1ab_0.conda#abc",
                                "https://repo.example/linux-64/libstdcxx-ng-13.2.0-h7e041cc_5.tar.bz2"]))
    assert keep == ["numpy==1.26.4", "torch==2.8.0+cu128"] and dropped == 1     # six: a conda package of the env
    local = [b for b in build if b.startswith("# homebuilt:")]                  # not in the conda inventory
    assert len(local) == 1 and "does not record" in local[0] and "file://" not in local[0]
    venv = ex.rewrite_pip(["six @ file:///somewhere/build/six_1/work"])         # a venv has no conda inventory
    assert venv[2] == 0 and any(b.startswith("# six:") and "not a conda" in b for b in venv[1])
    assert ex.conda_package_names(["https://x/y/python-dateutil-2.9.0-pyhd8ed1ab_0.conda"]) == {"python-dateutil"}
    assert any("editable xpkg" in b and "a" * 40 in b for b in build)
    assert any("detectron2 @ deadbeef" in b for b in build)
    assert any(b.startswith("# lietorch==0.2") for b in build)
    assert any(b.startswith("# chumpy @") and "--no-build-isolation" in b for b in build)   # R4
    assert unused == ["# openai==2.15.0    <- no importer"]
    assert not any(HOST_PATH.search(b) for b in build)
    keep2, _, _, unused2 = ex.rewrite_pip(["openai==2.15.0"])          # the drop list is per env
    assert keep2 == ["openai==2.15.0"] and unused2 == []


def test_host_path_regex_covers_common_roots_and_spares_urls() -> None:
    ex = _load("export_env_locks_rx", REL / "env" / "export_env_locks.py")
    assert ex.HOST_PATH_RX.pattern == HOST_PATH.pattern
    for hit in ("/tmp/x", "x /Users/a/b", "/mnt/d", "/scratch/u", "/data/v", "/opt/conda", "/usr/local/cuda",
                "file:///x", "/" + "home/user"):     # joined so the release identity scan stays clean
        assert HOST_PATH.search(hit), hit
    for miss in ("https://example.org/data/x", "$FASTDO_HAWOR_ROOT/_DATA/data/mano_mean_params.npz",
                 "https://download.pytorch.org/whl/cu128", "~/opt/x"):
        assert not HOST_PATH.search(miss), miss


def test_locks_keep_the_source_builds_and_drop_the_unused() -> None:
    env = REL / "env"
    hawor = (env / "hawor.pip.txt").read_text()
    assert not re.search(r"^chumpy", hawor, re.MULTILINE)
    assert re.search(r"^# chumpy @ git\+\S+@[0-9a-f]{40} .*--no-build-isolation", hawor, re.MULTILINE)
    sam3d = (env / "sam3d.pip.txt").read_text()
    for name in ("openai", "google-genai", "google-auth", "google_genai", "google_auth"):
        assert not re.search(rf"^{name}==", sam3d, re.MULTILINE), name
    assert re.search(r"^gradio==", sam3d, re.MULTILINE)         # moge 1.0.0 requires it: kept pinned


def test_the_exporters_write_both_lock_kinds_from_fake_envs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """export_conda and export_venv end to end on canned tool output (no env is read)."""
    ex = _load("export_env_locks_run", REL / "env" / "export_env_locks.py")

    def fake_run(cmd: list[str]) -> str:
        if "platform.python_version()" in cmd[-1]:
            return "3.11.0\n"
        if cmd[1:3] == ["list", "--explicit"]:
            return "# header\n@EXPLICIT\nhttps://conda.anaconda.org/conda-forge/linux-64/x-1.0-0.conda#" + "a" * 32 + "\n"
        return "numpy==1.26.4\nopenai==2.15.0\nchumpy @ git+https://github.com/mattloper/chumpy@" + "c" * 40 + "\n"
    monkeypatch.setattr(ex, "_run", fake_run)
    ex.export_conda(Path("conda"), tmp_path / "prefix", "sam3d", tmp_path, {})
    ex.export_venv("uv", tmp_path / "venv", "hands23", tmp_path, {})
    sam3d = (tmp_path / "sam3d.pip.txt").read_text()
    assert "\nnumpy==1.26.4\n" in sam3d and "\n# openai==2.15.0    <- no importer" in sam3d
    assert re.search(r"^# chumpy @ .*--no-build-isolation", sam3d, re.MULTILINE)
    hands23 = (tmp_path / "hands23.pip.txt").read_text()
    assert "\nopenai==2.15.0\n" in hands23                      # the drop list is sam3d's only
    assert "@EXPLICIT" in (tmp_path / "sam3d.conda.txt").read_text()

