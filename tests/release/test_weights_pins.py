"""Load-time pinning (R1, R2): the loaders read the revisions `scripts/fetch_weights.py` downloads at, from one table
(`fastdo.model_pins`). CPU only; a fake Hugging Face cache, no network."""
from __future__ import annotations

import functools
import json
import stat
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fastdo import model_pins
from fastdo.shape import sam3d_backend
from fastdo.shape.backend import ShapeConfig

REPO = Path(__file__).resolve().parents[2]
SAM3D = model_pins.SAM3D_REPO
SAM3D_REV = model_pins.HF_REVISIONS[SAM3D]


def _fetched_layout(cache: Path, repo: str, rev: str, files: dict[str, bytes], unfetched: tuple[str, ...]) -> Path:
    """The cache `snapshot_download(repo, revision=<commit>, allow_patterns=["checkpoints/*"])` leaves: blobs,
    `snapshots/<commit>/...` symlinks, and the tree listing of the WHOLE repository (`trees/<commit>.json`), but no
    `refs/main` (a commit-hash download writes no ref)."""
    storage = cache / ("models--" + repo.replace("/", "--"))
    tree: dict[str, dict[str, Any]] = {}
    for i, (name, payload) in enumerate(files.items()):
        blob = storage / "blobs" / f"{i:040x}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(payload)
        link = storage / "snapshots" / rev / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("../" * (len(Path(name).parts) + 1)) / "blobs" / blob.name)
        tree[name] = {"size": len(payload), "blob_id": f"{i:040x}"}
    for j, name in enumerate(unfetched):                   # README, doc images: listed upstream, not fetched
        tree[name] = {"size": 1, "blob_id": f"{1000 + j:040x}"}
    (storage / "trees").mkdir(parents=True)
    (storage / "trees" / f"{rev}.json").write_text(json.dumps({"format_version": 1, "files": tree}))
    return storage


@pytest.fixture
def fetched_cache(tmp_path: Path) -> Path:
    pytest.importorskip("huggingface_hub")
    cache = tmp_path / "hub"
    _fetched_layout(cache, SAM3D, SAM3D_REV,
                    {"checkpoints/pipeline.yaml": b"depth_model: {}\n", "checkpoints/ss_generator.ckpt": b"w",
                     "checkpoints/ss_encoder.safetensors": b""},
                    ("README.md", "LICENSE", "doc/arch.png"))
    return cache


def test_the_fetch_layout_resolves_at_the_pinned_revision(fetched_cache: Path) -> None:
    from huggingface_hub import snapshot_download

    snap = functools.partial(snapshot_download, cache_dir=str(fetched_cache))
    config = sam3d_backend.resolve_sam3d_pipeline_config(SAM3D, snapshot_fn=snap)
    assert config.is_file() and config.parent.parent.name == SAM3D_REV
    ident = sam3d_backend.sam3d_checkpoint_identity(SAM3D, snapshot_fn=snap)
    assert ident.startswith(f"sam3d:{SAM3D_REV}:") and "pipeline.yaml=" in ident


def test_the_two_lookups_that_failed_before_the_pin(fetched_cache: Path) -> None:
    """The defect, kept visible: no revision reads refs/main, which the fetch never wrote; and the revision without
    the fetch's pattern is an incomplete snapshot (README and doc images are listed but were never fetched)."""
    from huggingface_hub import snapshot_download

    with pytest.raises(OSError):
        snapshot_download(SAM3D, local_files_only=True, cache_dir=str(fetched_cache))
    with pytest.raises(OSError):
        snapshot_download(SAM3D, revision=SAM3D_REV, local_files_only=True, cache_dir=str(fetched_cache))
    with pytest.raises(RuntimeError, match=sam3d_backend.SAM3D_GATED_MESSAGE):     # reported, never a crash
        sam3d_backend.resolve_sam3d_pipeline_config(
            SAM3D, snapshot_fn=functools.partial(snapshot_download, cache_dir=str(fetched_cache)), revision=None)


def test_the_resolver_passes_the_pin_and_the_pattern(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "pipeline.yaml").write_text("x: 1\n")

    def snap(**kwargs: Any) -> str:
        seen.append(kwargs)
        return str(tmp_path)
    sam3d_backend.resolve_sam3d_pipeline_config(SAM3D, snapshot_fn=snap)
    sam3d_backend.resolve_sam3d_pipeline_config("x/unpinned", snapshot_fn=snap)
    assert seen[0] == {"repo_id": SAM3D, "revision": SAM3D_REV, "allow_patterns": ["checkpoints/*"],
                       "local_files_only": True}
    assert seen[1]["revision"] is None                                 # an unpinned repository reads main


def test_the_worker_command_carries_the_moge_v1_pin(tmp_path: Path) -> None:
    record = tmp_path / "argv.txt"
    script = tmp_path / "fake_python.sh"
    script.write_text("#!/bin/sh\n"
                      f"printf '%s\\n' \"$@\" > '{record}'\n"
                      "echo '{\"ready\": true, \"load_seconds\": 0.0, \"spconv_kernels\": \"pinned\"}'\n"
                      "read line\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "snap" / "checkpoints").mkdir(parents=True)
    (tmp_path / "snap" / "checkpoints" / "pipeline.yaml").write_text("x: 1\n")
    cfg = ShapeConfig(backend="sam3d", sam3d_python=str(script), device="cpu")
    client = sam3d_backend.start_sam3d_worker(cfg, snapshot_fn=lambda **_k: str(tmp_path / "snap"),
                                              startup_timeout_s=60.0)
    assert client.close(timeout_s=30.0) == 0
    argv = record.read_text().split()
    i = argv.index("--hf-revision")
    assert argv[i + 1] == f"{model_pins.MOGE1_REPO}={model_pins.HF_REVISIONS[model_pins.MOGE1_REPO]}"


def test_the_worker_pins_the_pipeline_yamls_from_pretrained_loaders() -> None:
    from omegaconf import OmegaConf

    from fastdo.shape import sam3d_worker as W

    rev = model_pins.HF_REVISIONS[model_pins.MOGE1_REPO]
    # SAM 3D's checkpoints/pipeline.yaml at the pinned revision, its depth_model block verbatim
    config = OmegaConf.create({"depth_model": {
        "_target_": "sam3d_objects.pipeline.depth_models.moge.MoGe",
        "model": {"_target_": "moge.model.v1.MoGeModel.from_pretrained",
                  "pretrained_model_name_or_path": "Ruicheng/moge-vitl"}},
        "other": [{"_target_": "torch.hub.from_pretrained", "pretrained_model_name_or_path": "someone/else"}]})
    assert W.pin_hf_revisions(config, W.parse_hf_revisions([f"Ruicheng/moge-vitl={rev}"])) == ["Ruicheng/moge-vitl"]
    assert config.depth_model.model.revision == rev and "revision" not in config.other[0]
    W.check_hydra_targets(config, resolve=lambda _target: None)         # the pin adds no target
    config.depth_model.model.revision = "0" * 40
    with pytest.raises(RuntimeError, match="pins Ruicheng/moge-vitl"):
        W.pin_hf_revisions(config, {"Ruicheng/moge-vitl": rev})
    for bad in (["no-equals"], ["=x"], ["a=1", "a=2"]):
        with pytest.raises(ValueError):
            W.parse_hf_revisions(bad)


def test_sam2_loaders_read_the_pinned_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sam2")
    import huggingface_hub
    from sam2 import build_sam, sam2_image_predictor

    from fastdo.shape import masks_sam2

    downloads: list[dict[str, Any]] = []
    builds: list[tuple[str, dict[str, Any]]] = []

    def fake_download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        return "/ckpt/" + kwargs["filename"]

    def fake_builder(name: str):
        def build(config_file: str, ckpt_path: str, **kwargs: Any) -> str:
            builds.append((name, {"config_file": config_file, "ckpt_path": ckpt_path, **kwargs}))
            return f"{name}-model"
        return build

    class FakeImagePredictor:
        def __init__(self, model: Any, **kwargs: Any) -> None:
            self.model, self.kwargs = model, kwargs

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(build_sam, "build_sam2_video_predictor", fake_builder("video"))
    monkeypatch.setattr(build_sam, "build_sam2", fake_builder("image"))
    monkeypatch.setattr(sam2_image_predictor, "SAM2ImagePredictor", FakeImagePredictor)
    monkeypatch.setattr(masks_sam2, "_under_sam2_hydra", lambda fn: fn())      # the guard: tested in tests/shape
    rev = model_pins.HF_REVISIONS[masks_sam2.MODEL_ID]
    assert masks_sam2.load_sam2_predictor(device="cpu") == "video-model"
    image = masks_sam2.load_sam2_image_predictor(device="cpu")
    assert image.model == "image-model" and image.kwargs == {"device": "cpu"}
    config, ckpt = build_sam.HF_MODEL_ID_TO_FILENAMES[masks_sam2.MODEL_ID]
    assert downloads == [{"repo_id": masks_sam2.MODEL_ID, "filename": ckpt, "revision": rev}] * 2
    assert builds == [("video", {"config_file": config, "ckpt_path": "/ckpt/" + ckpt, "device": "cpu"}),
                      ("image", {"config_file": config, "ckpt_path": "/ckpt/" + ckpt, "device": "cpu"})]


def test_moge2_loads_at_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastdo.geometry import moge_model

    seen: dict[str, Any] = {}

    class FakeMoGe:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> FakeMoGe:
            seen["name"], seen["kwargs"] = name, kwargs
            return cls()

        def to(self, _device: str) -> FakeMoGe:
            return self

        def eval(self) -> FakeMoGe:
            return self

    v2 = types.ModuleType("moge.model.v2")
    v2.MoGeModel = FakeMoGe
    monkeypatch.setitem(sys.modules, "moge", types.ModuleType("moge"))
    monkeypatch.setitem(sys.modules, "moge.model", types.ModuleType("moge.model"))
    monkeypatch.setitem(sys.modules, "moge.model.v2", v2)
    assert isinstance(moge_model._load_model("cpu"), FakeMoGe)
    assert seen == {"name": moge_model.MODEL_ID,
                    "kwargs": {"revision": model_pins.HF_REVISIONS[moge_model.MODEL_ID]}}


def test_every_pinned_model_has_a_loader_that_reads_its_pin() -> None:
    """The table covers exactly the HF models the pipeline loads by id, and their loaders name the table."""
    from fastdo.geometry import moge_model
    from fastdo.shape import masks_sam2

    assert set(model_pins.HF_REVISIONS) == {masks_sam2.MODEL_ID, moge_model.MODEL_ID, SAM3D, model_pins.MOGE1_REPO}
    for rev in model_pins.HF_REVISIONS.values():
        assert len(rev) == 40 and int(rev, 16) >= 0
    for path in ("src/fastdo/shape/masks_sam2.py", "src/fastdo/geometry/moge_model.py",
                 "src/fastdo/shape/sam3d_backend.py"):
        assert "hf_revision" in (REPO / path).read_text(), path
