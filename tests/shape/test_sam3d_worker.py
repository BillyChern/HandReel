"""The SAM 3D geometry-only driver and its persistent worker process, with a fake pipeline (D4).

Only `test_real_sam3d_geometry_profile` touches real weights; it skips without CUDA, and the GPU run uses a
GPU budget above SAM 3D's measured 18.67 GiB peak.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra.utils
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastdo.shape import ShapeConfig
from fastdo.shape.backend import GEOMETRY_PHASES
from fastdo.shape.sam3d_backend import WORKER_SCRIPT, Sam3dBackend, Sam3dWorkerClient
from fastdo.shape.sam3d_worker import (
    GEOMETRY_POSTPROCESS,
    SAM3D_MIN_GPU_GB,
    check_hydra_targets,
    load_sam3d_pipeline,
    reconstruct_geometry,
    require_gpu_cap,
)
from tests.shape.fake_sam3d import FakeSam3dPipeline

_REPO = Path(__file__).resolve().parents[2]
_SAM3D_PHASES = {"pointmap", *GEOMETRY_PHASES}
_FAKE_FACTORY = ["--factory", "tests.shape.fake_sam3d:make_fake_pipeline"]


def _frame_and_mask() -> tuple[np.ndarray, np.ndarray]:
    image = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:15, 8:22] = True
    return image, mask


def _extents(vertices: np.ndarray) -> tuple[float, ...]:
    return tuple(float(x) for x in np.round(vertices.max(0) - vertices.min(0), 6))


def _reconstruct(pipe: FakeSam3dPipeline, **overrides: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    params: dict[str, Any] = {"seed": 0, "sparse_structure_steps": 25, "structured_latent_steps": 25,
                              "solver": "euler", "mesh_postprocess": False} | overrides
    image, mask = _frame_and_mask()
    return reconstruct_geometry(pipe, image, mask, **params)


def _events(pipe: FakeSam3dPipeline, name: str) -> list[dict[str, Any]]:
    return [event for event in pipe.events if event["event"] == name]


# --- the in-process driver ----------------------------------------------------------------------------


def test_reconstruct_geometry_runs_only_the_geometry_phases() -> None:
    pipe, post_calls, post_grad = FakeSam3dPipeline(), [], []

    def _post(vertices: np.ndarray, faces: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        post_calls.append(kwargs)
        post_grad.append(torch.is_grad_enabled())
        return vertices, faces

    image, mask = _frame_and_mask()
    vertices, faces, phases = reconstruct_geometry(pipe, image, mask, seed=0, sparse_structure_steps=25,
                                                   structured_latent_steps=25, solver="euler",
                                                   mesh_postprocess=True, postprocess_fn=_post)

    assert [e["event"] for e in pipe.events] == ["pointmap", "preprocess", "preprocess", "sparse_structure",
                                                 "structured_latent", "decode"]
    (pointmap,) = _events(pipe, "pointmap")                       # the full frame, alpha = mask, as the baseline
    assert pointmap["shape"] == [24, 32, 4] and not pointmap["pointmap_given"]
    assert pointmap["alpha_sum"] == int(mask.sum()) * 255
    assert pointmap["rgb_sum"] == int(image.astype(np.int64).sum())
    assert [(e["preprocessor"], e["with_pointmap"]) for e in _events(pipe, "preprocess")] == [
        ("ss_pre", True), ("slat_pre", False)]
    assert [(e["steps"], e["use_distillation"]) for e in _events(pipe, "sparse_structure")] == [(25, False)]
    assert [(e["steps"], e["use_distillation"]) for e in _events(pipe, "structured_latent")] == [(25, False)]
    assert [e["formats"] for e in _events(pipe, "decode")] == [["mesh"]]
    assert post_calls == [GEOMETRY_POSTPROCESS]
    assert set(phases) == _SAM3D_PHASES
    assert _extents(vertices) == (1.0, 3.0, 2.0)                  # z-up -> y-up, as the baseline's exported mesh
    assert vertices.dtype == np.float64 and faces.dtype == np.int64
    assert [e["grad"] for e in pipe.events] == [False] * 6 and post_grad == [False]   # no autograd graphs
    assert torch.is_grad_enabled()                                # restored afterwards


def test_reconstruct_geometry_seeds_the_flow_noise() -> None:
    pipe = FakeSam3dPipeline()
    _reconstruct(pipe, seed=7)
    _reconstruct(pipe, seed=7)
    _reconstruct(pipe, seed=8)
    noise = [e["noise"] for e in _events(pipe, "sparse_structure")]
    assert noise[0] == noise[1] != noise[2]


def test_reconstruct_geometry_switches_to_midpoint_only_while_sampling() -> None:
    pipe = FakeSam3dPipeline()
    _reconstruct(pipe, solver="midpoint", sparse_structure_steps=8, structured_latent_steps=8)
    assert _events(pipe, "sparse_structure")[0]["solvers"] == ["midpoint", "midpoint"]
    assert _events(pipe, "structured_latent")[0]["solver"] == "midpoint"
    assert [e["steps"] for e in pipe.events if "steps" in e] == [8, 8]
    assert all(g._solver_method == "euler" for g in pipe.models.values())    # restored afterwards

    pipe.fail_in = "structured_latent"
    with pytest.raises(RuntimeError, match="injected"):
        _reconstruct(pipe, solver="midpoint")
    assert all(g._solver_method == "euler" for g in pipe.models.values())    # restored on failure too


# --- loading safeguards -------------------------------------------------------------------------------


def test_check_hydra_targets_mirrors_the_notebook_safety_filter() -> None:
    allowed = OmegaConf.create({
        "_target_": "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap",
        "depth_model": {"_target_": "moge.model.v1.MoGeModel.from_pretrained"},
        "transforms": [{"_target_": "torchvision.transforms.Resize", "size": 518},
                       {"nested": [{"_target_": "torch.nn.Identity"}]}]})
    resolved: list[str] = []
    check_hydra_targets(allowed, resolve=resolved.append)
    assert len(resolved) == 4                                     # every target, at any depth

    for target in ("os.system", "builtins.exec", "subprocess.Popen", "fastdo.engine.workers.shape_worker"):
        config = OmegaConf.create({"_target_": "torch.nn.Identity", "modules": [{"inner": {"_target_": target}}]})
        with pytest.raises(RuntimeError, match="not allowed"):
            check_hydra_targets(config, resolve=lambda _: object())
    aliased = OmegaConf.create({"_target_": "torch.nn.Identity", "hooks": [{"_target_": "torch.os.system"}]})
    with pytest.raises(RuntimeError, match="torch.os.system"):
        check_hydra_targets(aliased)                              # hydra resolves it to os.system: blacklisted
    check_hydra_targets(OmegaConf.create({"_target_": "torch.nn.Identity"}))


def test_load_sam3d_pipeline_checks_every_target_before_instantiating(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instantiated: list[Any] = []

    def _instantiate(config: Any) -> str:
        instantiated.append(config)
        return "pipeline"

    monkeypatch.setitem(sys.modules, "sam3d_objects", ModuleType("sam3d_objects"))
    monkeypatch.setattr(hydra.utils, "instantiate", _instantiate)
    for name in ("LIDRA_SKIP_INIT", "CUDA_HOME"):
        monkeypatch.setenv(name, "restored after the test")      # the loader overwrites both
    (tmp_path / "checkpoints").mkdir()
    bad = tmp_path / "checkpoints" / "bad.yaml"
    bad.write_text("_target_: torch.nn.Identity\nhooks:\n  - _target_: os.system\n    command: echo pwned\n")
    with pytest.raises(RuntimeError, match="os.system"):
        load_sam3d_pipeline(str(bad), "cpu")
    assert instantiated == []

    good = tmp_path / "checkpoints" / "pipeline.yaml"
    good.write_text("_target_: torch.nn.Identity\n")
    assert load_sam3d_pipeline(str(good), "cpu") == "pipeline"
    (config,) = instantiated
    assert (config.rendering_engine, config.compile_model, config.workspace_dir, config.device) == (
        "pytorch3d", False, str(tmp_path / "checkpoints"), "cpu")



def test_load_sam3d_pipeline_refuses_a_requested_pin_that_no_loader_reads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A YAML without the MoGe v1 `depth_model` node pins nothing; the load stops instead of reading `main`."""
    from fastdo import model_pins

    instantiated: list[Any] = []
    monkeypatch.setitem(sys.modules, "sam3d_objects", ModuleType("sam3d_objects"))
    monkeypatch.setattr(hydra.utils, "instantiate", lambda config: instantiated.append(config) or "pipeline")
    for name in ("LIDRA_SKIP_INIT", "CUDA_HOME"):
        monkeypatch.setenv(name, "restored after the test")      # the loader overwrites both
    (tmp_path / "checkpoints").mkdir()
    yaml = tmp_path / "checkpoints" / "pipeline.yaml"
    yaml.write_text("_target_: torch.nn.Identity\n")             # no `*.from_pretrained` node at all
    pins = model_pins.sam3d_hf_revisions()
    with pytest.raises(RuntimeError, match=r"Ruicheng/moge-vitl.*not pinned"):
        load_sam3d_pipeline(str(yaml), "cpu", hf_revisions=pins)
    assert instantiated == []
    from fastdo.shape import sam3d_worker

    checked: list[Any] = []                                            # moge is not importable in the uv env
    monkeypatch.setattr(sam3d_worker, "check_hydra_targets", checked.append)
    yaml.write_text("_target_: torch.nn.Identity\ndepth_model:\n  _target_: torch.nn.Identity\n  model:\n"
                    "    _target_: moge.model.v1.MoGeModel.from_pretrained\n"
                    "    pretrained_model_name_or_path: Ruicheng/moge-vitl\n")
    assert load_sam3d_pipeline(str(yaml), "cpu", hf_revisions=pins) == "pipeline"
    (config,) = instantiated
    assert config.depth_model.model.revision == pins[model_pins.MOGE1_REPO] and checked == [config]
    assert load_sam3d_pipeline(str(yaml), "cpu") == "pipeline"         # no pin requested: nothing to check

def _worker_env(log: Path) -> dict[str, str]:
    return dict(os.environ, FAKE_SAM3D_LOG=str(log), CUDA_VISIBLE_DEVICES="",
                PYTHONPATH=os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")]))


def test_worker_refuses_to_load_below_the_sam3d_memory_floor(tmp_path: Path) -> None:
    require_gpu_cap(None, SAM3D_MIN_GPU_GB)                       # uncapped or not on CUDA: nothing to check
    require_gpu_cap(21.0, SAM3D_MIN_GPU_GB)
    with pytest.raises(RuntimeError, match="FASTDO_GPU_MEM_GB>=19"):
        require_gpu_cap(12.0, SAM3D_MIN_GPU_GB)

    log = tmp_path / "events.jsonl"                               # the worker script with a 12 GiB CUDA cap applied
    patched = ("import sys; sys.path.insert(0, sys.argv.pop(1)); import sam3d_worker as w; "
               "w.apply_gpu_cap = lambda device: 12.0; sys.exit(w.main(sys.argv[1:]))")
    command = [sys.executable, "-c", patched, str(WORKER_SCRIPT.parent), "--config", "fake.yaml", "--device", "cuda",
               *_FAKE_FACTORY]
    with pytest.raises(RuntimeError, match=r"cap is 12\.0 GiB, below the 19 GiB"):
        Sam3dWorkerClient(command, env=_worker_env(log), startup_timeout_s=120.0)
    assert not log.exists()                                       # refused before the pipeline load


# --- the persistent worker process --------------------------------------------------------------------


def _worker(tmp_path: Path, config: str = "fake.yaml") -> tuple[Sam3dWorkerClient, Path]:
    log = tmp_path / "events.jsonl"
    command = [sys.executable, str(WORKER_SCRIPT), "--config", config, "--device", "cpu", *_FAKE_FACTORY]
    return Sam3dWorkerClient(command, env=_worker_env(log), startup_timeout_s=120.0, request_timeout_s=120.0), log


def test_worker_process_loads_once_and_serves_every_clip_geometry_only(tmp_path: Path) -> None:
    client, log = _worker(tmp_path)
    backend = Sam3dBackend(ShapeConfig(backend="sam3d", device="cpu", mesh_postprocess=False, spconv_kernel_pin=False),
                           client=client)                 # the fake worker starts in its default tuner mode
    image, mask = _frame_and_mask()
    try:
        first = backend.reconstruct(image, mask)
        second = backend.reconstruct(image[::-1].copy(), mask[::-1].copy())
    finally:
        returncode = backend.close()

    assert returncode == 0
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["event"] for e in events].count("load") == 1       # the pipeline loads once for both clips
    assert [e["formats"] for e in events if e["event"] == "decode"] == [["mesh"], ["mesh"]]
    assert [e["steps"] for e in events if e["event"] == "sparse_structure"] == [25, 25]
    assert not any(e["grad"] for e in events if e["event"] != "load")
    assert client.load_seconds >= 0.0
    for result in (first, second):
        assert result.texture_path is None and set(result.phase_seconds) == _SAM3D_PHASES
        assert _extents(result.vertices) == (1.0, 3.0, 2.0)


def test_worker_process_reports_a_failed_request_and_keeps_serving(tmp_path: Path) -> None:
    client, _ = _worker(tmp_path)
    image, mask = _frame_and_mask()
    params = {"seed": 0, "sparse_structure_steps": 25, "structured_latent_steps": 25, "solver": "euler",
              "mesh_postprocess": False}
    try:
        with pytest.raises(RuntimeError, match="empty object mask"):
            client.reconstruct(image, np.zeros_like(mask), params)
        assert client.reconstruct(image, mask, params).faces.shape == (12, 3)
    finally:
        assert client.close() == 0


def test_worker_process_startup_failure_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _worker(tmp_path, config="fail")


# --- real weights (GPU) -------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SAM 3D Objects needs CUDA (and > 17 GiB of allocator memory)")
def test_real_sam3d_geometry_profile() -> None:
    if not os.environ.get("FASTDO_SAM3D_PYTHON"):
        pytest.skip("FASTDO_SAM3D_PYTHON (the SAM 3D environment's interpreter) is not set")
    if float(os.environ.get("FASTDO_GPU_MEM_GB", "12")) < SAM3D_MIN_GPU_GB:
        pytest.skip("SAM 3D peaks at 18.67 GiB allocated; run with FASTDO_GPU_MEM_GB >= 19")
    height, width = 480, 640
    yy, xx = np.mgrid[:height, :width]
    mask = (yy - 240) ** 2 + (xx - 320) ** 2 < 90 ** 2
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    image[mask] = (200, 60, 40)
    backend = Sam3dBackend(ShapeConfig(backend="sam3d"))
    try:
        first = backend.reconstruct(image, mask)
        second = backend.reconstruct(image, mask)                 # warm: the per-phase numbers to report
    finally:
        backend.close()
    for result in (first, second):
        assert result.texture_path is None and len(result.faces) > 0
        assert _SAM3D_PHASES <= set(result.phase_seconds)
