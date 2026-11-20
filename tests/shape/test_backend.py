"""Shape backend registry, config, and the TRELLIS / SAM 3D backends with injected fakes (D4; no weights, no GPU)."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import torch
import trimesh
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

from fastdo.shape import SHAPE_BACKENDS, ShapeConfig, ShapeResult, make_shape_backend
from fastdo.shape.backend import GEOMETRY_PHASES, masked_crop_rgba
from fastdo.shape.sam3d_backend import (
    SAM3D_GATED_MESSAGE,
    Sam3dBackend,
    resolve_sam3d_pipeline_config,
    start_sam3d_worker,
)
from fastdo.shape.sam3d_worker import GEOMETRY_POSTPROCESS
from fastdo.shape.trellis_backend import (
    TrellisBackend,
    load_trellis_pipeline,
    resolve_trellis_weights,
)


def _frame_and_mask() -> tuple[np.ndarray, np.ndarray]:
    image = np.random.default_rng(0).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:14, 10:20] = True
    return image, mask


def _extents(vertices: np.ndarray) -> tuple[float, ...]:
    return tuple(float(x) for x in np.round(vertices.max(0) - vertices.min(0), 6))


# --- registry and config ------------------------------------------------------------------------------


def test_shape_backend_registry_selects_trellis_by_default() -> None:
    cfg = ShapeConfig()
    assert cfg.backend == "trellis"
    assert SHAPE_BACKENDS == {"trellis": TrellisBackend, "sam3d": Sam3dBackend}
    backend = make_shape_backend(cfg)
    assert isinstance(backend, TrellisBackend)
    assert not backend.loaded                                     # constructing a backend loads nothing


def test_make_shape_backend_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown shape backend"):
        make_shape_backend(ShapeConfig(backend="nerf"))


def test_shape_config_default_is_the_25_step_euler_geometry_profile() -> None:
    cfg = ShapeConfig()
    assert (cfg.sparse_structure_steps, cfg.structured_latent_steps, cfg.solver) == (25, 25, "euler")
    assert cfg.mesh_postprocess
    assert cfg.seed == 42                                         # the baseline generate_mesh_sam3d.py --seed default
    invalid: list[dict[str, Any]] = [
        {"structured_latent_steps": 0}, {"solver": "heun"}, {"sparse_structure_steps": 12.5},
        {"structured_latent_steps": True}, {"sparse_structure_steps": "25"}, {"seed": 1.5}, {"seed": True}]
    for fields in invalid:
        with pytest.raises(ValueError):
            ShapeConfig(**fields)


def test_masked_crop_rgba_squares_and_pads_around_the_mask() -> None:
    image, mask = _frame_and_mask()                               # mask rows 6..13, cols 10..19
    rgba = masked_crop_rgba(image, mask)
    assert rgba.shape == (12, 12, 4)                              # ceil(1.2 * 10)
    np.testing.assert_array_equal(rgba[..., :3], image[4:16, 9:21])
    np.testing.assert_array_equal(rgba[..., 3] == 255, mask[4:16, 9:21])

    corner = np.zeros_like(mask)
    corner[0:4, 0:4] = True
    crop = masked_crop_rgba(image, corner)                        # the square starts one pixel off the frame
    assert crop.shape == (5, 5, 4)
    assert not crop[0].any() and not crop[:, 0].any()             # outside the frame: transparent black
    np.testing.assert_array_equal(crop[1:, 1:, :3], image[0:4, 0:4])
    with pytest.raises(ValueError, match="empty"):
        masked_crop_rgba(image, np.zeros_like(mask))


# --- TRELLIS ------------------------------------------------------------------------------------------


class _FakeTrellis:
    """`TrellisImageTo3DPipeline`-shaped; `run` (which decodes Gaussians and radiance fields too) must not be used.

    Every model call records `torch.is_grad_enabled()`: upstream `run()` is `@torch.no_grad()`, and its decoders are not
    no_grad on their own."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.grad_enabled: list[bool] = []

    def _record(self, name: str, payload: Any) -> None:
        self.calls.append((name, payload))
        self.grad_enabled.append(torch.is_grad_enabled())

    def preprocess_image(self, image: Any) -> str:
        self._record("preprocess_image", np.asarray(image))
        return "prepared"

    def get_cond(self, images: list[Any]) -> dict[str, Any]:
        self._record("get_cond", list(images))
        return {"cond": "c"}

    def sample_sparse_structure(self, cond: dict[str, Any], num_samples: int = 1,
                                sampler_params: dict[str, Any] | None = None) -> str:
        self._record("sample_sparse_structure", (num_samples, dict(sampler_params or {}), float(torch.rand(()))))
        return "coords"

    def sample_slat(self, cond: dict[str, Any], coords: Any, sampler_params: dict[str, Any] | None = None) -> str:
        self._record("sample_slat", dict(sampler_params or {}))
        return "slat"

    def decode_slat(self, slat: Any, formats: list[str]) -> dict[str, Any]:
        self._record("decode_slat", list(formats))
        box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        return {"mesh": [SimpleNamespace(vertices=torch.as_tensor(box.vertices, dtype=torch.float32),
                                         faces=torch.as_tensor(box.faces, dtype=torch.int64))]}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TRELLIS run() decodes every format; the geometry-only profile decodes the mesh only")

    def of(self, name: str) -> list[Any]:
        return [payload for call, payload in self.calls if call == name]


def test_trellis_backend_geometry_only_profile_loads_once() -> None:
    image, mask = _frame_and_mask()
    fake, loads, post_calls, post_grad = _FakeTrellis(), [], [], []

    def _load() -> _FakeTrellis:
        loads.append("load")
        return fake

    def _post(vertices: np.ndarray, faces: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        post_calls.append(kwargs)
        post_grad.append(torch.is_grad_enabled())
        return vertices, faces

    backend = TrellisBackend(ShapeConfig(seed=5), loader=_load, postprocess_fn=_post)
    assert loads == [] and not backend.loaded
    first = backend.reconstruct(image, mask)
    second = backend.reconstruct(image, mask)

    assert loads == ["load"] and backend.loaded                    # one load, reused for the next clip
    assert isinstance(first, ShapeResult) and first.texture_path is None
    assert set(GEOMETRY_PHASES) <= set(first.phase_seconds)
    np.testing.assert_array_equal(fake.of("preprocess_image")[0], masked_crop_rgba(image, mask))
    ss = fake.of("sample_sparse_structure")
    assert [(n, params) for n, params, _ in ss] == [(1, {"steps": 25})] * 2
    assert ss[0][2] == ss[1][2]                                   # the flow noise is seeded per object
    assert fake.of("sample_slat") == [{"steps": 25}] * 2
    assert fake.of("decode_slat") == [["mesh"]] * 2
    assert post_calls == [GEOMETRY_POSTPROCESS] * 2
    assert _extents(first.vertices) == (1.0, 3.0, 2.0)            # z-up -> y-up, as both pipelines export
    np.testing.assert_array_equal(first.vertices, second.vertices)

    assert len(fake.grad_enabled) == 10 and not any(fake.grad_enabled)   # every model call under no_grad
    assert post_grad == [False, False]
    assert torch.is_grad_enabled()                                # grad mode is restored after reconstruct
    fake.decode_slat("slat", ["mesh"])
    assert fake.grad_enabled[-1]                                  # control: the fake does see grad mode outside


def test_trellis_backend_passes_configured_steps_and_can_skip_the_mesh_postprocess() -> None:
    image, mask = _frame_and_mask()
    fake = _FakeTrellis()
    cfg = ShapeConfig(sparse_structure_steps=12, structured_latent_steps=8, mesh_postprocess=False)
    backend = TrellisBackend(cfg, pipeline=fake, postprocess_fn=lambda *a, **k: pytest.fail("postprocess ran"))
    backend.reconstruct(image, mask)
    assert fake.of("sample_sparse_structure")[0][1] == {"steps": 12}
    assert fake.of("sample_slat") == [{"steps": 8}]


def test_trellis_backend_rejects_the_midpoint_solver() -> None:
    with pytest.raises(ValueError, match="euler"):
        TrellisBackend(ShapeConfig(solver="midpoint"))


def test_trellis_backend_without_the_package_raises_instead_of_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "trellis", None)             # `import trellis...` now raises ImportError
    image, mask = _frame_and_mask()
    with pytest.raises(RuntimeError, match="TRELLIS"):
        TrellisBackend(ShapeConfig(device="cpu")).reconstruct(image, mask)


def test_load_trellis_pipeline_reads_the_weights_from_the_local_cache_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[tuple[str, Any]] = []

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, path: str) -> _Pipeline:
            loaded.append(("from_pretrained", path))
            return cls()

        def to(self, device: torch.device) -> None:
            loaded.append(("to", device))

    pipelines = ModuleType("trellis.pipelines")
    pipelines.__dict__["TrellisImageTo3DPipeline"] = _Pipeline
    monkeypatch.setitem(sys.modules, "trellis", ModuleType("trellis"))
    monkeypatch.setitem(sys.modules, "trellis.pipelines", pipelines)
    seen: list[dict[str, Any]] = []

    def _cached(**kwargs: Any) -> str:
        seen.append(kwargs)
        return str(tmp_path)

    load_trellis_pipeline("microsoft/TRELLIS-image-large", "cpu", snapshot_fn=_cached)
    assert seen == [{"repo_id": "microsoft/TRELLIS-image-large", "local_files_only": True}]
    assert loaded == [("from_pretrained", str(tmp_path)), ("to", torch.device("cpu"))]
    local = resolve_trellis_weights(str(tmp_path), snapshot_fn=lambda **k: pytest.fail("a directory needs no lookup"))
    assert local == str(tmp_path)

    empty_cache = functools.partial(snapshot_download, cache_dir=str(tmp_path / "hf_cache"))   # real resolver, offline
    with pytest.raises(RuntimeError, match="not in the local Hugging Face cache"):
        load_trellis_pipeline("microsoft/TRELLIS-image-large", "cpu", snapshot_fn=empty_cache)
    assert len(loaded) == 2                                       # nothing loaded, nothing downloaded


# --- SAM 3D -------------------------------------------------------------------------------------------


def _gated_error() -> GatedRepoError:
    url = "https://huggingface.co/facebook/sam-3d-objects/resolve/main/checkpoints/pipeline.yaml"
    response = httpx.Response(403, request=httpx.Request("GET", url))
    return GatedRepoError("403 Client Error: access to model facebook/sam-3d-objects is restricted",
                          response=response)


@pytest.mark.parametrize("failure", ["gated_repo", "not_in_local_cache"])
def test_sam3d_backend_raises_when_gated(failure: str) -> None:
    error = _gated_error() if failure == "gated_repo" else LocalEntryNotFoundError("no local snapshot")
    seen: list[dict[str, Any]] = []

    def _unauthorized(*args: Any, **kwargs: Any) -> str:
        seen.append(kwargs)
        raise error

    cfg = ShapeConfig(backend="sam3d", sam3d_python=sys.executable)
    backend = Sam3dBackend(cfg, loader=lambda: start_sam3d_worker(cfg, snapshot_fn=_unauthorized))
    image, mask = _frame_and_mask()
    with pytest.raises(RuntimeError) as info:
        backend.reconstruct(image, mask)

    message = str(info.value)
    assert SAM3D_GATED_MESSAGE in message and "authorized" in message and "gated" in message
    assert seen and all(kwargs.get("local_files_only") is True for kwargs in seen)   # never downloads


def test_resolve_sam3d_pipeline_config_needs_the_checkpoint_yaml(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not authorized"):
        resolve_sam3d_pipeline_config("facebook/sam-3d-objects", snapshot_fn=lambda *a, **k: str(tmp_path))
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "pipeline.yaml").write_text("_target_: x\n")
    resolved = resolve_sam3d_pipeline_config("facebook/sam-3d-objects", snapshot_fn=lambda *a, **k: str(tmp_path))
    assert resolved == tmp_path / "checkpoints" / "pipeline.yaml"


def test_start_sam3d_worker_needs_an_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "pipeline.yaml").write_text("_target_: x\n")
    monkeypatch.delenv("FASTDO_SAM3D_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="FASTDO_SAM3D_PYTHON"):
        start_sam3d_worker(ShapeConfig(backend="sam3d"), snapshot_fn=lambda *a, **k: str(tmp_path))


class _FakeClient:
    def __init__(self) -> None:
        self.requests: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
        self.closed = False

    def reconstruct(self, image: np.ndarray, mask: np.ndarray, params: dict[str, Any]) -> ShapeResult:
        self.requests.append((image.copy(), mask.copy(), dict(params)))
        return ShapeResult(np.zeros((4, 3)), np.array([[0, 1, 2]]), None, {"decode": 0.5})

    def close(self) -> int:
        self.closed = True
        return 0


def test_sam3d_backend_starts_one_worker_across_clips_and_forwards_the_profile() -> None:
    image, mask = _frame_and_mask()
    client, starts = _FakeClient(), []

    def _start() -> _FakeClient:
        starts.append("start")
        return client

    backend = Sam3dBackend(ShapeConfig(backend="sam3d", structured_latent_steps=12, solver="midpoint"), loader=_start)
    assert starts == []
    backend.reconstruct(image, mask)
    result = backend.reconstruct(image, ~mask)

    assert starts == ["start"]                                    # one persistent worker for every clip
    assert result.texture_path is None and result.phase_seconds == {"decode": 0.5}
    assert [params for _, _, params in client.requests] == [
        {"seed": 42, "sparse_structure_steps": 25, "structured_latent_steps": 12, "solver": "midpoint",
         "mesh_postprocess": True, "deterministic": True,
         # the fast-path options ride along per request, each at the value that is today's path, so
         # one worker process can serve every evaluation arm without restarting.
         "mesh_postprocess_impl": "official", "fast_decimator_min_raw_faces": 250000,
         "min_component_area_frac": 0.0, "wall_qc": "off",
         "wall_qc_max_removed": 0.25, "wall_qc_min_area_ratio": 1.40}] * 2
    np.testing.assert_array_equal(client.requests[1][1], ~mask)
    assert backend.close() == 0 and client.closed
    with pytest.raises(ValueError, match="empty"):
        backend.reconstruct(image, np.zeros_like(mask))
