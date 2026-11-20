"""A SAM 3D Objects pipeline stand-in for the D4 tests; no weights, no GPU.

`FakeSam3dPipeline` has the `InferencePipelinePointMap` methods the geometry-only driver calls. The methods that
profile must never reach (full `run`, texture/GLB baking, layout pose decoding and post-optimization) fail loudly.
The worker process loads it with `--factory tests.shape.fake_sam3d:make_fake_pipeline`; `FAKE_SAM3D_LOG` names a
JSON-lines event log the test reads back. It prints to stdout on purpose: the worker must keep stdout for its protocol.

`make_fake_sam3d_modules` builds stand-ins for the three SAM 3D modules `deterministic_sam3d` patches (F2). Every
pipeline event records `det`, whether that patch is installed at the time of the call.
"""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
import trimesh

from fastdo.shape import sam3d_determinism as det

BOX_EXTENTS = (1.0, 2.0, 3.0)   # z-up box; the driver rotates it to y-up, extents (1, 3, 2)


def box_mesh() -> tuple[np.ndarray, np.ndarray]:
    box = trimesh.creation.box(extents=BOX_EXTENTS)
    return np.asarray(box.vertices, dtype=np.float32), np.asarray(box.faces, dtype=np.int32)


def make_fake_sam3d_modules() -> dict[str, ModuleType]:
    """Fake `spatial`, `utils_cube` and `flexicubes` modules; their `SEEN` lists record what each call saw."""
    spatial, utils_cube, flexicubes = (ModuleType(name) for name in (det.SPATIAL_MODULE, det.UTILS_CUBE_MODULE,
                                                                     det.FLEXICUBES_MODULE))

    class SparseDownsample:
        def forward(self, x: Any) -> Any:
            spatial.SEEN.append(torch.scatter_reduce)       # looked up at call time, as the real forward does
            return x

    class FlexiCubes:
        def __call__(self, voxelgrid_vertices: Any, scalar_field: Any, *args: Any, **kwargs: Any) -> str:
            flexicubes.SEEN.append(torch.are_deterministic_algorithms_enabled())
            return "flexicubes mesh"

    def cubes_to_verts(num_verts: int, cubes: Any, value: Any, reduce: str = "mean") -> str:
        return "reference cubes_to_verts"

    spatial.SEEN, spatial.SparseDownsample = [], SparseDownsample
    utils_cube.cubes_to_verts = cubes_to_verts
    flexicubes.SEEN, flexicubes.FlexiCubes = [], FlexiCubes
    return {det.SPATIAL_MODULE: spatial, det.UTILS_CUBE_MODULE: utils_cube, det.FLEXICUBES_MODULE: flexicubes}


def det_patch_installed() -> bool:
    """Whether `deterministic_sam3d`'s patch is installed on the (fake) `utils_cube` module right now."""
    module = sys.modules.get(det.UTILS_CUBE_MODULE)
    return module is not None and bool(getattr(module.cubes_to_verts, det.PATCH_SENTINEL, False))


class FakeGenerator:
    """A flow generator's solver attributes, as `FlowMatching` keeps them."""

    def __init__(self) -> None:
        self._solver_method, self._solver = "euler", "solver:euler"

    def _get_solver(self, method: str, kwargs: dict[str, Any]) -> tuple[str, str]:
        return method, f"solver:{method}"


class FakeSam3dPipeline:
    def __init__(self, log_path: str | None = None) -> None:
        self.device = torch.device("cpu")
        self.ss_preprocessor, self.slat_preprocessor = "ss_pre", "slat_pre"
        self.models = {"ss_generator": FakeGenerator(), "slat_generator": FakeGenerator()}
        self.events: list[dict[str, Any]] = []
        self.log_path = log_path
        self.fail_in: str | None = None

    def _event(self, name: str, **fields: Any) -> None:
        event = {"event": name, **fields, "grad": torch.is_grad_enabled(), "det": det_patch_installed()}   # model calls must not build graphs
        self.events.append(event)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        if self.fail_in == name:
            raise RuntimeError(f"injected failure in {name}")

    def compute_pointmap(self, image: np.ndarray, pointmap: Any = None) -> dict[str, Any]:
        image = np.asarray(image)
        if not (image[..., 3] > 0).any():
            raise ValueError("empty object mask")
        self._event("pointmap", shape=list(image.shape), alpha_sum=int(image[..., 3].astype(np.int64).sum()),
                    rgb_sum=int(image[..., :3].astype(np.int64).sum()), pointmap_given=pointmap is not None)
        return {"pointmap": torch.zeros(3, image.shape[0], image.shape[1]), "pts_color": torch.zeros(3, 1, 1)}

    def preprocess_image(self, image: np.ndarray, preprocessor: str, pointmap: Any = None) -> dict[str, Any]:
        self._event("preprocess", preprocessor=preprocessor, with_pointmap=pointmap is not None)
        return {"image": preprocessor}

    def sample_sparse_structure(self, ss_input_dict: dict[str, Any], inference_steps: int | None = None,
                                use_distillation: bool = False) -> dict[str, Any]:
        print("stdout noise from the sparse-structure sampler")
        self._event("sparse_structure", steps=inference_steps, use_distillation=use_distillation,
                    solvers=[self.models["ss_generator"]._solver_method, self.models["slat_generator"]._solver_method],
                    noise=float(torch.rand(())))
        return {"coords": torch.zeros(4, 4, dtype=torch.int32)}

    def sample_slat(self, slat_input: dict[str, Any], coords: Any, inference_steps: int | None = 25,
                    use_distillation: bool = False) -> Any:
        self._event("structured_latent", steps=inference_steps, use_distillation=use_distillation,
                    solver=self.models["slat_generator"]._solver_method)
        return SimpleNamespace(feats=torch.arange(16, dtype=torch.float32).reshape(2, 8))

    def decode_slat(self, slat: Any, formats: list[str]) -> dict[str, Any]:
        self._event("decode", formats=list(formats))
        flexicubes = sys.modules.get(det.FLEXICUBES_MODULE)
        if flexicubes is not None:                                # the mesh extractor, as SparseFeatures2Mesh calls it
            flexicubes.FlexiCubes()(torch.zeros(8, 3), torch.zeros(8))
        vertices, faces = box_mesh()
        return {"mesh": [SimpleNamespace(vertices=torch.from_numpy(vertices), faces=torch.from_numpy(faces))]}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the full run() bakes textures and runs layout post-optimization")

    def postprocess_slat_output(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("texture/GLB baking is not part of the geometry-only profile")

    def run_post_optimization(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("layout post-optimization is not part of the geometry-only profile")

    def pose_decoder(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("layout pose decoding is not part of the geometry-only profile")


def make_fake_pipeline(config_path: str, device: str) -> FakeSam3dPipeline:
    print("stdout noise while loading")
    if config_path == "fail":
        raise RuntimeError("boom: injected load failure")
    for name, module in make_fake_sam3d_modules().items():   # the worker process has no real sam3d_objects
        sys.modules.setdefault(name, module)
    pipeline = FakeSam3dPipeline(os.environ.get("FAKE_SAM3D_LOG"))
    pipeline._event("load", config=config_path, device=device)
    return pipeline
