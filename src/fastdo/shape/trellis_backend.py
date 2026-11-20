"""TRELLIS image-to-3D shape backend, the open-weight default (D4), under `fastdo.shape.backend`'s geometry-only profile.

`TrellisBackend.reconstruct` runs the body of TRELLIS's `TrellisImageTo3DPipeline.run` step by step, so each phase is
timed:
1. `preprocess_image` and `get_cond`.
2. The sparse-structure flow.
3. The structured-latent flow.
4. `decode_slat(formats=["mesh"])`.

The input is a square RGBA crop around the mask with alpha = mask, so TRELLIS crops by alpha and never loads its
background remover. There is no Gaussian or radiance-field decode, no `to_glb` texture baking, and no layout step.
The mesh goes through `sam3d_worker.geometry_postprocess` with TRELLIS's `postprocess_mesh` (the simplification and
hole filling `to_glb` applies), so both backends produce meshes the same way. TRELLIS's samplers are Euler flow
samplers; only the step counts are configurable. Every model call runs under `torch.no_grad()`, as upstream `run()` is
decorated: TRELLIS's sparse-structure decoder and `decode_slat` would otherwise build autograd graphs.

Nothing loads at construction. The pipeline loads on the first `reconstruct` or `load()` and is reused for every later
clip. Loading needs the `trellis` package, which this repository does not install. The pipeline weights come from the
local Hugging Face cache only (`resolve_trellis_weights`), so the engine never downloads them. TRELLIS itself loads its
DINOv2 conditioning encoder through `torch.hub`, which is not guarded here: populate `torch.hub.get_dir()` with that
repository and its weights before an engine run on a host without network access.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from fastdo.shape.backend import ShapeConfig, ShapeResult, masked_crop_rgba, register_shape_backend
from fastdo.shape.sam3d_worker import PhaseTimer, cuda_sync, geometry_postprocess

__all__ = ["TrellisBackend", "load_trellis_pipeline", "resolve_trellis_weights"]

logger = logging.getLogger(__name__)

PostprocessFn = Callable[..., tuple[Any, Any]]
_INSTALL_HINT = "install TRELLIS (github.com/microsoft/TRELLIS) in this environment to use the 'trellis' shape backend"


def resolve_trellis_weights(model_id: str, snapshot_fn: Callable[..., str] | None = None) -> str:
    """The local directory of the TRELLIS pipeline: `model_id` itself if it is a directory, else its snapshot in the
    local Hugging Face cache (`local_files_only=True`, never a download).

    Raises:
        RuntimeError: If `model_id` is neither a directory nor cached.
    """
    if os.path.isdir(model_id):
        return model_id
    if snapshot_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_fn = snapshot_download
    try:
        return str(snapshot_fn(repo_id=model_id, local_files_only=True))
    except (OSError, ValueError) as exc:  # LocalEntryNotFoundError is both
        raise RuntimeError(
            f"TRELLIS weights {model_id!r} are not in the local Hugging Face cache ({type(exc).__name__}: {exc}); "
            f"download them outside the engine (huggingface-cli download {model_id}) before a run") from exc


def load_trellis_pipeline(model_id: str, device: str, snapshot_fn: Callable[..., str] | None = None) -> Any:
    """`TrellisImageTo3DPipeline.from_pretrained(<local weights of model_id>)` moved to `device`.

    Raises:
        RuntimeError: If the `trellis` package is not importable or the weights are not cached locally.
    """
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except ImportError as exc:
        raise RuntimeError(f"TRELLIS is not installed: {_INSTALL_HINT}") from exc
    pipeline = TrellisImageTo3DPipeline.from_pretrained(resolve_trellis_weights(model_id, snapshot_fn))
    pipeline.to(torch.device(device))
    return pipeline


def _trellis_postprocess_mesh() -> PostprocessFn:
    try:
        from trellis.utils.postprocessing_utils import postprocess_mesh
    except ImportError as exc:
        raise RuntimeError(f"TRELLIS is not installed: {_INSTALL_HINT}") from exc
    return postprocess_mesh  # type: ignore[no-any-return]


@register_shape_backend("trellis")
class TrellisBackend:
    """The `trellis` shape backend (see the module docstring).

    Args:
        cfg: Shape settings; `cfg.solver` must be `"euler"`.
        pipeline: An already-loaded `TrellisImageTo3DPipeline`-compatible object (tests inject fakes).
        loader: Zero-argument callable returning the pipeline, called once on first use instead of
            `load_trellis_pipeline`.
        postprocess_fn: `postprocess_mesh` replacement; defaults to TRELLIS's own.

    Raises:
        ValueError: If `cfg.solver` is not `"euler"`, or `cfg.deterministic` is True (the F2 patch is SAM 3D's only).
    """

    def __init__(self, cfg: ShapeConfig | None = None, pipeline: Any = None, loader: Callable[[], Any] | None = None,
                 postprocess_fn: PostprocessFn | None = None) -> None:
        self.cfg = cfg if cfg is not None else ShapeConfig()
        if self.cfg.solver != "euler":
            raise ValueError(f"TRELLIS's flow samplers are Euler-only (solver='euler'); got solver={self.cfg.solver!r}")
        if self.cfg.deterministic:
            raise ValueError("deterministic=True is implemented for the sam3d backend only; TRELLIS is not patched")
        self._pipeline = pipeline
        self._loader = loader
        self._postprocess_fn = postprocess_fn

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> Any:
        """The pipeline, loaded on first call and reused."""
        if self._pipeline is None:
            loader = self._loader or (lambda: load_trellis_pipeline(self.cfg.trellis_model, self.cfg.device))
            self._pipeline = loader()
            logger.info("TRELLIS pipeline %s loaded on %s", self.cfg.trellis_model, self.cfg.device)
        return self._pipeline

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        from PIL import Image

        rgba = masked_crop_rgba(image, mask)
        pipeline = self.load()
        postprocess = self._postprocess()
        timer = PhaseTimer(cuda_sync(self.cfg.device))
        with torch.no_grad():                     # upstream run()'s decorator; decoders are not no_grad on their own
            with timer.phase("preprocess"):
                cond = pipeline.get_cond([pipeline.preprocess_image(Image.fromarray(rgba))])
            torch.manual_seed(self.cfg.seed)
            with timer.phase("sparse_structure_flow"):
                coords = pipeline.sample_sparse_structure(cond, 1, {"steps": self.cfg.sparse_structure_steps})
            with timer.phase("structured_latent_flow"):
                slat = pipeline.sample_slat(cond, coords, {"steps": self.cfg.structured_latent_steps})
            with timer.phase("decode"):
                mesh = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
            with timer.phase("postprocess"):
                vertices, faces = geometry_postprocess(mesh.vertices, mesh.faces, postprocess)
        logger.info("TRELLIS mesh: %d vertices, %d faces, phases %s", len(vertices), len(faces),
                    {name: round(seconds, 3) for name, seconds in timer.seconds.items()})
        return ShapeResult(vertices, faces, None, dict(timer.seconds))

    def _postprocess(self) -> PostprocessFn | None:
        if not self.cfg.mesh_postprocess:
            return None
        if self._postprocess_fn is None:
            self._postprocess_fn = _trellis_postprocess_mesh()
        return self._postprocess_fn
