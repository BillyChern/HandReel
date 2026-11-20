"""SAM 3D Objects geometry-only driver and its persistent worker process (D4).

SAM 3D Objects needs its own environment (the baseline's `sam3d` conda env with pytorch3d, kaolin and spconv), so this
file runs as a script under that environment's interpreter:

    <sam3d python> sam3d_worker.py --config <snapshot>/checkpoints/pipeline.yaml [--device cuda] [--min-gpu-gb 19]
        [--hf-revision <repo id>=<commit> ...]

It imports nothing from `fastdo`. `fastdo.shape.trellis_backend` imports its geometry helpers (`PhaseTimer`,
`geometry_postprocess`, `GEOMETRY_POSTPROCESS`) so both backends share one profile. The worker applies the engine's
CUDA cap, refuses to start when that cap is below `--min-gpu-gb` (`require_gpu_cap`), instantiates the pipeline YAML
only after `check_hydra_targets` accepts every `_target_`, then loads the pipeline once and serves until shutdown.
`--hf-revision` pins a Hugging Face loader of the pipeline YAML (`pin_hf_revisions`): SAM 3D's MoGe v1 pointmap model
is `moge.model.v1.MoGeModel.from_pretrained` of `Ruicheng/moge-vitl`, which reads `main` unless given a `revision`.
A repository given to `--hf-revision` that no loader of the YAML reads stops the load (nothing reads `main`).

`reconstruct_geometry` is the pipeline's `run()` reduced to geometry, each phase timed and every call under
`torch.no_grad()`. It is the only place the geometry-only profile is spelled out:
- the pointmap, then preprocessing;
- the sparse-structure flow, then the structured-latent flow, at the configured step counts and ODE update;
- a mesh-only decode;
- the mesh half of `to_glb`: simplification and hole filling when enabled, the z-up to y-up rotation, and trimesh's
  vertex merge.
It skips pose decoding, the Gaussian decode, texture baking and layout post-optimization.

Protocol, one JSON object per line:
- Worker to client, once: `{"ready": true, "load_seconds": s, "spconv_kernels": "pinned" | "tuner"}`, or
  `{"ready": false, "error": "..."}` followed by exit 1. `--spconv-kernels pinned` installs
  `sam3d_spconv_pin.pin_spconv_kernels` for the process lifetime before the pipeline load; a failed
  precondition is a failed start. `tuner` (default) keeps spconv's timing tuner.
- Client to worker: `{"cmd": "reconstruct", "id": n, "input": <npz with image (H,W,3) uint8 and mask (H,W) bool>,
  "output": <npz path>, "params": {seed, sparse_structure_steps, structured_latent_steps, solver, mesh_postprocess,
  [deterministic], [digest], [mesh_postprocess_impl], [fast_decimator_min_raw_faces], [min_component_area_frac], [wall_qc],
  [wall_qc_max_removed], [wall_qc_min_area_ratio], [sam3d_cuda_graph_ss], [sam3d_cuda_graph_slat]}}`. `deterministic`
  (default false) runs the structured-latent flow and the decode under `deterministic_sam3d`; `digest` (default false) adds sha256 `digests` of
  the structured latent's features, the FlexiCubes input and the raw mesh, plus `params` (`request_params_digest`)
  and `kernels` (`pinned_kernels()`, {} under the tuner). After a QC fallback those four describe the fallback's mesh, the one that
  is stored, and the discarded primary pass's are kept beside them under `primary_*`. The fast-path options
  are per request, so one process can serve every EVAL arm, and each defaults to today's path
  (`resolve_shape_options`), which refuses anything `ShapeConfig` would refuse rather than coercing it. The worker
  writes `vertices` (V,3) float64 and `faces` (F,3) int64 to `output`, then replies `{"id": n, "ok": true,
  "phase_seconds": {...}, "peak_gpu_gib": g, "peak_gpu_gib_reserved": r}`, the GATED pass's peak allocated and
  reserved CUDA memory in GiB (null off CUDA). A fast
  post-process adds `postprocess_stats` (the decimator, the face counts, `removed`, `area_ratio` and the `qc`
  outcome), and a QC fallback adds `fallback_phase_seconds` plus `fallback_peak_gpu_gib`(`_reserved`), each kept
  apart from the gated pass's own (seconds and peaks). `sam3d_cuda_graph_ss`
  (default false) replays the sparse-structure backbone from CUDA graphs (`sam3d_cuda_graph`), and the reply
  adds `cuda_graph_ss`: the request's `captures`, `capture_seconds` (included in `sparse_structure_flow`), `replays`
  and `eager_calls`, summed over a QC fallback's pass. `sam3d_cuda_graph_slat` (default false) replays the
  structured-latent flow's transformer blocks from per-request CUDA graphs (`sam3d_cuda_graph_slat`), and the reply
  adds `cuda_graph_slat` with the same counters plus `release_seconds`, all included in `structured_latent_flow`. On
  failure it replies
  `{"id": n, "ok": false, "error": "...", "peak_gpu_gib": g, "peak_gpu_gib_reserved": r}` - the peak since the last
  reset, gated or not, since a failed request is no arm's measurement - and keeps serving.
- `{"cmd": "shutdown"}`, or end of input: the worker exits 0.

Stdout carries only the protocol. At startup the worker moves the protocol to a private duplicate of fd 1 and points fd
1 at stderr, so anything the pipeline prints goes to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any

import numpy as np

if __package__:
    from . import (
        fast_postprocess,
        sam3d_cuda_graph,
        sam3d_cuda_graph_slat,
        sam3d_determinism,
        sam3d_multistep,
        sam3d_spconv_pin,
        wall_qc,
    )
else:                                   # run as a script under the SAM 3D env: the sibling files, never `fastdo`
    import fast_postprocess  # type: ignore[no-redef]
    import sam3d_cuda_graph  # type: ignore[no-redef]
    import sam3d_cuda_graph_slat  # type: ignore[no-redef]
    import sam3d_determinism  # type: ignore[no-redef]
    import sam3d_multistep  # type: ignore[no-redef]
    import sam3d_spconv_pin  # type: ignore[no-redef]
    import wall_qc  # type: ignore[no-redef]

if TYPE_CHECKING:                       # the sibling import above is a module object to the checker, not a namespace
    from fastdo.shape.wall_qc import WallQcThresholds

deterministic_sam3d = sam3d_determinism.deterministic_sam3d

__all__ = [
    "AB2R_RIGHT_LIMIT_EPS",
    "GEOMETRY_POSTPROCESS",
    "HYDRA_TARGET_PACKAGES",
    "SAM3D_MIN_GPU_GB",
    "SLAT_GRAPH_STATS_PREFIX",
    "Z_UP_TO_Y_UP",
    "PhaseTimer",
    "ShapeOptions",
    "apply_gpu_cap",
    "check_hydra_targets",
    "cuda_sync",
    "deterministic_sam3d",
    "geometry_postprocess",
    "load_sam3d_pipeline",
    "main",
    "parse_hf_revisions",
    "pin_hf_revisions",
    "reconstruct_geometry",
    "request_params_digest",
    "require_gpu_cap",
    "resolve_shape_options",
    "serve",
    "solver_override",
    "sparse_structure_ab2r",
    "structured_latent_ab2r",
]

logger = logging.getLogger(__name__)

# `to_glb(simplify=0.95, fill_holes=True, fill_holes_max_size=0.04)`'s call to `postprocess_mesh`, in both SAM 3D's
# and TRELLIS's `postprocessing_utils`.
GEOMETRY_POSTPROCESS: dict[str, Any] = {
    "simplify": True,
    "simplify_ratio": 0.95,
    "fill_holes": True,
    "fill_holes_max_hole_size": 0.04,
    "fill_holes_max_hole_nbe": int(250 * math.sqrt(1 - 0.95)),
    "fill_holes_resolution": 1024,
    "fill_holes_num_views": 1000,
    "verbose": False,
}
# `to_glb`'s final rotation of row-vector vertices: `vertices @ Z_UP_TO_Y_UP`.
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
GPU_MEM_ENV = "FASTDO_GPU_MEM_GB"      # the same variable and default as fastdo.bench.gpu_budget
DEFAULT_GPU_MEM_GB = 12.0
SAM3D_MIN_GPU_GB = 19.0                # SAM 3D Objects' measured peak is 18.67 GiB allocated (whisking clip)
SLAT_GRAPH_STATS_PREFIX = "slat_"      # SLAT runner counters in `graph_stats`, apart from the SS ones
AB2R_RIGHT_LIMIT_EPS = 1e-6            # the right-limited AB2's eps
STRUCTURED_LATENT_SOLVERS = ("ab2r",)  # `backend.STRUCTURED_LATENT_SOLVERS`, repeated: this file imports no `fastdo`
SPARSE_STRUCTURE_SOLVERS = ("ab2r",)   # `backend.SPARSE_STRUCTURE_SOLVERS`, likewise repeated
# `notebook/inference.py`'s WHITELIST_FILTERS: the packages a pipeline YAML `_target_` may come from.
HYDRA_TARGET_PACKAGES = frozenset({"sam3d_objects", "torch", "torchvision", "moge"})

PostprocessFn = Callable[..., tuple[Any, Any]]


class PhaseTimer:
    """Wall seconds per named phase. `sync` (e.g. `torch.cuda.synchronize`) runs at both ends of every phase, so
    queued GPU work is charged to the phase that queued it."""

    def __init__(self, sync: Callable[[], None] | None = None) -> None:
        self.sync = sync
        self.seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.sync is not None:
            self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync is not None:
                self.sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


def cuda_sync(device: str) -> Callable[[], None] | None:
    """`torch.cuda.synchronize` when `device` is a CUDA device and CUDA is available, else None."""
    if not str(device).startswith("cuda"):
        return None
    import torch

    return torch.cuda.synchronize if torch.cuda.is_available() else None


def _as_array(x: Any, dtype: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def geometry_postprocess(vertices: Any, faces: Any, postprocess_fn: PostprocessFn | None) -> tuple[np.ndarray, np.ndarray]:
    """The mesh half of `to_glb` without its texture baking.

    Applies `postprocess_fn(vertices, faces, **GEOMETRY_POSTPROCESS)` when given, then the z-up to y-up rotation, then
    `trimesh.Trimesh`'s default processing (vertex merge), as `to_glb` builds its mesh. Returns
    `(vertices (V,3) float64, faces (F,3) int64)`.
    """
    import trimesh

    v, f = _as_array(vertices, np.float32), _as_array(faces, np.int64)
    if postprocess_fn is not None:
        v, f = postprocess_fn(v, f, **GEOMETRY_POSTPROCESS)
    mesh = trimesh.Trimesh(np.asarray(v) @ Z_UP_TO_Y_UP, np.asarray(f))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


@contextmanager
def solver_override(generators: Iterable[Any], solver: str) -> Iterator[None]:
    """Switches each flow generator's ODE update (`FlowMatching._get_solver`) to `solver`, restoring it on exit."""
    saved = [(generator, generator._solver_method, generator._solver) for generator in generators]
    try:
        for generator, method, _ in saved:
            if method != solver:
                generator._solver_method, generator._solver = generator._get_solver(solver, {})
        yield
    finally:
        for generator, method, instance in saved:
            generator._solver_method, generator._solver = method, instance


@contextmanager
def structured_latent_ab2r(pipeline: Any, generator: Any) -> Iterator[None]:
    """The structured-latent generator's ODE update replaced by the right-limited AB2 for one block, restored on exit.

    The regime is SAM 3D's guidance
    switch on the SLAT generator (`cfg_regime(reverse_fn.interval, time_scale, pipeline.slat_cfg_strength)`).
    """
    saved = generator._solver
    regime = sam3d_multistep.cfg_regime(generator.reverse_fn.interval, generator.time_scale,
                                        pipeline.slat_cfg_strength)
    generator._solver = sam3d_multistep.AdamsBashforth2(regime, right_limit_eps=AB2R_RIGHT_LIMIT_EPS)
    try:
        yield
    finally:
        generator._solver = saved


@contextmanager
def sparse_structure_ab2r(pipeline: Any, generator: Any) -> Iterator[None]:
    """The sparse-structure generator's ODE update replaced by the right-limited AB2 for one block, restored on exit.

    `structured_latent_ab2r` on the SS flow: the regime is SAM 3D's guidance switch on the SS generator
    (`cfg_regime(reverse_fn.interval, time_scale, pipeline.ss_cfg_strength)`). The ShortCut generator's guided path
    samples through `self._solver.solve_iter`, the interface the SLAT substitution replaces.
    """
    saved = generator._solver
    regime = sam3d_multistep.cfg_regime(generator.reverse_fn.interval, generator.time_scale,
                                        pipeline.ss_cfg_strength)
    generator._solver = sam3d_multistep.AdamsBashforth2(regime, right_limit_eps=AB2R_RIGHT_LIMIT_EPS)
    try:
        yield
    finally:
        generator._solver = saved


def _sha256(tensors: Sequence[Any]) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        array = np.ascontiguousarray(_as_array(tensor, None))
        h.update(f"{array.dtype.str}{array.shape}".encode() + array.tobytes())
    return h.hexdigest()


def _sam3d_postprocess_mesh() -> PostprocessFn:
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import postprocess_mesh

    return postprocess_mesh  # type: ignore[no-any-return]


@dataclass(frozen=True)
class ShapeOptions:
    """One request's resolved fast-path options."""

    postprocess_impl: str
    fast_decimator_min_raw_faces: int
    wall_qc: str
    thresholds: WallQcThresholds
    min_component_area_frac: float = fast_postprocess.DEFAULT_MIN_COMPONENT_AREA_FRAC
    cuda_graph_ss: bool = False         # output-preserving, so not a mesh parameter (request_params_digest)
    cuda_graph_slat: bool = False       # likewise
    ss_shortcut: bool = False           # the sparse-structure flow's shortcut path (a mesh parameter)
    slat_solver: str | None = None      # the structured-latent flow's own update (a mesh parameter)
    ss_solver: str | None = None        # the sparse-structure flow's own update (a mesh parameter)


def _is_int(value: Any) -> bool:
    """A plain int, not a bool: `ShapeConfig`'s rule, repeated because the worker imports nothing from `fastdo`."""
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_shape_options(params: Mapping[str, Any]) -> ShapeOptions:
    """The fast-path options of one request, each defaulting to today's path when the key is absent.

    Nothing is coerced: this boundary fails closed on exactly what `ShapeConfig` refuses, with the same messages.
    The EVAL drives the worker directly, so a value that only `ShapeConfig` would catch
    never reaches a check otherwise - and a coerced `fast_decimator_min_raw_faces` of 1.9 truncates to 1, which
    with `mesh_postprocess_impl='fast'` routes *every* mesh to `fast_simplification` and silently inverts arm
    A0v's VTK-only mode: a swapped arm, not a rejected request.

    Raises:
        ValueError: On an unknown implementation or QC mode, a threshold that is not a finite number in range, a
            `fast_decimator_min_raw_faces` that is not an int >= 0, `wall_qc` without the instrumented fast
            post-process that produces the statistics it reads, or the fast path with `mesh_postprocess=False`.
    """
    impl = params.get("mesh_postprocess_impl", "official")
    if impl not in fast_postprocess.POSTPROCESS_IMPLS:
        raise ValueError(f"mesh_postprocess_impl must be one of {fast_postprocess.POSTPROCESS_IMPLS}, got {impl!r}")
    # Without this the worker answered `ok=True` with `postprocess_stats={"qc": "pass"}` for a mesh no
    # post-process ever touched - no decimator, no removed count, no area ratio - and an EVAL reading that record
    # would score a pass that never happened.
    if impl == "fast" and not params.get("mesh_postprocess", True):
        raise ValueError("mesh_postprocess_impl='fast' needs mesh_postprocess=True: with the post-process off "
                         "there is no decimation, mincut or hole filling for the fast path to do")
    mode = params.get("wall_qc", "off")
    if mode not in wall_qc.WALL_QC_MODES:
        raise ValueError(f"wall_qc must be one of {wall_qc.WALL_QC_MODES}, got {mode!r}")
    if mode != "off" and impl != "fast":
        raise ValueError(f"wall_qc={mode!r} needs mesh_postprocess_impl='fast': only the instrumented fast "
                         "post-process reports the mincut-removed fraction and the area ratio it thresholds")
    thresholds = wall_qc.WallQcThresholds(params.get("wall_qc_max_removed", wall_qc.DEFAULT_MAX_REMOVED),
                                          params.get("wall_qc_min_area_ratio", wall_qc.DEFAULT_MIN_AREA_RATIO))
    min_raw = params.get("fast_decimator_min_raw_faces", fast_postprocess.DEFAULT_MIN_RAW_FACES)
    if not _is_int(min_raw) or min_raw < 0:
        raise ValueError(f"fast_decimator_min_raw_faces must be an int >= 0, got {min_raw!r}")
    frac = fast_postprocess.check_min_component_area_frac(
        params.get("min_component_area_frac", fast_postprocess.DEFAULT_MIN_COMPONENT_AREA_FRAC))
    if frac > 0 and impl != "fast":
        raise ValueError("min_component_area_frac > 0 needs mesh_postprocess_impl='fast': the fragment filter is "
                         "a step of the fast post-process")
    graph = params.get("sam3d_cuda_graph_ss", False)
    if not isinstance(graph, bool):
        raise ValueError(f"sam3d_cuda_graph_ss must be a bool, got {graph!r}")  # noqa: TRY004  # as ShapeConfig
    graph_slat = params.get("sam3d_cuda_graph_slat", False)
    if not isinstance(graph_slat, bool):
        raise ValueError(f"sam3d_cuda_graph_slat must be a bool, got {graph_slat!r}")  # noqa: TRY004  # as ShapeConfig
    # `ShapeConfig._check_sampler_options`'s messages, string for string (its backend check has no twin
    # here: this worker is the sam3d backend).
    shortcut = params.get("sparse_structure_shortcut", False)
    if not isinstance(shortcut, bool):
        raise ValueError(f"sparse_structure_shortcut must be a bool, got "  # noqa: TRY004  # as ShapeConfig
                         f"{shortcut!r}")
    slat_solver = params.get("structured_latent_solver")
    if slat_solver is not None and slat_solver not in STRUCTURED_LATENT_SOLVERS:
        raise ValueError(f"structured_latent_solver must be None or one of {STRUCTURED_LATENT_SOLVERS}, got "
                         f"{slat_solver!r}")
    solver = params.get("solver", "euler")
    if (shortcut or slat_solver is not None) and solver != "euler":
        raise ValueError(f"sparse_structure_shortcut and structured_latent_solver are measured with solver='euler' "
                         f"only, got solver {solver!r}")
    ss_solver = params.get("sparse_structure_solver")   # likewise, string for string
    if ss_solver is not None and ss_solver not in SPARSE_STRUCTURE_SOLVERS:
        raise ValueError(f"sparse_structure_solver must be None or one of {SPARSE_STRUCTURE_SOLVERS}, got "
                         f"{ss_solver!r}")
    if ss_solver is not None and solver != "euler":
        raise ValueError(f"sparse_structure_solver is measured with solver='euler' only, got solver "
                         f"{solver!r}")
    if ss_solver is not None and shortcut:
        raise ValueError(f"sparse_structure_solver={ss_solver!r} and "
                         "sparse_structure_shortcut=True are mutually exclusive: the shortcut path samples a "
                         "different (distilled, unguided) flow")
    return ShapeOptions(impl, min_raw, mode, thresholds, frac, graph, graph_slat, shortcut, slat_solver, ss_solver)


def request_params_digest(params: Mapping[str, Any]) -> str:
    """sha256 over every resolved parameter that can change the mesh, so a record names the arm that produced it.

    Resolved, not literal: passing an option at its default gives the digest of a request that omits it. `digest`
    itself is left out, since asking for extra reporting cannot change the output, and so are `sam3d_cuda_graph_ss`
    and `sam3d_cuda_graph_slat`, which replay the same kernels and are gated on bit-identical
    meshes.

    Precisely, it names the arm that was REQUESTED. After a wall-QC fallback the stored mesh came from the 25+25 Euler
    path the fallback ran, not from this arm; `postprocess_stats["qc"]` says so, and the reply's `primary_*` digests and
    statistics separate the discarded pass from the one that was kept.
    """
    options = resolve_shape_options(params)
    resolved = {"seed": int(params["seed"]),
                "sparse_structure_steps": int(params["sparse_structure_steps"]),
                "structured_latent_steps": int(params["structured_latent_steps"]),
                "solver": str(params["solver"]), "mesh_postprocess": bool(params["mesh_postprocess"]),
                "deterministic": bool(params.get("deterministic", False)),
                "mesh_postprocess_impl": options.postprocess_impl,
                "fast_decimator_min_raw_faces": options.fast_decimator_min_raw_faces,
                "wall_qc": options.wall_qc, "wall_qc_max_removed": options.thresholds.max_removed,
                "wall_qc_min_area_ratio": options.thresholds.min_area_ratio}
    if options.min_component_area_frac != fast_postprocess.DEFAULT_MIN_COMPONENT_AREA_FRAC:
        # Only when set, so every request made without it keeps its recorded digest.
        resolved["min_component_area_frac"] = options.min_component_area_frac
    if options.ss_shortcut:             # likewise, so every earlier request keeps its recorded digest
        resolved["sparse_structure_shortcut"] = True
    if options.slat_solver is not None:
        resolved["structured_latent_solver"] = options.slat_solver
    if options.ss_solver is not None:   # likewise
        resolved["sparse_structure_solver"] = options.ss_solver
    return hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _call_kwargs(params: Mapping[str, Any], options: ShapeOptions, sync: Callable[[], None] | None,
                 digests: dict[str, str] | None, stats: dict[str, Any] | None,
                 graph_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """`reconstruct_geometry`'s keyword arguments for one request."""
    return {"seed": int(params["seed"]), "sparse_structure_steps": int(params["sparse_structure_steps"]),
            "structured_latent_steps": int(params["structured_latent_steps"]), "solver": str(params["solver"]),
            "mesh_postprocess": bool(params["mesh_postprocess"]), "sync": sync,
            "deterministic": bool(params.get("deterministic", False)), "digests": digests,
            "postprocess_impl": options.postprocess_impl,
            "fast_decimator_min_raw_faces": options.fast_decimator_min_raw_faces,
            "min_component_area_frac": options.min_component_area_frac, "stats": stats,
            "cuda_graph_ss": options.cuda_graph_ss, "cuda_graph_slat": options.cuda_graph_slat,
            "graph_stats": graph_stats, "sparse_structure_shortcut": options.ss_shortcut,
            "structured_latent_solver": options.slat_solver, "sparse_structure_solver": options.ss_solver}


def _fast_postprocess_fn(min_raw_faces: int, stats: dict[str, Any] | None,
                         min_component_area_frac: float = fast_postprocess.DEFAULT_MIN_COMPONENT_AREA_FRAC
                         ) -> PostprocessFn:
    """`fast_postprocess` bound to this request's decimator threshold, fragment filter and statistics dict."""

    def run(vertices: Any, faces: Any, **kwargs: Any) -> tuple[Any, Any]:
        if min_component_area_frac == fast_postprocess.DEFAULT_MIN_COMPONENT_AREA_FRAC:   # today's call, unchanged
            return fast_postprocess.fast_postprocess(vertices, faces, min_raw_faces=min_raw_faces, stats=stats,
                                                     **kwargs)
        return fast_postprocess.fast_postprocess(vertices, faces, min_raw_faces=min_raw_faces, stats=stats,
                                                 min_component_area_frac=min_component_area_frac, **kwargs)

    return run


def _add_graph_stats(graph_stats: dict[str, Any] | None, runner: Any, prefix: str = "") -> None:
    """Adds a graph runner's counters (none when `runner` is None) into `graph_stats`, each name after `prefix`."""
    taken = runner.take_stats() if runner is not None else {}
    if graph_stats is not None:
        for name, value in taken.items():
            graph_stats[prefix + name] = graph_stats.get(prefix + name, 0) + value


def reconstruct_geometry(pipeline: Any, image: np.ndarray, mask: np.ndarray, *, seed: int, sparse_structure_steps: int,
                         structured_latent_steps: int, solver: str, mesh_postprocess: bool,
                         postprocess_fn: PostprocessFn | None = None, sync: Callable[[], None] | None = None,
                         deterministic: bool = False, digests: dict[str, str] | None = None,
                         postprocess_impl: str = "official",
                         fast_decimator_min_raw_faces: int = fast_postprocess.DEFAULT_MIN_RAW_FACES,
                         stats: dict[str, Any] | None = None,
                         min_component_area_frac: float = fast_postprocess.DEFAULT_MIN_COMPONENT_AREA_FRAC,
                         cuda_graph_ss: bool = False, graph_stats: dict[str, Any] | None = None,
                         cuda_graph_slat: bool = False, sparse_structure_shortcut: bool = False,
                         structured_latent_solver: str | None = None, sparse_structure_solver: str | None = None
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """One object's mesh from an `InferencePipelinePointMap` under the geometry-only profile.

    Args:
        pipeline: A loaded SAM 3D Objects pipeline.
        image: `(H,W,3)` uint8 RGB frame; the full frame, as the baseline passes it (the pointmap needs it).
        mask: `(H,W)` bool object mask, packed into the alpha channel.
        seed: `torch.manual_seed` after preprocessing and before the flows, where `run()` seeds.
        sparse_structure_steps: Sparse-structure flow steps.
        structured_latent_steps: Structured-latent flow steps.
        solver: `"euler"` or another `FlowMatching.SOLVER_METHODS` key (e.g. `"midpoint"`), for both flows.
        mesh_postprocess: Apply the baseline's simplification and hole filling.
        postprocess_fn: `postprocess_mesh` replacement; defaults to SAM 3D's own when `mesh_postprocess`.
        sync: Called at phase boundaries (see `PhaseTimer`).
        deterministic: Run exactly the `structured_latent_flow` and `decode` phases under `deterministic_sam3d`; the
            pointmap, the sparse-structure flow and the postprocess stay unpatched, and any overhead is charged to
            those two phases.
        digests: When given, filled with sha256 hex digests `slat_feats`, `flexicubes_input`, `mesh_vertices` and
            `mesh_faces` (the raw decoded mesh), hashed outside the timed phases.
        postprocess_impl: `"official"` (SAM 3D's `postprocess_mesh`, the default) or `"fast"`
            (`fast_postprocess`); ignored when `postprocess_fn` is given.
        fast_decimator_min_raw_faces: The fast path's decimator threshold (`fast_postprocess.choose_decimator`).
        min_component_area_frac: The fast path's fragment filter (0.0 = off, today's path).
        stats: When given and the fast path runs, filled with its decimator, face counts, refused cut loops, per-step
            seconds and wall-QC statistics.
        cuda_graph_ss: Run the sparse-structure flow's backbone from CUDA graphs (`sam3d_cuda_graph`);
            capture happens on the first call per signature and is charged to `sparse_structure_flow`.
        graph_stats: When given, the graph runners' counters for this call are added into it: `cuda_graph_ss`'s under
            their own names, `cuda_graph_slat`'s under `SLAT_GRAPH_STATS_PREFIX` + name.
        cuda_graph_slat: Run the structured-latent flow's transformer blocks from CUDA graphs captured for this call
            and dropped at the end of its phase (`sam3d_cuda_graph_slat`); capture and release are charged
            to `structured_latent_flow`.
        sparse_structure_shortcut: Sample the sparse-structure flow through the checkpoint's shortcut path
            (`use_distillation=True`: step-size conditioning, guidance off) at `sparse_structure_steps` steps.
            False is today's call, `use_distillation=False`.
        structured_latent_solver: None (today) or `"ab2r"`: the structured-latent flow under
            `structured_latent_ab2r`, inside the flow's timed phase.
        sparse_structure_solver: None (today) or `"ab2r"`: the sparse-structure flow under `sparse_structure_ab2r`,
            inside the flow's timed phase and inside `cuda_graph_ss`'s graph context (the
            graph replays the backbone; the solver only chooses where it is evaluated). Refused together with
            `sparse_structure_shortcut`.

    Returns:
        `(vertices (V,3) float64, faces (F,3) int64, phase seconds)`, with phases `pointmap`, `preprocess`,
        `sparse_structure_flow`, `structured_latent_flow`, `decode` and `postprocess`.
    """
    import torch

    image, mask = np.asarray(image), np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] < 3 or mask.shape != image.shape[:2]:
        raise ValueError(f"image must be (H,W,3) and mask (H,W) of the same size, got {image.shape} and {mask.shape}")
    rgba = np.concatenate([image[..., :3].astype(np.uint8), mask.astype(np.uint8)[..., None] * 255], axis=-1)
    if mesh_postprocess and postprocess_fn is None:
        postprocess_fn = (_fast_postprocess_fn(fast_decimator_min_raw_faces, stats, min_component_area_frac)
                          if postprocess_impl == "fast" else _sam3d_postprocess_mesh())
    timer = PhaseTimer(sync)
    device = getattr(pipeline, "device", None)
    generators = (pipeline.models["ss_generator"], pipeline.models["slat_generator"])
    patch = deterministic_sam3d if deterministic else nullcontext
    graphed = ((lambda: sam3d_cuda_graph.graphed_ss_backbone(generators[0])) if cuda_graph_ss
               else nullcontext)
    graphed_slat = ((lambda: sam3d_cuda_graph_slat.graphed_slat_blocks(generators[1])) if cuda_graph_slat
                    else nullcontext)
    slat_update = ((lambda: structured_latent_ab2r(pipeline, generators[1])) if structured_latent_solver == "ab2r"
                   else nullcontext)
    if structured_latent_solver not in (None, "ab2r"):
        raise ValueError(f"structured_latent_solver must be None or one of {STRUCTURED_LATENT_SOLVERS}, got "
                         f"{structured_latent_solver!r}")
    ss_update = ((lambda: sparse_structure_ab2r(pipeline, generators[0])) if sparse_structure_solver == "ab2r"
                 else nullcontext)
    if sparse_structure_solver not in (None, "ab2r"):
        raise ValueError(f"sparse_structure_solver must be None or one of {SPARSE_STRUCTURE_SOLVERS}, got "
                         f"{sparse_structure_solver!r}")
    if sparse_structure_solver is not None and sparse_structure_shortcut:
        raise ValueError(f"sparse_structure_solver={sparse_structure_solver!r} and "
                         "sparse_structure_shortcut=True are mutually exclusive: the shortcut path samples a "
                         "different (distilled, unguided) flow")
    captured: list[Any] = []
    capture = (lambda: sam3d_determinism.capture_flexicubes_input(captured)) if digests is not None else nullcontext
    with torch.no_grad(),(device if isinstance(device, torch.device) else nullcontext()):
        with timer.phase("pointmap"):
            pointmap = pipeline.compute_pointmap(rgba, None)["pointmap"]
        with timer.phase("preprocess"):
            ss_input = pipeline.preprocess_image(rgba, pipeline.ss_preprocessor, pointmap=pointmap)
            slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
        torch.manual_seed(seed)
        with solver_override(generators, solver):
            with timer.phase("sparse_structure_flow"), graphed() as runner, ss_update():
                coords = pipeline.sample_sparse_structure(ss_input, inference_steps=sparse_structure_steps,
                                                          use_distillation=bool(sparse_structure_shortcut))["coords"]
            _add_graph_stats(graph_stats, runner)
            with timer.phase("structured_latent_flow"), patch(), graphed_slat() as slat_runner, slat_update():
                slat = pipeline.sample_slat(slat_input, coords, inference_steps=structured_latent_steps,
                                            use_distillation=False)
            _add_graph_stats(graph_stats, slat_runner, SLAT_GRAPH_STATS_PREFIX)
        with timer.phase("decode"), patch(), capture():
            mesh = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
        if digests is not None:
            digests.update(slat_feats=_sha256([slat.feats]), flexicubes_input=_sha256(captured),
                           mesh_vertices=_sha256([mesh.vertices]), mesh_faces=_sha256([mesh.faces]))
        with timer.phase("postprocess"):
            vertices, faces = geometry_postprocess(mesh.vertices, mesh.faces,
                                                   postprocess_fn if mesh_postprocess else None)
    return vertices, faces, dict(timer.seconds)


def _forbidden_targets() -> set[Any]:
    """`notebook/inference.py`'s BLACKLIST_FILTERS: code-execution, process and file-system primitives."""
    import builtins
    import shutil
    import subprocess

    names: dict[Any, tuple[str, ...]] = {
        builtins: ("exec", "eval", "__import__", "help"), shutil: ("rmtree", "move", "chown"), subprocess: ("Popen",),
        os: ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
             "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
             "chroot", "lchown", "getcwd", "chdir")}
    return {getattr(module, name) for module, attrs in names.items() for name in attrs if hasattr(module, name)}


def check_hydra_targets(config: Any, resolve: Callable[[str], Any] | None = None) -> None:
    """`notebook/inference.py`'s `check_hydra_safety`: the pipeline YAML sits in a user-writable cache.

    Walks every mapping and list (`DictConfig` and `ListConfig` included). A `_target_` passes when its first dotted
    segment is in `HYDRA_TARGET_PACKAGES` and `resolve(target)` (default `hydra.utils.get_method`, which imports it) is
    not a blacklisted primitive; the second check catches aliases such as `torch.os.system`.

    Raises:
        RuntimeError: On the first target that does not pass.
    """
    if resolve is None:
        from hydra.utils import get_method

        resolve = get_method
    forbidden = _forbidden_targets()
    stack: list[Any] = [config]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            stack.extend(node.values())
            target = node.get("_target_")
            if target is not None and (str(target).split(".", 1)[0] not in HYDRA_TARGET_PACKAGES
                                       or resolve(str(target)) in forbidden):
                raise RuntimeError(f"hydra target {str(target)!r} is not allowed to be instantiated (allowed packages "
                                   f"{sorted(HYDRA_TARGET_PACKAGES)}, no process or file-system primitives)")
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            stack.extend(node)


def parse_hf_revisions(items: Sequence[str]) -> dict[str, str]:
    """`["<repo id>=<commit>", ...]` -> `{repo id: commit}`.

    Raises:
        ValueError: On an item without `=`, an empty side, or a repository named twice.
    """
    out: dict[str, str] = {}
    for item in items:
        repo, sep, rev = item.partition("=")
        if not sep or not repo or not rev or repo in out:
            raise ValueError(f"--hf-revision takes distinct <repo id>=<commit> items, got {item!r}")
        out[repo] = rev
    return out


def pin_hf_revisions(config: Any, revisions: Mapping[str, str]) -> list[str]:
    """Sets `revision` on every `*.from_pretrained` node of `config` whose `pretrained_model_name_or_path` is a key of
    `revisions`; returns the repositories it pinned, in config order. A node that already names another revision is
    refused rather than overridden, and a requested repository that no node loads is refused too: its loader would
    otherwise read `main` unpinned (a YAML without the MoGe v1 `depth_model` node, say).

    Raises:
        RuntimeError: On a node whose own `revision` differs from the pin, or a key of `revisions` no node loads.
    """
    pinned: list[str] = []
    stack: list[Any] = [config]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            repo = node.get("pretrained_model_name_or_path")
            if str(node.get("_target_", "")).endswith(".from_pretrained") and repo in revisions:
                current = node.get("revision")
                if current is not None and str(current) != revisions[repo]:
                    raise RuntimeError(f"the pipeline YAML pins {repo} at {current}, not at {revisions[repo]}")
                node["revision"] = revisions[repo]
                pinned.append(str(repo))
            stack.extend(node.values())
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            stack.extend(node)
    unpinned = sorted(set(revisions) - set(pinned))
    if unpinned:
        raise RuntimeError(f"--hf-revision names {unpinned}, but no `*.from_pretrained` node of the pipeline YAML "
                           "loads them, so they were not pinned; refusing to load rather than read an unpinned `main`. "
                           "Check that the YAML is SAM 3D's checkpoints/pipeline.yaml at the pinned revision")
    return pinned


def load_sam3d_pipeline(config_path: str, device: str, hf_revisions: Mapping[str, str] | None = None) -> Any:
    """`notebook/inference.py`'s `Inference(config, compile=False)` without its notebook-only imports.

    Loads the checkpoint's pipeline YAML with pytorch3d rendering, no torch.compile, and the YAML's directory as the
    workspace, pins its Hugging Face loaders to `hf_revisions` (`pin_hf_revisions`), then instantiates it once
    `check_hydra_targets` accepts it.
    """
    os.environ["LIDRA_SKIP_INIT"] = "true"
    os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", sys.prefix)
    import sam3d_objects  # noqa: F401  # the package import registers its modules (notebook/inference.py)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = os.path.dirname(config_path)
    config.device = device
    if hf_revisions:
        logger.info("pinned Hugging Face loaders: %s", pin_hf_revisions(config, hf_revisions))
    check_hydra_targets(config)
    return instantiate(config)


def apply_gpu_cap(device: str) -> float | None:
    """`fastdo.bench.gpu_budget.apply_gpu_budget`'s rule in this process: `FASTDO_GPU_MEM_GB` GiB, default 12, <= 0
    disables. Returns the applied cap in GiB (at most the device's memory), or None when uncapped or not on CUDA."""
    if not str(device).startswith("cuda"):
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    gb = float(os.environ.get(GPU_MEM_ENV, DEFAULT_GPU_MEM_GB))
    if gb <= 0:
        return None
    index = torch.device(device).index or 0
    total = float(torch.cuda.get_device_properties(index).total_memory)
    fraction = min(1.0, gb * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    logger.info("CUDA per-process cap %.1f GiB (%.3f of device %d)", gb, fraction, index)
    return fraction * total / 2**30


def require_gpu_cap(cap_gb: float | None, min_gb: float) -> None:
    """Fails before the pipeline load when the applied CUDA cap (`apply_gpu_cap`; None means uncapped or not on CUDA) is
    below `min_gb`, instead of loading for about a minute and running out of memory on the first request.

    Raises:
        RuntimeError: If `cap_gb < min_gb`.
    """
    if cap_gb is not None and cap_gb < min_gb:
        raise RuntimeError(f"the CUDA per-process cap is {cap_gb:.1f} GiB, below the {min_gb:g} GiB SAM 3D Objects "
                           f"needs (18.67 GiB measured peak): set {GPU_MEM_ENV}>={min_gb:g}, or pass --min-gpu-gb 0")


def _reply(stream: IO[str], message: dict[str, Any]) -> None:
    stream.write(json.dumps(message) + "\n")
    stream.flush()


def _peak_memory(device_stats: bool, reset: bool) -> dict[str, float | None]:
    """The CUDA peak allocated and reserved GiB since the last reset (None values without CUDA); `reset` starts anew."""
    if not device_stats:
        return {"peak_gpu_gib": None, "peak_gpu_gib_reserved": None}
    import torch

    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {"peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_gpu_gib_reserved": torch.cuda.max_memory_reserved() / 2**30}


def _save_npz(path: str, **arrays: np.ndarray) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
    os.replace(tmp, path)


def _graph_replies(graph_stats: Mapping[str, Any], options: ShapeOptions) -> dict[str, dict[str, Any]]:
    """`cuda_graph_ss` (unprefixed counters) and `cuda_graph_slat`, each only when enabled."""
    replies: dict[str, dict[str, Any]] = {}
    if options.cuda_graph_ss:
        replies["cuda_graph_ss"] = {name: value for name, value in graph_stats.items()
                                    if not name.startswith(SLAT_GRAPH_STATS_PREFIX)}
    if options.cuda_graph_slat:
        replies["cuda_graph_slat"] = {name.removeprefix(SLAT_GRAPH_STATS_PREFIX): value
                                      for name, value in graph_stats.items()
                                      if name.startswith(SLAT_GRAPH_STATS_PREFIX)}
    return replies


def serve(pipeline: Any, requests: IO[str], replies: IO[str], sync: Callable[[], None] | None = None,
          kernels: Callable[[], dict[str, str]] | None = None) -> int:
    """Answers `reconstruct` requests from `requests` until `shutdown` or end of input; returns the exit code 0.
    Peak memory is reported only when `sync` is set, i.e. the worker runs on an available CUDA device; digest replies
    carry `kernels()` ({} when None).

    The peak counters are reset BETWEEN the passes of a flagged request, so `peak_gpu_gib*` is the gated pass's own
    peak and the fallback's is reported beside it as `fallback_peak_gpu_gib*` (the memory twin of the
    `phase_seconds` separation): the memory gate reads the reserved peak as a non-inferiority comparison, and
    one window spanning both passes would report the fallback's higher peak under the gated arm's name.
    """
    on_cuda = sync is not None
    for line in requests:
        if not line.strip():
            continue
        request_id, digests = None, None  # type: ignore[var-annotated]
        qc_stats: dict[str, Any] | None = None
        fallback_phases: dict[str, float] | None = None
        fallback_peaks: dict[str, float | None] | None = None
        try:
            request = json.loads(line)
            if request.get("cmd") == "shutdown":
                return 0
            request_id = request.get("id")
            if request.get("cmd") != "reconstruct":
                raise ValueError(f"unknown command {request.get('cmd')!r}")
            with np.load(request["input"], allow_pickle=False) as data:
                image, mask = np.asarray(data["image"]), np.asarray(data["mask"], dtype=bool)
            params = request["params"]
            options = resolve_shape_options(params)
            digests = {} if bool(params.get("digest", False)) else None
            stats: dict[str, Any] | None = {} if options.postprocess_impl == "fast" else None
            graph_stats: dict[str, Any] | None = {} if options.cuda_graph_ss or options.cuda_graph_slat else None
            _peak_memory(on_cuda, reset=True)
            vertices, faces, phases = reconstruct_geometry(pipeline, image, mask,
                                                          **_call_kwargs(params, options, sync, digests, stats,
                                                                         graph_stats))
            peaks = _peak_memory(on_cuda, reset=False)         # the GATED pass's peak, read before any fallback
            if stats is not None:
                qc_stats = dict(stats)
                outcome = wall_qc.QC_PASS
                if options.wall_qc == "fallback" and wall_qc.flagged(stats, options.thresholds):
                    # One re-run on the 25+25 Euler path, whose output is used whether or not it flags again.
                    # Its phases are recorded apart from the gated pass and never added to it.
                    fallback = wall_qc.fallback_params(params)
                    fallback_stats: dict[str, Any] = {}
                    # Both calls fill the same digests dict, so keep the discarded primary pass's digests
                    # before the re-run overwrites them; the plain names then describe the mesh that was stored.
                    primary_digests = {} if digests is None else {f"primary_{name}": value
                                                                  for name, value in digests.items()}
                    # A new peak window for the fallback, so the gated peak above stays the gated
                    # pass's own and the fallback's is reported beside it, never instead of it.
                    _peak_memory(on_cuda, reset=True)
                    vertices, faces, fallback_phases = reconstruct_geometry(
                        pipeline, image, mask,
                        **_call_kwargs(fallback, resolve_shape_options(fallback), sync, digests, fallback_stats,
                                       graph_stats))
                    fallback_peaks = {f"fallback_{name}": value
                                      for name, value in _peak_memory(on_cuda, reset=False).items()}
                    if digests is not None:
                        digests.update(primary_digests)
                    outcome = (wall_qc.QC_FALLBACK_FLAGGED if wall_qc.flagged(fallback_stats, options.thresholds)
                               else wall_qc.QC_FALLBACK)
                    qc_stats["fallback"] = dict(fallback_stats)
                qc_stats["qc"] = outcome
            _save_npz(request["output"], vertices=vertices, faces=faces)
        except Exception as exc:  # noqa: BLE001 - reported to the client, which raises; the worker keeps serving
            traceback.print_exc(file=sys.stderr)
            _reply(replies, {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                             **_peak_memory(on_cuda, reset=False)})
            continue
        logger.info("request %s: %d vertices, %d faces, phases %s", request_id, len(vertices), len(faces),
                    {name: round(seconds, 3) for name, seconds in phases.items()})
        reply: dict[str, Any] = {"id": request_id, "ok": True, "phase_seconds": phases, **peaks}
        if qc_stats is not None:
            reply["postprocess_stats"] = qc_stats
        if fallback_phases is not None:
            reply["fallback_phase_seconds"] = fallback_phases
        if fallback_peaks is not None:
            reply.update(fallback_peaks)
        if graph_stats is not None:
            reply.update(_graph_replies(graph_stats, options))
        if digests is not None:
            digests["params"] = request_params_digest(params)
            reply.update(digests=digests, kernels=kernels() if kernels is not None else {})
        _reply(replies, reply)
    return 0


def _import_factory(spec: str) -> Callable[[str, str], Any]:
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"--factory must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module), name)  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent SAM 3D Objects geometry-only worker (JSON lines).")
    parser.add_argument("--config", required=True, help="SAM 3D Objects checkpoints/pipeline.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-gpu-gb", type=float, default=SAM3D_MIN_GPU_GB,
                        help="refuse to start when the CUDA cap is below this many GiB (0 disables the check)")
    parser.add_argument("--factory", default=None,
                        help="module:function(config_path, device) building the pipeline (tests); default: SAM 3D")
    parser.add_argument("--spconv-kernels", choices=("tuner", "pinned"), default="tuner",
                        help="pinned: deterministic spconv kernel rule for the process (sam3d_spconv_pin)")
    parser.add_argument("--hf-revision", action="append", default=[], metavar="REPO=COMMIT",
                        help="pin the pipeline YAML's from_pretrained loader of REPO to COMMIT (repeatable)")
    args = parser.parse_args(argv)

    sys.stdout.flush()
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())          # prints from the pipeline land on stderr
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[sam3d_worker] %(asctime)s %(message)s")
    try:
        revisions = parse_hf_revisions(args.hf_revision)
        factory = (_import_factory(args.factory) if args.factory else
                   lambda config, device: load_sam3d_pipeline(config, device, hf_revisions=revisions))
        require_gpu_cap(apply_gpu_cap(args.device), args.min_gpu_gb)
        if args.spconv_kernels == "pinned":           # before any conv runs; never restored in this process
            sam3d_spconv_pin.pin_spconv_kernels()
        start = time.perf_counter()
        pipeline = factory(args.config, args.device)
        load_seconds = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - reported to the client before exiting
        traceback.print_exc(file=sys.stderr)
        _reply(replies, {"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    logger.info("pipeline loaded in %.1f s", load_seconds)
    _reply(replies, {"ready": True, "load_seconds": load_seconds, "spconv_kernels": args.spconv_kernels})
    kernels = sam3d_spconv_pin.pinned_kernels if args.spconv_kernels == "pinned" else None
    return serve(pipeline, sys.stdin, replies, sync=cuda_sync(args.device), kernels=kernels)


if __name__ == "__main__":
    sys.exit(main())
