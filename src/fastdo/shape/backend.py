"""Shape backend interface (D4): `ShapeResult`, `ShapeBackend`, `ShapeConfig`, and the backend registry.

TRELLIS (`trellis_backend.TrellisBackend`) is the default: its weights are open. SAM 3D Objects
(`sam3d_backend.Sam3dBackend`) is gated and exists for the parity comparison with DO AS I DO.

Geometry-only profile. The baseline's SAM 3D call costs 46-48 s per object on the
whisking clip. Texture baking takes 21-31 s of that and layout post-optimization about 3 s; no stage here uses either
(the M1 solver does the layout's job). Every backend therefore:
- decodes the structured latent to a mesh only (no Gaussian or radiance-field decode);
- never bakes a texture or writes a GLB (`ShapeResult.texture_path` is always None);
- never runs layout post-optimization;
- keeps the mesh post-process (simplification and hole filling) that the baseline applies to its geometry, toggled
  by `ShapeConfig.mesh_postprocess`;
- lives inside a persistent worker that loads the pipeline once and reuses it across clips.

The flow step counts are config. The default stays at the baseline's 25 sparse-structure and 25 structured-latent
steps, with Euler updates. Fewer steps, or SAM 3D's second-order midpoint update, are experiments a GPU run
accepts or rejects; see the amendments for the acceptance rule.

Backends register with `register_shape_backend`. `fastdo.shape` imports both backend modules, so `SHAPE_BACKENDS`
is complete whenever anything in the package has been imported.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np

from fastdo.shape.fast_postprocess import (
    DEFAULT_MIN_COMPONENT_AREA_FRAC,
    DEFAULT_MIN_RAW_FACES,
    POSTPROCESS_IMPLS,
    check_min_component_area_frac,
)
from fastdo.shape.wall_qc import (
    DEFAULT_MAX_REMOVED,
    DEFAULT_MIN_AREA_RATIO,
    WALL_QC_MODES,
    WallQcThresholds,
)

__all__ = [
    "GEOMETRY_PHASES",
    "SAMPLER_OPTION_DEFAULTS",
    "SHAPE_BACKENDS",
    "SHAPE_OPTION_DEFAULTS",
    "SOLVERS",
    "SPARSE_STRUCTURE_SOLVERS",
    "STRUCTURED_LATENT_SOLVERS",
    "ShapeBackend",
    "ShapeConfig",
    "ShapeResult",
    "make_shape_backend",
    "masked_crop_rgba",
    "non_default_sampler_options",
    "non_default_shape_options",
    "register_shape_backend",
    "shape_options",
]

# Per-object phases every backend times with a warm model (the amendments' report rows, plus preprocessing).
GEOMETRY_PHASES: tuple[str, ...] = ("preprocess", "sparse_structure_flow", "structured_latent_flow", "decode",
                                    "postprocess")
# Flow-ODE update rules. Only SAM 3D's samplers take "midpoint" (second order, two model evaluations per step);
# TRELLIS's samplers are Euler-only.
SOLVERS: tuple[str, ...] = ("euler", "midpoint")
# The structured-latent flow's own update, overriding `solver` for that flow only. "ab2r" is the
# right-limited Adams-Bashforth 2 (`sam3d_multistep.AdamsBashforth2(regime, right_limit_eps=1e-6)`).
STRUCTURED_LATENT_SOLVERS: tuple[str, ...] = ("ab2r",)
# The same substitution on the sparse-structure flow: "ab2r" is the right-limited AB2 with the
# regime of SAM 3D's guidance switch on the SS generator (`cfg_regime(..., pipeline.ss_cfg_strength)`).
SPARSE_STRUCTURE_SOLVERS: tuple[str, ...] = ("ab2r",)


@dataclass(frozen=True)
class ShapeResult:
    """A reconstructed object mesh in the backend's own object frame (y-up, arbitrary scale)."""

    vertices: np.ndarray                   # (V,3) float64
    faces: np.ndarray                      # (F,3) int64
    texture_path: str | None = None        # always None under the geometry-only profile
    phase_seconds: dict[str, float] = field(default_factory=dict)   # GEOMETRY_PHASES (and extras) -> seconds
    # A wall-QC fallback's own phases, never merged into `phase_seconds`, so `phase_seconds` always sums to the
    # GATED pass's worker seconds - what the timing gate reads, and the only timing an arm
    # is charged; the fallback is charged once, through t_eff. None when no fallback ran.
    fallback_phase_seconds: dict[str, float] | None = None


class ShapeBackend(Protocol):
    """Image-to-3D on one keyframe: `image` (H,W,3) uint8 RGB and `mask` (H,W) bool select the object."""

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult: ...


@dataclass(frozen=True)
class ShapeConfig:
    """Shape-stage settings shared by every backend.

    Attributes:
        backend: Key of `SHAPE_BACKENDS`.
        sparse_structure_steps: Sparse-structure flow steps (baseline: 25).
        structured_latent_steps: Structured-latent flow steps (baseline: 25).
        solver: Flow-ODE update, an entry of `SOLVERS`.
        seed: `torch.manual_seed` of the flow noise. The default 42 is the baseline's `generate_mesh_sam3d.py --seed`,
            which seeds object `i` of a frame with `42 + i`; a mesh-for-mesh parity run on object `i` needs `42 + i`.
        mesh_postprocess: Apply the baseline's mesh simplification (keep 5% of faces) and hole filling.
        device: Torch device of the pipeline.
        trellis_model: Hugging Face id (or local directory) of the TRELLIS image-to-3D pipeline, read from the local
            cache only.
        sam3d_repo: Hugging Face id of the gated SAM 3D Objects checkpoints, read from the local cache only.
        sam3d_python: Interpreter of the SAM 3D environment that runs the worker process; None reads
            `FASTDO_SAM3D_PYTHON`. The `sam3d` backend needs `FASTDO_GPU_MEM_GB >= 19` on CUDA (18.67 GiB measured
            peak); below that the worker refuses to start, before its 69 s pipeline load.
        sam3d_pythonpath: Directories prepended, in order, to the SAM 3D worker's PYTHONPATH ahead of any inherited
            entries (the worker process only, never the engine's interpreter). None reads `FASTDO_SAM3D_PYTHONPATH`
            (os.pathsep-separated; unset or empty means no entries). On an H100 SAM 3D selects flash_attn, which
            (with diff_gaussian_rasterization) may be installed only in machine-local directories.
        deterministic: Run SAM 3D's structured-latent flow and mesh decode with the order-fixed mean reductions of
            `sam3d_determinism`, so identical requests give byte-identical meshes. None (default) resolves to True
            for `sam3d`, False otherwise: the determinism gates passed on DexYCB 135545/135841 (bit identity in and
            across workers, reserved 19.07 GiB <= 21, +0.15/+0.28 s median flow + decode), and an interleaved
            re-timing found no unpatched-phase slowdown. TRELLIS is not patched and refuses True.
        spconv_kernel_pin: Start the SAM 3D worker with `sam3d_spconv_pin`'s deterministic spconv kernel rule instead
            of spconv's timing tuner, whose choice depends on the first request and timing noise
            (stock gave 3 artifact states per clip over 6 fresh processes; pinned, 12 fresh
            processes x 4 request orders gave 1 state per clip, first request -2.84 s, reserved 20.63 GiB). None
            (default) resolves to True for `sam3d`, False otherwise; True with another backend is refused.
        mesh_postprocess_impl: Which mesh post-process runs: `"official"` (SAM 3D's own
            `postprocess_mesh`, today's path) or `"fast"` (`fast_postprocess`, the instrumented replacement that
            equals it bit for bit below the decimator threshold). `"fast"` is measured on `sam3d` only.
        fast_decimator_min_raw_faces: The fast path decimates with VTK below this many raw FlexiCubes faces and with
            `fast_simplification` at or above it. A value above any mesh size is the instrumented
            VTK-only mode.
        min_component_area_frac: When > 0, the fast path's last mesh step drops every connected
            component (shared-edge connectivity) below this fraction of the total surface area, keeping the
            largest. 0.0 (default) is today's path, byte for byte. Needs `mesh_postprocess_impl="fast"`.
        wall_qc: Per-output double-wall QC: `"off"`, or `"fallback"` to re-run a flagged request once at
            25+25 Euler steps on the fast path. It needs `mesh_postprocess_impl="fast"`, which is what reports the
            mincut and area statistics it thresholds.
        wall_qc_max_removed: The QC flags when the mincut removed a smaller fraction of the decimated faces than this.
        wall_qc_min_area_ratio: ... and the mesh area is at least this multiple of its convex hull's area.
        sam3d_expandable_segments: Start the SAM 3D worker with
            `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` instead of the default allocator, and record that in
            the worker's environment identity. `sam3d` only.
        sam3d_cuda_graph_ss: Replay the SAM 3D sparse-structure flow's dense backbone from CUDA graphs
            (`sam3d_cuda_graph`), captured once per worker process. Output-preserving by construction and gated on
            bit-identical coords and meshes; False (default) is today's eager path. `sam3d` only.
        sam3d_cuda_graph_slat: Replay the SAM 3D structured-latent flow's 24 transformer blocks from CUDA
            graphs (`sam3d_cuda_graph_slat`), captured per request (the token count is per object) and dropped at the
            end of its SLAT phase. Output-preserving by construction and to be gated on bit-identical meshes; False
            (default) is today's eager path. `sam3d` only.
        sparse_structure_shortcut: Sample the sparse-structure flow through the checkpoint's
            trained shortcut path (`pipeline.sample_sparse_structure(..., use_distillation=True)`: step size
            conditioning `d = 1/sparse_structure_steps`, guidance off, as SAM 3D's own switch sets it) at
            `sparse_structure_steps` steps. False (default) is today's 25-step guided flow. `sam3d` and `solver="euler"`
            only.
        structured_latent_solver: None (default) integrates the structured-latent flow
            with `solver`, as today; `"ab2r"` with the right-limited AB2 (`sam3d_multistep.AdamsBashforth2`,
            `right_limit_eps=1e-6`) at `structured_latent_steps` steps. `sam3d` and `solver="euler"` only.
        sparse_structure_solver: None (default) integrates the sparse-structure flow with
            `solver`, as today; `"ab2r"` with the same right-limited AB2, its guidance regime read from the SS generator
            (`ss_cfg_strength`), at `sparse_structure_steps` steps. `sam3d` and `solver="euler"` only, and not together
            with `sparse_structure_shortcut` (the shortcut path is a different flow). Opt-in until its shape gate,
            the pose gate and an Ego4D gauge pass.

    Raises:
        ValueError: On a step count that is not an int >= 1, a seed that is not an int, an unknown solver, a
            `sam3d_pythonpath` that is not a tuple (or list) of non-empty strings without os.pathsep, a
            `deterministic` or `spconv_kernel_pin` that is not a bool, `spconv_kernel_pin=True` off `sam3d`, or any
            invalid fast-path option or combination of them (`_check_fast_path_options`).
    """

    backend: str = "trellis"
    sparse_structure_steps: int = 25
    structured_latent_steps: int = 25
    solver: str = "euler"
    seed: int = 42
    mesh_postprocess: bool = True
    device: str = "cuda"
    trellis_model: str = "microsoft/TRELLIS-image-large"
    sam3d_repo: str = "facebook/sam-3d-objects"
    sam3d_python: str | None = None
    sam3d_pythonpath: tuple[str, ...] | None = None
    deterministic: bool | None = None
    spconv_kernel_pin: bool | None = None
    # Fast-path options: every one is off, or at today's value, by default.
    mesh_postprocess_impl: str = "official"
    fast_decimator_min_raw_faces: int = DEFAULT_MIN_RAW_FACES
    min_component_area_frac: float = DEFAULT_MIN_COMPONENT_AREA_FRAC
    wall_qc: str = "off"
    wall_qc_max_removed: float = DEFAULT_MAX_REMOVED
    wall_qc_min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    sam3d_expandable_segments: bool = False
    sam3d_cuda_graph_ss: bool = False
    sam3d_cuda_graph_slat: bool = False
    # Sampler options. Off by default; recorded and keyed only when set.
    sparse_structure_shortcut: bool = False
    structured_latent_solver: str | None = None
    sparse_structure_solver: str | None = None

    def _check_fast_path_options(self) -> None:
        """Validates the fast-path options and refuses the combinations that cannot produce their statistics.

        **Every message below is duplicated, string for string, by `sam3d_worker.resolve_shape_options`** - the other
        boundary a request enters through, since the EVAL drives the worker directly and a value only
        this check would catch never reaches it otherwise.
        `tests/shape/test_shape_fast_options.py::test_the_worker_boundary_refuses_exactly_what_shapeconfig_refuses`
        asserts that equality value by value. It is kept exact on purpose: editing a message here without editing the
        worker's SHOULD fail that test, which is the coupling working, not a maintenance burden.
        """
        if self.mesh_postprocess_impl not in POSTPROCESS_IMPLS:
            raise ValueError(f"mesh_postprocess_impl must be one of {POSTPROCESS_IMPLS}, got "
                             f"{self.mesh_postprocess_impl!r}")
        if self.mesh_postprocess_impl == "fast" and self.backend != "sam3d":
            raise ValueError(f"mesh_postprocess_impl='fast' is the SAM 3D post-process, so it is allowed only for "
                             f"sam3d, got backend {self.backend!r}")
        if self.mesh_postprocess_impl == "fast" and not self.mesh_postprocess:
            raise ValueError("mesh_postprocess_impl='fast' needs mesh_postprocess=True: with the post-process off "
                             "there is no decimation, mincut or hole filling for the fast path to do")
        if self.wall_qc not in WALL_QC_MODES:
            raise ValueError(f"wall_qc must be one of {WALL_QC_MODES}, got {self.wall_qc!r}")
        if self.wall_qc != "off" and self.mesh_postprocess_impl != "fast":
            raise ValueError(f"wall_qc={self.wall_qc!r} needs mesh_postprocess_impl='fast': only the instrumented "
                             "fast post-process reports the mincut-removed fraction and the area ratio it thresholds")
        WallQcThresholds(self.wall_qc_max_removed, self.wall_qc_min_area_ratio)   # raises, naming its own field
        if not _is_int(self.fast_decimator_min_raw_faces) or self.fast_decimator_min_raw_faces < 0:
            raise ValueError(f"fast_decimator_min_raw_faces must be an int >= 0, got "
                             f"{self.fast_decimator_min_raw_faces!r}")
        if check_min_component_area_frac(self.min_component_area_frac) > 0 and self.mesh_postprocess_impl != "fast":
            raise ValueError("min_component_area_frac > 0 needs mesh_postprocess_impl='fast': the fragment filter is "
                             "a step of the fast post-process")
        if not isinstance(self.sam3d_expandable_segments, bool) or (self.sam3d_expandable_segments
                                                                    and self.backend != "sam3d"):
            raise ValueError(f"sam3d_expandable_segments must be a bool, and True only for sam3d, got "
                             f"{self.sam3d_expandable_segments!r} for backend {self.backend!r}")
        if not isinstance(self.sam3d_cuda_graph_ss, bool):
            raise ValueError(f"sam3d_cuda_graph_ss must be a bool, got {self.sam3d_cuda_graph_ss!r}")  # noqa: TRY004
        if self.sam3d_cuda_graph_ss and self.backend != "sam3d":
            raise ValueError(f"sam3d_cuda_graph_ss=True is a SAM 3D worker option, so it is allowed only for sam3d, "
                             f"got backend {self.backend!r}")
        if not isinstance(self.sam3d_cuda_graph_slat, bool):
            raise ValueError(f"sam3d_cuda_graph_slat must be a bool, got {self.sam3d_cuda_graph_slat!r}")  # noqa: TRY004
        if self.sam3d_cuda_graph_slat and self.backend != "sam3d":
            raise ValueError(f"sam3d_cuda_graph_slat=True is a SAM 3D worker option, so it is allowed only for "
                             f"sam3d, got backend {self.backend!r}")

    def _check_sampler_options(self) -> None:
        """Validates the sampler options. As for the fast-path options,
        `sam3d_worker.resolve_shape_options` repeats every message string for string
        (`tests/shape/test_sam3d_sampler_options.py` asserts the equality)."""
        if not isinstance(self.sparse_structure_shortcut, bool):
            raise ValueError(f"sparse_structure_shortcut must be a bool, got "  # noqa: TRY004  # spec: ValueError
                             f"{self.sparse_structure_shortcut!r}")
        if self.structured_latent_solver is not None and self.structured_latent_solver not in STRUCTURED_LATENT_SOLVERS:
            raise ValueError(f"structured_latent_solver must be None or one of {STRUCTURED_LATENT_SOLVERS}, got "
                             f"{self.structured_latent_solver!r}")
        if (self.sparse_structure_shortcut or self.structured_latent_solver is not None) and self.solver != "euler":
            raise ValueError(f"sparse_structure_shortcut and structured_latent_solver are measured with solver='euler' "
                             f"only, got solver {self.solver!r}")
        # The sparse-structure twin of `structured_latent_solver`, the same three checks plus exclusion.
        if self.sparse_structure_solver is not None and self.sparse_structure_solver not in SPARSE_STRUCTURE_SOLVERS:
            raise ValueError(f"sparse_structure_solver must be None or one of {SPARSE_STRUCTURE_SOLVERS}, got "
                             f"{self.sparse_structure_solver!r}")
        if self.sparse_structure_solver is not None and self.solver != "euler":
            raise ValueError(f"sparse_structure_solver is measured with solver='euler' only, got solver "
                             f"{self.solver!r}")
        if self.sparse_structure_solver is not None and self.sparse_structure_shortcut:
            raise ValueError(f"sparse_structure_solver={self.sparse_structure_solver!r} and "
                             "sparse_structure_shortcut=True are mutually exclusive: the shortcut path samples a "
                             "different (distilled, unguided) flow")
        if non_default_sampler_options(self) and self.backend != "sam3d":
            raise ValueError(f"sparse_structure_shortcut, structured_latent_solver and sparse_structure_solver are "
                             f"SAM 3D sampler options, so they are allowed only for sam3d, got backend "
                             f"{self.backend!r}")

    def __post_init__(self) -> None:
        self._check_fast_path_options()
        self._check_sampler_options()
        if self.spconv_kernel_pin is None:            # the backend default (docstring): measured on sam3d only
            object.__setattr__(self, "spconv_kernel_pin", self.backend == "sam3d")
        if not isinstance(self.spconv_kernel_pin, bool) or (self.spconv_kernel_pin and self.backend != "sam3d"):
            raise ValueError(f"spconv_kernel_pin must be a bool, and True only for sam3d, got "
                             f"{self.spconv_kernel_pin!r} for backend {self.backend!r}")
        if self.deterministic is None:                # the backend default (docstring): the gates measured sam3d only
            object.__setattr__(self, "deterministic", self.backend == "sam3d")
        if not isinstance(self.deterministic, bool):
            raise ValueError(f"deterministic must be a bool, got {self.deterministic!r}")  # noqa: TRY004  # spec: ValueError
        for name in ("sparse_structure_steps", "structured_latent_steps"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        if not _is_int(self.seed):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        entries = self.sam3d_pythonpath
        if entries is not None:                   # a Hydra list becomes a tuple; a str is refused, never split
            if not isinstance(entries, tuple | list) or not all(
                    isinstance(entry, str) and entry and os.pathsep not in entry for entry in entries):
                raise ValueError(f"sam3d_pythonpath must be None or a tuple of non-empty directory strings (one "
                                 f"per entry, no {os.pathsep!r}), got {entries!r}")
            object.__setattr__(self, "sam3d_pythonpath", tuple(entries))


def _is_int(value: object) -> bool:
    """A plain int, not a bool (the samplers and the worker's JSON protocol take Python ints)."""
    return isinstance(value, int) and not isinstance(value, bool)


# The fast-path options and the values that mean "today's path". `non_default_shape_options` returning `{}` is the
# definition of a default request: the shape cache key leaves these fields out and shape.json records none of them, so
# a default run stays byte-identical to one made before these options existed (the same rule as for `deterministic`).
SHAPE_OPTION_DEFAULTS: dict[str, Any] = {
    "mesh_postprocess_impl": "official",
    "fast_decimator_min_raw_faces": DEFAULT_MIN_RAW_FACES,
    "min_component_area_frac": DEFAULT_MIN_COMPONENT_AREA_FRAC,
    "wall_qc": "off",
    "wall_qc_max_removed": DEFAULT_MAX_REMOVED,
    "wall_qc_min_area_ratio": DEFAULT_MIN_AREA_RATIO,
    "sam3d_expandable_segments": False,
    "sam3d_cuda_graph_ss": False,
    "sam3d_cuda_graph_slat": False,
}


# The sampler options and their "today's path" values. Left out of the cache key at their defaults, like the
# fast-path options, but recorded in shape.json only when set (never all together, as `shape_options` are):
# production sets fast-path options, so its shape.json records every `SHAPE_OPTION_DEFAULTS` key, and a new key there
# would change it.
SAMPLER_OPTION_DEFAULTS: dict[str, Any] = {
    "sparse_structure_shortcut": False,
    "structured_latent_solver": None,
    "sparse_structure_solver": None,
}


def non_default_sampler_options(cfg: ShapeConfig) -> dict[str, Any]:
    """The sampler options of `cfg` that are off their default; `{}` for today's sampler."""
    return {name: getattr(cfg, name) for name, default in SAMPLER_OPTION_DEFAULTS.items()
            if getattr(cfg, name) != default}


def shape_options(cfg: ShapeConfig) -> dict[str, Any]:
    """Every fast-path option of `cfg`, at its resolved value."""
    return {name: getattr(cfg, name) for name in SHAPE_OPTION_DEFAULTS}


def non_default_shape_options(cfg: ShapeConfig) -> dict[str, Any]:
    """The fast-path options of `cfg` that are off their default; `{}` for a default request."""
    return {name: value for name, value in shape_options(cfg).items() if value != SHAPE_OPTION_DEFAULTS[name]}


SHAPE_BACKENDS: dict[str, type[ShapeBackend]] = {}

_B = TypeVar("_B")


def register_shape_backend(name: str) -> Callable[[type[_B]], type[_B]]:
    """Class decorator adding a backend to `SHAPE_BACKENDS` under `name`."""

    def decorator(cls: type[_B]) -> type[_B]:
        SHAPE_BACKENDS[name] = cls  # type: ignore[assignment]  # structural: the class implements ShapeBackend
        return cls

    return decorator


def make_shape_backend(cfg: ShapeConfig) -> ShapeBackend:
    """The configured backend, constructed without loading anything (backends load on first use).

    Raises:
        ValueError: If `cfg.backend` is not registered.
    """
    try:
        backend_cls = SHAPE_BACKENDS[cfg.backend]
    except KeyError:
        raise ValueError(f"unknown shape backend {cfg.backend!r}; registered: {sorted(SHAPE_BACKENDS)}") from None
    return backend_cls(cfg)  # type: ignore[call-arg]  # every registered backend takes the config first


def masked_crop_rgba(image: np.ndarray, mask: np.ndarray, pad: float = 1.2) -> np.ndarray:
    """A square RGBA crop around `mask` with alpha = mask, `pad` times the mask's longer bounding-box side.

    Parts of the square outside the frame are transparent black.

    Raises:
        ValueError: On mismatched shapes or an empty mask.
    """
    image = np.asarray(image)
    mask = np.asarray(mask).astype(bool)
    if image.ndim != 3 or image.shape[2] < 3 or mask.shape != image.shape[:2]:
        raise ValueError(f"image must be (H,W,3) and mask (H,W) of the same size, got {image.shape} and {mask.shape}")
    if not mask.any():
        raise ValueError("mask is empty: the shape backend needs at least one object pixel")
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    side = max(1, int(np.ceil(max(r1 - r0, c1 - c0) * pad)))
    top, left = (r0 + r1 - side) // 2, (c0 + c1 - side) // 2
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    h, w = mask.shape
    sr0, sr1, sc0, sc1 = max(top, 0), min(top + side, h), max(left, 0), min(left + side, w)
    rgba[sr0 - top:sr1 - top, sc0 - left:sc1 - left, :3] = image[sr0:sr1, sc0:sc1, :3]
    rgba[sr0 - top:sr1 - top, sc0 - left:sc1 - left, 3] = mask[sr0:sr1, sc0:sc1].astype(np.uint8) * 255
    return rgba
