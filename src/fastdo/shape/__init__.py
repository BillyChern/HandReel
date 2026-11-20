"""Object shape: SAM 2 masks, and the shape backends (TRELLIS default, SAM 3D optional) with mesh preparation."""

from .backend import (
    SHAPE_BACKENDS,
    ShapeBackend,
    ShapeConfig,
    ShapeResult,
    make_shape_backend,
    register_shape_backend,
)
from .masks_sam2 import propagate_masks_sam2
from .mesh_utils import canonical_samples, check_watertight, normalize_mesh
from .sam3d_backend import Sam3dBackend
from .trellis_backend import TrellisBackend

__all__: list[str] = [
    "SHAPE_BACKENDS",
    "Sam3dBackend",
    "ShapeBackend",
    "ShapeConfig",
    "ShapeResult",
    "TrellisBackend",
    "canonical_samples",
    "check_watertight",
    "make_shape_backend",
    "normalize_mesh",
    "propagate_masks_sam2",
    "register_shape_backend",
]
