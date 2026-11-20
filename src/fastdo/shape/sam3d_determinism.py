"""Order-fixed replacements for SAM 3D's nondeterministic CUDA mean reductions.

SAM 3D's structured-latent flow (`SparseDownsample.forward`, patch point P1) and its mesh decoder (`cubes_to_verts`, P2)
average with CUDA `torch.scatter_reduce(reduce="mean")`, whose accumulation order changes from call to call: 20/20
distinct results on real coordinates. `segment_mean` computes the same mean as a fixed left
fold in ascending original index (stable integer argsort, unweighted `bincount`, integer `cumsum`, at most K elementwise
`where(acc + src)` steps, K <= 8 at both sites). Every op is deterministic on CUDA; on CPU the result is bitwise equal to
`torch.scatter_reduce` in float32 and float64. The FlexiCubes `index_add_` calls (P3) run under `scoped_deterministic`.

`deterministic_sam3d` installs the three patches on the loaded SAM 3D modules for one block. This module imports torch
and the standard library only: the SAM 3D worker imports it from its own script directory, never through `fastdo`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

__all__ = [
    "FLEXICUBES_MODULE",
    "PATCH_SENTINEL",
    "SPATIAL_MODULE",
    "UTILS_CUBE_MODULE",
    "capture_flexicubes_input",
    "deterministic_cubes_to_verts",
    "deterministic_sam3d",
    "deterministic_scatter_reduce",
    "scoped_deterministic",
    "segment_mean",
]

_TDFY = "sam3d_objects.model.backbone.tdfy_dit"
SPATIAL_MODULE = f"{_TDFY}.modules.sparse.spatial"                            # SparseDownsample (P1)
UTILS_CUBE_MODULE = f"{_TDFY}.representations.mesh.utils_cube"                # cubes_to_verts (P2)
FLEXICUBES_MODULE = f"{_TDFY}.representations.mesh.flexicubes.flexicubes"    # FlexiCubes (P3)
PATCH_SENTINEL = "__fastdo_det__"       # attribute set on every installed wrapper


def segment_mean(input: torch.Tensor, index: torch.Tensor, src: torch.Tensor, include_self: bool) -> torch.Tensor:
    """`torch.scatter_reduce(input, 0, index[:, None].expand_as(src), src, "mean", include_self=include_self)` with a
    fixed accumulation order.

    Args:
        input: `(M, *F)` values; a segment with no index keeps its input value.
        index: `(N,)` int64 segment of each `src` row, in `[0, M)`.
        src: `(N, *F)` rows, same dtype and device as `input`.
        include_self: Count `input` as one more member of each non-empty segment, as `scatter_reduce` does.

    Returns:
        `(M, *F)`: per segment `((x + s_i1) + s_i2) + ...` over its rows in ascending original index (x = input if
        `include_self`, else +0), divided by the member count in the input dtype.
    """
    m, n = input.shape[0], index.shape[0]
    order = torch.argsort(index, stable=True)
    counts = torch.bincount(index, minlength=m)
    starts = torch.cumsum(counts, 0) - counts
    k = int(counts.max().item()) if n else 0
    src_sorted = src[order]
    view = (m,) + (1,) * (input.dim() - 1)
    acc = input.clone() if include_self else torch.zeros_like(input)
    for rank in range(k):                                     # Check L2: no (M, K, *F) buffer, same fold and bytes
        valid = (counts > rank).view(view)
        position = (starts + rank).clamp(max=n - 1)
        acc = torch.where(valid, acc + src_sorted[position], acc)
    divisor = (counts + int(include_self)).clamp_min(1).to(input.dtype).view(view)
    return torch.where((counts > 0).view(view), acc / divisor, input)


def deterministic_scatter_reduce(input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor,
                                 reduce: str, *, include_self: bool = True) -> torch.Tensor:
    """`torch.scatter_reduce` for P1's call shape only: `dim` 0, `reduce` "mean", a CUDA floating `input` and an index
    whose columns are one expanded column. Any other call raises instead of falling back.

    Raises:
        NotImplementedError: On any call outside that shape.
    """
    if dim != 0:
        raise NotImplementedError(f"deterministic_scatter_reduce supports dim=0 only, got dim={dim}")
    if reduce != "mean":
        raise NotImplementedError(f"deterministic_scatter_reduce supports reduce='mean' only, got {reduce!r}")
    if index.dim() != 2 or index.dtype != torch.int64 or index.shape != src.shape or input.dim() != 2:
        raise NotImplementedError(f"deterministic_scatter_reduce needs a 2-D int64 index shaped like src and a 2-D "
                                  f"input, got index {tuple(index.shape)} {index.dtype}, src {tuple(src.shape)}, input "
                                  f"{tuple(input.shape)}")
    if not (index.shape[1] == 1 or index.stride(1) == 0):
        raise NotImplementedError("deterministic_scatter_reduce needs a column-constant index (an expand view)")
    if not input.is_floating_point() or input.dtype != src.dtype:
        raise NotImplementedError(f"deterministic_scatter_reduce needs one floating dtype, got {input.dtype} and "
                                  f"{src.dtype}")
    if not input.is_cuda or index.device != input.device or src.device != input.device:
        raise NotImplementedError(f"deterministic_scatter_reduce is for CUDA tensors on one device, got input on "
                                  f"{input.device}, index on {index.device}, src on {src.device}")
    return segment_mean(input, index[:, 0], src, include_self)


def deterministic_cubes_to_verts(num_verts: int, cubes: torch.Tensor, value: torch.Tensor,
                                 reduce: str = "mean") -> torch.Tensor:
    """`utils_cube.cubes_to_verts` (P2): the mean of `value` `(N, 8, M)` over each grid vertex listed in `cubes`
    `(N, 8)`, without materialising the `(8N, M)` expanded index.

    Raises:
        NotImplementedError: If `reduce` is not "mean".
    """
    if reduce != "mean":
        raise NotImplementedError(f"deterministic_cubes_to_verts supports reduce='mean' only, got {reduce!r}")
    zeros = torch.zeros(num_verts, value.shape[2], device=cubes.device, dtype=value.dtype)
    return segment_mean(zeros, cubes.flatten().to(torch.int64), value.flatten(0, 1), include_self=False)


def _mark(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Sets `PATCH_SENTINEL` (not identity: the worker script imports this file as a second, top-level module)."""
    setattr(fn, PATCH_SENTINEL, True)
    return fn


_mark(deterministic_scatter_reduce), _mark(deterministic_cubes_to_verts)


@contextmanager
def scoped_deterministic() -> Iterator[None]:
    """`torch.use_deterministic_algorithms(True, warn_only=True)` for the block; the previous mode (enabled and
    warn-only) is restored on exit, also on an exception."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@contextmanager
def deterministic_sam3d() -> Iterator[None]:
    """Installs the patch points on the loaded SAM 3D modules and restores the originals on exit.

    - P1: `SparseDownsample.forward` runs the original forward with `torch.scatter_reduce` swapped for
      `deterministic_scatter_reduce` for that call only.
    - P2: `utils_cube.cubes_to_verts` becomes `deterministic_cubes_to_verts` (`sparse_cube2verts` resolves it from the
      module globals at call time).
    - P3: `FlexiCubes.__call__` runs under `scoped_deterministic`, for its `index_add_` calls.
    The originals, `torch.scatter_reduce` and the deterministic-algorithms mode are restored by identity, also when
    the block raises. third_party files are never modified.

    Raises:
        RuntimeError: If a patch is already installed (no nesting).
    """
    spatial = importlib.import_module(SPATIAL_MODULE)
    utils_cube = importlib.import_module(UTILS_CUBE_MODULE)
    flexicubes = importlib.import_module(FLEXICUBES_MODULE)
    downsample, extractor = spatial.SparseDownsample, flexicubes.FlexiCubes
    orig_forward, orig_cubes_to_verts = downsample.forward, utils_cube.cubes_to_verts
    orig_call, torch_scatter_reduce = extractor.__call__, torch.scatter_reduce
    if any(getattr(fn, PATCH_SENTINEL, False) for fn in (orig_forward, orig_call, orig_cubes_to_verts,
                                                          torch_scatter_reduce)):
        raise RuntimeError("the deterministic SAM 3D patch is already installed")
    mode = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        saved = torch.scatter_reduce
        torch.scatter_reduce = deterministic_scatter_reduce  # type: ignore[assignment]
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            torch.scatter_reduce = saved

    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        with scoped_deterministic():
            return orig_call(self, *args, **kwargs)

    try:
        downsample.forward = _mark(forward)
        setattr(utils_cube, "cubes_to_verts", deterministic_cubes_to_verts)  # noqa: B010  # a module global
        extractor.__call__ = _mark(call)
        yield
    finally:
        downsample.forward, extractor.__call__ = orig_forward, orig_call
        setattr(utils_cube, "cubes_to_verts", orig_cubes_to_verts)  # noqa: B010
        torch.scatter_reduce = torch_scatter_reduce  # type: ignore[assignment]
        torch.use_deterministic_algorithms(mode[0], warn_only=mode[1])


@contextmanager
def capture_flexicubes_input(sink: list[Any]) -> Iterator[None]:
    """Appends clones of each FlexiCubes call's `voxelgrid_vertices` and `scalar_field`."""
    extractor = importlib.import_module(FLEXICUBES_MODULE).FlexiCubes
    orig_call = extractor.__call__

    def call(self: Any, voxelgrid_vertices: Any, scalar_field: Any, *args: Any, **kwargs: Any) -> Any:
        sink.extend((voxelgrid_vertices.detach().clone(), scalar_field.detach().clone()))
        return orig_call(self, voxelgrid_vertices, scalar_field, *args, **kwargs)

    extractor.__call__ = call
    try:
        yield
    finally:
        extractor.__call__ = orig_call
