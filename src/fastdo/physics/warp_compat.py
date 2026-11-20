"""Warp 1.10 `wp.copy` semantics on newer Warp, for DO AS I DO's code under `--sim-stack mjwarp314`.

DO AS I DO (written against Warp 1.10.1) copies torch-backed float32 arrays into vector-typed mujoco_warp arrays,
e.g. `wp.copy(data_wp.xfrc_applied, wp.from_torch(t))` with `xfrc_applied` of dtype spatial_vectorf and the source
float32 of shape (nworld, nbody, 6) (`retargeting/utils/mjwp.py::_broadcast_state`). Warp 1.10.1's `copy`, for two
CONTIGUOUS arrays, is a raw memcpy of `count * sizeof(src.dtype)` bytes (count defaults to src.size) at byte offsets
`*_offset * sizeof(dtype)`, with byte-size bounds checks and no dtype check. Warp 1.17.0's `copy` is the same code
plus one new check that refuses differing element sizes. `install_copy_compat` wraps `warp.copy` so that exactly that
case (both contiguous, element sizes differ) performs Warp 1.10.1's memcpy, through byte views of both buffers and
the original `copy` (same bytes, offsets, bounds checks and stream). Every other call is passed through unchanged.
Non-contiguous arrays with differing element sizes were refused by 1.10.1 as well and still are.
"""

from __future__ import annotations

from typing import Any

__all__ = ["install_copy_compat", "stats"]

_STATS = {"compat_copies": 0}


def stats() -> dict[str, int]:
    return dict(_STATS)


def install_copy_compat(wp: Any) -> bool:
    """Wrap `wp.copy` (idempotent). Returns True if it installed the wrapper now."""
    if getattr(wp.copy, "_fastdo_compat", False):
        return False
    original = wp.copy
    from warp._src.types import type_size_in_bytes

    def copy(dest: Any, src: Any, dest_offset: int = 0, src_offset: int = 0, count: int = 0,
             stream: Any = None) -> None:
        if (hasattr(dest, "is_contiguous") and hasattr(src, "is_contiguous") and src.is_contiguous
                and dest.is_contiguous):
            ss, ds = type_size_in_bytes(src.dtype), type_size_in_bytes(dest.dtype)
            if ss != ds:
                n = src.size if count <= 0 else count
                if n == 0:
                    return
                sb = wp.array(ptr=src.ptr, dtype=wp.uint8, shape=(src.size * ss,), device=src.device)
                db = wp.array(ptr=dest.ptr, dtype=wp.uint8, shape=(dest.size * ds,), device=dest.device)
                _STATS["compat_copies"] += 1
                original(db, sb, dest_offset=dest_offset * ds, src_offset=src_offset * ss, count=n * ss,
                         stream=stream)
                return
        original(dest, src, dest_offset=dest_offset, src_offset=src_offset, count=count, stream=stream)

    copy._fastdo_compat = True  # type: ignore[attr-defined]
    copy.__wrapped__ = original  # type: ignore[attr-defined]
    wp.copy = copy
    return True
