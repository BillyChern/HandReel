"""Row reads of one array of an uncompressed `.npz`, without materialising the rest of it.

`decode.npz` holds `frames` (N,H,W,3) uint8 -- 0.6-1.5 GB per 100-row window -- and several stages need only a
few rows of it (the prompt detector's candidates, the intrinsics keyframes, the shape keyframe). `np.load(path)
["frames"]` reads the whole member before any row is selected. The engine writes artifacts with uncompressed
`np.savez` (ZIP_STORED), so a member's `.npy` bytes sit verbatim in the file: this module finds them through the
member's LOCAL header (its extra field is not the central directory's -- `np.savez` writes a 20-byte zip64 extra
field locally and none centrally), parses the `.npy` header there, memory-maps only that region read-only and
copies out the requested rows. A compressed member, a Fortran-ordered or object array, an empty array or an
`.npy` format version other than 1.0/2.0 falls back to the full `np.load`; either way the bytes returned are those
of `np.load(path)[name][rows]`.

This file imports only the standard library and numpy: the Hands23 detector runs in its own interpreter, which
cannot import `fastdo`, and loads this file by path (`fastdo/prompt/hands23_detect.py`).
"""

from __future__ import annotations

import math
import struct
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = ["npz_array_shape", "read_npz_rows"]

_LOCAL_HEADER = struct.Struct("<4s22xHH")   # signature, 22 bytes of versions/flags/sizes, name length, extra length
_LOCAL_SIGNATURE = b"PK\x03\x04"


def _read_header(f: object) -> tuple[tuple[int, ...], bool, np.dtype] | None:
    """`(shape, fortran_order, dtype)` of the `.npy` header at `f`'s position; None for a version other than 1.0/2.0."""
    version = np.lib.format.read_magic(f)       # type: ignore[arg-type]
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)   # type: ignore[arg-type,return-value]
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)   # type: ignore[arg-type,return-value]
    return None


def npz_array_shape(path: Path, name: str) -> tuple[int, ...]:
    """The shape of member `name` of the `.npz` at `path`, read from its `.npy` header only (stored or compressed).

    Raises:
        KeyError: If the archive has no member `name`.
        ValueError: If the member's `.npy` format version is not 1.0 or 2.0.
    """
    with zipfile.ZipFile(Path(path)) as archive, archive.open(f"{name}.npy") as f:
        header = _read_header(f)
    if header is None:
        raise ValueError(f"{path}: {name}.npy has an unsupported .npy format version")
    return tuple(int(s) for s in header[0])


def _stored_layout(path: Path, name: str) -> tuple[int, tuple[int, ...], np.dtype] | None:
    """`(file offset of the array data, shape, dtype)` of a ZIP_STORED, C-ordered, non-empty member; else None.

    Raises:
        KeyError: If the archive has no member `name`.
        ValueError: If the local header is not where the central directory says, or the member is shorter than
            its `.npy` header claims (a truncated file, which `np.load` would also refuse).
    """
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{name}.npy")
    if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:        # compressed or encrypted
        return None
    with Path(path).open("rb") as f:
        f.seek(info.header_offset)
        local = f.read(_LOCAL_HEADER.size)
        if len(local) != _LOCAL_HEADER.size or local[:4] != _LOCAL_SIGNATURE:
            raise ValueError(f"{path}: no local zip header for {name}.npy at offset {info.header_offset}")
        _, name_len, extra_len = _LOCAL_HEADER.unpack(local)
        start = info.header_offset + _LOCAL_HEADER.size + name_len + extra_len
        f.seek(start)
        header = _read_header(f)
        offset = f.tell()
    if header is None:
        return None
    shape, fortran_order, dtype = header
    if fortran_order or dtype.hasobject or math.prod(shape) == 0:
        return None
    if offset - start + math.prod(shape) * dtype.itemsize > info.file_size:
        raise ValueError(f"{path}: {name}.npy holds {info.file_size} bytes, fewer than its header's "
                         f"{tuple(shape)} {dtype} array needs")
    return offset, tuple(int(s) for s in shape), dtype


def read_npz_rows(path: Path, name: str, rows: Sequence[int] | np.ndarray) -> np.ndarray:
    """`np.load(path)[name][rows]`, as a new C-contiguous array, reading only the selected rows when it can.

    `rows` is a 1-D sequence of integer row indices along axis 0, in any order, repeats and negatives allowed,
    with numpy's semantics (an out-of-range row raises IndexError); an empty `rows` gives a `(0, ...)` array.
    No memory map outlives the call.

    Raises:
        KeyError: If the archive has no member `name`.
        TypeError: If `rows` is not a 1-D sequence of integers.
        ValueError: As `_stored_layout`.
    """
    idx = np.asarray(rows)
    if idx.size == 0:
        idx = idx.astype(np.intp).reshape(0)
    if idx.ndim != 1 or idx.dtype.kind not in "iu":
        raise TypeError(f"rows must be a 1-D sequence of integers, got {idx.dtype} with shape {idx.shape}")
    path = Path(path)
    layout = _stored_layout(path, name)
    if layout is None:
        with np.load(path, allow_pickle=False) as data:
            return np.ascontiguousarray(np.asarray(data[name])[idx])
    offset, shape, dtype = layout
    mapped = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="C")
    try:
        out = np.asarray(mapped[idx])       # integer-array indexing gathers into a fresh array, never a view of the map
    finally:
        del mapped
    return out
