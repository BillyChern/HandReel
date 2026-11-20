"""The `geometry.npz` pointmap codec: dense (the original format) or masked.

A dense artifact stores `pointmaps` (N,H,W,3) float32: 2.49 GB for a 100-frame 1080p Ego4D window, whose
write (4.38 s) and read (7.51 s) cost ~20% of the measured 58.22 s/window budget (quiet card, 2026-09-22).
The solver consumes none of it except the OBJECT pixels of every frame (`batch.py`: it packs
`pointmaps[k][masks[k] & finite]`) and the KEYFRAME, whose pointmap it also samples under the track queries
(`assemble_real_observations`) and registers against.

So the masked format stores exactly that: every frame's pixels under its object mask (row-major, NaNs kept,
so `mask & finite` re-applies identically), plus the keyframe dense. `unpack_pointmaps` rebuilds the dense
array with NaN everywhere else -- the same semantics a dense artifact has there -- so every downstream
consumer sees an identical tensor on every pixel it reads. The solve stage skips that rebuild: it reads
`object_points` as the solver's `ObjectPixels` (`masked_object_points`) and the keyframe from `keyframe_pointmap`.
The mask the payload was cut with is pinned by digest: a `mask.npz` regenerated after `geometry` would otherwise
scatter the points onto the wrong pixels, silently, and the reader refuses instead.

NOT for consumers that read off-object pixels: driftqc samples pointmaps at TRACKED pixels, which can leave
the mask (that is what it detects), so it refuses a masked artifact.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

__all__ = [
    "MASKED_FORMAT",
    "check_masked_shapes",
    "is_masked",
    "mask_digest",
    "masked_object_points",
    "masked_payload",
    "pack_masked",
    "unpack_pointmaps",
]

MASKED_FORMAT = "masked_v1"


def mask_digest(masks: np.ndarray) -> str:
    """sha256 of a (N,H,W) boolean mask stack, including its shape."""
    m = np.ascontiguousarray(np.asarray(masks, dtype=bool))
    h = hashlib.sha256(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(np.packbits(m, axis=None).tobytes())
    return h.hexdigest()


def is_masked(geometry: Mapping[str, np.ndarray]) -> bool:
    return "pointmaps_format" in geometry


def pack_masked(pointmaps: np.ndarray, masks: np.ndarray, keyframe_index: int) -> dict[str, np.ndarray]:
    """The masked encoding of dense `pointmaps` (N,H,W,3) under `masks` (N,H,W), keeping frame
    `keyframe_index` dense. Returns the arrays that replace `pointmaps` in `geometry.npz`.

    Raises:
        ValueError: On mismatched shapes or an out-of-range keyframe.
    """
    pm = np.asarray(pointmaps, dtype=np.float32)
    mk = np.asarray(masks, dtype=bool)
    check_masked_shapes(pm.shape, mk.shape, keyframe_index)
    # boolean indexing is row-major, frame-major
    return masked_payload(pm[mk], pm[keyframe_index].copy(), mk, keyframe_index)


def check_masked_shapes(pointmaps_shape: tuple[int, ...], masks_shape: tuple[int, ...], keyframe_index: int) -> None:
    """`pack_masked`'s checks on shapes alone, for a caller that never holds the dense pointmaps.

    Raises:
        ValueError: On mismatched shapes or an out-of-range keyframe.
    """
    if len(pointmaps_shape) != 4 or pointmaps_shape[-1] != 3 or tuple(masks_shape) != tuple(pointmaps_shape[:3]):
        raise ValueError(f"pointmaps {tuple(pointmaps_shape)} and masks {tuple(masks_shape)} must be (N,H,W,3) and "
                         "(N,H,W)")
    n = pointmaps_shape[0]
    if not 0 <= keyframe_index < n:
        raise ValueError(f"keyframe_index {keyframe_index} out of range [0,{n})")


def masked_payload(object_points: np.ndarray, keyframe_pointmap: np.ndarray, masks: np.ndarray,
                   keyframe_index: int) -> dict[str, np.ndarray]:
    """`pack_masked`'s arrays from what it keeps: `object_points` (P,3) float32, the pixels under `masks` (N,H,W) in
    `pointmaps[masks]` order, and `keyframe_pointmap` (H,W,3) float32, frame `keyframe_index` (the geometry
    stage gathers exactly these on the GPU). The caller has validated the shapes (`check_masked_shapes`)."""
    mk = np.asarray(masks, dtype=bool)
    n = mk.shape[0]
    counts = mk.reshape(n, -1).sum(axis=1).astype(np.int64)
    return {
        "pointmaps_format": np.array(MASKED_FORMAT),
        "object_points": object_points,
        "object_offsets": np.concatenate([np.zeros(1, np.int64), np.cumsum(counts)]),
        "keyframe_pointmap": keyframe_pointmap,
        "keyframe_index": np.array(keyframe_index, dtype=np.int64),
        "image_shape": np.asarray(mk.shape, dtype=np.int64),
        "mask_sha256": np.array(mask_digest(mk)),
    }


def masked_object_points(geometry: Mapping[str, np.ndarray], masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A masked artifact's `(object_points (P0,3) float32, object_offsets (N+1,) int64)`, validated against `masks`
    (N,H,W): frame k's pixels, row-major and NaNs kept, are `object_points[offsets[k]:offsets[k+1]]` (the
    solver reads these directly instead of a rebuilt dense array).

    Raises:
        ValueError: If the artifact is not masked, was cut with a different mask than `masks`, or is malformed.
    """
    if not is_masked(geometry):
        raise ValueError("geometry.npz is dense: it has no object_points")
    fmt = str(np.asarray(geometry["pointmaps_format"]))
    if fmt != MASKED_FORMAT:
        raise ValueError(f"unknown geometry pointmaps_format {fmt!r}; this reader knows {MASKED_FORMAT!r}")
    mk = np.asarray(masks, dtype=bool)
    shape = tuple(int(v) for v in np.asarray(geometry["image_shape"]))
    if mk.shape != shape:
        raise ValueError(f"masks {mk.shape} do not match the masked artifact's frames {shape}")
    if mask_digest(mk) != str(np.asarray(geometry["mask_sha256"])):
        raise ValueError("geometry.npz was cut with a different mask than mask.npz (was the mask stage re-run "
                         "after geometry?); re-run geometry, since the stored points would land on the wrong pixels")
    counts = np.count_nonzero(mk.reshape(shape[0], -1), axis=1).astype(np.int64)
    n_points = int(counts.sum())
    points = np.asarray(geometry["object_points"], dtype=np.float32)
    if points.shape != (n_points, 3):
        raise ValueError(f"object_points {points.shape} does not match the mask's {n_points} pixels")
    offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(counts)])
    if not np.array_equal(np.asarray(geometry["object_offsets"]), offsets):
        raise ValueError("object_offsets do not match the mask's per-frame pixel counts")
    return points, offsets


def unpack_pointmaps(geometry: Mapping[str, np.ndarray], masks: np.ndarray) -> np.ndarray:
    """Dense (N,H,W,3) float32 pointmaps from either format. A dense artifact is returned as stored.

    Raises:
        ValueError: If a masked artifact was cut with a different mask than `masks`, or is malformed.
    """
    if not is_masked(geometry):
        return np.asarray(geometry["pointmaps"], dtype=np.float32)
    points, _ = masked_object_points(geometry, masks)
    mk = np.asarray(masks, dtype=bool)
    dense = np.full((*mk.shape, 3), np.nan, dtype=np.float32)
    dense[mk] = points
    k = int(np.asarray(geometry["keyframe_index"]))
    dense[k] = np.asarray(geometry["keyframe_pointmap"], dtype=np.float32)
    return dense
