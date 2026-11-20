"""The batched hand npz (D3b): hold rows, the 23-key file contract and MANO's closed wrist.

Shared module: numpy and the standard library only; nothing from `fastdo` or the other shared modules.

Keys, for T rows:
- the 16 keys HaWoR's `demo.py` writes, per side (`left`, `right`): `{side}_vertices` (T,778,3) float32,
  `{side}_faces` (1552,3) int32, `{side}_joints` (T,21,3) float32, `{side}_trans` (T,3), `{side}_rot` (T,3),
  `{side}_hand_pose` (T,45) and `{side}_betas` (T,10) float32, and `{side}_valid` (T,) bool;
- `frame_idx` (T,) int64, strictly increasing, and `fps`, 0-d float64;
- `{side}_infilled` (T,) bool and `{side}_box` (T,5) float32: the filtered `[x1,y1,x2,y2,conf]` at detected rows,
  zeros elsewhere;
- `producer`: 0-d unicode, `PRODUCER`.

Every float array is finite. `valid` means detected or interior-filled. HaWoR marks every row valid after its infiller;
here leading and trailing gap rows are `False` and hold copies of the nearest valid row, so consumers must mask them.
`fastdo.hands.hawor_npz.load_hawor_npz` reads the file unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "NPZ_SIDE_KEYS",
    "PRODUCER",
    "SideArrays",
    "assemble_side",
    "closed_faces",
    "nearest_valid_rows",
    "validate_hand_npz",
    "write_hand_npz",
]

NPZ_SIDE_KEYS: tuple[str, ...] = ("vertices", "faces", "joints", "trans", "rot", "hand_pose", "betas", "valid",
                                  "infilled", "box")
PRODUCER = "fastdo.hands.hawor_batched/1"
_SIDES = ("left", "right")
_N_FACES = 1552


@dataclass(frozen=True)
class SideArrays:
    """One hand's arrays for `write_hand_npz` (faces are passed separately, per side)."""

    vertices: np.ndarray
    joints: np.ndarray
    trans: np.ndarray
    rot: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    valid: np.ndarray
    infilled: np.ndarray
    box: np.ndarray


def nearest_valid_rows(valid: np.ndarray) -> np.ndarray:
    """`(T,)` int64: the nearest valid row by index (ties to the lower index), -1 everywhere when none is valid."""
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    good = np.flatnonzero(mask)
    if good.size == 0:
        return np.full(mask.shape, -1, dtype=np.int64)
    idx = np.arange(mask.size)
    pos = np.searchsorted(good, idx)
    after = good[np.minimum(pos, good.size - 1)]
    before = good[np.maximum(pos - 1, 0)]
    return np.where(np.abs(idx - before) <= np.abs(after - idx), before, after).astype(np.int64)


def assemble_side(rows: Mapping[str, np.ndarray], vertices: np.ndarray, joints: np.ndarray, infilled: np.ndarray,
                  box: np.ndarray) -> SideArrays:
    """Completes one hand: invalid rows copy `trans`, `rot`, `hand_pose`, `vertices` and `joints` from their nearest
    valid row, and get the valid rows' mean `betas`; a side with no valid row is all zeros. `box` rows with NaN become
    zeros."""
    valid = np.asarray(rows["valid"], dtype=bool).reshape(-1)
    t = valid.size
    has_valid = bool(valid.any())
    nearest = nearest_valid_rows(valid)
    invalid = ~valid

    def held(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        arr = np.array(values, dtype=np.float32, copy=True).reshape((t, *shape))
        if not has_valid:
            return np.zeros_like(arr)
        arr[invalid] = arr[nearest[invalid]]
        return arr

    betas = np.array(rows["betas"], dtype=np.float32, copy=True).reshape(t, 10)
    if has_valid:
        betas[invalid] = betas[valid].astype(np.float64).mean(axis=0).astype(np.float32)
    else:
        betas = np.zeros_like(betas)
    box_arr = np.asarray(box, dtype=np.float64).reshape(t, 5)
    detected = np.isfinite(box_arr).all(axis=1) & has_valid
    return SideArrays(vertices=held(vertices, (778, 3)), joints=held(joints, (21, 3)), trans=held(rows["trans"], (3,)),
                      rot=held(rows["rot"], (3,)), hand_pose=held(rows["hand_pose"], (45,)), betas=betas,
                      valid=valid.copy(), infilled=np.asarray(infilled, dtype=bool).reshape(t).copy(),
                      box=np.where(detected[:, None], np.nan_to_num(box_arr), 0.0).astype(np.float32))


def _expected(t: int) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    shapes = {"vertices": (t, 778, 3), "faces": (_N_FACES, 3), "joints": (t, 21, 3), "trans": (t, 3), "rot": (t, 3),
              "hand_pose": (t, 45), "betas": (t, 10), "valid": (t,), "infilled": (t,), "box": (t, 5)}
    dtypes: dict[str, np.dtype[Any]] = {"faces": np.dtype(np.int32), "valid": np.dtype(bool),
                                        "infilled": np.dtype(bool)}
    spec: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {"frame_idx": ((t,), np.dtype(np.int64)),
                                                              "fps": ((), np.dtype(np.float64))}
    for side in _SIDES:
        for key in NPZ_SIDE_KEYS:
            spec[f"{side}_{key}"] = (shapes[key], dtypes.get(key, np.dtype(np.float32)))
    return spec


def _check(arrays: Mapping[str, np.ndarray], source: str) -> int:
    if "frame_idx" not in arrays:
        raise ValueError(f"{source}: missing key frame_idx")
    frame_idx = np.asarray(arrays["frame_idx"])
    if frame_idx.ndim != 1:
        raise ValueError(f"{source}: frame_idx must be (T,), got {frame_idx.shape}")
    if frame_idx.size > 1 and not bool((np.diff(frame_idx) > 0).all()):
        raise ValueError(f"{source}: frame_idx is not strictly increasing")
    spec = _expected(int(frame_idx.size))
    for key, (shape, dtype) in spec.items():
        if key not in arrays:
            raise ValueError(f"{source}: missing key {key}")
        arr = np.asarray(arrays[key])
        if arr.shape != shape or arr.dtype != dtype:
            raise ValueError(f"{source}: {key} must be {shape} {dtype}, got {arr.shape} {arr.dtype}")
        if arr.dtype.kind == "f" and not bool(np.isfinite(arr).all()):
            raise ValueError(f"{source}: {key} has non-finite values")
    if not float(arrays["fps"]) > 0:
        raise ValueError(f"{source}: fps must be positive")
    producer = np.asarray(arrays.get("producer", np.array(0)))
    if producer.ndim != 0 or producer.dtype.kind != "U":
        raise ValueError(f"{source}: producer must be a 0-d unicode array")
    extra = sorted(set(arrays) - set(spec) - {"producer"})
    if extra:
        raise ValueError(f"{source}: unexpected key {extra[0]}")
    return int(frame_idx.size)


def write_hand_npz(path: Path, frame_idx: np.ndarray, fps: float, left: SideArrays, right: SideArrays,
                   faces: Mapping[str, np.ndarray]) -> None:
    """Checks every key, then writes an uncompressed npz to `path.with_suffix(".tmp.npz")` and moves it into place.

    Raises:
        ValueError: Naming the first key with a wrong shape or dtype or a non-finite value; nothing is written then.
    """
    arrays: dict[str, np.ndarray] = {"frame_idx": np.asarray(frame_idx), "fps": np.array(fps, dtype=np.float64),
                                     "producer": np.array(PRODUCER)}
    for side, data in (("left", left), ("right", right)):
        for key in NPZ_SIDE_KEYS:
            arrays[f"{side}_{key}"] = np.asarray(faces[side] if key == "faces" else getattr(data, key))
    path = Path(path)
    _check(arrays, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    try:
        with tmp.open("wb") as f:
            np.savez(f, **arrays)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def validate_hand_npz(path: Path) -> int:
    """Reopens a hand npz without pickle, checks all 23 keys and `frame_idx` ordering, and returns T.

    Raises:
        ValueError: Naming the first bad or missing key.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return _check(arrays, str(path))


def closed_faces(mano_faces: np.ndarray) -> dict[str, np.ndarray]:
    """Right and left face arrays with the mesh's single boundary loop capped by a fan.

    Boundary directed edges are those whose reverse appears in no face. The loop `L` starts at its smallest vertex `v0`
    and follows them; the cap is `[v0, L[k+1], L[k]]` for `k = 1 .. len(L) - 2`, so every directed edge appears once
    with its reverse present. Right faces are `concat(mano_faces, cap)` as int32; left faces are `right[:, [0, 2, 1]]`.

    Raises:
        ValueError: If the boundary edges do not form exactly one loop.
    """
    faces = np.asarray(mano_faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).tolist()
    directed = {(int(i), int(j)) for i, j in edges}
    successor: dict[int, int] = {}
    for i, j in directed:
        if (j, i) not in directed:
            if i in successor:
                raise ValueError(f"boundary vertex {i} starts two boundary edges; not a single loop")
            successor[i] = j
    if not successor:
        raise ValueError("the mesh has no boundary loop to cap")
    v0 = min(successor)
    loop = [v0]
    while successor.get(loop[-1], v0) != v0 and len(loop) <= len(successor):
        loop.append(successor[loop[-1]])
    if loop[-1] not in successor or successor[loop[-1]] != v0 or len(loop) != len(successor):
        raise ValueError(f"boundary edges do not form exactly one loop ({len(successor)} edges, loop of {len(loop)})")
    cap = np.array([[v0, loop[k + 1], loop[k]] for k in range(1, len(loop) - 1)], dtype=np.int64).reshape(-1, 3)
    right = np.concatenate([faces, cap]).astype(np.int32)
    return {"right": right, "left": np.ascontiguousarray(right[:, [0, 2, 1]])}
