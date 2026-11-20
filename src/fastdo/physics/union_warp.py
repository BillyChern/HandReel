"""mujoco_warp side of the slot-union model: per-world rows on ONE `mjwarp.Model` (retargeting env only).

`put_union_model` calls `mjwarp.put_model` on the compiled union and replaces, BEFORE any step or graph capture (the
broadphase filter specializes on `geom_aabb.shape[0] > 1` when its kernel is built), the batched fields by
(nworld, ...) arrays whose row w is world w's clip's row (`union_rows.UnionTable`):
`geom_pos`, `BODY_FIELDS`, `DOF_FIELDS`, `opt.tolerance` (meaninertia-rescaled), `opt.gravity` and `eq_data`.
Every kernel reads these as `arr[worldid % arr.shape[0]]` (pinned mujoco_warp 5d5a6450), so world w steps with
exactly its clip's rows. EXCEPTION: geoms of the world body (the pedestals) are static; mujoco_warp computes their
`geom_xpos` once in `put_data` and its kinematics skips them (`smooth.py` `_geom_local_to_global`), so their per-world
parking lives in the DATA: `write_static_poses` writes each world's static `geom_xpos` rows (`geom_pos` rows of a
static geom are never read). `world_clip[w] = -1` marks an EMPTY world: it gets clip 0's rows with every clip-specific
geom parked (a hand alone), so an unfilled slot never makes contacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastdo.physics.union_rows import BODY_FIELDS, DOF_FIELDS, UnionTable

__all__ = [
    "EMPTY",
    "put_union_model",
    "static_geoms",
    "static_xpos_rows",
    "tolerance_floor",
    "union_data0",
    "world_rows",
    "write_static_poses",
]

EMPTY = -1


def tolerance_floor() -> float:
    """mujoco_warp's float32 clamp of `opt.tolerance` (`io.put_model`: max(tolerance, _TOLERANCE_F32))."""
    from mujoco_warp._src import io as mjw_io

    return float(mjw_io._TOLERANCE_F32)


def world_rows(table: UnionTable, world_clip: np.ndarray, tol_floor: float = 0.0) -> dict[str, np.ndarray]:
    """Per-world float64 rows for `world_clip` (clip index per world; EMPTY = no clip: everything parked)."""
    wc = np.asarray(world_clip, dtype=np.int64)
    if wc.ndim != 1 or wc.size == 0:
        raise ValueError("world_clip must be a non-empty 1-D array")
    K = table.geom_pos.shape[0]
    if ((wc < EMPTY) | (wc >= K)).any():
        raise ValueError(f"world_clip entries must be in [-1, {K})")
    src = np.where(wc == EMPTY, 0, wc)
    rows: dict[str, np.ndarray] = {"geom_pos": table.geom_pos[src].copy()}
    rows["geom_pos"][wc == EMPTY] = table.empty_geom_pos
    for f in BODY_FIELDS:
        rows[f] = table.body[f][src].copy()
    for f in DOF_FIELDS:
        rows[f] = table.dof[f][src].copy()
    rows["tolerance"] = table.tolerance_rows(tol_floor)[src].copy()
    return rows


def _batched(wp: Any, like: Any, rows: np.ndarray) -> Any:
    arr = wp.array(rows, dtype=like.dtype, device=like.device)
    arr._is_batched = True
    return arr


def put_union_model(mjm_union: Any, table: UnionTable, world_clip: np.ndarray, eq_data: np.ndarray | None = None,
                    gravity: np.ndarray | None = None, device: str | None = None) -> Any:
    """`mjwarp.put_model(mjm_union)` with per-world rows (module docstring). `eq_data` (nworld, neq, 11) and `gravity`
    (nworld, 3) default to the union's compiled rows broadcast; callers overwrite them per slot at run time."""
    import mujoco_warp as mjwarp
    import warp as wp

    nw = len(world_clip)
    rows = world_rows(table, world_clip, tolerance_floor())
    with wp.ScopedDevice(device):
        m = mjwarp.put_model(mjm_union)
        m.geom_pos = _batched(wp, m.geom_pos, rows["geom_pos"])
        for f in BODY_FIELDS + DOF_FIELDS:
            setattr(m, f, _batched(wp, getattr(m, f), rows[f]))
        m.opt.tolerance = _batched(wp, m.opt.tolerance, rows["tolerance"])
        g = np.repeat(np.asarray(mjm_union.opt.gravity, dtype=np.float64)[None], nw, 0) if gravity is None else gravity
        m.opt.gravity = _batched(wp, m.opt.gravity, np.asarray(g, dtype=np.float64).reshape(nw, 3))
        e = (np.repeat(np.asarray(mjm_union.eq_data, dtype=np.float64)[None], nw, 0) if eq_data is None else eq_data)
        m.eq_data = _batched(wp, m.eq_data, np.asarray(e, dtype=np.float64).reshape(nw, mjm_union.neq, -1))
    return m


def static_geoms(mjm: Any) -> np.ndarray:
    """(ngeom,) bool: geoms mujoco_warp treats as static (welded to the world, not under a mocap body)."""
    body = np.asarray(mjm.geom_bodyid)
    return (np.asarray(mjm.body_weldid)[body] == 0) & (np.asarray(mjm.body_mocapid)[np.asarray(mjm.body_rootid)[body]]
                                                        == -1)


def static_xpos_rows(mjm_union: Any, table: UnionTable, world_clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(static geom ids, (nworld, nstatic, 3) float64 geom_xpos): compiled world pose, parked unless active."""
    import mujoco

    from fastdo.physics.union_scene import PARK_WORLD_OFFSET

    d = mujoco.MjData(mjm_union)
    mujoco.mj_kinematics(mjm_union, d)
    ids = np.flatnonzero(static_geoms(mjm_union))
    wc = np.asarray(world_clip, dtype=np.int64)
    base = np.asarray(d.geom_xpos, dtype=np.float64)[ids]
    out = np.repeat(base[None], len(wc), 0)
    for w, k in enumerate(wc):
        act = (table.active[k][ids] if k != EMPTY
               else np.array([n not in table.spec.owner for n in _names(mjm_union, ids)], dtype=bool))
        out[w, ~act] += np.asarray(PARK_WORLD_OFFSET)
    return ids, out


def _names(mjm: Any, ids: np.ndarray) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or "" for i in ids]


def write_static_poses(d: Any, mjm_union: Any, table: UnionTable, world_clip: np.ndarray,
                       worlds: slice = slice(None)) -> None:
    """Write `geom_xpos` of the static geoms for `worlds` (the clip of world w is `world_clip[w]`, full length)."""
    import torch
    import warp as wp

    ids, rows = static_xpos_rows(mjm_union, table, world_clip)
    gx = wp.to_torch(d.geom_xpos)
    idx = torch.as_tensor(ids, device=gx.device)
    vals = torch.as_tensor(rows[worlds], dtype=gx.dtype, device=gx.device)
    w = torch.arange(gx.shape[0], device=gx.device)[worlds]
    gx[w[:, None], idx[None, :]] = vals


def union_data0(mjm_union: Any) -> Any:
    """The host MjData `mjwarp.put_data` starts the union from: kinematics only (static geom poses), NO collision.
    `mj_forward` would collide every clip's unparked geoms at qpos0 with each other and the hand (CPU MuJoCo has no
    per-world rows), and `put_data` refuses nefc > njmax; no union state is ever taken from this data (each slot's
    state is copied from its clip's own env, `union_batch.adopt`)."""
    import mujoco

    d = mujoco.MjData(mjm_union)
    mujoco.mj_kinematics(mjm_union, d)
    return d
