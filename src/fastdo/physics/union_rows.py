"""Per-clip model rows of a slot-union model: which world parameters make world w represent clip k.

`union_table` reads the compiled union and the K compiled clip models (each configured as DO AS I DO's
`setup_mj_model` configures it) and returns, per clip, the rows of the per-world (batched) fields:
- `geom_pos` (ngeom,): clip k's own geoms and the shared geoms at their compiled pose (equal to clip k's), every other
  clip's piece/support parked by `PARK_BODY_OFFSET` (body frame) and every other clip's pedestal by
  `PARK_WORLD_OFFSET` (world frame);
- every `BODY_FIELDS` row (nbody,) and `DOF_FIELDS` row (nv,): clip k's model's own rows (the union's object body has
  the inertia of ALL clips' pieces; only these rows depend on it); `CPU_*_FIELDS` are the further rows CPU MuJoCo reads
  (sameframe/simple shortcuts, dof_M0, dof_length) and mujoco_warp's step does not;
- `meaninertia` (scalar per clip): CPU MuJoCo reads it from the model; mujoco_warp passes ONE Python float to its solver
  (not batched), so its per-world `opt.tolerance` is rescaled instead: tolerance_k * meaninertia_k / meaninertia_union
  gives the same thresholds (`solver.py` `_rescale`, `linesearch_iterative` gtol) up to float32 rounding.
`check_union_compatible` refuses a union whose shared structure differs from any clip, or whose geoms, meshes or pairs
do not reproduce a clip's exactly. `apply_clip_rows` writes clip k's rows into a CPU `MjModel` (the CPU twin).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fastdo.physics.union_scene import (
    PARK_BODY_OFFSET,
    PARK_WORLD_OFFSET,
    UnionSpec,
    clip_specific_kind,
)

__all__ = ["BODY_FIELDS", "CPU_BODY_FIELDS", "CPU_DOF_FIELDS", "DOF_FIELDS", "UnionTable", "apply_clip_rows",
           "check_union_compatible", "geom_names", "union_table"]

BODY_FIELDS = ("body_mass", "body_subtreemass", "body_inertia", "body_ipos", "body_iquat", "body_invweight0")
CPU_BODY_FIELDS = ("body_sameframe", "body_simple")          # CPU MuJoCo shortcuts; mujoco_warp never reads them
DOF_FIELDS = ("dof_invweight0",)
CPU_DOF_FIELDS = ("dof_M0", "dof_length")                     # read by CPU MuJoCo only (not by mujoco_warp's step)
GEOM_EXACT_FIELDS = ("geom_type", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority", "geom_size",
                     "geom_pos", "geom_quat", "geom_aabb", "geom_rbound", "geom_friction", "geom_margin", "geom_gap",
                     "geom_solref", "geom_solimp", "geom_solmix", "geom_group")
PAIR_FIELDS = ("pair_dim", "pair_friction", "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin",
               "pair_gap")
MESH_FIELDS = ("mesh_vertnum", "mesh_graphadr", "mesh_polynum")
STRUCT = ("nq", "nv", "nu", "na", "nbody", "njnt", "neq", "nsite", "nmocap", "ntendon", "nsensor")
SHARED_PREFIXES = ("jnt_", "dof_", "actuator_", "eq_", "site_", "body_parentid", "body_rootid", "body_weldid",
                   "body_jntnum", "body_jntadr", "body_dofnum", "body_dofadr", "body_pos", "body_quat",
                   "body_mocapid", "body_gravcomp", "qpos0", "qpos_spring")
NOT_SHARED = {"dof_invweight0", "dof_M0", "dof_length", "actuator_acc0", "eq_data"}  # per-clip rows / runtime / unused


@dataclass(frozen=True)
class UnionTable:
    spec: UnionSpec
    geom_map: tuple[np.ndarray, ...]     # per clip: (ngeom_clip,) union geom id, -1 for a dropped geom
    active: np.ndarray                   # (K, ngeom_union) bool: geom is shared or clip k's own
    geom_pos: np.ndarray                 # (K, ngeom_union, 3) float64
    geom_sameframe: np.ndarray           # (K, ngeom_union) CPU MuJoCo only: clip k's flags on its geoms
    empty_geom_pos: np.ndarray           # (ngeom_union, 3): every clip-specific geom parked (an unfilled world)
    body: dict[str, np.ndarray]          # BODY_FIELDS + CPU_BODY_FIELDS -> (K, nbody, ...)
    dof: dict[str, np.ndarray]           # DOF_FIELDS + CPU_DOF_FIELDS -> (K, nv)
    meaninertia: np.ndarray              # (K,)
    union_meaninertia: float
    tolerance: np.ndarray                # (K,) clip opt.tolerance (the registered per-clip value)

    def tolerance_rows(self, floor: float = 0.0) -> np.ndarray:
        """(K,) mujoco_warp `opt.tolerance` for a world of clip k under the union's single `stat.meaninertia`.
        `floor`: the engine's own lower clamp of the tolerance (mujoco_warp `put_model` uses max(t, 1e-6)), applied
        BEFORE the rescale so that the effective thresholds equal the single-clip model's."""
        return np.maximum(self.tolerance, floor) * self.meaninertia / self.union_meaninertia


def geom_names(model: Any) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]


def _body_names(model: Any) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]


def _map_geoms(spec: UnionSpec, k: int, clip_names: list[str], union_names: list[str]) -> np.ndarray:
    """(ngeom_clip,) union geom id of each of clip k's geoms (-1: dropped). Clip-specific geoms map by their union
    name; shared geoms (some unnamed) map in order: the i-th shared geom of the clip is the union's i-th."""
    side = spec.side
    uidx = {n: i for i, n in enumerate(union_names) if n in spec.owner}
    shared_u = [i for i, n in enumerate(union_names) if n not in spec.owner]
    out = np.full(len(clip_names), -1, dtype=np.int64)
    j = 0
    for g, name in enumerate(clip_names):
        if clip_specific_kind(name, side) is None:
            if j >= len(shared_u) or union_names[shared_u[j]] != name:
                raise ValueError(f"clip {k}: shared geom {g} ({name!r}) does not match the union's shared geom order")
            out[g] = shared_u[j]
            j += 1
            continue
        un = spec.union_name(k, name)
        if un is None:
            continue
        if un not in uidx:
            raise ValueError(f"clip {k}: geom {name!r} ({un!r}) missing from the union")
        out[g] = uidx[un]
    if j != len(shared_u):
        raise ValueError(f"clip {k}: {len(shared_u) - j} shared union geoms have no counterpart in the clip")
    return out


def _pairs(model: Any, names: list[str]) -> list[tuple[str, str]]:
    return [(names[a], names[b]) for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True)]


def check_union_compatible(union: Any, clips: Sequence[Any], spec: UnionSpec) -> None:
    """Raise ValueError naming the clip and the field where clip k is not reproduced by the union's rows."""
    un = geom_names(union)
    upairs = _pairs(union, un)
    # mjSAMEFRAME_BODY (1) / _INERTIA (2) make CPU MuJoCo copy the body frame and ignore geom_pos (so the parking)
    framed = [n for i, n in enumerate(un) if spec.n_clips > 1 and n in spec.owner and union.geom_sameframe[i] in (1, 2)]
    if framed:
        raise ValueError(f"clip-specific union geoms whose geom_sameframe makes CPU MuJoCo ignore geom_pos: {framed}")
    for k, m in enumerate(clips):
        def bad(what: str, k: int = k) -> ValueError:
            return ValueError(f"clip {k} ({spec.scenes[k]}): {what}")

        for s in STRUCT:
            if getattr(m, s) != getattr(union, s):
                raise bad(f"{s} {getattr(m, s)} != union {getattr(union, s)}")
        if _body_names(m) != _body_names(union):
            raise bad("body names/order differ")
        for f in dir(m.opt):
            if f.startswith("_"):
                continue
            a, b = getattr(m.opt, f), getattr(union.opt, f)
            if callable(a):
                continue
            if f != "tolerance" and not np.array_equal(np.asarray(a), np.asarray(b)):
                raise bad(f"opt.{f} {a} != union {b}")
        for f in dir(m):
            if not f.startswith(SHARED_PREFIXES) or f in NOT_SHARED:
                continue
            a = getattr(m, f)
            if isinstance(a, np.ndarray) and not np.array_equal(a, getattr(union, f)):
                raise bad(f"{f} differs")
        cn = geom_names(m)
        gmap = _map_geoms(spec, k, cn, un)
        keep = gmap >= 0
        for f in GEOM_EXACT_FIELDS:
            if not np.array_equal(getattr(m, f)[keep], getattr(union, f)[gmap[keep]]):
                raise bad(f"{f} rows differ")
        bodyid_c = m.geom_bodyid[keep]
        if not np.array_equal(bodyid_c, union.geom_bodyid[gmap[keep]]):
            raise bad("geom_bodyid differs")
        for g in np.flatnonzero(keep & (m.geom_dataid >= 0)):
            mc, mu = m.geom_dataid[g], union.geom_dataid[gmap[g]]
            for f in MESH_FIELDS:
                if (getattr(m, f)[mc] >= 0) != (getattr(union, f)[mu] >= 0) or (
                        f != "mesh_graphadr" and getattr(m, f)[mc] != getattr(union, f)[mu]):
                    raise bad(f"{f} of mesh for geom {cn[g]!r} differs")
            vc = m.mesh_vert[m.mesh_vertadr[mc]: m.mesh_vertadr[mc] + m.mesh_vertnum[mc]]
            vu = union.mesh_vert[union.mesh_vertadr[mu]: union.mesh_vertadr[mu] + union.mesh_vertnum[mu]]
            if not np.array_equal(vc, vu):
                raise bad(f"mesh vertices of geom {cn[g]!r} differ")
        cpairs = [(spec.union_name(k, a), spec.union_name(k, b)) for a, b in _pairs(m, cn)]
        act = {spec.union_name(k, n) for n in cn} - {None}
        sel = [i for i, (a, b) in enumerate(upairs) if a in act and b in act]
        if [upairs[i] for i in sel] != cpairs:
            raise bad("the union's pairs among this clip's geoms are not the clip's pairs in the clip's order")
        for f in PAIR_FIELDS:
            if not np.array_equal(getattr(m, f), getattr(union, f)[sel]):
                raise bad(f"{f} rows differ")


def union_table(union: Any, clips: Sequence[Any], spec: UnionSpec, check: bool = True) -> UnionTable:
    """Per-clip rows of the batched fields (module docstring); `clips[k]` is clip k's configured model."""
    if check:
        check_union_compatible(union, clips, spec)
    un = geom_names(union)
    K, ng = len(clips), union.ngeom
    owner = np.array([spec.owner.get(n, -1) for n in un])
    is_world = union.geom_bodyid == 0
    geom_pos = np.repeat(np.asarray(union.geom_pos, dtype=np.float64)[None], K, axis=0)
    active = np.zeros((K, ng), dtype=bool)
    sameframe = np.repeat(np.asarray(union.geom_sameframe)[None], K, axis=0)
    gmaps = []
    for k, m in enumerate(clips):
        gmap = _map_geoms(spec, k, geom_names(m), un)
        gmaps.append(gmap)
        active[k] = (owner == -1) | (owner == k)
        park = ~active[k]
        geom_pos[k, park & ~is_world] += np.asarray(PARK_BODY_OFFSET)
        geom_pos[k, park & is_world] += np.asarray(PARK_WORLD_OFFSET)
        keep = gmap >= 0
        geom_pos[k, gmap[keep]] = m.geom_pos[keep]
        sameframe[k, gmap[keep]] = m.geom_sameframe[keep]
    empty = np.asarray(union.geom_pos, dtype=np.float64).copy()
    empty[(owner >= 0) & ~is_world] += np.asarray(PARK_BODY_OFFSET)
    empty[(owner >= 0) & is_world] += np.asarray(PARK_WORLD_OFFSET)
    body = {f: np.stack([np.asarray(getattr(m, f)).copy() for m in clips]) for f in BODY_FIELDS + CPU_BODY_FIELDS}
    dof = {f: np.stack([np.asarray(getattr(m, f)).copy() for m in clips]) for f in DOF_FIELDS + CPU_DOF_FIELDS}
    return UnionTable(spec=spec, geom_map=tuple(gmaps), active=active, geom_pos=geom_pos, geom_sameframe=sameframe,
                      empty_geom_pos=empty, body=body, dof=dof,
                      meaninertia=np.array([m.stat.meaninertia for m in clips], dtype=np.float64),
                      union_meaninertia=float(union.stat.meaninertia),
                      tolerance=np.array([m.opt.tolerance for m in clips], dtype=np.float64))


def apply_clip_rows(model: Any, table: UnionTable, k: int) -> None:
    """CPU twin: make `model` (a copy of the union) represent clip k (every per-clip row, meaninertia exact)."""
    model.geom_pos[:] = table.geom_pos[k]
    model.geom_sameframe[:] = table.geom_sameframe[k]
    for f, rows in table.body.items():
        getattr(model, f)[:] = rows[k]
    for f, rows in table.dof.items():
        getattr(model, f)[:] = rows[k]
    model.stat.meaninertia = float(table.meaninertia[k])
    model.opt.tolerance = float(table.tolerance[k])
