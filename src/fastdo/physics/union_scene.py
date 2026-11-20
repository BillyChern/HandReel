"""Slot-union MJCF for continuous batching of DO AS I DO's physics MPC across clips.

One MuJoCo model whose worlds can each represent any one of K clips of the same hand side. DO AS I DO's per-clip
scenes share the hand, actuators, weld and options and differ in the object (CoACD hull pieces on one free body),
the stabiliser supports on that body and the world-fixed pedestals, plus every explicit `<pair>` that touches them.

Mechanism (checked against the pinned mujoco_warp 5d5a6450 in the `retargeting` env; see `union_rows`):
- the union carries EVERY clip's pieces, supports and pedestals as separate geoms (renamed `<name>__u<k>`, meshes
  absolutized), and the union of the K pair lists merged in an order that keeps each clip's own pair order;
- a world of clip k keeps k's geoms at their compiled pose and PARKS every other clip's geom through the per-world
  (batched) `geom_pos` row: object-body geoms 1 km away in the body frame, pedestals 100 km away in the world frame,
  so the broadphase rejects every pair that touches a parked geom (no pair joins two parked geoms of DIFFERENT
  distance classes, and pairs only join geoms of one clip or the shared hand);
- the object body's inertial rows (and every body/dof row that depends on them) are per-world rows copied from clip
  k's own compiled model.
Why not a per-world mesh pointer: in this mujoco_warp `geom_dataid` and `stat.meaninertia` are NOT batched (plain
arrays / a Python float), while `geom_pos`, `body_*` inertials, `dof_invweight0`, `opt.tolerance`, `opt.gravity` and
`eq_data` are (`io.py` `create_nmodel_batched_array`; kernels read `arr[worldid % arr.shape[0]]`).

Geom ids are assigned so that each clip's geoms keep their relative order (pedestals < hand < pieces < supports, as in
the clip's own scene), and the merged pair list keeps each clip's relative pair order: CPU MuJoCo then produces each
active world's contacts in the clip's own order, and mujoco_warp's geom-pair (upper-triangle) order is preserved too.

Pure XML + numpy (no warp): importable in the main env and in the `retargeting` env.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PARK_BODY_OFFSET", "PARK_WORLD_OFFSET", "UnionSpec", "build_union_xml", "clip_specific_kind",
           "merge_orders"]

PARK_BODY_OFFSET = (1000.0, 0.0, 0.0)       # body frame; object-body geoms (pieces, supports) of other clips
PARK_WORLD_OFFSET = (0.0, 0.0, -1.0e5)      # world frame; pedestals of other clips


@dataclass(frozen=True)
class UnionSpec:
    """The union scene and how each clip's geoms are named in it."""

    xml_path: Path
    side: str
    scenes: tuple[Path, ...]
    rename: tuple[dict[str, str], ...]      # per clip: clip geom name -> union geom name (clip-specific geoms only)
    dropped: tuple[tuple[str, ...], ...]    # per clip: clip geom names absent from the union (the object visual)
    owner: dict[str, int]                   # union geom name -> owning clip (clip-specific geoms only)
    kind: dict[str, str]                    # union geom name -> "piece" | "support" | "pedestal"

    @property
    def n_clips(self) -> int:
        return len(self.scenes)

    def union_name(self, k: int, clip_name: str) -> str | None:
        """The union name of clip k's geom (None for a dropped one; shared geoms keep their name)."""
        if clip_name in self.dropped[k]:
            return None
        return self.rename[k].get(clip_name, clip_name)


def clip_specific_kind(name: str, side: str) -> str | None:
    """Kind of a clip-specific geom name in a DO AS I DO scene, or None for a shared (hand / scene) geom."""
    if re.fullmatch(rf"{side}_object_\d+", name):
        return "piece"
    if name == f"{side}_object_visual":
        return "visual"
    if name.startswith(f"{side}_support_"):
        return "support"
    if name.startswith(f"{side}_pedestal_"):
        return "pedestal"
    return None


def _is_clip_mesh(name: str, side: str) -> bool:
    return bool(re.fullmatch(rf"{side}_\d+", name)) or name == f"{side}_visual"


def merge_orders(seqs: Sequence[Sequence[str]]) -> list[str]:
    """One sequence containing every key of `seqs` that keeps each input's relative order (a stable topological
    merge: among ready keys the one first seen in the earliest input wins). Raises on contradictory orders."""
    rank: dict[str, tuple[int, int]] = {}
    succ: dict[str, set[str]] = {}
    indeg: dict[str, int] = {}
    for i, seq in enumerate(seqs):
        if len(set(seq)) != len(seq):
            raise ValueError(f"sequence {i} repeats a key")
        for j, key in enumerate(seq):
            rank.setdefault(key, (i, j))
            succ.setdefault(key, set())
            indeg.setdefault(key, 0)
        for a, b in itertools.pairwise(seq):
            if b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1
    ready = sorted((k for k, d in indeg.items() if d == 0), key=rank.__getitem__)
    out: list[str] = []
    while ready:
        key = ready.pop(0)
        out.append(key)
        for b in succ[key]:
            indeg[b] -= 1
            if indeg[b] == 0:
                ready.append(b)
        ready.sort(key=rank.__getitem__)
    if len(out) != len(indeg):
        raise ValueError("the inputs order their shared keys differently (no common supersequence)")
    return out


def _find(el: ET.Element, path: str) -> ET.Element:
    found = el.find(path)
    if found is None:
        raise ValueError(f"scene has no <{path}>")
    return found


def _get(el: ET.Element, key: str) -> str:
    v = el.get(key)
    if v is None:
        raise ValueError(f"<{el.tag}> has no {key!r}")
    return v


def _side(root: ET.Element) -> str:
    sides = [s for s in ("right", "left") if root.find(f".//body[@name='{s}_object']") is not None]
    if len(sides) != 1:
        raise ValueError(f"expected exactly one object body (right_object or left_object), found {sides}")
    return sides[0]


def _meshdir(root: ET.Element, scene: Path) -> Path:
    comp = root.find("compiler")
    rel = comp.get("meshdir", "") if comp is not None else ""
    rel = rel or ""
    return (scene.parent / rel).resolve()


def _pair_key(g1: str, g2: str) -> str:
    return f"{g1}|{g2}"


def build_union_xml(scenes: Sequence[str | Path], out_dir: str | Path, keep_visual: bool = False) -> UnionSpec:
    """Write the union of K same-side DO AS I DO scenes (clip 0 is the template) and return its `UnionSpec`.

    The written path is `<out_dir>/<12 hex of a sha256 over the scene XML bytes and every piece mesh>/union.xml`;
    an existing file there is reused. The object visual geoms and meshes are dropped (density 0, never in a pair)."""
    paths: tuple[Path, ...] = tuple(Path(s).resolve() for s in scenes)
    if not paths:
        raise ValueError("no scenes")
    roots = [ET.parse(s).getroot() for s in paths]
    side = _side(roots[0])
    comp0 = {k: v for k, v in (_find(roots[0], "compiler").attrib.items()) if k != "meshdir"}
    for k, (s, r) in enumerate(zip(paths, roots, strict=True)):
        if _side(r) != side:
            raise ValueError(f"clip {k} ({s}) is a {_side(r)}-hand scene; the union is {side}")
        comp = {a: v for a, v in _find(r, "compiler").attrib.items() if a != "meshdir"}
        if comp != comp0:
            raise ValueError(f"clip {k} ({s}) compiler settings {comp} differ from the template's {comp0}")
    union = copy.deepcopy(roots[0])
    _find(union, "compiler").set("meshdir", str(_meshdir(roots[0], paths[0])) + "/")
    asset = _find(union, "asset")
    for mesh in list(asset.findall("mesh")):
        if _is_clip_mesh(mesh.get("name", ""), side):
            asset.remove(mesh)
    world = _find(union, "worldbody")
    obj = _find(union, f".//body[@name='{side}_object']")
    for parent in (world, obj):
        for g in list(parent.findall("geom")):
            if clip_specific_kind(g.get("name", ""), side) is not None:
                parent.remove(g)
    contact = _find(union, "contact")
    for p in list(contact):
        contact.remove(p)

    rename: list[dict[str, str]] = []
    dropped: list[tuple[str, ...]] = []
    owner: dict[str, int] = {}
    kind: dict[str, str] = {}
    peds: list[ET.Element] = []
    pieces: list[ET.Element] = []
    supports: list[ET.Element] = []
    pair_seqs: list[list[str]] = []
    pair_elems: dict[str, ET.Element] = {}
    digest = hashlib.sha256(b"keep_visual" if keep_visual else b"")
    for k, (scene, root) in enumerate(zip(paths, roots, strict=True)):
        digest.update(scene.read_bytes())
        mdir = _meshdir(root, scene)
        ren: dict[str, str] = {}
        drop: list[str] = []
        mesh_ren: dict[str, str] = {}
        for mesh in _find(root, "asset").findall("mesh"):
            name = mesh.get("name", "")
            if not (re.fullmatch(rf"{side}_\d+", name) or (keep_visual and name == f"{side}_visual")):
                continue
            m2 = copy.deepcopy(mesh)
            m2.set("name", f"{name}__u{k}")
            f = (mdir / _get(mesh, "file")).resolve()
            digest.update(f.read_bytes())
            m2.set("file", str(f))
            mesh_ren[name] = _get(m2, "name")
            asset.append(m2)
        for parent in (_find(root, "worldbody"), _find(root, f".//body[@name='{side}_object']")):
            for g in parent.findall("geom"):
                name = g.get("name", "")
                kd = clip_specific_kind(name, side)
                if kd is None:
                    continue
                if kd == "visual" and not keep_visual:
                    drop.append(name)
                    continue
                g2 = copy.deepcopy(g)
                uname = f"{name}__u{k}"
                g2.set("name", uname)
                if kd in ("piece", "visual"):
                    g2.set("mesh", mesh_ren[_get(g, "mesh")])
                ren[name] = uname
                owner[uname] = k
                kind[uname] = kd
                (peds if kd == "pedestal" else pieces if kd in ("piece", "visual") else supports).append(g2)
        seq: list[str] = []
        for p in _find(root, "contact").findall("pair"):
            a1, a2 = _get(p, "geom1"), _get(p, "geom2")
            g1, g2n = ren.get(a1, a1), ren.get(a2, a2)
            key = _pair_key(g1, g2n)
            pe = copy.deepcopy(p)
            pe.set("geom1", g1)
            pe.set("geom2", g2n)
            specific = g1 in owner or g2n in owner
            if specific:
                pe.set("name", f"{p.get('name')}__u{k}")
            if key in pair_elems:
                if specific or pair_elems[key].attrib != pe.attrib:
                    raise ValueError(f"clip {k} ({scene}): pair {key} differs from an earlier clip's")
            else:
                pair_elems[key] = pe
            seq.append(key)
        pair_seqs.append(seq)
        rename.append(ren)
        dropped.append(tuple(drop))
    for i, g in enumerate(peds):
        world.insert(i, g)
    first_site = next((i for i, c in enumerate(list(obj)) if c.tag == "site"), len(list(obj)))
    for i, g in enumerate(pieces + supports):
        obj.insert(first_site + i, g)
    for key in merge_orders(pair_seqs):
        contact.append(pair_elems[key])
    out = Path(out_dir) / digest.hexdigest()[:12]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "union.xml"
    text = ET.tostring(union, encoding="unicode")
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return UnionSpec(xml_path=path, side=side, scenes=paths, rename=tuple(rename), dropped=tuple(dropped),
                     owner=owner, kind=kind)
