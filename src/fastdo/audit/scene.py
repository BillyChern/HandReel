# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""The MuJoCo scene side of the audit: geom masks, the hand-object contact census and contact removal.

Contact removal has three constructions:
- `remove_pairs` on the scene XML: delete every `<pair>` element whose
  geoms are a hand geom and an object geom. This removes all contact with the object only when no other path is
  left: default (contype / conaffinity) collisions, or pairs with dynamic geoms outside the hand mask;
  `contact_paths` counts them all. If the executor edits the compiled model, pass it as `executor_model`.
- `nullaudit.mujoco_adapter.MujocoExecutor(absent_mode="pairs")` with the hand moved away (CPU).
- `MujocoExecutor(absent_mode="displace")`: every hand geom moved 1 km.
Whichever is used, the N0 ledger of the rollout decides whether the run was a null: any post-release force on the
object from the hand mask or from any other dynamic geom makes it `invalid`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputError

__all__ = ["classify_pairs", "compile_without_pairs", "contact_paths", "free_object_geom", "geom_mask",
           "hand_object_pair_names", "ka_h", "load_model", "remove_pairs", "start_pedestal"]

_PAIR_RE = re.compile(r'<pair\b[^>]*\bname="([^"]+)"[^>]*/>')
_DIR_RE = re.compile(r'\b(meshdir|texturedir|assetdir)="([^"]*)"')
_INCLUDE_RE = re.compile(r'(<include\b[^>]*\bfile=")([^"]+)(")')


def load_model(path: str | Path) -> Any:
    """An MjModel from a scene `.xml` or a compiled `.mjb` file. A missing file or one MuJoCo cannot load raises
    `InputError`."""
    import mujoco

    p = Path(path)
    if not p.is_file():
        raise InputError(f"scene file {p} does not exist")
    try:
        if p.suffix == ".mjb":
            return mujoco.MjModel.from_binary_path(str(p))
        return mujoco.MjModel.from_xml_path(str(p))
    except ValueError as e:                             # MuJoCo's compile and load errors
        raise InputError(f"{p}: MuJoCo cannot load it: {e}") from e


def geom_mask(model: Any, pattern: str) -> np.ndarray:
    """(ngeom,) bool: geoms whose name contains a match of the regular expression `pattern` (the studies' hand rule
    is the substring "collision_hand_")."""
    import mujoco

    rx = re.compile(pattern)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    return np.array([bool(rx.search(n)) for n in names], bool)


def free_object_geom(model: Any) -> np.ndarray:
    """(ngeom,) bool: the geoms of the body that carries the scene's only free joint (the object)."""
    import mujoco

    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise InputError(f"expected exactly one free joint (the object), found {len(free)}")
    return np.asarray(model.geom_bodyid == model.jnt_bodyid[free[0]])


def classify_pairs(model: Any, hand_geom: np.ndarray | None = None,
                   object_geom: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Every `<pair>` of a compiled scene with its kinds, classified as the contact-removal study classified them:
    "object" = a geom of a `*_object` body (or of a body carrying an `object_geom` geom, so support markers on the
    object body count as object), "pedestal" = a geom or body whose name contains
    "pedestal", "world:<geom>" = a geom of body 0, "static:<geom>" = another body welded to the world, "hand" = a
    geom on a dynamic body (or in `hand_geom`). A third dynamic body is refused, not guessed: with `hand_geom`, a
    dynamic geom outside the mask and the object raises; without it, a scene with more than one free joint raises."""
    import mujoco
    from nullaudit.mujoco_adapter import dynamic_geoms

    G, B, P = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_PAIR
    m = model
    dyn = dynamic_geoms(m)
    obj_bodies = set() if object_geom is None else {int(b) for b in m.geom_bodyid[np.asarray(object_geom, bool)]}
    if hand_geom is None and sum(int(t == mujoco.mjtJoint.mjJNT_FREE) for t in m.jnt_type) > 1:
        raise InputError("more than one free joint: pass hand_geom and object_geom to say which body is which")

    def kind(g: int, pair: str) -> str:
        b = mujoco.mj_id2name(m, B, int(m.geom_bodyid[g])) or ""
        n = mujoco.mj_id2name(m, G, g) or ""
        if (object_geom is not None and int(m.geom_bodyid[g]) in obj_bodies) or (object_geom is None
                                                                                 and b.endswith("_object")):
            return "object"
        if "pedestal" in n or "pedestal" in b:
            return "pedestal"
        if int(m.geom_bodyid[g]) == 0:
            return "world:" + n
        if not dyn[g]:
            return "static:" + n
        if hand_geom is not None and not hand_geom[g]:
            raise InputError(f"pair {pair!r}: geom {n or g} is on a dynamic body outside the hand mask and the "
                             "object; a third dynamic body is not classified")
        return "hand"

    out = []
    for p in range(m.npair):
        g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
        name = mujoco.mj_id2name(m, P, p) or ""
        out.append({"name": name, "kinds": tuple(sorted((kind(g1, name), kind(g2, name)))),
                    "geoms": (mujoco.mj_id2name(m, G, g1) or "", mujoco.mj_id2name(m, G, g2) or "")})
    return out


def hand_object_pair_names(pairs: Sequence[dict[str, Any]]) -> list[str]:
    """Names of the hand|object pairs in a `classify_pairs` list."""
    return [p["name"] for p in pairs if p["kinds"] == ("hand", "object")]


def start_pedestal(pairs: Sequence[dict[str, Any]]) -> bool:
    """Start-pedestal rule: an object|pedestal pair whose pedestal geom is named `*pedestal_start*`."""
    ped = {g for p in pairs if p["kinds"] == ("object", "pedestal") for g in p["geoms"] if "pedestal" in g}
    return any("pedestal_start" in g for g in ped)


def contact_paths(model: Any, hand_geom: np.ndarray, object_geom: np.ndarray) -> dict[str, int]:
    """How contact with the object can arise from the N0 ledger's geoms (the hand mask plus every other dynamic
    geom, `nullaudit.mujoco_adapter.contact_ledger_geoms`): explicit `<pair>` elements, and geom pairs that
    MuJoCo's default contype / conaffinity filter lets collide; `outside_mask_*` counts the part outside the hand
    mask. Removing the hand|object pairs removes all contact only when every count is 0 afterwards (DO AS I
    DO's scenes: 0 on every clip)."""
    from nullaudit.mujoco_adapter import contact_ledger_geoms

    hand, obj = np.asarray(hand_geom, bool), np.asarray(object_geom, bool)
    led = contact_ledger_geoms(model, hand, obj)
    pairs = [(int(model.pair_geom1[p]), int(model.pair_geom2[p])) for p in range(model.npair)]
    other = [g2 if obj[g1] else g1 for g1, g2 in pairs if obj[g1] != obj[g2] and led[g2 if obj[g1] else g1]]
    ct, ca = model.geom_contype, model.geom_conaffinity
    dflt = [h for h in np.flatnonzero(led) for o in np.flatnonzero(obj) if (ct[h] & ca[o]) or (ct[o] & ca[h])]
    return {"explicit_pairs": len(other), "default_collision_pairs": len(dflt),
            "outside_mask_explicit_pairs": sum(1 for g in other if not hand[g]),
            "outside_mask_default_collision_pairs": sum(1 for g in dflt if not hand[g])}


def remove_pairs(xml_text: str, names: Sequence[str]) -> tuple[str, list[str]]:
    """Delete each single-line `<pair ... name="NAME" .../>` element. Returns (new text,
    removed lines). Raises if a name is empty, duplicated, missing or ambiguous, or if a pair line carries other
    content, so `ka_h` holds by construction."""
    lines = xml_text.splitlines(keepends=True)
    want = set(names)
    if len(want) != len(names) or "" in want:
        raise InputError("pair names must be unique and non-empty")
    keep, removed = [], []
    found: dict[str, int] = {}
    for ln in lines:
        m = _PAIR_RE.search(ln)
        if m and m.group(1) in want:
            if ln.strip() != m.group(0):
                raise InputError(f"pair line carries other content: {ln.strip()[:80]}")
            found[m.group(1)] = found.get(m.group(1), 0) + 1
            removed.append(ln)
            continue
        keep.append(ln)
    bad = sorted(n for n in want if found.get(n) != 1)
    if bad:
        raise InputError(f"{len(bad)} pair(s) not found exactly once, e.g. {bad[:3]}")
    return "".join(keep), removed


def ka_h(original: str, edited: str, removed: Sequence[str], census_count: int) -> dict[str, Any]:
    """KA-H: the edited scene equals the original minus exactly the removed pair lines, and
    their number equals the pair census."""
    rest = original.splitlines(keepends=True)
    for ln in removed:
        rest.remove(ln)
    same = "".join(rest) == edited
    return {"pass": bool(same and len(removed) == census_count), "removed": len(removed), "census": census_count,
            "text_equal_after_removal": same}


_ASSET_FILE_RE = re.compile(r'(<(mesh|hfield|skin|texture)\b[^>]*?\bfile=")([^"]+)(")')


def _absolute_paths(text: str, base: Path) -> str:
    """The XML with relative asset directories, includes and (where no asset directory applies) asset files made
    absolute, so that it compiles from a string."""
    def fix(p: str) -> str:
        return p if not p or Path(p).is_absolute() else str((base / p).resolve()) + ("/" if p.endswith("/") else "")

    dirs = {m.group(1) for m in _DIR_RE.finditer(text)}
    text = _DIR_RE.sub(lambda m: f'{m.group(1)}="{fix(m.group(2))}"', text)
    text = _INCLUDE_RE.sub(lambda m: m.group(1) + fix(m.group(2)) + m.group(3), text)

    def asset(m: re.Match[str]) -> str:
        own = "texturedir" if m.group(2) == "texture" else "meshdir"
        return m.group(0) if own in dirs or "assetdir" in dirs else m.group(1) + fix(m.group(3)) + m.group(4)

    return _ASSET_FILE_RE.sub(asset, text)


def _compile_edited(edited: str, scene: Path) -> Any:
    """Compile the edited XML from a string with absolute asset paths. If MuJoCo still cannot resolve a file, the
    XML is compiled from a PRIVATE temporary directory that mirrors the scene directory by symbolic links; nothing
    is written next to the scene."""
    import os
    import shutil
    import tempfile

    import mujoco

    try:
        return mujoco.MjModel.from_xml_string(_absolute_paths(edited, scene.parent))
    except ValueError:
        tmp = Path(tempfile.mkdtemp(prefix="fastdo_audit_"))
        try:
            for entry in scene.parent.iterdir():
                os.symlink(entry.resolve(), tmp / entry.name)
            (tmp / scene.name).unlink()
            (tmp / scene.name).write_text(edited)
            return mujoco.MjModel.from_xml_path(str(tmp / scene.name))
        finally:
            shutil.rmtree(tmp)


def compile_without_pairs(scene: str | Path, hand_geom: np.ndarray | None = None,
                          object_geom: np.ndarray | None = None,
                          executor_model: Any = None) -> tuple[Any, dict[str, Any]]:
    """Compile the scene XML with every hand|object `<pair>` deleted. Returns the model and a
    record (pairs removed, KA-H, remaining contact paths of the N0 ledger). Nothing is written next to the scene
    (`_compile_edited`).

    `executor_model`: the executor's model, when it is not the XML compiled as is. Its options, numeric arrays,
    statistics and the rows of every surviving pair are copied onto the result (`match.match_executor`), and the
    record lists what differed; a layout mismatch raises. `audit_run(removal="model")` refuses a removed model
    that differs from the executor's outside names and the removed pairs."""
    from .match import match_executor

    scene = Path(scene)
    model = load_model(scene)
    original = scene.read_text()
    pairs = classify_pairs(model, hand_geom, object_geom)
    names = hand_object_pair_names(pairs)
    edited, removed = remove_pairs(original, names)
    out = _compile_edited(edited, scene)
    matched = None if executor_model is None else match_executor(out, executor_model)
    after = None if hand_geom is None or object_geom is None else contact_paths(out, hand_geom, object_geom)
    rec = {"pairs_removed": len(removed), "ka_h": ka_h(original, edited, removed, len(names)),
           "start_pedestal": start_pedestal(pairs), "after": after, "matched_from_executor": matched}
    return out, rec
