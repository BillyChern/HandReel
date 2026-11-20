"""Inputs of the DexYCB policy references: per-object scenes, per-sequence GT hands, object geometry.

Layout. Both builders write it and `fastdo.policy.dexycb_refs` reads it.
- Scenes: `<scenes_root>/<object>/scene_<side>.json`, written by `run/pipeline/build_ycb_scenes.py`. It holds `object`,
  `side`, `scene_xml` (absolute), `pass` (the scene validation) and the build's provenance. The baseline's output tree
  for the object sits next to it.
- Hands: `<hands_root>/<sequence>/all_hand_meshes.npz` and `all_hand_meshes_validation.json`, written by
  `run/pipeline/build_dexycb_hands.py`. `<sequence>` is the DexYCB sequence directory name, e.g. `20200820_135250`.
  - The npz is in HaWoR's format, in the camera frame of one serial.
  - The json holds `conversion` (sequence, serial, hand side, MANO model) and `validation` (`pass`: joint error <= 2 mm
    and vertex-in-hand-label fraction >= 0.8 on 5 frames).

`scene_skip_reason` and `hands_skip_reason` turn a missing or failed record into the reason a clip is skipped.
`object_geometry` builds the SDF grid and surface samples of the kinematic reference from a YCB mesh. It first merges
the mesh's texture-seam duplicate vertices: DexYCB's `textured_simple.obj` is watertight only after that merge.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch import Tensor

from fastdo.solver.sdf import SdfGrid, mesh_sdf_grid

__all__ = ["HANDS_FILE", "HANDS_RECORD", "SIDES", "ObjectGeometry", "hands_npz_path", "hands_record_path",
           "hands_skip_reason", "load_hand_vertices", "load_object_mesh", "object_geometry", "read_hands_record",
           "read_scene_record", "scene_record_path", "scene_skip_reason", "sequence_name", "write_scene_record"]

SIDES = ("left", "right")
HANDS_FILE = "all_hand_meshes.npz"
HANDS_RECORD = "all_hand_meshes_validation.json"
_N_MANO_VERTS = 778


def _check_side(side: str | None) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return str(side)


def scene_record_path(scenes_root: str | Path, object_name: str, side: str) -> Path:
    return Path(scenes_root) / object_name / f"scene_{_check_side(side)}.json"


def write_scene_record(scenes_root: str | Path, record: Mapping[str, Any]) -> Path:
    """Write `record` (needs `object`, `side`, `scene_xml`, `pass`) to its `scene_record_path`; returns the path.

    Raises:
        ValueError: If a required key is missing.
    """
    missing = [k for k in ("object", "side", "scene_xml", "pass") if k not in record]
    if missing:
        raise ValueError(f"scene record lacks {missing}")
    path = scene_record_path(scenes_root, str(record["object"]), str(record["side"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(record), indent=1))
    os.replace(tmp, path)
    return path


def read_scene_record(scenes_root: str | Path, object_name: str, side: str) -> dict[str, Any] | None:
    path = scene_record_path(scenes_root, object_name, side)
    return json.loads(path.read_text()) if path.is_file() else None


def scene_skip_reason(record: Mapping[str, Any] | None, object_name: str, side: str) -> str | None:
    """Why a clip of `object_name` held by the `side` hand cannot use its scene record, or None."""
    if record is None:
        return f"no scene for {object_name} ({side} hand)"
    if not record.get("pass"):
        reasons = (record.get("validation") or {}).get("reasons") or record.get("error") or "unknown"
        return f"scene failed validation for {object_name} ({side} hand): {reasons}"
    return None


def sequence_name(sequence: str) -> str:
    """The DexYCB sequence directory name of `"<subject>/<sequence>[/<serial>]"`."""
    parts = sequence.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"expected <subject>/<sequence>[/<serial>], got {sequence!r}")
    return parts[1]


def hands_npz_path(hands_root: str | Path, sequence: str) -> Path:
    return Path(hands_root) / sequence_name(sequence) / HANDS_FILE


def hands_record_path(hands_root: str | Path, sequence: str) -> Path:
    return Path(hands_root) / sequence_name(sequence) / HANDS_RECORD


def read_hands_record(hands_root: str | Path, sequence: str) -> dict[str, Any] | None:
    path = hands_record_path(hands_root, sequence)
    return json.loads(path.read_text()) if path.is_file() else None


def hands_skip_reason(hands_root: str | Path, clip_id: str, side: str) -> str | None:
    """Why the GT hands of `"<subject>/<sequence>/<serial>"` cannot serve a `side`-hand clip, or None."""
    sequence, serial = "/".join(clip_id.split("/")[:2]), clip_id.split("/")[-1]
    record = read_hands_record(hands_root, sequence)
    if record is None or not hands_npz_path(hands_root, sequence).is_file():
        return f"hands not built for {sequence}"
    conversion, validation = record.get("conversion", {}), record.get("validation", {})
    if not validation.get("pass"):
        return (f"hands failed validation for {sequence}: joint error {validation.get('joint_err_mm_mean_5f')} mm, "
                f"label fraction {validation.get('frac_all')}")
    if str(conversion.get("serial")) != serial:
        return f"hands of {sequence} are in the frame of camera serial {conversion.get('serial')}, the clip's is {serial}"
    if conversion.get("hand_side") != side:
        return f"hands of {sequence} hold the {conversion.get('hand_side')} hand, the clip's hand side is {side}"
    return None


def load_hand_vertices(hands_root: str | Path, sequence: str, side: str, valid: np.ndarray) -> Tensor:
    """`(T,778,3)` float32 camera-frame `{side}_vertices` of the sequence's hands npz, one row per record row.

    Raises:
        ValueError: If the rows or the valid flags differ from the record's hand track `valid (T,)`.
    """
    flags = np.asarray(valid, dtype=bool).reshape(-1)
    with np.load(hands_npz_path(hands_root, sequence)) as npz:
        verts = np.asarray(npz[f"{_check_side(side)}_vertices"], dtype=np.float32)
        npz_valid = np.asarray(npz[f"{side}_valid"], dtype=bool)
    if verts.shape != (flags.shape[0], _N_MANO_VERTS, 3):
        raise ValueError(f"{sequence}: {side}_vertices {verts.shape} for {flags.shape[0]} record rows")
    if not np.array_equal(npz_valid, flags):
        raise ValueError(f"{sequence}: the hands npz marks rows valid that the record's {side} hand track does not")
    return torch.from_numpy(verts)


@dataclass(frozen=True)
class ObjectGeometry:
    """Canonical (metre) object geometry of the kinematic reference: SDF grid, surface samples and their normals."""

    sdf: SdfGrid
    surface: Tensor
    normals: Tensor
    n_vertices: int
    watertight: bool


def load_object_mesh(path: str | Path) -> trimesh.Trimesh:
    """A mesh file as one `Trimesh` with texture-seam duplicate vertices merged (module docstring)."""
    mesh = trimesh.load(path, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def object_geometry(mesh_path: str | Path, resolution: int = 64, n_surface: int = 1024, seed: int = 0) -> ObjectGeometry:
    """`mesh_sdf_grid` at `resolution` and `n_surface` area-weighted surface samples (`default_rng(seed)`) with face
    normals, as `tests/retarget/whisking_data.build_whisk_sdf_and_surface` builds them for the whisk."""
    mesh = load_object_mesh(mesh_path)
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=resolution)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_surface, seed=np.random.default_rng(seed))
    return ObjectGeometry(sdf=sdf, surface=torch.as_tensor(np.asarray(points), dtype=torch.float64),
                          normals=torch.as_tensor(np.asarray(mesh.face_normals[face_idx]), dtype=torch.float64),
                          n_vertices=len(mesh.vertices), watertight=bool(mesh.is_watertight))
