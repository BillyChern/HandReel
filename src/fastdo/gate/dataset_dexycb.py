"""DexYCB readers of the gate dataset: id checks and exclusions before any I/O, and frame labels.

`check_dexycb_ids` is pure string work on the caller's id list: it runs before any DexYCB path is built, opened or
listed. Exclusions come from JSON files, such as the solver studies' evaluation sets
and the QC calibration ids: every DexYCB
sequence name (`YYYYMMDD_HHMMSS`) anywhere in such a file is refused. Sequence names are unique across DexYCB
subjects, so bare names and full `<subject>/<sequence>/<serial>` ids both match. Only camera `DEXYCB_SERIAL` is used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import trimesh

from fastdo.gate.labels import DexycbFrameLabels, dexycb_frame_labels
from fastdo.io.datasets.dexycb import load_dexycb_yaml, ycb_object_name

__all__ = ["DEXYCB_SERIAL", "check_dexycb_ids", "excluded_dexycb_sequences", "read_dexycb_frame_labels",
           "read_dexycb_id_file"]

DEXYCB_MESH_FILE = "textured_simple.obj"
DEXYCB_SERIAL = "836212060125"          # the gate's one DexYCB camera
_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")
_SEQUENCE_NAME = re.compile(r"(?<!\d)\d{8}_\d{6}(?!\d)")


def check_dexycb_ids(seq_ids: Sequence[str], allowed_subjects: Sequence[str],
                     excluded_sequences: Collection[str] = (), serial: str = DEXYCB_SERIAL) -> list[str]:
    """The ids, unchanged, after pure string checks: each is `<subject>/<sequence>/<serial>`, `<subject>` is allowed,
    `<sequence>` is not excluded, `<serial>` is `serial`, and no id repeats. Nothing on disk is touched.

    Raises:
        ValueError: Naming the first id that fails a check.
    """
    allowed = {str(subject) for subject in allowed_subjects}
    excluded = frozenset(str(name) for name in excluded_sequences)
    seen: set[str] = set()
    for raw in seq_ids:
        seq_id = str(raw)
        parts = seq_id.split("/")
        if len(parts) != 3 or any(not _SEGMENT.fullmatch(p) or p in {".", ".."} for p in parts):
            raise ValueError(f"DexYCB id {seq_id!r} is not <subject>/<sequence>/<serial>")
        if parts[0] not in allowed:
            raise ValueError(f"DexYCB id {seq_id!r}: subject {parts[0]!r} is not in gate_data.dexycb_allowed_subjects "
                             f"{sorted(allowed)}")
        if parts[1] in excluded:
            raise ValueError(f"DexYCB id {seq_id!r}: sequence {parts[1]} is excluded (a pre-registered evaluation or "
                             "calibration set listed in gate_data.dexycb_exclude_files)")
        if parts[2] != serial:
            raise ValueError(f"DexYCB id {seq_id!r}: camera serial {parts[2]!r} is not {serial}")
        if seq_id in seen:
            raise ValueError(f"DexYCB id {seq_id!r} is listed twice")
        seen.add(seq_id)
    return [str(seq_id) for seq_id in seq_ids]


def excluded_dexycb_sequences(paths: Sequence[Path]) -> frozenset[str]:
    """Every DexYCB sequence name (`YYYYMMDD_HHMMSS`) in the JSON files at `paths` (module docstring).

    Raises:
        FileNotFoundError: If a file is absent: a listed exclusion set is never silently dropped.
        ValueError: If a file is not JSON.
    """
    names: set[str] = set()
    for path in paths:
        text = Path(path).read_text()
        json.loads(text)
        names.update(_SEQUENCE_NAME.findall(text))
    return frozenset(names)


def read_dexycb_id_file(path: Path) -> list[str]:
    """The ids of a DexYCB id file: a JSON list of strings, or an object whose `"sequences"` is one.

    Raises:
        ValueError: On any other layout.
    """
    data = json.loads(Path(path).read_text())
    ids = data.get("sequences") if isinstance(data, dict) else data
    if not isinstance(ids, list) or not all(isinstance(seq_id, str) for seq_id in ids):
        raise ValueError(f"{path} must be a JSON list of DexYCB ids, or an object with a 'sequences' list")
    return list(ids)


def read_dexycb_frame_labels(root: Path, seq_id: str, n_points: int = 2048, seed: int = 0) -> DexycbFrameLabels:
    """`dexycb_frame_labels` of one camera sequence under `root` (`cfg.dexycb.root`).

    Reads `meta.yml` (`ycb_ids`, `ycb_grasp_ind`), samples `n_points` surface points of the grasped object's
    `models/<name>/textured_simple.obj` (`trimesh.sample.sample_surface(seed=seed)`), and reads `pose_y[ycb_grasp_ind]`,
    `joint_3d[0]` and `(seg == object_id).sum()` from every `labels_*.npz` (sorted) of the camera directory.

    Raises:
        FileNotFoundError: If the mesh or every label file is missing.
    """
    root = Path(root)
    camera = root / seq_id
    meta = load_dexycb_yaml(camera.parent / "meta.yml")
    grasp = int(meta["ycb_grasp_ind"])
    object_id = int(meta["ycb_ids"][grasp])
    mesh_path = root / "models" / ycb_object_name(object_id) / DEXYCB_MESH_FILE
    if not mesh_path.is_file():
        raise FileNotFoundError(f"{mesh_path} not found: the grasped object's mesh of {seq_id}")
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    points = np.asarray(trimesh.sample.sample_surface(mesh, int(n_points), seed=seed)[0], dtype=np.float64)
    files = sorted(camera.glob("labels_*.npz"))
    if not files:
        raise FileNotFoundError(f"no labels_*.npz under {camera}")
    joints, poses, pixels = [], [], []
    for file in files:
        with np.load(file, allow_pickle=False) as labels:
            poses.append(np.asarray(labels["pose_y"])[grasp])
            joints.append(np.asarray(labels["joint_3d"])[0])
            pixels.append(int((np.asarray(labels["seg"]) == object_id).sum()))
    return dexycb_frame_labels(np.stack(joints), points, np.stack(poses), np.asarray(pixels))
