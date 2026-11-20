"""A synthetic HOI4D tree for the HOI4D reader and bench tests (no real HOI4D file is read)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

SEQ = "ZY20210800001/H1/C2/N01/S10/s01/T1"          # a Mug: rigid, CAD rigid/Mug/001.obj
SEQ_PADDED = "ZY20210800002/H2/C2/N01/S11/s02/T2"   # the same instance, 5-digit objpose names
SEQ_LAPTOP = "ZY20210800001/H1/C3/N01/S12/s01/T1"   # articulated: never a benchmark sequence
K = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]])
HW = (240, 320)
N_FRAMES = 12
EULER = [0.3, -0.2, 0.5]


def gt_pose(t: int) -> np.ndarray:
    """The synthetic GT object-to-camera pose of frame t: XYZ Euler EULER, centre moving along x."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("XYZ", EULER).as_matrix()
    pose[:3, 3] = [0.01 * t - 0.05, 0.0, 0.6]
    return pose


def box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.08, 0.1, 0.06))


def write_tree(root: Path, n_frames: int = N_FRAMES, invalid: tuple[int, ...] = (3,), missing: tuple[int, ...] = (5,),
               render_masks: bool = True) -> Path:
    """Writes the tree; frames in `invalid` have isEffective 0, frames in `missing` have no objpose file."""
    from fastdo.io.datasets.hoi4d_gt_check import silhouette

    (root / "meta").mkdir(parents=True)
    (root / "meta" / "release.txt").write_bytes(f"{SEQ}\r\n{SEQ_PADDED}\r\n{SEQ_LAPTOP}\r\n{SEQ}\r\n".encode())
    for cam in ("ZY20210800001", "ZY20210800002"):
        (root / "camera_params" / cam).mkdir(parents=True)
        np.save(root / "camera_params" / cam / "intrin.npy", K)
    cad = root / "HOI4D_CAD_Model_for_release" / "rigid" / "Mug"
    cad.mkdir(parents=True)
    mesh = box()
    mesh.export(cad / "001.obj")
    for seq, padded in ((SEQ, False), (SEQ_PADDED, True)):
        ann = root / "HOI4D_annotations" / seq
        (ann / "objpose").mkdir(parents=True)
        (ann / "2Dseg" / ("shift_mask" if padded else "mask")).mkdir(parents=True)
        (ann / "action").mkdir(parents=True)
        (ann / "action" / "color.json").write_text(json.dumps({"events": [
            {"event": "rest", "startTime": 0.0, "endTime": 0.2}, {"event": "Grasp", "startTime": 0.4, "endTime": 0.6}],
            "info": {"duration": n_frames / 15}}))
        for t in range(n_frames):
            if t in missing:
                continue
            pose = gt_pose(t)
            angles = Rotation.from_matrix(pose[:3, :3]).as_euler("XYZ")
            item = {"center": dict(zip("xyz", pose[:3, 3].tolist(), strict=True)),
                    "rotation": dict(zip("xyz", angles.tolist(), strict=True)),
                    "dimensions": {"length": 0.1, "width": 0.08, "height": 0.06}, "label": "mug"}
            name = f"{t:05d}.json" if padded else f"{t}.json"
            (ann / "objpose" / name).write_text(json.dumps(
                {"isEffective": 0 if t in invalid else 1, "dataList": [item]}))
            lab = np.zeros((*HW, 3), np.uint8)
            if render_masks:
                sil = silhouette(np.asarray(mesh.vertices), np.asarray(mesh.faces), pose, K, HW)
                lab[sil] = (128, 0, 0)
                lab[:30, :40] = (0, 128, 0)              # a "hand" that does not touch the object
            Image.fromarray(lab).save(ann / "2Dseg" / ("shift_mask" if padded else "mask") / f"{t:05d}.png")
    return root
