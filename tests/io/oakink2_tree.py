"""A tiny fake OakInk2 root (`fastdo.io.datasets.oakink2` layout) for unit tests: no real data is read."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

SEQ = "scene_09__Z001++seq__feedfacefeedfacefeed__2023-01-01-00-00-00"
SERIALS = {"egocentric": "900000000002", "allocentric_top": "900000000001"}
# image i sits at mocap id 4 i + 2 (the real ~30 fps images of a 120 Hz mocap, mean step 4)
N_IMAGES = 700
HW = (12, 16)
PROGRAM = {
    "(None, (100, 1300))": {"primitive": "grasp", "obj_list": ["O01@0001@00001"], "interaction_mode": "rh_main",
                            "primitive_lh": None, "primitive_rh": "grasp", "obj_list_lh": None,
                            "obj_list_rh": ["O01@0001@00001"]},
    "((1250, 2500), None)": {"primitive": "place_onto", "obj_list": ["C12001"], "interaction_mode": "lh_main",
                             "primitive_lh": "place_onto", "primitive_rh": None, "obj_list_lh": ["C12001"],
                             "obj_list_rh": None},
}


def mocap_id(i: int) -> int:
    return 4 * i + 2


def write_tree(root: Path, n_images: int = N_IMAGES, seq: str = SEQ, with_anno: bool = True) -> Path:
    """Write the fake root and return it. Image i of view v is a constant colour (v index, i % 256, 0)."""
    root = Path(root)
    seq_dir = root / "data" / seq
    for v, (view, serial) in enumerate(SERIALS.items()):
        d = seq_dir / serial
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_images):
            img = np.zeros((*HW, 3), dtype=np.uint8)
            img[..., 0], img[..., 1] = v, i % 256
            Image.fromarray(img).save(d / f"{mocap_id(i):06d}.png")
    views, serials = ",".join(SERIALS), ",".join(SERIALS.values())
    (seq_dir / ".subset_info").write_text(f"views={views} serials={serials} png_counts={{}} "
                                          f"frame_id_list={n_images} tar_sha256=0\n")
    prog = root / "program" / "program_info"
    prog.mkdir(parents=True, exist_ok=True)
    (prog / f"{seq}.json").write_text(json.dumps(PROGRAM))
    for obj, name in (("O01@0001@00001", "mug"), ("C12001", "bowl")):
        mdir = root / "object_raw" / "align_ds" / obj
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "scan.ply").write_text("ply\n")
    (root / "object_repair" / "align_ds" / "C12001").mkdir(parents=True, exist_ok=True)
    (root / "object_repair" / "align_ds" / "C12001" / "model.obj").write_text("v 0 0 0\n")
    (root / "object_raw" / "obj_desc.json").write_text(json.dumps(
        {"O01@0001@00001": {"obj_id": "O01@0001@00001", "obj_name": "mug"}}))
    if with_anno:
        write_anno(root, seq, n_images)
    return root


def write_anno(root: Path, seq: str = SEQ, n_images: int = N_IMAGES) -> None:
    """A numpy-only annotation pickle: a static allocentric camera, a moving egocentric one, one object."""
    ids = [mocap_id(i) for i in range(n_images)]
    extr_allo = np.eye(4)
    extr_allo[2, 3] = 1.0
    extr_ego = {}
    for fid in ids:
        e = np.eye(4)
        e[0, 3], e[2, 3] = 0.001 * fid, 0.5
        extr_ego[fid] = e
    obj = {}
    for fid in range(ids[-1] + 1):
        t = np.eye(4)
        t[:3, 3] = (0.1, 0.0, 0.2)
        obj[fid] = t
    obj[ids[3]] = np.full((4, 4), np.nan)
    K = np.array([[600.0, 0, 8], [0, 600.0, 6], [0, 0, 1]])
    anno = {"cam_def": {s: v for v, s in SERIALS.items()}, "cam_selection": list(SERIALS), "frame_id_list": ids,
            "cam_intr": {v: dict.fromkeys(ids, K) for v in SERIALS},
            "cam_extr": {"allocentric_top": dict.fromkeys(ids, extr_allo), "egocentric": extr_ego},
            "obj_list": ["O01@0001@00001"], "obj_transf": {"O01@0001@00001": obj}}
    path = root / "anno_preview" / f"{seq}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(anno))
