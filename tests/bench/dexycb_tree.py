"""A DexYCB-shaped tree for the benchmark tests (formerly `test_bench_clips._fake_dexycb`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SERIAL = "836212060125"
SEQ = f"20200820-subject-03/20200820_135508/{SERIAL}"


def write_fake_dexycb(root: Path, n: int = 5) -> str:
    """A DexYCB-shaped tree: one sequence, one camera, `n` colour frames and label maps (object id 5 grasped)."""
    intrinsics = root / "calibration" / "intrinsics"
    intrinsics.mkdir(parents=True)
    (intrinsics / f"{SERIAL}_640x480.yml").write_text(
        yaml.safe_dump({"color": {"fx": 600.0, "fy": 610.0, "ppx": 320.5, "ppy": 240.5}}))
    camera = root / SEQ
    camera.mkdir(parents=True)
    (camera.parent / "meta.yml").write_text(
        yaml.safe_dump({"serials": [SERIAL], "num_frames": n, "ycb_ids": [2, 5, 9], "ycb_grasp_ind": 1}))
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)).save(camera / f"color_{i:06d}.jpg")
        seg = np.zeros((12, 16), np.uint8)
        seg[2:6, 3:9] = 5 if i == 0 else 9          # only the keyframe's labels hold the grasped object
        seg[8:10, 0:4] = 255
        np.savez(camera / f"labels_{i:06d}.npz", seg=seg)
    return SEQ
