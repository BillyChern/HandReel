from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastdo.io.datasets.dexycb_frames import DexYCBFrames


def _write_seq(root: Path, n: int) -> None:
    d = root / "20200709-subject-01" / "20200709_141754" / "836212060125"
    d.mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.zeros((8, 10, 3), np.uint8)).save(d / f"color_{i:06d}.jpg")
        depth = np.zeros((8, 10), np.uint16); depth[2:6, 3:8] = 500          # 0.5 m block
        Image.fromarray(depth).save(d / f"aligned_depth_to_color_{i:06d}.png")
        seg = np.zeros((8, 10), np.uint8); seg[2:4, 3:8] = 5; seg[4:6, 3:8] = 255   # object id 5, hand 255
        np.savez(d / f"labels_{i:06d}.npz", seg=seg)


def test_frame_paths_and_observations(tmp_path: Path) -> None:
    _write_seq(tmp_path, 6)
    frames = DexYCBFrames(tmp_path)
    seq = "20200709-subject-01/20200709_141754/836212060125"
    paths = frames.frame_paths(seq)
    assert len(paths) == 6 and paths[0].color.name == "color_000000.jpg"
    K = torch.tensor([[10.0, 0, 5.0], [0, 10.0, 4.0], [0, 0, 1.0]])
    masks, hands, pmaps = frames.load_observations(seq, K, object_id=5, stride=3)
    assert masks.shape == (2, 8, 10) and hands.shape == (2, 8, 10) and pmaps.shape == (2, 8, 10, 3)
    assert masks[0, 2, 4] and not masks[0, 4, 4] and hands[0, 4, 4]
    assert abs(float(pmaps[0, 2, 4, 2]) - 0.5) < 1e-6 and torch.isnan(pmaps[0, 0, 0, 2])
    assert abs(float(pmaps[0, 2, 4, 0]) - ((4 - 5.0) * 0.5 / 10.0)) < 1e-6
