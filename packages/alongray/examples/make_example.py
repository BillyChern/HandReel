# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Write the quickstart's synthetic inputs, pred.npz and gt.npz, into the current directory (or the one given).

The scene is the README's minimal example: 30 frames of an object 0.45-0.75 m from the camera and a 21-joint hand
around it (joint 0 = wrist). The predicted object is the truth scaled by 1.1 about the camera centre (10% too far and
10% too big: a pure gauge move); the predicted hand is the true hand moved 10% farther along its wrist ray at its
true size. Seeded, so the files are the same on every run. No dataset is read.

    python examples/make_example.py [OUT_DIR]
    alongray pred.npz gt.npz --intrinsics 600,600,320,240
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(out: Path) -> None:
    rng = np.random.default_rng(0)
    T = 30
    gt_obj = np.column_stack([rng.normal(0, 0.05, T), rng.normal(0, 0.05, T), rng.uniform(0.45, 0.75, T)])
    gt_hand = gt_obj[:, None, :] + 0.03 * rng.standard_normal((T, 21, 3))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "gt.npz", object=gt_obj, hand=gt_hand)
    np.savez(out / "pred.npz", object=1.1 * gt_obj, hand=gt_hand + 0.1 * gt_hand[:, :1])
    print(f"wrote {out / 'pred.npz'} and {out / 'gt.npz'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
