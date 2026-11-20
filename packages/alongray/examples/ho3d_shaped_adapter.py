# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""An adapter from HO3D-SHAPED annotations to alongray inputs, on generated synthetic data only.

No dataset file, model file or mesh is read or written: every array below is drawn from a seeded generator, in the
SHAPE of a per-frame hand-object annotation (an object rotation and translation per frame, 21 hand joints, a camera
matrix, frame numbers taken from file names). Replace `synthetic_sequence` with your own loader; check every
convention below against the documentation of the dataset copy you use, since nothing here verifies it.

The five conventions the adapter makes explicit:

1. Object pose: a (4, 4) OBJECT-TO-CAMERA transform per frame, `x_cam = R x_obj + t`, bottom row [0, 0, 0, 1].
2. Same physical object point: the reference point is `R o + t` for ONE model-frame point `o` (here the mesh
   centroid), applied to prediction and truth alike. A template origin on one side and a centroid on the other is
   scored as error (see `origin_mismatch` in the output).
3. Metres: inputs in millimetres are converted (`MM`); alongray warns when the RMS range exceeds 10 m.
4. Wrist root: the hand is (T, 21, 3) with the wrist at joint 0 (`hand_root=0`), in the same joint order on both
   sides; reorder first if your annotation puts the wrist elsewhere.
5. Frame ids: integer ids, one per row, from the file names (`0000.pkl` -> 0); prediction rows arrive in any order
   and are joined on the ids (identical id sets required). Camera axes: this synthetic annotation, like some
   real ones, uses OpenGL-style axes (-z forward); `GL_TO_CV` flips y and z to OpenCV (+z forward), which the
   reprojection contrast needs. `e_par` itself does not depend on the axes, provided both sides use the same ones.

    python examples/ho3d_shaped_adapter.py [OUT_DIR]
    alongray OUT_DIR/pred.npz OUT_DIR/gt.npz --require-frame-ids
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import alongray

MM = 1e-3                                       # millimetres -> metres
GL_TO_CV = np.diag([1.0, -1.0, -1.0])           # OpenGL-style camera axes (y up, -z forward) -> OpenCV
SCALE = 1.05                                    # the synthetic prediction: a pure gauge move, 5% too far and big


def _rodrigues(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th == 0.0:
        return np.eye(3)
    k = v / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * kx + (1.0 - np.cos(th)) * kx @ kx


def synthetic_sequence(n: int = 16, seed: int = 0) -> dict:
    """One synthetic sequence in the annotation's own layout (OpenGL-style axes, metres, axis-angle rotations)."""
    rng = np.random.default_rng(seed)
    centroid_obj = np.array([0.004, -0.012, 0.031])             # mesh centroid in the model frame, metres
    frames = {}
    for f in range(n):
        t_cv = np.array([rng.normal(0, 0.04), rng.normal(0, 0.04), rng.uniform(0.45, 0.7)])
        hand_cv = t_cv + 0.03 * rng.standard_normal((21, 3))   # joint 0 = wrist
        frames[f"{f:04d}.pkl"] = {"objRot": rng.normal(0, 0.6, 3), "objTrans": GL_TO_CV @ t_cv,
                                  "handJoints3D": hand_cv @ GL_TO_CV.T}
    return {"frames": frames, "camMat": np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]]),
            "centroid_obj": centroid_obj}


def object_to_camera(rot_axis_angle: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(4, 4) object-to-camera pose in OpenCV axes from a rotation and translation given in OpenGL-style axes."""
    pose = np.eye(4)
    pose[:3, :3] = GL_TO_CV @ _rodrigues(np.asarray(rot_axis_angle, np.float64))
    pose[:3, 3] = GL_TO_CV @ np.asarray(trans, np.float64)
    return pose


def reference_pose(pose: np.ndarray, point_obj: np.ndarray) -> np.ndarray:
    """The same pose re-anchored at a model-frame point: its translation becomes ``R point_obj + t``."""
    out = pose.copy()
    out[:3, 3] = pose[:3, :3] @ point_obj + pose[:3, 3]
    return out


def to_arrays(seq: dict, point_obj: np.ndarray) -> dict[str, np.ndarray]:
    """``object`` (T, 4, 4), ``hand`` (T, 21, 3), ``frame_ids`` (T,), ``K``, OpenCV axes, metres."""
    names = sorted(seq["frames"])
    ids = np.array([int(Path(n).stem) for n in names], dtype=np.int64)
    obj = np.stack([reference_pose(object_to_camera(seq["frames"][n]["objRot"], seq["frames"][n]["objTrans"]),
                                   point_obj) for n in names])
    hand = np.stack([seq["frames"][n]["handJoints3D"] @ GL_TO_CV.T for n in names])
    return {"object": obj, "hand": hand, "frame_ids": ids, "K": seq["camMat"]}


def synthetic_prediction(gt: dict[str, np.ndarray], seed: int = 1) -> dict[str, np.ndarray]:
    """A method's output: the truth moved by the gauge SCALE, written in millimetres, rows in shuffled order."""
    order = np.random.default_rng(seed).permutation(len(gt["frame_ids"]))
    obj = gt["object"].copy()
    obj[:, :3, 3] *= SCALE / MM                                  # translations in millimetres
    return {"object_mm": obj[order], "hand_mm": SCALE * gt["hand"][order] / MM, "frame_ids": gt["frame_ids"][order]}


def main(out: Path) -> dict:
    seq = synthetic_sequence()
    gt = to_arrays(seq, seq["centroid_obj"])
    raw = synthetic_prediction(gt)
    pred_obj = raw["object_mm"].copy()
    pred_obj[:, :3, 3] *= MM                                     # convention 3: metres
    pred = {"object": pred_obj, "hand": raw["hand_mm"] * MM, "frame_ids": raw["frame_ids"]}
    res = alongray.score(pred["object"], gt["object"], pred["hand"], gt["hand"], gt["K"], hand_root=0,
                         pred_ids=pred["frame_ids"], gt_ids=gt["frame_ids"])
    # convention 2 broken on purpose: the truth anchored at the template origin, the prediction at the centroid
    gt_origin = to_arrays(seq, np.zeros(3))
    wrong = alongray.score(pred["object"], gt_origin["object"], pred_ids=pred["frame_ids"],
                           gt_ids=gt_origin["frame_ids"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "gt.npz", **gt)
    np.savez(out / "pred.npz", **pred)
    summary = {"object": res["object"].as_dict(), "hand": res["hand"].as_dict(),
               "origin_mismatch": {"off_orbit_part": wrong["object"].off_orbit_part, "lam": wrong["object"].lam}}
    print(json.dumps({b: {k: summary[b][k] for k in ("e_par", "gauge_part", "off_orbit_part", "lam")}
                      for b in ("object", "hand")}, indent=1))
    print(f"origin mismatch adds an off-orbit term of {wrong['object'].off_orbit_part * 1e3:.2f} mm")
    print(f"wrote {out / 'pred.npz'} and {out / 'gt.npz'} (synthetic, with frame_ids)")
    return summary


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ho3d_shaped_out"))
