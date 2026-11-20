"""MuJoCo-world-frame inputs for `fastdo.retarget.kinematic`, read from the baseline's own npz files.

Frame convention (revised after measurement -- see
"Frame source: a correction" below). The robot's own object qpos block (`q[28:35]`) and its
wrist-prior target must be expressed in whatever frame the compiled `scene.xml`/MuJoCo scene itself
lives in, since `RobotScene.point_kinematics`/`mujoco.mj_jac` operate directly in that frame and
the resulting `qpos` goes into the baseline's own `optimize_physics.py` on the *same* scene,
unchanged. This module reads the baseline's own Stage-1 `trajectory_keypoints.npz`, whose
`qpos_obj_right`/`qpos_wrist_right` arrays are already expressed in that target (MuJoCo world) frame
by construction: they are exactly what the baseline's own `solve_ik.py` feeds as `mink` IK site
targets. It does *not* attempt to derive that transform from scratch (`process_dataset.py`'s own
per-clip `world_offset` search over the object mesh's and both hands' vertices' min-z across the
whole trajectory) -- see the correction below for why that search turns out to be unnecessary here.

**Frame source: a correction.** The original plan was to pair `qpos_obj_right`/
`qpos_wrist_right` (world frame) with `fastdo.teacher.layout_reader.read_baseline_layout`'s poses and
`all_hand_meshes.npz`'s vertices (a *different*, camera-frame reconstruction) for the human contact
measure, on the reasoning that `human_contact_measure`'s math only needs its two inputs to share a
frame with each other, not with the robot. Measured on the whisking demo, this produced a severe
regression (thumb clearance ~30mm, worse than the baseline's own kinematic-only solve) -- because
`read_baseline_layout`'s poses and `qpos_obj_right` are not, in fact, the same object trajectory
viewed from two fixed frames: their rotation "difference" at a few sample frames was 62-118 degrees
and *not consistent frame to frame* (verified directly, not assumed), meaning they are independent
reconstructions that disagree on the object's actual orientation, not a simple recalibration apart.
Pairing HaWoR vertices with `read_baseline_layout`'s rotation while placing the robot at
`qpos_obj_right`'s (different) rotation scrambles the human/robot correspondence the contact-
transport term depends on.

The fix: `trajectory_keypoints.npz` *also* carries `mano_verts_right` `(T,778,3)` -- the same MANO
vertices `all_hand_meshes.npz` has, but already shifted into the exact same world frame as
`qpos_obj_right` (`process_dataset.py` computes `mano_verts_right = right_vertices - world_offset`,
the identical `world_offset` subtracted from `qpos_obj`/`qpos_wrist`). Using `mano_verts_right`
instead of `all_hand_meshes.npz`'s vertices makes the human contact measure and the robot's object
placement mutually consistent *by construction*, with no `world_offset` derivation needed at all
(it is already baked into every array this module reads). Verified directly: `human_contact_measure`'s
canonical mapping applied to (`mano_verts_right`, `qpos_obj_right`) gives physically sensible
per-frame signed distances (small positive values on approach, negative -- penetrating -- values at
a known real contact frame), unlike the mismatched pairing above.

A general (non-demo) retargeting path that needs to handle an arbitrary new clip without a
pre-existing `trajectory_keypoints.npz` would still need to reproduce `process_dataset.py`'s
`world_offset` derivation (or an equivalent) to place a *different* hand-vertex source in this same
frame -- out of scope here (C4/engine scope).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastdo.geometry.lie import quat_wxyz_to_matrix

__all__ = ["baseline_world_inputs"]


def _qpos7_to_pose4x4(qpos7: np.ndarray) -> Tensor:
    """`(T,7)` `[pos(3), quat_wxyz(4)]` -> `(T,4,4)` SE(3) poses."""
    q = torch.as_tensor(qpos7, dtype=torch.float64)
    pose = torch.zeros(q.shape[0], 4, 4, dtype=torch.float64)
    pose[:, :3, :3] = quat_wxyz_to_matrix(q[:, 3:7])
    pose[:, :3, 3] = q[:, :3]
    pose[:, 3, 3] = 1.0
    return pose


def baseline_world_inputs(keypoints_npz_path: str | Path, trajectory_npz_path: str | Path,
                         ) -> tuple[Tensor, Tensor, Tensor, float]:
    """Object/wrist/hand-vertex inputs in the baseline's MuJoCo world frame, plus its clip frequency.

    Args:
        keypoints_npz_path: The baseline's Stage-1 `trajectory_keypoints.npz`: `qpos_obj_right`,
            `qpos_wrist_right` (both `(T,7)` `[pos(3), quat_wxyz(4)]`), and `mano_verts_right`
            (`(T,778,3)`), all already in the MuJoCo world frame and mutually consistent (see module
            docstring).
        trajectory_npz_path: The baseline's own Stage-4 `trajectory_kinematic.npz`, whose scalar
            `frequency` field this function reads directly: the
            clip's actual frequency (50 Hz for the whisking demo) must come from the shipped data,
            not a hardcoded assumption that could silently drift from what the demo actually used.

    Returns:
        `(obj_poses_world (T,4,4), wrist_poses_world (T,4,4), hand_verts_world (T,778,3), frequency)`.
        `T` = however many frames `keypoints_npz_path` has (138 for the whisking demo -- *not*
        pre-aligned to the baseline's own 135-frame `trajectory_kinematic.npz`; see that file's own
        frame-offset note and `tests/retarget/test_baseline_io.py`'s offset-assertion test for the
        relationship between the two frame counts).
    """
    with np.load(Path(keypoints_npz_path)) as data:
        qpos_obj = np.asarray(data["qpos_obj_right"], dtype=np.float64)
        qpos_wrist = np.asarray(data["qpos_wrist_right"], dtype=np.float64)
        hand_verts_world = torch.as_tensor(np.asarray(data["mano_verts_right"], dtype=np.float64))
    with np.load(Path(trajectory_npz_path)) as data:
        frequency = float(np.asarray(data["frequency"]))
    return _qpos7_to_pose4x4(qpos_obj), _qpos7_to_pose4x4(qpos_wrist), hand_verts_world, frequency
