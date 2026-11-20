"""Wrist targets of the kinematic reference's wrist prior.

`retarget_kinematic` pulls the robot wrist towards one SE(3) target per frame. Two conventions exist:

- `ik_site` (the default): the baseline's IK-site target, `process_dataset.py::_build_side_qpos`. It is the target
  the kinematic retargeting solved against and the physics stage validated
  (`trajectory_keypoints.npz` `qpos_wrist_{side}`).
  - Position: the wrist joint.
  - Orientation: MANO's global orient `R_mano(t)` times one fixed offset `R_off`, the rotation from the MANO wrist
    frame to the MCP geometric frame of `compute_wrist_rotation`. The MCP joints are children of MANO's root, so
    `R_mano(t)^-1 R_geom(t)` is the rest hand's geometric frame, constant up to fit noise. `R_off` is its average over
    rows: sign-aligned unit quaternions, top eigenvector of their scatter matrix (`_wrist_frame_offset`).
- `mcp`: `fastdo.retarget.kinematic_terms.human_wrist_poses`, the per-frame MCP geometric frame `R_geom(t)` itself.
  It has not passed physics; switching the default needs a C4 physics re-run with it.

Both targets are equivariant under one rigid change of frame applied to the joints and the global orient. `R_off` is
invariant, since `(R R_mano)^-1 (R R_geom) = R_mano^-1 R_geom`. So the retarget stage computes them in the camera
frame and moves them to the world frame afterwards.

Two differences from the baseline, both on inputs the baseline does not see:
- `R_off` averages only the rows marked valid. D3b fills invalid rows with copies of the nearest valid row, which
  would re-weight the average.
- A quaternion exactly orthogonal to the first row's keeps its sign instead of dropping out (`np.sign(0) = 0`).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from fastdo.retarget.kinematic_terms import human_wrist_poses

__all__ = ["DEFAULT_WRIST_TARGET", "WRIST_TARGETS", "ik_site_wrist_poses", "wrist_frame_offset", "wrist_target_poses"]

WRIST_TARGETS = ("ik_site", "mcp")
DEFAULT_WRIST_TARGET = "ik_site"
_DT = torch.float64


def _valid_rows(joints: Tensor, rotvec: Tensor, valid: np.ndarray | None) -> np.ndarray:
    """Indices of the valid rows, after the shape and finiteness checks (ValueError)."""
    t = int(joints.shape[0]) if joints.dim() == 3 else -1
    if joints.dim() != 3 or tuple(joints.shape[1:]) != (21, 3) or tuple(rotvec.shape) != (t, 3):
        raise ValueError(f"expected joints (T,21,3) and rotvec (T,3), got {tuple(joints.shape)} and "
                         f"{tuple(rotvec.shape)}")
    mask = np.ones(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if mask.shape[0] != t:
        raise ValueError(f"{mask.shape[0]} validity flags for {t} rows")
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError("no valid row to compute the IK-site wrist target from")
    index = torch.from_numpy(rows)
    for name, values in (("joints", joints), ("rotvec", rotvec)):
        finite = torch.isfinite(values[index].reshape(rows.size, -1)).all(dim=1).numpy()
        if not bool(finite.all()):
            raise ValueError(f"{name} is non-finite on valid rows {rows[~finite][:6].tolist()}")
    return rows


def wrist_frame_offset(joints: Tensor, rotvec: Tensor, is_right: bool = True,
                       valid: np.ndarray | None = None) -> np.ndarray:
    """`(3,3)` float64 `R_off`, the average over the valid rows of `R_mano^-1 R_geom` (module docstring).

    Args:
        joints: `(T,21,3)` OpenPose-order hand joints (0 wrist, 5 index MCP, 9 middle MCP, 13 ring MCP).
        rotvec: `(T,3)` MANO global orient axis-angle, in the joints' frame (`HandTrack.params[:, :3]`).
        is_right: Hand side (the index-ring MCP axis direction).
        valid: `(T,)` bool rows to average over; None means all.

    Raises:
        ValueError: On bad shapes, no valid row, or non-finite joints or global orient on a valid row.
    """
    index = torch.from_numpy(_valid_rows(joints, rotvec, valid))
    r_geom = human_wrist_poses(joints.to(_DT)[index], is_right=is_right)[:, :3, :3].numpy()
    r_mano = Rotation.from_rotvec(rotvec.to(_DT)[index].numpy())
    quat = (r_mano.inv() * Rotation.from_matrix(r_geom)).as_quat()
    quat = quat * np.where(quat @ quat[0] < 0.0, -1.0, 1.0)[:, None]
    _, eigvecs = np.linalg.eigh(quat.T @ quat)
    return np.asarray(Rotation.from_quat(eigvecs[:, -1]).as_matrix())


def ik_site_wrist_poses(joints: Tensor, rotvec: Tensor, is_right: bool = True,
                        valid: np.ndarray | None = None) -> Tensor:
    """`(T,4,4)` float64 IK-site wrist targets: position `joints[:, 0]`, rotation `R_mano(t) R_off`.

    Rows outside `valid` hold the identity pose; the retarget stage never reads them. Arguments and errors as in
    `wrist_frame_offset`.
    """
    rows = _valid_rows(joints, rotvec, valid)
    r_off = Rotation.from_matrix(wrist_frame_offset(joints, rotvec, is_right, valid))
    index = torch.from_numpy(rows)
    out = torch.eye(4, dtype=_DT).repeat(int(joints.shape[0]), 1, 1)
    out[index, :3, :3] = torch.from_numpy((Rotation.from_rotvec(rotvec.to(_DT)[index].numpy()) * r_off).as_matrix())
    out[index, :3, 3] = joints.to(_DT)[index, 0]
    return out


def wrist_target_poses(target: str, joints: Tensor, rotvec: Tensor, is_right: bool = True,
                       valid: np.ndarray | None = None) -> Tensor:
    """`(T,4,4)` float64 wrist targets of convention `target` (`WRIST_TARGETS`), in the joints' frame.

    `mcp` ignores `rotvec` and `valid` and returns `human_wrist_poses(joints)` on every row.

    Raises:
        ValueError: On an unknown `target`, or (`ik_site`) as in `wrist_frame_offset`.
    """
    if target == "mcp":
        return human_wrist_poses(joints.to(_DT), is_right=is_right)
    if target == "ik_site":
        return ik_site_wrist_poses(joints, rotvec, is_right, valid)
    raise ValueError(f"wrist target must be one of {WRIST_TARGETS}, got {target!r}")
