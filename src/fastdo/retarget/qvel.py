"""Finite-difference velocities of a kinematic reference (moved verbatim from `fastdo.retarget.kinematic`).

`kinematic_qvel` is the qvel of the baseline's `trajectory_kinematic.npz` format that both the legacy and the fast
path of `retarget_kinematic` emit: central differences of the 28 robot dofs and the object position, and the
object's angular velocity from `so3_log` of consecutive rotations, one-sided at the two endpoints.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fastdo.geometry.lie import quat_wxyz_to_matrix, so3_log

__all__ = ["kinematic_qvel"]


def _central_diff(x: Tensor, dt: float) -> Tensor:
    """`(T,k)` -> `(T,k)` central differences; one-sided at the two endpoints."""
    v = torch.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def _angular_velocity(quat_wxyz: Tensor, dt: float) -> Tensor:
    """`(T,4)` wxyz quaternions -> `(T,3)` angular velocity via `so3_log` of consecutive rotations."""
    R = quat_wxyz_to_matrix(quat_wxyz.to(torch.float64))
    omega = torch.zeros(quat_wxyz.shape[0], 3, dtype=torch.float64)
    rel_mid = R[2:] @ R[:-2].transpose(-1, -2)
    omega[1:-1] = so3_log(rel_mid) / (2.0 * dt)
    omega[0] = so3_log(R[1] @ R[0].transpose(-1, -2)) / dt
    omega[-1] = so3_log(R[-1] @ R[-2].transpose(-1, -2)) / dt
    return omega


def kinematic_qvel(qpos: Tensor, frequency: float) -> Tensor:
    """`(T, 35)` float64 qpos `[robot (28), object pos (3), object quat wxyz (4)]` -> `(T, 34)` qvel at `frequency`."""
    T = qpos.shape[0]
    dt = 1.0 / frequency
    qvel = torch.zeros(T, 34, dtype=torch.float64)
    qvel[:, :28] = _central_diff(qpos[:, :28], dt)
    qvel[:, 28:31] = _central_diff(qpos[:, 28:31], dt)
    qvel[:, 31:34] = _angular_velocity(qpos[:, 31:35], dt)
    return qvel
