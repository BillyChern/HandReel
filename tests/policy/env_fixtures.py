"""Shared fixtures for the environment tests: the shipped whisking scene and a reference built from it.

The reference is the baseline's own kinematic trajectory (`trajectory_kinematic.npz`, 135 frames at 50 Hz) with
the object poses read from its free-joint block, a synthetic contact plan (thumb and index from frame 60) and a
synthetic per-frame posterior covariance (1 mm, 0.01 rad, log-scale std 0.01). At frame 0 the hand does not touch
the object (probed: 15 contacts, all hand self-contacts), so the welded warm-up holds the object exactly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.geometry.lie import quat_wxyz_to_matrix
from fastdo.policy.reference import PolicyReference, resample_reference

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "sharpa" / "right"
             / "whisking" / "0" / "scene.xml")
TRAJECTORY_NPZ = SCENE_XML.parent / "trajectory_kinematic.npz"
scene_missing = pytest.mark.skipif(not TRAJECTORY_NPZ.exists(),
                                   reason="baseline submodule whisking scene not checked out")

N_LABELS = 6
KP, KD = 300.0, 35.0          # the scene's finger stiffness and ~critical damping at armature 1


def whisking_reference(n_frames: int | None = None, covariance: bool = True, start: int = 0) -> PolicyReference:
    """The shipped whisking reference at 50 Hz from frame `start` (optionally `n_frames` long), float64 on the CPU.
    From frame 60 on the hand penetrates the object (C4 finding), so those frames are contact-rich."""
    from fastdo.retarget.kinematic import KinematicTrajectory

    with np.load(TRAJECTORY_NPZ) as data:
        qpos, qvel, freq = data["qpos"][start:], data["qvel"][start:], float(data["frequency"])
    if n_frames is not None:
        qpos, qvel = qpos[:n_frames], qvel[:n_frames]
    q = torch.as_tensor(qpos, dtype=torch.float64)
    T = int(q.shape[0])
    obj_pose = torch.eye(4, dtype=torch.float64).repeat(T, 1, 1)
    obj_pose[:, :3, :3] = quat_wxyz_to_matrix(q[:, 31:35])
    obj_pose[:, :3, 3] = q[:, 28:31]
    plan = torch.zeros(T, N_LABELS, dtype=torch.bool)
    plan[max(60 - start, 0):, 1:3] = True
    cov = None
    if covariance:
        cov = torch.diag(torch.tensor([1e-6] * 3 + [1e-4] * 3 + [1e-4], dtype=torch.float64)).repeat(T, 1, 1)
    kin = KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=freq)
    return resample_reference(kin, obj_pose, plan, cov, hz=50.0)


def env_config(**overrides: object):  # -> EnvConfig (imported lazily so collection works without the env)
    from fastdo.policy.env import EnvConfig

    base: dict[str, object] = {"n_worlds": 2, "kp": KP, "kd": KD, "randomize": False}
    base.update(overrides)
    return EnvConfig(**base)  # type: ignore[arg-type]


def moved_reference(ref: PolicyReference, yaw: float, shift: tuple[float, float, float]) -> PolicyReference:
    """`ref` rotated by `yaw` about world z and translated by `shift` (object poses and the wrist block of q_ref, with
    the XYZ Euler angles recomputed and unwrapped); q_ref keeps only the robot block (28)."""
    from scipy.spatial.transform import Rotation

    h = torch.eye(4, dtype=torch.float64)
    h[:3, :3] = torch.as_tensor(Rotation.from_euler("z", yaw).as_matrix())
    h[:3, 3] = torch.tensor(shift, dtype=torch.float64)
    q = ref.q_ref[:, :28].clone()
    wrist_rot = Rotation.from_euler("XYZ", q[:, 3:6].numpy())
    q[:, :3] = q[:, :3] @ h[:3, :3].T + h[:3, 3]
    new_rot = Rotation.from_matrix(h[:3, :3].numpy() @ wrist_rot.as_matrix())
    q[:, 3:6] = torch.as_tensor(np.unwrap(new_rot.as_euler("XYZ"), axis=0))
    return dataclasses.replace(ref, q_ref=q, obj_pose=h @ ref.obj_pose)


def with_object_block(ref: PolicyReference, value: float) -> PolicyReference:
    """`ref` with every column of q_ref past the robot block (the object free joint) overwritten by `value`."""
    q = ref.q_ref.clone()
    q[:, 28:] = value
    return dataclasses.replace(ref, q_ref=q)
