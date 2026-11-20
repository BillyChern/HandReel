"""Synthetic `ClipRecord` inputs and a capturing `retarget_kinematic` stand-in shared by the D6 qc tests (no data)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.qc.export import DEFAULT_GRAVITY_CAM, export_clip_record
from fastdo.retarget.kinematic import KinematicTrajectory
from fastdo.solver.api import TrajectorySolution
from fastdo.solver.sdf import SdfGrid

DT = torch.float64
N_VERTS = 778
K_PIX = torch.tensor([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def rigid(n: int, seed: int, spread: float = 1.0) -> Tensor:
    """`(n,4,4)` float64 independent random rigid transforms (rotation vectors and translations ~ N(0, spread))."""
    rng = np.random.default_rng(seed)
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = Rotation.from_rotvec(spread * rng.normal(size=(n, 3))).as_matrix()
    out[:, :3, 3] = spread * rng.normal(size=(n, 3))
    return torch.from_numpy(out)


def smooth_rigid(n: int, seed: int) -> Tensor:
    """`(n,4,4)` float64 rigid transforms along a smooth, non-affine path (a random pose, drifting and oscillating)."""
    rng = np.random.default_rng(seed)
    k = np.arange(n, dtype=np.float64)[:, None]
    rotvec = rng.normal(size=3) + 0.05 * k * rng.normal(size=3) + 0.02 * np.sin(k) * rng.normal(size=3)
    trans = rng.normal(size=3) + 0.01 * k * rng.normal(size=3) + 0.005 * np.cos(k) * rng.normal(size=3)
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    out[:, :3, 3] = trans
    return torch.from_numpy(out)


def solution(t: int, seed: int = 0, confidence: Tensor | None = None) -> TrajectorySolution:
    """A `TrajectorySolution`-shaped object track (float32 poses in front of nothing in particular)."""
    return TrajectorySolution(poses=smooth_rigid(t, seed).float(), scale=0.18,
                              covariance=1e-6 * torch.eye(6).expand(t, 6, 6).clone(),
                              per_frame_cost=torch.full((t,), 2e-3),
                              confidence=torch.linspace(0.3, 0.9, t) if confidence is None else confidence,
                              iterations=7, seconds=0.4)


def hand(t: int, seed: int, valid: np.ndarray | None = None) -> HandTrack:
    """A `HandTrack`: one fixed 21-joint hand (5 cm spread) carried along a smooth rigid path."""
    rng = np.random.default_rng(seed)
    base = 0.05 * rng.normal(size=(21, 3))
    motion = smooth_rigid(t, seed + 100).numpy()
    joints = np.einsum("tij,vj->tvi", motion[:, :3, :3], base) + motion[:, None, :3, 3]
    return HandTrack(params=rng.normal(size=(t, 48)).astype(np.float32), betas=rng.normal(size=10).astype(np.float32),
                     trans=joints[:, 0].astype(np.float32), joints=joints.astype(np.float32),
                     valid=np.ones(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool))


def hand_verts(t: int, seed: int) -> Tensor:
    """`(t,778,3)` float32 camera-frame "MANO vertices": a random cloud carried along a smooth rigid path."""
    rng = np.random.default_rng(seed)
    base = 0.04 * rng.normal(size=(N_VERTS, 3))
    motion = smooth_rigid(t, seed + 200).numpy()
    verts = np.einsum("tij,vj->tvi", motion[:, :3, :3], base) + motion[:, None, :3, 3]
    return torch.from_numpy(verts.astype(np.float32))


def hand_labels() -> Tensor:
    return (torch.arange(N_VERTS) % 6).to(torch.int8)


def make_record(frame_idx: np.ndarray, seed: int = 0, hand_right: HandTrack | None = None,
                hand_left: HandTrack | None = None, confidence: Tensor | None = None, fps: float = 30.0) -> ClipRecord:
    """A validated `ClipRecord` with a non-identity, moving `cam_to_world`."""
    idx = np.asarray(frame_idx, dtype=np.int64)
    t = int(idx.shape[0])
    return export_clip_record(f"clip{seed}", solution(t, seed, confidence), "/artifacts/mesh.obj", K_PIX,
                              smooth_rigid(t, seed + 1), torch.from_numpy(DEFAULT_GRAVITY_CAM), hand_left, hand_right,
                              0.8, idx, idx / fps)


def tiny_sdf() -> SdfGrid:
    return SdfGrid(values=torch.zeros(2, 2, 2, dtype=DT), origin=torch.zeros(3, dtype=DT), step=1.0)


class CapturingKinematic:
    """Stand-in for `fastdo.qc.export.retarget_kinematic`: records every call and returns a `(N,35)` trajectory."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.results: list[KinematicTrajectory] = []

    def __call__(self, *args: Any, **kwargs: Any) -> KinematicTrajectory:
        self.calls.append((args, kwargs))
        n = int(args[0].shape[0])
        result = KinematicTrajectory(qpos=np.full((n, 35), 0.25), qvel=np.zeros((n, 34)), frequency=float(args[9]))
        self.results.append(result)
        return result
