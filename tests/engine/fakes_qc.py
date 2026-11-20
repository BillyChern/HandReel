"""Artifact writers for the D6 `qc` and `retarget` worker tests: one solved clip's artifact directory (no models)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from fastdo.engine.worker_support import save_npz
from fastdo.engine.workers import write_hands_artifact, write_solve_artifact
from fastdo.geometry.intrinsics import camera_matrix, write_intrinsics
from fastdo.solver.api import TrajectorySolution
from tests.engine.fakes import make_hawor_npz, random_frames, write_decode

IMAGE_H, IMAGE_W = 24, 32
SOURCE_FPS = 30.0


def _side_valid(engine_valid: np.ndarray | None, frame_idx: np.ndarray, n_source: int) -> np.ndarray:
    out = np.ones(n_source, dtype=bool)
    if engine_valid is not None:
        out[frame_idx] = np.asarray(engine_valid, dtype=bool)
    return out


def write_solved_clip(artifact_dir: Path, t: int = 6, stride: int = 3, seed: int = 0,
                      right_valid: np.ndarray | None = None, left_valid: np.ndarray | None = None) -> TrajectorySolution:
    """Writes `decode.npz` (stride-`stride` rows of a 30 fps source), `intrinsics.npz`, `mask.npz`, the shape
    artifacts (`mesh.obj`, `samples.npz`, `sdf.npz`), `solve.npz` and `hands.npz` (from a HaWoR-format npz with one
    row per source frame; `right_valid`/`left_valid` give the engine rows' validity). Returns the solution."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frame_idx = stride * np.arange(t, dtype=np.int64)
    write_decode(artifact_dir, random_frames(t, IMAGE_H, IMAGE_W, seed), frame_idx=frame_idx.tolist(),
                 fps=SOURCE_FPS / stride, stride=stride)
    write_intrinsics(artifact_dir, camera_matrix(40.0, 40.0, 15.5, 11.5), "given")
    masks = np.zeros((t, IMAGE_H, IMAGE_W), dtype=bool)
    masks[:, 8:16, 10:22] = True
    save_npz(artifact_dir / "mask.npz", masks=masks, frame_idx=frame_idx, keyframe_index=np.array(0, dtype=np.int64))

    points = rng.uniform(-0.3, 0.3, size=(64, 3))
    save_npz(artifact_dir / "samples.npz", points=points.astype(np.float32),
             normals=(points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32))
    save_npz(artifact_dir / "sdf.npz", values=np.full((4, 4, 4), 0.5), origin=np.full(3, -0.5),
             step=np.array(1.0 / 3.0))
    (artifact_dir / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    poses = np.tile(np.eye(4), (t, 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec(0.05 * np.arange(t)[:, None] * np.array([0.0, 1.0, 0.0])).as_matrix()
    poses[:, :3, 3] = np.stack([0.01 * np.arange(t), np.zeros(t), np.full(t, 0.6)], axis=1)
    solution = TrajectorySolution(poses=torch.from_numpy(poses).float(), scale=0.2,
                                  covariance=1e-6 * torch.eye(6).expand(t, 6, 6).clone(),
                                  per_frame_cost=torch.full((t,), 2e-3), confidence=torch.linspace(0.4, 0.9, t),
                                  iterations=5, seconds=0.1)
    write_solve_artifact(artifact_dir, solution, frame_idx, torch.from_numpy(rng.uniform(size=(t, 16))).float())

    n_source = stride * t
    hawor = make_hawor_npz(artifact_dir / "all_hand_meshes.npz", n_source, seed=seed,
                           right_valid=_side_valid(right_valid, frame_idx, n_source),
                           left_valid=_side_valid(left_valid, frame_idx, n_source))
    write_hands_artifact(hawor, artifact_dir)
    return solution
