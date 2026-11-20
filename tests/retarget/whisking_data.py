"""Shared whisking-demo paths, markers and loaders for the retargeting tests (no fixtures, no tests).

The paths, `whisking_data_missing` and the input loader are copied from `tests/retarget/test_kinematic.py`
(which stays unchanged); the scene / SDF / surface builders are that module's fixtures as plain functions.
Test modules define their own module-scoped fixtures around these builders, so nothing fixture-like is imported.

The baseline data resolve against this checkout's `third_party/do-as-i-do`, or against `$FASTDO_BASELINE_ROOT` when
this checkout has no demo scene and the variable is set (a worktree without the submodule checked out).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from torch import Tensor

from fastdo.retarget.baseline_io import baseline_world_inputs
from fastdo.retarget.labels import mano_finger_labels
from fastdo.retarget.mj_model import RobotScene
from fastdo.solver.sdf import SdfGrid, mesh_sdf_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENE_REL = Path("retargeting") / "outputs" / "sharpa" / "right" / "whisking" / "0" / "scene.xml"


def _baseline_root() -> Path:
    local = REPO_ROOT / "third_party" / "do-as-i-do"
    env = os.environ.get("FASTDO_BASELINE_ROOT")
    return Path(env) if env and not (local / _SCENE_REL).is_file() else local


BASELINE_ROOT = _baseline_root()
SCENE_XML = BASELINE_ROOT / _SCENE_REL
KEYPOINTS_NPZ = (BASELINE_ROOT / "retargeting" / "outputs" / "mano" / "right" / "whisking" / "0"
                 / "trajectory_keypoints.npz")
TRAJECTORY_NPZ = SCENE_XML.parent / "trajectory_kinematic.npz"
WHISK_OBJ = (BASELINE_ROOT / "reconstruction" / "whisking" / "video_segmentation" / "masks" / "frame_000125_masks"
             / "whisk" / "whisk.obj")
_MANO_RIGHT_CANDIDATES = (
    REPO_ROOT / "data" / "mano" / "MANO_RIGHT.pkl",
    Path("/opt/fastdo/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl"),
)
MANO_RIGHT_PKL = next((p for p in _MANO_RIGHT_CANDIDATES if p.exists()), None)
MESH_SCALE = 0.18086324855404615  # measured from the baseline layout; not re-derived here
REFERENCE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "retarget" / "c3b_whisking_reference.npz"

_WHISKING_FILES = (SCENE_XML, KEYPOINTS_NPZ, TRAJECTORY_NPZ, WHISK_OBJ)
whisking_data_missing = pytest.mark.skipif(
    not all(p.exists() for p in _WHISKING_FILES) or MANO_RIGHT_PKL is None,
    reason=f"shipped whisking demo data or MANO pkl not available: "
           f"{[str(p) for p in _WHISKING_FILES if not p.exists()]}",
)
scene_missing = pytest.mark.skipif(not SCENE_XML.exists(), reason="baseline scene.xml not checked out")


def load_whisking_inputs(frames: np.ndarray | None = None) -> dict[str, Tensor | float]:
    """Every `retarget_kinematic` input in the single MuJoCo world frame of `baseline_world_inputs`, with the
    per-frame tensors indexed by `frames` when given (keys as `test_kinematic._load_whisking_inputs`)."""
    obj_poses, wrist_poses, hand_verts, frequency = baseline_world_inputs(KEYPOINTS_NPZ, TRAJECTORY_NPZ)
    assert MANO_RIGHT_PKL is not None
    labels = mano_finger_labels(MANO_RIGHT_PKL)
    if frames is not None:
        idx = torch.as_tensor(np.asarray(frames, dtype=np.int64))
        obj_poses, wrist_poses, hand_verts = obj_poses[idx], wrist_poses[idx], hand_verts[idx]
    return {
        "obj_poses": obj_poses,
        "wrist_poses": wrist_poses,
        "mesh_scale": MESH_SCALE,
        "hand_verts": hand_verts,
        "hand_labels": labels,
        "frequency": frequency,
    }


def build_scene() -> RobotScene:
    return RobotScene.from_xml(SCENE_XML)


def build_whisk_sdf_and_surface() -> tuple[SdfGrid, Tensor, Tensor]:
    """64^3 whisk SDF grid, 1024 surface samples (`default_rng(0)`) and their face normals."""
    mesh = trimesh.load(WHISK_OBJ, process=False)
    assert isinstance(mesh, trimesh.Trimesh)
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64)
    surf_pts, face_idx = trimesh.sample.sample_surface(mesh, 1024, seed=np.random.default_rng(0))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    surface_normals = torch.as_tensor(np.asarray(mesh.face_normals[face_idx]), dtype=torch.float64)
    return sdf, surface, surface_normals


def load_reference_fixture() -> dict[str, np.ndarray]:
    """The legacy reference written by `run/pipeline/profile_retarget_kinematic.py`.

    The release does not ship this demo-derived fixture (robot qpos derived from DO AS I DO's demo video,
    whose licence is unverified), so a missing file skips."""
    if not REFERENCE_FIXTURE.exists():
        pytest.skip(f"kinematic reference fixture not shipped (demo-derived): {REFERENCE_FIXTURE}")
    with np.load(REFERENCE_FIXTURE) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def engine_10s_frames() -> np.ndarray:
    """The 100-frame 10 fps per-10 s clip: the 47 whisking keys forward, backward, forward again."""
    from fastdo.retarget.rate import engine_key_frames

    keys = engine_key_frames(138, 3)
    return np.concatenate([keys, keys[::-1][1:], keys[1:]])[:100]


def full_10s_frames() -> np.ndarray:
    """The 300-frame full-rate per-10 s clip, the same forward-backward-forward construction on `arange(138)`."""
    full = np.arange(138, dtype=np.int64)
    return np.concatenate([full, full[::-1][1:], full[1:]])[:300]
