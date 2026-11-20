"""Tests for `fastdo.retarget.baseline_io.baseline_world_inputs`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fastdo.retarget.baseline_io import baseline_world_inputs

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "sharpa"
             / "right" / "whisking" / "0")
KEYPOINTS_NPZ = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "mano"
                 / "right" / "whisking" / "0" / "trajectory_keypoints.npz")
TRAJECTORY_NPZ = SCENE_DIR / "trajectory_kinematic.npz"

whisking_data_missing = pytest.mark.skipif(
    not (KEYPOINTS_NPZ.exists() and TRAJECTORY_NPZ.exists()),
    reason="shipped whisking demo trajectory_keypoints.npz/trajectory_kinematic.npz not available",
)


@whisking_data_missing
def test_baseline_world_inputs_shapes_and_demo_frequency() -> None:
    obj_poses, wrist_poses, hand_verts, frequency = baseline_world_inputs(KEYPOINTS_NPZ, TRAJECTORY_NPZ)
    with np.load(KEYPOINTS_NPZ) as d:
        n = np.asarray(d["qpos_obj_right"]).shape[0]
    assert obj_poses.shape == (n, 4, 4)
    assert wrist_poses.shape == (n, 4, 4)
    assert hand_verts.shape == (n, 778, 3)
    assert obj_poses.dtype == torch.float64
    assert hand_verts.dtype == torch.float64
    assert frequency == 50.0  # the whisking demo's shipped trajectory_kinematic.npz frequency


@whisking_data_missing
def test_baseline_world_inputs_hand_verts_come_from_mano_verts_right_not_all_hand_meshes() -> None:
    """The measured-correction fix (see module docstring): `hand_verts` must be `mano_verts_right`
    (already world-frame-shifted, consistent with `qpos_obj_right`), not `all_hand_meshes.npz`'s
    raw (different-frame) vertices."""
    _, _, hand_verts, _ = baseline_world_inputs(KEYPOINTS_NPZ, TRAJECTORY_NPZ)
    with np.load(KEYPOINTS_NPZ) as d:
        expected = np.asarray(d["mano_verts_right"], dtype=np.float64)
    np.testing.assert_allclose(hand_verts.numpy(), expected, atol=1e-6)


@whisking_data_missing
def test_baseline_world_inputs_pose_matches_raw_qpos_fields() -> None:
    obj_poses, wrist_poses, _, _ = baseline_world_inputs(KEYPOINTS_NPZ, TRAJECTORY_NPZ)
    with np.load(KEYPOINTS_NPZ) as d:
        qpos_obj = np.asarray(d["qpos_obj_right"], dtype=np.float64)
        qpos_wrist = np.asarray(d["qpos_wrist_right"], dtype=np.float64)

    for t in (0, 1, 50, qpos_obj.shape[0] - 1):
        expected_R = Rotation.from_quat(qpos_obj[t, 3:7][[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(obj_poses[t, :3, :3].numpy(), expected_R, atol=1e-9)
        np.testing.assert_allclose(obj_poses[t, :3, 3].numpy(), qpos_obj[t, :3], atol=1e-12)
        assert obj_poses[t, 3, 3] == 1.0

        expected_R_wrist = Rotation.from_quat(qpos_wrist[t, 3:7][[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(wrist_poses[t, :3, :3].numpy(), expected_R_wrist, atol=1e-9)
        np.testing.assert_allclose(wrist_poses[t, :3, 3].numpy(), qpos_wrist[t, :3], atol=1e-12)


@whisking_data_missing
def test_baseline_world_inputs_gives_physically_sensible_contact_distances() -> None:
    """Direct evidence for the frame-consistency fix (module docstring's "a correction" section):
    mapping `hand_verts`/`obj_poses` through `human_contact_measure`'s canonical map should show
    small (near-zero, some negative -- penetrating) signed distances at a known real contact frame,
    and *not* every frame in permanent/near contact -- i.e. sensible, varying per-frame geometry, not
    the ~30mm-off, effectively-random distances the mismatched (camera-frame-poses, world-frame-qpos)
    pairing produced when measured.
    """
    import trimesh

    from fastdo.retarget.contact import human_contact_measure
    from fastdo.retarget.labels import PALM
    from fastdo.solver.sdf import mesh_sdf_grid

    whisk_obj = (REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "whisking"
                 / "video_segmentation" / "masks" / "frame_000125_masks" / "whisk" / "whisk.obj")
    if not whisk_obj.exists():
        pytest.skip("whisk.obj mesh not available")

    obj_poses, _, hand_verts, _ = baseline_world_inputs(KEYPOINTS_NPZ, TRAJECTORY_NPZ)
    mesh = trimesh.load(whisk_obj, process=False)
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64)
    surf_pts, _ = trimesh.sample.sample_surface(mesh, 1024, seed=np.random.default_rng(0))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    labels = torch.full((778,), PALM, dtype=torch.int8)  # labels irrelevant here, only sdf_values used
    mesh_scale = 0.180863  # from the baseline layout; not re-derived here, just needs to be in the right ballpark

    measure = human_contact_measure(hand_verts, obj_poses, mesh_scale, labels, sdf, surface,
                                    sigma=0.004, tau=0.008)
    # A known real hand-object contact configuration (frame 60, per test_contact.py); the mismatched
    # pairing gave uniformly-large, no-contact-looking distances everywhere.
    assert float(measure.sdf_values[60].min()) < 0.0, "expected some penetrating (contact) vertices at frame 60"
    assert float(measure.mass[60]) > 0.0


@whisking_data_missing
def test_trajectory_keypoints_and_trajectory_kinematic_frame_offset_is_still_plus_one() -> None:
    """Regression guard for C2's own frame-offset finding (not consumed by this module -- see its
    docstring -- but documented and re-verified here so a future change to either file's frame
    trimming/count fails loudly instead of silently invalidating that documented relationship)."""
    with np.load(KEYPOINTS_NPZ) as d:
        qpos_obj_keypoints = np.asarray(d["qpos_obj_right"], dtype=np.float64)
    with np.load(TRAJECTORY_NPZ) as d:
        qpos_kinematic = np.asarray(d["qpos"], dtype=np.float64)

    n_kin = qpos_kinematic.shape[0]
    best_offset, best_residual = None, float("inf")
    for offset in range(-2, 3):
        lo = max(0, -offset)
        hi = min(n_kin, qpos_obj_keypoints.shape[0] - offset)
        if hi <= lo:
            continue
        kin_slice = qpos_kinematic[lo:hi, 28:31]
        kp_slice = qpos_obj_keypoints[lo + offset:hi + offset, :3]
        residual = float(np.abs(kin_slice - kp_slice).mean())
        if residual < best_residual:
            best_residual, best_offset = residual, offset

    assert best_offset == 1, f"expected the documented +1 offset, got {best_offset} (residual {best_residual})"
