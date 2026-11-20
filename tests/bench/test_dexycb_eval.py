import dataclasses

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.bench.alignment import scaled_icp
from fastdo.bench.dexycb_eval import (
    ALIGNMENTS,
    _posed,
    evaluate_sequence,
    evaluate_sequence_protocols,
    sample_mesh_points,
)
from fastdo.bench.metrics import chamfer_distance, f_score
from fastdo.io.clip_record import ClipRecord


def _record(t: int, poses: np.ndarray) -> ClipRecord:
    return ClipRecord(clip_id="s", frame_idx=np.arange(t, dtype=np.int64), timestamps=np.arange(t) / 30.0,
                      intrinsics=np.eye(3, dtype=np.float32), cam_to_world=np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)),
                      object_mesh_path="m.obj", object_pose=poses.astype(np.float32), object_scale=1.0,
                      gravity_cam=np.array([0, -1, 0], np.float32), confidence=np.ones(t, np.float32), gate_score=1.0,
                      hand_left=None, hand_right=None)


def test_perfect_prediction_scores_one() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    t = 3
    poses = np.tile(np.eye(4), (t, 1, 1)); poses[:, 2, 3] = 0.6
    rec = _record(t, poses)
    m = evaluate_sequence(rec, mesh.vertices, mesh.faces, mesh.vertices, mesh.faces, poses, 1.0, n_points=2000)
    assert m.n_frames == 3 and m.f5 > 0.99 and m.f10 > 0.99 and m.chamfer_cm2 < 0.01


def test_offset_prediction_scores_lower() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    poses = np.tile(np.eye(4), (2, 1, 1)); poses[:, 2, 3] = 0.6
    rec = _record(2, poses)
    shifted = poses.copy(); shifted[:, 0, 3] += 0.02
    m = evaluate_sequence(rec, mesh.vertices, mesh.faces, mesh.vertices, mesh.faces, shifted, 1.0, n_points=2000)
    assert m.f5 < 0.5 and m.chamfer_cm2 > 1.0


def test_sampling_is_deterministic() -> None:
    mesh = trimesh.creation.box()
    a = sample_mesh_points(mesh.vertices, mesh.faces, 100, seed=3)
    b = sample_mesh_points(mesh.vertices, mesh.faces, 100, seed=3)
    assert np.array_equal(a, b) and a.shape == (100, 3)


def _box_scene(t: int) -> tuple[trimesh.Trimesh, ClipRecord, np.ndarray]:
    """A box posed differently in each of `t` frames; returns the mesh, the record and its float64 GT poses."""
    mesh = trimesh.creation.box(extents=[0.16, 0.21, 0.06])
    poses = np.tile(np.eye(4), (t, 1, 1))
    for k in range(t):
        poses[k, :3, :3] = Rotation.from_euler("xyz", (6.0 * k, -5.0 * k, 4.0 * k), degrees=True).as_matrix()
        poses[k, :3, 3] = (0.01 * k, -0.01 * k, 0.5)
    rec = _record(t, poses)
    return mesh, rec, rec.object_pose.astype(np.float64)


def test_protocols_none_equals_evaluate_sequence() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    t = 3
    poses = np.tile(np.eye(4), (t, 1, 1)); poses[:, 2, 3] = 0.6
    shifted = poses.copy(); shifted[:, 0, 3] += 0.004; shifted[1, 1, 3] -= 0.003
    for rec in (_record(t, poses), dataclasses.replace(_record(t, poses), confidence=np.array([1, 0, 1], np.float32))):
        args = (rec, mesh.vertices, mesh.faces, mesh.vertices, mesh.faces, shifted, 1.02)
        got = evaluate_sequence_protocols(*args, n_points=2000, seed=3, solver_seconds=1.5)
        want = evaluate_sequence(*args, n_points=2000, seed=3, solver_seconds=1.5)
        assert tuple(got) == ALIGNMENTS == ("none", "icp_frame", "icp_clip")
        for field in dataclasses.fields(want):
            assert getattr(got["none"], field.name) == getattr(want, field.name), field.name
        assert all(m.n_frames == want.n_frames and m.solver_seconds == 1.5 for m in got.values())
    assert want.n_frames == 2


_GLOBAL_SCALE = 1.05


def _global_similarity_poses(gt: np.ndarray) -> np.ndarray:
    """`gt` composed with one global similarity (scale `_GLOBAL_SCALE`, 3 deg, 20 mm), for pred_scale `_GLOBAL_SCALE`."""
    axis = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)
    r_g = Rotation.from_rotvec(np.deg2rad(3.0) * axis).as_matrix()
    t_g = 0.02 * np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
    pred = gt.copy()                                     # x_pred = s R_g (R_k p + t_k) + t_g
    pred[:, :3, :3] = r_g @ gt[:, :3, :3]
    pred[:, :3, 3] = _GLOBAL_SCALE * gt[:, :3, 3] @ r_g.T + t_g
    return pred


def test_protocols_forgive_global_similarity() -> None:
    mesh, rec, gt = _box_scene(4)
    m = evaluate_sequence_protocols(rec, mesh.vertices, mesh.faces, mesh.vertices, mesh.faces,
                                    _global_similarity_poses(gt), _GLOBAL_SCALE, n_points=5000)
    assert m["none"].f5 < 0.9
    assert m["icp_clip"].f5 == 1.0
    assert m["icp_frame"].f5 == 1.0


def _metrics_path(rec: ClipRecord, mesh: trimesh.Trimesh, pred_poses: np.ndarray, pred_scale: float, n_points: int,
                  seed: int) -> dict[str, tuple[float, float, float, int]]:
    """Every protocol scored through `f_score` / `chamfer_distance` on the same sampling, posing and ICP fits."""
    gt_pts = sample_mesh_points(mesh.vertices, mesh.faces, n_points, seed)
    pred_pts = sample_mesh_points(mesh.vertices, mesh.faces, n_points, seed)
    keep = [k for k in range(rec.n_frames) if rec.confidence[k] > 0]
    gts = [_posed(gt_pts, rec.object_pose[k].astype(np.float64), 1.0) for k in keep]
    preds = [_posed(pred_pts, pred_poses[k].astype(np.float64), pred_scale) for k in keep]
    aligned = {"none": preds, "icp_frame": [scaled_icp([p], [g])[0][0] for p, g in zip(preds, gts, strict=True)],
               "icp_clip": scaled_icp(preds, gts)[0]}
    out: dict[str, tuple[float, float, float, int]] = {}
    for name, clouds in aligned.items():
        pairs = list(zip(clouds, gts, strict=True))
        out[name] = (float(np.mean([f_score(p, g, 0.005) for p, g in pairs])),
                     float(np.mean([f_score(p, g, 0.010) for p, g in pairs])),
                     float(np.mean([chamfer_distance(p, g, squared=True) * 1e4 for p, g in pairs])), len(pairs))
    return out


def test_protocol_scores_bitwise_equal_the_metrics_path() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 2, 3] = 0.6
    shifted = poses.copy()
    shifted[:, 0, 3] += 0.004
    shifted[1, 1, 3] -= 0.003
    masked = dataclasses.replace(_record(3, poses), confidence=np.array([1, 0, 1], np.float32))
    box4, rec4, gt4 = _box_scene(4)
    box6, rec6, gt6 = _box_scene(6)
    tracking = gt6.copy()
    tracking[1::2, :3, 3] += 0.015 * np.ones(3) / np.sqrt(3.0)
    cases = [(_record(3, poses), sphere, shifted, 1.02, 2000, 3), (masked, sphere, shifted, 1.02, 2000, 3),
             (rec4, box4, _global_similarity_poses(gt4), _GLOBAL_SCALE, 5000, 0), (rec6, box6, tracking, 1.0, 5000, 0)]
    for i, (rec, mesh, pred, scale, n_points, seed) in enumerate(cases):
        got = evaluate_sequence_protocols(rec, mesh.vertices, mesh.faces, mesh.vertices, mesh.faces, pred, scale,
                                          n_points=n_points, seed=seed)
        want = _metrics_path(rec, mesh, pred, scale, n_points, seed)
        assert tuple(want) == ALIGNMENTS
        for name in ALIGNMENTS:
            m = got[name]
            assert (m.f5, m.f10, m.chamfer_cm2, m.n_frames) == want[name], (i, name)


@pytest.mark.parametrize("direction", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)],
                         ids=["x", "y", "z", "diagonal"])
def test_protocols_clip_counts_tracking_error(direction: tuple[float, float, float]) -> None:
    mesh, rec, gt = _box_scene(6)
    pred = gt.copy()
    pred[1::2, :3, 3] += 0.015 * np.asarray(direction) / np.linalg.norm(direction)   # 15 mm, every other frame
    m = evaluate_sequence_protocols(rec, mesh.vertices, mesh.faces, mesh.vertices, mesh.faces, pred, 1.0,
                                    n_points=5000)
    frame, clip = m["icp_frame"], m["icp_clip"]
    assert frame.f10 == 1.0
    if direction == (1.0, 1.0, 1.0):
        # Along a single camera axis the clip fit leaves residuals under 10 mm, so F-10 ties at 1.0 there.
        assert clip.f10 < frame.f10
    assert clip.f5 < frame.f5 - 0.1
    assert clip.chamfer_cm2 > frame.chamfer_cm2 + 0.1
