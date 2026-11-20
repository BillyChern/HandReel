"""Tests for `fastdo.robot.world_frame`: the record rebuilt from engine artifacts and the ego-motion
camera (Sim(3) per row folded into rigid poses)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.io.clip_record import ClipRecord
from fastdo.robot.world_frame import apply_ego_motion, load_reconstruction, record_from_engine_dir

T = 12
# Real-clip known answer (optional): FASTDO_WORLD_FRAME_REAL names an engine output directory holding qc-stage windows,
# and FASTDO_WORLD_FRAME_WINDOWS a comma-separated list of window directory names in it. Unset, the test skips.
_REAL = Path(os.environ.get("FASTDO_WORLD_FRAME_REAL") or "/nonexistent/world-frame-windows")
_WINDOWS = [w for w in (os.environ.get("FASTDO_WORLD_FRAME_WINDOWS") or "<window-a>,<window-b>").split(",") if w]


def _engine_dir(root: Path, seed: int = 0, stride: int = 3) -> Path:
    """An engine clip directory with the solve, hand and intrinsics artifacts but no `clip_record.npz`."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=[1.0, 0.6, 0.4]).export(root / "mesh.obj")
    poses = np.tile(np.eye(4), (T, 1, 1))
    poses[:, :3, :3] = Rotation.random(T, random_state=seed).as_matrix()
    poses[:, :3, 3] = rng.normal(0, 0.1, (T, 3)) + [0.0, 0.0, 0.5]
    frame_idx = (300 + stride * np.arange(T)).astype(np.int64)
    np.savez(root / "solve.npz", poses=poses.astype(np.float32), scale=np.float64(0.07),
             covariance=np.zeros((T, 6, 6)), per_frame_cost=rng.random(T).astype(np.float32),
             confidence=rng.random(T).astype(np.float32), iterations=np.int64(7), seconds=np.float64(1.5),
             frame_idx=frame_idx, track_weight=np.ones((T, 4), np.float32))
    hands = {"frame_idx": frame_idx}
    for side, valid in (("left", np.zeros(T, bool)), ("right", np.ones(T, bool))):
        joints = rng.normal(0, 0.05, (T, 21, 3)) + [0.0, 0.0, 0.45]
        hands.update({f"{side}_params": rng.normal(size=(T, 48)).astype(np.float32),
                      f"{side}_betas": rng.normal(size=10).astype(np.float32),
                      f"{side}_trans": joints[:, 0].astype(np.float32), f"{side}_joints": joints.astype(np.float32),
                      f"{side}_valid": valid, f"{side}_vertices": (rng.normal(0, 0.05, (T, 778, 3))
                                                                  + [0.0, 0.0, 0.45]).astype(np.float32)})
    np.savez(root / "hands.npz", **hands)
    np.savez(root / "intrinsics.npz", K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float32))
    return root


def _ego(path: Path, frame_idx: np.ndarray, seed: int = 1, ok: bool = True, valid: np.ndarray | None = None
         ) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Write an `ego_motion.npz` with a random Sim(3) per row; returns (path, s, Q, u)."""
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(0, 0.08, len(frame_idx)))
    q = Rotation.random(len(frame_idx), random_state=seed).as_matrix()
    u = rng.normal(0, 0.3, (len(frame_idx), 3))
    c2w = np.tile(np.eye(4), (len(frame_idx), 1, 1))
    c2w[:, :3, :3] = s[:, None, None] * q
    c2w[:, :3, 3] = u
    summary = {"ok": ok, "cam_path_m": 1.2, "cam_rot_extent_deg": 30.0}
    np.savez(path, cam_to_world=c2w, valid=np.ones(len(frame_idx), bool) if valid is None else valid,
             reference_row=np.int64(0), frame_idx=frame_idx, summary=np.array(json.dumps(summary)))
    return path, s, q, u


def test_record_is_rebuilt_from_engine_artifacts(tmp_path: Path) -> None:
    root = _engine_dir(tmp_path / "clip")
    rec = record_from_engine_dir(root)
    with np.load(root / "solve.npz") as s:
        assert np.array_equal(rec.object_pose, s["poses"]) and rec.object_scale == float(s["scale"])
        assert np.array_equal(rec.confidence, s["confidence"]) and np.array_equal(rec.frame_idx, s["frame_idx"])
    assert np.allclose(rec.timestamps, rec.frame_idx / 30.0)
    assert rec.hand_left is None and rec.hand_right is not None          # a side with no valid row is None
    with np.load(root / "hands.npz") as h:
        assert np.array_equal(rec.hand_right.joints, h["right_joints"])
    assert np.allclose(rec.cam_to_world[0, :3, :3] @ rec.gravity_cam, [0, 0, 1], atol=1e-6)   # static, up -> +z
    assert np.isnan(rec.gate_score) and rec.clip_id == "clip"
    with pytest.raises(ValueError, match="source frames apart"):
        record_from_engine_dir(_engine_dir(tmp_path / "bad", stride=2))


@pytest.mark.skipif(not all((_REAL / w / "clip_record.npz").is_file() for w in _WINDOWS),
                    reason="set FASTDO_WORLD_FRAME_REAL and FASTDO_WORLD_FRAME_WINDOWS to real qc-stage windows")
@pytest.mark.parametrize("window", _WINDOWS)
def test_rebuilt_record_equals_the_qc_stage_record(window: str) -> None:
    """Known answer: on two windows whose qc stage wrote `clip_record.npz`, the rebuild is byte-identical."""
    stored = ClipRecord.load(_REAL / window / "clip_record.npz")
    rebuilt = record_from_engine_dir(_REAL / window)
    for name in ("frame_idx", "timestamps", "intrinsics", "cam_to_world", "object_pose", "gravity_cam", "confidence"):
        a, b = getattr(stored, name), getattr(rebuilt, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert stored.object_scale == rebuilt.object_scale
    for side in ("hand_left", "hand_right"):
        a, b = getattr(stored, side), getattr(rebuilt, side)
        assert (a is None) == (b is None)
        if a is not None:
            assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("params", "betas", "trans", "joints",
                                                                              "valid"))


def test_ego_motion_puts_every_point_where_the_sim3_puts_it(tmp_path: Path) -> None:
    """The record's rigid camera applied to the rescaled camera-frame quantities equals the Sim(3) X_w = s Q X + u
    on the original ones: hand joints, vertices, the object centre; the object's world rotation is Q R."""
    root = _engine_dir(tmp_path / "clip")
    rec = record_from_engine_dir(root)
    with np.load(root / "hands.npz") as h:
        verts = np.asarray(h["right_vertices"], dtype=np.float64)
    path, s, q, u = _ego(tmp_path / "ego.npz", rec.frame_idx)
    out, verts_out, info = apply_ego_motion(rec, verts, path)
    c2w = out.cam_to_world.astype(np.float64)
    assert np.allclose(np.einsum("tij,tkj->tik", c2w[:, :3, :3], c2w[:, :3, :3]), np.eye(3), atol=1e-5)  # rigid

    def sim3(x: np.ndarray) -> np.ndarray:
        return s[:, None, None] * np.einsum("tij,tvj->tvi", q, x) + u[:, None]

    def rigid(x: np.ndarray) -> np.ndarray:
        return np.einsum("tij,tvj->tvi", c2w[:, :3, :3], x) + c2w[:, None, :3, 3]

    joints = rec.hand_right.joints.astype(np.float64)
    assert np.allclose(rigid(out.hand_right.joints.astype(np.float64)), sim3(joints), atol=1e-5)
    assert np.allclose(rigid(verts_out), sim3(verts), atol=1e-5)
    obj_c = rec.object_pose[:, None, :3, 3].astype(np.float64)
    assert np.allclose(rigid(out.object_pose[:, None, :3, 3].astype(np.float64)), sim3(obj_c), atol=1e-5)
    world_r = np.einsum("tij,tjk->tik", c2w[:, :3, :3], out.object_pose[:, :3, :3].astype(np.float64))
    assert np.allclose(world_r, np.einsum("tij,tjk->tik", q, rec.object_pose[:, :3, :3].astype(np.float64)), atol=1e-5)
    assert out.object_scale == rec.object_scale
    assert info.ok and info.scale_range == pytest.approx((s.min(), s.max()))


def test_ego_motion_is_refused_when_unusable(tmp_path: Path) -> None:
    rec = record_from_engine_dir(_engine_dir(tmp_path / "clip"))
    bad_ok = _ego(tmp_path / "a.npz", rec.frame_idx, ok=False)[0]
    with pytest.raises(ValueError, match="self-check failed"):
        apply_ego_motion(rec, None, bad_ok)
    valid = np.ones(T, bool)
    valid[3] = False
    with pytest.raises(ValueError, match="off the camera trajectory"):
        apply_ego_motion(rec, None, _ego(tmp_path / "b.npz", rec.frame_idx, valid=valid)[0])
    with pytest.raises(ValueError, match="rows are not the record"):
        apply_ego_motion(rec, None, _ego(tmp_path / "c.npz", rec.frame_idx + 3)[0])


def test_load_reconstruction_rebuilds_and_applies_the_ego_motion(tmp_path: Path) -> None:
    root = _engine_dir(tmp_path / "clip")
    path = _ego(tmp_path / "ego.npz", record_from_engine_dir(root).frame_idx)[0]
    plain = load_reconstruction(root)
    assert plain.record_source == "rebuilt" and plain.ego is None and plain.camera_source == "unknown"
    ego = load_reconstruction(root, path)
    assert ego.camera_source == "ego_motion" and ego.ego is not None and ego.ego.ok
    assert not np.allclose(ego.right_vertices_cam, plain.right_vertices_cam)


def test_gap_fill_is_opt_in_and_bounded(tmp_path: Path) -> None:
    """max_gap_rows=0 (default) refuses as before; >0 fills a short interior gap (rows equal to the
    filler's own output), leaves the valid rows exactly as without the gap, and still refuses a gap above the cap."""
    from fastdo.geometry.ego_gaps import fill_gaps

    rec = record_from_engine_dir(_engine_dir(tmp_path / "clip"))
    valid = np.ones(T, bool)
    valid[[4, 5]] = False
    gap = _ego(tmp_path / "gap.npz", rec.frame_idx, valid=valid)[0]
    full = _ego(tmp_path / "full.npz", rec.frame_idx)[0]                       # same seed: same poses, no gap
    with pytest.raises(ValueError, match="off the camera trajectory"):
        apply_ego_motion(rec, None, gap)
    with pytest.raises(ValueError, match="2 rows are off"):
        apply_ego_motion(rec, None, gap, max_gap_rows=1)
    out, _, info = apply_ego_motion(rec, None, gap, max_gap_rows=2)
    ref, _, ref_info = apply_ego_motion(rec, None, full)
    assert info.filled_rows == 2 and info.valid_rows == T and info.to_dict()["filled_rows"] == 2
    assert "filled_rows" not in ref_info.to_dict()                              # stored stats unchanged by default
    assert np.array_equal(out.cam_to_world[valid], ref.cam_to_world[valid])
    assert np.array_equal(out.object_pose[valid], ref.object_pose[valid])
    with np.load(gap) as z:
        filled = fill_gaps(z["cam_to_world"], z["valid"], 2)[0][[4, 5]]
    s = np.cbrt(np.linalg.det(filled[:, :3, :3]))
    assert np.allclose(out.cam_to_world[[4, 5], :3, :3], filled[:, :3, :3] / s[:, None, None], atol=1e-6)
    assert np.allclose(out.object_pose[[4, 5], :3, 3], rec.object_pose[[4, 5], :3, 3] * s[:, None], atol=1e-6)
    recon = load_reconstruction(tmp_path / "clip", gap, ego_max_gap_rows=2)         # reachable from the loader
    assert recon.ego is not None and recon.ego.filled_rows == 2 and recon.camera_source == "ego_motion"
