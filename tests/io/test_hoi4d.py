"""HOI4D reader and reprojection check (synthetic tree; no real HOI4D file is read)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.io.datasets.hoi4d import (
    EXCLUDED_CATEGORIES,
    RIGID_CATEGORIES,
    HOI4DReader,
    contact_time,
    hand_track,
    parse_events,
    parse_objpose,
    parse_sequence_id,
)
from fastdo.io.datasets.hoi4d_gt_check import check_frame, silhouette
from tests.io.hoi4d_tree import (
    HW,
    N_FRAMES,
    SEQ,
    SEQ_LAPTOP,
    SEQ_PADDED,
    K,
    box,
    gt_pose,
    write_tree,
)


def test_parse_sequence_id() -> None:
    sid = parse_sequence_id("ZY20210800003/H3/C13/N45/S200/s05/T4")
    assert (sid.camera, sid.human, sid.category, sid.instance, sid.room, sid.layout, sid.task) == (3, 3, 13, 45, 200, 5, 4)
    assert sid.camera_dir == "ZY20210800003"
    for bad in ("ZY20210800003/H3/C13/N45/S200/s05", "zy/H3/C13/N45/S200/s05/T4", "ZY20210800003/H3/C13/N45/S200/s05/T4/"):
        with pytest.raises(ValueError):
            parse_sequence_id(bad)


def test_rigid_and_excluded_categories_partition_the_sixteen() -> None:
    assert set(RIGID_CATEGORIES) == {1, 2, 5, 7, 12, 13}
    assert not set(RIGID_CATEGORIES) & set(EXCLUDED_CATEGORIES)
    assert len(RIGID_CATEGORIES) + len(EXCLUDED_CATEGORIES) == 16


def test_parse_objpose_is_xyz_euler_object_to_camera() -> None:
    ang, ctr = [0.4, -1.1, 2.0], [0.1, -0.2, 0.7]
    pose, ok = parse_objpose({"isEffective": 1, "dataList": [{"center": dict(zip("xyz", ctr, strict=True)),
                                                              "rotation": dict(zip("xyz", ang, strict=True))}]})
    assert ok
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_euler("XYZ", ang).as_matrix(), atol=1e-6)
    np.testing.assert_allclose(pose[:3, 3], ctr, atol=1e-6)
    # the intrinsic XYZ order is not the extrinsic xyz one: a wrong order would be caught
    assert not np.allclose(pose[:3, :3], Rotation.from_euler("xyz", ang).as_matrix(), atol=1e-3)


def test_parse_objpose_invalid_and_malformed() -> None:
    item = {"center": {"x": 0, "y": 0, "z": 1}, "rotation": {"x": 0, "y": 0, "z": 0}}
    assert parse_objpose({"isEffective": 0, "dataList": [item]})[1] is False
    assert parse_objpose({"isEffective": 1, "dataList": []})[1] is False
    assert parse_objpose({"isEffective": 1, "dataList": [{**item, "center": {"x": float("nan"), "y": 0, "z": 1}}]})[1] is False
    assert parse_objpose({"objects": [item]})[1] is True        # the README's alternative key
    with pytest.raises(ValueError):
        parse_objpose({"isEffective": 1, "dataList": [{"center": {"x": 0, "y": 0, "z": 1}}]})


def test_parse_events_both_layouts_and_contact_time() -> None:
    a = parse_events({"events": [{"event": "rest", "startTime": 0, "endTime": 1},
                                 {"event": "Reachout", "startTime": 1, "endTime": 2.5},
                                 {"event": "Grasp", "startTime": 2.5, "endTime": 3}]})
    b = parse_events({"markResult": {"marks": [{"event": "push", "hdTimeStart": 4.0, "hdTimeEnd": 5.0}]}})
    assert contact_time(a) == 2.5 and contact_time(b) == 4.0
    assert contact_time([("rest", 0.0, 1.0), ("Reachout", 1.0, 2.0)]) is None
    with pytest.raises(ValueError):
        parse_events({"info": {}})


def test_reader_sequences_filter_categories_and_dedupe(tmp_path: Path) -> None:
    reader = HOI4DReader(write_tree(tmp_path / "hoi4d"))
    assert reader.sequences() == [SEQ, SEQ_PADDED]
    assert SEQ_LAPTOP in reader.sequences(categories=(3,))
    with pytest.raises(ValueError, match="not a rigid"):
        reader.mesh_path(SEQ_LAPTOP)


@pytest.mark.parametrize("seq", [SEQ, SEQ_PADDED])
def test_reader_load_full_length_absolute_frames(tmp_path: Path, seq: str) -> None:
    reader = HOI4DReader(write_tree(tmp_path / "hoi4d", invalid=(3,), missing=(5,)))
    rec = reader.load(seq)
    n = N_FRAMES                                  # frame 5 has no file: a missing frame, not a shorter sequence
    assert rec.n_frames == n and rec.frame_idx.tolist() == list(range(n))
    np.testing.assert_allclose(rec.timestamps, np.arange(n) / 15.0)
    assert rec.confidence[3] == 0 and rec.confidence[5] == 0 and rec.confidence[[0, 1, 2, 4, 6]].min() == 1
    np.testing.assert_allclose(rec.object_pose[3], np.eye(4))
    for t in (0, 4, 11):
        np.testing.assert_allclose(rec.object_pose[t], gt_pose(t), atol=1e-5)
    assert rec.object_mesh_path.endswith("rigid/Mug/001.obj") and rec.hand_right is None
    np.testing.assert_allclose(rec.intrinsics, K)


def test_intrinsics_shift_moves_only_the_principal_point(tmp_path: Path) -> None:
    reader = HOI4DReader(write_tree(tmp_path / "hoi4d"))
    k = reader.intrinsics(SEQ, (3.0, -2.5))
    np.testing.assert_allclose(k, K + np.array([[0, 0, 3.0], [0, 0, -2.5], [0, 0, 0]]))


def test_pose_registration_moves_the_object_laterally_and_keeps_the_shipped_k(tmp_path: Path) -> None:
    """The registration corrects the OBJECT pose (the hands and image agree with the shipped K): its origin projects
    (dx, dy) px further, rotation and depth unchanged, invalid frames untouched."""
    reader = HOI4DReader(write_tree(tmp_path / "hoi4d"))
    rec = reader.load(SEQ, pose_shift_px=(3.0, -2.5))
    np.testing.assert_allclose(rec.intrinsics, K)
    for t in (0, 4, 11):
        p, q = gt_pose(t), rec.object_pose[t].astype(np.float64)
        np.testing.assert_allclose(q[:3, :3], p[:3, :3], atol=1e-6)
        np.testing.assert_allclose(q[2, 3], p[2, 3], atol=1e-6)
        uv = lambda x: (K @ x)[:2] / x[2]
        np.testing.assert_allclose(uv(q[:3, 3]) - uv(p[:3, 3]), [3.0, -2.5], atol=1e-3)
    np.testing.assert_allclose(rec.object_pose[3], np.eye(4))


def test_object_mask_colours(tmp_path: Path) -> None:
    reader = HOI4DReader(write_tree(tmp_path / "hoi4d"))
    obj, other = reader.object_mask(SEQ, 0)
    assert obj.any() and other[:30, :40].all() and not (obj & other).any()
    assert reader.mask_dir(SEQ).name == "mask" and reader.mask_dir(SEQ_PADDED).name == "shift_mask"


def test_hand_track_reads_the_hawor_format(tmp_path: Path) -> None:
    n = 6
    rng = np.random.default_rng(0)
    arrays = {"right_rot": rng.normal(size=(n, 3)), "right_hand_pose": rng.normal(size=(n, 45)),
              "right_trans": rng.normal(size=(n, 3)), "right_joints": rng.normal(size=(n, 21, 3)),
              "right_betas": np.tile(rng.normal(size=10), (n, 1)), "right_valid": np.array([0, 1, 1, 1, 0, 1], bool)}
    np.savez(tmp_path / "h.npz", **{k: v.astype(np.float32) if v.dtype != bool else v for k, v in arrays.items()})
    track = hand_track(tmp_path / "h.npz", n)
    track.validate(n, "hand_right")
    np.testing.assert_allclose(track.params[:, :3], arrays["right_rot"], atol=1e-6)
    np.testing.assert_allclose(track.params[:, 3:], arrays["right_hand_pose"], atol=1e-6)
    assert track.valid.tolist() == arrays["right_valid"].tolist()
    with pytest.raises(ValueError):
        hand_track(tmp_path / "h.npz", n + 1)


def test_known_answer_gt_pose_reprojects_onto_its_own_mask(tmp_path: Path) -> None:
    """KA: the GT pose projected with K onto a mask rendered from the same pose: IoU 1, no shift, 0 mm."""
    reader = HOI4DReader(write_tree(tmp_path / "hoi4d"))
    mesh = box()
    pose, ok = reader.objpose(SEQ, 2)
    assert ok
    obj, other = reader.object_mask(SEQ, 2)
    sil = silhouette(np.asarray(mesh.vertices), np.asarray(mesh.faces), pose, reader.intrinsics(SEQ), HW)
    chk = check_frame(sil, obj, other, float(pose[2, 3]), K[0, 0])
    assert chk.iou_raw == 1.0 and chk.iou_occ == 1.0 and chk.shift_px == (0, 0) and chk.lateral_mm == 0.0


def test_check_frame_recovers_a_known_shift_in_mm() -> None:
    sil = np.zeros((200, 300), bool)
    sil[60:120, 100:180] = True
    obj = np.roll(np.roll(sil, -4, axis=0), 9, axis=1)
    other = np.zeros_like(sil)
    other[60:80, 90:200] = True                   # an occluder over the object's top rows
    obj &= ~other
    chk = check_frame(sil, obj, other, depth_m=0.8, focal_px=1000.0)
    assert chk.shift_px == (-4, 9)
    assert chk.iou_best == pytest.approx(1.0)
    assert chk.lateral_mm == pytest.approx(np.hypot(4, 9) * 0.8, rel=1e-9)
    assert chk.iou_occ > chk.iou_raw               # the occluder no longer counts against the pose
    with pytest.raises(ValueError):
        check_frame(sil, obj | other, other, 0.8, 1000.0)


def test_release_list_is_read_with_crlf(tmp_path: Path) -> None:
    root = write_tree(tmp_path / "hoi4d")
    assert b"\r\n" in (root / "meta" / "release.txt").read_bytes()
    assert json.loads((root / "HOI4D_annotations" / SEQ / "action" / "color.json").read_text())["events"]
