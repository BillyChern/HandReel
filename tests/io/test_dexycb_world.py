"""The DexYCB extrinsics world frame (`fastdo.io.datasets.dexycb_world`).

Convention: `calibration/extrinsics_<id>/extrinsics.yml` holds, per camera serial, the 3x4 camera-to-master transform
(the master camera's entry is the identity) and, under `apriltag`, the tag-to-master transform. The world frame is the
table's AprilTag frame, `T_world_cam = T_master_tag^-1 T_master_cam`. The synthetic tests check the algebra on known
poses; the real-data test checks the convention itself on subject 03's calibration: the grasped object's GT pose from
all 8 views lands on one world pose, every camera sits above the table plane (z up) and the object rests on z = 0 at
frame 0. The world frame is not D6's static-camera fallback (`gravity_aligned_cam_to_world([0, -1, 0])`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from fastdo.io.datasets.dexycb import load_dexycb_yaml
from fastdo.io.datasets.dexycb_world import (
    dexycb_cam_to_world,
    load_dexycb_extrinsics,
    select_rows,
    with_world_frame,
)
from fastdo.qc.export import DEFAULT_GRAVITY_CAM, gravity_aligned_cam_to_world
from tests.policy.dexycb_data import (
    DEXYCB_ROOT,
    SEQ_RIGHT,
    SERIAL,
    assert_not_reserved,
    dexycb_missing,
)
from tests.qc.records import hand, make_record

logger = logging.getLogger(__name__)

SUBJECT, SEQUENCE, EXT_ID = "20200101-subject-00", "20200101_000000", "20200101_000000"
SERIALS = ("111", "222", "333")
TOL = 1e-10


def _h(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, trans
    return out


def _write_calibration(root: Path, cam_to_master: dict[str, np.ndarray], tag_to_master: np.ndarray) -> None:
    """`extrinsics.yml` in DexYCB's layout: `!!python/tuple` rows of 12 floats, written so they parse back exactly."""
    lines = ["extrinsics:"]
    for name, transform in {**cam_to_master, "apriltag": tag_to_master}.items():
        lines.append(f"  '{name}': !!python/tuple")
        lines += [f"  - {np.format_float_positional(v, unique=True)}" for v in transform[:3, :4].reshape(-1)]
    lines.append(f"master: '{SERIALS[0]}'")
    path = root / "calibration" / f"extrinsics_{EXT_ID}" / "extrinsics.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    seq_dir = root / SUBJECT / SEQUENCE
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "meta.yml").write_text(yaml.safe_dump({"serials": list(SERIALS), "num_frames": 2, "extrinsics": EXT_ID,
                                                      "ycb_ids": [1], "ycb_grasp_ind": 0}))


def test_extrinsics_world_frame_round_trips_known_poses(tmp_path: Path) -> None:
    """Known camera-to-world transforms and a known tag-to-master transform written as DexYCB calibration: the loader
    recovers every camera-to-world within 1e-10, maps object-to-camera poses of known world poses back onto them and
    round-trips world -> camera -> world, all within 1e-10. A sequence without extrinsics raises."""
    rng = np.random.default_rng(0)
    cam_to_world = {s: _h(Rotation.random(random_state=i).as_matrix(), rng.normal(size=3))
                    for i, s in enumerate(SERIALS)}
    cam_to_world[SERIALS[0]] = np.linalg.inv(_h(Rotation.random(random_state=9).as_matrix(), rng.normal(size=3)))
    tag_to_master = np.linalg.inv(cam_to_world[SERIALS[0]])             # the master camera's own entry is identity
    cam_to_master = {s: tag_to_master @ c2w for s, c2w in cam_to_world.items()}
    _write_calibration(tmp_path, cam_to_master, tag_to_master)
    world_obj = np.stack([_h(Rotation.random(random_state=20 + i).as_matrix(), rng.normal(size=3)) for i in range(3)])

    cams, tag = load_dexycb_extrinsics(tmp_path, EXT_ID)
    assert set(cams) == set(SERIALS) and np.abs(tag - tag_to_master).max() <= TOL
    assert np.abs(cams[SERIALS[0]] - np.eye(4)).max() <= TOL
    for serial in SERIALS:
        c2w = dexycb_cam_to_world(tmp_path, f"{SUBJECT}/{SEQUENCE}/{serial}")
        assert c2w.dtype == np.float64 and np.abs(c2w - cam_to_world[serial]).max() <= TOL
        obj_cam = np.linalg.inv(cam_to_world[serial]) @ world_obj
        assert np.abs(c2w @ obj_cam - world_obj).max() <= TOL
        assert np.abs(np.linalg.inv(c2w) @ (c2w @ obj_cam) - obj_cam).max() <= TOL
    (tmp_path / SUBJECT / "20200101_000001").mkdir()
    (tmp_path / SUBJECT / "20200101_000001" / "meta.yml").write_text(yaml.safe_dump({"serials": ["111"]}))
    with pytest.raises(ValueError, match="extrinsics"):
        dexycb_cam_to_world(tmp_path, f"{SUBJECT}/20200101_000001/111")


def test_with_world_frame_sets_the_camera_and_its_up_vector() -> None:
    """The record's camera becomes the static extrinsics transform (float32 storage) and `gravity_cam` the world +z
    axis in the camera frame; object poses, hands and ids are untouched and the input record is not modified. A
    transform that is not (4,4) raises."""
    record = make_record(np.arange(4), seed=1, hand_right=hand(4, 3))
    before = record.cam_to_world.copy()
    c2w = _h(Rotation.from_rotvec([0.3, -1.2, 0.5]).as_matrix(), np.array([0.4, -0.2, 0.9]))
    out = with_world_frame(record, c2w)
    out.validate()
    assert out.cam_to_world.dtype == np.float32 and np.array_equal(out.cam_to_world, np.tile(c2w.astype(np.float32),
                                                                                             (4, 1, 1)))
    assert np.abs(c2w[:3, :3] @ out.gravity_cam.astype(np.float64) - [0.0, 0.0, 1.0]).max() <= 1e-6
    assert np.array_equal(out.object_pose, record.object_pose) and out.hand_right is record.hand_right
    assert out.clip_id == record.clip_id and np.array_equal(record.cam_to_world, before)
    with pytest.raises(ValueError, match="4, 4"):
        with_world_frame(record, np.eye(3))


def test_select_rows_keeps_the_engine_rows() -> None:
    """Every per-row array (camera, pose, confidence, time, source index, hand parameters, joints, validity) keeps the
    selected rows; per-clip fields (betas, intrinsics, scale) stay; rows that do not increase raise."""
    record = make_record(np.arange(10), seed=2, hand_right=hand(10, 3))
    rows = np.arange(0, 10, 3)
    out = select_rows(record, rows)
    out.validate()
    for name in ("frame_idx", "timestamps", "cam_to_world", "object_pose", "confidence"):
        assert np.array_equal(getattr(out, name), getattr(record, name)[rows]), name
    assert out.hand_right is not None and record.hand_right is not None
    for name in ("params", "trans", "joints", "valid"):
        assert np.array_equal(getattr(out.hand_right, name), getattr(record.hand_right, name)[rows]), name
    assert np.array_equal(out.hand_right.betas, record.hand_right.betas) and out.hand_left is None
    assert np.array_equal(out.intrinsics, record.intrinsics) and out.object_scale == record.object_scale
    with pytest.raises(ValueError, match="increase"):
        select_rows(record, np.array([0, 3, 3]))


@dexycb_missing
def test_world_frame_convention_on_subject_03_calibration() -> None:
    """Real calibration (subject 03, 20200820_135250): at frames 0 and 40 the grasped object's GT pose from all 8 views
    agrees in the world frame within 1e-5 (float32 labels); every camera is above the table (z > 0.05 m); the object's
    lowest vertex at frame 0 is within 5 mm of the table plane z = 0; the frame differs from D6's fallback."""
    assert DEXYCB_ROOT is not None
    assert_not_reserved(SEQ_RIGHT)
    meta = load_dexycb_yaml(DEXYCB_ROOT / SEQ_RIGHT / "meta.yml")
    grasp = int(meta["ycb_grasp_ind"])
    heights = {}
    for frame in (0, 40):
        world = []
        for serial in meta["serials"]:
            c2w = dexycb_cam_to_world(DEXYCB_ROOT, f"{SEQ_RIGHT}/{serial}")
            heights[serial] = float(c2w[2, 3])
            with np.load(DEXYCB_ROOT / SEQ_RIGHT / serial / f"labels_{frame:06d}.npz") as label:
                world.append(c2w @ np.vstack([label["pose_y"][grasp].astype(np.float64), [0.0, 0.0, 0.0, 1.0]]))
        poses = np.stack(world)
        spread = float(np.abs(poses - poses[0]).max())
        logger.info("frame %d: world pose spread over %d views %.3g", frame, len(poses), spread)
        assert spread <= 1e-5
        if frame == 0:
            mesh = trimesh.load(DEXYCB_ROOT / "models" / "002_master_chef_can" / "textured_simple.obj", force="mesh")
            assert isinstance(mesh, trimesh.Trimesh)
            verts = np.asarray(mesh.vertices) @ poses[0, :3, :3].T + poses[0, :3, 3]
            logger.info("frame 0: object z range [%.4f, %.4f] m", verts[:, 2].min(), verts[:, 2].max())
            assert abs(float(verts[:, 2].min())) <= 5e-3
    logger.info("camera heights above the table: %s", {s: round(h, 3) for s, h in heights.items()})
    assert min(heights.values()) > 0.05
    c2w = dexycb_cam_to_world(DEXYCB_ROOT, f"{SEQ_RIGHT}/{SERIAL}")
    fallback = gravity_aligned_cam_to_world(torch.from_numpy(DEFAULT_GRAVITY_CAM), 1)[0].numpy()
    angle = float(np.linalg.norm(Rotation.from_matrix(c2w[:3, :3].T @ fallback[:3, :3]).as_rotvec()))
    logger.info("extrinsics frame vs D6 fallback: rotation %.3f rad, translation %.3f m", angle,
                float(np.linalg.norm(c2w[:3, 3])))
    assert np.abs(c2w - fallback).max() > 0.1
