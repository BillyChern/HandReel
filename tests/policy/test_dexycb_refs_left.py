"""DexYCB policy references on a left-hand DexYCB clip: `fastdo.policy.dexycb_refs.make_reference_fn`.

Clip: subject 09, 20201015_143556 (005_tomato_soup_can, left hand); its GT hands pass the ported regtracks
validation. Robot scene: a left Sharpa scene built by `run/pipeline/build_ycb_scenes.py` from a synthetic box, recorded
as the object's left scene. The robot's kinematics and collision primitives come from the scene; the object's geometry
comes from its YCB mesh.

What is checked:
- the reference is kept, on the left hand, through D6's `run_retarget_stage`, with the IK-site wrist target that D6
  computes with `is_right=False`. The hand side changes the target's fixed offset `R_off`, so the flag is load-bearing;
- world frame: every row's camera-to-world is the DexYCB extrinsics transform (stored float32, as `ClipRecord`
  requires), not D6's static-camera fallback. At the engine key frames, the solver's wrist target is that stored
  transform applied to the camera-frame IK-site target, placed at the left wrist joint, and the reference's object pose
  is the GT world pose (float64 extrinsics), all within 1e-6;
- clearance is reported, not asserted (the measured left clip penetrates; a follow-up investigation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

import fastdo.qc.export as export_module
from fastdo.io.datasets.dexycb import DexYCBReader, load_dexycb_yaml
from fastdo.io.datasets.dexycb_world import dexycb_cam_to_world
from fastdo.policy.dexycb_inputs import write_scene_record
from fastdo.policy.dexycb_refs import DexYCBRefConfig, make_reference_fn
from fastdo.qc.export import gravity_aligned_cam_to_world
from fastdo.retarget.wrist_target import wrist_frame_offset
from run.pipeline.build_dexycb_hands import HandsBuildConfig, build_hands
from run.pipeline.build_ycb_scenes import SceneBuildConfig, build_object_scene
from tests.policy.dexycb_data import (
    BASELINE_ROOT,
    DEXYCB_ROOT,
    HAWOR_PYTHON,
    HAWOR_ROOT,
    MANO_LEFT_PKL,
    MANO_PCA,
    MANO_RIGHT_PKL,
    RETARGETING_PYTHON,
    SEQ_LEFT,
    SERIAL,
    assert_not_reserved,
)

logger = logging.getLogger(__name__)
OBJECT, STRIDE, SOURCE_HZ, HZ = "005_tomato_soup_can", 3, 30.0, 50.0


@pytest.fixture(scope="module")
def left_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """`(scenes_root, hands_root)`: the clip's GT left hand and a validated left Sharpa scene recorded for its object."""
    needed = (DEXYCB_ROOT, MANO_PCA, HAWOR_PYTHON, HAWOR_ROOT, MANO_RIGHT_PKL, MANO_LEFT_PKL, RETARGETING_PYTHON,
              BASELINE_ROOT)
    if any(p is None for p in needed):
        pytest.skip("DexYCB data, the hawor or retargeting env, the baseline checkout or MANO models are not on this host")
    assert_not_reserved(SEQ_LEFT)
    root = tmp_path_factory.mktemp("dexycb_refs_left")
    summary = build_hands([SEQ_LEFT], HandsBuildConfig(
        out_root=str(root / "refs"), dexycb_root=str(DEXYCB_ROOT), mano_pca=str(MANO_PCA),
        hawor_python=str(HAWOR_PYTHON), hawor_root=str(HAWOR_ROOT), mano_right_pkl=str(MANO_RIGHT_PKL),
        mano_left_pkl=str(MANO_LEFT_PKL)))
    assert summary["n_pass"] == 1 and summary["sides"] == {"left": 1, "right": 0}, summary
    mesh = root / "box.obj"
    trimesh.creation.box(extents=(0.08, 0.06, 0.04)).export(mesh)
    scene = build_object_scene("box", mesh, "left", SceneBuildConfig(
        out_root=str(root / "box_scene"), baseline_root=str(BASELINE_ROOT), python=str(RETARGETING_PYTHON)))
    assert scene["pass"] and scene["side"] == "left", scene.get("validation") or scene.get("error")
    write_scene_record(root / "scenes", {"object": OBJECT, "side": "left", "scene_xml": scene["scene_xml"], "pass": True,
                                         "note": "left Sharpa scene of a synthetic box as a stand-in (test)"})
    return root / "scenes", root / "refs"


def test_left_clip_reference_uses_the_left_ik_site_target_in_the_extrinsics_world_frame(
        left_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEXYCB_ROOT is not None and MANO_PCA is not None
    scenes_root, hands_root = left_inputs
    clip = f"{SEQ_LEFT}/{SERIAL}"
    record = DexYCBReader(DEXYCB_ROOT, MANO_PCA, subjects=("subject-09",)).load(clip)
    assert record.hand_right is None and record.hand_left is not None
    targets: list[tuple[Any, ...]] = []
    solves: list[tuple[Any, Any]] = []
    real_target, real_prepare = export_module.wrist_target_poses, export_module.prepare_retarget_stage

    def target(name: str, joints: Any, rotvec: Any, is_right: bool = True, valid: Any = None) -> Any:
        out = real_target(name, joints, rotvec, is_right, valid)
        targets.append((name, is_right, joints, rotvec, valid, out))
        return out

    def stage_prepare(engine: Any, *args: Any, **kwargs: Any) -> Any:  # the prepare half inside run_retarget_stage
        out = real_prepare(engine, *args, **kwargs)
        solves.append((engine, out))
        return out

    monkeypatch.setattr(export_module, "wrist_target_poses", target)
    monkeypatch.setattr(export_module, "prepare_retarget_stage", stage_prepare)
    fn = make_reference_fn(scenes_root, hands_root, DexYCBRefConfig(
        dexycb_root=str(DEXYCB_ROOT), mano_right_pkl=str(MANO_RIGHT_PKL), mano_left_pkl=str(MANO_LEFT_PKL), hz=HZ,
        engine_stride=STRIDE))
    ref = fn(record)
    kept = fn.records[-1]
    logger.info("left reference record: %s", kept)
    assert ref is not None and kept["status"] == "kept" and (kept["object"], kept["side"]) == (OBJECT, "left")
    assert kept["wrist_target"] == "ik_site" and kept["world_frame"] == "dexycb_apriltag"
    n_rows, n_full = len(range(0, record.n_frames, STRIDE)), (record.n_frames - 1) // STRIDE * STRIDE + 1
    n_policy = int(np.floor((n_full - 1) * HZ / SOURCE_HZ + 1e-9)) + 1
    assert (kept["n_rows"], kept["n_frames"], kept["n_policy_frames"], ref.num_frames) == (n_rows, n_full, n_policy,
                                                                                          n_policy)

    # one target per prepare call: inside run_retarget_stage (the solver's inputs), then the reference function's own
    assert [(t[0], t[1]) for t in targets] == [("ik_site", False), ("ik_site", False)]
    assert bool((targets[0][-1] == targets[1][-1]).all())
    _, _, joints, rotvec, valid, wrist_cam = targets[0]
    offset_left = wrist_frame_offset(joints, rotvec, is_right=False, valid=valid)
    offset_right = wrist_frame_offset(joints, rotvec, is_right=True, valid=valid)
    logger.info("left R_off vs right-hand R_off on the same rows: max |diff| %.3f", np.abs(offset_left - offset_right).max())
    assert np.abs(offset_left - offset_right).max() > 0.1

    (engine, inputs), = solves
    c2w = np.asarray(dexycb_cam_to_world(DEXYCB_ROOT, clip), dtype=np.float64)
    stored = c2w.astype(np.float32)                       # ClipRecord.cam_to_world is float32 (T, 4, 4)
    assert engine.cam_to_world.dtype == np.float32 and bool((engine.cam_to_world == stored[None]).all())
    c2w_stage = stored.astype(np.float64)                 # what the retarget stage upcasts and applies
    fallback = gravity_aligned_cam_to_world(np.array([0.0, -1.0, 0.0]), 1)[0].double().numpy()
    assert np.abs(c2w - fallback).max() > 0.1
    meta = load_dexycb_yaml(DEXYCB_ROOT / SEQ_LEFT / "meta.yml")
    grasp = int(meta["ycb_grasp_ind"])
    wrist_err, joint_err, pose_err = [], [], []
    for j, source in enumerate(range(0, n_full, STRIDE)):
        solved_wrist = inputs.wrist_poses[STRIDE * j].double().numpy()
        wrist_err.append(float(np.abs(solved_wrist - c2w_stage @ wrist_cam[j].double().numpy()).max()))
        wrist_joint = (c2w_stage[:3, :3] @ np.asarray(record.hand_left.joints[source, 0], dtype=np.float64)
                       + c2w_stage[:3, 3])
        joint_err.append(float(np.abs(solved_wrist[:3, 3] - wrist_joint).max()))
        with np.load(DEXYCB_ROOT / SEQ_LEFT / SERIAL / f"labels_{source:06d}.npz") as label:
            gt = c2w @ np.vstack([label["pose_y"][grasp].astype(np.float64), [0.0, 0.0, 0.0, 1.0]])
        pose_err.append(float(np.abs(ref.obj_pose[5 * j].double().numpy() - gt).max()))
    logger.info("key frames %d: wrist target vs world IK-site target %.3g, vs left wrist joint %.3g, object vs GT %.3g",
                len(pose_err), max(wrist_err), max(joint_err), max(pose_err))
    assert max(wrist_err) <= 1e-6 and max(joint_err) <= 1e-6 and max(pose_err) <= 1e-6
    clearance = kept["clearance"]
    logger.info("left clip clearance (reported, not asserted): min %.3f mm, %.3f of frames >= 0.5 mm, "
                "%.3f penetrating", clearance["frame_min_mm_min"], clearance["frac_frames_clear_0p5mm"],
                clearance["frac_frames_penetrating"])
