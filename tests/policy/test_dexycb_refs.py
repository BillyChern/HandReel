"""DexYCB policy references on real clips: `fastdo.policy.dexycb_refs.make_reference_fn` (subject 03, right hand).

Inputs. Each sequence's GT hands come from `run/pipeline/build_dexycb_hands.py` (hawor env, built once per module).
The robot scene recorded for every object is the shipped whisking Sharpa scene: the kinematic reference and the
clearance statistics read the robot's kinematics and collision primitives from the scene and the object's geometry from
its YCB mesh (SDF and surface samples), so the whisk's convex parts in that scene take no part. The per-object scene
build itself is tested in `tests/run/test_build_ycb_scenes.py`.

What is checked:
- the PolicyReference's object pose equals the GT world pose (DexYCB extrinsics frame) at the engine key frames within
  1e-6, and its length is the policy-rate length of the upsampled full-rate reference (stride-3 engine rows 0..69 ->
  70 source frames at 30 Hz -> 116 frames at 50 Hz);
- spies: the reference function calls the composed `run_retarget_stage` once and never the solve
  half on its own. The kinematic solve inside it runs once, at the full rate, on inputs whose splines only ever see the
  24 engine knots of positions and quaternions; the trajectory handed to the policy-rate resampling is that call's own
  output, so robot joint references are never spline-upsampled. The reference-rate object poses and frame mask come
  from the same inputs as the solve's;
- `build_reference_clips` with the real function on 4 objects and `n_held_out = 1` never lets the held-out object into
  training; clearance is reported per object and hand side (Amendment 1: not gated on 1 mm).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fastdo.qc.export as export_module
from fastdo.io.datasets.dexycb import DexYCBReader
from fastdo.io.datasets.dexycb_world import dexycb_cam_to_world
from fastdo.policy import dexycb_refs
from fastdo.policy.data import ReferenceSetConfig, build_reference_clips, object_split
from fastdo.policy.dexycb_inputs import write_scene_record
from fastdo.policy.dexycb_refs import DexYCBRefConfig, clearance_report, make_reference_fn
from fastdo.retarget import rate
from run.pipeline.build_dexycb_hands import HandsBuildConfig, build_hands
from tests.policy.dexycb_data import (
    DEXYCB_ROOT,
    FOUR_OBJECT_SEQS,
    HAWOR_PYTHON,
    HAWOR_ROOT,
    MANO_LEFT_PKL,
    MANO_PCA,
    MANO_RIGHT_PKL,
    SEQ_RIGHT,
    SERIAL,
    WHISKING_SCENE_XML,
    assert_not_reserved,
)

logger = logging.getLogger(__name__)
STRIDE, SOURCE_HZ, HZ = 3, 30.0, 50.0


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """`(scenes_root, hands_root)`: GT hands of the four sequences and a validated scene record per object."""
    needed = (DEXYCB_ROOT, MANO_PCA, HAWOR_PYTHON, HAWOR_ROOT, MANO_RIGHT_PKL, MANO_LEFT_PKL, WHISKING_SCENE_XML)
    if any(p is None for p in needed):
        pytest.skip("DexYCB data, the hawor env, MANO models or the whisking scene are not on this host")
    root = tmp_path_factory.mktemp("dexycb_refs")
    for seq, _ in FOUR_OBJECT_SEQS:
        assert_not_reserved(seq)
    summary = build_hands([seq for seq, _ in FOUR_OBJECT_SEQS], HandsBuildConfig(
        out_root=str(root / "refs"), dexycb_root=str(DEXYCB_ROOT), mano_pca=str(MANO_PCA),
        hawor_python=str(HAWOR_PYTHON), hawor_root=str(HAWOR_ROOT), mano_right_pkl=str(MANO_RIGHT_PKL),
        mano_left_pkl=str(MANO_LEFT_PKL)))
    assert summary["n_pass"] == len(FOUR_OBJECT_SEQS), summary["failed"]
    for _, name in FOUR_OBJECT_SEQS:
        write_scene_record(root / "scenes", {"object": name, "side": "right", "scene_xml": str(WHISKING_SCENE_XML),
                                             "pass": True, "note": "whisking robot scene as a stand-in (test)"})
    return root / "scenes", root / "refs"


def _cfg() -> DexYCBRefConfig:
    return DexYCBRefConfig(dexycb_root=str(DEXYCB_ROOT), mano_right_pkl=str(MANO_RIGHT_PKL),
                           mano_left_pkl=str(MANO_LEFT_PKL), hz=HZ, engine_stride=STRIDE)


def _reader() -> DexYCBReader:
    assert DEXYCB_ROOT is not None and MANO_PCA is not None
    return DexYCBReader(DEXYCB_ROOT, MANO_PCA, subjects=("subject-03",))


def test_reference_fn_on_one_sequence(inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEXYCB_ROOT is not None
    scenes_root, hands_root = inputs
    record = _reader().load(f"{SEQ_RIGHT}/{SERIAL}")
    calls: dict[str, list[Any]] = {"spline": [], "quaternion": [], "kinematic": [], "resample": [], "run": [],
                                   "stage_prepare": [], "own_prepare": []}
    real_spline, real_quat = rate.natural_spline, rate.quaternion_spline
    real_kinematic, real_resample = export_module.retarget_kinematic, dexycb_refs.resample_reference
    real_run, real_prepare = dexycb_refs.run_retarget_stage, export_module.prepare_retarget_stage

    def spline(values: np.ndarray, keys: np.ndarray, out: np.ndarray) -> np.ndarray:
        calls["spline"].append(np.asarray(values).shape)
        return real_spline(values, keys, out)

    def quaternion(values: np.ndarray, keys: np.ndarray, out: np.ndarray) -> np.ndarray:
        calls["quaternion"].append(np.asarray(values).shape)
        return real_quat(values, keys, out)

    def kinematic(*args: Any, **kwargs: Any) -> Any:
        out = real_kinematic(*args, **kwargs)
        calls["kinematic"].append((tuple(args[0].shape), float(args[9]), out))
        return out

    def resample(kin: Any, *args: Any, **kwargs: Any) -> Any:
        calls["resample"].append(kin)
        return real_resample(kin, *args, **kwargs)

    def run_stage(*args: Any, **kwargs: Any) -> Any:
        out = real_run(*args, **kwargs)
        calls["run"].append((args, kwargs, out))
        return out

    def stage_prepare(*args: Any, **kwargs: Any) -> Any:           # the prepare half inside run_retarget_stage
        out = real_prepare(*args, **kwargs)
        calls["stage_prepare"].append(out)
        return out

    def own_prepare(*args: Any, **kwargs: Any) -> Any:             # the reference function's own inputs
        out = real_prepare(*args, **kwargs)
        calls["own_prepare"].append(out)
        return out

    monkeypatch.setattr(rate, "natural_spline", spline)
    monkeypatch.setattr(rate, "quaternion_spline", quaternion)
    monkeypatch.setattr(export_module, "retarget_kinematic", kinematic)
    monkeypatch.setattr(dexycb_refs, "resample_reference", resample)
    monkeypatch.setattr(dexycb_refs, "run_retarget_stage", run_stage)
    monkeypatch.setattr(export_module, "prepare_retarget_stage", stage_prepare)
    monkeypatch.setattr(dexycb_refs, "prepare_retarget_stage", own_prepare)
    fn = make_reference_fn(scenes_root, hands_root, _cfg())
    ref = fn(record)

    assert ref is not None
    n_rows, n_full = len(range(0, record.n_frames, STRIDE)), (record.n_frames - 1) // STRIDE * STRIDE + 1
    assert (n_rows, n_full) == (24, 70)
    assert [shape for shape, _, _ in calls["kinematic"]] == [(n_full, 4, 4)]
    assert calls["kinematic"][0][1] == pytest.approx(SOURCE_HZ, rel=1e-12)
    solved = calls["kinematic"][0][2]
    assert len(calls["resample"]) == 1 and calls["resample"][0] is solved and solved.qpos.shape == (n_full, 35)
    # one composed run_retarget_stage call produced that trajectory; the solve half is never called alone
    assert not hasattr(dexycb_refs, "solve_retarget_stage") and len(calls["run"]) == 1
    run_args, run_kwargs, run_out = calls["run"][0]
    assert run_out is solved and run_kwargs == {"side": "right", "wrist_target": "ik_site"}
    assert run_args[0].n_frames == n_rows and tuple(run_args[1].shape) == (n_rows, 778, 3)
    assert run_args[7] == pytest.approx(SOURCE_HZ, rel=1e-12) and run_args[8] is fn.cfg.retarget
    (stage_inputs,), (own_inputs,) = calls["stage_prepare"], calls["own_prepare"]
    assert stage_inputs is not own_inputs and own_inputs.n_frames == stage_inputs.n_frames == n_full
    for key in ("obj_poses", "wrist_poses", "hand_verts", "frame_valid"):
        assert bool((getattr(own_inputs, key) == getattr(stage_inputs, key)).all()), key
    # every spline sees the engine knots of positions (3 columns) or of quaternions (4, routed through natural_spline),
    # never a joint array (35 qpos, 28 robot or 22 finger columns)
    assert calls["spline"] and all(len(s) == 2 and s[0] == n_rows and s[1] in (3, 4) for s in calls["spline"])
    assert calls["quaternion"] and all(s == (n_rows, 4) for s in calls["quaternion"])
    n_policy = int(np.floor((n_full - 1) * HZ / SOURCE_HZ + 1e-9)) + 1
    assert n_policy == 116 and ref.num_frames == n_policy and ref.hz == HZ and tuple(ref.q_ref.shape) == (n_policy, 35)
    assert tuple(ref.contact_plan.shape) == (n_policy, 6) and ref.covariance is not None
    assert tuple(ref.covariance.shape) == (n_policy, 7, 7) and ref.log_scale == 0.0
    c2w = dexycb_cam_to_world(DEXYCB_ROOT, f"{SEQ_RIGHT}/{SERIAL}")
    grasp = 0
    errors = []
    for j, source in enumerate(range(0, n_full, STRIDE)):
        with np.load(DEXYCB_ROOT / SEQ_RIGHT / SERIAL / f"labels_{source:06d}.npz") as label:
            gt = c2w @ np.vstack([label["pose_y"][grasp].astype(np.float64), [0.0, 0.0, 0.0, 1.0]])
        errors.append(float(np.abs(ref.obj_pose[5 * j].double().numpy() - gt).max()))
    logger.info("object pose vs GT world pose at %d key frames: max |diff| %.3g", len(errors), max(errors))
    assert max(errors) <= 1e-6
    kept = fn.records[-1]
    logger.info("reference record: %s", kept)
    assert kept["status"] == "kept" and kept["object"] == "002_master_chef_can" and kept["side"] == "right"
    assert (kept["n_rows"], kept["n_frames"], kept["n_policy_frames"]) == (n_rows, n_full, n_policy)
    assert kept["world_frame"] == "dexycb_apriltag" and kept["wrist_target"] == "ik_site"
    assert {"frame_min_mm_min", "frac_frames_clear_0p5mm", "frac_frames_penetrating"} <= set(kept["clearance"])


def test_build_reference_set_with_the_real_reference_fn_never_trains_on_the_held_out_object(
        inputs: tuple[Path, Path]) -> None:
    assert DEXYCB_ROOT is not None
    scenes_root, hands_root = inputs
    objects = [name for _, name in FOUR_OBJECT_SEQS]
    train_objects, held_out = object_split(objects, 1, seed=0)
    fn = make_reference_fn(scenes_root, hands_root, _cfg())
    base = ReferenceSetConfig(held_out_objects=held_out, split="train", reference_fn=fn, hand_side="right",
                              skip_reason=fn.skip_reason)
    train = build_reference_clips(DEXYCB_ROOT, objects, 0, base, reader=_reader())
    held = build_reference_clips(DEXYCB_ROOT, objects, 0, dataclasses.replace(base, split="held_out"), reader=_reader())
    logger.info("train %s, held out %s", [(c.clip_id, c.object_name) for c in train],
                [(c.clip_id, c.object_name) for c in held])
    assert sorted(c.object_name for c in train) == sorted(train_objects) and not {c.object_name for c in train} & set(
        held_out)
    assert [c.object_name for c in held] == list(held_out)
    assert {c.clip_id.rsplit("/", 1)[0] for c in [*train, *held]} == {seq for seq, _ in FOUR_OBJECT_SEQS}
    report = clearance_report(fn.records)
    logger.info("per-object clearance (not gated): %s", report)
    assert set(report) == {f"{o}/right" for o in objects}
    for name in objects:
        row = report[f"{name}/right"]
        assert (row["object"], row["side"], row["n_clips"]) == (name, "right", 1)
        assert 0.0 <= row["frac_frames_clear_0p5mm"] <= 1.0
