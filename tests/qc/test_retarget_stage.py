"""Tests for the retargeting stage hook: rate composition, masking, the wrist target and input errors before the
solve.

- Rate composition: the INPUTS go from the engine rate to the reference rate (one reference frame per source frame
  from the first to the last engine row) through `fastdo.retarget.rate`: object and wrist SE(3) by C2 natural splines
  on sign-continuous quaternions, hand vertices lerped per vertex in the wrist frame. The kinematic reference is then
  solved once, at full rate, on them; its joints are never upsampled.
- Masking: hand rows with `valid` False (the hands stage fills them with finite copies of the nearest valid row) are
  not spline knots, and every reference frame whose bracketing engine rows include one reaches the solver as
  `frame_valid` False.
- Wrist target: the IK-site target by default, the MCP geometric frame on request (`fastdo.retarget.wrist_target`).
- A non-finite input on a row the stage uses raises ValueError before any solve.

CPU only, with a capturing `retarget_kinematic` stand-in.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor

import fastdo.qc.export as export_module
from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.qc.export import points_to_world, pose_to_world, retarget_hand_side, run_retarget_stage
from fastdo.qc.reference_rate import (
    reference_frame_valid,
    reference_key_frames,
    upsample_reference_inputs,
)
from fastdo.retarget.kinematic import RetargetConfig
from fastdo.retarget.rate import upsample_hand_track, upsample_object_poses
from fastdo.retarget.wrist_target import wrist_target_poses
from tests.qc.records import (
    CapturingKinematic,
    hand,
    hand_labels,
    hand_verts,
    make_record,
    tiny_sdf,
)

SCENE = cast(Any, object())


def _run(record: ClipRecord, verts: Tensor, monkeypatch: pytest.MonkeyPatch, frequency: float = 30.0,
         side: str | None = None, wrist_target: str | None = None) -> tuple[CapturingKinematic, Any]:
    """`run_retarget_stage` with a capturing solver; `wrist_target=None` leaves the stage's default in place."""
    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)
    target = {} if wrist_target is None else {"wrist_target": wrist_target}
    out = run_retarget_stage(record, verts, hand_labels(), SCENE, tiny_sdf(), torch.zeros(8, 3), torch.ones(8, 3),
                             frequency, RetargetConfig(), side=side, **target)
    return spy, out


def _world_rows(record: ClipRecord, track: HandTrack, verts: Tensor, is_right: bool,
                wrist_target: str = "ik_site") -> tuple[Tensor, Tensor, Tensor]:
    c2w = torch.from_numpy(record.cam_to_world).double()
    obj = pose_to_world(c2w, torch.from_numpy(record.object_pose).double())
    wrist_cam = wrist_target_poses(wrist_target, torch.from_numpy(track.joints).double(),
                                   torch.from_numpy(np.ascontiguousarray(track.params[:, :3])).double(),
                                   is_right=is_right, valid=track.valid)
    return obj, pose_to_world(c2w, wrist_cam), points_to_world(c2w, verts.double())


def test_reference_key_frames_and_frame_valid() -> None:
    keys = reference_key_frames(np.array([30, 33, 36, 39, 41]))
    assert keys.dtype == np.int64 and keys.tolist() == [0, 3, 6, 9, 11]
    for bad in (np.array([3, 3, 6]), np.array([6, 3]), np.array([], dtype=np.int64)):
        with pytest.raises(ValueError):
            reference_key_frames(bad)
    valid = np.array([True, True, False, True, True])
    expected = [True, True, True, True, False, False, False, False, False, True, True, True]
    assert reference_frame_valid(keys, valid).tolist() == expected
    assert reference_frame_valid(keys, np.ones(5, bool)).all() and reference_frame_valid(keys, np.ones(5, bool)).size == 12
    assert reference_frame_valid(np.arange(4), np.array([True, False, True, True])).tolist() == [True, False, True, True]
    with pytest.raises(ValueError):
        reference_frame_valid(keys, valid[:-1])


def test_run_retarget_stage_upsamples_inputs_to_the_reference_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_idx = np.array([30, 33, 36, 39, 42, 44])
    keys, n = frame_idx - 30, 15
    record = make_record(frame_idx, seed=5, hand_right=hand(6, 6))
    assert record.hand_right is not None
    verts_cam = hand_verts(6, 7)

    spy, out = _run(record, verts_cam, monkeypatch, frequency=29.97)

    (args, kwargs), = spy.calls
    assert out is spy.results[0] and out.qpos.shape == (n, 35)       # the full-rate solve's own output, untouched
    obj_rows, wrist_rows, verts_rows = _world_rows(record, record.hand_right, verts_cam, is_right=True)
    wrist_up, verts_up = upsample_hand_track(wrist_rows, verts_rows, keys, n)
    assert args[0].dtype == torch.float64 and args[0].shape == (n, 4, 4) and args[3].shape == (n, 778, 3)
    assert np.array_equal(args[0].numpy(), upsample_object_poses(obj_rows, keys, n))
    assert np.array_equal(args[1].numpy(), wrist_up) and np.array_equal(args[3].numpy(), verts_up)
    # Key rows reproduce the engine rows to 1e-6: the record's float32 rotations (object pose, cam_to_world) are
    # orthonormal only to ~1e-7, and the quaternion round trip re-orthonormalizes them (object and, through the wrist
    # frame, vertices; measured 4.7e-8 and 1.5e-7).
    assert (args[0][keys] - obj_rows).abs().max() <= 1e-6 and (args[3][keys] - verts_rows).abs().max() <= 1e-6
    assert args[9] == 29.97 and kwargs["frame_valid"].shape == (n,) and bool(kwargs["frame_valid"].all())


def test_run_retarget_stage_masks_invalid_hand_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_idx = 30 + 3 * np.arange(6)
    keys, n = frame_idx - 30, 16
    valid = np.array([True, True, False, False, True, True])
    track = hand(6, 8, valid=valid)
    joints, verts = track.joints.copy(), hand_verts(6, 9).numpy()
    joints[2], joints[3], verts[2], verts[3] = joints[1], joints[4], verts[1], verts[4]    # D3b: nearest valid row
    filled = dataclasses.replace(track, joints=joints)
    record = make_record(frame_idx, seed=10, hand_right=filled)

    spy, _ = _run(record, torch.from_numpy(verts), monkeypatch)

    (args, kwargs), = spy.calls
    expected_valid = reference_frame_valid(keys, valid)
    assert np.flatnonzero(~expected_valid).tolist() == list(range(4, 12))
    assert kwargs["frame_valid"].dtype == torch.bool and kwargs["frame_valid"].numpy().tolist() == expected_valid.tolist()
    _, wrist_rows, verts_rows = _world_rows(record, filled, torch.from_numpy(verts), is_right=True)
    rows = np.flatnonzero(valid)
    wrist_up, verts_up = upsample_hand_track(wrist_rows[rows], verts_rows[rows], keys[rows], n)    # valid knots only
    assert np.array_equal(args[1].numpy(), wrist_up) and np.array_equal(args[3].numpy(), verts_up)

    rng = np.random.default_rng(0)
    joints_bad, verts_bad, params_bad = joints.copy(), verts.copy(), track.params.copy()
    joints_bad[2:4] = 5.0 + rng.normal(size=(2, 21, 3))
    verts_bad[2:4] = -3.0 + rng.normal(size=(2, 778, 3))
    params_bad[2:4] = 9.0 * rng.normal(size=(2, 48))           # global orient too: the IK-site offset averages valid rows
    garbage = make_record(frame_idx, seed=10, hand_right=dataclasses.replace(
        track, joints=joints_bad.astype(np.float32), params=params_bad.astype(np.float32)))
    spy_bad, _ = _run(garbage, torch.from_numpy(verts_bad.astype(np.float32)), monkeypatch)
    (bad_args, bad_kwargs), = spy_bad.calls
    for i in (0, 1, 3):
        assert torch.equal(bad_args[i], args[i]), i
    assert torch.equal(bad_kwargs["frame_valid"], kwargs["frame_valid"])


def test_run_retarget_stage_holds_edges_and_skips_clips_without_valid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_idx = 3 * np.arange(6)
    valid = np.array([False, True, True, True, False, False])
    track = hand(6, 11, valid=valid)
    verts = hand_verts(6, 12)
    record = make_record(frame_idx, seed=13, hand_right=track)

    spy, _ = _run(record, verts, monkeypatch)

    (args, kwargs), = spy.calls
    assert np.flatnonzero(kwargs["frame_valid"].numpy()).tolist() == list(range(3, 10))
    _, wrist_rows, verts_rows = _world_rows(record, track, verts, is_right=True)
    wrist_up, verts_up = upsample_hand_track(wrist_rows[1:4], verts_rows[1:4], np.array([0, 3, 6]), 7)
    assert np.array_equal(args[1][3:10].numpy(), wrist_up) and np.array_equal(args[3][3:10].numpy(), verts_up)
    for frame in range(3):
        assert torch.equal(args[1][frame], args[1][3]) and torch.equal(args[3][frame], args[3][3])
    for frame in range(10, 16):
        assert torch.equal(args[1][frame], args[1][9]) and torch.equal(args[3][frame], args[3][9])
    inputs = upsample_reference_inputs(*_world_rows(record, track, verts, is_right=True), valid,
                                       reference_key_frames(frame_idx))
    assert (inputs.masked_rows, inputs.masked_frames, inputs.hand_verts.shape[0]) == (3, 9, 16)

    none_valid = make_record(frame_idx, seed=13, hand_right=hand(6, 11, valid=np.zeros(6, bool)))
    spy_none, out = _run(none_valid, verts, monkeypatch)
    assert out is None and spy_none.calls == []


def test_run_retarget_stage_selects_the_hand_side(monkeypatch: pytest.MonkeyPatch) -> None:
    t = 4
    left, right = hand(t, 20), hand(t, 21)
    both = make_record(np.arange(t), seed=22, hand_left=left, hand_right=right)
    verts = hand_verts(t, 23)
    assert retarget_hand_side(both, None) == ("right", right) and retarget_hand_side(both, "left") == ("left", left)
    for side, track, is_right in ((None, right, True), ("left", left, False)):
        spy, out = _run(both, verts, monkeypatch, side=side)
        (args, _), = spy.calls
        _, wrist_rows, _ = _world_rows(both, track, verts, is_right=is_right)
        assert out is spy.results[0] and (args[1] - wrist_rows).abs().max() <= 1e-12, side

    only_left = make_record(np.arange(t), seed=22, hand_left=left)
    assert retarget_hand_side(only_left, None) == ("left", left)
    spy, out = _run(only_left, verts, monkeypatch, side="right")
    assert out is None and spy.calls == []
    with pytest.raises(ValueError):
        retarget_hand_side(both, "middle")
    with pytest.raises(ValueError):
        _run(both, verts[:-1], monkeypatch)


def test_run_retarget_stage_wrist_target_is_ik_site_unless_mcp_is_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrist target. Consecutive rows, so the solver's wrist input is the world-frame target itself."""
    t = 5
    record = make_record(np.arange(t), seed=30, hand_right=hand(t, 31))
    assert record.hand_right is not None
    verts = hand_verts(t, 32)
    targets = {name: _world_rows(record, record.hand_right, verts, True, name)[1] for name in ("ik_site", "mcp")}
    assert float((targets["ik_site"] - targets["mcp"]).abs().max()) > 1e-3
    for requested, expected in ((None, "ik_site"), ("ik_site", "ik_site"), ("mcp", "mcp")):
        spy, _ = _run(record, verts, monkeypatch, wrist_target=requested)
        (args, _), = spy.calls
        assert float((args[1] - targets[expected]).abs().max()) <= 1e-12, requested

    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)
    with pytest.raises(ValueError, match="wrist_target"):
        run_retarget_stage(record, verts, hand_labels(), SCENE, tiny_sdf(), torch.zeros(8, 3), torch.ones(8, 3), 30.0,
                           RetargetConfig(), wrist_target="palm")
    assert spy.calls == []


@pytest.mark.parametrize(("field", "valid_row", "match"), [
    ("cam_to_world", True, "cam_to_world"), ("object_pose", True, "object_pose"),
    ("joints", True, "hand_right.joints"), ("params", True, "global orient"), ("verts", True, "hand_verts_cam"),
    ("joints", False, None), ("params", False, None), ("verts", False, None),
])
def test_run_retarget_stage_rejects_non_finite_inputs_before_the_solve(monkeypatch: pytest.MonkeyPatch, field: str,
                                                                      valid_row: bool, match: str | None) -> None:
    """A NaN on a row the stage uses raises ValueError naming the field, with no solver call; it used to
    fail inside the upsampling (SVD did not converge). A NaN on an invalid hand row is never read."""
    valid = np.array([True, True, True, False, True, True])
    track, verts = hand(6, 40, valid=valid), hand_verts(6, 41)
    record = make_record(3 * np.arange(6), seed=42, hand_right=track)
    row = 2 if valid_row else 3
    if field in ("cam_to_world", "object_pose"):
        values = getattr(record, field).copy()
        values[row, 1, 2] = np.nan
        record = dataclasses.replace(record, **{field: values})
    elif field in ("joints", "params"):
        values = getattr(track, field).copy()
        values[row, 9 if field == "joints" else 1] = np.nan
        record = dataclasses.replace(record, hand_right=dataclasses.replace(track, **{field: values}))
    else:
        verts[row, 10, 1] = float("nan")

    if match is None:
        spy, _ = _run(record, verts, monkeypatch)
        (args, _), = spy.calls
        assert all(bool(torch.isfinite(args[i]).all()) for i in (0, 1, 3))
        return
    spy = CapturingKinematic()
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)
    with pytest.raises(ValueError, match=match):
        run_retarget_stage(record, verts, hand_labels(), SCENE, tiny_sdf(), torch.zeros(8, 3), torch.ones(8, 3), 30.0,
                           RetargetConfig())
    assert spy.calls == []
