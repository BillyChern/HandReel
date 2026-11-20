"""DexYCB references, no data: `make_reference_fn` skips and logs every clip whose scene or hands failed validation
(never keeping it), and `clearance_report` aggregates the clearance of Amendment 1 per object and hand side without
gating."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fastdo.policy.dexycb_inputs import (
    hands_npz_path,
    hands_record_path,
    scene_record_path,
    write_scene_record,
)
from fastdo.policy.dexycb_refs import DexYCBRefConfig, clearance_report, make_reference_fn
from tests.qc.records import hand, make_record

OBJECT, SEQUENCE, SERIAL = "002_master_chef_can", "20200101-subject-00/20200101_000000", "111"
CLIP = f"{SEQUENCE}/{SERIAL}"


def _record(tmp_path: Path, confidence: np.ndarray | None = None, left: bool = False) -> object:
    track = hand(6, 5)
    base = make_record(np.arange(6), seed=4, hand_right=None if left else track, hand_left=track if left else None)
    fields: dict[str, Any] = {"clip_id": CLIP,
                              "object_mesh_path": str(tmp_path / "models" / OBJECT / "textured_simple.obj")}
    if confidence is not None:
        fields["confidence"] = confidence.astype(np.float32)
    return dataclasses.replace(base, **fields)


def _write_hands(hands_root: Path, passed: bool = True, serial: str = SERIAL, side: str = "right") -> None:
    path = hands_npz_path(hands_root, SEQUENCE)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, right_vertices=np.zeros((6, 778, 3), np.float32))
    hands_record_path(hands_root, SEQUENCE).write_text(json.dumps({
        "conversion": {"seq": SEQUENCE, "serial": serial, "hand_side": side, "num_frames": 6},
        "validation": {"pass": passed, "frac_all": 0.9 if passed else 0.5, "joint_err_mm_mean_5f": 0.7}}))


def test_reference_fn_skips_and_logs_clips_whose_inputs_failed_validation(tmp_path: Path,
                                                                           caplog: pytest.LogCaptureFixture) -> None:
    scenes, hands = tmp_path / "scenes", tmp_path / "refs"
    fn = make_reference_fn(scenes, hands, DexYCBRefConfig(dexycb_root=str(tmp_path), mano_right_pkl="MANO_RIGHT.pkl"))
    record = _record(tmp_path)

    def expect_skip(fragment: str, rec: object = record, side: str = "right") -> None:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="fastdo.policy.dexycb_refs"):
            assert fn(rec) is None  # type: ignore[arg-type]
        reason = fn.records[-1]["reason"]
        assert fn.records[-1]["status"] == "skipped" and fragment in reason, reason
        assert fn.records[-1]["clip_id"] == CLIP and fn.records[-1]["object"] == OBJECT
        assert any(CLIP in m and fragment in m for m in caplog.messages), caplog.messages
        if "object pose" not in fragment:
            assert fragment in (fn.skip_reason(CLIP, OBJECT, side) or "")

    expect_skip("no scene")
    write_scene_record(scenes, {"object": OBJECT, "side": "right", "scene_xml": "scene.xml", "pass": False,
                                "validation": {"reasons": ["no object contact geoms"]}})
    expect_skip("scene failed validation")
    write_scene_record(scenes, {"object": OBJECT, "side": "right", "scene_xml": "scene.xml", "pass": True})
    assert json.loads(scene_record_path(scenes, OBJECT, "right").read_text())["pass"] is True
    expect_skip("hands not built")
    _write_hands(hands, passed=False)
    expect_skip("hands failed validation")
    _write_hands(hands, serial="222")
    expect_skip("serial")
    _write_hands(hands, side="left")
    expect_skip("hand side")
    _write_hands(hands)
    assert fn.skip_reason(CLIP, OBJECT, "right") is None
    expect_skip("object pose missing on rows [2]", _record(tmp_path, confidence=np.array([1, 1, 0, 1, 1, 1])))
    write_scene_record(scenes, {"object": OBJECT, "side": "left", "scene_xml": "scene.xml", "pass": True})
    _write_hands(hands, side="left")
    expect_skip("no MANO model configured for the left hand", _record(tmp_path, left=True), side="left")
    assert all(r["status"] == "skipped" for r in fn.records) and len(fn.records) == 8


def test_clearance_report_aggregates_per_object_and_hand_side_without_a_gate() -> None:
    """Per object and hand side (key `<object>/<side>`): kept clips, skipped clips, pooled frames, the minimum clearance
    over its clips and the frame-weighted fractions of frames clearing 0.5 mm and penetrating. An object's left and
    right clips are separate rows; a row with no kept clip reports None."""
    records = [
        {"status": "kept", "object": "a", "side": "right",
         "clearance": {"n_frames": 10, "frame_min_mm_min": 0.7, "frac_frames_clear_0p5mm": 0.9,
                       "frac_frames_penetrating": 0.0}},
        {"status": "kept", "object": "a", "side": "right",
         "clearance": {"n_frames": 30, "frame_min_mm_min": -0.2, "frac_frames_clear_0p5mm": 0.5,
                       "frac_frames_penetrating": 0.1}},
        {"status": "kept", "object": "a", "side": "left",
         "clearance": {"n_frames": 20, "frame_min_mm_min": -2.4, "frac_frames_clear_0p5mm": 0.8,
                       "frac_frames_penetrating": 0.05}},
        {"status": "skipped", "object": "a", "side": "right", "reason": "hands failed validation"},
        {"status": "skipped", "object": "b", "side": "right", "reason": "no scene"},
    ]
    report = clearance_report(records)
    assert set(report) == {"a/right", "a/left", "b/right"}
    assert report["a/right"] == {"object": "a", "side": "right", "n_clips": 2, "n_skipped": 1, "n_frames": 40,
                                 "min_clearance_mm": -0.2, "frac_frames_clear_0p5mm": pytest.approx(24 / 40),
                                 "frac_frames_penetrating": pytest.approx(3 / 40)}
    assert report["a/left"] == {"object": "a", "side": "left", "n_clips": 1, "n_skipped": 0, "n_frames": 20,
                                "min_clearance_mm": -2.4, "frac_frames_clear_0p5mm": pytest.approx(0.8),
                                "frac_frames_penetrating": pytest.approx(0.05)}
    assert report["b/right"] == {"object": "b", "side": "right", "n_clips": 0, "n_skipped": 1, "n_frames": 0,
                                 "min_clearance_mm": None, "frac_frames_clear_0p5mm": None,
                                 "frac_frames_penetrating": None}
