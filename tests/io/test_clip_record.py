from pathlib import Path

import numpy as np
import pytest

from fastdo.io.clip_record import ClipRecord, HandTrack


def _hand(t: int) -> HandTrack:
    return HandTrack(params=np.zeros((t, 48), np.float32), betas=np.zeros(10, np.float32),
                     trans=np.zeros((t, 3), np.float32), joints=np.zeros((t, 21, 3), np.float32),
                     valid=np.ones(t, bool))


def _record(t: int = 4) -> ClipRecord:
    eye = np.tile(np.eye(4, dtype=np.float32), (t, 1, 1))
    return ClipRecord(clip_id="c", frame_idx=np.arange(t, dtype=np.int64),
                      timestamps=np.arange(t, dtype=np.float64) / 10.0,
                      intrinsics=np.eye(3, dtype=np.float32), cam_to_world=eye,
                      object_mesh_path="obj.obj", object_pose=eye, object_scale=1.0,
                      gravity_cam=np.array([0, -1, 0], np.float32), confidence=np.ones(t, np.float32),
                      gate_score=0.9, hand_left=None, hand_right=_hand(t))


def test_roundtrip(tmp_path: Path) -> None:
    rec = _record()
    p = tmp_path / "c.npz"
    rec.save(p)
    back = ClipRecord.load(p)
    assert back.clip_id == "c" and back.n_frames == 4
    assert back.hand_left is None and back.hand_right is not None
    np.testing.assert_allclose(back.object_pose, rec.object_pose)
    np.testing.assert_allclose(back.hand_right.joints, rec.hand_right.joints)
    assert back.object_mesh_path == "obj.obj" and back.gate_score == pytest.approx(0.9)


def test_validate_rejects_shape_mismatch() -> None:
    rec = _record()
    bad = ClipRecord(**{**rec.__dict__, "confidence": np.ones(3, np.float32)})
    with pytest.raises(ValueError, match="confidence"):
        bad.validate()


def test_roundtrip_without_npz_suffix(tmp_path: Path) -> None:
    rec = _record()
    p = tmp_path / "clip_no_ext"
    rec.save(p)
    back = ClipRecord.load(p)
    assert back.clip_id == "c" and back.n_frames == 4
    np.testing.assert_allclose(back.object_pose, rec.object_pose)


def test_validate_rejects_wrong_dtype() -> None:
    rec = _record()
    bad = ClipRecord(**{**rec.__dict__, "confidence": np.ones(4, np.float64)})
    with pytest.raises(ValueError, match="confidence"):
        bad.validate()
