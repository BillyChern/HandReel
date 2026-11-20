"""Tests for the DexYCB reader.

The fixtures below mirror the on-disk DexYCB layout as documented in the
dex-ycb-toolkit README and source (verified 2026-09-12 via
https://github.com/NVlabs/dex-ycb-toolkit, README.md and
dex_ycb_toolkit/{dex_ycb,sequence_loader,grasp_eval}.py):

- ``labels_XXXXXX.npz``'s ``pose_y`` is ``(num_obj, 3, 4)`` float32, each
  row ``[R | t]`` (rotation matrix + translation, object-to-camera,
  metres) -- README.md L158, corroborated by ``grasp_eval.py`` L170-172
  which does ``np.vstack((pose_y[grasp_ind], [[0, 0, 0, 1]]))`` to build a
  4x4 (only valid if the per-object slice is already ``(3, 4)``).

Also covered:
- ``HandTrack.params`` must hold axis-angle, but DexYCB's
  ``pose_m[:, 3:48]`` is 45 MANO PCA coefficients (see ``dexycb.py``'s
  module docstring for the citation). These tests supply a synthetic MANO
  PCA basis via ``DexYCBReader(root, mano_pca=...)`` and assert the
  converted values, and assert ``load()`` raises without ``mano_pca``.
- ``_load_gravity``'s extrinsics-derived branch was removed (unverified
  convention, no prior test coverage); ``gravity_cam`` is always the
  default, and a warning is emitted when an extrinsics file exists.
- The intrinsics file is now matched by glob (not a hardcoded
  ``640x480`` suffix), and a missing ``serials`` key in ``meta.yml`` now
  warns and skips the sequence.

Found on real data:
- Real downloaded ``intrinsics/*.yml`` (and ``extrinsics.yml``) tag some
  values ``!!python/tuple``, which plain ``yaml.safe_load`` rejects.
  ``load_dexycb_yaml`` now uses a ``SafeLoader`` subclass that additionally
  accepts that one tag; see ``dexycb.py``'s module docstring.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from fastdo.io.clip_record import ClipRecord
from fastdo.io.datasets.dexycb import DexYCBReader

_SUBJECT = "20200709-subject-01"
_SEQUENCE = "20200709_141754"
_SERIAL = "839512060362"
_SEQ_ID = f"{_SUBJECT}/{_SEQUENCE}/{_SERIAL}"
_LOGGER_NAME = "fastdo.io.datasets.dexycb"


def _write_intrinsics(root: Path, suffix: str = "640x480") -> None:
    path = root / "calibration" / "intrinsics" / f"{_SERIAL}_{suffix}.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"color": {"fx": 612.0, "fy": 612.5, "ppx": 320.1, "ppy": 240.2}}))


def _write_mano_calib(root: Path, calib_id: str, betas: list[float]) -> None:
    path = root / "calibration" / f"mano_{calib_id}" / "mano.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"betas": betas}))


def _write_mano_pca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic MANO PCA basis: components = 2*I(45), mean = 0.1."""
    components = 2.0 * np.eye(45, dtype=np.float32)
    mean = np.full(45, 0.1, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, hands_components=components, hands_mean=mean)
    return components, mean


def _write_meta(seq_dir: Path, num_frames: int, extrinsics: str | None = None) -> None:
    meta: dict[str, object] = {
        "serials": [_SERIAL],
        "num_frames": num_frames,
        "ycb_ids": [5],
        "ycb_grasp_ind": 0,
        "mano_sides": ["right"],
        "mano_calib": ["subj0"],
    }
    if extrinsics is not None:
        meta["extrinsics"] = extrinsics
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "meta.yml").write_text(yaml.safe_dump(meta))


def _write_label(cam_dir: Path, frame: int, pose_y_row: np.ndarray) -> None:
    cam_dir.mkdir(parents=True, exist_ok=True)
    pose_m = np.zeros((1, 51), np.float32)
    pose_m[0, :48] = 0.01 * np.arange(48, dtype=np.float32)
    pose_m[0, 48:51] = [0.0, 0.0, 0.5]
    joint_3d = np.full((1, 21, 3), float(frame), dtype=np.float32)
    np.savez(
        cam_dir / f"labels_{frame:06d}.npz",
        seg=np.zeros((8, 8), np.uint8),
        pose_y=pose_y_row[np.newaxis, ...],
        pose_m=pose_m,
        joint_3d=joint_3d,
        joint_2d=np.zeros((1, 21, 2), np.float32),
    )


def _rt(translation: np.ndarray) -> np.ndarray:
    rt = np.eye(3, 4, dtype=np.float32)
    rt[:, 3] = translation
    return rt


def _build_fixture(
    tmp_path: Path, second_translation: np.ndarray, extrinsics: str | None = None,
) -> tuple[Path, np.ndarray, Path]:
    """Writes a 2-frame, 1-subject, 1-sequence, 1-camera synthetic DexYCB tree.

    Returns ``(root, per_frame_object_translations, mano_pca_npz_path)``.
    """
    root = tmp_path
    seq_dir = root / _SUBJECT / _SEQUENCE
    _write_meta(seq_dir, num_frames=2, extrinsics=extrinsics)
    _write_intrinsics(root)
    _write_mano_calib(root, "subj0", betas=[0.1] * 10)
    mano_pca_path = root / "mano_pca.npz"
    _write_mano_pca(mano_pca_path)

    cam_dir = seq_dir / _SERIAL
    first = np.array([0.10, 0.20, 0.30], np.float32)
    _write_label(cam_dir, 0, _rt(first))
    _write_label(cam_dir, 1, _rt(second_translation))

    return root, np.stack([first, second_translation]), mano_pca_path


def test_sequences_lists_subject_sequence_serial(tmp_path: Path) -> None:
    root, _, _ = _build_fixture(tmp_path, np.array([0.4, 0.5, 0.6], np.float32))
    assert DexYCBReader(root).sequences() == [_SEQ_ID]


def test_load_returns_valid_cliprecord(tmp_path: Path) -> None:
    root, translations, mano_pca = _build_fixture(tmp_path, np.array([0.4, 0.5, 0.6], np.float32))
    rec = DexYCBReader(root, mano_pca=mano_pca).load(_SEQ_ID)
    rec.validate()

    assert isinstance(rec, ClipRecord)
    assert rec.clip_id == _SEQ_ID
    assert rec.n_frames == 2
    np.testing.assert_allclose(rec.object_pose[:, :3, 3], translations)
    np.testing.assert_allclose(rec.object_pose[:, 3, :], np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)))
    np.testing.assert_allclose(rec.cam_to_world, np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)))
    np.testing.assert_allclose(rec.gravity_cam, np.array([0.0, -1.0, 0.0], np.float32))
    np.testing.assert_allclose(rec.confidence, np.ones(2, np.float32))
    assert rec.gate_score == pytest.approx(1.0)
    np.testing.assert_allclose(rec.intrinsics, np.array(
        [[612.0, 0.0, 320.1], [0.0, 612.5, 240.2], [0.0, 0.0, 1.0]], np.float32))

    assert rec.hand_left is None
    assert rec.hand_right is not None
    np.testing.assert_allclose(rec.hand_right.betas, np.full(10, 0.1, np.float32))
    np.testing.assert_allclose(rec.hand_right.trans[0], [0.0, 0.0, 0.5])
    assert bool(rec.hand_right.valid[0])

    # pose_m[0, :48] fixture = 0.01 * arange(48): [:3] global orient (already
    # axis-angle), [3:48] the 45 PCA coefficients converted via
    # coeffs @ (2*I) + 0.1 == 2*coeffs + 0.1.
    expected_orient = 0.01 * np.arange(3, dtype=np.float32)
    expected_pca_coeffs = 0.01 * np.arange(3, 48, dtype=np.float32)
    expected_fingers = 2.0 * expected_pca_coeffs + 0.1
    np.testing.assert_allclose(rec.hand_right.params[0, :3], expected_orient, atol=1e-6)
    np.testing.assert_allclose(rec.hand_right.params[0, 3:], expected_fingers, atol=1e-5)


def test_load_without_mano_pca_raises(tmp_path: Path) -> None:
    root, _, _ = _build_fixture(tmp_path, np.array([0.4, 0.5, 0.6], np.float32))
    with pytest.raises(ValueError, match="mano_pca"):
        DexYCBReader(root).load(_SEQ_ID)


def test_invalid_object_pose_frame_gets_zero_confidence_and_identity(tmp_path: Path) -> None:
    root = tmp_path
    seq_dir = root / _SUBJECT / _SEQUENCE
    _write_meta(seq_dir, num_frames=2)
    _write_intrinsics(root)
    _write_mano_calib(root, "subj0", betas=[0.1] * 10)
    mano_pca_path = root / "mano_pca.npz"
    _write_mano_pca(mano_pca_path)
    cam_dir = seq_dir / _SERIAL

    _write_label(cam_dir, 0, _rt(np.array([0.1, 0.2, 0.3], np.float32)))
    _write_label(cam_dir, 1, np.full((3, 4), -1.0, dtype=np.float32))

    rec = DexYCBReader(root, mano_pca=mano_pca_path).load(_SEQ_ID)
    rec.validate()

    assert rec.confidence[0] == pytest.approx(1.0)
    assert rec.confidence[1] == pytest.approx(0.0)
    np.testing.assert_allclose(rec.object_pose[1], np.eye(4, dtype=np.float32))


def test_no_extrinsics_file_yields_default_gravity(tmp_path: Path) -> None:
    root, _, mano_pca = _build_fixture(tmp_path, np.array([0.4, 0.5, 0.6], np.float32))
    assert not (root / "calibration" / "extrinsics_dummy").exists()
    rec = DexYCBReader(root, mano_pca=mano_pca).load(_SEQ_ID)
    np.testing.assert_allclose(rec.gravity_cam, np.array([0.0, -1.0, 0.0], np.float32))


def test_extrinsics_present_still_uses_default_gravity_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    root, _, mano_pca = _build_fixture(
        tmp_path, np.array([0.4, 0.5, 0.6], np.float32), extrinsics="20200709",
    )
    extr_path = root / "calibration" / "extrinsics_20200709" / "extrinsics.yml"
    extr_path.parent.mkdir(parents=True, exist_ok=True)
    extr_path.write_text(yaml.safe_dump({
        "master": _SERIAL,
        "extrinsics": {_SERIAL: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]},
    }))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        rec = DexYCBReader(root, mano_pca=mano_pca).load(_SEQ_ID)

    np.testing.assert_allclose(rec.gravity_cam, np.array([0.0, -1.0, 0.0], np.float32))
    assert any("extrinsics" in m and "unverified" in m for m in caplog.messages)


def test_intrinsics_glob_matches_any_resolution(tmp_path: Path) -> None:
    root, _, mano_pca = _build_fixture(tmp_path, np.array([0.4, 0.5, 0.6], np.float32))
    old_path = root / "calibration" / "intrinsics" / f"{_SERIAL}_640x480.yml"
    new_path = root / "calibration" / "intrinsics" / f"{_SERIAL}_1280x720.yml"
    old_path.rename(new_path)

    rec = DexYCBReader(root, mano_pca=mano_pca).load(_SEQ_ID)
    np.testing.assert_allclose(rec.intrinsics, np.array(
        [[612.0, 0.0, 320.1], [0.0, 612.5, 240.2], [0.0, 0.0, 1.0]], np.float32))


def test_load_accepts_python_tuple_tag_in_calibration_yaml(tmp_path: Path) -> None:
    """Real DexYCB calibration files (e.g. ``intrinsics/*.yml``) tag some
    values ``!!python/tuple``, which ``yaml.safe_load`` rejects with
    ``ConstructorError``. ``load()`` must still succeed and still recover
    the ``color`` block used to build ``intrinsics``.
    """
    root, _, mano_pca = _build_fixture(tmp_path, np.array([0.4, 0.5, 0.6], np.float32))
    intr_path = root / "calibration" / "intrinsics" / f"{_SERIAL}_640x480.yml"
    intr_path.write_text(
        "color: {fx: 600.0, fy: 600.0, ppx: 320.0, ppy: 240.0, res: !!python/tuple [640, 480]}\n"
        "depth: {fx: 600.0, fy: 600.0, ppx: 320.0, ppy: 240.0}\n"
        "extrinsics: !!python/tuple\n"
        "- 1.0\n- 0.0\n- 0.0\n- 0.0\n- 0.0\n- 1.0\n- 0.0\n- 0.0\n- 0.0\n- 0.0\n- 1.0\n- 0.0\n"
    )

    rec = DexYCBReader(root, mano_pca=mano_pca).load(_SEQ_ID)
    np.testing.assert_allclose(rec.intrinsics, np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], np.float32))


def test_excluded_subjects_are_never_listed_or_loaded(tmp_path: Path) -> None:
    """Subjects reserved for calibration and confirmation are skipped before their `meta.yml` is read
    (`exclude_subjects`, suffix match), and `load` refuses them; `subjects` is the allowlist form. The reserved
    subject's `meta.yml` is not YAML, so any read of it would raise, as the unfiltered reader shows."""
    root, _, mano_pca = _build_fixture(tmp_path, np.array([0.4, 0.5, 0.6], np.float32))
    reserved = root / "20200928-subject-07" / "20200928_000000"
    reserved.mkdir(parents=True)
    (reserved / "meta.yml").write_text("{not yaml: [")
    reader = DexYCBReader(root, mano_pca=mano_pca, exclude_subjects=("subject-07",))
    assert reader.sequences() == [_SEQ_ID]
    with pytest.raises(ValueError, match="excluded"):
        reader.load(f"20200928-subject-07/20200928_000000/{_SERIAL}")
    assert DexYCBReader(root, subjects=("subject-01",)).sequences() == [_SEQ_ID]
    with pytest.raises(ValueError, match="excluded"):
        DexYCBReader(root, subjects=("subject-01",)).load(f"20200928-subject-07/20200928_000000/{_SERIAL}")
    with pytest.raises(yaml.YAMLError):
        DexYCBReader(root).sequences()


def test_missing_serials_warns_and_skips_sequence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path
    seq_dir = root / _SUBJECT / _SEQUENCE
    seq_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "num_frames": 2, "ycb_ids": [5], "ycb_grasp_ind": 0,
        "mano_sides": ["right"], "mano_calib": ["subj0"],
    }
    (seq_dir / "meta.yml").write_text(yaml.safe_dump(meta))

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = DexYCBReader(root).sequences()

    assert result == []
    assert any("serials" in m for m in caplog.messages)
