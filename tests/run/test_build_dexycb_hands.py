"""DexYCB ground-truth MANO hands as HaWoR-format `all_hand_meshes.npz` (`run/pipeline/build_dexycb_hands.py`).

The builder runs its hawor-env driver (`run/pipeline/_dexycb_hands_driver.py`, the regtracks converter's right- and
left-hand paths with every model path from config) as a CPU subprocess at a pinned thread count (`policy_hands.threads`,
2). The float32 MANO output depends on the thread count, and both earlier converters ran at 2 threads. Expectations are
their validated numbers within 1e-4 mm, with the test's own thread count pinned to 2:

- right hand, with the parent environment asking for 4 threads:
  - 20200820_135250: mean joint error over all 72 frames 0.761648416519165 mm, in-hand-label fraction
    0.8442159383033419 on frames [0, 18, 36, 53, 71];
  - 20200903_104655 (subject 04): 0.759232223033905 mm over the 63 frames with a GT hand, label fraction
    0.8802056555269923 on frames [11, 27, 42, 57, 73]; rows 0-10 have no GT hand and take row 11's parameters;
- left hand (20201015_143556, subject 09): 5-frame joint error 0.6866231560707092 mm, label fraction
  0.8125964010282777 on frames [0, 18, 36, 55, 73], pass; 20201015_142815: label fraction 0.5709511568123393, fail.

When the earlier converters' npz files are on the host, every float array must match within 1e-7 m (1e-4 mm) and
every int and bool array exactly.
Without real data: the driver's environment and command pin the threads whatever the parent asks; a crashed or timed-out
driver becomes a failed row and the build goes on; a stored record is reused only at the configured thread count.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from fastdo.policy.dexycb_inputs import (
    hands_npz_path,
    hands_record_path,
    hands_skip_reason,
    read_hands_record,
)
from run.pipeline.build_dexycb_hands import (
    HandsBuildConfig,
    _driver_env,
    build_hands,
    convert_sequence,
)
from tests.policy.dexycb_data import (
    DEXYCB_ROOT,
    H2H_CLIPS,
    HAWOR_PYTHON,
    HAWOR_ROOT,
    MANO_LEFT_PKL,
    MANO_PCA,
    MANO_RIGHT_PKL,
    REGTRACKS_HANDS,
    SEQ_LEFT,
    SEQ_LEFT_OCCLUDED,
    SEQ_RIGHT,
    SEQ_RIGHT_FILLED,
    SERIAL,
    assert_not_reserved,
    hawor_missing,
)
from tests.run.test_train_policy_smoke import _compose

logger = logging.getLogger(__name__)
TOL_MM = 1e-4          # joint errors (mm) and label fractions against the earlier converters
TOL_NPZ = 1e-7         # npz float arrays, metres (1e-4 mm)
THREADS = 2            # the earlier converters' thread count, pinned by this test's configuration
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
H2H_RIGHT: dict[str, dict[str, Any]] = {       # the earlier converter's validation records, per sequence
    SEQ_RIGHT: {"joint_all": 0.761648416519165, "label": 0.8442159383033419, "frames": [0, 18, 36, 53, 71],
                "num_frames": 72, "filled": []},
    SEQ_RIGHT_FILLED: {"joint_all": 0.759232223033905, "label": 0.8802056555269923, "frames": [11, 27, 42, 57, 73],
                       "num_frames": 74, "filled": list(range(11))},
}


def _cfg(tmp_path: Path, **overrides: Any) -> HandsBuildConfig:
    fields: dict[str, Any] = {"out_root": str(tmp_path / "refs"), "dexycb_root": str(DEXYCB_ROOT),
                              "mano_pca": str(MANO_PCA), "hawor_python": str(HAWOR_PYTHON),
                              "hawor_root": str(HAWOR_ROOT), "mano_right_pkl": str(MANO_RIGHT_PKL),
                              "mano_left_pkl": str(MANO_LEFT_PKL), "threads": THREADS}
    return HandsBuildConfig(**{**fields, **overrides})


def _four_thread_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent process asks for 4 threads, as under slot.sh's heavy.lock slot."""
    for name in THREAD_VARS:
        monkeypatch.setenv(name, "4")


def _fake_sequence(root: Path, sequence: str, side: str = "right") -> str:
    """A meta-only DexYCB sequence `<subject>/<sequence>` (no labels: any real driver run fails on it)."""
    (root / sequence).mkdir(parents=True)
    (root / sequence / "meta.yml").write_text(yaml.safe_dump({"serials": [SERIAL], "num_frames": 2, "mano_sides": [side],
                                                             "ycb_ids": [1], "ycb_grasp_ind": 0}))
    return sequence


def _stand_in_cfg(tmp_path: Path, dexycb_root: Path) -> HandsBuildConfig:
    return HandsBuildConfig(out_root=str(tmp_path / "refs"), dexycb_root=str(dexycb_root), mano_pca="p",
                            hawor_python="/nonexistent/python", hawor_root="h", mano_right_pkl="r.pkl",
                            mano_left_pkl="l.pkl")


def _assert_npz_match(new: Path, reference: Path) -> None:
    with np.load(new) as a, np.load(reference) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            x, y = a[key], b[key]
            assert x.shape == y.shape and x.dtype == y.dtype, key
            if x.dtype.kind == "f":
                diff = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max(initial=0.0))
                logger.debug("%s: max |diff| %.3g", key, diff)
                assert diff <= TOL_NPZ, (key, diff)
            else:
                assert np.array_equal(x, y), key


@hawor_missing
@pytest.mark.parametrize("sequence", list(H2H_RIGHT))
def test_right_hand_sequences_reproduce_h2h_from_a_four_thread_parent(sequence: str, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    assert_not_reserved(sequence)
    _four_thread_parent(monkeypatch)
    expected, cfg = H2H_RIGHT[sequence], _cfg(tmp_path)
    row = convert_sequence(sequence, cfg)
    record = read_hands_record(cfg.out_root, sequence)
    assert record is not None and row["pass"] and record["validation"]["pass"]
    conversion, validation = record["conversion"], record["validation"]
    logger.info("right %s: joint error all frames %.9f mm, label fraction %.10f, frames %s, threads %s", sequence,
                conversion["joint3d_err_mm_all_frames"]["mean"], validation["frac_all"], validation["frames"],
                conversion["threads"])
    n = expected["num_frames"]
    assert (conversion["hand_side"], conversion["serial"], conversion["num_frames"]) == ("right", SERIAL, n)
    assert conversion["mano_model"] == cfg.mano_right_pkl and conversion["threads"] == THREADS
    assert conversion["filled_from_nearest_valid"] == expected["filled"]
    assert abs(conversion["joint3d_err_mm_all_frames"]["mean"] - expected["joint_all"]) <= TOL_MM
    assert abs(validation["frac_all"] - expected["label"]) <= TOL_MM and validation["frames"] == expected["frames"]
    first = len(expected["filled"])
    with np.load(hands_npz_path(cfg.out_root, sequence)) as npz:
        assert npz["right_vertices"].shape == (n, 778, 3) and not bool(npz["left_valid"].any())
        assert np.flatnonzero(~npz["right_valid"]).tolist() == expected["filled"]
        for key in ("right_trans", "right_rot", "right_hand_pose"):    # rows without a GT hand: nearest valid row
            filled, source = npz[key][:first], np.repeat(npz[key][first:first + 1], first, axis=0)
            if key == "right_hand_pose":        # PCA @ basis: a float32 GEMM need not give identical rows bit for bit
                assert np.allclose(filled, source, rtol=0.0, atol=1e-6), key
            else:                               # copies of the nearest valid row's GT parameters
                assert np.array_equal(filled, source), key
    if H2H_CLIPS is not None:
        _assert_npz_match(hands_npz_path(cfg.out_root, sequence),
                          H2H_CLIPS / sequence.split("/")[1] / "all_hand_meshes.npz")


@hawor_missing
def test_left_hand_sequences_reproduce_the_regtracks_validation(tmp_path: Path) -> None:
    """One passing and one occluded left-hand sequence: the summary counts 1 pass, lists the failure, and is the
    `validation.json` on disk; both conversions use the configured MANO_LEFT at the pinned thread count and match
    regtracks within 1e-4 mm."""
    cfg = _cfg(tmp_path)
    summary = build_hands([SEQ_LEFT, SEQ_LEFT_OCCLUDED], cfg)
    assert (summary["n"], summary["n_pass"], summary["failed"]) == (2, 1, [SEQ_LEFT_OCCLUDED])
    assert json.loads((Path(cfg.out_root) / "validation.json").read_text()) == summary
    expected = {SEQ_LEFT: (0.6866231560707092, 0.8125964010282777, [0, 18, 36, 55, 73], True),
                SEQ_LEFT_OCCLUDED: (0.686482846736908, 0.5709511568123393, [0, 18, 36, 53, 71], False)}
    for seq, (joint, label, frames, passed) in expected.items():
        record = read_hands_record(cfg.out_root, seq)
        assert record is not None
        conversion, validation = record["conversion"], record["validation"]
        logger.info("left %s: 5-frame joint error %.9f mm, label fraction %.10f, pass %s", seq,
                    validation["joint_err_mm_mean_5f"], validation["frac_all"], validation["pass"])
        assert conversion["hand_side"] == "left" and conversion["mano_model"] == cfg.mano_left_pkl
        assert conversion["threads"] == THREADS
        assert not conversion["shapedirs_fix_applied"] and conversion["pca_basis_check"]["hands_mean_equals_mano_layer"]
        assert abs(validation["joint_err_mm_mean_5f"] - joint) <= TOL_MM
        assert abs(validation["frac_all"] - label) <= TOL_MM
        assert validation["frames"] == frames and validation["pass"] is passed and validation["pass_joint"]
        assert summary["sequences"][seq]["pass"] is passed
        with np.load(hands_npz_path(cfg.out_root, seq)) as npz:
            assert bool(npz["left_valid"].all()) and not bool(npz["right_valid"].any())
        if REGTRACKS_HANDS is not None:
            _assert_npz_match(hands_npz_path(cfg.out_root, seq),
                              REGTRACKS_HANDS / seq.split("/")[1] / "all_hand_meshes.npz")


def test_hands_config_takes_the_mano_models_from_config(tmp_path: Path) -> None:
    """The Hydra block's MANO paths are `retarget_stage.mano_pkl_path` (the left one HaWoR's data_left tree, Amendment
    1) and its thread count is 2; a left-hand sequence with no left model configured, or a reserved subject, raises
    before any subprocess."""
    root = _compose(tmp_path, [])
    cfg = HandsBuildConfig.from_mapping(root.policy_hands)
    assert cfg.mano_left_pkl == str(root.retarget_stage.mano_pkl_path.left) == str(root.isolation.mano_left_pkl)
    assert cfg.mano_right_pkl == str(root.retarget_stage.mano_pkl_path.right)
    assert cfg.mano_left_pkl.endswith("data_left/mano_left/MANO_LEFT.pkl") and cfg.exclude_subjects == ("subject-07",
                                                                                                      "subject-08")
    assert cfg.threads == root.policy_hands.threads == 2
    sequence = _fake_sequence(tmp_path / "dexycb", "20200101-subject-00/20200101_000000", side="left")
    no_left = HandsBuildConfig(out_root=str(tmp_path / "refs"), dexycb_root=str(tmp_path / "dexycb"), mano_pca="p",
                               hawor_python="/nonexistent/python", hawor_root="h", mano_right_pkl="r.pkl",
                               mano_left_pkl="")
    with pytest.raises(ValueError, match="mano_pkl_path.left"):
        convert_sequence(sequence, no_left)
    with pytest.raises(ValueError, match="reserved"):
        convert_sequence("20200928-subject-07/20200928_000000", no_left)


def test_the_driver_runs_at_the_configured_threads_whatever_the_parent_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    _four_thread_parent(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    env = _driver_env(2)
    assert [env[name] for name in THREAD_VARS] == ["2", "2", "2"]
    assert env["CUDA_VISIBLE_DEVICES"] == "" and "PYTHONPATH" not in env
    assert _stand_in_cfg(Path("o"), Path("d")).threads == 2


def test_a_crashed_driver_is_a_failed_row_and_the_build_goes_on(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver that exits without its validation record (here exit 3 with an IndexError), or times out, gives a row
    with `pass` False and the error; the next sequence still runs, and `validation.json` lists both. With no record
    on disk the reference function skips the sequence. Every run is pinned to 2 threads in its command and env."""
    _four_thread_parent(monkeypatch)
    root = tmp_path / "dexycb"
    sequences = [_fake_sequence(root, f"20200101-subject-00/20200101_00000{i}") for i in range(2)]
    cfg = _stand_in_cfg(tmp_path, root)
    runs: list[tuple[list[str], dict[str, str]]] = []

    def crash(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        runs.append((cmd, kwargs["env"]))
        if len(runs) == 2:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="IndexError: list index out of range")

    monkeypatch.setattr(subprocess, "run", crash)
    summary = build_hands(sequences, cfg)
    assert (summary["n"], summary["n_pass"], summary["failed"]) == (2, 0, sequences)
    first, second = (summary["sequences"][s] for s in sequences)
    logger.info("failed rows: %s | %s", first, second)
    assert first["pass"] is False and "exited 3" in first["error"] and "IndexError" in first["error"]
    assert second["pass"] is False and "timed out" in second["error"]
    assert all((row["hand_side"], row["serial"]) == ("right", SERIAL) for row in (first, second))
    assert json.loads((Path(cfg.out_root) / "validation.json").read_text()) == summary
    assert len(runs) == 2
    for cmd, env in runs:
        assert cmd[cmd.index("--threads") + 1] == "2" and [env[name] for name in THREAD_VARS] == ["2", "2", "2"]
    for sequence in sequences:
        assert read_hands_record(cfg.out_root, sequence) is None
        assert hands_skip_reason(cfg.out_root, f"{sequence}/{SERIAL}", "right") is not None


def test_a_stored_record_is_reused_only_at_the_configured_thread_count(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored record of the same serial built at 2 threads is reused without a driver run; one built at 4 threads, or
    before the thread count was recorded, is rebuilt (here by a stand-in driver that fails)."""
    root = tmp_path / "dexycb"
    sequence = _fake_sequence(root, "20200101-subject-00/20200101_000000")
    cfg = _stand_in_cfg(tmp_path, root)
    runs: list[list[str]] = []

    def crash(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="stand-in driver")

    monkeypatch.setattr(subprocess, "run", crash)
    npz, path = hands_npz_path(cfg.out_root, sequence), hands_record_path(cfg.out_root, sequence)
    path.parent.mkdir(parents=True)
    conversion = {"hand_side": "right", "serial": SERIAL, "num_frames": 2, "n_valid": 2,
                  "joint3d_err_mm_all_frames": {"mean": 0.7}}
    validation = {"joint_err_mm_mean_5f": 0.7, "frac_all": 0.9, "frames": [0, 1], "pass_joint": True,
                  "pass_label": True, "pass": True}
    for threads, reused in ((2, True), (4, False), (None, False)):
        stored = {**conversion, "threads": threads} if threads is not None else conversion
        npz.write_bytes(b"stand-in")
        path.write_text(json.dumps({"conversion": stored, "validation": validation}))
        before = len(runs)
        row = convert_sequence(sequence, cfg)
        assert row["pass"] is reused and len(runs) == before + (0 if reused else 1), (threads, row)
