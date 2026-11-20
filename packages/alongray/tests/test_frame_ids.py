# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""Optional frame ids in `score` and the command line: rows are joined on integer frame ids, with the rules of
`alongray.change` (identical id sets, no duplicates, integers only, increasing id order). Without ids the rows are
taken as synchronized row by row and every output equals 0.3.1's (tests/data/no_ids_golden_0.3.1.json, written by
the unchanged 0.3.1 code with numpy 2.1.2): bit for bit in the same environment (the ids-versus-no-ids tests below),
and to a few ulp against the recorded file, since another numpy or BLAS build rounds a few values differently."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from alongray import score
from alongray.cli import main, run

PKG = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
K = (600.0, 600.0, 320.0, 240.0)
T = 12


def _scene(seed: int = 3) -> dict[str, np.ndarray]:
    """A gauge-plus-noise prediction whose rows are NOT interchangeable (every frame differs)."""
    rng = np.random.default_rng(seed)
    g = np.column_stack([rng.normal(0, 0.05, T), rng.normal(0, 0.05, T), rng.uniform(0.45, 0.75, T)])
    h = g[:, None, :] + 0.03 * rng.standard_normal((T, 21, 3))
    p = 1.08 * g + 0.004 * rng.standard_normal((T, 3))
    ph = h + 0.05 * h[:, :1] + 0.003 * rng.standard_normal((T, 21, 3))
    return {"p": p, "g": g, "ph": ph, "h": h}


def _hex(o: object) -> object:
    if isinstance(o, dict):
        return {k: _hex(v) for k, v in o.items()}
    if isinstance(o, float):
        return float.hex(o)
    return o


def _same(a: object, b: object) -> bool:
    """Bit-identical, NaN included."""
    return json.dumps(_hex(a), sort_keys=True) == json.dumps(_hex(b), sort_keys=True)


# --- API -------------------------------------------------------------------------------------------------------------

def test_api_shift_by_one_is_refused() -> None:
    """Equal-length arrays whose ids are shifted by one frame: the sets differ, so the call is refused."""
    s = _scene()
    with pytest.raises(ValueError, match="same set"):
        score(s["p"], s["g"], pred_ids=np.arange(T) + 1, gt_ids=np.arange(T))
    with pytest.raises(ValueError, match="same set"):
        score(s["p"], s["g"], s["ph"], s["h"], pred_ids=np.arange(T) + 1, gt_ids=np.arange(T))


def test_api_missing_frame_is_refused() -> None:
    """A prediction that lacks one frame of the truth is refused (never intersected or trimmed)."""
    s = _scene()
    keep = np.delete(np.arange(T), 5)
    with pytest.raises(ValueError, match="same set"):
        score(s["p"][keep], s["g"], pred_ids=keep, gt_ids=np.arange(T))
    with pytest.raises(ValueError, match="same set"):
        score(s["p"], s["g"][keep], pred_ids=np.arange(T), gt_ids=keep)


def test_api_duplicate_ids_are_refused() -> None:
    """Duplicates are refused on either side, also when both sides hold the same multiset."""
    s = _scene()
    dup = np.arange(T)
    dup[3] = 2
    with pytest.raises(ValueError, match="duplicate"):
        score(s["p"], s["g"], pred_ids=dup, gt_ids=dup.copy())
    with pytest.raises(ValueError, match="duplicate"):
        score(s["p"], s["g"], pred_ids=np.arange(T), gt_ids=dup)


def test_api_non_integer_and_malformed_ids_are_refused() -> None:
    s = _scene()
    with pytest.raises(ValueError, match="integer"):
        score(s["p"], s["g"], pred_ids=np.arange(T) + 0.0, gt_ids=np.arange(T))
    with pytest.raises(ValueError, match="integer"):
        score(s["p"], s["g"], pred_ids=np.arange(T), gt_ids=np.ones(T, bool))
    with pytest.raises(ValueError, match="one id per row"):
        score(s["p"], s["g"], pred_ids=np.arange(T - 1), gt_ids=np.arange(T))
    with pytest.raises(ValueError, match="one id per row"):
        score(s["p"], s["g"], pred_ids=np.arange(T)[:, None], gt_ids=np.arange(T))
    with pytest.raises(ValueError, match="both pred_ids and gt_ids"):
        score(s["p"], s["g"], pred_ids=np.arange(T))
    with pytest.raises(ValueError, match="both pred_ids and gt_ids"):
        score(s["p"], s["g"], gt_ids=np.arange(T))


def test_api_shuffled_ids_give_the_sorted_result_bit_for_bit() -> None:
    """Each side shuffled by its own permutation, with its ids: the result equals the sorted, id-free call exactly."""
    s = _scene()
    rng = np.random.default_rng(11)
    ip, ig = rng.permutation(T), rng.permutation(T)
    ids = np.arange(T) * 3 + 100                       # sparse, non-zero-based frame ids
    ref = score(s["p"], s["g"], s["ph"], s["h"], K).as_dict()
    got = score(s["p"][ip], s["g"][ig], s["ph"][ip], s["h"][ig], K, pred_ids=ids[ip], gt_ids=ids[ig]).as_dict()
    assert _same(got, ref)
    # poses and point clouds follow the same join
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose_p, pose_g = pose.copy(), pose.copy()
    pose_p[:, :3, 3], pose_g[:, :3, 3] = s["p"], s["g"]
    cloud_p = [s["p"][t] + 0.01 * np.eye(3) for t in range(T)]
    cloud_g = [s["g"][t] + 0.01 * np.eye(3) for t in range(T)]
    ref2 = score(pose_p, pose_g, pred_object_points=cloud_p, gt_object_points=cloud_g).as_dict()
    got2 = score(pose_p[ip], pose_g[ig], pred_object_points=[cloud_p[i] for i in ip],
                 gt_object_points=[cloud_g[i] for i in ig], pred_ids=ids[ip], gt_ids=ids[ig]).as_dict()
    assert _same(got2, ref2)


def test_api_sorted_ids_equal_no_ids_bit_for_bit() -> None:
    s = _scene()
    ref = score(s["p"], s["g"], s["ph"], s["h"], K).as_dict()
    assert _same(score(s["p"], s["g"], s["ph"], s["h"], K, pred_ids=np.arange(T), gt_ids=np.arange(T)).as_dict(),
                 ref)
    assert _same(score(s["p"], s["g"], s["ph"], s["h"], K, pred_ids=np.arange(T, dtype=np.uint16),
                       gt_ids=np.arange(T, dtype=np.int32)).as_dict(), ref)


# --- no ids: bit-identical to 0.3.1 ----------------------------------------------------------------------------

def _golden() -> dict:
    return json.loads((DATA / "no_ids_golden_0.3.1.json").read_text())


MAX_ULP = 8


def _match_golden(got: object, want: object, path: str = "") -> None:
    """Equal to the recorded 0.3.1 output: integers, booleans, zeros and NaN exactly; other floats to MAX_ULP units in
    the last place (numpy 2.1.2 and 2.5.4 differ by 1 ulp on some values, also for the unchanged 0.3.1 code); a value
    recorded below 1e-9 in magnitude (rounding of an exact zero, build-dependent) must stay below 1e-9."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), path
        for k in want:
            _match_golden(got[k], want[k], f"{path}.{k}")
        return
    if isinstance(want, str):                           # a float, as float.hex
        w = float.fromhex(want)
        assert isinstance(got, float), path
        if w != 0.0 and abs(w) < 1e-9:
            assert abs(got) < 1e-9, path
        elif w == 0.0 or not np.isfinite(w):
            assert float.hex(got) == want, (path, float.hex(got), want)
        else:
            assert abs(got - w) <= MAX_ULP * np.spacing(abs(w)), (path, float.hex(got), want)
        return
    assert got == want, path


def test_no_ids_quickstart_is_bit_identical_to_0_3_1(tmp_path: Path) -> None:
    """The README quickstart (examples/make_example.py, then the command) without ids, against 0.3.1's output."""
    runpy.run_path(str(PKG / "examples" / "make_example.py"), run_name="example")["main"](tmp_path)
    out = run([str(tmp_path / "pred.npz"), str(tmp_path / "gt.npz"), "--intrinsics", "600,600,320,240"])
    _match_golden(out, _golden()["quickstart_cli"])


def test_no_ids_readme_example_and_known_answers_are_bit_identical_to_0_3_1() -> None:
    """The README's minimal example and the KNOWN_ANSWERS scale case, called without ids, against 0.3.1's output."""
    gold = _golden()
    rng = np.random.default_rng(0)
    n = 30
    gt_obj = np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])
    gt_hand = gt_obj[:, None, :] + 0.03 * rng.standard_normal((n, 21, 3))
    res = score(1.1 * gt_obj, gt_obj, gt_hand + 0.1 * gt_hand[:, :1], gt_hand, k=K).as_dict()
    _match_golden(res, gold["readme_minimal_example"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import test_scale_known_answer as ka
    finally:
        sys.path.pop(0)
    f = ka._fixture()
    scaled = f["base"] + (ka.K_SCALE - 1.0) * f["base"][:, :1]
    _match_golden(score(f["obj"], f["obj"], f["base"], f["gt"]).as_dict(), gold["known_answer_scale_base"])
    _match_golden(score(f["obj"], f["obj"], scaled, f["gt"]).as_dict(), gold["known_answer_scale_shifted"])


# --- command line ----------------------------------------------------------------------------------------------------

def _cli(tmp: Path, pred: dict, gt: dict, *extra: str) -> tuple[int, dict]:
    np.savez(tmp / "p.npz", **pred)
    np.savez(tmp / "g.npz", **gt)
    out = tmp / "o.json"
    out.unlink(missing_ok=True)
    rc = main([str(tmp / "p.npz"), str(tmp / "g.npz"), "--out", str(out), *extra])
    return rc, (json.loads(out.read_text()) if rc == 0 else {})


def test_cli_shift_by_one_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = _scene()
    rc, _ = _cli(tmp_path, {"object": s["p"], "frame_ids": np.arange(T) + 1},
                 {"object": s["g"], "frame_ids": np.arange(T)})
    assert rc == 2 and "same set" in capsys.readouterr().err


def test_cli_missing_frame_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = _scene()
    keep = np.delete(np.arange(T), 5)
    rc, _ = _cli(tmp_path, {"object": s["p"][keep], "frame_ids": keep}, {"object": s["g"], "frame_ids": np.arange(T)})
    assert rc == 2 and "same set" in capsys.readouterr().err


def test_cli_duplicate_ids_are_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = _scene()
    dup = np.arange(T)
    dup[3] = 2
    rc, _ = _cli(tmp_path, {"object": s["p"], "frame_ids": dup}, {"object": s["g"], "frame_ids": dup})
    assert rc == 2 and "duplicate" in capsys.readouterr().err


@pytest.mark.parametrize("case", ["one_sided", "float_ids", "with_frame_idx"])
def test_cli_malformed_frame_ids_are_refused(tmp_path: Path, case: str, capsys: pytest.CaptureFixture[str]) -> None:
    s = _scene()
    p, g = {"object": s["p"], "frame_ids": np.arange(T)}, {"object": s["g"], "frame_ids": np.arange(T)}
    if case == "one_sided":
        del g["frame_ids"]
        msg = "'frame_ids' must be in both files or in neither"
    elif case == "float_ids":
        p["frame_ids"] = np.arange(T) + 0.0
        msg = "integer"
    else:
        p["frame_idx"] = g["frame_idx"] = np.arange(T)
        msg = "not both"
    rc, _ = _cli(tmp_path, p, g)
    assert rc == 2 and msg in capsys.readouterr().err


def test_cli_shuffled_ids_give_the_sorted_result_bit_for_bit(tmp_path: Path) -> None:
    s = _scene()
    rng = np.random.default_rng(5)
    ip, ig = rng.permutation(T), rng.permutation(T)
    ids = np.arange(T) + 40
    rc0, ref = _cli(tmp_path, {"object": s["p"], "hand": s["ph"]}, {"object": s["g"], "hand": s["h"]},
                    "--intrinsics", "600,600,320,240", "--scene")
    rc1, got = _cli(tmp_path, {"object": s["p"][ip], "hand": s["ph"][ip], "frame_ids": ids[ip]},
                    {"object": s["g"][ig], "hand": s["h"][ig], "frame_ids": ids[ig]},
                    "--intrinsics", "600,600,320,240", "--scene", "--require-frame-ids")
    assert rc0 == rc1 == 0
    assert _same(got, ref)


def test_cli_frame_ids_with_valid_mask(tmp_path: Path) -> None:
    """`valid` masks each file's own rows before the join."""
    s = _scene()
    ip = np.random.default_rng(2).permutation(T)
    vp = np.ones(T, bool)
    vp[ip == 4] = False                                  # the prediction's row that holds frame 4
    rc, got = _cli(tmp_path, {"object": s["p"][ip], "frame_ids": ip, "valid": vp},
                   {"object": s["g"], "frame_ids": np.arange(T)})
    keep = np.arange(T) != 4
    ref = score(s["p"][keep], s["g"][keep]).as_dict()
    assert rc == 0 and got["object"]["n_frames"] == T - 1 and got["object"]["n_frames_input"] == T
    assert got["object"]["e_par"] == ref["object"]["e_par"] and got["object"]["lam"] == ref["object"]["lam"]


def test_cli_require_frame_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = _scene()
    rc, _ = _cli(tmp_path, {"object": s["p"]}, {"object": s["g"]}, "--require-frame-ids")
    assert rc == 2 and "--require-frame-ids" in capsys.readouterr().err
    rc, _ = _cli(tmp_path, {"object": s["p"], "frame_idx": np.arange(T)}, {"object": s["g"], "frame_idx": np.arange(T)},
                 "--require-frame-ids")
    assert rc == 2 and "--require-frame-ids" in capsys.readouterr().err
    ids = np.arange(T)
    rc, out = _cli(tmp_path, {"object": s["p"], "frame_ids": ids}, {"object": s["g"], "frame_ids": ids},
                   "--require-frame-ids")
    assert rc == 0 and _same(out, _cli(tmp_path, {"object": s["p"]}, {"object": s["g"]})[1])


def test_uint64_ids_beyond_int64_are_refused() -> None:
    """A uint64 id above the int64 range would wrap to a negative id; it is refused, not wrapped."""
    s = _scene()
    big = np.arange(T, dtype=np.uint64)
    big[0] = np.uint64(2 ** 63)
    wrapped = np.arange(T, dtype=np.int64)
    wrapped[0] = -(2 ** 63)
    with pytest.raises(ValueError, match="int64"):
        score(s["p"], s["g"], pred_ids=big, gt_ids=wrapped)
