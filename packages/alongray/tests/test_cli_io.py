# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""CLI input handling: 0.1 compatibility, frame_idx matching, frame masks, the camera-frame check, --scene, low_n."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from alongray import e_par_split, score
from alongray.cli import _jsonable, main

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _traj(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def _run(tmp: Path, pred: dict, gt: dict, *extra: str) -> tuple[int, dict]:
    np.savez(tmp / "p.npz", **pred)
    np.savez(tmp / "g.npz", **gt)
    out = tmp / "o.json"
    out.unlink(missing_ok=True)
    rc = main([str(tmp / "p.npz"), str(tmp / "g.npz"), "--out", str(out), *extra])
    return rc, (json.loads(out.read_text()) if rc == 0 else {})


GOLDEN_01 = Path(__file__).resolve().parent / "data/v01_unequal_T.json"


def test_01_style_unequal_body_lengths_still_score(tmp_path: Path) -> None:
    """0.1 scored each body on its own: an object over 10 frames and a hand over 8 must still give rc 0, and every
    0.1 field must equal the golden output of the 0.1 package (tests/data/v01_unequal_T.json)."""
    g, h = _traj(10, 1), _traj(8, 2)
    p, ph = 1.1 * g, 0.9 * h
    rc, out = _run(tmp_path, {"object": p, "hand": ph}, {"object": g, "hand": h})
    assert rc == 0
    golden = json.loads(GOLDEN_01.read_text())["output"]
    for body, fields in golden.items():
        for key, value in fields.items():
            # floats to 1e-12 (the golden file was written with one numpy/LAPACK build; residuals of ~1e-16 differ
            # between builds), everything else exactly
            if isinstance(value, (dict, float)):
                assert out[body][key] == pytest.approx(value, rel=1e-12, abs=1e-12), (body, key)
            else:
                assert out[body][key] == value, (body, key)
    assert out == json.loads(json.dumps(_jsonable(score(p, g, ph, h).as_dict())))


def test_frame_idx_matching_and_masks(tmp_path: Path) -> None:
    g = _traj()
    rng = np.random.default_rng(3)
    p = 1.1 * g + rng.normal(scale=0.005, size=g.shape)
    order = rng.permutation(10)[:7]                      # prediction has 7 of the 10 frames, shuffled
    vg = np.ones(10, bool)
    vg[order[0]] = False                                 # a GT-masked frame inside the overlap
    vp = np.array([1, 0, 1, 1, 1, 1, 1], np.int64)       # a prediction-masked frame, 0/1 integers
    rc, out = _run(tmp_path, {"object": p[order], "frame_idx": order, "valid": vp},
                   {"object": g, "frame_idx": np.arange(10), "valid": vg})
    assert rc == 0
    keep = np.sort(order[2:])
    assert out["object"]["n_frames"] == 5
    assert out["object"]["e_par"] == pytest.approx(e_par_split(p[keep], g[keep]).e_par, rel=1e-12)


CASES = {
    "one_sided_frame_idx": "'frame_idx' must be in both files or in neither",
    "dup_frame_idx": "frame_idx has duplicates",
    "no_overlap": "share no frame_idx",
    "float_frame_idx": "1-D integer array",
    "short_frame_idx": "one id per row (10), got 9",
    "long_frame_idx": "one id per row (10), got 11",
    "huge_frame_idx": "must fit in int64",
    "mask_shape": "frame mask",
    "mask_values": "boolean or 0/1",
    "count_mismatch_with_valid": "frame count",
    "sign_change": "changes sign by more than",
    "mirrored_prediction": "looks mirrored",
    "bad_image_size": "'image_size' must be (W, H)",
    "scene_without_hand": "--scene needs a 'hand'",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_refusals_exit_2_with_their_message(tmp_path: Path, case: str, capsys: pytest.CaptureFixture[str]) -> None:
    g = _traj()
    p, gt = {"object": 1.1 * g}, {"object": g}
    extra: list[str] = []
    if case == "one_sided_frame_idx":
        p["frame_idx"] = np.arange(10)
    elif case == "dup_frame_idx":
        p["frame_idx"] = gt["frame_idx"] = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    elif case == "no_overlap":
        p["frame_idx"], gt["frame_idx"] = np.arange(10), np.arange(10) + 50
    elif case == "float_frame_idx":
        p["frame_idx"] = gt["frame_idx"] = np.arange(10) + 0.5
    elif case == "short_frame_idx":                 # one row would have no id and be dropped silently
        p["frame_idx"], gt["frame_idx"] = np.arange(9), np.arange(10)
    elif case == "long_frame_idx":                  # an id for a row that does not exist
        p["frame_idx"], gt["frame_idx"] = np.arange(10), np.arange(11)
    elif case == "huge_frame_idx":
        p["frame_idx"] = gt["frame_idx"] = np.arange(10, dtype=np.uint64) + np.uint64(2**63)
    elif case == "mask_shape":
        gt["valid"] = np.ones((10, 2), bool)
    elif case == "mask_values":
        gt["valid"] = np.full(10, 2, np.int64)
    elif case == "count_mismatch_with_valid":
        p["hand"], gt["hand"], gt["valid"] = 1.1 * g[:8], g[:8], np.ones(10, bool)
    elif case == "sign_change":
        w = g.copy()
        w[:5, 2] *= -1.0
        p, gt = {"object": 1.1 * w}, {"object": w}
    elif case == "mirrored_prediction":
        p = {"object": 1.1 * g * np.array([1.0, -1.0, -1.0])}
    elif case == "bad_image_size":
        gt["image_size"] = np.array([640.0, 0.0])
    else:
        extra = ["--scene"]
    rc, _ = _run(tmp_path, p, gt, *extra)
    assert rc == 2
    assert CASES[case] in capsys.readouterr().err


def _warned(tmp: Path, pred: dict, gt: dict, match: str) -> dict:
    with pytest.warns(UserWarning, match=match):
        rc, out = _run(tmp, pred, gt)
    assert rc == 0
    return out


def test_z_up_world_frame_warns_with_and_without_image_bounds(tmp_path: Path) -> None:
    """Objects resting on a z-up table (z 2-15 cm, x/y 0.3-0.8 m): far outside any camera cone, but only warned."""
    rng = np.random.default_rng(5)
    w = np.column_stack([rng.uniform(0.3, 0.8, 10), rng.uniform(0.3, 0.8, 10), rng.uniform(0.02, 0.15, 10)])
    _warned(tmp_path, {"object": 1.05 * w}, {"object": w, "K": K, "image_size": np.array([640, 480])},
            "project near the")
    _warned(tmp_path, {"object": 1.05 * w}, {"object": w}, "wider than any ordinary lens")


def test_wrist_out_of_frame_warns_for_the_hand_only(tmp_path: Path) -> None:
    """A wrist outside the image (only the fingers in view) is a valid camera-frame input: rc 0, a hand warning."""
    g = _traj(seed=1)
    joints = g[:, None, :] + np.random.default_rng(0).normal(scale=0.03, size=(10, 21, 3))
    joints[:, 0, 0] += 1.0                                # wrist 1 m to the side at ~0.6 m: outside the image
    with pytest.warns(UserWarning) as rec:
        rc, out = _run(tmp_path, {"object": 1.1 * g, "hand": 1.1 * joints},
                       {"object": g, "hand": joints, "K": K, "image_size": np.array([640, 480])})
    assert rc == 0 and out["hand"]["lam"] == pytest.approx(1.1)
    msgs = [str(r.message) for r in rec if "project near" in str(r.message)]
    assert msgs and all("hand" in m for m in msgs)


def test_off_centre_principal_point_is_never_refused(tmp_path: Path) -> None:
    """cx = cy = 0 (principal point at the image corner) must not refuse: bounds never come from cx, cy."""
    g = _traj(seed=2)
    k0 = np.array([[600.0, 0.0, 0.0], [0.0, 600.0, 0.0], [0.0, 0.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rc, out = _run(tmp_path, {"object": 1.1 * g}, {"object": g, "K": k0})
    assert rc == 0 and out["object"]["lam"] == pytest.approx(1.1)


def test_bad_prediction_on_the_wrong_side_is_scored_with_a_warning(tmp_path: Path) -> None:
    """6 of 10 predicted frames at z = -0.02 is a bad prediction, not a mirrored one: scored, with a warning."""
    g = _traj(seed=3)
    p = 1.1 * g
    p[:6, 2] = -0.02
    _warned(tmp_path, {"object": p}, {"object": g}, "other side of the camera")
    p[:6, 2] = -0.3                                      # far behind, but 60% < 90%: still not mirrored
    _warned(tmp_path, {"object": p}, {"object": g}, "other side of the camera")


def test_origin_just_behind_the_camera_is_accepted_with_a_warning(tmp_path: Path) -> None:
    g = _traj()
    g[:, 2] = np.linspace(-0.02, 0.12, 10)
    g[:, :2] *= 0.1
    with pytest.warns(UserWarning, match="just behind the camera"):
        rc, out = _run(tmp_path, {"object": 1.1 * g}, {"object": g})
    assert rc == 0 and out["object"]["lam"] == pytest.approx(1.1)


def test_opengl_axes_still_accepted(tmp_path: Path) -> None:
    g = _traj() * np.array([1.0, -1.0, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rc, out = _run(tmp_path, {"object": 1.1 * g}, {"object": g})
    assert rc == 0 and out["object"]["lam"] == pytest.approx(1.1)


def test_scene_flag_and_low_n_warning(tmp_path: Path) -> None:
    g, h = _traj(seed=1), _traj(seed=2)
    rc, out = _run(tmp_path, {"object": 1.2 * g, "hand": 0.8 * h}, {"object": g, "hand": h}, "--scene")
    assert rc == 0 and out["scene"]["relative_log_gauge"]["object/hand"] == pytest.approx(np.log(1.5), rel=1e-12)
    with pytest.warns(UserWarning, match="low_n"):
        rc, out = _run(tmp_path, {"object": 1.2 * g[:2]}, {"object": g[:2]})
    assert rc == 0 and out["object"]["low_n"] is True
