# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Intrinsics must be a finite pinhole matrix with the canonical bottom row (A6).

The reprojection divides the first two image coordinates by the camera-frame depth z, which is the homogeneous
coordinate only when K's bottom row is [0, 0, 1]. Any other bottom row, a non-zero K[1, 0], a non-finite entry or a
non-positive focal length is refused; a finite skew K[0, 1] is a valid pinhole and is kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alongray import score
from alongray.cli import main
from alongray.score import intrinsics_matrix, reprojection_error_px

K_OK = np.array([[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])


def _traj(t: int = 8) -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.column_stack([rng.normal(0, 0.05, t), rng.normal(0, 0.05, t), rng.uniform(0.45, 0.75, t)])


@pytest.mark.parametrize("k, match", [
    (np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 2.0]]), "bottom row"),
    (np.array([[600.0, 0, 320], [0, 600, 240], [0.1, 0, 1.0]]), "bottom row"),
    (np.array([[600.0, 0, 320], [5.0, 600, 240], [0, 0, 1.0]]), r"K\[1, 0\]"),
    (np.array([[600.0, 0, np.nan], [0, 600, 240], [0, 0, 1.0]]), "finite"),
    (np.array([[np.inf, 0, 320], [0, 600, 240], [0, 0, 1.0]]), "finite"),
    ((600.0, np.nan, 320.0, 240.0), "finite"),
    ((600.0, -600.0, 320.0, 240.0), "focal"),
    ((600.0, 600.0, 320.0), "shape"),
])
def test_non_canonical_or_non_finite_intrinsics_are_refused(k, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        intrinsics_matrix(k)
    g = _traj()
    with pytest.raises(ValueError, match=match):
        score(1.1 * g, g, k=k)


def test_a_canonical_pinhole_with_skew_matches_the_homogeneous_projection() -> None:
    k = K_OK.copy()
    k[0, 1] = 2.5                                           # skew: still a pinhole with the canonical bottom row
    g = _traj()
    p = g + [0.004, -0.003, 0.02]
    hp, hg = p @ k.T, g @ k.T                               # homogeneous image points
    want = np.linalg.norm(hp[:, :2] / hp[:, 2:] - hg[:, :2] / hg[:, 2:], axis=1)
    np.testing.assert_allclose(reprojection_error_px(p, g, k), want, rtol=1e-12, atol=0)
    assert np.allclose(reprojection_error_px(1.3 * g, g, k), 0.0, atol=1e-9)   # a pure gauge moves no pixel


def test_a_point_behind_the_camera_is_not_projected() -> None:
    g = _traj()
    p = g.copy()
    p[2, 2] = -0.5
    err = reprojection_error_px(p, g, K_OK)
    assert np.isnan(err[2]) and np.isfinite(np.delete(err, 2)).all()


def test_the_command_line_refuses_bad_intrinsics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    g = _traj()
    bad = K_OK.copy()
    bad[2, 2] = 0.0
    np.savez(tmp_path / "p.npz", object=1.1 * g, K=bad)
    np.savez(tmp_path / "g.npz", object=g)
    assert main([str(tmp_path / "p.npz"), str(tmp_path / "g.npz")]) == 2
    assert "bottom row" in capsys.readouterr().err
    np.savez(tmp_path / "p.npz", object=1.1 * g, K=K_OK)
    assert main([str(tmp_path / "p.npz"), str(tmp_path / "g.npz"), "--out", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["object"]["reproj_rms_px"] == pytest.approx(0.0, abs=1e-9)
