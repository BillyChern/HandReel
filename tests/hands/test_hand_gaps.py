"""Tests for `fastdo.hands.hand_gaps`: MANO-parameter lerp/slerp across interior detection gaps."""

from __future__ import annotations

import numpy as np

from fastdo.hands.hand_gaps import fill_interior_gaps, slerp_rotvec


def _rows(valid: list[bool]) -> dict[str, np.ndarray]:
    t = len(valid)
    mask = np.array(valid)
    rows = {"rot": np.full((t, 3), np.nan, np.float32), "hand_pose": np.full((t, 45), np.nan, np.float32),
            "trans": np.full((t, 3), np.nan, np.float32), "betas": np.full((t, 10), np.nan, np.float32),
            "valid": mask}
    for key in ("rot", "hand_pose", "trans", "betas"):
        rows[key][mask] = 0.0
    return rows


def test_fill_interior_gap_lerp_and_slerp() -> None:
    rows = _rows([True, True, False, False, True, True])
    rows["trans"][1] = [0.0, 0.0, 1.0]
    rows["trans"][4] = [0.3, 0.0, 1.0]
    rows["rot"][4] = [0.0, 0.0, np.pi / 2]
    rows["hand_pose"][4] = np.tile([0.0, 0.0, np.pi / 2], 15)
    rows["betas"][4] = 1.0
    frame_idx = np.array([0, 3, 6, 9, 12, 15], dtype=np.int64)
    before = {k: v.copy() for k, v in rows.items()}

    out, filled = fill_interior_gaps(rows, frame_idx)

    assert filled.tolist() == [False, False, True, True, False, False]
    assert out["valid"].all()
    for row, third in ((2, 1), (3, 2)):
        np.testing.assert_allclose(out["rot"][row], [0.0, 0.0, third * np.pi / 6], atol=1e-6)
        np.testing.assert_allclose(out["hand_pose"][row].reshape(15, 3), np.tile([0.0, 0.0, third * np.pi / 6], (15, 1)),
                                   atol=1e-6)
        np.testing.assert_allclose(out["trans"][row], [0.1 * third, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(out["betas"][row], np.full(10, third / 3), atol=1e-6)
    assert all(out[k].dtype == np.float32 for k in ("rot", "hand_pose", "trans", "betas"))
    for key, value in before.items():                                # inputs unchanged
        np.testing.assert_array_equal(rows[key], value)


def test_fill_respects_max_gap() -> None:
    frame_idx = np.array([0, 121, 242], dtype=np.int64)
    rows = _rows([True, False, True])

    out, filled = fill_interior_gaps(rows, frame_idx, max_gap=120)
    assert filled.tolist() == [False, False, False] and out["valid"].tolist() == [True, False, True]
    assert np.isnan(out["trans"][1]).all()

    out, filled = fill_interior_gaps(rows, frame_idx, max_gap=242)
    assert filled.tolist() == [False, True, False] and out["valid"].all()


def test_fill_leaves_edges_and_empty_sides_invalid() -> None:
    frame_idx = np.arange(5, dtype=np.int64) * 3
    out, filled = fill_interior_gaps(_rows([False, True, False, True, False]), frame_idx)
    assert filled.tolist() == [False, False, True, False, False]
    assert out["valid"].tolist() == [False, True, True, True, False]

    empty = _rows([False] * 5)
    out, filled = fill_interior_gaps(empty, frame_idx)
    assert not filled.any() and not out["valid"].any()
    for key in ("rot", "hand_pose", "trans", "betas"):
        np.testing.assert_array_equal(out[key], empty[key])


def test_slerp_rotvec_shortest_arc() -> None:
    r0 = np.array([[0.0, 0.0, np.deg2rad(170.0)]])
    r1 = np.array([[0.0, 0.0, np.deg2rad(-170.0)]])

    mid = slerp_rotvec(r0, r1, np.array([0.5]))

    assert mid.shape == (1, 1, 3) and mid.dtype == np.float64
    assert abs(np.linalg.norm(mid[0, 0]) - np.pi) <= 1e-6
    from scipy.spatial.transform import Rotation

    angle = (Rotation.from_rotvec(r0[0]).inv() * Rotation.from_rotvec(mid[0, 0])).magnitude()
    assert abs(angle - np.deg2rad(10.0)) <= 1e-6
