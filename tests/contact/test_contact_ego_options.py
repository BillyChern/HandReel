"""The two opt-in options of the contact ruler: bias-free constants and the touch gate.

Both must leave the promoted behaviour exactly as it was when they are off, and do only what they say when on.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.contact import (
    ContactConstants,
    contact_correction,
    contact_profile,
    correct_poses,
    posterior_mean_shift,
)

C = ContactConstants(a=0.68, b_mm=-25.0, mu_mm=-40.0, sig_mm=86.0, tau_mm=20.0)
GRID_MM = np.arange(-300, 301, 2, dtype=float)


def _scene(t: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A 4 cm sphere at 0.6 m; frames 0-2 have a hand touching it, frames 3-4 a hand 25 cm to the side."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=(800, 3))
    mesh = 0.04 * s / np.linalg.norm(s, axis=1, keepdims=True)
    poses = np.tile(np.eye(4), (t, 1, 1))
    poses[:, :3, 3] = (0.0, 0.0, 0.60)
    joints = np.zeros((t, 21, 3))
    joints[:3] = (0.0, 0.0, 0.66) + 0.005 * rng.normal(size=(3, 21, 3))
    joints[3:] = (0.25, 0.0, 0.60) + 0.005 * rng.normal(size=(t - 3, 21, 3))
    return poses, mesh, joints, np.ones(t, bool)


def test_bias_free_zeros_exactly_the_two_bias_terms() -> None:
    bf = C.bias_free()
    assert (bf.b_mm, bf.mu_mm) == (0.0, 0.0)
    assert (bf.a, bf.sig_mm, bf.tau_mm) == (C.a, C.sig_mm, C.tau_mm)
    assert (C.b_mm, C.mu_mm) == (-25.0, -40.0)                  # frozen: the original is untouched


def test_bias_free_on_a_flat_profile_returns_about_zero_not_the_prior_mean() -> None:
    """With no contact information the promoted ruler returns ~mu (-40 mm); the bias-free one returns ~0."""
    flat = np.full_like(GRID_MM, 7.0)
    assert posterior_mean_shift(flat, GRID_MM, C) < -30.0
    assert abs(posterior_mean_shift(flat, GRID_MM, C.bias_free())) < 1.0


def test_no_gate_is_bit_identical_to_correct_poses() -> None:
    poses, mesh, joints, valid = _scene()
    ref_poses, ref_shift = correct_poses(poses, 1.0, mesh, joints, valid, C)
    res = contact_correction(poses, 1.0, mesh, joints, valid, C)
    assert res.poses.tobytes() == ref_poses.tobytes()
    assert res.shift_mm.tobytes() == ref_shift.tobytes()
    assert not res.gated.any()


def test_min_cost_is_the_profile_minimum() -> None:
    poses, mesh, joints, valid = _scene()
    res = contact_correction(poses, 1.0, mesh, joints, valid, C)
    for t in range(len(poses)):
        cp = poses[t, :3, :3] @ mesh.mean(0) + poses[t, :3, 3]
        prof = contact_profile(mesh + poses[t, :3, 3], joints[t], cp / np.linalg.norm(cp), GRID_MM / 1e3)
        assert res.min_cost_mm[t] == pytest.approx(prof.min(), abs=1e-9)
    assert np.all(res.min_cost_mm[:3] < 15.0) and np.all(res.min_cost_mm[3:] > 150.0)


def test_the_gate_corrects_touching_frames_and_leaves_the_others_exactly_as_solved() -> None:
    poses, mesh, joints, valid = _scene()
    ungated = contact_correction(poses, 1.0, mesh, joints, valid, C)
    gated = contact_correction(poses, 1.0, mesh, joints, valid, C, touch_gate_mm=40.0)
    assert gated.gated.tolist() == [False, False, False, True, True]
    assert gated.poses[:3].tobytes() == ungated.poses[:3].tobytes()
    assert gated.poses[3:].tobytes() == poses[3:].tobytes()
    assert np.all(gated.shift_mm[3:] == 0) and np.all(ungated.shift_mm[3:] != 0)


def test_frames_without_a_hand_are_neither_corrected_nor_counted_as_gated() -> None:
    poses, mesh, joints, valid = _scene()
    valid[1] = False
    res = contact_correction(poses, 1.0, mesh, joints, valid, C, touch_gate_mm=40.0)
    assert np.isnan(res.min_cost_mm[1]) and not res.gated[1] and res.shift_mm[1] == 0


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_a_nonsense_gate_is_refused(bad: float) -> None:
    poses, mesh, joints, valid = _scene()
    with pytest.raises(ValueError, match="touch_gate_mm"):
        contact_correction(poses, 1.0, mesh, joints, valid, C, touch_gate_mm=bad)
