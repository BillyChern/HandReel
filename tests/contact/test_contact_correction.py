"""The promoted contact correction: properties the estimator must have, not a copy of its formula.

The promoted correction is the Bayesian form: the
along-ray shift is the posterior mean of the correction under a Gaussian prior and a likelihood built from the
contact-cost profile. These tests pin the behaviour that makes it safe in a pipeline:
the two limits (a sharp profile follows the data, a flat profile follows the prior), monotonicity, and no overflow.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.contact import (
    ContactConstants,
    contact_identifiability,
    contact_profile,
    correct_poses,
    posterior_mean_shift,
    posterior_shift_moments,
)

GRID_MM = np.arange(-300, 301, 2, dtype=float)
C = ContactConstants(a=0.68, b_mm=-25.0, mu_mm=-40.0, sig_mm=86.0, tau_mm=20.0)


def peaked(centre_mm: float, width_mm: float = 8.0) -> np.ndarray:
    """A contact profile whose minimum sits at `centre_mm` (mm units, like the real cost)."""
    return 10.0 + ((GRID_MM - centre_mm) / width_mm) ** 2


def _peaked_on(grid_mm: np.ndarray, centre_mm: float, width_mm: float) -> np.ndarray:
    """`peaked` for a grid other than GRID_MM, which the narrow-grid trap test needs."""
    return 10.0 + ((grid_mm - centre_mm) / width_mm) ** 2


def _on_grid_prior_mean(grid_mm: np.ndarray) -> float:
    """E[c] under the prior RESTRICTED TO THE GRID -- what a flat profile must return."""
    c_grid = C.a * grid_mm + C.b_mm
    prior = np.exp(-((c_grid - C.mu_mm) ** 2) / (2.0 * C.sig_mm ** 2))
    return float((prior * c_grid).sum() / prior.sum())


def test_a_sharp_profile_follows_the_data() -> None:
    """tau -> 0 makes the likelihood dominate: the shift becomes the calibrated map of the profile's minimum."""
    sharp = ContactConstants(a=C.a, b_mm=C.b_mm, mu_mm=C.mu_mm, sig_mm=C.sig_mm, tau_mm=0.05)
    for centre in (-120.0, -40.0, 0.0, 60.0, 180.0):
        got = posterior_mean_shift(peaked(centre), GRID_MM, sharp)
        assert got == pytest.approx(C.a * centre + C.b_mm, abs=1.0)


def test_a_flat_profile_follows_the_prior() -> None:
    """No contact information must leave the PRIOR, truncated to the grid's calibrated image.

    The truncation is not a detail: with a = 0.68 and b = -25 mm the grid covers [-229, +179] mm, which is
    asymmetric about mu = -40 mm, so the honest expectation is a truncated-normal mean (-38.3 mm), not mu.
    """
    from scipy import stats

    flat = np.full_like(GRID_MM, 7.0)
    lo, hi = C.a * GRID_MM[0] + C.b_mm, C.a * GRID_MM[-1] + C.b_mm
    want = stats.truncnorm.mean((lo - C.mu_mm) / C.sig_mm, (hi - C.mu_mm) / C.sig_mm, C.mu_mm, C.sig_mm)
    assert posterior_mean_shift(flat, GRID_MM, C) == pytest.approx(float(want), abs=0.2)


def test_the_shift_is_monotone_in_the_profile_minimum() -> None:
    shifts = [posterior_mean_shift(peaked(c), GRID_MM, C) for c in np.arange(-200, 201, 25.0)]
    assert all(b > a for a, b in zip(shifts, shifts[1:]))


def test_the_shift_stays_inside_the_calibrated_range() -> None:
    """A shift outside the grid's calibrated image would be an extrapolation the fit never justified."""
    lo, hi = sorted((C.a * GRID_MM[0] + C.b_mm, C.a * GRID_MM[-1] + C.b_mm))
    for c in (-300.0, -299.0, 0.0, 299.0, 300.0):
        assert lo - 1e-6 <= posterior_mean_shift(peaked(c), GRID_MM, C) <= hi + 1e-6


def test_no_overflow_on_a_huge_cost_range() -> None:
    profile = peaked(0.0) * 1e6                     # (C - Cmin)^2 / tau^2 overflows a naive exp
    got = posterior_mean_shift(profile, GRID_MM, C)
    assert np.isfinite(got)


def test_profile_minimum_sits_at_the_true_offset() -> None:
    """contact_profile on a sphere: joints ON the surface make delta = 0 the minimum, and sliding moves it."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3))
    surface = 0.05 * d / np.linalg.norm(d, axis=1, keepdims=True) + np.array([0.0, 0.0, 0.6])
    ray = np.array([0.0, 0.0, 1.0])
    joints = surface[:21]                            # a "hand" lying on the sphere
    prof = contact_profile(surface, joints, ray, GRID_MM / 1000.0)
    assert GRID_MM[int(np.argmin(prof))] == pytest.approx(0.0, abs=2.0)

    moved = joints + 0.040 * ray                     # hand 40 mm along the ray from the surface
    prof2 = contact_profile(surface, moved, ray, GRID_MM / 1000.0)
    assert GRID_MM[int(np.argmin(prof2))] == pytest.approx(40.0, abs=4.0)


def test_correct_poses_shifts_along_the_predicted_centroid_ray_only() -> None:
    rng = np.random.default_rng(1)
    mesh = rng.normal(scale=0.03, size=(500, 3))
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, 3] = np.array([0.1, 0.05, 0.7])
    joints = np.tile(np.array([0.1, 0.05, 0.75]), (4, 21, 1))
    valid = np.ones(4, dtype=bool)

    out, shifts = correct_poses(poses, 1.0, mesh, joints, valid, C)

    assert out.shape == poses.shape and shifts.shape == (4,)
    centroid_cam = poses[0, :3, :3] @ mesh.mean(axis=0) + poses[0, :3, 3]   # the ray is to the CENTROID
    for t in range(4):
        delta = out[t, :3, 3] - poses[t, :3, 3]
        u = centroid_cam / np.linalg.norm(centroid_cam)
        assert np.linalg.norm(delta - (delta @ u) * u) < 1e-9        # no perpendicular component
        assert np.allclose(out[t, :3, :3], poses[t, :3, :3])          # rotation untouched
    assert np.allclose(shifts, shifts[0])                            # identical frames -> identical shift


def test_invalid_hand_frames_are_left_alone() -> None:
    mesh = np.random.default_rng(2).normal(scale=0.03, size=(200, 3))
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, 3] = np.array([0.0, 0.0, 0.6])
    joints = np.full((3, 21, 3), np.nan)
    joints[1] = np.array([0.0, 0.0, 0.65])
    valid = np.array([False, True, False])

    out, shifts = correct_poses(poses, 1.0, mesh, joints, valid, C)

    assert np.allclose(out[0], poses[0]) and np.allclose(out[2], poses[2])
    assert shifts[0] == 0.0 and shifts[2] == 0.0 and shifts[1] != 0.0


def test_constants_reject_a_missing_or_nonsense_field() -> None:
    with pytest.raises(ValueError, match="tau_mm"):
        ContactConstants(a=0.7, b_mm=0.0, mu_mm=0.0, sig_mm=50.0, tau_mm=0.0)
    with pytest.raises(ValueError, match="sig_mm"):
        ContactConstants(a=0.7, b_mm=0.0, mu_mm=0.0, sig_mm=-1.0, tau_mm=20.0)


def test_the_shipped_production_constants_load_for_every_cell() -> None:
    """The engine will read these; a missing or nonsense entry must fail here, not in a run."""
    from pathlib import Path

    from fastdo.contact import load_constants

    path = Path(__file__).resolve().parents[2] / "src/fastdo/contact/constants_production.json"
    for cell in ("prod", "oraclescale"):
        for hands in ("hawor", "gt"):
            c = load_constants(path, cell, hands)
            assert 0.3 < c.a < 1.5 and abs(c.b_mm) < 60 and 20 < c.sig_mm < 200 and c.tau_mm == 20.0
    with pytest.raises(KeyError, match="nosuch"):
        load_constants(path, "nosuch", "hawor")


def test_the_posterior_spread_is_the_half_the_point_estimate_throws_away() -> None:
    """The mean must be bit-identical to the shipped estimator, and the sd must track how sharp the profile is.

    The mean is the PROMOTED quantity with a measured parity of 5.7e-14 mm against the study scorer, so
    exposing the moments must not perturb it by a single ulp.
    """
    for profile in (peaked(0.0), peaked(120.0, width_mm=40.0), np.zeros_like(GRID_MM)):
        mean, sd = posterior_shift_moments(profile, GRID_MM, C)
        assert mean == posterior_mean_shift(profile, GRID_MM, C)     # bit-identical, not approx
        assert sd > 0.0

    # a sharper profile pins the correction harder: the spread must fall as the well narrows
    spreads = [posterior_shift_moments(peaked(0.0, width_mm=w), GRID_MM, C)[1] for w in (5.0, 20.0, 80.0, 320.0)]
    assert spreads == sorted(spreads), spreads


def test_identifiability_is_zero_on_an_uninformative_profile_and_one_on_a_sharp_one() -> None:
    """The two limits the module docstring names, now measured rather than asserted in prose."""
    flat = np.zeros_like(GRID_MM)
    assert contact_identifiability(flat, GRID_MM, C) == pytest.approx(0.0, abs=1e-12)
    # A flat profile leaves the point estimate at the prior mean -- but the ON-GRID one. The grid spans
    # c = -229..179 mm around mu = -40, so truncation pulls the mean up to -38.27; comparing against mu
    # itself would be comparing against a prior this estimator never actually uses.
    assert posterior_mean_shift(flat, GRID_MM, C) == pytest.approx(_on_grid_prior_mean(GRID_MM), abs=1e-9)

    assert contact_identifiability(peaked(0.0, width_mm=4.0), GRID_MM, C) > 0.98   # measured 0.9892
    # and it is monotone in sharpness
    scores = [contact_identifiability(peaked(0.0, width_mm=w), GRID_MM, C) for w in (4.0, 20.0, 80.0, 320.0)]
    assert scores == sorted(scores, reverse=True), scores


def test_a_bimodal_profile_scores_below_zero_because_the_contact_is_ambiguous() -> None:
    """Two wells at opposite ends spread the posterior WIDER than the prior. Negative is the honest answer:
    the contact evidence is actively ambiguous, which is exactly what an acceptance screen wants to hear."""
    bimodal = np.minimum(peaked(-280.0, width_mm=10.0), peaked(280.0, width_mm=10.0))
    assert contact_identifiability(bimodal, GRID_MM, C) < 0.0


def test_a_narrow_grid_cannot_fake_identifiability() -> None:
    """THE TRAP. Normalising by sig_mm**2 would make a narrow grid look well identified no matter the profile.

    On a grid far narrower than the prior (+-20 mm against sig 86 mm) an UNINFORMATIVE profile still has a
    posterior variance far below sig**2, so `1 - Var/sig**2` would read ~0.99 -- a confident screen built on
    the grid's width rather than on any evidence. Normalising by the prior ON THE SAME GRID gives 0, correctly.
    """
    narrow = np.arange(-20, 21, 2, dtype=float)
    flat = np.zeros_like(narrow)
    assert contact_identifiability(flat, narrow, C) == pytest.approx(0.0, abs=1e-12)

    _, sd = posterior_shift_moments(flat, narrow, C)
    would_be = 1.0 - sd ** 2 / C.sig_mm ** 2          # what the sig-normalised version would have reported
    assert would_be > 0.98, would_be                  # measured 0.991: a confident screen built on nothing

    # and the measure still works on that narrow grid: a sharp profile there is still identified
    assert contact_identifiability(_peaked_on(narrow, 0.0, 1.0), narrow, C) > 0.9


def test_identifiability_refuses_a_grid_with_no_prior_spread() -> None:
    one = np.array([0.0])
    with pytest.raises(ValueError, match="no variance on this grid"):
        contact_identifiability(np.zeros_like(one), one, C)
