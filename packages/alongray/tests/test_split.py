# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Known answers and mutations for the exact split e_par^2 = (lam - 1)^2 <Z^2> + <o^2> (alongray.core)."""

from __future__ import annotations

import numpy as np
import pytest

from alongray import along_ray_components, apply_gauge, e_par_split, gauge_decomposition


def _traj(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def test_split_known_answer_in_closed_form() -> None:
    """Z = (1, 2, 3) on the optical axis, o = (1, 1, -1) with sum Z o = 0, lam = 1.5, plus a perpendicular offset.

    Then lam is recovered exactly, gauge^2 = 0.25 * 14/3 = 7/6, off^2 = 1, e_par^2 = (2.25 + 4 + 0.25)/3 = 13/6.
    """
    z = np.array([1.0, 2.0, 3.0])
    o = np.array([1.0, 1.0, -1.0])
    gt = np.column_stack([np.zeros(3), np.zeros(3), z])
    perp = np.array([[0.3, 0.0, 0.0], [0.0, -0.2, 0.0], [0.1, 0.4, 0.0]])   # orthogonal to every true ray
    pred = 1.5 * gt + o[:, None] * np.array([0.0, 0.0, 1.0]) + perp
    sp = e_par_split(pred, gt)
    assert sp.lam == 1.5
    assert sp.gauge_part ** 2 == pytest.approx(7.0 / 6.0, rel=1e-15)
    assert sp.off_orbit_part ** 2 == pytest.approx(1.0, rel=1e-15)
    assert sp.e_par ** 2 == pytest.approx(13.0 / 6.0, rel=1e-15)
    assert sp.e_par ** 2 == pytest.approx(sp.gauge_part ** 2 + sp.off_orbit_part ** 2, rel=1e-15)
    assert sp.oracle_floor == pytest.approx(1.0 / 1.5, rel=1e-15)
    assert sp.z_rms ** 2 == pytest.approx(14.0 / 3.0, rel=1e-15)
    assert sp.n_frames == 3


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kappa", [1.0, 0.93, 1.2])
def test_split_sums_exactly_on_random_trajectories(seed: int, kappa: float) -> None:
    rng = np.random.default_rng(100 + seed)
    gt = _traj(40, seed)
    pred = rng.uniform(0.8, 1.2) * gt + rng.normal(0, 0.02, gt.shape)
    sp = e_par_split(pred, gt, kappa)
    assert sp.e_par ** 2 == pytest.approx(sp.gauge_part ** 2 + sp.off_orbit_part ** 2, rel=1e-12)
    # the closed form under a gauge move: e_par(kappa)^2 = (kappa lam - 1)^2 <Z^2> + kappa^2 <o^2>
    base = e_par_split(pred, gt)
    expect = (kappa * base.lam - 1.0) ** 2 * base.z_rms ** 2 + kappa ** 2 * base.off_orbit_part ** 2
    assert sp.e_par ** 2 == pytest.approx(expect, rel=1e-10)


def test_normal_equation_is_what_makes_the_split_exact() -> None:
    """Mutation: at any other non-trivial orbit point w the cross term 2 (w - 1) <Z o_w> is not zero.

    (w = 1 is a trivial split, all of e_par in the second term, so it is not a useful mutant.)
    """
    rng = np.random.default_rng(5)
    gt = _traj(40, 5)
    pred = 1.09 * gt + rng.normal(0, 0.02, gt.shape)
    sp = e_par_split(pred, gt)
    z = np.linalg.norm(gt, axis=1)
    u = gt / z[:, None]
    o = np.einsum("ij,ij->i", pred - sp.lam * gt, u)
    assert abs(np.sum(z * o)) < 1e-12 * np.sum(z * z)
    for wrong in (sp.lam * 1.01, 0.5 * (1.0 + sp.lam), 1.0 + 2.0 * (sp.lam - 1.0)):
        o_w = np.einsum("ij,ij->i", pred - wrong * gt, u)
        split_w = (wrong - 1.0) ** 2 * np.mean(z ** 2) + np.mean(o_w ** 2)
        assert split_w != pytest.approx(sp.e_par ** 2, rel=1e-3)
        cross = 2.0 * (wrong - 1.0) * np.mean(z * o_w)
        assert split_w + cross == pytest.approx(sp.e_par ** 2, rel=1e-12)   # the general identity still holds


@pytest.mark.parametrize("lam", [0.7, 0.95, 1.1, 1.5])
def test_pure_scale_error_lands_entirely_in_the_gauge_part(lam: float) -> None:
    gt = _traj()
    sp = e_par_split(lam * gt, gt)
    assert sp.lam == pytest.approx(lam, rel=1e-14)
    assert sp.off_orbit_part == pytest.approx(0.0, abs=1e-14)
    assert sp.gauge_part == pytest.approx(abs(lam - 1.0) * sp.z_rms, rel=1e-14)
    assert sp.e_par == pytest.approx(sp.gauge_part, rel=1e-12)
    assert sp.oracle_floor == pytest.approx(0.0, abs=1e-14)
    assert e_par_split(lam * gt, gt, 1.0 / lam).e_par == pytest.approx(0.0, abs=1e-14)   # the oracle undoes it


def test_constant_along_ray_offset_puts_collinearity_of_its_square_in_the_gauge_part() -> None:
    """'Right size, wrong distance': +50 mm along every ray puts exactly collinearity * d^2 of its mean square in the
    gauge part (README, "The exact split"). On this window the range varies little (collinearity about 0.98), so the
    offset is mostly gauge, and only the remainder is off-orbit."""
    gt = _traj()
    u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    sp = e_par_split(gt + 0.05 * u, gt)
    assert sp.e_par == pytest.approx(0.05, rel=1e-12)
    assert sp.gauge_part ** 2 == pytest.approx(sp.collinearity * 0.05 ** 2, rel=1e-9)
    assert sp.off_orbit_part > 0.0 and sp.gauge_part > sp.off_orbit_part
    assert sp.e_par ** 2 == pytest.approx(sp.gauge_part ** 2 + sp.off_orbit_part ** 2, rel=1e-12)


def test_perpendicular_error_is_invisible_to_e_par() -> None:
    gt = _traj()
    u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    d = np.cross(u, np.array([0.0, 1.0, 0.0]))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sp = e_par_split(gt + 0.03 * d, gt)
    assert sp.e_par == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(along_ray_components(gt + 0.03 * d, gt).perp, 0.03, atol=1e-12)


def test_nan_frames_are_dropped_consistently() -> None:
    gt = _traj()
    pred = 1.1 * gt
    pred[3] = np.nan
    gt2 = gt.copy()
    gt2[7, 0] = np.inf
    sp = e_par_split(pred, gt2)
    keep = np.ones(len(gt), bool)
    keep[[3, 7]] = False
    ref = e_par_split(pred[keep], gt[keep])
    assert sp.n_frames == len(gt) - 2
    assert sp == ref


def test_errors() -> None:
    with pytest.raises(ValueError):
        e_par_split(np.ones((2, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError):
        e_par_split(np.full((2, 3), np.nan), np.ones((2, 3)))
    with pytest.raises(ValueError), np.errstate(invalid="ignore"):
        e_par_split(np.ones((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        apply_gauge(np.tile(np.eye(4), (2, 1, 1)), 0.1, 0.0)


def test_apply_gauge_is_on_the_orbit() -> None:
    gt = _traj()
    poses = np.tile(np.eye(4), (len(gt), 1, 1))
    poses[:, :3, 3] = gt
    moved, s = apply_gauge(poses, 0.12, 1.17)
    dec = gauge_decomposition(moved[:, :3, 3], gt, pred_size=s, gt_size=0.12)
    assert dec.lam == pytest.approx(1.17, abs=1e-12)
    assert dec.off_orbit_rms == pytest.approx(0.0, abs=1e-12)
    assert dec.orbit_coherence == pytest.approx(0.0, abs=1e-12)


def test_signed_along_has_the_sign_of_far_minus_near() -> None:
    """Mutation guard: e_par squares a_t, so only a direct check pins the sign convention (far = positive)."""
    gt = _traj()
    u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    np.testing.assert_allclose(along_ray_components(gt + 0.05 * u, gt).signed_along, 0.05, rtol=1e-12)
    np.testing.assert_allclose(along_ray_components(gt - 0.02 * u, gt).signed_along, -0.02, rtol=1e-12)
    np.testing.assert_allclose(along_ray_components(1.1 * gt, gt).log_depth_ratio, np.log(1.1), rtol=1e-12)


@pytest.mark.parametrize("factor", [-1.0, 0.0])
def test_non_positive_lambda_is_refused_with_a_convention_hint(factor: float) -> None:
    gt = _traj()
    with pytest.raises(ValueError, match="zero-filled"):
        e_par_split(factor * gt, gt)
    with pytest.raises(ValueError, match="units"):
        gauge_decomposition(factor * gt, gt)


@pytest.mark.parametrize("kappa", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_kappa_is_refused(kappa: float) -> None:
    gt = _traj()
    with pytest.raises(ValueError, match="kappa"):
        e_par_split(1.1 * gt, gt, kappa)


@pytest.mark.parametrize("shape", [(3,), (10, 6), (10, 1, 3), (10, 2)])
def test_non_trajectory_shapes_are_refused_not_reshaped(shape: tuple[int, ...]) -> None:
    x = np.ones(shape)
    for fn in (e_par_split, gauge_decomposition, along_ray_components):
        with pytest.raises(ValueError, match=r"\(T, 3\)"):
            fn(x, x)


def test_warnings_for_millimetres_and_implausible_lambda() -> None:
    gt = _traj()
    with pytest.warns(UserWarning, match="millimetres"):
        e_par_split(1000.0 * 1.01 * gt, 1000.0 * gt)
    with pytest.warns(UserWarning, match="outside"):
        e_par_split(20.0 * gt, gt)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e_par_split(1.1 * gt, gt)                       # a plausible input stays silent


def test_single_frame_puts_everything_in_the_gauge_part() -> None:
    """The README's T = 1 caveat: with one frame the normal equation forces o = 0."""
    gt = _traj(1)
    pred = gt + np.array([[0.01, -0.02, 0.07]])
    sp = e_par_split(pred, gt)
    assert sp.off_orbit_part == pytest.approx(0.0, abs=1e-15)
    assert sp.e_par == pytest.approx(sp.gauge_part, rel=1e-12)
