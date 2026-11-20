"""R1 known answers on synthetic windows: recovery up to lambda (KA recovery), KA-G equivariance,
M-ANCHOR and M-TRACK. Each check is paired with a mutation of the estimator under which it must FAIL, so a passing
check is not vacuous."""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.bench.metric_frame import along_ray_components, gauge_decomposition
from fastdo.tba.bundle import apply_log_gauge, gauge_k
from fastdo.tba.config import TbaConfig
from fastdo.tba.window import finish, prepare, run_tba, surface_centroid
from tests.tba.synth import as_window, corrupt, make_window

FAST: dict = {}


def _centres(poses: np.ndarray, scale: float, cm: np.ndarray) -> np.ndarray:
    return scale * poses[:, :3, :3] @ cm + poses[:, :3, 3]


@pytest.fixture(scope="module")
def case() -> dict:
    sw = make_window(n_rows=12, theta_deg=40.0, noise_px=0.3, seed=3)
    cm = surface_centroid(sw.mesh_v, sw.mesh_f)
    bp, bs = corrupt(sw.poses, sw.scale, cm, lam=1.08, along_sd_m=0.02, seed=4)
    return {"sw": sw, "cm": cm, "bp": bp, "bs": bs, "cg": _centres(sw.poses, sw.scale, cm)}


def test_ka_recovery_off_orbit_error_removed_up_to_lambda(case: dict) -> None:
    sw, cm, bp, bs, cg = case["sw"], case["cm"], case["bp"], case["bs"], case["cg"]
    base = gauge_decomposition(_centres(bp, bs, cm), cg)
    out = run_tba(as_window(sw, bp, bs), TbaConfig(carve=False, **FAST))
    got = gauge_decomposition(_centres(out.poses, out.scale, cm), cg)
    assert base.off_orbit_rms > 0.010                      # the corruption is real (>= 10 mm off the orbit)
    assert got.off_orbit_rms < 0.003                       # KA: <= 3 mm off-orbit after TBA
    assert got.lam == pytest.approx(base.lam, rel=0.01)    # gauge K: BASE's lambda is held, not restored
    # mutation: no BA row (min_live above the track count) leaves BASE's off-orbit error in place -> the check fires
    dead = run_tba(as_window(sw, bp, bs), TbaConfig(carve=False, min_live=1000, **FAST))
    assert gauge_decomposition(_centres(dead.poses, dead.scale, cm), cg).off_orbit_rms > 0.003


@pytest.mark.parametrize("beta", [0.9, 1.1])
def test_ka_g_output_is_gauge_equivariant(case: dict, beta: float) -> None:
    sw, bp, bs = case["sw"], case["bp"], case["bs"]
    cfg = TbaConfig(carve=False, **FAST)
    ref = run_tba(as_window(sw, bp, bs), cfg)
    sp = bp.copy()
    sp[:, :3, 3] *= beta
    got = run_tba(as_window(sw, sp, bs * beta), cfg)
    assert got.scale == pytest.approx(beta * ref.scale, rel=1e-6)
    assert np.allclose(got.poses[:, :3, :3], ref.poses[:, :3, :3], atol=1e-6)
    assert np.allclose(got.poses[:, :3, 3], beta * ref.poses[:, :3, 3], rtol=1e-6, atol=1e-9)
    # mutation: an ABSOLUTE-length anchor prior (25 mm whatever the gauge) is not equivariant -> the check fires
    import fastdo.tba.bundle as B

    orig = B._Problem.fun

    def absolute(self, x):                                  # prior residual in metres / 25 mm
        r = orig(self, x)
        n_obs, n_anc = 2 * len(self.ok), len(self.anc)
        a = x[6 * len(self.free):]
        r[n_obs:n_obs + n_anc] = np.expm1(a) * self.p.d_base[self.anc] / 0.025
        return r

    B._Problem.fun = absolute
    try:
        m1 = run_tba(as_window(sw, bp, bs), cfg)
        mb = run_tba(as_window(sw, sp, bs * beta), cfg)
    finally:
        B._Problem.fun = orig
    assert not np.allclose(mb.poses[:, :3, 3], beta * m1.poses[:, :3, 3], rtol=1e-6, atol=1e-9)


def _gauged_depths(prep) -> np.ndarray:
    """The anchors TBA outputs: the BA's depths in gauge K (what `finish` re-poses the template on)."""
    return apply_log_gauge(prep.result, gauge_k(prep.problem, prep.result, prep.d_hit)).depths


def test_m_anchor_offset_anchors_are_restored(case: dict) -> None:
    """M-ANCHOR: +50 mm along the keyframe ray on 20% of anchors (prior centre and init); TBA's output anchors, ALL of
    them, are within 5 mm RMS of the unmutated TBA solution's. Compared in gauge K (the output): before it the mutated
    prior also drags the soft BA gauge. All anchors, not only the mutated ones: gauge K's median can carry a wrong
    solution's mutated anchors back onto the reference while moving the other 80% (the tight-prior mutation below
    does exactly that: mutated-only RMS 2.4 mm, all-anchor RMS 45 mm)."""
    sw, bp, bs = case["sw"], case["bp"], case["bs"]
    win = as_window(sw, bp, bs)
    cfg = TbaConfig(carve=False, **FAST)
    ref = prepare(win, cfg)
    rng = np.random.default_rng(7)
    idx = rng.choice(np.flatnonzero(ref.anchored), int(0.2 * ref.anchored.sum()), replace=False)
    off = np.zeros(len(ref.anchored))
    off[idx] = 0.05
    mut = prepare(win, cfg, anchor_offsets=off)
    a = ref.anchored
    rms = float(np.sqrt(np.mean((_gauged_depths(mut)[a] - _gauged_depths(ref)[a]) ** 2)))
    assert rms < 0.005
    # mutation: a prior so tight the tracks cannot move the anchors -> they stay ~50 mm off and the check fires
    tight = TbaConfig(carve=False, prior_sd_m=1e-4, **FAST)
    ref_t, mut_t = prepare(win, tight), prepare(win, tight, anchor_offsets=off)
    assert float(np.sqrt(np.mean((_gauged_depths(mut_t)[a] - _gauged_depths(ref_t)[a]) ** 2))) > 0.02


def _swap(sw, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    j = sw.tracks.shape[1]
    idx = rng.choice(j, int(frac * j), replace=False)
    trk = sw.tracks.copy()
    trk[1:, idx] = sw.tracks[1:, np.roll(idx, 1)]         # identities swapped after the keyframe (~50 px moves)
    return trk, idx


def _e1_mm(out, cm: np.ndarray, cg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(along_ray_components(_centres(out.poses, out.scale, cm), cg).signed_along ** 2))
                 * 1e3)


def _m_track(cfg: TbaConfig, seeds: range) -> tuple[list[float], list[float]]:
    d_e1, below = [], []
    for seed in seeds:
        sw = make_window(n_rows=12, theta_deg=40.0, noise_px=0.3, seed=seed)
        cm = surface_centroid(sw.mesh_v, sw.mesh_f)
        bp, bs = corrupt(sw.poses, sw.scale, cm, lam=1.08, along_sd_m=0.02, seed=seed + 10)
        win, cg = as_window(sw, bp, bs), _centres(sw.poses, sw.scale, cm)
        trk, idx = _swap(sw, 0.1, 11 + seed)
        ref, mut = prepare(win, cfg), prepare(win, cfg, tracks=trk)
        d_e1.append(abs(_e1_mm(finish(win, cfg, ref), cm, cg) - _e1_mm(finish(win, cfg, mut), cm, cg)))
        w = mut.result.weights[1:]
        obs = win.vis[1:] & mut.anchored[None, :]
        per = np.array([w[obs[:, i], i].mean() if obs[:, i].any() else np.nan for i in range(len(mut.anchored))])
        below.append(float(np.mean(per[idx] < np.nanmedian(per))))
    return d_e1, below


def test_m_track_swapped_tracks_are_down_weighted() -> None:
    """M-TRACK: swap 10% of track identities after the keyframe. Over 6 synthetic windows the median |E1 change| is
    <= 2 mm AND the swapped tracks' final weights sit below the median weight on >= 80% of them (the robust loss
    actually fires)."""
    d_e1, below = _m_track(TbaConfig(carve=False), range(6))
    assert float(np.median(d_e1)) <= 2.0
    assert float(np.median(below)) >= 0.8
    # mutation: no robust loss -> the swapped tracks drag E1 (measured 10-23 mm per window) and the check fires
    d_nr, _ = _m_track(TbaConfig(carve=False, robust=False), range(3))
    assert float(np.median(d_nr)) > 2.0


def test_the_analytic_jacobian_matches_central_differences(case: dict) -> None:
    """The solver's Jacobian (tracks, anchor prior, centroid and rotation acceleration rows) against central differences
    of the residual at a random non-zero state; a mutated Jacobian (anchor column sign flipped) is caught."""
    from fastdo.tba.bundle import _ba_rows, _Problem

    sw, bp, bs = case["sw"], case["bp"], case["bs"]
    win = as_window(sw, bp, bs)
    cfg = TbaConfig(carve=False)
    prep = prepare(win, cfg)
    prob = _Problem(prep.problem, cfg, _ba_rows(prep.problem, cfg))
    rng = np.random.default_rng(5)
    prob.w = rng.uniform(0.2, 1.0, len(prob.ok))
    x = prob.x0() + rng.normal(0.0, 0.01, prob.n_x)
    J = prob.jac(x).toarray()
    h = 1e-6
    num = np.empty_like(J)
    for c in range(prob.n_x):
        e = np.zeros(prob.n_x)
        e[c] = h
        num[:, c] = (prob.fun(x + e) - prob.fun(x - e)) / (2 * h)
    scale = np.abs(num).max()
    assert np.abs(J - num).max() <= 1e-5 * scale
    # mutation: the anchor block's sign flipped -> the check fires
    bad = J.copy()
    bad[:, 6 * len(prob.free):] *= -1.0
    assert np.abs(bad - num).max() > 1e-2 * scale
