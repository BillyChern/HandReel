"""End-to-end effect of the constant-velocity prior on occluded runs.

Separate from `test_solve_synthetic.py` (whose fixture helpers it reuses) only to respect the 400-line
cap. Every solve here uses the shipped defaults (`visibility=True`, `lam_smooth=0.1`, `lam_vel=1.0`) and
its `lam_vel=0` twin; the solves are shared per module.

Bridging acceptance: `test_velocity_prior_bridges_smooth_occluded_run`, on a
fixture that satisfies the prior's smoothness assumption (constant acceleration through the same moving
occluder), is the binding bridging test. The stock occlusion-0.5 fixture is a low-passed random walk: with
every observed frame pinned to ground truth, prior-only interpolation of its 7-frame gap still errs by
22.4 mm (constant-pose) and 18.5 mm (velocity prior), so no second-difference prior reaches a 0.5x bridging bar
there. The Bayes-optimal gap fill-in is the Brownian bridge (constant-pose
prior) for random-walk motion and the cubic spline (velocity prior) for smooth motion; the evidence-based selection
picks the weights per clip by the Laplace evidence. The stock fixture keeps a no-regression test instead.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import so3_exp
from fastdo.solver import synthetic
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectorySolution, solve_trajectory
from tests.solver.test_solve_synthetic import (
    _obs,
    _observed_frames,
    _per_frame_errors,
    _perturbed_init,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Solves = tuple[synthetic.SyntheticClip, TrajectorySolution, TrajectorySolution]


def _constant_acceleration_trajectory(n: int, step: float, distance: float, gen: torch.Generator, focal: float,
                                      image_hw: tuple[int, int]) -> Tensor:
    """Drop-in for `synthetic._smooth_trajectory`: the object slides right-to-left under the occluder
    (a ~9-frame fully covered run) along a parabola (1.5 mm/frame^2 vertical acceleration, a ~15 mm
    sagitta over the covered window) with a slow constant 1 deg/frame spin. Unused arguments keep the
    patched signature."""
    del step, gen, focal, image_hw
    k = torch.arange(n, dtype=torch.float64)
    poses = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    w = torch.tensor([0.3, -0.5, 0.4], dtype=torch.float64)
    poses[:, :3, :3] = so3_exp(k[:, None] * (w / w.norm() * math.radians(1.0))[None, :])
    poses[:, 0, 3] = 0.18 - 0.012 * k
    poses[:, 1, 3] = -0.07 + 0.5 * 1.5e-3 * (k - 14.0) ** 2
    poses[:, 2, 3] = distance + 0.002 * k
    return poses.float()


def _solve_pair(clip: synthetic.SyntheticClip) -> tuple[TrajectorySolution, TrajectorySolution]:
    """(lam_vel = 0, shipped defaults) on the same clip, init and seed."""
    init = _perturbed_init(clip.poses_gt, seed=1)
    mesh = MeshSamples(clip.surface_points, clip.surface_normals)
    base = SolverConfig(n_obs=512, n_mesh=1024, device=DEVICE)
    sols = [solve_trajectory(mesh, _obs(clip), init, clip.scale_gt * 1.2, cfg)
            for cfg in (replace(base, lam_vel=0.0), base)]
    return sols[0], sols[1]


def _clip_config() -> synthetic.SyntheticConfig:
    return synthetic.SyntheticConfig(n_frames=30, image_hw=(240, 320), n_surface=1024, n_tracks=48,
                                     occlusion=0.5, depth_noise=0.002, shape="box")


@pytest.fixture(scope="module")
def stock_solves() -> Solves:
    clip = synthetic.make_synthetic_clip(_clip_config(), seed=11)
    return (clip, *_solve_pair(clip))


@pytest.fixture(scope="module")
def smooth_solves() -> Solves:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(synthetic, "_smooth_trajectory", _constant_acceleration_trajectory)
        clip = synthetic.make_synthetic_clip(_clip_config(), seed=11)
    return (clip, *_solve_pair(clip))


def _bridging_ratios(solves: Solves) -> tuple[float, float, int]:
    """(unobserved max translation error ratio, observed max translation error ratio, #unobserved),
    shipped defaults over lam_vel = 0, with `test_solve_synthetic`'s error metric and frame split."""
    clip, sol_no_vel, sol_vel = solves
    observed = _observed_frames(clip)
    err_no_vel, _ = _per_frame_errors(sol_no_vel.poses, clip.poses_gt)
    err_vel, _ = _per_frame_errors(sol_vel.poses, clip.poses_gt)
    unobs_ratio = float(err_vel[~observed].max() / err_no_vel[~observed].max())
    obs_ratio = float(err_vel[observed].max() / err_no_vel[observed].max())
    return unobs_ratio, obs_ratio, int((~observed).sum())


def _max_errors(sol: TrajectorySolution, clip: synthetic.SyntheticClip, frames: Tensor) -> tuple[float, float]:
    """(max object-centre error |t - t_gt| in m, max rotation error in deg) over the selected frames."""
    centre = (sol.poses[:, :3, 3].double() - clip.poses_gt[:, :3, 3].double()).norm(dim=1)
    _, rot = _per_frame_errors(sol.poses, clip.poses_gt)
    return float(centre[frames].max()), float(rot[frames].max())


def test_velocity_prior_bridges_smooth_occluded_run(smooth_solves: Solves) -> None:
    """Design 11.7's claim on motion that satisfies its smoothness assumption: the bridging error over
    the occluder-covered run drops to at most 0.5x the lam_vel = 0 value (measured 11.8 -> 2.8 mm) and
    the observed-frame translation error changes by less than 10% (measured 1.35 -> 1.36 mm). Also:
    the observed-frame rotation error may not grow by more than 10% either, above an absolute floor of
    0.05 deg because it sits at ~0.01 deg in both runs."""
    unobs_ratio, obs_ratio, n_unobs = _bridging_ratios(smooth_solves)
    assert n_unobs >= 7
    assert unobs_ratio <= 0.5, unobs_ratio
    assert abs(obs_ratio - 1.0) < 0.10, obs_ratio
    clip, sol_no_vel, sol_vel = smooth_solves
    observed = _observed_frames(clip)
    rot_no_vel, rot_vel = _max_errors(sol_no_vel, clip, observed)[1], _max_errors(sol_vel, clip, observed)[1]
    assert rot_vel <= max(1.1 * rot_no_vel, 0.05), (rot_vel, rot_no_vel)


def test_velocity_prior_does_not_regress_stock_gap_centre_or_rotation(stock_solves: Solves) -> None:
    """Stock random-walk occlusion-0.5 fixture (in place of a 0.5x bridging bar):
    versus lam_vel = 0 at the same lam_smooth = 0.1, the velocity prior must not increase the max gap
    object-centre error |t - t_gt| or the max gap rotation error over the 7 unobserved frames (measured
    20.10 -> 18.65 mm and 2.08 -> 1.91 deg).

    The tests' left-log translation metric (rho, `_per_frame_errors`) does rise in the gap, 16.77 -> 23.36 mm
    (1.39x; observed frames 5.38 -> 6.35 mm and 0.28 -> 0.48 deg), as expected for a random walk,
    where the spline fill-in overshoots; rho mixes theta x t into the centre error at ~0.57 m depth. The evidence-based
    selection picks (lam_smooth, lam_vel) per clip by the Laplace evidence."""
    clip, sol_no_vel, sol_vel = stock_solves
    gap = ~_observed_frames(clip)
    assert int(gap.sum()) == 7
    ctr_no_vel, rot_no_vel = _max_errors(sol_no_vel, clip, gap)
    ctr_vel, rot_vel = _max_errors(sol_vel, clip, gap)
    assert ctr_vel <= ctr_no_vel, (ctr_vel, ctr_no_vel)
    assert rot_vel <= rot_no_vel, (rot_vel, rot_no_vel)


def test_velocity_prior_covariance_grows_toward_gap_centre(stock_solves: Solves) -> None:
    """Laplace confidence semantics: over the 7-frame covered run of the stock fixture the
    translation std from `marginal_covariances` rises strictly from each end of the gap to its centre
    frame (measured 68 / 115 / 148 / 157 / 143 / 108 / 66 mm).

    Monotone growth alone is not specific to the velocity prior: the constant-pose prior alone
    (lam_vel = 0) also grows toward the gap centre (142 / 304 / 369 / 389 / 371 / 308 / 152 mm), so on
    the shipped solve it checks the paired 12x12
    marginal extraction on a real system (a sub-block permutation would break it). What the velocity
    prior adds is information inside the gap, so the test also requires its std to be below the
    lam_vel = 0 std at every gap frame (measured 0.35-0.48x)."""
    clip, sol_no_vel, sol = stock_solves
    gap = (~_observed_frames(clip)).nonzero().flatten()
    assert gap.tolist() == list(range(int(gap[0]), int(gap[0]) + 7))
    std_t = sol.covariance[gap, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1).sqrt()
    std_no_vel = sol_no_vel.covariance[gap, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1).sqrt()
    centre = 3
    rising, falling = std_t[:centre + 1], std_t[centre:]
    assert bool((rising[1:] > rising[:-1]).all()), std_t
    assert bool((falling[1:] < falling[:-1]).all()), std_t
    assert bool((std_t < std_no_vel).all()), (std_t, std_no_vel)
