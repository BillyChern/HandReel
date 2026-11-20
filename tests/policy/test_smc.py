"""Twisted SMC with the policy as proposal (design Section 9b).

The particle maths (tempering by ESS, systematic resampling, the twisted potential) is checked against brute force;
the particle system against the independent-chunk formulas on the slip chain, against potentials recomputed from the
twist chain's trace, and on the CPU twin. The world-state copy it relies on is tested in `test_env_state.py`.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.policy.networks import ActorCritic
from fastdo.policy.smc import (
    SmcConfig,
    chunk_log_potentials,
    ess_temperature,
    run_best_of_k,
    run_smc,
    systematic_resample,
)
from tests.policy.toy_env import (
    SlipChainConfig,
    SlipChainEnv,
    TwistChainConfig,
    TwistChainEnv,
    TwistCritic,
)

logger = logging.getLogger(__name__)

N_CLIPS = 2000


def _brute_force_ess(log_potentials: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """ESS (sum w)^2 / sum w^2 of w = exp(v / lambda) for every lambda (G,), dead (-inf) potentials weightless."""
    v = log_potentials[np.isfinite(log_potentials)]
    w = np.exp((v[None, :] - v.max()) / lam[:, None])
    return w.sum(1) ** 2 / (w * w).sum(1)


def test_ess_temperature_hits_target() -> None:
    """K = 64 random log-potentials at three scales, 5 of them dead: the returned lambda achieves ESS 32 within 1% and
    lies between the two neighbours of a 0.12%-step brute-force lambda grid where the grid ESS crosses 32. Equal
    potentials, or fewer live particles than the target, need no tempering: inf."""
    gen = torch.Generator().manual_seed(0)
    grid = np.logspace(-6.0, 6.0, 24001)
    for scale in (1e-3, 1.0, 50.0):
        log_pot = scale * torch.randn(64, generator=gen, dtype=torch.float64)
        log_pot[:5] = -math.inf
        lam = ess_temperature(log_pot, 0.5)
        v = log_pot.numpy()
        achieved = float(_brute_force_ess(v, np.array([lam]))[0])
        assert abs(achieved - 32.0) <= 0.01 * 32.0, (scale, lam, achieved)
        ess_grid = _brute_force_ess(v, scale * grid)
        assert bool(np.all(np.diff(ess_grid) >= -1e-9))            # ESS grows with lambda, so bisection is valid
        i = int(np.searchsorted(ess_grid, 32.0))
        assert 0 < i < grid.size and scale * grid[i - 1] <= lam <= scale * grid[i], (scale, lam)
    assert ess_temperature(torch.full((64,), 0.7), 0.5) == math.inf
    jitter = 1e-8 * torch.randn(64, generator=gen, dtype=torch.float64)      # float32-level noise is no preference
    assert ess_temperature(0.7 + jitter, 0.5) == math.inf
    few = torch.full((64,), -math.inf, dtype=torch.float64)
    few[:20] = torch.randn(20, generator=gen, dtype=torch.float64)
    assert ess_temperature(few, 0.5) == math.inf
    with pytest.raises(ValueError):
        ess_temperature(torch.full((8,), -math.inf), 0.5)
    with pytest.raises(ValueError):
        ess_temperature(torch.tensor([0.0, math.nan]), 0.5)


def test_systematic_resample_unbiased() -> None:
    """20k resamples of K = 8 fixed weights (expected offspring 0, 1, 1.25, 1.5, 1.75, 1.125, 1.375, 0): the mean
    offspring count of particle i is within 1% of K w_i, every count is floor or ceil of K w_i (the low variance of
    systematic resampling), and zero weights never reproduce. Uniform weights return every particle once, in place."""
    gen = torch.Generator().manual_seed(0)
    expected = torch.tensor([0.0, 1.0, 1.25, 1.5, 1.75, 1.125, 1.375, 0.0], dtype=torch.float64)
    weights = expected / 8.0
    counts = torch.stack([torch.bincount(systematic_resample(weights, gen), minlength=8)
                          for _ in range(20_000)]).double()
    mean = counts.mean(0)
    assert bool(((mean - expected).abs() <= 0.01 * expected).all()), mean.tolist()
    assert bool((counts >= expected.floor()).all()) and bool((counts <= expected.ceil()).all())
    random_w = torch.distributions.Dirichlet(torch.ones(64, dtype=torch.float64)).sample()
    ancestors = systematic_resample(random_w, gen)
    assert ancestors.dtype == torch.long and ancestors.shape == (64,)
    offspring = torch.bincount(ancestors, minlength=64).double()
    assert bool((offspring >= (64 * random_w).floor() - 1e-9).all() and (offspring <= (64 * random_w).ceil()).all())
    assert torch.equal(systematic_resample(torch.full((64,), 1.0 / 64), gen), torch.arange(64))


def test_chunk_potential_is_the_twisted_increment() -> None:
    """log w = R_chunk + gamma^H V(s_end) - V(s_start) with the twist, R_chunk without; dead particles and non-finite
    critic values get -inf, so they are never resampled."""
    reward = torch.tensor([1.0, 0.5, 2.0, 0.0, 1.0], dtype=torch.float64)
    v_start = torch.tensor([0.2, 0.1, 0.3, 0.4, math.nan], dtype=torch.float64)
    v_end = torch.tensor([0.5, 0.7, -0.1, 0.0, 0.0], dtype=torch.float64)
    alive = torch.tensor([True, True, True, False, True])
    got = chunk_log_potentials(reward, v_start, v_end, n_steps=25, gamma=0.99, alive=alive, twist=True)
    twisted = reward + 0.99**25 * v_end - v_start
    torch.testing.assert_close(got[:3], twisted[:3])
    assert got[3] == -math.inf and got[4] == -math.inf
    plain = chunk_log_potentials(reward, v_start, v_end, n_steps=25, gamma=0.99, alive=alive, twist=False)
    torch.testing.assert_close(plain[[0, 1, 2, 4]], reward[[0, 1, 2, 4]])
    assert plain[3] == -math.inf


def _toy_model(env: Any) -> ActorCritic:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=(16,))


def test_smc_beats_best_of_k_on_toy_chain() -> None:
    """Every chunk slips with probability 0.2 whatever the action, N = 20 chunks, K = 16 particles, 2000 clips. With
    independent chunks, SMC fails only if all K particles slip in one chunk, success (1 - p^K)^N = 1.000, while
    best-of-K needs one particle to survive all N chunks, 1 - (1 - (1 - p)^N)^K = 0.17."""
    cfg = SlipChainConfig()
    env = SlipChainEnv(cfg)
    model = _toy_model(env)
    k, n, p = cfg.n_worlds, cfg.n_chunks, cfg.p_slip
    smc_formula, bok_formula = (1.0 - p**k) ** n, 1.0 - (1.0 - (1.0 - p) ** n) ** k
    assert f"{smc_formula:.3f}" == "1.000" and f"{bok_formula:.2f}" == "0.17"
    gen = torch.Generator().manual_seed(0)
    smc = [run_smc(env, model, env.reference, SmcConfig(n_particles=k), gen) for _ in range(N_CLIPS)]
    bok = [run_best_of_k(env, model, env.reference, k, gen) for _ in range(N_CLIPS)]
    smc_rate = float(np.mean([r.success for r in smc]))
    bok_rate = float(np.mean([r.success for r in bok]))
    logger.info("slip chain: SMC %.4f (formula %.6f), best-of-K %.4f (formula %.4f), mean resamples %.2f", smc_rate,
                smc_formula, bok_rate, bok_formula, np.mean([r.n_resamples for r in smc]))
    assert smc_rate >= 0.99 and bok_rate <= 0.40, (smc_rate, bok_rate)
    assert smc_rate >= 0.999 and abs(bok_rate - bok_formula) < 4.0 * math.sqrt(bok_formula * (1 - bok_formula) / N_CLIPS)
    assert all(r.n_resamples == 0 for r in bok) and np.mean([r.n_resamples for r in smc]) > 1.0


def test_smc_output_is_one_ancestral_path() -> None:
    """The returned trajectory is the ancestral path of one final particle: one lineage tag throughout, every frame
    once and in order, and the object on its reference (never a dropped pose), although particles are copied between
    worlds after most chunks; its reset state `start_qpos` (frame 0) carries the same lineage tag. Chunks of 2 steps:
    the slip happens on the first, the copy after the second."""
    env = SlipChainEnv(SlipChainConfig(n_worlds=8, chunk_steps=2, hz=4.0, p_slip=0.3))
    model = _toy_model(env)
    gen = torch.Generator().manual_seed(1)
    steps = env.n_frames - 1
    ref_pose = env.reference.obj_pose[1:].numpy()
    successes, copies = 0, 0
    for _ in range(40):
        res = run_smc(env, model, env.reference, SmcConfig(n_particles=8), gen)
        assert res.qpos.shape == (steps, 2 + 7) and res.obj_pose.shape == (steps, 4, 4)
        assert res.ess_per_chunk.shape == (20,) and res.lambda_per_chunk.shape == (20,)
        if not res.success:
            continue
        successes, copies = successes + 1, copies + res.n_resamples
        assert bool(np.all(res.qpos[:, 0] == res.qpos[0, 0]))
        np.testing.assert_array_equal(res.qpos[:, 1], np.arange(1, steps + 1))
        np.testing.assert_array_equal(res.obj_pose, ref_pose)
        np.testing.assert_allclose(res.qpos[:, 2:5], ref_pose[:, :3, 3])
        assert res.start_qpos.shape == (2 + 7,) and res.start_qpos[0] == res.qpos[0, 0] and res.start_qpos[1] == 0.0
        np.testing.assert_allclose(res.start_qpos[2:5], env.reference.obj_pose[0, :3, 3].numpy())
        assert bool(np.all(res.ess_per_chunk <= 8.0)) and bool(np.all(np.isinf(res.lambda_per_chunk)))
    assert successes >= 38 and copies > 0


def test_run_smc_on_cpu_env_end_to_end() -> None:
    """3 particles in the CPU twin, a 6-frame clip (W = 5, chunks of 2 steps: N = 3), reference noise from the clip's
    covariance and the critic twist: every tempered chunk has ESS K / 2, and one executed trajectory of W + T - 1
    steps whose qpos holds the object pose it reports; `success` is `clip_success` on it with the documented alignment
    (step W - 1 + j is frame j). On a 60-frame clip the unheld object drops in every particle: a failure, with the
    remaining chunks NaN."""
    from fastdo.policy.env import make_env
    from fastdo.policy.rewards import clip_success
    from fastdo.policy.scene import object_pose
    from tests.policy.env_fixtures import SCENE_XML, TRAJECTORY_NPZ, env_config, whisking_reference

    if not TRAJECTORY_NPZ.exists():
        pytest.skip("baseline submodule whisking scene not checked out")
    env = make_env(env_config(n_worlds=3, warmup_duration=0.1), SCENE_XML, "cpu")
    model = _toy_model(env)
    gen = torch.Generator().manual_seed(0)
    ref, W, T = whisking_reference(n_frames=6), 5, 6
    res = run_smc(env, model, ref, SmcConfig(n_particles=3, chunk_seconds=0.04), gen)
    assert res.qpos.shape == (W + T - 1, 35) and res.obj_pose.shape == (W + T - 1, 4, 4)
    assert res.ess_per_chunk.shape == (3,) and res.lambda_per_chunk.shape == (3,)
    finite = np.isfinite(res.lambda_per_chunk)                     # tempered chunks have ESS K / 2
    assert bool(finite.any()) and np.allclose(res.ess_per_chunk[finite], 1.5, rtol=0.01), res.ess_per_chunk
    np.testing.assert_allclose(object_pose(torch.as_tensor(res.qpos[:, 28:35])).numpy(), res.obj_pose, atol=1e-5)
    frames = np.clip(np.arange(W + T - 1) - W + 1, 0, T - 1)
    assert res.success and clip_success(res.obj_pose, ref.obj_pose.numpy()[frames], W - 1)[0]
    dropped = run_best_of_k(env, model, whisking_reference(n_frames=60), 3, gen)
    assert not dropped.success and bool(np.isnan(dropped.ess_per_chunk[-1])) and bool(np.isnan(dropped.qpos[-1]).all())
    with pytest.raises(ValueError, match="n_particles"):
        run_smc(env, model, ref, SmcConfig(n_particles=4), gen)
    randomized = make_env(env_config(n_worlds=3, warmup_duration=0.1, randomize=True), SCENE_XML, "cpu")
    with pytest.raises(ValueError, match="randomize"):
        run_smc(randomized, model, ref, SmcConfig(n_particles=3), gen)


def _chunk_log_potential(trace: list[dict[str, Tensor]], s0: int, s1: int, critic: TwistCritic, gamma: float,
                         twist: bool, last: bool) -> np.ndarray:
    """log G of steps [s0, s1) from the twist chain's trace, independently of `chunk_log_potentials`:
    sum_h gamma^h r_h, plus gamma^H V(s_end) - V(s_start) with the twist (V(s_end) = 0 after the last chunk)."""
    def value(y: Tensor, doomed: Tensor) -> Tensor:
        return critic.value_scale * y - critic.doom_penalty * doomed.to(torch.float64)

    log_g = sum(gamma**h * trace[s]["reward"] for h, s in enumerate(range(s0, s1)))
    assert isinstance(log_g, Tensor)
    if twist:
        v_end = 0.0 if last else gamma ** (s1 - s0) * value(trace[s1 - 1]["y_next"], trace[s1 - 1]["doomed_next"])
        log_g = log_g + v_end - value(trace[s0]["y"], trace[s0]["doomed"])
    return log_g.numpy()


def test_run_smc_weights_follow_twisted_tempered_potentials() -> None:
    """Particles whose potentials differ: the twist chain with a drifting feature y, reward y and critic V = 2 y
    (K = 16, N = 8 chunks of 2 steps, 120 clips with and 120 without the twist). With log G recomputed per chunk from
    the environment's trace, every chunk's lambda is finite and exp(log G / lambda) has ESS K/2 within 1%, as
    `ess_per_chunk` reports. At every resampling each particle's offspring count (the worlds holding its end-of-chunk y
    after the copy) is the floor or ceil of K w_i, and the counts are unbiased over all resamplings; a particle with
    offspring keeps its world, and each copy writes exactly the worlds of the particles without offspring."""
    k, n_chunks, steps, gamma, n_clips = 16, 8, 2, 0.99, 120
    env = TwistChainEnv(TwistChainConfig(n_worlds=k, n_chunks=n_chunks, chunk_steps=steps, drift_std=0.5))
    critic = TwistCritic(value_scale=2.0, doom_penalty=0.0)
    gen = torch.Generator().manual_seed(0)
    expected_all, counts_all = [], []
    for twist in (True, False):
        cfg = SmcConfig(n_particles=k, gamma=gamma, twist=twist)
        for _ in range(n_clips):
            res = run_smc(env, critic, env.reference, cfg, gen)
            copies = []
            for j in range(n_chunks):
                s0, s1, last = j * steps, (j + 1) * steps, j == n_chunks - 1
                log_g = _chunk_log_potential(env.trace, s0, s1, critic, gamma, twist, last)
                lam = float(res.lambda_per_chunk[j])
                w = np.exp((log_g - log_g.max()) / lam)
                ess = w.sum() ** 2 / (w * w).sum()
                assert math.isfinite(lam) and abs(ess - k / 2) <= 0.01 * k / 2, (twist, j, lam, ess)
                assert abs(res.ess_per_chunk[j] - k / 2) <= 0.01 * k / 2, (twist, j, res.ess_per_chunk[j])
                if last:
                    break
                expected = k * w / w.sum()
                # y after the chunk is continuous, so unique per particle (tags repeat once copies share a lineage)
                before, after = env.trace[s1 - 1]["y_next"].numpy(), env.trace[s1]["y"].numpy()
                assert np.unique(before).size == k
                counts = (after[None, :] == before[:, None]).sum(1)
                assert np.all(counts >= np.floor(expected - 1e-6)) and np.all(counts <= np.ceil(expected + 1e-6)), (
                    twist, j, expected.round(3).tolist(), counts.tolist())
                np.testing.assert_array_equal(after[counts > 0], before[counts > 0])
                copies += [int((counts == 0).sum())] if bool((counts == 0).any()) else []
                expected_all.append(expected)
                counts_all.append(counts)
            assert env.copies == copies and res.n_resamples == len(copies)
    expected, counts = np.concatenate(expected_all), np.concatenate(counts_all)
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, math.inf)):
        sel = (expected >= lo) & (expected < hi)
        gap = abs(float(counts[sel].mean() - expected[sel].mean()))
        assert sel.sum() >= 100 and gap <= 4.0 * 0.5 / math.sqrt(sel.sum()), (lo, int(sel.sum()), gap)


def test_twist_removes_doomed_lineages_before_they_fail() -> None:
    """The twist chain with doom (K = 16, N = 8, 60 clips): on the first step of a chunk a healthy episode becomes
    doomed with probability 1/4, and a doomed one fails on the first step of the next chunk. Rewards are equal and
    the critic sees the doom (V = -5 doomed). With the twist a newly doomed particle's potential is gamma^2 5 below a
    healthy one's, so resampling removes it before it fails; without the twist every live particle weighs the same,
    and about p K doomed particles fail in each of the N - 1 later chunks."""
    k, n_chunks, p, n_clips = 16, 8, 0.25, 60
    env = TwistChainEnv(TwistChainConfig(n_worlds=k, n_chunks=n_chunks, p_doom=p))
    critic = TwistCritic(value_scale=0.0, doom_penalty=5.0)
    gen = torch.Generator().manual_seed(0)
    failures = {}
    for twist in (True, False):
        start = env.n_failures
        for _ in range(n_clips):
            run_smc(env, critic, env.reference, SmcConfig(n_particles=k, twist=twist), gen)
        failures[twist] = (env.n_failures - start) / n_clips
    logger.info("doom chain: failures per clip %.3f with the twist, %.3f without (p K (N - 1) = %.1f)", failures[True],
                failures[False], p * k * (n_chunks - 1))
    assert failures[False] >= 0.8 * p * k * (n_chunks - 1) and failures[True] <= 0.02 * failures[False], failures
