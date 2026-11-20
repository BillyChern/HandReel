"""Defaults B7, B8, B10 of the twisted SMC of `fastdo.policy.smc`, and the sigma_V calibration.

B7: chunk 0 is weighted with R_0 + gamma^H V(s_end), without -V(s_0), unless
`subtract_start_potential`. B8: lambda_j = max(lambda_ESS(K/2), lambda_floor_c sigma_V), sigma_V the sd of
the chunk residual over held-out rollouts, written next to the checkpoint. B10: a warning when reference noise is on
but the reference has no covariance. Potentials are recomputed from the twist chain's trace, independently of the code.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.policy.networks import ActorCritic
from fastdo.policy.smc import SmcConfig, ess_temperature, run_best_of_k, run_smc
from fastdo.policy.smc_calibration import (
    calibrate_sigma_v,
    read_sigma_v,
    sigma_v_path,
    with_calibrated_floor,
    write_sigma_v,
)
from tests.policy.toy_env import (
    SlipChainConfig,
    SlipChainEnv,
    TwistChainConfig,
    TwistChainEnv,
    TwistCritic,
)

logger = logging.getLogger(__name__)

K, N_CHUNKS, STEPS, GAMMA, SCALE = 16, 4, 2, 0.99, 2.0


def _ess(log_g: np.ndarray, lam: float) -> float:
    v = log_g[np.isfinite(log_g)]
    w = np.ones_like(v) if math.isinf(lam) else np.exp((v - v.max()) / lam)
    return float(w.sum() ** 2 / (w * w).sum())


def _log_potential(trace: list[dict[str, Tensor]], j: int, start_potential: bool) -> np.ndarray:
    """log G of chunk j per world from the trace: sum_h gamma^h r_h + gamma^H V(y_end) (0 after the last chunk), minus
    V(y_start) when `start_potential`; V = SCALE y."""
    s0, s1 = j * STEPS, (j + 1) * STEPS
    log_g = torch.stack([GAMMA**h * trace[s]["reward"] for h, s in enumerate(range(s0, s1))]).sum(0).numpy().copy()
    if j < N_CHUNKS - 1:
        log_g += GAMMA**STEPS * SCALE * trace[s1 - 1]["y_next"].numpy()
    if start_potential:
        log_g -= SCALE * trace[s0]["y"].numpy()
    return log_g


def _twist_env(init_std: float) -> TwistChainEnv:
    return TwistChainEnv(TwistChainConfig(n_worlds=K, n_chunks=N_CHUNKS, chunk_steps=STEPS, drift_std=0.5,
                                          init_std=init_std))


def test_smc_config_defaults_pin_b7_b8_b10() -> None:
    """B7-B8: `SmcConfig` defaults to no chunk-0 start potential (`subtract_start_potential`
    False) and a tempering floor at c = 1 (`lambda_floor_c` 1.0), both as field defaults and on a default-built
    configuration; the ablations switch them explicitly."""
    defaults = {f.name: f.default for f in dataclasses.fields(SmcConfig)}
    assert defaults["subtract_start_potential"] is False and defaults["lambda_floor_c"] == 1.0
    cfg = SmcConfig()
    assert cfg.subtract_start_potential is False and cfg.lambda_floor_c == 1.0


def test_chunk_zero_weights_omit_the_start_potential_by_default() -> None:
    """Particles start with different values (y_0 ~ N(0, 1), V = 2 y): by default chunk 0's reported lambda gives the
    recomputed R_0 + gamma^H V(s_end) weights ESS K/2 (1 %), while the difference form at that lambda does not; with
    `subtract_start_potential` it is the other way round. Before the first resampling world i is particle i, so the
    trace rows are the particles' own."""
    env, critic = _twist_env(init_std=1.0), TwistCritic(value_scale=SCALE, doom_penalty=0.0)
    gen = torch.Generator().manual_seed(0)
    separated = 0
    for subtract in (False, True):
        cfg = SmcConfig(n_particles=K, gamma=GAMMA, subtract_start_potential=subtract, lambda_floor_c=0.0)
        for _ in range(40):
            res = run_smc(env, critic, env.reference, cfg, gen)
            lam = float(res.lambda_per_chunk[0])
            assert math.isfinite(lam)
            chosen = _log_potential(env.trace, 0, start_potential=subtract)
            other = _log_potential(env.trace, 0, start_potential=not subtract)
            assert _ess(chosen, lam) == pytest.approx(K / 2, rel=0.01)
            assert res.ess_per_chunk[0] == pytest.approx(K / 2, rel=0.01)
            separated += abs(_ess(other, lam) - K / 2) > 0.05 * K
    assert separated >= 70, separated                  # the two formulas are distinguishable on almost every clip


def test_later_chunks_keep_the_difference_form_and_best_of_k_counts_successes() -> None:
    """Without resampling (best-of-K) world i is particle i: every chunk j >= 1 has ESS K/2 at its lambda for the
    difference form R + gamma^H V(s_end) - V(s_start), and `chunk_residuals` equals that form on every chunk (chunk 0
    included). On the slip chain, `n_success` is the number of particles that never failed."""
    env, critic = _twist_env(init_std=1.0), TwistCritic(value_scale=SCALE, doom_penalty=0.0)
    gen = torch.Generator().manual_seed(1)
    for _ in range(10):
        res = run_best_of_k(env, critic, env.reference, K, gen, cfg=SmcConfig(gamma=GAMMA, lambda_floor_c=0.0))
        for j in range(N_CHUNKS):
            diff = _log_potential(env.trace, j, start_potential=True)
            np.testing.assert_allclose(res.chunk_residuals[j], diff, rtol=0.0, atol=1e-9)
            if j >= 1:
                assert _ess(diff, float(res.lambda_per_chunk[j])) == pytest.approx(K / 2, rel=0.01)

    class Recording(SlipChainEnv):
        failed_worlds: Tensor

        def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
            out = super().step(action)
            self.failed_worlds |= out[3]["terminated"]
            return out

    slip = Recording(SlipChainConfig(n_worlds=K, n_chunks=5, p_slip=0.2))
    model = ActorCritic(slip.obs_dim, slip.priv_dim, slip.act_dim, hidden=(4,))
    counts = []
    for _ in range(20):
        slip.failed_worlds = torch.zeros(K, dtype=torch.bool)
        res = run_best_of_k(slip, model, slip.reference, K, gen, cfg=SmcConfig(twist=False, reference_noise=False))
        assert res.n_success == K - int(slip.failed_worlds.sum()) and res.success == (res.n_success > 0)
        counts.append(res.n_success)
    assert 0 < min(counts) < max(counts) < K                      # informative: partial survival varies


def test_tempering_floor_is_the_max_of_ess_temperature_and_c_sigma_v() -> None:
    """lambda_j = max(lambda_ESS, c sigma_V) on every chunk, with lambda_ESS recomputed from the trace: a large sigma_V
    binds (ESS above K/2), c = 0 or a negligible sigma_V leaves the ESS rule."""
    env, critic = _twist_env(init_std=1.0), TwistCritic(value_scale=SCALE, doom_penalty=0.0)
    gen = torch.Generator().manual_seed(2)
    bound = 0
    for c, sigma_v in ((1.0, 50.0), (0.0, 50.0), (1.0, 1e-9)):
        cfg = SmcConfig(n_particles=K, gamma=GAMMA, lambda_floor_c=c, sigma_v=sigma_v, subtract_start_potential=True)
        for _ in range(10):
            res = run_best_of_k(env, critic, env.reference, K, gen, cfg=cfg)
            for j in range(N_CHUNKS):
                log_g = _log_potential(env.trace, j, start_potential=True)
                lam_ess = ess_temperature(torch.as_tensor(log_g), 0.5)
                assert res.lambda_per_chunk[j] == pytest.approx(max(lam_ess, c * sigma_v), rel=1e-12)
                if c * sigma_v > lam_ess:
                    bound += 1
                    assert res.ess_per_chunk[j] > K / 2
    assert bound >= 30
    with pytest.raises(ValueError):
        SmcConfig(sigma_v=-1.0)


def test_calibrate_sigma_v_and_write_it_next_to_the_checkpoint(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """sigma_V is the sample sd of the difference-form chunk residuals of chunks 1..N-1 of one best-of-K pass per
    held-out clip (recomputed from the trace; chunk 0 is excluded); written as `<stem>.sigma_v.json`, read
    back, and applied as `SmcConfig.sigma_v`; a checkpoint without a calibration file leaves the floor off with a
    warning."""
    env, critic = _twist_env(init_std=1.0), TwistCritic(value_scale=SCALE, doom_penalty=0.0)
    cfg = SmcConfig(n_particles=K, gamma=GAMMA, twist=False)                   # calibration turns the twist on
    sigma_v, n = calibrate_sigma_v(env, critic, [env.reference], cfg, torch.Generator().manual_seed(3))
    residuals = np.concatenate([_log_potential(env.trace, j, start_potential=True) for j in range(1, N_CHUNKS)])
    assert n == K * (N_CHUNKS - 1) and sigma_v == pytest.approx(float(residuals.std(ddof=1)), rel=1e-9)
    checkpoint = tmp_path / "update_000050.pt"
    path = write_sigma_v(checkpoint, sigma_v, n, cfg)
    assert path == sigma_v_path(checkpoint) == tmp_path / "update_000050.sigma_v.json"
    assert read_sigma_v(checkpoint) == sigma_v and with_calibrated_floor(cfg, checkpoint).sigma_v == sigma_v
    with caplog.at_level(logging.WARNING, logger="fastdo.policy.smc_calibration"):
        assert with_calibrated_floor(cfg, tmp_path / "other.pt").sigma_v == 0.0
    assert "without a tempering floor" in caplog.text
    logger.info("sigma_V %.4f from %d residuals", sigma_v, n)


def test_reference_noise_without_covariance_warns_once_per_environment(caplog: pytest.LogCaptureFixture) -> None:
    """B10: `reference_noise` on a reference without covariance logs one warning per environment."""
    model = ActorCritic(SlipChainEnv.obs_dim, SlipChainEnv.priv_dim, SlipChainEnv.act_dim, hidden=(4,))
    cfg = SmcConfig(n_particles=4, twist=False, reference_noise=True)
    gen = torch.Generator().manual_seed(4)
    envs = [SlipChainEnv(SlipChainConfig(n_worlds=4, n_chunks=2)) for _ in range(2)]
    with caplog.at_level(logging.WARNING, logger="fastdo.policy.smc"):
        for env in (envs[0], envs[0], envs[1]):
            run_smc(env, model, env.reference, cfg, gen)
    warnings = [r for r in caplog.records if "no covariance" in r.getMessage()]
    assert len(warnings) == 2
