"""The sigma_V calibration of `fastdo.policy.smc_calibration` pools chunk-horizon residuals only.

Chunk 0 spans the warm-up plus one chunk, and a last chunk can be shorter than the chunk horizon; neither residual is
taken at the horizon the tempering floor applies to, so neither may move sigma_V. The best-of-K pass is replaced by a
fake that returns designed `chunk_residuals`; the chunk bounds come from the real schedule of `fastdo.policy.smc`.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.policy import smc_calibration
from fastdo.policy.reference import PolicyReference
from fastdo.policy.smc import SmcConfig, SmcResult

logger = logging.getLogger(__name__)

K, WARMUP, HZ, CHUNK_SECONDS = 3, 5, 50.0, 0.04          # chunk horizon round(0.04 x 50) = 2 steps


def _reference(n_frames: int) -> PolicyReference:
    return PolicyReference(q_ref=torch.zeros(n_frames, 28), obj_pose=torch.eye(4).repeat(n_frames, 1, 1),
                           log_scale=0.0, contact_plan=torch.zeros(n_frames, 6, dtype=torch.bool), covariance=None,
                           hz=HZ)


def _calibrate(monkeypatch: pytest.MonkeyPatch, residuals: np.ndarray, n_frames: int) -> tuple[float, int]:
    """sigma_V over one clip of `n_frames` at W = WARMUP whose best-of-K pass records `residuals (N, K)`."""

    def fake_best_of_k(env: Any, model: Any, ref: Any, k: int, generator: Any, cfg: SmcConfig | None = None
                       ) -> SmcResult:
        assert cfg is not None and cfg.twist and k == K
        n = residuals.shape[0]
        return SmcResult(qpos=np.zeros((0, 35)), obj_pose=np.zeros((0, 4, 4)), success=True,
                         ess_per_chunk=np.full(n, float(K)), lambda_per_chunk=np.full(n, math.inf), n_resamples=0,
                         chunk_residuals=residuals)

    monkeypatch.setattr(smc_calibration, "run_best_of_k", fake_best_of_k)
    env: Any = SimpleNamespace(n_worlds=K, cfg=SimpleNamespace(warmup_ctrl_steps=WARMUP, randomize=False))
    model: Any = None
    cfg = SmcConfig(n_particles=K, chunk_seconds=CHUNK_SECONDS, twist=False)
    return smc_calibration.calibrate_sigma_v(env, model, [_reference(n_frames)], cfg, torch.Generator())


def test_sigma_v_ignores_chunk_zero_and_a_partial_last_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """8 frames at W = 5 and a 2-step horizon: chunk 0 is steps [0, 7) (warm-up plus one chunk), chunks 1-2 hold 2 steps
    and chunk 3 one step. sigma_V and the residual count come from chunks 1-2 alone (a dead particle's NaN dropped): a
    chunk-0 residual of 1e6 and a last-chunk residual of -1e6 give exactly the values of zeros there, the sample sd of
    the chunk-horizon residuals."""
    body = np.random.default_rng(0).normal(0.0, 0.7, size=(2, K))
    body[1, 2] = math.nan

    def residuals(chunk_0: float, last: float) -> np.ndarray:
        return np.vstack([np.full((1, K), chunk_0), body, np.full((1, K), last)])

    clean = _calibrate(monkeypatch, residuals(0.0, 0.0), n_frames=8)
    spiked = _calibrate(monkeypatch, residuals(1e6, -1e6), n_frames=8)
    horizon = body[np.isfinite(body)]
    assert spiked == clean
    assert clean[1] == horizon.size == 5 and clean[0] == pytest.approx(float(horizon.std(ddof=1)), rel=1e-12)
    logger.info("sigma_V %.4f from %d chunk-horizon residuals; chunk 0 and the partial last chunk excluded", *clean)


def test_sigma_v_pools_the_residuals_of_every_scene_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Held-out clips of several objects run in their own object scenes. `calibrate_sigma_v_scenes` pools the
    chunk-horizon residuals of every (environment, references) group into one sample sd, which is not a mean of the
    per-scene values when the scenes' residuals differ in mean; `calibrate_sigma_v` is its one-group case."""
    rng = np.random.default_rng(1)
    tables = {"a": np.vstack([np.zeros((1, K)), rng.normal(1.0, 0.2, size=(2, K)), np.zeros((1, K))]),
              "b": np.vstack([np.zeros((1, K)), rng.normal(-1.0, 0.2, size=(2, K)), np.zeros((1, K))])}

    def fake_best_of_k(env: Any, model: Any, ref: Any, k: int, generator: Any, cfg: SmcConfig | None = None
                       ) -> SmcResult:
        table = tables[env.name]
        return SmcResult(qpos=np.zeros((0, 35)), obj_pose=np.zeros((0, 4, 4)), success=True,
                         ess_per_chunk=np.full(4, float(K)), lambda_per_chunk=np.full(4, math.inf), n_resamples=0,
                         chunk_residuals=table)

    monkeypatch.setattr(smc_calibration, "run_best_of_k", fake_best_of_k)
    envs: dict[str, Any] = {name: SimpleNamespace(name=name, n_worlds=K, cfg=SimpleNamespace(warmup_ctrl_steps=WARMUP,
                                                                                           randomize=False))
                            for name in tables}
    cfg = SmcConfig(n_particles=K, chunk_seconds=CHUNK_SECONDS, twist=False)
    groups = [(envs["a"], [_reference(8)]), (envs["b"], [_reference(8), _reference(8)])]
    sigma_v, n = smc_calibration.calibrate_sigma_v_scenes(groups, None, cfg, torch.Generator())  # type: ignore[arg-type]
    pooled = np.concatenate([tables["a"][1:3].ravel(), tables["b"][1:3].ravel(), tables["b"][1:3].ravel()])
    assert n == pooled.size == 18 and sigma_v == pytest.approx(float(pooled.std(ddof=1)), rel=1e-12)
    one = [smc_calibration.calibrate_sigma_v(env, None, refs, cfg, torch.Generator())  # type: ignore[arg-type]
           for env, refs in groups]
    assert sigma_v > 2.0 * max(s for s, _ in one)                      # between-scene spread is part of the pool
    logger.info("pooled sigma_V %.4f over %d residuals; per scene %s", sigma_v, n, one)


def test_sigma_v_refuses_a_clip_without_chunk_horizon_residuals(monkeypatch: pytest.MonkeyPatch) -> None:
    """3 frames: the clip is chunk 0 alone, so however large its residuals, no chunk-horizon residual exists and the
    calibration refuses; a residual table whose chunk count disagrees with the schedule refuses too."""
    with pytest.raises(ValueError, match="chunk-horizon"):
        _calibrate(monkeypatch, np.full((1, K), 5.0), n_frames=3)
    with pytest.raises(ValueError, match="chunks"):
        _calibrate(monkeypatch, np.zeros((3, K)), n_frames=8)
