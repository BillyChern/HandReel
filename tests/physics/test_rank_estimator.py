"""D1 estimator: antithetic noise, Blom rank weights (same expected step, lower variance), registered update."""

from __future__ import annotations

import math

import pytest
import torch

from fastdo.physics.rank_estimator import (
    antithetic_knot_noise,
    blom_scores,
    knot_interp,
    noisy_rows,
    rank_update,
    rank_weights,
    registered_expected_step,
    registered_update,
    registered_weights,
)


def test_registered_expected_step_value() -> None:
    assert registered_expected_step(0.1) == pytest.approx(1.7550, abs=1e-3)


def test_blom_scores_centred_ascending() -> None:
    b = blom_scores(10)
    assert b.shape == (10,)
    assert float(b.sum()) == pytest.approx(0.0, abs=1e-12)
    assert bool((b[1:] > b[:-1]).all())
    with pytest.raises(ValueError):
        blom_scores(1)


def test_rank_weights_structure() -> None:
    rews = torch.tensor([0.0, 3.0, 1.0, 2.0, 0.0, -1.0, 5.0, float("nan")])
    noisy = torch.tensor([False, True, True, True, False, True, True, True])
    w = rank_weights(rews, noisy, q=0.1)
    assert float(w[0]) == 0.0 and float(w[4]) == 0.0
    # monotone in reward over the noisy rows; NaN ranks lowest
    order = [7, 5, 2, 3, 1, 6]
    vals = [float(w[i]) for i in order]
    assert vals == sorted(vals)
    assert float(w.sum()) == pytest.approx(0.0, abs=1e-9)
    b = blom_scores(6)
    assert float((w[noisy].double().sort().values * b).sum()) == pytest.approx(registered_expected_step(0.1), rel=1e-6)


def test_antithetic_rows_mirror_and_nominal_slots() -> None:
    ns = torch.ones(8, 4, 3)
    ns[0] = 0.0
    g = torch.Generator().manual_seed(0)
    k = antithetic_knot_noise(ns, 0.5, generator=g)
    assert k.shape == (8, 4, 3)
    assert torch.equal(k[4:], -k[:4])
    assert float(k[0].abs().max()) == 0.0 and float(k[4].abs().max()) == 0.0
    assert noisy_rows(k).tolist() == [False, True, True, True, False, True, True, True]
    with pytest.raises(ValueError):
        antithetic_knot_noise(torch.ones(7, 4, 3), 1.0)


def test_knot_interp_matches_linear_upsampling() -> None:
    knots = torch.tensor([[[0.0], [1.0], [3.0]]])
    out = knot_interp(knots, 4)
    assert out.shape == (1, 12, 1)
    assert float(out[0, 0, 0]) == 0.0 and float(out[0, -1, 0]) == 3.0
    assert bool((out[0, 1:, 0] >= out[0, :-1, 0]).all())


def test_rank_update_and_registered_update() -> None:
    base = torch.zeros(2, 3)
    samples = torch.stack([torch.full((2, 3), float(i)) for i in range(4)])
    w = torch.tensor([0.0, -1.0, 0.5, 0.5])
    assert torch.allclose(rank_update(base, samples, w), torch.full((2, 3), 1.5))
    rews = torch.arange(20, dtype=torch.float32)
    rw, nan = registered_weights(rews, 20, 1.0)
    assert not bool(nan.any())
    assert float(rw.sum()) == pytest.approx(1.0, abs=1e-6)
    assert int((rw > 0).sum()) == 2          # top 10% of 20
    s = torch.randn(20, 2, 3)
    assert torch.allclose(registered_update(s, rw), (rw[:, None, None] * s).sum(0))


def _trial(n: int, d: int, trials: int, antithetic: bool, kappa: float, seed: int) -> tuple[float, float, float, float]:
    """f(eps) = eps_0 + kappa * |eps|^2. Returns (signal, SNR) for rank weights and for the uniform top-10% mean."""
    gen = torch.Generator().manual_seed(seed)
    sig_r, perp_r, sig_t, perp_t = [], [], [], []
    for _ in range(trials):
        if antithetic:
            half = torch.randn(n // 2, d, generator=gen)
            half[0] = 0.0
            eps = torch.cat([half, -half])
        else:
            eps = torch.randn(n, d, generator=gen)
            eps[0] = 0.0
        rews = eps[:, 0] + kappa * (eps ** 2).sum(1)
        w = rank_weights(rews, eps.abs().amax(1) > 0, 0.1)
        u = (w[:, None] * eps).sum(0)
        sig_r.append(float(u[0]))
        perp_r.append(float((u[1:] ** 2).sum()))
        ut = eps[torch.topk(rews, int(0.1 * n)).indices].mean(0)
        sig_t.append(float(ut[0]))
        perp_t.append(float((ut[1:] ** 2).sum()))
    m = lambda x: sum(x) / len(x)  # noqa: E731
    return m(sig_r), m(sig_r) ** 2 / m(perp_r), m(sig_t), m(sig_t) ** 2 / m(perp_t)


def test_linear_regime_same_step_lower_variance() -> None:
    sig_r, snr_r, sig_t, snr_t = _trial(256, 60, 200, antithetic=False, kappa=0.0, seed=3)
    assert sig_r == pytest.approx(sig_t, rel=0.08)       # same expected step as the registered top-10% mean
    assert snr_r / snr_t > 2.5                            # theory 3.25 at large N


def test_antithetic_is_curvature_immune() -> None:
    """Antithetic + rank keeps its SNR whatever the curvature; the registered estimator collapses under curvature.

    In the purely linear model antithetic pairs halve the independent directions (SNR ~ half of rank-only), still
    above the registered estimator; under strong curvature they are several times better than both."""
    _, snr_al, _, snr_tl = _trial(256, 60, 200, antithetic=True, kappa=0.0, seed=4)
    _, snr_ac, _, snr_tc = _trial(256, 60, 200, antithetic=True, kappa=-0.3, seed=5)
    _, snr_rc, _, _ = _trial(256, 60, 200, antithetic=False, kappa=-0.3, seed=6)
    assert snr_al / snr_tl > 1.4
    assert snr_ac / snr_tc > 5.0
    assert snr_ac / snr_rc > 3.0
    assert math.isfinite(snr_ac)
