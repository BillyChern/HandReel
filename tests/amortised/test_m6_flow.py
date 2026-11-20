"""M6: the conditional flow-matching prior (known answers and mutations).

Known answers (synthetic, CPU): 2-mode recovery (6 clips, each with two modes +mu_c / -mu_c; mode recall >= 0.9 at
S = 64); the conditional switch (>= 0.9 of each clip's samples sit nearest one of ITS OWN modes; without
conditioning the prior can only learn the 12-mode mixture); overfit one trajectory (one clip, one solution: every
sample lands on it); padding invariance (v on the valid knots is the same at any pad length: training pads to K_max,
sampling runs at the clip's own K); overfit with HETEROGENEOUS K (clips of 8 and 16 knots, each sampled at its own
length, lands on its solution). Further mutation target: GroupNorm statistics over the padded knots (the
heterogeneous-K overfit and padding invariance fail). Mutation targets: conditioning dropped (switch fails); the tau convention swapped between
training and sampling (the overfit known answer fails; the 2-mode recovery, at its 0.25 RMS tolerance, does NOT catch
it: batch-C mutation record).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastdo.amortised.flow_prior import (
    FlowConfig,
    FlowNet,
    PriorClip,
    PriorData,
    sample,
    sample_batch,
    train_prior,
)

C, K, R, CK, G = 6, 8, 3, 4, 2
TOL = 0.25                      # RMS distance to a mode (modes have RMS 1)


def _data(seed: int = 0) -> tuple[PriorData, list[list[np.ndarray]]]:
    rng = np.random.default_rng(seed)
    clips, modes = [], []
    for _ in range(C):
        kc, gc = rng.normal(size=(K, CK)), rng.normal(size=G)
        mu = rng.normal(size=(K, R))
        mu /= np.sqrt((mu ** 2).mean())
        sols = [s * mu + 0.03 * rng.normal(size=(K, R)) for s in (1, -1) for _ in range(10)]
        clips.append(PriorClip(kc, gc, sols))
        modes.append([mu, -mu])
    return PriorData(clips), modes


def _cfg() -> FlowConfig:
    return FlowConfig(R=R, knot_cond=CK, glob_cond=G, channels=32, levels=2, film=64)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float32)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    data, modes = _data()
    net = FlowNet(_cfg())
    train_prior(net, data, steps=900, batch=96, lr=2e-3, seed=0)
    return net, data, modes


def _rms(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.sqrt(((x - m) ** 2).mean((1, 2)))


def test_two_mode_recovery_recall_at_s64(trained) -> None:
    net, data, modes = trained
    hit = 0
    for c, clip in enumerate(data.clips):
        x = sample(net, _t(clip.kcond), _t(clip.gcond), torch.ones(K), 64, seed=1).numpy()
        hit += sum(_rms(x, m).min() < TOL for m in modes[c])
    assert hit / (2 * C) >= 0.9, hit


def test_conditional_switch(trained) -> None:
    net, data, modes = trained
    allm = [(c, m) for c in range(C) for m in modes[c]]
    own = tot = 0
    for c, clip in enumerate(data.clips):
        x = sample(net, _t(clip.kcond), _t(clip.gcond), torch.ones(K), 64, seed=2).numpy()
        d = np.stack([_rms(x, m) for _, m in allm], 1)
        own += sum(allm[j][0] == c for j in d.argmin(1))
        tot += len(x)
    assert own / tot >= 0.9, own / tot


def test_overfit_one_trajectory() -> None:
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=(K, R))
    data = PriorData([PriorClip(rng.normal(size=(K, CK)), rng.normal(size=G), [x1])])
    torch.manual_seed(1)
    net = FlowNet(_cfg())
    train_prior(net, data, steps=400, batch=64, lr=2e-3, seed=1)
    x = sample(net, _t(data.clips[0].kcond), _t(data.clips[0].gcond), torch.ones(K), 32, seed=4).numpy()
    assert _rms(x, x1).max() < 0.1, _rms(x, x1).max()


def test_samples_are_prefix_consistent_and_masked(trained) -> None:
    net, data, _ = trained
    clip = data.clips[0]
    a = sample(net, _t(clip.kcond), _t(clip.gcond), torch.ones(K), 4, seed=9)
    b = sample(net, _t(clip.kcond), _t(clip.gcond), torch.ones(K), 16, seed=9)
    assert torch.allclose(a, b[:4], atol=1e-5)
    m = torch.ones(K)
    m[-3:] = 0.0
    c = sample(net, _t(clip.kcond), _t(clip.gcond), m, 4, seed=9)
    assert torch.all(c[:, -3:] == 0.0)


def test_batches_draw_clips_uniformly_then_a_solution() -> None:
    rng = np.random.default_rng(0)
    clips = [PriorClip(np.zeros((K, CK)), np.zeros(G), [np.full((K, R), float(i))] * n)
             for i, n in enumerate((1, 50, 5))]
    _, _, _, _, _, ids = sample_batch(PriorData(clips), 6000, rng)
    freq = np.bincount(ids, minlength=3) / 6000
    np.testing.assert_allclose(freq, 1 / 3, atol=0.03)


def test_velocity_on_valid_knots_does_not_depend_on_the_padding() -> None:
    torch.manual_seed(0)
    net = FlowNet(FlowConfig(R=4, knot_cond=6, glob_cond=3)).eval()
    k = 10
    x, kc, gc, tau = torch.randn(1, k, 4), torch.randn(1, k, 6), torch.randn(1, 3), torch.tensor([0.3])
    with torch.no_grad():
        v_own = net(x, tau, kc, gc, torch.ones(1, k))
        assert float(v_own.abs().mean()) > 0.05
        for k_max in (12, 16, 24, 68):
            xp, kp, mp = torch.zeros(1, k_max, 4), torch.zeros(1, k_max, 6), torch.zeros(1, k_max)
            xp[:, :k], kp[:, :k], mp[:, :k] = x, kc, 1.0
            v_pad = net(xp, tau, kp, gc, mp)
            torch.testing.assert_close(v_pad[:, :k], v_own, atol=1e-5, rtol=0)
            assert torch.all(v_pad[:, k:] == 0.0)


def test_overfit_with_heterogeneous_knot_counts() -> None:
    """Two clips of K = 8 and K = 16 (trained padded to 16), each sampled at ITS OWN length."""
    rng = np.random.default_rng(3)
    xs = {k: rng.normal(size=(k, R)) for k in (8, 16)}
    clips = [PriorClip(rng.normal(size=(k, CK)), rng.normal(size=G), [xs[k]]) for k in (8, 16)]
    torch.manual_seed(1)
    net = FlowNet(_cfg())
    train_prior(net, PriorData(clips), steps=800, batch=64, lr=2e-3, seed=1)
    for c, k in zip(clips, (8, 16), strict=True):
        x = sample(net, _t(c.kcond), _t(c.gcond), torch.ones(k), 32, seed=4).numpy()
        assert _rms(x, xs[k]).max() < 0.1, (k, _rms(x, xs[k]).max())
