"""Own PPO with asymmetric actor-critic, GAE and the TrackingEnv protocol (design Sections 7-8)."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import pytest
import torch

from fastdo.policy.networks import LOG_STD_MAX, ActorCritic
from fastdo.policy.ppo import (
    PpoConfig,
    clipped_surrogate,
    clipped_value_loss,
    compute_gae,
    ppo_loss,
    ppo_update,
    train,
)
from fastdo.policy.protocol import TIME_OUTS_KEY, TrackingEnv
from fastdo.policy.rollout import EpisodeTracker, PpoBatch, collect_rollout
from tests.policy.toy_env import ToyEnvConfig, ToyTrackingEnv

logger = logging.getLogger(__name__)

DT = torch.float64
N_UPDATES = 150
TOY_HIDDEN = (256, 256, 256)
SEED = 0     # env seed; the learner, the held-out env and its noise use SEED + 1, + 2, + 3 (distinct streams)


def test_toy_env_implements_protocol_deterministically() -> None:
    env_a, env_b = ToyTrackingEnv(), ToyTrackingEnv()
    assert isinstance(env_a, TrackingEnv)
    assert not isinstance(object(), TrackingEnv)
    gen = torch.Generator().manual_seed(3)
    n_failed = 0
    for _ in range(env_a.cfg.episode_steps):
        action = torch.randn(env_a.n_worlds, env_a.act_dim, generator=gen)
        obs_a, rew_a, done_a, info_a = env_a.step(action)
        obs_b, rew_b, done_b, _ = env_b.step(action)
        assert obs_a.shape == (64, env_a.obs_dim) and rew_a.shape == (64,) and done_a.dtype == torch.bool
        assert torch.equal(obs_a, obs_b) and torch.equal(rew_a, rew_b) and torch.equal(done_a, done_b)
        assert not bool((info_a[TIME_OUTS_KEY] & ~done_a).any())
        n_failed += int((done_a & ~info_a[TIME_OUTS_KEY]).sum())
    assert n_failed > 0                                                  # random actions lose the track
    assert env_a.privileged().shape == (64, env_a.priv_dim)
    # Without the deviation limit every episode is truncated after exactly episode_steps, then auto-resets.
    env = ToyTrackingEnv(ToyEnvConfig(max_error=math.inf))
    for k in range(env.cfg.episode_steps):
        _, _, done, info = env.step(torch.randn(env.n_worlds, 1, generator=gen))
        assert bool(done.all()) == (k == env.cfg.episode_steps - 1) and torch.equal(info[TIME_OUTS_KEY], done)
    assert torch.equal(env.privileged()[:, 1], torch.zeros(64))
    other = ToyTrackingEnv(ToyEnvConfig(seed=1))
    assert not torch.equal(other.reset(None, None), ToyTrackingEnv().reset(None, None))


def test_gae_matches_hand_computation() -> None:
    """3-step scalar example, gamma 0.9, lam 0.8 (gamma lam = 0.72), rewards [1, 0, 2], values
    [0.5, 0.4, 0.3], last value 0.2:
        delta_2 = 2 + 0.9 * 0.2 - 0.3 = 1.88,  delta_1 = 0 + 0.9 * 0.3 - 0.4 = -0.13,
        delta_0 = 1 + 0.9 * 0.4 - 0.5 = 0.86,
        A_2 = 1.88,  A_1 = -0.13 + 0.72 * 1.88 = 1.2236,  A_0 = 0.86 + 0.72 * 1.2236 = 1.740992.
    With the episode ending at t = 1 (s_2 is a fresh episode): delta_1 = -0.4 and the recursion is cut,
        A_2 = 1.88,  A_1 = -0.4,  A_0 = 0.86 + 0.72 * (-0.4) = 0.572."""
    rewards = torch.tensor([1.0, 0.0, 2.0], dtype=DT)
    values = torch.tensor([0.5, 0.4, 0.3], dtype=DT)
    last_value = torch.tensor(0.2, dtype=DT)
    adv, ret = compute_gae(rewards, values, torch.zeros(3, dtype=torch.bool), last_value, 0.9, 0.8)
    expected = torch.tensor([1.740992, 1.2236, 1.88], dtype=DT)
    torch.testing.assert_close(adv, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(ret, expected + values, rtol=0.0, atol=1e-12)
    dones = torch.tensor([False, True, False])
    adv, ret = compute_gae(rewards, values, dones, last_value, 0.9, 0.8)
    expected = torch.tensor([0.572, -0.4, 1.88], dtype=DT)
    torch.testing.assert_close(adv, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(ret, expected + values, rtol=0.0, atol=1e-12)


def test_gae_batched_matches_truncated_discounted_sum() -> None:
    """A_t = sum_l (gamma lam)^l prod_{j<l} (1 - done_{t+j}) delta_{t+l} on (T, N) with random dones."""
    gen = torch.Generator().manual_seed(0)
    T, N, gamma, lam = 17, 5, 0.99, 0.95
    rewards = torch.randn(T, N, dtype=DT, generator=gen)
    values = torch.randn(T, N, dtype=DT, generator=gen)
    dones = torch.rand(T, N, generator=gen) < 0.2
    last_value = torch.randn(N, dtype=DT, generator=gen)
    d = dones.to(DT)
    next_values = torch.cat([values[1:], last_value[None]])
    delta = rewards + gamma * (1.0 - d) * next_values - values
    expected = torch.zeros(T, N, dtype=DT)
    for t in range(T):
        coef = torch.ones(N, dtype=DT)
        for lag in range(T - t):
            expected[t] += coef * delta[t + lag]
            coef = coef * gamma * lam * (1.0 - d[t + lag])
    adv, ret = compute_gae(rewards, values, dones, last_value, gamma, lam)
    torch.testing.assert_close(adv, expected, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(ret, expected + values, rtol=0.0, atol=1e-12)


def test_ppo_ratio_clipped() -> None:
    """A logp gap of 2.0 puts the ratio at e^2 (or e^-2), outside [1 - clip, 1 + clip]."""
    # Scalar form: positive advantage and ratio e^2 -> the clipped branch (1 + clip) A, zero gradient;
    # negative advantage -> the pessimistic unclipped branch e^2 A, d loss / d logp = -A e^2 = e^2.
    logp = torch.zeros(1, dtype=DT, requires_grad=True)
    loss = clipped_surrogate(logp, torch.full((1,), -2.0, dtype=DT), torch.ones(1, dtype=DT), 0.2)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(-1.2, abs=1e-12)
    assert logp.grad is not None and float(logp.grad.abs().max()) == 0.0
    logp.grad = None
    loss = clipped_surrogate(logp, torch.full((1,), -2.0, dtype=DT), -torch.ones(1, dtype=DT), 0.2)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(math.exp(2.0), abs=1e-12)
    assert logp.grad is not None and float(logp.grad[0]) == pytest.approx(math.exp(2.0), abs=1e-12)

    # Model form, through the loss ppo_update minimizes (per-minibatch advantage normalization included):
    # the gap follows the sign of the normalized advantage, so every sample is clipped and the actor
    # (mean network and log-std) receives exactly zero gradient; the opposite gap is unclipped.
    torch.manual_seed(0)
    model = ActorCritic(3, 2, 2, hidden=(16, 16)).double()
    B = 64
    obs, priv, action = torch.randn(B, 3, dtype=DT), torch.randn(B, 2, dtype=DT), torch.randn(B, 2, dtype=DT)
    adv = torch.randn(B, dtype=DT)
    with torch.no_grad():
        logp_now, _, value_now = model.evaluate(obs, priv, action)
    sign = torch.sign(adv - adv.mean())
    cfg = PpoConfig(ent_coef=0.0)
    actor_params = [p for n, p in model.named_parameters() if not n.startswith("critic")]
    for gap_sign, clipped in ((1.0, True), (-1.0, False)):
        batch = PpoBatch(obs=obs, priv=priv, actions=action, logp_old=logp_now - gap_sign * 2.0 * sign,
                         values_old=value_now, advantages=adv, returns=value_now)
        model.zero_grad()
        loss, stats = ppo_loss(model, batch, cfg)
        loss.backward()
        grad_norm = torch.sqrt(sum((p.grad**2).sum() for p in actor_params if p.grad is not None))
        if clipped:
            assert float(grad_norm) == 0.0
            assert float(stats["clip_fraction"]) == 1.0
            norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            factor = torch.where(norm_adv > 0, 1.0 + cfg.clip, 1.0 - cfg.clip)
            torch.testing.assert_close(stats["policy_loss"], -(factor * norm_adv).mean())
        else:
            assert float(grad_norm) > 1e-3


def test_value_loss_clipped() -> None:
    """Old value 0, return 1, clip 0.2: a new value 0.5 is clipped to 0.2 and the larger clipped error
    0.5 (1 - 0.2)^2 = 0.32 is used, with zero gradient; a new value 0.1 is inside the band, gradient
    v - R = -0.9."""
    old, ret = torch.zeros(1, dtype=DT), torch.ones(1, dtype=DT)
    v = torch.full((1,), 0.5, dtype=DT, requires_grad=True)
    loss = clipped_value_loss(v, old, ret, 0.2)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.32, abs=1e-12)
    assert v.grad is not None and float(v.grad[0]) == 0.0
    v = torch.full((1,), 0.1, dtype=DT, requires_grad=True)
    loss = clipped_value_loss(v, old, ret, 0.2)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.405, abs=1e-12)
    assert v.grad is not None and float(v.grad[0]) == pytest.approx(-0.9, abs=1e-12)
    # value_clip = inf disables the clipping: plain 0.5 (V - R)^2 = 0.125, gradient V - R = -0.5.
    v = torch.full((1,), 0.5, dtype=DT, requires_grad=True)
    loss = clipped_value_loss(v, old, ret, math.inf)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.125, abs=1e-12)
    assert v.grad is not None and float(v.grad[0]) == pytest.approx(-0.5, abs=1e-12)


def test_actor_critic_gaussian_and_asymmetric_critic() -> None:
    torch.manual_seed(0)
    model = ActorCritic(4, 3, 2, hidden=(32, 32), init_log_std=-0.5).double()
    obs, priv, action = torch.randn(10, 4, dtype=DT), torch.randn(10, 3, dtype=DT), torch.randn(10, 2, dtype=DT)
    logp, entropy, value = model.evaluate(obs, priv, action)
    dist = torch.distributions.Normal(model.mean_action(obs), model.log_std.exp())
    torch.testing.assert_close(logp, dist.log_prob(action).sum(-1), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(entropy, dist.entropy().sum(-1), rtol=0.0, atol=1e-12)
    assert value.shape == (10,)
    # The critic sees the privileged features, the actor does not.
    logp_2, _, value_2 = model.evaluate(obs, priv + 1.0, action)
    torch.testing.assert_close(logp_2, logp, rtol=0.0, atol=0.0)
    assert float((value_2 - value).detach().abs().min()) > 0.0
    gen = torch.Generator().manual_seed(1)
    a, lp, v = model.act(obs, priv, generator=gen)
    torch.testing.assert_close(lp, model.evaluate(obs, priv, a)[0], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(v, model.value(obs, priv), rtol=0.0, atol=1e-12)
    a_det, _, v_none = model.act(obs, deterministic=True)
    torch.testing.assert_close(a_det, model.mean_action(obs), rtol=0.0, atol=0.0)
    assert bool(torch.isnan(v_none).all())                # no privileged features -> no critic value,
    no_dones = torch.zeros(1, 10, dtype=torch.bool)       # which GAE refuses instead of propagating it
    with pytest.raises(ValueError, match="values"):
        compute_gae(torch.zeros(1, 10, dtype=DT), v_none[None], no_dones, v.detach(), 0.99, 0.95)
    with pytest.raises(ValueError, match="last_value"):
        compute_gae(torch.zeros(1, 10, dtype=DT), v[None].detach(), no_dones, v_none, 0.99, 0.95)
    # A critic without privileged features (priv_dim 0) needs no priv: act(obs) returns its value.
    no_priv = ActorCritic(4, 0, 2, hidden=(32, 32)).double()
    _, _, v_obs = no_priv.act(obs)
    torch.testing.assert_close(v_obs, no_priv.value(obs, obs[:, :0]), rtol=0.0, atol=0.0)


def test_log_std_projected_into_bounds_by_ppo_update() -> None:
    """A PPO step pushing the log-std past LOG_STD_MAX leaves the parameter on the bound, not beyond it
    (where the clamp's gradient vanishes); on the bound the gradient flows; `action_std` is the used std."""
    with pytest.raises(ValueError, match="init_log_std"):
        ActorCritic(2, 1, 1, init_log_std=LOG_STD_MAX + 1.0)
    torch.manual_seed(0)
    model = ActorCritic(2, 1, 1, hidden=(8,), init_log_std=LOG_STD_MAX - 0.05).double()
    B = 64
    obs, priv = torch.randn(B, 2, dtype=DT), torch.randn(B, 1, dtype=DT)
    far = (torch.arange(B) % 2 == 0)[:, None]
    with torch.no_grad():
        mean = model.mean_action(obs)
        actions = torch.where(far, mean + 10.0 * model.action_std(), mean)
        logp, _, values = model.evaluate(obs, priv, actions)
    # Positive advantages on the far samples (z = 10), negative on the mean: widening the Gaussian helps,
    # and Adam's first step moves the log-std up by about lr = 0.1, past the bound.
    batch = PpoBatch(obs=obs, priv=priv, actions=actions, logp_old=logp, values_old=values,
                     advantages=2.0 * far[:, 0].to(DT) - 1.0, returns=values)
    ppo_update(model, torch.optim.Adam(model.parameters(), lr=0.1), batch, PpoConfig(n_epochs=1,
                                                                                        n_minibatches=1))
    assert float(model.log_std.detach()) == LOG_STD_MAX
    torch.testing.assert_close(model.action_std(), torch.full((1,), math.exp(LOG_STD_MAX), dtype=DT))
    model.zero_grad()
    model.evaluate(obs, priv, mean)[0].sum().backward()   # samples on the mean favour a narrower Gaussian
    assert model.log_std.grad is not None and float(model.log_std.grad) < 0.0


def test_rollout_bootstraps_time_outs() -> None:
    """Truncated episode ends get r_t + gamma V(s_t) (the terminal state's value is approximated by the
    value of the state the last action was taken in); the episode tracker sees the raw rewards."""
    cfg_env = ToyEnvConfig(n_worlds=8, episode_steps=3, max_error=math.inf)
    env, replay = ToyTrackingEnv(cfg_env), ToyTrackingEnv(cfg_env)
    torch.manual_seed(0)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=(16, 16))
    tracker = EpisodeTracker(env.n_worlds)
    gamma = 0.9
    obs0 = env.reset(None, None)
    assert torch.equal(replay.reset(None, None), obs0)
    rollout, _ = collect_rollout(env, model, obs0, 4, gamma, tracker, torch.Generator().manual_seed(0))
    raw = torch.stack([replay.step(rollout.actions[t])[1] for t in range(4)])
    assert torch.equal(rollout.dones[2], torch.ones(8, dtype=torch.bool))
    with torch.no_grad():
        v2 = model.value(rollout.obs[2], rollout.priv[2])
    torch.testing.assert_close(rollout.rewards[2], raw[2] + gamma * v2)
    torch.testing.assert_close(rollout.rewards[[0, 1, 3]], raw[[0, 1, 3]])
    returns, lengths = tracker.pop()
    torch.testing.assert_close(returns, raw[:3].double().sum(0))
    assert torch.equal(lengths, torch.full((8,), 3))


def test_rollout_does_not_bootstrap_failures() -> None:
    """In a rollout with both episode ends, failures (done without time-out) keep the raw reward and
    time-outs get raw + gamma V(s_t), so bootstrapping every done would fail here."""
    cfg_env = ToyEnvConfig(n_worlds=8, episode_steps=3, max_error=0.05)
    env, replay = ToyTrackingEnv(cfg_env), ToyTrackingEnv(cfg_env)
    torch.manual_seed(0)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=(16, 16))
    gamma, n_steps = 0.9, 6
    obs0 = env.reset(None, None)
    assert torch.equal(replay.reset(None, None), obs0)
    rollout, _ = collect_rollout(env, model, obs0, n_steps, gamma, None, torch.Generator().manual_seed(0))
    steps = [replay.step(rollout.actions[t]) for t in range(n_steps)]
    raw = torch.stack([s[1] for s in steps])
    assert torch.equal(rollout.dones, torch.stack([s[2] for s in steps]))
    time_outs = torch.stack([s[3][TIME_OUTS_KEY] for s in steps])
    failures = rollout.dones & ~time_outs
    assert int(time_outs.sum()) > 0 and int(failures.sum()) > 0
    with torch.no_grad():
        values = model.value(rollout.obs, rollout.priv)
    assert float((gamma * values[failures]).abs().min()) > 1e-3
    torch.testing.assert_close(rollout.rewards[failures], raw[failures], rtol=0.0, atol=0.0)
    torch.testing.assert_close(rollout.rewards[~rollout.dones], raw[~rollout.dones], rtol=0.0, atol=0.0)
    torch.testing.assert_close(rollout.rewards[time_outs], raw[time_outs] + gamma * values[time_outs])


class _TimeOutWithoutDone(ToyTrackingEnv):
    """A broken environment: flags a time-out on a step that did not end the episode."""

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        broken = info[TIME_OUTS_KEY].clone()
        broken[0] = bool(self.k[0] == 2)
        return obs, reward, done & ~broken, {TIME_OUTS_KEY: broken}


def test_rollout_rejects_time_outs_without_done() -> None:
    env = _TimeOutWithoutDone(ToyEnvConfig(n_worlds=4, max_error=float("inf")))
    torch.manual_seed(0)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=(16, 16))
    with pytest.raises(ValueError, match="time_outs"):
        collect_rollout(env, model, env.reset(None, None), 4, 0.99)
    good = ToyTrackingEnv(ToyEnvConfig(n_worlds=4))
    collect_rollout(good, model, good.reset(None, None), 4, 0.99)


def test_train_steps_a_given_optimizer_to_resume() -> None:
    """`train` steps a caller-owned optimizer, so its state survives across calls (checkpoint, resume)."""
    env = ToyTrackingEnv(ToyEnvConfig(n_worlds=8))
    cfg = PpoConfig(n_steps=4, n_epochs=2, n_minibatches=2, hidden=(16, 16))
    torch.manual_seed(0)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=cfg.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(1)
    for n_updates in (2, 1):
        assert train(env, model, cfg, n_updates, optimizer=optimizer, generator=gen) is model
    steps = {float(state["step"]) for state in optimizer.state.values()}
    assert steps == {3.0 * cfg.n_epochs * cfg.n_minibatches}


def _mean_episode_return(model: ActorCritic, deterministic: bool) -> float:
    """Mean return of the first episode of each of 256 held-out worlds (env seed SEED + 2, action noise
    seed SEED + 3), so short failed episodes are not over-counted."""
    env: TrackingEnv = ToyTrackingEnv(ToyEnvConfig(n_worlds=256, seed=SEED + 2))
    gen = torch.Generator().manual_seed(SEED + 3)
    obs = env.reset(None, None)
    returns = torch.zeros(env.n_worlds, dtype=DT)
    finished = torch.zeros(env.n_worlds, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(ToyEnvConfig().episode_steps):
            action, _, _ = model.act(obs, deterministic=deterministic, generator=gen)
            obs, reward, done, _ = env.step(action)
            returns += torch.where(finished, 0.0, reward.to(DT))
            finished |= done
    assert bool(finished.all())
    return float(returns.mean())


@pytest.fixture(scope="module")
def toy_training() -> dict[str, Any]:
    # Default PpoConfig except the width: 3x256 instead of 3x512 keeps the depth at a quarter of the cost
    # for the toy's 4-D observation (the default 3x512 also passes, measured).
    cfg = PpoConfig(hidden=TOY_HIDDEN)
    env = ToyTrackingEnv(ToyEnvConfig(seed=SEED))
    torch.manual_seed(SEED)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=cfg.hidden)
    before = {det: _mean_episode_return(model, det) for det in (False, True)}
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    train(env, model, cfg, N_UPDATES, history.append, generator=torch.Generator().manual_seed(SEED + 1))
    elapsed = time.perf_counter() - t0
    after = {det: _mean_episode_return(model, det) for det in (False, True)}
    logger.info("toy PPO: %.1f s, stochastic %.4f -> %.4f, deterministic %.4f -> %.4f",
                elapsed, before[False], after[False], before[True], after[True])
    # Step-size maxima over every update (the module's log lines subsample the history).
    kl, clip_frac = [h["approx_kl"] for h in history], [h["clip_fraction"] for h in history]
    logger.info("toy PPO step size: max approx_kl %.4f (update %d), max clip_fraction %.3f; updates >= 10: "
                "max approx_kl %.4f, max clip_fraction %.3f", max(kl), kl.index(max(kl)), max(clip_frac),
                max(kl[10:]), max(clip_frac[10:]))
    return {"before": before, "after": after, "elapsed": elapsed, "history": history}


def test_ppo_solves_toy_tracking_env(toy_training: dict[str, Any]) -> None:
    """64 worlds, CPU, seed 0, PpoConfig defaults except hidden=(256, 256, 256): the mean episode
    return of the stochastic policy (the PPO objective) on 256 held-out episodes improves by more than 5x
    within 150 updates, reaching over 80% of the maximal return, and the deterministic policy improves."""
    before, after, history = toy_training["before"], toy_training["after"], toy_training["history"]
    assert len(history) == N_UPDATES
    assert all(math.isfinite(h["policy_loss"]) and math.isfinite(h["value_loss"]) for h in history)
    assert after[False] > 5.0 * before[False]
    max_return = ToyEnvConfig().episode_steps * ToyEnvConfig().dt
    assert after[False] > 0.8 * max_return
    assert after[True] > before[True]


@pytest.mark.timing
def test_ppo_toy_training_under_60_s(toy_training: dict[str, Any]) -> None:
    """The 150 updates of the toy run finish in under 60 s under the 4-thread cap."""
    assert toy_training["elapsed"] < 60.0, toy_training["elapsed"]

