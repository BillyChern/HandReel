"""PPO rollouts on the real CPU environment: bootstrapping, failures, NaN values, package exports.

The time-out subset rule is tested on the environment in test_env.py and on `collect_rollout` in test_ppo.py.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from fastdo.policy.env import make_env
from fastdo.policy.networks import ActorCritic
from fastdo.policy.ppo import compute_gae
from fastdo.policy.protocol import TIME_OUTS_KEY, TrackingEnv
from fastdo.policy.rollout import collect_rollout
from tests.policy.env_fixtures import SCENE_XML, env_config, scene_missing, whisking_reference

HIDDEN = (32, 32)


def _model(env: TrackingEnv, init_log_std: float = -3.0) -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=HIDDEN, init_log_std=init_log_std)


@scene_missing
def test_rollout_last_value_is_value_of_final_observation() -> None:
    """last_value must be V(o_T, p_T) of the observation after the last step, not V(o_{T-1}, p_{T-1})."""
    env = make_env(env_config(n_worlds=4, warmup_duration=0.04), SCENE_XML, "cpu")
    model = _model(env, init_log_std=-1.0)
    obs = env.reset(whisking_reference(), None)
    rollout, obs_T = collect_rollout(env, model, obs, 5, 0.99, None, torch.Generator().manual_seed(0))
    with torch.no_grad():
        v_T = model.value(obs_T, env.privileged())
        v_prev = model.value(rollout.obs[-1], rollout.priv[-1])
    torch.testing.assert_close(rollout.last_value, v_T, rtol=0.0, atol=0.0)
    assert float((v_T - v_prev).abs().min()) > 1e-6            # the two candidates are distinguishable here


@scene_missing
def test_rollout_does_not_bootstrap_env_failures() -> None:
    """World 0 tracks a 4-frame clip (times out every 3 steps before the object has fallen 0.1 m), world 1 a long
    clip whose unheld object drops (a failure): only the time-outs get gamma V(s_t) added."""
    cfg = env_config(n_worlds=2, warmup_duration=0.0)
    env, replay = make_env(cfg, SCENE_XML, "cpu"), make_env(cfg, SCENE_XML, "cpu")
    short, long = whisking_reference(n_frames=4), whisking_reference(n_frames=80)
    for e in (env, replay):
        e.reset(short, torch.tensor([0]))
        e.reset(long, torch.tensor([1]))
    model, gamma, n_steps = _model(env), 0.9, 12
    obs = env.reset(None, None)
    assert torch.equal(obs, replay.reset(None, None))
    rollout, _ = collect_rollout(env, model, obs, n_steps, gamma, None, torch.Generator().manual_seed(0))
    steps = [replay.step(rollout.actions[t]) for t in range(n_steps)]
    raw = torch.stack([s[1] for s in steps])
    time_outs = torch.stack([s[3][TIME_OUTS_KEY] for s in steps])
    assert torch.equal(rollout.dones, torch.stack([s[2] for s in steps]))
    failures = rollout.dones & ~time_outs
    assert bool(time_outs[:, 0].any()) and bool(failures[:, 1].any())
    with torch.no_grad():
        values = model.value(rollout.obs, rollout.priv)
    assert float((gamma * values[failures]).abs().min()) > 1e-4
    torch.testing.assert_close(rollout.rewards[failures], raw[failures], rtol=0.0, atol=0.0)
    torch.testing.assert_close(rollout.rewards[~rollout.dones], raw[~rollout.dones], rtol=0.0, atol=0.0)
    torch.testing.assert_close(rollout.rewards[time_outs], raw[time_outs] + gamma * values[time_outs])


@scene_missing
def test_nan_critic_value_without_priv_rejected_on_env() -> None:
    env = make_env(env_config(n_worlds=2), SCENE_XML, "cpu")
    model = _model(env)
    obs = env.reset(whisking_reference(), None)
    action, _, value = model.act(obs)                           # deployment form: no privileged features
    assert bool(torch.isnan(value).all())
    _, reward, done, _ = env.step(action)
    with torch.no_grad():
        last = model.value(env.reset(None, None), env.privileged())
    with pytest.raises(ValueError, match="values"):
        compute_gae(reward[None], value[None], done[None], last, 0.99, 0.95)


def test_package_exports_policy_api_without_simulator_imports() -> None:
    from fastdo import policy

    expected = {"ActorCritic", "PpoConfig", "compute_gae", "ppo_update", "train", "TrackingEnv", "TIME_OUTS_KEY",
                "collect_rollout", "EpisodeTracker", "PpoBatch", "Rollout", "RewardConfig", "StepObservation",
                "step_reward", "terminated", "clip_success", "pose_errors", "EnvConfig", "make_env", "obs_layout",
                "PolicyReference", "canonical_frame", "future_window"}
    assert expected <= set(policy.__all__)
    assert all(hasattr(policy, name) for name in policy.__all__)
    code = "import sys, fastdo.policy; print(any(m in sys.modules for m in ('warp', 'mujoco_warp')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == "False"
