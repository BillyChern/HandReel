"""Failure paths and reference bookkeeping of the CPU twin (non-finite state, a drop on the last
frame, the step guard and the bounded clip registry)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from fastdo.policy.env import MujocoCpuEnv, make_env
from fastdo.policy.protocol import TIME_OUTS_KEY
from fastdo.policy.reference import PolicyReference
from tests.policy.env_fixtures import SCENE_XML, env_config, scene_missing, whisking_reference

pytestmark = scene_missing


def _zeros(n: int) -> torch.Tensor:
    return torch.zeros(n, 28)


def test_non_finite_state_is_a_failure_and_the_world_restarts_cleanly(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)                           # MuJoCo writes its divergence warning to MUJOCO_LOG.TXT
    cfg = env_config(n_worlds=2, warmup_duration=0.0)
    env, twin = make_env(cfg, SCENE_XML, "cpu"), make_env(env_config(n_worlds=1, warmup_duration=0.0), SCENE_XML, "cpu")
    assert isinstance(env, MujocoCpuEnv)
    ref = whisking_reference(n_frames=30)
    env.reset(ref, None)
    fresh = twin.reset(ref, None)
    env.step(_zeros(2))
    env._datas[1].qvel[:] = math.nan                    # a diverged world (MuJoCo's own auto-reset is disabled)
    obs, reward, done, info = env.step(_zeros(2))
    assert not bool(torch.isfinite(info["q"][1]).all())                   # the divergence reached the environment
    assert done.tolist() == [False, True] and info["terminated"].tolist() == [False, True]
    assert not bool(info[TIME_OUTS_KEY].any())
    assert float(reward[1]) == pytest.approx(-cfg.reward.w_drop * cfg.reward_scale + float(info["shaping"][1]))
    assert bool(torch.isfinite(obs).all()) and bool(torch.isfinite(env.privileged()).all())
    torch.testing.assert_close(obs[1], fresh[0], rtol=0.0, atol=0.0)          # a clean new episode at k = 0
    _, reward, done, info = env.step(_zeros(2))
    assert bool(torch.isfinite(reward).all()) and not bool(done.any()) and info["k"].tolist() == [3, 1]


def test_drop_on_the_last_frame_is_a_failure_not_a_time_out() -> None:
    env = make_env(env_config(n_worlds=1, warmup_duration=0.0), SCENE_XML, "cpu")

    def run(n_frames: int) -> tuple[int, dict]:
        env.reset(whisking_reference(n_frames=n_frames), None)
        for step in range(40):
            _, _, done, info = env.step(_zeros(1))
            if bool(done[0]):
                return step, info
        raise AssertionError("the unheld object never dropped")

    step, info = run(60)
    k_drop = int(info["k"][0])
    assert bool(info["terminated"][0]) and not bool(info[TIME_OUTS_KEY][0]) and k_drop < 59
    same_step, info = run(k_drop + 1)                     # the clip now ends on the frame the object drops
    assert same_step == step and int(info["k"][0]) == k_drop
    assert bool(info["terminated"][0]) and not bool(info[TIME_OUTS_KEY][0])


def test_step_needs_every_world_started_and_the_clip_registry_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(env_config(n_worlds=2, warmup_duration=0.0), SCENE_XML, "cpu")
    with pytest.raises(RuntimeError, match="reset"):
        env.step(_zeros(2))
    ref = whisking_reference(n_frames=12)
    env.reset(ref, torch.tensor([0]))
    with pytest.raises(RuntimeError, match="reset"):
        env.step(_zeros(2))                               # world 1 has no episode yet

    calls: list[PolicyReference] = []
    validate = env._bank.validate

    def counting_validate(r: PolicyReference) -> None:
        calls.append(r)
        validate(r)

    monkeypatch.setattr(env._bank, "validate", counting_validate)
    for _ in range(3):
        env.reset(ref, None)
        env.step(_zeros(2))
    assert calls == []                                    # validated once, when first registered
    for _ in range(4):
        env.reset(whisking_reference(n_frames=12), None)  # fresh objects every reset (e.g. a curriculum)
    assert len(env._clips) == 1 and len(calls) == 4
    library = [whisking_reference(n_frames=12, start=s) for s in (0, 10, 20)]
    env.set_references(library)
    env.reset(None, None)
    assert len(env._clips) == 3 and len(calls) == 7
