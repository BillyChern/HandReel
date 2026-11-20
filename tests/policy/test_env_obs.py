"""Observation content and per-world reference packing on the CPU twin.

Every check recomputes the quantity from the simulator state and from `env.reference(world)` alone, one world at a
time with the frame and reward functions, so a packing error in `fastdo.policy.bank` (wrong stride, wrong clamp) or a
mis-assembled observation field disagrees with it.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import fastdo.policy.env_obs as env_obs_module
from fastdo.policy.bank import ReferenceBank
from fastdo.policy.canonical import canonical_frame
from fastdo.policy.env import BatchedTrackingEnv, make_env
from fastdo.policy.env_obs import observation_from_state, observation_state, shape_descriptor
from fastdo.policy.protocol import TIME_OUTS_KEY
from fastdo.policy.reference import future_window
from fastdo.policy.rewards import REWARD_TERMS, StepObservation, step_reward, terminated
from fastdo.policy.scene import object_pose, object_twist, rotation_6d, wrist_pose, wrist_twist
from tests.policy.env_fixtures import (
    SCENE_XML,
    env_config,
    moved_reference,
    scene_missing,
    whisking_reference,
)

pytestmark = scene_missing
H = 10
ATOL = 2e-5


def _expected_fields(env: BatchedTrackingEnv, world: int, k: int, prev_x: torch.Tensor, force: torch.Tensor,
                     delta: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The observation and privileged features of `world` recomputed from its state and its own reference, and F."""
    qpos, qvel = (x[world:world + 1] for x in env._read_state())
    q, qd = qpos[:, :28], qvel[:, :28]
    T_obj, tw_obj = object_pose(qpos[:, 28:35]), object_twist(qpos[:, 28:35], qvel[:, 28:34])
    T_w, tw_w = wrist_pose(q[:, :6]), wrist_twist(q[:, :6], qd[:, :6])
    F = canonical_frame(T_obj, T_w[:, :3, 3], env._gravity, prev_x[world:world + 1])
    Rt = F[:, :3, :3].transpose(1, 2)

    def rot(v: torch.Tensor) -> torch.Tensor:
        return (Rt @ v[..., None])[..., 0]

    ref = env.reference(world)
    T, kk, W = ref.num_frames, torch.tensor([k]), env.cfg.warmup_ctrl_steps
    obj_window, dq = future_window(ref, kk, T_obj, q, H)
    wrist_ref = dataclasses.replace(ref, obj_pose=wrist_pose(ref.q_ref[:, :6].double()).float())
    wrist_window, _ = future_window(wrist_ref, kk, T_w, q, H)
    frames = (kk + torch.arange(H + 1)).clamp_max(T - 1)
    obj_twist_f = torch.cat([rot(tw_obj[:, :3]), rot(tw_obj[:, 3:])], dim=1)
    fields = {
        "wrist_pose": torch.cat([rot(T_w[:, :3, 3] - T_obj[:, :3, 3]), rotation_6d(Rt @ T_w[:, :3, :3])], dim=1),
        "wrist_twist": torch.cat([rot(tw_w[:, :3]), rot(tw_w[:, 3:])], dim=1),
        "finger_q": q[:, 6:], "finger_qd": qd[:, 6:], "obj_rot": rotation_6d(Rt @ T_obj[:, :3, :3]),
        "obj_twist": obj_twist_f, "obj_window": obj_window.reshape(1, -1), "wrist_window": wrist_window.reshape(1, -1),
        "finger_window": dq[:, :, 6:].reshape(1, -1), "contact_plan": ref.contact_plan[frames].float().reshape(1, -1),
        "shape": torch.cat([torch.as_tensor([*env.scene.obj_extents, env.scene.obj_mass]).float()[None],
                            ref.contact_plan.float().mean(0)[None]], dim=1),
        "prev_action": delta[world:world + 1],
        "phase": torch.tensor([[max(W - int(env._t[world]), 0) / max(W, 1), k / (T - 1)]]),
        "privileged": torch.cat([torch.log1p(force[world:world + 1]), obj_twist_f], dim=1),
    }
    return fields, F[0]


def test_observation_fields_match_direct_computation() -> None:
    """Three worlds on different clips (one yawed 0.9 rad, one shorter than the window, one in contact) are stepped
    with random residuals; after every step each field of every live world equals its direct computation."""
    cfg = env_config(n_worlds=3, warmup_duration=0.04, warmup_open_hand=False)
    env = make_env(cfg, SCENE_XML, "cpu")
    clips = [moved_reference(whisking_reference(n_frames=40, start=50), 0.9, (0.2, -0.1, 0.03)),
             whisking_reference(n_frames=8, start=40), whisking_reference(n_frames=30, start=60)]
    for world, clip in enumerate(clips):
        env.reset(clip, torch.tensor([world]))
    layout = env.obs_layout()
    gen = torch.Generator().manual_seed(3)
    checked, max_speed, max_force, max_rotation = 0, 0.0, 0.0, 0.0
    for _ in range(6):
        action = 0.5 * torch.randn(3, 28, generator=gen)
        prev_x = env._prev_x.clone()
        obs, _, done, info = env.step(action)
        delta = cfg.delta_max * torch.tanh(action)
        for world in (~done).nonzero()[:, 0].tolist():
            want, F = _expected_fields(env, world, int(info["k"][world]), prev_x, info["contact_force"], delta)
            for name, value in want.items():
                got = env.privileged()[world:world + 1] if name == "privileged" else obs[world:world + 1, layout[name]]
                torch.testing.assert_close(got, value.float(), rtol=0.0, atol=ATOL, msg=f"{name}, world {world}")
            checked += 1
            max_speed = max(max_speed, float(want["obj_twist"].abs().max()), float(want["finger_qd"].abs().max()))
            max_force = max(max_force, float(info["contact_force"][world].max()))
            max_rotation = max(max_rotation, float((F[:3, :3] - torch.eye(3)).abs().max()))
    # The checks were informative: moving worlds, contact forces, frames F that rotate twists, windows past the end.
    assert checked >= 12 and max_speed > 0.05 and max_force > 1.0 and max_rotation > 0.1
    assert int(info["k"][1]) + H > clips[1].num_frames - 1


def test_each_world_is_scored_against_its_own_reference() -> None:
    """Heterogeneous clips and staggered k: per-world reward terms and terminations equal `step_reward` and
    `terminated` on that world's own reference; the short clip times out on its own last frame."""
    cfg = env_config(n_worlds=3, warmup_duration=0.04, warmup_open_hand=False)
    env = make_env(cfg, SCENE_XML, "cpu")
    clips = [whisking_reference(n_frames=30), whisking_reference(n_frames=5, start=40),
             moved_reference(whisking_reference(n_frames=30, start=20), -0.6, (0.1, 0.1, 0.0))]
    for world, clip in enumerate(clips):
        env.reset(clip, torch.tensor([world]))
    gen = torch.Generator().manual_seed(5)
    staggered, time_out_worlds, checked = False, set(), 0
    for step in range(8):
        if step == 1:
            env.reset(None, torch.tensor([2]))                     # world 2 restarts one step behind world 0
        prev = env._prev_delta.clone()
        refs = [env.reference(w) for w in range(3)]
        action = 0.5 * torch.randn(3, 28, generator=gen)
        _, _, done, info = env.step(action)
        qpos, qvel = env._read_state()
        staggered |= not bool(done.any()) and int(info["k"][0]) != int(info["k"][2])
        for world in range(3):
            k = int(info["k"][world])
            if bool(info[TIME_OUTS_KEY][world]):
                assert k == refs[world].num_frames - 1 and not bool(info["terminated"][world])
                time_out_worlds.add(world)
            if bool(done[world]):
                continue
            s = slice(world, world + 1)
            measured = StepObservation(
                q=qpos[s, :28], qd=qvel[s, :28], obj_pose=object_pose(qpos[s, 28:35]),
                obj_twist=object_twist(qpos[s, 28:35], qvel[s, 28:34]), contact_force=info["contact_force"][s],
                action=cfg.delta_max * torch.tanh(action[s]), prev_action=prev[s], k=torch.tensor([k]))
            _, terms = step_reward(cfg.reward, refs[world], measured)
            for name in (*REWARD_TERMS, "e_pos", "e_rot"):
                torch.testing.assert_close(info["reward_terms"][name][s], terms[name].float(), rtol=0.0, atol=1e-5,
                                           msg=f"{name}, world {world}, step {step}")
            assert bool(info["terminated"][world]) == bool(terminated(cfg.reward, refs[world], measured)[0])
            checked += 1
    assert time_out_worlds == {1} and checked >= 18 and staggered


def test_canonical_frame_carries_the_previous_x_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every observation passes the x axis of the previous observation's frame F to `canonical_frame` (its fallback
    when the object's first axis and the wrist direction are both near vertical, Section 3), and zero for a world that
    has just restarted."""
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def spy(obj_pose: torch.Tensor, wrist_pos: torch.Tensor, gravity: torch.Tensor,
            prev_x: torch.Tensor | None) -> torch.Tensor:
        assert prev_x is not None
        F = canonical_frame(obj_pose, wrist_pos, gravity, prev_x)
        calls.append((prev_x.clone(), F[:, :3, 0].clone()))
        return F

    monkeypatch.setattr(env_obs_module, "canonical_frame", spy)
    env = make_env(env_config(n_worlds=2, warmup_duration=0.04, warmup_open_hand=False), SCENE_XML, "cpu")
    env.reset(whisking_reference(n_frames=30), None)
    for _ in range(4):
        _, _, done, _ = env.step(torch.full((2, 28), 0.5))
        assert not bool(done.any())
    env.reset(None, torch.tensor([1]))
    assert len(calls) == 6 and bool((calls[0][0] == 0).all())              # a fresh episode has no previous axis
    assert float(calls[4][1].norm(dim=1).min()) > 0.99
    for (prev_x, _), (_, x_before) in zip(calls[1:5], calls[:4], strict=True):
        torch.testing.assert_close(prev_x, x_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(calls[5][0][0], calls[4][1][0], rtol=0.0, atol=0.0)      # world 0 carries on
    assert bool((calls[5][0][1] == 0).all())                                  # world 1 restarted


def test_observation_from_state_reproduces_the_environment_bitwise() -> None:
    """Over 50 steps of three worlds on different clips (one yawed and shifted, one short enough to
    time out and restart inside the window), every observation and privileged feature the CPU twin returns equals
    `observation_from_state` on logged inputs, bit for bit: the simulator state, the frame from `info["k"]`, a step
    count, a `ReferenceBank` written with the same clips, the carried canonical x-axis, the squashed residual and the
    contact forces (all zeroed for a restarted world, as the environment does)."""
    cfg = env_config(n_worlds=3, warmup_duration=0.1, warmup_open_hand=False)
    env = make_env(cfg, SCENE_XML, "cpu")
    clips = [moved_reference(whisking_reference(n_frames=70, start=50), 0.9, (0.2, -0.1, 0.03)),
             whisking_reference(n_frames=12, start=40), whisking_reference(n_frames=70, start=60)]
    cpu, n = torch.device("cpu"), 3
    bank = ReferenceBank(n, 28, 6, cfg.hz, cpu)
    shape, gravity = shape_descriptor(env.scene, cpu), torch.as_tensor(env.scene.gravity, dtype=torch.float32)
    k, t = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    prev_x, prev_action, force = torch.zeros(n, 3), torch.zeros(n, 28), torch.zeros(n, 6)

    def rebuild() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal prev_x
        qpos, qvel = env._read_state()
        state = observation_state(env.scene, bank, qpos, qvel, k, t, force, cfg.horizon, cfg.warmup_ctrl_steps,
                                  gravity, shape)
        window = state.reference.obj.contact_plan.reshape(n, cfg.horizon + 1, 6)
        obs, priv, prev_x = observation_from_state(state, prev_x, prev_action, window)
        return obs, priv

    for world, clip in enumerate(clips):                      # every reset refreshes all worlds' observations
        obs = env.reset(clip, torch.tensor([world]))
        bank.write(world, clip)
        prev_x[world] = 0.0
        want, _ = rebuild()
    assert torch.equal(obs, want)
    gen = torch.Generator().manual_seed(7)
    restarts = 0
    for step in range(50):
        action = 0.3 * torch.randn(n, 28, generator=gen)
        obs, _, done, info = env.step(action)
        live = ~done
        k = torch.where(live, info["k"], torch.zeros_like(k))
        t = torch.where(live, t + 1, torch.zeros_like(t))
        prev_action = torch.where(live[:, None], cfg.delta_max * torch.tanh(action), torch.zeros_like(prev_action))
        force = torch.where(live[:, None], info["contact_force"], torch.zeros_like(force))
        prev_x = torch.where(live[:, None], prev_x, torch.zeros_like(prev_x))
        want, want_priv = rebuild()
        assert torch.equal(obs, want) and torch.equal(env.privileged(), want_priv), step
        restarts += int(done.sum())
    assert restarts >= 1
