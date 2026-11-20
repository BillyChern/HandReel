"""Batched tracking environment, plain-MuJoCo CPU twin (design Sections 3-7; the Warp twin is test_env_warp)."""

from __future__ import annotations

import dataclasses
import itertools
import math

import mujoco
import numpy as np
import pytest
import torch

from fastdo.policy.env import EnvConfig, MujocoCpuEnv, make_env, obs_layout
from fastdo.policy.protocol import TIME_OUTS_KEY, TrackingEnv
from fastdo.policy.reference import PolicyReference
from fastdo.policy.rewards import REWARD_TERMS, pose_errors
from tests.policy.env_fixtures import (
    SCENE_XML,
    env_config,
    moved_reference,
    scene_missing,
    whisking_reference,
    with_object_block,
)

pytestmark = scene_missing

EXPECTED_WIDTHS = {"wrist_pose": 9, "wrist_twist": 6, "finger_q": 22, "finger_qd": 22, "obj_rot": 6,
                   "obj_twist": 6, "obj_window": 60, "wrist_window": 60, "finger_window": 220,
                   "contact_plan": 66, "shape": 10, "prev_action": 28, "phase": 2}


def _zeros(env: TrackingEnv) -> torch.Tensor:
    return torch.zeros(env.n_worlds, env.act_dim)


def test_env_config_validates_and_documents_reward_scale() -> None:
    cfg = env_config()
    assert (cfg.hz, cfg.n_substeps, cfg.horizon, cfg.delta_max, cfg.warmup_duration) == (50.0, 10, 10, 0.3, 3.0)
    # The baseline's warm-up lasts 3.0 s (its 600 steps at sim_dt 5 ms): 150 control steps, 1500 steps at 2 ms.
    assert (cfg.warmup_ctrl_steps, cfg.warmup_sim_steps) == (150, 1500)
    assert cfg.warmup_sim_steps / (cfg.hz * cfg.n_substeps) == pytest.approx(3.0)
    assert cfg.warmup_open_hand and cfg.warmup_clearance == 0.1 and cfg.warmup_reward_scale == 0.1
    # The default scale keeps a clip's discounted return (gamma 0.99) at most ~4, inside PPO's O(1-10) band.
    best_step = cfg.reward_scale * (cfg.reward.w_pose + cfg.reward.w_joint + cfg.reward.w_contact)
    assert cfg.reward_scale == pytest.approx(1.0 / cfg.hz) and best_step / (1.0 - 0.99) <= 10.0
    for bad in ({"warmup_duration": 0.03}, {"warmup_duration": -0.02}, {"warmup_clearance": -0.1}, {"n_worlds": 0},
                {"reward_scale": 0.0}, {"horizon": 0}, {"warmup_reward_scale": -0.1}):
        with pytest.raises(ValueError):
            env_config(**bad)
    with pytest.raises(ValueError, match="n_worlds"):
        make_env(env_config(n_worlds=9), SCENE_XML, "cpu")


def test_cpu_env_obs_layout_and_shapes() -> None:
    cfg = env_config(n_worlds=3)
    env = make_env(cfg, SCENE_XML, "cpu")
    assert isinstance(env, MujocoCpuEnv) and isinstance(env, TrackingEnv)
    layout = env.obs_layout()
    assert layout == obs_layout(cfg.horizon)
    assert {name: s.stop - s.start for name, s in layout.items()} == EXPECTED_WIDTHS
    spans = sorted((s.start, s.stop) for s in layout.values())
    assert spans[0][0] == 0 and all(a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert env.obs_dim == sum(s.stop - s.start for s in layout.values()) == spans[-1][1]
    assert (env.act_dim, env.priv_dim, env.n_worlds) == (28, 12, 3)

    ref = whisking_reference()
    obs = env.reset(ref, None)
    assert obs.shape == (3, env.obs_dim) and obs.dtype == torch.float32 and bool(torch.isfinite(obs).all())
    action = torch.full((3, 28), 0.5)
    obs, reward, done, info = env.step(action)
    assert obs.shape == (3, env.obs_dim) and reward.shape == (3,) and done.shape == (3,)
    assert done.dtype == torch.bool and info[TIME_OUTS_KEY].shape == (3,)
    assert info["obj_pose"].shape == (3, 4, 4) and info["contact_force"].shape == (3, 6)
    assert info["k"].shape == (3,) and not info["k"].dtype.is_floating_point and info["q"].shape == (3, 28)
    assert env.privileged().shape == (3, env.priv_dim)
    # Slice semantics: the squashed residual of the last action, the plan window k..k+H, the remaining warm-up.
    torch.testing.assert_close(obs[:, layout["prev_action"]], torch.full((3, 28), 0.3 * math.tanh(0.5)))
    plan = obs[:, layout["contact_plan"]].reshape(3, cfg.horizon + 1, 6)
    torch.testing.assert_close(plan[0], ref.contact_plan[: cfg.horizon + 1].float())
    W = cfg.warmup_ctrl_steps
    torch.testing.assert_close(obs[:, layout["phase"]], torch.tensor([[(W - 1) / W, 0.0]]).expand(3, 2))


def _warmup_rollout(cfg: EnvConfig, ref: PolicyReference) -> tuple[torch.Tensor, list[dict], torch.Tensor]:
    """Reset two worlds on `ref` and run the whole warm-up with zero residual: (start q, per-step info, rewards)."""
    env = make_env(cfg, SCENE_XML, "cpu")
    obs = env.reset(ref, None)
    torch.testing.assert_close(obs[:, env.obs_layout()["phase"]], torch.tensor([[1.0, 0.0]]).expand(2, 2))
    start = env._read_state()[0][:, :28]
    infos, rewards = [], []
    for _ in range(cfg.warmup_ctrl_steps):
        _, reward, done, info = env.step(_zeros(env))
        assert not bool(done.any()) and bool(info["warmup"].all()) and bool((info["k"] == 0).all())
        assert set(info["reward_terms"]) == set(REWARD_TERMS) | {"e_pos", "e_rot"}
        infos.append(info)
        rewards.append(reward)
    for step in range(5):                           # release: weld off, gravity back, k advances
        _, reward, _, info = env.step(_zeros(env))
        assert not bool(info["warmup"].any()) and bool((info["k"] == step + 1).all())
        terms = info["reward_terms"]
        torch.testing.assert_close(reward, cfg.reward_scale * sum(terms[name] for name in REWARD_TERMS)
                                   + info["shaping"])
    infos.append(info)
    return start, infos, torch.stack(rewards)


def test_cpu_env_tracks_reference_under_zero_residual_with_weld() -> None:
    """Hand held on frame 0 (`warmup_open_hand=False`): the welded object stays on obj_pose[0] within 1 mm,
    and far tighter, because gravity is off in the welded world; then it falls once released."""
    cfg = env_config(n_worlds=2, warmup_open_hand=False)
    ref = whisking_reference()
    start, infos, rewards = _warmup_rollout(cfg, ref)
    torch.testing.assert_close(start, ref.q_ref[0, :28].float().expand(2, 28), rtol=0.0, atol=1e-6)
    target = ref.obj_pose[0].to(torch.float32).expand(2, 4, 4)
    errors = torch.stack([torch.stack(pose_errors(info["obj_pose"], target)) for info in infos[:-1]])
    assert float(errors.max()) < 1e-3                                       # the 1e-3 bound, position and angle
    assert float(errors[:, 0].max()) < 1e-5, float(errors[:, 0].max())     # gravity left on would sag ~1 mm
    # Warm-up steps count with the baseline's warm-up reward multiplier (its do_as_i_do config: 0.1).
    for info, reward in zip(infos[:-1], rewards, strict=True):
        scale = cfg.reward_scale * cfg.warmup_reward_scale
        torch.testing.assert_close(reward, scale * sum(info["reward_terms"][name] for name in REWARD_TERMS)
                                   + info["shaping"])
    assert float((infos[-2]["q"][:, :3] - ref.q_ref[0, :3].float()).abs().max()) < 2e-3
    assert float(target[0, 2, 3]) - float(infos[-1]["obj_pose"][:, 2, 3].max()) > 0.01


def test_cpu_env_baseline_warmup_starts_open_and_reaches_frame_zero() -> None:
    """The default (baseline) warm-up: 3.0 s from the open, backed-off hand, joint reference driven linearly to frame
    0. Closing onto frame 0 the distal phalanges touch the object (~100 N near step 120) and push the soft baseline
    weld (solref 0.01) by up to ~1 cm, which it restores before the release."""
    cfg = env_config(n_worlds=2)
    ref = whisking_reference()
    W, q_first = cfg.warmup_ctrl_steps, ref.q_ref[0, :28].float()
    start, infos, _ = _warmup_rollout(cfg, ref)
    backoff = float((start[0, :3] - q_first[:3]).norm())
    assert float(start[:, 6:].abs().max()) == 0.0 and backoff > 0.1                  # fingers open, wrist backed off
    torch.testing.assert_close(start[:, 3:6], q_first[3:6].expand(2, 3), rtol=0.0, atol=1e-6)
    wrist_err = []
    for j, info in enumerate(infos[:-1]):
        # The joint term tracks the interpolated reference start + (j + 1)/W (q_ref[0] - start).
        warm_ref = start + (j + 1) / W * (q_first - start)
        joint = cfg.reward.w_joint * torch.exp(-((info["q"] - warm_ref) ** 2).sum(-1) / cfg.reward.sigma_joint**2)
        torch.testing.assert_close(info["reward_terms"]["joint"], joint, rtol=0.0, atol=1e-5)
        wrist_err.append(float((info["q"][:, :3] - q_first[:3]).norm(dim=1).max()))
        e_pos, e_rot = pose_errors(info["obj_pose"], ref.obj_pose[0].float().expand(2, 4, 4))
        assert float(e_pos.max()) < 0.02 and float(e_rot.max()) < 0.02                 # measured max 1.02e-2
    assert float(e_pos.max()) < 1e-3 and float(e_rot.max()) < 1e-3                     # restored at the release
    # The hand follows the ramp and reaches frame 0 at the end (PD lag of a few mm; hand self-contacts on the fingers).
    assert 0.3 * backoff < wrist_err[W // 2] < 0.7 * backoff and wrist_err[-1] < 1e-2, (backoff, wrist_err[-5:])
    assert float((infos[-2]["q"][:, 6:] - q_first[6:]).abs().max()) < 0.15


def test_cpu_env_reset_randomizes_only_when_enabled() -> None:
    ref = whisking_reference()
    env = make_env(env_config(randomize=False), SCENE_XML, "cpu")
    env.reset(ref, None)
    first = env.reference(0)
    env.reset(ref, None)
    assert torch.equal(first.obj_pose, env.reference(0).obj_pose)
    assert torch.equal(first.obj_pose, env.reference(1).obj_pose)
    torch.testing.assert_close(first.obj_pose, ref.obj_pose.float())

    def draws(seed: int) -> list[torch.Tensor]:
        env_r = make_env(env_config(randomize=True, seed=seed), SCENE_XML, "cpu")
        out = []
        for _ in range(2):
            env_r.reset(ref, None)
            out += [env_r.reference(0).obj_pose, env_r.reference(1).obj_pose]
            assert torch.equal(env_r.reference(0).q_ref, first.q_ref)
            assert torch.equal(env_r.reference(0).contact_plan, first.contact_plan)
        return out

    a, b, c = draws(7), draws(7), draws(8)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)                                          # same seed, same sequence
    gaps = [float((x - y).abs().max()) for i, x in enumerate(a) for y in a[i + 1:]]
    assert min(gaps) > 1e-4 and float((a[0] - c[0]).abs().max()) > 1e-4   # every draw and seed differs
    shift = (a[0][:, :3, 3] - ref.obj_pose[:, :3, 3].float()).norm(dim=-1)
    assert 1e-4 < float(shift.mean()) < 1e-2                              # ~ the 1 mm posterior std


def test_time_outs_at_clip_end_are_a_subset_of_done() -> None:
    cfg = env_config(n_worlds=2, warmup_duration=0.04, warmup_open_hand=False)   # 2 welded control steps
    env = make_env(cfg, SCENE_XML, "cpu")
    ref = whisking_reference(n_frames=6)                          # reaches its last frame 5 steps after release
    env.reset(ref, None)
    events = []
    for _ in range(10):
        _, _, done, info = env.step(_zeros(env))
        time_outs = info[TIME_OUTS_KEY]
        assert time_outs.dtype == torch.bool and not bool((time_outs & ~done).any())
        events.append((bool(done.all()), bool(time_outs.all()), bool(done.any()), bool(info["warmup"].all()),
                       int(info["k"].max())))
    assert [e[2] for e in events] == [False] * 6 + [True] + [False] * 3
    assert events[6][:2] == (True, True) and events[6][4] == ref.num_frames - 1
    # Auto-reset: the steps after the time-out belong to a new episode, welded again at k = 0.
    assert [e[3:] for e in events[5:]] == [(False, 4), (False, 5), (True, 0), (True, 0), (False, 1)]


def test_auto_reset_keeps_the_final_state_in_info() -> None:
    """On a contact-rich clip end, info holds the forces of the final state while the new episode starts clean."""
    env = make_env(env_config(n_worlds=2, warmup_duration=0.0), SCENE_XML, "cpu")
    env.reset(whisking_reference(n_frames=4, start=60), None)
    for _ in range(3):
        _, _, done, info = env.step(_zeros(env))
    assert bool(done.all()) and bool(info[TIME_OUTS_KEY].all())
    assert float(info["contact_force"].sum(dim=1).min()) > 1.0 and bool((info["k"] == 3).all())
    torch.testing.assert_close(env.privileged()[:, :6], torch.zeros(2, 6), rtol=0.0, atol=0.0)


def test_failure_is_done_without_time_out() -> None:
    env = make_env(env_config(n_worlds=2, warmup_duration=0.0), SCENE_XML, "cpu")
    env.reset(whisking_reference(n_frames=60), None)
    for step in range(40):
        _, reward, done, info = env.step(_zeros(env))
        if bool(done.any()):
            break
    assert 3 <= step < 20 and bool(done.all())                    # the unheld object drops 0.1 m in ~0.14 s
    assert not bool(info[TIME_OUTS_KEY].any()) and bool(info["terminated"].all())
    assert bool(torch.isfinite(reward).all())


def test_reset_none_none_uses_installed_references() -> None:
    ref = whisking_reference()
    env = make_env(env_config(n_worlds=4), SCENE_XML, "cpu")
    with pytest.raises(RuntimeError, match="reference"):
        env.reset(None, None)
    obs = env.reset(ref, None)
    torch.testing.assert_close(env.reset(None, None), obs, rtol=0.0, atol=0.0)
    env.step(torch.full((4, 28), 0.3))
    moved = env.reset(None, torch.tensor([2]))                    # partial reset: only world 2 restarts
    torch.testing.assert_close(moved[2], obs[2], rtol=0.0, atol=0.0)
    assert float((moved[0] - obs[0]).abs().max()) > 1e-3

    short = whisking_reference(n_frames=40)
    lib_env = make_env(env_config(n_worlds=8), SCENE_XML, "cpu", references=[ref, short])
    lib_env.reset(None, None)
    lengths = {lib_env.reference(i).num_frames for i in range(8)}
    assert lengths <= {ref.num_frames, short.num_frames} and len(lengths) == 2


def test_object_state_comes_from_obj_pose_not_q_ref() -> None:
    ref = whisking_reference()
    env = make_env(env_config(n_worlds=1), SCENE_XML, "cpu")
    robot_only = dataclasses.replace(ref, q_ref=ref.q_ref[:, :28])
    obs = [env.reset(r, None) for r in (ref, with_object_block(ref, 1e3), with_object_block(ref, math.nan),
                                         robot_only)]
    assert all(torch.equal(o, obs[0]) for o in obs[1:]) and bool(torch.isfinite(obs[0]).all())
    assert env.reference(0).q_ref.shape == (ref.num_frames, 28)
    _, _, _, info = env.step(_zeros(env))
    torch.testing.assert_close(info["obj_pose"][0], ref.obj_pose[0].float(), rtol=0.0, atol=1e-5)
    with pytest.raises(ValueError, match="hz"):
        env.reset(dataclasses.replace(ref, hz=30.0), None)


def test_observation_invariant_under_gravity_fixing_group() -> None:
    """Rotating a clip about gravity and translating it leaves the reset observation unchanged (Section 3)."""
    ref = whisking_reference(n_frames=30, covariance=False)
    moved = moved_reference(ref, 0.7, (0.3, -0.2, 0.05))
    env = make_env(env_config(n_worlds=1), SCENE_XML, "cpu")
    obs_ref, obs_moved = env.reset(ref, None), env.reset(moved, None)
    torch.testing.assert_close(obs_moved, obs_ref, rtol=0.0, atol=2e-4)


def test_label_contact_forces_match_mj_contact_force() -> None:
    from fastdo.policy.scene import load_policy_scene, mjdata_label_forces

    cfg = env_config()
    scene = load_policy_scene(SCENE_XML, cfg.hz, cfg.n_substeps, cfg.kp, cfg.kd, cfg.wrist_kp, cfg.wrist_kd)
    with np.load(SCENE_XML.parent / "trajectory_kinematic.npz") as data:
        qpos = data["qpos"]
    datas = [mujoco.MjData(scene.model) for _ in range(2)]
    for d, frame in zip(datas, (60, 90), strict=True):          # both penetrate the object (C4 finding)
        d.qpos[:] = qpos[frame]
        d.ctrl[:] = qpos[frame, :28]
        mujoco.mj_step(scene.model, d)
    expected = np.zeros((2, 6))
    force = np.zeros(6)
    for w, d in enumerate(datas):
        for i in range(d.ncon):
            g1, g2 = (int(g) for g in d.contact.geom[i])
            labels = [int(scene.geom_label[g]) for g in (g1, g2) if not scene.geom_is_object[g]]
            if scene.geom_is_object[g1] != scene.geom_is_object[g2] and labels[0] >= 0:
                mujoco.mj_contactForce(scene.model, d, i, force)
                expected[w, labels[0]] += force[0]
    got = mjdata_label_forces(scene, datas)
    assert got.shape == (2, 6) and float(expected.sum()) > 1.0
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_scene_kinematics_match_mujoco() -> None:
    from fastdo.policy.scene import (
        load_policy_scene,
        object_pose,
        object_twist,
        wrist_pose,
        wrist_twist,
    )

    cfg = env_config()
    scene = load_policy_scene(SCENE_XML, cfg.hz, cfg.n_substeps, cfg.kp, cfg.kd, cfg.wrist_kp, cfg.wrist_kd)
    m, d = scene.model, mujoco.MjData(scene.model)
    with np.load(SCENE_XML.parent / "trajectory_kinematic.npz") as data:
        d.qpos[:] = data["qpos"][40]
    d.qvel[:] = np.random.default_rng(0).normal(size=m.nv)
    mujoco.mj_forward(m, d)
    qpos, qvel = torch.as_tensor(d.qpos)[None], torch.as_tensor(d.qvel)[None]
    vel = np.zeros(6)
    for body, pose, twist in (
        (scene.wrist_body, wrist_pose(qpos[:, :6]), wrist_twist(qpos[:, :6], qvel[:, :6])),
        (scene.obj_body, object_pose(qpos[:, 28:35]), object_twist(qpos[:, 28:35], qvel[:, 28:34])),
    ):
        np.testing.assert_allclose(pose[0, :3, 3].numpy(), d.xpos[body], atol=1e-12)
        np.testing.assert_allclose(pose[0, :3, :3].numpy(), d.xmat[body].reshape(3, 3), atol=1e-12)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)   # [omega, v at the body COM]
        v_origin = vel[3:] - np.cross(vel[:3], d.xipos[body] - d.xpos[body])
        np.testing.assert_allclose(twist[0].numpy(), np.concatenate([v_origin, vel[:3]]), atol=1e-10)


def test_scene_requires_gravity_along_minus_z() -> None:
    from fastdo.policy.scene import check_gravity

    body = "<worldbody><body><freejoint/><geom size='0.1'/></body></worldbody>"
    check_gravity(mujoco.MjModel.from_xml_string(f"<mujoco><option gravity='0 0 -9.81'/>{body}</mujoco>"))
    for gravity in ("1 0 -9.81", "0 0 0", "0 0 9.81"):
        with pytest.raises(ValueError, match="gravity"):
            check_gravity(mujoco.MjModel.from_xml_string(f"<mujoco><option gravity='{gravity}'/>{body}</mujoco>"))


def test_env_config_is_frozen() -> None:
    cfg = EnvConfig(n_worlds=1, kp=1.0, kd=0.0, randomize=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_worlds = 2  # type: ignore[misc]
