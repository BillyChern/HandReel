"""The MuJoCo Warp environment against its CPU twin, its contact decode, its overflow bits and its GPU memory
(skipped without CUDA or the `policy` extra).

Run with the extra (`uv sync --extra dev --extra policy`). The first run compiles the MuJoCo Warp kernels (minutes on
a busy card; cached under the Warp kernel cache afterwards). At most 64 worlds, per the shared-host rule.
"""

from __future__ import annotations

import dataclasses
import gc
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.policy.env import MujocoCpuEnv, make_env
from fastdo.policy.protocol import TIME_OUTS_KEY
from fastdo.policy.reference import PolicyReference
from fastdo.policy.scene import load_policy_scene
from tests.policy.env_fixtures import SCENE_XML, env_config, scene_missing, whisking_reference

pytestmark = [scene_missing, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]
pytest.importorskip("mujoco_warp", reason="the `policy` extra (mujoco-warp, warp-lang) is not installed")

N_WORLDS = 64
GIB = 2**30
TOL = 1e-3
FLAGS = (TIME_OUTS_KEY, "terminated", "warmup", "k", "overflow")


def _gaps(info_w: dict, info_c: dict, rew_w: torch.Tensor, rew_c: torch.Tensor, rows: slice) -> tuple[float, ...]:
    q = float((info_w["q"][rows].cpu() - info_c["q"]).abs().max())
    pose = float((info_w["obj_pose"][rows].cpu() - info_c["obj_pose"]).abs().max())
    return q, pose, float((rew_w[rows].cpu() - rew_c).abs().max())


def _shifted(ref: PolicyReference, shift: tuple[float, float, float]) -> PolicyReference:
    """`ref` translated by `shift` (object poses and the wrist slides; the joint angles are unchanged)."""
    offset = torch.tensor(shift, dtype=torch.float64)
    q = ref.q_ref[:, :28].clone()
    q[:, :3] += offset.to(q.dtype)
    pose = ref.obj_pose.clone()
    pose[:, :3, 3] += offset.to(pose.dtype)
    return dataclasses.replace(ref, q_ref=q, obj_pose=pose)


def test_warp_env_matches_cpu_env() -> None:
    """64 identical Warp worlds vs the CPU twin on one scripted rollout that crosses the weld release (10 welded
    steps), the drop failure, the auto-reset and the next welded warm-up. The hand starts on frame 0 here: the open-hand
    start is shared code (`BatchedTrackingEnv._start_episodes`), and hand-object contact is where the backends are
    known to drift apart (report concern 1), which is not what this test is about."""
    from fastdo.policy.env_warp import WarpEnv

    ref = whisking_reference()                                   # built on the CPU; the env moves it to CUDA
    cfg_w = env_config(n_worlds=N_WORLDS, warmup_duration=0.2, warmup_open_hand=False)
    cfg_c = env_config(n_worlds=1, warmup_duration=0.2, warmup_open_hand=False)
    warp_env, cpu_env = make_env(cfg_w, SCENE_XML, "cuda"), make_env(cfg_c, SCENE_XML, "cpu")
    assert isinstance(warp_env, WarpEnv)
    obs_w, obs_c = warp_env.reset(ref, None), cpu_env.reset(ref, None)
    assert obs_w.shape == (N_WORLDS, cpu_env.obs_dim) and obs_w.device.type == "cuda"
    world_ref = warp_env.reference(3)
    assert world_ref.q_ref.device.type == "cuda" and world_ref.contact_plan.device.type == "cuda"
    torch.testing.assert_close(obs_w[0].cpu(), obs_c[0], rtol=0.0, atol=1e-4)

    actions = 0.5 * torch.randn(30, cpu_env.act_dim, generator=torch.Generator().manual_seed(0))
    worst, phases = [0.0, 0.0, 0.0], []
    for t in range(actions.shape[0]):
        _, rew_w, done_w, info_w = warp_env.step(actions[t].expand(N_WORLDS, -1).cuda())
        _, rew_c, done_c, info_c = cpu_env.step(actions[t][None])
        assert not bool((info_w[TIME_OUTS_KEY] & ~done_w).any())
        worst = [max(a, b) for a, b in zip(worst, _gaps(info_w, info_c, rew_w, rew_c, slice(0, 1)), strict=True)]
        assert bool(done_w[0].cpu() == done_c[0])
        for key in FLAGS:
            assert torch.equal(info_w[key][:1].cpu(), info_c[key]), (t, key)
        phases.append((bool(info_c["warmup"][0]), bool(done_c[0])))
    assert max(worst) < TOL, worst
    first_done = [done for _, done in phases].index(True)                 # raises if the object never dropped
    assert phases.index((False, False)) == 10                             # the weld is released at step 10
    assert any(warm for warm, _ in phases[first_done + 1:])               # welded again after the auto-reset
    # Every world got the same reference and actions, so the batch agrees with its world 0.
    torch.testing.assert_close(info_w["q"], info_w["q"][:1].expand_as(info_w["q"]), rtol=0.0, atol=1e-4)


def test_warp_env_staggered_worlds_match_cpu_env() -> None:
    """4 worlds per backend, both built from one scene object (the CPU twin keeps its model changes private): two clip
    lengths whose first object poses are 6.4 cm apart, a partial reset during the other worlds' warm-up, per-world
    actions; every world's state, reward and episode flags agree step by step through time-outs, drops and auto-resets.

    The short clip is a pure translation, so the per-world weld targets differ while the joint-space motion is the
    same. Yawed 0.5 rad as well (wrist Euler angles recomputed), its random residuals bring the hand into contact with
    the falling object, where the backends differ by up to 1.9e-3 in object rotation (report concern 1)."""
    from fastdo.policy.env_warp import WarpEnv

    cfg = env_config(n_worlds=4, warmup_duration=0.1, warmup_open_hand=False)
    scene = load_policy_scene(SCENE_XML, cfg.hz, cfg.n_substeps, cfg.kp, cfg.kd, cfg.wrist_kp, cfg.wrist_kd)
    cpu_env = MujocoCpuEnv(cfg, scene)
    warp_env = WarpEnv(cfg, scene, torch.device("cuda"))
    full, short = whisking_reference(), _shifted(whisking_reference(n_frames=6), (0.05, -0.04, 0.0))
    assert float((full.obj_pose[0, :3, 3] - short.obj_pose[0, :3, 3]).norm()) > 0.05       # weld targets differ
    for env in (warp_env, cpu_env):
        env.reset(full, torch.tensor([0, 1]))
        env.reset(short, torch.tensor([2, 3]))
    actions = 0.5 * torch.randn(30, 4, 28, generator=torch.Generator().manual_seed(1))
    worst, events = [0.0, 0.0, 0.0], {"time_out": 0, "failure": 0}
    for t in range(actions.shape[0]):
        if t == 3:
            obs_w, obs_c = (env.reset(None, torch.tensor([1, 3])) for env in (warp_env, cpu_env))
            torch.testing.assert_close(obs_w.cpu(), obs_c, rtol=0.0, atol=5e-3)
        _, rew_w, done_w, info_w = warp_env.step(actions[t].cuda())
        _, rew_c, done_c, info_c = cpu_env.step(actions[t])
        worst = [max(a, b) for a, b in zip(worst, _gaps(info_w, info_c, rew_w, rew_c, slice(None)), strict=True)]
        assert torch.equal(done_w.cpu(), done_c), t
        for key in FLAGS:
            assert torch.equal(info_w[key].cpu(), info_c[key]), (t, key)
        events["time_out"] += int(info_c[TIME_OUTS_KEY].sum())
        events["failure"] += int((done_c & ~info_c[TIME_OUTS_KEY]).sum())
    assert max(worst) < TOL, worst
    assert events["time_out"] >= 2 and events["failure"] >= 2, events


def test_warp_contact_decode_matches_mujoco_warp_contact_force() -> None:
    """Per-(world, label) normal forces decoded from Data.contact/efc equal mujoco_warp.contact_force sums at
    contact-rich frames (the hand penetrates the object from frame 60) after the contact count has shrunk, so the slots
    past nacon hold stale contacts; copies of live hand-object contacts planted there do not change the decode."""
    import mujoco_warp as mjw
    import warp

    from fastdo.policy.env_warp import WarpEnv

    wp: Any = warp
    env = make_env(env_config(n_worlds=4, warmup_duration=0.0), SCENE_XML, "cuda")
    assert isinstance(env, WarpEnv)
    data, scene = env._data, env.scene
    for world, start in enumerate((60, 90, 90, 75)):                        # every world in contact first
        env.reset(whisking_reference(n_frames=20, start=start), torch.tensor([world]))
    env.step(torch.zeros(4, 28, device="cuda"))
    nacon_rich = int(data.nacon.numpy()[0])
    env.reset(whisking_reference(n_frames=20, start=0), torch.tensor([2]))     # world 2 leaves contact
    env.step(torch.zeros(4, 28, device="cuda"))
    decoded = env._backend_contact_forces().double().cpu().numpy()
    nacon = int(data.nacon.numpy()[0])
    assert nacon < nacon_rich, (nacon, nacon_rich)                          # stale slots past nacon exist
    ids = wp.array(np.arange(nacon, dtype=np.int32), dtype=int, device=env._wp_device)
    force = wp.zeros(nacon, dtype=wp.spatial_vector, device=env._wp_device)
    mjw.contact_force(env._model, data, ids, False, force)
    normal = force.numpy()[:, 0]
    geom, world_of = data.contact.geom.numpy()[:nacon], data.contact.worldid.numpy()[:nacon]
    efc_address = data.contact.efc_address.numpy()[:nacon, 0]
    expected, live = np.zeros((4, 6)), []
    for i in range(nacon):
        g1, g2 = int(geom[i, 0]), int(geom[i, 1])
        if efc_address[i] < 0 or scene.geom_is_object[g1] == scene.geom_is_object[g2]:
            continue
        label = int(scene.geom_label[g2] if scene.geom_is_object[g1] else scene.geom_label[g1])
        if label >= 0:
            expected[int(world_of[i]), label] += normal[i]
            live.append(i)
    totals = expected.sum(axis=1)
    assert totals[2] == 0.0 and bool((totals[[0, 1, 3]] > 1.0).all()), totals
    np.testing.assert_allclose(decoded, expected, rtol=1e-5, atol=1e-2)
    n_plant = min(len(live), int(data.naconmax) - nacon)
    assert n_plant > 0, (len(live), data.naconmax, nacon)
    src, dst = torch.as_tensor(live[:n_plant], device="cuda"), torch.arange(nacon, nacon + n_plant, device="cuda")
    for array in (data.contact.geom, data.contact.dim, data.contact.efc_address, data.contact.worldid):
        view = wp.to_torch(array)
        view[dst] = view[src]
    np.testing.assert_allclose(env._backend_contact_forces().double().cpu().numpy(), decoded, rtol=1e-6, atol=1e-6)


def test_warp_overflow_fails_the_world_and_clears_on_restart() -> None:
    """mujoco_warp's per-world overflow bits (sticky until a reset) fail that world like a non-finite state; the
    restart clears them and the new episode runs on."""
    import warp as wp

    from fastdo.policy.env_warp import WarpEnv

    cfg = env_config(n_worlds=2, warmup_duration=0.0)
    env = make_env(cfg, SCENE_XML, "cuda")
    assert isinstance(env, WarpEnv)
    fresh = env.reset(whisking_reference(n_frames=30), None).clone()
    env.step(torch.zeros(2, 28, device="cuda"))
    overflow = wp.to_torch(env._data.overflow)
    overflow[1] = 1                                    # OverflowType.NEFC: world 1 dropped constraint rows
    obs, reward, done, info = env.step(torch.zeros(2, 28, device="cuda"))
    assert info["overflow"].tolist() == [False, True] and done.tolist() == [False, True]
    assert info["terminated"].tolist() == [False, True] and not bool(info[TIME_OUTS_KEY].any())
    assert float(reward[1]) == pytest.approx(-cfg.reward.w_drop * cfg.reward_scale + float(info["shaping"][1]))
    torch.testing.assert_close(obs[1], fresh[1], rtol=0.0, atol=1e-6)
    assert int(overflow[1]) == 0
    _, _, done, info = env.step(torch.zeros(2, 28, device="cuda"))
    assert not bool(info["overflow"].any()) and not bool(done.any())


def test_warp_env_memory_under_budget() -> None:
    import warp as wp

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    env = make_env(env_config(n_worlds=N_WORLDS), SCENE_XML, "cuda")
    env.reset(whisking_reference(), None)
    for _ in range(10):
        env.step(torch.zeros(N_WORLDS, env.act_dim, device="cuda"))
    warp_peak = int(wp.get_mempool_used_mem_high("cuda:0")) if wp.is_mempool_enabled("cuda:0") else 0
    torch_peak = int(torch.cuda.max_memory_allocated())
    assert warp_peak > 0 and warp_peak + torch_peak < 4 * GIB, (warp_peak / GIB, torch_peak / GIB)
