"""What the CPU twin commands and simulates per world.

Per-world warm-up weld targets and gravity, the release on the step that ends a world's warm-up, the frame and clip
of the PD target, buffer overflows as failures, and the caller's `MjModel` left untouched.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from fastdo.policy.env import MujocoCpuEnv, make_env
from fastdo.policy.protocol import TIME_OUTS_KEY
from fastdo.policy.rewards import pose_errors
from fastdo.policy.scene import load_policy_scene
from tests.policy.env_fixtures import (
    SCENE_XML,
    env_config,
    moved_reference,
    scene_missing,
    whisking_reference,
)

pytestmark = scene_missing

HOLD = 1e-5          # welded hold with gravity off (measured ~1e-13 m); gravity on in a welded world sags ~1 mm
AUTORESET = int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)


def test_each_world_keeps_its_own_weld_target_and_gravity_through_a_staggered_release() -> None:
    """Two clips 6 cm and 0.5 rad apart start in one reset call and two worlds restart mid-warm-up. Every welded object
    stays on its own obj_pose[0]. A world's weld and gravity switch on the step that ends its warm-up, so its first free
    step falls 10 substeps of h from rest (55 g h^2), while the worlds still welded do not move."""
    cfg = env_config(n_worlds=4, warmup_duration=0.1, warmup_open_hand=False)                 # W = 5
    env = make_env(cfg, SCENE_XML, "cpu")
    assert isinstance(env, MujocoCpuEnv)
    full, moved = whisking_reference(), moved_reference(whisking_reference(), 0.5, (0.05, -0.04, 0.0))
    assert float((full.obj_pose[0, :3, 3] - moved.obj_pose[0, :3, 3]).norm()) > 0.05
    env.set_references([full, moved])
    s, W, h = env.scene, cfg.warmup_ctrl_steps, float(env.scene.model.opt.timestep)
    fall = 0.5 * cfg.n_substeps * (cfg.n_substeps + 1) * float(np.linalg.norm(s.gravity)) * h**2

    def targets() -> list[torch.Tensor]:
        return [env.reference(w).obj_pose[0] for w in range(4)]

    env.reset(None, None)
    target = targets()
    on_moved = [bool(torch.equal(tgt, moved.obj_pose[0].float())) for tgt in target]
    assert 0 < sum(on_moved) < 4, on_moved                           # both clips in the one reset call
    t = [0] * 4
    first_free: set[int] = set()
    for step in range(9):
        if step == 2:
            env.reset(None, torch.tensor([1, 2]))                   # these two restart their warm-up
            target, t[1], t[2] = targets(), 0, 0
        z_before = env._read_state()[0][:, s.obj_qpos + 2].clone()
        _, _, done, info = env.step(torch.zeros(4, 28))
        assert not bool(done.any())
        for w in range(4):
            t[w] += 1
            assert bool(info["warmup"][w]) == (t[w] <= W)
            if t[w] <= W:
                e_pos, e_rot = pose_errors(info["obj_pose"][w:w + 1].double(), target[w][None].double())
                assert float(e_pos) < HOLD and float(e_rot) < 1e-4, (step, w, float(e_pos), float(e_rot))
            released = t[w] >= W                                     # switched right after step W, not later
            assert bool(env._datas[w].eq_active[s.weld_eq]) != released, (step, w)
            np.testing.assert_array_equal(env._world_gravity[w], s.gravity if released else np.zeros(3))
            if t[w] == W + 1:
                drop = float(z_before[w] - info["obj_pose"][w, 2, 3])
                assert drop == pytest.approx(fall, rel=0.02), (step, w, drop, fall)
                first_free.add(w)
    assert first_free == {0, 1, 2, 3}


def test_pd_target_is_the_reference_at_the_frame_the_step_moves_to(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Worlds at different k (one restarted mid-run, one on a clip that times out and restarts): the ctrl a step applies
    is clip(q_ref[k'] + delta_max tanh(a)) with k' = min(k + 1, T - 1), the frame the step moves to, and on warm-up
    steps clip(start + (t + 1)/W (q_ref[0] - start) + delta) from the open-hand start."""
    monkeypatch.chdir(tmp_path)                           # a diverging world makes MuJoCo write MUJOCO_LOG.TXT
    cfg = env_config(n_worlds=3, warmup_duration=0.06)                                         # W = 3
    env = make_env(cfg, SCENE_XML, "cpu")
    assert isinstance(env, MujocoCpuEnv)
    env.reset(whisking_reference(n_frames=40), torch.tensor([0, 1]))
    env.reset(whisking_reference(n_frames=6, start=10), torch.tensor([2]))
    applied: list[torch.Tensor] = []
    backend_step = env._backend_step

    def spy(ctrl: torch.Tensor) -> None:
        applied.append(ctrl.clone())
        backend_step(ctrl)

    monkeypatch.setattr(env, "_backend_step", spy)
    s, W = env.scene, cfg.warmup_ctrl_steps
    low, high = (torch.as_tensor(x, dtype=torch.float32) for x in (s.ctrl_low, s.ctrl_high))
    start, refs = env._read_state()[0][:, :28].clone(), [env.reference(w) for w in range(3)]
    t, k = [0] * 3, [0] * 3
    scale = torch.cat([torch.full((6,), 0.3), torch.full((22,), 3.0)])     # saturating finger residuals
    gen = torch.Generator().manual_seed(11)
    lag = clipped = ramp = 0.0
    mixed = staggered = False
    for step in range(12):
        if step == 2:
            env.reset(None, torch.tensor([1]))
            start[1], t[1], k[1], refs[1] = env._read_state()[0][1, :28], 0, 0, env.reference(1)
        action = scale * torch.randn(3, 28, generator=gen)
        _, _, done, info = env.step(action)
        mixed |= bool(info["warmup"].any()) and not bool(info["warmup"].all())
        for w in range(3):
            q_ref = refs[w].q_ref
            if t[w] < W:
                base = start[w] + (t[w] + 1) / W * (q_ref[0] - start[w])
                ramp = max(ramp, float((q_ref[0] - start[w]).abs().max()))
            else:
                k[w] = min(k[w] + 1, refs[w].num_frames - 1)
                base = q_ref[k[w]]
                lag = max(lag, float((q_ref[k[w]] - q_ref[k[w] - 1]).abs().max()))
            t[w] += 1
            assert int(info["k"][w]) == k[w], (step, w)
            raw = base + cfg.delta_max * torch.tanh(action[w])
            want = torch.minimum(torch.maximum(raw, low), high)
            clipped = max(clipped, float((raw - want).abs().max()))
            torch.testing.assert_close(applied[-1][w], want, rtol=0.0, atol=2e-6, msg=f"step {step}, world {w}")
            if bool(done[w]):                             # the auto-reset rewrote ctrl: follow the new episode
                start[w], t[w], k[w], refs[w] = env._read_state()[0][w, :28], 0, 0, env.reference(w)
            else:
                np.testing.assert_allclose(env._datas[w].ctrl, want.double().numpy(), rtol=0.0, atol=2e-6)
        staggered |= len(set(k)) == 3
    # Informative: consecutive frames differ, the clip bound, the warm-up ramp moves, k and warm-up differ per world.
    assert lag > 1e-3 and clipped > 1e-2 and ramp > 0.1 and mixed and staggered, (lag, clipped, ramp, mixed, staggered)


def test_buffer_overflow_is_a_failure_and_clears_on_restart() -> None:
    """A world whose contact/constraint buffers overflowed (the CPU arena's full warning; MuJoCo Warp's overflow bits)
    fails like a non-finite state even though its state is finite, and restarts with the flag cleared."""
    cfg = env_config(n_worlds=2, warmup_duration=0.0)
    env = make_env(cfg, SCENE_XML, "cpu")
    assert isinstance(env, MujocoCpuEnv)
    ref = whisking_reference(n_frames=30)
    fresh = make_env(env_config(n_worlds=1, warmup_duration=0.0), SCENE_XML, "cpu").reset(ref, None)
    env.reset(ref, None)
    env.step(torch.zeros(2, 28))
    env._datas[1].warning[int(mujoco.mjtWarning.mjWARN_CNSTRFULL)].number = 1
    obs, reward, done, info = env.step(torch.zeros(2, 28))
    assert info["overflow"].tolist() == [False, True] and done.tolist() == [False, True]
    assert info["terminated"].tolist() == [False, True] and not bool(info[TIME_OUTS_KEY].any())
    assert bool(torch.isfinite(info["q"]).all()) and bool(torch.isfinite(reward).all())
    assert float(reward[1]) == pytest.approx(-cfg.reward.w_drop * cfg.reward_scale + float(info["shaping"][1]))
    assert float(reward[0]) > -1e-3
    torch.testing.assert_close(obs[1], fresh[0], rtol=0.0, atol=0.0)
    _, _, done, info = env.step(torch.zeros(2, 28))
    assert not bool(info["overflow"].any()) and not bool(done.any())


def test_cpu_env_leaves_the_callers_scene_model_untouched() -> None:
    """The CPU twin disables MuJoCo's auto-reset and writes per-world gravity and weld rows on its own model copy, so a
    `WarpEnv` can still be built from the same scene (`put_model` rejects the auto-reset flag)."""
    cfg = env_config(n_worlds=2, warmup_duration=0.04, warmup_open_hand=False)
    scene = load_policy_scene(SCENE_XML, cfg.hz, cfg.n_substeps, cfg.kp, cfg.kd, cfg.wrist_kp, cfg.wrist_kd)
    flags, eq_data = int(scene.model.opt.disableflags), scene.model.eq_data.copy()
    gravity = scene.model.opt.gravity.copy()
    env = MujocoCpuEnv(cfg, scene)
    env.reset(moved_reference(whisking_reference(n_frames=10), 0.5, (0.05, -0.04, 0.0)), None)
    for _ in range(3):
        env.step(torch.zeros(2, 28))
    assert not flags & AUTORESET and int(scene.model.opt.disableflags) == flags
    assert env.scene.model is not scene.model and int(env.scene.model.opt.disableflags) & AUTORESET
    assert not np.array_equal(env.scene.model.eq_data, eq_data)             # the copy did carry per-world rows
    np.testing.assert_array_equal(scene.model.eq_data, eq_data)
    np.testing.assert_array_equal(scene.model.opt.gravity, gravity)
