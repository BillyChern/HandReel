"""The engine's retargeting stage on the tracking policy (`fastdo.policy.stage.retarget_stage_policy`)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch

from fastdo.policy.checkpoint import save_checkpoint
from fastdo.policy.networks import ActorCritic
from fastdo.policy.rewards import clip_success
from fastdo.policy.scene import object_pose
from fastdo.policy.smc import SmcConfig
from fastdo.policy.stage import (
    RETARGET_FILE,
    ClipRejected,
    make_policy_stage,
    retarget_stage_policy,
)
from tests.policy.env_fixtures import SCENE_XML, env_config, scene_missing, whisking_reference

if TYPE_CHECKING:
    from fastdo.engine.orchestrator import StageFn


@scene_missing
def test_retarget_stage_policy_returns_the_selected_trajectory(tmp_path: Path) -> None:
    """Three particles on the CPU twin (W = 5) with a freshly initialized policy (near-zero residuals) from a
    checkpoint: a 6-frame clip comes back as a `KinematicTrajectory` of 6 frames at 50 Hz whose object poses pass the
    baseline criterion against the reference and whose velocities are the central differences of its positions; a
    60-frame clip, where the unheld object drops in every particle, is rejected. As the engine's `StageFn`, the same
    stage (same seed) writes that trajectory to `<artifact_dir>/retarget.npz`."""
    from fastdo.policy.env import make_env

    env = make_env(env_config(n_worlds=3, warmup_duration=0.1), SCENE_XML, "cpu")
    torch.manual_seed(0)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=(16,))
    checkpoint = save_checkpoint(tmp_path / "update_000001.pt", model, None, {"ppo": {"gamma": 0.99}})
    cfg = SmcConfig(n_particles=3, chunk_seconds=0.04)
    short = whisking_reference(n_frames=6)
    record = cast(Any, SimpleNamespace(clip_id="whisking/0"))
    traj = retarget_stage_policy(record, checkpoint, reference_fn=lambda _: short, env=env, smc_cfg=cfg,
                                 generator=torch.Generator().manual_seed(0))
    assert traj.qpos.shape == (6, 35) and traj.qvel.shape == (6, 34) and traj.frequency == 50.0
    assert bool(np.isfinite(traj.qpos).all()) and bool(np.isfinite(traj.qvel).all())
    poses = object_pose(torch.as_tensor(traj.qpos[:, 28:35])).numpy()
    assert clip_success(poses, short.obj_pose.numpy(), 0)[0]
    np.testing.assert_allclose(traj.qvel[1:-1, :31], (traj.qpos[2:, :31] - traj.qpos[:-2, :31]) * 25.0, atol=1e-9)
    stage: StageFn = make_policy_stage(checkpoint, record_fn=lambda clip_id, _: record, reference_fn=lambda _: short,
                                       env=env, smc_cfg=cfg, seed=0)
    stage("whisking/0", tmp_path / "clips" / "whisking-0")
    with np.load(tmp_path / "clips" / "whisking-0" / RETARGET_FILE) as saved:
        np.testing.assert_array_equal(saved["qpos"], traj.qpos)
        np.testing.assert_array_equal(saved["qvel"], traj.qvel)
        assert float(saved["frequency"]) == 50.0
    with pytest.raises(ClipRejected, match="whisking/0"):
        retarget_stage_policy(record, checkpoint, reference_fn=lambda _: whisking_reference(n_frames=60), env=env,
                              smc_cfg=cfg, generator=torch.Generator().manual_seed(1))


@scene_missing
def test_retarget_stage_policy_without_warm_up_starts_on_the_reset_state(tmp_path: Path) -> None:
    """W = 0 on the CPU twin with two particles and reference noise: frame 0 of the trajectory is the actual
    reset state of the selected particle, one of the two world states `world_kinematics` reported after the particle
    draws (not the reference's frame 0, which the draw moves), and frames 1..T-1 are the particle path's steps 0..T-2
    (the same particle run again from the same seed). Without a draw, `world_kinematics` right after a reset reads the
    reference's frame 0 (joints and object pose), which ties the hook to the simulator state."""
    from fastdo.policy.env import make_env
    from fastdo.policy.scene import object_qpos
    from fastdo.policy.smc import run_smc

    env = make_env(env_config(n_worlds=2, warmup_duration=0.0), SCENE_XML, "cpu")
    torch.manual_seed(0)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim, hidden=(16,))
    checkpoint = save_checkpoint(tmp_path / "update_000001.pt", model, None, {"ppo": {"gamma": 0.99}})
    cfg, ref = SmcConfig(n_particles=2, chunk_seconds=0.04), whisking_reference(n_frames=4)
    reads: list[np.ndarray] = []
    original = env.world_kinematics

    def spy() -> tuple[torch.Tensor, torch.Tensor]:
        q, pose = original()
        reads.append(np.concatenate([q.double().numpy(), object_qpos(pose.double().numpy())], axis=1))
        return q, pose

    env.world_kinematics = spy  # type: ignore[method-assign]
    record = cast(Any, SimpleNamespace(clip_id="whisking/0"))
    traj = retarget_stage_policy(record, checkpoint, reference_fn=lambda _: ref, env=env, smc_cfg=cfg,
                                 generator=torch.Generator().manual_seed(0))
    assert traj.qpos.shape == (4, 35) and len(reads) == 1 and not np.allclose(reads[0][0], reads[0][1])
    assert min(float(np.abs(traj.qpos[0] - row).max()) for row in reads[0]) < 1e-9
    assert float(np.abs(traj.qpos[0, 28:31] - ref.obj_pose[0, :3, 3].numpy()).max()) > 1e-6
    rerun = run_smc(env, model, ref, cfg, torch.Generator().manual_seed(0))
    np.testing.assert_array_equal(traj.qpos[1:], rerun.qpos[:3])
    np.testing.assert_array_equal(traj.qpos[0], rerun.start_qpos)
    env.reset(ref, None)                     # no particle draw: `world_kinematics` reads frame 0 of the reference
    q0, pose0 = original()
    torch.testing.assert_close(q0.double(), ref.q_ref[:1, :28].expand(2, -1), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(pose0.double(), ref.obj_pose[:1].expand(2, -1, -1), rtol=0.0, atol=1e-5)


def test_retarget_stage_policy_refuses_w0_without_a_reset_state(tmp_path: Path) -> None:
    """W = 0 on an environment that cannot report its reset state (no `world_kinematics`): the stage refuses before
    loading anything instead of substituting the reference's frame 0."""
    env = cast(Any, SimpleNamespace(cfg=SimpleNamespace(warmup_ctrl_steps=0), n_worlds=2, device=torch.device("cpu")))
    record = cast(Any, SimpleNamespace(clip_id="c0"))
    with pytest.raises(ValueError, match="reset state"):
        retarget_stage_policy(record, tmp_path / "absent.pt", reference_fn=cast(Any, lambda _: None), env=env)


def test_retarget_stage_policy_rejects_a_declined_reference(tmp_path: Path) -> None:
    """A reference function may decline a clip (the DexYCB function returns None when the clip's scene or hands
    failed validation); the stage rejects that clip with `ClipRejected` and never runs SMC on a missing reference."""
    checkpoint = save_checkpoint(tmp_path / "update_000001.pt", ActorCritic(4, 2, 28, hidden=(4,)), None,
                                 {"ppo": {"gamma": 0.99}})
    env = cast(Any, SimpleNamespace(cfg=SimpleNamespace(warmup_ctrl_steps=5), n_worlds=2, device=torch.device("cpu")))
    record = cast(Any, SimpleNamespace(clip_id="20200820-subject-03/20200820_135250/836212060125"))
    with pytest.raises(ClipRejected, match="declined"):
        retarget_stage_policy(record, checkpoint, reference_fn=lambda _: None, env=env)
