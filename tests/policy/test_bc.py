"""Behaviour-cloning warm start from the baseline MPC's executed rollouts, on the environment's control grid."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastdo.policy.bc import (
    BcConfig,
    MpcRollout,
    bc_dataset,
    bc_pretrain,
    load_mpc_rollout,
    load_mpc_states,
    mpc_reference_frequency,
    mpc_steps_per_frame,
)
from fastdo.policy.env_config import obs_layout
from fastdo.policy.networks import ActorCritic
from fastdo.policy.reference import resample_reference
from fastdo.retarget.kinematic import KinematicTrajectory
from tests.policy.env_fixtures import SCENE_XML, TRAJECTORY_NPZ, scene_missing

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("theirs", "ours_clearance")
NA, N_LABELS, OBS_DIM = 28, 6, 517
SIM_DT, REF_DT, N_SRC, W_MPC, W_ENV, TAIL = 0.005, 1.0 / 30.0, 20, 6, 3, 9


def _outputs_roots() -> list[Path]:
    """`outputs/` of this checkout and, from a git worktree, of the main checkout (the artifacts are untracked)."""
    roots = [REPO_ROOT / "outputs"]
    git = REPO_ROOT / ".git"
    if git.is_file():                        # worktree: "gitdir: <main>/.git/worktrees/<name>"
        gitdir = Path(git.read_text().split("gitdir:", 1)[1].strip())
        roots.append(gitdir.parents[1].parent / "outputs")
    return roots


def _arm_dir(arm: str) -> Path | None:
    for root in _outputs_roots():
        d = root / "isolation" / arm / "sharpa" / "right" / "whisking_c4" / "0"
        if (d / "trajectory_mjwp.npz").exists():
            return d
    return None


def _write_chunks(path: Path, n_chunks: int, n_ctrl: int, dt: float, file_order: list[int],
                  sim_steps: list[int] | None = None, time_offset: float = 0.0) -> None:
    """A baseline-layout npz: chunk c covers simulator steps c n_ctrl .. (c+1) n_ctrl - 1; `qpos[..., 0]` is the
    time after each step, `ctrl[..., 0]` the step index and `qvel[..., 0]` minus it; chunks are written in `file_order`."""
    steps = np.arange(n_chunks * n_ctrl).reshape(n_chunks, n_ctrl)
    time = (steps + 1) * dt + time_offset
    qpos = np.zeros((n_chunks, n_ctrl, 5))
    qpos[..., 0], qpos[..., 1] = time, steps
    ctrl = np.zeros((n_chunks, n_ctrl, 3))
    ctrl[..., 0] = steps
    qvel = np.zeros((n_chunks, n_ctrl, 4))
    qvel[..., 0] = -steps
    sim_step = np.asarray(sim_steps if sim_steps is not None else n_ctrl * np.arange(1, n_chunks + 1))
    o = np.asarray(file_order)
    np.savez(path, qpos=qpos[o], qvel=qvel[o], ctrl=ctrl[o], time=time[o], sim_step=sim_step[o],
             opt_steps=np.ones((n_chunks, 1), dtype=np.int64))


def test_load_mpc_rollout_concatenates_chunks_in_order(tmp_path: Path) -> None:
    path = tmp_path / "trajectory_mjwp.npz"
    _write_chunks(path, n_chunks=3, n_ctrl=4, dt=0.005, file_order=[2, 0, 1])
    with np.load(path) as raw:
        assert not np.all(np.diff(raw["sim_step"]) > 0)        # the file order really is shuffled
    q, ctrl, dt = load_mpc_rollout(path)
    assert q.shape == (12, 5) and ctrl.shape == (12, 3)
    assert dt == pytest.approx(0.005, abs=1e-12)
    assert np.all(np.diff(q[:, 0]) > 0)                         # monotone time
    np.testing.assert_array_equal(ctrl[:, 0], np.arange(12))    # row j is simulator step j
    np.testing.assert_array_equal(q[:, 1], np.arange(12))
    states = load_mpc_states(path)
    assert states.qvel.shape == (12, 4) and states.dt == dt
    np.testing.assert_array_equal(states.qvel[:, 0], -np.arange(12))   # qvel follows the same chunk order


@pytest.mark.parametrize(("sim_steps", "time_offset", "match"), [
    ([4, 12, 8], 0.0, None),              # valid control case, shuffled
    ([4, 8, 8], 0.0, "duplicate"),
    ([4, 8, 16], 0.0, "contiguous"),      # a missing chunk
    ([8, 12, 16], 0.0, "contiguous"),     # does not start at simulator step 0
    ([4, 8, 12], 0.5, "time"),            # time inconsistent with sim_step
])
def test_load_mpc_rollout_validates_chunk_tiling(tmp_path: Path, sim_steps: list[int], time_offset: float,
                                                 match: str | None) -> None:
    path = tmp_path / "trajectory_mjwp.npz"
    order = [0, 2, 1] if match is None else [0, 1, 2]
    _write_chunks(path, 3, 4, 0.005, order, sim_steps=sorted(sim_steps) if match is None else sim_steps,
                  time_offset=time_offset)
    if match is None:
        assert load_mpc_rollout(path)[0].shape == (12, 5)
        return
    with pytest.raises(ValueError, match=match):
        load_mpc_rollout(path)


def test_load_mpc_rollout_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, qpos=np.zeros((1, 4, 5)), qvel=np.zeros((1, 4, 4)), time=np.zeros((1, 4)), sim_step=np.array([4]))
    with pytest.raises(ValueError, match="ctrl"):
        load_mpc_rollout(path)


@pytest.mark.parametrize(("n_frames", "ref_dt"), [(17, 0.0333), (135, 0.0333), (9, 0.005), (12, 0.02)])
def test_mpc_reference_grid_matches_baseline_alignment(n_frames: int, ref_dt: float) -> None:
    """`resample_reference` at `mpc_reference_frequency` reproduces the baseline's `align_to_sim_dt` grid
    (`round(ref_dt / sim_dt)`-fold `F.interpolate(linear, align_corners=True)`) sample for sample."""
    sim_dt = 0.005
    gen = torch.Generator().manual_seed(n_frames)
    qpos = torch.randn(n_frames, 35, generator=gen, dtype=torch.float64)
    factor = max(1, round(ref_dt / sim_dt))
    expected = qpos if factor == 1 else F.interpolate(qpos.T[None], size=factor * n_frames, mode="linear",
                                                      align_corners=True)[0].T
    kin = KinematicTrajectory(qpos=qpos.numpy(), qvel=np.zeros((n_frames, 34)),
                              frequency=mpc_reference_frequency(n_frames, ref_dt, sim_dt))
    ref = resample_reference(kin, torch.eye(4, dtype=torch.float64).repeat(n_frames, 1, 1),
                             torch.zeros(n_frames, N_LABELS, dtype=torch.bool), None, hz=1.0 / sim_dt)
    assert ref.num_frames == expected.shape[0]
    torch.testing.assert_close(ref.q_ref, expected, atol=1e-12, rtol=0.0)


def test_mpc_reference_frequency_rejects_unsupported() -> None:
    with pytest.raises(ValueError):
        mpc_reference_frequency(1, 0.0333, 0.005)
    with pytest.raises(ValueError):
        mpc_reference_frequency(10, 0.002, 0.005)           # the baseline subsamples; not a supported grid


def _mpc_setup() -> dict[str, Any]:
    """A synthetic MPC rollout on the whisking scene: the first 20 shipped kinematic frames (50 Hz) are the source, the
    MPC's nominal is the baseline grid (7-fold, `mpc_reference_frequency`), states follow the nominal after W warm-up
    rows, and the environment's reference is the source at 50 Hz with a contact plan from frame 8. Commands outside
    the post-warm-up nominal are poisoned (1e3)."""
    from fastdo.policy.scene import load_policy_scene, object_pose

    scene = load_policy_scene(SCENE_XML, 50.0, 10, 300.0, 35.0, 1000.0, 100.0)
    with np.load(TRAJECTORY_NPZ) as data:
        qpos, qvel, freq = data["qpos"][:N_SRC], data["qvel"][:N_SRC], float(data["frequency"])
    pose = object_pose(torch.as_tensor(qpos[:, NA:NA + 7], dtype=torch.float64))
    plan = torch.zeros(N_SRC, N_LABELS, dtype=torch.bool)
    plan[8:, 1:3] = True
    mpc_kin = KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=mpc_reference_frequency(N_SRC, REF_DT, SIM_DT))
    mpc_ref = resample_reference(mpc_kin, pose, plan, None, hz=1.0 / SIM_DT)
    env_ref = resample_reference(KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=freq), pose, plan, None, hz=50.0)
    n_post = mpc_ref.num_frames
    rows = torch.cat([mpc_ref.q_ref[:1].expand(W_MPC, -1), mpc_ref.q_ref, mpc_ref.q_ref[-1:].expand(TAIL, -1)])
    nominal = torch.full((int(rows.shape[0]), NA), 1e3, dtype=torch.float64)
    nominal[W_MPC:W_MPC + n_post] = mpc_ref.q_ref[:, :NA]
    return {"scene": scene, "q": rows.numpy(), "qvel": np.zeros((int(rows.shape[0]), 34)), "nominal": nominal,
            "mpc_ref": mpc_ref, "env_ref": env_ref, "n_post": n_post,
            "spf": mpc_steps_per_frame(N_SRC, REF_DT, freq, 50.0, SIM_DT)}


def _dataset(s: dict[str, Any], ctrl: torch.Tensor, cfg: BcConfig) -> Any:
    rollout = MpcRollout(q=s["q"], qvel=s["qvel"], ctrl=ctrl.numpy(), dt=SIM_DT)
    return bc_dataset(s["scene"], rollout, s["mpc_ref"], s["env_ref"], s["spf"], cfg)


@scene_missing
def test_bc_dataset_zero_residual_reproduces_the_next_frame() -> None:
    """An MPC commanding exactly its nominal control gives zero actions, and the PD targets q_ref[k + 1] + delta
    reproduce the environment's reference q_ref[k + 1] within 1e-6 on every control period. At 7.316 MPC steps per
    50 Hz frame the 140-step nominal holds 19 periods of the 20-frame clip; warm-up and tail commands never enter.

    Time base: each target is also the MPC's own command at rollout row W + p_{k+1}, the reference instant of frame
    k + 1, within the error of re-interpolating the 7-fold grid at a source knot (at most 0.25 |slope jump| per grid
    step). The unshifted instant p_k, step-index resampling (row W + k + 1) and a 7-step period each break that bound
    on most entries."""
    s = _mpc_setup()
    assert s["n_post"] == 140 and s["spf"] == pytest.approx(139.0 / 19.0, rel=1e-12)
    cfg = BcConfig(env_warmup_steps=W_ENV, mpc_warmup_steps=W_MPC, clamp_ctrl=False)
    ds = _dataset(s, s["nominal"], cfg)
    K = N_SRC - 1
    assert ds.obs.shape == (K, OBS_DIM) and ds.action.shape == (K, NA) and torch.equal(ds.frames, torch.arange(K))
    assert torch.equal(ds.action, torch.zeros(K, NA)) and ds.clipped_fraction == 0.0
    torch.testing.assert_close(ds.target, s["env_ref"].q_ref[1:K + 1, :NA], rtol=0.0, atol=1e-6)

    command = s["nominal"][W_MPC:W_MPC + s["n_post"]].numpy()                 # rollout rows W .. W + n_post - 1

    def command_at(position: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(np.stack([np.interp(position.numpy(), np.arange(s["n_post"]), command[:, c])
                                         for c in range(NA)], axis=1))

    source = s["env_ref"].q_ref[:, :NA].to(torch.float64)                     # the 50 Hz source frames
    slope = (source[1:] - source[:-1]) / s["spf"]                             # per MPC step
    bound = torch.zeros_like(source)
    bound[1:-1] = 0.25 * (slope[1:] - slope[:-1]).abs()
    bound = bound[1:K + 1]                                                    # at knots k + 1
    frames = torch.arange(K + 1, dtype=torch.float64)
    gap = (ds.target - command_at(frames[1:] * s["spf"])).abs()
    assert bool((gap <= bound + 1e-9).all()), float((gap - bound).max())
    for name, position in (("unshifted", frames[:-1] * s["spf"]), ("step index", frames[1:]),
                           ("7-step period", frames[1:] * 7.0)):
        wrong = (ds.target - command_at(position)).abs()
        assert float((wrong > bound + 1e-9).double().mean()) > 0.5, name
    logger.info("zero-residual time base: |target - command(W + p_k+1)| max %.3g (bound max %.3g)", float(gap.max()),
                float(bound.max()))


@scene_missing
def test_bc_dataset_resamples_in_reference_time_and_shifts_one_period() -> None:
    """With a residual r(j) = 0.01 sin(j / 13) on every joint (j the post-warm-up simulator step): action k is r at
    the MPC step of frame k + 1, p_{k+1} = 7.316 (k + 1) (linear interpolation error < 1e-5), which differs from the
    unshifted r(p_k) and from step-index resampling r(k + 1) by more than 1e-3. The observation of pair k shows frame
    k (phase k / (T - 1)), the finger joints of the MPC state at row W - 1 + p_k, the contact plan from frame k, and
    the previous pair's action as its previous residual."""
    s = _mpc_setup()
    j = torch.arange(s["n_post"], dtype=torch.float64)
    ctrl = s["nominal"].clone()
    ctrl[W_MPC:W_MPC + s["n_post"]] += 0.01 * torch.sin(j / 13.0)[:, None]
    ds = _dataset(s, ctrl, BcConfig(env_warmup_steps=W_ENV, mpc_warmup_steps=W_MPC, clamp_ctrl=False))
    K, T = int(ds.action.shape[0]), s["env_ref"].num_frames
    p = torch.arange(K + 1, dtype=torch.float64) * s["spf"]
    expected = 0.01 * torch.sin(p[1:] / 13.0)
    torch.testing.assert_close(ds.action.double(), expected[:, None].expand(K, NA), rtol=0.0, atol=1e-5)
    assert float((expected - 0.01 * torch.sin(p[:-1] / 13.0)).abs().max()) > 1e-3
    assert float((expected - 0.01 * torch.sin(torch.arange(1, K + 1) / 13.0)).abs().max()) > 1e-3
    layout = obs_layout(10)
    phase = torch.stack([torch.zeros(K), torch.arange(K) / (T - 1)], dim=1).float()
    torch.testing.assert_close(ds.obs[:, layout["phase"]], phase)
    rows = (W_MPC - 1 + p[:-1]).numpy()
    fingers = np.stack([np.interp(rows, np.arange(s["q"].shape[0]), s["q"][:, c]) for c in range(6, NA)], axis=1)
    torch.testing.assert_close(ds.obs[:, layout["finger_q"]], torch.as_tensor(fingers).float(), rtol=0.0, atol=1e-6)
    frames = (torch.arange(K)[:, None] + torch.arange(11)).clamp_max(T - 1)
    plan = ds.obs[:, layout["contact_plan"]].reshape(K, 11, N_LABELS)
    torch.testing.assert_close(plan, s["env_ref"].contact_plan[frames].float())
    torch.testing.assert_close(ds.obs[1:, layout["prev_action"]], ds.action[:-1])
    assert float(ds.obs[0, layout["prev_action"]].abs().max()) < 1e-7            # r(p_0) = 0.01 sin(0)


@scene_missing
def test_bc_dataset_saturates_targets_and_clamps_ctrl_by_default() -> None:
    """A 0.5 rad residual on one finger joint (above delta_max 0.3) is clipped to 0.95 delta_max, 1/28 of the
    entries. ctrl is clamped to the actuator range by default, so a command 10 above a joint's upper limit
    reconstructs a PD target at the limit (within the nominal's interpolation error); without clamping it keeps the
    excess. A nominal that is not on the simulator grid is rejected."""
    s = _mpc_setup()
    assert BcConfig().clamp_ctrl and BcConfig().saturation == 0.95
    post = slice(W_MPC, W_MPC + s["n_post"])
    ctrl = s["nominal"].clone()
    ctrl[post, 10] += 0.5
    ds = _dataset(s, ctrl, BcConfig(env_warmup_steps=W_ENV, mpc_warmup_steps=W_MPC, clamp_ctrl=False))
    torch.testing.assert_close(ds.action[:, 10].double(), torch.full((N_SRC - 1,), 0.95 * 0.3, dtype=torch.float64))
    assert ds.clipped_fraction == pytest.approx(1.0 / NA)
    assert float(ds.action[:, [c for c in range(NA) if c != 10]].abs().max()) < 1e-6

    joint, high = 7, float(s["scene"].ctrl_high[7])
    ctrl = s["nominal"].clone()
    ctrl[post, joint] = high + 10.0
    wide = BcConfig(delta_max=100.0, saturation=1.0, env_warmup_steps=W_ENV, mpc_warmup_steps=W_MPC)
    clamped = _dataset(s, ctrl, wide)
    np.testing.assert_allclose(clamped.target[:, joint].numpy(), high, rtol=0.0, atol=5e-3)
    unclamped = _dataset(s, ctrl, dataclasses.replace(wide, clamp_ctrl=False))
    assert float((unclamped.target[:, joint] - high).min()) > 9.0
    rollout = MpcRollout(q=s["q"], qvel=s["qvel"], ctrl=ctrl.numpy(), dt=SIM_DT)
    with pytest.raises(ValueError, match="hz"):
        bc_dataset(s["scene"], rollout, s["env_ref"], s["env_ref"], s["spf"], wide)


@scene_missing
def test_bc_dataset_observes_velocities_on_the_environment_clock() -> None:
    """Observation time base: the MPC plays the reference c = steps_per_frame dt hz = 1.829 times slower than
    the environment here, so its qvel is rescaled by c. A finger joint moving as A sin(omega tau) in reference time,
    logged with its MPC-clock velocity A omega cos(omega tau) / c, shows the 50 Hz velocity A omega cos(omega k / 50) in
    `finger_qd` (linear-interpolation error < 5e-4); with random velocities the velocity channels (wrist twist, finger
    qd, object twist) are exactly c times those without rescaling, and every other channel is bitwise unchanged."""
    s = _mpc_setup()
    c = s["spf"] * SIM_DT * 50.0
    assert c == pytest.approx(139.0 / 19.0 * SIM_DT * 50.0, rel=1e-12)
    n_rows = int(s["q"].shape[0])
    tau = (np.arange(n_rows) - W_MPC + 1.0) / (s["spf"] * 50.0)       # row W - 1 + p: post-warm-up step p
    amplitude, omega, joint = 0.05, 2.0 * np.pi / 0.4, 10
    q, qvel = s["q"].copy(), 0.3 * np.random.default_rng(5).standard_normal((n_rows, 34))
    q[:, joint], qvel[:, joint] = amplitude * np.sin(omega * tau), amplitude * omega * np.cos(omega * tau) / c
    rollout = MpcRollout(q=q, qvel=qvel, ctrl=s["nominal"].numpy(), dt=SIM_DT)
    cfg = BcConfig(env_warmup_steps=W_ENV, mpc_warmup_steps=W_MPC, clamp_ctrl=False)
    on, off = (bc_dataset(s["scene"], rollout, s["mpc_ref"], s["env_ref"], s["spf"],
                          dataclasses.replace(cfg, rescale_velocity=flag)) for flag in (True, False))
    assert BcConfig().rescale_velocity and on.velocity_scale == pytest.approx(c) and off.velocity_scale == 1.0
    layout, K = obs_layout(10), int(on.obs.shape[0])
    finger_qd = on.obs[:, layout["finger_qd"]][:, joint - 6].double().numpy()
    np.testing.assert_allclose(finger_qd, amplitude * omega * np.cos(omega * np.arange(K) / 50.0), rtol=0.0, atol=5e-4)
    velocity = torch.zeros(OBS_DIM, dtype=torch.bool)
    for name in ("wrist_twist", "finger_qd", "obj_twist"):
        velocity[layout[name]] = True
    torch.testing.assert_close(on.obs[:, velocity], c * off.obs[:, velocity], rtol=1e-5, atol=1e-6)
    assert float((on.obs[:, velocity] - off.obs[:, velocity]).abs().max()) > 0.1
    assert torch.equal(on.obs[:, ~velocity], off.obs[:, ~velocity])


@pytest.mark.parametrize("delta_max", [None, 0.3])
def test_bc_reduces_action_error(delta_max: float | None) -> None:
    """200 (obs, action) pairs from a linear teacher (squashed like the environment's residual when `delta_max`
    is set): the actor's action error drops by more than 10x in 200 epochs; the critic and log-std are untouched."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    obs_dim, act_dim, n = 16, 4, 200
    obs = torch.randn(n, obs_dim, generator=gen)
    teacher = 0.5 * torch.randn(act_dim, obs_dim, generator=gen)
    raw = obs @ teacher.T
    action = raw if delta_max is None else delta_max * torch.tanh(raw)
    model = ActorCritic(obs_dim, 3, act_dim)
    critic_before = [p.detach().clone() for p in model.critic.parameters()]
    log_std_before = model.log_std.detach().clone()

    def error() -> float:
        with torch.no_grad():
            mean = model.mean_action(obs)
            pred = mean if delta_max is None else delta_max * torch.tanh(mean)
            return float(F.mse_loss(pred, action))

    before = error()
    stats = bc_pretrain(model, (obs, action), n_epochs=200, lr=1e-3, delta_max=delta_max)
    after = error()
    logger.info("BC linear teacher (delta_max %s): MSE %.3e -> %.3e (%.0fx), %d updates", delta_max, before,
                after, before / after, int(stats["n_updates"]))
    assert after < before / 10.0
    assert stats["mse_initial"] == pytest.approx(before, rel=1e-4)
    assert stats["mse_final"] == pytest.approx(after, rel=1e-4)
    assert stats["n_samples"] == n and stats["n_epochs"] == 200
    for p, p0 in zip(model.critic.parameters(), critic_before, strict=True):
        torch.testing.assert_close(p.detach(), p0, atol=0.0, rtol=0.0)
    torch.testing.assert_close(model.log_std.detach(), log_std_before, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("arm", ARMS)
def test_bc_on_c4_artifact_if_present(arm: str) -> None:
    """Both C4 isolation arms (read-only): the executed MPC rollout against the arm's kinematic reference at the
    environment's 50 Hz (the baseline played it on its 28.4 Hz grid), resampled in reference time and shifted one
    control period, observations from `observation_from_state`, targets saturated; one BC epoch."""
    import yaml

    from fastdo.policy.scene import load_policy_scene, object_pose

    arm_dir = _arm_dir(arm)
    if arm_dir is None:
        pytest.skip(f"C4 isolation artifact for {arm} not present")
    cfg = yaml.safe_load((arm_dir / "config.yaml").read_text())
    sim_dt, ref_dt, warmup = float(cfg["sim_dt"]), float(cfg["ref_dt"]), int(cfg["warmup_steps"])
    rollout = load_mpc_states(arm_dir / "trajectory_mjwp.npz")
    assert rollout.dt == pytest.approx(sim_dt, abs=1e-9) and warmup == 600
    assert rollout.q.shape[1] == NA + 7 and rollout.qvel.shape[1] == NA + 6 and rollout.ctrl.shape[1] == NA
    with np.load(arm_dir / "trajectory_kinematic.npz") as data:
        qpos, qvel, freq = data["qpos"], data["qvel"], float(data["frequency"])
    T = int(qpos.shape[0])
    pose = object_pose(torch.as_tensor(qpos[:, NA:NA + 7], dtype=torch.float64))
    plan = torch.zeros(T, N_LABELS, dtype=torch.bool)
    mpc_kin = KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=mpc_reference_frequency(T, ref_dt, sim_dt))
    mpc_ref = resample_reference(mpc_kin, pose, plan, None, hz=1.0 / sim_dt)
    env_ref = resample_reference(KinematicTrajectory(qpos=qpos, qvel=qvel, frequency=freq), pose, plan, None, hz=50.0)
    scene = load_policy_scene(arm_dir / "scene.xml", 50.0, 10, 300.0, 35.0, 1000.0, 100.0)
    spf = mpc_steps_per_frame(T, ref_dt, freq, 50.0, sim_dt)
    ds = bc_dataset(scene, rollout, mpc_ref, env_ref, spf, BcConfig(mpc_warmup_steps=warmup))
    assert ds.velocity_scale == pytest.approx(spf * sim_dt * 50.0) and 1.7 < ds.velocity_scale < 1.8   # 35.2 / 20 ms
    n_post = min(rollout.ctrl.shape[0] - warmup, mpc_ref.num_frames)
    n_pairs = min(int(np.floor((n_post - 1) / spf + 1e-9)), env_ref.num_frames - 1)
    assert ds.obs.shape == (n_pairs, OBS_DIM) and bool(torch.isfinite(ds.obs).all())
    wrist, fingers = ds.action[:, :6].abs(), ds.action[:, 6:].abs()
    # The MPC stays within a few cm / rad of its own nominal wrist control; a wrong time base inflates this far past it.
    assert float(wrist.max()) < 0.05
    assert float(fingers.max()) <= 0.95 * 0.3 + 1e-6 and 0.0 < ds.clipped_fraction < 0.1
    model = ActorCritic(OBS_DIM, 12, NA)
    stats = bc_pretrain(model, (ds.obs, ds.action), n_epochs=1, lr=3e-4, delta_max=0.3)
    assert all(np.isfinite(v) for v in stats.values())
    logger.info("C4 %s: %d control-grid pairs (%d rollout rows, %d frames, %.4f MPC steps per frame, qvel x %.4f), "
                "|residual| wrist max %.4f, fingers mean %.4f; %.2f%% of entries clipped at 0.285; one epoch MSE "
                "%.4e -> %.4e", arm, n_pairs, rollout.ctrl.shape[0], T, spf, ds.velocity_scale, float(wrist.max()),
                float(fingers.mean()), 100.0 * ds.clipped_fraction, stats["mse_initial"], stats["mse_final"])
