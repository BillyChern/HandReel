"""D1 optimize loop on a fake rollout (CPU torch) and the tier-0 controller's offsets / servo / plan."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastdo.physics.c2_rollout import TierZeroController, quat_angle
from fastdo.physics.d1 import D1Stats, make_d1_optimize_fn, make_d1_optimize_once_fn
from fastdo.physics.planner_config import D1Config

N, KNOTS, KS, NU, ITERS = 32, 4, 5, 3, 6


def _config() -> SimpleNamespace:
    ns = torch.full((N, KNOTS, NU), 0.3)
    ns[:, 0] = 0.0
    ns[0] = 0.0
    return SimpleNamespace(noise_scale=ns, device="cpu", knot_steps=KS, num_samples=N, num_perturb_samples=2,
                           temperature=1.0, max_num_iterations=ITERS, beta_traj=0.9, env_params_list=[[{}]] * ITERS,
                           improvement_check_steps=2, improvement_threshold=1e-4, num_trace_uniform_samples=4,
                           num_trace_topk_samples=2)


TARGET = torch.ones(KNOTS * KS, NU)


def _rollout(config, env, samples, ref_slice, env_param):
    rews = -((samples - TARGET) ** 2).mean(dim=(1, 2))
    return samples, rews, {"qpos_dist": -rews}


def test_d1_optimize_improves_and_keeps_their_info_layout() -> None:
    torch.manual_seed(0)
    cfg = D1Config(enabled=True)
    stats = D1Stats()
    once = make_d1_optimize_once_fn(_rollout, cfg, stats)
    opt = make_d1_optimize_fn(once, cfg, stats)
    c = _config()
    ctrls = torch.zeros(KNOTS * KS, NU)
    err0 = float(((ctrls - TARGET) ** 2).mean())
    for _ in range(4):
        ctrls, info = opt(c, None, ctrls, ())
    assert float(((ctrls - TARGET) ** 2).mean()) < 0.7 * err0
    assert info["rew_max"].shape == (ITERS,) and info["d1_reverted"].shape == (ITERS,)
    assert info["opt_steps"].shape == (1,) and 1 <= int(info["opt_steps"][0]) <= ITERS
    assert stats.rollout_world_steps == stats.rollouts * N * KNOTS * KS * 2
    assert stats.optimize_calls == 4


def test_acceptance_keeps_a_better_previous_iterate() -> None:
    torch.manual_seed(1)
    once = make_d1_optimize_once_fn(_rollout, D1Config(enabled=True))
    c = _config()
    far = torch.full((KNOTS * KS, NU), -3.0)
    new, info, kept = once(c, None, far, (), [{}], {"global_noise_scale": 1.0}, prev_ctrls=TARGET.clone())
    assert kept and torch.equal(new, TARGET)
    assert info["improvement"] <= 1e-9
    new2, _, kept2 = once(c, None, TARGET.clone(), (), [{}], {"global_noise_scale": 1.0}, prev_ctrls=far)
    assert not kept2


def test_equivalence_mode_is_the_registered_weighted_mean() -> None:
    from fastdo.physics.rank_estimator import knot_interp, registered_update, registered_weights

    c = _config()
    cfg = D1Config(enabled=True, antithetic=False, rank_weights=False, accept_never_worse=False)
    once = make_d1_optimize_once_fn(_rollout, cfg)
    base = torch.zeros(KNOTS * KS, NU)
    torch.manual_seed(5)
    new, _, _ = once(c, None, base, (), [{}], {"global_noise_scale": 1.0})
    torch.manual_seed(5)
    knots = torch.randn_like(c.noise_scale) * c.noise_scale * torch.tensor(1.0)
    samples = base + knot_interp(knots, KS)
    _, rews, _ = _rollout(c, None, samples, (), {})
    w, _ = registered_weights(rews, N, 1.0)
    assert torch.allclose(new, registered_update(samples, w))


def _ctl(t: int = 50) -> TierZeroController:
    base = torch.zeros(t, 8)
    base[:, 2] = 0.02
    finger = torch.zeros(5, t, 8)
    finger[1, :, 6] = 0.1
    finger[0, :, 7] = -0.2
    ref = torch.zeros(t, 3)
    return TierZeroController(ctrl_ref=torch.zeros(t, 8), base=base, finger=finger,
                              finger_cols=torch.tensor([6, 7]), z_col=2, joint_max=0.3, ref_obj_pos=ref,
                              obj_pos_slice=slice(8, 11), release_step=10, servo_sat=0.02)


def test_offsets_scaling_clamp_and_z() -> None:
    ctl = _ctl()
    th = torch.zeros(2, 7)
    th[1, :5] = torch.log(torch.tensor(4.0))
    th[1, 6] = 0.005
    off = ctl.offsets(20, th)
    assert off[0, 6] == pytest.approx(0.1) and off[0, 7] == pytest.approx(-0.2) and off[0, 2] == pytest.approx(0.02)
    assert off[1, 6] == pytest.approx(0.3) and off[1, 7] == pytest.approx(-0.3)    # clamped at joint_max
    assert off[1, 2] == pytest.approx(0.025)
    plan = ctl.plan(45, 10, torch.zeros(7))
    assert plan.shape == (10, 8) and torch.allclose(plan[-1], plan[4])               # padding holds the last row


def test_servo_saturates_and_waits_for_release() -> None:
    ctl = _ctl()
    qpos = torch.zeros(2, 15)
    qpos[:, 8:11] = torch.tensor([0.0, 0.0, -0.1])
    corr = ctl.servo(qpos, torch.tensor([5, 20]), torch.tensor([1.0, 1.0]))
    assert torch.equal(corr[0], torch.zeros(3))
    assert corr[1, 2] == pytest.approx(0.02) and float(corr[1].norm()) == pytest.approx(0.02)


def test_quat_angle() -> None:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [np.cos(0.25), 0.0, 0.0, np.sin(0.25)]], dtype=torch.float32)
    a = quat_angle(q, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert a[0] == pytest.approx(0.0, abs=1e-3) and a[1] == pytest.approx(0.5, abs=1e-4)
