"""Batched tracking rewards, termination and the baseline success metric (design Section 5)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.bench.metrics import retarget_errors
from fastdo.geometry.lie import se3_exp, se3_inv, se3_log, so3_exp
from fastdo.policy.reference import PolicyReference
from fastdo.policy.rewards import (
    REWARD_TERMS,
    RewardConfig,
    StepObservation,
    clip_success,
    pose_errors,
    step_reward,
    terminated,
)

DT = torch.float64
HZ, NQ, NA, L = 50.0, 5, 4, 4
CFG = RewardConfig()


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _reference(T: int, gen: torch.Generator) -> PolicyReference:
    poses = se3_exp(torch.randn(T, 6, dtype=DT, generator=gen))
    plan = torch.rand(T, L, generator=gen) < 0.5
    return PolicyReference(q_ref=torch.randn(T, NQ, dtype=DT, generator=gen), obj_pose=poses,
                           log_scale=0.0, contact_plan=plan, covariance=None, hz=HZ)


def _obs(ref: PolicyReference, k: Tensor, obj_pose: Tensor, gen: torch.Generator,
         **overrides: Tensor) -> StepObservation:
    """A batch at reference steps `k` with random joints/contacts/actions near plausible magnitudes."""
    n = int(k.shape[0])
    q_near = ref.q_ref[k.clamp_max(ref.num_frames - 1)] + 0.2 * torch.randn(n, NQ, dtype=DT, generator=gen)
    fields: dict[str, Tensor] = {
        "q": q_near,
        "qd": torch.randn(n, NQ, dtype=DT, generator=gen),
        "obj_pose": obj_pose,
        "obj_twist": torch.randn(n, 6, dtype=DT, generator=gen),
        "contact_force": 2.0 * torch.rand(n, L, dtype=DT, generator=gen),
        "action": torch.randn(n, NA, dtype=DT, generator=gen),
        "prev_action": torch.randn(n, NA, dtype=DT, generator=gen),
        "k": k,
    }
    fields.update(overrides)
    return StepObservation(**fields)


def _perturb(poses: Tensor, rot: float, tr: float, gen: torch.Generator) -> Tensor:
    """Left-perturb each pose by a rotation of angle `rot` and a translation offset of norm `tr`."""
    n = poses.shape[0]
    axis = torch.nn.functional.normalize(torch.randn(n, 3, dtype=DT, generator=gen), dim=-1)
    offset = torch.nn.functional.normalize(torch.randn(n, 3, dtype=DT, generator=gen), dim=-1)
    out = poses.clone()
    out[:, :3, :3] = so3_exp(rot * axis) @ poses[:, :3, :3]
    out[:, :3, 3] = poses[:, :3, 3] + tr * offset
    return out


def test_pose_reward_left_invariant() -> None:
    gen = _gen(0)
    T, N = 20, 128
    ref = _reference(T, gen)
    k = torch.randint(0, T, (N,), generator=gen)
    rot = 0.1 * torch.rand(N, dtype=DT, generator=gen)
    obj = ref.obj_pose[k].clone()
    for i in range(N):                                            # errors spread over [0, 0.1] rad, [0, 2] cm
        obj[i:i + 1] = _perturb(ref.obj_pose[k[i]][None], float(rot[i]), 0.2 * float(rot[i]), gen)
    obs = _obs(ref, k, obj, gen)
    reward, terms = step_reward(CFG, ref, obs)
    assert reward.shape == (N,) and reward.dtype == DT
    assert float(terms["pose"].std()) > 0.05                      # the pose term is not saturated
    assert torch.equal(terms["drop"], torch.zeros(N, dtype=DT))  # 2 cm offsets never reach drop_height

    for seed in range(3):
        g = se3_exp(2.0 * torch.randn(6, dtype=DT, generator=_gen(100 + seed)))
        ref_g = dataclasses.replace(ref, obj_pose=g @ ref.obj_pose)
        obs_g = dataclasses.replace(obs, obj_pose=g @ obs.obj_pose)
        reward_g, terms_g = step_reward(CFG, ref_g, obs_g)
        assert (reward_g - reward).abs().max() < 1e-9
        for name in (*REWARD_TERMS, "e_pos", "e_rot"):
            assert (terms_g[name] - terms[name]).abs().max() < 1e-9, name


def test_pose_distance_matches_baseline_errors() -> None:
    gen = _gen(1)
    T = 60
    ref = _reference(T, gen)
    angles = 0.05 + 2.45 * torch.rand(T, dtype=DT, generator=gen)   # arccos in retarget_errors is exact here
    pred = torch.cat([_perturb(ref.obj_pose[i][None], float(angles[i]), 0.08 * float(angles[i]), gen)
                      for i in range(T)])
    k = torch.arange(T)
    _, terms = step_reward(CFG, ref, _obs(ref, k, pred, gen))

    pred_np, ref_np = pred.numpy(), ref.obj_pose.numpy()
    for i in range(T):
        per_frame = retarget_errors(pred_np[i:i + 1], ref_np[i:i + 1])
        assert abs(float(terms["e_pos"][i]) - per_frame.e_pos) < 1e-9
        assert abs(float(terms["e_rot"][i]) - per_frame.e_rot) < 1e-9
    whole = retarget_errors(pred_np, ref_np)
    assert abs(float(terms["e_pos"].mean()) - whole.e_pos) < 1e-9
    assert abs(float(terms["e_rot"].mean()) - whole.e_rot) < 1e-9
    assert torch.allclose(torch.stack(pose_errors(pred, ref.obj_pose)),
                          torch.stack([terms["e_pos"], terms["e_rot"]]), atol=1e-14, rtol=0.0)

    d2 = CFG.w_rot * terms["e_rot"] ** 2 + CFG.w_tr * terms["e_pos"] ** 2
    assert torch.allclose(terms["pose"], CFG.w_pose * torch.exp(-d2 / CFG.sigma_pose**2), atol=1e-12)
    # Regression guard: the se(3) log's translational part is a different quantity once omega != 0.
    rho = se3_log(se3_inv(ref.obj_pose) @ pred)[:, :3].norm(dim=-1)
    assert (rho - terms["e_pos"]).abs().max() > 1e-2


def test_pose_reward_peaks_at_reference() -> None:
    gen = _gen(2)
    T, N = 10, 64
    ref = _reference(T, gen)
    k = torch.randint(0, T, (N,), generator=gen)
    base = _obs(ref, k, ref.obj_pose[k].clone(), gen)
    reward0, terms0 = step_reward(CFG, ref, base)
    assert torch.equal(terms0["pose"], torch.full((N,), CFG.w_pose, dtype=DT))
    assert torch.equal(terms0["e_pos"], torch.zeros(N, dtype=DT))

    cases = ((1e-4, 0.0), (0.0, 1e-4), (0.02, 0.0), (0.0, 0.01), (0.3, 0.05), (math.pi - 1e-3, 0.2))
    for rot, tr in cases:
        perturbed = dataclasses.replace(base, obj_pose=_perturb(ref.obj_pose[k], rot, tr, gen))
        reward, terms = step_reward(CFG, ref, perturbed)
        assert bool((reward < reward0).all()), (rot, tr)
        assert bool((terms["pose"] < terms0["pose"]).all()), (rot, tr)


def test_contact_reward_sign_cases() -> None:
    gen = _gen(3)
    ref = _reference(5, gen)
    plan = torch.tensor([[True, False, False, False],        # row 0: planned + touching
                         [False, False, False, False],       # row 1: unplanned + touching
                         [True, True, False, False],         # row 2: planned + not touching (below f_min)
                         [False, False, False, False],       # row 3: nothing planned, nothing touching
                         [True, False, True, False]])        # row 4: force exactly f_min is not a contact
    ref = dataclasses.replace(ref, contact_plan=plan)
    force = torch.zeros(5, L, dtype=DT)
    force[0, 0] = 3.0
    force[1, 1] = 3.0
    force[2, :2] = 0.4 * CFG.f_min
    force[4, 0] = CFG.f_min
    k = torch.arange(5)
    _, terms = step_reward(CFG, ref, _obs(ref, k, ref.obj_pose[k], gen, contact_force=force))
    contact = terms["contact"]
    assert float(contact[0]) > 0.0 and float(contact[1]) < 0.0
    assert float(contact[2]) == 0.0 and float(contact[3]) == 0.0 and float(contact[4]) == 0.0
    assert float(contact[0]) == pytest.approx(CFG.w_contact / L, abs=1e-15)
    assert float(contact[1]) == pytest.approx(-CFG.w_contact * CFG.lambda_contact / L, abs=1e-15)


def test_joint_action_terms_and_total_reward() -> None:
    gen = _gen(4)
    T = 6
    ref = _reference(T, gen)
    k = torch.tensor([0, 3, T - 1, T + 7])                          # past the end clamps to the last frame
    idx = k.clamp_max(T - 1)
    dq = torch.zeros(4, NQ, dtype=DT)
    dq[:, 0] = CFG.sigma_joint
    act = torch.zeros(4, NA, dtype=DT)
    prev = act.clone()
    prev[:, :2] = torch.tensor([0.1, -0.2], dtype=DT)
    obs = _obs(ref, k, ref.obj_pose[idx], gen, q=ref.q_ref[idx] + dq, action=act, prev_action=prev)
    reward, terms = step_reward(CFG, ref, obs)
    assert torch.allclose(terms["joint"], torch.full((4,), CFG.w_joint * math.exp(-1.0), dtype=DT),
                          atol=1e-15)
    assert torch.allclose(terms["action_rate"], torch.full((4,), -CFG.w_action * 0.05, dtype=DT),
                          atol=1e-15)
    assert torch.equal(terms["pose"], torch.full((4,), CFG.w_pose, dtype=DT))
    assert torch.allclose(reward, torch.stack([terms[name] for name in REWARD_TERMS]).sum(0), atol=1e-15)


def test_terminated_on_drop() -> None:
    gen = _gen(5)
    T = 8
    ref = _reference(T, gen)
    k = torch.full((7,), 2)
    obj = ref.obj_pose[k].clone()
    obj[1, 2, 3] -= 0.2                                             # 0.2 m below: drop (translation < 0.3)
    obj[2, 2, 3] -= 0.05                                            # 5 cm below: neither
    obj[3, 2, 3] += 0.2                                             # 0.2 m above: not a drop
    obj[4, 0, 3] += 0.35                                            # 0.35 m lateral: tracking lost
    obj[5, 1, 3] += 0.25                                            # 0.25 m lateral: neither
    # Row 6: 2.5 rad world yaw plus a 0.25 m horizontal offset. |p - p_ref| = 0.25 m keeps tracking; the
    # se(3) log's translation is (theta/2)/sin(theta/2) = 1.32x longer (offset orthogonal to omega).
    obj[6, :3, :3] = so3_exp(torch.tensor([0.0, 0.0, 2.5], dtype=DT)) @ obj[6, :3, :3]
    obj[6, 0, 3] += 0.25
    obs = _obs(ref, k, obj, gen)

    done = terminated(CFG, ref, obs)
    assert done.dtype == torch.bool and done.shape == (7,)
    assert done.tolist() == [False, True, False, False, True, False, False]
    rho = se3_log(se3_inv(ref.obj_pose[k]) @ obj)[:, :3].norm(dim=-1)
    assert float(rho[6]) > CFG.max_pos_error                        # the product metric, not the se(3) log
    _, terms = step_reward(CFG, ref, obs)
    assert terms["drop"].tolist() == [0.0, -CFG.w_drop, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Invariant under the gravity-fixing group (yaw about +z and any translation).
    h = torch.eye(4, dtype=DT)
    h[:3, :3] = so3_exp(torch.tensor([0.0, 0.0, 2.1], dtype=DT))
    h[:3, 3] = torch.tensor([0.7, -1.2, 0.4], dtype=DT)
    ref_h = dataclasses.replace(ref, obj_pose=h @ ref.obj_pose)
    assert torch.equal(terminated(CFG, ref_h, dataclasses.replace(obs, obj_pose=h @ obj)), done)


def _trajectory_with_error(T: int, pos_err: float, rot_err: float, gen: np.random.Generator) -> np.ndarray:
    """Reference-relative poses whose every frame has exactly `pos_err` m and `rot_err` rad of error."""
    offsets = gen.normal(size=(T, 3))
    offsets *= pos_err / np.linalg.norm(offsets, axis=1, keepdims=True)
    axes = gen.normal(size=(T, 3))
    axes *= rot_err / np.linalg.norm(axes, axis=1, keepdims=True)
    delta = np.tile(np.eye(4), (T, 1, 1))
    delta[:, :3, :3] = so3_exp(torch.from_numpy(axes)).numpy()
    delta[:, :3, 3] = offsets
    return delta


def test_clip_success_matches_retarget_errors() -> None:
    rng = np.random.default_rng(6)
    T, warmup = 120, 40
    ref = se3_exp(torch.randn(T, 6, dtype=DT, generator=_gen(6))).numpy()
    for pos_err, expect in ((0.05, True), (0.15, False)):
        d = _trajectory_with_error(T, pos_err, 0.2, rng)
        pred = ref.copy()
        pred[:, :3, :3] = d[:, :3, :3] @ ref[:, :3, :3]
        pred[:, :3, 3] = ref[:, :3, 3] + d[:, :3, 3]
        pred[:warmup, :3, 3] += 2.0                                  # garbage during the welded warm-up
        ok, errors = clip_success(pred, ref, warmup)
        assert ok is expect and errors.success is expect
        assert errors == retarget_errors(pred[warmup:], ref[warmup:])
        assert errors.e_pos == pytest.approx(pos_err, abs=1e-12)
        assert errors.e_rot == pytest.approx(0.2, abs=1e-9)
        assert not retarget_errors(pred, ref).success                 # the warm-up slice matters
    with pytest.raises(ValueError):
        clip_success(ref, ref, T)
    with pytest.raises(ValueError):
        clip_success(ref, ref, -1)
    with pytest.raises(ValueError):
        clip_success(ref[1:], ref, 0)


def test_step_reward_validates_batch_shapes() -> None:
    gen = _gen(7)
    ref = _reference(5, gen)
    k = torch.arange(3)
    good = _obs(ref, k, ref.obj_pose[k], gen)
    bad_fields: dict[str, Tensor] = {
        "q": torch.zeros(3, NQ + 1, dtype=DT),
        "qd": torch.zeros(2, NQ, dtype=DT),
        "obj_pose": ref.obj_pose[:2],
        "obj_twist": torch.zeros(3, 7, dtype=DT),
        "contact_force": torch.zeros(3, L - 1, dtype=DT),
        "action": torch.zeros(3, NA, 1, dtype=DT),
        "prev_action": torch.zeros(3, NA + 1, dtype=DT),
        "k": torch.arange(2),
    }
    for name, value in bad_fields.items():
        bad = dataclasses.replace(good, **{name: value})
        with pytest.raises(ValueError):
            step_reward(CFG, ref, bad)
        with pytest.raises(ValueError):
            terminated(CFG, ref, bad)
    with pytest.raises(ValueError):
        RewardConfig(sigma_pose=0.0)
