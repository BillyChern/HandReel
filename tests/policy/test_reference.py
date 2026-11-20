"""Geodesic resampling of the reference and the future-window chart (design Section 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp, se3_inv, se3_log, so3_exp
from fastdo.policy.reference import PolicyReference, future_window, resample_reference
from fastdo.retarget.kinematic import KinematicTrajectory

DT = torch.float64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _rand_pose(n: int, gen: torch.Generator, rot_scale: float = 1.0, tr_scale: float = 0.5) -> Tensor:
    xi = torch.randn(n, 6, dtype=DT, generator=gen)
    xi[:, :3] *= tr_scale
    xi[:, 3:] *= rot_scale
    return se3_exp(xi)


def _kin(qpos: Tensor, frequency: float) -> KinematicTrajectory:
    q = qpos.numpy()
    return KinematicTrajectory(qpos=q, qvel=np.zeros((q.shape[0], q.shape[1] - 1)), frequency=frequency)


def _gravity_group_element(gen: torch.Generator, gravity: Tensor) -> Tensor:
    """Random h in SE(2) x R: a rotation about the gravity axis plus an arbitrary translation."""
    u = -gravity / gravity.norm()
    yaw = (torch.rand((), dtype=DT, generator=gen) * 2.0 - 1.0) * math.pi
    h = torch.eye(4, dtype=DT)
    h[:3, :3] = so3_exp(yaw * u)
    h[:3, 3] = torch.randn(3, dtype=DT, generator=gen)
    return h


@pytest.mark.parametrize("f_src", [10.0, 30.0, 7.0])       # 10 fps is the C3b rate; others: non-integer ratios
def test_resample_screw_motion_exact(f_src: float) -> None:
    gen = _gen(0)
    T0 = _rand_pose(1, gen)[0]
    xi = torch.tensor([0.08, -0.05, 0.12, 0.30, -0.45, 0.20], dtype=DT)   # one source step, 0.61 rad
    n_src, hz = 12, 50.0
    steps = torch.arange(n_src, dtype=DT)
    src = T0 @ se3_exp(steps[:, None] * xi)                                  # T_i = T_0 Exp(i xi)
    T_src = src.shape[0]
    kin = _kin(torch.zeros(T_src, 5, dtype=DT), f_src)
    plan = torch.zeros(T_src, 4, dtype=torch.bool)

    ref = resample_reference(kin, src, plan, None, hz=hz)

    n_out = math.floor((n_src - 1) * hz / f_src + 1e-9) + 1                # same time span, t <= t_last
    assert ref.obj_pose.shape == (n_out, 4, 4)
    if f_src == 10.0:
        assert n_out == (n_src - 1) * 5 + 1
    t = torch.arange(n_out, dtype=DT) * f_src / hz                          # source-frame units
    expected = T0 @ se3_exp(t[:, None] * xi)
    err = (ref.obj_pose - expected).abs().amax()
    assert err < 1e-9, f"geodesic resampling error {err:.3e}"
    assert ref.hz == hz
    assert ref.covariance is None


def test_resample_joints_plan_covariance_channels() -> None:
    gen = _gen(1)
    T_src, nq, L, f_src, hz = 6, 7, 5, 10.0, 50.0
    poses = _rand_pose(T_src, gen, rot_scale=0.3)
    q = torch.randn(T_src, nq, dtype=DT, generator=gen)
    plan = torch.rand(T_src, L, generator=gen) > 0.5
    A = torch.randn(T_src, 7, 7, dtype=DT, generator=gen)
    cov = A @ A.transpose(-1, -2) + 0.1 * torch.eye(7, dtype=DT)

    ref = resample_reference(_kin(q, f_src), poses, plan, cov, hz=hz, log_scale=0.25)

    n_out = (T_src - 1) * 5 + 1
    assert isinstance(ref, PolicyReference)
    assert ref.q_ref.shape == (n_out, nq) and ref.q_ref.dtype == DT
    assert ref.contact_plan.shape == (n_out, L) and ref.contact_plan.dtype == torch.bool
    assert ref.covariance is not None and ref.covariance.shape == (n_out, 7, 7)
    assert ref.log_scale == 0.25
    for k in range(n_out):
        i, r = divmod(k, 5)
        alpha = r / 5.0
        j = min(i + 1, T_src - 1)
        assert torch.allclose(ref.q_ref[k], (1 - alpha) * q[i] + alpha * q[j], atol=1e-12)
        assert torch.allclose(ref.covariance[k], (1 - alpha) * cov[i] + alpha * cov[j], atol=1e-12)
        nearest = i + 1 if alpha > 0.5 else i
        assert torch.equal(ref.contact_plan[k], plan[nearest])
    # Source frames are reproduced exactly (alpha = 0).
    assert torch.allclose(ref.obj_pose[::5], poses, atol=1e-12)


def test_resample_rejects_mismatched_lengths() -> None:
    gen = _gen(2)
    poses = _rand_pose(5, gen)
    with pytest.raises(ValueError):
        resample_reference(_kin(torch.zeros(4, 3, dtype=DT), 10.0), poses,
                           torch.zeros(5, 2, dtype=torch.bool), None)
    with pytest.raises(ValueError):
        resample_reference(_kin(torch.zeros(5, 3, dtype=DT), 10.0), poses,
                           torch.zeros(5, 2, dtype=torch.bool), torch.zeros(4, 7, 7, dtype=DT))


def test_future_window_invariant_under_group() -> None:
    gen = _gen(3)
    T, nq, N, H = 25, 6, 8, 10
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=DT)
    ref = PolicyReference(q_ref=torch.randn(T, nq, dtype=DT, generator=gen),
                          obj_pose=_rand_pose(T, gen, rot_scale=0.4),
                          log_scale=0.0,
                          contact_plan=torch.zeros(T, 3, dtype=torch.bool),
                          covariance=None, hz=50.0)
    k = torch.tensor([0, 3, 7, 12, 14, T - 5, T - 2, T - 1])
    cur_pose = ref.obj_pose[k] @ _rand_pose(N, gen, rot_scale=0.2, tr_scale=0.05)
    cur_q = torch.randn(N, nq, dtype=DT, generator=gen)

    twists, dq = future_window(ref, k, cur_pose, cur_q, horizon=H)
    assert twists.shape == (N, H, 6) and dq.shape == (N, H, nq)

    # Definition: xi_{k,h} = Log(T_cur^{-1} T_ref,min(k+h, T-1)), h = 1..H; joint deltas likewise.
    idx = (k[:, None] + torch.arange(1, H + 1)[None]).clamp_max(T - 1)
    expected = se3_log(se3_inv(cur_pose)[:, None] @ ref.obj_pose[idx])
    assert torch.allclose(twists, expected, atol=1e-12)
    assert torch.allclose(dq, ref.q_ref[idx] - cur_q[:, None], atol=1e-12)

    # Clamped at the last frame: every step past T-1 repeats the last frame's twist.
    last = N - 1                                                            # k = T-1
    assert torch.allclose(twists[last], twists[last, :1].expand(H, 6), atol=0.0)
    row = N - 2                                                             # k = T-2: h=1 hits T-1
    assert torch.allclose(twists[row, 1:], twists[row, :1].expand(H - 1, 6), atol=0.0)

    # Invariance under the gravity-fixing group SE(2) x R (and in fact under all of SE(3)).
    elements = [_gravity_group_element(gen, gravity) for _ in range(3)] + list(_rand_pose(2, gen))
    for h in elements:
        moved = PolicyReference(q_ref=ref.q_ref, obj_pose=h @ ref.obj_pose, log_scale=ref.log_scale,
                                contact_plan=ref.contact_plan, covariance=None, hz=ref.hz)
        tw_h, dq_h = future_window(moved, k, h @ cur_pose, cur_q, horizon=H)
        err = (tw_h - twists).abs().amax()
        assert err < 1e-9, f"window twists changed under the group action by {err:.3e}"
        assert torch.equal(dq_h, dq)
