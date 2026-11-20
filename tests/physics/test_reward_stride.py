"""`executor.rollout` with a reward stride, on CPU with a fake core.

The physics step, perturbation schedule and device waits are stubbed; `reward_step` is replaced by a deterministic
per-(world, step) reward so the accumulation can be checked against independent formulas:
- stride 1 (the default) is BITWISE the registered accumulation (DO AS I DO's `make_rollout_fn`, as the executor
  writes it), and calls the reward on every step;
- stride s equals the probe partial sum `sub{s}` (terminal step at weight 1 x its terminal scale, non-terminal
  steps with t % s == s - 1 at weight s, warmup weight of that step) and
  calls the reward on H // s steps plus the terminal one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastdo.physics import executor as X
from fastdo.physics.lockstep import RngSlot

N, K = 6, 2


def _reward(abs_step: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + abs_step)
    return -torch.rand(N * K, generator=g, dtype=torch.float32)


class _Sched:
    def verify(self) -> None:
        pass


def _core(monkeypatch: pytest.MonkeyPatch, stride: int, warmup_sim_step: int, calls: list[int]) -> X.ClipCore:
    config = SimpleNamespace(num_perturb_samples=K, device="cpu", warmup_steps=600, warmup_sim_step=warmup_sim_step,
                             terminal_rew_scale=10.0, warmup_rew_multiplier=0.1, default_gravity=[0.0, 0.0, -9.81])
    env = SimpleNamespace(_perturb_active={}, _perturb_force={}, _perturb_torque={}, graph=None)
    M = SimpleNamespace(save_state=lambda e: None, save_env_params=lambda c, e: {}, load_env_params=lambda c, e, p: e,
                        set_weld_active=lambda c, e, on: None, set_gravity=lambda e, g: None,
                        load_state=lambda e, s: e)

    def fake_reward(config, rstat, qpos, qvel, site_xpos, cv, ref, abs_step, mul_quat):
        calls.append(abs_step)
        r = _reward(abs_step)
        return r, {"qpos_dist": -r, "pen_penalty": torch.zeros_like(r)}

    monkeypatch.setattr(X, "reward_step", fake_reward)
    monkeypatch.setattr(X, "build_schedule", lambda plan, rng, H, s0: _Sched())
    monkeypatch.setattr(X, "apply_row", lambda xfrc, sched, plan, t: None)
    monkeypatch.setattr(X, "_event", lambda: None)

    def no_wait(ev):
        return iter(())

    monkeypatch.setattr(X, "wait_event", no_wait)
    z = torch.zeros(N * K, 3)
    return X.ClipCore(config=config, env=env, M=M, S=None, mul_quat=None,
                      wp=SimpleNamespace(capture_launch=lambda g: None), rslot=RngSlot(active=False), rng=None,
                      plan=None, rstat=None, cv=None, qpos=z, qvel=z, ctrl=torch.zeros(N * K, 4), xfrc=z,
                      site_xpos=z, log=None, reward_stride=stride)


def _run(core: X.ClipCore, H: int) -> tuple[torch.Tensor, dict]:
    ctrls = torch.zeros(N, H, 4)
    ref_slice = (torch.zeros(H, 3),)
    gen = X.rollout(core, ctrls, ref_slice, {})
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        _, rew, info = stop.value
    return rew, info


def _warmup_steps_in_rollout(s0: int, H: int, W: int = 600) -> int:
    if s0 >= W:
        return 0
    tt = W - s0 - 1
    return H if tt >= H else max(tt, 0)


def _registered(s0: int, H: int) -> torch.Tensor:
    """The registered (stride-1) accumulation, written out independently, same op order."""
    wr = _warmup_steps_in_rollout(s0, H)
    cum = torch.zeros(N * K)
    for t in range(H):
        r = _reward(s0 + t + 1)
        if t == H - 1:
            r = 10.0 * r
        cum += 0.1 * r if t < wr else r
    eff = (H - wr) + 0.1 * wr
    return (cum / max(eff, 1)).view(N, K).mean(dim=-1)


def _probe_sub(s0: int, H: int, s: int) -> torch.Tensor:
    """The probe's `sub{s}`: w = warmup weight of step t; terminal adds w*r (r carries the terminal scale)."""
    wr = _warmup_steps_in_rollout(s0, H)
    acc = torch.zeros(N * K, dtype=torch.float64)
    for t in range(H):
        r = _reward(s0 + t + 1).double()
        terminal = t == H - 1
        if terminal:
            r = 10.0 * r
        w = 0.1 if t < wr else 1.0
        if terminal:
            acc += w * r
        elif t % s == s - 1:
            acc += s * w * r
    eff = (H - wr) + 0.1 * wr
    return (acc / max(eff, 1)).view(N, K).mean(dim=-1)


@pytest.mark.parametrize("s0", [700, 100, 300, 599])
def test_stride_one_is_bitwise_the_registered_accumulation(monkeypatch: pytest.MonkeyPatch, s0: int) -> None:
    calls: list[int] = []
    rew, _ = _run(_core(monkeypatch, 1, s0, calls), H=600)
    assert torch.equal(rew, _registered(s0, 600))
    assert calls == [s0 + t + 1 for t in range(600)]


@pytest.mark.parametrize("stride", [2, 5, 10])
@pytest.mark.parametrize("s0", [700, 100, 300, 599])
def test_stride_equals_probe_partial_sum(monkeypatch: pytest.MonkeyPatch, stride: int, s0: int) -> None:
    calls: list[int] = []
    rew, _ = _run(_core(monkeypatch, stride, s0, calls), H=600)
    torch.testing.assert_close(rew.double(), _probe_sub(s0, 600, stride), rtol=1e-5, atol=1e-7)
    evaluated = [t for t in range(600) if t == 599 or t % stride == stride - 1]
    assert calls == [s0 + t + 1 for t in evaluated]
    assert len(calls) == 600 // stride + (0 if 599 % stride == stride - 1 else 1)


def test_stride_five_calls_the_reward_120_times(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    _run(_core(monkeypatch, 5, 700, calls), H=600)
    assert len(calls) == 120          # t = 4, 9, ..., 599 (599 is the terminal step)


def test_stride_info_is_scaled_like_the_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Info terms of a non-terminal evaluated step are scaled by s (their per-step mean stays comparable)."""
    calls: list[int] = []
    _, info = _run(_core(monkeypatch, 5, 700, calls), H=600)
    exp = torch.zeros(N * K, dtype=torch.float64)
    for t in range(600):
        if t == 599:
            exp += -_reward(701 + t).double()
        elif t % 5 == 4:
            exp += 5 * -_reward(701 + t).double()
    torch.testing.assert_close(info["qpos_dist"].double(), (exp / 600).view(N, K).mean(-1), rtol=1e-5, atol=1e-7)


def test_stride_changes_nothing_but_the_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rollout yields once per step whatever the stride (the scheduler's step accounting is unchanged)."""
    for stride in (1, 5):
        core = _core(monkeypatch, stride, 700, [])
        gen = X.rollout(core, torch.zeros(N, 600, 4), (torch.zeros(600, 3),), {})
        n = 0
        try:
            while True:
                next(gen)
                n += 1
        except StopIteration:
            pass
        assert n == 600
        assert core.stats == {"rollouts": 1, "steps": 600}


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "5", None])
def test_check_reward_stride_refuses_non_positive_ints(bad: object) -> None:
    with pytest.raises(ValueError):
        X.check_reward_stride(bad)


def test_reward_evaluated_pattern() -> None:
    assert all(X.reward_evaluated(t, 600, 1) for t in range(600))
    assert [t for t in range(20) if X.reward_evaluated(t, 20, 5)] == [4, 9, 14, 19]
    assert [t for t in range(12) if X.reward_evaluated(t, 12, 5)] == [4, 9, 11]
