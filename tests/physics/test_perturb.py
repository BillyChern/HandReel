"""The pre-built perturbation schedule reproduces DO AS I DO's `apply_perturbation` bit for bit and consumes the
generator exactly as it does (CPU generator, `decide="device"`); Philox4x32-10 matches Random123's known answers."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from fastdo.physics import perturb as P

XML = """
<mujoco>
  <asset><mesh name="m" vertex="0 0 0 0.05 0 0 0 0.05 0 0 0 0.05"/></asset>
  <worldbody>
    <body name="right_object" pos="0.1 0 0.1"><freejoint/><geom type="mesh" mesh="m" mass="0.3"/></body>
    <body name="left_object" pos="0.2 0 0.1"><freejoint/><geom type="sphere" size="0.01"/></body>
  </worldbody>
</mujoco>
"""


def test_philox_known_answers() -> None:
    """Random123 kat_vectors for philox4x32_10."""
    assert P.philox4x32_10((0, 0, 0, 0), (0, 0)) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    f = 0xFFFFFFFF
    assert P.philox4x32_10((f, f, f, f), (f, f)) == (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)
    assert P.philox4x32_10((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0)) == (
        0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)


def test_uniform_mapping_range() -> None:
    u = P.torch_uniform_at(0, 0, 64)
    assert u.dtype == np.float32 and (u > 0).all() and (u < 1).all() and len(set(u.tolist())) == 64
    assert not np.array_equal(P.torch_uniform_at(0, 4, 2), P.torch_uniform_at(0, 0, 2))
    with pytest.raises(AssertionError):
        P.torch_uniform_at(0, 2, 2)


def _setup(K: int, n_ctrl: int, T: int, gate_bits: np.ndarray, prob: float, cont: float, warmup: int):
    model = mujoco.MjModel.from_xml_string(XML)
    N = K * n_ctrl
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    env = SimpleNamespace(model_cpu=model, num_worlds=N,
                          data_wp=SimpleNamespace(xfrc_applied=torch.full((N, model.nbody, 6), 7.0)),
                          _perturb_active={}, _perturb_force={}, _perturb_torque={},
                          _perturb_gate_mask={rid: torch.as_tensor(gate_bits)})
    cfg = SimpleNamespace(num_perturb_samples=K, default_gravity=[0.0, 0.0, -9.81], perturb_force=0.0,
                          perturb_torque=0.0, perturb_force_scale=0.5, perturb_torque_scale=0.5, perturb_prob=prob,
                          perturb_continue_prob=cont, warmup_steps=warmup)
    return cfg, env


@pytest.mark.parametrize("prob,cont,ws", [(0.3, 0.7, 0), (0.05, 0.95, 3), (0.9, 0.2, 10)])
def test_schedule_matches_apply_perturbation(daid, prob: float, cont: float, ws: int) -> None:
    K, n_ctrl, T, H = 2, 5, 40, 24
    rng = np.random.default_rng(int(prob * 100) + ws)
    gate = rng.random(T) < 0.7
    gate[15:19] = False
    cfg, env = _setup(K, n_ctrl, T, gate, prob, cont, warmup=8)
    from retargeting.utils.in_hand import body_has_mesh_geom

    plan = P.plan_from_env(cfg, env, body_has_mesh_geom,
                           lambda n: mujoco.mj_name2id(env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, n))
    x0 = env.data_wp.xfrc_applied.clone()
    for rollout in range(3):                           # state resets between rollouts, as `rollout` does
        torch.manual_seed(100 + rollout)
        env.data_wp.xfrc_applied = x0.clone()
        for d in (env._perturb_active, env._perturb_force, env._perturb_torque):
            for v in d.values():
                v.zero_()
        ref_trace = []
        for t in range(H):
            abs_step = ws + t + 1
            if abs_step >= cfg.warmup_steps:
                daid.M.apply_perturbation(cfg, env, sim_step=abs_step)
            ref_trace.append(env.data_wp.xfrc_applied.clone())
        ref_state = torch.get_rng_state()

        torch.manual_seed(100 + rollout)
        x = x0.clone()
        sched = P.build_schedule(plan, P.Rng("cpu", decide="device"), H, ws)
        assert torch.equal(torch.get_rng_state(), ref_state)      # identical generator consumption
        sched.verify()
        for t in range(H):
            P.apply_row(x, sched, plan, t)
            assert torch.equal(x.view(torch.int32), ref_trace[t].view(torch.int32)), (rollout, t)


def test_schedule_emulate_refuses_cpu() -> None:
    with pytest.raises(ValueError):
        P.Rng("cpu", decide="emulate")


def test_verify_detects_divergence() -> None:
    s = P.Schedule(rows={}, step_row=[], drawn=[torch.tensor([[0.5], [0.25]])],
                   predicted=[np.array([[0.5], [0.2500001]], dtype=np.float32)])
    with pytest.raises(RuntimeError):
        s.verify()
