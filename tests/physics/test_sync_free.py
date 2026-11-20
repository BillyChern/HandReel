"""The sync-free reward reproduces DO AS I DO's `get_reward` BIT FOR BIT (CPU, DO AS I DO as the oracle)."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from fastdo.physics import sync_free as SF

XML = """
<mujoco>
  <asset><mesh name="m" vertex="0 0 0 0.05 0 0 0 0.05 0 0 0 0.05"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <geom name="right_pedestal_0" type="cylinder" size="0.05 0.02" pos="0.3 0 0.02"/>
    <body name="right_hand" pos="0 0 0.3">
      <freejoint/><geom name="collision_hand_palm" type="sphere" size="0.02"/>
    </body>
    <body name="right_object" pos="0.1 0 0.1">
      <freejoint/><geom name="right_object_mesh" type="mesh" mesh="m"/>
    </body>
  </worldbody>
</mujoco>
"""
N, NGEOM, L = 12, 4, 40
NQ, NV = 16, 15          # robot 9 hinge-like dofs (6 wrist + 3 joint) + object freejoint (7 / 6)


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(XML)


def _config(daid, model, penalty: float = 0.5) -> SimpleNamespace:
    cfg = SimpleNamespace(
        embodiment_type="right", nq_obj=7, nv=NV, nu=9, device="cpu", vel_rew_scale=1e-4,
        base_pos_rew_scale=1.0, base_rot_rew_scale=0.3, joint_rew_scale=0.003, pos_rew_scale=1.0, rot_rew_scale=0.3,
        contact_rew_scale=0.0, contact_site_ids=[], penetration_penalty_scale=3000.0, penetration_margin=0.005,
        drop_penalty_scale=0.0, drop_z_thresh=0.0, pedestal_penalty_scale=penalty, floor_penalty_scale=0.0,
        warmup_steps=5)
    daid.M.precompute_hand_object_geom_mask(cfg, model)
    return cfg


def _env(model, rng: np.random.Generator, nacon: int) -> SimpleNamespace:
    dist = torch.as_tensor(rng.normal(0.0, 0.01, L), dtype=torch.float32)
    dist[nacon + 1] = float("nan")                          # garbage beyond nacon must not leak
    dist[nacon + 2] = -5.0
    geom = torch.as_tensor(rng.integers(-2, NGEOM + 2, (L, 2)), dtype=torch.int32)
    worldid = torch.as_tensor(rng.integers(-3, N + 3, L), dtype=torch.int32)
    data = SimpleNamespace(
        qpos=torch.as_tensor(rng.normal(size=(N, NQ)), dtype=torch.float32),
        qvel=torch.as_tensor(rng.normal(size=(N, NV)), dtype=torch.float32),
        site_xpos=torch.zeros(N, 1, 3), nacon=torch.tensor([nacon], dtype=torch.int32),
        contact=SimpleNamespace(dist=dist, geom=geom, worldid=worldid))
    data.qpos[:, -4:] = data.qpos[:, -4:] / data.qpos[:, -4:].norm(dim=1, keepdim=True)
    T = 30
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    inhand = torch.as_tensor(rng.random(T) < 0.5)
    ped = torch.as_tensor(rng.random(T) < 0.5)
    return SimpleNamespace(data_wp=data, num_worlds=N, model_cpu=model, _side_object_body_id={"right": bid},
                           _inhand_gate_mask={bid: inhand}, _obj_pedestal_gate_mask={bid: ped},
                           _obj_floor_gate_mask={bid: torch.zeros(T, dtype=torch.bool)})


def _static(daid, cfg, env) -> SF.RewardStatic:
    cfg._qpos_weight_cache = daid.M._weight_diff_qpos(cfg)
    cfg._qpos_group_slices = daid.M._build_qpos_group_slices(cfg)
    bid = env._side_object_body_id["right"]
    host = {k: v.numpy() for k, v in ((0, env._inhand_gate_mask[bid]), (1, env._obj_pedestal_gate_mask[bid]),
                                      (2, env._obj_floor_gate_mask[bid]))}
    any_ped = cfg._is_right_pedestal_geom | cfg._is_left_pedestal_geom
    return SF.RewardStatic(
        group_index={k: torch.as_tensor(v, dtype=torch.long) for k, v in cfg._qpos_group_slices.items()},
        weight=cfg._qpos_weight_cache, is_hand=cfg._is_hand_geom, is_object=cfg._is_object_geom,
        any_pedestal=any_ped, any_pedestal_host=bool(any_ped.any()), is_floor=cfg._is_floor_geom,
        is_floor_host=bool(cfg._is_floor_geom.any()), hand_host=bool(cfg._is_hand_geom.any()),
        side_body={"right": bid}, body_obj_mask={bid: cfg._is_right_object_geom},
        body_obj_mask_host={bid: bool(cfg._is_right_object_geom.any())},
        ref_masks={bid: SF.RefMasks(inhand=host[0], pedestal=host[1], floor=host[2])})


def test_reward_static_matches_hand_built(daid) -> None:
    """`sync_free.reward_static` (what `executor.build_core` uses) resolves exactly the hand-built constants."""
    model = _model()
    cfg = _config(daid, model)
    cfg.contact_rew_scale, cfg.contact_site_ids = 0.0, []
    env = _env(model, np.random.default_rng(3), 5)
    want = _static(daid, cfg, env)
    got = SF.reward_static(cfg, env, lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
    for f in ("weight", "is_hand", "is_object", "any_pedestal", "is_floor"):
        assert torch.equal(getattr(got, f), getattr(want, f)), f
    for f in ("any_pedestal_host", "is_floor_host", "hand_host", "side_body", "body_obj_mask_host"):
        assert getattr(got, f) == getattr(want, f), f
    assert got.group_index.keys() == want.group_index.keys()
    assert all(torch.equal(got.group_index[k], want.group_index[k]) and got.group_index[k].dtype == torch.long
               for k in got.group_index)
    for b, m in want.ref_masks.items():
        g = got.ref_masks[b]
        assert all(np.array_equal(getattr(g, f), getattr(m, f)) for f in ("inhand", "pedestal", "floor"))
        assert torch.equal(got.body_obj_mask[b], want.body_obj_mask[b])
    assert got.contact_site_index is None
    cfg.contact_rew_scale, cfg.contact_site_ids = 1.0, [0]
    assert torch.equal(SF.reward_static(cfg, env, lambda b: "right_object").contact_site_index, torch.tensor([0]))


def _cv(env) -> SF.ContactView:
    c = env.data_wp.contact
    return SF.ContactView(dist=c.dist, geom=c.geom.reshape(-1, 2), worldid=c.worldid, nacon=env.data_wp.nacon)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a.view(torch.int32) if a.dtype ==
                                                                           torch.float32 else a,
                                                                           b.view(torch.int32) if b.dtype ==
                                                                           torch.float32 else b))


def test_quat_to_vel_bitwise(daid) -> None:
    """No zero-axis rows: the original's boolean compaction is the identity, so even CPU kernels see the same
    element positions and the results must be bitwise equal (incl. the angle > pi branch)."""
    g = torch.Generator().manual_seed(0)
    q = torch.randn(257, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    q[3] = torch.tensor([-0.2, 0.5, -0.6, 0.3])
    assert _same(SF.quat_to_vel(q), daid.math.quat_to_vel(q))
    assert _same(SF.quat_sub(q, q.flip(0), daid.math.mul_quat), daid.math.quat_sub(q, q.flip(0)))


def test_quat_to_vel_zero_rows(daid) -> None:
    """Zero-axis rows give exact zeros. Other rows: on CPU the original's compaction moves elements between the
    vectorised body and the scalar tail of `atan2` (last-bit differences, a CPU artefact; a CUDA elementwise kernel
    applies one device function per element, checked on the GPU by a separate probe), so 1e-6 here."""
    g = torch.Generator().manual_seed(1)
    q = torch.randn(257, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    q[::7, 1:] = 0.0
    q[::7, 0] = 1.0
    a, b = SF.quat_to_vel(q), daid.math.quat_to_vel(q)
    assert torch.equal(a[::7], torch.zeros_like(a[::7])) and torch.equal(b[::7], a[::7])
    assert torch.allclose(a, b, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("nacon", [0, 1, 17, 33])
def test_reward_bitwise_every_branch(daid, nacon: int) -> None:
    model = _model()
    rng = np.random.default_rng(265 + nacon)
    cfg = _config(daid, model)
    env = _env(model, rng, nacon)
    # make some valid contacts hand-object and object-pedestal touching
    c = env.data_wp.contact
    hand = int(np.flatnonzero(cfg._is_hand_geom.numpy())[0])
    obj = int(np.flatnonzero(cfg._is_right_object_geom.numpy())[0])
    ped = int(np.flatnonzero(cfg._is_right_pedestal_geom.numpy())[0])
    for i in range(min(nacon, 6)):
        c.geom.view(-1, 2)[i] = torch.tensor([hand, obj] if i % 2 else [obj, ped], dtype=torch.int32)
        c.dist[i] = -0.01 * (i + 1)
        c.worldid[i] = i % N
    st = _static(daid, cfg, env)
    cv = _cv(env)
    d = env.data_wp
    ref = (torch.randn(NQ), torch.randn(NV), torch.randn(9), torch.zeros(10), torch.zeros(10, 3))
    ref[0][-4:] = ref[0][-4:] / ref[0][-4:].norm()
    seen = set()
    for step in range(35):                                # warmup, both branches, neither, clamp past the end
        r_ref, i_ref = daid.M.get_reward(cfg, env, ref, sim_step=step)
        r_new, i_new = SF.reward_step(cfg, st, d.qpos, d.qvel, d.site_xpos, cv, ref, step, daid.math.mul_quat)
        assert _same(r_new, r_ref), step
        assert list(i_new) == list(i_ref)
        for k in i_ref:
            assert _same(i_new[k], i_ref[k]), (step, k)
        seen.add(st.ref_masks[env._side_object_body_id["right"]].state(step)[1:3])
    assert len(seen) >= 3


def test_penetration_matches_check_penetration_amax(daid) -> None:
    model = _model()
    rng = np.random.default_rng(7)
    cfg = _config(daid, model)
    for nacon in (0, 5, 39, 40):
        env = _env(model, rng, min(nacon, L - 3))
        env.data_wp.nacon[0] = nacon
        st = _static(daid, cfg, env)
        pen_r, wid_r = daid.M.check_penetration(cfg, env)
        a = torch.zeros(N).scatter_reduce_(0, wid_r, pen_r, reduce="amax")
        pen_n, wid_n = SF.penetration(st, _cv(env), N)
        b = torch.zeros(N).scatter_reduce_(0, wid_n, pen_n, reduce="amax")
        assert _same(a, b)


def test_pair_contact_worlds_bitwise(daid) -> None:
    model = _model()
    rng = np.random.default_rng(11)
    cfg = _config(daid, model)
    for nacon in (0, 3, 20):
        env = _env(model, rng, nacon)
        for a_mask, b_mask in ((cfg._is_hand_geom, cfg._is_right_object_geom),
                               (cfg._is_right_pedestal_geom, cfg._is_right_object_geom),
                               (cfg._is_left_object_geom, cfg._is_hand_geom)):
            ref = daid.M._pair_contact_worlds(env, a_mask, b_mask)
            new = SF.pair_contact_worlds(_cv(env), a_mask, b_mask, bool(a_mask.any()), bool(b_mask.any()), N)
            assert torch.equal(ref, new)
