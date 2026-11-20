"""M9 (P1) known answers for the device terms (`fastdo.amortised.tracker_dev`) against their host
specification (`tracker_obs`, `se3`, `contact_ref.GridSDF`), on torch's CPU device in float64: observation, reward,
step penalty, SDF (inside the grid and in the far field, clips with different grid sizes), SE(3) error and the
training termination, on random states of the toy clips. Mutation targets: the look-ahead offsets of
the observation, the SDF far-field branch, the envelope counter reset, the futility window start."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from fastdo.amortised import tracker_dev as D
from fastdo.amortised import tracker_obs as O
from fastdo.amortised.contact_ref import GridSDF
from fastdo.amortised.se3 import object_error
from tests.amortised import m5_toy

CFG = m5_toy.CFG


@pytest.fixture(scope="module")
def toy():
    inp = m5_toy.inputs()
    table = O.RefTable.stack(list(inp.entries))
    return inp, table, D.DevTable.from_host(table, "cpu")


def _quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _records(table: O.RefTable, rng, n: int, nq: int, nv: int) -> dict[str, np.ndarray]:
    R, F = table.R, table.F
    f = O.record_fields(nq, nv, R, F)
    clip = rng.integers(0, len(table.T), n)
    qpos = rng.normal(scale=0.05, size=(n, nq))
    qpos[:, R + 3:R + 7] = _quat(rng, n)
    raw = {"qpos": qpos, "qvel": rng.normal(size=(n, nv)), "wrist": np.concatenate([rng.normal(scale=0.05, size=(n, 3)),
                                                                               _quat(rng, n)], 1),
           "tips": rng.normal(scale=0.05, size=(n, F, 3)), "tip_sdf": rng.normal(scale=0.02, size=(n, F, 4)),
           "contact": rng.random((n, 3)), "prev_delta": rng.normal(scale=0.1, size=(n, R)),
           "alpha": rng.random(n), "clip": clip, "row": rng.integers(0, table.T[clip] + 3),
           "tip_force": rng.random((n, F)) * 5, "obj_vel": rng.normal(size=(n, 6))}
    return {k: np.asarray(raw[k]).astype(dt) for k, (_, dt) in f.items()}


def test_observe_t_equals_the_host_observation(toy) -> None:
    inp, table, dt = toy
    m = inp.specs[0].ex.model
    rec = _records(table, np.random.default_rng(0), 64, m.nq, m.nv)
    a_h, c_h = O.observe(table, rec, CFG)
    a_d, c_d = D.observe_t(dt, {k: torch.as_tensor(v) for k, v in rec.items()}, CFG)
    assert a_d.dtype == torch.float32 and a_d.shape == a_h.shape and c_d.shape == c_h.shape
    assert np.allclose(a_d.numpy(), a_h, rtol=1e-6, atol=1e-6)
    assert np.allclose(c_d.numpy(), c_h, rtol=1e-6, atol=1e-6)
    # every float32 component equal or one ulp apart (float64 libm differences only)
    assert (np.abs(a_d.numpy() - a_h) <= np.spacing(np.abs(a_h)) * 2).mean() > 0.999


def test_row_rewards_and_step_penalty_equal_the_host(toy) -> None:
    inp, table, dt = toy
    rng = np.random.default_rng(1)
    n, F, R = 50, table.F, table.R
    nq = inp.specs[0].ex.model.nq
    for c in range(len(table.T)):
        rows = rng.integers(0, int(table.T[c]), n)
        q = rng.normal(scale=0.05, size=(n, nq))
        q[:, R + 3:R + 7] = _quat(rng, n)
        tips, d_tip, pen = rng.normal(scale=0.05, size=(n, F, 3)), rng.normal(scale=0.02, size=(n, F)), rng.random(n) * 0.01
        h = O.row_rewards(table, CFG, c, rows, q, tips, d_tip, pen)
        t = lambda x: torch.as_tensor(x)
        d = D.row_rewards_t(dt, CFG, torch.full((n,), c), t(rows), t(q), t(tips), t(d_tip), t(pen))
        for a, b in zip(h, d, strict=True):
            assert np.allclose(b.numpy(), a, rtol=1e-12, atol=1e-12)
    dl, dp = rng.normal(size=(7, R)), rng.normal(size=(7, R))
    assert np.allclose(D.step_penalty_t(CFG, torch.as_tensor(dl), torch.as_tensor(dp)).numpy(),
                       O.step_penalty(CFG, dl, dp), rtol=1e-12)


def test_object_error_t_equals_se3() -> None:
    rng = np.random.default_rng(2)
    n = 200
    p, pr = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    q, qr = _quat(rng, n), _quat(rng, n)
    qr[:20] = q[:20]                                       # small-angle branch
    h = object_error(p, q, pr, qr)
    d = D.object_error_t(*(torch.as_tensor(x) for x in (p, q, pr, qr)))
    for a, b in zip(h, d, strict=True):
        assert np.allclose(b.numpy(), a, rtol=1e-10, atol=1e-12)


def test_dev_sdf_equals_grid_queries_inside_and_in_the_far_field() -> None:
    import trimesh

    box = trimesh.creation.box((0.04, 0.04, 0.04))
    s1 = GridSDF.from_mesh(np.asarray(box.vertices), np.asarray(box.faces), resolution=16)
    ball = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
    s2 = GridSDF.from_mesh(np.asarray(ball.vertices), np.asarray(ball.faces), resolution=24)
    dev = D.DevSDF.from_grids([s1, s2], "cpu")
    rng = np.random.default_rng(3)
    pts = rng.normal(scale=0.06, size=(40, 5, 3))           # many beyond the 5 cm pad: the far-field branch
    clip = rng.integers(0, 2, 40)
    phi, grad = dev.query(torch.as_tensor(clip), torch.as_tensor(pts))
    far = 0
    for b in range(40):
        g = (s1, s2)[clip[b]]
        ph, gr = g.query(pts[b])
        assert np.allclose(phi[b].numpy(), ph, rtol=1e-12, atol=1e-12)
        assert np.allclose(grad[b].numpy(), gr, rtol=1e-10, atol=1e-10)
        u = (pts[b] - g.origin) / g.step
        far += int(np.any((u < 0) | (u > g.values.shape[0] - 1), axis=-1).sum())
    assert far > 20                                          # informative: the far field was exercised


def test_term_dev_equals_term_state_row_by_row() -> None:
    rng = np.random.default_rng(4)
    cfg = dataclasses.replace(CFG, env_rows=3, grace=2, fut_pos=0.004)
    # The envelope's object term acts only in the welded warmup (rows < W; the post-release term is
    # dropped), so W is long enough here for the envelope to fire (it was 3-8 rows when the term covered every row)
    n, W, T = 6, np.array([20, 20, 15, 15, 12, 12]), np.array([40, 40, 30, 30, 25, 25])
    for evaluation in (False, True):
        host = O.TermState(n)
        dev = D.TermDev(n, "cpu")
        live = np.ones(n, bool)
        reasons = {}
        for r in range(45):
            rn, th = rng.random(n) * 0.09, rng.random(n) * 0.9
            pe = rng.random(n) * 0.05
            want = np.zeros(n, np.int64)
            for i in np.flatnonzero(live):
                used, why = host.rows(i, cfg, np.array([r]), rn[i:i + 1], th[i:i + 1], pe[i:i + 1], int(W[i]),
                                      int(T[i]), evaluation)
                code = {None: 0, "clip_end": D.END_CLIP, "envelope": D.END_ENVELOPE, "futile": D.END_FUTILE}[why]
                want[i] = code
            t = lambda x: torch.as_tensor(x)
            got = dev.row(cfg, t(live), torch.full((n,), r), t(rn), t(th), t(pe), t(W), t(T), evaluation).numpy()
            assert np.array_equal(got, want), (evaluation, r, got, want)
            for i in np.flatnonzero(want):
                reasons[int(want[i])] = reasons.get(int(want[i]), 0) + 1
            live &= want == 0
            assert np.array_equal(dev.env_count.numpy()[live], host.env_count[live])
        if not evaluation:
            assert D.END_ENVELOPE in reasons and D.END_FUTILE in reasons, reasons
        else:
            assert set(reasons) == {D.END_CLIP}, reasons
