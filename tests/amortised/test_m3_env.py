"""M3: environment v2 (known answers and mutations) on the toy scene through the transductive CPU
backend (`tests/transductive/toy.py`).

Known answers: a zero action on a clip whose warm start IS its reference tracks it exactly (observation errors at fp32
rounding, reward 2.0 per row = kappa (w_o + w_h + w_q) + w_c); the replay's rebuild equals the observation the policy
saw BITWISE (records and next records, episode ends inside the n-step window, batch-independent); futility never
ends a passing episode (adversarial sums at the boundary, against the scorer's own mean). Mutation targets: an
off-by-one rebuild row (record or next record), the envelope used at evaluation, the observation reading the wrong
reference row; the SE(3) frame mutation is `test_m3_se3.py`'s.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import trimesh

from fastdo.amortised import tracker_obs as O
from fastdo.amortised.contact_ref import GridSDF
from fastdo.amortised.replay import HostReplay
from fastdo.amortised.tracker_env import TipSpec, TrackerEnv, clip_table_entry
from tests.transductive.toy import toy_clip

CFG = O.TrackerConfig(k=5, horizon=4)
REF_ROWS = ("obj_p", "obj_q", "obj_v", "q_ref", "tips_ref", "dref", "kappa")


def _scorer_rows(entry: dict) -> dict:
    """The toy's reference is in DO AS I DO's reward convention (row s = state x_s); the scorer's (executed row r =
    x_{r+1} against reference row r) is one row earlier: drop the first reference row and the last control row."""
    return {k: (v[1:] if k in REF_ROWS else v[:-1] if k == "ctrl_ref" else v) for k, v in entry.items()}


def _tips(spec) -> TipSpec:
    import mujoco

    m = spec.ex.model
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    return TipSpec(wrist_body=bid("right_palm"), tip_body=(bid("right_finger"), bid("right_palm")),
                   tip_offset=np.array([[0.0, 0.0, -0.05], [0.04, 0.0, -0.01]]))


@pytest.fixture(scope="module")
def world():
    box = trimesh.creation.box((0.04, 0.04, 0.04))
    sdf = GridSDF.from_mesh(np.asarray(box.vertices), np.asarray(box.faces), resolution=24)
    pts = np.asarray(trimesh.sample.sample_surface_even(box, 64, seed=0)[0])[:8]
    specs = [toy_clip("a", T=80, W=20), toy_clip("b", T=60, W=15, ref_shift=0.03)]
    tips = [_tips(sp) for sp in specs]
    table = O.RefTable.stack([_scorer_rows(clip_table_entry(sp, t, sdf, pts))
                             for sp, t in zip(specs, tips, strict=True)])
    return specs, tips, sdf, table


def _env(world, world_clip, evaluation=False, table=None):
    specs, tips, sdf, tab = world
    return TrackerEnv(CFG, specs, table or tab, tips, [sdf, sdf], world_clip, evaluation=evaluation)


def test_zero_action_on_its_own_reference_tracks_exactly(world) -> None:
    env = _env(world, [0])
    rec = env.reset_all()
    tab = world[3]
    R = tab.R
    for step in range(15):
        a, _c = O.observe(tab, rec, CFG)
        if step > 0:                     # the start state x_0 is no executed row: its reference is clamped to row 0
            assert np.abs(a[0, :6]).max() <= 2e-5            # xi at the current reference row (fp32 qpos)
            assert np.abs(a[0, 12:12 + R]).max() <= 2e-5     # q - q_ref
        out = env.step(np.zeros((1, R)))
        assert out.end[0] is None
        assert out.reward[0] == pytest.approx(2.0 * out.rows[0], abs=1e-9)
        rec = out.record


def test_reward_known_values(world) -> None:
    tab = world[3]
    R = tab.R
    rows = np.arange(30, 33)
    i = tab.idx(np.zeros(3, np.int64), rows)
    q = np.concatenate([tab.q_ref[i], tab.obj_p[i], tab.obj_q[i]], 1)
    r, rn, th = O.row_rewards(tab, CFG, 0, rows, q, tab.tips_ref[i], tab.dref[i], np.zeros(3))
    np.testing.assert_allclose(r, 2.0, atol=1e-12)
    assert rn.max() <= 1e-12 and th.max() <= 1e-12
    k2 = dataclasses.replace(tab, kappa=np.full_like(tab.kappa, 0.25))
    r2, _, _ = O.row_rewards(k2, CFG, 0, rows, q, tab.tips_ref[i], tab.dref[i], np.full(3, 0.004))
    np.testing.assert_allclose(r2, 0.25 * 1.5 + 0.5 - 0.5 * (0.004 - 0.002) / 0.002, atol=1e-12)
    q_off = q.copy()
    q_off[:, R] += 0.05
    r3, rn3, _ = O.row_rewards(tab, CFG, 0, rows, q_off, tab.tips_ref[i], tab.dref[i], np.zeros(3))
    assert np.all(r3 < 2.0) and np.allclose(rn3, 0.05)
    assert O.step_penalty(CFG, np.full(R, 0.05), np.zeros(R))[()] < 0.0


def _rollout(world, world_clip, steps: int, seed: int, evaluation=False, table=None):
    """Run the env with random actions; store every transition and every direct observation."""
    env = _env(world, world_clip, evaluation, table)
    tab = table or world[3]
    rng = np.random.default_rng(seed)
    rec = env.reset_all()
    rep = HostReplay({k: v for k, v in env.fields.items()}, env.n, 1000, tab.R, 3, 0.97)
    direct, final_obs, ends = [], {}, []
    for s in range(steps):
        direct.append(O.observe(tab, rec, CFG))
        a = rng.normal(size=(env.n, tab.R)) * 2.0
        out = env.step(a)
        rep.add(rec, a.astype(np.float32), out.reward, out.term, out.trunc, final=out.final)
        for w in np.flatnonzero(out.term):
            one = {k: v[w:w + 1] for k, v in out.final.items()}
            final_obs[(s, int(w))] = O.observe(tab, one, CFG)
        ends.append(out.end)
        rec = out.record
    return rep, direct, final_obs, ends


def test_rebuild_is_bitwise_the_observation_the_policy_saw(world) -> None:
    rep, direct, final_obs, ends = _rollout(world, [0, 1, 1, 0], steps=40, seed=1)
    tab = world[3]
    assert sum(e is not None for row in ends for e in row) >= 4               # episode ends inside the run
    lo, hi = rep.sample_range()
    steps = np.repeat(np.arange(lo, hi), rep.n_worlds)
    worlds = np.tile(np.arange(rep.n_worlds), hi - lo)
    perm = np.random.default_rng(0).permutation(steps.size)                    # batch composition differs
    b = rep.gather(steps[perm], worlds[perm])
    obs, cobs = O.observe(tab, b["rec"], CFG)
    nobs, ncobs = O.observe(tab, b["next_rec"], CFG)
    j = b["jstar"].numpy()
    ended = (b["terminated"].numpy() > 0) | (b["truncated"].numpy() > 0)
    assert ended.sum() >= 4 and (~ended).sum() >= 4
    for n_, (s, w) in enumerate(zip(steps[perm], worlds[perm], strict=True)):
        da, dc = direct[s]
        assert np.array_equal(obs[n_], da[w]) and np.array_equal(cobs[n_], dc[w]), (s, w)
        e = s + int(j[n_])
        na, nc = (final_obs[(e, w)][0][0], final_obs[(e, w)][1][0]) if ended[n_] else (direct[e + 1][0][w],
                                                                                         direct[e + 1][1][w])
        assert np.array_equal(nobs[n_], na) and np.array_equal(ncobs[n_], nc), (s, w, int(j[n_]))
    one, _ = O.observe(tab, {k: v[:1] for k, v in b["rec"].items()}, CFG)         # a batch of one
    assert np.array_equal(one[0], obs[0])


def test_envelope_ends_training_but_never_evaluation(world) -> None:
    """The object reference is displaced 0.1 m in the welded warmup: training ends by the envelope, evaluation runs
    the clip to its end (only the executor's termination applies). Displaced 0.1 m after the release, the envelope no
    longer ends training (the post-release object term is dropped); futility does (its window mean
    exceeds 3 cm), and evaluation still runs to the clip end."""
    tab = world[3]
    a, W, n = int(tab.offset[0]), int(tab.W[0]), int(tab.n_rows[0])
    for lo, hi, train_end in ((1, W, "envelope"), (W + 1, n, "futile")):
        p = tab.obj_p.copy()
        p[a + lo:a + hi, 0] += 0.10
        moved = dataclasses.replace(tab, obj_p=p)
        for evaluation, expect in ((False, train_end), (True, "clip_end")):
            env = _env(world, [0], evaluation, moved)
            env.reset_all()
            ends = []
            for _ in range(20):
                out = env.step(np.zeros((1, tab.R)))
                ends.append(out.end[0])
                if out.end[0] is not None:
                    break
            assert ends[-1] == expect, (lo, evaluation, ends)


def test_futility_never_ends_a_passing_episode() -> None:
    """Against the scorer's own window mean: random error sequences (many near the boundary) and the exact boundary
    (constant 3 cm); the rule ends an episode only if its window mean exceeds 3 cm, and at the first row whose partial
    sum makes that certain."""
    cfg = O.TrackerConfig()
    W, T = 30, 400
    lo = W + cfg.grace
    rng = np.random.default_rng(7)
    seqs = [np.full(T, 0.03), np.full(T, 0.03 + 1e-12), np.r_[np.zeros(lo), np.full(T - lo, 0.03)]]
    for _ in range(300):
        base = rng.uniform(0.0, 0.06, T)
        seqs.append(base * (0.03 / base[lo:].mean()) * rng.choice([0.999999, 1.0, 1.000001, 0.9, 1.1]))
    fired = passed = 0
    for e in seqs:
        st = O.TermState(1)
        used, reason = st.rows(0, cfg, np.arange(T), np.zeros(T), np.zeros(T), e, W, T, evaluation=False)
        mean = float(e[lo:T].mean())                                           # the scorer's p_err
        if mean <= cfg.fut_pos:
            passed += 1
            assert reason == "clip_end" and used == T, (mean, reason, used)
        if reason == "futile":
            fired += 1
            assert mean > cfg.fut_pos
            assert e[lo:used].sum() > cfg.fut_pos * (T - lo)
    assert fired >= 50 and passed >= 50


def test_record_size_and_widths(world) -> None:
    f = O.record_fields(35, 34, 28, 5)
    per = sum(np.dtype(dt).itemsize * int(np.prod(shape)) for shape, dt in f.values())
    assert per == 412
    rep = HostReplay(f, 2, 10, 28, 5, 0.97)
    assert rep.nbytes_per_transition == 536
    env = _env(world, [0, 1])
    a, c = O.observe(world[3], env.reset_all(), CFG)
    assert (a.shape[1], c.shape[1]) == O.obs_dims(CFG, world[3].R, world[3].F, world[3].P)
    assert a.dtype == np.float32 and np.isfinite(c).all()


def test_kappa_rows_follow_the_support_flags_on_the_reference_grid() -> None:
    """kappa: 0.25 on unsupported frames, 1 elsewhere, sampled like the reference (frame 0 held through
    the warmup, linear between frames, the last frame repeated)."""
    from fastdo.amortised.tracker_env import KAPPA_UNSUPPORTED, kappa_rows

    valid = np.array([0, 1, 1, 0, 0, 1, 1, 1], bool)
    W, k = 10, 7
    rows = kappa_rows(valid, 0.0333, 0.005, W, W + len(valid) * k + 5)
    assert KAPPA_UNSUPPORTED == 0.25
    assert np.all(rows[:W + 1] == 0.25)                             # frame 0 is unsupported: held through warmup
    x = np.linspace(0.0, len(valid) - 1, len(valid) * k)
    np.testing.assert_allclose(rows[W:W + len(x)], np.interp(x, np.arange(len(valid)), np.where(valid, 1.0, 0.25)))
    assert rows[-1] == 1.0 and rows.min() == 0.25 and rows.max() == 1.0
