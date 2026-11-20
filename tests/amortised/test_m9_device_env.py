"""M9 (P1) known answers for the device environment on the CPU: `DeviceTrackerEnv`
with the `cpu-pool` backend (CPU MuJoCo per-clip pools, the device terms of `tracker_dev` on torch's CPU device)
executes EXACTLY `TrackerEnv`'s rows: the same controls and states bit for bit, the same episode ends and rows, the
same rewards (float64 terms; float32 output) and the same records, on the toy clips, with the assist on some
worlds, in training and in evaluation mode, and with RSI starts from the reset hook. Then `DeviceTrainer` (the
device trainer loop) runs end to end on the same backend: acting with the hash noise, learner updates, the store and
the CPU certificate of a stored candidate regenerate the same noise.

Mutation targets (batch F): the rows after an executor end counted; the record taken after the last LOCK-STEP row
instead of the last executed row; the term check after the executor's end; the step penalty dropped."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from fastdo.amortised import tracker_obs as O
from fastdo.amortised.tracker_env import TrackerEnv
from fastdo.amortised.tracker_gpu import DeviceTrackerEnv, grouped_world_clip
from tests.amortised import m5_toy

CFG = m5_toy.CFG


@pytest.fixture(scope="module")
def toy():
    inp = m5_toy.inputs()
    return inp, O.RefTable.stack(list(inp.entries))


# a tight envelope and futility bound, and clip b's drop height just above its start: every end reason occurs,
# including training ends followed by an executor end inside the same step (the record row then moves)
TIGHT = dataclasses.replace(CFG, env_pos=0.004, env_rows=3, fut_pos=0.002)


def _dropping(specs):
    sp = specs[1]
    z0 = float(sp.ex.q0[sp.ex.obj_qadr + 2])
    return [specs[0], dataclasses.replace(sp, terminate_z=z0 + 1e-3)]


def _pair(toy, world_clip, evaluation=False, hook=None, cfg=CFG, drop=False):
    inp, table = toy
    specs = _dropping(inp.specs) if drop else inp.specs
    cpu = TrackerEnv(cfg, specs, table, inp.tips, inp.sdfs, world_clip, evaluation=evaluation,
                     reset_hook=hook)
    dev = DeviceTrackerEnv(cfg, specs, table, inp.tips, inp.sdfs, world_clip, backend="cpu-pool",
                           device="cpu", evaluation=evaluation, reset_hook=hook)
    return cpu, dev


def _same_records(a: dict, b: dict, where: str) -> None:
    for k in a:
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        assert x.shape == y.shape, (where, k)
        if k in ("tip_sdf", "tips", "wrist"):
            # device float64 SDF/FK terms equal the host's to a few ulp; after the fp16 cast at most one half-ulp step
            assert np.allclose(x, y, rtol=2e-3, atol=1e-6), (where, k, np.abs(x - y).max())
        else:
            assert np.array_equal(x, y), (where, k, np.abs(x - y).max())


def _run(cpu, dev, steps: int, seed: int, alpha: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if alpha is not None:
        cpu.next_alpha[:] = alpha
        dev.next_alpha[:] = alpha
    _same_records(cpu.reset_all(), dev.reset_all(), "reset")
    ends: dict[str, int] = {}
    for s in range(steps):
        a = rng.normal(scale=1.5, size=(cpu.n, cpu.R)).astype(np.float32)
        oc, od = cpu.step(a), dev.step(a)
        assert oc.end == od.end, (s, oc.end, od.end)
        assert np.array_equal(oc.rows, od.rows), s
        assert np.array_equal(oc.term, od.term) and np.array_equal(oc.trunc, od.trunc)
        assert np.allclose(oc.reward, od.reward, rtol=1e-6, atol=1e-6), (s, np.abs(oc.reward - od.reward).max())
        for i in range(cpu.n):
            assert np.array_equal(oc.qpos_rows[i], od.qpos_rows[i]), (s, i)
            assert np.array_equal(oc.ctrl_rows[i], od.ctrl_rows[i]), (s, i)
        _same_records(oc.record, od.record, f"record {s}")
        _same_records(oc.final, od.final, f"final {s}")
        for e in oc.end:
            if e is not None:
                ends[e] = ends.get(e, 0) + 1
    return ends


def test_device_env_executes_the_cpu_envs_rows_in_training(toy) -> None:
    wc = grouped_world_clip(6, 2)
    cpu, dev = _pair(toy, wc)
    ends = _run(cpu, dev, 60, seed=1)
    assert ends.get("clip_end", 0) >= 6, ends        # informative: episodes ran to the end and were reset


def _rsi_hook(inp, wc, tag, starts):
    rng = np.random.default_rng(11)

    def hook(i):
        sp = inp.specs[wc[i]]
        row = int(rng.integers(sp.W, sp.T - 10)) if rng.random() < 0.5 else 0
        starts[tag].append(row)
        return wc[i], 0.0, row
    return hook


def test_device_env_matches_on_every_end_reason(toy) -> None:
    """Training ends, drops (mid-step after RSI starts that are not aligned to k) and clip ends, all matching."""
    wc = grouped_world_clip(6, 2)
    inp, _ = toy
    starts: dict[str, list[int]] = {"c": [], "d": []}
    cpu, _ = _pair(toy, wc, cfg=TIGHT, drop=True, hook=_rsi_hook(inp, wc, "c", starts))
    _, dev = _pair(toy, wc, cfg=TIGHT, drop=True, hook=_rsi_hook(inp, wc, "d", starts))
    ends = _run(cpu, dev, 80, seed=5)
    assert {"envelope", "drop", "clip_end"} <= set(ends), ends
    assert any(r % CFG.k for r in starts["d"] if r > 0), starts["d"]      # mid-step executor ends occur


def test_a_training_end_and_an_executor_end_on_the_same_row_name_the_training_end(toy) -> None:
    """Envelope (|rho| > 0 for 5 rows) and the drop rule (block ends every 5 rows, drop height above the start) both
    fire on row 4 of every episode of clip b: `TrackerEnv` names the training end, and so must the device env."""
    inp, table = toy
    cfg = dataclasses.replace(CFG, env_pos=-1.0, env_rows=5)
    sb = inp.specs[1]
    specs = [inp.specs[0], dataclasses.replace(sb, ctrl_steps=5, terminate_z=float(sb.ex.q0[sb.ex.obj_qadr + 2]) + 1.0)]
    wc = grouped_world_clip(4, 2)
    cpu = TrackerEnv(cfg, specs, table, inp.tips, inp.sdfs, wc)
    dev = DeviceTrackerEnv(cfg, specs, table, inp.tips, inp.sdfs, wc, backend="cpu-pool", device="cpu")
    ends = _run(cpu, dev, 3, seed=6)
    assert ends == {"envelope": 12}, ends


def test_device_env_matches_with_the_assist_on_some_worlds(toy) -> None:
    wc = grouped_world_clip(6, 2)
    cpu, dev = _pair(toy, wc)
    _run(cpu, dev, 40, seed=2, alpha=np.array([1.0, 0.0, 0.5, 0.9, 0.0, 0.3]))


def test_device_env_matches_in_evaluation_mode(toy) -> None:
    wc = grouped_world_clip(4, 2)
    cpu, dev = _pair(toy, wc, evaluation=True)
    ends = _run(cpu, dev, 30, seed=3)
    assert set(ends) <= {"clip_end", "drop", "invalid"}, ends
    assert ends.get("clip_end", 0) >= 1, ends


def test_device_env_matches_with_rsi_starts_from_the_hook(toy) -> None:
    wc = grouped_world_clip(4, 2)
    inp, table = toy
    starts: dict[str, list[int]] = {"c": [], "d": []}

    def make_hook(tag):
        rng = np.random.default_rng(7)

        def hook(i):
            sp = inp.specs[wc[i]]
            row = int(rng.integers(sp.W, sp.T - 10)) if rng.random() < 0.5 else 0
            starts[tag].append(row)
            return wc[i], float(rng.random() < 0.5), row
        return hook

    cpu = TrackerEnv(CFG, inp.specs, table, inp.tips, inp.sdfs, wc, reset_hook=make_hook("c"))
    dev = DeviceTrackerEnv(CFG, inp.specs, table, inp.tips, inp.sdfs, wc, backend="cpu-pool", device="cpu",
                           reset_hook=make_hook("d"))
    _run(cpu, dev, 50, seed=4)
    assert starts["c"] == starts["d"]
    assert sum(r > 0 for r in starts["d"]) >= 3, starts["d"]      # informative: several RSI episodes ran


def test_device_env_refuses_ungrouped_worlds_and_a_rebinding_hook(toy) -> None:
    inp, table = toy
    with pytest.raises(ValueError, match="grouped"):
        DeviceTrackerEnv(CFG, inp.specs, table, inp.tips, inp.sdfs, [0, 1, 0, 1], backend="cpu-pool", device="cpu")
    dev = DeviceTrackerEnv(CFG, inp.specs, table, inp.tips, inp.sdfs, [0, 0, 1, 1], backend="cpu-pool",
                           device="cpu", reset_hook=lambda i: (1 - [0, 0, 1, 1][i], 0.0, 0))
    with pytest.raises(ValueError, match="rebind"):
        dev.reset_all()


def test_grouped_world_clip_is_contiguous_and_balanced() -> None:
    wc = grouped_world_clip(8, 3)
    assert wc == sorted(wc) and set(wc) == {0, 1, 2}
    counts = [wc.count(c) for c in range(3)]
    assert max(counts) - min(counts) <= 1
    with pytest.raises(ValueError):
        grouped_world_clip(2, 3)


def test_device_trainer_runs_end_to_end_and_its_certificates_regenerate_the_hash_noise(tmp_path) -> None:
    """Device acting (one batched pass, hash noise) certified by the CPU certifier (batch-of-one, the candidate's
    noise regenerated from its seed): on the same simulator the certificate follows the training rollout to float32
    acting round-off (~1e-8 here); with any other noise it would leave it by the residual scale (cm)."""
    from fastdo.amortised.success import SuccessStore
    from fastdo.amortised.trainer_dev import DeviceTrainer

    inp = m5_toy.inputs()
    cfg = m5_toy.config(max_steps=80, threads=1, assist=False)
    t = DeviceTrainer(cfg, inp, tmp_path / "run", backend="cpu-pool", device="cpu")
    keep: dict[int, np.ndarray] = {}
    t.on_episode = lambda ep, reason, score: keep.__setitem__(ep.episode, np.concatenate(ep.qpos))
    s = t.run()
    assert s["updates"] > 0 and s["rows"] > 0
    assert s["offers"].get("stored", 0) >= 3, s           # the toy scorer passes about half the full episodes
    cands = SuccessStore(tmp_path / "run" / "successes").candidates()
    assert cands and all(c.noise == "hash" for c in cands)
    certs = sorted((tmp_path / "run" / "certificates").rglob("*.npz"))
    assert len(certs) == len(cands) == s["certificates"]
    for p in certs:
        q, tr = np.load(p)["qpos"], keep[int(p.stem)]
        assert q.shape == tr.shape
        assert np.abs(q - tr).max() < 1e-5, (p.stem, np.abs(q - tr).max())
    assert s["certified"] == len(certs)
    t.save_checkpoint()
    with pytest.raises(RuntimeError, match="does not resume"):
        DeviceTrainer(cfg, inp, tmp_path / "run", backend="cpu-pool", device="cpu").resume()


def test_hash_noise_is_a_pure_function_of_seed_step_and_dim() -> None:
    from fastdo.amortised.acting import noise_for, noise_hash

    seeds = np.array([1, 2, 3, 2**62 + 5], np.int64)
    steps = np.array([0, 7, 7, 1000], np.int64)
    a = noise_hash(seeds, steps, 28)
    b = np.stack([noise_hash(seeds[i:i + 1], steps[i:i + 1], 28)[0] for i in range(4)])
    assert a.dtype == np.float32 and np.array_equal(a, b)
    assert np.array_equal(noise_for("hash", 3, 7, 28).numpy(), a[2])
    assert not np.array_equal(a[1], a[2])                    # the seed matters
    assert not np.array_equal(noise_hash(seeds[:1], np.array([7]), 28), noise_hash(seeds[:1], np.array([8]), 28))
    assert abs(float(noise_hash(np.arange(4000), np.zeros(4000, np.int64), 28).std()) - 1.0) < 0.02
    with pytest.raises(ValueError):
        noise_for("other", 0, 0, 3)
    from fastdo.amortised.acting import step_noise

    assert torch.equal(noise_for("torch", 3, 7, 28), step_noise(3, 7, 28))
