"""M5: success store and closed-loop CPU certification (known answers and mutations) on the
toy clips through environment v2 (`tests/amortised/m5_toy.py`).

Known answers: certification of a stored candidate is bit-identical across two re-executions and equals the training
rollout (batch-of-one acting on the CPU executor); the trainer stores only alpha = 0 episodes that started at row 0.
Mutation targets: an alpha > 0 success stored; an RSI (start row > 0) success stored; a certificate produced from
other weights than the stored versions accepted; a certificate accepted although its re-execution did not run the
clip to its end (an executor termination, or the stored versions running out) under a scorer that passes.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest
import torch

from fastdo.amortised.acting import (
    PolicyVersions,
    VersionError,
    act,
    episode_seed,
    state_bytes,
    step_noise,
)
from fastdo.amortised.sac import Actor, SacConfig
from fastdo.amortised.success import Candidate, SuccessStore, certify
from fastdo.amortised.trainer import Trainer
from tests.amortised import m5_toy


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One toy run with the assist on and RSI at 50%: candidates of every kind are offered."""
    out = tmp_path_factory.mktemp("m5run")
    t = Trainer(m5_toy.config(max_steps=150, rsi_prob=0.3), m5_toy.inputs(), out)
    t.run()
    return t


def _cand(**kw) -> Candidate:
    base = {"task": "a", "clip": 0, "episode": 7, "noise_seed": 11, "start_row": 0, "alpha": 0.0, "versions": ("x",),
            "rows": 80, "traj_sha": "s", "score": {"success": True}}
    return Candidate(**{**base, **kw})


def test_store_refuses_an_assisted_success(tmp_path) -> None:
    store = SuccessStore(tmp_path)
    vers = PolicyVersions()
    assert store.offer(_cand(alpha=0.3), vers) == (False, "assist")
    assert store.offer(_cand(alpha=1e-9), vers) == (False, "assist")
    assert not (tmp_path / "a").exists()


def test_store_refuses_an_rsi_success(tmp_path) -> None:
    store = SuccessStore(tmp_path)
    assert store.offer(_cand(start_row=40), PolicyVersions()) == (False, "rsi")
    assert not (tmp_path / "a").exists()


def test_store_is_idempotent_and_refuses_non_successes(tmp_path) -> None:
    actor = Actor(5, 2, SacConfig(actor_hidden=8))
    vers = PolicyVersions()
    sha = vers.snapshot(actor)
    store = SuccessStore(tmp_path)
    assert store.offer(_cand(versions=(sha, sha), score={"success": False}), vers) == (False, "not_success")
    c = _cand(versions=(sha, sha))
    assert store.offer(c, vers) == (True, "stored")
    assert store.offer(c, vers) == (True, "exists")
    assert store.candidates() == [c]
    with pytest.raises(RuntimeError):
        store.offer(dataclasses.replace(c, rows=79), vers)


def test_trainer_stores_only_alpha0_row0_successes(run) -> None:
    offers = run.stats["offers"]
    assert offers.get("assist", 0) >= 1 and offers.get("rsi", 0) >= 1 and offers.get("stored", 0) >= 2, offers
    stored = run.store.candidates()
    assert stored and all(c.alpha == 0.0 and c.start_row == 0 for c in stored)
    log = [json.loads(x) for x in run.log_path.read_text().splitlines()]
    eps = {(r["task"], r["episode"]) for r in log if r["offer"] in ("stored", "exists")}
    assert eps == {(c.task, c.episode) for c in stored}
    assert any(r["start_row"] > 0 for r in log) and any(r["alpha"] > 0 for r in log)


def _recert(run, cand: Candidate, versions: PolicyVersions | None = None):
    i = cand.clip
    inp = run.inp
    return certify(cand, inp.specs[i], inp.tips[i], inp.sdfs[i], inp.entries[i], run.cfg.tracker,
                   copy.deepcopy(run.agent.actor), versions or run.store.versions(), inp.scorers[i])


def test_certification_is_bit_identical_across_two_reexecutions(run) -> None:
    for cand in run.store.candidates()[:3]:
        a, b = _recert(run, cand), _recert(run, cand)
        assert a.certified and b.certified
        assert a.sha == b.sha and np.array_equal(a.qpos, b.qpos) and np.array_equal(a.ctrl, b.ctrl)
        assert a.sha == cand.traj_sha                    # = the training rollout (batch-of-one acting on CPU)
        assert a.qpos.shape[0] == cand.rows


def test_certificate_from_different_weights_is_refused(run, tmp_path) -> None:
    """The certifier loads every version by its sha256: other weights stored under a candidate's version name are
    refused, never executed."""
    cand = run.store.candidates()[0]
    snaps = tmp_path / "snapshots"
    run.store.versions().write(cand.versions, snaps)
    other = copy.deepcopy(run.agent.actor)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.01)
    for sha in set(cand.versions):
        (snaps / f"{sha}.bin").write_bytes(state_bytes(other))
    with pytest.raises(VersionError):
        _recert(run, cand, PolicyVersions(snaps))


def test_noise_is_counter_based_and_acting_is_batch_invariant() -> None:
    s = episode_seed(3, 17)
    assert torch.equal(step_noise(s, 5, 7), step_noise(s, 5, 7))
    assert not torch.equal(step_noise(s, 5, 7), step_noise(s, 6, 7))
    assert episode_seed(3, 17) != episode_seed(3, 18) != episode_seed(4, 17)
    actor = Actor(9, 3, SacConfig(actor_hidden=16))
    obs = np.random.default_rng(0).normal(size=(6, 9)).astype(np.float32)
    nz = torch.randn(6, 3, generator=torch.Generator().manual_seed(0))
    full = act(actor, obs, nz)
    for i in range(6):
        assert np.array_equal(act(actor, obs[i:i + 1], nz[i:i + 1])[0], full[i])


def _always_pass(q: np.ndarray) -> dict:
    return {"success": True}


def test_a_terminated_reexecution_is_not_certified_even_if_the_scorer_passes(run, monkeypatch) -> None:
    """certify's rule: certified only if the re-execution ran the clip to its END. The toy scorer passes partial
    trajectories, so an executor termination (forced here at the 3rd policy step) must still refuse."""
    from fastdo.amortised import success as S

    class Terminating(S.TrackerEnv):
        def step(self, actions):
            out = super().step(actions)
            self._n = getattr(self, "_n", 0) + 1
            if self._n == 3 and out.end[0] is None:
                out.end[0] = "drop"
            return out

    monkeypatch.setattr(S, "TrackerEnv", Terminating)
    cert = _recert_with(run, run.store.candidates()[0], _always_pass)
    assert cert.end == "drop" and not cert.certified and cert.reason == "drop", cert.summary()
    assert cert.score.get("success") is True                 # the scorer passed; the END refused it


def test_versions_running_out_before_the_clip_end_is_not_certified(run) -> None:
    cand = run.store.candidates()[0]
    short = dataclasses.replace(cand, versions=cand.versions[:2])
    cert = _recert_with(run, short, _always_pass)
    assert not cert.certified and cert.reason == "versions_exhausted" and cert.qpos.shape[0] < cand.rows


def _recert_with(run, cand: Candidate, scorer):
    i = cand.clip
    inp = run.inp
    return certify(cand, inp.specs[i], inp.tips[i], inp.sdfs[i], inp.entries[i], run.cfg.tracker,
                   copy.deepcopy(run.agent.actor), run.store.versions(), scorer)
