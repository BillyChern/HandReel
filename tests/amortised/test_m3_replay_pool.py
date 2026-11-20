"""M3: the replay's vectorised final-record pool (the per-world Python loop in `HostReplay.add`
would not scale to M9's 8k worlds). A small pool gives exactly the default (dense) pool's batches, and a pool too
small for the episode-end rate raises instead of handing back another episode's final record."""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.amortised.replay import HostReplay

FIELDS = {"x": ((3,), np.float32), "row": ((), np.int32)}


def _fill(rep: HostReplay, steps: int, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for t in range(steps):
        rec = {"x": rng.normal(size=(n, 3)).astype(np.float32), "row": np.full(n, t, np.int32)}
        done = rng.random(n) < 0.1
        term = (done & (rng.random(n) < 0.5)).astype(np.float32)
        trunc = (done & (term == 0)).astype(np.float32)
        # a final record names its own (step, world): row = -(1000 t + w) - 1
        fin = {"x": rng.normal(size=(n, 3)).astype(np.float32), "row": -(1000 * t + np.arange(n, dtype=np.int32)) - 1}
        rep.add(rec, rng.normal(size=(n, 2)).astype(np.float32), rng.normal(size=n).astype(np.float32), term, trunc,
                final=fin)


def test_small_pool_equals_the_dense_pool() -> None:
    n, cap = 16, 40
    dense = HostReplay(FIELDS, n, cap, 2, 3, 0.9)
    small = HostReplay(FIELDS, n, cap, 2, 3, 0.9, final_capacity=cap * n // 4)
    _fill(dense, 300, n)
    _fill(small, 300, n)
    rng = np.random.default_rng(1)
    steps, worlds = dense.sample_indices(rng, 4000)
    a, b = dense.gather(steps, worlds), small.gather(steps, worlds)
    ended = (a["terminated"].numpy() > 0) | (a["truncated"].numpy() > 0)
    assert ended.sum() > 100
    # an ended window's next record is the final record of ITS world at its end step s + j*
    want = -(1000 * (steps + a["jstar"].numpy()) + worlds) - 1
    assert np.array_equal(b["next_rec"]["row"][ended], want[ended])
    for k in FIELDS:
        assert np.array_equal(a["next_rec"][k], b["next_rec"][k]), k
        assert np.array_equal(a["rec"][k], b["rec"][k]), k


def test_pool_overflow_raises() -> None:
    n, cap = 16, 40
    rep = HostReplay(FIELDS, n, cap, 2, 3, 0.9, final_capacity=n)       # one end per world for the whole ring
    with pytest.raises(ValueError, match="overflow"):
        _fill(rep, 300, n)
