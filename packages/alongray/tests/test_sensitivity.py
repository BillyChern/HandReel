# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Window- and frame-scope alignments and the pure-gauge sensitivity table (alongray.sensitivity)."""

from __future__ import annotations

import numpy as np
import pytest

from alongray import ALIGNMENTS, aligned_errors, gauge_sensitivity


def _hand(t: int = 40, n: int = 21, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tt = np.linspace(0.0, 1.0, t)
    c = np.stack([0.1 * np.sin(2 * np.pi * tt), 0.05 * np.cos(2 * np.pi * tt), 0.6 + 0.15 * tt], axis=1)
    shape = rng.normal(scale=0.04, size=(n, 3))
    rots = np.stack([[[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]] for a in 0.8 * tt])
    return c[:, None, :] + np.einsum("tij,nj->tni", rots, shape)


def test_pure_gauge_known_values() -> None:
    p = _hand()
    lam = 1.25
    t = gauge_sensitivity(p, (lam,), root=0)
    key = repr(lam)
    assert t["none"][key] == pytest.approx((lam - 1) * float(np.mean(np.linalg.norm(p.reshape(-1, 3), axis=1))),
                                           rel=1e-12)
    assert t["root_frame"][key] == pytest.approx((lam - 1) * float(np.linalg.norm(p - p[:, :1], axis=2).mean()),
                                                 rel=1e-12)
    for a in ("scale", "sim3", "sim3_frame"):
        assert t[a][key] == pytest.approx(0.0, abs=1e-12)
    d = p.reshape(-1, 3) - p.reshape(-1, 3).mean(0)
    assert t["translation"][key] == pytest.approx((lam - 1) * float(np.mean(np.linalg.norm(d, axis=1))), rel=1e-12)
    static = np.tile([[[0.0, 0.0, 0.6]]], (5, 1, 1))
    assert gauge_sensitivity(static, (lam,))["translation"][key] == pytest.approx(0.0, abs=1e-15)


def test_scale_and_sim3_residuals_are_gauge_free() -> None:
    rng = np.random.default_rng(2)
    c = _hand()[:, 0]
    p = 1.3 * c + rng.normal(scale=0.01, size=c.shape)
    a, b = aligned_errors(p, c), aligned_errors(2.0 * p, c)
    assert a["scale"]["rms"] == pytest.approx(b["scale"]["rms"], rel=1e-12)
    assert a["sim3"]["rms"] == pytest.approx(b["sim3"]["rms"], rel=1e-9)
    assert a["root_frame"] is None and a["sim3_frame"] is None and set(ALIGNMENTS) <= set(a)


@pytest.mark.parametrize("lateral", [0.0, 1e-9])
def test_collinear_and_nearly_collinear_centres_have_no_sim3(lateral: float) -> None:
    c = np.column_stack([lateral * np.sin(np.arange(10.0)), np.zeros(10), np.linspace(0.5, 0.8, 10)])
    assert aligned_errors(1.1 * c, c)["sim3"] is None


def test_mask_nan_and_refusals() -> None:
    p = _hand(t=12)
    q = p.copy()
    q[2, 4] = np.nan
    v = np.ones(12, bool)
    v[7] = False
    assert aligned_errors(q, p, v)["n_frames"] == 10
    with pytest.raises(ValueError, match="frame mask"):
        aligned_errors(p, p, np.ones((12, 21), bool))
    with pytest.raises(ValueError):
        aligned_errors(p[:-1], p)
    with pytest.raises(ValueError):
        aligned_errors(p, p, np.zeros(12, bool))
    with pytest.raises(ValueError):
        gauge_sensitivity(p, (0.0,))
