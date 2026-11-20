"""Tests for `fastdo.gate.calibrate`: the split-conformal recall threshold, its guarantee and the rank AUROC."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from fastdo.gate.calibrate import auroc, calibrate_gate, conformal_recall_threshold, operating_point
from fastdo.gate.labels import GateLabels
from tests.gate.fakes import StubGate

_POS19 = torch.arange(1, 20, dtype=torch.float64) / 10          # 0.1, 0.2, ..., 1.9


def test_conformal_threshold_order_statistic() -> None:
    tau, k = conformal_recall_threshold(_POS19, 0.1)
    assert (tau, k) == (pytest.approx(0.2), 2)
    assert calibrate_gate(_POS19, torch.ones(19), 0.1).guaranteed_recall == pytest.approx(0.9)
    alpha = 1 - 0.9
    assert alpha == 0.09999999999999998
    assert conformal_recall_threshold(_POS19, alpha) == (pytest.approx(0.2), 2)
    grid = torch.arange(41, dtype=torch.float64) / 40
    assert conformal_recall_threshold(grid, 0.05) == (pytest.approx(1 / 40), 2)
    assert calibrate_gate(grid, torch.ones(41), 0.05).guaranteed_recall == pytest.approx(1 - 2 / 42)
    assert conformal_recall_threshold(grid, 0.1) == (pytest.approx(3 / 40), 4)
    assert calibrate_gate(grid, torch.ones(41), 0.1).guaranteed_recall == pytest.approx(1 - 4 / 42)


def test_conformal_threshold_refuses_unattainable() -> None:
    with pytest.raises(ValueError, match="at least 9 calibration positives"):
        conformal_recall_threshold(torch.rand(8), 0.1)
    with pytest.raises(ValueError, match="at least 19 calibration positives"):
        conformal_recall_threshold(torch.rand(18), 0.05)
    with pytest.raises(ValueError, match="at least 9 calibration positives"):
        conformal_recall_threshold(torch.rand(8), 1 - 0.9)


def test_conformal_recall_guarantee_by_simulation() -> None:
    x = torch.rand(5000, 21, generator=torch.Generator().manual_seed(0))
    accept = torch.zeros(5000)
    ks: set[int] = set()
    for i in range(5000):
        tau, k = conformal_recall_threshold(x[i, :20], 0.1)
        ks.add(k)
        accept[i] = float(float(x[i, 20]) >= tau)
    assert ks == {2}
    assert abs(float(accept.mean()) - (1 - 2 / 21)) <= 0.0166


def test_operating_point_monotone() -> None:
    gen = torch.Generator().manual_seed(0)
    pos = 0.3 + 0.7 * torch.rand(50, generator=gen)
    neg = 0.7 * torch.rand(50, generator=gen)
    stub = StubGate(torch.cat([pos, neg]).tolist(), token_dim=16)
    features = torch.zeros(100, 8, 17)
    mask = torch.tensor([True, False, False, False]).expand(100, 4).clone()
    labels = GateLabels(interact=torch.cat([torch.ones(50), torch.zeros(50)]), rigid=torch.zeros(100),
                        visibility=torch.zeros(100), keyframe=torch.zeros(100, 8), mask=mask)

    thresholds = [operating_point(stub, features, labels, r) for r in (0.80, 0.85, 0.90, 0.95)]
    fprs = [float((neg >= t).float().mean()) for t in thresholds]

    assert all(a >= b for a, b in pairwise(thresholds))
    assert all(a <= b for a, b in pairwise(fprs))
    stub19 = StubGate(_POS19.tolist(), token_dim=16)
    labels19 = GateLabels(interact=torch.ones(19), rigid=torch.zeros(19), visibility=torch.zeros(19),
                          keyframe=torch.zeros(19, 8), mask=mask[:19])
    assert operating_point(stub19, torch.zeros(19, 8, 17), labels19, 0.9) == pytest.approx(0.2, abs=1e-6)


def test_calibrate_gate_fpr() -> None:
    no_negatives = calibrate_gate(_POS19, torch.ones(19), 0.1)
    assert no_negatives.empirical_fpr is None and (no_negatives.n_pos, no_negatives.n_neg) == (19, 0)
    assert no_negatives.threshold == pytest.approx(0.2) and no_negatives.k == 2
    assert no_negatives.empirical_recall == 18 / 19            # the positive equal to the threshold is accepted
    interact = torch.tensor([1.0] * 10 + [0.0, 0.0])
    scores = torch.tensor([0.5] * 9 + [0.7, 0.1, 0.9], dtype=torch.float64)    # n = 10 at alpha 0.1: k = 1
    at_half = calibrate_gate(scores, interact, 0.1)
    assert at_half.threshold == 0.5 and at_half.empirical_recall == 1.0 and at_half.empirical_fpr == 0.5


def test_auroc_exact() -> None:
    assert auroc(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1])) == 0.75
    assert auroc(torch.full((6,), 0.3), torch.tensor([0, 1, 0, 1, 1, 0])) == 0.5
