"""Gate threshold calibration: a split-conformal recall bound on calibration positives, and the rank AUROC.

For `n` calibration positives and `k = floor(alpha * (n + 1))`, the threshold is the k-th smallest positive score and a
clip is accepted iff `score >= threshold`. A new positive exchangeable with the calibration positives is then accepted
with probability at least `1 - k / (n + 1) >= 1 - alpha`. `k == 0` has no guarantee and is refused: there is no
accept-all fallback (as for the qc stage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor

from fastdo.gate.labels import GateLabels
from fastdo.gate.model import InteractionGate
from fastdo.gate.train import gate_scores

__all__ = ["GateCalibration", "auroc", "calibrate_gate", "conformal_recall_threshold", "operating_point"]

_EPS = 1e-9      # absorbs float round-off: (1 - 0.9) * 20 = 1.9999999999999996, 1 / (1 - 0.9) = 10.000000000000002


@dataclass(frozen=True)
class GateCalibration:
    """The calibrated threshold, its conformal rank `k` and guarantee, and the calibration set's empirical rates."""

    threshold: float
    alpha: float
    k: int
    n_pos: int
    n_neg: int
    guaranteed_recall: float
    empirical_recall: float
    empirical_fpr: float | None


def conformal_recall_threshold(pos_scores: Tensor, alpha: float) -> tuple[float, int]:
    """`(threshold, k)`: the k-th smallest of the `n` positive scores, `k = floor(alpha * (n + 1) + 1e-9)`.

    Raises:
        ValueError: If `alpha` is outside (0, 1), a score is non-finite, or `k == 0` (too few positives).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    scores = pos_scores.detach().reshape(-1).double()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("calibration scores must be finite")
    n = int(scores.numel())
    k = min(math.floor(alpha * (n + 1) + _EPS), n)
    if k == 0:
        need = math.ceil(1 / alpha - _EPS) - 1
        raise ValueError(f"conformal recall threshold needs at least {need} calibration positives at alpha {alpha:g}, "
                         f"got {n}")
    return float(torch.sort(scores).values[k - 1]), k


def calibrate_gate(scores: Tensor, interact: Tensor, alpha: float) -> GateCalibration:
    """The conformal threshold on the positives (`interact > 0.5`) and the empirical recall and FPR at it
    (`empirical_fpr` is None without negatives).

    Raises:
        ValueError: If the shapes differ, or as `conformal_recall_threshold`.
    """
    flat = scores.detach().reshape(-1).double()
    positive = interact.detach().reshape(-1) > 0.5
    if flat.shape != positive.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and interact {tuple(interact.shape)} must match")
    pos, neg = flat[positive], flat[~positive]
    threshold, k = conformal_recall_threshold(pos, alpha)
    n_pos, n_neg = int(pos.numel()), int(neg.numel())
    return GateCalibration(threshold=threshold, alpha=float(alpha), k=k, n_pos=n_pos, n_neg=n_neg,
                           guaranteed_recall=1.0 - k / (n_pos + 1),
                           empirical_recall=float((pos >= threshold).double().mean()),
                           empirical_fpr=float((neg >= threshold).double().mean()) if n_neg else None)


def operating_point(model: InteractionGate, features: Tensor, labels: GateLabels, target_recall: float) -> float:
    """The conformal threshold on the interact-labelled positives at `alpha = round(1 - target_recall, 12)`.

    Raises:
        ValueError: As `conformal_recall_threshold`.
    """
    labelled = labels.mask[:, 0].bool()
    scores = gate_scores(model, features)[labelled]
    positives = scores[labels.interact[labelled] > 0.5]
    return conformal_recall_threshold(positives, round(1.0 - target_recall, 12))[0]


def auroc(scores: Tensor, labels: Tensor) -> float:
    """The rank (Mann-Whitney) AUROC of `scores` for `labels > 0.5`, ties counted as 0.5.

    Raises:
        ValueError: If the shapes differ or either class is empty.
    """
    flat = scores.detach().reshape(-1).double().numpy()
    positive = labels.detach().reshape(-1).numpy() > 0.5
    if flat.shape != positive.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and labels {tuple(labels.shape)} must match")
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"AUROC needs both classes, got {n_pos} positives and {n_neg} negatives")
    ranks = np.asarray(rankdata(flat), dtype=np.float64)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
