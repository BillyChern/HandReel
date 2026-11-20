"""Gate fit: masked multi-head loss and a seeded AdamW loop on CPU.

Each source labels different heads (`GateLabels.mask`): masked BCE-with-logits on `interact_logit` (with `pos_weight`)
and on `rigid_logit`, masked MSE on `visibility`, and a clip-internal ranking hinge on `keyframe_scores`. A head without
a labelled sample in the batch contributes an exact 0.0 and no gradient.

Determinism: minibatches come from `torch.Generator().manual_seed(cfg.seed)`, and the loop runs under
`torch.random.fork_rng` with the CPU default generator (and, for CUDA features, that device's generator) seeded with
`cfg.seed` inside, so dropout is seeded and the caller's generators are untouched. `torch.manual_seed` is not used: it
also reseeds every CUDA generator (lazily, before CUDA init), which `fork_rng(devices=[])` does not restore. Weight init
is the caller's (`InteractionGate(cfg)` draws from the global RNG): given the same initial state_dict, features, labels
and config, the trained state_dict is bit-identical on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.gate.labels import GateLabels
from fastdo.gate.model import GateOutput, InteractionGate

__all__ = ["GateTrainConfig", "GateTrainResult", "gate_loss", "gate_scores", "keyframe_hinge", "train_gate"]


@dataclass(frozen=True)
class GateTrainConfig:
    """Optimizer, sampling and per-head loss weights of `train_gate`."""

    steps: int = 400
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42
    w_interact: float = 1.0
    w_rigid: float = 0.5
    w_visibility: float = 0.5
    w_keyframe: float = 0.5
    keyframe_margin: float = 0.1


@dataclass(frozen=True)
class GateTrainResult:
    """`losses[i]`: the minibatch loss before update i (train mode); `final_loss`: the loss on every sample after the
    last update (eval mode); `pos_weight = n_neg / n_pos` over the interact-labelled samples."""

    losses: tuple[float, ...]
    final_loss: float
    pos_weight: float


def keyframe_hinge(scores: Tensor, targets: Tensor, margin: float) -> Tensor:
    """Mean over clip-internal pairs with `targets_i > targets_j + margin` of `relu(margin - (scores_i - scores_j))`;
    zero (no gradient) when there is no pair.

    Raises:
        ValueError: If `scores` and `targets` are not the same `(B, F)` shape.
    """
    if scores.ndim != 2 or scores.shape != targets.shape:
        raise ValueError(f"scores and targets must both be (B, F), got {tuple(scores.shape)} and {tuple(targets.shape)}")
    pairs = (targets[:, :, None] - targets[:, None, :]) > margin
    if not bool(pairs.any()):
        return scores.new_zeros(())
    return torch.relu(margin - (scores[:, :, None] - scores[:, None, :]))[pairs].mean()


def gate_loss(out: GateOutput, labels: GateLabels, cfg: GateTrainConfig,
              pos_weight: float) -> tuple[Tensor, dict[str, float]]:
    """The weighted masked loss and its per-head terms (module docstring)."""
    mask = labels.mask.bool()
    zero = out.interact_logit.new_zeros(())
    dtype = out.interact_logit.dtype
    terms: dict[str, Tensor] = dict.fromkeys(("interact", "rigid", "visibility", "keyframe"), zero)
    if bool(mask[:, 0].any()):
        sel = mask[:, 0]
        terms["interact"] = F.binary_cross_entropy_with_logits(out.interact_logit[sel], labels.interact[sel].to(dtype),
                                                               pos_weight=torch.tensor(pos_weight, dtype=dtype))
    if bool(mask[:, 1].any()):
        sel = mask[:, 1]
        terms["rigid"] = F.binary_cross_entropy_with_logits(out.rigid_logit[sel], labels.rigid[sel].to(dtype))
    if bool(mask[:, 2].any()):
        sel = mask[:, 2]
        terms["visibility"] = F.mse_loss(out.visibility[sel], labels.visibility[sel].to(dtype))
    if bool(mask[:, 3].any()):
        sel = mask[:, 3]
        terms["keyframe"] = keyframe_hinge(out.keyframe_scores[sel], labels.keyframe[sel].to(dtype),
                                           cfg.keyframe_margin)
    total = (cfg.w_interact * terms["interact"] + cfg.w_rigid * terms["rigid"]
             + cfg.w_visibility * terms["visibility"] + cfg.w_keyframe * terms["keyframe"])
    return total, {name: float(term.detach()) for name, term in terms.items()}


def _take(labels: GateLabels, idx: Tensor) -> GateLabels:
    return GateLabels(interact=labels.interact[idx], rigid=labels.rigid[idx], visibility=labels.visibility[idx],
                      keyframe=labels.keyframe[idx], mask=labels.mask[idx])


def train_gate(model: InteractionGate, features: Tensor, labels: GateLabels, cfg: GateTrainConfig) -> GateTrainResult:
    """Fits `model` in place with AdamW on seeded minibatches (module docstring); the model's train/eval mode is
    restored afterwards.

    Raises:
        ValueError: If the labels do not match the features, `steps < 0` or `batch_size < 1`, or the interact-labelled
            samples lack a positive or a negative.
    """
    n = int(features.shape[0])
    if any(int(t.shape[0]) != n for t in (labels.interact, labels.rigid, labels.visibility, labels.keyframe,
                                          labels.mask)):
        raise ValueError(f"labels must have one row per feature row ({n})")
    if cfg.steps < 0 or cfg.batch_size < 1:
        raise ValueError(f"steps must be >= 0 and batch_size >= 1, got {cfg.steps} and {cfg.batch_size}")
    labelled = labels.mask[:, 0].bool()
    n_pos = int((labels.interact[labelled] > 0.5).sum())
    n_neg = int(labelled.sum()) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"train_gate needs interact-labelled positives and negatives, got {n_pos} and {n_neg}")
    pos_weight = n_neg / n_pos
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    batch = min(cfg.batch_size, n)
    losses: list[float] = []
    was_training = model.training
    with torch.random.fork_rng(devices=[features.device] if features.is_cuda else []):
        torch.default_generator.manual_seed(cfg.seed)
        if features.is_cuda:
            with torch.cuda.device(features.device):
                torch.cuda.manual_seed(cfg.seed)
        model.train()
        for _ in range(cfg.steps):
            idx = torch.randperm(n, generator=generator)[:batch]
            loss, _ = gate_loss(model(features[idx]), _take(labels, idx), cfg, pos_weight)
            losses.append(float(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            final, _ = gate_loss(model(features), labels, cfg, pos_weight)
    model.train(was_training)
    return GateTrainResult(losses=tuple(losses), final_loss=float(final), pos_weight=pos_weight)


def gate_scores(model: InteractionGate, features: Tensor) -> Tensor:
    """`p_interact` `(B,)` in eval mode under `torch.no_grad`; the model's mode is restored afterwards."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out: GateOutput = model(features)
            return out.p_interact.detach().clone()
    finally:
        model.train(was_training)
