"""Tests for `fastdo.gate.train`: masked multi-head loss, keyframe hinge, deterministic fit and class separation."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from fastdo.gate.calibrate import auroc
from fastdo.gate.model import GateConfig, InteractionGate
from fastdo.gate.train import GateTrainConfig, gate_loss, gate_scores, keyframe_hinge, train_gate
from tests.gate.fakes import fabricated_gate_batch


def test_gate_loss_masks_unlabelled_heads() -> None:
    torch.manual_seed(0)
    model = InteractionGate(GateConfig(token_dim=16))
    features, labels = fabricated_gate_batch(3, 3, 16, 0)

    loss, terms = gate_loss(model(features), labels, GateTrainConfig(), pos_weight=1.0)
    loss.backward()

    assert terms["rigid"] == 0.0 and terms["visibility"] == 0.0 and terms["keyframe"] == 0.0
    assert terms["interact"] > 0.0
    for head in (model.rigid_head, model.visibility_head, model.keyframe_head):
        grad = head.weight.grad
        assert grad is None or bool((grad == 0).all())


def test_keyframe_hinge_value() -> None:
    value = keyframe_hinge(torch.tensor([[0.0, 0.5]]), torch.tensor([[1.0, 0.0]]), 0.1)
    assert float(value) == pytest.approx(0.6, abs=1e-7)
    assert float(keyframe_hinge(torch.tensor([[0.0, 0.5]]), torch.tensor([[0.5, 0.45]]), 0.1)) == 0.0


def test_train_gate_reduces_loss_on_stand_in_data() -> None:
    torch.manual_seed(0)
    model = InteractionGate(GateConfig(token_dim=64, dropout=0.0))
    features, labels = fabricated_gate_batch(16, 16, 64, 0)

    result = train_gate(model, features, labels, GateTrainConfig(steps=3, batch_size=32, seed=0))

    assert len(result.losses) == 3
    assert result.losses[2] < result.losses[0]


def _run(seed: int) -> tuple[dict[str, torch.Tensor], tuple[float, ...], bool]:
    """A 5-step fit (default dropout 0.1) of a model built after `torch.manual_seed(0)`; also whether the global RNG
    state was left untouched by `train_gate`."""
    torch.manual_seed(0)
    model = InteractionGate(GateConfig(token_dim=16))
    features, labels = fabricated_gate_batch(12, 12, 16, 3)
    before = torch.random.get_rng_state()
    result = train_gate(model, features, labels, GateTrainConfig(steps=5, batch_size=8, seed=seed))
    return model.state_dict(), result.losses, torch.equal(torch.random.get_rng_state(), before)


def test_train_gate_is_deterministic() -> None:
    first_state, first_losses, first_rng_kept = _run(0)
    second_state, second_losses, second_rng_kept = _run(0)

    assert all(torch.equal(first_state[k], second_state[k]) for k in first_state)
    assert first_losses == second_losses
    assert first_rng_kept and second_rng_kept
    assert _run(1)[1] != first_losses


def test_train_gate_leaves_accelerator_generators_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU fit seeds only the CPU generator: `torch.manual_seed` would also (lazily) reseed every CUDA generator,
    which `fork_rng(devices=[])` does not restore."""
    torch.manual_seed(0)
    model = InteractionGate(GateConfig(token_dim=16))
    features, labels = fabricated_gate_batch(12, 12, 16, 3)
    seeded: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded.append)
    monkeypatch.setattr(torch.cuda, "manual_seed", seeded.append)

    train_gate(model, features, labels, GateTrainConfig(steps=2, batch_size=8))

    assert seeded == []


def test_trained_gate_separates_stand_in_classes() -> None:
    torch.manual_seed(0)
    model = InteractionGate(GateConfig(token_dim=64))
    features, labels = fabricated_gate_batch(64, 64, 64, 0)
    train_gate(model, features, labels, GateTrainConfig(steps=200))

    test_features, test_labels = fabricated_gate_batch(32, 32, 64, 1)

    assert auroc(gate_scores(model, test_features), test_labels.interact) >= 0.95


def test_pos_weight_balances_classes() -> None:
    torch.manual_seed(0)
    model = InteractionGate(GateConfig(token_dim=8))
    features, labels = fabricated_gate_batch(30, 10, 8, 0)
    result = train_gate(model, features, labels, GateTrainConfig(steps=1))
    assert result.pos_weight == 10 / 30
    only_positive = dataclasses.replace(labels, interact=torch.ones(40))
    with pytest.raises(ValueError):
        train_gate(model, features, only_positive, GateTrainConfig(steps=1))
