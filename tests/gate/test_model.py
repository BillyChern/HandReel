"""Tests for `fastdo.gate.model`: the interaction gate's shapes, frame-order invariance and LSE pooling."""

from __future__ import annotations

import statistics
import time

import pytest
import torch

from fastdo.gate.model import GateConfig, InteractionGate, lse_pool


def test_interaction_gate_forward_shapes() -> None:
    torch.manual_seed(0)
    model = InteractionGate(GateConfig(token_dim=384)).eval()

    out = model(torch.randn(4, 8, 385))

    for head in (out.p_interact, out.p_rigid, out.visibility):
        assert head.shape == (4,) and bool(((head > 0) & (head < 1)).all())
    assert out.keyframe_scores.shape == (4, 8)
    assert torch.allclose(torch.sigmoid(out.interact_logit), out.p_interact, atol=1e-6)
    assert torch.allclose(torch.sigmoid(out.rigid_logit), out.p_rigid, atol=1e-6)


def test_gate_parameter_count_matches_the_design() -> None:
    model = InteractionGate(GateConfig(token_dim=2048))
    n = sum(p.numel() for p in model.parameters())
    assert 2.7e5 < n < 2.9e5


def test_gate_is_frame_permutation_invariant() -> None:
    torch.manual_seed(1)
    model = InteractionGate(GateConfig(token_dim=32)).eval()
    x = torch.randn(3, 8, 33)
    perm = [7, 0, 6, 1, 5, 2, 4, 3]

    a, b = model(x), model(x[:, perm])

    for name in ("p_interact", "p_rigid", "visibility", "interact_logit", "rigid_logit"):
        assert torch.allclose(getattr(a, name), getattr(b, name), atol=1e-6), name
    assert torch.allclose(b.keyframe_scores, a.keyframe_scores[:, perm], atol=1e-6)


def test_lse_pool_matches_formula() -> None:
    z = torch.tensor([[0.0, 0, 0, 0, 0, 0, 0, 10.0]])

    value = float(lse_pool(z, 5.0))

    assert value == pytest.approx(9.584111691664033, abs=1e-5)
    assert 1.25 <= value <= 10.0
    assert torch.allclose(lse_pool(torch.full((2, 8), 0.7), 5.0), torch.full((2,), 0.7), atol=1e-6)


def test_gate_rejects_wrong_feature_dims() -> None:
    model = InteractionGate(GateConfig(token_dim=384))
    with pytest.raises(ValueError, match="385"):
        model(torch.randn(4, 8, 384))
    with pytest.raises(ValueError, match="8"):
        model(torch.randn(4, 7, 385))


def test_eval_forward_is_deterministic() -> None:
    model = InteractionGate(GateConfig(token_dim=64)).eval()
    x = torch.randn(5, 8, 65)
    a, b = model(x), model(x)
    assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


@pytest.mark.timing
def test_gate_forward_is_cheap_on_cpu() -> None:
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = InteractionGate(GateConfig(token_dim=2048)).eval()
        x = torch.randn(1, 8, 2049)
        seconds: list[float] = []
        with torch.no_grad():
            model(x)
            for _ in range(50):
                start = time.perf_counter()
                model(x)
                seconds.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(threads)
    assert statistics.median(seconds) < 0.005
