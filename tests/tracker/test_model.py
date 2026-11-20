"""Tracker CPU tests M1-M11: config checks, output shapes, ranges, initialization, determinism, budget, host reads."""

from __future__ import annotations

import dataclasses
import inspect
import logging

import pytest
import torch
from torch import Tensor, nn

from fastdo.tracker.config import TrackerConfig
from fastdo.tracker.features import frame_feature_dim
from fastdo.tracker.heads import TrackerOutput
from fastdo.tracker.model import AmortizedTracker
from fastdo.tracker.pack import TrackerInputs
from tests.tracker.tracker_fixtures import (
    SMALL_CONFIG,
    random_tracker_inputs,
    randomize_parameters,
    tracker_loss,
)

F32, F64 = torch.float32, torch.float64


def _m2_inputs(dtype: torch.dtype = F64) -> TrackerInputs:
    return random_tracker_inputs(2, (20, 13), 64, 256, 64, seed=0, dtype=dtype)


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _saturated() -> tuple[AmortizedTracker, TrackerInputs, TrackerOutput]:
    torch.manual_seed(0)
    model = AmortizedTracker()
    randomize_parameters(model, std=1.0, seed=0)
    base = _m2_inputs()
    inputs = dataclasses.replace(base, frame=base.frame * 1e3, track=base.track * 1e3, point=base.point * 1e3,
                                 cluster=base.cluster * 1e3)
    with torch.no_grad():
        return model, inputs, model(inputs)


def test_tracker_config_validation() -> None:
    cfg = TrackerConfig()
    assert cfg.feature_dim == 110 and cfg.damping_init == 1e-3 and cfg.weight_init == 0.9
    with pytest.raises(ValueError):
        TrackerConfig(n_params=8)
    assert TrackerConfig(n_params=8, feature_dim=114).feature_dim == 114
    bad = [{"n_heads": 5}, {"weight_init": 1e-3}, {"weight_floor": 0.0}, {"damping_init": 1e3},
           {"damping_log10_range": (2.0, -6.0)}, {"damping_log10_range": (-3.0, -3.0)}, {"precision_floor": 0.0},
           {"n_params": 6}, {"n_params": 6, "feature_dim": frame_feature_dim(6)}, {"max_tracks": 0},
           {"n_layers": 0}, {"weight_init": 1.0}]
    for kwargs in bad:
        with pytest.raises(ValueError):
            TrackerConfig(**kwargs)    # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.width = 1                  # type: ignore[misc]


def test_tracker_output_shapes() -> None:
    torch.manual_seed(0)
    model = AmortizedTracker()
    inputs = _m2_inputs(F32)
    with torch.no_grad():
        z = model.backbone(inputs)
        out = model(inputs)
        out64 = model(_m2_inputs(F64))                    # float64 inputs are cast to the parameter dtype
    assert z.shape == (2, 20, 384) and z.dtype == F32
    shapes = {"point_weights": (2, 20, 256), "track_weights": (2, 20, 64), "damping": (2,), "xi_hat": (2, 20, 6),
              "sigma_hat": (2,), "precision_diag": (2, 20, 6), "dual_g": (2, 20, 64), "dual_f": (2, 20, 256)}
    assert tuple(shapes) == TrackerOutput._fields
    for name, shape in shapes.items():
        value: Tensor = getattr(out, name)
        assert value.shape == shape and value.dtype == F32, name
        assert bool(torch.isfinite(value).all()), name
        assert torch.equal(getattr(out64, name), value), name


def test_tracker_weights_in_unit_interval_and_masked() -> None:
    model, inputs, out = _saturated()
    floor, fm = model.cfg.weight_floor, inputs.frame_mask
    point_valid = inputs.point_mask & fm[..., None]
    track_valid = inputs.track_mask & (inputs.track[..., 5] > 0) & fm[..., None]
    for weights, valid in ((out.point_weights, point_valid), (out.track_weights, track_valid)):
        w = weights[valid]
        assert bool((w >= floor).all()) and bool((w <= 1.0).all())
        assert float(w.min()) < 0.01 and float(w.max()) > 0.99
        assert bool((weights[~valid] == 0.0).all())
    assert bool((~inputs.track_mask | (inputs.track[..., 5] == 0) | ~fm[..., None])[~track_valid].all())


def test_tracker_precision_strictly_positive() -> None:
    model, inputs, out = _saturated()
    p, fm = out.precision_diag, inputs.frame_mask
    assert model.cfg.precision_floor > 0
    assert bool(torch.isfinite(p).all()) and bool((p >= model.cfg.precision_floor).all())
    assert bool((p[~fm] == 1.0).all()) and bool((~fm).any())
    assert float(p[fm].max()) > 1.0 and float(p[fm].min()) < 1e-3


def test_tracker_initialization_contract() -> None:
    torch.manual_seed(0)
    model = AmortizedTracker()
    inputs = _m2_inputs()
    with torch.no_grad():
        out = model(inputs)
    fm = inputs.frame_mask
    torch.testing.assert_close(out.damping, torch.full((2,), 1e-3), rtol=1e-5, atol=0.0)
    point_valid = inputs.point_mask & fm[..., None]
    track_valid = inputs.track_mask & (inputs.track[..., 5] > 0) & fm[..., None]
    for weights, valid in ((out.point_weights, point_valid), (out.track_weights, track_valid)):
        assert float((weights[valid] - 0.9).abs().max()) <= 1e-6
    assert float((out.precision_diag[fm] - 0.6931482).abs().max()) <= 1e-6
    for name in ("xi_hat", "sigma_hat", "dual_f", "dual_g"):
        value = getattr(out, name)
        assert torch.equal(value, torch.zeros_like(value)), name
    _, _, sat = _saturated()
    assert bool((sat.damping >= 1e-6).all()) and bool((sat.damping <= 1e2).all())


def test_tracker_deterministic_with_fixed_seed() -> None:
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        model = AmortizedTracker()
        with torch.no_grad():
            runs.append((model.state_dict(), model(_m2_inputs())))
    (sd0, out0), (sd1, out1) = runs
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    for name in TrackerOutput._fields:
        assert torch.equal(getattr(out0, name), getattr(out1, name)), name
    randomize_parameters(model, std=0.05, seed=0)
    with torch.no_grad():
        a, b = model(_m2_inputs()), model(_m2_inputs())
    for name in TrackerOutput._fields:
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_tracker_parameter_count_within_budget(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="fastdo.tracker.model"):
        model = AmortizedTracker()
    total = _count(model)
    assert total == 11_622_338 and 5e6 <= total <= 1.5e7
    bb, hd = model.backbone, model.heads
    assert _count(bb.frame_enc) == 18_048
    assert _count(bb.track_pool) == 300_678 and _count(bb.cluster_pool) == 300_294
    assert _count(bb.ln_in) == 768 and _count(bb.ln_out) == 768
    assert len(bb.blocks) == 6 and all(_count(b) == 1_774_662 for b in bb.blocks)
    assert sum(_count(b) for b in bb.blocks) == 10_647_972 and _count(bb) == 11_268_528
    assert (_count(hd.track_head), _count(hd.point_head), _count(hd.cluster_head)) == (66_817, 67_074, 66_689)
    assert _count(hd.frame_head) == 153_230 and _count(hd) == 353_810
    assert _count(AmortizedTracker(SMALL_CONFIG)) == 76_098
    assert "11622338" in caplog.text


def test_tracker_rejects_bad_inputs() -> None:
    model = AmortizedTracker()
    base = random_tracker_inputs(1, (4,), 8, 16, 64, seed=0)
    bad = [random_tracker_inputs(1, (4,), 129, 16, 64, seed=0),
           dataclasses.replace(base, frame=base.frame[..., :109]),
           dataclasses.replace(base, track_mask=base.track_mask.float()),
           dataclasses.replace(base, point_mask=base.point_mask[..., :-1]),
           random_tracker_inputs(1, (4,), 8, 16, 63, seed=0),
           dataclasses.replace(base, point=base.point[:, :3], point_mask=base.point_mask[:, :3])]
    for inputs in bad:
        with pytest.raises(ValueError):
            model(inputs)


def test_tracker_forward_reads_nothing_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = AmortizedTracker()
    randomize_parameters(model, std=0.05, seed=0)
    inputs = _m2_inputs()
    with torch.no_grad():
        ref = model(inputs)

    def host_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("host read")

    with monkeypatch.context() as m, torch.no_grad():
        for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__float__", "__int__"):
            m.setattr(torch.Tensor, name, host_read)
        out = model(inputs)
    for name in TrackerOutput._fields:
        assert torch.equal(getattr(out, name), getattr(ref, name)), name


def test_tracker_gradients_finite_with_empty_sets() -> None:
    model = AmortizedTracker(SMALL_CONFIG).double()
    randomize_parameters(model, std=0.05, seed=0)
    base = random_tracker_inputs(2, (8, 1), 16, 32, 64, seed=3)      # clip 1 has one valid frame
    track_mask, point_mask = base.track_mask.clone(), base.point_mask.clone()
    track_mask[0, 3] = False
    point_mask[0, 5] = False
    inputs = dataclasses.replace(base, track_mask=track_mask, point_mask=point_mask)
    out = model(inputs)
    assert torch.equal(out.track_weights[0, 3], torch.zeros(16, dtype=F64))
    assert torch.equal(out.point_weights[0, 5], torch.zeros(32, dtype=F64))
    tracker_loss(out).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
    bb, hd = model.backbone, model.heads
    groups: dict[str, nn.Module] = {"frame_enc": bb.frame_enc, "track_pool": bb.track_pool,
                                    "cluster_pool": bb.cluster_pool, "frame_head": hd.frame_head,
                                    "track_head": hd.track_head, "point_head": hd.point_head,
                                    "cluster_head": hd.cluster_head}
    groups.update({f"block{i}": block for i, block in enumerate(bb.blocks)})
    for name, module in groups.items():
        assert any(bool((p.grad != 0).any()) for p in module.parameters() if p.grad is not None), name


def test_model_sees_no_scale_inputs() -> None:
    assert list(inspect.signature(AmortizedTracker.forward).parameters) == ["self", "inputs"]
    assert [f.name for f in dataclasses.fields(TrackerInputs)] == [
        "frame", "frame_mask", "track", "track_mask", "point", "point_mask", "cluster", "cluster_mask"]
    names = [f.name for f in dataclasses.fields(TrackerConfig)]
    assert "K" not in names
    for word in ("focal", "image", "diameter", "scale", "depth"):
        assert not [n for n in names if word in n.lower()], word
