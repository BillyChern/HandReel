"""Tests for `fastdo.geometry.moge_model.run_moge` against an injected fake model (no GPU/weights)."""

from __future__ import annotations

import builtins
import math
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.geometry import moge_model
from fastdo.geometry.moge_model import fov_x_from_intrinsics, run_moge, run_moge_batches
from fastdo.solver import project


class _FakeMoGeModel:
    """Duck-typed `moge.model.v2.MoGeModel`: `.infer(image (B,3,H,W), **kw) -> {"points": ...}`.

    Returns the input image itself (permuted back to (B,H,W,3)) times a fixed constant, so the test
    can check the output is exactly what the "model" produced -- i.e. `run_moge` applies no scaling
    of its own.
    """

    def __init__(self, factor: float = 7.0) -> None:
        self.factor = factor
        self.calls: list[Tensor] = []
        self.kwargs: list[dict[str, Any]] = []

    def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(image)
        self.kwargs.append(kwargs)
        return {"points": image.permute(0, 2, 3, 1) * self.factor}


def test_run_moge_shape_and_scale_invariance() -> None:
    n, h, w = 5, 8, 8
    frames = (torch.arange(n * h * w * 3) % 256).to(torch.uint8).reshape(n, h, w, 3)
    model = _FakeMoGeModel(factor=7.0)

    out = run_moge(frames, device="cpu", model=model)

    assert out.shape == (n, h, w, 3)
    assert out.dtype == torch.float32
    assert sum(int(c.shape[0]) for c in model.calls) == n          # chunked, every frame covered once
    assert all(c.shape[1:] == (3, h, w) for c in model.calls)
    expected = (frames.to(torch.float32) / 255.0) * 7.0              # the fake's own output, unscaled
    assert torch.allclose(out, expected, atol=1e-5)
    assert all("fov_x" not in kw for kw in model.kwargs)             # no fov_x unless asked for


def test_run_moge_marks_masked_pixels_nan_and_forwards_fov() -> None:
    class _MaskingModel(_FakeMoGeModel):
        def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Any]:
            out = super().infer(image, **kwargs)
            out["points"][:, 0, 0] = torch.inf                       # MoGe's own apply_mask convention
            return out

    K = torch.tensor([[5.0, 0.0, 2.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]])
    for intrinsics in (None, K):
        model = _MaskingModel()
        out = run_moge(torch.full((2, 4, 4, 3), 128, dtype=torch.uint8), device="cpu", model=model,
                       intrinsics=intrinsics)
        assert torch.isnan(out[:, 0, 0]).all()
        assert torch.isfinite(out[:, 1:, 1:]).all()
    assert all(kw.get("fov_x") == pytest.approx(fov_x_from_intrinsics(K, 4)) for kw in model.kwargs)


class _CentredProjectionModel:
    """MoGe-2's `force_projection` output: depth back-projected with focal `f` about the image centre."""

    def __init__(self, depth: Tensor, f: float) -> None:
        self.depth, self.f = depth, f

    def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Any]:
        b, _, h, w = image.shape
        vv, uu = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                                indexing="ij")
        z = self.depth[:b]
        points = torch.stack([(uu - (w - 1) / 2) * z / self.f, (vv - (h - 1) / 2) * z / self.f, z], dim=-1)
        return {"points": points, "depth": z}


def test_run_moge_pointmap_projects_onto_its_pixels_with_an_off_centre_principal_point() -> None:
    n, h, w = 2, 24, 32
    K = torch.tensor([[40.0, 0.0, 12.3], [0.0, 41.0, 14.6], [0.0, 0.0, 1.0]])   # ~4 px off centre
    depth = 0.5 + torch.rand(n, h, w, generator=torch.Generator().manual_seed(0))
    model = _CentredProjectionModel(depth, f=40.0)

    raw = run_moge(torch.zeros(n, h, w, 3, dtype=torch.uint8), device="cpu", model=model)
    out = run_moge(torch.zeros(n, h, w, 3, dtype=torch.uint8), device="cpu", model=model, intrinsics=K)

    vv, uu = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    pixels = torch.stack([uu, vv], dim=-1).expand(n, h, w, 2)
    assert torch.allclose(project(out, K), pixels, atol=1e-3)            # every point lands on its own pixel
    assert torch.equal(out[..., 2], depth)                              # MoGe's depth is kept as is
    assert (project(raw, K) - pixels).abs().amax() > 3.0                 # the centred map does not


def test_fov_x_from_intrinsics() -> None:
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    assert fov_x_from_intrinsics(K, width=640) == pytest.approx(math.degrees(2 * math.atan(320 / 500)))


def test_run_moge_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="N,H,W,3"):
        run_moge(torch.zeros(3, 8, 8), device="cpu", model=_FakeMoGeModel())
    with pytest.raises(ValueError, match="3,3"):
        run_moge(torch.zeros(1, 8, 8, 3), device="cpu", model=_FakeMoGeModel(), intrinsics=torch.eye(4))


def test_run_moge_missing_package_raises_actionable_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("moge"):
            raise ImportError("simulated missing 'moge' package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(RuntimeError, match="frontend"):
        run_moge(torch.zeros(1, 4, 4, 3, dtype=torch.uint8), device="cpu")


def test_model_id_is_the_open_moge2_vitl_checkpoint() -> None:
    assert moge_model.MODEL_ID == "Ruicheng/moge-2-vitl"


def test_moge_intrinsics_passthrough() -> None:
    """D3 amendment: `moge_intrinsics` returns MoGe-2's normalized `infer(...)["intrinsics"]` unchanged, in
    `_BATCH_SIZE` chunks. N = 5 covers two full batches and a remainder at the current `_BATCH_SIZE` of 2 (R2)."""
    from fastdo.geometry.moge_model import _BATCH_SIZE, moge_intrinsics

    fixed = torch.tensor([[0.79, 0.0, 0.5], [0.0, 1.40, 0.5], [0.0, 0.0, 1.0]])

    class _IntrinsicsModel:
        def __init__(self) -> None:
            self.batches: list[tuple[int, ...]] = []

        def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Any]:
            self.batches.append(tuple(image.shape))
            return {"intrinsics": fixed.expand(image.shape[0], 3, 3).clone()}

    n = 5
    model = _IntrinsicsModel()
    out = moge_intrinsics(torch.zeros(n, 6, 8, 3, dtype=torch.uint8), device="cpu", model=model)

    assert out.shape == (n, 3, 3) and out.dtype == torch.float32
    assert torch.equal(out, fixed.expand(n, 3, 3))
    assert [b[0] for b in model.batches] == [_BATCH_SIZE] * (n // _BATCH_SIZE) + [n % _BATCH_SIZE] * (n % _BATCH_SIZE > 0)
    assert len(model.batches) >= 3 and all(b[1:] == (3, 6, 8) for b in model.batches)


def _old_postprocess(points: np.ndarray, K: Tensor | None) -> np.ndarray:
    """The earlier whole-array numpy path: mask non-finite pixels, then back-project MoGe's depth through K."""
    out = points.astype(np.float32).copy()
    out[~np.isfinite(out).all(axis=-1)] = np.nan
    return out if K is None else moge_model._points_through_intrinsics(out[..., 2], K)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"))])
def test_device_post_processing_is_bit_identical_to_the_numpy_path(device: str) -> None:
    """Masking and back-projection moved into the batch loop must not move a single bit."""
    n, h, w = 5, 18, 26
    g = torch.Generator().manual_seed(3)
    points = torch.randn(n, h, w, 3, generator=g) * 2.0
    points[..., 2] = 0.3 + torch.rand(n, h, w, generator=g) * 3.0
    points[0, 1, 2] = torch.inf                      # MoGe's apply_mask: the whole pixel
    points[1, 4, 5, 0] = torch.inf                   # a non-finite x with a finite depth: still invalid
    points[3, 7, 7, 1] = torch.nan

    class _Replay:
        def __init__(self) -> None:
            self.at = 0

        def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Any]:
            b = image.shape[0]
            self.at += b
            return {"points": points[self.at - b:self.at].to(image.device)}

    K = torch.tensor([[31.7, 0.0, 12.9], [0.0, 30.2, 8.4], [0.0, 0.0, 1.0]])
    frames = torch.zeros(n, h, w, 3, dtype=torch.uint8)
    for intr in (None, K):
        got = run_moge(frames, device=device, model=_Replay(), intrinsics=intr).numpy()
        want = _old_postprocess(points.numpy(), intr)
        assert np.array_equal(got, want, equal_nan=True)
        assert np.isnan(got[0, 1, 2]).all() and np.isnan(got[1, 4, 5]).all() and np.isnan(got[3, 7, 7]).all()


def test_run_moge_batches_hands_the_sink_run_moges_blocks() -> None:
    """The sink sees, per `_BATCH_SIZE` block, exactly the rows `run_moge` returns (NaN pixels included)."""
    n, h, w = 5, 24, 32
    K = torch.tensor([[40.0, 0.0, 12.3], [0.0, 41.0, 14.6], [0.0, 0.0, 1.0]])
    depth = 0.5 + torch.rand(n, h, w, generator=torch.Generator().manual_seed(0))
    depth[:, 3, 4] = torch.inf
    model = _CentredProjectionModel(depth, f=40.0)
    frames = torch.zeros(n, h, w, 3, dtype=torch.uint8)
    for intrinsics in (None, K):
        expected = run_moge(frames, device="cpu", model=model, intrinsics=intrinsics)
        got = torch.full_like(expected, 7.0)
        starts: list[int] = []

        def sink(start: int, points: Tensor, got: Tensor = got, starts: list[int] = starts) -> None:
            starts.append(start)
            got[start:start + points.shape[0]] = points

        run_moge_batches(frames, sink, device="cpu", model=model, intrinsics=intrinsics)
        assert starts == list(range(0, n, moge_model._BATCH_SIZE))
        assert got.numpy().tobytes() == expected.numpy().tobytes() and torch.isnan(got[:, 3, 4]).all()


# --- the MoGe batch size as a parameter (geometry_stage.moge_batch_size) ---------------------------------------


class _TaggedModel:
    """Records each forward's frame tags (pixel (0,0) channel 0); returns points and intrinsics that depend on the tag
    alone, so the outputs are batch-independent and any difference is the batching, not the fake."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Any]:
        tags = (image[:, 0, 0, 0] * 255.0).round().long()
        self.batches.append(tags.tolist())
        b, _, h, w = image.shape
        points = torch.ones(b, h, w, 3) * (1.0 + tags.float())[:, None, None, None]
        k = torch.eye(3).expand(b, 3, 3) * (1.0 + tags.float())[:, None, None]
        return {"points": points, "intrinsics": k.clone()}


def _tagged_frames(n: int) -> Tensor:
    frames = torch.zeros(n, 4, 6, 3, dtype=torch.uint8)
    frames[:, 0, 0, 0] = torch.arange(n, dtype=torch.uint8)
    return frames


def _sink_run(frames: Tensor, model: Any, **kwargs: Any) -> list[int]:
    starts: list[int] = []
    run_moge_batches(frames, lambda start, points: starts.append(start), device="cpu", model=model, **kwargs)
    return starts


def test_batch_size_default_is_todays_batch_composition_and_8_differs() -> None:
    """The default (no argument) and an explicit 2 make exactly today's forwards on all three entry points; 8 makes
    fewer, larger ones over the same rows, with identical outputs from a batch-independent model."""
    from fastdo.geometry.moge_model import moge_intrinsics

    n = 11
    frames = _tagged_frames(n)
    today = [list(range(s, min(s + 2, n))) for s in range(0, n, 2)]            # [[0,1],[2,3],...,[10]]
    eight = [list(range(8)), list(range(8, 11))]
    K = torch.tensor([[5.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]])
    outputs: dict[Any, tuple[Tensor, Tensor]] = {}
    for batch_size, want in ((None, today), (2, today), (8, eight)):
        kwargs = {} if batch_size is None else {"batch_size": batch_size}
        points_model, k_model, sink_model = _TaggedModel(), _TaggedModel(), _TaggedModel()
        points = run_moge(frames, device="cpu", model=points_model, intrinsics=K, **kwargs)
        intr = moge_intrinsics(frames, device="cpu", model=k_model, **kwargs)
        starts = _sink_run(frames, sink_model, **kwargs)
        assert points_model.batches == k_model.batches == sink_model.batches == want, batch_size
        assert starts == [b[0] for b in want]
        outputs[batch_size] = (points, intr)
    for batch_size in (2, 8):
        assert outputs[batch_size][0].numpy().tobytes() == outputs[None][0].numpy().tobytes()
        assert outputs[batch_size][1].numpy().tobytes() == outputs[None][1].numpy().tobytes()


def test_batch_size_default_follows_the_module_constant_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """`None` reads `_BATCH_SIZE` when called, not when defined: a `--batch` style override keeps working."""
    monkeypatch.setattr(moge_model, "_BATCH_SIZE", 4)
    model = _TaggedModel()
    run_moge(_tagged_frames(6), device="cpu", model=model)
    assert model.batches == [[0, 1, 2, 3], [4, 5]]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "8"])
def test_batch_size_is_validated(bad: Any) -> None:
    from fastdo.geometry.moge_model import moge_intrinsics

    frames = _tagged_frames(3)
    for call in (lambda: run_moge(frames, device="cpu", model=_TaggedModel(), batch_size=bad),
                 lambda: moge_intrinsics(frames, device="cpu", model=_TaggedModel(), batch_size=bad),
                 lambda: _sink_run(frames, _TaggedModel(), batch_size=bad)):
        with pytest.raises(ValueError, match="batch_size"):
            call()
