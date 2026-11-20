"""The CUDA-graph replay of the MoGe-2 forward (`fastdo.geometry.moge_cuda_graph`).

CPU tests: installation (idempotent, refuses an overridden forward, restorable, holds the model weakly), the signature
rules (what is capturable and what runs eager), and the eager fallback, which must return exactly the eager output and
count an eager call. The CUDA tests (skipped without a GPU) capture a small MoGe-shaped module through an `infer` that
mirrors `MoGeModel.infer` (inference mode + fp16 autocast) and check replay against eager bit for bit over two
signatures; the real-model check needs a GPU and the MoGe-2 weights.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.geometry import moge_cuda_graph as graph_mod
from fastdo.geometry.moge_cuda_graph import (
    GraphedMoGeForward,
    eager_moge_forward,
    graphed_moge_forward,
    installed_runner,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class TinyMoGe(torch.nn.Module):
    """MoGe-2-shaped: `forward(image, num_tokens)` sizes its token grid from `num_tokens` and the aspect ratio, resizes
    with antialiasing, builds a UV grid on the device, and returns `points` (a permuted view), `mask` and
    `metric_scale`; `infer` calls `self.forward` under inference mode and fp16 autocast, as `MoGeModel.infer` does,
    then post-processes in place. `forward_calls` counts Python executions of `forward` (a replay runs none)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, padding_mode="replicate")
        self.head = torch.nn.Conv2d(10, 4, 1)
        self.scale_head = torch.nn.Linear(8, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.forward_calls = 0

    def forward(self, image: Tensor, num_tokens: int) -> dict[str, Tensor]:
        self.forward_calls += 1
        b, _, h, w = image.shape
        aspect = w / h
        base_h, base_w = round((num_tokens / aspect) ** 0.5), round((num_tokens * aspect) ** 0.5)
        x = F.interpolate(image, (base_h * 2, base_w * 2), mode="bilinear", align_corners=False, antialias=True)
        x = self.conv(x - self.mean).relu()
        u = torch.linspace(-1.0, 1.0, base_w * 2, dtype=x.dtype, device=x.device)
        v = torch.linspace(-1.0, 1.0, base_h * 2, dtype=x.dtype, device=x.device)
        uv = torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=0)[None].expand(b, -1, -1, -1)
        out = self.head(torch.cat([x, uv], dim=1))
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=False)
        return {"points": out[:, :3].permute(0, 2, 3, 1), "mask": out[:, 3].sigmoid(),
                "metric_scale": self.scale_head(x.mean(dim=(2, 3))).squeeze(1).exp()}

    @torch.inference_mode()
    def infer(self, image: Tensor, num_tokens: int = 60, use_fp16: bool = True) -> dict[str, Tensor]:
        with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=use_fp16 and image.is_cuda):
            output = self.forward(image, num_tokens=num_tokens)
        points = output["points"].float()
        points[..., 2] += 1.0                           # in place, as MoGe's shift
        return {"points": points * output["metric_scale"].float()[:, None, None, None],
                "mask": output["mask"].float() > 0.5}


def _image(b: int = 2, h: int = 24, w: int = 32, device: str = "cpu", seed: int = 0) -> Tensor:
    """`(B,3,H,W)` float in [0,1] with the permuted strides `moge_model._model_input` produces."""
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, 256, (b, h, w, 3), generator=g, dtype=torch.uint8)
    return frames.to(device).permute(0, 3, 1, 2).float() / 255.0


def _assert_same(a: dict[str, Tensor], b: dict[str, Tensor]) -> None:
    assert a.keys() == b.keys()
    for key, value in a.items():
        assert value.stride() == b[key].stride(), key
        assert torch.equal(value, b[key]), key


# --- installation -------------------------------------------------------------------------------------


def test_install_is_idempotent_restorable_and_reuses_the_runner() -> None:
    model = TinyMoGe()
    assert installed_runner(model) is None and "forward" not in vars(model)
    runner = graphed_moge_forward(model)
    assert isinstance(runner, GraphedMoGeForward)
    assert vars(model)["forward"] is runner and installed_runner(model) is runner
    assert graphed_moge_forward(model) is runner                          # idempotent
    eager_moge_forward(model)
    assert "forward" not in vars(model) and installed_runner(model) is None
    eager_moge_forward(model)                                             # nothing installed: a no-op
    assert graphed_moge_forward(model) is runner                          # the same runner (and graphs) again


def test_install_refuses_a_foreign_override_and_a_non_module() -> None:
    model = TinyMoGe()
    model.forward = lambda image, num_tokens: {}                          # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="already overridden"):
        graphed_moge_forward(model)
    eager_moge_forward(model)                                             # not ours: left in place
    assert "forward" in vars(model)
    with pytest.raises(TypeError, match="torch.nn.Module"):
        graphed_moge_forward(object())                                    # type: ignore[arg-type]


def test_the_runner_holds_the_model_weakly() -> None:
    model = TinyMoGe()
    runner = graphed_moge_forward(model)
    del model
    gc.collect()
    with pytest.raises(RuntimeError, match="no longer exists"):
        runner.eager(_image(), num_tokens=60)


# --- the eager fallback ---------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["inference", "no_grad", "grad"])
def test_cpu_calls_return_exactly_the_eager_output_and_count_an_eager_call(mode: str) -> None:
    model = TinyMoGe()
    image = _image()
    context = {"inference": torch.inference_mode, "no_grad": torch.no_grad, "grad": torch.enable_grad}[mode]
    with context():
        expected = model(image, num_tokens=60)
        runner = graphed_moge_forward(model)
        got = model(image, num_tokens=60)                                 # nn.Module.__call__ -> the runner
        got_infer = model.infer(image)
    eager_moge_forward(model)
    _assert_same(got, expected)
    _assert_same(got_infer, TinyMoGe.infer(model, image))
    assert runner.take_stats() == {"captures": 0, "capture_seconds": 0.0, "replays": 0, "eager_calls": 2}
    assert runner.take_stats()["eager_calls"] == 0                        # take_stats resets


@pytest.fixture
def any_tensor_is_capturable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lets `signature` see a CPU image as capturable, so its other rules can be checked without a GPU."""
    monkeypatch.setattr(graph_mod, "_sig", lambda x: ("image",) if isinstance(x, Tensor) else None)


@pytest.mark.usefixtures("any_tensor_is_capturable")
def test_the_signature_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TinyMoGe().eval()
    runner = GraphedMoGeForward(model)
    image = _image()
    with torch.inference_mode():
        key = runner.signature(image, (), {"num_tokens": 3600})
        assert key is not None and key[1] == 3600
        assert runner.signature(image, (3600,), {}) == key                # positional num_tokens: the same key
        assert runner.signature(image, (), {"num_tokens": 1200}) != key
        with monkeypatch.context() as patch:
            patch.setattr(torch, "is_autocast_enabled", lambda device_type=None: not key[2][0])
            assert runner.signature(image, (), {"num_tokens": 3600}) != key       # autocast state is in the key
        uncapturable: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((), {"num_tokens": torch.tensor(3600)}),                     # data-dependent token grid
            ((), {"num_tokens": True}),
            ((), {"num_tokens": 3600.0}),
            ((), {"num_tokens": 3600, "extra": 1}),
            ((3600,), {"num_tokens": 3600}),
            ((3600, 1), {}),
            ((), {}),
        ]
        for args, kwargs in uncapturable:
            assert runner.signature(image, args, kwargs) is None, (args, kwargs)
        model.train()
        assert runner.signature(image, (), {"num_tokens": 3600}) is None  # training mode: never graphed
        model.eval()
    with torch.no_grad():
        assert runner.signature(image, (), {"num_tokens": 3600}) not in (None, key)  # inference mode is in the key
    with torch.enable_grad():
        assert runner.signature(image, (), {"num_tokens": 3600}) is None  # grad mode: never graphed
    assert runner.signature("not a tensor", (), {"num_tokens": 3600}) is None


@pytest.mark.usefixtures("any_tensor_is_capturable")
def test_backend_flags_are_in_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TinyMoGe().eval()                                              # the runner holds it weakly
    runner = GraphedMoGeForward(model)
    with torch.inference_mode():
        key = runner.signature(_image(), (), {"num_tokens": 60})
        monkeypatch.setattr(torch.backends.cudnn, "benchmark", not torch.backends.cudnn.benchmark)
        assert runner.signature(_image(), (), {"num_tokens": 60}) != key


# --- CUDA: replay equals eager ---------------------------------------------------------------------------


@cuda
def test_replay_equals_eager_bit_for_bit_over_two_signatures() -> None:
    torch.manual_seed(0)
    model = TinyMoGe().cuda().eval()
    images = [_image(b, device="cuda", seed=s) for s in range(3) for b in (2, 1)]    # batch 2 and a tail batch
    expected = [TinyMoGe.infer(model, image) for image in images]
    runner = graphed_moge_forward(model)
    calls_before = model.forward_calls
    for _ in range(2):
        got = [model.infer(image) for image in images]
        for g, e in zip(got, expected, strict=True):
            _assert_same(g, e)
    stats = runner.take_stats()
    assert stats["captures"] == 2 and stats["eager_calls"] == 0
    assert stats["replays"] == 2 * len(images)
    assert model.forward_calls - calls_before == 2 * (graph_mod.WARMUP_CALLS + 1)    # replays run no Python
    eager_moge_forward(model)


@cuda
def test_replay_outputs_are_clones_and_moved_weights_are_recaptured() -> None:
    torch.manual_seed(0)
    model = TinyMoGe().cuda().eval()
    image = _image(device="cuda")
    expected = TinyMoGe.infer(model, image)
    runner = graphed_moge_forward(model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        first = model(image, num_tokens=60)
        first["points"].fill_(123.0)                                      # the caller owns what it gets
        second = model(image, num_tokens=60)
    assert not torch.equal(first["points"], second["points"])
    _assert_same(model.infer(image), expected)
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.clone()                                       # the same values at new addresses
    _assert_same(model.infer(image), expected)
    stats = runner.take_stats()
    assert stats["captures"] == 2 and stats["eager_calls"] == 0          # the second capture: after the move
    eager_moge_forward(model)


@cuda
def test_a_forward_that_draws_random_numbers_stays_eager_and_consumes_no_extra_randomness() -> None:
    class Noisy(TinyMoGe):
        def forward(self, image: Tensor, num_tokens: int) -> dict[str, Tensor]:
            out = super().forward(image, num_tokens)
            return out | {"mask": out["mask"] + torch.rand_like(out["mask"])}

    model = Noisy().cuda().eval()
    image = _image(device="cuda")
    torch.cuda.manual_seed(7)
    expected = TinyMoGe.infer(model, image)
    state = torch.cuda.get_rng_state()
    runner = graphed_moge_forward(model)
    torch.cuda.manual_seed(7)
    got = model.infer(image)
    assert torch.equal(torch.cuda.get_rng_state(), state)                # the warm-up's draws were rolled back
    _assert_same(got, expected)
    got_again = model.infer(image)
    assert got_again["points"].shape == expected["points"].shape
    stats = runner.take_stats()
    assert stats["captures"] == 1 and stats["replays"] == 0 and stats["eager_calls"] == 2
    eager_moge_forward(model)
