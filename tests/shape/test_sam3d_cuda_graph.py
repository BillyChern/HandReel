"""The opt-in CUDA-graph replay of SAM 3D's sparse-structure flow backbone.

CPU tests: the `sam3d_cuda_graph_ss` flag from `ShapeConfig` to the worker's `reconstruct_geometry`, the graph path
entered only when the flag is set, and the runner's eager fallback for calls it cannot capture. The CUDA test (skipped
without a GPU) captures a small stand-in backbone and checks replay against the eager forward bit for bit; the real
SAM 3D bit-identity check needs a GPU and the real weights.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.shape import ShapeConfig
from fastdo.shape import sam3d_cuda_graph as graph_mod
from fastdo.shape import sam3d_worker as worker_mod
from fastdo.shape.backend import non_default_shape_options
from fastdo.shape.sam3d_backend import Sam3dBackend
from fastdo.shape.sam3d_worker import (
    reconstruct_geometry,
    request_params_digest,
    resolve_shape_options,
    serve,
)
from tests.shape.fake_sam3d import FakeSam3dPipeline


def _params(**overrides: Any) -> dict[str, Any]:
    return {"seed": 42, "sparse_structure_steps": 25, "structured_latent_steps": 25, "solver": "euler",
            "mesh_postprocess": True} | overrides


# --- the flag ------------------------------------------------------------------------------------------


def test_the_flag_is_off_by_default_and_a_default_request_is_unchanged_on_the_wire() -> None:
    cfg = ShapeConfig(backend="sam3d")
    assert cfg.sam3d_cuda_graph_ss is False
    assert "sam3d_cuda_graph_ss" not in Sam3dBackend(cfg).sampling_params
    assert non_default_shape_options(cfg) == {}
    assert resolve_shape_options(_params()).cuda_graph_ss is False


def test_the_flag_reaches_the_request_and_is_recorded_when_set() -> None:
    cfg = ShapeConfig(backend="sam3d", sam3d_cuda_graph_ss=True)
    params = Sam3dBackend(cfg).sampling_params
    assert params["sam3d_cuda_graph_ss"] is True
    assert resolve_shape_options(params).cuda_graph_ss is True
    assert non_default_shape_options(cfg) == {"sam3d_cuda_graph_ss": True}


def test_the_flag_does_not_move_the_params_digest() -> None:
    """Output-preserving: the arm digest names the mesh parameters, and the graph path is not one."""
    assert request_params_digest(_params(sam3d_cuda_graph_ss=True)) == request_params_digest(_params())


@pytest.mark.parametrize("value", [1, "true", None])
def test_a_non_bool_flag_is_refused_with_the_same_message_at_both_boundaries(value: Any) -> None:
    with pytest.raises(ValueError) as config_error:
        ShapeConfig(backend="sam3d", sam3d_cuda_graph_ss=value)
    with pytest.raises(ValueError) as worker_error:
        resolve_shape_options(_params(sam3d_cuda_graph_ss=value))
    assert str(worker_error.value) == str(config_error.value)
    assert "sam3d_cuda_graph_ss" in str(worker_error.value)


def test_the_flag_is_refused_off_sam3d() -> None:
    with pytest.raises(ValueError, match="sam3d_cuda_graph_ss"):
        ShapeConfig(backend="trellis", sam3d_cuda_graph_ss=True)
    ShapeConfig(backend="trellis", sam3d_cuda_graph_ss=False)


# --- the graph path is entered only when enabled -------------------------------------------------------


class _RecordingGraph:
    def __init__(self) -> None:
        self.entered: list[Any] = []
        self.active = False

    @contextmanager
    def __call__(self, ss_generator: Any) -> Any:
        self.entered.append(ss_generator)
        self.active = True
        try:
            yield SimpleNamespace(take_stats=lambda: {"captures": 1, "capture_seconds": 0.5, "replays": 38,
                                                      "eager_calls": 0})
        finally:
            self.active = False


def _reconstruct(pipe: FakeSam3dPipeline, **overrides: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    image = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:15, 8:22] = True
    params: dict[str, Any] = {"seed": 0, "sparse_structure_steps": 25, "structured_latent_steps": 25,
                              "solver": "euler", "mesh_postprocess": False} | overrides
    return reconstruct_geometry(pipe, image, mask, **params)


@pytest.mark.parametrize("enabled", [False, True])
def test_the_graph_wraps_exactly_the_sparse_structure_flow_when_enabled(monkeypatch: pytest.MonkeyPatch,
                                                                         enabled: bool) -> None:
    recorder = _RecordingGraph()
    monkeypatch.setattr(worker_mod.sam3d_cuda_graph, "graphed_ss_backbone", recorder)
    pipe = FakeSam3dPipeline()
    seen: dict[str, bool] = {}
    for name in ("sample_sparse_structure", "sample_slat", "decode_slat"):
        original = getattr(pipe, name)

        def spy(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            seen[_name] = recorder.active
            return _original(*args, **kwargs)

        setattr(pipe, name, spy)
    graph_stats: dict[str, Any] = {}
    _reconstruct(pipe, cuda_graph_ss=enabled, graph_stats=graph_stats)
    assert recorder.entered == ([pipe.models["ss_generator"]] if enabled else [])
    assert seen == {"sample_sparse_structure": enabled, "sample_slat": False, "decode_slat": False}
    assert graph_stats == ({"captures": 1, "capture_seconds": 0.5, "replays": 38, "eager_calls": 0} if enabled
                           else {})


@pytest.mark.parametrize("enabled", [False, True])
def test_the_worker_replies_with_graph_counters_only_when_asked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                               enabled: bool) -> None:
    calls: list[dict[str, Any]] = []

    def fake_reconstruct(pipeline: Any, image: Any, mask: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs["graph_stats"] is not None:
            kwargs["graph_stats"].update(captures=0, replays=38)
        return np.zeros((3, 3)), np.zeros((1, 3), np.int64), {"sparse_structure_flow": 1.0}

    monkeypatch.setattr(worker_mod, "reconstruct_geometry", fake_reconstruct)
    request_npz = tmp_path / "in.npz"
    np.savez(request_npz, image=np.zeros((4, 4, 3), np.uint8), mask=np.ones((4, 4), bool))
    params = _params(sam3d_cuda_graph_ss=True) if enabled else _params()
    line = json.dumps({"cmd": "reconstruct", "id": 1, "input": str(request_npz),
                       "output": str(tmp_path / "out.npz"), "params": params})
    replies = io.StringIO()
    assert serve(object(), io.StringIO(line + "\n"), replies) == 0
    reply = json.loads(replies.getvalue())
    assert reply["ok"] and calls[0]["cuda_graph_ss"] is enabled
    assert reply.get("cuda_graph_ss") == ({"captures": 0, "replays": 38} if enabled else None)


# --- the runner ----------------------------------------------------------------------------------------


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.calls = 0

    def forward(self, latents: dict[str, torch.Tensor], t: torch.Tensor, *cond: torch.Tensor,
                **kwargs: Any) -> dict[str, torch.Tensor]:
        self.calls += 1
        scale = 0.0 if kwargs.get("cfg", False) else 1.0
        return {name: self.linear(x) + t[:, None, None] + scale * cond[0].mean() for name, x in latents.items()}


def _generator(backbone: torch.nn.Module) -> Any:
    return SimpleNamespace(reverse_fn=SimpleNamespace(backbone=backbone))


def test_uncapturable_calls_run_eager_and_the_forward_is_restored() -> None:
    backbone = _Backbone()
    latents, t, cond = {"shape": torch.randn(1, 4, 8)}, torch.tensor([3.0]), torch.randn(1, 5, 8)
    with torch.no_grad():
        expected = backbone(latents, t, cond, d=torch.tensor([0.0]))
        with graph_mod.graphed_ss_backbone(_generator(backbone)) as runner:
            assert "forward" in vars(backbone)
            out = backbone(latents, t, cond, d=torch.tensor([0.0]))    # CPU tensors: never captured
            stats = runner.take_stats()
    assert "forward" not in vars(backbone)
    assert stats == {"captures": 0, "capture_seconds": 0.0, "replays": 0, "eager_calls": 1}
    assert torch.equal(out["shape"], expected["shape"])


def test_nesting_is_refused() -> None:
    generator = _generator(_Backbone())
    with graph_mod.graphed_ss_backbone(generator), pytest.raises(RuntimeError, match="already overridden"), \
            graph_mod.graphed_ss_backbone(generator):
        pass
    assert "forward" not in vars(generator.reverse_fn.backbone)


@pytest.fixture
def fake_timestep_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A stand-in for SAM 3D's `timestep_embedder` whose `timestep_embedding` is the original's expression."""
    module = ModuleType(graph_mod.TIMESTEP_EMBEDDER_MODULE)

    class TimestepEmbedder:
        @staticmethod
        def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
            half = dim // 2
            freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
                              / half).to(device=t.device)
            args = t[:, None].float() * freqs[None]
            return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    module.TimestepEmbedder = TimestepEmbedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, graph_mod.TIMESTEP_EMBEDDER_MODULE, module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_replay_equals_the_eager_forward_bit_for_bit(fake_timestep_module: ModuleType) -> None:
    embedder = fake_timestep_module.TimestepEmbedder

    class Timed(_Backbone):
        def forward(self, latents: dict[str, torch.Tensor], t: torch.Tensor, *cond: torch.Tensor,
                    **kwargs: Any) -> dict[str, torch.Tensor]:
            emb = embedder.timestep_embedding(t, 8)[:, None, :]
            return {name: value + emb for name, value in super().forward(latents, t, *cond, **kwargs).items()}

    original = embedder.__dict__["timestep_embedding"]
    backbone = Timed().cuda()
    cond = torch.randn(1, 5, 8, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for step in range(4):
            latents = {"shape": torch.randn(1, 4, 8, device="cuda")}
            t = torch.tensor([100.0 * step], device="cuda")
            for cfg in ({}, {"cfg": True}):
                expected = backbone(latents, t, cond, d=torch.zeros(1, device="cuda"), **cfg)
                with graph_mod.graphed_ss_backbone(_generator(backbone)) as runner:
                    out = backbone(latents, t, cond, d=torch.zeros(1, device="cuda"), **cfg)
                    stats = runner.take_stats()
                assert torch.equal(out["shape"], expected["shape"])
                assert stats["replays"] == 1 and stats["eager_calls"] == 0
                assert stats["captures"] == (1 if step == 0 else 0)
    assert embedder.__dict__["timestep_embedding"] is original                          # restored on exit
