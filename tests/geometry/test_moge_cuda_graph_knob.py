"""`geometry_stage.moge_cuda_graph` from the config to the `geometry` and `intrinsics` workers.

Default False must leave the model untouched (the same object, no instance attribute) and the artifacts byte-identical;
True installs the graphed forward on the worker's model once (idempotent across calls and across two workers sharing a
model). On the CPU every forward falls back to eager, so True must also give byte-identical artifacts here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

from fastdo.bench.bench_stages import build_stage_fns, geometry_moge_cuda_graph
from fastdo.engine import workers_geometry
from fastdo.engine.workers import make_geometry_worker, make_intrinsics_worker
from fastdo.geometry.moge_cuda_graph import GraphedMoGeForward, installed_runner
from tests.engine.fakes import (
    NORMALIZED_K,
    FakeMoGe,
    random_frames,
    write_decode,
    write_geometry_scene,
)
from tests.engine.test_workers_geometry import _masked_scene

ROOT = Path(__file__).resolve().parents[2]


class ModuleMoGe(torch.nn.Module):
    """A `FakeMoGe` behind a module whose `infer` goes through `self.forward(image, num_tokens=...)` under inference
    mode, as `MoGeModel.infer` does; the forward is exact (`image * 1.0`), so the fake's tags survive it."""

    def __init__(self, fake: FakeMoGe) -> None:
        super().__init__()
        self.fake = fake
        self.one = torch.nn.Parameter(torch.ones(()))

    def forward(self, image: Tensor, num_tokens: int) -> dict[str, Tensor]:
        return {"image": image * self.one}

    @torch.inference_mode()
    def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        return self.fake.infer(self.forward(image, num_tokens=3600)["image"], **kwargs)


def _geometry(clip: Path, **kwargs: Any) -> tuple[ModuleMoGe, bytes]:
    model = ModuleMoGe(write_geometry_scene(clip))
    make_geometry_worker(device="cpu", model=model, **kwargs)("clip0", clip)
    return model, (clip / "geometry.npz").read_bytes()


def _geometry_masked(clip: Path, **kwargs: Any) -> tuple[ModuleMoGe, bytes]:
    """The production storage: the gathered `run_moge_batches` path."""
    model = ModuleMoGe(_masked_scene(clip))
    make_geometry_worker(device="cpu", model=model, pointmap_storage="masked", **kwargs)("clip0", clip)
    return model, (clip / "geometry.npz").read_bytes()


def _intrinsics(clip: Path, **kwargs: Any) -> tuple[ModuleMoGe, bytes]:
    frames = np.zeros((10, 6, 8, 3), np.uint8)
    frames[:, 0, 0, 0] = np.arange(10)
    write_decode(clip, frames)
    model = ModuleMoGe(FakeMoGe())
    make_intrinsics_worker(n_keyframes=3, device="cpu", model=model, **kwargs)("clip0", clip)
    return model, (clip / "intrinsics.npz").read_bytes()


@pytest.mark.parametrize("run", [_geometry, _geometry_masked, _intrinsics])
def test_default_false_leaves_the_model_untouched(tmp_path: Path, run: Any) -> None:
    for name, kwargs in (("default", {}), ("false", {"moge_cuda_graph": False})):
        model, _ = run(tmp_path / name, **kwargs)
        assert "forward" not in vars(model) and installed_runner(model) is None
        assert model.fake.batches                                         # MoGe did run


@pytest.mark.parametrize("run", [_geometry, _geometry_masked, _intrinsics])
def test_true_wraps_the_model_once_and_the_cpu_fallback_is_byte_identical(tmp_path: Path, run: Any,
                                                                         caplog: pytest.LogCaptureFixture) -> None:
    _, plain = run(tmp_path / "plain")
    with caplog.at_level(logging.INFO, logger=workers_geometry.__name__):
        model, graphed = run(tmp_path / "graphed", moge_cuda_graph=True)
    runner = installed_runner(model)
    assert isinstance(runner, GraphedMoGeForward)
    assert graphed == plain
    assert len(model.fake.batches) == 2                                   # 4 scene rows or 3 keyframes, batch 2
    assert "0 replay(s), 2 eager call(s)" in caplog.text                  # CPU: every forward eager, and counted


def test_true_is_idempotent_across_calls_and_workers_sharing_a_model(tmp_path: Path) -> None:
    model = ModuleMoGe(write_geometry_scene(tmp_path / "a"))
    model.fake.depth |= write_geometry_scene(tmp_path / "b", tag_offset=4).depth
    geometry = make_geometry_worker(device="cpu", model=model, moge_cuda_graph=True)
    geometry("a", tmp_path / "a")
    runner = installed_runner(model)
    assert runner is not None
    geometry("b", tmp_path / "b")
    assert installed_runner(model) is runner
    write_decode(tmp_path / "c", random_frames(4, h=6, w=8))
    make_intrinsics_worker(device="cpu", model=model, moge_cuda_graph=True)("c", tmp_path / "c")
    assert installed_runner(model) is runner


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_a_non_bool_knob_is_refused_at_construction_and_in_the_config(bad: Any) -> None:
    with pytest.raises(ValueError, match="moge_cuda_graph"):
        make_geometry_worker(device="cpu", moge_cuda_graph=bad)
    with pytest.raises(ValueError, match="moge_cuda_graph"):
        make_intrinsics_worker(device="cpu", moge_cuda_graph=bad)
    with pytest.raises(ValueError, match="geometry_stage.moge_cuda_graph"):
        geometry_moge_cuda_graph(OmegaConf.create({"geometry_stage": {"moge_cuda_graph": bad}}))


def test_the_knob_reaches_both_workers_through_the_registered_stage_builder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachability: `geometry_stage.moge_cuda_graph` must arrive at the workers `build_stage_fns` builds -- the
    geometry worker's attribute, and whether the model the intrinsics worker hands `moge_intrinsics` is graphed (spied:
    the bench builds on cuda). The production default is False."""
    cfg = OmegaConf.load(ROOT / "run" / "conf" / "config.yaml")
    assert geometry_moge_cuda_graph(cfg) is False                         # production default: off
    assert geometry_moge_cuda_graph(OmegaConf.create({})) is False        # no block: off
    graphed: list[bool] = []

    def spy(frames: Tensor, device: str = "cuda", model: Any = None, batch_size: int | None = None) -> Tensor:
        graphed.append(installed_runner(model) is not None)
        return torch.from_numpy(np.repeat(NORMALIZED_K[None], frames.shape[0], axis=0))

    monkeypatch.setattr(workers_geometry.moge_model, "moge_intrinsics", spy)
    monkeypatch.setattr(workers_geometry.moge_model, "_load_model", lambda device: ModuleMoGe(FakeMoGe()))
    write_decode(tmp_path, random_frames(4, h=6, w=8))
    for enabled in (False, True):
        OmegaConf.update(cfg, "geometry_stage.moge_cuda_graph", enabled)
        fns = build_stage_fns(cfg, ROOT, [])[0]
        assert fns["geometry"].moge_cuda_graph is enabled
        fns["intrinsics"]("clip0", tmp_path)
    assert graphed == [False, True]
