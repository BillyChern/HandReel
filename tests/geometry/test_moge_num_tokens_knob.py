"""`geometry_stage.moge_num_tokens` from the config to BOTH MoGe-2 workers and both geometry paths.

The geometry worker runs MoGe through the REAL `MoGeModel.infer` post-processing (`SceneMoGe`, as in the overlap test),
on both storage paths (dense `run_moge`, masked gathered `run_moge_batches`) and both MoGe paths (serial,
overlapped): the default and an explicit None must write the same `geometry.npz` bytes with every forward at infer's
own 3600 tokens, and a value must reach every forward. The intrinsics worker hands its value to `infer` (FakeMoGe
records the kwargs) and never runs MoGe on a clip with `camera.json`. `build_stage_fns` must deliver the config value
to both workers; production is null.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

pytest.importorskip("moge")

from fastdo.bench.bench_stages import build_stage_fns, geometry_moge_num_tokens
from fastdo.engine import workers_geometry
from fastdo.engine.workers import make_geometry_worker, make_intrinsics_worker
from fastdo.geometry.intrinsics import camera_matrix, read_intrinsics
from fastdo.geometry.moge_model import fov_x_from_intrinsics
from tests.engine.fakes import (
    NORMALIZED_K,
    SCENE_H,
    SCENE_K,
    SCENE_W,
    FakeMoGe,
    write_decode,
    write_geometry_scene,
)
from tests.engine.test_workers_geometry import _masked_scene
from tests.geometry.moge_fakes import SceneMoGe

ROOT = Path(__file__).resolve().parents[2]


def _run(clip: Path, storage: str, **kwargs: Any) -> tuple[SceneMoGe, bytes]:
    fake = _masked_scene(clip) if storage == "masked" else write_geometry_scene(clip)
    fov_x = fov_x_from_intrinsics(torch.as_tensor(SCENE_K, dtype=torch.float32), SCENE_W)
    model = SceneMoGe(fake.depth, fov_x, SCENE_H, SCENE_W)
    make_geometry_worker(device="cpu", model=model, pointmap_storage=storage, **kwargs)("clip0", clip)
    return model, (clip / "geometry.npz").read_bytes()


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("storage", ["dense", "masked"])
def test_geometry_worker_default_is_untouched_and_a_value_reaches_every_forward(tmp_path: Path, storage: str,
                                                                                overlap: bool) -> None:
    default_model, default = _run(tmp_path / "default", storage, moge_overlap=overlap)
    none_model, none = _run(tmp_path / "none", storage, moge_overlap=overlap, moge_num_tokens=None)
    assert default == none
    assert default_model.tokens == none_model.tokens == [3600, 3600]          # 4 scene rows in batches of 2
    model, _ = _run(tmp_path / "t2500", storage, moge_overlap=overlap, moge_num_tokens=2500)
    assert model.tokens == [2500, 2500]


def test_intrinsics_worker_passes_the_value_and_skips_camera_json(tmp_path: Path) -> None:
    frames = np.zeros((10, 6, 8, 3), np.uint8)
    frames[:, 0, 0, 0] = np.arange(10)
    got: dict[Any, tuple[list[dict[str, Any]], bytes]] = {}
    for tokens in (None, 2500):
        clip = tmp_path / f"t{tokens}"
        write_decode(clip, frames)
        model = FakeMoGe()
        make_intrinsics_worker(n_keyframes=3, device="cpu", model=model, moge_num_tokens=tokens)("clip0", clip)
        got[tokens] = (model.kwargs, (clip / "intrinsics.npz").read_bytes())
    assert got[None][0] == [{}, {}]                                            # today's call, keyword absent
    assert got[2500][0] == [{"num_tokens": 2500}] * 2
    assert got[None][1] == got[2500][1]                                        # the fake's K ignores the tokens
    clip = tmp_path / "calibrated"
    write_decode(clip, frames)
    (clip / "camera.json").write_text(json.dumps({"fx": 600.0, "fy": 600.0, "cx": 4.0, "cy": 3.0}))
    model = FakeMoGe()
    make_intrinsics_worker(device="cpu", model=model, moge_num_tokens=2500)("clip0", clip)
    assert model.kwargs == []                                                  # calibrated K: MoGe never runs
    K, source = read_intrinsics(clip)
    assert source == "given" and np.array_equal(K, camera_matrix(600.0, 600.0, 4.0, 3.0))


def test_the_knobs_reach_their_own_workers_through_the_registered_stage_builder(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachability (split knobs): `moge_num_tokens` reaches the geometry worker ONLY, and
    `intrinsics_moge_num_tokens` is what the intrinsics worker hands `moge_intrinsics` (spied: the bench builds on
    cuda). On internet video K comes from MoGe, so the intrinsics tokens are their own knob."""
    cfg = OmegaConf.load(ROOT / "run" / "conf" / "config.yaml")
    assert "moge_num_tokens" in cfg.geometry_stage                            # the knob is registered ...
    assert geometry_moge_num_tokens(cfg) is None                                # ... and production is null
    assert geometry_moge_num_tokens(OmegaConf.create({})) is None               # no block: MoGe's default
    seen: list[Any] = []

    def spy(frames: Tensor, device: str = "cuda", model: Any = None, batch_size: int | None = None,
            num_tokens: int | None = None) -> Tensor:
        seen.append(num_tokens)
        return torch.from_numpy(np.repeat(NORMALIZED_K[None], frames.shape[0], axis=0))

    monkeypatch.setattr(workers_geometry.moge_model, "moge_intrinsics", spy)
    monkeypatch.setattr(workers_geometry.moge_model, "_load_model", lambda device: FakeMoGe())
    write_decode(tmp_path, np.zeros((4, 6, 8, 3), np.uint8))
    for tokens in (None, 2500, 1800):
        OmegaConf.update(cfg, "geometry_stage.moge_num_tokens", tokens)
        fns = build_stage_fns(cfg, ROOT, [])[0]
        assert fns["geometry"].moge_num_tokens == tokens
        fns["intrinsics"]("clip0", tmp_path)
    assert seen == [None, None, None]                                          # geometry's knob never reaches K
    OmegaConf.update(cfg, "geometry_stage.moge_num_tokens", None)
    for tokens in (2500, None):
        OmegaConf.update(cfg, "geometry_stage.intrinsics_moge_num_tokens", tokens)
        fns = build_stage_fns(cfg, ROOT, [])[0]
        assert fns["geometry"].moge_num_tokens is None
        fns["intrinsics"]("clip0", tmp_path)
    assert seen == [None, None, None, 2500, None]
    assert "moge_num_tokens" in inspect.signature(make_intrinsics_worker).parameters


@pytest.mark.parametrize("bad", [1199, 3601, 0, 2500.0, True, "2500"])
def test_a_bad_value_is_refused_at_construction_and_in_the_config(bad: Any) -> None:
    with pytest.raises(ValueError, match="moge_num_tokens"):
        make_geometry_worker(device="cpu", moge_num_tokens=bad)
    with pytest.raises(ValueError, match="moge_num_tokens"):
        make_intrinsics_worker(device="cpu", moge_num_tokens=bad)
    with pytest.raises(ValueError, match="geometry_stage.moge_num_tokens"):
        geometry_moge_num_tokens(OmegaConf.create({"geometry_stage": {"moge_num_tokens": bad}}))
