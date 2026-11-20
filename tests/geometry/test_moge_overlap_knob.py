"""`geometry_stage.moge_overlap` from the config to the `geometry` worker's MoGe calls.

The geometry worker runs MoGe through the REAL `MoGeModel.infer` post-processing here (`SceneMoGe`: the engine fakes'
scene depth behind a MoGe-shaped forward), on both storage paths -- dense (`run_moge`) and masked (the gathered
`run_moge_batches`). Default and False must take the serial path and True the pipelined one (spied), all three writing
the same `geometry.npz` bytes, with and without the CUDA-graphed forward (eager on the CPU). The intrinsics worker has
no such knob.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("moge")

from fastdo.bench.bench_stages import build_stage_fns, geometry_moge_overlap
from fastdo.engine.workers import make_geometry_worker, make_intrinsics_worker
from fastdo.geometry import moge_overlap
from fastdo.geometry.moge_model import fov_x_from_intrinsics
from tests.engine.fakes import SCENE_H, SCENE_K, SCENE_W, write_geometry_scene
from tests.engine.test_workers_geometry import _masked_scene
from tests.geometry.moge_fakes import SceneMoGe

ROOT = Path(__file__).resolve().parents[2]


def _spy_stage_b(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = moge_overlap.stage_b

    def spied(state: Any) -> Any:
        calls.append(1)
        return real(state)

    monkeypatch.setattr(moge_overlap, "stage_b", spied)
    return calls


def _run(clip: Path, storage: str, **kwargs: Any) -> tuple[SceneMoGe, bytes]:
    fake = _masked_scene(clip) if storage == "masked" else write_geometry_scene(clip)
    fov_x = fov_x_from_intrinsics(torch.as_tensor(SCENE_K, dtype=torch.float32), SCENE_W)
    model = SceneMoGe(fake.depth, fov_x, SCENE_H, SCENE_W)
    make_geometry_worker(device="cpu", model=model, pointmap_storage=storage, **kwargs)("clip0", clip)
    return model, (clip / "geometry.npz").read_bytes()


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("storage", ["dense", "masked"])
def test_overlap_takes_the_pipelined_path_and_writes_the_same_bytes(tmp_path: Path, storage: str, cuda_graph: bool,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_stage_b(monkeypatch)
    graph = {"moge_cuda_graph": True} if cuda_graph else {}
    _, default = _run(tmp_path / "default", storage, **graph)
    _, off = _run(tmp_path / "off", storage, moge_overlap=False, **graph)
    assert calls == []                                                  # default and False: the serial path
    model, on = _run(tmp_path / "on", storage, moge_overlap=True, **graph)
    assert len(calls) == 2                                              # 4 scene rows in batches of 2, pipelined
    assert [entry[0] for entry in model.log] == ["forward", "forward"]
    assert default == off == on


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_a_non_bool_knob_is_refused_at_construction_and_in_the_config(bad: Any) -> None:
    with pytest.raises(ValueError, match="moge_overlap"):
        make_geometry_worker(device="cpu", moge_overlap=bad)
    with pytest.raises(ValueError, match="geometry_stage.moge_overlap"):
        geometry_moge_overlap(OmegaConf.create({"geometry_stage": {"moge_overlap": bad}}))


def test_the_knob_reaches_the_geometry_worker_through_the_registered_stage_builder() -> None:
    """Reachability: `geometry_stage.moge_overlap` must arrive at the geometry worker `build_stage_fns` builds. The
    production config turns it on; the intrinsics worker takes no such argument."""
    cfg = OmegaConf.load(ROOT / "run" / "conf" / "config.yaml")
    assert geometry_moge_overlap(cfg) is True                             # production: on
    assert geometry_moge_overlap(OmegaConf.create({})) is False           # no block: off
    for enabled in (False, True):
        OmegaConf.update(cfg, "geometry_stage.moge_overlap", enabled)
        fns = build_stage_fns(cfg, ROOT, [])[0]
        assert fns["geometry"].moge_overlap is enabled
    assert "moge_overlap" not in inspect.signature(make_intrinsics_worker).parameters
