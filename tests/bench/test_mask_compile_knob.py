"""`bench.mask_compile` from the config to the loader call of the `mask` workers the engine builds.

`none` (the production default) must reach the loader as exactly the call it always made -- `(device, model_id)`, no
`mask_compile` keyword -- so its masks stay byte-identical; a vos mode must reach it as `mask_compile=<mode>`, both
through `build_stage_fns` and through the batched runner's slot pool (`build_batched_stages`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastdo.bench.bench_batched import build_batched_stages
from fastdo.bench.bench_stages import build_stage_fns, mask_compile
from fastdo.engine.workers_video import make_mask_worker, write_mask_prompt
from tests.engine.fakes import random_frames, write_decode
from tests.shape.test_masks_sam2 import _FakeSam2Predictor

_REPO = Path(__file__).resolve().parents[2]


def _cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[f"paths.root={_REPO}", *overrides])


def _spy_loader(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def load(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _FakeSam2Predictor({}, {})

    monkeypatch.setattr("fastdo.shape.masks_sam2.load_sam2_predictor", load)
    return calls


def _clip(root: Path) -> Path:
    pytest.importorskip("PIL")                                  # SAM 2 is staged through JPEG frames
    write_decode(root, random_frames(3, h=8, w=8))
    prompt = np.zeros((8, 8), bool)
    prompt[2:6, 2:6] = True
    write_mask_prompt(root, 0, prompt)
    return root


def test_the_production_default_is_none_and_a_missing_key_means_none() -> None:
    # vos_nocg was once the default and was reverted (not reproducible across processes).
    assert mask_compile(OmegaConf.load(_REPO / "run/conf/config.yaml")) == "none"
    assert mask_compile(OmegaConf.create({})) == "none"
    assert mask_compile(OmegaConf.create({"bench": {}})) == "none"
    assert mask_compile(OmegaConf.create({"bench": {"mask_compile": "vos_nocg"}})) == "vos_nocg"


@pytest.mark.parametrize("bad", ["vos", "max-autotune", "", None, True])
def test_an_unknown_value_is_refused_by_the_reader_the_worker_and_the_stage_builder(bad: Any) -> None:
    with pytest.raises(ValueError, match="mask_compile"):
        mask_compile(OmegaConf.create({"bench": {"mask_compile": bad}}))
    with pytest.raises(ValueError, match="mask_compile"):
        make_mask_worker(device="cpu", mask_compile=bad)                 # at construction, before any load
    cfg = _cfg()
    OmegaConf.update(cfg, "bench.mask_compile", bad)
    with pytest.raises(ValueError, match="mask_compile"):
        build_stage_fns(cfg, _REPO, [])


def test_the_worker_keeps_the_old_loader_call_for_none_and_passes_vos_modes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_loader(monkeypatch)
    clip = _clip(tmp_path)
    for mode in ("none", "vos_nocg", "vos_cg_clone"):
        make_mask_worker(variant="tiny", device="cpu", mask_compile=mode)("c0", clip)
    make_mask_worker(variant="tiny", device="cpu")("c0", clip)            # the default is none
    old = {"device": "cpu", "model_id": "facebook/sam2.1-hiera-tiny"}
    assert calls == [old, {**old, "mask_compile": "vos_nocg"}, {**old, "mask_compile": "vos_cg_clone"}, old]


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")   # the bench builds on cuda; CPU here
@pytest.mark.parametrize("mode", ["none", "vos_nocg", "vos_cg_clone"])
def test_the_knob_reaches_the_loader_through_both_registered_builders(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    calls = _spy_loader(monkeypatch)
    clip = _clip(tmp_path / "c0")
    cfg = _cfg(f"bench.mask_compile={mode}")
    fns, labels = build_stage_fns(cfg, _REPO, [])
    fns["mask"]("c0", clip)                                               # the bench builds on cuda: spied loader
    batched = build_batched_stages(cfg, fns, tmp_path / "jobs", tmp_path / ".gpu.lock")
    batched["mask"].pool._make_worker()("c0", clip)
    want = {"device": "cuda", "model_id": "facebook/sam2.1-hiera-large"}
    want = want if mode == "none" else {**want, "mask_compile": mode}
    assert calls == [want, want]
    assert (labels["mask"] == "computed") is (mode == "none")
    assert mode == "none" or mode in labels["mask"]
