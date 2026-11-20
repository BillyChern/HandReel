"""`solve_stage.fixed_scale` / `SolverConfig.fix_scale`: default off byte for byte, the held scale, the mutation
check, and the config block."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fastdo.engine.depth_anchor_gate import gated_solver_config
from fastdo.engine.instance_scale import write_instance_scale
from fastdo.engine.worker_support import load_npz
from fastdo.engine.workers_solve import (
    FIXED_SCALE_KEY,
    DepthAnchorGateConfig,
    FixedScaleConfig,
    RegistrationConfig,
    SolveStageConfig,
    make_solve_worker,
    prepare_solve,
)
from fastdo.solver import lm
from fastdo.solver.api import SolverConfig, _levels
from fastdo.solver.lm import LMConfig
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import SMALL_CLIP, SMALL_SOLVE, write_synthetic_solve_inputs

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
FAST = dataclasses.replace(SMALL_SOLVE, solver={"iters_coarse": 3, "iters_mid": 2, "iters_fine": 3})
FIXED = dataclasses.replace(FAST, fixed_scale=FixedScaleConfig())
GATE = DepthAnchorGateConfig(threshold=0.25, lam=500.0, band=0.2)


@pytest.fixture(scope="module")
def small_clip() -> SyntheticClip:
    return make_synthetic_clip(SMALL_CLIP, seed=0)


def _npz_identical(a: Path, b: Path, skip: tuple[str, ...] = ()) -> bool:
    x, y = load_npz(a), load_npz(b)
    return x.keys() == y.keys() and all(x[k].dtype == y[k].dtype and x[k].tobytes() == y[k].tobytes()
                                        for k in x if k not in skip)


def _meta(d: Path) -> dict:
    meta = json.loads((d / "solve.json").read_text())
    meta["solver"].pop("seconds")
    return meta


def test_config_defaults_off_and_validates() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    assert SolveStageConfig.from_mapping(root.solve_stage).fixed_scale is None
    assert SolveStageConfig().fixed_scale is None and SolverConfig().fix_scale is False
    assert SolveStageConfig().solver_config() == SolverConfig(device="cuda", seed=42)
    root.solve_stage.fixed_scale = {"file": "instance_scale.json"}
    cfg = SolveStageConfig.from_mapping(root.solve_stage)
    assert cfg.fixed_scale == FixedScaleConfig() and cfg.solver_config().fix_scale is True
    assert dataclasses.replace(cfg.solver_config(), fix_scale=False) == dataclasses.replace(cfg, fixed_scale=None
                                                                                            ).solver_config()
    for bad, match in (({"file": "../x.json"}, "plain file name"), ({"file": ""}, "plain file name"), ({}, "missing"),
                       ({"file": "a.json", "k": 3}, "unknown"), ([1], "null or")):
        with pytest.raises(ValueError, match=match):
            SolveStageConfig.from_mapping({"fixed_scale": bad})
    with pytest.raises(ValueError, match="estimate_scale"):
        SolveStageConfig(fixed_scale=FixedScaleConfig(), registration=RegistrationConfig(estimate_scale=True))
    with pytest.raises(ValueError, match="fix_scale=true is refused"):             # the top-level refusal is untouched
        SolveStageConfig(fix_scale=True)
    with pytest.raises(ValueError, match="top-level"):
        SolveStageConfig(solver={"fix_scale": True})


def test_solver_plumbs_fix_scale_to_every_level_and_through_the_gate() -> None:
    off, on = _levels(SolverConfig()), _levels(SolverConfig(fix_scale=True))
    # default off: the LMConfigs are the ones built before fix_scale existed (fix_scale False is LMConfig's own default)
    assert [lmc for _, lmc in off] == [LMConfig(max_iter=10, min_median_depth=0.05),
                                       LMConfig(max_iter=5, lam0=3e-3, min_median_depth=0.05),
                                       LMConfig(max_iter=20, lam0=1e-3, min_median_depth=0.05)]
    assert all(lmc.fix_scale for _, lmc in on)
    assert [dataclasses.replace(lmc, fix_scale=False) for _, lmc in on] == [lmc for _, lmc in off]
    assert gated_solver_config(SolverConfig(fix_scale=True), GATE).fix_scale is True


def test_default_off_ignores_a_stray_file_and_writes_todays_solve(tmp_path: Path, small_clip: SyntheticClip) -> None:
    a, b = tmp_path / "a" / "c", tmp_path / "b" / "c"
    write_synthetic_solve_inputs(a, small_clip)
    shutil.copytree(a, b)
    write_instance_scale(b, {"scale": 9.0})                     # present, but the block is off: never read
    assert FIXED_SCALE_KEY not in prepare_solve("c", a, FAST).context
    make_solve_worker(FAST)("c", a)
    make_solve_worker(FAST)("c", b)
    assert _npz_identical(a / "solve.npz", b / "solve.npz", skip=("seconds",))
    assert _meta(a) == _meta(b) and FIXED_SCALE_KEY not in _meta(a)


def test_fixed_scale_is_registered_at_and_held_through_the_solve(tmp_path: Path, small_clip: SyntheticClip) -> None:
    prod, fixed = tmp_path / "p" / "c", tmp_path / "m" / "c"
    write_synthetic_solve_inputs(prod, small_clip)
    shutil.copytree(prod, fixed)
    make_solve_worker(FAST)("c", prod)
    s_prod = float(load_npz(prod / "solve.npz")["scale"])
    target = 1.3 * s_prod
    record = {"scale": target, "instance": "I", "k": 3, "members": ["c", "d", "e"], "fused_size_m": 0.1}
    write_instance_scale(fixed, record)
    make_solve_worker(FIXED)("c", fixed)
    s_fixed = float(load_npz(fixed / "solve.npz")["scale"])
    assert s_fixed == pytest.approx(target, rel=1e-12)
    meta = _meta(fixed)
    assert meta[FIXED_SCALE_KEY] == record
    assert (meta["registration"]["scale_source"], meta["registration"]["init_scale"], meta["registration"]["scale"]
            ) == ("fixed", target, target)
    assert not np.allclose(load_npz(fixed / "solve.npz")["poses"], load_npz(prod / "solve.npz")["poses"])


def test_mutation_the_scale_moves_when_the_lm_does_not_hold_it(tmp_path: Path, small_clip: SyntheticClip,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """The held-scale assertion above is sensitive: with `_fix_scale` mutated to a no-op the solve pulls the scale
    away from a 30% wrong start, and the same assertion fails."""
    d = tmp_path / "c"
    write_synthetic_solve_inputs(d, small_clip)
    make_solve_worker(FAST)("c", d)
    target = 1.3 * float(load_npz(d / "solve.npz")["scale"])
    write_instance_scale(d, {"scale": target})
    monkeypatch.setattr(lm, "_fix_scale", lambda sys: sys)
    make_solve_worker(FIXED)("c", d)
    moved = float(load_npz(d / "solve.npz")["scale"])
    assert abs(moved / target - 1.0) > 1e-3


def test_missing_file_is_an_error_and_writes_nothing(tmp_path: Path, small_clip: SyntheticClip) -> None:
    d = tmp_path / "c"
    write_synthetic_solve_inputs(d, small_clip)
    with pytest.raises(FileNotFoundError, match="instance_scale.json"):
        make_solve_worker(FIXED)("c", d)
    assert not (d / "solve.npz").exists() and not (d / "solve.json").exists()
