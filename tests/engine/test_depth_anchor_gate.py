"""Certificate-gated depth anchor (`fastdo.engine.depth_anchor_gate`, `solve_stage.depth_anchor_gate`).

The certificate against hand-computed depths; the config block; both engine solve paths: gate off and gate on with an
unflagged clip write a byte-identical solve.npz; a flagged clip is re-solved from the same `SolveCall` with only the
anchor fields changed, and its certificate falls; a path that does not run the gate is refused by `finish_solve`.

A flagged synthetic clip: the synthetic solver recovers depth from any init we tried (x1.4-x2.0 translations, CPU
probe), so the production solve's along-ray OFFSET (an absorbing state of the solve) is injected by the test's
solve function -- the real solve with its translations scaled by `OFFSET` -- while the gated re-solve is the real
solver from the unmodified init.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from fastdo.engine import depth_anchor_gate as gate_mod
from fastdo.engine.batched_stages import SolveBatchStage
from fastdo.engine.depth_anchor_gate import (
    GATE_KEY,
    MIN_VALID_FRAMES,
    DepthCertificate,
    depth_certificate,
    gated_solver_config,
)
from fastdo.engine.worker_support import load_npz
from fastdo.engine.workers_solve import (
    DepthAnchorGateConfig,
    SolveCall,
    SolveStageConfig,
    finish_solve,
    make_solve_worker,
    prepare_solve,
)
from fastdo.solver.api import (
    MeshSamples,
    SolverConfig,
    TrajectoryObservations,
    TrajectorySolution,
    solve_trajectory,
)
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    SMALL_SOLVE,
    fake_solution,
    write_synthetic_solve_inputs,
)

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
GATE = DepthAnchorGateConfig(threshold=0.25, lam=500.0, band=0.2)
FAST = dataclasses.replace(SMALL_SOLVE, solver={"iters_coarse": 3, "iters_mid": 2, "iters_fine": 3})
OFFSET = 1.5          # the injected along-ray offset of the production solve: log 1.5 = 0.405 >= 0.25


@pytest.fixture(scope="module")
def small_clip() -> SyntheticClip:
    return make_synthetic_clip(SMALL_CLIP, seed=0)


def _clip_dir(root: Path, clip: SyntheticClip, name: str = "c") -> Path:
    write_synthetic_solve_inputs(root / name, clip)
    return root / name


def _offset(solution: TrajectorySolution, factor: float = OFFSET) -> TrajectorySolution:
    poses = solution.poses.clone()
    poses[:, :3, 3] *= factor
    return dataclasses.replace(solution, poses=poses)


def _npz_identical(a: Path, b: Path, skip: tuple[str, ...] = ()) -> bool:
    """Every array of two npz files byte-identical (dtype, shape, bytes), except the keys in `skip`."""
    x, y = load_npz(a), load_npz(b)
    return x.keys() == y.keys() and all(x[k].dtype == y[k].dtype and x[k].shape == y[k].shape
                                        and x[k].tobytes() == y[k].tobytes() for k in x if k not in skip)


# --- the certificate -------------------------------------------------------------------------------------------------


def _depth_obs(z_obs: list[float], hw: tuple[int, int] = (6, 8), empty: tuple[int, ...] = ()) -> TrajectoryObservations:
    """Frames whose masked pointmap pixels all sit at depth z_obs[t] (an empty mask on the frames in `empty`)."""
    t, (h, w) = len(z_obs), hw
    masks = torch.zeros(t, h, w, dtype=torch.bool)
    masks[:, 1:5, 2:6] = True                                        # 16 pixels >= DEPTH_MIN_POINTS
    for k in empty:
        masks[k] = False
    pm = torch.zeros(t, h, w, 3)
    pm[..., 2] = torch.tensor(z_obs)[:, None, None]
    return TrajectoryObservations(intrinsics=torch.eye(3), masks=masks, hand_masks=torch.zeros_like(masks),
                                  pointmaps=pm, tracks=torch.zeros(t, 1, 2), track_vis=torch.zeros(t, 1, dtype=torch.bool),
                                  anchors=torch.zeros(1, 3))


def _placed(z_sol: list[float], scale: float = 2.0) -> tuple[Tensor, TrajectorySolution]:
    """Mesh points with centroid (0.1, 0.1, 0.1) and poses placing it at camera depth z_sol[t] (identity rotation)."""
    points = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.1, 0.0, 0.2], [0.1, 0.2, 0.0]])
    poses = torch.eye(4).repeat(len(z_sol), 1, 1)
    poses[:, 2, 3] = torch.tensor(z_sol) - scale * 0.1
    return points, fake_solution(poses, scale)


def test_certificate_is_the_median_log_depth_offset_of_the_posed_centroid() -> None:
    z_obs = [0.8, 0.7, 0.75, 0.9, 0.6]
    d = [0.30, 0.10, 0.28, -0.05, 0.40]
    points, sol = _placed([z * math.exp(e) for z, e in zip(z_obs, d, strict=True)])

    cert = depth_certificate(points, _depth_obs(z_obs), sol)

    assert (cert.n_valid, cert.n_frames) == (5, 5)
    assert cert.median_d is not None and cert.median_d == pytest.approx(float(np.median(d)), abs=1e-6)
    assert cert.value == abs(cert.median_d)
    assert cert.flagged(0.25) and not cert.flagged(0.29)
    neg = depth_certificate(points, _depth_obs(z_obs), _placed([z * math.exp(-0.3) for z in z_obs])[1])
    assert neg.median_d == pytest.approx(-0.3, abs=1e-6) and neg.value == pytest.approx(0.3, abs=1e-6)


def test_certificate_excludes_frames_without_depth_and_needs_three() -> None:
    z_obs = [0.8, 0.8, 0.8, 0.8]
    points, sol = _placed([0.8, 0.8 * math.exp(0.3), 0.8 * math.exp(0.3), 0.8 * math.exp(0.3)])

    cert = depth_certificate(points, _depth_obs(z_obs, empty=(1,)), sol)
    assert (cert.n_valid, cert.median_d) == (3, pytest.approx(0.3, abs=1e-6))     # frame 1 excluded, not 0.3 x 3/4
    none = depth_certificate(points, _depth_obs(z_obs, empty=(1, 2)), sol)
    assert MIN_VALID_FRAMES == 3 and (none.n_valid, none.value, none.median_d) == (2, None, None)
    assert not none.flagged(0.0)


def test_flagged_is_value_at_or_above_the_threshold() -> None:
    at = DepthCertificate(value=0.25, median_d=-0.25, n_valid=5, n_frames=5)
    assert at.flagged(0.25) and not at.flagged(0.2500001)
    assert not DepthCertificate(value=0.2499999, median_d=0.2499999, n_valid=5, n_frames=5).flagged(0.25)


# --- the config block ------------------------------------------------------------------------------------------------


def test_config_block_defaults_on_and_validates() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    assert SolveStageConfig.from_mapping(root.solve_stage).depth_anchor_gate == GATE     # production default
    root.solve_stage.depth_anchor_gate = None                                          # null restores the old stage
    assert SolveStageConfig.from_mapping(root.solve_stage).depth_anchor_gate is None
    root.solve_stage.depth_anchor_gate = {"threshold": 0.25, "lam": 500, "band": 0.2}
    cfg = SolveStageConfig.from_mapping(root.solve_stage)
    assert cfg.depth_anchor_gate == GATE and isinstance(cfg.depth_anchor_gate.lam, float)
    assert cfg.solver_config() == SolveStageConfig.from_mapping(OmegaConf.load(_ROOT_CONFIG).solve_stage).solver_config()
    for bad, match in (({"threshold": 0.25, "lam": 500.0}, "missing"), ({**dataclasses.asdict(GATE), "x": 1}, "unknown"),
                       ({"threshold": 0.0, "lam": 500.0, "band": 0.2}, "threshold"),
                       ({"threshold": 0.25, "lam": float("nan"), "band": 0.2}, "lam"),
                       ({"threshold": 0.25, "lam": 500.0, "band": -1.0}, "band"), (True, "null or")):
        with pytest.raises(ValueError, match=match):
            SolveStageConfig.from_mapping({"depth_anchor_gate": bad})
    with pytest.raises(ValueError, match="production first solve"):
        SolveStageConfig(solver={"lam_depth_anchor": 500.0}, depth_anchor_gate=GATE)
    with pytest.raises(ValueError, match="already has"):
        gated_solver_config(SolverConfig(lam_depth_anchor=5.0), GATE)
    gated = gated_solver_config(SolverConfig(seed=3), GATE)
    assert gated == dataclasses.replace(SolverConfig(seed=3), lam_depth_anchor=500.0, depth_anchor_band=0.2)


# --- the per-clip worker (single-stream engine path) ----------------------------------------------------------------


class _Solves:
    """The real solver, recording each call; `offset` scales the translations of the PRODUCTION solve's output (the
    injected along-ray offset), never the anchored re-solve's."""

    def __init__(self, offset: float | None = None) -> None:
        self.offset = offset
        self.calls: list[tuple[Tensor, float, SolverConfig]] = []
        self.returned: list[TrajectorySolution] = []

    def __call__(self, mesh: MeshSamples, obs: TrajectoryObservations, init_poses: Tensor, init_scale: float,
                 cfg: SolverConfig) -> TrajectorySolution:
        self.calls.append((init_poses.clone(), init_scale, cfg))
        solution = solve_trajectory(mesh, obs, init_poses, init_scale, cfg)
        if self.offset is not None and cfg.lam_depth_anchor == 0:
            solution = _offset(solution, self.offset)
        self.returned.append(solution)
        return solution


def test_worker_gate_off_and_unflagged_write_the_production_solve(tmp_path: Path, small_clip: SyntheticClip) -> None:
    off = _clip_dir(tmp_path / "off", small_clip)
    on = tmp_path / "on" / "c"
    shutil.copytree(off, on)
    solves = _Solves()

    make_solve_worker(FAST, solve_fn=solves)("c", off)
    make_solve_worker(dataclasses.replace(FAST, depth_anchor_gate=GATE), solve_fn=solves)("c", on)

    assert len(solves.calls) == 2 and all(c[2].lam_depth_anchor == 0.0 for c in solves.calls)
    # two separate real solves: `seconds` is their wall clock (qc37's SOLVE_VOLATILE); every other array is bytes-equal
    assert _npz_identical(off / "solve.npz", on / "solve.npz", skip=("seconds",))
    meta_off, meta_on = (json.loads((d / "solve.json").read_text()) for d in (off, on))
    assert GATE_KEY not in meta_off
    block = meta_on.pop(GATE_KEY)
    for meta in (meta_off, meta_on):
        meta["solver"].pop("seconds")
    assert meta_on == meta_off
    assert block["flagged"] is False and 0.0 <= block["certificate"]["value"] < 0.25
    assert (block["threshold"], block["lam"], block["band"]) == (0.25, 500.0, 0.2)
    assert "resolve_certificate" not in block and "production_solve" not in block


def test_worker_flagged_clip_is_resolved_from_the_same_init(tmp_path: Path, small_clip: SyntheticClip) -> None:
    clip_dir = _clip_dir(tmp_path, small_clip)
    solves = _Solves(offset=OFFSET)

    make_solve_worker(dataclasses.replace(FAST, depth_anchor_gate=GATE), solve_fn=solves)("c", clip_dir)

    (p0, s0, c0), (p1, s1, c1) = solves.calls
    assert torch.equal(p0, p1) and s0 == s1                                # the production solve's init, unchanged
    assert c0 == FAST.solver_config() and c1 == gated_solver_config(c0, GATE)
    block = json.loads((clip_dir / "solve.json").read_text())[GATE_KEY]
    assert block["flagged"] is True and block["certificate"]["value"] >= 0.25
    assert block["resolve_certificate"]["value"] < 0.5 * block["certificate"]["value"]
    written = load_npz(clip_dir / "solve.npz")
    assert np.array_equal(written["poses"], solves.returned[1].poses.numpy())  # the re-solve is what was written
    assert float(written["scale"]) == solves.returned[1].scale


# --- the batched engine path ----------------------------------------------------------------------------------------


class _FakeMany:
    """`solve_many` returning each clip's init (`fake_solution`), with the translations of the clips in `offset`
    scaled by OFFSET on the production call only; records (clip count, config) per call."""

    def __init__(self, offset: tuple[int, ...] = ()) -> None:
        self.offset, self.calls = offset, []  # type: ignore[var-annotated]

    def __call__(self, meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
                 init_scales: list[float], config: SolverConfig) -> list[TrajectorySolution]:
        self.calls.append(([p.clone() for p in init_poses], config))
        out = [fake_solution(p, s) for p, s in zip(init_poses, init_scales, strict=True)]
        if config.lam_depth_anchor == 0:
            out = [_offset(s) if i in self.offset else s for i, s in enumerate(out)]
        return out


def _batched_dirs(root: Path, clip: SyntheticClip, n: int = 2) -> list[tuple[str, Path]]:
    first = _clip_dir(root, clip, "c0")
    for i in range(1, n):
        shutil.copytree(first, root / f"c{i}")
    return [(f"c{i}", root / f"c{i}") for i in range(n)]


def _stage(cfg: SolveStageConfig, fake: _FakeMany, gate: DepthAnchorGateConfig | None) -> SolveBatchStage:
    return SolveBatchStage(functools.partial(prepare_solve, cfg=cfg), functools.partial(finish_solve, cfg=cfg),
                           cfg.solver_config(), fake, depth_anchor_gate=gate)


def test_batched_stage_resolves_only_flagged_clips(tmp_path: Path, small_clip: SyntheticClip) -> None:
    off = _batched_dirs(tmp_path / "off", small_clip)
    on = _batched_dirs(tmp_path / "on", small_clip)
    gated = dataclasses.replace(SMALL_SOLVE, depth_anchor_gate=GATE)
    fake_off, fake_on = _FakeMany(offset=(1,)), _FakeMany(offset=(1,))

    assert _stage(SMALL_SOLVE, fake_off, None).run_batch(off, "b-off").errors == {}
    assert _stage(gated, fake_on, GATE).run_batch(on, "b-on").errors == {}

    assert len(fake_off.calls) == 1 and len(fake_on.calls) == 2
    (inits, cfg0), (re_inits, cfg1) = fake_on.calls
    assert len(re_inits) == 1 and torch.equal(re_inits[0], inits[1])      # only c1, from its own init
    assert cfg0 == SMALL_SOLVE.solver_config() and cfg1 == gated_solver_config(cfg0, GATE)
    assert _npz_identical(off[0][1] / "solve.npz", on[0][1] / "solve.npz")  # unflagged: the production solve
    assert not _npz_identical(off[1][1] / "solve.npz", on[1][1] / "solve.npz")
    blocks = [json.loads((d / "solve.json").read_text())[GATE_KEY] for _, d in on]
    assert [b["flagged"] for b in blocks] == [False, True]
    assert blocks[1]["resolve_certificate"]["value"] < 0.25 <= blocks[1]["certificate"]["value"]
    assert all(GATE_KEY not in json.loads((d / "solve.json").read_text()) for _, d in off)


def test_a_path_that_skips_the_gate_is_refused(tmp_path: Path, small_clip: SyntheticClip) -> None:
    clips = _batched_dirs(tmp_path, small_clip, n=1)
    gated = dataclasses.replace(SMALL_SOLVE, depth_anchor_gate=GATE)

    outcome = _stage(gated, _FakeMany(), None).run_batch(clips, "b-skip")    # finish built with the gate, stage without

    assert "does not honour the gate" in outcome.errors["c0"] and not (clips[0][1] / "solve.npz").exists()
    call = prepare_solve("c0", clips[0][1], SMALL_SOLVE)
    marked = gate_mod.with_gate_record(call, {"flagged": False})
    assert isinstance(marked, SolveCall) and call.context.get(GATE_KEY) is None
    with pytest.raises(ValueError, match="carries a"):
        finish_solve("c0", clips[0][1], marked, fake_solution(call.init_poses, call.init_scale), SMALL_SOLVE)


def test_build_batched_stages_passes_the_gate(tmp_path: Path) -> None:
    from fastdo.bench.bench_batched import build_batched_stages

    cfg = OmegaConf.load(_ROOT_CONFIG)
    assert isinstance(cfg, DictConfig)
    OmegaConf.update(cfg, "paths.root", str(tmp_path))
    cfg.solve_stage.depth_anchor_gate = dataclasses.asdict(GATE)
    stages = build_batched_stages(cfg, {"solve": lambda c, d: None}, tmp_path / "jobs", tmp_path / "gpu.lock")
    assert isinstance(stages["solve"], SolveBatchStage) and stages["solve"].depth_anchor_gate == GATE
    cfg.solve_stage.depth_anchor_gate = None
    stages = build_batched_stages(cfg, {"solve": lambda c, d: None}, tmp_path / "jobs", tmp_path / "gpu.lock")
    assert stages["solve"].depth_anchor_gate is None  # type: ignore[union-attr]
