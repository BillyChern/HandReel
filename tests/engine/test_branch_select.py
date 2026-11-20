"""`solve_stage.branch_select`: default off byte for byte, K=1 is production, the ranked registration hypotheses,
the chosen candidate reaches solve.npz on both solve paths, and the refusals."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastdo.engine import branch_select as bs
from fastdo.engine.batched_stages import SolveBatchStage
from fastdo.engine.worker_support import load_npz
from fastdo.engine.workers_solve import (
    BranchSelectConfig,
    DepthAnchorGateConfig,
    SolveStageConfig,
    finish_solve,
    make_solve_worker,
    prepare_solve,
    read_solve_inputs,
)
from fastdo.solver.api import solve_trajectory
from fastdo.solver.keyframe import register_keyframe_hypotheses, register_keyframe_pose
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import SMALL_CLIP, SMALL_SOLVE, write_synthetic_solve_inputs

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
FAST = dataclasses.replace(SMALL_SOLVE, solver={"iters_coarse": 3, "iters_mid": 2, "iters_fine": 3},
                           depth_anchor_gate=DepthAnchorGateConfig())
BLOCK = {"top_k": 1, "seed_offsets": [0], "scale_mults": [1.0], "selector": "mf_score", "min_rot_deg": 10.0,
         "min_shift": 0.005, "margin": 0.0}


def _with(**kw: object) -> SolveStageConfig:
    return dataclasses.replace(FAST, branch_select=BranchSelectConfig(**{**BLOCK, **kw}))  # type: ignore[arg-type]


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
    assert SolveStageConfig.from_mapping(root.solve_stage).branch_select is None
    assert SolveStageConfig().branch_select is None
    root.solve_stage.branch_select = dict(BLOCK, top_k=3, seed_offsets=[0, 1], scale_mults=[0.98, 1.0])
    cfg = SolveStageConfig.from_mapping(root.solve_stage).branch_select
    assert cfg == BranchSelectConfig(top_k=3, seed_offsets=(0, 1), scale_mults=(0.98, 1.0), selector="mf_score")
    assert cfg.runs() == ((0, 1.0), (0, 0.98), (1, 0.98), (1, 1.0))
    for bad, match in (({**BLOCK, "top_k": 0}, "top_k"), ({**BLOCK, "seed_offsets": [1]}, "seed_offsets"),
                       ({**BLOCK, "scale_mults": [0.99]}, "scale_mults"), ({**BLOCK, "selector": "gt"}, "selector"),
                       ({**BLOCK, "min_shift": 0.0}, "min_shift"), ({**BLOCK, "margin": 1.0}, "margin"),
                       ({**BLOCK, "margin": -0.1}, "margin"), ({"top_k": 2}, "missing"),
                       ({**BLOCK, "k": 1}, "unknown"), ([1], "null or")):
        with pytest.raises(ValueError, match=match):
            SolveStageConfig.from_mapping({"branch_select": bad})


def test_hypotheses_element_zero_is_register_keyframe_pose(small_clip: SyntheticClip, tmp_path: Path) -> None:
    d = tmp_path / "c"
    write_synthetic_solve_inputs(d, small_clip)
    inp = read_solve_inputs(d, FAST.n_anchor_surface, FAST.seed)
    base, k = inp.observations.observations, inp.keyframe_index
    observed = base.pointmaps[k][base.masks[k] & torch.isfinite(base.pointmaps[k]).all(-1)]
    r = FAST.registration
    args = (inp.mesh.points, inp.mesh.normals, observed, inp.K, base.masks[k])
    kw = {"scale": 1.0, "n_obs": r.n_obs, "n_mesh": r.n_mesh, "n_iters": r.n_iters, "seed": 3}
    one = register_keyframe_pose(*args, **kw)
    many = register_keyframe_hypotheses(*args, **kw, top_k=5)
    assert many[0].pose.numpy().tobytes() == one.pose.numpy().tobytes()
    assert (many[0].scale, many[0].score) == (one.scale, one.score)
    assert [h.score for h in many] == sorted(h.score for h in many) and 1 <= len(many) <= 5
    c = inp.mesh.points.double().mean(0)
    for i in range(len(many)):
        for j in range(i):
            a, b = many[i].pose.double(), many[j].pose.double()
            cos = float(((a[:3, :3].T @ b[:3, :3]).trace() - 1) / 2)
            shift = float((a[:3, :3] @ c + a[:3, 3] - b[:3, :3] @ c - b[:3, 3]).norm())
            assert cos < math.cos(math.radians(10.0)) or shift > 0.005
    assert len(register_keyframe_hypotheses(*args, **kw, top_k=5, min_rot_deg=180.0, min_shift=10.0)) == 1
    with pytest.raises(ValueError, match="top_k"):
        register_keyframe_hypotheses(*args, **kw, top_k=0)


def test_k1_is_production_byte_for_byte(tmp_path: Path, small_clip: SyntheticClip) -> None:
    a, b = tmp_path / "a" / "c", tmp_path / "b" / "c"
    write_synthetic_solve_inputs(a, small_clip)
    shutil.copytree(a, b)
    make_solve_worker(FAST)("c", a)
    make_solve_worker(_with())("c", b)
    assert _npz_identical(a / "solve.npz", b / "solve.npz", skip=("seconds",))
    ma, mb = _meta(a), _meta(b)
    block = mb.pop(bs.BRANCH_KEY)
    assert ma == mb and bs.BRANCH_KEY not in ma
    assert (block["n_candidates"], block["chosen"], block["production_kept"]) == (1, 0, True)


def test_the_chosen_candidate_is_written_on_both_paths(tmp_path: Path, small_clip: SyntheticClip,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _with(top_k=3, seed_offsets=[0, 1])
    base = tmp_path / "base" / "c"
    write_synthetic_solve_inputs(base, small_clip)
    call = prepare_solve("c", base, cfg)
    n = 1 + len(call.context[bs.BRANCH_KEY]["candidates"])
    assert n >= 3
    # force the LAST candidate: the written solve must be its registration's, on the per-clip and the batched path
    monkeypatch.setattr(bs, "candidate_values", lambda sel, calls, sols, dense: [float(len(calls) - i)
                                                                                 for i in range(len(calls))])
    one, two, prod = (tmp_path / x / "c" for x in ("one", "two", "prod"))
    for d in (one, two, prod):
        shutil.copytree(base, d)
    make_solve_worker(cfg)("c", one)
    stage = SolveBatchStage(lambda cid, p: prepare_solve(cid, p, cfg),
                            lambda cid, p, c, s: finish_solve(cid, p, c, s, cfg), cfg.solver_config(),
                            solve_many=lambda m, o, p, s, c: [solve_trajectory(*x, c) for x in zip(m, o, p, s)],
                            depth_anchor_gate=cfg.depth_anchor_gate, branch_select=cfg.branch_select)
    assert not stage.run_batch([("c", two)], "b0").errors
    make_solve_worker(FAST)("c", prod)
    m1, m2 = _meta(one), _meta(two)
    assert m1 == m2 and m1[bs.BRANCH_KEY]["chosen"] == n - 1 and not m1[bs.BRANCH_KEY]["production_kept"]
    assert m1["registration"]["score"] == m1[bs.BRANCH_KEY]["candidates"][n - 1]["reg_score"]
    assert _npz_identical(one / "solve.npz", two / "solve.npz", skip=("seconds",))
    # the mutation check: the forced candidate is a different solve from production's
    assert not np.allclose(load_npz(one / "solve.npz")["poses"], load_npz(prod / "solve.npz")["poses"])


def test_finish_refuses_a_path_that_ignored_branch_select(tmp_path: Path, small_clip: SyntheticClip) -> None:
    d = tmp_path / "c"
    write_synthetic_solve_inputs(d, small_clip)
    cfg = _with(top_k=2)
    call = prepare_solve("c", d, cfg)
    sol = solve_trajectory(call.mesh, call.obs, call.init_poses, call.init_scale, cfg.solver_config())
    from fastdo.engine.depth_anchor_gate import apply_depth_anchor_gate

    (call,), (sol,) = apply_depth_anchor_gate(["c"], [call], [sol], cfg.depth_anchor_gate, cfg.solver_config(),
                                              lambda m, o, p, s, c: [solve_trajectory(*x, c) for x in zip(m, o, p, s)])
    with pytest.raises(ValueError, match="unselected"):
        finish_solve("c", d, call, sol, cfg)
    with pytest.raises(ValueError, match="branch_select"):
        finish_solve("c", d, call, sol, FAST)
    assert not (d / "solve.npz").exists()


def test_selectors_prefer_the_true_pose(tmp_path: Path, small_clip: SyntheticClip) -> None:
    """mf_score and sil_iou are lower at the solved pose than after a 90 deg in-place turn of every frame."""
    d = tmp_path / "c"
    write_synthetic_solve_inputs(d, small_clip)
    call = prepare_solve("c", d, _with())
    sol = solve_trajectory(call.mesh, call.obs, call.init_poses, call.init_scale, FAST.solver_config())
    turn = torch.eye(4, dtype=sol.poses.dtype)
    turn[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=sol.poses.dtype)
    c = call.mesh.points.to(sol.poses.dtype).mean(0) * sol.scale
    bad_poses = sol.poses.clone()
    bad_poses[:, :3, :3] = sol.poses[:, :3, :3] @ turn[:3, :3]
    bad_poses[:, :3, 3] = sol.poses[:, :3, 3] + (sol.poses[:, :3, :3] @ c) - (bad_poses[:, :3, :3] @ c)
    bad = dataclasses.replace(sol, poses=bad_poses)
    dense = call.context[bs.BRANCH_KEY]["dense"]
    for sel in ("mf_score", "sil_iou", "sil_cov"):
        good_v, bad_v = bs.candidate_values(sel, [call, call], [sol, bad], dense)
        assert good_v < bad_v, sel
    far_poses = sol.poses.clone()
    far_poses[:, 2, 3] += 0.05                                   # the medoid reads centroid tracks: move one
    far = dataclasses.replace(sol, poses=far_poses)
    med = bs.candidate_values("medoid", [call, call, call], [sol, sol, far], dense)
    assert med[0] == med[1] < med[2]


def test_the_margin_keeps_production_unless_clearly_better(tmp_path: Path, small_clip: SyntheticClip,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Values [1.0, 0.95, 0.85]: margin 0 takes candidate 2 (argmin); margin 0.1 also takes it (0.85 < 0.9); margin
    0.2 keeps production (0.85 >= 0.8)."""
    base = tmp_path / "base" / "c"
    write_synthetic_solve_inputs(base, small_clip)
    monkeypatch.setattr(bs, "candidate_values", lambda sel, calls, sols, dense: [1.0, 0.95, 0.85][:len(calls)])
    for margin, want in ((0.0, 2), (0.1, 2), (0.2, 0)):
        d = tmp_path / f"m{margin}" / "c"
        shutil.copytree(base, d)
        cfg = _with(top_k=1, seed_offsets=[0, 1, 2], margin=margin)
        make_solve_worker(cfg)("c", d)
        block = _meta(d)[bs.BRANCH_KEY]
        assert (block["n_candidates"], block["chosen"], block["margin"]) == (3, want, margin)
