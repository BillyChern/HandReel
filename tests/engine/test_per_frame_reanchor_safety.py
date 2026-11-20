"""Safety options of the per-frame re-anchor G (`occluded_area_frac`, `max_log_shift`).

Both default to None, which is the original G byte for byte (the solve.json block gains no key). With them set: a
low-visibility frame (mask area below the fraction of the clip's median mask area) is never pushed away from the
camera, and every frame's along-ray move is capped on its own target ray. Hand-built observations for the frame logic,
the config block, and the engine solve path.
"""

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

from fastdo.engine.per_frame_reanchor import (
    REANCHOR_KEY,
    TRACKER_VIS_KEY,
    FrameObservations,
    apply_per_frame_reanchor,
    low_visibility,
    reanchor_poses,
)
from fastdo.engine.worker_support import load_npz
from fastdo.engine.workers_solve import PerFrameReanchorConfig, SolveStageConfig, make_solve_worker
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import SMALL_CLIP
from tests.engine.test_per_frame_reanchor import (
    _ROOT_CONFIG,
    FAST,
    G,
    K,
    _centroids,
    _clip_dir,
    _clip_solution,
    _depth_obs,
    _npz_identical,
    _poses,
    _ray_point,
    _Solve,
    _written_centroid_z,
)

C = np.array([0.01, -0.02, 0.03])
OCC = dataclasses.replace(G, occluded_area_frac=0.25)
CAP = dataclasses.replace(G, max_log_shift=0.3)


@pytest.fixture(scope="module")
def small_clip() -> SyntheticClip:
    return make_synthetic_clip(SMALL_CLIP, seed=0)


def _obs(z_obs: list[float], nvis: list[int], r: list[float], centroid: tuple[float, float] = (60.0, 45.0)
         ) -> FrameObservations:
    n = len(z_obs)
    return FrameObservations(z_obs=np.array(z_obs, float), mask_centroid=np.tile(centroid, (n, 1)).astype(float),
                             r_mask=np.array(r, float), nvis=np.array(nvis, np.int64), K=K)


def test_low_visibility_is_relative_to_the_median_non_empty_mask() -> None:
    ob = _obs([1.0] * 6, [0] * 6, [4.0, 4.0, 4.0, 1.9, 2.1, 0.0])    # median non-empty area 16 pi: floor 4 pi
    assert low_visibility(ob, 0.25).tolist() == [False, False, False, True, False, True]
    assert not low_visibility(ob, None).any()
    assert not low_visibility(_obs([1.0], [0], [0.0]), 0.25).any()     # no non-empty mask: nothing is low-visibility


def test_an_occluded_frame_is_never_pushed_away_but_may_come_nearer_or_sideways() -> None:
    z = 0.8
    on_mask = (60.0, 45.0)
    points = [_ray_point(*on_mask, z * math.exp(-0.5)),    # 0 low-vis, z_obs FARTHER than the solve -> untouched
              _ray_point(*on_mask, z * math.exp(0.5)),     # 1 low-vis, z_obs nearer                -> z_obs
              _ray_point(*on_mask, z * math.exp(-0.5)),    # 2 visible, z_obs farther               -> z_obs (G's move)
              _ray_point(60.0, 50.0, z * math.exp(0.1))]   # 3 low-vis, in band, off mask (dead)    -> sideways
    points += [_ray_point(*on_mask, z)] * 3                # 4-6 visible, in band, on mask          -> untouched
    poses = _poses(points)
    ob = _obs([z] * 7, [0] * 7, [1.0, 1.0, 4.0, 1.0, 4.0, 4.0, 4.0])   # frames 0, 1, 3: 1/16 of the median area
    for flagged, expect in ((True, [1, 2, 3]), (False, [1, 2, 3])):
        new, moved, _ = reanchor_poses(poses, 2.0, C, ob, flagged, OCC)
        g_new, g_moved, _ = reanchor_poses(poses, 2.0, C, ob, flagged, G)
        assert 0 in g_moved and moved == expect
        assert np.array_equal(new[0], poses[0])
        np.testing.assert_array_equal(new[[1, 2]], g_new[[1, 2]])         # nearer / visible moves are G's
    new, _, _ = reanchor_poses(poses, 2.0, C, ob, False, OCC)
    p = _centroids(new)[3]
    assert p[2] == pytest.approx(_centroids(poses)[3][2], rel=1e-12)       # same depth, moved onto the mask ray
    np.testing.assert_allclose(p, _ray_point(*on_mask, p[2]), atol=1e-12)


def test_max_log_shift_caps_the_depth_change_on_the_target_ray() -> None:
    z = 0.7
    on_mask = (60.0, 45.0)
    points = [_ray_point(*on_mask, z * math.exp(0.5)),     # target 0.5 nearer   -> capped at solved z e^-0.3
              _ray_point(80.0, 45.0, z * math.exp(-0.6)),  # target 0.6 farther  -> capped at solved z e^+0.3, mask ray
              _ray_point(*on_mask, z * math.exp(0.25))]    # target 0.25 nearer  -> z_obs (under the cap)
    poses = _poses(points)
    new, moved, _ = reanchor_poses(poses, 2.0, C, _obs([z] * 3, [40] * 3, [4.0] * 3), True, CAP)
    assert moved == [0, 1, 2]
    p = _centroids(new)
    np.testing.assert_allclose(p[0], _ray_point(*on_mask, z * math.exp(0.5) * math.exp(-0.3)), atol=1e-12)
    np.testing.assert_allclose(p[1], _ray_point(*on_mask, z * math.exp(-0.6) * math.exp(0.3)), atol=1e-12)
    np.testing.assert_allclose(p[2], _ray_point(*on_mask, z), atol=1e-12)
    assert np.array_equal(new[:, :3, :3], poses[:, :3, :3])


def test_g_record_is_unchanged_without_options_and_names_them_when_set() -> None:
    obs = _depth_obs([1.0, 1.0, 1.0, 1.0, 0.0])
    mesh, solution = _clip_solution([1.0, 1.0, 1.5, 1.0, 1.0])
    ctx = {TRACKER_VIS_KEY: np.array([9, 9, 0, 9, 9]), "K": torch.eye(3)}
    _, rec = apply_per_frame_reanchor(mesh, obs, solution, ctx, G)
    assert list(rec)[:6] == ["threshold", "band", "perp_k", "vis_min", "min_points", "certificate"]
    assert not {"occluded_area_frac", "max_log_shift", "low_visibility_frames"} & set(rec)
    both = dataclasses.replace(G, occluded_area_frac=0.25, max_log_shift=0.3)
    kept, rec = apply_per_frame_reanchor(mesh, obs, solution, ctx, both)
    # frame 4's mask is empty (area 0): low-visibility, and never moved anyway (no observed depth)
    assert (rec["occluded_area_frac"], rec["max_log_shift"], rec["low_visibility_frames"]) == (0.25, 0.3, [4])
    assert rec["frames"] == [2]
    assert kept.poses[2, 2, 3].item() == pytest.approx(math.exp(0.2) - 0.1, abs=1e-6)   # band edge (under the cap)


def test_config_block_accepts_the_optional_keys_and_validates_them() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    root.solve_stage.per_frame_reanchor = {"threshold": 0.25, "band": 0.2, "perp_k": 1.0, "vis_min": 5}
    cfg = SolveStageConfig.from_mapping(root.solve_stage).per_frame_reanchor
    assert cfg == G and (cfg.occluded_area_frac, cfg.max_log_shift) == (None, None) and cfg.safety_options() == {}
    root.solve_stage.per_frame_reanchor = {**dataclasses.asdict(G), "occluded_area_frac": 0.25, "max_log_shift": 1}
    cfg = SolveStageConfig.from_mapping(root.solve_stage).per_frame_reanchor
    assert cfg.safety_options() == {"occluded_area_frac": 0.25, "max_log_shift": 1.0}
    assert isinstance(cfg.max_log_shift, float)
    base = {"threshold": 0.25, "band": 0.2, "perp_k": 1.0, "vis_min": 5}
    for key, bad in (("occluded_area_frac", 0.0), ("occluded_area_frac", 1.5), ("occluded_area_frac", True),
                     ("max_log_shift", 0.0), ("max_log_shift", -0.3), ("max_log_shift", math.inf),
                     ("max_log_shift", math.nan)):
        with pytest.raises(ValueError, match=key):
            SolveStageConfig.from_mapping({"per_frame_reanchor": {**base, key: bad}})
    with pytest.raises(ValueError, match="missing"):
        SolveStageConfig.from_mapping({"per_frame_reanchor": {"threshold": 0.25, "band": 0.2, "perp_k": 1.0,
                                                              "occluded_area_frac": 0.25}})


def test_engine_path_honours_the_options(tmp_path: Path, small_clip: SyntheticClip) -> None:
    g_dir = _clip_dir(tmp_path / "g", small_clip)
    occ_dir, cap_dir = tmp_path / "occ" / "c", tmp_path / "cap" / "c"
    shutil.copytree(g_dir, occ_dir)
    shutil.copytree(g_dir, cap_dir)
    produced = {}
    for d, pfr in ((g_dir, G), (occ_dir, OCC), (cap_dir, CAP)):
        solve = _Solve(factor=1.5)                           # every frame 50 % far: flagged, every frame moves
        make_solve_worker(dataclasses.replace(FAST, per_frame_reanchor=pfr), solve_fn=solve)("c", d)
        produced[d] = solve.returned[0]
    blocks = {d: json.loads((d / "solve.json").read_text())[REANCHOR_KEY] for d in (g_dir, occ_dir, cap_dir)}
    assert all(b["flagged"] and b["frames"] == list(range(6)) for b in blocks.values())
    # the synthetic masks are all visible: the occlusion rule changes nothing, and its block names it
    assert blocks[occ_dir]["low_visibility_frames"] == [] and blocks[occ_dir]["occluded_area_frac"] == 0.25
    assert _npz_identical(g_dir / "solve.npz", occ_dir / "solve.npz", skip=("seconds",))
    # the 0.405 log move toward z_obs is capped at 0.3 on the mask ray
    samples = load_npz(cap_dir / "samples.npz")["points"].astype(np.float64)
    sol = produced[cap_dir]
    p = float(sol.scale) * sol.poses.numpy().astype(np.float64)[:, :3, :3] @ samples.mean(0) + \
        sol.poses.numpy().astype(np.float64)[:, :3, 3]
    np.testing.assert_allclose(_written_centroid_z(cap_dir), p[:, 2] * math.exp(-0.3), rtol=1e-5)
    assert blocks[cap_dir]["max_log_shift"] == 0.3 and "low_visibility_frames" not in blocks[cap_dir]


def test_default_config_file_ships_v2() -> None:
    # after a subject-07 one-shot PASS the production config ships V2; the dataclass default stays G-off.
    v2 = PerFrameReanchorConfig(threshold=0.25, band=0.2, perp_k=1.0, vis_min=5, occluded_area_frac=0.25,
                                max_log_shift=0.3)
    assert SolveStageConfig.from_mapping(OmegaConf.load(_ROOT_CONFIG).solve_stage).per_frame_reanchor == v2
    assert SolveStageConfig().per_frame_reanchor is None
    assert PerFrameReanchorConfig().safety_options() == {}
