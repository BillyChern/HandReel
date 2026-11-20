"""An option of the per-frame re-anchor: `dead_frame_branch` (None = the branches as merged, byte for byte).

False turns off the UNFLAGGED (dead-frame) branch only: an unflagged clip keeps its solve, a flagged clip is
re-anchored exactly as before. The record names the option only when it is set.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastdo.engine.per_frame_reanchor import (
    TRACKER_VIS_KEY,
    apply_per_frame_reanchor,
    reanchor_poses,
)
from fastdo.engine.workers_solve import PerFrameReanchorConfig, SolveStageConfig
from tests.engine.test_per_frame_reanchor import (
    _ROOT_CONFIG,
    G,
    _clip_solution,
    _depth_obs,
    _obs,
    _poses,
    _ray_point,
)

C = np.array([0.01, -0.02, 0.03])
V2 = dataclasses.replace(G, occluded_area_frac=0.25, max_log_shift=0.3)
V3 = dataclasses.replace(V2, dead_frame_branch=False)


def _clip() -> tuple[np.ndarray, object]:
    z = 0.8
    on_mask = (60.0, 45.0)
    points = [_ray_point(*on_mask, z * math.exp(0.5)),     # 0 alive, out of band
              _ray_point(*on_mask, z * math.exp(0.5)),     # 1 dead, far          -> band edge under V2
              _ray_point(80.0, 45.0, z),                   # 2 dead, off mask     -> mask ray under V2
              _ray_point(*on_mask, z)]                     # 3 alive, on mask, in band
    return _poses(points), _obs([z] * 4, [10, 0, 0, 10])


def test_off_keeps_every_unflagged_frame() -> None:
    poses, ob = _clip()
    _, moved_v2, _ = reanchor_poses(poses, 2.0, C, ob, False, V2)
    new, moved, shifts = reanchor_poses(poses, 2.0, C, ob, False, V3)
    assert moved_v2 == [1, 2]
    assert moved == [] and shifts.shape == (0, 3) and np.array_equal(new, poses)


def test_off_leaves_the_flagged_branch_unchanged() -> None:
    poses, ob = _clip()
    for pfr in (V2, dataclasses.replace(V2, dead_frame_branch=True)):
        want, want_moved, want_shifts = reanchor_poses(poses, 2.0, C, ob, True, pfr)
        new, moved, shifts = reanchor_poses(poses, 2.0, C, ob, True, V3)
        assert moved == want_moved == [0, 1, 2]
        assert np.array_equal(new, want) and np.array_equal(shifts, want_shifts)


def test_true_equals_none() -> None:
    poses, ob = _clip()
    on = dataclasses.replace(V2, dead_frame_branch=True)
    for flagged in (False, True):
        a = reanchor_poses(poses, 2.0, C, ob, flagged, V2)
        b = reanchor_poses(poses, 2.0, C, ob, flagged, on)
        assert a[1] == b[1] and np.array_equal(a[0], b[0])


def test_record_names_the_option_only_when_set() -> None:
    obs = _depth_obs([1.0, 1.0, 1.0, 1.0, 0.0])
    mesh, solution = _clip_solution([1.0, 1.0, 1.5, 1.0, 1.0])     # unflagged, frame 2 dead and 50 % far
    ctx = {TRACKER_VIS_KEY: np.array([9, 9, 0, 9, 9]), "K": torch.eye(3)}
    kept, rec = apply_per_frame_reanchor(mesh, obs, solution, ctx, V2)
    assert "dead_frame_branch" not in rec and rec["frames"] == [2]
    kept, rec = apply_per_frame_reanchor(mesh, obs, solution, ctx, V3)
    assert rec["dead_frame_branch"] is False and rec["frames"] == [] and rec["dead_frames"] == [2]
    assert kept is solution


def test_config_block_accepts_and_validates_the_key() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    root.solve_stage.per_frame_reanchor = {**dataclasses.asdict(V2), "dead_frame_branch": False}
    cfg = SolveStageConfig.from_mapping(root.solve_stage).per_frame_reanchor
    assert cfg == V3 and cfg.safety_options()["dead_frame_branch"] is False
    base = {"threshold": 0.25, "band": 0.2, "perp_k": 1.0, "vis_min": 5}
    for bad in (0, 1, "false", 0.5):
        with pytest.raises(ValueError, match="dead_frame_branch"):
            SolveStageConfig.from_mapping({"per_frame_reanchor": {**base, "dead_frame_branch": bad}})
    assert PerFrameReanchorConfig().dead_frame_branch is None
    # the shipped production default is V2 with the branch as merged (key absent)
    assert SolveStageConfig.from_mapping(OmegaConf.load(_ROOT_CONFIG).solve_stage).per_frame_reanchor == V2
