"""Per-frame re-anchor "G" (`fastdo.engine.per_frame_reanchor`, `solve_stage.per_frame_reanchor`).

The dead-frame / band / mask-ray logic on hand-built observations; the observations from real `TrajectoryObservations`;
the clip flag's source; the config block; both engine solve paths: G off writes today's solve stage, G on with nothing
to move is byte-identical to G off, dead frames of an unflagged clip and every out frame of a flagged clip move as
specified, and a call prepared without G is refused by `finish_solve`.
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
from omegaconf import OmegaConf
from torch import Tensor

from fastdo.engine.batched_stages import SolveBatchStage
from fastdo.engine.depth_anchor_gate import GATE_KEY
from fastdo.engine.per_frame_reanchor import (
    REANCHOR_KEY,
    TRACKER_VIS_KEY,
    FrameObservations,
    apply_per_frame_reanchor,
    frame_observations,
    reanchor_poses,
)
from fastdo.engine.worker_support import load_npz, save_npz
from fastdo.engine.workers_solve import (
    DepthAnchorGateConfig,
    PerFrameReanchorConfig,
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
from fastdo.solver.batch import frame_depths, valid_pixels_of
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    SMALL_SOLVE,
    fake_solution,
    write_synthetic_solve_inputs,
)

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
G = PerFrameReanchorConfig(threshold=0.25, band=0.2, perp_k=1.0, vis_min=5)
GATE = DepthAnchorGateConfig(threshold=0.25, lam=500.0, band=0.2)
FAST = dataclasses.replace(SMALL_SOLVE, solver={"iters_coarse": 3, "iters_mid": 2, "iters_fine": 3})
F, CX, CY = 100.0, 50.0, 40.0
K = np.array([[F, 0.0, CX], [0.0, F, CY], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def small_clip() -> SyntheticClip:
    return make_synthetic_clip(SMALL_CLIP, seed=0)


# --- the frame logic on hand-built observations ----------------------------------------------------------------------


def _ray_point(u: float, v: float, z: float) -> np.ndarray:
    return np.array([(u - CX) / F * z, (v - CY) / F * z, z])


def _obs(z_obs: list[float], nvis: list[int], centroid: tuple[float, float] = (60.0, 45.0), r: float = 4.0
         ) -> FrameObservations:
    n = len(z_obs)
    return FrameObservations(z_obs=np.array(z_obs, float), mask_centroid=np.tile(centroid, (n, 1)).astype(float),
                             r_mask=np.full(n, r), nvis=np.array(nvis, np.int64), K=K)


def _poses(points: list[np.ndarray]) -> np.ndarray:
    """Poses with a 30-degree rotation about z whose template centroid (c below, scale 2) lands at `points`."""
    a = math.radians(30.0)
    rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    poses = np.tile(np.eye(4), (len(points), 1, 1))
    for t, p in enumerate(points):
        poses[t, :3, :3] = rot
        poses[t, :3, 3] = p - 2.0 * rot @ C
    return poses


C = np.array([0.01, -0.02, 0.03])


def _centroids(poses: np.ndarray) -> np.ndarray:
    return 2.0 * poses[:, :3, :3] @ C + poses[:, :3, 3]


def test_unflagged_moves_only_dead_frames_out_of_band_or_off_mask() -> None:
    z = 0.8
    on_mask = (60.0, 45.0)
    points = [_ray_point(*on_mask, z * math.exp(0.5)),     # 0 alive, out of band        -> untouched
              _ray_point(*on_mask, z * math.exp(0.1)),     # 1 dead, in band, on mask     -> untouched
              _ray_point(*on_mask, z * math.exp(0.5)),     # 2 dead, far                  -> band edge, own ray
              _ray_point(80.0, 45.0, z),                   # 3 dead, in band, 5 r off mask -> mask ray, same depth
              _ray_point(*on_mask, z * math.exp(0.5)),     # 4 dead, no observed depth    -> untouched
              _ray_point(*on_mask, z * math.exp(-0.5)),    # 5 nvis == vis_min: alive     -> untouched
              _ray_point(*on_mask, z * math.exp(0.19))]    # 6 dead, just inside the band -> untouched
    poses = _poses(points)
    ob = _obs([z, z, z, z, math.nan, z, z], [10, 0, 4, 0, 0, 5, 0])

    new, moved, shifts = reanchor_poses(poses, 2.0, C, ob, False, G)

    assert moved == [2, 3]
    p = _centroids(new)
    np.testing.assert_allclose(p[2], _ray_point(*on_mask, z * math.exp(0.2)), atol=1e-12)
    np.testing.assert_allclose(p[3], _ray_point(*on_mask, z), atol=1e-12)
    untouched = [0, 1, 4, 5, 6]
    assert np.array_equal(new[untouched], poses[untouched])
    assert np.array_equal(new[:, :3, :3], poses[:, :3, :3]) and np.array_equal(new[:, 3], poses[:, 3])
    np.testing.assert_allclose(shifts, p[[2, 3]] - _centroids(poses)[[2, 3]], atol=1e-15)


def test_unflagged_near_frame_slides_to_the_lower_band_edge_on_its_own_ray() -> None:
    z = 0.6
    poses = _poses([_ray_point(55.0, 42.0, z * math.exp(-0.7))])     # q = 5.4 / 4 > 1: also moved onto the mask ray
    new, moved, _ = reanchor_poses(poses, 2.0, C, _obs([z], [0]), False, G)
    assert moved == [0]
    np.testing.assert_allclose(_centroids(new)[0], _ray_point(60.0, 45.0, z * math.exp(-0.2)), atol=1e-12)
    inside = _poses([_ray_point(61.0, 45.0, z * math.exp(-0.7))])    # q = 0.25: only the depth moves
    p0 = _centroids(inside)[0]
    p1 = _centroids(reanchor_poses(inside, 2.0, C, _obs([z], [0]), False, G)[0])[0]
    np.testing.assert_allclose(p1, p0 * (z * math.exp(-0.2) / p0[2]), atol=1e-12)


def test_flagged_places_every_out_frame_at_its_observation_on_the_mask_ray() -> None:
    z = 0.7
    on_mask = (60.0, 45.0)
    points = [_ray_point(*on_mask, z * math.exp(0.5)),     # alive, out of band -> z_obs on the mask ray
              _ray_point(61.0, 45.0, z * math.exp(0.1)),   # in band, on mask   -> untouched
              _ray_point(80.0, 45.0, z * math.exp(0.1)),   # in band, off mask  -> z_obs (not its own depth)
              _ray_point(*on_mask, -0.3)]                  # behind the camera  -> untouched
    poses = _poses(points)

    new, moved, _ = reanchor_poses(poses, 2.0, C, _obs([z] * 4, [40] * 4), True, G)

    assert moved == [0, 2]
    for t in moved:
        np.testing.assert_allclose(_centroids(new)[t], _ray_point(*on_mask, z), atol=1e-12)
    assert np.array_equal(new[[1, 3]], poses[[1, 3]])


# --- observations and the clip flag ----------------------------------------------------------------------------------


def _depth_obs(z_obs: list[float], hw: tuple[int, int] = (6, 8)) -> TrajectoryObservations:
    t, (h, w) = len(z_obs), hw
    masks = torch.zeros(t, h, w, dtype=torch.bool)
    masks[:, 1:5, 2:6] = True                                        # 16 pixels, centroid (3.5, 2.5)
    masks[-1] = False
    pm = torch.zeros(t, h, w, 3)
    pm[..., 2] = torch.tensor(z_obs)[:, None, None]
    return TrajectoryObservations(intrinsics=torch.eye(3), masks=masks, hand_masks=torch.zeros_like(masks),
                                  pointmaps=pm, tracks=torch.zeros(t, 1, 2), track_vis=torch.zeros(t, 1, dtype=torch.bool),
                                  anchors=torch.zeros(1, 3))


def test_frame_observations_read_masks_pointmap_and_raw_visibility() -> None:
    ob = frame_observations(_depth_obs([0.8, 0.7, 0.9]), np.array([3, 0, 7]), torch.from_numpy(K).float())
    np.testing.assert_allclose(ob.z_obs[:2], [0.8, 0.7], rtol=1e-6)
    assert math.isnan(ob.z_obs[2]) and np.isnan(ob.mask_centroid[2]).all() and ob.r_mask[2] == 0.0
    np.testing.assert_array_equal(ob.mask_centroid[:2], [[3.5, 2.5], [3.5, 2.5]])
    np.testing.assert_allclose(ob.r_mask[:2], math.sqrt(16 / math.pi))
    assert ob.nvis.tolist() == [3, 0, 7] and ob.K.dtype == np.float64
    with pytest.raises(ValueError, match="visible-track counts"):
        frame_observations(_depth_obs([0.8, 0.7]), np.array([1, 2, 3]), K)


def _clip_solution(z_sol: list[float]) -> tuple[MeshSamples, TrajectorySolution]:
    points = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.1, 0.0, 0.2], [0.1, 0.2, 0.0]])
    poses = torch.eye(4).repeat(len(z_sol), 1, 1)
    poses[:, 0, 3] = 3.5 - 0.1          # centroid (0.1, 0.1, 0.1): lands on the mask centroid with K = I, z = 1
    poses[:, 1, 3] = 2.5 - 0.1
    poses[:, 2, 3] = torch.tensor(z_sol) - 0.1
    return MeshSamples(points=points, normals=points.clone()), fake_solution(poses, 1.0)


def test_flag_comes_from_the_gate_record_when_the_gate_ran() -> None:
    obs = _depth_obs([1.0, 1.0, 1.0, 1.0, 0.0])
    mesh, solution = _clip_solution([1.0, 1.0, 1.0, 1.0, 1.0])          # certificate 0: in band everywhere
    ctx = {TRACKER_VIS_KEY: np.array([9, 9, 9, 9, 9]), "K": torch.eye(3)}

    kept, rec = apply_per_frame_reanchor(mesh, obs, solution, ctx, G)
    assert kept is solution and (rec["flagged"], rec["certificate_source"], rec["frames"]) == (False, "solution", [])
    assert rec["certificate"] == pytest.approx(0.0, abs=1e-6) and rec["shifts_m"] == [] and rec["max_shift_m"] == 0.0

    gated = {**ctx, GATE_KEY: {"certificate": {"value": 0.3, "median_d": -0.3}, "flagged": True}}
    kept, rec = apply_per_frame_reanchor(mesh, obs, solution, gated, G)
    assert (rec["flagged"], rec["certificate_source"], rec["certificate"]) == (True, "depth_anchor_gate", 0.3)
    assert kept is solution                              # flagged, but every frame is in band and on its mask
    unobserved = {**ctx, GATE_KEY: {"certificate": {"value": None}}}
    assert apply_per_frame_reanchor(mesh, obs, solution, unobserved, G)[1]["flagged"] is False


def test_moved_solution_keeps_dtype_and_every_other_field() -> None:
    obs = _depth_obs([1.0, 1.0, 1.0, 1.0, 0.0])
    mesh, solution = _clip_solution([1.0, 1.0, 1.5, 1.0, 1.0])
    ctx = {TRACKER_VIS_KEY: np.array([9, 9, 0, 9, 9]), "K": torch.eye(3)}
    kept, rec = apply_per_frame_reanchor(mesh, obs, solution, ctx, G)
    assert rec["frames"] == [2] and rec["dead_frames"] == [2] and rec["flagged"] is False
    assert kept.poses.dtype == solution.poses.dtype and kept.scale == solution.scale
    assert kept.poses[2, 2, 3].item() == pytest.approx(math.exp(0.2) - 0.1, abs=1e-6)
    assert torch.equal(kept.covariance, solution.covariance) and torch.equal(kept.confidence, solution.confidence)
    assert torch.equal(kept.poses[[0, 1, 3, 4]], solution.poses[[0, 1, 3, 4]])
    aniso = dataclasses.replace(solution, shape_log_stretch=torch.zeros(3), shape_axes=torch.eye(3),
                                shape_centroid=torch.zeros(3))
    with pytest.raises(ValueError, match="isotropic"):
        apply_per_frame_reanchor(mesh, obs, aniso, ctx, G)


# --- the config block ------------------------------------------------------------------------------------------------


def test_config_block_defaults_off_and_validates() -> None:
    root = OmegaConf.load(_ROOT_CONFIG)
    # the production config ships V2 (G + occluded_area_frac 0.25 + max_log_shift 0.3); the dataclass stays off.
    v2 = dataclasses.replace(G, occluded_area_frac=0.25, max_log_shift=0.3)
    assert SolveStageConfig.from_mapping(root.solve_stage).per_frame_reanchor == v2
    assert SolveStageConfig().per_frame_reanchor is None
    root.solve_stage.per_frame_reanchor = {"threshold": 0.25, "band": 0.2, "perp_k": 1, "vis_min": 5}
    cfg = SolveStageConfig.from_mapping(root.solve_stage)
    assert cfg.per_frame_reanchor == G and isinstance(cfg.per_frame_reanchor.perp_k, float)
    assert cfg.depth_anchor_gate == GATE        # independent of the gate
    for bad, match in (({"threshold": 0.25, "band": 0.2, "perp_k": 1.0}, "missing"),
                       ({**dataclasses.asdict(G), "x": 1}, "unknown"),
                       ({**dataclasses.asdict(G), "band": 0.0}, "band"),
                       ({**dataclasses.asdict(G), "threshold": math.inf}, "threshold"),
                       ({**dataclasses.asdict(G), "vis_min": 0}, "vis_min"),
                       ({**dataclasses.asdict(G), "vis_min": 2.5}, "vis_min"), ([1], "null or")):
        with pytest.raises(ValueError, match=match):
            SolveStageConfig.from_mapping({"per_frame_reanchor": bad})


# --- the engine solve paths ------------------------------------------------------------------------------------------


def _clip_dir(root: Path, clip: SyntheticClip, name: str = "c", dead: tuple[int, ...] = ()) -> Path:
    d = root / name
    write_synthetic_solve_inputs(d, clip)
    if dead:
        tracks = dict(load_npz(d / "tracks.npz"))
        vis = tracks["vis"].copy()
        vis[list(dead)] = False
        save_npz(d / "tracks.npz", **{**tracks, "vis": vis})
    return d


def _dead(clip_dir: Path) -> list[int]:
    """The frames with < 5 visible tracks in the clip's tracks.npz (the synthetic clip has occluded tracks)."""
    return np.flatnonzero(load_npz(clip_dir / "tracks.npz")["vis"].sum(1) < G.vis_min).tolist()


def _npz_identical(a: Path, b: Path, skip: tuple[str, ...] = ()) -> bool:
    x, y = load_npz(a), load_npz(b)
    return x.keys() == y.keys() and all(x[k].dtype == y[k].dtype and x[k].shape == y[k].shape
                                        and x[k].tobytes() == y[k].tobytes() for k in x if k not in skip)


class _Solve:
    """The real solver; `factor` scales the returned translations of the frames in `frames` (all when None)."""

    def __init__(self, factor: float = 1.0, frames: tuple[int, ...] | None = None) -> None:
        self.factor, self.frames = factor, frames
        self.returned: list[TrajectorySolution] = []

    def __call__(self, mesh: MeshSamples, obs: TrajectoryObservations, init: Tensor, scale: float,
                 cfg: SolverConfig) -> TrajectorySolution:
        solution = solve_trajectory(mesh, obs, init, scale, cfg)
        if self.factor != 1.0:
            poses = solution.poses.clone()
            rows = slice(None) if self.frames is None else list(self.frames)
            poses[rows, :3, 3] *= self.factor
            solution = dataclasses.replace(solution, poses=poses)
        self.returned.append(solution)
        return solution


def test_default_off_and_nothing_to_move_write_the_production_solve(tmp_path: Path, small_clip: SyntheticClip
                                                                    ) -> None:
    off = _clip_dir(tmp_path / "off", small_clip)
    on = tmp_path / "on" / "c"
    shutil.copytree(off, on)
    assert TRACKER_VIS_KEY not in prepare_solve("c", off, FAST).context       # default path: nothing new in the call

    make_solve_worker(FAST, solve_fn=_Solve())("c", off)
    make_solve_worker(dataclasses.replace(FAST, per_frame_reanchor=G), solve_fn=_Solve())("c", on)

    assert _npz_identical(off / "solve.npz", on / "solve.npz", skip=("seconds",))
    meta_off, meta_on = (json.loads((d / "solve.json").read_text()) for d in (off, on))
    assert REANCHOR_KEY not in meta_off
    block = meta_on.pop(REANCHOR_KEY)
    for meta in (meta_off, meta_on):
        meta["solver"].pop("seconds")
    assert meta_on == meta_off
    assert block["flagged"] is False and block["frames"] == [] and block["dead_frames"] == _dead(off)
    assert (block["threshold"], block["band"], block["perp_k"], block["vis_min"], block["min_points"]) == (
        0.25, 0.2, 1.0, 5, 10)


def _z_obs(clip_dir: Path, cfg: SolveStageConfig) -> np.ndarray:
    obs = prepare_solve("c", clip_dir, cfg).obs
    points, counts = valid_pixels_of(obs)
    return frame_depths(points, torch.cat([counts.new_zeros(1), counts.cumsum(0)])).numpy()


def _written_centroid_z(clip_dir: Path) -> np.ndarray:
    poses = load_npz(clip_dir / "solve.npz")["poses"].astype(np.float64)
    c = load_npz(clip_dir / "samples.npz")["points"].astype(np.float64).mean(0)
    scale = float(load_npz(clip_dir / "solve.npz")["scale"])
    return (scale * poses[:, :3, :3] @ c + poses[:, :3, 3])[:, 2]


def test_unflagged_clip_re_anchors_only_its_dead_frames(tmp_path: Path, small_clip: SyntheticClip) -> None:
    d = _clip_dir(tmp_path, small_clip, dead=(3, 4))
    solve = _Solve(factor=1.5, frames=(3, 4))                   # the dead frames drift 50 % along the ray
    cfg = dataclasses.replace(FAST, per_frame_reanchor=G)

    make_solve_worker(cfg, solve_fn=solve)("c", d)

    block = json.loads((d / "solve.json").read_text())[REANCHOR_KEY]
    assert block["flagged"] is False and block["certificate_source"] == "solution"
    dead = _dead(d)
    assert {3, 4} <= set(dead) and block["dead_frames"] == dead
    assert {3, 4} <= set(block["frames"]) <= set(dead) and len(block["shifts_m"]) == len(block["frames"])
    written = load_npz(d / "solve.npz")["poses"]
    produced = solve.returned[0].poses.numpy()
    alive = [t for t in range(6) if t not in dead]
    assert alive and np.array_equal(written[alive], produced[alive])
    assert np.array_equal(written[:, :3, :3], produced[:, :3, :3])
    np.testing.assert_allclose(_written_centroid_z(d)[[3, 4]], _z_obs(d, cfg)[[3, 4]] * math.exp(0.2), rtol=1e-5)


def test_flagged_clip_re_anchors_every_out_frame(tmp_path: Path, small_clip: SyntheticClip) -> None:
    d = _clip_dir(tmp_path, small_clip)
    cfg = dataclasses.replace(FAST, per_frame_reanchor=G)

    make_solve_worker(cfg, solve_fn=_Solve(factor=1.5))("c", d)   # every frame 50 % far: certificate ~0.4

    block = json.loads((d / "solve.json").read_text())[REANCHOR_KEY]
    assert block["flagged"] is True and block["certificate"] >= 0.25
    assert block["frames"] == list(range(6)) and block["dead_frames"] == _dead(d)
    np.testing.assert_allclose(_written_centroid_z(d), _z_obs(d, cfg), rtol=1e-5)


class _FakeMany:
    """`solve_many` at each clip's init; clip `offset`'s production translations scaled by 1.5 (not the re-solve's)."""

    def __init__(self, offset: int | None = None) -> None:
        self.offset = offset

    def __call__(self, meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
                 init_scales: list[float], config: SolverConfig) -> list[TrajectorySolution]:
        out = [fake_solution(p, s) for p, s in zip(init_poses, init_scales, strict=True)]
        if config.lam_depth_anchor == 0 and self.offset is not None:
            poses = out[self.offset].poses.clone()
            poses[:, :3, 3] *= 1.5
            out[self.offset] = dataclasses.replace(out[self.offset], poses=poses)
        return out


def _stage(cfg: SolveStageConfig, fake: _FakeMany) -> SolveBatchStage:
    return SolveBatchStage(functools.partial(prepare_solve, cfg=cfg), functools.partial(finish_solve, cfg=cfg),
                           cfg.solver_config(), fake, depth_anchor_gate=cfg.depth_anchor_gate)


def test_batched_path_runs_g_after_the_gate(tmp_path: Path, small_clip: SyntheticClip) -> None:
    first = _clip_dir(tmp_path / "off", small_clip, "c0")
    shutil.copytree(first, tmp_path / "off" / "c1")
    shutil.copytree(tmp_path / "off", tmp_path / "on")
    off = [(f"c{i}", tmp_path / "off" / f"c{i}") for i in range(2)]
    on = [(f"c{i}", tmp_path / "on" / f"c{i}") for i in range(2)]
    gated = dataclasses.replace(SMALL_SOLVE, depth_anchor_gate=GATE)

    assert _stage(gated, _FakeMany(offset=1)).run_batch(off, "b-off").errors == {}
    assert _stage(dataclasses.replace(gated, per_frame_reanchor=G), _FakeMany(offset=1)).run_batch(on, "b-on").errors == {}

    blocks = [json.loads((d / "solve.json").read_text()) for _, d in on]
    assert [b[REANCHOR_KEY]["flagged"] for b in blocks] == [False, True]
    assert all(b[REANCHOR_KEY]["certificate_source"] == "depth_anchor_gate" for b in blocks)
    assert blocks[1][REANCHOR_KEY]["certificate"] == blocks[1][GATE_KEY]["certificate"]["value"]
    assert _npz_identical(off[0][1] / "solve.npz", on[0][1] / "solve.npz")   # unflagged, no dead frame: unchanged


def test_a_call_prepared_without_g_is_refused(tmp_path: Path, small_clip: SyntheticClip) -> None:
    d = _clip_dir(tmp_path, small_clip)
    call = prepare_solve("c", d, SMALL_SOLVE)
    with pytest.raises(KeyError, match=TRACKER_VIS_KEY):
        finish_solve("c", d, call, fake_solution(call.init_poses, call.init_scale),
                     dataclasses.replace(SMALL_SOLVE, per_frame_reanchor=G))
    assert not (d / "solve.npz").exists()
