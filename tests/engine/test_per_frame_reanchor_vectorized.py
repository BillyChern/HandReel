"""`solve_stage.per_frame_reanchor_impl: vectorized` computes G's per-frame observations with the SAME
bytes as the reference loop (`fastdo.engine.per_frame_reanchor`, module docstring IMPLEMENTATION).

Equivalence is byte-for-byte (dtype, shape and `tobytes()` of every `FrameObservations` field; solve.npz and solve.json
without their wall-clock `seconds`), on randomized masks and object pixels (empty, single-pixel, full and border masks;
NaN / inf / non-positive depths; frames below `DEPTH_MIN_POINTS`; odd and even counts; ties), on the dense and the
masked (`object_pixels`) paths, on non-boolean masks, at native Ego4D resolution, and through both engine solve
paths. The comparator itself is shown to see a one-ulp change, so an equivalence pass is not vacuous.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from fastdo.engine.batched_stages import SolveBatchStage
from fastdo.engine.depth_anchor_gate import GATE_KEY
from fastdo.engine.per_frame_reanchor import (
    REANCHOR_IMPLS,
    REANCHOR_KEY,
    TRACKER_VIS_KEY,
    FrameObservations,
    apply_per_frame_reanchor,
    frame_observations,
    frame_observations_vectorized,
    mask_moments,
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
from fastdo.solver.batch import DEPTH_MIN_POINTS, ObjectPixels, object_pixels_of
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    SMALL_SOLVE,
    fake_solution,
    write_synthetic_solve_inputs,
)

V2 = PerFrameReanchorConfig(threshold=0.25, band=0.2, perp_k=1.0, vis_min=5, occluded_area_frac=0.25,
                            max_log_shift=0.3)
G = PerFrameReanchorConfig(threshold=0.25, band=0.2, perp_k=1.0, vis_min=5)
GATE = DepthAnchorGateConfig(threshold=0.25, lam=500.0, band=0.2)
FAST = dataclasses.replace(SMALL_SOLVE, solver={"iters_coarse": 3, "iters_mid": 2, "iters_fine": 3})
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
FIELDS = ("z_obs", "mask_centroid", "r_mask", "nvis", "K")


def _same(a: FrameObservations, b: FrameObservations) -> list[str]:
    """The fields of `a` and `b` whose dtype, shape or bytes differ (empty = byte-identical)."""
    out = []
    for name in FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            out.append(name)
    return out


# --- randomized observations -----------------------------------------------------------------------------------------


def _random_masks(rng: np.random.Generator, t: int, h: int, w: int) -> np.ndarray:
    masks = np.zeros((t, h, w), dtype=bool)
    for k in range(t):
        kind = rng.integers(0, 7)
        if kind == 0:
            continue                                                    # empty frame
        if kind == 1:
            masks[k, rng.integers(0, h), rng.integers(0, w)] = True     # single pixel
        elif kind == 2:
            masks[k] = True                                             # full frame
        elif kind == 3:                                                 # touching the borders
            masks[k, : rng.integers(1, h + 1), : rng.integers(1, 4)] = True
            masks[k, -1, -rng.integers(1, w + 1):] = True
        elif kind == 4:
            masks[k] = rng.random((h, w)) < rng.uniform(0.01, 0.9)      # speckle
        else:                                                           # a noisy blob
            y0, x0 = rng.integers(0, h), rng.integers(0, w)
            y1, x1 = rng.integers(y0 + 1, h + 1), rng.integers(x0 + 1, w + 1)
            masks[k, y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0)) < rng.uniform(0.3, 1.0)
    return masks


def _random_points(rng: np.random.Generator, n: int, dtype: type = np.float32) -> np.ndarray:
    """Depths with ties, NaN / +-inf in any coordinate, zero and negative z."""
    z = rng.choice([rng.uniform(0.2, 2.0), 0.5], size=n) if rng.random() < 0.3 else rng.uniform(-0.2, 2.0, size=n)
    pts = np.stack([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n), z], axis=1).astype(dtype)
    bad = rng.random(n) < rng.uniform(0.0, 0.3)
    pts[bad, rng.integers(0, 3, int(bad.sum()))] = rng.choice([np.nan, np.inf, -np.inf], int(bad.sum()))
    pts[rng.random(n) < 0.02, 2] = 0.0
    return pts


def _observations(rng: np.random.Generator, t: int, h: int, w: int, masked: bool,
                  mask_dtype: torch.dtype = torch.bool) -> TrajectoryObservations:
    masks = _random_masks(rng, t, h, w)
    # a few frames with fewer than DEPTH_MIN_POINTS valid points
    dense = _random_points(rng, t * h * w).reshape(t, h, w, 3)
    m = torch.from_numpy(masks)
    pointmaps = torch.from_numpy(dense)
    px = object_pixels_of(m, pointmaps) if masked else None
    stored = m if mask_dtype == torch.bool else m.to(mask_dtype) * (2 if mask_dtype != torch.float32 else 0.7)
    nan_maps = torch.full((1, h, w, 3), float("nan")).expand(t, h, w, 3) if masked else pointmaps
    return TrajectoryObservations(intrinsics=torch.from_numpy(K).float(), masks=stored,
                                  hand_masks=torch.zeros(t, h, w, dtype=torch.bool), pointmaps=nan_maps,
                                  tracks=torch.zeros(t, 4, 2), track_vis=torch.ones(t, 4, dtype=torch.bool),
                                  anchors=torch.zeros(4, 3), object_pixels=px)


@pytest.mark.parametrize("seed", range(40))
def test_vectorized_observations_are_byte_identical_on_random_clips(seed: int) -> None:
    rng = np.random.default_rng(seed)
    t, h, w = int(rng.integers(1, 12)), int(rng.integers(1, 40)), int(rng.integers(1, 50))
    obs = _observations(rng, t, h, w, masked=bool(seed % 2))
    nvis = rng.integers(0, 20, size=t)
    Kt = torch.from_numpy(K) if seed % 3 else K
    assert _same(frame_observations(obs, nvis, Kt), frame_observations_vectorized(obs, nvis, Kt)) == []


@pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.float32, torch.int64])
def test_non_boolean_masks_are_read_as_nonzero_like_the_reference(mask_dtype: torch.dtype) -> None:
    rng = np.random.default_rng(7)
    obs = _observations(rng, 6, 17, 23, masked=False, mask_dtype=mask_dtype)
    nvis = rng.integers(0, 20, size=6)
    assert _same(frame_observations(obs, nvis, K), frame_observations_vectorized(obs, nvis, K)) == []


def test_min_points_boundary_and_even_odd_medians() -> None:
    """Frames with exactly DEPTH_MIN_POINTS - 1, DEPTH_MIN_POINTS and +1 valid points, and tied middle values."""
    t, h, w = 4, 4, 8
    masks = torch.zeros(t, h, w, dtype=torch.bool)
    counts = [DEPTH_MIN_POINTS - 1, DEPTH_MIN_POINTS, DEPTH_MIN_POINTS + 1, 32]
    for k, c in enumerate(counts):
        masks[k].view(-1)[:c] = True
    rows = []
    for k, c in enumerate(counts):
        z = np.linspace(0.3, 1.7, c).astype(np.float32)
        if k == 3:
            z[:] = 0.8                                       # every value tied
        rows.append(np.stack([np.zeros(c), np.zeros(c), z], 1).astype(np.float32))
    pts = torch.from_numpy(np.concatenate(rows))
    offsets = torch.tensor(np.concatenate([[0], np.cumsum(counts)]), dtype=torch.int64)
    obs = TrajectoryObservations(torch.from_numpy(K).float(), masks, torch.zeros_like(masks),
                                 torch.full((1, h, w, 3), float("nan")).expand(t, h, w, 3), torch.zeros(t, 1, 2),
                                 torch.ones(t, 1, dtype=torch.bool), torch.zeros(1, 3), ObjectPixels(pts, offsets))
    a, b = frame_observations(obs, np.full(t, 9), K), frame_observations_vectorized(obs, np.full(t, 9), K)
    assert _same(a, b) == []
    assert np.isnan(b.z_obs[0]) and np.isfinite(b.z_obs[1:]).all() and b.z_obs[3] == np.float64(np.float32(0.8))


def test_native_ego4d_resolution_index_sums_stay_exact() -> None:
    """2560 x 1920 frames (the largest native Ego4D windows), full and near-full masks: the column / row index sums
    reach ~9.4e9, far above uint16/uint32 but exact in int64 and in float64 (< 2**53)."""
    t, h, w = 3, 1920, 2560
    masks = np.zeros((t, h, w), dtype=bool)
    masks[0] = True
    masks[1, 5:, 7:] = True
    masks[2, 1919, 2559] = True
    count, su, sv = mask_moments(masks)
    for k in range(t):
        vv, uu = np.nonzero(masks[k])
        assert (int(count[k]), int(su[k]), int(sv[k])) == (vv.size, int(uu.sum()), int(vv.sum()))
    obs = TrajectoryObservations(torch.from_numpy(K).float(), torch.from_numpy(masks), torch.zeros(1),
                                 torch.full((1, h, w, 3), float("nan")).expand(t, h, w, 3), torch.zeros(t, 1, 2),
                                 torch.ones(t, 1, dtype=torch.bool), torch.zeros(1, 3),
                                 object_pixels_of(torch.from_numpy(masks),
                                                  torch.ones(1, h, w, 3).expand(t, h, w, 3)))
    assert _same(frame_observations(obs, np.zeros(t), K), frame_observations_vectorized(obs, np.zeros(t), K)) == []


def test_mask_moments_equal_brute_force_sums() -> None:
    rng = np.random.default_rng(3)
    masks = _random_masks(rng, 30, 23, 31)
    count, su, sv = mask_moments(masks)
    for k in range(30):
        vv, uu = np.nonzero(masks[k])
        assert (int(count[k]), int(su[k]), int(sv[k])) == (vv.size, int(uu.sum()), int(vv.sum()))
    assert count.dtype == su.dtype == sv.dtype == np.int64


def test_the_comparator_sees_a_one_ulp_change() -> None:
    """Mutation check inside the suite: the equivalence comparator is not vacuous."""
    rng = np.random.default_rng(11)
    obs = _observations(rng, 6, 30, 40, masked=True)
    ref = frame_observations(obs, np.full(6, 3), K)
    for name in ("z_obs", "mask_centroid", "r_mask"):
        arr = getattr(ref, name).copy()
        flat = arr.reshape(-1)
        finite = np.flatnonzero(np.isfinite(flat) & (flat != 0))
        assert finite.size, f"{name}: no finite non-zero entry to perturb"
        i = int(finite[0])
        flat[i] = np.nextafter(flat[i], np.inf)
        assert _same(ref, dataclasses.replace(ref, **{name: arr})) == [name]


def test_nvis_length_is_checked_as_the_reference_does() -> None:
    obs = _observations(np.random.default_rng(0), 4, 5, 6, masked=True)
    for fn in (frame_observations, frame_observations_vectorized):
        with pytest.raises(ValueError, match="3 visible-track counts for 4 frames"):
            fn(obs, np.zeros(3), K)


# --- apply_per_frame_reanchor: both impls, same solution and record ---------------------------------------------------


def _strip(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "seconds"}


@pytest.mark.parametrize("cfg", [G, V2], ids=["G", "V2"])
@pytest.mark.parametrize("factor", [1.0, 1.6, 0.6])
@pytest.mark.parametrize("gate", [False, True])
def test_apply_gives_the_same_solution_and_record(cfg: PerFrameReanchorConfig, factor: float, gate: bool) -> None:
    rng = np.random.default_rng(int(factor * 10) + gate)
    t, h, w = 8, 48, 64
    masks = np.zeros((t, h, w), dtype=bool)
    for k in range(t):
        masks[k, 10 + k: 30 + k, 20 + k: 44 - (k % 3)] = rng.random((20, 24 - (k % 3) - k)) < 0.9
    masks[5, :, :] = False
    masks[6, 20:22, 30:31] = True                               # a collapsed (low-visibility) mask
    uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float64)
    z = 0.7 + 0.05 * rng.standard_normal((t, h, w))
    rays = np.concatenate([(uv - K[:2, 2]) / K[0, 0], np.ones((h, w, 1))], -1)
    dense = torch.from_numpy((rays[None] * z[..., None]).astype(np.float32))
    m = torch.from_numpy(masks)
    obs = TrajectoryObservations(torch.from_numpy(K).float(), m, torch.zeros_like(m),
                                 torch.full((1, h, w, 3), float("nan")).expand(t, h, w, 3), torch.zeros(t, 1, 2),
                                 torch.ones(t, 1, dtype=torch.bool), torch.zeros(1, 3), object_pixels_of(m, dense))
    mesh = MeshSamples(points=torch.randn(64, 3, dtype=torch.float64) * 0.02,
                       normals=torch.nn.functional.normalize(torch.randn(64, 3, dtype=torch.float64), dim=1))
    poses = torch.eye(4, dtype=torch.float32).repeat(t, 1, 1)
    poses[:, :3, 3] = torch.tensor([0.0, 0.0, 0.7 * factor]) + 0.01 * torch.randn(t, 3)
    solution = fake_solution(poses, 1.0)
    context: dict = {TRACKER_VIS_KEY: np.array([9, 9, 1, 0, 9, 2, 0, 9]), "K": torch.from_numpy(K).float()}
    if gate:
        context[GATE_KEY] = {"certificate": {"value": 0.4 if factor != 1.0 else 0.1}}
    kept_r, rec_r = apply_per_frame_reanchor(mesh, obs, solution, context, cfg, impl="reference")
    kept_v, rec_v = apply_per_frame_reanchor(mesh, obs, solution, context, cfg, impl="vectorized")
    assert (kept_r is solution) == (kept_v is solution)
    assert kept_r.poses.dtype == kept_v.poses.dtype and kept_r.poses.numpy().tobytes() == kept_v.poses.numpy().tobytes()
    assert _strip(dict(rec_r)) == _strip(dict(rec_v))
    assert json.dumps(_strip(dict(rec_r))) == json.dumps(_strip(dict(rec_v)))


def test_each_impl_runs_its_own_code_and_the_default_is_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two impls are equivalent by design, so output tests cannot tell which ran: spy on the dispatch."""
    import fastdo.engine.per_frame_reanchor as pfr

    obs = _observations(np.random.default_rng(1), 3, 6, 7, masked=True)
    mesh = MeshSamples(points=torch.zeros(4, 3, dtype=torch.float64), normals=torch.zeros(4, 3, dtype=torch.float64))
    sol = fake_solution(torch.eye(4).repeat(3, 1, 1), 1.0)
    ran: list[str] = []
    monkeypatch.setattr(pfr, "frame_observations", lambda *a: (ran.append("reference"), frame_observations(*a))[1])
    monkeypatch.setattr(pfr, "frame_observations_vectorized",
                        lambda *a: (ran.append("vectorized"), frame_observations_vectorized(*a))[1])
    context = {TRACKER_VIS_KEY: np.zeros(3), "K": K}
    pfr.apply_per_frame_reanchor(mesh, obs, sol, context, G)
    pfr.apply_per_frame_reanchor(mesh, obs, sol, context, G, impl="reference")
    pfr.apply_per_frame_reanchor(mesh, obs, sol, context, G, impl="vectorized")
    assert ran == ["reference", "reference", "vectorized"]


def test_an_unknown_impl_is_refused() -> None:
    obs = _observations(np.random.default_rng(0), 2, 4, 4, masked=True)
    mesh = MeshSamples(points=torch.zeros(4, 3, dtype=torch.float64), normals=torch.zeros(4, 3, dtype=torch.float64))
    sol = fake_solution(torch.eye(4).repeat(2, 1, 1), 1.0)
    with pytest.raises(ValueError, match="impl must be one of"):
        apply_per_frame_reanchor(mesh, obs, sol, {TRACKER_VIS_KEY: np.zeros(2), "K": K}, G, impl="fast")


# --- config -------------------------------------------------------------------------------------------------------


def test_config_key_defaults_to_the_reference_and_validates() -> None:
    assert SolveStageConfig().per_frame_reanchor_impl == "reference"
    assert REANCHOR_IMPLS == ("reference", "vectorized")
    cfg = SolveStageConfig.from_mapping({"device": "cpu", "per_frame_reanchor_impl": "vectorized"})
    assert cfg.per_frame_reanchor_impl == "vectorized"
    with pytest.raises(ValueError, match="per_frame_reanchor_impl must be one of"):
        SolveStageConfig(per_frame_reanchor_impl="loop")


# --- both engine solve paths: solve.npz / solve.json identical -----------------------------------------------------


@pytest.fixture(scope="module")
def small_clip() -> SyntheticClip:
    return make_synthetic_clip(SMALL_CLIP, seed=0)


def _npz_bytes(path: Path, skip: tuple[str, ...] = ("seconds",)) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    data = load_npz(path)
    return {k: (str(v.dtype), v.shape, v.tobytes()) for k, v in data.items() if k not in skip}


def _meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    meta["solver"].pop("seconds")
    meta.get(REANCHOR_KEY, {}).pop("seconds", None)
    return meta


class _Solve:
    def __init__(self, factor: float = 1.0, frames: tuple[int, ...] | None = None) -> None:
        self.factor, self.frames = factor, frames

    def __call__(self, mesh: MeshSamples, obs: TrajectoryObservations, init: Tensor, scale: float,
                 cfg: SolverConfig) -> TrajectorySolution:
        solution = solve_trajectory(mesh, obs, init, scale, cfg)
        if self.factor != 1.0:
            poses = solution.poses.clone()
            poses[slice(None) if self.frames is None else list(self.frames), :3, 3] *= self.factor
            solution = dataclasses.replace(solution, poses=poses)
        return solution


def _two_dirs(root: Path, clip: SyntheticClip, dead: tuple[int, ...]) -> tuple[Path, Path]:
    ref = root / "ref" / "c"
    write_synthetic_solve_inputs(ref, clip)
    if dead:
        tracks = dict(load_npz(ref / "tracks.npz"))
        vis = tracks["vis"].copy()
        vis[list(dead)] = False
        save_npz(ref / "tracks.npz", **{**tracks, "vis": vis})
    vec = root / "vec" / "c"
    shutil.copytree(ref, vec)
    return ref, vec


@pytest.mark.parametrize(("factor", "frames", "dead"), [(1.5, (3, 4), (3, 4)), (1.5, None, ()), (1.0, None, ())],
                         ids=["unflagged-dead", "flagged", "nothing-to-move"])
def test_per_clip_worker_writes_identical_artifacts(tmp_path: Path, small_clip: SyntheticClip, factor: float,
                                                    frames: tuple[int, ...] | None, dead: tuple[int, ...],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import fastdo.engine.per_frame_reanchor as pfr

    ran: list[str] = []
    monkeypatch.setattr(pfr, "frame_observations_vectorized",
                        lambda *a: (ran.append("vectorized"), frame_observations_vectorized(*a))[1])
    ref, vec = _two_dirs(tmp_path, small_clip, dead)
    base = dataclasses.replace(FAST, per_frame_reanchor=V2)
    make_solve_worker(base, solve_fn=_Solve(factor, frames))("c", ref)
    assert ran == []
    make_solve_worker(dataclasses.replace(base, per_frame_reanchor_impl="vectorized"),
                      solve_fn=_Solve(factor, frames))("c", vec)
    assert ran == ["vectorized"]                                  # the config key reaches G through finish_solve
    assert _npz_bytes(ref / "solve.npz") == _npz_bytes(vec / "solve.npz")
    assert _meta(ref / "solve.json") == _meta(vec / "solve.json")
    if factor != 1.0:
        assert json.loads((ref / "solve.json").read_text())[REANCHOR_KEY]["frames"]      # something did move


class _FakeMany:
    def __call__(self, meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
                 init_scales: list[float], config: SolverConfig) -> list[TrajectorySolution]:
        out = [fake_solution(p, s) for p, s in zip(init_poses, init_scales, strict=True)]
        if config.lam_depth_anchor == 0:
            poses = out[0].poses.clone()
            poses[:, :3, 3] *= 1.5
            out[0] = dataclasses.replace(out[0], poses=poses)
        return out


def test_batched_path_with_the_gate_writes_identical_artifacts(tmp_path: Path, small_clip: SyntheticClip) -> None:
    ref, vec = _two_dirs(tmp_path, small_clip, (2,))
    for d, impl in ((ref, "reference"), (vec, "vectorized")):
        cfg = dataclasses.replace(FAST, per_frame_reanchor=V2, depth_anchor_gate=GATE, per_frame_reanchor_impl=impl)
        stage = SolveBatchStage(functools.partial(prepare_solve, cfg=cfg), functools.partial(finish_solve, cfg=cfg),
                                cfg.solver_config(), _FakeMany(), depth_anchor_gate=cfg.depth_anchor_gate)
        assert stage.run_batch([("c", d)], "solve-00000").errors == {}
    assert _npz_bytes(ref / "solve.npz") == _npz_bytes(vec / "solve.npz")
    assert _meta(ref / "solve.json") == _meta(vec / "solve.json")
