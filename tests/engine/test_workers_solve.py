"""Tests for the solve stage worker (`fastdo.engine.workers_solve`): D2 real mode's order on engine artifacts (CPU)."""

from __future__ import annotations

import dataclasses

import inspect
import json
import logging
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import trimesh
from omegaconf import OmegaConf

from fastdo.engine import workers_solve
from fastdo.engine.geometry_io import pack_masked
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import STAGE_GPU, EngineConfig, gpu_lock, run_clip
from fastdo.engine.qc_calibration import template_frame_poses
from fastdo.engine.worker_support import load_npz, save_npz
from fastdo.engine.workers_qc import read_solve_artifact
from fastdo.engine.workers_shape import read_mesh_samples, read_shape_meta
from fastdo.engine.workers_solve import (
    RegistrationConfig,
    SolveStageConfig,
    final_track_weights,
    finish_solve,
    make_solve_worker,
    prepare_solve,
    registration_init_scale,
)
from fastdo.geometry.intrinsics import read_intrinsics
from fastdo.io.datasets.dexycb_real import assemble_real_observations
from fastdo.shape.mesh_utils import canonical_samples
from fastdo.solver import SolverConfig, TrajectorySolution, scale_from_extent
from fastdo.solver.api import TrajectoryObservations, solve_trajectory
from fastdo.solver.batch import valid_pixels_of
from fastdo.solver.synthetic import SyntheticClip, SyntheticConfig, make_synthetic_clip
from run.pipeline.eval_solver_dexycb import _real_init_poses
from tests.engine.fakes_solve import (
    SMALL_CLIP,
    SMALL_SOLVE,
    RecordingSolve,
    write_synthetic_solve_inputs,
)

_ROOT_CONFIG = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
_SPIED = ("register_keyframe_pose", "anchors_from_pose", "pnp_initialize", "gate_init_poses")


@pytest.fixture(scope="module")
def small_clip() -> SyntheticClip:
    return make_synthetic_clip(SMALL_CLIP, seed=0)


def _clip_dir(root: Path, clip: SyntheticClip, name: str = "c") -> Path:
    write_synthetic_solve_inputs(root / name, clip)
    return root / name


class _Spy:
    """Wraps the four D2 helpers in `workers_solve`'s namespace: records their bound arguments and results."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
        self.args: dict[str, dict[str, Any]] = {}
        self.results: dict[str, Any] = {}
        for name in _SPIED:
            monkeypatch.setattr(workers_solve, name, self._wrap(name, getattr(workers_solve, name), order))

    def _wrap(self, name: str, original: Callable[..., Any], order: list[str]) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            order.append(name)
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            self.args[name] = dict(bound.arguments)
            self.results[name] = original(*args, **kwargs)
            return self.results[name]

        return spy


def _run_solve_stage(root: Path, worker: Callable[[str, Path], None]) -> ClipManifest:
    manifest = ClipManifest.create(["c"], ["synthetic"], [6])
    run_clip("c", EngineConfig(artifact_root=root, stages=("solve",)), {"solve": worker}, manifest)
    return manifest


def test_solve_stage_config_from_the_hydra_block() -> None:
    cfg = SolveStageConfig.from_mapping(OmegaConf.load(_ROOT_CONFIG).solve_stage)

    # production carries the certificate-gated depth anchor, the V2 per-frame re-anchor and its vectorized
    # implementation; everything else is the dataclass default
    assert cfg.depth_anchor_gate is not None and (cfg.depth_anchor_gate.threshold, cfg.depth_anchor_gate.lam,
                                                  cfg.depth_anchor_gate.band) == (0.25, 500.0, 0.2)
    pfr = cfg.per_frame_reanchor
    assert pfr is not None and (pfr.threshold, pfr.band, pfr.perp_k, pfr.vis_min, pfr.occluded_area_frac,
                                pfr.max_log_shift) == (0.25, 0.2, 1.0, 5, 0.25, 0.3)
    assert cfg.per_frame_reanchor_impl == "vectorized" and SolveStageConfig().per_frame_reanchor_impl == "reference"
    assert dataclasses.replace(cfg, depth_anchor_gate=None, per_frame_reanchor=None,
                               per_frame_reanchor_impl="reference") == SolveStageConfig()
    assert cfg.registration == RegistrationConfig()
    assert (cfg.device, cfg.seed, cfg.fix_scale, cfg.registration.scale_source) == ("cuda", 42, False, "auto")
    assert cfg.registration.scale_range == (0.5, 2.0) and dict(cfg.solver) == {}
    assert cfg.solver_config() == SolverConfig(device="cuda", seed=42)


def test_solve_stage_config_rejects_unmerged_and_unknown_options() -> None:
    with pytest.raises(ValueError, match="fix_scale") as info:
        SolveStageConfig(fix_scale=True)
    assert "undecided" in str(info.value) and "solve_stage.fixed_scale" in str(info.value)
    for bad in ({"solver": {"fix_scale": True}}, {"solver": {"device": "cpu"}},
                {"registration": {"scale_source": "icp"}}, {"registration": {"scale_range": [2.0, 0.5]}}):
        with pytest.raises(ValueError):
            SolveStageConfig.from_mapping(bad)
    with pytest.raises(ValueError, match="lam_vel"):
        SolveStageConfig(solver={"lam_wnoa": 1.0})
    with pytest.raises(ValueError):
        SolveStageConfig(registration=RegistrationConfig(scale_range=(2.0, 0.5)))

    cfg = SolveStageConfig(device="cpu", solver={"lam_vel": 0.5, "lam_smooth": 0.05, "iters_fine": 7})

    assert cfg.solver_config() == SolverConfig(device="cpu", seed=42, lam_vel=0.5, lam_smooth=0.05, iters_fine=7)


def test_registration_init_scale_sources(small_clip: SyntheticClip) -> None:
    pointmap, mask, points = small_clip.pointmaps[0], small_clip.masks[0], small_clip.surface_points
    # The declared metres-per-canonical-unit must AGREE with what the keyframe pointmap sees, or the
    # cross-check below refuses. The old fixture used 2.5 (an arbitrary constant, 3.3x from this clip's
    # own extent): it exercised source RESOLUTION only, and nothing made the two rulers interact.
    extent = scale_from_extent(pointmap, mask, points)
    template = {"backend": "template", "normalization_scale": 1.0 / extent}
    sam3d = {"backend": "sam3d", "normalization_scale": 1.0 / extent}

    for source in ("auto", "metric_template"):
        scale, resolved = registration_init_scale(source, template, pointmap, mask, points)
        assert scale == pytest.approx(extent), "a consistent template keeps its own metric scale"
        assert resolved == "metric_template"
    assert registration_init_scale("auto", sam3d, pointmap, mask, points) == (extent, "extent")
    with pytest.raises(ValueError, match="template"):
        registration_init_scale("metric_template", sam3d, pointmap, mask, points)


def test_a_non_metric_template_is_refused_not_silently_scaled(small_clip: SyntheticClip) -> None:
    """`auto` inferred metric-ness from the BACKEND NAME, so a normalized mesh was read as metres.

    DO AS I DO's `template.obj` is normalized to unit AABB diagonal, so `1/normalization_scale` claimed a
    1 m mustard bottle against a true 0.22 m -- a silent 4.5x error, and unrecoverable, because the
    optimiser moves scale by a median 3.1% (the image cannot constrain the scale-depth orbit).

    The guard cross-checks the two INDEPENDENT rulers already available here and refuses on disagreement.
    A plausible-RANGE test would not catch it: a unit-normalized mesh gives exactly 1.0 m, a perfectly
    plausible object size. The DISAGREEMENT carries the signal, not the value.

    Threshold 2.0 is set from real clips, not taste: metric GT YCB templates agree to 1.03-1.37x (P2, n=5)
    and the unit-normalized template disagrees by 4.35-6.16x (the three h2h clips).
    """
    pointmap, mask, points = small_clip.pointmaps[0], small_clip.masks[0], small_clip.surface_points
    extent = scale_from_extent(pointmap, mask, points)
    unit_normalized = {"backend": "template", "normalization_scale": 1.0 / (extent * 5.0)}

    with pytest.raises(ValueError, match="disagree"):
        registration_init_scale("auto", unit_normalized, pointmap, mask, points)

    # and a template that agrees within the factor is still accepted, so the guard is not a blanket ban
    mild = {"backend": "template", "normalization_scale": 1.0 / (extent * 1.5)}
    scale, resolved = registration_init_scale("auto", mild, pointmap, mask, points)
    assert resolved == "metric_template" and scale == pytest.approx(extent * 1.5)


def test_a_contaminated_extent_does_not_refuse_a_metric_template(small_clip: SyntheticClip) -> None:
    """The metric-template guard was TWO-sided, and refused 5 of 71 labelled P2 clips whose templates are metric.

    In all five the template was right and the extent ruler was wrong -- it read hand-held YCB objects as
    1.2 m, 3.6 m and 6.5 m. Depth-edge ("flying") pixels only ever ADD extent, so that pathology is always
    on the HIGH side of observed/declared. A unit-normalized mesh declares ~1 m for a ~0.2 m object, so its
    signature is always on the LOW side. Hand occlusion also lowers the extent, but boundedly: the worst
    metric clip is declared/observed = 2.376, the best unit template 4.353, over n=71 and n=3 (qc18).

    So the guard is one-sided at 3.0. This pins the real failure: an extent 10x too LARGE is accepted.
    """
    pointmap, mask, points = small_clip.pointmaps[0], small_clip.masks[0], small_clip.surface_points
    extent = scale_from_extent(pointmap, mask, points)

    contaminated = {"backend": "template", "normalization_scale": 1.0 / (extent / 10.0)}
    scale, resolved = registration_init_scale("auto", contaminated, pointmap, mask, points)
    assert resolved == "metric_template" and scale == pytest.approx(extent / 10.0), \
        "an over-read extent must not overrule a metric template -- the template wins"

    occluded = {"backend": "template", "normalization_scale": 1.0 / (extent * 2.5)}   # extent under-read 2.5x
    scale, _ = registration_init_scale("auto", occluded, pointmap, mask, points)
    assert scale == pytest.approx(extent * 2.5), "bounded occlusion on the low side is tolerated"

    with pytest.raises(ValueError, match="disagree"):
        registration_init_scale("auto", {"backend": "template", "normalization_scale": 1.0 / (extent * 3.5)},
                                pointmap, mask, points)


def test_final_track_weights_are_the_cauchy_irls_weights() -> None:
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    anchors = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    tracks = torch.tensor([[50.0, 50.0], [54.0, 50.0], [58.0, 50.0]]).expand(2, 3, 2).clone()
    vis = torch.ones(2, 3, dtype=torch.bool)
    vis[1, 2] = False
    solution = TrajectorySolution(poses=torch.eye(4).expand(2, 4, 4).clone(), scale=1.0, covariance=torch.zeros(2, 6, 6),
                                  per_frame_cost=torch.zeros(2), confidence=torch.ones(2), iterations=0, seconds=0.0)

    weight = final_track_weights(anchors, tracks, vis, solution, K, 4.0)

    assert weight.dtype == torch.float32
    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.5, 0.2], [1.0, 0.5, 0.0]]), rtol=0.0, atol=1e-6)


def test_solve_worker_follows_d2_real_mode_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 small_clip: SyntheticClip) -> None:
    clip_dir = _clip_dir(tmp_path, small_clip)
    order: list[str] = []
    spy, solve = _Spy(monkeypatch, order), RecordingSolve(order=order)

    make_solve_worker(SMALL_SOLVE, solve_fn=solve)("c", clip_dir)

    assert order == [*_SPIED, "solve_fn"]
    samples, meta, K = read_mesh_samples(clip_dir), read_shape_meta(clip_dir), torch.from_numpy(read_intrinsics(clip_dir)[0])
    geometry, masks, tracks = (load_npz(clip_dir / f"{name}.npz") for name in ("geometry", "mask", "tracks"))
    pointmap, mask = torch.from_numpy(geometry["pointmaps"][0]), torch.from_numpy(masks["masks"][0])
    reg_args, reg = spy.args["register_keyframe_pose"], spy.results["register_keyframe_pose"]
    assert torch.equal(reg_args["points"], samples.points) and torch.equal(reg_args["normals"], samples.normals)
    assert torch.equal(reg_args["observed"], pointmap[mask & torch.isfinite(pointmap).all(-1)])
    assert reg_args["scale"] == 1.0 / meta["normalization_scale"]
    assert (reg_args["seed"], reg_args["device"], reg_args["estimate_scale"]) == (0, "cpu", False)

    obj = cast("trimesh.Trimesh", trimesh.load(str(clip_dir / "mesh.obj"), process=False, force="mesh"))
    dense_points, dense_normals = canonical_samples(np.asarray(obj.vertices), np.asarray(obj.faces), 4096, seed=2)
    anchor_args = spy.args["anchors_from_pose"]
    assert torch.equal(anchor_args["pose"], reg.pose) and anchor_args["scale"] == reg.scale
    assert torch.equal(anchor_args["points"], dense_points) and torch.equal(anchor_args["normals"], dense_normals)
    assert torch.equal(anchor_args["query_uv"], torch.from_numpy(tracks["tracks"][0]))
    assert anchor_args["image_hw"] == (120, 160) and anchor_args["max_px"] == 2.0

    anchors, valid = spy.results["anchors_from_pose"]
    real = assemble_real_observations(K, torch.from_numpy(masks["masks"]), torch.from_numpy(geometry["pointmaps"]),
                                      torch.from_numpy(tracks["tracks"]), torch.from_numpy(tracks["vis"]),
                                      torch.from_numpy(tracks["query_uv"]), 0)
    pnp_args = spy.args["pnp_initialize"]
    assert torch.equal(pnp_args["vis"], real.anchored(anchors, valid).track_vis)
    assert pnp_args["scale"] == reg.scale and pnp_args["min_inliers"] == 6
    gate_args = spy.args["gate_init_poses"]
    assert torch.equal(gate_args["points"], samples.points) and gate_args["keyframe_index"] == 0
    assert torch.equal(gate_args["keyframe_pose"], reg.pose)
    ((_, _, _, init_scale, solver_cfg),) = solve.calls
    assert init_scale == reg.scale and solver_cfg == SolverConfig(device="cpu", seed=0, iters_fine=3)


@pytest.fixture(scope="module")
def pnp_clip() -> SyntheticClip:
    """SMALL_CLIP seed 1: 11-13 of its 24 tracks are visible per frame, so PnP solves every frame. The order test's
    seed 0 shows only 4-5 per frame, below pnp_initialize's min_inliers 6, so there PnP solves no frame (measured
    2026-09-14, the worker and `_real_init_poses` alike) and only the fallback would be exercised."""
    return make_synthetic_clip(SMALL_CLIP, seed=1)


@pytest.mark.parametrize("case", ["pnp", "pnp_fails"])
def test_solve_worker_init_matches_d2_real_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                caplog: pytest.LogCaptureFixture, pnp_clip: SyntheticClip,
                                                case: str) -> None:
    clip_dir = _clip_dir(tmp_path, pnp_clip)
    if case == "pnp_fails":
        tracks = load_npz(clip_dir / "tracks.npz")
        save_npz(clip_dir / "tracks.npz", **{**tracks, "vis": np.zeros_like(tracks["vis"])})
    spy, solve = _Spy(monkeypatch, []), RecordingSolve()

    with caplog.at_level(logging.WARNING, logger=workers_solve.__name__):
        make_solve_worker(SMALL_SOLVE, solve_fn=solve)("c", clip_dir)

    ((_, obs, init, _, _),) = solve.calls
    reg, (_, valid) = spy.results["register_keyframe_pose"], spy.results["anchors_from_pose"]
    K = torch.from_numpy(read_intrinsics(clip_dir)[0])
    assert torch.equal(init, _real_init_poses(obs, valid, K, reg, read_mesh_samples(clip_dir).points, 0))
    meta = json.loads((clip_dir / "solve.json").read_text())
    warned = [r for r in caplog.records if r.name == workers_solve.__name__
              and "pnp_initialize solved no frame" in r.getMessage() and r.levelno == logging.WARNING]
    assert (meta["pnp_fallback"], len(warned)) == ((True, 1) if case == "pnp_fails" else (False, 0))
    assert (meta["n_pnp_solved"] > 0) is (case == "pnp")
    assert (clip_dir / "solve.npz").is_file() and meta["registration"]["scale_source"] == "metric_template"


def test_solve_worker_recovers_synthetic_trajectory(tmp_path: Path) -> None:
    clip = make_synthetic_clip(SyntheticConfig(n_frames=8, shape="box", depth_noise=0.001), seed=11)
    clip_dir = _clip_dir(tmp_path, clip)
    cfg = SolveStageConfig(device="cpu", seed=0, registration=RegistrationConfig(n_obs=256, n_mesh=512, n_iters=40))

    make_solve_worker(cfg)("c", clip_dir)

    artifact = read_solve_artifact(clip_dir)
    poses = artifact.solution.poses
    assert poses.shape == (8, 4, 4) and bool(torch.isfinite(poses).all())
    assert artifact.frame_idx.tolist() == [0, 3, 6, 9, 12, 15, 18, 21]
    weight = artifact.track_weight
    assert weight.shape == (8, 64) and float(weight.min()) >= 0.0 and float(weight.max()) <= 1.0
    raw, metric_scale = template_frame_poses(poses, artifact.solution.scale, read_shape_meta(clip_dir))
    err = np.linalg.norm(raw[:, :3, 3] - clip.poses_gt.double().numpy()[:, :3, 3], axis=1)
    assert float(err.max()) <= 0.010, f"per-frame translation error (m): {err.tolist()}"
    assert abs(metric_scale - 1.0) <= 0.03, f"metric scale {metric_scale}"


@pytest.mark.parametrize(("artifact", "key"), [("geometry.npz", "frame_idx"), ("tracks.npz", "keyframe_index")])
def test_solve_worker_requires_decode_rows_and_matching_keyframes(tmp_path: Path, small_clip: SyntheticClip,
                                                                  artifact: str, key: str) -> None:
    clip_dir = _clip_dir(tmp_path, small_clip)
    data = load_npz(clip_dir / artifact)
    save_npz(clip_dir / artifact, **{**data, key: data[key] + 1})       # tracks keyframe_index 0 -> 1
    worker = make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve())

    with pytest.raises(ValueError, match=artifact):
        worker("c", clip_dir)
    status = _run_solve_stage(tmp_path, worker).status("c", "solve")

    assert status.state == "failed" and artifact in status.error
    assert not (clip_dir / "solve.npz").exists()


@pytest.mark.parametrize("field", ["poses", "covariance"])
def test_solve_worker_refuses_non_finite_solutions(tmp_path: Path, small_clip: SyntheticClip, field: str) -> None:
    clip_dir = _clip_dir(tmp_path, small_clip)
    (clip_dir / "solve.npz").write_bytes(b"stale")
    (clip_dir / "solve.json").write_text("{}")
    worker = make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve(nan_field=field))

    status = _run_solve_stage(tmp_path, worker).status("c", "solve")

    assert status.state == "failed" and field in status.error and "non-finite" in status.error
    assert not (clip_dir / "solve.npz").exists() and not (clip_dir / "solve.json").exists()
    with pytest.raises(FloatingPointError, match=field):
        worker("c", clip_dir)


def test_solve_stage_holds_the_gpu_lock(tmp_path: Path, small_clip: SyntheticClip) -> None:
    assert STAGE_GPU["solve"] is True
    clip_dir = _clip_dir(tmp_path, small_clip)
    contention, probed = threading.Event(), threading.Event()

    def probe() -> None:
        try:
            with gpu_lock(tmp_path / ".gpu.lock", blocking=False):
                pass
        except BlockingIOError:
            contention.set()
        finally:
            probed.set()

    def probe_from_another_thread() -> None:
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(timeout=5.0)

    manifest = _run_solve_stage(tmp_path, make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve(
        on_call=probe_from_another_thread)))

    assert probed.is_set() and contention.is_set(), "a different thread must see the lock held during the solve"
    assert manifest.status("c", "solve").state == "done" and (clip_dir / "solve.npz").is_file()


def test_prepare_and_finish_compose_to_the_worker(tmp_path: Path, small_clip: SyntheticClip) -> None:
    a = _clip_dir(tmp_path / "a", small_clip)
    b = tmp_path / "b" / "c"
    shutil.copytree(a, b)
    fake = RecordingSolve()

    make_solve_worker(SMALL_SOLVE, solve_fn=fake)("c", a)
    call = prepare_solve("c", b, SMALL_SOLVE)
    solution = fake(call.mesh, call.obs, call.init_poses, call.init_scale, SMALL_SOLVE.solver_config())
    finish_solve("c", b, call, solution, SMALL_SOLVE)

    npz_a, npz_b = load_npz(a / "solve.npz"), load_npz(b / "solve.npz")
    assert npz_a.keys() == npz_b.keys()
    for key in npz_a:
        assert npz_a[key].dtype == npz_b[key].dtype and np.array_equal(npz_a[key], npz_b[key]), key
    assert json.loads((a / "solve.json").read_text()) == json.loads((b / "solve.json").read_text())
    assert call.init_scale == call.context["registration"].scale
    assert np.array_equal(call.context["frame_idx"], 3 * np.arange(6))


def test_solve_worker_logs_phase_timing(tmp_path: Path, small_clip: SyntheticClip,
                                        caplog: pytest.LogCaptureFixture) -> None:
    """One parseable line per clip decomposes the stage; solve.json stays byte-identical (no wall clock)."""
    clip_dir = _clip_dir(tmp_path, small_clip)
    with caplog.at_level(logging.INFO, logger="fastdo.engine.workers_solve"):
        make_solve_worker(SMALL_SOLVE, solve_fn=RecordingSolve())("c", clip_dir)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("solve timing c: ")]
    assert len(lines) == 1
    timing = json.loads(lines[0].removeprefix("solve timing c: "))
    assert set(timing) == {"read_inputs", "registration", "anchors", "pnp", "gate", "solve_fn", "write_npz"}
    assert all(v >= 0 for v in timing.values())
    assert "timing" not in json.loads((clip_dir / "solve.json").read_text())


def _dense_and_masked_clip(root: Path, clip: SyntheticClip, k: int = 2) -> tuple[Path, Path]:
    """One clip written twice, with dense and with masked `geometry.npz`, keyframe `k` (not 0, so a read
    of row 0 in place of the keyframe cannot pass). The dense pointmaps get FINITE off-mask pixels on every frame: the
    masked format keeps them only on the keyframe, so an identical solve shows no stage reads them elsewhere. A tenth
    of the object pixels turn NaN (MoGe-invalid), so the finite filter is exercised too."""
    dense = _clip_dir(root / "dense", clip)
    mask, tracks, geometry = (load_npz(dense / f"{name}.npz") for name in ("mask", "tracks", "geometry"))
    save_npz(dense / "mask.npz", **{**mask, "keyframe_index": np.array(k, dtype=np.int64)})
    save_npz(dense / "tracks.npz", **{**tracks, "keyframe_index": np.array(k, dtype=np.int64),
                                      "query_uv": tracks["tracks"][k].copy()})
    rng = np.random.default_rng(0)
    pm, on = geometry.pop("pointmaps").copy(), mask["masks"].astype(bool)
    pm[~on] = 3.0 + rng.standard_normal((int((~on).sum()), 3)).astype(np.float32)
    pm[on & (rng.random(on.shape) < 0.1)] = np.nan
    save_npz(dense / "geometry.npz", **geometry, pointmaps=pm)
    masked = root / "masked" / "c"
    shutil.copytree(dense, masked)
    save_npz(masked / "geometry.npz", **geometry, **pack_masked(pm, mask["masks"], k))
    return dense, masked


@pytest.mark.parametrize("solver", ["recording", "solve_trajectory"])
def test_masked_geometry_solves_bit_identically_to_dense(tmp_path: Path, small_clip: SyntheticClip,
                                                        solver: str) -> None:
    """The solve stage reads a masked artifact's object points directly (no dense rebuild) and writes the same
    `solve.npz` and `solve.json` as from the dense artifact, bit for bit (wall-clock `seconds` aside)."""
    dense, masked = _dense_and_masked_clip(tmp_path, small_clip)
    recorded: dict[Path, TrajectoryObservations] = {}
    for clip_dir in (dense, masked):
        fake = RecordingSolve()

        def solve_fn(mesh: Any, obs: TrajectoryObservations, init: Any, scale: float, cfg: SolverConfig,
                     fake: RecordingSolve = fake, clip_dir: Path = clip_dir) -> TrajectorySolution:
            recorded[clip_dir] = obs
            return fake(mesh, obs, init, scale, cfg) if solver == "recording" else solve_trajectory(
                mesh, obs, init, scale, cfg)

        make_solve_worker(SMALL_SOLVE, solve_fn=solve_fn)("c", clip_dir)

    obs_dense, obs_masked = recorded[dense], recorded[masked]
    assert obs_dense.object_pixels is None
    assert obs_masked.object_pixels is not None and obs_masked.pointmaps.stride()[0] == 0, "the packed path ran"
    for a, b in zip(valid_pixels_of(obs_dense), valid_pixels_of(obs_masked), strict=True):
        assert a.dtype == b.dtype and torch.equal(a, b)
    npz_a, npz_b = load_npz(dense / "solve.npz"), load_npz(masked / "solve.npz")
    assert npz_a.keys() == npz_b.keys()
    for key in npz_a.keys() - {"seconds"}:
        assert npz_a[key].dtype == npz_b[key].dtype and np.array_equal(npz_a[key], npz_b[key]), key
    json_a, json_b = (json.loads((d / "solve.json").read_text()) for d in (dense, masked))
    for meta in (json_a, json_b):
        meta["solver"].pop("seconds")
    assert json_a == json_b and json_a["keyframe_index"] == 2
