"""Pure-function tests for the DexYCB oracle-observation builder and the batch failure isolation (no real
dataset needed)."""

from __future__ import annotations

import contextlib
import inspect
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

import run.pipeline.eval_solver_dexycb as dexycb_script
from fastdo.bench import Profiler, evaluate_sequence
from fastdo.bench.dexycb_report import GT_MESH_CAVEAT, PROTOCOL_KEYS, summary_lines
from fastdo.io.clip_record import ClipRecord
from fastdo.solver import (
    GpuBudgetError,
    KeyframeRegistration,
    SolverConfig,
    TrajectoryObservations,
    TrajectorySolution,
)
from fastdo.solver.memory import BatchPlan
from run.pipeline.eval_solver_dexycb import (
    _BASELINE,
    _BASELINE_LABEL,
    PreparedSequence,
    SequenceRow,
    _grasped_object_id,
    _real_init_poses,
    _sequence_row,
    _solve_batch,
    _stage_seconds,
    build_oracle_observations,
    run_isolated,
)

_PROTOCOL_FIELDS = ("f5_icp_frame", "f10_icp_frame", "chamfer_icp_frame_cm2",
                    "f5_icp_clip", "f10_icp_clip", "chamfer_icp_clip_cm2")


def test_protocol_keys_are_the_row_fields() -> None:
    assert PROTOCOL_KEYS == _PROTOCOL_FIELDS
    assert set(PROTOCOL_KEYS) <= set(SequenceRow.__dataclass_fields__)


def test_grasped_object_id_selects_the_grasp_index() -> None:
    meta = {"ycb_ids": [1, 7, 13, 21], "ycb_grasp_ind": 2}
    assert _grasped_object_id(meta) == 13


def _scene() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """1 frame, 8x8 image, object origin at (0,0,1m) directly in front of the camera."""
    h, w = 8, 8
    K = torch.tensor([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4).unsqueeze(0)
    pose[0, 2, 3] = 1.0
    masks = torch.zeros(1, h, w, dtype=torch.bool)
    hand_masks = torch.zeros(1, h, w, dtype=torch.bool)
    pointmaps = torch.full((1, h, w, 3), float("nan"))
    return K, pose, masks, hand_masks, pointmaps


def test_pointmap_backprojection_matches_and_visible_anchor_is_flagged() -> None:
    K, pose, masks, hand_masks, pointmaps = _scene()
    masks[0, 4, 4] = True
    pointmaps[0, 4, 4] = torch.tensor([0.0, 0.0, 1.0])   # exact GT depth at the anchor's own pixel
    anchors = torch.zeros(1, 3)                          # the object origin itself
    obs = build_oracle_observations(masks, hand_masks, pointmaps, pose, K, anchors)
    assert torch.allclose(obs.tracks[0, 0], torch.tensor([4.0, 4.0]), atol=1e-5)
    assert bool(obs.track_vis[0, 0])
    assert obs.masks is masks and obs.pointmaps is pointmaps   # passed through unchanged (NaNs preclude ==)


def test_occluded_anchor_is_flagged_invisible() -> None:
    K, pose, masks, hand_masks, pointmaps = _scene()
    masks[0, 4, 4] = True
    pointmaps[0, 4, 4] = torch.tensor([0.0, 0.0, 0.4])   # something 60cm nearer occludes the anchor
    anchors = torch.zeros(1, 3)
    obs = build_oracle_observations(masks, hand_masks, pointmaps, pose, K, anchors)
    assert not bool(obs.track_vis[0, 0])


def test_anchor_outside_mask_is_flagged_invisible() -> None:
    K, pose, masks, hand_masks, pointmaps = _scene()
    pointmaps[0, 4, 4] = torch.tensor([0.0, 0.0, 1.0])   # depth matches exactly...
    # ...but masks[0, 4, 4] is left False: the pixel is not attributed to the tracked object.
    anchors = torch.zeros(1, 3)
    obs = build_oracle_observations(masks, hand_masks, pointmaps, pose, K, anchors)
    assert not bool(obs.track_vis[0, 0])


def test_failed_batch_is_retried_one_sequence_at_a_time() -> None:
    """A batched solve that raises does not take its batch-mates down; the bad sequence is
    skipped, as BASE skipped it, and a clean batch is solved once."""
    calls: list[list[str]] = []

    def run(items: list[str]) -> list[str]:
        calls.append(list(items))
        if "bad" in items:
            raise ValueError("bad sequence")
        return [f"row:{item}" for item in items]

    assert run_isolated(["a", "bad", "c"], run, str) == ["row:a", "row:c"]
    assert calls == [["a", "bad", "c"], ["a"], ["bad"], ["c"]]
    calls.clear()
    assert run_isolated(["a", "c"], run, str) == ["row:a", "row:c"] and calls == [["a", "c"]]


def test_budget_errors_are_skipped_and_other_errors_propagate() -> None:
    def over_budget(items: list[str]) -> list[str]:
        raise GpuBudgetError("does not fit")

    def bug(items: list[str]) -> list[str]:
        raise RuntimeError("solver bug")

    assert run_isolated(["a", "b"], over_budget, str) == []
    with pytest.raises(RuntimeError, match="solver bug"):
        run_isolated(["a", "b"], bug, str)


def test_rows_record_the_planned_sub_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """`batch_clips` is the sub-batch the solver planned (`TrajectorySolution.plan`), not the number of
    sequences passed in, so three sequences that the CUDA budget or a busy card split into 2 + 1 are not reported as
    one batch; `card_limited` marks a split by the card. Merge with main: scoring is `_sequence_row`, which gets
    `protocol_metrics`, and each clip's initial scale comes from its prepared sequence (real mode's registration)."""
    pair, single = BatchPlan((0, 1), 1, 1, card_bytes=1, card_limited=True), BatchPlan((2,), 1, 1, card_bytes=1)
    sols = [SimpleNamespace(plan=pair), SimpleNamespace(plan=pair), SimpleNamespace(plan=single)]
    scored: list[tuple[str, int, bool, bool]] = []
    solve_args: list[tuple[Any, ...]] = []

    def score(seq: str, record: Any, gt_vertices: Any, gt_faces: Any, sol: Any, *, protocol_metrics: bool,
              batch_clips: int, card_limited: bool) -> str:
        scored.append((seq, batch_clips, card_limited, protocol_metrics))
        return seq

    def solve(*args: Any) -> list[Any]:
        solve_args.append(args)
        return sols

    monkeypatch.setattr(dexycb_script, "solve_trajectories", solve)
    monkeypatch.setattr(dexycb_script, "_sequence_row", score)
    batch = [SimpleNamespace(sequence=s, record=None, gt_vertices=None, gt_faces=None, mesh=None, obs=None,
                             init_poses=torch.zeros(4, 4, 4), init_scale=float(k + 1)) for k, s in enumerate("abc")]
    profiler = SimpleNamespace(stage=lambda *args, **kwargs: contextlib.nullcontext())
    rows = _solve_batch(cast("list[PreparedSequence]", batch), SolverConfig(device="cpu"), "pnp", cast("Any", profiler),
                        protocol_metrics=True)
    assert rows == ["a", "b", "c"]
    assert scored == [("a", 2, True, True), ("b", 2, True, True), ("c", 1, False, True)]
    assert len(solve_args) == 1 and solve_args[0][3] == [1.0, 2.0, 3.0]


def test_real_init_poses_falls_back_to_the_keyframe_registration_when_pnp_solves_nothing() -> None:
    t, j = 3, 8
    obs = TrajectoryObservations(intrinsics=torch.eye(3), masks=torch.zeros(t, 4, 4, dtype=torch.bool),
                                 hand_masks=torch.zeros(t, 4, 4, dtype=torch.bool),
                                 pointmaps=torch.zeros(t, 4, 4, 3), tracks=torch.zeros(t, j, 2),
                                 track_vis=torch.zeros(t, j, dtype=torch.bool), anchors=torch.zeros(j, 3))
    keyframe_pose = torch.eye(4)
    keyframe_pose[2, 3] = 0.7
    reg = KeyframeRegistration(pose=keyframe_pose, scale=1.1, score=0.01)
    K = torch.tensor([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]])
    mesh_points = torch.randn(64, 3, generator=torch.Generator().manual_seed(0)) * 0.05
    poses = _real_init_poses(obs, torch.ones(j, dtype=torch.bool), K, reg, mesh_points, keyframe_index=1)
    assert poses.shape == (t, 4, 4)
    assert torch.allclose(poses, keyframe_pose.expand(t, 4, 4))   # no tracks, no pointmap support anywhere


def test_stage_seconds_groups_front_end_stages_and_the_solve() -> None:
    profiler = Profiler(clip_id="c", system="s", n_frames=1, fps=10.0, sync_cuda=False)
    for seq, base in (("a/b/1", 1.0), ("a/b/2", 3.0)):
        profiler.add(f"{seq}/sam2", base)
        profiler.add(f"{seq}/moge", 2 * base)
        profiler.add(seq, 10 * base)
    assert _stage_seconds(profiler, ["a/b/1", "a/b/2"]) == {"sam2": 2.0, "moge": 4.0, "solver": 20.0}


def _synthetic_solve(t: int = 2) -> tuple[ClipRecord, trimesh.Trimesh, TrajectorySolution]:
    """The synthetic sphere record of tests/bench/test_dexycb_eval.py and a stub solution 4 mm / 1 % off."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    poses = np.tile(np.eye(4), (t, 1, 1))
    poses[:, 2, 3] = 0.6
    record = ClipRecord(clip_id="s", frame_idx=np.arange(t, dtype=np.int64), timestamps=np.arange(t) / 30.0,
                        intrinsics=np.eye(3, dtype=np.float32),
                        cam_to_world=np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)), object_mesh_path="m.obj",
                        object_pose=poses.astype(np.float32), object_scale=1.0,
                        gravity_cam=np.array([0, -1, 0], np.float32), confidence=np.ones(t, np.float32),
                        gate_score=1.0, hand_left=None, hand_right=None)
    pred = torch.as_tensor(poses, dtype=torch.float32).clone()
    pred[:, 0, 3] += 0.004
    sol = TrajectorySolution(poses=pred, scale=1.01, covariance=torch.zeros(t, 6, 6), per_frame_cost=torch.zeros(t),
                             confidence=torch.full((t,), 0.8), iterations=7, seconds=0.25)
    return record, mesh, sol


def test_eval_row_has_protocol_fields() -> None:
    record, mesh, sol = _synthetic_solve()
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    row = _sequence_row("a/b/1", record, v, f, sol, protocol_metrics=True)
    strict = evaluate_sequence(record, v, f, v, f, sol.poses.numpy(), sol.scale, solver_seconds=sol.seconds)
    assert (row.f5, row.f10, row.chamfer_cm2, row.n_frames) == (strict.f5, strict.f10, strict.chamfer_cm2,
                                                                 strict.n_frames)
    for name in _PROTOCOL_FIELDS:
        value = getattr(row, name)
        assert isinstance(value, float) and math.isfinite(value), name
    assert row.chamfer_icp_frame_cm2 is not None and row.chamfer_icp_frame_cm2 < row.chamfer_cm2
    assert row.chamfer_icp_clip_cm2 is not None and row.chamfer_icp_clip_cm2 < row.chamfer_cm2
    assert (row.sequence, row.scale, row.seconds, row.iterations) == ("a/b/1", 1.01, 0.25, 7)
    assert math.isclose(row.mean_confidence, 0.8, rel_tol=1e-6)
    as_json = json.loads(json.dumps(asdict(row)))
    assert set(_PROTOCOL_FIELDS) <= set(as_json)
    assert row.shape_source == as_json["shape_source"] == "gt_mesh"   # the eval scores the GT mesh as the shape


def test_eval_row_emits_per_frame_confidence_next_to_translation_error() -> None:
    """Each row carries the solver's per-frame confidence and the per-frame translation error |t - t_gt|
    (the stub is 4 mm off along x), null where the ground-truth pose is missing (record confidence 0)."""
    record, mesh, sol = _synthetic_solve(t=3)
    record = ClipRecord(**{**record.__dict__, "confidence": np.array([1.0, 0.0, 1.0], np.float32)})
    sol = TrajectorySolution(**{**sol.__dict__, "confidence": torch.tensor([0.9, 0.0, 0.25])})
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    row = _sequence_row("a/b/1", record, v, f, sol)
    assert row.frame_confidence == pytest.approx((0.9, 0.0, 0.25))
    assert row.frame_translation_error_m[1] is None
    assert row.frame_translation_error_m[0] == pytest.approx(0.004, abs=1e-6)
    assert row.frame_translation_error_m[2] == pytest.approx(0.004, abs=1e-6)
    as_json = json.loads(json.dumps(asdict(row), allow_nan=False))
    assert as_json["frame_translation_error_m"][1] is None and len(as_json["frame_confidence"]) == 3


def test_protocol_metrics_flag_is_keyword_only_and_off_by_default() -> None:
    for fn in (_solve_batch, _sequence_row):
        param = inspect.signature(fn).parameters["protocol_metrics"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is False, fn.__name__


def test_eval_row_skips_protocol_metrics_when_disabled() -> None:
    record, mesh, sol = _synthetic_solve()
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    row = _sequence_row("a/b/1", record, v, f, sol, protocol_metrics=False)
    assert _sequence_row("a/b/1", record, v, f, sol) == row             # the default is the flag-off row
    assert all(getattr(row, name) is None for name in _PROTOCOL_FIELDS)
    assert row.f5 == evaluate_sequence(record, v, f, v, f, sol.poses.numpy(), sol.scale).f5
    assert json.loads(json.dumps(asdict(row)))["f5_icp_frame"] is None


def _box_solve() -> tuple[ClipRecord, trimesh.Trimesh, TrajectorySolution]:
    """A box posed differently in 3 frames and a stub solution whose frame k is 20 mm off along camera axis k.

    The offsets are independent per frame: each per-frame scaled ICP removes its frame's offset, while the one
    similarity of the per-clip ICP cannot remove all three, so the two protocols score far apart.
    """
    t = 3
    mesh = trimesh.creation.box(extents=[0.16, 0.21, 0.06])
    poses = np.tile(np.eye(4), (t, 1, 1))
    for k in range(t):
        poses[k, :3, :3] = Rotation.from_euler("xyz", (6.0 * k, -5.0 * k, 4.0 * k),
                                               degrees=True).as_matrix()
        poses[k, :3, 3] = (0.01 * k, -0.01 * k, 0.5)
    record = ClipRecord(clip_id="s", frame_idx=np.arange(t, dtype=np.int64), timestamps=np.arange(t) / 30.0,
                        intrinsics=np.eye(3, dtype=np.float32),
                        cam_to_world=np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)), object_mesh_path="m.obj",
                        object_pose=poses.astype(np.float32), object_scale=1.0,
                        gravity_cam=np.array([0, -1, 0], np.float32), confidence=np.ones(t, np.float32),
                        gate_score=1.0, hand_left=None, hand_right=None)
    pred = torch.as_tensor(record.object_pose).clone()
    pred[:, :3, 3] += 0.02 * torch.eye(3)
    sol = TrajectorySolution(poses=pred, scale=1.0, covariance=torch.zeros(t, 6, 6), per_frame_cost=torch.zeros(t),
                             confidence=torch.full((t,), 0.8), iterations=7, seconds=0.25)
    return record, mesh, sol


def test_eval_row_takes_frame_and_clip_fields_from_their_own_protocol() -> None:
    record, mesh, sol = _box_solve()
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    row = _sequence_row("a/b/1", record, v, f, sol, protocol_metrics=True)
    # Measured: icp_frame F-5 1.0, F-10 1.0, CD ~4e-27 cm^2; icp_clip F-5 0.247, F-10 0.672, CD 1.89 cm^2.
    assert row.f5_icp_frame == 1.0 and row.f10_icp_frame == 1.0
    assert row.chamfer_icp_frame_cm2 is not None and row.chamfer_icp_frame_cm2 < 1e-6
    assert row.f5_icp_clip is not None and row.f5_icp_clip < 0.9
    assert row.f10_icp_clip is not None and row.f10_icp_clip < 0.9
    assert row.chamfer_icp_clip_cm2 is not None and row.chamfer_icp_clip_cm2 > 0.5


_SUMMARY_MEANS = {"f5": 0.2, "f10": 0.4, "chamfer_cm2": 9.0, "seconds": 1.5, "f5_icp_frame": 0.6,
                  "f10_icp_frame": 0.8, "chamfer_icp_frame_cm2": 1.0, "f5_icp_clip": 0.3, "f10_icp_clip": 0.5,
                  "chamfer_icp_clip_cm2": 4.0}


def test_summary_flags_gt_mesh_shapes_under_the_per_frame_icp_block() -> None:
    caveat = ("caveat: predicted shape = GT mesh; scaled ICP per frame forgives almost all pose error, so this "
              "block is NOT comparable with DO AS I DO Table 2 (reconstructed shapes); compare tracking with the "
              "per-clip block")
    lines = summary_lines(_SUMMARY_MEANS, 2, 0, _BASELINE, _BASELINE_LABEL, shape_sources=["gt_mesh", "gt_mesh"])
    assert GT_MESH_CAVEAT == caveat and lines.count(caveat) == 1
    frame = lines.index("scaled ICP per frame (Table 2 protocol)")
    assert lines[frame + 2].startswith("mean")          # block title, header, mean row ...
    assert lines[frame + 3] == caveat                   # ... then the caveat directly under the block ...
    assert _BASELINE_LABEL in lines[frame + 4]          # ... and only then the baseline row
    for sources in (["gt_mesh", "reconstructed"], ["reconstructed", "reconstructed"]):
        other = summary_lines(_SUMMARY_MEANS, 2, 0, _BASELINE, _BASELINE_LABEL, shape_sources=sources)
        assert other == [line for line in lines if line != caveat], sources


def test_summary_prints_baseline_only_under_the_per_frame_icp_block() -> None:
    means = _SUMMARY_MEANS
    assert _BASELINE == {"f5": 0.71, "f10": 0.93, "chamfer_cm2": 0.66}
    lines = summary_lines(means, 3, 1, _BASELINE, _BASELINE_LABEL, shape_sources=["reconstructed"] * 3)

    def at(label: str) -> int:
        hits = [i for i, line in enumerate(lines) if label in line]
        assert len(hits) == 1, (label, lines)
        return hits[0]

    strict, frame = at("strict (camera frame, no alignment)"), at("scaled ICP per frame (Table 2 protocol)")
    baseline, clip = at("DO AS I DO Table 2 (scaled-ICP protocol, their shapes)"), at("scaled ICP per clip")
    assert strict < frame < baseline < clip
    assert "0.710" in lines[baseline] and "0.930" in lines[baseline] and "0.660" in lines[baseline]
    assert "0.600" in "\n".join(lines[frame:baseline]) and "0.300" in "\n".join(lines[clip:])

    strict_only = summary_lines({k: means[k] for k in ("f5", "f10", "chamfer_cm2", "seconds")}, 3, 0, _BASELINE,
                                _BASELINE_LABEL, shape_sources=["gt_mesh"] * 3)
    text = "\n".join(strict_only)
    assert "strict (camera frame, no alignment)" in text
    assert "scaled ICP" not in text and "0.710" not in text and GT_MESH_CAVEAT not in text


def test_config_enables_protocol_metrics_by_default() -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml")
    assert cfg.dexycb.protocol_metrics is True


def test_stage_seconds_splits_a_batched_solve_over_its_sequences() -> None:
    """Batched stage seconds: a batched solve's stage is named by its sequences joined with "+" and counts once
    per sequence at its wall time divided by the number of sequences, like the rows' `seconds`."""
    profiler = Profiler(clip_id="c", system="s", n_frames=1, fps=10.0, sync_cuda=False)
    profiler.add("a/b/1/sam2", 1.0)
    profiler.add("a/b/2/sam2", 3.0)
    profiler.add("a/b/1+a/b/2", 30.0)
    profiler.add("a/b/3", 9.0)
    assert _stage_seconds(profiler, ["a/b/1", "a/b/2", "a/b/3"]) == {"sam2": 2.0, "solver": 13.0}


def test_failures_are_recorded_for_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batched solve vs `failed_sequences`: a sequence that still fails alone after its batch is retried
    (a `LinAlgError` included), or whose scoring fails, is recorded once; its batch-mates are not."""
    def run(items: list[str]) -> list[str]:
        if "bad" in items:
            raise torch.linalg.LinAlgError("singular block")
        return [f"row:{item}" for item in items]

    failed: list[dict[str, str]] = []
    assert run_isolated(["a", "bad", "c"], run, str, failed) == ["row:a", "row:c"]
    assert [f["sequence"] for f in failed] == ["bad"] and "singular block" in failed[0]["error"]

    def score(seq: str, *args: Any, **kwargs: Any) -> str:
        if seq == "b":
            raise KeyError("no metric")
        return seq

    monkeypatch.setattr(dexycb_script, "solve_trajectories", lambda *args: [SimpleNamespace(plan=None)] * 2)
    monkeypatch.setattr(dexycb_script, "_sequence_row", score)
    batch = [SimpleNamespace(sequence=s, record=None, gt_vertices=None, gt_faces=None, mesh=None, obs=None,
                             init_poses=torch.zeros(4, 4, 4), init_scale=1.0) for s in "ab"]
    profiler = SimpleNamespace(stage=lambda *args, **kwargs: contextlib.nullcontext())
    failed.clear()
    rows = _solve_batch(cast("list[PreparedSequence]", batch), SolverConfig(device="cpu"), "pnp", cast("Any", profiler),
                        failed=failed)
    assert rows == ["a"] and failed == [{"sequence": "b", "error": "KeyError: 'no metric'"}]
