"""Tests for the placement search (`fastdo.robot.reach`, `fastdo.robot.reach_eval`) and the task span
(`fastdo.robot.kinematic_rows.task_span`)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.robot.kinematic_rows import task_span, wrist_targets_world
from fastdo.robot.reach import (
    PlacementChoice,
    ReachSearchConfig,
    ReachTask,
    approach_retreat,
    choice_targets,
    physical_check,
    score_outcome,
    search_placement,
    table_depths,
    table_violation,
)
from fastdo.robot.reach_eval import IkJob, IkOutcome, MarginConfig, ik_margins, row_margins, run_job
from tests.robot import synthetic

TX, TY = (0.18, 0.78), (-0.60, 0.40)
_C2W = np.eye(4)
_C2W[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]       # the synthetic clip's camera (looks along +x)


def _outcome(pos, rot, jl=None, sig=None, clr=None) -> IkOutcome:
    n = len(pos)
    return IkOutcome(key=0, q=np.zeros((n, 7)), pos_err=np.asarray(pos, float), rot_err=np.asarray(rot, float),
                     clearance=np.full(n, np.nan) if clr is None else np.asarray(clr, float),
                     jl_margin=np.full(n, 1.0) if jl is None else np.asarray(jl, float),
                     sigma_min=np.full(n, 1.0) if sig is None else np.asarray(sig, float))


def test_row_margin_known_values() -> None:
    cfg = MarginConfig()
    m = row_margins(np.array([0.0, 0.005, 0.03, 0.0, 0.0]), np.array([0.0, 0.05, 0.0, 0.25, 0.0]),
                    np.array([0.5, 0.1, 0.5, 0.5, 0.5]), np.array([0.1, 0.017, 0.1, 0.1, 0.1]),
                    np.array([np.nan, 0.004, np.nan, np.nan, -0.002]), cfg)
    assert m[0] == 1.0                                  # every margin saturated
    assert m[1] == pytest.approx(0.4)                   # min(0.1/0.2, 0.017/0.034, 0.004/0.01) = 0.4
    assert m[2] == pytest.approx(-3.0)                  # 3 cm out of reach
    assert m[3] == pytest.approx(-2.5)                  # 0.25 rad out of orientation
    assert m[4] == pytest.approx(-1.0)                  # 2 mm penetration: infeasible, floor -1


def test_one_infeasible_row_ranks_below_any_feasible_path() -> None:
    """Minimax objective: a path perfect except one row 1.5 cm out of reach loses to a path whose every row is
    barely feasible; between two infeasible paths the smaller worst violation wins. Mutation (mean instead of min)
    would rank the first path best."""
    cfg = MarginConfig()
    spike = score_outcome(_outcome([0.0] * 19 + [0.015], [0.0] * 20), cfg)
    meh = score_outcome(_outcome([0.009] * 20, [0.09] * 20, jl=[0.02] * 20), cfg)
    worse = score_outcome(_outcome([0.0] * 19 + [0.05], [0.0] * 20), cfg)
    assert meh["score"] >= 0.0 > spike["score"] > worse["score"]
    assert spike["infeasible_rows"] == 1 and spike["mean_margin"] > meh["mean_margin"]


def test_ik_margins_at_known_poses(g1) -> None:
    q = np.zeros((2, 7))
    q[1, 3] = -1.0472 + 0.05                            # elbow 0.05 rad above its lower limit
    jl, sig = ik_margins(g1, q, 1.0 / np.sqrt(50.0))
    assert jl[1] == pytest.approx(0.05, abs=1e-6) and jl[0] == pytest.approx(1.0472, abs=1e-4)
    assert sig[0] == pytest.approx(0.0605, abs=5e-4)    # q = 0, measured once (reach_eval docstring scale)


def test_task_span_keeps_the_contact_span() -> None:
    t = np.arange(50) * 0.1
    used = np.ones(50, bool)
    used[:3] = False
    dist = np.full(50, 0.2)
    dist[20:31] = 0.005
    assert task_span(dist, used, t, 0.015, 1.0, 1.0, 1.0, True) == (10, 40, True)
    assert task_span(dist, used, t, 0.015, 1.0, 1.0, 1.0, False) == (0, 49, True)
    assert task_span(np.full(50, 0.2), used, t, 0.015, 1.0, 1.0, 1.0, True) == (0, 49, False)
    dist2 = dist.copy()
    dist2[5:8] = 0.0                                     # touched 0.2 s after the first used row: not resting
    lo, hi, rest = task_span(dist2, used, t, 0.015, 1.0, 1.0, 1.0, True)
    assert (lo, hi, rest) == (3, 40, False)


def _task(g1, n: int = synthetic.N_ROWS, rest: bool = False) -> ReachTask:
    joints = synthetic.hand_joints(g1, synthetic.arm_path(n))
    obj = np.tile(np.eye(4), (n, 1, 1))
    obj[:, :3, 3] = [synthetic.OBJECT_XY[0], synthetic.OBJECT_XY[1], synthetic.TABLE_Z + 0.031]
    box = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    return ReachTask(wrist_world=wrist_targets_world(joints), object_world=obj, joints_world=joints, anchor=0,
                     rows=np.ones(n, bool), times=np.arange(n) * 0.1, hand_q=np.tile(synthetic.HAND_Q, (n, 1)),
                     mesh_vertices=box, scale=0.06, cam_to_world0=_C2W, up_record=np.array([0.0, 0.0, 1.0]),
                     rest_check=rest)


def test_choice_targets_move_the_task_rigidly(g1) -> None:
    task = _task(g1)
    base = PlacementChoice(table_xy=synthetic.OBJECT_XY, table_z=synthetic.TABLE_Z)
    _, w0, o0 = choice_targets(task, base)
    assert np.allclose(w0[:, :3, 3], task.wrist_world[:, :3, 3], atol=2e-3)        # identity up to the clearance
    pl2, w1, o1 = choice_targets(task, replace(base, yaw_deg=90.0, table_xy=(0.30, -0.20), table_z=0.75))
    assert np.allclose(pl2.object_start[:2, 3], (0.30, -0.20)) and pl2.table_z == 0.75
    rel0 = np.linalg.solve(o0, w0)
    rel1 = np.linalg.solve(o1, w1)
    assert np.allclose(rel0, rel1, atol=1e-9)                                     # hand-object relation kept
    d0 = w0[:, :3, 3] - o0[:, :3, 3]
    d1 = w1[:, :3, 3] - o1[:, :3, 3]
    assert np.allclose(d1[:, 0], -d0[:, 1], atol=1e-9) and np.allclose(d1[:, 1], d0[:, 0], atol=1e-9)


def test_table_depths_flag_a_hand_under_the_table_and_a_sunk_object(g1) -> None:
    """The hand path lowered 25 cm goes through the table top: every row over the table is a table violation (depth
    / 1 cm), and the score of a perfect-reach path drops below 0. The object sunk 5 cm on rows 5-9 counts only where
    its confidence is at least 0.1. Only an overhanging footprint is a hard refusal."""
    cfg = ReachSearchConfig()
    task = _task(g1)
    low = replace(task, joints_world=task.joints_world - [0.0, 0.0, 0.25],
                  wrist_world=task.wrist_world - np.array([[0, 0, 0, 0]] * 2 + [[0, 0, 0, 0.25], [0, 0, 0, 0]]))
    ch = PlacementChoice(table_xy=synthetic.OBJECT_XY, table_z=synthetic.TABLE_Z)
    pl, w, o = choice_targets(low, ch)
    chk = physical_check(low, pl, w, o, TX, TY, cfg)
    assert chk["ok"] and chk["hand_below_table_m"] > 0.05 and chk["table_bad_rows"] > 0
    tv = table_violation(low, pl, w, o, TX, TY, cfg)
    assert (tv[tv > 0] > 2.0).all() and (tv > 0).sum() == chk["table_bad_rows"]
    perfect = _outcome([0.0] * synthetic.N_ROWS, [0.0] * synthetic.N_ROWS)
    assert score_outcome(perfect, MarginConfig())["score"] == 1.0
    assert score_outcome(perfect, MarginConfig(), tv)["score"] == pytest.approx(-tv.max())

    obj = task.object_world.copy()
    obj[5:10, 2, 3] -= 0.05
    conf = np.ones(synthetic.N_ROWS)
    conf[7:10] = 0.05
    sunk = replace(task, object_world=obj, object_conf=conf)
    pl, w, o = choice_targets(sunk, ch)
    _, obj_d = table_depths(sunk, pl, w, o, TX, TY, cfg)
    assert (obj_d[5:7] > 0.04).all() and (obj_d[7:10] == 0).all() and (obj_d[:5] == 0).all()
    pl, w, o = choice_targets(task, PlacementChoice(table_xy=(0.19, 0.0), table_z=synthetic.TABLE_Z))
    assert not physical_check(task, pl, w, o, TX, TY, cfg)["ok"]


def test_approach_and_retreat_replace_only_the_contactless_rows() -> None:
    """Known answer: rows before the first contact descend vertically onto it from `height` (smoothstep), rows after
    the last contact rise from it; contact rows are untouched, orientation is held."""
    n = 11
    wrist = np.tile(np.eye(4), (n, 1, 1))
    wrist[:, :3, 3] = np.arange(n)[:, None] * [0.01, 0.02, -0.01]
    wrist[:, :3, :3] = Rotation.from_rotvec(np.arange(n)[:, None] * [0.0, 0.0, 0.1]).as_matrix()
    out = approach_retreat(wrist, np.arange(n) * 0.1, (3, 7), 0.08)
    assert np.allclose(out[3:8], wrist[3:8])
    assert np.allclose(out[0, :3, 3], wrist[3, :3, 3] + [0, 0, 0.08]) and np.allclose(out[0, :3, :3], wrist[3, :3, :3])
    assert np.allclose(out[10, :3, 3], wrist[7, :3, 3] + [0, 0, 0.08]) and np.allclose(out[9, :3, :3], wrist[7, :3, :3])
    assert np.allclose(out[:3, :2, 3], wrist[3, :2, 3]) and (np.diff(out[:4, 2, 3]) < 0).all()
    assert out[8, 2, 3] - wrist[7, 2, 3] == pytest.approx(0.08 * (1 / 9) * (3 - 2 / 3))


def test_run_job_reproduces_a_reachable_path(g1) -> None:
    """Known answer: the synthetic wrist path is the FK of a known arm path, so the IK lands on it (both with and
    without the table collider) with full reach margin on every row."""
    task = _task(g1)
    _, w, _ = choice_targets(task, PlacementChoice(table_xy=synthetic.OBJECT_XY, table_z=synthetic.TABLE_Z))
    for table_z in (None, synthetic.TABLE_Z):
        out = run_job(IkJob(key=0, targets=w, times=task.times, hand_q=task.hand_q, table_z=table_z))
        s = score_outcome(out, MarginConfig())
        assert s["infeasible_rows"] == 0 and s["pos_err_max_m"] < 3e-3, s


def test_search_finds_a_feasible_placement_and_records_every_candidate(g1) -> None:
    """A small search on the synthetic task: the chosen placement is fully feasible on every row (final stage with the
    collider), it is physically consistent, and every evaluated candidate is recorded with its stage."""
    task = _task(g1)
    _, w, _ = choice_targets(task, PlacementChoice(table_xy=synthetic.OBJECT_XY, table_z=synthetic.TABLE_Z))
    centre = tuple(float(v) for v in w[:, :3, 3].mean(axis=0))      # yaw 0 at this point = the generating placement
    cfg = ReachSearchConfig(pitch_deg=(None,), yaw_deg=(-60.0, 0.0, 60.0), anchor_points=(centre,),
                            coarse_stride=4, top_coarse=2, top_final=1, max_rounds=1, step_xy=0.02, min_step_xy=0.02)
    res = search_placement(task, cfg, TX, TY)
    assert res.final["infeasible_rows"] == 0 and res.score >= 0.0 and res.final["physical"]["ok"]
    assert res.final["rows"] == synthetic.N_ROWS
    stages = {r["stage"] for r in res.records}
    assert stages == {"coarse", "local", "final"}
    assert sum(r["stage"] == "coarse" for r in res.records) == 3
    assert all("score" in r and "physical" in r for r in res.records)
    assert cfg.table_z_range[0] <= res.choice.table_z <= cfg.table_z_range[1]


def test_search_refuses_when_nothing_fits_on_the_table(g1, tmp_path: Path) -> None:
    task = _task(g1)
    with pytest.raises(ValueError, match="footprint on the table"):
        search_placement(task, ReachSearchConfig(pitch_deg=(None,), yaw_deg=(0.0,)), (0.18, 0.20), TY)


def test_leximin_breaks_a_shared_worst_row() -> None:
    """Two placements share the same worst row (an object row below the table at every placement); leximin prefers
    the one whose SECOND worst row is better. Mutation (rank on the minimum only) keeps the input order."""
    from fastdo.robot.reach import _rank
    cfg = MarginConfig()
    tv = np.zeros(10)
    tv[0] = 5.0
    a = score_outcome(_outcome([0.0] + [0.03] + [0.0] * 8, [0.0] * 10), cfg, tv)
    b = score_outcome(_outcome([0.0] + [0.012] + [0.0] * 8, [0.0] * 10), cfg, tv)
    assert a["score"] == b["score"] == -5.0
    ranked = _rank([dict(a, name="a"), dict(b, name="b")])
    assert [r["name"] for r in ranked] == ["b", "a"]


def test_hand_below_the_table_plane_but_off_the_table_is_allowed(g1) -> None:
    """The robot table is a rectangle: a hand below its plane but in front of its near edge collides with nothing."""
    task = _task(g1)
    low = replace(task, joints_world=task.joints_world - [0.0, 0.0, 0.25],
                  wrist_world=task.wrist_world - np.array([[0, 0, 0, 0]] * 2 + [[0, 0, 0, 0.25], [0, 0, 0, 0]]))
    pl, w, o = choice_targets(low, PlacementChoice(table_xy=synthetic.OBJECT_XY, table_z=synthetic.TABLE_Z))
    hand_d, _ = table_depths(low, pl, w, o, (0.60, 0.78), TY, ReachSearchConfig())      # table starts at x = 0.60
    assert (hand_d == 0.0).all()


def test_local_search_walks_to_the_best_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a stand-in objective (residual = distance of the wrist centroid from a goal point), the coarse stage starts
    8 cm away and the pattern search must walk to within its last step (1 cm) of the goal. Every job of the coarse
    and local stages carries the table height (collider on); with `collider=False` only the final stage does."""
    import fastdo.robot.reach as R
    goal = np.array([0.33, -0.27])
    seen: list[tuple[str, float | None]] = []

    def fake(jobs, pool=None):
        out = []
        for j in jobs:
            d = float(np.linalg.norm(j.targets[:, :2, 3].mean(axis=0) - goal))
            seen.append((len(j.targets), j.table_z))
            n = len(j.targets)
            out.append(IkOutcome(key=j.key, q=np.zeros((n, 7)), pos_err=np.full(n, d), rot_err=np.zeros(n),
                                 clearance=np.full(n, np.nan), jl_margin=np.ones(n), sigma_min=np.ones(n)))
        return out

    monkeypatch.setattr(R, "evaluate_jobs", fake)
    n = 12
    task = ReachTask(wrist_world=np.tile(np.eye(4), (n, 1, 1)), object_world=np.tile(np.eye(4), (n, 1, 1)),
                     joints_world=np.zeros((n, 21, 3)), anchor=0, rows=np.ones(n, bool), times=np.arange(n) * 0.1,
                     hand_q=np.zeros((n, 7)), mesh_vertices=np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                                                                     for z in (-.5, .5)]), scale=0.04,
                     cam_to_world0=_C2W, up_record=np.array([0.0, 0.0, 1.0]), rest_check=False)
    cfg = ReachSearchConfig(pitch_deg=(None,), yaw_deg=(0.0,), anchor_points=((0.25, -0.27, 0.9),), top_coarse=1,
                            top_final=1, step_xy=0.04, min_step_xy=0.01, max_rounds=10)
    res = R.search_placement(task, cfg, TX, TY)
    _, w, _ = choice_targets(task, res.choice)
    assert np.linalg.norm(w[:, :2, 3].mean(axis=0) - goal) <= 0.01 + 1e-9
    assert all(z is not None for _, z in seen)
    seen.clear()
    R.search_placement(task, replace(cfg, collider=False), TX, TY)
    assert all((z is not None) == (rows == n) for rows, z in seen)


def test_moved_table_equals_a_freshly_built_table(g1) -> None:
    """The worker keeps ONE collider model per table rectangle and moves the table to each job's height: the IK and
    the clearances equal those of a model built at that height, whatever height ran before. Mutation (table never
    moved) keeps the first height's table. Many heights leave one cached collider model (memory)."""
    import fastdo.robot.reach_eval as E
    from fastdo.robot.arm_ik import IkCollider, solve_arm_ik
    from fastdo.robot.replay import SceneConfig, build_ik_model
    task = _task(g1)
    _, w, _ = choice_targets(task, PlacementChoice(table_xy=synthetic.OBJECT_XY, table_z=synthetic.TABLE_Z))
    E._WORKER.clear()
    for z in (0.70, 0.93, 0.80, 0.93):
        out = run_job(IkJob(key=0, targets=w, times=task.times, hand_q=task.hand_q, table_z=z))
        ref_g1 = build_ik_model(SceneConfig(table_z=z))
        ref = solve_arm_ik(ref_g1, w, collider=IkCollider(ref_g1), times=task.times, hand_q=task.hand_q)
        assert np.allclose(out.q, ref.q, atol=1e-10) and np.allclose(out.clearance, ref.clearance, atol=1e-10)
    assert out.clearance.min() < 0.01                       # at 0.93 m the table is within d_safe of the arm
    assert len([k for k in E._WORKER if k is not None]) == 1


def _die_once(marker: str) -> int:
    import os
    if not Path(marker).exists():
        Path(marker).write_text("x")
        os._exit(1)
    return 7


def test_retry_pool_survives_a_killed_worker(tmp_path: Path) -> None:
    from fastdo.robot.reach_eval import RetryPool
    marker = str(tmp_path / "died")
    with RetryPool(2) as pool:
        assert pool.map(_die_once, [marker, marker, marker]) == [7, 7, 7]
        assert pool.rebuilds == 1


def _box_task(rot: np.ndarray) -> tuple[ReachTask, object]:
    box = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    obj = np.tile(np.eye(4), (4, 1, 1))
    obj[:, :3, :3] = rot
    task = ReachTask(wrist_world=np.tile(np.eye(4), (4, 1, 1)), object_world=obj, joints_world=np.zeros((4, 21, 3)),
                     anchor=0, rows=np.ones(4, bool), times=np.arange(4) * 0.1, hand_q=np.zeros((4, 7)),
                     mesh_vertices=box, scale=0.06, cam_to_world0=_C2W, up_record=np.array([0.0, 0.0, 1.0]),
                     rest_check=True, hull_vertices=0.06 * box, hull_com=np.zeros(3))
    pl, _, _ = choice_targets(task, PlacementChoice(table_xy=(0.4, -0.1), table_z=0.8))
    return task, pl


def test_rest_check_known_answers() -> None:
    """A 6 cm cube flat on a face: CoM 3 cm inside its support square (stable). Turned 45 deg about x it stands on an
    edge right under its CoM (inside 0: unstable, margin 5 mm). Turned 30 deg it rests on one edge with the CoM
    horizontally 3*sqrt(2)*sin(15 deg) ~ 1.10 cm beyond it (would topple)."""
    from scipy.spatial.transform import Rotation as Rot

    from fastdo.robot.reach import rest_inside, rest_violation
    cfg = ReachSearchConfig()
    task, pl = _box_task(np.eye(3))
    assert rest_inside(task, pl, cfg) == pytest.approx(0.03, abs=1e-9) and rest_violation(task, pl, cfg) == 0.0
    task, pl = _box_task(Rot.from_euler("x", 45, degrees=True).as_matrix())
    assert rest_inside(task, pl, cfg) == pytest.approx(0.0, abs=1e-9)
    assert rest_violation(task, pl, cfg) == pytest.approx(0.5, abs=1e-6)
    task, pl = _box_task(Rot.from_euler("x", 30, degrees=True).as_matrix())
    assert rest_inside(task, pl, cfg) == pytest.approx(-0.03 * np.sqrt(2) * np.sin(np.radians(15)), abs=1e-6)
    chk = physical_check(task, pl, task.wrist_world, task.object_world, TX, TY, cfg)
    assert chk["rest_stable"] is False and chk["ok"]


def test_unstable_start_ranks_below_a_stable_one() -> None:
    cfg = MarginConfig()
    perfect = _outcome([0.0] * 5, [0.0] * 5)
    assert score_outcome(perfect, cfg, None, 0.0)["score"] == 1.0
    assert score_outcome(perfect, cfg, None, 2.5)["score"] == -2.5
    assert score_outcome(perfect, cfg, None, 0.3)["score"] == -1.0


def test_evaluator_scores_an_unstable_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """A perfect-reach candidate whose start pose stands on an edge (cube turned 30 deg) scores the unstable-start
    violation (1.6 cm short of 5 mm inside) instead of a full margin."""
    from scipy.spatial.transform import Rotation as Rot

    import fastdo.robot.reach as R

    def perfect(jobs, pool=None):
        return [_outcome([0.0] * len(j.targets), [0.0] * len(j.targets)) for j in jobs]

    monkeypatch.setattr(R, "evaluate_jobs", perfect)
    task, _ = _box_task(Rot.from_euler("x", 30, degrees=True).as_matrix())
    rec = R._Evaluator(task, ReachSearchConfig(), TX, TY, None).run(
        [PlacementChoice(table_xy=(0.4, -0.1), table_z=0.8)], "coarse")[0]
    expected = (0.005 + 0.03 * np.sqrt(2) * np.sin(np.radians(15))) / 0.01
    assert rec["rest_violation"] == pytest.approx(expected, abs=1e-6) and rec["score"] == pytest.approx(-expected)
    assert rec["infeasible_rows"] == 0 and rec["physical"]["rest_stable"] is False


def test_search_tries_resting_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    """With gravity unestimated, the coarse stage tries every head pitch AND every resting-face up; without
    (`allow_pitch=False`) only the record's up."""
    import fastdo.robot.reach as R
    from fastdo.robot.rest import StableUp

    monkeypatch.setattr(R, "evaluate_jobs", lambda jobs, pool=None: [_outcome([0.0] * len(j.targets),
                                                                              [0.0] * len(j.targets)) for j in jobs])
    task, _ = _box_task(np.eye(3))
    faces = (StableUp(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.5, 0.0),
             StableUp(np.array([-1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 0.2, 15.0))
    task = replace(task, stable_ups=faces)
    cfg = ReachSearchConfig(pitch_deg=(None, 30.0), yaw_deg=(0.0,), anchor_points=((0.3, -0.2, 0.9),), max_rounds=1)
    res = R.search_placement(task, cfg, TX, TY)
    coarse = [r["choice"] for r in res.records if r["stage"] == "coarse"]
    assert {(c["pitch_deg"], c["up_face"]) for c in coarse} == {(None, None), (30.0, None), (None, 0), (None, 1)}
    res = R.search_placement(task, cfg, TX, TY, allow_pitch=False)
    assert {(r["choice"]["pitch_deg"], r["choice"]["up_face"]) for r in res.records if r["stage"] == "coarse"} == {
        (None, None)}


def test_object_depth_reads_every_hull_vertex() -> None:
    """Round-3 MINOR: a 1,000-vertex mesh whose lowest point is vertex 1, 3 cm below every other vertex. The old
    400-vertex strided subsample skipped it (non-conservative): sunk 5 cm, the object reads 5 cm deep (not 2 cm)."""
    import trimesh
    g = np.linspace(-0.5, 0.5, 10)
    pts = np.stack(np.meshgrid(g, g, np.linspace(0.0, 1.0, 10), indexing="ij"), axis=-1).reshape(-1, 3)
    pts[1] = [0.0, 0.0, -0.5]                                   # 0.5 units x 0.06 = 3 cm below the bottom face
    assert 1 not in np.unique(np.linspace(0, len(pts) - 1, 400).astype(int))
    obj = np.tile(np.eye(4), (4, 1, 1))
    obj[2:, 2, 3] -= 0.05
    hull = trimesh.convex.convex_hull(0.06 * pts)
    task = ReachTask(wrist_world=np.tile(np.eye(4), (4, 1, 1)), object_world=obj, joints_world=np.zeros((4, 21, 3)),
                     anchor=0, rows=np.ones(4, bool), times=np.arange(4) * 0.1, hand_q=np.zeros((4, 7)),
                     mesh_vertices=pts, scale=0.06, cam_to_world0=_C2W, up_record=np.array([0.0, 0.0, 1.0]),
                     rest_check=True, hull_vertices=np.asarray(hull.vertices), hull_com=np.asarray(hull.center_mass))
    pl, w, o = choice_targets(task, PlacementChoice(table_xy=(0.4, -0.1), table_z=0.8))
    _, obj_d = table_depths(task, pl, w, o, TX, TY, ReachSearchConfig())
    assert obj_d[:2] == pytest.approx([0.0, 0.0], abs=2e-3)
    assert obj_d[2:] == pytest.approx([0.05, 0.05], abs=2e-3)
