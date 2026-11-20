"""The run script's moves between a home and the trajectory (TO_HOLD, GO_HOME) are planned
collision-free in the package's scene (`fastdo.robot.transfer`) and simulated exactly as the script commands them
(`fastdo.robot.replay.simulate_run`); a straight move through the table fails the execution gate.

Known answer (synthetic scene: table top 0.80 m, a 6 cm box at the synthetic object start, the synthetic arm's first
pose as the hold pose): the straight joint-space line from the `arms_hanging` home to the hold pose sweeps the hand
>= 5 cm into the table (an earlier run script's TO_HOLD); the planned path keeps >= 1.5 cm everywhere."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import trimesh

from fastdo.robot import g1_constants as C
from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.replay import SceneConfig, build_replay_scene, simulate_run
from fastdo.robot.trajectory import RobotTrajectory, TrajectoryLimits, gravity_torque
from fastdo.robot.transfer import (
    HOMES,
    PlannedPath,
    TransferChecker,
    TransferConfig,
    plan_fields,
    plan_go_home,
    plan_to_hold,
)
from tests.robot import synthetic

HANGING = np.asarray(dict(HOMES)["arms_hanging"])


@pytest.fixture(scope="module")
def setup(g1):
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    start = np.eye(4)
    start[:3, 3] = [synthetic.OBJECT_XY[0], synthetic.OBJECT_XY[1], synthetic.TABLE_Z + 0.031]
    scene = build_replay_scene(box.vertices, box.faces, 0.06, start, SceneConfig(table_z=synthetic.TABLE_Z))
    q = synthetic.arm_path()[0]
    at, ht = np.arange(251) / 250.0, np.arange(101) / 100.0
    tau = gravity_torque(g1, q[None], synthetic.HAND_Q[None])[0]
    traj = RobotTrajectory(arm_t=at, arm_q=np.tile(q, (len(at), 1)), arm_dq=np.zeros((len(at), 7)),
                           arm_tau_ff=np.tile(tau, (len(at), 1)), hand_t=ht, hand_q=np.tile(synthetic.HAND_Q, (len(ht), 1)),
                           hand_dq=np.zeros((len(ht), 7)), time_scale=1.0, hold_s=0.0, source_t=np.array([0.0, 1.0]))
    hold = np.concatenate([q, C.LEFT_ARM_PARK, synthetic.HAND_Q])
    return scene, start, traj, hold


def _sweep(chk: TransferChecker, verts: np.ndarray, res: float = 0.002) -> tuple[float, float, float]:
    worst = [np.inf, np.inf, 0.0]
    for a, b in itertools.pairwise(verts):
        for t in np.linspace(0.0, 1.0, max(int(np.ceil(np.abs(b - a).max() / res)), 1) + 1):
            m = chk.measure(a + (b - a) * t)
            worst = [min(worst[0], m[0]), min(worst[1], m[1]), max(worst[2], m[2])]
    return worst[0], worst[1], worst[2]


def _straight(a: np.ndarray, b: np.ndarray) -> PlannedPath:
    return PlannedPath(ok=True, reason=None, vertices=np.stack([a, b]), methods=("direct",), min_table_m=0.0,
                       min_object_m=0.0, max_self_m=0.0, length_rad=0.0, tolerance_box_ok=True)


def test_straight_move_from_the_hanging_home_goes_through_the_table(setup) -> None:
    scene, start, _, hold = setup
    chk = TransferChecker(scene, [start], 0.02)
    table, _, _ = _sweep(chk, np.stack([np.concatenate([HANGING[:14], synthetic.HAND_Q]), hold]))
    assert table < -0.05


def test_planned_move_to_the_hold_pose_is_clear(setup, g1) -> None:
    """The planner needs a detour (RRT) for the right arm; an independent 0.002 rad sweep of its path keeps >= 1.5 cm
    from the table and the box, the path starts at the home (hand 0) and ends at the hold pose, and the 0.08 rad box
    around the home is checked."""
    scene, start, _, hold = setup
    chk = TransferChecker(scene, [start], 0.02)
    th = plan_to_hold(chk, HANGING, hold, g1, TrajectoryLimits(), TransferConfig())
    assert th.ok and th.tolerance_box_ok and "rrt" in th.methods and len(th.vertices) >= 3
    assert np.allclose(th.vertices[0], np.concatenate([HANGING[:14], np.zeros(7)]))
    assert np.allclose(th.vertices[-1], hold)
    table, obj, self_d = _sweep(TransferChecker(scene, [start], 0.02), th.vertices)
    assert table >= 0.015 and obj >= 0.015 and self_d <= 0.0005
    gh = plan_go_home(TransferChecker(scene, [start], 0.02), hold, HANGING, g1, TrajectoryLimits(), TransferConfig())
    assert gh.ok and np.allclose(gh.vertices[-1][:14], HANGING[:14])
    table, obj, _ = _sweep(TransferChecker(scene, [start], 0.02), gh.vertices)
    assert table >= 0.015 and obj >= 0.015


def test_a_home_inside_the_table_is_not_planned(setup, g1) -> None:
    """The official zero posture puts the forearms horizontal at ~0.86-0.92 m: under a 0.90 m table top they are
    inside it, so no path from there is planned (and the package refuses that home)."""
    _, start, _, hold = setup
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    high = start.copy()
    high[2, 3] = 0.90 + 0.031
    sc = build_replay_scene(box.vertices, box.faces, 0.06, high, SceneConfig(table_z=0.90))
    th = plan_to_hold(TransferChecker(sc, [high], 0.02), np.zeros(17), hold, g1, TrajectoryLimits(), TransferConfig())
    assert not th.ok and "from the table" in th.reason


def _run(setup, g1, th: PlannedPath, gh: PlannedPath, stop=None) -> dict:
    scene, start, traj, _ = setup
    fields = plan_fields(traj, TrajectoryLimits(), g1, [(HANGING, th, gh)], 0.08)
    plan = S.Plan(**{k: fields[k] for k in S.Plan.__dataclass_fields__})
    assert plan.validate() == []
    return simulate_run(scene, plan, 0, start, stop=stop)


def _stats(run: dict, drills: list[dict]) -> dict:
    """Gate input for one home; the table-tolerance runs are stand-ins (the same run at both edges: the gate
    refuses a package planned below the 1 cm tolerance), and so are the object placement corners and the
    measured-state runs (the same run, recorded at each corner / edge), with passing object criteria (the static
    trajectory never moves the box)."""
    crit = {"reference_displacement_m": 0.1, "displacement_m": 0.1, "final_error_m": 0.0, "fell": False}
    tol = [run | {"case": "table_high", "dz_m": 0.01, "grow_m": 0.01}, run | {"case": "table_low", "dz_m": -0.01,
                                                                              "grow_m": 0.0}]
    corners = [run | {"case": c, "dx_m": 0.01 * sx, "dy_m": 0.01 * sy, "dyaw_deg": 5.0 * sa, "object_criteria": crit}
               for c, (sx, sy, sa) in S.PLACEMENT_CASES.items()]
    states = []
    for name in S.state_run_names(4):
        arm, hand = np.zeros(14), np.zeros(7)
        if name.startswith("arm_axis_"):
            arm[int(name[9:11])] = 0.07992 * (1 if name.endswith("+") else -1)
        elif name.startswith("arm_corner_"):
            arm[:] = 0.07992
        else:
            hand[:] = 0.0999 * (1 if name == "hand+" else -1)
        measured = {"q17": (HANGING + np.concatenate([arm, np.zeros(3)])).tolist(), "hand": hand.tolist()}
        states.append(run | {"name": name, "home_tol": 0.08, "arm_offset": arm.tolist(), "hand_offset": hand.tolist(),
                             "clipped": [], "measured_start": measured})
    return {"transfers": {"table_tol_m": 0.01, "home_tol_rad": 0.08, "hold_speed_rad_s": 0.3, "speed_scale": 1.0,
                          "start_hand": [0.0] * 7,
                          "send_tau_ff": True, "max_track_err_rad": 0.3, "hold_timeout_s": 120.0,
                          "object_xy_tol_m": 0.01, "object_yaw_tol_deg": 5.0,
                          "homes": [{"name": "arms_hanging", "home_q": HANGING.tolist(), "index": 0,
                                     "to_hold_plan": {"ok": True}, "go_home_plan": {"ok": True},
                                     "run": run | {"object_criteria": crit}, "drills": drills,
                                     "table_tolerance_runs": tol, "placement_runs": corners, "state_runs": states}]}}


def test_a_straight_to_hold_through_the_table_fails_the_gate(setup, g1) -> None:
    """The run script simulated from the hanging home along the straight TO_HOLD line (the early behaviour) presses
    the hand into the table and the gate refuses that home; along the planned path (control) the same simulation has
    no table, object or self contact outside the playback and the home passes, operator-stop drills included."""
    scene, start, traj, hold = setup
    home21 = np.concatenate([HANGING[:14], np.zeros(7)])
    end = np.concatenate([traj.arm_q[-1], C.LEFT_ARM_PARK, traj.hand_q[-1]])
    cfg = TransferConfig()
    gh = plan_go_home(TransferChecker(scene, [start], 0.02), end, HANGING, g1, TrajectoryLimits(), cfg)
    bad = _run(setup, g1, _straight(home21, hold), gh)
    assert bad["phases"]["to_hold"]["table_contact_steps"] > 0                 # the hand is pressed onto the table
    assert bad["status"] == "aborted" and bad["abort_phase"] == "to_hold" and "tracking" in bad["reason"]
    allowed, fails = S.transfer_gate(_stats(bad, [bad | {"status": "stopped"}]), HANGING[None])
    assert allowed == [] and any("did not complete" in f for f in fails)
    finished = bad | {"status": "completed"}                                  # even had it not aborted
    allowed, fails = S.transfer_gate(_stats(finished, [finished | {"status": "stopped"}]), HANGING[None])
    assert allowed == [] and any("to_hold: the robot touches the table" in f for f in fails)

    th = plan_to_hold(TransferChecker(scene, [start], 0.02), HANGING, hold, g1, TrajectoryLimits(), cfg)
    good = _run(setup, g1, th, gh)
    assert good["status"] == "completed"
    for ph in S.TRANSFER_PHASES:
        m = good["phases"][ph]
        assert m["table_contact_steps"] == 0 and m["object_contact_steps"] == 0 and m["self_collision_max_depth"] == 0.0
    drills = [_run(setup, g1, th, gh, stop=s) for s in cfg.drills]
    assert all(d["status"] == "stopped" for d in drills)
    assert S.transfer_gate(_stats(good, drills), HANGING[None]) == ([0], [])


def test_the_checker_treats_the_table_as_a_solid_block(g1) -> None:
    """Only the 4 cm top is simulated; the planner's checker extends the table to the floor (legs and frame are
    unknown). Known answer: under a 1.05 m table top the right hand reaching forward and down (wrist at x 0.21 m,
    z 0.74 m: over the table, 27 cm under its top) touches nothing in the replay scene but is deep inside the
    checker's table."""
    import mujoco

    from fastdo.robot.arm_ik import site_pose
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    far = np.eye(4)
    far[:3, 3] = [0.6, 0.3, 1.05 + 0.031]
    scene = build_replay_scene(box.vertices, box.faces, 0.06, far, SceneConfig(table_z=1.05))
    chk = TransferChecker(scene, [far], 0.02)
    q = np.array([-0.3, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0])
    wrist = site_pose(g1, q)[0, :3, 3]
    assert 0.2 < wrist[0] < 0.78 and wrist[2] < 1.05 - 0.25
    assert chk.measure(np.concatenate([q, C.LEFT_ARM_PARK, np.zeros(7)]))[0] < -0.05
    g = scene.table_geom
    assert chk.data.geom_xpos[g][2] == pytest.approx(1.05 / 2) and chk.model.geom_size[g][2] == pytest.approx(1.05 / 2)
    d = mujoco.MjData(scene.g1.model)
    d.qpos[:] = scene.g1.default_qpos()
    d.qpos[scene.g1.right_arm_qpos] = q
    mujoco.mj_forward(scene.g1.model, d)
    assert not any(g in (int(c.geom[0]), int(c.geom[1])) for c in d.contact[: d.ncon])


def test_finger_finger_contact_is_not_an_arm_collision(setup) -> None:
    """Pairs inside the right hand are not counted (they do not depend on the arm path). Known answer: an earlier C4
    re-export's first hand command presses the thumb tip 3.3 mm into the index tip; at the hanging home the replay
    scene sees exactly that contact, the checker reads no self-collision."""
    import mujoco
    scene, start, _, _ = setup
    shape = np.array([-0.5117, -0.2302, -1.4214, 0.6691, 1.7353, 0.8044, 1.6566])
    m = scene.g1.model
    d = mujoco.MjData(m)
    d.qpos[:] = scene.g1.default_qpos()
    d.qpos[scene.g1.right_arm_qpos] = HANGING[:7]
    d.qpos[scene.g1.right_hand_qpos] = shape
    mujoco.mj_forward(m, d)
    names = {tuple(sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[int(g)]) for g in c.geom))
             for c in d.contact[: d.ncon] if c.dist < -0.003}
    assert names == {("right_hand_index_1_link", "right_hand_thumb_2_link")}
    assert TransferChecker(scene, [start], 0.02).measure(np.concatenate([HANGING[:14], shape]))[2] == 0.0


def test_a_home_whose_tolerance_box_hits_the_table_is_not_planned(setup, g1) -> None:
    """Known answer: the official zero posture holds the forearms ~6 cm above the 0.80 m table top. Its own paths
    exist (the nominal home is clear), but the run script accepts a measured pose up to 0.08 rad away, and from
    corners of that box the straight move to the path touches the table: neither path is planned (mutation: the box
    unchecked)."""
    scene, start, traj, hold = setup
    zeros = np.zeros(17)
    chk = TransferChecker(scene, [start], 0.02)
    assert chk.measure(np.concatenate([zeros[:14], np.zeros(7)]))[0] > 0.001          # the home itself is clear
    th = plan_to_hold(chk, zeros, hold, g1, TrajectoryLimits(), TransferConfig())
    assert not th.ok and not th.tolerance_box_ok and th.methods and "home tolerance" in th.reason
    end = np.concatenate([traj.arm_q[-1], C.LEFT_ARM_PARK, traj.hand_q[-1]])
    gh = plan_go_home(TransferChecker(scene, [start], 0.02), end, zeros, g1, TrajectoryLimits(), TransferConfig())
    assert not gh.ok and not gh.tolerance_box_ok and gh.methods


# --- the README lets the lab place the table +-1 cm; the planner and the gate
# --- must hold for every such table, not only the nominal one ------------------------------------------------------

def _raised(g1):
    """The synthetic hold pose 6.4 mm above the table top (table at 0.9157 m; the box moved out of the hand's way)."""
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    tz = 0.80 + 0.1207 - 0.005
    start = np.eye(4)
    start[:3, 3] = [0.70, 0.30, tz + 0.031]
    scene = build_replay_scene(box.vertices, box.faces, 0.06, start, SceneConfig(table_z=tz))
    return scene, start


def test_a_hold_pose_inside_the_table_tolerance_is_not_planned(setup, g1) -> None:
    """Known answer: the hold pose is 6.4 mm above the nominal table. Planned against the nominal table only (the
    earlier planner, `table_tol` 0) the path is found, the run script simulated on the nominal table never touches it,
    but on a table 1 cm higher (inside the README's tolerance) it presses the hand onto the table in TO_HOLD. With
    the tolerance (default 1 cm) the hold pose is refused: the endpoint relaxation never goes below the tolerance
    (mutation: the checker's table not grown)."""
    from dataclasses import replace

    from fastdo.robot.replay import perturbed_table_scene
    from fastdo.robot.transfer import z_shifted
    _, _, traj, hold = setup
    scene, start = _raised(g1)
    chk = TransferChecker(scene, [start], 0.02)
    assert 0.005 < chk.measure(hold)[0] < 0.01
    old = replace(TransferConfig(), table_tol=0.0)
    th = plan_to_hold(chk, HANGING, hold, g1, TrajectoryLimits(), old)
    end = np.concatenate([traj.arm_q[-1], C.LEFT_ARM_PARK, traj.hand_q[-1]])
    gh = plan_go_home(TransferChecker(scene, [start], 0.02), end, HANGING, g1, TrajectoryLimits(), old)
    assert th.ok and gh.ok
    fields = plan_fields(traj, TrajectoryLimits(), g1, [(HANGING, th, gh)], 0.08)
    plan = S.Plan(**{k: fields[k] for k in S.Plan.__dataclass_fields__})
    nominal = simulate_run(scene, plan, 0, start)
    assert nominal["status"] == "completed" and nominal["phases"]["to_hold"]["table_contact_steps"] == 0
    up = start.copy()
    up[2, 3] += 0.01
    high = simulate_run(perturbed_table_scene(scene, 0.01, 0.01), plan, 0, up)
    assert high["phases"]["to_hold"]["table_contact_steps"] > 0

    cfg = TransferConfig()
    assert cfg.table_tol == pytest.approx(0.01)
    th = plan_to_hold(TransferChecker(scene, z_shifted([start], cfg.table_tol), 0.02, cfg.table_tol), HANGING, hold,
                      g1, TrajectoryLimits(), cfg)
    assert not th.ok and "from the table" in th.reason


def test_the_checker_grows_the_table_by_the_tolerance(g1) -> None:
    """Known answer: a configuration 6.4 mm above the nominal top reads 6.4 mm, and 3.6 mm INSIDE the table grown by
    1 cm; the object poses are also checked 1 cm up and down."""
    from fastdo.robot.transfer import z_shifted
    scene, start = _raised(g1)
    hold = np.concatenate([synthetic.arm_path()[0], C.LEFT_ARM_PARK, synthetic.HAND_Q])
    t0 = TransferChecker(scene, [start], 0.02).measure(hold)[0]
    t1 = TransferChecker(scene, [start], 0.02, 0.01).measure(hold)[0]
    assert t1 == pytest.approx(t0 - 0.01, abs=2e-4)
    poses = z_shifted([start], 0.01)
    assert len(poses) == 3 and sorted(p[2, 3] - start[2, 3] for p in poses) == pytest.approx([-0.01, 0.0, 0.01])
    assert len(z_shifted([start], 0.0)) == 1


# --- surviving mutants N3 and N6 -----------------------------------------------------------------------------------

def test_the_way_home_is_checked_against_the_object_end_poses(setup, g1) -> None:
    """Known answer (table at 0.60 m, box START far away at (0.70, 0.30)): the box's END pose (reference and
    simulated) sits on the hand's way home, 30 % along the right arm's straight line from the hold pose to the
    hanging home. Planned against the start pose only, the way home passes 38 mm into the end box; `check_transfers`
    plans it against the end poses, so its stored path keeps >= 1.5 cm from the end box (mutation N3: the GO_HOME
    checker given the object start pose only)."""
    from dataclasses import replace

    from fastdo.robot.arm_ik import site_pose
    from fastdo.robot.transfer import check_transfers
    _, _, traj, hold = setup
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    start = np.eye(4)
    start[:3, 3] = [0.70, 0.30, 0.60 + 0.031]
    scene = build_replay_scene(box.vertices, box.faces, 0.06, start, SceneConfig(table_z=0.60))
    end = np.eye(4)
    end[:3, 3] = site_pose(g1, (hold[:7] + 0.3 * (HANGING[:7] - hold[:7]))[None])[0, :3, 3]
    blind = plan_go_home(TransferChecker(scene, [start], 0.02, 0.01), hold, HANGING, g1, TrajectoryLimits(),
                         TransferConfig())
    assert blind.ok and _sweep(TransferChecker(scene, [end], 0.02), blind.vertices)[1] < -0.02
    cfg = replace(TransferConfig(), homes=(("arms_hanging", tuple(HANGING)),), drills=())
    rep = check_transfers(scene, traj, TrajectoryLimits(), g1, start, end, end, cfg)
    assert rep.homes[0]["go_home_plan"]["ok"]
    rows = rep.fields["go_home_path"]
    verts = np.concatenate([rows[:, S.RIGHT], rows[:, S.LEFT], rows[:, S.HAND]], axis=1)
    assert _sweep(TransferChecker(scene, [end], 0.02), verts)[1] >= 0.015


def test_the_whole_tolerance_box_is_checked(setup, g1) -> None:
    """Known answer (official zero posture, table top 0.76 m): the box of +-0.04 rad around the home is clear, the
    run script's +-0.08 rad box is not, so the home is planned with a 0.04 rad tolerance and refused with 0.08 (mutation
    N6: the box sampled at half the tolerance)."""
    from dataclasses import replace
    _, _, _, hold = setup
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    start = np.eye(4)
    start[:3, 3] = [synthetic.OBJECT_XY[0], synthetic.OBJECT_XY[1], 0.76 + 0.031]
    scene = build_replay_scene(box.vertices, box.faces, 0.06, start, SceneConfig(table_z=0.76))
    chk = TransferChecker(scene, [start], 0.02, 0.01)
    half = plan_to_hold(chk, np.zeros(17), hold, g1, TrajectoryLimits(), replace(TransferConfig(), home_tol=0.04))
    assert half.ok and half.tolerance_box_ok
    full = plan_to_hold(chk, np.zeros(17), hold, g1, TrajectoryLimits(), TransferConfig())
    assert TransferConfig().home_tol == pytest.approx(0.08)
    assert not full.ok and not full.tolerance_box_ok and "home tolerance" in full.reason


def test_go_home_box_checks_the_last_commanded_segment(setup, g1, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-3 review MINOR: at run time GO_HOME's last row is the MEASURED home, so the segment actually commanded is
    vertex[-2] -> (a point of the tolerance box). `plan_go_home` checks exactly that segment from every box point
    (mutation: the old check, box point -> the checked home)."""
    from fastdo.robot import transfer as T
    scene, start, traj, _ = setup
    end = np.concatenate([traj.arm_q[-1], C.LEFT_ARM_PARK, traj.hand_q[-1]])
    calls: list[tuple] = []
    real = T._Segments.edge

    def spy(self, a, b, lim):
        calls.append((np.array(a), np.array(b), lim))
        return real(self, a, b, lim)

    seen: dict = {}
    real_box = T._box_ok

    def box_spy(seg, home, hand_home, target, cfg, rng, lim, last_lim=None):
        seen.update(target=np.array(target), last_lim=last_lim)
        monkeypatch.setattr(T._Segments, "edge", spy)
        try:
            return real_box(seg, home, hand_home, target, cfg, rng, lim, last_lim)
        finally:
            monkeypatch.setattr(T._Segments, "edge", real)

    monkeypatch.setattr(T, "_box_ok", box_spy)
    gh = plan_go_home(TransferChecker(scene, [start], 0.02), end, HANGING, g1, TrajectoryLimits(), TransferConfig())
    assert gh.ok and len(gh.vertices) >= 2
    assert np.allclose(seen["target"], gh.vertices[-2]) and seen["last_lim"] is not None
    cfg = TransferConfig()
    assert len(calls) == cfg.tol_samples + 28                                    # one per box point
    for a, b, lim in calls:
        assert np.allclose(a, gh.vertices[-2]) and lim is seen["last_lim"]       # FROM the vertex before home
        assert np.abs(b[:14] - HANGING[:14]).max() == pytest.approx(cfg.home_tol)  # TO a point of the box


# --- check_transfers itself plans against the grown table -----------------------------

def test_check_transfers_does_not_plan_a_hold_pose_inside_the_grown_table(setup, g1) -> None:
    """Known answer (`_raised`: the hold pose 6.4 mm above the nominal table, 3.6 mm inside the table grown by the 1 cm
    tolerance): `check_transfers` refuses the home, TO_HOLD not planned (mutation R3B2b: its TO_HOLD checker given the
    nominal table and the object start only plans it; only a golden digest caught that earlier)."""
    from dataclasses import replace

    from fastdo.robot.transfer import check_transfers
    _, _, traj, _ = setup
    scene, start = _raised(g1)
    cfg = replace(TransferConfig(), homes=(("arms_hanging", tuple(HANGING)),), drills=())
    rep = check_transfers(scene, traj, TrajectoryLimits(), g1, start, start, start, cfg)
    th = rep.homes[0]["to_hold_plan"]
    assert not th["ok"] and "from the table grown by the 1 cm placement tolerance" in th["reason"], th
    assert rep.homes[0]["in_package"] is False and rep.homes[0]["index"] is None and len(rep.fields["home_q"]) == 0


# --- the object placement tolerance is planned (envelope) and simulated (corners) ----------------

def _beside_the_hand(g1):
    """The 6 cm box on a 0.90 m table 6.6 mm beside the synthetic hold pose's hand also raised / lowered by 1 cm (the
    hand 12.1 mm above the table grown by 1 cm); at the placement corner x-y+yaw- (1 cm toward -x / +y, -5 deg) the hand
    is 2.9 mm inside it."""
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    start = np.eye(4)
    start[:3, 3] = [0.28, -0.25, 0.90 + 0.031]
    return build_replay_scene(box.vertices, box.faces, 0.06, start, SceneConfig(table_z=0.90)), start


def test_the_planner_keeps_the_object_placement_envelope(setup, g1) -> None:
    """Known answer: the hold pose clears the box at its stated pose (and 1 cm up / down) by 6.6 mm, so the earlier
    planner (no object tolerance; endpoint relaxation down to 1 mm) plans TO_HOLD, and the run script simulated with
    the box at the corner x-y+yaw- touches the box outside the playback (the gate refuses such a corner run); with the ruled
    tolerance the clearance drops by the box's envelope (17.8 mm) and `check_transfers` refuses the home (mutation:
    the envelope not applied)."""
    from dataclasses import replace

    from fastdo.robot.tolerance_runs import placement_pose
    from fastdo.robot.transfer import check_transfers, z_shifted
    _, _, traj, hold = setup
    scene, start = _beside_the_hand(g1)
    plain = TransferChecker(scene, z_shifted([start], 0.01), 0.02, 0.01)
    wide = TransferChecker(scene, z_shifted([start], 0.01), 0.02, 0.01, (0.01, 5.0))
    assert plain.measure(hold)[1] == pytest.approx(0.0066, abs=3e-4)
    assert wide.envelope[0] == pytest.approx(0.0178, abs=2e-4)
    # the same distance less the envelope (to 1e-6 m: MuJoCo's mesh distance converges within its margin)
    assert wide.measure(hold)[1] == pytest.approx(plain.measure(hold)[1] - wide.envelope[0], abs=1e-6)
    corner = placement_pose(start, -0.01, 0.01, -5.0)
    assert TransferChecker(scene, [corner], 0.02).measure(hold)[1] < 0.0

    old = TransferConfig()
    th = plan_to_hold(plain, HANGING, hold, g1, TrajectoryLimits(), old)
    end = np.concatenate([traj.arm_q[-1], C.LEFT_ARM_PARK, traj.hand_q[-1]])
    gh = plan_go_home(plain, end, HANGING, g1, TrajectoryLimits(), old)
    assert th.ok and gh.ok
    fields = plan_fields(traj, TrajectoryLimits(), g1, [(HANGING, th, gh)], 0.08)
    plan = S.Plan(**{k: fields[k] for k in S.Plan.__dataclass_fields__})
    nominal = simulate_run(scene, plan, 0, start)
    assert nominal["status"] == "completed"
    assert all(v["object_contact_steps"] == 0 for k, v in nominal["phases"].items() if k != "play")
    moved = simulate_run(scene, plan, 0, corner)
    touched = sum(v["object_contact_steps"] for k, v in moved["phases"].items() if k != "play")
    assert touched > 0
    assert any("touches the object" in f for f in S._table_like_failures("corner", moved))

    cfg = replace(TransferConfig(), homes=(("arms_hanging", tuple(HANGING)),), drills=())
    rep = check_transfers(scene, traj, TrajectoryLimits(), g1, start, start, start, cfg)
    assert not rep.homes[0]["to_hold_plan"]["ok"] and "from the object" in rep.homes[0]["to_hold_plan"]["reason"]
    assert rep.homes[0]["in_package"] is False
    assert rep.object_envelope_m == pytest.approx(0.0178, abs=2e-4)


def test_the_run_object_criteria_use_the_reference_rows(setup, g1, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wiring: the run's object criteria read the reference from `object_ref_start` to `object_end_ref`
    (the replay's first and last reference rows), not from the placed start pose. Known answer with a stubbed
    simulation (the box played from (0.70, 0.20) to (0.70, 0.28)) and a reference from (0.70, 0.20) to (0.70, 0.30),
    away from the arm's way home: reference 10 cm, moved 8 cm, final error 2 cm (mutation: the reference start taken
    as the placed start (0.40, -0.05): reference 46.1 cm)."""
    from dataclasses import replace

    from fastdo.robot import transfer as T
    from fastdo.robot.transfer import check_transfers
    scene, start, traj, _ = setup

    def stub(*args, **kwargs) -> dict:
        return {"status": "completed", "object": {"play_start_m": [0.70, 0.20, 0.831],
                                                  "play_end_m": [0.70, 0.28, 0.831],
                                                  "run_end_m": [0.70, 0.28, 0.831]}, "phases": {}}
    monkeypatch.setattr(T, "simulate_run", stub)
    ref0, ref1 = np.eye(4), np.eye(4)
    ref0[:3, 3] = [0.70, 0.20, 0.831]
    ref1[:3, 3] = [0.70, 0.30, 0.831]
    cfg = replace(TransferConfig(), homes=(("arms_hanging", tuple(HANGING)),), drills=(), state_corners=0)
    rep = check_transfers(scene, traj, TrajectoryLimits(), g1, start, ref1, ref1, cfg, object_ref_start=ref0)
    assert rep.homes[0]["index"] == 0, rep.homes[0]
    crit = rep.homes[0]["run"]["object_criteria"]
    assert crit["reference_displacement_m"] == pytest.approx(0.10) and crit["displacement_m"] == pytest.approx(0.08)
    assert crit["final_error_m"] == pytest.approx(0.02) and crit["fell"] is False
    assert all(r["object_criteria"]["reference_displacement_m"] == pytest.approx(0.10)
               for r in rep.homes[0]["placement_runs"])
