"""End to end on a synthetic clip generated from the robot (`tests/robot/synthetic.py`): the kinematic path gives back
the wrist poses and fingertip vectors it was built from, the package is self-contained, and the copied run script
validates, dry-runs, aborts on a tracking fault and refuses invalid trajectories or the wrong robot."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.robot import g1_sdk_replay as S
from fastdo.robot.arm_ik import site_pose
from fastdo.robot.kinematic_path import (
    KinematicPathConfig,
    contact_anchor,
    hand_object_distance,
    load_reconstruction,
    robot_frame_targets,
    run_kinematic_path,
    spike_rows,
    wrist_targets_world,
)
from fastdo.robot.package import write_package
from tests.robot import synthetic
from tests.robot.plans import body, with_home


def test_kinematic_path_recovers_the_generating_motion(g1, built) -> None:
    _, _, result, _ = built
    assert np.allclose(result.placement.T_rv[:3, :3], np.eye(3), atol=1e-9)
    assert np.abs(result.placement.T_rv[:3, 3]).max() < 2e-3
    assert np.allclose(result.wrist_targets[:, :3, 3], site_pose(g1, synthetic.arm_path())[:, :3, 3], atol=2e-3)
    assert np.median(result.ik.pos_err) < 1e-3 and result.ik.pos_err.max() < 5e-3
    assert np.median(result.hand_vec_err) < 2e-3
    assert result.used_rows.all() and not result.spike_rows.any()
    m = result.metrics
    assert m.limit_violation_ticks == 0 and m.self_collision_steps == 0 and m.arm_rms < 0.02
    assert not m.success and "object_not_moved" in m.failed_criteria          # the hand never touches the box


def test_wrist_targets_follow_the_hand_frame(g1) -> None:
    joints = synthetic.hand_joints(g1, synthetic.arm_path(5))
    assert np.allclose(wrist_targets_world(joints), site_pose(g1, synthetic.arm_path(5)), atol=1e-9)


def test_spike_rows_flag_single_frame_jumps() -> None:
    pos = np.zeros((9, 3))
    pos[4] = [0.3, 0.0, 0.0]
    valid = np.ones(9, dtype=bool)
    assert np.flatnonzero(spike_rows(pos, valid, 5, 0.1)).tolist() == [4]
    valid[4] = False
    assert not spike_rows(pos, valid, 5, 0.1).any()


def test_spike_rows_flag_multi_frame_bursts() -> None:
    """A 3-frame burst 0.5 m away beats a 5-row median but not the 2.5 m/s speed gate; a slow drift is kept."""
    times = np.arange(12) * 0.1
    pos = np.zeros((12, 3))
    pos[:, 0] = np.linspace(0.0, 0.2, 12)                    # 0.18 m/s drift
    pos[1:4, 1] = 0.5                                        # burst at rows 1-3
    valid = np.ones(12, dtype=bool)
    assert not spike_rows(pos, valid, 5, 0.6)[1:4].all()
    flagged = spike_rows(pos, valid, 5, 0.6, times, 2.5)
    assert np.flatnonzero(flagged).tolist() == [1, 2, 3]


def test_package_is_self_contained(built) -> None:
    _, _, result, pkg = built
    names = {p.name for p in pkg.iterdir()}
    assert names == {"README.md", "trajectory.npz", "arm_250hz.csv", "hand_100hz.csv", "sim_stats.json",
                     "package.json", "object_mesh.obj", "run_g1_replay.py", "NOTICE.md"}
    assert "NOT TESTED ON HARDWARE" in (pkg / "README.md").read_text()
    meta = json.loads((pkg / "package.json").read_text())
    assert meta["unitree_ros"]["sha"] == "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
    assert np.allclose(meta["object_start_robot"]["position_m"], result.placement.object_start[:3, 3])
    arm = np.loadtxt(pkg / "arm_250hz.csv", delimiter=",", skiprows=1)
    assert arm.shape == (len(result.trajectory.arm_t), 22)
    assert np.allclose(arm[:, 1:8], result.trajectory.arm_q, atol=1e-6)
    assert (pkg / "run_g1_replay.py").read_text() == Path(S.__file__).read_text()
    assert "import fastdo" not in (pkg / "run_g1_replay.py").read_text()
    assert S.Plan.load(pkg / "trajectory.npz").validate() == []
    with pytest.raises(FileExistsError):
        write_package(pkg, result, built[1])


def test_run_script_dry_run_in_a_fresh_interpreter(built, tmp_path: Path) -> None:
    """The copied script runs standalone (numpy only) and defaults to dry-run."""
    pkg = tmp_path / "copy"
    shutil.copytree(built[3], pkg)
    proc = subprocess.run([sys.executable, str(pkg / "run_g1_replay.py")], capture_output=True, text=True,
                          timeout=300, cwd=tmp_path, check=False)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["status"] == "completed" and out["mode"].startswith("dry-run")


def test_run_script_refuses_an_invalid_trajectory(built, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Mutation: a 0.1 rad bump in the middle of the arm stream (25 rad/s in one 4 ms tick, start and end unchanged so
    the checked paths still join) is refused by the velocity check before anything is sent."""
    pkg = tmp_path / "bad"
    shutil.copytree(built[3], pkg)
    with np.load(pkg / "trajectory.npz") as z:
        data = dict(z)
    data["arm_q"] = data["arm_q"].copy()
    data["arm_q"][200:300, 1] += 0.1
    np.savez(pkg / "trajectory.npz", **data)
    assert S.main(["--package", str(pkg)]) == 2
    err = capsys.readouterr().err
    assert "velocity" in err and "does not join" not in err


def test_replayer_aborts_on_a_tracking_fault(built) -> None:
    """At the package's checked home with the right shoulder pitch stuck (the path to the hold pose moves it > 0.3
    rad): abort on tracking during TO_HOLD."""
    plan = S.Plan.load(built[3] / "trajectory.npz")
    assert abs(plan.to_hold_rows(0)[:, 0] - plan.home_q[0, 0]).max() > 0.35
    io = S.SimulatedIO(q0=body(plan), hand0=np.zeros(7), fault={"stuck_joint": S.RIGHT_ARM_DDS[0], "after_s": 0.0})
    result = S.Replayer(plan, io, S.Config()).run()
    assert result["status"] == "aborted" and "tracking error" in result["reason"]
    assert result["abort_phase"] == "to_hold" and io.weight == 0.0


def test_replayer_completes_goes_home_and_releases(built) -> None:
    """From the package's checked home: TO_HOLD passes through every planned waypoint (within one tick's step) and no
    joint steps faster than the hold speed, the whole trajectory is commanded (its last pose reached), the way home
    passes through the planned waypoints, and the arms end at the measured start pose before the weight is released.
    Mutation (TO_HOLD / GO_HOME as one straight line, the pre-fix behaviour) misses the waypoints."""
    plan = S.Plan.load(built[3] / "trajectory.npz")
    cmds: list[tuple[str, np.ndarray]] = []

    class Logged(S.SimulatedIO):
        def send_arm(self, q, kp, kd, tau, weight, dt) -> None:
            cmds.append((rp.phase, np.array([q[i] for i in S.RIGHT_ARM_DDS + S.LEFT_ARM_DDS])))
            super().send_arm(q, kp, kd, tau, weight, dt)

    io = Logged(q0=body(plan), hand0=np.zeros(7))
    cfg = S.Config()
    rp = S.Replayer(plan, io, cfg)
    result = rp.run()
    assert result["status"] == "completed" and io.weight == 0.0
    assert set(result["phases"]) >= {"ramp_up", "to_hold", "wait_go", "play", "go_home", "release"}
    assert io.sent_arm >= len(plan.arm_q) and io.sent_hand >= len(plan.hand_q) - 1
    arms = np.asarray([c for _, c in cmds])
    assert np.abs(arms[:, :7] - plan.arm_q[-1]).max(axis=1).min() < 1e-9
    home = plan.home_q[0, :14]
    assert np.abs(arms[-1] - home).max() < 1e-12 and np.abs(io.q[list(S.RIGHT_ARM_DDS)] - home[:7]).max() < 0.02
    step = cfg.hold_speed * float(plan.arm_t[1] - plan.arm_t[0])
    for phase, rows in (("to_hold", plan.to_hold_rows(0)), ("go_home", plan.go_home_rows(0))):
        seg = np.asarray([c for ph, c in cmds if ph == phase])
        assert len(rows) >= 3                                          # the planner needed waypoints here
        assert (np.abs(np.diff(seg, axis=0)).max(axis=1) <= step * (1 + 1e-6)).all()
        hits = [int(np.argmin(np.abs(seg - w[:14]).max(axis=1))) for w in rows[1:-1]]
        assert all(np.abs(seg[h] - w[:14]).max() <= step + 1e-9 for h, w in zip(hits, rows[1:-1], strict=True))
        assert hits == sorted(hits)                                    # in order


def test_replayer_refuses_the_wrong_robot_and_stale_state(built) -> None:
    plan = S.Plan.load(built[3] / "trajectory.npz")
    io = S.SimulatedIO(q0=body(plan), mode_machine=13)
    res = S.Replayer(plan, io, S.Config()).run()
    assert res["status"] == "refused" and "mode_machine" in res["reason"] and io.sent_arm == 0

    class Stale(S.SimulatedIO):
        field = "stamp"

        def state(self) -> S.State:
            st = super().state()
            return dataclasses.replace(st, **{self.field: getattr(st, self.field) - 1.0})

    for field in ("stamp", "hand_stamp"):
        io = Stale(q0=body(plan), hand0=np.zeros(7))                  # at a checked home: only the staleness refuses
        io.field = field
        res = S.Replayer(plan, io, S.Config()).run()
        assert res["status"] == "refused" and "fresh" in res["reason"] and io.sent_arm == 0


def test_execute_needs_an_interface(built) -> None:
    assert S.main(["--package", str(built[3]), "--execute"]) == 2
    assert S.main(["--package", str(built[3]), "--speed-scale", "2"]) == 2


def test_hand_object_distance_known_answer() -> None:
    """A 6 cm box (unit mesh x 0.06) at a rotated pose: a point 5 cm along its local x is 2 cm from the surface."""
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    pose[:3, 3] = [0.3, -0.2, 0.9]
    pts = np.array([[[0.0, 0.05, 0.0], [0.0, 0.0, 0.2]]]) + pose[:3, 3]      # world +y = box local x
    dist = hand_object_distance(pts, pose[None], box.vertices, box.faces, 0.06, np.array([True]))
    assert dist[0] == pytest.approx(0.02, abs=1e-9)
    assert np.isnan(hand_object_distance(pts, pose[None], box.vertices, box.faces, 0.06, np.array([False]))[0])


def test_contact_anchor_is_the_first_used_contact_row() -> None:
    dist = np.array([0.001, 0.03, 0.01, 0.0])
    assert contact_anchor(dist, np.array([False, True, True, True]), 0.015) == (2, 2)      # row 0 is not used
    assert contact_anchor(dist, np.array([False, True, False, False]), 0.015) == (1, None)
    with pytest.raises(ValueError):
        contact_anchor(dist, np.zeros(4, dtype=bool), 0.015)


def test_robot_frame_targets_hold_the_object_until_the_anchor() -> None:
    rng = np.random.default_rng(3)
    n, anchor = 6, 3
    obj_w = np.tile(np.eye(4), (n, 1, 1))
    wrist_w = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        obj_w[k, :3, :3] = Rotation.random(random_state=rng).as_matrix()
        wrist_w[k, :3, :3] = Rotation.random(random_state=rng).as_matrix()
        obj_w[k, :3, 3], wrist_w[k, :3, 3] = rng.normal(size=3), rng.normal(size=3)
    T_rv = np.eye(4)
    T_rv[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T_rv[:3, 3] = rng.normal(size=3)
    start = T_rv @ obj_w[anchor]
    wrist_r, obj_r = robot_frame_targets(T_rv, start, wrist_w, obj_w, anchor)
    rel_h = np.linalg.solve(obj_w, wrist_w)
    assert np.allclose(np.linalg.solve(obj_r, wrist_r), rel_h, atol=1e-9)        # hand-object relation everywhere
    assert np.allclose(obj_r[:anchor + 1], start, atol=1e-12)                      # static until the anchor
    assert np.allclose(wrist_r[anchor:], T_rv @ wrist_w[anchor:], atol=1e-9)       # rigid from the anchor on
    plain_w, _ = robot_frame_targets(T_rv, start, wrist_w, obj_w, 0)
    assert np.allclose(plain_w, T_rv @ wrist_w, atol=1e-9)


@pytest.mark.parametrize("contact_m", [0.015, 0.0])
def test_anchor_puts_the_sim_object_where_the_hand_grasps(g1, tmp_path: Path, contact_m: float) -> None:
    """Known answer: the index tip touches the box from row 10, and before that the reconstructed box drifts 1 m away.
    The anchor is row 10, so the sim object (object_ref[0]) is the box at the grasp; the drifting rows are dropped
    (measured: 1 m from the contact pose) and refilled by synthetic approach rows. Mutation `contact_m=0` (no contact
    found: anchored at the first row, the pre-fix behaviour) leaves the sim object ~1 m from the grasp."""
    contact_row, drift = 10, 1.0
    clip = synthetic.write_clip(g1, tmp_path / "contact", synthetic.contact_object_path(g1, contact_row, drift))
    result = run_kinematic_path(load_reconstruction(clip),
                                KinematicPathConfig(table_xy=synthetic.OBJECT_XY, contact_m=contact_m), g1)
    gap = float(np.linalg.norm(result.object_ref[contact_row, :3, 3] - result.object_ref[0, :3, 3]))
    if contact_m > 0:
        assert result.anchor_row == contact_row and result.contact_row == contact_row
        assert result.hand_object_dist[contact_row] == pytest.approx(0.001, abs=1e-6)
        assert np.isnan(result.hand_object_dist[:contact_row]).all() and (result.source_rows[:contact_row] < 0).all()
        assert gap < 1e-9 and result.pre_anchor_object_motion == 0.0
        assert result.object_motion["pre_contact_dropped_m"] == pytest.approx(drift, abs=1e-6)
        assert np.allclose(result.placement.object_start[:2, 3], synthetic.OBJECT_XY, atol=1e-9)
    else:
        assert result.anchor_row == 0 and result.contact_row is None
        assert gap == pytest.approx(drift, abs=1e-6)
    summary = result.summary()
    assert summary["anchor_row"] == result.anchor_row and "object_contact_steps" in summary["replay"]


def test_placement_search_is_reported_and_an_overhang_refused(g1, tmp_path: Path) -> None:
    """With `table_xy=None` the placement is searched (`fastdo.robot.reach`): the result carries the search records,
    the choice keeps the box on the table, the export path follows the chosen table height, and the adaptations name
    the searched placement. An explicit overhanging `table_xy` is refused."""
    from fastdo.robot.reach import ReachSearchConfig
    recon = load_reconstruction(synthetic.write_clip(g1, tmp_path / "grid"))
    small = ReachSearchConfig(pitch_deg=(None,), yaw_deg=(0.0, 30.0), anchor_points=((0.28, -0.22, 0.95),),
                              top_coarse=1, top_final=1, max_rounds=1, min_step_xy=0.04)
    result = run_kinematic_path(recon, KinematicPathConfig(search=small), g1)
    assert result.search is not None and len(result.search.records) >= 2
    assert {r["stage"] for r in result.search.records} == {"coarse", "local", "final"}
    assert result.placement.table_z == result.choice.table_z and 0.70 <= result.choice.table_z <= 1.00
    assert result.table_check is not None and result.table_check["on_table"]
    assert any(a.startswith("placement (chosen by the placement search)") for a in result.adaptations)
    summary = result.summary()
    assert summary["search"]["evaluated"] == len(result.search.records) and summary["choice"]["table_z"] == result.choice.table_z
    pkg = write_package(tmp_path / "pkg", result, recon)
    records = json.loads((pkg / "placement_search.json").read_text())
    assert len(records) == len(result.search.records) and all("leximin" not in r for r in records)
    readme = (pkg / "README.md").read_text()
    assert "## Adaptations" in readme and all(a in readme for a in result.adaptations)
    meta = json.loads((pkg / "package.json").read_text())
    assert meta["choice"] == result.choice.to_dict() and meta["adaptations"] == list(result.adaptations)
    cmd = meta["reexport_command"]
    assert f"--yaw-offset-deg {result.choice.yaw_deg:g}" in cmd and "--object-xy X Y --table-z Z" in cmd
    stats = json.loads((pkg / "sim_stats.json").read_text())
    assert stats["reach_margin"]["infeasible_rows"] == len(stats["table_check"]["infeasible"])
    with pytest.raises(ValueError, match="overhangs"):
        run_kinematic_path(recon, KinematicPathConfig(table_xy=(0.20, 0.0)), g1)


def test_contact_span_trims_rows_and_synthesises_the_approach(g1, tmp_path: Path) -> None:
    """Known answer: the index tip touches the box from row 10 to the end (19), the box static before (a 1.5 mm
    contact threshold: the static box sits 2 mm under the tip on row 9). With the 1 s lead the kept rows are 0-19
    (row 0 is exactly 1 s before), the robot wrist descends vertically onto the row-10 pose from 8 cm above, and the
    adaptations say so. With `approach_m=0` the human's own pre-contact path stays. With a box drifting 1 m before a
    contact at row 15 (default threshold), rows 5-14 are dropped and 10 synthetic rows keep the 1 s approach (same
    row count and duration)."""
    clip = synthetic.write_clip(g1, tmp_path / "span", synthetic.contact_object_path(g1, 10, 0.0))
    recon = load_reconstruction(clip)
    res = run_kinematic_path(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY, contact_m=0.0015), g1)
    assert res.row_range == (0, 19) and res.anchor_row == 10
    w = res.wrist_targets
    assert np.allclose(w[0, :3, 3], w[10, :3, 3] + [0, 0, 0.08], atol=1e-9)
    assert np.allclose(w[:10, :2, 3], w[10, :2, 3], atol=1e-9) and np.allclose(w[:10, :3, :3], w[10, :3, :3])
    assert any(a.startswith("approach / retreat") for a in res.adaptations)
    human = run_kinematic_path(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY, approach_m=0.0,
                                                          contact_m=0.0015), g1)
    assert not np.allclose(human.wrist_targets[:10, :3, 3], human.wrist_targets[10, :3, 3])
    assert not any(a.startswith("approach / retreat") for a in human.adaptations)
    late = run_kinematic_path(load_reconstruction(synthetic.write_clip(
        g1, tmp_path / "late", synthetic.contact_object_path(g1, 15, 1.0))), KinematicPathConfig(
        table_xy=synthetic.OBJECT_XY), g1)
    assert late.row_range == (15, 19) and late.used_rows.size == 15 and late.anchor_row == 10
    assert (late.source_rows[:10] == -1).all() and late.source_rows[10:].tolist() == list(range(15, 20))
    assert late.trajectory.source_t[0] == 0.0 and late.trajectory.source_t[-1] == pytest.approx(1.4)


def test_warnings_name_an_unestimated_gravity_and_a_static_camera(built, tmp_path: Path) -> None:
    from fastdo.robot.kinematic_path import path_warnings
    clip, _, result, pkg = built
    assert result.warnings == () and result.gravity_source == "unknown" and result.up_source == "record"
    w = path_warnings("gravity_aligned_static", "default", None, -0.025)
    assert len(w) == 3 and w[0].startswith("GRAVITY NOT ESTIMATED") and "CAMERA ASSUMED STATIC" in w[1]
    assert "2.5 cm BELOW" in w[2]
    assert path_warnings("geometry", "geometry", None, 0.05) == ()
    assert "ASSUMED head pitch of 45 deg" in path_warnings("geometry", "default", 45.0, 0.05)[0]
    copy = tmp_path / "with_qc"
    shutil.copytree(clip, copy)
    (copy / "qc.json").write_text(json.dumps({"camera_source": "gravity_aligned_static", "gravity_source": "default"}))
    recon = load_reconstruction(copy)
    assert (recon.camera_source, recon.gravity_source) == ("gravity_aligned_static", "default")
    assert "## Warnings" in (pkg / "README.md").read_text()


def test_hand_moves_to_the_hold_pose_at_the_hold_speed(built) -> None:
    """Known answer: arm, left arm and waist already at the hold pose, the right hand 0.8 rad away on thumb_0. The
    move to the hold pose must still take the hand there at <= `hold_speed` (mutation, the hand left out of the span:
    one 4 ms tick, 200 rad/s), and no hand command of the whole run may exceed the plan's hand velocity limit."""
    plan = S.Plan.load(built[3] / "trajectory.npz")
    q0 = np.zeros(35)
    q0[list(S.RIGHT_ARM_DDS)] = plan.arm_q[0]
    q0[list(S.LEFT_ARM_DDS)] = plan.left_park
    plan = with_home(plan, q0)                                         # a checked home AT the hold pose
    hand0 = plan.hand_q[0].copy()
    hand0[0] += -0.8 if hand0[0] > 0 else 0.8
    assert plan.hand_limits[0, 0] <= hand0[0] <= plan.hand_limits[0, 1]

    sent: list[tuple[float, np.ndarray]] = []

    class Logged(S.SimulatedIO):
        def send_hand(self, q: np.ndarray, dt: float) -> None:
            sent.append((self.t, q.copy()))
            super().send_hand(q, dt)

    io = Logged(q0=q0, hand0=hand0)
    cfg = S.Config()
    result = S.Replayer(plan, io, cfg).run()
    assert result["status"] == "completed"
    t = np.array([e[0] for e in sent])
    q = np.array([e[1] for e in sent])
    speed = np.abs(np.diff(q, axis=0)).max(axis=1) / np.maximum(np.diff(t), 1e-9)
    to_hold = int(np.argmax(np.abs(q - plan.hand_q[0]).max(axis=1) < 1e-12))     # first command AT the hold pose
    dt = float(plan.arm_t[1] - plan.arm_t[0])
    assert np.abs(q[0] - hand0).max() <= cfg.hold_speed * dt * (1 + 1e-6)          # first step from the measured hand
    assert to_hold > 1 and speed[:to_hold].max() <= cfg.hold_speed * (1 + 1e-6)
    assert speed.max() <= plan.hand_vmax.max() * (1 + 1e-3)


def test_hand_state_topic_override_is_accepted(built) -> None:
    """`rt/dex3/right/state` vs the SDK example's `rt/lf/dex3/right/state` is UNVERIFIED: the lab can switch without
    editing the script; the dry-run accepts the flag and an empty topic is refused."""
    assert S.main(["--package", str(built[3]), "--hand-state-topic", "rt/lf/dex3/right/state"]) == 0
    assert S.main(["--package", str(built[3]), "--hand-state-topic", ""]) == 2


# --- kinematic_path: the pre-contact freeze and the approach / retreat need a MEASURED static object --

def test_object_moving_before_the_contact_is_measured_and_dropped(g1, tmp_path: Path) -> None:
    """The reconstructed box drifts up to 1 m (row 0) before the contact at row 10, while the hand is away. Its motion
    is measured, the drifting rows are dropped with a warning that names it, 10 synthetic rows keep the approach's
    1 s, and the adaptation states the measured motion over the rows the reference holds static."""
    clip = synthetic.write_clip(g1, tmp_path / "drift", synthetic.contact_object_path(g1, 10, 1.0))
    res = run_kinematic_path(load_reconstruction(clip), KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1)
    assert res.row_range == (10, 19) and (res.source_rows[:10] == -1).all() and res.anchor_row == 10
    assert res.trajectory.source_t[-1] == pytest.approx(1.9)                      # the approach keeps its duration
    assert np.allclose(res.wrist_targets[0, :3, 3], res.wrist_targets[10, :3, 3] + [0, 0, 0.08], atol=1e-9)
    assert np.allclose(res.object_ref[:11], res.placement.object_start, atol=1e-12)
    mo = res.object_motion
    assert mo["pre_contact_dropped_m"] == pytest.approx(1.0, abs=1e-6) and mo["pre_contact_kept_m"] == 0.0
    assert mo["synthetic_rows_before"] == 10 and mo["synthetic_rows_after"] == 0
    assert any("moves up to 100.0 cm" in w and "rows 0-9" in w for w in res.warnings)
    assert any(a.startswith("approach / retreat") and "10 / 0 synthetic rows" in a for a in res.adaptations)
    # no video row is kept before the onset: nothing was measured there, and no "0.0 cm" says it was
    assert not any("moves at most 0.0 cm" in a for a in res.adaptations)
    assert any(a.startswith("approach / retreat") and "no video rows are kept before the first contact (10 synthetic"
               in a for a in res.adaptations)
    assert any(a.startswith("object reference held") and "no video rows are kept before it" in a
               for a in res.adaptations)
    summary = res.summary()
    assert summary["object_motion"] == mo and summary["synthetic_rows"] == 10 and summary["anchor_clip_row"] == 10


def test_object_motion_within_the_tolerance_keeps_every_row(g1, tmp_path: Path) -> None:
    """A 1.5 cm pre-contact drift (tolerance 2 cm; contact at row 10 with a 1.5 mm threshold, as in
    `test_contact_span_trims_rows_and_synthesises_the_approach`) keeps rows 0-19; the adaptation states the measured
    1.5 cm."""
    clip = synthetic.write_clip(g1, tmp_path / "jitter", synthetic.contact_object_path(g1, 10, 0.015))
    res = run_kinematic_path(load_reconstruction(clip), KinematicPathConfig(table_xy=synthetic.OBJECT_XY,
                                                                            contact_m=0.0015), g1)
    assert res.row_range == (0, 19) and res.anchor_row == 10 and (res.source_rows == np.arange(20)).all()
    assert res.object_motion["pre_contact_kept_m"] == pytest.approx(0.015, abs=1e-6)
    assert res.object_motion["pre_contact_dropped_m"] == 0.0
    assert any("moves at most 1.5 cm" in a for a in res.adaptations)
    assert not any("dropped" in w for w in res.warnings)


def test_object_moving_after_the_last_contact_is_dropped_from_the_retreat(g1, tmp_path: Path) -> None:
    """The box is carried, released (last contact at row 15), then jumps 10 cm on rows 16-19 with the hand away:
    the retreat rows are trimmed to row 15 and refilled by 4 synthetic rows; the reference end is the release pose."""
    obj = synthetic.contact_object_path(g1, 5, 0.0)
    obj[13:] = obj[12]
    obj[16:, 1, 3] += 0.10
    clip = synthetic.write_clip(g1, tmp_path / "post", obj)
    res = run_kinematic_path(load_reconstruction(clip), KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1)
    mo = res.object_motion
    assert mo["contact_end_row"] == 15 and res.row_range[1] == 15 and mo["synthetic_rows_after"] == 4
    assert (res.source_rows[-4:] == -1).all() and mo["post_contact_dropped_m"] == pytest.approx(0.10, abs=1e-6)
    assert np.allclose(res.object_ref[-1], res.object_ref[-5], atol=1e-12)        # nothing moves after the release
    assert any("after the last contact (clip row 15)" in w and "10.0 cm" in w for w in res.warnings)


def test_an_object_rotating_before_the_contact_is_dropped(g1, tmp_path: Path) -> None:
    """Round-3 review mutant N7: the box turns 20 deg about the vertical through its centre on rows 0-9 (it does not
    move: 0 cm) before the contact at row 10 (1.5 mm threshold, as in
    `test_object_motion_within_the_tolerance_keeps_every_row`). Rotation alone beyond the 10 deg tolerance drops those rows (the 2 cm
    displacement test alone keeps them); a 5 deg turn keeps every row."""
    from scipy.spatial.transform import Rotation
    for deg, dropped in ((20.0, True), (5.0, False)):
        obj = synthetic.contact_object_path(g1, 10, 0.0)
        obj[:10, :3, :3] = Rotation.from_euler("z", deg, degrees=True).as_matrix() @ obj[:10, :3, :3]
        clip = synthetic.write_clip(g1, tmp_path / f"turn{deg:g}", obj)
        res = run_kinematic_path(load_reconstruction(clip), KinematicPathConfig(table_xy=synthetic.OBJECT_XY,
                                                                                contact_m=0.0015), g1)
        mo = res.object_motion
        assert res.contact_row == 10
        if dropped:
            assert res.row_range[0] == 10 and mo["pre_contact_dropped_deg"] == pytest.approx(20.0, abs=1e-3)
            assert mo["pre_contact_dropped_m"] == pytest.approx(0.0, abs=1e-6)
            assert any("/ 20 deg" in w for w in res.warnings)
        else:
            assert res.row_range[0] < 10 and mo["pre_contact_dropped_deg"] == 0.0
            assert mo["pre_contact_kept_deg"] == pytest.approx(5.0, abs=1e-3)
