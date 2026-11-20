"""Start-pose sidecars (R3): the export CLI (`run/robot/export_g1_package.py`, the registered entry
point) takes the window's A1 start pose (`start_pose_a1.json`) and its `support_plane.json`; an accepted plane
gives up and switches the pitch / resting-face search off; without sidecars the export is unchanged.

Known answers on the synthetic clip (`tests/robot/synthetic.py`; the box carried from row 0, so the contact onset is
row 0 = source frame 0):
- the sidecar pose P = W O_0 (W: 7 deg about the vertical through the box, plus 1 cm) and a plane whose `gravity_cam`
  is 6 deg off the record's give the SAME package as a clip whose record was moved by W on every row and whose
  `gravity_cam` is the plane's, exported without sidecars;
- with a search, `search_placement` receives P at the anchor, up = cam_to_world[0] gravity_cam, and allow_pitch False
  (True without the plane, for a clip whose qc gravity_source is "default").

Also: the same known answer with the contact onset at row 12 (the anchor is not
row 0), 15 cm of pre-contact object drift and a camera that pans, tilts and translates on every row; refusal cases
for a static sidecar on an ego-motion export and a snapped sidecar on a plane the robot path refuses. And the
record's ruled rest verdict and the correction bound (8.6 cm / 12.2 deg), on the record and on the export's own W.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.io.clip_record import ClipRecord
from tests.robot import synthetic

_PATH = Path(__file__).resolve().parents[2] / "run" / "robot" / "export_g1_package.py"
# digest of the synthetic explicit-placement export before R3 (the start-pose sidecars), `_digest` below.
# Planning TO_HOLD against the object's placement envelope (R5-2) came later; it changes the planned `to_hold_path`
# (and so the simulated runs) of the synthetic export.
GOLDEN_PRE_R3 = {"trajectory": "0b03347668db6ee4deae5848429095d0affe0e67057595fdb5c73823715eb3a7",
                 "sim_stats": "4de28a6d15d4884c3644cc55cc03b5c66a800913a5cf70700ebd2aecd36f87ff",
                 "package": "82228c28c77c10a84fa438e7a15ce0b3f993997ee900b63bd9ad39b9f70c085e"}
GOLDEN = {"trajectory": "3a95e1a6334ed3824fba7e6d89e53eab0ec02450f5130a4365726ced4f1d9add",
          "sim_stats": "833e29f0af046ba42619f874e1233617400e4fe954c4789ce26a5c8663a3d238",
          "package": "a5b751b6c7f934c9217a58a208d74c02faa06f177417da23a73f06dc97be7839"}
# every trajectory.npz array but the planned paths: unchanged since before R3
GOLDEN_STREAMS = "f5acd36962bbe0e77c6d7037c9737c23f6eee29d62139d71fb8bafb45227ab04"
PLANNED_PATHS = ("to_hold_path", "go_home_path")
NEW_KEYS = ("start_pose", "gravity")      # added by R3 to sim_stats.json and package.json
# the gate keys: the recorded Dex3 start hand, the gate's new thresholds and labels
R5_GATE_KEYS = ("table_tol_min_m", "waist_tol_rad", "start_hand_tol_rad")
# the grasp keys, written even with their flags off: the approach / closure records and the stated mass
R6_1_STATS_KEYS = ("approach", "grasp_closure", "object_mass_kg", "object_mass_note")
R6_1_PACKAGE_KEYS = ("approach", "grasp_closure", "object_mass_note")     # object_mass_kg was already in package.json


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_g1_package", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def g1():
    from fastdo.robot.g1_model import load_g1_upper, resolve_unitree_ros
    try:
        resolve_unitree_ros()
    except FileNotFoundError:
        pytest.skip("official unitree_ros clone not available")
    return load_g1_upper()


def _streams_digest(pkg: Path) -> str:
    """sha256 of every trajectory.npz array except the planned paths (`PLANNED_PATHS`)."""
    with np.load(pkg / "trajectory.npz") as z:
        h = hashlib.sha256()
        for k in sorted(set(z.files) - set(PLANNED_PATHS)):
            a = np.asarray(z[k])
            h.update(k.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _digest(pkg: Path) -> dict:
    with np.load(pkg / "trajectory.npz") as z:
        h = hashlib.sha256()
        for k in sorted(z.files):
            a = np.asarray(z[k])
            h.update(k.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    stats = json.loads((pkg / "sim_stats.json").read_text())
    stats.pop("trajectory_sha256")
    meta = json.loads((pkg / "package.json").read_text())
    for k in ("created_utc", "fastdo_commit", "source_dir", "reexport_command"):
        meta.pop(k)
    for d in (stats, meta):
        for k in NEW_KEYS:
            d.pop(k, None)
    stats["transfers"].pop("start_hand", None)
    for k in R6_1_STATS_KEYS:
        stats.pop(k, None)
    for k in R6_1_PACKAGE_KEYS:
        meta.pop(k, None)
    gate = meta["sim_gate"]
    gate.pop("notes", None)
    gate["simulated_run_options"].pop("start_hand", None)
    for k in R5_GATE_KEYS:
        gate["thresholds"].pop(k, None)
    return {"trajectory": h.hexdigest(),
            "sim_stats": hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest(),
            "package": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()}


EXPLICIT = ["--object-xy", *map(str, synthetic.OBJECT_XY), "--table-z", str(synthetic.TABLE_Z), "--arm-vmax", "0.5"]


def test_without_sidecars_the_export_is_unchanged(g1, tmp_path: Path) -> None:
    """Byte-identical (trajectory arrays, sim_stats.json and package.json apart from the two R3 keys and the
    run-specific fields) to the current export without sidecars, and every trajectory stream but the planned paths
    byte-identical to the export before R3 (mutation: an identity correction applied without a sidecar would still
    pass, so the KA test below carries the override). The full digest was re-baselined: the R5-2 envelope changes
    the synthetic `to_hold_path`, and with it the simulated runs (`GOLDEN_PRE_R3` kept for the record)."""
    cli = _load()
    clip = synthetic.write_clip(g1, tmp_path / "clip")
    assert cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT]) == 0
    assert _streams_digest(tmp_path / "pkg") == GOLDEN_STREAMS
    assert _digest(tmp_path / "pkg") == GOLDEN
    meta = json.loads((tmp_path / "pkg" / "package.json").read_text())
    assert meta["start_pose"] == {"source": "the contact-onset row's reconstructed pose (no start-pose sidecar)"}
    assert meta["gravity"]["source"].startswith("record") and "support_plane" not in meta["gravity"]
    # R6-1 off by default: vertical approach, no closure (the digest above strips only these added records)
    assert meta["approach"]["mode"] == "vertical" and meta["grasp_closure"] is None


# --- the known answer ------------------------------------------------------------------------------------------------

def _carried_clip(g1, root: Path, w: np.ndarray | None = None, gravity_cam: np.ndarray | None = None) -> Path:
    """The synthetic clip with the box carried from row 0; optionally every row moved by the world transform `w` and
    the record's `gravity_cam` replaced."""
    clip = synthetic.write_clip(g1, root, synthetic.contact_object_path(g1, 0, 0.0))
    if w is None and gravity_cam is None:
        return clip
    rec = ClipRecord.load(clip / "clip_record.npz")
    c2w = rec.cam_to_world.astype(np.float64)
    out = rec
    if w is not None:
        to_cam = np.einsum("tij,jk->tik", np.linalg.inv(c2w), w)            # world move expressed per camera
        pose = np.einsum("tij,tjk,tkl->til", to_cam, c2w, rec.object_pose.astype(np.float64))
        hr = rec.hand_right
        j = hr.joints.astype(np.float64)
        jw = np.einsum("tij,tvj->tvi", c2w[:, :3, :3], j) + c2w[:, None, :3, 3]
        jw = jw @ w[:3, :3].T + w[:3, 3]
        jc = np.einsum("tji,tvj->tvi", c2w[:, :3, :3], jw - c2w[:, None, :3, 3])
        out = replace(out, object_pose=pose.astype(np.float32),
                      hand_right=replace(hr, joints=jc.astype(np.float32), trans=jc[:, 0].astype(np.float32)))
    if gravity_cam is not None:
        out = replace(out, gravity_cam=np.asarray(gravity_cam, dtype=np.float32))
    out.save(clip / "clip_record.npz")
    return clip


def _correction(o0: np.ndarray) -> np.ndarray:
    """7 deg about the vertical through the box at row 0 (`o0`), then 1 cm in +y (world)."""
    w = np.eye(4)
    w[:3, :3] = Rotation.from_euler("z", 7.0, degrees=True).as_matrix()
    c = o0[:3, 3]
    w[:3, 3] = c - w[:3, :3] @ c + np.array([0.0, 0.01, 0.0])
    return w


def _tilted_gravity(rec: ClipRecord, deg: float = 6.0) -> np.ndarray:
    g = rec.gravity_cam.astype(np.float64)
    return Rotation.from_euler("x", deg, degrees=True).as_matrix() @ g


def write_start_pose(path: Path, window: str, pose_world: np.ndarray, c0_frame: int = 0, /, **edits: object) -> Path:
    rec = {"version": 1, "kind": "start_pose_a1", "window": window, "amendment": "rest check on the decoded rows",
           "p7_ruled_pass": True,
           "start_pose_source": "fallback", "pose": "consolidated", "pose_world": np.asarray(pose_world).tolist(),
           "c0_frame_idx": c0_frame, "world": {"camera": "static", "ego_motion_sha256": None},
           "support_plane": {"status": "no plane", "rows": None, "sha256": None}, "snap": None,
           "caps": {"snap_max_m": 0.02, "snap_max_deg": 15.0}, "refused": False, "refusal": None,
           "start_vs_c0": {"disp_m": 0.01, "rot_deg": 7.0}}
    rec.update(edits)
    path.write_text(json.dumps(rec))
    return path


def write_plane(path: Path, window: str, gravity_cam: np.ndarray, rows: tuple[int, ...] = (10, 15),
                sd: float = 0.5, accepted: bool = True) -> Path:
    path.write_text(json.dumps({"version": 1, "window": window, "rows": list(rows), "gravity_cam": list(gravity_cam),
                                "normal_cam0": list(gravity_cam), "offset_cam0": 0.5, "normal_sd_deg": sd,
                                "accepted": accepted, "refusal": [] if accepted else ["inliers"],
                                "camera": "static_assumed"}))
    return path


def snapped_on(plane: Path) -> dict:
    """The A1 fields of a pose snapped on `plane` (the snap record is reported, not re-checked, by the robot path)."""
    from fastdo.robot.start_pose import file_sha256
    return {"pose": "snapped", "support_plane": {"status": "accepted", "rows": 2, "sha256": file_sha256(plane)},
            "snap": {"face_index": 0, "face_probability": 0.4, "rot_deg": 7.0, "disp_m": 0.01, "refused": False}}


def _traj(pkg: Path) -> dict:
    with np.load(pkg / "trajectory.npz") as z:
        return {k: np.asarray(z[k]) for k in z.files}


def test_sidecar_pose_and_gravity_reach_the_packaged_trajectory(g1, tmp_path: Path) -> None:
    """The sidecars give the package of the equivalently moved clip (module docstring), and differ from the plain
    export (mutations: the correction not applied; the plane's gravity ignored)."""
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "a" / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    w = _correction(o0)
    g = _tilted_gravity(rec)
    pl = write_plane(tmp_path / "plane.json", "win", g)
    sp = write_start_pose(tmp_path / "sp.json", "win", w @ o0, **snapped_on(pl))
    assert cli.main([str(clip), "--out", str(tmp_path / "with"), *EXPLICIT, "--start-pose", str(sp),
                     "--support-plane", str(pl)]) == 0
    moved = _carried_clip(g1, tmp_path / "b" / "win", w=w, gravity_cam=g)
    assert cli.main([str(moved), "--out", str(tmp_path / "moved"), *EXPLICIT]) == 0
    assert cli.main([str(clip), "--out", str(tmp_path / "plain"), *EXPLICIT]) == 0
    a, b, plain = (_traj(tmp_path / n) for n in ("with", "moved", "plain"))
    for key in ("arm_q", "hand_q", "to_hold_path", "go_home_path"):
        assert a[key].shape == b[key].shape and np.allclose(a[key], b[key], atol=1e-4), key
    assert a["arm_q"].shape != plain["arm_q"].shape or not np.allclose(a["arm_q"], plain["arm_q"], atol=1e-3)
    ma, mb = (json.loads((tmp_path / n / "package.json").read_text()) for n in ("with", "moved"))
    assert np.allclose(ma["T_robot_from_video_world"], mb["T_robot_from_video_world"], atol=1e-6)
    assert np.allclose(ma["object_start_robot"]["position_m"], mb["object_start_robot"]["position_m"], atol=1e-6)
    T = np.asarray(ma["T_robot_from_video_world"])
    up = rec.cam_to_world[0].astype(np.float64)[:3, :3] @ g
    assert np.allclose(T[:3, :3] @ up, [0.0, 0.0, 1.0], atol=1e-9)                     # the plane's up is robot +z
    assert ma["gravity_source"] == "support_plane" and ma["gravity"]["source"] == "support_plane"
    assert ma["start_pose"]["kind"] == "snapped" and np.allclose(ma["start_pose"]["correction_world"], w,
                                                                      atol=1e-6)
    assert ma["start_pose"]["correction_rot_deg"] == pytest.approx(7.0, abs=1e-4)
    assert ma["start_pose"]["correction_shift_m"] == pytest.approx(0.01, abs=1e-6)      # rotation about the box
    readme = (tmp_path / "with" / "README.md").read_text()
    assert "## Start pose and gravity" in readme and "ESTIMATED support plane" in readme and "SNAPPED" in readme
    assert "--start-pose" in ma["reexport_command"] and "--support-plane" in ma["reexport_command"]


def test_an_accepted_plane_switches_the_pitch_search_off(g1, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the CLI with a searched placement: `search_placement` gets the sidecar pose at the anchor, the plane's up and
    allow_pitch False; without the plane (qc gravity_source "default") it gets the record's up and allow_pitch True
    (mutation: allow_pitch left on the record's gravity source)."""
    from fastdo.robot import kinematic_path as kp
    from fastdo.robot.reach import PlacementChoice, SearchResult
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    (clip / "qc.json").write_text(json.dumps({"gravity_source": "default", "camera_source": "gravity_aligned_static"}))
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    pose = _correction(o0) @ o0
    g = _tilted_gravity(rec)
    seen: list[dict] = []

    def spy(task, cfg, tx, ty, pool=None, allow_pitch=True):
        seen.append({"up": task.up_record.copy(), "pose": task.object_world[task.anchor].copy(),
                     "allow_pitch": allow_pitch})
        ch = PlacementChoice(table_xy=synthetic.OBJECT_XY, table_z=synthetic.TABLE_Z)
        return SearchResult(choice=ch, score=0.0, records=[], final={})

    monkeypatch.setattr(kp, "search_placement", spy)
    pl = write_plane(tmp_path / "plane.json", "win", g)
    snapped = write_start_pose(tmp_path / "snapped.json", "win", pose, **snapped_on(pl))
    consolidated = write_start_pose(tmp_path / "consolidated.json", "win", pose)
    c2w0 = rec.cam_to_world[0].astype(np.float64)
    assert cli.main([str(clip), "--out", str(tmp_path / "p1"), "--start-pose", str(snapped), "--support-plane",
                     str(pl), "--workers", "0"]) == 0
    assert cli.main([str(clip), "--out", str(tmp_path / "p2"), "--start-pose", str(consolidated), "--workers",
                     "0"]) == 0
    with_plane, without = seen
    assert with_plane["allow_pitch"] is False and np.allclose(with_plane["up"], c2w0[:3, :3] @ g, atol=1e-9)
    assert np.allclose(with_plane["pose"], pose, atol=1e-6) and np.allclose(without["pose"], pose, atol=1e-6)
    assert without["allow_pitch"] is True
    assert np.allclose(without["up"], c2w0[:3, :3] @ rec.gravity_cam.astype(np.float64), atol=1e-6)
    meta = json.loads((tmp_path / "p1" / "package.json").read_text())
    assert meta["up_source"] == "support_plane"
    assert not any("GRAVITY NOT ESTIMATED" in w or "up from an ASSUMED" in w for w in meta["warnings"])
    meta2 = json.loads((tmp_path / "p2" / "package.json").read_text())
    assert any("GRAVITY NOT ESTIMATED" in w for w in meta2["warnings"])          # control: no plane, the warning stays


@pytest.mark.parametrize(("case", "needle"), [
    ("refused", "screen refused"),
    ("wrong_frame", "source frame"),
    ("snapped_without_plane", "snapped on support plane"),
    ("ego_world", "ego-motion"),
    ("other_window", "not of win"),
    ("plane_with_pitch", "assumption"),
    ("consolidated_with_accepted_plane", "not snapped"),
    ("snapped_on_another_plane", "was snapped on support plane"),
    ("not_rigid", "rigid"),
])
def test_bad_sidecars_are_refused(g1, tmp_path: Path, caplog: pytest.LogCaptureFixture, case: str,
                                  needle: str) -> None:
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    pl = write_plane(tmp_path / "plane.json", "win", _tilted_gravity(rec))
    edits: dict = {"refused": {"refused": True, "refusal": "snap rotation > 15 deg"}, "wrong_frame": {"c0_frame_idx": 3},
                   "snapped_without_plane": {"pose": "snapped", "support_plane": {"status": "accepted",
                                                                                  "sha256": "0" * 64}},
                   "ego_world": {"world": {"camera": "ego_motion", "ego_motion_sha256": "0" * 64}},
                   "other_window": {"window": "other"}, "plane_with_pitch": snapped_on(pl),
                   "consolidated_with_accepted_plane": {},
                   "snapped_on_another_plane": {**snapped_on(pl), "support_plane": {"status": "accepted", "rows": 2,
                                                                                   "sha256": "f" * 64}},
                   "not_rigid": {"pose_world": (2.0 * np.eye(4)).tolist()}}[case]
    sp = write_start_pose(tmp_path / "sp.json", "win", o0, **edits)
    extra = {"plane_with_pitch": ["--camera-pitch-deg", "30", "--support-plane", str(pl)],
             "consolidated_with_accepted_plane": ["--support-plane", str(pl)],
             "snapped_on_another_plane": ["--support-plane", str(pl)]}.get(case, [])
    caplog.set_level("ERROR")
    assert cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--start-pose", str(sp), *extra]) == 2
    assert needle in caplog.text, caplog.text
    assert not (tmp_path / "pkg").exists()


def test_a_single_row_plane_is_refused_and_keeps_the_record_gravity(g1, tmp_path: Path) -> None:
    """Row minimum: a plane fitted on one rest row (or with normals 3+ deg apart) is reported and REFUSED; up stays the
    record's and the package says why (mutation: the row minimum not enforced by the robot path)."""
    from fastdo.robot.start_pose import load_support_plane
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    g = _tilted_gravity(rec)
    one = load_support_plane(write_plane(tmp_path / "one.json", "win", g, rows=(10,), sd=0.0))
    assert not one.accepted and any("plane rows: 1 rest row" in r for r in one.refusal)
    wide = load_support_plane(write_plane(tmp_path / "wide.json", "win", g, rows=(10, 15), sd=3.2))
    assert not wide.accepted and any("sd 3.2" in r for r in wide.refusal)
    assert load_support_plane(write_plane(tmp_path / "ok.json", "win", g, rows=(10, 15), sd=3.0)).accepted
    assert cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--support-plane",
                     str(tmp_path / "one.json")]) == 0
    meta = json.loads((tmp_path / "pkg" / "package.json").read_text())
    assert meta["gravity"]["source"].startswith("record") and meta["gravity_source"] != "support_plane"
    assert any("support plane REFUSED" in w and "plane rows: 1 rest row" in w for w in meta["warnings"])
    T = np.asarray(meta["T_robot_from_video_world"])
    up = rec.cam_to_world[0].astype(np.float64)[:3, :3] @ rec.gravity_cam.astype(np.float64)
    assert np.allclose(T[:3, :3] @ up, [0.0, 0.0, 1.0], atol=1e-6)


# --- onset > 0, pre-contact drift, a moving camera ---------------------

CONTACT_ROW, DRIFT = 12, 0.15


def moving_camera_clip(g1, root: Path, w: np.ndarray | None = None, gravity_cam: np.ndarray | None = None) -> Path:
    """The synthetic clip with the box touched from row `CONTACT_ROW` on (drifting `DRIFT` in +x before it) and a
    camera that pans, tilts and translates on every row (the world is unchanged); optionally every row moved by the
    world transform `w` and `gravity_cam` replaced (after a reviewer's probe)."""
    clip = synthetic.write_clip(g1, root, synthetic.contact_object_path(g1, CONTACT_ROW, DRIFT))
    rec = ClipRecord.load(clip / "clip_record.npz")
    c2w_old = rec.cam_to_world.astype(np.float64)
    n = len(c2w_old)
    c2w = c2w_old.copy()
    for t in range(n):
        r = Rotation.from_euler("zy", [0.4 * t / n, 0.15 * np.sin(t)]).as_matrix()
        c2w[t, :3, :3] = r @ c2w_old[t, :3, :3]
        c2w[t, :3, 3] = c2w_old[t, :3, 3] + np.array([0.0, 0.01 * t, 0.002 * t])
    obj_w = np.einsum("tij,tjk->tik", c2w_old, rec.object_pose.astype(np.float64))
    hr = rec.hand_right
    jw = np.einsum("tij,tvj->tvi", c2w_old[:, :3, :3], hr.joints.astype(np.float64)) + c2w_old[:, None, :3, 3]
    if w is not None:
        obj_w = np.einsum("ij,tjk->tik", w, obj_w)
        jw = jw @ w[:3, :3].T + w[:3, 3]
    obj_c = np.einsum("tij,tjk->tik", np.linalg.inv(c2w), obj_w)
    jc = np.einsum("tji,tvj->tvi", c2w[:, :3, :3], jw - c2w[:, None, :3, 3])
    g = c2w[0, :3, :3].T @ np.array([0.0, 0.0, 1.0]) if gravity_cam is None else np.asarray(gravity_cam)
    out = replace(rec, cam_to_world=c2w.astype(np.float32), object_pose=obj_c.astype(np.float32),
                  hand_right=replace(hr, joints=jc.astype(np.float32), trans=jc[:, 0].astype(np.float32)),
                  gravity_cam=np.asarray(g, dtype=np.float32))
    out.save(clip / "clip_record.npz")
    return clip


def _onset(g1, clip: Path) -> tuple[int, np.ndarray, int]:
    """`(source frame, world object pose, task row)` of the export's contact-onset anchor (no sidecar)."""
    from fastdo.robot.kinematic_task import KinematicPathConfig, prepare_task
    from fastdo.robot.world_frame import load_reconstruction
    recon = load_reconstruction(clip, None)
    prep = prepare_task(recon, KinematicPathConfig(), g1)
    a = prep.task.anchor
    return int(recon.record.frame_idx[prep.source_rows[a]]), prep.task.object_world[a].copy(), a


def _tilted_w(o_c0: np.ndarray) -> np.ndarray:
    """12 deg about a tilted axis through the onset pose's origin, then (2, -2, 1) cm: 3.0 cm / 12 deg (the correction
    bound is 8.6 cm / 12.2 deg)."""
    w = np.eye(4)
    axis = np.array([0.3, 0.2, 0.93])
    w[:3, :3] = Rotation.from_rotvec(np.radians(12.0) * axis / np.linalg.norm(axis)).as_matrix()
    c = o_c0[:3, 3]
    w[:3, 3] = c - w[:3, :3] @ c + np.array([0.02, -0.02, 0.01])
    return w


def test_known_answer_with_a_later_onset_drift_and_a_moving_camera(g1, tmp_path: Path) -> None:
    """Onset at row 12 (anchor > 0), 15 cm of pre-contact drift, a per-row moving camera: the sidecars give the task
    and the package of the clip moved by W on every row with the plane's gravity_cam, exported without sidecars.
    Kills: RV1 (W from row 0 instead of the anchor: the anchor pose is not P), RV3 (gravity rotated
    by the last camera: the plane's up is not robot +z), RV4 (hand joints not moved by W: joints_world and the
    table check differ)."""
    from fastdo.robot.kinematic_task import KinematicPathConfig, prepare_task
    from fastdo.robot.start_pose import load_start_pose, load_support_plane
    from fastdo.robot.world_frame import load_reconstruction
    cli = _load()
    base = moving_camera_clip(g1, tmp_path / "a" / "win")
    c0f, o_c0, anchor = _onset(g1, base)
    assert anchor > 0 and c0f > 0
    rec = ClipRecord.load(base / "clip_record.npz")
    assert np.abs(rec.cam_to_world[-1] - rec.cam_to_world[0]).max() > 0.1           # the camera moves
    obj0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    assert np.linalg.norm(obj0[:3, 3] - o_c0[:3, 3]) > 0.1                          # row 0 is not the onset pose
    w = _tilted_w(o_c0)
    g = Rotation.from_euler("xy", [9.0, -4.0], degrees=True).as_matrix() @ rec.gravity_cam.astype(np.float64)
    pl = write_plane(tmp_path / "plane.json", "win", g)
    sp = write_start_pose(tmp_path / "sp.json", "win", w @ o_c0, c0f, **snapped_on(pl),
                          start_vs_c0={"disp_m": 0.03, "rot_deg": 12.0})
    moved = moving_camera_clip(g1, tmp_path / "b" / "win", w=w, gravity_cam=g)

    pa = prepare_task(load_reconstruction(base, None), KinematicPathConfig(start_pose=load_start_pose(sp),
                                                                           support_plane=load_support_plane(pl)), g1)
    pb = prepare_task(load_reconstruction(moved, None), KinematicPathConfig(), g1)
    ta, tb = pa.task, pb.task
    assert ta.anchor == tb.anchor == anchor and np.array_equal(ta.rows, tb.rows)
    assert np.abs(ta.object_world[anchor] - w @ o_c0).max() < 1e-9                  # the anchor pose IS P (RV1)
    # W moves the task rows from the onset on; before it the wrist keeps its pose relative to the per-row object
    # (`fastdo.robot.start_pose`), so there only the packaged trajectory is compared (below); hand_q and up: all
    for f, rows in (("wrist_world", slice(anchor, None)), ("object_world", slice(anchor, None)),
                    ("joints_world", slice(anchor, None)), ("hand_q", slice(None)), ("up_record", ...)):
        x, y = np.asarray(getattr(ta, f))[rows], np.asarray(getattr(tb, f))[rows]       # joints_world: RV4
        assert x.shape == y.shape and np.nanmax(np.abs(x - y)) < 1e-4, f

    assert cli.main([str(base), "--out", str(tmp_path / "with"), *EXPLICIT, "--start-pose", str(sp),
                     "--support-plane", str(pl)]) == 0
    assert cli.main([str(moved), "--out", str(tmp_path / "moved"), *EXPLICIT]) == 0
    a, b = _traj(tmp_path / "with"), _traj(tmp_path / "moved")
    for key in ("arm_q", "hand_q", "to_hold_path", "go_home_path"):
        assert a[key].shape == b[key].shape and np.allclose(a[key], b[key], atol=1e-4), key
    ma, mb = (json.loads((tmp_path / n / "package.json").read_text()) for n in ("with", "moved"))
    sa, sb = (json.loads((tmp_path / n / "sim_stats.json").read_text()) for n in ("with", "moved"))
    assert np.allclose(ma["T_robot_from_video_world"], mb["T_robot_from_video_world"], atol=1e-6)
    assert np.allclose(ma["object_start_robot"]["position_m"], mb["object_start_robot"]["position_m"], atol=1e-6)
    T = np.asarray(ma["T_robot_from_video_world"])
    up = rec.cam_to_world[0].astype(np.float64)[:3, :3] @ (g / np.linalg.norm(g))
    assert np.allclose(T[:3, :3] @ up, [0.0, 0.0, 1.0], atol=1e-9)                  # row 0's camera (RV3)
    tca, tcb = sa["table_check"], sb["table_check"]                                   # hand moved with W (RV4)
    for k in ("hand_below_table_m", "object_below_table_m", "table_bad_rows", "rest_com_inside_m"):
        assert tca[k] == pytest.approx(tcb[k], abs=1e-6), k
    assert ma["start_pose"]["correction_rot_deg"] == pytest.approx(12.0, abs=1e-6)
    assert ma["start_pose"]["correction_shift_m"] == pytest.approx(0.03, abs=1e-6)


def _ego_motion_file(clip: Path, path: Path) -> Path:
    """An accepted ego-motion of the clip's rows equal to its static camera (scale 1)."""
    rec = ClipRecord.load(clip / "clip_record.npz")
    np.savez(path, cam_to_world=rec.cam_to_world.astype(np.float64), frame_idx=rec.frame_idx,
             valid=np.ones(len(rec.frame_idx), dtype=bool), reference_row=np.asarray(0),
             summary=np.asarray(json.dumps({"ok": True, "cam_path_m": 0.0, "cam_rot_extent_deg": 0.0})))
    return path


def test_a_static_sidecar_on_an_ego_motion_export_is_refused(g1, tmp_path: Path,
                                                             caplog: pytest.LogCaptureFixture) -> None:
    """RV2: a start pose in the static camera's world, exported with `--ego-motion`, exits 2; the same sidecar
    without `--ego-motion` exports (control)."""
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    sp = write_start_pose(tmp_path / "sp.json", "win", o0)
    ego = _ego_motion_file(clip, tmp_path / "ego_motion.npz")
    caplog.set_level("ERROR")
    assert cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--start-pose", str(sp),
                     "--ego-motion", str(ego)]) == 2
    assert "static camera's world" in caplog.text and not (tmp_path / "pkg").exists()


def test_a_snapped_sidecar_on_a_refused_plane_is_refused(g1, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """RV5: a pose snapped on a plane the robot path refuses (one rest row: R4) exits 2, even though the plane
    file is the one it was snapped on (sha256 matches)."""
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    pl = write_plane(tmp_path / "plane.json", "win", _tilted_gravity(rec), rows=(10,), sd=0.0)
    sp = write_start_pose(tmp_path / "sp.json", "win", o0, **snapped_on(pl))
    caplog.set_level("ERROR")
    assert cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--start-pose", str(sp),
                     "--support-plane", str(pl)]) == 2
    assert "a plane the robot path refuses" in caplog.text and "plane rows: 1 rest row" in caplog.text


# --- the record's ruled rest verdict and the correction bound ----------------------------------

@pytest.mark.parametrize(("edits", "needle"), [
    ({"p7_ruled_pass": False}, "p7_ruled_pass is False"),          # the 76 stored sidecars
    ({"p7_ruled_pass": None}, "p7_ruled_pass is None"),
    ({"p7_ruled_pass": "true"}, "p7_ruled_pass is 'true'"),
    ({"start_vs_c0": {"disp_m": 0.691, "rot_deg": 28.2}}, "69.1 cm / 28.2 deg"),   # a stored failing case (0b215326:k000)
    ({"start_vs_c0": {"disp_m": 0.0861, "rot_deg": 1.0}}, "8.6 cm"),
    ({"start_vs_c0": {"disp_m": 0.01, "rot_deg": 12.21}}, "12.2 deg"),
    ({"start_vs_c0": None}, "nan cm"),
    ({"window": None}, "names no window"),
])
def test_a_record_failing_its_rest_verdict_is_refused(g1, tmp_path: Path, caplog: pytest.LogCaptureFixture, edits: dict,
                                               needle: str) -> None:
    """The record's ruled verdict must be True and its own start-vs-onset distance within 8.6 cm / 12.2 deg; the
    refusal names the value (exit 2, nothing written). Control: the same record passing exports
    (`test_sidecar_pose_and_gravity_reach_the_packaged_trajectory`)."""
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    sp = write_start_pose(tmp_path / "sp.json", "win", o0, **edits)
    caplog.set_level("ERROR")
    assert cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--start-pose", str(sp)]) == 2
    assert needle in caplog.text, caplog.text
    assert not (tmp_path / "pkg").exists()


def test_a_record_without_the_verdict_field_is_refused(tmp_path: Path) -> None:
    """A record that has NO `p7_ruled_pass` key (an older writer's records) is refused like an explicit None; the
    same record with the field True loads (control)."""
    from fastdo.robot.start_pose import SidecarError, load_start_pose
    path = write_start_pose(tmp_path / "sp.json", "win", np.eye(4))
    assert load_start_pose(path).p7_ruled_pass is True
    rec = json.loads(path.read_text())
    del rec["p7_ruled_pass"]
    path.write_text(json.dumps(rec))
    with pytest.raises(SidecarError, match="p7_ruled_pass"):
        load_start_pose(path)


def test_the_bound_is_inclusive_and_the_verdict_is_read_from_the_record(tmp_path: Path) -> None:
    from fastdo.robot.start_pose import MAX_CORRECTION_DEG, MAX_CORRECTION_M, load_start_pose
    assert (MAX_CORRECTION_M, MAX_CORRECTION_DEG) == (0.086, 12.2)
    reading = "rest check on the decoded rows, with the lead-motion test"
    sp = load_start_pose(write_start_pose(tmp_path / "sp.json", "win", np.eye(4),
                                          start_vs_c0={"disp_m": 0.086, "rot_deg": 12.2}, amendment=reading))
    assert sp.p7_ruled_pass is True and sp.to_dict()["amendment"] == reading


@pytest.mark.parametrize(("deg", "shift", "needle"), [(13.0, 0.0, "13.0 deg"), (0.0, 0.09, "9.0 cm")])
def test_the_export_bounds_its_own_correction(g1, tmp_path: Path, caplog: pytest.LogCaptureFixture, deg: float,
                                              shift: float, needle: str) -> None:
    """A sidecar pose 13 deg (or 9 cm) from the export's onset pose exits 2 although the record's own start_vs_c0
    claims 1 cm / 7 deg (mutation: only the record's field checked)."""
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    w = _correction(o0)
    w[:3, :3] = Rotation.from_euler("z", deg, degrees=True).as_matrix()
    w[:3, 3] = o0[:3, 3] - w[:3, :3] @ o0[:3, 3] + np.array([shift, 0.0, 0.0])
    sp = write_start_pose(tmp_path / "sp.json", "win", w @ o0)
    caplog.set_level("ERROR")
    assert cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--start-pose", str(sp)]) == 2
    assert "correction W" in caplog.text and needle in caplog.text, caplog.text


# --- window, plane camera, plane without a start pose, the exception class ---------

@pytest.mark.parametrize(("case", "needle"), [
    ("plane_without_window", "support plane of window None"),
    ("plane_in_the_ego_world", "camera world 'ego_motion'"),
    ("plane_without_camera", "camera world None"),
    ("accepted_plane_alone", "no start pose was given"),
])
def test_plane_sidecar_gaps_are_refused(g1, tmp_path: Path, caplog: pytest.LogCaptureFixture, case: str,
                                        needle: str) -> None:
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    pl = write_plane(tmp_path / "plane.json", "win", _tilted_gravity(rec))
    doc = json.loads(pl.read_text())
    if case == "plane_without_window":
        doc.pop("window")
    elif case == "plane_in_the_ego_world":
        doc["camera"] = "ego_motion"
    elif case == "plane_without_camera":
        doc.pop("camera")
    pl.write_text(json.dumps(doc))
    args = [str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--support-plane", str(pl)]
    if case != "accepted_plane_alone":
        args += ["--start-pose", str(write_start_pose(tmp_path / "sp.json", "win", o0, **snapped_on(pl)))]
    caplog.set_level("ERROR")
    assert cli.main(args) == 2
    assert needle in caplog.text, caplog.text


def test_only_a_sidecar_check_becomes_exit_2(g1, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any other ValueError in the kinematic path propagates (it is not labelled a sidecar refusal), also when a
    sidecar is given."""
    cli = _load()
    clip = _carried_clip(g1, tmp_path / "win")
    rec = ClipRecord.load(clip / "clip_record.npz")
    o0 = rec.cam_to_world[0].astype(np.float64) @ rec.object_pose[0].astype(np.float64)
    sp = write_start_pose(tmp_path / "sp.json", "win", o0)

    def boom(*a, **k):
        raise ValueError("not a sidecar problem")

    monkeypatch.setattr(cli, "run_kinematic_path", boom)
    with pytest.raises(ValueError, match="not a sidecar problem"):
        cli.main([str(clip), "--out", str(tmp_path / "pkg"), *EXPLICIT, "--start-pose", str(sp)])
