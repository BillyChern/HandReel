"""Per-object Sharpa scenes through the baseline's scene stages (`run/pipeline/build_ycb_scenes.py`).

The builder runs the baseline's `decompose_mesh`, `generate_scene`, `solve_ik` and `resolve_pedestal` inside the
retargeting conda env (imported at runtime from third_party, never copied) and validates the scene in this environment:
`RobotScene.from_xml` layout nq = 35, nv = 34, one free joint (the object) and object collision geoms in contact pairs
with the hand. A synthetic box runs through the whole chain; the real CoACD run on one YCB object is marked slow and
timing, so the merge gate (`-m "not timing"`) skips it. `validate_scene` is also checked directly: on the shipped
whisking scene, on a copy with the object's contact pairs removed, and on models without exactly one free joint.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from fastdo.policy.dexycb_inputs import read_scene_record, scene_record_path
from fastdo.policy.scene import load_policy_scene
from run.pipeline import build_ycb_scenes
from run.pipeline.build_ycb_scenes import (
    CoacdParams,
    SceneBuildConfig,
    build_object_scene,
    validate_scene,
)
from tests.policy.dexycb_data import (
    BASELINE_ROOT,
    DEXYCB_ROOT,
    RETARGETING_PYTHON,
    WHISKING_SCENE_XML,
    dexycb_missing,
    retargeting_env_missing,
    whisking_scene_missing,
)

logger = logging.getLogger(__name__)

STAGES = ["decompose_mesh", "generate_scene", "solve_ik", "resolve_pedestal"]


def _cfg(tmp_path: Path) -> SceneBuildConfig:
    assert RETARGETING_PYTHON is not None and BASELINE_ROOT is not None
    return SceneBuildConfig(out_root=str(tmp_path / "scenes"), baseline_root=str(BASELINE_ROOT),
                            python=str(RETARGETING_PYTHON))


def _vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    return np.asarray(mesh.vertices)


def _collision_bounds(scene_xml: str) -> np.ndarray:
    """`(2,3)` bounds of the object's convex parts (the files the scene's object collision meshes load)."""
    parts = sorted((Path(scene_xml).parents[4] / "assets" / "objects").glob("*/convex/*.obj"))
    verts = np.concatenate([_vertices(p) for p in parts])
    return np.stack([verts.min(axis=0), verts.max(axis=0)])


@whisking_scene_missing
def test_validate_scene_checks_the_layout_and_the_object_contacts(tmp_path: Path) -> None:
    """The whisking scene passes (nq 35, nv 34, one free joint, every object collision geom paired with the hand);
    the same scene without its object contact pairs fails with that reason."""
    assert WHISKING_SCENE_XML is not None
    ok = validate_scene(WHISKING_SCENE_XML)
    assert ok["pass"] and (ok["nq"], ok["nv"], ok["n_free_joints"]) == (35, 34, 1) and not ok["reasons"]
    assert ok["n_object_contact_geoms"] == ok["n_object_collision_geoms"] > 0
    text = WHISKING_SCENE_XML.read_text()
    assets = (WHISKING_SCENE_XML.parent / "../../../../assets").resolve()
    stripped = re.sub(r'meshdir="[^"]*"', f'meshdir="{assets}/"', text, count=1)
    stripped = re.sub(r"<pair [^>]*right_object_\d+[^>]*/>", "", stripped)
    path = tmp_path / "scene_no_object_pairs.xml"
    path.write_text(stripped)
    bad = validate_scene(path)
    assert not bad["pass"] and bad["n_object_contact_geoms"] == 0 and (bad["nq"], bad["nv"]) == (35, 34)
    assert any("object contact" in r for r in bad["reasons"])


@pytest.mark.parametrize("n_free", [0, 2])
def test_validate_scene_records_a_scene_without_exactly_one_free_joint(n_free: int,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """A model with no free joint, or two, fails validation with that reason instead of raising IndexError."""
    bodies = "".join(f'<body pos="{i} 0 0"><freejoint/><geom size="0.01"/></body>' for i in range(n_free))
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="hinge"/><geom size="0.01"/></body>'
                                           f"{bodies}</worldbody></mujoco>")
    stub = SimpleNamespace(model=model, hand_collision_geoms=list)
    monkeypatch.setattr(build_ycb_scenes, "RobotScene", SimpleNamespace(from_xml=lambda path: stub))
    result = validate_scene("stub.xml")
    assert not result["pass"] and result["n_free_joints"] == n_free
    assert any("exactly one free joint" in reason for reason in result["reasons"]), result["reasons"]


@retargeting_env_missing
def test_synthetic_box_goes_through_the_scene_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An 8 x 6 x 4 cm box through all four baseline stages: the stored record passes the layout checks, names the
    stages, records the CoACD parameters and the part count (the convex files on disk), and loads as a policy scene
    (28 actuated robot joints). The per-object scene carries no pedestal, and the convex parts cover the box within
    the thicken + dilate margins. A second build reuses the stored scene without running the stages again."""
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.08, 0.06, 0.04)).export(mesh_path)
    cfg = _cfg(tmp_path)
    record = build_object_scene("box", mesh_path, "right", cfg)
    logger.info("box scene: %d parts, stage seconds %s", record["n_parts"], record["stage_seconds"])
    assert record["pass"] and record["stages"] == STAGES and record["object"] == "box" and record["side"] == "right"
    assert (record["validation"]["nq"], record["validation"]["nv"], record["validation"]["n_free_joints"]) == (35, 34, 1)
    assert record["coacd"] == {"method": "coacd", "threshold": 0.05, "max_convex_hull": 32, "max_ch_vertex": 256,
                               "preprocess_resolution": 100, "thicken": 0.002, "dilate": 0.002}
    parts = sorted((Path(cfg.out_root) / "box" / "assets" / "objects" / "box" / "convex").glob("*.obj"))
    assert record["n_parts"] == len(parts) >= 1
    assert record["use_pedestal"] is False and "pedestal" not in Path(record["scene_xml"]).read_text()
    assert read_scene_record(cfg.out_root, "box", "right") == record
    assert json.loads(scene_record_path(cfg.out_root, "box", "right").read_text()) == record
    scene = load_policy_scene(record["scene_xml"], 50.0, 10, 300.0, 35.0, 1000.0, 100.0)
    assert scene.nq_robot == 28
    bounds = _collision_bounds(record["scene_xml"])
    half = np.array([0.04, 0.03, 0.02])
    assert np.all(np.abs(bounds[0] + half) <= 0.005) and np.all(np.abs(bounds[1] - half) <= 0.005)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the stored scene should be reused")

    monkeypatch.setattr(subprocess, "run", refuse)
    assert build_object_scene("box", mesh_path, "right", cfg) == record


def test_scene_build_config_defaults_are_the_launch_parameters() -> None:
    """The CoACD parameters are launch.py's stage-2 call (thicken and dilate 2 mm) on decompose_mesh's defaults."""
    assert CoacdParams() == CoacdParams(threshold=0.05, max_convex_hull=32, max_ch_vertex=256, preprocess_resolution=100,
                                        thicken=0.002, dilate=0.002)
    cfg = SceneBuildConfig(out_root="o", baseline_root="b", python="p")
    assert (cfg.friction_scale, cfg.add_ur3_arm, cfg.force) == (1.5, True, False)
    with pytest.raises(ValueError, match="side"):
        build_object_scene("box", "box.obj", "both", cfg)


@pytest.mark.slow
@pytest.mark.timing
@retargeting_env_missing
@dexycb_missing
def test_real_coacd_scene_for_one_ycb_object(tmp_path: Path) -> None:
    """The real CoACD decomposition of one YCB object (061_foam_brick) through the builder: the scene passes and the
    convex parts cover the object's mesh within the thicken + dilate margins."""
    assert DEXYCB_ROOT is not None
    name = "061_foam_brick"
    mesh_path = DEXYCB_ROOT / "models" / name / "textured_simple.obj"
    record = build_object_scene(name, mesh_path, "right", _cfg(tmp_path))
    logger.info("%s: %d CoACD parts, stage seconds %s", name, record["n_parts"], record["stage_seconds"])
    assert record["pass"] and record["n_parts"] >= 1 and record["mesh"]["watertight"]
    verts = _vertices(mesh_path)
    bounds = _collision_bounds(record["scene_xml"])
    assert np.all(np.abs(bounds[0] - verts.min(axis=0)) <= 0.006)
    assert np.all(np.abs(bounds[1] - verts.max(axis=0)) <= 0.006)


def _default_cpu() -> int:
    """The CPU an unrestricted build of this process is pinned to: `sorted(mask)[pid % len(mask)]`."""
    allowed = sorted(os.sched_getaffinity(0))
    return allowed[os.getpid() % len(allowed)]


def test_scene_driver_runs_under_a_one_cpu_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    """oneTBB inside CoACD sizes its pool from the affinity mask, so the driver is launched under
    `taskset -c <cpu>`: by default `sorted(mask)[pid % len(mask)]` (concurrent builder processes spread over the
    mask), or `cpu`; `pin=False` leaves the command as it was, and a CPU outside the mask is refused. A child
    launched this way sees exactly that one CPU."""
    allowed = os.sched_getaffinity(0)
    cfg = SceneBuildConfig(out_root="o", baseline_root="b", python="p")
    assert (cfg.pin, cfg.cpu) == (True, None)
    assert build_ycb_scenes._driver_cpu(cfg) == _default_cpu()
    for pid in (0, 1, len(allowed), 7 * len(allowed) + 3):
        monkeypatch.setattr(os, "getpid", lambda pid=pid: pid)
        assert build_ycb_scenes._driver_cpu(cfg) == sorted(allowed)[pid % len(allowed)]
    monkeypatch.undo()
    cmd = ["p", "driver.py", "--x"]
    pinned = build_ycb_scenes._pinned(cmd, build_ycb_scenes._driver_cpu(cfg))
    assert Path(pinned[0]).name == "taskset" and pinned[1:] == ["-c", str(_default_cpu()), *cmd]
    assert build_ycb_scenes._driver_cpu(dataclasses.replace(cfg, pin=False)) is None
    assert build_ycb_scenes._pinned(cmd, None) == cmd
    assert build_ycb_scenes._driver_cpu(dataclasses.replace(cfg, cpu=max(allowed))) == max(allowed)
    with pytest.raises(ValueError, match="affinity mask"):
        build_ycb_scenes._driver_cpu(dataclasses.replace(cfg, cpu=max(allowed) + 1))
    probe = [sys.executable, "-c", "import os; print(sorted(os.sched_getaffinity(0)))"]
    for cpu in sorted(allowed)[:1] + sorted(allowed)[-1:]:
        out = subprocess.run(build_ycb_scenes._pinned(probe, cpu), capture_output=True, text=True, check=True).stdout
        assert out.strip() == f"[{cpu}]"
    node = {"out_root": "o", "baseline_root": "b", "python": "p", "friction_scale": 1.5, "add_ur3_arm": True,
            "force": False, "timeout_s": 1.0}
    assert (SceneBuildConfig.from_mapping(node).pin, SceneBuildConfig.from_mapping(node).cpu) == (True, None)
    assert SceneBuildConfig.from_mapping({**node, "pin": False, "cpu": 3}).pin is False
    assert SceneBuildConfig.from_mapping({**node, "pin": False, "cpu": 3}).cpu == 3
    assert SceneBuildConfig.from_mapping({**node, "pin": True, "cpu": None}).cpu is None


def test_a_missing_taskset_is_refused_before_any_log_is_opened(monkeypatch: pytest.MonkeyPatch,
                                                               tmp_path: Path) -> None:
    """Without `taskset` on PATH a pinned config is refused when it is built, so no build log is opened and no object
    of the loop starts; an unpinned config still builds."""
    monkeypatch.setattr(build_ycb_scenes.shutil, "which", lambda name: None)
    node = {"out_root": str(tmp_path / "scenes"), "baseline_root": "b", "python": "p", "friction_scale": 1.5,
            "add_ur3_arm": True, "force": False, "timeout_s": 1.0}
    with pytest.raises(FileNotFoundError, match="taskset"):
        SceneBuildConfig.from_mapping(node)
    with pytest.raises(FileNotFoundError, match="taskset"):
        SceneBuildConfig(out_root="o", baseline_root="b", python="p")
    assert SceneBuildConfig.from_mapping({**node, "pin": False}).pin is False
    assert not (tmp_path / "scenes").exists()


def _assert_recorded_affinity(pinned: dict, unpinned: dict) -> None:
    """The pinned driver ran on exactly the chosen CPU (recorded by the driver itself), the unpinned one on the
    inherited mask; the unpinned half is skipped when this process has only one CPU."""
    assert pinned["driver_cpu"] == _default_cpu() and pinned["affinity"] == [_default_cpu()]
    assert unpinned["driver_cpu"] is None
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("the test process has one CPU: an unpinned build cannot be told from a pinned one")
    assert len(unpinned["affinity"]) > 1


def _hull_bytes(cfg: SceneBuildConfig, name: str) -> dict[str, bytes]:
    convex = Path(cfg.out_root) / name / "assets" / "objects" / name / "convex"
    return {p.name: p.read_bytes() for p in sorted(convex.glob("*.obj"))}


@retargeting_env_missing
def test_pinned_scene_build_writes_byte_identical_hulls(tmp_path: Path) -> None:
    """The one-CPU launch changes cost only: the box's CoACD hulls are byte-identical to an unpinned build's. The
    driver's recorded affinity shows the pin reached it: exactly the chosen CPU pinned, the inherited mask unpinned."""
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.08, 0.06, 0.04)).export(mesh_path)
    pinned = _cfg(tmp_path / "pinned")
    unpinned = dataclasses.replace(_cfg(tmp_path / "unpinned"), pin=False)
    pinned_record = build_object_scene("box", mesh_path, "right", pinned)
    unpinned_record = build_object_scene("box", mesh_path, "right", unpinned)
    assert pinned_record["pass"] and unpinned_record["pass"]
    hulls = _hull_bytes(pinned, "box")
    assert hulls and hulls == _hull_bytes(unpinned, "box")
    _assert_recorded_affinity(pinned_record, unpinned_record)


@pytest.mark.slow
@pytest.mark.timing
@retargeting_env_missing
@dexycb_missing
def test_pinned_real_coacd_hulls_are_byte_identical(tmp_path: Path) -> None:
    """The same checks on a real CoACD run (061_foam_brick, the oneTBB preprocess and MCTS search both exercised)."""
    assert DEXYCB_ROOT is not None
    name = "061_foam_brick"
    mesh_path = DEXYCB_ROOT / "models" / name / "textured_simple.obj"
    pinned = _cfg(tmp_path / "pinned")
    unpinned = dataclasses.replace(_cfg(tmp_path / "unpinned"), pin=False)
    pinned_record = build_object_scene(name, mesh_path, "right", pinned)
    unpinned_record = build_object_scene(name, mesh_path, "right", unpinned)
    assert pinned_record["pass"] and unpinned_record["pass"]
    hulls = _hull_bytes(pinned, name)
    assert hulls and hulls == _hull_bytes(unpinned, name)
    _assert_recorded_affinity(pinned_record, unpinned_record)
