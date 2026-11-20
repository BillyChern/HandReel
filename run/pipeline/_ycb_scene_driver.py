"""Build one object's Sharpa scene with the DO AS I DO baseline's scene stages (runs in the retargeting conda env).

It imports the baseline package from `--baseline-root` (third_party/do-as-i-do/retargeting, read-only, never copied) and
nothing from `fastdo`. `run/pipeline/build_ycb_scenes.py` starts it as a CPU subprocess and validates the scene
afterwards in the project environment.

Output tree under `--output-root` (one per object), in the baseline's own layout with the object name as the task:

    assets/objects/<object>/visual.obj                the mesh, texture-seam vertices merged (the model's metric frame)
    assets/objects/<object>/convex/<i>.obj            CoACD parts (decompose_mesh)
    mano/<side>/<object>/task_info.json               the baseline's task record
    mano/<side>/<object>/0/trajectory_keypoints.npz   neutral 3-frame stand-in (below)
    sharpa/<side>/<object>/0/scene_ik.xml             generate_scene: the pose-independent structural scene
    sharpa/<side>/<object>/0/trajectory_kinematic.npz solve_ik on the stand-in: a structural check, never a reference
    sharpa/<side>/<object>/0/scene.xml, scene_eq.xml  resolve_scene_pedestal with use_pedestal False

Stages, with launch.py's arguments except where noted:
1. decompose_mesh with thicken and dilate 2 mm and the CoACD parameters given on the command line;
2. generate_scene with robot sharpa, friction_scale 1.5, no floor collisions, the UR3 arm cylinder, no viewer;
3. solve_ik without smoothing, on the stand-in;
4. resolve_scene_pedestal with use_pedestal and use_support False.

Why a stand-in, and why no pedestal. launch.py builds one scene per clip: the pedestal and its welded support sit at that
clip's IK endpoint object pose. A policy scene is shared by every clip of its object, so it carries no clip's pedestal;
the shipped whisking scene has none either.
- generate_scene reads the task's keypoint file only for the contact-site positions, which are zeros when absent.
- resolve_scene_pedestal loads the keypoints and an IK output even with use_pedestal False.
So the driver writes a neutral 3-frame keypoint file (object at the origin, wrist and fingertips 0.2 m above it) and runs
solve_ik on it. That run shows mink solves on scene_ik.xml and produces the file resolve_scene_pedestal needs.

The last stdout line is the JSON record, also written to `--out-json`; the stages log through loguru to stderr.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

STAGES = ("decompose_mesh", "generate_scene", "solve_ik", "resolve_pedestal")
PREPROCESS_RESOLUTION = 100          # coacd_convex_decomp's default; decompose_mesh.main does not expose it
DATASET = "do_as_i_do"
ROBOT = "sharpa"


def write_stand_in_keypoints(path: Path) -> None:
    """The neutral 3-frame keypoint file of the module docstring (solve_ik uses frames 0 and 1)."""
    n = 3
    obj = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (n, 1))
    wrist = np.tile(np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]), (n, 1))
    fingers = np.tile(wrist[:, None, :], (1, 5, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, qpos_finger_right=fingers, qpos_finger_left=fingers, qpos_wrist_right=wrist, qpos_wrist_left=wrist,
             qpos_obj_right=obj, qpos_obj_left=obj, contact_pos_right=np.zeros((5, 3)),
             contact_pos_left=np.zeros((5, 3)))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--side", required=True, choices=("right", "left"))
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--max-convex-hull", type=int, required=True)
    parser.add_argument("--max-ch-vertex", type=int, required=True)
    parser.add_argument("--preprocess-resolution", type=int, default=PREPROCESS_RESOLUTION)
    parser.add_argument("--thicken", type=float, required=True)
    parser.add_argument("--dilate", type=float, required=True)
    parser.add_argument("--friction-scale", type=float, required=True)
    parser.add_argument("--add-ur3-arm", type=int, choices=(0, 1), required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.preprocess_resolution != PREPROCESS_RESOLUTION:
        raise SystemExit(f"preprocess_resolution is fixed at {PREPROCESS_RESOLUTION} inside the baseline's decompose_mesh")
    sys.dont_write_bytecode = True                       # nothing lands in the read-only third_party tree
    sys.path.insert(0, str(Path(args.baseline_root).resolve()))
    import coacd
    import mujoco
    import trimesh
    from retargeting.pipeline import decompose_mesh, generate_scene, solve_ik
    from retargeting.pipeline.resolve_pedestal import resolve_scene_pedestal

    root, name, side = Path(args.output_root).resolve(), str(args.object), str(args.side)
    other = "left" if side == "right" else "right"
    mesh_dir = root / "assets" / "objects" / name
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.mesh, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), f"{args.mesh} did not load as one mesh"
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.export(mesh_dir / "visual.obj")
    task_dir = root / "mano" / side / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_info.json").write_text(json.dumps({"task": name, "dataset_name": DATASET,
                                                         f"{side}_object_mesh_dir": os.path.relpath(mesh_dir, root),
                                                         f"{other}_object_mesh_dir": None}, indent=2))
    write_stand_in_keypoints(task_dir / "0" / "trajectory_keypoints.npz")
    common: dict[str, Any] = {"output_root_dir": str(root), "dataset_name": DATASET, "task": name, "data_id": 0,
                              "embodiment_type": side}
    seconds: dict[str, float] = {}

    def timed(stage: str, fn: Any, **kwargs: Any) -> None:
        start = time.perf_counter()
        fn(**kwargs)
        seconds[stage] = time.perf_counter() - start

    timed("decompose_mesh", decompose_mesh.main, **common, threshold=args.threshold,
          max_convex_hull=args.max_convex_hull, max_ch_vertex=args.max_ch_vertex, thicken=args.thicken,
          dilate=args.dilate, force=True)
    timed("generate_scene", generate_scene.main, **common, robot_type=ROBOT, show_viewer=False,
          friction_scale=args.friction_scale, object_floor_collision=False, hand_floor_collision=False,
          use_pedestal=True, use_support=True, force=True, add_ur3_arm=bool(args.add_ur3_arm))
    timed("solve_ik", solve_ik.main, **common, robot_type=ROBOT, show_viewer=False, force=True, smoothing=False)
    timed("resolve_pedestal", resolve_scene_pedestal, **common, robot_type=ROBOT, use_pedestal=False, use_support=False,
          friction_scale=args.friction_scale, force=True)
    processed = root / ROBOT / side / name / "0"
    record = {
        "object": name, "side": side, "mesh_source": str(Path(args.mesh).resolve()),
        "scene_xml": str(processed / "scene.xml"), "scene_ik_xml": str(processed / "scene_ik.xml"),
        "scene_eq_xml": str(processed / "scene_eq.xml"), "stages": list(STAGES), "stage_seconds": seconds,
        "coacd": {"method": "coacd", "threshold": args.threshold, "max_convex_hull": args.max_convex_hull,
                  "max_ch_vertex": args.max_ch_vertex, "preprocess_resolution": PREPROCESS_RESOLUTION,
                  "thicken": args.thicken, "dilate": args.dilate},
        "n_parts": len(sorted((mesh_dir / "convex").glob("*.obj"))), "use_pedestal": False,
        "add_ur3_arm": bool(args.add_ur3_arm), "friction_scale": args.friction_scale,
        "ik_stand_in": "3-frame neutral keypoints; the IK output is a structural check, never a reference",
        "mesh": {"n_vertices": len(mesh.vertices), "n_faces": len(mesh.faces),
                 "watertight": bool(mesh.is_watertight), "extents_m": [float(v) for v in mesh.extents]},
        "affinity": sorted(os.sched_getaffinity(0)),   # provenance: the CPU mask the stages ran under
        "baseline_root": str(Path(args.baseline_root).resolve()), "mujoco_version": str(mujoco.__version__),
        "coacd_version": str(getattr(coacd, "__version__", "unknown")),
        "built_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    Path(args.out_json).write_text(json.dumps(record, indent=1))
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
