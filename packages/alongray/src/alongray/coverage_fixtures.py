# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic known-answer cases for :mod:`alongray.coverage` (``alongray-coverage --write-known-answers DIR``).

Every case is a folder ``DIR/<case>/`` with ``protocol.json``, method folders ``a/`` and ``b/``, a ground-truth
folder ``ref/`` in the method format, and ``expected.json`` (the verdict and the flag codes the validator must
raise). Method ``a`` equals the truth; ``b`` is the truth under one mutation. An adapter author can point their own
adapter at ``ref/`` data, write a folder, and check that the validator reads it as ``a``.

The synthetic protocol: two sequences, three units on a stride-2 frame grid, 18 assigned and 17 annotated frames
(frame 24 of ``S0:w1`` has no annotation), positions about 0.5 m in front of the camera, 30 fps. The stride-1 case
uses four 20-frame units instead (80 assigned and annotated frames).

``expected.json`` lists the EXACT flag codes (``flags``, which set COMPARABLE_WITH_FLAGS) and the exact informational
codes (``info``, which never change the verdict).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CASES", "EXPECTED", "EXPECTED_INFO", "misread_quaternion_order", "truth", "write_known_answers"]

FPS = 30.0
UNITS = [("S0:w0", "S0", list(range(0, 12, 2)), list(range(0, 12, 2))),
         ("S0:w1", "S0", [20, 22, 24, 26], [20, 22, 26]),
         ("S1:w0", "S1", list(range(0, 16, 2)), list(range(0, 16, 2)))]
DENSE_UNITS = [("S0:w0", "S0", list(range(20)), list(range(20))),
               ("S0:w1", "S0", list(range(40, 60)), list(range(40, 60))),
               ("S1:w0", "S1", list(range(20)), list(range(20))),
               ("S1:w1", "S1", list(range(30, 50)), list(range(30, 50)))]
CONV = {"length_unit": "m", "frame": "camera", "camera_axes": "opencv", "frame_index_base": 0,
        "clock": "source_frame_index", "object_origin": "mesh_area_centroid", "scale": "metric",
        "symmetry": "min_over_group", "registration": "none", "alignment_fit_support": "none (unaligned)",
        "aggregation": "sequence"}
PROV = {"dataset": "SYNTH", "split": "test", "gt_release": "synth-1"}
SCORES = {"S0:w0": 10.0, "S0:w1": 40.0, "S1:w0": 50.0}
GL = np.diag([1.0, -1.0, -1.0])


def _rotvec(v: np.ndarray) -> np.ndarray:
    t = float(np.linalg.norm(v))
    k = v / t
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * kx + (1 - np.cos(t)) * kx @ kx


RZ_PI = _rotvec(np.array([0.0, 0.0, np.pi]))


def truth(unit: int, frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Object positions (N, 3) in metres and rotations (N, 3, 3), OpenCV camera frame."""
    f = np.asarray(frames, float)
    p = np.stack([0.05 + 0.01 * np.sin(0.1 * f + unit), -0.02 + 0.01 * np.cos(0.13 * f),
                  0.5 + 0.02 * np.sin(0.07 * f + unit)], 1)
    r = np.stack([_rotvec(np.array([0.1 + 0.02 * x, -0.05 * unit - 0.05, 0.3 + 0.01 * x])) for x in f])
    return p, r


def quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix with trace > -1."""
    w = np.sqrt(max(0.0, 1.0 + float(np.trace(r)))) / 2.0
    return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)])


def rot_from_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def misread_quaternion_order(rot: np.ndarray) -> np.ndarray:
    """Rotations written as (w, x, y, z) and read back as if stored (x, y, z, w): an adapter convention error."""
    return np.stack([rot_from_wxyz(np.roll(quat_wxyz(r), -1)) for r in rot])


def _protocol(dense: bool = False, **kw: Any) -> dict[str, Any]:
    units = [{"unit_id": u, "sequence": s, "camera": "c0", "object": "box", "frames": fr, "scorable": sc}
             for u, s, fr, sc in (DENSE_UNITS if dense else UNITS)]
    d = {"schema": "alongray-coverage-protocol/1", "name": "synthetic", "provenance": dict(PROV),
         "modality": {"input": "rgb_monocular", "compare": ["object_position"]}, "conventions": dict(CONV),
         "cluster": "sequence", "frame_rate": FPS, "units": units}
    d.update(kw)
    return d


def _method(name: str, proto: dict[str, Any]) -> dict[str, Any]:
    """Manifest and arrays of the truth on every assigned frame of every unit."""
    man = {"schema": "alongray-coverage-method/1", "method": name, "outputs": "per_frame",
           "provenance": dict(PROV), "modality": {"input": "rgb_monocular",
                                                  "outputs": ["object_position", "object_rotation"]},
           "inputs": {"intrinsics": "dataset"}, "conventions": dict(CONV), "units": {}, "scores": {}}
    arrays = {}
    for i, u in enumerate(proto["units"]):
        uid = u["unit_id"]
        p, r = truth(i if i < 3 else int(uid.endswith("c")), u["frames"])
        arrays[uid] = {"frame_ids": np.asarray(u["frames"], np.int64), "object_position": p, "object_rotation": r,
                       "timestamps": np.asarray(u["frames"], float) / FPS}
        man["units"][uid] = {"status": "delivered", "file": f"{uid.replace(':', '_')}.npz",
                             "sequence": u["sequence"], "camera": u["camera"], "object": u["object"]}
        man["scores"][uid] = {"n_frames": len(u["scorable"]), "err_mm": SCORES.get(uid, 30.0)}
    return {"manifest": man, "arrays": arrays}


def _keep(b: dict[str, Any], uid: str, keep: Callable[[int], bool]) -> None:
    a = b["arrays"][uid]
    m = np.array([keep(int(f)) for f in a["frame_ids"]])
    b["arrays"][uid] = {k: v[m] for k, v in a.items()}


def _map(b: dict[str, Any], fn: Callable[[dict[str, np.ndarray]], None]) -> None:
    for a in b["arrays"].values():
        fn(a)


def _cam_world(a: dict[str, np.ndarray]) -> None:
    rc, tc = _rotvec(np.array([0.0, np.pi / 6, 0.0])), np.array([0.1, 0.2, -0.3])
    n = len(a["frame_ids"])
    m = np.tile(np.eye(4), (n, 1, 1))
    m[:, :3, :3], m[:, :3, 3] = rc, tc
    a["object_position"] = (a["object_position"] - tc) @ rc
    a["object_rotation"] = rc.T @ a["object_rotation"]
    a["camera_from_world"] = m


def _gl(a: dict[str, np.ndarray]) -> None:
    a["object_position"], a["object_rotation"] = a["object_position"] @ GL.T, GL @ a["object_rotation"]


def _shift_origin(a: dict[str, np.ndarray]) -> None:
    a["object_position"] = a["object_position"] + a["object_rotation"] @ np.array([0.03, 0.0, 0.0])


def _mutate(case: str, p: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> None:
    bm, bc = b["manifest"], b["manifest"]["conventions"]
    if case == "origin_shift":
        _map(b, _shift_origin)
    elif case in ("unit_swap", "unit_declared"):
        _map(b, lambda x: x.__setitem__("object_position", x["object_position"] * 1e3))
        bc["length_unit"] = "m" if case == "unit_swap" else "mm"
    elif case == "missing_frames":
        _keep(b, "S0:w0", lambda f: f <= 4)
        bm["units"]["S0:w1"] = {"status": "failed", "reason": "qc_rejected"}
        bm["scores"]["S0:w0"]["n_frames"] = 3
        del bm["scores"]["S0:w1"]
    elif case == "aggregation":
        a["manifest"]["conventions"]["aggregation"] = "frame"
    elif case == "symmetric":
        p["objects"] = {"can": {"symmetry": [np.eye(3).tolist(), RZ_PI.tolist()]}}
        p["modality"]["compare"] = ["object_position", "object_rotation"]
        for u in p["units"]:
            u["object"] = "can"
        for m in (a, b):
            for u in m["manifest"]["units"].values():
                u["object"] = "can"
        _map(b, lambda x: x.__setitem__("object_rotation", x["object_rotation"] @ RZ_PI))
        a["manifest"]["conventions"]["symmetry"] = "none"
    elif case == "modality":
        bm["modality"]["input"] = "rgbd"
    elif case == "provenance":
        bm["provenance"]["split"] = "train"
    elif case == "reported_only":
        b["arrays"] = {}
        b["manifest"] = {"schema": "alongray-coverage-method/1", "method": "b", "outputs": "reported_only",
                         "provenance": dict(PROV), "modality": {"input": "rgb_monocular", "outputs": []},
                         "conventions": dict(CONV), "reported": {
                             "source": "a paper table", "cohort": "every 4th frame",
                             "evaluated_frames": {"S0": [0, 4, 8, 20, 24], "S1": [0, 4, 8, 12]}}}
    elif case == "counts_only":
        p.update(units=[], published_counts={"assigned": 20, "run": 16, "not_run": 4, "removed": 1, "analysed": 15,
                                             "both": 9})
        for m, n in ((a, 11), (b, 12)):
            m["arrays"] = {}
            m["manifest"] = {"method": m["manifest"]["method"], "outputs": "counts_only", "provenance": dict(PROV),
                             "modality": {"input": "rgb_monocular", "outputs": []}, "conventions": dict(CONV),
                             "counts": {"delivered": n}}
    elif case == "timing":
        a["manifest"]["wall_seconds"] = 12.5
        for u in a["manifest"]["units"].values():
            u["gpu_seconds"] = 3.0
        bm["fps"] = 30.0
    elif case == "camera_world":
        _map(b, _cam_world)
        bc["frame"] = "world"
    elif case in ("camera_gl_undeclared", "camera_gl_declared"):
        _map(b, _gl)
        bc["camera_axes"] = "opencv" if case == "camera_gl_undeclared" else "opengl"
    elif case in ("frame_base_undeclared", "frame_base_declared", "frame_base_undeclared_stride1"):
        _map(b, lambda x: x.__setitem__("frame_ids", x["frame_ids"] + 1))
        bc["frame_index_base"] = 1 if case == "frame_base_declared" else 0
        if case == "frame_base_undeclared_stride1":
            a["manifest"]["scores"], bm["scores"] = {}, {}
    elif case == "rotation_convention":
        p["modality"]["compare"] = ["object_position", "object_rotation"]
        _map(b, lambda x: x.__setitem__("object_rotation", misread_quaternion_order(x["object_rotation"])))
    elif case == "clock":
        _map(b, lambda x: x.__setitem__("timestamps", x["frame_ids"] / 25.0))
    elif case == "asymmetric_release":
        for uid in ("S0:w0", "S0:w1"):
            _keep(b, uid, lambda f: f % 4 == 0)
        bm["scores"]["S0:w0"]["n_frames"], bm["scores"]["S0:w1"]["n_frames"] = 3, 1
    elif case == "vanishing_failure":
        del bm["units"]["S0:w1"], bm["scores"]["S0:w1"]
        del b["arrays"]["S0:w1"]


def _overlap_protocol(policy: str) -> dict[str, Any]:
    p = _protocol(overlap_policy=policy)
    p["units"].append({"unit_id": "S0:w0b", "sequence": "S0", "camera": "c0", "object": "box", "frames": [8, 10, 12],
                       "scorable": [8, 10, 12]})
    return p


def _collision_protocol() -> dict[str, Any]:
    p = _protocol()
    p["units"].append({"unit_id": "S0:w0c", "sequence": "S0", "camera": "c1", "object": "box",
                       "frames": list(range(0, 12, 2)), "scorable": list(range(0, 12, 2))})
    return p


EXPECTED: dict[str, tuple[str, list[str]]] = {
    "identical": ("COMPARABLE", []),
    "origin_shift": ("COMPARABLE_WITH_FLAGS", ["ORIGIN_OFFSET_SUSPECT"]),
    "unit_swap": ("COMPARABLE_WITH_FLAGS", ["UNIT_OR_SCALE_SUSPECT"]),
    "unit_declared": ("COMPARABLE", []),
    "missing_frames": ("COMPARABLE_WITH_FLAGS", ["SCORES_NOT_ON_COMMON_FRAMES", "COVERAGE_ASYMMETRIC"]),
    "aggregation": ("COMPARABLE_WITH_FLAGS", ["CONVENTION_MISMATCH"]),
    "symmetric": ("COMPARABLE_WITH_FLAGS", ["CONVENTION_MISMATCH", "SYMMETRY_SENSITIVE"]),
    "modality": ("REFUSED", []),
    "provenance": ("REFUSED", []),
    "reported_only": ("NOT_PAIRABLE", ["INPUT_DIFFERS"]),
    "counts_only": ("COUNTS_ONLY", []),
    "timing": ("COMPARABLE", []),
    "camera_world": ("COMPARABLE", []),
    "camera_gl_undeclared": ("COMPARABLE_WITH_FLAGS", ["CAMERA_AXES_SUSPECT", "FRAME_OR_TRANSFORM_SUSPECT"]),
    "camera_gl_declared": ("COMPARABLE", []),
    "frame_base_undeclared": ("NOT_PAIRABLE", ["FRAME_ID_OFFSET_SUSPECT", "COVERAGE_ASYMMETRIC"]),
    "frame_base_undeclared_stride1": ("COMPARABLE_WITH_FLAGS", ["FRAME_ID_OFFSET_SUSPECT"]),
    "frame_base_declared": ("COMPARABLE", []),
    "clock": ("COMPARABLE_WITH_FLAGS", ["CLOCK_MISMATCH"]),
    "duplicate_keys_refused": ("REFUSED", []),
    "duplicate_keys_kept": ("COMPARABLE", []),
    "id_collision": ("COMPARABLE_WITH_FLAGS", ["SOURCE_MISMATCH", "COVERAGE_ASYMMETRIC"]),
    "asymmetric_release": ("COMPARABLE_WITH_FLAGS", ["COVERAGE_ASYMMETRIC", "SCORES_NOT_ON_COMMON_FRAMES"]),
    "vanishing_failure": ("COMPARABLE_WITH_FLAGS", ["UNDECLARED_UNITS", "COVERAGE_ASYMMETRIC"]),
    "rotation_convention": ("COMPARABLE_WITH_FLAGS", ["ROTATION_DISAGREEMENT"]),
}
# Informational codes (never change the verdict); cases not listed raise none.
EXPECTED_INFO: dict[str, list[str]] = {
    case: ["DELIVERY_DIFFERS"] for case in ("missing_frames", "counts_only", "frame_base_undeclared",
                                            "frame_base_undeclared_stride1", "id_collision", "asymmetric_release",
                                            "vanishing_failure")}
CASES = tuple(EXPECTED)


def _write(folder: Path, m: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(m["manifest"], indent=1) + "\n")
    for uid, arr in m["arrays"].items():
        f = m["manifest"].get("units", {}).get(uid, {}).get("file")
        if f is not None:
            np.savez(folder / f, **arr)


def build(case: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Protocol, method a, method b and the reference of one case (in memory)."""
    if case == "duplicate_keys_refused":
        p = _overlap_protocol("refuse")
    elif case == "duplicate_keys_kept":
        p = _overlap_protocol("keep_with_multiplicity")
    elif case == "id_collision":
        p = _collision_protocol()
    elif case == "frame_base_undeclared_stride1":
        p = _protocol(dense=True)
    else:
        p = _protocol()
    a, b = _method("a", p), _method("b", p)
    if case == "id_collision":
        b["manifest"]["units"]["S0:w0c"]["camera"] = "c0"
        b["manifest"]["units"]["S1:w0"]["object"] = "can"
    _mutate(case, p, a, b)
    return p, a, b, _method("ref", copy.deepcopy(p))  # the truth of the final protocol (identities included)


def write_known_answers(root: str | Path) -> dict[str, dict[str, Any]]:
    """Write every case under ``root`` and return ``{case: {"verdict": ..., "flags": [...]}}``."""
    root = Path(root)
    out = {}
    for case in CASES:
        p, a, b, ref = build(case)
        d = root / case
        d.mkdir(parents=True, exist_ok=True)
        (d / "protocol.json").write_text(json.dumps(p, indent=1) + "\n")
        for name, m in (("a", a), ("b", b), ("ref", ref)):
            _write(d / name, m)
        verdict, flags = EXPECTED[case]
        out[case] = {"verdict": verdict, "flags": flags, "info": EXPECTED_INFO.get(case, [])}
        (d / "expected.json").write_text(json.dumps(out[case], indent=1) + "\n")
    return out
