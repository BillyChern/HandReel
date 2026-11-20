"""The F2.5 gate verdict logic on synthetic reps: CPU only, no worker."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from tests.shape.sam3d_det_gate import G2_LIMIT_GIB, evaluate_gates

_BOX = trimesh.creation.box(extents=(0.3, 0.2, 0.4))


def _rep(mode: str, shift: float = 0.0, gate_s: float = 3.0, reserved: float | None = 19.0) -> dict[str, Any]:
    """One successful request; `shift` moves the mesh, so equal shifts give byte-identical meshes."""
    vertices = np.asarray(_BOX.vertices, dtype=np.float64) + shift
    return {"mode": mode, "vertices": vertices, "faces": np.asarray(_BOX.faces, dtype=np.int64),
            "phase_seconds": {"sparse_structure_flow": 3.0, "structured_latent_flow": gate_s - 0.1, "decode": 0.1},
            "wall_seconds": 10.0, "peak_gpu_gib_allocated": 18.0, "peak_gpu_gib_reserved": reserved,
            "digests": {"mesh_vertices": f"{mode}{shift}"}}


def _process1(a_shifts: tuple[float, float, float] = (0.0, 0.01, 0.02), d_shift: float = 0.5,
              d_gate_s: float = 3.2) -> list[dict[str, Any]]:
    a = [_rep("A", s) for s in a_shifts]
    d = [_rep("D", d_shift, gate_s=d_gate_s) for _ in range(3)]
    return [a[0], d[0], a[1], d[1], a[2], d[2]]                  # A D A D A D, as run_f2_gates orders them


def test_gates_pass_with_positive_control() -> None:
    p1 = {"c1": _process1(), "c2": _process1(a_shifts=(0.0, 0.0, 0.0))}
    p2 = {"c1": [_rep("D", 0.5), _rep("D", 0.5)], "c2": [_rep("D", 0.5), _rep("D", 0.5)]}
    record = evaluate_gates(p1, p2)
    assert record["gates"] == {"G1": "pass", "G2": "pass", "G3": "pass"}
    assert record["positive_control"] is True                    # one clip whose A reps differ is enough (M3)
    assert record["clips"]["c1"]["A_all_equal"] is False and record["clips"]["c2"]["A_all_equal"] is True
    assert record["max_peak_gpu_gib_reserved_D"] == 19.0
    assert abs(record["clips"]["c1"]["delta_gate_seconds"] - 0.2) < 1e-9


def test_g1_inconclusive_when_every_a_rep_is_equal() -> None:
    p1 = {"c1": _process1(a_shifts=(0.0, 0.0, 0.0))}
    record = evaluate_gates(p1, {"c1": [_rep("D", 0.5), _rep("D", 0.5)]})
    assert record["positive_control"] is False
    assert record["gates"] == {"G1": "inconclusive", "G2": "pass", "G3": "pass"}


def test_g1_fails_on_a_d_mismatch_in_process_or_across_processes() -> None:
    p1 = _process1()
    p1[3] = _rep("D", 0.6, gate_s=3.2)                            # the second D mesh differs
    record = evaluate_gates({"c1": p1}, {"c1": [_rep("D", 0.5), _rep("D", 0.5)]})
    assert record["gates"]["G1"] == "fail" and record["G1_in_process_only"] is False
    record = evaluate_gates({"c1": _process1()}, {"c1": [_rep("D", 0.5), _rep("D", 0.7)]})   # H2: fresh process
    assert record["gates"]["G1"] == "fail" and record["G1_in_process_only"] is True
    assert record["clips"]["c1"]["D_cross_process_equal"] is False


def test_errored_request_fails_g1_and_g2() -> None:
    p1 = _process1()
    p1[5] = {"mode": "D", "error": "RuntimeError: SAM 3D worker failed on request 6: CUDA out of memory"}
    record = evaluate_gates({"c1": p1}, {"c1": [_rep("D", 0.5), _rep("D", 0.5)]})
    assert record["clips"]["c1"]["errors"] == [p1[5]["error"]]
    assert record["gates"]["G1"] == "fail"                        # only two D reps: never "equal"
    assert record["gates"]["G2"] == "fail" and record["gates"]["G3"] == "pass"


def test_g2_and_g3_limits() -> None:
    p2_over = {"c1": [_rep("D", 0.5), _rep("D", 0.5, reserved=G2_LIMIT_GIB + 0.01)]}
    assert evaluate_gates({"c1": _process1()}, p2_over)["gates"]["G2"] == "fail"
    p2_none = {"c1": [_rep("D", 0.5), _rep("D", 0.5, reserved=None)]}          # off-CUDA peak: never a pass
    assert evaluate_gates({"c1": _process1()}, p2_none)["gates"]["G2"] == "fail"
    slow = evaluate_gates({"c1": _process1(d_gate_s=3.51)}, {"c1": [_rep("D", 0.5), _rep("D", 0.5)]})
    assert slow["gates"]["G3"] == "fail" and slow["gates"]["G1"] == "pass"
