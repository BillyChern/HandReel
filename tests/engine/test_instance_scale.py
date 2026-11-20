"""Instance-size fusion (`fastdo.engine.instance_scale`): the size functional, the median rule and its files."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from fastdo.engine.instance_scale import (
    MIN_K,
    plan_instance_scales,
    read_fixed_scale,
    template_size,
    window_size,
    write_instance_scale,
)
from fastdo.engine.worker_support import save_npz


def _box(a: float, b: float, c: float) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(a, b, c))


def _box_sigma(a: float, b: float, c: float) -> float:
    """The exact surface-uniform principal deviations of an a x b x c box: its two faces normal to an axis sit at
    +-half that side, the other four faces spread uniformly over it."""
    area = 2 * (b * c + a * c + a * b)
    var = [(2 * q * r * p**2 / 4 + 2 * (area / 2 - q * r) * p**2 / 12) / area for p, q, r in ((a, b, c), (b, a, c),
                                                                                              (c, a, b))]
    return float(np.exp(np.mean(np.log(np.sqrt(var)))))


def test_template_size_known_answer_and_equivariance() -> None:
    box = _box(0.3, 0.2, 0.1)
    sigma = template_size(box.vertices, box.faces)
    assert sigma == pytest.approx(_box_sigma(0.3, 0.2, 0.1), rel=0.02)          # 10k samples: ~1% noise
    # the same seed samples the same faces and barycentres, so scaling the mesh scales sigma exactly
    assert template_size(2.5 * box.vertices, box.faces) == pytest.approx(2.5 * sigma, rel=1e-12)
    rot = trimesh.transformations.random_rotation_matrix(np.random.default_rng(1).random(3))[:3, :3]
    assert template_size(box.vertices @ rot.T, box.faces) == pytest.approx(sigma, rel=0.03)   # PCA: frame-free
    assert template_size(box.vertices, box.faces) == sigma                                     # seeded: repeatable
    flat = box.vertices * np.array([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        template_size(flat, box.faces)


def _sizes(ms: dict[str, float], sigma: float = 0.1) -> dict[str, dict[str, float]]:
    return {w: {"scale": m / sigma, "template_size": sigma, "size_m": m} for w, m in ms.items()}


def test_plan_takes_the_log_median_and_gives_each_window_its_own_scale() -> None:
    sizes = _sizes({"a": 0.10, "b": 0.12, "c": 1.50})                 # "c" is a 12x outlier
    sizes["b"]["template_size"], sizes["b"]["scale"] = 0.2, 0.6         # another template size, the same metric size
    plan = plan_instance_scales(sizes, {"a": "I", "b": "I", "c": "I"})
    assert set(plan) == {"a", "b", "c"}
    for w, rec in plan.items():
        assert rec["fused_size_m"] == pytest.approx(0.12, rel=1e-12)    # median, not mean: the outlier does not move it
        assert rec["scale"] == pytest.approx(0.12 / sizes[w]["template_size"], rel=1e-12)
        assert (rec["k"], rec["members"], rec["instance"]) == (3, ["a", "b", "c"], "I")
        assert rec["production_scale"] == sizes[w]["scale"]
    even = plan_instance_scales(_sizes({"a": 0.1, "b": 0.2, "c": 0.4, "d": 0.8}), dict.fromkeys("abcd", "J"))
    assert even["a"]["fused_size_m"] == pytest.approx(math.sqrt(0.2 * 0.4), rel=1e-12)   # mean of the middle LOGS


def test_plan_needs_k_min_solved_members_and_an_instance_key() -> None:
    assert MIN_K == 3
    sizes = _sizes({"a": 0.1, "b": 0.2, "c": 0.3, "x": 0.1, "y": 0.2})
    keys = {"a": "I", "b": "I", "c": "I", "x": "J", "y": "J", "z": "J"}   # "z" was grouped but never solved
    plan = plan_instance_scales(sizes, keys)
    assert set(plan) == {"a", "b", "c"}                                   # J has K = 2 < 3: production stays
    assert set(plan_instance_scales(sizes, keys, min_k=2)) == {"a", "b", "c", "x", "y"}
    with pytest.raises(ValueError, match="instance key"):
        plan_instance_scales(sizes, {"a": "I"})
    with pytest.raises(ValueError, match="min_k"):
        plan_instance_scales(sizes, keys, min_k=0)


def test_window_size_and_the_fixed_scale_file_round_trip(tmp_path: Path) -> None:
    box = _box(0.3, 0.2, 0.1)
    box.export(tmp_path / "mesh.obj")
    save_npz(tmp_path / "solve.npz", scale=np.float64(0.7), poses=np.eye(4)[None])
    ws = window_size(tmp_path)
    mesh = trimesh.load(tmp_path / "mesh.obj", force="mesh", process=False)
    assert ws["scale"] == 0.7 and ws["template_size"] == template_size(mesh.vertices, mesh.faces)
    assert ws["size_m"] == pytest.approx(0.7 * ws["template_size"], rel=1e-15)
    rec = plan_instance_scales({"w": ws, "v": ws, "u": ws}, dict.fromkeys("wvu", "I"))["w"]
    path = write_instance_scale(tmp_path, rec)
    scale, back = read_fixed_scale(path)
    assert scale == rec["scale"] and back == json.loads(json.dumps(rec))
    (tmp_path / "solve.npz").unlink()
    with pytest.raises(FileNotFoundError, match="solve.npz"):
        window_size(tmp_path)
    with pytest.raises(FileNotFoundError, match="fusion post-pass"):
        read_fixed_scale(tmp_path / "absent.json")
    for bad in ([1.0], {"k": 3}, {"scale": 0.0}, {"scale": "nan"}):
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            read_fixed_scale(tmp_path / "bad.json")
