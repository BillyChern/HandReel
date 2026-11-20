"""Opt-in cross-window shape reuse (fastdo.engine.shape_reuse) and its reachability from build_stage_fns."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastdo.engine.shape_reuse import (
    REUSED_FILES,
    load_donor_map,
    make_reuse_shape_worker,
    plan_donors,
)


def _fake_inner(calls: list[str]):
    def inner(clip_id: str, artifact_dir: Path) -> None:
        calls.append(clip_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for f in REUSED_FILES:
            (artifact_dir / f).write_bytes(f"own-{clip_id}-{f}".encode())
        (artifact_dir / "shape.json").write_text(json.dumps({"clip_id": clip_id, "keyframe_index": 3, "frame_idx": 9}))
    return inner


def test_plan_donors_first_k_generate_and_the_largest_mask_donates() -> None:
    inst = {"a": "v:cup", "b": "v:cup", "c": "v:cup", "d": "v:cup", "e": "v:box", "f": None}
    order = ["a", "b", "e", "c", "d", "f"]
    area = {"a": 0.01, "b": 0.03, "e": 0.2, "c": 0.5, "d": 0.9}
    assert plan_donors(inst, order, area, k_generators=2) == {"c": "b", "d": "b"}
    assert plan_donors(inst, order, area, k_generators=1) == {"b": "a", "c": "a", "d": "a"}
    assert plan_donors(inst, order, {**area, "b": 0.01}, k_generators=2) == {"c": "a", "d": "a"}   # tie -> earlier
    with pytest.raises(ValueError, match="k_generators"):
        plan_donors(inst, order, area, k_generators=0)
    with pytest.raises(ValueError, match="repeats"):
        plan_donors(inst, ["a", "a"], area)


def test_unlisted_clips_run_the_inner_worker_unchanged(tmp_path: Path) -> None:
    calls: list[str] = []
    worker = make_reuse_shape_worker(_fake_inner(calls), {"t": "g"})
    worker("x", tmp_path / "x")
    assert calls == ["x"] and "reuse" not in json.loads((tmp_path / "x" / "shape.json").read_text())


def test_a_listed_clip_copies_the_donor_bytes_and_records_provenance(tmp_path: Path) -> None:
    calls: list[str] = []
    inner = _fake_inner(calls)
    inner("g", tmp_path / "g")
    calls.clear()
    make_reuse_shape_worker(inner, {"t": "g"})("t", tmp_path / "t")
    assert calls == []                                                     # no model call
    for f in REUSED_FILES:
        assert (tmp_path / "t" / f).read_bytes() == (tmp_path / "g" / f).read_bytes()
    meta = json.loads((tmp_path / "t" / "shape.json").read_text())
    assert meta["clip_id"] == "t" and meta["reuse"] == {"donor": "g", "donor_keyframe_index": 3, "donor_frame_idx": 9}


def test_a_missing_donor_falls_back_to_the_own_shape(tmp_path: Path) -> None:
    calls: list[str] = []
    make_reuse_shape_worker(_fake_inner(calls), {"t": "g"})("t", tmp_path / "t")
    assert calls == ["t"]
    meta = json.loads((tmp_path / "t" / "shape.json").read_text())
    assert meta["reuse"]["donor"] == "g" and "fallback" in meta["reuse"]
    assert (tmp_path / "t" / "mesh.obj").read_bytes() == b"own-t-mesh.obj"


def test_load_donor_map_refuses_self_and_chained_donors(tmp_path: Path) -> None:
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"t": "g", "u": "g"}))
    assert load_donor_map(p) == {"t": "g", "u": "g"}
    for bad in ({"t": "t"}, {"t": "g", "g": "h"}):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="donor"):
            load_donor_map(p)
    p.write_text(json.dumps(["t"]))
    with pytest.raises(ValueError, match="JSON object"):
        load_donor_map(p)


def test_the_shipped_config_has_reuse_off_and_build_stage_fns_reaches_the_wrapper(tmp_path: Path) -> None:
    """Default null: the shape stage is the plain model worker (its status label unchanged). Set: the registered
    entry point wraps it (reachability, not just an implementation)."""
    from hydra import compose, initialize_config_dir

    from fastdo.bench.bench_stages import build_stage_fns
    repo = Path(__file__).resolve().parents[2]
    base = [f"paths.root={repo}", f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}"]
    with initialize_config_dir(config_dir=str(repo / "run/conf"), version_base=None):
        off = compose(config_name="config", overrides=base)
        donors = tmp_path / "donors.json"
        donors.write_text(json.dumps({"t": "g"}))
        on = compose(config_name="config", overrides=[*base, f"bench.shape_reuse.donors={donors}"])
    assert off.bench.shape_reuse.donors is None
    fns_off, status_off = build_stage_fns(off, repo, [])
    fns_on, status_on = build_stage_fns(on, repo, [])
    assert status_off["shape"] == "computed (sam3d)"
    assert status_on["shape"] == "computed (sam3d); 1 windows reuse a donor's template (bench.shape_reuse)"
    assert fns_on["shape"].__name__ == "reuse_shape_worker" and fns_off["shape"].__name__ != "reuse_shape_worker"
