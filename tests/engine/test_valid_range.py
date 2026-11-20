"""driftqc row-range slicing helper.

Synthetic arrays only (N = 24, H = W = 8, J = 5). The driftqc.json / valid_range.json fixtures come from the local
writer `write_driftqc_pair`, keyed by `VALID_RANGE_KEYS` (the stage-worker tests import those names).
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.engine import valid_range as vr
from fastdo.engine.valid_range import (
    CLIP_LEVEL,
    DRIFTQC_FILE,
    PER_ROW,
    VALID_RANGE_FILE,
    VALID_RANGE_KEYS,
    RowRange,
    frame_idx_sha256,
    read_valid_range,
    resolve_valid_range,
    slice_artifact,
    slice_observations,
)
from fastdo.solver.api import TrajectoryObservations
from fastdo.solver.batch import object_pixels_of

N, H, W, J = 24, 8, 8, 5
ENGINE_SRC = Path(vr.__file__).resolve().parent
_HAND_ROW_KEYS = ("params", "trans", "joints", "valid", "vertices", "infilled")


def _decode_idx() -> np.ndarray:
    return (np.arange(N, dtype=np.int64) * 3 + 100).astype(np.int64)


def _artifacts(seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Every PER_ROW and CLIP_LEVEL key of the four artifacts, with the writers' dtypes (section 4.1)."""
    rng = np.random.default_rng(seed)
    idx = _decode_idx()
    hands: dict[str, np.ndarray] = {"frame_idx": idx.copy()}
    for side in ("left", "right"):
        hands[f"{side}_params"] = rng.standard_normal((N, 48)).astype(np.float32)
        hands[f"{side}_trans"] = rng.standard_normal((N, 3)).astype(np.float32)
        hands[f"{side}_joints"] = rng.standard_normal((N, 21, 3)).astype(np.float32)
        hands[f"{side}_valid"] = rng.random(N) > 0.3
        hands[f"{side}_vertices"] = rng.standard_normal((N, 778, 3)).astype(np.float32)
        hands[f"{side}_infilled"] = rng.random(N) > 0.8
        hands[f"{side}_betas"] = rng.standard_normal(10).astype(np.float32)
        hands[f"{side}_faces"] = rng.integers(0, 778, size=(1552, 3)).astype(np.int32)
    for key in ("camera_K", "hawor_camera_K"):                      # clip-level cameras
        hands[key] = np.array([[600.0, 0.0, 4.0], [0.0, 600.0, 4.0], [0.0, 0.0, 1.0]])
    pointmaps = rng.standard_normal((N, H, W, 3)).astype(np.float32)
    pointmaps[3, 2, 2] = np.nan
    return {
        "mask.npz": {"masks": rng.random((N, H, W)) > 0.5, "frame_idx": idx.copy(),
                     "keyframe_index": np.asarray(0, dtype=np.int64)},
        "geometry.npz": {"pointmaps": pointmaps, "frame_idx": idx.copy(),
                         "frame_scale": (1.0 + 0.1 * rng.random(N)).astype(np.float32),
                         "cam_to_world": rng.standard_normal((N, 4, 4)).astype(np.float32),
                         "scale": np.asarray(1.25, dtype=np.float32),
                         "gravity_cam": rng.standard_normal(3).astype(np.float32)},
        "tracks.npz": {"tracks": rng.standard_normal((N, J, 2)).astype(np.float32), "vis": rng.random((N, J)) > 0.2,
                       "query_uv": rng.standard_normal((J, 2)).astype(np.float32), "frame_idx": idx.copy(),
                       "keyframe_index": np.asarray(0, dtype=np.int64)},
        "hands.npz": hands,
    }


def write_driftqc_pair(artifact_dir: Path, decode_idx: np.ndarray, b: int, clip_id: str = "clip0",
                       **overrides: Any) -> dict[str, Any]:
    """Writes a driftqc.json and a matching valid_range.json with range [0, b]; `overrides` replace record fields."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    n = int(decode_idx.size)
    decision = "ACCEPT" if b == n - 1 else "TRUNCATE"
    driftqc = {"version": 1, "clip_id": clip_id, "segment_id": "#s0", "n_rows": n, "decision": decision,
               "range": [0, b]}
    driftqc_bytes = json.dumps(driftqc, sort_keys=True).encode()
    (artifact_dir / DRIFTQC_FILE).write_bytes(driftqc_bytes)
    values: dict[str, Any] = {
        "version": 1, "clip_id": clip_id, "segment_id": "#s0", "n_rows": n,
        "frame_idx_sha256": hashlib.sha256(np.asarray(decode_idx, np.int64).reshape(-1).tobytes()).hexdigest(),
        "range": [0, b], "decision": decision, "driftqc_sha256": hashlib.sha256(driftqc_bytes).hexdigest(),
    }
    record = {key: values[key] for key in VALID_RANGE_KEYS}
    record.update(overrides)
    (artifact_dir / VALID_RANGE_FILE).write_text(json.dumps(record))
    return record


def test_schema_constants_frozen() -> None:
    assert VALID_RANGE_FILE == "valid_range.json" and DRIFTQC_FILE == "driftqc.json"
    assert VALID_RANGE_KEYS == ("version", "clip_id", "segment_id", "n_rows", "frame_idx_sha256", "range",
                                "decision", "driftqc_sha256")
    hand_rows = {f"{s}_{k}" for s in ("left", "right") for k in _HAND_ROW_KEYS}
    assert dict(PER_ROW) == {
        "mask.npz": frozenset({"masks", "frame_idx"}),
        "geometry.npz": frozenset({"pointmaps", "frame_idx", "frame_scale", "cam_to_world"}),
        "tracks.npz": frozenset({"tracks", "vis", "frame_idx"}),
        "hands.npz": frozenset({"frame_idx"} | hand_rows),
        # the contact stage: its corrected poses and per-frame shifts are decode rows like any other
        "contact.npz": frozenset({"poses", "shift_mm", "frame_idx", "min_cost_mm", "gated"}),
    }
    assert dict(CLIP_LEVEL) == {
        "mask.npz": frozenset({"keyframe_index"}),
        "geometry.npz": frozenset({"scale", "gravity_cam"}),
        "tracks.npz": frozenset({"query_uv", "keyframe_index"}),
        "hands.npz": frozenset({f"{s}_{k}" for s in ("left", "right") for k in ("betas", "faces")}
                               | {"camera_K", "hawor_camera_K"}),
        "contact.npz": frozenset({"side", "n_corrected", "solve_sha256", "constants", "centroid_obj", "bias_free",
                                  "touch_gate_mm", "geometric"}),
    }


def test_row_range_prefix_slice() -> None:
    rng = RowRange(stop=7, n_rows=N)
    assert rng.start == 0
    assert rng.as_slice() == slice(0, 8)
    for stop in (-1, N):
        with pytest.raises(ValueError, match=r"RowRange stop .* outside \[0, "):
            RowRange(stop=stop, n_rows=N)


def test_frame_idx_sha256() -> None:
    idx = _decode_idx()
    expected = hashlib.sha256(np.asarray(idx, np.int64).reshape(-1).tobytes()).hexdigest()
    assert frame_idx_sha256(idx) == expected
    assert frame_idx_sha256(idx.astype(np.int32)) == expected


def test_slice_artifact_mask_geometry_tracks_hands() -> None:
    rng = RowRange(stop=7, n_rows=N)
    for artifact, arrays in _artifacts().items():
        out = slice_artifact(arrays, artifact, rng)
        assert set(out) == set(arrays), artifact
        for key in PER_ROW[artifact]:
            assert out[key].shape[0] == 8, (artifact, key)
            assert out[key].shape[1:] == arrays[key].shape[1:], (artifact, key)
            assert np.array_equal(out[key], arrays[key][:8], equal_nan=out[key].dtype.kind == "f"), (artifact, key)
            assert out[key].dtype == arrays[key].dtype, (artifact, key)
        for key in CLIP_LEVEL[artifact]:
            assert out[key] is arrays[key], (artifact, key)


def test_slice_artifact_full_range_identity() -> None:
    rng = RowRange(stop=N - 1, n_rows=N)
    for artifact, arrays in _artifacts(seed=1).items():
        out = slice_artifact(arrays, artifact, rng)
        for key in PER_ROW[artifact]:
            assert np.array_equal(out[key], arrays[key], equal_nan=out[key].dtype.kind == "f"), (artifact, key)
            assert out[key].dtype == arrays[key].dtype, (artifact, key)
            assert out[key].tobytes() == arrays[key].tobytes(), (artifact, key)
        for key in CLIP_LEVEL[artifact]:
            assert out[key] is arrays[key], (artifact, key)


def test_frame_scale_sliced_bitwise_not_applied() -> None:
    b = 7
    geometry = _artifacts(seed=2)["geometry.npz"]
    perturbed = dict(geometry)
    frame_scale = geometry["frame_scale"].copy()
    frame_scale[1::2] *= np.float32(1.7)
    perturbed["frame_scale"] = frame_scale
    rng = RowRange(stop=b, n_rows=N)
    out = slice_artifact(perturbed, "geometry.npz", rng)
    plain = slice_artifact(geometry, "geometry.npz", rng)
    assert out["pointmaps"].tobytes() == geometry["pointmaps"][: b + 1].tobytes()
    assert out["pointmaps"].tobytes() == plain["pointmaps"].tobytes()
    assert out["frame_scale"].tobytes() == frame_scale[: b + 1].tobytes()
    assert out["frame_scale"].dtype == np.float32


def test_unknown_key_raises() -> None:
    tracks = dict(_artifacts()["tracks.npz"])
    tracks["foo"] = np.zeros(N, dtype=np.float32)
    with pytest.raises(ValueError, match="foo"):
        slice_artifact(tracks, "tracks.npz", RowRange(stop=7, n_rows=N))


def test_unknown_artifact_raises() -> None:
    with pytest.raises(ValueError, match="decode.npz"):
        slice_artifact({"frame_idx": _decode_idx()}, "decode.npz", RowRange(stop=7, n_rows=N))


def test_per_row_length_mismatch_raises() -> None:
    tracks = dict(_artifacts()["tracks.npz"])
    tracks["vis"] = tracks["vis"][: N - 1]
    with pytest.raises(ValueError, match="vis"):
        slice_artifact(tracks, "tracks.npz", RowRange(stop=7, n_rows=N))


def test_read_valid_range_absent_none(tmp_path: Path) -> None:
    assert read_valid_range(tmp_path, _decode_idx()) is None


def test_read_valid_range_full_range_none(tmp_path: Path) -> None:
    write_driftqc_pair(tmp_path, _decode_idx(), b=N - 1)
    assert read_valid_range(tmp_path, _decode_idx()) is None


def test_read_valid_range_prefix(tmp_path: Path) -> None:
    record = write_driftqc_pair(tmp_path, _decode_idx(), b=7)
    assert set(record) == set(VALID_RANGE_KEYS)
    assert read_valid_range(tmp_path, _decode_idx()) == RowRange(stop=7, n_rows=24)


def _missing_driftqc(d: Path) -> None:
    write_driftqc_pair(d, _decode_idx(), b=7)
    (d / DRIFTQC_FILE).unlink()


def _driftqc_bytes_changed(d: Path) -> None:
    write_driftqc_pair(d, _decode_idx(), b=7)
    (d / DRIFTQC_FILE).write_bytes((d / DRIFTQC_FILE).read_bytes() + b" ")


_REJECTS: dict[str, Any] = {
    "missing_driftqc_json": _missing_driftqc,
    "driftqc_sha256_mismatch": _driftqc_bytes_changed,
    "n_rows_23_vs_24": lambda d: write_driftqc_pair(d, _decode_idx(), b=7, n_rows=23),
    "frame_idx_sha256_mismatch": lambda d: write_driftqc_pair(d, _decode_idx() + 1, b=7),
    "range_1_7": lambda d: write_driftqc_pair(d, _decode_idx(), b=7, range=[1, 7]),
    "range_0_24": lambda d: write_driftqc_pair(d, _decode_idx(), b=7, range=[0, 24]),
    "range_negative_stop": lambda d: write_driftqc_pair(d, _decode_idx(), b=7, range=[0, -1]),
    "range_not_pair": lambda d: write_driftqc_pair(d, _decode_idx(), b=7, range=[0, 7, 9]),
    "version_2": lambda d: write_driftqc_pair(d, _decode_idx(), b=7, version=2),
}


_REJECT_MATCH: dict[str, str] = {
    "missing_driftqc_json": "driftqc.json absent next to valid_range.json",
    "driftqc_sha256_mismatch": "driftqc_sha256 .* != sha256 of driftqc.json",
    "n_rows_23_vs_24": "n_rows 23 != decode rows 24",
    "frame_idx_sha256_mismatch": "frame_idx_sha256 does not match",
    "range_1_7": r"range \[1, 7\] is not \[0, b\]",
    "range_0_24": r"range \[0, 24\] is not \[0, b\]",
    "range_negative_stop": r"range \[0, -1\] is not \[0, b\]",
    "range_not_pair": r"range \[0, 7, 9\] is not \[0, b\]",
    "version_2": "version 2 != ",
}


@pytest.mark.parametrize("case", sorted(_REJECTS))
def test_read_valid_range_rejects(tmp_path: Path, case: str) -> None:
    assert set(_REJECT_MATCH) == set(_REJECTS)
    _REJECTS[case](tmp_path)
    with pytest.raises(ValueError, match=_REJECT_MATCH[case]):
        read_valid_range(tmp_path, _decode_idx())


def _observations(t: int = N) -> TrajectoryObservations:
    gen = torch.Generator().manual_seed(0)
    return TrajectoryObservations(
        intrinsics=torch.eye(3, dtype=torch.float64),
        masks=torch.rand(t, H, W, generator=gen) > 0.5,
        hand_masks=torch.rand(t, H, W, generator=gen) > 0.7,
        pointmaps=torch.randn(t, H, W, 3, generator=gen),
        tracks=torch.randn(t, J, 2, generator=gen),
        track_vis=torch.rand(t, J, generator=gen) > 0.2,
        anchors=torch.randn(J, 3, generator=gen),
    )


def test_slice_observations_per_row_and_clip_level() -> None:
    obs = _observations()
    out = slice_observations(obs, RowRange(stop=7, n_rows=N))
    for name in ("masks", "hand_masks", "pointmaps", "tracks", "track_vis"):
        before, after = getattr(obs, name), getattr(out, name)
        assert after.shape == (8, *before.shape[1:]), name
        assert after.dtype == before.dtype, name
        assert torch.equal(after, before[:8]), name
    assert out.intrinsics is obs.intrinsics
    assert out.anchors is obs.anchors
    short_tracks = dataclasses.replace(obs, tracks=obs.tracks[: N - 1])
    with pytest.raises(ValueError, match="tracks"):
        slice_observations(short_tracks, RowRange(stop=7, n_rows=N))


def test_slice_observations_keeps_the_object_pixels_of_the_kept_frames() -> None:
    """Sliced `object_pixels` are exactly `object_pixels_of` the sliced masks and pointmaps."""
    obs = _observations()
    obs = dataclasses.replace(obs, object_pixels=object_pixels_of(obs.masks, obs.pointmaps))
    out = slice_observations(obs, RowRange(stop=7, n_rows=N))
    ref = object_pixels_of(out.masks, out.pointmaps)
    assert out.object_pixels is not None and torch.equal(out.object_pixels.offsets, ref.offsets)
    assert out.object_pixels.points.numpy().tobytes() == ref.points.numpy().tobytes()
    short = dataclasses.replace(obs, object_pixels=object_pixels_of(obs.masks[:-1], obs.pointmaps[:-1]))
    with pytest.raises(ValueError, match="object_pixels"):
        slice_observations(short, RowRange(stop=7, n_rows=N))


def test_resolve_disabled_without_files_none(tmp_path: Path) -> None:
    assert resolve_valid_range(tmp_path, _decode_idx(), driftqc_enabled=False) is None


@pytest.mark.parametrize("leftover", ["driftqc.json", "valid_range.json", "both"])
def test_resolve_disabled_with_stale_files_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  leftover: str) -> None:
    write_driftqc_pair(tmp_path, _decode_idx(), b=7)
    if leftover != "both":
        (tmp_path / (VALID_RANGE_FILE if leftover == DRIFTQC_FILE else DRIFTQC_FILE)).unlink()
    calls: list[object] = []
    monkeypatch.setattr(vr, "read_valid_range", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="stale driftqc artifacts"):
        resolve_valid_range(tmp_path, _decode_idx(), driftqc_enabled=False)
    assert calls == []


def test_resolve_enabled_without_valid_range_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="driftqc enabled but valid_range.json absent"):
        resolve_valid_range(tmp_path, _decode_idx(), driftqc_enabled=True)
    write_driftqc_pair(tmp_path, _decode_idx(), b=7)
    (tmp_path / VALID_RANGE_FILE).unlink()
    with pytest.raises(ValueError, match="driftqc enabled but valid_range.json absent"):
        resolve_valid_range(tmp_path, _decode_idx(), driftqc_enabled=True)


def test_resolve_enabled_reads_range(tmp_path: Path) -> None:
    write_driftqc_pair(tmp_path, _decode_idx(), b=7)
    assert resolve_valid_range(tmp_path, _decode_idx(), driftqc_enabled=True) == RowRange(stop=7, n_rows=N)
    write_driftqc_pair(tmp_path, _decode_idx(), b=N - 1)
    assert resolve_valid_range(tmp_path, _decode_idx(), driftqc_enabled=True) is None


def _frame_scale_arithmetic(path: Path) -> list[str]:
    source = path.read_text()
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp):
            operands: tuple[ast.AST, ...] = (node.left, node.right)
        elif isinstance(node, ast.AugAssign):
            operands = (node.target, node.value)
        else:
            continue
        for operand in operands:
            segment = ast.get_source_segment(source, operand) or ""
            if "frame_scale" in segment:
                hits.append(f"{path.name}:{node.lineno}: {segment}")
    return hits


def test_no_arithmetic_on_frame_scale_ast(tmp_path: Path) -> None:
    files = [ENGINE_SRC / "valid_range.py", ENGINE_SRC / "workers_driftqc.py"]
    assert files[0].is_file()
    present = [p for p in files if p.is_file()]  # workers_driftqc.py arrives in E2/E3; valid_range.py is required
    hits = [hit for path in present for hit in _frame_scale_arithmetic(path)]
    assert hits == []
    probe = tmp_path / "probe.py"  # the check itself discriminates: both arithmetic forms are caught
    probe.write_text("x = pointmaps / arrays['frame_scale'][:, None]\ny = 2.0\ny *= frame_scale\nz = scale * 2\n")
    assert len(_frame_scale_arithmetic(probe)) == 2


def test_slice_artifact_geometric_contact_npz(tmp_path: Path) -> None:
    """A non-default ruler's contact.npz (geometric / bias_free / touch gate) slices like the posterior-mean one."""
    from fastdo.engine.workers_contact import CONTACT_FILE, ContactStageConfig, make_contact_worker
    from tests.engine.test_workers_contact import _clip

    d = _clip(tmp_path)
    make_contact_worker(ContactStageConfig(enabled=True, geometric=True))("clip", d)
    arrays = dict(np.load(d / CONTACT_FILE))
    n = arrays["poses"].shape[0]
    out = slice_artifact(arrays, CONTACT_FILE, RowRange(stop=n - 2, n_rows=n))
    for k in ("poses", "shift_mm", "frame_idx", "min_cost_mm", "gated"):
        assert out[k].shape[0] == n - 1 and np.array_equal(out[k], arrays[k][: n - 1])
    for k in ("geometric", "bias_free", "touch_gate_mm", "constants", "side"):
        assert out[k] is arrays[k]
