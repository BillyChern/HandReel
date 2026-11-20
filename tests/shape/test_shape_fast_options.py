"""The default-off shape fast-path options and their plumbing (CPU, fakes; no GPU).

The options are `mesh_postprocess_impl`, `wall_qc` (with its two thresholds),
`fast_decimator_min_raw_faces`, the per-stage flow step counts, and the worker's allocator pin. Every one is off (or at
today's value) by default, and a default request must produce byte-identical artifacts, shape.json, request digest and
shape cache key.

The GPU body of `fast_postprocess` is not exercised here: its bit-equality against SAM 3D's own postprocess is the
evaluation's A0v/G-rep gate. What is tested here is every decision the options drive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from fastdo.engine import shape_cache as cache_mod
from fastdo.engine.shape_cache import SHAPE_CODE_FILES, ShapeCacheKeyer
from fastdo.engine.workers_shape import read_shape_meta
from fastdo.shape import ShapeConfig, ShapeResult
from fastdo.shape import fast_postprocess as fp
from fastdo.shape import sam3d_backend as backend_mod
from fastdo.shape import wall_qc as qc
from fastdo.shape.backend import SHAPE_OPTION_DEFAULTS, non_default_shape_options
from fastdo.shape.sam3d_backend import (
    ALLOC_CONF_ENV,
    EXPANDABLE_SEGMENTS,
    Sam3dBackend,
    Sam3dWorkerClient,
    sam3d_worker_env,
)
from fastdo.shape.sam3d_worker import request_params_digest, resolve_shape_options, serve

# DESIGN 8.2: the per-request row helper the EVAL imports (section 9).
from tests.shape import sam3d_kernel_pin_gate as gate

# --- helpers -----------------------------------------------------------------------------------------

_PREP = {"sdf_resolution": 16, "sdf_padding": 0.15, "n_samples": 8, "samples_seed": 0}
# Computed on main (74ae6c9) before this change: the `shape_config` part digest of a default config, and the whole
# key with the code identity held fixed. A default-option request must still produce these.
_MAIN_CONFIG_PART = {"trellis": "45ed80440d49978dd789fa9b3f5bfc2229963a04667b466e52e84e2f7d683b68",
                     "sam3d": "d12da77f40026ed9878cf7cde2fc70140608479c1a845f9808966165555aa9d5",
                     "template": "030486769ba59176836355715867691f7888bcc4ee707ffd41dee9a2e0526086"}
_MAIN_KEY_FIXED_CODE = {"trellis": "3b96f285af7eddacd4639bcd0ce3238a81394d1f690fee5af5d579c7ee16c9e6",
                        "sam3d": "2ede4fa29727da573b9b0c32892c46bbdcb4027ff084800d52b8140deb346d35"}


class _Identities:
    def cache_identity(self) -> str:
        return "ckpt"

    def cache_env_identity(self) -> str:
        return "env"


def _keyframe() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool)


def _key(cfg: ShapeConfig) -> tuple[str, dict[str, str]]:
    return ShapeCacheKeyer(cfg, _Identities(), _PREP).key(*_keyframe())


def _box(side: float) -> tuple[np.ndarray, np.ndarray]:
    box = trimesh.creation.box(extents=(side, side, side))
    return np.asarray(box.vertices, dtype=np.float64), np.asarray(box.faces, dtype=np.int64)


def _double_walled(outer: float = 1.0, inner: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """A wall case: one closed shell inside another, as a kept inner wall leaves it (DESIGN 3.1, 4.8)."""
    ov, of = _box(outer)
    iv, i_f = _box(inner)
    return np.concatenate([ov, iv]), np.concatenate([of, i_f + len(ov)])


def _params(**overrides: Any) -> dict[str, Any]:
    base = {"seed": 42, "sparse_structure_steps": 25, "structured_latent_steps": 25, "solver": "euler",
            "mesh_postprocess": True}
    return base | overrides


# --- 1. defaults change nothing ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["trellis", "sam3d", "template"])
def test_a_default_config_keeps_todays_shape_config_part(backend: str) -> None:
    """The new fields are omitted from the key's `shape_config` part while they hold their defaults."""
    cfg = ShapeConfig(backend=backend) if backend != "template" else ShapeConfig(backend=backend,
                                                                                mesh_postprocess=False)
    assert _key(cfg)[1]["shape_config"] == _MAIN_CONFIG_PART[backend]


@pytest.mark.parametrize("backend", ["trellis", "sam3d"])
def test_a_default_request_keeps_todays_cache_key(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """With the code identity held fixed, a default-option request's whole key is the one main computes.

    The code identity itself necessarily moves: every file the fast path touches is in SHAPE_CODE_FILES, exactly as
    the kernel pin field moved it.
    """
    monkeypatch.setattr(cache_mod, "_code_identity", lambda: "CODE-FIXED")
    assert _key(ShapeConfig(backend=backend))[0] == _MAIN_KEY_FIXED_CODE[backend]


def test_every_new_option_changes_the_cache_key_when_it_is_set() -> None:
    """Off the default each option enters the key, exactly as the kernel pin field does."""
    settings: list[dict[str, Any]] = [
        {"mesh_postprocess_impl": "fast"}, {"fast_decimator_min_raw_faces": 10**12},
        {"mesh_postprocess_impl": "fast", "wall_qc": "fallback"},
        {"mesh_postprocess_impl": "fast", "wall_qc": "fallback", "wall_qc_max_removed": 0.30},
        {"mesh_postprocess_impl": "fast", "wall_qc": "fallback", "wall_qc_min_area_ratio": 1.50},
        {"sam3d_expandable_segments": True}, {"sparse_structure_steps": 8}, {"structured_latent_steps": 9}]
    keys = {_key(ShapeConfig(backend="sam3d"))[0]}
    for setting in settings:
        keys.add(_key(ShapeConfig(backend="sam3d", **setting))[0])
    assert len(keys) == len(settings) + 1, "an option does not reach the cache key"


def test_the_new_modules_are_in_shape_code_files() -> None:
    for rel in ("shape/fast_postprocess.py", "shape/wall_qc.py"):
        assert rel in SHAPE_CODE_FILES


def test_a_default_request_digest_is_stable_and_every_option_moves_it() -> None:
    base = request_params_digest(_params())
    assert base == request_params_digest(_params(mesh_postprocess_impl="official", wall_qc="off",
                                                 fast_decimator_min_raw_faces=250000))
    for setting in ({"mesh_postprocess_impl": "fast"}, {"sparse_structure_steps": 8},
                    {"structured_latent_steps": 9}, {"fast_decimator_min_raw_faces": 10**12},
                    {"wall_qc_max_removed": 0.3}, {"wall_qc_min_area_ratio": 1.5}, {"seed": 43}):
        assert request_params_digest(_params(**setting)) != base, setting
    fast = _params(mesh_postprocess_impl="fast")                               # the QC needs the fast path
    assert request_params_digest(_params(mesh_postprocess_impl="fast", wall_qc="fallback")) != \
        request_params_digest(fast)


def test_default_sampling_params_and_worker_env_are_unchanged() -> None:
    cfg = ShapeConfig(backend="sam3d")
    params = Sam3dBackend(cfg).sampling_params
    assert {k: params[k] for k in SHAPE_OPTION_DEFAULTS if k in params} == {
        k: v for k, v in SHAPE_OPTION_DEFAULTS.items() if k in params}
    assert sam3d_worker_env((), {"PATH": "/bin"}) is None                      # inherits, as today
    assert non_default_shape_options(cfg) == {}


# --- 2. the decimator threshold ----------------------------------------------------------------------


@pytest.mark.parametrize(("raw_faces", "expected"), [
    (0, "vtk"), (104_536, "vtk"), (249_999, "vtk"), (250_000, "fast_simplification"),
    (306_000, "fast_simplification"), (907_516, "fast_simplification")])
def test_the_decimator_routes_by_raw_face_count(raw_faces: int, expected: str) -> None:
    """DESIGN 3.7: VTK below 250,000 raw faces, fast_simplification at or above it."""
    assert fp.choose_decimator(raw_faces, fp.DEFAULT_MIN_RAW_FACES) == expected


def test_an_arbitrarily_high_threshold_is_the_instrumented_vtk_only_mode() -> None:
    """DESIGN 8.2: `fast_decimator_min_raw_faces` above any mesh size serves arm A0v."""
    assert fp.choose_decimator(2_000_000, 10**12) == "vtk"
    assert fp.choose_decimator(0, 0) == "fast_simplification"                  # 0: never VTK


def test_the_decimator_refuses_a_negative_face_count() -> None:
    with pytest.raises(ValueError, match="raw face count"):
        fp.choose_decimator(-1, fp.DEFAULT_MIN_RAW_FACES)


# --- 3. the wall QC statistic, its thresholds and its fallback ---------------------------------------


def test_the_qc_statistics_on_a_clean_mesh_and_on_a_wall() -> None:
    """DESIGN 4.8: removed = mincut faces / decimated faces, rho = mesh area / convex hull area."""
    clean = qc.statistics(*_box(1.0), faces_decimated=1000, faces_removed_mincut=630)
    assert clean["removed"] == pytest.approx(0.63)
    assert clean["area_ratio"] == pytest.approx(1.0, abs=1e-6)                 # a box is its own hull

    wall = qc.statistics(*_double_walled(), faces_decimated=1000, faces_removed_mincut=20)
    assert wall["removed"] == pytest.approx(0.02)
    assert wall["area_ratio"] == pytest.approx(1.64, abs=0.02)                 # 1 + 0.8^2, both shells


def test_the_qc_flags_the_wall_and_passes_the_clean_mesh_at_the_designs_thresholds() -> None:
    thresholds = qc.WallQcThresholds()
    assert (thresholds.max_removed, thresholds.min_area_ratio) == (0.25, 1.40)
    assert not qc.flagged(qc.statistics(*_box(1.0), faces_decimated=1000, faces_removed_mincut=630), thresholds)
    assert qc.flagged(qc.statistics(*_double_walled(), faces_decimated=1000, faces_removed_mincut=20), thresholds)


def test_the_qc_needs_both_conditions() -> None:
    """flag <=> removed < max_removed AND rho >= min_area_ratio (DESIGN 4.8)."""
    thresholds = qc.WallQcThresholds()
    assert qc.flagged({"removed": 0.02, "area_ratio": 1.60}, thresholds)
    assert not qc.flagged({"removed": 0.02, "area_ratio": 1.19}, thresholds)   # thin object: low rho
    assert not qc.flagged({"removed": 0.63, "area_ratio": 1.60}, thresholds)   # the mincut did its work
    assert qc.flagged({"removed": 0.249, "area_ratio": 1.40}, thresholds)      # both boundaries
    assert not qc.flagged({"removed": 0.25, "area_ratio": 1.40}, thresholds)


def test_the_thin_whisk_calibration_rows_do_not_flag_at_the_designs_thresholds() -> None:
    """DESIGN 4.8's calibration: singles reach rho 1.19, walls start at 1.60."""
    thresholds = qc.WallQcThresholds()
    for removed, ratio in ((0.00, 0.58), (0.07, 0.74), (0.14, 1.93)):          # whisk, fast_simplification
        assert qc.flagged({"removed": removed, "area_ratio": ratio}, thresholds) is (ratio >= 1.40)
    for removed, ratio in ((0.37, 0.93), (0.44, 0.99), (0.54, 1.19)):          # DexYCB single-walled
        assert not qc.flagged({"removed": removed, "area_ratio": ratio}, thresholds)


def test_the_fallback_is_the_a1_path_at_the_same_seed() -> None:
    """DESIGN 4.8: a flagged request is re-run at 25+25 Euler with fast_postprocess, same seed, QC off."""
    original = _params(seed=7, sparse_structure_steps=8, structured_latent_steps=8, mesh_postprocess_impl="fast",
                       wall_qc="fallback")
    fallback = qc.fallback_params(original)
    assert fallback["seed"] == 7 and fallback["solver"] == "euler"
    assert (fallback["sparse_structure_steps"], fallback["structured_latent_steps"]) == (25, 25)
    assert fallback["mesh_postprocess_impl"] == "fast" and fallback["wall_qc"] == "off"


def test_custom_thresholds_are_carried_and_validated() -> None:
    assert qc.WallQcThresholds(0.30, 1.50).max_removed == 0.30
    for bad, match in (({"max_removed": -0.1}, "wall_qc_max_removed"), ({"max_removed": 1.5}, "wall_qc_max_removed"),
                       ({"min_area_ratio": 0.0}, "wall_qc_min_area_ratio"),
                       ({"min_area_ratio": float("nan")}, "wall_qc_min_area_ratio")):
        with pytest.raises(ValueError, match=match):
            qc.WallQcThresholds(**bad)


# --- 4. ShapeConfig: validation and fail-closed invalid combinations ----------------------------------


def test_the_option_defaults_are_the_designs() -> None:
    cfg = ShapeConfig(backend="sam3d")
    assert cfg.mesh_postprocess_impl == "official" and cfg.wall_qc == "off"
    assert cfg.fast_decimator_min_raw_faces == 250000
    assert (cfg.wall_qc_max_removed, cfg.wall_qc_min_area_ratio) == (0.25, 1.40)
    assert cfg.sam3d_expandable_segments is False
    assert (cfg.sparse_structure_steps, cfg.structured_latent_steps, cfg.solver) == (25, 25, "euler")


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"mesh_postprocess_impl": "nope"}, "mesh_postprocess_impl"),
    ({"wall_qc": "on"}, "wall_qc"),
    ({"fast_decimator_min_raw_faces": -1}, "fast_decimator_min_raw_faces"),
    ({"fast_decimator_min_raw_faces": 1.5}, "fast_decimator_min_raw_faces"),
    ({"wall_qc_max_removed": 2.0}, "wall_qc_max_removed"),
    ({"wall_qc_min_area_ratio": -1.0}, "wall_qc_min_area_ratio"),
    ({"sam3d_expandable_segments": 1}, "sam3d_expandable_segments"),
    # invalid combinations
    ({"wall_qc": "fallback"}, "wall_qc"),                                      # needs the instrumented fast path
    ({"mesh_postprocess_impl": "fast", "mesh_postprocess": False}, "mesh_postprocess"),
])
def test_invalid_options_and_combinations_fail_closed(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ShapeConfig(backend="sam3d", **kwargs)


@pytest.mark.parametrize("field", ["mesh_postprocess_impl", "sam3d_expandable_segments"])
def test_the_sam3d_only_options_are_refused_off_sam3d(field: str) -> None:
    value: Any = "fast" if field == "mesh_postprocess_impl" else True
    with pytest.raises(ValueError, match=field):
        ShapeConfig(backend="trellis", **{field: value})
    ShapeConfig(backend="sam3d", **{field: value})                             # allowed on sam3d


def test_a_valid_fast_path_with_the_qc_is_accepted() -> None:
    cfg = ShapeConfig(backend="sam3d", sparse_structure_steps=8, structured_latent_steps=8,
                      mesh_postprocess_impl="fast", wall_qc="fallback", sam3d_expandable_segments=True)
    assert non_default_shape_options(cfg) == {"mesh_postprocess_impl": "fast", "wall_qc": "fallback",
                                              "sam3d_expandable_segments": True}


# --- 5. the allocator pin ----------------------------------------------------------------------------


def test_the_allocator_is_selectable_at_worker_start_and_default_unchanged() -> None:
    environ = {"PATH": "/bin"}
    assert sam3d_worker_env((), environ) is None
    assert sam3d_worker_env((), environ, alloc_conf=None) is None
    pinned = sam3d_worker_env((), environ, alloc_conf=EXPANDABLE_SEGMENTS)
    assert pinned is not None and pinned[ALLOC_CONF_ENV] == "expandable_segments:True"
    both = sam3d_worker_env(("/a",), environ, alloc_conf=EXPANDABLE_SEGMENTS)
    assert both is not None and both["PYTHONPATH"] == "/a" and both[ALLOC_CONF_ENV] == EXPANDABLE_SEGMENTS


def test_the_config_resolves_the_allocator_setting() -> None:
    assert backend_mod.sam3d_alloc_conf(ShapeConfig(backend="sam3d")) is None
    assert backend_mod.sam3d_alloc_conf(ShapeConfig(backend="sam3d",
                                                    sam3d_expandable_segments=True)) == EXPANDABLE_SEGMENTS


def test_the_allocator_reaches_the_worker_start_and_the_env_identity(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "pipeline.yaml").write_text("_target_: x\n")
    started: list[dict[str, str] | None] = []

    def _client(command: list[str], env: dict[str, str] | None = None, **kwargs: Any) -> None:
        started.append(env)

    monkeypatch.setattr(backend_mod, "Sam3dWorkerClient", _client)
    for flag in (False, True):
        cfg = ShapeConfig(backend="sam3d", sam3d_python="/env/python", sam3d_expandable_segments=flag)
        backend_mod.start_sam3d_worker(cfg, snapshot_fn=lambda *a, **k: str(tmp_path))
    assert started[0] is None or ALLOC_CONF_ENV not in started[0]
    assert started[1] is not None and started[1][ALLOC_CONF_ENV] == EXPANDABLE_SEGMENTS


def _fake_python(tmp_path: Path) -> Path:
    python = tmp_path / "python"
    python.write_bytes(b"#!/bin/sh\n")
    return python


def _probe_identity(monkeypatch: pytest.MonkeyPatch, cfg: ShapeConfig) -> tuple[dict[str, Any],
                                                                               dict[str, str] | None]:
    """`sam3d_env_identity` with the interpreter probe faked: the identity, and the environment the probe ran in."""
    probe_env: list[dict[str, str] | None] = []

    class _Proc:
        returncode = 0
        stdout = '{"executable": "/p", "env": {}}'
        stderr = ""

    def _run(argv: list[str], **kwargs: Any) -> _Proc:
        probe_env.append(kwargs.get("env"))
        return _Proc()

    monkeypatch.setattr(backend_mod.subprocess, "run", _run)
    monkeypatch.setattr(backend_mod, "_gpu_names", lambda: None)
    identity: dict[str, Any] = json.loads(backend_mod.sam3d_env_identity(cfg))
    return identity, probe_env[-1]


def test_the_env_identity_records_the_allocator_only_when_it_is_pinned(monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path) -> None:
    python = _fake_python(tmp_path)
    monkeypatch.delenv(ALLOC_CONF_ENV, raising=False)             # no ambient allocator: the default case
    plain, plain_env = _probe_identity(monkeypatch, ShapeConfig(backend="sam3d", sam3d_python=str(python)))
    pinned, pinned_env = _probe_identity(monkeypatch, ShapeConfig(backend="sam3d", sam3d_python=str(python),
                                                                  sam3d_expandable_segments=True))
    assert "alloc_conf" not in plain                                           # today's identity, unchanged
    assert pinned["alloc_conf"] == EXPANDABLE_SEGMENTS
    assert plain_env is None or ALLOC_CONF_ENV not in plain_env
    assert pinned_env is not None and pinned_env[ALLOC_CONF_ENV] == EXPANDABLE_SEGMENTS       # probe sees it too


def test_an_ambient_allocator_is_recorded_in_the_env_identity_too(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    """A worker that inherits an exported PYTORCH_CUDA_ALLOC_CONF really runs under it, so the
    identity, the cache key and shape.json must not claim the default allocator."""
    python = _fake_python(tmp_path)
    ambient = "expandable_segments:True,max_split_size_mb:128"
    monkeypatch.setenv(ALLOC_CONF_ENV, ambient)
    inherited, inherited_env = _probe_identity(monkeypatch, ShapeConfig(backend="sam3d", sam3d_python=str(python)))
    assert inherited["alloc_conf"] == ambient                     # not silently "the default allocator"
    assert inherited_env is None or inherited_env[ALLOC_CONF_ENV] == ambient   # the probe inherits it, as the worker
    pinned, _ = _probe_identity(monkeypatch, ShapeConfig(backend="sam3d", sam3d_python=str(python),
                                                         sam3d_expandable_segments=True))
    assert pinned["alloc_conf"] == EXPANDABLE_SEGMENTS            # the config's pin is what the worker starts with

    monkeypatch.setenv(ALLOC_CONF_ENV, "")                        # exported but empty: no allocator setting
    empty, _ = _probe_identity(monkeypatch, ShapeConfig(backend="sam3d", sam3d_python=str(python)))
    assert "alloc_conf" not in empty                              # so the default request's cache key is unchanged


# --- 6. the worker: per-request options, the QC decision and the fallback ----------------------------


class _FakeReconstruct:
    """Stands in for `reconstruct_geometry`; records each call and returns scripted postprocess statistics.

    `on_call` is handed the 0-based index of each pass before it runs, so a test can let the pass consume memory the
    way the real one does (section 9).
    """

    def __init__(self, scripted: list[dict[str, Any]], on_call: Any = None) -> None:
        self.scripted, self.calls, self.on_call = scripted, [], on_call  # type: ignore[var-annotated]

    def __call__(self, pipeline: Any, image: np.ndarray, mask: np.ndarray, **kwargs: Any) -> Any:
        if self.on_call is not None:
            self.on_call(len(self.calls))
        self.calls.append(kwargs)
        stats = kwargs.get("stats")
        scripted = self.scripted[min(len(self.calls) - 1, len(self.scripted) - 1)]
        if stats is not None:
            stats.update(scripted)
        n = len(self.calls)
        digests = kwargs.get("digests")
        if digests is not None:                    # as `reconstruct_geometry` does: one set per call, overwriting
            digests.update(slat_feats=f"slat{n}", flexicubes_input=f"flexi{n}", mesh_vertices=f"mv{n}",
                           mesh_faces=f"mf{n}")
        return np.zeros((3, 3)), np.array([[0, 1, 2]]), {"decode": 0.1 * n, "postprocess": 0.2 * n}


def _serve_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, params: dict[str, Any],
               scripted: list[dict[str, Any]], sync: Any = None,
               on_call: Any = None) -> tuple[dict[str, Any], _FakeReconstruct]:
    import io

    from fastdo.shape import sam3d_worker as worker_mod

    fake = _FakeReconstruct(scripted, on_call)
    monkeypatch.setattr(worker_mod, "reconstruct_geometry", fake)
    request_npz, out_npz = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(request_npz, image=np.zeros((4, 4, 3), np.uint8), mask=np.ones((4, 4), bool))
    line = json.dumps({"cmd": "reconstruct", "id": 1, "input": str(request_npz), "output": str(out_npz),
                       "params": params})
    replies = io.StringIO()
    assert serve(object(), io.StringIO(line + "\n"), replies, sync=sync) == 0
    reply: dict[str, Any] = json.loads(replies.getvalue().splitlines()[0])
    assert reply["ok"], reply
    return reply, fake


def test_a_default_request_takes_todays_path_and_reports_todays_reply(monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path) -> None:
    reply, fake = _serve_one(monkeypatch, tmp_path, _params(), [{}])
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["postprocess_impl"] == "official" and call["stats"] is None
    assert set(reply) == {"id", "ok", "phase_seconds", "peak_gpu_gib", "peak_gpu_gib_reserved"}


def test_a_fast_request_routes_the_impl_and_the_threshold_and_reports_its_statistics(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripted = [{"decimator": "fast_simplification", "faces_decimated": 1000, "faces_removed_mincut": 630,
                 "removed": 0.63, "area_ratio": 1.02}]
    reply, fake = _serve_one(monkeypatch, tmp_path, _params(sparse_structure_steps=8, structured_latent_steps=8,
                                                            mesh_postprocess_impl="fast", wall_qc="fallback"),
                             scripted)
    call = fake.calls[0]
    assert call["postprocess_impl"] == "fast" and call["fast_decimator_min_raw_faces"] == 250000
    assert (call["sparse_structure_steps"], call["structured_latent_steps"]) == (8, 8)
    assert len(fake.calls) == 1                                                # no wall: no fallback
    assert reply["postprocess_stats"]["decimator"] == "fast_simplification"
    assert reply["postprocess_stats"]["qc"] == "pass"
    assert reply["postprocess_stats"]["removed"] == pytest.approx(0.63)
    assert "fallback_phase_seconds" not in reply


def test_a_flagged_request_reruns_the_a1_path_and_splits_the_phase_seconds(monkeypatch: pytest.MonkeyPatch,
                                                                           tmp_path: Path) -> None:
    """The primary pass runs through the QC decision; the fallback's phases are recorded apart."""
    walled = {"decimator": "fast_simplification", "faces_decimated": 1000, "faces_removed_mincut": 20,
              "removed": 0.02, "area_ratio": 1.88}
    clean = dict(walled, removed=0.60, area_ratio=1.00)
    reply, fake = _serve_one(monkeypatch, tmp_path,
                             _params(sparse_structure_steps=8, structured_latent_steps=8,
                                     mesh_postprocess_impl="fast", wall_qc="fallback"), [walled, clean])
    assert len(fake.calls) == 2
    primary, fallback = fake.calls
    assert (primary["sparse_structure_steps"], primary["structured_latent_steps"]) == (8, 8)
    assert (fallback["sparse_structure_steps"], fallback["structured_latent_steps"]) == (25, 25)
    assert fallback["seed"] == primary["seed"] and fallback["postprocess_impl"] == "fast"
    assert reply["postprocess_stats"]["qc"] == "fallback"
    assert reply["phase_seconds"] == {"decode": 0.1, "postprocess": 0.2}       # gated: the primary pass only
    assert reply["fallback_phase_seconds"] == {"decode": 0.2, "postprocess": 0.4}


def test_a_fallback_that_flags_again_is_used_and_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DESIGN 4.8: the fallback output is used whether or not it flags again."""
    walled = {"decimator": "vtk", "faces_decimated": 1000, "faces_removed_mincut": 20, "removed": 0.02,
              "area_ratio": 1.88}
    reply, fake = _serve_one(monkeypatch, tmp_path,
                             _params(sparse_structure_steps=8, structured_latent_steps=8,
                                     mesh_postprocess_impl="fast", wall_qc="fallback"), [walled])
    assert len(fake.calls) == 2                                                # exactly one fallback, never a loop
    assert reply["postprocess_stats"]["qc"] == "fallback_flagged"
    assert reply["postprocess_stats"]["fallback"]["removed"] == pytest.approx(0.02)


@pytest.mark.parametrize(("bad", "named"), [
    ({"wall_qc": "fallback"}, "wall_qc"),                                      # fallback without the fast path
    # the worker used to answer this one `ok=True` with `qc="pass"` - a record
    # labelled "fast path, QC passed" for a mesh that no post-process ever touched.
    ({"mesh_postprocess_impl": "fast", "mesh_postprocess": False}, "mesh_postprocess"),
])
def test_the_worker_refuses_an_invalid_option_combination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                          bad: dict[str, Any], named: str) -> None:
    import io

    from fastdo.shape import sam3d_worker as worker_mod

    monkeypatch.setattr(worker_mod, "reconstruct_geometry", _FakeReconstruct([{}]))
    request_npz = tmp_path / "in.npz"
    np.savez(request_npz, image=np.zeros((4, 4, 3), np.uint8), mask=np.ones((4, 4), bool))
    line = json.dumps({"cmd": "reconstruct", "id": 2, "input": str(request_npz), "output": str(tmp_path / "o.npz"),
                       "params": _params(**bad)})
    replies = io.StringIO()
    assert serve(object(), io.StringIO(line + "\n"), replies) == 0
    reply = json.loads(replies.getvalue().splitlines()[0])
    assert reply["ok"] is False and named in reply["error"]
    assert "postprocess_stats" not in reply                                    # no statistics, so no QC record


def test_resolve_shape_options_defaults_every_absent_key() -> None:
    options = resolve_shape_options({})
    assert options.postprocess_impl == "official" and options.wall_qc == "off"
    assert options.fast_decimator_min_raw_faces == 250000
    assert options.thresholds == qc.WallQcThresholds()


def test_the_digest_reply_carries_the_request_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    params = _params(mesh_postprocess_impl="fast", wall_qc="fallback", digest=True)
    reply, _ = _serve_one(monkeypatch, tmp_path, params,
                          [{"decimator": "vtk", "faces_decimated": 10, "faces_removed_mincut": 6, "removed": 0.6,
                            "area_ratio": 1.0}])
    assert reply["digests"]["params"] == request_params_digest(params)


# --- 7. shape.json ------------------------------------------------------------------------------------


class _StatsBackend:
    """A shape backend that reports postprocess statistics, as the SAM 3D client does after a fast request."""

    def __init__(self, stats: dict[str, Any] | None) -> None:
        box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        self.vertices = np.asarray(box.vertices)
        self.faces = np.asarray(box.faces)
        self.last_postprocess_stats = stats

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        return ShapeResult(self.vertices, self.faces, None, {"decode": 0.25})


def _run_shape_worker(tmp_path: Path, cfg: ShapeConfig, stats: dict[str, Any] | None) -> dict[str, Any]:
    from fastdo.engine.workers import make_shape_worker
    from tests.engine.fakes import random_frames, write_decode

    clip = tmp_path / "clip"
    frames = random_frames(3, h=16, w=20, seed=0)
    write_decode(clip, frames)
    masks = np.zeros((3, 16, 20), dtype=bool)
    masks[:, 2:8, 3:12] = True
    np.savez(clip / "mask.npz", masks=masks, frame_idx=np.arange(3, dtype=np.int64),
             keyframe_index=np.array(1, dtype=np.int64))
    make_shape_worker(cfg, backend=_StatsBackend(stats), sdf_resolution=16, n_samples=64)("clip", clip)
    return read_shape_meta(clip)


def test_shape_json_of_a_default_run_keeps_todays_key_set(tmp_path: Path) -> None:
    meta = _run_shape_worker(tmp_path, ShapeConfig(), None)
    assert not set(meta) & set(SHAPE_OPTION_DEFAULTS)
    assert "postprocess_impl" not in meta and "wall_qc" not in meta and "decimator" not in meta


def test_shape_json_records_every_option_and_the_qc_outcome_when_one_is_set(tmp_path: Path) -> None:
    cfg = ShapeConfig(backend="sam3d", sparse_structure_steps=8, structured_latent_steps=8,
                      mesh_postprocess_impl="fast", wall_qc="fallback", sam3d_expandable_segments=True)
    stats = {"decimator": "fast_simplification", "faces_decimated": 1000, "faces_removed_mincut": 20,
             "removed": 0.02, "area_ratio": 1.88, "qc": "fallback"}
    meta = _run_shape_worker(tmp_path, cfg, stats)
    assert meta["mesh_postprocess_impl"] == "fast" and meta["wall_qc"] == "fallback"
    assert meta["fast_decimator_min_raw_faces"] == 250000                      # the kept default is recorded too
    assert (meta["wall_qc_max_removed"], meta["wall_qc_min_area_ratio"]) == (0.25, 1.40)
    assert meta["sam3d_expandable_segments"] is True
    assert meta["decimator"] == "fast_simplification" and meta["qc"] == "fallback"
    assert meta["removed"] == pytest.approx(0.02) and meta["area_ratio"] == pytest.approx(1.88)
    assert (meta["sparse_structure_steps"], meta["structured_latent_steps"]) == (8, 8)


def test_the_hydra_config_exposes_every_new_option() -> None:
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
    shape = OmegaConf.load(root).bench.shape
    for name in SHAPE_OPTION_DEFAULTS:
        assert name in shape, name
    assert os.path.isfile(root)


# --- 8. fallback records, error coercion, digests and decimator routing --------------------------------------


class _ScriptedWorker(Sam3dWorkerClient):
    """A `Sam3dWorkerClient` answering one scripted reply, with the real `reconstruct`: no process, no GPU."""

    def __init__(self, reply: dict[str, Any], workdir: Path) -> None:
        self._workdir = Path(workdir)
        self._next_id = 0
        self._reply: dict[str, Any] = reply
        self._pending: dict[str, Any] = {}
        self.request_timeout_s = 1.0
        self.last_reply = {}
        self.last_peak_gpu_gib = None
        self.last_peak_gpu_gib_reserved = None
        self.last_digests = None
        self.last_postprocess_stats = None
        self.last_fallback_peak_gpu_gib = None
        self.last_fallback_peak_gpu_gib_reserved = None

    def _send(self, message: dict[str, Any]) -> None:
        np.savez(message["output"], vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
        self._pending = dict(self._reply, id=message["id"])

    def _read_reply(self, timeout_s: float, what: str) -> dict[str, Any]:
        return self._pending


_GATED = {"pointmap": 1.25, "preprocess": 0.5, "sparse_structure_flow": 2.0, "structured_latent_flow": 3.0,
          "decode": 1.5, "postprocess": 0.75}
_FALLBACK = {"pointmap": 1.1, "preprocess": 0.4, "sparse_structure_flow": 5.5, "structured_latent_flow": 6.25,
             "decode": 1.4, "postprocess": 0.9}


def test_the_client_keeps_a_fallbacks_phases_out_of_the_gated_phase_seconds(tmp_path: Path) -> None:
    """Final review F1: G-time and t_eff are computed from worker seconds and RR2-M1 charges the fallback only
    through t_eff, so `ShapeResult.phase_seconds` must stay the gated pass and still sum to the gated seconds."""
    client = _ScriptedWorker({"ok": True, "phase_seconds": _GATED, "fallback_phase_seconds": _FALLBACK,
                              "postprocess_stats": {"qc": "fallback"}}, tmp_path)
    result = client.reconstruct(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool), _params())

    # the pattern DESIGN 8.2's EVAL helper uses (tests/shape/sam3d_kernel_pin_gate.py): gated, not inflated
    assert sum(result.phase_seconds.values()) == pytest.approx(sum(_GATED.values()))
    assert result.phase_seconds == pytest.approx(_GATED)
    assert not [name for name in result.phase_seconds if name.startswith("fallback")]
    fallback = result.fallback_phase_seconds                      # ... and reachable, separately, for t_eff
    assert fallback is not None and fallback == pytest.approx(_FALLBACK)
    assert sum(fallback.values()) == pytest.approx(sum(_FALLBACK.values()))


def test_a_request_without_a_fallback_reports_no_fallback_phases(tmp_path: Path) -> None:
    client = _ScriptedWorker({"ok": True, "phase_seconds": _GATED}, tmp_path)
    result = client.reconstruct(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool), _params())
    assert result.phase_seconds == pytest.approx(_GATED)
    assert result.fallback_phase_seconds is None                  # today's runs, unchanged
    assert ShapeResult(np.zeros((3, 3)), np.array([[0, 1, 2]])).fallback_phase_seconds is None


_REFUSED_OPTIONS: tuple[dict[str, Any], ...] = (
    {"fast_decimator_min_raw_faces": -5},
    {"fast_decimator_min_raw_faces": "250000"},
    {"fast_decimator_min_raw_faces": 1.9},
    {"fast_decimator_min_raw_faces": 250000.0},
    {"fast_decimator_min_raw_faces": True},
    {"wall_qc_max_removed": "0.25"},
    {"wall_qc_max_removed": 2.0},
    {"wall_qc_max_removed": -0.1},
    {"wall_qc_min_area_ratio": "1.40"},
    {"wall_qc_min_area_ratio": 0.0},
    {"mesh_postprocess_impl": 5},
    {"wall_qc": 1},
    # `ShapeConfig` refuses the fast path with the post-process off; the worker
    # boundary accepted it and then reported `qc="pass"` for a mesh that was never post-processed.
    {"mesh_postprocess": False},
)


@pytest.mark.parametrize("bad", _REFUSED_OPTIONS,
                         ids=[f"{next(iter(b))}={next(iter(b.values()))!r}" for b in _REFUSED_OPTIONS])
def test_the_worker_boundary_refuses_exactly_what_shapeconfig_refuses(bad: dict[str, Any]) -> None:
    """`resolve_shape_options` coerced where `ShapeConfig` refuses, and `1.9` silently truncated
    to `1` routes every mesh to fast_simplification - A0v's VTK-only mode inverted. Same value, same named error."""
    settings = {"mesh_postprocess_impl": "fast"} | bad
    with pytest.raises(ValueError) as config_error:
        ShapeConfig(backend="sam3d", **settings)
    with pytest.raises(ValueError) as worker_error:
        resolve_shape_options(_params(**settings))
    assert str(worker_error.value) == str(config_error.value)
    assert next(iter(bad)) in str(worker_error.value)


def test_resolve_shape_options_still_accepts_every_valid_value() -> None:
    options = resolve_shape_options(_params(mesh_postprocess_impl="fast", wall_qc="fallback",
                                            fast_decimator_min_raw_faces=10**12, wall_qc_max_removed=0.30,
                                            wall_qc_min_area_ratio=1.5))
    assert options.postprocess_impl == "fast" and options.wall_qc == "fallback"
    assert options.fast_decimator_min_raw_faces == 10**12
    assert (options.thresholds.max_removed, options.thresholds.min_area_ratio) == (0.30, 1.5)
    assert resolve_shape_options({"fast_decimator_min_raw_faces": 0}).fast_decimator_min_raw_faces == 0
    assert resolve_shape_options({"wall_qc_max_removed": 1}).thresholds.max_removed == 1   # an int is a number


_FALLBACK_STATS: dict[str, Any] = {
    "decimator": "fast_simplification", "faces_raw": 306_000, "faces_decimated": 1000, "faces_removed_mincut": 20,
    "removed": 0.02, "area_ratio": 1.88, "qc": "fallback",
    "fallback": {"decimator": "vtk", "faces_raw": 104_536, "faces_decimated": 900, "faces_removed_mincut": 540,
                 "removed": 0.60, "area_ratio": 1.01}}


def _fallback_cfg() -> ShapeConfig:
    return ShapeConfig(backend="sam3d", sparse_structure_steps=8, structured_latent_steps=8,
                       mesh_postprocess_impl="fast", wall_qc="fallback")


def test_shape_json_after_a_fallback_describes_the_mesh_it_shipped(tmp_path: Path) -> None:
    """The kept mesh is the fallback's, so the flat statistics beside
    `mesh.obj`, `n_faces` and `n_vertices` must be the fallback's - and `_POSTPROCESS_STAT_KEYS`, copying only flat
    keys, dropped the nested `fallback` block entirely."""
    meta = _run_shape_worker(tmp_path, _fallback_cfg(), _FALLBACK_STATS)
    assert meta["decimator"] == "vtk"                             # G-topo (c) reads the decimator from shape.json
    assert meta["faces_raw"] == 104_536 and meta["faces_decimated"] == 900
    assert meta["faces_removed_mincut"] == 540
    assert (meta["removed"], meta["area_ratio"]) == pytest.approx((0.60, 1.01))
    assert meta["qc"] == "fallback"


def test_shape_json_keeps_the_discarded_primarys_statistics_under_primary(tmp_path: Path) -> None:
    meta = _run_shape_worker(tmp_path, _fallback_cfg(), _FALLBACK_STATS)
    assert meta["primary_decimator"] == "fast_simplification"
    assert meta["primary_faces_raw"] == 306_000 and meta["primary_faces_decimated"] == 1000
    assert meta["primary_faces_removed_mincut"] == 20
    assert (meta["primary_removed"], meta["primary_area_ratio"]) == pytest.approx((0.02, 1.88))
    assert "primary_qc" not in meta                               # one outcome per request, not one per pass


def test_shape_json_of_a_passing_fast_request_has_no_primary_keys(tmp_path: Path) -> None:
    stats = {k: v for k, v in _FALLBACK_STATS.items() if k != "fallback"} | {"qc": "pass"}
    meta = _run_shape_worker(tmp_path, _fallback_cfg(), stats)
    assert meta["decimator"] == "fast_simplification" and meta["qc"] == "pass"
    assert not [key for key in meta if key.startswith("primary_")]


def test_a_fallbacks_digests_keep_the_discarded_primarys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`serve` hands one digests dict to both calls, so a fallback silently
    replaced the primary's mesh digests while `digests["params"]` still named the original request."""
    walled = {"decimator": "fast_simplification", "faces_decimated": 1000, "faces_removed_mincut": 20,
              "removed": 0.02, "area_ratio": 1.88}
    clean = dict(walled, removed=0.60, area_ratio=1.00)
    params = _params(sparse_structure_steps=8, structured_latent_steps=8, mesh_postprocess_impl="fast",
                     wall_qc="fallback", digest=True)
    reply, fake = _serve_one(monkeypatch, tmp_path, params, [walled, clean])
    assert len(fake.calls) == 2
    digests = reply["digests"]
    for name in ("slat_feats", "flexicubes_input", "mesh_vertices", "mesh_faces"):
        assert digests[name].endswith("2"), name                  # the fallback: the mesh that was stored
        assert digests[f"primary_{name}"].endswith("1"), name     # the gated pass that was discarded
    assert digests["params"] == request_params_digest(params)     # the arm that was asked for
    assert "primary_params" not in digests


def test_a_request_without_a_fallback_has_no_primary_digests(monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: Path) -> None:
    params = _params(mesh_postprocess_impl="fast", wall_qc="fallback", digest=True)
    reply, _ = _serve_one(monkeypatch, tmp_path, params,
                          [{"decimator": "vtk", "faces_decimated": 10, "faces_removed_mincut": 6, "removed": 0.6,
                            "area_ratio": 1.0}])
    assert not [key for key in reply["digests"] if key.startswith("primary_")]


def _recording_decimate(calls: list[tuple[str, int, int]]) -> Any:
    """A `fast_postprocess.decimate` that records how it was routed and returns its input unchanged."""

    def decimate(vertices: np.ndarray, faces: np.ndarray, decimator: str,
                 simplify_ratio: float) -> tuple[np.ndarray, np.ndarray]:
        calls.append((decimator, len(np.asarray(faces)), len(np.asarray(vertices))))
        return np.asarray(vertices, np.float32), np.asarray(faces, np.int64)

    return decimate


@pytest.mark.parametrize(("extra_vertices", "min_raw_faces", "expected"),
                         [(0, 10, "fast_simplification"), (20, 20, "vtk")])
def test_fast_postprocess_routes_its_decimator_by_the_raw_face_count(monkeypatch: pytest.MonkeyPatch,
                                                                     extra_vertices: int, min_raw_faces: int,
                                                                     expected: str) -> None:
    """`choose_decimator` is well tested, but its call site inside
    `fast_postprocess` was not, so routing by vertex count instead of raw face count passed the whole suite. Each
    case puts the two counts on opposite sides of the threshold, so a vertex-count router gets both backwards.

    The GPU half stays out of reach on CPU: `fill_holes=False`, and the decimator is faked (neither pyvista nor
    fast_simplification is installed in this environment).
    """
    vertices, faces = _box(1.0)                                   # 8 vertices, 12 faces
    if extra_vertices:                                            # unreferenced interior points: 28 vertices, 12 faces
        offsets = np.linspace(-0.2, 0.2, extra_vertices)
        vertices = np.concatenate([vertices, np.stack([offsets, offsets * 0.5, offsets * -0.25], axis=1)])
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(fp, "decimate", _recording_decimate(calls))
    stats: dict[str, Any] = {}
    out_v, out_f = fp.fast_postprocess(vertices, faces, fill_holes=False, min_raw_faces=min_raw_faces, stats=stats)

    assert calls == [(expected, len(faces), len(vertices))]       # the body chose, and it chose on the face count
    assert stats["decimator"] == expected
    assert stats["faces_raw"] == len(faces) == 12
    assert (len(out_v), len(out_f)) == (len(vertices), len(faces))
    assert stats["faces_decimated"] == 12 and stats["faces_final"] == 12
    assert stats["faces_removed_mincut"] == 0 and stats["removed"] == 0.0
    assert stats["area_ratio"] == pytest.approx(1.0, abs=1e-6)    # a box is its own hull
    assert "decimate" in stats["seconds"]


# --- 9. peaks, t_eff inputs and the post-process-off refusal -----------------------------------------


class _PeakCounters:
    """A stand-in for the CUDA peak counters: `reset` opens a new window, and each pass raises its high-water mark.

    `torch.cuda.max_memory_reserved()` is monotone inside a window and only `reset_peak_memory_stats()` starts a new
    one, so a window that spans both passes reports the LARGER of them - the defect tested here. `allocate` is
    the fake `reconstruct_geometry`'s `on_call` hook, so the passes consume memory in the order `serve` runs them.
    """

    def __init__(self, per_pass: list[float]) -> None:
        self.per_pass, self.window = per_pass, 0.0

    def allocate(self, index: int) -> None:
        self.window = max(self.window, self.per_pass[min(index, len(self.per_pass) - 1)])

    def peak_memory(self, device_stats: bool, reset: bool) -> dict[str, float | None]:
        if not device_stats:
            return {"peak_gpu_gib": None, "peak_gpu_gib_reserved": None}
        if reset:
            self.window = 0.0
        return {"peak_gpu_gib": self.window * 0.9, "peak_gpu_gib_reserved": self.window}


def _flagged_params() -> dict[str, Any]:
    return _params(sparse_structure_steps=8, structured_latent_steps=8, mesh_postprocess_impl="fast",
                   wall_qc="fallback")


_WALLED = {"decimator": "fast_simplification", "faces_decimated": 1000, "faces_removed_mincut": 20,
           "removed": 0.02, "area_ratio": 1.88}
_CLEAN = dict(_WALLED, removed=0.60, area_ratio=1.00)
_GATED_PEAK, _FALLBACK_PEAK = 18.0, 20.5              # the fallback is 25+25 on the A1 path: the higher arm


def test_a_flagged_requests_peak_is_the_gated_passs_and_not_the_fallbacks(monkeypatch: pytest.MonkeyPatch,
                                                                          tmp_path: Path) -> None:
    """The memory twin of the phase-seconds split.

    `serve` reset the peak once, before the primary pass, and read it after a possible fallback, so a flagged row
    carried an A1-path peak under the 8+8 arm's name. That is the quantity G-mem reads, and G-mem is a
    non-inferiority gate: a gated peak inflated by the fallback compares the wrong arm against the control.
    """
    from fastdo.shape import sam3d_worker as worker_mod

    peaks = _PeakCounters([_GATED_PEAK, _FALLBACK_PEAK])
    monkeypatch.setattr(worker_mod, "_peak_memory", peaks.peak_memory)
    reply, fake = _serve_one(monkeypatch, tmp_path, _flagged_params(), [_WALLED, _CLEAN],
                             sync=lambda: None, on_call=peaks.allocate)

    assert len(fake.calls) == 2 and reply["postprocess_stats"]["qc"] == "fallback"
    assert reply["peak_gpu_gib_reserved"] == _GATED_PEAK           # the gated pass alone: what G-mem reads
    assert reply["peak_gpu_gib_reserved"] != _FALLBACK_PEAK        # ... never the high-water mark over both passes
    assert reply["fallback_peak_gpu_gib_reserved"] == _FALLBACK_PEAK          # beside it, as F1 did for seconds
    assert reply["peak_gpu_gib"] == pytest.approx(_GATED_PEAK * 0.9)
    assert reply["fallback_peak_gpu_gib"] == pytest.approx(_FALLBACK_PEAK * 0.9)


def test_a_passing_requests_peak_is_reported_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No fallback, no second window: today's reply keys, and no fallback peak to misread."""
    from fastdo.shape import sam3d_worker as worker_mod

    peaks = _PeakCounters([_GATED_PEAK])
    monkeypatch.setattr(worker_mod, "_peak_memory", peaks.peak_memory)
    reply, fake = _serve_one(monkeypatch, tmp_path, _flagged_params(), [_CLEAN],
                             sync=lambda: None, on_call=peaks.allocate)

    assert len(fake.calls) == 1 and reply["postprocess_stats"]["qc"] == "pass"
    assert reply["peak_gpu_gib_reserved"] == _GATED_PEAK
    assert not [key for key in reply if key.startswith("fallback_peak")]


def test_the_client_exposes_the_fallbacks_peak_beside_the_gated_one(tmp_path: Path) -> None:
    client = _ScriptedWorker({"ok": True, "phase_seconds": _GATED, "fallback_phase_seconds": _FALLBACK,
                              "peak_gpu_gib": _GATED_PEAK * 0.9, "peak_gpu_gib_reserved": _GATED_PEAK,
                              "fallback_peak_gpu_gib": _FALLBACK_PEAK * 0.9,
                              "fallback_peak_gpu_gib_reserved": _FALLBACK_PEAK,
                              "postprocess_stats": {"qc": "fallback"}}, tmp_path)
    client.reconstruct(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool), _flagged_params())

    assert client.last_peak_gpu_gib_reserved == _GATED_PEAK        # G-mem's input stays the gated pass
    assert client.last_fallback_peak_gpu_gib_reserved == _FALLBACK_PEAK
    assert client.last_peak_gpu_gib == pytest.approx(_GATED_PEAK * 0.9)
    assert client.last_fallback_peak_gpu_gib == pytest.approx(_FALLBACK_PEAK * 0.9)


def _measured_row(reply: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    client = _ScriptedWorker(reply, tmp_path)
    result = client.reconstruct(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool), _flagged_params())
    return gate.request_measurements(result, client)


def test_an_eval_row_carries_both_passes_so_t_eff_has_a_source(tmp_path: Path) -> None:
    """The helper the evaluation imports had the gated numerator and
    nothing to charge t_eff with - it never read `fallback_phase_seconds` or `postprocess_stats`. t_eff is
    the ONE place a fallback's seconds are charged, so a T-block harness built on it could not compute one."""
    import statistics

    flagged = _measured_row({"ok": True, "phase_seconds": _GATED, "fallback_phase_seconds": _FALLBACK,
                             "peak_gpu_gib_reserved": _GATED_PEAK,
                             "fallback_peak_gpu_gib_reserved": _FALLBACK_PEAK,
                             "postprocess_stats": dict(_WALLED, qc="fallback",
                                                       fallback=dict(_CLEAN))}, tmp_path)
    passing = _measured_row({"ok": True, "phase_seconds": _GATED, "peak_gpu_gib_reserved": _GATED_PEAK,
                             "postprocess_stats": dict(_CLEAN, qc="pass")}, tmp_path)

    assert flagged["worker_seconds"] == pytest.approx(sum(_GATED.values()))    # gated: G-time's numerator
    assert flagged["fallback_phase_seconds"] == pytest.approx(_FALLBACK)
    assert flagged["fallback_worker_seconds"] == pytest.approx(sum(_FALLBACK.values()))
    assert flagged["qc"] == "fallback" and passing["qc"] == "pass"
    assert flagged["peak_gpu_gib_reserved"] == _GATED_PEAK                     # G-mem's input, gated
    assert flagged["fallback_peak_gpu_gib_reserved"] == _FALLBACK_PEAK
    assert passing["fallback_worker_seconds"] is None and passing["fallback_phase_seconds"] is None
    assert passing["fallback_peak_gpu_gib_reserved"] is None

    # ... and a block of these rows produces a t_eff: median gated seconds + flag rate x A1's median.
    rows = [flagged, passing]
    flagged_rows = [r for r in rows if r["qc"] in ("fallback", "fallback_flagged")]
    flag_rate = len(flagged_rows) / len(rows)
    a1_median = statistics.median([r["fallback_worker_seconds"] for r in flagged_rows])
    t_eff = statistics.median([r["worker_seconds"] for r in rows]) + flag_rate * a1_median
    assert t_eff == pytest.approx(sum(_GATED.values()) + 0.5 * sum(_FALLBACK.values()))
    assert t_eff > statistics.median([r["worker_seconds"] for r in rows])      # the fallback is charged, once
