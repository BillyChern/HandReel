"""The default-off SAM 3D sampler options `sparse_structure_shortcut` and `structured_latent_solver`.

They carry D4 (the sparse-structure flow through the checkpoint's shortcut path at 4 steps) and D4S12 (D4 plus
the structured-latent flow at 12 steps of the right-limited AB2) from `ShapeConfig` through the request to the
worker's `reconstruct_geometry`, by wrapping
`pipeline.sample_sparse_structure` / `pipeline.sample_slat`. CPU tests with the fake pipeline: a default request must
be unchanged on the wire, in its digest, its cache key, its shape.json and its calls; a set option must reach the call.
Whether the real meshes equal the reference ones is a GPU check.

`sparse_structure_solver` is tested in `test_sam3d_ss_solver.py`; here it is only checked to stay off (and
absent) in every request, digest, record and production config these tests build.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import trimesh

from fastdo.engine.shape_cache import SHAPE_CODE_FILES, ShapeCacheKeyer
from fastdo.engine.workers_shape import read_shape_meta
from fastdo.shape import ShapeConfig, ShapeResult
from fastdo.shape import sam3d_multistep as ms
from fastdo.shape import sam3d_worker as worker_mod
from fastdo.shape import wall_qc as qc
from fastdo.shape.backend import (
    SAMPLER_OPTION_DEFAULTS,
    SHAPE_OPTION_DEFAULTS,
    non_default_sampler_options,
    non_default_shape_options,
)
from fastdo.shape.sam3d_backend import Sam3dBackend
from fastdo.shape.sam3d_worker import (
    AB2R_RIGHT_LIMIT_EPS,
    reconstruct_geometry,
    request_params_digest,
    resolve_shape_options,
    serve,
)
from tests.shape.fake_sam3d import FakeSam3dPipeline

CONFIG_YAML = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
D4 = {"sparse_structure_steps": 4, "sparse_structure_shortcut": True}
D4S12 = D4 | {"structured_latent_steps": 12, "structured_latent_solver": "ab2r"}
NEW_KEYS = ("sparse_structure_shortcut", "structured_latent_solver", "sparse_structure_solver")


def _params(**overrides: Any) -> dict[str, Any]:
    return {"seed": 42, "sparse_structure_steps": 25, "structured_latent_steps": 25, "solver": "euler",
            "mesh_postprocess": True} | overrides


def _pre157_digest(params: dict[str, Any]) -> str:
    """`request_params_digest` as it was before the sampler options were added (the 11 keys it hashed then, all at their defaults)."""
    resolved = {"seed": int(params["seed"]), "sparse_structure_steps": int(params["sparse_structure_steps"]),
                "structured_latent_steps": int(params["structured_latent_steps"]), "solver": str(params["solver"]),
                "mesh_postprocess": bool(params["mesh_postprocess"]), "deterministic": False,
                "mesh_postprocess_impl": "official", "fast_decimator_min_raw_faces": 250000, "wall_qc": "off",
                "wall_qc_max_removed": 0.25, "wall_qc_min_area_ratio": 1.40}
    return hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _production_shape_config() -> ShapeConfig:
    from omegaconf import OmegaConf

    from fastdo.bench.bench_stages import shape_config
    return shape_config(OmegaConf.load(CONFIG_YAML))


# --- 1. defaults change nothing -----------------------------------------------------------------------------


def test_the_options_are_off_by_default_and_absent_from_a_default_request() -> None:
    cfg = ShapeConfig(backend="sam3d")
    assert (cfg.sparse_structure_shortcut, cfg.structured_latent_solver) == (False, None)
    assert non_default_sampler_options(cfg) == {} and non_default_shape_options(cfg) == {}
    assert not set(Sam3dBackend(cfg).sampling_params) & set(NEW_KEYS)
    options = resolve_shape_options(_params())
    assert (options.ss_shortcut, options.slat_solver) == (False, None)
    kwargs = worker_mod._call_kwargs(_params(), options, None, None, None)
    assert (kwargs["sparse_structure_shortcut"], kwargs["structured_latent_solver"]) == (False, None)


def test_the_production_config_is_s12_and_its_request_carries_only_the_solver() -> None:
    """S12 is the production setting: SLAT at 12 right-limited AB2 steps, the shortcut path still off."""
    cfg = _production_shape_config()
    assert (cfg.sparse_structure_shortcut, cfg.structured_latent_solver) == (False, "ab2r")
    assert cfg.sparse_structure_solver is None                  # opt-in until its gates pass
    params = Sam3dBackend(cfg).sampling_params
    assert {k: params[k] for k in NEW_KEYS if k in params} == {"structured_latent_solver": "ab2r"}
    assert (params["sparse_structure_steps"], params["structured_latent_steps"], params["solver"]) == (25, 12, "euler")


def test_a_default_request_keeps_its_pre157_digest() -> None:
    assert request_params_digest(_params()) == _pre157_digest(_params())
    assert request_params_digest(_params(sparse_structure_shortcut=False, structured_latent_solver=None)) == \
        _pre157_digest(_params())


def test_the_sampler_defaults_are_not_shape_option_defaults() -> None:
    """shape.json records every SHAPE_OPTION_DEFAULTS key once any is set (production sets them), so the sampler
    options must live apart or a production shape.json would gain two keys."""
    assert not set(SAMPLER_OPTION_DEFAULTS) & set(SHAPE_OPTION_DEFAULTS)
    assert SAMPLER_OPTION_DEFAULTS == {"sparse_structure_shortcut": False, "structured_latent_solver": None,
                                       "sparse_structure_solver": None}


def test_the_hydra_config_exposes_both_options_at_the_promoted_values() -> None:
    from omegaconf import OmegaConf

    shape = OmegaConf.load(CONFIG_YAML).bench.shape
    assert shape.sparse_structure_shortcut is False and shape.structured_latent_solver == "ab2r"
    assert shape.structured_latent_steps == 12
    assert shape.sparse_structure_solver is None and "sparse_structure_steps" not in shape   # opt-in, SS at 25


# --- 2. set, they reach the request, the digest and the call ------------------------------------------------


@pytest.mark.parametrize(("arm", "expected"), [("D4", {"sparse_structure_shortcut": True}),
                                               ("D4S12", {"sparse_structure_shortcut": True,
                                                          "structured_latent_solver": "ab2r"})])
def test_each_arm_reaches_the_request_and_the_worker_call(arm: str, expected: dict[str, Any]) -> None:
    settings = {"D4": D4, "D4S12": D4S12}[arm]
    cfg = ShapeConfig(backend="sam3d", **settings)
    assert non_default_sampler_options(cfg) == expected
    params = Sam3dBackend(cfg).sampling_params
    assert {k: params[k] for k in NEW_KEYS if k in params} == expected
    assert params["sparse_structure_steps"] == 4
    assert params["structured_latent_steps"] == settings.get("structured_latent_steps", 25)
    options = resolve_shape_options(params)
    kwargs = worker_mod._call_kwargs(params, options, None, None, None)
    assert kwargs["sparse_structure_shortcut"] is True
    assert kwargs["structured_latent_solver"] == expected.get("structured_latent_solver")
    assert request_params_digest(params) != request_params_digest(params | {"sparse_structure_shortcut": False,
                                                                             "structured_latent_solver": None})


def test_each_option_moves_the_digest_on_its_own() -> None:
    base = request_params_digest(_params())
    digests = {request_params_digest(_params(sparse_structure_shortcut=True)),
               request_params_digest(_params(structured_latent_solver="ab2r")),
               request_params_digest(_params(sparse_structure_shortcut=True, structured_latent_solver="ab2r"))}
    assert base not in digests and len(digests) == 3


_REFUSED: tuple[dict[str, Any], ...] = (
    {"sparse_structure_shortcut": 1}, {"sparse_structure_shortcut": "true"}, {"sparse_structure_shortcut": None},
    {"structured_latent_solver": "ab2"}, {"structured_latent_solver": "euler"}, {"structured_latent_solver": 2},
    {"sparse_structure_shortcut": True, "solver": "midpoint"}, {"structured_latent_solver": "ab2r", "solver": "midpoint"},
)


@pytest.mark.parametrize("bad", _REFUSED, ids=[repr(b) for b in _REFUSED])
def test_the_worker_boundary_refuses_exactly_what_shapeconfig_refuses(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError) as config_error:
        ShapeConfig(backend="sam3d", **bad)
    with pytest.raises(ValueError) as worker_error:
        resolve_shape_options(_params(**bad))
    assert str(worker_error.value) == str(config_error.value)
    assert any(k in str(worker_error.value) for k in NEW_KEYS)


@pytest.mark.parametrize("backend", ["trellis", "template"])
def test_the_options_are_refused_off_sam3d(backend: str) -> None:
    extra = {"mesh_postprocess": False} if backend == "template" else {}
    for settings in ({"sparse_structure_shortcut": True}, {"structured_latent_solver": "ab2r"}):
        with pytest.raises(ValueError, match="allowed only for sam3d"):
            ShapeConfig(backend=backend, **settings, **extra)
    ShapeConfig(backend=backend, sparse_structure_shortcut=False, structured_latent_solver=None, **extra)


# --- 3. the calls the driver makes ----------------------------------------------------------------------------


def _pipeline() -> tuple[FakeSam3dPipeline, dict[str, Any]]:
    """The fake pipeline with the SLAT generator attributes `cfg_regime` reads, and a spy recording the SLAT
    generator's update at the moment `sample_slat` runs."""
    pipe = FakeSam3dPipeline()
    gen = pipe.models["slat_generator"]
    gen.reverse_fn = SimpleNamespace(interval=[0.0, 500.0])
    gen.time_scale = 1000.0
    pipe.slat_cfg_strength = 1.0
    seen: dict[str, Any] = {}
    original = pipe.sample_slat

    def sample_slat(*args: Any, **kwargs: Any) -> Any:
        seen["solver"] = gen._solver
        return original(*args, **kwargs)

    pipe.sample_slat = sample_slat  # type: ignore[method-assign]
    return pipe, seen


def _reconstruct(pipe: FakeSam3dPipeline, **overrides: Any) -> Any:
    image = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:15, 8:22] = True
    params: dict[str, Any] = {"seed": 0, "sparse_structure_steps": 25, "structured_latent_steps": 25,
                              "solver": "euler", "mesh_postprocess": False} | overrides
    return reconstruct_geometry(pipe, image, mask, **params)


def _events(pipe: FakeSam3dPipeline, name: str) -> list[dict[str, Any]]:
    return [e for e in pipe.events if e["event"] == name]


def test_a_default_call_is_todays_call() -> None:
    pipe, seen = _pipeline()
    _reconstruct(pipe)
    (ss,), (slat,) = _events(pipe, "sparse_structure"), _events(pipe, "structured_latent")
    assert (ss["steps"], ss["use_distillation"]) == (25, False)
    assert (slat["steps"], slat["use_distillation"]) == (25, False)
    assert seen["solver"] == "solver:euler"                     # the generator's own update, untouched


def test_the_shortcut_samples_the_sparse_structure_via_the_distilled_path() -> None:
    pipe, seen = _pipeline()
    _reconstruct(pipe, sparse_structure_steps=4, sparse_structure_shortcut=True)
    (ss,), (slat,) = _events(pipe, "sparse_structure"), _events(pipe, "structured_latent")
    assert (ss["steps"], ss["use_distillation"]) == (4, True)
    assert (slat["steps"], slat["use_distillation"]) == (25, False)       # the SLAT flow stays today's
    assert seen["solver"] == "solver:euler"


def test_ab2r_installs_the_right_limited_ab2_for_the_slat_flow_only_and_restores_it() -> None:
    pipe, seen = _pipeline()
    gen = pipe.models["slat_generator"]
    _reconstruct(pipe, sparse_structure_steps=4, sparse_structure_shortcut=True, structured_latent_steps=12,
                 structured_latent_solver="ab2r")
    solver = seen["solver"]
    assert isinstance(solver, ms.AdamsBashforth2)
    assert solver.right_limit_eps == AB2R_RIGHT_LIMIT_EPS == 1e-6
    assert solver.regime is not None and solver.regime(0.4) is True and solver.regime(0.6) is False  # 0..500 on t*1000
    (slat,) = _events(pipe, "structured_latent")
    assert (slat["steps"], slat["use_distillation"]) == (12, False)
    assert gen._solver == "solver:euler" and pipe.models["ss_generator"]._solver == "solver:euler"


def test_ab2r_restores_the_update_when_the_slat_flow_fails() -> None:
    pipe, _ = _pipeline()
    pipe.fail_in = "structured_latent"
    with pytest.raises(RuntimeError, match="injected"):
        _reconstruct(pipe, structured_latent_solver="ab2r")
    assert pipe.models["slat_generator"]._solver == "solver:euler"


def test_the_driver_refuses_an_unknown_slat_solver() -> None:
    pipe, _ = _pipeline()
    with pytest.raises(ValueError, match="structured_latent_solver"):
        _reconstruct(pipe, structured_latent_solver="ab3")


def test_the_worker_passes_the_options_from_the_request_to_the_driver(monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_reconstruct(pipeline: Any, image: Any, mask: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return np.zeros((3, 3)), np.zeros((1, 3), np.int64), {"sparse_structure_flow": 1.0}

    monkeypatch.setattr(worker_mod, "reconstruct_geometry", fake_reconstruct)
    request_npz = tmp_path / "in.npz"
    np.savez(request_npz, image=np.zeros((4, 4, 3), np.uint8), mask=np.ones((4, 4), bool))
    lines = [json.dumps({"cmd": "reconstruct", "id": i, "input": str(request_npz),
                         "output": str(tmp_path / f"out{i}.npz"), "params": p})
             for i, p in enumerate((_params(), _params(**D4), _params(**D4S12)))]
    replies = io.StringIO()
    assert serve(object(), io.StringIO("\n".join(lines) + "\n"), replies) == 0
    assert all(json.loads(r)["ok"] for r in replies.getvalue().splitlines())
    got = [(c["sparse_structure_steps"], c["sparse_structure_shortcut"], c["structured_latent_steps"],
            c["structured_latent_solver"]) for c in calls]
    assert got == [(25, False, 25, None), (4, True, 25, None), (4, True, 12, "ab2r")]


def test_a_wall_qc_fallback_samples_with_todays_sampler() -> None:
    fallback = qc.fallback_params(_params(**D4S12, mesh_postprocess_impl="fast", wall_qc="fallback"))
    assert not set(fallback) & set(NEW_KEYS)
    assert (fallback["sparse_structure_steps"], fallback["structured_latent_steps"]) == (25, 25)
    assert qc.fallback_params(_params()) == _params() | {"mesh_postprocess_impl": "fast", "wall_qc": "off"}


# --- 4. cache key and shape.json -----------------------------------------------------------------------------


class _Identities:
    def cache_identity(self) -> str:
        return "ckpt"

    def cache_env_identity(self) -> str:
        return "env"


def _key(cfg: ShapeConfig) -> tuple[str, dict[str, str]]:
    prep = {"sdf_resolution": 16, "sdf_padding": 0.15, "n_samples": 8, "samples_seed": 0}
    return ShapeCacheKeyer(cfg, _Identities(), prep).key(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool))


def test_the_cache_key_omits_the_defaults_and_takes_each_set_option() -> None:
    base = _key(ShapeConfig(backend="sam3d"))
    assert _key(ShapeConfig(backend="sam3d", sparse_structure_shortcut=False,
                            structured_latent_solver=None))[0] == base[0]
    keys = {base[0], _key(ShapeConfig(backend="sam3d", sparse_structure_shortcut=True))[0],
            _key(ShapeConfig(backend="sam3d", structured_latent_solver="ab2r"))[0],
            _key(ShapeConfig(backend="sam3d", **D4S12))[0]}
    assert len(keys) == 4
    assert "shape/sam3d_multistep.py" in SHAPE_CODE_FILES      # the worker now runs it


class _BoxBackend:
    def __init__(self) -> None:
        box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        self.vertices, self.faces = np.asarray(box.vertices), np.asarray(box.faces)
        self.last_postprocess_stats = None

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        return ShapeResult(self.vertices, self.faces, None, {"decode": 0.25})


def _shape_json(tmp_path: Path, cfg: ShapeConfig) -> dict[str, Any]:
    from fastdo.engine.workers import make_shape_worker
    from tests.engine.fakes import random_frames, write_decode

    clip = tmp_path / "clip"
    write_decode(clip, random_frames(3, h=16, w=20, seed=0))
    masks = np.zeros((3, 16, 20), dtype=bool)
    masks[:, 2:8, 3:12] = True
    np.savez(clip / "mask.npz", masks=masks, frame_idx=np.arange(3, dtype=np.int64),
             keyframe_index=np.array(1, dtype=np.int64))
    make_shape_worker(cfg, backend=_BoxBackend(), sdf_resolution=16, n_samples=64)("clip", clip)
    return read_shape_meta(clip)


def test_a_production_shape_json_records_the_promoted_solver_only(tmp_path: Path) -> None:
    meta = _shape_json(tmp_path, _production_shape_config())
    assert {k: meta[k] for k in NEW_KEYS if k in meta} == {"structured_latent_solver": "ab2r"}
    assert meta["structured_latent_steps"] == 12
    assert set(SHAPE_OPTION_DEFAULTS) <= set(meta)             # production's fast-path options are recorded, as before


@pytest.mark.parametrize(("settings", "recorded"), [(D4, {"sparse_structure_shortcut": True}),
                                                    (D4S12, {"sparse_structure_shortcut": True,
                                                             "structured_latent_solver": "ab2r"})])
def test_shape_json_records_a_set_option_and_the_steps(tmp_path: Path, settings: dict[str, Any],
                                                       recorded: dict[str, Any]) -> None:
    meta = _shape_json(tmp_path, ShapeConfig(backend="sam3d", **settings))
    assert {k: meta[k] for k in NEW_KEYS if k in meta} == recorded
    assert meta["sparse_structure_steps"] == 4
    assert meta["structured_latent_steps"] == settings.get("structured_latent_steps", 25)
