"""The default-off SAM 3D sampler option `sparse_structure_solver` (None | "ab2r").

The sparse-structure twin of `structured_latent_solver`: the right-limited AB2 on the SS flow, with the
regime of SAM 3D's guidance switch on the SS generator (`cfg_regime(..., pipeline.ss_cfg_strength)`). It travels
from `ShapeConfig` through the request to `reconstruct_geometry` like the other sampler options
(`test_sam3d_sampler_options.py`, whose helpers these tests reuse): a default request is unchanged on the wire, in its
digest, its cache key, its shape.json and its calls; a set option reaches the call; the wall-QC fallback drops it; and
it is refused together with `sparse_structure_shortcut` (a different, distilled flow) at every boundary. Whether the
real meshes equal the reference ones is a GPU check (arm set P,SS12).
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fastdo.shape import ShapeConfig
from fastdo.shape import sam3d_multistep as ms
from fastdo.shape import sam3d_worker as worker_mod
from fastdo.shape import wall_qc as qc
from fastdo.shape.backend import SPARSE_STRUCTURE_SOLVERS, non_default_sampler_options
from fastdo.shape.sam3d_backend import Sam3dBackend
from fastdo.shape.sam3d_worker import (
    AB2R_RIGHT_LIMIT_EPS,
    request_params_digest,
    resolve_shape_options,
    serve,
)
from tests.shape.fake_sam3d import FakeSam3dPipeline
from tests.shape.test_sam3d_sampler_options import (
    _events,
    _key,
    _params,
    _pre157_digest,
    _production_shape_config,
    _reconstruct,
    _shape_json,
)

KEY = "sparse_structure_solver"
S12 = {"structured_latent_steps": 12, "structured_latent_solver": "ab2r"}      # production
SS12 = S12 | {"sparse_structure_steps": 12, KEY: "ab2r"}                     # the treatment arm


# --- 1. defaults change nothing -----------------------------------------------------------------------------


def test_the_option_is_off_by_default_and_absent_from_a_default_request() -> None:
    assert SPARSE_STRUCTURE_SOLVERS == worker_mod.SPARSE_STRUCTURE_SOLVERS == ("ab2r",)
    cfg = ShapeConfig(backend="sam3d")
    assert cfg.sparse_structure_solver is None and non_default_sampler_options(cfg) == {}
    assert KEY not in Sam3dBackend(cfg).sampling_params
    options = resolve_shape_options(_params())
    assert options.ss_solver is None
    assert worker_mod._call_kwargs(_params(), options, None, None, None)[KEY] is None


def test_the_production_request_does_not_carry_it() -> None:
    cfg = _production_shape_config()
    assert cfg.sparse_structure_solver is None and cfg.sparse_structure_steps == 25
    assert KEY not in Sam3dBackend(cfg).sampling_params


def test_a_default_request_keeps_its_digest() -> None:
    assert request_params_digest(_params(sparse_structure_solver=None)) == _pre157_digest(_params())
    production = Sam3dBackend(_production_shape_config()).sampling_params
    assert request_params_digest(production | {KEY: None}) == request_params_digest(production)


# --- 2. set, it reaches the request, the digest and the call ------------------------------------------------


@pytest.mark.parametrize("settings", [{KEY: "ab2r"}, SS12], ids=["alone", "SS12"])
def test_it_reaches_the_request_and_the_worker_call(settings: dict[str, Any]) -> None:
    cfg = ShapeConfig(backend="sam3d", **settings)
    assert non_default_sampler_options(cfg) == {k: v for k, v in settings.items()
                                                if k in (KEY, "structured_latent_solver")}
    params = Sam3dBackend(cfg).sampling_params
    assert params[KEY] == "ab2r"
    assert params["sparse_structure_steps"] == settings.get("sparse_structure_steps", 25)
    options = resolve_shape_options(params)
    assert options.ss_solver == "ab2r" and options.ss_shortcut is False
    kwargs = worker_mod._call_kwargs(params, options, None, None, None)
    assert (kwargs[KEY], kwargs["sparse_structure_shortcut"]) == ("ab2r", False)
    assert kwargs["structured_latent_solver"] == settings.get("structured_latent_solver")
    assert request_params_digest(params) != request_params_digest(params | {KEY: None})


def test_it_moves_the_digest_on_its_own_and_apart_from_the_slat_solver() -> None:
    digests = {request_params_digest(_params()), request_params_digest(_params(sparse_structure_solver="ab2r")),
               request_params_digest(_params(structured_latent_solver="ab2r")),
               request_params_digest(_params(sparse_structure_solver="ab2r", structured_latent_solver="ab2r"))}
    assert len(digests) == 4


_REFUSED: tuple[dict[str, Any], ...] = (
    {KEY: "ab2"}, {KEY: "euler"}, {KEY: 2}, {KEY: ""}, {KEY: "ab2r", "solver": "midpoint"},
    {KEY: "ab2r", "sparse_structure_shortcut": True},
    {KEY: "ab2r", "sparse_structure_shortcut": True, "sparse_structure_steps": 4},
)


@pytest.mark.parametrize("bad", _REFUSED, ids=[repr(b) for b in _REFUSED])
def test_the_worker_boundary_refuses_exactly_what_shapeconfig_refuses(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError) as config_error:
        ShapeConfig(backend="sam3d", **bad)
    with pytest.raises(ValueError) as worker_error:
        resolve_shape_options(_params(**bad))
    assert str(worker_error.value) == str(config_error.value)
    assert KEY in str(worker_error.value)


def test_the_shortcut_and_the_solver_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ShapeConfig(backend="sam3d", sparse_structure_steps=4, sparse_structure_shortcut=True,
                    sparse_structure_solver="ab2r")
    ShapeConfig(backend="sam3d", sparse_structure_steps=4, sparse_structure_shortcut=True)          # each alone is fine
    ShapeConfig(backend="sam3d", sparse_structure_steps=12, sparse_structure_solver="ab2r")


@pytest.mark.parametrize("backend", ["trellis", "template"])
def test_the_option_is_refused_off_sam3d(backend: str) -> None:
    extra = {"mesh_postprocess": False} if backend == "template" else {}
    with pytest.raises(ValueError, match="allowed only for sam3d"):
        ShapeConfig(backend=backend, sparse_structure_solver="ab2r", **extra)
    ShapeConfig(backend=backend, sparse_structure_solver=None, **extra)


# --- 3. the calls the driver makes ----------------------------------------------------------------------------


def _pipeline() -> tuple[FakeSam3dPipeline, dict[str, Any]]:
    """The fake pipeline with the generator attributes `cfg_regime` reads, on both generators, and a spy recording
    each generator's update at the moment its flow samples. The SS and SLAT guidance strengths differ so the test
    can tell which one the SS regime read (it reads the interval and the time scale only through the regime)."""
    pipe = FakeSam3dPipeline()
    for name, interval in (("ss_generator", [0.0, 300.0]), ("slat_generator", [0.0, 500.0])):
        gen = pipe.models[name]
        gen.reverse_fn = SimpleNamespace(interval=interval)
        gen.time_scale = 1000.0
    pipe.ss_cfg_strength, pipe.slat_cfg_strength = 7.0, 1.0
    seen: dict[str, Any] = {}
    for method, gen_name in (("sample_sparse_structure", "ss_generator"), ("sample_slat", "slat_generator")):
        original = getattr(pipe, method)

        def spy(*args: Any, _gen: str = gen_name, _original: Any = original, **kwargs: Any) -> Any:
            seen[_gen] = pipe.models[_gen]._solver
            return _original(*args, **kwargs)

        setattr(pipe, method, spy)
    return pipe, seen


def test_a_default_call_leaves_the_ss_update_alone() -> None:
    pipe, seen = _pipeline()
    _reconstruct(pipe)
    assert seen == {"ss_generator": "solver:euler", "slat_generator": "solver:euler"}


def test_ab2r_installs_the_right_limited_ab2_for_the_ss_flow_only_and_restores_it() -> None:
    pipe, seen = _pipeline()
    _reconstruct(pipe, sparse_structure_steps=12, sparse_structure_solver="ab2r")
    solver = seen["ss_generator"]
    assert isinstance(solver, ms.AdamsBashforth2)
    assert solver.right_limit_eps == AB2R_RIGHT_LIMIT_EPS == 1e-6
    # The SS generator's interval [0, 300] on t * 1000: guidance on below 0.3, off above.
    assert solver.regime is not None and solver.regime(0.2) is True and solver.regime(0.4) is False
    assert seen["slat_generator"] == "solver:euler"                       # the SLAT flow stays `solver`
    (ss,), (slat,) = _events(pipe, "sparse_structure"), _events(pipe, "structured_latent")
    assert (ss["steps"], ss["use_distillation"]) == (12, False)           # the guided flow, not the shortcut
    assert (slat["steps"], slat["use_distillation"]) == (25, False)
    assert pipe.models["ss_generator"]._solver == "solver:euler"
    assert pipe.models["slat_generator"]._solver == "solver:euler"


def test_the_ss_regime_reads_the_ss_guidance_strength(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    real = ms.cfg_regime

    def recording(*args: Any) -> Any:
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(worker_mod.sam3d_multistep, "cfg_regime", recording)
    pipe, _ = _pipeline()
    _reconstruct(pipe, sparse_structure_steps=12, sparse_structure_solver="ab2r")
    assert calls == [([0.0, 300.0], 1000.0, 7.0)]


def test_ss12_stacks_both_solvers_each_on_its_own_generator() -> None:
    pipe, seen = _pipeline()
    _reconstruct(pipe, sparse_structure_steps=12, sparse_structure_solver="ab2r", structured_latent_steps=12,
                 structured_latent_solver="ab2r")
    ss_solver, slat_solver = seen["ss_generator"], seen["slat_generator"]
    assert isinstance(ss_solver, ms.AdamsBashforth2) and isinstance(slat_solver, ms.AdamsBashforth2)
    assert ss_solver is not slat_solver
    assert ss_solver.regime(0.4) is False and slat_solver.regime(0.4) is True      # 300 vs 500 on t * 1000
    (ss,), (slat,) = _events(pipe, "sparse_structure"), _events(pipe, "structured_latent")
    assert (ss["steps"], slat["steps"]) == (12, 12)
    assert all(pipe.models[g]._solver == "solver:euler" for g in ("ss_generator", "slat_generator"))


def test_ab2r_restores_the_update_when_the_ss_flow_fails() -> None:
    pipe, _ = _pipeline()
    pipe.fail_in = "sparse_structure"
    with pytest.raises(RuntimeError, match="injected"):
        _reconstruct(pipe, sparse_structure_solver="ab2r")
    assert pipe.models["ss_generator"]._solver == "solver:euler"


def test_ab2r_is_active_only_inside_the_sparse_structure_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entered after the SS CUDA-graph context (inside it) and inside the timed phase; exited before the SLAT flow."""
    order: list[str] = []

    @contextmanager
    def graph(generator: Any) -> Any:
        order.append("graph enter")
        try:
            yield SimpleNamespace(take_stats=dict)
        finally:
            order.append("graph exit")

    real = worker_mod.sparse_structure_ab2r

    @contextmanager
    def solver_ctx(pipeline: Any, generator: Any) -> Any:
        order.append("ab2r enter")
        with real(pipeline, generator):
            yield
        order.append("ab2r exit")

    monkeypatch.setattr(worker_mod.sam3d_cuda_graph, "graphed_ss_backbone", graph)
    monkeypatch.setattr(worker_mod, "sparse_structure_ab2r", solver_ctx)
    pipe, seen = _pipeline()
    for method, label in (("sample_sparse_structure", "ss flow"), ("sample_slat", "slat flow")):
        original = getattr(pipe, method)

        def logged(*args: Any, _label: str = label, _original: Any = original, **kwargs: Any) -> Any:
            order.append(_label)
            return _original(*args, **kwargs)

        setattr(pipe, method, logged)
    _, _, seconds = _reconstruct(pipe, sparse_structure_solver="ab2r", cuda_graph_ss=True)
    assert order == ["graph enter", "ab2r enter", "ss flow", "ab2r exit", "graph exit", "slat flow"]
    assert isinstance(seen["ss_generator"], ms.AdamsBashforth2)
    assert "sparse_structure_flow" in seconds


def test_the_driver_refuses_an_unknown_ss_solver_and_the_shortcut_together() -> None:
    pipe, _ = _pipeline()
    with pytest.raises(ValueError, match=KEY):
        _reconstruct(pipe, sparse_structure_solver="ab3")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _reconstruct(pipe, sparse_structure_steps=4, sparse_structure_shortcut=True, sparse_structure_solver="ab2r")
    assert _events(pipe, "pointmap") == []                                # refused before any model call


def test_the_worker_passes_the_option_from_the_request_to_the_driver(monkeypatch: pytest.MonkeyPatch,
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
             for i, p in enumerate((_params(**S12), _params(**SS12)))]
    replies = io.StringIO()
    assert serve(object(), io.StringIO("\n".join(lines) + "\n"), replies) == 0
    assert all(json.loads(r)["ok"] for r in replies.getvalue().splitlines())
    got = [(c["sparse_structure_steps"], c[KEY], c["sparse_structure_shortcut"], c["structured_latent_steps"],
            c["structured_latent_solver"]) for c in calls]
    assert got == [(25, None, False, 12, "ab2r"), (12, "ab2r", False, 12, "ab2r")]


def test_a_wall_qc_fallback_samples_with_todays_sampler() -> None:
    assert KEY in qc.FALLBACK_DROPPED
    fallback = qc.fallback_params(_params(**SS12, mesh_postprocess_impl="fast", wall_qc="fallback"))
    assert KEY not in fallback and "structured_latent_solver" not in fallback
    assert (fallback["sparse_structure_steps"], fallback["structured_latent_steps"]) == (25, 25)
    assert resolve_shape_options(fallback).ss_solver is None


# --- 4. cache key and shape.json -----------------------------------------------------------------------------


def test_the_cache_key_omits_the_default_and_takes_the_set_option() -> None:
    base = _key(ShapeConfig(backend="sam3d"))
    assert _key(ShapeConfig(backend="sam3d", sparse_structure_solver=None))[0] == base[0]
    s12 = _key(ShapeConfig(backend="sam3d", **S12))[0]
    keys = {base[0], s12, _key(ShapeConfig(backend="sam3d", sparse_structure_solver="ab2r"))[0],
            _key(ShapeConfig(backend="sam3d", **SS12))[0],
            _key(ShapeConfig(backend="sam3d", **S12, sparse_structure_solver="ab2r"))[0]}   # differs from S12 only here
    assert len(keys) == 5


def test_a_production_shape_json_does_not_record_it(tmp_path: Path) -> None:
    assert KEY not in _shape_json(tmp_path, _production_shape_config())


def test_shape_json_records_the_set_option_and_the_steps(tmp_path: Path) -> None:
    meta = _shape_json(tmp_path, ShapeConfig(backend="sam3d", **SS12))
    assert (meta[KEY], meta["structured_latent_solver"]) == ("ab2r", "ab2r")
    assert (meta["sparse_structure_steps"], meta["structured_latent_steps"]) == (12, 12)
    assert "sparse_structure_shortcut" not in meta
