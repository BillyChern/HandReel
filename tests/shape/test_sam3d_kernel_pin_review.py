"""The G-pin gate design, its host-load record, the backend's kernel-mode check, fp32_accum,
the SPCONV_ALGO=native refusal and the Hydra `spconv_kernel_pin` entry."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from fastdo.bench import bench_stages
from fastdo.shape import ShapeConfig, ShapeResult
from fastdo.shape import sam3d_spconv_pin as pin
from fastdo.shape.sam3d_backend import Sam3dBackend
from tests.shape import fake_spconv as fs
from tests.shape import sam3d_kernel_pin_gate as gate
from tests.shape.test_sam3d_spconv_pin import _pin_worker

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    mods = fs.make_fake_spconv_modules()
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    return mods


# --- the gate plan: counterbalanced, so pin mode is not confounded with host drift -------------------


def test_gate_plan_is_abba_counterbalanced_with_first_requests_in_both_modes() -> None:
    modes = [mode for mode, _ in gate.PLAN]
    assert Counter(modes) == {"pinned": 6, "tuner": 6}
    pairs = [gate.PLAN[i:i + 2] for i in range(0, len(gate.PLAN), 2)]
    assert all(a[1] == b[1] and {a[0], b[0]} == {"pinned", "tuner"} for a, b in pairs)   # same order in a pair
    assert [a[0] for a, _ in pairs] == ["pinned", "tuner"] * 3                           # ABBA: alternate the lead
    index = {m: [i for i, (mode, _) in enumerate(gate.PLAN) if mode == m] for m in ("pinned", "tuner")}
    assert sum(index["pinned"]) == sum(index["tuner"])                                  # equal mean position in time
    for mode in ("pinned", "tuner"):
        firsts = Counter(gate.ORDERS[order][0] for m, order in gate.PLAN if m == mode)
        assert set(firsts) == set(gate.CLIP_FILES), (mode, firsts)                      # every clip starts a process
        positions = Counter((clip, pos) for m, order in gate.PLAN if m == mode
                            for pos, clip in enumerate(gate.ORDERS[order]))
        assert all(positions[(clip, 1)] + positions[(clip, 2)] >= 2 for clip in gate.CLIP_FILES)


def _row(mode: str, pos: int, seconds: float = 10.0, state: str = "s") -> dict[str, Any]:
    return {"mode": mode, "clip": "c", "pos": pos, "proc": 0, "state": state, "faces": 10, "worker_seconds": seconds,
            "peak_gpu_gib_reserved": 20.6, "kernels": {"k": "a"}, "error": None}


def test_g_time_is_not_passed_without_an_unpinned_first_request() -> None:
    rows = [_row("pinned", 0), _row("pinned", 1), _row("tuner", 1, state="t1"), _row("tuner", 2, state="t2")]
    record = gate.evaluate(rows)
    assert record["clips"]["c"]["delta_first_s"] is None
    assert record["gates"]["G-time"] == "fail" and record["verdict"] == "fail"


# --- the gate runner records host load around every worker (timing contamination evidence) ----------


class _FakeClient:
    pid, load_seconds, spconv_kernels = 4242, 1.5, "pinned"

    def __init__(self) -> None:
        self.last_peak_gpu_gib_reserved, self.last_reply, self.last_digests = 20.6, {"kernels": {"k": "a"}}, {}
        self.last_fallback_peak_gpu_gib_reserved = None        # the G-pin gate never runs the fast path's wall QC
        self.last_postprocess_stats = None

    def reconstruct(self, image: np.ndarray, mask: np.ndarray, params: dict[str, Any]) -> ShapeResult:
        return ShapeResult(np.zeros((4, 3)), np.zeros((2, 3), np.int64), phase_seconds={"decode": 1.0})

    def close(self) -> int:
        return 0


def test_gate_worker_record_carries_host_load_and_cpu_pressure(monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    monkeypatch.setattr(gate, "start_sam3d_worker", lambda cfg: _FakeClient())
    monkeypatch.setattr(gate, "artifact_state", lambda v, f, w: {"files": {}, "state": "s", "faces": 2,
                                                                 "watertight": True})
    loads = iter([(20.0, 19.0, 18.0), (25.0, 21.0, 19.0)])
    monkeypatch.setattr(gate.os, "getloadavg", lambda: next(loads))
    clip = (np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool))
    rows, info = gate._run_worker(ShapeConfig(backend="sam3d"), dict.fromkeys(gate.CLIP_FILES, clip), 0, "pinned",
                                  "D1", tmp_path)
    assert len(rows) == 3 and info["loadavg_start"] == [20.0, 19.0, 18.0] and info["loadavg_end"] == [25.0, 21.0, 19.0]


# --- Sam3dBackend refuses a worker whose kernel mode differs from its config (shape.json, cache key) --


class _ModeClient:
    def __init__(self, mode: Any) -> None:
        self.spconv_kernels, self.closed = mode, False

    def close(self) -> int:
        self.closed = True
        return 0


@pytest.mark.parametrize("pin_flag, mode", [(True, "tuner"), (False, "pinned")])
def test_backend_refuses_a_worker_in_another_kernel_mode(pin_flag: bool, mode: str) -> None:
    cfg = ShapeConfig(backend="sam3d", spconv_kernel_pin=pin_flag)
    client = _ModeClient(mode)
    with pytest.raises(RuntimeError, match="spconv kernels"):
        Sam3dBackend(cfg, loader=lambda: client).load()
    assert client.closed
    with pytest.raises(RuntimeError, match="spconv kernels"):
        Sam3dBackend(cfg, client=_ModeClient(mode))


@pytest.mark.parametrize("mode", ["pinned", None])
def test_backend_accepts_a_matching_or_unreported_kernel_mode(mode: str | None) -> None:
    client = _ModeClient(mode)
    assert Sam3dBackend(ShapeConfig(backend="sam3d"), loader=lambda: client).load() is client and not client.closed


# --- the pin honours an explicit fp32_accum; None keeps the C++ default (False) ----------------------


@pytest.mark.parametrize("given, expected", [(None, False), (True, True), (False, False)])
def test_pinned_tuner_passes_an_explicit_fp32_accum(fake: dict[str, ModuleType], given: bool | None,
                                                    expected: bool) -> None:
    conv = fake["spconv.algo"].CONV
    pin.pin_spconv_kernels()
    tensors = fs.FakeTensor((10, 32)), fs.FakeTensor((64, 27, 32)), fs.FakeTensor((10, 64))
    conv.tune_and_cache(fs.ConvOpType.kForward, *tensors, 0, 0, 0, (8, 0), None, None, None, False, fp32_accum=given)
    assert [call["fp32_accum"] for call in conv.available_calls] == [expected]


# --- review r2 m1: SPCONV_ALGO=native builds ConvAlgo.Native convs, whose Python gemm tuner the pin does not cover --


def test_pin_refuses_spconv_algo_native_and_changes_nothing(fake: dict[str, ModuleType],
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    ops, conv = fake["spconv.pytorch.ops"], fake["spconv.algo"].CONV
    monkeypatch.setenv("SPCONV_ALGO", "native")
    with pytest.raises(RuntimeError, match="SPCONV_ALGO=native"):
        pin.pin_spconv_kernels()
    assert ops.SPCONV_CPP_GEMM is True and "tune_and_cache" not in vars(conv)


@pytest.mark.parametrize("algo", [None, "auto", "implicit_gemm"])
def test_pin_installs_when_sam3d_builds_implicit_gemm_convs(fake: dict[str, ModuleType],
                                                            monkeypatch: pytest.MonkeyPatch, algo: str | None) -> None:
    if algo is None:
        monkeypatch.delenv("SPCONV_ALGO", raising=False)
    else:
        monkeypatch.setenv("SPCONV_ALGO", algo)
    pin.pin_spconv_kernels()
    assert fake["spconv.pytorch.ops"].SPCONV_CPP_GEMM is False


def test_pinned_worker_refuses_to_start_under_spconv_algo_native(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPCONV_ALGO", "native")
    with pytest.raises(RuntimeError, match="failed to load.*SPCONV_ALGO=native"):
        _pin_worker(tmp_path, "pinned", expect="pinned")
    assert not (tmp_path / "events.jsonl").exists()               # refused before the pipeline load
    client = _pin_worker(tmp_path, None, expect="tuner")          # the tuner makes no pin claim: it starts
    assert client.close() == 0


# --- review r2 m3: run/conf/config.yaml exposes spconv_kernel_pin, and the key reaches ShapeConfig ----------------


@pytest.mark.parametrize("override, expected", [(None, True), ("false", False), ("true", True)])
def test_hydra_spconv_kernel_pin_reaches_shape_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                      override: str | None, expected: bool) -> None:
    seen: list[ShapeConfig] = []
    monkeypatch.setattr(bench_stages, "make_shape_worker", lambda cfg, **kw: seen.append(cfg) or (lambda c, d: None))
    overrides = [f"paths.root={_REPO}", "bench.shape.backend=sam3d"]
    overrides += [] if override is None else [f"bench.shape.spconv_kernel_pin={override}"]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    assert "spconv_kernel_pin" in cfg.bench.shape
    bench_stages._shape_stage(cfg, tmp_path, [])
    assert [shape_cfg.spconv_kernel_pin for shape_cfg in seen] == [expected]
