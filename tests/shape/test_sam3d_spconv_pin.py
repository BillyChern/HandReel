"""The spconv kernel pin for SAM 3D; fakes unless marked GPU."""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.engine.shape_cache import SHAPE_CODE_FILES, ShapeCacheKeyer
from fastdo.shape import ShapeConfig
from fastdo.shape import sam3d_backend as backend_mod
from fastdo.shape import sam3d_spconv_pin as pin
from fastdo.shape.sam3d_backend import WORKER_SCRIPT, Sam3dWorkerClient
from fastdo.shape.sam3d_worker import reconstruct_geometry
from tests.shape import fake_spconv as fs
from tests.shape.fake_sam3d import FakeSam3dPipeline

_REPO = Path(__file__).resolve().parents[2]
_F16, _F32, _FWD = fs.FLOAT16, fs.FLOAT32, fs.ConvOpType.kForward


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    mods = fs.make_fake_spconv_modules()
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    return mods


def _conv(mods: dict[str, ModuleType]) -> Any:
    return mods["spconv.algo"].CONV


def _call(conv: Any, op: Any = _FWD, k: int = 64, c: int = 32, kv: int = 27) -> tuple[Any, float]:
    tensors = fs.FakeTensor((10, c)), fs.FakeTensor((k, kv, c)), fs.FakeTensor((10, k))
    return conv.tune_and_cache(op, *tensors, 0, 0, 0, (8, 0), None, None, None, False)


def _frame() -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    return np.zeros((8, 8, 3), np.uint8), mask


# --- T1-T3: install, restore, idempotence, fail closed -----------------------------------------------


def test_install_sets_the_python_path_flag_and_the_sentinel_tuner(fake: dict[str, ModuleType]) -> None:   # T1
    ops, conv = fake["spconv.pytorch.ops"], _conv(fake)
    restore = pin.pin_spconv_kernels()
    assert ops.SPCONV_CPP_GEMM is False
    assert getattr(vars(conv)["tune_and_cache"], pin.PATCH_SENTINEL) is True
    _call(conv)
    restore()
    assert ops.SPCONV_CPP_GEMM is True and "tune_and_cache" not in vars(conv)
    assert len(conv.kc_forward_cache) == 1                        # restore leaves the cache as it is


def test_install_is_idempotent_and_restore_is_single_shot(fake: dict[str, ModuleType]) -> None:       # T2
    ops, conv = fake["spconv.pytorch.ops"], _conv(fake)
    restore = pin.pin_spconv_kernels()
    installed = vars(conv)["tune_and_cache"]
    pin.pin_spconv_kernels()()                                    # a second install and its restore: no-ops
    assert vars(conv)["tune_and_cache"] is installed and ops.SPCONV_CPP_GEMM is False
    restore()
    restore()
    assert ops.SPCONV_CPP_GEMM is True and "tune_and_cache" not in vars(conv)
    restore2 = pin.pin_spconv_kernels()
    assert ops.SPCONV_CPP_GEMM is False and getattr(vars(conv)["tune_and_cache"], pin.PATCH_SENTINEL)
    restore2()
    assert ops.SPCONV_CPP_GEMM is True and "tune_and_cache" not in vars(conv)


@pytest.mark.parametrize("breakage", ["version", "ops_attr", "conv_attr", "best_class", "warm_cache"])
def test_failed_precondition_raises_and_changes_nothing(fake: dict[str, ModuleType], breakage: str) -> None:  # T3
    ops, algo, conv = fake["spconv.pytorch.ops"], fake["spconv.algo"], _conv(fake)
    if breakage == "version":
        fake["spconv"].__version__ = "2.3.8"  # type: ignore[attr-defined]
    elif breakage == "ops_attr":
        del ops.implicit_gemm
    elif breakage == "conv_attr":
        del conv.lock
    elif breakage == "best_class":
        del algo.BestConvAlgoByProfile
    else:
        conv.kc_forward_cache[(7, 7, 7, 64, 32, 8, 0, -1, False)] = "timed"

    def snapshot() -> list[dict[str, Any]]:
        return [dict(vars(ops)), dict(vars(conv)), dict(vars(algo)), dict(conv.kc_forward_cache)]

    before = snapshot()
    with pytest.raises(RuntimeError, match="spconv"):
        pin.pin_spconv_kernels()
    assert ops.SPCONV_CPP_GEMM is True
    for old, new in zip(before, snapshot(), strict=True):
        assert new.keys() == old.keys() and all(new[key] is old[key] for key in old)


# --- T4-T5: the rule and the pinned tuner ------------------------------------------------------------


def test_fixed_rule_is_order_invariant_and_breaks_ties_by_name() -> None:                             # T4
    desps = [fs.FakeDesp("Turing_a", "Turing", (7, 5), tile_shape=(128, 256, 32)),
             fs.FakeDesp("Ampere_small", tile_shape=(32, 64, 32)),
             fs.FakeDesp("Ampere_big_f32acc", dacc=_F32, tile_shape=(128, 256, 32)),
             fs.FakeDesp("Ampere_b", tile_shape=(64, 128, 32)), fs.FakeDesp("Ampere_a", tile_shape=(128, 64, 32))]
    assert {str(pin.fixed_rule(list(p))) for p in itertools.permutations(desps)} == {"Ampere_b"}


def test_candidates_drop_simt_only_for_fp16_on_sm70_and_force_f16_accumulators(fake: dict[str, ModuleType]) -> None:
    conv = _conv(fake)

    def names(dtype: int, arch: tuple[int, int]) -> set[str]:
        tensors = [fs.FakeTensor((10, 32), dtype), fs.FakeTensor((64, 27, 32), dtype), fs.FakeTensor((10, 64), dtype)]
        return {str(d) for d in pin.candidates(conv, *tensors, 0, 0, 0, arch, _FWD, -1, True, fs.FakeTensor(),
                                               fs.FakeTensor())}

    everything = {str(d) for d in conv.avail}
    assert names(_F16, (8, 0)) == names(_F16, (7, 0)) == everything - {"Simt_m128n256"}
    assert names(_F32, (8, 0)) == everything and names(_F16, (6, 1)) == everything
    assert {call["fp32_accum"] for call in conv.available_calls} == {False}


def test_pinned_tuner_caches_spconvs_key_without_running_a_kernel(fake: dict[str, ModuleType]) -> None:  # T5
    conv = _conv(fake)
    pin.pin_spconv_kernels()
    res, seconds = _call(conv, k=64, c=32, kv=27)
    assert seconds == 0.0 and res == fs.BestConvAlgoByProfile(conv.avail[1], (8, 0), 1)   # Ampere_m64n128_f16
    assert list(conv.kc_forward_cache) == [(_F16, _F16, _F16, 64, 32, 8, 0, -1, False)]
    assert conv.get_tuned_algo(_FWD, _F16, _F16, _F16, 64, 32, (8, 0)) is res
    _call(conv, k=96, c=48, kv=64)                                # kv > 32: the dynamic-mask key
    assert (_F16, _F16, _F16, 96, 48, 8, 0, -1, True) in conv.kc_forward_cache
    assert pin.pinned_kernels() == {"i7w7o7:k64:c32:sm80": "Ampere_m64n128_f16",
                                    "i7w7o7:k96:c48:sm80": "Ampere_m64n128_f16"}
    for op in (fs.ConvOpType.kBackwardInput, fs.ConvOpType.kBackwardWeight):
        with pytest.raises(NotImplementedError):
            _call(conv, op=op)
    conv.avail = []
    with pytest.raises(ValueError, match="no spconv kernel"):
        _call(conv, k=128)


# --- T6: the worker process --------------------------------------------------------------------------


def _pin_worker(tmp_path: Path, mode: str | None, expect: str | None, version: str = "2.3.6") -> Sam3dWorkerClient:
    script = ("import sys; sys.path.insert(0, sys.argv.pop(1)); from tests.shape.fake_spconv import install_fake_spconv; "
              "install_fake_spconv(); import sam3d_worker as w; sys.exit(w.main(sys.argv[1:]))")
    command = [sys.executable, "-c", script, str(WORKER_SCRIPT.parent), "--config", "fake.yaml", "--device", "cpu",
               "--factory", "tests.shape.fake_spconv:make_pin_recording_pipeline",
               *(["--spconv-kernels", mode] if mode else [])]
    env = dict(os.environ, FAKE_SAM3D_LOG=str(tmp_path / "events.jsonl"), CUDA_VISIBLE_DEVICES="",
               FAKE_SPCONV_VERSION=version, PYTHONPATH=os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")]))
    return Sam3dWorkerClient(command, env=env, startup_timeout_s=120.0, request_timeout_s=120.0, spconv_kernels=expect)


@pytest.mark.parametrize("mode", ["pinned", None])
def test_worker_installs_the_pin_before_the_pipeline_loads(tmp_path: Path, mode: str | None) -> None:     # T6
    client = _pin_worker(tmp_path, mode, expect=mode or "tuner")
    params = {"seed": 0, "sparse_structure_steps": 2, "structured_latent_steps": 2, "solver": "euler",
              "mesh_postprocess": False, "digest": True}
    try:
        assert client.spconv_kernels == (mode or "tuner")
        client.reconstruct(*_frame(), params)
        kernels = client.last_reply["kernels"]
    finally:
        assert client.close() == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["cpp_gemm"] for e in events if e["event"] == "spconv"] == [mode != "pinned"]
    assert kernels == ({"i7w7o7:k64:c32:sm80": "Ampere_m64n128_f16"} if mode == "pinned" else {})


def test_worker_refuses_to_start_on_a_failed_precondition_and_client_checks_the_mode(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="failed to load.*spconv"):
        _pin_worker(tmp_path, "pinned", expect="pinned", version="2.3.8")
    assert not (tmp_path / "events.jsonl").exists()               # refused before the pipeline load
    with pytest.raises(RuntimeError, match="expected 'pinned'"):
        _pin_worker(tmp_path, None, expect="pinned")


# --- T7-T8: ShapeConfig, start_sam3d_worker, the shape cache -----------------------------------------


def test_shape_config_spconv_kernel_pin_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:   # T7
    assert ShapeConfig(backend="sam3d").spconv_kernel_pin is True
    assert ShapeConfig().spconv_kernel_pin is False and ShapeConfig(backend="template").spconv_kernel_pin is False
    with pytest.raises(ValueError, match="spconv_kernel_pin"):
        ShapeConfig(backend="trellis", spconv_kernel_pin=True)
    with pytest.raises(ValueError, match="spconv_kernel_pin"):
        ShapeConfig(backend="sam3d", spconv_kernel_pin=1)  # type: ignore[arg-type]

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "pipeline.yaml").write_text("_target_: x\n")
    started: list[tuple[list[str], str | None]] = []

    def _client(command: list[str], spconv_kernels: str | None = None, **kwargs: Any) -> None:
        started.append((list(command), spconv_kernels))

    monkeypatch.setattr(backend_mod, "Sam3dWorkerClient", _client)
    for flag in (True, False):
        cfg = ShapeConfig(backend="sam3d", sam3d_python="/env/python", spconv_kernel_pin=flag)
        backend_mod.start_sam3d_worker(cfg, snapshot_fn=lambda *a, **k: str(tmp_path))
    assert [cmd[cmd.index("--spconv-kernels") + 1] for cmd, _ in started] == ["pinned", "tuner"]
    assert [expected for _, expected in started] == ["pinned", "tuner"]


def test_cache_key_covers_the_pin_and_its_code() -> None:                                              # T8
    class _Identities:
        def cache_identity(self) -> str:
            return "ckpt"

        def cache_env_identity(self) -> str:
            return "env"

    image, mask = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool)
    prep = {"sdf_resolution": 16, "sdf_padding": 0.15, "n_samples": 8, "samples_seed": 0}
    keys = {ShapeCacheKeyer(ShapeConfig(backend="sam3d", spconv_kernel_pin=flag), _Identities(), prep).key(image, mask)[0]
            for flag in (True, False)}
    assert len(keys) == 2
    assert "shape/sam3d_spconv_pin.py" in SHAPE_CODE_FILES


# --- T9: the torch.device context of every model phase --------------------------------------


class _DevicePipeline(FakeSam3dPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("meta")
        self.defaults: dict[str, torch.device] = {}

    def _event(self, name: str, **fields: Any) -> None:
        self.defaults[name] = torch.get_default_device()
        super()._event(name, **fields)

    def sample_sparse_structure(self, ss_input_dict: dict[str, Any], inference_steps: int | None = None,
                                use_distillation: bool = False) -> dict[str, Any]:
        self._event("sparse_structure")                           # no torch.rand: meta tensors have no values
        return {"coords": torch.zeros(4, 4, dtype=torch.int32)}


def test_every_model_phase_runs_under_the_pipeline_device_context() -> None:                           # T9
    pipe, outside = _DevicePipeline(), torch.get_default_device()
    reconstruct_geometry(pipe, *_frame(), seed=0, sparse_structure_steps=2, structured_latent_steps=2,
                         solver="euler", mesh_postprocess=False)
    assert set(pipe.defaults) == {"pointmap", "preprocess", "sparse_structure", "structured_latent", "decode"}
    assert all(device == pipe.device for device in pipe.defaults.values()), pipe.defaults
    assert torch.get_default_device() == outside != pipe.device


# --- the gate evaluation and the real gate (GPU) -----------------------------------------------------


def _row(mode: str, pos: int, state: str = "s", seconds: float = 10.0, reserved: float = 20.6,
         kernels: dict[str, str] | None = None, error: str | None = None) -> dict[str, Any]:
    return {"mode": mode, "clip": "c", "pos": pos, "proc": 0, "state": state, "faces": 10, "worker_seconds": seconds,
            "peak_gpu_gib_reserved": reserved, "kernels": {"k": "a"} if kernels is None else kernels, "error": error}


def test_gate_evaluation_verdicts() -> None:
    from tests.shape.sam3d_kernel_pin_gate import evaluate

    base = [_row("pinned", p) for p in (0, 1, 1)] + [_row("tuner", 0, "t1"), _row("tuner", 1, "t2")]
    assert evaluate(base)["gates"] == {"G-id": "pass", "G-ctrl": "pass", "G-mem": "pass", "G-time": "pass"}
    assert evaluate(base)["verdict"] == "pass"
    no_ctrl = evaluate(base[:3] + [_row("tuner", 0), _row("tuner", 1)])
    assert no_ctrl["gates"]["G-ctrl"] == no_ctrl["verdict"] == "inconclusive"
    assert evaluate(base + [_row("pinned", 2, "other")])["gates"]["G-id"] == "fail"
    assert evaluate(base + [_row("pinned", 2, kernels={"k": "b"})])["gates"]["G-id"] == "fail"
    assert evaluate(base + [_row("pinned", 2, reserved=21.5)])["gates"]["G-mem"] == "fail"
    assert evaluate(base + [_row("tuner", 2, error="boom")])["gates"]["G-mem"] == "fail"
    slow = evaluate([_row("pinned", p, seconds=10.6 if p else 10.0) for p in (0, 1, 1)] + base[3:])
    assert slow["gates"]["G-time"] == slow["verdict"] == "fail"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SAM 3D Objects needs CUDA (and > 17 GiB of allocator memory)")
def test_real_sam3d_kernel_pin_gate(tmp_path: Path) -> None:
    if not os.environ.get("FASTDO_SAM3D_PYTHON"):
        pytest.skip("FASTDO_SAM3D_PYTHON (the SAM 3D environment's interpreter) is not set")
    if float(os.environ.get("FASTDO_GPU_MEM_GB", "12")) < 19:
        pytest.skip("SAM 3D peaks at 18.67 GiB allocated; run with FASTDO_GPU_MEM_GB >= 19")
    inputs = os.environ.get("FASTDO_KERNEL_PIN_INPUTS")
    if not inputs:
        pytest.skip("FASTDO_KERNEL_PIN_INPUTS (kf_135250.npz, kf_135545.npz, whisk.npz) is not set")
    from tests.shape.sam3d_kernel_pin_gate import run_kernel_pin_gate

    record = run_kernel_pin_gate(Path(inputs), tmp_path)
    log = os.environ.get("FASTDO_KERNEL_PIN_LOG")
    if log:
        Path(log).write_text(json.dumps(record, indent=1, default=str))
    assert record["gates"] == {"G-id": "pass", "G-ctrl": "pass", "G-mem": "pass", "G-time": "pass"}, record["gates"]
