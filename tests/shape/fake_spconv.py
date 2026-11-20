"""spconv and cumm stand-ins for the kernel pin tests; no CUDA and no spconv install.

`make_fake_spconv_modules`: `spconv`, `spconv.pytorch.ops`, `spconv.algo`, `cumm.tensorview` and `cumm.conv.bases` with
the attributes `sam3d_spconv_pin` touches; the timing tuner and `ConvMainUnitTest.implicit_gemm2` raise if called.
Worker-process tests: `install_fake_spconv` (version from `FAKE_SPCONV_VERSION`) and the `--factory`
`make_pin_recording_pipeline`, which records the P4 flag at load and, when pinned, runs one forward conv through the pin.
"""

from __future__ import annotations

import enum
import os
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

FLOAT16, FLOAT32 = 7, 0


class FakeDesp:
    def __init__(self, name: str, algo: str = "Ampere", min_arch: tuple[int, int] = (8, 0), dacc: int = FLOAT16,
                 dtype_input: int = FLOAT16, tile_shape: tuple[int, int, int] = (64, 128, 32)) -> None:
        self.name, self.algo, self.min_arch = name, algo, min_arch
        self.dacc, self.dtype_input, self.tile_shape = dacc, dtype_input, tile_shape

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__


class FakeTensor:
    """`tv.Tensor`: `shape`, `ndim`, `dtype` and `dim(i)`; `FakeTensor()` is the empty tensor."""

    def __init__(self, shape: tuple[int, ...] = (), dtype: int = FLOAT16) -> None:
        self.shape, self.ndim, self.dtype = list(shape), len(shape), dtype

    def dim(self, i: int) -> int:
        return self.shape[i]


class ConvOpType(enum.Enum):
    kForward = 0
    kBackwardInput = 1
    kBackwardWeight = 2


class BestConvAlgoByProfile:
    def __init__(self, algo_desp: Any, arch: tuple[int, int], splitk: int = 1) -> None:
        self.algo_desp, self.arch, self.splitk = algo_desp, arch, splitk

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BestConvAlgoByProfile) and (self.algo_desp, tuple(self.arch), self.splitk) == (
            other.algo_desp, tuple(other.arch), other.splitk)


class FakeSimpleConv:
    def __init__(self, avail: list[FakeDesp]) -> None:
        self.kc_forward_cache: dict[tuple[Any, ...], BestConvAlgoByProfile] = {}
        self.lock = threading.Lock()
        self.avail, self.available_calls = list(avail), []  # type: ignore[var-annotated]

    def get_tuned_algo(self, op_type: Any, i: int, w: int, o: int, k: int, c: int, arch: tuple[int, int],
                       mask_width: int = -1, need_dynamic_mask: bool = False) -> Any:
        return self.kc_forward_cache.get((i, w, o, k, c, arch[0], arch[1], -1, need_dynamic_mask))

    def get_all_available(self, inp: Any, weight: Any, out: Any, layout_i: Any, layout_w: Any, layout_o: Any,
                          arch: tuple[int, int], op_type: Any, mask_width: int, fp32_accum: bool | None = None,
                          use_tf32: bool = True, bias: Any = None, scale: Any = None) -> list[FakeDesp]:
        self.available_calls.append({"fp32_accum": fp32_accum, "use_tf32": use_tf32, "arch": arch, "op": op_type})
        return list(self.avail)

    def tune_and_cache(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("spconv's timing tuner must not run under the kernel pin")


def default_desps() -> list[FakeDesp]:
    return [FakeDesp("Turing_m32n64", "Turing", (7, 5), tile_shape=(32, 64, 32)),
            FakeDesp("Ampere_m64n128_f16", tile_shape=(64, 128, 32)),
            FakeDesp("Ampere_m64n64_f32acc", dacc=FLOAT32, tile_shape=(128, 128, 32)),
            FakeDesp("Simt_m128n256", "Simt", (8, 0), tile_shape=(128, 256, 32))]


def make_fake_spconv_modules(avail: list[FakeDesp] | None = None, version: str = "2.3.6") -> dict[str, ModuleType]:
    mods = {name: ModuleType(name) for name in ("spconv", "spconv.pytorch", "spconv.pytorch.ops", "spconv.algo",
                                                 "cumm", "cumm.tensorview", "cumm.conv", "cumm.conv.bases")}
    conv = FakeSimpleConv(default_desps() if avail is None else avail)

    def implicit_gemm2(params: Any) -> None:
        raise AssertionError("no kernel may be executed or timed while choosing one")

    mods["spconv"].__version__ = version  # type: ignore[attr-defined]
    ops, algo = mods["spconv.pytorch.ops"], mods["spconv.algo"]
    ops.SPCONV_CPP_GEMM, ops.implicit_gemm, ops.CONV = True, lambda *a, **k: None, conv  # type: ignore[attr-defined]
    algo.CONV, algo.BestConvAlgoByProfile = conv, BestConvAlgoByProfile  # type: ignore[attr-defined]
    algo.ConvMainUnitTest = SimpleNamespace(implicit_gemm2=implicit_gemm2)  # type: ignore[attr-defined]
    tv = mods["cumm.tensorview"]
    tv.float16, tv.float32, tv.Tensor = FLOAT16, FLOAT32, FakeTensor  # type: ignore[attr-defined]
    mods["cumm.conv.bases"].ConvOpType = ConvOpType  # type: ignore[attr-defined]
    return mods


def install_fake_spconv() -> dict[str, ModuleType]:
    mods = make_fake_spconv_modules(version=os.environ.get("FAKE_SPCONV_VERSION", "2.3.6"))
    sys.modules.update(mods)
    return mods


def make_pin_recording_pipeline(config_path: str, device: str) -> Any:
    from tests.shape.fake_sam3d import make_fake_pipeline

    ops = sys.modules["spconv.pytorch.ops"]
    pinned = ops.SPCONV_CPP_GEMM is False
    if pinned:                                  # one forward conv (k=64, c=32, kv=27) through the pinned tuner
        ops.CONV.tune_and_cache(ConvOpType.kForward, FakeTensor((10, 32)), FakeTensor((64, 27, 32)),
                                FakeTensor((10, 64)), 0, 0, 0, (8, 0), None, None, None, False)
    pipeline = make_fake_pipeline(config_path, device)
    pipeline._event("spconv", cpp_gemm=ops.SPCONV_CPP_GEMM)
    return pipeline
