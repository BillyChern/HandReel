"""`bench.mask_compile`'s compile-mode patch (`fastdo.shape.sam2_compile`) and the SAM 2 loader it reaches.

The patch must rewrite only sam2's compile calls, only while the predictor is built, and restore `torch.compile` on
exit -- also when the build raises. `none` must keep the loader's call exactly as it was (no `vos_optimized`, no patch).
No compilation happens here: every test injects a fake `compile`.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from fastdo.shape import masks_sam2
from fastdo.shape.sam2_compile import (
    DET_OPTIONS,
    MASK_COMPILE_MODES,
    NOCG_MODE,
    VOS_COMPILED,
    CudagraphStep,
    check_mask_compile,
    compile_mode_patch,
)


def _sam2_fn(qualname: str, out: Any = None) -> Any:
    """A plain function that claims to live in sam2 (as `SAM2VideoPredictorVOS`'s bound forwards do)."""
    def fn(*args: Any, **kwargs: Any) -> Any:
        return out
    fn.__module__, fn.__qualname__, fn.__name__ = "sam2.modeling.fake", qualname, qualname.split(".")[-1]
    return fn


class FakeCompile:
    """Records every call; returns a callable tagged with the compile kwargs it was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, fn: Any = None, **kwargs: Any) -> Any:
        self.calls.append((fn, kwargs))

        def compiled(*a: Any, **k: Any) -> Any:
            return fn(*a, **k)
        compiled.__name__ = compiled.__qualname__ = getattr(fn, "__qualname__", "compiled")
        compiled.compile_kwargs = kwargs  # type: ignore[attr-defined]
        return compiled


def _module() -> types.SimpleNamespace:
    return types.SimpleNamespace(compile=FakeCompile())


# --- knob values -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", MASK_COMPILE_MODES)
def test_every_registered_mode_is_accepted(value: str) -> None:
    assert check_mask_compile(value) == value


@pytest.mark.parametrize("bad", ["", "vos", "NONE", "max-autotune", None, 0, True])
def test_unknown_modes_are_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="mask_compile"):
        check_mask_compile(bad)


# --- the patch ---------------------------------------------------------------------------------------------------


def test_nocg_rewrites_only_the_mode_of_sam2_calls_and_passes_others_through() -> None:
    mod = _module()
    fake = mod.compile
    sam2_fn, other = _sam2_fn("MemoryEncoder.forward"), (lambda x: x)
    with compile_mode_patch("vos_nocg", module=mod) as rewritten:
        assert mod.compile is not fake
        a = mod.compile(sam2_fn, mode="max-autotune", fullgraph=True, dynamic=False)
        b = mod.compile(other, mode="max-autotune", dynamic=True)
    assert mod.compile is fake                                              # restored
    assert rewritten == ["MemoryEncoder.forward"]
    assert a.compile_kwargs == {"mode": NOCG_MODE, "fullgraph": True, "dynamic": False}
    assert b.compile_kwargs == {"mode": "max-autotune", "dynamic": True}    # not sam2: untouched
    assert not isinstance(a, CudagraphStep) and not isinstance(b, CudagraphStep)


def test_det_drops_the_mode_for_deterministic_inductor_options_on_sam2_calls_only() -> None:
    """torch.compile refuses mode AND options together, so the mode goes and the options pin every kernel."""
    mod = _module()
    sam2_fn, other = _sam2_fn("MaskDecoder.forward"), (lambda x: x)
    with compile_mode_patch("vos_det", module=mod) as rewritten:
        a = mod.compile(sam2_fn, mode="max-autotune", fullgraph=True, dynamic=False)
        b = mod.compile(other, mode="max-autotune")
    assert rewritten == ["MaskDecoder.forward"]
    assert a.compile_kwargs == {"options": {"triton.autotune_pointwise": False, "max_autotune": False},
                                "fullgraph": True, "dynamic": False}
    assert "mode" not in a.compile_kwargs and b.compile_kwargs == {"mode": "max-autotune"}
    assert a.compile_kwargs["options"] == DET_OPTIONS and a.compile_kwargs["options"] is not DET_OPTIONS  # a copy
    assert not isinstance(a, CudagraphStep)


def test_cg_clone_keeps_upstream_kwargs_and_wraps_each_compiled_callable() -> None:
    mod = _module()
    fns = [_sam2_fn(name) for name in sorted(VOS_COMPILED)]
    with compile_mode_patch("vos_cg_clone", module=mod) as rewritten:
        wrapped = [mod.compile(fn, mode="max-autotune", fullgraph=True, dynamic=False) for fn in fns]
        plain = mod.compile(lambda x: x, mode="max-autotune")
    assert sorted(rewritten) == sorted(VOS_COMPILED)
    assert all(isinstance(w, CudagraphStep) for w in wrapped) and not isinstance(plain, CudagraphStep)
    assert len({id(w) for w in wrapped}) == len(fns)                       # one wrapper per compiled callable
    for fn, w, (seen, kwargs) in zip(fns, wrapped, mod.compile.calls, strict=False):
        assert seen is fn and kwargs == {"mode": "max-autotune", "fullgraph": True, "dynamic": False}
        assert w.__wrapped__ is w.compiled and w.compiled.compile_kwargs == kwargs
        assert w.__qualname__ == fn.__qualname__                           # update_wrapper copied the metadata


def test_the_original_compile_is_restored_when_the_build_raises() -> None:
    mod = _module()
    fake = mod.compile
    with pytest.raises(RuntimeError, match="build failed"), compile_mode_patch("vos_cg_clone", module=mod):
        mod.compile(_sam2_fn("MaskDecoder.forward"))
        raise RuntimeError("build failed")
    assert mod.compile is fake


def test_the_real_torch_compile_is_patched_only_inside_the_context(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCompile()
    monkeypatch.setattr(torch, "compile", fake)
    with compile_mode_patch("vos_nocg") as rewritten:
        assert torch.compile is not fake
        torch.compile(_sam2_fn("PromptEncoder.forward"), mode="max-autotune")
    assert torch.compile is fake and rewritten == ["PromptEncoder.forward"]
    assert fake.calls[0][1] == {"mode": NOCG_MODE}
    with pytest.raises(KeyError), compile_mode_patch("vos_nocg"):
        raise KeyError("boom")
    assert torch.compile is fake


@pytest.mark.parametrize("mode", ["none", "bogus"])
def test_none_and_unknown_modes_patch_nothing(mode: str) -> None:
    mod = _module()
    fake = mod.compile
    with pytest.raises(ValueError, match="mask_compile|vos mode"), compile_mode_patch(mode, module=mod):
        pass
    assert mod.compile is fake


def test_cudagraph_step_marks_the_step_first_and_clones_every_tensor_output() -> None:
    events: list[str] = []
    held = torch.arange(6.0)
    out = {"vision_features": held, "vision_pos_enc": [held * 2, (held + 1, 3)], "n": 7}

    def compiled(x: torch.Tensor, scale: float = 1.0) -> Any:
        events.append(f"call {scale}")
        return out

    step = CudagraphStep(compiled, mark_step=lambda: events.append("mark"))
    got = step(torch.ones(1), scale=2.0)
    assert events == ["mark", "call 2.0"]
    assert got["n"] == 7 and isinstance(got["vision_pos_enc"][1], tuple) and got["vision_pos_enc"][1][1] == 3
    leaves = [(got["vision_features"], held), (got["vision_pos_enc"][0], out["vision_pos_enc"][0]),
              (got["vision_pos_enc"][1][0], out["vision_pos_enc"][1][0])]
    for clone, source in leaves:
        assert torch.equal(clone, source) and clone.data_ptr() != source.data_ptr()
    held.add_(100.0)                                                      # the next replay overwrites its output
    assert torch.equal(got["vision_features"], torch.arange(6.0))
    step(torch.ones(1))
    assert events == ["mark", "call 2.0", "mark", "call 1.0"]            # one mark per call


# --- the loader ------------------------------------------------------------------------------------------------------


def _fake_build(record: list[dict[str, Any]], compile_names: tuple[str, ...] = tuple(sorted(VOS_COMPILED)),
                raise_after: bool = False) -> Any:
    def build(model_id: str, **kwargs: Any) -> Any:
        record.append({"model_id": model_id, **kwargs, "compile": torch.compile})
        for name in compile_names:                        # what SAM2VideoPredictorVOS's constructor does
            torch.compile(_sam2_fn(name), mode="max-autotune", fullgraph=True, dynamic=False)
        if raise_after:
            raise RuntimeError("checkpoint load failed")
        return "predictor"
    return build


def _patch_build(monkeypatch: pytest.MonkeyPatch, build: Any) -> None:
    real = masks_sam2._sam2_symbol
    monkeypatch.setattr(masks_sam2, "_sam2_symbol",
                        lambda module, name: build if name == "build_sam2_video_predictor_hf" else real(module, name))
    monkeypatch.setattr(masks_sam2, "_under_sam2_hydra", lambda fn: fn())   # Hydra guard: tested in test_masks_sam2


def test_none_keeps_the_loader_call_exactly_and_never_touches_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCompile()
    monkeypatch.setattr(torch, "compile", fake)
    record: list[dict[str, Any]] = []
    _patch_build(monkeypatch, _fake_build(record, compile_names=()))
    assert masks_sam2.load_sam2_predictor(device="cpu", model_id="m") == "predictor"
    assert masks_sam2.load_sam2_predictor(device="cpu", model_id="m", mask_compile="none") == "predictor"
    assert record == [{"model_id": "m", "device": "cpu", "compile": fake}] * 2   # no vos_optimized, no patch


@pytest.mark.parametrize("mode", ["vos_nocg", "vos_cg_clone", "vos_det"])
def test_vos_modes_build_the_vos_predictor_under_the_patch_and_restore_it(
        monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    fake = FakeCompile()
    monkeypatch.setattr(torch, "compile", fake)
    record: list[dict[str, Any]] = []
    _patch_build(monkeypatch, _fake_build(record))
    assert masks_sam2.load_sam2_predictor(device="cpu", model_id="m", mask_compile=mode) == "predictor"
    assert record[0]["vos_optimized"] is True and record[0]["device"] == "cpu" and record[0]["model_id"] == "m"
    assert record[0]["compile"] is not fake                                # patched during the build ...
    assert torch.compile is fake                                            # ... and only then
    if mode == "vos_det":
        assert all("mode" not in kw and kw["options"] == DET_OPTIONS for _, kw in fake.calls)
        assert len(fake.calls) == len(VOS_COMPILED)
        return
    want = NOCG_MODE if mode == "vos_nocg" else "max-autotune"
    assert [kw["mode"] for _, kw in fake.calls] == [want] * len(VOS_COMPILED)


def test_a_failing_vos_build_restores_compile_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCompile()
    monkeypatch.setattr(torch, "compile", fake)
    _patch_build(monkeypatch, _fake_build([], raise_after=True))
    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        masks_sam2.load_sam2_predictor(device="cpu", model_id="m", mask_compile="vos_cg_clone")
    assert torch.compile is fake


@pytest.mark.parametrize("names", [(), ("MemoryEncoder.forward",), (*sorted(VOS_COMPILED), "Extra.forward")])
def test_a_vos_build_that_does_not_compile_the_five_forwards_is_refused(
        monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    monkeypatch.setattr(torch, "compile", FakeCompile())
    _patch_build(monkeypatch, _fake_build([], compile_names=names))
    with pytest.raises(RuntimeError, match="sam2's VOS build changed"):
        masks_sam2.load_sam2_predictor(device="cpu", model_id="m", mask_compile="vos_nocg")


def test_an_unknown_mode_is_refused_before_sam2_is_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_import(module: str, name: str) -> Any:
        raise AssertionError("sam2 imported")

    monkeypatch.setattr(masks_sam2, "_sam2_symbol", no_import)
    with pytest.raises(ValueError, match="mask_compile"):
        masks_sam2.load_sam2_predictor(device="cpu", mask_compile="max-autotune")


def test_the_real_vos_build_compiles_exactly_the_five_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachability against the installed sam2: its real `SAM2VideoPredictorVOS` (tiny config, no checkpoint, CPU),
    built through `load_sam2_predictor` and the Hydra guard; the fake compile returns the eager forward."""
    pytest.importorskip("sam2")
    from sam2 import build_sam

    fake = FakeCompile()
    monkeypatch.setattr(torch, "compile", fake)
    monkeypatch.setattr(build_sam, "_hf_download",
                        lambda model_id: (build_sam.HF_MODEL_ID_TO_FILENAMES[model_id][0], None))   # no weights
    pred = masks_sam2.load_sam2_predictor(device="cpu", model_id="facebook/sam2.1-hiera-tiny",
                                          mask_compile="vos_cg_clone")
    assert type(pred).__name__ == "SAM2VideoPredictorVOS" and torch.compile is fake
    assert sorted(fn.__qualname__ for fn, _ in fake.calls) == sorted(VOS_COMPILED)
    for owner in (pred.image_encoder, pred.memory_encoder, pred.memory_attention, pred.sam_prompt_encoder,
                  pred.sam_mask_decoder):
        assert isinstance(owner.forward, CudagraphStep)
        assert owner.forward.compiled.compile_kwargs["mode"] == "max-autotune"
