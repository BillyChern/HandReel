"""Overlapped MoGe-2 inference (`fastdo.geometry.moge_overlap`, `run_moge_batches(overlap=True)`).

CPU tests on modules whose `infer` IS moge's `MoGeModel.infer` (`tests.geometry.moge_fakes`):

- the transcription (`stage_a` + `stage_b`) against the real `infer` on the same forward outputs, every returned tensor
  bit for bit, through both scipy solves (`solve_optimal_focal_shift` without `fov_x`, `solve_optimal_shift` with it),
  an empty-mask frame, a frame with negative depth, and infer's no-mask and normal branches;
- `run_moge_batches(overlap=True)` against `overlap=False`: every sink block bit-identical over odd batch counts (a tail
  batch), with and without intrinsics, and the pipeline order A(0), A(1), B(0), A(2), B(1), ...;
- the guards: the moge version, commit and transcribed-source hash are pinned, and the overlap refuses a model whose
  infer is not `MoGeModel.infer` and a moge whose source moved.

The GPU side (pinned/async copies, streams, the CUDA graph) needs a GPU and the real model; it is not tested here.
"""

from __future__ import annotations

import importlib.metadata
import json
from typing import Any

import pytest
import torch
from torch import Tensor

pytest.importorskip("moge")

import scipy.optimize

from fastdo.geometry import moge_overlap
from fastdo.geometry.moge_model import run_moge, run_moge_batches
from fastdo.geometry.moge_overlap import (
    PINNED_MOGE_COMMIT,
    PINNED_SOURCE_SHA256,
    infer_transcribed,
    moge_source_sha256,
)
from tests.geometry.moge_fakes import TinyMoGe, tag_frames

K = torch.tensor([[14.0, 0.0, 7.2], [0.0, 13.0, 5.9], [0.0, 0.0, 1.0]])
FOV_X = 61.3


def _bits(t: Tensor) -> tuple[str, tuple[int, ...], bytes]:
    return str(t.dtype), tuple(t.shape), t.contiguous().numpy().tobytes()


def _image(n: int, seed: int = 0) -> Tensor:
    return tag_frames(n, seed=seed).permute(0, 3, 1, 2).float() / 255.0


def _count_solves(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts scipy `least_squares` calls (moge's solves import it from `scipy.optimize` at call time)."""
    calls: list[int] = []
    real = scipy.optimize.least_squares

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "least_squares", counted)
    return calls


# --- the transcription against the real MoGeModel.infer ----------------------------------------------------------------


@pytest.mark.parametrize("fov_x", [None, FOV_X])
@pytest.mark.parametrize("variant", [{}, {"with_normal": True}, {"with_mask": False}])
def test_transcription_is_bit_identical_to_the_real_infer(fov_x: float | None, variant: dict[str, Any],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Frame 1 has an empty mask, frame 2 negative depth under its mask; every tensor infer returns, bit for bit."""
    model = TinyMoGe(empty=[1], negative=[2], **variant)
    image = _image(4)
    solves = _count_solves(monkeypatch)
    kwargs = {} if fov_x is None else {"fov_x": fov_x}
    want = model.infer(image, **kwargs)
    n_real = len(solves)
    got = infer_transcribed(model, image, fov_x)
    assert sorted(got) == sorted(want)
    expected_keys = {"points", "intrinsics", "depth"} | ({"mask"} if variant.get("with_mask", True) else set())
    assert set(want) == expected_keys | ({"normal"} if variant.get("with_normal") else set())
    for key in want:
        assert _bits(got[key]) == _bits(want[key]), key
    # The scipy path ran for every frame with >= 2 masked low-res pixels -- all but the empty-mask frame -- in both.
    assert n_real == len(solves) - n_real == (4 if not variant.get("with_mask", True) else 3)
    if variant.get("with_mask", True):
        assert torch.isinf(got["points"][1]).all() and torch.isinf(got["depth"][1]).all()   # the empty-mask frame
        assert not got["mask"][1].any()


def test_the_negative_depth_frame_exercises_infer_s_mask_update() -> None:
    """Sanity of the fixture: in the negative frame `mask_binary &= z > 0` removes pixels the raw mask kept."""
    model = TinyMoGe(negative=[2])
    image = _image(4)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16):
        raw = model.forward(image, num_tokens=3600)
    out = model.infer(image)
    assert int((raw["mask"][2].float() > 0.5).sum()) > int(out["mask"][2].sum())


# --- run_moge_batches(overlap=True) vs overlap=False --------------------------------------------------------------------


def _blocks(model: TinyMoGe, frames: Tensor, overlap: bool, **kwargs: Any) -> list[tuple[int, Any]]:
    blocks: list[tuple[int, Any]] = []

    def sink(start: int, points: Tensor) -> None:
        model.log.append(("sink", start))
        blocks.append((start, _bits(points)))

    run_moge_batches(frames, sink, device="cpu", model=model, overlap=overlap, **kwargs)
    return blocks


@pytest.mark.parametrize("intrinsics", [None, K])
@pytest.mark.parametrize(("n", "batch_size"), [(5, 2), (7, 3), (1, 2), (4, 2)])
def test_overlap_sink_blocks_are_bit_identical_including_the_tail_batch(intrinsics: Tensor | None, n: int,
                                                                        batch_size: int) -> None:
    frames = tag_frames(n)
    plain = _blocks(TinyMoGe(empty=[1], negative=[2]), frames, False, intrinsics=intrinsics, batch_size=batch_size)
    overlapped = _blocks(TinyMoGe(empty=[1], negative=[2]), frames, True, intrinsics=intrinsics,
                         batch_size=batch_size)
    assert [start for start, _ in plain] == list(range(0, n, batch_size))
    assert overlapped == plain


def test_overlap_runs_the_pipeline_order() -> None:
    """A(k+1)'s forward is issued before B(k)'s sink: F0 F1 S0 F2 S1 F3 S2 S3 (in batch starts)."""
    frames = tag_frames(7)
    serial, overlapped = TinyMoGe(), TinyMoGe()
    _blocks(serial, frames, False, batch_size=2)
    _blocks(overlapped, frames, True, batch_size=2)
    forwards = [("forward", [s, s + 1] if s < 6 else [6]) for s in range(0, 7, 2)]
    sinks = [("sink", s) for s in range(0, 7, 2)]
    assert serial.log == [x for pair in zip(forwards, sinks, strict=True) for x in pair]
    assert overlapped.log == [forwards[0], forwards[1], sinks[0], forwards[2], sinks[1], forwards[3], sinks[2],
                              sinks[3]]


def test_run_moge_passes_overlap_through() -> None:
    frames = tag_frames(5)
    for intrinsics in (None, K):
        plain = run_moge(frames, device="cpu", model=TinyMoGe(empty=[3]), intrinsics=intrinsics)
        overlapped = run_moge(frames, device="cpu", model=TinyMoGe(empty=[3]), intrinsics=intrinsics, overlap=True)
        assert overlapped.numpy().tobytes() == plain.numpy().tobytes()
        assert torch.isnan(plain[3]).all() and torch.isfinite(plain[0]).any()


# --- the guards ---------------------------------------------------------------------------------------------------------


def test_the_transcribed_moge_is_pinned() -> None:
    """A moge upgrade must fail here: re-check moge_overlap's transcription line by line, then re-pin."""
    assert importlib.metadata.version("moge") == "2.0.0"
    direct = json.loads(importlib.metadata.distribution("moge").read_text("direct_url.json") or "{}")
    assert direct.get("vcs_info", {}).get("commit_id") == PINNED_MOGE_COMMIT
    assert moge_source_sha256() == PINNED_SOURCE_SHA256


class _NotMoGe:
    def infer(self, image: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        return {"points": image.permute(0, 2, 3, 1).contiguous()}


def test_overlap_refuses_a_model_whose_infer_is_not_moge_s() -> None:
    frames = tag_frames(3)
    with pytest.raises(ValueError, match="MoGeModel.infer"):
        run_moge_batches(frames, lambda start, points: None, device="cpu", model=_NotMoGe(), overlap=True)
    model = TinyMoGe()
    model.infer = lambda image, **kwargs: {}  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="MoGeModel.infer"):
        run_moge_batches(frames, lambda start, points: None, device="cpu", model=model, overlap=True)
    run_moge_batches(frames, lambda start, points: None, device="cpu", model=_NotMoGe())   # the serial path: any model


def test_overlap_refuses_a_moved_moge_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(moge_overlap, "PINNED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="re-pin"):
        run_moge_batches(tag_frames(3), lambda start, points: None, device="cpu", model=TinyMoGe(), overlap=True)


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_overlap_must_be_a_bool(bad: Any) -> None:
    with pytest.raises(ValueError, match="overlap"):
        run_moge_batches(tag_frames(3), lambda start, points: None, device="cpu", model=TinyMoGe(), overlap=bad)
