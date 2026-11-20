"""MoGe-2's base ViT token count (`num_tokens`) through `fastdo.geometry.moge_model` and the MoGe overlap.

- The transcription: `moge_overlap.infer_transcribed(..., num_tokens=N)` equals moge's REAL `MoGeModel.infer(...,
  num_tokens=N)` bit for bit on the fake-model harness (`tests.geometry.moge_fakes.TinyMoGe`, whose depth moves
  with the token count), for N in {None, 2500}, with and without `fov_x`; None is infer's resolution_level-9 formula
  (3600 tokens) and 2500 is used as given.
- The paths: `run_moge_batches(overlap=True)` equals `overlap=False` at 2500 (tail batch, with/without intrinsics), and
  the value reaches every forward on both; `None` never reaches `infer` as a keyword (the knob-free call), for
  `run_moge`, `run_moge_batches` and `moge_intrinsics`.
- Validation: None or an int in [1200, 3600]; anything else refused before any model call.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

pytest.importorskip("moge")

from fastdo.geometry.moge_model import (
    NUM_TOKENS_RANGE,
    check_num_tokens,
    moge_intrinsics,
    run_moge,
    run_moge_batches,
)
from fastdo.geometry.moge_overlap import infer_transcribed
from tests.engine.fakes import FakeMoGe
from tests.geometry.moge_fakes import TinyMoGe, tag_frames

K = torch.tensor([[14.0, 0.0, 7.2], [0.0, 13.0, 5.9], [0.0, 0.0, 1.0]])
FOV_X = 61.3
DEFAULT_TOKENS = 3600                     # infer's resolution_level 9 on MoGe-2's num_tokens_range [1200, 3600]


def _bits(t: Tensor) -> tuple[str, tuple[int, ...], bytes]:
    return str(t.dtype), tuple(t.shape), t.contiguous().numpy().tobytes()


def _image(n: int, seed: int = 0) -> Tensor:
    return tag_frames(n, seed=seed).permute(0, 3, 1, 2).float() / 255.0


def test_the_range_is_moge_s_own() -> None:
    import inspect

    from moge.model.v2 import MoGeModel

    default = inspect.signature(MoGeModel.__init__).parameters["num_tokens_range"].default
    assert tuple(default) == NUM_TOKENS_RANGE == (1200, DEFAULT_TOKENS)


# --- the transcription against the real MoGeModel.infer ----------------------------------------------------------------


@pytest.mark.parametrize("fov_x", [None, FOV_X])
@pytest.mark.parametrize("num_tokens", [None, 2500])
def test_transcription_equals_the_real_infer_at_a_token_count(num_tokens: int | None, fov_x: float | None) -> None:
    """overlap's `stage_a`+`stage_b` at `num_tokens=N` == `MoGeModel.infer(num_tokens=N)`, every tensor bit for bit;
    frame 1 has an empty mask and frame 2 negative depth, as in the overlap transcription test."""
    real, transcribed = TinyMoGe(empty=[1], negative=[2]), TinyMoGe(empty=[1], negative=[2])
    image = _image(4)
    kwargs: dict[str, Any] = {} if fov_x is None else {"fov_x": fov_x}
    if num_tokens is not None:
        kwargs["num_tokens"] = num_tokens
    want = real.infer(image, **kwargs)
    got = infer_transcribed(transcribed, image, fov_x, num_tokens=num_tokens)
    assert sorted(got) == sorted(want) and {"points", "intrinsics", "depth", "mask"} <= set(want)
    for key in want:
        assert _bits(got[key]) == _bits(want[key]), key
    expected = DEFAULT_TOKENS if num_tokens is None else num_tokens
    assert real.tokens == transcribed.tokens == [expected]


def test_the_fake_s_output_moves_with_the_token_count() -> None:
    """Sanity of the fixture: without it the transcription test could not tell a dropped `num_tokens` from a kept one."""
    image = _image(4)
    default = infer_transcribed(TinyMoGe(), image, FOV_X)
    explicit_default = infer_transcribed(TinyMoGe(), image, FOV_X, num_tokens=DEFAULT_TOKENS)
    fewer = infer_transcribed(TinyMoGe(), image, FOV_X, num_tokens=2500)
    assert _bits(default["points"]) == _bits(explicit_default["points"])      # 3600 IS the default
    assert _bits(default["depth"]) != _bits(fewer["depth"])


# --- run_moge_batches: both paths, and None never reaches infer ---------------------------------------------------------


def _blocks(model: TinyMoGe, frames: Tensor, overlap: bool, **kwargs: Any) -> list[tuple[int, Any]]:
    blocks: list[tuple[int, Any]] = []
    run_moge_batches(frames, lambda start, points: blocks.append((start, _bits(points))), device="cpu", model=model,
                     overlap=overlap, **kwargs)
    return blocks


@pytest.mark.parametrize("intrinsics", [None, K])
@pytest.mark.parametrize(("n", "batch_size"), [(5, 2), (4, 3)])
def test_overlap_equals_serial_at_2500_tokens(intrinsics: Tensor | None, n: int, batch_size: int) -> None:
    frames = tag_frames(n)
    serial, overlapped = TinyMoGe(empty=[1], negative=[2]), TinyMoGe(empty=[1], negative=[2])
    plain = _blocks(serial, frames, False, intrinsics=intrinsics, batch_size=batch_size, num_tokens=2500)
    piped = _blocks(overlapped, frames, True, intrinsics=intrinsics, batch_size=batch_size, num_tokens=2500)
    assert piped == plain and [start for start, _ in plain] == list(range(0, n, batch_size))
    n_batches = len(plain)
    assert serial.tokens == overlapped.tokens == [2500] * n_batches
    default = _blocks(TinyMoGe(empty=[1], negative=[2]), frames, True, intrinsics=intrinsics, batch_size=batch_size)
    assert default != plain                                                  # the knob acts on the overlapped path


@pytest.mark.parametrize("overlap", [False, True])
def test_none_is_today_s_call_on_both_paths(overlap: bool) -> None:
    """Default and explicit None: the same blocks, and every forward at infer's own 3600."""
    frames = tag_frames(5)
    for intrinsics in (None, K):
        implicit, explicit = TinyMoGe(empty=[3]), TinyMoGe(empty=[3])
        a = _blocks(implicit, frames, overlap, intrinsics=intrinsics)
        b = _blocks(explicit, frames, overlap, intrinsics=intrinsics, num_tokens=None)
        assert a == b and implicit.tokens == explicit.tokens == [DEFAULT_TOKENS] * 3


def test_none_is_never_passed_to_infer_and_a_value_is() -> None:
    """The serial path's `infer` kwargs (FakeMoGe records them): None leaves them exactly as without the knob."""
    frames = torch.from_numpy(tag_frames(3).numpy())
    for tokens, extra in ((None, {}), (2500, {"num_tokens": 2500})):
        for intrinsics, base in ((None, set()), (K, {"fov_x"})):
            model = FakeMoGe()
            run_moge(frames, device="cpu", model=model, intrinsics=intrinsics, num_tokens=tokens)
            assert [set(kw) for kw in model.kwargs] == [base | set(extra)] * 2
            assert all(kw.get("num_tokens") == extra.get("num_tokens") for kw in model.kwargs)
        model = FakeMoGe()
        moge_intrinsics(frames, device="cpu", model=model, num_tokens=tokens)
        assert model.kwargs == [extra, extra]


def test_run_moge_passes_num_tokens_through_bit_identically() -> None:
    frames = tag_frames(5)
    for overlap in (False, True):
        via_run = run_moge(frames, device="cpu", model=TinyMoGe(), intrinsics=K, overlap=overlap, num_tokens=1800)
        blocks: list[Tensor] = []

        def sink(start: int, points: Tensor, blocks: list[Tensor] = blocks) -> None:
            blocks.append(points.clone())

        run_moge_batches(frames, sink, device="cpu", model=TinyMoGe(), intrinsics=K, overlap=overlap, num_tokens=1800)
        assert via_run.numpy().tobytes() == torch.cat(blocks).numpy().tobytes()


# --- validation ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("good", [None, 1200, 2500, 3600])
def test_check_num_tokens_accepts_none_and_the_range(good: int | None) -> None:
    assert check_num_tokens(good) == good


@pytest.mark.parametrize("bad", [1199, 3601, 0, -2500, 2500.0, True, "2500"])
def test_every_entry_point_refuses_a_bad_token_count_before_the_model(bad: Any) -> None:
    frames = tag_frames(3)
    model = FakeMoGe()
    for call in (lambda: run_moge(frames, device="cpu", model=model, num_tokens=bad),
                 lambda: run_moge(frames, device="cpu", model=model, overlap=True, num_tokens=bad),
                 lambda: run_moge_batches(frames, lambda start, points: None, device="cpu", model=model,
                                          num_tokens=bad),
                 lambda: moge_intrinsics(frames, device="cpu", model=model, num_tokens=bad)):
        with pytest.raises(ValueError, match="num_tokens"):
            call()
    assert model.kwargs == []
