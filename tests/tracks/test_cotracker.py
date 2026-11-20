"""Tests for the pure helpers in fastdo.tracks.cotracker and a
regression test for `_run_online`'s windowing against a duck-typed fake model.

`track_points_cotracker` itself needs the real CoTracker3 model (torch.hub download) and a GPU
to be meaningful, so it is exercised by the real-demo smoke run; here we only check its cheap,
model-independent input validation, the query-sampling helper, and `_run_online`'s
windowing logic in isolation via a fake model -- `_run_online` already takes `model` as a plain
parameter, so no refactor was needed to make it testable this way.
"""

from __future__ import annotations

import pytest
import torch

from fastdo.tracks.cotracker import (
    _run_online,
    sample_query_points_in_mask,
    track_points_cotracker,
    tracks_in_masks,
)


def test_sample_query_points_in_mask_are_inside_the_mask() -> None:
    mask = torch.zeros(20, 30, dtype=torch.bool)
    mask[5:10, 8:12] = True
    gen = torch.Generator().manual_seed(0)

    pts = sample_query_points_in_mask(mask, n_points=16, generator=gen)

    assert pts.shape == (16, 2)
    x, y = pts[:, 0].long(), pts[:, 1].long()
    assert bool(mask[y, x].all())


def test_sample_query_points_with_replacement_when_mask_is_small() -> None:
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 1] = True
    gen = torch.Generator().manual_seed(1)

    pts = sample_query_points_in_mask(mask, n_points=10, generator=gen)

    assert pts.shape == (10, 2)  # sampled with replacement from only 2 True pixels


def test_sample_query_points_empty_mask_raises() -> None:
    mask = torch.zeros(4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="no True pixels"):
        sample_query_points_in_mask(mask, n_points=5)


def test_sample_query_points_wrong_ndim_raises() -> None:
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="H,W"):
        sample_query_points_in_mask(mask, n_points=5)


def test_track_points_out_of_range_keyframe_raises_without_loading_the_model() -> None:
    frames = torch.zeros(5, 8, 8, 3, dtype=torch.uint8)
    query_uv = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="keyframe_index"):
        track_points_cotracker(frames, keyframe_index=5, query_uv=query_uv)


# --- `_run_online` windowing regression --------------------------------------------------


class _FakeCoTrackerModel:
    """Duck-typed ``CoTrackerOnlinePredictor`` stand-in -- no GPU, no weights, no network.

    Records the ``(start, end)`` frame range of every non-priming call, decoded straight from
    the video content (each frame's single pixel holds its own frame index -- see
    ``_make_content_encoded_clip``), and reproduces the *real* online model's output-length
    growth rule, reverse engineered empirically from ``facebookresearch/co-tracker``'s
    ``cotracker3_online`` (see ``tracks/cotracker.py``'s module docstring for the citation): the
    first real call's output covers exactly its own chunk length; every later call's output grows
    by ``chunk_length - (window - step)`` over the previous call's output length, regardless of
    which frames the chunk actually contains -- i.e. the real model assumes each call brings
    exactly ``chunk_length - (window - step)`` *new* frames, which only holds when every chunk's
    start is step-aligned. This is exactly what made the old, buggy ``start = max(end - window,
    0)`` windowing corrupt the frame count: it kept every chunk (including the last) at the full
    window length, so the model "believed" it saw a full `step` new frames on that final call
    too, producing 48 output frames for a 46-frame input instead of 46 (empirically confirmed
    against the real model).
    """

    def __init__(self, step: int = 8, n_queries: int = 1) -> None:
        self.step = step
        self.n_queries = n_queries
        self.calls: list[tuple[int, int]] = []
        self._total = 0

    def __call__(self, video_chunk: torch.Tensor, is_first_step: bool = False,
                queries: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if is_first_step:
            return None, None
        length = video_chunk.shape[1]
        frame_ids = video_chunk[0, :, 0, 0, 0].round().long().tolist()
        start, end = frame_ids[0], frame_ids[-1] + 1
        self.calls.append((start, end))
        overlap = 2 * self.step - self.step  # = window - step
        self._total = length if self._total == 0 else self._total + (length - overlap)
        tracks = torch.zeros(1, self._total, self.n_queries, 2)
        vis = torch.zeros(1, self._total, self.n_queries, dtype=torch.bool)
        return tracks, vis


def _make_content_encoded_clip(t: int) -> torch.Tensor:
    """``(T,1,1,3)`` float; channel 0 holds each frame's own index, for the fake model to decode."""
    clip = torch.zeros(t, 1, 1, 3)
    clip[:, 0, 0, 0] = torch.arange(t, dtype=torch.float32)
    return clip


def test_run_online_keeps_start_fixed_and_only_shrinks_end_for_a_non_multiple_length() -> None:
    clip = _make_content_encoded_clip(46)  # the real-demo stride-3 10 fps frame count
    model = _FakeCoTrackerModel(step=8, n_queries=2)
    query_uv = torch.zeros(2, 2)

    tracks, vis = _run_online(model, clip, query_uv, device="cpu")

    assert model.calls == [(0, 16), (8, 24), (16, 32), (24, 40), (32, 46)]
    assert tracks.shape[0] == 46 and vis.shape[0] == 46  # regression: was 48 with the old windowing


def test_run_online_exact_window_multiple_length() -> None:
    clip = _make_content_encoded_clip(40)
    model = _FakeCoTrackerModel(step=8, n_queries=1)
    query_uv = torch.zeros(1, 2)

    tracks, vis = _run_online(model, clip, query_uv, device="cpu")

    assert model.calls == [(0, 16), (8, 24), (16, 32), (24, 40)]
    assert tracks.shape[0] == 40 and vis.shape[0] == 40


def test_tracks_in_masks_requires_in_image_and_in_mask() -> None:
    masks = torch.zeros(2, 4, 5, dtype=torch.bool)
    masks[0, 1, 2] = True
    masks[1, 3, 4] = True
    tracks = torch.tensor([[[2.2, 0.9], [4.0, 3.0], [-0.6, 1.0]],
                           [[2.0, 1.0], [4.4, 3.2], [9.0, 3.0]]])
    hit = tracks_in_masks(masks, tracks)
    assert hit.tolist() == [[True, False, False], [False, True, False]]


# --- windows cast to float32 on the device, not a whole-clip float32 upload ----------------------------------


class _RecordingModel(_FakeCoTrackerModel):
    """Keeps every tensor the model is handed (priming call included)."""

    def __init__(self, step: int = 8, n_queries: int = 1) -> None:
        super().__init__(step, n_queries)
        self.seen: list[torch.Tensor] = []

    def __call__(self, video_chunk: torch.Tensor, is_first_step: bool = False,
                 queries: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self.seen.append(video_chunk.clone(memory_format=torch.preserve_format))
        return super().__call__(video_chunk, is_first_step, queries)


def _run_online_whole_clip_reference(model, clip: torch.Tensor, query_uv: torch.Tensor) -> None:
    """The earlier feeding: the whole clip cast to float32 before any window is sliced."""
    t = int(clip.shape[0])
    video = clip.permute(0, 3, 1, 2).to(device="cpu", dtype=torch.float32)[None]
    queries = torch.cat([torch.zeros(query_uv.shape[0], 1), query_uv.float()], dim=1)[None]
    model(video[:, :1], is_first_step=True, queries=queries)
    ind = 0
    while ind < t:
        end = min(ind + 2 * model.step, t)
        model(video[:, ind:end], is_first_step=False)
        if end == t:
            break
        ind += model.step


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64])
def test_run_online_feeds_the_model_exactly_what_the_whole_clip_cast_fed(dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(208)
    clip = torch.randint(0, 256, (37, 6, 10, 3), generator=g).to(dtype)
    clip[:, 0, 0, 0] = torch.arange(37).to(dtype)          # the fake model decodes frame ids from this pixel
    if dtype == torch.float64:
        clip = clip + torch.rand(clip.shape, generator=g, dtype=dtype) / 3   # values float32 must round
    query_uv = torch.rand(2, 2, generator=g) * 5
    new, ref = _RecordingModel(step=8, n_queries=2), _RecordingModel(step=8, n_queries=2)
    _run_online(new, clip, query_uv, device="cpu")
    _run_online_whole_clip_reference(ref, clip, query_uv)
    assert len(new.seen) == len(ref.seen) == 1 + len(ref.calls) and new.calls == ref.calls
    for a, b in zip(new.seen, ref.seen, strict=True):
        assert a.dtype == b.dtype == torch.float32 and a.stride() == b.stride() and torch.equal(a, b)
