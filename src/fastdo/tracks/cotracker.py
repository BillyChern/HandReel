"""CoTracker3 online 2D point tracking from queries sampled in a keyframe mask.

Loads ``cotracker3_online`` via ``torch.hub.load("facebookresearch/co-tracker:<commit>", "cotracker3_online")`` at the
pinned commit (``cotracker_hub_ref``; cached under ``$TORCH_HOME`` by ``scripts/fetch_weights.py``) and drives its
windowed online API.

``CoTrackerOnlinePredictor`` is causal: a single call only extends tracks *forward* in time, so
``track_points_cotracker`` runs it twice -- once on ``frames[keyframe_index:]`` and once on the
reversed prefix ``frames[:keyframe_index+1][::-1]`` -- and stitches the two halves back into a
full ``(N,J,...)`` array, mirroring the forward/reverse pattern used for SAM 2 mask propagation.

The online model's windowed loop (verified empirically against ``T`` in
``{1, 5, 16, 17, 46, 138}``, matching this repo's real 10 fps and full-rate whisking-demo frame
counts): feed a priming call with ``is_first_step=True`` and the query points (only that call's
frame's ``H, W`` matter -- the frame content itself is ignored for priming), then repeatedly call
with ``video[:, ind:min(ind + window, T)]`` for ``ind = 0, step, 2*step, ...`` where
``window = 2 * model.step``, stopping once a chunk reaches ``T``. Each call's output covers
``[0, min(ind + window, T))`` (it accumulates, not a sliding view), so the *last* call's output is
already the full-length result -- no separate concatenation step is needed. An earlier version of
this loop instead kept every chunk at exactly ``window`` frames by shifting the *start* index back
for the final chunk when ``T`` was not a multiple of ``model.step``; that breaks the model's
internal step accounting (observed: a 46-frame clip returned 48 frames of output instead of 46).
Keeping ``start = ind`` fixed and only shrinking the *end* of the final chunk is what reproduces
the reference ``online_demo.py`` script's (more convoluted, but equivalent) windowing exactly.
A regression test covers this against a duck-typed fake model (no GPU/weights
needed) that reproduces the real model's growth-per-call rule; see ``tests/tracks/
test_cotracker.py``. ``_run_online`` already took ``model`` as a plain parameter, so no source
change was needed to make that test possible; ``track_points_cotracker`` additionally gained an
injectable ``model`` parameter for symmetry with ``shape.masks_sam2.propagate_masks_sam2``.

No PyPI package is needed for CoTracker3 (``torch`` -- a base dependency -- is enough), but the
``torch.hub.load`` call does need network access to github.com/pytorch.org on first use, or a
pre-warmed ``$TORCH_HOME`` cache;
``_load_cotracker_model`` wraps a failed load in an actionable ``RuntimeError`` instead of letting
whatever ``torch.hub`` raises surface directly.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from fastdo.model_pins import COTRACKER_HUB_COMMIT as _PINNED_COTRACKER_HUB_COMMIT

__all__ = ["COTRACKER_HUB_COMMIT", "COTRACKER_HUB_REPO", "cotracker_hub_ref", "sample_query_points_in_mask",
           "track_points_cotracker", "tracks_in_masks"]

COTRACKER_HUB_REPO = "facebookresearch/co-tracker"
# The commit the paper's runs loaded: their cached torch.hub `main` equals it file for file (74/74 tracked files, git
# blob hashes, checked 2026-10-27; upstream main has not moved since 2025-01-21). It is the default;
# `scripts/fetch_weights.py` fetches exactly this commit. FASTDO_COTRACKER_HUB_REF overrides it (`main` loads the
# moving branch). The value lives in `fastdo.model_pins`, the table fetch_weights.py and run_example.py read.
COTRACKER_HUB_COMMIT = _PINNED_COTRACKER_HUB_COMMIT


def cotracker_hub_ref() -> str:
    """The `torch.hub` repo string `owner/repo:<ref>`: `$FASTDO_COTRACKER_HUB_REF` when set, else the pinned
    `COTRACKER_HUB_COMMIT`. `main` gives the unpinned repo string (torch.hub's moving `main`).

    Raises:
        ValueError: When the variable is set but empty.
    """
    ref = os.environ.get("FASTDO_COTRACKER_HUB_REF")
    if ref is None:
        return f"{COTRACKER_HUB_REPO}:{COTRACKER_HUB_COMMIT}"
    if ref == "main":
        return COTRACKER_HUB_REPO
    if not ref:
        raise ValueError("FASTDO_COTRACKER_HUB_REF is set but empty; unset it, or set a commit such as "
                         f"{COTRACKER_HUB_COMMIT}")
    return f"{COTRACKER_HUB_REPO}:{ref}"


def sample_query_points_in_mask(mask: Tensor, n_points: int, generator: torch.Generator | None = None) -> Tensor:
    """Pure helper: samples pixel ``(x,y)`` query points uniformly at random inside a mask.

    Args:
        mask: ``(H,W)`` boolean (or 0/1) mask.
        n_points: Number of query points to sample. Sampled with replacement when this exceeds
            the number of True pixels in ``mask``.
        generator: Optional torch RNG for reproducibility.

    Returns:
        ``(n_points,2)`` float32 pixel coordinates ``(x,y)``.

    Raises:
        ValueError: If ``mask`` is not 2D, or has no True pixels.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H,W), got {tuple(mask.shape)}")
    ys, xs = torch.nonzero(mask.to(torch.bool), as_tuple=True)
    if ys.numel() == 0:
        raise ValueError("mask has no True pixels to sample query points from")
    if n_points > ys.numel():
        idx = torch.randint(0, ys.numel(), (n_points,), generator=generator)
    else:
        idx = torch.randperm(ys.numel(), generator=generator)[:n_points]
    return torch.stack([xs[idx].to(torch.float32), ys[idx].to(torch.float32)], dim=-1)


def tracks_in_masks(masks: Tensor, tracks: Tensor) -> Tensor:
    """Pure helper: whether each tracked pixel lies inside the image and inside its frame's mask.

    Args:
        masks: ``(T,H,W)`` boolean per-frame object masks.
        tracks: ``(T,J,2)`` pixel ``(x,y)`` track locations (rounded to the nearest pixel).

    Returns:
        ``(T,J)`` bool.
    """
    t, h, w = masks.shape
    u = tracks[..., 0].round().long()
    v = tracks[..., 1].round().long()
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    frame_idx = torch.arange(t, device=masks.device).view(t, 1).expand_as(u)
    return inside & masks.to(torch.bool)[frame_idx, v.clamp(0, h - 1), u.clamp(0, w - 1)]


def track_points_cotracker(frames: Tensor, keyframe_index: int, query_uv: Tensor, device: str = "cuda",
                           model: Any = None) -> tuple[Tensor, Tensor]:
    """Tracks 2D points forward and backward from a keyframe with CoTracker3 online.

    Args:
        frames: ``(N,H,W,3)`` uint8 (or float) RGB frames.
        keyframe_index: Index into ``frames`` the query points are defined at.
        query_uv: ``(J,2)`` pixel ``(x,y)`` query coordinates at ``keyframe_index``.
        device: torch device string for the model (unused when ``model`` is given).
        model: A ``CoTrackerOnlinePredictor``-compatible object (callable, with a ``.step``
            attribute). ``None`` (the default) loads the real CoTracker3 online model via
            ``torch.hub``; tests can inject a fake here.

    Returns:
        ``(tracks, vis)``: ``tracks`` is ``(N,J,2)`` float32 pixel coordinates; ``vis`` is
        ``(N,J)`` bool visibility, both defined for every frame in ``frames``.

    Raises:
        ValueError: If ``keyframe_index`` is out of range for ``frames``.
    """
    n = int(frames.shape[0])
    if not 0 <= keyframe_index < n:
        raise ValueError(f"keyframe_index {keyframe_index} out of range [0,{n})")

    if model is None:
        model = _load_cotracker_model(device)

    forward_clip = frames[keyframe_index:]
    backward_clip = frames[: keyframe_index + 1].flip(0)

    fwd_tracks, fwd_vis = _run_online(model, forward_clip, query_uv, device)
    bwd_tracks, bwd_vis = _run_online(model, backward_clip, query_uv, device)

    j = query_uv.shape[0]
    tracks = torch.zeros(n, j, 2, dtype=torch.float32)
    vis = torch.zeros(n, j, dtype=torch.bool)
    tracks[keyframe_index:] = fwd_tracks
    vis[keyframe_index:] = fwd_vis
    tracks[: keyframe_index + 1] = bwd_tracks.flip(0)
    vis[: keyframe_index + 1] = bwd_vis.flip(0)
    return tracks, vis


def _load_cotracker_model(device: str) -> Any:
    """Loads CoTracker3 online via ``torch.hub`` -- untyped, no type stubs.

    Raises:
        RuntimeError: If the hub load fails, with an actionable hint (this needs no PyPI package,
            but does need network access to github.com/pytorch.org on first
            use, or a pre-warmed ``$TORCH_HOME`` cache).
    """
    repo = cotracker_hub_ref()
    try:
        model = torch.hub.load(repo, "cotracker3_online")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CoTracker3 via torch.hub ({repo!r}, "
            "'cotracker3_online'). This needs network access to github.com/pytorch.org on first "
            "use, or a pre-warmed $TORCH_HOME cache."
        ) from exc
    return model.to(device).eval()


def _run_online(model: Any, clip: Tensor, query_uv: Tensor, device: str) -> tuple[Tensor, Tensor]:
    """Feeds `clip` (T,H,W,3), queried at its own frame 0, through the online windowed API.

    ``model`` is a ``cotracker.predictor.CoTrackerOnlinePredictor`` -- untyped (``Any``) since
    that package ships no type stubs and its ``.step`` attribute isn't part of ``nn.Module``'s
    static interface.

    The clip goes to the device in its own dtype (uint8 for decoded video) and each window is cast to float32
    there. The model receives the same float32 values in the same layout as a whole-clip float32 upload, but a
    100-frame 1440x1920 clip no longer costs a 3.3 GB CPU-side float32 copy plus its transfer (scout: identical
    tracks and visibility, 3.1-3.6 s -> 1.34 s per clip, reserved 6.9-7.4 -> 3.5 GiB).
    """
    t = int(clip.shape[0])
    video = clip.permute(0, 3, 1, 2).to(device=device)[None]  # (1,T,3,H,W), source dtype
    j = query_uv.shape[0]
    queries = torch.cat([torch.zeros(j, 1, device=device), query_uv.to(device=device, dtype=torch.float32)],
                        dim=1)[None]  # (1,J,3): (t=0, x, y)

    def chunk(start: int, end: int) -> Tensor:
        return video[:, start:end].to(dtype=torch.float32)

    window = 2 * model.step
    model(chunk(0, 1), is_first_step=True, queries=queries)

    tracks: Tensor | None = None
    vis: Tensor | None = None
    ind = 0
    while ind < t:
        end = min(ind + window, t)
        pred_tracks, pred_vis = model(chunk(ind, end), is_first_step=False)
        if pred_tracks is not None:
            tracks, vis = pred_tracks, pred_vis
        if end == t:
            break
        ind += model.step

    if tracks is None or vis is None:
        raise RuntimeError("CoTracker online model produced no output")
    return tracks[0].detach().cpu(), vis[0].detach().cpu()
