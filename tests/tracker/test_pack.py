"""Tracker-model CPU tests P1-P4: frame/clip packing, TrackerInputs shapes and masks, non-finite frame isolation."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor

from fastdo.solver.banded import ClipLayout
from fastdo.solver.terms import TermOutput
from fastdo.tracker.clusters import mesh_cluster_ids
from fastdo.tracker.features import build_frame_tokens, plan_row_mass
from fastdo.tracker.normalize import cluster_geometry, cluster_tokens, point_tokens
from fastdo.tracker.pack import (
    TrackerInputs,
    build_tracker_inputs,
    clip_frame_mask,
    clips_to_frames,
    frames_to_clips,
)
from tests.tracker.synthetic_terms import make_terms_fixture
from tests.tracker.tracker_fixtures import (
    base_scene,
    fixture_tracker_inputs,
    nan_injection,
    two_clip_args,
)

F64 = torch.float64
TOKENS = ("frame", "track", "point", "cluster")
MASKS = ("track_mask", "point_mask", "cluster_mask")
NAMES = tuple(f.name for f in dataclasses.fields(TrackerInputs))


def _close(a: Tensor, b: Tensor) -> None:
    if b.dtype == torch.bool:
        assert torch.equal(a, b)
    else:
        torch.testing.assert_close(a, b, rtol=1e-12, atol=0.0)


def test_frames_to_clips_roundtrip() -> None:
    layout = ClipLayout.build((5, 3, 8), "cpu")
    assert layout.width == 8 and layout.n_clips == 3
    x = torch.arange(16 * 4).view(16, 4)
    clips = frames_to_clips(x, layout)
    mask = clip_frame_mask(layout)
    assert clips.shape == (3, 8, 4) and mask.shape == (3, 8) and mask.dtype == torch.bool
    assert mask.sum(1).tolist() == [5, 3, 8]
    assert torch.equal(clips[~mask], torch.zeros(24 - 16, 4, dtype=x.dtype))
    assert torch.equal(clips[mask], x)
    assert torch.equal(clips_to_frames(clips, layout), x)
    xb = x[:, 0] % 3 == 0
    back = clips_to_frames(frames_to_clips(xb, layout), layout)
    assert back.dtype == torch.bool and torch.equal(back, xb)
    single = ClipLayout.build((7,), "cpu")
    ms = clip_frame_mask(single)
    assert single.width == 8 and single.n_clips == 1 and ms.shape == (1, 8)
    assert not bool(ms[0, 7]) and bool(ms[0, :7].all())
    assert float(frames_to_clips(torch.arange(1.0, 8.0), single)[0, 7]) == 0.0


def test_build_tracker_inputs_shapes_and_masks() -> None:
    inp = build_tracker_inputs(**two_clip_args(6))
    shapes = {"frame": (2, 6, 110), "track": (2, 6, 16, 6), "point": (2, 6, 96, 7), "cluster": (2, 6, 64, 5)}
    for name, shape in shapes.items():
        assert getattr(inp, name).shape == shape and getattr(inp, name).dtype == F64, name
        mask = getattr(inp, "frame_mask" if name == "frame" else f"{name}_mask")
        assert mask.dtype == torch.bool and mask.shape == shape[:-1], name
    assert bool(inp.frame_mask.all())
    solo = fixture_tracker_inputs(make_terms_fixture(knn=16, seed=1))
    for name in NAMES:
        assert torch.equal(getattr(inp, name)[1], getattr(solo, name)[0]), name
    inp11 = build_tracker_inputs(**two_clip_args(5))            # frame 5 of fixture 0 dropped: T = 11, W = 6
    assert not bool(inp11.frame_mask[0, 5]) and bool(inp11.frame_mask[0, :5].all())
    for name in TOKENS:
        slot = getattr(inp11, name)[0, 5]
        assert torch.equal(slot, torch.zeros_like(slot)), name
    for name in MASKS:
        assert not bool(getattr(inp11, name)[0, 5].any()), name
    for name in NAMES:
        _close(getattr(inp11, name)[0, :5], getattr(inp, name)[0, :5])
        _close(getattr(inp11, name)[1], getattr(solo, name)[0])


def test_build_tracker_inputs_rejects_bad_shapes() -> None:
    args = two_clip_args(6)
    bad = [{"K": torch.eye(4, dtype=F64)}, {"centroids": torch.zeros(64, 2, dtype=F64)},
           {"centroids": args["centroids"][0], "fraction": torch.zeros(63, dtype=F64)},
           {"layout": ClipLayout.build((6, 5), "cpu")}, {"track_valid": torch.ones(12, 15, dtype=torch.bool)}]
    for change in bad:
        with pytest.raises(ValueError):
            build_tracker_inputs(**{**args, **change})


def _assert_frame_zeroed(inp: TrackerInputs, k: int) -> None:
    assert not bool(inp.frame_mask[0, k])
    for name in TOKENS:
        slot = getattr(inp, name)[0, k]
        assert torch.equal(slot, torch.zeros_like(slot)), name
    for name in MASKS:
        assert not bool(getattr(inp, name)[0, k].any()), name


def _assert_others_equal(inp: TrackerInputs, ref: TrackerInputs, k: int) -> None:
    others = [j for j in range(6) if j != k]
    for name in NAMES:
        _close(getattr(inp, name)[0, others], getattr(ref, name)[0, others])


def test_build_tracker_inputs_masks_nonfinite_frame() -> None:
    fs = make_terms_fixture(knn=16)
    pm_c, pm_n, nan_plan = nan_injection(fs)
    clean = fixture_tracker_inputs(fs, dataclasses.replace(base_scene(fs), pm=pm_c))
    bad = fixture_tracker_inputs(fs, dataclasses.replace(base_scene(fs), pm=pm_n), plan=nan_plan)
    # Preconditions: the injection reaches a positive-weight residual row and the raw point and cluster tokens.
    bad_rows = ~torch.isfinite(pm_n.residual).all(-1)
    assert bool((bad_rows[4] & (pm_n.weight[4] > 0)).any()) and not bool(bad_rows[[0, 1, 2, 3, 5]].any())
    L = fs.log_scale.exp() * fs.template_diameter
    ids = mesh_cluster_ids(fs.Y, 64, 0)
    pt = point_tokens(fs.X, pm_n, plan_row_mass(nan_plan), fs.poses, L)
    cen, frac = cluster_geometry(fs.Y, ids, fs.template_diameter)
    ct = cluster_tokens(build_frame_tokens(pm_n, fs.tr_raw, nan_plan, ids, fs.template_diameter, fs.poses,
                                           fs.log_scale), cen, frac)
    nonfinite = [False, False, False, False, True, False]
    assert (~torch.isfinite(pt)).flatten(1).any(1).tolist() == nonfinite
    assert (~torch.isfinite(ct)).flatten(1).any(1).tolist() == nonfinite
    for name in TOKENS:
        assert bool(torch.isfinite(getattr(bad, name)).all()), name
    assert bool(clean.frame_mask.all())
    _assert_frame_zeroed(bad, 4)
    _assert_others_equal(bad, clean, 4)
    # Point-only case: finite frame tokens, one NaN residual on a positive-weight row of frame 2.
    row = int(torch.nonzero(pm_c.weight[2] > 0)[0, 0])
    residual = pm_c.residual.clone()
    residual[2, row] = float("nan")
    point_only = fixture_tracker_inputs(fs, dataclasses.replace(base_scene(fs), pm=pm_c),
                                        point_pm=TermOutput(residual, pm_c.jacobian, pm_c.weight))
    for name in TOKENS:
        assert bool(torch.isfinite(getattr(point_only, name)).all()), name
    _assert_frame_zeroed(point_only, 2)
    _assert_others_equal(point_only, clean, 2)
