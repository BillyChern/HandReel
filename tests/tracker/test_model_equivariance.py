"""Model CPU tests E1-E8: permutation, relabeling, time-reversal, padding, similarity, rescale and non-finite isolation.

float64, `SMALL_CONFIG`, `randomize_parameters(std=0.05, seed=0)`. Every test also shows that a perturbation which
should matter changes the outputs, so a constant or input-ignoring model cannot pass.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver.terms import pointmap_term, track_term
from fastdo.tracker.heads import TrackerOutput
from fastdo.tracker.model import AmortizedTracker
from fastdo.tracker.pack import TrackerInputs
from tests.tracker.synthetic_terms import CAUCHY_C, TermsFixture, make_terms_fixture
from tests.tracker.tracker_fixtures import (
    SMALL_CONFIG,
    Scene,
    base_scene,
    fixture_tracker_inputs,
    max_rel,
    nan_injection,
    pixel_rescaled_scene,
    random_tracker_inputs,
    randomize_parameters,
    scaled_scene,
    tracker_loss,
)

F64 = torch.float64
FIELDS = TrackerOutput._fields
NAMES = tuple(f.name for f in dataclasses.fields(TrackerInputs))
PER_FRAME = ("point_weights", "track_weights", "xi_hat", "precision_diag", "dual_g", "dual_f")
PER_CLIP = ("damping", "sigma_hat")


def _model() -> AmortizedTracker:
    model = AmortizedTracker(SMALL_CONFIG).double()
    randomize_parameters(model, std=0.05, seed=0)
    return model


def _run(model: AmortizedTracker, inputs: TrackerInputs) -> TrackerOutput:
    with torch.no_grad():
        return model(inputs)


def _assert_outputs_close(out: TrackerOutput, ref: TrackerOutput, rel: float, skip: tuple[str, ...] = ()) -> None:
    for name in FIELDS:
        if name not in skip:
            assert max_rel(getattr(out, name), getattr(ref, name)) <= rel, (name, max_rel(getattr(out, name),
                                                                                          getattr(ref, name)))


def _max_change(out: TrackerOutput, ref: TrackerOutput, names: tuple[str, ...] = FIELDS) -> float:
    return max(max_rel(getattr(out, n), getattr(ref, n)) for n in names)


def _gather_set(tokens: Tensor, mask: Tensor, perm: Tensor) -> tuple[Tensor, Tensor]:
    """Permute slot axis 2 of (B,W,S,d) tokens and (B,W,S) masks by a (B,W,S) index."""
    return torch.gather(tokens, 2, perm[..., None].expand_as(tokens)), torch.gather(mask, 2, perm)


def _perms(shape: tuple[int, int], n: int, seed: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(n, generator=gen) for _ in range(shape[0] * shape[1])]).view(*shape, n)


def test_track_permutation_equivariance() -> None:
    model, inputs = _model(), random_tracker_inputs(2, (12, 9), 16, 32, 64, seed=1)
    perm = _perms((2, 12), 16, seed=11)
    track, track_mask = _gather_set(inputs.track, inputs.track_mask, perm)
    ref, out = _run(model, inputs), _run(model, dataclasses.replace(inputs, track=track, track_mask=track_mask))
    assert max_rel(out.track_weights, torch.gather(ref.track_weights, 2, perm)) <= 1e-12
    assert max_rel(out.track_weights, ref.track_weights) > 1e-6            # the permutation is not the identity
    _assert_outputs_close(out, ref, 1e-12, skip=("track_weights",))
    j = int(torch.nonzero(inputs.track_mask[0, 4])[0, 0])
    changed = inputs.track.clone()
    changed[0, 4, j, 3] += 1.0
    moved = _run(model, dataclasses.replace(inputs, track=changed))
    assert max_rel(moved.damping[0], ref.damping[0]) > 1e-6


def test_point_permutation_equivariance() -> None:
    model, inputs = _model(), random_tracker_inputs(2, (12, 9), 16, 32, 64, seed=2)
    perm = _perms((2, 12), 32, seed=12)
    point, point_mask = _gather_set(inputs.point, inputs.point_mask, perm)
    permuted = dataclasses.replace(inputs, point=point, point_mask=point_mask)
    with torch.no_grad():
        assert torch.equal(model.backbone(permuted), model.backbone(inputs))
    ref, out = _run(model, inputs), _run(model, permuted)
    for name in ("point_weights", "dual_f"):
        assert max_rel(getattr(out, name), torch.gather(getattr(ref, name), 2, perm)) <= 1e-12, name
    for name in ("damping", "xi_hat", "sigma_hat", "precision_diag", "track_weights", "dual_g"):
        assert torch.equal(getattr(out, name), getattr(ref, name)), name
    i = int(torch.nonzero(inputs.point_mask[0, 3])[0, 0])
    changed = inputs.point.clone()
    changed[0, 3, i, 0] += 0.5
    moved = _run(model, dataclasses.replace(inputs, point=changed))
    assert max_rel(moved.point_weights[0, 3, i], ref.point_weights[0, 3, i]) > 1e-6


def test_cluster_relabeling_equivariance() -> None:
    model, inputs = _model(), random_tracker_inputs(2, (12, 9), 16, 32, 64, seed=3)
    perm = _perms((2, 1), 64, seed=13).expand(2, 12, 64)             # one relabeling per clip (template)
    cluster, cluster_mask = _gather_set(inputs.cluster, inputs.cluster_mask, perm)
    frame = inputs.frame.clone()
    frame[..., 44:108] = torch.gather(inputs.frame[..., 44:108], 2, perm)
    relabeled = dataclasses.replace(inputs, frame=frame, cluster=cluster, cluster_mask=cluster_mask)
    ref, out = _run(model, inputs), _run(model, relabeled)
    assert max_rel(out.dual_g, torch.gather(ref.dual_g, 2, perm)) <= 1e-12
    _assert_outputs_close(out, ref, 1e-12, skip=("dual_g",))
    c = int(torch.nonzero(inputs.cluster_mask[0, 0])[0, 0])
    changed = inputs.cluster.clone()
    changed[0, :, c, 1] += 0.1                                        # move one centroid of clip 0
    moved = _run(model, dataclasses.replace(inputs, cluster=changed))
    assert max_rel(moved.damping[0], ref.damping[0]) > 1e-6


def _reorder(inputs: TrackerInputs, index: Tensor) -> TrackerInputs:
    return TrackerInputs(*(getattr(inputs, n).index_select(1, index) for n in NAMES))


def test_time_reversal_equivariance() -> None:
    model, inputs = _model(), random_tracker_inputs(1, (20,), 16, 32, 64, seed=4)
    assert bool(inputs.frame_mask.all())
    reverse = torch.arange(19, -1, -1)
    ref, out = _run(model, inputs), _run(model, _reorder(inputs, reverse))
    for name in PER_FRAME:
        assert max_rel(getattr(out, name), getattr(ref, name).index_select(1, reverse)) <= 1e-12, name
    for name in PER_CLIP:
        assert max_rel(getattr(out, name), getattr(ref, name)) <= 1e-12, name
    swap = torch.arange(20)
    swap[2], swap[9] = 9, 2                                           # frames 2 and 9 are not mirror images
    swapped = _run(model, _reorder(inputs, swap))
    expected = TrackerOutput(*(getattr(ref, n).index_select(1, swap) if n in PER_FRAME else getattr(ref, n)
                               for n in FIELDS))
    assert _max_change(swapped, expected, PER_FRAME) > 1e-6


def _e5_inputs() -> tuple[TrackerInputs, TrackerInputs, TrackerInputs]:
    """(solo A with W=20, solo B with W=13, batched [A, B] at W=20 with garbage and True set masks in B's padding)."""
    clip_a = random_tracker_inputs(1, (20,), 16, 32, 64, seed=5)
    clip_b = random_tracker_inputs(1, (13,), 16, 32, 64, seed=6)
    garbage = random_tracker_inputs(1, (20,), 16, 32, 64, seed=7)
    padded = {}
    for name in NAMES:
        tail = getattr(garbage, name)[:, 13:]
        if name == "frame_mask":
            tail = torch.zeros_like(tail)
        elif name.endswith("_mask"):
            tail = torch.ones_like(tail)
        padded[name] = torch.cat([getattr(clip_b, name), tail], dim=1)
    batched = TrackerInputs(*(torch.cat([getattr(clip_a, n), padded[n]], dim=0) for n in NAMES))
    return clip_a, clip_b, batched


def test_padding_and_batch_independence() -> None:
    model = _model()
    clip_a, clip_b, batched = _e5_inputs()
    out, out_a, out_b = _run(model, batched), _run(model, clip_a), _run(model, clip_b)
    for name in PER_FRAME:
        assert max_rel(getattr(out, name)[0], getattr(out_a, name)[0]) <= 1e-12, name
        assert max_rel(getattr(out, name)[1, :13], getattr(out_b, name)[0]) <= 1e-12, name
    for name in PER_CLIP:
        assert max_rel(getattr(out, name)[0], getattr(out_a, name)[0]) <= 1e-12, name
        assert max_rel(getattr(out, name)[1], getattr(out_b, name)[0]) <= 1e-12, name
    for name in ("point_weights", "track_weights", "dual_g", "dual_f", "xi_hat"):
        pad = getattr(out, name)[1, 13:]
        assert torch.equal(pad, torch.zeros_like(pad)), name
    assert torch.equal(out.precision_diag[1, 13:], torch.ones(7, 6, dtype=F64))
    assert max_rel(out_a.damping, out_b.damping) > 1e-6


def _rotated_scene(fx: TermsFixture, angle: float) -> Scene:
    """Every pose rotated by `angle` rad about its body x axis (a real state change), plan held fixed."""
    poses = fx.poses @ se3_exp(torch.tensor([0.0, 0.0, 0.0, angle, 0.0, 0.0], dtype=F64))
    pm = pointmap_term(fx.X, fx.X_valid, fx.Y, poses, fx.log_scale, fx.eps, fx.rho_x, fx.rho_y, 0, plan=fx.plan)[0]
    tr = track_term(fx.anchors, fx.tracks, fx.track_vis, poses, fx.log_scale, fx.K, CAUCHY_C)
    return dataclasses.replace(base_scene(fx), poses=poses, pm=pm, tr_raw=tr)


def test_tracker_outputs_invariant_to_scene_similarity() -> None:
    fx, model = make_terms_fixture(knn=16), _model()
    ref = _run(model, fixture_tracker_inputs(fx))
    out = _run(model, fixture_tracker_inputs(fx, scaled_scene(fx, 2.0)))
    _assert_outputs_close(out, ref, 1e-10)
    rotated = _run(model, fixture_tracker_inputs(fx, _rotated_scene(fx, 0.05)))
    assert _max_change(rotated, ref) > 1e-6


def test_tracker_outputs_invariant_to_pixel_rescale() -> None:
    fx, model = make_terms_fixture(knn=16), _model()
    ref = _run(model, fixture_tracker_inputs(fx))
    out = _run(model, fixture_tracker_inputs(fx, pixel_rescaled_scene(fx, 2.0)))
    _assert_outputs_close(out, ref, 1e-10)
    assert max_rel(pixel_rescaled_scene(fx, 2.0).tr_raw.jacobian, 2.0 * fx.tr_raw.jacobian) <= 1e-12


def _mask_frame(inputs: TrackerInputs, b: int, k: int) -> TrackerInputs:
    fields = {n: getattr(inputs, n).clone() for n in NAMES}
    for value in fields.values():
        value[b, k] = False if value.dtype == torch.bool else 0.0
    return TrackerInputs(**fields)


def _assert_masked_frame_outputs(out: TrackerOutput, b: int, k: int) -> None:
    for name in ("point_weights", "track_weights", "dual_g", "dual_f", "xi_hat"):
        value = getattr(out, name)[b, k]
        assert torch.equal(value, torch.zeros_like(value)), name
    assert torch.equal(out.precision_diag[b, k], torch.ones(6, dtype=F64))


def test_nonfinite_frame_is_isolated() -> None:
    model = _model()
    # (a) Full chain: token test 14's injection through pointmap_term, the feature tokens and build_tracker_inputs.
    fs = make_terms_fixture(knn=16)
    pm_c, pm_n, nan_plan = nan_injection(fs)
    clean = fixture_tracker_inputs(fs, dataclasses.replace(base_scene(fs), pm=pm_c))
    bad = fixture_tracker_inputs(fs, dataclasses.replace(base_scene(fs), pm=pm_n), plan=nan_plan)
    out, ref, out_clean = _run(model, bad), _run(model, _mask_frame(clean, 0, 4)), _run(model, clean)
    assert not bool(bad.frame_mask[0, 4])
    for name in FIELDS:
        assert bool(torch.isfinite(getattr(out, name)).all()), name
    others = [0, 1, 2, 3, 5]
    for name in PER_FRAME:
        assert max_rel(getattr(out, name)[:, others], getattr(ref, name)[:, others]) <= 1e-12, name
    for name in PER_CLIP:
        assert max_rel(getattr(out, name), getattr(ref, name)) <= 1e-12, name
    _assert_masked_frame_outputs(out, 0, 4)
    assert max_rel(out_clean.damping, ref.damping) > 1e-6
    # (b) Model entry: NaN, then +inf in a seeded half, in every masked slot of the E5 batch.
    _, _, batched = _e5_inputs()
    frame_mask = batched.frame_mask.clone()
    frame_mask[0, 4] = False
    base = dataclasses.replace(batched, frame_mask=frame_mask)
    gen = torch.Generator().manual_seed(17)
    poisoned, zeroed = {}, {}
    for tok_name, mask_name in (("frame", None), ("track", "track_mask"), ("point", "point_mask"),
                                ("cluster", "cluster_mask")):
        tok = getattr(base, tok_name)
        keep = frame_mask if mask_name is None else getattr(base, mask_name) & frame_mask[..., None]
        slot = (~keep)[..., None].expand_as(tok)
        half = torch.rand(tok.shape, generator=gen) < 0.5
        nan = torch.where(slot, torch.full_like(tok, float("nan")), tok)
        poisoned[tok_name] = torch.where(slot & half, torch.full_like(tok, float("inf")), nan)
        zeroed[tok_name] = torch.where(slot, torch.zeros_like(tok), tok)
    assert bool(torch.isinf(poisoned["point"]).any()) and bool(torch.isnan(poisoned["point"]).any())
    out_p = _run(model, dataclasses.replace(base, **poisoned))
    out_z = _run(model, dataclasses.replace(base, **zeroed))
    for name in FIELDS:
        assert bool(torch.isfinite(getattr(out_p, name)).all()), name
        assert torch.equal(getattr(out_p, name), getattr(out_z, name)), name
    _assert_masked_frame_outputs(out_p, 0, 4)
    model.zero_grad(set_to_none=True)
    tracker_loss(model(dataclasses.replace(base, **poisoned))).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
