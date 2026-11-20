"""The masked geometry codec must be invisible to every consumer of the pointmaps."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fastdo.engine.geometry_io import (
    MASKED_FORMAT,
    is_masked,
    masked_object_points,
    pack_masked,
    unpack_pointmaps,
)
from fastdo.solver.api import TrajectoryObservations
from fastdo.solver.batch import ObjectPixels, object_pixels_of, valid_pixels, valid_pixels_of
from fastdo.solver.keyframe import gate_init_poses


def _clip(n: int = 5, h: int = 12, w: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pm = rng.standard_normal((n, h, w, 3)).astype(np.float32)
    pm[rng.random((n, h, w)) < 0.1] = np.nan                  # MoGe-invalid pixels, some under the mask
    masks = np.zeros((n, h, w), dtype=bool)
    for t in range(n):
        masks[t, 3 + t % 3: 8 + t % 3, 4:11] = True           # the object moves between frames
    return pm, masks


def test_every_pixel_a_consumer_reads_is_bit_identical() -> None:
    pm, masks = _clip()
    k = 2
    back = unpack_pointmaps(pack_masked(pm, masks, k), masks)
    assert back.shape == pm.shape and back.dtype == np.float32
    # the solver's own selection, mask & finite, is identical, values AND membership
    for t in range(pm.shape[0]):
        a = pm[t][masks[t] & np.isfinite(pm[t]).all(-1)]
        b = back[t][masks[t] & np.isfinite(back[t]).all(-1)]
        assert np.array_equal(a, b)
    # the keyframe is restored in full, OFF the mask too (track queries are sampled there)
    assert np.array_equal(back[k], pm[k], equal_nan=True)


def test_off_mask_pixels_read_as_nan_like_an_invalid_dense_pixel() -> None:
    pm, masks = _clip()
    back = unpack_pointmaps(pack_masked(pm, masks, 0), masks)
    off = ~masks.copy()
    off[0] = False                                            # the keyframe is dense
    assert np.isnan(back[off]).all(), "never zeros: a zero is a finite point at the camera centre"


def test_a_different_mask_is_refused_not_scattered() -> None:
    pm, masks = _clip()
    packed = pack_masked(pm, masks, 0)
    other = masks.copy()
    other[1, 0, 0] = True
    with pytest.raises(ValueError, match="different mask"):
        unpack_pointmaps(packed, other)


def test_a_dense_artifact_passes_through_unchanged() -> None:
    pm, masks = _clip()
    dense = {"pointmaps": pm, "frame_idx": np.arange(pm.shape[0])}
    assert not is_masked(dense)
    assert np.array_equal(unpack_pointmaps(dense, masks), pm, equal_nan=True)


def test_the_format_is_versioned_and_compact() -> None:
    pm, masks = _clip(n=4, h=48, w=64)
    packed = pack_masked(pm, masks, 1)
    assert is_masked(packed) and str(packed["pointmaps_format"]) == MASKED_FORMAT
    assert packed["object_points"].shape == (int(masks.sum()), 3)
    assert packed["object_offsets"][-1] == masks.sum()
    stored = sum(v.nbytes for v in packed.values())
    assert stored < 0.6 * pm.nbytes, "the whole point: a small fraction of the dense bytes"
    packed["pointmaps_format"] = np.array("masked_v999")
    with pytest.raises(ValueError, match="unknown"):
        unpack_pointmaps(packed, masks)


def test_shape_and_keyframe_are_validated() -> None:
    pm, masks = _clip()
    with pytest.raises(ValueError, match="out of range"):
        pack_masked(pm, masks, pm.shape[0])
    with pytest.raises(ValueError, match="must be"):
        pack_masked(pm, masks[:, :-1], 0)


def test_the_config_refuses_masked_storage_alongside_driftqc() -> None:
    from omegaconf import OmegaConf

    from fastdo.bench.bench_stages import geometry_pointmap_storage

    assert geometry_pointmap_storage(OmegaConf.create({})) == "dense"
    assert geometry_pointmap_storage(OmegaConf.create({"geometry_stage": {"pointmap_storage": "masked"}})) == "masked"
    with pytest.raises(ValueError, match="pointmap_storage"):
        geometry_pointmap_storage(OmegaConf.create({"geometry_stage": {"pointmap_storage": "zstd"}}))
    both = OmegaConf.create({"geometry_stage": {"pointmap_storage": "masked"},
                             "driftqc_stage": {"enabled": True, "constants_path": "c.json", "detector_path": "d.json"}})
    with pytest.raises(ValueError, match="driftqc"):
        geometry_pointmap_storage(both)


# the solve stage reads a masked artifact's object points directly (`ObjectPixels`), never a dense rebuild.
def _hard_clip(k: int) -> tuple[np.ndarray, np.ndarray]:
    pm, masks = _clip(n=6, seed=3)
    masks[1] = False                                          # an empty mask
    pm[4][masks[4]] = np.nan                                  # a frame whose object pixels are all invalid
    pm[k][~masks[k]] = 5.0                                    # finite off-mask keyframe pixels (kept dense)
    return pm, masks


def _observations(pm: np.ndarray, masks: np.ndarray, k: int) -> tuple[TrajectoryObservations, TrajectoryObservations]:
    """(dense, packed) observations of one clip, the packed one built the way `read_solve_inputs` builds it."""
    n, h, w = masks.shape
    common = {"intrinsics": torch.eye(3), "masks": torch.from_numpy(masks),
              "hand_masks": torch.zeros(n, h, w, dtype=torch.bool), "tracks": torch.zeros(n, 2, 2),
              "track_vis": torch.zeros(n, 2, dtype=torch.bool), "anchors": torch.zeros(2, 3)}
    artifact = pack_masked(pm, masks, k)
    dense = TrajectoryObservations(pointmaps=torch.from_numpy(unpack_pointmaps(artifact, masks)), **common)
    points, offsets = masked_object_points(artifact, masks)
    packed = TrajectoryObservations(pointmaps=torch.full((1, h, w, 3), np.nan).expand(n, h, w, 3),
                                    object_pixels=ObjectPixels(torch.from_numpy(points), torch.from_numpy(offsets)),
                                    **common)
    return dense, packed


def test_packed_valid_pixels_equal_the_dense_path_exactly() -> None:
    k = 2
    pm, masks = _hard_clip(k)
    dense, packed = _observations(pm, masks, k)
    assert packed.pointmaps.stride()[0] == 0, "the placeholder must not allocate (T,H,W,3)"
    ref_points, ref_counts = valid_pixels(dense.masks, dense.pointmaps)
    points, counts = valid_pixels_of(packed)
    assert points.dtype == ref_points.dtype and counts.dtype == ref_counts.dtype
    assert torch.equal(points, ref_points) and torch.equal(counts, ref_counts)          # values AND order
    assert counts[1] == 0 and counts[4] == 0 and int(counts.sum()) > 0
    dense_points, dense_counts = valid_pixels_of(dense)                               # no object_pixels: dense path
    assert torch.equal(dense_points, ref_points) and torch.equal(dense_counts, ref_counts)
    # the artifact's object points are, bit for bit, `object_pixels_of` the dense clip
    ref = object_pixels_of(torch.from_numpy(masks), torch.from_numpy(pm))
    assert torch.equal(ref.offsets, packed.object_pixels.offsets)
    assert ref.points.numpy().tobytes() == packed.object_pixels.points.numpy().tobytes()


def test_gate_reads_identical_object_pixels_from_either_path() -> None:
    k = 2
    pm, masks = _hard_clip(k)
    dense, packed = _observations(pm, masks, k)
    px = packed.object_pixels
    for t in range(masks.shape[0]):
        a = dense.pointmaps[t][dense.masks[t]]
        a = a[torch.isfinite(a).all(-1)]
        b = px.frame(t)
        b = b[torch.isfinite(b).all(-1)]
        assert torch.equal(a, b), t
    n = masks.shape[0]
    poses = torch.eye(4).repeat(n, 1, 1)
    for t in range(n):                                        # even frames on their observed median, odd ones far off
        finite = pm[t][masks[t] & np.isfinite(pm[t]).all(-1)]
        poses[t, :3, 3] = torch.from_numpy(np.median(finite, 0)) if finite.size and t % 2 == 0 else torch.tensor(
            [0.0, 0.0, 9.0])
    shape = 0.05 * torch.randn(64, 3, generator=torch.Generator().manual_seed(0))
    args = (poses, torch.full((n,), 10), shape, 1.0)
    kwargs = {"min_inliers": 6, "max_offset": 2.0, "min_points": 3}
    ref_poses, ref_accepted = gate_init_poses(*args, dense, k, poses[k], **kwargs)
    out_poses, accepted = gate_init_poses(*args, packed, k, poses[k], **kwargs)
    assert 0 < int(ref_accepted.sum()) < n, "the fixture must exercise both the accept and the re-seed branches"
    assert torch.equal(accepted, ref_accepted) and torch.equal(out_poses, ref_poses)


def test_masked_object_points_validate_like_unpack() -> None:
    pm, masks = _clip()
    packed = pack_masked(pm, masks, 0)
    points, offsets = masked_object_points(packed, masks)
    assert points.dtype == np.float32 and offsets.dtype == np.int64
    assert np.array_equal(offsets, packed["object_offsets"])
    other = masks.copy()
    other[1, 0, 0] = True
    with pytest.raises(ValueError, match="different mask"):
        masked_object_points(packed, other)
    with pytest.raises(ValueError, match="object_offsets"):
        masked_object_points({**packed, "object_offsets": packed["object_offsets"][::-1].copy()}, masks)
    with pytest.raises(ValueError, match="dense"):
        masked_object_points({"pointmaps": pm}, masks)
