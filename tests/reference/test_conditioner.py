"""Conditioner end to end on synthetic windows: segmentation, refusals, byte-identity, ego world, start pose."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.reference import (
    HandInput,
    condition_window,
    no_world,
    start_pose_at_onset,
    world_from_ego,
)
from fastdo.reference import frames as FR

from .synth import noisy, pose, window

N = 100
BASE = pose([0.2, -0.3, 0.4], [0.05, -0.02, 0.6])
MOVED = pose([0.9, 0.1, -0.2], [-0.1, 0.05, 0.55])


def _story(rng, n=N):
    """Rest 0-39 (hand far), contact 40-59 (object carried), rest 60-99 at a NEW pose (hand far)."""
    P = np.empty((n, 4, 4))
    P[:40] = noisy(BASE, 40, rng)
    for i in range(40, 60):
        a = (i - 40) / 20
        P[i] = pose(Rotation.from_matrix(BASE[:3, :3]).as_rotvec() * (1 - a) + a * np.array([0.9, 0.1, -0.2]),
                    BASE[:3, 3] * (1 - a) + a * MOVED[:3, 3])
    P[60:] = noisy(MOVED, n - 60, rng)
    d = np.full(n, 0.30)
    d[34:40] = np.linspace(0.09, 0.02, 6)          # approach: NEAR
    d[40:60] = 0.005                               # contact
    d[60:66] = np.linspace(0.02, 0.09, 6)          # release: NEAR
    return P, d


def test_rest_segments_are_consolidated_separately(cfg) -> None:
    rng = np.random.default_rng(0)
    P, d = _story(rng)
    ref = condition_window(window(P, d), cfg)
    cons = [s for s in ref.segments if s.status == "consolidated"]
    assert len(cons) == 2
    assert cons[0].rows[-1] < 40 and cons[1].rows[0] > 59           # never across the touched span
    assert np.linalg.norm(cons[0].p - cons[1].p) > 0.1
    assert not ref.changed[34:66].any()                              # near / guard / touched rows unchanged
    assert np.array_equal(ref.poses[~ref.changed], P[~ref.changed])
    assert set(ref.row_state[40:60]) == {"touched"}
    assert set(ref.row_state[37:40]) <= {"guard", "near"}
    true_c = BASE[:3, :3] @ ref.c_star + BASE[:3, 3]
    assert np.linalg.norm(cons[0].p - true_c) < 0.002


def test_guard_rows_are_never_rest(cfg) -> None:
    """A hand jumps from FAR straight to CONTACT: the 3 rows before and after the touch stay unchanged."""
    rng = np.random.default_rng(1)
    P = noisy(BASE, N, rng)
    d = np.full(N, 0.30)
    d[50:55] = 0.005
    ref = condition_window(window(P, d), cfg)
    assert set(ref.row_state[47:50]) == {"guard"} and set(ref.row_state[55:58]) == {"guard"}
    assert not ref.changed[47:58].any()


def test_near_hand_blocks_rest(cfg) -> None:
    """A hand hovering at 50 mm (NEAR) is neither rest nor contact."""
    rng = np.random.default_rng(2)
    P = noisy(BASE, 40, rng)
    ref = condition_window(window(P, np.full(40, 0.05)), cfg)
    assert set(ref.row_state) == {"near"} and not ref.changed.any()


def test_a_drifting_untouched_object_is_not_consolidated(cfg) -> None:
    rng = np.random.default_rng(3)
    P = noisy(BASE, 30, rng, pos_sd=0.001)
    P[:, :3, 3] += np.linspace(0, 0.03, 30)[:, None] * [1, 0, 0]
    ref = condition_window(window(P, np.full(30, 0.3)), cfg)
    assert ref.segments[0].status == "refused" and not ref.changed.any()


def test_unknown_hands(cfg) -> None:
    """A never-seen hand the dataset does not declare absent makes every row unknown; declared absent -> rest. A
    bridged gap (FAR on both sides within 1 s) is certified."""
    rng = np.random.default_rng(4)
    P = noisy(BASE, 30, rng)
    far = np.full(30, 0.3)
    w = window(P, far, other_dist=np.full(30, np.nan), declared_absent_other=False)
    ref = condition_window(w, cfg)
    assert set(ref.row_state) == {"unknown_hand"} and not ref.changed.any()
    w2 = window(P, far)
    assert condition_window(w2, cfg).changed.all()
    gap = far.copy()
    gap[10:14] = np.nan
    ref3 = condition_window(window(P, gap), cfg)
    assert ref3.changed.all()
    gap2 = far.copy()
    gap2[10:25] = np.nan                                     # 1.5 s gap: rows in the middle cannot be certified
    ref4 = condition_window(window(P, gap2), cfg)
    assert "unknown_hand" in set(ref4.row_state[10:25])


def test_nonfinite_rows_are_never_filled(cfg) -> None:
    rng = np.random.default_rng(5)
    P = noisy(BASE, 30, rng)
    P[12] = np.nan
    ref = condition_window(window(P, np.full(30, 0.3)), cfg)
    assert np.isnan(ref.poses[12]).all() and ref.row_state[12] == "no_pose" and not ref.changed[12]
    assert ref.changed[:12].all() and ref.changed[13:].all()


def test_low_confidence_rows_are_replaced_but_do_not_vote(cfg) -> None:
    rng = np.random.default_rng(6)
    P = noisy(BASE, 20, rng)
    conf = np.ones(20)
    conf[[3, 7]] = 0.0
    P[3, :3, 3] += [0.012, 0, 0]
    ref = condition_window(window(P, np.full(20, 0.3), conf=conf), cfg)
    s = ref.segments[0]
    assert s.n_vote == 18 and 3 in s.replaced and 3 not in s.vote_rows


def test_action_cap_keeps_far_rows(cfg) -> None:
    rng = np.random.default_rng(7)
    P = noisy(BASE, 20, rng, pos_sd=0.001)
    P[5, :3, 3] += [0.025, 0, 0]
    ref = condition_window(window(P, np.full(20, 0.3)), cfg)
    assert ref.row_state[5] == "outlier_kept" and not ref.changed[5]
    assert np.array_equal(ref.poses[5], P[5])


def test_no_world_frame_is_byte_identical(cfg) -> None:
    rng = np.random.default_rng(8)
    P = noisy(BASE, 30, rng).astype(np.float32)
    ref = condition_window(window(P, np.full(30, 0.3), world=no_world(30, "test")), cfg)
    assert not ref.changed.any() and ref.poses.tobytes() == P.tobytes()
    assert set(ref.row_state) == {"no_world_frame"}


def _ego_file(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    s = np.exp(np.linspace(-0.1, 0.15, n))
    q = Rotation.from_rotvec(np.stack([np.linspace(0, 0.4, n), np.zeros(n), np.linspace(0, -0.3, n)], 1)).as_matrix()
    u = np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, :3, :3] = s[:, None, None] * q
    c2w[:, :3, 3] = u
    np.savez(path, cam_to_world=c2w, valid=np.ones(n, bool), reference_row=np.int64(0),
             frame_idx=np.arange(n, dtype=np.int64), summary=np.array(json.dumps({"ok": True})))
    return s, q, u


def test_static_world_object_under_a_moving_camera(tmp_path: Path, cfg) -> None:
    """The object is static in the ego WORLD while the camera moves (sigma != 1): the conditioner consolidates in the
    world and maps back; the output poses, put through the ego Sim(3) convention, give one world centre."""
    n = 30
    _ego_file(tmp_path / "ego.npz", n)
    ws = world_from_ego(tmp_path / "ego.npz", np.arange(n, dtype=np.int64))
    rng = np.random.default_rng(12)
    Ow = Rotation.from_rotvec([0.3, 0.2, -0.1]).as_matrix()
    probe = window(np.tile(BASE, (n, 1, 1)), np.full(n, 0.3))
    c = probe.group.c_star
    pw = np.array([0.1, 0.0, 0.7])
    O = np.einsum("tij,jk->tik", Rotation.from_rotvec(rng.normal(0, np.radians(0.5), (n, 3))).as_matrix(), Ow)
    p = pw + rng.normal(0, 0.002, (n, 3))
    P = FR.to_record(p, O, c, ws)
    ref = condition_window(dataclasses.replace(probe, poses=P, world=ws), cfg)
    assert ref.changed.all()
    pc, _ = FR.to_world(ref.poses, c, ws)
    assert np.allclose(pc, pc[0], atol=1e-9) and np.linalg.norm(pc[0] - pw) < 0.002
    cam_centres = ref.poses[:, :3, :3] @ c + ref.poses[:, :3, 3]
    assert np.ptp(cam_centres, axis=0).max() > 0.01             # the camera-frame poses still move


def test_start_pose_at_onset(cfg) -> None:
    rng = np.random.default_rng(13)
    P, d = _story(rng)
    ref = condition_window(window(P, d), cfg)
    sp, why = start_pose_at_onset(ref, 40)
    assert why == "ok" and sp is not None
    true_c = BASE[:3, :3] @ ref.c_star + BASE[:3, 3]
    assert np.linalg.norm(sp.centre_world - true_c) < 0.002
    assert np.allclose(sp.pose_record_at_onset[:3, :3] @ ref.c_star + sp.pose_record_at_onset[:3, 3], sp.centre_world)
    assert sp.segment_rows[1] < 34 and sp.n_vote >= cfg.min_vote and "centre_p95_mm" in sp.spread
    assert start_pose_at_onset(ref, 0)[1] == "no_segment_before_onset"
    assert start_pose_at_onset(ref, 40, max_gap_rows=2)[1] == "gap_too_long"
    d2 = d.copy()
    d2[30] = np.nan                                     # an uncertifiable hand row between rest and onset
    d2[31:34] = 0.05
    ref2 = condition_window(window(P, d2, other_dist=np.full(N, 0.3)), cfg)
    assert start_pose_at_onset(ref2, 40)[0] is None
    ref3 = condition_window(window(P, d, world=no_world(N, "t")), cfg)
    assert start_pose_at_onset(ref3, 40) == (None, "no_world_frame")


def test_ctrl_shares_rows_with_cond(cfg) -> None:
    rng = np.random.default_rng(14)
    P, d = _story(rng)
    a = condition_window(window(P, d), cfg)
    b = condition_window(window(P, d), cfg, estimator="ctrl")
    assert np.array_equal(a.changed, b.changed)
    assert all(np.array_equal(x.replaced, y.replaced) for x, y in zip(a.segments, b.segments, strict=True))


def test_save_refuses_reconstruction_artifacts(tmp_path: Path, cfg) -> None:
    rng = np.random.default_rng(15)
    ref = condition_window(window(noisy(BASE, 10, rng), np.full(10, 0.3)), cfg)
    with pytest.raises(ValueError):
        ref.save(tmp_path / "solve.npz")
    ref.save(tmp_path / "reference_conditioning.npz")
    assert (tmp_path / "reference_conditioning.json").is_file()


def test_hand_vertices_path_matches_precomputed_distances(cfg) -> None:
    """Distances computed from MANO-like points through the export routine agree with the precomputed path."""
    rng = np.random.default_rng(16)
    P = noisy(BASE, 12, rng)
    centre = P[:, :3, 3]
    verts = np.repeat((centre + [0.0, 0.0, -0.4])[:, None, :], 778, axis=1)
    w = window(P, np.full(12, np.nan))
    w = dataclasses.replace(w, hands={"right": HandInput(verts, np.ones(12, bool)),
                                      "left": HandInput(None, np.zeros(12, bool), True)})
    ref = condition_window(w, cfg)
    assert np.all(ref.distances["right"] > 0.3) and ref.changed.all()
