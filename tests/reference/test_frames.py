"""World map and covariance transport: apply_ego_motion is the independent forward map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fastdo.geometry.lie import se3_exp
from fastdo.io.clip_record import ClipRecord, HandTrack
from fastdo.reference import frames as FR
from fastdo.reference.so3 import log
from fastdo.robot.world_frame import apply_ego_motion

T = 10


def _ego(path: Path, n: int, seed: int = 3) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(0, 0.1, n))
    q = Rotation.random(n, random_state=seed).as_matrix()
    u = rng.normal(0, 0.4, (n, 3))
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, :3, :3] = s[:, None, None] * q
    c2w[:, :3, 3] = u
    np.savez(path, cam_to_world=c2w, valid=np.ones(n, bool), reference_row=np.int64(0),
             frame_idx=np.arange(n, dtype=np.int64) * 3, summary=np.array(json.dumps({"ok": True})))
    return path, s, q, u


def _record(poses: np.ndarray) -> ClipRecord:
    n = len(poses)
    hand = HandTrack(params=np.zeros((n, 48), np.float32), betas=np.zeros(10, np.float32),
                     trans=np.zeros((n, 3), np.float32), joints=np.zeros((n, 21, 3), np.float32),
                     valid=np.ones(n, bool))
    return ClipRecord(clip_id="x", frame_idx=np.arange(n, dtype=np.int64) * 3, timestamps=np.arange(n) / 10.0,
                      intrinsics=np.eye(3, dtype=np.float32), cam_to_world=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
                      object_mesh_path="m.obj", object_pose=poses.astype(np.float32), object_scale=1.0,
                      gravity_cam=np.array([0, -1, 0], np.float32), confidence=np.ones(n, np.float32), gate_score=0.0,
                      hand_left=None, hand_right=hand)


def _poses(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, :3] = Rotation.random(T, random_state=seed).as_matrix()
    P[:, :3, 3] = rng.normal(0, 0.1, (T, 3)) + [0, 0, 0.6]
    return P


def test_world_centre_matches_apply_ego_motion(tmp_path: Path) -> None:
    """The module's world centre and rotation equal apply_ego_motion's output posed through its cam_to_world, with
    sigma != 1 and c* != 0 (major 7 / minor 9)."""
    P = _poses().astype(np.float32).astype(np.float64)
    c_star = np.array([0.03, -0.02, 0.01])
    path, *_ = _ego(tmp_path / "ego.npz", T)
    ws = FR.world_from_ego(path, np.arange(T, dtype=np.int64) * 3)
    assert ws.kind == "ego_motion" and not np.allclose(ws.sigma, 1.0)
    out, _, _ = apply_ego_motion(_record(P), None, path)
    c2w = out.cam_to_world.astype(np.float64)
    obj = out.object_pose.astype(np.float64)
    want_p = np.einsum("tij,tj->ti", c2w[:, :3, :3], obj[:, :3, :3] @ c_star + obj[:, :3, 3]) + c2w[:, :3, 3]
    want_O = c2w[:, :3, :3] @ obj[:, :3, :3]
    p, O = FR.to_world(P, c_star, ws)
    assert np.allclose(p, want_p, atol=2e-6)          # float32 record
    assert np.allclose(O, want_O, atol=2e-6)
    back = FR.to_record(p, O, c_star, ws)
    assert np.abs(back - P).max() < 1e-12


def test_transport_matches_finite_differences(tmp_path: Path) -> None:
    """Sigma^w = M Sigma^cam M^T with M the finite-difference Jacobian of [p; Log(O' O^T)] w.r.t. the camera-frame
    left perturbation xi (lie.se3_exp), sigma != 1, c* != 0."""
    P = _poses(1)
    c_star = np.array([0.02, 0.05, -0.03])
    path, *_ = _ego(tmp_path / "ego.npz", T, seed=7)
    ws = FR.world_from_ego(path, np.arange(T, dtype=np.int64) * 3)
    rng = np.random.default_rng(2)
    A = rng.normal(size=(T, 6, 6)) * 1e-3
    cov = A @ np.transpose(A, (0, 2, 1))
    got = FR.transport_covariance(cov, P, c_star, ws)
    _, O0 = FR.to_world(P, c_star, ws)
    h = 1e-6
    for t in range(T):
        M = np.zeros((6, 6))
        sub = FR.WorldSource(ws.kind, ws.Q[[t]], ws.u[[t]], ws.sigma[[t]])
        for j in range(6):
            xi = np.zeros(6)
            xi[j] = h
            Pp = se3_exp(torch.from_numpy(xi)).numpy() @ P[t]
            Pm = se3_exp(torch.from_numpy(-xi)).numpy() @ P[t]
            pp, Op = FR.to_world(Pp[None], c_star, sub)
            pm, Om = FR.to_world(Pm[None], c_star, sub)
            M[:3, j] = (pp[0] - pm[0]) / (2 * h)
            M[3:, j] = (log(Op[0] @ O0[t].T) - log(Om[0] @ O0[t].T)) / (2 * h)
        want = M @ cov[t] @ M.T
        assert np.allclose(got[t], want, rtol=1e-4, atol=1e-12)


def test_round_trip_keeps_nonfinite_rows_and_is_exact() -> None:
    P = _poses(4)
    P[3] = np.nan
    ws = FR.declared_static(T)
    out = FR.round_trip(P, np.array([0.01, 0.0, 0.02]), ws)
    assert np.isnan(out[3]).all()
    ok = np.arange(T) != 3
    assert np.abs(out[ok] - P[ok]).max() < 1e-12


def test_ego_refusals_match_apply_ego_motion(tmp_path: Path) -> None:
    path, *_ = _ego(tmp_path / "ego.npz", T)
    assert FR.world_from_ego(path, np.arange(T, dtype=np.int64) * 3 + 3).kind == "none"
    assert FR.world_from_ego(None, np.arange(T)).reason == "no_ego_motion_file"
    with np.load(path) as z:
        d = dict(z)
    d["summary"] = np.array(json.dumps({"ok": False}))
    np.savez(tmp_path / "bad.npz", **d)
    assert FR.world_from_ego(tmp_path / "bad.npz", np.arange(T, dtype=np.int64) * 3).reason == "ego_self_check_failed"
    with pytest.raises(ValueError):
        apply_ego_motion(_record(_poses()), None, tmp_path / "bad.npz")
