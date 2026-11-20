"""Tests for `run/pipeline/whisking_headtohead.py` (faked front end, plus one real timing run)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import trimesh
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import Tensor

from fastdo.solver import TrajectorySolution
from run.pipeline import whisking_headtohead as h2h

_REPO = Path(__file__).resolve().parents[2]
_H, _W, _N_FULL, _STRIDE = 32, 48, 7, 3
_IDX = [0, 3, 6]
_K = torch.tensor([[60.0, 0.0, 24.0], [0.0, 60.0, 16.0], [0.0, 0.0, 1.0]])


def _cfg(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def _layout() -> tuple[Tensor, float, Tensor]:
    poses = torch.eye(4).repeat(_N_FULL, 1, 1)
    poses[:, 2, 3] = 0.5
    poses[:, 0, 3] = torch.linspace(-0.02, 0.02, _N_FULL)
    return poses, 0.1, _K.clone()


def _write_demo(tmp_path: Path, with_caches: bool) -> list[str]:
    demo, adapters = tmp_path / "demo", tmp_path / "adapters"
    demo.mkdir()
    adapters.mkdir()
    trimesh.creation.box(extents=(1.0, 0.6, 0.4)).export(demo / "mesh.obj")
    t = _N_FULL
    joints = np.zeros((t, 21, 3), np.float32)
    joints[..., 2] = 0.5
    tri = np.array([[-0.2, -0.2, 0.4], [-0.15, -0.2, 0.4], [-0.2, -0.15, 0.4]], np.float32)
    np.savez(demo / "hands.npz", right_rot=np.zeros((t, 3), np.float32), right_hand_pose=np.zeros((t, 45), np.float32),
             right_trans=np.zeros((t, 3), np.float32), right_joints=joints, right_betas=np.zeros((t, 10), np.float32),
             right_valid=np.ones(t, bool), right_vertices=np.repeat(tri[None], t, 0),
             right_faces=np.array([[0, 1, 2]]))
    if with_caches:
        masks = np.zeros((3, _H, _W), bool)
        masks[:, 12:20, 18:30] = True
        np.savez(adapters / h2h._MASKS_CACHE, masks=masks, frame_idx=np.array(_IDX))
        uv = np.array([[20.0, 14.0], [26.0, 17.0], [22.0, 18.0], [28.0, 13.0]], np.float32)
        np.savez(adapters / h2h._TRACKS_CACHE, tracks=np.repeat(uv[None], 3, 0), vis=np.ones((3, 4), bool),
                 query_uv=uv, frame_idx=np.array(_IDX), keyframe_index=np.int64(1))
    return [f"paths.root={_REPO}", f"whisking.demo_root={demo}", "whisking.video=whisking.mp4",
            "whisking.mesh_path=mesh.obj", "whisking.hand_npz=hands.npz", "whisking.layout_json=layout.json",
            f"whisking.adapters_dir={adapters}", f"whisking.out_dir={tmp_path / 'out'}",
            f"whisking.notes_path={tmp_path / 'notes.md'}", f"whisking.stride={_STRIDE}",
            f"whisking.n_frames_full={_N_FULL}", "whisking.keyframe_frame_idx=4", "whisking.n_track_queries=4"]


def _fake_pointmaps(frames: Tensor, device: str = "cuda", model: Any = None,
                    intrinsics: Tensor | None = None) -> Tensor:
    vv, uu = torch.meshgrid(torch.arange(_H, dtype=torch.float32), torch.arange(_W, dtype=torch.float32),
                            indexing="ij")
    z = torch.full((_H, _W), 0.5)
    pm = torch.stack([(uu - _K[0, 2]) * z / _K[0, 0], (vv - _K[1, 2]) * z / _K[1, 1], z], dim=-1)
    return pm[None].expand(frames.shape[0], -1, -1, -1).clone()


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, calls: dict[str, int]) -> None:
    def _spy_layout(path: Path, image_width: int = 1280, image_height: int = 720) -> tuple[Tensor, float, Tensor]:
        calls["read_baseline_layout"] = calls.get("read_baseline_layout", 0) + 1
        return _layout()

    def _fake_decode(path: Path, stride: int = 1, max_frames: int | None = None) -> tuple[Tensor, list[int]]:
        return torch.zeros(len(_IDX), _H, _W, 3, dtype=torch.uint8), list(_IDX)

    def _fake_pnp(anchors: Tensor, valid: Tensor, tracks: Tensor, vis: Tensor, K: Tensor, scale: float,
                  min_inliers: int = 6) -> tuple[Tensor, Tensor]:
        return _layout()[0][_IDX], torch.full((len(_IDX),), 6)

    def _fake_solve(mesh: Any, obs: Any, init_poses: Tensor, init_scale: float, cfg: Any) -> TrajectorySolution:
        t = init_poses.shape[0]
        return TrajectorySolution(poses=init_poses.clone(), scale=init_scale, covariance=torch.zeros(t, 6, 6),
                                  per_frame_cost=torch.zeros(t), confidence=torch.ones(t), iterations=1, seconds=0.0)

    def _fake_keyframe_layout(path: Path, frame_idx: int | None = None) -> tuple[Tensor, float]:
        calls["read_keyframe_layout"] = calls.get("read_keyframe_layout", 0) + 1
        calls["keyframe_layout_frame"] = int(frame_idx) if frame_idx is not None else -1
        pose = torch.eye(4)
        pose[2, 3], pose[0, 3] = 0.5, 0.004                     # SAM 3D's own keyframe layout, 4 mm off
        return pose, 0.08                                        # and its pre-optimization scale

    monkeypatch.setattr(h2h, "read_baseline_layout", _spy_layout)
    monkeypatch.setattr(h2h, "read_keyframe_layout", _fake_keyframe_layout)
    monkeypatch.setattr(h2h, "decode_frames", _fake_decode)
    monkeypatch.setattr(h2h, "run_moge", _fake_pointmaps)
    monkeypatch.setattr(h2h, "pnp_initialize", _fake_pnp)
    monkeypatch.setattr(h2h, "solve_trajectory", _fake_solve)


def test_whisking_headtohead_reuses_layout_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {}
    _patch_pipeline(monkeypatch, calls)
    cfg = _cfg(_write_demo(tmp_path, with_caches=True))

    report = h2h.run_headtohead(cfg, _REPO)

    assert calls["read_baseline_layout"] == 1
    assert set(h2h.REPORT_KEYS) <= set(report)
    # ours == layout except the keyframe, which starts 4 mm off at the shape stage's own layout
    assert 0.0 < report["mean_translation_error_m"] < 0.004
    on_disk = json.loads((tmp_path / "out" / "headtohead_report.json").read_text())
    assert set(on_disk) == set(h2h.REPORT_KEYS)


def test_whisking_headtohead_starts_from_the_shape_stage_layout_not_the_baseline_answer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {}
    seen: dict[str, Any] = {}
    _patch_pipeline(monkeypatch, calls)

    def _spy_gate(poses: Tensor, n_inliers: Tensor, points: Tensor, scale: float, obs: Any, keyframe_index: int,
                  keyframe_pose: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        seen["gate_scale"], seen["gate_keyframe_pose"] = scale, keyframe_pose.clone()
        return poses, torch.ones(poses.shape[0], dtype=torch.bool)

    def _spy_solve(mesh: Any, obs: Any, init_poses: Tensor, init_scale: float, cfg: Any) -> TrajectorySolution:
        seen["init_scale"] = init_scale
        t = init_poses.shape[0]
        return TrajectorySolution(poses=init_poses.clone(), scale=init_scale, covariance=torch.zeros(t, 6, 6),
                                  per_frame_cost=torch.zeros(t), confidence=torch.ones(t), iterations=1, seconds=0.0)

    monkeypatch.setattr(h2h, "gate_init_poses", _spy_gate)
    monkeypatch.setattr(h2h, "solve_trajectory", _spy_solve)
    h2h.run_headtohead(_cfg(_write_demo(tmp_path, with_caches=True)), _REPO)

    assert calls["read_keyframe_layout"] == 1 and calls["keyframe_layout_frame"] == 4
    assert seen["init_scale"] == pytest.approx(0.08) and seen["gate_scale"] == pytest.approx(0.08)  # not 0.1
    assert float(seen["gate_keyframe_pose"][0, 3]) == pytest.approx(0.004)                          # not layout[3]
    details = json.loads((tmp_path / "out" / "headtohead_details.json").read_text())
    assert details["init_scale"] == pytest.approx(0.08) and details["baseline_mesh_scale"] == pytest.approx(0.1)
    notes = (tmp_path / "notes.md").read_text()
    assert "not evidence of better tracking" in notes and "0.0800" in notes


def test_whisking_headtohead_reuses_cached_adapters_and_rewrites_notes_in_place(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cached SAM 2 / CoTracker outputs must be reused, not recomputed")

    _patch_pipeline(monkeypatch, {})
    monkeypatch.setattr(h2h, "propagate_masks_sam2", _must_not_run)
    monkeypatch.setattr(h2h, "track_points_cotracker", _must_not_run)
    overrides = _write_demo(tmp_path, with_caches=True)
    (tmp_path / "notes.md").write_text("# Notes\n\n## Earlier section\n\nkeep me\n")

    h2h.run_headtohead(_cfg(overrides), _REPO)
    h2h.run_headtohead(_cfg(overrides), _REPO)

    notes = (tmp_path / "notes.md").read_text()
    assert notes.count(h2h._NOTES_HEADING) == 1
    assert "keep me" in notes and "`solver_seconds`" in notes


def test_whisking_headtohead_recomputes_without_caches_from_the_keyframe_prompt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def _spy_sam2(frames: Tensor, keyframe_index: int, keyframe_mask: Tensor, device: str = "cuda") -> Tensor:
        seen["sam2"] = (keyframe_index, keyframe_mask.clone())
        return keyframe_mask[None].expand(frames.shape[0], -1, -1).clone()

    def _spy_cotracker(frames: Tensor, keyframe_index: int, query_uv: Tensor, device: str = "cuda",
                       model: Any = None) -> tuple[Tensor, Tensor]:
        seen["cotracker"] = keyframe_index
        return query_uv[None].expand(frames.shape[0], -1, -1).clone(), torch.ones(frames.shape[0], len(query_uv),
                                                                                  dtype=torch.bool)

    _patch_pipeline(monkeypatch, {})
    monkeypatch.setattr(h2h, "propagate_masks_sam2", _spy_sam2)
    monkeypatch.setattr(h2h, "track_points_cotracker", _spy_cotracker)
    report = h2h.run_headtohead(_cfg(_write_demo(tmp_path, with_caches=False)), _REPO)

    keyframe_index, prompt = seen["sam2"]
    assert keyframe_index == 1 and seen["cotracker"] == 1           # frame 3 is nearest the keyframe (4)
    assert 0 < int(prompt.sum()) < _H * _W                           # mesh silhouette at the keyframe pose
    # Fake SAM 2 repeats the keyframe prompt; the layout drifts 2 cm, so IoU is high but below 1 off-keyframe.
    assert 0.5 < report["reprojection_iou_baseline"] < 1.0


def _cached_weights_missing() -> list[str]:
    """HF checkpoints the real run loads that are not in the local cache (no download is attempted)."""
    from huggingface_hub import try_to_load_from_cache

    from fastdo.geometry.moge_model import MODEL_ID

    wanted = [(MODEL_ID, "model.pt"), ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt")]
    return [f"{repo}/{name}" for repo, name in wanted if not isinstance(try_to_load_from_cache(repo, name), str)]


def _real_run_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "needs the GPU"
    for pkg in ("moge", "sam2", "huggingface_hub"):
        if importlib.util.find_spec(pkg) is None:
            return False, f"needs the frontend extra ({pkg})"
    if not (_REPO / "third_party/do-as-i-do/reconstruction/whisking/whisking.mp4").is_file():
        return False, "needs the shipped whisking demo"
    missing = _cached_weights_missing()
    if missing:
        return False, f"needs cached weights: {', '.join(missing)}"
    return True, ""


@pytest.mark.timing
@pytest.mark.skipif(not _real_run_available()[0], reason=_real_run_available()[1])
def test_whisking_headtohead_runs_and_reports(tmp_path: Path) -> None:
    outputs = Path(os.environ.get("FASTDO_OUTPUTS", "/opt/fastdo/handreel/outputs"))
    cfg = _cfg([f"paths.root={_REPO}", f"whisking.adapters_dir={outputs / 'whisking/adapters'}",
                f"whisking.out_dir={tmp_path}", f"whisking.notes_path={tmp_path / 'notes.md'}"])

    report = h2h.run_headtohead(cfg, _REPO)

    saved = json.loads((tmp_path / "headtohead_report.json").read_text())
    assert set(saved) == set(h2h.REPORT_KEYS)
    assert all(np.isfinite(saved[k]) for k in h2h.REPORT_KEYS)
    assert report["solver_seconds"] < 3.0
