"""`geometry_stage.intrinsics_focal_scale`, the opt-in multiplier of MoGe-2's ESTIMATED focal length.

Off (null, absent, or exactly 1.0) must be the uncorrected path byte for byte; on, fx and fy are multiplied exactly and
the principal point is untouched; the value is recorded in `intrinsics.npz`; a `camera.json` calibration is never
scaled; `build_stage_fns` delivers the config value (the study's per-corpus constant) to the intrinsics worker, and
production ships the pooled constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("moge")

from fastdo.bench.bench_stages import build_stage_fns, intrinsics_focal_scale
from fastdo.engine import workers_geometry
from fastdo.engine.workers import make_intrinsics_worker
from fastdo.geometry.intrinsics import (
    FOCAL_SCALE_RANGE,
    INTRINSICS_FILE,
    camera_matrix,
    check_focal_scale,
    pixel_intrinsics,
    read_intrinsics,
    write_intrinsics,
)
from tests.engine.fakes import NORMALIZED_K, FakeMoGe, write_decode

ROOT = Path(__file__).resolve().parents[2]
NORMALIZED = np.zeros((3, 3, 3), np.float32)
NORMALIZED[:, 0, 0], NORMALIZED[:, 1, 1] = [0.78, 0.80, 0.79], [1.41, 1.38, 1.40]
NORMALIZED[:, 0, 2] = NORMALIZED[:, 1, 2] = 0.5
NORMALIZED[:, 2, 2] = 1.0


@pytest.mark.parametrize("off", [None, 1.0, 1])
def test_off_is_none(off: Any) -> None:
    assert check_focal_scale(off) is None


@pytest.mark.parametrize("value", [0.8, 1.0975, 1.25, np.float64(1.1), 1.0000001])
def test_on_values_are_floats(value: Any) -> None:
    assert check_focal_scale(value) == float(value)


@pytest.mark.parametrize("bad", [True, False, "1.1", 0.79, 1.26, float("nan"), float("inf"), 0.0, -1.1, [1.1]])
def test_bad_values_are_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="focal_scale"):
        check_focal_scale(bad)
    with pytest.raises(ValueError, match="focal_scale"):
        pixel_intrinsics(NORMALIZED, width=1280, height=720, focal_scale=bad)
    with pytest.raises(ValueError, match="focal_scale"):
        make_intrinsics_worker(device="cpu", focal_scale=bad)
    with pytest.raises(ValueError, match="geometry_stage.intrinsics_focal_scale"):
        intrinsics_focal_scale(OmegaConf.create({"geometry_stage": {"intrinsics_focal_scale": bad}}))


def test_range_brackets_the_measured_corrections() -> None:
    lo, hi = FOCAL_SCALE_RANGE
    assert lo <= 1.0 <= hi and lo <= np.exp(0.12) <= hi and lo <= np.exp(-0.12) <= hi


@pytest.mark.parametrize("off", [None, 1.0])
def test_pixel_intrinsics_identity_when_off(off: Any) -> None:
    base = pixel_intrinsics(NORMALIZED, width=1280, height=720)
    assert pixel_intrinsics(NORMALIZED, width=1280, height=720, focal_scale=off).tobytes() == base.tobytes()


@pytest.mark.parametrize("s", [0.9, 1.0975, 1.2])
def test_pixel_intrinsics_scales_fx_fy_exactly_and_keeps_the_principal_point(s: float) -> None:
    base = pixel_intrinsics(NORMALIZED, width=1280, height=720)
    K = pixel_intrinsics(NORMALIZED, width=1280, height=720, focal_scale=s)
    fx = float(np.median(NORMALIZED[:, 0, 0].astype(np.float64))) * 1280 * s
    fy = float(np.median(NORMALIZED[:, 1, 1].astype(np.float64))) * 720 * s
    assert K[0, 0] == np.float32(fx) and K[1, 1] == np.float32(fy)
    assert K[0, 2] == base[0, 2] and K[1, 2] == base[1, 2]
    np.testing.assert_allclose(K[0, 0] / base[0, 0], s, rtol=1e-6)
    off_diag = K.copy()
    off_diag[0, 0] = off_diag[1, 1] = off_diag[0, 2] = off_diag[1, 2] = 0
    np.testing.assert_array_equal(off_diag, np.diag([0, 0, 1]).astype(np.float32))


def test_write_intrinsics_records_the_scale_only_when_on(tmp_path: Path) -> None:
    K = camera_matrix(1100.0, 1100.0, 640.0, 360.0)
    write_intrinsics(tmp_path / "off", K, "moge2", focal_scale=1.0)
    write_intrinsics(tmp_path / "plain", K, "moge2")
    assert (tmp_path / "off" / INTRINSICS_FILE).read_bytes() == (tmp_path / "plain" / INTRINSICS_FILE).read_bytes()
    write_intrinsics(tmp_path / "on", K, "moge2", focal_scale=1.0975)
    with np.load(tmp_path / "on" / INTRINSICS_FILE, allow_pickle=False) as z:
        assert set(z.files) == {"K", "source", "focal_scale"}
        assert z["focal_scale"].dtype == np.float64 and float(z["focal_scale"]) == 1.0975
    K_read, source = read_intrinsics(tmp_path / "on")
    assert source == "moge2" and np.array_equal(K_read, K)


def _worker_K(clip: Path, **kwargs: Any) -> tuple[bytes, np.ndarray]:
    write_decode(clip, np.zeros((4, 6, 8, 3), np.uint8))
    make_intrinsics_worker(device="cpu", model=FakeMoGe(), **kwargs)("clip0", clip)
    return (clip / INTRINSICS_FILE).read_bytes(), read_intrinsics(clip)[0]


def test_worker_identity_when_off_and_exact_scaling_when_on(tmp_path: Path) -> None:
    base_bytes, base = _worker_K(tmp_path / "default")
    for off in (None, 1.0):
        assert _worker_K(tmp_path / f"off{off}", focal_scale=off)[0] == base_bytes
    _, K = _worker_K(tmp_path / "on", focal_scale=1.0975)
    expect = pixel_intrinsics(np.repeat(NORMALIZED_K[None], 3, axis=0), width=8, height=6, focal_scale=1.0975)
    np.testing.assert_array_equal(K, expect)
    assert K[0, 2] == base[0, 2] and K[1, 2] == base[1, 2]
    with np.load(tmp_path / "on" / INTRINSICS_FILE, allow_pickle=False) as z:
        assert float(z["focal_scale"]) == 1.0975


def test_a_calibrated_clip_is_never_scaled(tmp_path: Path) -> None:
    write_decode(tmp_path, np.zeros((4, 6, 8, 3), np.uint8))
    (tmp_path / "camera.json").write_text(json.dumps({"fx": 600.0, "fy": 600.0, "cx": 4.0, "cy": 3.0}))
    model = FakeMoGe()
    make_intrinsics_worker(device="cpu", model=model, focal_scale=1.2)("clip0", tmp_path)
    K, source = read_intrinsics(tmp_path)
    assert source == "given" and np.array_equal(K, camera_matrix(600.0, 600.0, 4.0, 3.0))
    with np.load(tmp_path / INTRINSICS_FILE, allow_pickle=False) as z:
        assert set(z.files) == {"K", "source"}


def test_production_is_the_pooled_constant_and_the_config_value_reaches_the_worker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production ships the POOLED constant; an absent key is still off."""
    cfg = OmegaConf.load(ROOT / "run" / "conf" / "config.yaml")
    pooled = json.loads((ROOT / "tests/fixtures/records/focal-scale-constants.json").read_text())["constants"]["pooled"]
    assert cfg.geometry_stage.intrinsics_focal_scale == pooled == 1.1040399273866555
    assert intrinsics_focal_scale(cfg) == pooled and intrinsics_focal_scale(OmegaConf.create({})) is None

    def spy(frames: Any, device: str = "cuda", model: Any = None, batch_size: int | None = None,
            **kwargs: Any) -> torch.Tensor:          # the bench builds on cuda; the test host has none
        return torch.from_numpy(np.repeat(NORMALIZED_K[None], frames.shape[0], axis=0))

    monkeypatch.setattr(workers_geometry.moge_model, "moge_intrinsics", spy)
    monkeypatch.setattr(workers_geometry.moge_model, "_load_model", lambda device: FakeMoGe())
    write_decode(tmp_path, np.zeros((4, 6, 8, 3), np.uint8))
    seen = []
    for value in (None, 1.0, 1.0975):
        OmegaConf.update(cfg, "geometry_stage.intrinsics_focal_scale", value)
        fn = build_stage_fns(cfg, ROOT, [])[0]["intrinsics"]
        assert fn.focal_scale == check_focal_scale(value)
        fn("clip0", tmp_path)
        seen.append(read_intrinsics(tmp_path)[0])
    assert np.array_equal(seen[0], seen[1])
    np.testing.assert_allclose(seen[2][0, 0] / seen[0][0, 0], 1.0975, rtol=1e-6)
    with np.load(tmp_path / INTRINSICS_FILE, allow_pickle=False) as z:
        assert float(z["focal_scale"]) == 1.0975
