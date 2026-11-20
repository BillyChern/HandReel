"""Tests for batch-invariant HaWoR hands on CPU: fixed-shape backbone, heads and MANO calls.

`BatchSensitiveHawor` makes the float-level kernel effect explicit and exact: every module adds `EPS * (rows in the
call)` to its output, and is otherwise computed row by row with elementwise ops and per-row reductions, so a row's
output depends on the SIZE of the call it ran in and on nothing else. With padding on, a clip's outputs must then be
byte-identical whatever its batch mates; with padding off (the default), the same inputs must show the dependence (the
control that shows the effect). The MANO fake does the same per `run_mano` call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fastdo.hands.hawor_core import FeatureBank, crop_hands, gather_rows, run_heads
from fastdo.hands.hawor_plan import Slot, plan_clip
from tests.hands.test_hawor_core import OPS, SIDES, FakeHawor, _clips
from tests.hands.test_hawor_driver_cpu import DRIVER_FILE, REPO, _arrays, _FakeApi, _request, _run

EPS = 1e-3


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fastdo_test_hawor_driver_191", DRIVER_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shared(driver: ModuleType) -> dict[str, ModuleType]:
    return driver.load_shared_modules(REPO)


class _Backbone(nn.Module):
    """`(N,3,256,192)` -> `(N,1280,16,12)`: a 16x16 average pool, channel-tiled and scaled, plus `EPS * N`."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("w", torch.linspace(0.5, 1.5, 1280))

    def forward(self, x: Tensor) -> Tensor:
        pooled = F.avg_pool2d(x, 16)                                        # (N,3,16,12)
        tiled = pooled.repeat(1, 427, 1, 1)[:, :1280]
        return tiled * self.w[None, :, None, None] + EPS * x.shape[0]


class _Temporal(nn.Module):
    """`(M,16,1283)` -> `(M,16,1280)` elementwise, plus `EPS * M` (M = windows x 16 x 12 sequences)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("w", torch.linspace(0.8, 1.2, 1280))

    def forward(self, x: Tensor) -> Tensor:
        return x[..., :1280] * self.w + 0.1 * x[..., 1280:1281] + EPS * x.shape[0]


class _ManoHead(nn.Module):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pooled = x.mean(dim=(2, 3))
        cam = torch.cat([0.9 + 0.1 * torch.sigmoid(pooled[:, :1]), 0.1 * pooled[:, 1:3]], 1)
        return 0.1 * pooled[:, :96], pooled[:, 96:106], cam


class _Motion(nn.Module):
    """`(b,16,99)` -> `(b,16,96)`: a small perturbation of the identity 6D rotation of every joint (well conditioned)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(16))

    def forward(self, x: Tensor) -> Tensor:
        return self.identity + 0.1 * torch.tanh(x[..., :96]) + 0.01 * x[..., 96:99].sum(-1, keepdim=True)


class BatchSensitiveHawor(FakeHawor):
    """`FakeHawor`'s CLIFF `bbox_est` and `get_trans`, with call-size-dependent, otherwise row-wise modules."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone, self.st_module = _Backbone(), _Temporal()
        self.mano_head, self.motion_module = _ManoHead(), _Motion()


def _call_sizes(module: nn.Module) -> list[int]:
    sizes: list[int] = []
    module.register_forward_hook(lambda _m, inp, _out: sizes.append(int(inp[0].shape[0])))
    return sizes


def _crops(n: int, seed: int) -> Tensor:
    return torch.randn(n, 3, 256, 256, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- backbone


def _bank_features(model: nn.Module, parts: list[Tensor], pad: bool, batch: int = 4) -> Tensor:
    bank = FeatureBank(model, batch=batch, bf16=False, pad_to_batch=pad)       # type: ignore[arg-type]
    for part in parts:
        bank.push(part)
    bank.flush()
    return bank.features()


@pytest.mark.parametrize("pad", [False, True])
def test_feature_bank_padding_makes_a_crop_independent_of_its_fifo_mates(pad: bool) -> None:
    model = BatchSensitiveHawor()
    mine, mates = _crops(5, 0), _crops(6, 1)

    alone = _bank_features(model, [mine], pad)                        # calls 4, then a remainder of 1
    after_mates = _bank_features(model, [mates, mine], pad)[6:]       # calls 4, 4, then a remainder of 3

    if pad:
        assert torch.equal(alone, after_mates)
    else:                                                             # the control: the effect is there without it
        assert torch.equal(alone[:2], after_mates[:2])                 # both ran in calls of 4
        assert float((alone[2:4] - after_mates[2:4]).abs().max()) > 0.5 * EPS   # calls of 4 vs a remainder of 3


def test_feature_bank_pad_runs_one_full_batch_and_keeps_the_real_rows() -> None:
    model = BatchSensitiveHawor()
    sizes = _call_sizes(model.backbone)
    crops = _crops(7, 2)

    padded = _bank_features(model, [crops[:2], crops[2:]], pad=True)

    assert sizes == [4, 4] and padded.shape == (7, 1280, 16, 12)
    full = model.backbone(torch.cat([crops[4:], torch.zeros(1, 3, 256, 256)])[:, :, :, 32:-32])
    assert torch.equal(padded[4:], full[:3])


def test_feature_bank_default_flush_runs_exactly_the_remainder() -> None:
    model = BatchSensitiveHawor()
    sizes = _call_sizes(model.backbone)
    crops = _crops(7, 3)

    default = FeatureBank(model, batch=4, bf16=False)                 # type: ignore[arg-type]
    default.push(crops)
    default.flush()
    explicit = _bank_features(model, [crops], pad=False)

    assert sizes == [4, 3, 4, 3]
    assert torch.equal(default.features(), explicit)
    assert FeatureBank(model, batch=4, bf16=False, pad_to_batch=True).flush() == 0.0   # type: ignore[arg-type]


# ---------------------------------------------------------------- heads


def _group_rows(model: nn.Module, group: list[int], pad: bool, windows_per_batch: int = 4,
                backbone_batch: int = 7) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    clips = _clips()
    bank = FeatureBank(model, batch=backbone_batch, bf16=False, pad_to_batch=pad)   # type: ignore[arg-type]
    slots: list[Slot] = []
    for local, index in enumerate(group):
        clip = clips[index]
        items, clip_slots = plan_clip(local, clip.frame_idx, clip.boxes, clip.geom, bank_offset=bank.n_items)
        bank.push(crop_hands(torch.from_numpy(clip.frames), items))
        slots += clip_slots
    bank.flush()
    out = run_heads(model, OPS, bank.features(), slots, [clips[i].geom for i in group],  # type: ignore[arg-type]
                    windows_per_batch=windows_per_batch, pad_last=pad)
    return {(index, side): gather_rows(out, slots, local, side, len(clips[index].frame_idx))
            for local, index in enumerate(group) for side in SIDES}


def _max_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    return max(float(np.nanmax(np.abs(a[k] - b[k]))) for k in ("rot", "hand_pose", "trans", "betas"))


@pytest.mark.parametrize("pad", [False, True])
def test_clip_hands_do_not_depend_on_group_mates_with_padding(pad: bool) -> None:
    model = BatchSensitiveHawor()

    alone = _group_rows(model, [0], pad) | _group_rows(model, [1], pad)
    together = _group_rows(model, [0, 1], pad)
    reversed_order = _group_rows(model, [1, 0], pad)

    for key in alone:
        for other in (together, reversed_order):
            if pad:
                for name in ("rot", "hand_pose", "trans", "betas", "valid"):   # slot_index is a group position
                    np.testing.assert_array_equal(other[key][name], alone[key][name], err_msg=f"{key} {name}")
            else:
                np.testing.assert_array_equal(other[key]["valid"], alone[key]["valid"])
    if not pad:                                                        # the control: some clip moved with its mates
        assert max(_max_diff(alone[key], other[key]) for key in alone for other in (together, reversed_order)) > 0.0


def test_run_heads_pad_last_runs_full_shape_passes_without_nan() -> None:
    model = BatchSensitiveHawor()
    sizes = _call_sizes(model.st_module)
    clips = _clips()
    bank = FeatureBank(model, batch=64, bf16=False)                   # type: ignore[arg-type]
    items, slots = plan_clip(0, clips[0].frame_idx, clips[0].boxes, clips[0].geom, bank_offset=0)
    bank.push(crop_hands(torch.from_numpy(clips[0].frames), items))
    bank.flush()
    per, n_windows, seq = 4, len(slots) // 16, 16 * 12                # seq: temporal sequences per window
    assert n_windows % per != 0                                       # so the last pass is short
    geoms: Any = [clips[0].geom]

    padded = run_heads(model, OPS, bank.features(), slots, geoms, windows_per_batch=per, pad_last=True)
    plain = run_heads(model, OPS, bank.features(), slots, geoms, windows_per_batch=per)

    n_passes = -(-n_windows // per)
    assert sizes[:n_passes] == [per * seq] * n_passes
    assert sizes[n_passes:] == [per * seq] * (n_passes - 1) + [(n_windows % per) * seq]
    full = slice(0, (n_passes - 1) * per * 16)                        # the full passes are the same calls
    for name in ("rotmat", "aa", "betas", "trans"):
        a, b = getattr(padded, name), getattr(plain, name)
        assert a.shape == b.shape and a.shape[0] == len(slots) and bool(torch.isfinite(a).all()), name
        assert torch.equal(a[full], b[full]), name
        assert not torch.equal(a[full.stop:], b[full.stop:]), name    # the short pass moved (the fake's EPS * M)


# ---------------------------------------------------------------- driver and MANO


class _SensitiveApi(_FakeApi):
    """The CPU driver test's fake API with a call-size-dependent network and MANO, recording MANO call sizes."""

    def __init__(self, clips: list[Any]) -> None:
        super().__init__(clips)
        self.mano_rows: list[int] = []

    def load_models(self, hawor_root: Path, shared: Any) -> Any:
        models = super().load_models(hawor_root, shared)
        models.hawor = BatchSensitiveHawor()
        return models

    def run_mano(self, models: Any, side: str, rot: np.ndarray, hand_pose: np.ndarray, betas: np.ndarray,
                 trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.mano_rows.append(int(rot.shape[0]))
        vertices, joints = super().run_mano(models, side, rot, hand_pose, betas, trans)
        return vertices + np.float32(EPS * rot.shape[0]), joints + np.float32(EPS * rot.shape[0])


def _driver_outputs(driver: ModuleType, shared: Any, tmp_path: Path, name: str, indices: list[int],
                    **options: Any) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any], _SensitiveApi]:
    clips = _clips()
    run_dir = tmp_path / name
    run_dir.mkdir()
    api = _SensitiveApi(clips)
    request = _request(run_dir, [clips[i] for i in indices], **options)
    code, body = _run(driver, shared, run_dir, api, request)
    assert code == 0, body
    return {i: _arrays(spec["out_npz"]) for i, spec in zip(indices, request["clips"])}, body, api


@pytest.mark.parametrize("on", [False, True])
def test_driver_hands_npz_do_not_depend_on_batch_mates_when_on(driver: ModuleType, shared: Any, tmp_path: Path,
                                                               on: bool) -> None:
    options = {"batch_invariant": True, "mano_rows_per_batch": 5} if on else {}

    alone0, _, _ = _driver_outputs(driver, shared, tmp_path, "alone0", [0], **options)
    alone1, _, _ = _driver_outputs(driver, shared, tmp_path, "alone1", [1], **options)
    together, body, api = _driver_outputs(driver, shared, tmp_path, "together", [0, 1], **options)

    if on:
        assert body["batch_invariant"] is True and body["mano_rows_per_batch"] == 5
        assert set(api.mano_rows) == {5}
        for alone, index in ((alone0, 0), (alone1, 1)):
            assert alone[index].keys() == together[index].keys()
            for key, value in alone[index].items():
                np.testing.assert_array_equal(together[index][key], value, err_msg=f"clip {index} {key}")
    else:                                                              # the control, and the off response as before
        assert "batch_invariant" not in body and "mano_rows_per_batch" not in body
        assert len(api.mano_rows) == 2                                 # one MANO call per side, as before
        moved = [key for alone, index in ((alone0, 0), (alone1, 1)) for key in ("left_vertices", "right_trans")
                 if not np.array_equal(alone[index][key], together[index][key])]
        assert moved


def test_driver_mano_chunks_pad_with_the_chunk_first_row(driver: ModuleType, shared: Any) -> None:
    call = driver._Call({"options": {"batch_invariant": True, "mano_rows_per_batch": 4}}, Path("/nonexistent"), shared,
                        None, "cpu")
    seen: list[np.ndarray] = []

    class _Api:
        @staticmethod
        def run_mano(models: Any, side: str, rot: np.ndarray, hand_pose: np.ndarray, betas: np.ndarray,
                     trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            seen.append(rot[:, 0].copy())
            return np.repeat(rot[:, None], 778, 1).astype(np.float32), np.repeat(rot[:, None], 21, 1).astype(np.float32)

    call.api = _Api()
    rot = np.arange(30, dtype=np.float32).reshape(10, 3)
    cat = {"rot": rot, "hand_pose": np.zeros((10, 45), np.float32), "betas": np.zeros((10, 10), np.float32),
           "trans": np.zeros((10, 3), np.float32)}

    vertices, joints = call._mano_fixed("right", cat)

    assert [s.tolist() for s in seen] == [[0, 3, 6, 9], [12, 15, 18, 21], [24, 27, 24, 24]]
    np.testing.assert_array_equal(vertices[:, 0], rot)
    np.testing.assert_array_equal(joints[:, 5], rot)


def test_driver_default_request_reads_as_off(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    request = _request(tmp_path, _clips())
    assert "batch_invariant" not in request["options"]

    call = driver._Call(request, tmp_path / "response.json", shared, None, "cpu")

    assert call.batch_invariant is False and "batch_invariant" not in call.body
    assert json.loads(json.dumps(call.body)).keys() == {"version", "error", "load_seconds", "warmup_seconds",
                                                        "compute_seconds", "peak_cuda_gb", "clips"}
