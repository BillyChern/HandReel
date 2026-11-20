"""Tests for `fastdo.hands.hawor_core` on CPU with a seeded fake network (no HaWoR weights).

`FakeHawor` has HaWoR's interface with tiny stand-in modules. Its `bbox_est` is the CLIFF bounding-box feature
(Li et al., ECCV 2022) and its `get_trans` the full-frame weak-perspective translation, both written from the published
formulas. One test compares `crop_ulbr` with HaWoR's own `imutils.transform` in a hawor-python subprocess.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import Tensor, nn

from fastdo.hands.hawor_core import (
    CROP_SIZE,
    IMG_MEAN,
    IMG_STD,
    WINDOW,
    FeatureBank,
    HaworOps,
    crop_hands,
    crop_ulbr,
    gather_rows,
    run_backbone,
    run_heads,
)
from fastdo.hands.hawor_plan import ClipGeometry, CropItem, Slot, plan_clip, plan_groups
from tests.hands.hawor_env import HAWOR_PYTHON, cpu_subprocess_env, hawor_root

CORE_FILE = Path(__file__).resolve().parents[2] / "src" / "fastdo" / "hands" / "hawor_core.py"
SIDES = ("left", "right")


class _FakeManoHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pose, self.betas, self.cam = nn.Linear(1280, 96), nn.Linear(1280, 10), nn.Linear(1280, 3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pooled = x.mean(dim=(2, 3))
        cam = self.cam(pooled)
        return self.pose(pooled), self.betas(pooled), torch.cat([0.9 + 0.1 * torch.sigmoid(cam[:, :1]), cam[:, 1:]], 1)


class FakeHawor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.backbone = nn.Conv2d(3, 1280, 16, stride=16)
        self.st_module = nn.Linear(1283, 1280)
        self.mano_head = _FakeManoHead()
        self.motion_module = nn.Linear(99, 96)
        # The identity 6D rotation in the bias keeps every joint's 6D vector well conditioned. With random weights
        # alone some are nearly degenerate, and Gram-Schmidt amplifies float32 roundoff (4.5e-8) up to 2.7e-5.
        with torch.no_grad():
            self.motion_module.bias += torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(16)

    def bbox_est(self, center: Tensor, scale: Tensor, img_focal: Tensor, img_center: Tensor) -> Tensor:
        return torch.stack([(center[:, 0] - img_center[:, 0]) / img_focal, (center[:, 1] - img_center[:, 1]) / img_focal,
                            200.0 * scale / img_focal], dim=-1)

    def get_trans(self, pred_cam: Tensor, center: Tensor, scale: Tensor, img_focal: Tensor, img_center: Tensor) -> Tensor:
        s, tx, ty = pred_cam.unbind(-1)
        bs = 200.0 * scale * s
        return torch.stack([tx + 2 * (center[:, 0] - img_center[:, 0]) / bs, ty + 2 * (center[:, 1] - img_center[:, 1]) / bs,
                            2 * img_focal / bs], dim=-1)[:, None]


def _rot6d_to_rotmat(x: Tensor) -> Tensor:
    a = x.reshape(-1, 2, 3).permute(0, 2, 1)
    b1 = F.normalize(a[..., 0], dim=-1)
    b2 = F.normalize(a[..., 1] - (b1 * a[..., 1]).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


def _rotmat_to_aa(r: Tensor) -> Tensor:
    return torch.from_numpy(Rotation.from_matrix(r.detach().double().cpu().numpy()).as_rotvec()).to(r.device, r.dtype)


OPS = HaworOps(rot6d_to_rotmat=_rot6d_to_rotmat, rotmat_to_aa=_rotmat_to_aa)


@dataclass(frozen=True)
class _Clip:
    frames: np.ndarray
    frame_idx: np.ndarray
    boxes: dict[str, np.ndarray]
    geom: ClipGeometry


def _clip(n: int, stride: int, h: int, w: int, focal: float, missing_left: tuple[int, ...] = (), seed: int = 0) -> _Clip:
    rng = np.random.default_rng(seed)
    boxes = {}
    for side in SIDES:
        b = np.full((n, 5), np.nan)
        for row in range(n):
            if side == "left" and row in missing_left:
                continue
            size, cx, cy = rng.uniform(0.15, 0.35) * h, rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.2 * h, 0.8 * h)
            b[row] = [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, 0.8]
        boxes[side] = b
    frames = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
    return _Clip(frames, np.arange(n, dtype=np.int64) * stride, boxes, ClipGeometry(w, h, focal))


def _clips() -> list[_Clip]:
    return [_clip(12, 3, 360, 640, 500.0, seed=1), _clip(8, 2, 240, 320, 300.0, missing_left=(3,), seed=2)]


def _run_groups(model: FakeHawor, clips: list[_Clip], groups: list[list[int]], windows_per_batch: int = 16,
                backbone_batch: int = 64) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    results = {}
    for group in groups:
        bank = FeatureBank(model, batch=backbone_batch, bf16=False)
        slots: list[Slot] = []
        geoms = []
        for local, index in enumerate(group):
            clip = clips[index]
            items, clip_slots = plan_clip(local, clip.frame_idx, clip.boxes, clip.geom, bank_offset=bank.n_items)
            bank.push(crop_hands(torch.from_numpy(clip.frames), items))
            slots += clip_slots
            geoms.append(clip.geom)
        bank.flush()
        out = run_heads(model, OPS, bank.features(), slots, geoms, windows_per_batch=windows_per_batch)
        for local, index in enumerate(group):
            for side in SIDES:
                results[(index, side)] = gather_rows(out, slots, local, side, len(clips[index].frame_idx))
    return results


def _assert_rows_equal(a: dict[tuple[int, str], dict[str, np.ndarray]],
                       b: dict[tuple[int, str], dict[str, np.ndarray]]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key]["valid"], b[key]["valid"])
        for name in ("rot", "hand_pose", "trans", "betas"):
            np.testing.assert_allclose(a[key][name], b[key][name], rtol=0, atol=1e-5, err_msg=f"{key} {name}")


def test_constants() -> None:
    assert CROP_SIZE == 256 and WINDOW == 16
    assert IMG_MEAN == (0.485, 0.456, 0.406) and IMG_STD == (0.229, 0.224, 0.225)


def test_crop_ulbr_known_values() -> None:
    assert crop_ulbr((640.0, 360.0), 1.0) == ((539, 259), (740, 460))
    assert crop_ulbr((10.3, 700.8), 0.5) == ((-39, 650), (60, 750))
    assert crop_ulbr((1275.5, 3.25), 0.465) == ((1228, -43), (1322, 49))


_HAWOR_TRANSFORM = r"""
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("fastdo_shared_hawor_core", sys.argv[1])
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
sys.path.insert(0, ".")
from lib.utils.imutils import transform
rng = np.random.default_rng(0)
equal = 0
for _ in range(200):
    c = (float(rng.uniform(-50, 1330)), float(rng.uniform(-50, 770)))
    s = float(rng.uniform(0.2, 2.0))
    ul = np.array(transform([1, 1], c, s, [256, 256], invert=1)) - 1
    br = np.array(transform([257, 257], c, s, [256, 256], invert=1)) - 1
    equal += int(core.crop_ulbr(c, s) == ((int(ul[0]), int(ul[1])), (int(br[0]), int(br[1]))))
print(json.dumps({"equal": equal}))
"""


def test_crop_ulbr_matches_hawor_transform() -> None:
    root = hawor_root()
    if not HAWOR_PYTHON.is_file() or root is None or not (root / "lib" / "utils" / "imutils.py").is_file():
        pytest.skip("hawor python or $FASTDO_HAWOR_ROOT/lib/utils/imutils.py not available")
    proc = subprocess.run([str(HAWOR_PYTHON), "-c", _HAWOR_TRANSFORM, str(CORE_FILE)], capture_output=True, text=True,
                          cwd=root, env=cpu_subprocess_env(), timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"equal": 200}


@pytest.mark.parametrize(("center", "scale", "pad_cols", "inner_cols", "atol"), [
    ((0.0, 360.0), 0.5, (0, 101), (156, 256), 1e-6),                   # 100 px crop, half outside: boundary at col 128
    ((20.0, 360.0), 0.5, (0, 76), (78, 256), 1e-6),                    # 100 px crop, 30 px outside: boundary near 77
    ((20.0, 360.0), 1.6, (0, 101), (156, 256), 1 / 255 / min(IMG_STD) + 1e-6),  # 320 px crop, downsampled
])
def test_crop_hands_zero_pads_outside_frame(center: tuple[float, float], scale: float, pad_cols: tuple[int, int],
                                            inner_cols: tuple[int, int], atol: float) -> None:
    frame = torch.full((1, 720, 1280, 3), 128, dtype=torch.uint8)

    crop = crop_hands(frame, [CropItem(clip=0, row=0, side="right", center=center, scale=scale, flip=False)])

    assert crop.shape == (1, 3, 256, 256) and crop.dtype == torch.float32
    for c in range(3):
        pad, inner = -IMG_MEAN[c] / IMG_STD[c], (128 / 255 - IMG_MEAN[c]) / IMG_STD[c]
        assert float((crop[0, c, :, pad_cols[0]:pad_cols[1]] - pad).abs().max()) <= atol
        assert float((crop[0, c, :, inner_cols[0]:inner_cols[1]] - inner).abs().max()) <= atol


def test_crop_hands_flip_uses_mirrored_frame() -> None:
    frames = torch.from_numpy(np.random.default_rng(0).integers(0, 256, (2, 90, 160, 3), dtype=np.uint8))
    spots = [((30.5, 40.0), 0.4), ((100.0, 45.0), 0.3)]                # partly outside, fully inside
    left = [CropItem(0, 1, "left", c, s, True) for c, s in spots]
    right = [CropItem(0, 0, "right", c, s, False) for c, s in spots]

    assert float((crop_hands(frames, left) - crop_hands(frames[1:2].flip(2), right)).abs().max()) == 0.0


def test_crop_hands_chunked_upload_matches_single_upload() -> None:
    clip = _clips()[0]
    items, _ = plan_clip(0, clip.frame_idx, clip.boxes, clip.geom, bank_offset=0)
    frames = torch.from_numpy(clip.frames)

    whole = crop_hands(frames, items)
    chunked = crop_hands(frames, items, device="cpu", frames_per_chunk=5)

    assert whole.shape == (len(items), 3, 256, 256)
    assert torch.equal(whole, chunked)


def test_run_backbone_trims_width() -> None:
    model = FakeHawor()
    seen: list[tuple[int, ...]] = []
    model.backbone.register_forward_hook(lambda _m, inp, _out: seen.append(tuple(inp[0].shape)))
    crops = torch.randn(5, 3, 256, 256, generator=torch.Generator().manual_seed(0))

    feats = run_backbone(model, crops, batch=4, bf16=False)

    assert seen == [(4, 3, 256, 192), (1, 3, 256, 192)]
    assert feats.shape == (5, 1280, 16, 12) and feats.dtype == torch.float32
    assert run_backbone(model, crops[:2], bf16=True).dtype == torch.bfloat16


def test_feature_bank_fifo_matches_run_backbone() -> None:
    model = FakeHawor()
    sizes: list[int] = []
    model.backbone.register_forward_hook(lambda _m, inp, _out: sizes.append(int(inp[0].shape[0])))
    crops = torch.randn(13, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    bank = FeatureBank(model, batch=4, bf16=False)

    for part in (crops[:3], crops[3:10], crops[10:]):
        bank.push(part)
    with pytest.raises(RuntimeError, match="flush"):
        bank.features()
    assert sizes == [4, 4, 4] and bank.n_items == 13
    bank.flush()
    assert sizes == [4, 4, 4, 1]
    expected = run_backbone(model, crops, batch=64, bf16=False)
    assert float((bank.features() - expected).abs().max()) <= 1e-5
    assert FeatureBank(model, batch=4, bf16=False).features().shape == (0, 1280, 16, 12)


def test_run_heads_is_batch_invariant() -> None:
    model, clips = FakeHawor(), _clips()
    bank = FeatureBank(model, batch=64, bf16=False)
    slots: list[Slot] = []
    for local, clip in enumerate(clips):
        items, clip_slots = plan_clip(local, clip.frame_idx, clip.boxes, clip.geom, bank_offset=bank.n_items)
        bank.push(crop_hands(torch.from_numpy(clip.frames), items))
        slots += clip_slots
    bank.flush()

    outs = [run_heads(model, OPS, bank.features(), slots, [c.geom for c in clips], windows_per_batch=w)
            for w in (1, 3, 16)]

    s = len(slots)
    assert outs[0].rotmat.shape == (s, 16, 3, 3) and outs[0].aa.shape == (s, 16, 3)
    assert outs[0].betas.shape == (s, 10) and outs[0].trans.shape == (s, 3)
    assert all(getattr(outs[0], n).dtype == torch.float32 for n in ("rotmat", "aa", "betas", "trans"))
    for other in outs[1:]:
        for name in ("rotmat", "aa", "betas", "trans"):
            assert float((getattr(other, name) - getattr(outs[0], name)).abs().max()) <= 1e-5, name
    with pytest.raises(ValueError, match="16"):
        run_heads(model, OPS, bank.features(), slots[:15], [c.geom for c in clips])


def test_run_heads_flip_semantics() -> None:
    model, geom = FakeHawor(), ClipGeometry(640, 360, 500.0)
    bank = torch.randn(16, 1280, 16, 12, generator=torch.Generator().manual_seed(2))
    centers = [(100.0 + 20 * q, 150.0 + 5 * q) for q in range(16)]
    scales = [0.4 + 0.01 * q for q in range(16)]
    right = [Slot(0, "right", q, q, 0.0, centers[q], scales[q], False, q) for q in range(16)]
    left = [replace(slot, side="left", flip=True) for slot in right]

    out = run_heads(model, OPS, bank, right + left, [geom])

    with torch.no_grad():
        center, scale = torch.tensor(centers), torch.tensor(scales)
        focal, ic = torch.full((16,), 500.0), torch.tensor([[320.0, 180.0]] * 16)
        info = model.bbox_est(center, scale, focal, ic)
        x = torch.cat([bank, info[:, :, None, None].expand(-1, -1, 16, 12)], 1).permute(2, 3, 0, 1).reshape(192, 16, 1283)
        x = model.st_module(x).reshape(16, 12, 16, 1280).permute(2, 3, 0, 1)
        pose, betas, cam = model.mano_head(x)
        rotmat = _rot6d_to_rotmat(model.motion_module(torch.cat([pose, info], -1)[None])[0]).reshape(16, 16, 3, 3)
        cam[:, 1] *= -1
        unmirrored = torch.stack([2 * 320.0 - center[:, 0] - 1, center[:, 1]], 1)
        trans = model.get_trans(cam, unmirrored, scale, focal, ic)[:, 0]
        aa = _rotmat_to_aa(rotmat.reshape(-1, 3, 3)).reshape(16, 16, 3)
        aa[..., 1:] *= -1
    for name, expected in (("rotmat", rotmat), ("betas", betas), ("trans", trans), ("aa", aa)):
        assert float((getattr(out, name)[16:] - expected).abs().max()) <= 1e-6, name
    assert float((out.rotmat[16:] - out.rotmat[:16]).abs().max()) <= 1e-6       # rotmat itself is not negated
    assert float((out.aa[16:, :, 1:] + out.aa[:16, :, 1:]).abs().max()) <= 1e-6


def test_multi_clip_equals_single_clip_pipeline() -> None:
    model, clips = FakeHawor(), _clips()

    together = _run_groups(model, clips, [[0, 1]])
    alone = _run_groups(model, clips, [[0]]) | _run_groups(model, clips, [[1]])

    _assert_rows_equal(together, alone)
    rows = together[(1, "left")]
    assert rows["rot"].shape == (8, 3) and rows["rot"].dtype == np.float32 and rows["hand_pose"].shape == (8, 45)
    assert rows["trans"].shape == (8, 3) and rows["betas"].shape == (8, 10) and rows["slot_index"].dtype == np.int64
    assert rows["valid"].tolist() == [True, True, True, False, True, True, True, True]
    assert np.isnan(rows["trans"][3]).all() and rows["slot_index"][3] == -1
    assert np.isfinite(rows["trans"][rows["valid"]]).all() and (rows["slot_index"][rows["valid"]] >= 0).all()


def test_crop_limit_splits_groups_with_equal_outputs() -> None:
    model, clips = FakeHawor(), _clips()
    counts = [len(plan_clip(0, c.frame_idx, c.boxes, c.geom, bank_offset=0)[0]) for c in clips]

    groups = plan_groups(counts, clips_per_group=8, max_crops_per_group=sum(counts) - 1)

    assert groups == [[0], [1]]
    assert plan_groups(counts, clips_per_group=8, max_crops_per_group=sum(counts)) == [[0, 1]]
    _assert_rows_equal(_run_groups(model, clips, groups, backbone_batch=7),
                       _run_groups(model, clips, [[0, 1]], backbone_batch=7))
