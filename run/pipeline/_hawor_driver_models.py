"""Model-side helpers of the batched HaWoR driver; hawor env only, loaded by path from `_hawor_batched_driver.py`.

Top-level imports are torch, numpy and the standard library. HaWoR (`lib`, `hawor`, `scripts`), ultralytics and smplx
are imported inside the functions, from the HaWoR checkout the driver runs in (cwd = `hands.hawor_root`, first on
`sys.path`). Nothing from those projects is copied into this repository.

Runtime licences: HaWoR code and weights, CC BY-NC-ND 4.0 (`<hawor_root>/license.txt`); ultralytics (the YOLOv8 hand
detector runtime), AGPL-3.0; MANO model files, the MANO licence (non-commercial research use).
"""

from __future__ import annotations

import collections
import logging
import os
import sys
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("hawor_driver_models")

HAWOR_CKPT = "weights/hawor/checkpoints/hawor.ckpt"
DETECTOR = "weights/external/detector.pt"
MANO_DIRS = ("_DATA/data/mano", "_DATA/data_left/mano_left")
MANO_ROWS_PER_CALL = 8192
SELF_CHECK_FOCAL = 999.0292358398438


@dataclass(frozen=True)
class LoadedModels:
    """Everything the driver loads once per call: HaWoR, the YOLO hand detector, MANO layers, ops and closed faces."""

    hawor: Any
    yolo: Any
    mano: dict[str, Any]
    ops: Any
    faces: dict[str, np.ndarray]


def _safe_globals() -> list[Any]:
    """The ten globals `hawor.ckpt` needs under `torch.load(weights_only=True)`."""
    import omegaconf
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    return [typing.Any, collections.defaultdict, int, list, dict, omegaconf.dictconfig.DictConfig,
            omegaconf.listconfig.ListConfig, AnyNode, ContainerMetadata, Metadata]


def load_models(hawor_root: Path, shared: Mapping[str, ModuleType]) -> LoadedModels:
    """Loads HaWoR (fp32, eval, CUDA), the detector, both MANO layers and the closed MANO faces.

    Raises:
        FileNotFoundError: If a checkpoint or MANO directory is missing (the message starts with "weights missing").
        RuntimeError: If CUDA is unavailable.
    """
    root = Path(hawor_root).resolve()
    for required in (HAWOR_CKPT, DETECTOR, *MANO_DIRS):
        if not (root / required).exists():
            raise FileNotFoundError(f"weights missing: {root / required}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the HaWoR driver")
    os.chdir(root)                                                   # MANO and detector paths are root-relative
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from hawor.utils.process import get_mano_faces
    from hawor.utils.rotation import rotation_matrix_to_angle_axis
    from lib.models.mano_wrapper import MANO
    from lib.utils.geometry import rot6d_to_rotmat_hmr2
    from scripts.scripts_test_video.hawor_video import load_hawor
    from ultralytics import YOLO

    with torch.serialization.safe_globals(_safe_globals()):
        hawor, _ = load_hawor(str(root / HAWOR_CKPT))
    hawor = hawor.cuda().eval()
    yolo = YOLO("./" + DETECTOR)
    right = MANO(data_dir="_DATA/data/", model_path="_DATA/data/mano", gender="neutral", num_hand_joints=15,
                 create_body_pose=False).cuda()
    left = MANO(data_dir="_DATA/data_left/", model_path="_DATA/data_left/mano_left", gender="neutral",
                num_hand_joints=15, create_body_pose=False, is_rhand=False).cuda()
    with torch.no_grad():
        left.shapedirs[:, 0, :] *= -1
    faces = shared["hand_npz"].closed_faces(np.asarray(get_mano_faces()))
    ops = shared["hawor_core"].HaworOps(rot6d_to_rotmat=rot6d_to_rotmat_hmr2, rotmat_to_aa=rotation_matrix_to_angle_axis)
    logger.info("loaded HaWoR, detector and MANO from %s", root)
    return LoadedModels(hawor=hawor, yolo=yolo, mano={"left": left, "right": right}, ops=ops, faces=faces)


def warm_up(models: LoadedModels, shared: Mapping[str, ModuleType], det_frames_per_batch: int) -> None:
    """One detector call on black 720p frames, one bf16 backbone pass on 16 zero crops, one heads window, MANO."""
    black = np.zeros((720, 1280, 3), np.uint8)
    models.yolo.predict([black] * det_frames_per_batch, conf=0.2, half=True, verbose=False)
    core, plan = shared["hawor_core"], shared["hawor_plan"]
    bank = core.run_backbone(models.hawor, torch.zeros(16, 3, 256, 256, device="cuda"), batch=16, bf16=True)
    slots = [plan.Slot(0, "right", q, q, 0.0, (640.0, 360.0), 1.0, False, q) for q in range(16)]
    core.run_heads(models.hawor, models.ops, bank, slots, [plan.ClipGeometry(1280, 720, 1000.0)], windows_per_batch=1)
    zeros = np.zeros((16, 3), np.float32)
    for side in ("left", "right"):
        run_mano(models, side, zeros, np.zeros((16, 45), np.float32), np.zeros((16, 10), np.float32), zeros + 0.5)
    torch.cuda.synchronize()


def disable_nms_time_limit() -> bool:
    """YOLO's NMS (`options.deterministic`) without its wall-clock limit; returns True when newly applied.

    ultralytics' `non_max_suppression` loops over a predict call's images and BREAKS once `2.0 + max_time_img * batch`
    seconds (2.8 s for our 16-frame calls) have passed since the loop started, leaving every later image of the batch
    with no detections. A stall of the driver (CPU starvation of this niced process, a busy card) therefore silently
    drops the hands of a chunk's trailing frames, and the result depends on timing. This wraps the module
    attribute the detect predictor calls (`ultralytics.utils.nms.non_max_suppression`) so that every call passes
    `max_time_img=inf`; no installed file is changed. An NMS that finishes inside the limit returns exactly what it
    returned before, so outputs change only where the limit would have cut a batch short.
    """
    from ultralytics.utils import nms

    if getattr(nms.non_max_suppression, "fastdo_no_time_limit", False):
        return False
    original = nms.non_max_suppression

    def no_time_limit(*args: Any, **kwargs: Any) -> Any:
        kwargs["max_time_img"] = float("inf")
        return original(*args, **kwargs)

    no_time_limit.fastdo_no_time_limit = True                        # type: ignore[attr-defined]
    no_time_limit.__wrapped__ = original                             # type: ignore[attr-defined]
    nms.non_max_suppression = no_time_limit
    return True


def detect_clip(models: LoadedModels, shared: Mapping[str, ModuleType], frames: np.ndarray, det_conf: float,
                det_frames_per_batch: int) -> dict[str, np.ndarray]:
    """Raw top-1 boxes per side, `(N,5)` float64 with NaN rows, from fixed-size padded predict calls on one clip.

    Every call holds exactly `det_frames_per_batch` frames (the last is padded with copies of the clip's last frame and
    the padded results are dropped), so a clip's boxes depend only on its own frames and that chunk size.
    """
    import cv2

    boxes = shared["hawor_boxes"]
    n = int(frames.shape[0])
    # BGR per chunk into one reused buffer. Flipping the whole clip at once (`frames[..., ::-1]` made
    # contiguous) cost 2.8-3.9 s per 100 x 1080x1920 window -- a fresh 622 MB array -- against 0.16 s of detector;
    # the channel swap is a permutation, so every batch the detector sees is bit-identical to before.
    buffer = np.empty((det_frames_per_batch, *frames.shape[1:]), dtype=np.uint8)
    detections = []
    for start, stop, n_pad in boxes.padded_chunks(n, det_frames_per_batch):
        for i, row in enumerate([*range(start, stop), *[n - 1] * n_pad]):
            cv2.cvtColor(np.ascontiguousarray(frames[row]), cv2.COLOR_RGB2BGR, dst=buffer[i])
        results = models.yolo.predict(list(buffer), conf=det_conf, half=True, verbose=False)
        for result in results[: stop - start]:
            b = result.boxes
            detections.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()))
    return boxes.select_hand_boxes(detections, min_conf=det_conf)


def run_mano(models: LoadedModels, side: str, rot: np.ndarray, hand_pose: np.ndarray, betas: np.ndarray,
             trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame MANO `(R,778,3)` vertices and `(R,21,3)` joints (OpenPose order with tips), float32."""
    from hawor.utils.geometry import aa_to_rotmat

    layer = models.mano[side]
    vertices, joints = [], []
    with torch.no_grad():
        for start in range(0, int(rot.shape[0]), MANO_ROWS_PER_CALL):
            sl = slice(start, start + MANO_ROWS_PER_CALL)
            orient = aa_to_rotmat(torch.as_tensor(rot[sl], dtype=torch.float32, device="cuda"))[:, None]
            pose = aa_to_rotmat(torch.as_tensor(hand_pose[sl], dtype=torch.float32, device="cuda").reshape(-1, 3))
            out = layer(global_orient=orient, hand_pose=pose.view(-1, 15, 3, 3),
                        betas=torch.as_tensor(betas[sl], dtype=torch.float32, device="cuda"),
                        transl=torch.as_tensor(trans[sl], dtype=torch.float32, device="cuda"), pose2rot=False)
            vertices.append(out.vertices.float().cpu().numpy())
            joints.append(out.joints.float().cpu().numpy())
    if not vertices:
        return np.zeros((0, 778, 3), np.float32), np.zeros((0, 21, 3), np.float32)
    return np.concatenate(vertices), np.concatenate(joints)


def self_check(models: LoadedModels, shared: Mapping[str, ModuleType]) -> dict[str, float]:
    """Max abs difference of `run_heads` (per-slot flip, one batch) against `HAWOR.forward_step` per window, fp32.

    32 seeded random crops: window 0 right-hand slots, window 1 left-hand slots (`do_flip`); centres x in [300,900),
    y in [200,500), scale in [0.6,1.4), f = 999.0292358398438, 1280x720.
    """
    core, plan = shared["hawor_core"], shared["hawor_plan"]
    g = torch.Generator().manual_seed(0)
    imgs = torch.randn(32, 3, 256, 256, generator=g).cuda()
    center = torch.stack([300 + 600 * torch.rand(32, generator=g), 200 + 300 * torch.rand(32, generator=g)], 1).cuda()
    scale = (0.6 + 0.8 * torch.rand(32, generator=g)).cuda()
    focal = torch.full((32,), SELF_CHECK_FOCAL, device="cuda")
    img_center = torch.tensor([[640.0, 360.0]] * 32, device="cuda")
    with torch.no_grad():
        ref_rot, ref_betas, ref_trans = [], [], []
        for w in range(2):
            sl = slice(16 * w, 16 * (w + 1))
            batch = {"img": imgs[sl][None], "center": center[sl][None].clone(), "scale": scale[sl][None],
                     "img_focal": focal[sl][None], "img_center": img_center[sl][None]}
            if w == 1:
                batch["do_flip"] = torch.ones(1, 16, device="cuda")
            out = models.hawor.forward_step(batch)["out"]
            ref_rot.append(out["pred_rotmat"])
            ref_betas.append(out["pred_shape"])
            ref_trans.append(out["trans_full"][:, 0])
        bank = core.run_backbone(models.hawor, imgs, batch=32, bf16=False)
    geom = plan.ClipGeometry(1280, 720, SELF_CHECK_FOCAL)
    c, s = center.cpu().tolist(), scale.cpu().tolist()
    slots = [plan.Slot(0, "left" if q >= 16 else "right", q, q, 0.0, (c[q][0], c[q][1]), s[q], q >= 16, q)
             for q in range(32)]
    heads = core.run_heads(models.hawor, models.ops, bank, slots, [geom], windows_per_batch=2)
    return {"max_abs_rotmat": float((heads.rotmat - torch.cat(ref_rot)).abs().max()),
            "max_abs_betas": float((heads.betas - torch.cat(ref_betas)).abs().max()),
            "max_abs_trans_m": float((heads.trans - torch.cat(ref_trans)).abs().max())}


if __name__ == "__main__":
    print("usage: not a command. Model-side helpers of the batched HaWoR driver, loaded by path from "
          "run/pipeline/_hawor_batched_driver.py in the hawor env.")
    raise SystemExit(0 if sys.argv[1:] in ([], ["-h"], ["--help"]) else 2)
