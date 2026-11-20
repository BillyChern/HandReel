"""HOI4D (Liu et al., CVPR 2022) ground truth as `ClipRecord`s: the second labelled benchmark.

Layout under `root` (data/hoi4d, CC BY-NC 4.0, see its LICENSE_NOTE.md); `<seq>` is
`ZY2021080000<cam>/H<h>/C<c>/N<n>/S<s>/s<l>/T<t>` (camera, human, category, instance, room, layout, task):

- `HOI4D_release/<seq>/align_rgb/image.mp4`: the RGB video, 1920x1080 at 15 fps (300 frames, 20 s).
- `HOI4D_annotations/<seq>/objpose/<t>.json` (unpadded frame index `t`; 64 rigid sequences pad it to 5 digits): `dataList[0]` holds `center {x,y,z}`
  (metres, camera frame) and `rotation {x,y,z}`, Euler angles read with `Rotation.from_euler("XYZ", ...)` -- the
  official loader's call (HOI4D-Instructions README, `read_rtd`). Object-to-camera: `x_cam = R x_model + center`.
- `HOI4D_annotations/<seq>/2Dseg/{mask,shift_mask}/<t:05d>.png`: RGB label maps on the PASCAL-VOC palette; label 1
  (the category's object) is (128,0,0) and label 2 (the right hand) is (0,128,0) in every rigid category
  (`meta/hoi4d_motion_segmentation_label.csv`). Camera 1 ships `mask`, cameras 2-4 ship `shift_mask`.
- `HOI4D_annotations/<seq>/action/color.json`: action segments (`events` with `startTime`, or on a few sequences
  `markResult.marks` with `hdTimeStart`), in seconds.
- `HOI4D_CAD_Model_for_release/rigid/<Category>/<n:03d>.obj`: the instance's CAD model in metres, centred.
- `camera_params/<camera>/intrin.npy`: (3,3) pixel intrinsics of the RGB camera.

Object-pose registration (MEASURED). The CAD silhouette under
the GT object pose sits a near-constant image offset from the annotated object mask WITHIN a sequence (median absolute
deviation 2-3 px over its frames), but the offset varies BETWEEN sequences: medians of the sequence offsets
0/+11/+5/+14 px in x on cameras 1-4 (both samples; cameras 2-4 ship `shift_mask`), between-sequence spread 2-10 px, a
few sequences > 50 px. A per-camera constant fitted on one seeded sample missed camera 3 by 10-14 px on a held-out one,
so there is no per-camera constant. The GT HANDS mostly do not share the offset: their kps2D fall on the hand label
better with the shipped K than when moved by the object's offset on 37/54 DEV windows with offsets >= 8 px (cameras
2 and 4: 30/37; camera 1, which ships `mask` not `shift_mask`: 3/10), and our MANO forward kinematics reproduces kps2D
to 1.3 px with the shipped K. So on cameras 2-4 the image, the intrinsics and the hands agree and the OBJECT pose
(labelled on depth point clouds) is off by a per-sequence lateral offset; camera 1 is mixed. `load(seq, pose_shift_px=(dx, dy))` corrects it: every valid object translation moves
by (dx * z / fx, dy * z / fy, 0) at its own depth z (rotation and depth untouched), which moves the object's image by
(dx, dy) px up to its depth relief. The offset is measured from the sequence's own GT masks and poses alone
(`fastdo.bench.hoi4d_bench.estimate_registration`) and recorded per window in the split; the engine is given the
SHIPPED intrinsics.
- `Hand_pose/handpose_right_hand/<seq>/<t>.pickle`: MANO `poseCoeff` (48, axis-angle, `flat_hand_mean=True`, so the
  45 finger values are the FULL finger pose), `beta` (10), `trans` (camera frame, metres), `kps2D` (21,2).

Every convention above except the directory names was VERIFIED by reprojection, not assumed (CAD model posed with
the GT pose and projected with the intrinsics against the 2D object mask, per Euler order and frame offset).

Frame indexing: `ClipRecord.frame_idx` is the ABSOLUTE source frame (0..299), exactly as in `DexYCBReader`, so a
scorer that indexes `gt[frame_idx]` with a decode's `frame_idx` works unchanged on a window of the video. A frame
with no objpose file, `isEffective != 1` or a non-finite value gets an identity pose and confidence 0.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from fastdo.io.clip_record import ClipRecord, HandTrack

__all__ = [
    "EULER_ORDER",
    "EXCLUDED_CATEGORIES",
    "FPS",
    "HAND_COLOUR",
    "OBJECT_COLOUR",
    "RIGID_CATEGORIES",
    "HOI4DId",
    "HOI4DReader",
    "contact_time",
    "correct_object_pose",
    "hand_track",
    "parse_events",
    "parse_objpose",
    "parse_sequence_id",
]

logger = logging.getLogger(__name__)

FPS = 15.0
EULER_ORDER = "XYZ"
OBJECT_COLOUR = (128, 0, 0)
HAND_COLOUR = (0, 128, 0)

RIGID_CATEGORIES: Mapping[int, str] = MappingProxyType(
    {1: "ToyCar", 2: "Mug", 5: "Bottle", 7: "Bowl", 12: "Kettle", 13: "Knife"})
"""Hand-held rigid categories: category id -> CAD directory (never the objpose `label`, which reads "Toycar")."""

EXCLUDED_CATEGORIES: Mapping[int, str] = MappingProxyType({
    3: "Laptop (articulated: display hinge)",
    4: "StorageFurniture (articulated: doors and drawers)",
    6: "Safe (articulated: door)",
    8: "Bucket (articulated: handle)",
    9: "Scissors (articulated: two blades)",
    11: "Pliers (articulated: two jaws)",
    14: "TrashCan (articulated: lid)",
    17: "Lamp (articulated: links)",
    18: "Stapler (articulated: lid and base)",
    20: "Chair (rigid CAD, but a two-handed furniture carry: label 3 is the LEFT hand; out of scope for a "
        "right-hand dexterous benchmark and for the right-only geometry rule)",
})
"""Categories left out of the rigid benchmark, with the reason. Rigid first; articulated ones need part poses."""

_ID = re.compile(r"^ZY2021080000(\d)/H(\d+)/C(\d+)/N(\d+)/S(\d+)/s(\d+)/T(\d+)$")
_CONTACT_EVENTS = frozenset({"grasp", "push", "pull", "press", "pickup", "carry", "cut", "dump", "putdown"})


@dataclass(frozen=True)
class HOI4DId:
    """The parsed components of a sequence id."""

    camera: int
    human: int
    category: int
    instance: int
    room: int
    layout: int
    task: int

    @property
    def camera_dir(self) -> str:
        return f"ZY2021080000{self.camera}"


def parse_sequence_id(seq: str) -> HOI4DId:
    """`ZY2021080000<cam>/H../C../N../S../s../T..` -> `HOI4DId`.

    Raises:
        ValueError: If `seq` does not match the layout.
    """
    m = _ID.match(seq)
    if m is None:
        raise ValueError(f"not a HOI4D sequence id: {seq!r}")
    return HOI4DId(*(int(g) for g in m.groups()))


def parse_objpose(data: Mapping[str, Any], euler_order: str = EULER_ORDER) -> tuple[np.ndarray, bool]:
    """One objpose JSON as a (4,4) float32 object-to-camera pose and a validity flag.

    Invalid (identity, False) when `isEffective != 1`, `dataList` is empty, or a value is non-finite. Only
    `dataList[0]` is read: rigid-category files carry one item (checked on the pool by the conventions study).

    Raises:
        ValueError: If a required key is missing (a layout difference, not a missing frame).
    """
    identity = np.eye(4, dtype=np.float32)
    try:
        items = data["dataList"] if "dataList" in data else data["objects"]
        if int(data.get("isEffective", 1)) != 1 or len(items) == 0:
            return identity, False
        centre = [float(items[0]["center"][axis]) for axis in "xyz"]
        angles = [float(items[0]["rotation"][axis]) for axis in "xyz"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"objpose JSON does not match the HOI4D layout: {exc!r}") from exc
    if not all(math.isfinite(v) for v in centre + angles):
        return identity, False
    pose = identity.copy()
    pose[:3, :3] = Rotation.from_euler(euler_order, angles).as_matrix()
    pose[:3, 3] = centre
    return pose, True


def parse_events(data: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    """Action segments `(label, start_s, end_s)` in file order, from either action JSON layout.

    Raises:
        ValueError: If neither layout is present.
    """
    if "events" in data:
        return [(str(e["event"]), float(e["startTime"]), float(e["endTime"])) for e in data["events"]]
    marks = (data.get("markResult") or {}).get("marks")
    if marks is None:
        raise ValueError(f"action JSON has neither 'events' nor 'markResult.marks': keys {sorted(data)}")
    return [(str(e["event"]), float(e["hdTimeStart"]), float(e["hdTimeEnd"])) for e in marks]


def contact_time(events: Sequence[tuple[str, float, float]]) -> float | None:
    """Start (s) of the first hand-object contact segment (grasp, push, pull, press, ...), or None when there is none."""
    starts = [start for label, start, _ in events if label.lower() in _CONTACT_EVENTS]
    return min(starts) if starts else None


class HOI4DReader:
    """Reads HOI4D ground truth under `root`. Construction opens nothing."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # --- paths ------------------------------------------------------------------------------------------------------

    def video_path(self, seq: str) -> Path:
        parse_sequence_id(seq)
        return self.root / "HOI4D_release" / seq / "align_rgb" / "image.mp4"

    def mesh_path(self, seq: str) -> Path:
        """The instance's rigid CAD model. Raises ValueError for a category outside `RIGID_CATEGORIES`."""
        sid = parse_sequence_id(seq)
        if sid.category not in RIGID_CATEGORIES:
            raise ValueError(f"{seq}: category C{sid.category} is not a rigid benchmark category "
                             f"({EXCLUDED_CATEGORIES.get(sid.category, 'unknown')})")
        return (self.root / "HOI4D_CAD_Model_for_release" / "rigid" / RIGID_CATEGORIES[sid.category]
                / f"{sid.instance:03d}.obj")

    def hand_pickle(self, seq: str, t: int, side: str = "right") -> Path:
        parse_sequence_id(seq)
        return self.root / "Hand_pose" / f"handpose_{side}_hand" / seq / f"{int(t)}.pickle"

    def mask_dir(self, seq: str) -> Path:
        """`2Dseg/shift_mask` if present, else `2Dseg/mask`. Raises FileNotFoundError when neither exists."""
        seg = self.root / "HOI4D_annotations" / seq / "2Dseg"
        for name in ("shift_mask", "mask"):
            if (seg / name).is_dir():
                return seg / name
        raise FileNotFoundError(f"{seg}: no mask or shift_mask directory")

    # --- small reads ------------------------------------------------------------------------------------------------

    def sequences(self, categories: Sequence[int] = tuple(RIGID_CATEGORIES)) -> list[str]:
        """`meta/release.txt` ids in the given categories, file order, de-duplicated."""
        wanted, out, seen = set(categories), [], set()
        for line in (self.root / "meta" / "release.txt").read_text(encoding="utf-8").splitlines():
            seq = line.strip()
            if seq and seq not in seen and parse_sequence_id(seq).category in wanted:
                out.append(seq)
                seen.add(seq)
        return out

    def intrinsics(self, seq: str, shift_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """(3,3) float32 RGB intrinsics of the sequence's camera as shipped; `shift_px` = (dx, dy) moves the principal
        point (a diagnostic only: the benchmark corrects the object pose instead, see the module docstring)."""
        path = self.root / "camera_params" / parse_sequence_id(seq).camera_dir / "intrin.npy"
        K = np.array(np.load(path, allow_pickle=False), dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError(f"{path}: expected a finite (3,3) intrinsics matrix, got {K.shape}")
        K[0, 2] += float(shift_px[0])
        K[1, 2] += float(shift_px[1])
        return K.astype(np.float32)

    def events(self, seq: str) -> list[tuple[str, float, float]]:
        path = self.root / "HOI4D_annotations" / seq / "action" / "color.json"
        return parse_events(json.loads(path.read_text(encoding="utf-8")))

    def n_frames(self, seq: str) -> int:
        """The annotated length: 1 + the largest objpose frame index (300 on every rigid release sequence). A missing
        file inside that range is a missing frame, not a shorter sequence."""
        idx = [int(p.stem) for p in (self.root / "HOI4D_annotations" / seq / "objpose").glob("*.json")
               if p.stem.isdigit()]
        return max(idx) + 1 if idx else 0

    def objpose_path(self, seq: str, t: int) -> Path:
        """`objpose/<t>.json`, or `objpose/<t:05d>.json` on the sequences that pad (64 of the 1,194 rigid ones)."""
        base = self.root / "HOI4D_annotations" / seq / "objpose"
        plain = base / f"{int(t)}.json"
        return plain if plain.is_file() else base / f"{int(t):05d}.json"

    def objpose(self, seq: str, t: int) -> tuple[np.ndarray, bool]:
        path = self.objpose_path(seq, t)
        if not path.is_file():
            return np.eye(4, dtype=np.float32), False
        return parse_objpose(json.loads(path.read_text(encoding="utf-8")))

    def label_map(self, seq: str, t: int) -> np.ndarray:
        """(H,W,3) uint8 RGB label map of frame `t`."""
        path = self.mask_dir(seq) / f"{int(t):05d}.png"
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def object_mask(self, seq: str, t: int) -> tuple[np.ndarray, np.ndarray]:
        """`(object, other)` (H,W) bool: the category object's pixels, and every OTHER labelled pixel (hands and
        second objects), which can occlude it."""
        lab = self.label_map(seq, t)
        obj = np.all(lab == np.asarray(OBJECT_COLOUR, np.uint8), axis=-1)
        return obj, lab.any(axis=-1) & ~obj

    # --- the record -------------------------------------------------------------------------------------------------

    def load(self, seq: str, hands_npz: Path | None = None,
             pose_shift_px: tuple[float, float] = (0.0, 0.0)) -> ClipRecord:
        """The full-length GT record of `seq` (frame_idx 0..T-1, absolute), like `DexYCBReader.load`.

        `hands_npz`: the sequence's HaWoR-format GT hand file (run/pipeline/build_hoi4d_hands.py); `hand_right` is
        None without it. The object pose needs no hand file. `pose_shift_px`: the sequence's object-pose registration
        (module docstring), applied with `correct_object_pose`; (0, 0) leaves the annotation as shipped.
        """
        n = self.n_frames(seq)
        if n == 0:
            raise FileNotFoundError(f"{seq}: no objpose files")
        frame_idx = np.arange(n, dtype=np.int64)
        poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        confidence = np.zeros(n, dtype=np.float32)
        K = self.intrinsics(seq)
        for t in range(n):
            pose, ok = self.objpose(seq, t)
            poses[t] = correct_object_pose(pose, K, pose_shift_px) if ok else pose
            confidence[t] = 1.0 if ok else 0.0
        record = ClipRecord(
            clip_id=seq, frame_idx=frame_idx, timestamps=frame_idx.astype(np.float64) / FPS,
            intrinsics=K, cam_to_world=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
            object_mesh_path=str(self.mesh_path(seq)), object_pose=poses, object_scale=1.0,
            gravity_cam=np.array([0.0, -1.0, 0.0], dtype=np.float32), confidence=confidence, gate_score=1.0,
            hand_left=None, hand_right=None if hands_npz is None else hand_track(Path(hands_npz), n, "right"))
        record.validate()
        return record


def correct_object_pose(pose: np.ndarray, K: np.ndarray, shift_px: Sequence[float]) -> np.ndarray:
    """`pose` (4,4) with its translation moved laterally by (dx * z / fx, dy * z / fy, 0), z its own depth: the object
    projects (dx, dy) px further in the image (exactly at its origin's depth). Rotation and depth are untouched."""
    out = np.array(pose, dtype=np.float32, copy=True)
    z = float(pose[2, 3])
    out[0, 3] += float(shift_px[0]) * z / float(K[0, 0])
    out[1, 3] += float(shift_px[1]) * z / float(K[1, 1])
    return out


def hand_track(hands_npz: Path, n_frames: int, side: str = "right") -> HandTrack:
    """A `HandTrack` from a HaWoR-format hand file: params = rot (3) + full finger pose (45), camera-frame trans and
    OpenPose-order joints, `valid` from `<side>_valid`. `betas` is the first valid row's (HOI4D's per-frame betas are
    constant within a sequence, checked by the converter).

    Raises:
        ValueError: If the file's length differs from `n_frames` or it has no valid row.
    """
    with np.load(hands_npz, allow_pickle=False) as d:
        rot, pose = np.asarray(d[f"{side}_rot"], np.float32), np.asarray(d[f"{side}_hand_pose"], np.float32)
        trans, joints = np.asarray(d[f"{side}_trans"], np.float32), np.asarray(d[f"{side}_joints"], np.float32)
        betas, valid = np.asarray(d[f"{side}_betas"], np.float32), np.asarray(d[f"{side}_valid"], bool)
    if len(valid) != n_frames:
        raise ValueError(f"{hands_npz}: {len(valid)} rows, the record has {n_frames}")
    if not valid.any():
        raise ValueError(f"{hands_npz}: no valid {side}-hand row")
    return HandTrack(params=np.concatenate([rot, pose], axis=1), betas=betas[int(np.argmax(valid))].copy(),
                     trans=trans, joints=joints, valid=valid)
