"""DexYCB dataset reader: converts on-disk DexYCB ground truth into ClipRecords.

Layout and label-format assumptions verified against the dex-ycb-toolkit
source (https://github.com/NVlabs/dex-ycb-toolkit, README.md and
``dex_ycb_toolkit/{dex_ycb,sequence_loader,grasp_eval}.py``, fetched
2026-09-12):

- ``<root>/<subject>/<sequence>/meta.yml``: ``serials`` (camera ids present
  in this sequence), ``num_frames``, ``ycb_ids`` (YCB class ids in this
  sequence), ``ycb_grasp_ind`` (index into ``ycb_ids`` for the grasped
  object), ``mano_sides`` (``["right"]``, ``["left"]``, ...; DexYCB always
  uses only the first entry -- see ``dex_ycb.py`` L226), ``mano_calib``
  (calibration ids parallel to ``mano_sides``), and, when extrinsics were
  captured for this sequence, ``extrinsics`` (the calibration folder
  suffix, e.g. ``"20200709"``).
- ``<root>/<subject>/<sequence>/<serial>/labels_XXXXXX.npz`` per frame:
  ``seg`` ``(H, W)`` uint8; ``pose_y`` ``(num_obj, 3, 4)`` float32, each
  row ``[R | t]`` (3x3 rotation + 3x1 translation, object-to-camera,
  metres) -- NOT a quaternion. (An earlier draft of this reader
  assumed ``pose_y`` was ``(num_obj, 7)`` quaternion-xyzw + translation;
  README.md L158 explicitly documents ``[num_obj, 3, 4]`` = ``[R; t]``,
  and ``grasp_eval.py`` L170-172 does
  ``np.vstack((pose_y[grasp_ind], [[0, 0, 0, 1]]))`` to build a 4x4, which
  only type-checks if the per-object slice is already ``(3, 4)``. This
  reader follows the verified format.); ``pose_m`` ``(1, 51)`` float32:
  ``pose_m[:, 0:3]`` is the MANO global orientation, already axis-angle;
  ``pose_m[:, 3:48]`` is 45 MANO finger-pose **PCA** coefficients (README
  L159: "MANO pose coefficients ... in PCA representation"); ``pose_m[:,
  48:51]`` is translation. All zero when no hand is visible in the frame.
  ``joint_3d`` is ``(1, 21, 3)`` float32 camera-frame joints, ``-1`` when
  absent.
- ``<root>/calibration/intrinsics/<serial>_<W>x<H>.yml``: ``color: {fx,
  fy, ppx, ppy}``. The resolution suffix is matched by glob rather than
  hardcoded, since it need not be ``640x480`` for every capture. Real
  files also carry an ``extrinsics`` field (unused by this reader) tagged
  ``!!python/tuple`` -- see ``load_dexycb_yaml`` below.
- ``<root>/calibration/mano_<calib>/mano.yml``: ``betas`` (10,).
- ``<root>/calibration/extrinsics_<id>/extrinsics.yml``: per-serial (and
  ``apriltag``) transforms, also tagged ``!!python/tuple``. Not parsed by
  this reader (see the ``gravity_cam`` note below), but ``load_dexycb_yaml``
  handles the tag anyway since it is a shared helper.
- ``<root>/models/<ycb_name>/textured_simple.obj``: object mesh.

Real downloaded calibration YAML (``intrinsics/*.yml`` and
``extrinsics_*/extrinsics.yml``) tags some sequence values
``!!python/tuple``, which ``yaml.safe_load`` rejects
(``ConstructorError: could not determine a constructor for the tag
'tag:yaml.org,2002:python/tuple'``). Rather than switching to
``yaml.unsafe_load``/``yaml.Loader`` (which would also accept arbitrary
``!!python/object`` and similar tags -- a real deserialization-safety
concern, see ``rules/security.md``), ``load_dexycb_yaml`` uses a ``SafeLoader``
subclass that additionally constructs *only* ``!!python/tuple`` as a
plain Python ``tuple``. Verified against the actual downloaded dataset
(``grep -rhoE '!![a-zA-Z0-9/_.:]+' data/dexycb --include='*.yml' | sort
-u``) that ``!!python/tuple`` is the only non-standard tag present
anywhere in ``meta.yml``, ``mano.yml``, ``intrinsics/*.yml``, or
``extrinsics_*/extrinsics.yml``.

``ClipRecord.gravity_cam`` is always the default ``[0, -1, 0]`` (float32).
DexYCB's ``calibration/extrinsics_<id>/extrinsics.yml`` does carry a
per-serial transform, but this reader has not verified its convention
(world-to-camera vs. camera-to-world, or which world axis is "up") against
real downloaded data, and an unverified sign/axis error silently baked into
``gravity_cam`` would be worse than an admittedly-generic default. If an
extrinsics file exists for a sequence, ``load()`` emits one
``logger.warning`` and still returns the default.

A DexYCB frame's grasped-object pose is considered missing/invalid when
every entry of its ``[R | t]`` row is ``-1`` or ``NaN``; such frames get
``confidence = 0`` and an identity ``object_pose``.

DexYCB's hand pose (``pose_m[:, 3:48]``) is stored as MANO PCA
coefficients, not the axis-angle values ``HandTrack.params`` requires
(global orient (3) + 15 joints axis-angle (45), see ``clip_record.py``).
Converting requires the subject's MANO PCA basis (``hands_components``,
``hands_mean``), which is part of the MANO model distribution, not of
DexYCB's own calibration files. Pass the path to an ``.npz`` holding that
basis as ``DexYCBReader(root, mano_pca=...)``; ``load()`` raises
``ValueError`` if a hand is present in the sequence and no ``mano_pca`` was
given, rather than silently writing PCA coefficients into an axis-angle
field.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fastdo.io.clip_record import ClipRecord, HandTrack

logger = logging.getLogger(__name__)

__all__ = ["YCB_OBJECTS", "DexYCBReader", "load_dexycb_yaml", "ycb_object_name"]

_RESERVED_TOP_DIRS = frozenset({"calibration", "models"})
_FPS = 30.0  # DexYCB is captured with RealSense D415 cameras at 30 fps.
_DEFAULT_GRAVITY_CAM = np.array([0.0, -1.0, 0.0], dtype=np.float32)
_NUM_MANO_PCA_COMPS = 45

_YCB_CLASSES: dict[int, str] = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}
YCB_OBJECTS: tuple[str, ...] = tuple(_YCB_CLASSES[i] for i in sorted(_YCB_CLASSES))  # the 21 DexYCB objects


class DexYCBReader:
    """Reads DexYCB ground truth from disk and converts it into ClipRecords.

    ``subjects`` (None: all) and ``exclude_subjects`` filter subject directories by name or name suffix (e.g.
    ``"subject-07"``). An excluded subject is never listed (its ``meta.yml`` is not read) and ``load`` refuses it, so
    subjects reserved for calibration and confirmation stay unopened.
    """

    def __init__(self, root: Path, mano_pca: Path | None = None, subjects: Sequence[str] | None = None,
                 exclude_subjects: Sequence[str] = ()) -> None:
        self.root = Path(root)
        self.mano_pca = Path(mano_pca) if mano_pca is not None else None
        self.subjects = None if subjects is None else tuple(subjects)
        self.exclude_subjects = tuple(exclude_subjects)

    def subject_allowed(self, subject: str) -> bool:
        """Whether the subject directory name passes ``subjects`` and ``exclude_subjects`` (exact or suffix match)."""

        def matches(pattern: str) -> bool:
            return subject == pattern or subject.endswith(pattern)

        if any(matches(p) for p in self.exclude_subjects):
            return False
        return self.subjects is None or any(matches(p) for p in self.subjects)

    def sequences(self) -> list[str]:
        """Lists sequence ids as ``"<subject>/<sequence>/<serial>"`` of the allowed subjects."""
        seq_ids: list[str] = []
        for subject_dir in _sorted_subdirs(self.root, exclude=_RESERVED_TOP_DIRS):
            if not self.subject_allowed(subject_dir.name):
                continue
            for seq_dir in _sorted_subdirs(subject_dir):
                meta_path = seq_dir / "meta.yml"
                if not meta_path.is_file():
                    logger.warning("skipping %s: no meta.yml", seq_dir)
                    continue
                meta = load_dexycb_yaml(meta_path)
                serials = meta.get("serials")
                if not serials:
                    logger.warning("skipping %s: meta.yml has no 'serials'", seq_dir)
                    continue
                for serial in serials:
                    seq_ids.append(f"{subject_dir.name}/{seq_dir.name}/{serial}")
        return seq_ids

    def load(self, seq: str) -> ClipRecord:
        """Loads one ``"<subject>/<sequence>/<serial>"`` sequence as a ClipRecord."""
        subject, sequence, serial = seq.split("/")
        if not self.subject_allowed(subject):
            raise ValueError(f"{seq}: subject {subject} is excluded by this reader's subject filter")
        seq_dir = self.root / subject / sequence
        meta = load_dexycb_yaml(seq_dir / "meta.yml")

        num_frames = int(meta["num_frames"])
        ycb_ids = list(meta["ycb_ids"])
        grasp_ind = int(meta["ycb_grasp_ind"])
        object_mesh_path = str(self.root / "models" / ycb_object_name(ycb_ids[grasp_ind]) / "textured_simple.obj")

        intrinsics = _load_intrinsics(self.root, serial)
        gravity_cam = _load_gravity(self.root, meta, seq)

        frame_idx = np.arange(num_frames, dtype=np.int64)
        timestamps = frame_idx.astype(np.float64) / _FPS
        cam_to_world = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
        object_pose = np.zeros((num_frames, 4, 4), dtype=np.float32)
        confidence = np.zeros(num_frames, dtype=np.float32)

        mano_sides = list(meta.get("mano_sides", []))
        mano_calib = list(meta.get("mano_calib", []))
        has_hand = bool(mano_sides)
        mano_params = np.zeros((num_frames, 48), dtype=np.float32)
        mano_trans = np.zeros((num_frames, 3), dtype=np.float32)
        mano_joints = np.zeros((num_frames, 21, 3), dtype=np.float32)
        mano_valid = np.zeros(num_frames, dtype=bool)

        mano_components = np.zeros((_NUM_MANO_PCA_COMPS, _NUM_MANO_PCA_COMPS), dtype=np.float32)
        mano_mean = np.zeros(_NUM_MANO_PCA_COMPS, dtype=np.float32)
        if has_hand:
            if self.mano_pca is None:
                raise ValueError(
                    "DexYCB MANO poses are PCA coefficients; pass mano_pca=<npz> to convert to axis-angle"
                )
            mano_components, mano_mean = _load_mano_pca(self.mano_pca, mano_sides[0])

        cam_dir = seq_dir / serial
        for t in range(num_frames):
            with np.load(cam_dir / f"labels_{t:06d}.npz") as label:
                pose_y = np.asarray(label["pose_y"], dtype=np.float32)
                pose_m = np.asarray(label["pose_m"], dtype=np.float32)
                joint_3d = np.asarray(label["joint_3d"], dtype=np.float32)

            obj_rt = pose_y[grasp_ind]
            if _is_invalid(obj_rt):
                logger.debug("frame %d: grasped object pose missing in %s", t, seq)
                object_pose[t] = np.eye(4, dtype=np.float32)
                confidence[t] = 0.0
            else:
                object_pose[t, :3, :] = obj_rt
                object_pose[t, 3, :] = (0.0, 0.0, 0.0, 1.0)
                confidence[t] = 1.0

            if has_hand:
                mano_params[t, :3] = pose_m[0, :3]
                mano_params[t, 3:] = pose_m[0, 3:48] @ mano_components + mano_mean
                mano_trans[t] = pose_m[0, 48:51]
                mano_joints[t] = joint_3d[0]
                mano_valid[t] = not bool(np.all(pose_m == 0.0))

        hand_left: HandTrack | None = None
        hand_right: HandTrack | None = None
        if has_hand:
            side = mano_sides[0]
            calib_id = mano_calib[0] if mano_calib else None
            betas = _load_betas(self.root, calib_id) if calib_id is not None else np.zeros(10, np.float32)
            track = HandTrack(params=mano_params, betas=betas, trans=mano_trans, joints=mano_joints, valid=mano_valid)
            if side == "right":
                hand_right = track
            elif side == "left":
                hand_left = track
            else:
                raise ValueError(f"unknown mano side {side!r} in {seq_dir / 'meta.yml'}")

        record = ClipRecord(
            clip_id=seq,
            frame_idx=frame_idx,
            timestamps=timestamps,
            intrinsics=intrinsics,
            cam_to_world=cam_to_world,
            object_mesh_path=object_mesh_path,
            object_pose=object_pose,
            object_scale=1.0,
            gravity_cam=gravity_cam,
            confidence=confidence,
            gate_score=1.0,
            hand_left=hand_left,
            hand_right=hand_right,
        )
        record.validate()
        return record


class _DexYCBSafeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` plus a constructor for DexYCB's ``!!python/tuple``
    tag. Deliberately does *not* use ``yaml.Loader``/``yaml.UnsafeLoader``,
    which would also construct arbitrary ``!!python/object`` (and similar)
    tags from untrusted input.
    """


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


_DexYCBSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def _sorted_subdirs(path: Path, exclude: frozenset[str] = frozenset()) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and p.name not in exclude)


def load_dexycb_yaml(path: Path) -> dict[str, Any]:
    """A DexYCB YAML file (`meta.yml`, calibration) through the tuple-aware safe loader (module docstring)."""
    with path.open("r") as f:
        return yaml.load(f, Loader=_DexYCBSafeLoader)


def ycb_object_name(ycb_id: int) -> str:
    """The YCB model name (`models/` directory, e.g. `006_mustard_bottle`) of a DexYCB class id.

    Raises:
        ValueError: For an unknown id.
    """
    try:
        return _YCB_CLASSES[ycb_id]
    except KeyError as e:
        raise ValueError(f"unknown YCB class id {ycb_id}") from e


def _is_invalid(rt: np.ndarray) -> bool:
    """A grasped-object pose row is invalid when every entry is -1 or NaN."""
    return bool(np.all((rt == -1.0) | np.isnan(rt)))


def _load_intrinsics(root: Path, serial: str) -> np.ndarray:
    intr_dir = root / "calibration" / "intrinsics"
    matches = sorted(intr_dir.glob(f"{serial}_*.yml"))
    if not matches:
        raise FileNotFoundError(f"no intrinsics file for serial {serial!r} under {intr_dir}")
    color = load_dexycb_yaml(matches[0])["color"]
    return np.array(
        [[color["fx"], 0.0, color["ppx"]], [0.0, color["fy"], color["ppy"]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _load_betas(root: Path, calib_id: str) -> np.ndarray:
    betas = load_dexycb_yaml(root / "calibration" / f"mano_{calib_id}" / "mano.yml")["betas"]
    return np.asarray(betas, dtype=np.float32)


def _load_mano_pca(path: Path, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads the MANO PCA basis used to convert DexYCB's finger-pose PCA
    coefficients (``pose_m[:, 3:48]``) into axis-angle.

    ``path`` is an ``.npz`` with ``hands_components`` (45, 45) and
    ``hands_mean`` (45,) for the right hand, and optionally
    ``hands_components_left`` / ``hands_mean_left`` for the left hand.
    """
    with np.load(path) as data:
        if side == "left":
            if "hands_components_left" not in data:
                raise ValueError(f"{path}: no left-hand MANO PCA basis (hands_components_left/hands_mean_left)")
            components = np.asarray(data["hands_components_left"], dtype=np.float32)
            mean = np.asarray(data["hands_mean_left"], dtype=np.float32)
        else:
            components = np.asarray(data["hands_components"], dtype=np.float32)
            mean = np.asarray(data["hands_mean"], dtype=np.float32)
    return components, mean


def _load_gravity(root: Path, meta: dict[str, Any], seq: str) -> np.ndarray:
    """Camera-frame up-vector. Always the default ``[0, -1, 0]``.

    See the module docstring: the extrinsics-derived alternative was
    removed because its convention is unverified. Still warn once per
    sequence when an extrinsics file exists, so gaps in gravity accuracy
    are visible rather than silent.
    """
    extr_id = meta.get("extrinsics")
    if extr_id is not None:
        extr_path = root / "calibration" / f"extrinsics_{extr_id}" / "extrinsics.yml"
        if extr_path.is_file():
            logger.warning(
                "%s: extrinsics.yml present at %s but its camera-to-world convention is "
                "unverified; using default gravity_cam=[0,-1,0]",
                seq,
                extr_path,
            )
    return _DEFAULT_GRAVITY_CAM.copy()
