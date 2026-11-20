"""Hydra `policy` group -> policy objects, shared by `run/pipeline/train_policy.py` and `run/pipeline/eval_policy.py`.

The helpers live in the package, not in a pipeline script, so both scripts run as documented,
`uv run python run/pipeline/<script>.py +policy=default policy.run=<name>`, where `run.pipeline` is not importable.

- `env_config`: `EnvConfig` from `policy.env`, with the reward's shaping discount set to PPO's gamma.
- `ppo_config`: `PpoConfig` from `policy.ppo`.
- `load_reference_clips`: the clips of `policy.data` on one side of the object split.
- `dexycb_reference_fn`, `reference_set_config`: the DexYCB reference function and set configuration. They read
  `policy.data`, `policy.env.hz` and the root `retarget_stage` block, which supplies the MANO models, wrist target and
  solver overrides.
- `clip_scene_xml`, `training_scene_xml`: the robot scene a clip runs in and the one scene of a training
  environment. A clip runs in the demo's `policy.scene_xml`, or in the validated scene of its object and hand side.
- `check_training_scope`: DexYCB training holds exactly one object and one hand side per run.
  The batched environment holds one MjModel; heterogeneous multi-object training needs per-world model variation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fastdo.io.datasets.dexycb import YCB_OBJECTS
from fastdo.policy.data import (
    ReferenceClip,
    ReferenceFn,
    ReferenceSetConfig,
    build_reference_clips,
    object_split,
    scene_object_name,
    whisking_policy_reference,
)
from fastdo.policy.dexycb_inputs import SIDES, read_scene_record, scene_skip_reason
from fastdo.policy.dexycb_refs import DexYCBRefConfig, DexYCBReferenceFn, make_reference_fn
from fastdo.policy.env import EnvConfig
from fastdo.policy.ppo import PpoConfig
from fastdo.policy.rewards import RewardConfig
from fastdo.retarget.kinematic import RetargetConfig

__all__ = ["WHISKING_CLIP_ID", "check_training_scope", "clip_scene_xml", "config_dict", "dexycb_reference_fn",
           "env_config",
           "load_reference_clips", "ppo_config", "reference_set_config", "training_scene_xml"]

WHISKING_CLIP_ID = "whisking/0"


def config_dict(node: Any) -> dict[str, Any]:
    """A configuration node (`DictConfig`, resolved, or a mapping) as a plain dict with string keys."""
    out = OmegaConf.to_container(node, resolve=True) if isinstance(node, DictConfig) else dict(node)
    assert isinstance(out, dict)
    return {str(k): v for k, v in out.items()}


def env_config(node: Any, gamma: float) -> EnvConfig:
    """`EnvConfig` from `policy.env`; the reward's shaping discount is PPO's `gamma` (policy invariance)."""
    fields = config_dict(node)
    reward = dict(fields.pop("reward", None) or {})
    reward["gamma"] = gamma
    return EnvConfig(**fields, reward=RewardConfig(**reward))


def ppo_config(node: Any) -> PpoConfig:
    fields = config_dict(node)
    fields["hidden"] = tuple(int(h) for h in fields["hidden"])
    return PpoConfig(**fields)


def dexycb_reference_fn(cfg: DictConfig) -> DexYCBReferenceFn:
    """`make_reference_fn` over `policy.data.scenes_root` and `hands_root`. It uses the engine stride and synthetic
    posterior of `policy.data`, the policy rate `policy.env.hz`, and from `retarget_stage` the MANO models per side
    (`mano_pkl_path`), `wrist_target` and the `retarget` solver overrides."""
    data, stage = cfg.policy.data, cfg.retarget_stage
    paths = config_dict(stage.get("mano_pkl_path") or {})
    ref_cfg = DexYCBRefConfig(dexycb_root=str(data.dexycb_root), mano_right_pkl=str(paths.get("right") or ""),
                              mano_left_pkl=str(paths.get("left") or ""), hz=float(cfg.policy.env.hz),
                              engine_stride=int(data.engine_stride), wrist_target=str(stage.wrist_target),
                              retarget=RetargetConfig(**config_dict(stage.get("retarget") or {})),
                              pos_std=float(data.covariance_pos_std), rot_std=float(data.covariance_rot_std),
                              log_scale_std=float(data.covariance_log_scale_std))
    return make_reference_fn(str(data.scenes_root), str(data.hands_root), ref_cfg)


def reference_set_config(cfg: DictConfig, split: str, reference_fn: ReferenceFn | None) -> ReferenceSetConfig:
    """The set configuration of `policy.data` on `split`'s side of `object_split(YCB_OBJECTS, n_held_out, split_seed)`.
    A DexYCB reference function also declines sequences before they load (`skip_reason`)."""
    data = cfg.policy.data
    _, held_out = object_split(YCB_OBJECTS, int(data.n_held_out), int(data.split_seed))
    skip = reference_fn.skip_reason if isinstance(reference_fn, DexYCBReferenceFn) else None
    return ReferenceSetConfig(held_out_objects=held_out, split=split, reference_fn=reference_fn,
                              mano_pca=None if data.mano_pca is None else str(data.mano_pca),
                              hand_side=None if data.hand_side is None else str(data.hand_side), skip_reason=skip,
                              exclude_subjects=tuple(str(s) for s in data.exclude_subjects))


def _set_objects(cfg: DictConfig, split: str, held_out: Sequence[str]) -> tuple[str, ...]:
    """The objects a set may hold: `policy.data.train_objects` on the training side when set, else all of them."""
    names = cfg.policy.data.get("train_objects")
    if split != "train" or names is None:
        return YCB_OBJECTS
    objects = tuple(str(o) for o in names)
    unknown, held = sorted(set(objects) - set(YCB_OBJECTS)), sorted(set(objects) & set(held_out))
    if unknown or held:
        raise ValueError(f"policy.data.train_objects: unknown objects {unknown}, objects held out by the split {held}")
    return objects


def check_training_scope(cfg: DictConfig) -> None:
    """`policy.data.source=dexycb` training takes exactly one object (`policy.data.train_objects`, a
    one-element list) and one hand side (`policy.data.hand_side`, left or right) per run. Other sources pass.

    Raises:
        ValueError: Naming the constraint, for any other DexYCB training configuration.
    """
    data = cfg.policy.data
    if str(data.source) != "dexycb":
        return
    names, side = data.get("train_objects"), data.get("hand_side")
    objects = [] if names is None else [str(o) for o in names]
    if len(objects) != 1 or side not in SIDES:
        raise ValueError("policy.data.source=dexycb training supports exactly one object and one hand side per run: "
                         "set policy.data.train_objects to one object and policy.data.hand_side to "
                         f"left or right, got train_objects={None if names is None else objects} and hand_side={side}; "
                         "heterogeneous multi-object training is not supported yet")


def load_reference_clips(cfg: DictConfig, split: str = "train", reference_fn: ReferenceFn | None = None,
                         reader: Any = None) -> list[ReferenceClip]:
    """The clips of `policy.data` on `split`'s side of the object split.

    - `whisking`: the shipped kinematic reference, one clip (`whisking/0`) of the scene's object
      (`scene_object_name`). It is the clip the policy trains on, so it has no held-out side.
    - `dexycb`: `build_reference_clips` over `data.dexycb_root` with `reference_set_config`, up to `data.max_seqs`
      sequences (0: all). The training side holds `data.train_objects` when set. The reference function defaults to
      `dexycb_reference_fn(cfg)`, the reader to `DexYCBReader` with `data.mano_pca`, which never lists
      `data.exclude_subjects`.

    Raises:
        ValueError: For the whisking source's held-out side, bad `train_objects`, or an unknown source.
    """
    data = cfg.policy.data
    if str(data.source) == "whisking":
        if split != "train":
            raise ValueError("policy.data.source=whisking is a single training clip: it has no held-out side")
        ref = whisking_policy_reference(str(data.whisking_npz), float(cfg.policy.env.hz),
                                        None if data.n_frames is None else int(data.n_frames),
                                        float(data.covariance_pos_std), float(data.covariance_rot_std),
                                        float(data.covariance_log_scale_std))
        return [ReferenceClip(WHISKING_CLIP_ID, scene_object_name(str(cfg.policy.scene_xml)), ref)]
    if str(data.source) == "dexycb":
        fn = reference_fn if reference_fn is not None else dexycb_reference_fn(cfg)
        set_cfg = reference_set_config(cfg, split, fn)
        return build_reference_clips(str(data.dexycb_root), _set_objects(cfg, split, set_cfg.held_out_objects),
                                     int(data.max_seqs), set_cfg, reader=reader)
    raise ValueError(f"unknown policy.data.source {data.source!r}")


def clip_scene_xml(cfg: DictConfig, clip: ReferenceClip) -> str:
    """The robot scene `clip` runs in: `policy.scene_xml` for the whisking source, else the validated scene of the
    clip's object and hand side under `policy.data.scenes_root`.

    Raises:
        ValueError: If the clip has no hand side, or its scene is missing or failed validation.
    """
    if str(cfg.policy.data.source) != "dexycb":
        return str(cfg.policy.scene_xml)
    if clip.hand_side not in SIDES:
        raise ValueError(f"clip {clip.clip_id} has no hand side, so no scene")
    side = str(clip.hand_side)
    record = read_scene_record(str(cfg.policy.data.scenes_root), clip.object_name, side)
    reason = scene_skip_reason(record, clip.object_name, side)
    if reason is not None or record is None:
        raise ValueError(f"clip {clip.clip_id}: {reason}")
    return str(record["scene_xml"])


def training_scene_xml(cfg: DictConfig, clips: Sequence[ReferenceClip]) -> str:
    """The one scene of the training environment, shared by every training clip.

    Raises:
        ValueError: If the clips need no scene or several (DexYCB training takes one object and one hand side per
            run, `check_training_scope`), or as `clip_scene_xml`.
    """
    scenes = sorted({clip_scene_xml(cfg, clip) for clip in clips})
    if len(scenes) != 1:
        raise ValueError(f"the training environment holds one scene, the training clips need {len(scenes)} "
                         f"({scenes[:4]}): DexYCB training takes exactly one object and one hand side per run: "
                         "set policy.data.train_objects to one object and policy.data.hand_side to one side; "
                         "multi-object training is not supported yet")
    return scenes[0]
