"""DexYCB training end to end on the CPU: `run/pipeline/train_policy.py::train_from_config` with
`policy.data.source=dexycb`.

Inputs:
- DexYCB layout: the fake meta-only root of `tests/policy/test_data.py` (12 sequences, grasped YCB ids 1-6 twice), read
  by its injected reader.
- The injected reference function returns the shipped whisking kinematic reference (40 frames) for every clip.
- Every training-side object has a passing scene record. The object the smoke run trains on points at the whisking
  Sharpa scene; the others point at distinct paths that are never loaded.

What is checked (DexYCB training holds exactly one object and one hand side per run; multi-object training is
not supported here):
- the default configuration (every training-side object) is refused before any clip loads or any environment is built,
  with a ValueError naming the constraint;
- on one training object, 3 PPO updates on 8 CPU worlds run on exactly that object's clips, and the checkpoint's run
  metadata records the scene (`scene_xml`) and the number of references (`n_references`).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest

from fastdo.io.datasets.dexycb import YCB_OBJECTS
from fastdo.policy.checkpoint import load_checkpoint
from fastdo.policy.data import object_split, whisking_policy_reference
from fastdo.policy.dexycb_inputs import write_scene_record
from fastdo.policy.reference import PolicyReference
from tests.policy.dexycb_data import WHISKING_SCENE_XML, whisking_scene_missing
from tests.policy.test_data import _dexycb_root, _FakeReader
from tests.run.test_train_policy_smoke import _compose

logger = logging.getLogger(__name__)

SMOKE = ["policy.run=dexycb_smoke", "policy.device=cpu", "policy.env.n_worlds=8", "policy.n_updates=3",
         "policy.checkpoint_every=2", "policy.env.warmup_duration=0.1", "policy.curriculum.stage_updates=2",
         "policy.curriculum.strength_updates=2", "policy.curriculum.length_updates=4",
         "policy.curriculum.start_seconds=0.3", "policy.curriculum.end_seconds=0.6"]


@whisking_scene_missing
def test_train_policy_on_the_dexycb_references_of_one_training_object(tmp_path: Path) -> None:
    from run.pipeline.train_policy import train_from_config

    assert WHISKING_SCENE_XML is not None
    root, grasped = _dexycb_root(tmp_path / "dexycb")
    train_side, _ = object_split(YCB_OBJECTS, 4, 0)
    objects = sorted({name for name in grasped.values() if name in train_side})
    assert len(objects) >= 2, objects
    one = objects[0]
    for name in objects:
        scene_xml = WHISKING_SCENE_XML if name == one else tmp_path / f"{name}_scene.xml"
        write_scene_record(tmp_path / "scenes", {"object": name, "side": "right", "scene_xml": str(scene_xml),
                                                 "pass": True})
    npz = WHISKING_SCENE_XML.parent / "trajectory_kinematic.npz"
    loaded: list[str] = []

    def reference_fn(record: Any) -> PolicyReference:
        loaded.append(str(record.clip_id))
        return whisking_policy_reference(str(npz), 50.0, 40, 1e-3, 1e-2, 1e-2)

    data = ["policy.data.source=dexycb", f"policy.data.dexycb_root={root}",
            f"policy.data.scenes_root={tmp_path / 'scenes'}", f"policy.data.hands_root={tmp_path / 'refs'}"]
    with pytest.raises(ValueError, match="exactly one object and one hand side"):
        train_from_config(_compose(tmp_path, [*data, *SMOKE]), reference_fn, reader=_FakeReader(root))
    assert not loaded                                   # refused before any clip loads
    cfg = _compose(tmp_path, [*data, *SMOKE, f"policy.data.train_objects=[{one}]"])
    summary = train_from_config(cfg, reference_fn, reader=_FakeReader(root))
    expected = sorted(seq for seq, name in grasped.items() if name == one)
    assert sorted(clip.rsplit("/", 1)[0] for clip in loaded) == expected and len(expected) == 2
    rows = [json.loads(line) for line in (Path(summary["run_dir"]) / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == [1, 2, 3]
    assert all(r["policy_loss"] is not None and math.isfinite(r["policy_loss"]) for r in rows)
    meta = load_checkpoint(summary["checkpoints"][-1]).meta
    logger.info("DexYCB smoke run on %s: %d references, scene %s, last policy loss %.4g", one, meta["n_references"],
                meta["scene_xml"], rows[-1]["policy_loss"])
    assert meta["scene_xml"] == str(WHISKING_SCENE_XML) and meta["n_references"] == len(expected)
    assert [Path(c).name for c in summary["checkpoints"]] == ["update_000002.pt", "update_000003.pt"]
    assert meta["update"] == 3 and meta["env_steps"] == 3 * 32 * 8
