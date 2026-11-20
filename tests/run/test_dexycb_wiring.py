"""`policy.data.source=dexycb` wiring in `fastdo.policy.pipeline_config` (no simulator, no data).

The data keys are live: `n_held_out` and `split_seed` pick the held-out objects, `max_seqs` caps the set, `hand_side`
and `exclude_subjects` filter sequences, `train_objects` narrows the training side, and the reference function is built
from `policy.data` (scene and hand roots, engine stride), `policy.env.hz` and the `retarget_stage` block (MANO models,
IK-site wrist target). Each clip resolves to its object's validated scene; the training environment holds one scene,
so a training set spanning several object scenes is refused with the key that narrows it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from fastdo.io.datasets.dexycb import YCB_OBJECTS
from fastdo.policy.data import ReferenceClip, object_split
from fastdo.policy.dexycb_inputs import write_scene_record
from fastdo.policy.dexycb_refs import DexYCBReferenceFn
from fastdo.policy.pipeline_config import (
    check_training_scope,
    clip_scene_xml,
    dexycb_reference_fn,
    load_reference_clips,
    reference_set_config,
    training_scene_xml,
)
from tests.policy.dexycb_data import REPO_ROOT
from tests.policy.test_data import _dexycb_root, _FakeReader, _tagging_fn
from tests.run.test_train_policy_smoke import _compose


def _dexycb_cfg(tmp_path: Path, *overrides: str) -> Any:
    """The composed configuration on the fake DexYCB root `<tmp_path>/dexycb` (written on first use)."""
    root = tmp_path / "dexycb"
    if not root.exists():
        _dexycb_root(root)
    return _compose(tmp_path, ["policy.data.source=dexycb", f"policy.data.dexycb_root={root}",
                               f"policy.data.scenes_root={tmp_path / 'scenes'}",
                               f"policy.data.hands_root={tmp_path / 'refs'}", *overrides])


@pytest.mark.parametrize(("script", "block"), [("build_ycb_scenes.py", "policy_scenes:"),
                                               ("build_dexycb_hands.py", "policy_hands:")])
def test_builder_scripts_compose_their_config_as_scripts(script: str, block: str, tmp_path: Path) -> None:
    """The two builders run as documented scripts (subprocess, cwd outside the repository, no PYTHONPATH) through Hydra's
    `--cfg job` path: imports and composition only, nothing is built."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, str(REPO_ROOT / "run" / "pipeline" / script), f"paths.outputs={tmp_path / 'outputs'}",
               f"hydra.run.dir={tmp_path / 'hydra'}", "--cfg", "job"]
    done = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert block in done.stdout and "exclude_subjects" in done.stdout and not (tmp_path / "outputs").exists()


def test_dexycb_data_keys_are_live_and_build_the_reference_function(tmp_path: Path) -> None:
    cfg = _dexycb_cfg(tmp_path, "policy.data.n_held_out=2", "policy.data.split_seed=5", "policy.data.max_seqs=3",
                      "policy.data.engine_stride=2", "policy.env.hz=25")
    fn = dexycb_reference_fn(cfg)
    assert isinstance(fn, DexYCBReferenceFn)
    assert (str(fn.scenes_root), str(fn.hands_root)) == (str(tmp_path / "scenes"), str(tmp_path / "refs"))
    assert (fn.cfg.hz, fn.cfg.engine_stride, fn.cfg.wrist_target) == (25.0, 2, "ik_site")
    assert fn.cfg.mano_left_pkl == str(cfg.retarget_stage.mano_pkl_path.left)
    assert fn.cfg.mano_right_pkl == str(cfg.retarget_stage.mano_pkl_path.right)
    assert fn.cfg.dexycb_root == str(cfg.policy.data.dexycb_root)
    assert fn.cfg.pos_std == cfg.policy.data.covariance_pos_std
    held = reference_set_config(cfg, "held_out", fn)
    assert held.held_out_objects == object_split(YCB_OBJECTS, 2, 5)[1] and held.split == "held_out"
    assert held.skip_reason == fn.skip_reason and held.hand_side == "right"
    assert held.exclude_subjects == ("subject-07", "subject-08")
    assert reference_set_config(cfg, "train", _tagging_fn([])).skip_reason is None
    clips = load_reference_clips(cfg, "train", _tagging_fn([]), reader=_FakeReader(tmp_path / "dexycb"))
    assert len(clips) == 3 and not {c.object_name for c in clips} & set(held.held_out_objects)


def test_train_objects_narrow_the_training_side(tmp_path: Path) -> None:
    cfg = _dexycb_cfg(tmp_path, "policy.data.n_held_out=10", "policy.data.split_seed=0")
    train_side, held_out = object_split(YCB_OBJECTS, 10, 0)
    everything = load_reference_clips(cfg, "train", _tagging_fn([]), reader=_FakeReader(tmp_path / "dexycb"))
    one = everything[0].object_name
    narrowed = _dexycb_cfg(tmp_path, "policy.data.n_held_out=10", "policy.data.split_seed=0",
                           f"policy.data.train_objects=[{one}]")
    clips = load_reference_clips(narrowed, "train", _tagging_fn([]), reader=_FakeReader(tmp_path / "dexycb"))
    assert clips and {c.object_name for c in clips} == {one} and one in train_side
    held = load_reference_clips(narrowed, "held_out", _tagging_fn([]), reader=_FakeReader(tmp_path / "dexycb"))
    assert held and {c.object_name for c in held} <= set(held_out)        # train_objects never narrows the held-out side
    bad = _dexycb_cfg(tmp_path, "policy.data.n_held_out=10", "policy.data.split_seed=0",
                      f"policy.data.train_objects=[{held_out[0]}]")
    with pytest.raises(ValueError, match="held out"):
        load_reference_clips(bad, "train", _tagging_fn([]), reader=_FakeReader(tmp_path / "dexycb"))


def test_each_clip_runs_in_its_object_scene_and_training_holds_one_scene(tmp_path: Path) -> None:
    whisking = _compose(tmp_path, [])
    demo = ReferenceClip("whisking/0", "whisking", cast(Any, None))
    assert clip_scene_xml(whisking, demo) == training_scene_xml(whisking, [demo]) == str(whisking.policy.scene_xml)
    cfg = _dexycb_cfg(tmp_path)
    for name, passed in (("a_obj", True), ("b_obj", True), ("c_obj", False)):
        write_scene_record(tmp_path / "scenes", {"object": name, "side": "right", "scene_xml": f"/s/{name}.xml",
                                                 "pass": passed})
    a, b, c, d = (ReferenceClip(f"s/{n}/cam0", n, cast(Any, None), "right") for n in ("a_obj", "b_obj", "c_obj", "d_obj"))
    assert clip_scene_xml(cfg, a) == "/s/a_obj.xml" and training_scene_xml(cfg, [a, a]) == "/s/a_obj.xml"
    with pytest.raises(ValueError, match="train_objects"):
        training_scene_xml(cfg, [a, b])
    with pytest.raises(ValueError, match="failed validation"):
        clip_scene_xml(cfg, c)
    with pytest.raises(ValueError, match="no scene"):
        clip_scene_xml(cfg, d)


def test_dexycb_training_holds_exactly_one_object_and_one_hand_side(tmp_path: Path) -> None:
    """`policy.data.source=dexycb` training takes exactly one object (`train_objects`) and one hand
    side (`hand_side`); anything else is a ValueError naming the constraint. The whisking source is one clip already."""
    one = object_split(YCB_OBJECTS, 4, 0)[0][:2]
    check_training_scope(_compose(tmp_path, []))
    check_training_scope(_dexycb_cfg(tmp_path, f"policy.data.train_objects=[{one[0]}]"))
    check_training_scope(_dexycb_cfg(tmp_path, f"policy.data.train_objects=[{one[0]}]", "policy.data.hand_side=left"))
    for overrides in ([], [f"policy.data.train_objects=[{one[0]},{one[1]}]"], ["policy.data.train_objects=[]"],
                      [f"policy.data.train_objects=[{one[0]}]", "policy.data.hand_side=null"]):
        with pytest.raises(ValueError, match="exactly one object and one hand side") as info:
            check_training_scope(_dexycb_cfg(tmp_path, *overrides))
        message = str(info.value)
        assert "multi-object training is not supported yet" in message, overrides
        assert "policy.data.train_objects" in message, overrides
