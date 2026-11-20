"""The `retarget` worker's hand-side selection, per-side robot scenes and skip reasons (artifact files and fakes
only; the scene loader and the solve are replaced, no MuJoCo).

- `side_from_prompt` False (the default): the right-when-present rule, whatever `prompt.json` says (unchanged).
- `side_from_prompt` True: `prompt.json` `choice.side`, else a one-side `geometry.npz` `hand_sides`, else the rule; the
  side's MANO labels and scene are the ones used, and `retarget.json` records `side` and `side_source`.
- A prompt side whose hand has no valid row is SKIPPED with that reason, never retargeted on the other hand.
- A left hand with no `scene_xml_left` fails as an unusable configuration (no fallback marker).
- Skip reasons name the real cause (`retarget_skip_reason`), e.g. no row with a valid hand AND an observed object.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import fastdo.qc.export as export_module
from fastdo.engine import workers_retarget
from fastdo.engine.worker_support import load_npz, save_npz
from fastdo.engine.workers import RetargetStageConfig, make_retarget_worker
from fastdo.engine.workers_retarget import interacting_side
from fastdo.io.clip_record import ClipRecord
from fastdo.qc.export import retarget_skip_reason
from tests.engine.test_workers_qc import _accepted_clip
from tests.qc.records import CapturingKinematic

CFG = RetargetStageConfig(scene_xml="/fake/right_scene.xml", scene_xml_left="/fake/left_scene.xml",
                          mano_right_pkl="/fake/MANO_RIGHT.pkl", mano_left_pkl="/fake/MANO_LEFT.pkl")
PROMPT = dataclasses.replace(CFG, side_from_prompt=True)


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    seen: dict[str, list[Any]] = {"scenes": [], "labels": [], "solves": []}

    def load_scene(path: Any) -> object:
        seen["scenes"].append(str(path))
        return object()

    def labels(path: str) -> torch.Tensor:
        seen["labels"].append(str(path))
        return torch.zeros(778, dtype=torch.int8)

    monkeypatch.setattr(workers_retarget.RobotScene, "from_xml", staticmethod(load_scene))
    monkeypatch.setattr(workers_retarget, "mano_finger_labels", labels)
    spy = CapturingKinematic()
    seen["solves"] = spy.calls
    monkeypatch.setattr(export_module, "retarget_kinematic", spy)
    return seen


def _prompt(clip: Path, side: str | None) -> None:
    (clip / "prompt.json").write_text(json.dumps({"clip_id": "clip0", "choice": {"row": 0, "side": side}}))


def _report(clip: Path) -> dict[str, Any]:
    return json.loads((clip / "retarget.json").read_text())


def test_default_keeps_the_right_when_present_rule_whatever_the_prompt_says(tmp_path: Path, spies: dict) -> None:
    clip = _accepted_clip(tmp_path)
    _prompt(clip, "left")
    make_retarget_worker(CFG)("clip0", clip)
    report = _report(clip)
    assert (report["status"], report["side"], report["side_source"]) == ("done", "right", "rule")
    assert spies["scenes"] == [CFG.scene_xml] and spies["labels"] == [CFG.mano_right_pkl]


@pytest.mark.parametrize("side", ["left", "right"])
def test_prompt_side_selects_the_hand_the_labels_and_the_scene(tmp_path: Path, spies: dict, side: str) -> None:
    clip = _accepted_clip(tmp_path)
    _prompt(clip, side)
    make_retarget_worker(PROMPT)("clip0", clip)
    report = _report(clip)
    assert (report["status"], report["side"], report["side_source"]) == ("done", side, "prompt")
    assert spies["scenes"] == [PROMPT.scene_xml_for(side)] and spies["labels"] == [PROMPT.mano_pkl_path(side)]
    (args, _kwargs), = spies["solves"]
    record = ClipRecord.load(clip / "clip_record.npz")
    verts = torch.from_numpy(load_npz(clip / "hands.npz")[f"{side}_vertices"]).double()
    world = export_module.points_to_world(torch.from_numpy(record.cam_to_world).double(), verts)
    assert (args[3][0] - world[0]).abs().max() <= 1e-9                       # that side's vertices reach the solve


def test_without_a_prompt_the_geometry_anchor_side_is_used(tmp_path: Path, spies: dict) -> None:
    clip = _accepted_clip(tmp_path)
    save_npz(clip / "geometry.npz", hand_sides=np.array(["left"]))
    make_retarget_worker(PROMPT)("clip0", clip)
    assert (_report(clip)["side"], _report(clip)["side_source"]) == ("left", "geometry")
    assert interacting_side(clip) == ("left", "geometry")
    save_npz(clip / "geometry.npz", hand_sides=np.array(["right", "left"]))    # pooled sides: no single anchor
    assert interacting_side(clip) == (None, "rule")
    (clip / "geometry.npz").unlink()
    assert interacting_side(clip) == (None, "rule")
    _prompt(clip, None)                                                      # a prompt without a side falls through
    assert interacting_side(clip) == (None, "rule")
    _prompt(clip, "both")
    with pytest.raises(ValueError, match="choice.side"):
        interacting_side(clip)


def test_a_prompt_side_without_a_valid_hand_row_is_skipped_not_swapped(tmp_path: Path, spies: dict) -> None:
    clip = _accepted_clip(tmp_path)
    record = ClipRecord.load(clip / "clip_record.npz")
    left = dataclasses.replace(record.hand_left, valid=np.zeros_like(record.hand_left.valid))
    dataclasses.replace(record, hand_left=left).save(clip / "clip_record.npz")
    _prompt(clip, "left")
    make_retarget_worker(PROMPT)("clip0", clip)
    report = _report(clip)
    assert (report["status"], report["reason"]) == ("skipped", "no valid left hand row (side from prompt)")
    assert spies["solves"] == [] and spies["scenes"] == []


def test_a_left_hand_without_a_left_scene_is_an_unusable_configuration(tmp_path: Path, spies: dict) -> None:
    clip = _accepted_clip(tmp_path)
    _prompt(clip, "left")
    cfg = dataclasses.replace(PROMPT, scene_xml_left="")
    with pytest.raises(ValueError, match="scene_xml_left"):
        make_retarget_worker(cfg)("clip0", clip)
    report = _report(clip)
    assert (report["status"], report["retarget_fallback_needed"]) == ("failed", False)
    assert spies["solves"] == []


def test_skip_reason_names_an_unobserved_object(tmp_path: Path, spies: dict) -> None:
    clip = _accepted_clip(tmp_path)
    record = ClipRecord.load(clip / "clip_record.npz")
    unobserved = dataclasses.replace(record, confidence=np.where(np.arange(record.n_frames) == 0, 1.0, 0.0)
                                     .astype(np.float32))
    right = dataclasses.replace(record.hand_right, valid=np.arange(record.n_frames) >= 2)   # hand only where unobserved
    dataclasses.replace(unobserved, hand_right=right).save(clip / "clip_record.npz")
    make_retarget_worker(CFG)("clip0", clip)
    report = _report(clip)
    assert report["status"] == "skipped"
    assert report["reason"] == "no row with both a valid right hand and an observed (or bridged) object"
    assert spies["solves"] == []


def test_retarget_skip_reason_order(tmp_path: Path) -> None:
    record = ClipRecord.load(_accepted_clip(tmp_path) / "clip_record.npz")
    zero = dataclasses.replace(record, confidence=np.zeros_like(record.confidence))
    assert retarget_skip_reason(zero, "right") == "mean tracker confidence is not positive"
    assert retarget_skip_reason(record, "right") is None and retarget_skip_reason(record, "left") is None
    no_hand = dataclasses.replace(record, hand_right=dataclasses.replace(
        record.hand_right, valid=np.zeros_like(record.hand_right.valid)))
    assert retarget_skip_reason(no_hand, "right") == "no valid right hand row"
    assert retarget_skip_reason(dataclasses.replace(no_hand, confidence=zero.confidence), "right") == \
        "mean tracker confidence is not positive"                                     # checked first


def test_config_keys_from_the_hydra_block_and_validation() -> None:
    root = OmegaConf.load(Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml")
    cfg = RetargetStageConfig.from_mapping(root.retarget_stage)
    assert cfg.side_from_prompt is False and root.retarget_stage.enabled is False            # opt-in, off by default
    assert cfg.scene_xml_left.endswith("sharpa/left/retarget_placeholder/0/scene.xml")
    assert cfg.scene_xml_for("right") == cfg.scene_xml and cfg.scene_xml_for("left") == cfg.scene_xml_left
    with pytest.raises(ValueError, match="side_from_prompt"):
        RetargetStageConfig(scene_xml="s.xml", mano_right_pkl="r.pkl", side="right", side_from_prompt=True)
    with pytest.raises(ValueError, match="side_from_prompt"):
        RetargetStageConfig(scene_xml="s.xml", mano_right_pkl="r.pkl", side_from_prompt="yes")  # type: ignore[arg-type]
    assert RetargetStageConfig.from_mapping({"scene_xml": "s.xml", "side_from_prompt": True}).side_from_prompt is True
