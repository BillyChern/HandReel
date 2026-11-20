"""Reference sets with an object-level split, and the curriculum (`fastdo.policy.data`)."""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml

from fastdo.io.datasets.dexycb import ycb_object_name
from fastdo.policy.data import (
    CurriculumConfig,
    ReferenceSetConfig,
    build_reference_clips,
    build_reference_set,
    curriculum_reference,
    curriculum_stage,
    object_split,
    scene_object_name,
)
from fastdo.policy.reference import PolicyReference
from tests.policy.env_fixtures import SCENE_XML

N_SERIALS = 2


class _FakeReader:
    """DexYCB layout with `meta.yml` files only; `load` returns the fields `build_reference_set` reads."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.loaded: list[str] = []

    def sequences(self) -> list[str]:
        out = []
        for meta in sorted(self.root.glob("*/*/meta.yml")):
            out += [f"{meta.parent.parent.name}/{meta.parent.name}/{s}" for s in yaml.safe_load(meta.read_text())["serials"]]
        return out

    def load(self, seq: str) -> Any:
        self.loaded.append(seq)
        subject, sequence, _ = seq.split("/")
        meta = yaml.safe_load((self.root / subject / sequence / "meta.yml").read_text())
        name = ycb_object_name(meta["ycb_ids"][meta["ycb_grasp_ind"]])
        return SimpleNamespace(clip_id=seq, object_mesh_path=str(self.root / "models" / name / "textured_simple.obj"))


def _dexycb_root(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """12 sequences over 2 subjects: grasped YCB ids 1..6 twice, each sequence with 2 camera serials and 3 objects."""
    grasped: dict[str, str] = {}
    for i in range(12):
        seq_dir = tmp_path / f"subject-{i % 2}" / f"2020_{i:02d}"
        seq_dir.mkdir(parents=True)
        ids = [(i % 6) + 1, 10, 12]
        (seq_dir / "meta.yml").write_text(yaml.safe_dump({"serials": [f"cam{s}" for s in range(N_SERIALS)],
                                                          "ycb_ids": ids, "ycb_grasp_ind": 0, "num_frames": 3,
                                                          "mano_sides": ["right"]}))
        grasped[f"{seq_dir.parent.name}/{seq_dir.name}"] = ycb_object_name(ids[0])
    return tmp_path, grasped


def _tagging_fn(names: list[str]) -> Any:
    def reference_fn(record: Any) -> PolicyReference:
        names.append(Path(record.object_mesh_path).parent.name)
        T = 2
        return PolicyReference(q_ref=torch.zeros(T, 28), obj_pose=torch.eye(4).repeat(T, 1, 1), log_scale=float(len(names)),
                               contact_plan=torch.zeros(T, 6, dtype=torch.bool), covariance=None, hz=50.0)
    return reference_fn


def test_build_reference_set_object_split(tmp_path: Path) -> None:
    """Six objects, two held out: the training set never loads a held-out object (even when `objects` lists it), the
    held-out set loads only held-out objects, one camera view per sequence, `max_seqs` caps the count, and a missing
    reference function or an empty side of the split raises."""
    root, grasped = _dexycb_root(tmp_path)
    objects = sorted(set(grasped.values()))
    train_objects, held_out = object_split(objects, 2, seed=3)
    assert len(held_out) == 2 and set(train_objects) | set(held_out) == set(objects)
    assert not set(train_objects) & set(held_out) and object_split(objects, 2, seed=3) == (train_objects, held_out)

    for split, allowed in (("train", set(train_objects)), ("held_out", set(held_out))):
        names: list[str] = []
        reader = _FakeReader(root)
        cfg = ReferenceSetConfig(held_out_objects=held_out, split=split, reference_fn=_tagging_fn(names))
        refs = build_reference_set(root, objects, 0, cfg, reader=cast(Any, reader))
        assert len(refs) == len(names) == sum(name in allowed for name in grasped.values())
        assert set(names) == allowed
        assert all(seq.endswith("/cam0") for seq in reader.loaded)                 # one view per sequence
        if split == "train":
            assert not set(names) & set(held_out)
        clips = build_reference_clips(root, objects, 0, dataclasses.replace(cfg, reference_fn=_tagging_fn([])),
                                      reader=cast(Any, _FakeReader(root)))
        assert [c.object_name for c in clips] == names                          # each clip keyed by its object
        assert all(grasped[c.clip_id.rsplit("/", 1)[0]] == c.object_name and c.clip_id.endswith("/cam0") for c in clips)

    names = []
    capped = build_reference_set(root, objects, 3, ReferenceSetConfig(held_out_objects=held_out,
                                                                      reference_fn=_tagging_fn(names)),
                                 reader=cast(Any, _FakeReader(root)))
    assert len(capped) == 3 and not set(names) & set(held_out)
    with pytest.raises(ValueError, match="reference_fn"):
        build_reference_set(root, objects, 0, ReferenceSetConfig(held_out_objects=held_out), reader=cast(Any, reader))
    with pytest.raises(ValueError, match="no held_out object"):
        build_reference_set(root, train_objects, 0, ReferenceSetConfig(held_out_objects=held_out, split="held_out",
                                                                       reference_fn=_tagging_fn([])),
                            reader=cast(Any, reader))
    with pytest.raises(ValueError):
        object_split(objects, len(objects), seed=0)


def test_reference_set_skips_declined_clips_and_filters_the_hand_side(tmp_path: Path,
                                                                     caplog: pytest.LogCaptureFixture) -> None:
    """`hand_side` keeps only sequences whose `meta.yml` hand is that side, before loading; `skip_reason(seq,
    object, side)` declines a sequence before its record loads, logged with the reason; a reference function that
    returns None declines a loaded clip, logged and never kept, and `max_seqs` counts kept clips only. Every clip
    carries its hand side."""
    root = tmp_path
    for i in range(8):
        seq_dir = root / "subject-0" / f"2020_{i:02d}"
        seq_dir.mkdir(parents=True)
        (seq_dir / "meta.yml").write_text(yaml.safe_dump({"serials": ["cam0"], "ycb_ids": [(i % 4) + 1],
                                                          "ycb_grasp_ind": 0, "num_frames": 3,
                                                          "mano_sides": ["right" if i % 2 == 0 else "left"]}))
    objects = [ycb_object_name(i) for i in range(1, 5)]
    reader = _FakeReader(root)
    names: list[str] = []
    tagging = _tagging_fn(names)

    def declining(record: Any) -> PolicyReference | None:
        return None if record.clip_id == "subject-0/2020_02/cam0" else tagging(record)

    def no_hands(seq: str, name: str, side: str | None) -> str | None:
        return "hands not built" if seq.startswith("subject-0/2020_04/") else None

    cfg = ReferenceSetConfig(reference_fn=declining, hand_side="right", skip_reason=no_hands)
    with caplog.at_level(logging.INFO, logger="fastdo.policy.data"):
        clips = build_reference_clips(root, objects, 0, cfg, reader=cast(Any, reader))
    assert reader.loaded == ["subject-0/2020_00/cam0", "subject-0/2020_02/cam0", "subject-0/2020_06/cam0"]
    assert [c.clip_id for c in clips] == ["subject-0/2020_00/cam0", "subject-0/2020_06/cam0"]
    assert names == [objects[0], objects[2]] and all(c.hand_side == "right" for c in clips)
    assert any("hands not built" in m and "2020_04" in m for m in caplog.messages)
    assert any("declined" in m and "2020_02" in m for m in caplog.messages)
    capped = build_reference_clips(root, objects, 2, cfg, reader=cast(Any, _FakeReader(root)))
    assert [c.clip_id for c in capped] == ["subject-0/2020_00/cam0", "subject-0/2020_06/cam0"]   # declined: not counted
    one = build_reference_clips(root, objects, 1, cfg, reader=cast(Any, _FakeReader(root)))
    assert [c.clip_id for c in one] == ["subject-0/2020_00/cam0"]
    left = build_reference_clips(root, objects, 0, ReferenceSetConfig(reference_fn=_tagging_fn([]), hand_side="left"),
                                 reader=cast(Any, _FakeReader(root)))
    assert [c.clip_id.split("/")[1] for c in left] == ["2020_01", "2020_03", "2020_05", "2020_07"]
    assert all(c.hand_side == "left" for c in left)
    with pytest.raises(ValueError, match="hand_side"):
        ReferenceSetConfig(hand_side="both")


def test_curriculum_schedule_and_reference() -> None:
    """Strength ramps 0 -> 1 and the clip length 1 s -> 10 s over the configured updates, held within a stage, then the
    full clip; a stage crops the reference and scales its covariance by strength^2."""
    cfg = CurriculumConfig(strength_updates=100, start_seconds=1.0, end_seconds=10.0, length_updates=200,
                           stage_updates=50)
    assert curriculum_stage(0, cfg) == (0.0, 1.0)
    assert curriculum_stage(49, cfg) == (0.0, 1.0)
    assert curriculum_stage(50, cfg) == (0.5, pytest.approx(3.25))
    assert curriculum_stage(120, cfg) == (1.0, pytest.approx(5.5))
    strength, seconds = curriculum_stage(200, cfg)
    assert strength == 1.0 and math.isinf(seconds)

    T = 135
    cov = torch.eye(7, dtype=torch.float64).repeat(T, 1, 1) * 4e-4
    ref = PolicyReference(q_ref=torch.randn(T, 28), obj_pose=torch.eye(4).repeat(T, 1, 1), log_scale=0.0,
                          contact_plan=torch.zeros(T, 6, dtype=torch.bool), covariance=cov, hz=50.0)
    stage = curriculum_reference(ref, 0.5, 1.0)
    assert stage.num_frames == 51 and stage.q_ref.shape == (51, 28) and stage.contact_plan.shape == (51, 6)
    assert stage.covariance is not None and torch.equal(stage.covariance, cov[:51] * 0.25)
    assert torch.equal(stage.q_ref, ref.q_ref[:51])
    full = curriculum_reference(ref, 1.0, math.inf)
    assert full.num_frames == T and full.covariance is not None and torch.equal(full.covariance, cov)
    assert curriculum_reference(ref, 0.0, 0.001).num_frames == 2
    with pytest.raises(ValueError):
        curriculum_reference(ref, 1.5, 1.0)


def test_scene_object_name_reads_the_object_meshes(tmp_path: Path) -> None:
    """A scene's object is the `<name>` of its `objects/<name>/` meshes (robot meshes ignored): `whisking` for the
    shipped demo; a scene with no object or with two objects raises."""
    def scene(files: list[str]) -> Path:
        path = tmp_path / f"scene_{len(files)}_{abs(hash(tuple(files)))}.xml"
        meshes = "".join(f'<mesh name="m{i}" file="{f}"/>' for i, f in enumerate(files))
        path.write_text(f"<mujoco><asset>{meshes}</asset></mujoco>")
        return path

    assert scene_object_name(scene(["robots/sharpa/a.STL", "objects/mug/visual.obj", "objects/mug/convex/0.obj"])) == "mug"
    with pytest.raises(ValueError, match="one object"):
        scene_object_name(scene(["robots/sharpa/a.STL"]))
    with pytest.raises(ValueError, match="one object"):
        scene_object_name(scene(["objects/mug/visual.obj", "objects/bowl/visual.obj"]))
    if SCENE_XML.exists():
        assert scene_object_name(SCENE_XML) == "whisking"
