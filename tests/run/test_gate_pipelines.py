"""Tests for the gate pipelines (`run/pipeline/build_gate_dataset.py`, `run/pipeline/train_gate.py`): fake decode
and DINOv2, fake 100DOH and DexYCB trees, fabricated token indexes. No weights, no GPU, no real data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fastdo.bench.bench_clips import make_image_sequence_decode_worker
from fastdo.engine import StageFn, make_encode_worker
from fastdo.gate.bundle import load_gate_bundle
from fastdo.gate.dataset import load_gate_dataset, read_gate_index
from run.pipeline import build_gate_dataset as builder
from run.pipeline import train_gate as trainer
from tests.bench.fake_doh import install_file_access_spy, write_fake_doh_bundle
from tests.engine.fakes import FakeDinov2, random_frames, write_decode
from tests.gate.fakes import (
    SERIAL,
    add_doh_hands,
    write_fabricated_gate_index,
    write_gate_dexycb_tree,
)

_REPO = Path(__file__).resolve().parents[2]
_SEQS = [f"20200820-subject-03/20200820_135508/{SERIAL}", f"20200918-subject-06/20200918_110000/{SERIAL}"]
_THRESHOLDS = {"contact_m": 0.02, "free_m": 0.05, "still_m": 0.004, "min_negative_frames": 3}
_FIT_COUNTS = {"fit": (32, 32), "val": (10, 10), "val_seq": (10, 10), "calib": (41, 0), "check": (1, 1)}


def _cfg(out: Path, *overrides: str) -> DictConfig:
    base = [f"paths.root={_REPO}", f"paths.outputs={out / 'outputs'}", "gate_data.dexycb_exclude_files=[]"]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[*base, *overrides])


def _decode_factory(kind: str, sources: Any, stride: int) -> StageFn:
    if kind == "image_sequence":
        return make_image_sequence_decode_worker(sources, stride=stride)

    def video(clip_id: str, artifact_dir: Path) -> None:     # junk-byte clips: 100 rows, as 300 decoded frames at stride 3
        write_decode(artifact_dir, random_frames(100, h=18, w=32), frame_idx=[i * stride for i in range(100)],
                     stride=stride)

    return video


def _encode_factory(crop: str) -> StageFn:
    return make_encode_worker(device="cpu", crop_size=14, model=FakeDinov2(), crop=crop)


def _inputs_cfg(tmp_path: Path, *overrides: str) -> DictConfig:
    return _cfg(tmp_path, f"gate_data.doh_root={tmp_path / '100DOH'}", f"gate_data.dexycb_ids_file={tmp_path / 'ids.json'}",
                f"dexycb.root={tmp_path / 'dexycb'}", f"gate_data.dexycb_exclude_files=['{tmp_path / 'exclude.json'}']",
                "gate_data.n_object_points=256", *(f"gate_data.{k}={v}" for k, v in _THRESHOLDS.items()), *overrides)


def _build_inputs(tmp_path: Path, extra_ids: tuple[str, ...] = ()) -> DictConfig:
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    add_doh_hands(bundle.root, {"boardgame": [3], "drink": [4, 4], "cooking": [3], "misc": [3]})
    write_gate_dexycb_tree(tmp_path / "dexycb", _SEQS)
    (tmp_path / "ids.json").write_text(json.dumps([*_SEQS, *extra_ids]))
    (tmp_path / "exclude.json").write_text(json.dumps({"sets": {"old": {"sequences": ["20200903_104655"]}}}))
    return _inputs_cfg(tmp_path)


def test_build_gate_dataset_keeps_tokens_deletes_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build_inputs(tmp_path)
    seen: list[dict[str, Any]] = []
    real_items = builder.dexycb_gate_items

    def items_spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real_items(*args, **kwargs)

    monkeypatch.setattr(builder, "dexycb_gate_items", items_spy)
    spy = install_file_access_spy(monkeypatch)
    sha, provenance = builder.build_gate_dataset(cfg, _REPO, _decode_factory, _encode_factory)
    calls = list(spy.calls)
    monkeypatch.undo()

    index = tmp_path / "outputs" / "gate" / "index.json"
    items, recorded = read_gate_index(index)
    assert sha == hashlib.sha256(index.read_bytes()).hexdigest() and recorded == provenance
    assert all(Path(item.token_file).is_file() for item in items)
    assert not list((tmp_path / "outputs" / "gate" / "work").rglob("decode.npz"))
    assert provenance["doh_held_out"] == {"<missing>": 1, "test": 1}
    for marker in ("TESTaa-b_c", "TESTnosplit"):
        assert [call for call in calls if marker in call[1]] == []
    assert {k: provenance[k] for k in _THRESHOLDS} == _THRESHOLDS
    assert seen == [_THRESHOLDS] * 2
    assert sorted((i.partition, i.source) for i in items) == [("calib", "100doh"), ("check", "100doh"),
                                                              ("fit", "dexycb_neg"), ("fit", "dexycb_pos"),
                                                              ("val", "dexycb_neg"), ("val", "dexycb_pos")]
    assert provenance["crop"] == "letterbox" and list(provenance["dexycb_exclude_files"]) == [str(tmp_path / "exclude.json")]
    assert load_gate_dataset(index, {"fit", "val", "calib", "check"}).features.shape == (6, 8, 9)

    for name, refused, marker, reason in (                  # subjects 07/08 and the NEW/OLD evaluation sequences
            ("fresh", f"20200928-subject-07/x/{SERIAL}", "subject-07", "not in gate_data.dexycb_allowed_subjects"),
            ("fresh_eval", f"20200903-subject-04/20200903_104655/{SERIAL}", "20200903_104655", "is excluded")):
        fresh = tmp_path / name
        fresh.mkdir()
        cfg = _build_inputs(fresh, extra_ids=(refused,))
        spy = install_file_access_spy(monkeypatch)
        with pytest.raises(ValueError, match=f"{refused}.*{reason}"):
            builder.build_gate_dataset(cfg, _REPO, _decode_factory, _encode_factory)
        calls = list(spy.calls)
        monkeypatch.undo()
        assert [call for call in calls if marker in call[1]] == []
        assert not (fresh / "outputs").exists()


def test_default_gate_exclusions_are_the_evaluation_sets() -> None:
    """The release ships no evaluation set, so the default exclusion list is empty: the user supplies the JSON files
    of their own DexYCB evaluation and calibration sequences (`gate_data.dexycb_exclude_files=[...]`), and an empty
    list excludes nothing (the refusal itself is covered by the build tests above)."""
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"paths.root={_REPO}"])
    assert list(cfg.gate_data.dexycb_exclude_files) == []


def test_build_gate_dataset_refuses_existing_index(tmp_path: Path) -> None:
    cfg = _build_inputs(tmp_path)
    builder.build_gate_dataset(cfg, _REPO, _decode_factory, _encode_factory)
    encoded: list[str] = []

    def counting_encoder(crop: str) -> StageFn:
        worker = _encode_factory(crop)

        def counted(clip_id: str, artifact_dir: Path) -> None:
            encoded.append(clip_id)
            worker(clip_id, artifact_dir)

        return counted

    with pytest.raises(FileExistsError, match="overwrite"):
        builder.build_gate_dataset(cfg, _REPO, _decode_factory, counting_encoder)
    builder.build_gate_dataset(_inputs_cfg(tmp_path, "gate_data.overwrite=true"), _REPO, _decode_factory,
                               counting_encoder)
    assert encoded == []                                   # resumable: every token file already holds its rows
    builder.build_gate_dataset(_inputs_cfg(tmp_path, "gate_data.overwrite=true", "gate_data.stride=2"), _REPO,
                               _decode_factory, counting_encoder)
    assert len(encoded) == 2 and all(clip.startswith("gate_100doh_") for clip in encoded)   # stride-3 tokens are stale,
    assert any("x_Yz-9" in clip for clip in encoded)       # also for the entry without decoded_frames


def test_train_gate_writes_bundle_and_card(tmp_path: Path) -> None:
    index = write_fabricated_gate_index(tmp_path / "gate", _FIT_COUNTS)

    card = trainer.run_train_gate(_cfg(tmp_path, f"gate_train.index_path={index}"), _REPO)

    bundle_path, card_path = tmp_path / "outputs" / "gate" / "gate_bundle.pt", tmp_path / "outputs" / "gate" / "gate_card.json"
    assert card["val_auroc"] >= 0.90 and card["val_seq_auroc"] >= 0.90
    assert bundle_path.is_file() and json.loads(card_path.read_text()) == card
    assert (card["alpha"], card["k"], card["n_calib_pos"]) == (0.05, 2, 41)
    assert card["guaranteed_recall"] == round(1 - 2 / 42, 6)
    assert card["promoted"] is True and card["reasons"] == []
    assert card["bundle_sha256"] == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert set(card["check_scores"]) == {"check_000", "check_001"}
    assert card["crop"] == "letterbox" and card["hand_conf_source"] == "none" and card["rigid_calibrated"] is False
    bundle = load_gate_bundle(bundle_path)
    assert bundle.calibration.k == 2 and bundle.crop == "letterbox" and bundle.config.token_dim == 16


def test_train_gate_never_fits_on_100doh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = write_fabricated_gate_index(tmp_path / "gate", _FIT_COUNTS)
    seen: list[torch.Tensor] = []
    real_train = trainer.train_gate

    def train_spy(model: Any, features: torch.Tensor, labels: Any, cfg: Any) -> Any:
        seen.append(features.clone())
        return real_train(model, features, labels, cfg)

    monkeypatch.setattr(trainer, "train_gate", train_spy)
    trainer.run_train_gate(_cfg(tmp_path, f"gate_train.index_path={index}", "gate_train.steps=5"), _REPO)

    fit = load_gate_dataset(index, {"fit"})
    assert len(seen) == 1 and seen[0].shape[0] == 64 and torch.equal(seen[0], fit.features)
    assert not any(item_id.startswith("gate_100doh_") for item_id in fit.item_ids)
    assert set(fit.sources) == {"dexycb_pos", "dexycb_neg"}


@pytest.mark.parametrize(("calib", "val_delta", "overrides", "reason"), [
    ((8, 0), 5.0, ("gate_train.alpha=0.1",), "k = 0"),
    ((41, 0), -5.0, (), "val_auroc"),
])
def test_train_gate_refuses_without_guarantee_or_separation(tmp_path: Path, calib: tuple[int, int], val_delta: float,
                                                            overrides: tuple[str, ...], reason: str) -> None:
    index = write_fabricated_gate_index(tmp_path / "gate", {**_FIT_COUNTS, "calib": calib}, val_delta=val_delta)

    with pytest.raises(SystemExit) as info:
        trainer.main(_cfg(tmp_path, f"gate_train.index_path={index}", *overrides))

    assert info.value.code == 1
    card = json.loads((tmp_path / "outputs" / "gate" / "gate_card.json").read_text())
    assert card["promoted"] is False and card["bundle_sha256"] is None
    assert any(reason in r for r in card["reasons"]), card["reasons"]
    assert not (tmp_path / "outputs" / "gate" / "gate_bundle.pt").exists()
