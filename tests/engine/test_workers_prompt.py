"""The `prompt` stage worker (S2's P3), with both models faked: artifacts, refusals, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastdo.engine.workers_prompt import (
    PROMPT_FILE,
    PROMPT_META_FILE,
    PromptRefused,
    PromptStageConfig,
    make_prompt_worker,
)
from fastdo.prompt.hands23_client import Hands23Result
from tests.engine.fakes import (
    forbid_full_frames,
    full_load_decode_rows,
    random_frames,
    write_decode,
)

CFG = PromptStageConfig(enabled=True, min_score=0.1, candidate_stride=5)


def _decode(tmp_path: Path, n: int = 12, h: int = 48, w: int = 64) -> Path:
    d = tmp_path / "clip"
    d.mkdir(exist_ok=True)
    np.savez(d / "decode.npz", frames=np.zeros((n, h, w, 3), np.uint8), frame_idx=np.arange(n, dtype=np.int64),
             fps=np.array(10.0), stride=np.array(3, dtype=np.int64))
    return d


def _client(records: dict[int, list[dict]]):
    class Fake:
        def detect(self, decode_npz: Path, rows: list[int]) -> Hands23Result:
            self.rows = rows
            return Hands23Result(rows={r: records.get(r, []) for r in rows}, meta={"weights_sha256": "b" * 64})

    return Fake()


def _hand(held: float = 0.9, box=(10.0, 10.0, 40.0, 30.0)) -> dict:
    return {"hand_score": 0.99, "contact_idx": 3, "side": "right", "interaction_score": 0.95,
            "obj_score": 0.97, "p_held_like": held, "obj_box": list(box)}


def _mask_of(shape=(48, 64), area=(slice(10, 30), slice(10, 40))):
    def masker(image: np.ndarray, box: list[float]) -> np.ndarray:
        m = np.zeros(shape, bool)
        m[area] = True
        return m

    return masker


def test_it_writes_the_schema_the_mask_stage_reads(tmp_path: Path) -> None:
    d = _decode(tmp_path)
    make_prompt_worker(CFG, client=_client({5: [_hand()]}), masker=_mask_of())("clip", d)

    z = np.load(d / PROMPT_FILE)
    assert set(z) == {"keyframe_index", "mask"}
    assert z["keyframe_index"].shape == () and int(z["keyframe_index"]) == 5
    assert z["mask"].dtype == bool and z["mask"].shape == (48, 64)

    meta = json.loads((d / PROMPT_META_FILE).read_text())
    assert meta["choice"]["row"] == 5 and meta["candidate_rows"] == [0, 5, 10, 11]
    assert meta["detector"]["weights_sha256"] == "b" * 64
    assert 0.0 < meta["mask_area_frac"] < 1.0


def test_no_confident_contact_refuses_and_writes_nothing(tmp_path: Path) -> None:
    d = _decode(tmp_path)
    worker = make_prompt_worker(CFG, client=_client({}), masker=_mask_of())
    with pytest.raises(PromptRefused, match="no confident hand-object contact"):
        worker("clip", d)
    assert not (d / PROMPT_FILE).exists() and not (d / PROMPT_META_FILE).exists()


def test_an_implausible_mask_refuses(tmp_path: Path) -> None:
    d = _decode(tmp_path)
    whole_frame = lambda image, box: np.ones((48, 64), bool)          # noqa: E731  (the background, not an object)
    worker = make_prompt_worker(CFG, client=_client({5: [_hand()]}), masker=whole_frame)
    with pytest.raises(PromptRefused, match="implausible mask"):
        worker("clip", d)
    assert not (d / PROMPT_FILE).exists()


def test_the_best_held_object_wins_across_candidate_rows(tmp_path: Path) -> None:
    d = _decode(tmp_path)
    records = {0: [_hand(held=0.2)], 5: [_hand(held=0.95)], 10: [_hand(held=0.5)]}
    make_prompt_worker(CFG, client=_client(records), masker=_mask_of())("clip", d)
    assert int(np.load(d / PROMPT_FILE)["keyframe_index"]) == 5


def test_a_mask_of_the_wrong_shape_is_an_error(tmp_path: Path) -> None:
    d = _decode(tmp_path)
    wrong = lambda image, box: np.ones((10, 10), bool)                # noqa: E731
    with pytest.raises(ValueError, match="frames are"):
        make_prompt_worker(CFG, client=_client({5: [_hand()]}), masker=wrong)("clip", d)


def test_only_the_chosen_row_is_read_and_the_prompt_is_what_the_full_read_wrote(tmp_path: Path) -> None:
    """The SAM 2 image is read alone (the detector reads its own rows), with the same bytes and artifacts."""
    frames = random_frames(12, h=48, w=64, seed=5)
    seen: dict[str, np.ndarray] = {}
    out: dict[str, tuple[bytes, bytes]] = {}
    for arm in ("before", "rows"):
        d = tmp_path / arm
        write_decode(d, frames, stride=3)

        def pixel_mask(image: np.ndarray, box: list[float], arm: str = arm) -> np.ndarray:
            seen[arm] = image.copy()
            mask = np.zeros(image.shape[:2], bool)                     # inside the box, set by the pixels
            mask[10:30, 10:40] = image[10:30, 10:40, 0] > 100
            return mask

        with pytest.MonkeyPatch.context() as mp:
            if arm == "before":
                mp.setattr("fastdo.engine.workers_prompt.read_decode_rows", full_load_decode_rows)
            else:
                forbid_full_frames(mp)
            make_prompt_worker(CFG, client=_client({10: [_hand()]}), masker=pixel_mask)("clip", d)
        meta = json.loads((d / PROMPT_META_FILE).read_text())
        out[arm] = ((d / PROMPT_FILE).read_bytes(), json.dumps(meta, sort_keys=True).encode())

    assert seen["rows"].dtype == np.uint8 and seen["rows"].shape == (48, 64, 3)
    assert seen["rows"].tobytes() == seen["before"].tobytes() == frames[10].tobytes()
    assert out["rows"] == out["before"]


def test_min_score_must_be_set_before_the_stage_can_be_built() -> None:
    with pytest.raises(ValueError, match="min_score is null"):
        make_prompt_worker(PromptStageConfig(enabled=True))
    with pytest.raises(ValueError, match="unknown keys"):
        PromptStageConfig.from_mapping({"enabled": True, "min_score": 0.1, "typo": 1})
    with pytest.raises(ValueError, match="sam2_variant"):
        PromptStageConfig(sam2_variant="enormous")


def test_the_worker_is_deterministic(tmp_path: Path) -> None:
    d = _decode(tmp_path)
    worker = make_prompt_worker(CFG, client=_client({5: [_hand()], 10: [_hand(held=0.9)]}), masker=_mask_of())
    worker("clip", d)
    first = (d / PROMPT_FILE).read_bytes()
    worker("clip", d)
    assert (d / PROMPT_FILE).read_bytes() == first


# --- the refusal convention (S2 Gate A/B) -----------------------------------------------------------------
#
# `run_clip` marks a raising stage "failed" with `str(exc)` and no separate state, so a refusal and a genuine
# error are the SAME manifest state. The gate resolves that by starting the message with a
# stable prefix constant, which `gate_outcome_counts` and the benchmark's S2 status classify on. The prompt
# stage must follow the same convention or S2's Gate B criterion 5 -- "prompt refusal rate at most 50%" --
# cannot be computed from artefacts at all, only scraped from log lines.


def test_both_refusals_start_with_the_stable_prefix(tmp_path: Path) -> None:
    """Matching refusal PROSE would reclassify every refusal as an error the day someone rewords it."""
    from fastdo.engine.workers_prompt import PROMPT_REFUSED_PREFIX

    d = _decode(tmp_path)
    no_contact = make_prompt_worker(CFG, client=_client({}), masker=_mask_of())
    with pytest.raises(PromptRefused) as no_contact_exc:
        no_contact("clip", d)

    whole_frame = lambda image, box: np.ones((48, 64), bool)          # noqa: E731
    bad_mask = make_prompt_worker(CFG, client=_client({5: [_hand()]}), masker=whole_frame)
    with pytest.raises(PromptRefused) as bad_mask_exc:
        bad_mask("clip", d)

    for exc in (no_contact_exc, bad_mask_exc):
        assert str(exc.value).startswith(PROMPT_REFUSED_PREFIX), str(exc.value)
    # the clip id survives the move -- it is what names the window in a failure report
    assert "clip" in str(no_contact_exc.value)


def test_prompt_outcome_counts_tells_a_refusal_from_an_error() -> None:
    """The count Gate B reads. A refusal is a recorded outcome; an exception is not, and conflating them
    would let a broken detector report a 100% refusal rate as if the pipeline had merely declined."""
    from fastdo.engine.manifest import ClipManifest
    from fastdo.engine.workers_prompt import PROMPT_REFUSED_PREFIX, prompt_outcome_counts

    clips = ["built", "refused", "broke", "untouched"]
    manifest = ClipManifest.create(clips, ["ego4d"] * len(clips), [30] * len(clips))
    manifest.mark("built", "prompt", "done", 1.0)
    manifest.mark("refused", "prompt", "failed", 1.0, error=f"{PROMPT_REFUSED_PREFIX}: clip: no confident contact")
    manifest.mark("broke", "prompt", "failed", 1.0, error="CUDA out of memory")

    assert prompt_outcome_counts(manifest) == {"built": 1, "refused": 1, "error": 1, "pending": 1}


def test_detect_batch_defaults_to_the_one_row_loop_and_matches_the_yaml() -> None:
    """Batching is opt-in (dataclass and shipped config both 1). The shipped config prefetches 2 rows (an
    exact data-movement lever, promoted); the dataclass default stays the plain loop (0) for direct library use."""
    import yaml

    assert PromptStageConfig().detect_batch == 1 and PromptStageConfig().detect_prefetch == 0
    conf = yaml.safe_load((Path(__file__).resolve().parents[2] / "run/conf/config.yaml").read_text())
    assert conf["prompt_stage"]["detect_batch"] == 1 and conf["prompt_stage"]["detect_prefetch"] == 2
    parsed = PromptStageConfig.from_mapping(conf["prompt_stage"])
    assert parsed.detect_batch == 1 and parsed.detect_prefetch == 2


def test_the_shipped_config_builds_its_client_with_prefetch_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production path (run/conf -> from_mapping -> make_prompt_worker) reaches Hands23Client(prefetch=2)."""
    import yaml

    import fastdo.engine.workers_prompt as wp

    built: dict = {}

    class Spy:
        def __init__(self, **kwargs: object) -> None:
            built.update(kwargs)

    monkeypatch.setattr(wp, "Hands23Client", Spy)
    conf = yaml.safe_load((Path(__file__).resolve().parents[2] / "run/conf/config.yaml").read_text())["prompt_stage"]
    conf = {**conf, "enabled": True, "min_score": 0.0}
    make_prompt_worker(PromptStageConfig.from_mapping(conf), masker=_mask_of())
    assert built["prefetch"] == 2 and built["batch"] == 1


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_detect_batch_must_be_a_positive_integer(bad: object) -> None:
    with pytest.raises(ValueError, match="detect_batch"):
        PromptStageConfig(enabled=True, min_score=0.1, detect_batch=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1, 0.5, True])
def test_detect_prefetch_must_be_a_non_negative_integer(bad: object) -> None:
    with pytest.raises(ValueError, match="detect_prefetch"):
        PromptStageConfig(enabled=True, min_score=0.1, detect_prefetch=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(("batch", "prefetch"), [(1, 0), (8, 0), (1, 2), (4, 3)])
def test_the_worker_builds_its_client_with_the_configured_batch(batch: int, prefetch: int,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import fastdo.engine.workers_prompt as wp

    built: dict = {}

    class Spy:
        def __init__(self, **kwargs: object) -> None:
            built.update(kwargs)

    monkeypatch.setattr(wp, "Hands23Client", Spy)
    make_prompt_worker(PromptStageConfig(enabled=True, min_score=0.1, detect_batch=batch, detect_prefetch=prefetch),
                       masker=_mask_of())
    assert built["batch"] == batch and built["prefetch"] == prefetch


# --- prompt_stage.record_refusals (opt-in, record only) ----------------------------------------------

REC = PromptStageConfig(enabled=True, min_score=0.1, candidate_stride=5, record_refusals=True)


def test_record_refusals_writes_every_record_and_its_first_failed_gate(tmp_path: Path) -> None:
    from fastdo.engine.workers_prompt import PROMPT_REFUSAL_FILE

    d = _decode(tmp_path)
    records = {0: [_hand(held=0.0)], 5: [{**_hand(), "contact_idx": 1}], 10: [{**_hand(), "obj_score": 0.3}]}
    with pytest.raises(PromptRefused, match="no confident hand-object contact"):
        make_prompt_worker(REC, client=_client(records), masker=_mask_of())("clip", d)
    rec = json.loads((d / PROMPT_REFUSAL_FILE).read_text())
    assert rec["path"] == "no_contact" and rec["candidate_rows"] == [0, 5, 10, 11] and rec["n_records"] == 3
    assert [(r["row"], r["first_failed_gate"]) for r in rec["records"]] == [(0, "score"), (5, "contact_idx"),
                                                                            (10, "obj_score")]
    assert rec["first_failed_counts"]["eligible"] == 0 and rec["detector"]["weights_sha256"] == "b" * 64
    assert not (d / PROMPT_FILE).exists() and not (d / PROMPT_META_FILE).exists()


def test_record_refusals_on_the_mask_path_records_the_choice_and_area(tmp_path: Path) -> None:
    from fastdo.engine.workers_prompt import PROMPT_REFUSAL_FILE

    d = _decode(tmp_path)
    speck = _mask_of(area=(slice(10, 11), slice(10, 11)))              # 1 px of 3072: below the 0.2 % floor
    with pytest.raises(PromptRefused, match="implausible mask"):
        make_prompt_worker(REC, client=_client({5: [_hand()]}), masker=speck)("clip", d)
    rec = json.loads((d / PROMPT_REFUSAL_FILE).read_text())
    assert rec["path"] == "implausible_mask" and rec["choice"]["row"] == 5
    assert rec["mask_area_frac"] == pytest.approx(1 / (48 * 64)) and rec["mask_borders"] == 0
    assert rec["records"][0]["first_failed_gate"] is None


@pytest.mark.parametrize("records", [{5: [_hand()]}, {0: [_hand(held=0.2)], 5: [_hand(held=0.95)]}])
def test_record_refusals_leaves_decisions_and_built_artifacts_byte_identical(tmp_path: Path, records) -> None:
    from fastdo.engine.workers_prompt import PROMPT_REFUSAL_FILE

    out = {}
    for arm, cfg in (("off", CFG), ("on", REC)):
        (tmp_path / arm).mkdir()
        d = _decode(tmp_path / arm)
        (d / PROMPT_REFUSAL_FILE).write_text("stale")          # a rerun of a once-refused window
        make_prompt_worker(cfg, client=_client(records), masker=_mask_of())("clip", d)
        out[arm] = ((d / PROMPT_FILE).read_bytes(), (d / PROMPT_META_FILE).read_bytes())
        assert (d / PROMPT_REFUSAL_FILE).exists() is (arm == "off")   # off never touches it; on removes the stale one
    assert out["off"] == out["on"]


def test_record_refusals_off_writes_nothing_on_refusal(tmp_path: Path) -> None:
    from fastdo.engine.workers_prompt import PROMPT_REFUSAL_FILE

    for masker, records in ((_mask_of(), {}), (lambda image, box: np.ones((48, 64), bool), {5: [_hand()]})):
        (tmp_path / str(len(records))).mkdir()
        d = _decode(tmp_path / str(len(records)))
        with pytest.raises(PromptRefused):
            make_prompt_worker(CFG, client=_client(records), masker=masker)("clip", d)
        assert sorted(p.name for p in d.iterdir()) == ["decode.npz"]
        assert not (d / PROMPT_REFUSAL_FILE).exists()
    with pytest.raises(ValueError, match="record_refusals"):
        PromptStageConfig(record_refusals=1)  # type: ignore[arg-type]
