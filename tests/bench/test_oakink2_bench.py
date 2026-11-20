"""`fastdo.bench.oakink2_bench`: contiguous windows, GT identity per window, the image-window decode, the
typed source domain, and the registered bench entry point (`bench.oakink2_windows`)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fastdo.bench.bench_clips import SOURCES, BenchClip, window_offset
from fastdo.bench.bench_stages import build_stage_fns
from fastdo.bench.oakink2_bench import (
    SOURCE_ALLO,
    SOURCE_EGO,
    STRIDE,
    WINDOW_IMAGES,
    OakInk2Window,
    contiguous_windows,
    load_windows,
    make_image_window_decode_worker,
    oakink2_clips,
    sequence_windows,
    window_identity,
)
from fastdo.engine.source_domain import SOURCE_DOMAINS, SourceDomain, read_domain_record
from fastdo.engine.worker_support import DECODE_FILE
from fastdo.io.datasets.oakink2 import FPS, OakInk2Reader
from run.pipeline import bench_throughput as bench
from tests.io.oakink2_tree import SEQ, SERIALS, mocap_id, write_tree

_REPO = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, *overrides: str) -> DictConfig:
    base = [f"paths.root={_REPO}", f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}",
            "bench.batched.enabled=false", f"whisking.demo_root={tmp_path / 'demo'}"]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[*base, *overrides])


def test_constants_follow_the_s2_operating_rate() -> None:
    assert FPS == 30.0 and WINDOW_IMAGES == 300 and STRIDE == 3


def test_contiguous_windows_keep_only_complete_windows() -> None:
    assert contiguous_windows(1000, 50) == [(0, 50, 350), (1, 350, 650), (2, 650, 950)]
    assert contiguous_windows(349, 50) == []
    assert contiguous_windows(350, 50) == [(0, 50, 350)]
    with pytest.raises(ValueError):
        contiguous_windows(10, -1)


def test_sequence_windows_use_the_seeded_s2_offset_and_are_contiguous(tmp_path: Path) -> None:
    reader = OakInk2Reader(write_tree(tmp_path / "oak", n_images=1000))
    wins = sequence_windows(reader, SEQ)
    offset = round(window_offset(SEQ) * FPS)
    assert wins[0].start == offset and all(w.stop - w.start == 300 for w in wins)
    assert [w.k for w in wins] == list(range(len(wins))) and len(wins) == (1000 - offset) // 300
    for a, b in itertools.pairwise(wins):
        assert a.stop == b.start and b.rows[0] - a.rows[-1] == STRIDE           # one stride across the cut
    assert len(wins[0].rows) == 100


def test_window_identity_from_the_program(tmp_path: Path) -> None:
    reader = OakInk2Reader(write_tree(tmp_path / "oak", n_images=900))
    ids, prog = reader.frame_ids(SEQ), reader.program(SEQ)
    # images 0..299 = mocap 2..1198: only the right-hand grasp (100..1300) covers it
    w0 = OakInk2Window(SEQ, SOURCE_EGO, 0, 0, 300)
    got = window_identity(prog, ids, w0)
    assert got["primary"] == "O01@0001@00001" and got["interaction"] == {"O01@0001@00001": 1198 - 100 + 1}
    assert got["first_objects"] == [] and got["last_objects"] == ["O01@0001@00001"]
    # images 300..599 = mocap 1202..2398: grasp to 1300 (99 ids), place from 1250 (1149 ids)
    w1 = OakInk2Window(SEQ, SOURCE_EGO, 1, 300, 600)
    got = window_identity(prog, ids, w1)
    assert got["primary"] == "C12001" and got["interaction"] == {"C12001": 1149, "O01@0001@00001": 99}
    assert got["first_objects"] == ["O01@0001@00001"] and got["last_objects"] == ["C12001"]
    assert got["first_row_mocap_id"] == mocap_id(300) and got["last_row_mocap_id"] == mocap_id(597)
    # no hand interval: no primary object
    got = window_identity(prog, ids, OakInk2Window(SEQ, SOURCE_EGO, 2, 650, 900))
    assert got["primary"] is None and got["interaction"] == {}


def test_load_windows_validates_entries() -> None:
    entry = {"sequence": SEQ, "source": SOURCE_EGO, "k": 0, "start": 0, "stop": 300}
    (w,) = load_windows([entry])
    assert w.clip_id == "bench_oakink2_ego_scene_09__Z001--seq__feedfacefeedfacefeed__2023-01-01-00-00-00_k000"
    with pytest.raises(ValueError, match="missing"):
        load_windows([{"sequence": SEQ}])
    with pytest.raises(ValueError, match="repeated"):
        load_windows([entry, entry])
    with pytest.raises(ValueError, match="unknown OakInk2 source"):
        load_windows([{**entry, "source": "oakink2"}])
    with pytest.raises(PermissionError):
        load_windows([{**entry, "sequence": "20200813-subject-08"}])


def test_source_domains_follow_the_view(tmp_path: Path) -> None:
    assert SOURCE_EGO in SOURCES and SOURCE_ALLO in SOURCES
    assert SOURCE_DOMAINS[SOURCE_EGO] is SourceDomain.EGOCENTRIC
    assert SOURCE_DOMAINS[SOURCE_ALLO] is SourceDomain.THIRD_PERSON
    reader = OakInk2Reader(write_tree(tmp_path / "oak", n_images=400))
    ego, allo = oakink2_clips(reader, [OakInk2Window(SEQ, SOURCE_EGO, 0, 10, 310),
                                       OakInk2Window(SEQ, SOURCE_ALLO, 0, 10, 310)])
    assert (ego.kind, ego.window, ego.decode_stride) == ("image_window", (10, 310), 3)
    assert ego.domain is SourceDomain.EGOCENTRIC
    assert ego.path.name == SERIALS["egocentric"] and allo.path.name == SERIALS["allocentric_top"]
    assert allo.domain is SourceDomain.THIRD_PERSON
    with pytest.raises(FileNotFoundError, match="window ends"):
        oakink2_clips(reader, [OakInk2Window(SEQ, SOURCE_EGO, 0, 200, 500)])


def test_only_windowed_kinds_carry_a_window() -> None:
    BenchClip("c", SOURCE_EGO, "image_window", Path("/d"), 300, window=(0, 300), decode_stride=3)
    with pytest.raises(ValueError, match="only a video or image_window"):
        BenchClip("c", "dexycb", "image_sequence", Path("/d"), 1, window=(0, 1))


def test_image_window_decode_reads_every_third_image_of_the_range(tmp_path: Path) -> None:
    reader = OakInk2Reader(write_tree(tmp_path / "oak", n_images=400))
    (clip,) = oakink2_clips(reader, [OakInk2Window(SEQ, SOURCE_EGO, 0, 10, 310)])
    worker = make_image_window_decode_worker({clip.clip_id: (clip.path, 10, 310, 3)})
    out = tmp_path / "art"
    out.mkdir()
    worker(clip.clip_id, out)
    z = np.load(out / DECODE_FILE)
    assert z["frame_idx"].tolist() == list(range(10, 310, 3))
    assert z["frames"].shape == (100, 12, 16, 3) and float(z["fps"]) == 10.0 and int(z["stride"]) == 3
    assert z["frames"][:, 0, 0, 1].tolist() == [i % 256 for i in range(10, 310, 3)]      # the image-index-th PNG
    assert int(z["frames"][0, 0, 0, 0]) == list(SERIALS).index("egocentric")
    with pytest.raises(KeyError):
        worker("other", out)


def test_bench_entry_point_builds_declares_and_decodes_oakink2_windows(tmp_path: Path) -> None:
    root = write_tree(tmp_path / "oak", n_images=700)
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps({"windows": [
        {"sequence": SEQ, "source": SOURCE_EGO, "k": 0, "start": 5, "stop": 305},
        {"sequence": SEQ, "source": SOURCE_EGO, "k": 1, "start": 305, "stop": 605}]}))
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=0", f"oakink2.root={root}",
               f"bench.oakink2_windows={windows}", "bench.stages=[decode]")
    clips = bench.build_clips(cfg, _REPO)
    assert [c.window for c in clips] == [(5, 305), (305, 605)] and {c.source for c in clips} == {SOURCE_EGO}
    engine = tmp_path / "engine"
    prompts = bench.prepare_artifacts(cfg, _REPO, clips, engine, write_inputs=True)
    for c in clips:
        assert read_domain_record(engine / c.clip_id) is SourceDomain.EGOCENTRIC
        assert not (engine / c.clip_id / "mask_prompt.npz").exists()             # the prompt stage chooses
        assert not (engine / c.clip_id / "camera.json").exists()
        assert "prompt stage" in prompts[c.clip_id]
    fns, _ = build_stage_fns(cfg, _REPO, clips)
    fns["decode"](clips[1].clip_id, engine / clips[1].clip_id)
    z = np.load(engine / clips[1].clip_id / DECODE_FILE)
    assert z["frame_idx"][0] == 305 and z["frame_idx"][-1] == 602
