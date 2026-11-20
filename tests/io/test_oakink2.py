"""`fastdo.io.datasets.oakink2`: the read-only OakInk2 reader, on a fake tree, plus real-data checks when the
subset is on disk (views, egocentric motion, frame lists; numbers only, no pixel is read)."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from fastdo.io.datasets.oakink2 import OakInk2Reader, parse_program, refuse_reserved
from tests.io.oakink2_tree import N_IMAGES, SEQ, SERIALS, mocap_id, write_tree

REAL = Path("/opt/fastdo/handreel/data/oakink2")


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}


def test_reader_lists_views_frames_and_program(tmp_path: Path) -> None:
    reader = OakInk2Reader(write_tree(tmp_path / "oak"))
    assert reader.sequences() == [SEQ]
    assert reader.view_serials(SEQ) == SERIALS
    ids = reader.frame_ids(SEQ, "egocentric")
    assert ids.tolist() == [mocap_id(i) for i in range(N_IMAGES)]           # NUMERIC order of the mocap-id names
    assert reader.view_dir(SEQ, "allocentric_top").name == SERIALS["allocentric_top"]
    with pytest.raises(KeyError, match="not on disk"):
        reader.view_dir(SEQ, "allocentric_left")
    prog = reader.program(SEQ)
    assert [(p.lh, p.rh, p.obj_lh, p.obj_rh) for p in prog] == [
        (None, (100, 1300), (), ("O01@0001@00001",)), ((1250, 2500), None, ("C12001",), ())]


def test_parse_program_refuses_malformed_keys() -> None:
    with pytest.raises(ValueError, match="pair"):
        parse_program({"(1, 2, 3)": {}})
    with pytest.raises(ValueError, match="ends before"):
        parse_program({"((5, 1), None)": {"obj_list_lh": ["a"]}})


def test_mesh_paths_and_names(tmp_path: Path) -> None:
    reader = OakInk2Reader(write_tree(tmp_path / "oak"))
    assert reader.object_mesh_path("C12001").parts[-4:] == ("object_repair", "align_ds", "C12001", "model.obj")
    assert reader.object_mesh_path("O01@0001@00001").name == "scan.ply"
    assert reader.object_name("O01@0001@00001") == "mug" and reader.object_name("X") == "X"
    with pytest.raises(FileNotFoundError):
        reader.object_mesh_path("NOPE")


def test_reserved_names_are_refused(tmp_path: Path) -> None:
    for name in ("x/subject-07/y", "20200813-subject-08", "HOI4D/FINAL_TEST/a"):
        with pytest.raises(PermissionError):
            refuse_reserved(name)
    with pytest.raises(PermissionError):
        OakInk2Reader(tmp_path / "FINAL_TEST")


def test_annotation_unpickler_refuses_foreign_globals(tmp_path: Path) -> None:
    root = write_tree(tmp_path / "oak")
    reader = OakInk2Reader(root)
    anno = reader.annotation(SEQ)
    assert anno["cam_def"][SERIALS["egocentric"]] == "egocentric"
    (root / "anno_preview" / f"{SEQ}.pkl").write_bytes(pickle.dumps({"x": os.getcwd}))     # a callable global
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        reader.annotation(SEQ)


def test_object_pose_is_extrinsic_times_object_transform(tmp_path: Path) -> None:
    reader = OakInk2Reader(write_tree(tmp_path / "oak"))
    anno = reader.annotation(SEQ)
    ids = [mocap_id(i) for i in range(5)]
    pose, valid = reader.object_pose_in_camera(anno, "egocentric", "O01@0001@00001", ids)
    assert valid.tolist() == [True, True, True, False, True]                     # id 3 carries a NaN transform
    np.testing.assert_allclose(pose[3], np.eye(4))
    np.testing.assert_allclose(pose[1][:3, 3], (0.1 + 0.001 * ids[1], 0.0, 0.7))
    K = reader.intrinsics(anno, "egocentric", ids)
    assert K.shape == (5, 3, 3) and K[0, 0, 0] == 600.0


def test_reader_never_writes_under_the_root(tmp_path: Path) -> None:
    root = write_tree(tmp_path / "oak")
    before = _snapshot(root)
    reader = OakInk2Reader(root)
    reader.sequences(), reader.frame_ids(SEQ), reader.program(SEQ), reader.annotation(SEQ)
    reader.object_mesh_path("C12001"), reader.object_name("C12001")
    assert _snapshot(root) == before


# --- the real subset (skipped when absent): metadata and extrinsics only, never a pixel ----------------------------

@pytest.mark.skipif(not (REAL / "data").is_dir(), reason="OakInk2 subset not on this machine")
def test_real_subset_views_frames_and_egocentric_motion() -> None:
    reader = OakInk2Reader(REAL)
    seqs = reader.sequences()
    assert len(seqs) == 89
    seq = seqs[0]
    serials = reader.view_serials(seq)
    assert set(serials) == {"egocentric", "allocentric_top"}
    anno = reader.annotation(seq)
    assert {s: v for s, v in anno["cam_def"].items() if v in serials} == {s: v for v, s in serials.items()}
    ids = reader.frame_ids(seq, "egocentric")
    assert ids.tolist() == [int(x) for x in anno["frame_id_list"]]
    assert np.array_equal(ids, reader.frame_ids(seq, "allocentric_top"))
    t_sd = {v: np.stack([np.asarray(anno["cam_extr"][v][int(f)])[:3, 3] for f in ids]).std(0).max() for v in serials}
    assert t_sd["egocentric"] > 0.01 and t_sd["allocentric_top"] < 1e-6       # head-mounted vs static rig
