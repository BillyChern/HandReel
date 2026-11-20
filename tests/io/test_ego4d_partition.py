"""`fastdo.io.datasets.ego4d_partition`: the participant partition on synthetic lists, and the refusals when the
licensed Ego4D lists or manifest are not in place (they never ship)."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from fastdo import hostpaths
from fastdo.io.datasets import ego4d_partition as P


def _write(path: Path, doc: object) -> dict[str, str]:
    path.write_text(json.dumps(doc))
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _manifest(path: Path, rows: dict[str, str]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_uid", "fb_participant_id", "duration_sec"])
        w.writeheader()
        for uid, pid in rows.items():
            w.writerow({"video_uid": uid, "fb_participant_id": pid, "duration_sec": "60.0"})
    return path


@pytest.fixture
def lists_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "lists"
    d.mkdir()
    monkeypatch.setenv("FASTDO_EGO4D_LISTS", str(d))
    return d


def test_the_lists_do_not_ship_and_a_missing_list_says_how_to_obtain_it(lists_dir: Path) -> None:
    for call, what in ((P.fresh_lists, "fresh lists"), (P.l_candidate_videos, "L candidates"),
                       (P.x_clean_videos, "X_clean list")):
        with pytest.raises(P.Ego4DListMissing) as e:
            call()
        msg = str(e.value)
        assert what in msg and "FASTDO_EGO4D_LISTS" in msg and "Ego4D licence" in msg and str(lists_dir) in msg
        assert isinstance(e.value, FileNotFoundError)
    with pytest.raises(P.Ego4DListMissing, match=P.FRESH_LISTS["sha256"]):
        P.fresh_lists()
    with pytest.raises(P.Ego4DListMissing, match=r"python -m fastdo\.io\.datasets\.ego4d_partition"):
        P.x_clean_videos()
    with pytest.raises(P.Ego4DListMissing):           # x_pool needs the lists first
        P.x_pool()
    for spec in (P.FRESH_LISTS, P.L_CANDIDATES_LIST, P.X_CLEAN_LIST):
        assert not Path(spec["file"]).is_absolute() and "/" not in spec["file"]


def test_a_placeholder_lists_dir_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    if Path(hostpaths.HOST_ROOT).exists():
        pytest.skip(f"{hostpaths.HOST_ROOT} exists on this host")
    for var in ("FASTDO_EGO4D_LISTS", "FASTDO_PRIMARY_ROOT", "FASTDO_EXT_ROOT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(hostpaths.HostPathNotSet, match="FASTDO_EGO4D_LISTS"):
        P.fresh_lists()
    with pytest.raises(hostpaths.HostPathNotSet, match="FASTDO_PRIMARY_ROOT"):
        P.load_participants({"file": str(hostpaths.primary_root() / "data/ego4d/manifest.csv"), "sha256": "0"})
    with pytest.raises(hostpaths.HostPathNotSet, match="FASTDO_PRIMARY_ROOT"):
        P.s2_videos(hostpaths.primary_root() / "data/ego4d_s2/v2/full_scale")


def test_supplied_lists_load_in_order_and_are_verified(lists_dir: Path) -> None:
    fresh = _write(lists_dir / "fresh.json", {"H_eval_fresh": {"videos": ["h1", "h2"]},
                                              "X_reserve": {"videos": ["x2", "x1", "x3"]}})
    assert P.fresh_lists(fresh) == (["h1", "h2"], ["x2", "x1", "x3"])
    cand = _write(lists_dir / "l.json", {"chosen": [{"video_uid": "l1"}, {"video_uid": "l2"}]})
    assert P.l_candidate_videos(cand) == ["l1", "l2"]
    with pytest.raises(ValueError, match="not the recorded"):
        P.fresh_lists({**fresh, "sha256": "0" * 64})
    xc = lists_dir / "xc.json"
    xc.write_text(json.dumps({"videos": ["x1", "x3"], "n": 2}))
    digest = hashlib.sha256(json.dumps(["x1", "x3"]).encode()).hexdigest()
    assert P.x_clean_videos({"file": "xc.json", "videos_sha256": digest}) == ["x1", "x3"]
    with pytest.raises(ValueError, match="video-list sha256"):
        P.x_clean_videos({"file": "xc.json", "videos_sha256": "0" * 64})


def test_partition_refuses_shared_participants_and_derives_x_clean(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.csv", {"h1": "17.0", "h2": "25", "x1": "17", "x2": "31", "x3": "",
                                                     "s1": "40"})
    assert P.derive_x_clean(["h1", "h2"], ["x2", "x1", "x3"], manifest) == ["x2"]    # same id 17; blank dropped
    with pytest.raises(P.PartitionError, match="participant partition refused") as e:
        P.check_partition({"H": ["h1", "h2"], "X": ["x1", "x2"]}, manifest, eval_sets=("H",))
    assert e.value.report["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = P.check_partition({"H": ["h1", "h2"], "X": ["x2"], "S2": ["s1"]}, manifest, eval_sets=("H",))
    assert report.n_videos == {"H": 2, "X": 1, "S2": 1}


def test_regenerated_x_clean_verifies_against_the_recorded_pin(tmp_path: Path, lists_dir: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path / "manifest.csv", {"h1": "1", "x1": "1", "x2": "2", "x3": ""})
    fresh = _write(lists_dir / "fresh.json", {"H_eval_fresh": {"videos": ["h1"]},
                                              "X_reserve": {"videos": ["x1", "x2", "x3"]}})
    record = {"file": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    monkeypatch.setattr(P, "FRESH_LISTS", fresh)
    monkeypatch.setattr(P, "MANIFEST", record)
    monkeypatch.setattr(P.fresh_lists, "__defaults__", (fresh,))            # defaults are bound at definition
    monkeypatch.setattr(P.derive_x_clean, "__defaults__", (record,))
    doc = P._write_x_clean(lists_dir / "xc.json")
    assert doc["videos"] == ["x2"] and doc["source"]["fresh_lists_sha256"] == fresh["sha256"]
    pin = hashlib.sha256(json.dumps(["x2"]).encode()).hexdigest()
    assert P.x_clean_videos({"file": "xc.json", "videos_sha256": pin}) == ["x2"]


def test_the_admitted_blank_video_is_matched_by_digest_not_shipped(tmp_path: Path, monkeypatch) -> None:
    import inspect

    monkeypatch.setenv("FASTDO_EGO4D_LISTS", str(tmp_path))
    uid = "00000000-0000-4000-8000-000000000001"                       # synthetic, not an Ego4D uid
    digest = hashlib.sha256(uid.encode()).hexdigest()
    assert P.admitted_blank_uids(["a", uid, "b"], frozenset({digest})) == {uid}
    assert P.h_blank_admitted_digests() == frozenset()                  # no user file: nothing is admitted
    assert P.admitted_blank_uids(["a", uid, "b"]) == frozenset()
    (tmp_path / P.H_BLANK_ADMITTED_FILE).write_text(json.dumps({"sha256": [digest]}))
    assert P.admitted_blank_uids(["a", uid, "b"]) == {uid}
    (tmp_path / P.H_BLANK_ADMITTED_FILE).write_text(json.dumps({"sha256": ["not-a-digest"]}))
    with pytest.raises(ValueError, match="64 lowercase hex"):
        P.h_blank_admitted_digests()
    uuid = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    assert not re.search(r"[0-9a-f]{64}", inspect.getsource(P).split("FRESH_LISTS")[0].split("MANIFEST = ")[0])
    assert not uuid.search(inspect.getsource(P))                       # no Ego4D video uid in the module
