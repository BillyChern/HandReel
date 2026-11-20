"""T5 release prep: scripts/fetch_weights.py on fakes only (no network): table, --yes gate, MANO never fetched,
sha256 verification, torch.hub layout, archive safety, and the pins agreeing with the code and uv.lock."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import re
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("fetch_weights", REPO / "scripts" / "fetch_weights.py")
F = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = F                      # dataclasses resolve their module while being created
_SPEC.loader.exec_module(F)


def _no_network() -> F.Transports:
    def boom(*_a, **_k):
        raise AssertionError("a transport was called")
    return F.Transports(url=boom, hf=boom, git_zip=boom)


def _url_writer(payload: bytes, calls: list[str]):
    def fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    return fetch


def test_listing_without_yes_downloads_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    assert F.main([], transports=_no_network()) == 0
    out = capsys.readouterr().out
    assert "Nothing downloaded" in out and F.MANO_LICENSE_URL in out
    for m in F.MODELS:
        assert f"| {m.key} |" in out


def test_json_lists_every_required_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert F.main(["--json"], transports=_no_network()) == 0
    got = capsys.readouterr()
    keys = {m["key"] for m in __import__("json").loads(got.out)}       # stdout is one JSON document, nothing after it
    assert "Nothing downloaded" in got.err
    with pytest.raises(SystemExit):                                    # the manifest mode never downloads
        F.main(["--json", "--yes"], transports=_no_network())
    assert keys == {"sam2", "sam3d", "moge1_sam3d", "dinov2_sam3d", "moge2", "cotracker3", "hawor", "wilor_detector",
                    "hands23", "mano"}


def test_every_model_states_licence_and_pins() -> None:
    for m in F.MODELS:
        assert m.licence and m.licence_url.startswith("https://"), m.key
        if m.kind != "manual":
            assert m.files, m.key
            assert m.revision or m.key in ("hands23", "dinov2_sam3d"), m.key   # unversioned URLs (stated)
        for f in m.files:
            assert f.sha256 is None or re.fullmatch(r"[0-9a-f]{64}", f.sha256), (m.key, f.remote)
    assert by_gated() == {"sam3d"}
    assert {m.key for m in F.MODELS if not m.commercial} >= {"cotracker3", "hawor", "wilor_detector", "hands23",
                                                             "mano", "sam3d"}   # sam3d: kaolin, nvdiffrast (L1)
    for m in F.MODELS:                                  # every downloaded file is verified, except Hands23's code
        if m.kind in ("hf", "url"):
            assert all(f.sha256 for f in m.files), m.key


def by_gated() -> set[str]:
    return {m.key for m in F.MODELS if m.gated}


def test_cotracker_commit_matches_the_code_pin_and_sam2_rev_matches_uv_lock() -> None:
    from fastdo.tracks.cotracker import COTRACKER_HUB_COMMIT

    assert F.COTRACKER_HUB_COMMIT == COTRACKER_HUB_COMMIT == F.by_key()["cotracker3"].git_rev
    lock = (REPO / "uv.lock").read_text()
    m = re.search(r'name = "sam-2"\nversion = "[^"]+"\nsource = \{ git = "[^"#]+#([0-9a-f]{40})" \}', lock)
    assert m and m.group(1) == F.SAM2_GIT_REV


def test_mano_is_never_fetched_even_with_yes(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert F.main(["--yes", "--only", "mano", "--hub-dir", str(tmp_path)], transports=_no_network()) == 0
    assert F.MANO_LICENSE_URL in capsys.readouterr().out
    for m in F.MODELS:               # verify-only files (HaWoR's tracked mano_mean_params.npz) never reach a transport
        if m.key != "mano":
            fetched = [f for f in m.files if f.fetch]
            assert not any("mano" in s.lower() for s in (m.source, *(f.remote for f in fetched),
                                                         *(f.dest for f in fetched))), m.key
    with pytest.raises(RuntimeError, match="MANO is NEVER fetched"):
        F._refuse_mano("https://example.org/MANO_RIGHT.pkl")
    bad = F.Model("x", "x", "url", "https://example.org/", "r", "l", "https://l", False, root="hawor",
                  files=(F.File("mano_v1_2.zip", "m.zip", None),))
    with pytest.raises(RuntimeError, match="refusing"):
        F.fetch(bad, {"hawor": tmp_path}, _no_network())


def test_url_fetch_verifies_sha256_and_deletes_a_mismatch(tmp_path: Path) -> None:
    payload = b"weights"
    good = F.Model("w", "w", "url", "https://example.org/base/", "r1", "MIT", "https://l", True, root="hawor",
                   files=(F.File("a/w.pt", "weights/w.pt", hashlib.sha256(payload).hexdigest()),))
    calls: list[str] = []
    t = F.Transports(url=_url_writer(payload, calls), hf=_no_network().hf, git_zip=_no_network().git_zip)
    assert F.fetch(good, {"hawor": tmp_path}, t)
    assert calls == ["https://example.org/base/a/w.pt"] and (tmp_path / "weights/w.pt").read_bytes() == payload
    assert F.fetch(good, {"hawor": tmp_path}, t) and len(calls) == 1          # present and verified: no refetch
    wrong = F.Model("w", "w", "url", "https://example.org/base/", "r1", "MIT", "https://l", True, root="hawor",
                    files=(F.File("a/v.pt", "weights/v.pt", "0" * 64),))
    assert not F.fetch(wrong, {"hawor": tmp_path}, t)
    assert not (tmp_path / "weights/v.pt").exists()                       # never installed
    assert (tmp_path / "weights/v.pt.sha256-mismatch").read_bytes() == payload   # kept, not deleted


def test_a_present_mismatched_file_is_reported_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/w.pt").write_bytes(b"user's own copy")
    m = F.Model("w", "w", "url", "https://example.org/", "r1", "MIT", "https://l", True, root="hawor",
                files=(F.File("w.pt", "weights/w.pt", "0" * 64),))
    assert not F.fetch(m, {"hawor": tmp_path}, _no_network())             # no transport call at all
    assert (tmp_path / "weights/w.pt").read_bytes() == b"user's own copy"


def test_hf_fetch_passes_the_pinned_revision(tmp_path: Path) -> None:
    p = tmp_path / "snap/model.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"m")
    seen: list[tuple[str, str, str]] = []

    def hf(repo: str, pattern: str, rev: str) -> list[Path]:
        seen.append((repo, pattern, rev))
        return [p]
    m = F.by_key()["moge2"]
    t = F.Transports(url=_no_network().url, hf=hf, git_zip=_no_network().git_zip)
    assert not F.fetch(m, {}, t)                     # b"m" is not MoGe-2: verification fails
    assert seen == [("Ruicheng/moge-2-vitl", "model.pt", m.revision)]
    assert p.read_bytes() == b"m"                    # the HF-cache entry is reported, never deleted


def test_hf_cache_symlink_mismatch_is_left_alone_and_main_exits_nonzero(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    blob = tmp_path / "blobs/abc"
    blob.parent.mkdir()
    blob.write_bytes(b"not sam2")
    link = tmp_path / "snap/sam2.1_hiera_large.pt"
    link.parent.mkdir()
    link.symlink_to(blob)
    t = F.Transports(url=_no_network().url, hf=lambda *_a: [link], git_zip=_no_network().git_zip)
    monkeypatch.delenv("FASTDO_HAWOR_ROOT", raising=False)
    assert F.main(["--yes", "--only", "sam2"], transports=t) == 1
    assert link.is_symlink() and blob.read_bytes() == b"not sam2"


def test_cotracker_lands_where_torch_hub_looks(tmp_path: Path) -> None:
    m = F.by_key()["cotracker3"]
    calls: list[str] = []
    zips: list[tuple[str, str, Path]] = []

    def git_zip(repo: str, rev: str, dest: Path, _url) -> None:
        zips.append((repo, rev, dest))
        (dest / "hubconf.py").parent.mkdir(parents=True)
        (dest / "hubconf.py").write_text("")
    t = F.Transports(url=_url_writer(b"x", calls), hf=_no_network().hf, git_zip=git_zip)
    assert not F.fetch(m, {"torch_hub": tmp_path}, t)          # fake bytes fail the checkpoint's sha256
    # torch.hub.load("facebookresearch/co-tracker:<sha>") reads <hub>/facebookresearch_co-tracker_<sha>
    assert zips == [("facebookresearch/co-tracker", F.COTRACKER_HUB_COMMIT,
                     tmp_path / f"facebookresearch_co-tracker_{F.COTRACKER_HUB_COMMIT}")]
    assert calls == [m.files[0].remote] and "/resolve/" + m.revision + "/" in calls[0]
    assert not (tmp_path / "checkpoints/scaled_online.pth").exists()
    assert (tmp_path / "checkpoints/scaled_online.pth.sha256-mismatch").exists()


@pytest.mark.parametrize("key", ["hawor", "wilor_detector"])
def test_hawor_targets_need_an_explicit_root_with_yes(key: str, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("FASTDO_HAWOR_ROOT", raising=False)
    with pytest.raises(SystemExit) as e:
        F.main(["--yes", "--only", key], transports=_no_network())
    assert e.value.code == 2 and "--hawor-root or FASTDO_HAWOR_ROOT" in capsys.readouterr().err
    assert F.main(["--only", key], transports=_no_network()) == 0         # listing needs no root


def test_nothing_is_ever_written_under_third_party(monkeypatch: pytest.MonkeyPatch) -> None:
    root = REPO / "third_party" / "no_such_dir_for_fetch_weights_test"
    monkeypatch.setenv("FASTDO_HAWOR_ROOT", str(root))
    with pytest.raises(RuntimeError, match="never modifies"):
        F.main(["--yes", "--only", "hawor"], transports=_no_network())
    assert not root.exists()
    for m in F.MODELS:                                  # the code checkout path is guarded too
        if m.code_dest:
            with pytest.raises(RuntimeError, match="never modifies"):
                F.fetch(m, {m.root: REPO / "third_party" / "no_such_dir_for_fetch_weights_test"}, _no_network())
    assert not root.exists()


def _zip_bytes(names: dict[str, str]) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, t in names.items():
            z.writestr(n, t)
    return b.getvalue()


def test_git_zip_extracts_the_tree_and_refuses_unsafe_paths(tmp_path: Path) -> None:
    ok = _zip_bytes({"co-tracker-abc/hubconf.py": "x", "co-tracker-abc/cotracker/__init__.py": ""})
    seen: list[str] = []
    F._git_zip_fetch("o/r", "abc", tmp_path / "o_r_abc", _url_writer(ok, seen))
    assert seen == ["https://github.com/o/r/archive/abc.zip"]
    assert (tmp_path / "o_r_abc/hubconf.py").read_text() == "x" and not (tmp_path / "o_r_abc.zip").exists()
    with pytest.raises(RuntimeError, match="unsafe path"):
        F._git_zip_fetch("o/r", "abc", tmp_path / "bad", _url_writer(_zip_bytes({"../evil.py": ""}), []))


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(SystemExit):
        F.main(["--only", "nope"], transports=_no_network())


def _git(*a: str) -> None:
    import subprocess

    subprocess.run(["git", "-c", "user.name=x", "-c", "user.email=x@example.org", *a], check=True,
                   capture_output=True)


def test_third_party_of_every_worktree_is_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N4: run from a linked worktree, the PRIMARY checkout's third_party/ (and any sibling worktree's) is refused."""
    primary, wt, sib = tmp_path / "primary", tmp_path / "wt", tmp_path / "sib"
    (primary / "third_party").mkdir(parents=True)
    (primary / "third_party/README").write_text("x")
    _git("init", "-q", str(primary))
    _git("-C", str(primary), "add", "-A")
    _git("-C", str(primary), "commit", "-q", "-m", "init")
    _git("-C", str(primary), "worktree", "add", "-q", str(wt))
    _git("-C", str(primary), "worktree", "add", "-q", str(sib))
    dirs = F.third_party_dirs(wt)
    for d in (primary, wt, sib):
        assert (d / "third_party").resolve() in dirs
    monkeypatch.setattr(F, "REPO", wt)
    for d in (primary, sib, wt):
        with pytest.raises(RuntimeError, match="never modifies"):
            F._refuse_third_party(d / "third_party/do-as-i-do/HaWoR/weights/x.pt")
    F._refuse_third_party(tmp_path / "elsewhere/x.pt")                 # outside every third_party/: allowed
    target = primary / "third_party/HaWoR"
    monkeypatch.setenv("FASTDO_HAWOR_ROOT", str(target))
    with pytest.raises(RuntimeError, match="never modifies"):
        F.main(["--yes", "--only", "wilor_detector"], transports=_no_network())
    assert not target.exists()


def test_this_checkouts_third_party_is_in_the_guarded_set() -> None:
    assert (REPO / "third_party").resolve() in F.third_party_dirs(REPO)


@pytest.mark.parametrize("suffix", [".part", ".sha256-mismatch"])
def test_a_leftover_download_temp_is_refused_not_reused(tmp_path: Path, suffix: str) -> None:
    (tmp_path / "weights").mkdir()
    left = tmp_path / f"weights/w.pt{suffix}"
    left.write_bytes(b"old")
    m = F.Model("w", "w", "url", "https://example.org/", "r1", "MIT", "https://l", True, root="hawor",
                files=(F.File("w.pt", "weights/w.pt", "0" * 64),))
    with pytest.raises(RuntimeError, match="already exists"):
        F.fetch(m, {"hawor": tmp_path}, _no_network())                  # refused before any transport call
    assert left.read_bytes() == b"old" and not (tmp_path / "weights/w.pt").exists()


@pytest.mark.parametrize("suffix", [".zip", ".extract"])
def test_a_leftover_archive_temp_is_refused(tmp_path: Path, suffix: str) -> None:
    left = tmp_path / f"o_r_abc{suffix}"
    left.mkdir() if suffix == ".extract" else left.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="already exists"):
        F._git_zip_fetch("o/r", "abc", tmp_path / "o_r_abc", _no_network().url)
    assert left.exists() and not (tmp_path / "o_r_abc").exists()


def test_a_verify_only_file_is_checked_never_downloaded(tmp_path: Path) -> None:
    """HaWoR's mano_mean_params.npz is tracked in the HaWoR checkout: verified there, never fetched."""
    payload = b"tracked"
    m = F.Model("v", "v", "url", "https://example.org/", "r1", "MIT", "https://l", True, root="hawor",
                files=(F.File("", "_DATA/x.npz", hashlib.sha256(payload).hexdigest(), fetch=False),))
    assert not F.fetch(m, {"hawor": tmp_path}, _no_network())             # missing: a failure, no transport call
    (tmp_path / "_DATA").mkdir()
    (tmp_path / "_DATA/x.npz").write_bytes(payload)
    assert F.fetch(m, {"hawor": tmp_path}, _no_network())
    (tmp_path / "_DATA/x.npz").write_bytes(b"changed")
    assert not F.fetch(m, {"hawor": tmp_path}, _no_network())
    hawor = F.by_key()["hawor"]
    assert [f.dest for f in hawor.files if not f.fetch] == ["_DATA/data/mano_mean_params.npz"]


def test_hf_pins_come_from_the_shared_table() -> None:
    from fastdo import model_pins

    hf = {m.source: m.revision for m in F.MODELS if m.kind == "hf"}
    assert hf == model_pins.HF_REVISIONS                       # every pinned HF model is fetched at its pin, and only
    assert F.COTRACKER_HUB_COMMIT == model_pins.COTRACKER_HUB_COMMIT
    assert F.by_key()["dinov2_sam3d"].git_rev == model_pins.DINOV2_HUB_COMMIT


def test_dinov2_code_lands_where_sam3d_torch_hub_looks_and_an_existing_one_is_kept(tmp_path: Path) -> None:
    m = F.by_key()["dinov2_sam3d"]
    zips: list[tuple[str, str, Path]] = []

    def git_zip(repo: str, rev: str, dest: Path, _url) -> None:
        zips.append((repo, rev, dest))
        dest.mkdir(parents=True)
        (dest / "hubconf.py").write_text("")
    t = F.Transports(url=_url_writer(b"x", []), hf=_no_network().hf, git_zip=git_zip)
    assert not F.fetch(m, {"torch_hub": tmp_path}, t)          # fake bytes fail the checkpoint's sha256
    # torch.hub.load("facebookresearch/dinov2", ...) (no ref) reads <hub>/facebookresearch_dinov2_main when present
    assert zips == [("facebookresearch/dinov2", F.DINOV2_HUB_COMMIT, tmp_path / "facebookresearch_dinov2_main")]
    assert (tmp_path / "checkpoints/dinov2_vitl14_reg4_pretrain.pth.sha256-mismatch").exists()
    (tmp_path / "checkpoints/dinov2_vitl14_reg4_pretrain.pth.sha256-mismatch").unlink()
    t2 = F.Transports(url=_url_writer(b"x", []), hf=_no_network().hf, git_zip=_no_network().git_zip)
    F.fetch(m, {"torch_hub": tmp_path}, t2)                    # the present code directory is never replaced
    assert (tmp_path / "facebookresearch_dinov2_main/hubconf.py").is_file()


def test_an_unpacked_code_dir_gets_a_pin_file_and_a_present_one_is_checked(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The unpack writes `CODE_PIN_FILE`; a present directory without one is UNVERIFIED (left alone, not a failure),
    one whose pin names another commit is a MISMATCH that fails the model. Nothing is downloaded."""
    import dataclasses

    from fastdo import model_pins

    payload = b"x"
    base = F.by_key()["dinov2_sam3d"]
    m = dataclasses.replace(base, files=(dataclasses.replace(base.files[0],
                                                             sha256=hashlib.sha256(payload).hexdigest()),))

    def git_zip(repo: str, rev: str, dest: Path, _url) -> None:
        dest.mkdir(parents=True)
        (dest / "hubconf.py").write_text("")
    assert F.fetch(m, {"torch_hub": tmp_path}, F.Transports(url=_url_writer(payload, []), hf=_no_network().hf,
                                                            git_zip=git_zip))
    code = tmp_path / base.code_dest
    assert model_pins.code_pin_status(code, base.code_repo, base.git_rev)[0] == "verified"
    capsys.readouterr()
    assert F.fetch(m, {"torch_hub": tmp_path}, _no_network())                 # present and pinned: verified
    assert "verified" in capsys.readouterr().out
    (code / model_pins.CODE_PIN_FILE).unlink()                                # e.g. a torch.hub-cached `main`
    assert F.fetch(m, {"torch_hub": tmp_path}, _no_network())
    assert "UNVERIFIED" in capsys.readouterr().err and (code / "hubconf.py").is_file()
    model_pins.write_code_pin(code, base.code_repo, "0" * 40)
    assert not F.fetch(m, {"torch_hub": tmp_path}, _no_network())
    assert "MISMATCH" in capsys.readouterr().err and (code / "hubconf.py").is_file()


def test_a_repeated_hands23_setup_verifies_its_checkout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The pinned Hands23 detector is not a torch.hub repository: its root has `demo.py` and `hodetector/` and no
    `hubconf.py`. A second run over the tree the first run unpacked must report it verified, not a mismatch."""
    import dataclasses

    from fastdo import model_pins

    payload = b"x"
    base = F.by_key()["hands23"]
    m = dataclasses.replace(base, files=(dataclasses.replace(base.files[0],
                                                             sha256=hashlib.sha256(payload).hexdigest()),))

    def git_zip(repo: str, rev: str, dest: Path, _url) -> None:
        (dest / "hodetector").mkdir(parents=True)
        (dest / "demo.py").write_text("")
        (dest / "hodetector" / "__init__.py").write_text("")
    assert F.fetch(m, {"hands23": tmp_path}, F.Transports(url=_url_writer(payload, []), hf=_no_network().hf,
                                                          git_zip=git_zip))
    code = tmp_path / base.code_dest
    assert not (code / "hubconf.py").exists()
    capsys.readouterr()
    assert F.fetch(m, {"hands23": tmp_path}, _no_network())                   # the repeat: verified, no download
    got = capsys.readouterr()
    assert "verified" in got.out and "MISMATCH" not in got.err
    assert model_pins.code_pin_status(code, base.code_repo, base.git_rev, base.code_entry)[0] == "verified"
    assert model_pins.code_pin_status(code, base.code_repo, base.git_rev)[0] == "missing"   # torch.hub's entry is absent
    (code / "demo.py").write_text("# changed")
    assert not F.fetch(m, {"hands23": tmp_path}, _no_network())
    assert "MISMATCH" in capsys.readouterr().err


def test_a_mismatch_says_the_pipeline_reads_the_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = F.File("w", "w", "0" * 64)
    (tmp_path / "w").write_bytes(b"x")
    assert not F._verify(tmp_path / "w", f)
    assert "do not run it until this is resolved" in capsys.readouterr().err
    assert not F._verify(tmp_path / "w", f, read_by_pipeline=False)
    assert "not installed" in capsys.readouterr().err


def test_a_gated_key_does_not_stop_the_others(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without SAM 3D access the fetch carries on past the gated key and ends with a summary saying how to
    get it (terms, hf auth login, the pinned revision); the exit status is still 1."""
    class GatedRepoError(OSError):            # huggingface_hub's GatedRepoError is an OSError subclass
        pass

    hf_calls: list[str] = []

    def hf(repo: str, pattern: str, revision: str) -> list[Path]:
        hf_calls.append(repo)
        if repo == "facebook/sam-3d-objects":
            raise GatedRepoError("401 Client Error. Cannot access gated repo")
        return []                             # nothing matched: reported, not raised

    t = F.Transports(url=_url_writer(b"x", []), hf=hf, git_zip=_no_network().git_zip)
    rc = F.main(["--yes", "--only", "sam2,sam3d,moge1_sam3d", "--hub-dir", str(tmp_path)], transports=t)
    out = capsys.readouterr()
    assert rc == 1
    assert hf_calls.count("facebook/sam-3d-objects") == 1 and "Ruicheng/moge-vitl" in hf_calls   # carried on
    assert "sam3d: NOT FETCHED (GatedRepoError: 401" in out.err
    sam3d = F.by_key()["sam3d"]
    assert "Summary:" in out.out and "hf auth login" in out.out and sam3d.revision in out.out
    assert "https://huggingface.co/facebook/sam-3d-objects" in out.out and "MANO (never fetched here)" in out.out
