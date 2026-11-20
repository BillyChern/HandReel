"""CPU smoke test of release/example/run_example.py: argument parsing, config loading, the batched engine run with
placeholder workers (`--stub`), the real stage registry built lazily, and the schema of the report and the
timing-ledger row. No model, no network, no GPU."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("run_example", REPO / "release" / "example" / "run_example.py")
E = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = E
_SPEC.loader.exec_module(E)

# The cost ledger's enums (unit rows and timing bases), restated so the example row stays loadable by it.
MEASUREMENT = {"measured", "estimated_by_subtraction", "modelled"}
CARD = {"dedicated", "shared", "unrecorded"}
ATTRIBUTION = {"unit_wall", "allocated_share", "stage_sum", "kernel_model"}
CACHE = {"cold", "warm", "mixed", "not_separated"}
EXECUTION = {"serial", "pipelined"}


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv(E.GPU_MEM_ENV, E.STUB_GPU_MEM_GB)          # QUICKSTART's cap; the stub allocates nothing


def test_the_stub_sets_the_quickstart_cap_and_the_gpu_run_demands_one(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.delenv(E.GPU_MEM_ENV)
    rc, _ = _stub_run(tmp_path)
    assert rc == 0 and os.environ[E.GPU_MEM_ENV] == E.STUB_GPU_MEM_GB
    monkeypatch.delenv(E.GPU_MEM_ENV)
    with pytest.raises(SystemExit, match=E.GPU_MEM_ENV):
        E.main(["--out", str(tmp_path / "gpu"), "--no-fetch"], fetch=_no_fetch)


def _no_fetch(url: str, dest: Path) -> None:
    raise AssertionError(f"the example tried to download {url}")


def _stub_run(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a video: the stub never decodes it")
    out = tmp_path / "out"
    rc = E.main(["--stub", "--video", str(video), "--out", str(out), *extra], fetch=_no_fetch)
    return rc, out


def test_spec_is_a_pinned_redistributable_window() -> None:
    spec = E.load_spec()
    assert spec["dataset"]["licence"] == "CC BY 4.0" and spec["dataset"]["attribution"]
    assert re.fullmatch(r"[0-9a-f]{40}", spec["dataset"]["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
    start, stop = spec["window"]
    assert (stop - start) / spec["video"]["fps"] <= 10.0          # one ~10 s window at most
    x0, y0, x1, y1 = spec["box_xyxy"]
    assert 0 <= x0 < x1 <= spec["video"]["width"] and 0 <= y0 < y1 <= spec["video"]["height"]


def test_the_pinned_clip_carries_the_dataset_credit() -> None:
    """ATTRIBUTION.txt credits the V2D clip (CC BY 4.0 requires it) only when the input is that clip."""
    spec = E.load_spec()
    credited = E.attribution_text(spec, True, spec["file"])
    assert "CC BY 4.0" in credited and "creativecommons.org/licenses/by/4.0" in credited
    assert spec["dataset"]["revision"] in credited
    assert "Modified: reconstructed by this pipeline" in credited and "MODELS.md" in credited
    other = E.attribution_text(spec, False, "README.md")
    assert "README.md" in other and spec["dataset"]["revision"] not in other and "CC BY 4.0" not in other


def test_parse_args_defaults_and_qc_rule() -> None:
    a = E.parse_args(["--out", "o"])
    assert a.stages == E.DEFAULT_STAGES and not a.stub and a.card == "unrecorded"
    assert E.parse_args(["--out", "o", "--qc-rule", "r.json"]).stages[-1] == "qc"
    with pytest.raises(SystemExit):
        E.parse_args(["--out", "o", "--stages", "decode,qc"])         # qc without a rule
    with pytest.raises(SystemExit):
        E.parse_args(["--out", "o", "--stages", "decode,decode"])
    with pytest.raises(SystemExit):
        E.parse_args([])                                               # --out is required


def test_config_composes_with_the_run_overrides(tmp_path: Path) -> None:
    a = E.parse_args(["--out", str(tmp_path), "--set", "solve_stage.device=cpu"])
    cfg = E.compose_config(E.run_overrides(a, E.load_spec(), stride=2))
    assert list(cfg.bench.stages) == list(E.DEFAULT_STAGES)
    assert cfg.bench.stride == 2 and cfg.bench.overwrite is True and cfg.solve_stage.device == "cpu"
    assert Path(cfg.bench.artifact_root) == tmp_path.resolve() / "engine"
    assert cfg.bench.batched.enabled is True                           # the production path ...
    stub = E.compose_config(E.run_overrides(E.parse_args(["--out", str(tmp_path), "--stub"]), E.load_spec(), 2))
    assert stub.bench.batched.enabled is True                          # ... under --stub too (R6)


def test_stub_run_writes_report_and_ledger_row(tmp_path: Path) -> None:
    rc, out = _stub_run(tmp_path)
    assert rc == 0
    report = json.loads((out / E.REPORT_FILE).read_text())
    assert set(report) == set(E.REPORT_KEYS) and report["schema"] == E.REPORT_SCHEMA and report["stub"] is True
    assert report["stage_states"] == {s: "done" for s in E.DEFAULT_STAGES}
    assert report["input"]["sha256_matches_pin"] is False
    assert report["delivered"] is None                                 # qc did not run: undecided, not "delivered"
    assert report["stride"] == 2 and report["execution"]["execution"] == "batched"    # the production runner (R6)
    assert report["execution"]["batch_run"] is not None
    expected = {E.STAGE_ARTIFACTS[s] for s in E.DEFAULT_STAGES} | {"mask_prompt.npz", "source_domain.json",
                                                                   "profile.json"}
    assert expected <= set(report["artifacts"])
    assert (out / E.MANIFEST_FILE).is_file()
    assert report["ledger_row"] == {"file": E.LEDGER_FILE, "line": 1}          # relative to --out (S5)
    assert str(tmp_path) not in json.dumps(report["ledger_row"])
    assert report["pins"][E.COTRACKER_REF_ENV]                                 # S1: recorded, never unset
    assert report["camera"]["stream"] == "ego_cam_a" and report["camera"]["frames_undistorted"] is False
    attribution = (out / E.ATTRIBUTION_FILE).read_text()                       # L5: not the pinned clip, no credit
    assert "clip.mp4" in attribution and "no\ndataset credit is recorded" in attribution
    assert "CC BY 4.0" not in attribution and "NVIDIA" not in attribution and "MODELS.md" in attribution
    rows = [json.loads(line) for line in (out / E.LEDGER_FILE).read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(E.LEDGER_KEYS) and row["schema"] == E.LEDGER_SCHEMA
    assert re.fullmatch(r"u-[0-9a-f]{16}", row["unit_id"]) and row["unit_id"] == E.unit_id(report["clip_id"])
    assert row["assigned"] and row["run"] and row["analysed"] and row["exclusion_reason"] is None
    assert row["delivered"] is None and row["timing_basis"]["delivered_basis"].startswith("qc not run")
    assert row["frames"] == 137.0 and row["timing_basis"]["frames_basis"] == "source"
    assert row["timing_basis"]["frames_decoded"] == 69 and row["timing_basis"]["stride"] == 2
    assert row["seconds"] == pytest.approx(sum(row["stage_seconds"].values())) and row["seconds"] >= 0
    tb = row["timing_basis"]
    assert tb["measurement"] in MEASUREMENT and tb["measurement"] == "modelled"     # a stub is never a timing
    assert tb["card"] in CARD and tb["attribution"] in ATTRIBUTION and tb["cache"] in CACHE
    assert tb["execution"] in EXECUTION and tb["execution"] == "pipelined" and tb["stages"] == list(E.DEFAULT_STAGES)


def test_stub_with_qc_decides_delivered(tmp_path: Path) -> None:
    rc, out = _stub_run(tmp_path, "--stages", ",".join((*E.DEFAULT_STAGES, "qc")))
    assert rc == 0
    report = json.loads((out / E.REPORT_FILE).read_text())
    assert report["delivered"] is True and "clip_record.npz" in report["artifacts"]


def test_the_cotracker_pin_is_the_default_and_an_explicit_ref_wins(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from fastdo.model_pins import COTRACKER_HUB_COMMIT

    monkeypatch.delenv(E.COTRACKER_REF_ENV, raising=False)
    _, out = _stub_run(tmp_path / "a")
    assert json.loads((out / E.REPORT_FILE).read_text())["pins"][E.COTRACKER_REF_ENV] == COTRACKER_HUB_COMMIT
    monkeypatch.setenv(E.COTRACKER_REF_ENV, "main")
    _, out = _stub_run(tmp_path / "b")
    assert json.loads((out / E.REPORT_FILE).read_text())["pins"][E.COTRACKER_REF_ENV] == "main"



def _absolute_strings(value: object, where: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [hit for k, v in value.items() for hit in _absolute_strings(v, f"{where}.{k}")]
    if isinstance(value, list):
        return [hit for i, v in enumerate(value) for hit in _absolute_strings(v, f"{where}[{i}]")]
    return [(where, value)] if isinstance(value, str) and value.startswith("/") else []


def test_the_stub_report_and_row_name_no_absolute_path(tmp_path: Path) -> None:
    """Anonymity: the report and the ledger row are meant to be shared, so no host path; the batch log of the
    execution provenance is named relative to --out, like `ledger_row`."""
    rc, out = _stub_run(tmp_path)
    assert rc == 0
    report = json.loads((out / E.REPORT_FILE).read_text())
    row = json.loads((out / E.LEDGER_FILE).read_text().splitlines()[-1])
    assert _absolute_strings(report) == [] and _absolute_strings(row) == []
    assert str(tmp_path) not in json.dumps(report) and str(tmp_path) not in json.dumps(row)
    log = report["execution"]["batch_log"]
    assert log == report["execution"]["batch_run"]["batch_log"] and (out / log).is_file()


def test_relative_to_out_rewrites_only_paths_under_out(tmp_path: Path) -> None:
    out = tmp_path / "out"
    value = {"a": str(out / "x" / "b.jsonl"), "b": [str(out)], "c": "/elsewhere/f", "d": 3, "e": "rel/f"}
    assert E.relative_to_out(value, out) == {"a": "x/b.jsonl", "b": ["."], "c": "/elsewhere/f", "d": 3, "e": "rel/f"}


def test_the_gpu_run_stops_first_when_cotracker_is_not_fetched(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight: without the fetched CoTracker3 code and checkpoint, the GPU run exits before any download or model
    load with the fetch command, instead of letting torch.hub fetch unverified code at run time."""
    from fastdo.model_pins import COTRACKER_HUB_COMMIT

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setenv(E.COTRACKER_REF_ENV, COTRACKER_HUB_COMMIT)
    with pytest.raises(SystemExit, match=r"fetch_weights\.py --yes --only cotracker3"):
        E.main(["--out", str(tmp_path / "gpu"), "--no-fetch"], fetch=_no_fetch)
    assert not (tmp_path / "gpu").exists()                            # stopped before anything was written
    hub = tmp_path / "torch" / "hub"
    code = hub / f"facebookresearch_co-tracker_{COTRACKER_HUB_COMMIT}"
    spec = importlib.util.spec_from_file_location("fetch_weights_for_example", REPO / "scripts" / "fetch_weights.py")
    fw = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fw)                    # its dataclasses look the module up
    spec.loader.exec_module(fw)
    cotracker = fw.by_key()["cotracker3"]
    assert code == hub / cotracker.code_dest and hub / E.COTRACKER_CHECKPOINT == hub / cotracker.files[0].dest
    code.mkdir(parents=True)
    (code / "hubconf.py").write_text("")
    with pytest.raises(SystemExit, match="scaled_online.pth"):
        E.require_cotracker_hub(hub, COTRACKER_HUB_COMMIT)
    (hub / "checkpoints").mkdir()
    (hub / "checkpoints" / "scaled_online.pth").write_bytes(b"w")
    E.require_cotracker_hub(hub, COTRACKER_HUB_COMMIT)               # both present: passes
    with pytest.raises(SystemExit, match="facebookresearch_co-tracker_main"):
        E.require_cotracker_hub(hub, "main")                          # an explicit ref needs its own directory

def test_rerun_appends_a_row_and_replaces_the_clip_dir(tmp_path: Path) -> None:
    _stub_run(tmp_path)
    rc, out = _stub_run(tmp_path, "--stages", "decode,mask")
    assert rc == 0
    assert len((out / E.LEDGER_FILE).read_text().splitlines()) == 2
    report = json.loads((out / E.REPORT_FILE).read_text())
    assert "solve.npz" not in report["artifacts"] and report["delivered"] is None
    assert report["ledger_row"] == {"file": E.LEDGER_FILE, "line": 2}
    row = json.loads((out / E.LEDGER_FILE).read_text().splitlines()[-1])
    assert row["analysed"] is False and row["exclusion_reason"]


def test_failed_stage_is_recorded_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = E.stub_stage_fns

    def failing(stages):
        fns = real(stages)

        def boom(_clip_id: str, _dir: Path) -> None:
            raise RuntimeError("stub shape failure")
        fns["shape"] = boom
        return fns
    monkeypatch.setattr(E, "stub_stage_fns", failing)
    rc, out = _stub_run(tmp_path)
    assert rc == 1
    report = json.loads((out / E.REPORT_FILE).read_text())
    assert report["stage_states"]["shape"] == "failed" and report["stage_states"]["solve"] == "pending"
    assert "stub shape failure" in report["errors"]["shape"]
    row = json.loads((out / E.LEDGER_FILE).read_text())
    assert row["analysed"] is False and row["exclusion_reason"].startswith("stage shape")


def test_missing_input_is_fetched_from_the_pinned_revision_and_checked(tmp_path: Path) -> None:
    spec = E.load_spec()
    calls: list[str] = []

    def fake(url: str, dest: Path) -> None:
        calls.append(url)
        assert dest.name.endswith(".part")                    # never the final name before the digest check
        dest.write_bytes(b"wrong bytes")
    a = E.parse_args(["--out", str(tmp_path / "o"), "--data-dir", str(tmp_path / "data")])
    with pytest.raises(ValueError, match="sha256"):
        E.resolve_input(a, spec, fetch=fake)
    url = (f"https://huggingface.co/datasets/{spec['dataset']['hf_repo']}/resolve/"
           f"{spec['dataset']['revision']}/{spec['file']}")
    assert calls == [url]
    video = tmp_path / "data" / spec["file"]
    assert not video.exists() and not video.with_name(video.name + ".part").exists()       # S3
    assert video.with_name(video.name + ".sha256-mismatch").read_bytes() == b"wrong bytes"
    with pytest.raises(RuntimeError, match="already exists"):                             # a leftover is refused
        E.resolve_input(a, spec, fetch=fake)
    assert len(calls) == 1
    a = E.parse_args(["--out", str(tmp_path / "o"), "--data-dir", str(tmp_path / "none"), "--no-fetch"])
    with pytest.raises(FileNotFoundError):
        E.resolve_input(a, spec, fetch=_no_fetch)


def test_a_missing_clip_without_fetching_exits_with_one_line(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:                  # --stub, no --video, no clip in --data-dir
        E.main(["--stub", "--out", str(tmp_path / "o"), "--data-dir", str(tmp_path / "none")], fetch=_no_fetch)
    msg = str(exc.value.code)
    assert "\n" not in msg and "--stub never downloads it" in msg and "pass --video <file>" in msg
    assert not (tmp_path / "o").exists()                    # nothing written before the input is resolved
    a = E.parse_args(["--out", str(tmp_path / "o"), "--data-dir", str(tmp_path / "none"), "--no-fetch"])
    with pytest.raises(FileNotFoundError, match="--no-fetch is set"):
        E.resolve_input(a, spec=E.load_spec(), fetch=_no_fetch)


def test_stub_mask_is_the_box() -> None:
    m = E.stub_box_mask((10, 20), (2, 3, 5, 7))
    assert m.sum() == 3 * 4 and m[3, 2] and not m[7, 5]


def test_a_verified_download_is_renamed_into_place(tmp_path: Path) -> None:
    payload = b"the clip"
    dest = tmp_path / "d" / "clip.mp4"
    E.download_verified("https://example.org/clip.mp4", dest, hashlib.sha256(payload).hexdigest(),
                        lambda _url, part: part.write_bytes(payload))
    assert dest.read_bytes() == payload
    assert sorted(p.name for p in dest.parent.iterdir()) == ["clip.mp4"]


def test_the_clip_download_has_a_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def fake_urlopen(url: str, timeout: float | None = None):
        seen["timeout"] = timeout
        return _Resp(b"x")
    monkeypatch.setattr(E.urllib.request, "urlopen", fake_urlopen)
    E._url_fetch("https://example.org/x", tmp_path / "x.part")
    assert seen["timeout"] == E.URL_TIMEOUT_S and E.URL_TIMEOUT_S > 0


def test_stage_artifacts_are_the_engines_file_names() -> None:
    from fastdo.engine.worker_support import DECODE_FILE, SOLVE_FILE
    from fastdo.engine.workers_geometry import GEOMETRY_FILE
    from fastdo.engine.workers_hands import HANDS_FILE
    from fastdo.engine.workers_qc import CLIP_RECORD_FILE
    from fastdo.engine.workers_shape import MESH_FILE
    from fastdo.engine.workers_video import MASK_FILE, TRACKS_FILE
    from fastdo.geometry.intrinsics import INTRINSICS_FILE

    assert E.STAGE_ARTIFACTS == {"decode": DECODE_FILE, "mask": MASK_FILE, "shape": MESH_FILE,
                                 "intrinsics": INTRINSICS_FILE, "hands": HANDS_FILE, "geometry": GEOMETRY_FILE,
                                 "tracks": TRACKS_FILE, "solve": SOLVE_FILE, "qc": CLIP_RECORD_FILE}


def test_the_real_stage_registry_builds_lazily_on_cpu(tmp_path: Path) -> None:
    """R6: the non-stub path's registry. `build_stage_fns` loads no model (every worker loads on its first clip), so
    it runs on a CPU-only host: every default stage must be registered, and the default batched config must validate
    against an engine config of the example's stages."""
    from fastdo.bench.bench_batched import batched_engine_config
    from fastdo.bench.bench_clips import BenchClip, operating_stride
    from fastdo.bench.bench_stages import build_stage_fns
    from fastdo.engine import EngineConfig
    from fastdo.engine.source_domain import parse_domain

    spec = E.load_spec()
    stride = operating_stride(float(spec["video"]["fps"]))
    a = E.parse_args(["--out", str(tmp_path / "out")])
    cfg = E.compose_config(E.run_overrides(a, spec, stride))
    start, stop = spec["window"]
    clip = BenchClip(spec["clip_id"], spec["source"], "video", tmp_path / "clip.mp4", stop - start,
                     window=(start, stop), decode_stride=stride, domain=parse_domain(spec["domain"]))
    fns, _labels = build_stage_fns(cfg, E.REPO, [clip])
    assert set(E.DEFAULT_STAGES) <= set(fns), sorted(set(E.DEFAULT_STAGES) - set(fns))
    assert all(callable(fns[s]) for s in E.DEFAULT_STAGES)
    assert "qc" not in fns                                              # no rule: the example leaves qc out
    engine = EngineConfig(artifact_root=tmp_path / "out" / "engine", stages=E.DEFAULT_STAGES, system="fastdo_example",
                          manifest_path=tmp_path / "out" / E.MANIFEST_FILE)
    batched = batched_engine_config(cfg.bench, engine)
    assert batched.engine is engine


def test_box_mask_frees_the_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    import weakref

    refs: list[weakref.ref] = []

    class _Predictor:
        def set_image(self, frame) -> None:
            self.hw = frame.shape[:2]

        def predict(self, box, multimask_output: bool):
            m = np.zeros((1, *self.hw), np.float32)
            x0, y0, x1, y1 = (int(v) for v in box)
            m[0, y0:y1, x0:x1] = 1.0
            return m, None, None

    def loader(_device: str) -> _Predictor:
        p = _Predictor()
        refs.append(weakref.ref(p))
        return p
    mask = E.box_mask(np.zeros((8, 10, 3), np.uint8), (2, 1, 5, 4), "cpu", loader=loader)
    assert mask.sum() == 9 and refs[0]() is None                       # nothing holds the predictor afterwards


def test_read_frame_keeps_every_source_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    class _Done:
        stdout = bytes(2 * 3 * 3)

    def fake_run(cmd, **_kw):
        seen.append(cmd)
        return _Done()
    monkeypatch.setattr(E.subprocess, "run", fake_run)
    assert E.read_frame(Path("v.mp4"), 4, 3, 2).shape == (2, 3, 3)
    cmd = seen[0]
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough" and "-vsync" not in cmd


def test_commit_falls_back_to_the_zips_version_file(tmp_path: Path) -> None:
    assert E.git_commit(tmp_path).startswith("unknown")
    (tmp_path / E.VERSION_FILE).write_text("$Format:%H %cs$\n")     # an unfilled git archive placeholder
    assert E.git_commit(tmp_path).startswith("unknown")
    (tmp_path / E.VERSION_FILE).write_text("tree-sha256 " + "a" * 64 + "\n")
    assert E.git_commit(tmp_path) == "release archive: tree-sha256 " + "a" * 64


REFERENCE_DIR = "release/reference/v2d_t3_ep011_cam_a/"     # the one shipped run output (its README says why)


def test_no_generated_output_is_tracked() -> None:
    """S5: outputs and the downloaded clip are generated, never shipped, except the reference output of the example
    clip in `REFERENCE_DIR`; no video is shipped anywhere."""
    import subprocess

    if not (E.REPO / ".git").exists():
        pytest.skip("not a git checkout (an archive): no tracked-file list to read")
    tracked = subprocess.run(["git", "-C", str(E.REPO), "ls-files", "release", "outputs", "data"], check=True,
                             capture_output=True, text=True).stdout.split()
    generated = (E.REPORT_FILE, E.LEDGER_FILE, E.ATTRIBUTION_FILE, ".mp4", ".npz", ".parquet")
    assert not [t for t in tracked if t.endswith(generated) and not t.startswith(REFERENCE_DIR)]
    assert not [t for t in tracked if t.endswith((".mp4", ".parquet", "decode.npz", "geometry.npz", "mask.npz"))]
    on_disk = sorted(p.relative_to(E.REPO).as_posix() for p in (E.REPO / REFERENCE_DIR).iterdir() if p.is_file())
    assert [f for f in on_disk if f not in tracked] == []            # e.g. an npz left out by .gitignore
    assert {REFERENCE_DIR + n for n in ("solve.npz", "hands.npz", "intrinsics.npz", "mesh.obj")} <= set(tracked)


PROPRIETARY_VERSION = ("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.15  Tue Mar  5 22:23:56 UTC 2024\n"
                       "GCC version:  gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04)\n")
OPEN_VERSION = ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.91.07  Release Build  (builder@host)  "
                "Wed Jul 29 03:01:16 UTC 2026\nGCC version:  gcc version 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)\n")


@pytest.mark.parametrize("text,expected", [(PROPRIETARY_VERSION, "550.54.15"), (OPEN_VERSION, "595.91.07"),
                                           ("garbage\n", "unrecorded"), ("NVRM version: no number\n", "unrecorded")])
def test_the_driver_version_parses_from_both_kernel_module_lines(text: str, expected: str) -> None:
    assert E.parse_driver_version(text) == expected


def _fake_proc(tmp_path: Path, version: str = OPEN_VERSION) -> Path:
    proc = tmp_path / "nvidia"
    for bus, model, uuid in (("0000:07:00.0", "Card A", "GPU-aaaa1111"), ("0000:09:00.0", "Card B", "GPU-bbbb2222"),
                             ("0000:0b:00.0", "Card C", "GPU-cccc3333")):
        (proc / "gpus" / bus).mkdir(parents=True)
        (proc / "gpus" / bus / "information").write_text(
            f"Model: \t\t {model}\nIRQ:   \t\t 22\nGPU UUID: \t {uuid}\nBus Location: \t {bus}\n")
    (proc / "version").write_text(version)
    return proc


@pytest.mark.parametrize("visible,gpus", [
    (None, "Card A, Card B, Card C"),         # unset: every card
    ("", "none"),                             # empty: no GPU used
    ("1", "Card B"), ("2,0", "Card C, Card A"),
    ("GPU-cccc3333", "Card C"), ("GPU-bbbb", "Card B"),   # a UUID or a unique prefix of one
    ("0,-1,2", "Card A"), ("7", "none"),      # CUDA stops at the first entry that names no card
])
def test_gpu_info_records_only_the_cards_cuda_visible_devices_leaves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     visible: str | None, gpus: str) -> None:
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    info = E.gpu_info(_fake_proc(tmp_path))
    assert info["gpus"] == gpus and info["driver"] == "595.91.07" and info["cuda_visible_devices"] == str(visible)


def test_gpu_info_without_the_nvidia_driver(tmp_path: Path) -> None:
    info = E.gpu_info(tmp_path / "absent")
    assert info["gpus"] == "none" and info["driver"] == "none"


def test_the_stub_records_no_gpu_when_none_is_visible(tmp_path: Path) -> None:
    rc, out = _stub_run(tmp_path)                 # the autouse fixture sets CUDA_VISIBLE_DEVICES=""
    assert rc == 0
    report = json.loads((out / "example_report.json").read_text())
    row = json.loads((out / "timing_ledger.jsonl").read_text().splitlines()[-1])
    assert report["host"]["gpus"] == "none" and row["timing_basis"]["hardware"] == "none"
