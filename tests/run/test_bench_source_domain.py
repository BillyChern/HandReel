"""The source domain from every bench route to the stage configs, through the registered entry points.

Routes: whisking copies, `bench.dexycb_sequences`, `bench.doh_root`, `bench.windows_inventory` (S2 windowed) and
`bench.hoi4d_windows`, each alone and together through `build_clips`. Carrier: `prepare_artifacts` writes every clip's
`source_domain.json` before any stage (also for injected workers and on resume). Entry points: `run_bench` refuses an
egocentric run with contact on BEFORE it deletes anything (and runs it with the study override);
`build_stage_fns` refuses up front and installs the profiled solve / contact workers; `build_batched_stages` installs
the profiled batched solve; the engine's `run_manifest` / `run_clip` carries the domain through the artifact directory
(an undeclared clip fails its stage with the reason).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastdo.bench.bench_batched import build_batched_stages
from fastdo.bench.bench_clips import (
    BenchClip,
    dexycb_clips,
    doh_clips,
    whisking_clips,
    windowed_clips,
    windowed_clips_from_inventory,
)
from fastdo.bench.bench_stages import build_stage_fns
from fastdo.bench.hoi4d_bench import HOI4DWindow, hoi4d_clips
from fastdo.bench.yield_report import read_yield_report
from fastdo.engine.manifest import ClipManifest
from fastdo.engine.orchestrator import EngineConfig, run_manifest
from fastdo.engine.source_domain import (
    DOMAIN_FILE,
    SourceDomain,
    read_domain_record,
    write_domain_record,
)
from fastdo.engine.source_profile import PROFILE_FILE, SourceProfileConflict, UndeclaredSourceDomain
from fastdo.engine.workers_solve import SolveStageConfig, make_solve_worker
from fastdo.io.datasets.hoi4d import HOI4DReader
from fastdo.solver.synthetic import make_synthetic_clip
from run.pipeline import bench_throughput as bench
from tests.bench.dexycb_tree import write_fake_dexycb
from tests.bench.fake_doh import write_fake_doh_bundle
from tests.engine.fakes_solve import SMALL_CLIP
from tests.engine.test_per_frame_reanchor import _clip_dir, _npz_identical
from tests.engine.test_source_profile import _meta
from tests.io.hoi4d_tree import SEQ, write_tree

_REPO = Path(__file__).resolve().parents[2]
EGO, THIRD = SourceDomain.EGOCENTRIC, SourceDomain.THIRD_PERSON
# the production solve block (depth-anchor gate, V2) at the synthetic clip's small CPU sizes
_SMALL_SOLVE = ["solve_stage.device=cpu", "solve_stage.seed=0", "solve_stage.registration.n_obs=128",
                "solve_stage.registration.n_mesh=256", "solve_stage.registration.n_iters=10",
                "solve_stage.n_anchor_surface=4096", "+solve_stage.solver.iters_coarse=3",
                "+solve_stage.solver.iters_mid=2", "+solve_stage.solver.iters_fine=3"]


def _cfg(tmp_path: Path, *overrides: str) -> DictConfig:
    base = [f"paths.root={_REPO}", f"bench.out_dir={tmp_path / 'bench'}", f"bench.artifact_root={tmp_path / 'engine'}",
            "bench.batched.enabled=false", f"whisking.demo_root={tmp_path / 'demo'}"]
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[*base, *overrides])


def _hoi4d(tmp_path: Path) -> tuple[Path, Path]:
    """A fake HOI4D tree and a one-window `bench.hoi4d_windows` file."""
    root = write_tree(tmp_path / "hoi4d")
    reader = HOI4DReader(root)
    reader.video_path(SEQ).parent.mkdir(parents=True)
    reader.video_path(SEQ).write_bytes(b"")
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps([{"sequence": SEQ, "start": 2, "stop": 10, "shift_px": [0.0, 0.0]}]))
    return root, windows


# --- every route ---------------------------------------------------------------------------------------------------


def test_every_route_gives_its_domain(tmp_path: Path) -> None:
    assert {c.domain for c in whisking_clips(tmp_path / "w.mp4", 138, 2)} == {THIRD}
    seq = write_fake_dexycb(tmp_path / "dexycb")
    assert dexycb_clips(tmp_path / "dexycb", [seq])[0].domain is THIRD
    bundle = write_fake_doh_bundle(tmp_path / "100DOH")
    assert {c.domain for c in doh_clips(bundle.root, count_frames=lambda _p: 250)} == {THIRD}
    reader = HOI4DReader(write_tree(tmp_path / "hoi4d"))
    reader.video_path(SEQ).parent.mkdir(parents=True)
    reader.video_path(SEQ).write_bytes(b"")
    assert hoi4d_clips(reader, [HOI4DWindow(SEQ, 3, 153, (0.0, 0.0))])[0].domain is EGO
    # the S2 windowed route is egocentric whatever the source string, and refuses a known third-person source
    for source in ("ego4d", "epic_kitchens_100", "ek"):
        assert {c.domain for c in windowed_clips("v", Path("/v.mp4"), 35.0, 30.0, source)} == {EGO}
    with pytest.raises(ValueError, match="third_person, not egocentric"):
        windowed_clips("v", Path("/v.mp4"), 35.0, 30.0, "dexycb")
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps([{"video_id": "ego_a", "path": str(video), "source": "ego4d", "split": "CALIB"}]))
    windows = windowed_clips_from_inventory(inventory, fps_of=lambda _p: 30.0, count_frames=lambda _p: 1500)
    assert windows and {c.domain for c in windows} == {EGO}


def test_a_bench_clip_derives_or_checks_its_domain() -> None:
    assert BenchClip("c", "ego4d", "video", Path("/v"), 1).domain is EGO
    assert BenchClip("c", "dexycb", "image_sequence", Path("/d"), 1).domain is THIRD
    assert BenchClip("c", "ek", "video", Path("/v"), 1, domain="egocentric").domain is EGO       # parsed to the enum
    with pytest.raises(ValueError, match="declare"):
        BenchClip("c", "ek", "video", Path("/v"), 1)                    # an unknown source is never guessed
    with pytest.raises(ValueError, match="cannot contradict"):
        BenchClip("c", "hoi4d", "video", Path("/v"), 1, domain=THIRD)


def test_build_clips_carries_every_routes_domain(tmp_path: Path) -> None:
    seq = write_fake_dexycb(tmp_path / "dexycb")
    root, windows = _hoi4d(tmp_path)
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1", f"dexycb.root={tmp_path / 'dexycb'}",
               f"bench.dexycb_sequences=[{seq}]", f"hoi4d.root={root}", f"bench.hoi4d_windows={windows}")
    clips = bench.build_clips(cfg, _REPO)
    assert {c.source: c.domain for c in clips} == {"whisking": THIRD, "dexycb": THIRD, "hoi4d": EGO}


# --- the carrier ---------------------------------------------------------------------------------------------------


def test_prepare_artifacts_declares_every_clip_before_any_stage(tmp_path: Path) -> None:
    root, windows = _hoi4d(tmp_path)
    cfg = _cfg(tmp_path, "bench.doh_root=null", "bench.whisking_copies=1", f"hoi4d.root={root}",
               f"bench.hoi4d_windows={windows}")
    clips = bench.build_clips(cfg, _REPO)
    engine = tmp_path / "engine"
    bench.prepare_artifacts(cfg, _REPO, clips, engine, write_inputs=False)       # injected workers: still declared
    assert {c.clip_id: read_domain_record(engine / c.clip_id) for c in clips} == {c.clip_id: c.domain for c in clips}
    hoi = next(c for c in clips if c.source == "hoi4d")
    write_domain_record(engine / hoi.clip_id, hoi.clip_id, "hoi4d", THIRD)       # a kept directory that disagrees
    resume = OmegaConf.merge(cfg, {"bench": {"overwrite": False}})
    with pytest.raises(ValueError, match="cannot change domain"):
        bench.prepare_artifacts(resume, _REPO, clips, engine, write_inputs=False, resume=True)


# --- run_bench -----------------------------------------------------------------------------------------------------


def test_run_bench_refuses_contact_on_an_egocentric_run_before_deleting_anything(tmp_path: Path) -> None:
    root, windows = _hoi4d(tmp_path)
    args = ("bench.doh_root=null", "bench.whisking_copies=0", f"hoi4d.root={root}", f"bench.hoi4d_windows={windows}",
            "bench.stages=[decode]")
    (clip,) = bench.build_clips(_cfg(tmp_path, *args), _REPO)
    kept = tmp_path / "engine" / clip.clip_id / "solve.npz"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"an earlier run's artifact")
    with pytest.raises(SourceProfileConflict, match="contact_stage.enabled=true"):
        bench.run_bench(_cfg(tmp_path, *args, "contact_stage.enabled=true"), _REPO,
                        stage_fns={"decode": lambda _c, _d: None})
    assert kept.read_bytes() == b"an earlier run's artifact"                    # bench.overwrite never reached it
    with pytest.raises(SourceProfileConflict, match="dead_frame_branch=true"):
        bench.run_bench(_cfg(tmp_path, *args, "+solve_stage.per_frame_reanchor.dead_frame_branch=true"), _REPO,
                        stage_fns={"decode": lambda _c, _d: None})
    bench.run_bench(_cfg(tmp_path, *args, "contact_stage.enabled=true", "source_profile.study_override=true"), _REPO,
                    stage_fns={"decode": lambda _c, _d: None})                  # a study-override run
    provenance = read_yield_report(tmp_path / "bench" / bench.REPORT_FILE)
    assert provenance["source_domains"] == {"egocentric": 1, "third_person": 0}
    assert provenance["source_profile"] == {"study_override": True}
    assert read_domain_record(tmp_path / "engine" / clip.clip_id) is EGO


# --- build_stage_fns / build_batched_stages and the engine runners --------------------------------------------------


def _ego_clip() -> BenchClip:
    return BenchClip("v:k001", "ego4d", "video", Path("/v.mp4"), 300, window=(0, 300), decode_stride=3)


def test_build_stage_fns_refuses_up_front_and_honours_the_override(tmp_path: Path) -> None:
    lab = BenchClip("lab", "whisking", "video", Path("/w.mp4"), 138)
    with pytest.raises(SourceProfileConflict, match=r"1 egocentric clips of this run"):
        build_stage_fns(_cfg(tmp_path, "contact_stage.enabled=true"), _REPO, [lab, _ego_clip()])
    fns, labels = build_stage_fns(_cfg(tmp_path, "contact_stage.enabled=true"), _REPO, [lab])
    assert "contact" in fns and "contact" in labels
    fns, _ = build_stage_fns(_cfg(tmp_path, "contact_stage.enabled=true", "source_profile.study_override=true"),
                             _REPO, [lab, _ego_clip()])
    assert "contact" in fns
    with pytest.raises(ValueError, match="cannot relabel"):
        build_stage_fns(_cfg(tmp_path, "source_profile.domain=third_person"), _REPO, [_ego_clip()])
    build_stage_fns(_cfg(tmp_path, "contact_stage.enabled=true"), _REPO, [])   # a harness's clips=[]: per-clip only


def _manual_solve(cfg: DictConfig) -> SolveStageConfig:
    """The registered solve block with `dead_frame_branch: false` typed by hand."""
    block = OmegaConf.to_container(cfg.solve_stage, resolve=True)
    block["per_frame_reanchor"]["dead_frame_branch"] = False
    return SolveStageConfig.from_mapping(block)


def _same(a: Path, b: Path) -> bool:
    return _npz_identical(a / "solve.npz", b / "solve.npz", skip=("seconds",)) and _meta(a) == _meta(b)


def test_the_registered_workers_solve_egocentric_clips_as_the_manual_overrides(tmp_path: Path) -> None:
    clip = make_synthetic_clip(SMALL_CLIP, seed=0)
    cfg = _cfg(tmp_path, *_SMALL_SOLVE)
    fns, _ = build_stage_fns(cfg, _REPO, [])
    ego = _clip_dir(tmp_path / "engine", clip, "ego", dead=(3, 4))
    lab = tmp_path / "engine" / "lab"
    shutil.copytree(ego, lab)
    undeclared = tmp_path / "engine" / "old"
    shutil.copytree(ego, undeclared)
    manual, plain = tmp_path / "ref" / "ego", tmp_path / "ref" / "lab"
    shutil.copytree(ego, manual)
    shutil.copytree(ego, plain)
    write_domain_record(ego, "ego", "ego4d", EGO)
    write_domain_record(lab, "lab", "dexycb", THIRD)

    # the engine's registered runner: the domain reaches the solve through the artifact directory
    manifest = ClipManifest.create(["ego", "lab", "old"], ["ego4d", "dexycb", "ego4d"], [18, 18, 18])
    run_manifest(manifest, EngineConfig(artifact_root=tmp_path / "engine", stages=("solve",)), {"solve": fns["solve"]})
    assert [manifest.status(c, "solve").state for c in ("ego", "lab", "old")] == ["done", "done", "failed"]
    assert "no source domain" in manifest.status("old", "solve").error
    make_solve_worker(_manual_solve(cfg))("ego", manual)
    make_solve_worker(SolveStageConfig.from_mapping(cfg.solve_stage))("lab", plain)
    assert _same(ego, manual) and _same(lab, plain)
    assert _meta(ego)["per_frame_reanchor"]["dead_frame_branch"] is False
    assert "dead_frame_branch" not in _meta(lab)["per_frame_reanchor"]
    assert json.loads((ego / PROFILE_FILE).read_text())["domain"] == "egocentric"

    # the batched solve of build_batched_stages is profiled the same way
    batched = build_batched_stages(cfg, fns, tmp_path / "jobs", tmp_path / "gpu.lock")["solve"]
    again = tmp_path / "batched" / "ego"
    shutil.copytree(manual, again)
    for name in ("solve.npz", "solve.json"):
        (again / name).unlink()
    write_domain_record(again, "ego", "ego4d", EGO)
    assert batched.run_batch([("ego", again)], "b").errors == {}
    assert _same(again, manual)
    with pytest.raises(UndeclaredSourceDomain):
        batched.prepare("old", undeclared)
    assert (again / DOMAIN_FILE).is_file() and (again / PROFILE_FILE).is_file()
