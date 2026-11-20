"""Source-domain profiles (`fastdo.engine.source_domain`, `fastdo.engine.source_profile`).

The domain record; the profile's resolution (third-person = the config as given, egocentric = contact off and V2's
dead-frame branch off, an EXPLICIT contradicting value refused unless the study override); the per-clip domain
resolution; the profiled single-stream and batched solve workers (egocentric output == the two overrides set by hand,
byte for byte; third-person output == the unprofiled worker, byte for byte); the profiled contact worker; the
mechanisms' backstop; and `source_profile.json`.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from fastdo.bench.bench_clips import SOURCES
from fastdo.engine.batched_stages import SolveBatchStage
from fastdo.engine.per_frame_reanchor import REANCHOR_KEY
from fastdo.engine.source_domain import (
    DOMAIN_FILE,
    SOURCE_DOMAINS,
    DeclaredEgocentric,
    SourceDomain,
    domain_of_source,
    parse_domain,
    read_domain_record,
    refuse_declared_egocentric,
    write_domain_record,
)
from fastdo.engine.source_profile import (
    CONTACT_KEY,
    DEAD_FRAME_KEY,
    PROFILE_FILE,
    SourceProfileConfig,
    SourceProfileConflict,
    UndeclaredSourceDomain,
    apply_source_profile,
    check_run_profile,
    make_profiled_contact_worker,
    make_profiled_solve_worker,
    profiled_solve_halves,
    resolve_domain,
)
from fastdo.engine.workers_contact import CONTACT_FILE, ContactStageConfig, make_contact_worker
from fastdo.engine.workers_solve import (
    PerFrameReanchorConfig,
    SolveStageConfig,
    finish_solve,
    make_solve_worker,
    prepare_solve,
)
from fastdo.solver.synthetic import SyntheticClip, make_synthetic_clip
from tests.engine.fakes_solve import SMALL_CLIP, fake_solution
from tests.engine.test_per_frame_reanchor import FAST, _clip_dir, _npz_identical, _Solve
from tests.engine.test_workers_contact import _clip as _clip_in

EGO, THIRD = SourceDomain.EGOCENTRIC, SourceDomain.THIRD_PERSON
V2 = PerFrameReanchorConfig(threshold=0.25, band=0.2, perp_k=1.0, vis_min=5, occluded_area_frac=0.25,
                            max_log_shift=0.3)
V2_SOLVE = dataclasses.replace(FAST, per_frame_reanchor=V2)
MANUAL = dataclasses.replace(FAST, per_frame_reanchor=dataclasses.replace(V2, dead_frame_branch=False))
OFF, OVERRIDE = SourceProfileConfig(), SourceProfileConfig(study_override=True)


@pytest.fixture(scope="module")
def small_clip() -> SyntheticClip:
    return make_synthetic_clip(SMALL_CLIP, seed=0)


# --- the domain and its record --------------------------------------------------------------------------------------


def test_every_bench_source_has_exactly_one_domain() -> None:
    assert set(SOURCE_DOMAINS) == set(SOURCES)
    assert {s for s, d in SOURCE_DOMAINS.items() if d is EGO} == {"ego4d", "epic_kitchens_100", "hoi4d", "oakink2_ego"}
    assert {s for s, d in SOURCE_DOMAINS.items() if d is THIRD} == {"dexycb", "whisking", "100doh", "oakink2_allo"}
    with pytest.raises(ValueError, match="declare"):
        domain_of_source("ek")
    assert parse_domain("egocentric") is EGO and parse_domain(THIRD) is THIRD
    with pytest.raises(ValueError, match="third_person"):
        parse_domain("exocentric")


def test_the_record_round_trips_and_a_damaged_one_raises(tmp_path: Path) -> None:
    assert read_domain_record(tmp_path / "c") is None
    write_domain_record(tmp_path / "c", "v:k001", "ego4d", EGO)
    assert read_domain_record(tmp_path / "c") is EGO
    assert json.loads((tmp_path / "c" / DOMAIN_FILE).read_text()) == {"clip_id": "v:k001", "source": "ego4d",
                                                                      "domain": "egocentric"}
    (tmp_path / "c" / DOMAIN_FILE).write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_domain_record(tmp_path / "c")
    (tmp_path / "c" / DOMAIN_FILE).write_text(json.dumps({"domain": "somewhere"}))
    with pytest.raises(ValueError, match="source domain"):
        read_domain_record(tmp_path / "c")


def test_the_backstop_refuses_only_a_declared_egocentric_clip(tmp_path: Path) -> None:
    refuse_declared_egocentric(tmp_path / "none", "c", "x")                  # undeclared: not this check's job
    write_domain_record(tmp_path / "lab", "c", "dexycb", THIRD)
    refuse_declared_egocentric(tmp_path / "lab", "c", "x")
    write_domain_record(tmp_path / "ego", "c", "hoi4d", EGO)
    with pytest.raises(DeclaredEgocentric, match="study_override"):
        refuse_declared_egocentric(tmp_path / "ego", "c", "x")


# --- the profile ---------------------------------------------------------------------------------------------------


def test_third_person_keeps_the_config_as_given() -> None:
    for solve, contact in ((V2_SOLVE, True), (V2_SOLVE, False), (FAST, True)):
        applied = apply_source_profile(THIRD, solve, contact, OFF)
        assert applied.solve is solve and applied.contact_enabled is contact and applied.overridden == ()
        assert applied.rules[CONTACT_KEY]["profile"] is None and applied.rules[DEAD_FRAME_KEY]["profile"] is None


def test_egocentric_fills_the_unset_dead_frame_branch_with_false() -> None:
    applied = apply_source_profile(EGO, V2_SOLVE, False, OFF)
    assert applied.solve == MANUAL and applied.contact_enabled is False and applied.overridden == ()
    dead_ledger = "dead-frame branch evaluation under induced track death and on Ego4D windows"
    contact_ledger = "egocentric contact-stage evaluation on fresh HOI4D clips (median +24.5 mm with contact on)"
    assert dict(applied.rules[DEAD_FRAME_KEY]) == {"per_frame_reanchor": True, "configured": None, "applied": False,
                                                    "profile": False, "ledger": dead_ledger}
    assert dict(applied.rules[CONTACT_KEY]) == {"configured": False, "applied": False, "profile": False,
                                                 "ledger": contact_ledger}
    assert apply_source_profile(EGO, MANUAL, False, OFF).solve is MANUAL     # already false: nothing to change
    none = apply_source_profile(EGO, FAST, False, OFF)                         # V2 off entirely stays off
    assert none.solve is FAST and none.rules[DEAD_FRAME_KEY]["per_frame_reanchor"] is False


def test_the_profiled_config_is_the_one_the_manual_overrides_build() -> None:
    """The egocentric profile of the SHIPPED block equals the block with `dead_frame_branch: false` typed by hand."""
    block = {"threshold": 0.25, "band": 0.2, "perp_k": 1.0, "vis_min": 5, "occluded_area_frac": 0.25,
             "max_log_shift": 0.3}
    shipped = SolveStageConfig.from_mapping({"per_frame_reanchor": block})
    by_hand = SolveStageConfig.from_mapping({"per_frame_reanchor": {**block, "dead_frame_branch": False}})
    assert apply_source_profile(EGO, shipped, False, OFF).solve == by_hand
    assert apply_source_profile(THIRD, shipped, False, OFF).solve == shipped != by_hand


@pytest.mark.parametrize(("solve", "contact", "key"), [
    (FAST, True, CONTACT_KEY),
    (dataclasses.replace(FAST, per_frame_reanchor=dataclasses.replace(V2, dead_frame_branch=True)), False,
     DEAD_FRAME_KEY)])
def test_an_explicit_contradiction_is_refused_unless_the_study_override(solve: SolveStageConfig, contact: bool,
                                                                       key: str) -> None:
    with pytest.raises(SourceProfileConflict, match=key.replace(".", r"\.") + "=true") as err:
        apply_source_profile(EGO, solve, contact, OFF, "3 egocentric clips")
    assert "study_override" in str(err.value) and "3 egocentric clips" in str(err.value)
    kept = apply_source_profile(EGO, solve, contact, OVERRIDE)
    assert kept.overridden == (key,) and kept.solve is solve and kept.contact_enabled is contact
    assert apply_source_profile(THIRD, solve, contact, OFF).overridden == ()     # third-person: nothing to refuse


def test_the_run_check_refuses_a_mixed_run_with_contact_on_and_a_relabel() -> None:
    domains = {"lab": THIRD, "v:k001": EGO}
    with pytest.raises(SourceProfileConflict, match=r"1 egocentric clips of this run \(e\.g\. \['v:k001'\]\)"):
        check_run_profile(domains, V2_SOLVE, True, OFF)
    assert check_run_profile({"lab": THIRD}, V2_SOLVE, True, OFF)[THIRD].contact_enabled is True
    applied = check_run_profile(domains, V2_SOLVE, False, OFF)
    assert applied[EGO].solve == MANUAL and applied[THIRD].solve is V2_SOLVE
    with pytest.raises(ValueError, match="cannot relabel"):
        check_run_profile(domains, V2_SOLVE, False, SourceProfileConfig(domain=EGO))
    assert check_run_profile({}, V2_SOLVE, True, OFF) == {}          # a harness with clips=[]: nothing to check here


def test_the_config_block_validates() -> None:
    assert SourceProfileConfig.from_mapping(None) == OFF
    assert SourceProfileConfig.from_mapping({"domain": "egocentric"}).domain is EGO
    with pytest.raises(ValueError, match="unknown keys"):
        SourceProfileConfig.from_mapping({"overide": True})
    with pytest.raises(ValueError, match="true or false"):
        SourceProfileConfig.from_mapping({"study_override": "yes"})
    with pytest.raises(ValueError, match="source domain"):
        SourceProfileConfig.from_mapping({"domain": "lab"})


def test_a_clip_domain_comes_from_its_record_else_the_config_else_nowhere(tmp_path: Path) -> None:
    write_domain_record(tmp_path / "ego", "c", "ego4d", EGO)
    assert resolve_domain(tmp_path / "ego", OFF) == (EGO, DOMAIN_FILE)
    assert resolve_domain(tmp_path / "ego", SourceProfileConfig(domain=EGO)) == (EGO, DOMAIN_FILE)
    with pytest.raises(ValueError, match="cannot relabel"):
        resolve_domain(tmp_path / "ego", SourceProfileConfig(domain=THIRD), "c")
    assert resolve_domain(tmp_path / "old", SourceProfileConfig(domain=THIRD)) == (THIRD, "source_profile.domain")
    with pytest.raises(UndeclaredSourceDomain, match="source_profile.domain"):
        resolve_domain(tmp_path / "old", OFF, "c")


# --- the profiled solve workers ------------------------------------------------------------------------------------


def _dead_frame_clip(root: Path, clip: SyntheticClip, domain: SourceDomain | None) -> Path:
    """A clip whose frames 3 and 4 are track-dead (the `_Solve(1.5, (3, 4))` solve drifts them): V2's dead-frame
    branch moves them."""
    d = _clip_dir(root, clip, dead=(3, 4))
    if domain is not None:
        write_domain_record(d, "c", "ego4d" if domain is EGO else "dexycb", domain)
    return d


def _drift() -> _Solve:
    return _Solve(factor=1.5, frames=(3, 4))


def _drift_many(meshes, obs, init_poses, init_scales, config):
    """`solve_many` at each clip's init, the dead frames 3 and 4 drifted 50 % along the ray (V2's branch moves them)."""
    out = []
    for p, s in zip(init_poses, init_scales, strict=True):
        solution = fake_solution(p, s)
        poses = solution.poses.clone()
        poses[[3, 4], :3, 3] *= 1.5
        out.append(dataclasses.replace(solution, poses=poses))
    return out


def _contact_clip(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return _clip_in(root)


def _strip_seconds(value: object) -> object:
    if isinstance(value, dict):
        return {k: _strip_seconds(v) for k, v in value.items() if k != "seconds"}
    return [_strip_seconds(v) for v in value] if isinstance(value, list) else value


def _meta(d: Path) -> dict:
    """solve.json without its wall-clock `seconds` (the solver's and the re-anchor's), at every depth."""
    return _strip_seconds(json.loads((d / "solve.json").read_text()))


def _same_solve(a: Path, b: Path) -> bool:
    return _npz_identical(a / "solve.npz", b / "solve.npz", skip=("seconds",)) and _meta(a) == _meta(b)


def test_egocentric_solve_equals_the_manual_overrides_and_third_person_equals_the_plain_worker(
        tmp_path: Path, small_clip: SyntheticClip) -> None:
    dirs = {name: _dead_frame_clip(tmp_path / name, small_clip, dom) for name, dom in
            (("ego", EGO), ("manual", None), ("lab", THIRD), ("plain", None))}
    make_profiled_solve_worker(V2_SOLVE, False, OFF, solve_fn=_drift())("c", dirs["ego"])
    make_solve_worker(MANUAL, solve_fn=_drift())("c", dirs["manual"])
    make_profiled_solve_worker(V2_SOLVE, False, OFF, solve_fn=_drift())("c", dirs["lab"])
    make_solve_worker(V2_SOLVE, solve_fn=_drift())("c", dirs["plain"])

    assert _same_solve(dirs["ego"], dirs["manual"]) and _same_solve(dirs["lab"], dirs["plain"])
    assert _meta(dirs["ego"])[REANCHOR_KEY]["frames"] == [] and _meta(dirs["ego"])[REANCHOR_KEY]["dead_frame_branch"] is False
    assert {3, 4} <= set(_meta(dirs["lab"])[REANCHOR_KEY]["frames"])          # the branch fired: the test is not vacuous
    assert not _same_solve(dirs["ego"], dirs["lab"])

    ego, lab = (json.loads((dirs[n] / PROFILE_FILE).read_text()) for n in ("ego", "lab"))
    assert (ego["domain"], ego["domain_from"], ego["study_override"], ego["overridden"]) == (
        "egocentric", DOMAIN_FILE, False, [])
    assert ego["rules"][DEAD_FRAME_KEY]["applied"] is False and ego["rules"][CONTACT_KEY]["applied"] is False
    assert lab["domain"] == "third_person" and lab["rules"][DEAD_FRAME_KEY]["applied"] is None
    assert not (dirs["manual"] / PROFILE_FILE).exists()                        # the mechanism writes no record


def test_the_batched_halves_equal_the_single_stream_worker(tmp_path: Path, small_clip: SyntheticClip) -> None:
    first = _dead_frame_clip(tmp_path / "batch", small_clip, None)
    shutil.copytree(first, tmp_path / "batch" / "d")
    write_domain_record(first, "c", "ego4d", EGO)
    write_domain_record(tmp_path / "batch" / "d", "d", "dexycb", THIRD)
    manual = _dead_frame_clip(tmp_path / "manual", small_clip, None)
    plain = _dead_frame_clip(tmp_path / "plain", small_clip, None)
    prepare, finish = profiled_solve_halves(V2_SOLVE, False, OFF)
    stage = SolveBatchStage(prepare, finish, V2_SOLVE.solver_config(), _drift_many)
    assert stage.run_batch([("c", first), ("d", tmp_path / "batch" / "d")], "b").errors == {}
    for cfg, d, cid in ((MANUAL, manual, "c"), (V2_SOLVE, plain, "d")):
        ref = SolveBatchStage(functools.partial(prepare_solve, cfg=cfg), functools.partial(finish_solve, cfg=cfg),
                              cfg.solver_config(), _drift_many)
        assert ref.run_batch([(cid, d)], "r").errors == {}
    assert _same_solve(first, manual) and _same_solve(tmp_path / "batch" / "d", plain)
    assert _meta(first)[REANCHOR_KEY]["frames"] == [] and _meta(plain)[REANCHOR_KEY]["frames"]   # not vacuous
    assert json.loads((first / PROFILE_FILE).read_text())["domain"] == "egocentric"


def test_an_undeclared_or_refused_clip_writes_nothing_and_a_stale_record_goes(tmp_path: Path,
                                                                             small_clip: SyntheticClip) -> None:
    d = _dead_frame_clip(tmp_path, small_clip, None)
    with pytest.raises(UndeclaredSourceDomain):
        make_profiled_solve_worker(V2_SOLVE, False, OFF, solve_fn=_drift())("c", d)
    write_domain_record(d, "c", "ego4d", EGO)
    (d / PROFILE_FILE).write_text("{}")                                       # a stale record from an earlier run
    with pytest.raises(SourceProfileConflict):
        make_profiled_solve_worker(V2_SOLVE, True, OFF, solve_fn=_drift())("c", d)
    assert not (d / PROFILE_FILE).exists() and not (d / "solve.npz").exists()
    prepare, _ = profiled_solve_halves(V2_SOLVE, True, OFF)
    with pytest.raises(SourceProfileConflict):
        prepare("c", d)
    make_profiled_solve_worker(V2_SOLVE, False, SourceProfileConfig(domain=EGO), solve_fn=_drift())  # builds fine


def test_the_study_override_keeps_the_dead_frame_branch_on_egocentric_clips(tmp_path: Path,
                                                                            small_clip: SyntheticClip) -> None:
    on = dataclasses.replace(FAST, per_frame_reanchor=dataclasses.replace(V2, dead_frame_branch=True))
    ego = _dead_frame_clip(tmp_path / "ego", small_clip, EGO)
    ref = _dead_frame_clip(tmp_path / "ref", small_clip, None)
    with pytest.raises(SourceProfileConflict, match="dead_frame_branch=true"):
        make_profiled_solve_worker(on, False, OFF, solve_fn=_drift())("c", ego)
    make_profiled_solve_worker(on, False, OVERRIDE, solve_fn=_drift())("c", ego)
    make_solve_worker(on, solve_fn=_drift())("c", ref)
    assert _same_solve(ego, ref) and {3, 4} <= set(_meta(ego)[REANCHOR_KEY]["frames"])
    record = json.loads((ego / PROFILE_FILE).read_text())
    assert record["study_override"] is True and record["overridden"] == [DEAD_FRAME_KEY]


def test_the_mechanism_refuses_the_dead_frame_branch_on_a_declared_egocentric_clip(
        tmp_path: Path, small_clip: SyntheticClip) -> None:
    d = _dead_frame_clip(tmp_path, small_clip, EGO)
    with pytest.raises(DeclaredEgocentric, match="dead-frame branch"):
        make_solve_worker(V2_SOLVE, solve_fn=_drift())("c", d)
    assert not (d / "solve.npz").exists()
    make_solve_worker(MANUAL, solve_fn=_drift())("c", d)                       # branch off: allowed
    make_solve_worker(FAST, solve_fn=_drift())("c", d)                         # no V2: allowed
    make_solve_worker(V2_SOLVE, solve_fn=_drift(), allow_egocentric_dead_frames=True)("c", d)


# --- the profiled contact worker -----------------------------------------------------------------------------------


def test_contact_runs_on_third_person_and_is_refused_on_egocentric_unless_overridden(tmp_path: Path) -> None:
    on = ContactStageConfig(enabled=True)
    lab = _contact_clip(tmp_path / "lab")
    write_domain_record(lab, "clip", "dexycb", THIRD)
    make_profiled_contact_worker(on, V2_SOLVE, OFF)("clip", lab)
    ref = _contact_clip(tmp_path / "ref")
    make_contact_worker(on)("clip", ref)
    assert _npz_identical(lab / CONTACT_FILE, ref / CONTACT_FILE)

    ego = _contact_clip(tmp_path / "ego")
    write_domain_record(ego, "clip", "hoi4d", EGO)
    with pytest.raises(SourceProfileConflict, match="contact_stage.enabled=true"):
        make_profiled_contact_worker(on, V2_SOLVE, OFF)("clip", ego)
    with pytest.raises(DeclaredEgocentric, match="contact stage"):
        make_contact_worker(on)("clip", ego)                                   # the mechanism's own backstop
    assert not (ego / CONTACT_FILE).exists()
    make_profiled_contact_worker(on, V2_SOLVE, OVERRIDE)("clip", ego)          # a study-style override
    assert _npz_identical(ego / CONTACT_FILE, ref / CONTACT_FILE)
    with pytest.raises(ValueError, match="not registered"):
        make_profiled_contact_worker(ContactStageConfig(), V2_SOLVE, OFF)
    undeclared = _contact_clip(tmp_path / "old")
    with pytest.raises(UndeclaredSourceDomain):
        make_profiled_contact_worker(on, V2_SOLVE, OFF)("clip", undeclared)
    make_contact_worker(on)("clip", undeclared)                                # the mechanism alone: config as given
    assert np.load(undeclared / CONTACT_FILE)["poses"].shape == (4, 4, 4)
