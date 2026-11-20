"""Source-domain PROFILES: the per-domain production rules, applied per clip and enforced in code.

The EGOCENTRIC profile (clip domains: `fastdo.engine.source_domain`) sets two stage options:

| key | egocentric value | evidence |
|---|---|---|
| `contact_stage.enabled` | false | the contact ruler harms egocentric video (fresh HOI4D median +24.5 mm) |
| `solve_stage.per_frame_reanchor.dead_frame_branch` | false | the branch fails under induced track death and snaps single Ego4D frames onto moving / truncated masks |

(`per_frame_reanchor: null` stays null: no V2 at all has no dead-frame branch.) The THIRD_PERSON profile keeps the
config as given: contact where the config enables it, full V2.

**Unset vs explicit.** The base config leaves `contact_stage.enabled` false and `dead_frame_branch` unset (null), so on an
egocentric clip the profile fills the unset value in. A config that EXPLICITLY asks an egocentric clip for the opposite
(`contact_stage.enabled=true`, `dead_frame_branch=true`) is refused (`SourceProfileConflict`) unless
`source_profile.study_override` is true: a study may run contact on HOI4D on purpose. The run-wide
`contact_stage.enabled=true` counts as explicit for every clip of the run, so a run mixing lab and egocentric clips with
contact on is refused too (split it). With the override the explicit value is kept and the record lists it.

**Where it is applied.** `make_profiled_solve_worker` / `profiled_solve_halves` (single-stream and batched solve) and
`make_profiled_contact_worker`, which the production builders install (`fastdo.bench.bench_stages.build_stage_fns`,
`fastdo.bench.bench_batched.build_batched_stages`) and so every engine runner they feed (`run_clip`, `run_manifest`,
`run_manifest_batched`, `PipelinedRunner`). A clip's domain comes from its `source_domain.json` (written by the bench
before any stage), else from `source_profile.domain` (a harness re-solving directories written before domains were
recorded), else the clip is refused (`UndeclaredSourceDomain`). Each solved clip gets `source_profile.json`
(`PROFILE_FILE`): its domain, where the domain came from, and every rule's configured and applied value. `solve.json`
itself is unchanged.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fastdo.engine.orchestrator import StageFn
from fastdo.engine.solve_stage_config import SolveStageConfig
from fastdo.engine.source_domain import DOMAIN_FILE, SourceDomain, parse_domain, read_domain_record
from fastdo.engine.workers_contact import ContactStageConfig, make_contact_worker
from fastdo.engine.workers_solve import (
    SolveCall,
    SolveFn,
    finish_solve,
    make_solve_worker,
    prepare_solve,
)
from fastdo.solver.api import TrajectorySolution, solve_trajectory

__all__ = ["CONTACT_KEY", "DEAD_FRAME_KEY", "PROFILE_FILE", "AppliedProfile", "SourceProfileConfig",
           "SourceProfileConflict", "UndeclaredSourceDomain", "apply_source_profile", "check_run_profile",
           "make_profiled_contact_worker", "make_profiled_solve_worker", "profiled_solve_halves", "resolve_domain"]

PROFILE_FILE = "source_profile.json"
CONTACT_KEY = "contact_stage.enabled"
DEAD_FRAME_KEY = "solve_stage.per_frame_reanchor.dead_frame_branch"
LEDGER: Mapping[str, str] = MappingProxyType({
    CONTACT_KEY: "egocentric contact-stage evaluation on fresh HOI4D clips (median +24.5 mm with contact on)",
    DEAD_FRAME_KEY: "dead-frame branch evaluation under induced track death and on Ego4D windows"})
_WHY: Mapping[str, str] = MappingProxyType({
    CONTACT_KEY: "the contact along-ray correction harms egocentric video (fresh HOI4D clips: median +24.5 mm vs off, "
                 "53/18 worse) and no egocentric-safe variant passed",
    DEAD_FRAME_KEY: "V2's dead-frame branch fails the induced track-death test and on Ego4D snaps single "
                    "frames onto moving or border-truncated masks (0.5-0.86 m); egocentric sources run it OFF"})
_OVERRIDE_HINT = ("set source_profile.study_override=true for a study that means to run it on egocentric video, or drop "
                  "the explicit value (the egocentric profile sets it)")


class SourceProfileConflict(ValueError):
    """A config explicitly asks egocentric clips for a value the egocentric profile forbids, without the override."""


class UndeclaredSourceDomain(ValueError):
    """A clip reached a profiled worker with no source domain (no `source_domain.json`, no `source_profile.domain`)."""


@dataclass(frozen=True)
class SourceProfileConfig:
    """The `source_profile` config block.

    Attributes:
        study_override: Keep an explicit egocentric-contradicting value (`contact_stage.enabled=true`,
            `dead_frame_branch=true`) instead of refusing it. Studies only; recorded per clip.
        domain: The domain of clip directories WITHOUT `source_domain.json` (a harness re-solving older artifact
            directories). None = such a clip is refused. A clip whose file disagrees is refused either way.
    """

    study_override: bool = False
    domain: SourceDomain | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.study_override, bool):
            raise ValueError(f"source_profile.study_override must be true or false, got {self.study_override!r}")  # noqa: TRY004
        if self.domain is not None:
            object.__setattr__(self, "domain", parse_domain(self.domain))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> SourceProfileConfig:
        """From the `source_profile` block (dict or OmegaConf node; absent = the defaults); unknown keys raise."""
        if not mapping:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        data = {str(k): mapping[k] for k in mapping}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"source_profile has unknown keys {unknown}; known keys: {sorted(known)}")
        return cls(**data)


@dataclass(frozen=True)
class AppliedProfile:
    """One domain's resolved stage options: `solve` (differs from the configured block only in `per_frame_reanchor`),
    `contact_enabled`, the rule keys whose explicit contradicting value the study override kept, and per-rule values."""

    domain: SourceDomain
    solve: SolveStageConfig
    contact_enabled: bool
    overridden: tuple[str, ...]
    rules: Mapping[str, Mapping[str, Any]]

    def record(self, clip_id: str, domain_from: str, study_override: bool) -> dict[str, Any]:
        """The `source_profile.json` payload."""
        return {"clip_id": clip_id, "domain": self.domain.value, "domain_from": domain_from,
                "profile": self.domain.value, "study_override": study_override, "overridden": list(self.overridden),
                "rules": {key: dict(value) for key, value in self.rules.items()}}


def apply_source_profile(domain: SourceDomain, solve: SolveStageConfig, contact_enabled: bool,
                         profile: SourceProfileConfig, where: str = "") -> AppliedProfile:
    """`domain`'s profile applied to the configured `solve` block and `contact_stage.enabled` (module docstring).

    Raises:
        SourceProfileConflict: For an egocentric domain whose config explicitly sets a forbidden value, unless
            `profile.study_override`; the message names every conflicting key, its evidence and the override.
    """
    domain = parse_domain(domain)
    pfr = solve.per_frame_reanchor
    configured_dead = None if pfr is None else pfr.dead_frame_branch
    ego = domain is SourceDomain.EGOCENTRIC
    applied_solve, applied_contact, applied_dead = solve, contact_enabled, configured_dead
    overridden: list[str] = []
    conflicts: list[str] = []
    if ego and contact_enabled:
        (overridden if profile.study_override else conflicts).append(CONTACT_KEY)
    if ego and pfr is not None:
        if configured_dead is True:
            (overridden if profile.study_override else conflicts).append(DEAD_FRAME_KEY)
        elif configured_dead is None:
            applied_dead = False
            applied_solve = dataclasses.replace(solve, per_frame_reanchor=dataclasses.replace(pfr, dead_frame_branch=False))
    if conflicts:
        detail = "; ".join(f"{key}=true ({_WHY[key]})" for key in conflicts)
        raise SourceProfileConflict(f"{where + ': ' if where else ''}the config EXPLICITLY asks egocentric clips for "
                                    f"{detail}. Refused: {_OVERRIDE_HINT}.")
    profile_value = False if ego else None
    rules = {
        CONTACT_KEY: {"configured": contact_enabled, "applied": applied_contact, "profile": profile_value,
                      "ledger": LEDGER[CONTACT_KEY]},
        DEAD_FRAME_KEY: {"per_frame_reanchor": pfr is not None, "configured": configured_dead,
                         "applied": applied_dead, "profile": profile_value if pfr is not None else None,
                         "ledger": LEDGER[DEAD_FRAME_KEY]}}
    return AppliedProfile(domain=domain, solve=applied_solve, contact_enabled=applied_contact,
                          overridden=tuple(overridden), rules=MappingProxyType(rules))


def check_run_profile(domains: Mapping[str, SourceDomain], solve: SolveStageConfig, contact_enabled: bool,
                      profile: SourceProfileConfig) -> dict[SourceDomain, AppliedProfile]:
    """The run-level check, before any stage runs: every clip domain's profile resolves (conflicts raise, naming how many
    clips of that domain the run holds), and `source_profile.domain`, when set, agrees with every clip's domain.

    Raises:
        SourceProfileConflict: As `apply_source_profile`.
        ValueError: If `profile.domain` disagrees with a clip's domain.
    """
    if profile.domain is not None:
        other = sorted(cid for cid, d in domains.items() if d is not profile.domain)
        if other:
            raise ValueError(f"source_profile.domain is {profile.domain.value}, but {len(other)} clips of this run are "
                             f"not (e.g. {other[:3]}): a clip's domain comes from how it enters the bench and a config "
                             "cannot relabel it. Leave source_profile.domain null for bench runs.")
    applied: dict[SourceDomain, AppliedProfile] = {}
    for domain in sorted(set(domains.values())):
        ids = sorted(cid for cid, d in domains.items() if d is domain)
        where = f"{len(ids)} {domain.value} clips of this run (e.g. {ids[:3]})"
        applied[domain] = apply_source_profile(domain, solve, contact_enabled, profile, where)
    return applied


def resolve_domain(artifact_dir: Path, profile: SourceProfileConfig, clip_id: str = "") -> tuple[SourceDomain, str]:
    """`(domain, where it came from)` for one clip: its `source_domain.json`, else `profile.domain`.

    Raises:
        UndeclaredSourceDomain: With neither.
        ValueError: If both are present and disagree, or the file is damaged.
    """
    declared = read_domain_record(artifact_dir)
    if declared is not None:
        if profile.domain is not None and profile.domain is not declared:
            raise ValueError(f"{clip_id}: {Path(artifact_dir) / DOMAIN_FILE} declares {declared.value}, but "
                             f"source_profile.domain is {profile.domain.value}; a config cannot relabel a clip")
        return declared, DOMAIN_FILE
    if profile.domain is not None:
        return profile.domain, "source_profile.domain"
    raise UndeclaredSourceDomain(
        f"{clip_id}: no source domain: {Path(artifact_dir) / DOMAIN_FILE} is absent and source_profile.domain is null. "
        "The production rules differ by domain (the contact stage, the dead-frame branch), so the clip is not "
        "solved blind. Bench runs write the file before any stage; a harness re-solving an older directory sets "
        "source_profile.domain=egocentric|third_person.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


class _Resolver:
    """Per-clip profile resolution shared by the profiled workers (the AppliedProfile per domain is cached)."""

    def __init__(self, solve: SolveStageConfig, contact_enabled: bool, profile: SourceProfileConfig) -> None:
        self.solve, self.contact_enabled, self.profile = solve, contact_enabled, profile
        self._cache: dict[SourceDomain, AppliedProfile] = {}

    def __call__(self, clip_id: str, artifact_dir: Path) -> tuple[AppliedProfile, str]:
        domain, domain_from = resolve_domain(artifact_dir, self.profile, clip_id)
        if domain not in self._cache:
            self._cache[domain] = apply_source_profile(domain, self.solve, self.contact_enabled, self.profile, clip_id)
        return self._cache[domain], domain_from

    def write_record(self, clip_id: str, artifact_dir: Path, applied: AppliedProfile, domain_from: str) -> None:
        _write_json(Path(artifact_dir) / PROFILE_FILE,
                    applied.record(clip_id, domain_from, self.profile.study_override))


def make_profiled_solve_worker(solve: SolveStageConfig, contact_enabled: bool, profile: SourceProfileConfig,
                               solve_fn: SolveFn = solve_trajectory, *, driftqc_enabled: bool = False) -> StageFn:
    """The production single-stream `solve` worker: per clip, the domain's profile applied to `solve`, then
    `make_solve_worker` of the result, then `source_profile.json`. A stale record is removed first, so a failed or
    refused clip leaves none."""
    resolver = _Resolver(solve, contact_enabled, profile)
    workers: dict[SourceDomain, StageFn] = {}

    def solve_worker(clip_id: str, artifact_dir: Path) -> None:
        (Path(artifact_dir) / PROFILE_FILE).unlink(missing_ok=True)
        applied, domain_from = resolver(clip_id, artifact_dir)
        if applied.domain not in workers:
            workers[applied.domain] = make_solve_worker(
                applied.solve, solve_fn, driftqc_enabled=driftqc_enabled,
                allow_egocentric_dead_frames=DEAD_FRAME_KEY in applied.overridden)
        workers[applied.domain](clip_id, artifact_dir)
        resolver.write_record(clip_id, artifact_dir, applied, domain_from)

    return solve_worker


def _profiled_prepare(clip_id: str, artifact_dir: Path, *, resolver: _Resolver, driftqc_enabled: bool) -> SolveCall:
    (Path(artifact_dir) / PROFILE_FILE).unlink(missing_ok=True)
    applied, _ = resolver(clip_id, artifact_dir)              # refuses before any work
    return prepare_solve(clip_id, artifact_dir, applied.solve, driftqc_enabled)


def _profiled_finish(clip_id: str, artifact_dir: Path, call: SolveCall, solution: TrajectorySolution,
                     solve_seconds: float | None = None, *, resolver: _Resolver) -> None:
    applied, domain_from = resolver(clip_id, artifact_dir)
    finish_solve(clip_id, artifact_dir, call, solution, applied.solve, solve_seconds,
                 allow_egocentric_dead_frames=DEAD_FRAME_KEY in applied.overridden)
    resolver.write_record(clip_id, artifact_dir, applied, domain_from)


def profiled_solve_halves(solve: SolveStageConfig, contact_enabled: bool, profile: SourceProfileConfig, *,
                          driftqc_enabled: bool = False
                          ) -> tuple[Callable[[str, Path], SolveCall],
                                     Callable[[str, Path, SolveCall, TrajectorySolution], None]]:
    """`(prepare, finish)` for the batched `SolveBatchStage`: `prepare_solve` / `finish_solve` with each clip's profiled
    config, `finish` then writing `source_profile.json`. The profile changes only `per_frame_reanchor`, which only
    `finish_solve` reads, so the batch's shared solver config, depth-anchor gate and branch selection stay the stage's.
    Both are `functools.partial`s of module functions, so the wiring (`resolver.solve`, `driftqc_enabled`) is
    inspectable."""
    resolver = _Resolver(solve, contact_enabled, profile)
    return (functools.partial(_profiled_prepare, resolver=resolver, driftqc_enabled=driftqc_enabled),
            functools.partial(_profiled_finish, resolver=resolver))


def make_profiled_contact_worker(contact: ContactStageConfig, solve: SolveStageConfig,
                                 profile: SourceProfileConfig) -> StageFn:
    """The production `contact` worker (registered only when `contact_stage.enabled`): per clip, the domain's profile
    decides. Third-person clips are corrected; an egocentric clip is refused (`SourceProfileConflict`) unless the
    study override kept contact on for it, in which case it is corrected (the solve's record lists the override).

    Raises:
        ValueError: If `contact.enabled` is false: a disabled contact stage is never registered.
    """
    if not contact.enabled:
        raise ValueError("make_profiled_contact_worker needs contact_stage.enabled=true: a disabled contact stage is "
                         "not registered")
    resolver = _Resolver(solve, contact.enabled, profile)
    plain = make_contact_worker(contact)
    allowed = make_contact_worker(contact, allow_egocentric=True) if profile.study_override else plain

    def contact_worker(clip_id: str, artifact_dir: Path) -> None:
        applied, _ = resolver(clip_id, artifact_dir)          # an egocentric clip without the override raises here
        (allowed if CONTACT_KEY in applied.overridden else plain)(clip_id, artifact_dir)

    return contact_worker
