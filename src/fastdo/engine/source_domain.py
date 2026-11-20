"""The typed SOURCE DOMAIN of a clip: `egocentric` or `third_person`, and the per-clip record that carries it.

Two production rules differ by source domain (the profiles themselves: `fastdo.engine.source_profile`):
- the contact stage is OFF for egocentric video and ON only for third-person lab video;
- V2's dead-frame branch (`solve_stage.per_frame_reanchor.dead_frame_branch`) is OFF for egocentric video.

Both were documentation (config comments, ledger entries), which memory says fails. The domain is therefore a value a
clip CARRIES, fixed by how it enters the bench, never by what a pipeline stage sees:

| route into the bench | source | domain |
|---|---|---|
| `bench.windows_inventory` (S2 windowed: Ego4D, EK-100) | any | egocentric (the S2 route IS egocentric) |
| `bench.hoi4d_windows` | hoi4d | egocentric (head-mounted camera) |
| `bench.dexycb_sequences` | dexycb | third_person (lab rig) |
| whisking copies | whisking | third_person (the DO AS I DO demo) |
| `bench.doh_root` | 100doh | third_person (YouTube; not runnable yet: no prompt source) |
| `bench.oakink2_windows` | oakink2_ego | egocentric (OakInk2's head-mounted view: its extrinsics move, sd 3-5 cm) |
| `bench.oakink2_windows` | oakink2_allo | third_person (OakInk2's static `allocentric_top` rig camera) |

`SOURCE_DOMAINS` is that table by source name; a windowed clip is egocentric by route and refuses a source this table
calls third-person (`fastdo.bench.bench_clips.windowed_clips`).

The bench writes `<clip>/source_domain.json` (`write_domain_record`) before any stage runs; the profiled workers read it
per clip (`read_domain_record`), and the low-level mechanisms (`finish_solve`, the contact worker) refuse to act against
a declared egocentric domain unless the profile authorised it (`refuse_declared_egocentric`).
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = ["DOMAIN_FILE", "SOURCE_DOMAINS", "DeclaredEgocentric", "SourceDomain", "domain_of_source", "parse_domain",
           "read_domain_record", "refuse_declared_egocentric", "write_domain_record"]

DOMAIN_FILE = "source_domain.json"


class SourceDomain(StrEnum):
    """Where the camera is: on the actor's head (`egocentric`) or looking at the actor (`third_person`)."""

    EGOCENTRIC = "egocentric"
    THIRD_PERSON = "third_person"


SOURCE_DOMAINS: MappingProxyType[str, SourceDomain] = MappingProxyType({
    "whisking": SourceDomain.THIRD_PERSON,
    "dexycb": SourceDomain.THIRD_PERSON,
    "100doh": SourceDomain.THIRD_PERSON,
    "epic_kitchens_100": SourceDomain.EGOCENTRIC,
    "ego4d": SourceDomain.EGOCENTRIC,
    "hoi4d": SourceDomain.EGOCENTRIC,
    "oakink2_ego": SourceDomain.EGOCENTRIC,       # the view decides the domain, so it is in the source name
    "oakink2_allo": SourceDomain.THIRD_PERSON,
})
"""Every `fastdo.bench.bench_clips.SOURCES` entry's domain (a test pins that the two stay in step)."""


class DeclaredEgocentric(ValueError):
    """A low-level mechanism was asked to apply a third-person-only rule to a clip that declares itself egocentric."""


def parse_domain(value: Any) -> SourceDomain:
    """`value` (a `SourceDomain` or its string) as a `SourceDomain`.

    Raises:
        ValueError: On anything else, naming the valid values.
    """
    try:
        return SourceDomain(value)
    except ValueError:
        raise ValueError(f"a source domain is one of {[d.value for d in SourceDomain]}, got {value!r}") from None


def domain_of_source(source: str) -> SourceDomain:
    """The domain of a bench `source` name (`SOURCE_DOMAINS`).

    Raises:
        ValueError: On a source the table does not know: its domain must be declared, never guessed.
    """
    try:
        return SOURCE_DOMAINS[source]
    except KeyError:
        raise ValueError(f"source {source!r} has no source domain in SOURCE_DOMAINS {dict(SOURCE_DOMAINS)}; declare "
                         "the clip's domain explicitly") from None


def write_domain_record(artifact_dir: Path, clip_id: str, source: str, domain: SourceDomain) -> None:
    """`<artifact_dir>/source_domain.json` (tmp sibling, then `os.replace`); the directory is created if needed."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {"clip_id": str(clip_id), "source": str(source), "domain": parse_domain(domain).value}
    tmp = artifact_dir / (DOMAIN_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, artifact_dir / DOMAIN_FILE)


def read_domain_record(artifact_dir: Path) -> SourceDomain | None:
    """The domain `<artifact_dir>/source_domain.json` declares, or None when the file is absent.

    Raises:
        ValueError: If the file exists but is not an object with a valid `domain` (a damaged declaration is an error,
            not an absent one).
    """
    path = Path(artifact_dir) / DOMAIN_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "domain" not in data:
        raise ValueError(f"{path} must be an object with a 'domain' key, got {data!r}")
    return parse_domain(data["domain"])


def refuse_declared_egocentric(artifact_dir: Path, clip_id: str, what: str) -> None:
    """The mechanisms' backstop: raise if `artifact_dir` declares an EGOCENTRIC clip.

    Called by `finish_solve` before it applies V2's dead-frame branch and by the contact worker before it corrects,
    unless the source profile authorised the step (`source_profile.study_override`). A directory without a declaration
    (a unit-test fixture, a study's scratch copy of an older run) is not checked here: the production builders refuse
    an undeclared clip before it reaches a mechanism (`fastdo.engine.source_profile.resolve_domain`).

    Raises:
        DeclaredEgocentric: Naming `what`, the rule and the override.
    """
    if read_domain_record(artifact_dir) is SourceDomain.EGOCENTRIC:
        raise DeclaredEgocentric(
            f"{clip_id}: {what} is refused on this clip: {Path(artifact_dir) / DOMAIN_FILE} declares it EGOCENTRIC, "
            "and the egocentric profile turns that step off (the contact stage; the dead-frame branch). Run the clip "
            "through the production builders (fastdo.bench.bench_stages.build_stage_fns), which apply the profile, and "
            "set source_profile.study_override=true only for a study that means to run the step on egocentric video.")
