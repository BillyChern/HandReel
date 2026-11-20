"""GPU admission rule for amortised-physics units, enforced in code.

Our jobs take turns with E1 on gpu.lock: admission requires a free lock and memory headroom, not that e1_theirs is
complete.

Admission is opt-in: it applies only when `FASTDO_GPU_TOOLS_DIR` names a shared GPU tools directory
(`fastdo.hostpaths.gpu_tools_dir`). With it unset, `TOOLS`, `STATE_PATH`, `LOCK_PATH` and `PEER_HOLD` are None and
`fastdo.amortised.unit` runs its work with no shared lock and no admission check.

State source: `gpu_priority.json` in the GPU tools directory (`STATE_PATH`), written by the job owners:

    {"holders": [{"name": "e1_theirs", "pid": 3177825, "argv1": "<script path>", "phase": "running",
                  "updated_utc": "2026-10-11T20:00:00Z"}, ...]}

`phase` is one of running / pending / complete; `pid` may be null for a holder that has no process (pending or
complete), never for a RUNNING one: a running holder without a pid or argv1 makes the whole file unparsable. A missing
or unparsable file, a holder older than `max_stale_s` (30 min), or one stamped in the future refuses (fail closed).

ADMIT(unit) iff every rule holds (all are evaluated; `Decision.reasons` lists each failure):
(a) rule 0/1, the holder record is FRESH and CONSISTENT with /proc (liveness: `/proc/<pid>/cmdline` argv[1] ==
    `argv1` EXACTLY, no pgrep -f, no substring):
    - a holder whose pid is alive while its phase is pending/complete refuses (a live process is not complete);
    - a "running" holder whose pid is not confirmed alive refuses (its owner must flip the phase; the 30-min
      staleness rule forces that refresh);
    - a "running" holder that IS alive refuses unless it is an E1 holder (`E1_INTERLEAVED`: e1_theirs, e1_ours),
      which takes turns with us on gpu.lock. Other priority holders (freeze headline passes, FDR-h2h timing
      holds) still exclude us: the interleaving covers E1 only.
    e1_theirs must be LISTED (the record describes it); its phase is RECORDED in every decision
    (`Decision.e1_theirs`), no longer required to be "complete";
(b) rule 4, card free MiB (NVML total - used, all processes) >= own cap + 4,096 (`Rules.reserve_mib`); an unreadable
    card refuses. BEFORE the lock, `Facts.holder_mib` (GPU memory of the current lock holder's process tree, which is
    released with the lock; `gpu_lock.holder_mib`) is credited, as the E1-ours queue does; AFTER the lock
    (and at every mid-unit re-check) holder_mib is 0 and the plain card free must clear the bar;
(c) the job then takes gpu.lock by a BLOCKING flock bounded by the unit's `lock_wait_s` (`gpu_lock.acquire`; held by
    our own lineage -> no wait, never queued behind ourselves), and re-checks (a)-(b) and rules 3/5 under the lock;
plus the unchanged rules 3 (own_cap_mib <= 8,192; our other GPU jobs + 1 <= max_concurrent (2)) and 5 (`peer_hold.py`
reports "normal"; unreadable refuses). Rule 2 (an E1-ours date window) and a 30 GiB E1 reserve are RETIRED; the rule
numbers are kept so that recorded reasons stay comparable.

Nothing here reads E1's own outputs: liveness comes only from /proc/<pid>/cmdline.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastdo import hostpaths

__all__ = [
    "DEFAULT_RULES",
    "E1_INTERLEAVED",
    "POLICY",
    "Decision",
    "Facts",
    "Holder",
    "Rules",
    "admit",
    "card_free_mib",
    "decide",
    "gather",
    "load_state",
    "parse_utc",
    "peer_hold_state",
    "pid_alive_exact",
]

logger = logging.getLogger(__name__)

TOOLS = hostpaths.gpu_tools_dir()     # $FASTDO_GPU_TOOLS_DIR; None (unset): no shared lock, no admission
STATE_PATH = None if TOOLS is None else TOOLS / "gpu_priority.json"
LOCK_PATH = None if TOOLS is None else TOOLS / "gpu.lock"
PEER_HOLD = None if TOOLS is None else TOOLS / "peer_hold.py"
PHASES = ("running", "pending", "complete")
UTC = dt.UTC
E1_INTERLEAVED = ("e1_theirs", "e1_ours")          # these take turns with us on gpu.lock
POLICY = {"file": "shared-GPU admission rules", "item": "2026-10-14"}   # the rule set this module implements


def parse_utc(s: str) -> dt.datetime:
    """'YYYY-MM-DDTHH:MM:SSZ' (or any ISO-8601 with an offset) -> aware UTC datetime."""
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise ValueError(f"timestamp without a zone: {s}")
    return t.astimezone(UTC)


@dataclass(frozen=True)
class Rules:
    """Registered admission parameters. `from_file` may only tighten them. The retired fields (`e1_reserve_mib`,
    `blackout`) are refused by name, so an old overrides file cannot silently mean something else."""

    max_stale_s: float = 1800.0
    own_cap_max_mib: int = 8192
    max_concurrent: int = 2
    reserve_mib: int = 4096                          # (b): free >= own cap + 4 GiB, E1 running or not

    @classmethod
    def from_file(cls, path: Path) -> Rules:
        """A JSON file overriding some fields; every override must be at least as strict as the default."""
        raw = json.loads(Path(path).read_text())
        base = cls()
        unknown = sorted(set(raw) - set(base.__dict__))
        if unknown:
            raise ValueError(f"rules file {path} names fields that are not admission parameters: {unknown}")
        r = cls(**{**base.__dict__, **raw})
        if (r.max_stale_s > base.max_stale_s or r.own_cap_max_mib > base.own_cap_max_mib
                or r.max_concurrent > base.max_concurrent or r.reserve_mib < base.reserve_mib):
            raise ValueError(f"rules file {path} loosens a registered admission parameter")
        return r


DEFAULT_RULES = Rules()


@dataclass(frozen=True)
class Holder:
    name: str
    pid: int | None
    argv1: str
    phase: str
    updated_utc: dt.datetime


@dataclass(frozen=True)
class Facts:
    """Everything a decision needs, gathered once (`gather`) so `decide` is pure and testable."""

    now: dt.datetime
    holders: tuple[Holder, ...] | None
    state_error: str | None
    alive: dict[str, bool]
    own_cap_mib: int
    other_jobs: int
    free_mib: int | None
    peer_state: str
    holder_mib: int = 0              # (b): the lock holder's GPU MiB, credited BEFORE the lock only (0 after it)


@dataclass(frozen=True)
class Decision:
    admit: bool
    reasons: tuple[str, ...]
    facts: Facts | None = field(default=None, compare=False)
    e1_theirs: str | None = None     # e1_theirs's phase (or "not listed"): RECORDED, no longer a condition

    def summary(self) -> dict[str, Any]:
        f = self.facts
        return {"admit": self.admit, "reasons": list(self.reasons), "e1_theirs": self.e1_theirs,
                "free_mib": f.free_mib if f else None, "holder_mib": f.holder_mib if f else None,
                "policy": f"{POLICY['file']}#{POLICY['item']}"}


def load_state(path: Path, now: dt.datetime, rules: Rules = DEFAULT_RULES) -> tuple[tuple[Holder, ...] | None,
                                                                                     str | None]:
    """(holders, None) or (None, why the state file is refused)."""
    try:
        raw = json.loads(Path(path).read_text())
        rows = raw["holders"]
        if not isinstance(rows, list):
            raise TypeError("holders is not a list")
        holders = []
        for h in rows:
            pid = h.get("pid")
            if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0):
                raise ValueError(f"bad pid {pid!r}")
            if h["phase"] not in PHASES:
                raise ValueError(f"bad phase {h['phase']!r}")
            if pid is not None and not h.get("argv1"):
                raise ValueError(f"holder {h['name']} has a pid but no argv1")
            if h["phase"] == "running" and (pid is None or not h.get("argv1")):
                raise ValueError(f"holder {h['name']} is running without a pid and argv1")
            holders.append(Holder(str(h["name"]), pid, str(h.get("argv1") or ""), h["phase"],
                                  parse_utc(h["updated_utc"])))
    except FileNotFoundError:
        return None, f"state file {path} missing"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"state file {path} unparsable: {exc}"
    for h in holders:
        age = (now - h.updated_utc).total_seconds()
        if age > rules.max_stale_s:
            return None, f"state file stale: holder {h.name} updated {age:.0f} s ago (> {rules.max_stale_s:.0f})"
        if age < -300:
            return None, f"state file holder {h.name} is stamped in the future"
    names = [h.name for h in holders]
    if len(names) != len(set(names)):
        return None, "state file lists a holder twice"
    return tuple(holders), None


def pid_alive_exact(pid: int | None, argv1: str, proc: Path = Path("/proc")) -> bool:
    """True iff process `pid` exists and its argv[1] equals `argv1` exactly (an unreadable cmdline of an existing
    process counts as alive: fail closed)."""
    if pid is None:
        return False
    p = proc / str(pid)
    try:
        raw = (p / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    except OSError:
        return p.exists()
    argv = raw.split(b"\0")
    if argv and argv[-1] == b"":
        argv = argv[:-1]
    return len(argv) >= 2 and argv[1].decode(errors="replace") == argv1


def _nvidia_smi_card() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=False, timeout=15).stdout


def card_free_mib(query: Callable[[], str] = _nvidia_smi_card) -> int | None:
    """Free MiB of the card (total - used, all processes); the minimum over listed cards; None if unreadable."""
    try:
        out = query()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("nvidia-smi unavailable: %s", exc)
        return None
    free = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and all(x.isdigit() for x in parts):
            free.append(int(parts[0]) - int(parts[1]))
    return min(free) if free else None


def peer_hold_state(path: Path | None = PEER_HOLD) -> str:
    """`peer_hold.hold_state()[0]` of the shared tools file, loaded by path; "unknown" when it cannot be read."""
    if path is None:
        return "unknown"
    try:
        spec = importlib.util.spec_from_file_location("fastdo_peer_hold", path)
        if spec is None or spec.loader is None:
            return "unknown"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(mod.hold_state()[0])
    except Exception as exc:  # noqa: BLE001  (a foreign tools script: any failure means "do not admit")
        logger.warning("peer_hold unreadable: %s", exc)
        return "unknown"


def gather(own_cap_mib: int, *, other_jobs: int, state_path: Path | None = STATE_PATH, now: dt.datetime | None = None,
           free_query: Callable[[], int | None] = card_free_mib, peer_query: Callable[[], str] = peer_hold_state,
           alive: Callable[[int | None, str], bool] = pid_alive_exact, rules: Rules = DEFAULT_RULES,
           holder_query: Callable[[], int] | None = None) -> Facts:
    """The facts of one decision. `holder_query` (BEFORE the lock only: `gpu_lock.holder_mib`) gives the lock holder's
    GPU MiB to credit; None (under the lock, and at every re-check) credits nothing."""
    now = dt.datetime.now(UTC) if now is None else now
    holders, err = ((None, "no shared GPU tools directory (FASTDO_GPU_TOOLS_DIR is unset)") if state_path is None
                    else load_state(state_path, now, rules))
    live = {h.name: alive(h.pid, h.argv1) for h in holders} if holders else {}
    held = max(0, int(holder_query())) if holder_query is not None else 0
    return Facts(now=now, holders=holders, state_error=err, alive=live, own_cap_mib=int(own_cap_mib),
                 other_jobs=int(other_jobs), free_mib=free_query(), peer_state=peer_query(), holder_mib=held)


def decide(f: Facts, rules: Rules = DEFAULT_RULES) -> Decision:
    """Apply (a), (b) and rules 3, 5 (module docstring) to gathered facts; pure."""
    why: list[str] = []
    if f.state_error is not None or f.holders is None:
        why.append(f"rule 0: {f.state_error or 'no state'}")
        by: dict[str, Holder] = {}
    else:
        by = {h.name: h for h in f.holders}
    for h in by.values():
        alive = bool(f.alive.get(h.name))
        if alive and h.phase != "running":
            why.append(f"rule 1: holder {h.name} pid {h.pid} is alive ({h.phase}): the record is inconsistent")
        elif alive and h.name not in E1_INTERLEAVED:
            why.append(f"rule 1: holder {h.name} pid {h.pid} is alive ({h.phase}): not an E1 holder, so it "
                       "does not take turns with us on gpu.lock")
        elif h.phase == "running" and not alive:
            why.append(f"rule 1: holder {h.name} is running (pid {h.pid} not confirmed alive; its owner must flip "
                       "the phase)")
    e1t = by.get("e1_theirs")
    e1_theirs = None if f.holders is None else (e1t.phase if e1t else "not listed")
    if f.holders is not None and e1t is None:
        why.append("rule 1: e1_theirs is not listed (the record must describe it; its phase is recorded, any phase)")
    if f.own_cap_mib > rules.own_cap_max_mib or f.own_cap_mib <= 0:
        why.append(f"rule 3: own cap {f.own_cap_mib} MiB outside (0, {rules.own_cap_max_mib}]")
    if f.other_jobs + 1 > rules.max_concurrent:
        why.append(f"rule 3: {f.other_jobs} other jobs of ours already run (max {rules.max_concurrent} in all)")
    if f.free_mib is None:
        why.append("rule 4: card memory unreadable")
    elif f.free_mib + max(0, f.holder_mib) < rules.reserve_mib + f.own_cap_mib:
        why.append(f"rule 4: card free {f.free_mib} MiB + lock holder {max(0, f.holder_mib)} MiB < reserve "
                   f"{rules.reserve_mib} + cap {f.own_cap_mib}")
    if f.peer_state != "normal":
        why.append(f"rule 5: peer_hold is {f.peer_state}")
    return Decision(admit=not why, reasons=tuple(why), facts=f, e1_theirs=e1_theirs)


def admit(own_cap_mib: int, *, other_jobs: int, rules: Rules = DEFAULT_RULES, **kw: Any) -> Decision:
    """`gather` + `decide`."""
    return decide(gather(own_cap_mib, other_jobs=other_jobs, rules=rules, **kw), rules)
