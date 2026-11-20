"""Speed ledger of the amortised tracker.

A HOLD is one fresh unit process's lock-held interval (`fastdo.amortised.unit`: the status file records, on the
process's monotonic clock, `t_start` when the process starts, `t_lock` when `flock` succeeded and `t_release` when the
lock fd was closed). Its GPU-s are the LOCK-HELD wall t_release - t_lock: waits (admission refusals, a busy lock, the
driver's back-offs, the time before the lock) are never charged.

- C_ours = the sum of the hold walls of the CHARGED kinds (`CHARGED`: tracker training, prior training, pass@S
  verification, fallback MPC), per component and per domain (Ego4D, OakInk2 egocentric, OakInk2 MoCap;
  only the Ego4D-domain X enters the end-to-end formula).
- Programme GPU-h = the hold walls of EVERY GPU hold of the study, whatever its kind (TUNE, development, probes,
  timing holds, aborted holds), / 3600.
- Contamination: a reference-throughput probe brackets every training hold; a hold whose slowest probe ran
  > `CONTAMINATION` (1.3) x slower than the clean reference rate is CONTAMINATED and is excluded from the HEADLINE. The
  headline then lists the excluded holds and is marked incomplete (`complete` False): their work must be re-timed in
  a clean hold or the speed reported as a range with conditions; the all-holds total stays beside it.
- Holds that die holding the lock (the unit driver's per-hold timeout SIGKILL, the memory sampler's `os._exit(3)`)
  never write t_release; `fastdo.amortised.unit.drive` charges them up to its own reap time on the same clock, and
  `holds_from_drive` turns every launch of a drive into a hold (none is dropped once t_lock is set: aborted
  holds count). `hold_from_unit_status` RAISES on a status with t_lock but no t_release unless the caller supplies
  that end time.
- Domain allocation of the headline (only the Ego4D-domain X enters the end-to-end formula): a hold with a
  `domain` is charged to it whole; a hold without one but with `domain_rows` (executor rows per domain, e.g. a family
  training hold over Ego4D and OakInk2-ego clips at once) is split PRO RATA to those rows (a proposed
  rule); any other hold is reported in its own "unallocated" row. The per-domain rows plus
  "unallocated" sum to the headline (asserted).
- CPU core-s (certification, (a), scene builds) are reported per accepted trajectory, never converted into X.
- X_att = C_MPC / C_ours; X_acc = (C_MPC / A_MPC) / (C_ours / A_ours) (`speed_ratios`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CHARGED", "CONTAMINATION", "DOMAINS", "KINDS", "UNALLOCATED", "Hold", "SpeedLedger",
           "hold_from_unit_status", "holds_from_drive", "speed_ratios"]

KINDS = ("train", "prior", "verify", "fallback", "timing", "probe", "tune", "dev", "aborted")
CHARGED = ("train", "prior", "verify", "fallback")
DOMAINS = ("ego4d", "oakink2_ego_recon", "oakink2_mocap")
CONTAMINATION = 1.3
UNALLOCATED = "unallocated"


@dataclass(frozen=True)
class Hold:
    name: str
    kind: str
    t_start: float                    # process start (monotonic s)
    t_lock: float                     # lock acquired
    t_release: float                  # lock released
    domain: str | None = None
    gpu: bool = True
    probes_rows_s: tuple[float, ...] = ()     # bracketing reference-throughput probes (rows/s)
    clips: tuple[str, ...] = ()
    domain_rows: tuple[tuple[str, int], ...] = ()     # executor rows per domain (allocation of a multi-domain hold)
    release_source: str = "unit"              # "unit" | "driver_reap" | "driver_unknown" (`unit._hold_record`)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown hold kind {self.kind!r}")
        if self.domain is not None and self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}")
        for d, n in self.domain_rows:
            if d not in DOMAINS or int(n) < 0:
                raise ValueError(f"{self.name}: bad domain_rows entry {(d, n)!r}")
        if self.domain is not None and self.domain_rows:
            raise ValueError(f"{self.name}: give either domain or domain_rows")
        if not self.t_start <= self.t_lock <= self.t_release:
            raise ValueError(f"{self.name}: times out of order")

    def allocation(self) -> dict[str, float]:
        """This hold's wall split over domains (module docstring); {UNALLOCATED: wall} when it has no rule."""
        w = self.hold_wall_s
        if self.domain is not None:
            return {self.domain: w}
        tot = sum(int(n) for _, n in self.domain_rows)
        if tot <= 0:
            return {UNALLOCATED: w}
        out: dict[str, float] = {}
        for d, n in self.domain_rows:
            out[d] = out.get(d, 0.0) + w * int(n) / tot
        return out

    @property
    def lock_wait_s(self) -> float:
        return self.t_lock - self.t_start

    @property
    def hold_wall_s(self) -> float:
        return self.t_release - self.t_lock


def hold_from_unit_status(status: dict[str, Any], kind: str, domain: str | None = None,
                          probes_rows_s: Iterable[float] = (), clips: Iterable[str] = (),
                          t_end: float | None = None, name: str | None = None,
                          domain_rows: Iterable[tuple[str, int]] = ()) -> Hold | None:
    """A `Hold` from a unit status file (`fastdo.amortised.unit.run_unit`); None ONLY if the unit never held the lock.
    A status with t_lock but no t_release (the unit died holding it) needs `t_end`, the driver's reap time on the same
    monotonic clock; without it this raises instead of dropping the hold (module docstring)."""
    if status.get("t_lock") is None:
        return None
    release, source = status.get("t_release"), status.get("release_source") or "unit"
    if release is None:
        if t_end is None:
            raise ValueError(f"{status.get('name')}: held the lock but has no t_release; pass the driver's reap time")
        release, source = t_end, "driver_reap"
    return Hold(name=str(name if name is not None else status.get("name", "")), kind=kind,
                t_start=float(status["t_start"]), t_lock=float(status["t_lock"]), t_release=float(release),
                domain=domain, probes_rows_s=tuple(float(x) for x in probes_rows_s), clips=tuple(clips),
                domain_rows=tuple((str(d), int(n)) for d, n in domain_rows), release_source=str(source))


def holds_from_drive(history: Iterable[dict[str, Any]], kind: str, domain: str | None = None,
                     domain_rows: Iterable[tuple[str, int]] = ()) -> list[Hold]:
    """Every lock-holding launch of a `unit.drive` (its `DriveResult.history`, or the rows of its hold log) as a hold.
    A launch refused AFTER taking the lock (it did no work) is kind "aborted" (programme GPU-h only); every other
    lock-holding launch, timed-out and over-cap ones included, is `kind`."""
    out = []
    for row in history:
        if row.get("t_lock") is None:
            continue
        k = "aborted" if row.get("phase") == "refused_after_lock" else kind
        h = hold_from_unit_status(row, k, domain=domain, t_end=row.get("t_reaped"), name=row.get("name"),
                                  domain_rows=domain_rows)
        if h is not None:
            out.append(h)
    return out


@dataclass
class SpeedLedger:
    reference_rows_s: float | None = None      # the clean reference throughput (rows/s)
    holds: list[Hold] = field(default_factory=list)
    cpu_core_s: dict[str, float] = field(default_factory=dict)

    def add(self, h: Hold) -> None:
        if any(x.name == h.name for x in self.holds):
            raise ValueError(f"hold {h.name} already in the ledger")
        self.holds.append(h)

    def add_cpu(self, kind: str, core_s: float) -> None:
        self.cpu_core_s[kind] = self.cpu_core_s.get(kind, 0.0) + float(core_s)

    def slowdown(self, h: Hold) -> float | None:
        if not h.probes_rows_s or not self.reference_rows_s:
            return None
        return self.reference_rows_s / min(h.probes_rows_s)

    def contaminated(self, h: Hold) -> bool:
        """A probed hold slower than CONTAMINATION x the reference; a TRAINING hold without probes counts as
        contaminated (every training hold is bracketed)."""
        s = self.slowdown(h)
        if s is None:
            return h.kind == "train"
        return s > CONTAMINATION

    def _sum(self, holds: Iterable[Hold]) -> float:
        return float(sum(h.hold_wall_s for h in holds))

    def totals(self) -> dict[str, Any]:
        charged = [h for h in self.holds if h.kind in CHARGED]
        clean = [h for h in charged if not self.contaminated(h)]
        excluded = [h for h in charged if self.contaminated(h)]
        by_kind = {k: self._sum(h for h in clean if h.kind == k) for k in sorted({h.kind for h in clean})}
        by_domain: dict[str, float] = {}
        for h in clean:
            for d, w in h.allocation().items():
                by_domain[d] = by_domain.get(d, 0.0) + w
        by_domain = dict(sorted(by_domain.items()))
        headline = self._sum(clean)
        if abs(sum(by_domain.values()) - headline) > 1e-6 * max(1.0, headline):
            raise AssertionError("per-domain allocation does not sum to the headline")
        return {
            "c_ours_all_s": self._sum(charged),
            "headline_c_ours_s": headline,
            "headline_by_kind_s": by_kind,
            "headline_by_domain_s": by_domain,
            "complete": not excluded,
            "excluded_contaminated": [{"name": h.name, "kind": h.kind, "hold_wall_s": h.hold_wall_s,
                                       "slowdown": self.slowdown(h)} for h in excluded],
            "programme_gpu_h": self._sum(h for h in self.holds if h.gpu) / 3600.0,
            "lock_wait_s": float(sum(h.lock_wait_s for h in self.holds)),
            "holds": len(self.holds),
            "cpu_core_s": dict(self.cpu_core_s),
        }

    def cpu_core_s_per_accepted(self, accepted: int) -> float | None:
        return (sum(self.cpu_core_s.values()) / accepted) if accepted else None

    def save(self, path: Path) -> None:
        d = {"reference_rows_s": self.reference_rows_s, "holds": [asdict(h) for h in self.holds],
             "cpu_core_s": self.cpu_core_s}
        tmp = Path(path).with_name(Path(path).name + ".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> SpeedLedger:
        d = json.loads(Path(path).read_text())
        holds = [Hold(**{**h, "probes_rows_s": tuple(h["probes_rows_s"]), "clips": tuple(h["clips"]),
                         "domain_rows": tuple((str(a), int(b)) for a, b in h.get("domain_rows", ()))})
                 for h in d["holds"]]
        return SpeedLedger(d["reference_rows_s"], holds, d["cpu_core_s"])


def speed_ratios(c_mpc_s: float, a_mpc: int, c_ours_s: float, a_ours: int) -> dict[str, float | None]:
    """X_att = C_MPC / C_ours; X_acc = (C_MPC / A_MPC) / (C_ours / A_ours) (None when a count is 0)."""
    x_att = c_mpc_s / c_ours_s if c_ours_s > 0 else None
    x_acc = ((c_mpc_s / a_mpc) / (c_ours_s / a_ours)) if (a_mpc and a_ours and c_ours_s > 0) else None
    return {"x_att": x_att, "x_acc": x_acc}
