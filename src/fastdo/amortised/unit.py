"""A per-unit GPU process and its driver: bounded, resumable GPU work that takes turns on a shared GPU lock.

One UNIT = one fresh process (`python -m fastdo.amortised.unit <config.json>`) doing <= 15 min of work. Steps 1 and 2
apply only when `FASTDO_GPU_TOOLS_DIR` names a shared GPU tools directory (`fastdo.hostpaths.gpu_tools_dir`); with it
unset, `UnitConfig.lock_path` is "" and the unit takes no lock and checks no admission (`lock_source` "none").

1. admission (`fastdo.amortised.admission`) BEFORE the lock, crediting the lock holder's GPU memory
   (`admission.POLICY`); refused -> exit 75 (TEMPFAIL), nothing taken;
2. a BLOCKING flock on `$FASTDO_GPU_TOOLS_DIR/gpu.lock` (`fastdo.amortised.gpu_lock.acquire`: we take turns with the
   other jobs, which wait in a blocking flock too), bounded by `lock_wait_s` (status phase "waiting_lock" while
   blocked; still busy at the end -> "lock_busy", exit 75, nothing held); held by our own lineage -> no wait (never queued behind ourselves);
   admission AGAIN under the lock, with the plain card free (no credit);
3. memory enforcement before any allocation: FASTDO_GPU_MEM_GB = cap; torch's per-process fraction when a GPU is
   visible; `fastdo.bench.device_memory.ProcessMemorySampler(stop_mib=cap)` (NVML per-PID: CUDA context and Warp
   allocations included) exits the process past the cap; after warm-up the work calls `ctx.after_warmup()`, which
   exits 75 if NVML already reports more than the cap;
4. the work function (`module:function`) runs with a `UnitContext`: checkpoints <= 5 min (atomic JSON), admission
   re-checked every `recheck_s` by `ctx.tick(state)` (a failure saves `state` as the checkpoint, then the unit exits
   75), `time_up()` at the unit length;
5. exit 0 = the work's target reached (checkpoint `done`), 10 = progress made, more units needed; 3 = the NVML
   sampler found the process past its cap (the driver STOPS on it: relaunching would put the same over-cap
   allocation beside E1 again for up to one sampler interval).

Speed ledger (`fastdo.amortised.speed_ledger`): the status file carries `t_start` (process start), `t_lock` (flock
acquired; written at once, phase "locked") and `t_release` (lock fd closed) on the monotonic clock; a hold's GPU-s are
t_release - t_lock. The status also carries `t_wait0` (start of a blocking lock wait, None without one),
`lock_blocked_s` (t_lock - t_wait0) and `lock_source` ("own" or "ancestor:<pid>"). `drive` copies these into every
history row (and an optional hold log) before the next launch, and charges a unit that died holding the lock (timeout
SIGKILL, over-cap exit) up to the driver's reap time (`_hold_record`); `speed_ledger.holds_from_drive` turns the rows into ledger holds.

Whole-hold bound: the work calls `ctx.setup_done()` once its setup (imports, scene build, graph
capture) is over; the unit stamps `t_setup` (monotonic) in its status at once, and `drive` reports each launch's
`setup_s` = t_setup - t_launch (the span the driver's timeout also counts from the launch). `drive`'s
`hold_timeout_s` may be a callable read before EVERY launch, so the per-hold timeout can become c_cap + p95(setup)
once setup has been measured (`fastdo.amortised.fallback.HoldBound`); each row records the timeout it ran under,
and `on_hold(row)` sees every row before the next launch.

Blocking lock: the time a unit spends BLOCKED on gpu.lock is not part of its hold. The driver's clock for a launch
starts at the EFFECTIVE launch `t_launch_eff` = t_launch + lock_blocked_s (the real launch shifted by the blocking
wait; equal to t_launch when the lock was free), so the per-hold timeout, `setup_s` = t_setup - t_launch_eff and every
"from the launch" quantity keep the meaning they have without a wait (start-up, admission, setup, work). The driver
reads the unit's status while it runs: the deadline is t_launch + lock_blocked_s + timeout (a unit still waiting has
its wait added as it grows, bounded by `lock_wait_s` + `LOCK_GRACE_S`, past which it is killed). So a hold's charge
t_release - t_lock stays <= timeout + the reap latency, as before.

The lock is held by the unit process only (or, under a `flock gpu.lock` wrapper, by that ancestor) and is released
when it exits (also on SIGKILL). Children cannot keep it:
the lock fd is non-inheritable, so an exec'd child (subprocess, multiprocessing "spawn"/"forkserver") never has it,
and a child created by a bare fork() (multiprocessing "fork", fork-based DataLoader workers) closes its inherited copy
at once (`os.register_at_fork`), so the flock dies with the unit even if such a child outlives it. The DRIVER loop
(`drive`) runs outside the lock, launches one fresh unit process at a time in its own session, applies a per-hold
timeout that SIGKILLs the unit's whole process group (children holding GPU memory included), and budgets WAITS
(refused or lock-busy attempts, each followed by a back-off) separately from LAUNCHES (units that ran). Resume = the
work reloading its last checkpoint.

Test fakes (`UnitConfig.fakes`: now, free_mib, peer, nvml_mib) are honoured ONLY when CUDA_VISIBLE_DEVICES is set
and empty, i.e. when no GPU can be touched; with a GPU visible every fact is real. `free_mib` may be a list: the
k-th admission check of the process reads element k (the last one repeats).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastdo import hostpaths
from fastdo.amortised import admission as A
from fastdo.amortised import gpu_lock as G

__all__ = ["EXIT_DONE", "EXIT_MORE", "EXIT_OVER_CAP", "EXIT_TEMPFAIL", "LOCK_WAIT_S", "DriveResult", "TempFail",
           "UnitConfig", "UnitContext", "drive", "main", "toy_counter"]

logger = logging.getLogger(__name__)

EXIT_DONE, EXIT_MORE, EXIT_TEMPFAIL = 0, 10, 75
EXIT_OVER_CAP = 3                     # fastdo.bench.device_memory's exit status past `stop_mib`
UNIT_MAX_S, CHECKPOINT_MAX_S = 900.0, 300.0
LOCK_WAIT_S, LOCK_WAIT_MAX_S = 3600.0, 21600.0    # default / largest blocking wait for gpu.lock
LOCK_GRACE_S = 60.0                   # driver backstop beyond `lock_wait_s` for a unit still waiting
STATUS_POLL_S = 0.1                   # driver's status poll while a unit runs
REGISTRY = hostpaths.unit_registry_dir()   # $FASTDO_UNIT_REGISTRY, default <scratch root>/units
_HOST_VARS = "FASTDO_UNIT_REGISTRY (or FASTDO_SCRATCH_ROOT), or the unit config's own paths"
MODULE = "fastdo.amortised.unit"
WAIT_PHASES = ("refused", "lock_busy", "refused_after_lock")
LOCK_FIELDS = ("t_wait0", "lock_blocked_s", "lock_source", "lock_holders")


@dataclass(frozen=True)
class UnitConfig:
    name: str
    work: str
    own_cap_mib: int
    ckpt_path: str
    status_path: str
    work_args: dict[str, Any] = field(default_factory=dict)
    state_path: str = "" if A.STATE_PATH is None else str(A.STATE_PATH)   # "": no shared GPU tools directory
    lock_path: str = "" if A.LOCK_PATH is None else str(A.LOCK_PATH)      # "": no shared lock, no admission
    registry_dir: str = str(REGISTRY)
    unit_s: float = UNIT_MAX_S
    checkpoint_s: float = CHECKPOINT_MAX_S
    recheck_s: float = 30.0
    sampler_interval_s: float = 0.25
    fakes: dict[str, Any] = field(default_factory=dict)
    lock_wait_s: float = LOCK_WAIT_S

    def __post_init__(self) -> None:
        if not 0 < self.unit_s <= UNIT_MAX_S:
            raise ValueError(f"unit_s {self.unit_s} outside (0, {UNIT_MAX_S}]")
        if not 0 < self.checkpoint_s <= CHECKPOINT_MAX_S:
            raise ValueError(f"checkpoint_s {self.checkpoint_s} outside (0, {CHECKPOINT_MAX_S}]")
        if ":" not in self.work:
            raise ValueError("work must be 'module:function'")
        if not 0 <= self.lock_wait_s <= LOCK_WAIT_MAX_S:
            raise ValueError(f"lock_wait_s {self.lock_wait_s} outside [0, {LOCK_WAIT_MAX_S}]")

    def dump(self, path: Path) -> None:
        _atomic_write(Path(path), json.dumps(dataclasses.asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> UnitConfig:
        return cls(**json.loads(Path(path).read_text()))


class TempFail(Exception):
    """A rule failed mid-unit: checkpoint and exit 75."""


def _atomic_write(path: Path, text: str) -> None:
    hostpaths.require(path, _HOST_VARS, create=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _no_gpu_visible() -> bool:
    return os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def _fakes(cfg: UnitConfig) -> dict[str, Any]:
    return dict(cfg.fakes) if cfg.fakes and _no_gpu_visible() else {}


def _unit_alive(pid: int) -> bool:
    try:
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return len(argv) >= 3 and argv[1] == b"-m" and argv[2] == MODULE.encode()


def other_units(registry: Path, self_pid: int, prune: bool = True) -> int:
    """Our other live unit processes. With `prune`, registry files of dead units are removed; a read-only caller
    (an admission check that only reads) passes prune=False."""
    n = 0
    if not registry.is_dir():
        return 0
    for p in registry.glob("*.json"):
        try:
            pid = int(p.stem)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if _unit_alive(pid):
            n += 1
        elif prune:
            p.unlink(missing_ok=True)
    return n


def _admission(cfg: UnitConfig, fk: dict[str, Any], before_lock: bool = False) -> A.Decision:
    """One admission check. `before_lock`: the lock holder's GPU memory is credited (admission (b)); never under the
    lock (the post-lock check and every mid-unit re-check see the plain card free)."""
    kw: dict[str, Any] = {"state_path": Path(cfg.state_path)}
    if before_lock:
        if "holder_mib" in fk:
            kw["holder_query"] = lambda: int(fk["holder_mib"])
        else:
            kw["holder_query"] = lambda: G.holder_mib(Path(cfg.lock_path))
    if "now" in fk:
        kw["now"] = A.parse_utc(fk["now"])
    if "free_mib" in fk:
        free = fk["free_mib"]
        if isinstance(free, list):                                       # fk is this process's own copy
            k = min(fk.get("_free_calls", 0), len(free) - 1)
            fk["_free_calls"] = fk.get("_free_calls", 0) + 1
            kw["free_query"] = lambda: free[k]
        else:
            kw["free_query"] = lambda: free
    if "peer" in fk:
        kw["peer_query"] = lambda: fk["peer"]
    return A.admit(cfg.own_cap_mib, other_jobs=other_units(Path(cfg.registry_dir), os.getpid()), **kw)


def device_memory() -> Any:
    """`fastdo.bench.device_memory`, loaded from its file: importing the `fastdo.bench` package pulls in torch
    (~5 s), which a unit must not pay before its admission check and lock."""
    name = "fastdo.bench.device_memory"
    if name in sys.modules:
        return sys.modules[name]
    alias = "fastdo_amortised_device_memory"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, Path(__file__).resolve().parents[1] / "bench" /
                                                      "device_memory.py")
        if spec is None or spec.loader is None:
            raise ImportError("fastdo/bench/device_memory.py not found")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    return sys.modules[alias]


def _nvml_query(fk: dict[str, Any]) -> Callable[[], str]:
    if "nvml_mib" in fk:
        return lambda: f"{os.getpid()}, {int(fk['nvml_mib'])}\n"
    return device_memory()._nvidia_smi


class UnitContext:
    """What a work function may use (module docstring, step 4)."""

    def __init__(self, cfg: UnitConfig, fk: dict[str, Any]) -> None:
        self.cfg = cfg
        self._fk = fk
        self.started = time.monotonic()
        self._last_ckpt = self.started
        self._last_check = self.started
        self.checkpoints = 0
        self.on_setup: Callable[[], None] | None = None
        self.t_lock: float | None = None

    def setup_done(self) -> None:
        """Mark the end of the work's setup (R3): the unit stamps `t_setup` in its status file at once."""
        if self.on_setup is not None:
            self.on_setup()

    def load_checkpoint(self) -> dict[str, Any] | None:
        p = Path(self.cfg.ckpt_path)
        return json.loads(p.read_text()) if p.is_file() else None

    def save_checkpoint(self, state: dict[str, Any]) -> None:
        _atomic_write(Path(self.cfg.ckpt_path), json.dumps(state))
        self._last_ckpt = time.monotonic()
        self.checkpoints += 1

    def checkpoint_due(self) -> bool:
        return time.monotonic() - self._last_ckpt >= self.cfg.checkpoint_s

    def time_up(self) -> bool:
        return time.monotonic() - self.started >= self.cfg.unit_s

    def tick(self, state: dict[str, Any] | None = None) -> None:
        """Re-check admission every `recheck_s`; when a rule fails, save `state` (the work's consistent state at this
        call) as the checkpoint and raise `TempFail` (the unit then exits 75)."""
        if not self.cfg.lock_path or time.monotonic() - self._last_check < self.cfg.recheck_s:
            return
        self._last_check = time.monotonic()
        d = _admission(self.cfg, self._fk)
        if not d.admit:
            if state is not None:
                self.save_checkpoint(state)
            raise TempFail("; ".join(d.reasons))

    def after_warmup(self) -> int | None:
        """NVML per-process total must be <= the cap once everything is allocated. Returns the
        reading (MiB; None when NVML reported no row for this process), so the work can record what the assert saw
        (a verdict on the memory bound needs the reading, not only the absence of an over-cap exit)."""
        got = device_memory().nvidia_smi_process_mib(None, _nvml_query(self._fk))
        if got is not None and got[0] > self.cfg.own_cap_mib:
            raise TempFail(f"NVML per-process {got[0]} MiB > cap {self.cfg.own_cap_mib} after warm-up")
        return None if got is None else int(got[0])


def _set_torch_fraction(cap_mib: int) -> None:
    """torch's per-process fraction = cap / card total, only when a GPU is visible (never imports torch otherwise)."""
    if _no_gpu_visible():
        return
    import torch

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mib / total))


def _resolve(work: str) -> Callable[..., str]:
    mod, fn = work.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def run_unit(cfg: UnitConfig) -> int:
    """The unit process body (module docstring, steps 1-5). Returns the exit status."""
    fk = _fakes(cfg)
    status: dict[str, Any] = {"name": cfg.name, "pid": os.getpid(), "phase": "start", "checks": [],
                              "utc": dt.datetime.now(A.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "t_start": time.monotonic(), "t_lock": None, "t_release": None,  # speed ledger
                              "t_setup": None,                                                    # whole-hold bound
                              "t_wait0": None, "lock_blocked_s": 0.0, "lock_source": None,        # blocking-lock wait
                              "lock_holders": None}

    def put(phase: str, **kw: Any) -> None:
        status.update(phase=phase, **kw)
        _atomic_write(Path(cfg.status_path), json.dumps(status, indent=1))

    shared = bool(cfg.lock_path)                     # opt-in: no shared lock and no admission without a tools dir
    if shared:
        d = _admission(cfg, fk, before_lock=True)
        status["checks"].append(d.summary())
        if not d.admit:
            put("refused")
            return EXIT_TEMPFAIL

    def waiting(holders: list[int], t_wait0: float) -> None:
        put("waiting_lock", t_wait0=t_wait0, lock_holders=holders)

    try:
        grant = (G.acquire(Path(cfg.lock_path), cfg.lock_wait_s, on_wait=waiting) if shared
                 else G.LockGrant(None, "none", None, time.monotonic(), ()))
    except G.LockTimeout as exc:
        put("lock_busy", reason=str(exc))
        return EXIT_TEMPFAIL
    fd = grant.fd                                    # None: an ancestor holds the lock for us (never queue behind it)
    status.update(t_lock=grant.t_lock, t_wait0=grant.t_wait0, lock_blocked_s=grant.blocked_s,
                  lock_source=grant.source, lock_holders=list(grant.holders_seen))
    put("locked")                                    # the hold's start is on disk before anything can kill it
    held = {"fd": fd}
    os.register_at_fork(after_in_child=lambda: _close_quietly(held.pop("fd", None)))    # module docstring
    reg = Path(cfg.registry_dir) / f"{os.getpid()}.json"
    try:
        if shared:
            d = _admission(cfg, fk)
            status["checks"].append(d.summary())
            if not d.admit:
                put("refused_after_lock")
                return EXIT_TEMPFAIL
        _atomic_write(reg, json.dumps({"pid": os.getpid(), "name": cfg.name}))
        put("admitted")
        os.environ["FASTDO_GPU_MEM_GB"] = repr(cfg.own_cap_mib / 1024)
        _set_torch_fraction(cfg.own_cap_mib)
        ctx = UnitContext(cfg, fk)
        ctx.t_lock = status["t_lock"]                # the hold's lock time (monotonic): GPU-s so far = now - t_lock
        ctx.on_setup = lambda: put(status["phase"], t_setup=time.monotonic())
        with device_memory().ProcessMemorySampler(interval_s=cfg.sampler_interval_s, stop_mib=cfg.own_cap_mib,
                                                  query=_nvml_query(fk)):
            put("running")
            try:
                result = _resolve(cfg.work)(ctx, **cfg.work_args)
            except TempFail as exc:
                put("tempfail", reason=str(exc))
                return EXIT_TEMPFAIL
        put("finished", result=result, checkpoints=ctx.checkpoints)
        return EXIT_DONE if result == "done" else EXIT_MORE
    finally:
        reg.unlink(missing_ok=True)
        held.pop("fd", None)                                             # a later fork must not close a reused fd
        if fd is not None:
            os.close(fd)                                                 # releases the lock (an ancestor's stays)
        status["t_release"] = time.monotonic()
        _atomic_write(Path(cfg.status_path), json.dumps(status, indent=1))

def _close_quietly(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m fastdo.amortised.unit <unit_config.json>", file=sys.stderr)
        return 2
    return run_unit(UnitConfig.load(Path(args[0])))


# ------------------------------------------------------------------------------------------------ driver


@dataclass
class DriveResult:
    done: bool
    reason: str
    launches: int = 0
    waits: int = 0
    wait_s: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


def _blocked_s(st: dict[str, Any], now: float) -> float:
    """Seconds the unit spent BLOCKED on gpu.lock (module docstring, "Blocking lock"): t_lock - t_wait0 once it holds
    the lock, now - t_wait0 while it still waits, 0 when it never waited."""
    w0 = st.get("t_wait0")
    if w0 is None:
        return 0.0
    end = st.get("t_lock")
    return max(0.0, (float(end) if end is not None else float(now)) - float(w0))


def _read_status(cfg: UnitConfig) -> dict[str, Any]:
    try:
        st = json.loads(Path(cfg.status_path).read_text())
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def _wait_unit(proc: subprocess.Popen, cfg: UnitConfig, t_launch: float, timeout: float) -> int | str:
    """Wait for one unit; SIGKILL its group (returning "timeout") past t_launch + blocked + timeout, or when it has
    been waiting for the lock longer than `lock_wait_s` + LOCK_GRACE_S (module docstring)."""
    while True:
        try:
            return proc.wait(timeout=STATUS_POLL_S)
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        st = _read_status(cfg)
        w0 = st.get("t_wait0")
        if st.get("t_lock") is None and w0 is not None and now - float(w0) > cfg.lock_wait_s + LOCK_GRACE_S:
            break
        if now > t_launch + _blocked_s(st, now) + timeout:
            break
    _kill_group(proc)
    return "timeout"


def _hold_record(cfg: UnitConfig, seq: int, rc: int | str, st: dict[str, Any], t_launch: float,
                 t_reaped: float) -> dict[str, Any]:
    """One launch's history row, carrying the speed-ledger hold times (`speed_ledger.holds_from_drive`).

    The unit stamps t_start / t_lock / t_release on the monotonic clock (CLOCK_MONOTONIC, system-wide on Linux, so the
    driver's own readings are on the same clock). A unit that held the lock but never wrote t_release (SIGKILLed by
    the per-hold timeout, `os._exit(3)` from the memory sampler, any other death) is charged up to `t_reaped`, the
    driver's reading after the process was reaped: the lock is released by the process's death, which precedes the
    reap, so this over-charges by at most the reap latency and never drops the hold (aborted holds count). A status
    that cannot be read at all from a unit that did not exit TEMPFAIL is charged launch to reap (fail closed;
    `release_source` 'driver_unknown'). `t_launch_eff` is the launch shifted by the blocking lock wait and `setup_s`
    counts from it (module docstring, "Blocking lock"); the unit's lock fields are copied as they are."""
    t_eff = t_launch + _blocked_s(st, t_reaped)
    row: dict[str, Any] = {"seq": seq, "name": f"{cfg.name}#{seq}", "rc": rc, "phase": st.get("phase"),
                           "reasons": st.get("checks", [])[-1:], "t_launch": t_launch, "t_launch_eff": t_eff,
                           "t_reaped": t_reaped, "t_start": st.get("t_start"), "t_lock": st.get("t_lock"),
                           "t_release": st.get("t_release"), "release_source": None,
                           "t_setup": st.get("t_setup"), "reason": st.get("reason"),
                           "setup_s": (st["t_setup"] - t_eff) if st.get("t_setup") is not None else None,
                           **{k: st.get(k) for k in LOCK_FIELDS}}
    if row["t_lock"] is not None:
        if row["t_release"] is None:
            row["t_release"], row["release_source"] = t_reaped, "driver_reap"
        else:
            row["release_source"] = "unit"
    elif st.get("phase") == "unknown" and rc != EXIT_TEMPFAIL:
        row.update(t_start=t_launch, t_lock=t_launch, t_release=t_reaped, release_source="driver_unknown")
    return row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    hostpaths.require(path, _HOST_VARS, create=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _done(cfg: UnitConfig) -> bool:
    p = Path(cfg.ckpt_path)
    try:
        return bool(p.is_file() and json.loads(p.read_text()).get("done"))
    except (OSError, ValueError):
        return False


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the unit's process group (it leads its own session: `start_new_session=True`), then reap it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def drive(cfg: UnitConfig, cfg_path: Path, *, max_launches: int, max_wait_s: float, backoff_s: float = 60.0,
          hold_timeout_s: float | Callable[[], float] = UNIT_MAX_S + 120.0, python: str = sys.executable,
          env: dict[str, str] | None = None, between_units: Callable[[], None] | None = None,
          sleep: Callable[[float], None] = time.sleep, hold_log: Path | None = None,
          on_hold: Callable[[dict[str, Any]], None] | None = None) -> DriveResult:
    """Launch fresh unit processes until the work is done or a budget runs out (module docstring). Every launch's
    hold times go into `DriveResult.history` (`_hold_record`) and, with `hold_log`, are appended to that JSONL file
    before the next launch (the status file is unlinked before every launch). `hold_timeout_s` may be a callable,
    read before every launch (R3); `on_hold(row)` is called with every row before the next launch."""
    cfg.dump(Path(cfg_path))
    res = DriveResult(done=False, reason="")
    seq = 0
    while True:
        if _done(cfg):
            res.done, res.reason = True, "done"
            return res
        if res.launches >= max_launches:
            res.reason = "launch budget spent"
            return res
        if between_units is not None:
            between_units()
        Path(cfg.status_path).unlink(missing_ok=True)
        timeout = float(hold_timeout_s() if callable(hold_timeout_s) else hold_timeout_s)
        t_launch = time.monotonic()
        proc = subprocess.Popen([python, "-m", MODULE, str(cfg_path)], env=env, start_new_session=True)
        rc: int | str = _wait_unit(proc, cfg, t_launch, timeout)
        t_reaped = time.monotonic()
        st = _read_status(cfg) or {"phase": "unknown"}
        row = _hold_record(cfg, seq, rc, st, t_launch, t_reaped)
        row["timeout_s"] = timeout
        seq += 1
        res.history.append(row)
        if hold_log is not None:
            _append_jsonl(Path(hold_log), row)
        if on_hold is not None:
            on_hold(row)
        waited = rc == EXIT_TEMPFAIL and st.get("phase") in WAIT_PHASES
        if waited:
            res.waits += 1
        else:
            res.launches += 1
        if rc == EXIT_OVER_CAP:
            res.reason = f"unit exceeded its memory cap ({cfg.own_cap_mib} MiB; status 3): resize before relaunching"
            return res
        if rc in (EXIT_DONE, EXIT_MORE, "timeout") and not waited:
            continue
        if rc == EXIT_TEMPFAIL:
            if res.wait_s + backoff_s > max_wait_s:
                res.reason = "wait budget spent"
                return res
            sleep(backoff_s)
            res.wait_s += backoff_s
            continue
        res.reason = f"unit failed with status {rc}"
        return res


# ------------------------------------------------------------------------------------------------ toy work


def _start_child(mode: str, pidfile: str) -> None:
    """Test hook: leave a 600 s child behind, exec'd ("exec") or forked without exec ("fork")."""
    if mode == "exec":
        pid = subprocess.Popen(["sleep", "600"]).pid
    elif mode == "fork":
        pid = os.fork()
        if pid == 0:
            time.sleep(600)
            os._exit(0)
    else:
        raise ValueError(f"unknown child mode {mode}")
    _atomic_write(Path(pidfile), str(pid))


def toy_counter(ctx: UnitContext, n_steps: int, seed: int, step_s: float = 0.0, marker: str = "",
                child: str = "", child_pidfile: str = "", stall_unit: int = 0, steps_per_unit: int = 0,
                setup_s: float = 0.0, slow_setup_unit: int = 0, slow_setup_s: float = 0.0) -> str:
    """A deterministic resumable work function for the unit tests: acc = sum of n_steps seeded normals. `child`
    ("exec" or "fork") leaves a sleeping child process behind, its pid written to `child_pidfile`. `stall_unit` k > 0:
    the k-th unit checkpoints after its first step and then hangs (600 s), for the driver's per-hold timeout.
    `steps_per_unit` > 0: a unit returns "more" after that many steps (a deterministic unit length).
    `setup_s`: a fake setup (sleep) before `ctx.setup_done()`; the `slow_setup_unit`-th unit's setup takes
    `slow_setup_s` instead (the whole-hold timeout)."""
    import numpy as np

    if child:
        _start_child(child, child_pidfile)
    st = ctx.load_checkpoint() or {"i": 0, "acc": 0.0,
                                   "rng": np.random.default_rng(seed).bit_generator.state, "units": 0}
    st["units"] = int(st.get("units", 0)) + 1
    if setup_s or slow_setup_unit:
        time.sleep(slow_setup_s if slow_setup_unit and st["units"] == slow_setup_unit else setup_s)
        ctx.setup_done()
    rng = np.random.default_rng()
    rng.bit_generator.state = st["rng"]
    if marker:
        Path(marker).write_text(str(os.getpid()))
    first = st["i"]
    while st["i"] < n_steps:
        ctx.tick(st)
        st["acc"] += float(rng.normal())
        st["i"] += 1
        st["rng"] = rng.bit_generator.state
        if st["i"] == 1:
            ctx.after_warmup()
        if stall_unit and st["units"] == stall_unit:
            ctx.save_checkpoint(st)
            time.sleep(600)
        if step_s:
            time.sleep(step_s)
        if ctx.checkpoint_due():
            ctx.save_checkpoint(st)
        if (ctx.time_up() or (steps_per_unit and st["i"] - first >= steps_per_unit)) and st["i"] < n_steps:
            ctx.save_checkpoint(st)
            return "more"
    st["done"] = True
    ctx.save_checkpoint(st)
    return "done"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
