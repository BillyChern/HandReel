"""Admission step (c): take the shared `gpu.lock` (in `$FASTDO_GPU_TOOLS_DIR`) by a BLOCKING flock, never queued behind
ourselves.

Our GPU jobs take turns with the other jobs on gpu.lock (`fastdo.amortised.admission.POLICY`). Those jobs wait in a
blocking flock (`flock gpu.lock ...`), so a job that only TRIES the lock (`flock -n`) almost never gets a turn: on a
release the kernel wakes the blocked waiters at once. `acquire` therefore blocks too, bounded by `wait_s` (SIGALRM on
the main thread; a 50 ms LOCK_NB poll elsewhere), and a timeout leaves nothing held.

Never queue behind ourselves: when the lock is held by this process or one of its ANCESTORS (the launcher was started
under `flock gpu.lock ...`), blocking would wait for our own wrapper forever. `acquire` then takes no lock of its own
and returns a grant whose `source` is "ancestor:<pid>" (the exclusion is the ancestor's, which outlives the unit).

Holders are read from /proc/locks (FLOCK lines on the lock file's device and inode; the "->" lines of blocked waiters
are skipped). `holder_mib` is the GPU memory (nvidia-smi compute apps) of the current holders' process trees, our own
lineage excluded: memory that is released with the lock, which admission may credit BEFORE the lock
(`admission.gather(holder_query=...)`); after the lock only the plain card free counts.
Anything unreadable gives no credit (0) and no ancestor match: fail closed.
"""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastdo import hostpaths

__all__ = ["LockGrant", "LockTimeout", "acquire", "ancestors", "holder_mib", "lock_holders"]

logger = logging.getLogger(__name__)

POLL_S = 0.05                       # LOCK_NB poll period off the main thread (no SIGALRM there)
MAX_DEPTH = 256                     # ancestor walk guard


class LockTimeout(Exception):
    """The lock stayed busy for the whole wait (nothing is held)."""


@dataclass(frozen=True)
class LockGrant:
    fd: int | None                  # our own lock fd (None: an ancestor holds the lock for us)
    source: str                     # "own" | "ancestor:<pid>" | "none" (no shared lock: `unit` without a tools dir)
    t_wait0: float | None           # monotonic start of the blocking wait (None: no wait)
    t_lock: float                   # monotonic time the lock was ours
    holders_seen: tuple[int, ...]   # the holders when we started waiting

    @property
    def blocked_s(self) -> float:
        return 0.0 if self.t_wait0 is None else max(0.0, self.t_lock - self.t_wait0)


def lock_holders(lock_path: Path, locks: Path = Path("/proc/locks")) -> list[int] | None:
    """PIDs holding a FLOCK on `lock_path`; [] when the file does not exist; None when /proc/locks is unreadable."""
    try:
        st = os.stat(lock_path)
    except FileNotFoundError:
        return []
    except OSError:
        return None
    try:
        text = Path(locks).read_text()
    except OSError:
        return None
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 6 or f[1] == "->" or f[1] != "FLOCK":
            continue
        try:
            maj, mnr, ino = f[5].split(":")
            if (int(ino) == st.st_ino and int(maj, 16) == os.major(st.st_dev)
                    and int(mnr, 16) == os.minor(st.st_dev)):
                out.append(int(f[4]))
        except ValueError:
            continue
    return out


def _ppid(pid: int, proc: Path) -> int | None:
    try:
        return int((proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def ancestors(pid: int, proc: Path = Path("/proc")) -> list[int]:
    """Parent, grandparent, ... of `pid` (stops at pid 0, an unreadable entry, or a loop)."""
    out: list[int] = []
    cur = pid
    for _ in range(MAX_DEPTH):
        pp = _ppid(cur, proc)
        if pp is None or pp <= 0 or pp in out or pp == pid:
            break
        out.append(pp)
        cur = pp
    return out


def _children_map(proc: Path = Path("/proc")) -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        pp = _ppid(int(p.name), proc)
        if pp is not None:
            kids.setdefault(pp, []).append(int(p.name))
    return kids


def _tree(root: int, kids: dict[int, list[int]]) -> set[int]:
    seen, todo = set(), [root]
    while todo:
        p = todo.pop()
        if p not in seen:
            seen.add(p)
            todo.extend(kids.get(p, []))
    return seen


def _compute_apps() -> dict[int, int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False, timeout=15).stdout
    apps: dict[int, int] = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            apps[int(parts[0])] = apps.get(int(parts[0]), 0) + int(parts[1])
    return apps


def _own_lineage(self_pid: int | None = None) -> set[int]:
    pid = os.getpid() if self_pid is None else self_pid
    return {pid, *ancestors(pid)}


def holder_mib(lock_path: Path, *, holders: Callable[[Path], list[int] | None] = lock_holders,
               apps: Callable[[], dict[int, int]] = _compute_apps,
               kids: Callable[[], dict[int, list[int]]] = _children_map, self_pid: int | None = None) -> int:
    """GPU MiB of the current lock holders' process trees (our own lineage excluded); 0 when nothing is readable."""
    try:
        hs = [h for h in (holders(Path(lock_path)) or []) if h not in _own_lineage(self_pid)]
        if not hs:
            return 0
        k = kids()
        pids = set().union(*(_tree(h, k) for h in hs))
        return int(sum(m for p, m in apps().items() if p in pids))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("lock-holder memory unreadable (no credit): %s", exc)
        return 0


def _close(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _block(fd: int, wait_s: float) -> None:
    """flock(LOCK_EX) bounded by `wait_s`; LockTimeout when it expires (the caller closes the fd)."""
    if threading.current_thread() is threading.main_thread():
        def _alarm(signum: int, frame: object) -> None:
            raise LockTimeout(f"gpu.lock still busy after {wait_s:.0f} s")

        old = signal.signal(signal.SIGALRM, _alarm)
        try:
            signal.setitimer(signal.ITIMER_REAL, float(wait_s))
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, old)
        return
    deadline = time.monotonic() + float(wait_s)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise LockTimeout(f"gpu.lock still busy after {wait_s:.0f} s") from None
            time.sleep(POLL_S)


def acquire(lock_path: Path, wait_s: float, *, on_wait: Callable[[list[int], float], None] | None = None,
            holders: Callable[[Path], list[int] | None] = lock_holders, self_pid: int | None = None) -> LockGrant:
    """Take gpu.lock (module docstring). Free: at once. Held by our own lineage: no wait, no fd ("ancestor:<pid>").
    Held by anyone else: `on_wait(holders, t_wait0)` is called, then a blocking wait of at most `wait_s` seconds
    (`wait_s` <= 0: no wait); LockTimeout when it expires. The fd is non-inheritable across exec (PEP 446)."""
    path = hostpaths.require(lock_path, "FASTDO_GPU_TOOLS_DIR", create=True)    # never creates /opt/fastdo
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return LockGrant(fd, "own", None, time.monotonic(), ())
    except BlockingIOError:
        pass
    except BaseException:
        _close(fd)
        raise
    hs = holders(path) or []
    mine = [h for h in hs if h in _own_lineage(self_pid)]
    if mine:
        _close(fd)
        return LockGrant(None, f"ancestor:{mine[0]}", None, time.monotonic(), tuple(hs))
    if wait_s <= 0:
        _close(fd)
        raise LockTimeout("gpu.lock busy and no wait allowed")
    t_wait0 = time.monotonic()
    try:
        if on_wait is not None:
            on_wait(list(hs), t_wait0)
        _block(fd, wait_s)
    except BaseException:
        _close(fd)                  # a timeout (or anything else) leaves nothing held
        raise
    return LockGrant(fd, "own", t_wait0, time.monotonic(), tuple(hs))
