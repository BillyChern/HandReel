# E1 infrastructure amendment: no disk guards (2026-10-07T05:23:02Z)

**Type:** infrastructure only. Recorded before any held-out output exists: the E1 run directory holds no sealed archive and the sealed directory is empty.

**Authority:** amendment approved by the authors on 2026-10-07T05:20Z: remove every disk guard of the project and proceed.

**Change:**
- `run/study/e1_common.py`: DISK_FLOOR_GB (150 GB, from the 2026-10-06 amendment) and UNIT_RESERVATION_GB (15 GB) are removed. `disk_ok()` now always returns True; it is kept only so legacy callers still import.
- `run/study/e1_theirs.py`: the queue no longer stops with rc 3 on low disk, and a unit no longer refuses to start. The dry run prints free space for information only.
- `run/study/e1_ours.py`: the start-time disk refusal is removed.
- The supervisor (theirs_supervisor.sh, not tracked) is re-pinned to this commit. Its rc-3 "disk" branch can no longer fire.

**Unchanged:**
- every registered quantity: units, order, arms, budget (185 GPU-h), sealing, the OOM retry and INFRA_OOM rule, scoring, and the unseal date;
- the supervisor's GPU-memory admission gate (≥30 GB free on the card), which guards card memory, not disk;
- the calendar stop (10-21T23:59Z).

**Risk accepted by the authors:** if the volume fills, a unit can fail on a write (ENOSPC). Such a failure is recorded as an infrastructure failure like any other unit failure. It is not retried silently.

Supersedes 2026-10-06-e1-amendment-disk-floor.md (this folder).
