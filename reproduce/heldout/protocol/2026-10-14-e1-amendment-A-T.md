# E1 amendment A-T — a TIMEOUT under host contention is an infrastructure fault: one retry, scored on the retry (2026-10-14T08:06Z)

Prereg: 2026-10-02-prereg-e1-heldout.md (this folder), sections 3.2 and 12. This is an **infrastructure** amendment, informed
only by status lines (wall_s, failure class), which section 12 allows. It uses no pose, error or score of either method.
It favours the BASELINE. Companion: A-N (2026-10-14-e1-amendment-A-N.md, NMS-timeout sensitivity).

## Evidence (status lines only)

- The registered timeout is 3x the expected wall: D 2.0 h, H 3.6 h (section 3.2). That margin was set for an
  uncontended host.
- Median theirs wall per day (rc 0 units): 10-08 D 0.57 h / H 1.27 h; 10-09 D 0.60 / H 1.25. Later days: 10-11 D 1.10
  / H 2.28; 10-12 D 1.41 / H 2.72; 10-13 D 1.16 / H 2.12; 10-14 D 1.64 / H 2.68. That is about 2x slower. The cause is
  host CPU contention: load up to 52 on a 17-core quota, our other study jobs at nice 10-15 until 07:15Z today, and other
  tenants at nice 0. Meanwhile e1_theirs.py:445 pins every unit to nice 19.
- 2026-10-14T07:57:56Z: D:20201002_112404:932122062010 failed with `TIMEOUT` at s3_track (wall 7205 s). This is the
  first TIMEOUT of the run (1 of 21 failures).

## Change

1. From this amendment on, a `TIMEOUT` on either side is classed as an **infrastructure fault**, as section 3.2
   already does for OOM, host crash and disk guard. It gets one retry and counts as a method failure only if the retry
   also times out. This covers the unit above and any later TIMEOUT.
2. The retry runs before scoring, through the registered runner on the pinned tree:
   `e1_theirs.py --mode final --unit <id> --retry` under gpu.lock with FASTDO_STAGE=e1_theirs, so it is the next
   archive attempt. Nothing else changes: same code, inputs, N-prime rule and timeout.
   - It may run at niceness 0 instead of the queue's 19. The prereg does not register niceness.
   - The retry's archive is the scored attempt.
   - Retries happen before the A-U stop record. A retry still running at the A-U point delays the unseal until it
     finishes.
3. Reporting: the paper lists every TIMEOUT unit, its first and retry walls, and the verdict with and without the
   retried units. The latter is a descriptive sensitivity; the primary uses the retry.
4. The RAN count is unchanged: the unit was RAN with rc 1 and stays RAN.

Status: committed BEFORE any retry runs. Running the retry touches the sealed run, so the operator acts only after
the authors approve.
