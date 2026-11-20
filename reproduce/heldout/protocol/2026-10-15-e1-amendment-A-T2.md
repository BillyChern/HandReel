# E1 amendment A-T2 — retries run only in a clean GPU window; an unresolvable TIMEOUT is excluded from BOTH arms (2026-10-15T00:36Z)

Builds on A-T (2026-10-14-e1-amendment-A-T.md, this folder). This is infrastructure-only: it is informed by status lines and by host and GPU
occupancy, never by any pose, error or score. It favours fairness to the baseline.

## Evidence (status lines and occupancy only)

- Four TIMEOUTs on theirs so far:
  - D 07:57Z
  - H 16:00Z
  - D 20:23Z (10-14)
  - D 00:32Z (10-15)
- The last two D units held gpu.lock for the whole 2.0 h at moderate CPU load. Throughout, another tenant's training
  job (<another tenant's job>) held about 42 GB of the card and computed on it. The likely cause is GPU time-slicing
  with a foreign process, which neither side controls.
- Under A-T as written, a retry in the same conditions would probably time out again and then count as THEIR method
  failure. That would be a host artefact biased toward us.

## Change

1. **Clean window.** A-T retries of theirs (and of ours, should one occur) start only when nvidia-smi shows no compute
   process from any other tenant. The operator polls, and the wait budget runs to the calendar stop. Our own E1
   queue and M9 are paused for the retry by holding gpu.lock in the normal way.
2. **Unresolvable.** A TIMEOUT unit whose retry cannot start in a clean window before the A-U stop, or whose retry
   also times out with a foreign GPU process recorded during the attempt, is classed **INFRA-UNRESOLVED**:
   - It is EXCLUDED from BOTH arms of the primary analysis (paired removal).
   - It is listed in the paper with its walls and the occupancy record.
   - A sensitivity analysis counts these units as theirs' failures and reports the verdict both ways. If the two
     verdicts differ, the claim is reported as not robust.
3. **Retry cap.** Still one retry per unit, as in A-T.
4. **Occupancy record.** From this amendment on, the operator logs per-minute GPU occupancy to
   an occupancy log, gpu_occupancy.jsonl: PIDs, memory, and the owning tenant (from cwd). This is infrastructure, not an outcome.

Committed BEFORE any retry runs. The retries themselves still await the authors' approval (the A-T requirement).
