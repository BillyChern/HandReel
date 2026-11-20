# E1 amendment A-N — theirs' detection-timeout exposure: pre-specified sensitivity analysis S-N (2026-10-14T07:11Z)

Prereg: 2026-10-02-prereg-e1-heldout.md (this folder). This amendment is outcome-free (section 12): it uses no E1 output of either
method. Its inputs are the status lines' wall times (allowed for infrastructure), the host load, the pinned code of both
pipelines, and our own M347/M350 finding. The prereg's section-12 list must cite it when the A-F/A-U branch
(e1/freeze-1014) merges.

## Threat (found 2026-10-14 by the mandate audit)

- DO AS I DO's hand stage (HaWoR, env `hawor`, ultralytics 8.4.36) runs YOLO NMS with a wall-clock limit:
  `time_limit = 2.0 + 0.05 x batch` s. On a timeout it logs `NMS time limit ... exceeded` and **returns the
  detections found so far** (ultralytics/utils/nms.py:81, 162-164). Detections are lost silently.
- `run/study/e1_theirs.py:445` (the e1-run pin, an internal commit) launches every unit under `nice -n 19`. Host at the audit:
  load average 32 on a 17-core cgroup quota, with other tenants at nice 0. A nice-19 thread gets about 1/68 of a
  nice-0 thread's CFS weight. Recent theirs walls are D 1.09 h and H 2.01 h mean over the last 40 units. The last
  H units took 2.4-2.75 h.
- Our production runs the same NMS **without** the limit (`run/conf/config.yaml:172`, `hands.deterministic: true`,
  internal record). M347 found the limit firing in our own pipeline under driver stalls before M350 made it the default.
- So CPU starvation can degrade theirs and cannot degrade ours. **The bias favours us.** A reviewer would be right to
  ask for this check.

## Change (pre-specified sensitivity analysis S-N; the primary analysis is unchanged)

At the unseal, **before any score is computed**:

1. For every unit and each side, count the lines containing `NMS time limit` across all of that unit's stage logs.
   - Theirs: `*.out.log` and `*.err.log` in the unit's archive, all attempts of the attempt that is scored.
   - Ours: the same over our stage logs. Expected 0, because ours runs without the limit; verify that the run
     config had `hands.deterministic: true`.
   - A unit is **N-affected** if either side's count is >= 1.
   - Record per unit: (unit, corpus, theirs count, ours count). Commit the record before scoring.
2. Report, per corpus: the number of N-affected units out of the units analysed.
3. If a corpus has >= 1 N-affected unit, re-run the registered primary analysis on that corpus's **unaffected** units
   only. It uses the same `e1_stats` code, margin log 1.15, bad values, resampling, B and seed. This is S-N.
4. **Claim rule (conservative):**
   - A corpus' NON-INFERIOR claim enters the paper's headline only if the primary verdict AND the S-N verdict are
     both NON-INFERIOR.
   - If S-N gives anything else, the paper reports the primary verdict, puts S-N beside it, and states that the
     claim is not robust to the baseline's timing-dependent detection loss. This includes INCONCLUSIVE because of the
     smaller n.
   - If no unit of a corpus is N-affected, S-N equals the primary and no extra rule applies.
5. Also report the median theirs wall per corpus for N-affected versus unaffected units. This is descriptive only.

## Proposed infrastructure change (NOT applied; needs the authors' approval)

The automated permission policy refused a helper that would keep theirs' process tree at niceness 0. That helper
would have changed no code, unit, order, output or seal, and sent no signals. If the authors approve, it is applied
from the approval time, and S-N still covers every unit, before and after the change. Without it, S-N alone protects
the claim, and theirs keeps running at nice 19. That is slower, so the unseal comes later.
