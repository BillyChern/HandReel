# E1 amendment A-U: unseal at the registered 80% bar (2026-10-14)

**Written:** 2026-10-14T07:14:19Z (`date -u`). The commit that adds this file is the amendment record. It is committed
before any run of ours on a held-out unit, as the ruling requires ("recorded as amendment A-U, committed before our side
runs").

**Authority.** The authors' decisions of 2026-10-14 (internal record, not released), item 1. A-F
(2026-10-14-e1-amendment-A-F-freeze.md, this folder) names the freeze SHA, `fb6961213faf5600ba6c8a97acc31a6cc82eb038`.

## The unseal condition (A-U)

E1 unseals once BOTH of the following hold:

1. **Theirs' completeness bar.** Each corpus has at least 80% of its assigned units RUN by theirs. This is the registered
   completeness bar of prereg s. 6.4 and s. 10.4 (`e1_stats.MIN_RUN_FRACTION = 0.8`), unchanged. It means H ≥ 58 of 72
   (57/72 = 0.792 fails) and D ≥ 87 of 108 (86/108 = 0.796 fails). A unit is RUN when the status log status.jsonl holds a line
   for it with rc 0 or rc 1, delivered or failed. That is the scorer's rule (`e1_score.score_all`: run = has a status
   line). Units never started are NOT RUN.
2. **Ours is complete on every unit theirs RAN.** For each RUN unit, the last row of ours_EST.jsonl and of
   ours_CAL.jsonl was written at the freeze SHA, and neither is an infrastructure fault (`failure == "infra"`).
   Every other row is final: delivered, QC-rejected, a production refusal, or upstream refused (`upstream_missing`).
   An infrastructure fault is re-run (prereg s. 3.1, s. 10.4). While any RUN unit is unfinished on our side, E1 does
   not unseal.

The prereg s. 9 unseal preconditions also still apply, unchanged:
- (a) the freeze manifest is committed: 2026-10-14-freeze-manifest.md (this folder), in this commit;
- (b) every ours-side run is complete: condition 2;
- (c) the scorer and the analysis are committed, and the known answers pass.

Ruling 3 adds one ordering constraint: another study's FINAL prereg and its 48-unit subset draw are committed before the unseal.

Amendment A-N (2026-10-14-e1-amendment-A-N.md, this folder; the sensitivity analysis S-N) also
applies, unchanged by this amendment: its per-unit NMS-timeout record is committed after the unseal and before any
score.

## Theirs stops at the A-U point

- The ours-side queue (`run/study/e1_ours_queue.py` at the freeze SHA) evaluates conditions 1 and 2 after every turn.
  The first time both hold, it writes ours_AU.json with the time, the per-corpus counts and the RUN unit ids.
- The operator then stops theirs under the stop-record protocol (internal record, not released, R1). No
  new theirs unit starts. The operator writes the stop record e1_stop_confirmed.json only after both checks pass: the
  supervisor PID 3177825 is gone, and the last line of its log has been read verbatim.
- **Units in flight at the stop.** A theirs unit that was running at the A-U point counts only if it is sealed before the
  stop record is written. Sealed means its archive is in SEAL.sha256 and its status line is in status.jsonl. Such a unit
  is RUN. Ours runs it too: the queue keeps following status.jsonl until the stop record exists, and it exits only when
  A-U holds after that. A unit that is not sealed before the stop record is NOT RUN. It is listed and excluded from ITT
  and from complete-case, as in s. 10.4.
- **Final N** per corpus = the units RUN when the stop record is written. ours_AU.json records them again at exit
  (`final_run_units`).
- If theirs' calendar stop (2026-10-21T23:59Z, the supervisor) comes before condition 1 holds, the registered schedule
  stands: theirs resumes from 10-27 in the registered order, and A-U applies when the bar is reached.

## No interim look; the analysis at unseal is final

- No outcome of either method is read, scored or summarised before the unseal. Both conditions use run counts and our
  own completion only: status rc, row presence, the code SHA and the infra flag. They never use an error, a scale, a
  delivered fraction compared across methods, or a GT file.
- At the unseal there is ONE scoring pass of the committed scorer, then `e1_stats.summarise` (prereg s. 9, s. 6). Its
  verdicts are FINAL. After the unseal, no further E1 unit runs for either method, and there is still no early stop for
  success or futility.

## Unchanged

Everything else in the prereg is unchanged, including:
- units and run order; arms (O-EST primary, O-CAL secondary); the delivered rule and ITT bad values;
- the metric, endpoints, bootstrap, Holm on the calibrated thresholds, and INCOMPLETE below 80%;
- sealing, label-read logging and hygiene;
- the 185 GPU-h cap; the infrastructure-retry rules of s. 3.1, s. 3.2 and s. 10.4;
- the reporting of s. 11.

This amendment replaces only the dates of s. 10.3 for ours (10-27/28) and for the unseal (≥ 10-29), and the
pre-freeze sealing horizon of R-A ("until after our code/result freeze (~2026-10-26)"). The freeze is now A-F.

## Why this is outcome-independent

- The ruling fixes the condition, and the condition depends only on how many units ran, not on what they produced.
- At writing, the only E1 state read was status.jsonl as (unit, corpus, rc) counts: D 75/108, H 50/72 RUN.
- No sealed file and no ours-side held-out output exists or was read.
