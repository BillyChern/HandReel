# E1 amendment A-W (FINAL): the authors rule two "clean poll, no retry start" TIMEOUT events INFRA-UNRESOLVED (2026-10-22)

**Written:** 2026-10-22T11:28:49Z (`date -u`). The commit that adds this file is the amendment record. It is committed
before the harness change that lets the stop record carry this ruling, and before any stop record, au-reached or unseal
step.

**Revised 2026-10-23 (review fix round, READY-WITH-FIXES):** item 4 now states the theirs-side restriction in code and
the committed-doc requirement, and the "Unchanged" list names the two disk amendments. The scope (item 1), the class
(item 2) and the consequences (item 3) are unchanged.

**Outcome-free (prereg s. 12).** The inputs are the refusal text of `e1_at.py check` (rc 3, 2026-10-22T11:22:15Z;
its stderr file, internal, not released), which names units, poll utcs and the U-3 rule only, and the authors'
decision. No score, pose, error, delivered flag, sealed file or ours-side output was read. It is an infrastructure
amendment; it favours no side (the exclusion is paired, and the registered sensitivity counts both units as theirs'
failures).

## The authors' decision

- Decision approved by the authors on 2026-10-22 (11:1xZ).
- The approval accepted every recommendation then pending a decision (internal record, not released). Item 1 of that
  record, verbatim:

  > **E1 (A-V stop ruling):** the two A-T events whose retry never started before the calendar stop,
  > D:20201002_111644:841412060263 and D:20201002_113031:932122062010, are **INFRA-UNRESOLVED**: excluded from both arms
  > (paired), listed with walls and occupancy, and the registered sensitivity counts them as theirs' failures (A-T2 item 2).
  > The stop record may be written with that acknowledgement; the unseal follows the registered runbook after an
  > independent review of the stop record and pre-unseal records.

## Why an amendment is needed

- A-V ("Consequence under U-3") registered that a TIMEOUT whose only valid clean-now polls include a clean one, with no
  retry started before the calendar stop, is NOT INFRA-UNRESOLVED (a): the operator runs the retry, or the authors
  decide. The operator does not acknowledge such a unit with `--unresolved-ok`.
- Both units are in exactly that case. Their clean-now polls are clean records of 2026-10-17T06:18:59Z-06:19:00Z (taken
  while the retries awaited the authors' approval). No retry started before the calendar stop 2026-10-21T23:59:00Z.
  `e1_at.py check` therefore classes them `None` and refuses (rc 3), before `--unresolved-ok` is consulted; and
  `--unresolved-ok` accepts only INFRA-UNRESOLVED (a) with reason NO_CLEAN_RETRY.
- The authors have decided. The harness had no way to record such a decision; A-W adds one, limited to this case.

## A-W items (FINAL)

1. **Scope: exactly two unit ids, theirs side.**
   - D:20201002_111644:841412060263
   - D:20201002_113031:932122062010

   No other unit, and no other event class, is covered by this amendment.
2. **Class.** Each of the two events is **INFRA_UNRESOLVED** with reason **USER_RULED_NO_RETRY_START**. This is not
   NO_CLEAN_RETRY (A-T2 (a)): the poll evidence does not show that no clean window was seen. The class comes from the
   authors' decision, and the record says so.
3. **Consequences: those of A-T2 item 2, unchanged.**
   - Both units are EXCLUDED from BOTH arms of the primary analysis (paired removal), for O-EST and for S2 O-CAL.
   - They are listed in the INFRA-UNRESOLVED listing with their first walls, occupancy and poll history.
   - The registered sensitivity (A-T2 item 2) counts them as theirs' failures and reports the verdict both ways. If the
     two verdicts differ, the claim is reported as not robust (`not_robust_infra`, reported beside the headline; U-5).
4. **How the harness records it (test-first, after this commit).**
   - A new acknowledgement: `--user-ruled-unresolved <unit>` (one per unit, repeatable) plus `--ruling-doc <path>`, on
     `e1_at.py check` and on the `e1_unseal_day.py` stop-record step.
   - It is accepted ONLY for a THEIRS event in the U-3 case "a clean poll with no retry start", and ONLY when the
     ruling doc exists and names the unit. An ours-side event of a ruled unit is not covered and still refuses.
   - On the stop-record step the doc must also be a committed, clean file of the study repository (a relative path is
     resolved against the repo). Whole-id matching cannot read intent, so the doc must be a reviewed record.
   - It is refused for any other class or reason (RESOLVED, METHOD_FAILURE, INFRA_UNRESOLVED of either kind, PENDING,
     no poll, only errored polls, unknown TIMEOUT time), for a unit with no such event, for a unit the doc does not
     name, and for a missing doc.
   - The stop record and the A-T bookkeeping record the ruled units, the doc path and the doc's sha256. The stop
     record also records the doc's repo-relative path and the commit that added it.
   - The later unseal and verdict steps re-run A-T / A-T2 with the stop record's ruling. They refuse if the doc is no
     longer committed clean, or its sha256 or adding commit has changed.
   - Runbook step 7 rebuilds the ruling from the stop record (`$RULED`).
   - `--unresolved-ok` is unchanged.
   - The operator passes this file as the ruling doc:
     `--user-ruled-unresolved D:20201002_111644:841412060263 --user-ruled-unresolved D:20201002_113031:932122062010
     --ruling-doc <this file>`.
5. **Order.** The stop record and the unseal follow the registered runbook (internal record, not released) only
   after an independent review of A-W and its code. The unseal also requires an independent review of the stop record
   and the pre-unseal records (item 1 of the decision record).

## Unchanged

Everything else in the prereg stays as registered, and so does everything in:
- the disk amendments of 2026-10-06 (2026-10-06-e1-amendment-disk-floor.md, superseded) and 2026-10-07
  (2026-10-07-e1-amendment-no-disk-guards.md);
- A-F, A-N, A-T, A-T2, A-U and A-V;
- the A-T ours-operational note and the unseal-prep rulings U-1..U-8.

In particular:

- **The data.** Units, order, arms, the RUN set and the A-U 80% bar: H 58/72, D 87/108, with A-V's 35 start refusals
  NOT RUN.
- **The analysis.** Metric, bootstrap, Holm, thresholds, INCOMPLETE, the S-N rule, the headline rule (U-5), the one
  scoring pass, and no interim look.
- **Other events.** The other three A-V TIMEOUT events follow A-T / A-T2 as before. The one-retry cap, the clean-window
  start (U-4), the occupancy record and the A-U calendar stop are unchanged.
- **The U-3 rule** for every other unit. A clean poll with no retry start still refuses unless the authors rule, by unit
  id, in a doc passed with `--ruling-doc`. INFRA-UNRESOLVED (a) still needs per-unit, all-not-clean poll evidence and
  `--unresolved-ok <unit>`.
- **The A-V items:** NOT RUN start refusals, the spanning-unit sensitivity and the disclosure.
