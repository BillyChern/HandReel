# E1 amendment A-V (FINAL): the foreign edit to DO AS I DO's tree, the 35 start refusals, the 5 TIMEOUT retries (2026-10-19)

**Written:** 2026-10-19T07:27Z (`date -u`). The commit that adds this file is the amendment record. It is committed
before any A-T retry runs, before any A-V code change lands on main, and before any unseal step. It replaces the draft
of A-V (internal record, not released), which
stays as the incident record.

**Outcome-free (prereg s. 12).** The inputs are theirs' status-line fields (unit, corpus, rc, failure class, stage,
wall_s, utc, archive_attempt), file metadata (names, sizes, mtimes, sha256 of DO AS I DO's own config file), the process
table, nvidia-smi, git state of third_party, and the operator's own clean-now records (utc, unit, side, clean,
process owners). No score, pose, error, delivered flag, sealed file or ours-side output was read. It is an
infrastructure amendment; it favours no side (items 2 and 3 apply to both arms by paired removal).

## Decisions (as recorded; not reopened)

- **Authors, 2026-10-19T07:18Z:**
  - the authors ran the paths.sh restore (`cp .../paths.sh.bak.1016 .../paths.sh`);
  - approved by the authors: the paths.sh restore and the 5 retries;
  - other approvals given at the same time concern later work and are not listed here;
    none of them touches E1.
- **Operator, 07:2xZ: second step of the restore (the A-V addendum).**
  - The untracked `paths.sh.bak.1016` was moved out of the tree to
    <host path>, and kept as evidence.
  - `paths.sh` is byte-identical to that backup and to git HEAD.
  - `e1_theirs.third_party_state()` then returned ok=true: head 824591b8, porcelain = KNOWN_DIRTY only.
- **Operator decision, A-V item 2: option (b).**
  - The 35 units are NOT RUN, because they failed at stage `start` before any DO AS I DO code ran.
  - This is decided from the failure class alone, so it is outcome-independent. Their paired rows on our side are
    removed, as for any not-run unit.
  - Why: (a) would cost about 50-90 GPU-h and ~3 days, and both corpora already sit at the A-U 80% bar.
- **A-V item 5 (sensitivity) and item 6 (disclosure) apply as drafted.**

## Re-check of the restore (this amendment, 2026-10-19T07:24:50Z)

Run read-only from theirs' pinned runner tree (<host path>, an internal commit, `git status` clean):

```
e1_theirs.third_party_state() -> {"head": "824591b808c342b20079c3b4198a8c2bdf88c74e",
                                  "porcelain": ["M reconstruction/modules/HaWoR"], "ok": true}
```

- `M reconstruction/modules/HaWoR` is the registered KNOWN_DIRTY entry (e1_theirs.py:56). Inside it, the nested
  `thirdparty/DROID-SLAM/thirdparty/lietorch` shows as modified; this is what makes HaWoR `M`, it is covered by
  KNOWN_DIRTY, and its origin is not established (draft, "What happened"). It is not touched.
- sha256 `0e93de57…7dfef` for all three: `reconstruction/config/paths.sh` (mtime 2026-10-19T07:16:21Z, the authors'
  restore), the kept backup paths.sh.bak.1016, and `git show HEAD:reconstruction/config/paths.sh`.
- `git diff HEAD -- reconstruction/config/paths.sh` is empty; no untracked file remains in DO AS I DO's tree.
- Evidence kept (internal, not released): paths.sh.bak.1016 (mtime 2026-10-16T07:43:00Z) and an incident directory
  (the foreign version `paths.sh.foreign_20261016T0743Z`, its diff, a second copy of the backup, SHA256SUMS).

## A-V items (FINAL)

1. **Restore (done).** DO AS I DO's tree is in its registered state (above). Nothing else in third_party is modified by
   this amendment or by its code.
2. **The 35 start refusals are NOT RUN (option (b)).** A theirs status line whose failure class starts with `INFRA` and
   whose stage is `start` ran no DO AS I DO code (e1_theirs.run_unit raises at its pre-stage check). Such a line:
   - does not make its unit RUN, for the A-U bar, for the stop record's RUN set, and for the scorer's `run` flag;
   - is not an attempt for A-T / A-T2 bookkeeping: a retry refused at `start` is not "the one retry" (A-T2 item 3),
     and the unit's next genuine attempt is its retry;
   - an `INFRA*` class at any later stage is unchanged (a prereg s. 3.2 infrastructure retry or failure, never NOT RUN).
   With the 35 removed, theirs RAN H 58/72 and D 87/108 (0.806 each), at the A-U bar. Their rows on our side are
   removed pairwise. Completeness is the registered one (RUN / assigned), unchanged.
   - The ours queue at SHA_F counted the 35 as RUN (its rule predates A-V) and ran them; that is harmless, as our rows
     for NOT RUN units are dropped. Its `final_run_units` therefore equals the stop record's RUN set plus exactly the
     A-V refused units; the unseal-day check compares the two modulo exactly that set and refuses any other difference.
3. **The 5 TIMEOUTs follow A-T / A-T2 unchanged.** One retry each, started only in a clean window (in-lock clean-now
   record for that unit and side, U-4), occupancy logged per minute (A-T2 item 4; logger PID 1511633 is running), under
   gpu.lock, through the registered runner on theirs' pinned tree (`e1_theirs.py --mode final --unit <id> --retry`, at
   niceness 0 as A-T allows). The five events (status lines; theirs side):
   - D:20201002_112404:932122062010
   - H:bench_hoi4d_ZY20210800002_H2_C12_N25_S157_s02_T1_f034
   - D:20201002_112114:932122062010
   - D:20201002_111644:841412060263
   - D:20201002_113031:932122062010
4. **Code, test-first, before the unseal** (as drafted): `INFRA_*` failure classes at stage `start` count as NOT RUN in
   the A-U / stop-record RUN set, the scorer, e1_at and e1_verdicts, with pairwise removal; a mutation test shows that
   the old rule (only a literal `"infra"` is infrastructure; every status line makes its unit RUN) fails.
   - Disclosure: code for items 2, 5 and 6 was drafted on branch e1/unseal-prep (two internal commits, 2026-10-17T02:35Z
     and 02:40Z) against the DRAFT, after its item 4. It was not on main and no retry, stop record or unseal used it.
     It is completed (e1_at; the ours-exit comparison of item 2) and re-verified test-first after this commit, and
     merged with the registered unseal harness (runbook step 0: main with e1/unseal-prep merged).
5. **Sensitivity (as drafted):** the primary result is also reported WITHOUT the one D unit whose run window spanned
   2026-10-16T07:43:00Z (status utc - wall_s < 07:43:00Z < utc). Descriptive, flagged beside the primary
   (`without_av_spanning`, `differs_without_av_spanning`).
   - This is also the prereg s. 10.4 rule for "a deviation from section 3 found after the fact": prereg s. 3.2 says a
     third_party change between a unit's before/after `git status` fails the unit, but the pinned runner records
     `third_party_after` without failing the unit (e1_theirs.py:323, 382). The spanning unit is the only one affected;
     its verdict is given with and without it.
6. **Disclosure (as drafted):** the paper and the verdict record state the incident, the 35 units (21 D, 14 H) and the
   choice of (b) (`av_refused_units` and one `disclosures` line in the verdict record).

## Ruling: the stop that bounds the retry wait has NOT passed (written before acting on it)

- **A-U** defines the stop as the operator's stop record (e1_stop_confirmed.json), written only after the
  supervisor is gone and its log's last line is read. **A-T** item 2: retries happen before the A-U stop record.
  **A-T2** item 1: "the wait budget runs to the calendar stop"; item 2: a retry that cannot start in a clean window
  before the A-U stop is INFRA-UNRESOLVED (a).
- Facts (file metadata and the process table only, 2026-10-19T07:2xZ): e1_stop_confirmed.json does NOT
  exist. The supervisor (restart PID 45729) is gone: it exited on its own at 2026-10-16T09:36:32Z after "all units
  done", because every unit then had an archive (the 35 refusals included). That exit is not the stop record. No
  e1_theirs.py process is alive.
- **Ruling:** the governing bound is A-T2's calendar stop, **2026-10-21T23:59:00Z** (theirs_supervisor.sh
  `STOP_EPOCH`; A-U "theirs' calendar stop"). It has not passed. The operator writes the stop record when every one of
  the 5 events has its retry sealed, or at 2026-10-21T23:59:00Z, whichever comes first; a retry still running then
  delays the stop record (runbook step 3 refuses while any e1_theirs.py process is alive). A retry may START only before
  2026-10-21T23:59:00Z.
- **Consequence under U-3 (found while writing this).** at_clean_checks.jsonl holds five clean-now records at
  2026-10-17T06:18:59Z-06:19:00Z, one per TIMEOUT unit (side theirs), all `clean: true`, with no retry started (the
  retries then awaited the authors' approval). U-3: a clean poll with no retry start is NOT INFRA-UNRESOLVED (a); the
  operator runs the retry, or the authors decide. So none of the 5 can become INFRA-UNRESOLVED (a) on poll evidence
  alone. If a unit's retry cannot start in a clean window before the calendar stop, the stop record refuses for it and
  the authors decide on that unit; the operator does not acknowledge it with `--unresolved-ok`. A retry that starts
  cleanly and times out again with a foreign process recorded is INFRA-UNRESOLVED (b) by the code (A-T2 item 2).
- INFRA-UNRESOLVED units are excluded from both arms (paired), listed with walls and occupancy, and the sensitivity
  counts them as theirs' failures (A-T2 item 2).

## Unchanged

Everything else in the prereg and in A-F, A-N, A-T, A-T2, A-U, the A-T ours-operational note and the unseal-prep
rulings U-1..U-8 is unchanged: units, order, arms, metric, bootstrap, Holm, thresholds, INCOMPLETE, the S-N rule, the
headline rule, the one scoring pass, and no interim look.

## Operational notes for the unseal (the unseal runbook, internal record, not released, on e1/unseal-prep)

- The supervisor that ran to the end is the 10-16 restart, PID 45729 (theirs_supervisor.pid), not 3177825.
  The stop-record and au-reached steps take `--supervisor-pid 45729`; the supervisor log is
  theirs_supervisor.log (theirs_supervisor.sh `SLOG`).
- The ours queue (PID 46175) is not alive (the host reboot of 2026-10-16T17:33Z). Should check-ours-exit find that it
  never saw the stop record, the operator relaunches it once from its recorded launch line at SHA_F after the stop
  record exists. With every RUN unit complete it starts no GPU work; it writes `final_utc` and `stop_record_seen` and
  exits (e1_ours_queue at SHA_F, the A-U branch of `queue`).
- Prereg s. 12's amendment index gains A-T2 and A-V when this branch merges (runbook "Found while preparing" item 9).
