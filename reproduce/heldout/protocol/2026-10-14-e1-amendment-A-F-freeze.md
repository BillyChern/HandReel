# E1 amendment A-F: the freeze SHA (2026-10-14)

**Written:** 2026-10-14T07:14:19Z (`date -u`). The commit that adds this file is the amendment record. It is committed
before any run of ours on a held-out unit: the ours-side output directory does not exist, and no ours-side row exists in the E1 run directory.

**Authority.** The authors' decisions of 2026-10-14 (internal record, not released), item 1: "The
reconstruction pipeline is FROZEN NOW, at the current production commit, recorded by amendment A-F." Prereg s. 3.1
(2026-10-02-prereg-e1-heldout.md, this folder): "The SHA is entered by amendment A-F before ours runs; A-F may not change
anything else." This amendment enters the SHA. It also records what the freeze excludes, as the ruling requires. It
changes nothing else.

## The freeze

- **Freeze SHA (SHA_F): `fb6961213faf5600ba6c8a97acc31a6cc82eb038`** (branch e1/freeze-1014; an internal commit, not in this release's history). Its parent
  was the main branch when the decision was made (fetched 2026-10-14T06:58Z; internal commit, not released).
- SHA_F adds two files to its parent: `run/study/e1_ours_queue.py` (the ours-side queue, an E1 harness file) and
  `tests/run/test_e1_ours_queue.py`. It changes no production file.
- **Production trees at SHA_F** (`git rev-parse SHA_F:<dir>`):

  | dir | tree |
  |---|---|
  | src/ | `701ad56d0f067788cedb89cd8aaf88e2946d7bcb` |
  | run/conf | `26c4dc647de8b3b164f5af8bee12e15c6a933492` |
  | run/pipeline | `b90a56b30d346350cf6aa8b910922ddb1501fc85` |

- **Relation to BASE** (an earlier internal commit; internal record "freeze ships BASE"). The three trees are identical
  in BASE, in SHA_F's parent and in SHA_F. Between BASE and that parent there are 36 commits. They touch 64 files, all
  in the paper and the internal records. The last commit to touch src/, run/conf or run/pipeline is a lint commit, which is
  an ancestor of BASE. There is therefore no difference to list. Freezing the main branch's production is the same as
  freezing BASE's production. The commit is SHA_F rather than BASE because the harness commit has to sit on top of it.
- **Config and rule.** `run/conf/config.yaml` sha256 `0c4585e81ac3365a9790e80b8181497c0f3bfa8d4035a44950582c09d2138e52`
  (inside the run/conf tree). Production QC AcceptanceRule `outputs/qc/acceptance_rule.json` sha256
  `e564b52e92d3a5ad6392734e36d23efda2d305aed5f2d6ce119271a2816150f0`. This is the prereg s. 3.1 rule (e564b52e92d3…),
  and the file is unchanged. The source-domain profiles, deterministic hands (M350) and every other default are
  taken as merged in those trees.
- **How ours runs.** From a detached worktree of SHA_F, pinned by SHA and never by a branch name.
  `run/study/e1_ours_queue.py --mode final --pin SHA_F` drives `run/study/e1_ours.py --mode final --pin SHA_F`
  (prereg s. 3.1: upstream, then O-EST, then O-CAL). Both refuse unless HEAD == SHA_F, and src/, run/conf and
  run/pipeline are clean. The queue also refuses unless the three trees equal the table above.

## What the freeze excludes

- Any reconstruction change committed after SHA_F never counts toward E1, whatever its result. This covers code,
  config, defaults, the QC rule, and model or weight swaps. Examples are the MV-SAM3D shape change, a later fix, and any
  study branch merged into main after this commit. Such changes are post-freeze ablations, evaluated on other data.
  They are never evaluated on the E1 units as E1.
- Opt-in features that are OFF in the trees above stay OFF for E1.
- The analysis code (e1_score.py, e1_stats.py) is unchanged by this amendment. A scorer bug found on development data
  stays governed by prereg s. 12.

## Why this is outcome-independent

- It was written before any ours-side held-out run, and no sealed archive was opened.
- The only E1 state read was the status log status.jsonl as counts of (unit, corpus, rc): theirs RAN D 75/108 and H 50/72,
  the same counts as at the ruling.
- The SHA was fixed by the ruling's "current production commit". Production at that commit is BASE's production.

## Schedule effect

The ruling replaces the prereg s. 10.3 rows "10-22 → 10-26 freeze manifest; amendment A-F" and "10-27 → 10-28 ours".
Ours now runs while theirs is still running, interleaved on the shared GPU lock (ruling 2). The unseal condition is
amendment A-U (2026-10-14-e1-amendment-A-U-unseal.md, this folder). The freeze manifest is
2026-10-14-freeze-manifest.md (this folder).
