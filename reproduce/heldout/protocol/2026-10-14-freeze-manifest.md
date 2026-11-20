# Freeze manifest: the reconstruction pipeline as frozen for E1 (2026-10-14)

**Written:** 2026-10-14T07:14:19Z (`date -u`). Authority: the authors' decisions of 2026-10-14, item 1 (internal record, not released);
E1 amendments A-F and A-U (2026-10-14-e1-amendment-A-F-freeze.md and
2026-10-14-e1-amendment-A-U-unseal.md, this folder). This replaces the 10-22 → 10-26 freeze manifest of the earlier freeze plan (one venue name removed from this public copy)
(internal record, not released, s. 6 "Freeze manifest and Gate A"). For item 5 of that plan it records the parts
that apply now, and it lists what is deferred.

## Recorded

| item | value |
|---|---|
| freeze commit (SHA_F) | `fb6961213faf5600ba6c8a97acc31a6cc82eb038` (branch e1/freeze-1014; its parent = the main branch at the decision; internal commits, not in this release's history) |
| src/ tree | `701ad56d0f067788cedb89cd8aaf88e2946d7bcb` |
| run/conf tree | `26c4dc647de8b3b164f5af8bee12e15c6a933492` |
| run/pipeline tree | `b90a56b30d346350cf6aa8b910922ddb1501fc85` |
| identical to | BASE (an earlier internal commit; internal record "freeze ships BASE") and SHA_F's parent; these differ only in the paper and the internal records (36 commits, 64 files). The main branch as fetched 2026-10-14T07:15:57Z still has the same three trees: it added only internal records (A-N, the mandate audit) |
| last production commit | a lint commit (internal), an ancestor of BASE |
| run/conf/config.yaml sha256 | `0c4585e81ac3365a9790e80b8181497c0f3bfa8d4035a44950582c09d2138e52` |
| QC AcceptanceRule (outputs/qc/acceptance_rule.json) sha256 | `e564b52e92d3a5ad6392734e36d23efda2d305aed5f2d6ce119271a2816150f0` (= prereg s. 3.1) |
| contact constants | `src/fastdo/contact/constants_production.json` (config `contact_stage.constants: null`), inside the src/ tree |
| main interpreter | `<host path>`, Python 3.12.3; `uv pip freeze` (119 lines) sha256 `ec40f62880729d9f801011a73f9d72e2b3d1faea3106de83cdaaa0e5467221d2`. fastdo is imported from the pinned tree through `PYTHONPATH=<tree>/src`. Checked: it resolves to `<host path>` and not to the venv's editable install of main |
| SAM 3D worker | a wrapper script, sam3d_python_pylib.sh → `<host path>` (pylib + pylib_dgr on PYTHONPATH); `pip freeze` (281 lines) sha256 `71131da45b594e433df7ccbb86ab1b9d2d681c7e3ff35920978f16704cc4efe8` |
| E1 units / thresholds | the units file (prereg s. 2.2) sha256 `72465b68c75d87b593bc4a67332963d0e38906bd62c89565b43c3b5d007524fb`; the size record (released as ../e1-power-size.json) sha256 `def45c58916f0e8bdebe2d9e49440beb4d4487614693443334b412e79d861a65` (unchanged) |
| ours-side harness | `run/study/e1_ours.py` (unchanged since registration) driven by `run/study/e1_ours_queue.py` (new in SHA_F), from a detached worktree of SHA_F |

## What the freeze covers

- **Covered:** the reconstruction pipeline, meaning every file in the three trees above and the QC rule. This is the
  code and configuration that E1's ours side runs (prereg s. 3.1), and that any paper number labelled "production at
  the freeze" must come from.
- **Not covered (excluded):** every reconstruction change after SHA_F. Examples: the MV-SAM3D shape change, a later fix, and
  study-branch merges into main. Such changes are post-freeze ablations on other data and never count toward E1. Also
  not covered: non-reconstruction work (the physics, retargeting and scaling studies) and the paper text.

## Not recorded

- Model weight files are resolved by code defaults in src/ and by the envs. Their checksums are NOT recorded here.

## Freeze plan item 5 (the headline protocol): deferred, separate

None of item 5 is part of this freeze record. Each sub-item is separate and still open:
- **The clean-CPU timing headline:** at least 4 admitted 72-window passes under the admission rules (throttled_frac
  ≤ 0.05, foreign CPU ≤ 1 core, reference request ≤ 1.15x, memory PSI ≤ 1e-3). It needs its own prereg and its own
  admitted window, and it runs on the frozen trees above. The authors' decisions of 2026-09-28 make headline numbers clean-CPU only, with
  speed reported floor-primary. This manifest makes no speed claim.
- The CALIB rehearsal (719 spent windows, engine.prune on).
- The S2 TEST one-shot with its reporting-only extension prereg.
- Reconciling S2's registered B = 564.5 with the 92x headline.
- The marginal cost c from the flow-shop replay.

When these run, they run at SHA_F's trees, or they state the tree they used and why.
