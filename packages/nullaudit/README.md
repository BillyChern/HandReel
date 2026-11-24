# nullaudit

`nullaudit` checks whether a physics success test for hand-object trajectories passes **null** trajectories: object
motion that a hand-borne grasp did not have to produce. The core needs only numpy and runs on CPU. An optional
MuJoCo adapter runs the two nulls that need a simulator.

You supply:
- a reference object trajectory, `(T, 7)` (x, y, z, qw, qx, qy, qz) or `(T, 4, 4)`;
- a success-test callable that scores an executed object trajectory;
- optionally, a simulator callback, the clip's stored controls and the release row.

Three input cases are refused or reported, never scored as if they were valid. A `(T, 4, 4)` trajectory whose
rotation block is not a proper rotation (a zero block, a reflection, a scaled matrix) has no quaternion and raises.
The DO AS I DO object test raises when a compared reference row has a zero or non-finite quaternion, which the
component-wise midpoint of `+q` and `-q` produces; resample such a reference with a sign-consistent interpolation
first. With the release at row 0 no command precedes the release, so the plan-then-freeze null is reported
`not_run` with that reason and the other nulls keep their outcomes.

The audit builds the standard nulls, scores each with your test, and flags every **Z0** clip: a clip where the
static-at-start null passes. The headline is each null's pass rate, and it is always printed beside the Z0 base rate.

## Install

```bash
pip install ./packages/nullaudit            # numpy only
pip install "./packages/nullaudit[mujoco]"  # adds the MuJoCo adapter
```

## The pitfall, stated formally

This section derives the pose-test thresholds. It is not a reported result. For the checks on user-supplied runs and
their interpretation, see `docs/audit.md`.

**Notation.**
- Executor rows t = 0..n-1.
- Warm-up rows: t < W. During the warm-up the object is welded at the reference start pose r_0.
- Post-release rows: t >= W. The full clip has T rows.
- x_t = (p_t, q_t) is the executed object pose; r_t = (p^r_t, q^r_t) is the reference pose.

**C_D, DO AS I DO's object test** (paper section 4.1; the released code returns only the errors):

    E_pos = (1/n) sum_{t<n} |p_t - p^r_t| < 0.1 m   and   E_rot = (1/n) sum_{t<n} ang(q_t, q^r_t) < 0.5 rad,

The means run over the EXECUTED rows, warm-up included. A stricter reading also requires n >= T (`required_rows`).

**Null trajectory.** A post-release trajectory that a hand-borne grasp did not have to produce:
- (N0) no hand-object contact force on any post-release row: do nothing, hand absent, or open hand;
- (N1) a frozen hand command, u_t = u_{W-1} for t >= W (plan-then-freeze): no command adapts after the release;
- (K) a kinematic row sequence that no control sequence produces, or its robot rows replayed as position targets.

**Proposition 1 (C_D admits the static null).** The weld gives zero error on the W warm-up rows. So the static null
p_t = r_0 passes C_D's position rule iff

    (1/(n - W)) sum_{t >= W} |r_0 - p^r_t| < 0.1 / (1 - w),   w = W / n,

when n > W. The rotation rule has the same form with its own threshold: both post-release means must be below
their thresholds, 0.1 / (1 - w) m for position and 0.5 / (1 - w) rad for rotation (0.2 m and 1 rad at w = 0.5). A
valid nonempty prefix that lies entirely in warm-up has zero error but must still reach any required row count; an
empty execution fails. C_D is a function of x only. So on a Z0 clip, any mechanism that produces the same
static-at-start poses on the same rows receives the same passing verdict: a start pedestal with no hand, a frozen
grasp, or (prefix scoring only) a run that stops early. A trajectory that only stays near r_0 is not covered: score
both of its pose errors and apply any required row count. (Take a full execution with zero rotation error
throughout. At w = 0.5 with a post-release reference 0.199 m from r_0, the static null has mean error 0.0995 m and
passes; a pose 0.002 m on the far side of r_0 has 0.1005 m and fails.) An early-stop null is scored on the prefix it
executed: a pass on the full window does not certify every prefix.

**Conditions under which a null can pass a pose-only certificate (P1-P8).** A certificate P(x) admits a null
mechanism M on a clip iff the trajectory that M produces lies inside P's tolerance tube. P2 states that condition
for one mechanism. Each other item is a way in which it comes to hold or goes unseen, and none of them is enough
alone. Scoring (P1, P4-P6), clip and scene (P2, P3), load bounds (P7), executor (P8):
- **P1** P reads only object poses. It cannot tell hand-borne motion from support-borne, or static from controlled.
- **P2** the reference stays within tolerance of what M produces (static: Z0 / MT-null; free fall; sliding).
- **P3** the scene supplies M (a start pedestal; the welded warm-up rows).
- **P4** the average includes rows whose error is zero by construction (warm-up dilution).
- **P5** a run that stops early is scored on its prefix.
- **P6** the contact rule is a geometric distance on some of the moving rows, with no force and no load attribution.
- **P7** nothing bounds contact force or actuator torque.
- **P8** the executor's contacts carry large torsional friction (released model 0.15 m; MuJoCo default 0.005 m).

The project's historical re-execution certificate, which `nullaudit` does not ship and HandReel does not supply,
requires the full trajectory and scores it after warm-up and a grace interval. It checks hand and object tracking
and geometric contact. Its informative endpoint excludes clips on which an object held at its reference start
passes the object thresholds. The score has no force or torque criterion; torsional friction is an executor
setting. So it removes P4, P5 and, on informative clips, the static case of P2, and it does not have P1; P6-P8
remain. Under re-execution the poses come from replay under the chosen simulator and contact model. That does not make the
motion a grasp.

`nullaudit` tests P1-P5 directly: static and simulated nulls, warm-up dilution, prefix scoring. P6 and P7 are
properties of the score, which has no force, load or torque term, and P8 of the executor and the hand model. A null
passes through them only when your certificate and simulator reproduce them. N1 tests
continued command adaptation. A held command can keep a simulated grasp, so an N1 pass shows that the test does not
need the command to adapt after the release; it does not show that no grasp exists.

## Results

The package ships no study records. The HandReel paper reports a separate registered certificate count.

When you report "physics-certified" counts, say that "certified" means tracking re-executed under the executor's
contact and actuator model. It does not mean a force-bounded, physically realisable grasp.

## API

```python
import nullaudit as na

ref = na.reference_on_executor_grid(frames, ref_dt, sim_dt, warmup_rows=W, n_rows=n)  # DO AS I DO's grid
test = na.DoAsIDoObjectTest(ref, required_rows=T)    # or any callable(executed (n, 7)) -> bool / .passed
audit = na.audit_clip(ref, test, clip_id="clip-1",
                      simulator=sim, controls=u, release_row=W,             # optional: the simulated nulls
                      absent_controls=na.move_hand_away(u, column=0, offset=2.0))
audit.z0, audit.passing(), audit.outcomes["hand_absent"].status   # pass | fail | not_run | invalid

report = na.summarize([audit, ...])
print(report.headline())   # each null's rate beside "Z0 base rate k/N", split into Z0 and Z1
report.to_dict()           # every rate carries z0_clips/clips, its Wilson 95% interval and its sentence
```

- **Nulls.**
  - `static_at_start` always runs, because it defines Z0.
  - `time_shifted` (default lag: a quarter of the post-release rows) and `reversed` are sanity nulls. A passing
    delayed or reversed trajectory meets the test's tolerances on this clip. A slow clip can pass both under a
    time-indexed test, and the same clip moving ten times as far can fail both under the same rule.
  - `hand_absent` and `plan_then_freeze` call `simulator(controls, hand_object_contact=...)`, which returns a
    `Rollout(object_traj, hand_object_force, status)`.
- **Invalid rollouts.** A simulated null whose rollout reports `status == "invalid"` is `invalid` before any
  scoring. A run that stops before its release row is also `invalid`: no post-release row ran, so the hand-absent
  force check has nothing to read and no frozen command executed. For plan-then-freeze the prefix verdict is kept
  as `detail["prefix_passed"]`. An early stop at or after the release, such as `"drop"`, is scored as executed.
- **Verdicts.** A wrapped verdict (`.passed` or `["passed"]`) must be a bool; a string or a number raises.
- **N0 verification.** A hand-absent run is verified only if the simulator reports the hand-object force:
  - zero force on every row >= `release_row`: the null is verified and counted;
  - any post-release force: the run is `invalid` and is never counted as a pass;
  - a ledger whose length differs from the trajectory's: `invalid`;
  - a ledger that is not one finite, non-negative real value per row (a negative or non-finite entry on any row, a
    scalar or an array of another shape, or a complex, Boolean or non-numeric array): `invalid`;
  - force not reported: the run is counted but flagged `verified_n0=False`, and the rate's sentence says
    "k of which N0-unverified" (`NullRate.unverified`).
- **Positive control.** `audit.reference_passes` scores the reference itself, and the report warns when it fails.
  If the reference fails, check the row count, pose validity and the evidence the test requires before
  interpreting the nulls. Four valid reference rows compared with themselves correctly fail a five-row requirement.
- **CLI.** `python -m nullaudit clip.npz ... --json report.json` runs the kinematic audit of C_D on stored references.
  Each clip `.npz` holds `reference`, or `frames` + `ref_dt`, `sim_dt`, `warmup_rows`, `n_rows`, plus optional
  `required_rows` and `release_row`.
- **MuJoCo adapter.** `nullaudit.mujoco_adapter.MujocoExecutor(model, q0, hand_geom, ctrl0, warmup_rows, gravity,
  weld_eq_ids, ctrl_steps, terminate_z, absent_mode)`. It runs the registered certifier's executor loop:
  - the warm-up weld is on and gravity is zero during the warm-up;
  - controls are rounded to float32;
  - the run stops on a simulator warning ("invalid") or on an object drop at a block end ("drop");
  - it keeps a per-row hand-object force ledger.
  It has two hand-absent modes:
  - `"pairs"` (the default of the paper's re-execution): it edits the hand-object pairs. **Editing the listed pairs can leave hand-object force** (see the
    tests): move the hand clear or displace its collision geometries, then verify the post-release force ledger;
  - `"displace"`: it moves every hand geom 1 km away.

## What is model-free and what is not

- **Model-free (numpy only):**
  - C_D;
  - DO AS I DO's reference grid;
  - the static, time-shifted and reversed nulls;
  - Z0 and the report;
  - the CLI.
- **Not model-free:**
  - Hand-absent and plan-then-freeze need a simulator plus the clip's scene, start state and stored controls. The
    MuJoCo adapter supplies the loop, not the model. The package ships no scene, hand model or controls. The
    paper's numbers need DO AS I DO's released hand and scene models, which keep their own licence.
  - The hand-absent construction depends on the model. Pair edits can leave hand-object force. Check the post-release
    ledger for residual object-contact force from the monitored geometry.
    Only the force ledger makes the null a verified N0.
  - Only the CPU MuJoCo executor is reproduced (the baseline also runs on Warp). Rollouts are not bit-identical across
    MuJoCo versions.
  - Fidelity to the release's own tracking-error function rests on an earlier, post hoc metric check against it: on 112 stored arm
    runs (two per clip on 56 clips) the largest observed difference in whole-trace tracking error was 1.8e-7.
    NullAudit computes the tracking error without importing DO AS I DO's implementation; it imports NumPy, and the
    adapter imports MuJoCo.

## Tests

`pytest packages/nullaudit/tests`.
- **Known answers on synthetic clips (`tests/_ka.py`)**, all closed-form:
  - the static null gives 0.09 m / 0.36 rad and passes;
  - the same motion with no warm-up fails (P4);
  - the prefix pitfall (P5);
  - strict thresholds, sign invariance, padding, the freeze rows;
  - the report's base rate and its handling of a leaking run;
  - review fixes: an invalid rollout is never scored, a short force ledger is invalid, unverified passes are
    labelled, a zero quaternion fails.
- **Mutant check (`tests/test_mutants.py`).** Fourteen wrong implementations are each killed by at least one known
  answer:
  - the test: `<=` thresholds, warm-up excluded, prefix scored, rotation ignored, zero padding, signed quaternion
    angle, zero quaternion accepted;
  - the nulls: static at the last pose, freeze at u_W, an invalid rollout scored, a short force ledger accepted;
  - the report: an invalid run counted as a pass, a base rate taken over tested clips only, unverified passes pooled
    silently.
- **Parity with our study code on 3 stored development cases** was checked in the development repository against
  the study scorer and stored per-run records; those tests and their inputs are not part of this release.

## Citation

DO AS I DO is the source of the test reproduced here:

```bibtex
@misc{paliwal2026doasido,
  title         = {{Do as {I} Do}: Dexterous Manipulation Data from Everyday Human Videos},
  author        = {Bhawna Paliwal and Haritheja Etukuru and William Liang and Pieter Abbeel and Nur Muhammad Mahi Shafiullah and Jitendra Malik},
  howpublished  = {arXiv preprint arXiv:2606.19333},
  year          = {2026},
  eprint        = {2606.19333},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

## License

Apache License 2.0 (`LICENSE`). Copyright 2026 The HandReel authors.
