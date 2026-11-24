# Auditing a physics certificate (`fastdo.audit`)

Video-to-robot pipelines keep a trajectory when a simulated object follows the reconstruction
closely enough. Such a "physics certificate" is evidence about the hand only if it fails when the
hand does nothing useful. `fastdo.audit` runs three checks on certified runs, guards them with a
positive control, and writes one JSON report. It needs `numpy`, `mujoco` and `nullaudit` (the in-repo package
`packages/nullaudit`) and runs on CPU. `uv sync` installs all three into the main environment, so run it with `uv run`
from the repository root, as below.
When MuJoCo warns, it may write `MUJOCO_LOG.TXT` to the working directory.

## Worked example

```bash
PYTHONPATH=src uv run --frozen python -m fastdo.audit.example audit_example      # writes into ./audit_example
```

This writes a synthetic scene (a cube, a start pedestal and a hand plate on a vertical servo,
`fastdo/audit/toy.py`) and three synthetic run files. Two include a scene for simulated replay; the third has no
scene and still contributes to the static-at-start check. The command audits them under DO AS I DO's certificate and
prints (exit status 0; the last line, `Report: audit_example/report.json`, is left out):

```
Certificate: CD
Z0 base rate: 1/3 clips pass the static-at-start null.
hand_absent: passes on 1/2 tested clips (Z0 1/1, Z1 0/1); ...; 0 invalid, 1 not run
Positive control (with-contact replay): 2/2 pass
Grip census: median ledger normal force / weight, per-run medians 0.952-7.46 (median 4.21) over 2 runs; 1 runs excluded.
  pressed_still   Z0 True  contact-removed pass     grip/weight 7.46
  carried         Z0 False contact-removed fail     grip/weight 0.95
  kinematic_only  Z0 False contact-removed not_run  grip/weight -
```

On `pressed_still` the plate presses a cube that never moves. The certificate passes the
motionless object and passes the run with the hand removed, so on this clip it says nothing about
the hand, although the hand squeezes with 7.5 times the cube's weight. On `carried` the plate receives
a 0.3 m lift command: both nulls fail, and the clip can tell a hand that carries from one that does not.
`kinematic_only` has no scene; it counts in the denominator as "not run". The printed checks therefore use three
denominators: static-at-start 1/3, hand removal 1/2 with one not run, and the positive replay control 2/2. The grip
summary covers two runs.
The example builds its executed rows with the tool's own executor, so its replay anchor is
circular by construction: it shows the mechanics on a synthetic demo and is not evidence that a
replay matches some other pipeline's executor.

## Running it on your runs

```bash
PYTHONPATH=src uv run --frozen python -m fastdo.audit runs/*.npz --certificate CD --removal xml --json audit.json
```

Each run is an `.npz` with the reference object rows on the simulator grid (`reference`, (n, 7)
as x, y, z, qw, qx, qy, qz, warm-up included, at least T rows, unit quaternions to within 1e-3 of norm 1), `warmup_rows` (W)
and `required_rows` (T). With only these the motionless-object null runs, and the CLI then checks these three fields
only: a simulation field in a file that names neither `scene` nor `executor_model` is refused, because nothing
would read it. (A reference given as `frames` with `ref_dt`, `sim_dt` and `n_rows` needs no scene; `sim_dt` is read there. Give `reference` or the frame-grid fields, not both: a file with both is refused.) Each run is named by its file name without the extension, so two files with one name are refused. The simulated checks
need `scene` (an `.xml` or `.mjb` path relative to the `.npz`), `controls` (at least T finite
rows of nu columns), `q0` (nq) and `weld_eq_ids` (equality ids, an empty array if there is no
warm-up weld); `ctrl0` (nu), `gravity` (3), `ctrl_steps` (>= 1) and `terminate_z` default to
zeros, the model's gravity, 1 and no drop check, and every default used is reported. The executor
sets row 0's control before its first step, so a `ctrl0` that differs from it has no effect and the
report says so. With a scene or an executor model named, every simulation field is checked against the model's sizes (0 <= W < T as well). Store the
executed `qpos`, `qvel` and `ctrl` rows (finite, one row per step) and the run's `sim_dt` when you
have them: the positive control compares its replay with `qpos`, `qvel` and `ctrl` row by row and the model's
timestep with `sim_dt`. The comparison covers the rows that the replay and the stored trace both have: stored rows
beyond the supplied controls are not examined, so give controls for every stored row you want checked. Pass all three arrays; the CLI refuses one or two of them. `--removal pairs` needs
`--hand-away COLUMN OFFSET` or an npz `hand_away` (both given, they must agree); either one with
another removal is refused, as is an npz field the tool does not know (exit status 2).
`--allow-nonunit-quaternions` lets a reference with non-zero quaternions off unit norm be audited; the report still
lists it as a problem, and the exit status is 1 unless `--allow-warnings` is given.
`PYTHONPATH=src uv run --frozen python -m fastdo.audit --help` lists the options.

**Simulate the executor's model, not just its XML.** A pipeline's loader often edits the compiled
scene: DO AS I DO's sets the timestep, the integrator and the line-search iterations, and writes the
warm-up weld's pose. Store that model as an `.mjb` and name it in the npz as `executor_model`: the
CLI then simulates it, and `--removal xml` deletes the hand|object pairs from the XML and copies
the executor's options, arrays, statistics and surviving pair rows onto the result
(`compile_without_pairs(..., executor_model=model)` in Python; a layout mismatch raises). Without an
executor model the CLI simulates the XML as stored and relies on the positive control. A model and a passing
certificate alone do not show that the audit reproduced the executor's rows: only the comparison with the stored
`qpos`, `qvel` and `ctrl` does, and the report lists a control without them and without an executor model as
certificate-only.

**Positive control.** When controls and a scene or executor model are given, the run is first replayed once with contact on
the simulated model. The replay must not end with the status `invalid` (a solver warning, a contact or constraint
buffer overflow, a NaN, inf or huge control, a non-finite state, or a |qvel| above 1e3) and must pass
the certificate. A drop below `terminate_z` is a separate status, `drop`: it is not `invalid`, the replay stops
there, and the certificate scores the shortened trajectory; with stored executed rows its qpos, qvel and applied ctrl must each match them to
1e-6 (max |difference|, `--replay-tol`; a non-default tolerance is listed as a problem; a replay
that stops before the executed run counts as an infinite difference); with `sim_dt` the model's
timestep must equal it to 1e-12 s. If the control fails, the contact-removed null is `invalid` and the census is not
taken. Two overrides waive named reasons and the report lists what they waived:
`--allow-uncertified-run` waives only the certificate's rejection of the replay (to apply the nulls
to runs that were not certified), `--allow-replay-mismatch` an invalid replay, a departure from the
executed rows and a timestep mismatch. Only the executed rows or a stored `executor_model` tie the
replay to the run the executor certified: a control with neither rests on the certificate (and on
`sim_dt`, which another model with the same timestep also matches, e.g. one whose servo gains
differ), and the report lists it as a problem ("certificate-only").

**What "the hand" is.** The hand is the geom mask (`hand_geom`, or the regex `--hand-geom`,
default `collision_hand_`, which is reported as a default). The mask decides what the
contact-removed null displaces or unpairs. The N0 force ledger does not trust the mask: it sums
the normal force on the object from the mask plus every other geom on a non-static body
(`nullaudit.mujoco_adapter.contact_ledger_geoms`), so a hand geom left out of the mask makes the
hand-removal null `invalid` instead of a pass when that geom still exerts force on the object after
release (`other_object_force` in the report); one that exerts none does not. The grip census and
`contact_paths` count the same set.

```python
import fastdo.audit as fa

cert = fa.make_certificate("CD", ref, warmup_rows=W, required_rows=T)
run = fa.audit_run(ref, cert, warmup_rows=W, model=model, hand_geom=hand, controls=u, q0=q0,
                   ctrl0=c0, weld_eq_ids=(), gravity=g, ctrl_steps=1, terminate_z=-1e9,
                   executed=fa.Executed(qpos, qvel, ctrl), sim_dt=dt, removal="displace")
run.to_dict()["z0"], run.positive_control.passed, run.clip.outcomes["hand_absent"].status
fa.report_dict([run], tool_version=fa.__version__)        # the pooled JSON report
```

A certificate is any callable that scores executed object rows and returns a bool, an object with
`.passed`, or a mapping with `"passed"`. Presets: `CD`, DO AS I DO's object test (mean position
error < 0.1 m and mean angle < 0.5 rad over the whole executed trace; when `required_rows` is supplied, a run that
executed fewer rows than the clip has also fails), and `C289`, the registered
certificate (window means after a 50-row grace, the object-must-move rule, MT-informative clips
only; T must exceed W + 50). PLAUS and the hand half of C289 are checked only when the caller
supplies `plaus` and `hand_errors`; otherwise the report labels the certificate with the terms it
checked, `C289[obj+M_i+informative]`, a necessary condition only, and marks it incomplete. The
hooks see only the object rows they are called on, so on a null they would score the certified
run's hand; `audit_run` therefore refuses a C289 with hooks, and the nulls are audited with the
object half. On an MT-null clip C289 rejects its own reference by design; the report notes such
clips instead of warning. A JSON file such as `{"preset": "CD", "pos_m": 0.05}` changes
thresholds: it must name its `preset`, and each value must have its field's type (`false`, not
`"false"`). The applied thresholds are written into the report (`inputs.thresholds` and each run's
`settings.thresholds`). `--certificate pkg.module:factory` loads your own.

**Exit status.** 0 when the report lists no problems; 1 when the certificate rejects its own
reference on an informative clip, a run is invalid, a positive control fails or is
certificate-only, a default was used, a non-default `--replay-tol` was given, `ctrl0` has no
effect, the grip census stopped at a non-finite state or sampled no post-release row, or the
object is touched by geoms outside the hand mask in the certified replay (`--allow-warnings`
returns 0 and still prints them); 2 on an input the audit refuses (`fastdo.audit.InputError`: a
missing, non-finite or ill-shaped npz field such as `hand_away` or `sim_dt`, an unreadable run file
or scene, a bad certificate config or regex, a short reference or control, a zero quaternion, or one whose norm differs from 1 by more than
the default 1e-3, which `--allow-nonunit-quaternions` waives for non-zero quaternions); 3 on any other exception, an internal error, printed with its traceback. The JSON
report is strict JSON (NaN is null, an infinite value is "inf" or "-inf"), and it records the
tolerances and constants each run was audited with (`settings`).

## What each check detects, and what it does not certify

| Check | Detects | Does not certify |
|---|---|---|
| Positive control | That the audit simulates what the executor ran: the with-contact replay ends normally, passes the certificate, and matches the stored rows and timestep when given. | Anything about the hand. It only licenses the simulated checks below. Without stored rows or an executor model it can miss a model difference that leaves the certificate's verdict and the timestep unchanged; the report then lists it as certificate-only. |
| Motionless-object null (Z0) | Clips on which the certificate passes an object that never leaves its first pose. On such a clip a pass carries no information about the hand. | That a Z1 clip (the null fails) is hard, or that a pass there needs a grasp. It is kinematic: no simulation is run. |
| Contact-removed null | Runs whose own controls still pass the supplied certificate, in the supplied scene, with the hand-object contact removed: on that open-loop run the pass does not depend on the contact. The run counts only if the ledger shows zero post-release force on the object from the hand mask or any other dynamic geom (N0); otherwise it is `invalid`. | That the object needs no hand outside this replay, that a failing run used the hand well, or that the grasp is physically realisable. Open-loop replay: a re-plan without the hand could do better. |
| Grip census | The median ledger normal force on the object over its weight, after release, from a forward re-solve at the executed states with the warm-up welds off. | The force a task needs, or that a real hand could exert it. It is a property of the scene's contact and actuator model (solref, solimp, condim, servo gains), not of the trajectory alone. |

None of the checks proves a certificate sound. They find cases where it is uninformative. Quote a
null's pass rate beside the two Z0 figures printed with it. They have different denominators: the Z0 base rate counts
every audited clip, and the null's own Z0 / Z1 split counts only the clips on which that null passed or failed. In
the bundled example the base rate is 1/3, hand removal passes on 1/2 tested clips, and 1 of those 2 tested clips is Z0. The static-at-start test
compares each reference pose with the starting pose across the scored rows, so a clip that starts and ends on a
support can still fail it. The grip census starts at release and by default samples every fifth row; a sampled row
without ledger contact contributes zero, and the report takes each run's median normal force over object weight,
then the median across contributing runs. Its JSON field `runs_with_contact` counts runs with a positive median
ledger force, so brief contacts can be missed by the sampling or yield a zero run median. `packages/nullaudit` supplies the kinematic
nulls and the pooled null results; `fastdo.audit` adds the positive-control and grip summaries and assembles the
report.

## Results

The paper's main text reports no result of the three checks on this page. It reports one count from the
certificate check that its authors registered: in a run of 53 fresh Ego4D windows, 2 of the 53 trajectories passed
(Sec. 5.3). The study inputs and records behind that count do not ship with this release.

The tool's own checks ship: closed-form answers on the synthetic scene (`tests/audit/_ka.py`) kill 20 mutants of the
key checks (`tests/audit/test_mutants.py`), and `tests/audit/test_cli_executor_regressions.py`, `test_replay_control_regressions.py`,
`test_input_validation_regressions.py` and `test_unused_input_regressions.py` hold its regression tests.

## Licence

Apache-2.0, as the repository. Runtime dependencies: `numpy` (BSD-3-Clause), `mujoco`
(Apache-2.0) and the in-repo `nullaudit` (Apache-2.0), pinned as a path dependency so that a plain
`pip install .` installs the repository's copy rather than a PyPI package of that name. No GPL code
is used.
