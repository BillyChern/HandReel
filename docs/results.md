# Results in the paper, with their conditions

This page keeps the full statement of each result that the README's table summarises. HandReel replaces only the reconstruction stage of DO AS I DO, the first of its three stages (reconstruction, retargeting to a robot hand, and physics). Every comparison with DO AS I DO below is with that stage, except the full-pipeline sums of Table 5.

Selected results, in the words and with the status that the submitted paper gives each. [`docs/paper_exhibits.md`](paper_exhibits.md)
maps each number to its table, figure or section; some rest on the paper's prose alone.

- **Error on held-out data (object-centre error).** Across 58 analysed held-out HOI4D windows with ground-truth
  object prompts, the paired object-centre error ratio of HandReel to DO AS I DO's reconstruction stage was 0.50 under the registered failure
  rules, in which a failure to deliver counts as an error (Table 1). On DexYCB, HandReel's ratio was 0.85, which is non-inferior, but that verdict is not robust if
  joint failures count against HandReel, and the authors partly used this subject in development (Table 1).
- **Reconstruction time, one window at a time (Table 2, row 1).** On Ego4D, on one H100 on a shared host with
  a 17-core CPU quota, the paper timed both reconstructions one window at a time. DO AS I DO's reconstruction stage took about 30x as long as HandReel: about
  2,000 s against 66 s per window, on 20 windows. The study drew 74 windows; both reconstructions completed 25, and the
  paper analyses 20 of them, without 5 that an audit of the timed processes' CPU affinity and a screening of CPU
  use flagged. It prints the lower of the readings on 25 and 20 windows, by a rule that the authors set after the
  study ended and before they computed estimates. A window counts as completed when DO AS I DO's reconstruction stage finishes
  and HandReel's solve runs, even if HandReel's quality check then rejects it. The bracket that Table 2 prints,
  25 to 37, is an envelope.
- **Reconstruction time, the baseline restarts against HandReel's share of batched runs (Table 2, row 2).** The
  paper timed DO AS I DO's reconstruction stage with a restart for every window: 2,100 s per window. HandReel's side
  is not the time of one window alone. It is a 17 s per-window share: the mean wall time of batched runs, which the
  paper splits among windows in proportion to their stage seconds. Each batched run loaded its models once, and the
  timing includes that load. The ratio is 130x, with a 95% confidence interval of 97 to 170, on the 9 of 19
  attempted windows that both reconstructions completed. The supplement adds that nine is below the registered minimum
  of 10.
- **Reconstruction time, both in batched runs, where the paper models the baseline (Table 2, row 3).** The
  baseline's seconds in this row are a modelled value, about 1,600 s per window: the paper models two instances of
  DO AS I DO's reconstruction stage that run at once, with the 1.2x throughput gain that two instances showed on three window pairs.
  HandReel's side is a 14 s per-window share of its batched runs. The modelled ratio is about 120x, on the 20
  completed windows that the paper analyses; the bracket that Table 2 prints, 87 to 150, is an envelope.
- **One video through the full pipeline (Table 5).** By a sum of separately measured stage times, DO AS I DO's full
  pipeline would take about 3x as long per window with its own reconstruction as with HandReel's: 3,069 s against
  991 s. This comparison times neither pipeline end to end. The reconstruction terms of the sums are the 2,100 s
  and the 17 s per-window share of Table 2, row 2. Both sums share one 894 s physics time, which the authors
  measured after HandReel's reconstruction, at 2 perturbation samples (default 4). With HandReel
  reconstruction, physics is the largest term in the stage sum. The supplement adds that HandReel's sum also holds
  2.63 CPU core-seconds for the certificate check, that neither sum holds a gravity term, and that the 894 s is the
  median elapsed time of one planner arm, measured while two planner arms ran at once on the same GPU, so it is not
  a time for one arm alone on the GPU.

- **Orientation (Sec. 5.2).** Object-centre error ignores orientation. HandReel's rotation error was higher than that of
  DO AS I DO's reconstruction stage on the 13 DexYCB views of objects without symmetry. The supplement gives the evaluated views and the median
  errors.
- **Published methods on HO3D-v3 (Table 3).** Table 3 lists published timings for six methods beside the
  measured Ego4D times. AGILE reports the times for HOLD and MagicHOI. AGILE reports HOLD and MagicHOI accuracy on
  successful sequences (100.0% and 83.3% succeeded, respectively). Each published timing uses its source's own
  input, hardware and timing scope. On HO3D-v3, five of them (HOLD, MagicHOI, AGILE, Open-CHOIR and 4D-HOF) have a
  lower Chamfer distance, in the values that the 4D-HOF paper lists, than the paper measured for HandReel. The
  listed and the measured errors use different inputs and evaluation protocols, so they are not a paired ranking.

Table 2 computes each ratio as the baseline's seconds over HandReel's, from unrounded seconds, so dividing the
printed seconds need not give the printed ratio. For each row of Table 2, the supplement describes the windows
included in the row and explains how each time was obtained (the supplement, Additional Experiments).

Ego4D has no ground-truth object poses, so the Ego4D timing studies measure speed only. HandReel ran the held-out
test from one frozen code tree and both timing studies from a later one, and the authors did not compare the outputs
of the two trees (the supplement, Experiment Details). This release cannot rerun these studies.

