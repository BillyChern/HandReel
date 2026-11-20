<div align='center'>
<h1>HandReel: Fast Hand-Object Reconstruction from Video without Per-Frame Pose Sampling</h1>

[Shi Chen](https://shichen.world/)<sup>1,2</sup>, [Jeffrey Ichnowski](https://ichnow.ski/)<sup>1</sup>

<sup>1</sup>[Robotics Institute, Carnegie Mellon University](https://www.ri.cmu.edu/) &nbsp;&nbsp; <sup>2</sup>[Intuition](https://intuition.dev/)

<img src='https://img.shields.io/badge/arXiv-coming_soon-b31b1b' alt='arXiv: coming soon'>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache_2.0-blue' alt='License: Apache 2.0'></a>

</div>

![Six video frames, each with HandReel's reconstructed object and the estimated hand outlined in purple: a knife, a sugar box, a cracker box, a potted meat can, a tomato soup can and a pitcher](docs/assets/gallery.jpg)

<sub>HandReel takes a video of a hand moving an object and returns the object's 3D mesh and its pose in each processed frame, together with the hand. Purple outlines: the reconstructed object and the estimated hand, drawn over one frame of each of six clips. Frames from HOI4D and DexYCB (CC BY-NC 4.0); see [`DATA_NOTICES.md`](DATA_NOTICES.md).</sub>

## Citation

`CITATION.cff` and the entry below cite the paper, "HandReel: Fast Hand-Object Reconstruction from Video without
Per-Frame Pose Sampling". Please also cite DO AS I DO (arXiv 2606.19333) and the models
you use (SAM 2, SAM 3D, MoGe-2, CoTracker3, HaWoR, WiLoR, MANO, Hands23).

```bibtex
@misc{chen2026handreel,
  title  = {{HandReel}: Fast Hand-Object Reconstruction from Video without Per-Frame Pose Sampling},
  author = {Chen, Shi and Ichnowski, Jeffrey},
  year   = {2026}
}
```

## Overview

HandReel takes a short video of a hand using an object. It returns the object's 3D shape, its position and rotation
over time and its real-world size, with the motion of the right hand in the frames where the hand estimator finds
it. The picture at the top shows six such results, each drawn over one frame of its clip.

HandReel (Python package `fastdo`) is a staged engine (decode, mask, shape, intrinsics, hands, geometry, tracks, solve,
then the optional qc and retarget stages), configured with Hydra (`run/conf/config.yaml`). The mask, shape, intrinsics,
hands, geometry and tracks stages use pinned pretrained models (SAM 2.1, SAM 3D Objects, MoGe-2, HaWoR, CoTracker3).
Each stage writes its artifacts per window. The solve stage fits the object's poses and scale over the window. [`release/MODELS.md`](release/MODELS.md) lists the external
files for the released reconstruction configurations, their sources and licensing information, and the SHA-256 hashes
recorded on the authors' system. The release also contains research code for retargeting to a robot hand
and for physics-based tracking; these do not yet run from this release (see ["Known gaps"](#known-gaps)).

## Quick start

The fastest check needs no GPU and no model weights. A full run needs the installation and the weights below.

### Try it on CPU, no weights

The shipped reference reconstruction of the clip above opens in your browser, in 3D over the video's frames. It needs
no GPU, model weights or MANO. Unpack the release archive and work from its `handreel/` directory:

```bash
tar -xzf handreel-release-*.tar.gz && cd handreel
uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser
uv run --frozen hf download nvidia/video_to_data_challenge \
  track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4 \
  --repo-type dataset --revision 5f68335f3acc802033d1e80728c1633197521de8 --local-dir data/example
uv run --frozen --extra viser python scripts/view_clip.py release/reference/v2d_t3_ep011_cam_a \
  --video data/example/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4
```

Open the printed URL (<http://127.0.0.1:8080>; pass `--port N` if 8080 is taken); Ctrl-C stops it. The clip is CC BY 4.0 (V2D Challenge, NVIDIA).
If you supply a local copy of the clip, compare its SHA-256 with the one in `release/example/example_input.json`.
The viewer pairs frames by index and accepts any readable video that contains those indices.

**Viewer time is not reconstruction time.** The viewer draws a reconstruction that already exists, so the time it
takes to load and draw is viewer time. The GPU example of "Quick start" logs its reconstruction seconds apart from
any viewer time.

### Installation

Requirements: Linux x86_64, Python 3.12, [uv](https://docs.astral.sh/uv/), conda, `git`, and `ffmpeg` with `ffprobe`
on `PATH`. The model stages need an NVIDIA GPU with at least 24 GB of memory; the code was developed on one H100 80 GB
with CUDA 12.8 wheels, and only that build was tested. A cold build needs about 60 GB of disk before the weights.

```bash
cd handreel                              # the unpacked release archive
bash scripts/fetch_baseline.sh           # DO AS I DO at the pinned commit, into third_party/
bash scripts/setup_all.sh --dry-run      # prints every step of release/QUICKSTART.md step 2
bash scripts/setup_all.sh                # builds them: the uv env, the sam3d and hawor conda envs, HaWoR
```

`setup_all.sh` needs conda on `PATH`, or `FASTDO_CONDA_BASE` set to your conda or miniforge root. It clones HaWoR to
`../hawor`, next to this checkout.

The GPU example (`release/example/run_example.py`) needs exactly these four variables:

- `FASTDO_GPU_MEM_GB=32`;
- `FASTDO_CONDA_BASE`;
- `FASTDO_SAM3D_PYTHON`;
- `FASTDO_HAWOR_ROOT`.

[`docs/installation.md`](docs/installation.md) has the rest: a CPU-only install, building for a GPU other than
an H100, the environments and why there are several, every host path and environment variable, the flash_attn
requirement on A100, H100 and H200 cards, and the notes on building by hand. `release/QUICKSTART.md` walks through every step and says which commands were run for this release.

### Weights

`scripts/fetch_weights.py` pins every model to a revision and verifies each published sha256; it never writes under
`third_party/`.

```bash
uv run --frozen python scripts/fetch_weights.py            # the table: source, revision, licence, sha256
uv run --frozen hf auth login                              # after accepting SAM 3D's terms on Hugging Face
uv run --frozen python scripts/fetch_weights.py --yes \
  --only sam2,moge2,sam3d,moge1_sam3d,dinov2_sam3d,cotracker3,hawor,wilor_detector --hawor-root ../hawor
```

`../hawor` is the HaWoR checkout that `scripts/setup_all.sh` cloned ("Installation"); pass yours if it is elsewhere.
`--yes` means you accept the licence of each model you fetch; several are non-commercial (see "Licence"). SAM 3D
Objects is gated on Hugging Face.

**MANO** is never downloaded or shipped. Register at <https://mano.is.tue.mpg.de>, accept the licence
(<https://mano.is.tue.mpg.de/license.html>; non-commercial, no redistribution), put `MANO_RIGHT.pkl` and
`MANO_LEFT.pkl` in `FASTDO_MANO_DIR` and `FASTDO_MANO_LEFT_DIR`, and copy them into the HaWoR checkout
(`release/QUICKSTART.md` step 3).

[`docs/weights.md`](docs/weights.md) lists every model with its licence, whether commercial use is allowed and how
to get it. `release/MODELS.md` adds the pinned revision and sha256 of every file, and `THIRD_PARTY_LICENSES.md` the
licence texts.

### Run the example on a GPU

On CPU, with only the main environment, these run without weights, data or a GPU:

```bash
uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser
export CUDA_VISIBLE_DEVICES=
uv run --frozen pytest -q -m "not slow and not timing" \
  tests/release tests/test_package.py tests/test_hostpaths.py tests/run/test_build_hoi4d_hands.py tests/reproduce
uv run --frozen python run/pipeline/bench_throughput.py --cfg job > /dev/null
uv run --frozen python release/example/run_example.py --stub --video README.md --out outputs/example_stub
PYTHONPATH=src uv run --frozen python -m fastdo.audit.example outputs/audit_example
```

In order: a CPU subset of the test suite; Hydra config composition for the benchmark entry point without running it; the
example runner with every stage replaced by a placeholder (argument parsing, config loading, orchestration and output
schema only; in stub mode `--video` is never decoded, so any existing file works as a placeholder, here
`README.md`); and the physics-certificate audit (`docs/audit.md`) on a synthetic scene. The stub's stage artifacts are
empty placeholder files (0 bytes) and its seconds are not timings; the run also leaves an empty `engine/.gpu.lock`
(`docs/output.md`). To look at a real reconstruction on CPU, open the
reference output of "Interactive viewer". The whole CPU suite is `uv run --frozen pytest -q -m "not slow and not timing"`
(about 30 min on a 4-thread cap; measured on the development host); GPU tests are faked or skip, as do tests whose
dataset, conda env or baseline checkout is absent.

On a GPU, with the environments, weights and MANO in place and the four variables the GPU example needs ("Installation") set:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0   # the one card to use, numbered as /proc lists them
FASTDO_GPU_MEM_GB=32 uv run --frozen --extra dev --extra frontend \
  python release/example/run_example.py --out outputs/example --card dedicated
```

It replaces `outputs/example/engine/<clip>/` and the report on every run, so use a fresh `--out` per run you want to
keep. The two exports make the report name the card the run used (`release/QUICKSTART.md` step 4). Before any GPU
work the script checks that CoTracker3, the SAM 3D Objects files and MANO are in place and names whatever is
missing. At the end it logs the reconstruction time on your machine (the stage sum of this one cold window); viewer
time is not part of it, and no ratio to another method is computed. That sum covers the eight stages only, model
loading included. The SAM 2.1 box-to-mask prompt before the stages, the clip download, interpreter and uv start-up
and report writing fall outside it (about 11-12 s in one independent run). That boundary differs from every row of
the paper's Table 2 ([`docs/paper_exhibits.md`](docs/paper_exhibits.md), "Timing boundaries"). The shipped reference run has a stage sum of
60.3 s, and the Quickstart's "Run-to-run variation" paragraph gives the ranges that the authors report over six cold
runs.
It downloads one CC BY 4.0 clip (V2D Challenge Track 3 public split, at a pinned revision) and reconstructs it;
`release/QUICKSTART.md` step 4 describes the clip, the camera handling and the outputs. One run of this command on
an H100 produced the reference output in `release/reference/v2d_t3_ep011_cam_a/` (its README says how, and what was
left out).

### Expected output

On the example clip a successful run puts the object mesh on the beige cup and the hand points on the hands. The
picture at the top of this page is the shipped reference output at decode rows 0, 34 and 68, drawn over the clip's
frames (CC BY 4.0, V2D Challenge Track 3, NVIDIA) by `scripts/render_reference_png.py`. The PNG projects the solve poses. The interactive viewer uses contact-refined poses when available. To draw the same picture for your own run, after the clip has been downloaded:

```bash
uv run --frozen python scripts/render_reference_png.py outputs/example/engine/v2d_t3_ep011_cam_a --out my_run.png
```

It reads the frames from `decode.npz` (pass `--video` for a directory without it). Compare by eye: the overlay
should sit on the same pixels. It is not a numeric tolerance.

## Detailed usage

### Run on your own video

Write an input description for your clip, starting from `release/example/spec_template.json`: `clip_id`, `domain`
(`egocentric` or `third_person`), the video's `fps`, `n_frames`, `width` and `height` (the template's `_note` gives the
`ffprobe` command), the `window` in source frames (about 10 s at most), a `keyframe` inside it and a `box_xyxy` around
the object on that keyframe, in source pixels. The keyframe must be a frame the run decodes (every third frame from
the window start at 30 fps); otherwise the example stops before any model loads and names the nearest decoded frames. Then, on a GPU with the same setup as the example:

```bash
cp release/example/spec_template.json my_clip.json                   # then edit it for your video
FASTDO_GPU_MEM_GB=32 uv run --frozen --extra dev --extra frontend \
  python release/example/run_example.py --spec my_clip.json --video my_clip.mp4 --out outputs/my_clip --card dedicated
uv run --frozen --extra viser python scripts/view_clip.py outputs/my_clip/engine/my_clip
```

The last path is `<--out>/engine/<clip_id>/`. With `--spec`, nothing is downloaded and no sha256 is checked; the
script checks the spec's width and height against the file and exits with a one-line message on any input error.
`ATTRIBUTION.txt` then names no dataset: your video's licence and attribution are yours to state. The camera is
estimated per clip by MoGe-2, as for the example. Add `--stub` to check a spec on CPU without a model: it checks the
spec itself (its fields, window and keyframe, and the box inside the frame) but not its width and height against the
file, which only a GPU run probes.

### Input

- One or more windows of monocular RGB video (an mp4 file and a frame range), decoded at the working rate.
- A box on the manipulated object at one keyframe. For internet video, the opt-in prompt stage (Hands23 detector)
  proposes the keyframe and its box instead.
- Model weights (fetched by `scripts/fetch_weights.py`) and your own MANO files (never fetched or shipped).

`release/example/example_input.json` is a complete input description for one public clip. The clip is downloaded at
a pinned revision on first use; it is not shipped.

### Output

Per window, the run writes a directory `<bench.artifact_root>/<clip_id>/` (default `outputs/engine/<clip_id>/`) with
one artifact per stage (`decode.npz` to `solve.npz`, plus `mesh.obj`) and `profile.json`. The object's pose per frame
and its metric scale are in `solve.npz`; the hands are in `hands.npz`, in the camera of `intrinsics.npz`. The shipped
example stops at `solve.npz` (see ["Known gaps"](#known-gaps)). [`docs/output.md`](docs/output.md) gives the stage layout and the fields
and coordinate conventions of the solve and hand outputs.

## Interactive viewer

`scripts/view_clip.py` serves one clip directory in the browser (Viser, on 127.0.0.1): the object mesh and the hands
in 3D over the source frame of the same frame id, with play/pause, a frame slider and orbit (drag; scroll to zoom).
It needs the `viser` extra (in the CPU `uv sync` above) and no GPU. The shipped reference output, on CPU:

```bash
uv run --frozen --extra viser python scripts/view_clip.py release/reference/v2d_t3_ep011_cam_a
```

Open the printed URL (default <http://127.0.0.1:8080>; `--port N` changes it); Ctrl-C stops the server. To show the source frames in the
viewer, supply the matching clip with `--video` (download command in
`release/reference/v2d_t3_ep011_cam_a/README.md`). The PNG preview includes three source stills. Its hands are drawn as their 21 joints (right green, left
orange, as in the picture above), because the MANO mesh arrays (vertices and faces) are not shipped. Your own GPU run has `decode.npz` and the full hand arrays, so the viewer
shows its frames and hand meshes. The frame slider counts decode rows from 0, as the rest of this page does:

```bash
uv run --frozen --extra viser python scripts/view_clip.py outputs/example/engine/v2d_t3_ep011_cam_a
```

## Results in the paper

| Measured in the paper | DO AS I DO | HandReel | Ratio |
|---|---|---|---|
| Reconstruction time per window, both one window at a time (Ego4D, 20 windows) | about 2,000 s | 66 s | about 30x |
| Reconstruction time per window, baseline restarting for every window against HandReel's batched runs (Ego4D, 9 windows) | 2,100 s | 17 s per-window share | 130x |
| Object-centre error, HandReel over DO AS I DO, held-out HOI4D (58 windows) | | | 0.50 |
| Object-centre error, HandReel over DO AS I DO, held-out DexYCB | | | 0.85 |

Ego4D has no ground-truth object poses, so the two timing rows measure speed only. The DexYCB error ratio is
less certain than the HOI4D one (see the linked page). HandReel is not the most
accurate method in the paper's comparisons: on HO3D-v3, five published methods (HOLD, MagicHOI, AGILE, Open-CHOIR
and 4D-HOF) list a lower Chamfer distance than the paper measured for HandReel, and on the DexYCB views of objects
without symmetry HandReel had the higher median rotation error.

[`docs/results.md`](docs/results.md) states each result in full, with its conditions, and
[`docs/paper_exhibits.md`](docs/paper_exhibits.md) maps each number to its table, figure or section. This release
cannot rerun these studies.

## Evaluation

Most of the paper's records are not in this release, and no script here reruns a paper experiment. Two checks run
from this release:

- `python3 reproduce/regenerate_tables.py --out <new or empty dir>` checks all 14 aggregate-only records in
  `reproduce/records/`, which are proposed for release and await the authors' decision. It uses ten of them to
  write four tables of the paper source's table generator and writes a fifth from settings in the generator. It
  compares the five outputs with that generator's output at a pinned paper-source revision, ignoring comment lines. None of
  the five is a table of the submitted main paper, whose Tables 1 to 5 the paper source writes by hand.
- With per-unit errors that you regenerate after acquiring HOI4D and DexYCB,
  `reproduce/e1_verdict_from_scores.py` recomputes the error ratios, intervals and median errors of Table 1
  (`reproduce/heldout/README.md`).

[`reproduce/README.md`](reproduce/README.md) says what each check covers, and [`docs/paper_exhibits.md`](docs/paper_exhibits.md) maps every exhibit of the submitted
main paper to what this release holds toward it.

The benchmark entry point runs the reconstruction on DO AS I DO's whisking demo clip (from the baseline submodule). As
configured, it reruns 3 identical windows of that one clip:

```bash
FASTDO_GPU_MEM_GB=32 uv run --frozen --extra dev --extra frontend \
  python run/pipeline/bench_throughput.py paths.root=$PWD \
  'bench.stages=[decode,gate,mask,shape,intrinsics,hands,geometry,tracks,solve]'
```

The demo supplies the baseline clip, its SAM 3D object mesh and the pre-optimization keyframe layout. Before timing,
the runner projects the mesh to make one silhouette prompt for all three copies.

It logs a clips-per-GPU-hour figure. The logged rate divides the attempted copies by the summed stage hours, CPU stages
included; a copy that stops early contributes only its partial time. That figure is smoke throughput on duplicated windows, not an evaluation: the
three windows are the same clip, nothing is scored, and it is not comparable with the paper's reconstruction times. On this
clip the depth-anchor gate flags all three solves (the log reports FLAGGED; in one independent run it moved 31 of 46
frames, by up to 0.51 m), so do not read the benchmark's output as a reconstruction of good quality either.

The `qc` stage needs a fitted acceptance rule (`run/pipeline/calibrate_acceptance.py`, then
`qc_stage.rule_path=<rule>`); retarget is opt-in and runs only on qc-accepted windows (`retarget` in `bench.stages`,
`retarget_stage.enabled=true`). Physics tracking runs in DO AS I DO's `retargeting` env.

**Datasets.** None are redistributed. DexYCB (CC BY-NC 4.0), HOI4D, HO3D-v3, OakInk2, Ego4D and 100DOH each have their
own terms; obtain them from their owners. `run/data/` holds manifests and two downloaders (`download_dexycb.py`;
`download_100doh.py`, which fetches YouTube clips: you are responsible for complying with YouTube's Terms of Service
and the videos' copyright).

## Repository layout

```
run/conf/          Hydra configs
run/pipeline/      entry scripts (benchmark, data builders, calibration, evaluation)
run/paper/         e1_table.py: the paper's held-out-test macros from the verdict record
run/study/         e1_stats.py: the held-out-test estimator (no other study script is in this release)
run/robot/         Unitree G1 package export
run/setup/         baseline environment setup
run/data/          dataset manifests and downloaders
src/fastdo/        the package
packages/          alongray and nullaudit, standalone packages with their own LICENSE and NOTICE (uv sync: nullaudit only)
release/           pinned worker-environment locks, the weights manifest, a one-clip example and its reference output
reproduce/         the table regeneration check and its records; the held-out test's protocol, unit IDs and verdict check
scripts/           setup_all.sh, fetch_baseline.sh, fetch_weights.py, view_clip.py, render_reference_png.py, dependency_licences.py,
                   check_paper_quotes.py (cited paper numbers and wording patterns against the main PDF; not a claim check)
docs/              the audit tool, the timing protocol, the paper exhibits and the record names
licenses/          licence texts of the third-party code adapted here
third_party/       DO AS I DO official code as an unmodified git submodule (empty in the release tar)
tests/             pytest suite
```

`VERSION` reads `$Format:%H %cs$` in a git clone. Only `git archive` exports, such as the release tar, fill it in with
the commit and its date (`.gitattributes`: `export-subst`); in a clone, `git rev-parse HEAD` gives the commit.

## Acknowledgements

The reconstruction builds on DO AS I DO, SAM 2, SAM 3D Objects, MoGe, HaWoR, WiLoR, CoTracker3, MANO and Hands23, and
the physics code on MuJoCo and MuJoCo-Warp; `THIRD_PARTY_LICENSES.md` credits each with its licence.

## Known gaps

- **The gated SAM 3D download is untested.** The reference output (`release/reference/v2d_t3_ep011_cam_a/`) was made
  on the authors' machine. A second run, on environments that `scripts/setup_all.sh` built from nothing, reproduced
  its stage artifacts (timing fields aside), but that run copied the SAM 3D weights and MANO files from an existing
  machine.
- **The quality check and the robot-hand retargeting do not run from this release.** No acceptance rule for new
  video ships, and there is no path from a reconstruction to a physics run.
  `reproduce/heldout/qc_rule_constants.json` records the rule that the held-out test used. The authors fitted it on
  71 DexYCB development clips, where its five-fold cross-validated area under the curve was 0.4904 (0.5 is chance;
  the supplement, Limitations), so do not use it on new video.
- **This code is newer than the code the paper ran.** It is a later commit than the tree frozen for the held-out
  test and the tree timed in the Ego4D timing studies, and this release shows no comparison of its outputs with
  either.
- **No script here reruns a paper experiment.** [`docs/paper_exhibits.md`](docs/paper_exhibits.md) says what this
  release holds toward each table and figure, and [`reproduce/README.md`](reproduce/README.md) lists what is missing.
- **No container image.**

## Licence

Our code is Apache-2.0 (`LICENSE`, `NOTICE`). `packages/alongray` and `packages/nullaudit` carry their own `LICENSE` and
`NOTICE`. **The derived data files are not Apache-2.0**: `reproduce/heldout/` (unit IDs derived from HOI4D and DexYCB, CC
BY-NC 4.0: non-commercial use only; no per-unit error against their ground truth ships) and the reference output in `release/reference/` (derived from a CC BY 4.0 clip; its
hand arrays are HaWoR outputs on the MANO model and follow HaWoR's CC BY-NC-ND 4.0 and MANO's licence). We do not ship
MANO-derived meshes: the reference's hand vertices and faces are left out. `THIRD_PARTY_LICENSES.md`, "Derived data in
this repository", lists each file and its terms. The repository includes code adapted from DO AS I DO (MIT), MoGe (MIT) and SAM 3D Objects (SAM License: two
short expressions, and an adaptation of its mesh postprocess in `src/fastdo/shape/fast_postprocess.py`); each such file
says so in its header, and `THIRD_PARTY_LICENSES.md` lists them with the licence texts in `licenses/`. **We ship no
weights, MANO files or datasets.** `THIRD_PARTY_LICENSES.md` lists every model, runtime, robot asset and Python
dependency with its version and licence. Read it before use:

- **Non-commercial components in the default pipeline:** HaWoR (CC BY-NC-ND 4.0), the WiLoR detector (NC; NC-ND on the
  stricter reading), smplx (MPI non-commercial), MANO (non-commercial, no redistribution, no commercial training),
  CoTracker3 (CC BY-NC 4.0), the Hands23 weights (no licence text, NC training data), and in the shape stage
  nvdiffrast 0.4.0 (NVIDIA Source Code License, non-commercial) and kaolin 0.18.0 (its `non_commercial` package). As
  shipped, the pipeline and the data it produces are for non-commercial research only: the outputs are bound by these
  model licences too.
- **Copyleft packages imported at run time:** these files are Apache-2.0, apart from the SAM-adapted parts of
  `src/fastdo/shape/fast_postprocess.py`, which stay under the SAM License (THIRD_PARTY_LICENSES.md). They import,
  at run time, ultralytics
  (AGPL-3.0) in the HaWoR process, and pymeshfix (GPL-3.0), igraph (GPL-2.0-or-later) and plyfile (GPL-3.0-or-later)
  in the SAM 3D worker process. None of them is included or modified here. If you distribute a combined work, their
  copyleft terms apply to it. libigl is MPL-2.0: its sections 3.1 and 3.2 set the terms for distributing libigl's own files, in source or
  executable form and modified or not; that copyleft is per file and does not reach this project's files.
- **The DINOv2 code SAM 3D loads** (torch.hub, commit `7764ea0f`): the backbone code and weights are Apache-2.0; at
  that commit `hubconf.py` also imports the Cell-DINO modules, whose files state CC BY 4.0 and CC BY-NC. SAM 3D calls
  only `dinov2_vitl14_reg` and loads no Cell-DINO weights.
- **Robot assets:** the Sharpa Wave hand assets inside DO AS I DO carry no licence file; this repository ships none of
  them, and its outputs hold joint trajectories, not the meshes or XML.
- **Non-commercial baselines and ablations** (evaluation only; not in the default pipeline, never shipped):
  FoundationPose (NVIDIA Source Code License, non-commercial) and Open-CHOIR's assets.
- The example clip is from the V2D Challenge Track 3 public split (NVIDIA, CC BY 4.0); `release/example/run_example.py`
  writes the attribution its outputs need (`ATTRIBUTION.txt`).
- The conda locks of `release/env/` fetch from `repo.anaconda.com`, so Anaconda's Terms of Service apply to those
  packages.

This section states what the licence texts say as we read them. It is not legal advice.
