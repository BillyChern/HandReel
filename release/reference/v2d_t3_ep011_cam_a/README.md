# Reference output: the example clip, one H100 run

These files are the output of one run of the README's GPU quick-start command on the pinned example clip (episode 11
of the V2D Challenge Track 3 public split, `release/example/example_input.json`). They let you look at a real
reconstruction without a GPU:

```bash
uv run --frozen --extra viser python scripts/view_clip.py release/reference/v2d_t3_ep011_cam_a
```

`expected_projection.png` shows this reconstruction over the clip at decode rows 0, 34 and 68 (README "Expected
output"). `docs/output.md` gives the keys, shapes, frames and units of each file. This is a demonstration on a public clip with
no ground truth, not a calibrated measurement: do not compare it with the paper's numbers (release/QUICKSTART.md,
step 4, "The camera, and what we did about it").

## How it was made

- Source: this release's code, unpacked with `git archive` into an empty directory before the release was packaged.
  The commit that `example_report.json` and the ledger row record (`commit: release archive: ...`) is that
  pre-release source commit, which is not part of the public history. The reference came from that archived source.
  Since then the example checks its input file more strictly (whole-number frames and sizes, a keyframe on the
  decoded grid); the pinned input passes those checks. No run of the current tree is claimed: a run of the same
  command from a clean extract of an earlier release tar reproduced the listed outputs and fields, timings excluded,
  and so did an independent run on envs built from nothing; the comparison and its exclusions are listed below.
- Setup, from that extract: `bash scripts/fetch_baseline.sh` (DO AS I DO at `824591b8`, every submodule verified);
  the main env from `uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser`; CoTracker3 (code at
  the pinned commit `82e02e80`) and DINOv2 (code at `7764ea0f`) fetched by
  `scripts/fetch_weights.py --yes --only cotracker3,dinov2_sam3d` into an empty `TORCH_HOME`; the other weights read
  from the development host's Hugging Face cache and HaWoR checkout, each one matching its sha256 in
  `release/MODELS.md` (`release/env/verify.sh` lists them).
- The `sam3d` env: a fresh copy of the development host's `sam3d` env (the env the locks of `release/env/` were
  exported from; `release/env/verify.sh` passes its imports, it has no flash_attn, and `release/env/check_flash_attn.py` FAILs on it on this H100), with the two `sam3d` commands
  of `release/QUICKSTART.md` step 2 run in it: flash_attn 2.8.3 from the official wheel by its sha256, and the
  editable `sam3d_objects` from this extract's DO AS I DO submodule. `FASTDO_SAM3D_PYTHON` pointed at that copy's
  interpreter, with no extra `PYTHONPATH`. The `hawor` env and MANO were the development host's.
- Command, under `nice` (the ledger row records the priority), on one NVIDIA H100 80 GB HBM3, driver 595.91.07, the
  card otherwise idle (checked with `nvidia-smi` inside the GPU lock), 2026-11-16:

  ```bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
  FASTDO_GPU_MEM_GB=32 uv run --frozen --extra dev --extra frontend \
    python release/example/run_example.py --out outputs/example --card dedicated
  ```

  with `FASTDO_CONDA_BASE`, `FASTDO_SAM3D_PYTHON` and `FASTDO_HAWOR_ROOT` set and `FASTDO_COTRACKER_HUB_REF` unset
  (the report's `pins` record the pinned commit). The report's `host.gpus` names the one card the run used.
- The clip was downloaded by the script from the pinned dataset revision; its sha256 matched the pin
  (`508a4fc94747741f35fd12780e6d5e6e7f78a233e2bf379202515cb9bff32585`).
- Every stage finished (`decode`, `mask`, `shape`, `intrinsics`, `hands`, `geometry`, `tracks`, `solve`): 60.3 s
  summed over the stages, one cold window with model loading included (decode 4.2, mask 5.2, shape 33.9,
  intrinsics 4.1, hands 1.4, geometry 7.1, tracks 1.3, solve 3.0). This example ran without qc (no
  `--qc-rule` was given), so its `delivered` field is `null` and it produced no `clip_record.npz`. The held-out
  package documents the constants of a rule fitted on DexYCB development clips (`reproduce/heldout/README.md`); that
  rule is not validated on this clip. SAM 2 logged that its `_C` extension is missing and
  skipped its optional mask post-processing, as in every run of this release (`release/QUICKSTART.md` step 4).
- `profile.json`: `n_frames` 137 is the window's source frames (20 fps); `fps` 10.0 is the decode rate at stride 2
  (69 rows). Each stage's `peak_gpu_mem_gb` is the larger of the engine process's own peak (decimal GB) and a peak
  that the stage's worker reports. Only the hands worker reports one, in GiB (4.72 here), so stages whose model
  runs in another worker read 0.0 and the field is not a whole-card peak (`docs/output.md`).
- We checked the result by eye only: the object mesh projected with `intrinsics.npz` and `solve.npz` lies on the beige
  cup, and the hand vertices and joints on the hands, at rows 0, 34 and 68. Nothing else was checked.
  `expected_projection.png` is that check as a picture (object mesh and hand joints), drawn from the files of this
  directory and the clip by `uv run --frozen python scripts/render_reference_png.py
  release/reference/v2d_t3_ep011_cam_a --video <the clip> --out expected_projection.png`.

**The compared reconstruction outputs matched on this clip.** The files and fields listed here, as this run wrote them and before packaging, match the previous
reference run (2026-11-15, an earlier release candidate, the host's `sam3d` interpreter with a flash_attn on its
`PYTHONPATH`, CoTracker3's cached `main` code, eight cards listed): `solve.npz` (every array except its `seconds`),
`intrinsics.npz` (including `focal_scale` 1.10404), `hands.npz` (all arrays; the full file's sha256 is the same,
below; the comparison was made on the run's file, before any array was removed), `mesh.obj`, `source_domain.json` and `ATTRIBUTION.txt`, and `shape.json` and `solve.json` apart from their
timings. A third run, from an earlier commit, gave the same. Only timings, run IDs, the commit, the
listed cards and the CoTracker3 ref differ in `profile.json`, `example_report.json` and the ledger row.

## What is here, and what was left out

| file | as written by the run? |
|---|---|
| `solve.npz`, `intrinsics.npz`, `mesh.obj`, `shape.json`, `solve.json`, `source_domain.json` | byte for byte (`engine/v2d_t3_ep011_cam_a/`) |
| `profile.json` | every field except `clips_per_gpu_hour` (3600 divided by the timed stage sum of this one cold window, which leaves out the runner's overhead and the untimed prompt: not a throughput or GPU-hour figure, so it is left out) |
| `example_report.json`, `timing_ledger.jsonl`, `ATTRIBUTION.txt` | byte for byte (`--out`) |
| `hands.npz` | the frame indices, the two intrinsics matrices and, per side, the joints and the two flag arrays; the MANO mesh arrays (`left_vertices`, `right_vertices`, `left_faces`, `right_faces`) and the MANO parameter arrays (`{side}_params`, `{side}_betas`, `{side}_trans`) are left out |
| `expected_projection.png` | not written by the run: drawn afterwards by `scripts/render_reference_png.py` (above) |

The ledger row's `"execution": "pipelined"` names the engine path the example uses (the batched runner; QUICKSTART
step 4), not pipelining across windows: this run had one cold window, nothing ran beside it, and its `seconds` is
that window's stage sum.

`hands.npz` ships without the MANO mesh: the face arrays are the MANO mesh topology (plus HaWoR's wrist faces), and
the vertex arrays are the posed MANO mesh, and MANO's licence forbids redistributing the model. It also ships without
the predicted MANO pose, shape and translation: neither the hand model's terms nor the hand estimator's licence says
whether predicted parameters may be passed on, so this copy leaves them out. It keeps the 21 joints per hand and the
validity flags. `scripts/view_clip.py` therefore draws the hands of this
directory as their 21 joints; on your own runs, which have the vertices and faces, it draws meshes. The run's files
are kept by the authors. The table gives the sha256 of each run file, before any array was removed, beside the
sha256 of the file this directory ships:

| file | sha256 of the run's file (not shipped) | sha256 of the shipped file |
|---|---|---|
| `hands.npz` | `89ebc2856f3224588426e7be8f72690ff8c7b156fbfdb1ecfcd2d66cdbefaa84` | `86aa9b6769da8cfe860268d7e5beb4c2119debefb9170865507b1e20404f05e1` |
| `profile.json` | `84289ae6b878119961b86ae9ad812cb3413598707ec444fcbb7af12f4d6da06d` | `b71fb7c2d4685331a5f7b0090a05df1ed301963289e8f9c546feb389917aa847` |

`ATTRIBUTION.txt` is the run's file, unchanged. It names the hand meshes among the run's outputs; this directory
ships the hand joints and MANO parameters, not the meshes.

Not shipped: `decode.npz` (638 MB of decoded frames), `geometry.npz` (83 MB), `mask.npz` (213 MB), `mask_prompt.npz`,
`tracks.npz`, `sdf.npz`, `samples.npz`, `hands_hawor.npz` and `source_profile.json`; and the clip itself. For the
source frames in the viewer, download the clip (CC BY 4.0) and pass it with `--video`:

```bash
uv run --frozen hf download nvidia/video_to_data_challenge \
  track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4 \
  --repo-type dataset --revision 5f68335f3acc802033d1e80728c1633197521de8 --local-dir data/example
uv run --frozen --extra viser python scripts/view_clip.py release/reference/v2d_t3_ep011_cam_a \
  --video data/example/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4
```

## Licence

The input clip is from the Video to Data (V2D) Challenge Dataset, Track 3 public split, by NVIDIA, licensed CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>), <https://huggingface.co/datasets/nvidia/video_to_data_challenge>.
These files are derived from it (reconstructed by this pipeline); `ATTRIBUTION.txt` is the credit the run wrote, and
`expected_projection.png` shows three of its frames, credited in the image. `ATTRIBUTION.txt` is kept byte for byte, so
it still lists "the hand meshes" among the derived files: this directory does not ship them (the MANO mesh arrays were
removed, above), and the hand arrays it does ship are derived from the clip in the same way. The files are not covered by the
Apache-2.0 licence of the code. They are also outputs of third-party models whose licences bind them: several are
non-commercial, HaWoR's is NoDerivatives, MANO's forbids redistribution; the hand arrays follow HaWoR's and MANO's
terms. Use them for non-commercial research only, and read README "Licence", `release/MODELS.md` and
`THIRD_PARTY_LICENSES.md` ("Derived data in this repository") before you share them.
