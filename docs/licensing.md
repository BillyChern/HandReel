# Licensing notes

The full licence notes that the README summarises in its "Licence" section (moved here from the README on
2026-11-21). Paths are relative to the repository root. The README picture, `docs/assets/gallery.jpg`, holds dataset
frames under CC BY-NC 4.0, not Apache-2.0; `DATA_NOTICES.md` gives its terms.

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

This page states what the licence texts say as we read them. It is not legal advice.
