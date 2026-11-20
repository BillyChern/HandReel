# Third-party licences

This repository's own code is Apache-2.0 (see `LICENSE` and `NOTICE`). It includes code adapted from three
third-party projects, listed below with their licences (full texts in `licenses/`). It includes no third-party model
weights, MANO model files or datasets.

**Not included, and never redistributed by this repository:** model weights (SAM 2.1, SAM 3D Objects, MoGe-2,
CoTracker3, HaWoR, the WiLoR detector, Hands23), the MANO model files, smplx, and every dataset. Obtain each from its
source under its licence.

This is a listing of the licences as we read them, not legal advice.

## Code adapted from third-party projects (included)

| Source | Licence | Files (each carries an in-file notice naming the source) |
|---|---|---|
| DO AS I DO, `malik-group/do-as-i-do` @ `824591b`, Copyright (c) 2026 malik-group | MIT (`licenses/DO-AS-I-DO-MIT.txt`) | `src/fastdo/physics/clip.py`, `executor.py`, `sync_free.py`, `c2_rollout.py`, `union_slot.py`, `continuous.py`, `d1.py`, `rank_estimator.py`, `perturb.py`; `src/fastdo/policy/warmup.py`; `src/fastdo/retarget/isolation.py`; `src/fastdo/geometry/moge_pointmaps.py`; `run/pipeline/_baseline_physics_driver.py`; parity references in `tests/retarget/test_wrist_target.py` (`process_dataset.py`) and `tests/policy/test_warmup.py` (`warmup_analytical_init`) |
| MoGe, `microsoft/MoGe` @ `b942f00b` (`moge/model/v2.py`, `MoGeModel.infer`; `moge/utils/geometry_torch.py`, `recover_focal_shift`), Copyright (c) Microsoft Corporation | MIT (`licenses/MoGe-LICENSE.txt`, the upstream LICENSE file) | `src/fastdo/geometry/moge_overlap.py` |
| SAM 3D Objects, `facebookresearch/sam-3d-objects` @ `875b0100` (as pinned inside DO AS I DO), a SAM Material of Meta | SAM License (`licenses/SAM_LICENSE`); the reproduced expression is distributed under it, the rest of each file under Apache-2.0 | `src/fastdo/shape/sam3d_cuda_graph_slat.py` (`_original_cu_seqlens`: the `cu_seqlens` expression of `full_attn.sparse_scaled_dot_product_attention`) and `src/fastdo/shape/sam3d_cuda_graph.py` (`timestep_embedding`: the expression of `TimestepEmbedder.timestep_embedding`), each checked bit for bit against the original at run time; and the parity reference `_reference_cubes_to_verts` in `tests/shape/test_sam3d_determinism.py` (`utils_cube.cubes_to_verts`) |
| SAM 3D Objects, `facebookresearch/sam-3d-objects` @ `875b0100` (as pinned inside DO AS I DO), `sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py` (`postprocess_mesh`, `_fill_holes`), "Copyright (c) Meta Platforms, Inc. and affiliates." | SAM License (`licenses/SAM_LICENSE`); the adapted parts are distributed under it, the rest of the file under Apache-2.0 | `src/fastdo/shape/fast_postprocess.py`: an adaptation of the mesh postprocess. It keeps the outer-face visibility rule (quantile 0.75, clamped to [0.25, 0.5]), the dual-graph mincut with edge weights x 1000 and the meshfix hole fill (`fill_small_boundaries(nbe=..., refine=True)`), and rewrites the camera, rasterisation and graph-building code for speed |

`src/fastdo/amortised/sac.py` re-implements FlashSAC's training target (FlashDexRetarget, MIT; `rsl_rl_flashsac`,
BSD-3-Clause) from its description and is checked against it; it copies none of its code.

## Copyleft packages imported at run time (not included)

The importing files are Apache-2.0, apart from the SAM-adapted parts of `src/fastdo/shape/fast_postprocess.py`
listed above, which stay under the SAM License. They import, at run time, ultralytics (AGPL-3.0) in the HaWoR process, and pymeshfix
(GPL-3.0), igraph (GPL-2.0-or-later) and plyfile (GPL-3.0-or-later) in the SAM 3D worker process. None of these
packages is included or modified here. If you distribute a combined work, their copyleft terms apply to it.

## Models and runtimes in the production pipeline

| ID | Component | Version | Licence | Source | Obligation |
|---|---|---|---|---|---|
| M1 | SAM 2.1 Hiera-L (mask and prompt stages) | `sam2` git `2b90b9f5`; HF `facebook/sam2.1-hiera-large` @ `665f8e2a` | Apache-2.0 | sam2 README "License" | Apache notices if redistributed |
| M2 | SAM 3D Objects (shape stage) | HF `facebook/sam-3d-objects` @ `2e735550` (gated); GitHub LICENSE @ `f91db411` | SAM License (last updated 2025-11-19) | the repository LICENSE | Acknowledge SAM in publications; ship the licence with any SAM Materials; trade controls. The SAM License does not say whether generated meshes are derivative works. The model licence allows commercial use on our reading, but the **shape stage as run is non-commercial**: nvdiffrast and kaolin (below, "Shape-stage environment") run in its process |
| M3 | MoGe-2 ViT-L (intrinsics, geometry) | MoGe git `b942f00b`; HF `Ruicheng/moge-2-vitl` @ `39c4d5e9` | MIT (code and weights) | README; HF card | MIT notice |
| M4 | CoTracker3 online (tracks stage) | torch.hub `facebookresearch/co-tracker` at commit `82e02e80` by default (the commit the paper's runs loaded: their cached `main` equals it, 74/74 files; `FASTDO_COTRACKER_HUB_REF` overrides it); `scaled_online.pth` from HF `facebook/cotracker3` @ `bf55ea50` (sha256 `205d3478…`) | Checkpoint and most upstream code: CC BY-NC 4.0. Upstream lists separate MIT and Apache-2.0 components. | hub `LICENSE.md`; HF `facebook/cotracker3` | Attribution; non-commercial |
| M5 | HaWoR (hands stage) | HaWoR `2c3fa0c0` inside do-as-i-do `824591b`; HF `ThunderVVV/HaWoR` @ `da6335f4` | CC BY-NC-ND 4.0 | `HaWoR/license.txt`; README | Non-commercial; NoDerivatives; attribution. The licence does not say whether predicted MANO parameters are Adapted Material |
| M6 | WiLoR hand detector (via HaWoR) | `detector.pt` (sha256 `5ef3df44e42d...`) | **Conflicting:** `cc-by-nc-2.0` (HF space card `rolpotamias/WiLoR`) vs CC BY-NC-ND 4.0 (WiLoR GitHub README and `license.txt` @ `fcb91131`); ultralytics states its trained YOLO models are AGPL-3.0. The stricter reading is CC BY-NC-ND 4.0 | as stated | Attribution; non-commercial; NoDerivatives on the stricter reading |
| M7 | ultralytics (YOLO runtime for M6) | 8.4.36 | AGPL-3.0 | package METADATA; LICENSE | Not included. Our HaWoR driver files are Apache-2.0 and import it at run time (`run/pipeline/_hawor_batched_driver.py`, `_hawor_driver_models.py`, `_hawor_trace.py`; six more `src/fastdo` files run in the same process: `hands/hawor_boxes.py`, `hands/hawor_plan.py`, `hands/hawor_core.py`, `hands/hand_gaps.py`, `hands/hand_npz.py`, `bench/gpu_budget.py`). If you distribute a combined work, AGPL-3.0 applies to it (including section 13, network use) |
| M8 | smplx (MANO layer inside HaWoR) | 0.1.28 | MPI "Software Copyright License for non-commercial scientific research purposes" | dist-info `LICENSE` | Non-commercial; not redistributable; never vendored |
| M9 | MANO hand model | user's licensed copy (never committed) | MANO licence | https://mano.is.tue.mpg.de/license.html | Non-commercial; "may not be reproduced, modified and/or made available in any form to any third party"; cite MANO. Never ship the `.pkl`, its vertices or the finger-label cache |
| M10 | Hands23 detector (prompt stage, internet runs) | code `EvaCheng-cty/hands23_detector` @ `1bc0f919`; `model_hands23.pth` | code MIT; weights carry no licence text; training data CC BY-NC 4.0 | repository (README and LICENSE) | Non-commercial on the conservative reading; cite Hands23, COCO, VISOR, Internet Articulation |
| M11 | DINOv2-L (gate dataset only; not in production) | HF `facebook/dinov2-large` @ `47b73eef` | Apache-2.0 | HF card | Notice |
| M13 | MoGe ViT-L v1 (inside SAM 3D: the pointmap of `pipeline.compute_pointmap`) | HF `Ruicheng/moge-vitl` @ `ad326bfb`; moge git `a8c37341` (sam3d env) | MIT (MoGe repository) | SAM 3D `checkpoints/pipeline.yaml` `depth_model` | MIT notice. Fetched by `scripts/fetch_weights.py` and read at that revision (the SAM 3D worker's `--hf-revision`); see `release/MODELS.md` |
| M14 | DINOv2 ViT-L/14 with registers (inside SAM 3D: image conditioner of both flow generators) | `dinov2_vitl14_reg4_pretrain.pth` (sha256 `36e4deff…`) via torch.hub `facebookresearch/dinov2`, which SAM 3D loads with no commit; `scripts/fetch_weights.py` unpacks commit `7764ea0f` (the code the paper's runs loaded, file for file) into `$TORCH_HOME/hub/facebookresearch_dinov2_main` | Apache-2.0 (the backbone code and weights) | dinov2 `LICENSE` | Notice. What is loaded: at `7764ea0f`, `hubconf.py` also imports the Cell-DINO modules when torch.hub loads it. Those files state CC BY 4.0 (`LICENSE_CELL_DINO_CODE`) and, in `dinov2/hub/cell_dino/backbones.py`, CC BY-NC. SAM 3D calls only `dinov2_vitl14_reg`, and no Cell-DINO weights are loaded |
| M12 | MuJoCo / MuJoCo-Warp / Warp | `mujoco` 3.13.0, `mujoco-warp` 3.13.0, `warp-lang` 1.15.0 (baseline env `mujoco` 3.4.0) | Apache-2.0 | PyPI metadata | Notice |

## Robot models and retargeting assets

| ID | Component | Version | Licence | Obligation |
|---|---|---|---|---|
| R1 | Unitree G1 29-DoF + hand model (model of record) | `unitree_ros` @ `5994d4fa` (`robots/g1_description/g1_29dof_with_hand_rev_1_0.{xml,urdf}`) | BSD 3-Clause | Keep the copyright notice and disclaimer if redistributed |
| R2 | Dex3-1 URDF + DexPilot config (`xr_teleoperate`) | @ `817fb00c` | Apache-2.0 | NOTICE and licence copy |
| R3 | `unitree_sdk2_python` | @ `814556d1` | BSD 3-Clause | Notice |
| R4 | `dex-retargeting` (Unitree fork) | @ `d7753d3` | MIT (Unitree 2025; Yuzhe Qin 2023) | Notice |
| R5 | MuJoCo Menagerie `unitree_g1` (not the model of record) | menagerie @ `0059d43` | BSD 3-Clause | Notice |
| R6 | Sharpa Wave hand assets (inside DO AS I DO) | `deployment/mujoco_replay/assets/sharpa_assets/SharpaWave_URDF_XML_USD_V3.0.0`; `retargeting/retargeting/assets/robots/sharpa` | No licence file in either directory; the DO AS I DO top level is MIT, and it does not say whether the vendor's assets fall under it | This repository ships none of them; outputs it writes hold joint trajectories, not the meshes or XML |

## Code we run or compare against (not included; cloned separately)

| ID | Component | Version | Licence |
|---|---|---|---|
| C1 | DO AS I DO (baseline; a git submodule pointer, not its files) | `malik-group/do-as-i-do` @ `824591b` | MIT, "Copyright (c) 2026 malik-group". The whisking demo video carries no licence statement |
| C2 | SAM 3 (DO AS I DO's segmenter) | HF `facebook/sam3` @ `3c879f39` | SAM License |
| C3 | TAPNet (inside DO AS I DO) | as pinned there | The TAPNet code and Google DeepMind's linked BootsTAPIR v2 PyTorch checkpoint are published under Apache-2.0. |
| C4 | FoundationPose (external evaluation only; this repository has no code that runs it and ships no weights) | NVlabs/FoundationPose a1b694b83e633c2cb6115b9063d940a687759392; released refiner 2023-10-28-18-33-37 and scorer 2024-01-11-20-02-45 (the public weights) | NVIDIA Source Code License: non-commercial use only |
| C5 | Open-CHOIR and its assets (head-to-head; outputs never redistributed) | `open-choir` @ `dda0d727` | code MIT; assets include CC BY-NC 4.0 models (Depth-Anything-V2-L; DA-V2 Metric Hypersim L is Apache-2.0 on its HF card and CC BY-NC 4.0 in the upstream README), CoTracker3 (CC BY-NC 4.0), the WiLoR detector (as M6), and checkpoints whose licence is unstated |
| C6 | FlashDexRetarget (physics baseline) | `flashdexretarget` @ `7919f8a` | MIT (Holiday Robotics 2026) |
| C6a | `rsl_rl_flashsac` (inside FlashDexRetarget; imported read-only by `tests/amortised/vendored_flashsac.py` from `$FASTDO_FLASHSAC_ROOT`) | as pinned there | BSD-3-Clause |
| C7 | MV-SAM3D (ablation) | `mv-sam3d` @ `abb04b5` | SAM License (identical to M2's) |
| C8 | V2D toolkit (benchmark tooling) | `video_to_data` @ `193382a` | documentation CC BY-4.0; source code Apache-2.0 |
| C9 | Other Unitree repositories (reference only) | as cloned | BSD-3 (`unitree_mujoco`, `unitree_rl_gym`, `unitree_ros2`, `unitree_sdk2`); Apache-2.0 (`unitree_sim_isaaclab`); MIT (`televuer`) |

## Datasets (not included)

The download scripts in `run/data/` fetch from the datasets' own hosts; each dataset keeps its terms. DexYCB and
HOI4D are CC BY-NC 4.0. OakInk2, HO3D-v3 and Ego4D are under their own licences or agreements, and 100DOH clips are
YouTube videos (you are responsible for complying with YouTube's Terms of Service and the videos' copyright). The V2D
Track 3 public split, used by the example, is CC BY 4.0.

## Derived data in this repository (not Apache-2.0)

A few files hold data derived from third-party datasets or model outputs. They are not covered by the Apache-2.0
licence of the code; each follows the terms of what it was derived from:

- `reproduce/heldout/units_hoi4d_72.json` and `reproduce/heldout/units_dexycb_s08_108.json` (the two unit-list
  files; no per-unit error ships): they list identifiers of HOI4D and DexYCB sequences and windows, both datasets
  CC BY-NC 4.0, so non-commercial use only, with the attribution in `reproduce/heldout/README.md`.
- `reproduce/heldout/e1-power-size.json`: the study's simulation rows and the thresholds computed from them. It
  names the two corpora and contains no individual sequence, view or window identifiers.
- `reproduce/heldout/qc_rule_constants.json`: the QC rule's fitted constants, fitted on DexYCB (CC BY-NC 4.0)
  development clips; non-commercial use only.
- These four files are part of this release; `reproduce/heldout/README.md` gives their sources and sha256.
- `release/reference/v2d_t3_ep011_cam_a/`: derived from a CC BY 4.0 clip (V2D Challenge Track 3, NVIDIA; credit in
  its `ATTRIBUTION.txt`, and the frames in `expected_projection.png`) and produced by the pipeline's models. Its hand
  arrays (MANO pose, shape and translation, and joints) are HaWoR outputs on the MANO model: they follow HaWoR's
  CC BY-NC-ND 4.0 and MANO's licence (non-commercial; the HaWoR licence does not say whether such outputs are Adapted
  Material, M5 above). The MANO mesh arrays, vertices and faces, are not shipped.

## Python dependencies of `fastdo` (main env)

Generated by `scripts/dependency_licences.py` from `uv.lock` and the installed packages' metadata (run in the
environment `uv sync --frozen --extra dev --extra frontend --extra viser` builds). They are dependencies, not included
code, except the rows for `fastdo` and `nullaudit` themselves, which this repository ships. The source column
names PyPI for every package that comes from a registry; `torch` (2.9.1+cu128) in fact comes from the PyTorch cu128
index recorded in `uv.lock`. viser (MIT) and the packages it adds (websockets, msgspec, imageio, requests, zstandard and others) are installed
only by the `viser` extra, for `scripts/view_clip.py`. Notes:
`opencv-python` (pulled in by MoGe) ships prebuilt Qt and FFmpeg libraries inside its wheel, which are LGPL;
`libigl` is MPL-2.0 (sections 3.1 and 3.2 govern distribution of its own files, in source or executable form,
modified or not; the copyleft is per file); `pipeline` and `utils3d` are git
dependencies (EasternJournalist, MIT). For `cycler` and `scipy` the table displays the first line of the licence
field, which is a copyright notice for these packages. Beside it, as read in the installed packages on 2026-11-19: both carry the
classifier "BSD License", and the licence file of each is the three-clause BSD text. The `scipy` 1.18.1 wheel (Linux,
CPython 3.12) also bundles libraries with their own notices in its `LICENSE.txt`: OpenBLAS (BSD-3-Clause), LAPACK
inside it (BSD-3-Clause-Open-MPI), the GCC runtime library (GPL-3.0-or-later with the GCC runtime library exception)
and libquadmath (LGPL-2.1-or-later). Other platforms' wheels can bundle other files; read the wheel you install.

<!-- BEGIN generated by scripts/dependency_licences.py -->
| Package | Version | Source | Licence (declared metadata) |
|---|---|---|---|
| absl-py | 2.5.0 | PyPI | Apache-2.0 |
| annotated-doc | 0.0.5 | PyPI | MIT |
| annotated-types | 0.8.0 | PyPI | MIT |
| antlr4-python3-runtime | 4.9.3 | PyPI | BSD |
| anyio | 4.15.1 | PyPI | MIT |
| ast-serialize | 0.11.1 | PyPI | MIT |
| audioop-lts | 0.2.2 | PyPI | not installed in this environment (not read) |
| brotli | 1.2.0 | PyPI | MIT |
| certifi | 2026.7.22 | PyPI | MPL-2.0 |
| charset-normalizer | 3.5.2 | PyPI | MIT |
| click | 8.5.0 | PyPI | BSD-3-Clause |
| colorama | 0.4.6 | PyPI | not installed in this environment (not read) |
| contourpy | 1.4.0 | PyPI | BSD-3-Clause |
| cycler | 0.12.1 | PyPI | Copyright (c) 2015, matplotlib project |
| decord | 0.6.0 | PyPI | APACHE |
| etils | 1.14.0 | PyPI | Apache Software License |
| fastapi | 0.141.1 | PyPI | MIT |
| fastdo | 0.1.0 | this repository | this repository (Apache-2.0) |
| filelock | 3.32.6 | PyPI | MIT |
| fonttools | 4.65.0 | PyPI | MIT |
| fsspec | 2026.7.0 | PyPI | BSD-3-Clause |
| glcontext | 3.0.0 | PyPI | MIT |
| glfw | 2.10.2 | PyPI | MIT |
| gradio | 6.27.0 | PyPI | Apache-2.0 |
| gradio-client | 2.7.0 | PyPI | Apache-2.0 |
| groovy | 0.1.2 | PyPI | MIT License |
| h11 | 0.16.0 | PyPI | MIT |
| hf-gradio | 0.4.1 | PyPI | MIT |
| hf-xet | 1.6.0 | PyPI | Apache-2.0 |
| httpcore | 1.0.9 | PyPI | BSD-3-Clause |
| httpx | 0.28.1 | PyPI | BSD-3-Clause |
| huggingface-hub | 1.31.0 | PyPI | Apache-2.0 |
| hydra-core | 1.3.6 | PyPI | MIT |
| idna | 3.19 | PyPI | BSD-3-Clause |
| imageio | 2.38.1 | PyPI | BSD-2-Clause |
| iniconfig | 2.3.0 | PyPI | MIT |
| iopath | 0.1.10 | PyPI | MIT licensed, as found in the LICENSE file |
| jinja2 | 3.1.6 | PyPI | BSD License |
| kiwisolver | 1.5.1 | PyPI | BSD-3-Clause |
| libigl | 2.6.3 | PyPI | Mozilla Public License 2.0 (MPL 2.0) |
| librt | 0.15.0 | PyPI | MIT |
| markdown-it-py | 4.2.0 | PyPI | MIT License |
| markupsafe | 3.0.3 | PyPI | BSD-3-Clause |
| matplotlib | 3.11.2 | PyPI | License agreement for matplotlib versions 1.3.0 and later |
| mdurl | 0.1.2 | PyPI | MIT License |
| moderngl | 5.12.0 | PyPI | MIT |
| moge | 2.0.0 | git https://github.com/microsoft/MoGe.git @ b942f00b | MIT |
| mpmath | 1.3.0 | PyPI | BSD |
| msgspec | 0.22.0 | PyPI | BSD-3-Clause |
| mujoco | 3.13.0 | PyPI | Apache-2.0 |
| mujoco-warp | 3.13.0 | PyPI | not installed in this environment (not read) |
| mypy | 2.3.1 | PyPI | MIT |
| mypy-extensions | 1.1.0 | PyPI | MIT |
| networkx | 3.6.1 | PyPI | BSD-3-Clause |
| nullaudit | 0.1.0 | this repository | this repository (Apache-2.0) |
| numpy | 2.5.3 | PyPI | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-cublas-cu12 | 12.8.4.1 | PyPI | NVIDIA Proprietary Software |
| nvidia-cuda-cupti-cu12 | 12.8.90 | PyPI | NVIDIA Proprietary Software |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | PyPI | NVIDIA Proprietary Software |
| nvidia-cuda-runtime-cu12 | 12.8.90 | PyPI | NVIDIA Proprietary Software |
| nvidia-cudnn-cu12 | 9.10.2.21 | PyPI | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufft-cu12 | 11.3.3.83 | PyPI | NVIDIA Proprietary Software |
| nvidia-cufile-cu12 | 1.13.1.3 | PyPI | NVIDIA Proprietary Software |
| nvidia-curand-cu12 | 10.3.9.90 | PyPI | NVIDIA Proprietary Software |
| nvidia-cusolver-cu12 | 11.7.3.90 | PyPI | NVIDIA Proprietary Software |
| nvidia-cusparse-cu12 | 12.5.8.93 | PyPI | NVIDIA Proprietary Software |
| nvidia-cusparselt-cu12 | 0.7.1 | PyPI | NVIDIA Proprietary Software |
| nvidia-nccl-cu12 | 2.27.5 | PyPI | BSD-3-Clause |
| nvidia-nvjitlink-cu12 | 12.8.93 | PyPI | NVIDIA Proprietary Software |
| nvidia-nvshmem-cu12 | 3.3.20 | PyPI | BSD-3-Clause |
| nvidia-nvtx-cu12 | 12.8.90 | PyPI | Apache 2.0 |
| omegaconf | 2.3.1 | PyPI | BSD License |
| opencv-python | 5.0.0.93 | PyPI | Apache 2.0 |
| opencv-python-headless | 5.0.0.93 | PyPI | Apache 2.0 |
| orjson | 3.12.0 | PyPI | MPL-2.0 AND (Apache-2.0 OR MIT) |
| packaging | 26.3 | PyPI | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.5 | PyPI | BSD 3-Clause License |
| pathspec | 1.1.1 | PyPI | Mozilla Public License 2.0 (MPL 2.0) |
| pillow | 12.3.0 | PyPI | MIT-CMU |
| pipeline | 1.0.0 | git https://github.com/EasternJournalist/pipeline.git @ 866f059d | MIT |
| pluggy | 1.6.0 | PyPI | MIT |
| portalocker | 4.3.1 | PyPI | BSD-3-Clause |
| pyarrow | 25.0.1 | PyPI | Apache-2.0 |
| pydantic | 2.13.5 | PyPI | MIT |
| pydantic-core | 2.46.5 | PyPI | MIT |
| pydub | 0.25.1 | PyPI | MIT |
| pygments | 2.21.0 | PyPI | BSD-2-Clause |
| pyopengl | 3.1.10 | PyPI | BSD License |
| pyparsing | 3.3.2 | PyPI | MIT |
| pytest | 9.1.1 | PyPI | MIT |
| python-dateutil | 2.9.0.post0 | PyPI | Dual License |
| python-multipart | 0.0.32 | PyPI | Apache-2.0 |
| pytz | 2026.3.post1 | PyPI | MIT |
| pyyaml | 6.0.3 | PyPI | MIT |
| regex | 2026.9.10 | PyPI | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | PyPI | Apache-2.0 |
| rich | 14.3.4 | PyPI | MIT |
| rtree | 1.4.1 | PyPI | MIT |
| ruff | 0.16.7 | PyPI | MIT |
| safehttpx | 0.1.7 | PyPI | MIT License |
| safetensors | 0.8.0 | PyPI | Apache Software License |
| sam-2 | 1.0 | git https://github.com/facebookresearch/sam2.git @ 2b90b9f5 | Apache 2.0 |
| scipy | 1.18.1 | PyPI | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. |
| semantic-version | 2.10.0 | PyPI | BSD |
| setuptools | 84.0.0 | PyPI | MIT |
| shellingham | 1.5.4 | PyPI | ISC License |
| six | 1.17.0 | PyPI | MIT |
| starlette | 1.6.0 | PyPI | BSD-3-Clause |
| sympy | 1.14.0 | PyPI | BSD |
| threadpoolctl | 3.6.0 | PyPI | BSD-3-Clause |
| tokenizers | 0.23.2 | PyPI | Apache Software License |
| tomlkit | 0.14.0 | PyPI | MIT |
| torch | 2.9.1+cu128 | PyPI | BSD-3-Clause |
| torchvision | 0.24.1 | PyPI | BSD |
| tqdm | 4.70.1 | PyPI | MPL-2.0 AND MIT |
| transformers | 5.17.0 | PyPI | Apache 2.0 License |
| trimesh | 5.1.0 | PyPI | The MIT License (MIT) |
| triton | 3.5.1 | PyPI | MIT License |
| typer | 0.27.2 | PyPI | MIT |
| typing-extensions | 4.16.0 | PyPI | PSF-2.0 |
| typing-inspection | 0.4.4 | PyPI | MIT |
| tzdata | 2026.4 | PyPI | not installed in this environment (not read) |
| urllib3 | 2.8.0 | PyPI | MIT |
| utils3d | 1.3 | git https://github.com/EasternJournalist/utils3d.git @ 3fab839f | MIT |
| uvicorn | 0.52.4 | PyPI | BSD-3-Clause |
| viser | 1.1.1 | PyPI | MIT |
| warp-lang | 1.15.0 | PyPI | not installed in this environment (not read) |
| websockets | 16.1.1 | PyPI | BSD-3-Clause |
| yt-dlp | 2026.8.19 | PyPI | Unlicense |
| zipp | 4.1.0 | PyPI | MIT |
| zstandard | 0.25.0 | PyPI | BSD-3-Clause |
<!-- END generated by scripts/dependency_licences.py -->

### Shape-stage environment (`sam3d` env)

The SAM 3D worker (`src/fastdo/shape/sam3d_worker.py` and the 7 modules it imports: `fast_postprocess`,
`sam3d_cuda_graph`, `sam3d_cuda_graph_slat`, `sam3d_determinism`, `sam3d_multistep`, `sam3d_spconv_pin`, `wall_qc`)
runs in one process with pymeshfix and igraph. `fast_postprocess.py` is the only file of ours that imports them
directly; SAM 3D's own `postprocessing_utils` also imports both at module level. The same process runs NVIDIA's
non-commercial nvdiffrast (through `fast_postprocess.py:218`, utils3d's `RastContext`, in the hole filling that is on
by default) and, on a static reading, kaolin, whose import loads its non-commercial `non_commercial` package
(imported by SAM 3D's `representations/mesh/flexicubes/flexicubes.py`). **The shape stage is therefore
non-commercial as run** (nvdiffrast alone suffices).

Licences read from installed package METADATA at 2026-10-27T05:21:52Z.

| Package | Version | Licence |
|---|---|---|
| pymeshfix | 0.17.0 | **GPL-3.0**; imported at run time, not included |
| igraph (formerly python-igraph) | 0.11.8 | **GPL-2.0-or-later** (upstream; METADATA says "GNU General Public License (GPL)"); imported at run time, not included |
| pyvista | 0.46.4 | MIT |
| fast_simplification | 0.1.12 | MIT |
| utils3d | 0.0.2 | MIT |
| loguru | 0.7.2 | MIT |
| plyfile | 1.1.3 | **GPL-3.0-or-later** (METADATA). Imported at module level by SAM 3D's `representations/gaussian` (via `representations/__init__.py`), so on a static reading it is in-process in the SAM 3D worker; not verified at run time. Not imported by our code; not included |
| nvdiffrast | 0.4.0 (git `253ac4fc`) | **NVIDIA Source Code License (Nvdiffrast)**: use "non-commercially", research or evaluation only (clause 3.3). In-process, through our `fast_postprocess.py` (utils3d `RastContext`) and SAM 3D's `postprocessing_utils` |
| kaolin | 0.18.0 | Apache-2.0, **except `kaolin/non_commercial/`, under NVIDIA's `LICENSE.NSCL`** (non-commercial, research or evaluation only). `kaolin/__init__.py` imports `non_commercial` unconditionally, so importing kaolin loads that code. SAM 3D's FlexiCubes imports kaolin (on a static reading), which puts it in the shape process; not imported by our code |

Not read from metadata here: coacd and TRELLIS (sam3d env) and detectron2 (hands23 env). triton is in the main-env table above.

### Hand-stage environment (`hawor` env)

Licences read from installed package METADATA at 2026-11-07 (`release/MODELS.md`, "Copyleft and other licence
flags"). ultralytics (M7) and smplx (M8) are above.

| Package | Version | Licence |
|---|---|---|
| evo | 1.35.2 | **GPL-3.0**; installed, imported only by HaWoR's `lib/pipeline/masked_droid_slam.py`, which the hands driver does not import on a static reading |
| PyQt5 | 5.15.11 | **GPL-3.0**; installed, not imported by the pipeline on a static reading |
| plyfile | 1.1.3 | **GPL-3.0-or-later**; installed |
| easydict, dominate, PyQt5-Qt5 | 1.13, 2.9.1, 5.15.18 | LGPL-3.0 |

