# Installation

The full installation notes. The README's "Installation" gives the short path; paths in this file are relative to the repository root.

Requirements: Linux x86_64, Python 3.12, [uv](https://docs.astral.sh/uv/), conda (for the worker environments),
`git`, and `ffmpeg` and `ffprobe` on `PATH` (the ffmpeg package provides both). The model stages need an NVIDIA
GPU; the code was developed on one H100 80 GB with CUDA 12.8 wheels. A GPU run needs `FASTDO_GPU_MEM_GB=32`: the
default per-process cap is 12 GiB, below the stage budgets, and the shape stage (SAM 3D Objects) alone needs 19 GiB.
The GPU example needs a GPU with at least 24 GB of memory (`release/QUICKSTART.md`, "Commands on this page"). The
authors ran the model stages only on a GPU, and this page does not claim that they work on CPU. On a GPU other than
an H100, set `TORCH_CUDA_ARCH_LIST` before `scripts/setup_all.sh` (for example `8.0` on an A100): the script
defaults to `9.0`, the H100 the authors tested, and passes the value to both worker builds. Only the H100 build
was tested.

```bash
git clone https://github.com/BillyChern/HandReel && cd HandReel
bash scripts/fetch_baseline.sh           # DO AS I DO at the pinned commit, into third_party/do-as-i-do
bash scripts/setup_all.sh --dry-run      # prints every step of release/QUICKSTART.md step 2
bash scripts/setup_all.sh                # builds them: the uv env, the sam3d and hawor conda envs, HaWoR
```

Besides the envs, `setup_all.sh` clones HaWoR (the malik-group fork, at the commit DO AS I DO pins) to `../hawor`,
next to this checkout (`--hawor DIR` chooses another place; keep it outside `third_party/`). That clone is the HaWoR
checkout the rest of this page uses: `FASTDO_HAWOR_ROOT` (below) and `--hawor-root ../hawor` in "Weights". To clone
it by hand instead, follow `release/QUICKSTART.md` step 2 (`git clone https://github.com/malik-group/HaWoR.git ../hawor`,
then the pinned `checkout` and `submodule update`).

`setup_all.sh` needs conda: put it on `PATH`, or set `FASTDO_CONDA_BASE` to your conda or miniforge root (the script
then uses `$FASTDO_CONDA_BASE/bin/conda`). Without either, even `--dry-run` prints the plan and then exits 2 with
`preflight: FAIL: conda not found`. It refuses to start when your conda already has an env named `sam3d` or `hawor`
(`conda create -y` would replace it) unless you pass `--replace-envs`. A cold build needs about 60 GB of disk before
the weights (envs plus the conda and uv package caches; `release/QUICKSTART.md` step 0).

`scripts/fetch_baseline.sh` clones DO AS I DO at commit `824591b808c342b20079c3b4198a8c2bdf88c74e` (MIT), including
its demo clip, into `third_party/do-as-i-do`; the SAM 3D and HaWoR code the pipeline runs lives there. The repository
records no submodule commit for that directory, so `git submodule update` leaves it empty; the script works the same
in a git clone and in an archive without git metadata.

For a CPU-only look, `uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser` builds only the
main environment (it installs exactly what `uv.lock` pins; the torch wheel also runs without a GPU; the `viser` extra
is the 3D viewer of the README's "Try it on CPU, no weights").
`release/QUICKSTART.md` walks through every step by hand and says which commands were run for this release. Its
"Notes" cover installing from an archive without git metadata, the expected pip `ERROR:` line of the `sam3d` step,
uv's harmless decord reinstall message and the optional alongray package.

The pipeline runs in one uv environment plus conda environments, because several models pin incompatible stacks:

| env | what runs in it | recipe | how fastdo finds it |
|---|---|---|---|
| uv (`.venv`) | the engine, mask, intrinsics/geometry, tracks, solve, qc, retarget | `uv sync` above | no setting needed |
| `sam3d` (conda) | the SAM 3D shape worker (subprocess) | `scripts/setup_all.sh` (locks in `release/env/`) | `FASTDO_SAM3D_PYTHON=<conda>/envs/sam3d/bin/python` |
| `hawor` (conda) | the HaWoR hand driver (subprocess) | `scripts/setup_all.sh` (locks in `release/env/`) | config `hands.hawor_python`, `FASTDO_HAWOR_ROOT` |
| `retargeting` (conda) | DO AS I DO scene build and physics drivers | `run/setup/build_baseline_envs.sh retargeting` (no lock) | `${paths.conda_base}/envs/retargeting` |
| Hands23 (venv, detectron2) | the internet-video prompt detector (subprocess) | `release/env/hands23.pip.txt`, `release/QUICKSTART.md` step 2 | `FASTDO_HANDS23_ROOT` or `FASTDO_HANDS23_PYTHON` |
| `mjwarp314` site (optional) | MuJoCo-Warp 3.14 for the GPU physics arm | `run/setup/make_mjwarp314_site.sh <venv-site-packages> <dest>` | `FASTDO_MJWARP314_SITE=<dest>` |

`release/env/` holds the pinned worker-environment locks; `release/env/verify.sh` checks an installed env against
them. The lock-based route, `scripts/setup_all.sh` (`release/QUICKSTART.md` step 2), is the supported one.
`run/setup/build_baseline_envs.sh` is the from-source recipe. It installs nvdiffrast, the `hawor` pytorch3d, chumpy
and GeoCalib unpinned, so it is not byte-identical to the locks.

## Host paths

Every host-specific path is a config key with an environment override (`run/conf/config.yaml` `paths.*`, mirrored in
`fastdo.hostpaths`). The defaults name a placeholder layout under `/opt/fastdo`; an entry point that needs a path that
is still a placeholder stops and names the variable to set:

| variable | meaning |
|---|---|
| `FASTDO_ROOT` | this checkout (`paths.root`; default `.`) |
| `FASTDO_EXT_ROOT` | the directory that holds this checkout and its siblings (such as the HaWoR checkout `../hawor` of "Installation") |
| `FASTDO_PRIMARY_ROOT`, `FASTDO_SCRATCH_ROOT` | the primary checkout and its scratch (`temp/`); note 1 |
| `FASTDO_CONDA_BASE` | your conda or miniforge root |
| `FASTDO_SAM3D_PYTHON` | the `sam3d` env's interpreter |
| `FASTDO_MANO_DIR`, `FASTDO_MANO_LEFT_DIR` | the directories holding your `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` |
| `FASTDO_HAWOR_ROOT` | your HaWoR checkout, e.g. `../hawor` from `scripts/setup_all.sh`. Set it: the config falls back to the baseline's `third_party/do-as-i-do/reconstruction/modules/HaWoR`, but `scripts/fetch_weights.py` never writes under `third_party/` and needs `--hawor-root` or this variable, so the HaWoR weights are never there |
| `FASTDO_HAWOR_PYTHON` | the HaWoR interpreter for `run/pipeline/build_hoi4d_hands.py` |
| `FASTDO_HANDS23_ROOT` (`_PYTHON`, `_WEIGHTS`) | the Hands23 env, weights and repo |
| `FASTDO_UNITREE_ROS` | the official `unitree_ros` clone (Unitree G1 model) |
| `FASTDO_EGO4D_LISTS` | your copies of the Ego4D video-uid lists (licensed metadata, not shipped) |
| `FASTDO_V2D_TRACK3_ROOT` | the V2D Track 3 public split (`fastdo.bench.v2d_track3`) |
| `FASTDO_UNIT_REGISTRY` | the amortised-unit registry; note 2 |
| `FASTDO_GPU_MEM_GB` | the per-process CUDA memory cap in GiB; note 3 |
| `FASTDO_GPU_TOOLS_DIR` | optional: a directory shared by every GPU job on a host; note 4 |
| `FASTDO_COTRACKER_HUB_REF` | optional: another CoTracker3 torch.hub ref; note 5 |
| `FASTDO_DRIFTQC_ROOT` | tests only: the frozen drift-QC constants (not shipped); note 6 |
| `FASTDO_HOLD_EVAL` | tests only: a local copy of HOLD's evaluation module (not shipped); note 7 |

Notes:

1. The primary checkout is the one that holds data and outputs when you work from several git worktrees; with one
   checkout, it is that checkout.
2. A directory where each running GPU unit of `fastdo.amortised.unit` writes `<pid>.json`, so units can see each
   other's GPU memory; only those units read it.
3. Default 12; `<= 0` disables the cap. Set it to `32` for a GPU run: SAM 3D alone needs 19.
4. Set it to make amortised units take turns on `gpu.lock` there. Unset: no shared lock.
5. Default: the pinned commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`; `main` loads the moving branch.
6. A directory holding the frozen drift-QC constants: three calibration files of the authors, at
   `temp/driftqc/constants_v2_2.json`, `temp/trackweight/constants_v2.json` and `temp/trackweight/code/v1_weights.py`
   under it. When the variable is unset, the drift-QC tests that need them skip.
7. When unset, `tests/bench/test_metric_frame.py::test_cd_h_matches_hold_bit_for_bit` skips.

The GPU example (`release/example/run_example.py`) needs exactly these four variables:

- `FASTDO_GPU_MEM_GB=32`;
- `FASTDO_CONDA_BASE`;
- `FASTDO_SAM3D_PYTHON`;
- `FASTDO_HAWOR_ROOT`.

On an A100, H100 or H200 it also needs flash_attn 2.8.3 in the `sam3d` env: SAM 3D selects flash_attn on those cards
by name, and the shape stage fails without it. `scripts/setup_all.sh` and `release/QUICKSTART.md` step 2 install the
official prebuilt wheel, checked by sha256, and `release/env/verify.sh` fails on those cards when it does not import.

It sets `paths.root` to its own checkout, and HaWoR reads MANO from the HaWoR checkout (`release/QUICKSTART.md` step 3
copies the files there), so the example does not need `FASTDO_ROOT` or the MANO variables. The CPU commands of "Quick
start" need none of these variables.

The other variables (CPU threads, extra SAM 3D paths, the Hands23 files, the frame-table cache, HaWoR diagnostics)
are listed in `release/QUICKSTART.md`, "Environment variables".
