# Quickstart: reconstruct one window

This page takes you from a fresh clone to one reconstructed window of a public video, with its timing-ledger row.
It complements `README.md`, which has the installation summary, the host-path variables and a licence summary
(full notes: `docs/licensing.md`). Where the two overlap, the README is the reference. `scripts/setup_all.sh` runs step 2 below in one command (`--dry-run` prints
it).

**Status of this page.** Steps 1, 2 (main env), 3 (the table) and the CPU smoke test of step 4 run without a GPU. The
GPU run of step 4 was run for this release from clean extracts, with a `sam3d` env completed by step 2's flash_attn
and `sam3d_objects` commands and the development host's `hawor` env, weights and MANO, not with envs built here from
nothing (see "Reference output" below). An independent build of the envs from nothing, with `scripts/setup_all.sh` on a
fresh miniforge (about 25 min with a filled pip cache), passed `release/env/verify.sh`, and the GPU run of step 4 on
those envs gave stage artifacts that match the reference output apart from timing fields (the reference `README.md`
lists what was compared and what was excluded); its SAM 3D weights and MANO files were copied
from an existing host, so the gated SAM 3D download itself was not tested. Its stages are the ones the paper's runs
use. Step 5's one command (the released table check) runs without a GPU. "Commands on this
page", at the end, says which commands were run for this release and what each of the others needs.

## 0. What you need

| | requirement |
|---|---|
| OS | Linux x86_64 |
| GPU | one NVIDIA GPU with at least 24 GB. SAM 3D alone needs 19 GB; GPU runs set `FASTDO_GPU_MEM_GB=32` (step 4). Developed and timed on one H100 80 GB (sm_90); no other GPU has been tested |
| driver | tested with NVIDIA driver **595.91.07** on the H100 (`release/reference/v2d_t3_ep011_cam_a/example_report.json`); an earlier development phase ran a 570-series driver. No minimum driver has been established for this stack: check NVIDIA's CUDA 12.8 release notes for your card |
| CUDA | no system toolkit for the main env; CUDA 12.8 toolkit packages in the conda envs (note below) |
| disk | about 25 GB of weights (`release/MODELS.md`), about 18 GB for the `sam3d` env, 14 GB for `hawor`, 7-8 GB for the main env, 7 GB for the optional `hands23` venv. A cold build also fills about 18 GB of conda package cache and about 14 GB of uv cache: an independent cold build of step 2 used about 60 GB in all, before the weights |
| tools | [uv](https://docs.astral.sh/uv/) >= 0.9, miniforge or conda, `git`, `ffmpeg` and `ffprobe` on `PATH` (the ffmpeg package provides both) |
| accounts | a Hugging Face login that has accepted the SAM 3D Objects terms; a MANO registration |

Decoder: the main decoder uses the `ffmpeg` and `ffprobe` binaries on its `PATH`. The FFmpeg 7.1.0 lock applies to
the SAM 3D shape worker. For timing or frame comparisons, record the builds of both host tools.

CUDA: the main env needs no system toolkit, because the torch `+cu128` wheels bundle the CUDA 12.8 runtime. The
conda envs bring CUDA 12.8 toolkit packages (`nvcc` 12.8) for the extensions you build from source. **Untested:** a
host with no system toolkit at all; the development host had one. nvdiffrast compiles its CUDA plugin on first use
(JIT) inside the SAM 3D worker. The worker overrides any `CUDA_HOME` you export: it sets it to `CONDA_PREFIX` when
that variable is set, and to its own interpreter's prefix otherwise. A `CONDA_PREFIX` inherited from another
activated env therefore points the compile at that env, not at `sam3d`. If the compile fails, check which prefix the
worker used and repair the CUDA 12.8 toolkit packages there.

We recommend `export HF_HUB_DISABLE_TELEMETRY=1` (`release/example/run_example.py` sets it for its own process
only when the variable is unset; a value you exported stays).

## 1. Code and the DO AS I DO baseline

Clone this repository, then fetch the baseline from the repository root. `fetch_baseline.sh` clones DO AS I DO into
`third_party/do-as-i-do` at the pinned commit, fetches its own submodules (sam-3d-objects lives there) and verifies
them:

```bash
git clone https://github.com/BillyChern/HandReel && cd HandReel
bash scripts/fetch_baseline.sh
```

This repository records no submodule commit for `third_party/do-as-i-do`, so `git clone --recurse-submodules` and
`git submodule update` leave that directory empty; use the script. It works the same in an archive without git
metadata.

If the nested download stops after the baseline reaches commit `824591b808c342b20079c3b4198a8c2bdf88c74e`, run
`git -C third_party/do-as-i-do submodule update --init --recursive`, then `bash scripts/fetch_baseline.sh --check`.
The CPU checks do not need the baseline.

## 2. Environments (pinned)

The reconstruction runs in one uv env and two conda envs, plus an optional venv for the internet-video prompt stage.
Every worker lock below (`sam3d`, `hawor`, `hands23`) was exported from the environment the paper's runs used
(`release/env/export_env_locks.py`); the main env's `uv.lock` comes from `uv` and is not written by that script. The worker locks were then
edited as that script now writes it (chumpy moved to the source builds; nine unused packages dropped from `sam3d`).

| env | runs | lock | build |
|---|---|---|---|
| main (uv) | engine, mask, intrinsics, geometry, tracks, solve, qc; the viewer | `uv.lock` | `uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser` |
| `sam3d` (conda, Python 3.11) | SAM 3D shape worker | `release/env/sam3d.conda.txt` + `sam3d.pip.txt` | below |
| `hawor` (conda, Python 3.10) | HaWoR hand driver | `release/env/hawor.conda.txt` + `hawor.pip.txt` | below |
| `hands23` (venv, Python 3.12) | Hands23 prompt detector (opt-in) | `release/env/hands23.pip.txt` | below |

Pinned stacks: main env torch 2.9.1+cu128; `sam3d` torch 2.8.0+cu128; `hawor` torch 2.9.0+cu128; `hands23` torch
2.8.0+cu128.

**Package sources you are trusting.** The pip locks add `--extra-index-url https://download.pytorch.org/whl/cu128`
(all three) and `--find-links` to NVIDIA's kaolin wheel page (`sam3d`). pip treats an extra index like PyPI for every
package name, so a name published on both is taken from whichever has the pinned version. The locks pin exact
versions but carry no hashes. We did not hash-pin them: `uv pip compile --generate-hashes` cannot run offline from
these locks, because their git sources need the network. Review the two indexes before you install. The conda locks
fetch from `conda.anaconda.org` (conda-forge, nvidia) and `repo.anaconda.com/pkgs`, and Anaconda's Terms of Service
apply to the latter: organisations above its size threshold need a licence.

`conda create -y -n sam3d` replaces an existing env of that name without asking. If your conda already has a
`sam3d` or `hawor` env, remove or rename it first; `scripts/setup_all.sh` refuses to start in that case unless you
pass `--replace-envs`.

```bash
uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}   # your GPU: 9.0 H100, 8.0 A100, 8.6 A40/3090, 8.9 L40S/4090
for env in sam3d hawor; do
  conda create -y -n $env --file release/env/$env.conda.txt
  conda run -n $env pip install -r release/env/$env.pip.txt
done
conda env config vars set LIDRA_SKIP_INIT=1 -n sam3d
```

Two things in this step's output are expected. `conda run -n sam3d pip install -r release/env/sam3d.pip.txt` prints
one `ERROR:` line from pip's dependency resolver: ipykernel 7.1.0 asks for jupyter-client>=8, and the lock pins
7.4.9. The step still exits 0, nothing in the pipeline imports ipykernel, and `release/env/verify.sh` passes. And
`conda run` holds back a command's output until the command ends, so a long pip step can print nothing for minutes;
`conda run --no-capture-output` streams it instead.

Do not set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` in the `hawor` env. It turns off torch's weights-only checkpoint
loading. The pipeline starts the HaWoR driver with the env's own python, never through `conda run`, and removes the
variable from the driver's environment (`fastdo.hands.hawor_batched.driver_env`).

Clone HaWoR now: the `hawor` source builds below patch and build its DROID-SLAM. Its code is the **malik-group
fork** at the commit DO AS I DO pins. `fetch_weights.py` (step 3) never writes under `third_party/`, so clone the fork
**outside `third_party/`**:

```bash
HAWOR=$PWD/../hawor                                        # anywhere outside third_party/
git clone https://github.com/malik-group/HaWoR.git "$HAWOR"
git -C "$HAWOR" checkout 2c3fa0c06e54a1b192c1ff692eaa89d526cd6317
git -C "$HAWOR" submodule update --init --recursive        # DROID-SLAM, lietorch, eigen
```

Then build the packages listed at the end of each `.pip.txt` under "build from source". They have no installable
wheel, or need `--no-build-isolation`, and the extensions were compiled for sm_90 only, so rebuild them for your GPU.
These are the commits of the locks (`run/setup/build_baseline_envs.sh`, the route the development host took, installs
nvdiffrast, the `hawor` pytorch3d, chumpy and GeoCalib unpinned, so it is not byte-identical to the locks; it also
sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in `hawor`, as the DO AS I DO recipe does for that baseline's own
`conda run` launches, which this page does not use):

```bash
# sam3d (GeoCalib is already pinned in sam3d.pip.txt, at 97b8968e)
conda run -n sam3d env TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST pip install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
conda run -n sam3d env TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST FORCE_CUDA=1 pip install --no-build-isolation \
  "pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
conda run -n sam3d pip install --no-deps -e third_party/do-as-i-do/reconstruction/modules/sam-3d-objects
conda run -n sam3d python -m pip install --no-deps \
  "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl#sha256=3d41b2fc55753faa7f45d6568ea73a96b96afb48b82994ab9b49bcbcb6c87588"

# hawor (HAWOR is the HaWoR checkout cloned above)
conda run -n hawor pip install --no-build-isolation \
  "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
conda run -n hawor env TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST pip install --no-build-isolation torch-scatter==2.1.2
conda run -n hawor env TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST FORCE_CUDA=1 pip install --no-build-isolation \
  "pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
sed -i 's/at::ScalarType _st = ::detail::scalar_type(the_type);/at::ScalarType _st = the_type.scalarType();/' \
  "$HAWOR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h"      # lietorch on torch >= 2
(cd "$HAWOR/thirdparty/DROID-SLAM" && conda run -n hawor env TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST \
  python setup.py install)                                                           # lietorch and droid_backends
```

**flash_attn (required on A100, H100 and H200).** SAM 3D picks its attention backend from the card's name: on an A100,
H100 or H200 it selects flash_attn, and without it the shape stage stops with `No module named 'flash_attn'`. On other
cards it uses PyTorch's `sdpa`, a path this release has not run. The `flash_attn` line of the `sam3d` block above
installs flash_attn 2.8.3, the version SAM 3D's own requirements pin, from the official prebuilt wheel for this env
(CPython 3.11, torch 2.8, CUDA 12, C++11 ABI; the `sam3d` torch is 2.8.0+cu128). It is the same file, by sha256, that
our runs loaded, and pip refuses it if the `#sha256=` fragment does not match. No compiler is needed. We did not build
flash_attn from source: its build compiles many CUDA kernels, which flash_attn's own notes say can take hours, and it
would give a different binary from the one our runs used. `--no-deps` keeps pip from changing anything else in the env
(flash_attn's dependencies, torch and einops, are already pinned there). The wheel needs the CUDA 12 driver of step 0.
`release/env/verify.sh` (below) fails when device 0 is an A100, H100 or H200 and flash_attn does not import.

The optional `hands23` venv (internet-video prompt stage only) is its own venv: target it with `--python`, and let uv
pick each package from whichever index has the pinned version (`--index-strategy unsafe-best-match`; uv's default
takes the first index that has the name, and the PyTorch index hosts only some versions of common packages):

```bash
H23=<a Hands23 directory>                                  # becomes FASTDO_HANDS23_ROOT
uv venv --python 3.12 "$H23/env_hands23"
uv pip install --python "$H23/env_hands23/bin/python" --index-strategy unsafe-best-match -r release/env/hands23.pip.txt
git clone https://github.com/facebookresearch/detectron2 "$H23/src/detectron2"
git -C "$H23/src/detectron2" checkout a2f4a8771ab77e8411c26b27f24f9489a28a2453
TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST uv pip install --python "$H23/env_hands23/bin/python" \
  --no-build-isolation -e "$H23/src/detectron2"
```

Check what this page built, and only that, after step 3:

```bash
FASTDO_CONDA_BASE=<your conda root> FASTDO_HAWOR_ROOT=$HAWOR FASTDO_MANO_DIR=... FASTDO_MANO_LEFT_DIR=... \
  bash release/env/verify.sh            # --no-gpu for a CPU-only look; --sha256 also hashes every weight file
```

It imports the key packages of each env, checks flash_attn in `sam3d` (`release/env/check_flash_attn.py`: FAIL on an
A100/H100/H200 without it, a NOTE on other cards; with `--no-gpu` a NOTE), and checks every weight file at the path and size that `release/models.json` gives
(`release/env/check_weights.py`; it resolves `HF_HOME` and `TORCH_HOME` and does not follow `HF_HUB_CACHE`, so if
that variable names a directory other than `$HF_HOME/hub`, the Hugging Face hub files the pipeline reads are not
the ones it checks), and the two torch.hub code directories against the pin file `fetch_weights.py`
wrote (step 3). `run/setup/verify_baseline_envs.sh` is the DO AS I DO baseline's own check (the
`sam3`, `tapnet` and `retargeting` envs and the baseline's weight layout); you do not need it here.

## 3. Weights

`release/MODELS.md` lists every model with its source, licence, and the sha256 of the file our runs used. **Several
are non-commercial.** Read the licence before you fetch.

Pass the HaWoR checkout of step 2 as `--hawor-root`. The fetch must include `cotracker3` before the GPU run of
step 4: `release/example/run_example.py` stops before it starts when CoTracker3's code directory at the pinned commit
or its checkpoint is missing, rather than let torch.hub download unverified code and weights at run time.

```bash
uv run --frozen python scripts/fetch_weights.py            # the table; nothing is downloaded
uv run --frozen hf auth login                              # after accepting SAM 3D's terms on Hugging Face
uv run --frozen python scripts/fetch_weights.py --yes \
  --only sam2,moge2,sam3d,moge1_sam3d,dinov2_sam3d,cotracker3,hawor,wilor_detector --hawor-root "$HAWOR"
```

The script verifies every file against the sha256 of `release/MODELS.md`, including HaWoR's `model_config.yaml` and
the `mano_mean_params.npz` that the HaWoR repository tracks. On a mismatch it exits 1 and replaces nothing. The two
download routes differ. A direct-URL download that mismatches is kept as `<name>.sha256-mismatch`, which nothing
reads. A Hugging Face download is verified where the hub client put it: a mismatched file stays in the Hugging Face
cache, is only reported, and is the path the pipeline reads. The same holds for a mismatched file already at its
destination. Do not run the pipeline until you have resolved a reported mismatch. `release/env/check_weights.py` flags a wrong size every time it runs, and a wrong
sha256 with `--sha256`.

With `--yes`, the exit status covers the requested non-MANO model keys, including files already present and files the
script checks without downloading. It says nothing about MANO, which the script never fetches, and nothing about a
model left out of `--only` (the command above leaves out `hands23`).
`release/env/verify.sh` checks the two MANO files at their paths and sizes, and the Hands23 weight only when
`FASTDO_HANDS23_ROOT` is set.
Four model loaders (SAM 2.1, SAM 3D Objects, MoGe v1 and MoGe-2) download at a commit and read at the same commit
(`src/fastdo/model_pins.py`). That covers SAM 3D's own pointmap model, MoGe v1, which the SAM 3D worker pins through
`--hf-revision`. The hand worker opens local HaWoR and WiLoR files, listed with their source revisions and SHA-256
values in `release/MODELS.md`. Nothing is downloaded at run time, as long as step 3 ran. torch.hub still asks GitHub whether
DINOv2's `main` branch exists (it downloads nothing; offline, it falls back to the cached directory).

**DINOv2 inside SAM 3D.** SAM 3D's generators call `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")`
with no commit. torch.hub then uses `$TORCH_HOME/hub/facebookresearch_dinov2_main` if that directory exists (default
`TORCH_HOME` is `~/.cache/torch`) and downloads the moving `main` branch only if it does not. `fetch_weights.py`
unpacks DINOv2 at commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` into that directory, the code our runs used (file
for file). The checkpoint goes to `hub/checkpoints/`. Loading it with `source='local'` instead would mean editing the
generator YAMLs inside the gated snapshot (which changes their sha256) or the SAM 3D code, so the hub directory is the
only lever. An existing `facebookresearch_dinov2_main` is never replaced. It holds whatever `main` was when it was
cached, so move it aside before the fetch if you want the pinned one. `fetch_weights.py` writes a pin file
(`.fastdo_pin.json`: commit and tree digest) into each code directory it unpacks. `check_weights.py` reports a
torch.hub code directory without one, such as a `main` that torch.hub cached or the manual route below, as
UNVERIFIED, and fails one whose files changed since the unpack. A successful weight check can still report
UNVERIFIED for a code directory, and `--only` skips both torch.hub directory checks, whichever models it names. Run
the checker without `--only` and inspect both torch.hub directory statuses when confirming the source revisions. To
pin it by hand (it then reads UNVERIFIED):

```bash
HUB=${TORCH_HOME:-$HOME/.cache/torch}/hub; COMMIT=7764ea0f912e53c92e82eb78a2a1631e92725fc8
test ! -e "$HUB/facebookresearch_dinov2_main" && mkdir -p "$HUB" && \
  curl -fL -o "$HUB/dinov2.zip" "https://github.com/facebookresearch/dinov2/archive/$COMMIT.zip" && \
  unzip -q "$HUB/dinov2.zip" -d "$HUB" && mv "$HUB/dinov2-$COMMIT" "$HUB/facebookresearch_dinov2_main" && rm "$HUB/dinov2.zip"
```

At that commit `hubconf.py` also imports the Cell-DINO modules when torch.hub loads it; their files state CC BY 4.0
and CC BY-NC (THIRD_PARTY_LICENSES.md, M14). SAM 3D calls only `dinov2_vitl14_reg`, which is not Cell-DINO code, and
loads no Cell-DINO weights.

**MANO** is never downloaded for you. Register at <https://mano.is.tue.mpg.de> and accept the licence (non-commercial,
no redistribution). Put `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` in `FASTDO_MANO_DIR` and `FASTDO_MANO_LEFT_DIR`, and
copy them to `$HAWOR/_DATA/data/mano/` and `$HAWOR/_DATA/data_left/mano_left/`, where the HaWoR driver reads them.
Then check that your copies are the ones we used:

```bash
FASTDO_MANO_DIR=... FASTDO_MANO_LEFT_DIR=... FASTDO_HAWOR_ROOT=$HAWOR \
  uv run --frozen python release/env/check_weights.py --sha256sum-list --only mano | sha256sum -c -
```

It checks both places, so it needs all three variables: without `FASTDO_HAWOR_ROOT` it prints
`set FASTDO_HAWOR_ROOT first`, and `sha256sum` then reports `no properly formatted checksum lines`.

The same form checks any model by key, for example `--only sam3d,hawor` (the files must have been fetched).

**What you must supply yourself.** Two inputs need an account. After you accept the SAM License and log in, the fetcher downloads the SAM 3D
snapshot (the authors did not test an authenticated download from this release). Obtain MANO directly from its
provider:

| input | how to get it | where it must be |
|---|---|---|
| SAM 3D Objects, `facebook/sam-3d-objects` at revision `2e73555018d2741ccd486e56c24fac41155a1dc6` (19 files) | accept the SAM License at <https://huggingface.co/facebook/sam-3d-objects> with your Hugging Face account, `uv run --frozen hf auth login`, then `fetch_weights.py --yes --only sam3d` | the Hugging Face cache: `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/` (default `HF_HOME` is `~/.cache/huggingface`); `release/models.json` lists the 19 files with size and sha256 |
| MANO, `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` | register at <https://mano.is.tue.mpg.de> and accept the licence | `$HAWOR/_DATA/data/mano/MANO_RIGHT.pkl` and `$HAWOR/_DATA/data_left/mano_left/MANO_LEFT.pkl` (the GPU example reads only these), and, for `release/env/verify.sh` and the MANO sha256 check above, also in `FASTDO_MANO_DIR` and `FASTDO_MANO_LEFT_DIR` |

`fetch_weights.py` carries on past a key it cannot fetch (a gated repository without access, a network error) and ends
with a summary of every requested key and, for each missing one, how to get it. It exits 1 while anything is missing.

The opt-in Hands23 weights (`--only hands23`) come from an unversioned URL on the Hands23 authors' lab server
(`release/MODELS.md`). From some networks that server stalls during the TLS handshake or fails certificate
verification, so the fetch can fail there. We do not mirror the file: the weights carry no licence text. A copy you
obtain from the Hands23 authors another way is checked the same way, by the sha256 in `release/MODELS.md`. The example
does not use Hands23.
Manual download: the upstream README (<https://github.com/EvaCheng-cty/hands23_detector>, "Pre-trained model weights") gives this same URL for `wget`. On 2026-11-21 a HEAD request to it failed certificate verification ("unable to get local issuer certificate"); if your download fails the same way, get the file from the Hands23 authors and check it against the sha256 in `release/MODELS.md`.

## 4. Run the example

The input is one 6.85 s head-mounted clip (137 frames, 2028x1520, 20 fps): episode 11 of the **Video to Data (V2D)
Challenge, Track 3 public split**, by NVIDIA, licensed CC BY 4.0
(<https://huggingface.co/datasets/nvidia/video_to_data_challenge>). We do not ship it.
`release/example/run_example.py` downloads it from the pinned dataset revision into `data/example/` (a `.part` file,
renamed only when its sha256 is the pinned one; a mismatch is kept as `.sha256-mismatch`). The window is the whole
clip, decoded at stride 2 (69 rows, about 10 fps, the pipeline's operating rate). The object is the beige cup,
prompted by a box on frame 0 (`release/example/example_input.json`). We checked the box by drawing it on frame 0 of
the pinned clip: it encloses the cup.

**The camera, and what we did about it.** The clip is the V2D head camera `ego_cam_a`: a 108-degree HFOV lens with a
14-term rational distortion model, on a device the dataset's calibration marks `rotation_deg` 180. Track 3 has no
pinhole camera, and this repository has no undistortion step in its V2D path (only the calibration transforms of
`fastdo.bench.v2d_orientation`, used for ground truth). So the frames go in as recorded:

- **not rotated:** the stored frames are upright (frame 0 viewed: floor, table and hands at the bottom);
- **not undistorted:** the intrinsics stage estimates a pinhole K per clip with MoGe-2, as it does for internet video,
  and the default config then multiplies MoGe-2's `fx` and `fy` by 1.104 (`geometry_stage.intrinsics_focal_scale`,
  a pooled lab-camera correction; `intrinsics.npz` records it as `focal_scale`),
  and the factory calibration is not used. For this example, the input specification
  (`release/example/example_input.json`, `camera.distortion_at_box_px`) reports that applying the factory
  undistortion would move pixels near the cup by 1-44 px and pixels at the left and right image edges by about
  205-259 px. The pipeline uses the original frames, so that lens distortion goes into the reconstruction. The
  release does not independently recompute these ranges.

The run is a demonstration of the pipeline on a public clip, not a calibrated measurement. Do not compare its pose
error with the paper's numbers.

**CPU smoke test, no GPU and no weights.** This checks argument parsing, config loading, the real stage registry
(built without loading a model), the batched engine run with placeholder workers, and the output schema:

```bash
CUDA_VISIBLE_DEVICES="" uv run --frozen pytest -q tests/release
```

**GPU run.** Set the four variables that the README's "Installation" section lists for the GPU example
(`FASTDO_CONDA_BASE`, `FASTDO_SAM3D_PYTHON`, `FASTDO_HAWOR_ROOT=$HAWOR`, and `FASTDO_GPU_MEM_GB=32`), with your MANO
files copied into `$HAWOR` (step 3). `FASTDO_GPU_MEM_GB=32` is required: the default 12 GiB cap is below the default
config's stage budgets and SAM 3D's 19 GiB. On an A100, H100 or H200 the `sam3d` env must also have flash_attn (step
2; `release/env/verify.sh` checks it); no variable is needed for that. CoTracker3's torch.hub code is loaded at the
pinned commit `82e02e80` unless you set `FASTDO_COTRACKER_HUB_REF`, and the report records the ref used. Before any
download or GPU work the script checks that CoTracker3 (`scripts/fetch_weights.py --yes --only cotracker3`), the 19
SAM 3D Objects files and the two MANO files are where step 3 puts them; if one is missing it exits, names it and says
how to get it. The two exports name the card the run uses ("Which card the report names", below). Then:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0   # the one card to use, numbered as /proc lists them
FASTDO_GPU_MEM_GB=32 uv run --frozen --extra dev --extra frontend \
  python release/example/run_example.py --out outputs/example --card dedicated
```

**Before you run it.** The script deletes and rebuilds `<--out>/engine/v2d_t3_ep011_cam_a/` and overwrites
`manifest.parquet`, `example_report.json` and `ATTRIBUTION.txt` in `--out`; `timing_ledger.jsonl` and
`batches.jsonl` are appended.
Use a fresh `--out` for every run you want to keep.

**Which card the report names.** The report's `host` block and the ledger row's `hardware` list candidate devices
derived from the host: the card models in `/proc/driver/nvidia`, in PCI bus order, that `CUDA_VISIBLE_DEVICES`
leaves listed. They are not confirmed by the CUDA runtime, and they are not proof of which device ran. With
`CUDA_VISIBLE_DEVICES` unset, every card in `/proc` is listed; inside a container that can include cards the
container cannot use (the Nov 15 run listed eight H100s and used one; the Nov 16 reference below used a one-card mask). To make the list name the one card
the run uses, set both variables before the run, with a numeric mask:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA numbers cards in PCI bus order, as /proc lists them
export CUDA_VISIBLE_DEVICES=0                # one numeric index in that order
```

Without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, CUDA numbers cards fastest-first, so on a host with mixed card models a
numeric mask can name a different card than the one CUDA uses. A GPU UUID mask (`GPU-...`) is matched by UUID.

The script uses the production engine and its default configuration, `run/conf/config.yaml` (batched execution, native
resolution). The report records it as `execution.execution: batched`; the ledger row's `execution` field takes only
`serial` or `pipelined`, and records the batched run as `pipelined`. That names the engine path (the batched
runner), not an overlap: with one window nothing runs beside it, and `seconds` is that cold window's stage sum. It runs the stages decode, mask, shape, intrinsics,
hands, geometry, tracks and solve. Before the run, untimed, it turns the box into the keyframe mask with SAM 2.1 and
frees that model. Pass `--card shared` if anything else used the GPU during the run, and `--set key=value` for any
config override. For your own video, see README "Run on your own video" (`--spec`). qc runs only with `--qc-rule
<AcceptanceRule json>`, and no rule fitted for new video ships (README "Known gaps"). Exit code 0 means every stage finished.

| Files generated under `--out` (the reference README lists the subset included in the release) | contents |
|---|---|
| `engine/v2d_t3_ep011_cam_a/` | the stage artifacts, `profile.json` and side files; note 0 |
| `engine/.gpu.lock` | an empty lock file that the GPU stages hold while they run; it stays after the run |
| `batches.jsonl` | one line per stage batch attempt: its clips, wall time and errors; appended on every run |
| `manifest.parquet` | the engine manifest: state and seconds per stage |
| `example_report.json` | input with sha256, camera handling, window, stride, stage states and seconds, artifacts, pins, host, commit, and the ledger row; note 1 |
| `timing_ledger.jsonl` | One stage-cost row per example run. `seconds` sums the recorded stage durations. See note 2 and `docs/timing_protocol.md` for the separate per-attempt format. |
| `ATTRIBUTION.txt` | the V2D credit, its CC BY 4.0 licence URI, "modified: reconstructed by this pipeline", and a pointer to the model licences that also bind the outputs; note 3 |

Notes on the table:

0. The stage artifacts `decode.npz`, `mask.npz`, `mesh.obj`, `intrinsics.npz`, `hands.npz`, `geometry.npz`,
   `tracks.npz` and `solve.npz` (object pose and scale per frame); `profile.json` (seconds per stage);
   `mask_prompt.npz` (the keyframe mask made from the box) and `source_domain.json` (the clip's source domain), both
   written before the stages run; and the side files of some stages (`shape.json`, `sdf.npz`, `samples.npz`,
   `hands_hawor.npz`, `solve.json`, `source_profile.json`). `docs/output.md` gives the keys, shapes, frames and units.
1. `pins`: the CoTracker3 hub ref, the Hugging Face revisions and the DINOv2 commit. `host`: the driver and the
   candidate devices of "Which card the report names" above (`none` when `CUDA_VISIBLE_DEVICES` is empty). `commit`:
   the git commit, or in a tar or zip extract without git metadata `release archive: <version> <date>` from the
   `VERSION` file, which holds the release version and its date and names no commit. `ledger_row`:
   `{"file", "line"}`, relative to `--out`.
2. The unit-row fields: `unit_id`, `method`, `corpus`, `assigned`, `run`, `analysed`, `delivered`,
   `exclusion_reason`, `seconds`, `frames`, `cluster`, `metrics`, `stage_seconds`, `clip_id`, `input_sha256`,
   `commit`, `created_utc`. The timing basis, `timing_basis`, as key: value:
   - `measurement`: `measured` (`modelled` under `--stub`);
   - `card`: the `--card` value;
   - `attribution`: `stage_sum` (the row's `seconds` is the sum of the stage seconds);
   - `cache`: `cold` (model loading included);
   - `execution`: `pipelined` (the batched runner; see above);
   - `cpu_priority`: `nice <n>`, the process's nice value;
   - `stages`, `stride`, `frames_decoded`: the stages requested, the stride and the count of planned sampled rows (whether each
     stage completed is `stage_states` in the report, not this list);
   - `hardware`: the candidate devices, as in the report;
   - `frames_basis`: `source` (`frames` counts the window's source frames);
   - `source_rate_carry`: `null`; `delivered_basis` and `note`: text.

3. Another input file (a `--stub --video` placeholder, or the video of a `--spec`) gets no V2D credit.

`delivered` is `null` when qc did not run: a window is delivered only when the qc stage accepts it. `frames` is the
window's 137 source frames, and `timing_basis.frames_decoded` is the 69 rows the stages saw.

`profile.json` has the same split: `n_frames` is the window's 137 source frames (the clip's 20 fps), and `fps` is
10.0, the decode rate at stride 2 (the 69 rows). Its `clips_per_gpu_hour` is 3600 divided by `total_seconds`, the
timed stage sum of this one cold window. That sum leaves out the runner's overhead and the untimed prompt, so it is
not the command's elapsed time. The field is not a throughput, not a GPU-hour measurement, and not comparable with the paper's figures. The shipped reference `profile.json` leaves the field out.

**What the timing covers.** The logged reconstruction time and the ledger row's `seconds` are the sum of the eight
stage seconds of one cold window, model loading included. Outside that sum: the SAM 2.1 box-to-mask prompt before
the stages (untimed, above), the clip download, interpreter and uv start-up, and report writing; in one independent
run they took about 11-12 s. This boundary differs from every row of the paper's Table 2
(`docs/paper_exhibits.md`, "Timing boundaries").

**Run-to-run variation.** The authors report six cold runs of this command on the same H100 model, with stage sums
of 60.3-65.8 s, shape times of 33.9-38.0 s and decode times of 1.7-4.9 s. The shipped reference run
(`release/reference/`, `profile.json`) accounts for 60.3 s, including 33.9 s for shape and 4.2 s for decode. The
release includes no six-run summary or record linking decode time to page-cache state.

SAM 2 prints `cannot import name '_C' from 'sam2'` and "Skipping the post-processing step" during the mask stage.
`uv.lock` pins SAM 2 to a Git revision (`2b90b9f5ceec907a1c18123530e92e794ad901a4`), which uv builds from source. Its
compiled `_C` extension is optional and needs a CUDA toolchain at build time. The environment of this release did not
build it, so SAM 2 skipped its optional hole filling of the mask scores in every run for this release, the reference
included. If the message appears, the run matches that environment and nothing needs to be done. A fresh build that
does compile `_C` applies the step, which the reference runs did not.

Every GPU run also logs `WARNING ... memory ledger: costs_gb names 'prompt', which this run does not gate (not
registered, or a subprocess class); ignored`. The shipped config gives a memory cost to the opt-in prompt stage
(`run/conf/config.yaml`, `memory_ledger`), and the example does not run that stage, so the entry is unused. It is
harmless.

The row is one cold window, with model loading included. Do not compare it with the paper's reconstruction times:
the paper measured them under other conditions, which this page does not repeat. For those times, read the paper's
Table 2 (Sec. 5.2) for reconstruction time and its Table 5 (Sec. 5.3) for stage sums, and `docs/paper_exhibits.md`,
which restates their timing boundaries.

### Reference output

One run of the GPU command above, on 2026-11-16, is shipped in `release/reference/v2d_t3_ep011_cam_a/`:
`solve.npz`, `hands.npz` (without the four MANO mesh arrays, two vertex and two face arrays, and without the MANO pose, shape and translation arrays), `intrinsics.npz`,
`mesh.obj`, `source_domain.json`, the stage side files `shape.json` and `solve.json`, `profile.json`,
`example_report.json`, the `timing_ledger.jsonl` row and `ATTRIBUTION.txt`. Its `README.md` says how it was made and what was left out. In short:

- source: a `git archive` extract of this release's code, made before the release was packaged and recorded in
  `example_report.json`. The example has since gained stricter checks of its input file, which the pinned input
  passes (reference `README.md`). A run from an earlier `git archive` extract reproduced the listed outputs and
  fields, timings excluded;
  no run of the current tree is claimed;
- host: one H100 80 GB HBM3, driver 595.91.07, card otherwise idle, `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=0`; main env from `uv.lock` with the `viser` extra; CoTracker3 and DINOv2 code at their
  pinned commits, fetched by `fetch_weights.py`; the `sam3d` env a fresh copy of the development host's with this
  page's two `sam3d` commands of step 2 run in it (flash_attn from the wheel, `sam3d_objects` from the extract);
  the host's `hawor` env, the other weights (sha256-checked) and MANO;
- every stage done; stage seconds (one cold window, model loading included): decode 4.2, mask 5.2, shape 33.9,
  intrinsics 4.1, hands 1.4, geometry 7.1, tracks 1.3, solve 3.0; 60.3 s in total;
- the files and fields that the reference `README.md` lists match the previous reference run of 2026-11-15, timing fields excluded; that run used the host's own `sam3d`
  interpreter and CoTracker3's cached `main` code (reference `README.md`);
- `solve.npz`: 69 rows (source frames 0 to 136, stride 2), scale 0.2470 m per mesh unit, 18 fine iterations;
- `mesh.obj`: 16,878 vertices, 33,752 faces, sha256 `9057116cd65407b8d579cbe71968bb5fd6215d0869cfdf445eb545b60eb78791`.

A run on other hardware, drivers or environments can differ. These numbers are what this one run gave, not a
tolerance to test against. The CPU smoke test is not a reference: its workers are placeholders, its artifacts are
empty files and its seconds are not timings. `scripts/view_clip.py` opens the reference output on CPU (README "Interactive
viewer"). `expected_projection.png` in that directory shows the reference reconstruction over the clip at decode
rows 0, 34 and 68 (README "Expected output"); `scripts/render_reference_png.py` draws the same picture for your run.

## 5. Paper numbers

This release ships no study-wide cost or evaluation ledger; those are withheld. The `timing_ledger.jsonl` row of step 4 uses the field names of the
authors' cost-ledger unit rows; the paper table generator of this release does not read it (the example writer
appends to it and the release smoke test parses its row). Most of the paper's records do not ship with this
release. From this release alone, `python3 reproduce/regenerate_tables.py --out <new or empty dir>` checks all 14
aggregate-only records that this release includes in `reproduce/records/`. It uses ten of them to write four tables of
the paper source's table generator, writes a fifth from settings in the generator, and compares the five outputs
with that generator's output at a pinned paper-source revision, ignoring comment lines. None of the five is
a table of the paper's main text, whose tables the paper source writes by hand (README "Evaluation"). No command
here prints a main-paper table or one of the paper's reconstruction times. `reproduce/README.md` says what each
check takes and what it covers, and `docs/paper_exhibits.md` maps every exhibit of the paper's main text to what
this release holds toward it.

## 6. Licences, in one paragraph

The code is Apache-2.0. With its default models, the pipeline is **for non-commercial research only**. CoTracker3,
HaWoR, the WiLoR detector, smplx, MANO and the Hands23 weights are non-commercial or carry no licence. The **shape
stage is non-commercial as run**: nvdiffrast 0.4.0 (NVIDIA Source Code License, non-commercial, clause 3.3) runs in
the SAM 3D worker through the release's `fast_postprocess.py` (utils3d's `RastContext`, for hole filling, on by default).
kaolin 0.18.0 is Apache-2.0 except its `non_commercial` package (NVIDIA `LICENSE.NSCL`), which every kaolin import
loads, and SAM 3D's FlexiCubes code imports kaolin on a static reading. SAM 3D's own model licence allows commercial
use as we read it. The release's files are Apache-2.0; they import GPL and AGPL packages at run time, not included here:
pymeshfix, igraph and plyfile in `sam3d`, and ultralytics in `hawor`. If you distribute a combined work, their
copyleft terms apply to it.
What the pipeline produces is bound by these model licences too. If you publish results on the example clip,
attribute the V2D dataset as CC BY 4.0 requires (`ATTRIBUTION.txt` holds the credit). The conda locks also bring
Anaconda's Terms of Service (step 2). See `release/MODELS.md`, `THIRD_PARTY_LICENSES.md` and `NOTICE`. This is not
legal advice.

## CPU checks without weights

(Moved here from the README on 2026-11-21.)

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
reference output in the viewer (README, "Try it on CPU, no weights"). The whole CPU suite is `uv run --frozen pytest -q -m "not slow and not timing"`
(about 30 min on a 4-thread cap; measured on the development host); GPU tests are faked or skip, as do tests whose
dataset, conda env or baseline checkout is absent.

## Notes

- **From an archive without git metadata.** Run `bash scripts/fetch_baseline.sh` from its root, as in a clone (step 1;
  `--check` verifies an existing copy). Every other command, here and in `README.md`, is the same for a clone and
  for an archive.
- **Expected output of the environment build.** The `sam3d` pip step prints one `ERROR:` line from pip's dependency
  resolver (ipykernel 7.1.0 asks for jupyter-client>=8, and the lock pins 7.4.9); the step still exits 0 and
  `release/env/verify.sh` passes, because nothing in the pipeline imports ipykernel. `conda run` holds back a step's
  output until the step ends, so a long pip step can print nothing for minutes.
- **decord.** Every `uv run` may print `Uninstalled 1 package` and `Installed 1 package`: uv reinstalls decord 0.6.0
  each time, because it reads that wheel's platform tags as not matching. This is harmless; the version stays 0.6.0.
- **alongray.** `uv sync` installs `packages/nullaudit` but not `packages/alongray`. To use alongray in the main
  environment, add it with `uv pip install ./packages/alongray` (a later `uv sync` removes it again). alongray 0.4.1
  installs three commands: `alongray` (score one prediction), `alongray-pool` (pool per-unit rows with
  cluster-bootstrap intervals) and `alongray-coverage` (a unit-flow and conventions report for two methods' outputs on
  one protocol; it scores nothing).

## Environment variables

`docs/installation.md`, "Host paths", lists the path variables. These are the others. None is required: flash_attn goes into the
`sam3d` env itself (step 2), not through `FASTDO_SAM3D_PYTHONPATH`.

| variable | default | effect |
|---|---|---|
| `FASTDO_CPU_THREADS` | 4 | torch intra-op threads of each pipeline process and test session (at least 1). It is also the default of `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` when those are unset; the HaWoR driver sets all three to it |
| `FASTDO_SAM3D_PYTHONPATH` | unset: no entries | directories, separated by `:`, put first on the SAM 3D worker's `PYTHONPATH` (the worker only), for packages installed outside the `sam3d` env. Step 2 installs flash_attn inside the env, so the example does not need it. A non-null config `sam3d_pythonpath` (`bench.shape.sam3d_pythonpath` in `run/conf/config.yaml`) replaces it |
| `FASTDO_HANDS23_PYTHON` | `$FASTDO_HANDS23_ROOT/env_hands23/bin/python` | the interpreter of the Hands23 prompt detector |
| `FASTDO_HANDS23_WEIGHTS` | `$FASTDO_HANDS23_ROOT/weights/model_hands23.pth` | the Hands23 checkpoint |
| `FASTDO_FRAME_TABLE_CACHE` | `<root>/temp/cache/frame_tables`, where `<root>` is `FASTDO_ROOT` or else the current directory | where the per-video ffprobe frame tables are cached on disk. A cache miss probes the whole file again |
| `FASTDO_HAWOR_TRACE` | unset: off | diagnosis only. A path: the batched HaWoR driver appends a JSON-lines trace of sha1 digests there |
| `FASTDO_HAWOR_TRACE_LEVEL` | 1 | with the trace on. 1: numerics flags and digests of the detections, backbone features, heads and MANO outputs. 2: also a digest of every layer call, which forces a device sync per layer |
| `FASTDO_HAWOR_NMS_TRUNCATE` | unset: off | diagnosis only. `<clip_id>\|<chunk>\|<first_empty>` empties the detector's NMS output for images `first_empty` onward of predict call `chunk` of that clip. Never set it for a real run |

## Commands on this page

Checked for this release in a fresh clone on a Linux x86_64
host with no GPU visible, no `FASTDO_*` variable set and a filled uv cache, except the GPU run of step 4 (note 2 below
the table). Every other command is marked with what it needs.

| step | command | status |
|---|---|---|
| 1 | `bash scripts/fetch_baseline.sh` | run in a fresh clone: exit 0, the baseline at `824591b8` with every submodule at its recorded commit. An earlier version of the script gave the same in a clean `git archive` extract |
| 2 | `bash scripts/setup_all.sh --dry-run` (the whole step in one script) | run: prints every command of this step and runs none; note 1 |
| 2 | `uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser` | run: exit 0 |
| 2 | `conda create ...`, `conda run -n $env pip install ...`, `conda env config vars set ...` | not run: needs conda and network access to conda.anaconda.org, repo.anaconda.com, PyPI and the PyTorch index |
| 2 | `git clone` / `checkout` / `submodule update` of HaWoR | not run: needs network access to GitHub |
| 2 | the source builds for `sam3d` and `hawor` | not run: need the conda envs above, network access to GitHub, a CUDA 12.8 toolkit and `TORCH_CUDA_ARCH_LIST` for your GPU; the `sam-3d-objects` install also needs the baseline submodule of step 1 |
| 2 | the `hands23` venv | not run: needs network access to PyPI, the PyTorch index and GitHub, and a CUDA toolkit |
| 2 | `bash release/env/verify.sh` | run with no variables: exits 1 with `set FASTDO_CONDA_BASE to your conda or miniforge root`; the full check needs the envs of step 2, the HaWoR checkout, your MANO files and the weights of step 3 |
| 3 | `scripts/fetch_weights.py` (the table) | run: exit 0, nothing downloaded |
| 3 | `hf auth login` | not run: needs a Hugging Face account that accepted the SAM 3D terms, and network access |
| 3 | `scripts/fetch_weights.py --yes --only ...` | not run: needs network access, the Hugging Face login and `--hawor-root` |
| 3 | the manual DINOv2 pin (`curl ... unzip ...`) | not run: needs network access to GitHub |
| 3 | `release/env/check_weights.py --sha256sum-list --only mano` | run with `FASTDO_MANO_DIR` unset: exits 1 with `set FASTDO_MANO_DIR first`. Listing needs `FASTDO_MANO_DIR`, `FASTDO_MANO_LEFT_DIR` and `FASTDO_HAWOR_ROOT`. Verifying the four registered copies also needs the files and `sha256sum -c -` (an independent run of the full pipeline with all four copies in place: 4x OK) |
| 4 | `pytest -q tests/release` | run: all pass. In a fresh clone after `fetch_baseline.sh`: 101 passed. Before it: 100 passed and 1 skipped (the check against SAM 3D's own card list, which needs the DO AS I DO code). The skip prints its reason |
| 4 | `FASTDO_GPU_MEM_GB=32 uv run ... release/example/run_example.py --out outputs/example --card dedicated` | run: exit 0, every stage done; note 2 |
| 4 | `scripts/view_clip.py release/reference/v2d_t3_ep011_cam_a` (README "Try it on CPU, no weights") | run: serves on 127.0.0.1 and answers HTTP 200 (`tests/test_view_clip.py` does the same on a synthetic clip) |
| 5 | `python3 reproduce/regenerate_tables.py --out <new or empty dir>` | run: exit 0; the 14 released records match their pins and the 5 tables match the paper generator's output at the pinned paper-source revision without their `%` comment lines |

Notes on the table:

1. `setup_all.sh --dry-run` also runs the preflight checks (repository root, DO AS I DO submodule, conda, no existing
   `sam3d` or `hawor` env unless `--replace-envs`) and exits 2
   after the plan when one fails (`tests/test_setup_all.py`). Without `--dry-run` it needs what the rows below need.
2. The GPU example was run in a clean extract of the release source on one H100 80 GB, with a fresh copy of the
   development host's `sam3d` env completed by step 2's two `sam3d` commands (flash_attn, `sam3d_objects`), the
   host's `hawor` env, weights and MANO, and CoTracker3 at its pinned commit; the clip was downloaded and its sha256
   matched ("Reference output"). It was run again from a clean extract of an earlier `git archive`, with the same
   outputs and fields, timings excluded. It needs a GPU with at least 24 GB, the envs of step 2, the weights of step 3, MANO and the
   variables above. With a variable missing it stops before any download and names it (`tests/test_hostpaths_unset.py`).
