#!/usr/bin/env bash
# Build the 4 reconstruction envs + the retargeting env for the DO AS I DO baseline.
# Usage: run/setup/build_baseline_envs.sh [all|sam3|sam3d|hawor|tapnet|retargeting]
#
# This is the from-source recipe. It is not byte-identical to the locks in release/env/; the supported route is
# scripts/setup_all.sh (release/QUICKSTART.md step 2). Needs FASTDO_CONDA_BASE (your conda or miniforge root).
#
# Deviations from the upstream recipe:
#  - sam3d: built from the exact cu128 pins (env/sam3d.yml) instead of
#    ./setup/01_create_envs.sh sam3d (which installs from environments/default.yml,
#    pinned to torchaudio==2.5.1+cu121, no longer resolvable from the PyTorch wheel
#    index). env/README.md itself recommends env/sam3d.yml "on CUDA 12.8 GPUs".
#  - sam3d + hawor exact-pin YAMLs (env/sam3d.yml, env/hawor.yml) are full
#    `conda env export`s and include locally built packages that are not on PyPI
#    (chumpy==0.71, lietorch==0.2, droid-backends==0.0.0, diff-gaussian-rasterization==0.0.0,
#    nvdiffrast==0.4.0, sam3d-objects==0.0.1), so `pip install -r` on the raw file fails.
#    These lines are filtered out before `conda env create`, then the packages are
#    installed the documented way: chumpy from its git pin (per env/README.md),
#    lietorch/droid-backends built from DROID-SLAM's setup.py (which the lietorch
#    dispatch.h patch below needs anyway), sam3d-objects via `pip install -e`.
#  - diff-gaussian-rasterization (mip-splatting) is not built. It backs an optional
#    GLB/texture-baking renderer path (env/README.md's troubleshooting note) that the
#    reconstruction pipeline's run path does not import.
#  - kaolin==0.18.0, nvdiffrast==0.4.0, moge==1.0.0, pytorch3d==0.7.9 and utils3d==0.0.2
#    (sam3d.yml) are filtered out of the yml (none installs with plain `pip install -r`)
#    and installed by `post_sam3d()` below. The reconstruction pipeline's Stage 2/2.5/3
#    code imports all five (pytorch3d directly in generate_mesh_sam3d.py / track_object.py /
#    run_project_mesh_combined.py and throughout sam3d_objects; moge + utils3d via
#    get_pointmap_dir.py's MoGe depth model; kaolin + nvdiffrast via the DiT backbone's
#    mesh post-processing). See `post_sam3d()` for the commands and where each comes from.
#    None of the five, nor the hawor set below, has a matching PyPI release. The torch,
#    torchvision and torchaudio "+cu128" entries install through the yml's own
#    --extra-index-url.
#  - torch-scatter==2.1.2 (hawor.yml) is filtered out of the batch pip install and built
#    separately with --no-build-isolation: its setup.py imports torch to detect the CUDA
#    build, which fails under pip's default build isolation (no wheel exists for
#    torch 2.9+cu128, so pip falls back to a source build).
#  - pytorch3d==0.7.8 (hawor.yml) is filtered out of the batch install for the same
#    reason, then built from source (`git+...@stable`, TORCH_CUDA_ARCH_LIST=9.0
#    FORCE_CUDA=1, --no-build-isolation). The verify check (`import torch, lietorch`)
#    does not need it, but HaWoR's mesh renderer (lib/vis/renderer*.py) imports it, so it
#    is built best-effort; a failure here does not fail the env build.
#  - concurrent `conda create`/`conda env create` processes that share one conda pkgs
#    cache can fail with InvalidArchiveError (two builds extracting the same cached
#    package at once). Each target gets its own CONDA_PKGS_DIRS, so builds of different
#    envs can run at the same time.
#  - build_retargeting's `conda create -n retargeting` had no existence guard (unlike
#    build_sam3d/build_hawor's `create || update` fallback), so re-running just the
#    `retargeting` target after a partial success aborted under `set -e` before reaching
#    the torch cu128 pin / `pip install -e .`. Guarded with a `conda env list` check.
case "${1:-}" in
  -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
  ""|all|sam3|sam3d|hawor|tapnet|retargeting) ;;
  *) echo "usage: $0 [all|sam3|sam3d|hawor|tapnet|retargeting]" >&2; exit 1 ;;
esac
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="$ROOT/third_party/do-as-i-do"
RECON="$BASE/reconstruction"
CONDA_BASE="${FASTDO_CONDA_BASE:?set FASTDO_CONDA_BASE to your conda or miniforge root}"   # config paths.conda_base
source "$CONDA_BASE/etc/profile.d/conda.sh"
export PATH="$CONDA_BASE/bin:$PATH"
target="${1:-all}"

# Isolate the package cache per target so concurrent builds (different envs) never
# race on the same cached .conda file (see deviations note above).
export CONDA_PKGS_DIRS="$CONDA_BASE/pkgs_build_${target}"
mkdir -p "$CONDA_PKGS_DIRS"
# build_sam3d / build_hawor write their filtered yml via mktemp "$ROOT/temp/...": ensure
# the directory exists before the first mktemp call (fresh checkout may not have it yet).
mkdir -p "$ROOT/temp"

build_recon () {   # $1 = sam3|tapnet|all
  cd "$RECON"
  ./setup/01_create_envs.sh "$1"
}

build_sam3d () {
  # Exact cu128 pins (torch 2.8+cu128); environments/default.yml pins cu121 torchaudio
  # builds that are no longer available from the PyTorch wheel index.
  cd "$RECON"
  # conda's env-spec plugin picks its parser off the filename extension, so a real
  # .yml file is needed (not a process-substitution /dev/fd path).
  filtered_yml="$(mktemp "$ROOT/temp/sam3d_filtered.XXXXXX.yml")"
  grep -vE '^\s*-\s*(diff-gaussian-rasterization==0\.0\.0|nvdiffrast==0\.4\.0|sam3d-objects==0\.0\.1|kaolin==0\.18\.0|moge==1\.0\.0|pytorch3d==0\.7\.9|utils3d==0\.0\.2)\s*$' env/sam3d.yml > "$filtered_yml"
  conda env create -n sam3d -f "$filtered_yml" || conda env update -n sam3d -f "$filtered_yml"
  rm -f "$filtered_yml"
  conda activate sam3d
  # --no-deps: env/sam3d.yml's own pip section already installed sam3d_objects' full
  # transitive dependency closure (it's a flattened `conda env export`); a plain
  # `pip install -e` re-resolves against the module's pyproject.toml/requirements.txt,
  # which still pin torchaudio==2.5.1+cu121 (same unavailable build as default.yml).
  pip install -e "$RECON/modules/sam-3d-objects" --no-deps                      # exposes sam3d_objects (locally built, not on PyPI)
  pip install "geocalib @ git+https://github.com/cvg/GeoCalib.git"              # Stage 2 gravity (scripts/predict_video_gravity.py); env/sam3d.yml predates this dep
  # sam3d_objects/__init__.py does `import sam3d_objects.init` unless LIDRA_SKIP_INIT is
  # set, but sam3d_objects/init.py does not exist in this fork/commit (verified: `import
  # sam3d_objects` bare raises ModuleNotFoundError). The package itself documents this
  # escape hatch ("Allow skipping initialization for lightweight tools"); set it for the
  # env so plain `import sam3d_objects` (the verify script's check) works.
  conda env config vars set LIDRA_SKIP_INIT=1 -n sam3d
  conda deactivate
}

post_sam3d () {
  # Close the sam3d functionality gaps left by build_sam3d's filtered yml (deviation 2
  # above): moge, utils3d, kaolin, nvdiffrast, pytorch3d ARE imported by the reconstruction
  # pipeline's Stage-2/2.5/3 code paths (confirmed by grepping reconstruction/scripts/*.py,
  # modules/sam-3d-objects/generate_mesh_sam3d.py, modules/Fast-SAM3D/track_object.py, and
  # sam3d_objects/**/*.py for `import moge|utils3d|kaolin|nvdiffrast|pytorch3d`):
  #   - pytorch3d: generate_mesh_sam3d.py, track_object.py, run_project_mesh_combined.py,
  #     and most of sam3d_objects/{pipeline,data,utils/visualization}/*.py (mesh transforms,
  #     rendering, layout post-optimization).
  #   - moge (+ its own utils3d dependency): get_pointmap_dir.py's hydra-instantiated
  #     depth_model (checkpoints/hf/pipeline.yaml -> sam3d_objects.pipeline.depth_models.moge.MoGe
  #     -> moge.model.v1.MoGeModel.from_pretrained("Ruicheng/moge-vitl"), a public/non-gated
  #     HF repo, unlike the gated facebook/sam-3d-objects checkpoints).
  #   - utils3d (standalone): sam3d_objects/pipeline/utils/pointmap.py,
  #     sam3d_objects/pipeline/layout_post_optimization_utils.py,
  #     sam3d_objects/utils/visualization/image_mesh.py,
  #     sam3d_objects/model/backbone/tdfy_dit/utils/{postprocessing,render}_utils.py.
  #   - kaolin: sam3d_objects/model/backbone/tdfy_dit/representations/mesh/flexicubes/flexicubes.py
  #     (DiT backbone mesh postprocessing, part of generate_mesh_sam3d.py's Stage-2 path).
  #   - nvdiffrast: lazily imported inside postprocessing_utils.py (`import nvdiffrast.torch as dr`),
  #     same DiT backbone mesh-postprocessing path.
  # Versions/commits below match the exact pins filtered out of env/sam3d.yml so the env
  # ends up at the same versions the original `conda env export` recorded (0.0.2 / 1.0.0 /
  # 0.18.0 / 0.4.0 / 0.7.9-equivalent — pytorch3d has no 0.7.9 tag, so this uses the fork's
  # own requirements.p3d.txt pin instead, same as build_hawor's pytorch3d step).
  cd "$RECON"
  conda activate sam3d
  # kaolin: no PyPI wheel for our torch/cuda combo, but NVIDIA publishes prebuilt wheels
  # per (torch, cuda, python) combo at this index; torch-2.8.0_cu128/kaolin-0.18.0-cp311-...
  # exists and matches this env exactly (torch 2.8.0+cu128, Python 3.11) — no source build
  # needed, unlike pytorch3d/nvdiffrast below.
  pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
  # utils3d: pinned to the exact commit modules/Fast-SAM3D/doc/Setup.md documents (MoGe's
  # own dependency pin); NOT editable per that doc's explicit warning.
  pip install "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
  # moge: pinned to the commit modules/sam-3d-objects/requirements.txt names
  # (MoGe @ git+...@a8c37341...). --no-deps: utils3d is already installed above at the
  # pin MoGe expects; letting pip re-resolve would just re-fetch the same utils3d commit
  # via a slower path (MoGe's own pyproject re-pins the identical git+ URL).
  pip install --no-deps "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
  # nvdiffrast: pure-Python package that JIT-compiles its CUDA extension on first use, but
  # even `pip install` (metadata-only, no CUDA compile yet) refuses to proceed under pip's
  # default build isolation ("Cannot compile nvdiffrast CUDA extension... run with
  # --no-build-isolation") because its setup.py probes for an already-importable torch.
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
  # pytorch3d: no PyPI wheel for our torch/cuda combo; source build. Uses the exact commit
  # modules/sam-3d-objects/requirements.p3d.txt pins (0.7.8-equivalent tree, tagged 0.7.9 in
  # env/sam3d.yml's export but no such PyPI/git tag exists) rather than build_hawor's
  # `@stable`, so the sam3d env matches what this fork's own extras file specifies.
  # fvcore is a build-time dependency of pytorch3d's setup.py (iopath was already present
  # from env/sam3d.yml's own pip section).
  pip install fvcore
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" FORCE_CUDA=1 pip install --no-build-isolation "pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
  conda deactivate
}

build_hawor () {
  # Use the exact cu128 pins (torch 2.9+cu128) instead of the cu117 recipe; H100 needs sm_90.
  # env/hawor.yml pins chumpy==0.71 / lietorch==0.2 / droid-backends==0.0.0, none of which
  # are installable from PyPI as plain wheels (see deviations note above) — filtered out here.
  cd "$RECON"
  filtered_yml="$(mktemp "$ROOT/temp/hawor_filtered.XXXXXX.yml")"
  grep -vE '^\s*-\s*(chumpy==0\.71|lietorch==0\.2|droid-backends==0\.0\.0|pytorch3d==0\.7\.8|torch-scatter==2\.1\.2)\s*$' env/hawor.yml > "$filtered_yml"
  conda env create -n hawor -f "$filtered_yml" || conda env update -n hawor -f "$filtered_yml"
  rm -f "$filtered_yml"
  conda activate hawor
  # chumpy's git pin resolves to 0.71 (PyPI maxes at 0.70); needs --no-build-isolation
  # (its setup.py imports numpy). Per reconstruction/env/README.md / setup/01_create_envs.sh.
  pip install "chumpy@git+https://github.com/mattloper/chumpy" --no-build-isolation
  # torch-scatter's setup.py needs torch (already installed above) visible at build time.
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" pip install torch-scatter==2.1.2 --no-build-isolation
  # pytorch3d: no prebuilt wheel for our torch/cuda combo; build from source. Only used by
  # HaWoR's mesh renderer (lib/vis/renderer*.py), not by the verify check (`import torch,
  # lietorch`) — best-effort, does not fail the env build.
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" FORCE_CUDA=1 pip install "pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation \
    || echo "[build_baseline_envs] WARNING: pytorch3d source build failed (non-fatal; not required by verify_baseline_envs.sh)" >&2
  # lietorch dispatch.h patch for torch>=2.x (see reconstruction/env/README.md)
  f="$RECON/modules/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h"
  sed -i 's/at::ScalarType _st = ::detail::scalar_type(the_type);/at::ScalarType _st = the_type.scalarType();/' "$f"
  ( cd "$RECON/modules/HaWoR/thirdparty/DROID-SLAM" && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" python setup.py install )
  conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -n hawor
  conda deactivate
}

build_retargeting () {
  cd "$BASE/retargeting"
  # Guard creation so a re-run (e.g. retrying just this target after a partial `all`) does
  # not abort under `set -e`: a bare `conda create -n retargeting` fails outright if the env
  # already exists, unlike build_sam3d/build_hawor's `create || update` fallback.
  conda env list | awk '{print $1}' | grep -qx retargeting || conda create -y -n retargeting python=3.12
  conda activate retargeting
  # pyproject.toml pins "torch" with no version/index: a bare `pip install -e .` resolves
  # the newest torch (2.14+cu130) from PyPI, which needs a newer driver than a CUDA 12.8
  # one (570 series): torch imports but torch.cuda.is_available() is False.
  # Install the cu128 build first so the editable install's dependency resolution is a
  # no-op for torch (already-satisfied pin).
  pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
  pip install -e .
  conda deactivate
}

case "$target" in
  all) build_recon sam3; build_sam3d; post_sam3d; build_recon tapnet; build_hawor; build_retargeting ;;
  sam3|tapnet) build_recon "$target" ;;
  sam3d) build_sam3d; post_sam3d ;;
  hawor) build_hawor ;;
  retargeting) build_retargeting ;;
  *) echo "usage: $0 [all|sam3|sam3d|hawor|tapnet|retargeting]"; exit 1 ;;
esac

# Post-steps from reconstruction/env/README.md
conda activate sam3d && pip uninstall -y notebook || true; conda deactivate || true
echo "[build_baseline_envs] done: $target"
