#!/usr/bin/env bash
# Build the environments of release/QUICKSTART.md step 2: the main uv env, the sam3d and hawor conda envs with their
# source builds, the HaWoR checkout, and (with --hands23) the optional Hands23 venv.
#
#   bash scripts/setup_all.sh [--dry-run] [--hawor DIR] [--hands23 DIR] [--replace-envs]
#
# --dry-run prints every command and runs none. It also runs the preflight checks (repository root, DO AS I DO
# baseline, conda), reports each result, and exits 2 after the plan when one fails. Needs uv, conda
# (FASTDO_CONDA_BASE, or conda on PATH), git, network access, a CUDA 12.8 toolkit for the source builds, and the
# DO AS I DO baseline in third_party/do-as-i-do (run bash scripts/fetch_baseline.sh first). TORCH_CUDA_ARCH_LIST
# defaults to 9.0 (H100); set it for your GPU. Run from the repository root. It refuses to start when a conda env
# named sam3d or hawor already exists, because `conda create -y` would replace it; --replace-envs allows that. A cold
# build needs about 60 GB of disk, package caches included.
set -euo pipefail

DRY=0
HAWOR="$(pwd)/../hawor"
H23=""
REPLACE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --hawor) HAWOR="$2"; shift ;;
    --hands23) H23="$2"; shift ;;
    --replace-envs) REPLACE=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
HAWOR_COMMIT=2c3fa0c06e54a1b192c1ff692eaa89d526cd6317
NVDIFFRAST="git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
PYTORCH3D="pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
CHUMPY="chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
DETECTRON2_COMMIT=a2f4a8771ab77e8411c26b27f24f9489a28a2453
# The official flash_attn 2.8.3 wheel (cp311, torch 2.8, CUDA 12, C++11 ABI) for the sam3d env; pip checks the sha256
# fragment. SAM 3D selects flash_attn on A100/H100/H200. Same line as release/env/sam3d.pip.txt.
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl#sha256=3d41b2fc55753faa7f45d6568ea73a96b96afb48b82994ab9b49bcbcb6c87588"

run() {                                   # print the command; run it unless --dry-run
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "$DRY" -eq 0 ]; then "$@"; fi
}

# Preflight. A real run stops at the first failed check; --dry-run reports every result, prints the plan anyway,
# and then exits 2 if a check failed.
PREFLIGHT_FAILED=0
preflight_fail() {
  echo "$1" >&2
  if [ "$DRY" -eq 0 ]; then exit 2; fi
  echo "preflight: FAIL: $1"
  PREFLIGHT_FAILED=1
}
{ [ -f pyproject.toml ] && [ -f uv.lock ]; } || preflight_fail "run from the repository root"
[ -d third_party/do-as-i-do/reconstruction/modules/sam-3d-objects ] ||
  preflight_fail "the DO AS I DO baseline is missing from third_party/do-as-i-do: run bash scripts/fetch_baseline.sh first"
if [ -n "${FASTDO_CONDA_BASE:-}" ]; then export PATH="$FASTDO_CONDA_BASE/bin:$PATH"; fi
command -v conda >/dev/null || preflight_fail "conda not found: set FASTDO_CONDA_BASE or put conda on PATH"
conda_env_names() {                       # the names conda lists, one per line; fails when conda cannot list them
  local listing
  listing="$(conda env list 2>/dev/null)" || return 1
  awk '!/^#/ && NF {print $1}' <<<"$listing"
}
if command -v conda >/dev/null && [ "$REPLACE" -eq 0 ]; then
  if ! existing="$(conda_env_names)"; then  # a failed listing is not an empty list
    preflight_fail "conda env list failed, so this script cannot tell whether a conda env named sam3d or hawor exists; fix conda, or pass --replace-envs to let conda create replace one"
  else
    for env in sam3d hawor; do              # conda create -y would silently replace an existing env
      if grep -qx "$env" <<<"$existing"; then
        preflight_fail "a conda env named $env already exists; remove it, or pass --replace-envs to let conda create replace it"
      fi
    done
  fi
fi
if [ "$DRY" -eq 1 ] && [ "$PREFLIGHT_FAILED" -eq 0 ]; then
  if [ "$REPLACE" -eq 1 ]; then
    echo "preflight: ok (repository root, DO AS I DO baseline, conda; --replace-envs: existing sam3d and hawor envs are replaced)"
  else
    echo "preflight: ok (repository root, DO AS I DO baseline, conda, no existing sam3d or hawor env)"
  fi
fi

# main env
run uv sync --frozen --python 3.12 --extra dev --extra frontend --extra viser

# sam3d and hawor conda envs
for env in sam3d hawor; do
  if [ "$REPLACE" -eq 0 ] && [ "$DRY" -eq 0 ]; then   # checked again right before each create
    names="$(conda_env_names)" || { echo "conda env list failed before creating $env; nothing was created for it" >&2; exit 2; }
    if grep -qx "$env" <<<"$names"; then
      echo "a conda env named $env exists now; remove it, or pass --replace-envs" >&2
      exit 2
    fi
  fi
  run conda create -y -n "$env" --file "release/env/$env.conda.txt"
  run conda run -n "$env" pip install -r "release/env/$env.pip.txt"
done
run conda env config vars set LIDRA_SKIP_INIT=1 -n sam3d

# HaWoR checkout (outside third_party/)
if [ ! -d "$HAWOR/.git" ]; then run git clone https://github.com/malik-group/HaWoR.git "$HAWOR"; fi
run git -C "$HAWOR" checkout "$HAWOR_COMMIT"
run git -C "$HAWOR" submodule update --init --recursive

# source builds: sam3d
run conda run -n sam3d env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" pip install --no-build-isolation "$NVDIFFRAST"
run conda run -n sam3d env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" FORCE_CUDA=1 pip install --no-build-isolation \
  "$PYTORCH3D"
run conda run -n sam3d pip install --no-deps -e third_party/do-as-i-do/reconstruction/modules/sam-3d-objects
run conda run -n sam3d python -m pip install --no-deps "flash_attn @ $FLASH_ATTN_WHEEL"

# source builds: hawor
run conda run -n hawor pip install --no-build-isolation "$CHUMPY"
run conda run -n hawor env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" pip install --no-build-isolation \
  torch-scatter==2.1.2
run conda run -n hawor env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" FORCE_CUDA=1 pip install --no-build-isolation \
  "$PYTORCH3D"
run sed -i 's/at::ScalarType _st = ::detail::scalar_type(the_type);/at::ScalarType _st = the_type.scalarType();/' \
  "$HAWOR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h"
run conda run -n hawor --cwd "$HAWOR/thirdparty/DROID-SLAM" env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  python setup.py install

# optional: the Hands23 venv (internet-video prompt stage)
if [ -n "$H23" ]; then
  run uv venv --python 3.12 "$H23/env_hands23"
  run uv pip install --python "$H23/env_hands23/bin/python" --index-strategy unsafe-best-match \
    -r release/env/hands23.pip.txt
  if [ ! -d "$H23/src/detectron2/.git" ]; then
    run git clone https://github.com/facebookresearch/detectron2 "$H23/src/detectron2"
  fi
  run git -C "$H23/src/detectron2" checkout "$DETECTRON2_COMMIT"
  run env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" uv pip install --python "$H23/env_hands23/bin/python" \
    --no-build-isolation -e "$H23/src/detectron2"
fi

if [ "$DRY" -eq 1 ] && [ "$PREFLIGHT_FAILED" -eq 1 ]; then
  echo "dry run: a preflight check failed (see above); the real run would stop before its first step" >&2
  exit 2
fi
echo "done. Next: weights (release/QUICKSTART.md step 3), then bash release/env/verify.sh."
