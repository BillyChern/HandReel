#!/usr/bin/env bash
# Check what release/QUICKSTART.md builds, and nothing else: the main uv env, the sam3d and hawor conda envs (and the
# hands23 venv when FASTDO_HANDS23_ROOT is set), flash_attn in sam3d on an A100/H100/H200
# (release/env/check_flash_attn.py), the weights where the pipeline reads them, and MANO.
# Usage: FASTDO_CONDA_BASE=<conda root> FASTDO_HAWOR_ROOT=... FASTDO_MANO_DIR=... FASTDO_MANO_LEFT_DIR=... \
#     bash release/env/verify.sh [--no-gpu] [--sha256]
# --no-gpu skips the torch.cuda.is_available() checks (for a CPU-only look at the envs); --sha256 also hashes every
# weight file (slow). Exit status 0 when every check passes. run/setup/verify_baseline_envs.sh is the DO AS I DO
# baseline's check (sam3, tapnet, retargeting and the baseline's weight layout); QUICKSTART users do not need it.
for arg in "$@"; do
  case "$arg" in -h|--help) sed -n '2,9p' "$0"; exit 0 ;; esac
done
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${FASTDO_CONDA_BASE:?set FASTDO_CONDA_BASE to your conda or miniforge root}"
gpu=1; sha=()
for arg in "$@"; do
  case "$arg" in
    --no-gpu) gpu=0 ;;
    --sha256) sha=(--sha256) ;;
    *) echo "usage: $0 [--no-gpu] [--sha256]" >&2; exit 2 ;;
  esac
done
fail=0
check () { if "$@" >/dev/null 2>&1; then echo "OK   $*"; else echo "FAIL $*"; fail=1; fi; }
cuda () { if [ "$gpu" = 1 ]; then echo "; assert torch.cuda.is_available()"; fi; }
in_env () { local env="$1"; shift; "$CONDA_BASE/bin/conda" run -n "$env" python -c "$*"; }

cd "$ROOT" || exit 1
check uv run --frozen python -c "import fastdo, torch, sam2, moge.model.v2$(cuda)"
check in_env sam3d "import os; os.environ['LIDRA_SKIP_INIT'] = '1'; import sam3d_objects, torch$(cuda)"
check in_env sam3d "import pytorch3d, moge.model.v1, utils3d, kaolin, nvdiffrast.torch, pymeshfix, igraph, plyfile"
# flash_attn: SAM 3D selects it on A100/H100/H200 by card name; the check prints its own OK/NOTE/FAIL line.
flash=(release/env/check_flash_attn.py)
if [ "$gpu" = 0 ]; then flash+=(--no-gpu); fi
"$CONDA_BASE/bin/conda" run -n sam3d python "${flash[@]}" || fail=1
check in_env hawor "import torch, lietorch, droid_backends, torch_scatter, smplx, chumpy, ultralytics$(cuda)"
if [ -n "${FASTDO_HANDS23_ROOT:-}" ]; then
  check "$FASTDO_HANDS23_ROOT/env_hands23/bin/python" -c "import detectron2, torch$(cuda)"
fi
check test -n "${FASTDO_HAWOR_ROOT:-}"
check test -f "${FASTDO_HAWOR_ROOT:-/nonexistent}/lib/models/hawor.py"
weights=(uv run --frozen python release/env/check_weights.py "${sha[@]}")
if [ -n "${FASTDO_HANDS23_ROOT:-}" ]; then weights+=(--with-hands23); fi
"${weights[@]}" || fail=1
exit $fail
