#!/usr/bin/env bash
# Verify each baseline env imports its key package on GPU and required weights exist.
# Usage: FASTDO_CONDA_BASE=<conda root> run/setup/verify_baseline_envs.sh
# Checks the DO AS I DO baseline's envs (sam3, sam3d, hawor, tapnet, retargeting) and its weight layout. Exit status
# 0 when every check passes. release/env/verify.sh is the check for what release/QUICKSTART.md builds.
case "${1:-}" in -h|--help) sed -n '2,5p' "$0"; exit 0 ;; esac
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECON="$ROOT/third_party/do-as-i-do/reconstruction"
CONDA_BASE="${FASTDO_CONDA_BASE:?set FASTDO_CONDA_BASE to your conda or miniforge root}"   # config paths.conda_base
source "$CONDA_BASE/etc/profile.d/conda.sh"
fail=0
check () { if "$@" >/dev/null 2>&1; then echo "OK   $*"; else echo "FAIL $*"; fail=1; fi; }
# check_gated: same as check, but on failure appends a human-readable reason so an
# expected, license-gated FAIL (not a build/tooling bug) doesn't read as a silent
# regression. Still counts toward the non-zero exit code.
check_gated () {
  local reason="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "OK   $*"; else echo "FAIL $* (gated: $reason)"; fail=1; fi
}
check conda run -n sam3 python -c "import sam3, torch; assert torch.cuda.is_available()"
check conda run -n sam3d python -c "import sam3d_objects, torch; assert torch.cuda.is_available()"
# The five checks below cover the sam3d functionality gaps closed by post_sam3d()
# (in build_baseline_envs.sh): pytorch3d (mesh transforms/rendering, used throughout
# sam3d_objects + generate_mesh_sam3d.py/track_object.py/run_project_mesh_combined.py),
# moge + utils3d (get_pointmap_dir.py's MoGe pointmap model), kaolin + nvdiffrast (DiT
# backbone mesh postprocessing). A grep of the pipeline confirmed each is actually imported,
# not just present in the upstream env export.
check conda run -n sam3d python -c "import pytorch3d"
check conda run -n sam3d python -c "import moge, moge.model.v1"
check conda run -n sam3d python -c "import utils3d"
check conda run -n sam3d python -c "import kaolin"
check conda run -n sam3d python -c "import nvdiffrast, nvdiffrast.torch"
check conda run -n hawor python -c "import torch, lietorch; assert torch.cuda.is_available()"
check conda run -n tapnet python -c "import tapnet, torch; assert torch.cuda.is_available()"
check conda run -n retargeting python -c "import mujoco_warp, mink, retargeting"
check conda run -n retargeting python -c "import torch; assert torch.cuda.is_available()"
check test -f "$RECON/weights/tapnet/bootstapir_checkpoint_v2.pt"
check test -f "$RECON/modules/HaWoR/weights/hawor/checkpoints/hawor.ckpt"
check test -f "$RECON/modules/HaWoR/weights/external/droid.pth"
check test -f "$RECON/modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl"
# facebook/sam-3d-objects is a gated Hugging Face repo. This check fails until the account behind HF_TOKEN has
# accepted its terms at https://huggingface.co/facebook/sam-3d-objects and
# `./setup/02_fetch_weights.sh --download` has been re-run from the sam3d/base env.
check_gated "user must accept the license on huggingface.co for facebook/sam-3d-objects" \
  test -f "$RECON/weights/sam3d_shared/hf/slat_generator.ckpt"
exit $fail
