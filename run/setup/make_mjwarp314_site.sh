#!/usr/bin/env bash
# Build the simulator-stack SITE directory for `_baseline_physics_driver.py --sim-stack mjwarp314`.
#
# Usage: run/setup/make_mjwarp314_site.sh <venv-site-packages> <dest-dir>
#
# <venv-site-packages> is a Python 3.12 venv holding mujoco==3.14.0, mujoco-warp==3.14.0 and warp-lang==1.17.0
# (one can be built with: UV_CACHE_DIR=<cache> uv venv -p python3.12 <venv>;
#  uv pip install -p <venv>/bin/python mujoco==3.14.0 mujoco-warp==3.14.0 warp-lang==1.17.0).
# <dest-dir> receives symlinks to ONLY those three packages and their dist-info, so the `retargeting` env's torch,
# numpy and DO AS I DO dependencies are not shadowed. Point FASTDO_MJWARP314_SITE at <dest-dir>.
# Nothing in the `retargeting` env or the venv is modified.
case "${1:-}" in -h|--help) sed -n '2,11p' "$0"; exit 0 ;; esac
set -euo pipefail
src=${1:?venv site-packages}
dest=${2:?destination directory}
mkdir -p "$dest"
for pat in mujoco mujoco-3.14.0.dist-info mujoco_warp mujoco_warp-3.14.0.dist-info warp warp_lang-1.17.0.dist-info; do
    if [ ! -e "$src/$pat" ]; then
        echo "missing $src/$pat" >&2
        exit 1
    fi
    ln -sfn "$src/$pat" "$dest/$pat"
done
echo "site ready: $dest"
