#!/usr/bin/env bash
# Put the DO AS I DO baseline at its pinned commit into third_party/do-as-i-do.
#
#   bash scripts/fetch_baseline.sh [--dry-run] [--check]
#
# In a git clone of HandReel (github.com/BillyChern/HandReel) and in a copy without git metadata (an archive), the
# repository records no submodule commit for third_party/do-as-i-do, so the script clones the baseline there, checks
# out commit 824591b808c342b20079c3b4198a8c2bdf88c74e, fetches the baseline's own submodules at the commits that
# commit records (sam-3d-objects and HaWoR live there), then verifies the checkout. In a git checkout that does record
# the submodule commit, it runs `git submodule update --init --recursive` for that path instead, then verifies.
# --check only verifies an existing checkout. --dry-run prints the commands and runs none. FASTDO_BASELINE_URL
# replaces the clone URL (for a mirror); the commit is fixed. Needs git and network access to GitHub (and GitLab, for
# a submodule of HaWoR). Run from the repository root.
# Exit codes: 0 verified; 1 a check failed or a command failed; 2 wrong directory or a usage error.
set -uo pipefail

PIN=824591b808c342b20079c3b4198a8c2bdf88c74e
URL="${FASTDO_BASELINE_URL:-https://github.com/malik-group/do-as-i-do.git}"
DEST=third_party/do-as-i-do

DRY=0
CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

run() {                                   # print the command; run it unless --dry-run; stop on failure
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "$DRY" -eq 0 ]; then "$@" || { echo "failed: $*" >&2; exit 1; }; fi
}

fail() { echo "fetch_baseline: FAIL: $1" >&2; exit 1; }

verify() {
  local head status
  [ -e "$DEST/.git" ] || fail "$DEST is not a git checkout"
  head=$(git -C "$DEST" rev-parse HEAD 2>/dev/null) || fail "$DEST is not a git checkout"
  [ "$head" = "$PIN" ] || fail "$DEST is at $head, not the pinned $PIN"
  status=$(git -C "$DEST" submodule status --recursive) || fail "git submodule status failed in $DEST"
  # A line starting with '-' is a submodule that is not checked out, '+' one at another commit than recorded,
  # 'U' one with merge conflicts.
  if printf '%s\n' "$status" | grep -q '^[-+U]'; then
    printf '%s\n' "$status" | grep '^[-+U]' >&2
    fail "a submodule of $DEST is missing or not at the commit $PIN records"
  fi
  [ -d "$DEST/reconstruction/modules/sam-3d-objects" ] || fail "$DEST/reconstruction/modules/sam-3d-objects is missing"
  echo "fetch_baseline: ok: $DEST at $PIN, every submodule at its recorded commit"
}

{ [ -f pyproject.toml ] && [ -f scripts/setup_all.sh ]; } || { echo "run from the repository root" >&2; exit 2; }
command -v git >/dev/null || { echo "git not found" >&2; exit 2; }

# A submodule commit (gitlink) is recorded only when this directory is the top of a git checkout whose index holds
# third_party/do-as-i-do as a submodule entry (mode 160000). A plain clone of the public repository has none.
GITLINK=0
if [ -e .git ]; then
  top=$(git rev-parse --show-toplevel 2>/dev/null) || top=""
  if [ -n "$top" ] && [ "$(cd "$top" && pwd -P)" = "$(pwd -P)" ] &&
     git ls-files --stage -- "$DEST" 2>/dev/null | grep -q '^160000 '; then
    GITLINK=1
  fi
fi

if [ "$CHECK" -eq 1 ]; then
  verify
  exit 0
fi
if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST")" ]; then
  if [ -e "$DEST/.git" ]; then
    echo "$DEST is not empty; checking it"
    verify
    exit 0
  fi
  fail "$DEST is not empty and not a git checkout; move it aside first"
fi

if [ "$GITLINK" -eq 1 ]; then
  run git submodule update --init --recursive -- "$DEST"
  if [ "$DRY" -eq 1 ]; then
    echo "(dry run: nothing was fetched or verified)"
    exit 0
  fi
  verify
  exit 0
fi

run git clone --no-checkout "$URL" "$DEST"
run git -C "$DEST" checkout --detach "$PIN"
run git -C "$DEST" submodule update --init --recursive
if [ "$DRY" -eq 1 ]; then
  echo "(dry run: nothing was fetched or verified)"
  exit 0
fi
verify
