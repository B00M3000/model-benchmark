#!/usr/bin/env bash
# Clone and install the four packages this study needs, on a Jetson Orin.
#
#   ./scripts/setup_jetson.sh
#
# Every install uses --no-deps, deliberately. All four declare torch (and
# torch2trt declares tensorrt) as dependencies, and letting pip satisfy those
# replaces JetPack's builds with PyPI wheels compiled against a different CUDA
# -- which is what produces "The NVIDIA driver on your system is too old" and
# "operator torchvision::nms does not exist".
#
# If an install fails (JetPack's setuptools and packaging often disagree, which
# breaks editable installs), the clone still works: this script prints the
# repo_paths block to paste into config.yaml, and the app imports straight from
# the checkout with nothing installed.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
SRC_DIR="${SRC_DIR:-$REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }

if [[ "$($PYTHON -c 'import sys; print(sys.prefix != sys.base_prefix)')" != "True" ]]; then
  warn "Not using a virtualenv. Expected .venv/bin/python."
  warn "Create one with:  python3 -m venv .venv --system-site-packages"
fi

# name|git url|editable
REPOS=(
  "nanoowl|https://github.com/NVIDIA-AI-IOT/nanoowl|yes"
  "nanosam|https://github.com/NVIDIA-AI-IOT/nanosam|yes"
  "efficientvit|https://github.com/mit-han-lab/efficientvit|yes"
  "torch2trt|https://github.com/NVIDIA-AI-IOT/torch2trt|no"
)

NEEDS_REPO_PATHS=()

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url editable <<<"$entry"
  dest="$SRC_DIR/$name"

  log "$name"
  if [[ -d "$dest/.git" ]]; then
    ok "already cloned at $dest"
  else
    git clone --depth 1 "$url" "$dest" || {
      warn "clone failed — skipping $name"
      continue
    }
  fi

  if "$PYTHON" -c "import $name" >/dev/null 2>&1; then
    ok "already importable"
    continue
  fi

  if [[ "$editable" == "yes" ]]; then
    "$PYTHON" -m pip install -e "$dest" --no-deps
  else
    "$PYTHON" -m pip install "$dest" --no-deps
  fi

  if "$PYTHON" -c "import $name" >/dev/null 2>&1; then
    ok "installed"
  else
    warn "install did not take — will import from the clone instead"
    NEEDS_REPO_PATHS+=("$dest")
  fi
done

if (( ${#NEEDS_REPO_PATHS[@]} )); then
  cat <<EOF

Some packages are not installed. They work fine imported straight from the
clone -- add this to config.yaml and nothing else needs to change:

repo_paths:
EOF
  for path in "${NEEDS_REPO_PATHS[@]}"; do
    printf '  - %s\n' "$path"
  done
fi

log "Verifying"
"$PYTHON" scripts/doctor.py
