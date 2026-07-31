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
# The package list, editable flag and build-isolation setting all come from
# benchmark.models.registry.REQUIRED_MODULES rather than being duplicated
# here, so there is exactly one place that knows torch2trt needs
# --no-build-isolation (its setup.py imports tensorrt/torch to compile a CUDA
# extension, and pip's isolated build env doesn't inherit
# --system-site-packages, so it can't see either).
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

# NOT a plain `import $name` check. Every clone here is a directory that
# shares its name with the package nested one level inside it
# (nanosam/nanosam/__init__.py), so with the repo root on sys.path -- which
# it always is here -- `import nanosam` "succeeds" as an empty PEP 420
# namespace package even when nothing is actually installed. That false
# positive previously made this script skip real installs entirely.
# module_status() tells a genuine install apart from that shadow by checking
# spec.origin (None for a namespace package).
is_really_importable() {
  "$PYTHON" - "$1" <<PYEOF
import sys
sys.path.insert(0, "$REPO_ROOT")
from benchmark.models.registry import module_status
status, _ = module_status(sys.argv[1])
sys.exit(0 if status == "ok" else 1)
PYEOF
}

MODULES_TSV="$("$PYTHON" - <<PYEOF
import sys
sys.path.insert(0, "$REPO_ROOT")
from benchmark.models.registry import REQUIRED_MODULES
for m in REQUIRED_MODULES:
    print(f"{m.module}\t{m.repo}\t{int(m.editable)}\t{int(m.build_isolation)}")
PYEOF
)"

NEEDS_REPO_PATHS=()

while IFS=$'\t' read -r name url editable build_iso; do
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

  if is_really_importable "$name"; then
    ok "already importable"
    continue
  fi

  args=(-m pip install)
  [[ "$editable" == "1" ]] && args+=(-e)
  args+=("$dest" --no-deps)
  [[ "$build_iso" == "0" ]] && args+=(--no-build-isolation)
  "$PYTHON" "${args[@]}"

  if is_really_importable "$name"; then
    ok "installed"
  else
    warn "install did not take — will import from the clone instead"
    NEEDS_REPO_PATHS+=("$dest")
  fi
done <<<"$MODULES_TSV"

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

# transformers is not one of the four repos above -- it's a separate PyPI
# package NanoOWL imports at runtime (owl_predictor.py, for
# OwlViTForObjectDetection). NanoOWL's own setup.py declares no dependencies
# (it's installed with --no-deps above regardless, since letting pip resolve
# it would replace JetPack's torch), and NVIDIA's own README lists this as a
# separate, explicit step. Deliberately NOT --no-deps here: transformers'
# only unconstrained core dependency is numpy>=1.17, already satisfied by the
# pinned numpy<2 install, so pip leaves it alone rather than upgrading it --
# and letting pip resolve the rest (huggingface_hub, httpx, idna, ...)
# normally is what avoids discovering each missing piece one traceback at a
# time.
log "transformers (NanoOWL's runtime dependency)"
if "$PYTHON" -c 'from transformers.models.owlvit.modeling_owlvit import OwlViTForObjectDetection' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install transformers
fi

log "Verifying"
"$PYTHON" scripts/doctor.py
