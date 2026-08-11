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

# Every pip install below can decide it wants a different torch. --no-deps
# blocks that wherever we control the command, but a *transitive* dependency
# can still declare an unconstrained torch and pull a PyPI CUDA wheel over
# JetPack's build -- silently, mid-script, announced as nothing worse than
# "Successfully installed". It surfaces much later, and somewhere else: as
# "torchvision not importable", or as torch.cuda.is_available() going False,
# by which point the install that caused it has scrolled away.
#
# So fingerprint torch and re-check after each section. This does not prevent
# the damage; it names the section that did it, while the command is still on
# screen.
torch_fingerprint() {
  "$PYTHON" - <<'PYEOF' 2>/dev/null || echo "absent"
import torch
print(f"{torch.__version__} {torch.__file__}")
PYEOF
}

TORCH_BEFORE="$(torch_fingerprint)"

assert_torch_unchanged() {
  local now
  now="$(torch_fingerprint)"
  if [[ "$now" != "$TORCH_BEFORE" ]]; then
    warn "torch CHANGED while installing: $1"
    warn "  before:  $TORCH_BEFORE"
    warn "  after:   $now"
    if [[ "$now" == "absent" ]]; then
      warn "That install removed torch outright."
    else
      warn "A dependency replaced JetPack's torch, most likely with a PyPI wheel"
      warn "built against a CUDA this driver cannot run."
    fi
    warn "Every CUDA stage fails until this is undone:  $PYTHON scripts/doctor.py --fix"
    # Re-baseline so one replacement is reported once, not by every later check.
    TORCH_BEFORE="$now"
  fi
}

# JetPack's Ubuntu 22.04 ships packaging 21.3 in /usr/lib/python3/dist-packages,
# and setuptools (>=71) prefers an installed `packaging` over its own vendored
# copy. Building any sdist that has a MANIFEST.in then dies inside setuptools'
# own metadata path with:
#     TypeError: canonicalize_version() got an unexpected keyword argument
#     'strip_trailing_zero'
# -- a keyword that arrived in packaging 22.0. Nothing about the package being
# built is wrong, which is what makes it so confusing to read: CLIP fails this
# way every time (it has a MANIFEST.in) while segment-anything installs fine
# (it has none, so setuptools never reaches prune_file_list). The editable
# installs below go through the same metadata path.
#
# Installed into the venv, which precedes /usr/lib/python3/dist-packages on
# sys.path, so this shadows the old copy without sudo or touching JetPack.
# Pure Python, no torch, nothing here to break.
log "packaging (JetPack's 21.3 breaks sdist builds under modern setuptools)"
if "$PYTHON" - <<'PYEOF' 2>/dev/null
import inspect
from packaging.utils import canonicalize_version
assert "strip_trailing_zero" in inspect.signature(canonicalize_version).parameters
PYEOF
then
  ok "already new enough"
else
  "$PYTHON" -m pip install "packaging>=24.2"
fi

# The other half of the same class of problem, from the opposite direction.
# Ubuntu 22.04 -- and so every stock JetPack image and container built on it --
# ships setuptools 59.6.0, which predates PEP 621: it does not read a
# [project] table at all. Building a project that has one then succeeds and
# produces a package named UNKNOWN, version 0.0.0, containing nothing:
#
#     Building wheel for UNKNOWN (pyproject.toml) ... done
#     Created wheel for UNKNOWN: filename=UNKNOWN-0.0.0-py3-none-any.whl size=12921
#     Successfully installed UNKNOWN-0.0.0
#
# pip reports success, `import clip` then fails, and the 12 KB is the giveaway
# -- the real wheel is 1.4 MB, nearly all of it the BPE vocabulary. CLIP is
# built from git here and declares [project], so it hits this exactly.
# setuptools 61 added PEP 621; CLIP's own build-system asks for >=70.
log "setuptools (Ubuntu 22.04's 59.6.0 builds [project] packages as UNKNOWN-0.0.0)"
if "$PYTHON" -c 'import setuptools; assert int(setuptools.__version__.split(".")[0]) >= 70' 2>/dev/null; then
  ok "already new enough"
else
  "$PYTHON" -m pip install -U "setuptools>=70" wheel
fi

# Clean up after a previous run that hit the above. Left installed, this
# occupies the name pip checks and hides the real package.
if "$PYTHON" -m pip show UNKNOWN >/dev/null 2>&1; then
  warn "removing a stray UNKNOWN-0.0.0 left by an earlier failed build"
  "$PYTHON" -m pip uninstall -y UNKNOWN
fi

# torch has to import before anything below is meaningful -- every repo here
# imports it at install time or first use. Not an install step: see the module
# docstring for why installing a torch is the wrong move when one is already
# present but built for a different environment.
log "torch must actually import"
"$PYTHON" scripts/fix_torch.py || warn "torch is still not importable -- see above"

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

assert_torch_unchanged "the four model repos"

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

# onnx is not one of the four repos either -- it's what torch.onnx.export()
# needs to serialize a model, and NanoOWL's engine build calls it. Not
# --no-deps: onnx doesn't depend on torch, so there's nothing here for
# --no-deps to protect against.
#
# NanoSAM's mask decoder does NOT need this: build_engines.sh fetches a
# pre-built ONNX for it by default rather than exporting fresh, since a fresh
# export reliably produces a graph trtexec rejects on a modern torch (see
# the comment in build_engines.sh).
log "onnx (needed to build the NanoOWL engine)"
if "$PYTHON" -c 'import onnx' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install onnx
fi

# Everything below is needed by EfficientViT-SAM's own runtime import chain
# (efficientvit.models.efficientvit.sam), traced with a static AST analyzer
# after manual, file-by-file tracing missed three of these in a row --
# Python executes a package's __init__.py in full before any of its
# submodules are usable, and efficientvit's own __init__.py files pull in
# far more than the SAM predictor alone needs. None of these are declared by
# efficientvit (installed with --no-deps above), and none need --no-deps
# themselves -- none of them depend on torch.

# segment_anything (Meta's original SAM): models/efficientvit/sam.py imports
# it directly. efficientvit's setup.py even declares this, as a git
# dependency -- exactly what --no-deps skips. Not on PyPI under a
# trustworthy name; installed from the source repo, same as torch2trt.
log "segment_anything (efficientvit's SAM predictor needs it)"
if "$PYTHON" -c 'from segment_anything import SamAutomaticMaskGenerator' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install "git+https://github.com/facebookresearch/segment-anything.git"
fi

# pycocotools: segment_anything's OWN __init__.py unconditionally imports
# automatic_mask_generator.py, which needs this -- even though nothing in
# this app's actual usage (EfficientViTSamPredictor) ever calls automatic
# mask generation. Ships a real aarch64 wheel; not a from-source build.
log "pycocotools (segment_anything's own __init__.py pulls it in)"
if "$PYTHON" -c 'import pycocotools' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install pycocotools
fi

# omegaconf: models/efficientvit/__init__.py unconditionally does
# `from .dc_ae import *` alongside `from .sam import *` -- dc_ae.py imports
# omegaconf at module level. Reached just by importing the SAM predictor's
# own package, regardless of which model inside it is actually used.
log "omegaconf (pulled in by models/efficientvit/__init__.py alongside sam.py)"
if "$PYTHON" -c 'import omegaconf' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install omegaconf
fi

# onnxsim: models/nn/__init__.py -> drop.py -> apps.trainer.run_config ->
# apps.trainer (package __init__) -> apps.trainer.base -> apps.utils
# (package __init__) -> apps/utils/export.py, which does
# `from onnxsim import simplify` at module level. Never actually called by
# anything this app uses; only the import has to succeed.
log "onnxsim (reached via models/nn -> apps.trainer -> apps.utils.export)"
if "$PYTHON" -c 'import onnxsim' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install onnxsim
fi

# timm: continuing that same chain, apps.trainer.base also imports
# efficientvit.apps.data_provider, whose own __init__.py pulls in
# apps/data_provider/augment/color_aug.py, which does
# `from timm.data.auto_augment import rand_augment_transform`. This is a
# SEPARATE reason from NanoSAM's export script (which also needs timm, via
# its own unrelated vendored-MobileSAM path, only under
# NANOSAM_EXPORT_DECODER=1) -- efficientvit needs it unconditionally, for
# its default runtime path. --no-deps matters here, unlike everything else
# above: timm's pyproject.toml declares torch AND torchvision as hard,
# unconstrained dependencies, so a plain install risks replacing JetPack's
# build.
log "timm (efficientvit's own color-augmentation module needs it)"
if "$PYTHON" -c 'import timm' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install timm --no-deps
fi

# triton (OpenAI's GPU kernel compiler): models/nn/__init__.py also does
# `from .norm import *`, and norm.py unconditionally imports
# TritonRMSNorm2dFunc from triton_rms_norm.py -- so importing
# efficientvit.models.nn at all requires triton, even though the L0 SAM
# variant this project uses never actually selects triton-based
# normalization (it uses plain batchnorm; norm="bn2d" in sam_model_zoo.py).
# Only the import has to succeed, the kernel is never JIT-compiled or run.
log "triton (efficientvit.models.nn imports it unconditionally)"
if "$PYTHON" -c 'import triton' >/dev/null 2>&1; then
  ok "already importable"
else
  "$PYTHON" -m pip install triton
fi

assert_torch_unchanged "the NanoOWL / EfficientViT-SAM runtime dependencies"

# ── YOLO-World-S ─────────────────────────────────────────────────────────
# The second detector. Set SKIP_YOLOWORLD=1 to leave it out -- nothing in
# the default NanoOWL pairings needs any of the three installs below.
#
# ultralytics declares torch, torchvision AND opencv-python as hard
# dependencies, so --no-deps is doubly required here: the first two would
# replace JetPack's builds (the failure this whole script exists to avoid),
# and opencv-python would shadow JetPack's cv2, which is the one built with
# CUDA and GStreamer. Its remaining dependencies are installed by name
# below.
SKIP_YOLOWORLD="${SKIP_YOLOWORLD:-0}"
if [[ "$SKIP_YOLOWORLD" == "1" ]]; then
  log "SKIP_YOLOWORLD=1 -- skipping ultralytics, CLIP and their dependencies"
else
  log "ultralytics (YOLO-World-S detector)"
  if "$PYTHON" -c 'from ultralytics import YOLOWorld' >/dev/null 2>&1; then
    ok "already importable"
  else
    "$PYTHON" -m pip install ultralytics --no-deps
  fi

  # ultralytics' own dependencies, minus torch/torchvision/opencv-python.
  # numpy is already pinned <2 above and pillow/pyyaml/requests usually come
  # in with transformers, but naming them all keeps this independent of
  # what happened to be installed first. None of these eight declares torch.
  log "ultralytics' safe dependencies (no torch, no opencv-python)"
  "$PYTHON" -m pip install \
    filelock matplotlib pillow pyyaml requests psutil polars nvidia-ml-py

  # ultralytics-thop is the exception, and it gets its own line because of it:
  # it declares an unconstrained `torch`. Listed alongside the eight above it
  # re-opens the exact hole --no-deps was added to close -- pip is entitled to
  # satisfy that requirement from PyPI, and on a Jetson the newest PyPI torch
  # is a CUDA-13 wheel that the driver cannot run and that leaves torchvision
  # behind. thop's *only* dependency is torch, and torch is already installed,
  # so --no-deps skips nothing real here.
  #
  # ultralytics imports thop lazily and behind a try/except (nn/tasks.py, for
  # FLOPs profiling), so this is not load-bearing for detection either way.
  log "ultralytics-thop (--no-deps: it declares an unconstrained torch)"
  "$PYTHON" -m pip install ultralytics-thop --no-deps
  assert_torch_unchanged "ultralytics and its dependencies"

  # CLIP encodes the text prompts inside YOLOWorld.set_classes(). This is
  # NOT optional and it must be installed HERE, ahead of time: ultralytics
  # falls back to running `pip install git+.../CLIP.git` itself at the
  # moment set_classes is first called, without --no-deps -- so a missing
  # CLIP means PyPI torch landing on top of JetPack's build in the middle
  # of a benchmark run. CLIP declares torch and torchvision too, hence
  # --no-deps and its own three dependencies by name.
  log "CLIP (encodes YOLO-World's prompts; pre-installed so ultralytics never self-installs it)"
  if "$PYTHON" -c 'import clip' >/dev/null 2>&1; then
    ok "already importable"
  else
    # --no-build-isolation is what makes the setuptools pin above take effect.
    # With isolation, pip builds in a temp prefix whose sys.path the system
    # dist-packages still shadow, so Ubuntu's setuptools 59.6.0 can win over
    # the >=70 that CLIP's build-system asked for -- silently, producing
    # UNKNOWN-0.0.0. Without isolation the build uses the venv's setuptools,
    # which is the one just verified.
    "$PYTHON" -m pip install "git+https://github.com/ultralytics/CLIP.git" \
      --no-deps --no-build-isolation
    "$PYTHON" -m pip install ftfy regex tqdm
    if "$PYTHON" -c 'import clip' >/dev/null 2>&1; then
      ok "installed"
    else
      warn "CLIP did not install -- the YOLO-World pairings will be unavailable"
      warn "Everything else still works; the NanoOWL pairings do not touch CLIP."
    fi
  fi
  assert_torch_unchanged "CLIP"
fi

log "Verifying"
"$PYTHON" scripts/doctor.py
