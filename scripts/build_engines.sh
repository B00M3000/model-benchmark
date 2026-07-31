#!/usr/bin/env bash
# Fetch weights and build the TensorRT engines this study needs.
#
# Run once on the Jetson Orin, from the repository root. Engine files are
# tied to the exact TensorRT version and GPU they were built on, so they
# cannot be copied from another machine -- they must be built here.
#
# Expects nanoowl, nanosam and efficientvit to already be installed against
# the JetPack torch/TensorRT build. See README.md.
set -euo pipefail

cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# NOT a plain `import $1` check. This script's cwd is the repo root, which
# contains clone directories (nanosam/, torch2trt/, ...) sharing their name
# with the package one level inside them -- so a naive import can silently
# "succeed" against an empty PEP 420 namespace package instead of the real,
# pip-installed one. module_status() (benchmark/models/registry.py) tells
# them apart via spec.origin. See scripts/setup_jetson.sh for the fuller
# writeup of this exact failure mode.
require_importable() {
  local module="$1" hint="$2"
  if ! "$PYTHON" - "$module" <<PYEOF
import sys
sys.path.insert(0, ".")
from benchmark.models.registry import module_status
status, _ = module_status(sys.argv[1])
sys.exit(0 if status == "ok" else 1)
PYEOF
  then
    printf '\n%s is not really installed (only its clone directory is on sys.path,\nwhich resolves as an empty namespace package -- see the comment above).\n%s\n' \
      "$module" "$hint" >&2
    exit 1
  fi
}

if ! have trtexec; then
  export PATH="/usr/src/tensorrt/bin:$PATH"
fi

PYTHON="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

# Preflight. Engine builds take minutes and load torch on the way in, so a
# broken CUDA/torch pairing is worth catching now rather than three steps
# from here. SKIP_DOCTOR=1 bypasses it.
if [[ "${SKIP_DOCTOR:-0}" != "1" ]]; then
  if ! "$PYTHON" scripts/doctor.py; then
    cat <<'EOF'

Preflight found problems that will break the engine build.
Fix them, or re-run with SKIP_DOCTOR=1 to proceed anyway.

Missing weights and engines are expected on a first run -- this script
creates them. Problems with torch, CUDA or TensorRT are not, and will
cause the build below to fail.
EOF
    read -r -p "Continue anyway? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || exit 1
  fi
fi

# NanoOWL's builder loads the finished engine back through torch2trt as its
# last step, so a missing torch2trt wastes the whole build before failing.
# Check it up front.
require_importable torch2trt \
  "./scripts/setup_jetson.sh          # installs it along with the model repos
or by hand:
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    pip install ./torch2trt --no-deps --no-build-isolation
(--no-build-isolation: its setup.py imports tensorrt/torch to compile a CUDA
extension, and pip's isolated build env doesn't inherit --system-site-packages)"

# ── NanoOWL: OWL-ViT image encoder ────────────────────────────────────────
if [[ -f "$DATA_DIR/owl_image_encoder_patch32.engine" ]]; then
  log "NanoOWL engine already present, skipping"
else
  log "Building NanoOWL image encoder engine (several minutes)"
  "$PYTHON" -m nanoowl.build_image_encoder_engine \
    "$DATA_DIR/owl_image_encoder_patch32.engine"
fi

# ── NanoSAM: ResNet18 encoder + MobileSAM mask decoder ────────────────────
#
# Neither ONNX file ships in the nanosam repo -- the README points at a
# Google Drive link for the encoder, which needs a confirm-token dance for
# files this size and is unreachable from behind some proxies. NVIDIA's own
# jetson-containers build hit the same problem and switched to a GitHub
# mirror (github.com/johnnynunez/nanosam) as the working replacement; that
# mirror is used here, first, for the same reason. The Drive link is the
# fallback for anyone who'd rather verify against NVIDIA's own listing.
#
# resnet18_image_encoder.onnx sha256, for anyone who wants to check it:
#   d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a
RESNET18_ONNX_MIRROR="https://raw.githubusercontent.com/johnnynunez/nanosam/main/data/resnet18_image_encoder.onnx"
RESNET18_ONNX_DRIVE_ID="14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR"

if [[ -f "$DATA_DIR/resnet18_image_encoder.engine" ]]; then
  log "NanoSAM encoder engine already present, skipping"
else
  if [[ ! -f "$DATA_DIR/resnet18_image_encoder.onnx" ]]; then
    log "Fetching resnet18_image_encoder.onnx"
    if ! curl -fL --retry 4 --retry-delay 2 \
        -o "$DATA_DIR/resnet18_image_encoder.onnx" "$RESNET18_ONNX_MIRROR"; then
      rm -f "$DATA_DIR/resnet18_image_encoder.onnx"
      echo "Mirror fetch failed. Download by hand from NVIDIA's own listing:"
      echo "  https://drive.google.com/file/d/${RESNET18_ONNX_DRIVE_ID}/view"
      echo "and save it to $DATA_DIR/resnet18_image_encoder.onnx"
      exit 1
    fi
  fi
  log "Building NanoSAM image encoder engine"
  trtexec \
    --onnx="$DATA_DIR/resnet18_image_encoder.onnx" \
    --saveEngine="$DATA_DIR/resnet18_image_encoder.engine" \
    --fp16
fi

if [[ -f "$DATA_DIR/mobile_sam_mask_decoder.engine" ]]; then
  log "NanoSAM decoder engine already present, skipping"
else
  if [[ ! -f "$DATA_DIR/mobile_sam_mask_decoder.onnx" ]]; then
    # This one is produced with nanosam's own export script rather than
    # fetched pre-built, so nothing here depends on a third-party mirror:
    # only the mobile_sam.pt checkpoint, which ships in the nanosam repo
    # itself (assets/mobile_sam.pt).
    if [[ ! -f "$DATA_DIR/mobile_sam.pt" ]]; then
      log "Fetching MobileSAM checkpoint"
      curl -fL --retry 4 --retry-delay 2 \
        -o "$DATA_DIR/mobile_sam.pt" \
        "https://raw.githubusercontent.com/NVIDIA-AI-IOT/nanosam/main/assets/mobile_sam.pt"
    fi
    require_importable nanosam \
      "./scripts/setup_jetson.sh          # installs it along with the other model repos
or by hand:
    git clone https://github.com/NVIDIA-AI-IOT/nanosam
    pip install ./nanosam --no-deps"
    log "Exporting NanoSAM mask decoder to ONNX"
    "$PYTHON" -m nanosam.tools.export_sam_mask_decoder_onnx \
      --checkpoint="$DATA_DIR/mobile_sam.pt" \
      --model-type=vit_t \
      --output="$DATA_DIR/mobile_sam_mask_decoder.onnx"
  fi
  log "Building NanoSAM mask decoder engine"
  trtexec \
    --onnx="$DATA_DIR/mobile_sam_mask_decoder.onnx" \
    --saveEngine="$DATA_DIR/mobile_sam_mask_decoder.engine" \
    --fp16 \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x4x2,point_labels:1x4 \
    --maxShapes=point_coords:1x8x2,point_labels:1x8
fi

# ── EfficientViT-SAM ──────────────────────────────────────────────────────
# The PyTorch path works out of the box with just the checkpoint; the
# TensorRT engines are optional but markedly faster. config.yaml's
# runtime: auto picks the engines when they exist.
if [[ ! -f "$DATA_DIR/efficientvit_sam_l0.pt" ]]; then
  log "Downloading EfficientViT-SAM-L0 checkpoint"
  curl -fL --retry 4 --retry-delay 2 -o "$DATA_DIR/efficientvit_sam_l0.pt" \
    "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt"
else
  log "EfficientViT-SAM checkpoint already present, skipping"
fi

if [[ -f "$DATA_DIR/efficientvit_sam_l0_encoder.onnx" \
   && ! -f "$DATA_DIR/efficientvit_sam_l0_encoder.engine" ]]; then
  log "Building EfficientViT-SAM encoder engine"
  trtexec \
    --onnx="$DATA_DIR/efficientvit_sam_l0_encoder.onnx" \
    --saveEngine="$DATA_DIR/efficientvit_sam_l0_encoder.engine" \
    --fp16
fi

if [[ -f "$DATA_DIR/efficientvit_sam_l0_decoder.onnx" \
   && ! -f "$DATA_DIR/efficientvit_sam_l0_decoder.engine" ]]; then
  log "Building EfficientViT-SAM decoder engine"
  trtexec \
    --onnx="$DATA_DIR/efficientvit_sam_l0_decoder.onnx" \
    --saveEngine="$DATA_DIR/efficientvit_sam_l0_decoder.engine" \
    --fp16 \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x4x2,point_labels:1x4 \
    --maxShapes=point_coords:1x8x2,point_labels:1x8
fi

log "Done. Artefacts in $DATA_DIR:"
ls -lh "$DATA_DIR" | grep -E '\.(engine|pt)$' || true

cat <<'EOF'

Before benchmarking, pin the clocks or the numbers will be noise:

    sudo nvpmodel -m 0
    sudo jetson_clocks

Then start the server:

    ./scripts/run.sh
EOF
