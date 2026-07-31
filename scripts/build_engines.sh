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
if ! "$PYTHON" -c 'import torch2trt' >/dev/null 2>&1; then
  cat <<'EOF'

torch2trt is not installed, and the engine build needs it.

NanoOWL and NanoSAM both run their TensorRT engines through torch2trt's
TRTModule, but neither declares it as a dependency and it is not on PyPI.

    ./scripts/setup_jetson.sh          # installs it along with the model repos

or by hand:

    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    pip install ./torch2trt --no-deps

EOF
  exit 1
fi

# ── NanoOWL: OWL-ViT image encoder ────────────────────────────────────────
if [[ -f "$DATA_DIR/owl_image_encoder_patch32.engine" ]]; then
  log "NanoOWL engine already present, skipping"
else
  log "Building NanoOWL image encoder engine (several minutes)"
  "$PYTHON" -m nanoowl.build_image_encoder_engine \
    "$DATA_DIR/owl_image_encoder_patch32.engine"
fi

# ── NanoSAM: ResNet18 encoder + MobileSAM mask decoder ────────────────────
if [[ -f "$DATA_DIR/resnet18_image_encoder.engine" ]]; then
  log "NanoSAM encoder engine already present, skipping"
else
  log "Fetching NanoSAM ONNX artefacts"
  if [[ ! -f "$DATA_DIR/resnet18_image_encoder.onnx" ]]; then
    echo "MISSING: $DATA_DIR/resnet18_image_encoder.onnx"
    echo "Download it from the NanoSAM README (NVIDIA-AI-IOT/nanosam) and re-run."
    exit 1
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
    echo "MISSING: $DATA_DIR/mobile_sam_mask_decoder.onnx"
    echo "Export it with nanosam's export script, then re-run."
    exit 1
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
