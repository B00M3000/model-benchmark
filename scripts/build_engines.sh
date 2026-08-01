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

# Both the NanoOWL and NanoSAM engine builds export to ONNX via
# torch.onnx.export(), which needs the onnx package itself to serialize the
# result -- but neither repo declares it, so a --no-deps install lacks it.
# Checked here, not just left to the doctor preflight above, because that
# preflight can be waved past with "Continue anyway?", and without this the
# failure only surfaces after several minutes of tracing the model.
if ! "$PYTHON" -c 'import onnx' >/dev/null 2>&1; then
  cat <<'EOF' >&2

onnx is not installed, and both engine builds below need it.

torch.onnx.export() (used by NanoOWL's build_image_encoder_engine and
NanoSAM's export_sam_mask_decoder_onnx) needs the onnx package itself to
serialize its output, but neither repo declares it as a dependency.

    pip install onnx      # not --no-deps -- onnx doesn't depend on torch,
                           # so there's nothing here for --no-deps to protect

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

# Default: fetch a pre-built ONNX, exactly like the encoder above, rather
# than exporting fresh with nanosam's own export script. This isn't for the
# encoder's reason (avoiding Google Drive) -- exporting the mask decoder
# fresh reliably PRODUCES A GRAPH TRTEXEC REJECTS on a modern torch. The
# vendored MobileSAM PromptEncoder does boolean-mask assignment
# (point_embedding[labels == -1] = 0.0, in prompt_encoder.py), and torch's
# TorchScript-based ONNX exporter traces that differently depending on torch
# version: 2.4.1 emits Where/Equal/Not (fine); 2.8.0 emits a OneHot feeding
# into Tile's shape input, which trtexec rejects outright:
#   Error Code 4: Internal Error (/OneHot: an IIOneHotLayer cannot be used
#   to compute a shape tensor)
# This is a known, still-open upstream issue (NVIDIA-AI-IOT/nanosam#16) with
# no fix from either project -- NVIDIA's own jetson-containers build sidesteps
# it exactly this way, fetching a pre-exported ONNX instead of exporting on
# whatever torch happens to be installed. Confirmed the mirror's copy has no
# OneHot node and matches the exact input/output names
# (image_embeddings/point_coords/point_labels/mask_input/has_mask_input ->
# iou_predictions/low_res_masks) nanosam's own Predictor expects.
#
# mobile_sam_mask_decoder.onnx sha256, for anyone who wants to check it:
#   3bcf84c17762173110783980ccdb8c97d2fd463dc62091dbcd667682bd297de3
DECODER_ONNX_MIRROR="https://raw.githubusercontent.com/johnnynunez/nanosam/main/data/mobile_sam_mask_decoder.onnx"

# Set to export fresh instead -- e.g. to target a different checkpoint or
# model-type, or once the upstream OneHot issue is eventually fixed. Needs
# nanosam (--no-deps) and timm (--no-deps; declares torch/torchvision as
# hard deps, so DON'T drop --no-deps for it) -- neither is needed on the
# default, pre-built path, since Predictor only ever loads compiled engines.
NANOSAM_EXPORT_DECODER="${NANOSAM_EXPORT_DECODER:-0}"

if [[ -f "$DATA_DIR/mobile_sam_mask_decoder.engine" ]]; then
  log "NanoSAM decoder engine already present, skipping"
else
  if [[ ! -f "$DATA_DIR/mobile_sam_mask_decoder.onnx" ]]; then
    if [[ "$NANOSAM_EXPORT_DECODER" == "1" ]]; then
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

      # nanosam itself being installed doesn't mean its vendored MobileSAM
      # does. Checked as two independent, direct imports rather than one
      # `from nanosam.mobile_sam import sam_model_registry` -- that chain
      # hits timm (via modeling/tiny_vit_sam.py) before pycocotools (via
      # mobile_sam's own __init__.py unconditionally importing
      # automatic_mask_generator.py), so a single combined check would
      # misreport a missing pycocotools as a missing timm. Neither is
      # declared by nanosam (installed with --no-deps, like the rest).
      if ! "$PYTHON" -c 'import timm' >/dev/null 2>&1; then
        cat <<'EOF' >&2

timm is not installed, and nanosam's vendored MobileSAM needs it
(mobile_sam/modeling/tiny_vit_sam.py imports timm.models.layers).

    pip install timm --no-deps

--no-deps matters here: unlike most of this project's other runtime
dependencies, timm declares torch AND torchvision as hard dependencies, so
a plain install risks replacing JetPack's build. timm's other dependencies
(pyyaml, huggingface_hub, safetensors) are already installed via the
transformers step, so nothing is lost by skipping them.

EOF
        exit 1
      fi
      if ! "$PYTHON" -c 'import pycocotools' >/dev/null 2>&1; then
        cat <<'EOF' >&2

pycocotools is not installed. nanosam's vendored MobileSAM needs it too --
mobile_sam/__init__.py unconditionally imports automatic_mask_generator.py,
even though nothing this app uses ever calls automatic mask generation.

    pip install pycocotools

Not --no-deps: pycocotools doesn't depend on torch, so there's nothing here
for --no-deps to protect against. Ships a real aarch64 wheel, not a
from-source build.

EOF
        exit 1
      fi

      log "Exporting NanoSAM mask decoder to ONNX (NANOSAM_EXPORT_DECODER=1)"
      echo "Note: this reliably produces a graph trtexec rejects on torch >= ~2.5" \
           "-- see the comment above this block. Unset NANOSAM_EXPORT_DECODER to" \
           "use the known-working pre-built ONNX instead."
      "$PYTHON" -m nanosam.tools.export_sam_mask_decoder_onnx \
        --checkpoint="$DATA_DIR/mobile_sam.pt" \
        --model-type=vit_t \
        --output="$DATA_DIR/mobile_sam_mask_decoder.onnx"
    else
      log "Fetching mobile_sam_mask_decoder.onnx (pre-built, avoids a known trtexec/OneHot failure)"
      if ! curl -fL --retry 4 --retry-delay 2 \
          -o "$DATA_DIR/mobile_sam_mask_decoder.onnx" "$DECODER_ONNX_MIRROR"; then
        rm -f "$DATA_DIR/mobile_sam_mask_decoder.onnx"
        echo "Mirror fetch failed. Set NANOSAM_EXPORT_DECODER=1 to export fresh instead"
        echo "(see the comment above this block for why that may fail on a modern torch),"
        echo "or download by hand and save to $DATA_DIR/mobile_sam_mask_decoder.onnx"
        exit 1
      fi
    fi
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
# EfficientViT-SAM can run straight from the PyTorch checkpoint, but leaving
# it there would quietly bias the whole study: pairing A (NanoSAM) is
# TensorRT-only, so a PyTorch pairing B measures "EfficientViT-SAM without
# TensorRT" and reports it as "EfficientViT-SAM". The engines are built here
# so both pairings are compared on the same runtime; config.yaml's
# runtime: auto then picks them up.
#
# Set EFFICIENTVIT_BUILD_ENGINES=0 to stay on the PyTorch path (faster setup,
# but see the caveat above before comparing the numbers).
EFFICIENTVIT_BUILD_ENGINES="${EFFICIENTVIT_BUILD_ENGINES:-1}"
EVIT_MODEL="${EVIT_MODEL:-efficientvit-sam-l0}"
EVIT_SLUG="${EVIT_MODEL//-/_}"

if [[ ! -f "$DATA_DIR/$EVIT_SLUG.pt" ]]; then
  log "Downloading ${EVIT_MODEL} checkpoint"
  curl -fL --retry 4 --retry-delay 2 -o "$DATA_DIR/$EVIT_SLUG.pt" \
    "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/$EVIT_SLUG.pt"
else
  log "EfficientViT-SAM checkpoint already present, skipping"
fi

if [[ "$EFFICIENTVIT_BUILD_ENGINES" != "1" ]]; then
  log "EFFICIENTVIT_BUILD_ENGINES=0 -- skipping engines, EfficientViT-SAM will run via PyTorch"
elif [[ -f "$DATA_DIR/${EVIT_SLUG}_encoder.engine" \
     && -f "$DATA_DIR/${EVIT_SLUG}_decoder.engine" ]]; then
  log "EfficientViT-SAM engines already present, skipping"
else
  require_importable efficientvit \
    "./scripts/setup_jetson.sh          # installs it along with the other model repos"

  # Exported by our own script rather than efficientvit's, because the
  # upstream one declares a dynamic prompt batch -- which makes SAM's
  # repeat_interleave lower to a OneHot feeding a shape tensor, the exact
  # construct trtexec refuses (the NanoSAM failure, again). Both graphs come
  # out fully static. See scripts/export_efficientvit_sam.py for the
  # measurements behind that.
  log "Exporting ${EVIT_MODEL} to ONNX"
  "$PYTHON" scripts/export_efficientvit_sam.py \
    --model "$EVIT_MODEL" \
    --weights "$DATA_DIR/$EVIT_SLUG.pt" \
    --encoder-output "$DATA_DIR/${EVIT_SLUG}_encoder.onnx" \
    --decoder-output "$DATA_DIR/${EVIT_SLUG}_decoder.onnx"

  # No --minShapes/--optShapes/--maxShapes on either build: both graphs are
  # fully static, so there is no optimisation profile to specify.
  if [[ ! -f "$DATA_DIR/${EVIT_SLUG}_encoder.engine" ]]; then
    log "Building EfficientViT-SAM encoder engine (several minutes)"
    trtexec \
      --onnx="$DATA_DIR/${EVIT_SLUG}_encoder.onnx" \
      --saveEngine="$DATA_DIR/${EVIT_SLUG}_encoder.engine" \
      --fp16
  fi

  if [[ ! -f "$DATA_DIR/${EVIT_SLUG}_decoder.engine" ]]; then
    log "Building EfficientViT-SAM mask decoder engine"
    trtexec \
      --onnx="$DATA_DIR/${EVIT_SLUG}_decoder.onnx" \
      --saveEngine="$DATA_DIR/${EVIT_SLUG}_decoder.engine" \
      --fp16
  fi
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
