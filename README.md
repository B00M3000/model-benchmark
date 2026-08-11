# Model Ablation Study — NanoOWL + NanoSAM vs NanoOWL + EfficientViT-SAM

A web app that runs on a Jetson Orin. Upload a video, and both open-vocabulary
detect-then-segment pipelines are run over the same frames **sequentially in a
single process**, recording per-frame latency and FPS. A side-by-side
comparison video is rendered afterwards so mask quality can be judged by eye.

| | Detector | Segmenter |
|---|---|---|
| **Pairing A** | NanoOWL | NanoSAM (ResNet18 encoder) |
| **Pairing B** | NanoOWL | EfficientViT-SAM (L0) |

The detector is identical in both, so the study isolates the cost of the
segmentation head.

### Choosing what to compare

Those two are the default, not the limit. Each run picks two entries from a
catalogue covering both detectors against all three segmentation heads:

| | NanoSAM | EfficientViT-SAM-L0 | EfficientViT-SAM-L2 |
|---|---|---|---|
| **NanoOWL** (TensorRT) | ✓ default A | ✓ default B | ✓ |
| **YOLO-World-S** (PyTorch) | ✓ | ✓ | ✓ |

That grid is the one in Park, Kim & Ko,
[*Real-time open-vocabulary perception for mobile robots on edge devices*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12583037/)
(Front. Robot. AI 12, 2025), so their table can be reproduced pair by pair on
your own hardware and footage.

**Read the delta according to what you varied.** Hold the detector constant
and the gap is attributable to the segmentation head — that is the clean
ablation. Hold the segmenter constant and it is the detector. Change both
(NanoOWL + NanoSAM against YOLO-World-S + L2) and the difference cannot be
attributed to either alone; the UI says so under the selector rather than
letting the number stand unqualified.

**YOLO-World is the slower one by construction.** Ultralytics has no TensorRT
export that keeps the open-vocabulary text head, so it runs in PyTorch while
NanoOWL's image encoder runs through TensorRT. It buys understanding of
longer phrases than NanoOWL's noun-phrase encoder handles — the trade the
paper measures, and the reason both are here.

<br>

## What it measures

**Latency distribution, per stage.** Not just the mean — p50/p90/p95/p99,
min/max and standard deviation (jitter), broken down by stage:

```
decode → preprocess → detect (encode + head) → seg_encode → seg_decode → postprocess
```

Two headline numbers per frame:

- **`pipeline_ms`** — model stages only. This is what a drone with a live
  camera feed experiences, and it's what the UI leads with.
- **`e2e_ms`** — `pipeline_ms` plus video decode, i.e. the full offline cost.

Throughput is reported two ways because they differ and the difference is
meaningful: `frames ÷ total pipeline time`, and the mean of `1/latency`.

Detections per frame are recorded alongside. Both segmenters encode once per
frame and decode once per box, so segmentation latency scales with detection
count — the timings are uninterpretable without it.

<br>

## Measurement methodology

This is the part that has to be right.

- **CUDA is synchronised** around every GPU stage (`torch.cuda.synchronize()`).
  Kernel launches are asynchronous; without the sync a stage's cost silently
  leaks into whichever stage next touches the GPU.
- **Warm-up frames are excluded.** The first 10 frames (configurable) are
  executed but flagged and dropped from every aggregate. Without this the first
  TensorRT/cuDNN calls dominate the mean and both pairings look wrong. The UI
  states how many frames were excluded, and shades them in the timeline chart.
- **Both runs see identical frames** — same decode, same stride, same order.
- **Fully independent runs.** Pairing B executes its own NanoOWL forward pass
  on every frame; no detections are cached or shared between runs.
- **Text prompts are encoded once** at load, not per frame. A drone would cache
  them too, so the cost is reported separately rather than charged to frames.
- **Segmentation is skipped when there are no detections**, exactly as a real
  deployment would. `frames_with_no_detections` is reported so the averages stay
  interpretable.
- **Mask recording never inflates the timings.** The timed region ends before
  masks are handed to a background thread that RLE-encodes and writes them.
  Set `record_masks: false` for a zero-overhead pure-latency run.
- **One job at a time.** A global lock serialises benchmarks; concurrent GPU
  work would contaminate every number.

### Single-process model lifecycle

Both pairings run on the same worker thread in the same process, so the CUDA
context, TensorRT runtime and PyTorch allocator initialise **once per job**
rather than once per pairing.

- **NanoOWL is loaded once and reused across both runs.** The weights, engine
  and cached text encodings are identical either way, so reloading would only
  add engine-deserialisation time. Each pairing still runs its own detection
  pass — this is a loading optimisation, not a sharing of results.
- **Only the segmenter is swapped.** NanoSAM is released (`del` + `gc.collect()`
  + `torch.cuda.empty_cache()`) before EfficientViT-SAM loads, so pairing B
  never competes with pairing A's memory. The gap is reported as `swap_ms`
  rather than hidden.
- Backends get a `reset()` call at the start of each run to clear per-run
  state, so nothing carries over from the previous pairing.

Tune via `lifecycle` in `config.yaml`:

| Option | Default | Effect |
|---|---|---|
| `share_detector` | `true` | Keep NanoOWL resident across both runs |
| `preload_all_models` | `false` | Load both segmenters up front — smallest inter-run gap, higher peak VRAM |
| `reload_between_runs` | `false` | Full teardown/reload between runs for maximum isolation |

Model loading always sits outside the timed region, so none of these affect
per-frame numbers.

<br>

## Install on the Jetson Orin

Requires JetPack 5.1+ with CUDA, cuDNN and TensorRT.

```bash
git clone <this repo> && cd model-benchmark
python3 -m venv .venv --system-site-packages   # inherit JetPack's torch/TensorRT
.venv/bin/pip install -r requirements-jetson.txt
```

Use **`requirements-jetson.txt`**, not `requirements.txt` — the latter is for
x86 development and installs an OpenCV wheel that drags in NumPy 2.

`--system-site-packages` matters: the Jetson's `torch`, `torchvision`,
`tensorrt` and `cv2` come from JetPack and must not be replaced by pip wheels.
Two constraints follow, and both cause failures far from their cause:

- **NumPy must stay on 1.x.** JetPack's torch is compiled against NumPy 1.x.
  NumPy 2 leaves torch importable but breaks its C bridge, so `torch.from_numpy`
  fails with *"Numpy is not available"* — surfacing inside NanoOWL, not at
  install time.
- **OpenCV should come from JetPack** (`sudo apt install python3-opencv`). The
  PyPI wheel is CPU-only, lacks the hardware decoders, and 4.11+ requires
  NumPy 2, which reintroduces the problem above.

Then install the four model packages against that same environment:

```bash
./scripts/setup_jetson.sh
```

That clones and installs them, then runs the doctor. By hand it is:

```bash
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
git clone https://github.com/NVIDIA-AI-IOT/nanosam
git clone https://github.com/mit-han-lab/efficientvit
git clone https://github.com/NVIDIA-AI-IOT/torch2trt

.venv/bin/pip install -e nanoowl      --no-deps
.venv/bin/pip install -e nanosam      --no-deps
.venv/bin/pip install -e efficientvit --no-deps
.venv/bin/pip install    ./torch2trt  --no-deps --no-build-isolation

# None of these get --no-deps -- see below.
.venv/bin/pip install transformers
.venv/bin/pip install onnx
.venv/bin/pip install "git+https://github.com/facebookresearch/segment-anything.git"
.venv/bin/pip install pycocotools
.venv/bin/pip install omegaconf
.venv/bin/pip install onnxsim
.venv/bin/pip install triton

# timm is the one exception -- it DOES get --no-deps. See below.
.venv/bin/pip install timm --no-deps
```

**`torch2trt` is easy to miss.** It is not on PyPI and neither NanoOWL nor
NanoSAM declares it, so nothing installs it as a side effect — but both do
`from torch2trt import TRTModule` to execute their TensorRT engines. Without it
the NanoOWL engine build runs to completion and then fails on its last line,
loading the finished engine back (`ModuleNotFoundError: No module named
'torch2trt'`). The engine it just built is fine; only the verification step
failed.

**Always pass `--no-deps` for the four repos above.** All four declare `torch`
as a dependency (torch2trt also declares `tensorrt`), and pip will happily pull
a PyPI wheel over JetPack's build, which then fails at runtime with *"The
NVIDIA driver on your system is too old"*.

**`transformers` is the one exception — install it *without* `--no-deps`.**
NanoOWL's `owl_predictor.py` imports `OwlViTForObjectDetection` from it at
runtime, but nanoowl's own `setup.py` declares no dependencies at all, so
nothing pulls `transformers` in. Unlike the four repos above, `transformers`
does **not** declare `torch` as a hard dependency (only as an optional extra),
so there's nothing here for `--no-deps` to protect against — and skipping it
means discovering `transformers`' own dependency chain one missing piece at a
time: first `ModuleNotFoundError: No module named 'idna'`, fix that and hit
the next one, and so on, since `httpx` needs `idna`, `huggingface_hub` needs
`httpx`, and `transformers` needs `huggingface_hub`. A plain
`pip install transformers` resolves the whole chain correctly in one shot —
this is also exactly what [NanoOWL's own README](https://github.com/NVIDIA-AI-IOT/nanoowl#setup)
says to run. Its only unconstrained core dependency is `numpy>=1.17`, already
satisfied by the pinned `numpy<2` install, so pip leaves it alone rather than
upgrading it.

**`onnx` is needed only for *building* NanoOWL's engine, not for running a
benchmark.** `build_image_encoder_engine()` calls `torch.onnx.export()` to
produce the `.onnx` file `trtexec` then compiles, and PyTorch's ONNX exporter
needs the `onnx` package itself to serialize the result. NanoOWL doesn't
declare it, so it's easy to burn several minutes tracing the model only to
fail on the last line with `torch.onnx.OnnxExporterError: Module onnx is not
installed!`. Same reasoning as `transformers`: `onnx` doesn't depend on
`torch`, and its `numpy>=1.23.2` is already satisfied, so a plain
`pip install onnx` is safe and complete.

**The next six exist only because of EfficientViT-SAM's own runtime import
chain.** `efficientvit/models/efficientvit/sam.py` — the file
`EfficientViTSamSegmenter` actually imports — needs all of them, but not
because the SAM predictor itself uses them: Python fully executes a
package's `__init__.py` (and every ancestor package's `__init__.py`) before
any of its submodules become usable, and efficientvit's own `__init__.py`
files each pull in far more than the SAM predictor alone needs. These were
found with a static AST analyzer after **three separate rounds** of manual,
file-by-file tracing each missed a different branch of this same import
tree — worth knowing if you ever need to re-audit this yourself: don't trust
a partial manual trace here, the graph is deep.

- **`segment_anything`** (Meta's original SAM) — `sam.py` imports it
  directly. efficientvit's `setup.py` even declares this, as a git
  dependency, but that's exactly what `--no-deps` skips. Not on PyPI under a
  name worth trusting, so installed from the source repo, same as
  `torch2trt`. Zero dependencies of its own.
- **`pycocotools`** — `segment_anything`'s *own* `__init__.py` unconditionally
  imports `automatic_mask_generator.py`, which needs this, even though
  nothing this app uses ever calls automatic mask generation. Ships a real
  aarch64 wheel; not a from-source build.
- **`omegaconf`** — `models/efficientvit/__init__.py` unconditionally does
  `from .dc_ae import *` alongside `from .sam import *`; `dc_ae.py` imports
  `omegaconf` at module level.
- **`onnxsim`** — reached via `models/nn/__init__.py` → `.drop` →
  `apps.trainer.run_config` → (via `apps/trainer/__init__.py`'s own
  `from .base import *`) → `apps.trainer.base` → `efficientvit.apps.data_provider`
  → `apps/utils/export.py`, which does `from onnxsim import simplify`. Never
  actually called by anything this app uses.
- **`timm`** — continuing that same chain, `apps.trainer.base` also imports
  `efficientvit.apps.data_provider`, whose `__init__.py` pulls in
  `augment/color_aug.py`, which imports `timm.data.auto_augment`. **This is
  the one exception that keeps `--no-deps`** — unlike the other five here,
  `timm`'s `pyproject.toml` declares `torch` **and** `torchvision` as hard,
  unconstrained dependencies, so a plain install risks replacing JetPack's
  build. (NanoSAM's vendored MobileSAM *also* needs `timm`, for a completely
  unrelated reason — see the `NANOSAM_EXPORT_DECODER` note in
  Troubleshooting — but that path is opt-in only; efficientvit needs it
  unconditionally, on its default runtime path.)
- **`triton`** (OpenAI's GPU kernel compiler) — `models/nn/__init__.py` also
  does `from .norm import *`, and `norm.py` unconditionally imports
  `TritonRMSNorm2dFunc` from `triton_rms_norm.py`, even though the L0 SAM
  variant this project uses never actually selects triton-based
  normalization (`sam_model_zoo.py` builds it with `norm="bn2d"`) — only the
  *import* has to succeed, the kernel is never JIT-compiled or run. Ships
  aarch64 wheels for JetPack 6's Python (3.10), so this is a plain install,
  not a build from source.

A single combined "does `efficientvit.models.nn` import" check can't tell
these apart reliably: it only ever reveals whichever one is missing *first*
in execution order, misreporting every other gap as that one. `scripts/doctor.py`
checks each of these six independently for exactly that reason.

Check the environment at any point:

```bash
.venv/bin/python scripts/doctor.py
```

It verifies torch matches the driver's CUDA version and can convert NumPy
arrays, that TensorRT and `trtexec` are present, that all four packages import,
that each repo's own runtime dependencies (`transformers`, `onnx`,
`segment_anything`, `pycocotools`, `omegaconf`, `onnxsim`, `timm`, `triton`)
are actually reachable — not just the repo itself, and not just the first of
several missing ones — that torch2trt's `TRTModule` is new enough for the
installed TensorRT, which weights and engines exist, that an H.264 encoder is
available so the comparison video is actually playable, and whether the clocks
are pinned — each with the fix.

Build the TensorRT engines and fetch weights (once, on the Orin — engines are
tied to the exact TensorRT version and GPU that built them and cannot be copied
between machines):

```bash
./scripts/build_engines.sh
```

This builds four engines: NanoOWL's image encoder, NanoSAM's encoder and mask
decoder, and — exported on the spot from efficientvit's own
`applications/efficientvit_sam/deployment/onnx/` scripts — EfficientViT-SAM's
encoder and mask decoder.

**Why EfficientViT-SAM's engines are not optional.** EfficientViT-SAM runs
happily from its PyTorch checkpoint alone, so it is tempting to skip them.
Don't, if you intend to compare the two pairings: NanoSAM has no PyTorch path
at all — `nanosam.utils.predictor.Predictor` only ever loads compiled engines —
so a PyTorch pairing B measures *EfficientViT-SAM without TensorRT* and labels
the result *EfficientViT-SAM*. Part of the gap you would read as "the
segmentation backbone is slower" would just be "one side got TensorRT and the
other didn't", which is precisely the confound this study exists to avoid.
`doctor.py` reports missing EfficientViT-SAM engines as a problem for that
reason, unless `efficientvit.runtime` is explicitly set to `torch`.

`EFFICIENTVIT_BUILD_ENGINES=0 ./scripts/build_engines.sh` skips them anyway
(faster setup, biased comparison); `EVIT_MODEL=efficientvit-sam-l1` builds a
different variant.

**Switching runtime changes the speed, not the masks.** EfficientViT-SAM's
two runtimes are held to producing the same output: `benchmark/models/trt_sam.py`
reuses upstream's own resize/pad/normalise transform, coordinate frame and
mask postprocessing, and does the mask-token selection in Python so it
matches `MaskDecoder.forward` rather than the different choice
`--return-single-mask` bakes into the graph. Verified against
`EfficientViTSamPredictor` on identical weights across landscape, portrait
and square frames — bit-identical masks through the PyTorch modules, and
identical to within one boundary pixel in ~9.5M when driven through the real
exported ONNX. So a difference between pairing A and pairing B is a
difference in the model, not an artefact of how it was run.

### "No module named 'torch2trt'"

```
File "nanoowl/owl_predictor.py", line 383, in load_image_encoder_engine
    from torch2trt import TRTModule
ModuleNotFoundError: No module named 'torch2trt'
```

torch2trt executes the TensorRT engines for both NanoOWL and NanoSAM. It is not
on PyPI and neither repo declares it, so a clean install lacks it.

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
.venv/bin/pip install ./torch2trt --no-deps --no-build-isolation
```

**`--no-build-isolation` is required, not optional, for this one.** torch2trt's
`setup.py` does `import tensorrt` and `import torch` at the top level to compile
a CUDA extension. pip's normal isolated build environment contains only
declared build requirements — it does *not* inherit the outer venv's
`--system-site-packages` — so even though both packages are importable
everywhere else, the isolated build can't see either and fails with
`ModuleNotFoundError: No module named 'tensorrt'` during "Getting requirements
to build wheel". `--no-build-isolation` builds against the current environment
instead, where both are already present.

`--no-deps` matters here too — torch2trt declares `tensorrt`, and pip would
install the PyPI wheel over JetPack's.

If this appeared at the end of `build_engines.sh`, **the engine built
successfully**; the failure is in the step that loads it back to verify.
Installing torch2trt and re-running skips straight past it — the script sees the
existing engine file and moves on.

On TensorRT 10 (JetPack 6), install torch2trt from master rather than a release:
older releases drive engines through the removed binding API and fail at
inference with `no attribute 'num_bindings'`. The doctor flags this.

### "No module named 'idna'" (or `httpx`, `huggingface_hub`, or any other piece of `transformers`' dependency chain)

```
File "nanoowl/owl_predictor.py", line 24, in <module>
    from transformers.models.owlvit.modeling_owlvit import OwlViTForObjectDetection
  ...
  File ".../huggingface_hub/utils/__init__.py", line 16, in <module>
    from huggingface_hub.errors import (...)
  ...
  File ".../httpx/_urls.py", line 6, in <module>
    import idna
ModuleNotFoundError: No module named 'idna'
```

NanoOWL imports `transformers` at runtime, but nanoowl's own `setup.py`
declares no dependencies, so nothing installs it. Fixing this one traceback at
a time is a trap: `idna` is missing because `httpx` needs it, `httpx` is
missing because `huggingface_hub` needs it, and so on — each fix just reveals
the next link in the chain. Install `transformers` itself instead, and let pip
resolve the whole chain in one shot:

```bash
.venv/bin/pip install transformers
```

**Deliberately not `--no-deps` here** — unlike the four repos above,
`transformers` doesn't declare `torch` as a hard dependency (only as an
optional extra), so there's no risk of it replacing JetPack's build. Its only
unconstrained core dependency is `numpy>=1.17`, already satisfied by the
pinned `numpy<2` install, so pip leaves it in place rather than upgrading it.
`scripts/doctor.py` checks this by attempting the exact import NanoOWL
performs, so it names whichever link in the chain is actually missing rather
than guessing.

### "Module onnx is not installed!" (during an engine build)

```
File ".../nanoowl/owl_predictor.py", line 370, in export_image_encoder_onnx
    torch.onnx.export(
  ...
torch.onnx.OnnxExporterError: Module onnx is not installed!
```

Both NanoOWL's and NanoSAM's engine builds export to ONNX via
`torch.onnx.export()` before `trtexec` compiles the result, and PyTorch's
exporter needs the `onnx` package itself to serialize that output. Neither
repo declares it as a dependency, so a `--no-deps` install lacks it — and
because this only fails at the very end of `export_image_encoder_onnx()`,
you'll see several minutes of tracing output first (`Loading weights...`,
the `torch.meshgrid` warning, `TracerWarning`) before hitting this.

```bash
.venv/bin/pip install onnx
```

Not `--no-deps` — `onnx` doesn't depend on `torch` (only `numpy>=1.23.2`,
already satisfied by the pinned `numpy<2` install), so there's nothing here
for `--no-deps` to protect against.

If this happened via `./scripts/build_engines.sh`, note that the script has
`set -euo pipefail`: the moment the *first* step (NanoOWL's engine) fails,
the script stops immediately, so NanoSAM's engine and the EfficientViT
weights never get their turn either — `doctor.py` will still list all of
them as missing afterward even though only this one thing was actually
blocking anything. Re-running the script after installing `onnx` lets every
step after the first one finally run.

### "Error Code 4: Internal Error (/OneHot: an IIOneHotLayer cannot be used to compute a shape tensor)"

```
[TRT] ModelImporter.cpp:948: While parsing node number 146 [Tile -> "/Tile_output_0"]:
...
ITensor::getDimensions: Error Code 4: Internal Error (/OneHot: an IIOneHotLayer
cannot be used to compute a shape tensor)
[E] Failed to parse onnx file
```

...while building `mobile_sam_mask_decoder.engine`. This is **not** a missing
package — this is TensorRT rejecting the ONNX graph itself. This is a known,
still-open issue upstream ([NVIDIA-AI-IOT/nanosam#16](https://github.com/NVIDIA-AI-IOT/nanosam/issues/16)),
not anything specific to this project's setup.

**Root cause:** MobileSAM's `PromptEncoder._embed_points` does boolean-mask
assignment (`point_embedding[labels == -1] = 0.0`, in
`mobile_sam/modeling/prompt_encoder.py`). PyTorch's TorchScript-based ONNX
exporter traces that differently depending on torch version: torch 2.4.1
emits `Where`/`Equal`/`Not` (TensorRT parses this fine); torch 2.8.0 emits a
`OneHot` op feeding into `Tile`'s shape input, which TensorRT explicitly
refuses to use as a shape tensor. Same source code, different torch, different
(and for newer torch, broken) graph.

**Fix:** `build_engines.sh` no longer exports this ONNX file fresh by
default — it fetches a pre-built one instead, confirmed to contain no
`OneHot` node and to match the exact input/output names
(`image_embeddings`/`point_coords`/`point_labels`/`mask_input`/`has_mask_input`
→ `iou_predictions`/`low_res_masks`) `nanosam.utils.predictor.Predictor`
expects. If you already have a broken `data/mobile_sam_mask_decoder.onnx`
from an earlier run, delete it and re-run the script:

```bash
rm -f data/mobile_sam_mask_decoder.onnx data/mobile_sam_mask_decoder.engine
./scripts/build_engines.sh
```

If you specifically need a fresh export (a different checkpoint or
model-type, or once the upstream issue is eventually fixed), set
`NANOSAM_EXPORT_DECODER=1` — this is the one case where you'll actually need
`timm` (see below), since only the export script touches
`nanosam.mobile_sam` at all:

```bash
NANOSAM_EXPORT_DECODER=1 ./scripts/build_engines.sh
```

Expect this to reproduce the same TensorRT error on a modern torch, unless
you also pin an older torch just for the export step.

**EfficientViT-SAM's mask decoder hits the same wall, for a different
reason** — and `scripts/export_efficientvit_sam.py` exists to avoid it. There
the `OneHot` does not come from boolean-mask assignment at all: SAM's
`MaskDecoder.predict_masks` runs
`torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)`, and when
the prompt batch is a dynamic axis that repeat count is symbolic, so torch
lowers it to `OneHot` → `Tile` with the OneHot result feeding `Tile`'s
*repeats* input — a shape tensor, which is exactly what TensorRT refuses.
efficientvit's own `export_decoder.py` declares that dynamic batch axis, so
using it directly reproduces the failure.

Measured on the real graph, pinning the batch axis is *not* enough (the
repeat count stays symbolic and both `OneHot` nodes survive); exporting fully
static removes them, and folds the decoder from 1075 nodes to 436. Static
shapes cost nothing here, since this app segments one box at a time and a box
prompt is exactly one batch of two points. That is why the export is done by
this project's script rather than upstream's, and why neither `trtexec`
invocation for EfficientViT-SAM passes an optimisation profile.

### "No module named 'timm'"

```
File ".../efficientvit/apps/data_provider/augment/color_aug.py", line 6, in <module>
    from timm.data.auto_augment import rand_augment_transform
ModuleNotFoundError: No module named 'timm'
```

or, a completely unrelated way to hit the same missing package:

```
File ".../nanosam/mobile_sam/modeling/tiny_vit_sam.py", line 15, in <module>
    from timm.models.layers import DropPath as TimmDropPath,\
ModuleNotFoundError: No module named 'timm'
```

**`timm` is needed by default now** — it isn't optional. EfficientViT-SAM's
own runtime import chain reaches it unconditionally (see the explanation in
the install section above: `models/nn` → `.drop` → `apps.trainer` →
`apps.data_provider` → `augment/color_aug.py`). Separately, and for a
completely unrelated reason, NanoSAM's vendored MobileSAM *also* imports
`timm` — but only if you set `NANOSAM_EXPORT_DECODER=1` to export the mask
decoder fresh instead of using the pre-built ONNX `build_engines.sh` fetches
by default. Either way, the fix is the same:

```bash
.venv/bin/pip install timm --no-deps
```

**This one keeps `--no-deps`, unlike everything else in this section.**
`timm`'s `pyproject.toml` lists `torch` and `torchvision` as hard,
unconstrained dependencies, so installing it without `--no-deps` risks pip
replacing JetPack's build — the exact failure the rest of this guide exists
to prevent. `timm`'s other three dependencies (`pyyaml`, `huggingface_hub`,
`safetensors`) are already satisfied once `transformers` is installed, so
nothing real is lost by skipping them.

### "No module named 'segment_anything'"

```
File ".../efficientvit/models/efficientvit/sam.py", line 9, in <module>
    from segment_anything import SamAutomaticMaskGenerator
ModuleNotFoundError: No module named 'segment_anything'
```

EfficientViT-SAM's own predictor imports Meta's original SAM directly.
efficientvit's `setup.py` even declares this — as a git dependency — but
that's exactly what `--no-deps` (required for efficientvit itself) skips.

```bash
.venv/bin/pip install "git+https://github.com/facebookresearch/segment-anything.git"
```

Not `--no-deps` — `segment_anything`'s own `setup.py` declares zero
dependencies, so there's nothing here to protect against. It isn't installed
from PyPI because there's no way to confirm a PyPI package under that name is
actually Meta's; installing from the source repo (the same approach used for
`torch2trt`) avoids that ambiguity entirely.

### "No module named 'pycocotools'"

```
File ".../segment_anything/__init__.py", line 14, in <module>
    from .automatic_mask_generator import SamAutomaticMaskGenerator
  File ".../segment_anything/automatic_mask_generator.py", ...
    from pycocotools import mask as mask_utils
ModuleNotFoundError: No module named 'pycocotools'
```

`segment_anything`'s *own* `__init__.py` unconditionally imports
`automatic_mask_generator.py`, which needs this — even though nothing this
app's EfficientViT-SAM pairing ever calls automatic mask generation. The
same applies to NanoSAM's vendored copy of MobileSAM
(`mobile_sam/__init__.py` does the identical unconditional import), if you
set `NANOSAM_EXPORT_DECODER=1`.

```bash
.venv/bin/pip install pycocotools
```

Not `--no-deps` — doesn't depend on `torch`. Ships a real aarch64 wheel;
this is not a from-source Cython build, despite `pycocotools`' reputation
for being one on some platforms.

### "No module named 'omegaconf'"

```
File ".../efficientvit/models/efficientvit/dc_ae.py", line 6, in <module>
    from omegaconf import MISSING, OmegaConf
ModuleNotFoundError: No module named 'omegaconf'
```

`models/efficientvit/__init__.py` unconditionally does `from .dc_ae import
*` right alongside `from .sam import *` — so importing the SAM predictor's
own package pulls in the (unrelated) diffusion-autoencoder module too,
which needs `omegaconf`.

```bash
.venv/bin/pip install omegaconf
```

Not `--no-deps` — doesn't depend on `torch`.

### "No module named 'onnxsim'"

```
File ".../efficientvit/apps/utils/export.py", line 8, in <module>
    from onnxsim import simplify as simplify_func
ModuleNotFoundError: No module named 'onnxsim'
```

Reached via `models/nn/__init__.py` → `.drop` → `apps.trainer.run_config` →
(via `apps/trainer/__init__.py`'s own `from .base import *`) →
`apps.trainer.base` → `efficientvit.apps.utils` → `apps/utils/export.py`.
Never actually called by anything this app uses; only the import has to
succeed.

```bash
.venv/bin/pip install onnxsim
```

Not `--no-deps` — doesn't depend on `torch`.

**If you've just fixed one of `pycocotools`/`omegaconf`/`onnxsim`/`timm`/
`triton` and immediately hit the *next* one:** that's expected, not a sign
something is still wrong. All five sit on the same import chain
(`efficientvit.models.efficientvit.sam` → `models/nn` and
`models/efficientvit`'s own `__init__.py` files), which resolves them one at
a time in a fixed order — fixing one just reveals whichever is missing
next. `scripts/doctor.py` reports all of them in a single pass rather than
one crash at a time.

### "No module named 'triton'"

```
File ".../efficientvit/models/nn/triton_rms_norm.py", line 2, in <module>
    import triton
ModuleNotFoundError: No module named 'triton'
```

Surprising, since nothing in an EfficientViT-SAM-L0 run actually asks for
triton-based normalization — but `efficientvit/models/nn/__init__.py`
unconditionally does `from .norm import *`, and `norm.py` unconditionally
imports `TritonRMSNorm2dFunc` from `triton_rms_norm.py`. So importing
`efficientvit.models.nn` at all requires `triton` to be installed, even
though `sam_model_zoo.py` builds the L0 model with `norm="bn2d"` and the
triton kernel is never actually JIT-compiled or run.

```bash
.venv/bin/pip install triton
```

Not `--no-deps` — `triton`'s only unconstrained dependency is
`importlib-metadata`, and only for Python older than 3.10 (JetPack 6 ships
3.10). `triton` publishes aarch64 `manylinux` wheels for cp310, so this is a
plain, fast wheel install, not a build from source.

### "No module named 'nanosam.tools'" (or any `<pkg>.<submodule>` after a supposedly clean install)

```
python: Error while finding module specification for
'nanosam.tools.export_sam_mask_decoder_onnx' (ModuleNotFoundError: No module
named 'nanosam.tools')
```

Every model repo here is a clone whose root directory shares its name with the
package nested one level inside it — `nanosam/nanosam/__init__.py`. Run from
the repo root (which `setup_jetson.sh` and `build_engines.sh` both do), a plain
`import nanosam` can resolve to the *clone root* as an empty namespace package
(PEP 420) instead of erroring — Python reports success, but nothing real is in
it, so `nanosam.tools` doesn't exist even though `import nanosam` "worked".
This previously made `setup_jetson.sh` believe packages were already installed
and skip installing them for real.

Both scripts now check with `benchmark.models.registry.module_status()`
instead, which tells a genuine install apart from this by checking
`spec.origin` (`None` for a namespace package). If you hit this outside those
scripts, the fix is the same either way — actually install the package:

```bash
.venv/bin/pip install -e nanosam --no-deps
```

### No boxes or masks in the comparison video, and the reported FPS looks too good

Both symptoms, together, usually mean **the detector returned nothing**. The
pipeline deliberately skips the segmenter when a frame has no detections (a
real deployment would not encode an image it has nothing to segment), so zero
detections removes `seg_encode` and `seg_decode` from every frame at once —
which inflates FPS dramatically *and* leaves the render with nothing to draw.
It looks like a rendering bug and is actually a detection bug.

Check the per-frame data rather than the video:

```bash
curl -s localhost:8000/api/jobs/<id>/frames?pairing=a | head -c 600
```

If `num_detections` is `0` everywhere, the detector is the problem. If it is
non-zero but the boxes are small decimals (`[0.31, 0.44, 0.52, 0.68]`) rather
than pixel coordinates, boxes are being left in OWL-ViT's normalised 0..1
space — they collapse to a dot at the origin when drawn with `int()`, and give
both segmenters a sub-pixel prompt.

That was a real bug in this adapter, fixed by routing detection through
nanoowl's `encode_rois` instead of `encode_image`. `encode_image` looks like
the natural split point for timing the encoder separately from the head, but
it is the *middle* of the encode step: `preprocess_pil_image` does not resize,
so the resize to the model's 768×768 input happens inside `encode_rois` (via
`roi_align`), and so does the mapping of boxes from normalised coordinates
back into frame pixels. Skipping it hands a full-resolution frame to an engine
whose spatial dimensions are fixed at 768×768 (`--shapes=image:1x3x768x768`;
only the batch axis is dynamic) and then interprets the result in the wrong
coordinate space. Nothing raises — the detections just come back empty or
sub-pixel.

Two things worth knowing once detections are working:

- **NanoOWL does not apply NMS.** `decode()` returns every patch above the
  threshold, so one object can yield several overlapping boxes. Since the
  pipeline segments *per detection*, `num_detections` drives `seg_decode` cost
  directly — this is a real workload, but read the per-stage numbers alongside
  `num_detections` rather than on their own.
- **The default threshold is 0.1**, which is permissive. Raise it in the run
  config if you are getting a wall of low-confidence boxes.

### "No module named 'ultralytics'" / "No module named 'clip'"

Only the YOLO-World pairings need these; every NanoOWL pairing runs without
them, and `doctor.py` reports them as warnings rather than problems for that
reason.

```bash
pip install ultralytics --no-deps
pip install filelock matplotlib pillow pyyaml requests psutil polars nvidia-ml-py
pip install ultralytics-thop --no-deps        # see below: this one declares torch
pip install git+https://github.com/ultralytics/CLIP.git --no-deps
pip install ftfy regex tqdm
```

**`--no-deps` is not optional here, and CLIP is not optional either.**
ultralytics declares `torch`, `torchvision` **and** `opencv-python` as hard
dependencies: the first two would replace JetPack's builds — the failure the
rest of this document is about — and the third would shadow JetPack's `cv2`,
which is the one compiled with CUDA and GStreamer.

Seven of its remaining dependencies are safe and are installed by name above.
`ultralytics-thop` is the exception and gets its own line: it declares an
unconstrained `torch` of its own, so listing it alongside the other seven
re-opens the exact hole `--no-deps` was added to close — pip may satisfy that
requirement from PyPI, and the newest PyPI torch is a CUDA-13 wheel a JetPack 6
driver cannot run. Its *only* dependency is torch, which is already installed,
so `--no-deps` skips nothing real. (ultralytics imports thop lazily behind a
`try/except`, for FLOPs profiling — detection does not need it either way.)

CLIP matters for a subtler reason. YOLO-World encodes its prompts through
CLIP inside `set_classes()`, and if the import fails **ultralytics shells out
to pip itself** to install it — no `--no-deps` — the first time a prompt is
set. CLIP declares torch and torchvision too. So a missing CLIP does not
produce a clean error: it produces PyPI torch landing on top of JetPack's
build in the middle of a benchmark run. `scripts/setup_jetson.sh` installs it
ahead of time precisely so that path is never taken, and the backend refuses
to load rather than letting ultralytics reach for pip.

`build_engines.sh` also pre-fetches CLIP's ViT-B/32 weights, which are
otherwise downloaded on first use — worth having on a bench with no route
out. `SKIP_YOLOWORLD=1` skips all of it in both scripts.

### The comparison video downloads but will not open (blank player, or QuickTime refuses it)

Nothing errored, `has_video` is true, the file has a sensible size — and the
player in the results page stays blank. VLC plays it fine, which makes it look
like a player problem rather than a file problem.

It isn't. The file is a valid `.mp4` whose *video stream* is in a codec no
browser can decode. OpenCV's `VideoWriter` defaults — and this project's
original `codec: mp4v` — write **MPEG-4 Part 2**, a 1998-era codec that
`ffprobe` and VLC read happily and that Chrome, Firefox, Safari and QuickTime
all refuse. Browsers only decode H.264 (`avc1`), H.265, VP8/VP9 and AV1 in a
`<video>` element. Since nothing in the chain treats "unplayable codec" as an
error, the failure is completely silent.

Check what a file actually contains:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name data/jobs/<id>/comparison.mp4
# h264  -> fine.   mpeg4 -> this bug.
```

**Fix:** install ffmpeg and re-run the job. The renderer prefers ffmpeg
(H.264, `yuv420p`, and `-movflags +faststart` so playback can start before
the whole file downloads) and only falls back to OpenCV when it is missing:

```bash
sudo apt install ffmpeg      # or: python3 scripts/doctor.py --fix
```

`doctor.py` checks for an H.264 encoder under **Comparison video encoder** and
reports this before you spend a benchmark run on it. If no encoder can be
found at all, the render still succeeds and the UI now says the file needs
VLC, rather than showing an empty player.

Note that pip's `opencv-python` wheels ship **without** an H.264 encoder for
licensing reasons, so `avc1` fails there even though most distro OpenCV builds
support it. That is why ffmpeg is the primary path rather than a fallback.

### "Numpy is not available" / "compiled using NumPy 1.x cannot be run in NumPy 2"

NumPy 2 was installed over JetPack's NumPy 1.x. torch still imports, but its
numpy bridge is dead, so `torch.from_numpy` fails — which is why this surfaces
inside NanoOWL's `_owl_compute_box_bias` rather than at import.

```bash
.venv/bin/pip install 'numpy<2'
```

If pip then complains that OpenCV requires NumPy 2, use JetPack's OpenCV instead
of the wheel (`sudo apt install python3-opencv`, plus `--system-site-packages`
on the venv), or pin `opencv-python-headless==4.10.0.84`, the last release that
resolves against NumPy 1.x.

### doctor reports five problems and four of them are the same one

A report like this is one failure, not five:

```
✗ torch.cuda.is_available() is False
✗ segment_anything not importable
✗ timm not importable
✗ efficientvit.models.efficientvit.sam import failed   ModuleNotFoundError: No module named 'torchvision'
! ultralytics not importable                           PackageNotFoundError: No package metadata was found for torchvision
```

`segment_anything`, `timm`, `efficientvit` and `ultralytics` are all installed
and all fine. Each one imports torchvision — `segment_anything` from
`automatic_mask_generator.py`, `timm` from its data loaders, `ultralytics` by
resolving its declared dependencies at import — so when torchvision is missing,
each fails *under its own name*. Reinstalling any of them fixes nothing.

The cause is one bad torch: a PyPI wheel landed on top of JetPack's build and
took torchvision with it. Confirm with the version string —

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.__file__)"
```

`2.13.0+cu130` on a driver that supports CUDA 12.6 is a PyPI wheel. JetPack
builds carry an `.nv` suffix. Repair the pair and every one of those reports
clears at once:

```bash
python3 scripts/doctor.py --fix        # installs a driver-matched torch + torchvision into the venv
```

doctor now labels these `(torchvision fallout)` in its summary and withholds
the misleading per-package install hints, so the list says which one to chase.

**How torch gets replaced in the first place:** some transitive dependency
declares an unconstrained `torch` and pip is free to satisfy it from PyPI —
`ultralytics-thop` is one, which is why `setup_jetson.sh` installs it with
`--no-deps` on a line of its own. The script also fingerprints torch before and
after each install section and shouts if it changed, so the next occurrence is
named while the responsible command is still on screen rather than surfacing
five sections later as this.

### "operator torchvision::nms does not exist"

torchvision's compiled extension was built against a different torch than the
one being imported. Check where each one lives:

```bash
python3 -c "import torch, torchvision; print(torch.__file__); print(torchvision.__file__)"
```

If one is in `.venv/lib/...` and the other in `/usr/local/lib/python3.10/dist-packages`,
that's the problem — a venv torchvision paired with the system JetPack torch.
Install them **together, from the same index, with `--force-reinstall`**:

```bash
.venv/bin/pip install --no-cache-dir --force-reinstall --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision
```

**`--force-reinstall` is required, not optional.** With `--system-site-packages`,
an unconstrained `torchvision` (no version pin) can already be satisfied by
whatever mismatched build is sitting in system `dist-packages` — so a plain
`pip install torch torchvision` silently installs only the one pip thinks is
missing (usually just torch) and leaves the old, mismatched torchvision
exactly where it was, reproducing this same error. `doctor.py --fix` learned
this the hard way and now always passes `--force-reinstall`. `--no-deps`
keeps the reinstall to just these two wheels — without it, `--force-reinstall`
would also re-fetch every transitive dependency (numpy, pillow, sympy, ...)
from this Jetson-only index, which likely doesn't host them.

Importing torchvision is not sufficient proof that it works — the failure above
happens at import, but a subtler mismatch can import cleanly and only fail when
an op is called. `scripts/doctor.py` actually invokes `nms` to confirm.

**If you just ran that command and nothing changed, the venv is losing.** pip
reports `Successfully installed torch-2.11.0`, and `import torch` goes on
resolving to `/usr/local/lib/python3.10/dist-packages`. A venv is not
guaranteed to win: with `--system-site-packages` the system `dist-packages`
directories are on `sys.path` too, and `PYTHONPATH` — which Jetson setup guides
hand out freely — precedes *every* site directory regardless of what is
installed where. This is worse than a no-op, because the half of the pair that
is **not** shadowed does take effect: a venv torchvision loaded against a system
torch is exactly the `nms` error above.

Check which copy actually wins:

```bash
.venv/bin/python -c "
import os, sys, sysconfig
print('PYTHONPATH =', os.environ.get('PYTHONPATH'))
print('venv purelib =', sysconfig.get_paths()['purelib'])
for i, p in enumerate(sys.path): print(f'  {i}: {p}')
"
```

If `PYTHONPATH` names the shadowing directory, `unset PYTHONPATH` (and drop it
from `~/.bashrc`). Otherwise the shadowing copy has to be removed outright —
it is a PyPI wheel, not JetPack's, and the venv already has the replacement:

```bash
sudo python3 -m pip uninstall -y torch torchvision
```

`doctor.py` reports this as *"the venv's torch is installed but not the one
being imported"*, names which mechanism it is, and offers the matching repair.

### "The NVIDIA driver on your system is too old (found version 12060)"

The torch being imported was built against a newer CUDA than the Jetson driver
provides — `12060` means the driver supports CUDA 12.6. It is almost always a
PyPI torch wheel that has replaced the JetPack build. Confirm with:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.__file__)"
```

JetPack wheels carry an `.nv` suffix (`2.5.0a0+872d972e41.nv24.08`). A bare
`2.6.0` or `2.6.0+cu128` is a PyPI wheel and is the problem. Reinstall the
matched pair for your JetPack — for JetPack 6.x / CUDA 12.6:

```bash
pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision
```

`torch` and `torchvision` must come from the same index; a mismatched pair fails
at import. Then reinstall the model repos with `--no-deps` so pip cannot replace
torch again.

### "TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'"

An install — editable, or straight from git — dies inside setuptools itself,
with a traceback that never mentions the package being installed:

```
File ".../setuptools/_core_metadata.py", line 293, in _distribution_fullname
  canonicalize_version(version, strip_trailing_zero=False),
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

Nothing is wrong with the package. JetPack's Ubuntu 22.04 ships **packaging
21.3** in `/usr/lib/python3/dist-packages`, and setuptools (≥71) prefers an
installed `packaging` over its own vendored copy — so it calls a keyword that
only exists from **packaging 22.0** onward.

The confusing part is that it only bites *some* installs. setuptools reaches
that call from `prune_file_list()`, which runs only when the project has a
`MANIFEST.in`. `git+https://github.com/ultralytics/CLIP.git` has one and fails
every time; `git+.../segment-anything.git` has none and installs cleanly right
next to it, in the same run.

Fix it by shadowing the system copy inside the venv (no sudo, nothing JetPack
owns is touched):

```bash
.venv/bin/pip install "packaging>=24.2"
```

`setup_jetson.sh` now does this before any install, checking the actual
signature rather than a version string.

If it persists, pin setuptools back below the change instead:

```bash
.venv/bin/pip install -U "setuptools<69.3"
.venv/bin/pip install -e efficientvit --no-build-isolation
```

**Or skip the install entirely.** None of the three repos need to be installed —
point `repo_paths` in `config.yaml` at the clones and they are imported straight
from source:

```yaml
repo_paths:
  - ../nanoowl
  - ../nanosam
  - ../efficientvit
```

Each path is the repo root, i.e. the directory containing the package folder.
The UI badge names any module it still cannot import, so a half-finished install
is diagnosable at a glance rather than silently falling back to mock.

<br>

## Run

**Pin the clocks first.** Unpinned clocks are the most common cause of
meaningless Jetson benchmarks:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

```bash
./scripts/run.sh
```

Open `http://<orin-ip>:8000` from any browser on the network. The power mode is
captured into every `results.json` next to the numbers, so a run done on the
wrong profile is identifiable after the fact.

<br>

## Using it

1. **Upload & configure** — drop a video, then set the NanoOWL text prompts.
   NanoOWL is open-vocabulary, so prompts are required; they drive both
   pairings identically. Threshold, frame stride, max frames and warm-up count
   are all per-job.
2. **Processing** — a stepper tracks Upload → Validate → Pairing A → Pairing B →
   Aggregate → Video, with live frame counts, instantaneous FPS and a rolling
   latency sparkline.
3. **Results** — head-to-head stat cards with deltas, a per-stage breakdown,
   latency over time, the distribution with p50/p95 markers, and a full
   percentile table. Downloadable as `results.json` or per-frame CSV.
4. **Comparison video** — rendered *after* the measurements are already final.
   If the render fails the job reports `complete_no_video`; the benchmark data
   is unaffected.

<br>

## Development off the Jetson

The Jetson libraries can't be imported on a normal x86 box, so the app ships
CPU mock backends with configurable synthetic latency. The entire application —
upload, queue, WebSocket progress, aggregation, charts, video render, exports —
runs and is testable without a GPU.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
BENCH_BACKEND=mock .venv/bin/python -m uvicorn benchmark.server:app --port 8000
.venv/bin/python -m pytest tests/ -q
```

Backend selection is `auto` by default: real backends when importable, mock
otherwise. Set `backend: jetson` in `config.yaml` to refuse the fallback, so a
misconfigured Orin fails loudly instead of quietly producing synthetic numbers.

Whenever mocks are active the UI shows a prominent **MOCK BACKEND** badge, the
results carry `is_mock: true`, and a warning sits above the verdict. Mock
numbers cannot be mistaken for Orin measurements.

<br>

## Output layout

```
data/jobs/<job_id>/
  job.json          state machine + run config
  input.mp4         uploaded video
  frames_a.jsonl    per-frame records incl. timings and detections
  frames_b.jsonl
  masks_a.jsonl     RLE masks (only if record_masks)
  masks_b.jsonl
  results.json      aggregates, comparison, environment snapshot
  comparison.mp4    side-by-side render
```

Everything is plain files: a restart loses nothing, and a finished run can be
copied off the Orin with `scp`.

<br>

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Upload video + run config → `job_id` |
| `GET` | `/api/jobs` · `/api/jobs/{id}` | List / detail with live state |
| `GET` | `/api/jobs/{id}/results` | Aggregates and comparison |
| `GET` | `/api/jobs/{id}/frames?pairing=a` | Per-frame records |
| `GET` | `/api/jobs/{id}/export.csv` | Flat per-frame CSV |
| `GET` | `/api/jobs/{id}/video` | Comparison mp4 (range requests supported) |
| `POST` | `/api/jobs/{id}/cancel` | Cooperative cancel |
| `WS` | `/ws/jobs/{id}` | Progress stream; replays state on connect |

<br>

## Scope

Measured: latency distribution and per-stage breakdown, with detection counts
for context.

Not measured: power/energy, GPU utilisation, memory, and automated mask-quality
metrics (IoU agreement, temporal stability, ground truth). Mask quality is
assessed by eye from the comparison video. The data model and stage timer leave
room for these, but nothing is built for them.
