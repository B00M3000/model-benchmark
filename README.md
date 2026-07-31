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

# Neither of these gets --no-deps: see below.
.venv/bin/pip install transformers
.venv/bin/pip install onnx
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

**`onnx` is the other exception, needed only for *building* engines, not for
running a benchmark.** Both NanoOWL's `build_image_encoder_engine()` and
NanoSAM's `export_sam_mask_decoder_onnx.py` call `torch.onnx.export()` to
produce the `.onnx` file `trtexec` then compiles, and PyTorch's ONNX exporter
needs the `onnx` package itself to serialize the result. Neither repo declares
it, so it's easy to burn several minutes tracing the NanoOWL model only to
fail on the last line with `torch.onnx.OnnxExporterError: Module onnx is not
installed!`. Same reasoning as `transformers`: `onnx` doesn't depend on
`torch`, and its `numpy>=1.23.2` is already satisfied, so a plain
`pip install onnx` is safe and complete.

Check the environment at any point:

```bash
.venv/bin/python scripts/doctor.py
```

It verifies torch matches the driver's CUDA version and can convert NumPy
arrays, that TensorRT and `trtexec` are present, that all four packages import,
that torch2trt's `TRTModule` is new enough for the installed TensorRT, which
weights and engines exist, and whether the clocks are pinned — each with the fix.

Build the TensorRT engines and fetch weights (once, on the Orin — engines are
tied to the exact TensorRT version and GPU that built them and cannot be copied
between machines):

```bash
./scripts/build_engines.sh
```

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

### If `pip install -e` fails

JetPack images frequently ship a setuptools newer than their `packaging`, and
editable installs then die with:

```
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

`strip_trailing_zero` landed in packaging 23.2, so the fix is to align the pair
inside the venv (which shadows the system copies without touching JetPack):

```bash
.venv/bin/pip install -U pip setuptools wheel "packaging>=23.2"
```

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
