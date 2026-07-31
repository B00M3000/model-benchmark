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
.venv/bin/pip install    torch2trt    --no-deps
```

**`torch2trt` is easy to miss.** It is not on PyPI and neither NanoOWL nor
NanoSAM declares it, so nothing installs it as a side effect — but both do
`from torch2trt import TRTModule` to execute their TensorRT engines. Without it
the NanoOWL engine build runs to completion and then fails on its last line,
loading the finished engine back (`ModuleNotFoundError: No module named
'torch2trt'`). The engine it just built is fine; only the verification step
failed.

**Always pass `--no-deps`.** All four declare `torch` as a dependency (torch2trt
also declares `tensorrt`), and pip will happily pull a PyPI wheel over JetPack's
build, which then fails at runtime with *"The NVIDIA driver on your system is
too old"*. The cost is that their other requirements (`transformers`, `timm`, …)
aren't installed automatically — add those individually as the import errors
name them. That trade is worth making: a broken torch is far more painful to
unpick than a missing pure-Python package.

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
.venv/bin/pip install ./torch2trt --no-deps
```

`--no-deps` matters here too — torch2trt declares `tensorrt`, and pip would
install the PyPI wheel over JetPack's.

If this appeared at the end of `build_engines.sh`, **the engine built
successfully**; the failure is in the step that loads it back to verify.
Installing torch2trt and re-running skips straight past it — the script sees the
existing engine file and moves on.

On TensorRT 10 (JetPack 6), install torch2trt from master rather than a release:
older releases drive engines through the removed binding API and fail at
inference with `no attribute 'num_bindings'`. The doctor flags this.

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
Install them **together, from the same index**, never one alone:

```bash
.venv/bin/pip uninstall -y torch torchvision
.venv/bin/pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision
```

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
