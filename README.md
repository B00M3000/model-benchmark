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
.venv/bin/pip install -r requirements.txt
```

`--system-site-packages` matters: the Jetson's `torch`, `torchvision` and
`tensorrt` come from JetPack and must not be replaced by pip wheels.

Then install the three model repos against that same environment:

```bash
# NanoOWL
git clone https://github.com/NVIDIA-AI-IOT/nanoowl && .venv/bin/pip install -e nanoowl

# NanoSAM
git clone https://github.com/NVIDIA-AI-IOT/nanosam && .venv/bin/pip install -e nanosam

# EfficientViT
git clone https://github.com/mit-han-lab/efficientvit && .venv/bin/pip install -e efficientvit
```

Build the TensorRT engines and fetch weights (once, on the Orin — engines are
tied to the exact TensorRT version and GPU that built them and cannot be copied
between machines):

```bash
./scripts/build_engines.sh
```

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
