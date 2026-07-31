"""Job state machine, queue, and the single-process run orchestrator.

One worker thread drains the queue and a global lock guarantees that only
one benchmark executes at a time -- concurrent GPU work would contaminate
every latency number this tool produces.

Both pairings run inside this one process. The CUDA context, TensorRT
runtime and PyTorch allocator initialise once per job rather than once per
pairing, and NanoOWL (identical in both pairings) stays resident across the
swap. Only the segmentation head is torn down and rebuilt between runs.
"""

from __future__ import annotations

import asyncio
import platform
import queue
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .metrics import PairingSummary, compare, histogram
from .models.base import BackendUnavailable, Detector, Segmenter
from .models.registry import (
    PAIRING_A,
    PAIRING_B,
    PAIRINGS,
    build_detector,
    build_segmenter,
    resolve_backend,
)
from .pipeline import CancelledError, RunConfig, VideoInfo, probe_video, run_pairing
from .storage import JobPaths, load_masks, read_json, read_jsonl, write_json
from .video import COLOR_A, COLOR_B, PanelData, render_comparison

# --- states -----------------------------------------------------------------
QUEUED = "queued"
VALIDATING = "validating"
RUNNING_A = "running_a"
RUNNING_B = "running_b"
AGGREGATING = "aggregating"
RENDERING_VIDEO = "rendering_video"
COMPLETE = "complete"
COMPLETE_NO_VIDEO = "complete_no_video"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL_STATES = {COMPLETE, COMPLETE_NO_VIDEO, FAILED, CANCELLED}

#: Stages shown in the UI stepper, in order.
STAGE_SEQUENCE = (
    VALIDATING,
    RUNNING_A,
    RUNNING_B,
    AGGREGATING,
    RENDERING_VIDEO,
)


@dataclass
class Job:
    job_id: str
    filename: str
    state: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    backend: str = "unknown"
    run_config: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    queue_position: int | None = None
    #: Last non-terminal stage entered, so a failed job can show the UI
    #: exactly where it broke rather than just "failed".
    last_stage: str | None = None
    #: Per-pairing lifecycle notes (load and swap costs).
    lifecycle: dict[str, Any] = field(default_factory=dict)
    has_results: bool = False
    has_video: bool = False
    video_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "backend": self.backend,
            "run_config": self.run_config,
            "video": self.video,
            "environment": self.environment,
            "progress": self.progress,
            "queue_position": self.queue_position,
            "last_stage": self.last_stage,
            "lifecycle": self.lifecycle,
            "has_results": self.has_results,
            "has_video": self.has_video,
            "video_error": self.video_error,
            "stage_sequence": list(STAGE_SEQUENCE),
            "is_terminal": self.state in TERMINAL_STATES,
        }


class EventHub:
    """Fan-out from the worker thread to WebSocket subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._last: dict[str, dict[str, Any]] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers.setdefault(job_id, set()).add(q)
            # Replay the latest event so a reconnecting browser immediately
            # shows current state instead of waiting for the next frame.
            last = self._last.get(job_id)
        if last:
            q.put_nowait(last)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs:
                subs.discard(q)
                if not subs:
                    self._subscribers.pop(job_id, None)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Thread-safe publish from the worker thread."""
        with self._lock:
            self._last[job_id] = event
            targets = list(self._subscribers.get(job_id, ()))
            loop = self._loop
        if not targets or loop is None:
            return

        def _deliver() -> None:
            for q in targets:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # A stalled browser must never block the benchmark.
                    pass

        try:
            loop.call_soon_threadsafe(_deliver)
        except RuntimeError:
            pass


def environment_snapshot(backend: str) -> dict[str, Any]:
    """Provenance captured once per job.

    Unpinned clocks are the most common cause of meaningless Jetson
    benchmarks, so the power mode is recorded next to the numbers.
    """
    env: dict[str, Any] = {
        "backend": backend,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    def _run(cmd: list[str]) -> str | None:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return out.stdout.strip() or None
        except Exception:
            return None

    l4t = Path("/etc/nv_tegra_release")
    if l4t.exists():
        try:
            env["l4t"] = l4t.read_text().strip().splitlines()[0]
        except OSError:
            pass

    if nvp := _run(["nvpmodel", "-q"]):
        env["nvpmodel"] = " / ".join(line.strip() for line in nvp.splitlines()[:2])

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch"] = None

    try:
        import tensorrt

        env["tensorrt"] = tensorrt.__version__
    except Exception:
        env["tensorrt"] = None

    try:
        import cv2

        env["opencv"] = cv2.__version__
    except Exception:
        pass

    return env


class JobManager:
    """Owns the queue, the worker thread, and job persistence."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.data_dir = config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventHub()

        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._cancels: dict[str, threading.Event] = {}
        self._current: str | None = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="bench-worker")
        self._started = False

        self._restore()

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not self._started:
            self._worker.start()
            self._started = True

    def _restore(self) -> None:
        """Reload jobs from disk so a restart does not lose finished runs."""
        for job_dir in sorted(self.data_dir.glob("*/"), key=lambda p: p.stat().st_mtime):
            payload = read_json(job_dir / "job.json")
            if not payload:
                continue
            job = Job(
                job_id=payload.get("job_id", job_dir.name),
                filename=payload.get("filename", "input"),
            )
            for key, value in payload.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            # Anything mid-flight when the process died is not recoverable.
            if job.state not in TERMINAL_STATES:
                job.state = FAILED
                job.error = job.error or "Server restarted while this job was running."
            paths = JobPaths(self.data_dir, job.job_id)
            job.has_results = paths.results_json.exists()
            job.has_video = paths.video.exists()
            with self._lock:
                self._jobs[job.job_id] = job
                self._order.append(job.job_id)

    # --- accessors ---------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order) if j in self._jobs]

    def paths(self, job_id: str) -> JobPaths:
        return JobPaths(self.data_dir, job_id)

    def results(self, job_id: str) -> dict[str, Any] | None:
        return read_json(self.paths(job_id).results_json)

    def frames(self, job_id: str, pairing_id: str) -> list[dict[str, Any]]:
        return list(read_jsonl(self.paths(job_id).frames(pairing_id)))

    # --- job creation ------------------------------------------------------
    def create(self, filename: str, suffix: str, run: RunConfig) -> tuple[Job, Path]:
        job_id = uuid.uuid4().hex[:12]
        paths = JobPaths(self.data_dir, job_id).ensure()
        job = Job(
            job_id=job_id,
            filename=filename,
            run_config=run.to_dict(),
            backend=resolve_backend(self.config),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._cancels[job_id] = threading.Event()
        self._persist(job)
        return job, paths.dir / f"input{suffix}"

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)
        self._refresh_queue_positions()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            event = self._cancels.get(job_id)
        if not job or job.state in TERMINAL_STATES:
            return False
        if event:
            event.set()
        if job.state == QUEUED:
            self._transition(job, CANCELLED, error="Cancelled before it started.")
            job.finished_at = time.time()
            self._persist(job)
        return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.job_id == self._current:
                return False
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
            self._cancels.pop(job_id, None)
        JobPaths(self.data_dir, job_id).delete()
        return True

    # --- state helpers -----------------------------------------------------
    def _persist(self, job: Job) -> None:
        write_json(JobPaths(self.data_dir, job.job_id).job_json, job.to_dict())

    def _emit(self, job: Job, extra: dict[str, Any] | None = None) -> None:
        event = {"type": "job", "job": job.to_dict()}
        if extra:
            event.update(extra)
        self.events.publish(job.job_id, event)

    def _transition(self, job: Job, state: str, error: str | None = None) -> None:
        if state in STAGE_SEQUENCE:
            job.last_stage = state
        job.state = state
        if error:
            job.error = error
        job.queue_position = None
        self._persist(job)
        self._emit(job)

    def _refresh_queue_positions(self) -> None:
        with self._lock:
            pending = [jid for jid in self._order if self._jobs.get(jid, Job("", "")).state == QUEUED]
        for index, job_id in enumerate(pending):
            job = self.get(job_id)
            if job:
                job.queue_position = index + 1
                self._emit(job)

    # --- worker ------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            try:
                if job is None or job.state != QUEUED:
                    continue
                with self._lock:
                    self._current = job_id
                self._execute(job)
            except Exception:  # pragma: no cover - defensive
                if job is not None:
                    job.finished_at = time.time()
                    self._transition(job, FAILED, error=traceback.format_exc(limit=3))
            finally:
                with self._lock:
                    self._current = None
                self._queue.task_done()
                self._refresh_queue_positions()

    def _execute(self, job: Job) -> None:
        cancel = self._cancels.get(job.job_id) or threading.Event()
        paths = JobPaths(self.data_dir, job.job_id)
        job.started_at = time.time()

        run = RunConfig(**job.run_config)
        backend = resolve_backend(self.config)
        job.backend = backend
        job.environment = environment_snapshot(backend)

        # --- validate ------------------------------------------------------
        self._transition(job, VALIDATING)
        video_path = paths.input_video()
        if video_path is None or not video_path.exists():
            job.finished_at = time.time()
            self._transition(job, FAILED, error="Uploaded video is missing on disk.")
            return
        try:
            info = probe_video(video_path)
        except Exception as exc:
            job.finished_at = time.time()
            self._transition(job, FAILED, error=f"Could not read the video: {exc}")
            return
        if info.frame_count <= 0:
            job.finished_at = time.time()
            self._transition(job, FAILED, error="The video contains no decodable frames.")
            return
        job.video = info.to_dict()
        self._persist(job)
        self._emit(job)

        detector: Detector | None = None
        segmenters: dict[str, Segmenter] = {}
        summaries: dict[str, PairingSummary] = {}
        all_frames: dict[str, list[dict[str, Any]]] = {}
        lifecycle = self.config.lifecycle

        try:
            # --- detector: loaded once, reused across both pairings --------
            detector = build_detector(self.config, backend)
            detector.load()
            text_encode_ms = detector.set_prompts(run.prompts, run.threshold)
            job.lifecycle["detector_load_ms"] = round(detector.load_ms, 2)
            job.lifecycle["text_encode_ms"] = round(text_encode_ms, 2)
            job.lifecycle["share_detector"] = lifecycle.share_detector

            if lifecycle.preload_all_models:
                for pairing in PAIRINGS:
                    seg = build_segmenter(self.config, backend, pairing.segmenter)
                    seg.load()
                    segmenters[pairing.pairing_id] = seg
                job.lifecycle["preloaded"] = True

            swap_start: float | None = None
            for pairing in PAIRINGS:
                if cancel.is_set():
                    raise CancelledError("cancelled")

                state = RUNNING_A if pairing.pairing_id == PAIRING_A else RUNNING_B
                self._transition(job, state)

                if lifecycle.reload_between_runs and detector is not None and summaries:
                    # Strict isolation: rebuild everything between runs.
                    detector.unload()
                    detector = build_detector(self.config, backend)
                    detector.load()
                    text_encode_ms = detector.set_prompts(run.prompts, run.threshold)

                segmenter = segmenters.get(pairing.pairing_id)
                if segmenter is None:
                    segmenter = build_segmenter(self.config, backend, pairing.segmenter)
                    segmenter.load()
                    segmenters[pairing.pairing_id] = segmenter
                if swap_start is not None:
                    job.lifecycle["swap_ms"] = round((time.perf_counter() - swap_start) * 1000, 2)

                summary, records = run_pairing(
                    video_path,
                    pairing,
                    detector,
                    segmenter,
                    run,
                    frames_path=paths.frames(pairing.pairing_id),
                    masks_path=paths.masks(pairing.pairing_id),
                    text_encode_ms=text_encode_ms,
                    progress=self._make_progress_callback(job),
                    cancel=cancel,
                    video_info=info,
                )
                summaries[pairing.pairing_id] = summary
                all_frames[pairing.pairing_id] = records

                # Free the segmentation head before the next one loads, so
                # pairing B never competes with pairing A's memory.
                if not lifecycle.preload_all_models:
                    segmenter.unload()
                    segmenters.pop(pairing.pairing_id, None)
                swap_start = time.perf_counter()

            # --- aggregate -----------------------------------------------
            self._transition(job, AGGREGATING)
            results = self._build_results(job, info, run, summaries, all_frames)
            write_json(paths.results_json, results)
            job.has_results = True
            self._persist(job)
            self._emit(job)

        except CancelledError:
            job.finished_at = time.time()
            self._transition(job, CANCELLED, error="Cancelled.")
            return
        except BackendUnavailable as exc:
            job.finished_at = time.time()
            self._transition(job, FAILED, error=str(exc))
            return
        except Exception:
            job.finished_at = time.time()
            self._transition(job, FAILED, error=traceback.format_exc(limit=4))
            return
        finally:
            for seg in segmenters.values():
                try:
                    seg.unload()
                except Exception:
                    pass
            if detector is not None:
                try:
                    detector.unload()
                except Exception:
                    pass

        # --- video (non-critical) -----------------------------------------
        # Results are already on disk and visible in the UI; a failure here
        # downgrades the job, it does not invalidate it.
        if not self.config.video.enabled:
            job.finished_at = time.time()
            self._transition(job, COMPLETE_NO_VIDEO)
            return

        self._transition(job, RENDERING_VIDEO)
        try:
            self._render_video(job, paths, info, run, summaries, cancel)
            job.has_video = True
            job.finished_at = time.time()
            self._transition(job, COMPLETE)
        except Exception as exc:
            job.video_error = f"{type(exc).__name__}: {exc}"
            job.finished_at = time.time()
            self._transition(job, COMPLETE_NO_VIDEO)

    def _make_progress_callback(self, job: Job) -> Callable[[Any], None]:
        """Throttled progress emitter -- ~15 updates/sec is plenty for a UI."""
        state = {"last": 0.0}
        min_interval = 1.0 / 15

        def callback(progress: Any) -> None:
            job.progress = {
                "pairing_id": progress.pairing_id,
                "frame_index": progress.frame_index,
                "processed": progress.processed,
                "total": progress.total,
                "pct": (progress.processed / progress.total * 100.0) if progress.total else 0.0,
                "pipeline_ms": round(progress.pipeline_ms, 2),
                "fps_instant": round(progress.fps_instant, 2),
                "warmup": progress.warmup,
                "num_detections": progress.num_detections,
            }
            now = time.monotonic()
            is_last = progress.total and progress.processed >= progress.total
            if now - state["last"] < min_interval and not is_last:
                return
            state["last"] = now
            self.events.publish(
                job.job_id, {"type": "progress", "job_id": job.job_id, "progress": job.progress}
            )

        return callback

    def _build_results(
        self,
        job: Job,
        info: VideoInfo,
        run: RunConfig,
        summaries: dict[str, PairingSummary],
        all_frames: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        summary_a = summaries[PAIRING_A]
        summary_b = summaries[PAIRING_B]

        def measured(pairing_id: str) -> list[float]:
            return [
                f["pipeline_ms"] for f in all_frames[pairing_id] if not f.get("warmup", False)
            ]

        return {
            "job_id": job.job_id,
            "generated_at": time.time(),
            "backend": job.backend,
            "is_mock": job.backend == "mock",
            "video": info.to_dict(),
            "run_config": run.to_dict(),
            "environment": job.environment,
            "lifecycle": job.lifecycle,
            "pairings": {
                PAIRING_A: summary_a.to_dict(),
                PAIRING_B: summary_b.to_dict(),
            },
            "comparison": compare(summary_a, summary_b),
            "histograms": {
                PAIRING_A: histogram(measured(PAIRING_A)),
                PAIRING_B: histogram(measured(PAIRING_B)),
            },
            "series": {
                pairing_id: [
                    {
                        "seq": f["seq"],
                        "pipeline_ms": f["pipeline_ms"],
                        "warmup": f.get("warmup", False),
                        "num_detections": f.get("num_detections", 0),
                    }
                    for f in frames
                ]
                for pairing_id, frames in all_frames.items()
            },
        }

    def _render_video(
        self,
        job: Job,
        paths: JobPaths,
        info: VideoInfo,
        run: RunConfig,
        summaries: dict[str, PairingSummary],
        cancel: threading.Event,
    ) -> None:
        panels = []
        for pairing_id, color in ((PAIRING_A, COLOR_A), (PAIRING_B, COLOR_B)):
            summary = summaries[pairing_id]
            frames = {
                int(f["frame_index"]): f for f in read_jsonl(paths.frames(pairing_id))
            }
            panels.append(
                PanelData(
                    label=summary.label,
                    model=summary.segmenter,
                    color=color,
                    masks=load_masks(paths.masks(pairing_id)),
                    frames=frames,
                )
            )

        def progress(done: int, total: int) -> None:
            job.progress = {
                "pairing_id": "video",
                "processed": done,
                "total": total,
                "pct": (done / total * 100.0) if total else 0.0,
            }
            self.events.publish(
                job.job_id, {"type": "progress", "job_id": job.job_id, "progress": job.progress}
            )

        render_comparison(
            paths.input_video(),
            paths.video,
            panels[0],
            panels[1],
            self.config.video,
            stride=run.stride,
            source_fps=info.fps,
            progress=progress,
            cancel=cancel,
        )
