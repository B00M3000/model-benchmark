"""The measurement loop: run one pairing over one video.

Called once per pairing, sequentially, inside the same process. Nothing here
knows about HTTP or jobs -- it takes loaded backends and a video, and
produces per-frame records.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .metrics import PairingSummary, summarize_pairing
from .models.base import Detector, Segmenter
from .models.registry import Pairing
from .storage import MaskWriter, append_jsonl
from .timing import STAGE_DECODE, STAGE_POSTPROCESS, STAGE_PREPROCESS, StageTimer, torch_cuda_sync

ProgressCallback = Callable[["FrameProgress"], None]


@dataclass
class FrameProgress:
    pairing_id: str
    frame_index: int
    processed: int
    total: int
    pipeline_ms: float
    fps_instant: float
    warmup: bool
    num_detections: int


@dataclass
class RunConfig:
    """Per-job run parameters, all overridable from the UI."""

    prompts: list[str] = field(default_factory=lambda: ["a person"])
    threshold: float = 0.1
    stride: int = 1
    max_frames: int | None = None
    warmup_frames: int = 10
    record_masks: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompts": self.prompts,
            "threshold": self.threshold,
            "stride": self.stride,
            "max_frames": self.max_frames,
            "warmup_frames": self.warmup_frames,
            "record_masks": self.record_masks,
        }


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": Path(self.path).name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 2),
        }


class CancelledError(RuntimeError):
    """Raised when a run is cancelled from the UI."""


def probe_video(path: str | Path) -> VideoInfo:
    """Read basic properties without decoding the whole file."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            # Some containers do not report a frame count; fall back to a
            # decode-and-count pass so progress reporting stays honest.
            count = 0
            while capture.grab():
                count += 1
        return VideoInfo(
            path=str(path),
            width=width,
            height=height,
            fps=fps,
            frame_count=count,
            duration_s=count / fps if fps else 0.0,
        )
    finally:
        capture.release()


def planned_frame_count(info: VideoInfo, run: RunConfig) -> int:
    """How many frames a run will actually process, after stride and cap."""
    if info.frame_count <= 0:
        return 0
    total = (info.frame_count + run.stride - 1) // run.stride
    if run.max_frames is not None:
        total = min(total, run.max_frames)
    return total


def run_pairing(
    video_path: str | Path,
    pairing: Pairing,
    detector: Detector,
    segmenter: Segmenter,
    run: RunConfig,
    *,
    frames_path: Path | None = None,
    masks_path: Path | None = None,
    text_encode_ms: float = 0.0,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    video_info: VideoInfo | None = None,
) -> tuple[PairingSummary, list[dict[str, Any]]]:
    """Run one pairing end to end and return its summary plus frame records.

    Both backends must already be loaded -- loading is the caller's job so
    that a shared detector can survive across both pairings.
    """
    info = video_info or probe_video(video_path)
    total_planned = planned_frame_count(info, run)
    sync = torch_cuda_sync()

    # A shared detector carries state from the previous pairing; clear it so
    # run B sees exactly what run A saw.
    detector.reset()
    segmenter.reset()

    mask_writer: MaskWriter | None = None
    if run.record_masks and masks_path is not None:
        mask_writer = MaskWriter(masks_path).start()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    records: list[dict[str, Any]] = []
    processed = 0
    source_index = -1
    wall_start = time.perf_counter()

    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise CancelledError(f"pairing {pairing.pairing_id} cancelled")
            if run.max_frames is not None and processed >= run.max_frames:
                break

            timer = StageTimer(sync)

            # --- decode -------------------------------------------------
            # Timed but excluded from pipeline_ms: a live camera feed has no
            # container to demux.
            with timer.stage(STAGE_DECODE, sync=False):
                ok, frame_bgr = capture.read()
            if not ok:
                break
            source_index += 1
            if source_index % run.stride != 0:
                continue

            # --- preprocess ---------------------------------------------
            with timer.stage(STAGE_PREPROCESS, sync=False):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # --- detect ---------------------------------------------------
            detections = detector.detect(frame_rgb, timer)

            # --- segment --------------------------------------------------
            # A real deployment does not run the image encoder when there is
            # nothing to segment, so neither do we. frames_with_no_detections
            # is reported alongside so the averages stay interpretable.
            masks: list[np.ndarray] = []
            mask_scores: list[float] = []
            if detections:
                segmenter.set_image(frame_rgb, timer)
                for detection in detections:
                    result = segmenter.segment_box(detection.box, timer)
                    masks.append(result.mask)
                    mask_scores.append(result.score)

            # --- postprocess ---------------------------------------------
            with timer.stage(STAGE_POSTPROCESS, sync=False):
                mask_areas = [int(np.count_nonzero(m)) for m in masks]

            warmup = processed < run.warmup_frames
            record = {
                "frame_index": source_index,
                "seq": processed,
                "warmup": warmup,
                "pipeline_ms": round(timer.pipeline_ms(), 4),
                "e2e_ms": round(timer.e2e_ms(), 4),
                "stages": {k: round(v, 4) for k, v in timer.marks.items()},
                "num_detections": len(detections),
                "detections": [d.to_dict() for d in detections],
                "mask_scores": [round(s, 4) for s in mask_scores],
                "mask_areas": mask_areas,
            }
            records.append(record)
            if frames_path is not None:
                append_jsonl(frames_path, record)

            # Mask hand-off happens strictly after the timed region.
            if mask_writer is not None and masks:
                mask_writer.submit(
                    source_index,
                    masks,
                    [list(d.box) for d in detections],
                )

            processed += 1
            if progress is not None:
                pipeline_ms = record["pipeline_ms"]
                progress(
                    FrameProgress(
                        pairing_id=pairing.pairing_id,
                        frame_index=source_index,
                        processed=processed,
                        total=total_planned,
                        pipeline_ms=pipeline_ms,
                        fps_instant=(1000.0 / pipeline_ms) if pipeline_ms > 0 else 0.0,
                        warmup=warmup,
                        num_detections=len(detections),
                    )
                )
    finally:
        capture.release()
        if mask_writer is not None:
            mask_writer.close()

    wall_clock_s = time.perf_counter() - wall_start
    summary = summarize_pairing(
        pairing_id=pairing.pairing_id,
        label=pairing.label,
        detector=detector.variant or detector.name,
        segmenter=segmenter.variant or segmenter.name,
        frames=records,
        wall_clock_s=wall_clock_s,
        model_load_ms=detector.load_ms + segmenter.load_ms,
        text_encode_ms=text_encode_ms,
    )
    return summary, records
