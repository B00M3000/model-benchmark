"""CPU stand-ins for the Jetson backends.

These exist so the entire application -- upload, queue, WebSocket progress,
aggregation, charts, video render, exports -- can be developed and tested on
a machine with no GPU and none of the Jetson libraries installed.

The latencies they report are synthetic. Anything produced with these
backends is tagged ``mock`` end to end and the UI shows a prominent badge,
so mock numbers can never be mistaken for Orin measurements.
"""

from __future__ import annotations

import hashlib
import time
from typing import Sequence

import numpy as np

from ..timing import (
    STAGE_DETECT,
    STAGE_DETECT_DECODE,
    STAGE_DETECT_ENCODE,
    STAGE_SEG_DECODE,
    STAGE_SEG_ENCODE,
    StageTimer,
)
from .base import Detection, Detector, MaskResult, Segmenter


def _busy_sleep(ms: float) -> None:
    """Burn wall-clock time so the synthetic stage cost is really observable."""
    if ms <= 0:
        return
    deadline = time.perf_counter_ns() + int(ms * 1e6)
    while time.perf_counter_ns() < deadline:
        pass


class MockDetector(Detector):
    """Deterministic pseudo-detections that drift smoothly across frames."""

    name = "mock_detector"

    def __init__(
        self,
        encode_ms: float = 8.0,
        decode_ms: float = 2.0,
        jitter_ms: float = 1.5,
        max_objects: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.encode_ms = encode_ms
        self.decode_ms = decode_ms
        self.jitter_ms = jitter_ms
        self.max_objects = max_objects
        self.variant = "mock detector (synthetic latency)"
        self._prompts: list[str] = []
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._frame_index = 0

    def _load(self) -> None:
        _busy_sleep(50.0)

    def reset(self) -> None:
        # Both pairings must see identical pseudo-detections on identical
        # frames, exactly as a real detector would.
        self._frame_index = 0
        self._rng = np.random.default_rng(self._seed)

    def set_prompts(self, prompts: Sequence[str], threshold: float) -> float:
        start = time.perf_counter_ns()
        self._prompts = list(prompts) or ["object"]
        _busy_sleep(5.0)
        return (time.perf_counter_ns() - start) / 1e6

    def detect(self, frame_rgb: np.ndarray, timer: StageTimer) -> list[Detection]:
        height, width = frame_rgb.shape[:2]
        with timer.stage(STAGE_DETECT, sync=False):
            with timer.stage(STAGE_DETECT_ENCODE, sync=False):
                _busy_sleep(self.encode_ms + float(self._rng.normal(0, self.jitter_ms)))
            with timer.stage(STAGE_DETECT_DECODE, sync=False):
                _busy_sleep(self.decode_ms + float(self._rng.normal(0, self.jitter_ms / 3)))

        # Deterministic in the frame index so both pairings see comparable
        # object counts, with a couple of frames left empty on purpose to
        # exercise the no-detection path.
        t = self._frame_index
        self._frame_index += 1
        count = 0 if t % 17 == 0 else 1 + (t // 3) % self.max_objects

        detections: list[Detection] = []
        for i in range(count):
            phase = t * 0.05 + i * 2.0
            cx = width * (0.5 + 0.28 * np.sin(phase))
            cy = height * (0.5 + 0.20 * np.cos(phase * 0.7))
            bw, bh = width * 0.18, height * 0.24
            label = self._prompts[i % len(self._prompts)] if self._prompts else "object"
            detections.append(
                Detection(
                    box=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                    score=float(0.55 + 0.4 * abs(np.sin(phase * 1.3))),
                    label=label,
                    label_index=i % max(1, len(self._prompts)),
                )
            )
        return detections


class MockSegmenter(Segmenter):
    """Ellipse masks with configurable synthetic encode/decode cost."""

    name = "mock_segmenter"

    def __init__(
        self,
        label: str = "mock segmenter",
        encode_ms: float = 12.0,
        decode_ms: float = 4.0,
        jitter_ms: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.encode_ms = encode_ms
        self.decode_ms = decode_ms
        self.jitter_ms = jitter_ms
        self.variant = f"{label} (synthetic latency)"
        self._shape: tuple[int, int] = (0, 0)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Stable per-instance wobble so the two mock segmenters produce
        # visibly different masks in the comparison video.
        self._wobble = int(hashlib.sha1(label.encode()).hexdigest()[:4], 16) % 13

    def _load(self) -> None:
        _busy_sleep(80.0)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def set_image(self, frame_rgb: np.ndarray, timer: StageTimer) -> None:
        self._shape = frame_rgb.shape[:2]
        with timer.stage(STAGE_SEG_ENCODE, sync=False):
            _busy_sleep(self.encode_ms + float(self._rng.normal(0, self.jitter_ms)))

    def segment_box(
        self, box: tuple[float, float, float, float], timer: StageTimer
    ) -> MaskResult:
        with timer.stage(STAGE_SEG_DECODE, sync=False):
            _busy_sleep(self.decode_ms + float(self._rng.normal(0, self.jitter_ms / 2)))

        height, width = self._shape
        mask = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx = max(1.0, (x1 - x0) / 2.0 * (0.85 + self._wobble * 0.01))
        ry = max(1.0, (y1 - y0) / 2.0 * (0.90 - self._wobble * 0.005))

        ys = np.arange(height).reshape(-1, 1)
        xs = np.arange(width).reshape(1, -1)
        ellipse = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
        mask |= ellipse
        return MaskResult(mask=mask, score=float(0.7 + 0.25 * self._rng.random()))
