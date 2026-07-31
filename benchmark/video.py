"""Side-by-side comparison render.

Secondary deliverable by design: it runs only after both pairings have
finished and their results are already available in the UI, and a failure
here never invalidates a run.

Since mask quality is judged by eye rather than by a metric, the render
optimises for legibility -- filled masks with bright contours, per-panel
model labels, and burned-in per-frame latency so a visual difference can be
tied to its cost.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import rle
from .config import VideoConfig

# BGR. Pairing A reads teal, pairing B amber -- distinguishable side by side
# and for the most common forms of colour blindness.
COLOR_A = (196, 209, 45)
COLOR_B = (60, 170, 247)
HEADER_BG = (28, 24, 20)
PANEL_LABEL_BG = (44, 38, 32)
TEXT = (240, 240, 240)
MUTED = (168, 160, 152)
FONT = cv2.FONT_HERSHEY_SIMPLEX

RenderProgress = Callable[[int, int], None]


@dataclass
class PanelData:
    """Everything needed to draw one side of the comparison."""

    label: str
    model: str
    color: tuple[int, int, int]
    masks: dict[int, dict[str, Any]]
    frames: dict[int, dict[str, Any]]


def _draw_masks(
    frame: np.ndarray, record: dict[str, Any] | None, color: tuple[int, int, int], alpha: float
) -> np.ndarray:
    if not record:
        return frame
    overlay = frame.copy()
    color_arr = np.array(color, dtype=np.uint8)
    for encoded in record.get("masks", []):
        try:
            mask = rle.decode(encoded)
        except Exception:
            continue
        if mask.shape[:2] != frame.shape[:2]:
            mask = (
                cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )
        overlay[mask] = color_arr
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def _draw_boxes(
    frame: np.ndarray, record: dict[str, Any] | None, color: tuple[int, int, int]
) -> None:
    if not record:
        return
    for detection in record.get("detections", []):
        x0, y0, x1, y1 = (int(v) for v in detection["box"])
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
        caption = f"{detection['label']} {detection['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, FONT, 0.5, 1)
        cv2.rectangle(frame, (x0, max(0, y0 - th - 8)), (x0 + tw + 8, y0), color, -1)
        cv2.putText(
            frame, caption, (x0 + 4, max(th, y0 - 4)), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA
        )


def _draw_panel_footer(
    frame: np.ndarray, record: dict[str, Any] | None, color: tuple[int, int, int]
) -> None:
    """Burn per-frame cost into the panel so visuals and latency line up."""
    height, width = frame.shape[:2]
    bar_h = 34
    cv2.rectangle(frame, (0, height - bar_h), (width, height), PANEL_LABEL_BG, -1)
    if record:
        pipeline_ms = record.get("pipeline_ms", 0.0)
        fps = 1000.0 / pipeline_ms if pipeline_ms else 0.0
        text = (
            f"{pipeline_ms:6.1f} ms   {fps:5.1f} FPS   "
            f"{record.get('num_detections', 0)} det"
        )
        if record.get("warmup"):
            text += "   [warmup]"
    else:
        text = "no data for this frame"
    cv2.putText(
        frame, text, (12, height - 11), FONT, 0.55, color, 1, cv2.LINE_AA
    )


def _fit_panel(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    height = max(2, int(round(frame.shape[0] * scale)))
    height += height % 2
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def render_comparison(
    video_path: str | Path,
    output_path: str | Path,
    panel_a: PanelData,
    panel_b: PanelData,
    config: VideoConfig,
    *,
    stride: int = 1,
    source_fps: float = 30.0,
    progress: RenderProgress | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Compose the two annotated streams into one mp4."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video for rendering: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    header_h = 54
    divider = 4
    rendered = 0
    total = len(panel_a.frames) or len(panel_b.frames) or 1
    source_index = -1

    try:
        while True:
            if cancel is not None and cancel.is_set():
                break
            ok, frame = capture.read()
            if not ok:
                break
            source_index += 1
            # Only frames that were actually benchmarked appear in the render.
            if source_index % stride != 0:
                continue
            if source_index not in panel_a.frames and source_index not in panel_b.frames:
                continue

            panels = []
            for panel in (panel_a, panel_b):
                canvas = frame.copy()
                record = panel.frames.get(source_index)
                canvas = _draw_masks(
                    canvas, panel.masks.get(source_index), panel.color, config.mask_alpha
                )
                _draw_boxes(canvas, record, panel.color)
                canvas = _fit_panel(canvas, config.panel_width)
                if config.draw_stats:
                    _draw_panel_footer(canvas, record, panel.color)
                panels.append(canvas)

            panel_h = min(p.shape[0] for p in panels)
            panels = [p[:panel_h] for p in panels]
            gap = np.full((panel_h, divider, 3), HEADER_BG, dtype=np.uint8)
            body = np.hstack([panels[0], gap, panels[1]])

            composed = np.vstack(
                [
                    _make_header(body.shape[1], header_h, panel_a, panel_b, config.panel_width),
                    body,
                ]
            )

            if writer is None:
                fps = config.fps or (source_fps / stride if stride else source_fps)
                fourcc = cv2.VideoWriter_fourcc(*config.codec)
                writer = cv2.VideoWriter(
                    str(output_path), fourcc, max(1.0, fps), (composed.shape[1], composed.shape[0])
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open VideoWriter with codec {config.codec!r}"
                    )
            writer.write(composed)
            rendered += 1
            if progress is not None and rendered % 5 == 0:
                progress(rendered, total)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if rendered == 0:
        raise RuntimeError("No frames were rendered into the comparison video")
    if progress is not None:
        progress(rendered, total)
    return output_path


def _make_header(
    width: int, height: int, panel_a: PanelData, panel_b: PanelData, panel_width: int
) -> np.ndarray:
    header = np.full((height, width, 3), HEADER_BG, dtype=np.uint8)
    for index, panel in enumerate((panel_a, panel_b)):
        x = 16 + index * (panel_width + 4)
        cv2.rectangle(header, (x, 14), (x + 10, 38), panel.color, -1)
        cv2.putText(header, panel.label, (x + 20, 27), FONT, 0.6, TEXT, 1, cv2.LINE_AA)
        cv2.putText(header, panel.model, (x + 20, 44), FONT, 0.42, MUTED, 1, cv2.LINE_AA)
    return header
