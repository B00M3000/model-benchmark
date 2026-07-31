"""Shared fixtures. Everything runs against mock backends on CPU."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.config import AppConfig


@pytest.fixture
def synthetic_video(tmp_path):
    """A short clip with a moving bright square, written with OpenCV."""
    import cv2

    path = tmp_path / "clip.mp4"
    width, height, frames, fps = 320, 240, 24, 30.0
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for i in range(frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        x = int(20 + i * 6)
        cv2.rectangle(frame, (x, 90), (x + 60, 150), (60, 200, 220), -1)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def mock_config(tmp_path) -> AppConfig:
    """Mock backends, fast synthetic latencies, isolated data dir."""
    config = AppConfig()
    config.backend = "mock"
    config.server.data_dir = str(tmp_path / "jobs")
    config.run.warmup_frames = 2
    config.video.panel_width = 240
    # Keep the suite quick: these are busy-waits.
    config.mock.detector_encode_ms = 1.0
    config.mock.detector_decode_ms = 0.4
    config.mock.nanosam_encode_ms = 1.2
    config.mock.nanosam_decode_ms = 0.4
    config.mock.efficientvit_encode_ms = 2.0
    config.mock.efficientvit_decode_ms = 0.6
    config.mock.jitter_ms = 0.2
    return config
