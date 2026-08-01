"""Comparison-render tests, focused on producing a file that actually opens.

The renderer's historical failure was not a crash: it wrote a valid .mp4 in
MPEG-4 Part 2 ("mp4v"), which ffprobe and VLC read fine and no browser can
decode. Nothing reported an error -- the player was simply blank. These
tests assert on the codec that lands in the container, since that is the
part that determines whether the deliverable is usable.
"""

from __future__ import annotations

import shutil
import struct

import numpy as np
import pytest

from benchmark import rle
from benchmark.config import VideoConfig
from benchmark.video import COLOR_A, COLOR_B, PanelData, render_comparison

FRAMES = 8


@pytest.fixture
def clip(tmp_path):
    import cv2

    path = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened()
    for i in range(FRAMES):
        frame = np.full((240, 320, 3), 30, dtype=np.uint8)
        cv2.rectangle(frame, (20 + i * 6, 90), (80 + i * 6, 150), (60, 200, 220), -1)
        writer.write(frame)
    writer.release()
    return path


def _panels():
    mask = np.zeros((240, 320), dtype=bool)
    mask[100:140, 40:120] = True

    def one(label, color):
        return PanelData(
            label=label,
            model="model",
            color=color,
            masks={i: {"masks": [rle.encode(mask)]} for i in range(FRAMES)},
            frames={
                i: {
                    "pipeline_ms": 12.0 + i,
                    "num_detections": 1,
                    "warmup": i < 2,
                    "detections": [
                        {"box": [40, 100, 120, 140], "label": "thing", "score": 0.9}
                    ],
                }
                for i in range(FRAMES)
            },
        )

    return one("A", COLOR_A), one("B", COLOR_B)


def _boxes(path):
    """Top-level MP4 box order, so faststart is checkable."""
    data = path.read_bytes()
    order, offset = [], 0
    while offset < len(data) - 8:
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        order.append(data[offset + 4 : offset + 8].decode("latin1", "replace"))
        if size < 8:
            break
        offset += size
    return order, data


def _render(tmp_path, clip, **overrides):
    config = VideoConfig()
    config.panel_width = 240
    for key, value in overrides.items():
        setattr(config, key, value)
    panel_a, panel_b = _panels()
    return render_comparison(
        clip, tmp_path / "out.mp4", panel_a, panel_b, config, stride=1, source_fps=30.0
    )


def _has_ffmpeg_h264():
    from benchmark.video import _ffmpeg_h264_encoder

    binary = shutil.which("ffmpeg")
    return bool(binary and _ffmpeg_h264_encoder(binary))


def test_render_produces_a_decodable_video(tmp_path, clip):
    import cv2

    result = _render(tmp_path, clip)
    assert result.frames == FRAMES
    assert result.path.exists() and result.path.stat().st_size > 0

    capture = cv2.VideoCapture(str(result.path))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == FRAMES


def test_render_reports_whether_a_browser_can_play_the_result(tmp_path, clip):
    """browser_playable must reflect the codec actually written."""
    result = _render(tmp_path, clip)
    _, data = _boxes(result.path)
    wrote_h264 = b"avc1" in data
    assert result.browser_playable is wrote_h264
    # mp4v and H.264 are mutually exclusive here; catching both would mean
    # the claim is being read off the wrong box.
    assert wrote_h264 != (b"mp4v" in data)


@pytest.mark.skipif(not _has_ffmpeg_h264(), reason="no ffmpeg with an H.264 encoder")
def test_ffmpeg_path_writes_h264_with_faststart(tmp_path, clip):
    result = _render(tmp_path, clip)
    assert result.encoder.startswith("ffmpeg/")
    assert result.browser_playable is True

    order, data = _boxes(result.path)
    assert b"avc1" in data and b"avcC" in data, "not actually H.264"
    assert b"mp4v" not in data
    # moov ahead of mdat: the browser can start playing before the whole
    # file has arrived, and seeking works over range requests.
    assert order.index("moov") < order.index("mdat")


def test_falls_back_and_flags_it_when_no_h264_encoder_exists(tmp_path, clip, monkeypatch):
    """Without any H.264 encoder the render still succeeds -- but says so."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    # Force the OpenCV branch to land on mp4v even where avc1 would work.
    result = _render(tmp_path, clip, codec="mp4v")

    assert result.encoder == "opencv/mp4v"
    assert result.browser_playable is False
    assert result.frames == FRAMES  # still a usable file, just not in a browser


def test_output_dimensions_are_even(tmp_path, clip):
    """yuv420p cannot encode odd dimensions, so the composer must not emit them."""
    import cv2

    # An odd panel width makes the composed frame odd without the guard.
    result = _render(tmp_path, clip, panel_width=241)
    capture = cv2.VideoCapture(str(result.path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    assert width % 2 == 0 and height % 2 == 0
