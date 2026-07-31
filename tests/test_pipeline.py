"""Pipeline, timing and RLE behaviour, driven by the mock backends."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import rle
from benchmark.models.registry import PAIRINGS, build_detector, build_segmenter
from benchmark.pipeline import RunConfig, planned_frame_count, probe_video, run_pairing
from benchmark.storage import read_jsonl
from benchmark.timing import PIPELINE_STAGES, STAGE_DECODE, StageTimer


# ── timing ──────────────────────────────────────────────────────────────
def test_stage_timer_accumulates_repeated_stages():
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("seg_decode", sync=False):
            pass
    # Three decodes in one frame must sum, not overwrite -- that is what
    # makes seg_decode the per-frame total across every detection.
    assert timer.marks["seg_decode"] > 0
    assert len([k for k in timer.marks if k == "seg_decode"]) == 1


def test_pipeline_ms_excludes_decode():
    timer = StageTimer()
    timer.add(STAGE_DECODE, 100.0)
    for stage in PIPELINE_STAGES:
        timer.add(stage, 1.0)

    assert timer.pipeline_ms() == pytest.approx(len(PIPELINE_STAGES) * 1.0)
    assert timer.e2e_ms() == pytest.approx(timer.pipeline_ms() + 100.0)


def test_timer_sync_is_invoked_for_gpu_stages():
    calls = []
    timer = StageTimer(sync=lambda: calls.append(1))
    with timer.stage("detect"):
        pass
    # Once before and once after -- without this, async CUDA work lands in
    # the next stage's measurement.
    assert len(calls) == 2

    with timer.stage("decode", sync=False):
        pass
    assert len(calls) == 2


# ── RLE ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rle_roundtrip(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((40, 55)) > 0.6
    assert np.array_equal(rle.decode(rle.encode(mask)), mask)


def test_rle_handles_all_set_and_all_clear():
    for mask in (np.ones((8, 9), bool), np.zeros((8, 9), bool)):
        assert np.array_equal(rle.decode(rle.encode(mask)), mask)


def test_rle_mask_starting_with_one():
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    encoded = rle.encode(mask)
    assert encoded["counts"][0] == 0  # leading empty zero-run
    assert np.array_equal(rle.decode(encoded), mask)


def test_rle_area_matches_count():
    rng = np.random.default_rng(3)
    mask = rng.random((30, 30)) > 0.5
    assert rle.area(rle.encode(mask)) == int(np.count_nonzero(mask))


# ── video probing ───────────────────────────────────────────────────────
def test_probe_video(synthetic_video):
    info = probe_video(synthetic_video)
    assert info.width == 320
    assert info.height == 240
    assert info.frame_count == 24


def test_planned_frame_count_respects_stride_and_cap(synthetic_video):
    info = probe_video(synthetic_video)
    assert planned_frame_count(info, RunConfig(stride=1)) == 24
    assert planned_frame_count(info, RunConfig(stride=3)) == 8
    assert planned_frame_count(info, RunConfig(stride=1, max_frames=5)) == 5


# ── the run itself ──────────────────────────────────────────────────────
def test_run_pairing_produces_records(synthetic_video, mock_config, tmp_path):
    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    frames_path = tmp_path / "frames_a.jsonl"
    masks_path = tmp_path / "masks_a.jsonl"
    run = RunConfig(prompts=["a box"], warmup_frames=2, record_masks=True)

    summary, records = run_pairing(
        synthetic_video, pairing, detector, segmenter, run,
        frames_path=frames_path, masks_path=masks_path,
    )

    assert len(records) == 24
    assert summary.frames_warmup == 2
    assert summary.frames_measured == 22
    assert summary.pipeline.p50 > 0
    assert summary.throughput_fps > 0

    # Every pipeline stage must be present and the total must be consistent.
    first = records[5]
    for stage in ("preprocess", "detect", "postprocess"):
        assert stage in first["stages"]
    total = sum(first["stages"].get(s, 0.0) for s in PIPELINE_STAGES)
    assert first["pipeline_ms"] == pytest.approx(total, abs=0.01)
    assert first["e2e_ms"] >= first["pipeline_ms"]

    # Records were persisted for the video render and the CSV export.
    persisted = list(read_jsonl(frames_path))
    assert len(persisted) == 24
    assert masks_path.exists()

    masks = list(read_jsonl(masks_path))
    assert masks, "masks should have been written by the background writer"
    decoded = rle.decode(masks[0]["masks"][0])
    assert decoded.shape == (240, 320)
    assert decoded.any()


def test_stride_and_max_frames_are_honoured(synthetic_video, mock_config):
    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    run = RunConfig(stride=4, max_frames=3, warmup_frames=0, record_masks=False)
    _, records = run_pairing(synthetic_video, pairing, detector, segmenter, run)

    assert len(records) == 3
    assert [r["frame_index"] for r in records] == [0, 4, 8]


def test_record_masks_false_writes_nothing(synthetic_video, mock_config, tmp_path):
    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    masks_path = tmp_path / "masks.jsonl"
    run = RunConfig(warmup_frames=0, record_masks=False, max_frames=4)
    run_pairing(synthetic_video, pairing, detector, segmenter, run, masks_path=masks_path)

    assert not masks_path.exists()


def test_cancellation_stops_the_run(synthetic_video, mock_config):
    import threading

    from benchmark.pipeline import CancelledError

    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    cancel = threading.Event()
    seen = []

    def on_progress(progress):
        seen.append(progress)
        if len(seen) >= 3:
            cancel.set()

    with pytest.raises(CancelledError):
        run_pairing(
            synthetic_video, pairing, detector, segmenter,
            RunConfig(warmup_frames=0, record_masks=False),
            progress=on_progress, cancel=cancel,
        )
    assert len(seen) < 24


def test_segmenters_report_distinct_costs(synthetic_video, mock_config):
    """The whole study rests on the segment stage differing; assert it does."""
    results = {}
    for pairing in PAIRINGS:
        detector = build_detector(mock_config, "mock")
        detector.load()
        detector.set_prompts(["a box"], 0.1)
        segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
        segmenter.load()
        summary, _ = run_pairing(
            synthetic_video, pairing, detector, segmenter,
            RunConfig(warmup_frames=2, record_masks=False),
        )
        results[pairing.pairing_id] = summary

    a_seg = results["a"].stages["seg_encode"].mean
    b_seg = results["b"].stages["seg_encode"].mean
    # Mock EfficientViT is configured to be the heavier encoder.
    assert b_seg > a_seg
