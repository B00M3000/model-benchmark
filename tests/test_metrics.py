"""Aggregation maths, checked against values computed by hand."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.metrics import compare, histogram, summarize, summarize_pairing


def test_summarize_known_values():
    # 1..100: percentiles are exact and easy to verify.
    stats = summarize(list(range(1, 101)))
    assert stats.count == 100
    assert stats.min == 1
    assert stats.max == 100
    assert stats.mean == pytest.approx(50.5)
    assert stats.p50 == pytest.approx(50.5)
    assert stats.p90 == pytest.approx(90.1)
    assert stats.p95 == pytest.approx(95.05)
    assert stats.p99 == pytest.approx(99.01)
    assert stats.std == pytest.approx(np.std(np.arange(1, 101)))


def test_summarize_empty_is_zeroed():
    stats = summarize([])
    assert stats.count == 0
    assert stats.p99 == 0.0


def test_summarize_single_value():
    stats = summarize([7.5])
    assert stats.count == 1
    assert stats.p50 == stats.p99 == stats.max == 7.5
    assert stats.std == 0.0


def _frame(seq, pipeline_ms, warmup=False, dets=1):
    return {
        "seq": seq,
        "warmup": warmup,
        "pipeline_ms": pipeline_ms,
        "e2e_ms": pipeline_ms + 1.0,
        "stages": {"detect": pipeline_ms * 0.6, "seg_encode": pipeline_ms * 0.4},
        "num_detections": dets,
    }


def test_warmup_frames_are_excluded():
    frames = [_frame(0, 500.0, warmup=True), _frame(1, 500.0, warmup=True)]
    frames += [_frame(i, 10.0) for i in range(2, 12)]

    summary = summarize_pairing("a", "A", "det", "seg", frames)

    assert summary.frames_total == 12
    assert summary.frames_warmup == 2
    assert summary.frames_measured == 10
    # The 500 ms warm-up frames must not touch the reported numbers.
    assert summary.pipeline.mean == pytest.approx(10.0)
    assert summary.pipeline.max == pytest.approx(10.0)


def test_throughput_uses_measured_frames_only():
    frames = [_frame(i, 20.0) for i in range(10)]
    summary = summarize_pairing("a", "A", "det", "seg", frames)
    # 20 ms/frame -> 50 fps.
    assert summary.throughput_fps == pytest.approx(50.0)
    assert summary.mean_instantaneous_fps == pytest.approx(50.0)


def test_detection_context_is_recorded():
    frames = [_frame(0, 10.0, dets=2), _frame(1, 10.0, dets=0), _frame(2, 10.0, dets=4)]
    summary = summarize_pairing("a", "A", "det", "seg", frames)
    assert summary.detections_total == 6
    assert summary.mean_detections_per_frame == pytest.approx(2.0)
    assert summary.frames_with_no_detections == 1


def test_compare_direction_and_percentages():
    fast = summarize_pairing("a", "A", "d", "s", [_frame(i, 10.0) for i in range(5)])
    slow = summarize_pairing("b", "B", "d", "s", [_frame(i, 15.0) for i in range(5)])

    result = compare(fast, slow)
    assert result["faster_pairing"] == "a"
    # B is 50% slower than A.
    assert result["pipeline_p50"]["pct"] == pytest.approx(50.0)
    assert result["pipeline_p50"]["abs"] == pytest.approx(5.0)


def test_histogram_counts_all_values():
    hist = histogram([1.0, 2.0, 3.0, 4.0], bins=4)
    assert sum(hist["counts"]) == 4
    assert len(hist["edges"]) == 5


def test_histogram_empty():
    assert histogram([]) == {"counts": [], "edges": []}
