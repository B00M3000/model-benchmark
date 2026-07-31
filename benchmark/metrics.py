"""Aggregation of per-frame records into the reported latency statistics.

Deliberately narrow: latency distribution and per-stage breakdown, as
scoped. Warm-up frames are excluded from every aggregate here -- callers
pass only the measured frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .timing import ALL_STAGES, BREAKDOWN_STAGES

PERCENTILES = (50, 90, 95, 99)


@dataclass
class Stats:
    """Distribution summary for one series of per-frame durations."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize(values: Sequence[float] | np.ndarray) -> Stats:
    """Distribution summary. Empty input yields an all-zero ``Stats``."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return Stats()
    p50, p90, p95, p99 = np.percentile(arr, PERCENTILES)
    return Stats(
        count=int(arr.size),
        mean=float(arr.mean()),
        # Population std: we are describing the frames we measured, not
        # estimating a wider population.
        std=float(arr.std()),
        min=float(arr.min()),
        p50=float(p50),
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),
        max=float(arr.max()),
    )


@dataclass
class PairingSummary:
    """Everything reported for one pairing over one video."""

    pairing_id: str
    label: str
    detector: str
    segmenter: str

    frames_total: int = 0
    frames_warmup: int = 0
    frames_measured: int = 0

    # Headline latencies (ms), warm-up excluded.
    pipeline: Stats = field(default_factory=Stats)
    e2e: Stats = field(default_factory=Stats)
    stages: dict[str, Stats] = field(default_factory=dict)

    # Throughput. These two differ and the difference is meaningful, so both
    # are reported rather than collapsing to a single "FPS".
    throughput_fps: float = 0.0
    mean_instantaneous_fps: float = 0.0

    # Context needed to interpret segmentation latency at all.
    detections_total: int = 0
    mean_detections_per_frame: float = 0.0
    frames_with_no_detections: int = 0

    # Outside the timed region, reported for transparency.
    model_load_ms: float = 0.0
    text_encode_ms: float = 0.0
    wall_clock_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["pipeline"] = self.pipeline.to_dict()
        out["e2e"] = self.e2e.to_dict()
        out["stages"] = {k: v.to_dict() for k, v in self.stages.items()}
        return out


def summarize_pairing(
    pairing_id: str,
    label: str,
    detector: str,
    segmenter: str,
    frames: Iterable[dict[str, Any]],
    *,
    wall_clock_s: float = 0.0,
    model_load_ms: float = 0.0,
    text_encode_ms: float = 0.0,
) -> PairingSummary:
    """Aggregate per-frame records into a :class:`PairingSummary`.

    ``frames`` are the raw records emitted by the pipeline, warm-up frames
    included -- they are filtered out here so the caller cannot forget to.
    """
    records = list(frames)
    measured = [f for f in records if not f.get("warmup", False)]

    summary = PairingSummary(
        pairing_id=pairing_id,
        label=label,
        detector=detector,
        segmenter=segmenter,
        frames_total=len(records),
        frames_warmup=len(records) - len(measured),
        frames_measured=len(measured),
        wall_clock_s=wall_clock_s,
        model_load_ms=model_load_ms,
        text_encode_ms=text_encode_ms,
    )
    if not measured:
        return summary

    pipeline_ms = [f["pipeline_ms"] for f in measured]
    summary.pipeline = summarize(pipeline_ms)
    summary.e2e = summarize([f["e2e_ms"] for f in measured])
    summary.stages = {
        stage: summarize([f["stages"].get(stage, 0.0) for f in measured])
        for stage in ALL_STAGES
    }

    # Throughput measured over the frames we kept, using their own summed
    # pipeline cost -- not job wall-clock, which includes model loading.
    total_pipeline_s = sum(pipeline_ms) / 1000.0
    summary.throughput_fps = len(measured) / total_pipeline_s if total_pipeline_s > 0 else 0.0
    inst = [1000.0 / ms for ms in pipeline_ms if ms > 0]
    summary.mean_instantaneous_fps = float(np.mean(inst)) if inst else 0.0

    dets = [int(f.get("num_detections", 0)) for f in measured]
    summary.detections_total = int(sum(dets))
    summary.mean_detections_per_frame = float(np.mean(dets)) if dets else 0.0
    summary.frames_with_no_detections = int(sum(1 for d in dets if d == 0))
    return summary


def compare(a: PairingSummary, b: PairingSummary) -> dict[str, Any]:
    """Head-to-head deltas, expressed as B relative to A.

    Positive ``pct`` means B is slower/larger than A.
    """

    def delta(av: float, bv: float) -> dict[str, float]:
        return {
            "a": av,
            "b": bv,
            "abs": bv - av,
            "pct": ((bv - av) / av * 100.0) if av else 0.0,
        }

    faster = None
    if a.pipeline.p50 and b.pipeline.p50:
        faster = a.pairing_id if a.pipeline.p50 < b.pipeline.p50 else b.pairing_id

    return {
        "faster_pairing": faster,
        "pipeline_mean": delta(a.pipeline.mean, b.pipeline.mean),
        "pipeline_p50": delta(a.pipeline.p50, b.pipeline.p50),
        "pipeline_p95": delta(a.pipeline.p95, b.pipeline.p95),
        "pipeline_p99": delta(a.pipeline.p99, b.pipeline.p99),
        "jitter_std": delta(a.pipeline.std, b.pipeline.std),
        "throughput_fps": delta(a.throughput_fps, b.throughput_fps),
        "stages": {
            stage: delta(
                a.stages.get(stage, Stats()).mean,
                b.stages.get(stage, Stats()).mean,
            )
            for stage in BREAKDOWN_STAGES
        },
        "detections": delta(a.mean_detections_per_frame, b.mean_detections_per_frame),
    }


def histogram(values: Sequence[float], bins: int = 30) -> dict[str, list[float]]:
    """Histogram for the latency-distribution chart."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"counts": [], "edges": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}
