"""On-disk layout for jobs, plus the off-thread mask writer.

    data/jobs/<job_id>/
        job.json          state machine + run config, rewritten on transition
        input.<ext>       uploaded video
        frames_a.jsonl    per-frame records for pairing A
        frames_b.jsonl    per-frame records for pairing B
        masks_a.jsonl     RLE masks for pairing A (only if record_masks)
        masks_b.jsonl
        results.json      aggregates, comparison, environment snapshot
        comparison.mp4    side-by-side render

Everything is plain files: a restart loses nothing, and a finished job can
be copied off the Orin with scp.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Iterator


class JobPaths:
    """Resolves and creates the per-job directory layout."""

    def __init__(self, root: Path, job_id: str) -> None:
        self.root = Path(root)
        self.job_id = job_id
        self.dir = self.root / job_id

    def ensure(self) -> "JobPaths":
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def job_json(self) -> Path:
        return self.dir / "job.json"

    @property
    def results_json(self) -> Path:
        return self.dir / "results.json"

    @property
    def video(self) -> Path:
        return self.dir / "comparison.mp4"

    def input_video(self) -> Path | None:
        for candidate in sorted(self.dir.glob("input.*")):
            return candidate
        return None

    def frames(self, pairing_id: str) -> Path:
        return self.dir / f"frames_{pairing_id}.jsonl"

    def masks(self, pairing_id: str) -> Path:
        return self.dir / f"masks_{pairing_id}.jsonl"

    def delete(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def write_json(path: Path, payload: Any) -> None:
    """Atomic write, so a crash mid-write cannot corrupt a job's state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class MaskWriter:
    """Serialises masks to disk on a background thread.

    RLE encoding costs CPU. Doing it inline would inflate the very latencies
    this tool exists to measure, so the timed region ends before handing
    off: the pipeline drops raw masks into a bounded queue and this thread
    encodes and writes them on another core.

    The queue is bounded so a slow disk applies backpressure instead of
    growing until the Orin runs out of memory.
    """

    def __init__(self, path: Path, max_queue: int = 64) -> None:
        self.path = path
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True, name="mask-writer")
        self._stop = object()
        self._started = False
        self.dropped = 0

    def start(self) -> "MaskWriter":
        if not self._started:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread.start()
            self._started = True
        return self

    def submit(self, frame_index: int, masks: list[Any], boxes: list[Any]) -> None:
        """Hand off one frame's masks. Never call from inside a timed block."""
        if not self._started:
            return
        self._queue.put((frame_index, masks, boxes))

    def _run(self) -> None:
        from . import rle

        with open(self.path, "a") as handle:
            while True:
                item = self._queue.get()
                try:
                    if item is self._stop:
                        return
                    frame_index, masks, boxes = item
                    encoded = [rle.encode(m) for m in masks]
                    handle.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "masks": encoded,
                                "boxes": boxes,
                            }
                        )
                        + "\n"
                    )
                except Exception:
                    # Mask capture is best-effort: the latency data is the
                    # deliverable, and a write failure must not kill a run.
                    self.dropped += 1
                finally:
                    self._queue.task_done()

    def close(self) -> None:
        if not self._started:
            return
        self._queue.put(self._stop)
        self._thread.join(timeout=30)
        self._started = False


def load_masks(path: Path) -> dict[int, dict[str, Any]]:
    """Load a masks JSONL keyed by frame index, for video rendering."""
    return {int(rec["frame_index"]): rec for rec in read_jsonl(path)}
