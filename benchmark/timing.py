"""Per-stage timing with CUDA synchronisation.

The whole study rests on these numbers, so the rules are strict:

* Every GPU stage is bracketed by ``torch.cuda.synchronize()``. CUDA kernel
  launches are asynchronous -- without the sync, a stage's cost silently
  leaks into whichever stage happens to touch the GPU next, and the
  breakdown becomes fiction.
* ``time.perf_counter_ns()`` is the clock. It is monotonic and unaffected by
  wall-clock adjustments.
* Nothing outside the model call is inside the timed region.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator

# Stages that make up a single frame of work, in execution order.
STAGE_DECODE = "decode"
STAGE_PREPROCESS = "preprocess"
STAGE_DETECT_ENCODE = "detect_encode"
STAGE_DETECT_DECODE = "detect_decode"
STAGE_DETECT = "detect"
STAGE_SEG_ENCODE = "seg_encode"
STAGE_SEG_DECODE = "seg_decode"
STAGE_POSTPROCESS = "postprocess"

#: Stages summed into ``pipeline_ms`` -- the model-only cost, i.e. what the
#: drone would actually experience with a live camera feed. Video decode is
#: deliberately excluded; ``detect_encode``/``detect_decode`` are excluded
#: because they are sub-stages of ``detect`` and would double-count.
PIPELINE_STAGES = (
    STAGE_PREPROCESS,
    STAGE_DETECT,
    STAGE_SEG_ENCODE,
    STAGE_SEG_DECODE,
    STAGE_POSTPROCESS,
)

#: Order used for stacked-bar breakdowns in the UI.
BREAKDOWN_STAGES = PIPELINE_STAGES

#: Every stage we report on, including sub-stages and decode.
ALL_STAGES = (
    STAGE_DECODE,
    STAGE_PREPROCESS,
    STAGE_DETECT,
    STAGE_DETECT_ENCODE,
    STAGE_DETECT_DECODE,
    STAGE_SEG_ENCODE,
    STAGE_SEG_DECODE,
    STAGE_POSTPROCESS,
)


def null_sync() -> None:
    """No-op device sync, used by CPU/mock backends."""


def torch_cuda_sync() -> Callable[[], None]:
    """Return ``torch.cuda.synchronize`` if a CUDA device is present.

    Falls back to a no-op so the same timing code runs unchanged on an x86
    dev box without a GPU.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except Exception:
        pass
    return null_sync


class StageTimer:
    """Accumulates per-stage durations for a single frame.

    Repeated entries for the same stage add up -- that is what makes
    ``seg_decode`` the total mask-decode cost across every detection in the
    frame, rather than just the last one.
    """

    def __init__(self, sync: Callable[[], None] | None = None) -> None:
        self._sync = sync or null_sync
        self._marks: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, sync: bool = True) -> Iterator[None]:
        """Time a block and add it to ``name``.

        ``sync=False`` skips the device sync for stages known to be pure CPU
        (video decode, colour conversion), where a sync would only add cost.
        """
        if sync:
            self._sync()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            if sync:
                self._sync()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            self._marks[name] = self._marks.get(name, 0.0) + elapsed_ms

    def add(self, name: str, ms: float) -> None:
        """Record a duration measured elsewhere."""
        self._marks[name] = self._marks.get(name, 0.0) + ms

    def get(self, name: str) -> float:
        return self._marks.get(name, 0.0)

    @property
    def marks(self) -> dict[str, float]:
        return dict(self._marks)

    def pipeline_ms(self) -> float:
        """Model-only latency: what a live camera feed would see."""
        return sum(self._marks.get(s, 0.0) for s in PIPELINE_STAGES)

    def e2e_ms(self) -> float:
        """Pipeline plus video decode -- the full offline cost per frame."""
        return self.pipeline_ms() + self._marks.get(STAGE_DECODE, 0.0)
