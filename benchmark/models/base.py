"""Backend interfaces.

The pipeline only ever talks to :class:`Detector` and :class:`Segmenter`,
which is what lets the same measurement code drive NanoOWL/NanoSAM/
EfficientViT-SAM on the Orin and CPU mocks on a dev box.

Both segmenters share an encode-once-per-frame / decode-once-per-box shape,
so that split is baked into the interface and the two are directly
comparable stage by stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..timing import StageTimer


class BackendUnavailable(RuntimeError):
    """A backend's libraries or engine files are missing on this host.

    Raised at load time (never at import time) so the server keeps running
    and the UI can show a precise reason instead of the process dying.
    """


@dataclass
class Detection:
    """One detected object. Box is absolute pixels, ``x0, y0, x1, y1``."""

    box: tuple[float, float, float, float]
    score: float
    label: str
    label_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [round(float(v), 2) for v in self.box],
            "score": round(float(self.score), 4),
            "label": self.label,
            "label_index": self.label_index,
        }

    def clamped(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.box
        return (
            max(0, min(int(x0), width - 1)),
            max(0, min(int(y0), height - 1)),
            max(0, min(int(x1), width)),
            max(0, min(int(y1), height)),
        )


@dataclass
class MaskResult:
    """A single instance mask plus the decoder's own confidence."""

    mask: np.ndarray  # bool, full frame resolution
    score: float = 0.0


class Backend(ABC):
    """Shared load/unload lifecycle."""

    name: str = "backend"
    #: Human-readable model identity, filled in at load time.
    variant: str = ""

    def __init__(self) -> None:
        self._loaded = False
        self.load_ms: float = 0.0

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def _load(self) -> None: ...

    def load(self) -> None:
        """Idempotent load. Model loading is never inside a timed region."""
        if self._loaded:
            return
        import time

        start = time.perf_counter_ns()
        self._load()
        self.load_ms = (time.perf_counter_ns() - start) / 1e6
        self._loaded = True

    def unload(self) -> None:
        """Release device memory. Safe to call when not loaded."""
        self._loaded = False

    def reset(self) -> None:
        """Clear per-run state before a backend is reused for a new run.

        Matters because a shared detector survives across both pairings:
        anything it carries between runs would make run B differ from run A
        for reasons that have nothing to do with the segmenter under test.
        """

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "variant": self.variant, "load_ms": round(self.load_ms, 2)}


class Detector(Backend):
    """Open-vocabulary detector (NanoOWL)."""

    @abstractmethod
    def set_prompts(self, prompts: Sequence[str], threshold: float) -> float:
        """Encode text prompts once, up front.

        A drone would cache these too, so the cost is reported separately
        rather than charged to every frame. Returns the encode time in ms.
        """

    @abstractmethod
    def detect(self, frame_rgb: np.ndarray, timer: StageTimer) -> list[Detection]:
        """Detect on one RGB frame, recording ``detect`` into ``timer``.

        Implementations also record ``detect_encode``/``detect_decode`` when
        the underlying API exposes that split.
        """


class Segmenter(Backend):
    """Promptable segmenter (NanoSAM / EfficientViT-SAM)."""

    @abstractmethod
    def set_image(self, frame_rgb: np.ndarray, timer: StageTimer) -> None:
        """Run the image encoder once for this frame (``seg_encode``)."""

    @abstractmethod
    def segment_box(
        self, box: tuple[float, float, float, float], timer: StageTimer
    ) -> MaskResult:
        """Decode a mask for one box (``seg_decode``, accumulated)."""
