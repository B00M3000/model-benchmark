"""NanoOWL detector (NVIDIA-AI-IOT/nanoowl).

Shared by both pairings. In a single job this is loaded exactly once and
reused across both runs -- the weights, engine and cached text encodings are
identical either way, so reloading would only add engine-deserialisation
time. Each pairing still runs its own forward pass on every frame, so the
runs stay fully independent in the measurement sense.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from ..timing import STAGE_DETECT, STAGE_DETECT_DECODE, STAGE_DETECT_ENCODE, StageTimer
from .base import BackendUnavailable, Detection, Detector


class NanoOwlDetector(Detector):
    name = "nanoowl"

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        image_encoder_engine: str | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_encoder_engine = image_encoder_engine
        self.device = device
        self.variant = model_name
        self._predictor: Any = None
        self._prompts: list[str] = []
        self._text_encodings: Any = None
        self._threshold: float = 0.1
        # Set when the installed nanoowl exposes encode_image/decode, which
        # lets us split detection into encoder vs head.
        self._can_split = False

    def _load(self) -> None:
        try:
            from nanoowl.owl_predictor import OwlPredictor
        except ImportError as exc:  # pragma: no cover - Jetson-only path
            raise BackendUnavailable(
                "nanoowl is not importable on this host. Install "
                "NVIDIA-AI-IOT/nanoowl on the Jetson, or run with backend=mock."
            ) from exc

        if self.image_encoder_engine and not os.path.exists(self.image_encoder_engine):
            raise BackendUnavailable(
                f"NanoOWL TensorRT engine not found: {self.image_encoder_engine}. "
                "Build it with scripts/build_engines.sh."
            )

        kwargs: dict[str, Any] = {"model_name": self.model_name, "device": self.device}
        if self.image_encoder_engine:
            kwargs["image_encoder_engine"] = self.image_encoder_engine
        self._predictor = OwlPredictor(**kwargs)
        self._can_split = hasattr(self._predictor, "encode_image") and hasattr(
            self._predictor, "decode"
        )
        self.variant = f"{self.model_name}" + (
            " (TensorRT)" if self.image_encoder_engine else " (PyTorch)"
        )

    def set_prompts(self, prompts: Sequence[str], threshold: float) -> float:
        import time

        self._prompts = list(prompts)
        self._threshold = threshold
        start = time.perf_counter_ns()
        self._text_encodings = self._predictor.encode_text(self._prompts)
        return (time.perf_counter_ns() - start) / 1e6

    def detect(self, frame_rgb: np.ndarray, timer: StageTimer) -> list[Detection]:
        from PIL import Image

        image = Image.fromarray(frame_rgb)

        if self._can_split:
            return self._detect_split(image, timer)
        return self._detect_whole(image, timer)

    def _detect_split(self, image: Any, timer: StageTimer) -> list[Detection]:
        """Preferred path: separate encoder cost from detection-head cost."""
        predictor = self._predictor
        with timer.stage(STAGE_DETECT):
            with timer.stage(STAGE_DETECT_ENCODE):
                image_tensor = predictor.image_preprocessor.preprocess_pil_image(image)
                image_output = predictor.encode_image(image_tensor)
            with timer.stage(STAGE_DETECT_DECODE):
                output = predictor.decode(
                    image_output, self._text_encodings, threshold=self._threshold
                )
        return self._to_detections(output)

    def _detect_whole(self, image: Any, timer: StageTimer) -> list[Detection]:
        """Fallback for nanoowl builds without the split API."""
        with timer.stage(STAGE_DETECT):
            output = self._predictor.predict(
                image=image,
                text=self._prompts,
                text_encodings=self._text_encodings,
                threshold=self._threshold,
                pad_square=False,
            )
        return self._to_detections(output)

    def _to_detections(self, output: Any) -> list[Detection]:
        boxes = _to_numpy(getattr(output, "boxes", []))
        scores = _to_numpy(getattr(output, "scores", []))
        labels = _to_numpy(getattr(output, "labels", []))
        detections: list[Detection] = []
        for i in range(len(boxes)):
            idx = int(labels[i]) if i < len(labels) else 0
            score = float(scores[i]) if i < len(scores) else 0.0
            name = self._prompts[idx] if 0 <= idx < len(self._prompts) else str(idx)
            x0, y0, x1, y1 = (float(v) for v in boxes[i][:4])
            detections.append(
                Detection(box=(x0, y0, x1, y1), score=score, label=name, label_index=idx)
            )
        return detections

    def unload(self) -> None:
        self._predictor = None
        self._text_encodings = None
        super().unload()
        _free_cuda()


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _free_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
