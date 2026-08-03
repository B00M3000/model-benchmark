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

#: Pad the frame to a square before encoding, rather than stretching it.
#: OWL-ViT is trained on square inputs, and squashing 16:9 into 768x768
#: distorts every aspect ratio the model has learned. This is nanoowl's own
#: default in OwlPredictor.predict; both detect paths here use it so the
#: split and fallback paths cannot disagree about what was measured.
PAD_SQUARE = True


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
        # encode_rois, not encode_image: see _detect_split. encode_image alone
        # is not a usable split point.
        self._can_split = hasattr(self._predictor, "encode_rois") and hasattr(
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
        """Preferred path: separate encoder cost from detection-head cost.

        Goes through ``encode_rois`` rather than ``encode_image``, which is
        what ``OwlPredictor.predict`` itself does, because encode_image is
        not a usable split point on its own -- it is the middle of the
        encode step, not the whole of it. Two things live in encode_rois
        that nothing else does:

        * **Resizing.** ``preprocess_pil_image`` only converts and
          normalises; it does not resize. The resize to the model's
          768x768 input happens inside ``encode_rois``, via
          ``roi_align(..., output_size=get_image_size())``. Calling
          encode_image directly hands the encoder a full-resolution frame,
          and the TensorRT engine is built with the spatial dims fixed
          (``--shapes=image:1x3x768x768``; only the batch axis is dynamic),
          so it cannot answer meaningfully.
        * **Box coordinates.** ``encode_image`` returns pred_boxes straight
          out of a sigmoid -- normalised to 0..1. ``encode_rois`` applies
          ``boxes * [w, h, w, h] + [x0, y0, x0, y0]`` to put them back in
          the frame's pixel space, which is what Detection documents and
          what both segmenters and the renderer expect.

        Skipping it produced detections that were simultaneously garbage
        (wrong encoder input) and sub-pixel (normalised boxes drawn with
        int(), collapsing to a dot at the origin).
        """
        import torch

        predictor = self._predictor
        with timer.stage(STAGE_DETECT):
            with timer.stage(STAGE_DETECT_ENCODE):
                image_tensor = predictor.image_preprocessor.preprocess_pil_image(image)
                rois = torch.tensor(
                    [[0, 0, image.width, image.height]],
                    dtype=image_tensor.dtype,
                    device=image_tensor.device,
                )
                image_output = predictor.encode_rois(
                    image_tensor, rois, pad_square=PAD_SQUARE
                )
            with timer.stage(STAGE_DETECT_DECODE):
                output = predictor.decode(
                    image_output, self._text_encodings, threshold=self._threshold
                )
        return self._to_detections(output, image.width, image.height)

    def _detect_whole(self, image: Any, timer: StageTimer) -> list[Detection]:
        """Fallback for nanoowl builds without the split API."""
        with timer.stage(STAGE_DETECT):
            output = self._predictor.predict(
                image=image,
                text=self._prompts,
                text_encodings=self._text_encodings,
                threshold=self._threshold,
                pad_square=PAD_SQUARE,
            )
        return self._to_detections(output, image.width, image.height)

    def _to_detections(self, output: Any, width: int, height: int) -> list[Detection]:
        boxes = _to_numpy(getattr(output, "boxes", []))
        scores = _to_numpy(getattr(output, "scores", []))
        labels = _to_numpy(getattr(output, "labels", []))
        detections: list[Detection] = []
        for i in range(len(boxes)):
            idx = int(labels[i]) if i < len(labels) else 0
            score = float(scores[i]) if i < len(scores) else 0.0
            name = self._prompts[idx] if 0 <= idx < len(self._prompts) else str(idx)
            x0, y0, x1, y1 = (float(v) for v in boxes[i][:4])
            # Square padding makes the ROI extend past a non-square frame, so
            # a box can legitimately come back partly outside it. Clamp here
            # rather than leaving each segmenter to cope with prompts that
            # point off-image.
            x0 = min(max(x0, 0.0), float(width))
            y0 = min(max(y0, 0.0), float(height))
            x1 = min(max(x1, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            if x1 <= x0 or y1 <= y0:
                continue
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
