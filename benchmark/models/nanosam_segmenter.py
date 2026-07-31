"""NanoSAM segmenter (NVIDIA-AI-IOT/nanosam) -- pairing A's segmentation head.

NanoSAM is prompted with points rather than boxes, so a detection box is
converted to SAM's two-corner convention: the top-left corner is labelled 2
and the bottom-right corner 3, which SAM interprets as a box prompt.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..timing import STAGE_SEG_DECODE, STAGE_SEG_ENCODE, StageTimer
from .base import BackendUnavailable, MaskResult, Segmenter
from .nanoowl_detector import _free_cuda


class NanoSamSegmenter(Segmenter):
    name = "nanosam"

    def __init__(
        self,
        image_encoder_engine: str,
        mask_decoder_engine: str,
    ) -> None:
        super().__init__()
        self.image_encoder_engine = image_encoder_engine
        self.mask_decoder_engine = mask_decoder_engine
        self.variant = "NanoSAM (ResNet18 encoder, TensorRT)"
        self._predictor: Any = None
        self._frame_shape: tuple[int, int] = (0, 0)

    def _load(self) -> None:
        try:
            from nanosam.utils.predictor import Predictor
        except ImportError as exc:  # pragma: no cover - Jetson-only path
            raise BackendUnavailable(
                "nanosam is not importable on this host. Install "
                "NVIDIA-AI-IOT/nanosam on the Jetson, or run with backend=mock."
            ) from exc

        for path, what in (
            (self.image_encoder_engine, "image encoder"),
            (self.mask_decoder_engine, "mask decoder"),
        ):
            if not path or not os.path.exists(path):
                raise BackendUnavailable(
                    f"NanoSAM {what} engine not found: {path}. "
                    "Build it with scripts/build_engines.sh."
                )

        self._predictor = Predictor(self.image_encoder_engine, self.mask_decoder_engine)

    def set_image(self, frame_rgb: np.ndarray, timer: StageTimer) -> None:
        from PIL import Image

        self._frame_shape = frame_rgb.shape[:2]
        image = Image.fromarray(frame_rgb)
        with timer.stage(STAGE_SEG_ENCODE):
            self._predictor.set_image(image)

    def segment_box(
        self, box: tuple[float, float, float, float], timer: StageTimer
    ) -> MaskResult:
        x0, y0, x1, y1 = box
        points = np.array([[x0, y0], [x1, y1]], dtype=np.float32)
        # 2 = box top-left, 3 = box bottom-right (SAM's box-prompt encoding).
        point_labels = np.array([2, 3], dtype=np.float32)

        with timer.stage(STAGE_SEG_DECODE):
            mask, iou_pred, _ = self._predictor.predict(points, point_labels)

        mask_np = _to_numpy(mask)
        # NanoSAM returns logits shaped (1, C, H, W); take the first mask and
        # threshold at 0 as SAM does.
        while mask_np.ndim > 2:
            mask_np = mask_np[0]
        binary = mask_np > 0

        score = 0.0
        iou_np = _to_numpy(iou_pred).ravel()
        if iou_np.size:
            score = float(iou_np[0])

        return MaskResult(mask=_resize_to(binary, self._frame_shape), score=score)

    def unload(self) -> None:
        self._predictor = None
        super().unload()
        _free_cuda()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _resize_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Ensure the mask matches the frame, so both backends return like for like."""
    if not shape or mask.shape == shape:
        return mask.astype(bool)
    import cv2

    resized = cv2.resize(
        mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)
