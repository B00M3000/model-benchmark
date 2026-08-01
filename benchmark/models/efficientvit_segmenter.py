"""EfficientViT-SAM segmenter (mit-han-lab/efficientvit) -- pairing B's head.

Same encode-once / decode-per-box shape as NanoSAM, so ``seg_encode`` and
``seg_decode`` mean the same thing in both pairings and the stage-by-stage
comparison is apples to apples.

Supports the PyTorch model zoo and, when engines are present, the TensorRT
deployment. ``runtime="auto"`` prefers TensorRT.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..timing import STAGE_SEG_DECODE, STAGE_SEG_ENCODE, StageTimer
from .base import BackendUnavailable, MaskResult, Segmenter
from .nanoowl_detector import _free_cuda


class EfficientViTSamSegmenter(Segmenter):
    name = "efficientvit_sam"

    def __init__(
        self,
        model: str = "efficientvit-sam-l0",
        weights: str | None = None,
        runtime: str = "auto",
        encoder_engine: str | None = None,
        decoder_engine: str | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.model = model
        self.weights = weights
        self.runtime = runtime
        self.encoder_engine = encoder_engine
        self.decoder_engine = decoder_engine
        self.device = device
        self.variant = model
        self._predictor: Any = None
        self._model: Any = None
        self._frame_shape: tuple[int, int] = (0, 0)

    def _use_tensorrt(self) -> bool:
        have_engines = bool(
            self.encoder_engine
            and self.decoder_engine
            and os.path.exists(self.encoder_engine)
            and os.path.exists(self.decoder_engine)
        )
        if self.runtime == "tensorrt":
            if not have_engines:
                raise BackendUnavailable(
                    "EfficientViT-SAM runtime=tensorrt but engines are missing: "
                    f"{self.encoder_engine}, {self.decoder_engine}"
                )
            return True
        if self.runtime == "torch":
            return False
        return have_engines

    def _load(self) -> None:
        if self._use_tensorrt():
            self._load_tensorrt()
        else:
            self._load_torch()

    def _load_torch(self) -> None:
        try:
            from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
            from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        except ImportError as exc:  # pragma: no cover - Jetson-only path
            # Reporting exc matters: this catch fires both for "efficientvit
            # isn't installed" and for "it is, but one of its own runtime
            # dependencies is missing" (it pulls in segment_anything, timm,
            # triton and more before the SAM predictor is usable). Without
            # the cause, the two are indistinguishable and the message sends
            # you off reinstalling something that was never the problem.
            raise BackendUnavailable(
                f"Cannot import efficientvit's SAM predictor: {exc}. "
                "Run scripts/doctor.py -- it checks each of efficientvit's "
                "runtime dependencies separately -- or use backend=mock."
            ) from exc

        if self.weights and not os.path.exists(self.weights):
            raise BackendUnavailable(
                f"EfficientViT-SAM weights not found: {self.weights}. "
                "Download them with scripts/build_engines.sh."
            )

        model = create_efficientvit_sam_model(self.model, True, self.weights)
        self._model = model.to(self.device).eval()
        self._predictor = EfficientViTSamPredictor(self._model)
        self.variant = f"{self.model} (PyTorch)"

    def _load_tensorrt(self) -> None:  # pragma: no cover - Jetson-only path
        from .trt_sam import TrtSamPredictor

        self._predictor = TrtSamPredictor(
            self.encoder_engine,
            self.decoder_engine,
            model=self.model,
            device=self.device,
        )
        self.variant = f"{self.model} (TensorRT)"

    def set_image(self, frame_rgb: np.ndarray, timer: StageTimer) -> None:
        self._frame_shape = frame_rgb.shape[:2]
        with timer.stage(STAGE_SEG_ENCODE):
            self._predictor.set_image(frame_rgb)

    def segment_box(
        self, box: tuple[float, float, float, float], timer: StageTimer
    ) -> MaskResult:
        box_arr = np.array([box[0], box[1], box[2], box[3]], dtype=np.float32)
        with timer.stage(STAGE_SEG_DECODE):
            masks, scores, _ = self._predictor.predict(box=box_arr, multimask_output=False)

        mask_np = np.asarray(masks)
        while mask_np.ndim > 2:
            mask_np = mask_np[0]
        score_arr = np.asarray(scores).ravel()
        score = float(score_arr[0]) if score_arr.size else 0.0
        return MaskResult(mask=mask_np.astype(bool), score=score)

    def unload(self) -> None:
        self._predictor = None
        self._model = None
        super().unload()
        _free_cuda()
