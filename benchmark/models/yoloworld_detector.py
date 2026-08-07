"""YOLO-World-S detector (ultralytics), the second open-vocabulary detector.

Deliberately the slower of the two, and included for that reason: NanoOWL
runs its image encoder through TensorRT but only understands noun phrases,
while YOLO-World stays in PyTorch and understands longer descriptions. The
accuracy-latency trade-off between them is the thing worth measuring.

Only ``detect`` is timed as a single stage. NanoOWL splits into encode and
decode because nanoowl exposes those separately; ultralytics runs one
forward pass with no comparable seam, so ``detect_encode``/``detect_decode``
stay unset rather than being invented by splitting an arbitrary boundary.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from ..timing import STAGE_DETECT, StageTimer
from .base import BackendUnavailable, Detection, Detector


class YoloWorldDetector(Detector):
    name = "yoloworld"

    def __init__(
        self,
        model: str = "yolov8s-worldv2.pt",
        weights: str | None = None,
        imgsz: int = 640,
        device: str = "cuda",
        precision: str = "fp16",
        max_det: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.weights = weights
        self.imgsz = imgsz
        self.device = device
        self.precision = precision
        self.max_det = max_det
        self._precision_kwarg: dict[str, Any] = {}
        self.variant = model
        self._predictor: Any = None
        self._prompts: list[str] = []
        self._threshold: float = 0.1

    def _load(self) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:  # pragma: no cover - Jetson-only path
            raise BackendUnavailable(
                f"ultralytics is not importable on this host: {exc}. Install it "
                "with --no-deps (it declares torch, torchvision AND opencv-python "
                "as hard dependencies, all of which would replace JetPack's "
                "builds), or run with backend=mock."
            ) from exc

        # set_classes() encodes the prompts with CLIP, and ultralytics will
        # pip-install CLIP itself if the import fails -- without --no-deps,
        # mid-run. On a Jetson that means PyPI torch landing on top of
        # JetPack's build in the middle of a benchmark. Fail here instead.
        try:
            import clip  # noqa: F401
        except ImportError as exc:  # pragma: no cover - Jetson-only path
            raise BackendUnavailable(
                "The `clip` package is missing, and YOLO-World needs it to encode "
                "prompts. Do NOT let ultralytics install it on demand: it shells "
                "out to pip without --no-deps and CLIP declares torch and "
                "torchvision, so it would replace JetPack's build mid-run. "
                "Install it first:\n"
                "    pip install git+https://github.com/ultralytics/CLIP.git --no-deps\n"
                "    pip install ftfy regex tqdm"
            ) from exc

        source = self.weights if self.weights else self.model
        if self.weights and not os.path.exists(self.weights):
            raise BackendUnavailable(
                f"YOLO-World weights not found: {self.weights}. "
                "Fetch them with scripts/build_engines.sh."
            )

        self._predictor = YOLOWorld(source)
        if self.device:
            self._predictor.to(self.device)

        # `half` was renamed to `quantize` in ultralytics 8.4; passing the
        # old name still works but warns on *every* predict call, which on a
        # 3500-frame clip is 3500 warnings interleaved with the timings.
        # Picked once here rather than probed per frame.
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if self.precision == "fp16":
            if "quantize" in DEFAULT_CFG_DICT:
                self._precision_kwarg = {"quantize": 16}
            else:
                self._precision_kwarg = {"half": True}
        self.variant = f"{self.model} (PyTorch, {self.precision})"

    def set_prompts(self, prompts: Sequence[str], threshold: float) -> float:
        import time

        self._prompts = list(prompts)
        self._threshold = threshold
        start = time.perf_counter_ns()
        # Encodes the prompts with CLIP once, exactly like NanoOWL's
        # encode_text -- a drone would cache this too, so it is reported as
        # setup rather than charged to every frame.
        self._predictor.set_classes(self._prompts)
        return (time.perf_counter_ns() - start) / 1e6

    def detect(self, frame_rgb: np.ndarray, timer: StageTimer) -> list[Detection]:
        with timer.stage(STAGE_DETECT):
            results = self._predictor.predict(
                frame_rgb,
                imgsz=self.imgsz,
                conf=self._threshold,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
                **self._precision_kwarg,
            )
        return self._to_detections(results, frame_rgb.shape[1], frame_rgb.shape[0])

    def _to_detections(self, results: Any, width: int, height: int) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        # Ultralytics returns xyxy already scaled back to the source frame,
        # so unlike nanoowl there is no coordinate frame to undo here.
        xyxy = _to_numpy(boxes.xyxy)
        conf = _to_numpy(boxes.conf).ravel()
        cls = _to_numpy(boxes.cls).ravel()
        for i in range(len(xyxy)):
            idx = int(cls[i]) if i < len(cls) else 0
            score = float(conf[i]) if i < len(conf) else 0.0
            name = self._prompts[idx] if 0 <= idx < len(self._prompts) else str(idx)
            x0, y0, x1, y1 = (float(v) for v in xyxy[i][:4])
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
        super().unload()
        from .nanoowl_detector import _free_cuda

        _free_cuda()


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
