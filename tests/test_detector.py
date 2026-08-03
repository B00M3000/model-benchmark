"""NanoOWL adapter tests, against a stand-in for nanoowl's own API.

The real predictor needs a Jetson, a TensorRT engine and a HuggingFace
download, so these drive a fake that reproduces the two properties of
nanoowl's API that this adapter has to get right:

* ``preprocess_pil_image`` does NOT resize -- the resize to the model's
  768x768 input happens inside ``encode_rois``.
* ``encode_image`` yields box coordinates normalised to 0..1;
  ``encode_rois`` is what maps them back into frame pixels.

Calling encode_image directly therefore produces boxes that are both
meaningless (the encoder saw the wrong input size) and sub-pixel. Nothing
raises -- the detections simply come out empty or at the origin, which is
what these tests exist to catch.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

IMAGE_SIZE = 768
# What the fake encoder "detects", in normalised 0..1 corner coordinates.
NORMALISED_BOX = (0.25, 0.25, 0.75, 0.75)


class _Output:
    def __init__(self, pred_boxes):
        self.pred_boxes = pred_boxes
        self.image_class_embeds = None


class _Decoded:
    def __init__(self, boxes, scores, labels):
        self.boxes, self.scores, self.labels = boxes, scores, labels


def _install_fake_nanoowl(monkeypatch):
    """Register a fake nanoowl.owl_predictor mirroring the real API shape."""
    import torch

    seen: dict[str, tuple] = {}

    class FakeImagePreprocessor:
        def preprocess_pil_image(self, image):
            # Deliberately no resize -- exactly like the real one.
            array = np.array(image)  # copy: torch rejects non-writable arrays
            return torch.from_numpy(array).permute(2, 0, 1)[None, ...].float()

    class FakeOwlPredictor:
        image_size = IMAGE_SIZE

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.image_preprocessor = FakeImagePreprocessor()

        def encode_text(self, text):
            return types.SimpleNamespace(text_embeds=torch.zeros(len(text), 4))

        def encode_image(self, image):
            # Record what the encoder was actually handed: the real engine
            # is built with the spatial dims fixed at 768x768.
            seen["encode_image_shape"] = tuple(image.shape)
            return _Output(torch.tensor([list(NORMALISED_BOX)], dtype=torch.float32))

        def encode_rois(self, image, rois, pad_square=True, padding_scale=1.0):
            seen["pad_square"] = pad_square
            resized = torch.nn.functional.interpolate(
                image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
            )
            output = self.encode_image(resized)
            x0, y0, x1, y1 = (float(v) for v in rois[0].tolist())
            width, height = x1 - x0, y1 - y0
            # Mirrors _owl_box_roi_to_box_global.
            output.pred_boxes = output.pred_boxes * torch.tensor(
                [width, height, width, height]
            ) + torch.tensor([x0, y0, x0, y0])
            return output

        def decode(self, image_output, text_output, threshold=0.1):
            count = image_output.pred_boxes.shape[0]
            return _Decoded(
                boxes=image_output.pred_boxes,
                scores=torch.full((count,), 0.9),
                labels=torch.zeros(count, dtype=torch.long),
            )

    module = types.ModuleType("nanoowl.owl_predictor")
    module.OwlPredictor = FakeOwlPredictor
    package = types.ModuleType("nanoowl")
    package.owl_predictor = module
    monkeypatch.setitem(sys.modules, "nanoowl", package)
    monkeypatch.setitem(sys.modules, "nanoowl.owl_predictor", module)
    return seen


@pytest.fixture
def fake_nanoowl(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    return _install_fake_nanoowl(monkeypatch)


def _detect(width=640, height=480):
    from benchmark.models.nanoowl_detector import NanoOwlDetector
    from benchmark.timing import StageTimer

    detector = NanoOwlDetector()
    detector.load()
    detector.set_prompts(["a person"], threshold=0.1)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    return detector, detector.detect(frame, StageTimer(lambda: None))


def test_boxes_come_back_in_frame_pixels_not_normalised(fake_nanoowl):
    """The regression: normalised boxes drawn with int() collapse to a dot."""
    width, height = 640, 480
    _, detections = _detect(width, height)

    assert len(detections) == 1
    x0, y0, x1, y1 = detections[0].box
    expected = (
        NORMALISED_BOX[0] * width, NORMALISED_BOX[1] * height,
        NORMALISED_BOX[2] * width, NORMALISED_BOX[3] * height,
    )
    assert (x0, y0, x1, y1) == pytest.approx(expected, abs=1e-3)
    # The failure mode being guarded: every coordinate under 1px, so int()
    # rounds the whole box to nothing and it never appears in the render.
    assert max(x1 - x0, y1 - y0) > 1.0


def test_encoder_receives_the_size_its_engine_was_built_for(fake_nanoowl):
    """A full-resolution frame reaches an engine whose dims are fixed at 768."""
    _detect(640, 480)
    assert fake_nanoowl["encode_image_shape"] == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_frame_is_padded_square_rather_than_stretched(fake_nanoowl):
    _detect(1280, 720)
    assert fake_nanoowl["pad_square"] is True


def test_boxes_are_clamped_into_the_frame(fake_nanoowl, monkeypatch):
    """Square padding can push a box off a non-square frame."""
    import torch

    import benchmark.models.nanoowl_detector as detector_module

    width, height = 640, 480
    detector = detector_module.NanoOwlDetector()
    detector.load()
    detector.set_prompts(["a person"], threshold=0.1)

    # A box running off every edge, as square padding on 4:3 can produce.
    monkeypatch.setattr(
        detector._predictor,
        "decode",
        lambda *a, **k: _Decoded(
            boxes=torch.tensor([[-50.0, -80.0, width + 90.0, height + 40.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        ),
    )
    from benchmark.timing import StageTimer

    detections = detector.detect(
        np.zeros((height, width, 3), dtype=np.uint8), StageTimer(lambda: None)
    )
    assert detections[0].box == (0.0, 0.0, float(width), float(height))


def test_degenerate_boxes_are_dropped(fake_nanoowl, monkeypatch):
    """A box entirely off-frame clamps to zero area and must not be kept."""
    import torch

    import benchmark.models.nanoowl_detector as detector_module
    from benchmark.timing import StageTimer

    detector = detector_module.NanoOwlDetector()
    detector.load()
    detector.set_prompts(["a person"], threshold=0.1)
    monkeypatch.setattr(
        detector._predictor,
        "decode",
        lambda *a, **k: _Decoded(
            boxes=torch.tensor([[-40.0, -30.0, -5.0, -2.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        ),
    )
    detections = detector.detect(
        np.zeros((480, 640, 3), dtype=np.uint8), StageTimer(lambda: None)
    )
    assert detections == []
