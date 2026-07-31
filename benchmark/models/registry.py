"""Backend selection and the two pairings under study.

Both pairings share NanoOWL as the detector and differ only in the
segmentation head -- that is the whole point of the ablation.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

from ..config import AppConfig
from .base import Detector, Segmenter

PAIRING_A = "a"
PAIRING_B = "b"


@dataclass(frozen=True)
class Pairing:
    pairing_id: str
    label: str
    detector: str
    segmenter: str
    description: str


PAIRINGS: tuple[Pairing, ...] = (
    Pairing(
        pairing_id=PAIRING_A,
        label="NanoOWL + NanoSAM",
        detector="nanoowl",
        segmenter="nanosam",
        description="Open-vocab detection with a distilled ResNet18-encoder SAM head.",
    ),
    Pairing(
        pairing_id=PAIRING_B,
        label="NanoOWL + EfficientViT-SAM",
        detector="nanoowl",
        segmenter="efficientvit_sam",
        description="Same detector, EfficientViT-SAM segmentation head.",
    ),
)

PAIRING_BY_ID = {p.pairing_id: p for p in PAIRINGS}


def jetson_backends_available() -> bool:
    """True when all three Jetson packages are importable on this host."""
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ("nanoowl", "nanosam", "efficientvit")
    )


def resolve_backend(config: AppConfig) -> str:
    """Decide between real and mock backends.

    ``auto`` uses the real ones when importable and silently falls back to
    mock otherwise; ``jetson`` refuses to fall back, so a misconfigured Orin
    fails loudly instead of quietly producing synthetic numbers.
    """
    if config.backend == "mock":
        return "mock"
    if config.backend == "jetson":
        return "jetson"
    return "jetson" if jetson_backends_available() else "mock"


def build_detector(config: AppConfig, backend: str) -> Detector:
    if backend == "mock":
        from .mock import MockDetector

        return MockDetector(
            encode_ms=config.mock.detector_encode_ms,
            decode_ms=config.mock.detector_decode_ms,
            jitter_ms=config.mock.jitter_ms,
        )

    from .nanoowl_detector import NanoOwlDetector

    return NanoOwlDetector(
        model_name=config.nanoowl.model_name,
        image_encoder_engine=config.resolve_path(config.nanoowl.image_encoder_engine),
        device=config.nanoowl.device,
    )


def build_segmenter(config: AppConfig, backend: str, segmenter: str) -> Segmenter:
    if backend == "mock":
        from .mock import MockSegmenter

        if segmenter == "nanosam":
            return MockSegmenter(
                label="mock NanoSAM",
                encode_ms=config.mock.nanosam_encode_ms,
                decode_ms=config.mock.nanosam_decode_ms,
                jitter_ms=config.mock.jitter_ms,
                seed=1,
            )
        return MockSegmenter(
            label="mock EfficientViT-SAM",
            encode_ms=config.mock.efficientvit_encode_ms,
            decode_ms=config.mock.efficientvit_decode_ms,
            jitter_ms=config.mock.jitter_ms,
            seed=2,
        )

    if segmenter == "nanosam":
        from .nanosam_segmenter import NanoSamSegmenter

        return NanoSamSegmenter(
            image_encoder_engine=config.resolve_path(config.nanosam.image_encoder_engine),
            mask_decoder_engine=config.resolve_path(config.nanosam.mask_decoder_engine),
        )

    from .efficientvit_segmenter import EfficientViTSamSegmenter

    return EfficientViTSamSegmenter(
        model=config.efficientvit.model,
        weights=config.resolve_path(config.efficientvit.weights),
        runtime=config.efficientvit.runtime,
        encoder_engine=config.resolve_path(config.efficientvit.encoder_engine),
        decoder_engine=config.resolve_path(config.efficientvit.decoder_engine),
        device=config.efficientvit.device,
    )


SegmenterFactory = Callable[[], Segmenter]
