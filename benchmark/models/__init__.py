"""Model backends and the pairing registry."""

from .base import Backend, BackendUnavailable, Detection, Detector, MaskResult, Segmenter
from .registry import (
    PAIRING_A,
    PAIRING_B,
    PAIRING_BY_ID,
    PAIRINGS,
    Pairing,
    build_detector,
    build_segmenter,
    ensure_repo_paths,
    jetson_backends_available,
    missing_jetson_modules,
    resolve_backend,
)

__all__ = [
    "Backend",
    "BackendUnavailable",
    "Detection",
    "Detector",
    "MaskResult",
    "Segmenter",
    "PAIRING_A",
    "PAIRING_B",
    "PAIRING_BY_ID",
    "PAIRINGS",
    "Pairing",
    "build_detector",
    "build_segmenter",
    "ensure_repo_paths",
    "jetson_backends_available",
    "missing_jetson_modules",
    "resolve_backend",
]
