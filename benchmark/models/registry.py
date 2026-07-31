"""Backend selection and the two pairings under study.

Both pairings share NanoOWL as the detector and differ only in the
segmentation head -- that is the whole point of the ablation.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable

from ..config import AppConfig
from .base import Detector, Segmenter

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class RequiredModule:
    module: str
    repo: str
    why: str
    editable: bool = True

    @property
    def install_hint(self) -> str:
        flag = "-e " if self.editable else ""
        return f"git clone {self.repo} && pip install {flag}./{self.module} --no-deps"


# --no-deps on every one of these: all four declare torch (and torch2trt
# declares tensorrt) as dependencies, and letting pip satisfy those replaces
# JetPack's builds with PyPI wheels compiled for a different CUDA.
REQUIRED_MODULES: tuple[RequiredModule, ...] = (
    RequiredModule(
        module="nanoowl",
        repo="https://github.com/NVIDIA-AI-IOT/nanoowl",
        why="the detector, shared by both pairings",
    ),
    RequiredModule(
        module="nanosam",
        repo="https://github.com/NVIDIA-AI-IOT/nanosam",
        why="pairing A's segmentation head",
    ),
    RequiredModule(
        module="efficientvit",
        repo="https://github.com/mit-han-lab/efficientvit",
        why="pairing B's segmentation head",
    ),
    # Not on PyPI, and neither nanoowl nor nanosam declares it -- so nothing
    # installs it as a side effect. Both import TRTModule from it to run their
    # TensorRT engines, so a missing torch2trt breaks both pairings.
    RequiredModule(
        module="torch2trt",
        repo="https://github.com/NVIDIA-AI-IOT/torch2trt",
        why="runs the TensorRT engines for NanoOWL and NanoSAM",
        editable=False,
    ),
)

MODULE_BY_NAME = {m.module: m for m in REQUIRED_MODULES}

JETSON_MODULES = tuple(m.module for m in REQUIRED_MODULES)


def ensure_repo_paths(config: AppConfig) -> list[str]:
    """Prepend configured repo clones to ``sys.path``.

    Lets the model repos be imported straight from a git clone, so a failed
    ``pip install -e`` -- a common JetPack problem, where the system
    setuptools and packaging versions disagree -- does not block a run.

    Idempotent: safe to call on every backend resolution.
    """
    added: list[str] = []
    for raw in config.repo_paths:
        path = config.resolve_path(raw)
        if not path:
            continue
        if not os.path.isdir(path):
            logger.warning("repo_paths entry does not exist, skipping: %s", path)
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    if added:
        # find_spec caches directory listings; without this a path added
        # after startup would not be seen.
        importlib.invalidate_caches()
    return added


def module_status(module: str) -> tuple[str, str]:
    """(status, origin) for a required module: ok / shadowed / missing.

    ``shadowed`` is the trap. Every one of these packages is a git clone whose
    root directory shares its name with the package nested inside it -- so with
    the clone's *parent* on sys.path (which is the case whenever the process
    runs from the repo root, since the clones live there), Python resolves the
    name to the clone root as a PEP 420 namespace package. That import
    succeeds and find_spec returns a spec, but the package is empty: the real
    module is one directory deeper. It fails later at the first attribute
    access, e.g. ``from torch2trt import TRTModule``.

    A namespace package has ``origin is None``, which is how we tell.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return "missing", ""
    if spec is None:
        return "missing", ""
    if spec.origin is None:
        locations = list(getattr(spec, "submodule_search_locations", None) or [])
        return "shadowed", locations[0] if locations else ""
    return "ok", spec.origin


def missing_jetson_modules(config: AppConfig | None = None) -> list[str]:
    """Which required packages are not actually usable on this host.

    A shadowed module counts as missing: it imports, but nothing in it works.
    """
    if config is not None:
        ensure_repo_paths(config)
    return [m for m in JETSON_MODULES if module_status(m)[0] != "ok"]


def jetson_backends_available(config: AppConfig | None = None) -> bool:
    """True when all three Jetson packages are importable on this host."""
    return not missing_jetson_modules(config)


def resolve_backend(config: AppConfig) -> str:
    """Decide between real and mock backends.

    ``auto`` uses the real ones when importable and silently falls back to
    mock otherwise; ``jetson`` refuses to fall back, so a misconfigured Orin
    fails loudly instead of quietly producing synthetic numbers.
    """
    if config.backend == "mock":
        return "mock"
    ensure_repo_paths(config)
    if config.backend == "jetson":
        return "jetson"
    return "jetson" if jetson_backends_available(config) else "mock"


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
