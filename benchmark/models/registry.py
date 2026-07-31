"""Backend selection and the two pairings under study.

Both pairings share NanoOWL as the detector and differ only in the
segmentation head -- that is the whole point of the ablation.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
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
    # torch2trt's setup.py does `import tensorrt` and `import torch` at module
    # level to compile a CUDA extension. pip's PEP 517 build isolation builds
    # in a throwaway env containing only declared build-requires -- it does
    # NOT inherit the outer venv's --system-site-packages -- so that isolated
    # env has neither, and the build fails with "No module named 'tensorrt'"
    # despite both being importable everywhere else. --no-build-isolation
    # makes it build with the current environment instead.
    build_isolation: bool = True

    @property
    def install_hint(self) -> str:
        flag = "-e " if self.editable else ""
        iso = "" if self.build_isolation else " --no-build-isolation"
        return f"git clone {self.repo} && pip install {flag}./{self.module} --no-deps{iso}"


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
        build_isolation=False,
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

    Deliberately checked via ``importlib.metadata``, not ``importlib.util.
    find_spec``. find_spec asks ``sys.meta_path`` finders in order and uses
    the first non-None answer -- and for a *editable* pip install (the
    ``pip install -e`` used throughout this project), that answer can come
    from the wrong finder.

    Every one of these packages is a git clone whose root directory shares
    its name with the package nested inside it. Whenever that clone root is
    on ``sys.path`` (true for this whole app, which adds the repo root to
    import itself), Python's default ``PathFinder`` -- earlier in
    ``sys.meta_path`` than pip's own editable-install finder -- notices a
    directory named e.g. ``nanosam`` with no ``__init__.py`` and returns a
    namespace-package spec for it *before pip's finder is ever consulted*,
    even when the package is genuinely, correctly installed elsewhere via
    ``pip install -e``. So even right after a real, successful install,
    find_spec can still report the empty shadow instead of the real thing.

    ``importlib.metadata`` sidesteps all of this: it reads ``*.dist-info``
    directly rather than resolving an import, so it is unaffected by which
    meta_path finder gets asked first.
    """
    try:
        dist = importlib.metadata.distribution(module)
    except importlib.metadata.PackageNotFoundError:
        dist = None

    if dist is not None:
        origin = ""
        try:
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                info = json.loads(direct_url)
                url = info.get("url", "")
                if url.startswith("file://"):
                    origin = url[len("file://") :]
        except Exception:
            pass
        if not origin:
            location = dist.locate_file("")
            origin = str(location) if location else ""
        return "ok", origin

    # No pip distribution anywhere. Fall back to ordinary import resolution --
    # this is what makes config.repo_paths work with no pip install at all,
    # and what finds JetPack packages (torch, tensorrt) that were never pip-
    # installed and so have no dist-info for the check above to find. A clone
    # directory alone still "imports" successfully as an empty namespace
    # package (spec.origin is None); tell that apart from a genuine ordinary
    # import and from nothing being present at all.
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
