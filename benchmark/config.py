"""Configuration loaded from config.yaml, overridable by environment.

Every path and threshold that differs between a dev box and the Orin lives
here, so switching hosts never means editing code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class NanoOwlConfig(BaseModel):
    model_name: str = "google/owlvit-base-patch32"
    image_encoder_engine: str | None = "data/owl_image_encoder_patch32.engine"
    device: str = "cuda"


class NanoSamConfig(BaseModel):
    image_encoder_engine: str = "data/resnet18_image_encoder.engine"
    mask_decoder_engine: str = "data/mobile_sam_mask_decoder.engine"


class EfficientViTConfig(BaseModel):
    model: str = "efficientvit-sam-l0"
    weights: str | None = "data/efficientvit_sam_l0.pt"
    runtime: Literal["auto", "torch", "tensorrt"] = "auto"
    encoder_engine: str | None = "data/efficientvit_sam_l0_encoder.engine"
    decoder_engine: str | None = "data/efficientvit_sam_l0_decoder.engine"
    device: str = "cuda"


class MockConfig(BaseModel):
    """Synthetic latencies, only used when backend resolves to ``mock``."""

    detector_encode_ms: float = 8.0
    detector_decode_ms: float = 2.0
    nanosam_encode_ms: float = 10.0
    nanosam_decode_ms: float = 3.0
    efficientvit_encode_ms: float = 17.0
    efficientvit_decode_ms: float = 5.0
    jitter_ms: float = 1.5


class RunDefaults(BaseModel):
    """Defaults for a benchmark run; the UI can override each per job."""

    prompts: list[str] = Field(default_factory=lambda: ["a person"])
    threshold: float = 0.1
    #: Process every Nth frame. 1 = every frame.
    stride: int = 1
    #: Hard cap on processed frames; null means the whole video.
    max_frames: int | None = None
    #: Frames run but excluded from the headline stats. Without this, the
    #: first TensorRT/cuDNN calls dominate the mean and both pairings look
    #: wrong. Set 0 to include everything.
    warmup_frames: int = 10
    #: Persist masks so the comparison video can render without re-running
    #: inference. Set false for a zero-overhead pure-latency run.
    record_masks: bool = True


class ModelLifecycle(BaseModel):
    """How models are held across the two runs of a single job.

    Both pairings run in one process; these knobs control what is kept
    resident between them.
    """

    #: Keep NanoOWL loaded across both runs. Identical weights and text
    #: encodings either way, so reloading only costs engine deserialisation.
    #: Each pairing still runs its own detection pass -- no boxes are shared.
    share_detector: bool = True
    #: Load both segmenters up front for the smallest inter-run gap, at the
    #: cost of higher peak VRAM.
    preload_all_models: bool = False
    #: Tear down and reload everything between runs for maximum isolation.
    #: Overrides share_detector.
    reload_between_runs: bool = False


class VideoConfig(BaseModel):
    """Side-by-side comparison render. Non-critical: never blocks results."""

    enabled: bool = True
    #: Max width of each panel; output is roughly twice this plus padding.
    panel_width: int = 960
    fps: float | None = None  # None = inherit the source frame rate
    mask_alpha: float = 0.45
    codec: str = "mp4v"
    #: Burn per-frame latency and running FPS into each panel.
    draw_stats: bool = True


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = "data/jobs"
    max_upload_mb: int = 2048
    #: Jobs kept on disk; older ones are pruned at startup. 0 = unlimited.
    keep_jobs: int = 50


class AppConfig(BaseModel):
    #: "auto" picks real backends when importable, else mock. Force with
    #: "jetson" (fail loudly if unavailable) or "mock".
    backend: Literal["auto", "jetson", "mock"] = "auto"
    nanoowl: NanoOwlConfig = Field(default_factory=NanoOwlConfig)
    nanosam: NanoSamConfig = Field(default_factory=NanoSamConfig)
    efficientvit: EfficientViTConfig = Field(default_factory=EfficientViTConfig)
    mock: MockConfig = Field(default_factory=MockConfig)
    run: RunDefaults = Field(default_factory=RunDefaults)
    lifecycle: ModelLifecycle = Field(default_factory=ModelLifecycle)
    video: VideoConfig = Field(default_factory=VideoConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def resolve_path(self, path: str | None) -> str | None:
        """Resolve a configured path relative to the repo root."""
        if not path:
            return path
        p = Path(path)
        return str(p if p.is_absolute() else (REPO_ROOT / p))

    @property
    def data_dir(self) -> Path:
        return Path(self.resolve_path(self.server.data_dir) or "data/jobs")


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load config.yaml if present, then apply environment overrides.

    Environment overrides (handy for systemd units and quick experiments):
    ``BENCH_BACKEND``, ``BENCH_PORT``, ``BENCH_HOST``, ``BENCH_DATA_DIR``.
    """
    config_path = Path(path or os.environ.get("BENCH_CONFIG", DEFAULT_CONFIG_PATH))
    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as handle:
            raw = yaml.safe_load(handle) or {}

    config = AppConfig(**raw)

    if backend := os.environ.get("BENCH_BACKEND"):
        config.backend = backend  # type: ignore[assignment]
    if port := os.environ.get("BENCH_PORT"):
        config.server.port = int(port)
    if host := os.environ.get("BENCH_HOST"):
        config.server.host = host
    if data_dir := os.environ.get("BENCH_DATA_DIR"):
        config.server.data_dir = data_dir

    return config
