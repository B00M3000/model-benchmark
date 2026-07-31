#!/usr/bin/env python3
"""Environment preflight for the ablation benchmark.

Jetson environments break in a handful of specific, recognisable ways --
a PyPI torch wheel shadowing the JetPack build, a missing engine, unpinned
clocks. This reports each one with the fix rather than letting it surface
as a traceback halfway through an engine build or a benchmark run.

    python3 scripts/doctor.py

Exit code 0 if the host can run a real benchmark, 1 otherwise.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)

_problems: list[str] = []
_warnings: list[str] = []


def ok(label: str, detail: str = "") -> None:
    print(f"  {GREEN}✓{RESET} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def warn(label: str, detail: str = "", fix: str = "") -> None:
    print(f"  {YELLOW}!{RESET} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if fix:
        print(f"    {DIM}{fix}{RESET}")
    _warnings.append(label)


def bad(label: str, detail: str = "", fix: str = "") -> None:
    print(f"  {RED}✗{RESET} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if fix:
        print(f"    {DIM}{fix}{RESET}")
    _problems.append(label)


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def cuda_driver_version() -> int | None:
    """CUDA version the installed driver supports, e.g. 12060 for 12.6.

    Read straight from libcuda so it works even when torch refuses to
    initialise -- which is exactly the case we most need to diagnose.
    """
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        version = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(version)) == 0:
            return version.value
    return None


def fmt_cuda(version: int) -> str:
    return f"{version // 1000}.{(version % 1000) // 10}"


def check_platform() -> None:
    section("Platform")
    import platform

    machine = platform.machine()
    l4t = Path("/etc/nv_tegra_release")
    if l4t.exists():
        ok("Jetson (L4T)", l4t.read_text().strip().splitlines()[0])
    elif machine == "aarch64":
        warn("aarch64 but no /etc/nv_tegra_release", "not a JetPack image?")
    else:
        warn(f"Not a Jetson ({machine})", "mock backends only; numbers will be synthetic")

    ok("Python", f"{platform.python_version()}  {sys.executable}")
    if sys.prefix == sys.base_prefix:
        warn(
            "Not running inside the venv",
            fix="source .venv/bin/activate  (or use .venv/bin/python)",
        )


def check_torch() -> None:
    section("PyTorch / CUDA")

    driver = cuda_driver_version()
    if driver is None:
        warn("No CUDA driver found (libcuda)", "expected on a non-Jetson dev box")
    else:
        ok("CUDA driver", f"supports up to CUDA {fmt_cuda(driver)}")

    try:
        import torch
    except ImportError:
        bad(
            "torch not importable",
            fix="On the Jetson use a venv created with --system-site-packages so "
                "JetPack's torch is inherited.",
        )
        return

    version = torch.__version__
    built_for = torch.version.cuda
    is_jetpack_build = ".nv" in version or "tegra" in version.lower()

    ok("torch", f"{version}  (built for CUDA {built_for})  {torch.__file__}")

    if not is_jetpack_build and driver is not None:
        warn(
            "This does not look like a JetPack build of torch",
            "JetPack wheels carry an .nv suffix, e.g. 2.5.0a0+872d972e41.nv24.08",
            "A PyPI torch wheel will not match the Jetson driver -- see below.",
        )

    # The decisive check. Catch rather than crash: a driver mismatch raises
    # here, and that is the single most common Jetson breakage.
    try:
        available = torch.cuda.is_available()
    except RuntimeError as exc:
        message = str(exc)
        if "too old" in message:
            hint = (
                f"torch was built for CUDA {built_for} but the driver only supports "
                f"CUDA {fmt_cuda(driver)}.\n    "
                "Reinstall the JetPack-matched wheel, e.g. for JetPack 6.x / CUDA 12.6:\n    "
                "  pip install --no-cache-dir --index-url "
                "https://pypi.jetson-ai-lab.dev/jp6/cu126 torch torchvision\n    "
                "Then install the model repos with --no-deps so pip cannot replace it again."
            )
        else:
            hint = message
        bad("torch.cuda is unusable", message.splitlines()[0], hint)
        return

    if not available:
        bad(
            "torch.cuda.is_available() is False",
            fix="No usable GPU. On a Jetson this usually means a mismatched torch wheel.",
        )
        return

    ok("torch.cuda", f"{torch.cuda.get_device_name(0)}")

    try:
        import torchvision

        ok("torchvision", torchvision.__version__)
    except ImportError:
        bad(
            "torchvision not importable",
            "NanoOWL needs it (torchvision.ops.roi_align)",
            "Install the JetPack-matched torchvision -- the version must pair with torch.",
        )


def check_tensorrt() -> None:
    section("TensorRT")
    try:
        import tensorrt

        ok("tensorrt", tensorrt.__version__)
    except ImportError:
        bad(
            "tensorrt not importable",
            fix="Ships with JetPack. Create the venv with --system-site-packages.",
        )

    from shutil import which

    trtexec = None
    for candidate in ("trtexec", "/usr/src/tensorrt/bin/trtexec"):
        if which(candidate) or os.path.exists(candidate):
            trtexec = candidate
            break
    if trtexec:
        ok("trtexec", trtexec)
    else:
        warn(
            "trtexec not on PATH",
            fix='export PATH="/usr/src/tensorrt/bin:$PATH"',
        )


def check_models() -> None:
    section("Model packages")
    from benchmark.config import load_config
    from benchmark.models.registry import JETSON_MODULES, ensure_repo_paths

    config = load_config()
    added = ensure_repo_paths(config)
    if added:
        ok("repo_paths", ", ".join(added))

    import importlib.util

    for module in JETSON_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            ok(module, spec.origin or "")
        else:
            bad(
                module,
                "not importable",
                f"pip install -e ../{module} --no-deps   "
                f"(or add its clone to repo_paths in config.yaml)",
            )


def check_artifacts() -> None:
    section("Weights and engines")
    from benchmark.config import load_config

    config = load_config()
    entries = [
        ("NanoOWL image encoder", config.nanoowl.image_encoder_engine, True),
        ("NanoSAM image encoder", config.nanosam.image_encoder_engine, True),
        ("NanoSAM mask decoder", config.nanosam.mask_decoder_engine, True),
        ("EfficientViT-SAM weights", config.efficientvit.weights, True),
        ("EfficientViT-SAM encoder engine", config.efficientvit.encoder_engine, False),
        ("EfficientViT-SAM decoder engine", config.efficientvit.decoder_engine, False),
    ]
    for label, raw, required in entries:
        path = config.resolve_path(raw)
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1 << 20)
            ok(label, f"{size_mb:.0f} MB")
        elif required:
            bad(label, f"missing: {path}", "./scripts/build_engines.sh")
        else:
            warn(label, "absent — EfficientViT-SAM will run via PyTorch instead")


def check_clocks() -> None:
    section("Benchmark hygiene")
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        warn("nvpmodel not available", "not a Jetson, or not on PATH")
        return

    # `nvpmodel -q` prints the mode name then the mode number, e.g.
    #   NV Power Mode: MAXN
    #   0
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    mode = " / ".join(lines[:2])
    if lines and (lines[-1] == "0" or "MAXN" in out.upper()):
        ok("Power mode", mode)
    else:
        warn(
            "Power mode may not be maximum",
            mode,
            "sudo nvpmodel -m 0 && sudo jetson_clocks   "
            "(unpinned clocks are the usual cause of noisy Jetson numbers)",
        )


def main() -> int:
    print(f"{BOLD}Ablation benchmark — environment check{RESET}")
    for check in (
        check_platform,
        check_torch,
        check_tensorrt,
        check_models,
        check_artifacts,
        check_clocks,
    ):
        try:
            check()
        except Exception as exc:  # a broken check must not hide the others
            bad(f"{check.__name__} failed", f"{type(exc).__name__}: {exc}")

    print()
    if _problems:
        print(f"{RED}{len(_problems)} problem(s) block a real benchmark:{RESET}")
        for problem in _problems:
            print(f"  · {problem}")
        print(f"\n{DIM}The app still runs with mock backends until these are fixed.{RESET}")
        return 1

    if _warnings:
        print(f"{YELLOW}Ready, with {len(_warnings)} warning(s).{RESET}")
    else:
        print(f"{GREEN}All checks passed — ready to benchmark.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
