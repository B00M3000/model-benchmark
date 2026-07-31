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

    # Checked before the CUDA early-returns below: a broken numpy bridge
    # breaks NanoOWL regardless of whether CUDA works.
    check_numpy(torch)

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
    check_torchvision(torch)


def check_numpy(torch) -> None:
    """JetPack's torch is built against NumPy 1.x; NumPy 2 breaks its bridge.

    torch still imports cleanly, so this only shows up when something calls
    ``torch.from_numpy`` -- which NanoOWL does during construction. Exercise
    it here instead.
    """
    try:
        import numpy
    except ImportError:
        bad("numpy not importable", fix="pip install 'numpy<2'")
        return

    try:
        torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))
        ok("numpy", f"{numpy.__version__}  (torch.from_numpy works)")
    except Exception as exc:
        major = numpy.__version__.split(".")[0]
        detail = f"numpy {numpy.__version__}: {str(exc).splitlines()[0]}"
        if major.isdigit() and int(major) >= 2:
            fix = (
                "JetPack's torch is compiled against NumPy 1.x and cannot use "
                "NumPy 2.\n    pip install 'numpy<2'"
            )
        else:
            fix = "torch and numpy are incompatible; reinstall both."
        bad("torch cannot convert numpy arrays", detail, fix)


MATCHED_PAIR_FIX = (
    "torch and torchvision must be a matched pair from the same build.\n    "
    "Reinstall BOTH together, never one alone -- for JetPack 6.x / CUDA 12.6:\n    "
    "  pip uninstall -y torch torchvision\n    "
    "  pip install --no-cache-dir --index-url "
    "https://pypi.jetson-ai-lab.dev/jp6/cu126 torch torchvision"
)


def check_torchvision(torch) -> None:
    """NanoOWL imports torchvision.ops.roi_align, so this must actually work.

    A torchvision whose compiled extension was built against a different
    torch imports far enough to fail with "operator torchvision::nms does
    not exist" -- so importability alone proves nothing, and the ops get
    exercised below.
    """
    torch_root = Path(torch.__file__).resolve().parent.parent

    try:
        import torchvision
    except ImportError:
        bad(
            "torchvision not importable",
            "NanoOWL needs torchvision.ops.roi_align",
            MATCHED_PAIR_FIX,
        )
        return
    except RuntimeError as exc:
        bad(
            "torchvision failed to load",
            str(exc).splitlines()[0],
            "Its compiled ops were built against a different torch.\n    "
            + MATCHED_PAIR_FIX,
        )
        return

    ok("torchvision", f"{torchvision.__version__}  {torchvision.__file__}")

    # A venv torchvision alongside a system torch is the usual cause.
    tv_root = Path(torchvision.__file__).resolve().parent.parent
    if tv_root != torch_root:
        warn(
            "torch and torchvision are in different site-packages",
            f"torch={torch_root}  torchvision={tv_root}",
            "They are probably not a matched pair. " + MATCHED_PAIR_FIX,
        )

    try:
        from torchvision.ops import nms, roi_align  # noqa: F401

        nms(
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]]),
            torch.tensor([0.9, 0.8]),
            0.5,
        )
        ok("torchvision ops", "nms / roi_align registered")
    except Exception as exc:
        bad(
            "torchvision ops are not registered",
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            MATCHED_PAIR_FIX,
        )


def check_opencv() -> None:
    section("OpenCV")
    try:
        import cv2
    except ImportError:
        bad(
            "cv2 not importable",
            fix="On the Jetson: sudo apt install python3-opencv, and create the "
                "venv with --system-site-packages.",
        )
        return

    ok("cv2", f"{cv2.__version__}  {cv2.__file__}")
    if "site-packages" in cv2.__file__ and "dist-packages" not in cv2.__file__:
        warn(
            "cv2 looks like a pip wheel rather than JetPack's build",
            fix="The PyPI wheel is CPU-only, lacks the hardware decoders, and "
                "requires NumPy 2 -- which breaks JetPack's torch. Prefer "
                "python3-opencv.",
        )

    # Video decode is the one thing this app actually needs from OpenCV.
    if not hasattr(cv2, "VideoCapture"):
        bad("cv2.VideoCapture missing", "this OpenCV build cannot decode video")


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
        check_opencv,
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
