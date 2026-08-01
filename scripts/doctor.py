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

import argparse
import ctypes
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)

_problems: list[str] = []
_warnings: list[str] = []
# (label, callable) for everything --fix knows how to repair. Populated as the
# checks run, so a fix is only ever offered for a problem actually observed.
_fixes: list[tuple[str, Callable[[], bool]]] = []


def ok(label: str, detail: str = "") -> None:
    print(f"  {GREEN}✓{RESET} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def _record_fix(label: str, fix_action: Callable[[], bool] | None) -> None:
    if fix_action is not None:
        _fixes.append((label, fix_action))
        print(f"    {DIM}--fix can repair this{RESET}")


def warn(
    label: str,
    detail: str = "",
    fix: str = "",
    fix_action: Callable[[], bool] | None = None,
) -> None:
    print(f"  {YELLOW}!{RESET} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if fix:
        print(f"    {DIM}{fix}{RESET}")
    _record_fix(label, fix_action)
    _warnings.append(label)


def bad(
    label: str,
    detail: str = "",
    fix: str = "",
    fix_action: Callable[[], bool] | None = None,
) -> None:
    print(f"  {RED}✗{RESET} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if fix:
        print(f"    {DIM}{fix}{RESET}")
    _record_fix(label, fix_action)
    _problems.append(label)


def run_fix(what: str, args: list[str], use_sudo: bool = False) -> bool:
    """Run one repair command, echoing it first so it is never a black box."""
    if use_sudo and os.geteuid() != 0:
        args = ["sudo"] + args
    print(f"\n{BOLD}--> {what}{RESET}")
    print(f"    {DIM}{' '.join(args)}{RESET}")
    try:
        return subprocess.run(args).returncode == 0
    except Exception as exc:
        print(f"    {RED}failed: {type(exc).__name__}: {exc}{RESET}")
        return False


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


def jetson_identity() -> tuple[str, str]:
    """(board model, L4T release), either possibly empty.

    /etc/nv_tegra_release is the usual marker but is absent from some flashed
    and containerised images, so fall back to the device tree -- a Jetson
    always reports its board there.
    """
    model = ""
    try:
        raw = Path("/proc/device-tree/model").read_bytes()
        candidate = raw.decode("utf-8", "ignore").strip("\x00 \n")
        if "jetson" in candidate.lower() or "tegra" in candidate.lower():
            model = candidate
    except OSError:
        pass

    release = ""
    try:
        release = Path("/etc/nv_tegra_release").read_text().strip().splitlines()[0]
    except (OSError, IndexError):
        pass

    if not model and not release and Path("/etc/nv_boot_control.conf").exists():
        model = "Tegra"
    return model, release


def check_platform() -> None:
    section("Platform")
    import platform

    machine = platform.machine()
    model, release = jetson_identity()
    if model or release:
        ok("Jetson", "  ".join(part for part in (model, release) if part))
    elif machine == "aarch64":
        warn("aarch64, but no Jetson markers found", "not a JetPack image?")
    else:
        warn(f"Not a Jetson ({machine})", "mock backends only; numbers will be synthetic")

    ok("Python", f"{platform.python_version()}  {sys.executable}")
    if sys.prefix == sys.base_prefix:
        warn(
            "Not running inside the venv",
            fix="source .venv/bin/activate  (or use .venv/bin/python)",
        )


def parse_cuda(text: str | None) -> tuple[int, int] | None:
    """'12.6' -> (12, 6). None when torch is a CPU build."""
    if not text:
        return None
    parts = text.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None


def jetson_wheel_index(driver: int | None) -> str:
    """NVIDIA's wheel index matching the driver's CUDA version."""
    cuda = f"cu{fmt_cuda(driver).replace('.', '')}" if driver else "cu126"
    return f"https://pypi.jetson-ai-lab.io/jp6/{cuda}"


JETSON_TORCH_FIX = (
    "Install a Jetson-matched build instead, e.g. for JetPack 6.x / CUDA 12.6:\n    "
    "  pip install --no-cache-dir --force-reinstall --no-deps --index-url "
    "https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision\n    "
    "(--force-reinstall matters: with --system-site-packages, an unconstrained\n    "
    "'torchvision' can already be 'satisfied' by a mismatched system build, so\n    "
    "pip silently skips reinstalling it without this flag.)\n    "
    "Then install the model repos with --no-deps so pip cannot replace it again."
)


def install_matched_torch(driver: int | None):
    """Install a driver-matched torch/torchvision into the *venv*.

    Deliberately into the venv rather than over the system copy: the venv takes
    precedence over /usr/local/lib/.../dist-packages, so this shadows the bad
    build without sudo and without touching anything JetPack owns. Undo by
    deleting the venv.

    torch and torchvision go in together, always -- installing one alone is
    what produces "operator torchvision::nms does not exist". That failure
    mode is also what happens if this command runs WITHOUT --force-reinstall:
    with --system-site-packages, an unconstrained "torchvision" is already
    satisfied by whatever mismatched build sits in system dist-packages, so
    pip silently skips it and installs only torch -- pairing a fresh,
    driver-matched torch with the old, mismatched torchvision. --no-deps
    keeps this to exactly those two wheels; without it, --force-reinstall
    would also re-fetch every transitive dependency (numpy, pillow, sympy,
    ...) from this Jetson-only index, which likely doesn't host them.
    """
    if sys.prefix == sys.base_prefix:
        return None  # would modify the system install; too blunt to automate

    def run() -> bool:
        return run_fix(
            "install driver-matched torch + torchvision into the venv",
            [
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "--force-reinstall", "--no-deps",
                "--index-url", jetson_wheel_index(driver),
                "torch", "torchvision",
            ],
        )

    return run


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

    ok("torch", f"{version}  (built for CUDA {built_for})  {torch.__file__}")

    # Checked before the CUDA early-returns below: a broken numpy bridge
    # breaks NanoOWL regardless of whether CUDA works.
    check_numpy(torch)

    # Compare what torch was built against with what the driver supports.
    # This is the condition that actually produces "The NVIDIA driver on your
    # system is too old"; the wheel's filename or version suffix is not, since
    # working Jetson wheels are published without one.
    #
    # No fix_action here even though this is the real diagnosis: the
    # torch.cuda.is_available() check right below is about to hit the same
    # mismatch and offers the identical repair. Attaching it to both would
    # queue the same pip install twice under --fix.
    built = parse_cuda(built_for)
    if built is not None and driver is not None and built > (
        driver // 1000,
        (driver % 1000) // 10,
    ):
        warn(
            "torch is built for a newer CUDA than the driver supports",
            f"torch wants CUDA {built_for}, driver supports {fmt_cuda(driver)}",
            "This is a PyPI wheel, not a Jetson one. " + JETSON_TORCH_FIX,
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
                f"CUDA {fmt_cuda(driver) if driver else '?'}.\n    " + JETSON_TORCH_FIX
            )
        else:
            hint = message
        bad(
            "torch.cuda is unusable",
            message.splitlines()[0],
            hint,
            fix_action=install_matched_torch(driver) if "too old" in message else None,
        )
        return

    if not available:
        bad(
            "torch.cuda.is_available() is False",
            fix="No usable GPU. On a Jetson this usually means a mismatched torch "
                "wheel.\n    " + JETSON_TORCH_FIX,
            fix_action=install_matched_torch(driver),
        )
        return

    ok("torch.cuda", f"{torch.cuda.get_device_name(0)}")
    check_torchvision(torch, driver)


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
    "  pip install --no-cache-dir --force-reinstall --no-deps --index-url "
    "https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision\n    "
    "(--force-reinstall matters here: with --system-site-packages, pip can "
    "decide\n    an unconstrained 'torchvision' is already satisfied by the "
    "mismatched\n    build and silently skip reinstalling it.)"
)


def check_torchvision(torch, driver: int | None = None) -> None:
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
            fix_action=install_matched_torch(driver),
        )
        return
    except RuntimeError as exc:
        bad(
            "torchvision failed to load",
            str(exc).splitlines()[0],
            "Its compiled ops were built against a different torch.\n    "
            + MATCHED_PAIR_FIX,
            fix_action=install_matched_torch(driver),
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
            fix_action=install_matched_torch(driver),
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
            fix_action=install_matched_torch(driver),
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


# Modules that get a dedicated, deeper section below.
DETAILED_CHECKS = {"torch2trt"}


def check_models() -> None:
    section("Model packages")
    from benchmark.config import load_config
    from benchmark.models.registry import REQUIRED_MODULES, ensure_repo_paths

    config = load_config()
    added = ensure_repo_paths(config)
    if added:
        ok("repo_paths", ", ".join(added))

    from benchmark.models.registry import module_status

    for required in REQUIRED_MODULES:
        if required.module in DETAILED_CHECKS:
            continue  # reported by its own section, in more detail
        report_module(required, *module_status(required.module))


def report_module(required, status: str, origin: str) -> None:
    """Shared reporting for one required package."""
    if status == "ok":
        ok(required.module, origin)
    elif status == "shadowed":
        bad(
            required.module,
            f"not installed — {origin or 'the clone directory'} is shadowing it",
            "The clone's root directory has the same name as the package inside "
            "it,\n    so Python imports the empty root instead. Installing it "
            "fixes the ordering:\n    "
            f"  {required.install_hint}",
            fix_action=install_module(required, origin),
        )
    else:
        bad(
            required.module,
            f"not importable — {required.why}",
            f"{required.install_hint}\n    "
            "(or add the clone to repo_paths in config.yaml)",
            fix_action=install_module(required, origin),
        )


def install_module(required, origin: str):
    """A callable that pip-installs one required package, or None."""
    clone = Path(origin) if origin else REPO_ROOT / required.module
    if not (clone / "setup.py").exists() and not (clone / "pyproject.toml").exists():
        return None

    def run() -> bool:
        args = [sys.executable, "-m", "pip", "install"]
        if required.editable:
            args.append("-e")
        args += [str(clone), "--no-deps"]
        if not required.build_isolation:
            args.append("--no-build-isolation")
        return run_fix(f"install {required.module}", args)

    return run


def install_transformers() -> bool:
    # No --no-deps: unlike the four repos above, transformers does not
    # declare torch as a dependency (only as an optional extra), so a plain
    # install can't replace JetPack's build. Its only unconstrained core
    # dependency of note is numpy>=1.17 -- already satisfied by the pinned
    # numpy<2 install, so pip leaves it alone rather than upgrading it.
    return run_fix(
        "install transformers (NanoOWL's runtime dependency)",
        [sys.executable, "-m", "pip", "install", "transformers"],
    )


def check_nanoowl_runtime() -> None:
    """NanoOWL imports transformers at runtime, but nothing installs it.

    nanoowl's own setup.py declares zero dependencies, and it must be
    installed with --no-deps anyway (its README lists torch as a manual
    prerequisite, and letting pip resolve it would replace JetPack's build).
    NVIDIA's own README instead lists `pip install transformers` as a
    separate, explicit setup step -- deliberately without --no-deps, since
    that is what pulls in transformers' own working dependency chain
    (huggingface_hub, httpx, idna, and the rest). Skipping that step doesn't
    fail at nanoowl's own import; it fails one or two frames deeper, inside
    transformers' import of OwlViTForObjectDetection, with whichever piece of
    that chain happens to be missing (idna, httpx, huggingface_hub, ...).
    Since the fix is the same regardless of which one it is, this checks the
    exact import NanoOWL performs rather than guessing dependency names.
    """
    section("NanoOWL runtime (transformers)")
    try:
        from transformers.models.owlvit.modeling_owlvit import (  # noqa: F401
            OwlViTForObjectDetection,
        )
    except ModuleNotFoundError as exc:
        bad(
            f"{exc.name} not importable",
            "part of transformers' own dependency chain -- NanoOWL needs it "
            "to import OwlViTForObjectDetection",
            "pip install transformers      # deliberately not --no-deps here; "
            "see the comment on check_nanoowl_runtime for why that's safe",
            fix_action=install_transformers,
        )
        return
    except ImportError as exc:
        bad(
            "transformers import failed",
            str(exc).splitlines()[0],
            fix_action=install_transformers,
        )
        return
    ok("transformers", "OwlViTForObjectDetection importable")


def install_timm() -> bool:
    # --no-deps, unlike transformers/onnx: timm's pyproject.toml declares
    # torch AND torchvision as hard dependencies (unconstrained), so a plain
    # install risks pip replacing JetPack's build. --no-deps is safe here
    # specifically because timm's only OTHER dependencies (pyyaml,
    # huggingface_hub, safetensors) are already satisfied by the transformers
    # install above -- nothing legitimate is being skipped.
    return run_fix(
        "install timm (NanoSAM's vendored MobileSAM needs it) -- --no-deps, "
        "since timm declares torch/torchvision as hard dependencies",
        [sys.executable, "-m", "pip", "install", "timm", "--no-deps"],
    )


def check_nanosam_runtime() -> None:
    """NanoSAM's vendored MobileSAM imports timm, but nothing installs it.

    This is optional, not required: build_engines.sh fetches a pre-built
    mobile_sam_mask_decoder.onnx by default (exporting it fresh reliably
    produces a graph trtexec rejects on a modern torch -- see the comment in
    build_engines.sh), and nanosam.utils.predictor.Predictor -- the actual
    runtime class this app uses -- only ever loads compiled engines, never
    nanosam.mobile_sam. timm is only needed if you set
    NANOSAM_EXPORT_DECODER=1 to export fresh instead.

    nanosam/mobile_sam/modeling/tiny_vit_sam.py does
    `from timm.models.layers import DropPath` -- reached from
    `nanosam.mobile_sam.sam_model_registry`, which the export script imports
    at its very first line. nanosam's own setup.py declares no dependencies
    (like nanoowl, and for the same reason: it must be installed with
    --no-deps so pip can't replace JetPack's torch), so nothing pulls timm in.

    Unlike transformers and onnx, timm's own pyproject.toml lists torch and
    torchvision as hard, unconstrained dependencies -- so timm gets --no-deps
    instead. Its other three dependencies (pyyaml, huggingface_hub,
    safetensors) are already satisfied by the transformers install this
    doctor also checks for, so nothing is actually lost by skipping them.

    Skipped entirely if nanosam itself isn't really installed --
    check_models() already reports that, and this would just add a second,
    differently-worded warning about the same underlying problem.
    """
    from benchmark.models.registry import module_status

    if module_status("nanosam")[0] != "ok":
        return

    section("NanoSAM runtime (timm, optional -- only for NANOSAM_EXPORT_DECODER=1)")
    try:
        from nanosam.mobile_sam import sam_model_registry  # noqa: F401
    except ModuleNotFoundError as exc:
        warn(
            f"{exc.name} not importable",
            "only needed if exporting the mask decoder fresh "
            "(NANOSAM_EXPORT_DECODER=1) -- the default build_engines.sh path "
            "uses a pre-built ONNX and never touches nanosam.mobile_sam",
            "pip install timm --no-deps      # --no-deps: timm declares "
            "torch/torchvision as hard dependencies; see the comment on "
            "check_nanosam_runtime",
            fix_action=install_timm,
        )
        return
    except ImportError as exc:
        warn(
            "nanosam.mobile_sam import failed",
            str(exc).splitlines()[0],
            fix_action=install_timm,
        )
        return
    ok("timm", "nanosam.mobile_sam.sam_model_registry importable")


@dataclass(frozen=True)
class RuntimeDep:
    module: str
    why: str
    install_args: tuple[str, ...]  # passed to `pip install`, after the module name choice
    package_arg: str | None = None  # pip install target, if different from `module`

    @property
    def pip_target(self) -> str:
        return self.package_arg or self.module

    @property
    def install_hint(self) -> str:
        return f"pip install {self.pip_target} " + " ".join(self.install_args)


# Every one of these is reached only as a side effect of importing
# efficientvit.models.efficientvit.sam -- Python fully executes a package's
# __init__.py (and every __init__.py of every ancestor package) before any
# of its submodules are usable, and efficientvit's own __init__.py files
# each pull in far more than the SAM predictor alone needs:
#
#   models/efficientvit/__init__.py -- alongside `from .sam import *`, also
#     unconditionally does `from .dc_ae import *`, which imports omegaconf.
#   models/nn/__init__.py -- does `from .drop import *` and `from .norm
#     import *` unconditionally. .drop reaches apps.trainer.run_config,
#     which (via apps/trainer/__init__.py's own `from .base import *`)
#     reaches apps.trainer.base, which imports efficientvit.apps.data_provider
#     -- whose __init__.py pulls in augment/color_aug.py (needs timm) -- and
#     also imports efficientvit.apps.utils, whose __init__.py pulls in
#     apps/utils/export.py (needs onnxsim). .norm reaches triton_rms_norm.py
#     (needs triton) the same way.
#   segment_anything/__init__.py (not efficientvit's own code, but pulled in
#     by sam.py) unconditionally imports automatic_mask_generator.py, which
#     needs pycocotools.
#
# Traced with a static AST analyzer after three rounds of manual, file-by-
# file tracing each missed a different branch of this same tree. Checked
# here as independent, direct imports of each leaf package rather than one
# combined "does efficientvit.models.nn import" attempt -- a single combined
# check only ever reveals whichever one is missing *first* in execution
# order, misreporting every other gap as the wrong one when more than one
# is absent at once. This is exactly what happened during development here:
# a missing onnxsim was misreported as a missing triton, since drop.py (onnxsim)
# runs before norm.py (triton) inside the same __init__.py.
#
# None of these are declared by efficientvit itself (installed with
# --no-deps, like the other three repos). Only timm needs --no-deps of its
# own: its pyproject.toml declares torch and torchvision as hard,
# unconstrained dependencies, so a plain install risks replacing JetPack's
# build. None of the other five depend on torch, so there's nothing for
# --no-deps to protect against with them.
EFFICIENTVIT_RUNTIME_DEPS: tuple[RuntimeDep, ...] = (
    RuntimeDep(
        module="segment_anything",
        why="efficientvit's own SAM predictor imports it directly "
        "(models/efficientvit/sam.py) -- efficientvit's setup.py even "
        "declares this, as a git dependency, but that's exactly what "
        "--no-deps skips",
        install_args=(),
        package_arg="git+https://github.com/facebookresearch/segment-anything.git",
    ),
    RuntimeDep(
        module="pycocotools",
        why="segment_anything's own __init__.py unconditionally imports "
        "automatic_mask_generator.py, which needs this -- even though "
        "nothing this app uses ever calls automatic mask generation",
        install_args=(),
    ),
    RuntimeDep(
        module="omegaconf",
        why="models/efficientvit/__init__.py unconditionally does "
        "`from .dc_ae import *` alongside `from .sam import *` -- dc_ae.py "
        "imports omegaconf at module level",
        install_args=(),
    ),
    RuntimeDep(
        module="onnxsim",
        why="reached via models/nn -> .drop -> apps.trainer -> "
        "apps.utils.export, which does `from onnxsim import simplify` -- "
        "never actually called by anything this app uses",
        install_args=(),
    ),
    RuntimeDep(
        module="timm",
        why="continuing that same chain, apps.trainer.base imports "
        "efficientvit.apps.data_provider, whose __init__.py pulls in "
        "augment/color_aug.py, which imports timm.data.auto_augment. This "
        "is a separate reason from NanoSAM's own, unrelated need for timm "
        "(only under NANOSAM_EXPORT_DECODER=1) -- efficientvit needs it "
        "unconditionally, on its default runtime path",
        install_args=("--no-deps",),
    ),
    RuntimeDep(
        module="triton",
        why="models/nn/__init__.py also does `from .norm import *`, and "
        "norm.py unconditionally imports TritonRMSNorm2dFunc from "
        "triton_rms_norm.py, even though the L0 SAM variant this project "
        "uses never actually selects triton-based normalization "
        "(sam_model_zoo.py builds it with norm=\"bn2d\") -- only the "
        "*import* has to succeed, the kernel is never JIT-compiled or run",
        install_args=(),
    ),
)


def install_runtime_dep(dep: RuntimeDep):
    def run() -> bool:
        return run_fix(
            f"install {dep.module}",
            [sys.executable, "-m", "pip", "install", dep.pip_target, *dep.install_args],
        )

    return run


def check_efficientvit_runtime() -> None:
    """Check every one of EfficientViT-SAM's real runtime dependencies.

    See the comment on EFFICIENTVIT_RUNTIME_DEPS for how these were found
    and why each is checked independently rather than via one combined
    import attempt.

    Skipped entirely if efficientvit itself isn't really installed --
    check_models() already reports that.
    """
    from benchmark.models.registry import module_status

    if module_status("efficientvit")[0] != "ok":
        return

    section(
        "EfficientViT-SAM runtime ("
        + ", ".join(d.module for d in EFFICIENTVIT_RUNTIME_DEPS)
        + ")"
    )
    for dep in EFFICIENTVIT_RUNTIME_DEPS:
        try:
            importlib.import_module(dep.module)
        except ImportError as exc:
            bad(
                f"{dep.module} not importable",
                dep.why,
                dep.install_hint,
                fix_action=install_runtime_dep(dep),
            )
        else:
            ok(dep.module)

    # Final, holistic confirmation: the independent checks above cover
    # everything found so far, but a combined attempt at the real entry
    # point catches anything this audit still missed, rather than reporting
    # false confidence.
    try:
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor  # noqa: F401
    except Exception as exc:
        bad(
            "efficientvit.models.efficientvit.sam import failed",
            f"{type(exc).__name__}: {str(exc).splitlines()[0]} -- unexpected; "
            "not one of the dependencies above",
        )
    else:
        ok("efficientvit.models.efficientvit.sam", "EfficientViTSamPredictor importable")

    # Checked separately from the predictor because it fails separately:
    # this is the model *factory*, and upstream has renamed it before. A
    # rename raises ImportError from the same `except ImportError` that
    # catches "efficientvit isn't installed", so without its own check the
    # symptom is an install error for a package that is installed fine.
    try:
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: F401
    except Exception as exc:
        bad(
            "efficientvit.sam_model_zoo.create_efficientvit_sam_model missing",
            f"{type(exc).__name__}: {str(exc).splitlines()[0]} -- the installed "
            "efficientvit exposes a different model-factory name than this app calls",
        )
    else:
        ok("efficientvit.sam_model_zoo", "create_efficientvit_sam_model importable")


def install_onnx() -> bool:
    # No --no-deps: onnx doesn't declare torch, so there's nothing here for
    # --no-deps to protect against. Its numpy>=1.23.2 is already satisfied
    # by the pinned numpy<2 install, so pip leaves it alone.
    return run_fix(
        "install onnx (needed to export engines to ONNX before building them)",
        [sys.executable, "-m", "pip", "install", "onnx"],
    )


def check_onnx_export() -> None:
    """Both engine builds go through torch.onnx.export, which needs `onnx`.

    NanoOWL's build_image_encoder_engine() and NanoSAM's own
    export_sam_mask_decoder_onnx.py both call torch.onnx.export() to produce
    the .onnx file that trtexec then compiles. torch's ONNX exporter imports
    the onnx package itself to serialize the result -- but neither NanoOWL
    nor NanoSAM depends on it, so a clean --no-deps install lacks it. Unlike
    the runtime NanoOWL check, this doesn't block actually running a
    benchmark once engines exist; it only blocks *building* them, which is
    why it's checked next to the artifacts rather than next to the models.
    """
    section("ONNX export tooling")
    try:
        import onnx  # noqa: F401
    except ImportError:
        bad(
            "onnx not importable",
            "needed by torch.onnx.export(), which both NanoOWL's and "
            "NanoSAM's engine builds use to produce the .onnx file trtexec "
            "compiles",
            "pip install onnx      # not --no-deps; see check_onnx_export",
            fix_action=install_onnx,
        )
        return
    ok("onnx", getattr(onnx, "__version__", ""))


def check_torch2trt() -> None:
    """torch2trt is what actually executes the .engine files.

    NanoOWL and NanoSAM both do ``from torch2trt import TRTModule`` to wrap an
    engine as an nn.Module, but neither declares torch2trt as a dependency and
    it is not on PyPI -- so a clean install silently lacks it, and the gap only
    surfaces at the very end of the NanoOWL engine build.
    """
    section("torch2trt")
    from benchmark.models.registry import MODULE_BY_NAME, module_status

    required = MODULE_BY_NAME["torch2trt"]
    status, origin = module_status("torch2trt")
    if status != "ok":
        report_module(required, status, origin)
        return

    try:
        import torch2trt
    except Exception as exc:
        bad("torch2trt failed to import", f"{type(exc).__name__}: {exc}")
        return

    ok("torch2trt", getattr(torch2trt, "__file__", "") or origin)

    try:
        from torch2trt import TRTModule
    except ImportError:
        bad(
            "torch2trt.TRTModule missing",
            "installed torch2trt is too old or partially built",
            "Reinstall from master: pip install --force-reinstall --no-deps "
            "./torch2trt",
            fix_action=install_module(required, origin),
        )
        return

    # torch2trt releases predating TensorRT 10 drive engines through the
    # removed binding API, so they import cleanly and then fail at inference
    # with "no attribute 'num_bindings'". Detect that before a benchmark run
    # rather than during one.
    try:
        import inspect

        import tensorrt

        trt_major = int(tensorrt.__version__.split(".")[0])
        source = inspect.getsource(TRTModule)
        uses_trt10_api = "num_io_tensors" in source or "get_tensor_name" in source
        if trt_major >= 10 and not uses_trt10_api:
            warn(
                "torch2trt looks too old for TensorRT 10",
                f"TensorRT {tensorrt.__version__}, but TRTModule still uses the "
                "removed binding API",
                "Install torch2trt from master:\n    "
                "  pip install --force-reinstall --no-deps "
                "git+https://github.com/NVIDIA-AI-IOT/torch2trt",
            )
        else:
            ok("TRTModule", f"compatible with TensorRT {tensorrt.__version__}")
    except Exception:
        # Version probing is best-effort; the import above is the real check.
        ok("TRTModule", "importable")


def check_artifacts() -> None:
    section("Weights and engines")
    from benchmark.config import load_config

    config = load_config()
    # EfficientViT-SAM's engines are only genuinely optional when the config
    # asks for the PyTorch runtime outright. On "auto" they are missing
    # infrastructure, not a preference: the run still completes, but pairing
    # A is TensorRT and pairing B silently is not, so the comparison stops
    # measuring the segmentation head and starts measuring the runtime.
    evit_engines_required = config.efficientvit.runtime != "torch"
    entries = [
        ("NanoOWL image encoder", config.nanoowl.image_encoder_engine, True),
        ("NanoSAM image encoder", config.nanosam.image_encoder_engine, True),
        ("NanoSAM mask decoder", config.nanosam.mask_decoder_engine, True),
        ("EfficientViT-SAM weights", config.efficientvit.weights, True),
        (
            "EfficientViT-SAM encoder engine",
            config.efficientvit.encoder_engine,
            evit_engines_required,
        ),
        (
            "EfficientViT-SAM decoder engine",
            config.efficientvit.decoder_engine,
            evit_engines_required,
        ),
    ]
    for label, raw, required in entries:
        path = config.resolve_path(raw)
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1 << 20)
            ok(label, f"{size_mb:.0f} MB")
        elif required:
            bad(
                label,
                f"missing: {path}",
                "./scripts/build_engines.sh",
                fix_action=build_artifacts,
            )
        else:
            warn(label, "absent — runtime: torch is set, so this is expected")


def build_artifacts() -> bool:
    """Download the weights and build the engines, via the existing script.

    SKIP_DOCTOR=1: this is already running from inside a doctor check, so the
    script's own preflight would just re-run us -- and if an earlier fix in
    this same --fix pass had failed, its interactive "Continue anyway? [y/N]"
    prompt would block with no one watching stdin.
    """
    env = {**os.environ, "SKIP_DOCTOR": "1"}
    print(f"\n{BOLD}--> fetch weights and build TensorRT engines{RESET}")
    script = REPO_ROOT / "scripts" / "build_engines.sh"
    print(f"    {DIM}bash {script}{RESET}")
    try:
        return subprocess.run(["bash", str(script)], env=env).returncode == 0
    except Exception as exc:
        print(f"    {RED}failed: {type(exc).__name__}: {exc}{RESET}")
        return False


def check_clocks() -> None:
    section("Benchmark hygiene")
    # nvpmodel lives in /usr/sbin, which is not on a normal user's PATH.
    from shutil import which

    nvpmodel = which("nvpmodel") or next(
        (p for p in ("/usr/sbin/nvpmodel", "/usr/bin/nvpmodel") if os.path.exists(p)),
        None,
    )
    if nvpmodel is None:
        warn("nvpmodel not found", "not a Jetson?")
        return

    try:
        out = subprocess.run(
            [nvpmodel, "-q"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception as exc:
        warn("nvpmodel could not be queried", f"{type(exc).__name__}: {exc}")
        return

    # `nvpmodel -q` prints the mode name then the mode number, e.g.
    #   NV Power Mode: MAXN
    #   0
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    mode = " / ".join(lines[:2])
    if lines and (lines[-1] == "0" or "MAXN" in out.upper()):
        ok("Power mode", mode)
    else:
        def pin_clocks() -> bool:
            # Both, in order: nvpmodel raises the power ceiling, jetson_clocks
            # then pins the frequencies to it. Either alone leaves the clocks
            # free to drift mid-run.
            if not run_fix("set maximum power mode", [nvpmodel, "-m", "0"], use_sudo=True):
                return False
            clocks = which("jetson_clocks") or "/usr/bin/jetson_clocks"
            return run_fix("pin clocks", [clocks], use_sudo=True)

        warn(
            "Power mode may not be maximum",
            mode,
            "sudo nvpmodel -m 0 && sudo jetson_clocks   "
            "(unpinned clocks are the usual cause of noisy Jetson numbers)",
            fix_action=pin_clocks,
        )


CHECKS = (
    check_platform,
    check_torch,
    check_opencv,
    check_tensorrt,
    check_models,
    check_nanoowl_runtime,
    check_nanosam_runtime,
    check_efficientvit_runtime,
    check_torch2trt,
    check_onnx_export,
    check_artifacts,
    check_clocks,
)


def run_checks() -> None:
    for check in CHECKS:
        try:
            check()
        except Exception as exc:  # a broken check must not hide the others
            bad(f"{check.__name__} failed", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true",
        help="attempt each repairable problem, then re-check",
    )
    args = parser.parse_args()

    print(f"{BOLD}Ablation benchmark — environment check{RESET}")
    run_checks()

    if args.fix and _fixes:
        # Several problems can share one remedy -- four missing artifacts all
        # point at the same build_engines.sh call. Dedupe by identity so it
        # runs once instead of four times in a row.
        seen: set[int] = set()
        unique_fixes = []
        for label, fix_action in _fixes:
            if id(fix_action) in seen:
                continue
            seen.add(id(fix_action))
            unique_fixes.append((label, fix_action))

        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}--fix: attempting {len(unique_fixes)} repair(s){RESET}")
        applied = 0
        for label, fix_action in unique_fixes:
            if fix_action():
                applied += 1
            else:
                print(f"    {RED}did not complete — see output above{RESET}")

        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}Re-checking after {applied}/{len(unique_fixes)} repair(s){RESET}")
        _problems.clear()
        _warnings.clear()
        _fixes.clear()
        run_checks()

    print()
    if _problems:
        print(f"{RED}{len(_problems)} problem(s) block a real benchmark:{RESET}")
        for problem in _problems:
            print(f"  · {problem}")
        if _fixes and not args.fix:
            print(f"\n{DIM}{len(_fixes)} of these can be attempted automatically: "
                  f"python3 scripts/doctor.py --fix{RESET}")
        print(f"\n{DIM}The app still runs with mock backends until these are fixed.{RESET}")
        return 1

    if _warnings:
        print(f"{YELLOW}Ready, with {len(_warnings)} warning(s).{RESET}")
    else:
        print(f"{GREEN}All checks passed — ready to benchmark.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
