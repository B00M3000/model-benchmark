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
import re
import subprocess
import sys
import sysconfig
import traceback
from dataclasses import dataclass
from functools import lru_cache
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
# Set by check_torchvision when torchvision is missing or unusable, and by
# check_torch when torch cannot load a shared library it links. Read by
# cascade_reason() so the checks further down can tell "this package is
# missing" apart from "this package is fine, what it imports isn't".
_torchvision_failed = False
_torch_missing_lib: str | None = None


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


@lru_cache(maxsize=None)
def install_matched_torch(driver: int | None):
    """Install a driver-matched torch/torchvision into the *venv*.

    Deliberately into the venv rather than over the system copy: no sudo, and
    nothing JetPack owns is touched. Undo by deleting the venv.

    This is necessary but on some hosts not sufficient -- see
    check_shadowed_package(). Whether the venv actually wins over
    /usr/local/lib/.../dist-packages depends on sys.path order and on
    PYTHONPATH, and it is not safe to assume it does; that assumption used to
    live in this docstring and was wrong on a real Jetson, where --fix
    installed the correct wheels and the report did not move.

    Cached, and not merely as an optimization: both the torch.cuda failure and
    the torchvision failure hand this same repair to --fix, and --fix dedupes
    the queue by callable identity. A fresh closure per call defeated that and
    downloaded 232 MB twice, the second install undoing and redoing the first.

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


# The dynamic loader names the first library it cannot find, then stops.
MISSING_LIBRARY = re.compile(r"(lib[\w.+-]+\.so[\d.]*): cannot open shared object file")

# torchaudio's own extension loader, which raises OSError rather than
# ImportError and names the .so by full path.
STALE_LIBRARY = re.compile(r"Could not load this library: (\S+)")


def package_owning(path: str) -> str:
    """'/usr/local/.../dist-packages/torchaudio/lib/libtorchaudio.so' ->
    'torchaudio'. The import name is what the reader needs; the .so path
    identifies a file nobody installed by name."""
    parts = Path(path).parts
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return Path(path).stem.removeprefix("lib")


def run_fix_torch() -> bool:
    return run_fix(
        "resolve a torch that imports in this environment",
        [sys.executable, str(REPO_ROOT / "scripts" / "fix_torch.py")],
    )


def installed_version(package_dir: Path) -> str:
    """Version of a package sitting on disk, without importing it.

    Importing is exactly what we cannot do here -- the whole point is that
    some *other* copy of this package is the one that imports.
    """
    version_py = package_dir / "version.py"
    if version_py.exists():
        match = re.search(
            r"""^__version__\s*=\s*['"]([^'"]+)""",
            version_py.read_text(errors="ignore"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    for info in package_dir.parent.glob(f"{package_dir.name}-*.dist-info"):
        # "torch-2.11.0.dist-info" -> "2.11.0". The suffix has to come off
        # before splitting: ".dist-info" contains the separator too, so a
        # plain split("-")[1] yields "2.11.0.dist".
        return info.name.removesuffix(".dist-info").split("-", 1)[1]
    return "?"


def base_python() -> str:
    """The interpreter the venv was built from -- i.e. the one whose pip owns
    /usr/local/lib/pythonX.Y/dist-packages."""
    return getattr(sys, "_base_executable", None) or str(
        Path(sys.base_prefix) / "bin" / "python3"
    )


def uninstall_shadowing(names: tuple[str, ...]):
    def run() -> bool:
        return run_fix(
            f"remove the system-wide {', '.join(names)} shadowing the venv",
            [base_python(), "-m", "pip", "uninstall", "-y", *names],
            use_sudo=True,
        )

    return run


def check_shadowed_package(module, label: str) -> bool:
    """Report a venv copy of `label` that is installed but never imported.

    The failure this exists for: --fix installs a driver-matched torch into the
    venv, pip says "Successfully installed torch-2.11.0", and `import torch`
    goes on resolving to /usr/local/lib/python3.10/dist-packages. Every
    subsequent report is identical to before the repair, so --fix looks like it
    silently did nothing -- and the half of the pair that *isn't* shadowed
    (torchvision, which now exists only in the venv) gets loaded against the
    wrong torch, turning a clean "not importable" into "operator
    torchvision::nms does not exist".

    A venv is not guaranteed to win. With --system-site-packages the system
    dist-packages directories are on sys.path too, and PYTHONPATH -- which
    JetPack setup guides hand out freely -- lands ahead of every site directory
    regardless. Measured here rather than assumed.

    Returns True when shadowing was found.
    """
    if sys.prefix == sys.base_prefix:
        return False

    venv_copy = Path(sysconfig.get_paths()["purelib"]) / label
    if not venv_copy.is_dir():
        return False  # nothing in the venv to be shadowed

    imported = Path(module.__file__).resolve().parent
    if imported == venv_copy.resolve():
        return False

    shadowing_dir = str(imported.parent)
    pythonpath = os.environ.get("PYTHONPATH", "")
    via_pythonpath = any(
        part and Path(part).resolve() == imported.parent
        for part in pythonpath.split(os.pathsep)
    )

    if via_pythonpath:
        cause = (
            f"PYTHONPATH puts {shadowing_dir} ahead of the venv. PYTHONPATH "
            f"precedes every site-packages directory, so no amount of "
            f"installing into the venv can win.\n    "
            f"Unset it for this shell and re-run:  unset PYTHONPATH\n    "
            f"(and remove it from ~/.bashrc if it is set there)"
        )
        action = None
    else:
        cause = (
            f"{shadowing_dir} precedes the venv on sys.path. Installing into "
            f"the venv cannot fix this; the shadowing copy has to go:\n    "
            f"  sudo {base_python()} -m pip uninstall -y {label}\n    "
            f"That copy is a PyPI wheel, not JetPack's -- removing it loses "
            f"nothing the venv does not already have."
        )
        action = uninstall_shadowing((label,))

    bad(
        f"the venv's {label} is installed but not the one being imported",
        f"imported {installed_version(imported)} from {imported}; "
        f"venv has {installed_version(venv_copy)} at {venv_copy}",
        cause,
        fix_action=action,
    )
    return True


def check_torch() -> None:
    section("PyTorch / CUDA")

    driver = cuda_driver_version()
    if driver is None:
        warn("No CUDA driver found (libcuda)", "expected on a non-Jetson dev box")
    else:
        ok("CUDA driver", f"supports up to CUDA {fmt_cuda(driver)}")

    global _torch_missing_lib
    try:
        import torch
    except ImportError as exc:
        # "torch is not installed" and "torch is installed but cannot load a
        # library it links" are the same ImportError and completely different
        # problems. Conflating them sends you to fix the venv's
        # --system-site-packages flag when torch is sitting right there, fully
        # installed, built for an environment that has libcudss.
        missing = MISSING_LIBRARY.search(str(exc))
        if missing:
            _torch_missing_lib = missing.group(1)
            bad(
                "torch cannot load a library it was built against",
                str(exc).splitlines()[0],
                f"torch is installed and intact; this environment does not carry "
                f"{_torch_missing_lib}. Usually one venv shared between a Jetson "
                f"host and a container image over the same working tree.\n    "
                f"  python3 scripts/fix_torch.py",
                fix_action=run_fix_torch,
            )
        else:
            bad(
                "torch not importable",
                str(exc).splitlines()[0],
                fix="On the Jetson use a venv created with --system-site-packages "
                    "so JetPack's torch is inherited.",
            )
        return

    version = torch.__version__
    built_for = torch.version.cuda

    ok("torch", f"{version}  (built for CUDA {built_for})  {torch.__file__}")

    # Before anything that reads torch's version or CUDA build: if this is not
    # the copy the venv installed, every number below describes a package the
    # last repair already tried to replace, and the repair on offer is the one
    # that just failed to take.
    check_shadowed_package(torch, "torch")

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
    else:
        if available:
            ok("torch.cuda", f"{torch.cuda.get_device_name(0)}")
        else:
            bad(
                "torch.cuda.is_available() is False",
                fix="No usable GPU. On a Jetson this usually means a mismatched "
                    "torch wheel.\n    " + JETSON_TORCH_FIX,
                fix_action=install_matched_torch(driver),
            )

    # Reached whether or not CUDA came up -- deliberately, and this used to be
    # a bare `return` above instead. torchvision is a separate axis from CUDA:
    # a driver-mismatched torch wheel takes torchvision with it, but the
    # is_available() failure is the only symptom that gets reported, so the
    # missing torchvision goes unmentioned here and instead resurfaces four
    # sections later as segment_anything, timm, efficientvit and ultralytics
    # each "not importable" -- one root cause wearing four disguises. Check it
    # here, where it can still be named.
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


TORCHVISION_CASCADE = (
    "Installed and fine -- it failed because torchvision underneath it is "
    "broken (reported under PyTorch / CUDA above). Reinstalling this package "
    "will not help; repair torchvision and re-run."
)

# Labels reported only as fallout from a broken torch or torchvision, so the
# summary can say so instead of presenting five equal-looking problems.
_cascaded: list[str] = []


def report_cascade(label: str, detail: str = TORCHVISION_CASCADE, severity=None) -> None:
    """Report `label` as fallout from a failure already reported, not as its
    own problem. No fix_action, deliberately: every fix on offer here would
    reinstall something that is already installed correctly."""
    (severity or bad)(label, detail)
    _cascaded.append(label)


def torchvision_cascade(exc: BaseException) -> bool:
    """True when `exc` is really the torchvision failure already reported.

    A broken torchvision does not announce itself. It surfaces as four
    separate-looking failures, each named after some other package:
    segment_anything imports torchvision.ops.boxes from its
    automatic_mask_generator, timm imports it from its data loaders,
    ultralytics declares it and looks up its metadata at import. Reported at
    face value, one root cause becomes four "not importable" lines, each
    suggesting a reinstall of a package that is already present and correct
    -- so following the advice fixes nothing and the real cause stays
    invisible. Attribute them instead.
    """
    if not _torchvision_failed:
        return False
    # ModuleNotFoundError and importlib.metadata.PackageNotFoundError (a
    # subclass of it) both carry the offending name; plain ImportError does
    # not, hence the message fallback.
    name = (getattr(exc, "name", "") or "").split(".")[0]
    return name == "torchvision" or "torchvision" in str(exc)


def cascade_root() -> str:
    """Name the failure everything else was attributed to.

    Not hardcoded to torchvision: it was, and a host whose torch could not load
    libcupti got six reports summarised as "one root cause: torchvision" --
    with torchvision never mentioned anywhere above, because the run never got
    far enough to check it.
    """
    if _torch_missing_lib:
        return f"torch cannot load {_torch_missing_lib}"
    return "torchvision"


def cascade_reason(exc: BaseException) -> str | None:
    """Explain `exc` as fallout from an already-reported failure, or None.

    Two root causes reach here. A broken torchvision, above. And a torch that
    cannot load a shared library it links -- which every importer of torch
    then re-raises verbatim, so `import transformers`, `import ultralytics`
    and `import efficientvit...sam` all fail with the identical
    "libcudss.so.0: cannot open shared object file" and read as three
    unrelated broken packages.
    """
    if _torch_missing_lib and _torch_missing_lib in str(exc):
        return (
            f"Installed and fine -- it imports torch, and torch cannot load "
            f"{_torch_missing_lib} (reported under PyTorch / CUDA above). "
            f"Reinstalling this package will not help; fix torch and re-run."
        )
    if torchvision_cascade(exc):
        return TORCHVISION_CASCADE
    return None


def check_torchvision(torch, driver: int | None = None) -> None:
    """NanoOWL imports torchvision.ops.roi_align, so this must actually work.

    A torchvision whose compiled extension was built against a different
    torch imports far enough to fail with "operator torchvision::nms does
    not exist" -- so importability alone proves nothing, and the ops get
    exercised below.
    """
    global _torchvision_failed
    torch_root = Path(torch.__file__).resolve().parent.parent

    try:
        import torchvision
    except ImportError:
        _torchvision_failed = True
        bad(
            "torchvision not importable",
            "NanoOWL needs torchvision.ops.roi_align; segment_anything, timm "
            "and ultralytics all import it too",
            MATCHED_PAIR_FIX,
            fix_action=install_matched_torch(driver),
        )
        return
    except RuntimeError as exc:
        _torchvision_failed = True
        bad(
            "torchvision failed to load",
            str(exc).splitlines()[0],
            "Its compiled ops were built against a different torch.\n    "
            + MATCHED_PAIR_FIX,
            fix_action=install_matched_torch(driver),
        )
        return

    ok("torchvision", f"{torchvision.__version__}  {torchvision.__file__}")
    check_shadowed_package(torchvision, "torchvision")

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
        _torchvision_failed = True
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
        reason = cascade_reason(exc)
        if reason:
            report_cascade("transformers import failed", reason)
            return
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
        reason = cascade_reason(exc)
        if reason:
            report_cascade("transformers import failed", reason)
            return
        bad(
            "transformers import failed",
            str(exc).splitlines()[0],
            fix_action=install_transformers,
        )
        return
    except Exception as exc:
        # OSError, not ImportError -- torchaudio's extension loader raises its
        # own type, so this used to escape to run_checks' catch-all and report
        # "check_nanoowl_runtime failed", naming no package and offering
        # nothing. transformers reaches torchaudio on the way to
        # OwlViTForObjectDetection; nothing here uses audio, but a companion
        # left over from a torch that has since been replaced fails the import
        # all the same.
        stale = STALE_LIBRARY.search(str(exc))
        if stale:
            package = package_owning(stale.group(1))
            bad(
                f"{package} was built against a different torch",
                stale.group(1),
                f"transformers imports {package}; that copy links the torch "
                f"that used to be installed.\n    "
                f"Install one matching the torch in use -- into the venv, "
                f"which shadows it:\n    "
                f"  python3 scripts/fix_torch.py",
                fix_action=run_fix_torch,
            )
        else:
            bad("transformers import failed", f"{type(exc).__name__}: {exc}")
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
        reason = cascade_reason(exc)
        if reason:
            report_cascade("nanosam.mobile_sam not importable", reason, warn)
            return
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
        reason = cascade_reason(exc)
        if reason:
            report_cascade("nanosam.mobile_sam import failed", reason, warn)
            return
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
            reason = cascade_reason(exc)
            if reason:
                report_cascade(f"{dep.module} not importable", reason)
            else:
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
        reason = cascade_reason(exc)
        if reason:
            report_cascade("efficientvit.models.efficientvit.sam import failed", reason)
        else:
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
        reason = cascade_reason(exc)
        if reason:
            # Without this branch the message below accuses upstream of a
            # rename it did not make: the import never got far enough to look
            # for the symbol at all.
            report_cascade("efficientvit.sam_model_zoo import failed", reason)
        else:
            bad(
                "efficientvit.sam_model_zoo.create_efficientvit_sam_model missing",
                f"{type(exc).__name__}: {str(exc).splitlines()[0]} -- the installed "
                "efficientvit exposes a different model-factory name than this app calls",
            )
    else:
        ok("efficientvit.sam_model_zoo", "create_efficientvit_sam_model importable")


def install_yoloworld() -> bool:
    # --no-deps is mandatory: ultralytics declares torch, torchvision AND
    # opencv-python. The first two would replace JetPack's builds; the third
    # would shadow JetPack's cv2, which is the one built with CUDA and
    # GStreamer. Its remaining dependencies are safe and installed by name.
    ok_pkg = run_fix(
        "install ultralytics (YOLO-World-S detector)",
        [sys.executable, "-m", "pip", "install", "ultralytics", "--no-deps"],
    )
    ok_deps = run_fix(
        "install ultralytics' safe dependencies (no torch, no opencv-python)",
        [sys.executable, "-m", "pip", "install", "filelock", "matplotlib", "pillow",
         "pyyaml", "requests", "psutil", "polars", "nvidia-ml-py"],
    )
    # Separate, and --no-deps, because ultralytics-thop declares an
    # unconstrained torch. Bundled into the line above it hands pip licence to
    # fetch a PyPI CUDA wheel over JetPack's build -- a repair that breaks the
    # host worse than the warning it was fixing. torch is already installed, so
    # --no-deps skips nothing here.
    ok_thop = run_fix(
        "install ultralytics-thop (--no-deps: it declares an unconstrained torch)",
        [sys.executable, "-m", "pip", "install", "ultralytics-thop", "--no-deps"],
    )
    return ok_pkg and ok_deps and ok_thop


def install_clip() -> bool:
    # CLIP is the one install here built from an sdist, and it carries a
    # MANIFEST.in -- which is what sends setuptools through prune_file_list()
    # into canonicalize_version(strip_trailing_zero=...), a keyword that only
    # exists from packaging 22.0. JetPack's Ubuntu 22.04 ships packaging 21.3
    # and setuptools >=71 prefers the installed copy over its vendored one, so
    # this install fails on a stock image every time, with a traceback that
    # never mentions CLIP. Shadow the old copy in the venv first.
    run_fix(
        "ensure packaging is new enough to build an sdist "
        "(JetPack ships 21.3; setuptools needs >=22.0)",
        [sys.executable, "-m", "pip", "install", "packaging>=24.2"],
    )
    ok_pkg = run_fix(
        "install CLIP (encodes YOLO-World's prompts)",
        [sys.executable, "-m", "pip", "install",
         "git+https://github.com/ultralytics/CLIP.git", "--no-deps"],
    )
    ok_deps = run_fix(
        "install CLIP's own dependencies",
        [sys.executable, "-m", "pip", "install", "ftfy", "regex", "tqdm"],
    )
    return ok_pkg and ok_deps


def check_yoloworld_runtime() -> None:
    """YOLO-World-S, the optional second detector.

    Warnings rather than problems: nothing in the default NanoOWL pairings
    touches any of this, so a host that never selects a YOLO-World pairing
    is completely fine without it.
    """
    section("YOLO-World-S runtime (optional second detector)")
    try:
        from ultralytics import YOLOWorld  # noqa: F401
    except ImportError as exc:
        reason = cascade_reason(exc)
        if reason:
            # ultralytics resolves its declared dependencies at import time, so
            # a torchvision with no metadata raises PackageNotFoundError from
            # inside `import ultralytics` -- which reads as "ultralytics is not
            # installed" when it is installed and undamaged.
            report_cascade("ultralytics not importable", reason, warn)
            return
        warn(
            "ultralytics not importable",
            f"{type(exc).__name__}: {str(exc).splitlines()[0]} -- only needed for "
            "the YOLO-World pairings",
            "pip install ultralytics --no-deps   # --no-deps: it declares torch, "
            "torchvision and opencv-python",
            fix_action=install_yoloworld,
        )
        return
    ok("ultralytics", "YOLOWorld importable")

    # The important one. Without clip, ultralytics runs `pip install
    # git+.../CLIP.git` itself the first time set_classes() is called --
    # no --no-deps, and CLIP declares torch and torchvision. That is PyPI
    # torch landing on JetPack's build in the middle of a benchmark.
    try:
        import clip  # noqa: F401
    except ImportError:
        warn(
            "clip not importable",
            "YOLO-World encodes its prompts with CLIP. Left missing, ultralytics "
            "pip-installs it MID-RUN without --no-deps, and CLIP declares torch "
            "and torchvision -- so the first YOLO-World run would replace "
            "JetPack's torch with a PyPI wheel",
            "pip install git+https://github.com/ultralytics/CLIP.git --no-deps"
            " && pip install ftfy regex tqdm",
            fix_action=install_clip,
        )
    else:
        ok("clip", "prompt encoder present, so ultralytics will not self-install it")


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
        reason = cascade_reason(exc)
        if reason:
            report_cascade("torch2trt failed to import", reason)
            return
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


def install_ffmpeg() -> bool:
    return run_fix(
        "install ffmpeg (H.264 encoder for the comparison video)",
        ["apt-get", "install", "-y", "ffmpeg"],
        use_sudo=True,
    )


def opencv_can_write_h264() -> bool:
    """Can this OpenCV build write H.264 itself?

    Distro OpenCV usually links libx264 and can; the pip `opencv-python`
    wheels ship without an H.264 encoder for licensing reasons and cannot.
    VideoWriter.isOpened() is the reliable signal either way.
    """
    import tempfile

    try:
        import cv2
    except ImportError:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.mp4")
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), 30.0, (64, 64))
        except Exception:
            return False
        opened = writer.isOpened()
        writer.release()
        return bool(opened)


def check_video_encoder() -> None:
    """The comparison video has to be playable, not merely written.

    OpenCV's default mp4v codec produces MPEG-4 Part 2: a valid .mp4 that
    VLC plays and no browser can decode, with no error anywhere. Since the
    video is viewed in the browser UI, a missing H.264 encoder makes the
    deliverable silently useless -- worth catching here rather than after a
    full benchmark run.
    """
    section("Comparison video encoder")
    from shutil import which

    try:
        # Imported rather than reimplemented so the doctor probes for exactly
        # the encoders the renderer will accept. Pulls in cv2, which
        # check_opencv above has already reported on if it is missing.
        from benchmark.video import _ffmpeg_h264_encoder
    except ImportError as exc:
        warn("Could not probe the video encoder", f"{type(exc).__name__}: {exc}")
        return

    binary = which("ffmpeg")
    if binary:
        encoder = _ffmpeg_h264_encoder(binary)
        if encoder:
            ok("ffmpeg", f"{binary}  (H.264 via {encoder})")
            return
        warn("ffmpeg has no H.264 encoder", f"{binary} — checked libx264 and the NVIDIA ones")
    if opencv_can_write_h264():
        ok("OpenCV avc1", "can write H.264 directly; ffmpeg not required")
        return
    bad(
        "No H.264 encoder",
        "the comparison video will be written as mp4v, which plays in VLC "
        "but shows a blank player in any browser",
        "sudo apt install ffmpeg",
        fix_action=install_ffmpeg,
    )


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
    check_yoloworld_runtime,
    check_torch2trt,
    check_onnx_export,
    check_artifacts,
    check_video_encoder,
    check_clocks,
)


def run_checks() -> None:
    # Reset alongside _problems/_warnings so the --fix re-check starts clean:
    # left set from the first pass, a repaired torchvision would still be
    # blamed for every later failure.
    global _torchvision_failed, _torch_missing_lib
    _torchvision_failed = False
    _torch_missing_lib = None
    _cascaded.clear()

    for check in CHECKS:
        try:
            check()
        except Exception as exc:  # a broken check must not hide the others
            bad(f"{check.__name__} failed", f"{type(exc).__name__}: {exc}")
            # ...and the one line above must not hide the cause. A check that
            # crashes instead of reporting has hit something unanticipated by
            # definition, so the frames are the only information there is --
            # "check_nanoowl_runtime failed: OSError" says nothing about which
            # import reached the library that would not load.
            for line in traceback.format_exc().strip().splitlines()[-7:]:
                print(f"    {DIM}{line}{RESET}")


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
        # In a fresh interpreter, not in this one. Several repairs replace
        # modules this process imported minutes ago -- torch above all -- and
        # Python will not reload them. An in-process re-check therefore reports
        # the state from *before* the repair, indistinguishable from a repair
        # that did nothing, which is precisely the case it needs to tell apart.
        return subprocess.run([sys.executable, str(Path(__file__).resolve())]).returncode

    print()
    if _problems:
        print(f"{RED}{len(_problems)} problem(s) block a real benchmark:{RESET}")
        for problem in _problems:
            marker = "  ·"
            print(f"{marker} {problem}"
                  + (f"  {DIM}(fallout){RESET}" if problem in _cascaded else ""))
        # Counted against _problems, not against _cascaded: the optional
        # runtimes report their fallout as warnings, and those are not in the
        # list this line is describing.
        cascaded_here = [p for p in _problems if p in _cascaded]
        if cascaded_here:
            print(f"\n{DIM}{len(cascaded_here)} of these are one root cause: "
                  f"{cascade_root()}. Repair it first -- the rest should clear "
                  f"on their own.{RESET}")
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
