#!/usr/bin/env python3
"""Make `import torch` work in this environment, or say precisely why it can't.

    python3 scripts/fix_torch.py            # resolve, and report what it did
    python3 scripts/fix_torch.py --dry-run  # diagnose only, change nothing

One venv, two environments. This project's `.venv` lives inside the working
tree, so the *same* venv is used on the Jetson host and inside a
Singularity/Docker image bind-mounted over that tree -- and the two do not
agree about torch. A driver-matched wheel installed into the venv to repair
the host links shared libraries a container image need not carry, and the
container then cannot import torch at all:

    ImportError: libcudss.so.0: cannot open shared object file

Nothing is corrupt and nothing was installed wrongly. The wheel is simply for
the other environment. Installing yet another torch is the obvious move and
the wrong one -- it fixes whichever environment happens to be running and
re-breaks the other on the next switch.

So this does not pick a torch. It resolves one: probe the copies already
present, in order of least damage, and stop at the first that imports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)

# The dynamic loader names the first library it cannot find and stops, so this
# yields one name per probe even when several are absent.
MISSING_LIB = re.compile(r"(lib[\w.+-]+\.so[\d.]*): cannot open shared object file")

# Shared libraries a CUDA-12 torch build may link that a JetPack image or a
# container need not carry, mapped to the pip package that redistributes them.
# Only libraries NVIDIA actually publishes as aarch64 wheels are listed --
# offering to install something with no aarch64 artifact would just trade this
# error for a resolver error.
LIBRARY_WHEELS: dict[str, str] = {
    "libcudss.so": "nvidia-cudss-cu12",
    "libcusparseLt.so": "nvidia-cusparselt-cu12",
    "libcudnn.so": "nvidia-cudnn-cu12",
    "libnccl.so": "nvidia-nccl-cu12",
}

_PROBE = r"""
import json
out = {"ok": False}
try:
    import torch
except BaseException as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
else:
    out.update(ok=True, version=torch.__version__, path=torch.__file__,
               cuda_build=torch.version.cuda)
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
    except BaseException as exc:
        out["cuda_available"] = False
        out["cuda_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
print("<<<PROBE>>>" + json.dumps(out))
"""


def say(mark: str, colour: str, text: str, detail: str = "") -> None:
    print(f"  {colour}{mark}{RESET} {text}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


ok = lambda text, detail="": say("✓", GREEN, text, detail)          # noqa: E731
warn = lambda text, detail="": say("!", YELLOW, text, detail)        # noqa: E731
bad = lambda text, detail="": say("✗", RED, text, detail)            # noqa: E731


@dataclass
class Probe:
    """The result of actually importing torch, in a fresh interpreter."""

    ok: bool = False
    version: str = ""
    path: str = ""
    cuda_build: str | None = None
    cuda_available: bool = False
    error: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def missing_lib(self) -> str | None:
        match = MISSING_LIB.search(self.error)
        return match.group(1) if match else None

    def describe(self) -> str:
        if not self.ok:
            return self.error or "import failed"
        cuda = "CUDA ok" if self.cuda_available else "no usable CUDA"
        return f"{self.version} (built for CUDA {self.cuda_build}, {cuda})  {self.path}"


def probe_torch(python: str) -> Probe:
    """Import torch under `python` and report what happened.

    A subprocess every time, deliberately: this gets called after installs and
    uninstalls that change which torch is on disk, and a process that has
    already imported torch will not see any of it.
    """
    try:
        result = subprocess.run(
            [python, "-c", _PROBE], capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Probe(error=f"{type(exc).__name__}: {exc}")

    for line in result.stdout.splitlines():
        if line.startswith("<<<PROBE>>>"):
            data = json.loads(line[len("<<<PROBE>>>"):])
            return Probe(
                ok=data.get("ok", False),
                version=data.get("version", ""),
                path=data.get("path", ""),
                cuda_build=data.get("cuda_build"),
                cuda_available=data.get("cuda_available", False),
                error=data.get("error", ""),
                raw=data,
            )
    return Probe(error=(result.stderr.strip().splitlines() or ["no output"])[-1])


def base_python() -> str:
    """The interpreter this venv was built from."""
    return getattr(sys, "_base_executable", None) or str(
        Path(sys.base_prefix) / "bin" / "python3"
    )


def venv_package(name: str) -> Path | None:
    """A package installed into the venv itself, as opposed to inherited."""
    if sys.prefix == sys.base_prefix:
        return None
    candidate = Path(sysconfig.get_paths()["purelib"]) / name
    return candidate if candidate.is_dir() else None


def pip(*args: str) -> bool:
    command = [sys.executable, "-m", "pip", *args]
    print(f"    {DIM}{' '.join(command)}{RESET}")
    return subprocess.run(command).returncode == 0


def drop_venv_torch(probe: Probe, dry_run: bool) -> bool:
    """Remove the venv's own torch when the inherited one works.

    This is the repair for the host-venv-in-a-container case, and it is first
    because it is the only one that *removes* the mismatch rather than piling
    another layer on top of it. The venv is ours to edit; the image is not.

    torchvision goes with it, always. They are a matched pair -- leaving a venv
    torchvision shadowing an inherited torch is how you get "operator
    torchvision::nms does not exist" instead of a clean import error.
    """
    if venv_package("torch") is None:
        return False

    inherited = probe_torch(base_python())
    if not inherited.ok:
        return False

    print(f"\n{BOLD}--> the venv's torch is for a different environment; "
          f"using the inherited one{RESET}")
    ok("inherited torch imports cleanly", inherited.describe())
    if dry_run:
        warn("--dry-run: would uninstall torch and torchvision from the venv")
        return False
    return pip("uninstall", "-y", "torch", "torchvision")


def find_library(name: str) -> Path | None:
    """Locate a just-installed NVIDIA .so inside site-packages."""
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    return next(iter(sorted(root.glob(f"*/lib/{name}"))), None) if root.is_dir() else None


def supply_missing_library(probe: Probe, dry_run: bool) -> bool:
    """Install the wheel carrying the missing .so and put it where torch looks.

    torch's `_C` extension carries an RPATH that includes `$ORIGIN/lib` -- the
    `torch/lib` directory its own shared objects live in -- so a symlink there
    is resolved by the loader without LD_LIBRARY_PATH, and without depending on
    which `$ORIGIN/../../nvidia/*/lib` entries a particular build happens to
    have baked in. That matters here: these wheels install under
    `nvidia/cu12/lib`, and a build compiled against a different CUDA minor
    looks in a differently-named sibling.
    """
    missing = probe.missing_lib
    if missing is None:
        return False

    stem = missing.split(".so")[0] + ".so"
    package = LIBRARY_WHEELS.get(stem)
    if package is None:
        bad(f"no pip wheel known to carry {missing}",
            "not one of: " + ", ".join(sorted(LIBRARY_WHEELS.values())))
        return False

    torch_dir = venv_package("torch") or (
        Path(probe.path).parent if probe.path else None)
    if torch_dir is None or not (torch_dir / "lib").is_dir():
        bad(f"cannot find torch/lib to place {missing} into")
        return False

    print(f"\n{BOLD}--> supplying {missing} from {package}{RESET}")
    if dry_run:
        warn(f"--dry-run: would install {package} and link {missing} "
             f"into {torch_dir / 'lib'}")
        return False

    # --no-deps: these wheels declare nvidia-* siblings that would drag in a
    # second CUDA stack, and one mismatched torch is already the problem here.
    if not pip("install", package, "--no-deps"):
        return False

    source = find_library(missing)
    if source is None:
        bad(f"{package} installed but {missing} is not in it")
        return False

    target = torch_dir / "lib" / missing
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    ok(f"linked {missing}", f"{target} -> {source}")
    return True


STRATEGIES = (
    ("use the environment's own torch", drop_venv_torch),
    ("supply the missing library", supply_missing_library),
)


def resolve(dry_run: bool = False) -> Probe:
    probe = probe_torch(sys.executable)
    if probe.ok:
        ok("torch imports", probe.describe())
        if not probe.cuda_available:
            warn("torch has no usable CUDA",
                 probe.raw.get("cuda_error", "torch.cuda.is_available() is False"))
        return probe

    bad("torch does not import", probe.error)
    if probe.missing_lib:
        print(f"    {DIM}Missing shared library: {probe.missing_lib}. torch itself "
              f"is installed and intact -- it was built for an environment that "
              f"has this library.{RESET}")

    for label, strategy in STRATEGIES:
        try:
            changed = strategy(probe, dry_run)
        except Exception as exc:  # a failed repair must not hide the diagnosis
            bad(f"{label} failed", f"{type(exc).__name__}: {exc}")
            continue
        if not changed:
            continue
        probe = probe_torch(sys.executable)
        if probe.ok:
            ok("torch imports", probe.describe())
            if not probe.cuda_available:
                warn("torch has no usable CUDA",
                     probe.raw.get("cuda_error", "torch.cuda.is_available() is False"))
            return probe
        bad("still not importing after that", probe.error)

    return probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="diagnose only; install and uninstall nothing")
    args = parser.parse_args()

    print(f"{BOLD}torch{RESET}")
    probe = resolve(dry_run=args.dry_run)
    if probe.ok:
        return 0

    print()
    print(f"{RED}Could not make torch importable here.{RESET}")
    print(f"{DIM}Install a build matching this environment, e.g. for "
          f"JetPack 6.x / CUDA 12.6:{RESET}")
    print(f"{DIM}  pip install --no-cache-dir --force-reinstall --no-deps "
          f"--index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
