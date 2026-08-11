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

# Where a CUDA library already on this machine is likely to be sitting.
#
# CUPTI is the reason this list exists and is checked before any download.
# JetPack does ship libcupti.so.12 -- in /usr/local/cuda-*/extras/CUPTI/lib64,
# which is not on the loader path and is not in ld.so.conf. So a torch that
# links it fails on a host that has it, and "install the missing library" is
# the wrong instinct: the right one is already here, matched to this CUDA, and
# just needs to be findable.
LIBRARY_SEARCH_GLOBS: tuple[str, ...] = (
    "/usr/local/cuda*/extras/CUPTI/lib64/{name}",
    "/usr/local/cuda*/targets/*/lib/{name}",
    "/usr/local/cuda*/lib64/{name}",
    "/usr/local/cuda*/lib/{name}",
    "/usr/lib/aarch64-linux-gnu/{name}",
    "/usr/lib/aarch64-linux-gnu/nvidia/{name}",
    "/opt/nvidia/*/lib*/{name}",
)

# Shared libraries a CUDA-12 torch build may link that a JetPack image or a
# container need not carry, mapped to the pip package that redistributes them.
# The fallback for when the search above comes up empty. Only libraries NVIDIA
# actually publishes as aarch64 wheels are listed -- offering to install
# something with no aarch64 artifact just trades a loader error for a resolver
# error.
LIBRARY_WHEELS: dict[str, str] = {
    "libcudss.so": "nvidia-cudss-cu12",
    "libcusparseLt.so": "nvidia-cusparselt-cu12",
    "libcudnn.so": "nvidia-cudnn-cu12",
    "libnccl.so": "nvidia-nccl-cu12",
    "libcupti.so": "nvidia-cuda-cupti-cu12",
    "libcublas.so": "nvidia-cublas-cu12",
    "libcublasLt.so": "nvidia-cublas-cu12",
    "libcufft.so": "nvidia-cufft-cu12",
    "libcurand.so": "nvidia-curand-cu12",
    "libcusolver.so": "nvidia-cusolver-cu12",
    "libcusparse.so": "nvidia-cusparse-cu12",
    "libnvJitLink.so": "nvidia-nvjitlink-cu12",
    "libcudart.so": "nvidia-cuda-runtime-cu12",
    "libnvrtc.so": "nvidia-cuda-nvrtc-cu12",
    "libnvToolsExt.so": "nvidia-nvtx-cu12",
}

# One round per missing library. The loader reports them one at a time, so a
# torch missing three needs three passes -- but a chain longer than this is
# not a missing-library problem and looping is not going to find that out.
MAX_ROUNDS = 8

# torch's companions ship compiled extensions linked against one exact torch
# build, so a torch that was replaced leaves them loading against the wrong
# one. Neither is used by this project directly -- torchvision is reached
# through NanoOWL, torchaudio only because transformers touches it on the way
# to OwlViTForObjectDetection -- but a stale one fails the import all the same.
COMPANIONS = ("torchvision", "torchaudio")

# What "stale" looks like, as opposed to "absent". Absent is fine and must be
# left alone: transformers checks whether torchaudio is available and copes
# when it is not. Only a companion that is present and will not load is a
# problem, and only that shape should trigger a repair.
STALE_COMPANION = re.compile(
    r"Could not load this library"                  # torchaudio's own loader
    r"|undefined symbol"                            # ABI drift between builds
    r"|operator torchvision::\w+ does not exist"    # torchvision's op registry
    r"|cannot open shared object file"              # links a lib torch dropped
)

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

_COMPANION_PROBE = r"""
import json, sys
out = {"ok": False}
try:
    module = __import__(sys.argv[1])
except BaseException as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
else:
    out.update(ok=True, version=getattr(module, "__version__", "?"),
               path=getattr(module, "__file__", "") or "")
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
    return _run_probe([python, "-c", _PROBE])


def probe_module(python: str, name: str) -> Probe:
    return _run_probe([python, "-c", _COMPANION_PROBE, name])


def _run_probe(command: list[str]) -> Probe:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300
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


# Overridable so the search can be pointed at a fixture tree in tests without
# patching pathlib itself.
SEARCH_ROOT = Path("/")


def find_system_library(name: str, root: Path | None = None) -> Path | None:
    """Locate `name` somewhere on this machine, outside the loader path.

    Globbed against a fixed list rather than walked: a full filesystem search
    on a Jetson's eMMC costs minutes, and every real location is known.
    """
    base = root or SEARCH_ROOT
    for pattern in LIBRARY_SEARCH_GLOBS:
        for match in sorted(base.glob(pattern.format(name=name).lstrip("/"))):
            if match.is_file():
                return match
    return None


def link_into_torch(missing: str, source: Path, torch_dir: Path) -> bool:
    """Put `source` where torch's own loader will find it.

    torch's `_C` extension carries an RPATH including `$ORIGIN/lib` -- the
    `torch/lib` directory its own shared objects live in -- so a symlink there
    resolves without LD_LIBRARY_PATH, which matters because the server and the
    engine builds are launched separately and would each need the variable set.
    It also avoids depending on which `$ORIGIN/../../nvidia/*/lib` entries a
    particular build baked in: the pip wheels install under `nvidia/cu12/lib`,
    and a build compiled against another CUDA minor looks in a differently
    named sibling.
    """
    target = torch_dir / "lib" / missing
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    ok(f"linked {missing}", f"{target} -> {source}")
    return True


def torch_lib_dir(probe: Probe) -> Path | None:
    torch_dir = venv_package("torch") or (
        Path(probe.path).parent if probe.path else None)
    return torch_dir if torch_dir and (torch_dir / "lib").is_dir() else None


def link_local_library(probe: Probe, dry_run: bool) -> bool:
    """Use the copy already on this machine before downloading another.

    Preferred over the pip wheel because it is the build matched to this
    machine's CUDA, costs nothing, and works on a bench with no route out.
    """
    missing = probe.missing_lib
    if missing is None:
        return False

    source = find_system_library(missing)
    if source is None:
        return False

    torch_dir = torch_lib_dir(probe)
    if torch_dir is None:
        bad(f"cannot find torch/lib to place {missing} into")
        return False

    print(f"\n{BOLD}--> {missing} is already on this machine, just not on the "
          f"loader path{RESET}")
    ok("found", str(source))
    if dry_run:
        warn(f"--dry-run: would link it into {torch_dir / 'lib'}")
        return False
    return link_into_torch(missing, source, torch_dir)


def supply_missing_library(probe: Probe, dry_run: bool) -> bool:
    """Install the wheel carrying the missing .so, when this machine has none."""
    missing = probe.missing_lib
    if missing is None:
        return False

    stem = missing.split(".so")[0] + ".so"
    package = LIBRARY_WHEELS.get(stem)
    if package is None:
        bad(f"no pip wheel known to carry {missing}",
            "not one of: " + ", ".join(sorted(set(LIBRARY_WHEELS.values()))))
        return False

    torch_dir = torch_lib_dir(probe)
    if torch_dir is None:
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
    return link_into_torch(missing, source, torch_dir)


STRATEGIES = (
    ("use the environment's own torch", drop_venv_torch),
    ("link the library already on this machine", link_local_library),
    ("supply the missing library", supply_missing_library),
)


def report_success(probe: Probe) -> Probe:
    ok("torch imports", probe.describe())
    if not probe.cuda_available:
        warn("torch has no usable CUDA",
             probe.raw.get("cuda_error", "torch.cuda.is_available() is False"))
    return probe


def resolve(dry_run: bool = False) -> Probe:
    """Keep going until torch imports, nothing helps, or we start repeating.

    Looped, not a single pass: the dynamic loader names the *first* library it
    cannot find and stops, so a torch missing three of them reports one, and
    supplying it just reveals the next. Fixing one and reporting "still not
    importing" would be technically true and useless -- it is the same problem,
    one library further along.
    """
    probe = probe_torch(sys.executable)
    handled: set[str] = set()

    for _ in range(MAX_ROUNDS):
        if probe.ok:
            return report_success(probe)

        bad("torch does not import", probe.error)
        missing = probe.missing_lib
        if missing:
            print(f"    {DIM}Missing shared library: {missing}. torch itself is "
                  f"installed and intact -- it was built for an environment that "
                  f"has this library.{RESET}")
            if missing in handled:
                # Supplied once and still unresolved: linking it did not take,
                # and going round again would just relink the same file.
                bad(f"{missing} is still missing after being supplied",
                    "the copy that was linked is not one this torch can use")
                break
            handled.add(missing)

        changed = False
        for label, strategy in STRATEGIES:
            try:
                changed = strategy(probe, dry_run)
            except Exception as exc:  # a failed repair must not hide the diagnosis
                bad(f"{label} failed", f"{type(exc).__name__}: {exc}")
                continue
            if changed:
                break
        if not changed:
            break
        probe = probe_torch(sys.executable)

    return probe


def wheel_index(cuda_build: str | None) -> str:
    """The index that published the torch we ended up with.

    Derived from torch's own CUDA version rather than the driver's, so a
    companion is matched to torch, which is what its compiled extension
    actually links against.
    """
    tag = f"cu{cuda_build.replace('.', '')}" if cuda_build else "cu126"
    return os.environ.get(
        "TORCH_INDEX_URL", f"https://pypi.jetson-ai-lab.io/jp6/{tag}")


VERSIONS_LINE = re.compile(r"(?:Available versions|from versions):\s*([^)\n]+)")


def version_key(version: str) -> tuple:
    parts = version.split("+")[0].split(".")
    return tuple(int(p) if p.isdigit() else 0 for p in parts[:3])


def series_of(version: str) -> str:
    """'2.10.0+cu126' -> '2.10'. What a companion has to match."""
    return ".".join(version.split("+")[0].split(".")[:2])


def available_versions(package: str, index: str) -> list[str]:
    """Versions of `package` the index publishes, newest first.

    `pip index versions` is the documented route and is flagged experimental,
    so there is a fallback: the resolver's own failure carries the same list
    ("from versions: 2.8.0, 2.9.1, 2.10.0"). That message is what revealed the
    problem in the first place -- the Jetson index publishes torch 2.11.0 but
    no torchaudio past 2.10.0, so "install the companion matching torch" can
    never succeed and the version to move is torch's.
    """
    for command in (
        ["index", "versions", package, "--index-url", index],
        ["install", f"{package}==99999", "--no-deps", "--index-url", index],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "pip", *command],
            capture_output=True, text=True,
        )
        match = VERSIONS_LINE.search(result.stdout + result.stderr)
        if match:
            found = [v.strip() for v in match.group(1).split(",") if v.strip()]
            # Sorted here rather than trusted: `pip index versions` lists
            # newest first, the error message lists oldest first.
            return sorted(found, key=version_key, reverse=True)
    return []


def torchvision_series(torch_series: str) -> str:
    """torch 2.11 -> torchvision 0.26. Fifteen behind, since torch 2.0."""
    major, minor = torch_series.split(".")
    return f"0.{int(minor) + 15}" if major == "2" else ""


def align_trio(torch_probe: Probe, dry_run: bool = False) -> bool:
    """Move torch, torchvision and torchaudio to one series the index has all of.

    The last resort, and the only move left when a companion cannot be matched
    to the installed torch because the index simply does not publish one. That
    is not a broken environment -- the Jetson index tracks torch ahead of
    torchaudio -- so no amount of installing companions resolves it. torch is
    the version that has to give.

    Nothing here uses audio. torchaudio is in the way only because
    transformers/audio_utils.py does a module-level `import torchaudio` behind
    is_torchaudio_available(), which tests for metadata rather than for a
    library that loads -- so a stale copy in system dist-packages, which a
    read-only container image will not let you remove, fails the import of
    OwlViTForObjectDetection.
    """
    index = wheel_index(torch_probe.cuda_build)
    print(f"\n{BOLD}--> no companion matches torch {torch_probe.version}; "
          f"moving the trio to a series the index has all of{RESET}")

    have = {name: available_versions(name, index)
            for name in ("torch", "torchvision", "torchaudio")}
    for name, versions in have.items():
        ok(f"{name} on the index", ", ".join(versions[:6]) or "none found")
    if not all(have.values()):
        bad("could not read the index", f"tried {index}")
        return False

    audio_series = {series_of(v) for v in have["torchaudio"]}
    vision_series = {series_of(v) for v in have["torchvision"]}

    chosen = None
    for version in have["torch"]:
        series = series_of(version)
        if series in audio_series and torchvision_series(series) in vision_series:
            chosen = series
            break
    if chosen is None:
        bad("no series has all three", "torch, torchvision and torchaudio do "
                                      "not overlap on this index")
        return False

    pins = [f"torch=={chosen}.*", f"torchvision=={torchvision_series(chosen)}.*",
            f"torchaudio=={chosen}.*"]
    if series_of(torch_probe.version) == chosen:
        ok(f"torch is already on {chosen}", "installing the companions only")
        pins = pins[1:]
    else:
        warn(f"torch {torch_probe.version} -> {chosen}.*",
             f"the newest series this index publishes a torchaudio for")

    if dry_run:
        warn("--dry-run: would install " + " ".join(pins))
        return False

    # --force-reinstall so a version already present but wrong is replaced
    # rather than considered satisfied; --no-deps so this stays three wheels.
    if not pip("install", "--no-cache-dir", "--force-reinstall", "--no-deps",
               "--index-url", index, *pins):
        return False

    after = probe_torch(sys.executable)
    if not after.ok:
        bad("torch does not import after realignment", after.error)
        return False
    ok("torch imports", after.describe())
    return True


def repair_companions(torch_probe: Probe, dry_run: bool = False) -> bool:
    """Replace companions left over from a torch that is no longer here.

    Runs only once torch itself imports, because until then every companion
    fails for torch's reason and nothing can be told apart.

    Absent is not broken. transformers asks whether torchaudio is available
    and copes when it is not, so a missing companion is left missing --
    installing one nothing asked for is how you acquire the next mismatch.
    Only a companion that is present and refuses to load gets touched.

    Returns the companions still broken afterwards, which is the signal to
    realign the whole trio instead.
    """
    if not torch_probe.ok:
        return []

    # torch 2.11.0 -> torchvision/torchaudio 2.11.*: the trio is released in
    # lockstep and a companion only loads against its own torch minor.
    series = series_of(torch_probe.version)
    unresolved: list[str] = []

    for name in COMPANIONS:
        probe = probe_module(sys.executable, name)
        if probe.ok:
            continue
        if not STALE_COMPANION.search(probe.error):
            continue  # absent, or broken for a reason installing cannot fix

        print(f"\n{BOLD}--> {name} was built against a torch that is no longer "
              f"installed{RESET}")
        bad(f"{name} does not load", probe.error)
        index = wheel_index(torch_probe.cuda_build)
        if dry_run:
            warn(f"--dry-run: would install {name}=={series}.* from {index}")
            unresolved.append(name)
            continue

        # Into the venv, where it shadows the stale copy -- which usually sits
        # in system dist-packages that we neither own nor, inside a read-only
        # container image, can write to.
        if not pip("install", "--no-cache-dir", "--no-deps",
                   "--index-url", index, f"{name}=={series}.*"):
            warn(f"the index publishes no {name} {series}.*",
                 "torch is ahead of its companions here; realigning below")
            unresolved.append(name)
            continue

        after = probe_module(sys.executable, name)
        if after.ok:
            ok(f"{name} loads", f"{after.version}  {after.path}")
        else:
            bad(f"{name} still does not load", after.error)
            unresolved.append(name)

    return unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="diagnose only; install and uninstall nothing")
    args = parser.parse_args()

    print(f"{BOLD}torch{RESET}")
    probe = resolve(dry_run=args.dry_run)
    if probe.ok:
        unresolved = repair_companions(probe, dry_run=args.dry_run)
        if unresolved and not args.dry_run and align_trio(probe):
            # Realigning moved torch, so the companions are re-checked against
            # the version that is actually installed now.
            repair_companions(probe_torch(sys.executable))
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
