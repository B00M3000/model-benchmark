"""Doctor's torchvision root-cause attribution.

A broken torchvision is the one Jetson failure that lies about itself: it
surfaces named after four *other* packages, each of which is installed and
undamaged. These tests pin the behaviour that tells the difference, because
getting it wrong sends you off reinstalling things that were never broken.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_doctor():
    """Import scripts/doctor.py fresh, without running any checks.

    Not importable as `scripts.doctor` -- scripts/ is not a package -- and a
    fresh module each time keeps the module-level result lists from leaking
    between tests.
    """
    spec = importlib.util.spec_from_file_location(
        "doctor_under_test", REPO_ROOT / "scripts" / "doctor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def doctor():
    module = load_doctor()
    yield module
    sys.modules.pop("doctor_under_test", None)


def test_no_cascade_claimed_while_torchvision_is_fine(doctor):
    """The whole mechanism must stay off unless torchvision actually failed.

    Otherwise a genuinely missing segment_anything gets excused as someone
    else's fault and its install hint never prints.
    """
    assert doctor._torchvision_failed is False
    assert doctor.torchvision_cascade(ModuleNotFoundError(
        "No module named 'torchvision'", name="torchvision")) is False


@pytest.mark.parametrize(
    "exc",
    [
        # segment_anything and timm: raised from the failing import itself.
        ModuleNotFoundError("No module named 'torchvision'", name="torchvision"),
        ModuleNotFoundError(
            "No module named 'torchvision.ops'", name="torchvision.ops"),
        # ultralytics: resolves declared dependencies at import time, so a
        # torchvision with no metadata fails here instead.
        PackageNotFoundError("torchvision"),
        # A plain ImportError carries no .name at all -- only the message.
        ImportError("cannot import name 'nms' from 'torchvision.ops'"),
    ],
)
def test_torchvision_shaped_failures_are_attributed(doctor, exc):
    doctor._torchvision_failed = True
    assert doctor.torchvision_cascade(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ModuleNotFoundError("No module named 'pycocotools'", name="pycocotools"),
        ModuleNotFoundError("No module named 'omegaconf'", name="omegaconf"),
        ImportError("cannot import name 'create_efficientvit_sam_model'"),
    ],
)
def test_unrelated_failures_keep_their_own_diagnosis(doctor, exc):
    """Even with torchvision down, a genuinely missing package is still that.

    Blanket-attributing everything once torchvision breaks would be just as
    wrong in the other direction -- pycocotools really can be absent at the
    same time, and it needs its own install hint.
    """
    doctor._torchvision_failed = True
    assert doctor.torchvision_cascade(exc) is False


def test_cascade_reports_offer_no_fix(doctor, capsys):
    """The fix list drives --fix. A cascade entry must never enter it: every
    repair on offer would reinstall a package that is already correct, and
    --fix would report success having changed nothing."""
    doctor._torchvision_failed = True
    doctor.report_cascade("segment_anything not importable")

    assert doctor._fixes == []
    assert doctor._cascaded == ["segment_anything not importable"]
    assert "segment_anything not importable" in doctor._problems
    out = capsys.readouterr().out
    assert "will not help" in out


def test_cascade_can_be_reported_as_a_warning(doctor):
    """The optional runtimes (nanosam.mobile_sam, ultralytics) must not turn
    into hard blockers just because torchvision took them down with it."""
    doctor._torchvision_failed = True
    doctor.report_cascade("ultralytics not importable", doctor.warn)

    assert doctor._problems == []
    assert doctor._warnings == ["ultralytics not importable"]
    assert doctor._cascaded == ["ultralytics not importable"]


def test_run_checks_clears_cascade_state(doctor, monkeypatch):
    """--fix re-runs every check in the same process. Stale state would blame
    a freshly repaired torchvision for the next pass's failures."""
    doctor._torchvision_failed = True
    doctor._cascaded.append("stale")
    monkeypatch.setattr(doctor, "CHECKS", [])

    doctor.run_checks()

    assert doctor._torchvision_failed is False
    assert doctor._cascaded == []


def make_package(site_packages: Path, name: str, version: str) -> Path:
    """A package on disk, complete enough for installed_version() to read."""
    pkg = site_packages / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "version.py").write_text(f"__version__ = '{version}'\n")
    return pkg


@pytest.fixture
def venv(doctor, tmp_path, monkeypatch):
    """Pose as a venv whose site-packages is tmp_path/venv-sp."""
    site_packages = tmp_path / "venv-sp"
    site_packages.mkdir()
    monkeypatch.setattr(doctor.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(doctor.sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(
        doctor.sysconfig, "get_paths", lambda: {"purelib": str(site_packages)})
    monkeypatch.delenv("PYTHONPATH", raising=False)
    return site_packages


def fake_module(package_dir: Path):
    module = type(sys)(package_dir.name)
    module.__file__ = str(package_dir / "__init__.py")
    return module


def test_shadowed_venv_torch_is_reported(doctor, venv, tmp_path):
    """The regression --fix walked into: it installed 2.11.0 into the venv,
    pip said "Successfully installed", and `import torch` kept resolving to
    /usr/local. Reported identically to before the repair, it reads as a
    repair that did nothing."""
    make_package(venv, "torch", "2.11.0")
    system = make_package(tmp_path / "usr-local", "torch", "2.13.0+cu130")

    assert doctor.check_shadowed_package(fake_module(system), "torch") is True

    assert doctor._problems == [
        "the venv's torch is installed but not the one being imported"]
    assert doctor._fixes, "an uninstall of the shadowing copy should be offered"


def test_pythonpath_shadowing_offers_no_uninstall(doctor, venv, tmp_path, monkeypatch, capsys):
    """When PYTHONPATH is the cause, uninstalling is the wrong advice --
    PYTHONPATH precedes every site directory, so the next install lands in
    the same trap. Say so instead."""
    make_package(venv, "torch", "2.11.0")
    system = make_package(tmp_path / "usr-local", "torch", "2.13.0+cu130")
    monkeypatch.setenv("PYTHONPATH", str(system.parent))

    assert doctor.check_shadowed_package(fake_module(system), "torch") is True

    assert doctor._fixes == []
    out = capsys.readouterr().out
    assert "unset PYTHONPATH" in out


def test_no_shadowing_reported_when_the_venv_copy_is_the_one_imported(doctor, venv):
    pkg = make_package(venv, "torch", "2.11.0")
    assert doctor.check_shadowed_package(fake_module(pkg), "torch") is False
    assert doctor._problems == []


def test_no_shadowing_reported_when_the_venv_has_no_copy(doctor, venv, tmp_path):
    """A system torch inherited through --system-site-packages is the normal,
    intended arrangement on a Jetson -- not something to complain about."""
    system = make_package(tmp_path / "usr-local", "torch", "2.5.0a0+nv24.08")
    assert doctor.check_shadowed_package(fake_module(system), "torch") is False
    assert doctor._problems == []


def test_installed_version_falls_back_to_dist_info(doctor, tmp_path):
    site = tmp_path / "sp"
    pkg = site / "torch"
    pkg.mkdir(parents=True)
    (site / "torch-2.11.0.dist-info").mkdir()
    assert doctor.installed_version(pkg) == "2.11.0"


def test_torch_repair_is_offered_once_not_twice(doctor):
    """Both the torch.cuda failure and the torchvision failure hand --fix the
    same remedy. --fix dedupes by callable identity, so a fresh closure per
    call meant 232 MB downloaded twice, the second install undoing the first.
    """
    first = doctor.install_matched_torch(12060)
    second = doctor.install_matched_torch(12060)
    assert first is second
    assert doctor.install_matched_torch(12030) is not first


def test_torchvision_is_checked_even_when_cuda_is_down(doctor, monkeypatch):
    """The regression this exists for.

    On a Jetson whose torch has been replaced by a PyPI CUDA-13 wheel,
    torch.cuda.is_available() is False *and* torchvision is gone. check_torch
    used to return at the first of those, so the second -- the one that
    cascades into four more reports -- was never even looked at.
    """
    seen: list[str] = []
    monkeypatch.setattr(doctor, "check_torchvision",
                        lambda torch, driver=None: seen.append("checked"))
    monkeypatch.setattr(doctor, "check_numpy", lambda torch: None)
    monkeypatch.setattr(doctor, "cuda_driver_version", lambda: 12060)

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = type(sys)("torch")
    fake_torch.__version__ = "2.13.0+cu130"
    fake_torch.__file__ = "/usr/local/lib/python3.10/dist-packages/torch/__init__.py"
    fake_torch.version = type(sys)("torch.version")
    fake_torch.version.cuda = "13.0"
    fake_torch.cuda = FakeCuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    doctor.check_torch()

    assert seen == ["checked"]
    # And the CUDA mismatch is still reported -- both, not one instead of
    # the other.
    assert "torch.cuda.is_available() is False" in doctor._problems
    assert "torch is built for a newer CUDA than the driver supports" in doctor._warnings
