"""Resolving a torch that will not import.

The case these pin down: one `.venv` inside the working tree, used both on the
Jetson host and inside a container bind-mounted over it. A torch installed to
repair one environment links libraries the other does not have, and the fix is
never "install another torch" -- that repairs whichever environment is running
and re-breaks the other on the next switch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ft():
    spec = importlib.util.spec_from_file_location(
        "fix_torch_under_test", REPO_ROOT / "scripts" / "fix_torch.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("fix_torch_under_test", None)


CONTAINER_ERROR = (
    "ImportError: libcudss.so.0: cannot open shared object file: "
    "No such file or directory"
)


def test_missing_library_is_extracted_from_the_import_error(ft):
    assert ft.Probe(error=CONTAINER_ERROR).missing_lib == "libcudss.so.0"


@pytest.mark.parametrize(
    "error, expected",
    [
        ("ImportError: libcusparseLt.so.0: cannot open shared object file: x",
         "libcusparseLt.so.0"),
        ("ImportError: libcudnn.so.9: cannot open shared object file: x",
         "libcudnn.so.9"),
        # Unversioned, and a name with a digit in the middle.
        ("ImportError: libnvpl_blas_lp64_gomp.so: cannot open shared object file: x",
         "libnvpl_blas_lp64_gomp.so"),
        # Not this shape at all -- must not be mistaken for one.
        ("ModuleNotFoundError: No module named 'torch'", None),
        ("ImportError: undefined symbol: _ZN3c105ErrorC1E", None),
        ("", None),
    ],
)
def test_only_missing_library_errors_are_recognised(ft, error, expected):
    assert ft.Probe(error=error).missing_lib == expected


def test_probe_reads_the_marker_line_not_stray_output(ft, monkeypatch):
    """Importing torch drags in libraries that print banners and warnings of
    their own. Parsing whatever came out last would read those as the result."""
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0,
            stdout="A NumPy version >=1.17.3 is required\n"
                   '<<<PROBE>>>{"ok": true, "version": "2.5.0", "path": "/x/torch/'
                   '__init__.py", "cuda_build": "12.6", "cuda_available": true}\n'
                   "some trailing chatter\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe = ft.probe_torch("python3")
    assert probe.ok and probe.version == "2.5.0" and probe.cuda_available


def test_probe_reports_failure_when_no_marker_is_produced(ft, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="Traceback...\nImportError: boom\n"))
    probe = ft.probe_torch("python3")
    assert probe.ok is False
    assert "boom" in probe.error


def test_probe_survives_an_interpreter_that_cannot_run(ft, monkeypatch):
    import subprocess

    def explode(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", explode)
    assert ft.probe_torch("/nonexistent/python").ok is False


def test_dropping_the_venv_torch_needs_a_working_inherited_one(ft, monkeypatch, tmp_path):
    """Removing the venv's copy only helps if something else answers `import
    torch` afterwards. With nothing inherited it turns a broken torch into no
    torch."""
    monkeypatch.setattr(ft, "venv_package", lambda name: tmp_path / "torch")
    monkeypatch.setattr(ft, "probe_torch", lambda python: ft.Probe(error=CONTAINER_ERROR))
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not uninstall"))

    assert ft.drop_venv_torch(ft.Probe(error=CONTAINER_ERROR), dry_run=False) is False


def test_dropping_the_venv_torch_takes_torchvision_with_it(ft, monkeypatch, tmp_path):
    """A venv torchvision left shadowing an inherited torch is the "operator
    torchvision::nms does not exist" failure, which is harder to read than the
    import error it replaces."""
    calls: list[tuple] = []
    monkeypatch.setattr(ft, "venv_package", lambda name: tmp_path / name)
    monkeypatch.setattr(ft, "probe_torch", lambda python: ft.Probe(
        ok=True, version="2.5.0", path="/usr/lib/torch/__init__.py",
        cuda_build="12.6", cuda_available=True))
    monkeypatch.setattr(ft, "pip", lambda *a: calls.append(a) or True)

    assert ft.drop_venv_torch(ft.Probe(error=CONTAINER_ERROR), dry_run=False) is True
    assert calls == [("uninstall", "-y", "torch", "torchvision")]


def test_dry_run_changes_nothing(ft, monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "venv_package", lambda name: tmp_path / name)
    monkeypatch.setattr(ft, "probe_torch", lambda python: ft.Probe(
        ok=True, version="2.5.0", path="/usr/lib/torch/__init__.py"))
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not run pip"))

    assert ft.drop_venv_torch(ft.Probe(error=CONTAINER_ERROR), dry_run=True) is False


def test_venv_torch_is_not_dropped_when_it_is_the_only_one(ft, monkeypatch):
    monkeypatch.setattr(ft, "venv_package", lambda name: None)
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not uninstall"))
    assert ft.drop_venv_torch(ft.Probe(error=CONTAINER_ERROR), dry_run=False) is False


def test_unknown_library_is_reported_rather_than_guessed(ft, monkeypatch, capsys):
    """NVPL and other ARM performance libraries have no aarch64 pip wheel.
    Offering an install for them trades a loader error for a resolver error."""
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not install"))
    probe = ft.Probe(error="ImportError: libnvpl_blas_lp64_gomp.so: "
                           "cannot open shared object file: x")

    assert ft.supply_missing_library(probe, dry_run=False) is False
    assert "no pip wheel known" in capsys.readouterr().out


def test_library_is_linked_where_torch_actually_looks(ft, monkeypatch, tmp_path):
    """torch/_C carries RPATH $ORIGIN/lib, so torch/lib resolves without
    LD_LIBRARY_PATH and regardless of which nvidia/*/lib a given build has
    baked in."""
    torch_dir = tmp_path / "site-packages" / "torch"
    (torch_dir / "lib").mkdir(parents=True)
    wheel_lib = tmp_path / "site-packages" / "nvidia" / "cu12" / "lib"
    wheel_lib.mkdir(parents=True)
    (wheel_lib / "libcudss.so.0").write_bytes(b"\x7fELF")

    monkeypatch.setattr(ft, "venv_package", lambda name: torch_dir if name == "torch" else None)
    monkeypatch.setattr(ft, "pip", lambda *a: True)
    monkeypatch.setattr(ft, "find_library", lambda name: wheel_lib / name)

    assert ft.supply_missing_library(ft.Probe(error=CONTAINER_ERROR), dry_run=False) is True

    linked = torch_dir / "lib" / "libcudss.so.0"
    assert linked.is_symlink()
    assert linked.resolve() == (wheel_lib / "libcudss.so.0").resolve()


def test_relinking_over_an_existing_file_succeeds(ft, monkeypatch, tmp_path):
    """Re-running setup_jetson.sh must be idempotent, not fail on FileExists."""
    torch_dir = tmp_path / "site-packages" / "torch"
    (torch_dir / "lib").mkdir(parents=True)
    (torch_dir / "lib" / "libcudss.so.0").write_bytes(b"stale")
    wheel_lib = tmp_path / "nvidia" / "cu12" / "lib"
    wheel_lib.mkdir(parents=True)
    (wheel_lib / "libcudss.so.0").write_bytes(b"\x7fELF")

    monkeypatch.setattr(ft, "venv_package", lambda name: torch_dir if name == "torch" else None)
    monkeypatch.setattr(ft, "pip", lambda *a: True)
    monkeypatch.setattr(ft, "find_library", lambda name: wheel_lib / name)

    assert ft.supply_missing_library(ft.Probe(error=CONTAINER_ERROR), dry_run=False) is True
    assert (torch_dir / "lib" / "libcudss.so.0").resolve() == (
        wheel_lib / "libcudss.so.0").resolve()


def test_a_working_torch_is_left_alone(ft, monkeypatch):
    """The common case, and the one where doing anything is a bug."""
    monkeypatch.setattr(ft, "probe_torch", lambda python: ft.Probe(
        ok=True, version="2.5.0", path="/usr/lib/torch/__init__.py",
        cuda_build="12.6", cuda_available=True))
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not touch a working torch"))

    assert ft.resolve().ok is True


def test_strategies_stop_at_the_first_that_works(ft, monkeypatch):
    """Supplying a library on top of a torch that a plain uninstall already
    fixed would leave a stray 103 MB wheel and a symlink behind."""
    attempted: list[str] = []
    probes = iter([ft.Probe(error=CONTAINER_ERROR),
                   ft.Probe(ok=True, version="2.5.0", path="/usr/lib/torch/__init__.py",
                            cuda_available=True)])
    monkeypatch.setattr(ft, "probe_torch", lambda python: next(probes))
    monkeypatch.setattr(ft, "STRATEGIES", (
        ("first", lambda p, d: attempted.append("first") or True),
        ("second", lambda p, d: attempted.append("second") or True),
    ))

    assert ft.resolve().ok is True
    assert attempted == ["first"]


def test_a_raising_strategy_does_not_hide_the_diagnosis(ft, monkeypatch, capsys):
    def explode(probe, dry_run):
        raise RuntimeError("network down")

    monkeypatch.setattr(ft, "probe_torch", lambda python: ft.Probe(error=CONTAINER_ERROR))
    monkeypatch.setattr(ft, "STRATEGIES", (("exploding", explode),))

    assert ft.resolve().ok is False
    out = capsys.readouterr().out
    assert "network down" in out
    assert "libcudss.so.0" in out
