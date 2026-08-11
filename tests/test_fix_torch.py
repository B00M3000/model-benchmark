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


def test_a_library_already_on_the_machine_is_preferred_over_a_download(
        ft, monkeypatch, tmp_path):
    """JetPack does ship libcupti.so.12 -- in cuda-*/extras/CUPTI/lib64, which
    is not on the loader path. Downloading a 12.9 wheel to replace a 12.6
    library that is already here, matched to this CUDA, is the wrong move."""
    torch_dir = tmp_path / "site-packages" / "torch"
    (torch_dir / "lib").mkdir(parents=True)
    cupti = tmp_path / "usr/local/cuda-12.6/extras/CUPTI/lib64/libcupti.so.12"
    cupti.parent.mkdir(parents=True)
    cupti.write_bytes(b"\x7fELF")

    monkeypatch.setattr(ft, "venv_package", lambda name: torch_dir if name == "torch" else None)
    monkeypatch.setattr(ft, "find_system_library", lambda name: cupti)
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not download"))

    probe = ft.Probe(error="ImportError: libcupti.so.12: cannot open shared object file: x")
    assert ft.link_local_library(probe, dry_run=False) is True
    assert (torch_dir / "lib" / "libcupti.so.12").resolve() == cupti.resolve()


@pytest.mark.parametrize(
    "relative",
    [
        # The one that matters: JetPack ships CUPTI here, off the loader path.
        "usr/local/cuda-12.6/extras/CUPTI/lib64/libcupti.so.12",
        "usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12",
        "usr/local/cuda-12.6/targets/aarch64-linux/lib/libcupti.so.12",
        "usr/local/cuda-12.6/lib64/libcupti.so.12",
        "usr/lib/aarch64-linux-gnu/libcupti.so.12",
    ],
)
def test_local_search_looks_where_cuda_actually_puts_libraries(ft, tmp_path, relative):
    """The glob list is the whole mechanism; a path missing from it finds
    nothing silently and falls through to a needless download."""
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x7fELF")

    assert ft.find_system_library("libcupti.so.12", root=tmp_path) == target


def test_local_search_returns_none_when_absent(ft, tmp_path):
    assert ft.find_system_library("libcupti.so.12", root=tmp_path) is None


def test_local_search_ignores_a_directory_of_the_right_name(ft, tmp_path):
    (tmp_path / "usr/local/cuda-12.6/lib64/libcupti.so.12").mkdir(parents=True)
    assert ft.find_system_library("libcupti.so.12", root=tmp_path) is None


def test_download_is_the_fallback_when_the_machine_has_none(ft, monkeypatch, tmp_path):
    torch_dir = tmp_path / "site-packages" / "torch"
    (torch_dir / "lib").mkdir(parents=True)
    wheel_lib = tmp_path / "nvidia" / "cu12" / "lib"
    wheel_lib.mkdir(parents=True)
    (wheel_lib / "libcupti.so.12").write_bytes(b"\x7fELF")
    installed: list = []

    monkeypatch.setattr(ft, "venv_package", lambda name: torch_dir if name == "torch" else None)
    monkeypatch.setattr(ft, "pip", lambda *a: installed.append(a) or True)
    monkeypatch.setattr(ft, "find_library", lambda name: wheel_lib / name)

    probe = ft.Probe(error="ImportError: libcupti.so.12: cannot open shared object file: x")
    assert ft.supply_missing_library(probe, dry_run=False) is True
    assert installed == [("install", "nvidia-cuda-cupti-cu12", "--no-deps")]


def test_several_missing_libraries_are_resolved_in_one_run(ft, monkeypatch):
    """The loader names the first library it cannot find and stops. Supplying
    it reveals the next, so a single pass leaves torch just as broken."""
    errors = [
        "ImportError: libcupti.so.12: cannot open shared object file: x",
        "ImportError: libcudss.so.0: cannot open shared object file: x",
        "ImportError: libnccl.so.2: cannot open shared object file: x",
    ]
    probes = [ft.Probe(error=e) for e in errors] + [
        ft.Probe(ok=True, version="2.11.0", path="/venv/torch/__init__.py",
                 cuda_build="12.6", cuda_available=True)]
    it = iter(probes)
    linked: list[str] = []

    monkeypatch.setattr(ft, "probe_torch", lambda python: next(it))
    monkeypatch.setattr(ft, "STRATEGIES", (
        ("link", lambda p, d: linked.append(p.missing_lib) is None or True),))

    assert ft.resolve().ok is True
    assert linked == ["libcupti.so.12", "libcudss.so.0", "libnccl.so.2"]


def test_the_same_library_twice_stops_instead_of_looping(ft, monkeypatch, capsys):
    """A strategy that reports success without actually resolving the library
    would otherwise spin until MAX_ROUNDS, relinking the same file."""
    same = ft.Probe(error="ImportError: libcupti.so.12: cannot open shared object file: x")
    monkeypatch.setattr(ft, "probe_torch", lambda python: same)
    monkeypatch.setattr(ft, "STRATEGIES", (("useless", lambda p, d: True),))

    assert ft.resolve().ok is False
    assert "still missing after being supplied" in capsys.readouterr().out


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


TORCHAUDIO_STALE = (
    "OSError: Could not load this library: "
    "/usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so"
)


@pytest.mark.parametrize(
    "error",
    [
        TORCHAUDIO_STALE,
        "ImportError: /x/torchvision/_C.so: undefined symbol: _ZN3c105Error",
        "RuntimeError: operator torchvision::nms does not exist",
        "OSError: libc10.so: cannot open shared object file: x",
    ],
)
def test_stale_companions_are_recognised(ft, error):
    assert ft.STALE_COMPANION.search(error)


@pytest.mark.parametrize(
    "error",
    [
        # Absent is not broken. transformers asks whether torchaudio is
        # available and copes when it is not; installing one nothing asked for
        # just acquires the next mismatch.
        "ModuleNotFoundError: No module named 'torchaudio'",
        "ImportError: cannot import name 'pipeline' from 'torchaudio'",
    ],
)
def test_absent_or_unrelated_companions_are_left_alone(ft, error):
    assert ft.STALE_COMPANION.search(error) is None


def test_a_stale_companion_is_replaced_with_one_matching_torch(ft, monkeypatch):
    """torch 2.11.0 needs torchvision/torchaudio 2.11.*; the trio ships in
    lockstep and a companion only loads against its own torch minor."""
    installed: list[tuple] = []
    torch_probe = ft.Probe(ok=True, version="2.11.0", cuda_build="12.6",
                           path="/venv/torch/__init__.py", cuda_available=True)
    probes = iter([
        ft.Probe(ok=True, version="0.26.0", path="/venv/torchvision/__init__.py"),
        ft.Probe(error=TORCHAUDIO_STALE),
        ft.Probe(ok=True, version="2.11.0", path="/venv/torchaudio/__init__.py"),
    ])
    monkeypatch.setattr(ft, "probe_module", lambda py, name: next(probes))
    monkeypatch.setattr(ft, "pip", lambda *a: installed.append(a) or True)
    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)

    assert ft.repair_companions(torch_probe) == []   # nothing left unresolved
    assert installed == [(
        "install", "--no-cache-dir", "--no-deps",
        "--index-url", "https://pypi.jetson-ai-lab.io/jp6/cu126",
        "torchaudio==2.11.*",
    )]


def test_a_missing_companion_is_not_installed(ft, monkeypatch):
    monkeypatch.setattr(ft, "probe_module", lambda py, name: ft.Probe(
        error="ModuleNotFoundError: No module named 'torchaudio'"))
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not install"))

    torch_probe = ft.Probe(ok=True, version="2.11.0", cuda_build="12.6",
                           path="/venv/torch/__init__.py")
    assert ft.repair_companions(torch_probe) == []


def test_companions_are_not_touched_while_torch_itself_is_broken(ft, monkeypatch):
    """Until torch imports, every companion fails for torch's reason and
    nothing can be told apart."""
    monkeypatch.setattr(ft, "probe_module", lambda py, name: pytest.fail("too early"))
    assert ft.repair_companions(ft.Probe(error=CONTAINER_ERROR)) == []


def test_torch_local_version_does_not_leak_into_the_pin(ft, monkeypatch):
    """torch reports 2.11.0+cu126 on some builds; '2.11.0+cu126.*' matches
    nothing and the install fails for a reason that looks like a missing
    wheel."""
    installed: list[tuple] = []
    probes = iter([ft.Probe(error=TORCHAUDIO_STALE),
                   ft.Probe(ok=True, version="2.11.0", path="/venv/tv/__init__.py")])
    monkeypatch.setattr(ft, "COMPANIONS", ("torchvision",))
    monkeypatch.setattr(ft, "probe_module", lambda py, name: next(probes))
    monkeypatch.setattr(ft, "pip", lambda *a: installed.append(a) or True)
    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)

    ft.repair_companions(ft.Probe(ok=True, version="2.11.0+cu126",
                                  cuda_build="12.6", path="/venv/torch/__init__.py"))
    assert installed[0][-1] == "torchvision==2.11.*"


def test_versions_are_parsed_from_pip_index(ft, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0,
        stdout="torchaudio (2.10.0)\nAvailable versions: 2.10.0, 2.9.1, 2.8.0\n",
        stderr=""))
    assert ft.available_versions("torchaudio", "https://x") == ["2.10.0", "2.9.1", "2.8.0"]


def test_versions_are_parsed_from_the_resolver_error(ft, monkeypatch):
    """The fallback, and the message that revealed the problem: pip lists what
    it *could* have installed when the pin matches nothing."""
    import subprocess

    calls: list[list] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "index" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such command")
        return subprocess.CompletedProcess(
            cmd, 1, stdout="",
            stderr="ERROR: Could not find a version that satisfies the requirement "
                   "torchaudio==99999 (from versions: 2.8.0, 2.9.1, 2.10.0)\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Newest first, regardless of the order pip happened to print.
    assert ft.available_versions("torchaudio", "https://x") == ["2.10.0", "2.9.1", "2.8.0"]
    assert len(calls) == 2


def test_versions_survive_an_index_that_says_nothing(ft, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="connection refused"))
    assert ft.available_versions("torch", "https://x") == []


@pytest.mark.parametrize(
    "torch_series, expected",
    [("2.11", "0.26"), ("2.10", "0.25"), ("2.8", "0.23"), ("2.5", "0.20")],
)
def test_torchvision_series_tracks_torch(ft, torch_series, expected):
    assert ft.torchvision_series(torch_series) == expected


def test_trio_moves_to_the_newest_series_the_index_has_all_of(ft, monkeypatch):
    """The real case: the Jetson index publishes torch 2.11.0 but no
    torchaudio past 2.10.0, so no companion can ever match torch and torch is
    the version that has to give."""
    index_has = {
        "torch": ["2.11.0", "2.10.0", "2.9.1", "2.8.0"],
        "torchvision": ["0.26.0", "0.25.0", "0.24.1", "0.23.0"],
        "torchaudio": ["2.10.0", "2.9.1", "2.8.0"],
    }
    installed: list[tuple] = []
    monkeypatch.setattr(ft, "available_versions", lambda name, index: index_has[name])
    monkeypatch.setattr(ft, "pip", lambda *a: installed.append(a) or True)
    monkeypatch.setattr(ft, "probe_torch", lambda py: ft.Probe(
        ok=True, version="2.10.0", cuda_build="12.6", path="/venv/torch/__init__.py",
        cuda_available=True))
    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)

    probe = ft.Probe(ok=True, version="2.11.0", cuda_build="12.6",
                     path="/venv/torch/__init__.py", cuda_available=True)
    assert ft.align_trio(probe) is True
    assert installed[0][-3:] == ("torch==2.10.*", "torchvision==0.25.*", "torchaudio==2.10.*")
    assert "--force-reinstall" in installed[0]


def test_trio_skips_a_series_missing_a_torchvision(ft, monkeypatch):
    index_has = {
        "torch": ["2.11.0", "2.10.0"],
        "torchvision": ["0.26.0"],           # nothing for torch 2.10
        "torchaudio": ["2.10.0"],            # nothing for torch 2.11
    }
    monkeypatch.setattr(ft, "available_versions", lambda name, index: index_has[name])
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not install"))

    probe = ft.Probe(ok=True, version="2.11.0", cuda_build="12.6",
                     path="/venv/torch/__init__.py")
    assert ft.align_trio(probe) is False


def test_trio_only_installs_companions_when_torch_is_already_right(ft, monkeypatch):
    """Re-downloading a 232 MB torch that is already the chosen version is
    pure waste."""
    index_has = {
        "torch": ["2.10.0"], "torchvision": ["0.25.0"], "torchaudio": ["2.10.0"],
    }
    installed: list[tuple] = []
    monkeypatch.setattr(ft, "available_versions", lambda name, index: index_has[name])
    monkeypatch.setattr(ft, "pip", lambda *a: installed.append(a) or True)
    monkeypatch.setattr(ft, "probe_torch", lambda py: ft.Probe(
        ok=True, version="2.10.0", cuda_build="12.6", path="/venv/torch/__init__.py"))

    probe = ft.Probe(ok=True, version="2.10.0", cuda_build="12.6",
                     path="/venv/torch/__init__.py")
    assert ft.align_trio(probe) is True
    assert installed[0][-2:] == ("torchvision==0.25.*", "torchaudio==2.10.*")
    assert not any(p.startswith("torch==") for p in installed[0])


def test_trio_reports_an_unreachable_index_rather_than_guessing(ft, monkeypatch):
    monkeypatch.setattr(ft, "available_versions", lambda name, index: [])
    monkeypatch.setattr(ft, "pip", lambda *a: pytest.fail("must not install"))

    probe = ft.Probe(ok=True, version="2.11.0", cuda_build="12.6",
                     path="/venv/torch/__init__.py")
    assert ft.align_trio(probe) is False


def test_an_unmatchable_companion_is_flagged_for_realignment(ft, monkeypatch):
    """repair_companions returns what it could not fix; that is the signal to
    move torch rather than keep trying to match it."""
    monkeypatch.setattr(ft, "COMPANIONS", ("torchaudio",))
    monkeypatch.setattr(ft, "probe_module", lambda py, name: ft.Probe(error=TORCHAUDIO_STALE))
    monkeypatch.setattr(ft, "pip", lambda *a: False)  # index has no matching version

    probe = ft.Probe(ok=True, version="2.11.0", cuda_build="12.6",
                     path="/venv/torch/__init__.py")
    assert ft.repair_companions(probe) == ["torchaudio"]


def test_index_follows_torchs_cuda_not_the_drivers(ft, monkeypatch):
    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)
    assert ft.wheel_index("12.6").endswith("/jp6/cu126")
    assert ft.wheel_index("12.8").endswith("/jp6/cu128")
    assert ft.wheel_index(None).endswith("/jp6/cu126")
    monkeypatch.setenv("TORCH_INDEX_URL", "https://example.invalid/simple")
    assert ft.wheel_index("12.6") == "https://example.invalid/simple"


def test_a_raising_strategy_does_not_hide_the_diagnosis(ft, monkeypatch, capsys):
    def explode(probe, dry_run):
        raise RuntimeError("network down")

    monkeypatch.setattr(ft, "probe_torch", lambda python: ft.Probe(error=CONTAINER_ERROR))
    monkeypatch.setattr(ft, "STRATEGIES", (("exploding", explode),))

    assert ft.resolve().ok is False
    out = capsys.readouterr().out
    assert "network down" in out
    assert "libcudss.so.0" in out
