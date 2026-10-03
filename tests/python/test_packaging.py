"""Python/native packaging regressions."""

import re
import shutil
import typing
from pathlib import Path

import pytest
from generativeqc import _cuda_runtime, _native


def test_installed_package_library_finds_wheel_layout(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    package = tmp_path / "generativeqc"
    library = package / "lib" / "libgenerativeqc.so"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(_native, "PACKAGE_DIR", package)

    assert _native._installed_package_library() == library


def test_native_candidates_prefer_override_then_bundled(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    explicit = tmp_path / "explicit" / "libgenerativeqc.so"
    explicit.parent.mkdir(parents=True)
    explicit.touch()
    package = tmp_path / "site" / "generativeqc"
    bundled = package / "lib" / "libgenerativeqc.so"
    bundled.parent.mkdir(parents=True)
    bundled.touch()
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(explicit))
    monkeypatch.setattr(_native, "PACKAGE_DIR", package)

    assert list(_native._candidate_paths())[:2] == [explicit, bundled]


def test_native_explicit_library_skips_discovery_and_preserves_live_selection(
    tmp_path: Path, monkeypatch: typing.Any
) -> None:
    """Avoid unused scans without caching paths, handles or device profiles."""
    from unittest.mock import Mock

    from generativeqc import profiles

    explicit = tmp_path / "explicit.so"
    replacement = tmp_path / "replacement.so"
    bundled = tmp_path / "bundled.so"
    for path in (explicit, replacement, bundled):
        path.touch()
    loaded: list[Path] = []
    selections: list[int] = []

    def fake_cdll(path: str) -> Mock:
        loaded.append(Path(path))
        library = Mock()
        library.generativeqc_get_abi_version.return_value = _native.ABI_VERSION
        return library

    def select(library: Mock, device_id: int) -> tuple[Mock, dict]:
        selections.append(device_id)
        return library, {"device_id": device_id}

    def unexpected_discovery() -> Path:
        raise AssertionError("a usable explicit library must not scan the wheel")

    monkeypatch.setattr(_cuda_runtime, "preload_cuda_runtime_libraries", lambda: ())
    monkeypatch.setattr(_native.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(profiles, "select_library", select)
    monkeypatch.setattr(_native, "_installed_package_library", unexpected_discovery)
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(explicit))
    first = _native.load_library(device="cuda", device_id=0)
    second = _native.load_library(device="cuda", device_id=1)
    assert first is not second
    assert first._generativeqc_profile_diagnostics == {"device_id": 0}
    assert second._generativeqc_profile_diagnostics == {"device_id": 1}
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(replacement))
    _native.load_library()
    # A missing override must still discover the current installed fallback.
    replacement.unlink()
    monkeypatch.setattr(_native, "_installed_package_library", lambda: bundled)
    _native.load_library()
    assert loaded == [explicit, explicit, replacement, bundled]
    assert selections == [0, 1]


def test_cuda_runtime_search_finds_pypi_provider_dirs(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    site_packages = tmp_path / "site-packages"
    cublas = site_packages / "nvidia" / "cublas" / "lib"
    cusolver = site_packages / "nvidia" / "cusolver" / "lib"
    cublas.mkdir(parents=True)
    cusolver.mkdir(parents=True)

    monkeypatch.setattr(
        _cuda_runtime.site, "getsitepackages", lambda: [str(site_packages)]
    )
    monkeypatch.setattr(_cuda_runtime.site, "getusersitepackages", lambda: None)

    search_dirs = _cuda_runtime._runtime_search_dirs()
    assert cublas.resolve() in search_dirs
    assert cusolver.resolve() in search_dirs


def test_cuda_runtime_preload_uses_curated_sonames_not_driver(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    provider = tmp_path / "lib"
    provider.mkdir()
    sonames = [group[0] for group in _cuda_runtime._CUDA_RUNTIME_LIBRARY_GROUPS]
    for soname in sonames:
        (provider / soname).touch()

    loaded = []

    def fake_cdll(path: typing.Any, *, mode: typing.Any) -> typing.Any:
        loaded.append((Path(path).name, mode))
        return object()

    monkeypatch.setattr(_cuda_runtime, "_cuda_runtime_handles", {})
    monkeypatch.setattr(_cuda_runtime, "_runtime_search_dirs", lambda: [provider])
    monkeypatch.setattr(_cuda_runtime.ctypes, "CDLL", fake_cdll)

    assert _cuda_runtime.preload_cuda_runtime_libraries() == tuple(sonames)
    assert [name for name, _ in loaded] == sonames
    assert "libcuda.so.1" not in sonames

    def unexpected_discovery() -> list[Path]:
        raise AssertionError("retained providers must not trigger another scan")

    monkeypatch.setattr(_cuda_runtime, "_runtime_search_dirs", unexpected_discovery)
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()
    assert [name for name, _ in loaded] == sonames


def test_cuda_runtime_preload_retries_missing_groups(
    tmp_path: Path, monkeypatch: typing.Any
) -> None:
    """A failed or unavailable group must not turn into a cached discovery miss."""
    first, second = "libfirst.so", "libsecond.so"
    groups = ((first, "libfirst-alternative.so"), (second,))
    (tmp_path / first).touch()
    attempts = []
    scans = []
    broken = True

    def discover() -> list[Path]:
        scans.append(True)
        return [tmp_path]

    def fake_cdll(path: str, *, mode: int) -> object:
        attempts.append(Path(path).name)
        if Path(path).name == second and broken:
            raise OSError("provider cannot be loaded yet")
        return object()

    monkeypatch.setattr(_cuda_runtime, "_CUDA_RUNTIME_LIBRARY_GROUPS", groups)
    monkeypatch.setattr(_cuda_runtime, "_cuda_runtime_handles", {})
    monkeypatch.setattr(_cuda_runtime, "_runtime_search_dirs", discover)
    monkeypatch.setattr(_cuda_runtime.ctypes, "CDLL", fake_cdll)
    assert _cuda_runtime.preload_cuda_runtime_libraries() == (first,)
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()
    (tmp_path / second).touch()
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()
    broken = False
    assert _cuda_runtime.preload_cuda_runtime_libraries() == (second,)
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()
    assert len(scans) == 4
    assert attempts == [first, second, second]


def test_cuda_runtime_preload_is_noop_off_linux(monkeypatch: typing.Any) -> None:
    monkeypatch.setattr(_cuda_runtime.sys, "platform", "darwin")

    assert _cuda_runtime._runtime_search_dirs() == []
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()


def test_cuda_runtime_preload_falls_through_loader_errors(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    broken = tmp_path / "broken"
    working = tmp_path / "working"
    broken.mkdir()
    working.mkdir()
    soname = "libcudart.so.12"
    (broken / soname).touch()
    (working / soname).touch()
    attempts = []

    def fake_cdll(path: typing.Any, *, mode: typing.Any) -> typing.Any:
        attempts.append((Path(path), mode))
        if Path(path).parent == broken:
            raise OSError("broken provider")
        return object()

    monkeypatch.setattr(_cuda_runtime, "_CUDA_RUNTIME_LIBRARY_GROUPS", ((soname,),))
    monkeypatch.setattr(_cuda_runtime, "_cuda_runtime_handles", {})
    monkeypatch.setattr(
        _cuda_runtime, "_runtime_search_dirs", lambda: [broken, working]
    )
    monkeypatch.setattr(_cuda_runtime.ctypes, "CDLL", fake_cdll)

    assert _cuda_runtime.preload_cuda_runtime_libraries() == (soname,)
    assert [path.parent for path, _ in attempts] == [broken, working]
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()


def test_cuda_runtime_loader_install_is_idempotent() -> None:
    installed = _native.load_library
    _cuda_runtime.install_native_loader()

    assert _native.load_library is installed


def test_package_installs_cuda_runtime_wrapper() -> None:
    assert getattr(_native.load_library, "_generativeqc_cuda_runtime_loader", False)


def test_wheel_compiler_templates_include_local_dependencies(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    """Exercise JIT source preparation using only the declared wheel payload."""
    tomllib = pytest.importorskip("tomllib")
    from generativeqc_compiler.common import paths
    from generativeqc_compiler.dft.ao_cuda import emit_grid_source
    from generativeqc_compiler.tensor.cuda_execute import tensor_source_identity

    root = Path(__file__).resolve().parents[2]
    package = tmp_path / "generativeqc_compiler"
    shutil.copytree(root / "python/generativeqc_compiler", package)
    config = tomllib.loads((root / "pyproject.toml").read_text())
    for source, destination in config["tool"]["scikit-build"]["wheel"][
        "force-include"
    ].items():
        target = tmp_path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if (root / source).is_dir():
            shutil.copytree(root / source, target)
        else:
            shutil.copyfile(root / source, target)
    monkeypatch.setattr(paths, "PACKAGE", package)
    assert tensor_source_identity()
    assert all(header.is_file() for header in emit_grid_source()[2])
    assets = package / "assets"
    # Quoted native includes can resolve beside the template or through the
    # compiler's src/include roots; none may depend on a neighboring checkout.
    for source in (assets / "src").rglob("*"):
        if source.is_file():
            for name in re.findall(
                r'^#include "([^"]+)"', source.read_text(), re.MULTILINE
            ):
                assert any(
                    (base / name).is_file()
                    for base in (source.parent, assets / "src", assets / "include")
                ), (source, name)


def test_project_metadata_declares_cuda_runtime_providers() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    required = {
        "nvidia-cublas-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-nvjitlink-cu12",
    }
    assert required <= {dependency.split(">=", 1)[0] for dependency in dependencies}
    assert project["project"]["optional-dependencies"]["cuda12"] == []


def test_cibuildwheel_uses_base_dependencies_for_provider_smoke() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/wheels.yml").read_text()
    assert "CIBW_TEST_EXTRAS" not in workflow
