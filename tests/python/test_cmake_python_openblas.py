"""Discover an installed LP64 provider without importing it or bypassing policy."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from test_cmake_cpu_linalg_cache import _make_openblas_provider

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _distribution(root: Path) -> Path:
    """Expose metadata whose package import deliberately fails."""
    package = root / "scipy_openblas32"
    config = package / "lib/cmake/openblas"
    _make_openblas_provider(config, version="1.0", local_threads=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('provider must not be imported')\n"
    )
    metadata = root / "scipy_openblas32-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: scipy-openblas32\nVersion: 1.0\n"
    )
    (metadata / "RECORD").write_text(
        "scipy_openblas32/lib/cmake/openblas/OpenBLASConfig.cmake,,\n"
    )
    return config


def _configure(
    root: Path,
    native_cxx: NativeCxx,
    *,
    pythonpath: Path | None = None,
    prelude: str = "",
    extra: tuple[str, ...] = (),
    real: bool = False,
) -> tuple[Path, dict[str, str], str]:
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required")
    source = root / "source"
    source.mkdir()
    (source / "probe.cpp").write_text("int probe() {return 0;}\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(PythonOpenBlas LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 20)\n"
        # Disable ambient providers after project/compiler discovery. The only
        # automatic path left is the selected interpreter's distribution data.
        "set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig TRUE)\n"
        "set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)\n"
        "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)\n"
        "set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)\n"
        "set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)\n"
        "set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)\n"
        + prelude
        + f'include("{ROOT / "cmake/GenerativeQCCpuLinalg.cmake"}")\n'
        + (
            f'add_library(probe SHARED "{ROOT / "src/tensor/cpu_linalg.cpp"}")\n'
            f'target_include_directories(probe PUBLIC "{ROOT / "src"}")\n'
            f'add_executable(consumer "{ROOT / "tests/native/test_cpu_linalg.cpp"}")\n'
            "target_link_libraries(consumer PRIVATE probe)\n"
            if real
            else "add_library(probe STATIC probe.cpp)\n"
        )
        + "generativeqc_configure_cpu_linalg(probe)\n"
        + 'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/definitions.txt" '
        'CONTENT "$<JOIN:$<TARGET_PROPERTY:probe,COMPILE_DEFINITIONS>,\\n>")\n'
    )
    env = {
        **os.environ,
        "CMAKE_PREFIX_PATH": "",
        "CMAKE_CXX_COMPILER_LAUNCHER": native_cxx.cache,
    }
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    build = root / "build"
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            *extra,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return build, env, result.stdout


@pytest.mark.parametrize("mode", ["auto", "scalar", "cross", "disabled", "explicit"])
def test_installed_provider_discovery_respects_selection(
    tmp_path: Path, native_cxx: NativeCxx, mode: str
) -> None:
    """A host wheel fills missing discovery, but cannot replace target policy."""
    config = _distribution(tmp_path / "site")
    prelude = ""
    extra = ()
    if mode == "scalar":
        extra = ("-DGENERATIVEQC_CPU_LINALG_PROVIDER=scalar",)
    elif mode == "cross":
        extra = ("-DCMAKE_SYSTEM_NAME=Generic",)
    elif mode == "disabled":
        prelude = "set(CMAKE_DISABLE_FIND_PACKAGE_OpenBLAS TRUE)\n"
    elif mode == "explicit":
        chosen = tmp_path / "chosen"
        _make_openblas_provider(chosen, version="2.0", local_threads=False)
        extra = (f"-DOpenBLAS_DIR={chosen}",)
    build, _, _ = _configure(
        tmp_path, native_cxx, pythonpath=tmp_path / "site", prelude=prelude, extra=extra
    )
    definitions = (build / "definitions.txt").read_text()
    if mode in {"auto", "explicit"}:
        assert "GENERATIVEQC_HAS_OPENBLAS=1" in definitions
        assert "GENERATIVEQC_OPENBLAS_HAS_LAPACKE=1" in definitions
        assert (
            f"GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS={int(mode == 'auto')}"
            in definitions
        )
        selected = config if mode == "auto" else chosen
        assert any(
            line.startswith("OpenBLAS_DIR:") and line.partition("=")[2] == str(selected)
            for line in (build / "CMakeCache.txt").read_text().splitlines()
        )
    else:
        assert "GENERATIVEQC_HAS_OPENBLAS=0" in definitions


def test_real_installed_provider_executes_native_cpu_contract(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Link the actual production provider, including local-thread and LAPACK gates."""
    if importlib.util.find_spec("scipy_openblas32") is None:
        pytest.skip("optional scipy-openblas32 build dependency is absent")
    build, env, _ = _configure(tmp_path, native_cxx, real=True)
    assert "GENERATIVEQC_HAS_OPENBLAS=1" in (build / "definitions.txt").read_text()
    for command in (
        ["cmake", "--build", str(build), "--parallel", "2"],
        [str(build / "consumer")],
    ):
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=180, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
