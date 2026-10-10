"""Check optional cuTENSOR SDK discovery and incremental admission without CUDA."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _sdk(path: Path, *, version: str = "2.8.0", library_dir: str = "lib") -> None:
    """Create findable headers and a dummy SONAME for configure-only tests."""
    include = path / "include"
    include.mkdir(parents=True, exist_ok=True)
    major, minor, patch = version.split(".")
    (include / "cutensor.h").write_text(
        f"#define CUTENSOR_MAJOR {major}\n"
        f"#define CUTENSOR_MINOR {minor}\n"
        f"#define CUTENSOR_PATCH {patch}\n"
    )
    lib = path / library_dir
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "libcutensor.so.2").write_bytes(b"configure-only fixture\n")


def _project(path: Path) -> Path:
    src = path / "src"
    src.mkdir()
    (src / "empty.cpp").write_text("// Not compiled: configuration-only fixture.\n")
    (src / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(CutensorAdmission LANGUAGES CXX)\n"
        "add_library(owner STATIC EXCLUDE_FROM_ALL empty.cpp)\n"
        f'include("{ROOT / "cmake/GenerativeQCCutensor.cmake"}")\n'
        "generativeqc_configure_cutensor(owner)\n"
        "if(TARGET generativeqc_cutensor)\n"
        '  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/provider.txt" CONTENT\n'
        '    "$<TARGET_PROPERTY:generativeqc_cutensor,INTERFACE_INCLUDE_DIRECTORIES>\n'
        "$<TARGET_PROPERTY:generativeqc_cutensor_library,IMPORTED_LOCATION>\n"
        "$<TARGET_PROPERTY:generativeqc_cutensor,INTERFACE_COMPILE_DEFINITIONS>\n"
        '$<TARGET_PROPERTY:owner,INTERFACE_LINK_LIBRARIES>\n")\n'
        "endif()\n"
        'add_custom_target(admission ALL COMMAND "${CMAKE_COMMAND}" -E touch '
        '"${CMAKE_BINARY_DIR}/admitted.txt")\n'
    )
    return src


def _configure(
    src: Path,
    root: Path | None,
    *,
    enabled: bool = True,
    environment_root: Path | None = None,
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    enabled_value = "ON" if enabled else "OFF"
    cmd = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(src.parent / "build"),
        "-G",
        "Ninja",
        f"-DGENERATIVEQC_ENABLE_CUTENSOR={enabled_value}",
        "-DGENERATIVEQC_ENABLE_CUDA=ON",
        "-DGENERATIVEQC_CUDA_PROVIDER=nvidia",
        f"-DGENERATIVEQC_CUTENSOR_ROOT={root or ''}",
        *extra,
    ]
    env = os.environ.copy()
    env.pop("GENERATIVEQC_CUTENSOR_ROOT", None)
    if environment_root is not None:
        env["GENERATIVEQC_CUTENSOR_ROOT"] = str(environment_root)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=45, env=env, check=False
    )


def _assert_sdk(src: Path, root: Path, *, library_dir: str = "lib") -> None:
    header_dir, library, macros, links = (
        (src.parent / "build/provider.txt").read_text().splitlines()
    )
    assert header_dir == str(root / "include")
    assert library == str(root / library_dir / "libcutensor.so.2")
    assert macros == "GENERATIVEQC_HAS_CUTENSOR=1"
    assert links == "generativeqc_cutensor"
    cache = (src.parent / "build/CMakeCache.txt").read_text()
    assert f"GENERATIVEQC_CUTENSOR_INCLUDE_DIR:PATH={root / 'include'}" in cache
    assert f"GENERATIVEQC_CUTENSOR_LIBRARY:FILEPATH={library}" in cache


def _build(src: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmake", "--build", str(src.parent / "build")],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


@pytest.fixture(autouse=True)
def _require_cmake_and_ninja() -> None:
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        pytest.skip("CMake/Ninja required for SDK admission tests")


def test_opt_in_disabled_does_not_resolve_sdk(tmp_path: Path) -> None:
    src = _project(tmp_path)
    result = _configure(src, tmp_path / "missing", enabled=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "build/provider.txt").exists()


def test_root_switch_replaces_both_header_and_library(tmp_path: Path) -> None:
    src = _project(tmp_path)
    a, b = tmp_path / "sdk-a", tmp_path / "sdk-b"
    _sdk(a)
    _sdk(b, version="2.10.1", library_dir="lib64")
    result = _configure(src, a)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_sdk(src, a)
    # Plant legacy find_* values to ensure they cannot override the new ROOT.
    result = _configure(
        src,
        b,
        extra=(
            f"-DGENERATIVEQC_CUTENSOR_INCLUDE_DIR:PATH={a / 'include'}",
            f"-DGENERATIVEQC_CUTENSOR_LIBRARY:FILEPATH={a / 'lib/libcutensor.so.2'}",
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_sdk(src, b, library_dir="lib64")
    assert _configure(src, tmp_path / "missing").returncode != 0
    # Explicit root cannot fall back to the usable previous SDK or system SDK.
    assert _configure(src, a).returncode == 0
    _assert_sdk(src, a)


def test_environment_root_fallback_and_explicit_precedence(tmp_path: Path) -> None:
    src = _project(tmp_path)
    a, b = tmp_path / "sdk-env", tmp_path / "sdk-explicit"
    _sdk(a)
    _sdk(b)
    result = _configure(src, None, environment_root=a)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_sdk(src, a)
    result = _configure(src, b, environment_root=a)
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_sdk(src, b)


@pytest.mark.parametrize("version", ["2.7.9", "3.0.0", "1.99.1", "2.bad.0"])
def test_rejects_unqualified_header_versions(tmp_path: Path, version: str) -> None:
    src = _project(tmp_path)
    sdk = tmp_path / "sdk"
    _sdk(sdk, version=version)
    result = _configure(src, sdk)
    assert result.returncode != 0
    assert "cuTENSOR" in result.stderr


@pytest.mark.parametrize(
    "override,message",
    [
        ("-DGENERATIVEQC_ENABLE_CUDA=OFF", "requires the NVIDIA CUDA backend"),
        ("-DGENERATIVEQC_CUDA_PROVIDER=cumetal", "requires the NVIDIA CUDA backend"),
        (
            "-DGENERATIVEQC_PYTHON_WHEEL=ON",
            "wheel dependency packaging is not implemented",
        ),
    ],
)
def test_rejects_unsupported_build_modes(
    tmp_path: Path, override: str, message: str
) -> None:
    src = _project(tmp_path)
    sdk = tmp_path / "sdk"
    _sdk(sdk)
    result = _configure(src, sdk, extra=(override,))
    assert result.returncode != 0
    assert message in result.stderr


def test_in_place_header_upgrade_rechecks_on_incremental_build(tmp_path: Path) -> None:
    src = _project(tmp_path)
    sdk = tmp_path / "sdk"
    _sdk(sdk)
    result = _configure(src, sdk)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _build(src).returncode == 0
    header = sdk / "include/cutensor.h"
    header.write_text(
        header.read_text().replace("CUTENSOR_MAJOR 2", "CUTENSOR_MAJOR 3")
    )
    built = _build(src)
    assert built.returncode != 0
    assert "Re-running CMake" in built.stdout
    assert "requires cuTENSOR 2.8 or later in 2.x" in built.stderr
    header.write_text(
        header.read_text().replace("CUTENSOR_MAJOR 3", "CUTENSOR_MAJOR 2")
    )
    built = _build(src)
    assert built.returncode == 0, built.stdout + built.stderr
    assert "Re-running CMake" in built.stdout
    _assert_sdk(src, sdk)


def test_in_place_library_relocation_rediscovers_on_build(tmp_path: Path) -> None:
    src = _project(tmp_path)
    sdk = tmp_path / "sdk"
    _sdk(sdk)
    result = _configure(src, sdk)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _build(src).returncode == 0
    original = sdk / "lib/libcutensor.so.2"
    moved = sdk / "lib64/libcutensor.so.2"
    moved.parent.mkdir(parents=True)
    original.replace(moved)
    built = _build(src)
    assert built.returncode == 0, built.stdout + built.stderr
    _assert_sdk(src, sdk, library_dir="lib64")
