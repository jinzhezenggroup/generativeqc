"""Exercise optional SDK admission with real CMake configuration, without a GPU."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _configure(
    tmp_path: Path, definitions: list[str], *, version: str = "3.9.2"
) -> subprocess.CompletedProcess[str]:
    """Use fixture headers to test configuration only; never compile a fake SDK."""
    cache = shutil.which("ccache")
    assert cache, "CMake qualification requires ccache"
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    sdk = tmp_path / "sdk"
    headers = sdk / "include/cutlass"
    (headers / "gemm/device").mkdir(parents=True, exist_ok=True)
    (headers / "gemm/device/gemm_batched.h").touch()
    (headers / "version.h").write_text(
        "\n".join(
            f"#define CUTLASS_{part} {value}"
            for part, value in zip(
                ("MAJOR", "MINOR", "PATCH"), version.split("."), strict=True
            )
        )
        + "\n"
    )
    (tmp_path / "empty.cpp").write_text("// Configuration-only consumer.\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(OptionalCutlass LANGUAGES CXX)\n"
        "add_library(owner STATIC empty.cpp)\n"
        f'include("{ROOT / "cmake/GenerativeQCCutlass.cmake"}")\n'
        "generativeqc_configure_cutlass(owner)\n"
        "if(TARGET generativeqc_cutlass)\n"
        "  get_target_property(headers generativeqc_cutlass INTERFACE_INCLUDE_DIRECTORIES)\n"
        "  get_target_property(macros generativeqc_cutlass INTERFACE_COMPILE_DEFINITIONS)\n"
        "  get_target_property(deps owner INTERFACE_LINK_LIBRARIES)\n"
        '  file(WRITE "${CMAKE_BINARY_DIR}/provider.txt" "${headers}\n${macros}\n${deps}\n")\n'
        "endif()\n"
    )
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={cache}",
            f"-DCMAKE_CUDA_COMPILER_LAUNCHER={cache}",
            f"-DGENERATIVEQC_CUTLASS_ROOT={sdk}",
            "-DGENERATIVEQC_ENABLE_CUDA=ON",
            "-DGENERATIVEQC_CUDA_PROVIDER=nvidia",
            *definitions,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_optional_cutlass_default_does_not_probe(tmp_path: Path) -> None:
    result = _configure(tmp_path, ["-DGENERATIVEQC_CUTLASS_ROOT=/missing/sdk"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "build/provider.txt").exists()


def test_optional_cutlass_propagates_sdk_and_macro(tmp_path: Path) -> None:
    result = _configure(tmp_path, ["-DGENERATIVEQC_ENABLE_CUTLASS=ON"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "build/provider.txt").read_text().splitlines() == [
        str(tmp_path / "sdk/include"),
        "GENERATIVEQC_HAS_CUTLASS=1",
        "generativeqc_cutlass",
    ]
    # Reconfiguration must use a changed ROOT, not a cached successful find_path.
    result = _configure(
        tmp_path,
        [
            "-DGENERATIVEQC_ENABLE_CUTLASS=ON",
            "-DGENERATIVEQC_CUTLASS_ROOT=/missing/sdk",
        ],
    )
    assert result.returncode != 0
    assert "must name a CUTLASS 3.9.2 source tree" in result.stderr


@pytest.mark.parametrize(
    "definition,reason",
    [
        ("-DGENERATIVEQC_ENABLE_CUDA=OFF", "requires the NVIDIA CUDA backend"),
        ("-DGENERATIVEQC_CUDA_PROVIDER=cumetal", "requires the NVIDIA CUDA backend"),
        (
            "-DGENERATIVEQC_PYTHON_WHEEL=ON",
            "wheel artifact packaging is not implemented",
        ),
    ],
)
def test_optional_cutlass_rejects_unqualified_builds(
    tmp_path: Path, definition: str, reason: str
) -> None:
    result = _configure(tmp_path, ["-DGENERATIVEQC_ENABLE_CUTLASS=ON", definition])
    assert result.returncode != 0
    assert reason in result.stderr


@pytest.mark.parametrize("version", ["3.8.0", "4.0.0", "3.9.bad"])
def test_optional_cutlass_rejects_unqualified_headers(
    tmp_path: Path, version: str
) -> None:
    result = _configure(tmp_path, ["-DGENERATIVEQC_ENABLE_CUTLASS=ON"], version=version)
    assert result.returncode != 0
    assert "CUTLASS" in result.stderr


def test_optional_cutlass_revalidates_replaced_headers(tmp_path: Path) -> None:
    """An incremental build must re-admit an SDK replaced at the same path."""
    result = _configure(tmp_path, ["-DGENERATIVEQC_ENABLE_CUTLASS=ON"])
    assert result.returncode == 0, result.stdout + result.stderr
    version = tmp_path / "sdk/include/cutlass/version.h"
    version.write_text(
        version.read_text().replace("CUTLASS_MAJOR 3", "CUTLASS_MAJOR 4")
    )
    result = subprocess.run(
        ["cmake", "--build", str(tmp_path / "build")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "requires qualified headers 3.9.2" in result.stderr
