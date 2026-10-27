"""Execute CMake profile resolution for portable and mixed CUDA builds."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "architectures,profile,overrides,expected",
    [
        (
            "86-real",
            "portable",
            "",
            ["--target-architecture", "sm_86", "--profile", "portable"],
        ),
        (
            "120;80-real;120-virtual",
            "portable",
            "sm_120",
            [
                "--target-architecture",
                "sm_80",
                "--target-architecture",
                "sm_120",
                "--profile",
                "portable",
                "--profile-map",
                "sm_120=sm_120",
            ],
        ),
        (
            "90",
            "auto",
            "local_hopper",
            ["--target-architecture", "sm_90", "--profile-map", "sm_90=local_hopper"],
        ),
    ],
)
def test_cmake_propagates_target_and_profile_intent(
    tmp_path: Path,
    architectures: str,
    profile: str,
    overrides: str,
    expected: list[str],
) -> None:
    script = tmp_path / "profiles.cmake"
    result = tmp_path / "arguments.txt"
    script.write_text(f'''
cmake_minimum_required(VERSION 3.24)
include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")
set(_generativeqc_cuda_compile_architectures "{architectures}")
set(GENERATIVEQC_AOT_PROFILE "{profile}")
set(GENERATIVEQC_AOT_PROFILES "{overrides}")
generativeqc_cuda_profile_arguments(arguments)
file(WRITE "{result}" "${{arguments}}")
''')
    subprocess.run(
        ["cmake", "-P", str(script)], check=True, capture_output=True, timeout=30
    )
    assert result.read_text().split(";") == expected


def test_generic_presets_do_not_override_a_requested_gpu(tmp_path: Path) -> None:
    """Configure a tiny project using the actual presets, with an Ada target."""
    (tmp_path / "CMakePresets.json").write_text(
        (ROOT / "CMakePresets.json").read_text()
    )
    (tmp_path / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.24)
project(preset_contract NONE)
if(GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES)
  set(selected "${GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES}")
else()
  set(selected "${CMAKE_CUDA_ARCHITECTURES}")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/target.txt" "${selected};${GENERATIVEQC_AOT_PROFILE}")
""")
    presets = json.loads((ROOT / "CMakePresets.json").read_text())
    for name in ("cuda-dev-fast", "cuda-release"):
        assert any(preset["name"] == name for preset in presets["configurePresets"])
        subprocess.run(
            ["cmake", "--preset", name, "-DCMAKE_CUDA_ARCHITECTURES=89-real"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            timeout=30,
        )
        assert (
            tmp_path / "build" / name / "target.txt"
        ).read_text() == "89-real;portable"
