"""The bounded prerequisite reuses production dependencies in the final gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
from test_ci_cuda_build_budget import WORKFLOW, _assert_cold_budget, _job, _step
from test_ci_cuda_cache_handoff import _assert_transport
from test_ci_prose_workflow import assert_full_coverage_step, selected

ROOT = Path(__file__).resolve().parents[2]
PREREQUISITE = "Compile production CUDA prerequisite archive"
INTEGRATION = "Compile production CUDA and runtime CTest"


def _script(step: str) -> str:
    match = re.search(r"^        run: >-\n((?:          .+\n)+)", step, re.MULTILINE)
    assert match, "build must be an executable folded command"
    return " ".join(textwrap.dedent(match[1]).splitlines())


def _assert_phases(workflow: str) -> None:
    _assert_cold_budget(workflow)
    _assert_transport(workflow)
    job = _job(workflow)
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--preset")
    parser.add_argument("--target", nargs="+")
    parser.add_argument("--parallel", type=int)
    for name, target in (
        (PREREQUISITE, "generativeqc_direct_native"),
        (INTEGRATION, "generativeqc_cuda_runtime_tests"),
    ):
        step = _step(job, name)
        assert_full_coverage_step(step)
        assert not selected(step, {"success()": False})
        argv = shlex.split(_script(step))
        assert argv[0] == "cmake"
        args, unknown = parser.parse_known_args(argv[1:])
        assert not unknown, "no rebuild, cleanup, dry-run or failure swallowing"
        assert args.build and args.preset == "cuda-release-sm120"
        assert args.target == [target] and args.parallel == 3
        assert "working-directory:" not in step
        assert "CMAKE_" not in step and "CCACHE_DIR:" not in step


def test_phases_keep_the_same_build_tree_and_required_final_gate() -> None:
    _assert_phases(WORKFLOW)


@pytest.mark.parametrize(
    ("name", "before", "after"),
    [
        (PREREQUISITE, "timeout-minutes: 10", "timeout-minutes: 0"),
        (PREREQUISITE, "timeout-minutes: 10", "timeout-minutes: 30"),
        (PREREQUISITE, "        if:", "        continue-on-error: true\n        if:"),
        (
            PREREQUISITE,
            "--preset cuda-release-sm120",
            "--preset cuda-release-cli-sm120",
        ),
        (PREREQUISITE, "--parallel 3", "--parallel 3 --clean-first"),
        (PREREQUISITE, "--parallel 3", "--parallel 3 || true"),
        (PREREQUISITE, "--target generativeqc_direct_native", "--target help"),
        (PREREQUISITE, "github.event_name == 'push' && ", ""),
        (PREREQUISITE, "github.ref == 'refs/heads/master' && ", ""),
        (PREREQUISITE, 'CCACHE_NORESHARE: "true"', 'CCACHE_NORESHARE: "false"'),
        (INTEGRATION, "--parallel 3", "--parallel 3 --clean-first"),
        (
            INTEGRATION,
            "--target generativeqc_cuda_runtime_tests",
            "--target generativeqc_direct_native",
        ),
    ],
)
def test_phase_controls_reject_incomplete_or_separate_builds(
    name: str, before: str, after: str
) -> None:
    step = _step(_job(WORKFLOW), name)
    assert before in step
    broken = WORKFLOW.replace(step, step.replace(before, after))
    with pytest.raises((AssertionError, argparse.ArgumentError)):
        _assert_phases(broken)


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("cmake", "ninja", "ccache", "g++")),
    reason="requires CMake, Ninja, ccache and a host compiler",
)
@pytest.mark.parametrize("remove_header_dependency", [False, True])
def test_production_archive_dependency_is_built_once_and_consumed_by_final_gate(
    tmp_path: Path, remove_header_dependency: bool
) -> None:
    """Use the real CUDA target declarations with tiny host-only stand-ins.

    This checks representative header closure and incremental link ownership, not
    CUDA compilation or numerical behavior; those remain required in CI.
    """
    subprocess.run(["ccache", "--version"], check=True, capture_output=True, timeout=5)
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (tmp_path / "include").symlink_to(ROOT / "include", target_is_directory=True)
    (tmp_path / "header.in").write_text("#define FIXTURE_VALUE 42\n")
    (tmp_path / "prerequisite.cpp").write_text(
        '#include "weighted_eri.cuh"\nint prerequisite(){return FIXTURE_VALUE;}\n'
    )
    (tmp_path / "angular.cpp").write_text("int angular(){return 0;}\n")
    (tmp_path / "integration.cpp").write_text(
        "int prerequisite(); int integration(){return prerequisite();}\n"
    )
    (tmp_path / "main.cpp").write_text(
        "int integration(); int main(){return integration() != 42;}\n"
    )
    cmake = f"""
        cmake_minimum_required(VERSION 3.24)
        project(CudaPrerequisiteGraph LANGUAGES CXX)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        set(CMAKE_CUDA_COMPILER_ID NVIDIA)
        set(CMAKE_CUDA_ARCHITECTURES 120)
        set(_generativeqc_cuda_compile_architectures 120)
        set(GENERATIVEQC_CUDA_DIRECT_DEVICE_LINK ON)
        set(GENERATIVEQC_CUDA_SEPARABLE_COMPILATION OFF)
        set(GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS 1)
        function(generativeqc_select_stationary_aot_profiles output)
          set(${{output}} "" PARENT_SCOPE)
        endfunction()
        foreach(provider cudart cublas cusolver)
          add_library(CUDA::${{provider}} INTERFACE IMPORTED)
        endforeach()
        set(GENERATIVEQC_WEIGHTED_ERI_HEADER
            "${{CMAKE_CURRENT_BINARY_DIR}}/generated/weighted_eri.cuh")
        add_custom_command(OUTPUT "${{GENERATIVEQC_WEIGHTED_ERI_HEADER}}"
          COMMAND "${{CMAKE_COMMAND}}" -E make_directory "${{CMAKE_CURRENT_BINARY_DIR}}/generated"
          COMMAND "${{CMAKE_COMMAND}}" -E copy "${{CMAKE_CURRENT_SOURCE_DIR}}/header.in"
                  "${{GENERATIVEQC_WEIGHTED_ERI_HEADER}}"
          DEPENDS header.in)
        add_library(generativeqc SHARED integration.cpp)
        include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")
        generativeqc_configure_cuda_backend(generativeqc)
        # Preserve production target/dependency declarations. Only replace
        # numerical compilation with small host fixtures on non-CUDA runners.
        foreach(owner generativeqc generativeqc_direct_native generativeqc_direct_angular_force)
          get_target_property(sources ${{owner}} SOURCES)
          foreach(source IN LISTS sources)
            if(NOT source STREQUAL "integration.cpp" AND NOT source MATCHES "^\\$<")
              set_source_files_properties("${{source}}" PROPERTIES HEADER_FILE_ONLY ON)
            endif()
          endforeach()
        endforeach()
        target_sources(generativeqc_direct_native PRIVATE prerequisite.cpp)
        target_sources(generativeqc_direct_angular_force PRIVATE angular.cpp)
        add_executable(generativeqc_cuda_runtime_tests main.cpp)
        target_link_libraries(generativeqc_cuda_runtime_tests PRIVATE generativeqc)
    """
    if remove_header_dependency:
        cmake += """
            get_target_property(sources generativeqc_direct_native SOURCES)
            list(REMOVE_ITEM sources "${GENERATIVEQC_WEIGHTED_ERI_HEADER}")
            set_property(TARGET generativeqc_direct_native PROPERTY SOURCES "${sources}")
        """
    (tmp_path / "CMakeLists.txt").write_text(textwrap.dedent(cmake))
    (tmp_path / "CMakePresets.json").write_text(
        json.dumps(
            {
                "version": 5,
                "configurePresets": [
                    {
                        "name": "cuda-release-sm120",
                        "generator": "Ninja",
                        "binaryDir": "${sourceDir}/build/cuda-release-sm120",
                        "cacheVariables": {"CMAKE_CXX_COMPILER_LAUNCHER": "ccache"},
                    }
                ],
                "buildPresets": [
                    {
                        "name": "cuda-release-sm120",
                        "configurePreset": "cuda-release-sm120",
                    }
                ],
            }
        )
    )
    env = {**os.environ, "CCACHE_DIR": str(tmp_path / "cache")}
    subprocess.run(
        ["cmake", "--preset", "cuda-release-sm120"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    job = _job(WORKFLOW)
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _script(_step(job, PREREQUISITE))],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if remove_header_dependency:
        assert (
            result.returncode != 0
            and "weighted_eri.cuh" in result.stderr + result.stdout
        )
        return
    assert result.returncode == 0, result.stderr + result.stdout
    build = tmp_path / "build/cuda-release-sm120"
    archive = build / "libgenerativeqc_direct_native.a"
    before = archive.stat().st_mtime_ns
    first_log = (build / ".ninja_log").read_text().splitlines()
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _script(_step(job, INTEGRATION))],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [str(build / "generativeqc_cuda_runtime_tests")], check=True, timeout=5
    )
    assert archive.stat().st_mtime_ns == before
    later = (build / ".ninja_log").read_text().splitlines()[len(first_log) :]
    assert later and not any("generativeqc_direct_native" in line for line in later)
