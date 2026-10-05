"""Keep both CUDA jobs mandatory and resource sources owned by the CMake graph."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()


def _job(name: str) -> str:
    section = WORKFLOW.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-zA-Z0-9_-]+:\n", section, maxsplit=1)[0]


def _step(job: str, name: str) -> str:
    section = _job(job).split(f"      - name: {name}\n", 1)[1]
    return section.split("      - name: ", 1)[0]


def test_both_cuda_jobs_remain_required_with_the_same_source_and_budget() -> None:
    for name in ("cuda-compile", "cuda-resources"):
        job = _job(name)
        header = job.split("    steps:\n", 1)[0]
        assert "needs: merge_queue_liveness" in header
        assert "if: needs.merge_queue_liveness.outputs.active != 'false'" in header
        assert "timeout-minutes: 25" in header
        assert "continue-on-error:" not in header
        checkout = job.split("      - uses: actions/checkout@", 1)[1]
        checkout = checkout.split("      - name: ", 1)[0]
        assert "ref: ${{ github.sha }}" in checkout
        assert "persist-credentials: false" in checkout
        assert "CUDACXX: /usr/local/cuda-12.9/bin/nvcc" in header
    aggregate = _job("pass")
    assert (
        "needs: [merge_queue_liveness, cpu, cuda-compile, cuda-resources, python]"
        in aggregate
    )
    assert "jobs: ${{ toJSON(needs) }}" in aggregate
    assert "allowed-failures:" not in aggregate
    assert "allowed-skips:" not in aggregate


@pytest.mark.parametrize(
    "job,name,command",
    [
        (
            "cuda-compile",
            "Compile production CUDA and runtime CTest",
            "--target generativeqc_cuda_runtime_tests --parallel 3",
        ),
        (
            "cuda-compile",
            "Compile opted-in AO radial kernels with release resources",
            "test_cuda_ao_radial_reuse.py::test_emitted_ao_cuda_compiles_with_specialized_resources",
        ),
        (
            "cuda-compile",
            "Check Fock and proposal preflight without a GPU",
            "-R '^generativeqc_(fock_build|scf_proposal)_tests$' --output-on-failure",
        ),
        (
            "cuda-resources",
            "Compile five representative f classes with release resources",
            "tools/validate_f_shells.py --tier compile --smoke",
        ),
        (
            "cuda-resources",
            "Compile generated XC derivative consumers",
            "tools/validate_xc.py --tier cuda-compile",
        ),
        (
            "cuda-resources",
            "Compile native grid/XC region resources",
            "tools/validate_native_xc_resources.py",
        ),
        (
            "cuda-resources",
            "Compile generated r2SCAN stationary CUDA consumer",
            "tools/validate_stationary_cuda_compile.py",
        ),
        (
            "cuda-resources",
            "Compile generated TensorIR static-data artifacts",
            "tools/validate_tensor_cuda_compile.py",
        ),
    ],
)
def test_existing_gates_are_unconditional_and_owned_once(
    job: str, name: str, command: str
) -> None:
    step = _step(job, name)
    assert command in step
    assert "if:" not in step
    assert "continue-on-error:" not in step
    assert WORKFLOW.count(f"      - name: {name}\n") == 1


def test_resources_generate_the_native_cmake_source_without_rebuilding_library() -> (
    None
):
    production = _step("cuda-compile", "Configure production CUDA compile check")
    resources = _step("cuda-resources", "Configure native grid source generation")
    assert (
        production.split("        run: >-\n", 1)[1].strip()
        == resources.split("        run: >-\n", 1)[1].strip()
    )
    generation = _step(
        "cuda-resources", "Generate exact native grid source and prerequisite headers"
    )
    assert "cmake --build --preset cuda-release-sm120" in generation
    assert "--target generated/generated_grid_policy.cu --parallel 2" in generation
    assert "generativeqc_cuda_runtime_tests" not in _job("cuda-resources")
    validation = _step("cuda-resources", "Compile native grid/XC region resources")
    assert "--generated-dir build/cuda-release-sm120/generated" in validation
    assert (
        "--cmake-source build/cuda-release-sm120/generated/generated_grid_policy.cu"
        in validation
    )
    assert "--functional PBE --npoint 4096 --tile-points 256 --nao 96 --spins 2" in (
        validation
    )
    reports = _step("cuda-resources", "Preserve generated XC resource reports")
    assert "if: always()" in reports
    assert "build/xc-compile" in reports
    assert "build/native-xc-resource" in reports


@pytest.mark.skipif(
    not all(shutil.which(command) for command in ("cmake", "ninja", "ccache")),
    reason="requires the CI build tools",
)
def test_native_grid_file_target_generates_its_exact_header_closure(
    tmp_path: Path,
) -> None:
    """Exercise the real production declarations without enabling a CUDA compiler."""
    from generativeqc_compiler.dft.ao_cuda import emit_grid_source

    subprocess.run(["ccache", "--version"], check=True, capture_output=True, timeout=10)
    for name in ("tools", "python", "src", "include", "manifests", "upstream", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "dummy.cpp").write_text("int fixture;\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(CudaGridCodegen LANGUAGES CXX)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'include("{ROOT / "cmake/GenerativeQCGeneratedSources.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        "set(GENERATIVEQC_ENABLE_CUDA ON)\n"
        "set(CMAKE_CUDA_ARCHITECTURES 120)\n"
        "add_library(generativeqc STATIC dummy.cpp)\n"
        "generativeqc_register_host_generated_sources(generativeqc)\n"
        "generativeqc_register_cuda_generated_sources(generativeqc)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "generated/generated_grid_policy.cu",
            "--parallel",
            "2",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    generated = build / "generated"
    assert {path.name for path in generated.iterdir() if path.suffix != ".d"} == {
        "generated_grid_policy.cu",
        "generated_b3lyp_device.cuh",
        "generated_r2scan_device.cuh",
        "generated_wb97mv_device.cuh",
        "generated_split_hybrid_registry.cuh",
    }
    assert (generated / "generated_grid_policy.cu").read_text() == emit_grid_source(
        native_ks=True
    )[0]
    assert not list(build.rglob("*.o"))
    assert not list(build.glob("*.a"))
    assert not list(build.glob("*.so"))
