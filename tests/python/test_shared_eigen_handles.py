"""Run the real shared prepared handle owner against both supported host ABIs.

These probes qualify ownership, failure handling and host call order. They do
not claim GPU numerical, stream capture or real-device qualification.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "src/solver/cuda/symmetric_eigen_handles.cpp"
STUBS = ROOT / "tests/native/fixtures/shared_eigen_provider"


def _definition(source: str, prefix: str) -> str:
    start = source.index(prefix)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _consumer_lifetimes() -> str:
    gfn2 = (ROOT / "src/xtb/native/src/runtime/gfn2_cuda_execution.cu").read_text()
    rhf = (ROOT / "src/scf/cuda/resources.cpp").read_text()
    ordinary = (ROOT / "src/scf/cuda/eigensolver.cpp").read_text()
    # Keep complete production bodies unchanged. Only the enclosing GFN2 class
    # name changes; unrelated scientific fields become minimal host fixtures.
    return (
        "struct Gfn2Owner {\n"
        "int device_id{3}; cudaStream_t stream{}; cublasHandle_t blas{};\n"
        "shared::PreparedSymmetricEigenHandles eigen_handles;\n"
        "bool handles_created{}; std::unique_ptr<Prepared> prepared; std::mutex mutex;\n"
        + _definition(gfn2, "generativeqc_xtb_status_t ensure_handles(")
        + "\n"
        + _definition(gfn2, "~Impl()").replace("~Impl()", "~Gfn2Owner()", 1)
        + "\n};\n"
        + _definition(rhf, "CudaResources::~CudaResources()")
        + "\n"
        + _definition(ordinary, "void OrdinaryStreamEigensolver::cleanup()")
        + "\n"
        + _definition(
            ordinary, "OrdinaryStreamEigensolver::~OrdinaryStreamEigensolver()"
        )
        + "\n"
    )


@pytest.fixture(scope="module", params=("official", "cumetal-strided"))
def handle_probe(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: NativeCxx,
) -> Path:
    folder = tmp_path_factory.mktemp(f"shared-eigen-handles-{request.param}")
    (folder / "shared_eigen_lifetimes.inc").write_text(_consumer_lifetimes())
    (folder / "cuda_runtime_api.h").write_text(
        "#pragma once\n#include <cstddef>\n"
        "struct cudaStream; using cudaStream_t = cudaStream*;\n"
        "struct cudaEvent; using cudaEvent_t = cudaEvent*;\n"
        "enum cudaError_t { cudaSuccess = 0, cudaErrorUnknown = 999 };\n"
        "enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice,\n"
        "cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };\n"
        "cudaError_t cudaSetDevice(int); cudaError_t cudaGetDevice(int*);\n"
        "cudaError_t cudaStreamSynchronize(cudaStream_t);\n"
        "cudaError_t cudaStreamDestroy(cudaStream_t);\n"
        "cudaError_t cudaEventSynchronize(cudaEvent_t);\n"
        "cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);\n"
        "cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t,\n"
        "cudaMemcpyKind, cudaStream_t);\n"
    )
    (folder / "library_types.h").write_text(
        "#pragma once\nenum cudaDataType { CUDA_R_32F = 0, CUDA_R_64F = 1 };\n"
        "enum libraryPropertyType { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL };\n"
    )
    args = [
        "-std=c++20",
        "-O0",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I" + str(STUBS),
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
        "-I" + str(ROOT / "src/xtb/native/src"),
        "-I" + str(folder),
    ]
    if request.param == "cumetal-strided":
        args.append("-DTEST_CUMETAL_STRIDED")
    return required_native_cxx.build_executable(
        [
            OWNER,
            ROOT / "tests/native/test_shared_eigen_handles.cpp",
            ROOT / "tests/native/test_shared_eigen_handles_private_abi.cpp",
        ],
        folder / "probe",
        compile_args=args,
    )


@pytest.mark.parametrize("mode", ("gfn2", "scf-jacobi", "scf-generic"))
def test_configurations_preserve_order_and_settings(
    handle_probe: Path, mode: str
) -> None:
    subprocess.run([str(handle_probe), "success", mode], check=True, timeout=15)


@pytest.mark.parametrize("failure", range(1, 8))
@pytest.mark.parametrize("status", (1, 2, 6, 9))
def test_every_provider_failure_rolls_back_and_retries(
    handle_probe: Path, failure: int, status: int
) -> None:
    subprocess.run(
        [str(handle_probe), "failure", str(failure), str(status)],
        check=True,
        timeout=15,
    )


@pytest.mark.parametrize("operation", ("move", "duplicate", "private-abi"))
def test_exclusive_ownership_and_private_abi(
    handle_probe: Path, operation: str
) -> None:
    subprocess.run([str(handle_probe), operation], check=True, timeout=15)


@pytest.mark.parametrize("failure", range(10))
def test_actual_gfn2_setup_transaction_and_context_teardown(
    handle_probe: Path, failure: int
) -> None:
    subprocess.run(
        [str(handle_probe), "gfn2-lifetime", str(failure)], check=True, timeout=15
    )


@pytest.mark.parametrize("consumer", ("rhf-jacobi", "rhf-generic", "ordinary"))
def test_actual_scf_teardown_bodies(handle_probe: Path, consumer: str) -> None:
    subprocess.run(
        [str(handle_probe), "scf-lifetime", consumer], check=True, timeout=15
    )


def test_shared_reset_stays_inside_existing_release_scopes() -> None:
    rhf = (ROOT / "src/scf/cuda/resources.cpp").read_text()
    assert rhf.index("allocation_lock(") < rhf.index("eigen_handles_.reset()")
    df = (ROOT / "src/scf/cuda/df_plan_lifetime.cpp").read_text()
    release = _definition(df, "void release(")
    ordered = (
        "cudaSetDevice(plan.device_id)",
        "destroy_persistent_scf_state(",
        "destroy_ordinary_eigensystem(",
        "runtime::resource_cuda_free(plan.exchange_density_column_major)",
        "plan.eigen_handles.reset()",
        "cublasDestroy(plan.blas)",
        "cudaStreamDestroy(plan.stream)",
        "plan = {}",
    )
    assert [release.index(term) for term in ordered] == sorted(
        release.index(term) for term in ordered
    )


def test_handle_header_keeps_vendor_and_method_abi_private() -> None:
    header = OWNER.with_suffix(".hpp").read_text()
    includes = re.findall(r"^#include\s+[<\"]([^>\"]+)", header, re.MULTILINE)
    assert set(includes) <= {
        "cstddef",
        "cstdint",
        "utility",
        "solver/cuda/symmetric_eigen_provider.hpp",
    }
    for forbidden in (
        "cusolverDnHandle_t",
        "cudaStream_t",
        "generativeqc_status",
        "Gfn2Eigensolver",
    ):
        assert forbidden not in header


def test_handle_owner_has_no_numeric_or_execution_side_effects() -> None:
    source = OWNER.read_text()
    for name in (
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaGetDevice",
        "cudaSetDevice",
        "cudaStreamCreate",
        "cudaStreamDestroy",
        "malloc",
        "free",
        "query_symmetric_eigen",
        "launch_symmetric_eigen",
    ):
        assert not re.search(rf"\b{name}\w*\s*\(", source), name


def test_handle_sources_participate_in_build_and_prepared_identity() -> None:
    generator = (ROOT / "tools/generate_solver_lowering.py").read_text()
    dependencies = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    for source in (
        "src/solver/cuda/symmetric_eigen_handles.hpp",
        "src/solver/cuda/symmetric_eigen_handles.cpp",
    ):
        assert source in generator
        assert source in dependencies
    sources = (ROOT / "cmake/GenerativeQCSources.cmake").read_text()
    assert sources.count("src/solver/cuda/symmetric_eigen_handles.cpp") == 1


@pytest.mark.parametrize(
    "consumer",
    (
        "src/xtb/native/src/runtime/gfn2_cuda_execution.cu",
        "src/scf/cuda_rhf.cpp",
        "src/scf/cuda/resources.cpp",
        "src/scf/cuda/eigensolver.cpp",
        "src/scf/cuda/df_scf_library.cpp",
        "src/scf/cuda/df_scf_state.hpp",
        "src/scf/cuda/df_plan_setup.cpp",
        "src/scf/cuda/df_plan_lifetime.cpp",
    ),
)
def test_all_migrated_consumers_delete_raw_lifecycle_calls(consumer: str) -> None:
    source = (ROOT / consumer).read_text()
    raw_lifecycle = re.compile(
        r"\bcusolverDn(?:Create(?:Params|SyevjInfo)?|Destroy(?:Params|SyevjInfo)?"
        r"|SetStream|XsyevjSet(?:Tolerance|MaxSweeps|SortEig))\s*\("
    )
    assert not raw_lifecycle.search(source), consumer


@pytest.mark.parametrize(
    "owner",
    (
        "src/xtb/native/src/runtime/gfn2_cuda_execution.cu",
        "src/scf/cuda/resources.hpp",
        "src/scf/cuda/eigensolver.hpp",
        "src/scf/cuda/df_scf_state.hpp",
        "src/scf/cuda/df_plan_internal.hpp",
    ),
)
def test_all_prepared_owners_use_the_shared_resource(owner: str) -> None:
    assert "PreparedSymmetricEigenHandles" in (ROOT / owner).read_text()


@pytest.mark.parametrize(
    "header,owner,member",
    (
        (
            "scf/cuda/eigensolver.hpp",
            "generativeqc::scf::cuda_execution::OrdinaryStreamEigensolver",
            None,
        ),
        (
            "scf/cuda/resources.hpp",
            "generativeqc::scf::cuda_execution::CudaResources",
            "eigen_handles_",
        ),
        (
            "scf/cuda/df_scf_state.hpp",
            "generativeqc::scf::cuda_df::DeviceSolver",
            "handles",
        ),
        (
            "scf/cuda/df_plan_internal.hpp",
            "generativeqc::scf::CudaDensityFittingJkPlan",
            "eigen_handles",
        ),
    ),
)
def test_real_owner_headers_after_method_solver_namespace(
    header: str,
    owner: str,
    member: str | None,
    tmp_path: Path,
    required_native_cxx: NativeCxx,
) -> None:
    """Compile real headers when method-local solver already exists (CI order).

    Only vendor declarations are stubbed. No SCF or shared-owner declaration is
    extracted or replaced, so namespace lookup sees the real nested scopes.
    """
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\n#include <cstddef>\n"
        "struct cudaStream; using cudaStream_t = cudaStream*;\n"
        "struct cudaGraph; using cudaGraph_t = cudaGraph*;\n"
        "struct cudaGraphExec; using cudaGraphExec_t = cudaGraphExec*;\n"
        "enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, "
        "cudaErrorMemoryAllocation = 2, cudaErrorInvalidDevice = 101 };\n"
        "cudaError_t cudaGetDevice(int*); cudaError_t cudaSetDevice(int);\n"
        "cudaError_t cudaMalloc(void**, std::size_t);\n"
        "cudaError_t cudaFree(void*);\n"
        "cudaError_t cudaMallocAsync(void**, std::size_t, cudaStream_t);\n"
        "cudaError_t cudaFreeAsync(void*, cudaStream_t);\n"
        "cudaError_t cudaStreamSynchronize(cudaStream_t);\n"
        "cudaError_t cudaGraphDestroy(cudaGraph_t);\n"
        "cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);\n"
    )
    (tmp_path / "cuda_runtime.h").write_text('#include "cuda_runtime_api.h"\n')
    (tmp_path / "cublas_v2.h").write_text(
        "#pragma once\nstruct cublasContext; using cublasHandle_t = cublasContext*;\n"
    )
    source = tmp_path / "owner_include_order.cpp"
    source.write_text(
        "namespace generativeqc::scf::solver {}\n"
        '#include "' + header + '"\n'
        "#include <type_traits>\n#include <utility>\n"
        "using Shared = ::generativeqc::solver::cuda::PreparedSymmetricEigenHandles;\n"
        "using Owner = ::" + owner + ";\n"
        "static_assert(sizeof(Owner) >= sizeof(Shared));\n"
        + (
            "static_assert(std::is_same_v<decltype(std::declval<Owner>()."
            + member
            + "), Shared>);\n"
            if member
            else "static_assert(!std::is_copy_constructible_v<Owner>);\n"
        )
    )
    required_native_cxx.compile_object(
        source,
        tmp_path / "owner_include_order.o",
        args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(STUBS),
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
    )
