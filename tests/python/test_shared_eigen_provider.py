"""Trace real shared cuSOLVER lowering and consumer host adapters with fake ABI.

Both NVIDIA's opaque-pointer, enum-status ABI and CuMetal's explicit-stride
signature compile the actual production lowerer. These are host-call and routing
checks, not numerical, graph capture, real-device, or performance qualification.
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
LOWERER = ROOT / "src/solver/cuda/symmetric_eigen_provider.cpp"
STUBS = ROOT / "tests/native/fixtures/shared_eigen_provider"
CONSUMERS = (
    "src/xtb/native/src/backends/cuda/gfn2_eigensolver.cu",
    "src/scf/cuda/eigensolver.cpp",
    "src/scf/cuda/df_scf_library.cpp",
    "src/scf/cuda_rhf.cpp",
    "src/scf/cuda/df_eigensystem.cpp",
    "src/scf/cuda/df_plan_setup.cpp",
)


@pytest.fixture(scope="module", params=("official", "cumetal-strided"))
def provider_probe(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: NativeCxx,
) -> Path:
    folder = tmp_path_factory.mktemp(f"shared-eigen-{request.param}")
    (folder / "shared_eigen_consumers.inc").write_text(_consumer_definitions())
    (folder / "cuda_runtime_api.h").write_text(
        "#pragma once\nstruct cudaStream; using cudaStream_t = cudaStream*;\n"
    )
    (folder / "library_types.h").write_text(
        "#pragma once\nenum cudaDataType { CUDA_R_32F = 0, CUDA_R_64F = 1 };\n"
        "enum libraryPropertyType { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL };\n"
    )
    args = [
        "-std=c++17",
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
            LOWERER,
            ROOT / "tests/native/test_shared_eigen_provider.cpp",
            ROOT / "tests/native/test_shared_eigen_private_abi.cpp",
        ],
        folder / "probe",
        compile_args=args,
    )


@pytest.mark.parametrize("operation", ("query", "launch"))
@pytest.mark.parametrize("family", (0, 1, 2), ids=("jacobi", "xsyev-batched", "xsyevd"))
@pytest.mark.parametrize("vectors", (0, 1), ids=("values-only", "vectors"))
def test_actual_provider_calls(
    provider_probe: Path, operation: str, family: int, vectors: int
) -> None:
    subprocess.run(
        [str(provider_probe), operation, str(family), str(vectors)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_consumers_delegate_all_moved_raw_calls(consumer: str) -> None:
    source = (ROOT / consumer).read_text()
    raw_call = re.compile(
        r"\bcusolverDn(?:DsyevjBatched|XsyevBatched|Xsyevd)(?:_bufferSize)?\s*\("
    )
    assert not raw_call.search(source), (
        f"raw symmetric-eigen provider call left in {consumer}"
    )
    assert "solver/cuda/symmetric_eigen_provider.hpp" in source
    assert re.search(r"(?:query|launch)_symmetric_eigen\s*\(", source)


def test_shared_header_does_not_export_vendor_or_method_abi() -> None:
    header = (LOWERER.with_suffix(".hpp")).read_text()
    includes = re.findall(r"^#include\s+[<\"]([^>\"]+)", header, re.MULTILINE)
    assert set(includes) <= {"cstddef", "cstdint"}
    assert "cusolverDnHandle_t" not in header
    assert "generativeqc_status" not in header
    assert "Gfn2Eigensolver" not in header


def test_shared_lowering_has_no_ownership_or_synchronization() -> None:
    source = LOWERER.read_text()
    forbidden_calls = (
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cusolverDnCreate",
        "cusolverDnDestroy",
        "cusolverDnSetStream",
        "malloc",
        "free",
        "launch_sanitize_inactive_solver_input_kernel",
    )
    for name in forbidden_calls:
        assert not re.search(rf"\b{name}\w*\s*\(", source), name
    assert "cuda_compat::xsyev_batched" in source


def _definition(source: str, prefix: str) -> str:
    """Extract a complete production declaration/body verbatim, including braces."""
    start = source.index(prefix)
    opening = source.index("{", start)
    level = 1
    end = opening + 1
    while level:
        level += (source[end] == "{") - (source[end] == "}")
        end += 1
    if end < len(source) and source[end] == ";":
        end += 1
    return source[start:end]


def _consumer_definitions() -> str:
    """Keep adapters/status/shape logic real; substitute only other device work."""
    gfn2_path = ROOT / CONSUMERS[0]
    gfn2 = gfn2_path.read_text()
    gfn2_header = gfn2_path.with_suffix(".cuh").read_text()
    scf = (ROOT / CONSUMERS[1]).read_text()
    scf_header = (ROOT / "src/scf/cuda/eigensolver.hpp").read_text()
    batch_header = (ROOT / "src/scf/cuda_batch.hpp").read_text()
    parts = [
        "namespace gfn2 { namespace eigen_provider = ::generativeqc::solver::cuda;",
        "inline constexpr std::int64_t kMaximumInt64 = std::numeric_limits<std::int64_t>::max();",
    ]
    for declaration in (
        "enum class Gfn2EigensolverLaunchStatus",
        "struct Gfn2EigensolverLaunchResult",
        "struct Gfn2EigensolverBucket {",
        "struct Gfn2EigensolverBucketActivity",
        "struct Gfn2EigensolverDeviceWorkspace",
        "enum class Gfn2EigensolverStrategy",
        "struct Gfn2EigensolverOptions",
        "struct Gfn2EigensolverWorkspaceRequirements",
    ):
        parts.append(_definition(gfn2_header, declaration))
    begin = gfn2_header.index(
        "inline constexpr std::int32_t kGfn2JacobiMinimumOrbitals"
    )
    end = gfn2_header.index("/* Controls whether", begin)
    parts.append(gfn2_header[begin:end])
    for definition in (
        "template <typename T>\nbool is_aligned(",
        "bool checked_multiply(",
        "Gfn2EigensolverLaunchResult invalid_argument(",
        "Gfn2EigensolverLaunchResult cusolver_failure(",
        "Gfn2EigensolverLaunchResult launch_success(",
    ):
        parts.append(_definition(gfn2, definition))
    parts.append(GFN2_TRIDIAGONAL_STUB)
    for definition in (
        "Gfn2EigensolverLaunchResult symmetric_eigensolve(",
        "Gfn2EigensolverLaunchResult query_gfn2_eigensolver_bucket_workspace_cuda(",
        "Gfn2EigensolverLaunchResult query_gfn2_spin_eigensolver_bucket_workspace_cuda(",
        "Gfn2EigensolverLaunchResult query_gfn2_jacobi_bucket_workspace_cuda(",
    ):
        parts.append(_definition(gfn2, definition))
    parts.append("} // namespace gfn2")
    parts.append(
        "namespace scf { namespace eigen_provider = ::generativeqc::solver::cuda;"
    )
    parts.append(_definition(batch_header, "enum class CudaEigensolverFamily"))
    parts.append(_definition(scf_header, "struct EigensolverResources"))
    parts.append(_definition(scf_header, "struct EigensolverProfileLaunch"))
    for definition in (
        "generativeqc_status cuda_status(",
        "generativeqc_status solver_status(",
        "bool provider_eigensolver(",
        "generativeqc_status launch_solver(",
    ):
        parts.append(_definition(scf, definition))
    parts.append("} // namespace scf")
    parts.append(
        "namespace df { namespace eigen_provider = ::generativeqc::solver::cuda;"
    )
    df_state = (ROOT / "src/scf/cuda/df_scf_state.hpp").read_text()
    df_runtime = (ROOT / "src/scf/cuda/df_runtime.cpp").read_text()
    df_library = (ROOT / CONSUMERS[2]).read_text()
    parts.append(_definition(df_state, "struct DeviceSolver"))
    parts.append(_definition(df_runtime, "generativeqc_status solver_failure("))
    parts.append(_definition(df_library, "generativeqc_status solve_device_batch("))
    parts.append("} // namespace df")
    return "\n".join(parts)


GFN2_TRIDIAGONAL_STUB = r"""
int tridiagonal_calls = 0;
Gfn2EigensolverLaunchResult tridiagonal_symmetric_eigensolve(
    cusolverDnHandle_t, const Gfn2EigensolverBucket&, double*, double*,
    const Gfn2EigensolverDeviceWorkspace&, int*, cudaStream_t) noexcept {
  ++tridiagonal_calls;
  return launch_success();
}
"""


@pytest.mark.parametrize("family", (0, 1), ids=("jacobi", "xsyev-batched"))
@pytest.mark.parametrize("vectors", (0, 1), ids=("values-only", "vectors"))
@pytest.mark.parametrize("operation", ("gfn2-query", "gfn2-launch"))
def test_actual_gfn2_adapters(
    provider_probe: Path, operation: str, family: int, vectors: int
) -> None:
    subprocess.run(
        [str(provider_probe), operation, str(family), str(vectors)],
        check=True,
        timeout=15,
    )


@pytest.mark.parametrize("family", (0, 1, 2), ids=("jacobi", "xsyev-batched", "xsyevd"))
def test_actual_scf_launch_adapter(provider_probe: Path, family: int) -> None:
    subprocess.run(
        [str(provider_probe), "scf-launch", str(family), "1"], check=True, timeout=15
    )


def test_provider_sources_invalidate_generated_metadata() -> None:
    generator = (ROOT / "tools/generate_solver_lowering.py").read_text()
    cmake = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    for dependency in (
        "src/solver/cuda/symmetric_eigen_provider.hpp",
        "src/solver/cuda/symmetric_eigen_provider.cpp",
        "src/solver/cuda/cusolver_compat.hpp",
        "src/scf/cuda_eigensolver_policy.hpp",
    ):
        assert dependency in generator
        assert dependency in cmake
    sources = (ROOT / "cmake/GenerativeQCSources.cmake").read_text()
    assert sources.count("src/solver/cuda/symmetric_eigen_provider.cpp") == 1


@pytest.mark.parametrize("family", (0, 1), ids=("jacobi", "xsyev-batched"))
def test_actual_df_launch_adapter(provider_probe: Path, family: int) -> None:
    subprocess.run(
        [str(provider_probe), "df-launch", str(family), "1"], check=True, timeout=15
    )


def test_invalid_family_fails_without_provider_or_mutation(
    provider_probe: Path,
) -> None:
    subprocess.run(
        [str(provider_probe), "invalid-family", "0", "1"], check=True, timeout=15
    )


@pytest.mark.parametrize("family", (1, 2), ids=("xsyev-batched", "xsyevd"))
def test_generic_provider_retains_int64_dimensions(
    provider_probe: Path, family: int
) -> None:
    subprocess.run(
        [str(provider_probe), "dimension-width", str(family), "1"],
        check=True,
        timeout=15,
    )


@pytest.mark.parametrize("family", (0, 1, 2), ids=("jacobi", "xsyev-batched", "xsyevd"))
def test_private_uint32_abi_crosses_shared_provider(
    provider_probe: Path, family: int
) -> None:
    subprocess.run(
        [str(provider_probe), "private-abi", str(family), "1"], check=True, timeout=15
    )
