"""Full GFN host-call preservation, with device work replaced at launch boundaries.

The baseline is the unedited production bodies at 6a02e4b1. The live bodies,
validators, provider, workspace binding, and generalized lowering are compiled.
Every kernel argument and CUDA/cuBLAS/cuSOLVER host call is traced. This is a
host orchestration gate, never a device numerical or performance qualification.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from test_shared_eigen_provider import ROOT, STUBS, _consumer_definitions, _definition

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx

GFN = ROOT / "src/xtb/native/src/backends/cuda/gfn2_eigensolver.cu"
BASELINE = (
    ROOT / "tests/native/fixtures/shared_eigen_provider/gfn_generalized_baseline.inc"
)
ENTRY_POINTS = (
    "static Gfn2EigensolverLaunchResult solve_eigensystems_impl(",
    "static Gfn2EigensolverLaunchResult solve_spin_eigensystems_impl(",
    "Gfn2EigensolverLaunchResult enqueue_capacity_eigensolver_body(",
    "Gfn2EigensolverLaunchResult enqueue_capacity_backtransform_body(",
)


def test_baseline_and_shared_host_helpers_are_pinned() -> None:
    """A helper reused by both traces cannot silently change the baseline."""
    manifest = json.loads(BASELINE.with_suffix(".json").read_text())
    assert (
        hashlib.sha256(BASELINE.read_bytes()).hexdigest() == manifest["baseline_sha256"]
    )
    for helper in manifest["unchanged_host_helpers"]:
        source = (ROOT / helper["path"]).read_text()
        definition = _definition(source, helper["prefix"])
        assert _token_digest(definition) == helper["token_sha256"], (
            helper["path"],
            helper["prefix"],
        )


def test_all_gfn_transform_phases_use_the_shared_service() -> None:
    source = GFN.read_text()
    assert '#include "solver/cuda/generalized_eigen.hpp"' in source
    adapter = _definition(source, "Gfn2EigensolverLaunchResult generalized_transform(")
    assert "shared::reduce_generalized_eigen(" in adapter
    assert "shared::recover_generalized_eigen(" in adapter
    for prefix, expected in zip(ENTRY_POINTS, (2, 2, 1, 1), strict=True):
        body = _last_definition(source, prefix)
        assert body.count("generalized_transform(") == expected
        assert "triangular_solve(" not in body and "cublasDtrsmBatched(" not in body


def _last_definition(source: str, prefix: str) -> str:
    # Capacity bodies have forward declarations. Extract the last occurrence.
    return _definition(source[source.rindex(prefix) :], prefix)


def _launches(source: str) -> str:
    """Replace launch syntax only; every argument/expression remains production."""
    transformed = re.sub(
        r"\b(\w+)<<<(.*?)>>>\s*\(",
        lambda m: f'trace_kernel("{m[1]}", {m[2]} )(',
        source,
        flags=re.DOTALL,
    )
    assert "<<<" not in transformed and ">>>" not in transformed
    return transformed


def _host_definitions() -> tuple[str, str]:
    source = GFN.read_text()
    header = GFN.with_suffix(".cuh").read_text()
    schema = (
        ROOT / "src/xtb/native/src/backends/common/gfn2_plan_schema.hpp"
    ).read_text()
    geometry = (GFN.parent / "gfn2_geometry.cuh").read_text()
    types = [
        _definition(schema, "enum class Gfn2PlanMemorySpace"),
        _definition(schema, "struct Gfn2WavefunctionLayoutView"),
        _definition(geometry, "struct Gfn2GeometryEpochDevice"),
    ]
    for name in (
        "Gfn2EigensolverDeviceBatch",
        "Gfn2EigensolverOverlapCache",
        "Gfn2EigensolverDeviceResults",
    ):
        types.append(_definition(header, "struct " + name + " {"))
    # Serialization is field-wise, so padding cannot hide or fabricate changes.
    serialization = []
    for declaration in types + [
        _definition(header, "struct " + name + " {")
        for name in ("Gfn2EigensolverBucket", "Gfn2EigensolverDeviceWorkspace")
    ]:
        match = re.match(r"struct (\w+)", declaration)
        if not match:
            continue
        stripped = re.sub(r"/\*.*?\*/|//[^\n]*", "", declaration, flags=re.DOTALL)
        fields = re.findall(r"\b(\w+)\s*=.*?;", stripped)
        assert fields
        serialization.append(
            "inline void append(Trace& t, const gfn2::"
            + match[1]
            + "& v) {"
            + "".join("append(t, v." + field + ");" for field in fields)
            + "}"
        )
    common = [
        re.search(rf"constexpr int {name}\s*=.*?;", source)[0]
        for name in ("kThreadsPerSystem", "kSpinPrepareThreads")
    ]
    for prefix in (
        "bool checked_add(",
        "bool ranges_overlap(std::int64_t",
        "struct AddressRange",
        "bool make_range(",
        "bool make_byte_range(",
        "bool ranges_overlap(const AddressRange",
        "template <std::size_t Count>\nbool pairwise_disjoint(",
        "template <std::size_t FirstCount, std::size_t SecondCount>\nbool disjoint_sets(",
        "Gfn2EigensolverLaunchResult cuda_failure(",
        "Gfn2EigensolverLaunchResult cublas_failure(",
        "bool valid_options(",
        "bool valid_bucket_plan(",
        "bool valid_spin_bucket_plan(",
        "bool valid_workspace(",
        "bool valid_spin_workspace(",
        "bool valid_cache(",
        "bool valid_solve_ranges(",
        "bool valid_spin_solve_ranges(",
        "Gfn2EigensolverLaunchResult check_kernel_launch(",
        "Gfn2EigensolverLaunchResult prepare_launch_sequence(",
        "Gfn2EigensolverLaunchResult configure_solver(",
        "Gfn2EigensolverLaunchResult configure_blas(",
        "static Gfn2EigensolverLaunchResult validate_spin_solve_buckets(",
        "static Gfn2EigensolverLaunchResult prepare_spin_solve_bucket(",
    ):
        common.append(_definition(source, prefix))
    # The consumer owns status translation. The numerical lowering is included
    # from its real headers, not duplicated in the probe.
    for name in (
        "generalized_transform",
        "generalized_reduce",
        "generalized_recover",
        "triangular_solve",
    ):
        prefix = "Gfn2EigensolverLaunchResult " + name + "("
        if prefix in source:
            common.append(_definition(source, prefix))
    for prefix in (
        "struct TridiagonalProviderWorkspace",
        "bool append_workspace_bytes(",
        "bool make_tridiagonal_provider_workspace(",
    ):
        common.append(_definition(source, prefix))
    common.append(
        _definition(
            source, "Gfn2EigensolverLaunchResult tridiagonal_symmetric_eigensolve("
        ).replace(
            "tridiagonal_symmetric_eigensolve(",
            "actual_tridiagonal_symmetric_eigensolve(",
            1,
        )
    )
    common.extend(_last_definition(source, prefix) for prefix in ENTRY_POINTS)
    for name in (
        "enqueue_gfn2_eigensolver_capacity_cuda",
        "enqueue_gfn2_backtransform_capacity_cuda",
    ):
        common.append(_definition(source, "Gfn2EigensolverLaunchResult " + name + "("))
    baseline = BASELINE.read_text()
    baseline = "constexpr double baseline_kOne = 1.0;\n" + baseline
    baseline = (
        "constexpr int baseline_kThreadsPerSystem = 128, baseline_kSpinPrepareThreads = 256;\n"
        + baseline
    )
    for name in (
        "triangular_solve",
        "solve_eigensystems_impl",
        "solve_spin_eigensystems_impl",
        "enqueue_capacity_eigensolver_body",
        "enqueue_capacity_backtransform_body",
        "kThreadsPerSystem",
        "kSpinPrepareThreads",
        "kOne",
    ):
        baseline = re.sub(rf"\b{name}\b", "baseline_" + name, baseline)
    definitions = "namespace gfn2 {\n" + "\n".join(types) + "\n}\n"
    definitions += "namespace orchestration {\n" + "\n".join(serialization) + "\n}\n"
    bodies = (
        "namespace gfn2 {\n" + _launches("\n".join(common) + "\n" + baseline) + "\n}\n"
    )
    library = (ROOT / "src/scf/cuda/df_scf_library.cpp").read_text()
    runtime = (ROOT / "src/scf/cuda/df_runtime.cpp").read_text()
    bodies += "namespace df {\n"
    bodies += _definition(runtime, "generativeqc_status blas_failure(") + "\n"
    for name in ("scf_gemm_strided", "scf_gemm", "scf_generalized_transform"):
        bodies += _definition(library, "generativeqc_status " + name + "(") + "\n"
    bodies += "}\n" + _canonical_regions()
    return definitions, bodies


@pytest.fixture(scope="module", params=("official", "cumetal-strided"))
def orchestration_probe(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: NativeCxx,
) -> Path:
    folder = tmp_path_factory.mktemp("generalized-preservation-" + request.param)
    (folder / "shared_eigen_consumers.inc").write_text(_consumer_definitions())
    definitions, bodies = _host_definitions()
    (folder / "generalized_gfn_types.inc").write_text(definitions)
    (folder / "generalized_gfn_bodies.inc").write_text(bodies)
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
        "-DCUDART_VERSION=12080",
        "-I" + str(STUBS),
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
        "-I" + str(ROOT / "src/xtb/native/src"),
        "-I" + str(folder),
    ]
    if request.param == "cumetal-strided":
        args.append("-DTEST_CUMETAL_STRIDED")
    compiler = replace(required_native_cxx, base_dir=ROOT)
    return compiler.build_executable(
        [
            ROOT / "src/solver/cuda/generalized_eigen.cpp",
            ROOT / "src/solver/cuda/symmetric_eigen_provider.cpp",
            ROOT / "src/solver/cuda/symmetric_eigen_workspace.cpp",
            ROOT / "src/solver/cuda/symmetric_eigen_handles.cpp",
            ROOT / "tests/native/test_shared_eigen_private_abi.cpp",
            ROOT / "tests/native/test_generalized_eigen_private_abi.cpp",
            ROOT / "tests/native/test_generalized_eigen_preservation.cpp",
        ],
        folder / "probe",
        compile_args=args,
    )


@pytest.mark.parametrize(
    "scenario",
    (
        "restricted",
        "spin",
        "capacity-solve",
        "capacity-recover",
        "canonical",
        "canonical-adapter",
        "canonical-ordinary",
        "canonical-rhf",
        "canonical-uhf",
        "identity",
        "private-abi",
        "unused-capacity",
        "invalid",
    ),
)
def test_complete_gfn_host_preservation(
    orchestration_probe: Path, scenario: str
) -> None:
    subprocess.run(
        [str(orchestration_probe), scenario],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


# Preserve literals and multi-character operators while ignoring formatting and
# comments. Comparing the complete token stream retains call order, conditions,
# buffer expressions, scope lifetime and every surrounding orchestration step.
_CPP_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|'
    r"(?:u8|u|U|L)?'(?:\\.|[^'\\])*'|[A-Za-z_]\w*|"
    r"\d(?:[\w.']|[eEpP][+-])*|::|->\*|->|\.\.\.|<<=|>>=|"
    r"<=>|==|!=|<=|>=|&&|\|\||\+\+|--|<<|>>|[+*/%&|^!=-]=|\S",
    re.DOTALL,
)


def _token_digest(source: str) -> str:
    tokens = [
        token
        for token in _CPP_TOKEN.findall(source)
        if not token.startswith(("//", "/*"))
    ]
    return hashlib.sha256("\0".join(tokens).encode()).hexdigest()


def _expand_canonical(source: str, *, ordinary: bool) -> str:
    """Expand only the replaced 2/1-GEMM phases to their original call sites."""

    def expand(match: re.Match[str]) -> str:
        args = [arg.strip() for arg in match[2].split(",")]
        assert len(args) == 8, args
        plan, recovery, batch, n, matrix, x, temporary, detail = args
        assert recovery in ("false", "true")
        status = match[1]
        if recovery == "true":
            return (
                f"{status} = scf_gemm({plan}, false, {batch}, {n}, {x}, "
                f"{matrix}, {temporary}, {detail});"
            )
        first = (
            f"{status} = scf_gemm({plan}, false, {batch}, {n}, {matrix}, "
            f"{x}, {temporary}, {detail});"
        )
        second = (
            f"{status} = scf_gemm({plan}, true, {batch}, {n}, {x}, "
            f"{temporary}, {matrix}, {detail});"
        )
        if not ordinary:
            second = "{" + second + "}"
        return first + f"if ({status} == GENERATIVEQC_STATUS_SUCCESS) " + second

    expanded, count = re.subn(
        r"\b(\w+)\s*=\s*scf_generalized_transform\(([^;]+)\);", expand, source
    )
    assert count == (4 if "d_alpha_fock" in source else 2), count
    return expanded


@pytest.mark.parametrize("name", ("df_eigensystem", "df_rhf_scf", "df_uhf_scf"))
def test_full_canonical_consumer_orchestration_tokens(name: str) -> None:
    manifest = json.loads(BASELINE.with_suffix(".json").read_text())
    source = (ROOT / "src/scf/cuda" / (name + ".cpp")).read_text()
    expanded = _expand_canonical(source, ordinary=name == "df_eigensystem")
    assert _token_digest(expanded) == manifest["canonical_consumers"][name]


def _canonical_regions() -> str:
    parts = [
        (
            "namespace canonical { using df::scf_gemm; using df::scf_generalized_transform; "
            "using df::solve_device_batch; namespace cuda_execution = scf; "
            "using scf::CudaEigensolverFamily; namespace trace = runtime::cuda_trace; "
            "namespace host = runtime::host_trace;"
        )
    ]
    for name in ("df_eigensystem", "df_rhf_scf", "df_uhf_scf"):
        live = (ROOT / "src/scf/cuda" / (name + ".cpp")).read_text()
        for baseline in (False, True):
            source = (
                _expand_canonical(live, ordinary=name == "df_eigensystem")
                if baseline
                else live
            )
            function = ("baseline_" if baseline else "live_") + name
            if name == "df_eigensystem":
                start = source.index(
                    "    if (orthogonalizer) {",
                    source.index('"upload ordinary DF eigen inputs"'),
                )
                end = source.index("    {\n      trace::TraceRegion download(", start)
                parts.append(
                    f"generativeqc_status {function}(CudaDensityFittingJkPlan* plan, "
                    "dfao::OrdinaryEigensystem* state, bool orthogonalizer, double*& published, "
                    "std::string& detail) { const auto n = plan->nbf; "
                    "generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS; struct { int solver_calls{}; } diagnostic;\n"
                    + source[start:end]
                    + "published = output; return status; }\n"
                )
            else:
                first_call = "scf_gemm" if baseline else "scf_generalized_transform"
                start = source.index("    iteration_status = " + first_call + "(")
                if name == "df_rhf_scf":
                    end = source.index("    launch_build_device_density_kernel", start)
                else:
                    end = source.index(
                        "      launch_update_device_uhf_convergence_kernel", start
                    )
                region = source[start:end]
                if name == "df_uhf_scf":
                    region += "}\n"
                parts.append(
                    f"generativeqc_status {function}(CudaDensityFittingJkPlan* plan, "
                    "CanonicalState* state, CanonicalBuffers& buffers, bool occupied_exchange, "
                    "std::string& detail) { "
                    "const std::size_t batch_size = 3, expected = batch_size * plan->nbf * plan->nbf; "
                    "[[maybe_unused]] constexpr unsigned kThreads = 256; "
                    "[[maybe_unused]] const auto blocks_for = [](std::size_t n) { return unsigned((n+255)/256); }; "
                    "generativeqc_status iteration_status = GENERATIVEQC_STATUS_SUCCESS; "
                    + "".join(
                        f"[[maybe_unused]] auto* {field} = buffers.{field};"
                        for field in (
                            "d_fock",
                            "d_orthogonalizer",
                            "d_temporary",
                            "d_eigenvalues",
                            "d_info",
                            "d_alpha_fock",
                            "d_beta_fock",
                            "d_alpha_eigenvalues",
                            "d_beta_eigenvalues",
                            "d_alpha_info",
                            "d_beta_info",
                            "d_alpha_occupied",
                            "d_beta_occupied",
                            "d_next_alpha",
                            "d_next_beta",
                        )
                    )
                    + "(void)expected; (void)occupied_exchange;\n"
                    + region
                    + "return iteration_status; }\n"
                )
    return "\n".join(parts) + "}\n"
