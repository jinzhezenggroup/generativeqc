"""Real host DF policy/domain admission, without RHF, CUDA, or integral work."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <cassert>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "cc/solver.hpp"
#include "scf/cuda/df_source_domain.hpp"
namespace probe {
namespace core = generativeqc::core;
int rhf_calls = 0, source_calls = 0, allocations = 0;
unsigned expected_math = 0;
namespace runtime {
struct ExecutionContext {
  int backend() const { return GENERATIVEQC_BACKEND_CUDA; }
  bool cuda_requested() const { return true; }
  int device_id() const { return 0; }
};
namespace df_progress { struct Scope { explicit Scope(const char*) {} }; }
}
struct generativeqc_method_descriptor {};
struct MethodError : std::runtime_error {
  MethodError(int, const std::string& message) : std::runtime_error(message) {}
};
struct Reference {
  std::size_t reference_memory_budget_bytes = 100;
  int diis_history = 8;
  double screening_tolerance = 0;
};
namespace scf {
namespace cuda_execution = generativeqc::scf::cuda_execution;
using namespace cuda_execution;
enum class FockSpin { Restricted };
enum class FockBackend { Cpu, Cuda };
struct FockBuildSpec { unsigned derivative_order = 1; };
FockBuildSpec make_hf_fock_spec(FockSpin) { return {}; }
int resolve_fock_build(FockBuildSpec, FockBackend, double) { return 0; }
struct PreparedFockPlan {
  PreparedFockPlan(const core::System&, std::nullptr_t, int, int = -1) { ++allocations; }
};
struct CudaDensityFittingIntegralSourceImpl {};
struct CudaDensityFittingIntegralSource { void* implementation{}; };
void destroy_cuda_density_fitting_integral_source_impl(CudaDensityFittingIntegralSourceImpl* p) {
  delete p;
}
generativeqc_status create_cuda_density_fitting_integral_source_impl(
    int, const std::vector<core::System>&, const std::vector<core::System>&,
    CudaDensityFittingIntegralSourceImpl** source, std::vector<double>& metric,
    std::size_t& nbf, std::size_t& naux, std::string&, const CudaDfSourcePolicy& policy) {
  ++source_calls;
  assert(policy.value_math == expected_math);
  assert(policy.requested_value_mapping == 1);  // Frozen component mapping.
  *source = new CudaDensityFittingIntegralSourceImpl;
  metric = {1.0}; nbf = 1; naux = 1;
  return GENERATIVEQC_STATUS_SUCCESS;
}
"""

AFTER_FACTORY = r"""
}  // namespace scf
namespace posthf {
std::size_t source_capacity(const core::System&) { return 0; }
std::size_t rhf_reference_capacity(const core::System&, int, bool) { return 80; }
std::size_t checked_add(std::size_t a, std::size_t b) { return a + b; }
}
struct RccsdNativeState {
  std::size_t external_reservation_bytes = 19;
  struct { std::size_t numeric_capacity_bytes = 23; } diagnostic;
};
void validate_descriptor(const generativeqc_method_descriptor&, const runtime::ExecutionContext&) {}
std::size_t correlation_budget(const generativeqc_method_descriptor&) { return 100; }
using SolverOptions = generativeqc::cc::SolverOptions;
SolverOptions cc_options(const generativeqc_method_descriptor&, std::size_t) { return {}; }
Reference reference_options(const generativeqc_method_descriptor&, std::size_t) { return {}; }
RccsdNativeState execute_rccsd_prepared(
    runtime::ExecutionContext&, const core::System& orbital, Reference, SolverOptions options, std::size_t,
    scf::PreparedFockPlan*, const std::vector<double>*, bool* warm_fallback,
    std::unique_ptr<scf::PreparedFockPlan>*, const core::System* auxiliary, bool retain_df_response,
    const scf::cuda_execution::CudaDfSourcePolicy* policy) {
  ++rhf_calls;
  assert(auxiliary && policy && retain_df_response && !options.df_matrix_gemm);
  // A mid-RHF diagnostic change must not alter the already admitted source.
  setenv("GENERATIVEQC_DF_VALUE_MATH", "invalid-after-rhf", 1);
  setenv("GENERATIVEQC_DF_VALUE_MAPPING", "primitive", 1);
  scf::CudaDensityFittingIntegralSource* source = nullptr;
  std::vector<double> metric;
  std::size_t nbf = 0, naux = 0;
  std::string detail;
  const auto status = scf::create_cuda_density_fitting_integral_source(
      0, {orbital}, {*auxiliary}, &source, metric, nbf, naux, detail, policy);
  assert(status == GENERATIVEQC_STATUS_SUCCESS && source);
  scf::destroy_cuda_density_fitting_integral_source_impl(
      static_cast<scf::CudaDensityFittingIntegralSourceImpl*>(source->implementation));
  delete source;
  *warm_fallback = false;
  return {};
}
"""

MAIN = r"""
}  // namespace probe
int main(int argc, char** argv) {
  using namespace probe;
  assert(argc == 6);
  const std::string endpoint = argv[1], selection = argv[2];
  if (selection == "unset") unsetenv("GENERATIVEQC_DF_VALUE_MATH");
  else setenv("GENERATIVEQC_DF_VALUE_MATH", selection.c_str(), 1);
  setenv("GENERATIVEQC_DF_VALUE_MAPPING", "component", 1);
  const int expected = std::atoi(argv[5]);
  const bool admitted = expected >= 0;
  expected_math = admitted ? unsigned(expected) : 0U;
  core::System orbital, auxiliary;
  orbital.shells = {{0, unsigned(std::atoi(argv[3])), {{1.0, 1.0}}}};
  auxiliary.shells = {{0, unsigned(std::atoi(argv[4])), {{1.0, 1.0}}}};
  if (endpoint == "factory") {
    scf::CudaDensityFittingIntegralSource sentinel;
    auto* source = &sentinel;
    std::vector<double> metric{77.0};
    std::size_t nbf = 11, naux = 12;
    std::string detail;
    const auto status = scf::create_cuda_density_fitting_integral_source(
        0, {orbital}, {auxiliary}, &source, metric, nbf, naux, detail, nullptr);
    if (!admitted) {
      assert(status == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
      assert(source_calls == 0 && rhf_calls == 0);
      assert(source == &sentinel && metric == std::vector<double>{77.0});
      assert(nbf == 11 && naux == 12 && !detail.empty());
    } else {
      assert(status == GENERATIVEQC_STATUS_SUCCESS && source_calls == 1);
      scf::destroy_cuda_density_fitting_integral_source_impl(
          static_cast<scf::CudaDensityFittingIntegralSourceImpl*>(source->implementation));
      delete source;
    }
  } else {
    runtime::ExecutionContext execution;
    probe::generativeqc_method_descriptor descriptor;
    auto cache = std::make_unique<scf::PreparedFockPlan>(orbital, nullptr, 0);
    auto* original_cache = cache.get();
    allocations = 0;
    bool warm_fallback = true;
    const std::vector<double> density{17.0};
    RccsdNativeState output;
    try {
      output = run_rccsd_native_state(execution, orbital, descriptor, &cache, &density,
                                      &warm_fallback, 0, &auxiliary, true, false);
      assert(admitted && rhf_calls == 1 && source_calls == 1);
      assert(!warm_fallback);
    } catch (const MethodError& error) {
      assert(!admitted && std::string(error.what()).size() > 0);
      assert(rhf_calls == 0 && source_calls == 0 && allocations == 0);
      assert(cache.get() == original_cache && warm_fallback);
      assert(output.external_reservation_bytes == 19);
      assert(output.diagnostic.numeric_capacity_bytes == 23);
    }
    assert(density == std::vector<double>{17.0});
  }
}
"""


@pytest.fixture(scope="module")
def host_policy_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler/ccache unavailable")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    tmp = tmp_path_factory.mktemp("df-policy-preflight")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_one_electron_kernels.py"),
            "--derivatives",
            "--derivative-policy-output",
            str(tmp / "generated_one_electron_derivative_policy.cuh"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = (ROOT / "src/scf/cuda/df_source.cpp").read_text()
    start = source.index(
        "generativeqc_status create_cuda_density_fitting_integral_source("
    )
    end = source.index("\nvoid destroy_cuda_density_fitting_integral_source(", start)
    factory = source[start:end]
    source = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    start = source.index("RccsdNativeState run_rccsd_native_state(")
    end = source.index("\ngenerativeqc_status validate_rccsd_system", start)
    # Use the real declaration's defaults and SolverOptions fields while
    # retaining the intercepted RHF/source owners and their no-work assertions.
    header = (ROOT / "src/methods/rccsd_method.hpp").read_text()
    declaration = (
        "RccsdNativeState run_rccsd_native_state("
        + header.split("RccsdNativeState run_rccsd_native_state(", 1)[1].split(");", 1)[
            0
        ]
        + ");\n"
    )
    probe = tmp / "probe.cpp"
    probe.write_text(
        PREFIX + factory + AFTER_FACTORY + declaration + source[start:end] + MAIN
    )
    objects = []
    env = dict(os.environ, CCACHE_BASEDIR=str(ROOT))
    # Compile the complete real policy/domain translation units, not test reimplementations.
    for source in (
        probe,
        ROOT / "src/scf/cuda/df_source_domain.cpp",
        ROOT / "src/scf/cuda/rhf_policy.cpp",
    ):
        obj = tmp / (source.stem + ".o")
        subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O0",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffunction-sections",
                "-fdata-sections",
                "-DGENERATIVEQC_HAS_CUDA=1",
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
                f"-I{tmp}",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        objects.append(str(obj))
    executable = tmp / "probe"
    subprocess.run(
        [compiler, *objects, "-Wl,--gc-sections", "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("endpoint", ("admission", "factory"))
@pytest.mark.parametrize(
    "selection,orbital,auxiliary,expected",
    [
        ("unset", 3, 4, 0),
        ("auto", 3, 4, 0),
        ("generic", 3, 4, 0),
        ("polynomial", 3, 4, -1),
        ("rys", 3, 4, -1),
        ("candidate", 3, 4, -1),
        ("polynomial", 3, 3, 1),
        ("rys", 3, 3, 2),
        ("candidate", 3, 3, 3),
        ("invalid", 3, 3, -1),
        ("invalid", 3, 4, -1),
        ("auto", 4, 4, -1),
        ("generic", 3, 5, -1),
    ],
)
def test_real_policy_rejects_before_work_and_freezes_admitted_source(
    host_policy_probe: Path,
    endpoint: str,
    selection: str,
    orbital: int,
    auxiliary: int,
    expected: int,
) -> None:
    subprocess.run(
        [
            str(host_policy_probe),
            endpoint,
            selection,
            str(orbital),
            str(auxiliary),
            str(expected),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_source_setup_consumes_the_admitted_policy_without_environment_reads() -> None:
    setup = (ROOT / "src/scf/cuda/df_source_setup.cpp").read_text()
    assert "df_value_math_requested" not in setup
    assert "df_value_mapping_requested" not in setup
    assert "candidate->value_math = policy.value_math;" in setup
    assert "const auto requested_mapping = policy.requested_value_mapping;" in setup
    assert setup.index("cuda_df_value_domain(") < setup.index("cudaSetDevice(")
    method = " ".join((ROOT / "src/methods/rccsd_method.cpp").read_text().split())
    assert (
        "device, caller_bytes, retained_df_response != nullptr, correlation_policy"
        in method
    )
    assert (
        "provider_metrics, correlation_auxiliary, retain_df_response ? &state.df_source : nullptr, correlation_policy"
        in method
    )
    builder = (ROOT / "src/cc/df_source_cuda.cu").read_text()
    assert "source_n, source_q, detail, policy);" in builder


def test_policy_host_source_is_compiled_and_identity_bound() -> None:
    from generativeqc.autotune import _source_identity_paths

    source = "src/scf/cuda/df_source_domain.cpp"
    assert ROOT / source in _source_identity_paths(ROOT)
    cmake = (ROOT / "cmake/GenerativeQCSources.cmake").read_text()
    host_sources = cmake.split(
        "function(generativeqc_add_integrals_scf_sources target)", 1
    )[1].split("if(GENERATIVEQC_ENABLE_CUDA)", 1)[0]
    assert source in host_sources
