"""Execute CUDA DF bucket/seed owners with failing numerical providers.

The extracted production functions are compiled unchanged. Only integral,
CUDA and numerical operations are replaced, so this test needs no GPU and tests
status propagation, retry admission and arena lifetime rather than source order.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]

_SHIM = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "scf/density_fitting.hpp"
#include "scf/initial_guess/eigen_operation.hpp"

namespace generativeqc::runtime::df_progress {
void number(const char*, auto) {}
void label(const char*, const char*) {}
}
namespace generativeqc::scf {
using reference::Matrix;
using reference::EigenResult;
namespace host_trace {
struct Region { explicit Region(const char*) {} void finish() {} };
struct Item { explicit Item(std::size_t) {} };
enum class EigenReason { fallback };
template<class F> auto call(const char*, F f) { return f(); }
template<class F> auto with_reason(EigenReason, F f) { return f(); }
}

// Track externally meaningful attempts and concurrently live arena ownership.
// Two sources fail independently; a third successful neighbor must survive.
enum class Boundary { Seed, Iteration, Compact, FinalCompact, FinalHost };
Boundary boundary;
generativeqc_status injected_status{}, retry_status{};
std::array<unsigned, 3> core_attempts{}, core_densities{}, preparations{};
std::array<bool, 3> fail_source{true, true, false};
unsigned live_arenas{}, host_fock_calls{}, reference_eigen_calls{};
bool injection_enabled{true};
bool in_seed{}, in_finalization{};

std::size_t source_id(const core::System& system) {
  return static_cast<std::size_t>(system.atoms.at(0).atomic_number - 1);
}
struct CudaDensityFittingMetricDiagnostic { std::size_t system_index{}; };
struct CudaDensityFittingJkPlan { std::vector<std::size_t> sources; };
using CudaDensityFittingPlanPtr =
    std::unique_ptr<CudaDensityFittingJkPlan, void(*)(CudaDensityFittingJkPlan*)>;
void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan* plan) {
  if (plan) { assert(live_arenas > 0); --live_arenas; delete plan; }
}
std::size_t cuda_density_fitting_jk_plan_batch_size(const CudaDensityFittingJkPlan* p) {
  return p->sources.size();
}
bool cuda_density_fitting_scf_policy_matches(const CudaDensityFittingJkPlan*) { return true; }
auto cuda_density_fitting_pair_storage(const CudaDensityFittingJkPlan*) { return DfPairStorage::Dense; }
std::size_t cuda_density_fitting_scf_value_budget(const CudaDensityFittingJkPlan*) { return 0; }
unsigned cuda_density_fitting_scf_diis_history(const CudaDensityFittingJkPlan*) { return 8; }
bool one_electron_response_policy_matches(const DensityFittingScfData&, std::size_t, bool) { return true; }
bool df_response_policy_matches(const DensityFittingScfData&, std::size_t, double, bool) { return true; }
void trace_df_resolved_budget(const auto&) {}
void discard_density_fitting_tensor_storage(DensityFittingScfData&) {}
std::size_t df_seed_retained_capacity(const DensityFittingScfData&) { return 0; }
std::size_t cuda_density_fitting_seed_host_numeric_bytes(const CudaDensityFittingJkPlan*) { return 0; }
core::System density_fitting_auxiliary_for_geometry(const std::optional<core::System>&,
                                                   const core::System& system) { return system; }
DensityFittingScfData prepare_density_fitting_data(const core::System& system,
    const core::System&, double, int, std::size_t, bool, unsigned) {
  DensityFittingScfData data;
  data.raw.nbf = data.raw.naux = data.one_electron.nbf = 1;
  data.one_electron.overlap = {1.0};
  data.one_electron.hcore = {static_cast<double>(source_id(system) + 1)};
  return data;
}
std::vector<std::optional<DensityFittingScfData>> prepare_cuda_density_fitting_batch(
    const std::vector<core::System>& systems, const std::optional<core::System>&,
    double threshold, std::size_t budget, int device,
    std::vector<generativeqc_status>& statuses, bool forces, unsigned history) {
  std::vector<std::optional<DensityFittingScfData>> out;
  for (const auto& system : systems)
    out.emplace_back(prepare_density_fitting_data(system, system, threshold, device,
                                                 budget, forces, history));
  statuses.assign(systems.size(), GENERATIVEQC_STATUS_SUCCESS);
  return out;
}
CudaDensityFittingPlanPtr make_cuda_density_fitting_batch_plan(
    const std::vector<DensityFittingScfData>&, const ScfOptions&, int, std::size_t,
    bool, std::vector<CudaDensityFittingMetricDiagnostic>*,
    const std::vector<core::System>* systems, const std::vector<core::System>*) {
  auto* plan = new CudaDensityFittingJkPlan;
  for (const auto& system : *systems) plan->sources.push_back(source_id(system));
  ++live_arenas;
  return {plan, &destroy_cuda_density_fitting_jk_plan};
}
bool cuda_density_fitting_rhf_warm_matches(CudaDensityFittingJkPlan*, const Matrix&,
    const Matrix&, const Matrix&, const Matrix&, std::size_t, double) { return false; }
FockBuildSpec make_hf_fock_spec(FockSpin, FockApproximation) { return {}; }
ResolvedFockBuild resolve_fock_build(FockBuildSpec, FockBackend, double, double) { return {}; }

generativeqc_status solve_cuda_density_fitting_eigen(CudaDensityFittingJkPlan* plan,
    const Matrix&, const Matrix*, const Matrix*, Matrix& values, Matrix& vectors,
    CudaDfEigenDiagnostic&, std::string& detail, std::size_t slot) {
  const bool selected = in_seed ? boundary == Boundary::Seed
      : in_finalization ? (boundary == Boundary::FinalCompact || boundary == Boundary::FinalHost)
      : boundary == Boundary::Iteration;
  if (injection_enabled && selected && fail_source.at(plan->sources.at(slot))) {
    detail = "injected ordinary device eigen failure";
    return injected_status;
  }
  values = {1}; vectors = {1};
  return GENERATIVEQC_STATUS_SUCCESS;
}
EigenResult generalized_eigen(const Matrix&, const Matrix&, std::size_t) {
  ++reference_eigen_calls;
  throw std::runtime_error("unexpected reference eigensolver fallback");
}
namespace initial_guess {
class OverlapOrthogonalizer {};
Matrix prepare_overlap_orthogonalizer(const core::System&, const Matrix&, std::size_t,
                                     OverlapOrthogonalizer*, const EigenOperation&) { return {1}; }
Matrix normalized_warm_density(const core::System&, const auto&, const Matrix& d) { return d; }
void validate_preliminary_target(const core::System&, const ResolvedFockBuild&, const ScfOptions&) {}
std::optional<Matrix> prepare_huckel_initial_density(const core::System& system,
    const auto&, const PreliminaryOptions&, PreliminaryDiagnostic& diagnostic,
    const EigenOperation& eigen) {
  ++preparations.at(source_id(system));
  in_seed = true;
  try { eigen({1}, nullptr, nullptr, 1); }
  catch (...) { in_seed = false; throw; }
  in_seed = false;
  diagnostic.outcome = PreliminaryOutcome::Used;
  diagnostic.preliminary_iterations = 3;
  diagnostic.preliminary_fock_builds = 4;
  return Matrix{2};
}
}
int df_initial_orbital_request() { return 0; }
Matrix prepare_initial_density(const core::System& system, const auto&, const Matrix&,
    std::size_t, const Matrix*, std::optional<EigenResult>&, int,
    const initial_guess::EigenOperation&) {
  ++core_densities.at(source_id(system));
  return {2};
}
struct Diis { explicit Diis(unsigned) {} Matrix update(const Matrix& f, const Matrix&) { return f; } };
struct CudaDensityFittingDeviceScfItem {
  bool converged{}; unsigned iterations{}; double energy{}, energy_change{}, density_rms{};
};
unsigned cuda_df_iteration_diis_history(const ScfOptions& options) { return options.diis_history; }
generativeqc_status run_cuda_density_fitting_rhf_device_scf(CudaDensityFittingJkPlan* plan,
    const Matrix&, const Matrix&, const Matrix&, const std::vector<std::int32_t>&,
    const Matrix&, unsigned, double, double, Matrix& final_density,
    std::vector<CudaDensityFittingDeviceScfItem>& records, std::string&, const Matrix&, unsigned) {
  if (boundary == Boundary::Compact && injection_enabled) return injected_status;
  if (boundary == Boundary::Iteration || boundary == Boundary::FinalHost)
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  for (auto source : plan->sources) {
    final_density.push_back(2);
    records.push_back({true, 2, 20.0 + source, 0, 0});
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}
generativeqc_status execute_cuda_density_fitting_rhf_jk(CudaDensityFittingJkPlan* plan,
    const Matrix&, Matrix& j, Matrix& k, std::string&) {
  ++host_fock_calls;
  j.assign(plan->sources.size(), 0); k = j;
  return GENERATIVEQC_STATUS_SUCCESS;
}
double electronic_energy(const Matrix&, const Matrix& h, const Matrix&) { return 20 + h[0] - 1; }
Matrix commutator_residual(const Matrix&, const Matrix&, const Matrix&, std::size_t) { return {0}; }
Matrix density_from_orbitals(const Matrix&, std::size_t, std::size_t) { return {2}; }
double density_rms(const Matrix&, const Matrix&) { return 0; }

// The retry's physical solve is an injected numerical boundary. It asserts
// that the failed shared arena is already gone and the seed cannot recurse.
ScfResult run_rhf_density_fitting_cuda_impl(const core::System& system, const core::System&,
    const ScfOptions& options, int, const Matrix* density,
    initial_guess::OverlapOrthogonalizer*) {
  assert(live_arenas == 0 && !options.preliminary_guess && !density);
  ++core_attempts.at(source_id(system));
  if (retry_status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (retry_status != GENERATIVEQC_STATUS_SUCCESS && retry_status != GENERATIVEQC_STATUS_SCF_NOT_CONVERGED)
    throw generativeqc::Error(retry_status, "injected core target failure");
  ScfResult result;
  result.converged = retry_status == GENERATIVEQC_STATUS_SUCCESS;
  result.iterations = 5; result.fock_builds = 6;
  result.energy = 50 + source_id(system); result.density = {2};
  return result;
}
void finalize_density_fitting_rhf(const DensityFittingScfData&, const Matrix&, std::size_t,
    Matrix&, const ScfOptions&, ScfResult&, CudaDensityFittingJkPlan*, std::size_t, bool = false);
"""

_FINALIZE = r"""
void finalize_density_fitting_rhf(const DensityFittingScfData&, const Matrix&, std::size_t,
    Matrix& density, const ScfOptions&, ScfResult& result,
    CudaDensityFittingJkPlan* plan, std::size_t slot, bool) {
  in_finalization = true;
  try { device_df_eigen({1}, nullptr, nullptr, plan, slot); }
  catch (...) { in_finalization = false; throw; }
  in_finalization = false;
  result.density = density;
}
"""

_TAIL = r"""
} // namespace generativeqc::scf
using namespace generativeqc;
using namespace generativeqc::scf;

generativeqc_status status_from_name(const std::string& name) {
  if (name == "SUCCESS") return GENERATIVEQC_STATUS_SUCCESS;
  if (name == "CUDA_ERROR") return GENERATIVEQC_STATUS_CUDA_ERROR;
  if (name == "OUT_OF_MEMORY") return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  if (name == "NUMERICAL_FAILURE") return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  assert(name == "SCF_NOT_CONVERGED");
  return GENERATIVEQC_STATUS_SCF_NOT_CONVERGED;
}

int main(int argc, char** argv) {
  assert(argc == 5);
  boundary = static_cast<Boundary>(std::stoi(argv[1]));
  injected_status = status_from_name(argv[2]);
  retry_status = status_from_name(argv[3]);
  const bool cached = std::stoi(argv[4]) != 0;
  unsetenv("GENERATIVEQC_DF_REFERENCE_SETUP_EIGEN");
  unsetenv("GENERATIVEQC_DF_REFERENCE_ITERATION_EIGEN");
  std::vector<core::System> systems(3);
  for (std::size_t i = 0; i < systems.size(); ++i) {
    systems[i].atoms.resize(1);
    systems[i].atoms[0].atomic_number = static_cast<std::int32_t>(i + 1);
    systems[i].electron_count = 2;
  }
  ScfOptions options;
  options.max_iterations = 4;
  options.preliminary_guess.emplace();
  options.preliminary_guess->kind = initial_guess::PreliminaryKind::Huckel;
  std::vector<const Matrix*> densities(systems.size(), nullptr);
  CudaDensityFittingJkPlan* cached_plan = nullptr;
  std::vector<std::optional<DensityFittingScfData>> prepared_cache;
  const auto run = [&] {
    return cached ? run_rhf_density_fitting_cuda_bucket_cached(
        &cached_plan, systems, std::nullopt, options, densities, 0, nullptr,
        &prepared_cache, nullptr)
      : run_rhf_density_fitting_cuda_bucket(systems, std::nullopt, options, densities, 0, nullptr);
  };
  // Establish a same-orchestration success control for the untouched neighbor.
  injection_enabled = false;
  const auto control = run();
  destroy_cuda_density_fitting_jk_plan(cached_plan); cached_plan = nullptr;
  prepared_cache.clear();
  preparations.fill(0); core_attempts.fill(0); core_densities.fill(0);
  host_fock_calls = 0;
  injection_enabled = true;
  const auto results = run();
  assert(results.size() == systems.size());
  const bool terminal = injected_status == GENERATIVEQC_STATUS_CUDA_ERROR ||
                        injected_status == GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  if (boundary == Boundary::Compact && terminal) {
    for (const auto& result : results) assert(result.status == injected_status);
    assert(host_fock_calls == 0);
  } else {
    const auto& neighbor = results.back();
    const auto& expected = control.back();
    assert(neighbor.status == GENERATIVEQC_STATUS_SUCCESS);
    assert(neighbor.scf.energy == expected.scf.energy);
    assert(neighbor.scf.density == expected.scf.density);
    assert(neighbor.scf.iterations == expected.scf.iterations);
    assert(neighbor.scf.fock_builds == expected.scf.fock_builds);
    assert(neighbor.scf.preliminary_guess.outcome == initial_guess::PreliminaryOutcome::Used);
    assert(neighbor.scf.preliminary_guess.target_attempts == 1);
    assert(core_attempts.back() == 0);
  }
  for (std::size_t source = 0; source < systems.size(); ++source) {
    assert(preparations[source] == 1);
    if (!fail_source[source]) { assert(core_attempts[source] == 0); continue; }
    const auto& result = results[source];
    if (terminal) {
      assert(result.status == injected_status);
      assert(core_attempts[source] == 0 && core_densities[source] == 0);
      assert(result.scf.preliminary_guess.target_attempts <= 1);
    } else if (boundary == Boundary::Seed) {
      // An ordinary seed numerical rejection admits core D in the first target.
      assert(result.status == GENERATIVEQC_STATUS_SUCCESS);
      assert(result.scf.preliminary_guess.outcome == initial_guess::PreliminaryOutcome::PreparationFailed);
      assert(core_densities[source] == 1 && core_attempts[source] == 0);
    } else if (boundary == Boundary::Compact) {
      // Existing compact numerical refusal may use the host DIIS target path.
      assert(result.status == GENERATIVEQC_STATUS_SUCCESS);
      assert(host_fock_calls > 0 && core_attempts[source] == 0);
    } else {
      assert(core_attempts[source] == 1);
      assert(result.status == retry_status);
      assert(result.scf.preliminary_guess.outcome == initial_guess::PreliminaryOutcome::TargetRetried);
      assert(result.scf.preliminary_guess.target_attempts == 2);
      assert(result.scf.preliminary_guess.preliminary_iterations == 3);
      assert(result.scf.preliminary_guess.preliminary_fock_builds == 4);
      assert(!result.scf.preliminary_guess.work_counters_complete);
    }
  }
  assert(reference_eigen_calls == 0);
  destroy_cuda_density_fitting_jk_plan(cached_plan);
  assert(live_arenas == 0);
}
"""


def _owner_probe(source: str) -> str:
    """Compile whole owners; slice anchors identify them, never assert spelling."""

    def section(start: str, stop: str) -> str:
        begin = source.index(start)
        return source[begin : source.index(stop, begin)]

    execution = (ROOT / "src/scf/fock_execution.cpp").read_text()
    guard_start = execution.index("void reject_cuda_df_preliminary_guess(")
    guard_stop = execution.index(
        "\nResolvedFockBuild fock_strategy_for_execution(", guard_start
    )
    return (
        _SHIM
        + execution[guard_start:guard_stop]
        + section("EigenResult device_df_eigen(", "/** Native atomic preparation")
        + section(
            "[[maybe_unused]] std::optional<Matrix> prepare_df_huckel_density(",
            "/** Retained host tensors",
        )
        + section(
            "[[maybe_unused]] EigenResult iteration_df_eigen(",
            "/** Keep CPU/reference execution explicit",
        )
        + _FINALIZE
        + section(
            "std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket_impl(",
            "\nstd::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket_impl(",
        )
        + section("void retry_df_huckel_items(", "\n}  // namespace")
        + section(
            "std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket(",
            "\nstd::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket(",
        )
        + _TAIL
    )


@pytest.fixture(scope="module")
def df_error_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("requires host C++")
    directory = tmp_path_factory.mktemp("df_error_boundaries")
    probe = directory / "probe.cpp"
    probe.write_text(_owner_probe((ROOT / "src/scf/rhf.cpp").read_text()))
    binary = directory / "probe"
    compile_owner(compiler, directory, [probe], binary)
    return binary


@pytest.mark.parametrize("cached", [False, True], ids=["owned-arena", "cached-arena"])
@pytest.mark.parametrize(
    "failure", ["CUDA_ERROR", "OUT_OF_MEMORY", "NUMERICAL_FAILURE"]
)
@pytest.mark.parametrize(
    "boundary",
    range(5),
    ids=[
        "seed-admission",
        "target-eigen",
        "compact-dispatch",
        "compact-finalize",
        "host-finalize",
    ],
)
def test_bucket_typed_failure_boundaries(
    df_error_probe: Path, boundary: int, failure: str, cached: bool
) -> None:
    subprocess.run(
        [str(df_error_probe), str(boundary), failure, "SUCCESS", str(int(cached))],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "retry_failure",
    ["NUMERICAL_FAILURE", "CUDA_ERROR", "OUT_OF_MEMORY", "SCF_NOT_CONVERGED"],
)
@pytest.mark.parametrize(
    "boundary", [1, 3, 4], ids=["target-eigen", "compact-finalize", "host-finalize"]
)
def test_failed_core_retry_is_terminal(
    df_error_probe: Path, boundary: int, retry_failure: str
) -> None:
    subprocess.run(
        [str(df_error_probe), str(boundary), "NUMERICAL_FAILURE", retry_failure, "1"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("owner", ["fock_prepared.cpp", "cuda_fock_provider.cpp"])
def test_independent_provider_preserves_status(tmp_path: Path, owner: str) -> None:
    """The independent J/K and eigen adapters preserve terminal native errors."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("requires host C++")
    source = (ROOT / "src/scf" / owner).read_text()
    start = source.index("void checked(")
    end = source.index("\n}", start) + 2
    probe = tmp_path / "independent_status.cpp"
    probe.write_text(
        '#include <cassert>\n#include "generativeqc/generativeqc.hpp"\n'
        + source[start:end]
        + r"""
int main() {
  checked(GENERATIVEQC_STATUS_SUCCESS, "success");
  for (const auto status : {GENERATIVEQC_STATUS_CUDA_ERROR,
                           GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                           GENERATIVEQC_STATUS_NOT_IMPLEMENTED}) {
    try { checked(status, "native failure"); assert(false); }
    catch (const generativeqc::Error& error) { assert(error.status() == status); }
  }
  try { checked(GENERATIVEQC_STATUS_OUT_OF_MEMORY, "memory"); assert(false); }
  catch (const std::bad_alloc&) {}
  try { checked(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "input"); assert(false); }
  catch (const std::invalid_argument&) {}
}
"""
    )
    binary = tmp_path / "independent_status"
    compile_owner(compiler, tmp_path, [probe], binary)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
