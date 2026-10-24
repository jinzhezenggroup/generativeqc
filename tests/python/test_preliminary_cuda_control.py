"""Execute the shared KS scheduler and prepared lifecycle with host doubles.

The production control bodies are compiled unchanged. Provider/data doubles
exercise seed precedence, bounded retries, exception identity, history lifetime,
and final-state revocation; this is not CUDA or physical SCF qualification.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def between(text: str, begin: str, end: str) -> str:
    """Locate executable production code, never keep a copied control branch."""
    start = text.index(begin)
    return text[start : text.index(end, start)]


HARNESS = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "generativeqc/generativeqc.hpp"
#include "runtime/compiled_execution_region.hpp"
#include "scf/coulomb_exchange_work.hpp"
#include "scf/initial_guess/preliminary_types.hpp"
#include "scf/target_fock_work.hpp"
namespace scf = generativeqc::scf;
namespace initial_guess = scf::initial_guess;
namespace runtime {
using generativeqc::runtime::CompiledExecutionBinding;
using generativeqc::runtime::CompiledExecutionRegion;
struct ExecutionResourceSnapshot {};
struct CpuRetainedCapacity { explicit CpuRetainedCapacity(std::size_t) {} };
void sample_cpu_capacity(std::size_t) {}
}
namespace dft {
// CUDA_SEED_ORIGIN
bool semilocal_family_requires_tau(int) { return false; }
struct Diagnostic {
  scf::CoulombExchangeWork coulomb_exchange_work = scf::CoulombExchangeWork::empty();
  std::uint64_t fock_builds{};
  bool initial_density_used{};
  // Sentinel follows the exported history through the real result adapter.
  std::shared_ptr<int> history;
};
}
using dft::CudaKsSeedOrigin;
namespace nlc {
enum class Vv10Variant { vv10, rvv10 };
struct Parameters { Vv10Variant variant{Vv10Variant::vv10}; };
}
struct MethodError : generativeqc::Error { using Error::Error; };
namespace generativeqc::scf {
struct ScfResult {
  bool converged{}, initial_density_used{};
  unsigned iterations{}, precision{}, incremental_direct_jk{}, precision_work{};
  double energy{}, energy_change{}, density_rms{}, physical_residual_rms{};
  std::uint64_t fock_builds{};
  initial_guess::PreliminaryDiagnostic preliminary_guess;
  dft::Diagnostic dft_diagnostic;
};
struct HfWarmState {
  std::vector<double> coordinates, density;
  double energy{}, energy_change{}, density_rms{};
  unsigned iterations{};
};
}
struct Result {
  struct { unsigned iterations{}; double energy_change{}, residual_rms{}; bool converged{}; }
      convergence;
  double energy{}, physical_residual_rms{};
  generativeqc_backend executed_backend{};
  std::uint64_t fock_builds{};
  unsigned precision{}, incremental_direct_jk{}, precision_work{};
  initial_guess::PreliminaryDiagnostic preliminary_guess;
  std::optional<dft::Diagnostic> ks_diagnostic;
};
struct BatchItemResult {
  generativeqc_status status{GENERATIVEQC_STATUS_INTERNAL_ERROR};
  Result calculation;
  std::size_t bucket_id{};
  bool warm_start_used{}, warm_start_fallback{};
  std::optional<scf::TargetFockWork> target_fock_work;
};
using Coordinates = std::vector<std::optional<std::vector<double>>>;
// RESULT_ADAPTER
// CUDA_STATUS_CHECK
void check_seed_eigen_info(int info) {
  // CUDA_SEED_EIGEN_CHECK
}
std::string scenario;
std::weak_ptr<int> discarded_history;
struct Call { unsigned seed; bool reuse, warm, updates, export_density;
  CudaKsSeedOrigin origin;
  bool phase_source_cold{};
};
struct CudaKsPlan {
  // The real begin() admission also resets optional nonlocal initialization.
  // These controls stay disabled for this scheduler/lifecycle fixture.
  bool device_nonlocal{}, incremental_direct_jk{};
  unsigned spins{1};
  struct { unsigned max_iterations{100}; } options;
  struct { bool any_lower_precision() const { return false; } } precision_schedule;
  struct Nonlocal { nlc::Parameters parameters() const { return {}; } };
  Nonlocal* nonlocal_correlation{};
  bool nonlocal_seed_trace{}, nonlocal_outer_active{}, nonlocal_seed_stage{};
  bool nonlocal_frozen_stage{}, pending_nonlocal_frozen{}, pending_nonlocal_seed{};
  double nonlocal_inner_residual_tolerance{};
  unsigned nonlocal_inner_iterations{}, nonlocal_exact_epoch_iterations{};
  unsigned nonlocal_seed_iterations{}, nonlocal_exact_iterations{};
  struct Impl {
    bool is_active{}, is_failed{}, warm_ready{}, warm_updates{true};
    bool final_state_ready{}, final_frame_ready{}, final_stationary_weights_ready{};
    bool final_fitted_projection_ready{};
    unsigned final_generation{};
    void clear_warm_state() { warm_ready = false; }
  };
  std::unique_ptr<Impl> impl_ = std::make_unique<Impl>();
  std::vector<Call> calls;
  bool history_retained_at_retry{}, stale_state_at_begin{};
  bool failure_injection{true};
  unsigned warm_replacements{};
  bool has_warm_start() const noexcept;
  bool failed() const noexcept;
  void set_warm_start_updates(bool) noexcept;
  void clear_warm_start() noexcept;
  void invalidate_final_state() noexcept;
  int transfers() const { return 0; }
  void fail_if_requested(const std::string& stage) {
    if (!failure_injection) return;
    const bool second = scenario.starts_with("second-");
    auto error = second ? scenario.substr(7) : scenario;
    std::string failure_stage = "begin";
    for (const auto* prefix : {"enqueue-", "finish-"})
      if (error.starts_with(prefix)) {
        failure_stage = std::string(prefix).substr(0, std::string(prefix).size()-1);
        error = error.substr(std::string(prefix).size());
      }
    if (stage != failure_stage) return;
    if (error == "both-numerical")
      check(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "numerical failure");
    if (calls.size() != (second ? 2U : 1U)) return;
    if (error == "numerical") check(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "numerical failure");
    if (error == "seed-eigen") check_seed_eigen_info(1);
    if (error == "seed-eigen-invalid") check_seed_eigen_info(-1);
    if (error == "invalid") check(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "invalid seed");
    if (error == "device") check(GENERATIVEQC_STATUS_CUDA_ERROR, "device fault");
    if (error == "runtime") throw std::runtime_error("untyped runtime/device fault");
    if (error == "allocation") check(GENERATIVEQC_STATUS_OUT_OF_MEMORY, "allocation failure");
    if (error == "overflow") throw std::overflow_error("epoch exhausted");
    if (error == "logic") throw std::logic_error("pending iteration");
  }
  void prepare_phase_source(bool cold, bool = false) { calls.back().phase_source_cold = cold; }
  void begin(const std::vector<double>* seed, bool reuse, CudaKsSeedOrigin origin) {
    stale_state_at_begin |= impl_->final_state_ready;
    invalidate_final_state();
    if (!calls.empty()) history_retained_at_retry |= !discarded_history.expired();
    const auto* input = seed;
    const bool reuse_warm = reuse, warm_ready = impl_->warm_ready;
    const auto seed_origin = origin;
    // CUDA_WARM_ADMISSION
    calls.push_back({seed ? static_cast<unsigned>(seed->at(0)) : 0U,
                     reuse, use_warm, impl_->warm_updates, false, origin});
    bool phase_source_cold{};
    const bool retain_range_source = false;
    // CUDA_PHASE_SOURCE_ADMISSION
    impl_->is_failed = true;
    impl_->is_active = false;
    fail_if_requested("begin");
    impl_->is_failed = false;
    impl_->is_active = true;
  }
  void enqueue_iteration() { fail_if_requested("enqueue"); }
  bool finish_iteration() {
    fail_if_requested("finish");
    impl_->is_active = false;
    impl_->is_failed = scenario == "both-physical" ||
        (calls.size() == 1 && (scenario == "physical" || scenario == "failed-converged"));
    return false;
  }
  std::optional<std::uint64_t> completed_fock_builds() const {
    return calls.size() == 1 ? (std::uint64_t{1} << 33) + 7U : 5U;
  }
  scf::ScfResult result(bool export_density) {
    calls.back().export_density = export_density;
    const bool first = calls.size() == 1;
    const bool nonconverged = scenario == "both-nonconverged" ||
        (first && (scenario == "nonconverged" || scenario.starts_with("second-") ||
                   scenario == "census-unknown" || scenario == "census-overflow"));
    scf::ScfResult value;
    value.converged = !nonconverged && (!impl_->is_failed || scenario == "failed-converged");
    value.iterations = first ? 7U : 3U;
    value.fock_builds = *completed_fock_builds();
    value.dft_diagnostic.coulomb_exchange_work = {
        true, first ? (std::uint64_t{1} << 33) + 11 : 5,
        first ? (std::uint64_t{1} << 34) + 22 : 10};
    if (first && scenario == "census-unknown")
      value.dft_diagnostic.coulomb_exchange_work.complete = false;
    if (first && scenario == "census-overflow")
      value.dft_diagnostic.coulomb_exchange_work.coulomb_applications =
          std::numeric_limits<std::uint64_t>::max();
    if (scenario == "census-zero")
      value.dft_diagnostic.coulomb_exchange_work = scf::CoulombExchangeWork::empty();
    value.dft_diagnostic.history = std::make_shared<int>(1);
    if (first) discarded_history = value.dft_diagnostic.history;
    if (value.converged && !impl_->is_failed) {
      impl_->final_state_ready = true;
      if (impl_->warm_updates) { impl_->warm_ready = true; ++warm_replacements; }
    }
    return value;
  }
};
// CUDA_METHODS
struct System { std::vector<double> coordinates{0, 0, 0}; };
namespace core { using System = ::System; }
namespace geometry {
std::vector<double> coordinates(const System& system) { return system.coordinates; }
bool valid_coordinates(const std::vector<double>& positions, const System& system) {
  return positions.size() == system.coordinates.size() &&
      std::all_of(positions.begin(), positions.end(), [](double x) { return std::isfinite(x); });
}
void set_coordinates(System& system, const std::vector<double>& positions) {
  system.coordinates = positions;
}
// BIND_BATCH_ITEM
}
void add_transfers(int& previous, int current) { previous += current; }
struct ExecutionPlan { bool automatic_program{}; int semilocal_family{}; };
const char* ks_semilocal_family_name(const ExecutionPlan&) { return "probe"; }
struct Plan {
  System system_;
  CudaKsPlan cuda;
  std::optional<initial_guess::PreliminaryOptions> policy{std::in_place};
  unsigned preparations{};
  bool preparation_skipped{};
  Plan() { policy->kind = initial_guess::PreliminaryKind::Minao; }
  const System& system() const { return system_; }
  CudaKsPlan* cuda_plan() { return &cuda; }
  const auto& initial_guess_policy() const { return policy; }
  void invalidate_final_state() { cuda.invalidate_final_state(); }
  void clear_warm_start() { cuda.clear_warm_start(); }
  void add_prepared_native_forces(Result&) { throw std::logic_error("unexpected force request"); }
  void apply_d4(Result&) {}
  scf::ScfResult run_cpu(const std::vector<double>*, bool, bool, bool) {
    throw std::logic_error("unexpected CPU execution");
  }
  std::optional<std::vector<double>> prepare_cold_initial_guess(
      initial_guess::PreliminaryDiagnostic& diagnostic) {
    ++preparations;
    if (scenario == "preparation-runtime") throw std::runtime_error("preparation failed");
    diagnostic.outcome = preparation_skipped ? initial_guess::PreliminaryOutcome::BudgetSkipped
                                            : initial_guess::PreliminaryOutcome::Used;
    diagnostic.preparation_numeric_capacity = 123;
    diagnostic.preparation_seconds = 0.25;
    if (preparation_skipped) return std::nullopt;
    return std::vector<double>{2};
  }
};
class BatchWarmStarts {
 public:
  virtual ~BatchWarmStarts() = default;
  virtual void clear_warm_starts() = 0;
};
// PREPARED_BATCH_INTERFACE
// PREPARED_BATCH_LIFECYCLE
// FAILURE_CLASSIFICATION
struct Owner : PreparedBatch, BatchWarmStarts {
  struct Item {
    generativeqc_property_flags prepared_properties{};
    std::exception_ptr failure;
    std::unique_ptr<Plan> plan = std::make_unique<Plan>();
    std::optional<scf::HfWarmState> warm;
    bool resident_warm{};
    int retired_transfers{};
    std::optional<std::size_t> stationary_device_allowance;
  };
  std::vector<System> systems_{1};
  std::vector<Item> items_{1};
  bool warm_enabled_{true}, warm_updates_{true};
  generativeqc_backend backend_{GENERATIVEQC_BACKEND_CUDA};
  ExecutionPlan execution_plan_;
  std::size_t size() const noexcept override { return items_.size(); }
  std::size_t host_numeric_capacity(std::size_t = 0) { return 0; }
  void materialize_warm(std::size_t) {}
  std::unique_ptr<Plan> make_plan(const System&, std::optional<std::size_t>) {
    return std::make_unique<Plan>();
  }
  // SHARED_OWNER_CONTROL
  // CLEAR_WARM_STARTS
};
int main(int argc, char** argv) {
  if (argc != 3) return 100;
  const std::string route = argv[1];
  if (route == "eigen-info") {
    std::string status = "returned";
    try { check_seed_eigen_info(std::stoi(argv[2])); }
    catch (const std::invalid_argument&) { status = "invalid"; }
    catch (const generativeqc::Error& error) {
      status = error.status() == GENERATIVEQC_STATUS_NUMERICAL_FAILURE ? "numerical" : "other";
    }
    std::cout << "{\"status\":\"" << status << "\"}";
    return 0;
  }
  scenario = argv[2];
  if (route == "neighbor") {
    Owner owner;
    owner.systems_.resize(2);
    owner.items_.resize(2);
    owner.items_[1].plan->cuda.failure_injection = false;
    const auto results = owner.execute({}, false);
    const auto& failed = results[0].calculation.ks_diagnostic->coulomb_exchange_work;
    const auto& neighbor = results[1].calculation.ks_diagnostic->coulomb_exchange_work;
    std::cout << "{\"failed_complete\":" << failed.complete
              << ",\"neighbor_complete\":" << neighbor.complete
              << ",\"neighbor_j\":" << neighbor.coulomb_applications
              << ",\"neighbor_k\":" << neighbor.exchange_applications << "}";
    return 0;
  }
  Owner owner;
  auto& item = owner.items_[0];
  auto& plan = *item.plan;
  auto& cuda = plan.cuda;
  const bool resident = route == "warm" || route == "both" || route == "no-reuse";
  cuda.impl_->warm_ready = resident;
  if (resident || route == "explicit") {
    item.warm.emplace();
    item.warm->density = {3};
    item.resident_warm = resident && route != "both";
  }
  // Every public execution must revoke an old final result before admission.
  cuda.impl_->final_state_ready = true;
  owner.warm_enabled_ = route != "no-reuse";
  owner.warm_updates_ = route != "frozen";
  plan.preparation_skipped = route == "preparation-skipped";
  if (route == "disabled") plan.policy.reset();
  Result result;
  std::string status = "returned", initial_status;
  const auto execute = [&]() {
    try {
      result = owner.execute_single(false);
      status = "returned";
    } catch (const generativeqc::Error& error) {
      status = error.status() == GENERATIVEQC_STATUS_NUMERICAL_FAILURE ? "numerical" : "device";
    } catch (const std::bad_alloc&) { status = "allocation";
    } catch (const std::invalid_argument&) { status = "invalid";
    } catch (const std::overflow_error&) { status = "overflow";
    } catch (const std::logic_error&) { status = "logic";
    } catch (const std::runtime_error&) { status = "runtime"; }
  };
  execute();
  const bool revoked_after_first = !cuda.impl_->final_state_ready;
  if (route == "recover") {
    initial_status = status;
    scenario = "success";
    execute();
  }
  const auto& d = result.preliminary_guess;
  std::cout << "{\"status\":\"" << status << "\",\"calls\":" << cuda.calls.size()
            << ",\"initial_status\":\"" << initial_status << "\""
            << ",\"revoked_after_first\":" << revoked_after_first
            << ",\"preparations\":" << plan.preparations
            << ",\"converged\":" << result.convergence.converged
            << ",\"kind\":" << d.requested_kind
            << ",\"outcome\":" << static_cast<unsigned>(d.outcome)
            << ",\"attempts\":" << d.target_attempts
            << ",\"discarded_iterations\":" << d.discarded_target_iterations
            << ",\"discarded_focks\":" << d.discarded_target_fock_builds
            << ",\"complete\":" << d.work_counters_complete
            << ",\"capacity\":" << d.preparation_numeric_capacity
            << ",\"seconds\":" << d.preparation_seconds
            << ",\"returned_iterations\":" << result.convergence.iterations
            << ",\"returned_focks\":" << result.fock_builds
            << ",\"warm_ready\":" << cuda.has_warm_start()
            << ",\"warm_replacements\":" << cuda.warm_replacements
            << ",\"final_ready\":" << cuda.impl_->final_state_ready
            << ",\"stale_state\":" << cuda.stale_state_at_begin
            << ",\"retained_history\":" << cuda.history_retained_at_retry;
  const auto* value_work = result.ks_diagnostic ?
      &result.ks_diagnostic->coulomb_exchange_work : nullptr;
  const bool certified = value_work && value_work->complete;
  std::cout << ",\"jk_complete\":" << certified << ",\"jk_j\":";
  if (certified) std::cout << value_work->coulomb_applications;
  else std::cout << "null";
  std::cout << ",\"jk_k\":";
  if (certified) std::cout << value_work->exchange_applications;
  else std::cout << "null";
  std::cout << ",\"trace\":[";
  for (unsigned i = 0; i < cuda.calls.size(); ++i) {
    const auto& call = cuda.calls[i];
    if (i) std::cout << ",";
    std::cout << "{\"seed\":" << call.seed << ",\"reuse\":" << call.reuse
              << ",\"warm\":" << call.warm << ",\"updates\":" << call.updates
              << ",\"export\":" << call.export_density
              << ",\"native_cold\":" << (call.origin == CudaKsSeedOrigin::NativeColdGuess) << "}";
  }
  std::cout << "],\"phase_source_cold\":[";
  for (unsigned i = 0; i < cuda.calls.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << cuda.calls[i].phase_source_cold;
  }
  std::cout << "]}";
}
"""


@pytest.fixture(scope="module")
def cuda_control_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    cuda = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    header = (ROOT / "src/dft/cuda_ks.hpp").read_text()
    owner = (ROOT / "src/methods/dft_method.cpp").read_text()
    batch = owner[owner.index("class KsPreparedBatch final") :]
    lifecycle = (ROOT / "src/methods/prepared_execution.cpp").read_text()
    pieces = {
        "BIND_BATCH_ITEM": between(
            (ROOT / "src/methods/geometry.hpp").read_text(),
            "inline core::System bind_batch_item(",
            "\n}  // namespace generativeqc::methods::geometry",
        ),
        "PREPARED_BATCH_INTERFACE": between(
            (ROOT / "src/methods/method.hpp").read_text(),
            "class PreparedBatch {",
            "\n/** Exception carrying",
        ),
        # This fixture tests seed precedence, not the independent initializer.
        # Compile the real warm predicate without pulling unrelated optional
        # solver setup into the scheduler double.
        "CUDA_WARM_ADMISSION": between(cuda, "    const bool use_warm =", ";") + ";",
        "CUDA_PHASE_SOURCE_ADMISSION": between(
            cuda,
            "      phase_source_cold =",
            "\n      // The bounded device-control prototype",
        ),
        "CUDA_METHODS": "\n".join(
            [
                between(
                    cuda,
                    "bool CudaKsPlan::failed()",
                    "\nvoid CudaKsPlan::enqueue_iteration",
                ),
                between(
                    cuda,
                    "bool CudaKsPlan::has_warm_start()",
                    "\nstd::vector<double> CudaKsPlan::warm_density",
                ),
                between(
                    cuda,
                    "void CudaKsPlan::set_warm_start_updates",
                    "\ngenerativeqc_status CudaKsPlan::final_state_token",
                ),
            ]
        ),
        "CUDA_SEED_ORIGIN": between(header, "enum class CudaKsSeedOrigin", ";") + ";",
        "CUDA_STATUS_CHECK": between(
            cuda, "void check(generativeqc_status", "\ntemplate <class Function>"
        ),
        "CUDA_SEED_EIGEN_CHECK": between(
            cuda, "    if (info < 0)", "    // The solver emits column-major orbitals"
        ),
        "RESULT_ADAPTER": between(
            owner, "Result adapt_result(", "\n/** The method's global ledger"
        ),
        "FAILURE_CLASSIFICATION": between(
            (ROOT / "src/methods/execution_failure.hpp").read_text(),
            "inline generativeqc_status item_exception_status(",
            "\n}  // namespace generativeqc::methods",
        ),
        "SHARED_OWNER_CONTROL": between(
            batch,
            "  void invalidate_result()",
            "\n  /** Borrow the sole scientific port",
        ),
        "CLEAR_WARM_STARTS": between(
            batch, "  void clear_warm_starts()", "\n  std::size_t warm_density_size"
        ),
        "PREPARED_BATCH_LIFECYCLE": between(
            lifecycle,
            "runtime::CompiledExecutionBinding owner_binding(",
            "\n}  // namespace",
        )
        + "\n"
        + between(
            lifecycle,
            "PreparedBatch::PreparedBatch(",
            "\n}  // namespace generativeqc::methods",
        ),
    }
    source = HARNESS
    for marker, production in pieces.items():
        source = source.replace(f"// {marker}", production)
    directory = tmp_path_factory.mktemp("preliminary-cuda-control")
    path = directory / "probe.cpp"
    path.write_text(source)
    return native_cxx.build_executable(
        [path],
        directory / "probe",
        compile_args=[
            "-std=c++20",
            "-O0",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ],
    )


def probe(binary: Path, route: str = "cold", scenario: str = "success") -> dict:
    completed = subprocess.run(
        [str(binary), route, scenario],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(completed.stdout)


def test_single_execute_prepares_minao_through_the_shared_scheduler(
    cuda_control_probe: Path,
) -> None:
    result = probe(cuda_control_probe)
    assert result["calls"] == result["preparations"] == 1
    assert result["trace"] == [
        {"seed": 2, "reuse": 0, "warm": 0, "updates": 1, "export": 0, "native_cold": 1}
    ]
    assert result["status"] == "returned" and result["converged"]
    assert (result["kind"], result["outcome"], result["attempts"]) == (3, 2, 1)
    assert result["complete"] and not result["stale_state"]
    assert result["capacity"] == 123 and result["seconds"] == 0.25


@pytest.mark.parametrize(
    "route,seed,warm", [("warm", 0, 1), ("explicit", 3, 0), ("both", 3, 0)]
)
def test_explicit_and_available_warm_seeds_precede_preliminary_policy(
    cuda_control_probe: Path, route: str, seed: int, warm: int
) -> None:
    result = probe(cuda_control_probe, route)
    assert result["calls"] == 1 and result["preparations"] == 0
    assert result["trace"][0]["seed"] == seed and result["trace"][0]["warm"] == warm
    assert not result["trace"][0]["native_cold"]
    assert (result["kind"], result["outcome"], result["attempts"]) == (3, 1, 1)


@pytest.mark.parametrize("route", ["no-reuse", "frozen"])
def test_reuse_and_publication_permissions_do_not_skip_cold_preparation(
    cuda_control_probe: Path, route: str
) -> None:
    result = probe(cuda_control_probe, route)
    assert result["preparations"] == 1 and result["trace"][0]["seed"] == 2
    assert result["warm_replacements"] == 0
    assert result["trace"][0]["updates"] == 0
    assert result["warm_ready"] == (route == "no-reuse")


@pytest.mark.parametrize("scenario", ["nonconverged", "physical", "failed-converged"])
def test_prepared_cuda_failure_gets_one_fresh_hcore_attempt(
    cuda_control_probe: Path, scenario: str
) -> None:
    result = probe(cuda_control_probe, scenario=scenario)
    assert result["calls"] == 2 and result["preparations"] == 1
    assert result["trace"] == [
        {"seed": 2, "reuse": 0, "warm": 0, "updates": 1, "export": 0, "native_cold": 1},
        {"seed": 0, "reuse": 0, "warm": 0, "updates": 1, "export": 0, "native_cold": 0},
    ]
    assert result["status"] == "returned" and result["converged"]
    assert (result["kind"], result["outcome"], result["attempts"]) == (3, 5, 2)
    exported = scenario == "nonconverged"
    assert result["discarded_iterations"] == (7 if exported else 0)
    assert result["discarded_focks"] == ((1 << 33) + 7 if exported else 0)
    assert result["returned_iterations"] == 3 and result["returned_focks"] == 5
    assert result["complete"] == exported
    assert result["jk_complete"] == exported
    assert result["jk_j"] == ((1 << 33) + 16 if exported else None)
    assert result["jk_k"] == ((1 << 34) + 32 if exported else None)
    assert result["final_ready"]
    assert result["warm_replacements"] == 1
    assert not result["stale_state"] and not result["retained_history"]
    assert result["capacity"] == 123 and result["seconds"] == 0.25


@pytest.mark.parametrize(
    "scenario,status",
    [("both-nonconverged", "returned"), ("both-physical", "numerical")],
)
def test_unsuccessful_hcore_retry_is_terminal(
    cuda_control_probe: Path, scenario: str, status: str
) -> None:
    result = probe(cuda_control_probe, scenario=scenario)
    assert result["calls"] == 2 and result["preparations"] == 1
    assert result["status"] == status and not result["converged"]
    assert not result["final_ready"] and not result["warm_ready"]
    if status == "returned":
        assert result["attempts"] == 2 and result["complete"]


@pytest.mark.parametrize(
    "error,status",
    [
        ("device", "device"),
        ("runtime", "runtime"),
        ("allocation", "allocation"),
        ("overflow", "overflow"),
        ("logic", "logic"),
    ],
)
@pytest.mark.parametrize("prefix,calls", [("", 1), ("second-", 2)])
def test_non_numerical_exceptions_never_trigger_an_extra_attempt(
    cuda_control_probe: Path, error: str, status: str, prefix: str, calls: int
) -> None:
    result = probe(cuda_control_probe, scenario=prefix + error)
    assert result["calls"] == calls and result["status"] == status
    assert result["preparations"] == 1
    assert not result["final_ready"] and not result["warm_ready"]
    assert not result["stale_state"] and not result["retained_history"]


@pytest.mark.parametrize(
    "route",
    ["disabled", "preparation-skipped"],
)
def test_unseeded_physical_failures_do_not_retry(
    cuda_control_probe: Path, route: str
) -> None:
    result = probe(cuda_control_probe, route, "physical")
    assert result["calls"] == 1 and result["status"] == "numerical"
    assert result["preparations"] == (route == "preparation-skipped")
    assert result["warm_ready"] == (route in ("warm", "both"))
    assert result["warm_replacements"] == 0 and not result["final_ready"]


@pytest.mark.parametrize("route", ["disabled", "preparation-skipped"])
def test_disabled_or_skipped_preparation_keeps_core_behavior(
    cuda_control_probe: Path, route: str
) -> None:
    result = probe(cuda_control_probe, route)
    assert result["calls"] == 1 and result["trace"][0]["seed"] == 0
    assert result["preparations"] == (route == "preparation-skipped")
    assert result["outcome"] == (4 if route == "preparation-skipped" else 0)
    assert result["attempts"] == (1 if route == "preparation-skipped" else 0)


@pytest.mark.parametrize(
    "scenario", ["numerical", "seed-eigen", "invalid", "seed-eigen-invalid"]
)
@pytest.mark.parametrize("route", ["cold", "frozen"])
def test_typed_numerical_exception_retries_with_an_incomplete_work_census(
    cuda_control_probe: Path, scenario: str, route: str
) -> None:
    result = probe(cuda_control_probe, route, scenario)
    assert result["calls"] == 2 and result["preparations"] == 1
    assert result["status"] == "returned" and result["converged"]
    assert (result["kind"], result["outcome"], result["attempts"]) == (3, 5, 2)
    assert result["discarded_iterations"] == result["discarded_focks"] == 0
    assert not result["complete"]
    assert result["trace"][0]["seed"] == 2
    assert result["trace"][1]["seed"] == result["trace"][1]["reuse"] == 0
    assert result["warm_replacements"] == (route != "frozen")
    assert result["final_ready"] and not result["stale_state"]
    assert not result["retained_history"]
    assert result["capacity"] == 123 and result["seconds"] == 0.25


@pytest.mark.parametrize(
    "scenario,status",
    [
        ("both-numerical", "numerical"),
        ("second-numerical", "numerical"),
        ("second-seed-eigen", "numerical"),
        ("second-invalid", "invalid"),
        ("second-seed-eigen-invalid", "invalid"),
    ],
)
def test_hcore_numerical_exception_is_terminal(
    cuda_control_probe: Path, scenario: str, status: str
) -> None:
    result = probe(cuda_control_probe, scenario=scenario)
    assert result["calls"] == 2 and result["preparations"] == 1
    assert result["status"] == status
    assert not result["final_ready"] and not result["warm_ready"]


@pytest.mark.parametrize(
    "route",
    ["disabled", "preparation-skipped"],
)
def test_unprepared_numerical_exception_remains_visible(
    cuda_control_probe: Path, route: str
) -> None:
    result = probe(cuda_control_probe, route, "numerical")
    assert result["calls"] == 1 and result["status"] == "numerical"
    assert result["preparations"] == (route == "preparation-skipped")
    assert not result["final_ready"] and result["warm_replacements"] == 0
    assert result["warm_ready"] == (route in ("warm", "both"))


@pytest.mark.parametrize(
    "info,status",
    [
        (-7, "invalid"),
        (-1, "invalid"),
        (0, "returned"),
        (1, "numerical"),
        (7, "numerical"),
    ],
)
def test_seed_eigen_info_sign_preserves_solver_failure_classification(
    cuda_control_probe: Path, info: int, status: str
) -> None:
    assert probe(cuda_control_probe, "eigen-info", str(info))["status"] == status


@pytest.mark.parametrize("route", ["warm", "explicit", "both"])
@pytest.mark.parametrize(
    "scenario", ["nonconverged", "physical", "numerical", "invalid"]
)
def test_warm_and_restored_seed_failures_share_one_unprepared_cold_retry(
    cuda_control_probe: Path, route: str, scenario: str
) -> None:
    result = probe(cuda_control_probe, route, scenario)
    assert result["status"] == "returned" and result["converged"]
    assert result["calls"] == 2 and result["preparations"] == 0
    assert (result["kind"], result["outcome"], result["attempts"]) == (3, 1, 2)
    assert result["trace"][0]["seed"] == (0 if route == "warm" else 3)
    assert result["trace"][0]["warm"] == (route == "warm")
    assert result["trace"][1]["seed"] == result["trace"][1]["reuse"] == 0
    assert all(not attempt["native_cold"] for attempt in result["trace"])
    assert not result["retained_history"] and not result["stale_state"]
    assert result["complete"] == (scenario == "nonconverged")
    assert result["warm_replacements"] == 1 and result["final_ready"]


@pytest.mark.parametrize("stage", ["enqueue", "finish"])
@pytest.mark.parametrize("error", ["numerical", "runtime", "device", "allocation"])
def test_iteration_failures_preserve_retry_classification(
    cuda_control_probe: Path, stage: str, error: str
) -> None:
    result = probe(cuda_control_probe, scenario=f"{stage}-{error}")
    retryable = error == "numerical"
    assert result["calls"] == (2 if retryable else 1)
    assert result["status"] == ("returned" if retryable else error)
    assert result["final_ready"] == retryable
    assert not result["stale_state"] and not result["retained_history"]
    if retryable:
        assert result["attempts"] == 2 and not result["complete"]
        assert not result["jk_complete"]
        assert result["jk_j"] is result["jk_k"] is None


@pytest.mark.parametrize(
    "scenario,status",
    [
        ("runtime", "runtime"),
        ("device", "device"),
        ("allocation", "allocation"),
        ("overflow", "overflow"),
        ("logic", "logic"),
        ("preparation-runtime", "runtime"),
        ("both-physical", "numerical"),
    ],
)
def test_failed_replay_revokes_final_state_and_next_public_call_recovers(
    cuda_control_probe: Path, scenario: str, status: str
) -> None:
    result = probe(cuda_control_probe, "recover", scenario)
    assert result["initial_status"] == status and result["revoked_after_first"]
    assert result["status"] == "returned" and result["converged"]
    assert result["final_ready"] and result["warm_ready"]
    assert not result["stale_state"] and not result["retained_history"]
    assert result["preparations"] == 2 and result["warm_replacements"] == 1
    assert result["jk_complete"]
    first_cuda_attempt = scenario == "preparation-runtime"
    assert result["jk_j"] == ((1 << 33) + 11 if first_cuda_attempt else 5)
    assert result["jk_k"] == ((1 << 34) + 22 if first_cuda_attempt else 10)


@pytest.mark.parametrize("scenario", ["census-unknown", "census-overflow"])
def test_unobserved_or_overflowed_attempt_cannot_be_repaired_by_retry(
    cuda_control_probe: Path, scenario: str
) -> None:
    result = probe(cuda_control_probe, scenario=scenario)
    assert result["status"] == "returned" and result["calls"] == 2
    assert not result["jk_complete"]
    assert result["jk_j"] is result["jk_k"] is None


def test_certified_executed_zero_is_distinct_from_unavailable(
    cuda_control_probe: Path,
) -> None:
    result = probe(cuda_control_probe, scenario="census-zero")
    assert result["jk_complete"] and result["jk_j"] == result["jk_k"] == 0


@pytest.mark.parametrize("stage", ["enqueue", "finish"])
def test_failed_attempt_does_not_poison_a_neighbor_census(
    cuda_control_probe: Path, stage: str
) -> None:
    result = probe(cuda_control_probe, "neighbor", f"{stage}-numerical")
    assert not result["failed_complete"]
    assert result["neighbor_complete"]
    assert result["neighbor_j"] == (1 << 33) + 11
    assert result["neighbor_k"] == (1 << 34) + 22
