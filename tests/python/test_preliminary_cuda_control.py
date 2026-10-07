"""Run the real single-owner CUDA seed/retry branch with host provider doubles.

This qualifies control flow, precedence, counters and revocation only. It does
not qualify CUDA kernels, MINAO density numerics or physical SCF convergence.
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
    """Fail closed if a production seam moves; never keep a copied branch."""
    start = text.index(begin)
    return text[start : text.index(end, start)]


def cuda_control_source() -> str:
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    return (
        between(
            source,
            "  scf::ScfResult run(const std::vector<double>* initial_density",
            "\n#endif",
        )
        + '\n#endif\n    throw std::logic_error("missing CUDA owner");\n  }\n'
    )


HARNESS = r"""
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.hpp"
#include "scf/initial_guess/preliminary_types.hpp"
namespace scf = generativeqc::scf;
namespace initial_guess = scf::initial_guess;
struct MethodError : generativeqc::Error { using Error::Error; };
namespace generativeqc::scf {
struct ScfResult {
  bool converged{};
  unsigned iterations{};
  std::uint64_t fock_builds{};
  initial_guess::PreliminaryDiagnostic preliminary_guess;
  // Model ownership of the returned attempt's exported history.
  std::shared_ptr<int> history;
};
}
// CUDA_STATUS_CHECK
void check_seed_eigen_info(int info) {
  // CUDA_SEED_EIGEN_CHECK
}
std::string scenario;
std::weak_ptr<int> discarded_history;
struct Call { unsigned seed; bool reuse, warm, updates, export_density; };
struct CudaKsPlan {
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
  unsigned warm_replacements{};
  bool has_warm_start() const noexcept;
  bool failed() const noexcept;
  void set_warm_start_updates(bool) noexcept;
  void clear_warm_start() noexcept;
  void invalidate_final_state() noexcept;
  scf::ScfResult run(const std::vector<double>*, bool, bool);
  bool active() const { return impl_->is_active; }
  void begin(const std::vector<double>* seed, bool reuse) {
    stale_state_at_begin |= impl_->final_state_ready;
    invalidate_final_state();
    if (!calls.empty()) history_retained_at_retry |= !discarded_history.expired();
    calls.push_back({seed ? static_cast<unsigned>(seed->at(0)) : 0U,
                     reuse, !seed && reuse && impl_->warm_ready, impl_->warm_updates, false});
    impl_->is_failed = true;
    impl_->is_active = false;
    const bool second = scenario.starts_with("second-");
    const std::string error = second ? scenario.substr(7) : scenario;
    if (error == "both-numerical")
      check(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "numerical failure");
    if (calls.size() == (second ? 2U : 1U)) {
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
    impl_->is_failed = false;
    impl_->is_active = true;
  }
  void enqueue_iteration() {}
  void finish_iteration() { impl_->is_active = false; }
  scf::ScfResult result(bool export_density) {
    calls.back().export_density = export_density;
    const bool first = calls.size() == 1;
    const bool nonconverged = scenario == "both-nonconverged" ||
        (first && (scenario == "nonconverged" || scenario.starts_with("second-")));
    impl_->is_failed = scenario == "both-physical" ||
        (first && (scenario == "physical" || scenario == "failed-converged"));
    scf::ScfResult value;
    value.converged = !nonconverged && (!impl_->is_failed || scenario == "failed-converged");
    value.iterations = first ? 7U : 3U;
    value.fock_builds = first ? (std::uint64_t{1} << 33) + 7U : 5U;
    value.history = std::make_shared<int>(1);
    if (first) discarded_history = value.history;
    if (value.converged && !impl_->is_failed) {
      impl_->final_state_ready = true;
      if (impl_->warm_updates) { impl_->warm_ready = true; ++warm_replacements; }
    }
    return value;
  }
};
// CUDA_METHODS
struct Owner {
  std::unique_ptr<CudaKsPlan> cuda_ = std::make_unique<CudaKsPlan>();
  struct {
    std::optional<initial_guess::PreliminaryOptions> preliminary_guess{std::in_place};
  } options_;
  unsigned preparations{};
  bool preparation_skipped{};
  Owner() { options_.preliminary_guess->kind = initial_guess::PreliminaryKind::Minao; }
  void invalidate_final_state() { cuda_->invalidate_final_state(); }
  std::optional<std::vector<double>> prepare_cold_initial_guess(
      initial_guess::PreliminaryDiagnostic& diagnostic) {
    ++preparations;
    diagnostic.outcome = preparation_skipped ? initial_guess::PreliminaryOutcome::BudgetSkipped
                                            : initial_guess::PreliminaryOutcome::Used;
    diagnostic.preparation_numeric_capacity = 123;
    diagnostic.preparation_seconds = 0.25;
    if (preparation_skipped) return std::nullopt;
    return std::vector<double>{2};
  }
  // OWNER_RUN
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
  Owner owner;
  const bool resident = route == "warm" || route == "both" || route == "no-reuse";
  owner.cuda_->impl_->warm_ready = resident;
  // Every call must revoke the prior final result even if admission throws.
  owner.cuda_->impl_->final_state_ready = true;
  const std::vector<double> explicit_seed{3};
  const bool explicit_input = route == "explicit" || route == "both";
  const bool reuse = route != "no-reuse";
  const bool update = route != "frozen";
  const bool allow = route != "allow-disabled";
  owner.preparation_skipped = route == "preparation-skipped";
  if (route == "disabled") owner.options_.preliminary_guess.reset();
  scf::ScfResult result;
  std::string status = "returned";
  try {
    result = owner.run(explicit_input ? &explicit_seed : nullptr, reuse, update, allow);
  } catch (const generativeqc::Error& error) {
    status = error.status() == GENERATIVEQC_STATUS_NUMERICAL_FAILURE ? "numerical" : "device";
  } catch (const std::bad_alloc&) { status = "allocation";
  } catch (const std::invalid_argument&) { status = "invalid";
  } catch (const std::overflow_error&) { status = "overflow";
  } catch (const std::logic_error&) { status = "logic";
  } catch (const std::runtime_error&) { status = "runtime"; }
  const auto& d = result.preliminary_guess;
  const auto& cuda = *owner.cuda_;
  std::cout << "{\"status\":\"" << status << "\",\"calls\":" << cuda.calls.size()
            << ",\"preparations\":" << owner.preparations
            << ",\"converged\":" << result.converged
            << ",\"kind\":" << d.requested_kind
            << ",\"outcome\":" << static_cast<unsigned>(d.outcome)
            << ",\"attempts\":" << d.target_attempts
            << ",\"discarded_iterations\":" << d.discarded_target_iterations
            << ",\"discarded_focks\":" << d.discarded_target_fock_builds
            << ",\"complete\":" << d.work_counters_complete
            << ",\"capacity\":" << d.preparation_numeric_capacity
            << ",\"seconds\":" << d.preparation_seconds
            << ",\"returned_iterations\":" << result.iterations
            << ",\"returned_focks\":" << result.fock_builds
            << ",\"warm_ready\":" << cuda.has_warm_start()
            << ",\"warm_replacements\":" << cuda.warm_replacements
            << ",\"final_ready\":" << cuda.impl_->final_state_ready
            << ",\"stale_state\":" << cuda.stale_state_at_begin
            << ",\"retained_history\":" << cuda.history_retained_at_retry
            << ",\"trace\":[";
  for (unsigned i = 0; i < cuda.calls.size(); ++i) {
    const auto& call = cuda.calls[i];
    if (i) std::cout << ",";
    std::cout << "{\"seed\":" << call.seed << ",\"reuse\":" << call.reuse
              << ",\"warm\":" << call.warm << ",\"updates\":" << call.updates
              << ",\"export\":" << call.export_density << "}";
  }
  std::cout << "]}";
}
"""


@pytest.fixture(scope="module")
def cuda_control_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    cuda = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    methods = "\n".join(
        [
            between(
                cuda,
                "bool CudaKsPlan::failed()",
                "\nvoid CudaKsPlan::enqueue_iteration",
            ),
            between(
                cuda,
                "scf::ScfResult CudaKsPlan::run(",
                "\nstd::vector<double> CudaKsPlan::warm_density",
            ),
            between(
                cuda,
                "void CudaKsPlan::set_warm_start_updates",
                "\ngenerativeqc_status CudaKsPlan::final_state_token",
            ),
        ]
    )
    source = HARNESS.replace("// CUDA_METHODS", methods).replace(
        "// OWNER_RUN", cuda_control_source()
    )
    source = source.replace(
        "// CUDA_STATUS_CHECK",
        between(cuda, "void check(generativeqc_status", "\ntemplate <class Function>"),
    ).replace(
        "// CUDA_SEED_EIGEN_CHECK",
        between(
            cuda, "    if (info < 0)", "    // The solver emits column-major orbitals"
        ),
    )
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


def test_public_execute_arguments_prepare_minao_for_a_fresh_cuda_owner(
    cuda_control_probe: Path,
) -> None:
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    assert "auto result = adapt_result(run(nullptr, true, true), backend_);" in source
    result = probe(cuda_control_probe)
    assert result["calls"] == result["preparations"] == 1
    assert result["trace"] == [
        {"seed": 2, "reuse": 0, "warm": 0, "updates": 1, "export": 0}
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
    assert (result["kind"], result["outcome"], result["attempts"]) == (3, 1, 1)


@pytest.mark.parametrize("route", ["no-reuse", "frozen"])
def test_reuse_and_publication_permissions_do_not_skip_cold_preparation(
    cuda_control_probe: Path, route: str
) -> None:
    result = probe(cuda_control_probe, route)
    assert result["preparations"] == 1 and result["trace"][0]["seed"] == 2
    assert result["warm_replacements"] == (route != "frozen")
    assert result["warm_ready"] == (route != "frozen")


@pytest.mark.parametrize("scenario", ["nonconverged", "physical", "failed-converged"])
def test_prepared_cuda_failure_gets_one_fresh_hcore_attempt(
    cuda_control_probe: Path, scenario: str
) -> None:
    result = probe(cuda_control_probe, scenario=scenario)
    assert result["calls"] == 2 and result["preparations"] == 1
    assert result["trace"] == [
        {"seed": 2, "reuse": 0, "warm": 0, "updates": 1, "export": 0},
        {"seed": 0, "reuse": 0, "warm": 0, "updates": 1, "export": 0},
    ]
    assert result["status"] == "returned" and result["converged"]
    assert (result["kind"], result["outcome"], result["attempts"]) == (3, 5, 2)
    assert result["discarded_iterations"] == 7
    assert result["discarded_focks"] == (1 << 33) + 7
    assert result["returned_iterations"] == 3 and result["returned_focks"] == 5
    assert result["complete"] and result["final_ready"]
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
        ("invalid", "invalid"),
        ("seed-eigen-invalid", "invalid"),
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
    ["warm", "explicit", "both", "disabled", "allow-disabled", "preparation-skipped"],
)
def test_only_a_prepared_seed_can_activate_the_new_retry(
    cuda_control_probe: Path, route: str
) -> None:
    result = probe(cuda_control_probe, route, "physical")
    assert result["calls"] == 1 and result["status"] == "numerical"
    assert result["preparations"] == (route == "preparation-skipped")
    assert result["warm_ready"] == (route in ("warm", "both"))
    assert result["warm_replacements"] == 0 and not result["final_ready"]


@pytest.mark.parametrize("route", ["disabled", "allow-disabled", "preparation-skipped"])
def test_disabled_or_skipped_preparation_keeps_core_behavior(
    cuda_control_probe: Path, route: str
) -> None:
    result = probe(cuda_control_probe, route)
    assert result["calls"] == 1 and result["trace"][0]["seed"] == 0
    assert result["preparations"] == (route == "preparation-skipped")
    assert result["outcome"] == (4 if route == "preparation-skipped" else 0)
    assert result["attempts"] == (1 if route == "preparation-skipped" else 0)


@pytest.mark.parametrize("scenario", ["numerical", "seed-eigen"])
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
    "scenario", ["both-numerical", "second-numerical", "second-seed-eigen"]
)
def test_hcore_numerical_exception_is_terminal(
    cuda_control_probe: Path, scenario: str
) -> None:
    result = probe(cuda_control_probe, scenario=scenario)
    assert result["calls"] == 2 and result["preparations"] == 1
    assert result["status"] == "numerical"
    assert not result["final_ready"] and not result["warm_ready"]


@pytest.mark.parametrize(
    "route",
    ["warm", "explicit", "both", "disabled", "allow-disabled", "preparation-skipped"],
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
