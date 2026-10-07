"""Host control-flow contracts, NOT native SCF/numerical qualification.

The tiny C++ harness compiles the actual HF and DFT retry control statements
extracted from their production owners. Scientific providers and publication
are explicit doubles. This catches retry/option/counter drift without linking
an old library or claiming that a real density/energy was evaluated.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def between(text: str, begin: str, end: str) -> str:
    """Fail closed when a production seam moves instead of testing stale code."""
    start = text.index(begin)
    return text[start : text.index(end, start)]


def control_sources() -> tuple[str, str, str]:
    hf = (ROOT / "src/scf/fleet.cpp").read_text()
    dft = (ROOT / "src/methods/dft_method.cpp").read_text()
    hf_retry = between(
        hf,
        "    const bool has_warm_density = warm_starts_enabled_",
        "    if (item.status == GENERATIVEQC_STATUS_SUCCESS && warm_starts_enabled_",
    )
    dft_retry = (
        between(
            dft,
            "    for (unsigned attempt = 0; attempt < 2; ++attempt)",
            "#if GENERATIVEQC_HAS_CUDA\n      while",
        )
        + "\n}\n"
    )
    dft_options = between(
        dft,
        "    const auto* seed =\n        initial_density ? initial_density",
        "    scf::ScfResult native;",
    )
    return hf_retry, dft_retry, dft_options


def test_retry_seams_keep_preparation_opt_in_and_raw_seed_admission() -> None:
    hf, dft, options = control_sources()
    assert "if (warm_retry) attempt_options.preliminary_guess.reset();" in hf
    assert "attempt_options," in hf
    assert "warm_enabled_ && warm_updates_, !attempt)" in dft
    assert (
        "if (!allow_preliminary) execution_options.preliminary_guess.reset();"
        in options
    )
    source = (ROOT / "src/scf/preliminary_guess.cpp").read_text()
    preparation = between(
        source,
        "std::optional<std::vector<double>> prepare_impl",
        "\n}\n}  // namespace",
    )
    assert (
        "admit_preliminary_density(target, std::move(preliminary.density))"
        in preparation
    )
    admission = between(
        source,
        "std::vector<double> admit_preliminary_density",
        "\nvoid validate_preliminary_options",
    )
    assert "solver::validate_seed(target.one_electron().overlap, density" in admission
    assert "normalized_warm_density" not in admission
    assert "admissible_minao_density(system, ints, x, projected.density)" in preparation


HARNESS = r"""
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.h"
#include "scf/initial_guess/preliminary_types.hpp"
namespace scf = generativeqc::scf;
namespace initial_guess = scf::initial_guess;
namespace core { struct System {}; }
namespace runtime {
struct CpuRetainedCapacity { explicit CpuRetainedCapacity(std::size_t) {} };
template <class T> std::size_t vector_bytes(const std::vector<T>& values) {
  return values.capacity() * sizeof(T);
}
}
enum class FockBackend { Cpu, Cuda };
struct Strategy { FockBackend backend{FockBackend::Cpu}; };
bool policy_enabled = true;
struct Options {
  std::optional<initial_guess::PreliminaryOptions> preliminary_guess{std::in_place};
  std::optional<Strategy> resolved_fock_build{std::in_place};
  Options() { if (!policy_enabled) preliminary_guess.reset(); }
};
struct Native {
  bool converged{};
  unsigned iterations{};
  std::size_t fock_builds{};
  initial_guess::PreliminaryDiagnostic preliminary_guess;
};
struct Warm { std::vector<double> density{1}; };
struct Item {
  Native scf;
  bool warm_start_used{}, warm_start_fallback{};
  generativeqc_backend executed_backend{GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc_status status{GENERATIVEQC_STATUS_SUCCESS};
};
std::string scenario;
unsigned calls{}, preparations{};
std::vector<bool> policies, seeds;
generativeqc_status exception_status() {
  try { throw; }
  catch (const std::bad_alloc&) { return GENERATIVEQC_STATUS_OUT_OF_MEMORY; }
  catch (...) { return GENERATIVEQC_STATUS_NUMERICAL_FAILURE; }
}
auto item_exception_status() { return exception_status(); }
Native solve(const Options& options, const std::vector<double>* seed) {
  ++calls;
  policies.push_back(options.preliminary_guess.has_value());
  seeds.push_back(seed != nullptr);
  if (options.preliminary_guess && !seed) ++preparations;
  if (calls == 1 && scenario == "runtime_error") throw std::runtime_error("warm failure");
  if (calls == 2 && scenario == "core_runtime_error") throw std::runtime_error("core failure");
  if ((calls == 1 && scenario == "bad_alloc") ||
      (calls == 2 && scenario == "core_bad_alloc")) throw std::bad_alloc();
  Native result;
  result.converged = calls > 1 || scenario == "success";
  result.iterations = calls == 1 ? 7 : 3;
  result.fock_builds = calls == 1 ? (std::uint64_t{1} << 33) + 7 : 5;
  if (options.preliminary_guess) {
    result.preliminary_guess.requested_kind = 1;
    result.preliminary_guess.outcome = initial_guess::PreliminaryOutcome::ExplicitDensity;
    result.preliminary_guess.target_attempts = 1;
  }
  return result;
}
core::System auxiliary_for_geometry(int, const core::System&) { return {}; }
Native run_fock_strategy_cached(int, const core::System&, const core::System*,
                               const Options& options, int,
                               const std::vector<double>* seed, int*) {
  return solve(options, seed);
}
Item run_hf() {
  Item item;
  Options execution_options;
  bool warm_starts_enabled_ = true;
  std::vector<std::optional<Warm>> warm_densities_{Warm{}};
  std::vector<int> independent_fock_plans_{0}, cuda_df_orthogonalizers_;
  std::size_t system_index = 0;
  int auxiliary_template_ = 0, device_id_ = -1;
  core::System execution_system;
  // HF_RETRY
  return item;
}
struct StubKsPlan {
  Options options_;
  std::vector<double> warm_{1};
  Native run(const std::vector<double>* initial_density, bool reuse_warm,
             bool update_warm, bool allow_preliminary = true) {
    // DFT_OPTIONS
    return solve(execution_options, seed);
  }
};
struct Calculation {
  struct { unsigned iterations{}; } convergence;
  std::size_t fock_builds{};
  std::optional<int> ks_diagnostic;
};
struct BatchResult : Item { Calculation calculation; };
struct KsItem {
  std::unique_ptr<StubKsPlan> plan{std::make_unique<StubKsPlan>()};
  std::optional<Warm> warm{Warm{}};
  bool resident_warm{};
};
Item run_dft(bool resident) {
  Options options_;
  bool warm_enabled_ = true, warm_updates_ = true;
  std::vector<KsItem> items_(1);
  items_[0].resident_warm = resident;
  std::vector<BatchResult> results(1);
  results[0].warm_start_used = true;
  std::vector<bool> ready(1, true);
  std::vector<std::optional<std::vector<double>>> preliminary_seeds(1);
  std::vector<initial_guess::PreliminaryDiagnostic> preliminary_diagnostics(1);
  std::vector<bool> preliminary_requested(1, false);
  auto size = [] { return std::size_t{1}; };
  auto host_numeric_capacity = [](std::size_t) { return std::size_t{}; };
  const auto finish = [&](std::size_t i, Native native) {
    auto& result = results[i];
    result.calculation.convergence.iterations = native.iterations;
    result.calculation.fock_builds = native.fock_builds;
    result.status = native.converged ? GENERATIVEQC_STATUS_SUCCESS : GENERATIVEQC_STATUS_NOT_CONVERGED;
    result.scf = std::move(native);
  };
  // DFT_RETRY
  return results[0];
}
int main(int argc, char** argv) {
  const std::string route = argv[1];
  if (route == "layout") {
    using D = generativeqc_initial_guess_diagnostic;
    std::cout << "{\"size\":" << sizeof(D)
              << ",\"preliminary_fock_builds\":" << offsetof(D, preliminary_fock_builds)
              << ",\"discarded_target_fock_builds\":" << offsetof(D, discarded_target_fock_builds)
              << ",\"work_counters_complete\":" << offsetof(D, work_counters_complete) << "}";
    return 0;
  }
  scenario = argv[2];
  policy_enabled = argc < 4 || std::string(argv[3]) != "disabled";
  const auto item = route == "hf" ? run_hf() : run_dft(route == "dft-resident");
  const auto& d = item.scf.preliminary_guess;
  std::cout << "{\"calls\":" << calls << ",\"preparations\":" << preparations
            << ",\"status\":" << item.status << ",\"oom\":" << GENERATIVEQC_STATUS_OUT_OF_MEMORY
            << ",\"success\":" << GENERATIVEQC_STATUS_SUCCESS
            << ",\"numerical_failure\":" << GENERATIVEQC_STATUS_NUMERICAL_FAILURE
            << ",\"fallback\":" << item.warm_start_fallback
            << ",\"kind\":" << d.requested_kind
            << ",\"outcome\":" << static_cast<unsigned>(d.outcome)
            << ",\"attempts\":" << d.target_attempts
            << ",\"discarded_iterations\":" << d.discarded_target_iterations
            << ",\"discarded_focks\":" << d.discarded_target_fock_builds
            << ",\"complete\":" << d.work_counters_complete
            << ",\"first_policy\":" << policies.front()
            << ",\"last_policy\":" << policies.back()
            << ",\"first_seed\":" << seeds.front()
            << ",\"last_seed\":" << seeds.back() << "}";
}
"""


@pytest.fixture(scope="module")
def host_control_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ccache = shutil.which(os.environ.get("CCACHE", "ccache"))
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if ccache is None or compiler is None:
        pytest.skip("source-control harness requires ccache and a C++20 compiler")
    subprocess.run([ccache, "--version"], check=True, capture_output=True)
    hf, dft, options = control_sources()
    source = HARNESS.replace("// HF_RETRY", hf).replace("// DFT_RETRY", dft)
    source = source.replace("// DFT_OPTIONS", options)
    directory = tmp_path_factory.mktemp("preliminary-control")
    path = directory / "control.cpp"
    object_path = directory / "control.o"
    binary = directory / "control"
    path.write_text(source)
    subprocess.run(
        [
            ccache,
            compiler,
            "-std=c++20",
            "-O0",
            "-DGENERATIVEQC_HAS_CUDA=0",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
            "-c",
            str(path),
            "-o",
            str(object_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [compiler, str(object_path), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def probe(
    binary: Path, route: str, scenario: str = "success", *, policy: bool = True
) -> dict:
    completed = subprocess.run(
        [str(binary), route, scenario, "enabled" if policy else "disabled"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("route", ["hf", "dft-imported", "dft-resident"])
@pytest.mark.parametrize("scenario", ["success", "nonconverged", "runtime_error"])
def test_actual_warm_retry_controls_use_only_core_and_count_work(
    host_control_harness: Path, route: str, scenario: str
) -> None:
    result = probe(host_control_harness, route, scenario)
    retry = scenario != "success"
    assert result["calls"] == 1 + retry
    assert result["preparations"] == 0
    assert result["first_policy"] and result["first_seed"]
    assert result["last_policy"] == (not retry)
    assert result["last_seed"] == (not retry)
    assert result["status"] == result["success"]
    assert result["fallback"] == retry
    assert result["kind"] == 1 and result["outcome"] == 1
    assert result["attempts"] == 1 + retry
    assert result["complete"] == (scenario != "runtime_error")
    assert result["discarded_iterations"] == (7 if scenario == "nonconverged" else 0)
    assert result["discarded_focks"] == (
        (1 << 33) + 7 if scenario == "nonconverged" else 0
    )


@pytest.mark.parametrize("route", ["hf", "dft-imported", "dft-resident"])
@pytest.mark.parametrize("scenario,calls", [("bad_alloc", 1), ("core_bad_alloc", 2)])
def test_actual_warm_retry_controls_never_retry_allocation_failure(
    host_control_harness: Path, route: str, scenario: str, calls: int
) -> None:
    result = probe(host_control_harness, route, scenario)
    assert result["calls"] == calls
    assert result["preparations"] == 0
    assert result["status"] == result["oom"]


@pytest.mark.parametrize("route", ["hf", "dft-imported", "dft-resident"])
def test_failed_core_retry_is_terminal_for_active_preliminary_policy(
    host_control_harness: Path, route: str
) -> None:
    result = probe(host_control_harness, route, "core_runtime_error")
    assert result["calls"] == 2
    assert result["preparations"] == 0
    assert result["status"] == result["numerical_failure"]
    assert result["first_policy"] and not result["last_policy"]


@pytest.mark.parametrize(
    "scenario,calls",
    [("bad_alloc", 2), ("core_bad_alloc", 3), ("core_runtime_error", 3)],
)
def test_disabled_policy_preserves_existing_hf_outer_retry_behavior(
    host_control_harness: Path, scenario: str, calls: int
) -> None:
    # This records compatibility with the pre-existing HF catch-all lifecycle;
    # it is not an endorsement of retrying allocation errors in new policies.
    result = probe(host_control_harness, "hf", scenario, policy=False)
    assert result["calls"] == calls
    assert result["status"] == result["success"]
    assert not result["first_policy"] and not result["last_policy"]
    assert result["preparations"] == 0 and result["kind"] == 0


def test_widened_diagnostic_ctypes_layout_matches_public_c_header(
    host_control_harness: Path,
) -> None:
    from generativeqc import _native

    native = probe(host_control_harness, "layout")
    descriptor = _native.InitialGuessDiagnosticDescriptor
    assert native.pop("size") == ctypes.sizeof(descriptor)
    for name, offset in native.items():
        assert getattr(descriptor, name).offset == offset
