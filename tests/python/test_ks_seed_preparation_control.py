"""Execute cold-seed adapter wiring with explicit host provider doubles.

This tests callback ownership, opt-in admission and failure classification. The
native initial-guess suite and CUDA seed tests own scientific qualification.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r"""
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.hpp"
#include "scf/initial_guess/eigen_operation.hpp"
#include "scf/initial_guess/preliminary_types.hpp"
namespace scf = generativeqc::scf;
namespace initial_guess = scf::initial_guess;
namespace runtime {
struct DeviceCapacityScope { explicit DeviceCapacityScope(int) {} };
struct CpuRetainedCapacity { explicit CpuRetainedCapacity(std::size_t) {} };
}
struct Options {
  std::optional<initial_guess::PreliminaryOptions> preliminary_guess{std::in_place};
};
struct Fock { int strategy() const { return 0; } };
std::string scenario;
bool prepared{}, validated{}, used_eigen{}, used_products{};
namespace generativeqc::scf::initial_guess {
void validate_preliminary_target(int, int, const Options&) { validated = true; }
std::optional<std::vector<double>> prepare_preliminary_density(
    const Fock&, const PreliminaryOptions&, PreliminaryDiagnostic& diagnostic,
    const EigenOperation& eigen, const MatrixProductOperation& products) {
  prepared = true;
  if (eigen) used_eigen = eigen({1}, nullptr, nullptr, 1).values == reference::Matrix{7};
  if (products) used_products = products({2}, {3}, 1, true, false) == reference::Matrix{11};
  if (scenario == "numerical")
    throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "numerical");
  if (scenario == "device")
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, "device");
  if (scenario == "allocation") throw std::bad_alloc();
  if (scenario == "runtime") throw std::runtime_error("runtime");
  diagnostic.outcome = PreliminaryOutcome::Used;
  return std::vector<double>{13};
}
}  // namespace generativeqc::scf::initial_guess
struct CudaOwner {
  std::shared_ptr<int> lifetime = std::make_shared<int>(1);
  initial_guess::EigenOperation seed_eigen_operation() const {
    return [lease = lifetime](const auto&, const auto*, const auto*, std::size_t) {
      return scf::reference::EigenResult{{7}, {1}};
    };
  }
  initial_guess::MatrixProductOperation seed_product_operation() const {
    return [lease = lifetime](const auto&, const auto&, std::size_t, bool, bool) {
      return scf::reference::Matrix{11};
    };
  }
};
struct Plan {
  Options options_;
  int capacity_{}, system_{};
  Fock fock_;
  std::unique_ptr<CudaOwner> cuda_;
  std::size_t host_numeric_capacity() const { return 0; }
  // PREPARATION
};
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  Plan plan;
  if (std::string(argv[1]) == "cuda") plan.cuda_ = std::make_unique<CudaOwner>();
  scenario = argv[2];
  if (scenario == "disabled") plan.options_.preliminary_guess.reset();
  std::weak_ptr<int> lifetime;
  if (plan.cuda_) lifetime = plan.cuda_->lifetime;
  initial_guess::PreliminaryDiagnostic diagnostic;
  std::optional<std::vector<double>> density;
  std::string error = "none";
  try { density = plan.prepare_cold_initial_guess(diagnostic); }
  catch (const std::bad_alloc&) { error = "allocation"; }
  catch (const generativeqc::Error& e) {
    error = e.status() == GENERATIVEQC_STATUS_CUDA_ERROR ? "device" : "unexpected";
  }
  plan.cuda_.reset();
  std::cout << "{\"prepared\":" << prepared << ",\"validated\":" << validated
            << ",\"eigen\":" << used_eigen << ",\"products\":" << used_products
            << ",\"released\":" << lifetime.expired()
            << ",\"density\":" << (density == std::optional(std::vector<double>{13}))
            << ",\"failed\":" << (diagnostic.outcome == initial_guess::PreliminaryOutcome::PreparationFailed)
            << ",\"complete\":" << diagnostic.work_counters_complete
            << ",\"attempts\":" << diagnostic.target_attempts
            << ",\"error\":\"" << error << "\"}\n";
}
"""


@pytest.fixture(scope="module")
def preparation_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Compile the actual adapter; never copy its control branches into a test."""
    owner = (ROOT / "src/methods/dft_method.cpp").read_text()
    production = cpp_function_definition(owner, "prepare_cold_initial_guess")
    directory = tmp_path_factory.mktemp("ks-seed-preparation")
    source = directory / "probe.cpp"
    source.write_text(HARNESS.replace("// PREPARATION", production))
    return native_cxx.build_executable(
        [source],
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


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "scenario", ["success", "disabled", "numerical", "device", "allocation", "runtime"]
)
def test_preparation_borrows_providers_and_preserves_failures(
    preparation_probe: Path, backend: str, scenario: str
) -> None:
    """Callbacks are synchronous, optional, and released on every exit path."""
    completed = subprocess.run(
        [str(preparation_probe), backend, scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    enabled = scenario != "disabled"
    assert result["prepared"] == result["validated"] == enabled
    assert result["eigen"] == result["products"] == (enabled and backend == "cuda")
    assert result["released"]
    assert result["attempts"] == int(enabled)
    assert result["density"] == (scenario == "success")
    assert result["error"] == (
        scenario if scenario in {"device", "allocation"} else "none"
    )
    declined = scenario in {"numerical", "runtime"}
    assert result["failed"] == declined
    if declined:
        assert not result["complete"]
