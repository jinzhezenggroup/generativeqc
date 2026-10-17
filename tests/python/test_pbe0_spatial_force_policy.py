"""Execute the actual method-owned default predicate without native/GPU imports."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_direct_force_schedule_composition import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_method_authenticates_full_range_spatial_force_default(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Check every composition, approximation, size, spin and precision guard."""
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    predicate = _definition(source, "  bool qualified_spatial_force_default() const")
    probe = tmp_path / "policy.cpp"
    probe.write_text(PREFIX + predicate + "\n};\n" + DRIVER)
    executable = native_cxx.build_executable(
        [probe],
        tmp_path / "policy",
        compile_args=["-std=c++20", "-O2", f"-I{ROOT / 'src'}"],
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_only_authenticated_dft_consumers_supply_default_hint() -> None:
    """Both source and combined force callbacks preserve the resident token check."""
    source = (ROOT / "src/methods/dft_method.cpp").read_text()
    for marker in (
        "  generativeqc_status cuda_full_range_integral_derivatives(",
        "  generativeqc_status cuda_integral_gradient(",
    ):
        body = _definition(source, marker)
        assert body.index("resident_final_density(expected") < body.index(
            "qualified_spatial_force_default()"
        )
        assert (
            "execute_prepared_cuda_direct_shell_full_range_derivatives_device(" in body
        )
    for path in (
        "src/scf/cuda/direct_coulomb.hpp",
        "src/scf/cuda_direct_jk.hpp",
        "src/scf/cuda_fock_execution.hpp",
    ):
        assert "bool allow_spatial_default = false" in (ROOT / path).read_text()
    for path in (
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda_fock_execution.cpp",
    ):
        assert "separate_sources, allow_spatial_default" in (ROOT / path).read_text()
    native = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    rsh = _definition(native, "cudaError_t execute_generated_rsh_energy_derivatives(")
    assert "allow_spatial_default" not in rsh


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>
#include "dft/cuda_ks_final_validation_policy.hpp"
#include "dft/cuda_spd_force_policy.hpp"
constexpr int GENERATIVEQC_DENSITY_FITTING_NONE = 0;
constexpr int GENERATIVEQC_PRECISION_FP64 = 0;
namespace generativeqc::dft { enum class SemilocalFamily { Pbe, Other }; }
namespace generativeqc::scf {
enum class FockBackend { Cuda, Cpu };
enum class FockApproximation { Exact, DensityFitted };
enum class FockOperator { FullRange, LongRange };
struct Term {
  bool present{true};
  double coefficient{1.0};
  FockApproximation approximation{FockApproximation::Exact};
  FockOperator op{FockOperator::FullRange};
};
struct Spec { unsigned derivative_order{}; Term coulomb{}, exchange{true, -0.125}; };
struct Strategy { FockBackend backend{FockBackend::Cuda}; Spec spec{}; };
struct Diagnostic { std::size_t nbf{384}; };
struct Plan {
  Strategy strategy_value{};
  Diagnostic diagnostic_value{};
  const Strategy& strategy() const { return strategy_value; }
  const Diagnostic& diagnostic() const { return diagnostic_value; }
};
}
using namespace generativeqc;
struct Method {
  scf::Plan fock_;
  struct {
    dft::SemilocalFamily semilocal_family{dft::SemilocalFamily::Pbe};
    unsigned spin_channels{1};
    bool range_exchange{}, nonlocal_correlation{}, d4_correction{};
  } execution_plan_;
  struct {
    double semilocal_exchange_scale{0.75}, semilocal_correlation_scale{1.0};
    int density_fitting_mode{};
    std::optional<int> precision_mode{GENERATIVEQC_PRECISION_FP64};
  } options_;
  bool range_strategy_{};
  bool native_stationary_admitted{};
  unsigned gradient_qualification{};
  const Method* admitted_cuda_stationary_binding() const {
    return native_stationary_admitted ? this : nullptr;
  }
  unsigned application_gradient_qualification() const { return gradient_qualification; }
  struct Atom { int ecp_core{}; };
  struct { std::vector<int> ecp_terms; std::vector<Atom> atoms{Atom{}}; } system_;
"""

DRIVER = r"""
template<class Mutation> void reject_hybrid(Mutation mutate) {
  for (bool native_fp64 : {false, true}) {
    Method method;
    method.options_.precision_mode = native_fp64 ? 1 : GENERATIVEQC_PRECISION_FP64;
    method.native_stationary_admitted = native_fp64;
    mutate(method);
    assert(!method.qualified_spatial_force_default());
  }
}
template<class Mutation> void reject(Mutation mutate) {
  for (bool native_fp64 : {false, true})
    for (unsigned source : {0U, 1U, 2U}) {
      Method method;
      method.options_.precision_mode = native_fp64 ? 1 : GENERATIVEQC_PRECISION_FP64;
      method.native_stationary_admitted = native_fp64;
      method.fock_.strategy_value.spec.exchange.present = source == 0;
      method.execution_plan_.d4_correction = source == 2;
      mutate(method);
      assert(!method.qualified_spatial_force_default());
    }
}
int main() {
  Method method;
  assert(method.qualified_spatial_force_default());
  for (bool exchange_present : {false, true})
    for (std::size_t aos : {0U, 383U, 384U, 385U, 768U, 1536U}) {
      method.fock_.strategy_value.spec.exchange.present = exchange_present;
      method.fock_.diagnostic_value.nbf = aos;
      assert(method.qualified_spatial_force_default() == (aos >= 384));
    }
  reject([](auto& value) { value.execution_plan_.spin_channels = 0; });
  reject([](auto& value) { value.execution_plan_.spin_channels = 2; });
  // Eligible B3LYP has 20% exact K and no PBE label; the method catalog
  // has already authenticated its full semilocal/exact exchange composition.
  Method b3lyp;
  b3lyp.execution_plan_.semilocal_family = dft::SemilocalFamily::Other;
  b3lyp.fock_.strategy_value.spec.exchange.coefficient = -0.1;
  b3lyp.options_.semilocal_exchange_scale = 1.0;
  assert(b3lyp.qualified_spatial_force_default());
  reject([](auto& value) { value.fock_.strategy_value.backend = scf::FockBackend::Cpu; });
  reject([](auto& value) { value.fock_.strategy_value.spec.derivative_order = 1; });
  reject([](auto& value) { value.fock_.strategy_value.spec.coulomb.present = false; });
  reject([](auto& value) { value.fock_.strategy_value.spec.coulomb.coefficient = 0.5; });
  // Pure J remains valid when an independent D4/gCP owner adds its force.
  Method pure_j;
  pure_j.fock_.strategy_value.spec.exchange.present = false;
  assert(pure_j.qualified_spatial_force_default());
  pure_j.execution_plan_.d4_correction = true;
  assert(pure_j.qualified_spatial_force_default());
  pure_j.execution_plan_.nonlocal_correlation = true;
  assert(!pure_j.qualified_spatial_force_default());
  for (double coefficient : {0.0, -0.0, 0.25, -0.500001, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()})
    reject_hybrid([coefficient](auto& value) { value.fock_.strategy_value.spec.exchange.coefficient = coefficient; });
  for (double coefficient : {-0.5, -1e-300}) {
    Method live_exchange;
    live_exchange.fock_.strategy_value.spec.exchange.coefficient = coefficient;
    assert(live_exchange.qualified_spatial_force_default());
  }
  // Semilocal XC composition is orthogonal to the exact Direct J/K recurrence.
  Method scaled;
  scaled.options_.semilocal_exchange_scale = 1.0;
  scaled.options_.semilocal_correlation_scale = 0.5;
  assert(scaled.qualified_spatial_force_default());
  reject([](auto& value) { value.fock_.strategy_value.spec.coulomb.approximation = scf::FockApproximation::DensityFitted; });
  reject_hybrid([](auto& value) { value.fock_.strategy_value.spec.exchange.approximation = scf::FockApproximation::DensityFitted; });
  reject([](auto& value) { value.fock_.strategy_value.spec.coulomb.op = scf::FockOperator::LongRange; });
  reject_hybrid([](auto& value) { value.fock_.strategy_value.spec.exchange.op = scf::FockOperator::LongRange; });
  reject([](auto& value) { value.options_.density_fitting_mode = 1; });
  // Budget SCF may reuse the FP64 force source only through an authenticated
  // native binding; independently qualified mixed derivatives are excluded.
  for (unsigned source : {0U, 1U, 2U}) {
    for (bool admitted : {false, true})
      for (unsigned gradient : {0U, 1U}) {
        Method precision;
        precision.fock_.strategy_value.spec.exchange.present = source == 0;
        precision.execution_plan_.d4_correction = source == 2;
        precision.options_.precision_mode = 1;
        precision.native_stationary_admitted = admitted;
        precision.gradient_qualification = gradient;
        assert(precision.qualified_spatial_force_default() == (admitted && gradient == 0));
        precision.fock_.diagnostic_value.nbf = 383;
        assert(!precision.qualified_spatial_force_default());
      }
    Method unresolved;
    unresolved.fock_.strategy_value.spec.exchange.present = source == 0;
    unresolved.execution_plan_.d4_correction = source == 2;
    unresolved.options_.precision_mode.reset();
    assert(!unresolved.qualified_spatial_force_default());
  }
  reject([](auto& value) { value.execution_plan_.range_exchange = true; });
  reject([](auto& value) { value.range_strategy_ = true; });
  reject([](auto& value) { value.execution_plan_.nonlocal_correlation = true; });
  reject_hybrid([](auto& value) { value.execution_plan_.d4_correction = true; });
  reject([](auto& value) { value.system_.ecp_terms.push_back(1); });
  reject([](auto& value) { value.system_.atoms[0].ecp_core = 2; });
}
"""
