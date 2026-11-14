"""Execute native resource admission with independent capacity leaf stand-ins."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_fp64_resources_follow_selected_force_precision(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Budget presence cannot change strict bounds or admit uncounted mixed scratch.

    The production reservation and ledger execute unchanged. Leaf provider
    capacities are deterministic stand-ins: this is a lifecycle/admission test,
    not a numerical CUDA qualification or a measured device-memory peak.
    """
    production = (ROOT / "src/methods/dft_method.cpp").read_text()
    reservation = cpp_function_definition(production, "ks_device_capacity_reservation")
    source = tmp_path / "precision_resources.cpp"
    source.write_text(PREFIX + reservation + DRIVER)
    executable = required_native_cxx.build_executable(
        [source],
        tmp_path / "precision_resources",
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=15)


PREFIX = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <vector>
#include "core/types.hpp"
#include "dft/application_precision.hpp"
#include "dft/cuda_stationary_capacity.hpp"
#include "dft/ks_tile_policy.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/ks_device_resources.hpp"
#define GENERATIVEQC_HAS_CUDA 1
using namespace generativeqc;
namespace generativeqc::scf {
enum class FockApproximation { Exact, DensityFitted };
enum class FockOperator { FullRange, LongRange };
struct Term {
  bool present{true};
  FockApproximation approximation{FockApproximation::Exact};
  FockOperator op{FockOperator::FullRange};
  double omega{};
};
struct ResolvedFockBuild { struct { Term coulomb, exchange; } spec; };
struct ScfOptions {
  enum class XcExecutionSchedule { DeviceFused, HostUnfused };
  bool preliminary_guess{}, incremental_direct_jk{};
  bool nonlocal_grid_spec{};
  int precision_mode{GENERATIVEQC_PRECISION_FP64};
  XcExecutionSchedule xc_execution_schedule{XcExecutionSchedule::DeviceFused};
  std::optional<ResolvedFockBuild> resolved_fock_build{ResolvedFockBuild{}};
  std::array<double, 3> application_precision_budget{};
  double energy_tolerance{1e-10}, density_tolerance{1e-8}, screening_tolerance{1e-12};
  double semilocal_exchange_scale{1}, semilocal_correlation_scale{1};
  std::size_t xc_tile_points{256}, diis_history{8};
  std::size_t stationary_tile_points() const { return 256; }
};
struct Capacity { std::size_t retained_response_bytes{256}, setup_bytes{512}; };
template<class... Arguments> Capacity prepared_fock_device_capacity(Arguments&&...) { return {}; }
namespace cuda_execution {
struct MatrixLibraryOwner { static std::size_t provider_allowance(int) { return 512; } };
}
}
namespace generativeqc::molecule {
std::size_t ao_count(const core::System& system) { return system.shells.size(); }
std::size_t cartesian_ao_count(const core::System& system) { return ao_count(system); }
}
namespace generativeqc::dft {
struct GridSpec {};
std::size_t grid_point_count(const core::System&, const GridSpec&) { return 4096; }
std::size_t cuda_resident_grid_bytes(std::size_t, std::size_t) { return 1024; }
std::size_t cuda_quadrature_bytes(std::size_t, std::size_t) { return 512; }
template<class... Arguments> std::size_t cuda_ks_state_bytes(Arguments&&...) { return 512; }
enum class CudaXcAoPrecision { Fp64 };
struct Layout { std::size_t device_bytes{512}; };
template<class... Arguments> Layout cuda_xc_layout_shape(Arguments&&...) { return {}; }
namespace dispersion {
struct Usage { std::size_t device_bytes{}; };
template<class... Arguments> Usage d4_resource_usage(Arguments&&...) { return {}; }
}
}
struct NativeKsExecutionPlan {
  dft::SemilocalFamily semilocal_family{dft::SemilocalFamily::Pbe};
  bool automatic_program{}, generated_split_hybrid{}, range_exchange{}, nonlocal_correlation{};
  bool d4_correction{};
  struct { int kind{}; bool composite_profile{}; } dispersion;
  unsigned spin_channels{1};
};
enum class KsProviderPurpose { mandatory_scf };
template<class... Arguments> unsigned ks_direct_derivative_order(Arguments&&...) { return 1; }
template<class... Arguments> std::size_t ks_provider_bytes(Arguments&&...) { return 256; }
template<class... Arguments> std::size_t ks_fitted_provider_budget(Arguments&&...) { return 256; }
template<class... Arguments> int ks_fitted_projection_reservation(Arguments&&...) { return 0; }
std::optional<core::System> ks_auxiliary_for_system(const core::System&,
                                                  const std::optional<core::System>& auxiliary) {
  return auxiliary;
}
unsigned ks_xc_functional_code(const NativeKsExecutionPlan&) { return 1; }
dft::CudaStationaryBinding binding;
bool binding_available{true};
template<class... Arguments>
const dft::CudaStationaryBinding* native_cuda_stationary_binding(Arguments&&...) {
  return binding_available ? &binding : nullptr;
}
"""


DRIVER = r"""
int main() {
  binding.model = {1, 4, 1, 3, 1.0, 1.0, 0.0};
  binding.ao_jets = 4;
  binding.spin_blocks = 1;
  binding.numeric_bytes = +[](std::size_t, std::size_t, std::size_t, std::size_t,
                              std::size_t, std::size_t, std::size_t, bool cached) {
    return std::size_t{cached ? 2048U : 1024U};
  };
  binding.geometry_fp32 = +[](void*, const dft::GridTaskView*, const double*, std::size_t,
                              std::size_t, const double*, const double*, std::uint64_t,
                              std::uint64_t, char*, std::size_t) { return 0; };
  const std::array<double, 3> budget{1e-5, 1e-4, 3e-5};
  for (const auto approximation : {scf::FockApproximation::Exact,
                                   scf::FockApproximation::DensityFitted}) {
    for (const auto aos : {24U, 96U, 97U, 512U, 768U}) {
      core::System system;
      system.atoms.resize(12);
      system.shells.resize(aos);
      for (auto& shell : system.shells) shell.primitives.resize(1);
      const std::vector<core::System> systems{system, system};
      NativeKsExecutionPlan plan;
      scf::ScfOptions options;
      options.resolved_fock_build->spec.coulomb.approximation = approximation;
      options.resolved_fock_build->spec.exchange.approximation = approximation;
      const auto reserve = [&] {
        return ks_device_capacity_reservation(systems, plan, options, {},
                                              GENERATIVEQC_BACKEND_CUDA, std::nullopt);
      };
      auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
      ledger->limit = std::size_t{1} << 30;
      ledger->admission = runtime::DeviceResourceAdmission::NumericCap;
      runtime::active_device_resource_ledger = ledger;
      options.energy_tolerance = 1e-9;
      options.density_tolerance = 1e-5;
      auto standard = reserve();
      assert(standard && ledger->capacity_reserved);
      const auto strict_bytes = ledger->capacity_reserved;
      standard.reset();
      assert(!ledger->capacity_reserved);
      options.application_precision_budget = budget;
      auto fast = reserve();
      assert(fast && ledger->capacity_reserved == strict_bytes);
      fast.reset();
      for (const auto limit : {strict_bytes - 1, strict_bytes}) {
        ledger->limit = limit;
        auto bounded = reserve();
        assert(bool(bounded) == (limit == strict_bytes));
        bounded.reset();
        assert(!ledger->capacity_reserved && !ledger->live);
      }
      ledger->limit = std::size_t{1} << 30;
      options.energy_tolerance = 1e-10;
      options.density_tolerance = 1e-8;
      const auto qualification = dft::application_gradient_qualification(
          binding, budget, aos, 12, 0, {1e-10, 1e-8, 1e-12});
      auto selected = reserve();
      assert(bool(selected) == !qualification);
      selected.reset();
      assert(!ledger->capacity_reserved);
      options.precision_mode = GENERATIVEQC_PRECISION_AUTO;
      assert(!reserve() && !ledger->capacity_reserved);
      options.precision_mode = GENERATIVEQC_PRECISION_FP64;
      options.energy_tolerance = 1e-9;
      options.density_tolerance = 1e-5;
      plan.range_exchange = true;
      assert(!reserve() && !ledger->capacity_reserved);
      plan.range_exchange = false;
      options.preliminary_guess = true;
      assert(!reserve() && !ledger->capacity_reserved);
      options.preliminary_guess = false;
      ledger->admission = runtime::DeviceResourceAdmission::SelectedInventory;
      assert(!reserve() && !ledger->capacity_reserved);
      ledger->admission = runtime::DeviceResourceAdmission::NumericCap;
      binding_available = false;
      auto unbound = reserve();
      assert(bool(unbound) == (approximation == scf::FockApproximation::DensityFitted));
      unbound.reset();
      binding_available = true;
      assert(!ledger->capacity_reserved);
      runtime::active_device_resource_ledger.reset();
    }
  }
  // The same FP64 force tile applies with or without a budget. Large shapes
  // cannot borrow the separately qualified small mixed-gradient domain.
  for (const auto aos : {512U, 768U}) {
    for (const auto allowances : {std::array<double, 3>{}, budget}) {
      assert(!dft::application_gradient_qualification(binding, allowances, aos, 96, 2,
                                                       {1e-10, 1e-8, 1e-12}));
      assert(dft::automatic_ks_force_tile(true, true, true, 2, 96, aos, 1U << 20, 12) == 1024);
      assert(dft::automatic_ks_force_tile(true, false, true, 2, 96, aos, 1U << 20, 12) == 256);
    }
  }
}
"""
