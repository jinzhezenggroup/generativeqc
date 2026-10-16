"""Execute budget/domain admission and the backwards-compatible KS transport."""

import ctypes
import gc
import subprocess
from pathlib import Path
from typing import Any

from generativeqc import GridSpec, KsOptions, _native
from generativeqc.ks import native_ks_options, resolve_ks_options

ROOT = Path(__file__).resolve().parents[2]


def test_native_budget_domain_admission(
    required_native_cxx: Any, tmp_path: Path
) -> None:
    source = tmp_path / "admission.cpp"
    source.write_text(r"""
#include <cassert>
#include <limits>
#include "dft/application_precision.hpp"
#include "scf/application_precision.hpp"
#include "scf/fock_precision.hpp"
int main() {
  using generativeqc::dft::nonlocal_confirmation_tolerance;
  using generativeqc::dft::nonlocal_inner_tolerance;
  assert(nonlocal_confirmation_tolerance(false, 1e-4) == 1e-9);
  assert(nonlocal_confirmation_tolerance(true, 1e-4) == 1e-5);
  assert(nonlocal_inner_tolerance(false, 1e-2, 1e-4) == 1e-7);
  assert(nonlocal_inner_tolerance(false, 1e-6, 1e-4) == 1e-9);
  assert(nonlocal_inner_tolerance(true, 1e-2, 1e-4) == 1e-5);
  for (const auto density_tolerance : {1e-10, 1e-12}) {
    assert(nonlocal_confirmation_tolerance(false, density_tolerance) == density_tolerance);
    assert(nonlocal_confirmation_tolerance(true, density_tolerance) == density_tolerance);
    assert(nonlocal_inner_tolerance(true, 1e-2, density_tolerance) == density_tolerance);
  }
  using generativeqc::dft::admit_application_xc_warmup;
  using generativeqc::dft::admit_application_gradient;
  const std::array<double, 3> loose{1e-5, 1e-4, 3e-5};
  using generativeqc::dft::admit_application_fock;
  using namespace generativeqc::scf;
  ResolvedFockBuild direct;
  direct.backend = FockBackend::Cuda;
  direct.spec.derivative_order = 0;
  for (const auto schedule : {FockSchedule::CudaFused, FockSchedule::CudaIndependent}) {
    direct.schedule = schedule;
    for (const auto coefficient : {-0.5, -0.27, -0.125, -0.1, 0.0, 0.25}) {
      direct.spec.exchange.coefficient = coefficient;
      const auto modules = cuda_fock_precision_capabilities(direct);
      assert(modules.coulomb && modules.exchange);
    }
  }
  direct.schedule = FockSchedule::CudaDfResident;
  assert(!cuda_fock_precision_capabilities(direct).any_lower_precision());
  direct.schedule = FockSchedule::CudaIndependent;
  direct.spec.exchange.present = false;
  assert(cuda_fock_precision_capabilities(direct).coulomb);
  assert(!cuda_fock_precision_capabilities(direct).exchange);
  direct.spec.exchange.present = true;
  direct.spec.exchange.op = FockOperator::LongRange;
  direct.spec.exchange.omega = 0.3;
  assert(cuda_fock_precision_capabilities(direct).coulomb);
  assert(!cuda_fock_precision_capabilities(direct).exchange);
  direct.spec.coulomb.approximation = FockApproximation::DensityFitted;
  assert(!cuda_fock_precision_capabilities(direct).any_lower_precision());
  direct.spec.coulomb.approximation = FockApproximation::Exact;
  direct.spec.spin = FockSpin::Unrestricted;
  assert(!cuda_fock_precision_capabilities(direct).any_lower_precision());
  direct.spec.spin = FockSpin::Restricted;
  direct.spec.derivative_order = 1;
  assert(!cuda_fock_precision_capabilities(direct).any_lower_precision());
  direct.spec.derivative_order = 0;
  direct.backend = FockBackend::Cpu;
  assert(!cuda_fock_precision_capabilities(direct).any_lower_precision());
  auto range = direct;
  range.backend = FockBackend::Cuda;
  range.spec.coulomb.present = false;
  for (const auto op : {FockOperator::ShortRange, FockOperator::LongRange}) {
    range.spec.exchange.op = op;
    assert(cuda_range_exchange_precision_capability(range));
    range.spec.exchange.omega = 0.0;
    assert(!cuda_range_exchange_precision_capability(range));
    range.spec.exchange.omega = 0.3;
    range.spec.derivative_order = 1;
    assert(!cuda_range_exchange_precision_capability(range));
    range.spec.derivative_order = 0;
    range.spec.spin = FockSpin::Unrestricted;
    assert(!cuda_range_exchange_precision_capability(range));
    range.spec.spin = FockSpin::Restricted;
    range.spec.exchange.approximation = FockApproximation::DensityFitted;
    assert(!cuda_range_exchange_precision_capability(range));
    range.spec.exchange.approximation = FockApproximation::Exact;
    range.schedule = FockSchedule::CudaDfResident;
    assert(!cuda_range_exchange_precision_capability(range));
    range.schedule = FockSchedule::CudaIndependent;
  }
  for (const auto aos : {24u, 384u, 768u})
    assert(admit_application_fock(loose, aos, 96, 2, true, true, true));
  for (const auto aos : {0u, 23u, 769u})
    assert(!admit_application_fock(loose, aos, 96, 2, true, true, true));
  assert(!admit_application_fock(loose, 768, 97, 2, true, true, true));
  assert(!admit_application_fock(loose, 768, 1, 2, true, true, true));
  assert(!admit_application_fock(loose, 768, 96, 3, true, true, true));
  assert(!admit_application_fock(loose, 768, 96, 2, false, true, true));
  assert(!admit_application_fock(loose, 768, 96, 2, true, false, true));
  assert(!admit_application_fock(loose, 768, 96, 2, true, true, false));
  for (unsigned index = 0; index < 3; ++index) {
    for (const auto value : {loose[index] * 0.9, -1.0,
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      auto invalid = loose;
      invalid[index] = value;
      assert(!admit_application_fock(invalid, 768, 96, 2, true, true, true));
    }
  }
  using generativeqc::scf::admit_application_hf_gradient;
  for (const auto aos : {24u, 96u})
    assert(admit_application_hf_gradient(loose, aos, 24, 2, true, true));
  for (const auto aos : {0u, 23u, 97u})
    assert(!admit_application_hf_gradient(loose, aos, 24, 2, true, true));
  assert(!admit_application_hf_gradient(loose, 96, 24, 3, true, true));
  assert(!admit_application_hf_gradient(loose, 96, 24, 2, false, true));
  assert(!admit_application_hf_gradient(loose, 96, 24, 2, true, false));
  assert(!admit_application_hf_gradient(loose, 96, 49, 2, true, true));
  assert(!admit_application_hf_gradient({1e-5, 0, 0}, 96, 24, 2, true, true));
  using generativeqc::scf::admit_application_df_hf_gradient;
  assert(admit_application_df_hf_gradient(loose, 96, 464, 12, 2, 3, true, true));
  assert(!admit_application_df_hf_gradient(loose, 96, 513, 12, 2, 3, true, true));
  assert(!admit_application_df_hf_gradient(loose, 96, 0, 12, 2, 3, true, true));
  assert(!admit_application_df_hf_gradient(loose, 96, 464, 12, 2, 4, true, true));
  assert(!admit_application_df_hf_gradient(loose, 96, 464, 13, 2, 3, true, true));
  assert(!admit_application_df_hf_gradient(loose, 96, 464, 12, 2, 3, false, true));
  assert(!admit_application_df_hf_gradient(loose, 96, 464, 12, 2, 3, true, false));
  for (const auto aos : {24u, 96u, 192u, 384u})
    assert(admit_application_gradient(loose, aos, 24, 2, true));
  for (const auto aos : {0u, 23u, 385u})
    assert(!admit_application_gradient(loose, aos, 24, 2, true));
  assert(!admit_application_gradient(loose, 96, 49, 2, true));
  assert(!admit_application_gradient(loose, 96, 1, 2, true));
  assert(!admit_application_gradient(loose, 96, 24, 3, true));
  assert(!admit_application_gradient(loose, 96, 24, 2, false));
  assert(!admit_application_gradient({1e-5, 0, 0}, 96, 24, 2, true));
  assert(admit_application_gradient({1e-6, 1e-5, 0}, 96, 24, 2, true));
  for (const auto aos : {24u, 96u, 192u, 256u})
    assert(admit_application_xc_warmup(loose, aos, 2, true, true, true));
  for (const auto aos : {0u, 23u, 257u})
    assert(!admit_application_xc_warmup(loose, aos, 2, true, true, true));
  assert(!admit_application_xc_warmup(loose, 96, 3, true, true, true));
  assert(!admit_application_xc_warmup(loose, 96, 2, false, true, true));
  assert(!admit_application_xc_warmup(loose, 96, 2, true, false, true));
  assert(!admit_application_xc_warmup(loose, 96, 2, true, true, false));
  assert(admit_application_xc_warmup({1e-6, 1e-5, 3e-6}, 96, 2, true, true, true));
  assert(admit_application_xc_warmup({1e-6, 0, 0}, 96, 2, true, true, true));
  const std::array<double, 3> minimum{1e-6, 1e-5, 3e-6};
  for (unsigned index = 0; index < 3; ++index) {
    for (const auto value : {minimum[index] * 0.9, -1.0,
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      auto invalid = loose;
      invalid[index] = value;
      assert(!admit_application_xc_warmup(invalid, 96, 2, true, true, true));
      assert(!admit_application_gradient(invalid, 96, 24, 2, true));
      assert(!admit_application_hf_gradient(invalid, 96, 24, 2, true, true));
      assert(!admit_application_df_hf_gradient(invalid, 96, 464, 12, 2, 3, true, true));
    }
  }
  assert(offsetof(generativeqc::dft::ApplicationPrecisionKsOptions, energy) ==
         sizeof(generativeqc_ks_options));
  using generativeqc::dft::application_precision_requested;
  generativeqc_ks_options legacy{};
  legacy.struct_size = sizeof(legacy);
  legacy.abi_version = GENERATIVEQC_ABI_VERSION;
  assert(!application_precision_requested(nullptr));
  assert(!application_precision_requested(&legacy));
  generativeqc::dft::ApplicationPrecisionKsOptions extended{};
  extended.base = legacy;
  extended.energy = 1e-5;
  for (unsigned size = sizeof(legacy); size < sizeof(extended); ++size) {
    extended.base.struct_size = size;
    assert(!application_precision_requested(&extended.base));
  }
  extended.base.struct_size = sizeof(extended);
  assert(application_precision_requested(&extended.base));
  for (const auto energy : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    extended.energy = energy;
    assert(!application_precision_requested(&extended.base));
  }
  generativeqc::dft::CudaStationaryBinding binding;
  binding.geometry_fp32 = +[](void*, const generativeqc::dft::GridTaskView*, const double*,
                              std::size_t, std::size_t, const double*, const double*,
                              std::uint64_t, std::uint64_t, char*, std::size_t) { return 0; };
  using generativeqc::dft::application_gradient_qualification;
  const auto qualify = [&] (unsigned aos) {
    return application_gradient_qualification(binding, loose, aos, 12, 2);
  };
  binding.model = {1, 4, 1, 3, 0.75, 1.0, 0.25};
  assert(qualify(384) == 1);
  binding.model = {1, 4, 1, 3, 1.0, 1.0, 0.0};
  assert(qualify(24) == 2 && qualify(96) == 2 && qualify(97) == 0);
  binding.model = {3, 1, 1, 3, 1.0, 1.0, 0.2};
  assert(qualify(96) == 3);
  binding.model.full_range_exchange = 0.25;
  assert(qualify(96) == 0);
  binding.model = {2, 1, 1, 3, 1.0, 1.0, 0.0};
  assert(qualify(96) == 4);
  binding.model.spin_blocks = 2;
  assert(qualify(96) == 0);
  binding.model.spin_blocks = 1;
  binding.model = {0x201c2, 4, 1, 3, 1.0, 1.0, 0.54};
  assert(qualify(24) == 6 && qualify(96) == 6 && qualify(97) == 0);
  binding.model.domain_version = 1;
  assert(qualify(96) == 0);
  binding.model.domain_version = 4;
  binding.model.full_range_exchange = 0.5;
  assert(qualify(96) == 0);
  binding.model = {2, 1, 1, 3, 1.0, 1.0, 0.0};
  binding.geometry_fp32 = nullptr;
  assert(qualify(96) == 0);
  assert(generativeqc::dft::stationary_gradient_model_qualification(binding, 96) == 4);
}
""")
    executable = tmp_path / "admission"
    required_native_cxx.build_executable(
        [source],
        executable,
        compile_args=("-std=c++20", f"-I{ROOT / 'src'}", f"-I{ROOT / 'include'}"),
    )
    subprocess.run([executable], check=True, timeout=20)


def test_budget_transport_owns_extended_prefix_without_changing_legacy_size() -> None:
    grid = GridSpec(16, 8, 16)
    strict = native_ks_options(resolve_ks_options("pbe0-rks", KsOptions(grid=grid)))
    mixed = native_ks_options(
        resolve_ks_options(
            "pbe0-rks", KsOptions(grid=grid, _precision_budget=(1e-5, 1e-4, 3e-5))
        )
    )
    gc.collect()
    assert type(strict) is type(mixed) is _native.KsOptionsDescriptor
    assert strict.struct_size == ctypes.sizeof(_native.KsOptionsDescriptor)
    assert mixed.struct_size == ctypes.sizeof(
        _native.ApplicationPrecisionKsOptionsDescriptor
    )
    assert mixed.struct_size == strict.struct_size + 3 * ctypes.sizeof(ctypes.c_double)
    assert mixed._extension_owner.application_energy_budget == 1e-5
    assert mixed._extension_owner.application_max_force_budget == 1e-4
    assert mixed._extension_owner.application_rms_force_budget == 3e-5
    assert mixed.semilocal_component_count == strict.semilocal_component_count
