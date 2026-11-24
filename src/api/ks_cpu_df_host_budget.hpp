#pragma once

#include <algorithm>

#include "api/host_resources.hpp"
#include "api/named_ks_inputs.hpp"
#include "methods/dft_method_preparation.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/density_fitting.hpp"
#include "scf/ks_host_resources.hpp"

namespace generativeqc::api {

/** Revalidate an effective host limit using the systems and controls the native
 * prepared owner will actually copy. Fleet preparation/solve is serialized;
 * retained providers sum and the existing inventory takes the largest scratch.
 */
class CpuDfKsHostAdmission {
 public:
  CpuDfKsHostAdmission(const NamedKsInputs& named, generativeqc_backend backend)
      : limit_(named.selection.cpu_df_host_maximum_bytes),
        properties_(named.selection.cpu_df_host_properties),
        shared_(named.selection.host_resources ? named.selection.host_resources->ledger : nullptr) {
    if (properties_ != GENERATIVEQC_PROPERTY_ENERGY &&
        properties_ != (GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES))
      throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                                 "invalid CPU DF host property set");
    const auto& method = named.method.base;
    if (backend != GENERATIVEQC_BACKEND_CPU_REFERENCE ||
        !methods::named_ks::cpu_df_energy_program(named.program->identity) ||
        named.selection.cpu_packed_eri_maximum_bytes ||
        (method.initial_guess && method.initial_guess->kind) ||
        scf::cpu_materialized_df_derivatives_requested())
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "CPU DF host admission requires the native semilocal DF-J owner");
    methods::detail::NativeKsExecutionPlan plan;
    auto options = methods::detail::dft_options(method, backend, plan);
    if ((options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE &&
         options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_AUTO) ||
        options.precision_mode != GENERATIVEQC_PRECISION_FP64 || options.incremental_direct_jk)
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "CPU DF host admission requires ordinary FP64 DF-J replay");
    grid_ = methods::detail::ks_grid_options(method, options, plan);
    first_ao_derivatives_ = plan.semilocal_family != dft::SemilocalFamily::Lda;
    spins_ = plan.spin_channels;
    diis_ = options.diis_history;
    iterations_ = options.max_iterations;
    tile_ = std::max(options.xc_tile_points,
                     forces() ? options.stationary_tile_points() : std::size_t{0});
  }

  void append(const core::System& system, const core::System* auxiliary) {
    const auto orbital = describe(system);
    auto fit = auxiliary ? describe(*auxiliary) : orbital;
    fleet_.append({system.atoms.size(), orbital.aos, orbital.cartesian_aos, orbital.shells,
                   orbital.primitives, spins_, orbital.maximum_angular, 0,
                   dft::grid_point_count(system, grid_), grid_.radial_points, grid_.angular_polar,
                   tile_, diis_, iterations_, first_ao_derivatives_, false, 0, fit},
                  1, scf::cpu_ks_host_execution(false, forces()));
    if (fleet_.inventory()[14] > limit_)
      throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                 "native CPU DF KS modeled host capacity exceeds the limit");
  }

  generativeqc_property_flags properties() const noexcept { return properties_; }
  bool forces() const noexcept { return properties_ & GENERATIVEQC_PROPERTY_FORCES; }

  std::unique_lock<std::mutex> use_workspace() const {
    return shared_ ? shared_->use_workspace() : std::unique_lock<std::mutex>{};
  }

  /** The local per-owner gate has already authenticated this inventory. */
  std::shared_ptr<runtime::HostResourceLease> reserve() const {
    if (!shared_) return nullptr;
    const auto inventory = fleet_.inventory();
    try {
      return std::make_shared<runtime::HostResourceLease>(shared_, inventory[12], inventory[13]);
    } catch (const std::bad_alloc&) {
      throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                 "shared host capacity exceeds the limit");
    }
  }

 private:
  static scf::CpuKsDfHostShape describe(const core::System& system) {
    if (!system.ecp_terms.empty() ||
        std::ranges::any_of(system.atoms, [](const auto& atom) { return atom.ecp_core != 0; }))
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "CPU DF host admission excludes effective core potentials");
    scf::CpuKsDfHostShape shape{molecule::ao_count(system), molecule::cartesian_ao_count(system),
                                system.shells.size(), 0, 0};
    for (const auto& shell : system.shells) {
      shape.primitives = runtime::size_add(shape.primitives, shell.primitives.size());
      shape.maximum_angular =
          std::max<std::uint64_t>(shape.maximum_angular, shell.angular_momentum);
    }
    return shape;
  }

  std::uint64_t limit_;
  generativeqc_property_flags properties_;
  std::shared_ptr<runtime::HostResourceLedger> shared_;
  dft::GridSpec grid_;
  std::uint64_t spins_{}, diis_{}, iterations_{}, tile_{};
  bool first_ao_derivatives_{};
  scf::CpuKsHostFleetResources fleet_;
};

/** Recheck the environment-selected provider mode before a geometry rebuild. */
inline bool cpu_df_host_replay_allowed(generativeqc_property_flags admitted, bool forces) noexcept {
  return !admitted || ((!forces || (admitted & GENERATIVEQC_PROPERTY_FORCES)) &&
                       !scf::cpu_materialized_df_derivatives_requested());
}

}  // namespace generativeqc::api
