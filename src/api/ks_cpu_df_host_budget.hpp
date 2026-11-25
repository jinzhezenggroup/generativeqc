#pragma once

#include <algorithm>
#include <limits>
#include <optional>

#include "api/host_resources.hpp"
#include "api/named_ks_inputs.hpp"
#include "generativeqc/ks_hessian.h"
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
class CpuKsHostAdmission {
 public:
  CpuKsHostAdmission(const NamedKsInputs& named, generativeqc_backend backend)
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

  CpuKsHostAdmission(const NamedKsInputs& named, generativeqc_backend backend,
                     const generativeqc_host_resources& resources, std::uint64_t limit,
                     std::uint64_t derivative_workspace)
      : limit_(limit),
        properties_(GENERATIVEQC_PROPERTY_ENERGY),
        shared_(resources.ledger),
        derivative_workspace_(derivative_workspace),
        direct_d3_(named.program->dispersion.kind != 0) {
    const auto& method = named.method.base;
    if (!shared_ || !derivative_workspace_ || named.selection.cpu_df_host_properties ||
        limit_ > std::numeric_limits<std::int64_t>::max())
      throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                                 "invalid CPU Direct Hessian host admission controls");
    if (backend != GENERATIVEQC_BACKEND_CPU_REFERENCE ||
        named.selection.cpu_packed_eri_maximum_bytes || named.program->dispersion.kind == 3 ||
        !generativeqc_ks_hessian_cpu_method_v1(named.program->identity.data()))
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "host admission requires a linked ordinary Direct CPU Hessian");
    methods::detail::NativeKsExecutionPlan plan;
    auto options = methods::detail::dft_options(method, backend, plan);
    if (options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
        options.precision_mode != GENERATIVEQC_PRECISION_FP64 || options.incremental_direct_jk ||
        options.preliminary_guess || plan.nonlocal_correlation)
      throw methods::MethodError(
          GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "CPU Direct Hessian host admission excludes approximate, nonlocal, "
          "incremental and preliminary SCF providers");
    grid_ = methods::detail::ks_grid_options(method, options, plan);
    first_ao_derivatives_ = plan.semilocal_family != dft::SemilocalFamily::Lda;
    spins_ = plan.spin_channels;
    diis_ = options.diis_history;
    iterations_ = options.max_iterations;
    tile_ = options.xc_tile_points;
  }

  void append(const core::System& system, const core::System* auxiliary) {
    const auto orbital = describe(system);
    std::optional<scf::CpuKsDfHostShape> fit;
    if (!derivative_workspace_) fit = auxiliary ? describe(*auxiliary) : orbital;
    fleet_.append({system.atoms.size(), orbital.aos, orbital.cartesian_aos, orbital.shells,
                   orbital.primitives, spins_, orbital.maximum_angular, 0,
                   dft::grid_point_count(system, grid_), grid_.radial_points, grid_.angular_polar,
                   tile_, diis_, iterations_, first_ao_derivatives_, false, 0, fit},
                  1, scf::cpu_ks_host_execution(false, forces()));
    if (direct_d3_) append_direct_d3(system);
    const auto inventory = fleet_.inventory();
    if (runtime::size_add(inventory[12], workspace_bytes(inventory[13])) > limit_)
      throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                 derivative_workspace_
                                     ? "native CPU Direct Hessian host capacity exceeds the limit"
                                     : "native CPU DF KS modeled host capacity exceeds the limit");
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
      return std::make_shared<runtime::HostResourceLease>(shared_, inventory[12],
                                                          workspace_bytes(inventory[13]));
    } catch (const std::bad_alloc&) {
      throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                 "shared host capacity exceeds the limit");
    }
  }

 private:
  /** Count the derivative reservation in addition to the primal scratch. The
   * exclusive pool lock serializes derivative replay across leased items. */
  std::uint64_t workspace_bytes(std::uint64_t primal) const {
    return runtime::size_add(primal, derivative_workspace_);
  }

  void append_direct_d3(const core::System& system);

  static scf::CpuKsDfHostShape describe(const core::System& system) {
    if (!system.ecp_terms.empty() ||
        std::ranges::any_of(system.atoms, [](const auto& atom) { return atom.ecp_core != 0; }))
      throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                 "CPU KS host admission excludes effective core potentials");
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
  std::uint64_t derivative_workspace_{};
  bool direct_d3_{};
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
