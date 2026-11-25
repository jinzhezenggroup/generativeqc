#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <string_view>

#include "api/host_resources.hpp"
#include "api/ks_host_hessian.hpp"
#include "api/named_ks_inputs.hpp"
#include "methods/dft_method_preparation.hpp"
#include "methods/warm_seed_capacity.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/density_fitting.hpp"
#include "scf/ks_host_resources.hpp"
#include "scf/rks_cold_fock.hpp"
#include "scf/stationary_integral_gradient.hpp"

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
                     std::uint64_t derivative_workspace, bool reserve_forces,
                     bool reserve_seed_restore = false, bool saddle_ecp_forces = false)
      : limit_(limit),
        properties_(GENERATIVEQC_PROPERTY_ENERGY |
                    (reserve_forces ? GENERATIVEQC_PROPERTY_FORCES : 0)),
        shared_(resources.ledger),
        derivative_workspace_(derivative_workspace),
        hessian_identity_(named.program->identity),
        direct_d3_(named.program->dispersion.kind != 0),
        packed_(named.selection.cpu_packed_eri_maximum_bytes != 0),
        reserve_seed_restore_(reserve_seed_restore),
        saddle_ecp_forces_(saddle_ecp_forces),
        remaining_packed_bytes_(named.selection.cpu_packed_eri_maximum_bytes) {
    const auto& method = named.method.base;
    if (!shared_ || !derivative_workspace_ || named.selection.cpu_df_host_properties ||
        limit_ > std::numeric_limits<std::int64_t>::max())
      throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                                 "invalid CPU Direct Hessian host admission controls");
    if (backend != GENERATIVEQC_BACKEND_CPU_REFERENCE || named.program->dispersion.kind == 3 ||
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
    force_has_exact_exchange_ = named.program->exchange_count != 0;
  }

  void append(const core::System& system, const core::System* auxiliary,
              const scf::RksColdFockDamping* cold_damping = nullptr) {
    const auto orbital = describe(system);
    std::uint64_t ecp_workspace{};
    if (has_ecp(system)) {
      if (derivative_workspace_ && forces() && !saddle_ecp_forces_)
        throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                   "CPU Hessian host force reservation requires all electrons");
      if (!derivative_workspace_)
        throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                   "CPU DF KS host admission excludes effective core potentials");
      validate_host_hessian_system(system, hessian_identity_);
      if (generativeqc_resource_ecp_workspace_v1(orbital.aos, orbital.cartesian_aos,
                                                 system.atoms.size(), orbital.primitives,
                                                 system.ecp_terms.size(), 0, &ecp_workspace))
        throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                   "ECP setup inventory exceeds native capacity");
      if (saddle_ecp_forces_) {
        // The force fleet already reserves its complete source ceiling. Check
        // the same producer-specific bound that applies at replay.
        const auto force_bytes = scf::cpu_stationary_direct_integral_gradient_capacity(
            system, force_has_exact_exchange_, spins_ == 2);
        if (force_bytes > dft::cpu_stationary_force_source_bytes)
          throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                     "ECP Direct force source exceeds its native capacity");
      }
    }
    if (packed_) {
      if (system.basis_representation != GENERATIVEQC_BASIS_SPHERICAL ||
          orbital.maximum_angular > 3)
        throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                                   "packed Coulomb ERIs require a real spherical s/p/d/f basis");
      // Enforce the independent fleet ceiling before staging any retained values.
      const auto values = scf::cpu_ks_packed_value_bytes(orbital.aos);
      if (values > remaining_packed_bytes_)
        throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                   "packed spherical ERI capacity exceeded for retained KS batch");
      remaining_packed_bytes_ -= values;
    }
    std::optional<scf::CpuKsDfHostShape> fit;
    if (!derivative_workspace_) {
      if (auxiliary && has_ecp(*auxiliary))
        throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                                   "CPU DF KS host admission excludes auxiliary ECPs");
      fit = auxiliary ? describe(*auxiliary) : orbital;
    }
    fleet_.append({system.atoms.size(), orbital.aos, orbital.cartesian_aos, orbital.shells,
                   orbital.primitives, spins_, orbital.maximum_angular, system.ecp_terms.size(),
                   dft::grid_point_count(system, grid_), grid_.radial_points, grid_.angular_polar,
                   tile_, diis_, iterations_, first_ao_derivatives_, false, ecp_workspace, fit},
                  1, scf::cpu_ks_host_execution(packed_, forces()));
    if (direct_d3_) append_direct_d3(system);
    if (reserve_seed_restore_)
      seed_restore_workspace_ = cpu_seed_restore_bytes(system, orbital.aos, ecp_workspace);
    if (cold_damping)
      damping_workspace_ =
          std::max(damping_workspace_, cold_damping->proposal_bytes(orbital.aos, iterations_));
    const auto inventory = fleet_.inventory();
    if (runtime::size_add(inventory[12], workspace_bytes(inventory[13])) > limit_)
      throw methods::MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                                 derivative_workspace_
                                     ? "native CPU Direct Hessian host capacity exceeds the limit"
                                     : "native CPU DF KS modeled host capacity exceeds the limit");
  }

  generativeqc_property_flags properties() const noexcept { return properties_; }
  bool forces() const noexcept { return properties_ & GENERATIVEQC_PROPERTY_FORCES; }
  std::uint64_t derivative_workspace() const noexcept { return derivative_workspace_; }

  /** After admitting all mandatory providers and the floor, give spare local
   * capacity to optional derivative schedules without an extra prepare/retry.
   * The normal lease still atomically revalidates the shared ledger.
   */
  void select_derivative_workspace(std::uint64_t ceiling) {
    const auto inventory = fleet_.inventory();
    const auto mandatory_peak = runtime::size_add(inventory[12], workspace_bytes(inventory[13]));
    derivative_workspace_ += std::min(ceiling - derivative_workspace_, limit_ - mandatory_peak);
  }

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
  /** Single cold-prepared CPU batch. Restore copies the seed
   * candidate into the batch and validates it on its original geometry. The
   * copy with restored coordinates and RawSource's copy/expansion coexist
   * with the source metric, spin panels, products and scalar Jacobi solver.
   * The two detached CLI snapshots are charged outside this lease.
   */
  std::size_t cpu_seed_restore_bytes(const core::System& system, std::size_t basis,
                                     std::size_t ecp_setup_bytes) const {
    using runtime::size_add;
    using runtime::size_mul;
    const auto square = size_mul(basis, basis);
    const auto candidate =
        methods::warm_seed_item_capacity(size_mul(spins_, square), system.atoms.size());
    auto bytes =
        size_add(posthf::source_scratch_bytes, size_mul(2, posthf::source_capacity(system)));
    if (!system.ecp_terms.empty()) {
      // RawSource owns an ECP value matrix even though restore only reads
      // overlap. Its producer scratch and retained copy are distinct owners.
      bytes = size_add(bytes, std::max(ecp_setup_bytes, integrals::ecp_workspace_capacity(system)));
      bytes = size_add(bytes, size_mul(sizeof(double), square));
    }
    bytes = size_add(bytes, candidate.total_bytes);
    // The validator's spin copies, dense products and permutation are all
    // bounded, irrespective of seed occupation or coordinate displacement.
    const auto matrices = size_mul(16 + 2 * spins_, square);
    return size_add(bytes, size_mul(sizeof(double), size_add(matrices, size_mul(6, basis))));
  }

  /** The derivative conservatively overlaps the full primal peak. One fleet
   * item can consume the derivative reservation at a time under the pool lock. */
  std::uint64_t workspace_bytes(std::uint64_t primal) const {
    // The previous damped proposal stays live during SCF, but not during
    // source-metric seed validation or after the cold prefix completes.
    const auto primal_peak = runtime::size_add(primal, damping_workspace_);
    return runtime::size_add(std::max(primal_peak, seed_restore_workspace_), derivative_workspace_);
  }

  void append_direct_d3(const core::System& system);

  static bool has_ecp(const core::System& system) {
    return !system.ecp_terms.empty() ||
           std::ranges::any_of(system.atoms, [](const auto& atom) { return atom.ecp_core != 0; });
  }

  static scf::CpuKsDfHostShape describe(const core::System& system) {
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
  std::string_view hessian_identity_;
  bool direct_d3_{};
  bool packed_{};
  bool reserve_seed_restore_{};
  bool saddle_ecp_forces_{};
  std::uint64_t remaining_packed_bytes_{};
  dft::GridSpec grid_;
  std::uint64_t spins_{}, diis_{}, iterations_{}, tile_{};
  bool first_ao_derivatives_{};
  bool force_has_exact_exchange_{};
  std::size_t seed_restore_workspace_{};
  std::size_t damping_workspace_{};
  scf::CpuKsHostFleetResources fleet_;
};

/** Recheck the environment-selected provider mode before a geometry rebuild. */
inline bool cpu_df_host_replay_allowed(generativeqc_property_flags admitted, bool forces) noexcept {
  return !admitted || ((!forces || (admitted & GENERATIVEQC_PROPERTY_FORCES)) &&
                       !scf::cpu_materialized_df_derivatives_requested());
}

}  // namespace generativeqc::api
