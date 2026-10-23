#include "dft/cuda_ks.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "dft/application_precision.hpp"
#include "dft/cuda_ks_final_validation_policy.hpp"
#include "dft/cuda_ks_kernels.hpp"
#include "dft/cuda_ks_precision.hpp"
#include "dft/cuda_xc.hpp"
#include "dft/energy_change.hpp"
#include "dft/ks_incremental_policy.hpp"
#include "dft/ks_stopping_policy.hpp"
#include "dft/mixed_precision_warmup.hpp"
#include "dft/xc.hpp"
#include "generated_split_hybrid_registry.cuh"
#include "generativeqc/generativeqc.hpp"
#include "molecule/basis.hpp"
#include "runtime/compiled_execution_region.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "runtime/cuda_resources.cuh"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "runtime/solver_region_cuda.cuh"
#include "scf/convergence_limits.hpp"
#include "scf/cuda/direct_jk_source_lease.hpp"
#include "scf/cuda/direct_jk_source_policy.hpp"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "scf/cuda/mean_field_setup.hpp"
#include "scf/cuda/resident_final_validation.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/scf_constants.hpp"
#include "scf/cuda/scf_density_kernels.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda/structured_matrix.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/eigen_frame.hpp"
#include "scf/solver/proposal_control.hpp"
#include "xc_cpu_generated.hpp"

#if defined(GENERATIVEQC_TEST_HOOKS)
namespace {
// One-shot injection uses the real status mapper without poisoning the CUDA
// context, allowing the public API to verify explicit recovery and seed reuse.
thread_local bool fail_next_ks_runtime = false;
}  // namespace
extern "C" void ks_cuda_fail_next_runtime_for_test_v1() { fail_next_ks_runtime = true; }
#endif

namespace generativeqc::dft {
namespace {
using namespace scf::cuda_execution;
constexpr unsigned kMaximumFinalCorrections = 4;
constexpr unsigned kCudaKsChunkCapacity = 2;

// Bind native mathematical implementations once, outside the SCF hot path.
// Execution uses the manifest-owned semilocal code rather than name branches.
using HostRksXc = XcIntegral (*)(const AoBasis&, const MolecularGrid&, const std::vector<double>&,
                                 std::size_t, XcDensitySource);
using HostUksXc = SpinXcIntegral (*)(const AoBasis&, const MolecularGrid&,
                                     const std::vector<double>&, const std::vector<double>&,
                                     std::size_t);
struct HostXcExecutors {
  HostRksXc rks;
  HostUksXc uks;
};
const auto kHostXcExecutors = [] {
  std::array<HostXcExecutors, kSemilocalFamilyMetadata.size()> result{};
  result[semilocal_family_code(SemilocalFamily::Lda)] = {integrate_lda_xc_pw_rks,
                                                         integrate_lda_xc_pw_uks};
  result[semilocal_family_code(SemilocalFamily::Pbe)] = {integrate_pbe_rks_with_tail,
                                                         integrate_pbe_uks};
  result[semilocal_family_code(SemilocalFamily::R2scan)] = {integrate_r2scan_rks,
                                                            integrate_r2scan_uks};
  result[semilocal_family_code(SemilocalFamily::B3lyp)] = {integrate_b3lyp_rks,
                                                           integrate_b3lyp_uks};
  result[semilocal_family_code(SemilocalFamily::Wb97mv)] = {integrate_wb97mv_rks,
                                                            integrate_wb97mv_uks};
  return result;
}();

constexpr bool curated_cuda_ks_functional(std::uint32_t functional) noexcept {
  const auto* metadata = semilocal_family_metadata_from_code(functional);
  return metadata && metadata->cuda_ks;
}

constexpr bool is_semilocal_family(std::uint32_t functional, SemilocalFamily family) noexcept {
  return functional == semilocal_family_code(family);
}

void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}
void check(generativeqc_status status, const std::string& detail) {
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == GENERATIVEQC_STATUS_CUDA_ERROR || status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE)
    throw generativeqc::Error(status, detail);
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
template <class Function>
void run_resident_nonlocal_cuda(Function function) {
  try {
    function();
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::invalid_argument&) {
    throw;
  } catch (const std::overflow_error&) {
    throw;
  } catch (const generativeqc::Error&) {
    throw;
  } catch (const std::runtime_error& error) {
    // The resident VV10 seam uses runtime::cuda_resource_check internally.
    // Translate its untyped CUDA runtime failures at the KS owner boundary;
    // allocation failures already arrive as std::bad_alloc.
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, error.what());
  }
}
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("CUDA KS storage overflow");
  return a * b;
}
std::size_t sum(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::overflow_error("CUDA KS storage overflow");
  return a + b;
}
/** Numeric arena view shared by allocation and metadata-only planning. */
struct KsStateStorage {
  scf::cuda_df::ValidationPartial* final_validation_partial{};
  double *hcore{}, *overlap{}, *x{}, *j{}, *exchange{}, *range_exchange{}, *density{}, *proposal{},
      *warm{}, *warm_orbitals{}, *fock{}, *residual{}, *tmp1{}, *tmp2{}, *effective{},
      *masked_matrix_output{}, *fock_history{}, *residual_history{}, *gram{}, *raw_gram{},
      *weights{}, *eigenvalues{}, *final_coefficients{}, *final_eigenvalues{}, *cold_seed{},
      *incremental_anchor_density{}, *incremental_delta_density{}, *incremental_anchor_j{},
      *incremental_anchor_exchange{}, *incremental_anchor_range_exchange{},
      *incremental_max_abs_delta_density{};
  std::int32_t* occupied{};
  std::uint8_t *enabled{}, *spin_enabled{};
  std::uint32_t *history_count{}, *history_head{};
  int *solver_info{}, *final_solver_info{}, *jk_error{}, *range_jk_error{}, *staged_xc_error{};
  double* staged_xc_totals{};
  std::uint8_t *final_spin_enabled{}, *final_enabled{};
  cuda_ks_detail::Control* control{};
  cuda_ks_detail::Scalars* scalar_records{};
  /** The dry run and actual partition share one checked, typed layout. All
   * persistent and phase-local numeric buffers are explicitly charged. */
  std::size_t partition(std::size_t n, unsigned spins, unsigned history, bool exact_exchange,
                        bool range_correction, bool incremental_direct_jk,
                        bool incremental_diis_gram, void* storage) {
    const auto matrix = product(n, n), elements = product(spins, matrix);
    std::size_t bytes = 0;
    const auto reserve = [&](auto*& pointer, std::size_t count) {
      using T = std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>;
      const auto remainder = bytes % alignof(T);
      if (remainder) bytes = sum(bytes, alignof(T) - remainder);
      pointer = storage ? reinterpret_cast<T*>(static_cast<char*>(storage) + bytes) : nullptr;
      bytes = sum(bytes, product(count, sizeof(T)));
    };
    for (auto** pointer : {&hcore, &overlap, &x, &j}) reserve(*pointer, matrix);
    if (exact_exchange)
      reserve(exchange, elements);
    else
      exchange = nullptr;
    if (range_correction)
      reserve(range_exchange, elements);
    else
      range_exchange = nullptr;
    if (incremental_direct_jk) {
      reserve(incremental_anchor_density, elements);
      reserve(incremental_delta_density, elements);
      reserve(incremental_anchor_j, matrix);
      if (exact_exchange)
        reserve(incremental_anchor_exchange, elements);
      else
        incremental_anchor_exchange = nullptr;
      if (range_correction)
        reserve(incremental_anchor_range_exchange, elements);
      else
        incremental_anchor_range_exchange = nullptr;
      reserve(incremental_max_abs_delta_density, 1);
    } else {
      incremental_anchor_density = incremental_delta_density = incremental_anchor_j = nullptr;
      incremental_anchor_exchange = incremental_anchor_range_exchange = nullptr;
      incremental_max_abs_delta_density = nullptr;
    }
    for (auto** pointer :
         {&density, &proposal, &warm, &warm_orbitals, &fock, &residual, &tmp1, &tmp2, &effective})
      reserve(*pointer, elements);
    // Exact numeric capacity is shared with the public dry-run inventory.
    reserve(masked_matrix_output,
            MatrixLibraryOwner::provider_allowance(static_cast<int>(n)) ? elements : 0);
    // The generated cold guess stays resident across cold retries. It cannot
    // alias proposal/warm storage, which changes during every SCF trajectory.
    reserve(cold_seed, elements);
    reserve(fock_history, product(history, elements));
    reserve(residual_history, product(history, elements));
    reserve(gram, product(history + 1, history + 1));
    if (incremental_diis_gram)
      reserve(raw_gram, product(history, history));
    else
      raw_gram = nullptr;
    reserve(weights, history + 1);
    reserve(eigenvalues, product(spins, n));
    reserve(final_coefficients, elements);
    reserve(final_eigenvalues, product(spins, n));
    reserve(occupied, spins);
    reserve(enabled, 1);
    reserve(spin_enabled, spins);
    reserve(history_count, 1);
    reserve(history_head, 1);
    reserve(solver_info, spins);
    reserve(final_solver_info, spins);
    reserve(jk_error, 1);
    if (range_correction)
      reserve(range_jk_error, 1);
    else
      range_jk_error = nullptr;
    reserve(staged_xc_totals, 3);
    reserve(staged_xc_error, 1);
    reserve(final_spin_enabled, spins);
    reserve(final_enabled, 1);
    reserve(control, 1);
    reserve(scalar_records, kCudaKsChunkCapacity);
    // Reuse four existing iteration matrices after final-frame selection. Only
    // bounded scalar reductions need additional, publicly charged storage.
    // Tiny owners use their existing native products, not a new BLAS allowance.
    reserve(final_validation_partial, resident_final_validation_partial_count(n));
    return bytes;
  }
};

std::uint64_t next_ks_owner() noexcept {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  while (value && value != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
  return 0;
}
}  // namespace

std::size_t cuda_ks_state_bytes(std::size_t n, unsigned spins, unsigned history,
                                bool exact_exchange, bool range_correction,
                                bool incremental_direct_jk) {
  if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (spins != 1 && spins != 2) || history > 64)
    throw std::invalid_argument("invalid CUDA KS resource shape");
  KsStateStorage layout;
  return sum(
      layout.partition(
          n, spins, std::max(1U, history), exact_exchange, range_correction, incremental_direct_jk,
          history >= 2 && scf::cuda_execution::incremental_diis_gram_requested(), nullptr),
      n <= kSmallEigensolverLimit ? 0 : scf::ordinary_eigensolver_workspace_allowance(n));
}

struct CudaKsPlan::Impl : KsStateStorage {
  const scf::PreparedFockPlan& provider;
  const scf::PreparedFockPlan* range_provider{};
  const AoBasis& basis;
  const MolecularGrid& grid;
  const MolecularGrid& nonlocal_grid;
  scf::ScfOptions options;
  scf::PreparedCudaFockBinding fock_binding{};
  scf::PreparedCudaFockBinding range_fock_binding{};
  scf::PreparedCudaOccupiedFockBinding occupied_fock_binding{};
  MatrixLibraryOwner matrix_products;
  runtime::OwnedCudaEvent range_input_ready, range_output_ready;
  cudaStream_t stream{};
  int device{};
  std::size_t n{}, matrix{}, elements{};
  unsigned spins{}, history{};
  bool incremental_diis_gram{};
  bool ordered_diis_gram{};
  std::array<std::size_t, 2> occupations{};
  std::vector<double> host_xc_density, host_xc_alpha, host_xc_beta, host_xc_potential;
  // Async H2D copies retain these controls through the existing stream drain.
  std::array<double, 3> host_xc_totals{};
  int host_xc_error{};
  std::array<std::int32_t, 2> host_spin_counts{};
  std::array<std::uint8_t, 2> host_selected{}, host_all_spins{1, 1};
  std::uint8_t host_one{1};
  GridSpec grid_spec;
  CudaXcLayout xc_layout;
  CudaKsResources resource;
  CudaKsTransfers movement;
  void *arena{}, *xc_arena{}, *nonlocal_arena{}, *nonlocal_xc_arena{};
  std::size_t ks_arena_bytes{}, nonlocal_arena_bytes{};
  double *nonlocal_raw_density{}, *nonlocal_raw_gradient{}, *nonlocal_effective_weights{},
      *nonlocal_effective_density{}, *nonlocal_effective_gradient{}, *nonlocal_vrho{},
      *nonlocal_vsigma{}, *nonlocal_workspace{};
  int *nonlocal_domain_error{}, *nonlocal_pair_error{};
  double* nonlocal_energy{};
  int* nonlocal_cache_mismatch{};
  std::uint64_t* nonlocal_cache_counters{};
  bool nonlocal_cache_enabled{}, nonlocal_cache_ready{}, nonlocal_cache_mixed{};
  std::string nonlocal_cache_schedule;
  nlc::Vv10CudaDeviceLayout nonlocal_layout{};
  std::unique_ptr<CudaXcPlan> xc;
  std::unique_ptr<CudaXcPlan> nonlocal_xc;
  CudaXcLayout nonlocal_xc_layout;
  CudaXcAoSelectionWork prepared_ao_work;
  std::uint64_t initial_xc_evaluations{};
  std::unique_ptr<OrdinaryStreamEigensolver> eigensolver;
  scf::ScfResult output;
  bool is_active{}, is_pending{}, is_failed{}, warm_ready{}, warm_orbitals_ready{}, started{};
  bool warm_updates{true}, device_chunk_mode{};
  bool stabilize_occupations{}, final_closure{}, has_exchange{}, has_range_correction{};
  bool fitted_coulomb{}, fitted_exchange{}, occupied_fitted_factor_ready{};
  // Only this immutable owner can create a factor-backed warm seed. A separate
  // storage tag survives frozen replays and provider scratch/force writes.
  bool warm_factor_storage{};
  bool warm_orbitals_are_ao{};
  scf::PreparedCudaOccupiedFockBinding warm_factor_binding{};
  std::uint64_t warm_density_generation{}, warm_factor_generation{}, warm_factor_epoch{},
      warm_factor_owner{};
  std::size_t warm_factor_rank{};
  bool pending_fitted_occupied{}, final_fitted_projection_ready{};
  std::uint64_t pending_fitted_projection_scratch_generation{};
  std::uint64_t final_fitted_projection_scratch_generation{};
  runtime::ExecutionPrecisionSchedule precision_schedule{};
  MixedPrecisionWarmup mixed_warmup{};
  bool experimental_mixed_point{}, experimental_mixed_ao{}, pending_mixed_point{};
  bool experimental_mixed_history{};
  bool experimental_mixed_contractions{};
  bool pending_mixed_contractions{};
  bool budget_fock{}, budget_stationarity{}, pending_mixed_exchange{}, pending_mixed_resident{};
  bool pending_mixed_range_exchange{}, pending_mixed_range_resident{}, pending_mixed_nonlocal{};
  scf::IncrementalDirectJkPolicy incremental_direct_jk_policy{};
  bool incremental_direct_jk{}, incremental_anchored{}, pending_incremental_delta{},
      incremental_energy_refinement{};
  bool incremental_proposals_requested{true}, incremental_proposals_active{};
  bool incremental_coarse_screening_requested{true};
  bool fast_anchor_eligible{}, fast_anchor_active{}, fast_anchor_ready{}, pending_fast_delta{};
  void* fast_anchor_arena{};
  double *fast_anchor_density{}, *fast_delta_density{}, *fast_anchor_j{}, *fast_anchor_exchange{},
      *fast_anchor_range_exchange{}, *fast_delta_maximum{};
  unsigned incremental_energy_full_builds{};
  unsigned incremental_delta_updates{};
  double host_incremental_max_abs_delta_density{};
  bool strict_refinement{}, pending_mixed_coulomb{}, pending_mixed_density{},
      mixed_precision_executed{}, device_nonlocal{};
  // Diagnostic-only native seed experiment. The semilocal/RSH stage shares
  // this owner's basis, grid, FP64 kernels and budget; it can never return a
  // successful target state. Full VV10 starts a fresh energy/DIIS epoch.
  bool nonlocal_seed_stage{}, pending_nonlocal_seed{}, nonlocal_seed_trace{};
  unsigned nonlocal_seed_iterations{}, nonlocal_exact_iterations{};
  // Experimental nested solve: cheap inner iterations retain the last exact
  // VV10 potential. Every accepted endpoint still uses the unmodified full
  // target operator, consecutive physical energy checks and original budget.
  bool nonlocal_outer_active{}, nonlocal_frozen_stage{}, pending_nonlocal_frozen{};
  double nonlocal_inner_residual_tolerance{1e-9};
  unsigned nonlocal_inner_iterations{}, nonlocal_exact_epoch_iterations{};
  double exchange_coefficient{}, range_exchange_coefficient{};
  std::optional<scf::ResolvedFockBuild> range_correction;
  nlc::Vv10Plan* nonlocal_correlation{};
  nlc::Vv10DensityDomain nonlocal_domain{nlc::Vv10DensityDomain::StrictPositive};
  unsigned final_corrections{}, refinement_iterations{};
  std::uint32_t functional{semilocal_family_code(SemilocalFamily::Lda)};
  bool final_state_ready{}, final_frame_ready{}, final_stationary_weights_ready{};
  // Eligible strict solves may replace the conservative AO-maximum margin
  // only with a successful complete canonical final-state proof.
  bool early_final_probe_requested{}, standard_convergence_model{};
  mutable std::optional<CudaKsFinalStateToken> cached_final_token;
  KsFinalStateDiagnostic final_validation_diagnostic;
  std::uint64_t owner{next_ks_owner()}, solve_epoch{}, generation{}, final_generation{};
  double previous_energy{std::numeric_limits<double>::infinity()};
  double previous_energy_correction{};
  double warm_energy{std::numeric_limits<double>::infinity()};
  double warm_energy_correction{};
  bool warm_energy_baseline{};
  unsigned pending_iterations{};
  std::array<std::uint64_t, kCudaKsChunkCapacity> pending_generations{};
  runtime::SolverRegionCudaExecutor solver_region_executor;
  runtime::CompiledExecutionRegion device_chunk_region;
  std::unique_ptr<scf::cuda_execution::DirectJkSourceLease> phase_source;
  bool phase_source_cold{};
  bool phase_source_range{};
  double range_warm_energy{std::numeric_limits<double>::infinity()};
  double range_warm_energy_correction{};
  std::string range_bucket_policy, range_capacity_policy, range_selection_policy;

  static std::string range_policy_value(const char* key) {
    const auto* value = std::getenv(key);
    return value ? value : "";
  }

  /** Reuse only this immutable geometry owner's audited radial source.
   * Policy changes and force-memory eviction fall back to the separately
   * rebased incumbent energy. No density, Fock or force evaluation is skipped. */
  bool retain_range_source_for_replay(bool use_warm) const {
    const auto replay = range_policy_value("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY");
    // Admission follows the next warm consumer, not the cold proposal's
    // precision schedule. Application-budget warm revalidation starts strict;
    // it may borrow a strict lease, but never relabel a mixed recurrence lease.
    const bool strict_warm = !precision_schedule.any_lower_precision() ||
                             (options.application_precision_budget[0] > 0.0 &&
                              (experimental_mixed_point || budget_fock));
    return use_warm && range_source_active() && std::isfinite(range_warm_energy) &&
           !phase_source->mixed_recurrence() && strict_warm && (replay.empty() || replay == "1") &&
           range_policy_value("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES") == range_selection_policy &&
           range_policy_value("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK") == range_bucket_policy &&
           range_policy_value("GENERATIVEQC_KS_RESIDENT_MAX_BYTES") == range_capacity_policy &&
           !incremental_direct_jk && configured_chunk_width() == 1 &&
           !runtime::active_device_resource_ledger;
  }

  bool range_source_active() const noexcept {
    return phase_source_range && phase_source && phase_source->active();
  }
  std::uint64_t phase_submitted_begin{};
  std::size_t phase_device_bytes{}, phase_host_bytes{};

  /** A successful SCF closes captured iteration borrowers, not immutable ERIs.
   * The final-state token still owns D/W/XC and the primary derivative source;
   * post-SCF value consumers can reuse this independently audited value lease. */
  void audit_phase_source() {
    if (!phase_source || !phase_source->active()) return;
    device_chunk_region.invalidate();
    solver_region_executor.invalidate();
    const auto actions = movement.submitted_iterations - phase_submitted_begin;
    check(phase_source->audit(), "CUDA KS phase source finite/census audit failed");
    movement.phase_resident_fock_actions += actions;
    phase_source->observe_completed(actions);
    phase_submitted_begin = movement.submitted_iterations;
  }

  void retire_phase_source(bool audit) {
    range_warm_energy = std::numeric_limits<double>::infinity();
    phase_source_range = false;
    if (!phase_source) return;
    auto retired = std::move(phase_source);
    resource.provider_device_bytes -= phase_device_bytes;
    resource.retained_host_numeric_bytes -= phase_host_bytes;
    phase_device_bytes = phase_host_bytes = 0;
    // A future graph qualification cannot retain the provider beyond this
    // phase, including failure and explicit restart exits.
    device_chunk_region.invalidate();
    solver_region_executor.invalidate();
    if (audit && retired->active()) {
      const auto actions = movement.submitted_iterations - phase_submitted_begin;
      check(retired->audit(), "CUDA KS phase source finite/census audit failed");
      movement.phase_resident_fock_actions += actions;
      retired->observe_completed(actions);
    }
  }

  void prepare_phase_source(bool cold, bool frozen_warm = false) {
    using namespace scf::cuda_execution;
    const bool mixed_source = budget_fock && !strict_refinement;
    if (phase_source && phase_source->active() && phase_source->mixed_recurrence() != mixed_source)
      retire_phase_source(true);
    if (experimental_mixed_point && phase_source && phase_source->active()) return;
    // Range residency has its own bounded automatic policy. Never rebuild a
    // quartic source for a retained one-step replay, or compose it with the
    // incremental or device-chunk solvers. Precision-bound leases are retired
    // before strict refinement rather than excluding mixed recurrence sources.
    const auto* selection =
        std::getenv(has_range_correction ? "GENERATIVEQC_KS_RANGE_RESIDENT_VALUES"
                                         : "GENERATIVEQC_KS_RESIDENT_VALUES");
    // Unset follows the same bounded cold policy as explicit auto; 0 restores
    // the incumbent schedule without preparing any optional source storage.
    const auto requested = direct_jk_source_selection(selection);
    if (requested == DirectJkSourceSelection::invalid)
      throw std::invalid_argument("invalid KS resident value selection (expected auto, 0 or 1)");
    if (requested == DirectJkSourceSelection::disabled) return;
    if (has_range_correction &&
        (frozen_warm || spins != 1 || !device_nonlocal || !nonlocal_correlation ||
         nonlocal_correlation->parameters().variant != nlc::Vv10Variant::vv10 ||
         configured_chunk_width() != 1 ||
         (precision_schedule.any_lower_precision() && !mixed_source && !strict_refinement)))
      return;
    const auto& strategy = provider.strategy();
    const auto& source_strategy = has_range_correction ? *range_correction : strategy;
    // Public ledgers have no reserved quartic allowance yet. A forced request
    // must not bypass their complete resource plan or alter precision/ranges.
    if (fitted_coulomb || fitted_exchange || incremental_direct_jk ||
        (precision_schedule.any_lower_precision() && !strict_refinement &&
         !experimental_mixed_point && !mixed_source) ||
        runtime::active_device_resource_ledger ||
        strategy.spec.coulomb.op != scf::FockOperator::FullRange ||
        strategy.spec.exchange.op != scf::FockOperator::FullRange)
      return;
    const auto& system = provider.system();
    const auto direct_nbf = molecule::cartesian_ao_count(system);
    unsigned maximum_angular = 0;
    for (const auto& shell : system.shells)
      maximum_angular = std::max(maximum_angular, shell.angular_momentum);
    // Only hybrid endpoints qualify the smaller reuse threshold. Pure J keeps
    // its incumbent admission rather than inheriting an unmeasured cost model.
    const auto auto_policy = has_exchange ? DirectJkSourceAutoPolicy::screened_hybrid_cold
                                          : DirectJkSourceAutoPolicy::screened_cold;
    const bool automatic_range =
        has_range_correction && requested == DirectJkSourceSelection::automatic;
    // A bounded native initialization has repeated density-current range K
    // consumers even after a geometry move. Warm calls only retain an existing
    // audited source; they never construct a quartic one for a single Fock.
    // Complete endpoints qualify screened SPD sources on substantial grids.
    // Application-budget refinement publishes a strict lease for both its
    // remaining target work and same-owner warm consumers. The mixed lease is
    // retired first, so the optional ceiling and complete-resource bounds still
    // apply to one optional source. Ordinary AUTO keeps its streamed tail.
    const bool budget_range_refinement = strict_refinement && budget_fock && budget_stationarity;
    if (automatic_range &&
        (!nonlocal_outer_active || (strict_refinement && !budget_range_refinement) ||
         xc_layout.npoint < 65536 || maximum_angular > 2))
      return;
    const bool source_cold_phase = cold || (automatic_range && nonlocal_outer_active);
    const auto buckets = direct_jk_resident_bucket_policy(
        std::getenv("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK"),
        std::getenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES"),
        automatic_range ? 16ULL << 30
                        : direct_jk_source_value_ceiling(requested, auto_policy, maximum_angular));
    if (!buckets.valid)
      throw std::invalid_argument("invalid KS resident bucket mask or value ceiling");
    const auto generic_fock =
        maximum_angular == 3 && scf::cuda_direct_jk_value_census_available(
                                    provider.cuda_direct_source(), source_strategy.spec);
    // Pure J already has a generated strict consumer (and may own faster MD-J).
    // A quartic cache, especially an FP32 cache rebuilt in FP64 at handoff,
    // costs more than the saved J applications over a complete cold solve.
    // The short strict tail of a relaxed application-budget hybrid uses that same
    // generated provider. Rebuilding all FP32 source values in FP64 for two
    // closing Focks costs more than streaming those strict applications.
    // Mixed hybrid proposals, explicitly tighter stopping controls, uncovered
    // high-angular/range operators and forced requests keep their residency policy.
    if (requested == DirectJkSourceSelection::automatic && !has_range_correction &&
        (!has_exchange || !budget_fock || (strict_refinement && budget_stationarity)) &&
        maximum_angular <= 2 &&
        scf::cuda_direct_jk_generated_coulomb_available(provider.cuda_direct_source(),
                                                        source_strategy.spec))
      return;
    // Refuse warm/seeded or small candidates before cloning the provider or
    // sorting compact rows. A permitted iteration limit is not expected reuse.
    if (requested == DirectJkSourceSelection::automatic) {
      if (const auto* refusal = direct_jk_source_auto_refusal(
              molecule::ao_count(system), direct_nbf, options.max_iterations, maximum_angular,
              source_cold_phase, generic_fock, auto_policy, strategy.screening_tolerance)) {
        runtime::df_progress::Scope trace("cuda_ks_resident_values_skip", "host_decision");
        runtime::df_progress::Scope::label("selection", "auto");
        runtime::df_progress::Scope::label("admission", refusal);
        trace.finish("refused");
        return;
      }
    }
    // KS/XC/grid/DIIS/eigensolver and prepared-provider storage is admitted
    // before optional values. Include host setup, not just free device VRAM.
    auto required = sum(resource.state_device_bytes, resource.xc_device_bytes);
    required = sum(required, resource.grid_device_bytes);
    required = sum(required, resource.provider_device_bytes);
    required = sum(required, resource.retained_host_numeric_bytes);
    required = sum(required, provider.cpu_observation_capacity());
    required = sum(required, provider.diagnostic().direct.host_preparation_bytes);
    required = sum(required, runtime::vector_capacities(basis.packed, grid.points(), grid.weights(),
                                                        grid.owners()));
    phase_source = std::make_unique<DirectJkSourceLease>(
        stream,
        has_range_correction ? source_strategy.spec.exchange.op : scf::FockOperator::FullRange,
        has_range_correction ? source_strategy.spec.exchange.omega : 0.0, mixed_source);
    check(phase_source->prepare(
              device, {system}, options.max_iterations, direct_nbf, required,
              options.reference_memory_budget_bytes ? options.reference_memory_budget_bytes
                                                    : std::numeric_limits<std::size_t>::max(),
              selection,
              has_range_correction ? "cuda_ks_range_resident_values_prepare"
                                   : "cuda_ks_resident_values_prepare",
              source_cold_phase, generic_fock, source_strategy.screening_tolerance, buckets.ceiling,
              false, auto_policy, buckets.mask, spins,
              automatic_range && !std::getenv("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK")),
          phase_source->reason());
    movement.phase_resident_numeric_peak_bytes = std::max(
        movement.phase_resident_numeric_peak_bytes, sum(required, phase_source->capacity_bytes()));
    // Publication can refuse after ERIs have completed. Include that discarded
    // setup in endpoint work rather than counting only admitted cache reuse.
    movement.phase_resident_values_built += phase_source->completed_value_count();
    record_precision_operator(
        scf::PrecisionOperatorKind::CoulombRecurrence,
        mixed_source ? scf::PrecisionDtype::Fp32 : scf::PrecisionDtype::Fp64,
        mixed_source ? scf::PrecisionArithmeticMode::Mixed : scf::PrecisionArithmeticMode::Strict,
        phase_source->completed_value_count(),
        mixed_source ? scf::PrecisionDtype::Fp32 : scf::PrecisionDtype::Fp64);
    if (!phase_source->active()) {
      phase_source.reset();
      return;
    }
    phase_source_range = has_range_correction;
    if (phase_source_range) {
      range_selection_policy = range_policy_value("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES");
      range_bucket_policy = range_policy_value("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK");
      range_capacity_policy = range_policy_value("GENERATIVEQC_KS_RESIDENT_MAX_BYTES");
    }
    phase_device_bytes = phase_source->device_bytes();
    phase_host_bytes = phase_source->capacity_bytes() - phase_device_bytes;
    resource.provider_device_bytes = sum(resource.provider_device_bytes, phase_device_bytes);
    resource.retained_host_numeric_bytes =
        sum(resource.retained_host_numeric_bytes, phase_host_bytes);
    movement.phase_resident_value_bytes =
        std::max(movement.phase_resident_value_bytes, phase_source->value_bytes());
    phase_submitted_begin = movement.submitted_iterations;
  }

  generativeqc_status enqueue_primary_jk(const double* alpha, const double* beta, bool mixed,
                                         std::string& detail, bool use_source = true) {
    const bool resident_source = use_source && !phase_source_range && phase_source &&
                                 phase_source->active() &&
                                 mixed == phase_source->mixed_recurrence();
    pending_mixed_resident = mixed && resident_source;
    runtime::cuda_trace::TraceOperation trace(
        has_exchange ? "cuda_ks_exact_jk_shared" : "cuda_ks_exact_j", stream,
        {1, n, 0, resident_source, false});
    if (resident_source) {
      // Source construction already recorded its completed recurrences once.
      // A resident F[D] executes FP64 products/scatter and no new recurrences.
      if (mixed)
        check(cudaMemsetAsync(mixed_coulomb_work_counter(), 0, sizeof(std::uint64_t), stream));
      return phase_source->enqueue(provider.strategy().spec, alpha, j,
                                   has_exchange ? exchange : nullptr, jk_error,
                                   provider.strategy().screening_tolerance, detail, beta,
                                   has_exchange && spins == 2 ? exchange + matrix : nullptr);
    }
    // Intermediate delta operators only propose densities. Keep the prepared
    // target's cutoff immutable; full builds, warm replay and final proof use
    // the ordinary facade. Unsupported producers retain the original action.
    const double screening = ks_incremental_proposal_screening(
        incremental_proposals_active && incremental_coarse_screening_requested &&
            pending_incremental_delta && !mixed,
        output.incremental_direct_jk.delta_builds, provider.strategy().screening_tolerance);
    if (screening > provider.strategy().screening_tolerance) {
      const auto status = scf::enqueue_prepared_cuda_fock_proposal(
          provider, alpha, matrix, j, has_exchange ? exchange : nullptr, jk_error, screening,
          detail);
      if (status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
        runtime::cuda_trace::trace_counter("incremental_coarse_screening", 1);
        return status;
      }
    }
    return scf::enqueue_prepared_cuda_fock(
        provider, alpha, beta, matrix, j, exchange,
        has_exchange && spins == 2 ? exchange + matrix : nullptr, jk_error, mixed, detail,
        mixed ? mixed_coulomb_work_counter() : nullptr, mixed && pending_mixed_exchange);
  }

  /** Standard convergence requires a complete resident physical proof.
   * Incremental solves must first establish the separate consecutive full-Fock
   * energy history; reconstructed delta Focks cannot publish a final state.
   * Partial/nonlocal/mixed/ECP owners retain their independent policy. */
  bool standard_convergence_active() const noexcept {
    return standard_convergence_model &&
           scf::standard_scf_convergence_requested(options.energy_tolerance,
                                                   options.density_tolerance) &&
           spins == 1 && !budget_fock && !has_range_correction && !nonlocal_correlation &&
           !precision_schedule.any_lower_precision() && configured_chunk_width() == 1 &&
           final_validation_partial && matrix_products.library_enabled() &&
           provider.system().ecp_terms.empty() &&
           std::none_of(provider.system().atoms.begin(), provider.system().atoms.end(),
                        [](const auto& atom) { return atom.ecp_core != 0; });
  }

  double physical_stationarity_tolerance() const noexcept {
    if (standard_convergence_active())
      return scf::scf_stationarity_tolerance(options.energy_tolerance, options.density_tolerance,
                                             1e-9);
    return std::min(budget_stationarity ? 1e-4 : 1e-9, options.density_tolerance);
  }

  void multiply_matrix(const double* left, bool transpose_left, const double* right,
                       const std::uint8_t* active, double* output) {
    check(launch_matrix_product(matrix_products.view(masked_matrix_output, elements), 1,
                                static_cast<int>(n), left, transpose_left, right, active, output),
          "CUDA KS matrix product failed");
  }

  void multiply_spin(unsigned spin_count, const double* left, bool left_is_spin,
                     bool transpose_left, const double* right, bool right_is_spin,
                     const std::uint8_t* active, double* output) {
    check(launch_spin_matrix_product(matrix_products.view(masked_matrix_output, elements), 1,
                                     static_cast<int>(spin_count), static_cast<int>(n), left,
                                     left_is_spin, transpose_left, right, right_is_spin, active,
                                     output),
          "CUDA KS spin matrix product failed");
  }

  void build_density(const double* orbital_frame, const double* energies,
                     const std::uint8_t* active, double* destination) {
    // tmp2 is dead after back-transformation. Signed force reconstruction runs
    // after the final-frame solve/proof, when effective is also dead scratch.
    check(launch_structured_density(matrix_products.view(masked_matrix_output, elements), 1,
                                    static_cast<int>(spins), static_cast<int>(n), occupied,
                                    orbital_frame, energies, active, destination, tmp2, elements,
                                    energies ? effective : nullptr, energies ? elements : 0),
          "CUDA KS structured density failed");
  }

  void retain_final_fitted_projection() {
    final_fitted_projection_ready = false;
    if (!pending_fitted_occupied || spins != 1 || !warm_orbitals_ready || !occupations[0]) return;
    const auto projection =
        scf::prepared_cuda_occupied_projection_binding(provider, occupations[0]);
    if (!projection || projection.device_id != device || projection.stream != stream ||
        projection.source_identity != occupied_fock_binding.source_identity ||
        projection.scratch_generation != pending_fitted_projection_scratch_generation ||
        projection.nbf != n)
      return;
    // The candidate preserves the original AO factor that generated this D.
    // Keep the incumbent orthonormal-frame handoff unchanged otherwise. The
    // representation tag, not the current environment, selects the conversion.
    if (warm_orbitals_are_ao)
      check(cudaMemcpyAsync(proposal, warm_orbitals, matrix * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream));
    else
      multiply_spin(1, x, false, false, warm_orbitals, true, final_enabled, proposal);
    final_fitted_projection_scratch_generation = projection.scratch_generation;
    final_fitted_projection_ready = true;
    ++movement.fitted_final_projection_leases;
  }

  std::uint64_t* mixed_coulomb_work_counter() noexcept {
    static_assert(kCudaKsChunkCapacity >= 2);
    static_assert(sizeof(cuda_ks_detail::Scalars) >= 4 * sizeof(std::uint64_t));
    static_assert(alignof(cuda_ks_detail::Scalars) >= alignof(std::uint64_t));
    // AUTO is deliberately excluded from device_chunk_mode. The second scalar
    // record is therefore dead for the whole AUTO trajectory and can hold the
    // provider-owned recurrence counter without growing the numeric arena.
    return reinterpret_cast<std::uint64_t*>(scalar_records + 1);
  }

  bool complete_precision_inventory_domain() const noexcept {
    // Version-1 detailed census covers the ordinary host-controlled CUDA-KS
    // schedule with device-resident semilocal XC and, when present, the
    // device-resident nonlocal owner. Device chunks have a separate replay owner
    // and host-unfused XC/nonlocal work has arithmetic outside this census.
    return !device_chunk_mode && (!nonlocal_correlation || device_nonlocal) &&
           options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused;
  }

  void record_precision_operator(scf::PrecisionOperatorKind kind, scf::PrecisionDtype compute,
                                 scf::PrecisionArithmeticMode arithmetic_mode,
                                 std::uint64_t count = 1,
                                 scf::PrecisionDtype storage = scf::PrecisionDtype::Fp64,
                                 scf::PrecisionDtype accumulation = scf::PrecisionDtype::Fp64,
                                 scf::PrecisionDtype reduction = scf::PrecisionDtype::Fp64) {
    if (!count) return;
    auto& operators = output.precision_work.operators;
    const scf::PrecisionOperatorRecord signature{kind,      storage,         compute, accumulation,
                                                 reduction, arithmetic_mode, 0};
    auto found = std::find_if(operators.begin(), operators.end(), [&](const auto& item) {
      return item.kind == signature.kind && item.storage == signature.storage &&
             item.compute == signature.compute && item.accumulation == signature.accumulation &&
             item.reduction == signature.reduction &&
             item.arithmetic_mode == signature.arithmetic_mode;
    });
    if (found == operators.end()) {
      auto record = signature;
      record.count = count;
      operators.push_back(record);
      return;
    }
    if (count > std::numeric_limits<std::uint64_t>::max() - found->count)
      throw std::overflow_error("CUDA KS precision operator census overflow");
    found->count += count;
  }

  void record_fock_precision_work(std::uint64_t mixed_coulomb_recurrences,
                                  std::uint64_t mixed_range_recurrences,
                                  std::array<std::uint64_t, 2> nonlocal_arithmetic) {
    auto& work = output.precision_work;
    const bool mixed = pending_mixed_coulomb || pending_mixed_density || pending_mixed_point ||
                       pending_mixed_contractions || pending_mixed_range_resident ||
                       mixed_range_recurrences || nonlocal_arithmetic[0];
    const auto phase = mixed ? scf::PrecisionWorkPhase::Scf
                             : (mixed_precision_executed ? scf::PrecisionWorkPhase::Refinement
                                                         : scf::PrecisionWorkPhase::Scf);
    work.events.push_back(
        {mixed ? scf::PrecisionWorkEventKind::MixedFock : scf::PrecisionWorkEventKind::StrictFock,
         phase, static_cast<std::uint64_t>(work.events.size()), output.iterations, owner,
         solve_epoch, generation});

    if (complete_precision_inventory_domain()) {
      const auto j_mode = pending_mixed_coulomb ? scf::PrecisionArithmeticMode::Mixed
                                                : scf::PrecisionArithmeticMode::Strict;
      const auto j_compute =
          pending_mixed_coulomb ? scf::PrecisionDtype::Fp32 : scf::PrecisionDtype::Fp64;
      const auto source_storage =
          pending_mixed_resident ? scf::PrecisionDtype::Fp32 : scf::PrecisionDtype::Fp64;
      record_precision_operator(scf::PrecisionOperatorKind::CoulombJ, j_compute, j_mode, 1,
                                source_storage);
      record_precision_operator(
          scf::PrecisionOperatorKind::ExchangeK,
          pending_mixed_exchange ? scf::PrecisionDtype::Fp32 : scf::PrecisionDtype::Fp64,
          pending_mixed_exchange ? scf::PrecisionArithmeticMode::Mixed
                                 : scf::PrecisionArithmeticMode::Strict,
          pending_mixed_exchange && !mixed_coulomb_recurrences && !pending_mixed_resident
              ? 0
              : static_cast<std::uint64_t>(has_exchange),
          source_storage);
      if (has_range_correction)
        record_precision_operator(
            scf::PrecisionOperatorKind::ExchangeK,
            mixed_range_recurrences || pending_mixed_range_resident ? scf::PrecisionDtype::Fp32
                                                                    : scf::PrecisionDtype::Fp64,
            mixed_range_recurrences || pending_mixed_range_resident
                ? scf::PrecisionArithmeticMode::Mixed
                : scf::PrecisionArithmeticMode::Strict,
            1,
            pending_mixed_range_resident ? scf::PrecisionDtype::Fp32 : scf::PrecisionDtype::Fp64);
      record_precision_operator(
          scf::PrecisionOperatorKind::Xc,
          pending_mixed_point ? scf::PrecisionDtype::Fp32 : scf::PrecisionDtype::Fp64,
          pending_mixed_point ? scf::PrecisionArithmeticMode::Mixed
                              : scf::PrecisionArithmeticMode::Strict,
          1, scf::PrecisionDtype::Fp64,
          pending_mixed_point && experimental_mixed_contractions ? scf::PrecisionDtype::Fp32
                                                                 : scf::PrecisionDtype::Fp64,
          pending_mixed_point && experimental_mixed_contractions ? scf::PrecisionDtype::Fp32
                                                                 : scf::PrecisionDtype::Fp64);
      if (nonlocal_correlation) {
        if (nonlocal_arithmetic[0])
          record_precision_operator(scf::PrecisionOperatorKind::NonlocalCorrelation,
                                    scf::PrecisionDtype::Fp32, scf::PrecisionArithmeticMode::Mixed);
        if (!nonlocal_arithmetic[0] || nonlocal_arithmetic[1])
          record_precision_operator(scf::PrecisionOperatorKind::NonlocalCorrelation,
                                    scf::PrecisionDtype::Fp64,
                                    scf::PrecisionArithmeticMode::Strict);
      }
      record_precision_operator(scf::PrecisionOperatorKind::FockAssembly, scf::PrecisionDtype::Fp64,
                                scf::PrecisionArithmeticMode::Strict);
      record_precision_operator(scf::PrecisionOperatorKind::PhysicalResidual,
                                scf::PrecisionDtype::Fp64, scf::PrecisionArithmeticMode::Strict);
      const std::uint64_t strict_matrix_products = stabilize_occupations ? 9U : 7U;
      record_precision_operator(scf::PrecisionOperatorKind::MatrixProduct,
                                scf::PrecisionDtype::Fp64, scf::PrecisionArithmeticMode::Strict,
                                strict_matrix_products);
      if (pending_mixed_density)
        record_precision_operator(scf::PrecisionOperatorKind::MatrixProduct,
                                  scf::PrecisionDtype::Fp32, scf::PrecisionArithmeticMode::Mixed);
      if (pending_mixed_contractions) {
        record_precision_operator(scf::PrecisionOperatorKind::MatrixProduct,
                                  scf::PrecisionDtype::Fp32, scf::PrecisionArithmeticMode::Mixed, 1,
                                  scf::PrecisionDtype::Fp64, scf::PrecisionDtype::Fp32,
                                  scf::PrecisionDtype::Fp32);
        record_precision_operator(scf::PrecisionOperatorKind::MatrixProduct,
                                  scf::PrecisionDtype::Fp32, scf::PrecisionArithmeticMode::Mixed, 1,
                                  scf::PrecisionDtype::Fp64, scf::PrecisionDtype::Fp32);
      }
      if (!final_closure)
        record_precision_operator(scf::PrecisionOperatorKind::Diis, scf::PrecisionDtype::Fp64,
                                  scf::PrecisionArithmeticMode::Strict);
      if (stabilize_occupations)
        record_precision_operator(scf::PrecisionOperatorKind::OccupationStabilization,
                                  scf::PrecisionDtype::Fp64, scf::PrecisionArithmeticMode::Strict);
      record_precision_operator(scf::PrecisionOperatorKind::Eigensolver, scf::PrecisionDtype::Fp64,
                                scf::PrecisionArithmeticMode::Strict);
      record_precision_operator(scf::PrecisionOperatorKind::DensityBuild, scf::PrecisionDtype::Fp64,
                                scf::PrecisionArithmeticMode::Strict);
      record_precision_operator(scf::PrecisionOperatorKind::Diagnostics, scf::PrecisionDtype::Fp64,
                                scf::PrecisionArithmeticMode::Strict);
    }
    if (pending_mixed_coulomb && mixed_coulomb_recurrences)
      record_precision_operator(scf::PrecisionOperatorKind::CoulombRecurrence,
                                scf::PrecisionDtype::Fp32, scf::PrecisionArithmeticMode::Mixed,
                                mixed_coulomb_recurrences);
  }

  void record_precision_retry() {
    auto& work = output.precision_work;
    work.events.push_back({scf::PrecisionWorkEventKind::Retry, scf::PrecisionWorkPhase::Retry,
                           static_cast<std::uint64_t>(work.events.size()), output.iterations, owner,
                           solve_epoch, generation});
  }

  runtime::CompiledExecutionBinding device_chunk_binding() const {
    return {"cuda-ks-device-chunk-v1:" + std::to_string(n) + ":" + std::to_string(spins) + ":" +
                std::to_string(functional) + ":" + std::to_string(history) + ":" +
                std::to_string(xc_layout.tile_points),
            // The prepared facade owns provider lifetime and replay identity;
            // device chunks are admitted only for its direct-Fock binding.
            device, stream, arena,
            phase_source && phase_source->active() ? phase_source->source_identity()
                                                   : fock_binding.source_identity};
  }

  void invalidate_warm_orbitals() noexcept {
    if (warm_orbitals_ready) ++movement.warm_orbital_frame_invalidations;
    warm_orbitals_ready = false;
    warm_orbitals_are_ao = false;
  }

  void clear_warm_state() noexcept {
    range_warm_energy = std::numeric_limits<double>::infinity();
    warm_ready = false;
    warm_factor_storage = false;
    warm_factor_binding = {};
    warm_density_generation = warm_factor_generation = warm_factor_epoch = warm_factor_owner = 0;
    warm_factor_rank = 0;
    warm_energy = std::numeric_limits<double>::infinity();
    warm_energy_correction = 0.0;
    warm_energy_baseline = false;
    invalidate_warm_orbitals();
  }

  bool warm_factor_replay_enabled() const noexcept {
    // Default to exact-factor replay only within the existing qualification
    // guards. Explicit 0 and invalid values retain dense first-K rollback.
    const char* selection = std::getenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR");
    return !selection || std::strcmp(selection, "1") == 0;
  }

  bool qualified_warm_factor() const noexcept {
    const auto binding = scf::prepared_cuda_occupied_fock_binding(provider);
    return warm_ready && warm_factor_storage && fitted_exchange && spins == 1 &&
           !precision_schedule.any_lower_precision() && warm_factor_owner == owner &&
           warm_density_generation && warm_factor_generation == warm_density_generation &&
           warm_factor_epoch && warm_factor_epoch < solve_epoch && warm_factor_rank &&
           warm_factor_rank == occupations[0] && warm_factor_rank <= n && binding &&
           !binding.unrestricted && binding.nbf == n && binding.device_id == device &&
           binding.stream == stream &&
           binding.source_identity == warm_factor_binding.source_identity &&
           binding.nbf == warm_factor_binding.nbf &&
           binding.device_id == warm_factor_binding.device_id &&
           binding.stream == warm_factor_binding.stream && !warm_factor_binding.unrestricted;
  }

  void materialize_warm_density(double* target) {
    if (warm_factor_storage) {
      // The same kernel, occupation and original AO C produced the accepted D.
      // No eigensolve, PSD approximation or final-frame substitution occurs.
      launch_build_density_kernel(static_cast<unsigned>((matrix + 127) / 128), 128, 0, stream, 1, n,
                                  occupied, warm, final_enabled, target);
      check(cudaGetLastError());
    } else {
      check(cudaMemcpyAsync(target, warm, elements * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream));
    }
  }

  void restore_warm_seed() {
    materialize_warm_density(density);
    if (warm_factor_replay_enabled() && qualified_warm_factor()) {
      check(cudaMemcpyAsync(tmp1, warm, matrix * sizeof(double), cudaMemcpyDeviceToDevice, stream));
      check(cudaMemcpyAsync(warm_orbitals, warm, matrix * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream));
      warm_orbitals_ready = true;
      warm_orbitals_are_ao = true;
      occupied_fitted_factor_ready = true;
      ++movement.fitted_warm_factor_restores;
    } else if (fitted_exchange) {
      ++movement.fitted_warm_factor_fallbacks;
    }
  }

  void publish_warm_seed() {
    // The convergence controller owns its energy anchor and compensation;
    // changing the seed representation must not reinterpret that history.
    // A complete final projection lease proves full auxiliary metric rank and
    // admitted occupied storage. Borrow only its exact C, never its mutable U.
    // The PreparedFockPlan owns immutable geometry/basis/auxiliary/model data;
    // its source identity plus this owner/generation/occupation binds the cache.
    const bool factor = warm_factor_replay_enabled() && final_fitted_projection_ready &&
                        pending_fitted_occupied && warm_orbitals_ready && warm_orbitals_are_ao &&
                        spins == 1 && !precision_schedule.any_lower_precision();
    warm_ready = false;
    warm_factor_storage = false;
    check(cudaMemcpyAsync(warm, factor ? proposal : density, elements * sizeof(double),
                          cudaMemcpyDeviceToDevice, stream));
    warm_density_generation = generation;
    warm_factor_generation = factor ? generation : 0;
    warm_factor_epoch = factor ? solve_epoch : 0;
    warm_factor_owner = factor ? owner : 0;
    warm_factor_rank = factor ? occupations[0] : 0;
    warm_factor_binding = factor ? occupied_fock_binding : scf::PreparedCudaOccupiedFockBinding{};
    warm_factor_storage = factor;
    warm_ready = true;
    if (factor) ++movement.fitted_warm_factor_publications;
  }

  void audit_warm_seed_exchange() {
    // Diagnostic only: compare the first occupied K against the incumbent
    // dense route at exactly the same D. Dead iteration scratch holds dense
    // outputs; no retained device allocation or production oracle is added.
    runtime::cuda_trace::TraceRegion audit("warm_seed_matrix_audit", stream);
    std::vector<double> occupied_j(matrix), occupied_k(matrix), dense_j(matrix), dense_k(matrix);
    const auto bytes = matrix * sizeof(double);
    check(cudaMemcpyAsync(occupied_j.data(), j, bytes, cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(occupied_k.data(), exchange, bytes, cudaMemcpyDeviceToHost, stream));
    std::string detail;
    check(scf::enqueue_prepared_cuda_fock(provider, density, nullptr, matrix, effective, residual,
                                          nullptr, jk_error, false, detail),
          detail);
    check(cudaMemcpyAsync(dense_j.data(), effective, bytes, cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(dense_k.data(), residual, bytes, cudaMemcpyDeviceToHost, stream));
    int error = -1;
    check(cudaMemcpyAsync(&error, jk_error, sizeof(error), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    ++movement.synchronizations;
    movement.matrix_d2h_bytes += 4 * bytes;
    movement.scalar_d2h_bytes += sizeof(error);
    runtime::cuda_trace::trace_counter("explicit_synchronizations", 1);
    runtime::cuda_trace::trace_counter("warm_seed_audit_dense_k_builds", 1);
    runtime::cuda_trace::trace_counter("warm_seed_audit_host_numeric_bytes", 4 * bytes);
    double k_error = 0, fock_error = 0;
    for (std::size_t element = 0; element < matrix; ++element) {
      const double delta_j = occupied_j[element] - dense_j[element];
      const double delta_k = occupied_k[element] - dense_k[element];
      // Hcore, XC and any range correction are unchanged at this exact D.
      const double delta_fock = delta_j + exchange_coefficient * delta_k;
      if (!std::isfinite(delta_k) || !std::isfinite(delta_fock)) error = 1;
      k_error = std::max(k_error, std::abs(delta_k));
      fock_error = std::max(fock_error, std::abs(delta_fock));
    }
    if (error || k_error >= 2e-11 || fock_error >= 2e-11)
      throw std::runtime_error("CUDA KS warm-seed dense/occupied K/Fock audit failed");
    runtime::cuda_trace::trace_counter("warm_seed_audit_k_max_abs_e16_ceil",
                                       static_cast<std::uint64_t>(std::ceil(k_error * 1e16)));
    runtime::cuda_trace::trace_counter("warm_seed_audit_fock_max_abs_e16_ceil",
                                       static_cast<std::uint64_t>(std::ceil(fock_error * 1e16)));
    // The dense diagnostic can overwrite shared projection scratch. Replay
    // occupied K before publishing its lease, preserving the ordinary force
    // handoff even when this first physical build immediately converges.
    const scf::PreparedCudaOccupiedFockInput occupied{tmp1, nullptr, occupations[0], 0};
    check(scf::enqueue_prepared_cuda_occupied_fock(provider, density, nullptr, matrix, occupied, j,
                                                   exchange, nullptr, jk_error, detail),
          detail);
    runtime::cuda_trace::trace_counter("warm_seed_audit_extra_occupied_k_builds", 1);
  }

  void current_device() const {
    // Prepared owners select their bound device on every entry, as the common
    // Fock provider does. Another context may have changed this thread's device
    // between calls; borrowed XC views still enforce their own device identity.
    check(cudaSetDevice(device));
  }

  std::size_t partition_nonlocal(void* storage) {
    if (!device_nonlocal) {
      nonlocal_raw_density = nonlocal_raw_gradient = nonlocal_effective_weights =
          nonlocal_effective_density = nonlocal_effective_gradient = nonlocal_vrho =
              nonlocal_vsigma = nonlocal_workspace = nullptr;
      nonlocal_domain_error = nonlocal_pair_error = nullptr;
      return 0;
    }
    if (nonlocal_layout.workspace_bytes % sizeof(double))
      throw std::logic_error("resident VV10 workspace is not double-aligned");
    const auto points = nonlocal_grid.point_count();
    const auto workspace_doubles = nonlocal_layout.workspace_bytes / sizeof(double);
    const auto doubles =
        sum(product(11, points), sum(workspace_doubles, nonlocal_cache_enabled ? 7U : 2U));
    if (!storage) return product(doubles, sizeof(double));
    auto* cursor = static_cast<double*>(storage);
    auto take = [&](std::size_t count) {
      auto* out = cursor;
      cursor += count;
      return out;
    };
    nonlocal_raw_density = take(points);
    nonlocal_raw_gradient = take(product(3, points));
    nonlocal_effective_weights = take(points);
    nonlocal_effective_density = take(points);
    nonlocal_effective_gradient = take(product(3, points));
    nonlocal_vrho = take(points);
    nonlocal_vsigma = take(points);
    nonlocal_workspace = take(workspace_doubles);
    nonlocal_domain_error = reinterpret_cast<int*>(take(1));
    nonlocal_pair_error = reinterpret_cast<int*>(take(1));
    nonlocal_energy = nonlocal_workspace;
    if (nonlocal_cache_enabled) {
      // Retain scalar energy outside the dead pair scratch used by the key.
      // This fixed 40-byte optional lease is included in the owner bound.
      nonlocal_energy = take(1);
      nonlocal_cache_mismatch = reinterpret_cast<int*>(take(1));
      nonlocal_cache_counters = reinterpret_cast<std::uint64_t*>(take(3));
    }
    if (cursor != static_cast<double*>(storage) + doubles)
      throw std::logic_error("resident VV10 KS arena partition mismatch");
    return product(doubles, sizeof(double));
  }

  scf::reference::EigenResult seed_eigen(const scf::reference::Matrix& input,
                                         std::size_t dimension) {
    runtime::host_trace::Region trace("cuda_ks_seed_eigen", n);
    if (dimension != n || input.size() != matrix || is_active || is_pending)
      throw std::invalid_argument("CUDA KS seed eigen operation requires its idle AO owner");
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        const auto a = input[row * n + column], b = input[column * n + row];
        if (!std::isfinite(a) ||
            std::abs(a - b) > 1e-12 * std::max({1.0, std::abs(a), std::abs(b)}))
          throw std::invalid_argument("CUDA KS seed eigen input must be finite and symmetric");
      }
    }
    current_device();
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("CUDA KS seed admission requires an ordinary stream");

    // Every begin() discards DIIS history before submitting work. These two
    // charged history buffers are therefore dead while idle, unlike tmp1/tmp2
    // which can back a live stationary D/W lease. Keep all final coefficients,
    // eigenvalues, density, warm state and generation tokens untouched, even
    // when validation rejects the imported checkpoint.
    scf::reference::EigenResult frame;
    frame.values.resize(n);
    frame.vectors.resize(matrix);
    int info{};
    try {
      check(cudaMemcpyAsync(fock_history, input.data(), matrix * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      check(eigensolver->launch(1, fock_history, residual_history, eigenvalues, solver_info,
                                final_enabled),
            "CUDA KS seed eigensolver launch failed");
      check(cudaMemcpyAsync(frame.vectors.data(), fock_history, matrix * sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaMemcpyAsync(frame.values.data(), eigenvalues, n * sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaMemcpyAsync(&info, solver_info, sizeof(info), cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      // Input/output buffers belong to this synchronous call. Drain queued
      // copies before their host storage or a stack-backed status can expire.
      (void)cudaStreamSynchronize(stream);
      throw;
    }
    movement.setup_h2d_bytes += matrix * sizeof(double);
    movement.matrix_d2h_bytes += matrix * sizeof(double);
    movement.scalar_d2h_bytes += n * sizeof(double) + sizeof(info);
    ++movement.synchronizations;
    if (info < 0) throw std::invalid_argument("CUDA KS seed eigensolver rejected an argument");
    if (info > 0)
      throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                                "CUDA KS seed eigensolver did not converge");
    // The solver emits column-major orbitals; the common admission algebra
    // uses row-major C[ao, orbital]. Symmetric input needs no packing copy.
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t column = row + 1; column < n; ++column)
        std::swap(frame.vectors[row * n + column], frame.vectors[column * n + row]);
    // Large zero/two occupation clusters can contain provider-order inversions
    // at roundoff scale. Reorder pairs, then retain every original physical gate.
    (void)scf::solver::order_eigenpairs(frame.values, frame.vectors, n);
    scf::solver::EigenFrameDiagnostic diagnostic;
    std::string detail;
    if (!scf::solver::validate_eigen_frame(input, nullptr, frame.values, frame.vectors, n,
                                           diagnostic, detail))
      throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, detail);
    return frame;
  }

  std::vector<double> seed(const std::vector<double>* input) {
    using namespace scf::reference;
    if (!input) throw std::logic_error("CUDA cold guesses must use resident setup state");
    const auto& ints = provider.one_electron();
    if (options.strict_initial_density && input) {
      const std::vector<unsigned> counts =
          spins == 2
              ? std::vector<unsigned>{static_cast<unsigned>(occupations[0]),
                                      static_cast<unsigned>(occupations[1])}
              : std::vector<unsigned>{static_cast<unsigned>(provider.system().electron_count)};
      // RKS stores the spin-summed density, not the number of occupied orbitals.
      // begin() has not submitted an iteration yet. Strict seeded admission
      // uses the same charged GPU solver as checkpoint and MINAO construction.
      const scf::initial_guess::EigenOperation eigen = [this](const auto& matrix, const auto*,
                                                              const auto*, std::size_t dimension) {
        return seed_eigen(matrix, dimension);
      };
      scf::solver::validate_seed(ints.overlap, *input, n, counts, spins == 2 ? 1.0 : 2.0, eigen);
      return *input;
    }
    if (spins == 2) {
      const auto pair = scf::initial_guess::normalized_warm_uhf_density(ints, occupations[0],
                                                                        occupations[1], *input);
      return concatenate(pair.first, pair.second);
    }
    return scf::initial_guess::normalized_warm_density(provider.system(), ints, *input);
  }

  /** Construct X and the historical core guess with the ordinary GPU solver.
   * All working matrices borrow the existing arena before iteration begins;
   * only the immutable cold density adds retained storage. Host warm-input
   * normalization remains an explicit input-boundary operation, never a
   * reference eigen fallback. */
  void prepare_initial_state() {
    runtime::host_trace::Region trace("cuda_ks_initial_state", n);
    const auto multiply = [&](const double* a, bool transpose, const double* b, double* c) {
      multiply_matrix(a, transpose, b, final_enabled, c);
    };
    const auto solve = [&] {
      check(eigensolver->launch(1, tmp2, effective, eigenvalues, solver_info, final_enabled),
            "CUDA KS setup eigensolver launch failed");
    };
    const auto read_status = [&](bool metric) {
      // Stack-backed downloads are drained before any error escapes this scope.
      std::array<int, 2> status{};
      try {
        check(
            cudaMemcpyAsync(&status[0], solver_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
        if (metric)
          check(cudaMemcpyAsync(&status[1], jk_error, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
      } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
      }
      movement.scalar_d2h_bytes += sizeof(int) * (metric ? 2 : 1);
      ++movement.synchronizations;
      if (status[0]) throw std::runtime_error("CUDA KS setup eigensolver did not converge");
      if (status[1])
        throw std::runtime_error("overlap matrix is singular or failed its metric identity check");
    };
    check(
        cudaMemcpyAsync(tmp2, overlap, matrix * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    solve();
    check(cudaMemsetAsync(jk_error, 0, sizeof(int), stream));
    form_overlap_weights(stream, n, eigenvalues, jk_error);
    check(cudaGetLastError());
    form_weighted_projector(stream, n, 1, tmp2, eigenvalues, x);
    check(cudaGetLastError());
    multiply(overlap, false, x, tmp1);
    multiply(x, true, tmp1, residual);
    check_overlap_metric(stream, n, residual, jk_error);
    check(cudaGetLastError());
    read_status(true);  // Reject singular S before attempting a core solve.

    multiply(hcore, false, x, tmp1);
    multiply(x, true, tmp1, tmp2);
    solve();
    multiply(x, false, tmp2, tmp1);
    if (spins == 2) {
      check(cudaMemcpyAsync(tmp1 + matrix, tmp1, matrix * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream));
      // Reuse the common UHF frontier rotation, including beta=0/equal-spin
      // branches; the scientific seed policy is identical to HF and CPU KS.
      launch_mix_open_shell_guess_kernel(static_cast<unsigned>((n + 127) / 128), 128, 0, stream, 1,
                                         n, occupied, final_enabled, tmp1);
      check(cudaGetLastError());
    }
    form_occupation_weights(stream, n, spins, occupations[0], occupations[1], eigenvalues);
    check(cudaGetLastError());
    form_weighted_projector(stream, n, spins, tmp1, eigenvalues, cold_seed);
    check(cudaGetLastError());
    read_status(false);
  }

  Impl(const scf::PreparedFockPlan& plan, const AoBasis& basis, const MolecularGrid& grid,
       const scf::ScfOptions& control, std::uint32_t functional, std::size_t tile,
       const scf::ResolvedFockBuild* range, nlc::Vv10Plan* nonlocal, nlc::Vv10DensityDomain domain,
       CudaXcPreparationBudget xc_budget, const scf::PreparedFockPlan* separate_range_provider)
      : provider(plan),
        range_provider(separate_range_provider),
        basis(basis),
        grid(grid),
        nonlocal_grid(control.nonlocal_grid ? *control.nonlocal_grid : grid),
        options(control),
        grid_spec(grid.spec()),
        functional(functional),
        range_correction(range ? std::optional<scf::ResolvedFockBuild>(*range) : std::nullopt),
        nonlocal_correlation(nonlocal),
        nonlocal_domain(domain) {
    if (!curated_cuda_ks_functional(functional) && !generated::split_hybrid_registered(functional))
      throw std::invalid_argument("CUDA KS functional has no qualified device implementation");
    const auto& strategy = provider.strategy();
    scf::validate_resolved_fock_build(strategy);
    has_exchange = strategy.spec.exchange.present;
    exchange_coefficient = has_exchange ? strategy.spec.exchange.coefficient : 0.0;
    has_range_correction = range_correction.has_value();
    if (has_range_correction) {
      scf::validate_resolved_fock_build(*range_correction);
      range_exchange_coefficient = range_correction->spec.exchange.coefficient;
    }
    fitted_coulomb = strategy.spec.coulomb.approximation == scf::FockApproximation::DensityFitted;
    fitted_exchange = has_exchange &&
                      strategy.spec.exchange.approximation == scf::FockApproximation::DensityFitted;
    fock_binding = scf::prepared_cuda_fock_binding(provider);
    occupied_fock_binding = scf::prepared_cuda_occupied_fock_binding(provider);
    if (!owner || strategy.backend != scf::FockBackend::Cuda ||
        strategy.spec.derivative_order != 0 || !strategy.spec.coulomb.present ||
        strategy.spec.coulomb.coefficient != 1.0 ||
        (strategy.spec.coulomb.approximation != scf::FockApproximation::Exact && !fitted_coulomb) ||
        (has_exchange && (strategy.spec.exchange.op != scf::FockOperator::FullRange ||
                          (fitted_coulomb ? !fitted_exchange
                                          : strategy.spec.exchange.approximation !=
                                                scf::FockApproximation::Exact))) ||
        !fock_binding || (fitted_exchange && !occupied_fock_binding))
      throw std::invalid_argument(
          "CUDA KS requires one prepared Coulomb provider and matching full-range exchange");
    if (has_range_correction) {
      const auto& correction = *range_correction;
      const auto& spec = correction.spec;
      const bool correction_identity =
          fock_binding && correction.backend == scf::FockBackend::Cuda &&
          spec.spin == strategy.spec.spin && spec.derivative_order == 0 && !spec.coulomb.present &&
          spec.exchange.present && spec.exchange.approximation == scf::FockApproximation::Exact &&
          spec.exchange.op == scf::FockOperator::LongRange && spec.exchange.omega > 0.0 &&
          correction.screening_tolerance == strategy.screening_tolerance;
      if (!correction_identity)
        throw std::invalid_argument(
            "CUDA KS range correction must be one compatible exact long-range exchange term");
      if (fitted_coulomb) {
        if (!range_provider || range_provider->strategy() != correction ||
            !range_provider->matches_system(provider.system()))
          throw std::invalid_argument(
              "fitted CUDA RSH requires a separate prepared Direct range provider");
        range_fock_binding = scf::prepared_cuda_fock_binding(*range_provider);
        if (!range_fock_binding || range_fock_binding.device_id != fock_binding.device_id ||
            range_fock_binding.nbf != fock_binding.nbf)
          throw std::invalid_argument(
              "fitted CUDA RSH range provider is incompatible with the primary DF owner");
      } else if (range_provider) {
        throw std::invalid_argument(
            "Direct CUDA RSH must reuse its primary provider for the range correction");
      }
    } else if (range_provider) {
      throw std::invalid_argument("CUDA KS received an unused prepared range provider");
    }
    if (options.compute_forces || options.hooks || options.export_physical_reference ||
        options.xc_density_route != XcDensityRoute::DensityMatrix ||
        (options.precision_mode && *options.precision_mode != GENERATIVEQC_PRECISION_FP64 &&
         *options.precision_mode != GENERATIVEQC_PRECISION_AUTO))
      throw std::invalid_argument("CUDA KS received an unsupported execution policy");
    if (nonlocal_correlation) {
      const auto* family = semilocal_family_metadata_from_code(functional);
      if (!family || !family->native_nonlocal_correlation)
        throw std::invalid_argument("CUDA KS nonlocal composition has no native family capability");
      device_nonlocal =
          options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused;
      const auto expected_domain = family->molecular_nonlocal_domain
                                       ? nlc::Vv10DensityDomain::MolecularV1
                                       : nlc::Vv10DensityDomain::StrictPositive;
      if (device_nonlocal &&
          (!family->cuda_nonlocal_correlation || nonlocal_domain != expected_domain ||
           nonlocal_correlation->parameters().variant != nlc::Vv10Variant::vv10))
        throw std::invalid_argument(
            "device-resident CUDA nonlocal KS lacks a qualified family/domain capability");
      if (!device_nonlocal &&
          options.xc_execution_schedule != scf::ScfOptions::XcExecutionSchedule::HostUnfused)
        throw std::invalid_argument("unknown CUDA KS nonlocal execution schedule");
    }
    if (!options.max_iterations || !std::isfinite(options.energy_tolerance) ||
        !std::isfinite(options.density_tolerance) || options.energy_tolerance <= 0.0 ||
        options.density_tolerance <= 0.0 || options.diis_history > 64)
      throw std::invalid_argument("invalid CUDA KS convergence, DIIS or precision controls");
    if (!provider.matches_system(grid.system()) ||
        basis.packed != AoBasis(provider.system()).packed)
      throw std::invalid_argument(
          "CUDA KS refuses a stale geometry, basis, charge or spin binding");
    n = provider.one_electron().nbf;
    if (!n || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) || basis.nao != n)
      throw std::invalid_argument("invalid CUDA KS AO dimension");
    matrix = product(n, n);
    spins = strategy.spec.spin == scf::FockSpin::Unrestricted ? 2 : 1;
    elements = product(spins, matrix);
    const auto counts = scf::initial_guess::spin_occupations(provider.system());
    occupations = {counts.first, counts.second};
    if (!provider.system().electron_count || occupations[0] > n || occupations[1] > n ||
        (spins == 1 && (occupations[0] != occupations[1] || provider.system().multiplicity != 1)))
      throw std::invalid_argument("CUDA KS occupations do not match the spin/orbital space");
    // A valid overlap does not imply finite physical data: unlike two equal
    // H centers, coincident O/H centers can retain a nonsingular AO metric
    // while nuclear repulsion is infinite. Fail before staging a cold seed.
    const auto& integrals = provider.one_electron();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!finite(integrals.nuclear_repulsion) ||
        !std::all_of(integrals.overlap.begin(), integrals.overlap.end(), finite) ||
        !std::all_of(integrals.hcore.begin(), integrals.hcore.end(), finite))
      throw std::runtime_error("nonfinite CUDA KS one-electron or nuclear energy");
    history = std::max(1U, options.diis_history);
    incremental_diis_gram = history >= 2 && scf::cuda_execution::incremental_diis_gram_requested();
    ordered_diis_gram =
        incremental_diis_gram && scf::cuda_execution::ordered_incremental_diis_gram_requested();
    device = fock_binding.device_id;
    stream = fock_binding.stream;
    if (range_provider) {
      range_input_ready.create(device, cudaEventDisableTiming);
      range_output_ready.create(device, cudaEventDisableTiming);
    }
    if (nonlocal_correlation &&
        (nonlocal_correlation->backend() != GENERATIVEQC_BACKEND_CUDA ||
         nonlocal_correlation->device_id() != device ||
         nonlocal_correlation->resources().point_count != nonlocal_grid.point_count()))
      throw std::invalid_argument("CUDA KS nonlocal owner is incompatible with the grid or device");
    current_device();
    const auto resident_grid = grid.cuda_view();
    if (resident_grid && resident_grid.device != device)
      throw std::invalid_argument("CUDA KS resident grid belongs to a different device");
    const bool borrow_resident_grid = static_cast<bool>(resident_grid);
    xc_layout = cuda_xc_layout(basis, grid, functional, spins == 2, tile, CudaXcAoPrecision::Fp64,
                               options.semilocal_exchange_scale,
                               options.semilocal_correlation_scale, borrow_resident_grid);
    if (options.nonlocal_grid)
      nonlocal_xc_layout = cuda_xc_layout(
          basis, nonlocal_grid, semilocal_family_code(SemilocalFamily::Pbe), spins == 2, tile,
          CudaXcAoPrecision::Fp64, 0.0, 0.0, static_cast<bool>(nonlocal_grid.cuda_view()));
    const auto fock_precision = scf::cuda_fock_precision_capabilities(strategy);
    const bool application_precision = options.application_precision_budget[0] > 0.0;
    bool fitted_xc_budget = false;
    if (application_precision) {
      for (const auto* control :
           {"GENERATIVEQC_EXPERIMENTAL_PBE_POINT_FP32",
            "GENERATIVEQC_EXPERIMENTAL_PBE_POINT_KEEP_DIIS",
            "GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL", "GENERATIVEQC_KS_MIXED_MAX_ITERATIONS"})
        if (std::getenv(control))
          throw std::invalid_argument(
              "application precision cannot inherit experimental overrides");
      unsigned maximum_angular = 0;
      for (const auto& shell : provider.system().shells)
        maximum_angular = std::max(maximum_angular, shell.angular_momentum);
      // DF has no mixed J provider. A separate local pure-J qualification
      // admits only XC matrix products, with the existing bounded proposal and
      // strict physical closure. Never report a nonexistent FP32 DF recurrence.
      fitted_xc_budget =
          options.precision_mode == GENERATIVEQC_PRECISION_AUTO && fitted_coulomb &&
          !fitted_exchange && !has_exchange && !has_range_correction && !nonlocal_correlation &&
          grid_spec == default_grid_spec() &&
          (functional == semilocal_family_code(SemilocalFamily::Pbe) ||
           functional == semilocal_family_code(SemilocalFamily::R2scan)) &&
          options.semilocal_exchange_scale == 1.0 && options.semilocal_correlation_scale == 1.0 &&
          molecule::ao_count(provider.system()) >= 24 &&
          molecule::ao_count(provider.system()) <= 1200 && provider.system().atoms.size() >= 2 &&
          provider.system().atoms.size() <= 96 && maximum_angular <= 2 &&
          provider.system().ecp_terms.empty() &&
          admit_application_df_xc_budget(
              options.application_precision_budget,
              spins == 1 && options.xc_execution_schedule ==
                                scf::ScfOptions::XcExecutionSchedule::DeviceFused);
      budget_fock =
          options.precision_mode == GENERATIVEQC_PRECISION_AUTO &&
          (fitted_xc_budget ||
           admit_application_fock(
               options.application_precision_budget, molecule::ao_count(provider.system()),
               provider.system().atoms.size(), maximum_angular,
               spins == 1 && options.xc_execution_schedule ==
                                 scf::ScfOptions::XcExecutionSchedule::DeviceFused,
               provider.system().ecp_terms.empty(), fock_precision.coulomb));
      budget_stationarity =
          budget_fock && (options.energy_tolerance > 1e-10 || options.density_tolerance > 1e-8 ||
                          options.screening_tolerance > 1e-12);
      experimental_mixed_point =
          !budget_fock && options.precision_mode == GENERATIVEQC_PRECISION_AUTO &&
          admit_application_xc_warmup(
              options.application_precision_budget, molecule::ao_count(provider.system()),
              maximum_angular,
              spins == 1 && functional == static_cast<std::uint32_t>(SemilocalFamily::Pbe) &&
                  options.xc_execution_schedule ==
                      scf::ScfOptions::XcExecutionSchedule::DeviceFused &&
                  !nonlocal_correlation,
              provider.system().ecp_terms.empty(),
              !fitted_coulomb && !fitted_exchange && !has_range_correction);
      experimental_mixed_ao = experimental_mixed_point;
      experimental_mixed_contractions = experimental_mixed_point;
      experimental_mixed_history = experimental_mixed_point;
    }
    if (const auto* selected = std::getenv("GENERATIVEQC_EXPERIMENTAL_PBE_POINT_FP32")) {
      if (std::strcmp(selected, "0") != 0 && std::strcmp(selected, "1") != 0 &&
          std::strcmp(selected, "2") != 0 && std::strcmp(selected, "3") != 0)
        throw std::invalid_argument("experimental PBE point selection must be 0, 1, 2 or 3");
      experimental_mixed_point = std::strcmp(selected, "0") != 0;
      experimental_mixed_ao = std::strcmp(selected, "2") == 0 || std::strcmp(selected, "3") == 0;
      experimental_mixed_contractions = std::strcmp(selected, "3") == 0;
      if (experimental_mixed_point &&
          (spins != 1 || functional != static_cast<std::uint32_t>(SemilocalFamily::Pbe) ||
           options.xc_execution_schedule != scf::ScfOptions::XcExecutionSchedule::DeviceFused))
        throw std::invalid_argument("experimental FP32 points require device-fused PBE RKS");
    }
    if (const auto* selected = std::getenv("GENERATIVEQC_EXPERIMENTAL_PBE_POINT_KEEP_DIIS")) {
      if ((std::strcmp(selected, "0") != 0 && std::strcmp(selected, "1") != 0) ||
          !experimental_mixed_point)
        throw std::invalid_argument("experimental DIIS handoff requires mixed points and 0 or 1");
      experimental_mixed_history = std::strcmp(selected, "1") == 0;
    }
    precision_schedule = resolve_cuda_ks_precision_schedule(
        application_precision && !experimental_mixed_point && !budget_fock
            ? std::optional<generativeqc_precision_mode>{GENERATIVEQC_PRECISION_FP64}
            : options.precision_mode,
        xc_layout.fast_paths, fitted_coulomb, nonlocal_correlation != nullptr,
        experimental_mixed_point, experimental_mixed_contractions,
        budget_fock ? fock_precision : scf::CudaFockPrecisionCapabilities{}, has_range_correction,
        range_correction && scf::cuda_range_exchange_precision_capability(*range_correction),
        fitted_xc_budget);
    if (precision_schedule.any_lower_precision()) {
      mixed_warmup =
          resolve_mixed_precision_warmup(std::getenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL"),
                                         std::getenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS"));
      if (experimental_mixed_point || budget_fock) {
        if (!std::getenv("GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL"))
          mixed_warmup.switch_residual = 1e-4;
        if (!std::getenv("GENERATIVEQC_KS_MIXED_MAX_ITERATIONS")) mixed_warmup.max_iterations = 12;
      }
      // Early-handoff evidence currently covers restricted owners only. Keep
      // the incumbent UKS schedule until independent open-shell qualification.
      if (spins != 1 && (mixed_warmup.switch_residual > 0.0 || mixed_warmup.max_iterations))
        throw std::invalid_argument("bounded mixed-precision warmup requires restricted CUDA KS");
    }
    early_final_probe_requested = early_final_probe_default_model(
        functional, options.semilocal_exchange_scale, options.semilocal_correlation_scale,
        exchange_coefficient, grid_spec == default_grid_spec());
    // Primitive 450 is the complete M06-2X functional. Its solver error is
    // qualified separately from the retained default-grid error; exact mixing
    // and the complete final-state proof remain required.
    standard_convergence_model =
        early_final_probe_requested ||
        (functional == (0x20000U | 450U) && options.semilocal_exchange_scale == 1.0 &&
         options.semilocal_correlation_scale == 1.0 && exchange_coefficient == -0.27 &&
         grid_spec == default_grid_spec());
    if (const char* setting = std::getenv("GENERATIVEQC_KS_EARLY_FINAL_PROBE")) {
      if (std::strcmp(setting, "0") != 0 && std::strcmp(setting, "1") != 0)
        throw std::invalid_argument("GENERATIVEQC_KS_EARLY_FINAL_PROBE requires 0 or 1");
      early_final_probe_requested = std::strcmp(setting, "1") == 0;
    }
    const bool range_incremental_eligible =
        !has_range_correction ||
        (range_correction->backend == scf::FockBackend::Cuda &&
         !range_correction->spec.coulomb.present && range_correction->spec.exchange.present &&
         range_correction->spec.exchange.approximation == scf::FockApproximation::Exact);
    auto incremental_policy_options = options;
    // The prepared provider owns the executed screening contract. Native callers
    // may supply ScfOptions independently, so never let an options mismatch turn
    // a screened lower into an unbounded exact-linear delta chain.
    incremental_policy_options.screening_tolerance = strategy.screening_tolerance;
    // Keep the automatically promoted domain narrower than explicit opt-in.
    // In particular, public resource plans have no charged incremental arena
    // and must retain their incumbent solver; full physical closure is unchanged.
    const bool auto_through_d = std::all_of(
        provider.system().shells.begin(), provider.system().shells.end(),
        [](const auto& shell) { return shell.angular_momentum <= 2U; });
    const bool auto_all_electron =
        provider.system().ecp_terms.empty() &&
        std::none_of(provider.system().atoms.begin(), provider.system().atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; });
    const bool auto_incremental = admit_ks_incremental_cold_auto(
        {options.incremental_direct_jk_auto, n, provider.system().atoms.size(),
         static_cast<bool>(fock_binding) && !fitted_coulomb && !fitted_exchange &&
             has_exchange && spins == 1 && !has_range_correction && !nonlocal_correlation &&
             scf::direct_jk_incremental_exact_eligible(strategy) &&
             !precision_schedule.any_lower_precision(),
         standard_convergence_model && grid_spec == default_grid_spec() && !budget_fock &&
             configured_chunk_width() == 1 &&
             MatrixLibraryOwner::provider_allowance(static_cast<int>(n)),
         auto_through_d, auto_all_electron, !runtime::active_device_resource_ledger});
    incremental_policy_options.incremental_direct_jk |= auto_incremental;
    incremental_direct_jk_policy = scf::resolve_incremental_direct_jk_policy(
        incremental_policy_options,
        {static_cast<bool>(fock_binding) && !fitted_coulomb &&
             scf::direct_jk_incremental_exact_eligible(strategy) && range_incremental_eligible,
         true, precision_schedule.any_lower_precision()});
    incremental_direct_jk = incremental_direct_jk_policy.active;
    if (incremental_direct_jk_policy.requested && !incremental_direct_jk)
      throw std::invalid_argument(
          "CUDA KS incremental Direct-J/K requires one strict-FP64 exact Direct provider");
    if (const auto* setting = std::getenv("GENERATIVEQC_KS_INCREMENTAL_PROPOSALS")) {
      if (std::strcmp(setting, "0") != 0 && std::strcmp(setting, "1") != 0)
        throw std::invalid_argument("GENERATIVEQC_KS_INCREMENTAL_PROPOSALS requires 0 or 1");
      incremental_proposals_requested = std::strcmp(setting, "1") == 0;
    }
    if (const auto* setting = std::getenv("GENERATIVEQC_KS_INCREMENTAL_COARSE_SCREENING")) {
      if (std::strcmp(setting, "0") != 0 && std::strcmp(setting, "1") != 0)
        throw std::invalid_argument("GENERATIVEQC_KS_INCREMENTAL_COARSE_SCREENING requires 0 or 1");
      incremental_coarse_screening_requested = std::strcmp(setting, "1") == 0;
    }
    fast_anchor_eligible = admit_fast_anchor_delta(
        budget_fock, budget_stationarity, functional,
        static_cast<bool>(fock_binding) && !fitted_coulomb && !fitted_exchange && has_exchange &&
            has_range_correction && scf::direct_jk_incremental_exact_eligible(strategy) &&
            range_incremental_eligible && !incremental_direct_jk,
        spins,
        nonlocal_correlation &&
            nonlocal_correlation->parameters().variant == nlc::Vv10Variant::vv10,
        runtime::active_device_resource_ledger != nullptr, configured_chunk_width());
    if (fast_anchor_eligible) {
      if (const auto* selected = std::getenv("GENERATIVEQC_KS_FAST_ANCHOR_DELTA")) {
        if (std::strcmp(selected, "0") != 0 && std::strcmp(selected, "1") != 0)
          throw std::invalid_argument("Fast anchor delta accepts 0 or 1");
        fast_anchor_eligible = std::strcmp(selected, "1") == 0;
      } else {
        fast_anchor_eligible = default_fast_anchor_delta(n);
      }
    }
    const bool host_unfused =
        options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::HostUnfused;
    // Automatic local-AO requests use the XC owner's execution capability.
    // Keep 0 as a debugging opt-out and 1 as a fail-closed explicit request.
    // Capability describes legal execution, not endpoint profitability.
    // Fixed geometry maps belong to this owner, so a coordinate/grid rebuild
    // necessarily reruns discovery rather than reusing a pointer-based mask.
    const char* ao_selection = std::getenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO");
    const bool disable_ao = ao_selection && std::strcmp(ao_selection, "0") == 0;
    const bool request_ao = ao_selection && std::strcmp(ao_selection, "1") == 0;
    if (ao_selection && !disable_ao && !request_ao)
      throw std::invalid_argument("GENERATIVEQC_CUDA_KS_ACTIVE_AO accepts only 0 or 1");
    const bool capable_local_ao =
        !host_unfused && cuda_xc_execution_capabilities(xc_layout).local_ao_selection;
    if (request_ao && !capable_local_ao)
      throw std::invalid_argument(
          "local SCF AO maps require a device-fused physical FP64 XC layout");
    // Experimental schedule control is frozen at geometry preparation. It
    // changes only discovery ownership; the sampled predicate and dense
    // fallback admission remain the same for every functional.
    const char* discovery = std::getenv("GENERATIVEQC_CUDA_KS_AO_DISCOVERY");
    const bool device_discovery = discovery && std::strcmp(discovery, "device") == 0;
    if (discovery && !device_discovery && std::strcmp(discovery, "host") != 0)
      throw std::invalid_argument("GENERATIVEQC_CUDA_KS_AO_DISCOVERY accepts host or device");
    const bool select_ao = !disable_ao && capable_local_ao;
    constexpr std::size_t ao_map_host_budget = 64U << 20;
    CudaXcAoSelectionResources ao_selection_bound;
    if (select_ao) ao_selection_bound = cuda_xc_ao_selection_resources(xc_layout);
    // Public ledgers reserve dense XC, later fleet owners and force storage,
    // not optional retained maps. Spare capacity is not a map allowance, even
    // for an explicit request or an unlimited public ResourceBudget().
    bool admit_ao = select_ao && !runtime::active_device_resource_ledger &&
                    ao_selection_bound.host_peak_bytes <= ao_map_host_budget;
    if (host_unfused &&
        (options.semilocal_exchange_scale != 1.0 || options.semilocal_correlation_scale != 1.0))
      throw std::invalid_argument("scaled CUDA XC requires device-fused execution");
    const auto* host_family = semilocal_family_metadata_from_code(functional);
    const bool curated_requires_fused_hybrid =
        host_family && has_exchange && host_family->cuda_global_hybrid_exact_exchange > 0.0 &&
        !host_family->component_coefficients_are_native_scales;
    if (host_unfused &&
        (curated_requires_fused_hybrid || generated::split_hybrid_registered(functional)))
      throw std::invalid_argument(
          "generated/global-hybrid CUDA XC requires device-fused execution");
    if (host_unfused) {
      host_xc_density.resize(elements);
      host_xc_potential.resize(elements);
      if (spins == 2) {
        host_xc_alpha.resize(matrix);
        host_xc_beta.resize(matrix);
      }
    }
    if (device_nonlocal) {
      // Both ordinary and graph replays apply MolecularV1's domain first.
      // Its negative-zero weight marks a screened row whose energy and AO
      // potential are discarded. Preserve active positive-zero/signed weights
      // while avoiding that row's otherwise unnecessary partner traversal.
      nonlocal_layout = nlc::vv10_cuda_device_layout(nonlocal_grid.point_count(),
                                                     xc_layout.tile_points, true, false, true);
      const auto* cache_choice = std::getenv("GENERATIVEQC_VV10_SCF_COMPONENT_REUSE");
      const auto parameters = nonlocal_correlation->parameters();
      // Complete endpoint controls qualify the exact component cache once
      // quadratic pair work dominates its linear ingredient comparison.
      const bool cache_requested = cache_choice ? std::string_view(cache_choice) == "1"
                                                : nonlocal_grid.point_count() >= 65536;
      nonlocal_cache_enabled = cache_requested && nonlocal_grid.point_count() >= 3 &&
                               nonlocal_grid.point_count() <= (std::uint64_t{1} << 32) &&
                               parameters.variant == nlc::Vv10Variant::vv10 &&
                               std::fabs(parameters.coefficient) <= 0x1p32;
      nonlocal_arena_bytes = partition_nonlocal(nullptr);
      if (nonlocal_cache_enabled &&
          nonlocal_arena_bytes > nonlocal_correlation->resources().device_workspace_bytes) {
        nonlocal_cache_enabled = false;
        nonlocal_arena_bytes = partition_nonlocal(nullptr);
      }
      if (nonlocal_arena_bytes > nonlocal_correlation->resources().device_workspace_bytes)
        throw std::invalid_argument(
            "resident CUDA KS nonlocal workspace exceeds the prepared VV10 device bound");
    }
    ks_arena_bytes = partition(n, spins, history, has_exchange, has_range_correction,
                               incremental_direct_jk, incremental_diis_gram, nullptr);
    resource.state_device_bytes = sum(ks_arena_bytes, nonlocal_arena_bytes);
    resource.xc_device_bytes =
        host_unfused ? 0 : (admit_ao ? ao_selection_bound.device_bytes : xc_layout.device_bytes);
    resource.grid_device_bytes = borrow_resident_grid ? resident_grid.device_bytes : 0;
    resource.provider_device_bytes = provider.diagnostic().device_bytes;
    if (range_provider)
      resource.provider_device_bytes =
          sum(resource.provider_device_bytes, range_provider->diagnostic().device_bytes);
    const auto diagnostic_iterations =
        precision_schedule.any_lower_precision()
            ? sum(product(options.max_iterations, 2U), kMaximumFinalCorrections)
            : (incremental_direct_jk ? sum(options.max_iterations, kMaximumFinalCorrections)
                                     : options.max_iterations);
    output.dft_diagnostic.history.reserve(diagnostic_iterations);
    final_validation_diagnostic.determinant.eigenframes.reserve(spins);
    resource.retained_host_numeric_bytes =
        (host_xc_density.capacity() + host_xc_alpha.capacity() + host_xc_beta.capacity() +
         host_xc_potential.capacity()) *
            sizeof(double) +
        output.dft_diagnostic.history.capacity() * sizeof(ScfIteration) + sizeof(host_xc_totals) +
        sizeof(host_xc_error) + sizeof(host_spin_counts) + sizeof(host_selected) +
        sizeof(host_all_spins) + sizeof(host_one) +
        final_validation_diagnostic.determinant.eigenframes.capacity() *
            sizeof(scf::solver::EigenFrameDiagnostic) +
        tensor::CudaMatrixProductBinding::kPreparationHostBytes +
        tensor::CudaStructuredProductBinding::kPreparationHostBytes;
    if (!host_unfused)
      resource.retained_host_numeric_bytes =
          sum(resource.retained_host_numeric_bytes, CudaXcLayout::lowering_host_bytes);
    // Conservatively retain the setup peak in the owner's capacity report.
    // This experiment's fixed host cap does not implement public host-budget
    // admission: the Python resource planner currently rejects WB97M-V.
    if (admit_ao)
      resource.retained_host_numeric_bytes =
          sum(resource.retained_host_numeric_bytes, ao_selection_bound.host_peak_bytes);
    try {
      check(runtime::resource_cuda_malloc(&arena, ks_arena_bytes));
      partition(n, spins, history, has_exchange, has_range_correction, incremental_direct_jk,
                incremental_diis_gram, arena);
      if (nonlocal_arena_bytes) {
        bool host_oom = false;
        auto status =
            runtime::resource_cuda_malloc(&nonlocal_arena, nonlocal_arena_bytes, &host_oom);
        if (nonlocal_cache_enabled && status == cudaErrorMemoryAllocation && !host_oom) {
          // The optional scalar/key/counter lease must not make an otherwise
          // admissible complete solve fail. Preserve non-allocation errors.
          const auto pending = cudaGetLastError();
          if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) check(pending);
          const auto prior_bytes = nonlocal_arena_bytes;
          nonlocal_cache_enabled = false;
          nonlocal_arena_bytes = partition_nonlocal(nullptr);
          resource.state_device_bytes -= prior_bytes - nonlocal_arena_bytes;
          status = runtime::resource_cuda_malloc(&nonlocal_arena, nonlocal_arena_bytes);
        }
        check(status);
        partition_nonlocal(nonlocal_arena);
      }
      // Admit mandatory solver/VV10 storage before optional maps. A device
      // budget/allocation miss may retry the smaller dense XC arena; host
      // registry failures and other runtime errors must still propagate.
      check(matrix_products.prepare(stream, static_cast<int>(n)),
            "CUDA KS matrix provider preparation failed");
      if (matrix_products.library_enabled())
        resource.provider_device_bytes =
            sum(resource.provider_device_bytes, MatrixLibraryOwner::kProviderAllowance);
      eigensolver = std::make_unique<OrdinaryStreamEigensolver>(stream, n, tmp2, eigenvalues);
      resource.state_device_bytes = sum(resource.state_device_bytes, eigensolver->device_bytes());
      resource.retained_host_numeric_bytes =
          sum(resource.retained_host_numeric_bytes,
              sum(eigensolver->host_bytes(), eigensolver->metadata_bytes()));
      // Reserve the mandatory secondary transport before optional primary AO maps.
      // A map allocation miss must still be able to retry the dense primary arena.
      if (!host_unfused && options.nonlocal_grid)
        check(runtime::resource_cuda_malloc(&nonlocal_xc_arena, nonlocal_xc_layout.device_bytes));
      if (resource.xc_device_bytes) {
        bool host_oom = false;
        auto status = runtime::resource_cuda_malloc(&xc_arena, resource.xc_device_bytes, &host_oom);
        if (admit_ao && status == cudaErrorMemoryAllocation && !host_oom) {
          const auto pending = cudaGetLastError();
          if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) check(pending);
          admit_ao = false;
          resource.retained_host_numeric_bytes -= ao_selection_bound.host_peak_bytes;
          resource.xc_device_bytes = xc_layout.device_bytes;
          status = runtime::resource_cuda_malloc(&xc_arena, resource.xc_device_bytes);
        }
        check(status);
      }
      check(cudaMemsetAsync(arena, 0, ks_arena_bytes, stream));
      if (nonlocal_arena) check(cudaMemsetAsync(nonlocal_arena, 0, nonlocal_arena_bytes, stream));
      const auto upload = [&](void* destination, const void* source, std::size_t bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
        movement.setup_h2d_bytes += bytes;
      };
      upload(hcore, provider.one_electron().hcore.data(), matrix * sizeof(double));
      upload(overlap, provider.one_electron().overlap.data(), matrix * sizeof(double));
      host_spin_counts = {static_cast<std::int32_t>(occupations[0]),
                          static_cast<std::int32_t>(occupations[1])};
      host_selected = {static_cast<std::uint8_t>(occupations[0] > 0),
                       static_cast<std::uint8_t>(occupations[1] > 0)};
      upload(occupied, host_spin_counts.data(), spins * sizeof(std::int32_t));
      upload(spin_enabled, host_selected.data(), spins * sizeof(std::uint8_t));
      upload(final_spin_enabled, host_all_spins.data(), spins * sizeof(std::uint8_t));
      upload(final_enabled, &host_one, sizeof(host_one));
      if (!host_unfused) {
        // Device-fused XC setup drains this same stream.
        xc = std::make_unique<CudaXcPlan>(
            basis, grid, functional, spins == 2, tile, xc_arena, resource.xc_device_bytes, stream,
            CudaXcAoPrecision::Fp64, options.semilocal_exchange_scale,
            options.semilocal_correlation_scale, borrow_resident_grid);
        if (options.nonlocal_grid) {
          resource.xc_device_bytes = sum(resource.xc_device_bytes, nonlocal_xc_layout.device_bytes);
          resource.grid_device_bytes =
              sum(resource.grid_device_bytes, nonlocal_grid.cuda_view().device_bytes);
          resource.retained_host_numeric_bytes =
              sum(resource.retained_host_numeric_bytes, CudaXcLayout::lowering_host_bytes);
          // VV10 needs only total rho/grad-rho and their GGA AO pullback. The
          // qualified zero-scaled PBE transport supplies those without adding
          // PBE energy or requiring WB97M-V component-scaling capability.
          nonlocal_xc = std::make_unique<CudaXcPlan>(
              basis, nonlocal_grid, semilocal_family_code(SemilocalFamily::Pbe), spins == 2, tile,
              nonlocal_xc_arena, nonlocal_xc_layout.device_bytes, stream, CudaXcAoPrecision::Fp64,
              0.0, 0.0, static_cast<bool>(nonlocal_grid.cuda_view()));
        }
        if (experimental_mixed_point)
          xc->prepare_mixed_pbe_points(experimental_mixed_ao, experimental_mixed_contractions);
        // Keep the canonical resident quadrature for grid-response forces.
        // The optional semilocal execution copy changes only point order and
        // is excluded under a public ledger until a fleet planner reserves it.
        const auto* spatial_choice = std::getenv("GENERATIVEQC_CUDA_XC_SPATIAL_ORDER");
        // Unset selects the qualified spatial schedule; explicit 0 keeps the
        // canonical execution order for diagnostics and paired comparisons.
        if ((!spatial_choice || std::string_view(spatial_choice) == "1") && admit_ao &&
            borrow_resident_grid && !nonlocal_correlation &&
            !runtime::active_device_resource_ledger) {
          const auto spatial_bytes = product(xc_layout.npoint, 4 * sizeof(double));
          const auto host_bytes =
              sum(spatial_bytes, product(sum(product(2, xc_layout.npoint), product(4, basis.natom)),
                                         sizeof(std::size_t)));
          xc->prepare_spatial_grid(grid, host_bytes, spatial_bytes);
          resource.xc_device_bytes = sum(resource.xc_device_bytes, xc->spatial_grid_device_bytes());
        }
        if (admit_ao) {
          if (!xc->select_local_ao(1e-16, ao_map_host_budget,
                                   device_discovery ? CudaXcAoDiscovery::DeviceChunks
                                                    : CudaXcAoDiscovery::HostTiles))
            throw std::logic_error("admitted native SCF AO selection failed its resource check");
          xc_layout = xc->layout();
        }
        const auto admitted_precision = resolve_cuda_ks_iteration_precision(
            precision_schedule, false,
            cuda_xc_execution_capabilities(xc_layout).mixed_density_contraction);
        if (admitted_precision.uses_lower_precision(
                cuda_ks_precision_region::kXcMatrixContractions) &&
            cuda_xc_detail::mixed_matrix_contractions_supported(xc_layout))
          xc->prepare_mixed_matrix_contractions();
        // Host and device reservations must both precede optional preparation.
        // The lowering layer owns qualification and implementation selection.
        const auto density_budget =
            xc_budget.host_bytes >= tensor::PreparedPanelProduct::host_reservation
                ? xc_budget.device_bytes
                : 0;
        xc->prepare_density(*admitted_precision.find(cuda_ks_precision_region::kDensityContraction),
                            options.max_iterations, density_budget);
        if (const auto* prepared = xc->density_provider_diagnostic()) {
          resource.xc_device_bytes = sum(resource.xc_device_bytes, prepared->matrix_bytes);
          resource.provider_device_bytes =
              sum(resource.provider_device_bytes, prepared->provider_allowance);
          resource.retained_host_numeric_bytes =
              sum(resource.retained_host_numeric_bytes, prepared->host_bytes);
        }
        // Default batching retains the compiler's bounded admission and the
        // one-tile fallback. Overrides keep the incumbent available for ablations;
        // the public inventory currently reserves only incumbent XC storage.
        const auto point_batch_size = [](const char* name, std::size_t fallback) {
          const char* value = std::getenv(name);
          if (!value) return fallback;
          std::size_t result = 0;
          const auto* end = value + std::strlen(value);
          const auto parsed = std::from_chars(value, end, result);
          if (parsed.ec != std::errc{} || parsed.ptr != end)
            throw std::invalid_argument(std::string(name) + " requires a nonnegative integer");
          return result;
        };
        const auto point_batch_tiles = point_batch_size("GENERATIVEQC_CUDA_XC_BATCH_TILES", 32);
        if (point_batch_tiles > 1) {
          constexpr std::size_t incumbent_point_batch_budget = 32 * 1024 * 1024;
          const auto automatic_point_batch_budget =
              product(incumbent_point_batch_budget, xc_layout.work_jets);
          const auto point_batch_budget =
              point_batch_size("GENERATIVEQC_CUDA_XC_BATCH_BYTES", automatic_point_batch_budget);
          const bool explicit_point_batch_budget =
              std::getenv("GENERATIVEQC_CUDA_XC_BATCH_BYTES") != nullptr;
          const bool compact_point_batches =
              point_batch_size("GENERATIVEQC_CUDA_XC_COMPACT_BATCH", 1) != 0;
          const bool ragged_point_batches =
              point_batch_size("GENERATIVEQC_CUDA_XC_RAGGED_BATCH", 1) != 0;
          // A live public ledger also reserves later fleet owners and force
          // workspace. Its currently unused bytes are not an optional allowance,
          // even with an unlimited user budget. Keep the incumbent until a plan
          // explicitly accounts for optional panels across those lifetimes.
          xc->prepare_point_batches(point_batch_tiles,
                                    runtime::active_device_resource_ledger ? 0 : point_batch_budget,
                                    compact_point_batches, ragged_point_batches);
          if (!explicit_point_batch_budget && !runtime::active_device_resource_ledger &&
              automatic_point_batch_budget > incumbent_point_batch_budget &&
              xc->point_batch_plan().tiles == 1)
            xc->prepare_point_batches(point_batch_tiles, incumbent_point_batch_budget,
                                      compact_point_batches, ragged_point_batches);
          resource.xc_device_bytes =
              sum(resource.xc_device_bytes, xc->point_batch_plan().device_bytes);
        }
        prepared_ao_work = xc->ao_selection_work();
        prepared_ao_work.requested = select_ao;
        if (!admit_ao) {
          prepared_ao_work.cutoff = select_ao ? 1e-16 : 0;
          prepared_ao_work.tiles = 1 + (xc_layout.npoint - 1) / xc_layout.tile_points;
          prepared_ao_work.min_active = prepared_ao_work.max_active = xc_layout.nao;
          prepared_ao_work.active_sum = product(prepared_ao_work.tiles, xc_layout.nao);
          prepared_ao_work.point_ao_visits = product(xc_layout.npoint, xc_layout.nao);
          prepared_ao_work.point_ao_square_sum =
              product(prepared_ao_work.point_ao_visits, xc_layout.nao);
          prepared_ao_work.dense_point_ao_square_sum = prepared_ao_work.point_ao_square_sum;
          prepared_ao_work.reserved_device_bytes = resource.xc_device_bytes;
        }
      }
      prepare_initial_state();
      prepare_fast_anchor_storage();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  void cleanup() noexcept {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (range_fock_binding.stream && range_fock_binding.stream != stream)
      cudaStreamSynchronize(range_fock_binding.stream);
    try {
      retire_phase_source(false);
    } catch (...) {
      // Graph invalidation has already released captured handles; the local
      // retired lease drains even if diagnostic metadata cannot be allocated.
    }
    xc.reset();
    nonlocal_xc.reset();
    eigensolver.reset();
    matrix_products.reset();
    if (fast_anchor_arena) runtime::resource_cuda_free(fast_anchor_arena);
    if (nonlocal_arena) runtime::resource_cuda_free(nonlocal_arena);
    if (xc_arena) runtime::resource_cuda_free(xc_arena);
    if (nonlocal_xc_arena) runtime::resource_cuda_free(nonlocal_xc_arena);
    if (arena) runtime::resource_cuda_free(arena);
    nonlocal_arena = xc_arena = arena = nullptr;
    nonlocal_xc_arena = nullptr;
    fast_anchor_arena = nullptr;
    cudaSetDevice(previous);
  }
  ~Impl() { cleanup(); }

  void begin(const std::vector<double>* input, bool reuse_warm, CudaKsSeedOrigin seed_origin) {
    if (is_pending) throw std::logic_error("cannot replace a pending CUDA KS iteration");
    final_state_ready = final_frame_ready = final_stationary_weights_ready = false;
    final_fitted_projection_ready = false;
    final_generation = 0;
    occupied_fitted_factor_ready = false;
    pending_fitted_occupied = false;
    // Arbitrary imported/changed-geometry seeds never borrow a previous frame.
    // A same-owner factor-backed warm seed can restore its own exact AO C below.
    invalidate_warm_orbitals();
    if (solve_epoch == std::numeric_limits<std::uint64_t>::max()) {
      is_active = false;
      is_failed = true;
      throw std::overflow_error("CUDA KS solve epoch exhausted");
    }
    ++solve_epoch;
    is_active = false;
    is_failed = true;
    // A new attempt invalidates execution evidence immediately, before any
    // device selection, injected runtime failure, or seed validation can
    // escape.  Never leave the prior successful solve's precision receipt
    // observable after a failed begin().
    output.precision = {};
    output.precision.requested_mode = options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64);
    output.precision_work = {};
    output.precision_work.owner_id = owner;
    output.precision_work.operators.reserve(14);
    current_device();
    runtime::cuda_trace::TraceOperation trace("ks_begin", stream, {1, n, 0, fitted_coulomb, false});
    const auto initial_synchronizations = movement.synchronizations;
#if defined(GENERATIVEQC_TEST_HOOKS)
    if (fail_next_ks_runtime) {
      fail_next_ks_runtime = false;
      check(cudaErrorUnknown);
    }
#endif
    const bool use_warm = !input && reuse_warm && warm_ready;
    // Incremental storage is a prepared capability, but a previously audited
    // resident density needs the ordinary full-density warm proof. Re-entering
    // the delta controller would discard that energy history and force extra
    // full builds even when no delta was applied. External/moved seeds do not
    // carry this immutable owner's warm provenance and retain the cold policy.
    // Keep unqualified UKS/ECP/range/nonlocal owners on their existing policy.
    // Recompute on every begin so a later cold solve restores the capability.
    const bool ordinary_incremental_warm =
        use_warm && spins == 1 && !has_range_correction && !nonlocal_correlation &&
        provider.system().ecp_terms.empty() &&
        std::none_of(provider.system().atoms.begin(), provider.system().atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; });
    // Automatic promotion never borrows external/imported/moved seed provenance.
    // Retained warm states use the existing full-density one-Fock proof. Explicit
    // opt-in callers retain their original external-seed eligibility.
    incremental_direct_jk =
        incremental_direct_jk_policy.active && !ordinary_incremental_warm &&
        (!options.incremental_direct_jk_auto || input == nullptr);
    // Only an owner with the complete Standard full-density closure may use a
    // bounded chain of screened proposals. J/K linearity is independent of
    // the XC family; inherit the ordinary Standard owner's existing model
    // and proof admission without adding an incremental functional whitelist.
    // A chain never supplies the final
    // energy history or physical proof. Strict requests and unsupported owners
    // retain the ordinary one-delta refresh policy; warm performs no proposals.
    incremental_proposals_active = incremental_direct_jk && incremental_proposals_requested &&
                                   provider.strategy().screening_tolerance > 0.0 &&
                                   standard_convergence_active();
    runtime::cuda_trace::trace_counter("incremental_proposals", incremental_proposals_active);
    runtime::cuda_trace::trace_counter(
        "incremental_refresh_interval",
        incremental_direct_jk ? ks_incremental_proposal_interval(
                                    incremental_proposals_active,
                                    incremental_direct_jk_policy.requested_rebuild_interval,
                                    incremental_direct_jk_policy.effective_rebuild_interval)
                              : 0U);
    const auto* seed_selection = std::getenv("GENERATIVEQC_VV10_NATIVE_SEED");
    const auto* outer_selection = std::getenv("GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED");
    // Initialization only proposes densities; full target evaluations still
    // certify convergence. An explicit off setting keeps the ordinary solve,
    // and an explicitly selected device chunk retains its own solver policy.
    const bool outer_requested =
        outer_selection ? std::string_view(outer_selection) == "1" : configured_chunk_width() == 1;
    nonlocal_seed_trace = seed_selection && std::string_view(seed_selection) == "1";
    const bool nonlocal_initialization =
        (nonlocal_seed_trace || outer_requested) && device_nonlocal && spins == 1 &&
        nonlocal_correlation &&
        nonlocal_correlation->parameters().variant == nlc::Vv10Variant::vv10 &&
        (!precision_schedule.any_lower_precision() || budget_fock) && !incremental_direct_jk &&
        !use_warm && options.max_iterations > 2;
    nonlocal_outer_active = outer_requested && nonlocal_initialization;
    fast_anchor_active = fast_anchor_arena && nonlocal_outer_active &&
                         configured_chunk_width() == 1 && !runtime::active_device_resource_ledger;
    fast_anchor_ready = pending_fast_delta = false;
    if (fast_anchor_active) check(cudaMemsetAsync(fast_delta_maximum, 0, sizeof(double), stream));
    if (budget_fock && budget_stationarity) {
      // Finish one bounded mixed proposal before paying for the strict anchor.
      // Application budgets reject explicit warmup overrides at preparation;
      // ordinary AUTO and explicitly strict callers retain their own policy.
      mixed_warmup.max_iterations =
          nonlocal_outer_active ? nonlocal_mixed_proposal_limit(options.max_iterations) : 12;
      mixed_warmup.switch_residual =
          nonlocal_outer_active ? nonlocal_confirmation_tolerance(true, options.density_tolerance)
                                : 1e-4;
    }
    // A transported/external density may already approximate the full target.
    // Anchor it first instead of pulling it back to the semilocal-only solution.
    const bool native_cold_seed = !input || seed_origin == CudaKsSeedOrigin::NativeColdGuess;
    nonlocal_seed_stage = nonlocal_initialization && (!outer_requested || native_cold_seed);
    nonlocal_inner_residual_tolerance = 1e-9;
    nonlocal_frozen_stage = pending_nonlocal_frozen = false;
    nonlocal_inner_iterations = nonlocal_exact_epoch_iterations = 0;
    nonlocal_seed_iterations = nonlocal_exact_iterations = 0;
    pending_nonlocal_seed = false;
    const bool retain_range_source = retain_range_source_for_replay(use_warm);
    if (!retain_range_source) retire_phase_source(false);
    std::vector<double> prepared;
    if (input) prepared = seed(input);  // Validate before replacing current state.
    auto retained_history = std::move(output.dft_diagnostic.history);
    retained_history.clear();
    output = {};
    output.dft_diagnostic.cuda_ao_selection = prepared_ao_work;
    initial_xc_evaluations = xc ? xc->transfers().evaluations : 0;
    output.dft_diagnostic.history = std::move(retained_history);
    output.dft_diagnostic.occupations = occupations;
    output.dft_diagnostic.grid_points = xc_layout.npoint;
    output.dft_diagnostic.tile_points = xc_layout.tile_points;
    output.dft_diagnostic.ao_order = xc_layout.jets == 1 ? 0 : 1;
    // Scientific domain identity follows the functional, including B3LYP v2.
    output.dft_diagnostic.scf_domain_version =
        generated::split_hybrid_registered(functional)
            ? 4U
            : semilocal_family_domain_version(semilocal_family_from_code(functional));
    output.initial_density_used = input != nullptr || use_warm;
    output.incremental_direct_jk.requested = incremental_direct_jk_policy.requested;
    output.incremental_direct_jk.active = incremental_direct_jk;
    incremental_anchored = false;
    pending_incremental_delta = false;
    incremental_energy_refinement = false;
    incremental_energy_full_builds = 0;
    incremental_delta_updates = 0;
    host_incremental_max_abs_delta_density = 0.0;
    is_active = false;
    started = true;
    is_failed = false;
    stabilize_occupations = false;
    final_closure = false;
    // A resident, same-model warm proposal already passed the strict physical
    // gates. Re-evaluate it in FP64 before perturbing it with approximate J/K or XC:
    // a converged frozen replay should not manufacture another mixed SCF.
    // This selects arithmetic, not convergence: every ordinary energy,
    // density and residual gate still runs. Cold/external/moved seeds retain
    // the admitted warmup, and ordinary AUTO keeps its incumbent contract.
    strict_refinement = use_warm && options.application_precision_budget[0] > 0.0 &&
                        (experimental_mixed_point || budget_fock);
    pending_mixed_coulomb = false;
    pending_mixed_exchange = false;
    pending_mixed_range_exchange = false;
    pending_mixed_range_resident = false;
    pending_mixed_nonlocal = false;
    pending_mixed_resident = false;
    pending_mixed_density = false;
    pending_mixed_point = false;
    pending_mixed_contractions = false;
    mixed_precision_executed = false;
    final_corrections = 0;
    refinement_iterations = 0;
    output.precision.requested_mode = options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64);
    output.precision_work.owner_id = owner;
    output.precision_work.operators.reserve(14);
    pending_iterations = 0;
    try {
      // Native cold guesses arrive as a density too. Preserve their provenance
      // instead of treating every non-null pointer as a warm or external seed.
      phase_source_cold = (!input || seed_origin == CudaKsSeedOrigin::NativeColdGuess) && !use_warm;
      if (!retain_range_source) prepare_phase_source(phase_source_cold, use_warm);
      // The bounded device-control prototype is qualified only for strict-FP64
      // direct all-electron RKS. AUTO must stay on the legacy host-controlled
      // path so its FP32 mixed-J stage and independent FP64 refinement cannot be
      // bypassed by an opt-in two-iteration device chunk. In addition to the
      // original pure semilocal envelope, admit the manifest-qualified direct
      // global-hybrid composition so exact K can remain inside the same bounded
      // SolverRegion. Graph replay
      // stays disabled for global hybrids until the exchange provider is
      // independently capture-qualified.
      const bool pure_semilocal_chunk = !has_exchange && !has_range_correction &&
                                        options.semilocal_exchange_scale == 1.0 &&
                                        options.semilocal_correlation_scale == 1.0;
      const auto* chunk_family = semilocal_family_metadata_from_code(functional);
      const double chunk_exact_exchange =
          chunk_family ? chunk_family->cuda_global_hybrid_exact_exchange : -1.0;
      const double chunk_semilocal_exchange =
          chunk_family && chunk_family->component_coefficients_are_native_scales &&
                  chunk_exact_exchange > 0.0
              ? 1.0 - chunk_exact_exchange
              : 1.0;
      const bool curated_global_hybrid_chunk =
          has_exchange && !has_range_correction && chunk_exact_exchange > 0.0 &&
          options.semilocal_exchange_scale == chunk_semilocal_exchange &&
          options.semilocal_correlation_scale == 1.0 &&
          exchange_coefficient == -chunk_exact_exchange / (spins == 1 ? 2.0 : 1.0);
      // Range-separated exact exchange already stays device-resident on the
      // prepared Direct owner. Admit its ordinary two-call primary/correction
      // composition to the same bounded region; graph replay remains disabled
      // below because the range provider has not been capture-qualified.
      const bool rsh_chunk =
          has_exchange && has_range_correction && range_correction.has_value() &&
          range_correction->backend == scf::FockBackend::Cuda &&
          range_correction->spec.derivative_order == 0 && !range_correction->spec.coulomb.present &&
          range_correction->spec.exchange.present &&
          range_correction->spec.exchange.approximation == scf::FockApproximation::Exact &&
          (range_correction->spec.exchange.op == scf::FockOperator::ShortRange ||
           range_correction->spec.exchange.op == scf::FockOperator::LongRange) &&
          range_correction->spec.exchange.omega > 0.0;
      const bool resident_nonlocal_chunk =
          rsh_chunk && device_nonlocal && nonlocal_correlation != nullptr;
      device_chunk_mode =
          options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused &&
          !fitted_coulomb && (!nonlocal_correlation || resident_nonlocal_chunk) &&
          !precision_schedule.any_lower_precision() && !incremental_direct_jk && spins == 1 &&
          (pure_semilocal_chunk || curated_global_hybrid_chunk || rsh_chunk) &&
          provider.system().ecp_terms.empty() && configured_chunk_width() == kCudaKsChunkCapacity;
      if (nonlocal_initialization) device_chunk_mode = false;
      if (device_chunk_mode) {
        const auto binding = device_chunk_binding();
        if (!device_chunk_region.matches(binding))
          device_chunk_region.bind(binding);
        else if (device_chunk_region.failed())
          device_chunk_region.recover();
      } else if (device_chunk_region.bound()) {
        device_chunk_region.invalidate();
      }
      check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
      check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
      if (incremental_direct_jk)
        check(cudaMemsetAsync(incremental_max_abs_delta_density, 0, sizeof(double), stream));
      cuda_ks_detail::reset_control(stream, spins, static_cast<int>(occupations[0]),
                                    static_cast<int>(occupations[1]), control, enabled,
                                    spin_enabled);
      check(cudaGetLastError());
      if (nonlocal_seed_stage) {
        check(cudaMemsetAsync(nonlocal_domain_error, 0, sizeof(int), stream));
        check(cudaMemsetAsync(nonlocal_pair_error, 0, sizeof(int), stream));
      }
      if (nonlocal_cache_enabled)
        check(cudaMemsetAsync(nonlocal_cache_counters, 0, 3 * sizeof(std::uint64_t), stream));
      if (use_warm) {
        restore_warm_seed();
      } else if (input) {
        check(cudaMemcpyAsync(density, prepared.data(), elements * sizeof(double),
                              cudaMemcpyHostToDevice, stream));
        // Explicit initial-guess staging, never an iteration matrix transfer.
        check(cudaStreamSynchronize(stream));
        movement.density_h2d_bytes += elements * sizeof(double);
        ++movement.synchronizations;
      } else {
        check(cudaMemcpyAsync(density, cold_seed, elements * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream));
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_failed = true;
      retire_phase_source(false);
      if (device_chunk_mode && device_chunk_region.bound())
        device_chunk_region.mark_failure("CUDA KS device region preparation failed");
      throw;
    }
    // The retained radial source and incumbent provider each own an energy
    // baseline for the same accepted warm density. Never compare schedules at
    // the sub-ulp energy gate, including after force-memory cache retirement.
    const double baseline = retain_range_source ? range_warm_energy : warm_energy;
    const double correction =
        retain_range_source ? range_warm_energy_correction : warm_energy_correction;
    warm_energy_baseline = use_warm && !device_chunk_mode &&
                           (!phase_source || retain_range_source) && std::isfinite(baseline);
    previous_energy = warm_energy_baseline ? baseline : std::numeric_limits<double>::infinity();
    previous_energy_correction = warm_energy_baseline ? correction : 0.0;
    runtime::cuda_trace::trace_counter("owner", owner);
    runtime::cuda_trace::trace_counter("solve_epoch", solve_epoch);
    runtime::cuda_trace::trace_counter("warm_seed", use_warm);
    runtime::cuda_trace::trace_counter("explicit_seed", input != nullptr);
    runtime::cuda_trace::trace_counter("warm_density_generation", warm_density_generation);
    runtime::cuda_trace::trace_counter("warm_factor_generation", warm_factor_generation);
    runtime::cuda_trace::trace_counter("warm_factor_restored", occupied_fitted_factor_ready);
    runtime::cuda_trace::trace_counter("retained_state_bytes", resource.state_device_bytes);
    runtime::cuda_trace::trace_counter("retained_xc_bytes", resource.xc_device_bytes);
    runtime::cuda_trace::trace_counter("retained_provider_bytes", resource.provider_device_bytes);
    runtime::cuda_trace::trace_counter("explicit_synchronizations",
                                       movement.synchronizations - initial_synchronizations);
    is_active = true;
  }

  unsigned configured_chunk_width() const noexcept {
    const char* selection = std::getenv("GENERATIVEQC_CUDA_KS_CHUNK");
    if (selection != nullptr) {
      if (std::strcmp(selection, "0") == 0 || std::strcmp(selection, "1") == 0 ||
          std::strcmp(selection, "off") == 0 || std::strcmp(selection, "none") == 0)
        return 1;
      if (std::strcmp(selection, "2") == 0) return kCudaKsChunkCapacity;
    }
    // Complete cold/warm/changed-geometry endpoint measurements did not
    // establish a reproducible benefit for automatic promotion.
    return 1;
  }

  bool configured_replay_enabled() const noexcept {
    const char* selection = std::getenv("GENERATIVEQC_CUDA_KS_REPLAY");
    if (selection == nullptr) return false;
    return std::strcmp(selection, "1") == 0 || std::strcmp(selection, "on") == 0 ||
           std::strcmp(selection, "true") == 0 || std::strcmp(selection, "small-native") == 0;
  }

  runtime::SolverRegionCudaBinding solver_region_binding() const {
    const bool replay_point_program =
        cuda_xc_capability_qualified(xc_layout.fast_paths.graph_replay);
    // CUDA-Graph replay remains limited to the semilocal body qualified by
    // #1437. Global-hybrid chunks may use the shared bounded SolverRegion
    // without capturing the exact-exchange provider.
    const bool replay_semilocal_only =
        !has_exchange && !has_range_correction && !nonlocal_correlation && !fitted_coulomb &&
        !precision_schedule.any_lower_precision() && options.semilocal_exchange_scale == 1.0 &&
        options.semilocal_correlation_scale == 1.0;
    const bool replay = configured_replay_enabled() && replay_semilocal_only &&
                        replay_point_program &&
                        n <= static_cast<std::size_t>(scf::cuda_execution::kSmallEigensolverLimit);
    auto graph = device_chunk_binding();
    graph.qualification += warm_updates ? ":warm-updates" : ":frozen-warm";
    return {std::move(graph), kCudaKsChunkCapacity, runtime::SolverRegionCompletionMode::Scalar,
            replay};
  }

  unsigned submission_width() const noexcept {
    unsigned width = configured_chunk_width();
    if (width == 1 || output.iterations >= options.max_iterations) return 1;
    width = std::min<unsigned>(width, options.max_iterations - output.iterations);
    if (!output.dft_diagnostic.history.empty()) {
      const auto& last = output.dft_diagnostic.history.back();
      const double residual_gate = std::min(1e-9, options.density_tolerance);
      if (last.energy_change < 32.0 * options.energy_tolerance ||
          last.density_change < 32.0 * options.density_tolerance ||
          last.physical_residual < 32.0 * residual_gate)
        width = 1;
    }
    return std::max(1U, width);
  }

  void enqueue_one(unsigned slot) {
    if (slot >= kCudaKsChunkCapacity) throw std::logic_error("CUDA KS chunk slot overflow");
    std::string detail;
    if (has_range_correction) {
      auto status = scf::enqueue_prepared_cuda_rsh_values(
          provider, *range_correction, density, nullptr, matrix, j, exchange, nullptr,
          range_exchange, nullptr, jk_error, range_jk_error, detail);
      if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
        status = scf::enqueue_prepared_cuda_fock(provider, density, nullptr, matrix, j, exchange,
                                                 nullptr, jk_error, false, detail);
        if (status == GENERATIVEQC_STATUS_SUCCESS)
          status = scf::enqueue_prepared_cuda_exchange_correction(
              provider, *range_correction, density, nullptr, matrix, range_exchange, nullptr,
              range_jk_error, detail);
      }
      check(status, detail);
    } else {
      check(enqueue_primary_jk(density, nullptr, false, detail), detail);
    }
    CudaXcView potential;
    if (!device_nonlocal) {
      potential = xc->enqueue_replay_body(density, elements);
    } else {
      auto& nonlocal_source = nonlocal_xc ? *nonlocal_xc : *xc;
      if (nonlocal_xc) potential = xc->enqueue_replay_body(density, elements);
      const auto nonlocal_potential = nonlocal_source.enqueue_replay_density_features(
          density, elements, nonlocal_raw_density, nonlocal_raw_gradient);
      if (!nonlocal_xc) potential = nonlocal_potential;
      const auto quadrature = nonlocal_source.grid_view();
      run_resident_nonlocal_cuda([&] {
        nlc::enqueue_vv10_molecular_domain_cuda(
            stream, nonlocal_grid.point_count(), generated::kMolecularVv10DensityThreshold,
            quadrature.weights, nonlocal_raw_density, nonlocal_raw_gradient,
            nonlocal_effective_weights, nonlocal_effective_density, nonlocal_effective_gradient,
            nonlocal_domain_error);
      });
      enqueue_nonlocal_pairs(quadrature.points);
      nonlocal_source.enqueue_replay_nonlocal_potential(nonlocal_effective_weights,
                                                        nonlocal_effective_gradient, nonlocal_vrho,
                                                        nonlocal_vsigma, nonlocal_energy);
      if (nonlocal_xc) merge_grid_views(potential, nonlocal_potential);
    }
    cuda_ks_detail::assemble_fock(stream, n, spins, hcore, j, exchange, exchange_coefficient,
                                  range_exchange, range_exchange_coefficient, potential.potential,
                                  enabled, fock);
    check(cudaGetLastError());
    const auto blocks = static_cast<unsigned>((elements + 127) / 128);
    const auto multiply = [&](const double* a, bool a_spin, bool transpose, const double* b,
                              bool b_spin, const std::uint8_t* mask, double* c) {
      multiply_spin(spins, a, a_spin, transpose, b, b_spin, mask, c);
    };
    multiply(fock, true, false, density, true, enabled, tmp1);
    multiply(tmp1, true, false, overlap, false, enabled, residual);
    multiply(overlap, false, false, density, true, enabled, tmp1);
    multiply(tmp1, true, false, fock, true, enabled, tmp2);
    launch_subtract_matrix_batches_kernel(blocks, 128, 0, stream, 1, spins, n, tmp2, enabled,
                                          residual);
    check(cudaGetLastError());
    if (ordered_diis_gram) {
      launch_update_diis_cached_gram(1, 32, 0, stream, 1, n, spins, history, fock, residual,
                                     enabled, fock_history, residual_history, gram, weights,
                                     history_count, history_head, effective, raw_gram, true);
    } else {
      const bool incremental_gram = incremental_diis_gram;
      if (incremental_gram)
        check(launch_diis_pending_gram(stream, 1, n, spins, history, residual, residual_history,
                                       enabled, history_count, history_head, raw_gram));
      launch_update_diis_kernel(1, 32, 0, stream, 1, n, spins, history, fock, residual, enabled,
                                fock_history, residual_history, gram, weights, history_count,
                                history_head, effective, true, false, nullptr, 0, raw_gram);
    }
    check(cudaGetLastError());
    multiply(effective, true, false, x, false, enabled, tmp1);
    multiply(x, false, true, tmp1, true, enabled, tmp2);
    check(eigensolver->launch(spins, tmp2, effective, eigenvalues, solver_info, spin_enabled),
          "CUDA KS eigensolver launch failed");
    multiply(x, false, false, tmp2, true, enabled, tmp1);
    build_density(tmp1, nullptr, enabled, proposal);
    check(cudaGetLastError());
    auto* record = scalar_records + slot;
    cuda_ks_detail::diagnostics(
        stream, n, spins, density, proposal, residual, hcore, overlap, j, exchange,
        exchange_coefficient, range_exchange, range_exchange_coefficient, potential.totals,
        potential.error, jk_error, range_jk_error,
        device_nonlocal ? nonlocal_domain_error : nullptr,
        device_nonlocal ? nonlocal_pair_error : nullptr, solver_info, enabled, record);
    check(cudaGetLastError());
    cuda_ks_detail::advance(stream, n, spins, provider.one_electron().nuclear_repulsion,
                            static_cast<int>(occupations[0]), static_cast<int>(occupations[1]),
                            options.energy_tolerance, options.density_tolerance,
                            options.max_iterations, warm_updates, record, control, proposal,
                            density, warm, enabled, spin_enabled);
    check(cudaGetLastError());
  }

  void enqueue_device() {
    current_device();
    if (!is_active || is_pending) throw std::logic_error("CUDA KS iteration state is not ready");
    is_pending = true;
    pending_iterations = 0;
    try {
      const unsigned width = submission_width();
      const unsigned remaining = options.max_iterations - output.iterations;
      if (generation > std::numeric_limits<std::uint64_t>::max() - width)
        throw std::overflow_error("CUDA KS density generation exhausted");
      pending_iterations =
          solver_region_executor.submit(solver_region_binding(), width, remaining, false,
                                        [&](unsigned slot) { enqueue_one(slot); });
      movement.submitted_iterations += pending_iterations;
      // The replay body has no host publication side effects. Publish exactly
      // once for the physical warmup/capture/replay/fallback selected by the
      // shared runtime, even when capture internally probes the body twice.
      for (unsigned slot = 0; slot < pending_iterations; ++slot) {
        const auto submitted_generation = ++generation;
        xc->publish_submitted_generation(submitted_generation);
        if (nonlocal_xc) nonlocal_xc->publish_submitted_generation(submitted_generation);
        pending_generations[slot] = submitted_generation;
      }
    } catch (...) {
      cudaStreamSynchronize(stream);
      ++movement.synchronizations;
      solver_region_executor.invalidate();
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      device_chunk_region.mark_failure("CUDA KS device chunk submission failed");
      throw;
    }
  }

  bool finish_device() {
    current_device();
    if (!is_pending || pending_iterations == 0)
      throw std::logic_error("no pending CUDA KS iteration chunk");
    std::array<cuda_ks_detail::Scalars, kCudaKsChunkCapacity> physical{};
    cuda_ks_detail::Control device_control{};
    const unsigned submitted = pending_iterations;
    try {
      check(cudaMemcpyAsync(physical.data(), scalar_records, submitted * sizeof(physical[0]),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaMemcpyAsync(&device_control, control, sizeof(device_control),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      solver_region_executor.invalidate();
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      device_chunk_region.mark_failure("CUDA KS device chunk completion failed");
      throw;
    }
    movement.scalar_d2h_bytes += submitted * sizeof(physical[0]) + sizeof(device_control);
    ++movement.synchronizations;
    ++movement.iteration_synchronizations;
    ++movement.iteration_chunks;
    solver_region_executor.checkpoint();
    if (device_control.iterations <= output.iterations ||
        device_control.iterations > output.iterations + submitted) {
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      solver_region_executor.invalidate();
      device_chunk_region.mark_failure("CUDA KS device chunk returned an invalid iteration count");
      throw std::runtime_error("CUDA KS device chunk returned an invalid iteration count");
    }
    const unsigned completed = device_control.iterations - output.iterations;
    movement.iterations += completed;
    auto& diagnostic = output.dft_diagnostic;
    for (unsigned slot = 0; slot < completed; ++slot) {
      const auto& item = physical[slot];
      ++output.iterations;
      ++output.fock_builds;
      diagnostic.components = {provider.one_electron().nuclear_repulsion, item.one_electron,
                               item.hartree, item.xc, item.exact_exchange};
      diagnostic.physical_residual = item.residual;
      diagnostic.electrons = {item.electrons[0], item.electrons[1]};
      diagnostic.density_change = item.density_change;
      output.physical_residual_rms = item.residual_rms;
      output.energy = diagnostic.components.total();
      output.energy_change = item.energy_change;
      output.density_rms = item.density_rms;
      diagnostic.history.push_back({output.iterations, diagnostic.components, item.energy_change,
                                    item.density_change, item.residual, diagnostic.electrons,
                                    false});
    }
    is_pending = false;
    pending_iterations = 0;
    is_failed = device_control.failed != 0;
    is_active = device_control.active != 0;
    output.converged = device_control.converged != 0;
    if (output.converged && warm_updates) {
      warm_ready = true;
      warm_energy = physical[completed - 1].electronic_energy;
      warm_energy_correction = physical[completed - 1].electronic_energy_correction;
    }
    if (output.converged) {
      final_state_ready = true;
      final_generation = pending_generations[completed - 1U];
    }
    if (is_failed)
      device_chunk_region.mark_failure("CUDA KS device control reported failure");
    else
      device_chunk_region.mark_success();
    return is_active;
  }

  void enqueue() {
    try {
      if (device_chunk_mode)
        enqueue_device();
      else
        enqueue_legacy();
    } catch (...) {
      retire_phase_source(false);
      occupied_fitted_factor_ready = false;
      pending_fitted_occupied = false;
      final_fitted_projection_ready = false;
      invalidate_warm_orbitals();
      throw;
    }
  }
  bool finish() {
    // A partial density/frame copy or rejected iteration cannot lend the old
    // intermediate frame. Preserve the independent last-good warm density.
    bool auditing_phase = false;
    try {
      const bool active = device_chunk_mode ? finish_device() : finish_legacy();
      if (!active) {
        auditing_phase = !is_failed && phase_source && phase_source->active();
        if (is_failed || !output.converged)
          retire_phase_source(false);
        else
          audit_phase_source();
        if (auditing_phase && output.converged && warm_updates) {
          // Preserve the accepted source-schedule energy before preparing the
          // independent incumbent fallback. Both refer to the same warm D.
          if (range_source_active()) {
            range_warm_energy = warm_energy;
            range_warm_energy_correction = warm_energy_correction;
          }
          rebase_warm_energy();
        }
        auditing_phase = false;
      }
      if (is_failed) {
        occupied_fitted_factor_ready = false;
        pending_fitted_occupied = false;
        final_fitted_projection_ready = false;
        invalidate_warm_orbitals();
      }
      return active;
    } catch (...) {
      // Preparation at the mixed-to-strict handoff may throw before a
      // lower-level iteration guard owns the failure. Every completion error
      // must end the trajectory, including a partially published source lease.
      (void)cudaStreamSynchronize(stream);
      is_pending = is_active = false;
      is_failed = true;
      output.converged = final_state_ready = false;
      retire_phase_source(false);
      if (auditing_phase) clear_warm_state();
      occupied_fitted_factor_ready = false;
      pending_fitted_occupied = false;
      final_fitted_projection_ready = false;
      invalidate_warm_orbitals();
      throw;
    }
  }

  /** Anchor the reusable density's energy to the provider used by warm solves.
   * A phase source and the incumbent generated J/K have different FP64
   * summation/screening schedules. Comparing their energies can fail a strict
   * sub-ulp gate even at identical D. Evaluate the incumbent once at that D,
   * retaining its already-qualified XC. This is cache metadata, not a new
   * SCF proposal or convergence waiver: the next warm iteration must still
   * pass every physical gate. Keep the returned resident F[D] untouched.
   */
  void rebase_warm_energy() {
    runtime::df_progress::Scope trace("cuda_ks_warm_energy_rebase", "cuda_device");
    std::string status_detail;
    // Warm replay retires optional values before selecting a new source. Its
    // baseline must therefore use the incumbent without destroying the final
    // solve's value lease or changing the returned physical F[D]/XC frame.
    check(enqueue_primary_jk(density, spins == 2 ? density + matrix : nullptr, false, status_detail,
                             false),
          status_detail);
    // A warm replay uses the incumbent radial provider too. Rebase both raw
    // exchange terms, with their own coefficients, at this unchanged density.
    if (has_range_correction)
      enqueue_range_correction(density, spins == 2 ? density + matrix : nullptr, status_detail,
                               false);
    ++movement.warm_energy_rebase_fock_actions;
    ++output.fock_builds;
    ++output.precision.post_scf_fock_builds;
    output.precision_work.events.push_back(
        {scf::PrecisionWorkEventKind::PostScfFock, scf::PrecisionWorkPhase::Finalization,
         static_cast<std::uint64_t>(output.precision_work.events.size()), output.iterations, owner,
         solve_epoch, final_generation});
    for (const auto kind :
         {scf::PrecisionOperatorKind::CoulombJ, scf::PrecisionOperatorKind::ExchangeK,
          scf::PrecisionOperatorKind::Diagnostics})
      if (kind != scf::PrecisionOperatorKind::ExchangeK || has_exchange)
        record_precision_operator(kind, scf::PrecisionDtype::Fp64,
                                  scf::PrecisionArithmeticMode::Strict);
    if (has_range_correction)
      record_precision_operator(scf::PrecisionOperatorKind::ExchangeK, scf::PrecisionDtype::Fp64,
                                scf::PrecisionArithmeticMode::Strict);
    const auto potential =
        options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused
            ? xc->view(generation)
            : CudaXcView{generation, n, spins, nullptr, staged_xc_totals, staged_xc_error, stream};
    // Only the compensated energy is consumed. Proposal/residual still belong
    // to the resident final state and must not be exported as incumbent gates.
    cuda_ks_detail::diagnostics(stream, n, spins, density, proposal, residual, hcore, overlap, j,
                                exchange, exchange_coefficient, range_exchange,
                                range_exchange_coefficient, potential.totals, potential.error,
                                jk_error, range_jk_error, nullptr, nullptr, solver_info,
                                final_enabled, scalar_records);
    check(cudaGetLastError());
    cuda_ks_detail::Scalars rebased{};
    check(
        cudaMemcpyAsync(&rebased, scalar_records, sizeof(rebased), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    movement.scalar_d2h_bytes += sizeof(rebased);
    ++movement.synchronizations;
    if (rebased.failure || !std::isfinite(rebased.electronic_energy) ||
        !std::isfinite(rebased.electronic_energy_correction))
      throw std::runtime_error("CUDA KS warm energy rebase failed");
    runtime::df_progress::Scope::number("fock_actions", 1);
    if (trace.enabled()) {
      const auto label_energy = [](const char* key, double value) {
        char text[64]{};
        std::snprintf(text, sizeof(text), "%.17e", value);
        runtime::df_progress::Scope::label(key, text);
      };
      label_energy("energy_shift_Eh",
                   detail::electronic_energy_change(rebased.electronic_energy,
                                                    rebased.electronic_energy_correction,
                                                    warm_energy, warm_energy_correction));
      label_energy("exact_exchange_shift_Eh",
                   rebased.exact_exchange - output.dft_diagnostic.components.exact_exchange);
      label_energy("xc_shift_Eh", rebased.xc - output.dft_diagnostic.components.xc);
    }
    warm_energy = rebased.electronic_energy;
    warm_energy_correction = rebased.electronic_energy_correction;
    trace.finish("completed");
  }

  void enqueue_range_correction(const double* alpha_density, const double* beta_density,
                                std::string& detail, bool use_source = true,
                                bool mixed_exchange = false) {
    if (!has_range_correction || !range_correction)
      throw std::logic_error("CUDA KS range correction is unavailable");
    pending_mixed_range_resident = false;
    if (use_source && range_source_active() && mixed_exchange == phase_source->mixed_recurrence()) {
      // Cached FP32 sources are mixed applications, not new recurrences. Clear
      // the per-action counter and preserve their source storage in the census.
      if (mixed_exchange)
        check(cudaMemsetAsync(mixed_coulomb_work_counter() + 1, 0, sizeof(std::uint64_t), stream));
      check(phase_source->enqueue(range_correction->spec, alpha_density, nullptr, range_exchange,
                                  range_jk_error, range_correction->screening_tolerance, detail,
                                  beta_density, spins == 2 ? range_exchange + matrix : nullptr),
            detail);
      pending_mixed_range_resident = mixed_exchange;
      return;
    }
    if (!range_provider) {
      check(scf::enqueue_prepared_cuda_exchange_correction(
                provider, *range_correction, alpha_density, beta_density, matrix, range_exchange,
                spins == 2 ? range_exchange + matrix : nullptr, range_jk_error, detail,
                mixed_exchange, mixed_exchange ? mixed_coulomb_work_counter() + 1 : nullptr),
            detail);
      return;
    }

    const auto correction_stream = range_fock_binding.stream;
    bool submitted = false;
    try {
      range_input_ready.record(stream);
      check(cudaStreamWaitEvent(correction_stream, range_input_ready.get(), 0));
      submitted = true;
      const auto status = scf::enqueue_prepared_cuda_exchange_correction(
          *range_provider, *range_correction, alpha_density, beta_density, matrix, range_exchange,
          spins == 2 ? range_exchange + matrix : nullptr, range_jk_error, detail, mixed_exchange,
          mixed_exchange ? mixed_coulomb_work_counter() + 1 : nullptr);
      if (status != GENERATIVEQC_STATUS_SUCCESS) {
        (void)cudaStreamSynchronize(correction_stream);
        check(status, detail);
      }
      range_output_ready.record(correction_stream);
      check(cudaStreamWaitEvent(stream, range_output_ready.get(), 0));
    } catch (...) {
      if (submitted) (void)cudaStreamSynchronize(correction_stream);
      throw;
    }
  }

  /** Reuse only bitwise-identical pair ingredients on this fixed-grid owner.
   * Semilocal XC, J/K, AO assembly, physical residual and convergence all run
   * anew. A failed prior pair/domain generation or schedule change is a miss. */
  void enqueue_nonlocal_pairs(const double* points, bool mixed_pair_arithmetic = false) {
    const auto* selection = std::getenv("GENERATIVEQC_VV10_PAIR_SCHEDULE");
    const std::string schedule = selection ? selection : "";
    const int* mismatch = nullptr;
    run_resident_nonlocal_cuda([&] {
      if (nonlocal_cache_enabled && nonlocal_cache_ready && schedule == nonlocal_cache_schedule &&
          nonlocal_cache_mixed == mixed_pair_arithmetic) {
        nlc::enqueue_vv10_force_key_compare_cuda(
            stream, nonlocal_grid.point_count(), nonlocal_correlation->parameters(),
            nonlocal_effective_weights, nonlocal_effective_density, nonlocal_effective_gradient,
            nonlocal_workspace, nonlocal_domain_error, nonlocal_pair_error,
            nonlocal_cache_mismatch);
        mismatch = nonlocal_cache_mismatch;
      }
      if (nonlocal_cache_enabled)
        nlc::enqueue_vv10_cache_work_cuda(stream, mismatch, nonlocal_cache_counters);
    });
    nonlocal_cache_ready = false;
    run_resident_nonlocal_cuda([&] {
      nlc::enqueue_vv10_cuda_device(
          nonlocal_layout, nonlocal_correlation->parameters(), device, stream, points,
          nonlocal_effective_weights, nonlocal_effective_density, nonlocal_effective_gradient,
          nonlocal_workspace, nonlocal_layout.workspace_bytes, nonlocal_energy, nonlocal_vrho,
          nonlocal_vsigma, nullptr, nullptr, nonlocal_pair_error, mismatch, mixed_pair_arithmetic,
          mixed_pair_arithmetic ? mixed_coulomb_work_counter() + 2 : nullptr);
    });
    if (nonlocal_cache_enabled) {
      run_resident_nonlocal_cuda([&] {
        nlc::enqueue_vv10_store_force_key_cuda(
            stream, nonlocal_grid.point_count(), nonlocal_effective_density, nonlocal_workspace,
            nonlocal_domain_error, nonlocal_domain_error, nonlocal_pair_error);
      });
      nonlocal_cache_schedule = schedule;
      nonlocal_cache_mixed = mixed_pair_arithmetic;
      nonlocal_cache_ready = true;
    }
  }

  /** Both views belong exclusively to this owner and the same ordered stream.
   * Only E/V are additive: electron-count diagnostics retain the XC rule. */
  void merge_grid_views(const CudaXcView& semilocal, const CudaXcView& nonlocal) {
    if (semilocal.stream != stream || nonlocal.stream != stream || semilocal.nao != n ||
        nonlocal.nao != n || semilocal.spins != spins || nonlocal.spins != spins)
      throw std::logic_error("independent quadrature views do not share the KS frame");
    cuda_ks_detail::merge_grid_contributions(
        stream, elements, const_cast<double*>(semilocal.potential),
        const_cast<double*>(semilocal.totals), const_cast<int*>(semilocal.error),
        nonlocal.potential, nonlocal.totals, nonlocal.error);
    check(cudaGetLastError());
  }

  void merge_nonlocal_grid(std::uint64_t next_generation) {
    if (nonlocal_xc)
      merge_grid_views(xc->view(next_generation), nonlocal_xc->view(next_generation));
  }

  CudaXcView stage_xc(std::uint64_t next_generation,
                      generativeqc::runtime::PrecisionPhase phase =
                          generativeqc::runtime::PrecisionPhase::StrictAudit) {
    runtime::cuda_trace::TraceOperation trace("cuda_ks_xc_shared", stream, {1, n, 0, false, false});
    if (options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused) {
      if (!xc) throw std::logic_error("device-fused XC owner is unavailable");
      if (!device_nonlocal || nonlocal_seed_stage) {
        xc->enqueue(density, elements, next_generation, phase);
        return xc->view(next_generation);
      }
      // Keep AO/semilocal work separate from the quadratic nonlocal pair sweep
      // in opt-in diagnostics. Disabled tracing adds no events or fences.
      auto& nonlocal_source = nonlocal_xc ? *nonlocal_xc : *xc;
      if (nonlocal_xc) xc->enqueue(density, elements, next_generation, phase);
      runtime::cuda_trace::trace_call("semilocal_density_features", stream, [&] {
        nonlocal_source.enqueue_density_features(
            density, elements, next_generation, nonlocal_raw_density, nonlocal_raw_gradient, phase);
      });
      const auto quadrature = nonlocal_source.grid_view();
      if (nonlocal_frozen_stage) {
        // Keep the anchor gradient, screening mask and scalar pair derivatives
        // together: their AO assembly is precisely V_nlc[D_anchor]. Only raw
        // current-density features above are overwritten by this inner step.
        // Its energy is provisional and can never satisfy the target gate.
        nonlocal_source.enqueue_nonlocal_potential(next_generation, nonlocal_effective_weights,
                                                   nonlocal_effective_gradient, nonlocal_vrho,
                                                   nonlocal_vsigma, nonlocal_energy);
        merge_nonlocal_grid(next_generation);
        return xc->view(next_generation);
      }
      runtime::cuda_trace::TraceRegion domain_trace("nonlocal_density_domain", stream);
      run_resident_nonlocal_cuda([&] {
        nlc::enqueue_vv10_molecular_domain_cuda(
            stream, nonlocal_grid.point_count(), generated::kMolecularVv10DensityThreshold,
            quadrature.weights, nonlocal_raw_density, nonlocal_raw_gradient,
            nonlocal_effective_weights, nonlocal_effective_density, nonlocal_effective_gradient,
            nonlocal_domain_error);
      });
      domain_trace.finish();
      runtime::cuda_trace::TraceRegion pair_trace("nonlocal_pair_features", stream);
      enqueue_nonlocal_pairs(
          quadrature.points,
          phase == generativeqc::runtime::PrecisionPhase::Admitted && pending_mixed_nonlocal);
      pair_trace.finish();
      runtime::cuda_trace::trace_call("nonlocal_ao_potential", stream, [&] {
        nonlocal_source.enqueue_nonlocal_potential(next_generation, nonlocal_effective_weights,
                                                   nonlocal_effective_gradient, nonlocal_vrho,
                                                   nonlocal_vsigma, nonlocal_energy);
      });
      merge_nonlocal_grid(next_generation);
      return xc->view(next_generation);
    }
    const auto bytes = elements * sizeof(double);
    check(cudaMemcpyAsync(host_xc_density.data(), density, bytes, cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    movement.xc_host_d2h_bytes += bytes;
    ++movement.xc_host_synchronizations;
    ++movement.synchronizations;

    host_xc_totals.fill(0.0);
    if (spins == 1) {
      const auto& executor =
          kHostXcExecutors.at(semilocal_family_code(semilocal_family_from_code(functional)));
      if (!executor.rks) throw std::runtime_error("no registered host RKS XC evaluator");
      const auto value = executor.rks(basis, grid, host_xc_density, xc_layout.tile_points, {});
      if (value.potential.size() != matrix)
        throw std::runtime_error("host-unfused RKS XC potential size changed");
      std::copy(value.potential.begin(), value.potential.end(), host_xc_potential.begin());
      host_xc_totals = {value.energy, 0.5 * value.electrons, 0.5 * value.electrons};
      if (nonlocal_correlation) {
        const auto nonlocal =
            nlc::integrate_vv10_rks(basis, nonlocal_grid, host_xc_density, *nonlocal_correlation,
                                    xc_layout.tile_points, {}, nonlocal_domain);
        if (nonlocal.potential.size() != matrix)
          throw std::runtime_error("host-unfused RKS nonlocal potential size changed");
        for (std::size_t i = 0; i < matrix; ++i) host_xc_potential[i] += nonlocal.potential[i];
        host_xc_totals[0] += nonlocal.energy;
      }
    } else {
      std::copy_n(host_xc_density.begin(), matrix, host_xc_alpha.begin());
      std::copy_n(host_xc_density.begin() + matrix, matrix, host_xc_beta.begin());
      const auto& executor =
          kHostXcExecutors.at(semilocal_family_code(semilocal_family_from_code(functional)));
      if (!executor.uks) throw std::runtime_error("no registered host UKS XC evaluator");
      const auto value =
          executor.uks(basis, grid, host_xc_alpha, host_xc_beta, xc_layout.tile_points);
      if (value.potential[0].size() != matrix || value.potential[1].size() != matrix)
        throw std::runtime_error("host-unfused UKS XC potential size changed");
      std::copy(value.potential[0].begin(), value.potential[0].end(), host_xc_potential.begin());
      std::copy(value.potential[1].begin(), value.potential[1].end(),
                host_xc_potential.begin() + matrix);
      host_xc_totals = {value.energy, value.electrons[0], value.electrons[1]};
      if (nonlocal_correlation) {
        const auto nonlocal =
            nlc::integrate_vv10_uks(basis, nonlocal_grid, host_xc_alpha, host_xc_beta,
                                    *nonlocal_correlation, xc_layout.tile_points, nonlocal_domain);
        if (nonlocal.potential[0].size() != matrix || nonlocal.potential[1].size() != matrix)
          throw std::runtime_error("host-unfused UKS nonlocal potential size changed");
        for (std::size_t i = 0; i < matrix; ++i) {
          host_xc_potential[i] += nonlocal.potential[0][i];
          host_xc_potential[matrix + i] += nonlocal.potential[1][i];
        }
        host_xc_totals[0] += nonlocal.energy;
      }
    }

    check(cudaMemcpyAsync(tmp1, host_xc_potential.data(), bytes, cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(staged_xc_totals, host_xc_totals.data(), sizeof(host_xc_totals),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(staged_xc_error, &host_xc_error, sizeof(host_xc_error),
                          cudaMemcpyHostToDevice, stream));
    movement.xc_host_h2d_bytes += bytes + sizeof(host_xc_totals) + sizeof(host_xc_error);
    return {next_generation, n, spins, tmp1, staged_xc_totals, staged_xc_error, stream};
  }

  /** Allocate only after mandatory preparation. Refusing this optional lease
   * must retain the ordinary Fast solve, not turn memory pressure into a new
   * endpoint failure. A host reservation or non-allocation error still fails. */
  void prepare_fast_anchor_storage() {
    if (!fast_anchor_eligible) return;
    const auto bytes = sum(product(matrix, 5 * sizeof(double)), sizeof(double));
    bool host_oom = false;
    const auto status = runtime::resource_cuda_malloc(&fast_anchor_arena, bytes, &host_oom);
    if (status == cudaErrorMemoryAllocation && !host_oom) {
      const auto pending = cudaGetLastError();
      if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) check(pending);
      fast_anchor_eligible = false;
      ++movement.fast_anchor_allocation_fallbacks;
      return;
    }
    check(status);
    auto* storage = static_cast<double*>(fast_anchor_arena);
    fast_anchor_density = storage;
    fast_delta_density = storage + matrix;
    fast_anchor_j = storage + 2 * matrix;
    fast_anchor_exchange = storage + 3 * matrix;
    fast_anchor_range_exchange = storage + 4 * matrix;
    fast_delta_maximum = storage + 5 * matrix;
    resource.state_device_bytes = sum(resource.state_device_bytes, bytes);
  }

  /** The idle force consumer never borrows the optional cold-SCF anchor.
   * Release it only after the producer stream is drained. A later cold solve
   * keeps the ordinary Fast path; cumulative execution evidence is unchanged. */
  std::size_t retire_fast_anchor_storage() {
    if (!fast_anchor_arena) return 0;
    const auto bytes = sum(product(matrix, 5 * sizeof(double)), sizeof(double));
    check(runtime::resource_cuda_free(fast_anchor_arena));
    fast_anchor_arena = nullptr;
    fast_anchor_density = fast_delta_density = fast_anchor_j = fast_anchor_exchange =
        fast_anchor_range_exchange = fast_delta_maximum = nullptr;
    fast_anchor_eligible = fast_anchor_active = fast_anchor_ready = pending_fast_delta = false;
    resource.state_device_bytes -= bytes;
    return bytes;
  }

  /** Reconstruct linear operators from one fixed strict anchor. Full targets
   * always refresh FP64 J/K; nonlinear XC continues to consume total density.
   * Keeping the anchor fixed avoids chaining screened or rounded updates. */
  const double* prepare_fast_anchor_jk_density() {
    pending_fast_delta =
        fast_anchor_ready && !final_closure && (nonlocal_seed_stage || nonlocal_frozen_stage);
    if (!pending_fast_delta) {
      pending_mixed_coulomb = pending_mixed_exchange = pending_mixed_range_exchange = false;
      return density;
    }
    const auto blocks = static_cast<unsigned>((elements + 127) / 128);
    launch_prepare_incremental_density_kernel(blocks, 128, 0, stream, elements, density,
                                              fast_anchor_density, fast_delta_density,
                                              fast_delta_maximum);
    check(cudaGetLastError());
    return fast_delta_density;
  }

  void finalize_fast_anchor_jk_components() {
    const auto blocks = static_cast<unsigned>((elements + 127) / 128);
    if (pending_fast_delta) {
      launch_add_matrix_kernel(blocks, 128, 0, stream, matrix, fast_anchor_j, j);
      launch_add_matrix_kernel(blocks, 128, 0, stream, elements, fast_anchor_exchange, exchange);
      launch_add_matrix_kernel(blocks, 128, 0, stream, elements, fast_anchor_range_exchange,
                               range_exchange);
    } else {
      launch_copy_matrix_kernel(blocks, 128, 0, stream, elements, density, fast_anchor_density);
      launch_copy_matrix_kernel(blocks, 128, 0, stream, matrix, j, fast_anchor_j);
      launch_copy_matrix_kernel(blocks, 128, 0, stream, elements, exchange, fast_anchor_exchange);
      launch_copy_matrix_kernel(blocks, 128, 0, stream, elements, range_exchange,
                                fast_anchor_range_exchange);
    }
    check(cudaGetLastError());
  }

  /** Enqueued copies are not accepted anchors until physical validation
   * succeeds. A failed attempt must not lend stale buffers to the next one. */
  void finish_fast_anchor_iteration() {
    if (!fast_anchor_active) return;
    fast_anchor_ready = !is_failed;
    if (is_failed)
      ++movement.fast_anchor_invalidations;
    else if (pending_fast_delta)
      ++movement.fast_delta_builds;
    else
      ++movement.fast_anchor_builds;
  }

  bool incremental_delta_admitted() const noexcept {
    if (!incremental_direct_jk || !incremental_anchored || incremental_energy_refinement ||
        final_closure || strict_refinement)
      return false;
    const auto interval = ks_incremental_proposal_interval(
        incremental_proposals_active, incremental_direct_jk_policy.requested_rebuild_interval,
        incremental_direct_jk_policy.effective_rebuild_interval);
    if (interval != 0U && incremental_delta_updates >= interval) return false;
    if (incremental_direct_jk_policy.density_rms_threshold > 0.0 &&
        (!output.iterations || !std::isfinite(output.density_rms) ||
         output.density_rms > incremental_direct_jk_policy.density_rms_threshold))
      return false;
    return true;
  }

  /** Screened full and delta builds need not omit the same integrals. Their
   * reconstructed energies can therefore alternate above the energy tolerance
   * even after D and the physical residual are stationary. Finish energy
   * convergence with the full target operator instead of waiting for that
   * alternating energy gate before permitting a full-density audit. Keep DIIS
   * and the original convergence tolerances; final physical qualification
   * still requires consecutive full-density builds. */
  void refine_incremental_energy(double density_change, double residual,
                                 double maximum_residual) noexcept {
    const double residual_tolerance = physical_stationarity_tolerance();
    const bool standard = standard_convergence_active();
    // Start the safer full-density tail as soon as the density and RMS gates
    // predict stationarity. Waiting for the AO maximum as well can spend the
    // last useful iteration on a delta, then require two additional full Focks.
    // The AO-maximum and canonicality gates still decide final acceptance;
    // selecting full arithmetic early is not a convergence claim.
    if (incremental_direct_jk && density_change < options.density_tolerance &&
        residual < residual_tolerance && (standard || maximum_residual < residual_tolerance)) {
      // A just-completed full F[D] is already the first physical energy in
      // this refinement epoch. Count that real build instead of discarding it
      // and demanding two more. A reconstructed delta energy never qualifies.
      // The Standard stop still requires a second full build and the complete
      // canonical final-state proof; explicit strict controls keep their policy.
      if (!incremental_energy_refinement && !pending_incremental_delta && standard)
        incremental_energy_full_builds = 1;
      incremental_energy_refinement = true;
    }
  }

  const double* prepare_incremental_jk_density() {
    if (fast_anchor_active) return prepare_fast_anchor_jk_density();
    pending_incremental_delta = false;
    if (!incremental_direct_jk) return density;
    pending_incremental_delta = incremental_delta_admitted();
    if (!pending_incremental_delta) {
      if (incremental_energy_refinement && !final_closure) ++incremental_energy_full_builds;
      if (final_closure) {
        ++output.incremental_direct_jk.post_scf_full_builds;
      } else {
        if (incremental_anchored) ++output.incremental_direct_jk.periodic_rebuilds;
        ++output.incremental_direct_jk.anchor_full_builds;
      }
      return density;
    }
    const auto blocks = static_cast<unsigned>((elements + 127) / 128);
    launch_prepare_incremental_density_kernel(blocks, 128, 0, stream, elements, density,
                                              incremental_anchor_density, incremental_delta_density,
                                              incremental_max_abs_delta_density);
    check(cudaGetLastError());
    ++output.incremental_direct_jk.delta_builds;
    return incremental_delta_density;
  }

  /** The qualifying RKS build already evaluates the full physical operator.
   * Reclassify that actual build as final validation rather than launching a
   * redundant correction and imposing a second noisy energy qualification. */
  void account_incremental_full_energy_finalization() noexcept {
    --output.incremental_direct_jk.anchor_full_builds;
    --output.incremental_direct_jk.periodic_rebuilds;
    ++output.incremental_direct_jk.post_scf_full_builds;
  }

  void finalize_incremental_jk_components() {
    if (fast_anchor_active) {
      finalize_fast_anchor_jk_components();
      return;
    }
    if (!incremental_direct_jk || final_closure) return;
    const auto matrix_blocks = static_cast<unsigned>((matrix + 127) / 128);
    const auto element_blocks = static_cast<unsigned>((elements + 127) / 128);
    if (pending_incremental_delta) {
      launch_add_matrix_kernel(matrix_blocks, 128, 0, stream, matrix, incremental_anchor_j, j);
      if (has_exchange)
        launch_add_matrix_kernel(element_blocks, 128, 0, stream, elements,
                                 incremental_anchor_exchange, exchange);
      if (has_range_correction)
        launch_add_matrix_kernel(element_blocks, 128, 0, stream, elements,
                                 incremental_anchor_range_exchange, range_exchange);
      check(cudaGetLastError());
      ++incremental_delta_updates;
    } else {
      incremental_delta_updates = 0;
    }
    launch_copy_matrix_kernel(element_blocks, 128, 0, stream, elements, density,
                              incremental_anchor_density);
    launch_copy_matrix_kernel(matrix_blocks, 128, 0, stream, matrix, j, incremental_anchor_j);
    if (has_exchange)
      launch_copy_matrix_kernel(element_blocks, 128, 0, stream, elements, exchange,
                                incremental_anchor_exchange);
    if (has_range_correction)
      launch_copy_matrix_kernel(element_blocks, 128, 0, stream, elements, range_exchange,
                                incremental_anchor_range_exchange);
    check(cudaGetLastError());
    incremental_anchored = true;
  }

  void enqueue_legacy() {
    current_device();
    if (!is_active || is_pending) throw std::logic_error("CUDA KS iteration state is not ready");
    if (generation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CUDA KS density generation exhausted");
    runtime::cuda_trace::TraceOperation trace("ks_iteration", stream,
                                              {1, n, 0, fitted_coulomb, false});
    runtime::cuda_trace::trace_counter("owner", owner);
    runtime::cuda_trace::trace_counter("solve_epoch", solve_epoch);
    runtime::cuda_trace::trace_counter("density_generation", generation + 1);
    runtime::cuda_trace::trace_counter("fock_build", output.fock_builds + 1);
    is_pending = true;  // Any partial CUDA submission is drained on failure.
    try {
      // Mixed history is only a bounded proposal accelerator. Every physical
      // energy/residual after handoff is rebuilt in FP64; six unsuccessful
      // strict proposals discard all provisional history and retain the
      // incumbent exact refinement. No mixed operator certifies publication.
      if (experimental_mixed_history && strict_refinement && refinement_iterations == 6) {
        check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
        check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
      }
      std::string detail;
      const auto iteration_precision = resolve_cuda_ks_iteration_precision(
          precision_schedule, strict_refinement,
          cuda_xc_execution_capabilities(xc_layout).mixed_density_contraction);
      pending_mixed_coulomb =
          iteration_precision.uses_lower_precision(cuda_ks_precision_region::kCoulombJ);
      // Pure-J proposals benefit from the generated strict shell consumer:
      // generic mixed recurrence repeats substantially more component work.
      // Select before submission so the work ledger reports actual FP64 J.
      // Hybrid/resident schedules retain their independently qualified policy.
      if (pending_mixed_coulomb && !has_exchange && !has_range_correction &&
          !(phase_source && phase_source->active()) &&
          scf::cuda_direct_jk_generated_coulomb_available(provider.cuda_direct_source(),
                                                          provider.strategy().spec))
        pending_mixed_coulomb = false;
      pending_mixed_exchange = has_exchange && iteration_precision.uses_lower_precision(
                                                   cuda_ks_precision_region::kExactExchange);
      pending_mixed_range_exchange =
          has_range_correction &&
          iteration_precision.uses_lower_precision(cuda_ks_precision_region::kRangeExchange);
      pending_mixed_nonlocal =
          device_nonlocal &&
          iteration_precision.uses_lower_precision(cuda_ks_precision_region::kNonlocalCorrelation);
      if (pending_mixed_nonlocal)
        check(cudaMemsetAsync(mixed_coulomb_work_counter() + 2, 0, 2 * sizeof(std::uint64_t),
                              stream));
      pending_mixed_point =
          iteration_precision.uses_lower_precision(cuda_ks_precision_region::kXcPointAlgebra);
      pending_mixed_contractions = xc && prepared_ao_work.point_ao_visits != 0 &&
                                   iteration_precision.uses_lower_precision(
                                       cuda_ks_precision_region::kXcMatrixContractions) &&
                                   cuda_xc_detail::mixed_matrix_contractions_supported(xc_layout);
      const auto density_phase = strict_refinement
                                     ? generativeqc::runtime::PrecisionPhase::StrictAudit
                                     : generativeqc::runtime::PrecisionPhase::Admitted;
      pending_mixed_density =
          xc && !xc->density_binding(density_phase).precision.arithmetic.is_strict_fp64();
      // Provider selection stays inside the prepared Fock facade. For a fitted
      // hybrid, arbitrary cold/imported seeds have no trusted canonical factor
      // and stay dense. Verified same-owner warm seeds may restore the exact C.
      // After a successful proposal becomes the current density,
      // tmp1 still owns the exact AO canonical C that generated that proposal;
      // borrow it on the next iteration without a host round trip.
      const bool use_occupied_fitted =
          fitted_exchange && occupied_fitted_factor_ready && occupied_fock_binding;
      runtime::cuda_trace::trace_counter("dense_k", fitted_exchange && !use_occupied_fitted);
      runtime::cuda_trace::trace_counter("occupied_k", use_occupied_fitted);
      runtime::cuda_trace::trace_counter(
          "factor_density_generation",
          use_occupied_fitted ? (output.iterations == 0 ? warm_factor_generation : generation + 1)
                              : 0);
      runtime::cuda_trace::trace_counter(
          "factor_solve_epoch",
          use_occupied_fitted ? (output.iterations == 0 ? warm_factor_epoch : solve_epoch) : 0);
      runtime::cuda_trace::TraceRegion jk_trace("ks_jk", stream);
      pending_fitted_occupied = use_occupied_fitted;
      pending_fitted_projection_scratch_generation = 0;
      const double* jk_density = prepare_incremental_jk_density();
      generativeqc_status jk_status = GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
      bool fused_rsh_values = false;
      if (has_range_correction && !range_source_active() && !pending_mixed_coulomb &&
          !use_occupied_fitted) {
        jk_status = scf::enqueue_prepared_cuda_rsh_values(
            provider, *range_correction, jk_density, spins == 2 ? jk_density + matrix : nullptr,
            matrix, j, exchange, has_exchange && spins == 2 ? exchange + matrix : nullptr,
            range_exchange, spins == 2 ? range_exchange + matrix : nullptr, jk_error,
            range_jk_error, detail);
        fused_rsh_values = jk_status == GENERATIVEQC_STATUS_SUCCESS;
        if (!fused_rsh_values && jk_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED)
          check(jk_status, detail);
      }
      if (!fused_rsh_values) {
        if (use_occupied_fitted) {
          const scf::PreparedCudaOccupiedFockInput occupied{
              tmp1, spins == 2 ? tmp1 + matrix : nullptr, occupations[0],
              spins == 2 ? occupations[1] : 0};
          jk_status = scf::enqueue_prepared_cuda_occupied_fock(
              provider, jk_density, spins == 2 ? jk_density + matrix : nullptr, matrix, occupied, j,
              exchange, spins == 2 ? exchange + matrix : nullptr, jk_error, detail);
        } else {
          jk_status = enqueue_primary_jk(jk_density, spins == 2 ? jk_density + matrix : nullptr,
                                         pending_mixed_coulomb, detail);
        }
      }
      check(jk_status, detail);
      const char* audit = std::getenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT");
      if (use_occupied_fitted && output.iterations == 0 && spins == 1 && audit &&
          std::strcmp(audit, "1") == 0)
        audit_warm_seed_exchange();
      if (use_occupied_fitted) {
        // Bind the write submitted for this density before another KS owner
        // can reuse the provider between enqueue_iteration and finish_iteration.
        const auto projection =
            scf::prepared_cuda_occupied_projection_binding(provider, occupations[0]);
        if (projection)
          pending_fitted_projection_scratch_generation = projection.scratch_generation;
      }
      if (fitted_exchange) {
        if (use_occupied_fitted)
          ++movement.fitted_occupied_exchange_builds;
        else
          ++movement.fitted_dense_exchange_builds;
      }
      if (has_range_correction && !fused_rsh_values)
        enqueue_range_correction(jk_density, spins == 2 ? jk_density + matrix : nullptr, detail,
                                 true, pending_mixed_range_exchange);
      finalize_incremental_jk_components();
      jk_trace.finish();
      mixed_precision_executed = mixed_precision_executed || pending_mixed_coulomb ||
                                 pending_mixed_density || pending_mixed_point ||
                                 pending_mixed_contractions || pending_mixed_range_exchange ||
                                 pending_mixed_nonlocal;
      pending_nonlocal_seed = nonlocal_seed_stage;
      pending_nonlocal_frozen = nonlocal_frozen_stage;
      runtime::cuda_trace::TraceRegion xc_trace("ks_xc", stream);
      const auto potential = stage_xc(++generation, density_phase);
      xc_trace.finish();
      pending_generations[0] = generation;
      ++movement.submitted_iterations;
      pending_iterations = 1;
      cuda_ks_detail::assemble_fock(stream, n, spins, hcore, j, exchange, exchange_coefficient,
                                    range_exchange, range_exchange_coefficient, potential.potential,
                                    enabled, fock);
      check(cudaGetLastError());
      const auto blocks = static_cast<unsigned>((elements + 127) / 128);
      const auto multiply = [&](const double* a, bool a_spin, bool transpose, const double* b,
                                bool b_spin, double* c) {
        multiply_spin(spins, a, a_spin, transpose, b, b_spin, enabled, c);
      };
      // Physical residual is FDS-SDF, using the unchanged CURRENT density.
      multiply(fock, true, false, density, true, tmp1);
      multiply(tmp1, true, false, overlap, false, residual);
      multiply(overlap, false, false, density, true, tmp1);
      multiply(tmp1, true, false, fock, true, tmp2);
      launch_subtract_matrix_batches_kernel(blocks, 128, 0, stream, 1, spins, n, tmp2, enabled,
                                            residual);
      check(cudaGetLastError());
      if (final_closure) {
        // Discard DIIS history and rebuild the closure proposal from F[D].
        // A stationary UKS occupation cycle still needs its virtual-space
        // shift, as on CPU. Export separately validates the unshifted F[D].
        check(cudaMemcpyAsync(effective, fock, elements * sizeof(double), cudaMemcpyDeviceToDevice,
                              stream));
      } else {
        if (ordered_diis_gram) {
          launch_update_diis_cached_gram(1, 32, 0, stream, 1, n, spins, history, fock, residual,
                                         enabled, fock_history, residual_history, gram, weights,
                                         history_count, history_head, effective, raw_gram, true);
        } else {
          const bool incremental_gram = incremental_diis_gram;
          if (incremental_gram)
            check(launch_diis_pending_gram(stream, 1, n, spins, history, residual, residual_history,
                                           enabled, history_count, history_head, raw_gram));
          launch_update_diis_kernel(1, 32, 0, stream, 1, n, spins, history, fock, residual, enabled,
                                    fock_history, residual_history, gram, weights, history_count,
                                    history_head, effective, true, false, nullptr, 0, raw_gram);
        }
        check(cudaGetLastError());
      }
      if (stabilize_occupations) {
        // Match the CPU stationary-cycle policy. The unit-occupation virtual
        // projector is S-SDS for each spin. Shift only the DIIS proposal;
        // physical F/D/residual and the history above remain unmodified.
        multiply(overlap, false, false, density, true, tmp1);
        multiply(tmp1, true, false, overlap, false, tmp2);
        cuda_ks_detail::stabilize_uks_proposal(stream, n, overlap, tmp2, enabled, effective);
        check(cudaGetLastError());
        ++movement.occupation_stabilized_proposals;
      }
      multiply(effective, true, false, x, false, tmp1);
      multiply(x, false, true, tmp1, true, tmp2);
      check(eigensolver->launch(spins, tmp2, effective, eigenvalues, solver_info, spin_enabled),
            "CUDA KS eigensolver launch failed");
      multiply(x, false, false, tmp2, true, tmp1);
      build_density(tmp1, nullptr, enabled, proposal);
      check(cudaGetLastError());
      cuda_ks_detail::diagnostics(
          stream, n, spins, density, proposal, residual, hcore, overlap, j, exchange,
          exchange_coefficient, range_exchange, range_exchange_coefficient, potential.totals,
          potential.error, jk_error, range_jk_error,
          device_nonlocal ? nonlocal_domain_error : nullptr,
          device_nonlocal ? nonlocal_pair_error : nullptr, solver_info, enabled, scalar_records);
      check(cudaGetLastError());
    } catch (...) {
      cudaStreamSynchronize(stream);
      ++movement.synchronizations;
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      throw;
    }
  }

  bool finish_legacy() {
    current_device();
    if (!is_pending) throw std::logic_error("no pending CUDA KS iteration");
    runtime::cuda_trace::TraceOperation trace("ks_finish", stream,
                                              {1, n, 0, fitted_coulomb, false});
    runtime::cuda_trace::trace_counter("owner", owner);
    runtime::cuda_trace::trace_counter("solve_epoch", solve_epoch);
    runtime::cuda_trace::trace_counter("density_generation", generation);
    cuda_ks_detail::Scalars physical{};
    std::uint64_t mixed_coulomb_recurrences{};
    std::uint64_t mixed_range_recurrences{};
    std::array<std::uint64_t, 2> nonlocal_arithmetic{};
    std::array<std::uint64_t, 3> nonlocal_work{};
    try {
      check(cudaMemcpyAsync(&physical, scalar_records, sizeof(physical), cudaMemcpyDeviceToHost,
                            stream));
      if (pending_mixed_coulomb)
        check(cudaMemcpyAsync(&mixed_coulomb_recurrences, mixed_coulomb_work_counter(),
                              sizeof(mixed_coulomb_recurrences), cudaMemcpyDeviceToHost, stream));
      if (pending_mixed_range_exchange)
        check(cudaMemcpyAsync(&mixed_range_recurrences, mixed_coulomb_work_counter() + 1,
                              sizeof(mixed_range_recurrences), cudaMemcpyDeviceToHost, stream));
      if (pending_mixed_nonlocal)
        check(cudaMemcpyAsync(nonlocal_arithmetic.data(), mixed_coulomb_work_counter() + 2,
                              sizeof(nonlocal_arithmetic), cudaMemcpyDeviceToHost, stream));
      if (incremental_direct_jk)
        check(cudaMemcpyAsync(&host_incremental_max_abs_delta_density,
                              incremental_max_abs_delta_density, sizeof(double),
                              cudaMemcpyDeviceToHost, stream));
      if (nonlocal_cache_enabled)
        check(cudaMemcpyAsync(nonlocal_work.data(), nonlocal_cache_counters, sizeof(nonlocal_work),
                              cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_pending = is_active = false;
      is_failed = true;
      pending_iterations = 0;
      throw;
    }
    movement.scalar_d2h_bytes +=
        sizeof(physical) + (pending_mixed_coulomb ? sizeof(mixed_coulomb_recurrences) : 0U) +
        (pending_mixed_range_exchange ? sizeof(mixed_range_recurrences) : 0U) +
        (pending_mixed_nonlocal ? sizeof(nonlocal_arithmetic) : 0U) +
        (incremental_direct_jk ? sizeof(host_incremental_max_abs_delta_density) : 0U);
    if (nonlocal_cache_enabled) movement.scalar_d2h_bytes += sizeof(nonlocal_work);
    if (incremental_direct_jk)
      output.incremental_direct_jk.max_abs_delta_density = host_incremental_max_abs_delta_density;
    ++movement.synchronizations;
    ++movement.iteration_synchronizations;
    ++movement.iteration_chunks;
    ++movement.iterations;
    runtime::cuda_trace::trace_counter("explicit_synchronizations", 1);
    is_pending = false;
    pending_iterations = 0;
    ++output.iterations;
    ++output.fock_builds;
    if (pending_nonlocal_seed || pending_nonlocal_frozen)
      ++nonlocal_seed_iterations;
    else if (device_nonlocal) {
      ++nonlocal_exact_iterations;
      ++nonlocal_exact_epoch_iterations;
    }
    if (pending_nonlocal_frozen) ++nonlocal_inner_iterations;
    if (device_nonlocal)
      output.dft_diagnostic.nonlocal_work = {
          1,
          nonlocal_seed_iterations,
          nonlocal_exact_iterations,
          nonlocal_work[0],
          nonlocal_work[1],
          nonlocal_cache_enabled ? nonlocal_work[2] : nonlocal_exact_iterations};
    const bool completed_mixed = pending_mixed_coulomb || pending_mixed_density ||
                                 pending_mixed_point || pending_mixed_range_resident ||
                                 pending_mixed_contractions || mixed_range_recurrences ||
                                 nonlocal_arithmetic[0];
    if (completed_mixed)
      ++output.precision.mixed_stage_fock_builds;
    else
      ++output.precision.strict_stage_fock_builds;
    record_fock_precision_work(mixed_coulomb_recurrences, mixed_range_recurrences,
                               nonlocal_arithmetic);
    if (mixed_precision_executed && !completed_mixed) ++refinement_iterations;
    output.precision.requested_mode = options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64);
    output.precision.effective_bits = mixed_precision_executed ? 32U : 64U;
    output.precision.strict_refinement_applied =
        mixed_precision_executed && refinement_iterations > 0;
    output.precision.refinement_iterations = refinement_iterations;
    auto& diagnostic = output.dft_diagnostic;
    diagnostic.components = {provider.one_electron().nuclear_repulsion, physical.one_electron,
                             physical.hartree, physical.xc, physical.exact_exchange};
    diagnostic.physical_residual = physical.residual;
    diagnostic.electrons = {physical.electrons[0], physical.electrons[1]};
    diagnostic.density_change = physical.density_change;
    output.physical_residual_rms = physical.residual_rms;
    output.energy = diagnostic.components.total();
    output.energy_change = detail::electronic_energy_change(
        physical.electronic_energy, physical.electronic_energy_correction, previous_energy,
        previous_energy_correction);
    output.density_rms = physical.density_rms;
    const bool new_target_epoch =
        (mixed_precision_executed && refinement_iterations == 1) ||
        (nonlocal_seed_iterations && nonlocal_exact_epoch_iterations == 1 &&
         !pending_nonlocal_seed && !pending_nonlocal_frozen);
    const bool first_strict_energy =
        new_target_epoch && std::isinf(output.energy_change) && output.energy_change > 0.0;
    const double history_energy_change = first_strict_energy ? -1.0 : output.energy_change;
    diagnostic.history.push_back({output.iterations, diagnostic.components, history_energy_change,
                                  physical.density_change, physical.residual, diagnostic.electrons,
                                  stabilize_occupations});
    diagnostic.history.back().nonlocal_operator_scope =
        pending_nonlocal_seed ? 1U : (pending_nonlocal_frozen ? 2U : 0U);
    // The kernel validates electronic components; their host-side sum with
    // the nuclear term must also be finite before any convergence/cache gate.
    is_failed = physical.failure != 0 || !std::isfinite(output.energy);
    for (unsigned s = 0; s < 2; ++s)
      if (std::abs(physical.electrons[s] - occupations[s]) > 1e-8) is_failed = true;
    finish_fast_anchor_iteration();
    if (!is_failed && incremental_direct_jk && pending_incremental_delta)
      ++output.incremental_direct_jk.anchor_updates;
    if (is_failed) {
      pending_incremental_delta = false;
      if (pending_mixed_coulomb || pending_mixed_density || pending_mixed_point ||
          pending_mixed_contractions || pending_mixed_range_exchange || pending_mixed_nonlocal) {
        // Any failed low-precision attempt retries the same density with the
        // strict target operator. Do not publish or cache the failed proposal.
        ++output.precision.execution_retries;
        record_precision_retry();
        strict_refinement = true;
        // A numerical retry must discard the provisional operator as well as
        // its arithmetic. Never omit NLC or retain a mixed frozen potential in
        // the fresh strict target evaluation at this unchanged density.
        nonlocal_outer_active = nonlocal_seed_stage = nonlocal_frozen_stage = false;
        nonlocal_exact_epoch_iterations = 0;
        // A failed proposal cannot certify reuse or lend FP32 storage to the
        // strict retry. Retire it before admitting a fresh FP64 source.
        if (phase_source && phase_source->mixed_recurrence()) retire_phase_source(false);
        prepare_phase_source(phase_source_cold);
        stabilize_occupations = false;
        final_closure = false;
        final_corrections = 0;
        is_failed = false;
        is_active = true;
        check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
        check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
        previous_energy = std::numeric_limits<double>::infinity();
        previous_energy_correction = 0.0;
        return true;
      }
      is_active = false;
      occupied_fitted_factor_ready = false;
      pending_fitted_occupied = false;
      final_fitted_projection_ready = false;
      return false;
    }
    // A stationary physical state can still alternate integer occupations.
    // Enable the same 0.1-Eh proposal shift as CPU UKS only after both physical
    // gates pass. A subsequent density-change gate must still pass to finish.
    if (!final_closure && spins == 2 && output.iterations > 1 &&
        output.energy_change < options.energy_tolerance &&
        physical.residual < std::min(1e-9, options.density_tolerance) &&
        physical.density_change >= options.density_tolerance)
      stabilize_occupations = true;
    const bool has_energy_history =
        output.iterations > 1 || (output.iterations == 1 && warm_energy_baseline);
    refine_incremental_energy(physical.density_change, physical.residual,
                              physical.maximum_residual);
    const bool full_energy_history = !incremental_direct_jk || incremental_energy_full_builds >= 2;
    bool converged = ordinary_ks_converged(
        {output.iterations, warm_energy_baseline, full_energy_history,
         !pending_nonlocal_seed && !pending_nonlocal_frozen &&
             (!nonlocal_seed_iterations || nonlocal_exact_epoch_iterations >= 2),
         output.energy_change, physical.density_change, physical.residual,
         physical.maximum_residual, options.energy_tolerance, options.density_tolerance,
         physical_stationarity_tolerance()});
    bool early_final_probe_ran = false;
    // The final contract admits a 1e-8 AO maximum but separately checks the
    // true F[D] canonical density. The 1e-9 spin-RMS and public E/D gates stay
    // intact. Optional/device/backend cases without this qualified ownership
    // retain the incumbent conservative stopping policy.
    const bool early_final_probe_eligible =
        early_final_probe_requested && spins == 1 && !budget_fock && !fitted_exchange &&
        !has_range_correction && !nonlocal_correlation && !incremental_direct_jk &&
        !precision_schedule.any_lower_precision() && configured_chunk_width() == 1 &&
        final_validation_partial && matrix_products.library_enabled() &&
        provider.system().ecp_terms.empty() &&
        std::none_of(provider.system().atoms.begin(), provider.system().atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; });
    // Prove the force consumer's complete physical/canonical state before
    // publishing convergence or a warm seed. Rejection resumes the intact
    // DIIS proposal and invalidates only the borrowed orbital scratch.
    if (converged && standard_convergence_active()) {
      early_final_probe_ran = true;
      converged = probe_current_final_state();
    } else if (!converged && early_final_probe_eligible && has_energy_history &&
               full_energy_history && output.energy_change < options.energy_tolerance &&
               physical.density_change < options.density_tolerance &&
               physical.residual < physical_stationarity_tolerance() &&
               physical.maximum_residual < std::min(1e-8, options.density_tolerance)) {
      early_final_probe_ran = true;
      converged = probe_current_final_state();
    }
    // Once the initialized target already passes every physical stationarity
    // gate, confirm its energy with another complete F[D] at that same D.
    // Applying a sub-tolerance proposal can otherwise inject summation noise
    // and repeat expensive changing-density sweeps without improving the
    // accepted physical state. The original energy gate still has to pass;
    // this is neither an acceptance shortcut nor a provisional-state replay.
    const auto nonlocal_confirmation =
        nonlocal_confirmation_tolerance(budget_stationarity, options.density_tolerance);
    const bool confirm_nonlocal_stationary_density =
        nonlocal_seed_iterations && !pending_nonlocal_seed && !pending_nonlocal_frozen &&
        physical.density_change < options.density_tolerance &&
        physical.residual < nonlocal_confirmation &&
        physical.maximum_residual < nonlocal_confirmation;
    const bool confirm_direct_stationary_density =
        spins == 1 && !has_exchange && !fitted_coulomb && !fitted_exchange &&
        !has_range_correction && !nonlocal_correlation && !incremental_direct_jk && !budget_fock &&
        !precision_schedule.any_lower_precision() && provider.system().ecp_terms.empty() &&
        std::none_of(provider.system().atoms.begin(), provider.system().atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; }) &&
        detail::stationary_density_needs_energy_confirmation(
            has_energy_history, output.energy_change, physical.density_change, physical.residual,
            physical.maximum_residual, options.energy_tolerance, options.density_tolerance,
            physical_stationarity_tolerance());
    // Keep the existing RMS diagnostic, but do not publish an energy-only state
    // that the shared final-state validator will reject on the AO maximum norm.
    // Budget Fock already requires consecutive fresh FP64 physical Focks and
    // authenticated canonical export. Do not add a stricter fixed-point sweep
    // after those budget gates pass; explicit strict callers retain closure.
    const bool budget_final_validation =
        budget_fock && budget_stationarity && options.density_tolerance >= 1e-4;
    const bool strict_final_closure = (budget_fock && !budget_final_validation) ||
                                      incremental_direct_jk || spins == 2 ||
                                      !provider.system().ecp_terms.empty();
    const bool mixed_stage = precision_schedule.any_lower_precision() && !strict_refinement;
    // Inner residuals only control how soon to refresh the expensive operator;
    // they are never publication gates. Early fixed-point problems need not be
    // oversolved below their changing VV10-anchor error. The ordinary strict
    // target acceptance above remains unchanged.
    const double nonlocal_stage_tolerance = pending_nonlocal_seed
                                                ? (nonlocal_outer_active ? 1e-5 : 1e-9)
                                                : nonlocal_inner_residual_tolerance;
    const unsigned nonlocal_stage_limit =
        pending_nonlocal_seed && mixed_stage && budget_fock && budget_stationarity
            ? nonlocal_seed_iteration_limit
            : nonlocal_proposal_iteration_limit;
    const bool enter_nonlocal_target =
        (pending_nonlocal_seed || pending_nonlocal_frozen) &&
        (physical.maximum_residual < nonlocal_stage_tolerance ||
         mixed_warmup.target_refresh_due(mixed_stage, output.iterations, options.max_iterations) ||
         (pending_nonlocal_seed ? nonlocal_seed_iterations : nonlocal_inner_iterations) >=
             nonlocal_stage_limit ||
         output.iterations + (nonlocal_outer_active ? 20U : 2U) >= options.max_iterations);
    // A completed mixed frozen-potential proposal needs a strict target next,
    // not another mixed pair sweep immediately discarded at precision handoff.
    // Neither its residual nor its energy can authorize publication. A stalled
    // inner solve also hands off within the same finite proposal budget.
    const bool handoff_nonlocal_proposal = budget_fock && budget_stationarity && mixed_stage &&
                                           nonlocal_outer_active && pending_nonlocal_frozen &&
                                           enter_nonlocal_target;
    const bool enter_strict_refinement =
        handoff_nonlocal_proposal ||
        mixed_warmup.enter_refinement(mixed_stage, converged, output.iterations,
                                      options.max_iterations, physical.maximum_residual,
                                      pending_nonlocal_seed || pending_nonlocal_frozen);
    if (enter_strict_refinement) {
      // FP32 is only a proposal accelerator. The bounded point-only history
      // exception cannot certify a physical iterate; ordinary AUTO still
      // clears history and every route retains the full FP64 refinement gate.
      strict_refinement = true;
      // Discard the mixed anchor, but retain the admitted outer solver. The
      // next iteration must rebuild the full FP64 target before its potential
      // can seed any frozen inner steps. Those steps remain provisional and
      // cannot satisfy the consecutive fresh-target publication gates.
      nonlocal_seed_stage = nonlocal_frozen_stage = false;
      nonlocal_exact_epoch_iterations = 0;
      final_closure = false;
      final_corrections = 0;
      stabilize_occupations = false;
      output.converged = false;
      is_active = true;
      if (!experimental_mixed_history) {
        check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
        check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
      }
      // Mixed J needs a new strict source lease; an XC-only warmup may retain
      // its unchanged FP64 J/K source. Existing resource/seed guards still apply.
      prepare_phase_source(phase_source_cold);
    } else if (pending_nonlocal_seed || pending_nonlocal_frozen) {
      // Never export a provisional density, energy, residual or force as a
      // converged target. The original total iteration budget remains finite.
      output.converged = false;
      is_active = output.iterations < options.max_iterations;
      if (enter_nonlocal_target) {
        nonlocal_seed_stage = false;
        nonlocal_frozen_stage = false;
        nonlocal_exact_epoch_iterations = 0;
        // Leave a bounded ordinary full-target tail if the nested solve stalls.
        if (output.iterations + 20 >= options.max_iterations) nonlocal_outer_active = false;
        check(cudaMemsetAsync(history_count, 0, sizeof(*history_count), stream));
        check(cudaMemsetAsync(history_head, 0, sizeof(*history_head), stream));
      }
    } else if (nonlocal_outer_active && !converged) {
      output.converged = false;
      is_active = output.iterations < options.max_iterations;
      const double target_residual_gate = std::min(1e-9, options.density_tolerance);
      if (confirm_nonlocal_stationary_density || physical.maximum_residual < target_residual_gate ||
          output.iterations + 20 >= options.max_iterations) {
        // Close with consecutive fresh physical Focks and the original strict
        // energy/density/residual gates, never an inner or stale-potential gate.
        nonlocal_outer_active = false;
      } else {
        nonlocal_frozen_stage = true;
        nonlocal_inner_iterations = 0;
        nonlocal_inner_residual_tolerance = nonlocal_inner_tolerance(
            budget_stationarity, physical.maximum_residual, options.density_tolerance);
        // The just-computed target Fock is also the first exact Fock of this
        // fixed-potential inner problem. Retain that one valid DIIS entry.
      }
    } else if (incremental_direct_jk && converged && !final_closure && spins == 1 &&
               provider.system().ecp_terms.empty()) {
      // Both energies now belong to consecutive full-density builds. The
      // current F[D], density proposal and maximum residual pass the same
      // gates as ordinary RKS, so this build is already the strict full audit.
      // UKS/ECP retain their separate corrective-closure contract below.
      account_incremental_full_energy_finalization();
      final_closure = true;
      output.converged = true;
      is_active = false;
    } else if (strict_final_closure && converged && !final_closure) {
      // A DIIS proposal can satisfy the SCF gate before a fresh F[D] proposal
      // does. Preserve any established UKS occupation stabilization through
      // this bounded correction, just as CPU UKS does; clearing it restarts
      // the stationary occupation cycle. Physical energy/residual gates and
      // the separate unshifted final-state export validator stay unchanged.
      final_closure = true;
      final_corrections = 0;
      output.converged = false;
      is_active = true;
    } else if (final_closure) {
      ++final_corrections;
      output.converged = converged;
      is_active = !output.converged && final_corrections < kMaximumFinalCorrections;
    } else {
      output.converged = converged;
      const bool refinement_budget = strict_refinement && mixed_precision_executed
                                         ? refinement_iterations < options.max_iterations
                                         : output.iterations < options.max_iterations;
      is_active = !output.converged && refinement_budget;
    }
    try {
      // Final RI-K provenance is independent of whether the caller elects to
      // update the reusable warm density cache.
      if (output.converged) retain_final_fitted_projection();
      if (output.converged && warm_updates) {
        // E, F, residual and retained D all belong to this same generation.
        // A failed/unfinished solve can never overwrite the last-good cache.
        publish_warm_seed();
        warm_energy = physical.electronic_energy;
        warm_energy_correction = physical.electronic_energy_correction;
        occupied_fitted_factor_ready = false;
      } else if (is_active &&
                 (confirm_nonlocal_stationary_density || confirm_direct_stationary_density)) {
        // tmp2 belongs to the unaccepted proposal, not the held density. Do
        // not expose that orbital frame as an occupied-factor identity.
        occupied_fitted_factor_ready = false;
        invalidate_warm_orbitals();
      } else if (is_active) {
        check(cudaMemcpyAsync(density, proposal, elements * sizeof(double),
                              cudaMemcpyDeviceToDevice, stream));
        if (early_final_probe_ran) {
          // The rejected proof borrowed tmp1/tmp2/effective/residual. Proposal
          // D has separate storage, but its old eigen scratch is no longer a
          // valid warm orbital frame. The next ordinary sweep rebuilds it.
          occupied_fitted_factor_ready = false;
          invalidate_warm_orbitals();
        } else {
          // Only the candidate changes the frame representation. Preserve the
          // incumbent orthonormal path unless exact AO warm reuse was selected.
          const bool retain_ao = fitted_exchange && spins == 1 && warm_factor_replay_enabled();
          check(cudaMemcpyAsync(warm_orbitals, retain_ao ? tmp1 : tmp2, elements * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream));
          warm_orbitals_ready = true;
          warm_orbitals_are_ao = retain_ao;
          ++movement.warm_orbital_frames_retained;
          occupied_fitted_factor_ready = fitted_exchange;
        }
      } else {
        occupied_fitted_factor_ready = false;
        invalidate_warm_orbitals();
      }
      if (output.converged) {
        final_state_ready = true;
        final_generation = generation;
        output.precision_work.returned_solve_epoch = solve_epoch;
        output.precision_work.returned_state_generation = final_generation;
      } else {
        final_fitted_projection_ready = false;
      }
      pending_fitted_occupied = false;
    } catch (...) {
      cudaStreamSynchronize(stream);
      is_active = false;
      is_failed = true;
      output.converged = false;
      occupied_fitted_factor_ready = false;
      pending_fitted_occupied = false;
      final_fitted_projection_ready = false;
      throw;
    }
    pending_incremental_delta = false;
    // An approximate energy cannot be the baseline of a strict stopping test.
    // The first FP64 observation starts a new energy epoch regardless of DIIS.
    const bool new_energy_epoch = enter_strict_refinement || enter_nonlocal_target;
    previous_energy =
        new_energy_epoch ? std::numeric_limits<double>::infinity() : physical.electronic_energy;
    previous_energy_correction = new_energy_epoch ? 0.0 : physical.electronic_energy_correction;
    runtime::cuda_trace::trace_counter("fock_builds", output.fock_builds);
    runtime::cuda_trace::trace_counter("converged", output.converged);
    runtime::cuda_trace::trace_counter("full_density_audit", output.converged);
    try {
      if (output.converged && complete_precision_inventory_domain()) {
        auto& work = output.precision_work;
        // The diagnostics kernel of this final strict physical F[D] iteration
        // already evaluated the residual/convergence gates. FinalAudit records
        // that executed audit; it does not invent another Fock build or kernel.
        work.events.push_back({scf::PrecisionWorkEventKind::FinalAudit,
                               scf::PrecisionWorkPhase::Finalization,
                               static_cast<std::uint64_t>(work.events.size()), output.iterations,
                               owner, solve_epoch, final_generation});
        ++output.precision.final_residual_audits;
        work.complete = true;
        work.operator_inventory_complete = true;
        output.precision.operator_work_counters_valid = 1U;
      }
    } catch (...) {
      is_active = false;
      is_failed = true;
      output.converged = false;
      final_state_ready = false;
      occupied_fitted_factor_ready = false;
      throw;
    }
    return is_active;
  }

  KsFinalStateIdentity final_identity() const {
    KsFinalStateIdentity identity;
    identity.determinant.factor = {owner, 1, final_generation, final_generation};
    identity.determinant.solve_epoch = solve_epoch;
    identity.determinant.model = provider.strategy();
    identity.determinant.occupied = {occupations[0]};
    if (spins == 2) identity.determinant.occupied.push_back(occupations[1]);
    identity.model = {1,
                      generated::split_hybrid_registered(functional)
                          ? 4U
                          : semilocal_family_domain_version(semilocal_family_from_code(functional)),
                      grid_spec,
                      xc_layout.tile_points,
                      functional,
                      spins,
                      device,
                      owner};
    if (range_correction) identity.model.range_correction = *range_correction;
    if (nonlocal_correlation) {
      identity.model.nonlocal_correlation = nonlocal_correlation->parameters();
      identity.model.nonlocal_density_domain = nonlocal_domain;
      identity.model.nonlocal_grid = options.nonlocal_grid_spec;
    }
    // A snapshot must describe the same XC composition that built its Fock matrix.
    identity.model.semilocal_exchange_scale = options.semilocal_exchange_scale;
    identity.model.semilocal_correlation_scale = options.semilocal_correlation_scale;
    return identity;
  }

  const CudaKsFinalStateToken& token_reference() const {
    if (!final_state_ready || !output.converged || is_active || is_pending || is_failed ||
        !solve_epoch || !final_generation)
      throw std::invalid_argument("CUDA KS owner has no successful current final state");
    // Model/basis/grid are immutable for this owner. Cache only host identity
    // metadata, never numerical validation: the live state gates above remain
    // mandatory on every borrow. Solve epochs do not wrap, and a new generation
    // rebuilds the snapshot before it can authorize any response consumer.
    if (!cached_final_token ||
        cached_final_token->identity.determinant.solve_epoch != solve_epoch ||
        cached_final_token->identity.determinant.factor.density_generation != final_generation)
      cached_final_token = CudaKsFinalStateToken{1, final_identity()};
    return *cached_final_token;
  }

  CudaKsFinalStateToken token() const { return token_reference(); }

  void prepare_final_frame() {
    if (!final_frame_ready) {
      const auto multiply = [&](const double* a, bool a_spin, bool transpose, const double* b,
                                bool b_spin, double* c) {
        multiply_spin(spins, a, a_spin, transpose, b, b_spin, final_enabled, c);
      };
      multiply(fock, true, false, x, false, tmp1);
      multiply(x, false, true, tmp1, true, tmp2);
      check(eigensolver->launch(spins, tmp2, effective, final_eigenvalues, final_solver_info,
                                final_spin_enabled),
            "CUDA KS final-state eigensolver launch failed");
      multiply(x, false, false, tmp2, true, final_coefficients);
    }
  }

  /** Test the current physical F[D] before publishing a final generation.
   * Numerical rejection continues from the still-owned DIIS proposal; driver,
   * allocation and eigensolver failures propagate normally. No force weights
   * or warm seed are authorized by this speculative proof. */
  bool probe_current_final_state() {
    runtime::cuda_trace::TraceRegion trace("ks_early_final_probe", stream);
    runtime::cuda_trace::trace_counter("early_final_probes", 1);
    prepare_final_frame();
    scf::solver::FinalStateLimits limits;
    limits.density_tolerance = options.density_tolerance;
    if (standard_convergence_active())
      limits.stationarity_tolerance = physical_stationarity_tolerance();
    limits.energy_tolerance = options.energy_tolerance;
    limits.maximum_corrections = 0;
    limits.require_canonicality = true;
    scf::solver::FinalStateDiagnostic products;
    std::string detail;
    if (!resident_final_products(true, provider.one_electron().nuclear_repulsion, products, detail))
      throw std::runtime_error(detail.empty() ? "CUDA KS early final-state products failed"
                                              : detail);
    auto identity = final_identity();
    identity.determinant.factor = {owner, 1, generation, generation};
    const KsFinalStateEvidence evidence{identity,      generation,
                                        true,          output.dft_diagnostic.components,
                                        output.energy, output.dft_diagnostic.physical_residual};
    KsFinalStateDiagnostic diagnostic;
    const bool accepted = validate_ks_final_state_evidence(identity, n, evidence, limits, products,
                                                           diagnostic, detail);
    runtime::cuda_trace::trace_counter(accepted ? "early_final_accepts" : "early_final_rejects", 1);
    // Reuse only the verified physical eigenframe. The public force path still
    // performs its own token-bound proof before staging D/W in shared scratch.
    final_frame_ready = accepted;
    return accepted;
  }

  void stage_stationary_weights() {
    constexpr unsigned threads = 128;
    build_density(final_coefficients, final_eigenvalues, final_enabled, tmp1);
    check(cudaGetLastError());
    if (spins == 2) {
      const auto matrix_blocks = static_cast<unsigned>((matrix + threads - 1) / threads);
      // One batch permits the total to overwrite the first spin block.
      launch_sum_uhf_spin_matrices_kernel(matrix_blocks, threads, 0, stream, 1,
                                          static_cast<std::int32_t>(n), tmp1, final_enabled, tmp1);
      check(cudaGetLastError());
      launch_sum_uhf_spin_matrices_kernel(matrix_blocks, threads, 0, stream, 1,
                                          static_cast<std::int32_t>(n), density, final_enabled,
                                          tmp2);
      check(cudaGetLastError());
    }
  }

  bool resident_final_products(bool canonical, double nuclear,
                               scf::solver::FinalStateDiagnostic& diagnostic, std::string& detail) {
    std::array<scf::cuda_df::ValidationInputs, 2> inputs{};
    for (unsigned spin = 0; spin < spins; ++spin) {
      auto& input = inputs[spin];
      const auto offset = static_cast<std::size_t>(spin) * matrix;
      input.n = n;
      input.occupied = occupations[spin];
      input.weight = spins == 1 ? 2 : 1;
      input.f = fock + offset;
      input.s = overlap;
      input.h = hcore;
      input.d = density + offset;
      input.c = final_coefficients + offset;
      input.values = final_eigenvalues + static_cast<std::size_t>(spin) * n;
      input.info = final_solver_info + spin;
      input.physical_fock = true;
      input.transposed_operators = true;
    }
    const ResidentFinalValidationWorkspace workspace{{tmp1, tmp2, residual, effective},
                                                     final_validation_partial,
                                                     resident_final_validation_partial_count(n)};
    ResidentFinalValidationTransfers transfers;
    const auto record = [&] {
      movement.scalar_d2h_bytes += transfers.device_to_host_bytes;
      movement.final_state_d2h_bytes += transfers.device_to_host_bytes;
      movement.synchronizations += transfers.synchronizations;
      if (transfers.synchronizations &&
          transfers.device_to_host_bytes == spins * sizeof(scf::cuda_df::ValidationPartial))
        ++movement.resident_final_validations;
    };
    try {
      const bool valid = resident_final_state_products(
          matrix_products.view(), std::span(inputs.data(), spins), workspace, canonical, nuclear,
          diagnostic, detail, &transfers);
      record();
      return valid;
    } catch (...) {
      // Invalid solver/symmetry evidence is detected after the packet drain;
      // rejecting it must not erase the transport that actually happened.
      record();
      throw;
    }
  }

  KsFinalStateDiagnostic prepare_resident_weights(const CudaKsFinalStateToken& expected,
                                                  std::string& detail) {
    const auto current = token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument("CUDA KS resident final-state token is stale");
    current_device();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("CUDA KS resident final-state proof cannot capture");
    if (final_stationary_weights_ready) return final_validation_diagnostic;
    try {
      prepare_final_frame();
      scf::solver::FinalStateLimits limits;
      limits.density_tolerance = options.density_tolerance;
      if (budget_stationarity || standard_convergence_active())
        limits.stationarity_tolerance = physical_stationarity_tolerance();
      limits.energy_tolerance = options.energy_tolerance;
      limits.maximum_corrections = 0;
      limits.require_canonicality = true;
      scf::solver::FinalStateDiagnostic products;
      if (!resident_final_products(true, provider.one_electron().nuclear_repulsion, products,
                                   detail))
        throw std::runtime_error(detail.empty() ? "CUDA KS resident final-state products failed"
                                                : detail);
      const KsFinalStateEvidence evidence{current.identity,
                                          final_generation,
                                          true,
                                          output.dft_diagnostic.components,
                                          output.energy,
                                          output.dft_diagnostic.physical_residual};
      KsFinalStateDiagnostic diagnostic;
      if (token() != current ||
          !validate_ks_final_state_evidence(current.identity, n, evidence, limits, products,
                                            diagnostic, detail))
        throw std::runtime_error(detail.empty() ? "CUDA KS resident final-state proof failed"
                                                : detail);
      stage_stationary_weights();
      check(cudaMemsetAsync(staged_xc_error, 0, sizeof(int), stream));
      scf::cuda_df::launch_validation_weights_status(stream, matrix, spins == 1 ? density : tmp2,
                                                     tmp1, staged_xc_error);
      check(cudaGetLastError());
      int weights_status{};
      check(cudaMemcpyAsync(&weights_status, staged_xc_error, sizeof(weights_status),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
      ++movement.synchronizations;
      movement.scalar_d2h_bytes += sizeof(weights_status);
      movement.final_state_d2h_bytes += sizeof(weights_status);
      if (weights_status || token() != current)
        throw std::runtime_error("CUDA KS resident stationary D/W is nonfinite or lost its owner");
      final_validation_diagnostic = diagnostic;
      final_frame_ready = final_stationary_weights_ready = true;
      return diagnostic;
    } catch (...) {
      // Scratch is phase-local. Drain even an incomplete frame launch before
      // revoking authorization, without touching the separate warm density.
      (void)cudaStreamSynchronize(stream);
      final_state_ready = final_frame_ready = final_stationary_weights_ready = false;
      throw;
    }
  }

  VerifiedKsFinalState read_final(const CudaKsFinalStateToken& expected,
                                  bool compute_weighted_density, std::string& detail) {
    const auto current = token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument(
          "CUDA KS final-state token has stale owner, epoch, generation, model or occupations");
    current_device();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument(
          "CUDA KS final-state read requires an ordinary noncapturing stream");
    prepare_final_frame();

    // The final-state product proof is shared by every prepared full-range
    // global hybrid; the semilocal XC family does not change its matrix algebra.
    const auto& final_spec = provider.strategy().spec;
    // An admitted Standard incremental solve ends with consecutive ordinary
    // full-density Focks and a canonical proof. Its current operator may use
    // the same resident validation products as an ordinary solve; the earlier
    // delta trajectory does not require exporting matrices to re-prove them.
    const bool ordinary_final_operator =
        !incremental_direct_jk ||
        (standard_convergence_active() && final_closure && incremental_energy_full_builds >= 2);
    const bool direct_global_hybrid =
        has_exchange && restricted_global_hybrid_exchange_coefficient(exchange_coefficient) &&
        !fitted_coulomb && !fitted_exchange && !has_range_correction && !nonlocal_correlation &&
        ordinary_final_operator && provider.system().ecp_terms.empty() &&
        std::none_of(provider.system().atoms.begin(), provider.system().atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; }) &&
        final_spec.derivative_order == 0 && final_spec.coulomb.present &&
        final_spec.coulomb.coefficient == 1.0 &&
        final_spec.coulomb.approximation == scf::FockApproximation::Exact &&
        final_spec.coulomb.op == scf::FockOperator::FullRange && final_spec.exchange.present &&
        final_spec.exchange.approximation == scf::FockApproximation::Exact &&
        final_spec.exchange.op == scf::FockOperator::FullRange;
    // The proof consumes the final matrices, not the XC family or J builder.
    // Admit only complete all-electron full-range pure-J models; range,
    // unclosed incremental, nonlocal, ECP and lower-precision owners keep their policy.
    const bool full_range_coulomb_only =
        !has_exchange && !fitted_exchange && !has_range_correction && !nonlocal_correlation &&
        ordinary_final_operator && provider.system().ecp_terms.empty() &&
        std::none_of(provider.system().atoms.begin(), provider.system().atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; }) &&
        final_spec.derivative_order == 0 && final_spec.coulomb.present &&
        final_spec.coulomb.coefficient == 1.0 &&
        final_spec.coulomb.op == scf::FockOperator::FullRange && !final_spec.exchange.present &&
        (final_spec.coulomb.approximation == scf::FockApproximation::Exact ||
         final_spec.coulomb.approximation == scf::FockApproximation::DensityFitted);
    // AUTO can propose mixed iterates and then close with fresh FP64 F[D].
    // The final proof consumes that strict frame; the historical proposal
    // schedule does not change its algebra or relax any token/numerical gate.
    const bool strict_final_arithmetic =
        !precision_schedule.any_lower_precision() || strict_refinement;
    const bool default_device_validation = device_final_validation_default_eligible(
        n, spins, direct_global_hybrid, strict_final_arithmetic, full_range_coulomb_only);
    const bool device_validation = device_final_validation_requested(
                                       std::getenv("GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION"),
                                       default_device_validation) &&
                                   final_validation_partial && matrix_products.library_enabled() &&
                                   !final_stationary_weights_ready;
    // A previously published W/total-D lease stays valid for this token. A
    // repeated read must not borrow its storage, even if it could rebuild W.
    // A device proof borrows tmp1/tmp2 before staging the force consumer's W/D.
    // The reference fallback retains its original single-drain staging order.
    bool staged_stationary_weights = false;
    const auto stage_weights = [&] {
      stage_stationary_weights();
      staged_stationary_weights = true;
    };
    if (compute_weighted_density && !final_stationary_weights_ready && !device_validation)
      stage_weights();

    KsPhysicalState physical;
    KsFinalStateCandidate candidate;
    physical.identity = candidate.identity = current.identity;
    physical.physical = candidate.physical_origin = true;
    physical.components = output.dft_diagnostic.components;
    physical.reported_energy = output.energy;
    physical.physical_residual = output.dft_diagnostic.physical_residual;
    candidate.fock_density_generation = final_generation;
    physical.density.resize(spins);
    physical.fock.resize(spins);
    candidate.spins.resize(spins);
    for (unsigned spin = 0; spin < spins; ++spin) {
      physical.density[spin].resize(matrix);
      physical.fock[spin].resize(matrix);
      candidate.spins[spin].vectors.resize(matrix);
      candidate.spins[spin].values.resize(n);
    }
    int info[2]{};
    cudaError_t error = cudaSuccess;
    const auto copy = [&](void* target, const void* source, std::size_t bytes) {
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost, stream);
    };
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto offset = static_cast<std::size_t>(spin) * matrix;
      copy(physical.density[spin].data(), density + offset, matrix * sizeof(double));
      copy(physical.fock[spin].data(), fock + offset, matrix * sizeof(double));
      copy(candidate.spins[spin].vectors.data(), final_coefficients + offset,
           matrix * sizeof(double));
      copy(candidate.spins[spin].values.data(),
           final_eigenvalues + static_cast<std::size_t>(spin) * n, n * sizeof(double));
      copy(&info[spin], final_solver_info + spin, sizeof(int));
    }
    const auto drained = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) error = drained;
    check(error);
    ++movement.synchronizations;
    ++movement.final_state_reads;
    const auto numeric_bytes = spins * (3 * matrix + n) * sizeof(double);
    movement.matrix_d2h_bytes += spins * 3 * matrix * sizeof(double);
    movement.final_state_d2h_bytes += numeric_bytes + spins * sizeof(int);
    final_frame_ready = true;
    for (unsigned spin = 0; spin < spins; ++spin)
      if (info[spin] != 0) {
        final_state_ready = final_stationary_weights_ready = false;
        throw std::runtime_error("CUDA KS final-state eigensolver reported failure");
      }
    // CUDA matrix products/eigensolvers store columns contiguously, whereas
    // the detached reference frame uses row-major C[ao, orbital]. Symmetric
    // D/F need no conversion; copying C verbatim would validate its transpose
    // and reject even a converged two-orbital state.
    for (auto& frame : candidate.spins)
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = row + 1; column < n; ++column)
          std::swap(frame.vectors[row * n + column], frame.vectors[column * n + row]);
    if (token() != current)
      throw std::invalid_argument("CUDA KS final-state eligibility changed during export");

    scf::solver::FinalStateLimits limits;
    limits.density_tolerance = options.density_tolerance;
    if (budget_stationarity || standard_convergence_active())
      limits.stationarity_tolerance = physical_stationarity_tolerance();
    limits.energy_tolerance = options.energy_tolerance;
    limits.maximum_corrections = 0;
    limits.require_canonicality = true;
    scf::solver::FinalStateOperations operations;
    if (device_validation) {
      // The exported host frame and resident inputs were read under the same
      // live token above. The private callback admits only that exact request,
      // not arbitrary detached arrays carrying copied identity labels.
      final_stationary_weights_ready = false;
      operations.products = [&](const auto& identity, const auto& metric, const auto& core,
                                double nuclear, const auto& densities, const auto& physical_fock,
                                const auto& frame, const auto& validation_limits, auto& diagnostic,
                                auto& product_detail) {
        if (token() != current || identity != current.identity.determinant ||
            &metric != &provider.one_electron().overlap ||
            &core != &provider.one_electron().hcore || densities != physical.density ||
            physical_fock.spins != physical.fock || frame.spins.size() != candidate.spins.size())
          throw std::invalid_argument("CUDA KS validation products lost their exported owner");
        for (unsigned spin = 0; spin < spins; ++spin) {
          if (frame.spins[spin].vectors != candidate.spins[spin].vectors ||
              frame.spins[spin].values != candidate.spins[spin].values)
            throw std::invalid_argument("CUDA KS validation products lost their exported frame");
        }
        const bool valid = resident_final_products(validation_limits.require_canonicality, nuclear,
                                                   diagnostic, product_detail);
        if (token() != current)
          throw std::invalid_argument("CUDA KS final-state owner changed during validation");
        return valid;
      };
    }
    VerifiedKsFinalState verified;
    bool accepted = false;
    try {
      accepted = validate_ks_final_state(current.identity, provider.one_electron().overlap,
                                         provider.one_electron().hcore, physical, candidate, limits,
                                         compute_weighted_density, verified, detail,
                                         device_validation ? &operations : nullptr);
    } catch (...) {
      final_state_ready = final_stationary_weights_ready = false;
      throw;
    }
    if (!accepted) {
      final_state_ready = final_stationary_weights_ready = false;
      throw std::runtime_error(detail.empty() ? "CUDA KS final-state validation failed" : detail);
    }
    if (device_validation && compute_weighted_density) {
      try {
        stage_weights();
        check(cudaStreamSynchronize(stream));
        ++movement.synchronizations;
      } catch (...) {
        final_state_ready = final_stationary_weights_ready = false;
        throw;
      }
    }
    if (staged_stationary_weights) {
      final_validation_diagnostic = verified.diagnostic;
      final_stationary_weights_ready = true;
    }
    return verified;
  }

  std::vector<double> download(const double* source) {
    current_device();
    std::vector<double> data(elements);
    try {
      check(cudaMemcpyAsync(data.data(), source, elements * sizeof(double), cudaMemcpyDeviceToHost,
                            stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      cudaStreamSynchronize(stream);
      throw;
    }
    movement.matrix_d2h_bytes += elements * sizeof(double);
    ++movement.synchronizations;
    return data;
  }
};

CudaKsPlan::CudaKsPlan(const scf::PreparedFockPlan& fock, const AoBasis& basis,
                       const MolecularGrid& grid, const scf::ScfOptions& options,
                       std::uint32_t functional_code, std::size_t tile_points,
                       const scf::ResolvedFockBuild* range_correction,
                       nlc::Vv10Plan* nonlocal_correlation, nlc::Vv10DensityDomain nonlocal_domain,
                       CudaXcPreparationBudget xc_budget,
                       const scf::PreparedFockPlan* range_provider)
    : impl_(std::make_unique<Impl>(fock, basis, grid, options, functional_code, tile_points,
                                   range_correction, nonlocal_correlation, nonlocal_domain,
                                   xc_budget, range_provider)) {}

CudaKsPlan::CudaKsPlan(const scf::PreparedFockPlan& fock, const AoBasis& basis,
                       const MolecularGrid& grid, const scf::ScfOptions& options,
                       SemilocalFamily functional, std::size_t tile_points,
                       const scf::ResolvedFockBuild* range_correction,
                       nlc::Vv10Plan* nonlocal_correlation, nlc::Vv10DensityDomain nonlocal_domain,
                       CudaXcPreparationBudget xc_budget,
                       const scf::PreparedFockPlan* range_provider)
    : CudaKsPlan(fock, basis, grid, options, semilocal_family_code(functional), tile_points,
                 range_correction, nonlocal_correlation, nonlocal_domain, xc_budget,
                 range_provider) {}
CudaKsPlan::~CudaKsPlan() = default;
void CudaKsPlan::begin(const std::vector<double>* seed, bool reuse_warm,
                       CudaKsSeedOrigin seed_origin) {
  impl_->begin(seed, reuse_warm, seed_origin);
}
bool CudaKsPlan::active() const noexcept { return impl_->is_active; }
bool CudaKsPlan::pending() const noexcept { return impl_->is_pending; }
bool CudaKsPlan::failed() const noexcept { return impl_->is_failed; }
void CudaKsPlan::enqueue_iteration() { impl_->enqueue(); }
bool CudaKsPlan::finish_iteration() { return impl_->finish(); }
scf::ScfResult CudaKsPlan::result(bool export_density) {
  if (!impl_->started || impl_->is_active || impl_->is_pending)
    throw std::logic_error("CUDA KS result is not terminal");
  auto result = impl_->output;
  if (impl_->xc)
    result.dft_diagnostic.cuda_ao_selection.xc_evaluations =
        impl_->xc->transfers().evaluations - impl_->initial_xc_evaluations;
  if (export_density) result.density = impl_->download(impl_->density);
  return result;
}
std::optional<std::uint64_t> CudaKsPlan::completed_fock_builds() const noexcept {
  if (!impl_->started || impl_->is_active || impl_->is_pending || impl_->is_failed ||
      !impl_->complete_precision_inventory_domain())
    return std::nullopt;
  const auto& work = impl_->output.precision_work;
  const auto recorded =
      std::count_if(work.events.begin(), work.events.end(), [](const auto& event) {
        return event.kind == scf::PrecisionWorkEventKind::MixedFock ||
               event.kind == scf::PrecisionWorkEventKind::StrictFock ||
               event.kind == scf::PrecisionWorkEventKind::PostScfFock;
      });
  // A terminal nonconverged attempt has no FinalAudit. Its physical build
  // ledger must nevertheless cover every build; audit flags stay unchanged.
  if (static_cast<std::uint64_t>(recorded) != impl_->output.fock_builds) return std::nullopt;
  return impl_->output.fock_builds;
}
scf::ScfResult CudaKsPlan::run(const std::vector<double>* seed, bool reuse_warm,
                               bool export_density, CudaKsSeedOrigin seed_origin) {
  begin(seed, reuse_warm, seed_origin);
  while (active()) {
    enqueue_iteration();
    finish_iteration();
  }
  return result(export_density);
}
bool CudaKsPlan::has_warm_start() const noexcept { return impl_->warm_ready; }
std::vector<double> CudaKsPlan::warm_density() {
  if (impl_->is_pending)
    throw std::logic_error("cannot export warm state during a pending iteration");
  if (!impl_->warm_ready) return {};
  if (!impl_->warm_factor_storage) return impl_->download(impl_->warm);
  impl_->current_device();
  // Between submissions effective is dead eigensolver/DIIS scratch. Do not use
  // tmp1/tmp2, proposal or final_coefficients: stationary/force leases use them.
  impl_->materialize_warm_density(impl_->effective);
  return impl_->download(impl_->effective);
}
scf::initial_guess::EigenOperation CudaKsPlan::seed_eigen_operation() {
  return [this](const auto& matrix, const auto* overlap, const auto* orthogonalizer,
                std::size_t dimension) {
    if (overlap || orthogonalizer)
      throw std::invalid_argument("CUDA KS seed eigen operation accepts symmetric solves only");
    return impl_->seed_eigen(matrix, dimension);
  };
}
void CudaKsPlan::set_warm_start_updates(bool enabled) noexcept { impl_->warm_updates = enabled; }
void CudaKsPlan::clear_warm_start() noexcept { impl_->clear_warm_state(); }
void CudaKsPlan::invalidate_final_state() noexcept {
  impl_->final_state_ready = impl_->final_frame_ready = impl_->final_stationary_weights_ready =
      false;
  impl_->final_fitted_projection_ready = false;
  impl_->final_generation = 0;
}
generativeqc_status CudaKsPlan::final_state_token(CudaKsFinalStateToken& token,
                                                  std::string& detail) const {
  token = {};
  detail.clear();
  try {
    token = impl_->token();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA KS final-state token failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}
generativeqc_status CudaKsPlan::resident_final_density(const CudaKsFinalStateToken& expected,
                                                       CudaKsResidentDensityBinding& binding,
                                                       std::string& detail) const {
  binding = {};
  detail.clear();
  try {
    const auto current = impl_->token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument(
          "CUDA KS resident-density token has stale owner, epoch, generation or model");
    if (!impl_->density || !impl_->matrix || (impl_->spins != 1 && impl_->spins != 2))
      throw std::logic_error("CUDA KS resident-density storage is unavailable");
    binding = {impl_->device,
               impl_->density,
               impl_->spins == 2 ? impl_->density + impl_->matrix : nullptr,
               impl_->matrix,
               impl_->spins,
               reinterpret_cast<void*>(impl_->stream),
               impl_->owner,
               impl_->solve_epoch,
               impl_->final_generation};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}
generativeqc_status CudaKsPlan::resident_final_fitted_projection(
    const CudaKsFinalStateToken& expected, CudaKsResidentFittedProjectionBinding& binding,
    std::string& detail) const {
  binding = {};
  detail.clear();
  try {
    const auto current = impl_->token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument(
          "CUDA KS fitted-projection token has stale owner, epoch, generation or model");
    if (!impl_->final_fitted_projection_ready) {
      detail = "CUDA KS final state has no retained fitted occupied projection";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto projection =
        scf::prepared_cuda_occupied_projection_binding(impl_->provider, impl_->occupations[0]);
    if (!projection || projection.device_id != impl_->device ||
        projection.stream != impl_->stream ||
        projection.source_identity != impl_->occupied_fock_binding.source_identity ||
        projection.scratch_generation != impl_->final_fitted_projection_scratch_generation ||
        projection.nbf != impl_->n || !impl_->proposal) {
      detail = "CUDA KS fitted occupied projection lease was revoked";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    binding = {impl_->device,
               impl_->proposal,
               projection.projection,
               projection.nbf,
               projection.naux,
               projection.rank,
               reinterpret_cast<void*>(impl_->stream),
               impl_->owner,
               impl_->solve_epoch,
               impl_->final_generation};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

generativeqc_status CudaKsPlan::resident_final_stationary_weights(
    const CudaKsFinalStateToken& expected, CudaKsResidentStationaryWeightsBinding& binding,
    std::string& detail) const {
  binding = {};
  detail.clear();
  try {
    const auto current = impl_->token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument(
          "CUDA KS resident stationary-weight token has stale owner, epoch, generation or model");
    if (!impl_->final_stationary_weights_ready || !impl_->tmp1 || !impl_->matrix)
      throw std::logic_error(
          "CUDA KS resident stationary D/W requires successful final-state weight preparation");
    const auto* total_density = impl_->spins == 1 ? impl_->density : impl_->tmp2;
    if (!total_density)
      throw std::logic_error("CUDA KS resident stationary density storage is unavailable");
    binding = {impl_->device, total_density, impl_->tmp1,        impl_->matrix,
               impl_->spins,  impl_->owner,  impl_->solve_epoch, impl_->final_generation};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}
generativeqc_status CudaKsPlan::prepare_resident_final_stationary_weights(
    const CudaKsFinalStateToken& expected, CudaKsResidentStationaryWeightsBinding& binding,
    KsFinalStateDiagnostic& diagnostic, std::string& detail) {
  binding = {};
  diagnostic = {};
  detail.clear();
  try {
    auto accepted = impl_->prepare_resident_weights(expected, detail);
    const auto status = resident_final_stationary_weights(expected, binding, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    diagnostic = std::move(accepted);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA KS resident final-state diagnostics failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const generativeqc::Error& error) {
    detail = error.what();
    return error.status();
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    if (detail.empty()) detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

generativeqc_status CudaKsPlan::resident_final_orbital_frame(
    const CudaKsFinalStateToken& expected, CudaKsResidentOrbitalFrameBinding& binding,
    std::string& detail) const {
  binding = {};
  detail.clear();
  try {
    const auto& current = impl_->token_reference();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument("CUDA KS resident orbital-frame token is stale");
    if (!impl_->final_frame_ready || !impl_->final_stationary_weights_ready)
      throw std::invalid_argument("CUDA KS orbital frame requires resident canonical validation");
    CudaKsResidentOrbitalFrameBinding candidate{
        impl_->device,
        impl_->final_coefficients,
        impl_->final_eigenvalues,
        impl_->density,
        impl_->fock,
        impl_->overlap,
        impl_->n,
        impl_->spins,
        {impl_->occupations[0], impl_->spins == 2 ? impl_->occupations[1] : 0},
        reinterpret_cast<void*>(impl_->stream),
        impl_->owner,
        impl_->solve_epoch,
        impl_->final_generation};
    if (!candidate) throw std::logic_error("CUDA KS resident orbital-frame storage is unavailable");
    binding = candidate;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

generativeqc_status CudaKsPlan::resident_final_fock_sources(const CudaKsFinalStateToken& expected,
                                                            CudaKsResidentFockBinding& binding,
                                                            std::string& detail) const {
  binding = {};
  CudaKsResidentOrbitalFrameBinding frame;
  const auto status = resident_final_orbital_frame(expected, frame, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  CudaKsResidentFockBinding candidate{frame.device_id,
                                      &impl_->provider,
                                      impl_->range_correction ? &*impl_->range_correction : nullptr,
                                      impl_->hcore,
                                      impl_->fock,
                                      frame.nao,
                                      frame.spins,
                                      frame.stream,
                                      frame.owner,
                                      frame.solve_epoch,
                                      frame.generation};
  if (!candidate) {
    detail = "CUDA KS resident Fock sources are unavailable";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  binding = candidate;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status CudaKsPlan::resident_final_grid(const CudaKsFinalStateToken& expected,
                                                    CudaKsResidentGridBinding& binding,
                                                    std::string& detail) const {
  binding = {};
  CudaKsResidentOrbitalFrameBinding frame;
  const auto status = resident_final_orbital_frame(expected, frame, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  if (!impl_->xc) {
    detail = "resident basis/grid requires a device-fused CUDA XC owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  try {
    const auto& layout = impl_->xc->layout();
    // SCF may integrate a private exact permutation. Derivative ownership and
    // atomic measures use the retained canonical order, authenticated below.
    const auto grid = impl_->xc->canonical_grid_view();
    CudaKsResidentGridBinding candidate{frame.device_id,   impl_->xc->packed_basis(),
                                        grid.points,       grid.weights,
                                        frame.density,     layout.natom,
                                        layout.nprimitive, layout.nao,
                                        grid.point_count,  frame.spins,
                                        frame.stream,      frame.owner,
                                        frame.solve_epoch, frame.generation};
    if (!candidate || layout.response || layout.nao != frame.nao || layout.spins != frame.spins ||
        layout.npoint != grid.point_count || reinterpret_cast<void*>(grid.stream) != frame.stream)
      throw std::logic_error("CUDA KS resident basis/grid storage is inconsistent");
    const auto quadrature = impl_->grid.cuda_view();
    if (quadrature) {
      if (quadrature.device != frame.device_id || quadrature.points != grid.points ||
          quadrature.weights != grid.weights || quadrature.point_count != grid.point_count)
        throw std::logic_error("CUDA KS moving-grid source is not its canonical XC quadrature");
      const auto& spec = impl_->grid.spec();
      candidate.atomic_weights = quadrature.atomic_weights;
      candidate.points_per_atom = spec.version == 4 ? 0 : grid_points_per_atom(spec);
      candidate.partition_iterations = spec.partition_iterations;
      candidate.coincident_tolerance = spec.coincident_tolerance;
      candidate.quadrature_owner = quadrature.owner;
      candidate.atom_offsets = quadrature.atom_offsets;
      candidate.atomic_radii = quadrature.atomic_radii;
    }
    binding = candidate;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

generativeqc_status CudaKsPlan::prepare_resident_final_orbital_frame(
    const CudaKsFinalStateToken& expected, CudaKsResidentOrbitalFrameBinding& binding,
    KsFinalStateDiagnostic& diagnostic, std::string& detail) {
  binding = {};
  CudaKsResidentStationaryWeightsBinding weights;
  const auto status =
      prepare_resident_final_stationary_weights(expected, weights, diagnostic, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  return resident_final_orbital_frame(expected, binding, detail);
}

generativeqc_status CudaKsPlan::resident_final_nonlocal_features(
    const CudaKsFinalStateToken& expected, CudaKsResidentNonlocalFeaturesBinding& binding,
    std::string& detail) const {
  binding = {};
  detail.clear();
  try {
    const auto current = impl_->token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument(
          "CUDA KS resident-nonlocal token has stale owner, epoch, generation or model");
    if (!impl_->device_nonlocal) {
      detail = "CUDA KS final state has no device-resident nonlocal features";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    // The resident nonlocal replay body overwrites raw full-grid features in
    // stream order. A terminal iteration does not copy its proposal back into
    // density, so at most one already-submitted second body sees the unchanged
    // final density. Its feature overwrite is therefore still the final-state
    // lease even though logical generation publication includes that bounded
    // speculative body. Graph replay remains disabled for nonlocal composition.
    const bool exact_generation = impl_->generation == impl_->final_generation;
    const bool bounded_terminal_overwrite =
        impl_->device_chunk_mode && impl_->generation > impl_->final_generation &&
        impl_->generation - impl_->final_generation < kCudaKsChunkCapacity;
    if (!exact_generation && !bounded_terminal_overwrite)
      throw std::logic_error("CUDA KS resident nonlocal features are not the final generation");
    if (!impl_->nonlocal_raw_density || !impl_->nonlocal_raw_gradient ||
        !impl_->nonlocal_grid.point_count())
      throw std::logic_error("CUDA KS resident nonlocal feature storage is unavailable");
    binding = {impl_->device,
               impl_->nonlocal_raw_density,
               impl_->nonlocal_raw_gradient,
               impl_->nonlocal_grid.point_count(),
               reinterpret_cast<void*>(impl_->stream),
               impl_->owner,
               impl_->solve_epoch,
               impl_->final_generation};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}
generativeqc_status CudaKsPlan::profile_fixed_density_components(
    const CudaKsFinalStateToken& expected, CudaKsFixedDensityProfile& profile,
    std::string& detail) {
  profile = {};
  detail.clear();
  bool submitted = false, complete = false;
  const auto revoke_failed_profile = [&]() noexcept {
    if (!submitted || complete) return;
    // The diagnostic borrows the published owner's mutable J/K/XC buffers.
    // A partial replay cannot retain a lease on those buffers. Drain before
    // revoking all final-state views; the last-good warm seed stays separate.
    (void)cudaSetDevice(impl_->device);
    (void)cudaStreamSynchronize(impl_->stream);
    invalidate_final_state();
    profile = {};
  };
  runtime::ResourceScopeExit failed_profile(revoke_failed_profile);
  try {
    const auto current = impl_->token();
    if (expected.version != 1 || expected != current)
      throw std::invalid_argument(
          "CUDA KS fixed-density profile token has stale owner, epoch, generation or model");
    impl_->current_device();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(impl_->stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("CUDA KS fixed-density profile requires an ordinary stream");

    runtime::OwnedCudaEvent begin(impl_->device), end(impl_->device);
    const auto timed = [&](auto&& submit) {
      begin.record(impl_->stream);
      submitted = true;
      submit();
      end.record(impl_->stream);
      end.synchronize();
      return static_cast<double>(end.elapsed_since(begin));
    };
    const auto read_error = [&](const int* source, const char* name) {
      int value = 0;
      check(cudaMemcpy(&value, source, sizeof(value), cudaMemcpyDeviceToHost));
      if (value) throw std::runtime_error(std::string(name) + " reported a numerical failure");
    };

    const auto& strategy = impl_->provider.strategy();
    const auto* beta = impl_->spins == 2 ? impl_->density + impl_->matrix : nullptr;
    auto profile_primary = [&](bool want_j, bool want_k, unsigned slot) {
      auto spec = strategy.spec;
      spec.coulomb.present = want_j && strategy.spec.coulomb.present;
      spec.exchange.present = want_k && strategy.spec.exchange.present;
      if (!spec.coulomb.present && !spec.exchange.present) return;

      if (!impl_->phase_source_range && impl_->phase_source && impl_->phase_source->active() &&
          !impl_->phase_source->mixed_recurrence()) {
        profile.milliseconds[slot] = timed([&] {
          check(impl_->phase_source->enqueue(
                    spec, impl_->density, spec.coulomb.present ? impl_->j : nullptr,
                    spec.exchange.present ? impl_->exchange : nullptr, impl_->jk_error,
                    impl_->provider.strategy().screening_tolerance, detail, beta,
                    spec.exchange.present && impl_->spins == 2 ? impl_->exchange + impl_->matrix
                                                               : nullptr),
                detail);
        });
        check(impl_->phase_source->audit(), "fixed-density resident J/K audit failed");
        impl_->phase_source->observe_completed(1);
        read_error(impl_->jk_error, want_j ? "fixed-density J" : "fixed-density K");
      } else if (auto* direct = impl_->provider.cuda_direct_source()) {
        profile.milliseconds[slot] = timed([&] {
          check(scf::enqueue_cuda_direct_jk_device(
                    direct, spec, impl_->density, beta, impl_->matrix,
                    spec.coulomb.present ? impl_->j : nullptr,
                    spec.exchange.present ? impl_->exchange : nullptr,
                    spec.exchange.present && impl_->spins == 2 ? impl_->exchange + impl_->matrix
                                                               : nullptr,
                    impl_->jk_error, detail),
                detail);
        });
        read_error(impl_->jk_error, want_j ? "fixed-density J" : "fixed-density K");
      } else if (auto* fitted = impl_->provider.cuda_fitted_source()) {
        const scf::JkTermSelection terms{spec.coulomb.present, spec.exchange.present};
        profile.milliseconds[slot] = timed([&] {
          const auto status =
              impl_->spins == 2
                  ? scf::execute_cuda_density_fitting_uhf_jk_device(
                        fitted, impl_->density, beta, spec.coulomb.present ? impl_->j : nullptr,
                        spec.exchange.present ? impl_->exchange : nullptr,
                        spec.exchange.present ? impl_->exchange + impl_->matrix : nullptr, detail,
                        terms, scf::FockMatrixLayout::RowMajor)
                  : scf::execute_cuda_density_fitting_rhf_jk_device(
                        fitted, impl_->density, spec.coulomb.present ? impl_->j : nullptr,
                        spec.exchange.present ? impl_->exchange : nullptr, detail, terms,
                        scf::FockMatrixLayout::RowMajor);
          check(status, detail);
        });
      } else {
        throw std::runtime_error("CUDA KS fixed-density profile lost its prepared Fock provider");
      }
      profile.present_mask |= (1U << slot);
    };

    profile_primary(true, false, 0);
    profile_primary(false, true, 1);

    if (impl_->has_range_correction) {
      profile.milliseconds[2] =
          timed([&] { impl_->enqueue_range_correction(impl_->density, beta, detail); });
      if (impl_->range_source_active()) {
        check(impl_->phase_source->audit(), "fixed-density resident range K audit failed");
        impl_->phase_source->observe_completed(1);
      }
      read_error(impl_->range_jk_error, "fixed-density range K");
      profile.present_mask |= (1U << 2);
    }

    // A semilocal replay against the same D is state-equivalent for ordinary
    // device-fused KS. Nonlocal composition has already added VV10 to the XC
    // buffers, so a semilocal-only replay would corrupt that final view; leave
    // its XC component explicitly unavailable instead.
    if (impl_->xc && !impl_->nonlocal_correlation) {
      CudaXcView view;
      profile.milliseconds[3] = timed([&] {
        view = impl_->xc->enqueue_replay_body(impl_->density, impl_->elements,
                                              generativeqc::runtime::PrecisionPhase::StrictAudit);
      });
      read_error(view.error, "fixed-density XC");
      profile.present_mask |= (1U << 3);
    }

    if (impl_->token() != current)
      throw std::invalid_argument("CUDA KS final-state eligibility changed during profiling");
    complete = true;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "CUDA KS fixed-density component profile exceeded its resource budget";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const generativeqc::Error& error) {
    detail = error.what();
    return error.status();
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    if (detail.empty()) detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

generativeqc_status CudaKsPlan::read_final_state(const CudaKsFinalStateToken& expected,
                                                 bool compute_weighted_density,
                                                 VerifiedKsFinalState& state, std::string& detail) {
  state = {};
  detail.clear();
  try {
    state = impl_->read_final(expected, compute_weighted_density, detail);
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for detached CUDA KS final state failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const generativeqc::Error& error) {
    impl_->final_state_ready = false;
    detail = error.what();
    return error.status();
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    impl_->final_state_ready = false;
    if (detail.empty()) detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}
generativeqc_status CudaKsPlan::prepare_force_memory(const CudaKsFinalStateToken& expected,
                                                     std::size_t additional_device_bytes,
                                                     std::array<std::uint64_t, 4>& receipt,
                                                     std::string& detail) {
  receipt = {};
  detail.clear();
  try {
    if (expected.version != 1 || expected != impl_->token())
      throw std::invalid_argument("CUDA KS force-memory token is stale");
    impl_->current_device();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(impl_->stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::invalid_argument("CUDA KS force-memory handoff cannot capture");
    std::size_t free_before{}, free_after{}, total{}, retired{};
    check(cudaMemGetInfo(&free_before, &total));
    free_after = free_before;
    if (additional_device_bytes > free_before &&
        (impl_->fast_anchor_arena || impl_->phase_source)) {
      // The final-state token excludes active/pending SCF work. Drain its
      // producer before freeing optional cold-SCF storage. Downstream D/W/XC
      // leases borrow neither the anchor nor the independently owned ERIs.
      check(cudaStreamSynchronize(impl_->stream));
      ++impl_->movement.synchronizations;
      if (impl_->fast_anchor_arena) {
        retired = impl_->retire_fast_anchor_storage();
        check(cudaMemGetInfo(&free_after, &total));
      }
      if (additional_device_bytes > free_after && impl_->phase_source) {
        retired += impl_->phase_device_bytes;
        impl_->retire_phase_source(false);  // Successful SCF already audited it.
        check(cudaMemGetInfo(&free_after, &total));
      }
    }
    receipt = {additional_device_bytes, free_before, free_after, retired};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "CUDA KS force-memory handoff allocation failed";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const generativeqc::Error& error) {
    detail = error.what();
    return error.status();
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

const CudaKsResources& CudaKsPlan::resources() const noexcept { return impl_->resource; }
const tensor::PanelProductDiagnostic* CudaKsPlan::density_provider_diagnostic() const noexcept {
  return impl_->xc ? impl_->xc->density_provider_diagnostic() : nullptr;
}
CudaKsTransfers CudaKsPlan::transfers() const noexcept {
  auto out = impl_->movement;
  const auto& region = impl_->device_chunk_region.metrics();
  out.execution_region_bindings = region.bindings;
  out.execution_region_invalidations = region.invalidations;
  out.execution_region_executions = region.executions;
  out.execution_region_failures = region.failures;
  out.execution_region_recoveries = region.recoveries;
  const auto& replay = impl_->solver_region_executor.replay_metrics();
  out.execution_region_captures = replay.captures;
  out.execution_region_replays = replay.replays;
  out.execution_region_fallbacks = replay.fallbacks;
  if (impl_->xc) {
    const auto& xc = impl_->xc->transfers();
    out.setup_h2d_bytes += xc.setup_h2d_bytes;
    out.synchronizations += xc.synchronizations;
  }
  if (impl_->nonlocal_xc) {
    const auto& nonlocal = impl_->nonlocal_xc->transfers();
    out.setup_h2d_bytes += nonlocal.setup_h2d_bytes;
    out.synchronizations += nonlocal.synchronizations;
  }
  return out;
}
}  // namespace generativeqc::dft
