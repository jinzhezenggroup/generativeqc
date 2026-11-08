#include "scf/cuda/rhf_policy.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "generated_one_electron_derivative_policy.cuh"
#include "scf/cuda/hf_bounded_page_policy.hpp"
#include "scf/rhf_block_convergence_policy.hpp"

namespace generativeqc::scf::cuda_policy {

RhfBlockConvergenceRequest parse_rhf_block_convergence_request(const char* value) noexcept {
  if (value == nullptr || std::strcmp(value, "0") == 0) return RhfBlockConvergenceRequest::Off;
  return std::strcmp(value, "1") == 0 ? RhfBlockConvergenceRequest::On
                                      : RhfBlockConvergenceRequest::Invalid;
}

RhfBlockConvergenceRequest rhf_block_convergence_requested() noexcept {
  return parse_rhf_block_convergence_request(std::getenv("GENERATIVEQC_HF_BLOCK_CONVERGENCE"));
}

bool resolve_rhf_block_convergence(RhfBlockConvergenceRequest request, bool unrestricted,
                                   bool mixed_precision, bool incremental, bool quartet_direct,
                                   bool physical_reference) noexcept {
  return request == RhfBlockConvergenceRequest::On && !unrestricted && !mixed_precision &&
         !incremental && quartet_direct && !physical_reference;
}

SharedRysXyRequest parse_shared_rys_xy_request(const char* value) noexcept {
  if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0)
    return SharedRysXyRequest::Off;
  return std::strcmp(value, "1") == 0 ? SharedRysXyRequest::On : SharedRysXyRequest::Invalid;
}

SharedRysXyRequest shared_rys_xy_requested() noexcept {
  const char* xy = std::getenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_XY");
  const auto request =
      parse_shared_rys_xy_request(xy, std::getenv("GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z"));
  return request == SharedRysXyRequest::Off && (xy == nullptr || *xy == '\0')
             ? SharedRysXyRequest::Auto
             : request;
}

SharedRysXyRequest parse_shared_rys_xy_request(const char* xy, const char* weighted_z) noexcept {
  const auto primary = parse_shared_rys_xy_request(xy);
  const auto weighted = parse_shared_rys_xy_request(weighted_z);
  if (primary == SharedRysXyRequest::Invalid || weighted == SharedRysXyRequest::Invalid ||
      (weighted == SharedRysXyRequest::On && primary != SharedRysXyRequest::On))
    return SharedRysXyRequest::Invalid;
  return weighted == SharedRysXyRequest::On ? SharedRysXyRequest::WeightedZOn : primary;
}

SharedRysXyPolicy resolve_shared_rys_xy_policy(SharedRysXyRequest request,
                                               const SharedRysXyContext& c,
                                               const runtime::CudaTargetInfo& target) noexcept {
  using A = SharedRysXyAdmission;
  A admission = A::Off;
  // Shared scalar-FP64 recurrence storage is a throughput-class choice, not
  // an ISA identity. Unknown or reduced-rate FP64 retains the incumbent;
  // explicit trials may bypass profitability, never capability or legality.
  const bool full_rate_fp64 =
      target.fp32_fp64_ratio > 0 && target.fp32_fp64_ratio <= 2;
  if (request == SharedRysXyRequest::On || request == SharedRysXyRequest::WeightedZOn ||
      (request == SharedRysXyRequest::Auto && full_rate_fp64)) {
    if (c.unrestricted)
      admission = A::Unrestricted;
    else if (c.mixed_precision)
      admission = A::MixedPrecision;
    else if (c.capability_mask == 0)
      admission = A::UnsupportedProfile;
    // The generated capability owns per-kernel resource qualification. This
    // fixed family additionally requires 32-lane warps and its largest block.
    else if (target.warp_size != 32 || target.maximum_threads_per_block < 192 ||
             target.maximum_threads_per_sm < 192 || target.maximum_blocks_per_sm == 0)
      admission = A::UnsupportedResources;
    else if (c.persistent_eri)
      admission = A::PersistentEri;
    else if (!c.quartet_direct)
      admission = A::NonPaged;
    else if (c.incremental_jk)
      admission = A::IncrementalJk;
    else if (c.separate_jk)
      admission = A::SeparateJk;
    else if (c.streaming)
      admission = A::Streaming;
    else if (c.compensated_output)
      admission = A::CompensatedOutput;
    else
      admission = A::Active;
  }
  return {request, admission, admission == A::Active ? c.capability_mask : 0};
}

const char* shared_rys_xy_admission_name(SharedRysXyAdmission admission) noexcept {
  switch (admission) {
    case SharedRysXyAdmission::Off:
      return "off";
    case SharedRysXyAdmission::Active:
      return "active";
    case SharedRysXyAdmission::Unrestricted:
      return "unrestricted";
    case SharedRysXyAdmission::MixedPrecision:
      return "mixed_precision";
    case SharedRysXyAdmission::UnsupportedProfile:
      return "unsupported_profile";
    case SharedRysXyAdmission::UnsupportedResources:
      return "unsupported_resources";
    case SharedRysXyAdmission::NonPaged:
      return "non_paged";
    case SharedRysXyAdmission::IncrementalJk:
      return "incremental_jk";
    case SharedRysXyAdmission::SeparateJk:
      return "separate_jk";
    case SharedRysXyAdmission::Streaming:
      return "streaming";
    case SharedRysXyAdmission::PersistentEri:
      return "persistent_eri";
    case SharedRysXyAdmission::CompensatedOutput:
      return "compensated_output";
  }
  return "invalid";
}

namespace {

constexpr std::size_t kUnknownTargetFixedTopologyArenaBytes = std::size_t{256} << 20;
constexpr std::size_t kFixedTopologyArenaDeviceMemoryDivisor = 32;
constexpr std::size_t kUnknownTargetBoundedStreamingArenaBytes = std::size_t{256} << 20;
constexpr std::size_t kBoundedStreamingArenaDeviceMemoryDivisor = 16;
constexpr std::size_t kLargeBoundedStreamingArenaDeviceMemoryDivisor = 4;
constexpr std::size_t kLargeBoundedStreamingTargetBytes = std::size_t{24} << 30;
constexpr unsigned kUnknownTargetPersistentQuartetWarpsPerSm = 4;

std::uint64_t saturating_multiply(std::uint64_t first, std::uint64_t second) noexcept {
  if (first == 0 || second == 0) return 0;
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return second > maximum / first ? maximum : first * second;
}

constexpr double kDefaultMixedPrecisionFockThreshold = 1.0e-6;
/**
 * Ceiling for the resolved tile cutoff, anchored so the default 1.0e-10 target
 * can recover the legacy measured-accurate 1.0e-6 threshold. It is only an
 * upper bound: the accumulated-error budget below resolves the value used.
 */
constexpr double kAutoPrecisionFactor = 1.0e4;
/**
 * Fraction of the requested energy tolerance reserved for the accumulated FP32
 * rounding of the iterative mixed Fock. The target-precision refinement makes
 * the operator that produces the reported energy, orbitals, and forces exact,
 * so this bound covers only the perturbation the iterative density was converged
 * against.
 */
constexpr double kAutoMixedPrecisionErrorBudgetFraction = 6.25e-02;
constexpr double kTightConvergedFockReuseDensityRms = 1.0e-12;
constexpr double kExpandedConvergedFockReuseDensityTolerance = 1.0e-9;
constexpr double kExpandedConvergedFockReuseDensityRms = 2.0e-9;
using NucleusCooperativeSchedule =
    generated_one_electron_derivative_policy::NucleusCooperativeSchedule;

bool enabled(const char* variable) noexcept {
  const char* selection = std::getenv(variable);
  return selection == nullptr ||
         (std::strcmp(selection, "0") != 0 && std::strcmp(selection, "none") != 0);
}

bool selected(const char* variable, const char* value) noexcept {
  const char* selection = std::getenv(variable);
  return selection != nullptr &&
         (std::strcmp(selection, "1") == 0 || std::strcmp(selection, value) == 0);
}

DirectJkTuningProfile expanded_hf_direct_profile() noexcept {
  DirectJkTuningProfile profile;
  profile.bounded_streaming.maximum_task_capacity = 32U * 1024U * 1024U;
  profile.bounded_streaming.maximum_arena_bytes = std::size_t{6} << 30;
  return profile;
}

std::size_t resolve_target_memory_budget(std::size_t total_global_memory,
                                         std::size_t profile_ceiling, std::size_t unknown_fallback,
                                         std::size_t divisor) noexcept {
  if (total_global_memory == 0) return std::min(profile_ceiling, unknown_fallback);
  return std::min(profile_ceiling, total_global_memory / divisor);
}

std::optional<double> parsed_mixed_precision_override(double screening_tolerance) noexcept {
  // A user-supplied explicit numeric threshold from the legacy switch; the
  // absent / 0 / none / auto / invalid spellings yield std::nullopt.
  const char* selection = std::getenv("GENERATIVEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  if (selection == nullptr) return std::nullopt;
  if (std::strcmp(selection, "0") == 0 || std::strcmp(selection, "none") == 0 ||
      std::strcmp(selection, "auto") == 0) {
    return std::nullopt;
  }
  char* end = nullptr;
  const double value = std::strtod(selection, &end);
  if (end == selection || end == nullptr || *end != '\0') return std::nullopt;
  if (!std::isfinite(value) || value <= screening_tolerance) return std::nullopt;
  return value;
}
}  // namespace

SmallHfAnalyticEstimate estimate_small_hf_workload(const runtime::CudaTargetInfo& target,
                                                   const SmallHfWorkload& workload) noexcept {
  SmallHfAnalyticEstimate estimate;
  const std::uint64_t n = static_cast<std::uint64_t>(workload.nbf);
  if (n == 0) return estimate;

  const std::uint64_t n2 = saturating_multiply(n, n);
  const std::uint64_t n4 = saturating_multiply(n2, n2);
  const std::uint64_t systems = static_cast<std::uint64_t>(workload.system_count);
  const std::uint64_t fock_states = static_cast<std::uint64_t>(workload.fock_state_count);

  estimate.matrix =
      tensor::CudaMatrixProductAdmission::resolve(target, workload.nbf, workload.matrix_count)
          .estimate();

  estimate.eri_elements = saturating_multiply(systems, n4);
  estimate.eri_bytes = saturating_multiply(8U, estimate.eri_elements);
  estimate.cached_fock_ao_quartets = saturating_multiply(fock_states, n4);
  return estimate;
}

SmallHfProfitabilityPolicy resolve_small_hf_profitability(
    const runtime::CudaTargetInfo& target, const SmallHfWorkload& workload,
    SmallHfProfitabilityProfile profile) noexcept {
  SmallHfProfitabilityPolicy policy;
  policy.estimate = estimate_small_hf_workload(target, workload);
  if (workload.nbf == 0) return policy;

  // Persistent-ERI selection is intentionally still compatibility-only in this
  // slice. Its derived n^4 storage/work facts are exposed above, but the direct
  // route also needs screened shell-quartet work and expected SCF reuse before
  // a profitability comparison is meaningful.
  policy.persistent_eri = workload.nbf <= profile.fallback_persistent_eri_ao_limit;

  policy.matrix = tensor::CudaMatrixProductAdmission::resolve(
      target, workload.nbf, workload.matrix_count, profile.matrix);
  return policy;
}

DirectJkSchedulePolicy resolve_direct_jk_schedule_policy(const runtime::CudaTargetInfo& target,
                                                         DirectJkTuningProfile profile) noexcept {
  DirectJkSchedulePolicy policy;
  policy.cuda_stack_limit_bytes = profile.cuda_stack_limit_bytes;

  // Fixed-topology descriptors are long-lived prepared-state storage. Keep
  // their historical one-GiB profile ceiling under the conservative 1/32
  // target-memory budget.
  policy.fixed_topology.arena_maximum_bytes = resolve_target_memory_budget(
      target.total_global_memory, profile.fixed_topology.maximum_arena_bytes,
      kUnknownTargetFixedTopologyArenaBytes, kFixedTopologyArenaDeviceMemoryDivisor);

  // Bounded streaming has an independent reusable scratch budget. Large
  // opted-in targets can amortize compaction across a 32M-task page (6 GiB at the
  // current ABI). Smaller/unknown targets keep the conservative 1/16 budget.
  // The owner retries its incumbent page if the larger allocation is refused.
  policy.bounded_streaming.task_capacity_ceiling = profile.bounded_streaming.maximum_task_capacity;
  policy.bounded_streaming.arena_maximum_bytes = resolve_target_memory_budget(
      target.total_global_memory, profile.bounded_streaming.maximum_arena_bytes,
      kUnknownTargetBoundedStreamingArenaBytes,
      selected("GENERATIVEQC_HF_COLD_SCREENING", "1") &&
              target.total_global_memory >= kLargeBoundedStreamingTargetBytes
          ? kLargeBoundedStreamingArenaDeviceMemoryDivisor
          : kBoundedStreamingArenaDeviceMemoryDivisor);

  // Persistent workers are one warp per block. Respect both the resident block
  // ceiling and the SM thread ceiling; unknown facts choose a smaller
  // conservative fallback instead of assuming the measured 5090 occupancy.
  const bool complete_occupancy = target.warp_size != 0 && target.maximum_threads_per_sm != 0 &&
                                  target.maximum_blocks_per_sm != 0;
  unsigned legal_warps = kUnknownTargetPersistentQuartetWarpsPerSm;
  if (target.warp_size != 0 && target.maximum_threads_per_sm != 0) {
    const unsigned thread_ceiling = std::max(1U, target.maximum_threads_per_sm / target.warp_size);
    legal_warps = complete_occupancy ? thread_ceiling : std::min(legal_warps, thread_ceiling);
  }
  if (target.maximum_blocks_per_sm != 0) {
    legal_warps = std::min(legal_warps, target.maximum_blocks_per_sm);
  }
  policy.persistent_quartet_warps_per_sm =
      std::max(1U, std::min(profile.maximum_persistent_quartet_warps_per_sm, legal_warps));
  return policy;
}

std::size_t direct_jk_bounded_streaming_task_capacity_limit(
    const DirectJkSchedulePolicy& policy, std::size_t generated_task_bytes) noexcept {
  if (generated_task_bytes == 0) return 0;
  return std::min(policy.bounded_streaming.task_capacity_ceiling,
                  policy.bounded_streaming.arena_maximum_bytes / generated_task_bytes);
}

bool reuse_converged_fock_requested() noexcept {
  const char* force_rebuild = std::getenv("GENERATIVEQC_FINAL_FOCK_REBUILD");
  return force_rebuild == nullptr || std::strcmp(force_rebuild, "0") == 0 ||
         std::strcmp(force_rebuild, "none") == 0;
}

DirectJkTuningProfile resolve_hf_direct_tuning_profile(bool physical_reference) noexcept {
  return !physical_reference && selected("GENERATIVEQC_HF_COLD_SCREENING", "1")
             ? expanded_hf_direct_profile()
             : DirectJkTuningProfile{};
}

DirectJkSchedulePolicy resolve_hf_bounded_page_policy(const runtime::CudaTargetInfo& target,
                                                      const HfBoundedPageDomain& domain) noexcept {
  auto profile = resolve_hf_direct_tuning_profile(domain.physical_reference);
  const bool automatic = std::getenv("GENERATIVEQC_HF_COLD_SCREENING") == nullptr &&
                         domain.nbf >= 512 && domain.systems == 1 &&
                         domain.maximum_shell_angular <= 2 && domain.maximum_iterations >= 24 &&
                         !domain.unrestricted && !domain.physical_reference &&
                         !domain.mixed_precision && !domain.incremental_jk;
  if (automatic) {
    profile = expanded_hf_direct_profile();
  }
  auto policy = resolve_direct_jk_schedule_policy(target, profile);
  // The generic schedule remains conservative for other consumers and custom
  // profiles. Only this admitted HF owner may use the qualified larger fraction.
  if (automatic && target.total_global_memory >= kLargeBoundedStreamingTargetBytes) {
    policy.bounded_streaming.arena_maximum_bytes = resolve_target_memory_budget(
        target.total_global_memory, profile.bounded_streaming.maximum_arena_bytes,
        kUnknownTargetBoundedStreamingArenaBytes, kLargeBoundedStreamingArenaDeviceMemoryDivisor);
  }
  return policy;
}

ColdHfScreeningPolicy resolve_cold_hf_screening(const ColdHfScreeningDomain& domain,
                                                double target_screening) noexcept {
  constexpr double cutoff = 8.0e-8;
  const char* selection = std::getenv("GENERATIVEQC_HF_COLD_SCREENING");
  // Qualification covers large closed-shell through-d direct owners. Keep
  // small/batched/reference and seeded solves on their existing paths; never
  // consume a short user allowance with a proposal that needs a target tail.
  if ((selection && std::strcmp(selection, "1") != 0) || domain.nbf < 512 || domain.systems != 1 ||
      domain.maximum_shell_angular > 2 || domain.maximum_iterations < 24 || !domain.first_setup ||
      domain.warm_seed || domain.unrestricted || domain.physical_reference ||
      !domain.bounded_quartet_direct || domain.persistent_eri || domain.resident_values ||
      domain.mixed_precision || domain.incremental_jk || !std::isfinite(target_screening) ||
      target_screening < 0.0 || target_screening >= cutoff)
    return {};
  return {cutoff, 8U};
}

std::optional<double> configured_mixed_precision_fock_threshold(
    double screening_tolerance) noexcept {
  const char* selection = std::getenv("GENERATIVEQC_MIXED_PRECISION_FOCK_THRESHOLD");
  if (selection == nullptr || std::strcmp(selection, "0") == 0 ||
      std::strcmp(selection, "none") == 0) {
    return std::nullopt;
  }
  double threshold = 0.0;
  if (std::strcmp(selection, "auto") == 0) {
    threshold = kDefaultMixedPrecisionFockThreshold;
  } else {
    char* end = nullptr;
    threshold = std::strtod(selection, &end);
    if (end == selection || end == nullptr || *end != '\0') {
      return std::nullopt;
    }
  }
  if (!std::isfinite(threshold) || threshold <= screening_tolerance) {
    return std::nullopt;
  }
  return threshold;
}

AutoMixedPrecisionAdmission admit_auto_mixed_precision_fock(double energy_tolerance,
                                                            double screening_tolerance,
                                                            double eligible_tiles) noexcept {
  AutoMixedPrecisionAdmission admission;
  if (!(energy_tolerance > 0.0) || !std::isfinite(energy_tolerance)) return admission;
  // An unknown (or empty) census cannot be budgeted; refusing keeps the FP64
  // operator rather than admitting work whose accumulated error is unbounded.
  if (!(eligible_tiles >= 1.0) || !std::isfinite(eligible_tiles)) return admission;
  // Worst case: every eligible tile is admitted at the cutoff and carries the
  // full relative FP32 rounding, so the accumulated Fock error is bounded by
  // eps32 * cutoff * eligible_tiles. Solve that bound for the largest cutoff the
  // reserved budget certifies, and never exceed the legacy measured anchor.
  const double reserved_error = kAutoMixedPrecisionErrorBudgetFraction * energy_tolerance;
  const double budget_threshold =
      reserved_error / (kMixedPrecisionFloat32UnitRoundoff * eligible_tiles);
  const double anchor_threshold = kAutoPrecisionFactor * energy_tolerance;
  const double threshold = std::min(anchor_threshold, budget_threshold);
  if (!(threshold > screening_tolerance)) {
    // The budget certifies no useful tile at this requested accuracy, so the
    // operator stays FP64 instead of accumulating rounding it cannot bound.
    return admission;
  }
  admission.admitted = true;
  admission.threshold = threshold;
  admission.reserved_error = reserved_error;
  admission.eligible_tiles = eligible_tiles;
  return admission;
}
MixedPrecisionFockPolicy resolve_mixed_precision_fock_policy(
    std::optional<generativeqc_precision_mode> precision_mode, double energy_tolerance,
    double screening_tolerance, double eligible_tiles) noexcept {
  MixedPrecisionFockPolicy policy;
  if (!precision_mode.has_value()) {
    // No explicit public policy: preserve the legacy diagnostic switch exactly.
    // The diagnostic cutoff is item agnostic, so every item shares the ceiling.
    policy.threshold = configured_mixed_precision_fock_threshold(screening_tolerance);
    policy.item_cutoff_ceiling = policy.threshold.value_or(0.0);
    return policy;
  }
  switch (*precision_mode) {
    case GENERATIVEQC_PRECISION_FP64:
      return policy;
    case GENERATIVEQC_PRECISION_AUTO: {
      const std::optional<double> override_value =
          parsed_mixed_precision_override(screening_tolerance);
      if (override_value.has_value()) {
        // An explicit diagnostic cutoff is deliberately unbudgeted, so it is
        // reported as uncertified rather than as budgeted evidence, and it
        // applies to every item without a census.
        policy.threshold = override_value;
        policy.item_cutoff_ceiling = *override_value;
        return policy;
      }
      const AutoMixedPrecisionAdmission admission =
          admit_auto_mixed_precision_fock(energy_tolerance, screening_tolerance, eligible_tiles);
      if (admission.admitted) {
        policy.threshold = admission.threshold;
        policy.budget_certified = true;
        policy.reserved_error = admission.reserved_error;
        policy.eligible_tiles = admission.eligible_tiles;
        // The certified batch ceiling keeps the tolerance anchor; each item
        // tightens it with its own census below.
        policy.item_cutoff_ceiling = kAutoPrecisionFactor * energy_tolerance;
        policy.item_budget_error = admission.reserved_error;
      }
      return policy;
    }
    default:
      // An unrecognized public policy must never relax the default.
      return policy;
  }
}

MixedPrecisionItemPolicy resolve_mixed_precision_item(const MixedPrecisionFockPolicy& policy,
                                                      bool validated_warm_state,
                                                      std::size_t item_tile_census,
                                                      double screening_tolerance) noexcept {
  MixedPrecisionItemPolicy item;
  if (!policy.threshold.has_value()) return item;
  if (!(policy.item_budget_error > 0.0)) {
    // An explicit diagnostic cutoff is not a budget: keep it item agnostic.
    item.admitted = true;
    item.threshold = policy.item_cutoff_ceiling;
    item.census = 1U;
    return item;
  }
  // The accumulated bound needs the item's own census, and the reserved budget
  // only bounds the perturbation of a known state, so a cold item (or one whose
  // census is unavailable) keeps the exact operator for its whole solve.
  if (!validated_warm_state || item_tile_census == 0) return item;
  const double budget_threshold =
      policy.item_budget_error /
      (kMixedPrecisionFloat32UnitRoundoff * static_cast<double>(item_tile_census));
  const double threshold = std::min(policy.item_cutoff_ceiling, budget_threshold);
  if (!(threshold > screening_tolerance)) return item;
  item.admitted = true;
  item.threshold = threshold;
  item.census = static_cast<std::uint32_t>(item_tile_census);
  return item;
}

bool graph_native_eigensolver_override_requested() noexcept {
  const char* selection = std::getenv("GENERATIVEQC_GRAPH_EIGENSOLVER_OVERRIDE");
  return selection != nullptr && std::strcmp(selection, "graph_native") == 0;
}

bool xsyev_probe_skip_diagnostic_requested() noexcept {
  return selected("GENERATIVEQC_XSYEV_PROBE_SKIP_DIAGNOSTIC", "skip");
}

bool bounded_direct_streaming_override_requested() noexcept {
  return selected("GENERATIVEQC_BOUNDED_DIRECT_STREAMING", "force");
}

bool bounded_direct_count_diagnostic_requested() noexcept {
  return selected("GENERATIVEQC_BOUNDED_DIRECT_COUNT_DIAGNOSTIC", "count");
}

bool bounded_direct_aot_only_diagnostic_requested() noexcept {
  return selected("GENERATIVEQC_BOUNDED_DIRECT_AOT_ONLY_DIAGNOSTIC", "aot");
}

bool bounded_direct_fock_only_diagnostic_requested() noexcept {
  return selected("GENERATIVEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC", "fock");
}

std::optional<std::uint64_t> bounded_direct_primary_streaming_fock_mask_requested() noexcept {
  const char* selection = std::getenv("GENERATIVEQC_BOUNDED_DIRECT_PRIMARY_STREAMING_MASK");
  if (selection == nullptr || *selection == '\0' || std::strcmp(selection, "0") == 0 ||
      std::strcmp(selection, "none") == 0 || *selection == '-') {
    return std::nullopt;
  }
  if (std::strcmp(selection, "all") == 0) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  // strtoull accepts leading whitespace and signs, wrapping negative values
  // into the unsigned range. Diagnostic selection must reject those spellings.
  if (*selection < '0' || *selection > '9') return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(selection, &end, 0);
  if (errno == ERANGE || end == selection || *end != '\0' || value == 0ULL ||
      value > std::numeric_limits<std::uint64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(value);
}

bool bounded_fock_class_timing_requested() noexcept {
  return selected("GENERATIVEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE", "profile");
}

bool aot_shell_class_selection_override_requested() noexcept {
  const char* selection = std::getenv("GENERATIVEQC_AOT_SHELL_CLASSES");
  // Match the generated registry: absent, empty, and "all" all mean the full
  // compiled profile. Any other spelling intentionally narrows the force
  // registry and may therefore exercise the generic fallback for diagnostics.
  return selection != nullptr && *selection != '\0' && std::strcmp(selection, "all") != 0;
}

DirectTileValidationPolicy resolve_direct_tile_validation_policy() noexcept {
  const bool requested = selected("GENERATIVEQC_DIRECT_TILE_VALIDATION", "validate");
  return {
      .requested = requested,
      .produces_numerical_endpoint = !requested,
      .endpoint_status =
          requested ? GENERATIVEQC_STATUS_NOT_IMPLEMENTED : GENERATIVEQC_STATUS_SUCCESS,
  };
}

bool direct_tile_validation_requested() noexcept {
  return resolve_direct_tile_validation_policy().requested;
}

double converged_fock_reuse_density_rms(double density_tolerance) noexcept {
  return density_tolerance >= kExpandedConvergedFockReuseDensityTolerance
             ? kExpandedConvergedFockReuseDensityRms
             : kTightConvergedFockReuseDensityRms;
}

bool force_density_product_screening_requested() noexcept {
  return enabled("GENERATIVEQC_FORCE_DENSITY_PRODUCT_SCREENING");
}

bool bounded_schwarz_schedule_requested() noexcept {
  const char* selection = std::getenv("GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE");
  if (selection == nullptr) return true;
  if (std::strcmp(selection, "0") == 0 || std::strcmp(selection, "none") == 0) return false;
  return std::strcmp(selection, "1") == 0 || std::strcmp(selection, "indexed") == 0 ||
         std::strcmp(selection, "auto") == 0;
}

bool bounded_angular_force_requested() noexcept {
  return selected("GENERATIVEQC_BOUNDED_ANGULAR_FORCE", "angular");
}

bool canonical_rsh_values_requested() noexcept {
  const char* mode = std::getenv("GENERATIVEQC_CANONICAL_RSH_VALUES");
  return mode && (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "shared") == 0);
}

unsigned direct_coulomb_reachable_mode() noexcept {
  const char* mode = std::getenv("GENERATIVEQC_DIRECT_COULOMB_REACHABLE");
  if (mode == nullptr) return 0;
  if (std::strcmp(mode, "values") == 0) return 1;
  if (std::strcmp(mode, "forces") == 0) return 2;
  return std::strcmp(mode, "1") == 0 || std::strcmp(mode, "all") == 0 ||
                 std::strcmp(mode, "reachable") == 0
             ? 3
             : 0;
}

unsigned direct_hermite_convolution_mode() noexcept {
  const char* mode = std::getenv("GENERATIVEQC_DIRECT_HERMITE_CONVOLUTION");
  if (mode == nullptr) return 0;
  if (std::strcmp(mode, "values") == 0) return 1;
  if (std::strcmp(mode, "forces") == 0) return 2;
  return std::strcmp(mode, "1") == 0 || std::strcmp(mode, "all") == 0 ? 3 : 0;
}

bool direct_pair_materialized_values_requested() noexcept {
  return enabled("GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES");
}

bool direct_pair_materialized_derivatives_requested() noexcept {
  return enabled("GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_DERIVATIVES");
}

bool direct_pair_cooperative_derivatives_requested() noexcept {
  return enabled("GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES");
}

bool resident_ppps_bra_requested() noexcept { return enabled("GENERATIVEQC_PPPS_RESIDENT_BRA"); }

bool ppps_signature_bucketing_requested() noexcept {
  return enabled("GENERATIVEQC_PPPS_SIGNATURE_BUCKETING");
}

bool psps_signature_bucketing_requested() noexcept {
  return enabled("GENERATIVEQC_PSPS_SIGNATURE_BUCKETING");
}

bool ppss_signature_bucketing_requested() noexcept {
  return enabled("GENERATIVEQC_PPSS_SIGNATURE_BUCKETING");
}

unsigned ppps_resident_block_threads_requested() noexcept {
  const char* selection = std::getenv("GENERATIVEQC_PPPS_BLOCK_THREADS");
  if (selection == nullptr || std::strcmp(selection, "256") == 0) return 256U;
  if (std::strcmp(selection, "128") == 0) return 128U;
  if (std::strcmp(selection, "64") == 0) return 64U;
  if (std::strcmp(selection, "32") == 0) return 32U;
  return 0U;
}

OneElectronValuePolicy resolve_one_electron_value_policy(
    const runtime::CudaProviderCapabilities& provider) noexcept {
  OneElectronValuePolicy policy;
  const char* selection = std::getenv("GENERATIVEQC_ONE_ELECTRON_VALUE_MAPPING");
  if (selection == nullptr) {
    // Partition nuclear attraction across eight lanes using the same scalar
    // policy. Providers without qualified subgroup operations retain threads.
    policy.mapping = provider.templated_shell_warp_one_electron ? 2U : 0U;
    return policy;
  }
  policy.diagnostic_override = true;
  if (std::strcmp(selection, "nucleus_8") == 0 || std::strcmp(selection, "nucleus_32") == 0) {
    if (!provider.templated_shell_warp_one_electron) {
      policy.capability_fallback = true;
      return policy;
    }
    policy.mapping = std::strcmp(selection, "nucleus_8") == 0 ? 2U : 3U;
    return policy;
  }
  const bool requests_shell_warp =
      std::strcmp(selection, "1") == 0 || std::strcmp(selection, "shell_warp") == 0;
  if (requests_shell_warp && !provider.templated_shell_warp_one_electron) {
    policy.capability_fallback = true;
    return policy;
  }
  policy.mapping = requests_shell_warp ? 1U : 0U;
  return policy;
}

unsigned one_electron_value_mapping_requested() noexcept {
  return resolve_one_electron_value_policy(runtime::active_cuda_provider()).mapping;
}

unsigned one_electron_derivative_mapping_requested() noexcept {
  const char* selection = std::getenv("GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_MAPPING");
  if (selection == nullptr) return NucleusCooperativeSchedule::schedule_code;
  if (std::strcmp(selection, "serial") == 0 || std::strcmp(selection, "2") == 0) return 2U;
  if (std::strcmp(selection, "nucleus_cooperative") == 0 ||
      std::strcmp(selection, "cooperative") == 0 || std::strcmp(selection, "3") == 0)
    return NucleusCooperativeSchedule::schedule_code;
  if (std::strcmp(selection, "shell_warp") == 0 || std::strcmp(selection, "1") == 0) return 1U;
  return 0U;
}

bool resident_psss_bra_requested() noexcept { return enabled("GENERATIVEQC_PSSS_RESIDENT_BRA"); }

unsigned df_derivative_mapping_requested() noexcept {
  return selected("GENERATIVEQC_DF_DERIVATIVE_MAPPING", "serial") ? 1U : 0U;
}

bool df_value_math_requested(unsigned& math) noexcept {
  math = 0;
  const char* value = std::getenv("GENERATIVEQC_DF_VALUE_MATH");
  if (!value || selected("GENERATIVEQC_DF_VALUE_MATH", "auto") ||
      selected("GENERATIVEQC_DF_VALUE_MATH", "generic"))
    return true;
  if (selected("GENERATIVEQC_DF_VALUE_MATH", "candidate")) {
    math = 3;
    return true;
  }
  if (selected("GENERATIVEQC_DF_VALUE_MATH", "polynomial")) {
    math = 1;
    return true;
  }
  if (selected("GENERATIVEQC_DF_VALUE_MATH", "rys")) {
    math = 2;
    return true;
  }
  return false;
}

bool df_value_raw_lanes_requested(unsigned& lanes) noexcept {
  lanes = 1;
  const char* value = std::getenv("GENERATIVEQC_DF_VALUE_RAW_MAPPING");
  if (!value || selected("GENERATIVEQC_DF_VALUE_RAW_MAPPING", "auto") ||
      selected("GENERATIVEQC_DF_VALUE_RAW_MAPPING", "scalar"))
    return true;
  if (selected("GENERATIVEQC_DF_VALUE_RAW_MAPPING", "candidate")) {
    lanes = kDfCandidateRawSchedule;
    return true;
  }
  if (selected("GENERATIVEQC_DF_VALUE_RAW_MAPPING", "subgroup")) {
    lanes = 4;
    return true;
  }
  if (selected("GENERATIVEQC_DF_VALUE_RAW_MAPPING", "warp")) {
    lanes = 32;
    return true;
  }
  return false;
}

unsigned df_value_mapping_requested() noexcept {
  // Auto is resolved by the compiler separately for raw and transformed
  // consumers. A transformed output's auxiliary reduction is absent in raw
  // tiles; sharing its full warp schedule wastes lanes on short contractions.
  if (std::getenv("GENERATIVEQC_DF_VALUE_MAPPING") == nullptr ||
      selected("GENERATIVEQC_DF_VALUE_MAPPING", "auto"))
    return 3U;
  if (selected("GENERATIVEQC_DF_VALUE_MAPPING", "component")) return 1U;
  if (selected("GENERATIVEQC_DF_VALUE_MAPPING", "primitive")) return 2U;
  return 0U;
}

}  // namespace generativeqc::scf::cuda_policy
