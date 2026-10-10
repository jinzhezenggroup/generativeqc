#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <utility>

#include "runtime/residency_observer.hpp"

namespace generativeqc::runtime {

/** Versioned source declarations, not classifications inferred from addresses,
 * byte sizes, enclosing public execute phases or GPU-related filenames. */
enum class ResidencyOwner : std::uint64_t {
  hf_bucket = 1,
  hf_graph_setup = 2,
  hf_bucket_resources = 3,
  hf_eigensolver_resources = 4,
  device_resource_ledger = 5,
  posthf_df_source = 6,
};
enum class ResidencyRole : std::uint64_t {
  prepare = 1,
  iteration = 2,
  tile = 3,
  publication = 4,
  oracle = 5,
  compatibility = 6,
  lifetime = 7,
};
enum class ResidencySite : std::uint64_t {
  hf_results = 1,
  hf_forces = 2,
  hf_results_fence = 3,
  hf_active = 4,
  hf_active_fence = 5,
  hf_warm_invalid = 6,
  hf_warm_invalid_fence = 7,
  hf_final_fock_count = 8,
  hf_final_fock_fence = 9,
  hf_refinement = 10,
  hf_refinement_fence = 11,
  hf_profile = 12,
  hf_profile_fence = 13,
  hf_validation = 14,
  hf_validation_fence = 15,
  hf_shell_bounds = 16,
  hf_shell_bounds_fence = 17,
  hf_static_inputs = 18,
  hf_dynamic_inputs = 19,
  hf_positions_input = 20,
  hf_energy_seed_input = 21,
  hf_shell_order_inputs = 22,
  hf_graph_capture_fence = 23,
  hf_graph_upload_fence = 24,
  hf_bucket_release_fence = 25,
  hf_eigensolver_release_fence = 26,
  resource_ledger_release_fence = 27,
  resource_ledger_rollback_fence = 28,
  posthf_df_generation_fence = 29,
  posthf_df_raw_tile = 30,
  posthf_df_publication_fence = 31,
  posthf_df_release_fence = 32,
  posthf_df_handoff_fence = 33,
};
enum class ResidencyPayload : std::uint64_t {
  none = 0,
  energy = 1,
  energy_change = 2,
  density_rms = 3,
  density = 4,
  converged = 5,
  failed = 6,
  iterations = 7,
  forces = 8,
  incremental_max_delta = 9,
  incremental_full_count = 10,
  incremental_delta_count = 11,
  incremental_full_quartets = 12,
  incremental_delta_quartets = 13,
  incremental_full_tiles = 14,
  incremental_delta_tiles = 15,
  final_fock_reuse_mask = 16,
  final_audit_mask = 17,
  active = 18,
  warm_invalid = 19,
  final_fock_count = 20,
  shell_bounds = 21,
  validation = 22,
  task_counts = 23,
  task_overflow = 24,
  elapsed_ticks = 25,
  launch_counts = 26,
  fp64_work = 27,
  fp32_work = 28,
  inactive_profile_count = 29,
  inactive_profile = 30,
  shell_profile = 31,
  ppps_counts = 32,
  ppps_signatures = 33,
  input_atom_offsets = 34,
  input_atom_systems,
  input_atomic_numbers,
  input_system_shell_offsets,
  input_shell_atoms,
  input_shell_angular,
  input_shell_ao_offsets,
  input_shell_direct_ao_offsets,
  input_shell_primitive_offsets,
  input_system_shell_pair_offsets,
  input_system_shell_quartet_offsets,
  input_system_shell_pair_block_offsets,
  input_system_shell_pair_block_quartet_offsets,
  input_shell_pair_systems,
  input_shell_pair_first,
  input_shell_pair_second,
  input_shell_pair_primitive_offsets,
  input_psss_resident_tasks,
  input_psss_resident_ket_pairs,
  input_ao_shells,
  input_ao_term_counts,
  input_ao_term_angular,
  input_ao_term_coefficients,
  input_direct_ao_shells,
  input_direct_ao_angular,
  input_direct_ao_coefficients,
  input_ao_to_direct_transform,
  input_primitive_exponents,
  input_primitive_coefficients,
  input_occupied,
  input_shell_quartet_tile_offsets,
  input_fp32_shell_quartet_tile_offsets,
  input_bounded_generated_task_offsets,
  input_bounded_direct_shell_pair_order,
  input_bounded_stream_shell_pair_order,
  input_bounded_stream_pair_class_offsets,
  input_bounded_stream_topology,
  input_ao_pair_first,
  input_ao_pair_second,
  input_mixed_precision_item_census,
  input_warm_mask,
  input_warm_density,
  input_generated_fock_shell_class_mask,
  input_generated_mixed_fock_shell_class_mask,
  input_positions,
  input_previous_energy_seed,
  df_raw_three_center,
};
enum class ResidencyOperationKind : std::uint64_t { transfer = 1, stream_sync = 2, event_sync = 3 };
enum class ResidencyDirection : std::uint64_t { none = 0, h2d = 1, d2h = 2, d2d = 3 };
inline constexpr std::size_t kResidencyBoundaryFields = 14;
inline std::atomic<std::uint64_t> residency_boundary_generation{1};

/** Names are read from the same native artifact that emits the numeric tags.
 * No separately maintained Python table can silently reinterpret a source ID. */
inline const char* residency_boundary_name(std::uint64_t category, std::uint64_t value) noexcept {
  static constexpr std::array owners{"unknown",
                                     "hf_bucket",
                                     "hf_graph_setup",
                                     "hf_bucket_resources",
                                     "hf_eigensolver_resources",
                                     "device_resource_ledger",
                                     "posthf_df_source"};
  static constexpr std::array roles{"unknown",     "prepare", "iteration",     "tile",
                                    "publication", "oracle",  "compatibility", "lifetime"};
  static constexpr std::array sites{"unknown",
                                    "hf.results",
                                    "hf.forces",
                                    "hf.results-fence",
                                    "hf.active",
                                    "hf.active-fence",
                                    "hf.warm-invalid",
                                    "hf.warm-invalid-fence",
                                    "hf.final-fock-count",
                                    "hf.final-fock-fence",
                                    "hf.refinement",
                                    "hf.refinement-fence",
                                    "hf.profile",
                                    "hf.profile-fence",
                                    "hf.validation",
                                    "hf.validation-fence",
                                    "hf.shell-bounds",
                                    "hf.shell-bounds-fence",
                                    "hf.static-inputs",
                                    "hf.dynamic-inputs",
                                    "hf.positions-input",
                                    "hf.energy-seed-input",
                                    "hf.shell-order-inputs",
                                    "hf.graph-capture-fence",
                                    "hf.graph-upload-fence",
                                    "hf.bucket-release-fence",
                                    "hf.eigensolver-release-fence",
                                    "resource-ledger.release-fence",
                                    "resource-ledger.rollback-fence",
                                    "posthf.df.generation-fence",
                                    "posthf.df.raw-tile",
                                    "posthf.df.publication-fence",
                                    "posthf.df.release-fence",
                                    "posthf.df.handoff-fence"};
  static constexpr std::array payloads{"none",
                                       "energy",
                                       "energy_change",
                                       "density_rms",
                                       "density",
                                       "converged",
                                       "failed",
                                       "iterations",
                                       "forces",
                                       "incremental_max_delta",
                                       "incremental_full_count",
                                       "incremental_delta_count",
                                       "incremental_full_quartets",
                                       "incremental_delta_quartets",
                                       "incremental_full_tiles",
                                       "incremental_delta_tiles",
                                       "final_fock_reuse_mask",
                                       "final_audit_mask",
                                       "active",
                                       "warm_invalid",
                                       "final_fock_count",
                                       "shell_bounds",
                                       "validation",
                                       "task_counts",
                                       "task_overflow",
                                       "elapsed_ticks",
                                       "launch_counts",
                                       "fp64_work",
                                       "fp32_work",
                                       "inactive_profile_count",
                                       "inactive_profile",
                                       "shell_profile",
                                       "ppps_counts",
                                       "ppps_signatures",
                                       "input.atom_offsets",
                                       "input.atom_systems",
                                       "input.atomic_numbers",
                                       "input.system_shell_offsets",
                                       "input.shell_atoms",
                                       "input.shell_angular",
                                       "input.shell_ao_offsets",
                                       "input.shell_direct_ao_offsets",
                                       "input.shell_primitive_offsets",
                                       "input.system_shell_pair_offsets",
                                       "input.system_shell_quartet_offsets",
                                       "input.system_shell_pair_block_offsets",
                                       "input.system_shell_pair_block_quartet_offsets",
                                       "input.shell_pair_systems",
                                       "input.shell_pair_first",
                                       "input.shell_pair_second",
                                       "input.shell_pair_primitive_offsets",
                                       "input.psss_resident_tasks",
                                       "input.psss_resident_ket_pairs",
                                       "input.ao_shells",
                                       "input.ao_term_counts",
                                       "input.ao_term_angular",
                                       "input.ao_term_coefficients",
                                       "input.direct_ao_shells",
                                       "input.direct_ao_angular",
                                       "input.direct_ao_coefficients",
                                       "input.ao_to_direct_transform",
                                       "input.primitive_exponents",
                                       "input.primitive_coefficients",
                                       "input.occupied",
                                       "input.shell_quartet_tile_offsets",
                                       "input.fp32_shell_quartet_tile_offsets",
                                       "input.bounded_generated_task_offsets",
                                       "input.bounded_direct_shell_pair_order",
                                       "input.bounded_stream_shell_pair_order",
                                       "input.bounded_stream_pair_class_offsets",
                                       "input.bounded_stream_topology",
                                       "input.ao_pair_first",
                                       "input.ao_pair_second",
                                       "input.mixed_precision_item_census",
                                       "input.warm_mask",
                                       "input.warm_density",
                                       "input.generated_fock_shell_class_mask",
                                       "input.generated_mixed_fock_shell_class_mask",
                                       "input.positions",
                                       "input.previous_energy_seed",
                                       "df.raw_three_center"};
  static_assert(owners.size() == static_cast<std::size_t>(ResidencyOwner::posthf_df_source) + 1);
  static_assert(roles.size() == static_cast<std::size_t>(ResidencyRole::lifetime) + 1);
  static_assert(sites.size() ==
                static_cast<std::size_t>(ResidencySite::posthf_df_handoff_fence) + 1);
  static_assert(payloads.size() ==
                static_cast<std::size_t>(ResidencyPayload::df_raw_three_center) + 1);
  switch (category) {
    case 1:
      return value > 0 && value < owners.size() ? owners[value] : nullptr;
    case 2:
      return value > 0 && value < roles.size() ? roles[value] : nullptr;
    case 3:
      return value > 0 && value < sites.size() ? sites[value] : nullptr;
    case 4:
      return value < payloads.size() ? payloads[value] : nullptr;
    default:
      return nullptr;
  }
}

inline std::uint64_t next_residency_boundary_generation() noexcept {
  auto current = residency_boundary_generation.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<std::uint64_t>::max()) {
    if (residency_boundary_generation.compare_exchange_weak(current, current + 1,
                                                            std::memory_order_relaxed))
      return current;
  }
  return 0;
}

/** One declared source invocation, distinct from an enclosing public endpoint.
 * A generation groups explicit source fields without identifying them by memory
 * addresses. Closing the scope proves lifetime closure, not scientific success. */
class ResidencyExecution {
 public:
  explicit ResidencyExecution(ResidencyOwner owner) noexcept : owner_(owner) {
    if (residency_observation.callback) {
      generation_ = next_residency_boundary_generation();
      emit(3);
    }
  }
  ResidencyExecution(const ResidencyExecution&) = delete;
  ResidencyExecution& operator=(const ResidencyExecution&) = delete;
  ~ResidencyExecution() {
    if (generation_) emit(4);
  }
  std::uint64_t generation() const noexcept { return generation_; }
  ResidencyOwner owner() const noexcept { return owner_; }

 private:
  void emit(std::uint64_t kind) const noexcept {
    const std::array<std::uint64_t, kResidencyBoundaryFields> values{
        2, kind, 0, generation_, static_cast<std::uint64_t>(owner_)};
    dispatch_residency_source(values.data(), values.size());
  }
  ResidencyOwner owner_;
  std::uint64_t generation_{};
};

/** Wrap exactly one source operation, retaining failed/abandoned submissions.
 * Payload instances are source-owned generations of named fields, not proof of
 * producer kernels or host-transform dependencies. An optional dependency ID
 * must be supplied by its real owner; default zero means no link is asserted. */
class ResidencyBoundary {
 public:
  ResidencyBoundary(const ResidencyExecution& execution, ResidencyOperationKind operation,
                    ResidencyRole role, ResidencySite site, ResidencyPayload payload,
                    ResidencyDirection direction, std::uint64_t bytes,
                    std::uint64_t payload_instance = 0, std::uint64_t dependency = 0) noexcept {
    if (!execution.generation() || !residency_observation.callback) return;
    const auto generation = next_residency_boundary_generation();
    values_ = {
        2,
        1,
        generation,
        execution.generation(),
        static_cast<std::uint64_t>(execution.owner()),
        static_cast<std::uint64_t>(role),
        static_cast<std::uint64_t>(site),
        static_cast<std::uint64_t>(payload),
        payload == ResidencyPayload::none ? 0 : (payload_instance ? payload_instance : generation),
        dependency,
        static_cast<std::uint64_t>(operation),
        static_cast<std::uint64_t>(direction),
        bytes,
        0};
    active_ = true;
    dispatch_residency_source(values_.data(), values_.size());
  }
  ResidencyBoundary(const ResidencyBoundary&) = delete;
  ResidencyBoundary& operator=(const ResidencyBoundary&) = delete;
  ~ResidencyBoundary() {
    if (active_) finish(std::numeric_limits<std::uint64_t>::max());
  }
  void finish(std::uint64_t status) noexcept {
    if (!active_) return;
    values_[1] = 2;
    values_[13] = status;
    dispatch_residency_source(values_.data(), values_.size());
    active_ = false;
  }

 private:
  std::array<std::uint64_t, kResidencyBoundaryFields> values_{};
  bool active_{};
};

}  // namespace generativeqc::runtime
