#pragma once
#include <cstddef>
#include <cstdint>

namespace generativeqc::cc::phase_observation {
// Private source protocol, not a numerical/public C API. External collectors
// include only this contract; they must not instantiate another native TLS copy.
using Observer = void (*)(const std::uint64_t*, std::size_t, void*);
inline constexpr std::uint64_t kPhaseLayout = 0x434350480001ULL;
inline constexpr std::uint64_t kAdmissionLayout = 0x434350410001ULL;
inline constexpr std::uint64_t kDetailedContract = 0x434343500002ULL;
inline constexpr std::uint64_t kTriplesAdmissionLayout = 0x434354410001ULL;
inline constexpr std::size_t kPhaseWords = 12, kAdmissionWords = 14;
inline constexpr std::size_t kTriplesAdmissionWords = 17;
// Detailed contract v2 adds the previously inactive DF/(T) sites and the
// triples record. The original phase-layout opt-in preserves its Q-only view.
// Triples words: layout, event, ID, parent, o, v, q, panel slots, paired pointer
// bytes, selected lowering, storage/compute/accumulation bits, maximum budget,
// reserved layout bytes, permit, status. Reserved bytes are not live/peak usage.
// Admission words: layout, event, snapshot ID, parent, owner, naux, primal
// batch, effective replay width, route, maximum budget, numeric high-water bytes, device
// high-water bytes, permit, status. The last two capacity diagnostics preserve
// Owner's existing peak/max accounting (including provider peaks); they are
// not exact current live allocations and need not describe the budget's scope.
// Replay width is zero for no Q work, one for a positive-Q scalar fallback;
// it is projected through the same rule as the actual replay Q traversal.
enum class Site : std::uint64_t {
  endpoint_inclusive = 1,
  source_inclusive = 2,
  source_gpu_submission = 3,
  q_tile_inclusive = 4,
  q_host_preparation = 5,
  q_production_submission = 6,
  q_accumulation_submission = 7,
  triples_inclusive = 8,
  triples_host_preparation = 9,
  triples_panel_submission = 10,
  triples_moment_submission = 11,
  triples_epilogue_submission = 12,
  explicit_wait = 13,
  q_iteration_inclusive = 14,
  q_replay_inclusive = 15,
  ccsd_solve_inclusive = 16,
  setup_inclusive = 17,
  teardown_inclusive = 18,
  ccsd_preparation_submission = 19,
  ccsd_core_submission = 20,
  q_mixed_inclusive = 21,
  source_transform_submission = 22,
  source_pack_submission = 23,
  source_copy_submission = 24,
  source_validation = 25,
  source_host_preparation = 26,
  source_block_submission = 27,
  triples_tile_inclusive = 28,
  triples_pair_submission = 29,
  triples_reduction_submission = 30,
  triples_download_submission = 31,
  triples_seed_preparation = 32
};
enum class PlanOwner : std::uint64_t { ccsd = 1, triples = 2, df_source = 3 };
enum PlanRoute : std::uint64_t {
  hoisted = 1,
  matrix = 2,
  replay_matrix = 4,
  mixed = 8,
  occupied_pairs = 16,
  conventional = 32
};
}  // namespace generativeqc::cc::phase_observation

extern "C" {
// Validate an already-bound live native owner; never registers/replaces it.
int generativeqc_cc_phase_enable_v1(std::uint64_t contract,
                                    generativeqc::cc::phase_observation::Observer observer,
                                    void* context, std::uint64_t* permit);
// Refuse open source phases. Errors include changed binding or callback failure.
int generativeqc_cc_phase_finish_v1(std::uint64_t permit, std::uint64_t* errors);
}
