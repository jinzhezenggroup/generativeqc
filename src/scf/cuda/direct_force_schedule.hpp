#pragma once

#include <cstddef>

#include "scf/cuda/direct_metadata.hpp"

namespace generativeqc::scf::cuda_execution {

/** Optional resident-bra lease shared by HF and stationary force consumers.
 * Tasks and ket indices cover the complete psss domain, in immutable pair-cache
 * order. The plan owns charged storage and stream-ordered lifetime. An empty
 * lease, unsupported capacity or missing view retains bounded queue execution. */
struct DirectForceResidentBraSchedule {
  const PsssResidentTask* tasks{};
  const std::uint32_t* ket_pairs{};
  std::size_t task_count{};
  std::size_t bra_primitive_pair_capacity{};
};

/** Capacity admission is independent of method/output mode and performs no
 * allocation or GPU probe. It uses the compiler-owned resident schedule limit. */
bool direct_force_resident_bra_capacity_supported(std::size_t task_count,
                                                  std::size_t primitive_pair_capacity) noexcept;

/** A lease is usable only after both complete device views have been retained. */
bool direct_force_resident_bra_schedule_available(DirectForceResidentBraSchedule schedule) noexcept;

}  // namespace generativeqc::scf::cuda_execution
