#pragma once

#include "scf/cuda/direct_force_work.hpp"
#include "scf/cuda/direct_queue_index.cuh"

namespace generativeqc::scf::cuda_execution {

/** Resolve optional executed-work storage using the physical shell class. */
__device__ inline DirectForceClassWork* direct_force_class_work(const DeviceBatch& batch,
                                                                unsigned shell_class) {
  return batch.direct_force_work && shell_class < detail::kDirectQuartetShellClassCount
             ? &batch.direct_force_work->classes[shell_class]
             : nullptr;
}

__device__ inline DirectForceClassWork* direct_force_task_work(const DeviceBatch& batch,
                                                               const ActiveShellQuartetTile& task) {
  if (!batch.direct_force_work) return nullptr;
  const auto first = task.first_pair, second = task.second_pair;
  return direct_force_class_work(batch, direct_quartet_shell_class_device(
                                            batch.shell_angular[batch.shell_pair_first[first]],
                                            batch.shell_angular[batch.shell_pair_second[first]],
                                            batch.shell_angular[batch.shell_pair_first[second]],
                                            batch.shell_angular[batch.shell_pair_second[second]]));
}

/** Count primitive-pair products only after an actual consumer's final
 * admission. Low-order precontracted sources traverse this product once per
 * shell; component sources traverse it once per admitted AO component. The
 * caller supplies that multiplicity, not an unscreened shell cardinality.
 */
__device__ inline void profile_direct_force_primitive_products(const DeviceBatch& batch,
                                                               DirectForceClassWork* work,
                                                               std::size_t first,
                                                               std::size_t second,
                                                               unsigned long long traversals = 1) {
  if (!work) return;
  const auto bra =
      batch.shell_pair_primitive_offsets[first + 1] - batch.shell_pair_primitive_offsets[first];
  const auto ket =
      batch.shell_pair_primitive_offsets[second + 1] - batch.shell_pair_primitive_offsets[second];
  atomicAdd(&work->primitive_pair_products,
            traversals * static_cast<unsigned long long>(bra) * ket);
}

}  // namespace generativeqc::scf::cuda_execution
