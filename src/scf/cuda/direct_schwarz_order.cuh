#pragma once

#include "generated_direct_source_contraction.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** A Schwarz diagonal (ij|ij) has even total angular order. Separate launches
 * keep unreachable high-order recurrence frames out of the CUDA local-memory
 * live set. Each AO pair belongs to exactly one pass; normalization, spherical
 * expansions and primitive arithmetic remain in the existing evaluator.
 */
template <unsigned Order, bool Cartesian>
__device__ bool schwarz_order_matches(const DeviceBatch& batch, std::int32_t system, std::int32_t i,
                                      std::int32_t j) {
  const auto base = static_cast<std::int64_t>(system) * (Cartesian ? batch.direct_nbf : batch.nbf);
  const auto* shells = Cartesian ? batch.direct_ao_shells : batch.ao_shells;
  return 2U * (batch.shell_angular[shells[base + i]] + batch.shell_angular[shells[base + j]]) ==
         Order;
}

template <unsigned Order, bool Cartesian>
__device__ double schwarz_diagonal(const DeviceBatch& batch, std::int32_t system, std::int32_t i,
                                   std::int32_t j) {
  if constexpr (Cartesian) {
    const auto base = static_cast<std::int64_t>(system) * batch.direct_nbf;
    const auto a = batch.shell_angular[batch.direct_ao_shells[base + i]];
    const auto b = batch.shell_angular[batch.direct_ao_shells[base + j]];
    return dispatch_contracted_eri_cartesian_source_shell_class<Order, double>(
        direct_quartet_shell_class_device(a, b, a, b), batch, system, i, j, i, j, -1);
  } else {
    return contracted_eri_order<Order, double>(batch, system, i, j, i, j, -1);
  }
}

/** Unknown basis bounds keep all supported orders; proved bounds omit them
 * on the host, before loading or launching a larger private recurrence frame.
 */
template <unsigned Order = 0, class Launch>
void for_each_schwarz_order(const DeviceBatch& batch, const Launch& launch) {
  if constexpr (Order > 0) {
    if (Order > 4U * batch.direct_maximum_shell_angular) return;
  }
  launch.template operator()<Order>();
  if constexpr (Order < 12) for_each_schwarz_order<Order + 2>(batch, launch);
}

}  // namespace generativeqc::scf::cuda_execution
