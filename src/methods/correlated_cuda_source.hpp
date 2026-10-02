#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "posthf/capacity.hpp"

namespace generativeqc::methods::detail {

struct CorrelatedCudaSourcePlan {
  std::size_t device_bytes{};
  std::size_t one_electron_phase_bytes{};
  std::size_t direct_phase_bytes{};
  std::size_t peak_bytes{};
  bool admitted{};
};

/** Allocation-free envelope for one generic-only prepared Direct source.
 *
 * The native CUDA reference has already finished. Its detached arrays coexist
 * with both preparation phases. The one-electron exporter releases its device
 * buffers before Direct packing/allocation begins. Scalar coefficients below
 * cover the shared HostBatch's vector capacities and growth temporaries, its
 * public/Cartesian AO metadata, shell pairs, warm matrices and PSSS task table.
 * The task count uses the compiler-owned schedule width supplied by the caller.
 * Device bytes come from the existing authoritative Direct resource query;
 * passing precisely that allowance disables unused optional J/K owners.
 */
inline CorrelatedCudaSourcePlan plan_correlated_cuda_source(
    std::size_t n, std::size_t cart, std::size_t atoms, std::size_t shells, std::size_t primitives,
    std::size_t s_shells, std::size_t p_shells, std::size_t psss_threads,
    std::size_t molecular_source_bytes, std::size_t reference_bytes,
    std::size_t direct_device_bytes, std::size_t budget) {
  using posthf::checked_add;
  using posthf::checked_mul;
  if (!n || cart < n || !atoms || !shells || !primitives || !psss_threads || s_shells > shells ||
      p_shells > shells - s_shells || !direct_device_bytes)
    throw std::invalid_argument("invalid correlated CUDA source resource shape");
  const auto pairs = checked_mul(shells, checked_add(shells, 1)) / 2;
  const auto ket_pairs = checked_mul(s_shells, checked_add(s_shells, 1)) / 2;
  const auto ket_chunks = ket_pairs / psss_threads + (ket_pairs % psss_threads != 0);
  const auto tasks = checked_mul(checked_mul(s_shells, p_shells), ket_chunks);
  const auto host_packing = [&](std::size_t public_aos, std::size_t task_count) {
    auto bytes = checked_add(4096, checked_mul(4, molecular_source_bytes));
    for (const auto term :
         {checked_mul(128, atoms), checked_mul(128, shells), checked_mul(128, public_aos),
          checked_mul(64, cart), checked_mul(64, primitives), checked_mul(128, pairs),
          checked_mul(64, task_count), checked_mul(32, checked_mul(public_aos, public_aos)),
          checked_mul(16, checked_mul(public_aos, cart))})
      bytes = checked_add(bytes, term);
    return bytes;
  };
  CorrelatedCudaSourcePlan plan;
  plan.device_bytes = direct_device_bytes;
  // Matrix-only packing omits PSSS tasks. Its host and uploaded metadata,
  // Cartesian pair lists, output matrices and public transform coexist.
  const auto one_electron =
      checked_add(checked_mul(2, host_packing(cart, 0)), checked_mul(128, checked_mul(cart, cart)));
  plan.one_electron_phase_bytes = checked_add(reference_bytes, one_electron);
  // The two public one-electron matrices survive while Direct is prepared.
  const auto direct = checked_add(
      checked_add(host_packing(n, tasks), checked_mul(16, checked_mul(n, n))), direct_device_bytes);
  plan.direct_phase_bytes = checked_add(reference_bytes, direct);
  plan.peak_bytes = std::max(plan.one_electron_phase_bytes, plan.direct_phase_bytes);
  plan.admitted = plan.peak_bytes <= budget;
  return plan;
}

}  // namespace generativeqc::methods::detail
