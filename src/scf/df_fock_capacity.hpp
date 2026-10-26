#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "scf/density_fitting.hpp"

namespace generativeqc::scf {

/** Numeric capacity of one source-backed prepared Fock owner. The retained
 * allowance includes its future DF response and value-plan setup workspace.
 * Initial source construction instead uploads a temporary auxiliary metric
 * before that plan exists. Neither field includes one-electron export, SCF/XC,
 * opaque providers or independent method/derivative owners.
 */
struct DfFockDeviceCapacity {
  std::size_t retained_response_bytes{}, source_setup_bytes{};
};

/** Compose the actual native DF policy leaves without a live-memory probe.
 * exact_bytes bounds an already retained Direct provider (zero for pure DF).
 * The shared explicit budget includes that provider; automatic resident
 * promotion acts on the remaining DF subbudget and must add Direct back.
 * Source metadata must bound the real combined orbital/auxiliary/dummy upload.
 * This singleton method owner does not own the batched RHF DIIS/SCF state.
 */
inline DfFockDeviceCapacity df_fock_device_capacity(
    DfBudgetWorkload workload, std::size_t requested_bytes, std::size_t exact_bytes,
    std::size_t source_bytes, std::size_t restricted_rank, bool allow_zero_rank_coulomb,
    bool automatic_pair_storage) {
  if (!workload.nbf || !workload.naux || !workload.atoms || workload.batch != 1 ||
      workload.diis_history || restricted_rank > workload.nbf)
    throw std::invalid_argument("invalid singleton DF Fock capacity shape");
  const auto add = [](std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::overflow_error("DF Fock capacity overflows size_t");
    return a + b;
  };
  const auto targets = df_budget_targets(workload);
  const auto base =
      requested_bytes ? requested_bytes : std::max(targets.workload_bytes, targets.resident_bytes);
  // Mixed preparation gives Direct at most half the pre-promotion budget.
  // A supplied native Direct bound may be larger; such a preparation either
  // fits the smaller allowance or fails before it creates the DF source.
  const auto exact = std::min(exact_bytes, base / 2);
  auto retained = base;
  if (!requested_bytes && automatic_pair_storage && workload.forces &&
      (restricted_rank || allow_zero_rank_coulomb))
    retained = std::max(
        retained, add(exact, method_owned_df_resident_capacity(workload.nbf, workload.naux,
                                                               restricted_rank, source_bytes)));
  if (workload.naux > std::numeric_limits<std::size_t>::max() / sizeof(double) / workload.naux)
    throw std::overflow_error("DF source metric capacity overflows size_t");
  const auto metric = workload.naux * workload.naux * sizeof(double);
  return {retained, add(exact, add(source_bytes, metric))};
}

}  // namespace generativeqc::scf
