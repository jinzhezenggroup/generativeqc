#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "scf/density_fitting.hpp"
#include "scf/df_source_capacity.hpp"

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

/** Shape-only device inventory of the combined orbital/auxiliary/dummy source.
 * Use the same conservative public-AO transform bound as prepared Fock when
 * deciding whether to promote a streamed value owner to resident storage. */
inline std::size_t df_fock_source_metadata_bytes(const df_source_capacity::Shape& orbital,
                                                 const df_source_capacity::Shape& auxiliary) {
  using df_source_capacity::add;
  using df_source_capacity::mul;
  for (const auto* basis : {&orbital, &auxiliary}) {
    if (!basis->atoms || !basis->shells || basis->primitives < basis->shells ||
        basis->public_aos < basis->shells || basis->cartesian_aos < basis->public_aos)
      throw std::invalid_argument("invalid DF Fock basis capacity shape");
  }
  if (orbital.atoms != auxiliary.atoms)
    throw std::invalid_argument("DF Fock capacity requires matching orbital/auxiliary atom counts");
  return density_fitting_source_metadata_bytes(
      1, orbital.atoms, add(add(orbital.shells, auxiliary.shells), 1),
      add(add(orbital.cartesian_aos, auxiliary.cartesian_aos), 1),
      add(add(orbital.primitives, auxiliary.primitives), 1),
      add(mul(orbital.cartesian_aos, orbital.public_aos),
          mul(auxiliary.cartesian_aos, auxiliary.public_aos)));
}

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

/** Upper bound for one prepared radial owner (full or erf) and any exact
 * provider in that owner. Callers compose separate radial owners explicitly;
 * a shared basis does not imply shared fitted metric or value buffers. */
inline DfFockDeviceCapacity prepared_df_fock_device_capacity(
    const df_source_capacity::Shape& orbital, const df_source_capacity::Shape& auxiliary,
    std::size_t requested_bytes, std::size_t exact_bytes, std::size_t restricted_rank,
    bool retain_derivatives, bool allow_zero_rank_coulomb, bool automatic_pair_storage) {
  const DfBudgetWorkload workload{orbital.public_aos, auxiliary.public_aos, orbital.atoms, 1, 0,
                                  retain_derivatives};
  const auto bytes = df_fock_source_metadata_bytes(orbital, auxiliary);
  return df_fock_device_capacity(workload, requested_bytes, exact_bytes, bytes, restricted_rank,
                                 allow_zero_rank_coulomb, automatic_pair_storage);
}

/** Conservative complete pageable envelope for one CUDA source-backed Fock
 * owner. Reserved as retained by KS so no packing/setup/optional storage may
 * overlap other owners without being accounted for. */
std::size_t prepared_df_fock_host_capacity(const df_source_capacity::Shape& orbital,
                                           const df_source_capacity::Shape& auxiliary,
                                           std::size_t fitted_device_cap);

}  // namespace generativeqc::scf
