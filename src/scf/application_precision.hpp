#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace generativeqc::scf {

/** Executable shape admits a compatible native force owner, not an error claim.
 * Empirically qualified AO/atom sizes are tracked separately for publication. */
inline bool admit_application_hf_gradient_shape(std::size_t aos, std::size_t atoms,
                                                std::size_t maximum_angular, bool restricted,
                                                bool all_electron) noexcept {
  return restricted && all_electron && aos >= 24 && atoms >= 2 && maximum_angular <= 2;
}

/** Historical independent HF-gradient qualification, not an execution cap.
 * Out-of-cohort shape candidates carry an audit-required provenance code. */
inline constexpr bool independently_qualified_hf_gradient(std::size_t aos,
                                                          std::size_t atoms) noexcept {
  return aos <= 96 && atoms <= 48;
}

/** Restricted direct-HF derivative candidate at an authenticated strict final
 * state. Full-rank physical proof and bounded resources remain mandatory. */
inline bool admit_application_hf_gradient(const std::array<double, 3>& allowances, std::size_t aos,
                                          std::size_t atoms, std::size_t maximum_angular,
                                          bool restricted, bool all_electron) noexcept {
  for (const auto allowance : allowances)
    if (!std::isfinite(allowance) || allowance < 0.0) return false;
  return admit_application_hf_gradient_shape(aos, atoms, maximum_angular, restricted,
                                             all_electron) &&
         allowances[0] >= 1e-6 && allowances[1] >= 1e-5 &&
         (allowances[2] == 0.0 || allowances[2] >= 3e-6);
}

/** Fitted RHF selects a separate response, not direct-HF orders.
 * Metric conditioning, response weights, generic derivatives and resource
 * admission remain strict. The response validates its final density/metric.
 * Old one-endpoint evidence includes <=96 orbital AOs, <=512 auxiliaries and
 * <=12 atoms; expanded shapes require an independent FP64 E+F audit. */
inline constexpr bool independently_qualified_df_hf_gradient(std::size_t aos,
                                                             std::size_t auxiliary_aos,
                                                             std::size_t atoms) noexcept {
  return independently_qualified_hf_gradient(aos, atoms) && auxiliary_aos <= 512 && atoms <= 12;
}

inline bool admit_application_df_hf_gradient(const std::array<double, 3>& allowances,
                                             std::size_t aos, std::size_t auxiliary_aos,
                                             std::size_t atoms, std::size_t orbital_angular,
                                             std::size_t auxiliary_angular, bool restricted,
                                             bool all_electron) noexcept {
  return admit_application_hf_gradient(allowances, aos, atoms, orbital_angular, restricted,
                                       all_electron) &&
         auxiliary_aos > 0 && auxiliary_angular <= 3;
}

}  // namespace generativeqc::scf
