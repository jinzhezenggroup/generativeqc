#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace generativeqc::scf {

/** Shape qualification shared by arithmetic selection and derivative ownership.
 * Changing the force owner does not by itself authorize narrowed arithmetic. */
inline bool admit_application_hf_gradient_shape(std::size_t aos, std::size_t atoms,
                                                std::size_t maximum_angular, bool restricted,
                                                bool all_electron) noexcept {
  return restricted && all_electron && aos >= 24 && aos <= 96 && atoms >= 2 && atoms <= 48 &&
         maximum_angular <= 2;
}

/** Restricted direct-HF derivative candidate at an authenticated strict final
 * state. Qualification is bounded independently of the SCF precision request;
 * an explicit observable target still requires its separate endpoint audit. */
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

/** Fitted RHF qualifies a separate complete response, not direct-HF orders.
 * Metric conditioning, response weights and generic derivatives stay strict.
 * The force owner must additionally authenticate its final density and metric. */
inline bool admit_application_df_hf_gradient(const std::array<double, 3>& allowances,
                                             std::size_t aos, std::size_t auxiliary_aos,
                                             std::size_t atoms, std::size_t orbital_angular,
                                             std::size_t auxiliary_angular, bool restricted,
                                             bool all_electron) noexcept {
  return admit_application_hf_gradient(allowances, aos, atoms, orbital_angular, restricted,
                                       all_electron) &&
         atoms <= 12 && auxiliary_aos > 0 && auxiliary_aos <= 512 && auxiliary_angular <= 3;
}

}  // namespace generativeqc::scf
