#ifndef GENERATIVEQC_SCF_INITIAL_GUESS_AUTOMATIC_POLICY_HPP
#define GENERATIVEQC_SCF_INITIAL_GUESS_AUTOMATIC_POLICY_HPP

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf::initial_guess {

/** Execution-only MINAO policy shared by discovery and native admission.
 * Model-only discovery does not replace system, seed-budget, or density gates.
 */
inline bool automatic_minao_model_eligible(generativeqc_method method,
                                           generativeqc_method_family family,
                                           generativeqc_backend backend,
                                           generativeqc_precision_mode precision,
                                           generativeqc_density_fitting_mode density_fitting,
                                           bool restricted, bool has_ecp) noexcept {
  const bool hf = family == GENERATIVEQC_METHOD_FAMILY_HARTREE_FOCK;
  const bool ks = family == GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL;
  return (hf || ks) &&
         (backend == GENERATIVEQC_BACKEND_CPU_REFERENCE ||
          (backend == GENERATIVEQC_BACKEND_CUDA && ks)) &&
         precision == GENERATIVEQC_PRECISION_FP64 &&
         density_fitting == GENERATIVEQC_DENSITY_FITTING_NONE && restricted && !has_ecp &&
         (!hf || method == GENERATIVEQC_METHOD_RHF);
}

/** The occupied-ANO records are independently admitted only for H through Ar. */
inline bool automatic_minao_element_eligible(int32_t atomic_number) noexcept {
  return atomic_number >= 1 && atomic_number <= 18;
}

}  // namespace generativeqc::scf::initial_guess

#endif
