#pragma once

#include <cstddef>
#include <cstdlib>

#include "scf/cuda/rhf_policy.hpp"

namespace generativeqc::dft {

/** Large KS residuals amortize the pending-row launch and separately charged
 * physical-slot Gram cache. Small owners retain the serial schedule. The
 * resource query and constructor must use the same admission rule; the plan
 * then retains its reducer through warm replay and final closure.
 *
 * Explicit shared selectors keep their existing meaning, including the serial
 * rollback and alternative cooperative reducer. HF selection is unaffected. */
inline bool cuda_ks_incremental_diis_gram_requested(std::size_t aos, unsigned history) {
  if (history < 2) return false;
  if (std::getenv("GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM"))
    return scf::cuda_execution::incremental_diis_gram_requested();
  if (aos < 512) return false;
  // An automatically admitted cache also makes the subordinate selector active.
  // Validate it during capacity queries, before any owner can be allocated.
  (void)scf::cuda_execution::ordered_incremental_diis_gram_requested();
  return true;
}

}  // namespace generativeqc::dft
