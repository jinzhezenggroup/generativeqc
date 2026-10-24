#pragma once

#include <cstdint>
#include <optional>

#include "scf/types.hpp"

namespace generativeqc::scf {

/** Export completed HF work without estimating from returned SCF iterations.
 * Direct CUDA owns an executed application counter; CUDA DF RHF owns an explicit
 * assembly counter plus final physical evaluations. Failed/discarded attempts
 * remain unavailable until their own complete census can be joined safely.
 */
inline std::optional<std::uint64_t> completed_hf_fock_work(const ScfResult& result,
                                                           bool discarded_warm_attempt,
                                                           bool counted_cuda_df_rhf) noexcept {
  const auto& guess = result.preliminary_guess;
  if (!result.converged || discarded_warm_attempt || !guess.work_counters_complete ||
      guess.target_attempts > 1 || guess.discarded_target_iterations ||
      guess.discarded_target_fock_builds)
    return std::nullopt;
  if (counted_cuda_df_rhf)
    return result.fock_builds ? std::optional<std::uint64_t>{result.fock_builds} : std::nullopt;
  return result.executed_fock_builds && *result.executed_fock_builds ? result.executed_fock_builds
                                                                     : std::nullopt;
}

}  // namespace generativeqc::scf
