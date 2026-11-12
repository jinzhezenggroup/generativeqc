#pragma once

#include <stdexcept>

#include "core/types.hpp"
#include "scf/fock_build.hpp"

namespace generativeqc::scf::cuda_execution {

/** The shared g-shell integral source is qualified for closed-shell full DF,
 * not unrestricted Fock/SCF execution. Check the consumer before CUDA setup;
 * source value-math, range and auxiliary limits remain independent.
 */
inline void require_cuda_df_spin_domain(const core::System& orbital, FockSpin spin) {
  if (spin != FockSpin::Unrestricted) return;
  for (const auto& shell : orbital.shells)
    if (shell.angular_momentum > 3U)
      throw std::invalid_argument(
          "unrestricted CUDA DF execution supports orbital shells only through f (l <= 3)");
}

}  // namespace generativeqc::scf::cuda_execution
