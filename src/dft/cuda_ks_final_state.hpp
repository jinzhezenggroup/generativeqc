#pragma once

#include <cstdint>

#include "dft/ks_final_state.hpp"

namespace generativeqc::dft {

/** Detached authorization for exactly one successful solve on one immutable
 * native KS owner. A later begin invalidates it before scientific work.
 * The historical type name is retained for CUDA source compatibility; backend
 * and device semantics are carried by the complete identity. */
struct CudaKsFinalStateToken {
  std::uint32_t version{1};
  KsFinalStateIdentity identity;
  /** Evidence from the accepted CUDA generation, not the requested policy or
   * configured precision of its integral provider. A mixed proposal is not
   * a reference for strict stationary derivatives. */
  bool strict_physical_fock{};
  /** A force-error allowance may relax convergence even after FP64 handoff.
   * It does not certify the stationary point for second derivatives. */
  bool relaxed_stationarity{};
  bool operator==(const CudaKsFinalStateToken&) const = default;
};

}  // namespace generativeqc::dft
