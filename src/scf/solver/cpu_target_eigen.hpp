#pragma once

#include "scf/initial_guess/eigen_operation.hpp"

namespace generativeqc::scf::solver {
/** Numeric allocation bound including returned frame, excluding borrowed F/S/X.
 * It does not exceed the prior generalized-reference envelope. No allocator,
 * runtime, trace or exception bookkeeping is represented as numerical storage. */
std::size_t cpu_target_eigen_workspace_bytes(std::size_t n);

/** Fixed one-thread shared-scalar target solve. The 1e-14 off-diagonal cap is a
 * stopping rule, not an error certificate; original F/S residual and metric
 * checks independently gate every returned frame. No retry or provider switch.
 * Only primary CPU HF iteration/finalization opts in; Setup and exports do not. */
reference::EigenResult cpu_target_eigen(const reference::Matrix& matrix,
                                        const reference::Matrix* overlap,
                                        const reference::Matrix* orthogonalizer, std::size_t n);
}  // namespace generativeqc::scf::solver
