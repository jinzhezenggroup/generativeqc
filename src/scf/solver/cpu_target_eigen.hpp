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
 * Primary host-scalar HF iteration/finalization opts in, including independent
 * CUDA Fock execution with a Hückel policy. Resident CUDA targets use their GPU
 * owner; physical-reference exports retain their independent provider. */
reference::EigenResult cpu_target_eigen(const reference::Matrix& matrix,
                                        const reference::Matrix* overlap,
                                        const reference::Matrix* orthogonalizer, std::size_t n);
/** One-thread molecular preliminary solve with the available native LAPACK
 * provider and the same physical-frame acceptance gates. Atomic model solves
 * remain on cpu_target_eigen. The caller's preliminary numeric envelope covers
 * returned/working matrices and LAPACKE's row-major transpose plus dsyevd work;
 * no reference-oracle, extra molecular Fock owner or provider retry is used. */
reference::EigenResult cpu_preliminary_eigen(const reference::Matrix& matrix,
                                             const reference::Matrix* overlap,
                                             const reference::Matrix* orthogonalizer,
                                             std::size_t n);

/** Recover a closed-shell preliminary density from row-major n-by-occupied
 * coefficients. The production rank-k adapter owns provider selection and
 * returns both triangles for the caller's independent physical admission. */
reference::Matrix cpu_preliminary_density(const reference::Matrix& coefficients, std::size_t n,
                                          std::size_t occupied);
}  // namespace generativeqc::scf::solver
