#pragma once

#include <array>
#include <span>
#include <string>

#include "scf/cuda/final_validation_kernels.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "scf/solver/final_state.hpp"

namespace generativeqc::scf::cuda_execution {
/** Borrowed, serialized scratch; all four matrices hold n*n doubles and must
 * not alias inputs or one another. Partial storage contains the three bounded
 * reduction stages and one result packet. The caller charges and owns it. */
struct ResidentFinalValidationWorkspace {
  std::array<double*, 4> matrices{};
  cuda_df::ValidationPartial* partial{};
  std::size_t partial_count{};
};

inline std::size_t resident_final_validation_partial_count(std::size_t dimension) {
  return 3 * cuda_df::validation_block_count(dimension) + 1;
}

/** Compute evidence for the shared final-state gates, never accept a state.
 * Inputs belong to one authenticated physical frame. C is column-major;
 * transposed_operators explicitly marks row-major F/S/H/D buffer interpretation.
 * The owner proves identity/lifetime; this adapter checks solver status and
 * numerical products. Requires an existing cuBLAS handle, allocates no device
 * storage, rejects capture, and drains all submitted work even on exceptions.
 * Scratch is overwritten; consumers must stage their own retained data later. */
bool resident_final_state_products(MatrixLibraryResources resources,
                                   std::span<const cuda_df::ValidationInputs> spins,
                                   ResidentFinalValidationWorkspace workspace, bool canonical,
                                   double nuclear, solver::FinalStateDiagnostic& diagnostic,
                                   std::string& detail);
}  // namespace generativeqc::scf::cuda_execution
