#pragma once

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_ks_resident_fock.hpp"

namespace generativeqc::dft {
/** Shared physical-model admission for induced and nuclear WB97M-V sources.
 * Coefficients remain compiler-owned. A valid grid alone does not authenticate
 * the exact/fitted primary operator and mandatory exact LR correction. */
bool cuda_ks_wb97mv_model_matches(const CudaKsFinalStateToken& expected,
                                  const CudaKsResidentFockBinding& source);
}  // namespace generativeqc::dft
