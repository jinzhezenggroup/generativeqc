#pragma once

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_ks_resident_fock.hpp"
#include "dft/cuda_wb97mv_grid.hpp"

namespace generativeqc::dft {
/** Shared physical-model admission for induced and nuclear WB97M-V sources.
 * Coefficients remain compiler-owned. A valid grid alone does not authenticate
 * the exact/fitted primary operator and the accepted exact or fitted erf correction. */
bool cuda_ks_wb97mv_model_matches(const CudaKsFinalStateToken& expected,
                                  const CudaKsResidentFockBinding& source);

/** Authenticate the native second-order model and, if requested, populate only
 * the scientific grid settings. Caller-owned resource limits are unchanged. */
bool cuda_ks_hessian_model_matches(const CudaKsFinalStateToken& expected,
                                   const CudaKsResidentFockBinding& source,
                                   CudaWb97mvGridOptions* grid = nullptr);
}  // namespace generativeqc::dft
