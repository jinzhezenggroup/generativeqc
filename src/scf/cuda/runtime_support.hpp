#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf::cuda_execution {

/** Preserve the public CUDA/BLAS/solver status mapping at host launch boundaries. */
generativeqc_status cuda_status(cudaError_t status);

generativeqc_status solver_status(cusolverStatus_t status);

generativeqc_status blas_status(cublasStatus_t status);

/** Queue a nonempty host upload on the caller's stream; empty input is a no-op. */
generativeqc_status copy_to_device(void* destination, const void* source, std::size_t bytes,
                                   cudaStream_t stream);

}  // namespace generativeqc::scf::cuda_execution
