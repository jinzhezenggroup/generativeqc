#pragma once

// Scalar integral support is shared by native-host and CUDA lowering. Keep
// CUDA's original qualifiers, without requiring CUDA headers on CPU builds.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define GENERATIVEQC_ERI_DEVICE __device__
#define GENERATIVEQC_ERI_SHARED_HD __host__ __device__
#define GENERATIVEQC_ERI_INLINE __forceinline__
#else
#define GENERATIVEQC_ERI_DEVICE
#define GENERATIVEQC_ERI_SHARED_HD
#define GENERATIVEQC_ERI_INLINE inline
#endif
