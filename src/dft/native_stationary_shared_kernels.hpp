#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace generativeqc::dft {

// Shared CUDA source transport/reduction for native stationary derivatives.
// The enclosing prepared owner validates graph identity, source completeness,
// device pointers, ordered source slots and stream lifetime before calling.
cudaError_t launch_native_stationary_source_transport(
    const double* input, std::size_t coordinates, std::size_t channels, double sign,
    double* first, double* second, int* error, cudaStream_t stream) noexcept;

// The compiler admits this backend only when its complete ordered MethodIR
// reduction has unit coefficients. All graphs reuse one device implementation.
cudaError_t launch_native_stationary_source_reduction(
    const double* input, std::size_t atoms, std::size_t sources, double* output,
    int* error, cudaStream_t stream) noexcept;

}  // namespace generativeqc::dft
