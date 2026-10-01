#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_eri_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                             DeviceBatch batch, double* eri);

/** Enqueue one unscreened public-AO ERI tile in row-major [i,j,k,l] order.
 * Uses the same contracted_eri<double> evaluator as Direct J/K; no molecular
 * N^4 tensor, screening decision, allocation, or host transfer is introduced.
 */
void launch_build_eri_tile_kernel(cudaStream_t stream, DeviceBatch batch, std::int32_t system,
                                  const std::array<std::size_t, 4>& begin,
                                  const std::array<std::size_t, 4>& count, std::size_t elements,
                                  double* eri);

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                              std::int32_t batch_size, std::int32_t nbf, const double* hcore,
                              const double* eri, const double* density, const std::uint8_t* active,
                              double* fock);

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_uhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* hcore, const double* eri, const double* density,
                                  const std::uint8_t* active, double* fock);

}  // namespace generativeqc::scf::cuda_execution
