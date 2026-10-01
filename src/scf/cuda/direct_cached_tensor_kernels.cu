#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "generated_direct_contraction.cuh"
#include "scf/cuda/direct_cached_tensor_kernels.hpp"
#include "scf/cuda/eri_tensor_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

__global__ void build_eri_kernel(DeviceBatch batch, double* eri) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t eri_size = n * n * n * n;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * eri_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / eri_size);
  std::size_t local = element % eri_size;
  const std::int32_t l = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t k = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t j = static_cast<std::int32_t>(local % n);
  const std::int32_t i = static_cast<std::int32_t>(local / n);
  eri[element] = contracted_eri<double>(batch, system, i, j, k, l, -1);
}

__global__ void build_eri_tile_kernel(DeviceBatch batch, std::int32_t system, std::size_t b0,
                                      std::size_t b1, std::size_t b2, std::size_t b3,
                                      std::size_t c0, std::size_t c1, std::size_t c2,
                                      std::size_t c3, std::size_t elements, double* eri) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       element < elements; element += stride) {
    std::size_t local = element;
    const auto l = static_cast<std::int32_t>(b3 + local % c3);
    local /= c3;
    const auto k = static_cast<std::int32_t>(b2 + local % c2);
    local /= c2;
    const auto j = static_cast<std::int32_t>(b1 + local % c1);
    local /= c1;
    const auto i = static_cast<std::int32_t>(b0 + local);
    eri[element] = contracted_eri<double>(batch, system, i, j, k, l, -1);
  }
}

__global__ void build_fock_kernel(std::int32_t batch_size, std::int32_t nbf, const double* hcore,
                                  const double* eri, const double* density,
                                  const std::uint8_t* active, double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t eri_offset = static_cast<std::size_t>(system) * eri_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const double pkl = density[matrix_offset + matrix_index(k, l, n)];
      coulomb += pkl * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += pkl * eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[element] + coulomb - 0.5 * exchange;
}

__global__ void build_uhf_fock_kernel(std::int32_t batch_size, std::int32_t nbf,
                                      const double* hcore, const double* eri, const double* density,
                                      const std::uint8_t* active, double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * 2 * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / 2;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t spin = state % 2;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t physical_matrix_offset = system * matrix_size;
  const std::size_t eri_offset = system * eri_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t spin_offset = alpha_offset + spin * matrix_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const std::size_t kl = matrix_index(k, l, n);
      const double total = density[alpha_offset + kl] + density[beta_offset + kl];
      coulomb += total * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += density[spin_offset + kl] * eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[physical_matrix_offset + local] + coulomb - exchange;
}

void launch_build_eri_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                             DeviceBatch batch, double* eri) {
  build_eri_kernel<<<grid, block, shared_bytes, stream>>>(batch, eri);
}

void launch_build_eri_tile_kernel(cudaStream_t stream, DeviceBatch batch, std::int32_t system,
                                  const std::array<std::size_t, 4>& begin,
                                  const std::array<std::size_t, 4>& count, std::size_t elements,
                                  double* eri) {
  if (!elements) return;
  constexpr unsigned threads = 128;
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>((elements + threads - 1) / threads, 65535));
  build_eri_tile_kernel<<<blocks, threads, 0, stream>>>(batch, system, begin[0], begin[1], begin[2],
                                                        begin[3], count[0], count[1], count[2],
                                                        count[3], elements, eri);
}

void launch_build_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                              std::int32_t batch_size, std::int32_t nbf, const double* hcore,
                              const double* eri, const double* density, const std::uint8_t* active,
                              double* fock) {
  build_fock_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, nbf, hcore, eri, density,
                                                           active, fock);
}

void launch_build_uhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* hcore, const double* eri, const double* density,
                                  const std::uint8_t* active, double* fock) {
  build_uhf_fock_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, nbf, hcore, eri, density,
                                                               active, fock);
}

}  // namespace generativeqc::scf::cuda_execution
