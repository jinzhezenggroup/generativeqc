#include <math_constants.h>

#include <cmath>

#include "scf/cuda/basis_transform_kernels.hpp"
#include "scf/cuda/matrix_index.cuh"

namespace generativeqc::scf::cuda_execution {

__global__ void initialize_direct_fock_kernel(std::int32_t batch_size,
                                              std::int32_t matrices_per_system, std::int32_t nbf,
                                              const double* hcore, const std::uint8_t* active,
                                              double* fock) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t matrix_count = static_cast<std::size_t>(batch_size) * matrices_per_system;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(matrices_per_system);
  if (active != nullptr && active[system] == 0) return;
  fock[element] = hcore[system * matrix_size + element % matrix_size];
}

/** First stage of D_cart = C^T D_public C. */
__global__ void transform_density_to_direct_right_kernel(
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf,
    const double* transform, const double* density, const std::uint8_t* active, double* temporary,
    const std::int32_t* shell_spans) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count = static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * rectangular_size) return;
  const std::size_t state = element / rectangular_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % rectangular_size;
  const std::size_t row = local % n;
  const std::size_t direct_column = local / n;
  const std::size_t density_offset = state * n * n;
  const std::size_t transform_offset = system * rectangular_size;
  double value = 0.0;
  const auto span =
      2U * (static_cast<std::size_t>(batch_size) * n + system * direct_n + direct_column);
  const std::size_t begin = shell_spans ? shell_spans[span] : 0U;
  const std::size_t end = shell_spans ? shell_spans[span + 1U] : n;
  for (std::size_t column = begin; column < end; ++column) {
    value += density[density_offset + matrix_index(row, column, n)] *
             transform[transform_offset + column + direct_column * n];
  }
  temporary[element] = value;
}

/** Second stage of D_cart = C^T (D_public C). */
__global__ void transform_density_to_direct_left_kernel(
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf,
    const double* transform, const double* temporary, const std::uint8_t* active,
    double* direct_density, const std::int32_t* shell_spans) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t matrix_size = direct_n * direct_n;
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count = static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t direct_row = local % direct_n;
  const std::size_t direct_column = local / direct_n;
  const std::size_t transform_offset = system * rectangular_size;
  const std::size_t temporary_offset = state * rectangular_size;
  double value = 0.0;
  const auto span =
      2U * (static_cast<std::size_t>(batch_size) * n + system * direct_n + direct_row);
  const std::size_t begin = shell_spans ? shell_spans[span] : 0U;
  const std::size_t end = shell_spans ? shell_spans[span + 1U] : n;
  for (std::size_t row = begin; row < end; ++row) {
    value += transform[transform_offset + row + direct_row * n] *
             temporary[temporary_offset + row + direct_column * n];
  }
  direct_density[element] = value;
}

/** First stage of F_public = C F_cart C^T. */
__global__ void transform_direct_fock_left_kernel(std::int32_t batch_size, std::int32_t spin_count,
                                                  std::int32_t nbf, std::int32_t direct_nbf,
                                                  const double* transform,
                                                  const double* direct_fock,
                                                  const std::uint8_t* active, double* temporary,
                                                  const std::int32_t* shell_spans) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t direct_matrix_size = direct_n * direct_n;
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count = static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * rectangular_size) return;
  const std::size_t state = element / rectangular_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % rectangular_size;
  const std::size_t public_row = local % n;
  const std::size_t direct_column = local / n;
  const std::size_t transform_offset = system * rectangular_size;
  const std::size_t direct_offset = state * direct_matrix_size;
  double value = 0.0;
  const auto span = 2U * (system * n + public_row);
  const std::size_t begin = shell_spans ? shell_spans[span] : 0U;
  const std::size_t end = shell_spans ? shell_spans[span + 1U] : direct_n;
  for (std::size_t direct_row = begin; direct_row < end; ++direct_row) {
    value += transform[transform_offset + public_row + direct_row * n] *
             direct_fock[direct_offset + matrix_index(direct_row, direct_column, direct_n)];
  }
  temporary[element] = value;
}

/** Finish F_public = (C F_cart) C^T and restore the one-electron matrix. */
__global__ void transform_direct_fock_right_kernel(std::int32_t batch_size, std::int32_t spin_count,
                                                   std::int32_t nbf, std::int32_t direct_nbf,
                                                   const double* transform, const double* temporary,
                                                   const double* hcore, const std::uint8_t* active,
                                                   double* fock, const std::int32_t* shell_spans) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count = static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t public_row = local % n;
  const std::size_t public_column = local / n;
  const std::size_t transform_offset = system * rectangular_size;
  const std::size_t temporary_offset = state * rectangular_size;
  double value = hcore[system * matrix_size + local];
  const auto span = 2U * (system * n + public_column);
  const std::size_t begin = shell_spans ? shell_spans[span] : 0U;
  const std::size_t end = shell_spans ? shell_spans[span + 1U] : direct_n;
  for (std::size_t direct_column = begin; direct_column < end; ++direct_column) {
    value += temporary[temporary_offset + public_row + direct_column * n] *
             transform[transform_offset + public_column + direct_column * n];
  }
  fock[element] = value;
}

void launch_initialize_direct_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t matrices_per_system, std::int32_t nbf,
                                          const double* hcore, const std::uint8_t* active,
                                          double* fock) {
  initialize_direct_fock_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, matrices_per_system, nbf, hcore, active, fock);
}

void launch_transform_density_to_direct_right_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf, const double* transform,
    const double* density, const std::uint8_t* active, double* temporary,
    const std::int32_t* shell_spans) {
  transform_density_to_direct_right_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, spin_count, nbf, direct_nbf, transform, density, active, temporary, shell_spans);
}

void launch_transform_density_to_direct_left_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf, const double* transform,
    const double* temporary, const std::uint8_t* active, double* direct_density,
    const std::int32_t* shell_spans) {
  transform_density_to_direct_left_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, spin_count, nbf, direct_nbf, transform, temporary, active, direct_density,
      shell_spans);
}

void launch_transform_direct_fock_left_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                              cudaStream_t stream, std::int32_t batch_size,
                                              std::int32_t spin_count, std::int32_t nbf,
                                              std::int32_t direct_nbf, const double* transform,
                                              const double* direct_fock, const std::uint8_t* active,
                                              double* temporary, const std::int32_t* shell_spans) {
  transform_direct_fock_left_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, spin_count, nbf, direct_nbf, transform, direct_fock, active, temporary,
      shell_spans);
}

void launch_transform_direct_fock_right_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream, std::int32_t batch_size,
                                               std::int32_t spin_count, std::int32_t nbf,
                                               std::int32_t direct_nbf, const double* transform,
                                               const double* temporary, const double* hcore,
                                               const std::uint8_t* active, double* fock,
                                               const std::int32_t* shell_spans) {
  transform_direct_fock_right_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, spin_count, nbf, direct_nbf, transform, temporary, hcore, active, fock,
      shell_spans);
}

}  // namespace generativeqc::scf::cuda_execution
