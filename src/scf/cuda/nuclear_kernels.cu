#include "generated_nuclear_repulsion.hpp"
#include "scf/cuda/nuclear_kernels.hpp"
#include "scf/cuda/one_electron_export_kernels.hpp"

namespace generativeqc::scf::cuda_execution {

template <bool Derivative>
__global__ void build_cuda_nuclear_repulsion_kernel(DeviceBatch batch,
                                                    std::int64_t derivative_coordinate,
                                                    double* nuclear_repulsion) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch.batch_size) return;
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0 ? derivative_coordinate
                                : derivative_coordinate + batch.atom_offsets[system] * 3;
  if constexpr (Derivative) {
    if (system_derivative_coordinate < 0) {
      nuclear_repulsion[system] = 0.0;
      return;
    }
    double result = 0.0;
    const auto atom = system_derivative_coordinate / 3;
    if (atom < batch.atom_offsets[system] || atom >= batch.atom_offsets[system + 1]) {
      nuclear_repulsion[system] = 0.0;
      return;
    }
    // Only pairs containing the differentiated atom contribute. Visit partners
    // in ascending order to retain the original triangular sum and orientation
    // without evaluating every unrelated pair for each of the 3*N coordinates.
    for (std::int64_t partner = batch.atom_offsets[system];
         partner < batch.atom_offsets[system + 1]; ++partner) {
      if (partner == atom) continue;
      const auto first = partner < atom ? atom : partner;
      const auto second = partner < atom ? partner : atom;
      const double* a = batch.positions + 3 * first;
      const double* b = batch.positions + 3 * second;
      double gradient[3];
      ::generativeqc::generated::nuclear_repulsion::gradient(
          static_cast<double>(batch.atomic_numbers[first]) * batch.atomic_numbers[second],
          a[0] - b[0], a[1] - b[1], a[2] - b[2], gradient);
      if (atom == first)
        result += gradient[system_derivative_coordinate % 3];
      else
        result -= gradient[system_derivative_coordinate % 3];
    }
    nuclear_repulsion[system] = result;
  } else {
    double result = 0.0;
    for (std::int64_t first = batch.atom_offsets[system]; first < batch.atom_offsets[system + 1];
         ++first) {
      const double* a = batch.positions + 3 * first;
      for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
        const double* b = batch.positions + 3 * second;
        result += ::generativeqc::generated::nuclear_repulsion::energy(
            static_cast<double>(batch.atomic_numbers[first]) * batch.atomic_numbers[second],
            a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      }
    }
    nuclear_repulsion[system] = result;
  }
}

__global__ void build_nuclear_repulsion_kernel(DeviceBatch batch, double* nuclear_repulsion) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch.batch_size) return;
  double result = 0.0;
  for (std::int64_t first = batch.atom_offsets[system]; first < batch.atom_offsets[system + 1];
       ++first) {
    const double* a = batch.positions + 3 * first;
    for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
      const double* b = batch.positions + 3 * second;
      result += ::generativeqc::generated::nuclear_repulsion::energy(
          static_cast<double>(batch.atomic_numbers[first]) * batch.atomic_numbers[second],
          a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    }
  }
  nuclear_repulsion[system] = result;
}

__global__ void nuclear_force_kernel(DeviceBatch batch, const std::uint8_t* active,
                                     double* forces) {
  const std::int64_t coordinate = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (coordinate >= batch.total_atoms * 3) return;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  double derivative = 0.0;
  for (std::int64_t first = batch.atom_offsets[system]; first < batch.atom_offsets[system + 1];
       ++first) {
    const double* a = batch.positions + 3 * first;
    for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
      const double* b = batch.positions + 3 * second;
      double gradient[3];
      ::generativeqc::generated::nuclear_repulsion::gradient(
          static_cast<double>(batch.atomic_numbers[first]) * batch.atomic_numbers[second],
          a[0] - b[0], a[1] - b[1], a[2] - b[2], gradient);
      if (atom == first) derivative += gradient[coordinate % 3];
      if (atom == second) derivative -= gradient[coordinate % 3];
    }
  }
  forces[coordinate] = -derivative;
}

void launch_build_nuclear_repulsion_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                           cudaStream_t stream, DeviceBatch batch,
                                           double* nuclear_repulsion) {
  build_nuclear_repulsion_kernel<<<grid, block, shared_bytes, stream>>>(batch, nuclear_repulsion);
}

void launch_nuclear_force_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                 cudaStream_t stream, DeviceBatch batch, const std::uint8_t* active,
                                 double* forces) {
  nuclear_force_kernel<<<grid, block, shared_bytes, stream>>>(batch, active, forces);
}

void launch_build_cuda_nuclear_repulsion_kernel(bool derivative, dim3 grid, dim3 block,
                                                std::size_t shared_bytes, cudaStream_t stream,
                                                DeviceBatch batch,
                                                std::int64_t derivative_coordinate,
                                                double* nuclear_repulsion) {
  if (derivative) {
    build_cuda_nuclear_repulsion_kernel<true>
        <<<grid, block, shared_bytes, stream>>>(batch, derivative_coordinate, nuclear_repulsion);
  } else {
    build_cuda_nuclear_repulsion_kernel<false>
        <<<grid, block, shared_bytes, stream>>>(batch, derivative_coordinate, nuclear_repulsion);
  }
}

}  // namespace generativeqc::scf::cuda_execution
