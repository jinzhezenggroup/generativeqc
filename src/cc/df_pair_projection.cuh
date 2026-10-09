#pragma once

#include <cuda_runtime.h>

#include "cc/df_pair_bound.hpp"
#include "generated_df_ccsd_spectator_pairs_cpu.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc::pair_bound {

#if !defined(GENERATIVEQC_CUDA_PROVIDER_CUMETAL) || !GENERATIVEQC_CUDA_PROVIDER_CUMETAL

// Positive subtraction is rounded toward +infinity, including subnormals.
// Nonfinite metadata refuses this optional path; it never touches the physical
// sticky status maintained by preparation, contractions, and the original core.
static __device__ std::uint64_t outward_distance(double first, double second, unsigned& refused) {
  const auto distance = first >= second ? __dsub_ru(first, second) : __dsub_ru(second, first);
  const auto bits = static_cast<std::uint64_t>(__double_as_longlong(distance)) & magnitude_mask;
  if (bits >= infinity_bits) refused = 1;
  return bits;
}

static __global__ void project_tau_kernel(const double* tau, const double* singles,
                                          std::size_t occupied, std::size_t virtuals,
                                          std::size_t doubles_count, std::size_t singles_count,
                                          double* paired, ProjectionMaxima* maxima) {
  __shared__ ProjectionMaxima partial[256];
  ProjectionMaxima local{};
  const auto first = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = std::size_t(blockDim.x) * gridDim.x;
  const generated::RestrictedPairCoordinates coordinates{occupied, virtuals};
  for (auto index = first; index < singles_count; index += stride) {
    const auto bits =
        static_cast<std::uint64_t>(__double_as_longlong(singles[index])) & magnitude_mask;
    local.t1_magnitude_bits = max(local.t1_magnitude_bits, bits);
    if (bits >= infinity_bits) local.refused = 1;
  }
  for (auto index = first; index < doubles_count; index += stride) {
    const auto occupied_flat = index / virtuals / virtuals;
    if (occupied_flat / occupied > occupied_flat % occupied) continue;
    const auto mate = coordinates(index).partner;
    // Diagonal occupied blocks retain both virtual orders. Canonical mean
    // operand order makes their stored reflections exactly identical, even
    // when the ordered compiler expression is not commutative in FP64.
    const auto lower = min(index, mate), upper = max(index, mate);
    const double left = tau[lower], right = tau[upper];
    const auto mean = coordinates.project(left, right);
    paired[generated::dfpairs::occupied_ladder_offset(index, occupied, virtuals)] = mean;
    const auto left_bits = static_cast<std::uint64_t>(__double_as_longlong(left)) & magnitude_mask;
    const auto right_bits =
        static_cast<std::uint64_t>(__double_as_longlong(right)) & magnitude_mask;
    const auto mean_bits = static_cast<std::uint64_t>(__double_as_longlong(mean)) & magnitude_mask;
    if (left_bits >= infinity_bits || right_bits >= infinity_bits || mean_bits >= infinity_bits) {
      local.refused = 1;
      continue;
    }
    local.tau_error_bits = max(local.tau_error_bits, outward_distance(left, mean, local.refused));
    local.tau_error_bits = max(local.tau_error_bits, outward_distance(right, mean, local.refused));
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (unsigned offset = 128; offset; offset /= 2) {
    if (threadIdx.x < offset) {
      auto& target = partial[threadIdx.x];
      const auto source = partial[threadIdx.x + offset];
      target.tau_error_bits = max(target.tau_error_bits, source.tau_error_bits);
      target.t1_magnitude_bits = max(target.t1_magnitude_bits, source.t1_magnitude_bits);
      target.refused |= source.refused;
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    atomicMax(reinterpret_cast<unsigned long long*>(&maxima->tau_error_bits),
              static_cast<unsigned long long>(partial[0].tau_error_bits));
    atomicMax(reinterpret_cast<unsigned long long*>(&maxima->t1_magnitude_bits),
              static_cast<unsigned long long>(partial[0].t1_magnitude_bits));
    atomicOr(&maxima->refused, partial[0].refused);
  }
}

#endif

// Dimensions and storage are checked by the owner before launch. The metadata
// reset is per amplitude state; no geometry-only cache may bypass this audit.
inline void project_tau(const double* tau, const double* singles, std::size_t occupied,
                        std::size_t virtuals, std::size_t doubles_count, std::size_t singles_count,
                        double* paired, ProjectionMaxima* maxima, cudaStream_t stream) {
#if !defined(GENERATIVEQC_CUDA_PROVIDER_CUMETAL) || !GENERATIVEQC_CUDA_PROVIDER_CUMETAL
  generativeqc_tensor::cuda_check(cudaMemsetAsync(maxima, 0, sizeof(*maxima), stream));
  project_tau_kernel<<<generativeqc_tensor::blocks(
                           static_cast<generativeqc_tensor::I>(doubles_count), 256),
                       256, 0, stream>>>(tau, singles, occupied, virtuals, doubles_count,
                                         singles_count, paired, maxima);
  generativeqc_tensor::cuda_check(cudaGetLastError());
#else
  // CuMetal owners cannot admit this path until directed-DP metadata has its
  // own qualification. Keep the stub free of unavailable CUDA intrinsics.
  (void)tau;
  (void)singles;
  (void)occupied;
  (void)virtuals;
  (void)doubles_count;
  (void)singles_count;
  (void)paired;
  (void)maxima;
  (void)stream;
#endif
}

}  // namespace generativeqc::cc::pair_bound
