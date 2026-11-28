#include <cuda_runtime.h>

#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_response_pair_kernels.hpp"

namespace generativeqc::scf::cuda_execution {
namespace {

constexpr unsigned threads = 128, warp_width = 32;

__device__ std::size_t lower_pair(std::size_t first, std::size_t second) {
  if (first < second) {
    const auto saved = first;
    first = second;
    second = saved;
  }
  return first * (first + 1) / 2 + second;
}

struct ExchangeOrientations {
  double same, crossed;
};

/** ERI permutations identify both orientations of the density contraction. */
__device__ ExchangeOrientations exchange_orientations(const double* coulomb, std::size_t stride,
                                                      std::size_t first, std::size_t second,
                                                      std::size_t bra, std::size_t ket) {
  return {coulomb[lower_pair(first, bra) * stride + lower_pair(second, ket)],
          coulomb[lower_pair(first, ket) * stride + lower_pair(second, bra)]};
}

__global__ void store_symmetric_exchange(const double* coulomb, double* exchange,
                                         std::size_t stride, std::size_t pairs) {
  const auto input = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto output = std::size_t(blockIdx.y);
  if (input >= pairs) return;
  std::size_t first{}, second{}, bra{}, ket{};
  decode_lower_triangle(output, first, second);
  decode_lower_triangle(input, bra, ket);
  const auto value = exchange_orientations(coulomb, stride, first, second, bra, ket);
  exchange[output * stride + input] =
      value.same == value.crossed ? value.same : 0.5 * value.same + 0.5 * value.crossed;
}

__global__ void pack_hybrid_pair_density(const double* density, std::size_t nao, std::size_t pairs,
                                         double* packed) {
  const auto pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  std::size_t first{}, second{};
  decode_lower_triangle(pair, first, second);
  const auto lower = density[first * nao + second];
  const auto upper = first == second ? 0.0 : density[second * nao + first];
  packed[pair] = lower + upper;
  packed[pairs + pair] = first == second ? 0.0 : lower - upper;
}

/** Each warp owns both output orientations. Symmetric densities only read
 * coalesced rows; a nonzero odd coefficient gathers its two ERIs on demand. */
__global__ void apply_hybrid_pair_rows(const double* coulomb_rows, const double* exchange_rows,
                                       std::size_t stride, std::size_t nao, std::size_t pairs,
                                       const double* packed, double* coulomb, double* exchange) {
  const auto lane = threadIdx.x % warp_width;
  const auto output = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / warp_width;
  if (output >= pairs) return;
  std::size_t first{}, second{};
  decode_lower_triangle(output, first, second);
  const auto* coulomb_row = coulomb_rows + output * stride;
  const auto* exchange_row = exchange_rows + output * stride;
  double charge{}, even{}, odd{};
  for (std::size_t input = lane; input < pairs; input += warp_width) {
    const auto symmetric = packed[input];
    charge += coulomb_row[input] * symmetric;
    even += exchange_row[input] * symmetric;
    const auto antisymmetric = packed[pairs + input];
    if (first != second && antisymmetric != 0) {
      std::size_t bra{}, ket{};
      decode_lower_triangle(input, bra, ket);
      const auto value = exchange_orientations(coulomb_rows, stride, first, second, bra, ket);
      odd += (0.5 * value.same - 0.5 * value.crossed) * antisymmetric;
    }
  }
  for (unsigned offset = warp_width / 2; offset; offset /= 2) {
    charge += __shfl_down_sync(0xffffffffU, charge, offset);
    even += __shfl_down_sync(0xffffffffU, even, offset);
    odd += __shfl_down_sync(0xffffffffU, odd, offset);
  }
  if (lane == 0) {
    coulomb[first * nao + second] = charge;
    exchange[first * nao + second] = even + odd;
    if (first != second) {
      coulomb[second * nao + first] = charge;
      exchange[second * nao + first] = even - odd;
    }
  }
}

}  // namespace

cudaError_t prepare_exchange_pair_rows(cudaStream_t stream, const double* coulomb_rows,
                                       double* exchange_rows, std::size_t stride,
                                       std::size_t nao) noexcept {
  const auto pairs = nao * (nao + 1) / 2;
  const dim3 grid(static_cast<unsigned>((pairs + threads - 1) / threads),
                  static_cast<unsigned>(pairs));
  store_symmetric_exchange<<<grid, threads, 0, stream>>>(coulomb_rows, exchange_rows, stride,
                                                         pairs);
  return cudaGetLastError();
}

cudaError_t enqueue_hybrid_pair_rows(cudaStream_t stream, const double* coulomb_rows,
                                     const double* exchange_rows, std::size_t stride,
                                     std::size_t nao, const double* density, double* scratch,
                                     double* coulomb, double* exchange) noexcept {
  const auto pairs = nao * (nao + 1) / 2;
  constexpr unsigned warps = threads / warp_width;
  pack_hybrid_pair_density<<<(pairs + threads - 1) / threads, threads, 0, stream>>>(density, nao,
                                                                                    pairs, scratch);
  const auto packed = cudaGetLastError();
  if (packed != cudaSuccess) return packed;
  apply_hybrid_pair_rows<<<(pairs + warps - 1) / warps, threads, 0, stream>>>(
      coulomb_rows, exchange_rows, stride, nao, pairs, scratch, coulomb, exchange);
  return cudaGetLastError();
}

}  // namespace generativeqc::scf::cuda_execution
