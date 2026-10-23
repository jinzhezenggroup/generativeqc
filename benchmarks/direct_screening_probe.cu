/** Standalone scheduling experiment: FP64, FP32, INT8 and INT8 Tensor Cores.
 * Build with nvcc -std=c++20 -O3 -arch=sm_120 -Isrc and run through srun.
 * This synthetic sorted-bound workload is NOT molecular endpoint evidence.
 * INT8 uses tile-local power-of-two upper quantization. WMMA forms an outer
 * product with only one nonzero K lane, exposing its utilization/packing cost.
 * Prepared FP32 intervals resolve either outcome and refine only uncertain
 * candidates. Other coarse modes refine every survivor with the FP64 predicate.
 */
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "scf/cuda/direct_coarse_screening.cuh"

namespace {
using namespace generativeqc::scf::cuda_execution;
namespace wmma = nvcuda::wmma;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** Each 16-bound tile owns an exact power-of-two scale and 0..64 codes.
 * Nonfinite or unsupported exponent tiles fail open; they cannot reject.
 * Padding K to 16 is included in both allocation and measured preparation.
 */
__global__ void pack(const double* bounds, int n, signed char* codes, int* exponents) {
  const int base = blockIdx.x * 16;
  const int lane = threadIdx.x;
  float upper =
      lane < 16 && base + lane < n ? direct_screening_upper_float(bounds[base + lane]) : 0;
  float maximum = upper;
  for (int offset = 16; offset; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
  const int exponent = int((__float_as_uint(maximum) >> 23U) & 255U) - 127 - 5;
  const bool supported = exponent >= -120 && exponent <= 110 && isfinite(maximum);
  if (lane == 0) exponents[blockIdx.x] = supported ? exponent : 128;
  for (int slot = lane; slot < 256; slot += 32) {
    const int index = base * 16 + slot;
    if (index < n * 16) codes[index] = 0;
  }
  __syncthreads();
  if (lane < 16 && base + lane < n)
    codes[(base + lane) * 16] =
        supported
            ? static_cast<signed char>(__float2int_ru(__fdiv_ru(upper, scalbnf(1.0F, exponent))))
            : 1;
}

__device__ bool exact_survives(double a, double b, double density, double tolerance, bool force) {
  const double q = a * b;
  if (q < tolerance || q * density < tolerance) return false;
  return !force || !(q * density * density < fmin(tolerance, 1e-14));
}

__device__ bool integer_rejects(int product, int ea, int eb, double density, double tolerance,
                                bool force) {
  if (ea == 128 || eb == 128) return false;
  // The first rescaling is exact in the admitted exponent range. Directed
  // multiplication protects the second scaling and all density products.
  const float q = direct_screening_upper_product(scalbnf(float(product), ea), scalbnf(1.0F, eb));
  const float threshold = __double2float_rd(tolerance);
  if (q < threshold) return true;
  const float d = direct_screening_upper_float(density);
  const float f = direct_screening_upper_product(q, d);
  if (f < threshold) return true;
  return force && direct_screening_upper_product(f, d) < __double2float_rd(fmin(tolerance, 1e-14));
}

/** Convert once per bound, amortized over the matrix of block candidates. */
__global__ void pack_float(const double* bounds, int n, float* upper) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) upper[i] = direct_screening_upper_float(bounds[i]);
}

template <bool Interval>
__global__ void prepared_float(const double* bounds, const float* upper, int n, double density,
                               double tolerance, bool force, unsigned char* output) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * n) return;
  const int i = slot / n, j = slot % n;
  if constexpr (Interval) {
    const float d = direct_screening_upper_float(density);
    const auto decision =
        force ? direct_fp32_block_decision<true>(upper[i], upper[j], d, tolerance,
                                                 fmin(tolerance, 1e-14))
              : direct_fp32_block_decision<false>(upper[i], upper[j], d, tolerance, 0);
    // Bit zero is admission; the other values distinguish proven outcomes
    // from exact refinement without adding a second counter kernel.
    output[slot] = decision == DirectBlockDecision::Accept ? 5
                   : decision == DirectBlockDecision::Reject
                       ? 2
                       : unsigned(exact_survives(bounds[i], bounds[j], density, tolerance, force));
    return;
  }
  const float q = direct_screening_upper_product(upper[i], upper[j]);
  const float d = direct_screening_upper_float(density);
  const float f = direct_screening_upper_product(q, d);
  const float t = __double2float_rd(tolerance);
  const bool rejected =
      q < t || f < t ||
      (force && direct_screening_upper_product(f, d) < __double2float_rd(fmin(tolerance, 1e-14)));
  output[slot] =
      rejected ? 2 : unsigned(exact_survives(bounds[i], bounds[j], density, tolerance, force));
}

template <int Mode>
__global__ void scalar(const double* bounds, int n, const signed char* codes, const int* exponents,
                       double density, double tolerance, bool force, unsigned char* output) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= n * n) return;
  const int i = slot / n, j = slot % n;
  bool rejected = false;
  if constexpr (Mode == 1)
    rejected = force
                   ? direct_fp32_block_rejects<true>(bounds[i], bounds[j], density, tolerance,
                                                     fmin(tolerance, 1e-14))
                   : direct_fp32_block_rejects<false>(bounds[i], bounds[j], density, tolerance, 0);
  if constexpr (Mode == 2)
    rejected = integer_rejects(int(codes[i * 16]) * int(codes[j * 16]), exponents[i / 16],
                               exponents[j / 16], density, tolerance, force);
  output[slot] =
      rejected ? 2 : unsigned(exact_survives(bounds[i], bounds[j], density, tolerance, force));
}

/** One warp computes 256 independent pair bounds using exact INT32 MMA.
 * The remaining 15 reduction lanes are zero: scalar screening has no natural
 * long dot product. Retain this deliberately honest packing in the timing.
 */
template <bool Tensor>
__global__ void tiled(const double* bounds, int n, const signed char* codes, const int* exponents,
                      double density, double tolerance, bool force, unsigned char* output) {
  __shared__ __align__(32) int products[256];
  const int row = blockIdx.y * 16, col = blockIdx.x * 16;
  if constexpr (Tensor) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
    wmma::load_matrix_sync(a, codes + row * 16, 16);
    wmma::load_matrix_sync(b, codes + col * 16, 16);
    wmma::fill_fragment(c, 0);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(products, c, 16, wmma::mem_row_major);
    __syncwarp();
  }
  for (int index = threadIdx.x; index < 256; index += 32) {
    const int i = row + index / 16, j = col + index % 16;
    // Same warp/tile/refinement mapping isolates MMA from scheduling benefits.
    const int product = Tensor ? products[index] : int(codes[i * 16]) * int(codes[j * 16]);
    const bool rejected = integer_rejects(product, exponents[blockIdx.y], exponents[blockIdx.x],
                                          density, tolerance, force);
    output[i * n + j] =
        rejected ? 2 : unsigned(exact_survives(bounds[i], bounds[j], density, tolerance, force));
  }
}

/** Measure the complete packing + mask + exact-refinement GPU work. */
template <class Launch>
float timed(Launch launch) {
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));
  for (int i = 0; i < 3; ++i) launch();
  check(cudaEventRecord(begin));
  for (int i = 0; i < 30; ++i) launch();
  check(cudaEventRecord(end));
  check(cudaEventSynchronize(end));
  check(cudaGetLastError());
  float milliseconds = 0;
  check(cudaEventElapsedTime(&milliseconds, begin, end));
  check(cudaEventDestroy(begin));
  check(cudaEventDestroy(end));
  return milliseconds / 30;
}
}  // namespace

int main(int argc, char** argv) try {
  if (!std::getenv("SLURM_JOB_ID") || !std::getenv("CUDA_VISIBLE_DEVICES"))
    throw std::runtime_error("Run GPU experiments through a finite srun allocation");
  const int n = argc > 1 ? std::atoi(argv[1]) : 2048;
  const double tolerance = argc > 2 ? std::atof(argv[2]) : 1e-12;
  if (!std::isfinite(tolerance) || tolerance < 0)
    throw std::runtime_error("tolerance must be finite and nonnegative");
  if (n < 16 || n > 8192 || n % 16)
    throw std::runtime_error("n must be a multiple of 16 in [16,8192]");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0));
  if (properties.major < 7 || (properties.major == 7 && properties.minor < 5)) return 77;
  std::vector<double> host(n);
  std::mt19937_64 random(20261023);
  for (auto& x : host) x = std::exp2(-70.0 * std::generate_canonical<double, 53>(random));
  std::sort(host.begin(), host.end(), std::greater<double>());
  std::printf(
      "{\"schema\":\"direct-screening-probe.v1\",\"gpu\":\"%s\",\"slurm_job\":\"%s\","
      "\"visibility\":\"%s\",\"bounds\":%d,\"inner_repeats\":30}\n",
      properties.name, std::getenv("SLURM_JOB_ID"), std::getenv("CUDA_VISIBLE_DEVICES"), n);
  double* bounds{};
  float* upper{};
  signed char* codes{};
  int* exponents{};
  unsigned char* output{};
  check(cudaMalloc(&bounds, n * sizeof(double)));
  check(cudaMalloc(&upper, n * sizeof(float)));
  check(cudaMalloc(&codes, n * 16));
  check(cudaMalloc(&exponents, n / 16 * sizeof(int)));
  check(cudaMalloc(&output, std::size_t(n) * n));
  check(cudaMemcpy(bounds, host.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  const float packing_ms = timed([&] { pack<<<n / 16, 32>>>(bounds, n, codes, exponents); });
  const float float_packing_ms =
      timed([&] { pack_float<<<(n + 255) / 256, 256>>>(bounds, n, upper); });
  const char* names[]{"fp64",          "fp32",       "int8",         "int8_tensorcore",
                      "fp32_prepared", "int8_tiled", "fp32_interval"};
  for (bool force : {false, true}) {
    // A small force density makes the quadratic gate stricter than the linear gate.
    const double density = force ? 1e-4 : 0.125;
    std::vector<unsigned char> expected(std::size_t(n) * n);
    for (int mode = 0; mode < 7; ++mode) {
      auto launch = [&] {
        if (mode == 2 || mode == 3 || mode == 5) pack<<<n / 16, 32>>>(bounds, n, codes, exponents);
        const int grid = (n * n + 255) / 256;
        if (mode == 4 || mode == 6) {
          pack_float<<<(n + 255) / 256, 256>>>(bounds, n, upper);
          if (mode == 4)
            prepared_float<false>
                <<<grid, 256>>>(bounds, upper, n, density, tolerance, force, output);
          else
            prepared_float<true>
                <<<grid, 256>>>(bounds, upper, n, density, tolerance, force, output);
        }
        if (mode == 0)
          scalar<0><<<grid, 256>>>(bounds, n, codes, exponents, density, tolerance, force, output);
        if (mode == 1)
          scalar<1><<<grid, 256>>>(bounds, n, codes, exponents, density, tolerance, force, output);
        if (mode == 2)
          scalar<2><<<grid, 256>>>(bounds, n, codes, exponents, density, tolerance, force, output);
        if (mode == 3)
          tiled<true><<<dim3(n / 16, n / 16), 32>>>(bounds, n, codes, exponents, density, tolerance,
                                                    force, output);
        if (mode == 5)
          tiled<false><<<dim3(n / 16, n / 16), 32>>>(bounds, n, codes, exponents, density,
                                                     tolerance, force, output);
      };
      const float elapsed = timed(launch);
      std::vector<unsigned char> actual(expected.size());
      check(cudaMemcpy(actual.data(), output, actual.size(), cudaMemcpyDeviceToHost));
      std::size_t survivors = 0, rejected = 0, accepted = 0;
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const bool survives = (actual[i] & 1) != 0;
        if (mode == 0) {
          // Independent host FP64 admission, including every task identity.
          const double q = host[i / n] * host[i % n];
          const bool reference = !(q < tolerance || q * density < tolerance ||
                                   (force && q * density * density < std::fmin(tolerance, 1e-14)));
          if (survives != reference) throw std::runtime_error("FP64 host oracle mismatch");
          expected[i] = survives;
        } else if (survives != bool(expected[i]))
          throw std::runtime_error("task set changed");
        survivors += survives;
        rejected += actual[i] == 2;
        accepted += actual[i] == 5;
      }
      std::printf(
          "{\"mode\":\"%s\",\"force\":%s,\"bounds\":%d,\"candidates\":%zu,\"survivors\":%zu,"
          "\"coarse_rejections\":%zu,\"total_ms\":%.6f,\"packing_ms\":%.6f,\"tolerance\":%.17g,"
          "\"density\":%.17g,\"coarse_acceptances\":%zu,\"refinements\":%zu,\"mismatches\":0}\n",
          names[mode], force ? "true" : "false", n, actual.size(), survivors, rejected, elapsed,
          (mode == 4 || mode == 6) ? float_packing_ms
          : mode >= 2              ? packing_ms
                                   : 0,
          tolerance, density, accepted, actual.size() - accepted - rejected);
    }
  }
  check(cudaFree(bounds));
  check(cudaFree(upper));
  check(cudaFree(codes));
  check(cudaFree(exponents));
  check(cudaFree(output));
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
