#include <algorithm>
#include <array>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

#include "posthf/capacity.hpp"
#include "tensor/cuda_runtime.cuh"

namespace indexed_reduction_test {
using generativeqc::posthf::checked_add;
using generativeqc::posthf::checked_mul;

inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {
  std::size_t result = 1;
  for (const auto factor : factors) result = checked_mul(result, factor);
  return result;
}

struct ProbeState {
  std::size_t o{}, v{};
  const double* bar_gap{};
  double* response_arena{};
  int* error{};
  cudaStream_t stream{};
};
struct ProbeOutputs {
  const double *bar_eps_i{}, *bar_eps_j{}, *bar_eps_k{}, *bar_eps_v{};
};

#include "generated_indexed_reduction_probe.cuh"

/** Test-only device ownership; generated schedules receive the same detached seed. */
struct Resources {
  double *seed{}, *arena{};
  int* error{};
  cudaStream_t stream{};
  cudaEvent_t started{}, finished{};
  ~Resources() {
    if (stream) cudaStreamSynchronize(stream);
    cudaFree(seed);
    cudaFree(arena);
    cudaFree(error);
    if (started) cudaEventDestroy(started);
    if (finished) cudaEventDestroy(finished);
    if (stream) cudaStreamDestroy(stream);
  }
};
}  // namespace indexed_reduction_test

/** Compile and execute the actual serial/parallel compiler emission without CC. */
extern "C" int indexed_reduction_probe(std::size_t virtuals, const double* seed, bool parallel,
                                       double* occupied, double* virtual_output,
                                       std::size_t* counts, double* device_seconds, char* error,
                                       std::size_t error_size) noexcept {
  using namespace indexed_reduction_test;
  using generativeqc_tensor::cuda_check;
  try {
    if (!virtuals) throw std::invalid_argument("positive virtual dimension required");
    const auto elements = checked_product({virtuals, virtuals, virtuals});
    const auto workspace = parallel ? probe_parallel_arena_elements(1, virtuals)
                                    : probe_serial_arena_elements(1, virtuals);
    Resources resources;
    cuda_check(cudaStreamCreate(&resources.stream));
    cuda_check(cudaEventCreate(&resources.started));
    cuda_check(cudaEventCreate(&resources.finished));
    cuda_check(cudaMalloc(&resources.seed, checked_mul(elements, sizeof(double))));
    cuda_check(cudaMalloc(&resources.arena, checked_mul(workspace, sizeof(double))));
    cuda_check(cudaMalloc(&resources.error, sizeof(int)));
    cuda_check(cudaMemcpyAsync(resources.seed, seed, checked_mul(elements, sizeof(double)),
                               cudaMemcpyHostToDevice, resources.stream));
    cuda_check(cudaMemsetAsync(resources.error, 0, sizeof(int), resources.stream));
    ProbeState state{
        1, virtuals, resources.seed, resources.arena, resources.error, resources.stream};
    cuda_check(cudaEventRecord(resources.started, resources.stream));
    const auto output = parallel ? run_probe_parallel(state) : run_probe_serial(state);
    cuda_check(cudaEventRecord(resources.finished, resources.stream));
    int failed = 0;
    cuda_check(cudaMemcpyAsync(&failed, resources.error, sizeof(int), cudaMemcpyDeviceToHost,
                               resources.stream));
    cuda_check(cudaStreamSynchronize(resources.stream));
    if (failed) throw std::runtime_error("nonfinite generated reduction; outputs unpublished");
    float milliseconds = 0;
    cuda_check(cudaEventElapsedTime(&milliseconds, resources.started, resources.finished));
    *device_seconds = milliseconds * 1e-3;
    const std::array scalars{output.bar_eps_i, output.bar_eps_j, output.bar_eps_k};
    for (std::size_t index = 0; index < scalars.size(); ++index)
      cuda_check(cudaMemcpyAsync(occupied + index, scalars[index], sizeof(double),
                                 cudaMemcpyDeviceToHost, resources.stream));
    cuda_check(cudaMemcpyAsync(virtual_output, output.bar_eps_v,
                               checked_mul(virtuals, sizeof(double)), cudaMemcpyDeviceToHost,
                               resources.stream));
    cuda_check(cudaStreamSynchronize(resources.stream));
    counts[0] = checked_mul(workspace, sizeof(double));
    counts[1] = parallel ? probe_parallel_kernel_count(1, virtuals) : 8;
    counts[2] = parallel ? probe_parallel_materialized_elements(1, virtuals)
                         : checked_add(checked_mul(2, elements), checked_mul(4, virtuals));
    counts[3] = parallel ? probe_parallel_value_reads(1, virtuals)
                         : checked_add(checked_mul(6, elements), checked_mul(4, virtuals));
    counts[4] =
        parallel ? probe_parallel_value_writes(1, virtuals)
                 : checked_add(checked_add(checked_mul(2, elements), checked_mul(5, virtuals)), 1);
    counts[5] =
        parallel ? probe_parallel_reduction_summands(1, virtuals) : checked_mul(4, elements);
    return 0;
  } catch (const std::exception& failure) {
    if (error && error_size) std::snprintf(error, error_size, "%s", failure.what());
    return 1;
  }
}
