#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated_direct_resident_psss_schedule.cuh"
#include "scf/cuda/direct_tile_compaction.hpp"
#include "scf/direct_task_layout.hpp"

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

/** Keep the borrowed queue storage alive until every submitted reset/copy retires. */
struct DeviceStorage {
  std::uint32_t* data{};
  cudaStream_t stream{};

  ~DeviceStorage() {
    if (stream) cudaStreamSynchronize(stream);
    if (data) cudaFree(data);
    if (stream) cudaStreamDestroy(stream);
  }
};

}  // namespace

int main() {
  try {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && device_count == 0)) return 77;
    check_cuda(status, "cudaGetDeviceCount");
    using namespace generativeqc::scf;
    using namespace generativeqc::scf::cuda_execution;
    constexpr auto count_elements = detail::kDirectQuartetAngularOrderCount;
    constexpr auto head_elements = kPersistentFockAngularOrderCount;
    constexpr auto head_offset = count_elements + 1;
    constexpr auto fp32_count_offset = head_offset + head_elements + 1;
    constexpr auto fp32_head_offset = fp32_count_offset + count_elements + 1;
    constexpr auto total_elements = fp32_head_offset + head_elements + 1;
    constexpr std::uint32_t sentinel = 0x7f7f7f7fU;
    DeviceStorage storage;
    check_cuda(cudaStreamCreate(&storage.stream), "cudaStreamCreate");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&storage.data), total_elements * sizeof(std::uint32_t)),
        "cudaMalloc");
    std::vector<std::uint32_t> observed(total_elements);
    for (bool mixed_precision : {false, true}) {
      check_cuda(cudaMemsetAsync(storage.data, 0x7f, total_elements * sizeof(std::uint32_t),
                                 storage.stream),
                 "cudaMemsetAsync");
      launch_clear_active_shell_quartet_tile_counts_kernel(
          dim3(1), dim3(32), 0, storage.stream, storage.data, storage.data + head_offset,
          mixed_precision ? storage.data + fp32_count_offset : nullptr,
          mixed_precision ? storage.data + fp32_head_offset : nullptr);
      check_cuda(cudaGetLastError(), "clear queue heads");
      check_cuda(
          cudaMemcpyAsync(observed.data(), storage.data, total_elements * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, storage.stream),
          "cudaMemcpyAsync");
      check_cuda(cudaStreamSynchronize(storage.stream), "cudaStreamSynchronize");
      for (std::size_t index = 0; index != total_elements; ++index) {
        const bool reset = index < count_elements ||
                           (index >= head_offset && index < head_offset + head_elements) ||
                           (mixed_precision && index >= fp32_count_offset &&
                            index < fp32_count_offset + count_elements) ||
                           (mixed_precision && index >= fp32_head_offset &&
                            index < fp32_head_offset + head_elements);
        if (observed[index] != (reset ? 0U : sentinel))
          throw std::runtime_error("queue reset crossed its emitted profile/storage boundary");
      }
    }
    std::cout << "persistent queue reset passed: Fock orders=" << head_elements
              << ", force orders=" << kPersistentForceAngularOrderCount << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
