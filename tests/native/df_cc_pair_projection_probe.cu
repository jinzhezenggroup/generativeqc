#include <bit>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cc/df_pair_projection.cuh"
#include "tensor/cuda_error.hpp"

using generativeqc_tensor::cuda_check;
namespace folded = generativeqc::cc::generated::dfpairs;
namespace bound = generativeqc::cc::pair_bound;

struct Buffers {
  std::vector<void*> allocations;
  void* allocate(std::size_t bytes) {
    void* pointer{};
    cuda_check(cudaMalloc(&pointer, bytes));
    allocations.push_back(pointer);
    return pointer;
  }
  ~Buffers() {
    for (auto pointer : allocations) cudaFree(pointer);
  }
};

int main() {
  try {
    std::size_t occupied{}, virtuals{};
    std::cin >> occupied >> virtuals;
    if (!occupied || !virtuals) throw std::invalid_argument("empty dimensions");
    const auto singles_count = folded::checked_mul(occupied, virtuals);
    const auto doubles_count = folded::checked_mul(singles_count, singles_count);
    const auto paired_count = folded::occupied_tau_elements(occupied, virtuals);
    std::vector<std::uint64_t> singles(singles_count), tau(doubles_count);
    for (auto& bits : singles) std::cin >> bits;
    for (auto& bits : tau) std::cin >> bits;
    if (!std::cin) throw std::invalid_argument("truncated input");
    Buffers buffers;
    auto* device_singles = static_cast<double*>(buffers.allocate(singles_count * sizeof(double)));
    auto* device_tau = static_cast<double*>(buffers.allocate(doubles_count * sizeof(double)));
    auto* device_paired =
        static_cast<double*>(buffers.allocate((paired_count + 2) * sizeof(double)));
    auto* device_metadata =
        static_cast<bound::ProjectionMaxima*>(buffers.allocate(sizeof(bound::ProjectionMaxima)));
    cuda_check(cudaMemcpy(device_singles, singles.data(), singles_count * sizeof(double),
                          cudaMemcpyHostToDevice));
    cuda_check(
        cudaMemcpy(device_tau, tau.data(), doubles_count * sizeof(double), cudaMemcpyHostToDevice));
    constexpr auto canary = 0x3ff123456789abcdULL;
    std::vector<std::uint64_t> paired(paired_count + 2, canary);
    cuda_check(cudaMemcpy(device_paired, paired.data(), paired.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(device_metadata, 0xff, sizeof(bound::ProjectionMaxima)));
    bound::project_tau(device_tau, device_singles, occupied, virtuals, doubles_count, singles_count,
                       device_paired + 1, device_metadata, nullptr);
    bound::ProjectionMaxima metadata{};
    cuda_check(cudaMemcpy(&metadata, device_metadata, sizeof(metadata), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(paired.data(), device_paired, paired.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    std::vector<std::uint64_t> retained(doubles_count);
    cuda_check(cudaMemcpy(retained.data(), device_tau, retained.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    if (retained != tau) throw std::runtime_error("original tau was modified");
    if (paired.front() != canary || paired.back() != canary)
      throw std::runtime_error("projection crossed a canary");
    std::cout << metadata.refused << ' ' << metadata.tau_error_bits << ' '
              << metadata.t1_magnitude_bits << ' ' << metadata.tau_magnitude_bits << '\n';
    for (std::size_t index = 1; index <= paired_count; ++index) std::cout << paired[index] << ' ';
    std::cout << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
