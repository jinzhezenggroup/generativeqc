// Fault injection for asynchronous host staging; no CUDA device is accessed.
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <string>

#include "tensor/cuda_error.hpp"

namespace {
void* pending_destination = nullptr;
bool freed_before_drain = false;
unsigned copies = 0;
std::size_t available_bytes = 0;
cudaError_t synchronize_error = cudaSuccess;
cudaError_t mem_info_error = cudaSuccess;
bool synchronized = false;

cudaError_t mock_copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind,
                      cudaStream_t) {
  ++copies;
  if (copies <= 3) {
    std::memcpy(destination, source, bytes);  // convergence flags and iteration count
    return cudaSuccess;
  }
  if (copies == 4) {
    pending_destination = destination;
    return cudaSuccess;
  }
  return cudaErrorInvalidValue;  // next copy fails while the first is pending
}

cudaError_t mock_synchronize(cudaStream_t) {
  synchronized = true;
  pending_destination = nullptr;
  return synchronize_error;
}

cudaError_t mock_mem_info(std::size_t* available, std::size_t* total) {
  if (!synchronized) throw std::logic_error("memory snapshot preceded stream synchronization");
  *available = available_bytes;
  *total = std::numeric_limits<std::size_t>::max();
  return mem_info_error;
}
}  // namespace

// Observe destruction without dereferencing freed storage or relying on an
// allocator to overwrite it. Both sized and unsized vector deletion are covered.
void operator delete(void* pointer) noexcept {
  if (pointer && pointer == pending_destination) freed_before_drain = true;
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

#define cudaMemcpyAsync mock_copy
#define cudaStreamSynchronize mock_synchronize
#define cudaMemGetInfo mock_mem_info
#include "scf/cuda/reference_export.cuh"
#undef cudaMemcpyAsync
#undef cudaStreamSynchronize
#undef cudaMemGetInfo

namespace {
void retention_guard() {
  using generativeqc::scf::reference_detail::retained_bytes;
  available_bytes = 7167;
  bool refused = false;
  try {
    retained_bytes(nullptr, 8192, 1024);
  } catch (const std::length_error& error) {
    refused = true;
    const std::string message = error.what();
    if (message.find("CUDA RHF provider exceeds retained allocation allowance") != 0)
      throw std::runtime_error("provider refusal lost its recognizable prefix");
    for (const auto* field : {"free_before_bytes=8192", "free_after_bytes=7167",
                              "observed_delta_bytes=1025", "allowance_bytes=1024"}) {
      if (message.find(field) == std::string::npos)
        throw std::runtime_error(std::string("provider refusal omitted ") + field + ": " + message);
    }
  }
  if (!refused) throw std::runtime_error("one byte over provider allowance was accepted");

  struct Case {
    std::size_t before, after, allowance, expected;
  };
  const auto maximum = std::numeric_limits<std::size_t>::max();
  for (const auto& item :
       {Case{8192, 7168, 1024, 1024}, Case{8192, 8192, 0, 0}, Case{8192, 8193, 0, 0},
        Case{0, maximum, 0, 0}, Case{maximum, 0, maximum, maximum}}) {
    synchronized = false;
    available_bytes = item.after;
    if (retained_bytes(nullptr, item.before, item.allowance) != item.expected)
      throw std::runtime_error("provider retention boundary changed");
  }
  available_bytes = 0;
  try {
    retained_bytes(nullptr, maximum, maximum - 1);
    throw std::runtime_error("maximum byte count refusal was lost");
  } catch (const std::length_error& error) {
    const std::string message = error.what();
    for (const auto& field : {"free_before_bytes=" + std::to_string(maximum),
                              "observed_delta_bytes=" + std::to_string(maximum),
                              "allowance_bytes=" + std::to_string(maximum - 1)})
      if (message.find(field) == std::string::npos)
        throw std::runtime_error("maximum byte count was truncated");
  }
  // Whole-budget rejection must remain distinguishable from provider refusal.
  try {
    generativeqc::scf::reference_detail::check_capacity(1024, 1, 1024);
    throw std::runtime_error("numeric budget overrun was accepted");
  } catch (const std::length_error& error) {
    if (std::string(error.what()) != "CUDA RHF reference exceeds numeric memory budget")
      throw std::runtime_error("numeric budget refusal was reclassified");
  }
  for (bool fail_sync : {false, true}) {
    for (auto status : {cudaErrorMemoryAllocation, cudaErrorInvalidValue}) {
      synchronize_error = fail_sync ? status : cudaSuccess;
      mem_info_error = fail_sync ? cudaSuccess : status;
      bool preserved = false;
      try {
        retained_bytes(nullptr, 8192, 0);
      } catch (const generativeqc_tensor::DeviceAllocationError& error) {
        preserved = status == cudaErrorMemoryAllocation &&
                    std::string(error.what()) == cudaGetErrorString(status);
      } catch (const generativeqc_tensor::DeviceRuntimeError& error) {
        preserved = status == cudaErrorInvalidValue &&
                    std::string(error.what()) == cudaGetErrorString(status);
      }
      if (!preserved) throw std::runtime_error("CUDA error type or message was lost");
    }
  }
  synchronize_error = mem_info_error = cudaSuccess;
  std::cout << "Provider retention diagnostics, boundaries and CUDA errors passed\n";
}
}  // namespace

int main() {
  using generativeqc::scf::reference_detail::download;
  const std::array<const double*, 5> matrices{};
  const std::array<const double*, 3> scalars{};
  std::uint8_t converged = 0, failed = 0;
  std::uint32_t iterations = 1;
  generativeqc::scf::ScfResult result;
  try {
    retention_guard();
    // An unconverged solve must return its status before dimension checking or
    // any dense allocation, even when a reference could never fit in memory.
    auto status = download(nullptr, std::numeric_limits<std::size_t>::max(), 1, 0, matrices,
                           nullptr, scalars, &converged, &failed, &iterations, result);
    if (status != GENERATIVEQC_STATUS_NOT_CONVERGED || result.reference)
      throw std::runtime_error("unconverged reference was allocated/published");
    copies = 0;
    converged = 1;
    bool threw = false;
    try {
      download(nullptr, 2, 1, 0, matrices, nullptr, scalars, &converged, &failed, &iterations,
               result);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    if (!threw || copies != 5 || pending_destination || freed_before_drain || result.reference)
      throw std::runtime_error("failed reference copy did not drain before freeing staging");
    std::cout << "Reference convergence preflight and exceptional staging lifetime passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
