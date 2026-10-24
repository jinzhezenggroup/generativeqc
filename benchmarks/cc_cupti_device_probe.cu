// Bounded collector/device transport control. This is not a scientific endpoint.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <poll.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cc/phase_observation.hpp"
#include "cc_source_session.hpp"

namespace phase = generativeqc::cc::phase_observation;

__global__ void affine(const int* input, int* output) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 1024) output[index] = 3 * input[index] + 7;
}

void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) return 2;
  try {
    generativeqc::benchmarks::cc_trace::Session observation(argv[1], argv[2]);
    int count = 0;
    checked(cudaGetDeviceCount(&count));
    if (count != 1) throw std::runtime_error("probe requires one allocated visible device");
    cudaDeviceProp properties{};
    checked(cudaGetDeviceProperties(&properties, 0));
    if (std::string(properties.name).find("H200") == std::string::npos)
      throw std::runtime_error("probe requires the allocated H200 recipe");
    std::cout << "DEVICE_UUID=GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
      if (index == 4 || index == 6 || index == 8 || index == 10) std::cout << '-';
      std::cout << std::setw(2)
                << static_cast<unsigned>(static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    std::cout << std::dec << '\n' << std::flush;
    if (argc == 4) {
      if (std::string(argv[3]) != "--parent-ready-v1") return 2;
      pollfd ready{STDIN_FILENO, POLLIN, 0};
      char signal = 0;
      if (poll(&ready, 1, 15000) != 1 || read(STDIN_FILENO, &signal, 1) != 1 || signal != 'G')
        throw std::runtime_error("parent module attestation handshake failed");
    }
    {
      phase::Scope root(phase::Site::endpoint_inclusive);
      checked(cudaSetDevice(0));
      int* input = nullptr;
      int* output = nullptr;
      checked(cudaMalloc(&input, 1024 * sizeof(int)));
      try {
        checked(cudaMalloc(&output, 1024 * sizeof(int)));
        std::array<int, 1024> host{}, result{};
        for (std::size_t i = 0; i < host.size(); ++i) host[i] = static_cast<int>(i);
        {
          phase::Scope copy(phase::Site::source_copy_submission, 0, sizeof(host));
          checked(cudaMemset(output, 0, sizeof(host)));
          checked(cudaMemcpy(input, host.data(), sizeof(host), cudaMemcpyHostToDevice));
        }
        {
          phase::Scope producer(phase::Site::q_production_submission, 0, 1024);
          affine<<<4, 256>>>(input, output);
          checked(cudaGetLastError());
        }
        {
          phase::Scope wait(phase::Site::explicit_wait);
          checked(cudaDeviceSynchronize());
        }
        {
          phase::Scope copy(phase::Site::source_copy_submission, 1, sizeof(result));
          checked(cudaMemcpy(result.data(), output, sizeof(result), cudaMemcpyDeviceToHost));
        }
        for (std::size_t i = 0; i < result.size(); ++i)
          if (result[i] != 3 * static_cast<int>(i) + 7)
            throw std::runtime_error("device transport oracle mismatch");
        checked(cudaFree(output));
        output = nullptr;
      } catch (...) {
        if (output) (void)cudaFree(output);
        (void)cudaFree(input);
        throw;
      }
      checked(cudaFree(input));
    }
    if (!observation.finish()) return 3;
    std::cout << "DEVICE_TRANSPORT_CONTROL_PASS; SCIENCE_NOT_RUN\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
