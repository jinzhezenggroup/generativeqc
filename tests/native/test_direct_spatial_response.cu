#include <cuda_runtime.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "probe_shell_pair.cuh"

using namespace generativeqc::scf::cuda_execution;

struct Request {
  unsigned order, radial, mask, selected;
  unsigned powers[12];
  double positions[12], exponents[4];
};

void require_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

/** Instantiate the production shell-bounded consumer, with no launch tuning. */
template <unsigned Order>
__global__ void spatial_response_probe(const Request* requests, unsigned count, double* output) {
  const unsigned ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count || requests[ordinal].order != Order) return;
  const auto& request = requests[ordinal];
  Angular angular[4];
  Vec3<Dual3> centers[4];
  for (unsigned center = 0; center < 4; ++center) {
    angular[center] = {request.powers[3 * center], request.powers[3 * center + 1],
                       request.powers[3 * center + 2]};
    const double seed = (request.mask & (1U << center)) ? 1.0 : 0.0;
    centers[center] = {{request.positions[3 * center], seed, 0.0, 0.0},
                       {request.positions[3 * center + 1], 0.0, seed, 0.0},
                       {request.positions[3 * center + 2], 0.0, 0.0, seed}};
    if (request.mask & 16U) {
      centers[center].x.derivative_y = 0.25 * seed;
      centers[center].x.derivative_z = -0.5 * seed;
      centers[center].y.derivative_x = -0.75 * seed;
      centers[center].y.derivative_z = 0.125 * seed;
      centers[center].z.derivative_x = 0.375 * seed;
      centers[center].z.derivative_y = -0.25 * seed;
    }
  }
  constexpr unsigned shell = Order == 12 ? 3U : 2U;
  constexpr unsigned final_shell = Order == 7 ? 1U : shell;
  const Dual3 value = primitive_eri_cartesian_shell_pairs<shell, shell, shell, final_shell, Dual3>(
      request.exponents[0], centers[0], angular[0], request.exponents[1], centers[1], angular[1],
      request.exponents[2], centers[2], angular[2], request.exponents[3], centers[3], angular[3],
      static_cast<generativeqc::integrals::CoulombRange>(request.radial), 0.3,
      (request.selected & 1U) != 0U, (request.selected & 2U) != 0U);
  output[4 * ordinal] = value.value;
  output[4 * ordinal + 1] = value.derivative_x;
  output[4 * ordinal + 2] = value.derivative_y;
  output[4 * ordinal + 3] = value.derivative_z;
}

template <unsigned Order>
void measure(const Request* requests, unsigned count, double* output, bool separator) {
  cudaFuncAttributes attributes{};
  require_cuda(cudaFuncGetAttributes(&attributes, spatial_response_probe<Order>));
  cudaEvent_t begin, end;
  require_cuda(cudaEventCreate(&begin));
  require_cuda(cudaEventCreate(&end));
  require_cuda(cudaEventRecord(begin));
  spatial_response_probe<Order><<<(count + 127U) / 128U, 128>>>(requests, count, output);
  require_cuda(cudaGetLastError());
  require_cuda(cudaEventRecord(end));
  require_cuda(cudaEventSynchronize(end));
  float milliseconds;
  require_cuda(cudaEventElapsedTime(&milliseconds, begin, end));
  require_cuda(cudaEventDestroy(begin));
  require_cuda(cudaEventDestroy(end));
  std::cout << (separator ? "," : "") << "{\"order\":" << Order
            << ",\"registers_per_thread\":" << attributes.numRegs
            << ",\"local_bytes_per_thread\":" << attributes.localSizeBytes
            << ",\"threads_per_block\":128,\"kernel_ms\":" << milliseconds << "}";
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    std::ifstream input(argv[1]);
    std::vector<Request> requests;
    Request request;
    while (input >> request.order >> request.radial >> request.mask >> request.selected) {
      for (auto& power : request.powers) input >> power;
      for (auto& position : request.positions) input >> position;
      for (auto& exponent : request.exponents) input >> exponent;
      if (!input) throw std::runtime_error("incomplete request");
      requests.push_back(request);
    }
    if (requests.empty()) throw std::runtime_error("no requests");
    Request* device_requests;
    double* device_output;
    std::vector<double> output(requests.size() * 4U);
    require_cuda(cudaMalloc(&device_requests, requests.size() * sizeof(Request)));
    require_cuda(cudaMalloc(&device_output, output.size() * sizeof(double)));
    require_cuda(cudaMemset(device_output, 0, output.size() * sizeof(double)));
    require_cuda(cudaMemcpy(device_requests, requests.data(), requests.size() * sizeof(Request),
                            cudaMemcpyHostToDevice));
    std::cout << std::setprecision(17) << "{\"resources\":[";
    measure<7>(device_requests, requests.size(), device_output, false);
    measure<8>(device_requests, requests.size(), device_output, true);
    measure<12>(device_requests, requests.size(), device_output, true);
    require_cuda(cudaMemcpy(output.data(), device_output, output.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
    require_cuda(cudaFree(device_requests));
    require_cuda(cudaFree(device_output));
    std::cout << "],\"jets\":[";
    for (unsigned ordinal = 0; ordinal < requests.size(); ++ordinal) {
      std::cout << (ordinal ? ",[" : "[");
      for (unsigned axis = 0; axis < 4; ++axis)
        std::cout << (axis ? "," : "") << output[4 * ordinal + axis];
      std::cout << "]";
    }
    std::cout << "]}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
