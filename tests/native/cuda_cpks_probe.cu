#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "response/cuda_cpks.hpp"

namespace {
namespace response = generativeqc::response;

void require(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct DeviceStorage {
  std::vector<void*> allocations;
  ~DeviceStorage() {
    for (auto* pointer : allocations) cudaFree(pointer);
  }
  double* allocate(std::size_t elements) {
    double* pointer{};
    require(cudaMalloc(&pointer, std::max<std::size_t>(elements, 1) * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  }
  double* upload(const std::vector<double>& values) {
    auto* pointer = allocate(values.size());
    require(
        cudaMemcpy(pointer, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    return pointer;
  }
};

std::vector<double> read(std::size_t count) {
  std::vector<double> values(count);
  for (auto& value : values) {
    std::string token;
    if (!(std::cin >> token)) throw std::runtime_error("missing numerical input");
    value = std::stod(token);
  }
  return values;
}

// A deliberately independent, immutable linear test model, not a molecular
// induced-Fock implementation or a replacement for native integral/grid work.
__global__ void linear_fock(const double* operation, const double* density, double* fock,
                            std::size_t dimension, int failure) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < dimension;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double value{};
    for (std::size_t column = 0; column < dimension; ++column)
      value += operation[row * dimension + column] * density[column];
    if (failure == 1 && row == 0) value = nan("");
    if (failure == 2 && row == 1) value += 0.3;
    if (failure == 3 && row == 0) continue;
    fock[row] = value;
  }
}

struct Owner {
  const double* operation{};
  std::string mode;
  unsigned authentications{}, actions{};
};

cudaError_t authenticate(void* context) noexcept {
  auto& owner = *static_cast<Owner*>(context);
  ++owner.authentications;
  if (owner.mode == "stale-entry" || (owner.mode == "stale-publish" && owner.authentications > 1))
    return cudaErrorInvalidResourceHandle;
  return cudaSuccess;
}

cudaError_t induced(void* context, const double* density, double* fock,
                    const response::CudaCpksShape& shape, cudaStream_t stream) noexcept {
  auto& owner = *static_cast<Owner*>(context);
  ++owner.actions;
  if (owner.mode == "fail-provider" || (owner.mode == "late-provider" && owner.actions >= 3))
    return cudaErrorLaunchFailure;
  const auto dimension = shape.nao * shape.nao * shape.spins;
  const int failure = owner.mode == "nan-provider"          ? 1
                      : owner.mode == "asymmetric-provider" ? 2
                      : owner.mode == "partial-provider"    ? 3
                                                            : 0;
  linear_fock<<<static_cast<unsigned>((dimension + 127) / 128), 128, 0, stream>>>(
      owner.operation, density, fock, dimension, failure);
  return cudaGetLastError();
}

void array(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void contract() {
  response::CudaCpksShape shape{3, 2, {2, 1}};
  response::CudaCpksOptions options;
  std::size_t bytes{};
  require(response::cuda_cpks_workspace(shape, options, bytes));
  if (!bytes || bytes % sizeof(double)) throw std::runtime_error("invalid workspace");
  const auto refused = [&](response::CudaCpksShape invalid, response::CudaCpksOptions settings) {
    std::size_t size = 17;
    if (response::cuda_cpks_workspace(invalid, settings, size) != cudaErrorInvalidValue || size)
      throw std::runtime_error("admission accepted invalid contract");
  };
  refused({0, 1, {0, 0}}, options);
  refused({3, 3, {1, 1}}, options);
  refused({3, 1, {1, 1}}, options);
  refused({3, 2, {4, 1}}, options);
  refused({std::numeric_limits<std::size_t>::max(), 1, {1, 0}}, options);
  options.max_workspace_bytes = bytes - 1;
  refused(shape, options);
  options = {};
  options.gmres.restart = 0;
  refused(shape, options);
  options = {};
  options.symmetry_relative_tolerance = std::numeric_limits<double>::quiet_NaN();
  refused(shape, options);
  std::cout << "{\"contract\":\"passed\"}\n";
}

void solve(int argc, char** argv) {
  if (argc != 12) throw std::runtime_error("solve requires ten arguments");
  response::CudaCpksShape shape{std::stoull(argv[2]),
                                static_cast<unsigned>(std::stoul(argv[3])),
                                {std::stoull(argv[4]), std::stoull(argv[5])}};
  const bool column_major = std::string(argv[6]) == "column";
  const bool reference_only = std::string(argv[1]) == "reference";
  Owner owner{nullptr, argv[7]};
  response::CudaCpksOptions options;
  options.gmres.max_iterations = std::stoull(argv[8]);
  options.gmres.restart = std::stoull(argv[9]);
  options.gmres.relative_tolerance = 1e-11;
  options.gmres.absolute_tolerance = 1e-13;
  options.max_workspace_bytes = std::stoull(argv[10]);
  options.gmres.diagonal_floor = std::stod(argv[11]);
  const auto matrix = shape.nao * shape.nao;
  const auto matrices = matrix * shape.spins;
  auto coefficients = read(matrices);
  auto energies = read(shape.spins * shape.nao);
  auto frozen = read(matrices);
  auto overlap = read(matrix);
  auto operation = read(matrices * matrices);
  if (owner.mode == "nan-input") coefficients[0] = std::numeric_limits<double>::quiet_NaN();
  if (owner.mode == "nan-energy") energies[0] = std::numeric_limits<double>::quiet_NaN();
  if (owner.mode == "asymmetric-input") frozen[1] += 0.3;
  if (owner.mode == "zero-gap") energies[shape.occupied[0]] = energies[0];
  DeviceStorage storage;
  cudaStream_t stream{};
  require(cudaStreamCreate(&stream));
  auto* device_coefficients = storage.upload(coefficients);
  auto* device_energies = storage.upload(energies);
  auto* device_frozen = storage.upload(frozen);
  auto* device_overlap = storage.upload(overlap);
  owner.operation = storage.upload(operation);
  auto* density = storage.upload(std::vector<double>(matrices, 9817.0));
  auto* weights = storage.upload(std::vector<double>(matrices, -3277.0));
  std::size_t required{};
  const auto admitted = response::cuda_cpks_workspace(shape, options, required);
  auto* workspace = storage.allocate(required / sizeof(double) + 1);
  const response::CudaCpksReference reference{
      device_coefficients,
      device_energies,
      column_major ? response::CudaCpksLayout::column_major : response::CudaCpksLayout::row_major,
      stream,
      authenticate,
      &owner};
  response::CudaCpksResult result;
  auto* selected_workspace = owner.mode == "alias-workspace" ? density : workspace;
  auto* selected_weights = owner.mode == "alias-output" ? density : weights;
  if (reference_only && owner.mode == "alias-workspace") selected_workspace = device_coefficients;
  if (reference_only && owner.mode == "alias-output") selected_weights = device_coefficients;
  if (owner.mode == "misaligned-workspace")
    selected_workspace = reinterpret_cast<double*>(reinterpret_cast<char*>(workspace) + 1);
  if (owner.mode == "capture") require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  const auto error = admitted != cudaSuccess ? admitted
                     : reference_only
                         ? response::cuda_cpks_reference_weights(
                               shape, options, reference, selected_weights, selected_workspace,
                               owner.mode == "short-workspace" ? required - 1 : required,
                               owner.mode == "wrong-stream" ? nullptr : stream)
                         : response::solve_cuda_cpks(
                               shape, options, reference, device_frozen, device_overlap, induced,
                               &owner, density, selected_weights, selected_workspace,
                               owner.mode == "short-workspace" ? required - 1 : required,
                               owner.mode == "wrong-stream" ? nullptr : stream, result);
  if (owner.mode == "capture") {
    cudaGraph_t captured{};
    require(cudaStreamEndCapture(stream, &captured));
    require(cudaGraphDestroy(captured));
  }
  require(cudaStreamSynchronize(stream));
  std::vector<double> output_density(matrices), output_weights(matrices);
  require(cudaMemcpy(output_density.data(), density, matrices * sizeof(double),
                     cudaMemcpyDeviceToHost));
  require(cudaMemcpy(output_weights.data(), weights, matrices * sizeof(double),
                     cudaMemcpyDeviceToHost));
  require(cudaStreamDestroy(stream));
  std::cout << std::setprecision(17) << "{\"cuda_error\":" << static_cast<int>(error)
            << ",\"status\":" << static_cast<int>(result.status)
            << ",\"converged\":" << (result.converged() ? "true" : "false")
            << ",\"iterations\":" << result.gmres.iterations
            << ",\"residual\":" << result.gmres.residual_norm
            << ",\"tolerance\":" << result.gmres.tolerance
            << ",\"gmres_status\":" << static_cast<int>(result.gmres.status)
            << ",\"workspace_bytes\":" << result.workspace_bytes
            << ",\"authentications\":" << owner.authentications << ",\"actions\":" << owner.actions
            << ",\"density\":";
  array(output_density);
  std::cout << ",\"weights\":";
  array(output_weights);
  std::cout << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "contract")
      contract();
    else if (argc > 1 && (std::string(argv[1]) == "solve" || std::string(argv[1]) == "reference"))
      solve(argc, argv);
    else
      throw std::runtime_error("expected contract, solve or reference");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
