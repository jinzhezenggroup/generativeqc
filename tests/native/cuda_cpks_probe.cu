#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_xc_transition_integer.cuh"
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
                            std::size_t dimension, int failure, double shift) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < dimension;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double value{};
    for (std::size_t column = 0; column < dimension; ++column)
      value += operation[row * dimension + column] * density[column];
    if (failure == 1 && row == 0) value = nan("");
    if (failure == 2 && row == 1) value += 0.3;
    if (failure == 3 && row == 0) continue;
    if (failure == 4) value += 0.01 * density[row] * density[row];
    fock[row] = shift ? value - shift * density[row] : value;
  }
}

struct Owner {
  const double* operation{};
  std::string mode;
  unsigned authentications{}, actions{}, seeds{}, complements{};
};
constexpr double overlap_shift = 0.03125;

__global__ void seed_diagonal(double* output, std::size_t dimension, double diagonal) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < dimension * dimension)
    output[index] = index % dimension == index / dimension ? diagonal : 0;
}

__global__ void diagonal_fock(const double* density, double* fock, std::size_t dimension) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < dimension) fock[index] = overlap_shift * density[index];
}

cudaError_t seed(void* context, const response::CudaCpksShape& shape,
                 const response::CudaCpksReference&, double* output, cudaStream_t stream) noexcept {
  auto& owner = *static_cast<Owner*>(context);
  ++owner.seeds;
  const auto rotations = shape.occupied[0] * (shape.nao - shape.occupied[0]);
  const bool bad = owner.mode != "cached-seed" && owner.mode != "cached-seed-complete" &&
                   owner.mode != "cached-seed-complete-diagonal";
  seed_diagonal<<<static_cast<unsigned>((rotations * rotations + 127) / 128), 128, 0, stream>>>(
      output, rotations, (bad ? 16.0 : 2.0) * overlap_shift);
  auto error = cudaGetLastError();
  if (error == cudaSuccess && owner.mode.ends_with("-refused")) {
    error = cudaStreamSynchronize(stream);
    return error == cudaSuccess ? cudaErrorNotSupported : error;
  }
  return error;
}

cudaError_t complement(void* context, const double* density, double* fock,
                       const response::CudaCpksShape& shape, cudaStream_t stream) noexcept {
  auto& owner = *static_cast<Owner*>(context);
  ++owner.complements;
  const auto dimension = shape.nao * shape.nao;
  linear_fock<<<static_cast<unsigned>((dimension + 127) / 128), 128, 0, stream>>>(
      owner.operation, density, fock, dimension, 0, overlap_shift);
  return cudaGetLastError();
}

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
  if (owner.mode == "cached-seed-complete-diagonal") {
    diagonal_fock<<<static_cast<unsigned>((dimension + 127) / 128), 128, 0, stream>>>(density, fock,
                                                                                      dimension);
    return cudaGetLastError();
  }
  const int failure = owner.mode == "nan-provider"                 ? 1
                      : owner.mode == "asymmetric-provider"        ? 2
                      : owner.mode == "partial-provider"           ? 3
                      : owner.mode.starts_with("cached-nonlinear") ? 4
                                                                   : 0;
  linear_fock<<<static_cast<unsigned>((dimension + 127) / 128), 128, 0, stream>>>(
      owner.operation, density, fock, dimension, failure, 0.0);
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

void integer_projection() {
  namespace integer = generativeqc::dft::transition_integer;
  constexpr int order = 2 * integer::block_width, rank = 32, tail = 12;
  constexpr auto panel = std::size_t(order) * rank;
  constexpr auto square = std::size_t(order) * order;
  DeviceStorage storage;
  struct Library {
    cublasHandle_t handle{};
    ~Library() {
      if (handle) cublasDestroy(handle);
    }
  } blas;
  if (cublasCreate(&blas.handle) != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("integer projection BLAS initialization");
  auto* output = storage.allocate(square);
  auto* exponent = reinterpret_cast<int*>(storage.allocate(order));
  auto* product = reinterpret_cast<int*>(storage.allocate(square / 2));
  const auto digits = [&] {
    return reinterpret_cast<signed char*>(
        storage.allocate(panel * integer::digits / sizeof(double)));
  };
  integer::Workspace workspace{product, exponent, digits(), digits(), digits(), digits()};
  auto* status = reinterpret_cast<int*>(storage.allocate(1));
  require(cudaMemset(status, 0, sizeof(int)));
  std::vector<double> left(panel), right(panel), left_tail(order * tail), right_tail(order * tail);
  const auto populate = [](auto& values, int pitch, int shift) {
    for (int column = 0; column < order; ++column)
      for (int row = 0; row < pitch; ++row) {
        // Scale each rotation independently and mix positive and negative
        // mantissas, including digits well below the most significant plane.
        const auto sample = std::sin(0.17 * (5 + column * pitch + row) * shift);
        values[column * pitch + row] = std::ldexp(sample, (column + shift) % 35 - 17);
      }
  };
  populate(left, rank, 3);
  populate(right, rank, 4);
  populate(left_tail, tail, 7);
  populate(right_tail, tail, 9);
  require(integer::project(blas.handle, workspace, storage.upload(left), storage.upload(right),
                           output, order, rank, true, status, nullptr));
  require(integer::project(blas.handle, workspace, storage.upload(left_tail),
                           storage.upload(right_tail), output, order, tail, false, status,
                           nullptr));
  std::vector<double> actual(square);
  require(cudaMemcpy(actual.data(), output, square * sizeof(double), cudaMemcpyDeviceToHost));
  int refused{};
  require(cudaMemcpy(&refused, status, sizeof(int), cudaMemcpyDeviceToHost));
  if (refused) throw std::runtime_error("integer projection refused bounded input");
  double worst{};
  for (int column = 0; column < order; ++column)
    for (int row = column; row < order; ++row) {
      long double expected{}, magnitude{};
      const auto dot = [&](const auto& a, const auto& b, const int pitch) {
        for (int point = 0; point < pitch; ++point) {
          const auto term = 4.L * a[row * pitch + point] * b[column * pitch + point];
          expected += term;
          magnitude += std::abs(term);
        }
      };
      dot(left, right, rank);
      dot(left_tail, right_tail, tail);
      const auto relative = std::abs(actual[column * order + row] - expected) / magnitude;
      worst = std::max(worst, static_cast<double>(relative));
    }
  left[0] = 0x1p350;
  require(integer::project(blas.handle, workspace, storage.upload(left), storage.upload(right),
                           output, order, rank, true, status, nullptr));
  require(cudaMemcpy(&refused, status, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "{\"normalized_error\":" << worst << ",\"refused_extreme\":" << refused << "}\n";
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
  auto operation = read(owner.mode == "cached-seed-complete-diagonal" ? 0 : matrices * matrices);
  const bool metric_projection = owner.mode.starts_with("cached-metric-");
  const auto projected = metric_projection
                             ? read(shape.occupied[0] * (shape.nao - shape.occupied[0]))
                             : std::vector<double>{};
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
  response::CudaCpksOperatorCache cache;
  std::size_t cache_bytes{};
  const bool cached = owner.mode.starts_with("cached");
  if (cached && admitted == cudaSuccess) {
    require(response::cuda_cpks_operator_workspace(shape, options, cache_bytes));
    cache.reset(owner.mode == "cached-alias" ? workspace
                                             : storage.allocate(cache_bytes / sizeof(double) + 1),
                owner.mode == "cached-short" && cache_bytes ? cache_bytes - 1 : cache_bytes);
    if (owner.mode.starts_with("cached-seed")) {
      if (shape.spins != 1 || !shape.occupied[0] || shape.occupied[0] >= shape.nao)
        throw std::runtime_error("seed probes require an occupied and virtual RKS space");
      cache.set_build_decomposition(
          seed, owner.mode.starts_with("cached-seed-complete") ? nullptr : complement);
    }
  }
  auto* selected_workspace = owner.mode == "alias-workspace" ? density : workspace;
  auto* selected_weights = owner.mode == "alias-output" ? density : weights;
  int* upstream_status{};
  if (owner.mode.starts_with("upstream-")) {
    upstream_status = reinterpret_cast<int*>(storage.allocate(1));
    require(cudaMemsetAsync(upstream_status, 1, sizeof(int), stream));
  }
  if (reference_only && owner.mode == "alias-workspace") selected_workspace = device_coefficients;
  if (reference_only && owner.mode == "alias-output") selected_weights = device_coefficients;
  if (owner.mode == "misaligned-workspace")
    selected_workspace = reinterpret_cast<double*>(reinterpret_cast<char*>(workspace) + 1);
  if (owner.mode == "capture") require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  auto error = admitted != cudaSuccess ? admitted
               : reference_only
                   ? response::cuda_cpks_reference_weights(
                         shape, options, reference, selected_weights, selected_workspace,
                         owner.mode == "short-workspace" ? required - 1 : required,
                         owner.mode == "wrong-stream" ? nullptr : stream)
                   : response::solve_cuda_cpks(
                         shape, options, reference, device_frozen, device_overlap, induced, &owner,
                         density, selected_weights, selected_workspace,
                         owner.mode == "short-workspace" ? required - 1 : required,
                         owner.mode == "wrong-stream" ? nullptr : stream, result,
                         cached ? &cache : nullptr, upstream_status);
  if (owner.mode == "upstream-replay") {
    if (error != cudaSuccess || result.status != response::CudaCpksStatus::numerical_failure ||
        result.gmres.iterations)
      throw std::runtime_error("rejected upstream audit reached GMRES");
    require(cudaMemsetAsync(upstream_status, 0, sizeof(int), stream));
    error = response::solve_cuda_cpks(shape, options, reference, device_frozen, device_overlap,
                                      induced, &owner, density, weights, workspace, required,
                                      stream, result, nullptr, upstream_status);
  }
  if (owner.mode == "cached-nonlinear-replay") {
    if (error != cudaSuccess || result.status != response::CudaCpksStatus::linear_solve_failed ||
        !cache.physical_checks())
      throw std::runtime_error("invalid cached operator did not reach the physical residual gate");
    const auto rejected_columns = cache.columns();
    owner.mode = "cached-linear";
    error = response::solve_cuda_cpks(shape, options, reference, device_frozen, device_overlap,
                                      induced, &owner, density, weights, workspace, required,
                                      stream, result, &cache);
    if (error == cudaSuccess && result.converged() && cache.columns() <= rejected_columns)
      throw std::runtime_error("physical residual rejection left the cached operator authorized");
  }
  if ((owner.mode == "cached-reuse" || owner.mode.starts_with("cached-seed-complete")) &&
      error == cudaSuccess && result.converged())
    error = response::solve_cuda_cpks(shape, options, reference, device_frozen, device_overlap,
                                      induced, &owner, density, weights, workspace, required,
                                      stream, result, &cache);
  if (metric_projection && error == cudaSuccess && result.converged()) {
    const auto* projection_overlap =
        owner.mode == "cached-metric-other-overlap" ? storage.upload(overlap) : device_overlap;
    cache.set_metric_projection(projection_overlap, storage.upload(projected), complement);
    require(cudaMemcpyAsync(density, storage.upload(std::vector<double>(matrices, 9817.0)),
                            matrices * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    require(cudaMemcpyAsync(weights, storage.upload(std::vector<double>(matrices, -3277.0)),
                            matrices * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    error = response::solve_cuda_cpks(shape, options, reference, device_frozen, device_overlap,
                                      induced, &owner, density, weights, workspace, required,
                                      stream, result, &cache);
  }
  if (owner.mode == "cached-stale" && error == cudaSuccess && result.converged()) {
    auto replaced = reference;
    replaced.orbital_energies = storage.upload(energies);
    const std::vector<double> density_sentinel(matrices, 9817.0),
        weights_sentinel(matrices, -3277.0);
    require(cudaMemcpyAsync(density, density_sentinel.data(), matrices * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
    require(cudaMemcpyAsync(weights, weights_sentinel.data(), matrices * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
    require(cudaStreamSynchronize(stream));
    error = response::solve_cuda_cpks(shape, options, replaced, device_frozen, device_overlap,
                                      induced, &owner, density, weights, workspace, required,
                                      stream, result, &cache);
  }
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
            << ",\"cached_columns\":" << cache.columns()
            << ",\"cached_actions\":" << cache.actions()
            << ",\"physical_checks\":" << cache.physical_checks() << ",\"seeds\":" << owner.seeds
            << ",\"complements\":" << owner.complements
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
    else if (argc == 2 && std::string(argv[1]) == "integer-projection")
      integer_projection();
    else if (argc > 1 && (std::string(argv[1]) == "solve" || std::string(argv[1]) == "reference"))
      solve(argc, argv);
    else
      throw std::runtime_error("expected contract, integer-projection, solve or reference");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
