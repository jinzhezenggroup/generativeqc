/** Exercise real retained response with independently supplied metric factors.
 * No eigensolver or integral oracle inside this adapter constructs the expected
 * answer. Python supplies rotated full-rank metrics and checks both adjoints.
 */
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "scf/cuda/df_packed_values.hpp"
#include "scf/cuda/df_response_weights.cuh"

namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Storage {
  cudaStream_t stream{};
  cublasHandle_t blas{};
  std::vector<void*> allocations;
  ~Storage() {
    if (stream) cudaStreamSynchronize(stream);
    if (blas) cublasDestroy(blas);
    for (auto* pointer : allocations) cudaFree(pointer);
    if (stream) cudaStreamDestroy(stream);
  }
  double* allocate(std::size_t count, const double* host = nullptr) {
    double* pointer{};
    check(cudaMalloc(&pointer, count * sizeof(double)));
    allocations.push_back(pointer);
    if (host)
      check(cudaMemcpyAsync(pointer, host, count * sizeof(double), cudaMemcpyHostToDevice, stream));
    return pointer;
  }
};
}  // namespace

extern "C" int df_spectral_response_probe(std::size_t n, std::size_t a, std::size_t terms,
                                          std::size_t tile, std::size_t rank, unsigned mode,
                                          bool spectral, const double* forward,
                                          const double* densities, const double* coefficients,
                                          const double* coulomb, const double* exchange,
                                          const double* eigenvectors, const double* eigenvalues,
                                          const double* symmetric_root, const double* explicit_root,
                                          double* bar_a, double* bar_m, char* message,
                                          std::size_t message_size) noexcept {
  using namespace generativeqc;
  try {
    if (!n || !a || !terms || !tile || tile > a || n > 16 || a > 64 || terms > 3 || !rank ||
        rank > n || mode > 2 || (mode == 2 && terms != 1))
      throw std::invalid_argument("invalid spectral response probe shape");
    Storage owner;
    check(cudaStreamCreateWithFlags(&owner.stream, cudaStreamNonBlocking));
    if (cublasCreate(&owner.blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(owner.blas, owner.stream) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("response probe BLAS preparation failed");
    const auto matrix = n * n, pairs = n * (n + 1) / 2;
    const auto* device_density = owner.allocate(terms * matrix, densities);
    const auto* device_coefficients = owner.allocate(terms * n * rank, coefficients);
    const scf::CudaDfMetricView metric{owner.allocate(a * a, symmetric_root),
                                       owner.allocate(a * a, eigenvectors),
                                       owner.allocate(a, eigenvalues),
                                       1e-12,
                                       true,
                                       1};
    const scf::CudaDfWhitenedTensorView fitted{
        owner.allocate(pairs * a, forward),
        n,
        a,
        pairs,
        true,
        1,
        metric,
        spectral,
        explicit_root ? owner.allocate(a * a, explicit_root) : nullptr};
    std::vector<scf::DensityFittingDensityResponse> response;
    for (std::size_t t = 0; t < terms; ++t)
      response.push_back({{densities + t * matrix, matrix}, coulomb[t], exchange[t]});
    auto* workspace = owner.allocate(scf::cuda_df_response_workspace_elements(n, a, terms, tile));
    scf::CudaDfResponseBuffers occupied;
    if (mode) {
      occupied.occupied_response = true;
      occupied.fitted_occupied_source = &fitted;
      occupied.staging_elements = terms * a * rank * rank;
      occupied.exchange_elements = std::max(a * rank * rank, tile * (matrix + n * rank));
      occupied.raw_elements = matrix;
      occupied.staging_weights = owner.allocate(occupied.staging_elements);
      occupied.exchange_response = owner.allocate(occupied.exchange_elements);
      occupied.raw_auxiliary_major = owner.allocate(occupied.raw_elements);
      for (std::size_t t = 0; t < terms; ++t)
        occupied.occupied_factors[t] = {device_coefficients + t * n * rank, rank, 2.0};
      if (mode == 2) {
        auto* projection = owner.allocate(n * a * rank);
        scf::cuda_df::launch_project_packed_df(owner.stream, n, a, rank, true, fitted.data,
                                               device_coefficients, projection);
        check(cudaGetLastError());
        occupied.final_fitted_occupied_projection = projection;
      }
    }
    const auto status = scf::contract_cuda_df_response_weights(
        n, a, response, device_density, metric, tile, workspace, owner.stream, owner.blas, false,
        true,
        [](std::size_t, double*) {
          throw std::runtime_error("retained response reread raw values");
        },
        [&](unsigned kind, runtime::StridedRange range, std::size_t count, const double* weights) {
          if (kind == 0) {
            if (range.row_length != matrix || range.row_stride != 1 || range.column_stride != a)
              throw std::runtime_error("unexpected raw-adjoint layout");
            check(cudaMemcpyAsync(bar_a + range.offset * matrix, weights, count * sizeof(double),
                                  cudaMemcpyDeviceToHost, owner.stream));
          } else {
            if (kind != 1 || count != a * a) throw std::runtime_error("unexpected metric layout");
            check(cudaMemcpyAsync(bar_m, weights, count * sizeof(double), cudaMemcpyDeviceToHost,
                                  owner.stream));
          }
        },
        nullptr, {}, false, {}, 256, {}, false, mode ? &occupied : nullptr,
        mode ? nullptr : &fitted);
    check(cudaStreamSynchronize(owner.stream));
    return static_cast<int>(status);
  } catch (const std::exception& error) {
    if (message && message_size) std::snprintf(message, message_size, "%s", error.what());
    return -1;
  } catch (...) {
    return -2;
  }
}
