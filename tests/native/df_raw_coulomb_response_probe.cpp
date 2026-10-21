// Exercise the real device adjoint with externally supplied independent data.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

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

extern "C" int df_raw_coulomb_response_probe(std::size_t n, std::size_t a, std::size_t terms,
                                             std::size_t tile, const double* raw,
                                             const double* densities, const double* coefficients,
                                             const double* eigenvectors, const double* eigenvalues,
                                             bool full_rank, bool exchange, bool batched,
                                             double* bar_a, double* bar_m, char* message,
                                             std::size_t message_size) noexcept {
  using namespace generativeqc;
  try {
    if (!n || !a || !terms || !tile || tile > a || n > 16 || a > 64 || terms > 3)
      throw std::invalid_argument("invalid response probe shape");
    Storage owner;
    check(cudaStreamCreateWithFlags(&owner.stream, cudaStreamNonBlocking));
    if (cublasCreate(&owner.blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(owner.blas, owner.stream) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("response probe BLAS preparation failed");
    const auto matrix = n * n;
    const auto* device_raw = owner.allocate(a * matrix, raw);
    // The production panel ABI is pair-major, unlike the independent oracle's
    // auxiliary-major input. Keep both layouts immutable while checking ragged
    // reads; the callback must never borrow the response's live charge vectors.
    std::vector<double> pair_major(a * matrix);
    for (std::size_t ij = 0; ij < matrix; ++ij)
      for (std::size_t q = 0; q < a; ++q) pair_major[ij * a + q] = raw[q * matrix + ij];
    const auto* device_pair_raw = owner.allocate(a * matrix, pair_major.data());
    const auto* device_density = owner.allocate(terms * matrix, densities);
    const scf::CudaDfMetricView metric{nullptr,
                                       owner.allocate(a * a, eigenvectors),
                                       owner.allocate(a, eigenvalues),
                                       1e-12,
                                       full_rank,
                                       1};
    auto* workspace = owner.allocate(scf::cuda_df_response_workspace_elements(n, a, terms, tile));
    std::vector<scf::DensityFittingDensityResponse> response;
    for (std::size_t t = 0; t < terms; ++t)
      response.push_back({{densities + t * matrix, matrix}, coefficients[t], exchange ? .25 : 0.0});
    std::function<void(std::size_t, std::size_t, double*)> panel_reader;
    if (batched)
      panel_reader = [&](std::size_t begin, std::size_t count, double* output) {
        if (!count || count > tile || begin > a || count > a - begin)
          throw std::runtime_error("raw panel exceeds its admitted bounds");
        check(cudaMemcpy2DAsync(output, count * sizeof(double), device_pair_raw + begin,
                                a * sizeof(double), count * sizeof(double), matrix,
                                cudaMemcpyDeviceToDevice, owner.stream));
      };
    const auto error = scf::contract_cuda_df_response_weights(
        n, a, response, device_density, metric, tile, workspace, owner.stream, owner.blas, false,
        true,
        [&](std::size_t q, double* output) {
          check(cudaMemcpyAsync(output, device_raw + q * matrix, matrix * sizeof(double),
                                cudaMemcpyDeviceToDevice, owner.stream));
        },
        [&](unsigned kind, runtime::StridedRange range, std::size_t count, const double* weights) {
          // Native weights are auxiliary-major here; avoid an implementation
          // dependent scatter in the oracle-facing adapter.
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
        nullptr, {}, false, {}, 256, {}, false, nullptr, nullptr, true, panel_reader);
    check(cudaStreamSynchronize(owner.stream));
    return static_cast<int>(error);
  } catch (const std::exception& error) {
    if (message && message_size) std::snprintf(message, message_size, "%s", error.what());
    return -1;
  } catch (...) {
    return -2;
  }
}
