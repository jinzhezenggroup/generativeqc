#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "generated_integral_ad.cuh"
#include "scf/cuda/eigensolver.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "scf/explicit_integral_backend.hpp"
#include "tensor/cuda_matrix_product.hpp"

namespace generativeqc::scf {
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("explicit-integral cuBLAS failure");
}

struct Resources {
  int device;
  cudaStream_t stream{};
  cublasHandle_t blas{};
  double* data{};
  int* info{};
  std::uint8_t* active{};
  explicit Resources(int selected, std::size_t count) : device(selected) {
    try {
      check(cudaSetDevice(device));
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      check(cublasCreate(&blas));
      check(cublasSetStream(blas, stream));
      check(cudaMalloc(&data, count * sizeof(double)));
      check(cudaMalloc(&info, sizeof(int)));
      check(cudaMalloc(&active, 1));
      check(cudaMemset(active, 1, 1));
    } catch (...) {
      cleanup();
      throw;
    }
  }
  void cleanup() noexcept {
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (data) cudaFree(data);
    if (info) cudaFree(info);
    if (active) cudaFree(active);
    if (blas) cublasDestroy(blas);
    if (stream) cudaStreamDestroy(stream);
  }
  ~Resources() { cleanup(); }
};

class CudaExplicitIntegrals final : public ExplicitIntegralBackend {
  std::size_t n_, n2_;
  Resources resources_;
  double *eri_, *alpha_, *beta_, *j_, *ka_, *kb_, *matrix_, *x_, *temp_, *scratch_, *values_;
  std::unique_ptr<cuda_execution::OrdinaryStreamEigensolver> solver_;

 public:
  CudaExplicitIntegrals(int device, std::size_t n, const std::vector<double>& eri,
                        std::size_t budget)
      : n_(n), n2_(n * n), resources_(device, n * n * n * n + 9 * n * n + n) {
    auto* pointer = resources_.data;
    eri_ = pointer;
    pointer += n2_ * n2_;
    alpha_ = pointer;
    pointer += n2_;
    beta_ = pointer;
    pointer += n2_;
    j_ = pointer;
    pointer += n2_;
    ka_ = pointer;
    pointer += n2_;
    kb_ = pointer;
    pointer += n2_;
    matrix_ = pointer;
    pointer += n2_;
    x_ = pointer;
    pointer += n2_;
    temp_ = pointer;
    pointer += n2_;
    scratch_ = pointer;
    pointer += n2_;
    values_ = pointer;
    check(cudaMemcpy(eri_, eri.data(), eri.size() * sizeof(double), cudaMemcpyHostToDevice));
    solver_ = std::make_unique<cuda_execution::OrdinaryStreamEigensolver>(resources_.stream, int(n),
                                                                          matrix_, values_);
    if ((n2_ * n2_ + 9 * n2_ + n) * sizeof(double) + solver_->device_bytes() +
            solver_->host_bytes() + 5 >
        budget)
      throw std::invalid_argument("explicit CUDA eigensolver exceeds admitted numeric budget");
  }
  DirectJkMatrices build(const FockBuildSpec& spec, const std::vector<double>& da,
                         const std::vector<double>& db) override {
    const bool spin = spec.spin == FockSpin::Unrestricted;
    if (da.size() != n2_ || (spin && db.size() != n2_) || (!spin && !db.empty()) ||
        spec.derivative_order || spec.coulomb.op != FockOperator::FullRange ||
        spec.exchange.op != FockOperator::FullRange)
      throw std::invalid_argument("invalid explicit CUDA Fock inputs");
    for (const auto* density : {&da, &db})
      if (!std::all_of(density->begin(), density->end(), [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("nonfinite explicit CUDA density");
    check(cudaSetDevice(resources_.device));
    check(cudaMemcpy(alpha_, da.data(), n2_ * sizeof(double), cudaMemcpyHostToDevice));
    if (spin) check(cudaMemcpy(beta_, db.data(), n2_ * sizeof(double), cudaMemcpyHostToDevice));
    explicit_hf_contract<<<(n2_ + 127) / 128, 128, 0, resources_.stream>>>(n_, eri_, alpha_, beta_,
                                                                           j_, ka_, kb_, spin);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(resources_.stream));
    DirectJkMatrices result;
    result.nbf = n_;
    if (spec.coulomb.present) {
      result.coulomb.resize(n2_);
      check(cudaMemcpy(result.coulomb.data(), j_, n2_ * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (spec.exchange.present) {
      result.exchange_alpha.resize(n2_);
      check(cudaMemcpy(result.exchange_alpha.data(), ka_, n2_ * sizeof(double),
                       cudaMemcpyDeviceToHost));
      if (spin) {
        result.exchange_beta.resize(n2_);
        check(cudaMemcpy(result.exchange_beta.data(), kb_, n2_ * sizeof(double),
                         cudaMemcpyDeviceToHost));
      }
    }
    // Generated scalar rules reject intermediate overflow. A failed term or
    // overflowing reduction must fail before a partial Fock can reach SCF.
    for (const auto* values : {&result.coulomb, &result.exchange_alpha, &result.exchange_beta})
      if (!std::all_of(values->begin(), values->end(), [](double v) { return std::isfinite(v); }))
        throw std::runtime_error("nonfinite explicit CUDA Fock contraction");
    result.coulomb_exchange_work = {true, 1, spin ? 2u : 1u};
    return result;
  }
  reference::EigenResult eigen(const std::vector<double>& matrix,
                               const std::vector<double>* overlap, const std::vector<double>* x,
                               std::size_t n) override {
    if (n != n_ || matrix.size() != n2_ || (overlap == nullptr) != (x == nullptr) ||
        (x && (x->size() != n2_ || overlap->size() != n2_)))
      throw std::invalid_argument("invalid explicit CUDA eigen inputs");
    check(cudaSetDevice(resources_.device));
    check(cudaMemcpy(matrix_, matrix.data(), n2_ * sizeof(double), cudaMemcpyHostToDevice));
    const auto gemm = [&](bool transpose, const double* a, const double* b, double* c) {
      if (tensor::cuda::matrix_product(resources_.blas, 1, 1, int(n), a, false, transpose, b, false,
                                       c, 1.0) != GENERATIVEQC_STATUS_SUCCESS)
        throw std::runtime_error("explicit CUDA frame transform failed");
    };
    if (x) {
      check(cudaMemcpy(x_, x->data(), n2_ * sizeof(double), cudaMemcpyHostToDevice));
      gemm(false, matrix_, x_, temp_);
      gemm(true, x_, temp_, matrix_);
    }
    auto status =
        solver_->launch(1, matrix_, scratch_, values_, resources_.info, resources_.active);
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw std::runtime_error("explicit CUDA eigensolve failed");
    auto* output = matrix_;
    if (x) {
      gemm(false, x_, matrix_, temp_);
      output = temp_;
    }
    check(cudaStreamSynchronize(resources_.stream));
    int info;
    check(cudaMemcpy(&info, resources_.info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info) throw std::runtime_error("explicit CUDA eigensolve did not converge");
    reference::EigenResult result;
    result.values.resize(n);
    result.vectors.resize(n2_);
    check(cudaMemcpy(result.values.data(), values_, n * sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(result.vectors.data(), output, n2_ * sizeof(double), cudaMemcpyDeviceToHost));
    result.vectors = reference::transpose(result.vectors, n);
    return result;
  }
};
}  // namespace

std::unique_ptr<ExplicitIntegralBackend> make_cuda_explicit_integrals(
    int device, std::size_t n, const std::vector<double>& eri, std::size_t budget) {
  if (device < 0 || !n || n > 256 || eri.size() != n * n * n * n)
    throw std::invalid_argument("invalid explicit CUDA integral shape/device");
  // Bound all explicit arrays and the shared solver's workspace allowance before
  // acquiring device memory. Opaque CUDA/cuSOLVER bookkeeping is excluded.
  const auto required = (n * n * n * n + 9 * n * n + n) * sizeof(double) + 5 +
                        2 * ordinary_eigensolver_workspace_allowance(n);
  if (required > budget) throw std::invalid_argument("explicit CUDA integrals exceed max_bytes");
  return std::make_unique<CudaExplicitIntegrals>(device, n, eri, budget);
}
}  // namespace generativeqc::scf
