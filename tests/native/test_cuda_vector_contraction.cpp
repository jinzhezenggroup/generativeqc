#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_df_coulomb_lowering.hpp"

namespace {
using generativeqc::tensor::CudaVectorContraction;
namespace lowering = generativeqc::scf::cuda_df::coulomb_lowering;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
void blas_check(cublasStatus_t status) { require(status == CUBLAS_STATUS_SUCCESS, "BLAS failure"); }
struct Owner {
  cudaStream_t stream{};
  cublasHandle_t handle{};
  Owner() {
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    blas_check(cublasCreate(&handle));
    blas_check(cublasSetStream(handle, stream));
    blas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  }
  ~Owner() {
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
  }
};
struct Buffer {
  double* data{};
  std::size_t count{};
  explicit Buffer(std::size_t n) : count(n) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(double)));
  }
  ~Buffer() { cudaFree(data); }
  void upload(const std::vector<double>& values, cudaStream_t stream) {
    require(values.size() == count, "upload extent");
    cuda_check(cudaMemcpyAsync(data, values.data(), count * sizeof(double), cudaMemcpyHostToDevice,
                               stream));
  }
  std::vector<double> read(cudaStream_t stream) {
    std::vector<double> values(count);
    cuda_check(cudaMemcpyAsync(values.data(), data, count * sizeof(double), cudaMemcpyDeviceToHost,
                               stream));
    cuda_check(cudaStreamSynchronize(stream));
    return values;
  }
};
template <class F>
void rejects(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "invalid binding was admitted");
}
std::unique_ptr<CudaVectorContraction> alternative(const CudaVectorContraction& binding,
                                                   Owner& owner, std::size_t incumbent) {
  const auto& d = binding.diagnostic();
  return std::make_unique<CudaVectorContraction>(
      d.request, d.candidates, d.candidates[0].target_identity,
      d.candidates[0].compilation_identity, binding.resolved(), incumbent, owner.handle,
      owner.stream);
}
void run(bool packed, std::size_t batches, std::size_t nbf, std::size_t naux, bool alternate) {
  Owner owner;
  auto charge = packed ? lowering::packed_charge(batches, nbf, naux, owner.handle, owner.stream)
                       : lowering::dense_charge(batches, nbf, naux, owner.handle, owner.stream);
  auto coulomb = packed ? lowering::packed_coulomb(batches, nbf, naux, owner.handle, owner.stream)
                        : lowering::dense_coulomb(batches, nbf, naux, owner.handle, owner.stream);
  if (alternate) {
    charge = alternative(*charge, owner, packed ? 1 : 0);
    coulomb = alternative(*coulomb, owner, packed ? 1 : 0);
  }
  const auto& diagnostic = charge->diagnostic();
  require(diagnostic.provider_version > 0 && diagnostic.runtime_version > 0 &&
              diagnostic.prepare_ns > 0 && diagnostic.retained_incumbent &&
              !diagnostic.borrowed_provider_bytes_known,
          "incomplete provenance");
  for (const auto& candidate : diagnostic.candidates) {
    require(candidate.request_identity == diagnostic.request.identity &&
                candidate.precision_identity == charge->resolved().precision_identity &&
                candidate.workspace_bytes == 0 &&
                candidate.host_bytes == sizeof(CudaVectorContraction),
            "canonical identity/resource mismatch");
  }
  if (batches > 1) {
    require(!diagnostic.candidates[0].rejection.empty(), "batched GEMV must be rejected");
    rejects([&] { alternative(*charge, owner, 0); });
  }
  blas_check(cublasSetPointerMode(owner.handle, CUBLAS_POINTER_MODE_DEVICE));
  rejects([&] { alternative(*charge, owner, diagnostic.selected); });
  blas_check(cublasSetPointerMode(owner.handle, CUBLAS_POINTER_MODE_HOST));
  const auto pairs = packed ? nbf * (nbf + 1) / 2 : nbf * nbf;
  Buffer factor(batches * pairs * naux), density(batches * pairs), rho(batches * naux),
      j(batches * pairs);
  require(charge->launch(factor.data, density.data, density.data + (pairs > 1 ? 1 : 0)) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "partial output alias accepted");
  require(charge->launch(factor.data, density.data, factor.data) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "matrix output alias accepted");
  require(charge->launch(nullptr, density.data, rho.data) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null input accepted");
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  for (int repeat = 0; repeat < 3; ++repeat) {
    std::vector<double> b(factor.count), d(density.count);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.13 * (i + 1) + repeat) * 0.125;
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = std::cos(0.17 * (i + 1) - repeat);
    factor.upload(b, owner.stream);
    density.upload(d, owner.stream);
    cuda_check(cudaStreamSynchronize(owner.stream));
    if (repeat == 0) {
      // Preparation must reject before it can invalidate an enclosing capture.
      cuda_check(cudaStreamBeginCapture(owner.stream, cudaStreamCaptureModeThreadLocal));
      rejects([&] { alternative(*charge, owner, diagnostic.selected); });
      require(charge->launch(factor.data, density.data, rho.data) == GENERATIVEQC_STATUS_SUCCESS,
              "charge launch");
      require(coulomb->launch(factor.data, rho.data, j.data) == GENERATIVEQC_STATUS_SUCCESS,
              "Coulomb launch");
      cuda_check(cudaStreamEndCapture(owner.stream, &graph));
      cuda_check(cudaGraphInstantiate(&executable, graph, 0));
    }
    cuda_check(cudaGraphLaunch(executable, owner.stream));
    const auto actual_rho = rho.read(owner.stream), actual_j = j.read(owner.stream);
    // Independent scalar long-double oracle, with no shared transpose recipe.
    for (std::size_t system = 0; system < batches; ++system) {
      std::vector<long double> expected_rho(naux);
      for (std::size_t q = 0; q < naux; ++q) {
        for (std::size_t p = 0; p < pairs; ++p)
          expected_rho[q] +=
              static_cast<long double>(b[(system * pairs + p) * naux + q]) * d[system * pairs + p];
        require(std::abs(actual_rho[system * naux + q] - expected_rho[q]) <
                    2e-12L * (1 + std::abs(expected_rho[q])),
                "charge oracle mismatch");
      }
      for (std::size_t p = 0; p < pairs; ++p) {
        long double expected{};
        for (std::size_t q = 0; q < naux; ++q)
          expected +=
              static_cast<long double>(b[(system * pairs + p) * naux + q]) * expected_rho[q];
        require(
            std::abs(actual_j[system * pairs + p] - expected) < 2e-12L * (1 + std::abs(expected)),
            "Coulomb oracle mismatch");
      }
    }
  }
  cuda_check(cudaGraphExecDestroy(executable));
  cuda_check(cudaGraphDestroy(graph));
  std::cout << "packed=" << packed << " batch=" << batches << " nbf=" << nbf << " naux=" << naux
            << " algorithm=" << diagnostic.candidates[diagnostic.selected].algorithm
            << " cublas=" << diagnostic.provider_version
            << " host_bytes=" << sizeof(CudaVectorContraction)
            << " prepare_ns=" << diagnostic.prepare_ns << '\n';
}
// Exercise the physical full/tail metric recipes independently of the resident
// charge/J tests. NaN padding exposes accidental dense packing or wrong strides.
void run_metric(std::size_t operation, std::size_t naux, std::size_t panel, bool alternate) {
  Owner owner;
  const auto factories = std::array{lowering::metric_charge, lowering::metric_potential,
                                    lowering::metric_project, lowering::metric_rotate};
  auto binding = factories[operation](naux, panel, owner.handle, owner.stream);
  if (alternate) binding = alternative(*binding, owner, 1);
  const auto& diagnostic = binding->diagnostic();
  const bool accumulate = operation == 0;
  require(diagnostic.request.inputs == (accumulate ? 3U : 2U), "missing explicit seed read");
  auto invalid = binding->resolved();
  invalid.beta = accumulate ? 0.0 : 1.0;
  rejects([&] {
    CudaVectorContraction forged(diagnostic.request, diagnostic.candidates,
                                 diagnostic.candidates[0].target_identity,
                                 diagnostic.candidates[0].compilation_identity, invalid,
                                 diagnostic.selected, owner.handle, owner.stream);
  });
  const std::size_t outputs = operation == 1 ? panel : naux;
  const std::size_t summands = operation == 0 ? panel : naux;
  Buffer matrix(naux * naux), vector(summands), output(outputs);
  std::vector<long double> expected(outputs);
  std::vector<double> seed(outputs);
  for (std::size_t i = 0; i < outputs; ++i) seed[i] = expected[i] = 0.17 * (i + 1);
  output.upload(seed, owner.stream);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  for (int repeat = 0; repeat < 3; ++repeat) {
    std::vector<double> a(matrix.count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> x(summands);
    for (std::size_t i = 0; i < summands; ++i) x[i] = std::cos(0.13 * (i + 1) + repeat);
    for (std::size_t out = 0; out < outputs; ++out)
      for (std::size_t inner = 0; inner < summands; ++inner) {
        const auto index = operation == 2 ? out * naux + inner : inner * naux + out;
        a[index] = std::sin(0.07 * (index + 1) - repeat);
      }
    matrix.upload(a, owner.stream);
    vector.upload(x, owner.stream);
    cuda_check(cudaStreamSynchronize(owner.stream));
    if (repeat == 0) {
      cuda_check(cudaStreamBeginCapture(owner.stream, cudaStreamCaptureModeThreadLocal));
      require(binding->launch(matrix.data, vector.data, output.data) == GENERATIVEQC_STATUS_SUCCESS,
              "metric capture submission");
      cuda_check(cudaStreamEndCapture(owner.stream, &graph));
      cuda_check(cudaGraphInstantiate(&executable, graph, 0));
    }
    cuda_check(cudaGraphLaunch(executable, owner.stream));
    const auto actual = output.read(owner.stream);
    for (std::size_t out = 0; out < outputs; ++out) {
      if (!accumulate) expected[out] = 0;
      for (std::size_t inner = 0; inner < summands; ++inner) {
        const auto index = operation == 2 ? out * naux + inner : inner * naux + out;
        expected[out] += static_cast<long double>(a[index]) * x[inner];
      }
      require(std::isfinite(actual[out]) &&
                  std::abs(actual[out] - expected[out]) < 2e-12L * (1 + std::abs(expected[out])),
              "metric scalar oracle or seed accumulation mismatch");
    }
  }
  cuda_check(cudaGraphExecDestroy(executable));
  cuda_check(cudaGraphDestroy(graph));
  std::cout << "metric=" << operation << " naux=" << naux << " panel=" << panel
            << " alternate=" << alternate << " inputs=" << diagnostic.request.inputs << '\n';
}
}  // namespace
int main() {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  try {
    for (const auto packed : {false, true}) {
      run(packed, 1, 1, 1, false);
      run(packed, 1, 7, 13, false);
      run(packed, 1, 7, 13, true);
      run(packed, 1, 24, 73, false);
    }
    run(false, 3, 7, 13, false);
    run(false, 2, 24, 73, false);
    for (std::size_t operation = 0; operation != 4; ++operation)
      for (const auto alternate : {false, true}) {
        run_metric(operation, 1, 1, alternate);
        run_metric(operation, 7, 3, alternate);
        run_metric(operation, 13, 8, alternate);
        run_metric(operation, 32, 7, alternate);
      }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
