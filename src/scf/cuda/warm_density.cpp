#include "scf/cuda/warm_density.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "generativeqc/generativeqc.hpp"
#include "molecule/basis.hpp"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/eigensolver.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "scf/mean_field.hpp"
#include "scf/solver/eigen_frame.hpp"

namespace generativeqc::scf {
namespace {
void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}
}  // namespace

void validate_hf_warm_density_cuda(const core::System& source, generativeqc_method method,
                                   const std::vector<double>& density, int device) {
  const auto n = molecule::ao_count(source);
  runtime::host_trace::Region trace("cuda_hf_warm_density_validation", n);
  // Check all products before allocation. The provider separately admits its
  // queried workspace against the shared shape-only ordinary-solver bound.
  (void)ordinary_eigensolver_workspace_allowance(n);
  if (n > std::size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("CUDA HF seed dimension exceeds solver capacity");
  const auto matrix_size = n * n;
  int previous{};
  check(cudaGetDevice(&previous));
  check(cudaSetDevice(device));
  cudaStream_t stream{};
  double* storage{};
  std::unique_ptr<cuda_execution::OrdinaryStreamEigensolver> solver;
  auto cleanup = [&]() noexcept {
    // Drain before either host frames or device storage are released, including
    // exceptional copies/launches. No retained fleet buffer is borrowed here.
    if (stream) (void)cudaStreamSynchronize(stream);
    solver.reset();
    if (storage) (void)runtime::resource_cuda_free(storage);
    if (stream) (void)cudaStreamDestroy(stream);
    (void)cudaSetDevice(previous);
  };
  runtime::ResourceScopeExit guard(cleanup);
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto bytes = (2 * matrix_size + n + 2) * sizeof(double);
  check(runtime::resource_cuda_malloc(&storage, bytes));
  auto* matrix = storage;
  auto* scratch = matrix + matrix_size;
  auto* values = scratch + matrix_size;
  auto* info_device = reinterpret_cast<int*>(values + n);
  auto* active = reinterpret_cast<std::uint8_t*>(values + n + 1);
  check(cudaMemsetAsync(active, 1, 1, stream));
  solver =
      std::make_unique<cuda_execution::OrdinaryStreamEigensolver>(stream, int(n), matrix, values);
  runtime::CpuRetainedCapacity retained(solver->host_bytes() + solver->metadata_bytes());
  runtime::sample_cuda_arena_capacity(bytes + solver->device_bytes());
  const initial_guess::EigenOperation eigen = [&](const auto& input, const auto* overlap,
                                                  const auto* x, std::size_t dimension) {
    if (overlap || x || dimension != n || input.size() != matrix_size)
      throw std::invalid_argument("CUDA HF seed validation requires a matching symmetric solve");
    reference::EigenResult frame;
    frame.values.resize(n);
    frame.vectors.resize(matrix_size);
    int info{};
    try {
      check(cudaMemcpyAsync(matrix, input.data(), matrix_size * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      const auto status = solver->launch(1, matrix, scratch, values, info_device, active);
      if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
      if (status != GENERATIVEQC_STATUS_SUCCESS)
        throw generativeqc::Error(status, "CUDA HF warm density eigensolver launch failed");
      check(cudaMemcpyAsync(frame.vectors.data(), matrix, matrix_size * sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaMemcpyAsync(frame.values.data(), values, n * sizeof(double), cudaMemcpyDeviceToHost,
                            stream));
      check(cudaMemcpyAsync(&info, info_device, sizeof(info), cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      throw;
    }
    if (info != 0)
      throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                                "CUDA HF warm density eigensolver did not converge");
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t column = row + 1; column < n; ++column)
        std::swap(frame.vectors[row * n + column], frame.vectors[column * n + row]);
    (void)solver::order_eigenpairs(frame.values, frame.vectors, n);
    solver::EigenFrameDiagnostic diagnostic;
    std::string detail;
    if (!solver::validate_eigen_frame(input, nullptr, frame.values, frame.vectors, n, diagnostic,
                                      detail))
      throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, detail);
    return frame;
  };
  validate_hf_warm_density(source, method, density, eigen);
}
}  // namespace generativeqc::scf
