// Host ABI/call trace, not a numerical or real-device qualification.
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "generativeqc/generativeqc.h"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda/launch_geometry.hpp"
#include "solver/cuda/symmetric_eigen_provider.hpp"

namespace shared = generativeqc::solver::cuda;
std::uint32_t private_eigen_query(const shared::SymmetricEigenResources&,
                                  shared::SymmetricEigenFamily,
                                  const shared::SymmetricEigenProblem&, const double*,
                                  const double*, shared::SymmetricEigenWorkspace&);
std::uint32_t private_eigen_launch(const shared::SymmetricEigenResources&,
                                   shared::SymmetricEigenFamily,
                                   const shared::SymmetricEigenProblem&, double*, double*, int*,
                                   int);

namespace {
enum class Operation {
  query_jacobi,
  launch_jacobi,
  query_batched,
  launch_batched,
  query_serial,
  launch_serial
};
struct Call {
  Operation operation;
  cusolverDnHandle_t solver;
  cusolverDnParams_t parameters;
  syevjInfo_t jacobi;
  cusolverEigMode_t vectors;
  cublasFillMode_t triangle;
  std::int64_t n, lda, batch;
  const void* matrix;
  const void* values;
  cudaDataType matrix_type = CUDA_R_64F;
  cudaDataType values_type = CUDA_R_64F;
  cudaDataType compute_type = CUDA_R_64F;
  void* device = nullptr;
  std::size_t device_bytes = 0;
  void* host = nullptr;
  std::size_t host_bytes = 0;
  int* info = nullptr;
  int lwork = 0;
  std::int64_t matrix_stride = 0, values_stride = 0;
};
std::vector<Call> calls;
std::vector<std::size_t> query_device, query_host;
std::vector<int> query_elements;
std::size_t fail_at = 0;
cusolverStatus_t failure = CUSOLVER_STATUS_SUCCESS;
constexpr int queried_elements = 37;
constexpr std::size_t queried_device_bytes = 513, queried_host_bytes = 79;

cusolverStatus_t result() {
  return calls.size() - 1 == fail_at ? failure : CUSOLVER_STATUS_SUCCESS;
}
void write_sizes(std::size_t* device, std::size_t* host) {
  assert(device != nullptr && host != nullptr && device != host);
  *device = query_device.empty() ? queried_device_bytes : query_device.at(calls.size() - 1);
  *host = query_host.empty() ? queried_host_bytes : query_host.at(calls.size() - 1);
}

struct Fixture {
  // Real storage backs even opaque tokens so no integer-to-pointer assumptions
  // enter the production lowering or serialized pointer arithmetic.
  int solver_token = 0, params_token = 0, jacobi_token = 0;
  std::array<double, 75> matrices{};
  std::array<double, 15> values{};
  std::array<double, 256> device{};
  std::array<unsigned char, 192> host{};
  std::array<int, 3> info{{-71, -72, -73}};
  shared::SymmetricEigenResources resources{
      &solver_token, &params_token, &jacobi_token, device.data(), 1025, host.data(), 131};
  shared::SymmetricEigenProblem problem{5, 3, shared::Eigenvectors::values_and_vectors};
  static constexpr int launch_elements = 23;
  void reset(cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS, std::size_t index = 0) {
    calls.clear();
    query_device.clear();
    query_host.clear();
    query_elements.clear();
    failure = status;
    fail_at = index;
  }
  void check(const Call& call, shared::SymmetricEigenFamily family, bool query,
             std::size_t serial_index = 0, int expected_elements = launch_elements) const {
    const bool jacobi = family == shared::SymmetricEigenFamily::jacobi_batched;
    const bool serial = family == shared::SymmetricEigenFamily::xsyevd;
    const auto expected_operation =
        jacobi   ? (query ? Operation::query_jacobi : Operation::launch_jacobi)
        : serial ? (query ? Operation::query_serial : Operation::launch_serial)
                 : (query ? Operation::query_batched : Operation::launch_batched);
    assert(call.operation == expected_operation);
    assert(call.solver == reinterpret_cast<cusolverDnHandle_t>(resources.solver));
    assert(call.parameters ==
           (jacobi ? nullptr : reinterpret_cast<cusolverDnParams_t>(resources.parameters)));
    assert(call.jacobi == (jacobi ? reinterpret_cast<syevjInfo_t>(resources.jacobi) : nullptr));
    assert(call.vectors == (problem.vectors == shared::Eigenvectors::values_and_vectors
                                ? CUSOLVER_EIG_MODE_VECTOR
                                : CUSOLVER_EIG_MODE_NOVECTOR));
    assert(call.triangle == CUBLAS_FILL_MODE_LOWER);
    assert(call.n == problem.n && call.lda == problem.n);
    assert(call.batch == (serial ? 1 : problem.batch));
    const auto offset = serial && !query ? serial_index : 0;
    assert(call.matrix == matrices.data() + offset * problem.n * problem.n);
    assert(call.values == values.data() + offset * problem.n);
    assert(call.matrix_type == CUDA_R_64F && call.values_type == CUDA_R_64F &&
           call.compute_type == CUDA_R_64F);
#if defined(TEST_CUMETAL_STRIDED)
    if (!jacobi && !serial) {
      assert(call.matrix_stride == std::int64_t(problem.n) * problem.n);
      assert(call.values_stride == problem.n);
    }
#endif
    if (!query) {
      assert(call.info == info.data() + offset);
      assert(call.device == resources.device_workspace);
      if (jacobi) {
        assert(call.lwork == expected_elements);
        assert(call.host == nullptr && call.host_bytes == 0);
      } else {
        assert(call.device_bytes == resources.device_workspace_bytes);
        assert(call.host == resources.host_workspace &&
               call.host_bytes == resources.host_workspace_bytes);
      }
    }
  }
};

void test_private_abi(shared::SymmetricEigenFamily family) {
  Fixture fixture;
  for (auto status : {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_ALLOC_FAILED,
                      CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_NOT_SUPPORTED}) {
    fixture.reset(status);
    shared::SymmetricEigenWorkspace workspace;
    assert(private_eigen_query(fixture.resources, family, fixture.problem, fixture.matrices.data(),
                               fixture.values.data(),
                               workspace) == static_cast<std::uint32_t>(status));
    assert(calls.size() == 1);
    fixture.check(calls.front(), family, true);
    fixture.reset(status);
    assert(private_eigen_launch(fixture.resources, family, fixture.problem, fixture.matrices.data(),
                                fixture.values.data(), fixture.info.data(),
                                Fixture::launch_elements) == static_cast<std::uint32_t>(status));
    const auto count =
        family == shared::SymmetricEigenFamily::xsyevd && status == CUSOLVER_STATUS_SUCCESS ? 3U
                                                                                            : 1U;
    assert(calls.size() == count);
    for (std::size_t i = 0; i < calls.size(); ++i) fixture.check(calls[i], family, false, i);
  }
}

void test_query(shared::SymmetricEigenFamily family, shared::Eigenvectors vectors) {
  Fixture fixture;
  fixture.problem.vectors = vectors;
  for (auto status : {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_NOT_INITIALIZED,
                      CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_EXECUTION_FAILED,
                      CUSOLVER_STATUS_INTERNAL_ERROR, CUSOLVER_STATUS_NOT_SUPPORTED}) {
    fixture.reset(status);
    shared::SymmetricEigenWorkspace workspace{9991, 9993, 9997};
    const auto returned =
        shared::query_symmetric_eigen(fixture.resources, family, fixture.problem,
                                      fixture.matrices.data(), fixture.values.data(), workspace);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(returned)>, std::uint32_t>);
    assert(returned == static_cast<std::uint32_t>(status));
    assert(calls.size() == 1);  // No capacity sweep or per-matrix sizing belongs here.
    fixture.check(calls[0], family, true);
    if (status == CUSOLVER_STATUS_SUCCESS) {
      const bool jacobi = family == shared::SymmetricEigenFamily::jacobi_batched;
      assert(workspace.device_bytes == (jacobi ? 9991 : queried_device_bytes));
      assert(workspace.host_bytes == (jacobi ? 9993 : queried_host_bytes));
      assert(workspace.jacobi_elements == (jacobi ? queried_elements : 9997));
    }
  }
}

void test_launch(shared::SymmetricEigenFamily family, shared::Eigenvectors vectors) {
  Fixture fixture;
  fixture.problem.vectors = vectors;
  const bool serial = family == shared::SymmetricEigenFamily::xsyevd;
  for (auto status : {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_ALLOC_FAILED,
                      CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_EXECUTION_FAILED,
                      CUSOLVER_STATUS_INTERNAL_ERROR, CUSOLVER_STATUS_NOT_SUPPORTED}) {
    // Inject at every serialized index, not just the first call.
    for (std::size_t index = 0; index != (serial ? 3U : 1U); ++index) {
      fixture.reset(status, index);
      const auto returned = shared::launch_symmetric_eigen(
          fixture.resources, family, fixture.problem, fixture.matrices.data(),
          fixture.values.data(), fixture.info.data(), Fixture::launch_elements);
      static_assert(std::is_same_v<std::remove_cv_t<decltype(returned)>, std::uint32_t>);
      assert(returned == static_cast<std::uint32_t>(status));
      const auto count = serial ? (status == CUSOLVER_STATUS_SUCCESS ? 3U : index + 1) : 1U;
      assert(calls.size() == count);
      for (std::size_t i = 0; i != count; ++i) fixture.check(calls[i], family, false, i);
      // Lowering forwards buffers and device info; it neither diagnoses nor
      // synchronizes, clears, sanitizes, allocates, or overwrites client memory.
      assert((fixture.info == std::array<int, 3>{{-71, -72, -73}}));
      for (auto value : fixture.matrices) assert(value == 0.0);
      for (auto value : fixture.values) assert(value == 0.0);
    }
  }
}
}  // namespace

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t solver,
                                                    cusolverEigMode_t mode,
                                                    cublasFillMode_t triangle, int n,
                                                    const double* matrix, int lda,
                                                    const double* values, int* lwork,
                                                    syevjInfo_t jacobi, int batch) {
  calls.push_back({Operation::query_jacobi, solver, nullptr, jacobi, mode, triangle, n, lda, batch,
                   matrix, values});
  assert(lwork != nullptr);
  *lwork = query_elements.empty() ? queried_elements : query_elements.at(calls.size() - 1);
  return result();
}
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t solver, cusolverEigMode_t mode,
                                         cublasFillMode_t triangle, int n, double* matrix, int lda,
                                         double* values, double* device, int lwork, int* info,
                                         syevjInfo_t jacobi, int batch) {
  calls.push_back({Operation::launch_jacobi, solver, nullptr, jacobi, mode, triangle, n, lda, batch,
                   matrix, values});
  auto& call = calls.back();
  call.device = device;
  call.lwork = lwork;
  call.info = info;
  return result();
}
cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t solver, cusolverDnParams_t params,
                                             cusolverEigMode_t mode, cublasFillMode_t triangle,
                                             std::int64_t n, cudaDataType a_type,
                                             const void* matrix, std::int64_t lda,
                                             cudaDataType w_type, const void* values,
                                             cudaDataType compute_type, std::size_t* device_bytes,
                                             std::size_t* host_bytes) {
  calls.push_back({Operation::query_serial, solver, params, nullptr, mode, triangle, n, lda, 1,
                   matrix, values, a_type, w_type, compute_type});
  write_sizes(device_bytes, host_bytes);
  return result();
}
cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t solver, cusolverDnParams_t params,
                                  cusolverEigMode_t mode, cublasFillMode_t triangle, std::int64_t n,
                                  cudaDataType a_type, void* matrix, std::int64_t lda,
                                  cudaDataType w_type, void* values, cudaDataType compute_type,
                                  void* device, std::size_t device_bytes, void* host,
                                  std::size_t host_bytes, int* info) {
  calls.push_back({Operation::launch_serial, solver, params, nullptr, mode, triangle, n, lda, 1,
                   matrix, values, a_type, w_type, compute_type, device, device_bytes, host,
                   host_bytes, info});
  return result();
}

cusolverStatus_t cusolverDnXsyevBatched_bufferSize(
    cusolverDnHandle_t solver, cusolverDnParams_t params, cusolverEigMode_t mode,
    cublasFillMode_t triangle, std::int64_t n, cudaDataType a_type, const void* matrix,
    std::int64_t lda,
#if defined(TEST_CUMETAL_STRIDED)
    std::int64_t matrix_stride,
#endif
    cudaDataType w_type, const void* values,
#if defined(TEST_CUMETAL_STRIDED)
    std::int64_t values_stride, cudaDataType compute_type, std::int64_t batch,
    std::size_t* device_bytes, std::size_t* host_bytes) {
#else
    cudaDataType compute_type, std::size_t* device_bytes, std::size_t* host_bytes,
    std::int64_t batch) {
#endif
  calls.push_back({Operation::query_batched, solver, params, nullptr, mode, triangle, n, lda, batch,
                   matrix, values, a_type, w_type, compute_type});
#if defined(TEST_CUMETAL_STRIDED)
  calls.back().matrix_stride = matrix_stride;
  calls.back().values_stride = values_stride;
#endif
  write_sizes(device_bytes, host_bytes);
  return result();
}
cusolverStatus_t cusolverDnXsyevBatched(
    cusolverDnHandle_t solver, cusolverDnParams_t params, cusolverEigMode_t mode,
    cublasFillMode_t triangle, std::int64_t n, cudaDataType a_type, void* matrix, std::int64_t lda,
#if defined(TEST_CUMETAL_STRIDED)
    std::int64_t matrix_stride,
#endif
    cudaDataType w_type, void* values,
#if defined(TEST_CUMETAL_STRIDED)
    std::int64_t values_stride, cudaDataType compute_type, std::int64_t batch,
#else
    cudaDataType compute_type,
#endif
    void* device, std::size_t device_bytes, void* host, std::size_t host_bytes, int* info
#if !defined(TEST_CUMETAL_STRIDED)
    ,
    std::int64_t batch
#endif
) {
  calls.push_back({Operation::launch_batched, solver, params, nullptr, mode, triangle, n, lda,
                   batch, matrix, values, a_type, w_type, compute_type, device, device_bytes, host,
                   host_bytes, info});
#if defined(TEST_CUMETAL_STRIDED)
  calls.back().matrix_stride = matrix_stride;
  calls.back().values_stride = values_stride;
#endif
  return result();
}

// Other CUDA work is replaced only at its launch boundary. The functions and
// types in this generated include are extracted verbatim from production.
struct cudaStream;
using cudaStream_t = cudaStream*;
enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorUnknown = 999
};
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
using namespace generativeqc::scf::cuda_execution;
std::vector<std::pair<std::string, std::size_t>> stages;
cudaError_t kernel_error = cudaSuccess;
std::size_t kernel_fail_at = std::numeric_limits<std::size_t>::max();
cudaError_t cudaPeekAtLastError() {
  return stages.size() - 1 == kernel_fail_at ? kernel_error : cudaSuccess;
}
#define TRACE_KERNEL(name, label)             \
  template <class... Args>                    \
  void name(Args...) {                        \
    stages.emplace_back(label, calls.size()); \
  }
TRACE_KERNEL(launch_begin_inactive_eigensolver_profile_kernel, "begin")
TRACE_KERNEL(launch_sanitize_inactive_solver_input_kernel, "sanitize")
TRACE_KERNEL(launch_start_inactive_eigensolver_timer_kernel, "start")
TRACE_KERNEL(launch_finish_inactive_eigensolver_profile_kernel, "finish")
TRACE_KERNEL(launch_symmetric_eigen_small_kernel, "small")
TRACE_KERNEL(launch_symmetric_eigen_graph_maximum_pivot_kernel, "graph")
#undef TRACE_KERNEL
// DF's solver owner type and adapter are real; these independent tracing and
// destruction services do no CUDA work in this host harness.
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t) { return CUSOLVER_STATUS_SUCCESS; }
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t) { return CUSOLVER_STATUS_SUCCESS; }
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t) { return CUSOLVER_STATUS_SUCCESS; }
std::vector<std::pair<std::string, std::uint64_t>> counters;
std::string provider_label;
namespace runtime {
cudaError_t resource_cuda_free(void*) { return cudaSuccess; }
namespace cuda_trace {
struct TraceShape {
  std::size_t systems, nbf, naux;
  bool source_backed, streamed;
};
struct TraceOperation {
  TraceOperation(const char*, cudaStream_t, TraceShape) {
    stages.emplace_back("df-trace", calls.size());
  }
};
void trace_counter(const char* name, std::uint64_t value) { counters.emplace_back(name, value); }
}  // namespace cuda_trace
namespace df_progress {
void label(const char*, const char* value) { provider_label = value; }
}  // namespace df_progress
namespace host_trace {
struct Region {
  Region(const char*, std::size_t) { stages.emplace_back("df-provider", calls.size()); }
};
}  // namespace host_trace
}  // namespace runtime
struct CudaDensityFittingJkPlan {
  cudaStream_t stream{};
  std::size_t naux{7};
  void* integral_source{};
  bool streamed{};
};
#include "shared_eigen_consumers.inc"

static_assert(GENERATIVEQC_ABI_VERSION == 0);
static_assert(sizeof(generativeqc_status) == 4);
static_assert(GENERATIVEQC_STATUS_SUCCESS == 0 && GENERATIVEQC_STATUS_INVALID_ARGUMENT == 1);
static_assert(GENERATIVEQC_STATUS_CUDA_ERROR == 6 && GENERATIVEQC_STATUS_OUT_OF_MEMORY == 7);
static_assert(sizeof(gfn2::Gfn2EigensolverLaunchStatus) == 4);
static_assert(static_cast<std::uint32_t>(gfn2::Gfn2EigensolverLaunchStatus::kSuccess) == 0);
static_assert(static_cast<std::uint32_t>(gfn2::Gfn2EigensolverLaunchStatus::kInvalidArgument) == 1);
static_assert(static_cast<std::uint32_t>(gfn2::Gfn2EigensolverLaunchStatus::kCudaError) == 2);
static_assert(static_cast<std::uint32_t>(gfn2::Gfn2EigensolverLaunchStatus::kCublasError) == 3);
static_assert(static_cast<std::uint32_t>(gfn2::Gfn2EigensolverLaunchStatus::kCusolverError) == 4);
static_assert(std::is_standard_layout_v<gfn2::Gfn2EigensolverLaunchResult>);
static_assert(
    std::is_same_v<decltype(gfn2::Gfn2EigensolverLaunchResult::cusolver_status), cusolverStatus_t>);
static_assert(sizeof(scf::CudaEigensolverFamily) == 4);
static_assert(static_cast<std::uint32_t>(scf::CudaEigensolverFamily::small_native) == 0);
static_assert(static_cast<std::uint32_t>(scf::CudaEigensolverFamily::jacobi_batched) == 1);
static_assert(static_cast<std::uint32_t>(scf::CudaEigensolverFamily::xsyev_batched) == 2);
static_assert(static_cast<std::uint32_t>(scf::CudaEigensolverFamily::graph_native) == 3);
static_assert(static_cast<std::uint32_t>(scf::CudaEigensolverFamily::xsyevd) == 4);

void check_gfn2_status(const gfn2::Gfn2EigensolverLaunchResult& returned, cusolverStatus_t status) {
  assert(returned.success() == (status == CUSOLVER_STATUS_SUCCESS));
  assert(returned.status == (status == CUSOLVER_STATUS_SUCCESS
                                 ? gfn2::Gfn2EigensolverLaunchStatus::kSuccess
                                 : gfn2::Gfn2EigensolverLaunchStatus::kCusolverError));
  assert(returned.cusolver_status == status);
  assert(returned.cuda_status == cudaSuccess && returned.cublas_status == CUBLAS_STATUS_SUCCESS);
}

void test_gfn2_launch(shared::SymmetricEigenFamily family, shared::Eigenvectors vectors) {
  assert(family != shared::SymmetricEigenFamily::xsyevd);
  Fixture fixture;
  fixture.problem.vectors = vectors;
  gfn2::Gfn2EigensolverOptions options;
  const bool jacobi = family == shared::SymmetricEigenFamily::jacobi_batched;
  options.strategy = jacobi ? gfn2::Gfn2EigensolverStrategy::kBatchedJacobi
                            : gfn2::Gfn2EigensolverStrategy::kBatchedDivideAndConquer;
  options.jacobi = static_cast<syevjInfo_t>(fixture.resources.jacobi);
  gfn2::Gfn2EigensolverBucket bucket{5, 3};
  gfn2::Gfn2EigensolverDeviceWorkspace workspace;
  workspace.solver_device_workspace = fixture.resources.device_workspace;
  workspace.solver_device_workspace_bytes = fixture.resources.device_workspace_bytes;
  workspace.solver_host_workspace = fixture.resources.host_workspace;
  workspace.solver_host_workspace_bytes = fixture.resources.host_workspace_bytes;
  const auto run = [&] {
    return gfn2::symmetric_eigensolve(
        static_cast<cusolverDnHandle_t>(fixture.resources.solver),
        static_cast<cusolverDnParams_t>(fixture.resources.parameters),
        vectors == shared::Eigenvectors::values_only ? CUSOLVER_EIG_MODE_NOVECTOR
                                                     : CUSOLVER_EIG_MODE_VECTOR,
        bucket, bucket, fixture.matrices.data(), fixture.values.data(), options, workspace,
        fixture.info.data(), nullptr);
  };
  for (auto status : {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_ALLOC_FAILED,
                      CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_EXECUTION_FAILED}) {
    fixture.reset(status);
    check_gfn2_status(run(), status);
    assert(calls.size() == 1);
    fixture.check(calls[0], family, false, 0,
                  static_cast<int>(fixture.resources.device_workspace_bytes / sizeof(double)));
  }
  if (jacobi) {
    // The consumer retains its pre-existing admission and byte->element rules.
    for (int invalid : {0, 1, 2}) {
      fixture.reset();
      options.jacobi = invalid == 0 ? nullptr : static_cast<syevjInfo_t>(fixture.resources.jacobi);
      bucket.orbital_count = invalid == 1 ? 33 : 5;
      workspace.solver_device_workspace_bytes =
          invalid == 2 ? (std::size_t(std::numeric_limits<int>::max()) + 1) * sizeof(double)
                       : fixture.resources.device_workspace_bytes;
      const auto returned = run();
      assert(returned.status == gfn2::Gfn2EigensolverLaunchStatus::kInvalidArgument);
      assert(returned.cuda_status == cudaErrorInvalidValue);
      assert(calls.empty());
    }
  }
  // Values-only still takes the generic provider even under tridiagonal policy.
  options.strategy = gfn2::Gfn2EigensolverStrategy::kTridiagonalBisection;
  bucket = {5, 3};
  fixture.reset();
  gfn2::tridiagonal_calls = 0;
  check_gfn2_status(run(), CUSOLVER_STATUS_SUCCESS);
  assert(gfn2::tridiagonal_calls == (vectors == shared::Eigenvectors::values_and_vectors ? 1 : 0));
  assert(calls.size() == (vectors == shared::Eigenvectors::values_only ? 1U : 0U));
}

void test_gfn2_query(shared::SymmetricEigenFamily family) {
  assert(family != shared::SymmetricEigenFamily::xsyevd);
  const bool jacobi = family == shared::SymmetricEigenFamily::jacobi_batched;
  Fixture fixture;
  gfn2::Gfn2EigensolverBucket bucket{5, 3};
  auto solver = static_cast<cusolverDnHandle_t>(fixture.resources.solver);
  auto parameters = static_cast<cusolverDnParams_t>(fixture.resources.parameters);
  auto jacobi_handle = static_cast<syevjInfo_t>(fixture.resources.jacobi);
  const auto run = [&](gfn2::Gfn2EigensolverWorkspaceRequirements& requirements) {
    return jacobi ? gfn2::query_gfn2_jacobi_bucket_workspace_cuda(
                        solver, jacobi_handle, bucket, fixture.matrices.data(),
                        fixture.values.data(), requirements)
                  : gfn2::query_gfn2_eigensolver_bucket_workspace_cuda(
                        solver, parameters, bucket, fixture.matrices.data(), fixture.values.data(),
                        requirements);
  };
  for (auto status :
       {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_ALLOC_FAILED, CUSOLVER_STATUS_EXECUTION_FAILED}) {
    for (std::size_t stop = 0; stop != 6; ++stop) {
      for (bool retained_larger : {false, true}) {
        fixture.reset(status, stop);
        // Different non-monotonic peaks by mode and capacity defeat last-query
        // or dimension-only aggregation and accidental workspace unit changes.
        query_device = {90, 533, 17, 901, 220, 8};
        query_host = {77, 8, 101, 81, 41, 7};
        query_elements = {9, 73, 11, 37, 29, 5};
        const std::size_t prior_device = retained_larger ? 1999 : 5;
        const std::size_t prior_host = retained_larger ? 1997 : 7;
        gfn2::Gfn2EigensolverWorkspaceRequirements requirements{prior_device, prior_host};
        check_gfn2_status(run(requirements), status);
        const auto count = status == CUSOLVER_STATUS_SUCCESS ? 6U : stop + 1;
        assert(calls.size() == count);
        for (std::size_t i = 0; i != count; ++i) {
          fixture.problem.batch = static_cast<int>(i % 3) + 1;
          fixture.problem.vectors =
              i < 3 ? shared::Eigenvectors::values_only : shared::Eigenvectors::values_and_vectors;
          fixture.check(calls[i], family, true);
        }
        assert(requirements.solver_device_workspace_bytes ==
               (status != CUSOLVER_STATUS_SUCCESS
                    ? prior_device
                    : std::max(prior_device, jacobi ? 73 * sizeof(double) : 901U)));
        assert(requirements.solver_host_workspace_bytes ==
               (status != CUSOLVER_STATUS_SUCCESS || jacobi
                    ? prior_host
                    : std::max(prior_host, std::size_t(101))));
      }
    }
  }
  if (jacobi) {
    fixture.reset();
    query_elements = {9, -1};
    gfn2::Gfn2EigensolverWorkspaceRequirements requirements{13, 17};
    assert(run(requirements).status == gfn2::Gfn2EigensolverLaunchStatus::kInvalidArgument);
    assert(calls.size() == 2 && requirements.solver_device_workspace_bytes == 13);
    fixture.reset();
    bucket.orbital_count = 33;
    assert(run(requirements).success() && calls.empty());
  } else {
    fixture.reset();
    bucket.solve_count = 2;
    gfn2::Gfn2EigensolverWorkspaceRequirements requirements;
    assert(gfn2::query_gfn2_spin_eigensolver_bucket_workspace_cuda(
               solver, parameters, bucket, fixture.matrices.data(), fixture.values.data(),
               requirements)
               .success());
    assert(calls.size() == 4);
    for (std::size_t i = 0; i != calls.size(); ++i)
      assert(calls[i].batch == std::int64_t(i % 2) + 1);
  }
}

void test_scf_launch(shared::SymmetricEigenFamily family) {
  Fixture fixture;
  const auto consumer_family = family == shared::SymmetricEigenFamily::jacobi_batched
                                   ? scf::CudaEigensolverFamily::jacobi_batched
                               : family == shared::SymmetricEigenFamily::xsyev_batched
                                   ? scf::CudaEigensolverFamily::xsyev_batched
                                   : scf::CudaEigensolverFamily::xsyevd;
  scf::EigensolverResources resources{nullptr,
                                      static_cast<cusolverDnHandle_t>(fixture.resources.solver),
                                      static_cast<cusolverDnParams_t>(fixture.resources.parameters),
                                      static_cast<syevjInfo_t>(fixture.resources.jacobi),
                                      fixture.resources.device_workspace,
                                      fixture.resources.device_workspace_bytes,
                                      fixture.resources.host_workspace,
                                      fixture.resources.host_workspace_bytes};
  const std::uint8_t active[3]{1, 0, 1};
  scf::EigensolverProfileLaunch profile{};
  const auto run = [&](scf::CudaEigensolverFamily selected, bool profiled) {
    return scf::launch_solver(resources, selected, 5, 3, fixture.matrices.data(), nullptr,
                              fixture.values.data(), Fixture::launch_elements, fixture.info.data(),
                              active, profiled ? &profile : nullptr);
  };
  for (bool profiled : {false, true}) {
    for (auto status : {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_ALLOC_FAILED,
                        CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_EXECUTION_FAILED}) {
      fixture.reset(status);
      stages.clear();
      const auto returned = run(consumer_family, profiled);
      assert(returned == (status == CUSOLVER_STATUS_SUCCESS ? GENERATIVEQC_STATUS_SUCCESS
                          : status == CUSOLVER_STATUS_ALLOC_FAILED
                              ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                              : GENERATIVEQC_STATUS_CUDA_ERROR));
      const auto count =
          family == shared::SymmetricEigenFamily::xsyevd && status == CUSOLVER_STATUS_SUCCESS ? 3U
                                                                                              : 1U;
      assert(calls.size() == count);
      for (std::size_t i = 0; i != count; ++i) fixture.check(calls[i], family, false, i);
      std::vector<std::pair<std::string, std::size_t>> expected;
      if (profiled) expected.emplace_back("begin", 0);
      expected.emplace_back("sanitize", 0);
      if (profiled) expected.emplace_back("start", 0);
      if (profiled && status == CUSOLVER_STATUS_SUCCESS) expected.emplace_back("finish", count);
      assert(stages == expected);
    }
    for (const auto selected :
         {scf::CudaEigensolverFamily::small_native, scf::CudaEigensolverFamily::graph_native}) {
      fixture.reset();
      stages.clear();
      assert(run(selected, profiled) == GENERATIVEQC_STATUS_SUCCESS && calls.empty());
      const auto kernel_index = profiled ? 2U : 0U;
      assert(stages[kernel_index].first ==
             (selected == scf::CudaEigensolverFamily::small_native ? "small" : "graph"));
      for (const auto& stage : stages) assert(stage.first != "sanitize");
    }
  }
  // A failed earlier mask/profile stage never reaches the provider; finish-stage
  // failure propagates after the provider. Test the real CUDA-status translator.
  for (std::size_t stage = 0; stage != 4; ++stage) {
    fixture.reset();
    stages.clear();
    kernel_fail_at = stage;
    kernel_error = stage == 1 ? cudaErrorMemoryAllocation : cudaErrorUnknown;
    assert(run(consumer_family, true) ==
           (stage == 1 ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR));
    assert(calls.empty() == (stage < 3));
    assert(stages.size() == stage + 1);
  }
  kernel_fail_at = std::numeric_limits<std::size_t>::max();
  kernel_error = cudaSuccess;
}
void test_df_launch(shared::SymmetricEigenFamily family) {
  assert(family != shared::SymmetricEigenFamily::xsyevd);
  Fixture fixture;
  CudaDensityFittingJkPlan plan;
  df::DeviceSolver solver;
  solver.handle = static_cast<cusolverDnHandle_t>(fixture.resources.solver);
  solver.jacobi = static_cast<syevjInfo_t>(fixture.resources.jacobi);
  solver.parameters = static_cast<cusolverDnParams_t>(fixture.resources.parameters);
  solver.workspace = fixture.device.data();
  solver.workspace_bytes = fixture.resources.device_workspace_bytes;
  // The real owner's destructor frees host storage; preserve that contract.
  solver.host_workspace = std::malloc(fixture.resources.host_workspace_bytes);
  assert(solver.host_workspace != nullptr);
  fixture.resources.host_workspace = solver.host_workspace;
  solver.host_workspace_bytes = fixture.resources.host_workspace_bytes;
  solver.lwork = Fixture::launch_elements;
  solver.xsyev = family == shared::SymmetricEigenFamily::xsyev_batched;
  for (auto status : {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_ALLOC_FAILED,
                      CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_EXECUTION_FAILED}) {
    fixture.reset(status);
    stages.clear();
    counters.clear();
    std::string detail;
    const auto returned =
        df::solve_device_batch(plan, solver, 5, 3, fixture.matrices.data(), fixture.values.data(),
                               fixture.info.data(), detail);
    assert(returned == (status == CUSOLVER_STATUS_SUCCESS        ? GENERATIVEQC_STATUS_SUCCESS
                        : status == CUSOLVER_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                                 : GENERATIVEQC_STATUS_CUDA_ERROR));
    assert(calls.size() == 1);
    fixture.check(calls[0], family, false);
    if (status == CUSOLVER_STATUS_SUCCESS)
      assert(detail.empty());
    else
      assert(detail ==
             "CUDA DF SCF eigensolve failed with cuSOLVER status " + std::to_string(status));
    assert((stages ==
            std::vector<std::pair<std::string, std::size_t>>{{"df-trace", 0}, {"df-provider", 0}}));
    assert(
        (counters == std::vector<std::pair<std::string, std::uint64_t>>{
                         {"eigensystems", 3},
                         {"retained_solver_device_workspace_bytes", solver.workspace_bytes},
                         {"retained_solver_host_workspace_bytes", solver.host_workspace_bytes}}));
    assert(provider_label == (solver.xsyev ? "cusolverDnXsyevBatched" : "cusolverDnDsyevjBatched"));
  }
}

void test_invalid_family() {
  Fixture fixture;
  fixture.reset();
  shared::SymmetricEigenWorkspace workspace{59, 61, 67};
  const auto family = static_cast<shared::SymmetricEigenFamily>(999);
  assert(shared::query_symmetric_eigen(fixture.resources, family, fixture.problem,
                                       fixture.matrices.data(), fixture.values.data(),
                                       workspace) == CUSOLVER_STATUS_INVALID_VALUE);
  assert(workspace.device_bytes == 59 && workspace.host_bytes == 61 &&
         workspace.jacobi_elements == 67);
  assert(shared::launch_symmetric_eigen(fixture.resources, family, fixture.problem,
                                        fixture.matrices.data(), fixture.values.data(),
                                        fixture.info.data(), 23) == CUSOLVER_STATUS_INVALID_VALUE);
  assert(calls.empty());
  assert((fixture.info == std::array<int, 3>{{-71, -72, -73}}));
}

void test_generic_dimension_width(shared::SymmetricEigenFamily family) {
  static_assert(std::is_same_v<decltype(shared::SymmetricEigenProblem::n), std::int64_t>);
  static_assert(std::is_same_v<decltype(shared::SymmetricEigenProblem::batch), std::int64_t>);
  Fixture fixture;
  fixture.reset();
  fixture.problem.n = std::int64_t(std::numeric_limits<int>::max()) + 17;
  fixture.problem.batch = std::int64_t(std::numeric_limits<int>::max()) + 19;
  shared::SymmetricEigenWorkspace workspace;
  assert(shared::query_symmetric_eigen(fixture.resources, family, fixture.problem,
                                       fixture.matrices.data(), fixture.values.data(),
                                       workspace) == 0);
  assert(calls.size() == 1);
  fixture.check(calls[0], family, true);
  // Caller admission normally bounds these shapes; this ABI-only test makes
  // accidental narrowing in the shared generic path visible without allocating.
  if (family == shared::SymmetricEigenFamily::xsyev_batched) {
    fixture.reset();
    assert(shared::launch_symmetric_eigen(fixture.resources, family, fixture.problem,
                                          fixture.matrices.data(), fixture.values.data(),
                                          fixture.info.data(), 0) == 0);
    assert(calls.size() == 1);
    fixture.check(calls[0], family, false);
  }
}

int main(int argc, char** argv) {
  assert(argc == 4);
  const std::string operation = argv[1];
  const auto family = static_cast<shared::SymmetricEigenFamily>(std::atoi(argv[2]));
  const auto vectors = std::atoi(argv[3]) ? shared::Eigenvectors::values_and_vectors
                                          : shared::Eigenvectors::values_only;
  if (operation == "private-abi")
    test_private_abi(family);
  else if (operation == "query")
    test_query(family, vectors);
  else if (operation == "launch")
    test_launch(family, vectors);
  else if (operation == "gfn2-query")
    test_gfn2_query(family);
  else if (operation == "gfn2-launch")
    test_gfn2_launch(family, vectors);
  else if (operation == "scf-launch")
    test_scf_launch(family);
  else if (operation == "df-launch")
    test_df_launch(family);
  else if (operation == "invalid-family")
    test_invalid_family();
  else if (operation == "dimension-width")
    test_generic_dimension_width(family);
  else
    return 2;
  std::cout << "PASS " << operation << '\n';
}
