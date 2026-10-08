// Host ABI/call trace, not a numerical or real-device qualification.
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "generativeqc/generativeqc.h"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda/launch_geometry.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "solver/cuda/symmetric_eigen_handles.hpp"
#include "solver/cuda/symmetric_eigen_provider.hpp"
#include "solver/cuda/symmetric_eigen_workspace.hpp"

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
// Optional full-host orchestration trace; ordinary provider tests leave it null.
cusolverStatus_t (*provider_result_hook)() = nullptr;
cusolverStatus_t (*provider_stream_hook)(cusolverDnHandle_t, cudaStream_t) = nullptr;
constexpr int queried_elements = 37;
constexpr std::size_t queried_device_bytes = 513, queried_host_bytes = 79;

cusolverStatus_t result() {
  if (provider_result_hook != nullptr) return provider_result_hook();
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
shared::SymmetricEigenResources handle_tokens;
cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
  *handle = static_cast<cusolverDnHandle_t>(handle_tokens.solver);
  return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t solver, cudaStream_t stream) {
  if (provider_stream_hook != nullptr) return provider_stream_hook(solver, stream);
  return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* parameters) {
  *parameters = static_cast<cusolverDnParams_t>(handle_tokens.parameters);
  return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* jacobi) {
  *jacobi = static_cast<syevjInfo_t>(handle_tokens.jacobi);
  return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t, double) {
  return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t, int) { return CUSOLVER_STATUS_SUCCESS; }
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t, int) { return CUSOLVER_STATUS_SUCCESS; }
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
void number(const char* key, std::uint64_t value) { counters.emplace_back(key, value); }
}  // namespace df_progress
namespace host_trace {
struct Region {
  Region(const char*, std::size_t) { stages.emplace_back("df-provider", calls.size()); }
};
}  // namespace host_trace
}  // namespace runtime
struct CudaDensityFittingJkPlan {
  cudaStream_t stream{};
  cublasHandle_t blas{};
  std::size_t naux{7};
  void* integral_source{};
  bool streamed{};
  int device_id{};
  std::size_t nbf{5};
  void* ordinary_eigensystem{};
  shared::PreparedSymmetricEigenHandles eigen_handles;
};
using generativeqc::scf::df_scf_workspace_allowance;
std::vector<std::pair<std::size_t, std::size_t>> allocations;
std::vector<std::unique_ptr<unsigned char[]>> device_storage;
std::size_t allocation_fail_at = std::numeric_limits<std::size_t>::max();
generativeqc_status allocate_device(void** output, std::size_t bytes, const char*, std::string&) {
  allocations.emplace_back(bytes, calls.size());
  if (allocations.size() - 1 == allocation_fail_at) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  device_storage.emplace_back(std::make_unique<unsigned char[]>(bytes));
  *output = device_storage.back().get();
  return GENERATIVEQC_STATUS_SUCCESS;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
namespace runtime {
cudaError_t resource_cuda_malloc_async(void** output, std::size_t bytes, cudaStream_t) {
  std::string detail;
  return allocate_device(output, bytes, "", detail) == GENERATIVEQC_STATUS_SUCCESS
             ? cudaSuccess
             : cudaErrorMemoryAllocation;
}
}  // namespace runtime
struct RhfResources {
  shared::PreparedSymmetricEigenHandles eigen_handles_;
  cudaStream_t stream_{};
  void* blas_{};
  void* solver_workspace_{};
  std::size_t solver_workspace_bytes_{};
  void* solver_host_workspace_{};
  std::size_t solver_host_workspace_bytes_{};
  std::size_t reference_peak_bytes_{};
  ~RhfResources() { std::free(solver_host_workspace_); }
};
struct RhfPlan {
  int lwork{};
  bool retry_without_cublas{};
};
struct RhfOptions {
  bool export_physical_reference{};
  std::size_t reference_memory_budget_bytes{};
};
void fill_global_failure(generativeqc_status& output, generativeqc_status status) {
  output = status;
}
namespace posthf {
std::size_t checked_add(std::size_t a, std::size_t b) { return a + b; }
}  // namespace posthf
namespace reference_detail {
std::size_t check_capacity(std::size_t a, std::size_t b, std::size_t) { return a + b; }
}  // namespace reference_detail
cublasStatus_t cublasSetWorkspace(void*, void*, std::size_t bytes) {
  counters.emplace_back("blas-workspace", bytes);
  return CUBLAS_STATUS_SUCCESS;
}
generativeqc_status blas_status(cublasStatus_t) { return GENERATIVEQC_STATUS_CUDA_ERROR; }
struct OrdinaryResources {
  std::size_t solver_workspace_bytes_{}, solver_host_workspace_bytes_{};
};
struct OrdinaryDiagnostic {
  struct Candidate {
    std::size_t workspace_bytes{},
        host_bytes{generativeqc::scf::kOrdinaryEigensolverBindingHostBytes};
  };
  std::array<Candidate, 2> candidates;
};
struct MetricSetup {
  double* metrics{};
  double* eigenvalues{};
  int* solver_info{};
  void* solver_workspace{};
  std::vector<unsigned char> solver_host_workspace;
};
generativeqc_status fail_plan(CudaDensityFittingJkPlan*, generativeqc_status status) {
  return status;
}
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
  handle_tokens = fixture.resources;
  assert(solver.handles.create() == 0);
  assert(solver.handles.create_parameters() == 0);
  assert(solver.handles.configure_jacobi(1.0e-13, 100, 1) == 0);
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

void initialize_handles(shared::PreparedSymmetricEigenHandles& handles, Fixture& fixture) {
  handle_tokens = fixture.resources;
  assert(handles.create() == 0);
  assert(handles.create_parameters() == 0);
  assert(handles.configure_jacobi(1.e-13, 100, 1) == 0);
}

void reset_allocations() {
  allocations.clear();
  device_storage.clear();
  counters.clear();
  allocation_fail_at = std::numeric_limits<std::size_t>::max();
}

static_assert(sizeof(shared::PreparedSymmetricEigenWorkspace) ==
              sizeof(shared::SymmetricEigenWorkspace));
static_assert(std::is_trivially_copyable_v<shared::PreparedSymmetricEigenWorkspace>);

void test_workspace_envelope() {
  Fixture fixture;
  const shared::SymmetricEigenQueryRange ranges[]{{1, 3, shared::Eigenvectors::values_only},
                                                  {1, 3, shared::Eigenvectors::values_and_vectors}};
  for (auto family : {shared::SymmetricEigenFamily::jacobi_batched,
                      shared::SymmetricEigenFamily::xsyev_batched}) {
    const shared::SymmetricEigenQueryDomain domain{family, 5, ranges, 2};
    for (std::size_t index = 0; index != 6; ++index) {
      fixture.reset();
      shared::PreparedSymmetricEigenWorkspace prepared;
      assert(shared::prepare_symmetric_eigen_workspace(
                 fixture.resources, domain, fixture.matrices.data(), fixture.values.data(),
                 prepared, {1999, 1997, 101})
                 .success());
      const auto before = prepared.required();
      fixture.reset(CUSOLVER_STATUS_ALLOC_FAILED, index);
      // The fake provider writes its outputs even when it fails.
      const auto failed = shared::prepare_symmetric_eigen_workspace(
          fixture.resources, domain, fixture.matrices.data(), fixture.values.data(), prepared);
      assert(failed.error == shared::EigenWorkspaceError::provider_failure);
      assert(failed.provider_status == CUSOLVER_STATUS_ALLOC_FAILED);
      assert(calls.size() == index + 1);
      assert(prepared.required().device_bytes == before.device_bytes);
      assert(prepared.required().host_bytes == before.host_bytes);
      assert(prepared.required().jacobi_elements == before.jacobi_elements);
      if (family == shared::SymmetricEigenFamily::jacobi_batched) {
        fixture.reset();
        query_elements.assign(6, 7);
        query_elements[index] = -1;
        const auto negative = shared::prepare_symmetric_eigen_workspace(
            fixture.resources, domain, fixture.matrices.data(), fixture.values.data(), prepared);
        assert(negative.error == shared::EigenWorkspaceError::invalid_size);
        assert(calls.size() == index + 1 &&
               prepared.required().device_bytes == before.device_bytes);
      }
    }
  }
  const shared::SymmetricEigenQueryRange singleton{1, 1, shared::Eigenvectors::values_and_vectors};
  const auto maximum = std::numeric_limits<std::size_t>::max();
  shared::PreparedSymmetricEigenWorkspace prepared;
  fixture.reset();
  query_device = {maximum};
  query_host = {maximum - 1};
  assert(shared::prepare_symmetric_eigen_workspace(
             fixture.resources, {shared::SymmetricEigenFamily::xsyevd, 5, &singleton, 1},
             fixture.matrices.data(), fixture.values.data(), prepared)
             .success());
  assert(prepared.admit({maximum, maximum - 1, true}) == shared::EigenWorkspaceAdmission::accepted);
  assert(prepared.admit({maximum - 1, maximum}) == shared::EigenWorkspaceAdmission::exceeds_limit);
  assert(prepared.admit({maximum, maximum - 2}) == shared::EigenWorkspaceAdmission::exceeds_limit);
  // A singleton at INT64_MAX terminates without a signed increment overflow.
  const shared::SymmetricEigenQueryRange terminal{std::numeric_limits<std::int64_t>::max(),
                                                  std::numeric_limits<std::int64_t>::max(),
                                                  shared::Eigenvectors::values_only};
  fixture.reset();
  assert(shared::prepare_symmetric_eigen_workspace(
             fixture.resources, {shared::SymmetricEigenFamily::xsyev_batched, 5, &terminal, 1},
             fixture.matrices.data(), fixture.values.data(), prepared)
             .success());
  assert(calls.size() == 1 && calls[0].batch == terminal.first_batch);
  fixture.reset();
  assert(!shared::prepare_symmetric_eigen_workspace(
              fixture.resources, {shared::SymmetricEigenFamily::jacobi_batched, 5, &terminal, 1},
              fixture.matrices.data(), fixture.values.data(), prepared)
              .success());
  assert(calls.empty());
  for (auto invalid :
       {shared::SymmetricEigenQueryRange{0, 1, shared::Eigenvectors::values_only},
        shared::SymmetricEigenQueryRange{2, 1, shared::Eigenvectors::values_only},
        shared::SymmetricEigenQueryRange{1, 1, static_cast<shared::Eigenvectors>(-1)}}) {
    assert(!shared::prepare_symmetric_eigen_workspace(
                fixture.resources, {shared::SymmetricEigenFamily::xsyevd, 5, &invalid, 1},
                fixture.matrices.data(), fixture.values.data(), prepared)
                .success());
    assert(calls.empty());
  }
  for (auto family :
       {shared::SymmetricEigenFamily::jacobi_batched, shared::SymmetricEigenFamily::xsyevd}) {
    fixture.reset();
    query_elements = {0};
    query_device = {0};
    query_host = {0};
    assert(shared::prepare_symmetric_eigen_workspace(fixture.resources, {family, 5, &singleton, 1},
                                                     fixture.matrices.data(), fixture.values.data(),
                                                     prepared)
               .success());
    assert(prepared.admit({0, 0}) == shared::EigenWorkspaceAdmission::accepted);
    assert(prepared.admit({maximum, maximum, true}) ==
           shared::EigenWorkspaceAdmission::empty_device);
  }
  fixture.reset();
  query_elements = {std::numeric_limits<int>::max()};
  const auto largest_jacobi = shared::prepare_symmetric_eigen_workspace(
      fixture.resources, {shared::SymmetricEigenFamily::jacobi_batched, 5, &singleton, 1},
      fixture.matrices.data(), fixture.values.data(), prepared);
  if (maximum / sizeof(double) >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    assert(largest_jacobi.success());
    assert(prepared.required().device_bytes ==
           std::size_t(std::numeric_limits<int>::max()) * sizeof(double));
  } else {
    assert(largest_jacobi.error == shared::EigenWorkspaceError::invalid_size);
  }
}

void test_workspace_binding() {
  Fixture fixture;
  for (auto family :
       {shared::SymmetricEigenFamily::jacobi_batched, shared::SymmetricEigenFamily::xsyev_batched,
        shared::SymmetricEigenFamily::xsyevd}) {
    for (auto extent : {shared::JacobiWorkspaceExtent::queried_elements,
                        shared::JacobiWorkspaceExtent::device_capacity}) {
      shared::SymmetricEigenWorkspaceBinding binding;
      const shared::SymmetricEigenWorkspace required{513, 79, 37};
      assert(shared::bind_symmetric_eigen_workspace(fixture.resources, required, family, extent,
                                                    binding));
      assert(binding.resources.solver == fixture.resources.solver);
      assert(binding.resources.parameters == fixture.resources.parameters);
      assert(binding.resources.jacobi == fixture.resources.jacobi);
      assert(binding.resources.device_workspace == fixture.device.data());
      assert(binding.resources.device_workspace_bytes == 1025);
      assert(binding.resources.host_workspace == fixture.host.data());
      assert(binding.resources.host_workspace_bytes == 131);
      assert(binding.jacobi_elements == (family != shared::SymmetricEigenFamily::jacobi_batched ? 0
                                         : extent == shared::JacobiWorkspaceExtent::queried_elements
                                             ? 37
                                             : 128));
      for (int bad = 0; bad != 4; ++bad) {
        auto storage = fixture.resources;
        if (bad == 0) storage.device_workspace_bytes = 512;
        if (bad == 1) storage.host_workspace_bytes = 78;
        if (bad == 2) storage.device_workspace = nullptr;
        if (bad == 3) storage.host_workspace = nullptr;
        assert(!shared::bind_symmetric_eigen_workspace(storage, required, family, extent, binding));
        assert(binding.resources.device_workspace == fixture.device.data());
        assert(binding.resources.device_workspace_bytes == 1025);
      }
    }
  }
  shared::SymmetricEigenWorkspaceBinding binding;
  auto storage = fixture.resources;
  storage.device_workspace_bytes =
      (std::size_t(std::numeric_limits<int>::max()) + 1) * sizeof(double);
  assert(!shared::bind_symmetric_eigen_workspace(
      storage, {}, shared::SymmetricEigenFamily::jacobi_batched,
      shared::JacobiWorkspaceExtent::device_capacity, binding));
  for (int elements : {-1, 129}) {
    assert(!shared::bind_symmetric_eigen_workspace(
        fixture.resources, {0, 0, elements}, shared::SymmetricEigenFamily::jacobi_batched,
        shared::JacobiWorkspaceExtent::queried_elements, binding));
  }
  storage = {};
  storage.jacobi = fixture.resources.jacobi;
  assert(shared::bind_symmetric_eigen_workspace(
      storage, {}, shared::SymmetricEigenFamily::jacobi_batched,
      shared::JacobiWorkspaceExtent::queried_elements, binding));
  assert(binding.jacobi_elements == 0 && binding.resources.device_workspace == nullptr);
  assert(calls.empty());
}

void test_rhf_setup() {
  Fixture fixture;
  for (int kind = 0; kind != 3; ++kind) {
    for (auto spin_batch : {3U, 6U}) {
      const auto count = kind == 1 ? 2U : 1U;
      for (auto status : {CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_ALLOC_FAILED,
                          CUSOLVER_STATUS_EXECUTION_FAILED}) {
        for (std::size_t failure_index = 0; failure_index < count; ++failure_index) {
          fixture.reset(status, failure_index);
          reset_allocations();
          query_device = {901, 533};
          query_host = {79, 131};
          query_elements = {37};
          RhfResources resources;
          RhfPlan plan;
          initialize_handles(resources.eigen_handles_, fixture);
          const auto returned =
              prepare_rhf(kind == 0, kind == 2, 3, spin_batch, fixture.matrices.data(),
                          fixture.values.data(), resources, plan);
          assert(returned == (status == 0 ? GENERATIVEQC_STATUS_SUCCESS
                              : status == CUSOLVER_STATUS_ALLOC_FAILED
                                  ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                  : GENERATIVEQC_STATUS_CUDA_ERROR));
          assert(calls.size() == (status == 0 ? count : failure_index + 1));
          assert(calls[0].batch == (kind == 0 ? spin_batch : kind == 2 ? 1 : 3));
          if (calls.size() == 2) assert(calls[1].batch == spin_batch);
          for (const auto& call : calls) assert(call.vectors == CUSOLVER_EIG_MODE_VECTOR);
          if (status != 0) {
            assert(allocations.empty());
            assert(resources.solver_workspace_bytes_ == 0 &&
                   resources.solver_host_workspace_bytes_ == 0);
          } else {
            const std::size_t bytes = kind == 0 ? 37 * sizeof(double) : 901;
            assert(resources.solver_workspace_bytes_ == bytes);
            assert(resources.solver_host_workspace_bytes_ == (kind == 0   ? 0
                                                              : kind == 1 ? 131
                                                                          : 79));
            assert(plan.lwork == (kind == 0 ? 37 : 0));
            assert(allocations.size() == 1 && allocations[0].first == bytes &&
                   allocations[0].second == count);
            assert(counters.back() ==
                   std::make_pair(std::string("blas-workspace"), std::uint64_t(bytes)));
          }
        }
      }
    }
    for (int empty : {0, -1}) {
      fixture.reset();
      reset_allocations();
      query_device = {0, 0};
      query_host = {0, 0};
      query_elements = {empty};
      RhfResources resources;
      RhfPlan plan;
      initialize_handles(resources.eigen_handles_, fixture);
      assert(prepare_rhf(kind == 0, kind == 2, 3, 3, fixture.matrices.data(), fixture.values.data(),
                         resources, plan) == GENERATIVEQC_STATUS_CUDA_ERROR);
      assert(allocations.empty());
    }
  }
}

void test_df_setup() {
  Fixture fixture;
  for (auto n : {5U, 64U}) {
    for (int outcome = 0; outcome != 6; ++outcome) {
      fixture.reset(outcome == 1 ? CUSOLVER_STATUS_ALLOC_FAILED : CUSOLVER_STATUS_SUCCESS);
      reset_allocations();
      handle_tokens = fixture.resources;
      const auto allowance = df_scf_workspace_allowance(n, 3);
      query_device = {outcome == 2 ? 0U : outcome == 3 ? allowance + 1 : 513U};
      query_host = {outcome == 4 ? allowance + 1 : 79U};
      query_elements = {outcome == 2   ? 0
                        : outcome == 3 ? static_cast<int>(allowance / sizeof(double) + 1)
                        : outcome == 5 ? -1
                                       : 37};
      CudaDensityFittingJkPlan plan;
      df::DeviceSolver solver;
      std::string detail;
      const auto returned = df::setup_device_solver(plan, n, 3, fixture.matrices.data(),
                                                    fixture.values.data(), solver, detail);
      const bool bad = outcome == 1 || outcome == 2 || outcome == 3 || (outcome == 5 && n == 5);
      assert(calls.size() == 1 && calls[0].batch == 3 && calls[0].n == n);
      if (bad) {
        assert(returned == ((outcome == 1 || outcome == 3) ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                           : GENERATIVEQC_STATUS_CUDA_ERROR));
        assert(allocations.empty());
      } else {
        assert(returned == GENERATIVEQC_STATUS_SUCCESS);
        assert(allocations.size() == 1 && allocations[0].second == 1);
        assert(solver.workspace_bytes == (n == 5 ? 37 * sizeof(double) : 513));
        assert(solver.host_workspace_bytes == (n == 5 ? 0 : query_host[0]));
        assert(solver.lwork == (n == 5 ? 37 : 0));
      }
    }
  }
}

void test_ordinary_setup() {
  Fixture fixture;
  for (int outcome = 0; outcome != 5; ++outcome) {
    fixture.reset(outcome == 1 ? CUSOLVER_STATUS_ALLOC_FAILED : CUSOLVER_STATUS_SUCCESS);
    query_device = {outcome == 2 ? 0U : outcome == 3 ? 901U : 513U};
    query_host = {outcome == 2 ? 0U : outcome == 4 ? 901U : 79U};
    shared::PreparedSymmetricEigenHandles handles;
    initialize_handles(handles, fixture);
    OrdinaryResources resources;
    OrdinaryDiagnostic diagnostic;
    bool rejected = false;
    try {
      prepare_ordinary(5, fixture.matrices.data(), fixture.values.data(), handles, resources,
                       diagnostic, 900);
    } catch (const std::bad_alloc&) {
      assert(outcome == 3 || outcome == 4);
      rejected = true;
    } catch (generativeqc_status status) {
      assert(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY && outcome == 1);
      rejected = true;
    }
    assert(rejected == (outcome == 1 || outcome == 3 || outcome == 4));
    assert(calls.size() == 1 && calls[0].batch == 1 &&
           calls[0].vectors == CUSOLVER_EIG_MODE_VECTOR);
    if (!rejected) {
      assert(resources.solver_workspace_bytes_ == query_device[0]);
      assert(resources.solver_host_workspace_bytes_ == query_host[0]);
      assert(diagnostic.candidates[1].workspace_bytes == query_device[0]);
      assert(diagnostic.candidates[1].host_bytes ==
             generativeqc::scf::kOrdinaryEigensolverBindingHostBytes + query_host[0]);
    }
  }
}

void test_df_ao_setup() {
  Fixture fixture;
  for (int outcome = 0; outcome != 5; ++outcome) {
    for (std::size_t failed_allocation = 0; failed_allocation != 8; ++failed_allocation) {
      fixture.reset(outcome == 1 ? CUSOLVER_STATUS_ALLOC_FAILED : CUSOLVER_STATUS_SUCCESS);
      reset_allocations();
      allocation_fail_at = failed_allocation;
      const auto allowance = generativeqc::scf::df_eigen_workspace_allowance(5);
      query_device = {outcome == 2 ? 0U : outcome == 3 ? allowance + 1 : 513U};
      query_host = {outcome == 2 ? 0U : outcome == 4 ? allowance + 1 : 79U};
      CudaDensityFittingJkPlan plan;
      initialize_handles(plan.eigen_handles, fixture);
      dfao::OrdinaryEigensystem* state{};
      std::string detail;
      const auto returned = dfao::prepare(plan, state, detail);
      const std::size_t count = outcome == 2 ? 6 : 7;
      if (failed_allocation < count || outcome == 1 || outcome >= 3) {
        assert(returned == GENERATIVEQC_STATUS_OUT_OF_MEMORY);
        assert(state == nullptr && plan.ordinary_eigensystem == nullptr);
      } else {
        assert(returned == GENERATIVEQC_STATUS_SUCCESS);
        assert(state && state == plan.ordinary_eigensystem);
        assert(state->device_bytes ==
               3 * 25 * sizeof(double) + 5 * sizeof(double) + sizeof(int) + 1 + query_device[0]);
        assert(state->host_workspace_bytes == query_host[0]);
        const auto query_count = calls.size(), allocation_count = allocations.size();
        assert(dfao::prepare(plan, state, detail) == GENERATIVEQC_STATUS_SUCCESS);
        assert(calls.size() == query_count && allocations.size() == allocation_count);
        delete state;
      }
      if (failed_allocation < 6)
        assert(calls.empty());
      else {
        assert(calls.size() == 1 && calls[0].batch == 1 && calls[0].n == 5);
        if (outcome == 1 || outcome >= 3) assert(allocations.size() == 6);
      }
    }
  }
}

void test_metric_setup() {
  Fixture fixture;
  for (int outcome = 0; outcome != 5; ++outcome) {
    for (std::size_t fail_call = 0; fail_call != 4; ++fail_call) {
      fixture.reset(outcome == 1 ? CUSOLVER_STATUS_ALLOC_FAILED : CUSOLVER_STATUS_SUCCESS,
                    fail_call);
      reset_allocations();
      const auto allowance = generativeqc::scf::df_eigen_workspace_allowance(5);
      query_device = {outcome == 2 ? 0U : outcome == 3 ? allowance + 1 : 513U};
      query_host = {outcome == 2 ? 0U : outcome == 4 ? allowance + 1 : 79U};
      CudaDensityFittingJkPlan plan;
      initialize_handles(plan.eigen_handles, fixture);
      MetricSetup setup{
          fixture.matrices.data(), fixture.values.data(), fixture.info.data(), nullptr, {}};
      std::string detail;
      const auto status = prepare_metric(&plan, setup, 5, 3, detail);
      const bool failed = outcome == 1 || outcome == 3;
      assert(status == (failed ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_SUCCESS));
      assert(calls[0].operation == Operation::query_serial && calls[0].batch == 1);
      if (outcome == 3 || (outcome == 1 && fail_call == 0)) {
        assert(calls.size() == 1 && allocations.empty());
      } else {
        assert(calls.size() == (outcome == 1 ? fail_call + 1 : 4));
        if (outcome == 2)
          assert(allocations.empty());
        else
          assert(allocations.size() == 1 && allocations[0].second == 1);
        for (std::size_t index = 1; index < calls.size(); ++index) {
          const auto& call = calls[index];
          assert(call.operation == Operation::launch_serial && call.batch == 1);
          assert(call.matrix == fixture.matrices.data() + (index - 1) * 25);
          assert(call.values == fixture.values.data() + (index - 1) * 5);
          assert(call.info == fixture.info.data() + index - 1);
          assert(call.device == setup.solver_workspace && call.device_bytes == query_device[0]);
          assert(
              call.host ==
              (setup.solver_host_workspace.empty() ? nullptr : setup.solver_host_workspace.data()));
          assert(call.host_bytes == query_host[0]);
        }
      }
    }
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
  else if (operation == "metric-setup")
    test_metric_setup();
  else if (operation == "envelope")
    test_workspace_envelope();
  else if (operation == "binding")
    test_workspace_binding();
  else if (operation == "rhf-setup")
    test_rhf_setup();
  else if (operation == "df-setup")
    test_df_setup();
  else if (operation == "ordinary-setup")
    test_ordinary_setup();
  else if (operation == "df-ao-setup")
    test_df_ao_setup();
  else
    return 2;
  std::cout << "PASS " << operation << '\n';
  return 0;
}
