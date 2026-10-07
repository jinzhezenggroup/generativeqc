#include "scf/cuda/df_scf_library.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "runtime/host_component_trace.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "solver/cuda/symmetric_eigen_provider.hpp"
#include "solver/cuda/symmetric_eigen_workspace.hpp"

namespace generativeqc::scf::cuda_df {
namespace eigen_provider = ::generativeqc::solver::cuda;

generativeqc_status recover_scf_capture(cudaStream_t stream, cudaError_t capture_error,
                                        generativeqc_status iteration_status,
                                        bool& capture_rejected, std::string& detail) {
  const auto expected = [](cudaError_t error) {
    // CUDA assigns stable ABI values 900/901 to unsupported/invalidated stream
    // capture. Compare the values so compatible runtimes need not spell the
    // optional enum names in their public headers.
    const int value = static_cast<int>(error);
    return error == cudaSuccess || value == 900 || value == 901 || error == cudaErrorNotSupported;
  };
  // Assigning a local cudaSuccess does not clear the runtime's last-error
  // slot. Otherwise the next successful generated tile launch can still fail
  // its cudaPeekAtLastError check with the abandoned capture's error 901.
  const auto pending = cudaGetLastError();
  if (!expected(pending)) return cuda_failure(pending, "CUDA DF capture pending error", detail);
  if (!expected(capture_error))
    return cuda_failure(capture_error, "CUDA DF capture failure", detail);
  // A library can report failure without setting CUDA's last-error slot.
  // Preserve that failure unless an explicit capture/mode error explains it;
  // in particular a solver allocation failure must never become a retry.
  if (iteration_status != GENERATIVEQC_STATUS_SUCCESS &&
      (iteration_status != GENERATIVEQC_STATUS_CUDA_ERROR ||
       (capture_error == cudaSuccess && pending == cudaSuccess)))
    return iteration_status;
  cudaStreamCaptureStatus capture{};
  const auto status = cudaStreamIsCapturing(stream, &capture);
  if (status != cudaSuccess) return cuda_failure(status, "query recovered DF stream", detail);
  if (capture != cudaStreamCaptureStatusNone) {
    detail = "CUDA DF capture must be ended before ordinary execution";
    return GENERATIVEQC_STATUS_CUDA_ERROR;
  }
  capture_rejected = true;
  detail.clear();
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status scf_gemm_strided(CudaDensityFittingJkPlan& plan, bool transpose_left,
                                     std::size_t batch_size, std::size_t nbf, const double* left,
                                     std::size_t left_stride, const double* right,
                                     std::size_t right_stride, double* output,
                                     std::size_t output_stride, double alpha, double beta,
                                     const char* failure_context, std::string& detail) {
  const cublasStatus_t status = cublasDgemmStridedBatched(
      plan.blas, transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(nbf),
      static_cast<int>(nbf), static_cast<int>(nbf), &alpha, left, static_cast<int>(nbf),
      static_cast<long long>(left_stride), right, static_cast<int>(nbf),
      static_cast<long long>(right_stride), &beta, output, static_cast<int>(nbf),
      static_cast<long long>(output_stride), static_cast<int>(batch_size));
  return status == CUBLAS_STATUS_SUCCESS ? GENERATIVEQC_STATUS_SUCCESS
                                         : blas_failure(status, failure_context, detail);
}

generativeqc_status scf_gemm(CudaDensityFittingJkPlan& plan, bool transpose_left,
                             std::size_t batch_size, std::size_t nbf, const double* left,
                             const double* right, double* output, std::string& detail) {
  const std::size_t matrix_elements = nbf * nbf;
  return scf_gemm_strided(plan, transpose_left, batch_size, nbf, left, matrix_elements, right,
                          matrix_elements, output, matrix_elements, 1.0, 0.0,
                          "CUDA DF device matrix product", detail);
}

generativeqc_status setup_device_solver(CudaDensityFittingJkPlan& plan, std::size_t nbf,
                                        std::size_t batch_size, double* eigensystem,
                                        double* eigenvalues, DeviceSolver& solver,
                                        std::string& detail) {
  cusolverStatus_t status = static_cast<cusolverStatus_t>(solver.handles.create());
  if (status == CUSOLVER_STATUS_SUCCESS) {
    status = static_cast<cusolverStatus_t>(solver.handles.bind_stream(plan.stream));
  }
  solver.xsyev = nbf > 32;
  if (status == CUSOLVER_STATUS_SUCCESS && !solver.xsyev) {
    status = static_cast<cusolverStatus_t>(solver.handles.configure_jacobi(1.0e-13, 100, 1));
  } else if (status == CUSOLVER_STATUS_SUCCESS) {
    status = static_cast<cusolverStatus_t>(solver.handles.create_parameters());
  }
  if (status != CUSOLVER_STATUS_SUCCESS) {
    return solver_failure(status, "initialize CUDA DF SCF eigensolver", detail);
  }
  const auto family = solver.xsyev ? eigen_provider::SymmetricEigenFamily::xsyev_batched
                                   : eigen_provider::SymmetricEigenFamily::jacobi_batched;
  const eigen_provider::SymmetricEigenQueryRange range{
      static_cast<std::int64_t>(batch_size), static_cast<std::int64_t>(batch_size),
      eigen_provider::Eigenvectors::values_and_vectors};
  eigen_provider::PreparedSymmetricEigenWorkspace prepared;
  const auto queried = eigen_provider::prepare_symmetric_eigen_workspace(
      solver.handles.view(), {family, static_cast<std::int64_t>(nbf), &range, 1}, eigensystem,
      eigenvalues, prepared);
  eigen_provider::EigenWorkspaceLimits limits;
  limits.require_device = true;
  if (!queried.success() ||
      prepared.admit(limits) != eigen_provider::EigenWorkspaceAdmission::accepted) {
    return solver_failure(
        queried.error == eigen_provider::EigenWorkspaceError::provider_failure
            ? static_cast<cusolverStatus_t>(queried.provider_status)
            : CUSOLVER_STATUS_INTERNAL_ERROR,
        solver.xsyev ? "size CUDA DF SCF generic eigensolver" : "size CUDA DF SCF eigensolver",
        detail);
  }
  const auto device_bytes = prepared.required().device_bytes;
  const auto host_bytes = prepared.required().host_bytes;
  solver.lwork = prepared.required().jacobi_elements;
  solver.workspace_bytes = device_bytes;
  solver.host_workspace_bytes = host_bytes;
  runtime::df_progress::number("compact_solver_workspace_bytes", device_bytes);
  runtime::df_progress::number("compact_solver_workspace_allowance",
                               df_scf_workspace_allowance(nbf, batch_size));
  // Compact DF historically bounds device workspace only; host allocation and
  // its diagnostics remain explicit rather than silently adding a new cap.
  limits.device_bytes = df_scf_workspace_allowance(nbf, batch_size);
  if (prepared.admit(limits) != eigen_provider::EigenWorkspaceAdmission::accepted) {
    detail = solver.xsyev ? "CUDA DF SCF generic eigensolver query exceeds its planned workspace"
                          : "CUDA DF SCF eigensolver query exceeds its planned workspace";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  if (!solver.xsyev)
    return allocate_device(reinterpret_cast<void**>(&solver.workspace), device_bytes,
                           "allocate CUDA DF SCF eigensolver workspace", detail);
  generativeqc_status allocation =
      allocate_device(reinterpret_cast<void**>(&solver.workspace), device_bytes,
                      "allocate CUDA DF SCF generic eigensolver workspace", detail);
  if (allocation != GENERATIVEQC_STATUS_SUCCESS) return allocation;
  if (host_bytes != 0) {
    solver.host_workspace = std::malloc(host_bytes);
    if (solver.host_workspace == nullptr) {
      detail = "host allocation for CUDA DF SCF generic eigensolver failed";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status solve_device_batch(CudaDensityFittingJkPlan& plan, DeviceSolver& solver,
                                       std::size_t nbf, std::size_t batch_size, double* eigensystem,
                                       double* eigenvalues, int* info, std::string& detail) {
  // Capture counts describe submitted graph nodes. Actual graph iterations
  // remain the device readback count; only ordinary calls receive event timing.
  runtime::cuda_trace::TraceOperation trace(
      "compact_eigensolve", plan.stream,
      {batch_size, nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  runtime::cuda_trace::trace_counter("eigensystems", batch_size);
  runtime::cuda_trace::trace_counter("retained_solver_device_workspace_bytes",
                                     solver.workspace_bytes);
  runtime::cuda_trace::trace_counter("retained_solver_host_workspace_bytes",
                                     solver.host_workspace_bytes);
  runtime::df_progress::label("eigen_provider",
                              solver.xsyev ? "cusolverDnXsyevBatched" : "cusolverDnDsyevjBatched");
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  // All descriptors, occupation buffers and workspaces were prepared once.
  // The provider call can itself block on queued GPU work or perform host
  // orchestration. Keep its host interval separate from the eigensolve's GPU
  // events: their difference is not an unmeasured CPU eigenframe validation.
  runtime::host_trace::Region provider("compact_eigensolve_provider", nbf);
  const auto family = solver.xsyev ? eigen_provider::SymmetricEigenFamily::xsyev_batched
                                   : eigen_provider::SymmetricEigenFamily::jacobi_batched;
  eigen_provider::SymmetricEigenWorkspaceBinding binding;
  if (!eigen_provider::bind_symmetric_eigen_workspace(
          {solver.handles.view().solver, solver.handles.view().parameters,
           solver.handles.view().jacobi, solver.workspace, solver.workspace_bytes,
           solver.host_workspace, solver.host_workspace_bytes},
          {0, 0, solver.lwork}, family, eigen_provider::JacobiWorkspaceExtent::queried_elements,
          binding)) {
    detail = "invalid CUDA DF SCF eigensolver workspace binding";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  status = static_cast<cusolverStatus_t>(eigen_provider::launch_symmetric_eigen(
      binding.resources, family,
      {static_cast<std::int64_t>(nbf), static_cast<std::int64_t>(batch_size),
       eigen_provider::Eigenvectors::values_and_vectors},
      eigensystem, eigenvalues, info, binding.jacobi_elements));
  return status == CUSOLVER_STATUS_SUCCESS
             ? GENERATIVEQC_STATUS_SUCCESS
             : solver_failure(status, "CUDA DF SCF eigensolve", detail);
}

}  // namespace generativeqc::scf::cuda_df
