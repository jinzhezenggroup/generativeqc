#include "scf/cuda/eigensolver.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "generated_solver_lowering.hpp"
#include "generativeqc/generativeqc.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/eigensolver_kernels.hpp"
#include "scf/cuda/launch_geometry.hpp"
#include "scf/eigensolver_workspace.hpp"
#include "solver/cuda/symmetric_eigen_provider.hpp"

namespace generativeqc::scf::cuda_execution {
namespace eigen_provider = ::generativeqc::solver::cuda;

/** Host-only provider dispatch and profiling order. Graph eligibility is resolved at setup before
 * this execution boundary. */
namespace {
generativeqc_status cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return GENERATIVEQC_STATUS_SUCCESS;
  return status == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                             : GENERATIVEQC_STATUS_CUDA_ERROR;
}

generativeqc_status solver_status(cusolverStatus_t status) {
  if (status == CUSOLVER_STATUS_SUCCESS) return GENERATIVEQC_STATUS_SUCCESS;
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                : GENERATIVEQC_STATUS_CUDA_ERROR;
}

}  // namespace

bool provider_eigensolver(CudaEigensolverFamily family) {
  return family == CudaEigensolverFamily::jacobi_batched ||
         family == CudaEigensolverFamily::xsyev_batched || family == CudaEigensolverFamily::xsyevd;
}

OrdinaryStreamEigensolver::OrdinaryStreamEigensolver(cudaStream_t stream, int n,
                                                     const double* matrix,
                                                     const double* eigenvalues)
    : n_(n) {
  const auto started = std::chrono::steady_clock::now();
  if (n <= 0 || !stream || !matrix || !eigenvalues)
    throw std::invalid_argument("invalid ordinary eigensolver owner");
  const auto checked = [](generativeqc_status status) {
    if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw generativeqc::Error(status, "ordinary CUDA eigensolver preparation failed");
  };
  checked(cuda_status(cudaGetDevice(&device_)));
  cudaStreamCaptureStatus capture{};
  checked(cuda_status(cudaStreamIsCapturing(stream, &capture)));
  if (capture != cudaStreamCaptureStatusNone)
    // Reject before taking stream cleanup responsibility: synchronizing a
    // borrowed capturing stream would invalidate its owner's graph.
    throw std::invalid_argument("ordinary eigensolver preparation cannot capture");
  resources_.stream_ = stream;
  try {
    diagnostic_.request = ordinary_eigh_request;
    diagnostic_.candidates = ordinary_eigh_candidates;
    diagnostic_.dimension = n;
    diagnostic_.device = device_;
    cudaDeviceProp properties{};
    checked(cuda_status(cudaGetDeviceProperties(&properties, device_)));
    diagnostic_.compute_major = properties.major;
    diagnostic_.compute_minor = properties.minor;
    checked(cuda_status(cudaRuntimeGetVersion(&diagnostic_.runtime_version)));
    checked(cuda_status(cudaDriverGetVersion(&diagnostic_.driver_version)));
    const auto allowance = ordinary_eigensolver_workspace_allowance(n);
    diagnostic_.request.constraints.workspace_bytes = allowance;
    diagnostic_.request.constraints.host_bytes = allowance + metadata_bytes();
    for (auto& offer : diagnostic_.candidates) offer.host_bytes = metadata_bytes();
    const bool small = n <= kSmallEigensolverLimit;
    diagnostic_.opaque_provider_bytes_known = small;
    diagnostic_.candidates[small ? 1 : 0].rejection =
        small ? "retained small-native size domain" : "outside qualified small-native size domain";
    if (!small) {
      int major{}, minor{}, patch{};
      checked(solver_status(cusolverGetProperty(MAJOR_VERSION, &major)));
      checked(solver_status(cusolverGetProperty(MINOR_VERSION, &minor)));
      checked(solver_status(cusolverGetProperty(PATCH_LEVEL, &patch)));
      const int written = std::snprintf(provider_version_.data(), provider_version_.size(),
                                        "%d.%d.%d", major, minor, patch);
      if (written <= 0 || std::size_t(written) >= provider_version_.size())
        throw std::runtime_error("cuSOLVER runtime version is unavailable");
      diagnostic_.candidates[1].provider_version = provider_version_.data();
      checked(solver_status(static_cast<cusolverStatus_t>(handles_.create())));
      checked(solver_status(static_cast<cusolverStatus_t>(handles_.bind_stream(stream))));
      checked(solver_status(static_cast<cusolverStatus_t>(handles_.create_parameters())));
      eigen_provider::SymmetricEigenWorkspace queried;
      checked(solver_status(static_cast<cusolverStatus_t>(eigen_provider::query_symmetric_eigen(
          handles_.view(), eigen_provider::SymmetricEigenFamily::xsyevd,
          {n, 1, eigen_provider::Eigenvectors::values_and_vectors}, matrix, eigenvalues,
          queried))));
      resources_.solver_workspace_bytes_ = queried.device_bytes;
      resources_.solver_host_workspace_bytes_ = queried.host_bytes;
      if (resources_.solver_workspace_bytes_ > allowance ||
          resources_.solver_host_workspace_bytes_ > allowance)
        throw std::bad_alloc();
      diagnostic_.candidates[1].workspace_bytes = resources_.solver_workspace_bytes_;
      diagnostic_.candidates[1].host_bytes += resources_.solver_host_workspace_bytes_;
    }
    // Both offers consume the same canonical request. Unknown endpoint costs
    // retain the previously qualified size-domain incumbent; no performance
    // promotion, precision relaxation or maximum-pivot fallback is permitted.
    const auto decision = runtime::select_native_lowering(
        diagnostic_.request, diagnostic_.candidates, ordinary_eigh_target,
        ordinary_eigh_compilation, 1, small ? 0 : 1);
    diagnostic_.selected = decision.selected;
    diagnostic_.retained_incumbent = decision.retained_incumbent;
    for (const auto& rejection : decision.rejections)
      diagnostic_.rejections[rejection.candidate] = rejection.reason;
    family_ = decision.selected == 0 ? CudaEigensolverFamily::small_native
                                     : CudaEigensolverFamily::xsyevd;
    if (resources_.solver_workspace_bytes_)
      checked(cuda_status(runtime::resource_cuda_malloc(&resources_.solver_workspace_,
                                                        resources_.solver_workspace_bytes_)));
    host_workspace_.resize(resources_.solver_host_workspace_bytes_);
    if (host_workspace_.capacity() > allowance) throw std::bad_alloc();
    diagnostic_.candidates[decision.selected].host_bytes = host_bytes() + metadata_bytes();
    resources_.solver_host_workspace_ = host_workspace_.data();
    diagnostic_.prepare_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
  } catch (...) {
    cleanup();
    throw;
  }
}

std::size_t OrdinaryStreamEigensolver::metadata_bytes() const noexcept {
  static_assert(sizeof(OrdinaryStreamEigensolver) <= kOrdinaryEigensolverBindingHostBytes);
  return kOrdinaryEigensolverBindingHostBytes;
}

void OrdinaryStreamEigensolver::cleanup() noexcept {
  int previous = device_;
  (void)cudaGetDevice(&previous);
  (void)cudaSetDevice(device_);
  if (resources_.stream_) (void)cudaStreamSynchronize(resources_.stream_);
  if (resources_.solver_workspace_) (void)runtime::resource_cuda_free(resources_.solver_workspace_);
  handles_.reset();
  resources_ = {};
  (void)cudaSetDevice(previous);
}

OrdinaryStreamEigensolver::~OrdinaryStreamEigensolver() { cleanup(); }

generativeqc_status OrdinaryStreamEigensolver::launch(int batch, double* matrices,
                                                      double* native_workspace, double* eigenvalues,
                                                      int* info, const std::uint8_t* active) const {
  if (batch <= 0 || !matrices || !native_workspace || !eigenvalues || !info || !active)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  int device = -1;
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) return cuda_status(error);
  if (device != device_) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  cudaStreamCaptureStatus capture{};
  error = cudaStreamIsCapturing(resources_.stream_, &capture);
  if (error != cudaSuccess) return cuda_status(error);
  if (capture != cudaStreamCaptureStatusNone &&
      !diagnostic_.candidates[diagnostic_.selected].capture_safe)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto borrowed = resources_;
  borrowed.solver_ = static_cast<cusolverDnHandle_t>(handles_.view().solver);
  borrowed.solver_parameters_ = static_cast<cusolverDnParams_t>(handles_.view().parameters);
  return launch_solver(borrowed, family_, n_, batch, matrices, native_workspace, eigenvalues, 0,
                       info, active);
}

generativeqc_status launch_solver(const EigensolverResources& resources,
                                  CudaEigensolverFamily family, int nbf, int batch_size,
                                  double* matrices, double* eigenvector_workspace,
                                  double* eigenvalues, int lwork, int* info,
                                  const std::uint8_t* active,
                                  const EigensolverProfileLaunch* profile) {
  const bool provider_invoked = provider_eigensolver(family);
  if (profile != nullptr) {
    launch_begin_inactive_eigensolver_profile_kernel(
        1, 1, 0, resources.stream_, profile->physical_batch_size, batch_size,
        static_cast<std::uint32_t>(family), provider_invoked, profile->cublas_transformed_inactive,
        profile->physical_active, active, profile->capacity, profile->count, profile->entries);
    const cudaError_t profile_error = cudaPeekAtLastError();
    if (profile_error != cudaSuccess) return cuda_status(profile_error);
  }
  if (provider_invoked) {
    // One block per solver state returns immediately for active matrices. The
    // homogeneous fast path therefore pays one tiny mask kernel while a
    // divergent provider batch receives finite identity placeholders.
    launch_sanitize_inactive_solver_input_kernel(
        static_cast<unsigned>(batch_size), kCaptureSafeKernelThreads, 0, resources.stream_,
        batch_size, nbf, active, matrices, info, profile == nullptr ? 0U : profile->capacity,
        profile == nullptr ? nullptr : profile->count,
        profile == nullptr ? nullptr : profile->entries);
    const cudaError_t sanitize_error = cudaPeekAtLastError();
    if (sanitize_error != cudaSuccess) return cuda_status(sanitize_error);
  }
  if (profile != nullptr) {
    launch_start_inactive_eigensolver_timer_kernel(1, 1, 0, resources.stream_, profile->capacity,
                                                   profile->count, profile->entries);
    const cudaError_t profile_error = cudaPeekAtLastError();
    if (profile_error != cudaSuccess) return cuda_status(profile_error);
  }
  generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS;
  if (family == CudaEigensolverFamily::small_native) {
    launch_symmetric_eigen_small_kernel(static_cast<unsigned>(batch_size), 1, 0, resources.stream_,
                                        batch_size, nbf, matrices, eigenvalues, info, active);
    status = cuda_status(cudaPeekAtLastError());
  } else if (provider_invoked) {
    const auto provider_family = family == CudaEigensolverFamily::jacobi_batched
                                     ? eigen_provider::SymmetricEigenFamily::jacobi_batched
                                 : family == CudaEigensolverFamily::xsyev_batched
                                     ? eigen_provider::SymmetricEigenFamily::xsyev_batched
                                     : eigen_provider::SymmetricEigenFamily::xsyevd;
    const auto provider_status =
        static_cast<cusolverStatus_t>(eigen_provider::launch_symmetric_eigen(
            {resources.solver_, resources.solver_parameters_, resources.jacobi_,
             resources.solver_workspace_, resources.solver_workspace_bytes_,
             resources.solver_host_workspace_, resources.solver_host_workspace_bytes_},
            provider_family, {nbf, batch_size, eigen_provider::Eigenvectors::values_and_vectors},
            matrices, eigenvalues, info, lwork));
    if (provider_status != CUSOLVER_STATUS_SUCCESS) return solver_status(provider_status);
  } else {
    // API-ineligible or Graph-rejected signatures retain the unbounded native
    // implementation without treating a provider limitation as a calculation
    // failure.
    launch_symmetric_eigen_graph_maximum_pivot_kernel(
        static_cast<unsigned>(batch_size), kGraphEigensolverThreads, 0, resources.stream_,
        batch_size, nbf, matrices, eigenvector_workspace, eigenvalues, info, active);
    status = cuda_status(cudaPeekAtLastError());
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  if (profile != nullptr) {
    launch_finish_inactive_eigensolver_profile_kernel(1, 1, 0, resources.stream_, batch_size,
                                                      active, info, profile->capacity,
                                                      profile->count, profile->entries);
    status = cuda_status(cudaPeekAtLastError());
  }
  return status;
}

}  // namespace generativeqc::scf::cuda_execution
