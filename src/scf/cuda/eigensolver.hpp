#pragma once

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>
#include <vector>

#include "runtime/lowering_binding.hpp"
#include "scf/cuda/eigensolver_types.hpp"
#include "scf/cuda_batch.hpp"

namespace generativeqc::scf::cuda_execution {

/** Borrowed eigensolver resources; allocation, lifetime and exact-stack qualification remain with
 * the prepared owner. */
struct EigensolverResources {
  cudaStream_t stream_{};
  cusolverDnHandle_t solver_{};
  cusolverDnParams_t solver_parameters_{};
  syevjInfo_t jacobi_{};
  void* solver_workspace_{};
  std::size_t solver_workspace_bytes_{};
  void* solver_host_workspace_{};
  std::size_t solver_host_workspace_bytes_{};
};

struct EigensolverProfileLaunch {
  std::int32_t physical_batch_size{};
  const std::uint8_t* physical_active{};
  bool cublas_transformed_inactive{};
  std::uint32_t capacity{};
  std::uint32_t* count{};
  DeviceInactiveEigensolverProfileEntry* entries{};
};

/** Execute the resolved native/library family on the owning stream. Masks and diagnostics preserve
 * inactive terminal states. */
generativeqc_status launch_solver(const EigensolverResources& resources,
                                  CudaEigensolverFamily family, int nbf, int batch_size,
                                  double* matrices, double* eigenvector_workspace,
                                  double* eigenvalues, int lwork, int* info,
                                  const std::uint8_t* active,
                                  const EigensolverProfileLaunch* profile = nullptr);

/** Whether this family requires provider input sanitization and cuSOLVER workspace. */
bool provider_eigensolver(CudaEigensolverFamily family);

/** Immutable preparation evidence. Identities describe the AOT template;
 * dimension/device/toolkit and queried resources complete this owner's runtime
 * binding. There is no cross-owner executable cache. Residuals are qualified by
 * independent tests, not measured on every launch; info remains caller-visible.
 */
struct OrdinaryEigensolverDiagnostic {
  runtime::NativeLoweringRequest request;
  std::array<runtime::NativeLoweringCandidate, 2> candidates;
  std::array<std::string_view, 2> rejections;
  std::size_t selected{};
  bool retained_incumbent{};
  int dimension{}, device{}, compute_major{}, compute_minor{}, runtime_version{}, driver_version{};
  std::uint64_t prepare_ns{};
  // Numeric resources only. cuSOLVER's opaque allocations are not bounded by
  // Xsyevd_bufferSize and must never be reported as zero total provider storage.
  bool opaque_provider_bytes_known{};
};

/** Prepared ordinary-stream eigensolver with explicit numeric workspace.
 * Borrows its owner's stream and matrix/eigenvalue buffers. Small matrices
 * retain the capture-safe native path; larger matrices reuse the existing
 * Xsyevd dispatch. Graph capture is rejected explicitly for provider-backed
 * solves instead of silently substituting an unbounded maximum-pivot solve.
 * Construction queries and charges workspace once; repeated solves and
 * serialized spin states allocate no numeric buffers.
 */
class OrdinaryStreamEigensolver {
 public:
  OrdinaryStreamEigensolver(cudaStream_t stream, int n, const double* matrix,
                            const double* eigenvalues);
  ~OrdinaryStreamEigensolver();
  OrdinaryStreamEigensolver(const OrdinaryStreamEigensolver&) = delete;
  OrdinaryStreamEigensolver& operator=(const OrdinaryStreamEigensolver&) = delete;
  /** Borrow contiguous disjoint matrix/scratch (batch*n*n), eigenvalue
   * (batch*n), info and mask buffers until the owning stream completes. Scratch
   * is caller-owned and already charged in the KS arena, not provider workspace.
   */
  generativeqc_status launch(int batch, double* matrices, double* native_workspace,
                             double* eigenvalues, int* info, const std::uint8_t* active) const;
  std::size_t device_bytes() const noexcept { return resources_.solver_workspace_bytes_; }
  std::size_t host_bytes() const noexcept { return host_workspace_.capacity(); }
  std::size_t metadata_bytes() const noexcept;
  const OrdinaryEigensolverDiagnostic& diagnostic() const noexcept { return diagnostic_; }

 private:
  void cleanup() noexcept;
  int n_{}, device_{};
  CudaEigensolverFamily family_{};
  EigensolverResources resources_{};
  std::vector<unsigned char> host_workspace_;
  std::array<char, 48> provider_version_{};
  OrdinaryEigensolverDiagnostic diagnostic_{};
};

}  // namespace generativeqc::scf::cuda_execution
