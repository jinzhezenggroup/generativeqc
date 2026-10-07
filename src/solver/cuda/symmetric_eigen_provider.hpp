#pragma once

#include <cstddef>
#include <cstdint>

namespace generativeqc::solver::cuda {

/** Backend-internal lowerings. Method policy and exact-stack qualification choose
 * the already-admitted family before submission; this service never switches it. */
enum class SymmetricEigenFamily { jacobi_batched, xsyev_batched, xsyevd };
enum class Eigenvectors { values_only, values_and_vectors };

/** Borrowed provider resources, with no NVIDIA/CuMetal ABI in the shared header.
 * The original owner retains handle, stream binding and buffer lifetimes. No
 * handle creation, allocation, synchronization or stream mutation occurs here. */
struct SymmetricEigenResources {
  void* solver{};
  void* parameters{};
  void* jacobi{};
  void* device_workspace{};
  std::size_t device_workspace_bytes{};
  void* host_workspace{};
  std::size_t host_workspace_bytes{};
};

/** Contiguous FP64, column-major, lower-triangle input with lda=n. Eigenvalues
 * are ascending; vector mode overwrites matrices with eigenvector columns.
 * The caller validates dimensions/storage and owns inactive-input treatment,
 * generalized transforms, publication, failure interpretation and capture gates. */
struct SymmetricEigenProblem {
  std::int64_t n{};
  std::int64_t batch{};
  Eigenvectors vectors{Eigenvectors::values_and_vectors};
};

struct SymmetricEigenWorkspace {
  std::size_t device_bytes{};
  std::size_t host_bytes{};
  int jacobi_elements{};
};

/** Query exactly one shape/mode/capacity. Jacobi reports elements, generic
 * providers report bytes; callers retain their existing conversion/validation
 * and capacity aggregation order. No new shape query is introduced. An unknown
 * internal family returns provider invalid-value status without submission. */
std::uint32_t query_symmetric_eigen(const SymmetricEigenResources& resources,
                                    SymmetricEigenFamily family,
                                    const SymmetricEigenProblem& problem, const double* matrix,
                                    const double* values,
                                    SymmetricEigenWorkspace& workspace) noexcept;

/** Submit the exact borrowed workspace and Jacobi lwork selected by the owner.
 * Xsyevd serializes batch matrices on the existing stream and stops at the first
 * provider failure. Batched families make one call. The return is the unmodified
 * 32-bit provider status, not a numerical info or a method status. */
std::uint32_t launch_symmetric_eigen(const SymmetricEigenResources& resources,
                                     SymmetricEigenFamily family,
                                     const SymmetricEigenProblem& problem, double* matrix,
                                     double* values, int* info, int jacobi_elements) noexcept;

}  // namespace generativeqc::solver::cuda
