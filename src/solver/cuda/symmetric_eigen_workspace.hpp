#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "solver/cuda/symmetric_eigen_provider.hpp"

namespace generativeqc::solver::cuda {

/** An ordered inclusive capacity range for one vector mode. Ranges are visited
 * in declaration order, including overlaps/duplicates; no monotonic workspace
 * assumption or query deduplication is valid for the provider. */
struct SymmetricEigenQueryRange {
  std::int64_t first_batch;
  std::int64_t last_batch;
  Eigenvectors vectors;
};

/** Immutable borrowed setup domain. The ranges need live only through prepare;
 * the service stores no query pointers, matrix pointers, callbacks or allocator.
 * Method adapters declare only the family and reachable shape/mode capacities. */
struct SymmetricEigenQueryDomain {
  const SymmetricEigenFamily family;
  const std::int64_t n;
  const SymmetricEigenQueryRange* const ranges;
  const std::size_t range_count;
};

enum class EigenWorkspaceError { none, provider_failure, invalid_query, invalid_size };
struct EigenWorkspaceResult {
  EigenWorkspaceError error{EigenWorkspaceError::none};
  std::uint32_t provider_status{};
  bool success() const noexcept { return error == EigenWorkspaceError::none; }
};

struct EigenWorkspaceLimits {
  std::size_t device_bytes{std::numeric_limits<std::size_t>::max()};
  std::size_t host_bytes{std::numeric_limits<std::size_t>::max()};
  bool require_device{};
};
enum class EigenWorkspaceAdmission { accepted, empty_device, exceeds_limit };

/** Numeric workspace evidence only, not opaque-provider allocation accounting.
 * Published only after every query and signed-element conversion succeeds.
 * No allocation, stream action or provider selection occurs here. */
class PreparedSymmetricEigenWorkspace {
 public:
  const SymmetricEigenWorkspace& required() const noexcept { return required_; }
  EigenWorkspaceAdmission admit(const EigenWorkspaceLimits& limits) const noexcept;

 private:
  SymmetricEigenWorkspace required_{};
  friend EigenWorkspaceResult prepare_symmetric_eigen_workspace(const SymmetricEigenResources&,
                                                                const SymmetricEigenQueryDomain&,
                                                                const double*, const double*,
                                                                PreparedSymmetricEigenWorkspace&,
                                                                SymmetricEigenWorkspace) noexcept;
};

/** The floor carries previously admitted components (e.g. another provider in
 * a shared arena). Maxima are componentwise, never a byte sum. On any failure,
 * output and floor remain unchanged, even if a provider wrote partial sizes. */
EigenWorkspaceResult prepare_symmetric_eigen_workspace(const SymmetricEigenResources& resources,
                                                       const SymmetricEigenQueryDomain& domain,
                                                       const double* matrix, const double* values,
                                                       PreparedSymmetricEigenWorkspace& output,
                                                       SymmetricEigenWorkspace floor = {}) noexcept;

/** Explicit element count is the queried Jacobi lwork. Device capacity is the
 * padded allocation extent used by GFN2; it is deliberately not interchangeable
 * with queried lwork. Generic providers always receive the original extents. */
enum class JacobiWorkspaceExtent { queried_elements, device_capacity };
struct SymmetricEigenWorkspaceBinding {
  SymmetricEigenResources resources{};
  int jacobi_elements{};
};

/** Validate borrowed pointers/extents and publish the exact submission binding
 * transactionally. No buffers are allocated or owned. The required envelope
 * may be empty when an enclosing method has already admitted a combined arena. */
bool bind_symmetric_eigen_workspace(const SymmetricEigenResources& storage,
                                    const SymmetricEigenWorkspace& required,
                                    SymmetricEigenFamily family,
                                    JacobiWorkspaceExtent jacobi_extent,
                                    SymmetricEigenWorkspaceBinding& output) noexcept;

}  // namespace generativeqc::solver::cuda
