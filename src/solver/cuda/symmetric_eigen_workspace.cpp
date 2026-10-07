// Workspace preparation consolidates GenerativeQC and adapted xTBloom sizing.
// xTBloom portions: GPL-3.0-or-later; original scoped CUDA/MKL additional
// permission: LICENSES/xtbloom-CUDA_MKL_LINKING_EXCEPTION.txt.
#include "solver/cuda/symmetric_eigen_workspace.hpp"

#include <algorithm>
#include <limits>

namespace generativeqc::solver::cuda {

EigenWorkspaceResult prepare_symmetric_eigen_workspace(const SymmetricEigenResources& resources,
                                                       const SymmetricEigenQueryDomain& domain,
                                                       const double* matrix, const double* values,
                                                       PreparedSymmetricEigenWorkspace& output,
                                                       SymmetricEigenWorkspace floor) noexcept {
  if (domain.n <= 0 || domain.ranges == nullptr || domain.range_count == 0 ||
      (domain.family != SymmetricEigenFamily::jacobi_batched &&
       domain.family != SymmetricEigenFamily::xsyev_batched &&
       domain.family != SymmetricEigenFamily::xsyevd))
    return {EigenWorkspaceError::invalid_query};
  // Validate the declaration before the first provider call. The terminal
  // capacity is compared before incrementing, including INT64_MAX domains.
  for (std::size_t index = 0; index < domain.range_count; ++index) {
    const auto& range = domain.ranges[index];
    if (range.first_batch <= 0 || range.last_batch < range.first_batch ||
        (range.vectors != Eigenvectors::values_only &&
         range.vectors != Eigenvectors::values_and_vectors) ||
        (domain.family == SymmetricEigenFamily::jacobi_batched &&
         (domain.n > std::numeric_limits<int>::max() ||
          range.last_batch > std::numeric_limits<int>::max())))
      return {EigenWorkspaceError::invalid_query};
  }
  auto candidate = floor;
  for (std::size_t index = 0; index < domain.range_count; ++index) {
    const auto& range = domain.ranges[index];
    for (auto batch = range.first_batch;; ++batch) {
      SymmetricEigenWorkspace queried;
      const auto status = query_symmetric_eigen(
          resources, domain.family, {domain.n, batch, range.vectors}, matrix, values, queried);
      if (status != 0) return {EigenWorkspaceError::provider_failure, status};
      if (domain.family == SymmetricEigenFamily::jacobi_batched) {
        if (queried.jacobi_elements < 0 ||
            static_cast<std::size_t>(queried.jacobi_elements) >
                std::numeric_limits<std::size_t>::max() / sizeof(double))
          return {EigenWorkspaceError::invalid_size};
        queried.device_bytes = static_cast<std::size_t>(queried.jacobi_elements) * sizeof(double);
      }
      candidate.device_bytes = std::max(candidate.device_bytes, queried.device_bytes);
      candidate.host_bytes = std::max(candidate.host_bytes, queried.host_bytes);
      candidate.jacobi_elements = std::max(candidate.jacobi_elements, queried.jacobi_elements);
      if (batch == range.last_batch) break;
    }
  }
  output.required_ = candidate;
  return {};
}

EigenWorkspaceAdmission PreparedSymmetricEigenWorkspace::admit(
    const EigenWorkspaceLimits& limits) const noexcept {
  if (limits.require_device && required_.device_bytes == 0)
    return EigenWorkspaceAdmission::empty_device;
  if (required_.device_bytes > limits.device_bytes || required_.host_bytes > limits.host_bytes)
    return EigenWorkspaceAdmission::exceeds_limit;
  return EigenWorkspaceAdmission::accepted;
}

bool bind_symmetric_eigen_workspace(const SymmetricEigenResources& storage,
                                    const SymmetricEigenWorkspace& required,
                                    SymmetricEigenFamily family,
                                    JacobiWorkspaceExtent jacobi_extent,
                                    SymmetricEigenWorkspaceBinding& output) noexcept {
  if (storage.device_workspace_bytes < required.device_bytes ||
      storage.host_workspace_bytes < required.host_bytes ||
      (storage.device_workspace_bytes != 0 && storage.device_workspace == nullptr) ||
      (storage.host_workspace_bytes != 0 && storage.host_workspace == nullptr))
    return false;
  int elements = 0;
  if (family == SymmetricEigenFamily::jacobi_batched) {
    if (storage.jacobi == nullptr) return false;
    if (jacobi_extent == JacobiWorkspaceExtent::device_capacity) {
      const auto capacity = storage.device_workspace_bytes / sizeof(double);
      if (capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
      elements = static_cast<int>(capacity);
    } else {
      if (required.jacobi_elements < 0 || static_cast<std::size_t>(required.jacobi_elements) >
                                              storage.device_workspace_bytes / sizeof(double))
        return false;
      elements = required.jacobi_elements;
    }
  } else if (family != SymmetricEigenFamily::xsyev_batched &&
             family != SymmetricEigenFamily::xsyevd) {
    return false;
  }
  output = {storage, elements};
  return true;
}

}  // namespace generativeqc::solver::cuda
