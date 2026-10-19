#pragma once

#include <cstddef>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

namespace generativeqc::scf::projection {

/** Projection proposes an initial density; it never authenticates convergence
 * or changes the target Hamiltonian. Cutoffs match the public metric policy. */
struct Policy {
  double relative_threshold{1e-10};
  double validation_tolerance{1e-7};
  double maximum_residual{0.5};
  std::size_t maximum_ao{4096};
  std::size_t maximum_workspace_bytes{};
};

struct Rejected : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

struct BudgetExceeded : std::bad_alloc {
  const char* what() const noexcept override {
    return "occupied projection exceeds maximum_workspace_bytes";
  }
};

/** All inputs are borrowed row-major FP64 arrays, with target-by-source cross
 * overlaps. A zero occupied count is a valid empty UHF spin channel. */
struct Input {
  std::size_t source_ao{}, target_ao{}, occupied{};
  unsigned occupation{2};
  std::span<const double> source_overlap, target_overlap, cross_overlap, source_state;
};

struct Diagnostics {
  std::size_t source_metric_rank{}, target_metric_rank{}, occupied_rank{};
  double minimum_projected_norm_squared{}, projection_residual{};
  double normal_equation_residual{}, metric_orthogonality_error{};
  double electron_trace{}, metric_idempotency_error{};
};

/** Detached target-by-occupied coefficients and target-by-target density.
 * Coefficient gauges may differ across providers; density and metric invariants
 * are the physical publication. Refusal publishes no partially formed seed. */
struct Result {
  std::vector<double> coefficients, density;
  Diagnostics diagnostics;
};

/** Conservative owner-visible numeric capacity, including detached native
 * results and ordinary eigen scratch. Caller arrays, metadata, allocator and
 * provider-internal retained overhead are separate; this is not an RSS bound. */
struct Capacity {
  std::size_t coefficient_count{}, density_count{}, numeric_workspace_bytes{};
};

Capacity capacity(const Input& input, const Policy& policy = {});

Result project_occupied(const Input& input, const Policy& policy = {});
Result project_density(const Input& input, const Policy& policy = {});

}  // namespace generativeqc::scf::projection
