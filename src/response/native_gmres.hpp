#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "response/linear_problem.hpp"
#include "runtime/tracked_allocator.hpp"

namespace generativeqc::response {

enum class GmresStatus {
  initial_residual,
  converged,
  workspace_limit,
  max_iterations,
  breakdown,
  stagnation,
  nonfinite_input,
  nonfinite_operator,
  nonfinite_preconditioner,
};

/** Stable stopping reasons for diagnostics; convergence remains a true-residual gate. */
inline const char* gmres_status_name(GmresStatus status) noexcept {
  switch (status) {
    case GmresStatus::initial_residual:
      return "initial_residual";
    case GmresStatus::converged:
      return "converged";
    case GmresStatus::workspace_limit:
      return "workspace_limit";
    case GmresStatus::max_iterations:
      return "max_iterations";
    case GmresStatus::breakdown:
      return "breakdown";
    case GmresStatus::stagnation:
      return "stagnation";
    case GmresStatus::nonfinite_input:
      return "nonfinite_input";
    case GmresStatus::nonfinite_operator:
      return "nonfinite_operator";
    case GmresStatus::nonfinite_preconditioner:
      return "nonfinite_preconditioner";
  }
  return "unknown";
}

struct GmresOptions {
  double relative_tolerance{1e-10};
  double absolute_tolerance{};
  std::size_t restart{30};
  std::size_t max_iterations{200};
  std::size_t max_workspace_bytes{64ULL << 20};
  unsigned reorthogonalize{2};
  double breakdown_tolerance{1e-14};
  // Maximum iteration interval between full candidate-residual actions.
  // Predicted convergence, restart, breakdown and exhaustion also force a
  // fresh action. Only that true unpreconditioned residual can accept a solve.
  std::size_t true_residual_every{1};
  std::size_t stagnation_window{25};
  double stagnation_tolerance{1e-14};
  /** Optional owned diagnostics; workspace reserves each requested payload.
   * Disabled for existing native consumers unless explicitly requested.
   */
  bool collect_history{};
  bool collect_basis{};
  bool collect_timing{};
};

struct GmresPlan {
  std::size_t dimension{};
  std::size_t restart{};
  std::size_t workspace_bytes{};
  GmresOptions options;
};

/** Perturbation-specific borrowed inputs. Preconditioners act independently;
 * their callable/setup storage and workspace remain caller-owned.
 */
struct ResponseSolveRequest {
  std::span<const double> rhs;
  std::span<const double> initial_guess{};
  std::span<const double> diagonal_preconditioner{};
  LinearOperator right_preconditioner{};
};

struct GmresResult {
  /** Empty on workspace refusal; no dimension-sized output is allocated. */
  runtime::TrackedVector<double> solution;
  GmresStatus status{GmresStatus::nonfinite_input};
  double residual_norm{};
  double relative_residual{};
  std::size_t iterations{};
  std::size_t restarts{};
  std::size_t operator_actions{};
  std::size_t preconditioner_actions{};
  std::size_t workspace_bytes{};
  /** Actual high-water mark of GMRES-owned array payload, including its result.
   * Excludes input spans, operator callback allocations and allocator overhead;
   * this is not a complete MP2 endpoint measurement. Zero on admission refusal.
   */
  std::size_t measured_workspace_peak_bytes{};
  std::size_t workspace_allocation_count{};
  /** True-residual checkpoints, and column-major Arnoldi vectors associated
   * with the published best iterate. Empty when collection is disabled.
   */
  runtime::TrackedVector<double> history;
  runtime::TrackedVector<double> basis;
  std::size_t basis_columns{};
  double orthogonalization_seconds{};
  double operator_seconds{};

  [[nodiscard]] bool converged() const noexcept {
    return status == GmresStatus::initial_residual || status == GmresStatus::converged;
  }
};

double stable_norm(std::span<const double> values);
GmresPlan prepare_gmres(std::size_t dimension, const GmresOptions& options);
/** Solve with either a diagonal or a caller-owned right preconditioner.
 * The callback overwrites its output; nonfinite output is refused before the
 * physical operator runs. Callback exceptions propagate. Its storage/setup
 * belong to the caller's complete budget, not this controller's workspace.
 */
GmresResult solve_gmres(const GmresPlan& plan, const LinearOperator& apply,
                        std::span<const double> rhs, std::span<const double> initial_guess = {},
                        std::span<const double> diagonal_preconditioner = {},
                        const LinearOperator& right_preconditioner = {});

/** Conservative simultaneous numeric payload, including all retained results.
 * One to four independent states use exactly the single-RHS numerical owner.
 * No threads, coupled Krylov equations or residual-replay shortcuts are used.
 * Scalar/controller/allocator metadata and physical-action storage are excluded,
 * as for prepare_gmres; enclosing endpoint owners must admit them separately.
 */
std::size_t gmres_batch_workspace_bytes(const GmresPlan& plan, std::size_t request_count,
                                        std::size_t concurrent_requests);
/** Empty on aggregate/per-solve workspace refusal, before numeric allocation or
 * callbacks. Callback exceptions unwind every live state; no partial batch is
 * returned. Per-RHS operator timing includes wall time shared with other RHSs.
 */
std::vector<GmresResult> solve_gmres_batch(const GmresPlan& plan, const BatchLinearOperator& apply,
                                           std::span<const ResponseSolveRequest> requests,
                                           std::size_t concurrent_requests,
                                           std::size_t max_numeric_bytes);

}  // namespace generativeqc::response
